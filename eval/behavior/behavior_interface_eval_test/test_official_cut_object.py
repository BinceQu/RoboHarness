from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import behavior_interface_eval_test.tool.official_v2.tools as official_tools
from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.tool.official_v2.capabilities import (
    OfficialToolBoundaryError,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.dynamic_point_tracker import (
    CameraIntrinsics,
    TrackerFrame,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    eef_pose as local_eef_pose,
)
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_object_distance import (
    TrackObjectDistanceBindingError,
    TrackedObjectDistanceMemory,
)


class _LiveCutManager:
    def __init__(self, world, adapter, target_offset=(0.018, 0.0, 0.0)) -> None:
        self.world = world
        self.adapter = adapter
        self.session_id = ""
        self.image_id = ""
        self.target_offset = np.asarray(target_offset, dtype=np.float64)
        self.start_sequence = int(adapter.status()["sequence"])
        self.initial_source = self._eef_position("left")
        self.target = self.initial_source + self.target_offset
        self.target_moved = False
        self.move_target_after_sequence: int | None = None
        self.loss_after_sequence: int | None = None
        self.episode_failure_after_sequence: int | None = None
        self.binding_error: TrackObjectDistanceBindingError | None = None

    def _eef_position(self, side: str) -> np.ndarray:
        state = official_tools.local_robot_state(
            trunk_q=self.world.trunk_qpos(),
            arm_left_q=self.world.arm_qpos_list("left"),
            arm_right_q=self.world.arm_qpos_list("right"),
            gripper_left_q=self.world.gripper_qpos_list("left"),
            gripper_right_q=self.world.gripper_qpos_list("right"),
        )
        q = self.world.arm_qpos_list(side)
        position, _quaternion = local_eef_pose(state, side, q)
        return np.asarray(position, dtype=np.float64)

    def replace_points(self, points, **kwargs):
        if self.binding_error is not None:
            raise self.binding_error
        check_cancelled = kwargs.get("check_cancelled")
        if callable(check_cancelled):
            check_cancelled()
        self.session_id = str(kwargs["session_id"])
        self.image_id = str(kwargs["image_id"])
        sequence = int(self.adapter.status()["sequence"])
        return {
            "entries": {},
            "replaced_names": [],
            "session_id": self.session_id,
            "image_id": self.image_id,
            "capture_observation_sequence": int(
                kwargs["capture_observation_sequence"]
            ),
            "observation_sequence": sequence,
            "replayed_observation_count": 0,
            "episode_id": str(kwargs["capture_episode_id"]),
            "frame_binding": (
                "policy_owned_frozen_head_capture_then_ordered_live_replay"
            ),
            "frame_digest": str(kwargs["capture_frame_digest"]),
        }

    def observed_points_snapshot(
        self,
        names,
        *,
        session_id,
        image_id,
        episode_id,
    ):
        sequence = int(self.adapter.status()["sequence"])
        if self.loss_after_sequence is not None and sequence >= self.loss_after_sequence:
            return {
                "ok": False,
                "reason": "tracked_point_unavailable",
                "entries": {},
                "unavailable": {"cutting_tool_point": "point_unobserved"},
                "observation_sequence": sequence,
                "episode_id": episode_id,
            }
        if (
            self.episode_failure_after_sequence is not None
            and sequence >= self.episode_failure_after_sequence
        ):
            return {
                "ok": False,
                "reason": "episode_changed",
                "entries": {},
                "unavailable": {},
                "observation_sequence": sequence,
                "episode_id": "different-episode",
            }
        if (
            self.move_target_after_sequence is not None
            and sequence >= self.move_target_after_sequence
            and not self.target_moved
        ):
            self.target = self.target + np.array([0.004, 0.0, 0.0])
            self.target_moved = True
        source = self._eef_position("left")
        entries = {
            "cutting_tool_point": {
                "name": "cutting_tool_point",
                "status": "observed",
                "depth_m": 1.0,
                "xyz_in_robot_base_coord_m": source.tolist(),
                "observation_sequence": sequence,
            },
            "target_object_point": {
                "name": "target_object_point",
                "status": "observed",
                "depth_m": 1.0,
                "xyz_in_robot_base_coord_m": self.target.tolist(),
                "observation_sequence": sequence,
            },
        }
        return {
            "ok": True,
            "reason": None,
            "entries": {name: entries[name] for name in names},
            "unavailable": {},
            "observation_sequence": sequence,
            "episode_id": episode_id,
            "session_id": session_id,
            "image_id": image_id,
        }


class OfficialCutObjectTest(unittest.TestCase):
    @staticmethod
    def _ready_world():
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
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        arm_q = np.asarray(official_tools.GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES[f"gripper_{side}_qpos"]] = [0.02, 0.02]
            proprio[PROPRIO_SLICES[f"gripper_{side}_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.45, -0.4, 0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        return adapter, world

    @staticmethod
    def _ctx(world, raise_if_cancelled=lambda where="": None):
        result = {}
        return SimpleNamespace(
            world=world,
            task_name="cutting_test",
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=raise_if_cancelled,
        ), result

    @staticmethod
    def _binding(adapter, world):
        sequence = int(adapter.status()["sequence"])
        capture = SimpleNamespace(
            evaluator_sequence=sequence,
            depth=np.ones((12, 16), dtype=np.float32),
        )
        return {
            "capture": capture,
            "episode_id": world.episode_id(),
            "frame_digest": "a" * 64,
        }

    @staticmethod
    def _points():
        return [
            {"name": "cutting_tool_point", "u": 400, "v": 500},
            {"name": "target_object_point", "u": 600, "v": 500},
        ]

    @staticmethod
    def _drive(adapter, world, generator, *, follow_actions=True):
        actions = []
        for raw_action in generator:
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            if follow_actions:
                for side in ("left", "right"):
                    proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                        ACTION_SLICES[f"arm_{side}"]
                    ]
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    def test_contract_registry_and_fixed_roles(self) -> None:
        normalized = validate_submission(
            "cut_object",
            {
                "session_id": "cut-session",
                "image_id": "img_0001",
                "points": list(reversed(self._points())),
            },
        )
        self.assertEqual(
            [point["name"] for point in normalized["points"]],
            ["cutting_tool_point", "target_object_point"],
        )
        params = build_registry(None)["cut_object"].params
        self.assertEqual(
            [item["name"] for item in params],
            [
                "session_id",
                "image_id",
                "points",
                "pos_tol",
                "ori_tol_deg",
                "max_steps",
                "timeout_s",
            ],
        )
        with self.assertRaisesRegex(OfficialToolBoundaryError, "named exactly"):
            validate_submission(
                "cut_object",
                {
                    "session_id": "cut-session",
                    "image_id": "img_0001",
                    "points": [
                        {"name": "knife", "u": 1, "v": 2},
                        {"name": "food", "u": 3, "v": 4},
                    ],
                },
            )

    def test_atomic_live_snapshot_tracks_both_current_xyz(self) -> None:
        height, width = 144, 192
        rng = np.random.default_rng(71)
        tool_texture = rng.integers(25, 230, size=(43, 43), dtype=np.uint8)
        target_texture = rng.integers(25, 230, size=(43, 43), dtype=np.uint8)

        def frame(sequence, tool_center, target_center):
            gray = np.full((height, width), 8, dtype=np.uint8)
            depth = np.full((height, width), 4.0, dtype=np.float32)
            for center, texture, value in (
                (tool_center, tool_texture, 1.2),
                (target_center, target_texture, 1.5),
            ):
                x, y = center
                gray[y - 21:y + 22, x - 21:x + 22] = texture
                depth[y - 21:y + 22, x - 21:x + 22] = value
            return TrackerFrame(
                rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
                depth_linear=depth,
                intrinsics=CameraIntrinsics(
                    width=width,
                    height=height,
                    fx=81.6,
                    fy=81.6,
                    cx=width * 0.5,
                    cy=height * 0.5,
                ),
                camera_to_robot_base=np.eye(4, dtype=np.float64),
                camera_to_policy=np.eye(4, dtype=np.float64),
                sequence=sequence,
                timestamp_s=sequence / 30.0,
                episode_id="episode-a",
                camera_role="head",
                color_order="rgb",
            )

        manager = TrackedObjectDistanceMemory()
        manager.ingest(frame(1, (52, 60), (140, 82)))
        binding = manager.register_capture(
            session_id="cut-session",
            image_id="img_0001",
            observation_sequence=1,
            episode_id="episode-a",
            image_shape=(height, width),
        )
        manager.replace_points(
            [
                {
                    "name": "cutting_tool_point",
                    "u": 52 / (width - 1) * 1000,
                    "v": 60 / (height - 1) * 1000,
                },
                {
                    "name": "target_object_point",
                    "u": 140 / (width - 1) * 1000,
                    "v": 82 / (height - 1) * 1000,
                },
            ],
            session_id="cut-session",
            image_id="img_0001",
            capture_observation_sequence=1,
            capture_episode_id="episode-a",
            capture_image_shape=(height, width),
            capture_frame_digest=binding["frame_digest"],
        )
        manager.ingest(frame(2, (57, 62), (137, 80)))
        snapshot = manager.observed_points_snapshot(
            ["cutting_tool_point", "target_object_point"],
            session_id="cut-session",
            image_id="img_0001",
            episode_id="episode-a",
        )
        self.assertTrue(snapshot["ok"], snapshot)
        self.assertEqual(snapshot["observation_sequence"], 2)
        self.assertEqual(
            set(snapshot["entries"]),
            {"cutting_tool_point", "target_object_point"},
        )
        for entry in snapshot["entries"].values():
            self.assertEqual(entry["observation_sequence"], 2)
            self.assertTrue(
                np.isfinite(entry["xyz_in_robot_base_coord_m"]).all()
            )
        active = manager.observed_active_points_snapshot(
            ["cutting_tool_point", "target_object_point"],
            episode_id="episode-a",
        )
        self.assertTrue(active["ok"], active)
        self.assertEqual(active["session_id"], "cut-session")
        self.assertEqual(active["image_id"], "img_0001")
        self.assertEqual(active["observation_sequence"], 2)

    def test_live_servo_follows_moving_target_and_emits_legal_actions(self) -> None:
        adapter, world = self._ready_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        manager = _LiveCutManager(world, adapter)
        manager.move_target_after_sequence = int(adapter.status()["sequence"]) + 1
        world._official_tracked_object_distances = manager
        ctx, result = self._ctx(world)
        binding = self._binding(adapter, world)
        with mock.patch.object(
            official_tools,
            "_load_tracker_bound_capture",
            return_value=binding,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["cut_object"].fn(
                    ctx,
                    session_id="cut-session",
                    image_id="img_0001",
                    points=self._points(),
                    pos_tol=0.004,
                    max_steps=120,
                ),
            )
        self.assertTrue(result["ok"], result)
        self.assertTrue(manager.target_moved)
        self.assertEqual(result["selected_arm"], "left")
        self.assertEqual(
            result["gripper_evidence"]["left"]["state"],
            "fully_closed",
        )
        self.assertTrue(
            result["gripper_evidence"]["left"][
                "eligible_held_tool_evidence"
            ]
        )
        self.assertTrue(result["point_coincidence_verified"])
        self.assertLessEqual(
            result["visual_servo"]["final_point_distance_m"],
            0.004,
        )
        self.assertEqual(result["visual_servo"]["stable_steps"], 2)
        self.assertFalse(result["contact_truth_used"])
        self.assertFalse(result["bddl_truth_used"])
        self.assertFalse(result["collision_checked"])
        self.assertGreaterEqual(len(actions), 3)
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

    def test_tracking_loss_and_digest_mismatch_fail_closed(self) -> None:
        for mode in ("tracking_loss", "episode_change", "digest_mismatch"):
            with self.subTest(mode=mode):
                adapter, world = self._ready_world()
                manager = _LiveCutManager(world, adapter)
                if mode == "tracking_loss":
                    manager.loss_after_sequence = int(adapter.status()["sequence"]) + 1
                elif mode == "episode_change":
                    manager.episode_failure_after_sequence = (
                        int(adapter.status()["sequence"]) + 1
                    )
                else:
                    manager.binding_error = TrackObjectDistanceBindingError(
                        "digest_mismatch",
                        "frozen frame digest mismatch",
                    )
                world._official_tracked_object_distances = manager
                ctx, result = self._ctx(world)
                with mock.patch.object(
                    official_tools,
                    "_load_tracker_bound_capture",
                    return_value=self._binding(adapter, world),
                ):
                    actions = self._drive(
                        adapter,
                        world,
                        build_registry(adapter)["cut_object"].fn(
                            ctx,
                            session_id="cut-session",
                            image_id="img_0001",
                            points=self._points(),
                            pos_tol=0.004,
                            max_steps=30,
                        ),
                    )
                self.assertFalse(result["ok"], result)
                self.assertFalse(result.get("point_coincidence_verified", False))
                if mode == "tracking_loss":
                    self.assertEqual(result["reason"], "tracked_point_unavailable")
                    self.assertEqual(result["failure_stage"], "tracking")
                elif mode == "episode_change":
                    self.assertEqual(result["reason"], "episode_changed")
                    self.assertEqual(result["failure_stage"], "tracking")
                else:
                    self.assertEqual(result["reason"], "digest_mismatch")
                    self.assertEqual(result["failure_stage"], "tracking")
                self.assertNotIn("depth_m", result)
                self.assertNotIn("xyz_in_robot_base_coord_m", result)
                for action in actions:
                    self.assertTrue(np.isfinite(action).all())

    def test_stall_timeout_and_cancellation_hold(self) -> None:
        adapter, world = self._ready_world()
        manager = _LiveCutManager(world, adapter)
        world._official_tracked_object_distances = manager
        ctx, stalled = self._ctx(world)
        with mock.patch.object(
            official_tools,
            "_load_tracker_bound_capture",
            return_value=self._binding(adapter, world),
        ):
            stall_actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["cut_object"].fn(
                    ctx,
                    session_id="cut-session",
                    image_id="img_0001",
                    points=self._points(),
                    pos_tol=0.004,
                    max_steps=30,
                ),
                follow_actions=False,
            )
        self.assertFalse(stalled["ok"], stalled)
        self.assertEqual(stalled["reason"], "proprio_stall")
        self.assertGreaterEqual(len(stall_actions), 4)

        adapter, world = self._ready_world()
        manager = _LiveCutManager(world, adapter)
        world._official_tracked_object_distances = manager
        ctx, timed_out = self._ctx(world)
        clock = iter((0.0, 1.0, 1.0, 1.0))
        with mock.patch.object(
            official_tools,
            "_load_tracker_bound_capture",
            return_value=self._binding(adapter, world),
        ), mock.patch.object(
            official_tools.time,
            "monotonic",
            side_effect=lambda: next(clock, 1.0),
        ):
            timeout_actions = list(
                build_registry(adapter)["cut_object"].fn(
                    ctx,
                    session_id="cut-session",
                    image_id="img_0001",
                    points=self._points(),
                    pos_tol=0.004,
                    timeout_s=0.1,
                )
            )
        self.assertFalse(timed_out["ok"], timed_out)
        self.assertEqual(timed_out["reason"], "timeout")
        self.assertEqual(timed_out["failure_stage"], "timeout")
        self.assertEqual(len(timeout_actions), 1)

        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._ready_world()
        manager = _LiveCutManager(world, adapter)
        world._official_tracked_object_distances = manager
        ctx, cancelled = self._ctx(
            world,
            raise_if_cancelled=lambda where="": (_ for _ in ()).throw(
                SkillCancelled(f"cancelled at {where}")
            ),
        )
        with mock.patch.object(
            official_tools,
            "_load_tracker_bound_capture",
            return_value=self._binding(adapter, world),
        ):
            cancel_actions = list(
                build_registry(adapter)["cut_object"].fn(
                    ctx,
                    session_id="cut-session",
                    image_id="img_0001",
                    points=self._points(),
                )
            )
        self.assertFalse(cancelled["ok"], cancelled)
        self.assertEqual(cancelled["failure_stage"], "cancellation")
        self.assertEqual(cancelled["reason"], "cancelled")
        self.assertEqual(len(cancel_actions), 1)


if __name__ == "__main__":
    unittest.main()
