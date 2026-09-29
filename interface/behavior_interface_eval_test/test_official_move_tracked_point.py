from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from flask import Flask, jsonify

import behavior_interface_eval_test.tool.official_v2.tools as official_tools
import behavior_interface_eval_test.tool.official_v2.rgbd_grasp_lite as rgbd_lite
import behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local as tracked_motion
from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
    install_official_move_tracked_point_routes,
    install_official_v2_failure_capture_contract,
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
from behavior_interface_eval_test.tool.official_v2.eef_adjustment_local import (
    local_robot_state,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    quat_to_mat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    eef_pose,
)
from behavior_interface_eval_test.tool.official_v2.human_track_object_distance_ui import (
    install_track_object_distance_human_ui,
)
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    evaluate_target_constraints,
    plan_endpoint,
)


class _RigidTrackedManager:
    def __init__(self, world, adapter, points_world: dict[str, np.ndarray]) -> None:
        self.world = world
        self.adapter = adapter
        self.session_id = "tracked-session"
        self.image_id = "img_tracked_0001"
        self.arm = "left"
        self.loss_after_sequence: int | None = None
        self.loss_after_arm_motion_rad: float | None = None
        self.loss_until_sequence: int | None = None
        self.temporary_loss_after_arm_motion_rad: float | None = None
        self.temporary_loss_snapshot_count = 0
        self._temporary_loss_remaining: int | None = None
        self._motion_retention_leases: dict[str, dict] = {}
        self._motion_retention_counter = 0
        self.initial_arm_q = np.asarray(
            self.world.arm_qpos_list(self.arm), dtype=np.float64
        )
        position, quaternion = self._eef_pose()
        rotation = quat_to_mat_xyzw(quaternion)
        self.anchors = {
            name: rotation.T @ (np.asarray(point) - position)
            for name, point in points_world.items()
        }

    def _eef_pose(self):
        state = local_robot_state(
            trunk_q=self.world.trunk_qpos(),
            arm_left_q=self.world.arm_qpos_list("left"),
            arm_right_q=self.world.arm_qpos_list("right"),
            gripper_left_q=self.world.gripper_qpos_list("left"),
            gripper_right_q=self.world.gripper_qpos_list("right"),
        )
        return eef_pose(state, self.arm, self.world.arm_qpos_list(self.arm))

    def activate_rigid_pair(self, names, *, episode_id):
        if len(names) != 2:
            raise ValueError("rigid pair requires two names")
        missing = [name for name in names if name not in self.anchors]
        if missing:
            raise ValueError(f"missing rigid pair names: {missing}")
        position, quaternion = self._eef_pose()
        rotation = quat_to_mat_xyzw(quaternion)
        points = np.asarray(
            [position + rotation @ self.anchors[name] for name in names]
        )
        return {
            "ok": True,
            "names": list(names),
            "reference_distance_m": float(
                np.linalg.norm(points[1] - points[0])
            ),
            "episode_id": episode_id,
            "build": "unit_test_rigid_pair_provider",
        }

    def begin_motion_retention(self, names, *, episode_id, timeout_s):
        requested = tuple(str(name) for name in names)
        missing = [name for name in requested if name not in self.anchors]
        if missing:
            raise ValueError(f"missing motion retention names: {missing}")
        self._motion_retention_counter += 1
        lease_id = f"unit-motion-retention-{self._motion_retention_counter}"
        self._motion_retention_leases[lease_id] = {
            "names": requested,
            "episode_id": str(episode_id),
            "timeout_s": float(timeout_s),
        }
        return {
            "ok": True,
            "lease_id": lease_id,
            "names": list(requested),
            "episode_id": str(episode_id),
            "duration_s": float(timeout_s),
            "coordinates_published_while_unobserved": False,
        }

    def end_motion_retention(self, lease_id):
        released = self._motion_retention_leases.pop(str(lease_id), None) is not None
        return {
            "ok": released,
            "released": released,
            "lease_id": str(lease_id),
        }

    def observed_active_points_snapshot(self, names, *, episode_id):
        sequence = int(self.adapter.status()["sequence"])
        arm_motion = float(
            np.linalg.norm(
                np.asarray(
                    self.world.arm_qpos_list(self.arm), dtype=np.float64
                )
                - self.initial_arm_q,
                ord=np.inf,
            )
        )
        lost_after_motion = bool(
            self.loss_after_arm_motion_rad is not None
            and arm_motion >= self.loss_after_arm_motion_rad
        )
        temporary_loss = False
        if (
            self.temporary_loss_after_arm_motion_rad is not None
            and arm_motion >= self.temporary_loss_after_arm_motion_rad
        ):
            if self._temporary_loss_remaining is None:
                self._temporary_loss_remaining = int(
                    self.temporary_loss_snapshot_count
                )
            if self._temporary_loss_remaining > 0:
                self._temporary_loss_remaining -= 1
                temporary_loss = True
        sequence_loss = bool(
            self.loss_after_sequence is not None
            and sequence >= self.loss_after_sequence
            and (
                self.loss_until_sequence is None
                or sequence <= self.loss_until_sequence
            )
        )
        if (
            sequence_loss
            or lost_after_motion
            or temporary_loss
        ):
            return {
                "ok": False,
                "reason": "tracked_point_unavailable",
                "entries": {},
                "unavailable": {str(names[0]): "point_unobserved"},
                "observation_sequence": sequence,
                "episode_id": episode_id,
                "session_id": self.session_id,
                "image_id": self.image_id,
            }
        missing = [name for name in names if name not in self.anchors]
        if missing:
            return {
                "ok": False,
                "reason": "tracked_point_unavailable",
                "entries": {},
                "unavailable": {name: "point_not_registered_or_retired" for name in missing},
                "observation_sequence": sequence,
                "episode_id": episode_id,
                "session_id": self.session_id,
                "image_id": self.image_id,
            }
        position, quaternion = self._eef_pose()
        rotation = quat_to_mat_xyzw(quaternion)
        entries = {
            name: {
                "name": name,
                "status": "observed",
                "depth_m": 1.0,
                "xyz_in_robot_base_coord_m": (
                    position + rotation @ self.anchors[name]
                ).astype(float).tolist(),
                "observation_sequence": sequence,
            }
            for name in names
        }
        return {
            "ok": True,
            "reason": None,
            "entries": entries,
            "unavailable": {},
            "observation_sequence": sequence,
            "episode_id": episode_id,
            "session_id": self.session_id,
            "image_id": self.image_id,
            "frame": "current_robot_base",
        }


class _ScriptedRigidPairActivationManager(_RigidTrackedManager):
    """Expose deterministic fresh-frame rigid-pair reports to the tool."""

    def __init__(
        self,
        world,
        adapter,
        points_world: dict[str, np.ndarray],
        *,
        initial_report: dict,
        fresh_reports: list[dict],
    ) -> None:
        super().__init__(world, adapter, points_world)
        self.initial_report = dict(initial_report)
        self.fresh_reports = [dict(report) for report in fresh_reports]
        self.activation_sequences: list[int] = []
        self.snapshot_sequences: list[int] = []
        self.active_pair: dict | None = None
        self.deactivation_reasons: list[str] = []

    @staticmethod
    def _at_sequence(report: dict, sequence: int) -> dict:
        return {
            **dict(report),
            "observation_sequence": int(sequence),
        }

    def activate_rigid_pair(self, names, *, episode_id):
        sequence = int(self.adapter.status()["sequence"])
        self.activation_sequences.append(sequence)
        self.active_pair = {"names": list(names)}
        return {
            **self._at_sequence(self.initial_report, sequence),
            "names": list(names),
            "episode_id": str(episode_id),
        }

    def status(self):
        return {"active_rigid_pair": self.active_pair}

    def deactivate_rigid_pair(self, *, reason="rigid_pair_request_scope_ended"):
        previous = self.active_pair
        self.active_pair = None
        self.deactivation_reasons.append(str(reason))
        return {
            "ok": True,
            "active": False,
            "reason": str(reason),
            "previous_pair": previous,
        }

    def observed_active_points_snapshot(self, names, *, episode_id):
        snapshot = super().observed_active_points_snapshot(
            names,
            episode_id=episode_id,
        )
        sequence = int(self.adapter.status()["sequence"])
        self.snapshot_sequences.append(sequence)
        report_index = min(
            len(self.snapshot_sequences) - 1,
            max(0, len(self.fresh_reports) - 1),
        )
        report = (
            {"ok": True, "reason": None, "fusion_applied": True}
            if not self.fresh_reports
            else self.fresh_reports[report_index]
        )
        report = self._at_sequence(report, sequence)
        snapshot["rigid_pair_fusion"] = report
        snapshot["ok"] = bool(report.get("ok"))
        snapshot["reason"] = (
            None
            if snapshot["ok"]
            else str(report.get("reason") or "rigid_pair_measurement_invalid")
        )
        return snapshot


class OfficialMoveTrackedPointTest(unittest.TestCase):
    @staticmethod
    def _safe_overlap_context() -> dict:
        return {
            "session": {
                "session_id": "tracked-session",
                "image_id": "img_tracked_0001",
                "rgb_path": "/policy-owned/frozen.png",
                "depth_path": "/policy-owned/frozen.depth.npy",
            },
            "camera_pos": [0.45, 0.0, 0.6],
            "camera_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "focal_length": 1.0,
            "horizontal_aperture": 1.0,
        }

    @staticmethod
    def _safe_overlap_evaluation(*, poses, **_kwargs) -> dict:
        reports = []
        for index, pose in enumerate(poses):
            reports.append(
                {
                    "pose_index": index,
                    "eef_position_robot_base_m": np.asarray(
                        pose["eef_pos"], dtype=np.float64
                    ).astype(float).tolist(),
                    "eef_quaternion_xyzw": np.asarray(
                        pose["quat"], dtype=np.float64
                    ).astype(float).tolist(),
                    "overlap_vox": 0,
                    "overlap_vol_cm3": 0.0,
                    "original_overlap_ok": True,
                    "inflated_overlap_vox": 0,
                    "inflated_overlap_vol_cm3": 0.0,
                    "inflated_overlap_ok": True,
                    "inflated_query_skipped": False,
                    "inflated_query_skipped_reason": None,
                    "ok": True,
                    "reason": None,
                }
            )
        return {
            "ok": True,
            "build": rgbd_lite.LITE_POSE_OVERLAP_FILTER_BUILD,
            "pose_count": len(reports),
            "passing_pose_count": len(reports),
            "rejected_pose_count": 0,
            "passing_pose_indices": list(range(len(reports))),
            "original_overlap_threshold_cm3": 0.15,
            "original_overlap_comparison": "<=",
            "inflated_overlap_threshold_cm3": 1.6,
            "inflated_overlap_comparison": "<",
            "grasp_opening_volume_queried": False,
            "poses": reports,
        }
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
            task_name="chopping_wood",
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=raise_if_cancelled,
        ), result

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

    @staticmethod
    def _terminal_occlusion_bundle_fixture():
        entries = {
            "visible": {
                "xyz_in_robot_base_coord_m": [0.601, 0.0, 0.0],
            },
            "fixed_a": {
                "xyz_in_robot_base_coord_m": [0.70, 0.10, 0.0],
            },
            "fixed_b": {
                "xyz_in_robot_base_coord_m": [0.72, 0.12, 0.01],
            },
        }
        live = {
            "ok": False,
            "reason": "tracked_point_unavailable",
            "entries": entries,
            "unavailable": {"hidden": "point_unobserved"},
            "observation_sequence": 17,
            "rigid_pair_fusion": {
                "names": ["hidden", "visible"],
                "visual_tracker_pair": {
                    "track_ids": ["track_hidden", "track_visible"],
                    "per_point": {
                        "track_hidden": {
                            "status": "lost",
                            "rejected_candidates_before_acceptance": [
                                {
                                    "reason": "eef_prior_depth_layer_mismatch",
                                    "candidate_method": "previous_observation_feature",
                                    "candidate_status": "observed",
                                    "candidate_depth_m": 0.45,
                                    "predicted_depth_m": 0.50,
                                    "pixel_error_px": 4.0,
                                    "pixel_error_limit_px": 24.0,
                                }
                            ],
                        }
                    },
                },
            },
        }
        kwargs = {
            "controlled_names": ["hidden", "visible"],
            "fixed_names": ["fixed_a", "fixed_b"],
            "current_observation": {
                "position": np.array([0.5, 0.0, 0.0]),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            "anchors_eef": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64
            ),
            "current_sequence": 17,
            "anchor_tolerance_m": 0.05,
            "pos_tol_m": 0.012,
        }
        return live, kwargs

    @staticmethod
    def _terminal_eef_depth_adapter(
        expected_points_robot_base_m,
        *,
        sequence: int = 17,
        depth_offset_m: float = 0.0,
        status_sequences=None,
    ):
        """Build one synchronized evaluator-only RGB-D observation for tests."""

        expected = np.asarray(expected_points_robot_base_m, dtype=np.float64)
        width = height = 100
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), np.nan, dtype=np.float64)
        intrinsics = official_tools.head_camera_intrinsics(width, height)
        for point in expected:
            point_depth = -float(point[2]) + float(depth_offset_m)
            pixel_x = int(
                round(
                    intrinsics.cx
                    + intrinsics.fx * float(point[0]) / -float(point[2])
                )
            )
            pixel_y = int(
                round(
                    intrinsics.cy
                    - intrinsics.fy * float(point[1]) / -float(point[2])
                )
            )
            depth[pixel_y, pixel_x] = point_depth
        if status_sequences is None:
            status = lambda: {"sequence": int(sequence)}
        else:
            status_values = iter(status_sequences)
            status = lambda: {"sequence": int(next(status_values))}
        return SimpleNamespace(
            status=status,
            camera_frame=lambda role: rgb.copy() if role == "head" else None,
            camera_depth_frame=lambda role: (
                depth.copy() if role == "head" else None
            ),
            camera_relative_poses=lambda: {
                "head": {
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                }
            },
        )

    def test_public_wrapper_releases_on_hand_eef_prior_when_closed(self) -> None:
        released: list[str] = []
        motion_released: list[str] = []
        manager = SimpleNamespace(
            deactivate_on_hand_eef_observation_prior=lambda lease_id: released.append(
                lease_id
            ),
            end_motion_retention=lambda lease_id: (
                motion_released.append(lease_id)
                or {"ok": True, "released": True, "lease_id": lease_id}
            ),
        )
        ctx = SimpleNamespace(
            world=SimpleNamespace(_official_tracked_object_distances=manager)
        )

        def fake_implementation(
            _ctx,
            _points,
            *,
            _eef_prior_lease_state,
            _motion_retention_lease_state,
            **_kwargs,
        ):
            _eef_prior_lease_state["lease_id"] = "eef-prior-lease-test"
            _motion_retention_lease_state["lease_id"] = (
                "motion-retention-lease-test"
            )
            yield np.zeros(ACTION_DIM, dtype=np.float32)
            yield np.zeros(ACTION_DIM, dtype=np.float32)

        with mock.patch.object(
            official_tools,
            "_move_tracked_point_impl",
            side_effect=fake_implementation,
        ):
            generator = official_tools.move_tracked_point(ctx, points=[])
            self.assertEqual(np.asarray(next(generator)).shape, (ACTION_DIM,))
            generator.close()

        self.assertEqual(released, ["eef-prior-lease-test"])
        self.assertEqual(motion_released, ["motion-retention-lease-test"])

    def test_request_retention_spans_freeze_planning_and_final_snapshot(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        events: list[tuple[str, bool]] = []

        class RetentionRequiredManager(_RigidTrackedManager):
            def begin_motion_retention(self, names, *, episode_id, timeout_s):
                report = super().begin_motion_retention(
                    names,
                    episode_id=episode_id,
                    timeout_s=timeout_s,
                )
                events.append(("begin", True))
                return report

            def observed_active_points_snapshot(self, names, *, episode_id):
                retained = bool(self._motion_retention_leases)
                events.append(("snapshot", retained))
                if len([event for event, _active in events if event == "snapshot"]) > 1 and not retained:
                    return {
                        "ok": False,
                        "reason": "tracked_point_unavailable",
                        "entries": {},
                        "unavailable": {
                            str(name): "point_not_registered_or_retired"
                            for name in names
                        },
                        "observation_sequence": int(self.adapter.status()["sequence"]),
                        "episode_id": str(episode_id),
                        "session_id": self.session_id,
                        "image_id": self.image_id,
                    }
                return super().observed_active_points_snapshot(
                    names,
                    episode_id=episode_id,
                )

            def end_motion_retention(self, lease_id):
                events.append(("end", bool(self._motion_retention_leases)))
                return super().end_motion_retention(lease_id)

        manager = RetentionRequiredManager(world, adapter, {"known": position})
        world._official_tracked_object_distances = manager
        target = np.asarray(position, dtype=np.float64) + [0.008, 0.0, 0.0]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target.tolist()}],
                    max_steps=120,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(events[0], ("begin", True))
        snapshot_events = [event for event in events if event[0] == "snapshot"]
        self.assertGreaterEqual(len(snapshot_events), 2)
        self.assertTrue(all(active for _event, active in snapshot_events))
        self.assertNotIn(("end", True), events)
        release = result["motion_tracking_retention"]["release"]
        self.assertFalse(release["released"])
        self.assertEqual(
            release["reason"], "bounded_successor_identity_retention"
        )
        self.assertTrue(release["expires_without_successor"])
        self.assertTrue(release["superseded_by_next_matching_request"])
        self.assertFalse(release["coordinates_published_while_unobserved"])
        self.assertEqual(len(manager._motion_retention_leases), 1)

    @staticmethod
    def _left_eef(world):
        state = local_robot_state(
            trunk_q=world.trunk_qpos(),
            arm_left_q=world.arm_qpos_list("left"),
            arm_right_q=world.arm_qpos_list("right"),
            gripper_left_q=world.gripper_qpos_list("left"),
            gripper_right_q=world.gripper_qpos_list("right"),
        )
        return state, eef_pose(state, "left", world.arm_qpos_list("left"))

    def _run_full_n_point_translation_case(
        self,
        point_offsets: list[list[float]],
    ) -> tuple[dict, list[np.ndarray]]:
        """Run the official tool end to end with a rigid N-point bundle."""

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        source = np.asarray(position, dtype=np.float64) + np.asarray(
            point_offsets,
            dtype=np.float64,
        )
        names = [f"marker_{index + 1}" for index in range(len(source))]
        manager = _RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source)),
        )
        world._official_tracked_object_distances = manager
        translation = np.asarray([0.008, -0.004, 0.003], dtype=np.float64)
        targets = [
            {
                "name": name,
                "target_xyz_m": (point + translation).astype(float).tolist(),
            }
            for name, point in zip(names, source)
        ]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=420,
                ),
            )
        return result, actions

    def _start_gate_fixture(self):
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"known": position})
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            observations,
        )
        source_points = np.asarray(position, dtype=np.float64).reshape(1, 3)
        return adapter, world, manager, ctx, capture_state, source_points

    @staticmethod
    def _drive_start_gate(adapter, generator, qvel_for_action):
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                return actions, stopped.value
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES["arm_right_qvel"]][0] = float(
                qvel_for_action(len(actions))
            )
            adapter.update({"robot_r1::proprio": proprio})

    @staticmethod
    def _rigid_pair_activation_report(
        *,
        ok: bool,
        reason: str | None = None,
        max_point_correction_m: float = 0.0,
    ) -> dict:
        reference_distance_m = 0.080
        distance_error_m = 2.0 * float(max_point_correction_m)
        measured_distance_m = reference_distance_m + distance_error_m
        return {
            "ok": bool(ok),
            "reason": None if ok else str(reason),
            "reference_distance_m": reference_distance_m,
            "measured_distance_m": measured_distance_m,
            "fused_distance_m": reference_distance_m if ok else None,
            "distance_error_m": distance_error_m,
            "point_correction_m": [
                float(max_point_correction_m),
                float(max_point_correction_m),
            ],
            "max_point_correction_m": float(max_point_correction_m),
            "correction_limit_m": 0.035,
            "fusion_applied": bool(ok),
        }

    @staticmethod
    def _assert_rigid_pair_hold_actions(actions) -> None:
        for action in actions:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape != (ACTION_DIM,):
                raise AssertionError(
                    f"activation hold action shape {action.shape} != {(ACTION_DIM,)}"
                )
            if not np.all(np.isfinite(action)):
                raise AssertionError("activation hold action contains non-finite values")
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                for side in ("left", "right"):
                    if float(action[ACTION_SLICES[f"arm_{side}"]][7]) != 0.0:
                        raise AssertionError(f"{side} J8 was not locked to zero")

    def _rigid_pair_activation_fixture(self, *, initial_report, fresh_reports):
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        names = ["hand-a", "hand-b"]
        anchors = np.asarray(
            [[-0.040, 0.0, 0.0], [0.040, 0.0, 0.0]],
            dtype=np.float64,
        )
        points = np.asarray(position, dtype=np.float64)[None, :] + (
            rotation @ anchors.T
        ).T
        manager = _ScriptedRigidPairActivationManager(
            world,
            adapter,
            dict(zip(names, points)),
            initial_report=initial_report,
            fresh_reports=fresh_reports,
        )
        world._official_tracked_object_distances = manager
        ctx, result = self._ctx(world)
        sequence = int(adapter.status()["sequence"])
        return adapter, world, manager, ctx, result, names, points, sequence

    @staticmethod
    def _rigid_pair_activation_hold_state(ctx):
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        return official_tools._move_tracked_capture_state(ctx, observations)

    def test_registration_eef_bootstrap_uses_frozen_rgbd_and_capture_fk(
        self,
    ) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        current_state = self._rigid_pair_activation_hold_state(ctx)
        left_position = np.asarray(
            current_state["eef"]["left"]["position"], dtype=np.float64
        )
        left_rotation = quat_to_mat_xyzw(
            current_state["eef"]["left"]["quaternion"]
        )
        expected_anchors = np.asarray(
            [[-0.052, 0.004, 0.002], [0.052, -0.003, 0.001]],
            dtype=np.float64,
        )
        registration_points = left_position[None, :] + (
            left_rotation @ expected_anchors.T
        ).T
        registration_sequence = 37
        calls = {}

        def registered_points_snapshot(names, *, episode_id):
            calls["snapshot"] = (list(names), str(episode_id))
            return {
                "ok": True,
                "reason": None,
                "entries": {
                    name: {
                        "xyz_in_robot_base_coord_m": registration_points[index]
                        .astype(float)
                        .tolist()
                    }
                    for index, name in enumerate(names)
                },
                "session_id": "bootstrap-session",
                "image_id": "img_bootstrap",
                "episode_id": str(episode_id),
                "registration_observation_sequence": registration_sequence,
                "measurement_source": (
                    "policy_owned_frozen_head_rgbd_registration"
                ),
                "predictions_published": False,
            }

        def activate_prior(names, **kwargs):
            calls["activation"] = (list(names), dict(kwargs))
            return {
                "ok": True,
                "lease_id": "bootstrap-prior-lease",
                "activation_observation_sequence": int(
                    adapter.status()["sequence"]
                ),
                "max_pixel_error_px": 24.0,
                "max_depth_error_m": 0.025,
                "max_point_error_m": 0.030,
                "prediction_values_published_as_observation": False,
            }

        manager = SimpleNamespace(
            registered_points_snapshot=registered_points_snapshot,
            activate_on_hand_eef_observation_prior=activate_prior,
        )
        frozen_capture = SimpleNamespace(
            evaluator_sequence=registration_sequence
        )
        with mock.patch.object(
            official_tools,
            "_load_frozen_capture",
            return_value=frozen_capture,
        ) as load_capture, mock.patch.object(
            official_tools,
            "_move_point_capture_kinematics",
            return_value=current_state,
        ):
            report = official_tools._move_tracked_bootstrap_on_hand_eef_prior(
                manager,
                ["hand-a", "hand-b"],
                episode_id=str(ctx.world.episode_id()),
                current_capture_state=current_state,
            )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["active"], report)
        self.assertEqual(report["selected_arm"], "left")
        self.assertFalse(report["prediction_values_published_as_observation"])
        self.assertEqual(
            calls["snapshot"],
            (["hand-a", "hand-b"], str(ctx.world.episode_id())),
        )
        load_capture.assert_called_once_with(
            "bootstrap-session", "img_bootstrap"
        )
        activation_names, activation_kwargs = calls["activation"]
        self.assertEqual(activation_names, ["hand-a", "hand-b"])
        self.assertEqual(activation_kwargs["arm"], "left")
        np.testing.assert_allclose(
            activation_kwargs["anchors_eef_m"],
            expected_anchors,
            atol=1e-9,
        )

    def _registered_npoint_source_fixture(self):
        _adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        current_state = self._rigid_pair_activation_hold_state(ctx)
        left_position = np.asarray(
            current_state["eef"]["left"]["position"], dtype=np.float64
        )
        left_rotation = quat_to_mat_xyzw(
            current_state["eef"]["left"]["quaternion"]
        )
        names = ["hand-a", "hand-b", "hand-c", "hand-d"]
        anchors = np.asarray(
            [
                [-0.050, 0.004, 0.002],
                [0.050, -0.003, 0.001],
                [-0.012, 0.002, -0.055],
                [-0.025, 0.003, -0.115],
            ],
            dtype=np.float64,
        )
        registered = left_position[None, :] + (left_rotation @ anchors.T).T
        registration_sequence = 41

        def registered_points_snapshot(requested, *, episode_id):
            return {
                "ok": True,
                "reason": None,
                "entries": {
                    name: {
                        "xyz_in_robot_base_coord_m": registered[
                            names.index(name)
                        ].tolist()
                    }
                    for name in requested
                },
                "session_id": "npoint-registration-session",
                "image_id": "img_npoint_registration",
                "episode_id": str(episode_id),
                "registration_observation_sequence": registration_sequence,
                "measurement_source": (
                    "policy_owned_frozen_head_rgbd_registration"
                ),
            }

        manager = SimpleNamespace(
            registered_points_snapshot=registered_points_snapshot,
        )
        frozen_capture = SimpleNamespace(
            evaluator_sequence=registration_sequence,
        )
        return (
            ctx,
            current_state,
            manager,
            frozen_capture,
            names,
            registered,
        )

    def test_registered_npoint_source_recovers_one_bounded_tracker_outlier(
        self,
    ) -> None:
        (
            ctx,
            current_state,
            manager,
            frozen_capture,
            names,
            registered,
        ) = self._registered_npoint_source_fixture()
        live = registered.copy()
        live[0] += np.asarray([0.004, 0.0, 0.0])
        live[1] += np.asarray([0.0, -0.002, 0.0])
        live[2] += np.asarray([0.0, 0.023, 0.0])
        live[3] += np.asarray([0.0, 0.0, 0.001])
        with mock.patch.object(
            official_tools,
            "_load_frozen_capture",
            return_value=frozen_capture,
        ), mock.patch.object(
            official_tools,
            "_move_point_capture_kinematics",
            return_value=current_state,
        ):
            report = (
                official_tools._move_tracked_reconcile_registered_on_hand_source(
                    manager,
                    names,
                    episode_id=str(ctx.world.episode_id()),
                    current_capture_state=current_state,
                    live_points_robot_base_m=live,
                )
            )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["applicable"], report)
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["selected_arm"], "left")
        self.assertEqual(report["inlier_names"], ["hand-a", "hand-b", "hand-d"])
        self.assertEqual(report["recovered_point_names"], ["hand-c"])
        self.assertEqual(report["required_inlier_count"], 3)
        self.assertFalse(report["predictions_published_as_observation"])
        np.testing.assert_allclose(
            report["source_points_robot_base_m"],
            registered,
            atol=1e-9,
        )

    def test_registered_npoint_source_preserves_consistent_live_fast_path(
        self,
    ) -> None:
        (
            ctx,
            current_state,
            manager,
            frozen_capture,
            names,
            registered,
        ) = self._registered_npoint_source_fixture()
        live = registered + np.asarray(
            [
                [0.001, 0.0, 0.0],
                [0.0, -0.001, 0.0],
                [0.0, 0.0, 0.001],
                [-0.001, 0.0, 0.0],
            ]
        )
        with mock.patch.object(
            official_tools,
            "_load_frozen_capture",
            return_value=frozen_capture,
        ), mock.patch.object(
            official_tools,
            "_move_point_capture_kinematics",
            return_value=current_state,
        ):
            report = (
                official_tools._move_tracked_reconcile_registered_on_hand_source(
                    manager,
                    names,
                    episode_id=str(ctx.world.episode_id()),
                    current_capture_state=current_state,
                    live_points_robot_base_m=live,
                )
            )

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["applied"], report)
        self.assertEqual(
            report["reason"],
            "all_live_points_agree_with_registered_eef_anchors",
        )
        self.assertFalse(report["planning_source_is_model_derived"])
        np.testing.assert_allclose(
            report["source_points_robot_base_m"],
            live,
            atol=0.0,
        )

    def test_registered_npoint_source_recovery_fails_closed_outside_bounds(
        self,
    ) -> None:
        (
            ctx,
            current_state,
            manager,
            frozen_capture,
            names,
            registered,
        ) = self._registered_npoint_source_fixture()
        cases = {
            "excessive_outlier": (
                np.asarray(
                    [
                        [0.001, 0.0, 0.0],
                        [0.0, -0.001, 0.0],
                        [0.0, 0.031, 0.0],
                        [0.0, 0.0, 0.001],
                    ]
                ),
                "registered_eef_anchor_recovery_exceeded",
            ),
            "no_strict_majority": (
                np.asarray(
                    [
                        [0.001, 0.0, 0.0],
                        [0.0, 0.012, 0.0],
                        [0.0, 0.018, 0.0],
                        [0.0, 0.0, 0.001],
                    ]
                ),
                "registered_eef_anchor_consensus_insufficient",
            ),
        }
        for label, (delta, expected_reason) in cases.items():
            with self.subTest(label=label), mock.patch.object(
                official_tools,
                "_load_frozen_capture",
                return_value=frozen_capture,
            ), mock.patch.object(
                official_tools,
                "_move_point_capture_kinematics",
                return_value=current_state,
            ):
                report = (
                    official_tools._move_tracked_reconcile_registered_on_hand_source(
                        manager,
                        names,
                        episode_id=str(ctx.world.episode_id()),
                        current_capture_state=current_state,
                        live_points_robot_base_m=registered + delta,
                    )
                )
            self.assertFalse(report["ok"], report)
            self.assertFalse(report["applied"], report)
            self.assertEqual(report["reason"], expected_reason)

    def test_fresh_rigid_pair_activation_clean_fast_path_yields_no_actions(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=initial,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        self.assertEqual(actions, [])
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["required"])
        self.assertFalse(report["recovered"])
        self.assertEqual(report["attempt_count"], 1)
        self.assertEqual(report["recovery_hold_steps"], 0)
        self.assertEqual(report["confirmation_source"], "initial_activation_fast_path")
        self.assertEqual(manager.snapshot_sequences, [])

    def test_large_estimator_pair_projection_requires_fresh_confirmation(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(
            ok=True,
            max_point_correction_m=0.014787330308532654,
        )
        gated = official_tools._move_tracked_gate_rigid_pair_source_projection(
            initial
        )
        self.assertTrue(gated["tracker_estimator_ok"])
        self.assertFalse(gated["ok"])
        self.assertFalse(
            gated["direct_rgbd_projection_accepted_for_move_source"]
        )
        self.assertTrue(gated["fresh_observation_confirmation_required"])
        self.assertEqual(
            gated["reason"],
            "rigid_pair_projection_requires_fresh_confirmation",
        )
        self.assertAlmostEqual(
            gated["move_source_max_point_correction_m"],
            0.003,
        )

        direct = official_tools._move_tracked_gate_rigid_pair_source_projection(
            self._rigid_pair_activation_report(
                ok=True,
                max_point_correction_m=0.002999,
            )
        )
        self.assertTrue(direct["ok"], direct)
        self.assertTrue(direct["direct_rgbd_projection_accepted_for_move_source"])

    def test_fresh_rigid_pair_activation_honors_expired_request_deadline(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[self._rigid_pair_activation_report(ok=True)],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=failed,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
            deadline_monotonic=-1.0,
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        self.assertEqual(actions, [])
        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "rigid_pair_activation_recovery_timeout"
        )
        self.assertEqual(report["recovery_hold_steps"], 0)
        self.assertEqual(manager.snapshot_sequences, [])

    def test_fresh_rigid_pair_activation_checks_deadline_after_hold(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[self._rigid_pair_activation_report(ok=True)],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=failed,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
            deadline_monotonic=1.0,
        )

        with mock.patch.object(
            official_tools.time,
            "monotonic",
            side_effect=[0.0, 0.0, 2.0],
        ):
            actions, report = self._drive_start_gate(
                adapter,
                generator,
                lambda _action_index: 0.0,
            )

        self.assertEqual(len(actions), 1)
        self._assert_rigid_pair_hold_actions(actions)
        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "rigid_pair_activation_recovery_timeout"
        )
        self.assertEqual(report["recovery_hold_steps"], 1)
        self.assertEqual(manager.snapshot_sequences, [])

    def test_fresh_rigid_pair_activation_recovers_after_two_valid_frames(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        valid = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[valid, valid],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=initial,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        self.assertEqual(len(actions), 2)
        self._assert_rigid_pair_hold_actions(actions)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["required"])
        self.assertTrue(report["recovered"])
        self.assertEqual(report["attempt_count"], 3)
        self.assertEqual(report["recovery_hold_steps"], 2)
        self.assertEqual(report["stable_steps"], 2)
        self.assertAlmostEqual(
            report["first_failure"]["max_point_correction_m"],
            0.126,
        )
        self.assertEqual(
            manager.snapshot_sequences,
            [sequence + 1, sequence + 2],
        )
        self.assertTrue(
            all(
                later > earlier
                for earlier, later in zip(
                    [sequence, *manager.snapshot_sequences[:-1]],
                    manager.snapshot_sequences,
                )
            )
        )

    def test_fresh_rigid_pair_activation_recovers_initial_unobserved_pair(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_point_unobserved",
        )
        initial.update(
            {
                "unavailable": ["hand-b"],
                "fusion_applied": False,
            }
        )
        valid = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[valid, valid],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=initial,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        self.assertEqual(len(actions), 2)
        self._assert_rigid_pair_hold_actions(actions)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["recovered"])
        self.assertEqual(report["attempt_count"], 3)
        self.assertEqual(report["recovery_hold_steps"], 2)
        self.assertEqual(report["stable_steps"], 2)
        self.assertEqual(
            report["first_failure"]["reason"],
            "rigid_pair_point_unobserved",
        )
        self.assertEqual(
            manager.snapshot_sequences,
            [sequence + 1, sequence + 2],
        )

    def test_fresh_rigid_pair_activation_persistent_failure_is_bounded(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[failed],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=failed,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        expected_holds = (
            official_tools.MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS
        )
        self.assertEqual(len(actions), expected_holds)
        self._assert_rigid_pair_hold_actions(actions)
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "rigid_pair_activation_recovery_exhausted")
        self.assertEqual(report["attempt_count"], expected_holds + 1)
        self.assertEqual(report["recovery_hold_steps"], expected_holds)
        self.assertEqual(len(report["history"]), expected_holds)
        self.assertEqual(
            manager.snapshot_sequences,
            list(range(sequence + 1, sequence + expected_holds + 1)),
        )
        self.assertAlmostEqual(
            report["last_pair_report"]["max_point_correction_m"],
            0.126,
        )

    def test_non_pair_request_retires_prior_pair_without_deleting_points(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(ok=True)
        (
            _adapter,
            _world,
            manager,
            _ctx,
            _result,
            names,
            _points,
            _sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[],
        )
        manager.active_pair = {"names": list(names)}

        report = official_tools._move_tracked_clear_incompatible_rigid_pair(
            manager,
            [names[0]],
        )

        self.assertTrue(report["ok"], report)
        self.assertIsNone(manager.active_pair)
        self.assertEqual(
            manager.deactivation_reasons,
            ["rigid_pair_not_part_of_current_controlled_point_set"],
        )
        self.assertEqual(set(manager.anchors), set(names))

    def test_hot_reload_legacy_pair_retirement_restores_raw_rgbd_rows(
        self,
    ) -> None:
        class LegacyManager:
            def __init__(self):
                self._lock = threading.RLock()
                self._active_rigid_pair = {"names": ["A", "B"]}
                self._temporarily_unobserved = {}
                self._entries = {
                    "A": {
                        "name": "A",
                        "status": "observed",
                        "observation_sequence": 41,
                        "depth_m": 0.4,
                        "u": 10.0,
                        "v": 20.0,
                        "xyz_in_robot_base_coord_m": [9.0, 9.0, 9.0],
                        "xyz_source": "registration_rigid_pair_projection",
                        "raw_xyz_in_robot_base_coord_m": [1.0, 2.0, 3.0],
                        "raw_xyz_source": "current_evaluator_depth_linear",
                        "rigid_pair_fusion": {"ok": True},
                    },
                    "B": {
                        "name": "B",
                        "status": "observed",
                        "observation_sequence": 41,
                        "depth_m": 0.5,
                        "u": 30.0,
                        "v": 40.0,
                        "xyz_in_robot_base_coord_m": [8.0, 8.0, 8.0],
                        "xyz_source": "registration_rigid_pair_projection",
                        "rigid_pair_fusion": {"ok": True},
                    },
                    "other": {
                        "name": "other",
                        "status": "observed",
                        "xyz_in_robot_base_coord_m": [4.0, 5.0, 6.0],
                        "xyz_source": "current_evaluator_depth_linear",
                    },
                }

            def _retire_active_rigid_pair_locked(self, *, reason):
                previous = self._active_rigid_pair
                self._active_rigid_pair = None
                return {"reason": str(reason), "previous_pair": previous}

        manager = LegacyManager()
        report = official_tools._move_tracked_deactivate_rigid_pair(
            manager,
            reason="unit_test_scope_end",
            required=True,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["hot_reload_compatibility_path"])
        np.testing.assert_allclose(
            manager._entries["A"]["xyz_in_robot_base_coord_m"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(manager._entries["A"]["status"], "observed")
        self.assertIsNone(manager._entries["B"]["xyz_in_robot_base_coord_m"])
        self.assertEqual(manager._entries["B"]["status"], "temporarily_unobserved")
        self.assertIn("B", manager._temporarily_unobserved)
        np.testing.assert_allclose(
            manager._entries["other"]["xyz_in_robot_base_coord_m"],
            [4.0, 5.0, 6.0],
        )
        self.assertEqual(report["entry_retirement"]["restored_raw_names"], ["A"])
        self.assertEqual(
            report["entry_retirement"]["temporarily_unobserved_names"],
            ["B"],
        )

    def test_clear_incompatible_rigid_pair_respects_exact_pair_scope(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(ok=True)
        cases = (
            ("exact_pair", ["A", "B"], False),
            ("different_pair", ["C", "D"], True),
            ("single_point", ["A"], True),
            ("n_point_bundle", ["A", "B", "C"], True),
        )
        for case_name, requested_names, should_deactivate in cases:
            with self.subTest(case=case_name):
                (
                    _adapter,
                    _world,
                    manager,
                    _ctx,
                    _result,
                    registered_names,
                    _points,
                    _sequence,
                ) = self._rigid_pair_activation_fixture(
                    initial_report=initial,
                    fresh_reports=[],
                )
                manager.active_pair = {"names": ["A", "B"]}
                registered_before = {
                    name: np.asarray(anchor, dtype=np.float64).copy()
                    for name, anchor in manager.anchors.items()
                }

                report = official_tools._move_tracked_clear_incompatible_rigid_pair(
                    manager,
                    requested_names,
                )

                self.assertTrue(report["ok"], report)
                if should_deactivate:
                    self.assertIsNone(manager.active_pair)
                    self.assertEqual(
                        manager.deactivation_reasons,
                        ["rigid_pair_not_part_of_current_controlled_point_set"],
                    )
                else:
                    self.assertEqual(manager.active_pair, {"names": ["A", "B"]})
                    self.assertEqual(manager.deactivation_reasons, [])
                    self.assertEqual(
                        report["reason"],
                        "current_request_will_reactivate_exact_pair",
                    )
                self.assertEqual(set(manager.anchors), set(registered_names))
                for name, anchor in registered_before.items():
                    np.testing.assert_allclose(manager.anchors[name], anchor)

    def test_fresh_rigid_pair_activation_same_sequence_refuses_retry(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[self._rigid_pair_activation_report(ok=True)],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=failed,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        action = np.asarray(next(generator), dtype=np.float32)
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        report = stopped.exception.value

        self._assert_rigid_pair_hold_actions([action])
        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"],
            "rigid_pair_activation_observation_not_advanced",
        )
        self.assertEqual(report["attempt_count"], 1)
        self.assertEqual(report["recovery_hold_steps"], 1)
        self.assertEqual(manager.snapshot_sequences, [])

    def test_fresh_rigid_pair_activation_nonretryable_failure_yields_no_actions(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="registration_reference_unavailable",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            _world,
            manager,
            ctx,
            _result,
            names,
            _points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[self._rigid_pair_activation_report(ok=True)],
        )
        generator = official_tools._move_tracked_confirm_fresh_rigid_pair_activation(
            ctx,
            manager,
            names,
            episode_id=str(ctx.world.episode_id()),
            adapter=adapter,
            observation_sequence=sequence,
            initial_report=failed,
            hold_state=self._rigid_pair_activation_hold_state(ctx),
        )

        actions, report = self._drive_start_gate(
            adapter,
            generator,
            lambda _action_index: 0.0,
        )

        self.assertEqual(actions, [])
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "registration_reference_unavailable")
        self.assertEqual(report["attempt_count"], 1)
        self.assertEqual(report["recovery_hold_steps"], 0)
        self.assertEqual(manager.snapshot_sequences, [])

    def test_public_move_confirms_large_successful_pair_projection_before_planning(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(
            ok=True,
            max_point_correction_m=0.014787330308532654,
        )
        valid = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            world,
            manager,
            ctx,
            result,
            names,
            points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[valid, valid],
        )
        targets = [
            {
                "name": name,
                "target_xyz_m": point.astype(float).tolist(),
            }
            for name, point in zip(names, points)
        ]
        if ARM_DOF == 8:
            world.set_tool_roll_pin_qpos("left", 0.7)
            world.set_tool_roll_pin_qpos("right", -0.4)
            world.begin_tool_roll_motion("left")
            world.begin_tool_roll_motion("right")
        observed_arm_q = {
            side: np.asarray(world.arm_qpos_list(side), dtype=np.float64)
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            stale_pin = observed_arm_q[side].copy()
            stale_pin[0] += 0.20
            world.set_arm_pin_qpos(side, stale_pin)

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=120,
                    timeout_s=30.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(manager.activation_sequences, [sequence])
        self.assertGreaterEqual(len(actions), 2)
        self._assert_rigid_pair_hold_actions(actions[:2])
        for action in actions[:2]:
            for side in ("left", "right"):
                expected_q = observed_arm_q[side].copy()
                if ARM_DOF == 8:
                    expected_q[7] = 0.0
                np.testing.assert_allclose(
                    action[ACTION_SLICES[f"arm_{side}"]],
                    expected_q,
                    atol=1e-7,
                )
        confirmation = result["rigid_pair_activation_confirmation"]
        self.assertTrue(confirmation["ok"], confirmation)
        self.assertTrue(confirmation["recovered"])
        self.assertEqual(confirmation["recovery_hold_steps"], 2)
        self.assertEqual(result["rigid_pair_activation_hold_steps"], 2)
        self.assertEqual(manager.deactivation_reasons, [])
        self.assertEqual(
            manager.snapshot_sequences[:2],
            [sequence + 1, sequence + 2],
        )
        self.assertAlmostEqual(
            result["rigid_pair_activation"]["initial_activation_report"][
                "max_point_correction_m"
            ],
            0.014787330308532654,
        )
        self.assertEqual(
            result["rigid_pair_activation"]["initial_activation_report"]["reason"],
            "rigid_pair_projection_requires_fresh_confirmation",
        )
        self.assertAlmostEqual(
            result["rigid_pair_activation"]["max_point_correction_m"],
            0.0,
        )
        self.assertTrue(result["rigid_pair_activation"]["fusion_applied"])

    def test_public_move_persistent_rigid_pair_activation_failure_is_bounded(
        self,
    ) -> None:
        failed = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_rgbd_correction_exceeded",
            max_point_correction_m=0.126,
        )
        (
            adapter,
            world,
            manager,
            ctx,
            result,
            names,
            points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=failed,
            fresh_reports=[failed],
        )
        targets = [
            {
                "name": name,
                "target_xyz_m": point.astype(float).tolist(),
            }
            for name, point in zip(names, points)
        ]
        if ARM_DOF == 8:
            world.set_tool_roll_pin_qpos("left", 0.7)
            world.set_tool_roll_pin_qpos("right", -0.4)
            world.begin_tool_roll_motion("left")
            world.begin_tool_roll_motion("right")
        planner_calls = []

        def planner_must_not_run(**kwargs):
            planner_calls.append(kwargs)
            raise AssertionError("planner ran after rigid-pair activation failure")

        with mock.patch.object(
            official_tools,
            "_move_tracked_plan_frozen",
            side_effect=planner_must_not_run,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=120,
                    timeout_s=30.0,
                ),
            )

        expected_holds = (
            official_tools.MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failure_stage"], "observation validation")
        self.assertEqual(planner_calls, [])
        self.assertEqual(len(actions), expected_holds + 1)
        self._assert_rigid_pair_hold_actions(actions)
        self.assertEqual(manager.activation_sequences, [sequence])
        self.assertEqual(
            manager.snapshot_sequences,
            list(range(sequence + 1, sequence + expected_holds + 1)),
        )
        confirmation = result["rigid_pair_activation_confirmation"]
        self.assertFalse(confirmation["ok"], confirmation)
        self.assertEqual(
            confirmation["reason"],
            "rigid_pair_activation_recovery_exhausted",
        )
        self.assertEqual(
            confirmation["recovery_hold_steps"],
            expected_holds,
        )
        self.assertEqual(result["rigid_pair_activation_hold_steps"], expected_holds)
        self.assertEqual(
            manager.deactivation_reasons,
            ["rigid_pair_activation_recovery_failed"],
        )
        self.assertAlmostEqual(
            confirmation["first_failure"]["max_point_correction_m"],
            0.126,
        )
        self.assertIn("required_point_correction=0.126000m", result["error"])
        self.assertIn("correction_limit=0.035000m", result["error"])

    def test_public_move_close_during_first_recovery_hold_retires_pair(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(
            ok=False,
            reason="rigid_pair_point_unobserved",
        )
        valid = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            world,
            manager,
            ctx,
            _result,
            names,
            points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[valid, valid],
        )
        targets = [
            {
                "name": name,
                "target_xyz_m": point.astype(float).tolist(),
            }
            for name, point in zip(names, points)
        ]
        generator = build_registry(adapter)["move_tracked_point"].fn(
            ctx,
            points=targets,
            max_steps=120,
            timeout_s=30.0,
        )

        first_action = np.asarray(next(generator), dtype=np.float32)
        self._assert_rigid_pair_hold_actions([first_action])
        self.assertEqual(manager.activation_sequences, [sequence])
        self.assertEqual(manager.active_pair, {"names": names})
        self.assertEqual(manager.snapshot_sequences, [])

        generator.close()

        self.assertIsNone(manager.active_pair)
        self.assertEqual(
            manager.deactivation_reasons,
            ["rigid_pair_activation_recovery_interrupted"],
        )
        self.assertEqual(manager.snapshot_sequences, [])

    def test_public_move_planner_failure_after_clean_activation_retires_pair(
        self,
    ) -> None:
        initial = self._rigid_pair_activation_report(ok=True)
        (
            adapter,
            world,
            manager,
            ctx,
            result,
            names,
            points,
            sequence,
        ) = self._rigid_pair_activation_fixture(
            initial_report=initial,
            fresh_reports=[],
        )
        targets = [
            {
                "name": name,
                "target_xyz_m": point.astype(float).tolist(),
            }
            for name, point in zip(names, points)
        ]
        planner_calls = []

        def fail_planner(**kwargs):
            planner_calls.append(kwargs)
            raise RuntimeError("forced planner failure after clean pair activation")

        with mock.patch.object(
            official_tools,
            "_move_tracked_plan_frozen",
            side_effect=fail_planner,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=120,
                    timeout_s=30.0,
                ),
            )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failure_stage"], "planning")
        self.assertIn("forced planner failure", result["error"])
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(manager.activation_sequences, [sequence])
        self.assertIsNone(manager.active_pair)
        self.assertEqual(
            manager.deactivation_reasons,
            ["move_tracked_point_request_did_not_complete_successfully"],
        )
        self.assertGreaterEqual(len(actions), 1)
        self._assert_rigid_pair_hold_actions(actions)

    def test_contract_accepts_numeric_and_symbolic_targets(self) -> None:
        numeric = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {
                        "name": "axe_tip",
                        "target_xyz_m": {"x": 0.4, "y": -0.2, "z": 0.8},
                    }
                ]
            },
        )
        self.assertEqual(numeric["points"][0]["target_xyz_m"]["x"], 0.4)
        self.assertEqual(numeric["execution_mode"], "exec")
        self.assertEqual(numeric["pos_tol"], 0.03)
        self.assertEqual(numeric["ori_tol_deg"], 20.0)
        strict_override = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {
                        "name": "axe_tip",
                        "target_xyz_m": [0.4, -0.2, 0.8],
                    }
                ],
                "pos_tol": 0.012,
                "ori_tol_deg": 5.0,
            },
        )
        self.assertEqual(strict_override["pos_tol"], 0.012)
        self.assertEqual(strict_override["ori_tol_deg"], 5.0)
        plan_only = validate_submission(
            "move_tracked_point",
            {
                "execution_mode": " PLAN ",
                "points": [
                    {
                        "name": "axe_tip",
                        "target_xyz_m": [0.4, -0.2, 0.8],
                    }
                ],
            },
        )
        self.assertEqual(plan_only["execution_mode"], "plan")
        for invalid_execution_mode in (None, True, 1, "", "preview", "execute"):
            with self.subTest(invalid_execution_mode=invalid_execution_mode):
                with self.assertRaises(OfficialToolBoundaryError):
                    validate_submission(
                        "move_tracked_point",
                        {
                            "execution_mode": invalid_execution_mode,
                            "points": [
                                {
                                    "name": "axe_tip",
                                    "target_xyz_m": [0.4, -0.2, 0.8],
                                }
                            ],
                        },
                    )
        symbolic = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {
                        "name": "a",
                        "target_xyz_m": [
                            {"var": "common_x"},
                            {"var": "common_y"},
                            {"var": "za"},
                        ],
                    },
                    {
                        "name": "b",
                        "target_xyz_m": {
                            "x": {"var": "common_x"},
                            "y": {"var": "common_y"},
                            "z": {"var": "zb"},
                        },
                    },
                ]
            },
        )
        self.assertEqual(
            symbolic["points"][1]["target_xyz_m"]["x"],
            {"var": "common_x"},
        )
        symbolic_names = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {"name": "a", "target_xyz_m": ["x", "y", "za"]},
                    {"name": "b", "target_xyz_m": ["x", "y", "zb"]},
                ]
            },
        )
        self.assertEqual(
            symbolic_names["points"][0]["target_xyz_m"]["x"],
            {"var": "x"},
        )
        for invalid in (
            {"points": []},
            {
                "points": [
                    {"name": "a", "target_xyz_m": [0, 0, 0]},
                    {"name": "b", "target_xyz_m": [0, 0, 0]},
                    {"name": "c", "target_xyz_m": [0, 0, 0]},
                    {"name": "d", "target_xyz_m": [0, 0, 0]},
                    {"name": "e", "target_xyz_m": [0, 0, 0]},
                    {"name": "f", "target_xyz_m": [0, 0, 0]},
                    {"name": "g", "target_xyz_m": [0, 0, 0]},
                ]
            },
            {
                "points": [
                    {
                        "name": "a",
                        "target_xyz_m": ["x*y", 0, 0],
                    }
                ]
            },
            {
                "points": [
                    {
                        "name": "a",
                        "target_xyz_m": ["x0.1", 0, 0],
                    }
                ]
            },
            {
                "points": [
                    {
                        "name": "a",
                        "target_xyz_m": ["x+*0.1", 0, 0],
                    }
                ]
            },
            {
                "points": [
                    {
                        "name": "a",
                        "target_xyz_m": [{"var": "bad-name"}, 0, 0],
                    }
                ]
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(OfficialToolBoundaryError):
                    validate_submission("move_tracked_point", invalid)

        affine = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {"name": "head", "target_xyz_m": ["a", "b", "z"]},
                    {
                        "name": "tail",
                        "target_xyz_m": ["a", "b", "z+0.08"],
                    },
                ]
            },
        )
        self.assertEqual(
            affine["points"][1]["target_xyz_m"]["z"],
            {"expr": "z+0.08"},
        )
        for expression in ("2*z", "(a+b)/2", "z - 0.08", "-a+3*b"):
            with self.subTest(expression=expression):
                normalized = validate_submission(
                    "move_tracked_point",
                    {
                        "points": [
                            {
                                "name": "p",
                                "target_xyz_m": [expression, "?", "?"],
                            }
                        ]
                    },
                )
                self.assertEqual(
                    normalized["points"][0]["target_xyz_m"]["x"],
                    {"expr": expression.replace(" ", "")},
                )

    def test_http_route_and_dropdown_metadata_support_numeric_and_variables(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-move", "job-head"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    {
                        "ok": True,
                        "tool": "move_tracked_point",
                        "selected_arm": "left",
                    },
                    {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-after-move",
                        "rgb_main": "data:image/jpeg;base64,after-move",
                    },
                ]
            ),
        )
        runtime = SimpleNamespace(server=server)
        app = Flask(__name__)

        @app.get("/api/v2/tools")
        def api_v2_tools():
            return jsonify({
                "tool_version": "official_v2",
                "tools": [
                    {
                        "name": "track_object_distance",
                        "endpoint": "/api/v2/track_object_distance",
                        "args": [],
                    },
                    {
                        "name": "move_tracked_point",
                        "endpoint": "/stale",
                        "args": [],
                    },
                    {
                        "name": "open_gripper",
                        "endpoint": "/api/v2/open_gripper",
                        "args": [],
                    },
                ],
            })

        install_official_move_tracked_point_routes(app, runtime)
        client = app.test_client()
        metadata = client.get("/api/v2/tools").get_json()["tools"]
        self.assertEqual(
            sum(item["name"] == "move_tracked_point" for item in metadata),
            1,
        )
        metadata_names = [item["name"] for item in metadata]
        self.assertEqual(
            metadata_names.index("move_tracked_point"),
            metadata_names.index("track_object_distance") + 1,
        )
        move_metadata = next(
            item for item in metadata if item["name"] == "move_tracked_point"
        )
        self.assertEqual(
            move_metadata["endpoint"],
            "/api/v2/move_tracked_point",
        )
        points_metadata = next(
            arg for arg in move_metadata["args"] if arg["name"] == "points"
        )
        self.assertEqual(points_metadata["widget"], "tracked_target_points")
        self.assertEqual(points_metadata["min_points"], 1)
        self.assertEqual(points_metadata["max_points"], 6)
        vertical_metadata = next(
            item for item in points_metadata["quick_presets"]
            if item["type"] == "vertical_to_ground"
        )
        self.assertEqual(vertical_metadata["on_hand_count"], [2, 3])
        self.assertEqual(vertical_metadata["off_hand_count"], 0)
        faceto_metadata = next(
            item for item in points_metadata["quick_presets"]
            if item["type"] == "faceto"
        )
        self.assertEqual(faceto_metadata, {
            "type": "faceto", "on_hand_count": 3, "off_hand_count": 0,
        })
        reverse_faceto_metadata = next(
            item for item in points_metadata["quick_presets"]
            if item["type"] == "reverse_faceto"
        )
        self.assertEqual(reverse_faceto_metadata, {
            "type": "reverse_faceto", "on_hand_count": 3, "off_hand_count": 0,
        })
        execution_metadata = next(
            arg
            for arg in move_metadata["args"]
            if arg["name"] == "execution_mode"
        )
        self.assertEqual(execution_metadata["widget"], "select")
        self.assertEqual(execution_metadata["default"], "exec")
        self.assertEqual(execution_metadata["options"], ["exec", "plan"])
        inequality_metadata = next(
            arg
            for arg in move_metadata["args"]
            if arg["name"] == "inequalities"
        )
        self.assertEqual(inequality_metadata["default"], [])
        self.assertEqual(inequality_metadata["max_items"], 6)
        self.assertEqual(
            inequality_metadata["items"]["properties"]["op"]["enum"],
            [">", "<"],
        )
        pos_tol_metadata = next(
            arg for arg in move_metadata["args"] if arg["name"] == "pos_tol"
        )
        ori_tol_metadata = next(
            arg for arg in move_metadata["args"] if arg["name"] == "ori_tol_deg"
        )
        self.assertEqual(pos_tol_metadata["default"], 0.03)
        self.assertEqual(ori_tol_metadata["default"], 20.0)
        self.assertTrue(any(arg["name"] == "relations" for arg in move_metadata["args"]))
        self.assertIn("same rigid object", move_metadata["desc"])
        self.assertIn("identifier used once", move_metadata["desc"])

        response = client.post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "web-move",
                "points": [
                    {
                        "name": "axe_a",
                        "target_xyz_m": {
                            "x": "shared_x",
                            "y": 0.12,
                            "z": "za",
                        },
                    },
                    {
                        "name": "axe_b",
                        "target_xyz_m": {
                            "x": {"var": "shared_x"},
                            "y": 0.12,
                            "z": {"var": "zb"},
                        },
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["image_id"], "img-after-move")
        submitted = server.submit_skill.call_args_list[0]
        self.assertEqual(submitted.args[0], "move_tracked_point")
        submitted_args = submitted.args[1]
        self.assertEqual(
            submitted_args["points"][0]["target_xyz_m"]["x"],
            {"var": "shared_x"},
        )
        self.assertEqual(submitted_args["points"][0]["target_xyz_m"]["y"], 0.12)
        self.assertEqual(submitted_args["execution_mode"], "exec")
        self.assertEqual(submitted_args["pos_tol"], 0.03)
        self.assertEqual(submitted_args["ori_tol_deg"], 20.0)
        self.assertEqual(submitted_args["timeout_s"], 90.0)
        self.assertEqual(
            server.submit_skill.call_args_list[1],
            mock.call("capture_head_camera", {"session_id": "web-move"}),
        )

    def test_http_plan_mode_exposes_overlay_without_exit_capture(self) -> None:
        overlay_path = "/tmp/move-tracked-plan-overlay.png"
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-plan"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "tool": "move_tracked_point",
                    "execution_mode": "plan",
                    "plan_id": "plan_0042",
                    "image_id": "img-frozen-head",
                    "render_image": "data:image/png;base64,cGxhbg==",
                    "gripper_visualization": {
                        "ok": True,
                        "path": overlay_path,
                    },
                }
            ),
        )
        app = Flask(__name__)
        runtime = SimpleNamespace(server=server)
        install_official_move_tracked_point_routes(
            app,
            runtime,
        )
        install_official_v2_failure_capture_contract(app, runtime)

        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "web-plan",
                "execution_mode": "plan",
                "points": [
                    {
                        "name": "axe_tip",
                        "target_xyz_m": [0.5, 0.1, 0.7],
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["execution_mode"], "plan")
        self.assertEqual(payload["plan_id"], "plan_0042")
        self.assertEqual(payload["image_id"], "img-frozen-head")
        self.assertEqual(payload["marked_image_url"], overlay_path)
        self.assertEqual(
            payload["plan_preview_overlay"],
            "red_gripper_on_frozen_head",
        )
        server.submit_skill.assert_called_once()
        self.assertEqual(
            server.submit_skill.call_args.args,
            (
                "move_tracked_point",
                mock.ANY,
            ),
        )
        submitted_args = server.submit_skill.call_args.args[1]
        self.assertEqual(submitted_args["execution_mode"], "plan")
        server.wait_for_skill_result.assert_called_once()

    def test_http_route_rejects_invalid_variable_before_submission(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "web-move",
                "points": [
                    {
                        "name": "axe_tip",
                        "target_xyz_m": ["x*y", 0.0, 0.7],
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["failure_stage"], "input validation")
        server.submit_skill.assert_not_called()

    def test_http_route_returns_strict_json_for_nonfinite_diagnostics(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-nonfinite"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": False,
                    "error": "final live tracked-point goal was never observed",
                    "max_anchor_error_m": float("inf"),
                    "final_anchor_error_m": {
                        "p1": np.float32(np.inf),
                        "p2": np.float64(np.nan),
                    },
                    "nested": np.asarray([0.25, -np.inf], dtype=np.float32),
                }
            ),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "strict-json-session",
                "points": [
                    {"name": "p1", "target_xyz_m": ["x", "y", "z"]}
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        raw = response.get_data(as_text=True)
        self.assertNotIn("Infinity", raw)
        self.assertNotIn("NaN", raw)
        payload = response.get_json()
        self.assertIsNone(payload["max_anchor_error_m"])
        self.assertIsNone(payload["final_anchor_error_m"]["p1"])
        self.assertIsNone(payload["final_anchor_error_m"]["p2"])
        self.assertEqual(payload["nested"], [0.25, None])
        json.dumps(payload, allow_nan=False)

    def test_official_tool_result_boundary_sanitizes_nonfinite_values(self) -> None:
        ctx = SimpleNamespace(set_result=mock.Mock())
        with mock.patch.object(
            official_tools,
            "record_compliant_grasp_prediction",
        ):
            official_tools._set_result(
                ctx,
                "move_tracked_point",
                {
                    "ok": False,
                    "python_inf": float("inf"),
                    "numpy_nan": np.float32(np.nan),
                    "array": np.asarray([1.0, -np.inf]),
                },
            )

        result = ctx.set_result.call_args.args[0]
        self.assertIsNone(result["python_inf"])
        self.assertIsNone(result["numpy_nan"])
        self.assertEqual(result["array"], [1.0, None])
        json.dumps(result, allow_nan=False)

    def test_human_ui_renders_tracked_target_rows(self) -> None:
        app = Flask(__name__)

        @app.get("/")
        def index():
            return "<html><head></head><body>test interface</body></html>"

        install_track_object_distance_human_ui(app)
        client = app.test_client()
        page_response = client.get("/")
        page = page_response.get_data(as_text=True)
        self.assertIn("move_tracked_point_human_ui.css", page)
        self.assertIn("move_tracked_point_human_ui.js", page)
        self.assertIn("move_tracked_point_human_ui.css?v17", page)
        self.assertIn("move_tracked_point_human_ui.js?v17", page)
        self.assertEqual(
            page_response.headers.get("Cache-Control"),
            "no-cache, no-store, must-revalidate",
        )

        script_response = client.get(
            "/__official__/assets/move_tracked_point_human_ui.js"
        )
        script = script_response.get_data(as_text=True)
        style = client.get(
            "/__official__/assets/move_tracked_point_human_ui.css"
        ).get_data(as_text=True)
        self.assertEqual(
            script_response.headers.get("Cache-Control"),
            "no-cache, no-store, must-revalidate",
        )
        self.assertIn('const WIDGET = "tracked_target_points"', script)
        self.assertIn("body.points = points", script)
        self.assertIn("target_xyz_m", script)
        self.assertIn("XYZ coordinates", script)
        self.assertIn("coordinateTokens", script)
        self.assertIn('coordinates: ""', script)
        self.assertIn(
            "tracked_target_rows_v17_inequalities", script
        )
        self.assertIn('id="v2-tracked-execution-mode"', script)
        self.assertIn('value="exec" ${V2.trackedExecutionMode', script)
        self.assertIn('value="plan" ${V2.trackedExecutionMode', script)
        self.assertIn("body.execution_mode = V2.trackedExecutionMode", script)
        self.assertIn('data-arg="execution_mode"', script)
        self.assertIn('value="touch"', script)
        self.assertIn('value="flatwise"', script)
        self.assertIn('value="plane_parallel"', script)
        self.assertIn('value="collinear"', script)
        self.assertIn('id="v2-tracked-constraint-block-add"', script)
        self.assertIn('id="v2-tracked-inequality-add"', script)
        self.assertIn('id="v2-tracked-inequality-list"', script)
        self.assertIn('data-v2-inequality-field="lhs"', script)
        self.assertIn('data-v2-inequality-field="op"', script)
        self.assertIn('data-v2-inequality-field="rhs"', script)
        self.assertIn("body.inequalities = V2.trackedInequalities.map", script)
        self.assertIn("V2.trackedConstraintBlocks", script)
        self.assertIn("quickType: previous.quickType", script)
        self.assertIn("rows: previous.rows.map", script)
        self.assertIn("body.quick_constraints = quickGroups", script)
        self.assertIn("constraint.axial_point_order", script)
        self.assertIn("collinear axial order must contain every row exactly once", script)
        self.assertIn('data-v2-block-field="quickType"', script)
        self.assertIn('data-v2-block-add-point="${blockIndex}"', script)
        self.assertIn('data-v2-block-index="${blockIndex}"', script)
        self.assertIn('data-v2-complete-block-point-toolbar="${blockIndex}"', script)
        self.assertIn('data-v2-complete-block-point-list="${blockIndex}"', script)
        self.assertNotIn('data-v2-quick-field="onHandNames"', script)
        self.assertNotIn('data-v2-quick-field="offHandNames"', script)
        self.assertNotIn("on-hand points (ordered)", script)
        self.assertNotIn("off-hand points (ordered)", script)
        self.assertIn("pointByName = new Map()", script)
        self.assertIn("different roles in multiple constraints", script)
        self.assertIn("different XYZ coordinates in multiple constraints", script)
        self.assertIn('data-arg="quick_constraints"', script)
        self.assertIn('value="line_vertical_to_plane"', script)
        self.assertIn('value="vertical_to_ground"', script)
        self.assertIn('value="faceto"', script)
        self.assertIn('value="reverse_faceto"', script)
        self.assertIn('data-v2-target-field="role"', script)
        self.assertIn('data-arg="relations"', script)
        self.assertNotIn("v2-tracked-relation-add", script)
        self.assertNotIn("Direction sense", script)
        self.assertIn("Number.isFinite(numeric)", script)
        self.assertIn("VARIABLE_PATTERN", script)
        self.assertIn("free: true", script)
        self.assertIn("expr: text", script)
        self.assertIn("z+0.08", script)
        self.assertIn("2*z", script)
        self.assertIn("EXPRESSION_PATTERN", script)
        self.assertIn("v2-tracked-target-row", style)
        self.assertIn("v2-tracked-constraint-block", style)
        self.assertIn("v2-tracked-inequality-block", style)
        self.assertNotIn("v2-tracked-quick-group-row", style)

    def test_http_route_accepts_affine_expression_from_each_point_row(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-expression"),
            wait_for_skill_result=mock.Mock(
                return_value={"ok": True, "tool": "move_tracked_point"}
            ),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )
        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "expression-session",
                "points": [
                    {"name": "head", "target_xyz_m": ["a", "b", "z"]},
                    {
                        "name": "tail",
                        "target_xyz_m": ["a", "b", "z+0.08"],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        submitted = server.submit_skill.call_args_list[0].args[1]
        self.assertEqual(
            submitted["points"][1]["target_xyz_m"]["z"],
            {"expr": "z+0.08"},
        )

    def test_http_route_accepts_and_canonicalizes_affine_inequality(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-inequality"),
            wait_for_skill_result=mock.Mock(
                return_value={"ok": True, "tool": "move_tracked_point"}
            ),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )
        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "inequality-session",
                "points": [
                    {
                        "name": "upper",
                        "role": "on_hand",
                        "target_xyz_m": ["x", "y", "a"],
                    },
                    {
                        "name": "lower",
                        "role": "on_hand",
                        "target_xyz_m": ["x", "y", "b"],
                    },
                ],
                "inequalities": [{"lhs": " a - b ", "op": ">", "rhs": 0}],
            },
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        submitted = server.submit_skill.call_args_list[0].args[1]
        self.assertEqual(
            submitted["inequalities"],
            [{"lhs": "a-b", "op": ">", "rhs": 0.0}],
        )

    def test_endpoint_numeric_and_variable_constraints(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        source = np.asarray(position)[None, :]
        target = source[0] + np.array([0.008, 0.0, 0.0])
        numeric_spec = [
            {
                "name": "p",
                "target_xyz_m": {
                    "x": float(target[0]),
                    "y": float(target[1]),
                    "z": float(target[2]),
                },
            }
        ]
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=numeric_spec,
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )
        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        self.assertEqual(endpoint["nearest_free_coordinate_dimension"], 0)
        self.assertEqual(endpoint["nearest_free_coordinate_weight"], 0.0)
        self.assertLessEqual(
            endpoint["constraints"]["max_constraint_error_m"], 0.012
        )
        ranked = endpoint["_ranked_endpoint_candidates"]
        self.assertEqual(len(ranked), endpoint["feasible_candidate_count"])
        self.assertEqual(
            [item["selected_candidate"]["endpoint_score_rank"] for item in ranked],
            list(range(len(ranked))),
        )
        self.assertEqual(
            [item["selected_candidate"]["score"] for item in ranked],
            sorted(item["selected_candidate"]["score"] for item in ranked),
        )

        pair = np.asarray(position)[None, :] + np.array(
            [[0.0, 0.0, -0.04], [0.0, 0.0, 0.04]]
        )
        variable_spec = [
            {
                "name": "a",
                "target_xyz_m": {
                    "x": {"var": "x"},
                    "y": {"var": "y"},
                    "z": {"var": "za"},
                },
            },
            {
                "name": "b",
                "target_xyz_m": {
                    "x": {"var": "x"},
                    "y": {"var": "y"},
                    "z": {"var": "zb"},
                },
            },
        ]
        symbolic = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=pair,
            target_points=variable_spec,
            pos_tol_m=0.004,
            ori_tol_deg=5.0,
        )
        self.assertTrue(symbolic["constraints"]["ok"], symbolic)
        self.assertEqual(
            set(symbolic["constraints"]["resolved_variables"]),
            {"x", "y", "za", "zb"},
        )

        bad_numeric = [
            {"name": "a", "target_xyz_m": {"x": 0.0, "y": 0.0, "z": 0.0}},
            {"name": "b", "target_xyz_m": {"x": 1.0, "y": 0.0, "z": 0.0}},
        ]
        best_effort = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=pair,
            target_points=bad_numeric,
            pos_tol_m=0.004,
            ori_tol_deg=5.0,
        )
        self.assertFalse(best_effort["selected_is_precise"], best_effort)
        self.assertEqual(
            best_effort["selected_candidate"]["precision_class"],
            "best_effort",
        )
        self.assertFalse(best_effort["geometry_preflight"]["ok"])
        candidate_costs = [
            candidate["selected_candidate"]["accuracy_cost_mm_plus_deg"]
            for candidate in best_effort["_ranked_endpoint_candidates"]
        ]
        self.assertAlmostEqual(
            best_effort["selected_candidate"]["accuracy_cost_mm_plus_deg"],
            min(candidate_costs),
        )
        self.assertTrue(np.all(np.isfinite(best_effort["q_final"])))
        if ARM_DOF == 8:
            self.assertAlmostEqual(
                float(best_effort["q_final"][7]), 0.0, places=12
            )

    def test_single_point_partial_targets_preserve_free_source_coordinates(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        source = np.asarray(position, dtype=np.float64).reshape(1, 3)
        requested_delta = np.asarray([0.008, -0.007, 0.006], dtype=np.float64)

        for constrained_axes in (
            (0,),
            (1,),
            (2,),
            (0, 1),
            (0, 2),
            (1, 2),
        ):
            with self.subTest(constrained_axes=constrained_axes):
                target = ["?", "?", "?"]
                expected = source[0].copy()
                for axis in constrained_axes:
                    expected[axis] += requested_delta[axis]
                    target[axis] = float(expected[axis])

                endpoint = plan_endpoint(
                    state=state,
                    arm="left",
                    q_start=world.arm_qpos_list("left"),
                    source_points_robot_base_m=source,
                    target_points=[{"name": "partial", "target_xyz_m": target}],
                    pos_tol_m=0.003,
                    ori_tol_deg=5.0,
                )
                resolved = np.asarray(
                    endpoint["resolved_points_robot_base_m"], dtype=np.float64
                )[0]
                free_axes = tuple(
                    axis for axis in range(3) if axis not in constrained_axes
                )

                self.assertTrue(endpoint["constraints"]["ok"], endpoint)
                self.assertEqual(
                    endpoint["nearest_free_coordinate_dimension"],
                    len(free_axes),
                )
                self.assertEqual(endpoint["nearest_free_coordinate_weight"], 1.0)
                np.testing.assert_allclose(
                    resolved[list(constrained_axes)],
                    expected[list(constrained_axes)],
                    atol=5.0e-4,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    resolved[list(free_axes)],
                    source[0, list(free_axes)],
                    atol=5.0e-4,
                    rtol=0.0,
                )

        numeric_x = float(source[0, 0] + requested_delta[0])
        equivalent_targets = (
            [numeric_x, "?", "?"],
            [numeric_x, "single_use_y", "single_use_z"],
        )
        equivalent_resolved = []
        for target in equivalent_targets:
            endpoint = plan_endpoint(
                state=state,
                arm="left",
                q_start=world.arm_qpos_list("left"),
                source_points_robot_base_m=source,
                target_points=[{"name": "equivalent", "target_xyz_m": target}],
                pos_tol_m=0.003,
                ori_tol_deg=5.0,
            )
            self.assertTrue(endpoint["constraints"]["ok"], endpoint)
            self.assertEqual(endpoint["nearest_free_coordinate_dimension"], 2)
            equivalent_resolved.append(
                np.asarray(endpoint["resolved_points_robot_base_m"], dtype=np.float64)[0]
            )
        np.testing.assert_allclose(
            equivalent_resolved[1],
            equivalent_resolved[0],
            atol=1.0e-6,
            rtol=0.0,
        )

    @unittest.skipUnless(ARM_DOF == 8, "captured regression uses the 8-DOF profile")
    def test_plan_0096_z_only_target_preserves_captured_free_xy(self) -> None:
        """Regression for the live endpoint that previously drifted x by 23.8 mm."""

        trunk = [
            -0.9915795922279358,
            2.5302937030792236,
            0.16864080727100372,
            -1.6276665348868846e-08,
        ]
        left = [
            0.00012513319961726665,
            3.940023361792555e-06,
            9.619167394703254e-05,
            -2.0943994522094727,
            -8.66832269821316e-05,
            -1.0471980571746826,
            8.582485861552414e-06,
            0.0,
        ]
        right = [
            -1.4858343601226807,
            0.028372380882501602,
            0.5467805862426758,
            -1.4867103099822998,
            1.0230647325515747,
            -0.9647048711776733,
            -1.462199091911316,
            0.0,
        ]
        source = np.asarray(
            [[0.5435007596198076, -0.10752732575477096, 0.38345576193875475]],
            dtype=np.float64,
        )
        target_z = 0.3770956210862252
        state = local_robot_state(
            trunk_q=trunk,
            arm_left_q=left,
            arm_right_q=right,
            gripper_left_q=[0.049998246133327484, 0.049998365342617035],
            gripper_right_q=[0.04999835044145584, 0.0027673039585351944],
        )

        endpoint = plan_endpoint(
            state=state,
            arm="right",
            q_start=right,
            source_points_robot_base_m=source,
            target_points=[{"name": "1", "target_xyz_m": ["?", "?", target_z]}],
            pos_tol_m=0.006,
            ori_tol_deg=5.0,
        )
        resolved = np.asarray(
            endpoint["resolved_points_robot_base_m"], dtype=np.float64
        )[0]

        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        np.testing.assert_allclose(resolved[:2], source[0, :2], atol=5.0e-4, rtol=0.0)
        self.assertAlmostEqual(float(resolved[2]), target_z, delta=5.0e-4)
        self.assertEqual(endpoint["nearest_free_coordinate_dimension"], 2)
        self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    def test_relation_orientation_residual_blocks_false_noop(self) -> None:
        """Public tolerance is not a license to skip a finite pose change."""

        within_public_tolerance = {
            "ok": True,
            "max_constraint_error_m": 0.0,
            "max_relation_error_rad": math.radians(1.0),
            "relations": [{"type": "align_vector", "degenerate": False}],
        }
        self.assertFalse(
            tracked_motion._constraints_numerically_satisfied(
                within_public_tolerance,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
            )
        )
        numerically_zero = dict(within_public_tolerance)
        numerically_zero["max_relation_error_rad"] = 1.0e-10
        self.assertTrue(
            tracked_motion._constraints_numerically_satisfied(
                numerically_zero,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
            )
        )

    def test_endpoint_candidate_rank_is_strictly_accuracy_first(self) -> None:
        precise_small_motion = {
            "precise": True,
            "motion_score": 1.0,
            "accuracy_cost_mm_plus_deg": 7.9,
            "position_error_mm": 3.0,
            "selection_orientation_error_deg": 4.9,
        }
        precise_large_motion = {
            **precise_small_motion,
            "motion_score": 2.0,
            "accuracy_cost_mm_plus_deg": 0.0,
            "position_error_mm": 0.0,
            "selection_orientation_error_deg": 0.0,
        }
        best_effort_more_accurate = {
            "precise": False,
            "motion_score": 100.0,
            "accuracy_cost_mm_plus_deg": 14.0,
            "position_error_mm": 10.0,
            "selection_orientation_error_deg": 4.0,
        }
        best_effort_less_accurate = {
            **best_effort_more_accurate,
            "motion_score": 0.0,
            "accuracy_cost_mm_plus_deg": 24.0,
            "position_error_mm": 4.0,
            "selection_orientation_error_deg": 20.0,
        }

        ranked = sorted(
            [
                best_effort_less_accurate,
                precise_large_motion,
                best_effort_more_accurate,
                precise_small_motion,
            ],
            key=tracked_motion._candidate_rank_key,
        )
        self.assertIs(ranked[0], precise_small_motion)
        self.assertIs(ranked[1], precise_large_motion)
        self.assertIs(ranked[2], best_effort_more_accurate)
        self.assertIs(ranked[3], best_effort_less_accurate)

    def test_broad_public_tolerance_still_finds_nearest_precise_line_pose(self) -> None:
        """A 30 mm / 20 degree request must not stop at its coarse first pose."""

        _adapter, world = self._ready_world()
        state, (eef_position, eef_quaternion) = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        source = np.asarray(eef_position, dtype=np.float64) + np.asarray(
            [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
            dtype=np.float64,
        )
        anchors = tracked_motion.rigid_anchors_from_points(
            eef_position,
            eef_quaternion,
            source,
        )
        known_q = q_start.copy()
        known_q[0] += 0.06
        fixed, _position, _quaternion = tracked_motion._point_positions(
            state,
            "left",
            known_q,
            anchors,
        )
        relation = {
            "type": "line_segment_overlap",
            "point_names": ["on_a", "on_b"],
            "segment_start_robot_base_m": fixed[0].tolist(),
            "segment_end_robot_base_m": fixed[1].tolist(),
        }
        initial_relation = tracked_motion.evaluate_relations(
            source,
            [relation],
            point_names=["on_a", "on_b"],
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        self.assertFalse(initial_relation["ok"], initial_relation)
        self.assertGreater(initial_relation["max_relation_error_m"], 0.003)
        self.assertAlmostEqual(
            initial_relation["collinear_position_tolerance_m"], 0.001
        )

        free = lambda name: {"name": name, "target_xyz_m": ["?", "?", "?"]}
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=q_start,
            source_points_robot_base_m=source,
            target_points=[free("on_a"), free("on_b")],
            relations=[relation],
            fixed_points_robot_base_m=fixed,
            fixed_target_points=[free("off_a"), free("off_b")],
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
        )

        selected = endpoint["selected_candidate"]
        self.assertTrue(endpoint["selected_is_precise"], endpoint)
        self.assertLessEqual(selected["position_error_mm"], 1.0001)
        self.assertLessEqual(selected["selection_orientation_error_deg"], 5.0001)
        precise_motion_scores = [
            float(candidate["selected_candidate"]["motion_score"])
            for candidate in endpoint["_ranked_endpoint_candidates"]
            if candidate["selected_candidate"]["precise"]
        ]
        self.assertGreater(len(precise_motion_scores), 0)
        self.assertAlmostEqual(
            float(selected["motion_score"]),
            min(precise_motion_scores),
            places=12,
        )
        self.assertLess(
            float(np.max(np.abs(np.asarray(endpoint["q_final"]) - q_start))),
            0.08,
        )
        self.assertGreater(endpoint["precise_motion_refinement_evaluations"], 0)
        if ARM_DOF == 8:
            self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    @unittest.skipUnless(ARM_DOF == 8, "captured regression uses the 8-DOF profile")
    def test_plan_0130_collinear_endpoint_is_ranked_by_true_mm_plus_deg_error(self) -> None:
        trunk = [
            -0.9915008544921875,
            2.530683755874634,
            0.1684725135564804,
            -9.7162855539068e-09,
        ]
        left = [
            2.4484933192070457e-07,
            -2.0079178320031588e-09,
            -1.0355756252522497e-09,
            -2.0943994522094727,
            -1.262469231733121e-05,
            -1.0471980571746826,
            7.257850143105316e-07,
            0.0,
        ]
        right = [
            -0.33700066804885864,
            0.17165207862854004,
            0.22898170351982117,
            -2.094358444213867,
            0.35595110058784485,
            -0.6789150834083557,
            -0.14213155210018158,
            0.0,
        ]
        source = np.asarray(
            [
                [0.47985300642960826, -0.11341770687665773, 0.43014544600447896],
                [0.3772630361317413, -0.14053398575108852, 0.43426530548386316],
            ],
            dtype=np.float64,
        )
        reference = np.asarray(
            [
                [0.5463470110163741, 0.06515409671606069, 0.08601231759949624],
                [0.5627852400704483, 0.024628308914465262, 0.08373616627322011],
            ],
            dtype=np.float64,
        )
        state = local_robot_state(
            trunk_q=trunk,
            arm_left_q=left,
            arm_right_q=right,
            gripper_left_q=[0.049999, 0.049999],
            gripper_right_q=[0.049999, 0.02736],
        )
        free = lambda name: {"name": name, "target_xyz_m": ["?", "?", "?"]}
        relation = {
            "type": "line_segment_overlap",
            "point_names": ["1", "2"],
            "segment_start_robot_base_m": reference[0].tolist(),
            "segment_end_robot_base_m": reference[1].tolist(),
        }

        endpoint = plan_endpoint(
            state=state,
            arm="right",
            q_start=right,
            source_points_robot_base_m=source,
            target_points=[free("1"), free("2")],
            relations=[relation],
            fixed_points_robot_base_m=reference,
            fixed_target_points=[free("3"), free("4")],
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
        )

        selected = endpoint["selected_candidate"]
        resolved = np.asarray(
            endpoint["resolved_points_robot_base_m"], dtype=np.float64
        )
        controlled_axis = resolved[1] - resolved[0]
        reference_axis = reference[1] - reference[0]
        measured_angle_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(
                            float(
                                controlled_axis @ reference_axis
                                / (
                                    np.linalg.norm(controlled_axis)
                                    * np.linalg.norm(reference_axis)
                                )
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        ranked = endpoint["_ranked_endpoint_candidates"]
        rank_keys = [
            tracked_motion._candidate_rank_key(item["selected_candidate"])
            for item in ranked
        ]

        self.assertEqual(
            endpoint["selection_policy"],
            "precise_collinear_1mm_other_3mm_5deg_then_minimum_motion_else_"
            "minimum_position_mm_plus_orientation_deg",
        )
        self.assertEqual(rank_keys[0], min(rank_keys))
        self.assertAlmostEqual(
            selected["selection_orientation_error_deg"],
            measured_angle_deg,
            places=7,
        )
        self.assertAlmostEqual(
            selected["accuracy_cost_mm_plus_deg"],
            selected["position_error_mm"] + measured_angle_deg,
            places=7,
        )
        self.assertLess(selected["accuracy_cost_mm_plus_deg"], 34.5)
        # This fixed-arm capture has no 1 mm collinear solution.  Planning
        # mode must still rank and return the most accurate best-effort pose;
        # execution mode will reject it at final live validation.
        self.assertFalse(endpoint["selected_meets_requested_tolerance"], endpoint)
        self.assertFalse(selected["precise"])
        self.assertAlmostEqual(
            selected["collinear_position_tolerance_m"],
            tracked_motion.COLLINEAR_POSITION_TOLERANCE_M,
        )
        self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    @unittest.skipUnless(ARM_DOF == 8, "captured regression uses the 8-DOF profile")
    def test_trunk_joint_limit_normalization_is_numeric_only(self) -> None:
        limits = np.asarray(official_tools.TRUNK_LIMITS, dtype=np.float64)
        near_limit = np.asarray(
            [0.0, limits[1, 1] + 5.0e-8, 0.0, 5.0e-8],
            dtype=np.float64,
        )
        normalized, corrections = (
            official_tools._normalize_trunk_q_to_joint_limits(
                near_limit,
                label="test planned trunk",
                point="planned",
            )
        )
        self.assertEqual(len(corrections), 2)
        self.assertAlmostEqual(float(normalized[1]), float(limits[1, 1]), places=12)
        self.assertEqual(float(normalized[3]), 0.0)
        self.assertTrue(
            all(
                abs(float(item["correction_rad"]))
                <= official_tools.TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
                for item in corrections
            )
        )

        truly_invalid = near_limit.copy()
        truly_invalid[1] = limits[1, 1] + 2.0e-6
        with self.assertRaisesRegex(ValueError, "exceeds local trunk joint limits"):
            official_tools._normalize_trunk_q_to_joint_limits(
                truly_invalid,
                label="test invalid planned trunk",
                point="planned",
            )

        recovered, observed_corrections = (
            official_tools._normalize_trunk_q_to_joint_limits(
                truly_invalid,
                label="test observed trunk",
                point="observed",
                recover_observed_overshoot=True,
            )
        )
        self.assertAlmostEqual(float(recovered[1]), float(limits[1, 1]), places=12)
        self.assertTrue(
            any(
                item.get("recovered_observed_overshoot") is True
                for item in observed_corrections
            )
        )

    @unittest.skipUnless(ARM_DOF == 8, "captured regression uses the 8-DOF profile")
    def test_endpoint_does_not_newly_enter_joint_limit_buffer(self) -> None:
        """Regression for a live underconstrained pair that selected J5 low."""

        trunk = [
            0.4525195062160492,
            -0.3908095359802246,
            -0.0060771917924284935,
            0.0,
        ]
        left = [
            -0.0011921485420316458,
            -2.8935693990206346e-05,
            0.0006849583005532622,
            -2.0944039821624756,
            -0.00015898865240160376,
            -1.0463420152664185,
            -1.7416334230802022e-05,
            0.0,
        ]
        right = [
            -1.722091555595398,
            0.17449568212032318,
            1.5770337581634521,
            -0.4041549265384674,
            -2.0551280975341797,
            0.424024760723114,
            0.28556114435195923,
            0.0,
        ]
        source = np.asarray(
            [
                [0.8303407784148646, 0.19779528884868264, 1.5434417428297962],
                [0.8518700047138762, 0.16802173177609198, 1.5310249963221942],
            ],
            dtype=np.float64,
        )
        targets = [
            {
                "name": "arbitrary_a",
                "target_xyz_m": ["shared_x", "shared_y", "?"],
            },
            {
                "name": "arbitrary_b",
                "target_xyz_m": [
                    "shared_x+0.00593561814865",
                    "shared_y-0.0356275460591",
                    "?",
                ],
            },
        ]
        state = local_robot_state(
            trunk_q=trunk,
            arm_left_q=left,
            arm_right_q=right,
        )
        endpoint = plan_endpoint(
            state=state,
            arm="right",
            q_start=right,
            source_points_robot_base_m=source,
            target_points=targets,
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        lower, upper = tracked_motion.arm_joint_limits(state, "right")
        start_q = np.asarray(right, dtype=np.float64)[:7]
        final_q = np.asarray(endpoint["q_final"], dtype=np.float64)[:7]
        span = upper[:7] - lower[:7]
        required = np.minimum(
            tracked_motion.ENDPOINT_NEW_LIMIT_SAFETY_MARGIN_RAD,
            0.2 * span,
        )
        start_margin = np.minimum(
            start_q - lower[:7], upper[:7] - start_q
        )
        final_margin = np.minimum(
            final_q - lower[:7], upper[:7] - final_q
        )
        started_interior = start_margin >= required
        self.assertTrue(np.any(started_interior))
        self.assertTrue(
            np.all(final_margin[started_interior] >= required[started_interior] - 1e-8),
            (start_margin, final_margin, endpoint),
        )
        self.assertGreaterEqual(final_margin[4], 0.08 - 1e-8)
        self.assertLessEqual(
            max(
                endpoint["selected_candidate"][
                    "joint_limit_safety_worsening_rad"
                ]
            ),
            1e-8,
        )

    def test_numeric_pair_near_length_mismatch_uses_generic_projection(self) -> None:
        """Pointwise tolerance may make a nearly rigid numeric pair feasible.

        The exact rigid-transform family cannot map a pair whose requested
        separation is slightly different from the captured rigid separation.
        Both endpoints can nevertheless be inside the public L-infinity
        tolerance.  This must use the same constraint solver as symbolic
        input, rather than a value-specific fallback or an unconditional
        rejection.
        """

        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        source = np.asarray(
            [position + [-0.04, 0.0, 0.0], position + [0.04, 0.0, 0.0]],
            dtype=np.float64,
        )
        # The requested separation differs by 0.01m, but each endpoint can
        # move 0.005m and still satisfy a 0.006m pointwise tolerance.
        target = np.asarray(
            [position + [-0.045, 0.0, 0.0], position + [0.045, 0.0, 0.0]],
            dtype=np.float64,
        )
        targets = [
            {"name": "first", "target_xyz_m": target[0].tolist()},
            {"name": "second", "target_xyz_m": target[1].tolist()},
        ]
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=targets,
            pos_tol_m=0.006,
            ori_tol_deg=5.0,
        )
        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        self.assertEqual(
            endpoint["selected_candidate"]["mode"],
            "joint_space_affine_constraint_ik",
        )
        self.assertLessEqual(
            endpoint["constraints"]["max_constraint_error_m"],
            0.006,
        )
        self.assertTrue(np.isfinite(endpoint["q_final"]).all())

    def test_numeric_and_symbolic_inputs_share_one_endpoint_solver(self) -> None:
        """The representation of a target must not select a special planner."""

        _adapter, world = self._ready_world()
        state, (position, quaternion) = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)

        single_source = position[None, :]
        single_target = position + np.asarray([0.006, -0.004, 0.003])
        single = plan_endpoint(
            state=state,
            arm="left",
            q_start=q_start,
            source_points_robot_base_m=single_source,
            target_points=[
                {
                    "name": "arbitrary_single_name",
                    "target_xyz_m": single_target.tolist(),
                }
            ],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        anchors = tracked_motion.rigid_anchors_from_points(
            position,
            quaternion,
            np.asarray(
                [position + [-0.04, 0.0, 0.0], position + [0.04, 0.0, 0.0]],
                dtype=np.float64,
            ),
        )
        pair_source = np.asarray(
            [position + [-0.04, 0.0, 0.0], position + [0.04, 0.0, 0.0]],
            dtype=np.float64,
        )
        q_target = q_start.copy()
        q_target[:7] += np.asarray([0.02, -0.015, 0.01, 0.01, -0.02, 0.015, -0.01])
        pair_target, _eef_target, _quat_target = tracked_motion._point_positions(
            state,
            "left",
            q_target,
            anchors,
        )
        pair = plan_endpoint(
            state=state,
            arm="left",
            q_start=q_start,
            source_points_robot_base_m=pair_source,
            target_points=[
                {"name": "alpha_17", "target_xyz_m": pair_target[0].tolist()},
                {"name": "beta_93", "target_xyz_m": pair_target[1].tolist()},
            ],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        for endpoint in (single, pair):
            self.assertTrue(endpoint["constraints"]["ok"], endpoint)
            self.assertEqual(
                endpoint["selected_candidate"]["mode"],
                "joint_space_affine_constraint_ik",
            )

    def test_generic_affine_solver_handles_all_supported_constraint_shapes(self) -> None:
        """Coordinate names and constants select equations, never a solver hack."""

        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        q_start = world.arm_qpos_list("left")
        cases = (
            (
                "shared_y",
                np.asarray(
                    [position + [-0.04, -0.03, -0.03], position + [0.04, 0.03, 0.03]]
                ),
                [["xa", "y", "za"], ["xb", "y", "zb"]],
            ),
            (
                "shared_yz",
                np.asarray(
                    [position + [-0.04, -0.03, -0.03], position + [0.04, 0.03, 0.03]]
                ),
                [["xa", "y", "z"], ["xb", "y", "z"]],
            ),
            (
                "x_offset",
                np.asarray(
                    [position + [-0.05, 0.0, 0.0], position + [0.05, 0.0, 0.0]]
                ),
                [["x", "y", "z"], ["x+0.07", "?", "?"]],
            ),
            (
                "fixed_head_tail_free",
                np.asarray(
                    [position + [-0.04, 0.0, 0.0], position + [0.04, 0.0, 0.0]]
                ),
                [
                    [float(position[0] + 0.04), float(position[1]), float(position[2])],
                    ["?", "?", "?"],
                ],
            ),
        )
        for case_name, source, rows in cases:
            with self.subTest(case=case_name):
                targets = [
                    {
                        "name": name,
                        "target_xyz_m": dict(zip("xyz", row)),
                    }
                    for name, row in zip(("head", "tail"), rows)
                ]
                endpoint = plan_endpoint(
                    state=state,
                    arm="left",
                    q_start=q_start,
                    source_points_robot_base_m=source,
                    # Deliberately pass the public string form directly.  The
                    # local planner must apply the same normalization as the
                    # HTTP boundary.
                    target_points=targets,
                    pos_tol_m=0.006,
                    ori_tol_deg=5.0,
                )
                self.assertTrue(endpoint["constraints"]["ok"], endpoint)
                self.assertEqual(
                    endpoint["selected_candidate"]["mode"],
                    "joint_space_affine_constraint_ik",
                )
                self.assertTrue(
                    np.isfinite(endpoint["q_final"]).all(),
                    endpoint,
                )
                if case_name == "x_offset":
                    resolved = endpoint["resolved_points_robot_base_m"]
                    self.assertAlmostEqual(
                        float(resolved[1, 0] - resolved[0, 0]),
                        0.07,
                        delta=0.006,
                    )
                if case_name == "fixed_head_tail_free":
                    resolved = endpoint["resolved_points_robot_base_m"]
                    np.testing.assert_allclose(
                        resolved[0],
                        [position[0] + 0.04, position[1], position[2]],
                        atol=0.006,
                    )

    def test_affine_offset_is_not_tied_to_a_specific_constant_or_name(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        source = np.asarray(
            [position + [-0.045, 0.0, 0.0], position + [0.045, 0.0, 0.0]],
            dtype=np.float64,
        )
        offset = 0.061
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=[
                {
                    "name": "front_reference_17",
                    "target_xyz_m": ["anchor_value", "shared_lateral", "?"],
                },
                {
                    "name": "rear_reference_93",
                    "target_xyz_m": [
                        f"anchor_value+{offset}",
                        "shared_lateral",
                        "?",
                    ],
                },
            ],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )
        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        resolved = endpoint["resolved_points_robot_base_m"]
        self.assertAlmostEqual(
            float(resolved[1, 0] - resolved[0, 0]),
            offset,
            delta=0.012,
        )
        self.assertLessEqual(
            abs(float(resolved[1, 1] - resolved[0, 1])),
            0.012,
        )

    def test_infeasible_affine_constraint_returns_ranked_best_effort(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        source = np.asarray(
            [position + [-0.02, 0.0, 0.0], position + [0.02, 0.0, 0.0]],
            dtype=np.float64,
        )
        targets = [
            {"name": "head", "target_xyz_m": ["x", "y", "z"]},
            {"name": "tail", "target_xyz_m": ["x+0.07", "?", "?"]},
        ]
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=targets,
            pos_tol_m=0.001,
            ori_tol_deg=5.0,
        )
        self.assertFalse(endpoint["selected_is_precise"], endpoint)
        self.assertFalse(endpoint["selected_meets_requested_tolerance"], endpoint)
        self.assertFalse(endpoint["constraints"]["ok"], endpoint)
        self.assertEqual(
            endpoint["selection_policy"],
            "precise_3mm_5deg_then_minimum_motion_else_"
            "minimum_position_mm_plus_orientation_deg",
        )
        costs = [
            candidate["selected_candidate"]["accuracy_cost_mm_plus_deg"]
            for candidate in endpoint["_ranked_endpoint_candidates"]
        ]
        self.assertEqual(costs, sorted(costs))
        self.assertAlmostEqual(
            endpoint["selected_candidate"]["accuracy_cost_mm_plus_deg"],
            min(costs),
        )
        self.assertGreater(endpoint["best_effort_refinement_evaluations"], 0)
        if ARM_DOF == 8:
            self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    def test_affine_relation_residual_is_not_split_between_coordinates(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.05, 0.2, -0.1]],
            dtype=np.float64,
        )
        targets = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {"name": "head", "target_xyz_m": ["x", "?", "?"]},
                    {"name": "tail", "target_xyz_m": ["x+0.07", "?", "?"]},
                ]
            },
        )["points"]
        report = evaluate_target_constraints(points, targets, tolerance_m=0.001)
        self.assertFalse(report["ok"])
        # The reported error is the maximum physical coordinate correction
        # after projecting onto the shared-x affine subspace.  The two
        # conflicting coordinates therefore split the 20 mm discrepancy into
        # two 10 mm corrections; a raw reduced-equation residual would be
        # representation-dependent.
        self.assertAlmostEqual(report["max_constraint_error_m"], 0.01, places=9)
        self.assertEqual(report["constraint_residual_dimension"], 1)

    def test_affine_coordinate_error_is_invariant_to_equivalent_expression_scale(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.05, 0.2, -0.1]],
            dtype=np.float64,
        )
        base = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {"name": "a", "target_xyz_m": ["x", "?", "?"]},
                    {"name": "b", "target_xyz_m": ["x+0.07", "?", "?"]},
                ]
            },
        )["points"]
        scaled = validate_submission(
            "move_tracked_point",
            {
                "points": [
                    {"name": "a", "target_xyz_m": ["2*x/2", "?", "?"]},
                    {
                        "name": "b",
                        "target_xyz_m": ["(2*x+0.14)/2", "?", "?"],
                    },
                ]
            },
        )["points"]
        base_report = evaluate_target_constraints(
            points, base, tolerance_m=0.001
        )
        scaled_report = evaluate_target_constraints(
            points, scaled, tolerance_m=0.001
        )
        self.assertAlmostEqual(
            base_report["max_coordinate_constraint_error_m"],
            scaled_report["max_coordinate_constraint_error_m"],
            places=12,
        )
        self.assertAlmostEqual(
            base_report["max_constraint_error_m"], 0.01, places=9
        )

    def test_strict_affine_inequality_validation_and_live_evaluation(self) -> None:
        request = {
            "points": [
                {
                    "name": "upper",
                    "role": "on_hand",
                    "target_xyz_m": ["x", "y", "a"],
                },
                {
                    "name": "lower",
                    "role": "on_hand",
                    "target_xyz_m": ["x", "y", "b"],
                },
            ],
            "inequalities": [{"lhs": "a-b", "op": ">", "rhs": 0}],
        }
        normalized = validate_submission("move_tracked_point", request)
        self.assertEqual(
            normalized["inequalities"],
            [{"lhs": "a-b", "op": ">", "rhs": 0.0}],
        )
        targets = [
            {key: value for key, value in point.items() if key != "role"}
            for point in normalized["points"]
        ]
        for points, expected, expected_sign in (
            ([[0.5, 0.0, 0.6], [0.5, 0.0, 0.4]], True, 1),
            ([[0.5, 0.0, 0.4], [0.5, 0.0, 0.6]], False, -1),
            ([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]], False, 0),
        ):
            with self.subTest(points=points):
                report = evaluate_target_constraints(
                    points,
                    targets,
                    tolerance_m=0.03,
                    inequalities=normalized["inequalities"],
                )
                self.assertEqual(report["ok"], expected, report)
                clearance = float(report["minimum_inequality_clearance_m"])
                self.assertEqual(
                    1 if clearance > 0 else (-1 if clearance < 0 else 0),
                    expected_sign,
                )
                self.assertEqual(
                    report["inequality_constraints_ok"], expected
                )

        invalid = (
            [{"lhs": "missing-a", "op": ">", "rhs": 0}],
            [{"lhs": "a*b", "op": ">", "rhs": 0}],
            [{"lhs": "a-b", "op": ">=", "rhs": 0}],
            [{"lhs": "1", "op": ">", "rhs": 0}],
            [{"lhs": "a-b", "op": ">", "rhs": float("inf")}],
        )
        for inequalities in invalid:
            bad_request = {**request, "inequalities": inequalities}
            with self.subTest(inequalities=inequalities), self.assertRaises(
                (OfficialToolBoundaryError, ValueError)
            ):
                validate_submission("move_tracked_point", bad_request)

        too_many = {
            **request,
            "inequalities": [
                {"lhs": "a-b", "op": ">", "rhs": index * 0.001}
                for index in range(7)
            ],
        }
        with self.assertRaises((OfficialToolBoundaryError, ValueError)):
            validate_submission("move_tracked_point", too_many)

    def test_omitted_and_empty_inequalities_are_identical(self) -> None:
        request = {
            "points": [
                {
                    "name": "left_finger_tip",
                    "role": "on_hand",
                    "target_xyz_m": ["x", "y", "a"],
                },
                {
                    "name": "handle",
                    "role": "off_hand",
                    "target_xyz_m": ["x", "y", "z"],
                },
                {
                    "name": "right_finger_tip",
                    "role": "on_hand",
                    "target_xyz_m": ["x", "y", "b"],
                },
            ],
            "execution_mode": "plan",
            "pos_tol": 0.03,
            "ori_tol_deg": 20.0,
            "max_steps": 360,
            "timeout_s": 90.0,
        }
        omitted = validate_submission("move_tracked_point", request)
        explicit_empty = validate_submission(
            "move_tracked_point", {**request, "inequalities": []}
        )
        self.assertEqual(omitted, explicit_empty)
        self.assertEqual(omitted["inequalities"], [])

    def test_endpoint_solver_obeys_either_strict_vertical_sign(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]],
            dtype=np.float64,
        )
        source = np.asarray(position)[None, :] + (rotation @ anchors.T).T
        targets = [
            {"name": "upper", "target_xyz_m": ["x", "y", "a"]},
            {"name": "lower", "target_xyz_m": ["x", "y", "b"]},
        ]
        for operator, expected_sign in ((">", 1), ("<", -1)):
            with self.subTest(operator=operator):
                endpoint = plan_endpoint(
                    state=state,
                    arm="left",
                    q_start=world.arm_qpos_list("left"),
                    source_points_robot_base_m=source,
                    target_points=targets,
                    inequalities=[
                        {"lhs": "a-b", "op": operator, "rhs": 0.0}
                    ],
                    pos_tol_m=0.03,
                    ori_tol_deg=20.0,
                )
                resolved = np.asarray(
                    endpoint["resolved_points_robot_base_m"],
                    dtype=np.float64,
                )
                self.assertTrue(endpoint["constraints"]["ok"], endpoint)
                self.assertTrue(
                    endpoint["constraints"]["inequality_constraints_ok"],
                    endpoint,
                )
                self.assertLess(
                    float(np.linalg.norm(resolved[0, :2] - resolved[1, :2])),
                    0.03,
                )
                self.assertEqual(
                    1 if resolved[0, 2] > resolved[1, 2] else -1,
                    expected_sign,
                )

    def test_official_action_loop_executes_and_live_verifies_inequality(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]],
            dtype=np.float64,
        )
        source = np.asarray(position)[None, :] + (rotation @ anchors.T).T
        source_dz = float(source[0, 2] - source[1, 2])
        operator = "<" if source_dz >= 0.0 else ">"
        expected_sign = -1 if operator == "<" else 1
        names = ["upper", "lower"]
        manager = _RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source)),
        )
        world._official_tracked_object_distances = manager
        ctx, result = self._ctx(world)
        initial_right = np.asarray(world.arm_qpos_list("right"), dtype=np.float64)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": root},
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "upper",
                            "role": "on_hand",
                            "target_xyz_m": ["x", "y", "a"],
                        },
                        {
                            "name": "lower",
                            "role": "on_hand",
                            "target_xyz_m": ["x", "y", "b"],
                        },
                    ],
                    inequalities=[
                        {"lhs": "a-b", "op": operator, "rhs": 0.0}
                    ],
                    pos_tol=0.03,
                    ori_tol_deg=20.0,
                    max_steps=420,
                    timeout_s=90.0,
                ),
            )
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertTrue(
            result["final_constraints"]["inequality_constraints_ok"],
            result,
        )
        self.assertEqual(
            result["target_constraints"]["inequalities"][0]["op"],
            operator,
        )
        final = result["final_live_points_robot_base_m"]
        upper = np.asarray(final["upper"], dtype=np.float64)
        lower = np.asarray(final["lower"], dtype=np.float64)
        self.assertLess(float(np.linalg.norm(upper[:2] - lower[:2])), 0.03)
        self.assertEqual(1 if upper[2] > lower[2] else -1, expected_sign)
        self.assertGreater(len(actions), 1)
        for action in actions:
            self.assertEqual(np.shape(action), (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            np.testing.assert_allclose(action[ACTION_SLICES["arm_right"]], initial_right)
            if ARM_DOF == 8:
                self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)

    def test_shared_xy_free_z_uses_generic_affine_constraint_ik(self) -> None:
        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        source = np.asarray(position)[None, :] + np.array(
            [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
        )
        targets = [
            {
                "name": "ren",
                "target_xyz_m": {
                    "x": {"var": "x"},
                    "y": {"var": "y"},
                    "z": {"var": "za"},
                },
            },
            {
                "name": "bing",
                "target_xyz_m": {
                    "x": {"var": "x"},
                    "y": {"var": "y"},
                    "z": {"var": "zb"},
                },
            },
        ]

        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=targets,
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        self.assertEqual(
            endpoint["selected_candidate"]["mode"],
            "joint_space_affine_constraint_ik",
        )
        resolved = endpoint["resolved_points_robot_base_m"]
        self.assertLessEqual(abs(float(resolved[0, 0] - resolved[1, 0])), 0.012)
        self.assertLessEqual(abs(float(resolved[0, 1] - resolved[1, 1])), 0.012)
        self.assertEqual(endpoint["j8_participates"], False)
        if ARM_DOF == 8:
            self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    def test_underdetermined_pair_selects_nearest_rigid_point_transform(self) -> None:
        """A solved equality must not drift along unconstrained coordinates."""

        _adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        source = np.asarray(
            [
                position + [-0.04, -0.03, -0.03],
                position + [0.04, 0.03, 0.03],
            ],
            dtype=np.float64,
        )
        endpoint = plan_endpoint(
            state=state,
            arm="left",
            q_start=world.arm_qpos_list("left"),
            source_points_robot_base_m=source,
            target_points=[
                {
                    "name": "arbitrary_first",
                    "target_xyz_m": ["shared_axis", "first_y", "first_z"],
                },
                {
                    "name": "arbitrary_second",
                    "target_xyz_m": ["shared_axis", "second_y", "second_z"],
                },
            ],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        resolved = np.asarray(
            endpoint["resolved_points_robot_base_m"], dtype=np.float64
        )
        source_delta = source[1] - source[0]
        closest_delta = source_delta.copy()
        closest_delta[0] = 0.0
        closest_delta *= np.linalg.norm(source_delta) / np.linalg.norm(
            closest_delta
        )
        theoretical_point_motion = 0.5 * float(
            np.linalg.norm(closest_delta - source_delta)
        )
        measured_point_motion = np.linalg.norm(resolved - source, axis=1)

        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        self.assertLessEqual(
            float(np.max(measured_point_motion)),
            theoretical_point_motion + 0.006,
            endpoint,
        )
        self.assertLessEqual(
            float(
                np.linalg.norm(
                    np.mean(resolved, axis=0) - np.mean(source, axis=0)
                )
            ),
            0.006,
            endpoint,
        )

    def test_start_gate_absorbs_transient_arm_velocity(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qvel"]][0] = 0.106076956
        adapter.update({"robot_r1::proprio": proprio})

        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda action_index: 0.106076956 if action_index == 1 else 0.0,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_start_verified")
        self.assertEqual(
            report["stable_steps"],
            official_tools.MOVE_TRACKED_POINT_START_STABLE_STEPS,
        )
        self.assertGreaterEqual(
            report["settle_steps"],
            official_tools.MOVE_TRACKED_POINT_START_STABLE_STEPS - 1,
        )
        self.assertAlmostEqual(
            report["max_observed_errors"]["right_arm_qvel_inf_rad_s"],
            0.106076956,
            places=6,
        )
        self.assertIn("measured_qpos_brake", report["hold_modes"])
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

    def test_start_gate_accepts_rgbd_xyz_refinement_when_eef_rigid(self) -> None:
        """A fresh RGB-D sample may move while its physical EEF anchor does not."""

        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [[-0.045, 0.0, 0.0], [0.045, 0.0, 0.0]], dtype=np.float64
        )
        names = ["finger-left", "finger-right"]
        source_points = position[None, :] + (rotation @ anchors.T).T
        manager = _RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source_points)),
        )
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        real_snapshot = manager.observed_active_points_snapshot
        rgbd_refinement = np.asarray([0.0094, 0.0022, 0.0131])

        def refined_snapshot(requested_names, *, episode_id):
            snapshot = real_snapshot(requested_names, episode_id=episode_id)
            if snapshot.get("ok"):
                point = np.asarray(
                    snapshot["entries"]["finger-right"][
                        "xyz_in_robot_base_coord_m"
                    ],
                    dtype=np.float64,
                )
                snapshot["entries"]["finger-right"][
                    "xyz_in_robot_base_coord_m"
                ] = (point + rgbd_refinement).tolist()
            return snapshot

        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=refined_snapshot,
        ):
            gate = official_tools._move_tracked_wait_for_stable_start(
                ctx,
                capture_state=official_tools._move_tracked_capture_state(
                    ctx,
                    {
                        side: official_tools._adjust_kinematic_observation(ctx, side)
                        for side in ("left", "right")
                    },
                ),
                manager=manager,
                point_names=names,
                source_points=source_points,
                episode_id=world.episode_id(),
                session_id=manager.session_id,
                image_id=manager.image_id,
                pos_tol_m=0.012,
                source_drift_tolerance_m=0.012,
                rigid_eef_arm="left",
                rigid_anchors_eef=anchors,
                rigid_anchor_tolerance_m=0.030,
                rigid_pair_distance_tolerance_m=0.018,
                deadline_monotonic=time.monotonic() + 30.0,
            )
            actions, report = self._drive_start_gate(
                adapter,
                gate,
                lambda _action_index: 0.0,
            )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_start_verified")
        self.assertGreater(report["last_source_drift_m"], 0.012)
        self.assertLess(report["last_eef_anchor_error_m"], 0.030)
        self.assertTrue(report["last_rigid_geometry_consistency"]["ok"])
        self.assertTrue(
            report["raw_source_drift_accepted_as_rgbd_refinement"]
        )
        self.assertEqual(
            report["source_drift_validation_mode"],
            "live_eef_rigid_anchor_and_pair_geometry",
        )
        self.assertGreaterEqual(len(actions), 2)

    def test_start_gate_rejects_post_planning_rgbd_refinement(self) -> None:
        """A signed trajectory may not absorb a later source-coordinate change."""

        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [[-0.045, 0.0, 0.0], [0.045, 0.0, 0.0]], dtype=np.float64
        )
        names = ["finger-left", "finger-right"]
        source_points = position[None, :] + (rotation @ anchors.T).T
        manager = _RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source_points)),
        )
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        real_snapshot = manager.observed_active_points_snapshot
        rigid_rgbd_refinement = np.asarray([0.0, 0.015, 0.0])

        def refined_snapshot(requested_names, *, episode_id):
            snapshot = real_snapshot(requested_names, episode_id=episode_id)
            if snapshot.get("ok"):
                for name in names:
                    point = np.asarray(
                        snapshot["entries"][name][
                            "xyz_in_robot_base_coord_m"
                        ],
                        dtype=np.float64,
                    )
                    snapshot["entries"][name][
                        "xyz_in_robot_base_coord_m"
                    ] = (point + rigid_rgbd_refinement).tolist()
            return snapshot

        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=refined_snapshot,
        ):
            gate = official_tools._move_tracked_wait_for_stable_start(
                ctx,
                capture_state=official_tools._move_tracked_capture_state(
                    ctx,
                    {
                        side: official_tools._adjust_kinematic_observation(ctx, side)
                        for side in ("left", "right")
                    },
                ),
                manager=manager,
                point_names=names,
                source_points=source_points,
                episode_id=world.episode_id(),
                session_id=manager.session_id,
                image_id=manager.image_id,
                pos_tol_m=0.012,
                source_drift_tolerance_m=0.012,
                rigid_eef_arm="left",
                rigid_anchors_eef=anchors,
                rigid_anchor_tolerance_m=0.030,
                rigid_pair_distance_tolerance_m=0.018,
                accept_rigid_rgbd_refinement=False,
                deadline_monotonic=time.monotonic() + 30.0,
            )
            _actions, report = self._drive_start_gate(
                adapter,
                gate,
                lambda _action_index: 0.0,
            )

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "tracked_points_moved_after_source_sync"
        )
        self.assertFalse(report["rigid_rgbd_refinement_allowed"])
        self.assertGreater(report["last_source_drift_m"], 0.012)
        np.testing.assert_allclose(
            report["last_live_points_robot_base_m"],
            source_points + rigid_rgbd_refinement,
            atol=1e-7,
        )

    def test_eef_prior_refinement_is_frozen_before_planner_runs(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        names = ["finger-left", "finger-right"]
        anchors = np.asarray(
            [[-0.040, 0.0, 0.0], [0.040, 0.0, 0.0]], dtype=np.float64
        )
        initial_points = position[None, :] + (rotation @ anchors.T).T
        refinement_eef = np.asarray([0.0, 0.015, 0.0], dtype=np.float64)

        class RefiningPriorManager(_RigidTrackedManager):
            def __init__(self, *manager_args, **manager_kwargs):
                super().__init__(*manager_args, **manager_kwargs)
                self.prior_activation_count = 0
                self.active_prior_lease = None

            def activate_on_hand_eef_observation_prior(
                self,
                requested_names,
                *,
                episode_id,
                session_id,
                image_id,
                arm,
                anchors_eef_m,
            ):
                self.prior_activation_count += 1
                if self.prior_activation_count == 1:
                    for name in requested_names:
                        self.anchors[name] = (
                            np.asarray(self.anchors[name], dtype=np.float64)
                            + refinement_eef
                        )
                self.active_prior_lease = (
                    f"unit-eef-prior-{self.prior_activation_count}"
                )
                return {
                    "ok": True,
                    "lease_id": self.active_prior_lease,
                    "names": list(requested_names),
                    "episode_id": str(episode_id),
                    "session_id": str(session_id),
                    "image_id": str(image_id),
                    "arm": str(arm),
                    "anchors_eef_m": np.asarray(
                        anchors_eef_m, dtype=np.float64
                    ).tolist(),
                    "activation_observation_sequence": int(
                        self.adapter.status()["sequence"]
                    ),
                    "max_point_error_m": 0.030,
                }

            def deactivate_on_hand_eef_observation_prior(self, lease_id):
                released = str(lease_id) == str(self.active_prior_lease)
                self.active_prior_lease = None
                return {
                    "ok": released,
                    "released": released,
                    "lease_id": str(lease_id),
                }

        manager = RefiningPriorManager(
            world,
            adapter,
            dict(zip(names, initial_points)),
        )
        world._official_tracked_object_distances = manager
        captured_planner_sources = []

        def capture_then_stop_planner(**kwargs):
            captured_planner_sources.append(
                np.asarray(kwargs["source_points"], dtype=np.float64).copy()
            )
            raise RuntimeError("stop after synchronized planner input capture")

        ctx, result = self._ctx(world)
        with mock.patch.object(
            official_tools,
            "_move_tracked_plan_frozen",
            side_effect=capture_then_stop_planner,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": names[0],
                            "target_xyz_m": ["x1", "y1", "z1"],
                        },
                        {
                            "name": names[1],
                            "target_xyz_m": ["x2", "y2", "z2"],
                        },
                    ],
                    timeout_s=30.0,
                ),
            )

        self.assertFalse(result["ok"], result)
        self.assertIn("stop after synchronized", result["error"])
        self.assertEqual(len(captured_planner_sources), 1)
        expected_points = position[None, :] + (
            rotation @ (anchors + refinement_eef[None, :]).T
        ).T
        np.testing.assert_allclose(
            captured_planner_sources[0], expected_points, atol=1e-7
        )
        self.assertGreaterEqual(len(actions), 2)
        sync = result["preplanning_source_sync"]
        self.assertTrue(sync["planning_source_rebased"], sync)
        self.assertGreater(sync["last_source_drift_m"], 0.012)
        self.assertGreater(
            sync["planning_observation_sequence"],
            sync["initial_observation_sequence"],
        )

    def test_post_plan_coherent_rgbd_refinement_discards_and_replans(self) -> None:
        """A second coherent refinement replaces, rather than executes, a stale plan."""

        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        names = ["finger-left", "finger-right"]
        anchors = np.asarray(
            [[-0.040, 0.0, 0.0], [0.040, 0.0, 0.0]], dtype=np.float64
        )
        initial_points = position[None, :] + (rotation @ anchors.T).T
        late_refinement_eef = np.asarray(
            [0.0, 0.015, 0.0], dtype=np.float64
        )

        class LateRefiningPriorManager(_RigidTrackedManager):
            def __init__(self, *manager_args, **manager_kwargs):
                super().__init__(*manager_args, **manager_kwargs)
                self.prior_activation_count = 0
                self.active_prior_lease = None

            def activate_on_hand_eef_observation_prior(
                self,
                requested_names,
                *,
                episode_id,
                session_id,
                image_id,
                arm,
                anchors_eef_m,
            ):
                self.prior_activation_count += 1
                # Activation 1 happens before the first planner. Activation 2
                # publishes that plan's prior and emulates the real tracker
                # settling to a newer coherent RGB-D pair while the arm is
                # still stationary.
                if self.prior_activation_count == 2:
                    for name in requested_names:
                        self.anchors[name] = (
                            np.asarray(self.anchors[name], dtype=np.float64)
                            + late_refinement_eef
                        )
                self.active_prior_lease = (
                    f"unit-late-eef-prior-{self.prior_activation_count}"
                )
                return {
                    "ok": True,
                    "lease_id": self.active_prior_lease,
                    "names": list(requested_names),
                    "episode_id": str(episode_id),
                    "session_id": str(session_id),
                    "image_id": str(image_id),
                    "arm": str(arm),
                    "anchors_eef_m": np.asarray(
                        anchors_eef_m, dtype=np.float64
                    ).tolist(),
                    "activation_observation_sequence": int(
                        self.adapter.status()["sequence"]
                    ),
                    "max_point_error_m": 0.030,
                }

            def deactivate_on_hand_eef_observation_prior(self, lease_id):
                released = str(lease_id) == str(self.active_prior_lease)
                self.active_prior_lease = None
                return {
                    "ok": released,
                    "released": released,
                    "lease_id": str(lease_id),
                }

        manager = LateRefiningPriorManager(
            world,
            adapter,
            dict(zip(names, initial_points)),
        )
        world._official_tracked_object_distances = manager
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": names[0],
                            "target_xyz_m": ["x1", "y1", "z1"],
                        },
                        {
                            "name": names[1],
                            "target_xyz_m": ["x2", "y2", "z2"],
                        },
                    ],
                    pos_tol=0.012,
                    ori_tol_deg=5.0,
                    max_steps=120,
                    timeout_s=30.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["post_planning_source_replan_count"], 1)
        self.assertGreaterEqual(manager.prior_activation_count, 4)
        replan = result["post_planning_source_replans"][0]
        self.assertTrue(replan["ok"], replan)
        self.assertFalse(replan["discarded_plan_executed"])
        self.assertNotEqual(
            replan["discarded_plan_id"], replan["new_plan_id"]
        )
        self.assertNotEqual(
            replan["discarded_trajectory_digest"],
            replan["new_trajectory_digest"],
        )
        self.assertTrue(replan["new_plan_schema_validated"])
        self.assertTrue(replan["new_plan_j8_locked_zero"])
        self.assertGreater(replan["trigger_source_drift_m"], 0.012)
        expected_points = position[None, :] + (
            rotation @ (anchors + late_refinement_eef[None, :]).T
        ).T
        np.testing.assert_allclose(
            [
                result["source_points_robot_base_m"][name]
                for name in names
            ],
            expected_points,
            atol=1e-7,
        )
        self.assertTrue(actions)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            if ARM_DOF == 8:
                for side in ("left", "right"):
                    self.assertEqual(
                        float(action[ACTION_SLICES[f"arm_{side}"]][7]),
                        0.0,
                    )

    def test_start_gate_rejects_point_outside_live_eef_anchor(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [[-0.045, 0.0, 0.0], [0.045, 0.0, 0.0]], dtype=np.float64
        )
        names = ["finger-left", "finger-right"]
        source_points = position[None, :] + (rotation @ anchors.T).T
        manager = _RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source_points)),
        )
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        real_snapshot = manager.observed_active_points_snapshot

        def detached_snapshot(requested_names, *, episode_id):
            snapshot = real_snapshot(requested_names, episode_id=episode_id)
            if snapshot.get("ok"):
                for name in names:
                    point = np.asarray(
                        snapshot["entries"][name]["xyz_in_robot_base_coord_m"],
                        dtype=np.float64,
                    )
                    snapshot["entries"][name]["xyz_in_robot_base_coord_m"] = (
                        point + np.asarray([0.0, 0.0, 0.035])
                    ).tolist()
            return snapshot

        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            {
                side: official_tools._adjust_kinematic_observation(ctx, side)
                for side in ("left", "right")
            },
        )
        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=detached_snapshot,
        ):
            gate = official_tools._move_tracked_wait_for_stable_start(
                ctx,
                capture_state=capture_state,
                manager=manager,
                point_names=names,
                source_points=source_points,
                episode_id=world.episode_id(),
                session_id=manager.session_id,
                image_id=manager.image_id,
                pos_tol_m=0.012,
                source_drift_tolerance_m=0.012,
                rigid_eef_arm="left",
                rigid_anchors_eef=anchors,
                rigid_anchor_tolerance_m=0.030,
                rigid_pair_distance_tolerance_m=0.018,
                deadline_monotonic=time.monotonic() + 30.0,
            )
            actions, report = self._drive_start_gate(
                adapter,
                gate,
                lambda _action_index: 0.0,
            )

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "tracked_points_not_rigid_with_selected_eef"
        )
        self.assertGreater(report["last_eef_anchor_error_m"], 0.030)
        self.assertEqual(actions, [])

    def test_start_gate_without_eef_prior_keeps_absolute_xyz_gate(self) -> None:
        (
            _adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        real_snapshot = manager.observed_active_points_snapshot

        def moved_snapshot(requested_names, *, episode_id):
            snapshot = real_snapshot(requested_names, episode_id=episode_id)
            if snapshot.get("ok"):
                point = np.asarray(
                    snapshot["entries"]["known"]["xyz_in_robot_base_coord_m"],
                    dtype=np.float64,
                )
                snapshot["entries"]["known"]["xyz_in_robot_base_coord_m"] = (
                    point + np.asarray([0.0162, 0.0, 0.0])
                ).tolist()
            return snapshot

        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=moved_snapshot,
        ):
            gate = official_tools._move_tracked_wait_for_stable_start(
                ctx,
                capture_state=capture_state,
                manager=manager,
                point_names=["known"],
                source_points=source_points,
                episode_id=world.episode_id(),
                session_id=manager.session_id,
                image_id=manager.image_id,
                pos_tol_m=0.012,
                source_drift_tolerance_m=0.012,
                deadline_monotonic=time.monotonic() + 30.0,
            )
            with self.assertRaises(StopIteration) as stopped:
                next(gate)

        report = stopped.exception.value
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "tracked_points_moved_during_planning")
        self.assertEqual(
            report["source_drift_validation_mode"], "absolute_frozen_xyz"
        )
        self.assertFalse(
            report["raw_source_drift_accepted_as_rgbd_refinement"]
        )

    def test_start_gate_brakes_hard_velocity_until_stationary(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        observed_hard_velocity = 2.669563
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qvel"]][0] = (
            observed_hard_velocity
        )
        adapter.update({"robot_r1::proprio": proprio})

        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        velocity_after_action = (1.2, 0.10, 0.0, 0.0, 0.0)
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda action_index: velocity_after_action[
                min(action_index - 1, len(velocity_after_action) - 1)
            ],
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_start_verified")
        self.assertGreater(
            report["max_observed_errors"]["right_arm_qvel_inf_rad_s"],
            official_tools.MOVE_TRACKED_POINT_START_HANDOFF_QVEL_HARD_RAD_S,
        )
        self.assertGreaterEqual(len(actions), 3)
        self.assertIn("measured_qpos_brake", report["hold_modes"])
        first_target = actions[0][ACTION_SLICES["arm_right"]]
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]], first_target, atol=1e-8
            )

    def test_full_tool_absorbs_post_plan_active_arm_velocity_sample(self) -> None:
        adapter, world = self._ready_world()
        state, _left_pose = self._left_eef(world)
        right_position, _right_quaternion = eef_pose(
            state,
            "right",
            world.arm_qpos_list("right"),
        )
        manager = _RigidTrackedManager(
            world,
            adapter,
            {"known": right_position},
        )
        world._official_tracked_object_distances = manager
        ctx, result = self._ctx(world)
        real_loader = official_tools._move_tracked_load_plan

        def load_then_inject_velocity(session_id, plan_id):
            loaded = real_loader(session_id, plan_id)
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_right_qvel"]][0] = 0.106076956
            adapter.update({"robot_r1::proprio": proprio})
            return loaded

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_move_tracked_load_plan",
            side_effect=load_then_inject_velocity,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "known",
                            "target_xyz_m": np.asarray(
                                right_position,
                                dtype=np.float64,
                            ).tolist(),
                        }
                    ],
                    max_steps=120,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["arm"], "right")
        start = result["start_state_validation"]
        self.assertEqual(start["reason"], "stable_start_verified")
        self.assertEqual(
            start["stable_steps"],
            official_tools.MOVE_TRACKED_POINT_START_STABLE_STEPS,
        )
        self.assertGreaterEqual(start["settle_steps"], 2)
        self.assertAlmostEqual(
            start["max_observed_errors"]["right_arm_qvel_inf_rad_s"],
            0.106076956,
            places=6,
        )
        self.assertGreaterEqual(len(actions), start["settle_steps"])

    def test_start_gate_fails_closed_when_velocity_never_settles(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qvel"]][0] = (
            official_tools.MOVE_TRACKED_POINT_START_HANDOFF_QVEL_HARD_RAD_S
            + 0.1
        )
        adapter.update({"robot_r1::proprio": proprio})

        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda _action_index: (
                official_tools.MOVE_TRACKED_POINT_START_HANDOFF_QVEL_HARD_RAD_S
                + 0.1
            ),
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "start_state_not_stable")
        self.assertEqual(
            report["settle_steps"],
            official_tools.MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS,
        )
        self.assertEqual(len(actions), report["settle_steps"])
        self.assertIn("measured_qpos_brake", report["hold_modes"])

    def test_start_gate_accepts_stationary_positions_over_bounded_qvel_noise(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        bounded_qvel = 1.20
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qvel"]][4] = bounded_qvel
        proprio[PROPRIO_SLICES["arm_left_qvel"]][5] = -0.48
        adapter.update({"robot_r1::proprio": proprio})

        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda _action_index: bounded_qvel,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_start_verified")
        self.assertEqual(report["handoff_mode"], "position_derived_velocity")
        self.assertTrue(report["velocity_threshold_satisfied"])
        self.assertFalse(
            report["handoff_assessment"][
                "raw_velocity_threshold_satisfied"
            ]
        )
        self.assertTrue(
            report["handoff_assessment"][
                "derived_velocity_threshold_satisfied"
            ]
        )
        self.assertEqual(
            report["handoff_velocity_evidence"],
            "consecutive_position_delta_over_observation_dt",
        )
        self.assertAlmostEqual(
            report["last_handoff_derived_velocity_rad_s"], 0.0
        )
        self.assertIn("measured_qpos_brake", report["hold_modes"])
        self.assertEqual(len(actions), report["settle_steps"])

    def test_start_gate_rejects_real_motion_even_when_raw_qvel_is_zero(self) -> None:
        """Consecutive qpos observations remain authoritative for real motion."""
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        q_start = np.asarray(world.arm_qpos_list("right"), dtype=np.float64)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qpos"]][0] = q_start[0] + 0.019
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )

        actions = []
        while True:
            try:
                action = np.asarray(next(gate), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            next_proprio = adapter.proprio_vector().copy()
            next_proprio[PROPRIO_SLICES["arm_right_qpos"]] = q_start
            next_proprio[PROPRIO_SLICES["arm_right_qpos"]][0] = (
                q_start[0] + (0.019 if len(actions) % 2 == 0 else -0.019)
            )
            next_proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "start_state_not_stable")
        self.assertEqual(report["handoff_mode"], "braking")
        self.assertTrue(
            report["handoff_assessment"][
                "raw_velocity_threshold_satisfied"
            ]
        )
        self.assertFalse(
            report["handoff_assessment"]["position_step_satisfied"]
        )
        self.assertGreater(
            report["last_handoff_position_step_rad"],
            official_tools.MOVE_TRACKED_POINT_START_HANDOFF_POSITION_STEP_RAD,
        )
        self.assertEqual(
            report["settle_steps"],
            official_tools.MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS,
        )
        self.assertEqual(len(actions), report["settle_steps"])

    def test_start_gate_exception_is_structured_by_tool(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        ctx, result = self._ctx(world)
        real_gate = official_tools._move_tracked_wait_for_stable_start

        def raising_gate(*args, **kwargs):
            raise ValueError("synthetic malformed evaluator sample")

        target = (np.asarray(position) + [0.008, 0.0, 0.0]).tolist()
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_move_tracked_wait_for_stable_start",
            side_effect=raising_gate,
        ):
            actions = list(
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target}],
                    max_steps=120,
                )
            )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failure_stage"], "start-state validation")
        self.assertIn("synthetic malformed evaluator sample", result["error"])
        self.assertEqual(
            result["start_state_validation"]["exception_type"],
            "ValueError",
        )
        # Local planning still emits official hold actions while the patched
        # gate is reached; the important contract is the structured terminal
        # result and its final safe hold.
        self.assertGreaterEqual(len(actions), 1)

    def test_locked_j8_velocity_is_diagnostic_not_stationary_arm_velocity(self) -> None:
        (
            adapter,
            world,
            _manager,
            ctx,
            capture_state,
            _source_points,
        ) = self._start_gate_fixture()
        if ARM_DOF != 8:
            self.skipTest("J8 exists only in the custom 8DOF contract")

        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qvel"]][7] = 4.5
        proprio[PROPRIO_SLICES["arm_right_qvel"]][7] = -4.25
        adapter.update({"robot_r1::proprio": proprio})
        start_ok, validation = official_tools._move_point_start_state_validation(
            ctx,
            capture_state,
        )

        self.assertTrue(start_ok, validation)
        self.assertAlmostEqual(
            validation["errors"]["left_arm_qvel_inf_rad_s"],
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            validation["errors"]["right_arm_qvel_inf_rad_s"],
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            validation["errors"]["left_j8_qvel_abs_rad_s"],
            4.5,
            places=7,
        )
        assessment = official_tools._move_tracked_start_handoff_assessment(
            validation
        )
        self.assertTrue(assessment["ok"], assessment)
        self.assertEqual(assessment["mode"], "stationary_threshold")

    def test_locked_j8_position_still_fails_closed(self) -> None:
        if ARM_DOF != 8:
            self.skipTest("J8 exists only in the custom 8DOF contract")
        adapter, world, _manager, ctx, capture_state, _source_points = (
            self._start_gate_fixture()
        )
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]][7] = 0.04
        adapter.update({"robot_r1::proprio": proprio})
        start_ok, validation = official_tools._move_point_start_state_validation(
            ctx,
            capture_state,
        )
        self.assertFalse(start_ok)
        self.assertEqual(validation["reason"], "left_j8_not_locked")

    def test_move_tracked_resets_any_prior_j8_pin_to_zero(self) -> None:
        if ARM_DOF != 8:
            self.skipTest("J8 exists only in the custom 8DOF contract")
        _adapter, world = self._ready_world()
        world.set_tool_roll_pin_qpos("left", 0.7)
        world.set_tool_roll_pin_qpos("right", -0.4)
        world.begin_tool_roll_motion("left")
        world.begin_tool_roll_motion("right")
        ctx, _result = self._ctx(world)
        official_tools._move_tracked_prepare_j8_lock(ctx)
        self.assertEqual(float(world.tool_roll_pin_qpos("left")), 0.0)
        self.assertEqual(float(world.tool_roll_pin_qpos("right")), 0.0)
        self.assertNotIn("left", world._tool_roll_motion_enabled)
        self.assertNotIn("right", world._tool_roll_motion_enabled)

    def test_execution_bridge_rebases_large_handoff_without_rewriting_signed_plan(self) -> None:
        _adapter, world = self._ready_world()
        initial = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        signed_target = initial.copy()
        signed_target[0] += 0.31
        signed_target[4] -= 0.24
        if ARM_DOF == 8:
            signed_target[7] = 0.0
        execution, bridge_count = official_tools._move_tracked_execution_bridge(
            initial,
            [signed_target],
            arm="left",
        )
        self.assertGreater(bridge_count, 0)
        self.assertGreater(len(execution), 1)
        self.assertEqual(len(execution[-1]), ARM_DOF)
        np.testing.assert_allclose(execution[-1], signed_target, atol=1e-12)
        previous = initial
        for waypoint in execution:
            current = np.asarray(waypoint, dtype=np.float64)
            self.assertLessEqual(
                float(np.linalg.norm(current - previous, ord=np.inf)),
                official_tools.MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD + 1e-12,
            )
            if ARM_DOF == 8:
                self.assertEqual(float(current[7]), 0.0)
            previous = current

    def test_execution_bridge_does_not_expand_scheduler_roundoff(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        initial = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            initial[7] = 0.0
        scheduler_cap = 0.03564744450012222
        accepted_delta = scheduler_cap + 9.96e-10
        waypoints = []
        for index in range(31):
            target = initial.copy()
            if index % 2 == 0:
                target[0] += accepted_delta
            waypoints.append(target)

        execution, bridge_count = official_tools._move_tracked_execution_bridge(
            initial,
            waypoints,
            arm="right",
            max_step_rad=scheduler_cap,
        )

        self.assertEqual(bridge_count, 0)
        self.assertEqual(len(execution), len(waypoints))
        np.testing.assert_allclose(execution[-1], waypoints[-1], atol=1e-12)

    def test_execution_schedule_preflights_bridge_inside_same_budget(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observed = official_tools._adjust_kinematic_observation(ctx, "right")
        initial = np.asarray(observed["q"], dtype=np.float64)
        scheduler_cap = 0.03564744450012222
        accepted_delta = scheduler_cap + 9.96e-10
        waypoints = []
        for index in range(31):
            target = initial.copy()
            if index % 2 == 0:
                target[0] += accepted_delta
            waypoints.append(target.astype(float).tolist())
        scheduler_report = {
            "ok": True,
            "mode": "adaptive_monotone_subsample",
            "reason": "synthetic scheduler rounding regression",
            "waypoints": waypoints,
            "selected_indices": list(range(len(waypoints))),
            "input_waypoint_count": len(waypoints),
            "selected_waypoint_count": len(waypoints),
            "dropped_waypoint_count": 0,
            "budget_steps": len(waypoints),
            "time_budget_steps": len(waypoints),
            "max_joint_step_rad": scheduler_cap,
            "actual_max_joint_step_rad": accepted_delta,
        }
        trajectory = {
            "tracking": {
                "max_joint_step_rad": official_tools.MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
            },
            "waypoints": waypoints,
        }

        with mock.patch.object(
            official_tools,
            "_schedule_tracked_execution_waypoints",
            return_value=scheduler_report,
        ):
            report = official_tools._move_tracked_prepare_execution_schedule(
                trajectory=trajectory,
                state=observed["state"],
                arm="right",
                start_q=initial,
                max_steps=len(waypoints),
                available_time_s=90.0,
                observation_dt_s=1.0,
            )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["execution_budget_includes_bridge"])
        self.assertEqual(report["predicted_execution_bridge_waypoint_count"], 0)
        self.assertEqual(report["predicted_execution_waypoint_count"], 31)
        self.assertLessEqual(
            report["predicted_execution_waypoint_count"],
            report["execution_budget_steps_total"],
        )
        self.assertGreaterEqual(
            report["execution_max_joint_step_rad"],
            report["actual_max_joint_step_rad"],
        )

    def test_move_tracked_schedule_targets_eight_certified_motion_actions(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observed = official_tools._adjust_kinematic_observation(ctx, "right")
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        initial = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            initial[7] = 0.0
        progress = np.linspace(0.0, 1.0, 83)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        path = np.repeat(initial[None, :], len(progress), axis=0)
        path[:, 0] += 0.30 * smooth
        path[:, 1] += 0.06 * np.sin(np.pi * progress)
        path[:, 2] -= 0.12 * progress
        trajectory = {
            "tracking": {
                "max_joint_step_rad": (
                    official_tools.MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
                ),
            },
            "planner": {
                "same_branch_checked": True,
                "monotonic_progress_checked": True,
                "internal_stop_count": 0,
            },
            "waypoints": path.astype(float).tolist(),
        }

        report = official_tools._move_tracked_prepare_execution_schedule(
            trajectory=trajectory,
            state=observed["state"],
            arm="right",
            start_q=initial,
            max_steps=360,
            available_time_s=90.0,
            observation_dt_s=1.0,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["certified_sparse_execution"], report)
        self.assertTrue(report["preferred_budget_met"], report)
        self.assertLessEqual(
            report["selected_waypoint_count"],
            official_tools.MOVE_TRACKED_POINT_EXECUTION_PREFERRED_MAX_STEPS,
        )
        self.assertEqual(report["selected_indices"][-1], len(path) - 1)
        self.assertLessEqual(
            report["joint_corridor"]["max_deviation_rad"],
            official_tools.MOVE_TRACKED_POINT_EXECUTION_MAX_JOINT_CORRIDOR_DEVIATION_RAD
            + 1.0e-9,
        )
        if ARM_DOF == 8:
            self.assertTrue(
                all(abs(row[7]) <= 1.0e-12 for row in report["waypoints"])
            )

    def test_npoint_visual_schedule_caps_step_without_changing_two_point_path(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observed = official_tools._adjust_kinematic_observation(ctx, "right")
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        initial = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            initial[7] = 0.0
        progress = np.linspace(0.0, 1.0, 181)
        path = np.repeat(initial[None, :], len(progress), axis=0)
        path[:, 0] += 0.54 * progress
        path[:, 1] += 0.09 * np.sin(np.pi * progress)
        trajectory = {
            "tracking": {
                "max_joint_step_rad": (
                    official_tools.MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
                ),
            },
            "planner": {
                "same_branch_checked": True,
                "monotonic_progress_checked": True,
                "internal_stop_count": 0,
            },
            "waypoints": path.astype(float).tolist(),
        }
        kwargs = {
            "trajectory": trajectory,
            "state": observed["state"],
            "arm": "right",
            "start_q": initial,
            "max_steps": 360,
            "available_time_s": 90.0,
            "observation_dt_s": None,
        }

        legacy = official_tools._move_tracked_prepare_execution_schedule(
            **kwargs,
        )
        two_point = official_tools._move_tracked_prepare_execution_schedule(
            **kwargs,
            visual_tracked_point_count=2,
        )
        npoint = official_tools._move_tracked_prepare_execution_schedule(
            **kwargs,
            visual_tracked_point_count=3,
        )

        self.assertTrue(legacy["ok"], legacy)
        self.assertTrue(two_point["ok"], two_point)
        self.assertTrue(npoint["ok"], npoint)
        self.assertEqual(two_point["selected_indices"], legacy["selected_indices"])
        np.testing.assert_allclose(
            two_point["waypoints"], legacy["waypoints"], atol=0.0
        )
        self.assertEqual(
            npoint["visual_speedup_cap"],
            official_tools.MOVE_TRACKED_POINT_NPOINT_VISUAL_MAX_SPEEDUP_FACTOR,
        )
        self.assertLessEqual(
            npoint["actual_max_joint_step_rad"],
            official_tools.MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
            * official_tools.MOVE_TRACKED_POINT_NPOINT_VISUAL_MAX_SPEEDUP_FACTOR
            + 1.0e-8,
        )
        self.assertGreaterEqual(
            npoint["selected_waypoint_count"],
            two_point["selected_waypoint_count"],
        )
        self.assertEqual(npoint["selected_indices"][-1], len(path) - 1)
        if ARM_DOF == 8:
            self.assertTrue(
                all(abs(row[7]) <= 1.0e-12 for row in npoint["waypoints"])
            )

    def test_execution_cadence_uses_slowest_request_local_window(self) -> None:
        report = official_tools._move_tracked_execution_cadence_estimate(
            start_validation={
                "last_handoff_observation_dt_s": 0.7202258110046387,
                "handoff_observation_dt_samples_s": [0.70, 0.72],
            },
            preplanning_source_sync={
                "last_handoff_observation_dt_s": 0.7908244132995605,
                "handoff_observation_dt_samples_s": [0.76, 0.79],
            },
            planner={
                "planning_elapsed_s": 7.750208880053833,
                "planning_hold_steps": 9,
            },
        )

        planner_period = 7.750208880053833 / 9.0
        self.assertTrue(report["ok"], report)
        self.assertAlmostEqual(
            report["max_observed_observation_dt_s"], planner_period
        )
        self.assertAlmostEqual(
            report["observation_dt_s"],
            planner_period
            * official_tools.MOVE_TRACKED_POINT_EXECUTION_CADENCE_SAFETY_FACTOR,
        )
        self.assertGreater(report["observation_dt_s"], 1.0)

    def test_execution_bridge_recovers_small_observed_joint_limit_residual(self) -> None:
        limits = official_tools._ARM_LIMITS["left"][:ARM_DOF]
        max_step = official_tools.MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD
        initial = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            initial[7] = 0.0
        initial[0] = limits[0, 0] - (0.5 * max_step)
        signed_target = initial.copy()
        signed_target[0] = limits[0, 0] + max_step

        execution, bridge_count = official_tools._move_tracked_execution_bridge(
            initial,
            [signed_target],
            arm="left",
            max_step_rad=max_step,
        )

        self.assertGreaterEqual(bridge_count, 1)
        self.assertEqual(float(execution[0][0]), float(limits[0, 0]))
        np.testing.assert_allclose(execution[-1], signed_target, atol=1e-12)
        previous = initial
        for waypoint in execution:
            current = np.asarray(waypoint, dtype=np.float64)
            self.assertTrue(np.all(current >= limits[:, 0]))
            self.assertTrue(np.all(current <= limits[:, 1]))
            self.assertLessEqual(
                float(np.linalg.norm(current - previous, ord=np.inf)),
                max_step + 1e-12,
            )
            if ARM_DOF == 8:
                self.assertEqual(float(current[7]), 0.0)
            previous = current

    def test_execution_bridge_rejects_large_observed_joint_limit_residual(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        max_step = official_tools.MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD
        initial = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            initial[7] = 0.0
        initial[0] = limits[0, 0] - (2.0 * max_step)
        signed_target = np.clip(initial, limits[:, 0], limits[:, 1])

        with self.assertRaisesRegex(
            ValueError,
            "exceeds the recoverable local joint-limit residual",
        ):
            official_tools._move_tracked_execution_bridge(
                initial,
                [signed_target],
                arm="right",
                max_step_rad=max_step,
            )

    def test_frozen_planning_starts_inside_limits_after_observed_residual(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        observed = np.mean(limits, axis=1)
        if ARM_DOF == 8:
            observed[7] = 0.0
        observed[4] = limits[4, 0] - 0.0005
        expected_start = np.clip(observed, limits[:, 0], limits[:, 1])

        endpoint = {
            "q_start": expected_start.copy(),
            "q_final": expected_start.copy(),
            "start_eef_position_m": np.zeros(3, dtype=np.float64),
            "final_eef_position_m": np.zeros(3, dtype=np.float64),
            "selected_candidate": {"endpoint_score_rank": 0},
        }
        path_plan = {
            "waypoints": [expected_start.astype(float).tolist()],
            "joint_limits_checked": True,
        }
        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={"_ranked_endpoint_candidates": [endpoint]},
        ) as endpoint_solver, mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            return_value=path_plan,
        ) as path_solver:
            result = official_tools._move_tracked_plan_frozen(
                state=object(),
                arm="right",
                q_start=observed,
                source_points=np.asarray([[0.5, 0.0, 0.8]], dtype=np.float64),
                target_points=[
                    {
                        "name": "point",
                        "target_xyz_m": {"x": 0.5, "y": 0.0, "z": 0.81},
                    }
                ],
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_steps=360,
            )

        self.assertTrue(result["ok"])
        np.testing.assert_allclose(
            endpoint_solver.call_args.kwargs["q_start"],
            expected_start,
            atol=0.0,
        )
        np.testing.assert_allclose(
            path_solver.call_args.kwargs["q_start"],
            expected_start,
            atol=0.0,
        )
        recovery = result["path_plan"][
            "observed_start_joint_limit_recovery"
        ]
        self.assertEqual([item["joint"] for item in recovery], ["J5"])
        self.assertAlmostEqual(
            result["path_plan"][
                "observed_start_max_joint_limit_recovery_rad"
            ],
            0.0005,
            places=9,
        )

    def test_trajectory_safety_uses_long_horizon_odometry_drift_limit(self) -> None:
        _adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        before = official_tools._adjust_kinematic_observation(ctx, "right")
        start_pose = world.robot_pose()
        drifted_pose = SimpleNamespace(
            pos=np.asarray(start_pose.pos, dtype=np.float64)
            + np.asarray([0.002, 0.0, 0.0], dtype=np.float64),
            yaw=float(start_pose.yaw),
        )
        after = dict(before)
        after["q"] = before["q"].copy()
        after["base_qvel"] = np.zeros(3, dtype=np.float64)
        after["trunk"] = before["trunk"].copy()

        default_monitor = official_tools._adjust_safety_monitor(ctx, before)
        trajectory_monitor = official_tools._adjust_safety_monitor(ctx, before)
        with mock.patch.object(
            world,
            "robot_pose",
            return_value=drifted_pose,
        ):
            default_reason, _ = official_tools._adjust_safety_after_action(
                ctx,
                default_monitor,
                before=before,
                command=before["q"],
                after=after,
            )
            trajectory_reason, report = (
                official_tools._adjust_safety_after_action(
                    ctx,
                    trajectory_monitor,
                    before=before,
                    command=before["q"],
                    after=after,
                    base_translation_limit_m=(
                        official_tools.MOVE_POINT_TO_POINT_BASE_TRANSLATION_LIMIT_M
                    ),
                    base_yaw_limit_rad=(
                        official_tools.MOVE_POINT_TO_POINT_BASE_YAW_LIMIT_RAD
                    ),
                )
            )

        self.assertEqual(default_reason, "base_odometry_disturbance")
        self.assertIsNone(trajectory_reason)
        self.assertAlmostEqual(report["base_translation_m"], 0.002)
        self.assertEqual(
            report["base_translation_limit_m"],
            official_tools.MOVE_POINT_TO_POINT_BASE_TRANSLATION_LIMIT_M,
        )

    def test_waypoint_stall_requires_stationary_low_velocity_proprioception(self) -> None:
        before_q = np.zeros(ARM_DOF, dtype=np.float64)
        moving_q = before_q.copy()
        moving_q[4] = 0.006
        moving_qvel = np.zeros(ARM_DOF, dtype=np.float64)
        moving_qvel[4] = 0.536

        moving_stall, moving_delta, moving_speed = (
            official_tools._move_point_waypoint_stalled_sample(
                previous_error=0.050,
                error=0.052,
                before_q=before_q,
                after_q=moving_q,
                after_qvel=moving_qvel,
            )
        )
        stationary_stall, stationary_delta, stationary_speed = (
            official_tools._move_point_waypoint_stalled_sample(
                previous_error=0.050,
                error=0.052,
                before_q=before_q,
                after_q=before_q.copy(),
                after_qvel=np.zeros(ARM_DOF, dtype=np.float64),
            )
        )

        self.assertFalse(moving_stall)
        self.assertGreater(
            moving_delta,
            official_tools.MOVE_POINT_TO_POINT_STALL_POSITION_DELTA_RAD,
        )
        self.assertGreater(
            moving_speed,
            official_tools.MOVE_POINT_TO_POINT_STALL_QVEL_RAD_S,
        )
        self.assertTrue(stationary_stall)
        self.assertEqual(stationary_delta, 0.0)
        self.assertEqual(stationary_speed, 0.0)

    def test_trajectory_safety_does_not_stall_while_proprio_is_moving(self) -> None:
        _adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        before = official_tools._adjust_kinematic_observation(ctx, "right")
        command = before["q"].copy()
        command[4] += 0.02173097600289653
        after = dict(before)
        after["q"] = before["q"].copy()
        after["q"][4] += 0.0008356571197509766
        after["qvel"] = np.zeros(ARM_DOF, dtype=np.float64)
        after["qvel"][4] = 0.14601309597492218

        monitor = official_tools._adjust_safety_monitor(ctx, before)
        reason = None
        report = {}
        for _ in range(3):
            reason, report = official_tools._adjust_safety_after_action(
                ctx,
                monitor,
                before=before,
                command=command,
                after=after,
                base_translation_limit_m=(
                    official_tools.MOVE_POINT_TO_POINT_BASE_TRANSLATION_LIMIT_M
                ),
                base_yaw_limit_rad=(
                    official_tools.MOVE_POINT_TO_POINT_BASE_YAW_LIMIT_RAD
                ),
                proprio_stall_requires_stationary=True,
            )

        self.assertIsNone(reason)
        self.assertEqual(monitor.proprio_stall_steps, 0)
        self.assertTrue(report["proprio_stall_requires_stationary"])
        self.assertLess(
            report["command_progress_ratio"],
            official_tools.ADJUST_EEF_MIN_PROGRESS_RATIO,
        )

    def test_final_stability_holds_one_immutable_trajectory_target(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observed = official_tools._adjust_kinematic_observation(ctx, "right")
        target_q = observed["q"].copy()
        target_q[4] += 0.04
        target_position, target_quaternion = eef_pose(
            observed["state"], "right", target_q
        )
        generator = official_tools._move_point_verify_stable_pose(
            ctx,
            arm="right",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            target_position=target_position,
            target_quaternion=target_quaternion,
            target_q=target_q,
            pos_tol=0.001,
            ori_tol_deg=1.0,
            timeout_s=5.0,
            expected_episode_id=world.episode_id(),
        )

        first = np.asarray(next(generator), dtype=np.float64)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = (
            observed["q"] + 0.5 * (target_q - observed["q"])
        )
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.1
        adapter.update({"robot_r1::proprio": proprio})
        second = np.asarray(next(generator), dtype=np.float64)
        generator.close()

        np.testing.assert_allclose(
            first[ACTION_SLICES["arm_right"]], target_q, atol=0.0
        )
        np.testing.assert_allclose(
            second[ACTION_SLICES["arm_right"]], target_q, atol=0.0
        )
        self.assertEqual(
            official_tools.MOVE_POINT_TO_POINT_MAX_STABILITY_STEPS, 24
        )

    def test_tracked_final_stability_holds_immutable_planned_endpoint(self) -> None:
        adapter, world = self._ready_world()
        state, (position, quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(
            world,
            adapter,
            {"head": np.asarray(position, dtype=np.float64)},
        )
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        planned_q = measured_q.copy()
        target_points = [
            {
                "name": "head",
                "target_xyz_m": np.asarray(position, dtype=np.float64).tolist(),
            }
        ]
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=["head"],
            target_points=target_points,
            anchors_eef=np.zeros((1, 3), dtype=np.float64),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(position, dtype=np.float64),
            target_quaternion=np.asarray(quaternion, dtype=np.float64),
            target_q=planned_q,
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
            timeout_s=10.0,
            max_steps=20,
            expected_episode_id=world.episode_id(),
        )
        velocity_after_action = (0.09398, 0.06, 0.0, 0.0)
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                ACTION_SLICES["arm_left"]
            ]
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            proprio[PROPRIO_SLICES["arm_left_qvel"]][0] = (
                velocity_after_action[
                    min(len(actions) - 1, len(velocity_after_action) - 1)
                ]
            )
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["goal_latched"])
        self.assertGreaterEqual(report["verification_steps"], 4)
        self.assertEqual(report["planned_target_q"], report["hold_target_q"])
        self.assertEqual(report["final_load_compensation_round_count"], 0)
        self.assertEqual(report["final_visual_correction_round_count"], 0)
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]], planned_q, atol=1e-8
            )

    def test_tracked_final_goal_latch_brakes_loaded_oscillation(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        planned_q = measured_q.copy()
        planned_q[0] += 0.04
        manager = _RigidTrackedManager(
            world,
            adapter,
            {"head": np.asarray(position, dtype=np.float64)},
        )
        world._official_tracked_object_distances = manager
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qvel"]][0] = 0.12
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=["head"],
            target_points=[
                {
                    "name": "head",
                    "target_xyz_m": np.asarray(
                        position, dtype=np.float64
                    ).tolist(),
                }
            ],
            anchors_eef=np.zeros((1, 3), dtype=np.float64),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(position, dtype=np.float64),
            target_quaternion=np.asarray(quaternion, dtype=np.float64),
            target_q=planned_q,
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
            timeout_s=10.0,
            max_steps=12,
            expected_episode_id=world.episode_id(),
        )
        observed_samples = [
            (measured_q + np.eye(1, ARM_DOF, 0)[0] * 0.004, 0.12),
            (measured_q - np.eye(1, ARM_DOF, 0)[0] * 0.002, -0.08),
            (measured_q.copy(), 0.0),
            (measured_q.copy(), 0.0),
        ]
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            sample_q, sample_qvel = observed_samples[
                min(len(actions) - 1, len(observed_samples) - 1)
            ]
            next_proprio = adapter.proprio_vector().copy()
            next_proprio[PROPRIO_SLICES["arm_left_qpos"]] = sample_q
            next_proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            next_proprio[PROPRIO_SLICES["arm_left_qvel"]][0] = sample_qvel
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_tracked_target_converged")
        self.assertEqual(report["goal_latch_brake_event_count"], 1)
        event = report["goal_latch_brake_events"][0]
        self.assertTrue(event["brake_applied"])
        self.assertEqual(event["trigger"], "initial_goal_observation")
        self.assertAlmostEqual(
            event["previous_command_residual_inf_rad"], 0.04, places=7
        )
        np.testing.assert_allclose(report["hold_target_q"], measured_q, atol=1e-8)
        self.assertGreaterEqual(len(actions), 4)
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]], measured_q, atol=1e-8
            )
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )

    def test_tracked_final_goal_latch_persistent_motion_fails_bounded(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        planned_q = measured_q.copy()
        planned_q[0] += 0.04
        manager = _RigidTrackedManager(
            world,
            adapter,
            {"head": np.asarray(position, dtype=np.float64)},
        )
        world._official_tracked_object_distances = manager
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qvel"]][0] = 0.12
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=["head"],
            target_points=[
                {
                    "name": "head",
                    "target_xyz_m": np.asarray(
                        position, dtype=np.float64
                    ).tolist(),
                }
            ],
            anchors_eef=np.zeros((1, 3), dtype=np.float64),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(position, dtype=np.float64),
            target_quaternion=np.asarray(quaternion, dtype=np.float64),
            target_q=planned_q,
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
            timeout_s=10.0,
            max_steps=5,
            expected_episode_id=world.episode_id(),
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            direction = 1.0 if len(actions) % 2 else -1.0
            next_proprio = adapter.proprio_vector().copy()
            next_proprio[PROPRIO_SLICES["arm_left_qpos"]] = measured_q
            next_proprio[PROPRIO_SLICES["arm_left_qpos"]][0] += (
                direction * 0.004
            )
            next_proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            next_proprio[PROPRIO_SLICES["arm_left_qvel"]][0] = (
                direction * 0.12
            )
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "final_velocity_not_stable")
        self.assertEqual(report["verification_steps"], 5)
        self.assertEqual(len(actions), 5)
        self.assertEqual(report["goal_latch_brake_event_count"], 1)
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]], measured_q, atol=1e-8
            )
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )

    def test_tracked_final_coherent_bundle_offset_is_corrected_once(self) -> None:
        class _CoherentlyBiasedTrackedManager(_RigidTrackedManager):
            correction_loss_triggered = False
            correction_loss_remaining = 2

            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                current_q = np.asarray(
                    self.world.arm_qpos_list(self.arm), dtype=np.float64
                )
                arm_motion = float(
                    np.linalg.norm(
                        current_q - self.initial_arm_q,
                        ord=np.inf,
                    )
                )
                correction_motion = float(
                    np.linalg.norm(current_q - planned_q, ord=np.inf)
                )
                if (
                    snapshot["ok"]
                    and arm_motion > 1.0e-5
                    and correction_motion > 1.0e-6
                    and not self.correction_loss_triggered
                ):
                    self.correction_loss_triggered = True
                if (
                    self.correction_loss_triggered
                    and self.correction_loss_remaining > 0
                ):
                    self.correction_loss_remaining -= 1
                    return {
                        "ok": False,
                        "reason": "tracked_point_unavailable",
                        "entries": {},
                        "unavailable": {
                            str(name): "temporary_visual_occlusion"
                            for name in names
                        },
                        "observation_sequence": int(
                            self.adapter.status()["sequence"]
                        ),
                        "episode_id": episode_id,
                        "session_id": self.session_id,
                        "image_id": self.image_id,
                    }
                if snapshot["ok"] and arm_motion > 1.0e-5:
                    for name in names:
                        snapshot["entries"][str(name)][
                            "xyz_in_robot_base_coord_m"
                        ][0] += 0.010
                return snapshot

        adapter, world = self._ready_world()
        state, (position, quaternion) = self._left_eef(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        source = {
            "head": np.asarray(position) + np.array([0.0, -0.03, 0.0]),
            "tail": np.asarray(position) + np.array([0.0, 0.03, 0.0]),
        }
        manager = _CoherentlyBiasedTrackedManager(
            world, adapter, source
        )
        world._official_tracked_object_distances = manager

        planned_q = measured_q.copy()
        planned_q[0] += 0.04
        target_position, target_quaternion = eef_pose(
            state, "left", planned_q
        )
        target_rotation = quat_to_mat_xyzw(target_quaternion)
        point_names = ["head", "tail"]
        target_matrix = np.asarray(
            [
                target_position + target_rotation @ manager.anchors[name]
                for name in point_names
            ],
            dtype=np.float64,
        )
        target_points = [
            {
                "name": name,
                "target_xyz_m": target_matrix[index].astype(float).tolist(),
            }
            for index, name in enumerate(point_names)
        ]
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=point_names,
            target_points=target_points,
            anchors_eef=np.asarray(
                [manager.anchors[name] for name in point_names],
                dtype=np.float64,
            ),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(target_position, dtype=np.float64),
            target_quaternion=np.asarray(target_quaternion, dtype=np.float64),
            target_q=planned_q,
            pos_tol_m=0.004,
            ori_tol_deg=2.0,
            timeout_s=15.0,
            max_steps=80,
            expected_episode_id=world.episode_id(),
        )

        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            next_proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                next_proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                next_proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["final_load_compensation_round_count"], 0)
        self.assertEqual(report["final_visual_correction_round_count"], 1)
        correction = report["final_visual_correction_rounds"][0]
        self.assertTrue(correction["cartesian_replan_used"])
        self.assertEqual(correction["trajectory_schema_version"], 1)
        self.assertEqual(len(correction["trajectory_digest"]), 64)
        self.assertGreater(correction["execution_waypoint_count"], 0)
        self.assertIn("first_feasible_prefix", correction)
        self.assertEqual(
            correction["first_feasible_prefix"]["selection_rule"],
            "first_complete_constraint_feasible_fk_waypoint",
        )
        self.assertLessEqual(
            correction["eef_translation_m"],
            official_tools.MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_TRANSLATION_M,
        )
        recovery_events = report["final_visual_tracking_recovery_events"]
        self.assertEqual(len(recovery_events), 1, report)
        self.assertTrue(recovery_events[0]["recovered"], report)
        self.assertEqual(recovery_events[0]["wait_steps"], 2, report)
        self.assertTrue(
            report["last_live_goal_report"]["constraints"]["ok"], report
        )
        self.assertNotEqual(
            report["initial_planned_target_q"], report["planned_target_q"]
        )
        if ARM_DOF == 8:
            for action in actions:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )

    def test_final_visual_correction_uses_first_feasible_path_prefix(self) -> None:
        _adapter, world = self._ready_world()
        state, _pose = self._left_eef(world)
        q_start = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        q_final = q_start.copy()
        q_final[0] += 0.08
        waypoints = [
            q_start + (q_final - q_start) * fraction
            for fraction in np.linspace(0.1, 1.0, 10)
        ]
        if ARM_DOF == 8:
            for q in waypoints:
                q[7] = 0.0
            q_final[7] = 0.0
        poses = [eef_pose(state, "left", q) for q in waypoints]
        final_position = np.asarray(poses[-1][0], dtype=np.float64)
        coordinate_errors = [
            float(
                np.max(
                    np.abs(np.asarray(position, dtype=np.float64) - final_position)
                )
            )
            for position, _quaternion in poses
        ]
        expected_index = None
        tolerance = None
        for index in range(1, len(coordinate_errors) - 1):
            prior_minimum = min(coordinate_errors[:index])
            if coordinate_errors[index] + 1.0e-8 < prior_minimum:
                expected_index = index
                tolerance = 0.5 * (
                    coordinate_errors[index] + prior_minimum
                )
                break
        self.assertIsNotNone(expected_index, coordinate_errors)
        self.assertIsNotNone(tolerance, coordinate_errors)
        target_points = [
            {
                "name": "marker",
                "target_xyz_m": final_position.astype(float).tolist(),
            }
        ]
        endpoint = {
            "q_final": q_final,
            "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
            "final_eef_position_m": final_position,
            "final_eef_quaternion_xyzw": np.asarray(
                poses[-1][1], dtype=np.float64
            ),
            "constraints": {"ok": True},
        }
        trimmed = (
            official_tools._move_tracked_trim_visual_correction_to_first_feasible(
                state=state,
                arm="left",
                q_start=q_start,
                endpoint=endpoint,
                path_plan={
                    "waypoints": [q.astype(float).tolist() for q in waypoints],
                    "max_joint_step_rad": 0.018,
                },
                target_points=target_points,
                relations=[],
                fixed_points_robot_base_m=None,
                fixed_target_points=None,
                pos_tol_m=float(tolerance),
                ori_tol_deg=5.0,
            )
        )
        report = trimmed["report"]
        self.assertTrue(report["applied"], report)
        self.assertEqual(
            report["first_feasible_waypoint_index"], expected_index
        )
        self.assertEqual(
            len(trimmed["path_plan"]["waypoints"]), expected_index + 1
        )
        self.assertLess(
            report["selected_endpoint_translation_m"],
            report["original_endpoint_translation_m"],
        )
        self.assertTrue(trimmed["endpoint"]["constraints"]["ok"])
        np.testing.assert_allclose(
            trimmed["endpoint"]["q_final"],
            waypoints[expected_index],
            atol=1.0e-12,
        )
        if ARM_DOF == 8:
            self.assertEqual(float(trimmed["endpoint"]["q_final"][7]), 0.0)

    def test_terminal_occluded_on_hand_point_uses_strict_fk_completion(self) -> None:
        live, kwargs = self._terminal_occlusion_bundle_fixture()
        report = official_tools._move_tracked_final_observation_bundle(
            live, **kwargs
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["mode"],
            "partial_rgbd_with_occluded_on_hand_fk_completion",
        )
        self.assertEqual(report["measured_controlled_names"], ["visible"])
        self.assertEqual(report["fk_completed_controlled_names"], ["hidden"])
        np.testing.assert_allclose(report["points"][0], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(report["points"][1], [0.601, 0.0, 0.0])
        self.assertIn("current_evaluator_proprio_local_fk", report["point_sources"]["hidden"])
        self.assertIn("current_evaluator_rgbd", report["point_sources"]["visible"])
        self.assertFalse(report["prediction_values_published_as_observation"])
        evidence = report["occlusion_evidence"]["hidden"]
        self.assertTrue(evidence["ok"], evidence)
        self.assertAlmostEqual(
            evidence["accepted_candidate"]["foreground_depth_margin_m"],
            0.05,
        )

    def test_terminal_occlusion_rejects_background_or_feature_loss(self) -> None:
        live, kwargs = self._terminal_occlusion_bundle_fixture()
        candidate = live["rigid_pair_fusion"]["visual_tracker_pair"][
            "per_point"
        ]["track_hidden"]["rejected_candidates_before_acceptance"][0]
        candidate["candidate_depth_m"] = 0.51
        report = official_tools._move_tracked_final_observation_bundle(
            live, **kwargs
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"],
            "on_hand_point_unavailable_without_occlusion_evidence",
        )
        self.assertFalse(
            report["occlusion_evidence"]["hidden"]["ok"]
        )

    def test_terminal_occlusion_never_completes_off_hand_points(self) -> None:
        live, kwargs = self._terminal_occlusion_bundle_fixture()
        del live["entries"]["fixed_b"]
        live["unavailable"]["fixed_b"] = "point_unobserved"
        report = official_tools._move_tracked_final_observation_bundle(
            live, **kwargs
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["missing_fixed_names"], ["fixed_b"])
        self.assertEqual(report["fk_completed_controlled_names"], [])

    def test_terminal_fully_measured_bundle_preserves_existing_path(self) -> None:
        live, kwargs = self._terminal_occlusion_bundle_fixture()
        live["ok"] = True
        live["reason"] = None
        live["unavailable"] = {}
        live["entries"]["hidden"] = {
            "xyz_in_robot_base_coord_m": [0.499, 0.0, 0.0],
        }
        del live["rigid_pair_fusion"]
        report = official_tools._move_tracked_final_observation_bundle(
            live, **kwargs
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "all_controlled_points_measured_rgbd")
        self.assertEqual(report["fk_completed_controlled_names"], [])
        np.testing.assert_allclose(report["points"][0], [0.499, 0.0, 0.0])
        self.assertTrue(
            all(
                "current_evaluator_rgbd" in source
                for source in report["point_sources"].values()
            )
        )

    def test_terminal_eef_depth_reobservation_uses_one_synchronized_frame(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        adapter = self._terminal_eef_depth_adapter(expected)

        report = official_tools._move_tracked_eef_anchor_depth_reobservation(
            adapter,
            expected,
            current_sequence=17,
        )

        self.assertTrue(report["ok"], report)
        np.testing.assert_allclose(
            report["points_robot_base_m"], expected, atol=1.0e-12
        )
        self.assertEqual(report["sequence_before"], 17)
        self.assertEqual(report["sequence_after"], 17)
        self.assertFalse(report["prediction_values_published_as_observation"])
        self.assertTrue(
            all(item["ok"] for item in report["per_point"]), report
        )

    def test_terminal_eef_depth_reobservation_rejects_sequence_or_surface_mismatch(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        sequence_report = (
            official_tools._move_tracked_eef_anchor_depth_reobservation(
                self._terminal_eef_depth_adapter(
                    expected, status_sequences=[17, 18]
                ),
                expected,
                current_sequence=17,
            )
        )
        self.assertFalse(sequence_report["ok"], sequence_report)
        self.assertEqual(
            sequence_report["reason"], "evaluator_rgbd_sequence_mismatch"
        )

        surface_report = (
            official_tools._move_tracked_eef_anchor_depth_reobservation(
                self._terminal_eef_depth_adapter(
                    expected, depth_offset_m=0.02
                ),
                expected,
                current_sequence=17,
            )
        )
        self.assertFalse(surface_report["ok"], surface_report)
        self.assertEqual(
            surface_report["reason"],
            "complete_eef_anchor_depth_bundle_not_observed",
        )
        self.assertTrue(
            all(
                item["reason"] == "predicted_anchor_depth_surface_mismatch"
                for item in surface_report["per_point"]
            ),
            surface_report,
        )

    def test_terminal_wrong_feature_bundle_keeps_depth_probe_diagnostic(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        fixed_points = np.array(
            [[0.31, -0.02, -0.8], [0.35, 0.01, -0.81]], dtype=np.float64
        )
        tracker_points = expected + np.array([0.020, 0.0, 0.0])
        live = {
            "ok": True,
            "reason": None,
            "entries": {
                "on_a": {"xyz_in_robot_base_coord_m": tracker_points[0]},
                "on_b": {"xyz_in_robot_base_coord_m": tracker_points[1]},
                "off_a": {"xyz_in_robot_base_coord_m": fixed_points[0]},
                "off_b": {"xyz_in_robot_base_coord_m": fixed_points[1]},
            },
            "unavailable": {},
            "observation_sequence": 17,
        }
        report = official_tools._move_tracked_final_observation_bundle(
            live,
            controlled_names=["on_a", "on_b"],
            fixed_names=["off_a", "off_b"],
            current_observation={
                "position": np.zeros(3),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            anchors_eef=expected,
            current_sequence=17,
            anchor_tolerance_m=0.05,
            pos_tol_m=0.012,
            adapter=self._terminal_eef_depth_adapter(expected),
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "all_controlled_points_measured_rgbd")
        self.assertEqual(report["depth_reobserved_controlled_names"], [])
        np.testing.assert_allclose(report["points"], tracker_points, atol=1.0e-12)
        np.testing.assert_allclose(report["fixed_points"], fixed_points)
        self.assertEqual(
            report["point_sources"]["off_a"],
            "current_evaluator_rgbd_and_camera_relative_pose",
        )
        self.assertGreater(report["raw_tracker_anchor_error_m"]["on_a"], 0.019)
        self.assertTrue(report["eef_anchor_depth_reobservation"]["ok"])
        self.assertFalse(
            report["eef_anchor_depth_reobservation"][
                "authoritative_for_terminal_success"
            ]
        )
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_terminal_rejected_complete_bundle_depth_probe_cannot_succeed(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        fixed_points = np.array(
            [[0.31, -0.02, -0.8], [0.35, 0.01, -0.81]], dtype=np.float64
        )
        live = {
            "ok": False,
            "reason": "tracked_point_unavailable",
            "entries": {
                "off_a": {"xyz_in_robot_base_coord_m": fixed_points[0]},
                "off_b": {"xyz_in_robot_base_coord_m": fixed_points[1]},
            },
            "unavailable": {
                "on_a": "point_unobserved",
                "on_b": "point_unobserved",
            },
            "observation_sequence": 17,
        }
        report = official_tools._move_tracked_final_observation_bundle(
            live,
            controlled_names=["on_a", "on_b"],
            fixed_names=["off_a", "off_b"],
            current_observation={
                "position": np.zeros(3),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            anchors_eef=expected,
            current_sequence=17,
            anchor_tolerance_m=0.05,
            pos_tol_m=0.012,
            adapter=self._terminal_eef_depth_adapter(expected),
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["mode"], "unavailable")
        self.assertEqual(report["reason"], "tracked_point_unavailable")
        self.assertEqual(report["missing_controlled_names"], ["on_a", "on_b"])
        self.assertEqual(report["depth_reobserved_controlled_names"], [])
        self.assertEqual(
            report["eef_anchor_depth_reobservation"]["trigger"],
            "complete_controlled_tracker_bundle_unavailable",
        )
        self.assertTrue(report["eef_anchor_depth_reobservation"]["ok"])
        self.assertTrue(
            report["eef_anchor_depth_reobservation"][
                "reacquisition_diagnostic_only"
            ]
        )
        self.assertFalse(
            report["eef_anchor_depth_reobservation"][
                "authoritative_for_terminal_success"
            ]
        )
        self.assertNotIn("points", report)
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_terminal_rejected_bundle_depth_mismatch_still_fails(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        live = {
            "ok": False,
            "reason": "tracked_point_unavailable",
            "entries": {},
            "unavailable": {
                "on_a": "point_unobserved",
                "on_b": "point_unobserved",
            },
            "observation_sequence": 17,
        }
        report = official_tools._move_tracked_final_observation_bundle(
            live,
            controlled_names=["on_a", "on_b"],
            fixed_names=[],
            current_observation={
                "position": np.zeros(3),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            anchors_eef=expected,
            current_sequence=17,
            anchor_tolerance_m=0.05,
            pos_tol_m=0.012,
            adapter=self._terminal_eef_depth_adapter(
                expected, depth_offset_m=0.02
            ),
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "tracked_point_unavailable")
        self.assertEqual(report["missing_controlled_names"], ["on_a", "on_b"])
        self.assertFalse(report["eef_anchor_depth_reobservation"]["ok"])
        self.assertEqual(
            report["eef_anchor_depth_reobservation"]["reason"],
            "complete_eef_anchor_depth_bundle_not_observed",
        )
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_terminal_failed_depth_reobservation_does_not_publish_fk_prediction(self) -> None:
        expected = np.array(
            [[0.0, 0.0, -0.5], [4.0 * 0.5 / 42.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        tracker_points = expected + np.array([0.020, 0.0, 0.0])
        live = {
            "ok": True,
            "reason": None,
            "entries": {
                "on_a": {"xyz_in_robot_base_coord_m": tracker_points[0]},
                "on_b": {"xyz_in_robot_base_coord_m": tracker_points[1]},
            },
            "unavailable": {},
            "observation_sequence": 17,
        }
        report = official_tools._move_tracked_final_observation_bundle(
            live,
            controlled_names=["on_a", "on_b"],
            fixed_names=[],
            current_observation={
                "position": np.zeros(3),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            anchors_eef=expected,
            current_sequence=17,
            anchor_tolerance_m=0.05,
            pos_tol_m=0.012,
            adapter=self._terminal_eef_depth_adapter(
                expected, depth_offset_m=0.02
            ),
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "all_controlled_points_measured_rgbd")
        self.assertFalse(report["eef_anchor_depth_reobservation"]["ok"])
        np.testing.assert_allclose(report["points"], tracker_points)
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_terminal_occlusion_bundle_can_latch_stable_goal(self) -> None:
        class OneMarkerOccludedManager(_RigidTrackedManager):
            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                snapshot["entries"].pop("hidden")
                snapshot.update(
                    {
                        "ok": False,
                        "reason": "tracked_point_unavailable",
                        "unavailable": {"hidden": "point_unobserved"},
                        "rigid_pair_fusion": {
                            "names": ["hidden", "visible"],
                            "visual_tracker_pair": {
                                "track_ids": ["track_hidden", "track_visible"],
                                "per_point": {
                                    "track_hidden": {
                                        "status": "lost",
                                        "rejected_candidates_before_acceptance": [
                                            {
                                                "reason": "eef_prior_depth_layer_mismatch",
                                                "candidate_method": "depth_component",
                                                "candidate_status": "observed",
                                                "candidate_depth_m": 0.45,
                                                "predicted_depth_m": 0.50,
                                                "pixel_error_px": 3.0,
                                                "pixel_error_limit_px": 24.0,
                                            }
                                        ],
                                    }
                                },
                            },
                        },
                    }
                )
                return snapshot

        adapter, world = self._ready_world()
        state, (position, quaternion) = self._left_eef(world)
        source = {
            "hidden": np.asarray(position) + np.array([0.0, -0.03, 0.0]),
            "visible": np.asarray(position) + np.array([0.0, 0.03, 0.0]),
        }
        manager = OneMarkerOccludedManager(world, adapter, source)
        world._official_tracked_object_distances = manager
        target_q = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target_points = [
            {
                "name": name,
                "target_xyz_m": np.asarray(source[name], dtype=np.float64).tolist(),
            }
            for name in ("hidden", "visible")
        ]
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=["hidden", "visible"],
            target_points=target_points,
            anchors_eef=np.asarray(
                [manager.anchors["hidden"], manager.anchors["visible"]],
                dtype=np.float64,
            ),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(position, dtype=np.float64),
            target_quaternion=np.asarray(quaternion, dtype=np.float64),
            target_q=target_q,
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
            timeout_s=10.0,
            max_steps=10,
            expected_episode_id=world.episode_id(),
        )
        emitted = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                stable = stopped.value
                break
            emitted.append(action)
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                ACTION_SLICES["arm_left"]
            ]
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(stable["ok"], stable)
        self.assertTrue(stable["goal_latched"], stable)
        self.assertTrue(emitted)
        completion = stable["last_live_goal_report"][
            "on_hand_observation_completion"
        ]
        self.assertEqual(
            completion["mode"],
            "partial_rgbd_with_occluded_on_hand_fk_completion",
        )
        self.assertEqual(completion["fk_completed_controlled_names"], ["hidden"])

    def test_final_visual_correction_tracking_loss_holds_and_fails_bounded(self) -> None:
        adapter, world = self._ready_world()
        state, (position, _quaternion) = self._left_eef(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        source = {
            "head": np.asarray(position) + np.array([0.0, -0.03, 0.0]),
            "tail": np.asarray(position) + np.array([0.0, 0.03, 0.0]),
        }
        manager = _RigidTrackedManager(world, adapter, source)
        world._official_tracked_object_distances = manager
        planned_q = measured_q.copy()
        planned_q[0] += 0.04
        target_position, target_quaternion = eef_pose(
            state, "left", planned_q
        )
        target_rotation = quat_to_mat_xyzw(target_quaternion)
        point_names = ["head", "tail"]
        target_matrix = np.asarray(
            [
                target_position + target_rotation @ manager.anchors[name]
                for name in point_names
            ],
            dtype=np.float64,
        )
        target_points = [
            {
                "name": name,
                "target_xyz_m": target_matrix[index].astype(float).tolist(),
            }
            for index, name in enumerate(point_names)
        ]
        real_snapshot = manager.observed_active_points_snapshot
        correction_loss_triggered = False

        def persistently_lost_during_correction(names, *, episode_id):
            nonlocal correction_loss_triggered
            snapshot = real_snapshot(names, episode_id=episode_id)
            current_q = np.asarray(
                world.arm_qpos_list("left"), dtype=np.float64
            )
            arm_motion = float(
                np.linalg.norm(current_q - measured_q, ord=np.inf)
            )
            correction_motion = float(
                np.linalg.norm(current_q - planned_q, ord=np.inf)
            )
            if (
                arm_motion > 1.0e-5
                and correction_motion > 1.0e-6
            ):
                correction_loss_triggered = True
            if correction_loss_triggered:
                return {
                    "ok": False,
                    "reason": "tracked_point_unavailable",
                    "entries": {},
                    "unavailable": {
                        str(name): "persistent_visual_occlusion"
                        for name in names
                    },
                    "observation_sequence": int(adapter.status()["sequence"]),
                    "episode_id": episode_id,
                    "session_id": manager.session_id,
                    "image_id": manager.image_id,
                }
            if snapshot["ok"] and arm_motion > 1.0e-5:
                for name in names:
                    snapshot["entries"][str(name)][
                        "xyz_in_robot_base_coord_m"
                    ][0] += 0.010
            return snapshot

        ctx, _result = self._ctx(world)
        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=persistently_lost_during_correction,
        ), mock.patch.object(
            official_tools,
            "MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS",
            3,
        ):
            generator = official_tools._move_tracked_verify_stable_target(
                ctx,
                arm="left",
                adapter=adapter,
                observation_sequence=adapter.observation_metadata()[0],
                manager=manager,
                point_names=point_names,
                target_points=target_points,
                anchors_eef=np.asarray(
                    [manager.anchors[name] for name in point_names],
                    dtype=np.float64,
                ),
                anchor_tolerance_m=0.05,
                pair_distance_tolerance_m=0.018,
                target_position=np.asarray(target_position, dtype=np.float64),
                target_quaternion=np.asarray(
                    target_quaternion, dtype=np.float64
                ),
                target_q=planned_q,
                pos_tol_m=0.004,
                ori_tol_deg=2.0,
                timeout_s=15.0,
                max_steps=80,
                expected_episode_id=world.episode_id(),
            )
            actions = []
            while True:
                try:
                    action = np.asarray(next(generator), dtype=np.float64)
                except StopIteration as stopped:
                    report = stopped.value
                    break
                actions.append(action.copy())
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                    ACTION_SLICES["arm_left"]
                ]
                proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "final_visual_correction_tracking_lost"
        )
        self.assertIn("did not reacquire within 3", report["error"])
        self.assertEqual(len(report["final_visual_tracking_recovery_events"]), 1)
        event = report["final_visual_tracking_recovery_events"][0]
        self.assertFalse(event["recovered"])
        self.assertEqual(event["wait_steps"], 3)
        self.assertGreaterEqual(len(actions), 3)
        np.testing.assert_allclose(
            actions[-1][ACTION_SLICES["arm_left"]],
            actions[-2][ACTION_SLICES["arm_left"]],
            atol=1.0e-12,
        )
        if ARM_DOF == 8:
            for action in actions:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )

    def test_npoint_final_rigid_geometry_inconsistency_is_bounded(self) -> None:
        class _RigidGeometryDriftManager(_RigidTrackedManager):
            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                if snapshot["ok"] and len(names) == 3:
                    snapshot["entries"][str(names[2])][
                        "xyz_in_robot_base_coord_m"
                    ][1] += 0.025
                return snapshot

        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        names = ["marker_1", "marker_2", "marker_3"]
        source = np.asarray(position, dtype=np.float64) + np.asarray(
            [
                [0.0, 0.0, -0.04],
                [0.06, 0.0, -0.04],
                [0.0, 0.05, -0.04],
            ],
            dtype=np.float64,
        )
        manager = _RigidGeometryDriftManager(
            world, adapter, dict(zip(names, source))
        )
        world._official_tracked_object_distances = manager
        target_q = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target_points = [
            {
                "name": name,
                "target_xyz_m": source[index].astype(float).tolist(),
            }
            for index, name in enumerate(names)
        ]
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=names,
            target_points=target_points,
            anchors_eef=np.asarray(
                [manager.anchors[name] for name in names],
                dtype=np.float64,
            ),
            anchor_tolerance_m=0.03,
            pair_distance_tolerance_m=0.02,
            target_position=np.asarray(position, dtype=np.float64),
            target_quaternion=np.asarray(quaternion, dtype=np.float64),
            target_q=target_q,
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
            timeout_s=30.0,
            max_steps=30,
            expected_episode_id=world.episode_id(),
        )

        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["reason"], "final_visual_correction_tracking_lost"
        )
        self.assertEqual(
            report["verification_steps"],
            official_tools.MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS,
        )
        self.assertIn("rigid geometry remained inconsistent", report["error"])
        self.assertNotIn("velocity did not settle", report["error"])
        failure = report["final_visual_correction_failure"]
        self.assertEqual(
            failure["last_tracker_reason"],
            "tracked_point_rigid_geometry_inconsistent",
        )
        self.assertEqual(
            failure["recovery_steps"],
            official_tools.MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS,
        )
        self.assertEqual(
            len(actions),
            official_tools.MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS,
        )
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_right"]][7]), 0.0
                )

    def test_tracked_final_load_compensation_reaches_planned_pose(self) -> None:
        adapter, world = self._ready_world()
        state, _current_pose = self._left_eef(world)
        measured_q = np.asarray(
            world.arm_qpos_list("left"), dtype=np.float64
        )
        planned_q = measured_q.copy()
        planned_q[0] += 0.04
        residual = np.zeros(ARM_DOF, dtype=np.float64)
        residual[0] = 0.01

        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = planned_q - residual
        adapter.update({"robot_r1::proprio": proprio})
        loaded_state, (loaded_position, _loaded_quaternion) = self._left_eef(world)
        target_position, target_quaternion = eef_pose(
            loaded_state,
            "left",
            planned_q,
        )
        manager = _RigidTrackedManager(
            world,
            adapter,
            {"head": np.asarray(loaded_position, dtype=np.float64)},
        )
        world._official_tracked_object_distances = manager
        ctx, _result = self._ctx(world)
        generator = official_tools._move_tracked_verify_stable_target(
            ctx,
            arm="left",
            adapter=adapter,
            observation_sequence=adapter.observation_metadata()[0],
            manager=manager,
            point_names=["head"],
            target_points=[
                {
                    "name": "head",
                    "target_xyz_m": np.asarray(
                        target_position, dtype=np.float64
                    ).tolist(),
                }
            ],
            anchors_eef=np.zeros((1, 3), dtype=np.float64),
            anchor_tolerance_m=0.05,
            pair_distance_tolerance_m=0.018,
            target_position=np.asarray(target_position, dtype=np.float64),
            target_quaternion=np.asarray(target_quaternion, dtype=np.float64),
            target_q=planned_q,
            pos_tol_m=0.001,
            ori_tol_deg=1.0,
            timeout_s=10.0,
            max_steps=30,
            expected_episode_id=world.episode_id(),
        )

        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action.copy())
            next_proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                next_proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                next_proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            next_proprio[PROPRIO_SLICES["arm_left_qpos"]] -= residual
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["final_load_compensation_round_count"], 1)
        rounds = report["final_load_compensation_rounds"]
        self.assertFalse(rounds[0]["cartesian_replan_used"])
        self.assertLessEqual(
            rounds[0]["command_delta_inf_rad"],
            official_tools.MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD,
        )
        command_q = np.asarray(report["hold_target_q"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(report["planned_target_q"], dtype=np.float64),
            planned_q,
            atol=0.0,
        )
        self.assertAlmostEqual(command_q[0] - planned_q[0], 0.01, places=8)
        np.testing.assert_allclose(
            world.arm_qpos_list("left"), planned_q, atol=1e-8
        )
        command_history = np.asarray(
            [action[ACTION_SLICES["arm_left"]] for action in actions],
            dtype=np.float64,
        )
        if len(command_history) > 1:
            self.assertLessEqual(
                float(np.max(np.abs(np.diff(command_history, axis=0)))),
                official_tools.MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD + 1e-9,
            )
        self.assertTrue(np.all(np.diff(command_history[:, 0]) >= -1e-12))
        if ARM_DOF == 8:
            np.testing.assert_allclose(command_history[:, 7], 0.0, atol=0.0)

    def test_final_stability_filters_only_stationary_joint_limit_chatter(self) -> None:
        q = np.zeros(ARM_DOF, dtype=np.float64)
        q[4] = official_tools._ARM_LIMITS["right"][4, 0] - 0.0018
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[2] = 0.0213
        qvel[4] = 0.1152

        stationary_delta = np.zeros(ARM_DOF, dtype=np.float64)
        stationary_delta[4] = 0.0002
        raw, effective, step, filtered = (
            official_tools._move_point_effective_stability_qvel(
                arm="right",
                previous_q=q - stationary_delta,
                current_q=q,
                current_qvel=qvel,
            )
        )
        self.assertAlmostEqual(raw, 0.1152)
        self.assertAlmostEqual(effective, 0.0213)
        self.assertAlmostEqual(step, 0.0002)
        self.assertEqual(filtered, ["J5"])

        moving = q.copy()
        moving[4] -= 0.003
        _raw, moving_effective, _step, moving_filtered = (
            official_tools._move_point_effective_stability_qvel(
                arm="right",
                previous_q=moving,
                current_q=q,
                current_qvel=qvel,
            )
        )
        self.assertAlmostEqual(moving_effective, 0.1152)
        self.assertEqual(moving_filtered, [])

        interior = q.copy()
        interior[4] = 0.0
        _raw, interior_effective, _step, interior_filtered = (
            official_tools._move_point_effective_stability_qvel(
                arm="right",
                previous_q=interior.copy(),
                current_q=interior,
                current_qvel=qvel,
            )
        )
        self.assertAlmostEqual(interior_effective, 0.1152)
        self.assertEqual(interior_filtered, [])

    def test_tracked_final_stability_requires_consecutive_qvel_coherence(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        current = np.mean(limits, axis=1)
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[4] = 0.08
        qvel[5] = -0.05
        streak = np.zeros(ARM_DOF, dtype=np.int64)

        reports = []
        for _sample in range(
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
        ):
            previous = current.copy()
            current = current.copy()
            current[4] += 1.0e-5
            current[5] -= 1.0e-5
            report = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=previous,
                current_q=current,
                current_qvel=qvel,
                observation_dt_s=0.5,
                previous_incoherence_streak=streak,
                goal_satisfied=True,
            )
            reports.append(report)
            streak = np.asarray(
                report["qvel_incoherence_streak_by_joint"], dtype=np.int64
            )

        for report in reports[:-1]:
            self.assertAlmostEqual(report["effective_qvel_inf_rad_s"], 0.08)
            self.assertEqual(report["coherence_filtered_velocity_joints"], [])
        final = reports[-1]
        self.assertTrue(final["raw_velocity_hard_ok"])
        self.assertLessEqual(
            final["effective_qvel_inf_rad_s"],
            official_tools.MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
        )
        self.assertEqual(
            final["coherence_filtered_velocity_joints"], ["J5", "J6"]
        )

    def test_tracked_final_stability_never_filters_unbounded_qvel(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        current = np.mean(limits, axis=1)
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[4] = (
            official_tools.MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S
            + 0.001
        )
        streak = np.full(
            ARM_DOF,
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS,
            dtype=np.int64,
        )
        report = official_tools._move_tracked_coherent_stability_velocity(
            arm="right",
            previous_q=current - 1.0e-6,
            current_q=current,
            current_qvel=qvel,
            observation_dt_s=0.5,
            previous_incoherence_streak=streak,
            goal_satisfied=True,
        )

        self.assertFalse(report["raw_velocity_hard_ok"])
        self.assertGreater(
            report["effective_qvel_inf_rad_s"],
            official_tools.MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
        )
        self.assertEqual(report["coherence_filtered_velocity_joints"], [])
        self.assertEqual(report["qvel_incoherence_streak_by_joint"][4], 0)

    def test_tracked_final_stability_filters_loaded_interior_qvel_only_after_position_coherence(
        self,
    ) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        current = np.mean(limits, axis=1)
        hold = current.copy()
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[5] = -0.216
        streak = np.zeros(ARM_DOF, dtype=np.int64)

        reports = []
        for _sample in range(
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
        ):
            report = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=current.copy(),
                current_q=current.copy(),
                current_qvel=qvel,
                observation_dt_s=0.8,
                previous_incoherence_streak=streak,
                goal_satisfied=True,
                hold_q=hold,
            )
            reports.append(report)
            streak = np.asarray(
                report["qvel_incoherence_streak_by_joint"], dtype=np.int64
            )

        for report in reports[:-1]:
            self.assertFalse(report["raw_velocity_hard_ok"], report)
            self.assertEqual(report["coherence_filtered_velocity_joints"], [])
        final = reports[-1]
        self.assertTrue(final["raw_velocity_hard_ok"], final)
        self.assertEqual(final["coherence_filtered_velocity_joints"], ["J6"])
        self.assertEqual(
            final["command_limit_coherence_filtered_velocity_joints"], []
        )
        self.assertLessEqual(
            final["effective_qvel_inf_rad_s"],
            official_tools.MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
        )

        for unsafe_qvel, previous in (
            (
                official_tools.MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S
                + 0.01,
                current.copy(),
            ),
            (-0.216, current.copy() - 0.02),
        ):
            unsafe = qvel.copy()
            unsafe[5] = unsafe_qvel
            rejected = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=previous,
                current_q=current.copy(),
                current_qvel=unsafe,
                observation_dt_s=0.5,
                previous_incoherence_streak=np.full(
                    ARM_DOF, 20, dtype=np.int64
                ),
                goal_satisfied=True,
                hold_q=hold,
            )
            self.assertFalse(rejected["raw_velocity_hard_ok"], rejected)
            self.assertEqual(rejected["coherence_filtered_velocity_joints"], [])

    def test_tracked_final_velocity_evidence_survives_visual_goal_flicker(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        current = np.mean(limits, axis=1)
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[5] = 0.08
        streak = np.zeros(ARM_DOF, dtype=np.int64)

        for _sample in range(
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
        ):
            previous = current.copy()
            current = current.copy()
            current[5] += 1.0e-5
            report = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=previous,
                current_q=current,
                current_qvel=qvel,
                observation_dt_s=0.5,
                previous_incoherence_streak=streak,
                goal_satisfied=False,
            )
            streak = np.asarray(
                report["qvel_incoherence_streak_by_joint"], dtype=np.int64
            )

        self.assertEqual(
            int(streak[5]),
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS,
        )
        report = official_tools._move_tracked_coherent_stability_velocity(
            arm="right",
            previous_q=current.copy(),
            current_q=current.copy(),
            current_qvel=qvel,
            observation_dt_s=0.5,
            previous_incoherence_streak=streak,
            goal_satisfied=True,
        )
        self.assertLessEqual(
            report["effective_qvel_inf_rad_s"],
            official_tools.MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
        )
        self.assertEqual(
            report["coherence_filtered_velocity_joints"], ["J6"]
        )

    def test_tracked_final_stability_filters_loaded_limit_qvel_only(self) -> None:
        limits = official_tools._ARM_LIMITS["right"][:ARM_DOF]
        hold = np.mean(limits, axis=1)
        hold[5] = limits[5, 0]
        current = hold.copy()
        current[5] += 0.0069
        qvel = np.zeros(ARM_DOF, dtype=np.float64)
        qvel[5] = -0.24
        streak = np.zeros(ARM_DOF, dtype=np.int64)

        for sample in range(
            official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
        ):
            report = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=current.copy(),
                current_q=current.copy(),
                current_qvel=qvel,
                observation_dt_s=0.5,
                previous_incoherence_streak=streak,
                goal_satisfied=True,
                hold_q=hold,
            )
            streak = np.asarray(
                report["qvel_incoherence_streak_by_joint"], dtype=np.int64
            )
            if sample + 1 < (
                official_tools.MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
            ):
                self.assertFalse(report["raw_velocity_hard_ok"])

        self.assertTrue(report["raw_velocity_hard_ok"], report)
        self.assertLessEqual(
            report["effective_qvel_inf_rad_s"],
            official_tools.MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
        )
        self.assertEqual(
            report["command_limit_coherence_filtered_velocity_joints"],
            ["J6"],
        )

        away_from_limit = qvel.copy()
        away_from_limit[5] = 0.24
        generic_stationary = (
            official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=current.copy(),
                current_q=current.copy(),
                current_qvel=away_from_limit,
                observation_dt_s=0.5,
                previous_incoherence_streak=np.full(
                    ARM_DOF, 20, dtype=np.int64
                ),
                goal_satisfied=True,
                hold_q=hold,
            )
        )
        self.assertTrue(generic_stationary["raw_velocity_hard_ok"])
        self.assertEqual(
            generic_stationary[
                "command_limit_coherence_filtered_velocity_joints"
            ],
            [],
        )
        self.assertEqual(
            generic_stationary["coherence_filtered_velocity_joints"], ["J6"]
        )

        for unsafe_qvel, previous in (
            (-0.51, current.copy()),
            (-0.24, current.copy() - 0.02),
        ):
            unsafe = qvel.copy()
            unsafe[5] = unsafe_qvel
            rejected = official_tools._move_tracked_coherent_stability_velocity(
                arm="right",
                previous_q=previous,
                current_q=current.copy(),
                current_qvel=unsafe,
                observation_dt_s=0.5,
                previous_incoherence_streak=np.full(
                    ARM_DOF, 20, dtype=np.int64
                ),
                goal_satisfied=True,
                hold_q=hold,
            )
            self.assertFalse(rejected["raw_velocity_hard_ok"], rejected)
            self.assertEqual(
                rejected[
                    "command_limit_coherence_filtered_velocity_joints"
                ],
                [],
            )

    def test_full_execution_accepts_rebased_handoff_and_counts_bridge_waypoints(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            observations,
        )
        q_plan_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        signed_target = q_plan_start.copy()
        signed_target[0] += 0.10
        if ARM_DOF == 8:
            signed_target[7] = 0.0

        # Simulate the bounded post-planning handoff: the current proprio is
        # close to, but not equal to, the frozen planning start.
        proprio = adapter.proprio_vector().copy()
        handoff_q = q_plan_start.copy()
        handoff_q[0] += 0.045
        if ARM_DOF == 8:
            handoff_q[7] = 0.0
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = handoff_q
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.35,
            },
            "waypoints": [signed_target.astype(float).tolist()],
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=120,
            operation_name="move_tracked_point",
            allow_start_rebase=True,
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                motion = stopped.value
                break
            actions.append(action)
            next_proprio = adapter.proprio_vector().copy()
            next_proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                ACTION_SLICES["arm_left"]
            ]
            next_proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(motion["ok"], motion)
        self.assertGreater(motion["execution_bridge_waypoint_count"], 0)
        self.assertEqual(motion["signed_waypoint_count"], 1)
        self.assertEqual(
            motion["waypoints_completed"], motion["waypoint_count"]
        )
        self.assertGreater(motion["waypoint_count"], motion["signed_waypoint_count"])
        self.assertTrue(actions)

    def test_continuous_execution_dispatches_each_waypoint_once(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            observations,
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        waypoints = []
        for offset in (0.03, 0.06, 0.09):
            target = q_start.copy()
            target[0] += offset
            if ARM_DOF == 8:
                target[7] = 0.0
            waypoints.append(target)
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.12,
            },
            "waypoints": [target.astype(float).tolist() for target in waypoints],
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
        )
        commanded = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                motion = stopped.value
                break
            target = np.asarray(
                action[ACTION_SLICES["arm_left"]], dtype=np.float64
            )
            commanded.append(target.copy())
            proprio = adapter.proprio_vector().copy()
            observed = np.asarray(
                proprio[PROPRIO_SLICES["arm_left_qpos"]], dtype=np.float64
            )
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = (
                observed + 0.25 * (target - observed)
            )
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["execution_mode"], "continuous_waypoint_stream")
        self.assertEqual(motion["steps_executed"], len(waypoints))
        self.assertEqual(len(commanded), len(waypoints))
        self.assertFalse(motion["gripper_recovery_enabled"])
        self.assertEqual(motion["gripper_recovery_steps"], 0)
        self.assertEqual(motion["gripper_recovery_events"], [])
        self.assertEqual(motion["gripper_recovery_samples"], [])
        np.testing.assert_allclose(commanded, waypoints, atol=1e-6)

    def test_npoint_execution_holds_observed_pose_then_resumes_same_waypoint(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        first = q_start.copy()
        first[0] += 0.04
        final = q_start.copy()
        final[0] += 0.08
        if ARM_DOF == 8:
            first[7] = 0.0
            final[7] = 0.0
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.35,
            },
            "waypoints": [first.astype(float).tolist(), final.astype(float).tolist()],
        }
        monitor_calls = 0

        def monitor(_after, _sequence):
            nonlocal monitor_calls
            monitor_calls += 1
            unavailable = monitor_calls <= 2
            return None, {
                "temporarily_unobserved": unavailable,
                "tracker_reason": (
                    "tracked_point_unavailable" if unavailable else None
                ),
            }

        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            step_monitor=monitor,
            operation_name="move_tracked_point",
            tracker_reacquisition_max_steps=4,
        )
        commanded = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                motion = stopped.value
                break
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            if ARM_DOF == 8:
                self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)
                self.assertEqual(float(action[ACTION_SLICES["arm_right"]][7]), 0.0)
            target = np.asarray(action[ACTION_SLICES["arm_left"]], dtype=np.float64)
            commanded.append(target.copy())
            proprio = adapter.proprio_vector().copy()
            observed = np.asarray(
                proprio[PROPRIO_SLICES["arm_left_qpos"]], dtype=np.float64
            )
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = (
                observed + 0.5 * (target - observed)
            )
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["waypoints_completed"], 2)
        self.assertEqual(motion["tracker_reacquisition_hold_steps"], 2)
        self.assertEqual(motion["tracker_reacquisition_max_steps"], 4)
        self.assertEqual(
            [item["event"] for item in motion["tracker_reacquisition_events"]],
            ["hold_started", "observation_reacquired"],
        )
        self.assertGreaterEqual(len(commanded), 5)
        np.testing.assert_allclose(commanded[1], commanded[2], atol=1e-7)
        self.assertLess(
            float(np.linalg.norm(commanded[1] - first, ord=np.inf)),
            float(np.linalg.norm(q_start - first, ord=np.inf)),
        )
        np.testing.assert_allclose(commanded[-1], final, atol=1e-6)

    def test_npoint_execution_persistent_loss_has_bounded_hold(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.04
        if ARM_DOF == 8:
            target[7] = 0.0
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.35,
            },
            "waypoints": [target.astype(float).tolist()],
        }

        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            step_monitor=lambda _after, _sequence: (
                None,
                {
                    "temporarily_unobserved": True,
                    "tracker_reason": "tracked_point_unavailable",
                },
            ),
            operation_name="move_tracked_point",
            tracker_reacquisition_max_steps=3,
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32).reshape(-1)
            except StopIteration as stopped:
                motion = stopped.value
                break
            actions.append(action)
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                ACTION_SLICES["arm_left"]
            ]
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertFalse(motion["ok"], motion)
        self.assertEqual(motion["reason"], "tracked_point_reacquisition_failed")
        self.assertEqual(motion["tracker_reacquisition_hold_steps"], 3)
        self.assertEqual(len(actions), 4)
        self.assertEqual(
            motion["tracker_reacquisition_events"][-1]["event"],
            "reacquisition_exhausted",
        )
        self.assertEqual(motion["waypoints_completed"], 0)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            if ARM_DOF == 8:
                self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)
                self.assertEqual(float(action[ACTION_SLICES["arm_right"]][7]), 0.0)

    def test_near_open_effort_hold_never_emits_close_intent(self) -> None:
        capture_q = {
            "left": np.asarray([0.02, 0.02], dtype=np.float64),
            "right": np.asarray(
                [0.049810975790023804, 0.04953378066420555],
                dtype=np.float64,
            ),
        }
        observed_q = {
            "left": np.asarray([0.02, 0.02], dtype=np.float64),
            "right": np.asarray([0.04990, 0.03904276713728905]),
        }
        observed_qvel = {
            "left": np.zeros(2, dtype=np.float64),
            "right": np.asarray([0.01, 0.195], dtype=np.float64),
        }
        world = SimpleNamespace(
            gripper_close_keepalive_active=lambda _side: False,
            gripper_uses_effort=lambda _side: True,
            gripper_qpos_list=lambda side: observed_q[side].tolist(),
            gripper_qvel_list=lambda side: observed_qvel[side].tolist(),
        )
        overrides, report = official_tools._move_point_gripper_action_overrides(
            SimpleNamespace(world=world),
            {"gripper_q": capture_q},
            enable_open_recovery=True,
        )

        command = np.asarray(overrides["gripper_effort_right"])
        self.assertTrue(np.all(np.isfinite(command)))
        self.assertGreaterEqual(float(np.min(command)), 0.0)
        self.assertEqual(command[0], 0.0)
        self.assertEqual(
            command[1],
            official_tools.MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N,
        )
        self.assertEqual(report["right"]["outward_only_fingers"], [0, 1])
        self.assertEqual(report["right"]["boosted_open_fingers"], [1])

        intermediate_capture = {
            "gripper_q": {
                "left": capture_q["left"],
                "right": np.asarray([0.02, 0.02], dtype=np.float64),
            }
        }
        observed_q["right"] = np.asarray([0.021, 0.018], dtype=np.float64)
        observed_qvel["right"] = np.zeros(2, dtype=np.float64)
        intermediate, intermediate_report = (
            official_tools._move_point_gripper_action_overrides(
                SimpleNamespace(world=world),
                intermediate_capture,
                enable_open_recovery=True,
            )
        )
        intermediate_command = np.asarray(
            intermediate["gripper_effort_right"]
        )
        self.assertLess(intermediate_command[0], 0.0)
        self.assertAlmostEqual(intermediate_command[1], 0.08, places=9)
        self.assertLessEqual(
            abs(float(intermediate_command[0])),
            official_tools.MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N,
        )
        self.assertEqual(
            intermediate_report["right"]["outward_only_fingers"],
            [],
        )
        self.assertEqual(
            intermediate_report["right"]["boosted_open_fingers"],
            [],
        )

    def test_near_open_fast_path_clamps_velocity_close_intent_without_boost(self) -> None:
        capture_q = {
            "left": np.asarray([0.02, 0.02], dtype=np.float64),
            "right": np.asarray([0.0475, 0.0035], dtype=np.float64),
        }
        observed_q = {
            side: values.copy() for side, values in capture_q.items()
        }
        observed_qvel = {
            "left": np.zeros(2, dtype=np.float64),
            "right": np.asarray([0.02, 0.02], dtype=np.float64),
        }
        world = SimpleNamespace(
            gripper_close_keepalive_active=lambda _side: False,
            gripper_uses_effort=lambda _side: True,
            gripper_qpos_list=lambda side: observed_q[side].tolist(),
            gripper_qvel_list=lambda side: observed_qvel[side].tolist(),
        )

        overrides, report = official_tools._move_point_gripper_action_overrides(
            SimpleNamespace(world=world),
            {"gripper_q": capture_q},
            enable_open_recovery=False,
        )

        command = np.asarray(overrides["gripper_effort_right"])
        self.assertEqual(command[0], 0.0)
        self.assertLess(command[1], 0.0)
        self.assertGreaterEqual(
            command[1],
            -official_tools.MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N,
        )
        self.assertEqual(report["right"]["outward_only_fingers"], [0])
        self.assertEqual(report["right"]["boosted_open_fingers"], [])

    def test_close_keepalive_is_unchanged_by_open_recovery(self) -> None:
        capture_q = {
            "left": np.asarray([0.02, 0.02], dtype=np.float64),
            "right": np.asarray([0.0475, 0.0035], dtype=np.float64),
        }
        world = SimpleNamespace(
            gripper_close_keepalive_active=lambda side: side == "right",
            gripper_uses_effort=lambda _side: True,
            gripper_qpos_list=lambda side: capture_q[side].tolist(),
            gripper_qvel_list=lambda _side: [0.0, 0.0],
        )

        overrides, report = official_tools._move_point_gripper_action_overrides(
            SimpleNamespace(world=world),
            {"gripper_q": capture_q},
            enable_open_recovery=True,
        )

        self.assertNotIn("gripper_effort_right", overrides)
        self.assertEqual(report["right"]["mode"], "verified_close_keepalive")

    @unittest.skipUnless(
        ACTION_SLICES["gripper_left"].stop
        - ACTION_SLICES["gripper_left"].start
        == 2,
        "requires the custom effort-gripper action contract",
    )
    def test_continuous_execution_recovers_transient_gripper_drift(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            observations,
        )
        capture_state["gripper_q"]["right"] = np.asarray(
            [0.0475, 0.0035], dtype=np.float64
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.03
        if ARM_DOF == 8:
            target[7] = 0.0
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.12,
            },
            "waypoints": [target.astype(float).tolist()],
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
            allow_gripper_recovery=True,
        )

        first = np.asarray(next(generator), dtype=np.float32)
        proprio = adapter.proprio_vector().copy()
        partial_q = q_start.copy()
        partial_q[0] += 0.01
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = partial_q
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0405, 0.0035]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        recovery_one = np.asarray(next(generator), dtype=np.float32)
        np.testing.assert_allclose(
            recovery_one[ACTION_SLICES["arm_left"]],
            partial_q,
            atol=1e-6,
        )
        self.assertEqual(
            float(recovery_one[ACTION_SLICES["gripper_right"]][0]),
            official_tools.MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N,
        )
        self.assertLessEqual(
            abs(float(recovery_one[ACTION_SLICES["gripper_right"]][1])),
            official_tools.MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N,
        )
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = partial_q
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0448, 0.0035]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        recovery_two = np.asarray(next(generator), dtype=np.float32)
        np.testing.assert_allclose(
            recovery_two[ACTION_SLICES["arm_left"]],
            partial_q,
            atol=1e-6,
        )
        adapter.update({"robot_r1::proprio": proprio.copy()})

        resumed = np.asarray(next(generator), dtype=np.float32)
        np.testing.assert_allclose(
            resumed[ACTION_SLICES["arm_left"]],
            target,
            atol=1e-6,
        )
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = target
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        motion = stopped.exception.value

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["waypoints_completed"], 1)
        self.assertEqual(motion["steps_executed"], 4)
        self.assertEqual(motion["gripper_recovery_steps"], 2)
        self.assertLessEqual(
            motion["final_gripper_drift_inf_m"],
            official_tools.MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M,
        )
        self.assertEqual(
            [item["event"] for item in motion["gripper_recovery_events"]],
            ["hold_started", "recovered"],
        )
        self.assertTrue(np.all(np.isfinite(first)))
        if ARM_DOF == 8:
            self.assertTrue(
                all(
                    float(action[ACTION_SLICES["arm_left"]][7]) == 0.0
                    for action in (first, recovery_one, recovery_two, resumed)
                )
            )
        initial_inactive = np.asarray(
            capture_state["arm_q"]["right"], dtype=np.float64
        )
        for action in (first, recovery_one, recovery_two, resumed):
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]],
                initial_inactive,
                atol=1e-6,
            )

    @unittest.skipUnless(
        ACTION_SLICES["gripper_left"].stop
        - ACTION_SLICES["gripper_left"].start
        == 2,
        "requires the custom effort-gripper action contract",
    )
    def test_move_tracked_gripper_disturbance_never_stops_arm_path(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        capture_state["gripper_q"]["right"] = np.asarray(
            [0.0475, 0.0035], dtype=np.float64
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        targets = []
        for offset in (0.03, 0.06):
            target = q_start.copy()
            target[0] += offset
            if ARM_DOF == 8:
                target[7] = 0.0
            targets.append(target)
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory={
                "active_arm": "left",
                "start_state": {"episode_id": world.episode_id()},
                "tracking": {
                    "execution_mode": "continuous_waypoint_stream",
                    "waypoint_tolerance_rad": 0.018,
                    "max_tracking_error_rad": 0.12,
                },
                "waypoints": [target.tolist() for target in targets],
            },
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
            allow_gripper_recovery=True,
            continue_during_gripper_disturbance=True,
        )

        first = np.asarray(next(generator), dtype=np.float32)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = targets[0]
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.034, 0.0035]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        second = np.asarray(next(generator), dtype=np.float32)
        np.testing.assert_allclose(
            second[ACTION_SLICES["arm_left"]], targets[1], atol=1e-6
        )
        self.assertEqual(
            float(second[ACTION_SLICES["gripper_right"]][0]),
            official_tools.MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N,
        )
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = targets[1]
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.034, 0.0035]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        motion = stopped.exception.value

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["waypoints_completed"], 2)
        self.assertEqual(motion["steps_executed"], 2)
        self.assertTrue(motion["gripper_recovery_concurrent_with_arm"])
        self.assertGreater(
            motion["max_gripper_drift_inf_m"],
            official_tools.MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M,
        )
        self.assertEqual(
            motion["gripper_recovery_events"][0]["event"],
            "concurrent_recovery_started",
        )
        self.assertTrue(
            all(
                event.get("arm_trajectory_continued")
                for event in motion["gripper_recovery_events"]
            )
        )
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(np.isfinite(second)))

    def test_move_tracked_recovers_external_active_arm_displacement(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.03
        if ARM_DOF == 8:
            target[7] = 0.0
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory={
                "active_arm": "left",
                "start_state": {"episode_id": world.episode_id()},
                "tracking": {
                    "execution_mode": "continuous_waypoint_stream",
                    "waypoint_tolerance_rad": 0.018,
                    "max_tracking_error_rad": 0.12,
                },
                "waypoints": [target.tolist()],
            },
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
            recover_active_arm_disturbance=True,
        )

        first = np.asarray(next(generator), dtype=np.float32)
        disturbed_q = target.copy()
        disturbed_q[1] += 0.16
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = disturbed_q
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        retry = np.asarray(next(generator), dtype=np.float32)
        np.testing.assert_allclose(
            retry[ACTION_SLICES["arm_left"]], target, atol=1e-6
        )
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = target
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        motion = stopped.exception.value

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["steps_executed"], 2)
        self.assertEqual(motion["waypoints_completed"], 1)
        self.assertTrue(motion["active_arm_disturbance_recovery_enabled"])
        self.assertEqual(
            motion["tracking_recovery_events"][0]["reason"],
            "external_disturbance_target_reissued",
        )
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(np.isfinite(retry)))

    @unittest.skipUnless(
        ACTION_SLICES["gripper_left"].stop
        - ACTION_SLICES["gripper_left"].start
        == 2,
        "requires the custom effort-gripper action contract",
    )
    def test_persistent_gripper_drift_fails_after_bounded_recovery(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx,
            observations,
        )
        capture_state["gripper_q"]["right"] = np.asarray(
            [0.0475, 0.0035], dtype=np.float64
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.03
        if ARM_DOF == 8:
            target[7] = 0.0
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.12,
            },
            "waypoints": [target.astype(float).tolist()],
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=(
                official_tools.MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS
                + 5
            ),
            operation_name="move_tracked_point",
            allow_gripper_recovery=True,
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float32)
            except StopIteration as stopped:
                motion = stopped.value
                break
            actions.append(action)
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[
                ACTION_SLICES["arm_left"]
            ]
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0405, 0.0035]
            proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertFalse(motion["ok"])
        self.assertEqual(motion["reason"], "gripper_drift")
        self.assertEqual(
            motion["gripper_recovery_steps"],
            official_tools.MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS,
        )
        self.assertEqual(
            motion["gripper_recovery_events"][-1]["event"],
            "recovery_exhausted",
        )
        self.assertEqual(motion["waypoints_completed"], 0)
        self.assertEqual(
            len(actions),
            1 + official_tools.MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS,
        )

    @unittest.skipUnless(
        ACTION_SLICES["gripper_left"].stop
        - ACTION_SLICES["gripper_left"].start
        == 2,
        "requires the custom effort-gripper action contract",
    )
    def test_gripper_hard_envelope_fails_without_recovery_action(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        capture_state["gripper_q"]["right"] = np.asarray(
            [0.0475, 0.0035], dtype=np.float64
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.03
        if ARM_DOF == 8:
            target[7] = 0.0
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory={
                "active_arm": "left",
                "start_state": {"episode_id": world.episode_id()},
                "tracking": {
                    "execution_mode": "continuous_waypoint_stream",
                    "waypoint_tolerance_rad": 0.018,
                    "max_tracking_error_rad": 0.12,
                },
                "waypoints": [target.astype(float).tolist()],
            },
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
            allow_gripper_recovery=True,
        )

        next(generator)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = target
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0364, 0.0035]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        motion = stopped.exception.value

        self.assertFalse(motion["ok"])
        self.assertEqual(motion["reason"], "gripper_drift")
        self.assertEqual(motion["steps_executed"], 1)
        self.assertEqual(motion["gripper_recovery_steps"], 0)
        self.assertEqual(
            motion["gripper_recovery_events"][-1]["event"],
            "hard_limit_exceeded",
        )

    @unittest.skipUnless(
        ACTION_SLICES["gripper_left"].stop
        - ACTION_SLICES["gripper_left"].start
        == 2,
        "requires the custom effort-gripper action contract",
    )
    def test_intermediate_gripper_soft_drift_keeps_existing_servo_semantics(self) -> None:
        adapter, world = self._ready_world()
        ctx, _result = self._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = official_tools._move_tracked_capture_state(
            ctx, observations
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = q_start.copy()
        target[0] += 0.03
        if ARM_DOF == 8:
            target[7] = 0.0
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory={
                "active_arm": "left",
                "start_state": {"episode_id": world.episode_id()},
                "tracking": {
                    "execution_mode": "continuous_waypoint_stream",
                    "waypoint_tolerance_rad": 0.018,
                    "max_tracking_error_rad": 0.12,
                },
                "waypoints": [target.astype(float).tolist()],
            },
            capture_state=capture_state,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=30.0,
            max_steps=20,
            operation_name="move_tracked_point",
            allow_gripper_recovery=True,
        )

        next(generator)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = target
        proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.013, 0.02]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        motion = stopped.exception.value

        self.assertTrue(motion["ok"], motion)
        self.assertEqual(motion["steps_executed"], 1)
        self.assertEqual(motion["gripper_recovery_steps"], 0)
        self.assertEqual(motion["gripper_recovery_events"], [])

    def test_global_time_parameterization_has_no_internal_stops(self) -> None:
        q0 = np.zeros(ARM_DOF, dtype=np.float64)
        q1 = q0.copy()
        q2 = q0.copy()
        q1[0] = 0.12
        q1[1] = -0.04
        q2[0] = 0.24
        q2[1] = -0.08
        waypoints, fractions, report = (
            tracked_motion._global_time_parameterize_knots(
                [(0.0, q0), (0.5, q1), (1.0, q2)],
                max_joint_step_rad=0.012,
                max_waypoints=100,
                check_cancelled=lambda: None,
            )
        )
        path = np.vstack([q0, *waypoints])
        increments = np.diff(path[:, 0])
        self.assertTrue(np.all(increments > 0.0))
        self.assertTrue(np.all(np.diff(fractions) > 0.0))
        self.assertEqual(report["internal_stop_count"], 0)
        self.assertGreater(report["minimum_path_progress_increment_rad"], 0.0)
        self.assertLessEqual(report["actual_max_joint_step_rad"], 0.012)
        if ARM_DOF == 8:
            np.testing.assert_allclose(path[:, 7], 0.0, atol=0.0)

    def test_start_settle_brake_target_is_latched(self) -> None:
        (
            adapter,
            world,
            _manager,
            ctx,
            capture_state,
            _source_points,
        ) = self._start_gate_fixture()
        brake_targets: dict[str, np.ndarray] = {}
        first_validation = official_tools._move_point_start_state_validation(
            ctx,
            capture_state,
        )[1]
        first_action, first_mode = official_tools._move_tracked_start_settle_action(
            ctx,
            capture_state,
            first_validation,
            brake_targets=brake_targets,
        )
        self.assertEqual(first_mode, "measured_qpos_brake")
        first_target = first_action[ACTION_SLICES["arm_left"]].copy()

        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]][0] += 0.02
        adapter.update({"robot_r1::proprio": proprio})
        second_validation = official_tools._move_point_start_state_validation(
            ctx,
            capture_state,
        )[1]
        second_action, second_mode = official_tools._move_tracked_start_settle_action(
            ctx,
            capture_state,
            second_validation,
            brake_targets=brake_targets,
        )
        self.assertEqual(second_mode, "measured_qpos_brake")
        np.testing.assert_allclose(
            second_action[ACTION_SLICES["arm_left"]],
            first_target,
            atol=1e-7,
        )

    def test_start_gate_waits_without_stale_coordinates_and_recovers(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        manager.loss_after_sequence = int(adapter.status()["sequence"]) + 1
        manager.loss_until_sequence = manager.loss_after_sequence + 3
        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda _action_index: 0.0,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reason"], "stable_start_verified")
        self.assertEqual(report["tracker_wait_steps"], 4)
        self.assertEqual(
            report["tracker_wait_reason_counts"],
            {"tracked_point_unavailable": 4},
        )
        self.assertEqual(report["max_consecutive_tracker_wait_steps"], 4)
        self.assertFalse(report["stale_tracker_coordinates_used_while_waiting"])
        self.assertGreaterEqual(len(actions), 6)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())

    def test_start_gate_fails_closed_when_tracker_never_recovers(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        manager.loss_after_sequence = int(adapter.status()["sequence"]) + 1
        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda _action_index: 0.0,
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "tracked_start_not_synchronized")
        self.assertEqual(
            report["tracker_wait_steps"],
            official_tools.MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS,
        )
        self.assertEqual(
            len(actions),
            official_tools.MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS,
        )
        self.assertFalse(report["stale_tracker_coordinates_used_while_waiting"])

    def test_start_gate_reports_per_point_retirement_reason(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        manager.anchors.pop("known")
        gate = official_tools._move_tracked_wait_for_stable_start(
            ctx,
            capture_state=capture_state,
            manager=manager,
            point_names=["known"],
            source_points=source_points,
            episode_id=world.episode_id(),
            session_id=manager.session_id,
            image_id=manager.image_id,
            pos_tol_m=0.012,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        actions, report = self._drive_start_gate(
            adapter,
            gate,
            lambda _action_index: 0.0,
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "tracked_point_unavailable")
        self.assertEqual(actions, [])
        self.assertEqual(
            report["last_tracker_snapshot"]["unavailable"],
            {"known": "point_not_registered_or_retired"},
        )

    def test_start_gate_recovers_from_tracker_sequence_lag(self) -> None:
        (
            adapter,
            world,
            manager,
            ctx,
            capture_state,
            source_points,
        ) = self._start_gate_fixture()
        real_snapshot = manager.observed_active_points_snapshot
        lagged_calls = 3

        def snapshot_with_temporary_lag(names, *, episode_id):
            nonlocal lagged_calls
            snapshot = real_snapshot(names, episode_id=episode_id)
            if lagged_calls > 0 and bool(snapshot.get("ok")):
                lagged_calls -= 1
                snapshot["observation_sequence"] = int(
                    snapshot["observation_sequence"]
                ) - 1
            return snapshot

        with mock.patch.object(
            manager,
            "observed_active_points_snapshot",
            side_effect=snapshot_with_temporary_lag,
        ):
            gate = official_tools._move_tracked_wait_for_stable_start(
                ctx,
                capture_state=capture_state,
                manager=manager,
                point_names=["known"],
                source_points=source_points,
                episode_id=world.episode_id(),
                session_id=manager.session_id,
                image_id=manager.image_id,
                pos_tol_m=0.012,
                deadline_monotonic=time.monotonic() + 30.0,
            )
            actions, report = self._drive_start_gate(
                adapter,
                gate,
                lambda _action_index: 0.0,
            )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["tracker_wait_steps"], 3)
        self.assertEqual(
            report["tracker_wait_reason_counts"],
            {"tracker_observation_mismatch": 3},
        )
        self.assertFalse(report["stale_tracker_coordinates_used_while_waiting"])
        self.assertGreaterEqual(len(actions), 4)

    def test_vector_alignment_is_a_proper_rotation_for_random_directions(self) -> None:
        rng = np.random.default_rng(20260827)
        cases = [
            (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        ]
        cases.extend((rng.normal(size=3), rng.normal(size=3)) for _ in range(200))
        for source, target in cases:
            source_unit = source / np.linalg.norm(source)
            target_unit = target / np.linalg.norm(target)
            rotation = tracked_motion._align_vectors(source, target)
            np.testing.assert_allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-10,
            )
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)
            np.testing.assert_allclose(
                rotation @ source_unit,
                target_unit,
                atol=1e-9,
            )

    def test_real_stale_capture_sample_has_reachable_direction_endpoint(self) -> None:
        trunk = [0.4500338733, -0.3998764455, -0.00007977, -0.00000570]
        left_q = [
            -0.00001799,
            -0.00000050,
            0.00000576,
            -2.09440088,
            -0.00000057,
            -1.04718757,
            -0.00000037,
            0.0,
        ]
        right_q = [
            -0.93076229,
            0.16929936,
            0.54524904,
            -1.58869934,
            1.33912361,
            0.66437531,
            0.93078268,
            0.0,
        ]
        left_q = left_q[:ARM_DOF]
        right_q = right_q[:ARM_DOF]
        state = local_robot_state(
            trunk_q=trunk,
            arm_left_q=left_q,
            arm_right_q=right_q,
            gripper_left_q=[0.02, 0.02],
            gripper_right_q=[0.02, 0.02],
        )
        # The input order is part of the affine constraint semantics.  The
        # nearest point is only a reported geometric anchor, never a reason to
        # swap the submitted head/tail pairing.
        source = np.asarray(
            [
                [0.7112404284, 0.0392089001, 1.4747762011],
                [0.7299347397, 0.0675236846, 1.5743870312],
            ],
            dtype=np.float64,
        )
        targets = [
            {"name": "bing", "target_xyz_m": ["x", "y", "zb"]},
            {"name": "ren", "target_xyz_m": ["x", "y", "za"]},
        ]

        endpoint = plan_endpoint(
            state=state,
            arm="right",
            q_start=right_q,
            source_points_robot_base_m=source,
            target_points=validate_submission(
                "move_tracked_point", {"points": targets}
            )["points"],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
        )

        resolved = endpoint["resolved_points_robot_base_m"]
        self.assertTrue(endpoint["constraints"]["ok"], endpoint)
        self.assertEqual(
            endpoint["selected_candidate"]["mode"],
            "joint_space_affine_constraint_ik",
        )
        self.assertLessEqual(abs(float(resolved[0, 0] - resolved[1, 0])), 0.012)
        self.assertLessEqual(abs(float(resolved[0, 1] - resolved[1, 1])), 0.012)
        if ARM_DOF == 8:
            self.assertAlmostEqual(float(endpoint["q_final"][7]), 0.0, places=12)

    def test_frozen_planning_yields_hold_before_solver_completes(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world,
            adapter,
            {"known": position},
        )
        entered = threading.Event()
        release = threading.Event()
        real_planner = official_tools._move_tracked_plan_frozen

        def gated_planner(**kwargs):
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test planner gate timed out")
            return real_planner(**kwargs)

        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_move_tracked_plan_frozen",
            side_effect=gated_planner,
        ):
            generator = build_registry(adapter)["move_tracked_point"].fn(
                ctx,
                points=[
                    {
                        "name": "known",
                        "target_xyz_m": np.asarray(position).astype(float).tolist(),
                    }
                ],
                max_steps=120,
            )
            started = time.monotonic()
            first_action = np.asarray(next(generator), dtype=np.float32)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(entered.wait(timeout=0.5))
            self.assertFalse(result)
            self.assertEqual(first_action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(first_action).all())
            adapter.update(
                {"robot_r1::proprio": adapter.proprio_vector().copy()}
            )
            release.set()
            remaining_actions = self._drive(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["planner"]["planning_hold_steps"], 1)
        self.assertGreaterEqual(len(remaining_actions), 1)

    def test_cartesian_knots_accept_solution_within_public_tolerance(self) -> None:
        _adapter, world = self._ready_world()
        state, _pose = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.20
        real_solve = tracked_motion.solve_pose_target

        def strict_solver_miss(*args, **kwargs):
            q, report = real_solve(*args, **kwargs)
            return q, {
                **report,
                "ok": False,
                "pos_err_m": min(float(report["pos_err_m"]), 0.0038),
                "ori_err_deg": 4.49,
            }

        with mock.patch.object(
            tracked_motion,
            "solve_pose_target",
            side_effect=strict_solver_miss,
        ):
            path = tracked_motion.plan_cartesian_trajectory(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_final,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_joint_step_rad=0.10,
                max_waypoints=180,
            )

        accepted = [
            report
            for report in path["ik_reports"]
            if report.get("accepted_by_public_tolerance")
        ]
        self.assertTrue(accepted, path)
        self.assertTrue(all(report["ori_err_deg"] == 4.49 for report in accepted))

    def test_cartesian_path_honors_cancellation(self) -> None:
        _adapter, world = self._ready_world()
        state, _pose = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        callback_count = 0

        def cancel_requested() -> bool:
            nonlocal callback_count
            callback_count += 1
            return callback_count >= 3

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            tracked_motion.plan_cartesian_trajectory(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_start,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_joint_step_rad=0.012,
                max_waypoints=100,
                cancel_requested=cancel_requested,
            )
        self.assertGreaterEqual(callback_count, 3)

    def test_cartesian_path_rejects_flat_tracked_anchor_buffer(self) -> None:
        _adapter, world = self._ready_world()
        state, _pose = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "nested N x 3"):
            tracked_motion.plan_cartesian_trajectory(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_start,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_joint_step_rad=0.10,
                max_waypoints=8,
                tracked_anchors_eef=[0.0, 0.0, 0.0],
            )

    def test_runtime_point_bundle_validator_rejects_malformed_n_point_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "nested N x 3"):
            official_tools._move_tracked_strict_point_matrix(
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                expected_count=2,
                label="bundle",
            )
        with self.assertRaisesRegex(ValueError, "shape \(2, 3\)"):
            official_tools._move_tracked_strict_point_matrix(
                [[0.0, 0.0, 0.0]],
                expected_count=2,
                label="bundle",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            official_tools._move_tracked_strict_point_matrix(
                [[0.0, 0.0, float("nan")], [0.1, 0.0, 0.0]],
                expected_count=2,
                label="bundle",
            )

    def test_tool_tries_next_endpoint_when_best_path_is_infeasible(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        target = np.asarray(position) + [0.008, 0.0, 0.0]
        real_endpoint = official_tools._plan_tracked_endpoint
        real_path = official_tools._plan_tracked_cartesian_trajectory

        def two_ranked_endpoints(**kwargs):
            result = real_endpoint(**kwargs)
            first = dict(result["_ranked_endpoint_candidates"][0])
            first["selected_candidate"] = {
                **first["selected_candidate"],
                "endpoint_score_rank": 0,
            }
            second = dict(first)
            second["selected_candidate"] = {
                **first["selected_candidate"],
                "endpoint_score_rank": 1,
            }
            result["_ranked_endpoint_candidates"] = [first, second]
            result["feasible_candidate_count"] = 2
            return result

        path_calls = 0
        joint_fallback_calls = 0

        def first_path_fails(**kwargs):
            nonlocal path_calls
            path_calls += 1
            if path_calls == 1:
                raise ValueError("synthetic best-candidate branch failure")
            return real_path(**kwargs)

        def first_joint_fallback_fails(**kwargs):
            nonlocal joint_fallback_calls
            joint_fallback_calls += 1
            raise ValueError("synthetic best-candidate joint fallback failure")

        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            side_effect=two_ranked_endpoints,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=first_path_fails,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_joint_space_trajectory",
            side_effect=first_joint_fallback_fails,
        ):
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target.tolist()}],
                    max_steps=180,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(path_calls, 2)
        self.assertEqual(joint_fallback_calls, 1)
        self.assertEqual(result["planner"]["endpoint_path_candidate_rank"], 1)
        self.assertEqual(result["planner"]["endpoint_path_candidates_tested"], 2)
        self.assertIn(
            "synthetic best-candidate branch failure",
            result["planner"]["endpoint_path_candidate_failures"][0]["error"],
        )

    def test_tool_prefers_best_endpoint_joint_fallback_before_next_endpoint(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        target = np.asarray(position) + [0.008, 0.0, 0.0]
        real_endpoint = official_tools._plan_tracked_endpoint

        def two_ranked_endpoints(**kwargs):
            result = real_endpoint(**kwargs)
            first = dict(result["_ranked_endpoint_candidates"][0])
            first["selected_candidate"] = {
                **first["selected_candidate"],
                "endpoint_score_rank": 0,
            }
            second = dict(first)
            second["selected_candidate"] = {
                **first["selected_candidate"],
                "endpoint_score_rank": 1,
            }
            result["_ranked_endpoint_candidates"] = [first, second]
            result["feasible_candidate_count"] = 2
            return result

        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            side_effect=two_ranked_endpoints,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=ValueError("synthetic Cartesian path failure"),
        ) as cartesian_mock:
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target.tolist()}],
                    max_steps=180,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(cartesian_mock.call_count, 1)
        self.assertEqual(result["planner"]["endpoint_path_candidate_rank"], 0)
        self.assertEqual(result["planner"]["endpoint_path_candidates_tested"], 1)
        self.assertEqual(result["planner"]["path_mode"], "joint_space_fallback")
        self.assertIn(
            "synthetic Cartesian path failure",
            result["planner"]["endpoint_path_candidate_failures"][0]["error"],
        )

    def test_cartesian_planner_skips_finer_ik_after_severe_fk_deviation(self) -> None:
        _adapter, world = self._ready_world()
        state, _pose = self._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        severe = tracked_motion._CartesianPathDeviation(
            "joint interpolation leaves the verified Cartesian path by 0.1200m",
            measured=0.12,
            limit=0.03,
        )
        with mock.patch.object(
            tracked_motion,
            "_plan_cartesian_trajectory_once",
            side_effect=severe,
        ) as once_mock, self.assertRaisesRegex(
            ValueError,
            "after uniform sampling attempts",
        ):
            tracked_motion.plan_cartesian_trajectory(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_start,
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_joint_step_rad=0.018,
                max_waypoints=100,
            )
        self.assertEqual(once_mock.call_count, 1)

    def test_full_tool_moves_numeric_point_and_emits_legal_actions(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"axe_tip": position})
        world._official_tracked_object_distances = manager
        target = np.asarray(position) + np.array([0.008, 0.0, 0.0])
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "axe_tip",
                            "target_xyz_m": target.astype(float).tolist(),
                        }
                    ],
                    max_steps=180,
                ),
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["arm"], "left")
        self.assertTrue(result["final_constraints"]["ok"])
        self.assertTrue(result["final_anchor_ok"])
        self.assertEqual(
            result["trajectory_schema_version"],
            official_tools.MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertGreaterEqual(len(actions), 3)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                for side in ("left", "right"):
                    self.assertEqual(
                        float(action[ACTION_SLICES[f"arm_{side}"]][7]), 0.0
                    )

    def test_lite_pose_overlap_uses_exact_voxel_thresholds_and_short_circuit(
        self,
    ) -> None:
        class FakeOccupancy:
            voxel_m = 0.003
            metadata = {"occupancy_method": "synthetic_exact_counts"}

            def __init__(self) -> None:
                self.calls = []

            def counts_for_poses(self, poses, offsets):
                self.calls.append((list(poses), np.asarray(offsets).copy()))
                if len(self.calls) == 1:
                    return np.asarray([5, 6, 0], dtype=np.int32), {
                        "device": "synthetic",
                        "pose_count": 3,
                    }
                return np.asarray([59, 60], dtype=np.int32), {
                    "device": "synthetic",
                    "pose_count": 2,
                }

        occupancy = FakeOccupancy()
        geometry = {
            "gripper_voxels": np.asarray([[0.0, 0.0, 0.0]]),
            "components": {
                "palm": np.asarray([[0.0, 0.0, 0.0]]),
                "finger_positive_y": np.zeros((0, 3)),
                "finger_negative_y": np.zeros((0, 3)),
                "camera": np.zeros((0, 3)),
            },
        }
        inflated = np.asarray(
            [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]],
            dtype=np.float64,
        )
        poses = [
            {
                "eef_pos": [0.4 + 0.01 * index, 0.0, 0.7],
                "quat": [0.0, 0.0, 0.0, 1.0],
            }
            for index in range(3)
        ]
        occupancy_builder = mock.Mock(return_value=occupancy)
        with mock.patch.object(
            rgbd_lite,
            "gripper_geometry",
            return_value=geometry,
        ), mock.patch.object(
            rgbd_lite.production,
            "inflated_gripper_voxels_eef",
            return_value=(inflated, {}, {"policy": "test_exact_policy"}),
        ), mock.patch.object(
            rgbd_lite,
            "_build_local_scene_occupancy_lite",
            occupancy_builder,
        ):
            report = rgbd_lite.evaluate_pose_overlap_volumes_lite(
                session={"session_id": "s", "image_id": "i"},
                camera_pos=[0.0, 0.0, 0.0],
                camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                focal_length=1.0,
                horizontal_aperture=1.0,
                poses=poses,
                prepared_scene=(object(), None, None, {"geometry_version": "test"}),
            )

        self.assertEqual(len(occupancy.calls), 2)
        self.assertEqual(len(occupancy.calls[0][0]), 3)
        self.assertEqual(len(occupancy.calls[1][0]), 2)
        self.assertEqual(report["passing_pose_indices"], [0])
        self.assertAlmostEqual(report["poses"][0]["overlap_vol_cm3"], 0.135)
        self.assertAlmostEqual(
            report["poses"][0]["inflated_overlap_vol_cm3"], 1.593
        )
        self.assertTrue(report["poses"][0]["ok"])
        self.assertAlmostEqual(report["poses"][1]["overlap_vol_cm3"], 0.162)
        self.assertFalse(report["poses"][1]["ok"])
        self.assertIsNone(report["poses"][1]["inflated_overlap_vol_cm3"])
        self.assertTrue(report["poses"][1]["inflated_query_skipped"])
        self.assertAlmostEqual(
            report["poses"][2]["inflated_overlap_vol_cm3"], 1.62
        )
        self.assertFalse(report["poses"][2]["ok"])
        self.assertFalse(report["grasp_opening_volume_queried"])
        self.assertTrue(report["filter_result_equivalent_to_full_conjunction"])
        self.assertIs(
            occupancy_builder.call_args.kwargs["validate_hit"],
            False,
        )
        self.assertEqual(
            report["occupancy_center_semantics"],
            "eef_pose_envelope_center",
        )
        self.assertIs(report["occupancy_center_surface_validation"], False)

    def test_lite_occupancy_hit_validation_is_propagated_and_cached_separately(
        self,
    ) -> None:
        common_kwargs = {
            "hit": np.asarray([0.4, 0.0, 0.7]),
            "query_offsets": [np.asarray([[0.0, 0.0, 0.0]])],
            "axial_z_m": np.asarray([0.0]),
            "anchor_radius_m": 0.0,
            "voxel_m": 0.003,
        }
        cache_mesh = SimpleNamespace(
            _official_v2_lite_scene_key="synthetic-scene"
        )
        key_with_validation = rgbd_lite._lite_occupancy_cache_key(
            cache_mesh,
            {**common_kwargs, "validate_hit": True},
        )
        key_without_validation = rgbd_lite._lite_occupancy_cache_key(
            cache_mesh,
            {**common_kwargs, "validate_hit": False},
        )
        self.assertNotEqual(key_with_validation, key_without_validation)

        built = SimpleNamespace(
            origin=np.zeros(3, dtype=np.float64),
            occupancy=np.zeros((1, 1, 1), dtype=bool),
            voxel_m=0.003,
            metadata={"occupancy_method": "synthetic"},
            device_dense=None,
            device_origin=None,
            device_resources=None,
        )
        projective_scene = rgbd_lite.ProjectiveDepthScene(
            depth=np.ones((2, 2), dtype=np.float32),
            camera_pos=np.zeros(3, dtype=np.float64),
            camera_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            focal_length=1.0,
            horizontal_aperture=1.0,
            scene_key="projective-scene",
            metadata={},
        )
        with mock.patch.dict(
            os.environ,
            {"OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "0"},
        ), mock.patch.object(
            rgbd_lite,
            "build_projective_local_occupancy",
            return_value=built,
        ) as projective_builder:
            rgbd_lite._build_local_scene_occupancy_lite(
                projective_scene,
                **common_kwargs,
                validate_hit=False,
            )
        self.assertIs(projective_builder.call_args.kwargs["validate_hit"], False)

        with mock.patch.dict(
            os.environ,
            {
                "OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "0",
                rgbd_lite.LITE_GEOMETRY_BACKEND_ENV: (
                    rgbd_lite.LITE_GEOMETRY_BACKEND_WARP
                ),
            },
        ), mock.patch.object(
            rgbd_lite,
            "build_warp_local_occupancy",
            return_value=built,
        ) as warp_builder:
            rgbd_lite._build_local_scene_occupancy_lite(
                cache_mesh,
                **common_kwargs,
                validate_hit=False,
            )
        self.assertIs(warp_builder.call_args.kwargs["validate_hit"], False)

    def test_plan_overlap_filter_rejects_unsafe_candidate_before_path(self) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)

        def candidate(index: int) -> dict:
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            return {
                "q_start": q_start.copy(),
                "q_final": q_final,
                "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
                "final_eef_position_m": np.asarray(
                    [0.41 + 0.01 * index, 0.0, 0.70]
                ),
                "final_eef_quaternion_xyzw": np.asarray(
                    [0.0, 0.0, 0.0, 1.0]
                ),
                "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
                "selected_is_precise": True,
                "selected_candidate": {
                    "precise": True,
                    "motion_score": float(index),
                    "accuracy_cost_mm_plus_deg": 0.0,
                    "position_error_mm": 0.0,
                    "selection_orientation_error_deg": 0.0,
                    "endpoint_score_rank": index,
                },
            }

        first = candidate(0)
        second = candidate(1)

        def overlap_evaluation(*, poses, **_kwargs):
            report = self._safe_overlap_evaluation(poses=poses)
            report["poses"][0].update(
                {
                    "overlap_vox": 6,
                    "overlap_vol_cm3": 0.162,
                    "original_overlap_ok": False,
                    "inflated_overlap_vox": None,
                    "inflated_overlap_vol_cm3": None,
                    "inflated_overlap_ok": False,
                    "inflated_query_skipped": True,
                    "inflated_query_skipped_reason": "original_overlap_failed",
                    "ok": False,
                    "reason": "original_overlap_exceeded",
                }
            )
            report["passing_pose_count"] = 1
            report["rejected_pose_count"] = 1
            report["passing_pose_indices"] = [1]
            return report

        path_calls = []

        def path_planner(**kwargs):
            path_calls.append(np.asarray(kwargs["q_final"]).copy())
            return {
                "waypoints": [np.asarray(kwargs["q_final"]).astype(float).tolist()],
                "same_branch_checked": True,
            }

        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={
                **first,
                "_ranked_endpoint_candidates": [first, second],
            },
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=path_planner,
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.42, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertTrue(planned["ok"], planned)
        self.assertIs(planned["endpoint"].get("q_final"), second["q_final"])
        self.assertEqual(len(path_calls), 1)
        np.testing.assert_allclose(path_calls[0], second["q_final"])
        audit = planned["path_plan"]["frozen_rgbd_overlap_filter"]
        self.assertTrue(audit["ok"], audit)
        self.assertEqual(audit["candidate_count_evaluated"], 2)
        self.assertEqual(audit["candidate_count_passing"], 1)
        self.assertEqual(audit["candidate_count_rejected"], 1)
        self.assertEqual(
            planned["path_failures"][0]["path_mode"],
            "frozen_rgbd_overlap_filter",
        )

    def test_shared_line_pose_family_preserves_global_best_error(self) -> None:
        source = np.asarray(
            [
                [-0.05, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, -0.05, 0.004],
                [0.0, 0.05, 0.004],
            ],
            dtype=np.float64,
        )
        relations = [
            {
                "type": "line_through_point",
                "point_names": ["a", "b"],
                "target_point_robot_base_m": [0.2, -0.1, 0.3],
            },
            {
                "type": "line_through_point",
                "point_names": ["c", "d"],
                "target_point_robot_base_m": [0.2, -0.1, 0.3],
            },
        ]
        family = tracked_motion.shared_line_target_pose_family(
            source_points_robot_base_m=source,
            point_names=["a", "b", "c", "d"],
            relations=relations,
            start_eef_position_m=[0.0, 0.0, 0.1],
            start_eef_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            base_target_eef_quaternions_xyzw=[[0.0, 0.0, 0.0, 1.0]],
        )

        self.assertTrue(family["ok"], family)
        self.assertEqual(family["pose_count"], 125)
        np.testing.assert_allclose(
            family["source_pivot_robot_base_m"], [0.0, 0.0, 0.002]
        )
        self.assertAlmostEqual(family["invariant_max_line_error_m"], 0.002)
        self.assertIn(
            [0.0, 0.0, -20.0],
            [pose["local_euler_xyz_deg"] for pose in family["poses"]],
        )
        target = np.asarray([0.2, -0.1, 0.3])
        for pose in family["poses"][::31]:
            target_eef_position = np.asarray(
                pose["final_eef_position_m"], dtype=np.float64
            )
            target_eef_rotation = quat_to_mat_xyzw(
                pose["final_eef_quaternion_xyzw"]
            )
            transformed = target_eef_position + (
                target_eef_rotation @ (source - [0.0, 0.0, 0.1]).T
            ).T
            errors = []
            for first, second in ((0, 1), (2, 3)):
                direction = transformed[second] - transformed[first]
                direction /= np.linalg.norm(direction)
                delta = target - transformed[first]
                errors.append(
                    np.linalg.norm(delta - float(delta @ direction) * direction)
                )
            self.assertAlmostEqual(max(errors), 0.002, places=10)

    def test_plan_endpoint_accepts_prefiltered_pose_target_frontier(self) -> None:
        _adapter, world = self._ready_world()
        state = local_robot_state(
            trunk_q=world.trunk_qpos(),
            arm_left_q=world.arm_qpos_list("left"),
            arm_right_q=world.arm_qpos_list("right"),
            gripper_left_q=world.gripper_qpos_list("left"),
            gripper_right_q=world.gripper_qpos_list("right"),
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        position, quaternion = eef_pose(state, "left", q_start)
        rotation = quat_to_mat_xyzw(quaternion)
        anchors = np.asarray(
            [
                [-0.04, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.0, -0.04, 0.0],
                [0.0, 0.04, 0.0],
            ]
        )
        source = np.asarray(position) + (rotation @ anchors.T).T
        names = ["a", "b", "c", "d"]
        target_points = [
            {"name": name, "target_xyz_m": ["?", "?", "?"]}
            for name in names
        ]
        relations = [
            {
                "type": "line_through_point",
                "point_names": ["a", "b"],
                "target_point_robot_base_m": np.asarray(position).tolist(),
            },
            {
                "type": "line_through_point",
                "point_names": ["c", "d"],
                "target_point_robot_base_m": np.asarray(position).tolist(),
            },
        ]
        planned = plan_endpoint(
            state=state,
            arm="left",
            q_start=q_start,
            source_points_robot_base_m=source,
            target_points=target_points,
            relations=relations,
            pos_tol_m=0.03,
            ori_tol_deg=20.0,
            eef_pose_targets=[
                {
                    "final_eef_position_m": np.asarray(position).tolist(),
                    "final_eef_quaternion_xyzw": np.asarray(quaternion).tolist(),
                    "pose_family": "unit_test_shared_line_family",
                }
            ],
            pose_targets_only=True,
        )

        self.assertEqual(planned["pose_target_solve_count"], 1)
        self.assertEqual(planned["pose_target_failure_count"], 0)
        self.assertEqual(
            planned["selected_candidate"]["mode"],
            "explicit_eef_pose_target_frontier",
        )
        self.assertTrue(planned["constraints"]["ok"], planned)
        np.testing.assert_allclose(planned["q_final"], q_start, atol=1.0e-6)

    def test_plan_overlap_filter_searches_equivalent_pose_family_before_failing(
        self,
    ) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)
        start_position = np.asarray([0.0, 0.0, 0.10])

        def candidate(
            index: int,
            *,
            position=None,
            quaternion=None,
            precise: bool = True,
        ) -> dict:
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            return {
                "q_start": q_start.copy(),
                "q_final": q_final,
                "start_eef_position_m": start_position.copy(),
                "start_eef_quaternion_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "final_eef_position_m": np.asarray(
                    position if position is not None else [0.20, 0.0, 0.10]
                ),
                "final_eef_quaternion_xyzw": np.asarray(
                    quaternion if quaternion is not None else [0.0, 0.0, 0.0, 1.0]
                ),
                "anchors_eef_m": np.zeros((4, 3), dtype=np.float64),
                "selected_is_precise": bool(precise),
                "selected_candidate": {
                    "precise": bool(precise),
                    "motion_score": float(index),
                    "accuracy_cost_mm_plus_deg": 0.0,
                    "position_error_mm": 0.0,
                    "selection_orientation_error_deg": 0.0,
                    "endpoint_score_rank": index,
                },
            }

        primary = candidate(0, precise=False)
        fallback = candidate(1)
        endpoint_calls = []

        def endpoint_planner(**kwargs):
            endpoint_calls.append(kwargs)
            if not kwargs.get("pose_targets_only"):
                return primary if len(endpoint_calls) == 1 else fallback
            targets = list(kwargs["eef_pose_targets"])
            self.assertEqual(len(targets), 1)
            return candidate(
                2,
                position=targets[0]["final_eef_position_m"],
                quaternion=targets[0]["final_eef_quaternion_xyzw"],
            )

        overlap_call_count = 0

        def overlap_evaluation(*, poses, **_kwargs):
            nonlocal overlap_call_count
            overlap_call_count += 1
            report = self._safe_overlap_evaluation(poses=poses)
            passing = []
            for index, pose_report in enumerate(report["poses"]):
                should_pass = overlap_call_count >= 3 or (
                    overlap_call_count == 2 and index == 0
                )
                if should_pass:
                    passing.append(index)
                    continue
                pose_report.update(
                    {
                        "overlap_vox": 0,
                        "overlap_vol_cm3": 0.0,
                        "original_overlap_ok": True,
                        "inflated_overlap_vox": 61,
                        "inflated_overlap_vol_cm3": 1.647,
                        "inflated_overlap_ok": False,
                        "inflated_query_skipped": False,
                        "inflated_query_skipped_reason": None,
                        "ok": False,
                        "reason": "inflated_overlap_exceeded",
                    }
                )
            report["passing_pose_count"] = len(passing)
            report["rejected_pose_count"] = len(poses) - len(passing)
            report["passing_pose_indices"] = passing
            return report

        source = np.asarray(
            [
                [-0.05, 0.0, 0.10],
                [0.05, 0.0, 0.10],
                [0.0, -0.05, 0.104],
                [0.0, 0.05, 0.104],
            ]
        )
        relations = [
            {
                "type": "line_through_point",
                "point_names": ["a", "b"],
                "target_point_robot_base_m": [0.20, 0.0, 0.10],
            },
            {
                "type": "line_through_point",
                "point_names": ["c", "d"],
                "target_point_robot_base_m": [0.20, 0.0, 0.10],
            },
        ]
        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            side_effect=endpoint_planner,
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "_local_eef_pose",
            return_value=(start_position, np.asarray([0.0, 0.0, 0.0, 1.0])),
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            return_value={
                "waypoints": [candidate(2)["q_final"].astype(float).tolist()],
                "same_branch_checked": True,
            },
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=source,
                target_points=[
                    {"name": name, "target_xyz_m": ["?", "?", "?"]}
                    for name in ("a", "b", "c", "d")
                ],
                relations=relations,
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertTrue(planned["ok"], planned)
        self.assertEqual(len(endpoint_calls), 2)
        self.assertTrue(endpoint_calls[1]["pose_targets_only"])
        self.assertEqual(overlap_call_count, 3)
        self.assertEqual(
            planned["endpoint"]["path_endpoint_preference"],
            "overlap_aware_shared_line_pose_family",
        )
        family = planned["overlap_aware_pose_family"]
        self.assertTrue(family["overlap_prefilter_applied_before_ik"])
        self.assertEqual(family["overlap_prefilter_passing_count"], 1)
        self.assertEqual(family["ik_candidate_count"], 1)
        self.assertEqual(family["post_ik_overlap_passing_count"], 1)
        self.assertEqual(len(family["search_stages"]), 1)
        self.assertEqual(family["search_stages"][0]["search_stage"], "coarse")
        self.assertEqual(family["search_stages"][0]["pose_count"], 27)

    def test_plan_overlap_filter_uses_minimum_overlap_when_all_candidates_collide(
        self,
    ) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)

        def candidate(index: int) -> dict:
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            return {
                "q_start": q_start.copy(),
                "q_final": q_final,
                "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
                "final_eef_position_m": np.asarray(
                    [0.41 + 0.01 * index, 0.0, 0.70]
                ),
                "final_eef_quaternion_xyzw": np.asarray(
                    [0.0, 0.0, 0.0, 1.0]
                ),
                "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
                "selected_is_precise": True,
                "selected_candidate": {
                    "precise": True,
                    "motion_score": float(index),
                    "accuracy_cost_mm_plus_deg": 0.0,
                    "position_error_mm": 0.0,
                    "selection_orientation_error_deg": 0.0,
                    "endpoint_score_rank": index,
                },
            }

        candidates = [candidate(0), candidate(1)]

        def overlap_evaluation(*, poses, **_kwargs):
            report = self._safe_overlap_evaluation(poses=poses)
            for pose_report in report["poses"]:
                pose_report.update(
                    {
                        "overlap_vox": 6,
                        "overlap_vol_cm3": 0.324 if pose_report["pose_index"] == 0 else 0.162,
                        "original_overlap_ok": False,
                        "inflated_overlap_vox": None,
                        "inflated_overlap_vol_cm3": None,
                        "inflated_overlap_ok": False,
                        "inflated_query_skipped": True,
                        "inflated_query_skipped_reason": (
                            "original_overlap_failed"
                        ),
                        "ok": False,
                        "reason": "original_overlap_exceeded",
                    }
                )
            report["passing_pose_count"] = 0
            report["rejected_pose_count"] = len(poses)
            report["passing_pose_indices"] = []
            return report

        path_planner = mock.Mock(
            return_value={
                "waypoints": [candidates[1]["q_final"].tolist()],
                "same_branch_checked": True,
            }
        )
        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={
                **candidates[0],
                "_ranked_endpoint_candidates": candidates,
            },
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            path_planner,
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.42, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertTrue(planned["ok"], planned)
        path_planner.assert_called_once()
        np.testing.assert_allclose(
            path_planner.call_args.kwargs["q_final"], candidates[1]["q_final"]
        )
        audit = planned["path_plan"]["frozen_rgbd_overlap_filter"]
        self.assertFalse(audit["ok"])
        self.assertTrue(audit["fallback_applied"])
        self.assertEqual(audit["selected_pose"]["overlap_vol_cm3"], 0.162)
        self.assertIsNone(audit["selected_pose"]["inflated_overlap_vol_cm3"])
        self.assertTrue(audit["selected_pose"]["inflated_query_skipped"])
        self.assertEqual(audit["fallback_selection"]["accuracy_class"], "precise")
        self.assertEqual(audit["candidate_count_evaluated"], 2)
        self.assertEqual(audit["candidate_count_passing"], 0)
        self.assertEqual(audit["candidate_count_rejected"], 2)
        self.assertEqual(len(planned["path_failures"]), 2)

    def test_plan_overlap_filter_splits_oversized_batch_and_keeps_safe_pose(
        self,
    ) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)

        def candidate(index: int) -> dict:
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            return {
                "q_start": q_start.copy(),
                "q_final": q_final,
                "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
                "final_eef_position_m": np.asarray(
                    [0.41 + 0.20 * index, 0.0, 0.70]
                ),
                "final_eef_quaternion_xyzw": np.asarray(
                    [0.0, 0.0, 0.0, 1.0]
                ),
                "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
                "selected_is_precise": True,
                "selected_candidate": {
                    "precise": True,
                    "motion_score": float(index),
                    "accuracy_cost_mm_plus_deg": 0.0,
                    "position_error_mm": 0.0,
                    "selection_orientation_error_deg": 0.0,
                    "endpoint_score_rank": index,
                },
            }

        candidates = [candidate(0), candidate(1)]
        overlap_batch_sizes = []

        def overlap_evaluation(*, poses, **_kwargs):
            overlap_batch_sizes.append(len(poses))
            if len(poses) > 1:
                raise official_tools.ProjectiveOccupancyContractError(
                    "local occupancy grid has 11594252 voxels, limit is 4000000"
                )
            report = self._safe_overlap_evaluation(poses=poses)
            position_x = float(np.asarray(poses[0]["eef_pos"])[0])
            if position_x < 0.5:
                report["poses"][0].update(
                    {
                        "overlap_vox": 6,
                        "overlap_vol_cm3": 0.162,
                        "original_overlap_ok": False,
                        "inflated_overlap_vox": None,
                        "inflated_overlap_vol_cm3": None,
                        "inflated_overlap_ok": False,
                        "inflated_query_skipped": True,
                        "inflated_query_skipped_reason": (
                            "original_overlap_failed"
                        ),
                        "ok": False,
                        "reason": "original_overlap_exceeded",
                    }
                )
                report["passing_pose_count"] = 0
                report["rejected_pose_count"] = 1
                report["passing_pose_indices"] = []
            return report

        path_calls = []

        def path_planner(**kwargs):
            path_calls.append(np.asarray(kwargs["q_final"]).copy())
            return {
                "waypoints": [np.asarray(kwargs["q_final"]).astype(float).tolist()],
                "same_branch_checked": True,
            }

        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={
                **candidates[0],
                "_ranked_endpoint_candidates": candidates,
            },
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=path_planner,
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.42, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertTrue(planned["ok"], planned)
        self.assertEqual(overlap_batch_sizes, [2, 1, 1])
        self.assertEqual(len(path_calls), 1)
        np.testing.assert_allclose(path_calls[0], candidates[1]["q_final"])
        audit = planned["path_plan"]["frozen_rgbd_overlap_filter"]
        self.assertEqual(audit["adaptive_split_count"], 1)
        self.assertEqual(audit["batch_attempt_count"], 3)
        self.assertEqual(audit["successful_batch_count"], 2)
        self.assertEqual(audit["candidate_count_evaluation_unavailable"], 0)
        self.assertEqual(audit["maximum_attempted_grid_voxels"], 11594252)
        self.assertEqual(audit["grid_voxel_limit"], 4000000)
        self.assertEqual(audit["candidate_count_passing"], 1)
        self.assertEqual(audit["candidate_count_rejected"], 1)

    def test_plan_overlap_filter_skips_singleton_grid_failure_for_next_pose(
        self,
    ) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)

        def candidate(index: int) -> dict:
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            return {
                "q_start": q_start.copy(),
                "q_final": q_final,
                "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
                "final_eef_position_m": np.asarray(
                    [0.41 + 0.20 * index, 0.0, 0.70]
                ),
                "final_eef_quaternion_xyzw": np.asarray(
                    [0.0, 0.0, 0.0, 1.0]
                ),
                "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
                "selected_is_precise": True,
                "selected_candidate": {
                    "precise": True,
                    "motion_score": float(index),
                    "accuracy_cost_mm_plus_deg": 0.0,
                    "position_error_mm": 0.0,
                    "selection_orientation_error_deg": 0.0,
                    "endpoint_score_rank": index,
                },
            }

        candidates = [candidate(0), candidate(1)]

        def overlap_evaluation(*, poses, **_kwargs):
            if len(poses) > 1 or float(np.asarray(poses[0]["eef_pos"])[0]) < 0.5:
                raise official_tools.ProjectiveOccupancyContractError(
                    "local occupancy grid has 11594252 voxels, limit is 4000000"
                )
            return self._safe_overlap_evaluation(poses=poses)

        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={
                **candidates[0],
                "_ranked_endpoint_candidates": candidates,
            },
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            return_value={
                "waypoints": [candidates[1]["q_final"].astype(float).tolist()],
                "same_branch_checked": True,
            },
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.42, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertTrue(planned["ok"], planned)
        np.testing.assert_allclose(
            planned["endpoint"]["q_final"], candidates[1]["q_final"]
        )
        audit = planned["path_plan"]["frozen_rgbd_overlap_filter"]
        self.assertEqual(audit["candidate_count_evaluation_unavailable"], 1)
        self.assertEqual(audit["candidate_count_passing"], 1)
        self.assertEqual(audit["candidate_count_rejected"], 1)
        unavailable = [
            failure
            for failure in planned["path_failures"]
            if failure.get("reason") == "overlap_evaluation_grid_limit"
        ]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["requested_grid_voxels"], 11594252)

    def test_plan_overlap_filter_reports_all_singletons_unevaluable(self) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)
        candidates = []
        for index in range(2):
            q_final = q_start.copy()
            q_final[0] = 0.01 * (index + 1)
            candidates.append(
                {
                    "q_start": q_start.copy(),
                    "q_final": q_final,
                    "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
                    "final_eef_position_m": np.asarray(
                        [0.41 + 0.20 * index, 0.0, 0.70]
                    ),
                    "final_eef_quaternion_xyzw": np.asarray(
                        [0.0, 0.0, 0.0, 1.0]
                    ),
                    "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
                    "selected_is_precise": True,
                    "selected_candidate": {
                        "precise": True,
                        "motion_score": float(index),
                        "accuracy_cost_mm_plus_deg": 0.0,
                        "position_error_mm": 0.0,
                        "selection_orientation_error_deg": 0.0,
                        "endpoint_score_rank": index,
                    },
                }
            )

        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value={
                **candidates[0],
                "_ranked_endpoint_candidates": candidates,
            },
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=official_tools.ProjectiveOccupancyContractError(
                "local occupancy grid has 11594252 voxels, limit is 4000000"
            ),
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=AssertionError("unevaluable poses must not reach paths"),
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.42, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=self._safe_overlap_context(),
            )

        self.assertFalse(planned["ok"], planned)
        self.assertIn("no endpoint candidate passes", planned["error"])
        audit = planned["frozen_rgbd_overlap_filter"]
        self.assertEqual(audit["candidate_count_evaluation_unavailable"], 2)
        self.assertEqual(audit["candidate_count_rejected"], 2)
        self.assertEqual(audit["candidate_count_passing"], 0)

    def test_plan_overlap_filter_does_not_hide_non_grid_evaluator_failure(
        self,
    ) -> None:
        candidate = {
            "final_eef_position_m": np.asarray([0.4, 0.0, 0.7]),
            "final_eef_quaternion_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "selected_candidate": {"endpoint_score_rank": 0},
        }
        with mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=ValueError("synthetic evaluator implementation bug"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic evaluator implementation bug",
            ):
                official_tools._move_tracked_filter_endpoint_overlap_frontier(
                    [(candidate, "test")],
                    overlap_filter_context=self._safe_overlap_context(),
                    stage="test",
                )

    def test_plan_overlap_filter_does_not_split_non_grid_contract_failure(
        self,
    ) -> None:
        candidates = [
            (
                {
                    "final_eef_position_m": np.asarray(
                        [0.4 + 0.1 * index, 0.0, 0.7]
                    ),
                    "final_eef_quaternion_xyzw": np.asarray(
                        [0.0, 0.0, 0.0, 1.0]
                    ),
                    "selected_candidate": {"endpoint_score_rank": index},
                },
                "test",
            )
            for index in range(2)
        ]
        evaluator = mock.Mock(
            side_effect=official_tools.ProjectiveOccupancyContractError(
                "clicked RGB-D surface is missing from V53 Warp occupancy "
                "(nearest=Nonemm, limit=7.500mm)"
            )
        )
        with mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            evaluator,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "non-grid occupancy contract failure",
            ):
                official_tools._move_tracked_filter_endpoint_overlap_frontier(
                    candidates,
                    overlap_filter_context=self._safe_overlap_context(),
                    stage="test",
                )
        self.assertEqual(evaluator.call_count, 1)

    def test_exec_mode_planning_does_not_evaluate_rgbd_overlap(self) -> None:
        q_start = np.zeros(ARM_DOF, dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] = 0.01
        candidate = {
            "q_start": q_start.copy(),
            "q_final": q_final,
            "start_eef_position_m": np.asarray([0.40, 0.0, 0.70]),
            "final_eef_position_m": np.asarray([0.41, 0.0, 0.70]),
            "final_eef_quaternion_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "anchors_eef_m": np.zeros((1, 3), dtype=np.float64),
            "selected_is_precise": True,
            "selected_candidate": {
                "precise": True,
                "motion_score": 0.0,
                "accuracy_cost_mm_plus_deg": 0.0,
                "position_error_mm": 0.0,
                "selection_orientation_error_deg": 0.0,
                "endpoint_score_rank": 0,
            },
        }
        overlap_evaluator = mock.Mock(
            side_effect=AssertionError("exec mode must not evaluate overlap")
        )
        with mock.patch.object(
            official_tools,
            "_plan_tracked_endpoint",
            return_value=candidate,
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            overlap_evaluator,
        ), mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            return_value={
                "waypoints": [q_final.astype(float).tolist()],
                "same_branch_checked": True,
            },
        ):
            planned = official_tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=q_start,
                source_points=np.asarray([[0.4, 0.0, 0.7]]),
                target_points=[
                    {"name": "p", "target_xyz_m": [0.41, 0.0, 0.7]}
                ],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context=None,
            )

        self.assertTrue(planned["ok"], planned)
        overlap_evaluator.assert_not_called()
        self.assertNotIn(
            "frozen_rgbd_overlap_filter", planned["path_plan"]
        )

    def test_plan_mode_previews_exact_endpoint_and_exec_plan_pose_replays_it(
        self,
    ) -> None:
        self._assert_plan_preview_and_exec_replay(overlap_fallback=False)

    def test_plan_mode_overlap_fallback_previews_and_exec_plan_pose_replays_it(
        self,
    ) -> None:
        self._assert_plan_preview_and_exec_replay(overlap_fallback=True)

    def test_plan_mode_inflated_overlap_fallback_previews_and_replays_it(
        self,
    ) -> None:
        self._assert_plan_preview_and_exec_replay(
            overlap_fallback=True, inflated_only=True
        )

    def test_plan_mode_compacts_large_overlap_audit_before_persisting(self) -> None:
        self._assert_plan_preview_and_exec_replay(
            overlap_fallback=False,
            large_overlap_audit=True,
        )

    def _assert_plan_preview_and_exec_replay(
        self,
        *,
        overlap_fallback: bool,
        inflated_only: bool = False,
        large_overlap_audit: bool = False,
    ) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"axe_tip": position})
        world._official_tracked_object_distances = manager
        target = np.asarray(position, dtype=np.float64) + [0.008, 0.0, 0.0]
        initial_arms = {
            side: np.asarray(world.arm_qpos_list(side), dtype=np.float64).copy()
            for side in ("left", "right")
        }
        initial_trunk = np.asarray(world.trunk_qpos(), dtype=np.float64).copy()
        frozen_capture = official_tools._FrozenCapture(
            session_id=manager.session_id,
            image_id=manager.image_id,
            role="head",
            depth=np.ones((24, 32), dtype=np.float32),
            camera={
                "pos": [12.0, -8.0, 3.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "robot_relative_pose": {
                    "pos": [0.45, 0.0, 0.6],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
            },
            evaluator_sequence=int(adapter.status()["sequence"]),
        )
        overlay_call: dict = {}
        expected_overlap = (0.081 if inflated_only else 0.162) if overlap_fallback else 0.0
        expected_inflated = (1.701 if inflated_only else None) if overlap_fallback else 0.0

        def overlap_evaluation(*, poses, **kwargs):
            report = self._safe_overlap_evaluation(poses=poses, **kwargs)
            if large_overlap_audit:
                report["reconstruction"] = {
                    "topology_components": "x"
                    * (official_tools.MAX_PLAN_RECORD_BYTES // 2),
                }
            if overlap_fallback:
                for pose in report["poses"]:
                    pose.update({
                        "ok": False,
                        "overlap_vox": 3 if inflated_only else 6,
                        "overlap_vol_cm3": expected_overlap,
                        "original_overlap_ok": inflated_only,
                        "inflated_overlap_vox": 63 if inflated_only else None,
                        "inflated_overlap_vol_cm3": expected_inflated,
                        "inflated_overlap_ok": False,
                        "inflated_query_skipped": not inflated_only,
                        "inflated_query_skipped_reason": (
                            None if inflated_only else "original_overlap_failed"
                        ),
                        "reason": (
                            "inflated_overlap_exceeded"
                            if inflated_only else "original_overlap_exceeded"
                        ),
                    })
                report.update({
                    "passing_pose_count": 0,
                    "rejected_pose_count": len(poses),
                    "passing_pose_indices": [],
                })
            return report

        def render_overlay(**kwargs):
            overlay_call.update(kwargs)
            output_path = Path(kwargs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x89PNG\r\n\x1a\nmove-tracked-plan")
            return {
                "ok": True,
                "path": str(output_path),
                "visible_pixel_count": 64,
                "color_rgb": [255, 0, 0],
            }

        plan_ctx, plan_result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ), mock.patch.object(
            official_tools,
            "_run_tracked_plan_worker",
            # This contract test supplies in-process geometry/render doubles;
            # real process IPC/deadline coverage lives in its worker test suite.
            side_effect=lambda kwargs, **options: official_tools._move_tracked_plan_frozen(**kwargs),
        ), mock.patch.object(
            official_tools, "_move_tracked_freeze_plan_capture",
            return_value={"image_id": "img_plan_frozen", "observation_sequence": frozen_capture.evaluator_sequence},
        ), mock.patch.object(
            official_tools,
            "_load_frozen_capture",
            return_value=frozen_capture,
        ), mock.patch.object(
            official_tools,
            "_move_tracked_plan_overlap_context",
            return_value=self._safe_overlap_context(),
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "render_plan_gripper_overlay",
            side_effect=render_overlay,
        ):
            plan_actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    plan_ctx,
                    points=[
                        {
                            "name": "axe_tip",
                            "target_xyz_m": target.astype(float).tolist(),
                        }
                    ],
                    execution_mode="plan",
                    max_steps=180,
                ),
            )

            self.assertTrue(plan_result["ok"], plan_result)
            self.assertEqual(plan_result["execution_mode"], "plan")
            self.assertEqual(plan_result["image_id"], "img_plan_frozen")
            self.assertEqual(plan_result["execution"], "plan_only_no_robot_motion")
            self.assertEqual(plan_result["overlap_vol_cm3"], expected_overlap)
            self.assertEqual(plan_result["inflated_overlap_vol_cm3"], expected_inflated)
            self.assertEqual(plan_result["overlap_volume_filter"]["ok"], not overlap_fallback)
            self.assertEqual(plan_result["overlap_fallback_applied"], overlap_fallback)
            self.assertEqual(plan_result["overlap_thresholds_satisfied"], not overlap_fallback)
            self.assertEqual(bool(plan_result["warnings"]), overlap_fallback)
            self.assertEqual(
                plan_result["plan_preview_overlay"],
                "red_gripper_on_frozen_head",
            )
            self.assertTrue(plan_result["render_image"].startswith("data:image/png;base64,"))
            self.assertEqual(
                plan_result["marked_image_url"],
                plan_result["render_image_path"],
            )
            self.assertEqual(overlay_call["eef_pose"], plan_result["eef_pose"])
            self.assertEqual(
                overlay_call["camera"]["pos"],
                frozen_capture.camera["robot_relative_pose"]["pos"],
            )
            self.assertNotEqual(
                overlay_call["camera"]["pos"],
                frozen_capture.camera["pos"],
            )
            self.assertEqual(overlay_call["camera"]["frame"], "robot_base")
            self.assertEqual(
                plan_result["gripper_visualization"]["camera_frame"],
                "robot_base",
            )
            self.assertGreaterEqual(len(plan_actions), 1)
            for action in plan_actions:
                self.assertEqual(action.shape, (ACTION_DIM,))
                self.assertTrue(np.isfinite(action).all())
                np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                for side in ("left", "right"):
                    np.testing.assert_allclose(
                        action[ACTION_SLICES[f"arm_{side}"]],
                        initial_arms[side],
                        rtol=0.0,
                        atol=1e-6,
                    )
            for side in ("left", "right"):
                np.testing.assert_allclose(
                    world.arm_qpos_list(side),
                    initial_arms[side],
                    rtol=0.0,
                    atol=1e-6,
                )

            trajectory, plan_path, record = official_tools._load_plan_trajectory(
                manager.session_id,
                plan_result["plan_id"],
                include_record=True,
            )
            self.assertEqual(
                overlay_call["gripper_qpos"],
                trajectory["start_state"]["gripper_qpos"][
                    trajectory["active_arm"]
                ],
            )
            self.assertEqual(record["tool"], "move_tracked_point")
            self.assertEqual(trajectory["source_tool"], "move_tracked_point")
            self.assertLessEqual(
                Path(plan_path).stat().st_size,
                official_tools.MAX_PLAN_RECORD_BYTES,
            )
            if large_overlap_audit:
                planner = record["move_tracked_trajectory"]["planner"]
                compacted_batches = planner[
                    "frozen_rgbd_overlap_filter"
                ]["batches"]
                compacted = [
                    batch["reconstruction"]
                    for batch in compacted_batches
                    if "reconstruction" in batch
                ]
                self.assertTrue(compacted)
                for diagnostic in compacted:
                    self.assertTrue(diagnostic["plan_record_compacted"])
                    self.assertEqual(
                        diagnostic["reason"],
                        "non_execution_overlap_diagnostic",
                    )
                    self.assertGreater(
                        diagnostic["canonical_size_bytes"],
                        official_tools.MOVE_TRACKED_PLAN_INLINE_DIAGNOSTIC_BYTES,
                    )
                    self.assertEqual(len(diagnostic["sha256"]), 64)
            planned_safe = trajectory["safety_validation"]["planned_safe"]
            self.assertTrue(planned_safe["ok"])
            self.assertEqual(
                trajectory["parameters"]["back_m"],
                planned_safe["active_back_m"],
            )
            self.assertGreaterEqual(
                trajectory["parameters"]["back_m"],
                official_tools.EXEC_SAFE_BACK_MIN_M,
            )
            self.assertLessEqual(
                trajectory["parameters"]["back_m"],
                official_tools.EXEC_SAFE_BACK_MAX_M,
            )
            self.assertEqual(
                [segment["name"] for segment in trajectory["segments"]],
                ["tool_roll_reset", "current_to_safe", "safe_to_final"],
            )
            self.assertNotIn("trunk_assisted", trajectory["tracking"])
            self.assertNotIn("trunk_waypoints", trajectory)
            self.assertNotIn("final_trunk_q", trajectory)
            for segment in trajectory["segments"]:
                self.assertNotIn("trunk_waypoints", segment)
                self.assertNotIn("endpoint_trunk_q", segment)
            self.assertEqual(
                trajectory["safety_validation"]["frozen_rgbd_collision_checked"],
                not overlap_fallback,
            )
            self.assertTrue(trajectory["safety_validation"]["frozen_rgbd_collision_evaluated"])
            self.assertEqual(trajectory["safety_validation"]["overlap_fallback_applied"], overlap_fallback)
            if overlap_fallback:
                for mutation in ("unmarked", "pose", "accuracy", "passing", "policy"):
                    with self.subTest(tampering=mutation):
                        changed = json.loads(json.dumps(record["move_tracked_trajectory"]))
                        audit = changed["planner"]["frozen_rgbd_overlap_filter"]
                        if mutation == "unmarked":
                            audit["fallback_applied"] = False
                        elif mutation == "pose":
                            changed["target"]["overlap_volume_filter"]["eef_position_robot_base_m"][0] += 0.01
                        elif mutation == "accuracy":
                            changed["planner"]["endpoint_selected_candidate"]["accuracy_cost_mm_plus_deg"] += 1.0
                        elif mutation == "passing":
                            # The count is cumulative and may legitimately be
                            # positive when an overlap-passing endpoint was
                            # exhausted by path planning.  Tamper with the
                            # count by violating its upper-bound invariant
                            # instead of assuming that every fallback must
                            # have zero passing endpoints.
                            audit["endpoint_candidate_count_passing"] = (
                                audit["candidate_count_evaluated"] + 1
                            )
                        else:
                            audit["fallback_selection"]["policy"] = "arbitrary_bypass"
                        with self.assertRaises(ValueError):
                            official_tools._compile_move_tracked_exec_plan_trajectory(changed)
                # Regression: a valid fallback may follow one or more
                # overlap-passing endpoints whose paths were exhausted.  The
                # compiler must accept that cumulative evidence rather than
                # treating it as an inconsistent fallback report.
                positive_count = json.loads(
                    json.dumps(record["move_tracked_trajectory"])
                )
                positive_audit = positive_count["planner"][
                    "frozen_rgbd_overlap_filter"
                ]
                positive_audit["endpoint_candidate_count_passing"] = 1
                positive_audit["candidate_count_passing"] = max(
                    1, int(positive_audit["candidate_count_passing"])
                )
                self.assertTrue(
                    official_tools._compile_move_tracked_exec_plan_trajectory(
                        positive_count
                    )
                )
            self.assertEqual(plan_path, plan_result["plan_record_path"])
            self.assertEqual(
                trajectory["integrity"]["digest"],
                plan_result["trajectory_digest"],
            )
            np.testing.assert_allclose(
                record["move_tracked_trajectory"]["target"][
                    "eef_position_robot_base_m"
                ],
                plan_result["eef_pose"]["pos"],
                rtol=0.0,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                trajectory["segments"][-1]["endpoint_q"],
                record["move_tracked_trajectory"]["target"]["endpoint_q"],
                rtol=0.0,
                atol=1e-9,
            )

            exec_ctx, exec_result = self._ctx(world)
            exec_actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["exec_plan_pose"].fn(
                    exec_ctx,
                    session_id=manager.session_id,
                    plan_id=plan_result["plan_id"],
                ),
            )
            self.assertTrue(exec_result["ok"], exec_result)
            self.assertEqual(
                exec_result["execution"],
                "official_action_joint_trajectory_replay",
            )
            self.assertEqual(
                exec_result["active_back_m"],
                trajectory["parameters"]["back_m"],
            )
            self.assertFalse(exec_result["trunk_motion"])
            self.assertGreaterEqual(len(exec_actions), 1)
            np.testing.assert_allclose(
                world.arm_qpos_list(plan_result["arm"]),
                trajectory["segments"][-1]["endpoint_q"],
                rtol=0.0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                world.trunk_qpos(),
                initial_trunk,
                rtol=0.0,
                atol=1e-6,
            )
            for action in exec_actions:
                np.testing.assert_allclose(
                    action[ACTION_SLICES["trunk"]],
                    initial_trunk,
                    rtol=0.0,
                    atol=1e-6,
                )
            if ARM_DOF == 8:
                self.assertTrue(
                    all(
                        float(
                            action[
                                ACTION_SLICES[f"arm_{plan_result['arm']}"]
                            ][7]
                        )
                        == 0.0
                        for action in exec_actions
                    )
                )

    def test_two_point_plan_mode_freezes_current_frame_without_eef_prior_wait(
        self,
    ) -> None:
        adapter, world = self._ready_world()
        _state, (position, quaternion) = self._left_eef(world)
        rotation = quat_to_mat_xyzw(quaternion)
        names = ["edge-left", "edge-right"]
        anchors = np.asarray(
            [[-0.035, 0.0, 0.0], [0.035, 0.0, 0.0]],
            dtype=np.float64,
        )
        initial_points = position[None, :] + (rotation @ anchors.T).T

        class EdgeDroppingPriorManager(_RigidTrackedManager):
            def __init__(self, *manager_args, **manager_kwargs):
                super().__init__(*manager_args, **manager_kwargs)
                self.prior_activation_count = 0

            def activate_on_hand_eef_observation_prior(self, *args, **kwargs):
                self.prior_activation_count += 1
                self.loss_after_sequence = int(self.adapter.status()["sequence"]) + 1
                return {
                    "ok": True,
                    "active": True,
                    "lease_id": "edge-dropping-prior",
                    "activation_observation_sequence": int(
                        self.adapter.status()["sequence"]
                    ),
                    "max_point_error_m": 0.030,
                }

            def deactivate_on_hand_eef_observation_prior(self, lease_id):
                return {
                    "ok": True,
                    "released": True,
                    "lease_id": str(lease_id),
                }

        manager = EdgeDroppingPriorManager(
            world,
            adapter,
            dict(zip(names, initial_points)),
        )
        world._official_tracked_object_distances = manager
        target_points = initial_points + np.asarray([0.006, 0.0, 0.0])
        frozen_capture = official_tools._FrozenCapture(
            session_id=manager.session_id,
            image_id=manager.image_id,
            role="head",
            depth=np.ones((24, 32), dtype=np.float32),
            camera={
                "pos": [0.45, 0.0, 0.6],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "robot_relative_pose": {
                    "pos": [0.45, 0.0, 0.6],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
            },
            evaluator_sequence=int(adapter.status()["sequence"]),
        )

        def render_overlay(**kwargs):
            output_path = Path(kwargs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x89PNG\r\n\x1a\nordered-plan")
            return {
                "ok": True,
                "path": str(output_path),
                "visible_pixel_count": 64,
                "color_rgb": [255, 0, 0],
            }

        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ), mock.patch.object(
            official_tools,
            "_run_tracked_plan_worker",
            side_effect=lambda kwargs, **options: official_tools._move_tracked_plan_frozen(**kwargs),
        ), mock.patch.object(
            official_tools, "_move_tracked_freeze_plan_capture",
            return_value={"image_id": "img_plan_frozen", "observation_sequence": frozen_capture.evaluator_sequence},
        ), mock.patch.object(
            official_tools,
            "_load_frozen_capture",
            return_value=frozen_capture,
        ), mock.patch.object(
            official_tools,
            "_move_tracked_plan_overlap_context",
            return_value=self._safe_overlap_context(),
        ), mock.patch.object(
            official_tools,
            "_evaluate_rgbd_lite_pose_overlaps",
            side_effect=self._safe_overlap_evaluation,
        ), mock.patch.object(
            official_tools,
            "render_plan_gripper_overlay",
            side_effect=render_overlay,
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": name,
                            "target_xyz_m": target.astype(float).tolist(),
                        }
                        for name, target in zip(names, target_points)
                    ],
                    execution_mode="plan",
                    max_steps=180,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(manager.prior_activation_count, 0)
        self.assertEqual(
            result["preplanning_on_hand_eef_observation_prior"]["reason"],
            "plan_mode_uses_current_synchronized_tracker_frame",
        )
        source_sync = result["preplanning_source_sync"]
        self.assertTrue(source_sync["ok"], source_sync)
        self.assertTrue(source_sync["all_points_frozen_from_one_observation"])
        self.assertFalse(source_sync["planning_source_rebased"])
        self.assertEqual(
            source_sync["initial_observation_sequence"],
            source_sync["planning_observation_sequence"],
        )
        self.assertGreaterEqual(len(actions), 1)

    def test_full_tool_moves_three_point_rigid_bundle(self) -> None:
        with mock.patch.object(
            official_tools,
            "_move_tracked_activate_on_hand_eef_prior",
            wraps=official_tools._move_tracked_activate_on_hand_eef_prior,
        ) as activate_prior:
            result, actions = self._run_full_n_point_translation_case(
                [[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.05, 0.0]]
            )
        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(activate_prior.call_count, 2)
        self.assertTrue(
            all(len(call.args[1]) == 3 for call in activate_prior.call_args_list)
        )
        self.assertEqual(result["point_count"], 3)
        self.assertEqual(result["geometry_rank"], 2)
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        self.assertEqual(len(result["final_live_points_robot_base_m"]), 3)
        self.assertEqual(len(result["per_point_constraint_errors"]), 3)
        self.assertGreater(len(actions), 2)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())

    def test_full_tool_moves_four_point_rigid_bundle_and_checks_redundancy(self) -> None:
        result, actions = self._run_full_n_point_translation_case(
            [
                [0.0, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.04],
            ]
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["point_count"], 4)
        self.assertEqual(result["geometry_rank"], 3)
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        self.assertEqual(len(result["final_live_points_robot_base_m"]), 4)
        self.assertEqual(len(result["per_point_constraint_errors"]), 4)
        self.assertEqual(
            len(result["final_rigid_pair_consistency"]["pair_distances"]),
            6,
        )
        self.assertGreater(len(actions), 2)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]),
                    0.0,
                )

    def test_full_tool_moves_five_and_six_point_rigid_bundles(self) -> None:
        offsets = [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.04],
            [0.04, 0.03, 0.02],
            [-0.02, 0.025, 0.015],
        ]
        for point_count in (5, 6):
            with self.subTest(point_count=point_count):
                result, actions = self._run_full_n_point_translation_case(
                    offsets[:point_count]
                )
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["point_count"], point_count)
                self.assertTrue(result["final_constraints"]["ok"], result)
                self.assertTrue(
                    result["post_execution_track_validation"]["ok"], result
                )
                self.assertEqual(
                    len(result["final_live_points_robot_base_m"]), point_count
                )
                self.assertEqual(
                    len(result["per_point_constraint_errors"]), point_count
                )
                self.assertEqual(
                    len(result["final_rigid_pair_consistency"]["pair_distances"]),
                    point_count * (point_count - 1) // 2,
                )
                self.assertGreater(len(actions), 2)
                for action in actions:
                    self.assertEqual(action.shape, (ACTION_DIM,))
                    self.assertTrue(np.isfinite(action).all())
                    np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                    if ARM_DOF == 8:
                        self.assertEqual(
                            float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                        )

    def test_pair_distance_disturbance_continues_but_cannot_pass_final_gate(self) -> None:
        class _PairDistanceDriftTrackedManager(_RigidTrackedManager):
            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                arm_motion = float(
                    np.linalg.norm(
                        np.asarray(
                            self.world.arm_qpos_list(self.arm), dtype=np.float64
                        )
                        - self.initial_arm_q,
                        ord=np.inf,
                    )
                )
                if snapshot["ok"] and arm_motion > 1e-5 and len(names) == 2:
                    snapshot["entries"][str(names[1])][
                        "xyz_in_robot_base_coord_m"
                    ][2] += 0.025
                return snapshot

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        source = {
            "head": np.asarray(position) + np.array([0.0, 0.0, -0.04]),
            "tail": np.asarray(position) + np.array([0.0, 0.0, 0.04]),
        }
        manager = _PairDistanceDriftTrackedManager(world, adapter, source)
        world._official_tracked_object_distances = manager
        targets = [
            {
                "name": name,
                "target_xyz_m": (
                    point + np.array([0.008, 0.0, 0.0])
                ).astype(float).tolist(),
            }
            for name, point in source.items()
        ]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=180,
                ),
            )

        self.assertFalse(result["ok"], result)
        self.assertTrue(result["motion"]["ok"], result)
        self.assertEqual(result["motion"]["reason"], "trajectory_complete")
        self.assertGreater(
            result["motion"]["tracking_continuity"][
                "controlled_rigid_geometry_disturbance_steps"
            ],
            0,
        )
        self.assertFalse(result["post_execution_track_validation"]["ok"])
        self.assertFalse(result["final_rigid_pair_consistency"]["ok"])

    def test_npoint_rigid_geometry_disturbance_pauses_and_recovers(self) -> None:
        class _TransientRigidGeometryDriftManager(_RigidTrackedManager):
            disturbed_snapshots = 0

            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                arm_motion = float(
                    np.linalg.norm(
                        np.asarray(
                            self.world.arm_qpos_list(self.arm),
                            dtype=np.float64,
                        )
                        - self.initial_arm_q,
                        ord=np.inf,
                    )
                )
                if (
                    snapshot["ok"]
                    and arm_motion > 1.0e-5
                    and len(names) == 3
                    and self.disturbed_snapshots < 2
                ):
                    self.disturbed_snapshots += 1
                    snapshot["entries"][str(names[2])][
                        "xyz_in_robot_base_coord_m"
                    ][1] += 0.035
                return snapshot

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        names = ["marker_1", "marker_2", "marker_3"]
        source = np.asarray(position, dtype=np.float64) + np.asarray(
            [
                [0.0, 0.0, -0.04],
                [0.06, 0.0, -0.04],
                [0.0, 0.05, -0.04],
            ],
            dtype=np.float64,
        )
        manager = _TransientRigidGeometryDriftManager(
            world, adapter, dict(zip(names, source))
        )
        world._official_tracked_object_distances = manager
        targets = [
            {
                "name": name,
                "target_xyz_m": (
                    point + np.array([0.008, 0.0, 0.0])
                ).astype(float).tolist(),
            }
            for name, point in zip(names, source)
        ]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    max_steps=180,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["motion"]["ok"], result)
        self.assertEqual(result["motion"]["reason"], "trajectory_complete")
        self.assertEqual(manager.disturbed_snapshots, 2)
        self.assertGreaterEqual(
            result["motion"]["tracker_reacquisition_hold_steps"], 2
        )
        self.assertEqual(
            [
                event["event"]
                for event in result["motion"]["tracker_reacquisition_events"]
            ],
            ["hold_started", "observation_reacquired"],
        )
        self.assertGreater(
            result["motion"]["tracking_continuity"][
                "controlled_rigid_geometry_disturbance_steps"
            ],
            0,
        )
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                )
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_right"]][7]), 0.0
                )

    def test_final_live_coordinates_are_a_mandatory_success_gate(self) -> None:
        class _BiasedAfterMotionTrackedManager(_RigidTrackedManager):
            def observed_active_points_snapshot(self, names, *, episode_id):
                snapshot = super().observed_active_points_snapshot(
                    names, episode_id=episode_id
                )
                arm_motion = float(
                    np.linalg.norm(
                        np.asarray(
                            self.world.arm_qpos_list(self.arm), dtype=np.float64
                        )
                        - self.initial_arm_q,
                        ord=np.inf,
                    )
                )
                if snapshot["ok"] and arm_motion > 1e-5:
                    snapshot["entries"][str(names[0])][
                        "xyz_in_robot_base_coord_m"
                    ][0] += 0.010
                return snapshot

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _BiasedAfterMotionTrackedManager(
            world, adapter, {"axe_tip": position}
        )
        world._official_tracked_object_distances = manager
        target = np.asarray(position) + np.array([0.008, 0.0, 0.0])
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "axe_tip",
                            "target_xyz_m": target.astype(float).tolist(),
                        }
                    ],
                    pos_tol=0.004,
                    max_steps=180,
                ),
            )

        self.assertTrue(result["motion"]["ok"], result)
        self.assertFalse(result["stability"]["ok"], result)
        self.assertEqual(
            result["stability"]["reason"],
            "final_goal_not_reached",
        )
        self.assertTrue(result["final_anchor_ok"], result)
        self.assertFalse(result["final_constraints"]["ok"], result)
        self.assertFalse(result["ok"], result)
        validation = result["post_execution_track_validation"]
        self.assertFalse(validation["ok"], result)
        self.assertTrue(validation["required_for_success"])
        self.assertEqual(validation["source"], "subsequent_evaluator_observation")
        self.assertTrue(validation["synchronized_with_final_proprioception"])
        self.assertGreater(
            validation["max_constraint_error_m"], validation["tolerance_m"]
        )
        self.assertIn("axe_tip", validation["points_robot_base_m"])

    def test_symbolic_pair_noop_resolves_variables(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        points = {
            "a": np.asarray(position) + [0.0, 0.0, -0.04],
            "b": np.asarray(position) + [0.0, 0.0, 0.04],
        }
        manager = _RigidTrackedManager(world, adapter, points)
        world._official_tracked_object_distances = manager
        targets = [
            {
                "name": name,
                "target_xyz_m": {
                    "x": "common_x",
                    "y": "common_y",
                    "z": f"z_{name}",
                },
            }
            for name in ("a", "b")
        ]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx, points=targets, pos_tol=0.004, max_steps=120
                ),
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["internal_point_order"], ["a", "b"])
        self.assertEqual(
            set(result["resolved_variables"]),
            {"common_x", "common_y", "z_a", "z_b"},
        )

    def test_shared_xy_pair_plans_and_executes_nonzero_motion(self) -> None:
        """A large underconstrained rotation must fit the evaluator step budget."""

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        position = np.asarray(position, dtype=np.float64)
        source_points = {
            "reference_17": position + [-0.04, 0.0, 0.0],
            "reference_93": position + [0.04, 0.0, 0.0],
        }
        world._official_tracked_object_distances = _RigidTrackedManager(
            world,
            adapter,
            source_points,
        )
        targets = [
            {
                "name": "reference_17",
                "target_xyz_m": ["shared_u", "shared_v", "free_z_17"],
            },
            {
                "name": "reference_93",
                "target_xyz_m": ["shared_u", "shared_v", "free_z_93"],
            },
        ]
        ctx, result = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=targets,
                    pos_tol=0.012,
                    ori_tol_deg=5.0,
                    max_steps=360,
                    timeout_s=90.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["motion"]["ok"], result)
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertEqual(
            result["internal_point_order"],
            ["reference_17", "reference_93"],
        )
        self.assertGreater(result["trajectory_waypoint_count"], 1)
        self.assertLessEqual(result["trajectory_waypoint_count"], 358)
        if result["planner"].get("path_mode") == "joint_space_fallback":
            self.assertEqual(
                result["planner"]["fallback_reason"],
                "straight_cartesian_path_unavailable",
            )
            self.assertLessEqual(
                result["planner"]["actual_max_joint_step_rad"],
                result["planner"]["max_joint_step_rad"] + 1e-12,
            )
        else:
            self.assertGreaterEqual(
                result["planner"]["orientation_step_deg"],
                tracked_motion.DEFAULT_CARTESIAN_ORIENTATION_STEP_DEG,
            )
        final_points = result["final_constraints"]["resolved_points_xyz_m"]
        self.assertLessEqual(
            abs(
                float(final_points["reference_17"][0])
                - float(final_points["reference_93"][0])
            ),
            0.012,
        )
        self.assertLessEqual(
            abs(
                float(final_points["reference_17"][1])
                - float(final_points["reference_93"][1])
            ),
            0.012,
        )
        self.assertTrue(any(np.any(action != actions[0]) for action in actions[1:]))

    def test_supported_constraint_shapes_all_plan_and_execute(self) -> None:
        """Every public affine shape uses the same endpoint and path pipeline."""

        case_names = ("shared_y", "shared_yz", "x_offset", "fixed_first")
        for case_name in case_names:
            with self.subTest(case=case_name):
                adapter, world = self._ready_world()
                _state, (position, _quaternion) = self._left_eef(world)
                position = np.asarray(position, dtype=np.float64)
                if case_name in {"shared_y", "shared_yz"}:
                    source = np.asarray(
                        [
                            position + [-0.04, -0.03, -0.03],
                            position + [0.04, 0.03, 0.03],
                        ],
                        dtype=np.float64,
                    )
                elif case_name == "x_offset":
                    source = np.asarray(
                        [position + [-0.05, 0.0, 0.0], position + [0.05, 0.0, 0.0]],
                        dtype=np.float64,
                    )
                else:
                    source = np.asarray(
                        [position + [-0.04, 0.0, 0.0], position + [0.04, 0.0, 0.0]],
                        dtype=np.float64,
                    )

                if case_name == "shared_y":
                    rows = [
                        ["x_first", "same_y", "z_first"],
                        ["x_second", "same_y", "z_second"],
                    ]
                elif case_name == "shared_yz":
                    rows = [
                        ["x_first", "same_y", "same_z"],
                        ["x_second", "same_y", "same_z"],
                    ]
                elif case_name == "x_offset":
                    rows = [["origin_x", "?", "?"], ["origin_x+0.07", "?", "?"]]
                else:
                    rows = [source[1].astype(float).tolist(), ["?", "?", "?"]]

                names = ("first_reference", "second_reference")
                world._official_tracked_object_distances = _RigidTrackedManager(
                    world,
                    adapter,
                    dict(zip(names, source)),
                )
                targets = [
                    {"name": name, "target_xyz_m": row}
                    for name, row in zip(names, rows)
                ]
                initial_q = np.asarray(
                    world.arm_qpos_list("left"),
                    dtype=np.float64,
                )
                ctx, result = self._ctx(world)
                with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
                    os.environ,
                    {"BEHAVIOR_AGENT_RUNS": temp_root},
                ):
                    actions = self._drive(
                        adapter,
                        world,
                        build_registry(adapter)["move_tracked_point"].fn(
                            ctx,
                            points=targets,
                            pos_tol=0.012,
                            ori_tol_deg=5.0,
                            max_steps=520,
                            timeout_s=90.0,
                        ),
                    )

                self.assertTrue(result["ok"], result)
                self.assertTrue(result["motion"]["ok"], result)
                self.assertTrue(result["final_constraints"]["ok"], result)
                self.assertEqual(
                    result["planner"]["endpoint_selected_candidate"]["mode"],
                    "joint_space_affine_constraint_ik",
                )
                active_slice = ACTION_SLICES[f"arm_{result['arm']}"]
                self.assertTrue(
                    any(
                        float(
                            np.linalg.norm(
                                np.asarray(action[active_slice], dtype=np.float64)
                                - initial_q,
                                ord=np.inf,
                            )
                        )
                        > 1e-4
                        for action in actions
                    ),
                    result,
                )

                resolved = result["final_constraints"]["resolved_points_xyz_m"]
                first = np.asarray(resolved[names[0]], dtype=np.float64)
                second = np.asarray(resolved[names[1]], dtype=np.float64)
                if case_name == "shared_y":
                    self.assertLessEqual(abs(float(first[1] - second[1])), 0.012)
                elif case_name == "shared_yz":
                    np.testing.assert_allclose(first[1:], second[1:], atol=0.012)
                elif case_name == "x_offset":
                    self.assertAlmostEqual(
                        float(second[0] - first[0]),
                        0.07,
                        delta=0.012,
                    )
                else:
                    np.testing.assert_allclose(first, source[1], atol=0.012)

    def test_unknown_tracking_stall_tamper_and_cancellation_fail_closed(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"known": position})
        world._official_tracked_object_distances = manager
        ctx, unknown = self._ctx(world)
        actions = list(
            build_registry(adapter)["move_tracked_point"].fn(
                ctx,
                points=[{"name": "missing", "target_xyz_m": [0.0, 0.0, 0.0]}],
            )
        )
        self.assertFalse(unknown["ok"])
        self.assertEqual(unknown["failure_stage"], "observation validation")
        self.assertEqual(len(actions), 1)

        target = np.asarray(position) + [0.020, 0.0, 0.0]
        ctx, stalled = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            stall_actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target.tolist()}],
                    pos_tol=0.004,
                    ori_tol_deg=5.0,
                    max_steps=120,
                ),
                follow_actions=False,
            )
        self.assertFalse(stalled["ok"], stalled)
        self.assertEqual(stalled["motion"]["reason"], "proprio_stall")
        self.assertGreaterEqual(len(stall_actions), 4)

        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        ctx, cancelled = self._ctx(
            world,
            raise_if_cancelled=lambda where="": (_ for _ in ()).throw(
                SkillCancelled(f"cancelled at {where}")
            ),
        )
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            cancel_actions = list(
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "known",
                            "target_xyz_m": (np.asarray(position) + [0.008, 0, 0]).tolist(),
                        }
                    ],
                    max_steps=120,
                )
            )
        self.assertFalse(cancelled["ok"], cancelled)
        self.assertEqual(cancelled["failure_stage"], "cancellation")
        self.assertEqual(len(cancel_actions), 1)

    def test_transient_tracking_loss_continues_plan_and_recovers(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"known": position})
        manager.temporary_loss_after_arm_motion_rad = 1e-5
        manager.temporary_loss_snapshot_count = 4
        world._official_tracked_object_distances = manager
        target = np.asarray(position) + [0.008, 0.0, 0.0]
        ctx, recovered = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {"name": "known", "target_xyz_m": target.tolist()}
                    ],
                    max_steps=120,
                ),
            )
        self.assertTrue(recovered["ok"], recovered)
        self.assertTrue(recovered["motion"]["ok"])
        self.assertTrue(recovered["stability"]["ok"])
        self.assertGreaterEqual(
            recovered["tracking_continuity"]["temporarily_unobserved_steps"],
            1,
        )
        self.assertFalse(
            recovered["tracking_continuity"]["stale_coordinates_used"]
        )
        self.assertFalse(
            recovered["tracking_continuity"][
                "trajectory_changed_for_reacquisition"
            ]
        )
        self.assertEqual(
            recovered["motion"]["tracker_reacquisition_hold_steps"], 0
        )
        self.assertEqual(
            recovered["motion"]["tracker_reacquisition_events"], []
        )
        self.assertEqual(
            recovered["motion"]["waypoints_completed"],
            recovered["motion"]["waypoint_count"],
        )
        release = recovered["motion_tracking_retention"]["release"]
        self.assertFalse(release["released"])
        self.assertEqual(
            release["reason"], "bounded_successor_identity_retention"
        )
        self.assertFalse(release["coordinates_published_while_unobserved"])
        self.assertEqual(len(manager._motion_retention_leases), 1)
        self.assertGreaterEqual(len(actions), 2)

    def test_timeout_tracking_loss_and_episode_change_fail_closed(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"known": position})
        world._official_tracked_object_distances = manager
        target = np.asarray(position) + [0.008, 0.0, 0.0]

        ctx, timed_out = self._ctx(world)
        clock_calls = 0

        def fake_monotonic():
            nonlocal clock_calls
            clock_calls += 1
            return 0.0 if clock_calls <= 2 else 1.0

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools.time,
            "monotonic",
            side_effect=fake_monotonic,
        ):
            timeout_actions = list(
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {"name": "known", "target_xyz_m": target.tolist()}
                    ],
                    timeout_s=0.1,
                    max_steps=120,
                )
            )
        self.assertFalse(timed_out["ok"], timed_out)
        self.assertEqual(timed_out["failure_stage"], "timeout")
        self.assertEqual(timed_out["motion"]["reason"], "timeout")
        self.assertGreaterEqual(len(timeout_actions), 1)

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        manager = _RigidTrackedManager(world, adapter, {"known": position})
        manager.loss_after_arm_motion_rad = 1e-5
        world._official_tracked_object_distances = manager
        ctx, tracking_lost = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            loss_actions = self._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "known",
                            "target_xyz_m": (
                                np.asarray(position) + [0.008, 0.0, 0.0]
                            ).tolist(),
                        }
                    ],
                    max_steps=120,
                ),
            )
        self.assertFalse(tracking_lost["ok"], tracking_lost)
        self.assertEqual(tracking_lost["failure_stage"], "tracking")
        self.assertEqual(
            tracking_lost["stability"]["reason"],
            "tracked_point_unavailable",
        )
        self.assertTrue(tracking_lost["motion"]["ok"])
        self.assertEqual(
            tracking_lost["motion"]["waypoints_completed"],
            tracking_lost["motion"]["waypoint_count"],
        )
        self.assertGreater(
            tracking_lost["tracking_continuity"]["max_unobserved_streak"],
            official_tools.MOVE_TRACKED_POINT_MAX_CONSECUTIVE_REACQUIRE_STEPS,
        )
        self.assertFalse(
            tracking_lost["tracking_continuity"][
                "trajectory_changed_for_reacquisition"
            ]
        )
        self.assertFalse(
            tracking_lost["tracking_continuity"]["stale_coordinates_used"]
        )
        self.assertTrue(
            tracking_lost["motion_tracking_retention"]["release"]["released"]
        )
        self.assertEqual(manager._motion_retention_leases, {})
        self.assertGreaterEqual(len(loss_actions), 2)

        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        ctx, episode_changed = self._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            generator = build_registry(adapter)["move_tracked_point"].fn(
                ctx,
                points=[
                    {
                        "name": "known",
                        "target_xyz_m": (
                            np.asarray(position) + [0.008, 0.0, 0.0]
                        ).tolist(),
                    }
                ],
                max_steps=120,
            )
            first_action = np.asarray(next(generator), dtype=np.float32)
            world.reset_observation_state()
            adapter.update(
                {"robot_r1::proprio": adapter.proprio_vector().copy()}
            )
            remaining_actions = [
                np.asarray(action, dtype=np.float32) for action in generator
            ]
        self.assertFalse(episode_changed["ok"], episode_changed)
        self.assertEqual(
            episode_changed["failure_stage"], "start-state validation"
        )
        self.assertIn("episode_changed_during_planning", episode_changed["error"])
        for action in [first_action, *remaining_actions]:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())

    def test_plan_digest_tampering_is_rejected(self) -> None:
        adapter, world = self._ready_world()
        _state, (position, _quaternion) = self._left_eef(world)
        world._official_tracked_object_distances = _RigidTrackedManager(
            world, adapter, {"known": position}
        )
        target = np.asarray(position) + [0.008, 0.0, 0.0]
        ctx, result = self._ctx(world)
        real_loader = official_tools._move_tracked_load_plan

        def tampering_loader(session_id, plan_id):
            root = Path(os.environ["BEHAVIOR_AGENT_RUNS"])
            path = root / session_id / "plans" / f"{plan_id}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["trajectory"]["waypoints"][0][0] += 0.001
            path.write_text(json.dumps(record), encoding="utf-8")
            return real_loader(session_id, plan_id)

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ), mock.patch.object(
            official_tools,
            "_move_tracked_load_plan",
            side_effect=tampering_loader,
        ):
            actions = list(
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[{"name": "known", "target_xyz_m": target.tolist()}],
                    max_steps=120,
                )
            )
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failure_stage"], "planning")
        self.assertIn("digest mismatch", result["error"])
        self.assertGreaterEqual(len(actions), 1)
        for action in actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
