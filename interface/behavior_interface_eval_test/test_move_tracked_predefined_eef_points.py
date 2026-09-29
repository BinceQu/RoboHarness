from __future__ import annotations

import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

from behavior_interface_eval_test.tool.official_v2.contract import (
    validate_move_tracked_point_args,
)
from behavior_interface_eval_test.tool.official_v2.predefined_eef_points_local import (
    PREDEFINED_EEF_POINT_NAMES,
    is_predefined_eef_point_name,
    partition_predefined_eef_point_names,
    predefined_eef_point_local_m,
    predefined_eef_points_robot_base_m,
)
from behavior_interface_eval_test.tool.official_v2 import tools
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
from behavior_interface_eval_test.tool.official_v2.registry import build_registry


class _Adapter:
    def __init__(self, sequence: int = 17) -> None:
        self.sequence = int(sequence)

    def observation_metadata(self):
        return self.sequence, 1.0


class _Tracker:
    def __init__(self, adapter: _Adapter) -> None:
        self.adapter = adapter
        self.snapshot_names: list[list[str]] = []
        self.retention_names: list[list[str]] = []

    def observed_active_points_snapshot(self, names, *, episode_id):
        requested = [str(name) for name in names]
        self.snapshot_names.append(requested)
        return {
            "ok": True,
            "reason": None,
            "entries": {
                name: {
                    "name": name,
                    "status": "observed",
                    "xyz_in_robot_base_coord_m": [0.0, 0.82, 0.0],
                }
                for name in requested
            },
            "unavailable": {},
            "observation_sequence": self.adapter.sequence,
            "episode_id": str(episode_id),
            "session_id": "session_1",
            "image_id": "img_1",
            "measurement_source": "current_evaluator_rgbd",
        }

    def begin_motion_retention(self, names, *, episode_id, timeout_s):
        requested = [str(name) for name in names]
        self.retention_names.append(requested)
        return {
            "ok": True,
            "lease_id": "visual_lease",
            "names": requested,
            "duration_s": float(timeout_s),
        }

    def end_motion_retention(self, lease_id):
        return {"ok": True, "released": True, "lease_id": str(lease_id)}


def _observation(position, *, gripper=(0.05, 0.05)):
    return {
        "position": np.asarray(position, dtype=np.float64),
        "quaternion": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "gripper": np.asarray(gripper, dtype=np.float64),
    }


class PredefinedEefGeometryTests(unittest.TestCase):
    @staticmethod
    def _visual_mesh_component_triangles(component_id: int) -> np.ndarray:
        asset_path = os.path.join(
            os.path.dirname(tools.__file__),
            "assets",
            "r1pro_gripper_visual_mesh_open.npz",
        )
        with np.load(asset_path, allow_pickle=False) as asset:
            vertices = np.asarray(asset["vertices_eef"], dtype=np.float64)
            faces = np.asarray(asset["faces"], dtype=np.int64)
            face_component = np.asarray(asset["face_component"], dtype=np.int64)
        return vertices[faces[face_component == int(component_id)]]

    @staticmethod
    def _visual_mesh_tip(component_id: int) -> np.ndarray:
        triangles = PredefinedEefGeometryTests._visual_mesh_component_triangles(
            component_id
        )
        component_vertices = np.unique(triangles.reshape(-1, 3), axis=0)
        tip_z = float(np.max(component_vertices[:, 2]))
        cap = component_vertices[component_vertices[:, 2] >= tip_z - 0.00025]
        point = cap.mean(axis=0)
        point[2] = tip_z
        return point

    @staticmethod
    def _point_lies_on_horizontal_mesh_face(
        point: np.ndarray, triangles: np.ndarray, *, atol: float = 2.0e-9
    ) -> bool:
        target = np.asarray(point, dtype=np.float64).reshape(3)
        for triangle in np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3):
            if not np.all(np.abs(triangle[:, 2] - target[2]) <= atol):
                continue
            a, b, c = triangle[:, :2]
            v0 = c - a
            v1 = b - a
            v2 = target[:2] - a
            denominator = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(float(denominator)) <= 1.0e-15:
                continue
            u = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator
            v = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator
            if u >= -atol and v >= -atol and u + v <= 1.0 + atol:
                return True
        return False

    def test_catalog_and_partition_are_exact_and_ordered(self) -> None:
        self.assertEqual(
            PREDEFINED_EEF_POINT_NAMES,
            ("left_finger_tip", "right_finger_tip", "gripper_slide_center"),
        )
        self.assertTrue(is_predefined_eef_point_name("left_finger_tip"))
        self.assertFalse(is_predefined_eef_point_name("Left_finger_tip"))
        predefined, tracked = partition_predefined_eef_point_names(
            ["scene", "right_finger_tip", "left_finger_tip", "door"]
        )
        self.assertEqual(predefined, ["right_finger_tip", "left_finger_tip"])
        self.assertEqual(tracked, ["scene", "door"])

    def test_finger_tips_are_symmetric_and_slide_with_live_gripper_q(self) -> None:
        left_open = predefined_eef_point_local_m(
            "left_finger_tip", [0.05, 0.05]
        )
        right_open = predefined_eef_point_local_m(
            "right_finger_tip", [0.05, 0.05]
        )
        left_wider = predefined_eef_point_local_m(
            "left_finger_tip", [0.065, 0.05]
        )
        right_wider = predefined_eef_point_local_m(
            "right_finger_tip", [0.05, 0.065]
        )
        self.assertGreater(left_open[1], 0.0)
        self.assertLess(right_open[1], 0.0)
        self.assertAlmostEqual(left_open[2], right_open[2], places=9)
        self.assertAlmostEqual(left_wider[1] - left_open[1], 0.015, places=9)
        self.assertAlmostEqual(right_wider[1] - right_open[1], -0.015, places=9)
        np.testing.assert_allclose(
            predefined_eef_point_local_m(
                "gripper_slide_center", [0.01, 0.08]
            ),
            predefined_eef_point_local_m(
                "gripper_slide_center", [0.05, 0.05]
            ),
        )

    def test_fingertips_are_registered_to_overlay_mesh_front_caps(self) -> None:
        left = predefined_eef_point_local_m("left_finger_tip", [0.05, 0.05])
        right = predefined_eef_point_local_m("right_finger_tip", [0.05, 0.05])
        slide = predefined_eef_point_local_m(
            "gripper_slide_center", [0.05, 0.05]
        )
        np.testing.assert_allclose(left, self._visual_mesh_tip(1), atol=1.0e-12)
        np.testing.assert_allclose(right, self._visual_mesh_tip(2), atol=1.0e-12)
        self.assertGreater(left[2] - slide[2], 0.075)
        self.assertGreater(right[2] - slide[2], 0.075)
        self.assertTrue(
            self._point_lies_on_horizontal_mesh_face(
                slide,
                self._visual_mesh_component_triangles(0),
            ),
            msg=f"gripper_slide_center is not on the overlay slide surface: {slide}",
        )

        # At the open reference posture the two physical tips form the EEF-Y
        # grasp line.  This catches both a palm-side cap and a diagonal pair.
        tip_axis = right - left
        self.assertLess(abs(float(tip_axis[0])), 0.001)
        self.assertLess(abs(float(tip_axis[2])), 1.0e-9)
        self.assertGreater(abs(float(tip_axis[1])), 0.10)

    def test_closed_fingertips_remain_physical_tip_caps(self) -> None:
        left_open = predefined_eef_point_local_m(
            "left_finger_tip", [0.05, 0.05]
        )
        right_open = predefined_eef_point_local_m(
            "right_finger_tip", [0.05, 0.05]
        )
        left_closed = predefined_eef_point_local_m(
            "left_finger_tip", [0.0, 0.0]
        )
        right_closed = predefined_eef_point_local_m(
            "right_finger_tip", [0.0, 0.0]
        )
        np.testing.assert_allclose(left_closed[[0, 2]], left_open[[0, 2]])
        np.testing.assert_allclose(right_closed[[0, 2]], right_open[[0, 2]])
        self.assertAlmostEqual(left_open[1] - left_closed[1], 0.05, places=9)
        self.assertAlmostEqual(right_closed[1] - right_open[1], 0.05, places=9)

    def test_robot_base_transform_uses_xyzw_pose(self) -> None:
        points = predefined_eef_points_robot_base_m(
            PREDEFINED_EEF_POINT_NAMES,
            eef_position_robot_base_m=[1.0, 2.0, 3.0],
            eef_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            gripper_q_m=[0.05, 0.05],
        )
        for name in PREDEFINED_EEF_POINT_NAMES:
            np.testing.assert_allclose(
                points[name],
                np.asarray([1.0, 2.0, 3.0])
                + predefined_eef_point_local_m(name, [0.05, 0.05]),
            )


class PredefinedEefContractTests(unittest.TestCase):
    def test_reserved_names_need_no_extra_public_argument(self) -> None:
        result = validate_move_tracked_point_args(
            {
                "points": [
                    {"name": "left_finger_tip"},
                    {"name": "right_finger_tip", "role": "on_hand"},
                    {"name": "gripper_slide_center"},
                ],
                "execution_mode": "plan",
            }
        )
        self.assertEqual(
            [point["name"] for point in result["points"]],
            list(PREDEFINED_EEF_POINT_NAMES),
        )

    def test_reserved_name_cannot_be_off_hand(self) -> None:
        with self.assertRaisesRegex(ValueError, "predefined EEF geometry"):
            validate_move_tracked_point_args(
                {
                    "points": [
                        {"name": "left_finger_tip", "role": "off_hand"},
                        {"name": "visual", "role": "on_hand"},
                    ]
                }
            )


class PredefinedEefSourceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _Adapter()
        self.ctx = SimpleNamespace(world=SimpleNamespace())
        self.observations = {
            "left": _observation([0.0, 0.8, 0.0]),
            "right": _observation([0.0, -0.8, 0.0]),
        }

    def _patch_observation_helpers(self):
        return mock.patch.multiple(
            tools,
            _adjust_adapter=mock.DEFAULT,
            _adjust_kinematic_observation=mock.DEFAULT,
        )

    def test_all_predefined_snapshot_never_calls_visual_tracker(self) -> None:
        manager = tools._MoveTrackedPointSourceManager(
            self.ctx,
            None,
            predefined_names=PREDEFINED_EEF_POINT_NAMES,
            episode_id="episode",
        )
        with self._patch_observation_helpers() as patched:
            patched["_adjust_adapter"].return_value = (self.adapter, 17)
            patched["_adjust_kinematic_observation"].side_effect = (
                lambda _ctx, side: self.observations[side]
            )
            snapshot = manager.observed_active_points_snapshot(
                PREDEFINED_EEF_POINT_NAMES, episode_id="episode"
            )
        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["visual_tracked_point_names"], [])
        self.assertEqual(
            snapshot["predefined_eef_point_names"],
            list(PREDEFINED_EEF_POINT_NAMES),
        )
        self.assertTrue(all(
            snapshot["entries"][name]["predefined_eef_point"]
            for name in PREDEFINED_EEF_POINT_NAMES
        ))
        lease = manager.begin_motion_retention(
            PREDEFINED_EEF_POINT_NAMES,
            episode_id="episode",
            timeout_s=10.0,
        )
        self.assertTrue(lease["ok"])
        self.assertEqual(lease["names"], [])
        self.assertTrue(manager.end_motion_retention(lease["lease_id"])["released"])

    def test_mixed_snapshot_delegates_only_visual_names(self) -> None:
        tracker = _Tracker(self.adapter)
        manager = tools._MoveTrackedPointSourceManager(
            self.ctx,
            tracker,
            predefined_names=["left_finger_tip"],
            episode_id="episode",
        )
        with self._patch_observation_helpers() as patched:
            patched["_adjust_adapter"].return_value = (self.adapter, 17)
            patched["_adjust_kinematic_observation"].side_effect = (
                lambda _ctx, side: self.observations[side]
            )
            snapshot = manager.observed_active_points_snapshot(
                ["left_finger_tip", "door_lock"], episode_id="episode"
            )
        self.assertTrue(snapshot["ok"])
        self.assertEqual(tracker.snapshot_names, [["door_lock"]])
        self.assertEqual(list(snapshot["entries"]), ["left_finger_tip", "door_lock"])
        retention = manager.begin_motion_retention(
            ["left_finger_tip", "door_lock"],
            episode_id="episode",
            timeout_s=10.0,
        )
        self.assertEqual(tracker.retention_names, [["door_lock"]])
        self.assertEqual(retention["lease_id"], "visual_lease")

    def test_arm_selection_uses_nearest_off_hand_reference(self) -> None:
        arm, report = tools._move_tracked_select_predefined_eef_arm(
            points=[
                {"name": "left_finger_tip", "role": "on_hand"},
                {"name": "handle", "role": "off_hand"},
            ],
            roles_by_name={"left_finger_tip": "on_hand", "handle": "off_hand"},
            observations=self.observations,
            visual_entries={
                "handle": {"xyz_in_robot_base_coord_m": [0.0, 0.82, 0.0]}
            },
        )
        self.assertEqual(arm, "left")
        self.assertEqual(
            report["selection_basis"], "nearest_to_off_hand_reference_geometry"
        )


class PredefinedEefFullToolTests(unittest.TestCase):
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
        arm_q = np.asarray(tools.GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES[f"gripper_{side}_qpos"]] = [0.02, 0.02]
            proprio[PROPRIO_SLICES[f"gripper_{side}_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.45, -0.4, 0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        return adapter, world

    @staticmethod
    def _drive(adapter, generator):
        actions = []
        for raw_action in generator:
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    def test_exec_uses_predefined_point_without_tracker_registration(self) -> None:
        adapter, world = self._ready_world()
        result = {}
        ctx = SimpleNamespace(
            world=world,
            task_name="chopping_wood",
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )
        observed = tools._adjust_kinematic_observation(ctx, "right")
        current = predefined_eef_points_robot_base_m(
            ["right_finger_tip"],
            eef_position_robot_base_m=observed["position"],
            eef_quaternion_xyzw=observed["quaternion"],
            gripper_q_m=observed["gripper"],
        )["right_finger_tip"]
        target = current + np.asarray([0.004, 0.0, 0.0], dtype=np.float64)
        self.assertFalse(hasattr(world, "_official_tracked_object_distances"))

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = self._drive(
                adapter,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=[
                        {
                            "name": "right_finger_tip",
                            "role": "on_hand",
                            "target_xyz_m": target.astype(float).tolist(),
                        }
                    ],
                    max_steps=180,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["arm"], "right")
        self.assertEqual(
            result["predefined_eef_point_names"], ["right_finger_tip"]
        )
        self.assertEqual(result["visual_tracked_point_names"], [])
        self.assertTrue(result["final_constraints"]["ok"])
        self.assertGreaterEqual(len(actions), 2)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
