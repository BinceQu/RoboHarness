from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .runtime_fixture import InterfaceRuntime
from tools.util.x5_kinematics import (
    calibrated_pose_for_joints,
)
from tools.util.capabilities import PUBLIC_TOOLS

from .helpers import (
    apply_action,
    apply_action_fraction,
    drive_job,
    fake_observation,
    top_down_grasp_observation,
)


class OfficialV2ToolAcceptanceTest(unittest.TestCase):
    """One independently reported acceptance test for every public v2 tool."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.runtime = InterfaceRuntime(
            task_name="acceptance_task",
            runs_root=self.temporary.name,
        )
        self.observation = fake_observation()
        self.runtime.update_observation(self.observation)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_tool(self, name: str, arguments: dict):
        result, self.observation, actions = drive_job(
            self.runtime,
            self.observation,
            name,
            arguments,
        )
        self.assertEqual(result["tool"], name)
        self.assertEqual(result["tool_version"], "official_v2")
        self.assertEqual(result["backend"], "robodojo")
        self.assertFalse(result["direct_simulator_access"])
        self.assertTrue(result["exit_capture_attempted"])
        observation = result.get("observation")
        self.assertIsInstance(observation, dict, result)
        self.assertTrue(observation["ok"], observation)
        self.assertTrue(observation["image_id"])
        self.assertTrue(observation["rgb_main"].startswith("data:image/png;base64,"))
        self.assertEqual(result["image_id"], observation["image_id"])

        if name == "capture_left_wrist_camera":
            expected_feed = "left_wrist"
        elif name == "capture_right_wrist_camera":
            expected_feed = "right_wrist"
        elif name == "adjust_left_eef_pose_in_wrist_frame":
            expected_feed = "left_wrist"
        elif name == "adjust_right_eef_pose_in_wrist_frame":
            expected_feed = "right_wrist"
        elif name in ("open_gripper", "close_gripper"):
            expected_feed = f"{arguments.get('arm', 'right')}_wrist"
        elif name == "exec_plan_pose":
            expected_arm = result.get("arm") or arguments.get("arm") or "right"
            expected_feed = f"{expected_arm}_wrist"
        else:
            expected_feed = "head"
        self.assertEqual(observation["feed"], expected_feed, result)
        self.assertEqual(result["feed"], expected_feed, result)
        if expected_feed.endswith("_wrist"):
            overlay = observation["grasp_zone_overlay"]
            self.assertTrue(overlay["ok"], result)
            self.assertTrue(
                overlay["robot_mask_used"], result
            )
            self.assertIn(
                "self_rejected_pixel_count", overlay
            )
            registration = overlay["gripper_depth_registration"]
            self.assertEqual(
                registration["source"],
                "evaluator_wrist_depth_registered_official_x5_finger_mesh",
            )
            self.assertFalse(registration["segmentation_used"])
            self.assertFalse(registration["simulator_state_used"])
            self.assertNotEqual(
                observation["rgb_main_path"], observation["rgb_path"]
            )
            self.assertIn("red_grasp_volume_mask", observation["modalities"])
            self.assertTrue(Path(observation["red_mask_path"]).is_file())
            self.assertTrue(
                Path(observation["grasp_volume_three_views_path"]).is_file()
            )
            self.assertEqual(
                [
                    view["name"]
                    for view in observation["grasp_volume_three_views_metadata"][
                        "views"
                    ]
                ],
                ["top", "side", "front"],
            )
            self.assertEqual(
                observation["grasp_volume_three_views_metadata"][
                    "gripper_geometry_qpos_normalized"
                ],
                overlay["gripper_geometry_qpos_normalized"],
            )
            self.assertEqual(
                observation["grasp_volume_three_views_metadata"][
                    "gripper_qpos_normalized"
                ],
                overlay["gripper_qpos_normalized"],
            )
            self.assertEqual(
                result["grasp_volume_three_views_path"],
                observation["grasp_volume_three_views_path"],
            )
        return result, actions

    def capture_head(self, session_id: str = "acceptance"):
        result, actions = self.run_tool(
            "capture_head_camera", {"session_id": session_id}
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(actions, [])
        return result

    def common_click(self, session_id: str = "acceptance"):
        capture = self.capture_head(session_id)
        return {
            "session_id": session_id,
            "image_id": capture["image_id"],
            "u": 500,
            "v": 500,
        }

    def assert_motion(self, result: dict, actions: list[str]) -> None:
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["action_steps"], 0)
        self.assertEqual(result["action_steps"], len(actions))
        self.assertEqual(set(actions), {"joint"})

    def assert_unsupported(self, name: str, arguments: dict) -> None:
        result, actions = self.run_tool(name, arguments)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "unsupported_embodiment")
        self.assertFalse(result["action_emitted"])
        self.assertEqual(actions, [])

    def test_01_capture_head_camera(self) -> None:
        result = self.capture_head("head")
        self.assertEqual(result["feed"], "head")
        self.assertIn("depth", result["modalities"])

    def test_02_capture_left_wrist_camera(self) -> None:
        result, actions = self.run_tool(
            "capture_left_wrist_camera", {"session_id": "left_wrist"}
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["feed"], "left_wrist")
        self.assertEqual(actions, [])
        self.assertTrue(result["grasp_zone_overlay"]["ok"], result)
        self.assertIn("red_grasp_volume_mask", result["modalities"])
        self.assertNotEqual(result["rgb_main_path"], result["rgb_path"])
        self.assertTrue(Path(result["red_mask_path"]).is_file())
        self.assertTrue(Path(result["grasp_volume_three_views_path"]).is_file())

    def test_03_capture_right_wrist_camera(self) -> None:
        result, actions = self.run_tool(
            "capture_right_wrist_camera", {"session_id": "right_wrist"}
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["feed"], "right_wrist")
        self.assertEqual(actions, [])
        self.assertTrue(result["grasp_zone_overlay"]["ok"], result)
        self.assertIn("red_grasp_volume_mask", result["modalities"])
        self.assertNotEqual(result["rgb_main_path"], result["rgb_path"])
        self.assertTrue(Path(result["red_mask_path"]).is_file())
        self.assertTrue(Path(result["grasp_volume_three_views_path"]).is_file())

    def test_04_read_depth(self) -> None:
        click = self.common_click("depth")
        result, actions = self.run_tool("read_depth", click)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["depth_m"], 1.0)
        self.assertEqual(result["unit"], "m")
        self.assertEqual(actions, [])

    def test_05_track_object_distance(self) -> None:
        click = self.common_click("track")
        result, actions = self.run_tool(
            "track_object_distance",
            {
                "session_id": click["session_id"],
                "image_id": click["image_id"],
                "points": [{"name": "agent_point", "u": 500, "v": 500}],
            },
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["identity_source"], "model_annotation")
        self.assertFalse(result["identity_verified"])
        self.assertEqual(
            result["frame_alignment"],
            "required_session_id_image_id_and_tracker_frame_sha256",
        )
        self.assertEqual(result["replacement_policy"], "complete_named_set_on_success")
        self.assertEqual(result["cleared_names"], [])
        self.assertEqual(actions, [])

    def test_07_move_chassis_to_floor_point(self) -> None:
        self.assert_unsupported(
            "move_chassis_to_floor_point",
            {"session_id": "fixed", "image_id": "image", "u": 500, "v": 500},
        )

    def test_08_adjust_chassis(self) -> None:
        self.assert_unsupported("adjust_chassis", {"forward": 0.1})

    def test_09_adjust_pitch(self) -> None:
        self.assert_unsupported("adjust_pitch", {"degree": 5.0})

    def test_10_adjust_height(self) -> None:
        self.assert_unsupported("adjust_height", {"upward": 0.05})

    def test_11_spin_to_facing_point(self) -> None:
        self.assert_unsupported("spin_to_facing_point", {"u": 600, "v": 500})

    def test_12_open_gripper(self) -> None:
        result, actions = self.run_tool("open_gripper", {"arm": "left"})
        self.assert_motion(result, actions)
        self.assertLessEqual(result["tracking_error"], result["tracking_tolerance"])

    def test_13_close_gripper(self) -> None:
        result, actions = self.run_tool("close_gripper", {"arm": "right"})
        self.assert_motion(result, actions)
        self.assertEqual(result["grasp_detected"], "unknown")

    def assert_nonzero_eef_adjust(self, name: str, arguments: dict) -> None:
        result, actions = self.run_tool(name, arguments)
        self.assert_motion(result, actions)
        start = np.asarray(result["start_eef_pose"][:3])
        target = np.asarray(result["target_eef_pose"][:3])
        self.assertGreater(float(np.linalg.norm(target - start)), 0.005)
        self.assertLessEqual(
            result["position_error_m"], result["position_tolerance_m"]
        )
        self.assertIn("observed_joint_position", result["tracking_trace"][0])

    def test_14_adjust_left_eef_pose_in_head_frame(self) -> None:
        self.assert_nonzero_eef_adjust(
            "adjust_left_eef_pose_in_head_frame", {"y": 0.015}
        )

    def test_15_adjust_right_eef_pose_in_head_frame(self) -> None:
        self.assert_nonzero_eef_adjust(
            "adjust_right_eef_pose_in_head_frame", {"y": 0.015}
        )

    def test_16_adjust_left_eef_pose_in_wrist_frame(self) -> None:
        self.assert_nonzero_eef_adjust(
            "adjust_left_eef_pose_in_wrist_frame", {"leftward": 0.015}
        )

    def test_17_adjust_right_eef_pose_in_wrist_frame(self) -> None:
        self.assert_nonzero_eef_adjust(
            "adjust_right_eef_pose_in_wrist_frame", {"leftward": 0.015}
        )

    def test_18_move_point_to_point(self) -> None:
        capture = self.capture_head("point_to_point")
        result, actions = self.run_tool(
            "move_point_to_point",
            {
                "session_id": "point_to_point",
                "image_id": capture["image_id"],
                "points": [{"u": 400, "v": 500}, {"u": 600, "v": 500}],
            },
        )
        self.assert_motion(result, actions)
        self.assertGreater(
            float(np.linalg.norm(result["delta_robodojo_env_m"])), 0.005
        )

    def test_19_move_tracked_point(self) -> None:
        capture = self.capture_head("move_tracked")
        tracked, track_actions = self.run_tool(
            "track_object_distance",
            {
                "session_id": "move_tracked",
                "image_id": capture["image_id"],
                "points": [{"name": "move_marker", "u": 500, "v": 500}],
            },
        )
        self.assertTrue(tracked["ok"], tracked)
        self.assertEqual(track_actions, [])
        result, actions = self.run_tool(
            "move_tracked_point",
            {
                "points": [
                    {
                        "name": "move_marker",
                        "target_xyz_m": ["?", "?", "?"],
                    }
                ],
                "max_steps": 40,
                "timeout_s": 15.0,
            },
        )
        self.assertTrue(result["ok"], result)
        self.assertGreater(len(actions), 0)
        self.assertEqual(set(actions), {"joint"})
        self.assertGreater(result["motion"]["action_steps"], 0)
        self.assertTrue(result["start_state_validation"]["ok"], result)
        self.assertTrue(result["motion"]["ok"], result)
        self.assertTrue(result["stability"]["ok"], result)
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        self.assertTrue(result["final_anchor_ok"], result)
        self.assertTrue(Path(result["plan_record_path"]).is_file())

    def test_20_plan_eef_translation_to_uvd_point(self) -> None:
        result, actions = self.run_tool(
            "plan_eef_translation_to_uvd_point", self.common_click("translation")
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["trajectory_waypoint_count"], 1)
        self.assertEqual(actions, [])

    def test_22_move_to_reach_point(self) -> None:
        result, actions = self.run_tool(
            "move_to_reach_point", self.common_click("reach")
        )
        self.assertEqual(result["ok"], result["already_reachable"])
        self.assertFalse(result["action_emitted"])
        self.assertEqual(actions, [])

    def test_23_measure_shoulder_distance(self) -> None:
        result, actions = self.run_tool(
            "measure_shoulder_distance", self.common_click("distance")
        )
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["left_shoulder_to_object_m"], 0.0)
        self.assertIn(result["recommended_arm"], ("left", "right"))
        self.assertEqual(actions, [])

    def test_24_plan_grasp_point_filter_rgbd_lite(self) -> None:
        self.observation = top_down_grasp_observation()
        self.runtime.update_observation(self.observation)
        result, actions = self.run_tool(
            "plan_grasp_point_filter_rgbd_lite", self.common_click("grasp_lite")
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["planner_variant"],
            "rgbd_lite_1120_volume_paired_x5_ik",
        )
        self.assertEqual(result["candidate_count"], 1120)
        surface_anchor = result["plan_audit"]["surface_anchor"]
        self.assertEqual(
            surface_anchor["selection"],
            "original_visible_depth_click_for_vertical_pinch",
        )
        self.assertAlmostEqual(surface_anchor["diameter_m"], 0.030)
        self.assertGreater(surface_anchor["sphere_surface_point_count"], 0)
        self.assertEqual(result["input_pixel_uv"], [16, 12])
        self.assertEqual(result["pixel_uv"], surface_anchor["selected_pixel_uv"])
        self.assertGreater(result["grasp_vol_cm3"], 0.0)
        self.assertLessEqual(result["overlap_vol_cm3"], 0.30)
        self.assertLess(result["inflated_overlap_vol_cm3"], 1.6)
        validation = result["ik_solution"]["volume_validation"]
        self.assertTrue(validation["ok"], validation)
        self.assertTrue(validation["ideal_ok"], validation)
        self.assertTrue(validation["final_q_fk_ok"], validation)
        self.assertEqual(
            result["execution_pose_source"],
            "capture_calibrated_fk_of_signed_final_q",
        )
        np.testing.assert_allclose(
            [*result["eef_pose"]["pos"], *result["eef_pose"]["quat"]],
            result["ik_solution"]["final_q_fk_pose"],
            atol=1e-10,
        )
        self.assertAlmostEqual(
            result["grasp_vol_cm3"],
            validation["final_q_fk"]["grasp_vol_cm3"],
        )
        self.assertAlmostEqual(
            result["robust_grasp_vol_cm3"],
            min(
                validation["ideal"]["grasp_vol_cm3"],
                validation["final_q_fk"]["grasp_vol_cm3"],
            ),
        )
        self.assertTrue(result["gripper_visualization"]["ok"], result)
        self.assertEqual(result["gripper_visualization"]["color"], "red")
        self.assertGreater(
            result["gripper_visualization"]["visible_pixel_count"], 0
        )
        self.assertTrue(result["render_image"].startswith("data:image/png;base64,"))
        three_views = result["grasp_plan_three_views_metadata"]
        self.assertTrue(three_views["ok"], three_views)
        self.assertEqual(
            [view["name"] for view in three_views["views"]],
            ["front", "side", "top"],
        )
        self.assertTrue(three_views["metrics_match_planner"], three_views)
        self.assertEqual(three_views["opening_query_version"],
                         result["grip_fit"]["opening_query_version"])
        self.assertEqual(three_views["surface_query_radius_cells"], 0)
        self.assertTrue(result["grip_fit"]["raw_surface_evidence"]["ok"])
        self.assertFalse(result["grip_fit"]["raw_surface_evidence"]["raw_voxel_proxy_is_material_volume"])
        self.assertFalse(three_views["segmentation_used"])
        self.assertFalse(three_views["simulator_geometry_used"])
        self.assertTrue(Path(result["render_image_3d_path"]).is_file())
        self.assertEqual(
            result["render_image_3d"], result["render_image_3d_path"]
        )
        self.assertEqual(
            result["3d_point_render"], result["render_image_3d_path"]
        )
        self.assertEqual(
            result["debug_images"]["rgbd_mesh_three_views"],
            result["render_image_3d_path"],
        )
        self.assertTrue(
            result["grasp_volume_three_views"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertEqual(actions, [])

    def test_25_plan_press_point(self) -> None:
        result, actions = self.run_tool(
            "plan_press_point", self.common_click("press")
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["trajectory_waypoint_count"], 2)
        self.assertEqual(actions, [])

    def test_26_exec_plan_pose(self) -> None:
        self.observation = top_down_grasp_observation()
        self.runtime.update_observation(self.observation)
        click = self.common_click("execute")
        plan, _actions = self.run_tool(
            "plan_grasp_point_filter_rgbd_lite", click
        )
        gripper_before = self.observation["state"][
            f"{plan['arm']}_ee_joint_state"
        ].copy()
        result, actions = self.run_tool(
            "exec_plan_pose",
            {"session_id": "execute", "plan_id": plan["plan_id"]},
        )
        self.assert_motion(result, actions)
        self.assertEqual(result["trajectory_digest"], plan["trajectory_digest"])
        self.assertEqual(len(result["segments_executed"]), 2)
        self.assertEqual(result["position_tolerance_m"], 0.012)
        self.assertEqual(
            result["execution_controller"],
            "immutable_joint_target_replay",
        )
        self.assertEqual(result["exec_sequence"], "move_only")
        self.assertEqual(result["compat_gripper_transition"], "preserve")
        self.assertFalse(result["gripper_transition_handled"])
        self.assertEqual(result["gripper_transitions"], [])
        np.testing.assert_allclose(result["gripper_qpos_before"], gripper_before)
        np.testing.assert_allclose(result["gripper_qpos_after"], gripper_before)
        np.testing.assert_allclose(
            self.observation["state"][f"{plan['arm']}_ee_joint_state"],
            gripper_before,
        )
        self.assertTrue(
            all(
                segment["position_error_m"]
                <= result["position_tolerance_m"]
                for segment in result["segments_executed"]
            ),
            result,
        )
        self.assertTrue(
            all(
                segment["joint_target_source"]
                == "signed_planned_joint_target"
                for segment in result["segments_executed"]
            ),
            result,
        )
        self.assertEqual(
            result["segments_executed"][-1]["target_pose_source"],
            "capture_calibrated_fk_of_signed_final_q",
        )
        self.assertEqual(
            result["segments_executed"][-1][
                "control_tracking_position_error_m"
            ],
            result["segments_executed"][-1]["position_error_m"],
        )
        np.testing.assert_allclose(
            result["segments_executed"][-1]["target_pose"],
            [*plan["eef_pose"]["pos"], *plan["eef_pose"]["quat"]],
            atol=1e-10,
        )

    def test_exec_replays_signed_targets_with_slow_joint_tracking(self) -> None:
        self.observation = top_down_grasp_observation()
        self.runtime.update_observation(self.observation)
        plan, _actions = self.run_tool(
            "plan_grasp_point_filter_rgbd_lite",
            self.common_click("execute_slow_tracking"),
        )
        # Exercise slow servo tracking from a nearby, measured starting state.
        # The downward fixture requires a much longer wrist rotation than the
        # old upward-looking scene; its full 0.4-gain transit exceeds the hard
        # 50-action budget before the second waypoint. Keep that budget intact.
        warm_start = {
            key: np.asarray(self.observation["state"][key]).copy()
            for key in ("left_arm_joint_state", "right_arm_joint_state",
                        "left_ee_joint_state", "right_ee_joint_state")
        }
        arm_key = f"{plan['arm']}_arm_joint_state"
        warm_start[arm_key] = .4 * np.asarray(plan["ik_solution"]["safe_q"])
        self.observation = apply_action(self.observation, warm_start)
        self.runtime.update_observation(self.observation)

        def slow_tracking(observation, action):
            return apply_action_fraction(observation, action, fraction=0.4)

        with patch(
            "tools.exec_plan_pose."
            "solve_calibrated_x5_ik",
            side_effect=AssertionError(
                "exec must not resolve IK for a signed joint target"
            ),
        ) as runtime_ik:
            result, self.observation, action_types = drive_job(
                self.runtime,
                self.observation,
                "exec_plan_pose",
                {
                    "session_id": "execute_slow_tracking",
                    "plan_id": plan["plan_id"],
                },
                action_updater=slow_tracking,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(runtime_ik.call_count, 0)
        self.assertTrue(action_types)
        self.assertTrue(all(kind == "joint" for kind in action_types))
        self.assertGreater(result["action_steps"], 2)
        for segment in result["segments_executed"]:
            self.assertEqual(
                segment["joint_target_source"],
                "signed_planned_joint_target",
            )
            self.assertLessEqual(
                segment["terminal_joint_error_rad"],
                segment["joint_settle_tolerance_rad"],
            )

    def test_exec_reports_blocked_signed_joint_target(self) -> None:
        self.observation = top_down_grasp_observation()
        self.runtime.update_observation(self.observation)
        plan, _actions = self.run_tool(
            "plan_grasp_point_filter_rgbd_lite",
            self.common_click("execute_blocked_target"),
        )

        def ignore_joint_motion(observation, action):
            updated = deepcopy(observation)
            for key, value in action.items():
                updated["action"][key] = np.asarray(value).copy()
            return updated

        result, self.observation, action_types = drive_job(
            self.runtime,
            self.observation,
            "exec_plan_pose",
            {
                "session_id": "execute_blocked_target",
                "plan_id": plan["plan_id"],
            },
            action_updater=ignore_joint_motion,
        )
        self.assertFalse(result["ok"], result)
        self.assertTrue(action_types)
        segment = result["segments_executed"][-1]
        self.assertEqual(segment["stop_reason"], "joint_target_blocked")
        self.assertGreater(
            segment["max_joint_tracking_error_rad"],
            segment["joint_tracking_blocked_threshold_rad"],
        )

    def test_27_set_arm_to_grasp_position(self) -> None:
        displaced_q = np.array(
            [0.08, 0.10, -0.16, 0.06, -0.08, 0.04], dtype=np.float32
        )
        for side in ("left", "right"):
            reference_q = np.asarray(
                self.observation["state"][f"{side}_arm_joint_state"]
            )
            reference_pose = np.asarray(
                self.observation["state"][f"{side}_ee_pose"]
            )
            self.observation["state"][f"{side}_arm_joint_state"] = (
                displaced_q.copy()
            )
            self.observation["action"][f"{side}_arm_joint_state"] = (
                displaced_q.copy()
            )
            self.observation["state"][f"{side}_ee_pose"] = (
                calibrated_pose_for_joints(
                    reference_q,
                    reference_pose,
                    displaced_q,
                ).astype(np.float32)
            )
        self.runtime.update_observation(self.observation)
        result, actions = self.run_tool(
            "set_arm_to_grasp_position",
            {
                "arm": "both",
                "keep_ori_arm": "none",
                "gripper": "open",
                "max_dq_per_step": 0.2,
            },
        )
        self.assert_motion(result, actions)
        self.assertEqual(result["arms"], ["left", "right"])
        self.assertLessEqual(result["max_abs_error_rad"], result["tolerance_rad"])

    def test_set_arm_retraces_real_deep_grasp_path_before_neutral(self) -> None:
        safe_q = np.array(
            [-1.46428, 3.80132, 4.57430, -3.16735, -1.12348, -2.44564],
            dtype=np.float32,
        )
        deep_q = np.array(
            [-0.96083, 3.41994, 4.28726, -2.18490, 1.19705, -2.15099],
            dtype=np.float32,
        )

        outward = [
            *np.linspace(np.zeros(6), safe_q, 31, dtype=np.float32)[1:],
            *np.linspace(safe_q, deep_q, 17, dtype=np.float32)[1:],
        ]
        for waypoint in outward:
            action = self.runtime.world.make_joint_action(left_arm=waypoint)
            self.runtime.world.mark_action_dispatched(
                action,
                tool_name="exec_plan_pose",
            )
            self.observation = apply_action(self.observation, action)
            self.runtime.update_observation(self.observation)

        unlocked = False
        commands: list[np.ndarray] = []

        def block_direct_neutral_path(observation, action):
            nonlocal unlocked
            current = np.asarray(
                observation["state"]["left_arm_joint_state"],
                dtype=np.float64,
            )
            command = np.asarray(
                action["left_arm_joint_state"], dtype=np.float64
            )
            commands.append(command.copy())
            # The real failure released only when the final grasp descent was
            # reversed: J4 and J5 first move away from the direct-zero route.
            if command[3] < current[3] - 1.0e-3 and command[4] < current[4] - 1.0e-3:
                unlocked = True
            updated = apply_action(observation, action)
            if not unlocked:
                observed_q = np.asarray(
                    updated["state"]["left_arm_joint_state"],
                    dtype=np.float32,
                )
                observed_q[3:5] = current[3:5]
                updated["state"]["left_arm_joint_state"] = observed_q
                updated["state"]["left_ee_pose"] = calibrated_pose_for_joints(
                    current,
                    observation["state"]["left_ee_pose"],
                    observed_q,
                ).astype(np.float32)
            return updated

        result, self.observation, action_types = drive_job(
            self.runtime,
            self.observation,
            "set_arm_to_grasp_position",
            {
                "arm": "left",
                "gripper": "keep",
                "max_dq_per_step": 0.12,
                "timeout_s": 45.0,
            },
            action_updater=block_direct_neutral_path,
        )
        self.assertTrue(result["ok"], result)
        self.assertTrue(unlocked)
        self.assertTrue(result["retrace_used"], result)
        self.assertEqual(
            result["retrace_source"],
            "policy_recorded_evaluator_observed_qpos",
        )
        self.assertGreater(result["retrace_waypoints_completed"], 1)
        self.assertEqual(set(action_types), {"joint"})
        self.assertLess(commands[0][3], deep_q[3])
        self.assertLess(commands[0][4], deep_q[4])
        self.assertLessEqual(
            result["max_abs_error_rad"], result["tolerance_rad"]
        )
        self.assertEqual(
            self.runtime.world.arm_retrace_waypoints(
                "left",
                np.zeros(6),
                anchor_tolerance_rad=0.12,
                max_waypoints=56,
            ),
            [],
        )

    def test_set_arm_reports_the_specific_blocked_joint(self) -> None:
        reference_q = np.asarray(
            self.observation["state"]["left_arm_joint_state"]
        )
        reference_pose = np.asarray(
            self.observation["state"]["left_ee_pose"]
        )
        displaced = np.array(
            [0.30, 0.25, 0.20, -0.50, 0.15, -0.10], dtype=np.float32
        )
        self.observation["state"]["left_arm_joint_state"] = displaced.copy()
        self.observation["action"]["left_arm_joint_state"] = displaced.copy()
        self.observation["state"]["left_ee_pose"] = calibrated_pose_for_joints(
            reference_q,
            reference_pose,
            displaced,
        ).astype(np.float32)
        self.runtime.update_observation(self.observation)

        def freeze_j4(observation, action):
            current_j4 = float(
                observation["state"]["left_arm_joint_state"][3]
            )
            updated = apply_action(observation, action)
            observed_q = np.asarray(
                updated["state"]["left_arm_joint_state"], dtype=np.float32
            )
            observed_q[3] = current_j4
            updated["state"]["left_arm_joint_state"] = observed_q
            updated["state"]["left_ee_pose"] = calibrated_pose_for_joints(
                observation["state"]["left_arm_joint_state"],
                observation["state"]["left_ee_pose"],
                observed_q,
            ).astype(np.float32)
            return updated

        result, self.observation, _action_types = drive_job(
            self.runtime,
            self.observation,
            "set_arm_to_grasp_position",
            {"arm": "left", "timeout_s": 45.0},
            action_updater=freeze_j4,
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["stop_reason"], "joint_tracking_blocked")
        self.assertEqual(result["failure_stage"], "final_neutral_servo")
        self.assertEqual(
            [entry["name"] for entry in result["stalled_joints_by_arm"]["left"]],
            ["J4"],
        )
        self.assertIn("left:J4", result["error"])

    def test_28_reset_body(self) -> None:
        self.assert_unsupported("reset_body", {})

    def test_manifest_has_exactly_one_case_per_public_tool(self) -> None:
        methods = {
            name.removeprefix("test_").split("_", 1)[1]
            for name in dir(type(self))
            if name.startswith("test_")
            and name[5:7].isdigit()
        }
        self.assertEqual(methods, set(PUBLIC_TOOLS))

    def test_cartesian_servo_reports_blocked_joint_tracking(self) -> None:
        def ignore_joint_motion(observation, action):
            updated = deepcopy(observation)
            for key, value in action.items():
                updated["action"][key] = np.asarray(value).copy()
            return updated

        request_id = self.runtime.submit(
            "adjust_right_eef_pose_in_head_frame", {"y": 0.03}
        )
        actions = []
        for _ in range(16):
            pending = self.runtime.get_action()
            if pending:
                self.assertEqual(len(pending), 1)
                actions.append(pending[0])
                self.observation = ignore_joint_motion(
                    self.observation, pending[0]
                )
                self.runtime.update_observation(self.observation)
            if self.runtime.scheduler.job_status(request_id)["done"]:
                break
        result = self.runtime.wait(request_id, 0.1)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["stop_reason"], "joint_tracking_blocked")
        self.assertGreater(
            result["max_joint_tracking_error_rad"],
            result["joint_tracking_blocked_threshold_rad"],
        )
        self.assertGreater(len(actions), 1)

    def test_failed_exec_still_returns_requested_wrist_image(self) -> None:
        result, actions = self.run_tool(
            "exec_plan_pose",
            {
                "session_id": "failed_exec_capture",
                "plan_id": "missing_plan",
                "arm": "left",
            },
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(actions, [])
        self.assertEqual(result["observation"]["feed"], "left_wrist")
        self.assertEqual(
            result["exit_capture_skill"], "capture_left_wrist_camera"
        )


if __name__ == "__main__":
    unittest.main()
