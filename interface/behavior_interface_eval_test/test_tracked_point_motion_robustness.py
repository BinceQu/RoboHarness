"""Regression tests for tracked-point path recovery and fallback behavior."""

from __future__ import annotations

import unittest
import time
from unittest import mock

import numpy as np

import behavior_interface_eval_test.test_official_move_tracked_point as official_test
from behavior_interface_eval_test.robot_contract import (
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_SLICES,
)
import behavior_interface_eval_test.tool.official_v2.tools as official_tools
import behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local as tracked_motion
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    PlanningDeadlineExceeded,
)


class TrackedPointMotionRobustnessTest(unittest.TestCase):
    def _execution_fixture(self):
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        observations = {
            side: official_tools._adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture = official_tools._move_tracked_capture_state(ctx, observations)
        return adapter, world, ctx, result, capture

    def test_failed_intermediate_ik_knot_is_dropped_and_path_remains_local(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (_position, _quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.01

        def failed_ik(*_args, **_kwargs):
            return q_start.copy(), {
                "ok": False,
                "pos_err_m": 0.10,
                "ori_err_deg": 20.0,
                "solver": "synthetic_nonconvergence",
            }

        with mock.patch.object(
            tracked_motion,
            "solve_pose_target",
            side_effect=failed_ik,
        ):
            path = tracked_motion._plan_cartesian_trajectory_once(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_final,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_joint_step_rad=0.018,
                max_waypoints=40,
                translation_step_m=0.001,
                orientation_step_deg=1.0,
            )
        self.assertTrue(path["dropped_knot_count"] > 0, path)
        self.assertEqual(path["waypoints"][-1], q_final.astype(float).tolist())
        self.assertTrue(path["local_fk_finite_checked"])

    def test_endpoint_is_never_silently_dropped_on_large_same_branch_step(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, _pose = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 1.30

        # Use one coarse knot and deliberately broad path tolerances so this
        # isolates the endpoint bookkeeping rule.  A large endpoint step is
        # retained for full local path validation; it must never turn into a
        # trajectory that ends at q_start.
        path = tracked_motion._plan_cartesian_trajectory_once(
            state=state,
            arm="left",
            q_start=q_start,
            q_final=q_final,
            pos_tol_m=2.0,
            ori_tol_deg=180.0,
            max_joint_step_rad=2.0,
            max_waypoints=20,
            translation_step_m=100.0,
            orientation_step_deg=360.0,
        )
        self.assertTrue(path["endpoint_branch_jump_exceeded"], path)
        np.testing.assert_allclose(
            np.asarray(path["waypoints"][-1], dtype=np.float64),
            q_final,
            atol=1.0e-12,
        )

    def test_joint_fallback_is_available_after_cartesian_failure(self) -> None:
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (position, _quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        source = np.asarray(position, dtype=np.float64).reshape(1, 3)
        manager = official_test._RigidTrackedManager(world, adapter, {"known": source[0]})
        world._official_tracked_object_distances = manager
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.01
        target_position = np.asarray(position, dtype=np.float64) + np.array([0.005, 0.0, 0.0])

        with mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=ValueError("synthetic Cartesian continuation failure"),
        ):
            result = official_tools._move_tracked_plan_frozen(
                state=state,
                arm="left",
                q_start=q_start,
                source_points=source,
                target_points=[
                    {"name": "known", "target_xyz_m": target_position.tolist()}
                ],
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_steps=120,
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["path_plan"]["path_mode"], "joint_space_fallback")
        self.assertTrue(result["path_plan"]["joint_limits_checked"])
        self.assertTrue(result["path_plan"]["local_fk_finite_checked"])
        self.assertFalse(result["path_plan"]["simulator_collision_checked"])

    def test_planning_deadline_is_not_swallowed_by_density_retries(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, _pose = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.08

        with self.assertRaises(PlanningDeadlineExceeded):
            tracked_motion.plan_cartesian_trajectory(
                state=state,
                arm="left",
                q_start=q_start,
                q_final=q_final,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_joint_step_rad=0.05,
                max_waypoints=100,
                planning_deadline_monotonic=time.monotonic() - 1.0,
            )

    def test_pose_solver_propagates_cooperative_deadline_from_residual(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (position, quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        callback_calls = 0

        def stop_after_a_few_residuals() -> None:
            nonlocal callback_calls
            callback_calls += 1
            if callback_calls >= 3:
                raise PlanningDeadlineExceeded("synthetic residual deadline")

        with self.assertRaises(PlanningDeadlineExceeded):
            tracked_motion.solve_pose_target(
                state,
                "left",
                q_start,
                target_position=position,
                target_quaternion=quaternion,
                max_iterations=120,
                check_requested=stop_after_a_few_residuals,
            )
        self.assertGreaterEqual(callback_calls, 3)

    def test_cartesian_ik_deadline_is_not_swallowed_by_legacy_retry(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, _pose = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.50
        with mock.patch.object(
            tracked_motion,
            "solve_pose_target",
            side_effect=PlanningDeadlineExceeded("synthetic IK deadline"),
        ):
            with self.assertRaises(PlanningDeadlineExceeded):
                tracked_motion._plan_cartesian_trajectory_once(
                    state=state,
                    arm="left",
                    q_start=q_start,
                    q_final=q_final,
                    pos_tol_m=0.012,
                    ori_tol_deg=5.0,
                    max_joint_step_rad=0.05,
                    max_waypoints=80,
                    translation_step_m=0.05,
                    orientation_step_deg=10.0,
                    planning_deadline_monotonic=time.monotonic() + 30.0,
                )

    def test_soft_deadline_uses_verified_endpoint_joint_fallback(self) -> None:
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (position, _quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        source = np.asarray(position, dtype=np.float64).reshape(1, 3)
        world._official_tracked_object_distances = official_test._RigidTrackedManager(
            world, adapter, {"known": source[0]}
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = source[0] + np.array([0.005, 0.0, 0.0])

        with mock.patch.object(
            official_tools,
            "_plan_tracked_cartesian_trajectory",
            side_effect=PlanningDeadlineExceeded("synthetic planning budget"),
        ) as path_solver:
            result = official_tools._move_tracked_plan_frozen(
                state=state,
                arm="left",
                q_start=q_start,
                source_points=source,
                target_points=[
                    {"name": "known", "target_xyz_m": target.tolist()}
                ],
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
                max_steps=120,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["path_plan"]["path_mode"], "joint_space_fallback")
        self.assertTrue(result["path_plan"]["planning_budget_exhausted"])
        self.assertTrue(path_solver.called)

    def test_live_cartesian_path_prefers_fast_previous_knot_continuation(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, _pose = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        q_final = q_start.copy()
        q_final[0] += 0.20
        real_solver = tracked_motion.solve_pose_target
        calls: list[int] = []

        def counted_solver(*args, **kwargs):
            calls.append(int(kwargs.get("max_iterations", 0)))
            return real_solver(*args, **kwargs)

        with mock.patch.object(
            tracked_motion,
            "solve_pose_target",
            side_effect=counted_solver,
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
                planning_deadline_monotonic=time.monotonic() + 30.0,
            )

        self.assertGreater(path["fast_continuation_attempt_count"], 0, path)
        self.assertGreater(path["fast_continuation_success_count"], 0, path)
        self.assertEqual(path["legacy_ik_call_count"], 0, path)
        self.assertIn(
            tracked_motion.FAST_CARTESIAN_CONTINUATION_MAX_ITERATIONS,
            calls,
        )

    def test_cached_affine_operator_matches_lstsq_reference(self) -> None:
        """Caching the symbolic projection must preserve its old numerics."""

        model = tracked_motion._build_constraint_model(
            [
                {"name": "a", "target_xyz_m": ["x", "y", "za"]},
                {"name": "b", "target_xyz_m": ["x+0.07", "y", "zb"]},
                {"name": "c", "target_xyz_m": ["x", "yc", "zc"]},
            ]
        )
        points = np.asarray(
            [
                [0.41, -0.12, 0.83],
                [0.48, -0.12, 0.85],
                [0.43, -0.08, 0.81],
            ],
            dtype=np.float64,
        )
        cached_residual, cached_values = model.fit(points)
        actual = model._actual_values(points)
        rhs = actual - model.constants
        reference_values = np.linalg.lstsq(
            model.matrix,
            rhs,
            rcond=None,
        )[0]
        reference_residual = rhs - model.matrix @ reference_values
        np.testing.assert_allclose(cached_values, reference_values, atol=1e-12)
        np.testing.assert_allclose(cached_residual, reference_residual, atol=1e-12)
        # Repeated calls use the same immutable operator rather than fitting a
        # new decomposition for every FK evaluation.
        operator = model._variable_lstsq_operator
        model.fit(points + 1.0e-4)
        self.assertIs(model._variable_lstsq_operator, operator)

    def test_numeric_three_point_noop_skips_kabsch_without_relaxing_acceptance(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (eef_position, _eef_quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        source = np.asarray(eef_position, dtype=np.float64) + np.asarray(
            [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.03, 0.0]],
            dtype=np.float64,
        )
        target = [
            {"name": f"p{index}", "target_xyz_m": point.tolist()}
            for index, point in enumerate(source)
        ]
        with mock.patch.object(
            tracked_motion,
            "kabsch_rigid_transform",
            side_effect=AssertionError("no-op must not run Kabsch"),
        ):
            result = tracked_motion.plan_endpoint(
                state=state,
                arm="left",
                q_start=q_start,
                source_points_robot_base_m=source,
                target_points=target,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
            )
        self.assertEqual(
            result["selected_candidate"]["mode"],
            "constraints_already_satisfied_no_motion",
        )
        np.testing.assert_allclose(result["q_final"], q_start, atol=1e-12)
        self.assertTrue(result["constraints"]["ok"])

    def test_numeric_four_point_noop_skips_multistart_and_keeps_j8_locked(self) -> None:
        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (eef_position, _eef_quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        source = np.asarray(eef_position, dtype=np.float64) + np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.0, 0.02],
            ],
            dtype=np.float64,
        )
        target = [
            {"name": f"p{index}", "target_xyz_m": point.tolist()}
            for index, point in enumerate(source)
        ]
        with mock.patch.object(
            tracked_motion,
            "solve_pose_target",
            side_effect=AssertionError("no-op must not run IK"),
        ):
            result = tracked_motion.plan_endpoint(
                state=state,
                arm="left",
                q_start=q_start,
                source_points_robot_base_m=source,
                target_points=target,
                pos_tol_m=0.012,
                ori_tol_deg=5.0,
            )
        self.assertEqual(
            result["selected_candidate"]["mode"],
            "constraints_already_satisfied_no_motion",
        )
        self.assertEqual(np.asarray(result["q_final"]).shape, (ARM_DOF,))
        if ARM_DOF == 8:
            self.assertEqual(float(result["q_final"][7]), 0.0)
        self.assertTrue(result["constraints"]["ok"])

    def test_fallback_does_not_change_successful_cartesian_path(self) -> None:
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (position, _quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        source = np.asarray(position, dtype=np.float64).reshape(1, 3)
        world._official_tracked_object_distances = official_test._RigidTrackedManager(
            world, adapter, {"known": source[0]}
        )
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target_position = source[0] + np.array([0.003, 0.0, 0.0])
        result = official_tools._move_tracked_plan_frozen(
            state=state,
            arm="left",
            q_start=q_start,
            source_points=source,
            target_points=[
                {"name": "known", "target_xyz_m": target_position.tolist()}
            ],
            pos_tol_m=0.012,
            ori_tol_deg=5.0,
            max_steps=120,
        )
        self.assertTrue(result["ok"], result)
        self.assertNotEqual(
            result["path_plan"].get("path_mode"),
            "joint_space_fallback",
        )

    def test_execution_schedule_view_does_not_rewrite_signed_waypoints(self) -> None:
        adapter, world, ctx, _result, capture = self._execution_fixture()
        start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        dense = []
        for fraction in np.linspace(0.25, 1.0, 4):
            q = start.copy()
            q[0] += 0.04 * float(fraction)
            dense.append(q.astype(float).tolist())
        selected = [dense[1], dense[-1]]
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.12,
                "max_joint_step_rad": 0.018,
            },
            "waypoints": dense,
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=10.0,
            max_steps=10,
            operation_name="move_tracked_point",
            execution_waypoints=selected,
            execution_max_step_rad=0.05,
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action)
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = action[ACTION_SLICES["arm_left"]]
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["waypoint_count"], len(selected))
        self.assertEqual(report["signed_waypoint_count"], len(dense))
        self.assertTrue(report["execution_schedule_applied"])
        self.assertEqual(len(actions), len(selected))

    def test_execution_schedule_helper_relaxes_only_verified_path(self) -> None:
        _adapter, world, _ctx, _result, capture = self._execution_fixture()
        state = capture["state"]
        start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        dense = []
        for fraction in np.linspace(0.04, 1.0, 25):
            q = start.copy()
            q[0] += 0.35 * float(fraction)
            dense.append(q.astype(float).tolist())
        trajectory = {
            "active_arm": "left",
            "tracking": {"max_joint_step_rad": 0.018},
            "planner": {
                "same_branch_checked": True,
                "monotonic_progress_checked": True,
                "internal_stop_count": 0,
            },
            "waypoints": dense,
        }
        report = official_tools._move_tracked_prepare_execution_schedule(
            trajectory=trajectory,
            state=state,
            arm="left",
            start_q=start,
            max_steps=8,
            available_time_s=30.0,
            observation_dt_s=0.1,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "adaptive_monotone_subsample")
        self.assertLessEqual(report["selected_waypoint_count"], 8)
        self.assertEqual(report["selected_indices"][-1], len(dense) - 1)
        self.assertTrue(report["same_branch_inherited_from_signed_path"])
        self.assertTrue(report["transition_local_fk_finite_checked"])
        indices = report["selected_indices"]
        self.assertEqual(indices, sorted(set(indices)))

    def test_relaxed_execution_rejects_unsigned_path_metadata(self) -> None:
        _adapter, world, _ctx, _result, capture = self._execution_fixture()
        state = capture["state"]
        start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        dense = []
        for fraction in np.linspace(0.04, 1.0, 25):
            q = start.copy()
            q[0] += 0.35 * float(fraction)
            dense.append(q.astype(float).tolist())
        trajectory = {
            "active_arm": "left",
            "tracking": {"max_joint_step_rad": 0.018},
            "planner": {
                "same_branch_checked": False,
                "monotonic_progress_checked": True,
                "internal_stop_count": 0,
            },
            "waypoints": dense,
        }
        report = official_tools._move_tracked_prepare_execution_schedule(
            trajectory=trajectory,
            state=state,
            arm="left",
            start_q=start,
            max_steps=8,
            available_time_s=30.0,
            observation_dt_s=0.1,
        )
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "execution_schedule_invalid")
        self.assertIn("same-branch", report["error"])

    def test_stream_tracking_retries_finite_progress_before_failing(self) -> None:
        adapter, world, ctx, _result, capture = self._execution_fixture()
        start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        target = start.copy()
        target[0] += 0.04
        trajectory = {
            "active_arm": "left",
            "start_state": {"episode_id": world.episode_id()},
            "tracking": {
                "execution_mode": "continuous_waypoint_stream",
                "waypoint_tolerance_rad": 0.018,
                "max_tracking_error_rad": 0.01,
            },
            "waypoints": [target.astype(float).tolist()],
        }
        generator = official_tools._move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture,
            adapter=adapter,
            observation_sequence=int(adapter.status()["sequence"]),
            timeout_s=10.0,
            max_steps=10,
            operation_name="move_tracked_point",
        )
        actions = []
        while True:
            try:
                action = np.asarray(next(generator), dtype=np.float64)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action)
            proprio = adapter.proprio_vector().copy()
            current = np.asarray(proprio[PROPRIO_SLICES["arm_left_qpos"]], dtype=np.float64)
            command = np.asarray(action[ACTION_SLICES["arm_left"]], dtype=np.float64)
            # First sample moves halfway (still outside the tight corridor),
            # the bounded retry reaches the command exactly.
            factor = 0.5 if len(actions) == 1 else 1.0
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = current + factor * (command - current)
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        self.assertTrue(report["ok"], report)
        self.assertGreaterEqual(report["tracking_recovery_steps"], 1)
        self.assertEqual(report["reason"], "trajectory_complete")


if __name__ == "__main__":
    unittest.main()
