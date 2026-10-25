"""Unit tests for the pure tracked-point execution scheduler."""

from __future__ import annotations

import unittest

import numpy as np

from tools.util.tracked_point_execution_local import (
    schedule_execution_waypoints,
)


class TrackedPointExecutionLocalTest(unittest.TestCase):
    def _path(self, count: int = 25) -> np.ndarray:
        path = np.zeros((count, 8), dtype=np.float64)
        path[:, 0] = np.linspace(0.0, 0.18, count)
        path[:, 1] = np.linspace(0.0, -0.09, count)
        return path

    def test_identity_schedule_preserves_legacy_success(self) -> None:
        path = self._path(4)
        report = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=20,
            available_time_s=20.0,
            observation_dt_s=0.1,
            max_joint_step_rad=0.10,
            j8_index=7,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "legacy_identity")
        np.testing.assert_array_equal(np.asarray(report["waypoints"]), path)
        self.assertEqual(report["selected_indices"], [0, 1, 2, 3])

    def test_adaptive_schedule_is_monotone_bounded_and_keeps_endpoint(self) -> None:
        path = self._path()
        report = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=8,
            available_time_s=None,
            observation_dt_s=None,
            max_joint_step_rad=0.018,
            max_speedup_factor=3.0,
            j8_index=7,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "adaptive_monotone_subsample")
        indices = report["selected_indices"]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertLessEqual(len(indices), 8)
        selected = np.asarray(report["waypoints"], dtype=np.float64)
        self.assertEqual(selected.shape, (len(indices), 8))
        np.testing.assert_array_equal(selected[-1], path[-1])
        deltas = np.vstack((selected[0], np.diff(selected, axis=0)))
        self.assertLessEqual(
            float(np.max(np.linalg.norm(deltas, axis=1))),
            0.018 * 3.0 + 1.0e-9,
        )
        np.testing.assert_array_equal(selected[:, 7], 0.0)

    def test_time_budget_is_conservative_and_structured(self) -> None:
        path = self._path(10)
        report = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=100,
            available_time_s=0.25,
            observation_dt_s=0.1,
            max_joint_step_rad=0.018,
        )
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "execution_budget_infeasible")
        self.assertEqual(report["budget_steps"], 0)

    def test_speed_cap_rejects_an_unbounded_shortcut(self) -> None:
        path = self._path(20)
        report = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=2,
            max_joint_step_rad=0.018,
            max_speedup_factor=1.0,
        )
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "execution_budget_infeasible")

    def test_relaxed_pass_derives_minimal_cap_when_bounded_pass_cannot_fit(self) -> None:
        path = self._path(20)
        bounded = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=4,
            max_joint_step_rad=0.018,
            max_speedup_factor=2.5,
        )
        self.assertFalse(bounded["ok"], bounded)
        relaxed = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=4,
            max_joint_step_rad=0.018,
            max_speedup_factor=None,
        )
        self.assertTrue(relaxed["ok"], relaxed)
        self.assertEqual(relaxed["mode"], "adaptive_monotone_subsample_relaxed")
        self.assertLessEqual(relaxed["selected_waypoint_count"], 4)
        self.assertEqual(relaxed["selected_indices"][-1], len(path) - 1)
        self.assertGreater(relaxed["speedup_factor"], 2.5)

    def test_invalid_values_and_locked_joint_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            schedule_execution_waypoints(
                [[0.0, 0.0, float("nan")]],
                max_steps=1,
                max_joint_step_rad=0.018,
                j8_index=2,
                j8_value=0.0,
            )
        with self.assertRaisesRegex(ValueError, "locked"):
            schedule_execution_waypoints(
                np.asarray([[0.0, 0.0, 0.1]]),
                max_steps=1,
                max_joint_step_rad=0.018,
                j8_index=2,
                j8_value=0.0,
            )

    def test_schedule_is_deterministic_for_same_inputs(self) -> None:
        path = self._path(31)
        first = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=9,
            max_joint_step_rad=0.018,
            max_speedup_factor=2.5,
        )
        second = schedule_execution_waypoints(
            path,
            start_q=np.zeros(8),
            max_steps=9,
            max_joint_step_rad=0.018,
            max_speedup_factor=2.5,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
