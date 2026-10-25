from __future__ import annotations

import copy
import math
import threading
import unittest
from unittest import mock

import numpy as np

from behavior_interface_eval_test.robot_contract import ARM_DOF
from behavior_interface_eval_test.tool.official_v2 import tools


class MoveTrackedOverlapFallbackTest(unittest.TestCase):
    def record(self, index, *, precise=True, cost=0.0, overlap=0.2, inflated=None):
        q = np.zeros(ARM_DOF)
        q[0] = (index + 1) * 0.01
        return ({
            "q_start": np.zeros(ARM_DOF),
            "q_final": q,
            "start_eef_position_m": np.array([0.4, 0.0, 0.7]),
            "final_eef_position_m": np.array([0.41 + index * 0.01, 0.0, 0.7]),
            "final_eef_quaternion_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
            "anchors_eef_m": np.zeros((1, 3)),
            "selected_is_precise": precise,
            "selected_candidate": {
                "precise": precise,
                "accuracy_cost_mm_plus_deg": cost,
                "position_error_mm": cost,
                "selection_orientation_error_deg": 0.0,
                "motion_score": float(index),
                "endpoint_score_rank": index,
            },
            "overlap_volume_filter": {
                "ok": False,
                "overlap_vol_cm3": overlap,
                "inflated_overlap_vol_cm3": inflated,
                "original_overlap_ok": overlap <= 0.15,
                "inflated_overlap_ok": False,
                "inflated_query_skipped": inflated is None,
            },
        }, "test_frontier")

    def rank(self, records):
        return tools._move_tracked_overlap_fallback_records(records)

    def indices(self, ranked):
        return [candidate["selected_candidate"]["endpoint_score_rank"] for candidate, _ in ranked]

    def test_precise_class_precedes_overlap_and_motion(self):
        records = [
            self.record(0, cost=0.1, overlap=0.4),
            self.record(1, cost=0.2, overlap=0.2),
            self.record(2, precise=False, cost=4.0, overlap=0.01, inflated=2.0),
        ]
        ranked = self.rank(records)
        self.assertEqual(self.indices(ranked), [1, 0])
        self.assertEqual(ranked[0][0]["overlap_volume_filter"]["fallback_selection"]["accuracy_class"], "precise")

    def test_collinear_precision_is_not_reclassified_using_general_3mm_tolerance(self):
        records = [
            self.record(0, precise=False, cost=1.2, overlap=0.001, inflated=2.0),
            self.record(1, precise=True, cost=0.9, overlap=0.3),
        ]
        self.assertEqual(self.indices(self.rank(records)), [1])

    def test_no_precise_solution_retains_only_minimum_constraint_error(self):
        records = [
            self.record(0, precise=False, cost=4.0, overlap=0.3),
            self.record(1, precise=False, cost=4.0, overlap=0.2),
            self.record(2, precise=False, cost=4.001, overlap=0.01, inflated=2.0),
        ]
        ranked = self.rank(records)
        self.assertEqual(self.indices(ranked), [1, 0])
        self.assertEqual(ranked[0][0]["overlap_volume_filter"]["fallback_selection"]["accuracy_class"], "minimum_error")

    def test_passing_inaccurate_pose_cannot_displace_rejected_precise_pose(self):
        exact = self.record(0, cost=0.001, overlap=0.4)
        inaccurate = self.record(1, precise=False, cost=36.16, overlap=0.01, inflated=0.1)
        inaccurate[0]["overlap_volume_filter"]["ok"] = True
        self.assertEqual(tools._move_tracked_best_accuracy_records(
            [inaccurate], reference_records=[exact, inaccurate]), [])
        self.assertEqual(self.indices(self.rank(tools._move_tracked_best_accuracy_records(
            [exact], reference_records=[exact, inaccurate]))), [0])

    def test_passing_pose_cannot_displace_more_accurate_best_effort_pose(self):
        better = self.record(0, precise=False, cost=4.0, overlap=0.4)
        worse = self.record(1, precise=False, cost=36.0, overlap=0.01, inflated=0.1)
        self.assertEqual(tools._move_tracked_best_accuracy_records(
            [worse], reference_records=[better, worse]), [])

    def test_all_precise_passing_poses_keep_their_original_motion_order(self):
        records = [self.record(0, cost=0.9), self.record(1, cost=0.1)]
        self.assertEqual(tools._move_tracked_best_accuracy_records(records), records)

    def test_original_overlap_precedes_inflated_overlap_and_breaks_ties(self):
        records = [
            self.record(0, overlap=0.1, inflated=1.8),
            self.record(1, overlap=0.08, inflated=2.1),
            self.record(2, overlap=0.08, inflated=1.9),
        ]
        self.assertEqual(self.indices(self.rank(records)), [2, 1, 0])

    def test_fallback_does_not_mutate_rejected_reports_or_query_skipped_volumes(self):
        record = self.record(0)
        before = copy.deepcopy(record[0]["overlap_volume_filter"])
        ranked = self.rank([record])
        self.assertEqual(record[0]["overlap_volume_filter"], before)
        self.assertFalse(ranked[0][0]["overlap_volume_filter"]["ok"])
        self.assertIsNone(ranked[0][0]["overlap_volume_filter"]["inflated_overlap_vol_cm3"])

    def test_unmeasured_invalid_or_non_ik_candidates_are_never_fallbacks(self):
        for mutation in ("unavailable", "no_ik", "nan", "negative", "infinite_accuracy", "passing", "inflated_missing", "inflated_nan"):
            with self.subTest(mutation=mutation):
                record = self.record(0)
                candidate = record[0]
                report = candidate["overlap_volume_filter"]
                if mutation == "unavailable":
                    report["evaluation_available"] = False
                elif mutation == "no_ik":
                    candidate.pop("q_final")
                elif mutation == "nan":
                    report["overlap_vol_cm3"] = math.nan
                elif mutation == "negative":
                    report["overlap_vol_cm3"] = -1.0
                elif mutation == "infinite_accuracy":
                    candidate["selected_candidate"]["accuracy_cost_mm_plus_deg"] = math.inf
                elif mutation == "passing":
                    report["ok"] = True
                elif mutation == "inflated_missing":
                    report["inflated_query_skipped"] = False
                else:
                    report["inflated_overlap_vol_cm3"] = math.nan
                self.assertEqual(self.rank([record]), [])

    def plan(
        self, candidates, *, extra_candidates=(), cancel_event=None,
        cancel_in_filter=False, path_error=False, joint_error=False,
        endpoint_deadline=False, passing_indices=(),
    ):
        def endpoint(**kwargs):
            if endpoint_deadline and "eef_orientation_regularization_weight" in kwargs:
                raise tools.PlanningDeadlineExceeded("endpoint budget exhausted")
            frontier = list(extra_candidates) if "eef_orientation_regularization_weight" in kwargs else candidates
            if not frontier:
                raise ValueError("no additional endpoint")
            return {**frontier[0][0], "_ranked_endpoint_candidates": [c for c, _ in frontier]}

        def rejected_frontier(records, **kwargs):
            if cancel_in_filter:
                cancel_event.set()
            passing = [record for record in records if record[0]["selected_candidate"]["endpoint_score_rank"] in passing_indices]
            return passing, [], [], {
                "evaluation_unavailable_count": 0,
                "batch_attempt_count": 1,
                "successful_batch_count": 1,
                "adaptive_split_count": 0,
            }

        path = mock.Mock(return_value={"waypoints": [], "same_branch_checked": True})
        joint = mock.Mock(return_value={"waypoints": [], "same_branch_checked": True})
        if path_error:
            path.side_effect = ValueError("Cartesian route unavailable")
        if joint_error:
            joint.side_effect = ValueError("joint route unavailable")
        with mock.patch.object(tools, "_plan_tracked_endpoint", side_effect=endpoint), mock.patch.object(
            tools, "_move_tracked_filter_endpoint_overlap_frontier", side_effect=rejected_frontier
        ), mock.patch.object(tools, "_plan_tracked_cartesian_trajectory", path), mock.patch.object(
            tools, "_plan_tracked_joint_space_trajectory", joint
        ):
            planned = tools._move_tracked_plan_frozen(
                state=None,
                arm="left",
                q_start=np.zeros(ARM_DOF),
                source_points=np.array([[0.4, 0.0, 0.7], [0.5, 0.0, 0.7]]),
                target_points=[{"name": n, "target_xyz_m": ["?", "?", "?"]} for n in ("a", "b")],
                pos_tol_m=0.03,
                ori_tol_deg=20.0,
                max_steps=180,
                overlap_filter_context={},
                cancel_event=cancel_event,
            )
        return planned, path, joint

    def test_full_pipeline_rejects_passing_inaccurate_pose_in_favor_of_precise_fallback(self):
        exact = self.record(0, cost=0.001, overlap=0.4)
        inaccurate = self.record(1, precise=False, cost=36.16, overlap=0.01, inflated=0.1)
        inaccurate[0]["overlap_volume_filter"]["ok"] = True
        planned, path, _ = self.plan([exact, inaccurate], passing_indices=(1,))
        self.assertTrue(planned["ok"], planned)
        self.assertEqual(planned["endpoint"]["selected_candidate"]["endpoint_score_rank"], 0)
        self.assertTrue(planned["path_plan"]["frozen_rgbd_overlap_filter"]["fallback_applied"])
        path.assert_called_once()

    def test_fallback_uses_best_candidate_across_later_endpoint_frontiers(self):
        planned, path, _ = self.plan(
            [self.record(0, overlap=0.4)],
            extra_candidates=[self.record(1, overlap=0.2)],
        )
        self.assertTrue(planned["ok"], planned)
        self.assertEqual(planned["endpoint"]["selected_candidate"]["endpoint_score_rank"], 1)
        path.assert_called_once()

    def test_endpoint_search_failure_still_returns_evaluated_fallback(self):
        planned, path, _ = self.plan([self.record(0)])
        self.assertTrue(planned["ok"], planned)
        self.assertTrue(planned["path_plan"]["frozen_rgbd_overlap_filter"]["fallback_applied"])
        path.assert_called_once()

    def test_fallback_retains_same_branch_joint_path_recovery(self):
        planned, path, joint = self.plan([self.record(0)], path_error=True)
        self.assertTrue(planned["ok"], planned)
        path.assert_called_once()
        joint.assert_called_once()
        self.assertTrue(planned["path_plan"]["frozen_rgbd_overlap_filter"]["fallback_applied"])

    def test_endpoint_deadline_does_not_discard_already_evaluated_fallback(self):
        planned, path, _ = self.plan([self.record(0)], endpoint_deadline=True)
        self.assertTrue(planned["ok"], planned)
        self.assertTrue(planned["path_plan"]["planning_budget_exhausted"])
        self.assertTrue(planned["path_plan"]["frozen_rgbd_overlap_filter"]["fallback_applied"])
        path.assert_called_once()

    def test_overlap_fallback_cannot_bypass_invalid_paths(self):
        planned, path, joint = self.plan(
            [self.record(0), self.record(1)], path_error=True, joint_error=True
        )
        self.assertFalse(planned["ok"], planned)
        self.assertIn("no endpoint candidate has a valid", planned["error"])
        self.assertEqual(path.call_count, 2)
        self.assertEqual(joint.call_count, 2)

    def test_cancellation_cannot_trigger_overlap_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.plan([self.record(0)], cancel_event=threading.Event(), cancel_in_filter=True)


if __name__ == "__main__":
    unittest.main()
