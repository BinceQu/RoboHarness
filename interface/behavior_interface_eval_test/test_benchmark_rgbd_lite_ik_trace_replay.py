from __future__ import annotations

from behavior_interface_eval_test.benchmark_rgbd_lite_ik_trace_replay import (
    canonical_raw_result,
    first_difference,
)


def test_canonical_raw_result_removes_runtime_only_fields():
    value = {
        "arms": {"right": [{"ok": True, "q_arm": [1.0, 2.0]}]},
        "meta": {
            "fixed_solve_batch_size": 8,
            "final_solve_s": 2.0,
            "runtime_cache": {"goal_hits": 3},
        },
    }

    assert canonical_raw_result(value) == {
        "arms": {"right": [{"ok": True, "q_arm": [1.0, 2.0]}]},
        "meta": {"fixed_solve_batch_size": 8},
    }


def test_first_difference_reports_candidate_value_path():
    left = {"arms": {"right": [{"q_arm": [1.0, 2.0]}]}}
    right = {"arms": {"right": [{"q_arm": [1.0, 3.0]}]}}

    difference = first_difference(left, right)

    assert difference == "$.arms.right[0].q_arm[1]: 2.0 != 3.0"
