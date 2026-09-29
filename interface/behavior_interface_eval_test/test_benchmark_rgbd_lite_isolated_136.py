from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

import pytest

from behavior_interface_eval_test import benchmark_rgbd_lite_isolated_136 as benchmark


def test_manifest_has_68_clicks_forced_to_both_arms() -> None:
    manifest = benchmark.build_manifest()

    assert len(manifest) == 136
    assert len({row["source_case_id"] for row in manifest}) == 68
    assert sum(row["plan_arm"] == "left" for row in manifest) == 68
    assert sum(row["plan_arm"] == "right" for row in manifest) == 68
    assert all(row["seed"] == 42 for row in manifest)


def test_optimized_defaults_are_the_full136_verified_configuration() -> None:
    environment = {}

    benchmark._apply_optimized_runtime_defaults(environment)

    assert environment == benchmark.OPTIMIZED_RUNTIME_DEFAULTS
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_CONCURRENCY"] == "6"
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES"] == "4"
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"] == "64"
    assert (
        environment[
            "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_STAGES"
        ]
        == "closure2,translation"
    )
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_NORMAL_ALIGNMENT_CACHE"] == "1"
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_POSE_GENERATION_CACHE"] == "1"


def test_optimized_defaults_preserve_explicit_overrides() -> None:
    environment = {
        "OFFICIAL_V2_RGBD_LITE_TEST_CONCURRENCY": "3",
        "OFFICIAL_V2_RGBD_LITE_TEST_POSE_GENERATION_CACHE": "0",
    }

    benchmark._apply_optimized_runtime_defaults(environment)

    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_CONCURRENCY"] == "3"
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_POSE_GENERATION_CACHE"] == "0"
    assert environment["OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES"] == "4"


def test_pilot_manifest_has_10_clicks_and_20_arm_cases() -> None:
    manifest = benchmark.build_manifest(benchmark.SOURCE_CASE_IDS[:10])

    assert len(manifest) == 20
    assert len({row["source_case_id"] for row in manifest}) == 10
    assert {row["plan_arm"] for row in manifest} == {"left", "right"}


def test_single_frame_pilot_selects_10_5012_clicks_and_20_arm_cases() -> None:
    source_ids = benchmark.select_source_case_ids(scene="5012", limit=10)
    manifest = benchmark.build_manifest(source_ids)

    assert len(source_ids) == 10
    assert len(manifest) == 20
    assert {row["scene"] for row in manifest} == {"5012"}
    assert len({row["source_case_id"] for row in manifest}) == 10
    assert sum(row["plan_arm"] == "left" for row in manifest) == 10
    assert sum(row["plan_arm"] == "right" for row in manifest) == 10


def test_optimized_global_schedule_keeps_all_scenes_in_one_group(
    monkeypatch,
) -> None:
    manifest = benchmark.build_manifest(
        (
            "s5010_p01_u380_v315",
            "s5011_p01_u374_v289",
            "s5012_p01_u431_v361",
        )
    )
    monkeypatch.setenv(benchmark.GLOBAL_SCHEDULE_ENV, "1")

    groups = benchmark._optimized_manifest_groups(manifest)

    assert groups == [manifest]


def test_optimized_global_schedule_can_restore_scene_groups(monkeypatch) -> None:
    manifest = benchmark.build_manifest(
        (
            "s5010_p01_u380_v315",
            "s5011_p01_u374_v289",
        )
    )
    monkeypatch.setenv(benchmark.GLOBAL_SCHEDULE_ENV, "0")

    groups = benchmark._optimized_manifest_groups(manifest)

    assert len(groups) == 2
    assert [[row["scene"] for row in group] for group in groups] == [
        ["5010", "5010"],
        ["5011", "5011"],
    ]


def test_scene_prewarm_uses_each_frozen_fixture_once(monkeypatch) -> None:
    calls = []

    def fake_reconstruct(session, **kwargs):
        calls.append((session, kwargs))

    monkeypatch.setattr(
        benchmark.rgbd_grasp_lite,
        "_cached_reconstruct_v52_scene_from_session",
        fake_reconstruct,
    )
    fixtures = {
        scene: {
            "session": {"session_id": scene},
            "camera": {
                "pos": np.asarray([1.0, 2.0, 3.0]),
                "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "fl": 17.0,
                "ha": 40.0,
            },
        }
        for scene in ("5012", "5010")
    }

    elapsed = benchmark._prewarm_fixture_scenes(fixtures)

    assert list(elapsed) == ["5010", "5012"]
    assert [call[0]["session_id"] for call in calls] == ["5010", "5012"]
    assert all(call[1]["focal_length"] == 17.0 for call in calls)
    assert all(call[1]["horizontal_aperture"] == 40.0 for call in calls)


def test_explicit_source_click_subset_is_validated() -> None:
    requested = (
        "s5012_p02_u439_v361",
        "s5012_p05_u436_v366",
        "s5012_p07_u435_v359",
    )

    assert benchmark.select_source_case_ids(
        scene="5012",
        case_ids=requested,
    ) == requested
    with pytest.raises(ValueError, match="unknown fixed source clicks"):
        benchmark.select_source_case_ids(case_ids=("s9999_p01_u1_v1",))


def test_current_pid_is_owned_by_its_own_stage_root() -> None:
    assert benchmark._pid_descends_from(os.getpid(), os.getpid())


def test_descendant_pid_snapshot_walks_the_full_process_tree(monkeypatch) -> None:
    tree = {
        10: {11, 12},
        11: {13},
        12: set(),
        13: {14},
        14: set(),
    }
    monkeypatch.setattr(
        benchmark,
        "_direct_child_pids",
        lambda pid: set(tree.get(pid, set())),
    )

    assert benchmark._descendant_pids(10) == {11, 12, 13, 14}


def test_owned_pid_tracker_remembers_a_short_lived_descendant(
    monkeypatch,
) -> None:
    snapshots = [{101}, {101, 202}, set()]
    snapshot_lock = threading.Lock()

    def fake_descendants(_root_pid):
        with snapshot_lock:
            return snapshots.pop(0) if snapshots else set()

    monkeypatch.setattr(benchmark, "_descendant_pids", fake_descendants)
    tracker = benchmark._OwnedPidTracker(99, interval_s=0.005)
    tracker.start()
    deadline = time.monotonic() + 1.0
    try:
        while 202 not in tracker.snapshot() and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        tracker.stop()

    assert tracker.snapshot() == {99, 101, 202}


def test_foreign_gpu_process_filter_uses_process_ancestry(monkeypatch) -> None:
    processes = [
        {"pid": 101, "process_name": "own-worker", "used_memory_mib": 100},
        {"pid": 202, "process_name": "evaluator", "used_memory_mib": 12000},
    ]
    monkeypatch.setattr(
        benchmark,
        "gpu_compute_processes",
        lambda gpu_index: processes,
    )
    monkeypatch.setattr(
        benchmark,
        "_pid_descends_from",
        lambda pid, root_pid: pid == 101 and root_pid == 99,
    )

    assert benchmark.foreign_gpu_processes(5, root_pid=99) == [processes[1]]


def test_gpu_process_classification_separates_owned_allowed_and_unexpected(
    monkeypatch,
) -> None:
    processes = [
        {"pid": 101, "process_name": "own-worker", "used_memory_mib": 100},
        {"pid": 202, "process_name": "evaluator", "used_memory_mib": 12000},
        {"pid": 303, "process_name": "new-process", "used_memory_mib": 300},
        {"pid": 404, "process_name": "known-owned", "used_memory_mib": 400},
    ]
    monkeypatch.setattr(
        benchmark,
        "_pid_descends_from",
        lambda pid, root_pid: pid == 101 and root_pid == 99,
    )

    classified = benchmark.classify_gpu_processes(
        processes,
        root_pid=99,
        allowed_preexisting_pids={202},
        known_owned_pids={404},
    )

    assert classified["owned"] == [processes[0], processes[3]]
    assert classified["allowed_preexisting"] == [processes[1]]
    assert classified["unexpected"] == [processes[2]]


def test_watchdog_rejects_foreign_gpu_process(monkeypatch) -> None:
    foreign = [
        {"pid": 202, "process_name": "evaluator", "used_memory_mib": 12000}
    ]
    monkeypatch.setattr(
        benchmark,
        "foreign_gpu_processes",
        lambda gpu_index, root_pid: foreign,
    )
    watchdog = benchmark._GPUIsolationWatchdog(5, root_pid=99)

    with pytest.raises(RuntimeError, match="lost isolation"):
        watchdog.start()


def test_cli_passes_explicit_shared_gpu_mode_to_pair(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(gpu_index, output_dir, **kwargs):
        captured.update(
            gpu_index=gpu_index,
            output_dir=output_dir,
            **kwargs,
        )
        return {"summary": {}}

    monkeypatch.setattr(benchmark, "run_isolated_pair", fake_run)

    assert benchmark.main(
        [
            "--gpu",
            "5",
            "--allow-shared-gpu",
            "--optimized-only",
            "--source-scene",
            "5012",
            "--source-click-limit",
            "10",
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert captured == {
        "gpu_index": 5,
        "output_dir": tmp_path.resolve(),
        "source_click_limit": 10,
        "source_scene": "5012",
        "source_case_ids": None,
        "allow_shared_gpu": True,
        "optimized_only": True,
        "baseline_only": False,
    }


def test_cli_passes_baseline_only_mode_to_pair(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(gpu_index, output_dir, **kwargs):
        captured.update(
            gpu_index=gpu_index,
            output_dir=output_dir,
            **kwargs,
        )
        return {"summary": {}}

    monkeypatch.setattr(benchmark, "run_isolated_pair", fake_run)

    assert benchmark.main(
        [
            "--gpu",
            "5",
            "--allow-shared-gpu",
            "--baseline-only",
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert captured["baseline_only"] is True
    assert captured["optimized_only"] is False


def test_pair_rejects_conflicting_stage_only_modes(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        benchmark.run_isolated_pair(
            5,
            tmp_path,
            optimized_only=True,
            baseline_only=True,
        )


def test_comparison_does_not_count_matching_failures_as_exact_poses(
    tmp_path,
) -> None:
    pose = {
        "pos": [1.0, 2.0, 3.0],
        "quat": [0.0, 0.0, 0.0, 1.0],
        "approach": [0.0, 0.0, 1.0],
    }
    payload_path = tmp_path / "pose.json"
    payload_path.write_text(
        json.dumps({"ok": True, "arm": "right", "eef_pose": pose}),
        encoding="utf-8",
    )
    manifest = [
        {"case_id": "success__right", "scene": "5012", "plan_arm": "right"},
        {"case_id": "failure__left", "scene": "5012", "plan_arm": "left"},
    ]
    successful_run = {
        "ok": True,
        "wall_s": 2.0,
        "payload_path": str(payload_path),
        "image_hashes": {},
    }
    failed_run = {
        "ok": False,
        "wall_s": 1.0,
        "error_type": "PlanningError",
        "error": "unreachable",
        "payload_path": None,
        "image_hashes": {},
    }
    for stage, wall_s in (("baseline", 3.0), ("optimized", 1.0)):
        report = {
            "complete": True,
            "manifest": manifest,
            "official_implementation_sha256": "same",
            "wall_s": wall_s,
            "results": [
                {**manifest[0], "run": successful_run},
                {**manifest[1], "run": failed_run},
            ],
        }
        (tmp_path / f"{stage}.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    summary = benchmark.compare_reports(tmp_path)["summary"]

    assert summary["cases"] == 2
    assert summary["plan_pose_outcome_exact"] == 2
    assert summary["successful_plan_poses_compared"] == 1
    assert summary["successful_plan_poses_exact"] == 1
    assert summary["all_successful_plan_poses_exact"] is True
    assert summary["all_136_exact"] is False
    assert summary["target_met"] is False


def test_target_uses_exact_plan_pose_contract_not_internal_diagnostics(
    tmp_path,
) -> None:
    pose = {
        "pos": [1.0, 2.0, 3.0],
        "quat": [0.0, 0.0, 0.0, 1.0],
        "approach": [0.0, 0.0, 1.0],
    }
    baseline_payload = tmp_path / "baseline_payload.json"
    optimized_payload = tmp_path / "optimized_payload.json"
    baseline_payload.write_text(
        json.dumps(
            {
                "ok": True,
                "arm": "right",
                "eef_pose": pose,
                "candidates": [{"meta": {"diagnostic_count": 120}}],
            }
        ),
        encoding="utf-8",
    )
    optimized_payload.write_text(
        json.dumps(
            {
                "ok": True,
                "arm": "right",
                "eef_pose": pose,
                "candidates": [{"meta": {"diagnostic_count": 32}}],
            }
        ),
        encoding="utf-8",
    )
    manifest = [
        {
            "case_id": f"case_{index:03d}__right",
            "scene": "5012",
            "plan_arm": "right",
        }
        for index in range(136)
    ]
    for stage, wall_s, payload_path in (
        ("baseline", 500.0, baseline_payload),
        ("optimized", 100.0, optimized_payload),
    ):
        report = {
            "complete": True,
            "manifest": manifest,
            "official_implementation_sha256": "same",
            "wall_s": wall_s,
            "results": [
                {
                    **descriptor,
                    "run": {
                        "ok": True,
                        "wall_s": 1.0,
                        "payload_path": str(payload_path),
                        "image_hashes": {},
                    },
                }
                for descriptor in manifest
            ],
        }
        (tmp_path / f"{stage}.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    summary = benchmark.compare_reports(tmp_path)["summary"]

    assert summary["all_136_plan_pose_outcomes_exact"] is True
    assert summary["all_successful_plan_poses_exact"] is True
    assert summary["all_136_exact"] is False
    assert summary["speedup"] == 5.0
    assert summary["target_met"] is True


def test_require_target_returns_nonzero_for_incomplete_summary(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "compare_reports",
        lambda output_dir: {"summary": {"target_met": False}},
    )

    assert benchmark.main(
        ["--compare-only", "--require-target", "--output-dir", str(tmp_path)]
    ) == 2
