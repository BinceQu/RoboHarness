from __future__ import annotations

import concurrent.futures
import json
import threading
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from behavior_interface_eval_test.test_support import rgbd_lite_isolated_runtime as runtime


class FakeWorker:
    instances = []

    def __init__(self, *, arm: str, gpu: str, python_path: str, environment: Dict[str, str]):
        self.arm = arm
        self.gpu = gpu
        self.python_path = python_path
        self.environment = environment
        self.closed = False
        self.requests = []
        FakeWorker.instances.append(self)

    def alive(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def request(self, request: Dict[str, Any], **kwargs):
        self.requests.append((request, kwargs))
        return {"ok": True, "arms": {self.arm: []}, "meta": {}}


@pytest.fixture(autouse=True)
def clean_pool():
    runtime.close_test_policy_workers()
    runtime.close_test_render_executor()
    runtime.clear_test_component_cache()
    runtime.clear_test_camera_face_cache()
    runtime.clear_test_normal_alignment_cache()
    runtime.clear_test_pose_generation_cache()
    runtime._configure_candidate_compression(False)
    runtime.MeshOccupancyEvaluator.build = (
        runtime._ORIGINAL_MESH_OCCUPANCY_BUILD_DESCRIPTOR
    )
    runtime.rgbd_grasp_lite._lite_watertight_component_bounds = (
        runtime._ORIGINAL_WATERTIGHT_COMPONENT_BOUNDS
    )
    runtime.grasp_kinematics_local._WORKER_PATH = (
        runtime._ORIGINAL_IK_WORKER_PATH
    )
    runtime.grasp_kinematics_local._build_external_ik_request = (
        runtime._ORIGINAL_BUILD_EXTERNAL_IK_REQUEST
    )
    runtime.grasp_kinematics_local._ik_request_in_shared_memory = (
        runtime._ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY
    )
    runtime.rgbd_grasp_lite._lite_render_executor = (
        runtime._ORIGINAL_LITE_RENDER_EXECUTOR
    )
    runtime.rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite = (
        runtime._ORIGINAL_CAMERA_FACE_LITE
    )
    runtime.rgbd_grasp_planner.DenseSceneOccupancy.counts_for_poses = (
        runtime._ORIGINAL_COUNTS_FOR_POSES
    )
    runtime.rgbd_grasp_planner.filter_poses_safe_final_dual_arm_ik = (
        runtime._ORIGINAL_FILTER_SAFE_FINAL
    )
    runtime.rgbd_grasp_planner.attach_normal_alignment = (
        runtime._ORIGINAL_ATTACH_NORMAL_ALIGNMENT
    )
    runtime.rgbd_grasp_planner.generate_rgbd_filter_poses = (
        runtime._ORIGINAL_GENERATE_RGBD_FILTER_POSES
    )
    runtime.rgbd_grasp_planner.generate_local_orientation_refinements = (
        runtime._ORIGINAL_GENERATE_LOCAL_ORIENTATION_REFINEMENTS
    )
    FakeWorker.instances.clear()
    yield
    runtime.close_test_policy_workers()
    runtime.close_test_render_executor()
    runtime.clear_test_component_cache()
    runtime.clear_test_camera_face_cache()
    runtime.clear_test_normal_alignment_cache()
    runtime.clear_test_pose_generation_cache()
    runtime._configure_candidate_compression(False)
    runtime.MeshOccupancyEvaluator.build = (
        runtime._ORIGINAL_MESH_OCCUPANCY_BUILD_DESCRIPTOR
    )
    runtime.rgbd_grasp_lite._lite_watertight_component_bounds = (
        runtime._ORIGINAL_WATERTIGHT_COMPONENT_BOUNDS
    )
    runtime.grasp_kinematics_local._WORKER_PATH = (
        runtime._ORIGINAL_IK_WORKER_PATH
    )
    runtime.grasp_kinematics_local._build_external_ik_request = (
        runtime._ORIGINAL_BUILD_EXTERNAL_IK_REQUEST
    )
    runtime.grasp_kinematics_local._ik_request_in_shared_memory = (
        runtime._ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY
    )
    runtime.rgbd_grasp_lite._lite_render_executor = (
        runtime._ORIGINAL_LITE_RENDER_EXECUTOR
    )
    runtime.rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite = (
        runtime._ORIGINAL_CAMERA_FACE_LITE
    )
    runtime.rgbd_grasp_planner.DenseSceneOccupancy.counts_for_poses = (
        runtime._ORIGINAL_COUNTS_FOR_POSES
    )
    runtime.rgbd_grasp_planner.filter_poses_safe_final_dual_arm_ik = (
        runtime._ORIGINAL_FILTER_SAFE_FINAL
    )
    runtime.rgbd_grasp_planner.attach_normal_alignment = (
        runtime._ORIGINAL_ATTACH_NORMAL_ALIGNMENT
    )
    runtime.rgbd_grasp_planner.generate_rgbd_filter_poses = (
        runtime._ORIGINAL_GENERATE_RGBD_FILTER_POSES
    )
    runtime.rgbd_grasp_planner.generate_local_orientation_refinements = (
        runtime._ORIGINAL_GENERATE_LOCAL_ORIENTATION_REFINEMENTS
    )


def test_install_requires_explicit_test_mode(monkeypatch):
    monkeypatch.delenv(runtime.TEST_MODE_ENV, raising=False)
    with pytest.raises(RuntimeError, match=runtime.TEST_MODE_ENV):
        runtime.install_policy_pool_runtime()


def test_install_defaults_to_exact_parallel_fresh_child_runtime(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.delenv(runtime.POLICY_POOL_ENV, raising=False)
    monkeypatch.delenv(runtime.SKIP_REACHABLE_SE3_ENV, raising=False)
    monkeypatch.delenv(
        "OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES",
        raising=False,
    )
    monkeypatch.delenv(runtime.PHYSICAL_BATCH_ENV, raising=False)
    monkeypatch.delenv(runtime.SOLVER_RESET_ENV, raising=False)
    monkeypatch.delenv(runtime.EXACT_SOLVER_RELOAD_ENV, raising=False)
    monkeypatch.delenv(runtime.RENDER_PROCESSES_ENV, raising=False)
    monkeypatch.delenv(runtime.COMPONENT_CACHE_ENV, raising=False)
    monkeypatch.delenv(runtime.POLICY_SHARD_ENV, raising=False)
    monkeypatch.delenv(runtime.LANE_AFFINITY_ENV, raising=False)
    monkeypatch.delenv(runtime.PRE_IK_DEDUP_ENV, raising=False)
    monkeypatch.delenv(runtime.PREPARED_SIGNATURE_POOL_ENV, raising=False)
    monkeypatch.delenv(
        runtime.CAMERA_FACE_ORIENTATION_CACHE_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        runtime.NORMAL_ALIGNMENT_CACHE_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        runtime.POSE_GENERATION_CACHE_ENV,
        raising=False,
    )
    monkeypatch.delenv(runtime.CANDIDATE_COMPRESSION_ENV, raising=False)
    monkeypatch.delenv(
        runtime.CANDIDATE_COMPRESSION_SCOPE_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        runtime.CANDIDATE_COMPRESSION_STAGES_ENV,
        raising=False,
    )
    monkeypatch.delenv(runtime.CANDIDATE_POST_LIMIT_ENV, raising=False)
    monkeypatch.delenv(runtime.CANDIDATE_STAGE_LIMIT_ENV, raising=False)

    settings = runtime.install_policy_pool_runtime()

    assert settings["runtime"] == "exact_reload_parallel_ik_lanes"
    assert settings["policy_pool_enabled"] == "1"
    assert settings["forkserver_enabled"] == "1"
    assert settings["exact_solver_reload"] == "1"
    assert settings["OFFICIAL_V2_LITE_FORKSERVER_IK"] == "1"
    assert settings["OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES"] == "2"
    assert settings["OFFICIAL_V2_LITE_SCENE_CACHE_SIZE"] == "5"
    assert settings["lite_skip_reachable_se3_ik_stages"] == "0"
    assert settings[runtime.PHYSICAL_BATCH_ENV] == "0"
    assert settings[runtime.SOLVER_RESET_ENV] == "none"
    assert settings[runtime.RENDER_PROCESSES_ENV] == "1"
    assert settings[runtime.COMPONENT_CACHE_ENV] == "0"
    assert settings[runtime.POLICY_SHARD_ENV] == "0"
    assert settings[runtime.LANE_AFFINITY_ENV] == "0"
    assert settings[runtime.PRE_IK_DEDUP_ENV] == "0"
    assert settings[runtime.PRE_IK_COUNT_DEDUP_ENV] == "0"
    assert settings[runtime.PREPARED_SIGNATURE_POOL_ENV] == "0"
    assert settings[runtime.CAMERA_FACE_ORIENTATION_CACHE_ENV] == "0"
    assert settings[runtime.NORMAL_ALIGNMENT_CACHE_ENV] == "0"
    assert settings[runtime.POSE_GENERATION_CACHE_ENV] == "0"
    assert settings[runtime.CANDIDATE_COMPRESSION_ENV] == "0"
    assert settings[runtime.CANDIDATE_COMPRESSION_SCOPE_ENV] == "all"
    assert settings[runtime.CANDIDATE_COMPRESSION_STAGES_ENV] == ""
    assert settings[runtime.CANDIDATE_POST_LIMIT_ENV] == "64"
    assert settings[runtime.CANDIDATE_STAGE_LIMIT_ENV] == "64"
    assert settings[runtime.WORKER_CPU_THREADS_ENV] == "0"
    assert settings[runtime.SIGNATURE_STICKY_LANES_ENV] == "0"
    assert (
        runtime.grasp_kinematics_local._WORKER_PATH
        == runtime._ORIGINAL_IK_WORKER_PATH
    )
    assert (
        runtime.rgbd_grasp_lite._lite_render_executor
        is runtime._ORIGINAL_LITE_RENDER_EXECUTOR
    )


def test_candidate_compression_is_explicit_and_process_local(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.CANDIDATE_COMPRESSION_ENV, "1")
    monkeypatch.setenv(runtime.POLICY_POOL_ENV, "0")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.CANDIDATE_COMPRESSION_ENV] == "1"
    assert runtime.rgbd_grasp_lite.LITE_COMPRESS_INITIAL_STAGE is True
    assert runtime.rgbd_grasp_lite.LITE_COMPRESS_REFINEMENT_STAGES is True
    assert (
        runtime.rgbd_grasp_lite.rank_dedupe_top_ik_input_lite
        is runtime._TEST_INITIAL_RANKER
    )
    assert (
        runtime.rgbd_grasp_planner.rank_dedupe_top_ik_input
        is runtime._selective_lineage_ranker
    )
    assert (
        runtime.rgbd_grasp_planner.rank_seed_balanced_micro_ik_input
        is runtime._TEST_MICRO_RANKER
    )


def test_refinement_only_compression_preserves_initial_ranker(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.CANDIDATE_COMPRESSION_ENV, "1")
    monkeypatch.setenv(
        runtime.CANDIDATE_COMPRESSION_SCOPE_ENV,
        "refinement",
    )
    monkeypatch.setenv(runtime.POLICY_POOL_ENV, "0")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.CANDIDATE_COMPRESSION_SCOPE_ENV] == "refinement"
    assert (
        runtime.rgbd_grasp_lite.rank_dedupe_top_ik_input_lite
        is runtime._ORIGINAL_PRODUCTION_RANK_DEDUPE
    )
    assert (
        runtime.rgbd_grasp_planner.rank_dedupe_top_ik_input
        is runtime._selective_lineage_ranker
    )


def test_candidate_compression_can_target_only_late_stages(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.CANDIDATE_COMPRESSION_ENV, "1")
    monkeypatch.setenv(
        runtime.CANDIDATE_COMPRESSION_STAGES_ENV,
        "closure2,translation",
    )
    monkeypatch.setenv(runtime.CANDIDATE_STAGE_LIMIT_ENV, "48")
    monkeypatch.setenv(runtime.POLICY_POOL_ENV, "0")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.CANDIDATE_COMPRESSION_STAGES_ENV] == (
        "closure2,translation"
    )
    assert settings[runtime.CANDIDATE_STAGE_LIMIT_ENV] == "48"
    assert runtime.rgbd_grasp_lite.LITE_COMPRESS_INITIAL_STAGE is True
    assert (
        runtime.rgbd_grasp_lite.rank_dedupe_top_ik_input_lite
        is runtime._ORIGINAL_PRODUCTION_RANK_DEDUPE
    )
    assert runtime._candidate_stage_name(
        [{"reachable_se3_generation": 3}]
    ) == "closure2"
    assert runtime._candidate_stage_name(
        [{"translation_refined": True}]
    ) == "translation"


def test_pre_ik_singleflight_computes_once_for_two_arm_scopes():
    barrier = threading.Barrier(2)
    calls = []

    def invoke(value):
        with runtime.planner_case_scope("same-click"):
            key = runtime._pre_ik_call_key("pure-stage")
            assert key is not None
            barrier.wait()

            def compute():
                calls.append(value)
                return {"result": 7}

            return runtime._singleflight_copy(key, compute)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, (1, 2)))

    assert results == [{"result": 7}, {"result": 7}]
    assert len(calls) == 1
    assert runtime._PRE_IK_FLIGHTS == {}


def test_cached_camera_face_mutates_both_pose_lists_identically(monkeypatch):
    calls = []
    barrier = threading.Barrier(2)

    def fake_camera_face(poses, *, world, ctx=None):
        calls.append(world)
        for pose in poses:
            pose["camera_face"] = {"flipped": True}
            pose["eef_pos"] = np.asarray([1.0, 2.0, 3.0])
        return {"pose_count": len(poses), "flipped": len(poses), "skipped": 0}

    monkeypatch.setattr(runtime, "_ORIGINAL_CAMERA_FACE_LITE", fake_camera_face)

    def invoke(_arm):
        poses = [{"quat": np.zeros(4), "eef_pos": np.zeros(3)}]
        with runtime.planner_case_scope("same-click"):
            barrier.wait()
            audit = runtime._cached_camera_face_lite(
                poses,
                world="world",
                ctx=None,
            )
        return poses, audit

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, ("left", "right")))

    assert len(calls) == 1
    assert results[0][1] == results[1][1]
    assert results[0][0][0]["camera_face"] == {"flipped": True}
    np.testing.assert_array_equal(
        results[0][0][0]["eef_pos"],
        results[1][0][0]["eef_pos"],
    )


def test_orientation_cached_camera_face_is_float_exact():
    runtime.clear_test_camera_face_cache()
    anchor = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    original = runtime.rgbd_grasp_planner.generate_rgbd_filter_poses(anchor)
    optimized = __import__("copy").deepcopy(original)
    world = SimpleNamespace(robot_forward=np.asarray([0.7, -0.2, 0.0]))

    expected_audit = runtime._ORIGINAL_CAMERA_FACE_LITE(
        original,
        world=world,
        ctx=None,
    )
    actual_audit = runtime._orientation_cached_camera_face_lite(
        optimized,
        world=world,
        ctx=None,
    )

    assert actual_audit == expected_audit
    assert len(original) == len(optimized)
    for expected, actual in zip(original, optimized):
        assert set(expected) == set(actual)
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]


def test_orientation_cache_reuses_results_across_calls(monkeypatch):
    from behavior_interface_eval_test.tool.official_v2 import grasp_geometry_local

    runtime.clear_test_camera_face_cache()
    calls = []
    original_ensure = grasp_geometry_local.ensure_camera_face_forward

    def counted_ensure(*args, **kwargs):
        calls.append(1)
        return original_ensure(*args, **kwargs)

    monkeypatch.setattr(
        grasp_geometry_local,
        "ensure_camera_face_forward",
        counted_ensure,
    )
    world = SimpleNamespace(robot_forward=np.asarray([0.7, -0.2, 0.0]))
    first = runtime.rgbd_grasp_planner.generate_rgbd_filter_poses(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    )
    second = runtime.rgbd_grasp_planner.generate_rgbd_filter_poses(
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64)
    )

    runtime._orientation_cached_camera_face_lite(first, world=world, ctx=None)
    first_call_count = len(calls)
    runtime._orientation_cached_camera_face_lite(second, world=world, ctx=None)

    assert first_call_count > 0
    assert len(calls) == first_call_count


def test_orientation_cached_camera_face_random_anchors_are_float_exact():
    import copy

    runtime.clear_test_camera_face_cache()
    rng = np.random.default_rng(20260823)
    quaternions = rng.standard_normal((37, 4))
    poses = []
    for index in range(1000):
        poses.append(
            {
                "quat": quaternions[index % len(quaternions)].copy(),
                "anchor": rng.standard_normal(3),
                "anchor_local": rng.standard_normal(3),
                "R": rng.standard_normal((3, 3)),
                "eef_pos": rng.standard_normal(3),
                "ik_warm_start_q_by_arm": {"right": rng.standard_normal(7)},
            }
        )
    expected = copy.deepcopy(poses)
    actual = copy.deepcopy(poses)
    world = SimpleNamespace(robot_forward=np.asarray([0.3, -0.8, 0.1]))

    expected_audit = runtime._ORIGINAL_CAMERA_FACE_LITE(
        expected,
        world=world,
        ctx=None,
    )
    actual_audit = runtime._orientation_cached_camera_face_lite(
        actual,
        world=world,
        ctx=None,
    )

    assert actual_audit == expected_audit
    for expected_pose, actual_pose in zip(expected, actual):
        assert set(actual_pose) == set(expected_pose)
        for key, expected_value in expected_pose.items():
            actual_value = actual_pose[key]
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            elif isinstance(expected_value, dict):
                assert set(actual_value) == set(expected_value)
                for child_key, child_expected in expected_value.items():
                    child_actual = actual_value[child_key]
                    if isinstance(child_expected, np.ndarray):
                        np.testing.assert_array_equal(child_actual, child_expected)
                    else:
                        assert child_actual == child_expected
            else:
                assert actual_value == expected_value


def test_orientation_cached_camera_face_audits_remain_independent():
    runtime.clear_test_camera_face_cache()
    poses = [
        {
            "quat": np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float64),
            "anchor": np.asarray([float(index), 2.0, 3.0], dtype=np.float64),
            "anchor_local": np.asarray([0.0, 0.0, 0.04], dtype=np.float64),
        }
        for index in range(2)
    ]
    world = SimpleNamespace(robot_forward=np.asarray([0.7, -0.2, 0.0]))

    runtime._orientation_cached_camera_face_lite(poses, world=world, ctx=None)

    assert poses[0]["camera_face"] is not poses[1]["camera_face"]
    for key in (
        "robot_forward_xy",
        "camera_normal_xy_before",
        "camera_normal_xy_after",
    ):
        left = poses[0]["camera_face"].get(key)
        right = poses[1]["camera_face"].get(key)
        if isinstance(left, list):
            assert left is not right


def test_cached_normal_alignment_is_float_exact_and_independent():
    import copy

    runtime.clear_test_normal_alignment_cache()
    rng = np.random.default_rng(20260823)
    rotations = rng.standard_normal((31, 3, 3))
    poses = [
        {"R": rotations[index % len(rotations)].copy(), "id": index}
        for index in range(1000)
    ]
    expected = copy.deepcopy(poses)
    actual = copy.deepcopy(poses)
    outward = np.asarray([0.31, -0.72, 0.19], dtype=np.float64)

    expected_audit = runtime._ORIGINAL_ATTACH_NORMAL_ALIGNMENT(
        expected,
        outward,
    )
    actual_audit = runtime._cached_normal_alignment(actual, outward)

    assert actual_audit == expected_audit
    for expected_pose, actual_pose in zip(expected, actual):
        assert set(actual_pose) == set(expected_pose)
        assert actual_pose["normal_alignment_deg"] == expected_pose[
            "normal_alignment_deg"
        ]
        np.testing.assert_array_equal(
            actual_pose["gripper_vector_world"],
            expected_pose["gripper_vector_world"],
        )
    assert (
        actual[0]["gripper_vector_world"]
        is not actual[len(rotations)]["gripper_vector_world"]
    )


def test_normal_alignment_cache_reuses_unique_scalar_results(monkeypatch):
    runtime.clear_test_normal_alignment_cache()
    calls = []
    original = runtime._ORIGINAL_POSE_GRIPPER_VECTOR_WORLD

    def counted(pose):
        calls.append(1)
        return original(pose)

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_POSE_GRIPPER_VECTOR_WORLD",
        counted,
    )
    rotations = [np.eye(3), np.diag([1.0, -1.0, -1.0])]
    poses = [
        {"R": rotations[index % len(rotations)].copy()}
        for index in range(28)
    ]
    outward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    runtime._cached_normal_alignment(poses, outward)
    runtime._cached_normal_alignment(poses, outward)

    assert len(calls) == len(rotations)


@pytest.mark.parametrize(
    ("axial_z_m", "n_roll"),
    (
        (runtime.rgbd_grasp_planner.AXIAL_Z_M, 8),
        (np.asarray([-0.013, -0.0, 0.021], dtype=np.float64), 3),
    ),
)
def test_cached_rgbd_filter_pose_templates_are_float_exact(
    axial_z_m,
    n_roll,
):
    runtime.clear_test_pose_generation_cache()
    anchors = np.asarray(
        [[1.25, -2.5, 0.125], [-0.7, 4.0, 1.3]],
        dtype=np.float64,
    )

    expected = runtime._ORIGINAL_GENERATE_RGBD_FILTER_POSES(
        anchors,
        axial_z_m=axial_z_m,
        n_roll=n_roll,
    )
    actual = runtime._cached_rgbd_filter_pose_templates(
        anchors,
        axial_z_m=axial_z_m,
        n_roll=n_roll,
    )

    assert len(actual) == len(expected)
    for expected_pose, actual_pose in zip(expected, actual):
        assert set(actual_pose) == set(expected_pose)
        for key, expected_value in expected_pose.items():
            actual_value = actual_pose[key]
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert actual[0]["R"] is not actual[len(expected) // 2]["R"]


def test_rgbd_filter_pose_template_cache_builds_official_template_once(
    monkeypatch,
):
    runtime.clear_test_pose_generation_cache()
    calls = []
    original = runtime._ORIGINAL_GENERATE_RGBD_FILTER_POSES

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_GENERATE_RGBD_FILTER_POSES",
        counted,
    )
    first_anchor = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    second_anchor = np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64)

    runtime._cached_rgbd_filter_pose_templates(first_anchor)
    runtime._cached_rgbd_filter_pose_templates(second_anchor)

    assert len(calls) == 1


def test_cached_local_orientation_refinements_are_float_exact():
    runtime.clear_test_pose_generation_cache()
    seeds = runtime._ORIGINAL_GENERATE_RGBD_FILTER_POSES(
        np.asarray([[0.5, -1.5, 2.5]], dtype=np.float64),
        axial_z_m=np.asarray([-0.01, 0.02], dtype=np.float64),
        n_roll=2,
    )[:17]
    for index, seed in enumerate(seeds):
        seed["custom_scalar"] = int(index)
        seed["custom_array"] = np.asarray([index, -index], dtype=np.float64)
    tilt_deg = (-7.5, -0.0, 4.0)
    roll_deg = (-3.0, 0.0, 5.0)

    expected = runtime._ORIGINAL_GENERATE_LOCAL_ORIENTATION_REFINEMENTS(
        seeds,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    actual = runtime._cached_local_orientation_refinements(
        seeds,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )

    assert len(actual) == len(expected)
    for expected_pose, actual_pose in zip(expected, actual):
        assert set(actual_pose) == set(expected_pose)
        for key, expected_value in expected_pose.items():
            actual_value = actual_pose[key]
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_cached_counts_replays_logical_count_cache(monkeypatch):
    calls = []
    barrier = threading.Barrier(2)

    def fake_counts(
        occupancy,
        poses,
        offsets_eef,
        *,
        batch_size=32,
        prefer_cuda=True,
    ):
        calls.append(occupancy)
        counts = np.asarray([3 for _pose in poses], dtype=np.int32)
        runtime._replay_count_cache(
            occupancy,
            poses,
            offsets_eef,
            counts,
            {"device": "numpy"},
        )
        return counts, {
            "device": "numpy",
            "pose_count": len(poses),
            "unique_query_pose_count": len(poses),
            "cache_hit_count": 0,
        }

    monkeypatch.setattr(runtime, "_ORIGINAL_COUNTS_FOR_POSES", fake_counts)

    def invoke(_arm):
        occupancy = runtime.rgbd_grasp_planner.DenseSceneOccupancy(
            origin=np.zeros(3),
            occupancy=np.zeros((2, 2, 2), dtype=bool),
            voxel_m=0.01,
            metadata={},
        )
        poses = [{"eef_pos": np.zeros(3), "R": np.eye(3)}]
        offsets = np.asarray([[0.0, 0.0, 0.0]])
        with runtime.planner_case_scope("same-click"):
            barrier.wait()
            result = runtime._cached_counts_for_poses(
                occupancy,
                poses,
                offsets,
                prefer_cuda=False,
            )
        return occupancy, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, ("left", "right")))

    assert len(calls) == 1
    for occupancy, (counts, metadata) in results:
        np.testing.assert_array_equal(counts, np.asarray([3], dtype=np.int32))
        assert metadata["unique_query_pose_count"] == 1
        assert sum(len(cache) for cache in occupancy._count_cache.values()) == 1


def test_install_can_enable_count_only_pre_ik_dedup(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.PRE_IK_DEDUP_ENV, "0")
    monkeypatch.setenv(runtime.PRE_IK_COUNT_DEDUP_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.PRE_IK_DEDUP_ENV] == "0"
    assert settings[runtime.PRE_IK_COUNT_DEDUP_ENV] == "1"
    assert (
        runtime.rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite
        is runtime._ORIGINAL_CAMERA_FACE_LITE
    )
    assert (
        runtime.rgbd_grasp_planner.DenseSceneOccupancy.counts_for_poses
        is runtime._cached_counts_for_poses
    )
    assert (
        runtime.rgbd_grasp_planner.filter_poses_safe_final_dual_arm_ik
        is runtime._mark_post_prefix_filter
    )


def test_install_can_enable_normal_alignment_cache(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.NORMAL_ALIGNMENT_CACHE_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.NORMAL_ALIGNMENT_CACHE_ENV] == "1"
    assert (
        runtime.rgbd_grasp_planner.attach_normal_alignment
        is runtime._cached_normal_alignment
    )


def test_install_can_enable_pose_generation_cache(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.POSE_GENERATION_CACHE_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.POSE_GENERATION_CACHE_ENV] == "1"
    assert (
        runtime.rgbd_grasp_planner.generate_rgbd_filter_poses
        is runtime._cached_rgbd_filter_pose_templates
    )
    assert (
        runtime.rgbd_grasp_planner.generate_local_orientation_refinements
        is runtime._cached_local_orientation_refinements
    )


def test_worker_environment_can_bound_native_cpu_threads(monkeypatch):
    monkeypatch.setenv(runtime.WORKER_CPU_THREADS_ENV, "2")

    environment = runtime._worker_environment("5")

    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert environment[name] == "2"


def test_sticky_signature_lane_prepares_current_signature(monkeypatch):
    monkeypatch.setattr(
        runtime.grasp_kinematics_local,
        "_PersistentIKWorker",
        FakeWorker,
    )
    monkeypatch.setenv(runtime.SIGNATURE_STICKY_LANES_ENV, "1")
    monkeypatch.setenv("OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES", "1")
    request = {"poses": [{"eef_pos": [0.0, 0.0, 0.0]}]}

    runtime._run_policy_ik_arm(
        arm="right",
        gpu="5",
        python_path="python",
        environment={},
        request=request,
        solver_signature="split8-signature",
        timeout_s=1.0,
        prepare_next={
            "request": {"poses": []},
            "solver_signature": "warm32-signature",
        },
    )

    assert len(FakeWorker.instances) == 1
    _request, kwargs = FakeWorker.instances[0].requests[0]
    assert kwargs["prepare_next"] == {
        "request": request,
        "solver_signature": "split8-signature",
    }


def test_install_uses_test_worker_only_with_explicit_physical_batch(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.PHYSICAL_BATCH_ENV, "64")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.PHYSICAL_BATCH_ENV] == "64"
    assert settings["ik_worker_path"] == runtime._TEST_WORKER_PATH
    assert runtime.grasp_kinematics_local._WORKER_PATH == runtime._TEST_WORKER_PATH


def test_install_uses_test_worker_for_prepared_signature_pool(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.PREPARED_SIGNATURE_POOL_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.PREPARED_SIGNATURE_POOL_ENV] == "1"
    assert settings["ik_worker_path"] == runtime._TEST_WORKER_PATH


def test_install_can_reuse_solver_only_when_explicitly_requested(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.FORKSERVER_ENV, "0")
    monkeypatch.setenv(runtime.EXACT_SOLVER_RELOAD_ENV, "0")

    settings = runtime.install_policy_pool_runtime()

    assert settings["forkserver_enabled"] == "0"
    assert settings["exact_solver_reload"] == "0"
    assert settings["OFFICIAL_V2_LITE_FORKSERVER_IK"] == "0"
    assert settings["OFFICIAL_V2_LITE_EXACT_SOLVER_RELOAD"] == "0"


def test_install_uses_test_worker_for_explicit_solver_reset(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.SOLVER_RESET_ENV, "optimizer")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.PHYSICAL_BATCH_ENV] == "0"
    assert settings[runtime.SOLVER_RESET_ENV] == "optimizer"
    assert settings["ik_worker_path"] == runtime._TEST_WORKER_PATH


def test_install_rejects_physical_batch_over_logical_limit(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.PHYSICAL_BATCH_ENV, "65")

    with pytest.raises(ValueError, match="must be in"):
        runtime.install_policy_pool_runtime()


def test_install_can_enable_spawn_render_process_pool(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.RENDER_PROCESSES_ENV, "4")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.RENDER_PROCESSES_ENV] == "4"
    assert settings["render_policy"] == "async_spawn_process_pool"
    assert (
        runtime.rgbd_grasp_lite._lite_render_executor
        is runtime._test_render_executor
    )


def test_install_rejects_excessive_render_processes(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.RENDER_PROCESSES_ENV, "17")

    with pytest.raises(ValueError, match="must be in"):
        runtime.install_policy_pool_runtime()


def test_component_cache_reuses_one_exact_split(monkeypatch):
    class Component:
        is_watertight = True

        def __init__(self, lower, upper):
            self.bounds = __import__("numpy").asarray([lower, upper], dtype=float)

    class Mesh:
        is_watertight = True

        def __init__(self):
            self._official_v2_lite_scene_key = "scene-key"
            self.split_calls = 0
            self.components = (
                Component([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                Component([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]),
            )

        def split(self, *, only_watertight):
            assert only_watertight is False
            self.split_calls += 1
            return self.components

    mesh = Mesh()

    first = runtime._cached_component_record(mesh)
    second = runtime._cached_component_record(mesh)
    bounds = runtime._cached_watertight_component_bounds(mesh)

    assert first is second
    assert mesh.split_calls == 1
    assert bounds.shape == (2, 2, 3)


def test_install_can_enable_component_cache(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.COMPONENT_CACHE_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings[runtime.COMPONENT_CACHE_ENV] == "1"
    assert (
        runtime.rgbd_grasp_lite._lite_watertight_component_bounds
        is runtime._cached_watertight_component_bounds
    )


def test_install_can_keep_official_one_shot_ik(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.POLICY_POOL_ENV, "0")
    monkeypatch.setenv(runtime.SKIP_REACHABLE_SE3_ENV, "0")
    monkeypatch.setenv(runtime.SHARED_MEMORY_ENV, "1")

    settings = runtime.install_policy_pool_runtime()

    assert settings["runtime"] == "official_one_shot_ik"
    assert settings["policy_pool_enabled"] == "0"
    assert settings["lite_skip_reachable_se3_ik_stages"] == "0"
    assert settings["OFFICIAL_V2_LITE_IK_SHARED_MEMORY"] == "1"
    assert settings["OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE"] == "32"
    assert settings["OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE_SIZE"] == "32"
    assert (
        runtime.grasp_kinematics_local._run_persistent_ik_arm
        is runtime._ORIGINAL_RUN_PERSISTENT_IK_ARM
    )
    assert (
        runtime.rgbd_grasp_lite.prepare_external_ik
        is runtime._ORIGINAL_PREPARE_EXTERNAL_IK
    )


def test_install_can_disable_shared_memory_transport(monkeypatch):
    monkeypatch.setenv(runtime.TEST_MODE_ENV, "1")
    monkeypatch.setenv(runtime.POLICY_POOL_ENV, "1")
    monkeypatch.setenv(runtime.SHARED_MEMORY_ENV, "0")

    settings = runtime.install_policy_pool_runtime()

    assert settings["shared_memory_enabled"] == "0"
    assert settings["OFFICIAL_V2_LITE_IK_SHARED_MEMORY"] == "0"
    assert settings["forkserver_enabled"] == "1"
    assert settings["OFFICIAL_V2_LITE_FORKSERVER_IK"] == "1"


def test_policy_pool_reuses_idle_lane_and_separates_arms(monkeypatch):
    monkeypatch.setattr(runtime.grasp_kinematics_local, "_PersistentIKWorker", FakeWorker)

    first, first_reused = runtime._acquire_policy_worker(
        arm="left", gpu="5", timeout_s=1.0
    )
    runtime._release_policy_worker(
        ("5", "left"), first, completed=True
    )
    same, same_reused = runtime._acquire_policy_worker(
        arm="left", gpu="5", timeout_s=1.0
    )
    other_arm, other_arm_reused = runtime._acquire_policy_worker(
        arm="right", gpu="5", timeout_s=1.0
    )

    assert first is same
    assert first_reused is False
    assert same_reused is True
    assert other_arm is not first
    assert other_arm_reused is False
    assert len(FakeWorker.instances) == 2


def test_policy_pool_preserves_planner_lane_affinity(monkeypatch):
    monkeypatch.setattr(
        runtime.grasp_kinematics_local,
        "_PersistentIKWorker",
        FakeWorker,
    )
    monkeypatch.setenv("OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES", "2")

    first, _ = runtime._acquire_policy_worker(
        arm="left", gpu="5", timeout_s=1.0, affinity=101
    )
    runtime._release_policy_worker(("5", "left"), first, completed=True)
    second, _ = runtime._acquire_policy_worker(
        arm="left", gpu="5", timeout_s=1.0, affinity=202
    )
    runtime._release_policy_worker(("5", "left"), second, completed=True)
    same, reused = runtime._acquire_policy_worker(
        arm="left", gpu="5", timeout_s=1.0, affinity=101
    )

    assert second is not first
    assert same is first
    assert reused is True


def test_request_affinity_side_map_does_not_modify_request(monkeypatch):
    request = {"poses": [{"pos": [1.0, 2.0, 3.0]}], "seed": 42}
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_BUILD_EXTERNAL_IK_REQUEST",
        lambda *args, **kwargs: (request, "policy"),
    )

    built, policy = runtime._build_external_ik_request_with_affinity()
    affinity = runtime._consume_request_affinity(built)

    assert built is request
    assert built == {"poses": [{"pos": [1.0, 2.0, 3.0]}], "seed": 42}
    assert policy == "policy"
    assert affinity == threading.get_ident()


def test_shared_memory_transport_moves_request_affinity(monkeypatch):
    request = {"poses": [{"pos": [1.0, 2.0, 3.0]}]}
    transported = {"poses": [], "pose_shared_memory": {"name": "test"}}
    marker = object()
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY",
        lambda value: (transported, marker),
    )
    with runtime._REQUEST_AFFINITY_LOCK:
        runtime._REQUEST_AFFINITY[id(request)] = (request, 303)

    actual, actual_marker = (
        runtime._ik_request_in_shared_memory_with_affinity(request)
    )

    assert actual is transported
    assert actual_marker is marker
    assert runtime._consume_request_affinity(transported) == 303
    assert runtime._consume_request_affinity(request) is None


def test_policy_sharding_separates_solver_signatures(monkeypatch):
    monkeypatch.setattr(
        runtime.grasp_kinematics_local,
        "_PersistentIKWorker",
        FakeWorker,
    )
    monkeypatch.setenv(runtime.POLICY_SHARD_ENV, "1")

    split, _ = runtime._acquire_policy_worker(
        arm="left",
        gpu="5",
        timeout_s=1.0,
        solver_signature="split-signature",
    )
    warm, _ = runtime._acquire_policy_worker(
        arm="left",
        gpu="5",
        timeout_s=1.0,
        solver_signature="warm-signature",
    )

    assert split is not warm
    assert len(FakeWorker.instances) == 2


def test_policy_sharding_prepares_same_fresh_signature(monkeypatch):
    monkeypatch.setattr(
        runtime.grasp_kinematics_local,
        "_PersistentIKWorker",
        FakeWorker,
    )
    monkeypatch.setenv(runtime.POLICY_SHARD_ENV, "1")
    request = {
        "poses": [{"eef_pos": [1, 2, 3]}],
        "solver_policy": "cuda_graph_split8_rewarm",
    }

    runtime._run_policy_ik_arm(
        arm="right",
        gpu="5",
        python_path="ignored",
        environment={},
        request=request,
        solver_signature="split-signature",
        timeout_s=12.0,
        prepare_next={
            "request": {"poses": [], "solver_policy": "warm"},
            "solver_signature": "warm-signature",
        },
    )

    assert FakeWorker.instances[0].requests[0][1]["prepare_next"] == {
        "request": request,
        "solver_signature": "split-signature",
    }


def test_run_policy_arm_forwards_request_without_mutation(monkeypatch):
    monkeypatch.setattr(runtime.grasp_kinematics_local, "_PersistentIKWorker", FakeWorker)
    request = {"poses": [{"eef_pos": [1, 2, 3]}], "solver_policy": "split8"}
    prepare_next = {
        "request": {"poses": [], "solver_policy": "warm32x6"},
        "solver_signature": "next-sig",
    }

    result, reused = runtime._run_policy_ik_arm(
        arm="right",
        gpu="5",
        python_path="ignored",
        environment={},
        request=request,
        solver_signature="sig",
        timeout_s=12.0,
        prepare_next=prepare_next,
    )

    assert result["ok"] is True
    assert reused is False
    worker = FakeWorker.instances[0]
    assert worker.requests[0][0] is request
    assert worker.requests[0][1] == {
        "solver_signature": "sig",
        "timeout_s": 12.0,
        "prepare_next": prepare_next,
    }


def test_run_policy_arm_can_trace_unmodified_request_and_result(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        runtime.grasp_kinematics_local,
        "_PersistentIKWorker",
        FakeWorker,
    )
    monkeypatch.setenv(runtime.IK_TRACE_DIR_ENV, str(tmp_path))
    request = {
        "poses": [{"eef_pos": [1.0, 2.0, 3.0]}],
        "solver_policy": "cuda_graph_split8_rewarm",
    }

    result, _reused = runtime._run_policy_ik_arm(
        arm="right",
        gpu="5",
        python_path="ignored",
        environment={},
        request=request,
        solver_signature="signature",
        timeout_s=12.0,
    )

    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    trace = json.loads(paths[0].read_text())
    assert trace["request"] == request
    assert trace["result"] == result


def test_trace_rejects_ephemeral_shared_memory_descriptor(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.IK_TRACE_DIR_ENV, str(tmp_path))

    with pytest.raises(RuntimeError, match="SHARED_MEMORY=0"):
        runtime._write_ik_trace_request(
            arm="left",
            request={"poses": [], "pose_shared_memory": {"name": "gone"}},
            solver_signature="signature",
        )


def test_policy_pool_allows_configured_parallel_lanes(monkeypatch):
    monkeypatch.setattr(runtime.grasp_kinematics_local, "_PersistentIKWorker", FakeWorker)
    monkeypatch.setenv("OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES", "2")

    first, _ = runtime._acquire_policy_worker(
        arm="right", gpu="5", timeout_s=1.0
    )
    second, _ = runtime._acquire_policy_worker(
        arm="right", gpu="5", timeout_s=1.0
    )

    assert first is not second
    assert len(FakeWorker.instances) == 2
