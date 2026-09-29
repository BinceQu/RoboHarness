"""Deterministic full-loop benchmark for arbitrary tracked-point targets."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import behavior_interface_eval_test.test_official_move_tracked_point as official_test
from behavior_interface_eval_test.run_live_move_tracked_point_30 import (
    nontrivial_constraint_error_floor,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    arm_joint_limits,
)
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    _point_positions,
    evaluate_target_constraints,
    rigid_anchors_from_points,
)


BENCHMARK_SEED = 20260828
POSITION_TOLERANCE_M = 0.008
MIN_INITIAL_CONSTRAINT_ERROR_M = 0.014
# The chained generator uses the physical point-to-affine-subspace error.
# Partially constrained relative pairs can have a meaningful motion whose
# largest constrained-coordinate correction is below the old equation-residual
# threshold.  The shared helper keeps a small margin above the public
# tolerance, so generated cases are still genuinely outside the success band.
CHAINED_MIN_INITIAL_CONSTRAINT_ERROR_M = nontrivial_constraint_error_floor(
    POSITION_TOLERANCE_M
)
CASE_PATTERNS = (
    "single_numeric_xyz",
    "single_numeric_x_free_yz",
    "pair_numeric_xyz",
    "first_numeric_second_free",
    "relative_xyz",
    "relative_xy",
    "relative_yz",
    "relative_xz",
    "mixed_absolute_affine",
    "cross_axis_sum",
    "cross_axis_difference",
    "cross_axis_three_rows",
    "two_variable_affine",
    "mixed_numeric_slots",
)
CHAINED_CASE_PATTERNS = CASE_PATTERNS * 2 + CASE_PATTERNS[:2]


def _offset_expression(variable: str, offset: float) -> str:
    return f"{variable}{float(offset):+.12g}"


def _target_rows(
    pattern: str,
    target: np.ndarray,
    variable_prefix: str,
) -> list[list[float | str]]:
    first = np.asarray(target[0], dtype=np.float64)
    second = np.asarray(target[1], dtype=np.float64)
    delta = second - first
    vx = f"{variable_prefix}_x"
    vy = f"{variable_prefix}_y"
    vz = f"{variable_prefix}_z"

    if pattern == "pair_numeric_xyz":
        return [first.astype(float).tolist(), second.astype(float).tolist()]
    if pattern == "first_numeric_second_free":
        return [first.astype(float).tolist(), ["?", "?", "?"]]
    if pattern == "relative_xyz":
        return [
            [vx, vy, vz],
            [
                _offset_expression(vx, delta[0]),
                _offset_expression(vy, delta[1]),
                _offset_expression(vz, delta[2]),
            ],
        ]
    if pattern == "relative_xy":
        return [
            [vx, vy, "?"],
            [
                _offset_expression(vx, delta[0]),
                _offset_expression(vy, delta[1]),
                "?",
            ],
        ]
    if pattern == "relative_yz":
        return [
            ["?", vy, vz],
            [
                "?",
                _offset_expression(vy, delta[1]),
                _offset_expression(vz, delta[2]),
            ],
        ]
    if pattern == "relative_xz":
        return [
            [vx, "?", vz],
            [
                _offset_expression(vx, delta[0]),
                "?",
                _offset_expression(vz, delta[2]),
            ],
        ]
    if pattern == "mixed_absolute_affine":
        return [
            [float(first[0]), vy, "?"],
            [float(second[0]), _offset_expression(vy, delta[1]), "?"],
        ]
    if pattern == "cross_axis_sum":
        total = float(first[0] + second[1])
        return [
            [vx, "?", "?"],
            ["?", f"-{vx}{total:+.12g}", "?"],
        ]
    if pattern == "cross_axis_difference":
        offset = float(second[0] - first[2])
        return [
            ["?", "?", vz],
            [_offset_expression(vz, offset), "?", "?"],
        ]
    if pattern == "cross_axis_three_rows":
        return [
            [vx, _offset_expression(vx, first[1] - first[0]), "?"],
            ["?", "?", _offset_expression(vx, second[2] - first[0])],
        ]
    if pattern == "two_variable_affine":
        vu = f"{variable_prefix}_u"
        vv = f"{variable_prefix}_v"
        u_value = 0.5 * float(first[0])
        v_value = float(first[0]) - u_value
        first_y_offset = float(first[1] - u_value + v_value)
        second_z_offset = float(second[2] - v_value)
        return [
            [f"{vu}+{vv}", f"{vu}-{vv}{first_y_offset:+.12g}", "?"],
            ["?", "?", _offset_expression(vv, second_z_offset)],
        ]
    if pattern == "mixed_numeric_slots":
        return [
            [float(first[0]), "?", float(first[2])],
            ["?", float(second[1]), "?"],
        ]
    raise AssertionError(f"unsupported benchmark pattern {pattern!r}")


class MoveTrackedPointArbitraryBenchmarkTest(unittest.TestCase):
    """Run every generated input through the public official_v2 tool."""

    def _run_case(self, case_index: int, pattern: str) -> None:
        self.assertGreaterEqual(len(CASE_PATTERNS), 10)
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (eef_position, eef_quaternion) = (
            official_test.OfficialMoveTrackedPointTest._left_eef(world)
        )
        eef_position = np.asarray(eef_position, dtype=np.float64)
        q_start = np.asarray(world.arm_qpos_list("left"), dtype=np.float64)
        rng = np.random.default_rng(BENCHMARK_SEED + 1009 * case_index)

        if pattern.startswith("single_"):
            point_name = f"marker_{case_index:02d}_{int(rng.integers(1000, 9999))}"
            source = eef_position.reshape(1, 3)
            displacement = rng.uniform(-0.018, 0.018, size=3)
            displacement[np.argmax(np.abs(displacement))] += (
                0.018 if displacement[np.argmax(np.abs(displacement))] >= 0.0 else -0.018
            )
            if pattern == "single_numeric_xyz":
                rows: list[list[float | str]] = [
                    (eef_position + displacement).astype(float).tolist()
                ]
            else:
                x_offset = 0.018 + 0.004 * float(rng.random())
                rows = [[float(eef_position[0] + x_offset), "?", "?"]]
            names = (point_name,)
            target_points = [
                {"name": point_name, "target_xyz_m": rows[0]}
            ]
        else:
            direction = rng.normal(size=3)
            direction /= float(np.linalg.norm(direction))
            separation = float(rng.uniform(0.075, 0.115))
            center_offset = rng.uniform(-0.012, 0.012, size=3)
            source = np.asarray(
                [
                    eef_position + center_offset - 0.5 * separation * direction,
                    eef_position + center_offset + 0.5 * separation * direction,
                ],
                dtype=np.float64,
            )
            anchors = rigid_anchors_from_points(
                eef_position,
                eef_quaternion,
                source,
            )
            lower, upper = arm_joint_limits(state, "left")
            target = None
            rows = None
            for attempt in range(96):
                # Partial relative constraints may need a larger orientation
                # change before their constrained-coordinate error exceeds the
                # public tolerance.  Increase the deterministic search radius
                # gradually instead of weakening the acceptance assertion.
                amplitude = 0.07 + 0.003 * attempt
                q_target = q_start.copy()
                q_target[:7] = np.clip(
                    q_start[:7] + rng.uniform(-amplitude, amplitude, size=7),
                    np.asarray(lower[:7], dtype=np.float64),
                    np.asarray(upper[:7], dtype=np.float64),
                )
                if ARM_DOF == 8:
                    q_target[7] = 0.0
                candidate_target, _target_eef, _target_quaternion = _point_positions(
                    state,
                    "left",
                    q_target,
                    anchors,
                )
                variable_prefix = (
                    f"v{case_index}_{int(rng.integers(10000, 99999))}"
                )
                candidate_rows = _target_rows(
                    pattern,
                    candidate_target,
                    variable_prefix,
                )
                candidate_names = (
                    f"marker_{case_index:02d}_{int(rng.integers(1000, 9999))}_a",
                    f"marker_{case_index:02d}_{int(rng.integers(1000, 9999))}_b",
                )
                candidate_points = [
                    {"name": name, "target_xyz_m": row}
                    for name, row in zip(candidate_names, candidate_rows)
                ]
                target_report = evaluate_target_constraints(
                    candidate_target,
                    candidate_points,
                    tolerance_m=1e-8,
                )
                initial_report = evaluate_target_constraints(
                    source,
                    candidate_points,
                    tolerance_m=POSITION_TOLERANCE_M,
                )
                initial_error = float(initial_report["max_constraint_error_m"])
                moved_distance = float(
                    np.max(np.linalg.norm(candidate_target - source, axis=1))
                )
                if (
                    target_report["ok"]
                    and initial_error
                    >= nontrivial_constraint_error_floor(POSITION_TOLERANCE_M)
                    and moved_distance >= 0.014
                ):
                    target = candidate_target
                    rows = candidate_rows
                    names = candidate_names
                    target_points = candidate_points
                    break
            self.assertIsNotNone(
                target,
                f"seeded generator could not make nontrivial case {pattern}",
            )
            self.assertIsNotNone(rows)

        initial_report = evaluate_target_constraints(
            source,
            target_points,
            tolerance_m=POSITION_TOLERANCE_M,
        )
        self.assertFalse(initial_report["ok"], initial_report)
        world._official_tracked_object_distances = official_test._RigidTrackedManager(
            world,
            adapter,
            dict(zip(names, source)),
        )
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            actions = official_test.OfficialMoveTrackedPointTest._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    points=target_points,
                    pos_tol=POSITION_TOLERANCE_M,
                    ori_tol_deg=5.0,
                    max_steps=520,
                    timeout_s=90.0,
                ),
            )

        context = {
            "case_index": case_index,
            "pattern": pattern,
            "seed": BENCHMARK_SEED + 1009 * case_index,
            "target_points": target_points,
            "result": result,
        }
        self.assertTrue(result.get("ok"), context)
        self.assertTrue(result["motion"]["ok"], context)
        self.assertTrue(result["final_constraints"]["ok"], context)
        self.assertEqual(
            result["planner"]["endpoint_selected_candidate"]["mode"],
            "joint_space_affine_constraint_ik",
            context,
        )
        active_slice = ACTION_SLICES[f"arm_{result['arm']}"]
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,), context)
            self.assertTrue(np.isfinite(action).all(), context)
        self.assertTrue(
            any(
                float(
                    np.linalg.norm(
                        np.asarray(action[active_slice], dtype=np.float64)
                        - q_start,
                        ord=np.inf,
                    )
                )
                > 1e-4
                for action in actions
            ),
            context,
        )
        if ARM_DOF == 8:
            for action in actions:
                self.assertAlmostEqual(
                    float(action[active_slice][7]),
                    0.0,
                    places=12,
                    msg=str(context),
                )

    def test_30_chained_inputs_start_from_30_distinct_live_poses(self) -> None:
        self.assertEqual(len(CHAINED_CASE_PATTERNS), 30)
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (eef_position, _eef_quaternion) = (
            official_test.OfficialMoveTrackedPointTest._left_eef(world)
        )
        eef_position = np.asarray(eef_position, dtype=np.float64)
        names = ("chain_head", "chain_tail")
        initial_points = {
            names[0]: eef_position.copy(),
            names[1]: eef_position + np.array([0.035, 0.068, 0.024]),
        }
        manager = official_test._RigidTrackedManager(
            world, adapter, initial_points
        )
        world._official_tracked_object_distances = manager
        anchors = np.asarray(
            [manager.anchors[name] for name in names], dtype=np.float64
        )
        rng = np.random.default_rng(BENCHMARK_SEED + 30_000)
        initial_pose_keys: list[tuple[float, ...]] = []
        case_reports: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ):
            for case_index, pattern in enumerate(CHAINED_CASE_PATTERNS):
                live = manager.observed_active_points_snapshot(
                    names,
                    episode_id=world.episode_id(),
                )
                self.assertTrue(live["ok"], live)
                source = np.asarray(
                    [
                        live["entries"][name]["xyz_in_robot_base_coord_m"]
                        for name in names
                    ],
                    dtype=np.float64,
                )
                q_start = np.asarray(
                    world.arm_qpos_list("left"), dtype=np.float64
                )
                initial_pose_keys.append(
                    tuple(np.round(q_start[: min(7, ARM_DOF)], 6))
                )

                if pattern.startswith("single_"):
                    direction = np.array(
                        [
                            math.sin(0.7 * (case_index + 1)),
                            math.cos(0.9 * (case_index + 1)),
                            math.sin(1.3 * (case_index + 1)),
                        ],
                        dtype=np.float64,
                    )
                    direction /= float(np.linalg.norm(direction))
                    delta = direction * (0.014 + 0.001 * (case_index % 4))
                    if pattern == "single_numeric_x_free_yz":
                        delta[0] = math.copysign(
                            0.016 + 0.001 * (case_index % 3),
                            delta[0] if delta[0] != 0.0 else 1.0,
                        )
                        row: list[float | str] = [
                            float(source[0, 0] + delta[0]),
                            "?",
                            "?",
                        ]
                    else:
                        row = (source[0] + delta).astype(float).tolist()
                    target_points = [
                        {"name": names[0], "target_xyz_m": row}
                    ]
                else:
                    current_state, _current_eef = (
                        official_test.OfficialMoveTrackedPointTest._left_eef(
                            world
                        )
                    )
                    lower, upper = arm_joint_limits(current_state, "left")
                    target_points = None
                    for attempt in range(160):
                        amplitude = 0.065 + 0.002 * attempt
                        q_target = q_start.copy()
                        q_target[:7] = np.clip(
                            q_start[:7]
                            + rng.uniform(-amplitude, amplitude, size=7),
                            np.asarray(lower[:7], dtype=np.float64),
                            np.asarray(upper[:7], dtype=np.float64),
                        )
                        if ARM_DOF == 8:
                            q_target[7] = 0.0
                        candidate_target, _target_eef, _target_quaternion = (
                            _point_positions(
                                current_state,
                                "left",
                                q_target,
                                anchors,
                            )
                        )
                        rows = _target_rows(
                            pattern,
                            candidate_target,
                            f"chain_{case_index}_{attempt}",
                        )
                        candidate_points = [
                            {"name": name, "target_xyz_m": row}
                            for name, row in zip(names, rows)
                        ]
                        target_report = evaluate_target_constraints(
                            candidate_target,
                            candidate_points,
                            tolerance_m=1e-8,
                        )
                        initial_report = evaluate_target_constraints(
                            source,
                            candidate_points,
                            tolerance_m=POSITION_TOLERANCE_M,
                        )
                        initial_error = float(
                            initial_report["max_constraint_error_m"]
                        )
                        moved_distance = float(
                            np.max(
                                np.linalg.norm(candidate_target - source, axis=1)
                            )
                        )
                        if (
                            target_report["ok"]
                            and initial_error
                            >= nontrivial_constraint_error_floor(
                                POSITION_TOLERANCE_M
                            )
                            and moved_distance >= 0.014
                        ):
                            target_points = candidate_points
                            break
                    self.assertIsNotNone(
                        target_points,
                        f"could not generate chained case {case_index}: {pattern}",
                    )

                # A one-point request intentionally binds only its named
                # marker.  The live manager may retain additional markers for
                # later chained cases, but those unrequested points are not
                # part of this endpoint's source/target correspondence.
                request_source = source[:1] if pattern.startswith("single_") else source
                initial_report = evaluate_target_constraints(
                    request_source,
                    target_points,
                    tolerance_m=POSITION_TOLERANCE_M,
                )
                self.assertFalse(
                    initial_report["ok"],
                    {"case": case_index, "pattern": pattern, "report": initial_report},
                )
                ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(
                    world
                )
                actions = official_test.OfficialMoveTrackedPointTest._drive(
                    adapter,
                    world,
                    build_registry(adapter)["move_tracked_point"].fn(
                        ctx,
                        points=target_points,
                        pos_tol=POSITION_TOLERANCE_M,
                        ori_tol_deg=5.0,
                        max_steps=520,
                        timeout_s=90.0,
                    ),
                )
                context = {
                    "case_index": case_index,
                    "pattern": pattern,
                    "target_points": target_points,
                    "result": result,
                }
                self.assertTrue(result.get("ok"), context)
                self.assertTrue(result["motion"]["ok"], context)
                self.assertTrue(result["final_constraints"]["ok"], context)
                q_after = np.asarray(
                    world.arm_qpos_list("left"), dtype=np.float64
                )
                self.assertGreater(
                    float(np.linalg.norm(q_after - q_start, ord=np.inf)),
                    1e-5,
                    context,
                )
                self.assertTrue(actions, context)
                if ARM_DOF == 8:
                    active_slice = ACTION_SLICES["arm_left"]
                    for action in actions:
                        self.assertAlmostEqual(
                            float(action[active_slice][7]), 0.0, places=12
                        )
                case_reports.append(
                    {
                        "case_index": case_index,
                        "pattern": pattern,
                        "start_q": q_start.astype(float).tolist(),
                        "final_q": q_after.astype(float).tolist(),
                        "max_constraint_error_m": result[
                            "final_constraints"
                        ]["max_constraint_error_m"],
                    }
                )

        self.assertEqual(len(case_reports), 30)
        self.assertEqual(len(set(initial_pose_keys)), 30, case_reports)


def _install_case_test(case_index: int, pattern: str) -> None:
    def test_case(self: MoveTrackedPointArbitraryBenchmarkTest) -> None:
        self._run_case(case_index, pattern)

    test_case.__name__ = f"test_case_{case_index + 1:02d}_{pattern}"
    test_case.__qualname__ = (
        f"MoveTrackedPointArbitraryBenchmarkTest.{test_case.__name__}"
    )
    test_case.__doc__ = (
        f"Full-loop arbitrary-input case {case_index + 1}: {pattern}."
    )
    setattr(
        MoveTrackedPointArbitraryBenchmarkTest,
        test_case.__name__,
        test_case,
    )


for _case_index, _pattern in enumerate(CASE_PATTERNS):
    _install_case_test(_case_index, _pattern)


if __name__ == "__main__":
    unittest.main()
