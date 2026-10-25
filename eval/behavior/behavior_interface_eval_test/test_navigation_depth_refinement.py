"""Oriented chassis sweeps refine depth discretization, never erase obstacles."""

from copy import deepcopy
import math
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from behavior_interface_eval_test.robot_contract import ACTION_DIM, ACTION_SLICES
from behavior_interface_eval_test.tool.official_v2 import map_navigation_local as nav
from behavior_interface_eval_test.tool.official_v2 import navigation_footprint_local as fp
from behavior_interface_eval_test.tool.official_v2 import tools
from behavior_interface_eval_test import test_navigate_to_execution as fixtures


def room(gap=.80, angle=0):
    grid = np.zeros((100, 100), dtype=np.int8)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    y = np.r_[np.arange(-2.5, -gap/2, .025), np.arange(gap/2, 2.51, .025)]
    points = np.column_stack((np.zeros(len(y)), y)) @ rotation.T
    start, goal = np.array([[1.2, 0], [-1.2, 0]]) @ rotation.T
    mask = np.zeros_like(grid, dtype=bool)
    cells = np.floor((points + 2.5) / .05).astype(int)
    valid = (cells >= 0).all(1) & (cells < 100).all(1)
    mask[cells[valid, 1], cells[valid, 0]] = True
    return dict(occupancy=grid, origin=[-2.5, -2.5], resolution=.05,
                pose=dict(x=float(start[0]), y=float(start[1]), yaw_rad=angle),
                places=[dict(name="goal", x=float(goal[0]), y=float(goal[1]))],
                navigation_obstacle_mask=mask, navigation_depth_points_xy_m=points)


def refine(snapshot):
    base = dict(snapshot)
    base.pop("navigation_obstacle_mask")
    plan = nav.plan_clearance_path(base, "goal", safety_margin_fallbacks_m=(.06, .04, .02, 0),
                                   optimize_goal_standoff=True)
    assert plan["ok"], plan
    return nav.plan_depth_refined_path(snapshot, plan, fp.base_navigation_polygon(), fp.BASE_POLYGON_SHA256)


def traversed_stale_wall_room(*, live_gap=.80, angle=0):
    """A mapped wall contradicts a continuous earlier base crossing."""

    snapshot = room(gap=live_gap, angle=angle)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    wall_points = np.column_stack((
        np.zeros(161), np.linspace(-4.0, 4.0, 161),
    )) @ rotation.T
    wall = np.zeros_like(snapshot["occupancy"], dtype=bool)
    cells = np.floor((wall_points + 2.5) / .05).astype(int)
    valid = (cells >= 0).all(1) & (cells < 100).all(1)
    wall[cells[valid, 1], cells[valid, 0]] = True
    free = np.ones_like(wall)
    free[wall] = False
    history = [
        (
            np.array([
                [1.2, offset], [.6, offset], [0.0, offset],
                [-.6, offset], [-1.2, offset],
            ]) @ rotation.T
        ).tolist()
        for offset in (-.10, .10)
    ]
    snapshot.update(
        obstacle_mask=wall.copy(),
        free_mask=free,
        wall_mask=wall.copy(),
        traversed_paths_xy_m=history,
    )
    return snapshot


def test_polygon_and_rotation_sweep_check_middle_not_only_endpoints():
    square = np.array([[-.4, -.1], [.4, -.1], [.4, .1], [-.4, .1]])
    point = np.array([[.28, .28]])
    assert nav.convex_sweep_distances(point, square, [0, 0], [0, 0], 0, 0)[0] > .1
    assert nav.convex_sweep_distances(point, square, [0, 0], [0, 0], math.pi/2, math.pi/2)[0] > .1
    assert nav.convex_sweep_distances(point, square, [0, 0], [0, 0], 0, math.pi/2)[0] == 0


def test_halfspace_distance_is_a_conservative_lower_bound():
    rng = np.random.default_rng(15063)
    polygon = fp.base_navigation_polygon()
    for yaw in np.linspace(-math.pi, math.pi, 9):
        rotation = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        rotated = polygon @ rotation.T
        points = rng.uniform(-.8, .8, (2000, 2))
        lower = nav.convex_point_distance_lower_bounds(points, rotated)
        exact = nav.convex_point_distances(points, rotated)
        assert np.all(lower >= 0.0)
        assert np.all(lower <= exact + 1.0e-12)


def test_tight_retry_keeps_measurement_error_and_nonzero_tracking_room():
    assert nav.DEPTH_POINT_MARGIN_M == pytest.approx(0.02)
    assert nav.DEPTH_TRACKING_RESERVE_M == pytest.approx(0.02)
    assert nav.DEPTH_TIGHT_TRACKING_RESERVE_M == pytest.approx(0.01)
    assert (
        nav.DEPTH_POINT_MARGIN_M + nav.DEPTH_TIGHT_TRACKING_RESERVE_M
    ) == pytest.approx(0.03)


def _valid_partial_turn_certificate():
    path = [[0.0, 0.0], [.25, .10], [.50, .10]]
    arc_yaw = math.radians(-7.5)
    gaps = [1.0, 1.0]
    certificate = {
        "schema": nav.DEPTH_REFINEMENT_SCHEMA,
        "polygon_sha256": fp.BASE_POLYGON_SHA256,
        "segment_yaw_rad": [arc_yaw, arc_yaw],
        "segment_start_yaw_rad": [0.0, arc_yaw],
        "segment_end_yaw_rad": [arc_yaw, arc_yaw],
        "arc_segment_indices": [0],
        "point_margin_m": nav.DEPTH_POINT_MARGIN_M,
        "tracking_reserve_m": nav.DEPTH_TRACKING_RESERVE_M,
        "planning_margin_m": (
            nav.DEPTH_POINT_MARGIN_M + nav.DEPTH_TRACKING_RESERVE_M
        ),
        "rotation_clearance_m": 1.0,
        "segment_clearance_m": gaps,
        "heading_tolerance_deg": 2.0,
        "max_speed_mps": .35,
        "heading_policy": "face_chord_over_45_unless_depth_constrained",
        "orientation_mode": "se2_lattice_certified_partial_turn_and_arc",
        "orientation_constrained_segment_indices": [0],
        "orientation_constrained_max_speed_mps": .15,
        "refresh_depth_after_alignment_deg": 45.0,
        "static_map": {
            "schema": nav.STATIC_FOOTPRINT_SCHEMA,
            "polygon_sha256": fp.BASE_POLYGON_SHA256,
            "planning_margin_m": nav.START_EGRESS_SAFETY_MARGIN_M,
            "unknown_space_blocked": True,
            "grid_cell_obstacles_filled": True,
            "segment_clearance_m": gaps,
            "rotation_clearance_m": 1.0,
            "equivalent_center_clearance_m": [
                tools.BASE_FOOTPRINT_RADIUS_M + gap for gap in gaps
            ],
        },
    }
    return path, certificate


def test_partial_turn_certificate_is_executable_but_cannot_be_weakened():
    path, certificate = _valid_partial_turn_certificate()
    assert tools._navigate_to_validate_depth_refinement(
        certificate, path, failure_stage="planning"
    )["arc_segment_indices"] == [0]

    missing = deepcopy(certificate)
    missing.pop("segment_end_yaw_rad")
    with pytest.raises(tools._NavigateToFailure, match="oriented depth certificate"):
        tools._navigate_to_validate_depth_refinement(
            missing, path, failure_stage="planning"
        )

    too_long_path = deepcopy(path)
    too_long_path[1] = [.31, 0.0]
    with pytest.raises(tools._NavigateToFailure, match="oriented depth certificate"):
        tools._navigate_to_validate_depth_refinement(
            certificate, too_long_path, failure_stage="planning"
        )

    zero_turn = deepcopy(certificate)
    zero_turn["segment_start_yaw_rad"][0] = zero_turn[
        "segment_end_yaw_rad"
    ][0]
    with pytest.raises(tools._NavigateToFailure, match="oriented depth certificate"):
        tools._navigate_to_validate_depth_refinement(
            zero_turn, path, failure_stage="planning"
        )

    unmarked_turn = deepcopy(certificate)
    unmarked_turn["arc_segment_indices"] = []
    with pytest.raises(tools._NavigateToFailure, match="oriented depth certificate"):
        tools._navigate_to_validate_depth_refinement(
            unmarked_turn, path, failure_stage="planning"
        )


def test_tight_margin_certificate_is_explicit_and_speed_constrained():
    path, certificate = _valid_partial_turn_certificate()
    certificate["tracking_reserve_m"] = (
        nav.DEPTH_TIGHT_TRACKING_RESERVE_M
    )
    certificate["planning_margin_m"] = (
        nav.DEPTH_POINT_MARGIN_M + nav.DEPTH_TIGHT_TRACKING_RESERVE_M
    )
    certificate["margin_policy"] = nav.DEPTH_TIGHT_MARGIN_POLICY
    certificate["segment_clearance_m"] = [0.035, 1.0]
    certificate["tight_margin_segment_indices"] = [0]
    assert tools._navigate_to_validate_depth_refinement(
        certificate, path, failure_stage="planning"
    )["margin_policy"] == nav.DEPTH_TIGHT_MARGIN_POLICY

    unconstrained = deepcopy(certificate)
    unconstrained["orientation_constrained_segment_indices"] = []
    with pytest.raises(tools._NavigateToFailure):
        tools._navigate_to_validate_depth_refinement(
            unconstrained, path, failure_stage="planning"
        )

    unlabelled = deepcopy(certificate)
    unlabelled.pop("margin_policy")
    with pytest.raises(tools._NavigateToFailure):
        tools._navigate_to_validate_depth_refinement(
            unlabelled, path, failure_stage="planning"
        )


@pytest.mark.parametrize("angle", [0, math.radians(31)])
def test_narrow_door_uses_a_certified_orientation_at_any_scene_rotation(angle):
    snapshot = room(angle=angle)
    before = snapshot["occupancy"].copy()
    result = refine(snapshot)
    assert result is not None
    certificate = result["depth_refinement"]
    assert min(certificate["segment_clearance_m"]) >= certificate["planning_margin_m"]
    assert certificate["rotation_clearance_m"] >= certificate["planning_margin_m"]
    assert len(set(certificate["segment_yaw_rad"])) == 1
    assert certificate["heading_policy"] == (
        "face_chord_over_45_unless_depth_constrained"
    )
    previous_yaw = snapshot["pose"]["yaw_rad"]
    constrained = set(certificate["orientation_constrained_segment_indices"])
    for index, (start, end, segment_yaw) in enumerate(zip(
        result["path_xy_m"][:-1],
        result["path_xy_m"][1:],
        certificate["segment_yaw_rad"],
    )):
        chord_yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        chord_delta = (chord_yaw - previous_yaw + math.pi) % (2 * math.pi) - math.pi
        if index not in constrained and abs(chord_delta) > math.radians(45):
            assert segment_yaw == pytest.approx(chord_yaw)
        previous_yaw = segment_yaw
    assert result["minimum_clearance_m"] >= result["required_clearance_m"]
    np.testing.assert_array_equal(snapshot["occupancy"], before)
    tools._navigate_to_validate_depth_refinement(certificate, result["path_xy_m"], failure_stage="planning")


def test_too_small_gap_and_unknown_space_remain_blocked():
    assert refine(room(gap=.60)) is None
    snapshot = room()
    snapshot.pop("navigation_depth_points_xy_m")
    assert refine(snapshot) is None


@pytest.mark.parametrize("angle", [0, math.radians(31)])
def test_depth_refinement_replays_audited_traversed_map_evidence(angle):
    snapshot = traversed_stale_wall_room(angle=angle)
    obstacle_before = snapshot["obstacle_mask"].copy()
    free_before = snapshot["free_mask"].copy()
    map_only = dict(snapshot)
    map_only.pop("navigation_obstacle_mask")
    plan = nav.plan_clearance_path(
        map_only, "goal", safety_margin_fallbacks_m=(.06, .04, .02, 0),
        optimize_goal_standoff=True,
    )
    assert plan["ok"], plan
    assert plan["planning_evidence_trials"][
        plan["planning_evidence_trial_index"]
    ] == "traversed_swept_space_override"

    result = nav.plan_depth_refined_path(
        snapshot, plan, fp.base_navigation_polygon(), fp.BASE_POLYGON_SHA256,
    )

    assert result is not None
    assert result["traversed_route_direct_replay"] is False
    evidence = result["depth_refinement"]["map_evidence"]
    assert evidence["schema"] == "reference_traversed_evidence_v1"
    assert evidence["mode"] == "traversed_swept_space_override"
    assert evidence["overridden_wall_obstacle_cells"] > 0
    np.testing.assert_array_equal(snapshot["obstacle_mask"], obstacle_before)
    np.testing.assert_array_equal(snapshot["free_mask"], free_before)


def test_history_guidance_without_map_override_needs_no_evidence_replay():
    occupancy = np.zeros((140, 140), dtype=np.int8)
    occupancy[[0, -1], :] = 100
    occupancy[:, [0, -1]] = 100
    start = (2.05, 7.05)
    goal = (11.05, 7.05)
    history = [(2.05, y) for y in np.linspace(7.05, 3.05, 41)]
    history += [(x, 3.05) for x in np.linspace(2.15, 11.05, 90)]
    history += [(11.05, y) for y in np.linspace(3.15, 7.05, 40)]
    snapshot = {
        "occupancy": occupancy,
        "origin": [0.0, 0.0],
        "resolution": 0.10,
        "pose": {"x": start[0], "y": start[1], "yaw_rad": 0.0},
        "places": [{"name": "goal", "x": goal[0], "y": goal[1]}],
        "traversed_paths_xy_m": [history],
    }
    plan = nav.plan_clearance_path(
        snapshot,
        "goal",
        robot_radius_m=0.20,
        safety_margin_m=0.05,
        safety_margin_fallbacks_m=(0.0,),
        clearance_weight=0.0,
    )

    assert plan["ok"], plan
    assert plan["traversed_route_guidance_selected"]
    assert not plan["traversed_route_evidence_used"]
    replayed, audit = nav._replay_reference_traversed_evidence(
        snapshot, plan
    )
    assert audit is None
    assert replayed is snapshot


def test_traversed_map_evidence_does_not_erase_a_current_narrow_door():
    snapshot = traversed_stale_wall_room(live_gap=.60)
    map_only = dict(snapshot)
    map_only.pop("navigation_obstacle_mask")
    plan = nav.plan_clearance_path(
        map_only, "goal", safety_margin_fallbacks_m=(.06, .04, .02, 0),
        optimize_goal_standoff=True,
    )
    assert plan["ok"] and plan["traversed_route_evidence_used"]
    assert nav.plan_depth_refined_path(
        snapshot, plan, fp.base_navigation_polygon(), fp.BASE_POLYGON_SHA256,
    ) is None


def test_runtime_uses_active_margin_and_oriented_stopping_sweep():
    ctx = SimpleNamespace(world=SimpleNamespace())
    setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
    action = np.zeros(ACTION_DIM)
    action[ACTION_SLICES["base"]] = [.2, 0, 0]
    with mock.patch.object(tools, "_navigate_to_depth_obstacles", return_value=np.array([[.1, .47, .25]])):
        assert not tools._navigate_to_depth_motion_clear(ctx, {}, action)
        assert tools._navigate_to_depth_motion_clear(ctx, {}, action, {"clearance": {"effective_safety_margin_m": .02}})
    precise = {"depth_refinement": {"point_margin_m": .02}}
    with mock.patch.object(tools, "_navigate_to_depth_obstacles", return_value=np.array([[.1, .40, .25]])):
        assert tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)
    with mock.patch.object(tools, "_navigate_to_depth_obstacles", return_value=np.array([[.30, 0, .25]])):
        action[ACTION_SLICES["base"]] = [1, 0, 0]
        assert not tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)


def test_invalid_precise_certificate_is_not_executable():
    with pytest.raises(tools._NavigateToFailure, match="oriented depth certificate"):
        tools._navigate_to_validate_depth_refinement({"schema": nav.DEPTH_REFINEMENT_SCHEMA}, [[0, 0], [1, 0]], failure_stage="planning")


@pytest.mark.parametrize("yaw_correction", [0, -10])
def test_signed_heading_is_executed_without_turning_back_toward_the_chord(yaw_correction):
    fixture = fixtures.NavigateToExecutionTest
    adapter, world = fixture._adapter_world()
    snapshot = fixture._snapshot(world)
    state = fixture._install_snapshot_provider(adapter, snapshot)
    ctx, results, setter = fixture._ctx(world)
    calls = []
    original = tools.plan_clearance_path
    corrected = False

    def after_phase(kind, current, _world):
        nonlocal corrected
        if kind == "spin" and not corrected and yaw_correction:
            current["snapshot"]["pose"]["yaw_deg"] += yaw_correction
            corrected = True

    def planner(*args, **kwargs):
        result = original(*args, **kwargs)
        count = len(result["path_xy_m"]) - 1
        result["depth_refinement"] = {
            "schema": nav.DEPTH_REFINEMENT_SCHEMA, "polygon_sha256": fp.BASE_POLYGON_SHA256,
            "segment_yaw_rad": [math.pi/2] * count, "segment_clearance_m": [1.0] * count,
            "point_margin_m": .02, "tracking_reserve_m": .02, "planning_margin_m": .04,
            "rotation_clearance_m": 1.0, "heading_tolerance_deg": 2.0, "max_speed_mps": .35,
        }
        return result

    with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}):
        with mock.patch.object(tools, "plan_clearance_path", side_effect=planner):
            with mock.patch.object(tools, "_yield_adjust_chassis_controller", side_effect=fixture._controller_stub(state, calls, after_phase=after_phase)):
                list(fixtures.build_registry(adapter)["navigate_to"].fn(
                    ctx, name=fixtures.GOAL_NAME, session_id="oriented-navigation-test", timeout_s=10))
    assert setter.call_count == 1
    assert results[0]["ok"], results[0]
    spins = [call for call in calls if call["kind"] == "spin"]
    assert len(spins) == (2 if yaw_correction else 1)
    assert spins[0]["spin"] == pytest.approx(90)
    assert state["snapshot"]["pose"]["yaw_deg"] == pytest.approx(90)
    assert all(call["kwargs"]["vmax"] == .35 for call in calls if call["kind"] == "forward")
    assert all(call["kwargs"]["_componentwise_linear_limit"] is False
               for call in calls if call["kind"] == "forward")


def test_partial_turn_plan_is_dispatched_as_one_coupled_motion():
    fixture = fixtures.NavigateToExecutionTest
    adapter, world = fixture._adapter_world()
    snapshot = fixture._snapshot(
        world, start_cell=(30, 10), goal_cell=(31, 12)
    )
    state = fixture._install_snapshot_provider(adapter, snapshot)
    ctx, results, setter = fixture._ctx(world)
    original = tools.plan_clearance_path
    calls = []
    arc_yaw = math.radians(-7.5)

    def planner(*args, **kwargs):
        result = original(*args, **kwargs)
        count = len(result["path_xy_m"]) - 1
        assert count >= 1
        assert math.dist(
            result["path_xy_m"][0], result["path_xy_m"][1]
        ) <= nav.PARTIAL_TURN_MAX_TRANSLATION_M
        center_clearances = [
            float(value) for value in result["path_segment_clearance_m"]
        ]
        static_gaps = [
            value - tools.BASE_FOOTPRINT_RADIUS_M
            for value in center_clearances
        ]
        result["depth_refinement"] = {
            "schema": nav.DEPTH_REFINEMENT_SCHEMA,
            "polygon_sha256": fp.BASE_POLYGON_SHA256,
            "segment_yaw_rad": [arc_yaw] * count,
            "segment_start_yaw_rad": [0.0] + [arc_yaw] * (count - 1),
            "segment_end_yaw_rad": [arc_yaw] * count,
            "arc_segment_indices": [0],
            "segment_clearance_m": [1.0] * count,
            "point_margin_m": nav.DEPTH_POINT_MARGIN_M,
            "tracking_reserve_m": nav.DEPTH_TRACKING_RESERVE_M,
            "planning_margin_m": (
                nav.DEPTH_POINT_MARGIN_M + nav.DEPTH_TRACKING_RESERVE_M
            ),
            "rotation_clearance_m": 1.0,
            "heading_tolerance_deg": 2.0,
            "max_speed_mps": .35,
            "heading_policy": "face_chord_over_45_unless_depth_constrained",
            "orientation_mode": "se2_lattice_certified_partial_turn_and_arc",
            "orientation_constrained_segment_indices": [0],
            "orientation_constrained_max_speed_mps": .15,
            "refresh_depth_after_alignment_deg": 45.0,
            "static_map": {
                "schema": nav.STATIC_FOOTPRINT_SCHEMA,
                "polygon_sha256": fp.BASE_POLYGON_SHA256,
                "planning_margin_m": nav.START_EGRESS_SAFETY_MARGIN_M,
                "unknown_space_blocked": True,
                "grid_cell_obstacles_filled": True,
                "segment_clearance_m": static_gaps,
                "rotation_clearance_m": 1.0,
                "equivalent_center_clearance_m": center_clearances,
            },
        }
        return result

    def controller(inner_ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs):
        calls.append({
            "forward": float(forward),
            "translation": float(translation),
            "spin": float(spin),
            "kwargs": dict(kwargs),
        })

        def phase():
            yield inner_ctx.world.make_action(base=[.1, .1, -.1])
            pose = state["snapshot"]["pose"]
            yaw = math.radians(float(pose["yaw_deg"]))
            pose["x"] += (
                math.cos(yaw) * float(forward)
                - math.sin(yaw) * float(translation)
            )
            pose["y"] += (
                math.sin(yaw) * float(forward)
                + math.cos(yaw) * float(translation)
            )
            pose["yaw_deg"] += float(spin)
            fixture._advance_snapshot_pose(state["snapshot"])
            return {
                "ok": True,
                "action_steps": 1,
                "linear_remaining_m": 0.0,
                "obstacle_stop_reason": None,
            }

        return phase()

    with tempfile.TemporaryDirectory() as root, mock.patch.dict(
        os.environ, {"BEHAVIOR_AGENT_RUNS": root}
    ):
        with mock.patch.object(tools, "plan_clearance_path", side_effect=planner):
            with mock.patch.object(
                tools, "_yield_adjust_chassis_controller", side_effect=controller
            ):
                list(fixtures.build_registry(adapter)["navigate_to"].fn(
                    ctx,
                    name=fixtures.GOAL_NAME,
                    session_id="coupled-partial-turn-test",
                    timeout_s=10,
                    arrival_tolerance_m=.10,
                ))

    assert setter.call_count == 1
    assert results[0]["ok"], results[0]
    coupled = [
        call for call in calls if call["kwargs"].get("_coupled_linear_spin")
    ]
    assert len(coupled) == 1
    assert math.hypot(coupled[0]["forward"], coupled[0]["translation"]) > 0.0
    assert coupled[0]["spin"] == pytest.approx(-7.5)


def test_compressed_disks_enclose_every_original_depth_point():
    points = np.random.default_rng(92).uniform([-.3, .4], [.3, .41], (2000, 2))
    centers, radii = nav.compact_depth_disks(points)
    assert len(centers) < len(points) / 4
    distance = np.linalg.norm(points[:, None] - centers[None], axis=2) - radii
    assert np.all(distance.min(axis=1) <= 1e-12)
    polygon = fp.base_navigation_polygon()
    exact = nav.convex_sweep_distances(points, polygon, [0, 0], [.2, 0], 0, .3).min()
    conservative = (nav.convex_sweep_distances(centers, polygon, [0, 0], [.2, 0], 0, .3) - radii).min()
    assert conservative <= exact + 1e-12


def test_close_start_can_translate_away_but_not_spin_or_reduce_its_gap():
    snapshot = room()
    polygon = fp.base_navigation_polygon()
    wall_y = polygon[:, 1].max() + .018
    points = np.column_stack((np.linspace(-2.4, 2.4, 121), np.full(121, wall_y)))
    snapshot["navigation_depth_points_xy_m"] = points
    snapshot["places"][0]["y"] = -.5
    result = refine(snapshot)
    assert result is not None
    certificate = result["depth_refinement"]
    assert .018 <= certificate["start_egress"]["initial_clearance_m"] < .02
    assert certificate["segment_yaw_rad"][0] == 0
    path = result["path_xy_m"]
    initial = nav.convex_sweep_distances(points, polygon, path[0], path[0], 0, 0)
    swept = nav.convex_sweep_distances(points, polygon, path[0], path[1], 0, 0)
    assert np.all(swept >= np.minimum(initial, .04) - 1e-9)
    tools._navigate_to_validate_depth_refinement(certificate, path, failure_stage="planning")
    certificate["start_egress"]["separating"] = False
    with pytest.raises(tools._NavigateToFailure):
        tools._navigate_to_validate_depth_refinement(certificate, path, failure_stage="planning")


def test_precise_stopping_guard_allows_only_monotone_positive_gap_egress():
    ctx = SimpleNamespace(world=SimpleNamespace())
    setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
    polygon = fp.base_navigation_polygon()
    edge = int(np.argmax(polygon[:, 1]))
    close = np.array([[*polygon[edge], .25]])
    close[0, 1] += .015
    action = np.zeros(ACTION_DIM)
    precise = {"depth_refinement": {"point_margin_m": .02}}
    with mock.patch.object(tools, "_navigate_to_depth_obstacles", return_value=close):
        action[ACTION_SLICES["base"]] = [0, -.2, 0]
        assert tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)
        action[ACTION_SLICES["base"]] = [0, .2, 0]
        assert not tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)
        action[ACTION_SLICES["base"]] = [0, 0, .2]
        assert not tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)
        close[0, 1] -= .02
        action[ACTION_SLICES["base"]] = [0, -.2, 0]
        assert not tools._navigate_to_depth_motion_clear(ctx, {}, action, precise)
