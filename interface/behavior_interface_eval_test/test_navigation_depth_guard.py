"""Navigation-only live depth evidence must not mutate or be erased by SLAM."""

from types import SimpleNamespace
import itertools
import math
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from behavior_interface_eval_test.robot_contract import ACTION_DIM, ACTION_SLICES
from behavior_interface_eval_test.tool.official_v2 import tools
from behavior_interface_eval_test.tool.official_v2 import map_navigation_local as nav
from behavior_interface_eval_test.tool.official_v2 import navigation_footprint_local as footprint
from behavior_interface_eval_test.tool.official_v2.map_navigation_local import plan_clearance_path
from behavior_interface_eval_test.tool.official_v2.navigation_footprint_local import (
    filter_navigation_depth_obstacles,
)


class NavigationDepthGuardTest(unittest.TestCase):
    @staticmethod
    def _thin_barrier_fixture(angle=0.0, *, wall_length=0.8, z_high=1.4,
                              crossed_history=True, connected_jamb=False):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[c, -s], [s, c]])

        ys = np.linspace(-wall_length / 2, wall_length / 2, 19)
        zs = np.linspace(0.2, z_high, 17)
        door = np.asarray(
            [[0.8 + 0.004 * math.sin(3 * y + z), y, z] for y in ys for z in zs]
        )
        anchor = np.asarray(
            [[0.8, y, z] for y in np.linspace(wall_length / 2 + 0.24,
                                               wall_length / 2 + 0.54, 8)
             for z in np.linspace(0.2, 1.4, 8)]
        )
        groups = [door, anchor]
        if connected_jamb:
            jamb_face = np.asarray([
                [x, wall_length / 2 + .50, z]
                for x in np.linspace(.82, 1.12, 6)
                for z in np.linspace(.2, 1.4, 7)
            ])
            jamb_bridge = np.asarray([
                [.82, y, z]
                for y in np.linspace(wall_length / 2 + .04,
                                     wall_length / 2 + .50, 8)
                for z in np.linspace(.2, 1.4, 7)
            ])
            groups.extend((jamb_face, jamb_bridge))
        points = np.concatenate(groups)
        points[:, :2] = points[:, :2] @ rotation.T

        path = np.asarray([
            [0.0, 0.0], [0.4, -0.55], [0.8, -0.9],
            [1.2, -0.55], [2.0, 0.0],
        ]) @ rotation.T
        history = np.asarray([
            [-0.6, -0.9], [0.0, -0.9], [0.6, -0.9], [1.2, -0.9],
            [1.6, -0.9],
        ]) @ rotation.T
        if not crossed_history:
            history = np.asarray([
                [-0.4, -0.9], [0.0, -0.9], [0.3, -0.9],
            ]) @ rotation.T
        snapshot = {
            "pose": {"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            "traversed_paths_xy_m": [history.tolist()],
        }
        reference = {
            "ok": True,
            "traversed_route_evidence_used": True,
            "path_xy_m": path.tolist(),
        }
        return snapshot, reference, points, path

    @staticmethod
    def _clear_ended_thin_barrier_fixture(angle=0.0, *, wall_length=.56):
        snapshot, reference, points, _path = (
            NavigationDepthGuardTest._thin_barrier_fixture(
                angle, wall_length=wall_length
            )
        )
        # Replace the fixture leaf with an exactly planar one, then add one
        # noisy vertical image column at its low end plus supported but distant
        # background context.  The two noisy returns are part of the leaf's XY
        # component; they are not a jamb and must not reverse the endpoint.
        rotation = np.asarray([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        half_length = .5 * wall_length
        points = np.asarray([
            [.8, y, z]
            for y in np.linspace(-half_length, half_length, 19)
            for z in np.linspace(.2, 1.4, 17)
        ])
        singleton = np.asarray([
            [.86, -half_length - .04, .45],
            [.86, -half_length - .04, 1.25],
        ])
        background = np.asarray([
            [x, y, z]
            for x in np.linspace(.55, .85, 7)
            for y in np.linspace(1.25, 1.45, 3)
            for z in (.3, 1.2)
        ])
        for group in (points, singleton, background):
            group[:, :2] = group[:, :2] @ rotation.T
        points = np.concatenate((points, singleton, background))

        reference["path_xy_m"] = (
            np.asarray([[0.0, 0.0], [1.6, 0.0]]) @ rotation.T
        ).tolist()
        reference["traversed_route_evidence_used"] = False
        taught_y = -half_length - .56
        taught = np.asarray([
            [0.0, 0.0], [.3, taught_y], [.8, taught_y],
            [1.3, taught_y], [1.6, 0.0],
        ]) @ rotation.T
        snapshot["traversed_paths_xy_m"] = [
            nav._subdivide(taught.tolist(), .05)
        ]
        grid = np.zeros((160, 160), dtype=np.int8)
        snapshot.update({
            "occupancy": grid,
            "obstacle_mask": np.zeros_like(grid, dtype=bool),
            "free_mask": np.ones_like(grid, dtype=bool),
            "wall_mask": np.zeros_like(grid, dtype=bool),
            "resolution": .05,
            "origin": [-4.0, -4.0],
        })
        executable_y = -half_length - .50
        executable = np.asarray([
            [0.0, 0.0], [.3, executable_y], [.8, executable_y],
            [1.3, executable_y], [1.6, 0.0],
        ]) @ rotation.T
        return snapshot, reference, points, executable.tolist()

    def test_partly_observed_half_metre_leaf_retains_extent_guard(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle, wall_length=.46):
                snapshot, reference, points, executable_path = (
                    self._clear_ended_thin_barrier_fixture(
                        angle, wall_length=.46
                    )
                )
                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )
                self.assertIsNotNone(detection)
                self.assertGreaterEqual(
                    detection["certificate"]["tangent_span_m"],
                    nav.THIN_BARRIER_MIN_TANGENT_SPAN_M,
                )
                bound = nav.bind_traversed_thin_barrier_segments(
                    detection["certificate"],
                    executable_path,
                    robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)
                tools._navigate_to_validate_traversed_thin_barrier(
                    bound,
                    executable_path,
                    robot_radius_m=.42,
                    failure_stage="test",
                )

            with self.subTest(angle=angle, wall_length=.43):
                snapshot, reference, points, _path = (
                    self._clear_ended_thin_barrier_fixture(
                        angle, wall_length=.43
                    )
                )
                self.assertIsNone(nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                ))

    def test_traversed_thin_door_is_detected_at_arbitrary_map_rotations(self):
        for angle in (0.0, 0.37, 1.19, -2.1):
            with self.subTest(angle=angle):
                snapshot, reference, points, _path = self._thin_barrier_fixture(angle)
                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=0.42,
                )
                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(certificate["mode"], "free_endpoint_swept_contact")
                self.assertGreaterEqual(
                    certificate["free_endpoint_neighbour_distance_m"], .55
                )
                self.assertLessEqual(
                    certificate["anchor_endpoint_neighbour_distance_m"], .36
                )

    def test_verified_stall_recovers_a_door_censored_by_the_image_bottom(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        points = points.copy()
        points[:, 2] += .64
        lower_boundary = points[:, 2] <= np.quantile(points[:, 2], .12)

        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot,
            reference,
            points,
            robot_radius_m=.42,
        ))
        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot,
            reference,
            points,
            robot_radius_m=.42,
            lower_image_boundary_mask=lower_boundary,
            allow_censored_low_extent=True,
        ))

        detection = nav.detect_traversed_thin_barrier(
            snapshot,
            reference,
            points,
            robot_radius_m=.42,
            lower_image_boundary_mask=lower_boundary,
            verified_stall_xy_m=[.48, 0.0],
            allow_censored_low_extent=True,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["low_extent_evidence"],
            "verified_stall_lower_image_boundary_censoring",
        )
        self.assertGreaterEqual(
            certificate["lower_image_boundary_point_count"],
            nav.THIN_BARRIER_CENSORED_MIN_BOUNDARY_POINTS,
        )
        preferred = reference["path_xy_m"]
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate,
            preferred,
            robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            preferred,
            robot_radius_m=.42,
            failure_stage="test",
        )

    def test_verified_stall_recovers_close_cropped_door_from_history_and_map(self):
        snapshot, reference, points, _path = (
            self._clear_ended_thin_barrier_fixture()
        )
        points = points.copy()
        points[:, 2] += .38
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [.3, -.78], [.8, -.78], [1.3, -.78], [1.6, 0.0]],
            .05,
        )]

        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot,
            reference,
            points,
            robot_radius_m=.42,
            allow_censored_low_extent=True,
        ))
        detection = nav.detect_traversed_thin_barrier(
            snapshot,
            reference,
            points,
            robot_radius_m=.42,
            verified_stall_xy_m=[.48, 0.0],
            allow_censored_low_extent=True,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["low_extent_evidence"],
            "verified_stall_history_mapped_free_censoring",
        )
        self.assertEqual(
            certificate["route_source"],
            "traversed_history_changed_leaf_pose",
        )
        self.assertEqual(
            certificate["endpoint_topology_mode"],
            "history_mapped_free_clear_ends",
        )
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate,
            detection["preferred_path_xy_m"],
            robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            detection["preferred_path_xy_m"],
            robot_radius_m=.42,
            failure_stage="test",
        )

        blocked_snapshot = dict(snapshot)
        blocked_snapshot["free_mask"] = np.zeros_like(
            snapshot["free_mask"], dtype=bool
        )
        blocked_snapshot["wall_mask"] = np.ones_like(
            snapshot["wall_mask"], dtype=bool
        )
        self.assertIsNone(nav.detect_traversed_thin_barrier(
            blocked_snapshot,
            reference,
            points,
            robot_radius_m=.42,
            verified_stall_xy_m=[.48, 0.0],
            allow_censored_low_extent=True,
        ))

    def test_verified_stall_extracts_close_door_from_connected_corner(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle):
                snapshot, reference, points, _path = (
                    self._clear_ended_thin_barrier_fixture(angle)
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                points = points.copy()
                points[:, 2] += .38
                attached_wall = np.asarray([
                    [x, .28, z]
                    for x in np.linspace(.80, 1.20, 11)
                    for z in np.linspace(.58, 1.98, 17)
                ])
                attached_wall[:, :2] = attached_wall[:, :2] @ rotation.T
                points = np.concatenate((points, attached_wall))
                taught = np.asarray([
                    [0.0, 0.0], [.3, -.78], [.8, -.78],
                    [1.3, -.78], [1.6, 0.0],
                ]) @ rotation.T
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(taught.tolist(), .05)
                ]
                stall = np.asarray([.48, 0.0]) @ rotation.T

                self.assertIsNone(nav.detect_traversed_thin_barrier(
                    snapshot,
                    reference,
                    points,
                    robot_radius_m=.42,
                    allow_censored_low_extent=True,
                ))
                diagnostics = []
                detection = nav.detect_traversed_thin_barrier(
                    snapshot,
                    reference,
                    points,
                    robot_radius_m=.42,
                    verified_stall_xy_m=stall,
                    allow_censored_low_extent=True,
                    diagnostics=diagnostics,
                )

                self.assertIsNotNone(detection, diagnostics)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["robust_fit_method"],
                    "bounded_xy_vertical_plane_consensus",
                )
                self.assertGreaterEqual(
                    certificate["robust_inlier_ratio"],
                    nav.THIN_BARRIER_CONNECTED_SUBPLANE_MIN_INLIER_RATIO,
                )
                self.assertEqual(
                    certificate["route_source"],
                    "traversed_history_changed_leaf_pose",
                )
                bound = nav.bind_traversed_thin_barrier_segments(
                    certificate,
                    detection["preferred_path_xy_m"],
                    robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)
                tools._navigate_to_validate_traversed_thin_barrier(
                    bound,
                    detection["preferred_path_xy_m"],
                    robot_radius_m=.42,
                    failure_stage="test",
                )

                blocked = dict(snapshot)
                blocked["free_mask"] = np.zeros_like(
                    snapshot["free_mask"], dtype=bool
                )
                blocked["wall_mask"] = np.ones_like(
                    snapshot["wall_mask"], dtype=bool
                )
                self.assertIsNone(nav.detect_traversed_thin_barrier(
                    blocked,
                    reference,
                    points,
                    robot_radius_m=.42,
                    verified_stall_xy_m=stall,
                    allow_censored_low_extent=True,
                ))

    def test_continuous_history_is_evidence_even_when_map_astar_did_not_need_it(self):
        snapshot, reference, points, path = self._thin_barrier_fixture()
        snapshot["traversed_paths_xy_m"] = [
            nav._subdivide(path.tolist(), .05)
        ]
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        self.assertEqual(
            detection["certificate"]["mode"],
            "free_endpoint_swept_contact",
        )

    def test_local_door_crossing_does_not_require_history_to_reach_final_goal(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle):
                snapshot, reference, points, _path = self._thin_barrier_fixture(
                    angle
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                reference["path_xy_m"] = (
                    np.asarray([
                        [0.0, 0.0],
                        [.4, -.55],
                        [.8, -.9],
                        [1.2, -.55],
                        [2.8, 1.2],
                    ])
                    @ rotation.T
                ).tolist()
                # This taught route certifies the intermediate doorway only.
                # The final marked goal is deliberately far beyond its end.
                history = (
                    np.asarray([
                        [-.6, -.9],
                        [0.0, -.9],
                        [.6, -.9],
                        [1.2, -.9],
                        [1.6, -.9],
                    ])
                    @ rotation.T
                )
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(history.tolist(), .05)
                ]
                reference["traversed_route_evidence_used"] = False

                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )

                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["mode"], "free_endpoint_swept_contact"
                )
                self.assertLessEqual(
                    certificate["history"]["crossing_error_m"], .20
                )

    def test_history_free_end_replaces_a_reference_route_through_leaf_centre(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [.25, -.65], [.8, -.9], [1.35, -.65], [1.6, 0.0]],
            .05,
        )]
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(certificate["mode"], "free_endpoint_swept_contact")
        self.assertEqual(certificate["route_source"], "traversed_history")
        self.assertGreaterEqual(
            certificate["route_plane_crossing_tangent_offset_m"], 0.0
        )
        self.assertEqual(len(detection["preferred_path_xy_m"]), 3)

    def test_history_near_current_leaf_is_topology_not_an_executable_line(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        # A previously successful centreline passes only 0.19 m beyond the
        # current free endpoint.  That proves connectivity, but a 0.42 m base
        # would overlap today's leaf if the line were replayed literally.
        taught_y = -.59
        taught = [[0.0, 0.0], [.35, taught_y], [.8, taught_y],
                  [1.25, taught_y], [1.6, 0.0]]
        # The current map-only route still goes through the panel centre; only
        # the old traversal reaches the nominal free end.
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(taught, .05)]
        grid = np.zeros((100, 100), dtype=np.int8)
        snapshot.update({
            "occupancy": grid,
            "obstacle_mask": np.zeros_like(grid, dtype=bool),
            "free_mask": np.ones_like(grid, dtype=bool),
            "wall_mask": np.zeros_like(grid, dtype=bool),
            "resolution": .05,
            "origin": [-2.0, -2.0],
        })
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["route_source"],
            "traversed_history_changed_leaf_pose",
        )
        self.assertEqual(
            certificate["endpoint_topology_mode"], "absolute_clearance"
        )
        self.assertAlmostEqual(
            certificate["route_contact_distance_m"],
            .42 + nav.THIN_BARRIER_ROUTE_CONTACT_PAD_M,
            places=7,
        )
        self.assertGreater(
            certificate["route_contact_distance_m"], abs(taught_y) - .4
        )

    def test_changed_door_pose_routes_current_plan_around_the_free_end(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        # The base crossed this doorway while the leaf was at another angle.
        # Relative to the leaf's current plane both the old route and the
        # map-only route pass through its middle, so replaying the old line is
        # wrong even though the historical traversal proves one passage.
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [1.6, 0.0]], .05,
        )]
        grid = np.zeros((100, 100), dtype=np.int8)
        snapshot.update({
            "occupancy": grid,
            "obstacle_mask": np.zeros_like(grid, dtype=bool),
            "free_mask": np.ones_like(grid, dtype=bool),
            "wall_mask": np.zeros_like(grid, dtype=bool),
            "resolution": .05,
            "origin": [-2.0, -2.0],
        })
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["route_source"],
            "traversed_history_changed_leaf_pose",
        )
        self.assertEqual(
            certificate["endpoint_topology_mode"], "absolute_clearance"
        )
        history_crossing = np.asarray(
            certificate["history_panel_crossing_xy_m"]
        )
        free_endpoint = np.asarray(certificate["free_endpoint_xy_m"])
        current_crossing = np.asarray(certificate["route_plane_crossing_xy_m"])
        tangent = np.asarray(certificate["tangent_xy"])
        self.assertLessEqual(
            abs(float((history_crossing - certificate["centroid_xy_m"]) @ tangent)),
            .5 * certificate["tangent_span_m"] + 1e-6,
        )
        self.assertAlmostEqual(
            abs(float((current_crossing - free_endpoint) @ tangent)),
            .42 + nav.THIN_BARRIER_ROUTE_CONTACT_PAD_M,
            places=7,
        )
        preferred = detection["preferred_path_xy_m"]
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate, preferred, robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            preferred,
            robot_radius_m=.42,
            failure_stage="test",
        )
        tampered = dict(bound)
        tampered["changed_pose_map_evidence"] = dict(
            bound["changed_pose_map_evidence"],
            qualified=False,
        )
        with self.assertRaisesRegex(
            tools._NavigateToFailure,
            "thin-barrier certificate is invalid",
        ):
            tools._navigate_to_validate_traversed_thin_barrier(
                tampered,
                preferred,
                robot_radius_m=.42,
                failure_stage="test",
            )

    def test_changed_door_pose_rejects_a_corner_cut_into_current_leaf(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [1.6, 0.0]], .05,
        )]
        grid = np.zeros((100, 100), dtype=np.int8)
        snapshot.update({
            "occupancy": grid,
            "obstacle_mask": np.zeros_like(grid, dtype=bool),
            "free_mask": np.ones_like(grid, dtype=bool),
            "wall_mask": np.zeros_like(grid, dtype=bool),
            "resolution": .05,
            "origin": [-2.0, -2.0],
        })
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        self.assertIsNotNone(detection)
        self.assertEqual(
            detection["certificate"]["route_source"],
            "traversed_history_changed_leaf_pose",
        )

        # The plane crossing is outside the free tip, but the incoming chord
        # cuts the corner and comes within the physical footprint radius.
        corner_cut = [
            [0.0, 0.0],
            [.8, -.92],
            [1.6, 0.0],
        ]
        self.assertIsNone(nav.bind_traversed_thin_barrier_segments(
            detection["certificate"],
            corner_cut,
            robot_radius_m=.42,
        ))
        self.assertIsNotNone(nav.bind_traversed_thin_barrier_segments(
            detection["certificate"],
            detection["preferred_path_xy_m"],
            robot_radius_m=.42,
        ))

    def test_changed_door_pose_accepts_a_prior_sweep_just_past_current_panel(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        # The leaf rotated after this traversal.  The old 0.4 m footprint
        # touched the current panel while its centreline passed 0.50 m beyond
        # the now-anchored end.  Current free-space topology identifies the
        # opposite endpoint, so recovery must construct a new detour there
        # instead of demanding that the old centreline intersect the moved
        # panel itself.
        reference["path_xy_m"] = [
            [0.0, 0.0], [.35, .90], [1.25, .90], [1.6, 0.0]
        ]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            reference["path_xy_m"], .05,
        )]
        grid = np.zeros((100, 100), dtype=np.int8)
        snapshot.update({
            "occupancy": grid,
            "obstacle_mask": np.zeros_like(grid, dtype=bool),
            "free_mask": np.ones_like(grid, dtype=bool),
            "wall_mask": np.zeros_like(grid, dtype=bool),
            "resolution": .05,
            "origin": [-2.0, -2.0],
        })
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["route_source"],
            "traversed_history_changed_leaf_pose",
        )
        self.assertGreater(
            certificate["history_panel_crossing_distance_m"], 0.0
        )
        self.assertLessEqual(
            certificate["history_panel_crossing_distance_m"], .52 + 1e-9
        )
        history_crossing = np.asarray(
            certificate["history_panel_crossing_xy_m"]
        )
        free_endpoint = np.asarray(certificate["free_endpoint_xy_m"])
        current_crossing = np.asarray(certificate["route_plane_crossing_xy_m"])
        tangent = np.asarray(certificate["tangent_xy"])
        self.assertGreater(
            abs(float((history_crossing - certificate["centroid_xy_m"]) @ tangent)),
            .5 * certificate["tangent_span_m"],
        )
        self.assertAlmostEqual(
            abs(float((current_crossing - free_endpoint) @ tangent)),
            .42 + nav.THIN_BARRIER_ROUTE_CONTACT_PAD_M,
            places=7,
        )
        preferred = detection["preferred_path_xy_m"]
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate, preferred, robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            preferred,
            robot_radius_m=.42,
            failure_stage="test",
        )
        tampered = dict(
            bound,
            history_panel_crossing_distance_m=(
                bound["history_panel_crossing_distance_m"] + .05
            ),
        )
        with self.assertRaisesRegex(
            tools._NavigateToFailure,
            "thin-barrier certificate is invalid",
        ):
            tools._navigate_to_validate_traversed_thin_barrier(
                tampered,
                preferred,
                robot_radius_m=.42,
                failure_stage="test",
            )

    def test_changed_door_swept_contact_is_rotation_invariant(self):
        for angle in (0.0, .61, -1.37, 2.42):
            with self.subTest(angle=angle):
                snapshot, reference, points, _path = self._thin_barrier_fixture(
                    angle,
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                route = np.asarray([
                    [0.0, 0.0], [.35, .90],
                    [1.25, .90], [1.6, 0.0],
                ]) @ rotation.T
                reference["path_xy_m"] = route.tolist()
                reference["traversed_route_evidence_used"] = False
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(route.tolist(), .05)
                ]
                grid = np.zeros((160, 160), dtype=np.int8)
                snapshot.update({
                    "occupancy": grid,
                    "obstacle_mask": np.zeros_like(grid, dtype=bool),
                    "free_mask": np.ones_like(grid, dtype=bool),
                    "wall_mask": np.zeros_like(grid, dtype=bool),
                    "resolution": .05,
                    "origin": [-4.0, -4.0],
                })

                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )

                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["route_source"],
                    "traversed_history_changed_leaf_pose",
                )
                self.assertGreater(
                    certificate["history_panel_crossing_distance_m"], 0.0
                )
                preferred = detection["preferred_path_xy_m"]
                bound = nav.bind_traversed_thin_barrier_segments(
                    certificate, preferred, robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)
                tools._navigate_to_validate_traversed_thin_barrier(
                    bound,
                    preferred,
                    robot_radius_m=.42,
                    failure_stage="test",
                )

    def test_latest_depth_frame_recovers_a_leaf_hidden_by_temporal_fusion(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        with mock.patch.object(
            tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=points,
        ):
            detection, trace, attempts, source = (
                tools._navigate_to_detect_traversed_thin_barrier(
                    None,
                    snapshot,
                    reference,
                    np.empty((0, 3)),
                    robot_radius_m=.42,
                )
            )

        self.assertIsNotNone(detection)
        self.assertEqual(source, "latest_depth_frame")
        self.assertEqual(
            attempts,
            [
                {
                    "observation_source": "fused_recent_depth",
                    "point_count": 0,
                    "detected": False,
                },
                {
                    "observation_source": "latest_depth_frame",
                    "point_count": len(points),
                    "detected": True,
                },
            ],
        )
        self.assertTrue(any(
            event["observation_source"] == "latest_depth_frame"
            and event.get("accepted") is True
            for event in trace
        ))

    def test_older_individual_frame_recovers_when_latest_misses_the_leaf(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()
        latest = np.asarray([[.3, .1, .4]], dtype=np.float64)
        with mock.patch.multiple(
            tools,
            _navigate_to_latest_depth_obstacles=mock.Mock(return_value=latest),
            _navigate_to_recent_depth_obstacle_samples=mock.Mock(
                return_value=[(12, latest), (7, points)]
            ),
        ):
            detection, trace, attempts, source = (
                tools._navigate_to_detect_traversed_thin_barrier(
                    None,
                    snapshot,
                    reference,
                    np.empty((0, 3)),
                    robot_radius_m=.42,
                )
            )

        self.assertIsNotNone(detection)
        self.assertEqual(source, "recent_depth_frame_sequence_7")
        self.assertEqual(
            [attempt["observation_source"] for attempt in attempts],
            [
                "fused_recent_depth",
                "latest_depth_frame",
                "recent_depth_frame_sequence_7",
            ],
        )
        self.assertTrue(any(
            event["observation_source"] == "recent_depth_frame_sequence_7"
            and event.get("accepted") is True
            for event in trace
        ))

    def test_connected_door_and_jamb_use_robust_leaf_and_history_route(self):
        for angle in (0.0, .61, -1.37):
            with self.subTest(angle=angle):
                snapshot, reference, points, _path = self._thin_barrier_fixture(
                    angle, connected_jamb=True,
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                reference["path_xy_m"] = (
                    np.asarray([[0.0, 0.0], [1.6, 0.0]]) @ rotation.T
                ).tolist()
                history = np.asarray([
                    [0.0, 0.0], [.25, -.65], [.8, -.9],
                    [1.35, -.65], [1.6, 0.0],
                ]) @ rotation.T
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(history.tolist(), .05)
                ]
                reference["traversed_route_evidence_used"] = False
                diagnostics = []

                detection = nav.detect_traversed_thin_barrier(
                    snapshot,
                    reference,
                    points,
                    robot_radius_m=.42,
                    diagnostics=diagnostics,
                )

                self.assertIsNotNone(detection, diagnostics)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["route_source"], "traversed_history"
                )
                self.assertLess(
                    certificate["component_point_count"],
                    certificate["raw_component_point_count"],
                )
                self.assertGreaterEqual(
                    certificate["robust_inlier_ratio"],
                    nav.THIN_BARRIER_ROBUST_MIN_INLIER_RATIO,
                )

    def test_history_identifies_a_free_end_despite_sparse_leaf_edge_returns(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(
            connected_jamb=True,
        )
        # These sparse returns lie beyond the robust leaf endpoint.  Treating
        # their nearest distance as an absolute free-space test used to make a
        # previously traversed ajar doorway look closed.
        leaf_edge_returns = np.asarray([
            [.8, -.62, z] for z in np.linspace(.2, 1.4, 9)
        ])
        points = np.concatenate((points, leaf_edge_returns))
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [.25, -.65], [.8, -.9], [1.35, -.65], [1.6, 0.0]],
            .05,
        )]
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(
            certificate["endpoint_topology_mode"],
            "history_asymmetric_context",
        )
        self.assertEqual(certificate["route_source"], "traversed_history")
        self.assertLess(
            certificate["free_endpoint_neighbour_distance_m"],
            nav.THIN_BARRIER_FREE_END_MIN_DISTANCE_M,
        )
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate,
            detection["preferred_path_xy_m"],
            robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            detection["preferred_path_xy_m"],
            robot_radius_m=.42,
            failure_stage="test",
        )

    def test_history_selects_clear_end_when_one_noisy_xy_column_looks_like_jamb(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle):
                snapshot, reference, points, executable = (
                    self._clear_ended_thin_barrier_fixture(angle)
                )

                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )

                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["endpoint_topology_mode"],
                    "history_mapped_free_clear_ends",
                )
                self.assertEqual(
                    certificate["route_source"], "traversed_history"
                )
                self.assertGreaterEqual(
                    certificate["free_endpoint_neighbour_distance_m"], .55
                )
                self.assertGreaterEqual(
                    certificate["anchor_endpoint_neighbour_distance_m"], .55
                )
                self.assertGreater(
                    certificate["route_contact_distance_m"],
                    .42 + nav.THIN_BARRIER_ROUTE_CONTACT_PAD_M,
                )
                self.assertLessEqual(
                    certificate["route_contact_distance_m"],
                    .42 + nav.THIN_BARRIER_HISTORY_ROUTE_CONTACT_PAD_M,
                )
                bound = nav.bind_traversed_thin_barrier_segments(
                    certificate, executable, robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)
                tools._navigate_to_validate_traversed_thin_barrier(
                    bound,
                    executable,
                    robot_radius_m=.42,
                    failure_stage="test",
                )

    def test_rotated_clear_ended_leaf_uses_current_route_and_history_connectivity(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle):
                snapshot, reference, points, _executable = (
                    self._clear_ended_thin_barrier_fixture(angle)
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                reference_path = np.asarray([
                    [0.0, 0.0], [.35, -.75], [.8, -.75],
                    [1.25, -.75], [1.6, 0.0],
                ]) @ rotation.T
                taught_path = np.asarray([
                    [0.0, 0.0], [.35, -.92], [.8, -.92],
                    [1.25, -.92], [1.6, 0.0],
                ]) @ rotation.T
                reference["path_xy_m"] = reference_path.tolist()
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(taught_path.tolist(), .05)
                ]

                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )

                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["route_source"],
                    "reference_plan_with_traversed_connectivity",
                )
                self.assertEqual(
                    certificate["endpoint_topology_mode"],
                    "history_mapped_free_clear_ends",
                )
                self.assertGreater(
                    certificate[
                        "history_connectivity_crossing_tangent_offset_m"
                    ],
                    .42 + nav.THIN_BARRIER_HISTORY_ROUTE_CONTACT_PAD_M,
                )
                bound = nav.bind_traversed_thin_barrier_segments(
                    certificate,
                    reference_path.tolist(),
                    robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)
                tools._navigate_to_validate_traversed_thin_barrier(
                    bound,
                    reference_path.tolist(),
                    robot_radius_m=.42,
                    failure_stage="test",
                )
                tampered = dict(
                    bound,
                    history_connectivity_crossing_tangent_offset_m=(
                        bound[
                            "history_connectivity_crossing_tangent_offset_m"
                        ] + .05
                    ),
                )
                with self.assertRaisesRegex(
                    tools._NavigateToFailure,
                    "thin-barrier certificate is invalid",
                ):
                    tools._navigate_to_validate_traversed_thin_barrier(
                        tampered,
                        reference_path.tolist(),
                        robot_radius_m=.42,
                        failure_stage="test",
                    )

    def test_current_free_end_route_wins_when_old_route_is_also_eligible(self):
        for angle in (0.0, .73, -1.41):
            with self.subTest(angle=angle):
                snapshot, reference, points, _executable = (
                    self._clear_ended_thin_barrier_fixture(angle)
                )
                rotation = np.asarray([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                reference_path = np.asarray([
                    [0.0, 0.0], [.35, -.75], [.8, -.75],
                    [1.25, -.75], [1.6, 0.0],
                ]) @ rotation.T
                # This older route is close enough to construct its own
                # free-end candidate.  It remains connectivity evidence, not
                # a better gate than the route certified on the current map.
                taught_path = np.asarray([
                    [0.0, 0.0], [.35, -.70], [.8, -.70],
                    [1.25, -.70], [1.6, 0.0],
                ]) @ rotation.T
                reference["path_xy_m"] = reference_path.tolist()
                snapshot["traversed_paths_xy_m"] = [
                    nav._subdivide(taught_path.tolist(), .05)
                ]

                detection = nav.detect_traversed_thin_barrier(
                    snapshot, reference, points, robot_radius_m=.42,
                )

                self.assertIsNotNone(detection)
                certificate = detection["certificate"]
                self.assertEqual(
                    certificate["route_source"],
                    "reference_plan_with_traversed_connectivity",
                )
                bound = nav.bind_traversed_thin_barrier_segments(
                    certificate, reference_path.tolist(), robot_radius_m=.42,
                )
                self.assertIsNotNone(bound)

    def test_clear_ended_history_cannot_erase_a_new_static_wall(self):
        snapshot, reference, points, _executable = (
            self._clear_ended_thin_barrier_fixture()
        )
        snapshot["free_mask"].fill(False)
        snapshot["wall_mask"].fill(True)

        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        ))

    def test_clear_ended_certificate_requires_supported_context_and_map_evidence(self):
        snapshot, reference, points, executable = (
            self._clear_ended_thin_barrier_fixture()
        )
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        bound = nav.bind_traversed_thin_barrier_segments(
            detection["certificate"], executable, robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)

        weak_context = dict(
            bound,
            anchor_endpoint_neighbour_support_cells=1,
        )
        weak_map = dict(bound)
        weak_map["changed_pose_map_evidence"] = dict(
            bound["changed_pose_map_evidence"], qualified=False,
        )
        for tampered in (weak_context, weak_map):
            with self.subTest(tampered=tampered["endpoint_topology_mode"]):
                with self.assertRaisesRegex(
                    tools._NavigateToFailure,
                    "thin-barrier certificate is invalid",
                ):
                    tools._navigate_to_validate_traversed_thin_barrier(
                        tampered,
                        executable,
                        robot_radius_m=.42,
                        failure_stage="test",
                    )

    def test_history_does_not_override_symmetric_close_endpoint_context(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(
            connected_jamb=True,
        )
        close_context = np.asarray([
            [.8, -.54, z] for z in np.linspace(.2, 1.4, 9)
        ])
        points = np.concatenate((points, close_context))
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [.25, -.65], [.8, -.9], [1.35, -.65], [1.6, 0.0]],
            .05,
        )]
        reference["traversed_route_evidence_used"] = False

        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        ))

    def test_extended_width_plane_requires_the_free_end_proof(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(
            wall_length=1.85,
        )
        reference["path_xy_m"] = [[0.0, 0.0], [1.6, 0.0]]
        snapshot["traversed_paths_xy_m"] = [nav._subdivide(
            [[0.0, 0.0], [.25, -.75], [.8, -1.1], [1.35, -.75], [1.6, 0.0]],
            .05,
        )]
        reference["traversed_route_evidence_used"] = False

        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )

        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertGreater(
            certificate["tangent_span_m"],
            nav.THIN_BARRIER_MAX_TANGENT_SPAN_M,
        )
        self.assertLessEqual(
            certificate["tangent_span_m"],
            nav.THIN_BARRIER_MAX_FREE_END_TANGENT_SPAN_M,
        )
        self.assertEqual(certificate["mode"], "free_endpoint_swept_contact")
        self.assertEqual(certificate["route_source"], "traversed_history")

        self.assertIsNone(nav.bind_traversed_thin_barrier_segments(
            certificate,
            reference["path_xy_m"],
            robot_radius_m=.42,
        ))
        executable_path = detection["preferred_path_xy_m"]
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate,
            executable_path,
            robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        tools._navigate_to_validate_traversed_thin_barrier(
            bound,
            executable_path,
            robot_radius_m=.42,
            failure_stage="test",
        )

    def test_direct_crossing_requires_the_observed_leaf_endpoint(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture()

        def set_straight_crossing(y):
            reference["path_xy_m"] = [[0.0, y], [1.6, y]]
            snapshot["traversed_paths_xy_m"] = [nav._subdivide(
                [[-0.6, y], [2.0, y]], .05
            )]

        set_straight_crossing(0.0)
        self.assertIsNone(nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        ), "a historical path must not authorize crossing a closed leaf centre")

        set_straight_crossing(-.38)
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        self.assertIsNotNone(detection)
        certificate = detection["certificate"]
        self.assertEqual(certificate["mode"], "direct_plane_crossing")
        self.assertLessEqual(
            certificate["route_endpoint_distance_m"],
            nav.THIN_BARRIER_DIRECT_MAX_ENDPOINT_DISTANCE_M,
        )
        bound = nav.bind_traversed_thin_barrier_segments(
            certificate, reference["path_xy_m"], robot_radius_m=.42,
        )
        self.assertIsNotNone(bound)
        self.assertEqual(
            tools._navigate_to_validate_traversed_thin_barrier(
                bound,
                reference["path_xy_m"],
                robot_radius_m=.42,
                failure_stage="test",
            )["segment_indices"],
            bound["segment_indices"],
        )
        tampered = dict(bound, route_endpoint_distance_m=.25)
        with self.assertRaisesRegex(
            tools._NavigateToFailure,
            "thin-barrier certificate is invalid",
        ):
            tools._navigate_to_validate_traversed_thin_barrier(
                tampered,
                reference["path_xy_m"],
                robot_radius_m=.42,
                failure_stage="test",
            )

    def test_thin_barrier_rejects_long_wall_low_object_and_missing_crossing(self):
        cases = (
            {"wall_length": 2.2},
            {"z_high": 0.55},
            {"crossed_history": False},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                snapshot, reference, points, _path = self._thin_barrier_fixture(
                    **kwargs
                )
                self.assertIsNone(
                    nav.detect_traversed_thin_barrier(
                        snapshot, reference, points, robot_radius_m=.42,
                    )
                )

    def test_thin_barrier_override_is_bounded_and_does_not_mutate_map(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(.37)
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        self.assertIsNotNone(detection)
        grid = np.zeros((100, 100), dtype=np.int8)
        obstacle = np.zeros_like(grid, dtype=bool)
        free = np.ones_like(grid, dtype=bool)
        geometry = dict(
            snapshot,
            occupancy=grid,
            obstacle_mask=obstacle,
            free_mask=free,
            resolution=.05,
            origin=[-2.0, -2.0],
            navigation_depth_points_xy_m=points[:, :2],
            navigation_obstacle_mask=obstacle.copy(),
        )
        original_obstacle = geometry["obstacle_mask"].copy()
        original_depth = geometry["navigation_depth_points_xy_m"].copy()
        recovered = nav.apply_traversed_thin_barrier_override(geometry, detection)
        np.testing.assert_array_equal(geometry["obstacle_mask"], original_obstacle)
        np.testing.assert_array_equal(
            geometry["navigation_depth_points_xy_m"], original_depth
        )
        self.assertGreater(
            recovered["traversed_thin_barrier_removed_depth_points"], 0
        )
        certificate = detection["certificate"]
        override_span = (
            certificate["override_tangent_max_m"]
            - certificate["override_tangent_min_m"]
        )
        self.assertLess(
            certificate["override_tangent_max_m"],
            certificate["tangent_span_m"],
            "the anchored remainder of the leaf must stay occupied",
        )
        self.assertAlmostEqual(
            certificate["override_tangent_min_m"],
            -nav.THIN_BARRIER_FREE_END_TAIL_PAD_M,
            places=7,
        )
        self.assertLessEqual(
            override_span,
            .42
            + nav.THIN_BARRIER_FREE_END_OVERRIDE_CLEARANCE_PAD_M
            + nav.THIN_BARRIER_FREE_END_TAIL_PAD_M
            + 1e-9,
        )
        self.assertGreater(
            len(recovered["navigation_depth_points_xy_m"]), 0,
            "the anchored jamb must remain in the private depth layer",
        )
        self.assertGreater(
            recovered[
                "traversed_thin_barrier_observed_removed_depth_points"
            ],
            0,
        )
        preferred_cost = nav._preferred_path_cost(
            recovered,
            obstacle.shape,
            (-2.0, -2.0),
            .05,
            robot_radius_m=.42,
        )
        self.assertIsNotNone(preferred_cost)
        free_endpoint = np.asarray(certificate["crossing_xy_m"])
        normal = np.asarray(certificate["normal_xy"])
        tangent = np.asarray(certificate["tangent_xy"])
        inward = free_endpoint + .25 * tangent
        outward = free_endpoint - .25 * tangent
        self.assertTrue(np.isinf(preferred_cost[nav._cell(
            tuple(inward), (-2.0, -2.0), .05
        )]))
        self.assertTrue(np.isfinite(preferred_cost[nav._cell(
            tuple(outward), (-2.0, -2.0), .05
        )]))
        preferred = np.asarray(detection["preferred_path_xy_m"])
        gate = preferred[np.argmin(np.abs(
            (preferred - free_endpoint) @ normal
        ))]
        self.assertTrue(np.isfinite(preferred_cost[nav._cell(
            tuple(gate), (-2.0, -2.0), .05
        )]))
        # A route may not evade the certified free end by leaving the soft
        # preference neighbourhood and crossing the same plane elsewhere.
        far_plane_crossing = gate - 1.0 * tangent
        self.assertTrue(np.isinf(preferred_cost[nav._cell(
            tuple(far_plane_crossing), (-2.0, -2.0), .05
        )]))

    def test_thin_barrier_recovery_injects_observed_anchored_remainder(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(.37)
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        self.assertIsNotNone(detection)
        grid = np.zeros((100, 100), dtype=np.int8)
        obstacle = np.zeros_like(grid, dtype=bool)
        # Model the production range split: the full vertical leaf was close
        # enough for recognition, but no leaf return survived in the shorter
        # low-layer planning horizon.
        geometry = dict(
            snapshot,
            occupancy=grid,
            obstacle_mask=obstacle,
            free_mask=np.ones_like(grid, dtype=bool),
            resolution=.05,
            origin=[-2.0, -2.0],
            navigation_depth_points_xy_m=np.asarray(
                [[-1.5, -1.5], [-1.45, -1.5], [-1.5, -1.45], [-1.45, -1.45]],
                dtype=np.float64,
            ),
            navigation_obstacle_mask=obstacle.copy(),
        )

        recovered = nav.apply_traversed_thin_barrier_override(
            geometry, detection
        )

        self.assertEqual(
            recovered["traversed_thin_barrier_removed_depth_points"], 0
        )
        self.assertGreater(
            recovered[
                "traversed_thin_barrier_observed_removed_depth_points"
            ],
            0,
        )
        self.assertGreater(
            recovered["traversed_thin_barrier_injected_depth_points"], 0
        )
        certificate = detection["certificate"]
        crossing = np.asarray(certificate["crossing_xy_m"])
        tangent = np.asarray(certificate["tangent_xy"])
        normal = np.asarray(certificate["normal_xy"])
        retained = np.asarray(recovered["navigation_depth_points_xy_m"])
        relative = retained - crossing
        anchored = (
            (np.abs(relative @ normal) <= .10)
            & (
                relative @ tangent
                > certificate["override_tangent_max_m"] + .05
            )
        )
        self.assertTrue(
            anchored.any(),
            "the observed anchored half must remain a live-depth obstacle",
        )

    def test_thin_barrier_noncontact_retry_restores_all_observations(self):
        snapshot, reference, points, _path = self._thin_barrier_fixture(.37)
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        self.assertIsNotNone(detection)
        grid = np.zeros((100, 100), dtype=np.int8)
        obstacle = np.zeros_like(grid, dtype=bool)
        obstacle[5, 5] = True
        free = ~obstacle
        ordinary_points = np.asarray(
            [
                [-1.5, -1.5],
                [-1.45, -1.5],
                [-1.5, -1.45],
                [-1.45, -1.45],
            ],
            dtype=np.float64,
        )
        ordinary_mask = np.zeros_like(grid, dtype=bool)
        geometry = dict(
            snapshot,
            occupancy=grid,
            obstacle_mask=obstacle,
            free_mask=free,
            resolution=.05,
            origin=[-2.0, -2.0],
            navigation_depth_points_xy_m=ordinary_points,
            navigation_obstacle_mask=ordinary_mask,
        )

        restored = nav.restore_traversed_thin_barrier_observation(
            geometry, detection
        )

        np.testing.assert_array_equal(restored["obstacle_mask"], obstacle)
        np.testing.assert_array_equal(restored["free_mask"], free)
        self.assertEqual(
            len(restored["navigation_depth_points_xy_m"]),
            len(ordinary_points)
            + len(detection["raw_component_points_xy_m"]),
        )
        self.assertIn("navigation_preferred_path_context", restored)
        observed = np.asarray(detection["raw_component_points_xy_m"])
        cells = np.floor((observed - [-2.0, -2.0]) / .05).astype(int)
        valid = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < grid.shape[1])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < grid.shape[0])
        )
        self.assertTrue(np.all(
            restored["navigation_obstacle_mask"][
                cells[valid, 1], cells[valid, 0]
            ]
        ))

    def test_runtime_door_exception_is_signed_to_forward_contact_segments(self):
        snapshot, reference, points, path = self._thin_barrier_fixture()
        detection = nav.detect_traversed_thin_barrier(
            snapshot, reference, points, robot_radius_m=.42,
        )
        certificate = nav.bind_traversed_thin_barrier_segments(
            detection["certificate"], path, robot_radius_m=.42,
        )
        self.assertIsNotNone(certificate)
        segment_index = certificate["segment_indices"][0]
        trajectory = {
            "path_xy_m": path.tolist(),
            "traversed_thin_barrier": certificate,
        }
        pose_xy = np.asarray([.15, 0.0])
        runtime_snapshot = {
            "pose": {"x": pose_xy[0], "y": pose_xy[1], "yaw_rad": 0.0}
        }
        door_world = np.asarray([.8, -.35, .3])
        jamb_world = np.asarray([.8, .7, .3])
        observed = np.stack((door_world, jamb_world))
        observed[:, :2] -= pose_xy
        direction = path[segment_index + 1] - path[segment_index]
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]][:2] = direction / np.linalg.norm(direction)
        mask = tools._navigate_to_traversed_barrier_point_mask(
            runtime_snapshot,
            observed,
            action,
            trajectory,
            segment_index + 1,
        )
        np.testing.assert_array_equal(mask, [True, False])

        reverse = action.copy()
        reverse[ACTION_SLICES["base"]][:2] *= -1
        self.assertFalse(tools._navigate_to_traversed_barrier_point_mask(
            runtime_snapshot, observed, reverse, trajectory, segment_index + 1,
        ).any())
        spin = action.copy()
        spin[ACTION_SLICES["base"]][2] = .2
        self.assertFalse(tools._navigate_to_traversed_barrier_point_mask(
            runtime_snapshot, observed, spin, trajectory, segment_index + 1,
        ).any())
        self.assertFalse(tools._navigate_to_traversed_barrier_point_mask(
            runtime_snapshot, observed, action, trajectory, len(path) - 1,
        ).any())

    def test_only_observed_free_space_clears_historical_returns(self):
        depth = np.full((25, 25), 2.0)
        camera = dict(fx=10, fy=10, cx=12, cy=12, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        points = np.array([[0, 0, -1], [0, 0, -2], [0, 0, -3],
                           [0, 0, 1], [10, 0, -1], [0, 0, -1.95]])
        np.testing.assert_array_equal(
            tools._navigate_to_depth_proves_free(points, depth, camera),
            [True, False, False, False, False, False],
        )
        for invalid_or_occluding in (np.nan, np.inf, 0, .5, 1.05):
            with self.subTest(pixel=invalid_or_occluding):
                depth[11, 13] = invalid_or_occluding
                self.assertFalse(tools._navigate_to_depth_proves_free(points[:1], depth, camera)[0])

    def test_depth_clearing_uses_current_camera_extrinsics(self):
        camera = dict(fx=10, fy=10, cx=12, cy=12, pos=[.2, -.3, 1.2],
                      quat=[0, -math.sqrt(.5), 0, math.sqrt(.5)])
        depth = np.full((25, 25), 2.0)
        points = tools._depth_points(np.ones_like(depth), camera=camera, stride=12)
        visible = tools._navigate_to_depth_proves_free(points, depth, camera)
        np.testing.assert_array_equal(visible, [False, False, False, False, True, False, False, False, False])

    def test_new_ray_clears_cache_in_policy_frame_without_erasing_occluded_history(self):
        state = {"sequence": 5}
        depth = np.full((25, 25), 2.0)
        camera = dict(fx=10, fy=10, cx=12, cy=12, pos=[0, 0, 1],
                      quat=[0, -math.sqrt(.5), 0, math.sqrt(.5)])
        adapter = SimpleNamespace(camera_depth_frame=lambda _name: depth,
                                  camera_relative_poses=lambda: {"head": camera},
                                  status=lambda: state)
        ctx = SimpleNamespace(world=SimpleNamespace(_official_adapter=adapter))
        old = np.array([[1, 0, 1], [3, 0, 1]])
        # Policy origin is translated by (4, -2); points remain in that frame.
        old[:, :2] += [4, -2]
        guard = dict(owner=id(ctx), map_epoch="test", samples=[(0, old)],
                     last_sequence=0, checks=0, anchor_kind="policy_odometry",
                     merged_sequences=(0,), merged_points=old)
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, guard)
        with mock.patch.multiple(tools,
                _navigate_to_policy_local_pose=mock.Mock(return_value=(4, -2, 0)),
                _navigate_to_snapshot_sequence=mock.Mock(return_value=5),
                _camera_intrinsics=mock.Mock(return_value=camera),
                _navigate_to_observed_joints=mock.Mock(return_value={}),
                filter_navigation_depth_obstacles=mock.Mock(return_value=np.empty((0, 3)))):
            remaining = tools._navigate_to_depth_obstacles(ctx, {"map_epoch": "test"})
        np.testing.assert_allclose(remaining, [[3, 0, 1]])
        self.assertEqual(guard["cleared_points"], 1)

    def test_latest_depth_sample_is_not_contaminated_by_older_cached_points(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        older = np.asarray([[5.0, -2.0, .3]])
        latest = np.asarray([[6.0, -2.0, .4]])
        guard = {
            "owner": id(ctx),
            "map_epoch": "test",
            "anchor_kind": "policy_odometry",
            "samples": [(4, older), (5, latest)],
        }
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, guard)
        with mock.patch.multiple(
            tools,
            _navigate_to_policy_local_pose=mock.Mock(
                return_value=(4.0, -2.0, 0.0)
            ),
            _navigate_to_snapshot_sequence=mock.Mock(return_value=5),
        ):
            result = tools._navigate_to_latest_depth_obstacles(
                ctx, {"map_epoch": "test"}
            )
        np.testing.assert_allclose(result, [[2.0, 0.0, .4]])

    def test_recent_depth_samples_are_bounded_ordered_and_pose_transformed(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        samples = [
            (sequence, np.asarray([[4.0, -2.0 + sequence, .3]]))
            for sequence in range(1, 10)
        ]
        guard = {
            "owner": id(ctx),
            "map_epoch": "test",
            "anchor_kind": "policy_odometry",
            "samples": samples,
        }
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, guard)
        with mock.patch.multiple(
            tools,
            _navigate_to_policy_local_pose=mock.Mock(
                return_value=(4.0, -2.0, math.pi / 2.0)
            ),
            _navigate_to_snapshot_sequence=mock.Mock(return_value=10),
        ):
            result = tools._navigate_to_recent_depth_obstacle_samples(
                ctx, {"map_epoch": "test"}
            )

        self.assertEqual(
            [sequence for sequence, _points in result],
            [9, 8, 7, 6, 5, 4],
        )
        np.testing.assert_allclose(result[0][1], [[9.0, 0.0, .3]], atol=1e-12)

    def test_floor_and_high_ceiling_are_not_obstacles_and_j8_is_not_a_gate(self):
        q = {"trunk": np.zeros(4), "arm_left": np.zeros(8),
             "arm_right": np.zeros(8), "gripper_left": np.zeros(2),
             "gripper_right": np.zeros(2)}
        q["arm_left"][7] = 0.2
        q["arm_right"][7] = -0.3
        points = np.array([[2.0, 0.0, 0.01], [2.0, 0.0, 0.5],
                           [2.0, 0.0, 4.0], [0.0, 0.0, 0.2]])
        remaining = filter_navigation_depth_obstacles(points, q, arm_dof=8)
        np.testing.assert_allclose(remaining, [[2.0, 0.0, 0.5]])

    def test_guard_checks_stopping_sweep_not_only_current_footprint(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [1.0, 0.0, 0.0]
        for point, expected in (([0.7, 0.0], False), ([0.95, 0.0], True),
                                ([0.2, 0.49], False), ([0.2, 0.55], True),
                                ([-0.7, 0.0], True)):
            with self.subTest(point=point):
                with mock.patch.object(tools, "_navigate_to_depth_obstacles",
                                       return_value=np.asarray([[*point, 0.25]])):
                    with mock.patch.object(tools, "_navigate_to_observed_joints", return_value={}):
                        with mock.patch.object(tools, "navigation_upper_body_sweep_is_clear", return_value=True):
                            self.assertEqual(tools._navigate_to_depth_motion_clear(ctx, {}, action), expected)

    @staticmethod
    def _committed_history_trajectory(*, stall_points=0, stall_evidence=0):
        return {
            "planner": {
                "traversed_route_direct_replay": True,
                "committed_history_runtime_guard_selected": True,
                "execution_stall_depth_point_count": stall_points,
                "execution_stall_obstacle_evidence_count": stall_evidence,
                "route_sweep": {
                    "occupied_overlap_cells": 0,
                    "unknown_overlap_cells": 0,
                    "live_depth_overlap_cells": 0,
                    "execution_stall_overlap_cells": 0,
                },
            },
            "clearance": {
                "minimum_execution_clearance_m": .568,
                "required_clearance_m": .42,
            },
        }

    def test_committed_history_allows_only_one_sided_future_corner_contact(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [.4667, -.0043, 0.0]
        side_jamb = np.asarray([[.3726, .2989, .4366]])
        with mock.patch.object(
            tools, "_navigate_to_depth_obstacles", return_value=side_jamb
        ):
            self.assertTrue(tools._navigate_to_depth_motion_clear(
                ctx,
                {},
                action,
                self._committed_history_trajectory(),
                1,
            ))
        guard = getattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR)
        self.assertEqual(guard["history_edge_exempted_points"], 1)

    def test_committed_history_accepts_live_door_jamb_at_hysteresis_boundary(self):
        """A taught route must not double-charge the RGB-D Schmitt band."""

        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [
            .6666666865348816,
            .07913443446159363,
            0.0,
        ]
        # Captured from the repeated garage-egress stop.  It is a one-sided
        # future corner with 16.01 mm nominal deficit, 11.01 mm after the
        # 5 mm Schmitt band, and 160.6 mm current-footprint clearance.
        side_jamb = np.asarray([
            [.42819934562709694, -.3093765560169907, .4371481645007611]
        ])
        with mock.patch.object(
            tools, "_navigate_to_depth_obstacles", return_value=side_jamb
        ):
            self.assertTrue(tools._navigate_to_depth_motion_clear(
                ctx,
                {},
                action,
                self._committed_history_trajectory(),
                1,
            ))
        guard = getattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR)
        self.assertEqual(guard["history_edge_exempted_points"], 1)

    def test_history_edge_allowance_never_applies_to_centre_or_two_sides(self):
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [.4667, -.0043, 0.0]
        trajectory = self._committed_history_trajectory()
        for points in (
            [[.3726, 0.0, .4366]],
            [[.3726, .2989, .4366], [.3726, -.2989, .4366]],
        ):
            with self.subTest(points=points):
                ctx = SimpleNamespace(world=SimpleNamespace())
                setattr(
                    ctx.world,
                    tools.NAVIGATE_TO_DEPTH_GUARD_ATTR,
                    {"stops": 0},
                )
                with mock.patch.object(
                    tools,
                    "_navigate_to_depth_obstacles",
                    return_value=np.asarray(points),
                ):
                    self.assertFalse(tools._navigate_to_depth_motion_clear(
                        ctx, {}, action, trajectory, 1
                    ))

    def test_history_edge_allowance_rejects_a_cross_route_depth_slab(self):
        polygon = footprint.base_navigation_polygon()
        lateral_extent = float(np.max(np.abs(polygon[:, 1])))
        points = np.asarray(
            [
                [0.37, 0.80 * lateral_extent],
                [0.37, 0.35 * lateral_extent],
                [0.37, -0.35 * lateral_extent],
                [0.37, -0.80 * lateral_extent],
            ],
            dtype=np.float64,
        )
        mask = tools._navigate_to_direct_history_edge_point_mask(
            points,
            polygon,
            np.asarray([0.01, 0.04, 0.04, 0.04]),
            np.asarray([0.08, 0.08, 0.08, 0.08]),
            np.full(4, 0.02),
            np.asarray([0.15, 0.0]),
            self._committed_history_trajectory(),
        )

        self.assertFalse(np.any(mask))

    def test_history_edge_allowance_is_disabled_after_verified_stall(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [.4667, -.0043, 0.0]
        with mock.patch.object(
            tools,
            "_navigate_to_depth_obstacles",
            return_value=np.asarray([[.3726, .2989, .4366]]),
        ):
            self.assertFalse(tools._navigate_to_depth_motion_clear(
                ctx,
                {},
                action,
                self._committed_history_trajectory(stall_points=8),
                1,
            ))

    def test_low_depth_guard_allows_a_proven_separating_retreat(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [-1.0, 0.0, 0.0]
        with mock.patch.object(
            tools, "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.47, 0.0, 0.25]]),
        ):
            with mock.patch.object(
                tools, "_navigate_to_observed_joints", return_value={}
            ):
                with mock.patch.object(
                    tools,
                    "navigation_upper_body_sweep_is_clear",
                    return_value=True,
                ):
                    self.assertTrue(
                        tools._navigate_to_depth_motion_clear(ctx, {}, action)
                    )

        action[ACTION_SLICES["base"]] = [1.0, 0.0, 0.0]
        with mock.patch.object(
            tools, "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.47, 0.0, 0.25]]),
        ):
            with mock.patch.object(
                tools, "_navigate_to_observed_joints", return_value={}
            ):
                self.assertFalse(
                    tools._navigate_to_depth_motion_clear(ctx, {}, action)
                )

    def test_low_depth_guard_keeps_uncertain_overlap_blocked_when_retreating(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [-1.0, 0.0, 0.0]
        with mock.patch.object(
            tools, "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.45, 0.0, 0.25]]),
        ):
            self.assertFalse(
                tools._navigate_to_depth_motion_clear(ctx, {}, action)
            )

    def test_zero_action_does_not_require_depth_or_prevent_braking(self):
        self.assertTrue(tools._navigate_to_depth_motion_clear(None, {}, np.zeros(ACTION_DIM)))

    def test_live_layer_does_not_write_back_into_the_map(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        grid.setflags(write=False)
        snapshot = {"occupancy": grid, "resolution": 0.1, "origin": [0.0, 0.0],
                    "pose": {"x": 2.0, "y": 3.0, "yaw_deg": 90.0}}
        with mock.patch.object(tools, "_navigate_to_depth_obstacles",
                               return_value=np.array([[1.0, 0.0, 0.25], [0.0, 0.0, 1.2]])):
            planned = tools._navigate_to_depth_planning_snapshot(None, snapshot)
        self.assertNotIn("navigation_obstacle_mask", snapshot)
        self.assertIs(planned["occupancy"], grid)
        self.assertEqual(np.count_nonzero(grid), 0)
        self.assertTrue(planned["navigation_obstacle_mask"][40, 20])
        self.assertEqual(np.count_nonzero(planned["navigation_obstacle_mask"]), 1)

    def test_local_planning_horizon_does_not_disable_far_stopping_guard(self):
        grid = np.zeros((100, 100), dtype=np.int8)
        snapshot = {"occupancy": grid, "resolution": 0.1, "origin": [0.0, 0.0],
                    "pose": {"x": 4.0, "y": 4.0, "yaw_deg": 0.0}}
        points = np.array([[1.0, 0.0, 0.25], [2.0, 0.0, 0.25], [0.4, 0.0, 1.2]])
        with mock.patch.object(tools, "_navigate_to_depth_obstacles", return_value=points):
            planned = tools._navigate_to_depth_planning_snapshot(None, snapshot)
        self.assertEqual(int(planned["navigation_obstacle_mask"].sum()), 1)
        self.assertTrue(planned["navigation_obstacle_mask"][40, 50])
        self.assertFalse(planned["navigation_obstacle_mask"][40, 60])
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [1.0, 0.0, 0.0]
        with mock.patch.object(tools, "_navigate_to_depth_obstacles",
                               return_value=np.array([[0.7, 0.0, 0.25]])):
            self.assertFalse(tools._navigate_to_depth_motion_clear(ctx, snapshot, action))

    def test_depth_height_follows_static_chassis_geometry_with_margin(self):
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, tools.NAVIGATE_TO_DEPTH_GUARD_ATTR, {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [1.0, 0.0, 0.0]
        grid = np.zeros((80, 80), dtype=np.int8)
        snapshot = {"occupancy": grid, "resolution": 0.1, "origin": [0.0, 0.0],
                    "pose": {"x": 2.0, "y": 2.0, "yaw_deg": 0.0}}
        top = footprint.base_navigation_vertical_bounds_m()[1]
        for height, blocked in ((0.25, True), (top, True), (top + 0.049, True),
                                (top + 0.051, False), (0.70, False), (1.2, False)):
            with self.subTest(height=height):
                with mock.patch.object(tools, "_navigate_to_depth_obstacles",
                                       return_value=np.array([[0.7, 0.0, height]])):
                    planned = tools._navigate_to_depth_planning_snapshot(ctx, snapshot)
                    self.assertEqual(bool(np.any(planned.get("navigation_obstacle_mask", False))), blocked)
                    self.assertEqual(tools._navigate_to_depth_motion_clear(ctx, snapshot, action), not blocked)

    def test_clear_first_chord_cannot_authorize_a_depth_blocked_later_chord(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[:, 24] = True
        snapshot = {"occupancy": grid, "resolution": 0.1, "origin": [0.0, 0.0],
                    "pose": {"x": 1.0, "y": 1.0, "yaw_deg": 0.0},
                    "navigation_obstacle_mask": mask}
        failed = {"ok": False, "error": "unreachable"}
        diagnostic = {"ok": True, "path_xy_m": [[1.0, 1.0], [1.1, 1.0], [3.0, 1.0]]}
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.multiple(
                tools,
                _check_cancelled=mock.Mock(),
                _navigate_to_capture_inactive_limb_contract=mock.Mock(return_value=({}, None)),
                _navigate_to_base_footprint=mock.Mock(return_value={"radius_m": 0.42}),
                _navigate_to_depth_planning_snapshot=mock.Mock(return_value=snapshot),
                _navigate_to_depth_obstacles=mock.Mock(
                    return_value=np.empty((0, 3))
                ),
                _next_plan_id=mock.Mock(return_value="plan_test"),
                _plan_record_path=mock.Mock(return_value=root + "/plan_test.json"),
            ):
                with mock.patch.object(tools, "plan_clearance_path",
                                       side_effect=[failed, diagnostic]):
                    with self.assertRaises(tools._NavigateToFailure) as caught:
                        tools._navigate_to_new_plan(
                            None, args={"name": "goal", "arrival_tolerance_m": 0.25},
                            storage_session_id="test", snapshot=snapshot,
                            deadline=time.monotonic() + 10.0,
                        )
        self.assertEqual(caught.exception.details["planning_failure_code"],
                         "live_depth_blocks_map_route")
        self.assertTrue(caught.exception.details["map_only_route_available"])

    def test_successful_depth_detour_selects_a_shorter_certified_door_route(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[20, 20] = True
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": mask,
        }
        primary = {
            "ok": True,
            "path_length_m": 4.2,
            "direct_distance_m": 2.0,
        }
        map_only = {
            "ok": True,
            "path_length_m": 2.1,
            "direct_distance_m": 2.0,
        }
        recovered = {
            "ok": True,
            "path_length_m": 2.4,
            "direct_distance_m": 2.0,
            "traversed_thin_barrier": {"certificate": "test"},
        }
        compiled = {"schema": "test"}
        loaded = ({"ok": True}, "/tmp/plan_test.json")
        recovery = mock.Mock(return_value=(
            recovered,
            {"detected": True, "planned": True, "bound": True},
        ))
        compile_plan = mock.Mock(return_value=compiled)
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=recovery,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(return_value=loaded),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                result = tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                )

        self.assertEqual(result, loaded)
        recovery.assert_called_once()
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 2.4)
        self.assertTrue(selected["depth_detour_evaluated"])
        self.assertTrue(selected["depth_detour_candidate"])
        self.assertTrue(selected["depth_detour_recovery_selected"])
        self.assertAlmostEqual(selected["depth_detour_recovered_saving_m"], 1.8)

    def test_missing_wall_shortcut_runs_oriented_history_refinement(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[20, 20] = True
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": mask,
        }
        primary = {
            "ok": True,
            "path_length_m": 9.5,
            "direct_distance_m": 8.0,
            "ordinary_route_unsupported_by_history_m": 1.7,
            "ordinary_route_unsupported_by_history_fraction": 0.18,
        }
        map_only = {
            "ok": True,
            "path_length_m": 9.7,
            "direct_distance_m": 8.0,
            "traversed_route_direct_replay": True,
        }
        refined = {
            **map_only,
            "path_length_m": 9.8,
            "traversed_route_direct_replay": False,
            "depth_refinement": {"schema": nav.DEPTH_REFINEMENT_SCHEMA},
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        exact_refinement = mock.Mock(return_value=refined)
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=mock.Mock(
                return_value=(
                    None,
                    {"detected": False, "planned": False, "bound": False},
                )
            ),
            plan_depth_refined_path=exact_refinement,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                )

        exact_refinement.assert_called_once()
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 9.8)
        self.assertIsNotNone(selected["depth_refinement"])
        self.assertFalse(
            selected.get("committed_history_runtime_guard_selected", False)
        )

    def test_missing_wall_shortcut_falls_back_to_guarded_history(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[20, 20] = True
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": mask,
        }
        primary = {
            "ok": True,
            "path_length_m": 9.5,
            "direct_distance_m": 8.0,
            "ordinary_route_unsupported_by_history_m": 1.7,
            "ordinary_route_unsupported_by_history_fraction": 0.18,
        }
        map_only = {
            "ok": True,
            "path_length_m": 9.7,
            "direct_distance_m": 8.0,
            "traversed_route_direct_replay": True,
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        exact_refinement = mock.Mock(return_value=None)
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=mock.Mock(
                return_value=(
                    None,
                    {"detected": False, "planned": False, "bound": False},
                )
            ),
            plan_depth_refined_path=exact_refinement,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                )

        self.assertEqual(exact_refinement.call_count, 2)
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 9.7)
        self.assertTrue(selected["traversed_route_direct_replay"])
        self.assertTrue(selected["committed_history_runtime_guard_selected"])
        self.assertEqual(
            selected["history_runtime_guard_selection_reason"],
            "ordinary_route_leaves_traversed_corridor",
        )

    def test_depth_disconnection_falls_back_only_to_guarded_history(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[20, 20] = True
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": mask,
        }
        primary = {"ok": False, "error": "unreachable"}
        map_only = {
            "ok": True,
            "path_length_m": 7.5,
            "direct_distance_m": 7.0,
            "traversed_route_direct_replay": True,
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        exact_refinement = mock.Mock(return_value=None)
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=mock.Mock(
                return_value=(
                    None,
                    {"detected": False, "planned": False, "bound": False},
                )
            ),
            plan_depth_refined_path=exact_refinement,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                )

        self.assertEqual(exact_refinement.call_count, 2)
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 7.5)
        self.assertTrue(selected["traversed_route_direct_replay"])
        self.assertTrue(selected["committed_history_runtime_guard_selected"])
        self.assertEqual(
            selected["history_runtime_guard_selection_reason"],
            "live_depth_disconnects_traversed_corridor",
        )

    def test_carried_door_certificate_is_reused_before_redetection(self):
        canonical = {"raw_component_point_count": 321}
        redetect = mock.Mock()
        recovered = {
            "ok": True,
            "path_xy_m": [[0.0, 0.0], [1.0, 0.0]],
            "traversed_thin_barrier": canonical,
        }
        with mock.patch.multiple(
            tools,
            _navigate_to_validate_traversed_thin_barrier=mock.Mock(
                return_value=canonical
            ),
            _navigate_to_plan_traversed_thin_barrier_detection=mock.Mock(
                return_value=(
                    recovered,
                    {"planned": True, "bound": True},
                )
            ),
            _navigate_to_detect_traversed_thin_barrier=redetect,
        ):
            result, diagnostics = (
                tools._navigate_to_traversed_thin_barrier_recovery(
                    None,
                    snapshot={"kind": "current"},
                    map_only_plan={"ok": True},
                    footprint_radius_m=.42,
                    progress_check=lambda: None,
                    carried_certificate={"schema": "signed"},
                    carried_path_xy_m=[[0.0, 0.0], [1.0, 0.0]],
                )
            )

        self.assertIs(result, recovered)
        self.assertTrue(diagnostics["carried_certificate_evaluated"])
        self.assertTrue(diagnostics["carried_certificate_selected"])
        self.assertEqual(
            diagnostics["detection_source"],
            "integrity_checked_carried_certificate",
        )
        redetect.assert_not_called()

    def test_new_plan_keeps_a_carried_door_route_on_an_ordinary_replan(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": np.zeros_like(grid, dtype=bool),
        }
        primary = {
            "ok": True,
            "path_length_m": 2.0,
            "direct_distance_m": 2.0,
        }
        map_only = dict(primary)
        recovered = {
            "ok": True,
            "path_length_m": 2.2,
            "direct_distance_m": 2.0,
            "traversed_thin_barrier": {"schema": "signed"},
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        recovery = mock.Mock(return_value=(
            recovered,
            {
                "detected": True,
                "planned": True,
                "bound": True,
                "carried_certificate_selected": True,
            },
        ))
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": .42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=recovery,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": .25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                    carried_traversed_thin_barrier={"schema": "signed"},
                    carried_traversed_path_xy_m=[
                        [0.0, 0.0], [1.0, 0.0]
                    ],
                )

        recovery.assert_called_once()
        self.assertEqual(
            recovery.call_args.kwargs["carried_certificate"],
            {"schema": "signed"},
        )
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertIs(selected["traversed_thin_barrier"], recovered[
            "traversed_thin_barrier"
        ])
        self.assertTrue(selected["depth_detour_recovery_selected"])
        self.assertTrue(selected["traversed_thin_barrier_recovery_carried"])

    def test_alignment_refresh_is_scoped_to_the_signed_door_segments(self):
        trajectory = {
            "traversed_thin_barrier": {"segment_indices": [1, 2]}
        }
        self.assertFalse(
            tools._navigate_to_segment_has_traversed_thin_barrier(
                trajectory, 0
            )
        )
        self.assertTrue(
            tools._navigate_to_segment_has_traversed_thin_barrier(
                trajectory, 1
            )
        )
        self.assertFalse(
            tools._navigate_to_segment_has_traversed_thin_barrier({}, 1)
        )

    def test_successful_depth_detour_stays_safe_without_a_door_certificate(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        mask = np.zeros_like(grid, dtype=bool)
        mask[20, 20] = True
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_obstacle_mask": mask,
        }
        primary = {
            "ok": True,
            "path_length_m": 4.2,
            "direct_distance_m": 2.0,
        }
        map_only = {
            "ok": True,
            "path_length_m": 2.1,
            "direct_distance_m": 2.0,
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        regular_refinement = mock.Mock()
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=mock.Mock(
                return_value=(
                    None,
                    {"detected": False, "planned": False, "bound": False},
                )
            ),
            plan_depth_refined_path=regular_refinement,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                )

        regular_refinement.assert_not_called()
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 4.2)
        self.assertFalse(selected["depth_detour_recovery_selected"])

    def test_stall_plane_forces_exact_refinement_of_committed_history(self):
        grid = np.zeros((40, 40), dtype=np.int8)
        snapshot = {
            "occupancy": grid,
            "free_mask": np.ones_like(grid, dtype=bool),
            "navigation_stall_depth_points_xy_m": np.asarray(
                [[1.0, y] for y in np.linspace(0.6, 1.4, 17)],
                dtype=np.float64,
            ),
        }
        primary = {
            "ok": True,
            "path_length_m": 2.0,
            "direct_distance_m": 2.0,
            "traversed_route_direct_replay": True,
        }
        map_only = dict(primary)
        refined = {
            **primary,
            "path_length_m": 2.6,
            "traversed_route_direct_replay": False,
            "depth_refinement": {"schema": nav.DEPTH_REFINEMENT_SCHEMA},
        }
        compile_plan = mock.Mock(return_value={"schema": "test"})
        exact_refinement = mock.Mock(return_value=refined)
        with mock.patch.multiple(
            tools,
            _check_cancelled=mock.Mock(),
            _navigate_to_capture_inactive_limb_contract=mock.Mock(
                return_value=({}, None)
            ),
            _navigate_to_base_footprint=mock.Mock(
                return_value={"radius_m": 0.42}
            ),
            _navigate_to_depth_planning_snapshot=mock.Mock(
                return_value=snapshot
            ),
            _navigate_to_traversed_thin_barrier_recovery=mock.Mock(
                return_value=(
                    None,
                    {"detected": False, "planned": False, "bound": False},
                )
            ),
            plan_depth_refined_path=exact_refinement,
            _navigate_to_compile_trajectory=compile_plan,
            _navigate_to_write_plan=mock.Mock(),
            _navigate_to_load_plan=mock.Mock(
                return_value=({"ok": True}, "/tmp/plan_test.json")
            ),
            _next_plan_id=mock.Mock(return_value="plan_test"),
        ):
            with mock.patch.object(
                tools, "plan_clearance_path", side_effect=[primary, map_only]
            ):
                tools._navigate_to_new_plan(
                    None,
                    args={"name": "goal", "arrival_tolerance_m": 0.25},
                    storage_session_id="test",
                    snapshot=snapshot,
                    deadline=time.monotonic() + 10.0,
                    require_traversed_route=True,
                )

        exact_refinement.assert_called_once()
        selected = compile_plan.call_args.kwargs["planner_result"]
        self.assertEqual(selected["path_length_m"], 2.6)
        self.assertEqual(selected["execution_stall_depth_point_count"], 17)
        self.assertFalse(
            selected.get("committed_history_runtime_guard_selected", False)
        )

    def test_noncontact_door_route_requires_full_observation_reproof(self):
        detection = {
            "certificate": {"schema": nav.TRAVERSED_THIN_BARRIER_SCHEMA}
        }
        passage_snapshot = {"kind": "private_contact_override"}
        full_snapshot = {"kind": "full_observation"}
        contact_candidate = {
            "ok": True,
            "path_xy_m": [[0.0, 0.0], [1.0, 0.0]],
        }
        fully_certified = {
            "ok": True,
            "path_xy_m": [[0.0, 0.0], [0.5, 0.3], [1.0, 0.0]],
            "path_length_m": 1.2,
        }
        planner = mock.Mock(side_effect=[contact_candidate, fully_certified])
        with mock.patch.multiple(
            tools,
            _navigate_to_depth_obstacles=mock.Mock(
                return_value=np.zeros((4, 3), dtype=np.float64)
            ),
            _navigate_to_detect_traversed_thin_barrier=mock.Mock(
                return_value=(detection, [], [], "latest_depth_frame")
            ),
            apply_traversed_thin_barrier_override=mock.Mock(
                return_value=passage_snapshot
            ),
            bind_traversed_thin_barrier_segments=mock.Mock(return_value=None),
            restore_traversed_thin_barrier_observation=mock.Mock(
                return_value=full_snapshot
            ),
            plan_depth_refined_path=planner,
        ):
            recovered, diagnostics = (
                tools._navigate_to_traversed_thin_barrier_recovery(
                    None,
                    snapshot={"kind": "original"},
                    map_only_plan={"ok": True},
                    footprint_radius_m=.42,
                    progress_check=lambda: None,
                )
            )

        self.assertIsNotNone(recovered)
        self.assertEqual(
            recovered["traversed_thin_barrier_recovery_mode"],
            "full_observation_avoidance",
        )
        self.assertNotIn("traversed_thin_barrier", recovered)
        self.assertFalse(diagnostics["bound"])
        self.assertTrue(diagnostics["full_observation_certified"])
        self.assertEqual(planner.call_count, 2)
        self.assertIs(planner.call_args_list[1].args[0], full_snapshot)

    def _sweep(self, points, *, translation=(0.0, 0.0), yaw=0.0,
               center=(-0.7, 0.0, 1.2), initial_yaw=0.0, margin=0.02, point_error=0.0):
        corners = np.array(list(itertools.product(*[(v - 0.04, v + 0.04) for v in center])))
        transform = np.eye(4)
        c, s = math.cos(initial_yaw), math.sin(initial_yaw)
        transform[:2, :2] = [[c, -s], [s, c]]
        model = (0.42, (("upper", ("box",), corners),), 1, "", "")
        with mock.patch.object(footprint, "_observed_navigation_transforms", return_value={"upper": transform}):
            with mock.patch.object(footprint, "_collision_aabb_model", return_value=model):
                return footprint.navigation_upper_body_sweep_is_clear(
                    np.asarray(points), {}, arm_dof=8, translation_xy=translation,
                    yaw_delta_rad=yaw, margin_m=margin, point_error_m=point_error,
                )

    def test_overhang_above_base_does_not_block_a_displaced_upper_body(self):
        self.assertTrue(self._sweep([[0.0, 0.0, 1.2]], translation=(0.2, 0.0)))
        self.assertFalse(self._sweep([[-0.35, 0.0, 1.2]], translation=(0.4, 0.0)))

    def test_rotation_checks_the_arc_not_just_endpoint_boxes(self):
        self.assertFalse(self._sweep(
            [[0.0, 1.0, 1.2]], center=(1.0, 0.0, 1.2),
            initial_yaw=math.pi / 4.0, yaw=math.pi / 2.0,
        ))
        self.assertTrue(self._sweep([[0.0, 0.0, 2.0]], yaw=math.pi / 2.0))

    def test_margin_only_proximity_allows_proven_separating_retreat(self):
        point = [[-0.6, 0.0, 1.2]]
        self.assertTrue(self._sweep(point, translation=(-0.2, 0.0), margin=0.12, point_error=0.04))
        self.assertFalse(self._sweep(point, translation=(0.2, 0.0), margin=0.12, point_error=0.04))
        self.assertFalse(self._sweep([[-0.65, 0.0, 1.2]], translation=(-0.2, 0.0),
                                     margin=0.12, point_error=0.04))

    def test_translation_parallel_to_near_surface_preserves_clearance(self):
        self.assertTrue(self._sweep([[-0.6, 0.0, 1.2]], translation=(0.0, 0.2),
                                   margin=0.12, point_error=0.04))
        self.assertFalse(self._sweep([[-0.6, 0.0, 1.2]], translation=(0.02, 0.2),
                                    margin=0.12, point_error=0.04))

    def test_diagonal_translation_does_not_fill_the_endpoint_aabb(self):
        points = np.array([[0.3, -0.3, 1.2], [0.3, 0.3, 1.2]])
        distances = footprint._translated_box_sweep_distances(
            points, np.array([-0.04, -0.04, 1.16]),
            np.array([0.04, 0.04, 1.24]), np.array([0.6, 0.6, 0.0]),
        )
        np.testing.assert_allclose(distances, [math.sqrt(2.0) * 0.26, 0.0], atol=1e-12)
        self.assertTrue(self._sweep(points[:1], center=(0.0, 0.0, 1.2),
                                   translation=(0.6, 0.6), margin=0.35))
        self.assertFalse(self._sweep(points[1:], center=(0.0, 0.0, 1.2),
                                    translation=(0.6, 0.6), margin=0.02))

    def test_analytic_translation_matches_dense_independent_oracle(self):
        rng = np.random.default_rng(721)
        points = rng.uniform(-0.6, 0.6, (24, 3))
        lower, upper = np.full(3, -0.04), np.full(3, 0.04)
        times = np.linspace(0.0, 1.0, 4001)
        for translation in (np.zeros(3), np.array([0.7, 0.0, 0.0]),
                            np.array([-0.4, 0.6, -0.2])):
            with self.subTest(translation=translation):
                analytic = footprint._translated_box_sweep_distances(points, lower, upper, translation)
                sampled = points[:, None, :] - times[None, :, None] * translation
                sampled -= np.clip(sampled, lower, upper)
                oracle = np.linalg.norm(sampled, axis=2).min(axis=1)
                self.assertTrue(np.all(analytic <= oracle + 1.0e-12))
                np.testing.assert_allclose(analytic, oracle, atol=0.0002)

    def test_sweep_encloses_interior_pose_collisions_for_both_turn_directions(self):
        for angle in (-1.5, -0.3, 0.3, 1.5):
            for fraction in (0.1, 0.4, 0.7, 0.9):
                with self.subTest(angle=angle, fraction=fraction):
                    c, s = math.cos(angle * fraction), math.sin(angle * fraction)
                    point = [c + 0.2 * fraction, s - 0.1 * fraction, 1.2]
                    self.assertFalse(self._sweep(
                        [point], center=(1.0, 0.0, 1.2), yaw=angle, translation=(0.2, -0.1),
                    ))

    def test_walked_evidence_cannot_erase_a_freshly_observed_wall(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        protected = np.zeros_like(grid, dtype=bool)
        protected[:, 40] = True
        snapshot = {"occupancy": grid, "resolution": 0.1, "origin": [0.0, 0.0],
                    "pose": {"x": 2.0, "y": 4.0, "yaw_deg": 0.0},
                    "places": [{"name": "goal", "x": 6.0, "y": 4.0}],
                    "traversed_paths_xy_m": [[[2.0, 4.0], [6.0, 4.0]]],
                    "navigation_obstacle_mask": protected}
        result = plan_clearance_path(snapshot, "goal")
        self.assertFalse(result["ok"], result)
