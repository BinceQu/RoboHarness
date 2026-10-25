from __future__ import annotations

from pathlib import Path
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from behavior_interface_eval_test.navigation_map_bridge import (
    NAVIGATION_MAP_SCHEMA,
    NAVIGATION_MAP_SCHEMA_VERSION,
    ORDERED_POSE_PRODUCER_SCHEMA,
    NavigationMapBridge,
    normalize_navigation_map_snapshot,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
    OfficialPolicyRuntime,
)


def _external_snapshot() -> dict:
    return {
        "schema": NAVIGATION_MAP_SCHEMA,
        "schema_version": NAVIGATION_MAP_SCHEMA_VERSION,
        "episode_id": "episode-a",
        "map_epoch": "provider-epoch-a",
        "map_version": "map-a",
        "frame_version": "frame-a",
        "pose_version": "frame-a",
        "frame": "plugin-map",
        "origin": [-1.0, -2.0],
        "resolution": 0.1,
        "occupancy": np.asarray(
            [[-1, 0, 100], [0, 0, 100]],
            dtype=np.int8,
        ),
        "pose": {
            "x": -0.85,
            "y": -1.85,
            "yaw_deg": 10.0,
            "frame": "plugin-map",
            "global_confident": True,
        },
        "places": [{"name": "goal", "x": -0.75, "y": -1.85}],
    }


def _rtab_mapper(cell: int = 0) -> SimpleNamespace:
    occupancy = np.full((2, 3), cell, dtype=np.int8)
    result = SimpleNamespace(
        frame_id=1,
        map_updated=True,
        node_count=1,
        loop_count=0,
        occupancy=occupancy,
        low_obstacles=np.zeros_like(occupancy, dtype=np.uint8),
        high_obstacles=np.zeros_like(occupancy, dtype=np.uint8),
        x_min_m=0.0,
        y_min_m=0.0,
        cell_size_m=0.1,
        current_pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0),
        tracking_ok=True,
        recovery_hold=False,
        uncertain_hold=False,
        soft_localization=False,
        read_only_match=False,
        mapping_active=True,
        localized=False,
        visual_localized=False,
        geometric_localized=False,
    )
    return SimpleNamespace(
        _state_lock=threading.RLock(),
        _latest=result,
        _places_version=0,
        _start_place=None,
        _places=[],
    )


def _legacy_grid(*, occupied: bool, revision: int = 0) -> SimpleNamespace:
    obstacle = np.zeros((2, 2), dtype=np.bool_)
    obstacle[0, 0] = occupied
    free = ~obstacle
    return SimpleNamespace(
        _rev=revision,
        frames=revision,
        half_span_m=1.0,
        resolution_m=0.5,
        observed_free_mask=lambda: free,
        observed_occupied_mask=lambda: obstacle,
        observed_wall_mask=lambda: obstacle,
        observed_overhead_mask=lambda: np.zeros_like(obstacle),
    )


class NavigationMapBridgeTest(unittest.TestCase):
    def test_rtab_anchored_trail_appends_without_rebuilding_history(self) -> None:
        mapper = _rtab_mapper()
        mapper._latest.poses = (
            SimpleNamespace(node_id=1, x_m=2.0, y_m=3.0, yaw_rad=0.0),
        )

        class CountedTrailPose:
            def __init__(self, x_m: float) -> None:
                self.anchor_node_id = 1
                self._local_pose = SimpleNamespace(
                    x_m=x_m, y_m=0.0, yaw_rad=0.0
                )
                self.fallback_pose = SimpleNamespace(
                    x_m=99.0, y_m=99.0, yaw_rad=0.0
                )
                self.reads = 0

            @property
            def local_pose(self):
                self.reads += 1
                return self._local_pose

        first_item = CountedTrailPose(0.1)
        mapper._trail = [first_item]
        mapper._display_result = mock.Mock(
            side_effect=AssertionError("native full-trail resolver was used")
        )
        bridge = NavigationMapBridge()

        first = bridge.capture(
            mapper,
            episode_id="incremental-trail",
            observation_sequence=1,
        )
        assert first is not None
        self.assertEqual(first["traversed_paths_xy_m"], (((2.1, 3.0),),))
        self.assertEqual(first_item.reads, 1)

        second_item = CountedTrailPose(0.2)
        mapper._trail.append(second_item)
        mapper._latest.frame_id = 2
        second = bridge.capture(
            mapper,
            episode_id="incremental-trail",
            observation_sequence=2,
        )
        assert second is not None
        self.assertEqual(
            second["traversed_paths_xy_m"],
            (((2.1, 3.0), (2.2, 3.0)),),
        )
        self.assertEqual(first_item.reads, 1)
        self.assertEqual(second_item.reads, 1)

        mapper._latest.poses = (
            SimpleNamespace(node_id=1, x_m=4.0, y_m=3.0, yaw_rad=0.0),
        )
        mapper._latest.frame_id = 3
        corrected = bridge.capture(
            mapper,
            episode_id="incremental-trail",
            observation_sequence=3,
        )
        assert corrected is not None
        self.assertEqual(
            corrected["traversed_paths_xy_m"],
            (((4.1, 3.0), (4.2, 3.0)),),
        )
        self.assertEqual(first_item.reads, 2)
        self.assertEqual(second_item.reads, 2)

    def test_rtab_cached_frame_does_not_resolve_the_full_trail_again(self) -> None:
        mapper = _rtab_mapper()
        mapper._trail = [object(), object()]
        displayed = SimpleNamespace(
            poses=(SimpleNamespace(x_m=0.05, y_m=0.05),)
        )
        mapper._display_result = mock.Mock(return_value=displayed)
        bridge = NavigationMapBridge()

        first = bridge.capture(
            mapper,
            episode_id="cached-trail",
            observation_sequence=1,
        )
        second = bridge.capture(
            mapper,
            episode_id="cached-trail",
            observation_sequence=2,
        )

        assert first is not None and second is not None
        self.assertEqual(mapper._display_result.call_count, 1)
        self.assertIs(
            first["traversed_paths_xy_m"],
            second["traversed_paths_xy_m"],
        )
        self.assertEqual(second["observation_sequence"], 2)

    def test_rtab_traversed_route_is_immutable_split_and_versioned(self) -> None:
        mapper = _rtab_mapper()
        trail = [
            SimpleNamespace(x_m=0.05, y_m=0.05),
            SimpleNamespace(x_m=0.08, y_m=0.05),
            SimpleNamespace(x_m=0.25, y_m=0.05),
            SimpleNamespace(x_m=2.0, y_m=2.0),
            SimpleNamespace(x_m=2.15, y_m=2.0),
        ]
        mapper._trail = list(trail)
        mapper._display_result = lambda _heading_up: SimpleNamespace(
            poses=tuple(trail)
        )
        bridge = NavigationMapBridge()

        first = bridge.capture(
            mapper,
            episode_id="traversed-route",
            observation_sequence=1,
        )
        assert first is not None
        self.assertEqual(
            first["traversed_paths_xy_m"],
            (
                ((0.08, 0.05), (0.25, 0.05)),
                ((2.0, 2.0), (2.15, 2.0)),
            ),
        )
        self.assertIsInstance(first["traversed_paths_xy_m"], tuple)

        trail.append(SimpleNamespace(x_m=2.30, y_m=2.0))
        mapper._trail.append(object())
        second = bridge.capture(
            mapper,
            episode_id="traversed-route",
            observation_sequence=2,
        )
        assert second is not None
        self.assertNotEqual(first["map_version"], second["map_version"])
        self.assertEqual(second["traversed_paths_xy_m"][-1][-1], (2.30, 2.0))

    def test_rtab_layers_and_absolute_places_are_canonicalised(self) -> None:
        occupancy = np.asarray(
            [[-1, 0, 0, 100], [-1, 0, 25, 80]],
            dtype=np.int8,
        )
        low = np.zeros_like(occupancy, dtype=np.uint8)
        low[1, 0] = 1
        low[1, 3] = 1
        high = np.zeros_like(occupancy, dtype=np.uint8)
        high[0, 2] = 1
        result = SimpleNamespace(
            frame_id=7,
            loop_count=2,
            occupancy=occupancy,
            low_obstacles=low,
            high_obstacles=high,
            x_min_m=-3.0,
            y_min_m=-4.0,
            cell_size_m=0.05,
            current_pose=SimpleNamespace(x_m=1.2, y_m=-0.4, yaw_rad=0.25),
            tracking_ok=True,
            recovery_hold=False,
            mapping_active=True,
            localized=False,
            visual_localized=False,
            geometric_localized=False,
        )
        mapper = SimpleNamespace(
            _state_lock=threading.RLock(),
            _latest=result,
            _places_version=3,
            _start_place=SimpleNamespace(
                name="start",
                x_m=0.0,
                y_m=0.0,
                source="automatic",
                image_id="",
            ),
            _places=[
                SimpleNamespace(
                    name="desk",
                    x_m=2.25,
                    y_m=-1.5,
                    source="image_pick",
                    image_id="capture-2",
                )
            ],
        )
        bridge = NavigationMapBridge()

        snapshot = bridge.capture(
            mapper,
            episode_id="episode-r",
            observation_sequence=11,
            policy_local_pose=[0.3, 0.4, -0.1],
            captured_ts=20.0,
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        np.testing.assert_array_equal(
            snapshot["occupancy"],
            np.asarray(
                [[-1, 0, -1, 100], [100, 0, -1, 100]],
                dtype=np.int8,
            ),
        )
        self.assertFalse(snapshot["occupancy"].flags.writeable)
        np.testing.assert_array_equal(
            snapshot["free_mask"], snapshot["occupancy"] == 0
        )
        np.testing.assert_array_equal(
            snapshot["obstacle_mask"], snapshot["occupancy"] == 100
        )
        self.assertTrue(bool(snapshot["wall_mask"][1, 3]))
        self.assertEqual(snapshot["origin"], [-3.0, -4.0])
        self.assertEqual(snapshot["resolution"], 0.05)
        self.assertEqual(snapshot["places"][1]["name"], "desk")
        self.assertEqual(snapshot["places"][1]["x"], 2.25)
        self.assertEqual(snapshot["places"][1]["y"], -1.5)
        self.assertEqual(snapshot["pose"]["frame"], "rtabmap_contact_aware_q")
        self.assertAlmostEqual(snapshot["pose"]["yaw_deg"], 14.323944878)
        self.assertTrue(snapshot["lifecycle"]["pose_confident"])
        self.assertEqual(
            snapshot["policy_local_pose"]["frame"],
            "policy_local_odometry",
        )

        # Provider-owned arrays are copied before publication.
        occupancy[0, 3] = 0
        self.assertEqual(int(snapshot["occupancy"][0, 3]), 100)

    def test_grid_arrays_have_an_irreversibly_read_only_bytes_owner(self) -> None:
        snapshot = normalize_navigation_map_snapshot(_external_snapshot())

        for array in (
            snapshot["occupancy"],
            *snapshot["layers"].values(),
        ):
            owner = array
            while isinstance(owner, np.ndarray):
                self.assertFalse(owner.flags.writeable)
                with self.assertRaises(ValueError):
                    owner.setflags(write=True)
                owner = owner.base
            self.assertIsInstance(owner, bytes)

        adapter = ObservationActionAdapter()
        adapter.publish_navigation_map_snapshot(_external_snapshot())
        public_view = adapter.navigation_map_snapshot()
        assert public_view is not None
        owner = public_view["occupancy"]
        while isinstance(owner, np.ndarray):
            with self.assertRaises(ValueError):
                owner.setflags(write=True)
            owner = owner.base
        self.assertIsInstance(owner, bytes)

    def test_cached_grid_refreshes_pose_and_recovery_lifecycle(self) -> None:
        result = SimpleNamespace(
            frame_id=4,
            loop_count=0,
            occupancy=np.zeros((3, 4), dtype=np.int8),
            low_obstacles=np.zeros((3, 4), dtype=np.uint8),
            high_obstacles=np.zeros((3, 4), dtype=np.uint8),
            x_min_m=0.0,
            y_min_m=0.0,
            cell_size_m=0.1,
            current_pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0),
            tracking_ok=True,
            recovery_hold=False,
            mapping_active=True,
            localized=False,
            visual_localized=False,
            geometric_localized=False,
        )
        mapper = SimpleNamespace(
            _state_lock=threading.RLock(),
            _latest=result,
            _places_version=0,
            _start_place=None,
            _places=[],
        )
        bridge = NavigationMapBridge()
        first = bridge.capture(
            mapper,
            episode_id="episode-r",
            observation_sequence=1,
        )
        assert first is not None

        result.current_pose = SimpleNamespace(x_m=0.6, y_m=0.2, yaw_rad=0.5)
        result.frame_id = 5
        result.map_updated = False
        result.recovery_hold = True
        second = bridge.capture(
            mapper,
            episode_id="episode-r",
            observation_sequence=2,
        )
        assert second is not None

        self.assertIs(first["occupancy"], second["occupancy"])
        self.assertEqual(second["pose"]["x"], 0.6)
        self.assertAlmostEqual(second["pose"]["yaw_deg"], 28.647889756)
        self.assertFalse(second["pose"]["global_confident"])
        self.assertTrue(second["lifecycle"]["recovery_hold"])
        self.assertEqual(second["observation_sequence"], 2)

    def test_rtab_pose_first_seen_stays_fixed_until_latest_advances(self) -> None:
        mapper = _rtab_mapper()
        bridge = NavigationMapBridge()
        first = bridge.capture(
            mapper,
            episode_id="pose-freshness",
            observation_sequence=10,
            captured_ts=100.0,
        )
        assert first is not None
        self.assertEqual(first["pose_observed_sequence"], 10)
        self.assertEqual(first["pose_observed_ts"], 100.0)
        self.assertEqual(first["pose_age_observations"], 0)
        self.assertEqual(first["pose_age_s"], 0.0)

        frozen = bridge.capture(
            mapper,
            episode_id="pose-freshness",
            observation_sequence=14,
            captured_ts=100.8,
        )
        assert frozen is not None
        self.assertEqual(frozen["pose_version"], first["pose_version"])
        self.assertEqual(frozen["pose_observed_sequence"], 10)
        self.assertEqual(frozen["pose_observed_ts"], 100.0)
        self.assertEqual(frozen["pose_age_observations"], 4)
        self.assertAlmostEqual(frozen["pose_age_s"], 0.8)
        self.assertIs(frozen["occupancy"], first["occupancy"])

        mapper._latest = SimpleNamespace(
            **{
                **vars(mapper._latest),
                "frame_id": 2,
                "map_updated": False,
                "current_pose": SimpleNamespace(
                    x_m=0.2,
                    y_m=0.1,
                    yaw_rad=0.1,
                ),
            }
        )
        advanced = bridge.capture(
            mapper,
            episode_id="pose-freshness",
            observation_sequence=15,
            captured_ts=101.0,
        )
        assert advanced is not None
        self.assertNotEqual(advanced["pose_version"], first["pose_version"])
        self.assertEqual(advanced["pose_observed_sequence"], 15)
        self.assertEqual(advanced["pose_observed_ts"], 101.0)
        self.assertEqual(advanced["pose_age_observations"], 0)
        self.assertEqual(advanced["pose_age_s"], 0.0)
        self.assertEqual(advanced["pose"]["x"], 0.2)
        self.assertIs(advanced["occupancy"], first["occupancy"])

    def test_rtab_source_sequence_is_known_only_when_worker_is_caught_up(self) -> None:
        mapper = _rtab_mapper()
        mapper._input_lock = threading.RLock()
        mapper.frames_enqueued = 1
        mapper.frames_processed = 1
        mapper._last_sequence = 8
        bridge = NavigationMapBridge()

        caught_up = bridge.capture(
            mapper,
            episode_id="pose-source",
            observation_sequence=10,
            captured_ts=20.0,
        )
        assert caught_up is not None
        self.assertTrue(caught_up["pose_source_sequence_known"])
        self.assertEqual(caught_up["pose_source_observation_sequence"], 8)
        self.assertTrue(caught_up["source_frame"]["lag_known"])
        self.assertEqual(caught_up["source_frame"]["lag_observations"], 2)
        progress = caught_up["source_frame"]["ordered_pose_producer"]
        self.assertEqual(progress["schema"], ORDERED_POSE_PRODUCER_SCHEMA)
        self.assertEqual(progress["enqueued_frame_count"], 1)
        self.assertEqual(progress["processed_frame_count"], 1)
        self.assertEqual(progress["pose_frame_id"], 1)
        self.assertEqual(progress["last_enqueued_observation_sequence"], 8)

        mapper.frames_enqueued = 2
        mapper._last_sequence = 11
        queued = bridge.capture(
            mapper,
            episode_id="pose-source",
            observation_sequence=11,
            captured_ts=20.2,
        )
        assert queued is not None
        self.assertFalse(queued["pose_source_sequence_known"])
        self.assertNotIn("pose_source_observation_sequence", queued)
        self.assertFalse(queued["source_frame"]["lag_known"])
        queued_progress = queued["source_frame"]["ordered_pose_producer"]
        self.assertEqual(queued_progress["enqueued_frame_count"], 2)
        self.assertEqual(queued_progress["processed_frame_count"], 1)
        self.assertEqual(
            queued_progress["last_enqueued_observation_sequence"], 11
        )

        mapper._latest = SimpleNamespace(
            **{
                **vars(mapper._latest),
                "frame_id": 2,
                "map_updated": False,
            }
        )
        mapper.frames_processed = 2
        caught_up_again = bridge.capture(
            mapper,
            episode_id="pose-source",
            observation_sequence=12,
            captured_ts=20.4,
        )
        assert caught_up_again is not None
        self.assertTrue(caught_up_again["pose_source_sequence_known"])
        self.assertEqual(
            caught_up_again["pose_source_observation_sequence"], 11
        )
        completed_progress = caught_up_again["source_frame"][
            "ordered_pose_producer"
        ]
        self.assertEqual(completed_progress["enqueued_frame_count"], 2)
        self.assertEqual(completed_progress["processed_frame_count"], 2)

    def test_rtab_snapshot_marks_a_long_running_worker_request_stalled(self) -> None:
        mapper = _rtab_mapper()
        mapper._input_lock = threading.RLock()
        mapper.frames_enqueued = 2
        mapper.frames_processed = 1
        mapper._last_sequence = 11
        mapper._worker_started_monotonic_s = time.monotonic() - 11.0
        mapper.worker_stall_warning_s = 10.0
        mapper.worker_timed_out = False
        mapper.disabled = False

        snapshot = NavigationMapBridge().capture(
            mapper,
            episode_id="stalled-producer",
            observation_sequence=12,
        )

        assert snapshot is not None
        lifecycle = snapshot["lifecycle"]
        self.assertTrue(lifecycle["worker_busy"])
        self.assertGreaterEqual(lifecycle["worker_busy_s"], 10.0)
        self.assertTrue(lifecycle["worker_stalled"])
        self.assertFalse(lifecycle["worker_healthy"])

    def test_rtab_pending_frame_keeps_exact_accepted_pose_sequence(self) -> None:
        mapper = _rtab_mapper()
        mapper._input_lock = threading.RLock()
        mapper.frames_enqueued = 2
        mapper.frames_processed = 1
        mapper._last_sequence = 11
        mapper._last_processed_sequence = 8
        bridge = NavigationMapBridge()
        snapshot = bridge.capture(
            mapper, episode_id="delayed-pose", observation_sequence=12,
        )
        self.assertTrue(snapshot["pose_source_sequence_known"])
        self.assertEqual(snapshot["pose_source_observation_sequence"], 8)
        self.assertEqual(snapshot["source_frame"]["lag_observations"], 4)
        self.assertEqual(
            snapshot["source_frame"]["ordered_pose_producer"]
            ["last_enqueued_observation_sequence"], 11,
        )
        for invalid in (True, 12, -1, None, "8"):
            with self.subTest(processed_sequence=invalid):
                mapper._last_processed_sequence = invalid
                snapshot = bridge.capture(
                    mapper, episode_id="delayed-pose", observation_sequence=12,
                )
                self.assertFalse(snapshot["pose_source_sequence_known"])
                self.assertNotIn("pose_source_observation_sequence", snapshot)

    def test_malformed_ordered_pose_progress_is_rejected(self) -> None:
        snapshot = _external_snapshot()
        snapshot["observation_sequence"] = 12
        snapshot["source_frame"] = {
            "ordered_pose_producer": {
                "schema": ORDERED_POSE_PRODUCER_SCHEMA,
                "enqueued_frame_count": 3,
                "processed_frame_count": 2,
                "pose_frame_id": 1,
                "last_enqueued_observation_sequence": 11,
            }
        }

        with self.assertRaisesRegex(ValueError, "latest processed frame"):
            normalize_navigation_map_snapshot(snapshot)

    def test_rtab_uncertain_hold_marks_pose_untrusted(self) -> None:
        mapper = _rtab_mapper()
        bridge = NavigationMapBridge()
        trusted = bridge.capture(
            mapper,
            episode_id="episode-uncertain",
            observation_sequence=8,
        )
        assert trusted is not None
        self.assertTrue(trusted["lifecycle"]["pose_confident"])

        mapper._latest.frame_id = 2
        mapper._latest.uncertain_hold = True
        mapper._latest.map_updated = False
        snapshot = bridge.capture(
            mapper,
            episode_id="episode-uncertain",
            observation_sequence=9,
        )

        assert snapshot is not None
        self.assertIs(trusted["occupancy"], snapshot["occupancy"])
        self.assertFalse(snapshot["pose"]["global_confident"])
        self.assertFalse(snapshot["lifecycle"]["pose_confident"])
        self.assertTrue(snapshot["lifecycle"]["uncertain_hold"])
        self.assertFalse(snapshot["source_frame"]["lag_known"])
        self.assertFalse(snapshot["lifecycle"]["source_lag_known"])

    def test_rtab_localization_frames_keep_stable_geometry_version(self) -> None:
        result = SimpleNamespace(
            frame_id=10,
            map_updated=True,
            node_count=6,
            loop_count=1,
            occupancy=np.zeros((3, 4), dtype=np.int8),
            low_obstacles=np.zeros((3, 4), dtype=np.uint8),
            high_obstacles=np.zeros((3, 4), dtype=np.uint8),
            x_min_m=-1.0,
            y_min_m=-2.0,
            cell_size_m=0.1,
            current_pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0),
            tracking_ok=True,
            recovery_hold=False,
            mapping_active=False,
            localized=True,
            visual_localized=True,
            geometric_localized=True,
        )
        mapper = SimpleNamespace(
            _state_lock=threading.RLock(),
            _latest=result,
            _places_version=2,
            _start_place=None,
            _places=[],
        )
        bridge = NavigationMapBridge()
        first = bridge.capture(
            mapper,
            episode_id="episode-localization",
            observation_sequence=20,
        )
        assert first is not None

        result.frame_id = 11
        result.map_updated = False
        result.current_pose = SimpleNamespace(x_m=0.4, y_m=0.1, yaw_rad=0.2)
        # A native response owns fresh array objects even when it reports that
        # the frozen geometry did not change.  The bridge must not copy them.
        result.occupancy = result.occupancy.copy()
        result.low_obstacles = result.low_obstacles.copy()
        result.high_obstacles = result.high_obstacles.copy()
        localized = bridge.capture(
            mapper,
            episode_id="episode-localization",
            observation_sequence=21,
        )
        assert localized is not None

        self.assertEqual(localized["map_version"], first["map_version"])
        self.assertNotEqual(localized["frame_version"], first["frame_version"])
        self.assertTrue(localized["frame_version"].endswith("rtabmap-frame:11"))
        self.assertIs(localized["occupancy"], first["occupancy"])
        self.assertEqual(localized["pose"]["x"], 0.4)
        self.assertFalse(localized["lifecycle"]["map_updated_this_frame"])

        result.frame_id = 12
        result.map_updated = True
        result.occupancy[0, 0] = 100
        updated = bridge.capture(
            mapper,
            episode_id="episode-localization",
            observation_sequence=22,
        )
        assert updated is not None

        self.assertNotEqual(updated["map_version"], localized["map_version"])
        self.assertIsNot(updated["occupancy"], localized["occupancy"])
        self.assertFalse(
            np.shares_memory(updated["occupancy"], localized["occupancy"])
        )
        self.assertEqual(int(updated["occupancy"][0, 0]), 100)
        self.assertTrue(updated["lifecycle"]["map_updated_this_frame"])

    def test_same_episode_rtab_mapper_restart_changes_stream_identity(self) -> None:
        bridge = NavigationMapBridge()
        first_mapper = _rtab_mapper(0)
        second_mapper = _rtab_mapper(100)

        first = bridge.capture(
            first_mapper,
            episode_id="same-episode",
            observation_sequence=1,
        )
        restarted = bridge.capture(
            second_mapper,
            episode_id="same-episode",
            observation_sequence=2,
        )

        assert first is not None and restarted is not None
        self.assertNotEqual(first["map_epoch"], restarted["map_epoch"])
        self.assertNotEqual(first["map_version"], restarted["map_version"])
        self.assertEqual(int(first["occupancy"][0, 0]), 0)
        self.assertEqual(int(restarted["occupancy"][0, 0]), 100)
        self.assertIsNot(first["occupancy"], restarted["occupancy"])

    def test_legacy_egomap_masks_pose_and_marks_are_bridged(self) -> None:
        free = np.asarray(
            [[False, True, True], [False, True, False]], dtype=np.bool_
        )
        obstacle = np.asarray(
            [[True, False, False], [False, False, True]], dtype=np.bool_
        )
        wall = np.asarray(
            [[True, False, False], [False, False, False]], dtype=np.bool_
        )
        high = np.asarray(
            [[False, False, True], [False, False, False]], dtype=np.bool_
        )
        grid = SimpleNamespace(
            _rev=9,
            frames=5,
            half_span_m=2.0,
            resolution_m=0.2,
            observed_free_mask=lambda: free,
            observed_occupied_mask=lambda: obstacle,
            observed_wall_mask=lambda: wall,
            observed_overhead_mask=lambda: high,
        )
        ego = SimpleNamespace(
            grid=grid,
            x=0.5,
            y=-0.3,
            yaw_deg=-30.0,
            initialized=True,
            update_count=12,
            mapping_state="mapping",
            localization_state="mapping",
            landmarks={
                "start": SimpleNamespace(name="start", x=0.0, y=0.0)
            },
            places=[
                SimpleNamespace(
                    label="sink", x=1.25, y=-0.75, kind="picked", count=2
                )
            ],
        )
        mapper = SimpleNamespace(_ego=lambda: ego)

        snapshot = NavigationMapBridge().capture(
            mapper,
            episode_id="episode-e",
            observation_sequence=8,
        )

        assert snapshot is not None
        np.testing.assert_array_equal(
            snapshot["occupancy"],
            np.asarray([[100, 0, -1], [-1, 0, 100]], dtype=np.int8),
        )
        self.assertEqual(snapshot["origin"], [-2.0, -2.0])
        self.assertEqual(snapshot["pose"]["x"], 0.5)
        self.assertEqual(snapshot["places"][1]["name"], "sink")
        self.assertEqual(snapshot["places"][1]["x"], 1.25)
        self.assertTrue(snapshot["lifecycle"]["pose_confident"])
        self.assertTrue(snapshot["pose_source_sequence_known"])
        self.assertEqual(snapshot["pose_source_observation_sequence"], 8)
        self.assertTrue(snapshot["source_frame"]["lag_known"])
        self.assertEqual(snapshot["source_frame"]["lag_observations"], 0)

    def test_legacy_grid_and_ego_identity_invalidate_same_revision_cache(self) -> None:
        def ego_with(grid: SimpleNamespace) -> SimpleNamespace:
            return SimpleNamespace(
                grid=grid,
                x=0.0,
                y=0.0,
                yaw_deg=0.0,
                initialized=True,
                update_count=1,
                mapping_state="mapping",
                localization_state="mapping",
                landmarks={},
                places=[],
            )

        state = {"ego": ego_with(_legacy_grid(occupied=False, revision=0))}
        mapper = SimpleNamespace(_ego=lambda: state["ego"])
        bridge = NavigationMapBridge()
        first = bridge.capture(
            mapper,
            episode_id="legacy-episode",
            observation_sequence=1,
        )
        assert first is not None

        state["ego"].grid = _legacy_grid(occupied=True, revision=0)
        replaced_grid = bridge.capture(
            mapper,
            episode_id="legacy-episode",
            observation_sequence=2,
        )
        assert replaced_grid is not None
        self.assertEqual(first["map_epoch"], replaced_grid["map_epoch"])
        self.assertNotEqual(first["map_version"], replaced_grid["map_version"])
        self.assertEqual(int(first["occupancy"][0, 0]), 0)
        self.assertEqual(int(replaced_grid["occupancy"][0, 0]), 100)
        self.assertIsNot(first["occupancy"], replaced_grid["occupancy"])

        state["ego"] = ego_with(_legacy_grid(occupied=False, revision=0))
        replaced_ego = bridge.capture(
            mapper,
            episode_id="legacy-episode",
            observation_sequence=3,
        )
        assert replaced_ego is not None
        self.assertNotEqual(
            replaced_grid["map_epoch"], replaced_ego["map_epoch"]
        )
        self.assertNotEqual(
            replaced_grid["map_version"], replaced_ego["map_version"]
        )

    def test_legacy_pose_version_keeps_submillimeter_updates(self) -> None:
        grid = _legacy_grid(occupied=False)
        ego = SimpleNamespace(
            grid=grid,
            x=0.0,
            y=0.0,
            yaw_deg=0.0,
            initialized=True,
            update_count=1,
            mapping_state="mapping",
            localization_state="mapping",
            landmarks={},
            places=[],
        )
        mapper = SimpleNamespace(_ego=lambda: ego)
        bridge = NavigationMapBridge()
        first = bridge.capture(
            mapper,
            episode_id="legacy-small-motion",
            observation_sequence=1,
        )
        assert first is not None

        ego.x = 0.000001
        second = bridge.capture(
            mapper,
            episode_id="legacy-small-motion",
            observation_sequence=2,
        )
        assert second is not None
        self.assertNotEqual(first["pose_version"], second["pose_version"])
        self.assertEqual(second["pose"]["x"], 0.000001)
        self.assertEqual(second["pose_source_observation_sequence"], 2)
        self.assertIs(first["occupancy"], second["occupancy"])

    def test_external_provider_is_the_plugin_boundary(self) -> None:
        class Provider:
            def export_navigation_map_snapshot(self):
                return _external_snapshot()

        snapshot = NavigationMapBridge().capture(
            Provider(),
            episode_id="owned-episode",
            observation_sequence=44,
            policy_local_pose=[1.0, 2.0, 0.2],
        )

        assert snapshot is not None
        self.assertEqual(snapshot["schema"], NAVIGATION_MAP_SCHEMA)
        self.assertEqual(
            snapshot["schema_version"], NAVIGATION_MAP_SCHEMA_VERSION
        )
        self.assertEqual(snapshot["episode_id"], "owned-episode")
        self.assertEqual(snapshot["provider_map_epoch"], "provider-epoch-a")
        self.assertEqual(snapshot["provider_map_version"], "map-a")
        self.assertIn("owned-episode", snapshot["map_epoch"])
        self.assertIn("provider-epoch-a", snapshot["map_epoch"])
        self.assertIn(snapshot["map_epoch"], snapshot["map_version"])
        self.assertEqual(snapshot["observation_sequence"], 44)

    def test_external_provider_requires_canonical_schema_and_frame(self) -> None:
        class Provider:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def export_navigation_map_snapshot(self):
                return self.payload

        invalid_cases = []
        wrong_schema = _external_snapshot()
        wrong_schema["schema"] = "other.navigation.map"
        invalid_cases.append((wrong_schema, "schema"))
        wrong_version = _external_snapshot()
        wrong_version["schema_version"] = 9
        invalid_cases.append((wrong_version, "schema_version"))
        missing_epoch = _external_snapshot()
        missing_epoch.pop("map_epoch")
        invalid_cases.append((missing_epoch, "map_epoch"))
        missing_frame_version = _external_snapshot()
        missing_frame_version.pop("frame_version")
        invalid_cases.append((missing_frame_version, "frame_version"))
        mismatched_frame = _external_snapshot()
        mismatched_frame["pose"] = dict(mismatched_frame["pose"])
        mismatched_frame["pose"]["frame"] = "some-other-map"
        invalid_cases.append((mismatched_frame, "frame"))
        missing_pose_version = _external_snapshot()
        missing_pose_version.pop("pose_version")
        invalid_cases.append((missing_pose_version, "pose_version"))
        mismatched_pose_version = _external_snapshot()
        mismatched_pose_version["pose_version"] = "different-frame"
        invalid_cases.append((mismatched_pose_version, "pose_version"))

        for payload, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    NavigationMapBridge().capture(
                        Provider(payload),
                        episode_id="evaluator-episode",
                        observation_sequence=1,
                    )

    def test_external_provider_confidence_is_type_strict_and_fail_closed(self) -> None:
        missing = _external_snapshot()
        missing["pose"] = dict(missing["pose"])
        missing["pose"].pop("global_confident")
        normalized = normalize_navigation_map_snapshot(missing)
        self.assertFalse(normalized["pose"]["global_confident"])
        self.assertFalse(normalized["lifecycle"]["pose_confident"])

        invalid_cases = (
            ("pose", "global_confident"),
            ("lifecycle", "pose_confident"),
        )
        for owner, field in invalid_cases:
            with self.subTest(owner=owner):
                payload = _external_snapshot()
                payload[owner] = dict(payload.get(owner) or {})
                payload[owner][field] = "false"
                with self.assertRaisesRegex(ValueError, "must be boolean"):
                    normalize_navigation_map_snapshot(payload)

    def test_external_provider_may_declare_exact_pose_source_sequence(self) -> None:
        class Provider:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def export_navigation_map_snapshot(self):
                return self.payload

        declared = _external_snapshot()
        declared["pose_source_sequence_known"] = True
        declared["pose_source_observation_sequence"] = 6
        declared["source_frame"] = {"provider_detail": "kept"}
        snapshot = NavigationMapBridge().capture(
            Provider(declared),
            episode_id="evaluator-episode",
            observation_sequence=8,
        )
        assert snapshot is not None
        self.assertTrue(snapshot["pose_source_sequence_known"])
        self.assertEqual(snapshot["pose_source_observation_sequence"], 6)
        self.assertTrue(snapshot["lifecycle"]["source_lag_known"])
        self.assertEqual(snapshot["source_frame"]["lag_observations"], 2)
        self.assertEqual(snapshot["source_frame"]["provider_detail"], "kept")

        future = _external_snapshot()
        future["pose_source_sequence_known"] = True
        future["pose_source_observation_sequence"] = 9
        with self.assertRaisesRegex(
            ValueError, "pose_source_observation_sequence"
        ):
            NavigationMapBridge().capture(
                Provider(future),
                episode_id="evaluator-episode",
                observation_sequence=8,
            )

        wrong_type = _external_snapshot()
        wrong_type["pose_source_sequence_known"] = "true"
        wrong_type["pose_source_observation_sequence"] = 6
        with self.assertRaisesRegex(
            ValueError, "pose_source_sequence_known"
        ):
            NavigationMapBridge().capture(
                Provider(wrong_type),
                episode_id="evaluator-episode",
                observation_sequence=8,
            )

    def test_external_stable_version_reuses_grid_and_epoch_is_composed(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.payload = _external_snapshot()

            def export_navigation_map_snapshot(self):
                return self.payload

        provider = Provider()
        bridge = NavigationMapBridge()
        first = bridge.capture(
            provider,
            episode_id="evaluator-episode",
            observation_sequence=1,
            captured_ts=10.0,
        )
        assert first is not None

        provider.payload["occupancy"] = provider.payload["occupancy"].copy()
        provider.payload["pose"] = dict(provider.payload["pose"])
        provider.payload["pose"]["x"] = 0.4
        provider.payload["frame_version"] = "frame-b"
        provider.payload["pose_version"] = "frame-b"
        stable = bridge.capture(
            provider,
            episode_id="evaluator-episode",
            observation_sequence=2,
            captured_ts=10.5,
        )
        assert stable is not None
        self.assertIs(first["occupancy"], stable["occupancy"])
        self.assertEqual(stable["pose"]["x"], 0.4)
        self.assertEqual(stable["provider_frame_version"], "frame-b")
        self.assertEqual(stable["provider_pose_version"], "frame-b")
        self.assertTrue(stable["frame_version"].endswith("frame-b"))
        self.assertEqual(first["map_epoch"], stable["map_epoch"])
        self.assertEqual(first["map_version"], stable["map_version"])
        self.assertEqual(stable["pose_observed_sequence"], 2)
        self.assertEqual(stable["pose_observed_ts"], 10.5)

        still_stable = bridge.capture(
            provider,
            episode_id="evaluator-episode",
            observation_sequence=3,
            captured_ts=11.0,
        )
        assert still_stable is not None
        self.assertEqual(still_stable["pose_version"], stable["pose_version"])
        self.assertEqual(still_stable["pose_observed_sequence"], 2)
        self.assertEqual(still_stable["pose_observed_ts"], 10.5)
        self.assertEqual(still_stable["pose_age_observations"], 1)
        self.assertEqual(still_stable["pose_age_s"], 0.5)
        self.assertIs(still_stable["occupancy"], stable["occupancy"])

        provider.payload["origin"] = [-2.0, -2.0]
        with self.assertRaisesRegex(ValueError, "origin"):
            bridge.capture(
                provider,
                episode_id="evaluator-episode",
                observation_sequence=4,
                captured_ts=11.1,
            )

        provider.payload = _external_snapshot()
        provider.payload["map_version"] = "map-b"
        provider.payload["occupancy"][0, 0] = 100
        updated = bridge.capture(
            provider,
            episode_id="evaluator-episode",
            observation_sequence=5,
            captured_ts=11.2,
        )
        assert updated is not None
        self.assertEqual(first["map_epoch"], updated["map_epoch"])
        self.assertNotEqual(first["map_version"], updated["map_version"])
        self.assertIsNot(first["occupancy"], updated["occupancy"])
        self.assertEqual(int(updated["occupancy"][0, 0]), 100)

        provider.payload = _external_snapshot()
        provider.payload["map_epoch"] = "provider-epoch-b"
        provider.payload["map_version"] = "map-b"
        new_epoch = bridge.capture(
            provider,
            episode_id="evaluator-episode",
            observation_sequence=6,
            captured_ts=11.3,
        )
        assert new_epoch is not None
        self.assertEqual(new_epoch["provider_map_epoch"], "provider-epoch-b")
        self.assertNotEqual(updated["map_epoch"], new_epoch["map_epoch"])
        self.assertNotEqual(updated["map_version"], new_epoch["map_version"])

    def test_provider_replacement_resets_pose_first_seen_state(self) -> None:
        class Provider:
            def export_navigation_map_snapshot(self):
                return _external_snapshot()

        bridge = NavigationMapBridge()
        first = bridge.capture(
            Provider(),
            episode_id="same-episode",
            observation_sequence=3,
            captured_ts=10.0,
        )
        assert first is not None

        replacement = Provider()
        # Both producers start at the same provider-local pose revision.  A
        # stream replacement must still establish a new first-seen record.
        second = bridge.capture(
            replacement,
            episode_id="same-episode",
            observation_sequence=9,
            captured_ts=12.0,
        )
        assert second is not None
        self.assertNotEqual(first["map_epoch"], second["map_epoch"])
        self.assertNotEqual(first["pose_version"], second["pose_version"])
        self.assertEqual(second["provider_pose_version"], "frame-a")
        self.assertEqual(second["pose_observed_sequence"], 9)
        self.assertEqual(second["pose_observed_ts"], 12.0)
        self.assertEqual(second["pose_age_observations"], 0)
        self.assertEqual(second["pose_age_s"], 0.0)

    def test_adapter_storage_is_defensive_and_reset_clears_it(self) -> None:
        source = _external_snapshot()
        adapter = ObservationActionAdapter()
        adapter.publish_navigation_map_snapshot(source)
        source["origin"][0] = 99.0
        source["occupancy"][0, 1] = 100

        first = adapter.navigation_map_snapshot()
        assert first is not None
        self.assertEqual(first["origin"], [-1.0, -2.0])
        self.assertEqual(int(first["occupancy"][0, 1]), 0)
        self.assertFalse(first["occupancy"].flags.writeable)
        first["origin"][0] = 123.0
        second = adapter.spatial_map_snapshot()
        assert second is not None
        self.assertEqual(second["origin"], [-1.0, -2.0])
        self.assertIsNot(first["occupancy"], second["occupancy"])
        self.assertTrue(
            np.shares_memory(first["occupancy"], second["occupancy"])
        )
        with self.assertRaises(ValueError):
            first["occupancy"][0, 0] = 0
        with self.assertRaises(ValueError):
            first["occupancy"].setflags(write=True)
        self.assertIsNot(first["layers"]["wall"], second["layers"]["wall"])
        self.assertTrue(
            np.shares_memory(
                first["layers"]["wall"], second["layers"]["wall"]
            )
        )

        adapter.reset()
        self.assertIsNone(adapter.navigation_map_snapshot())

    def test_adapter_take_ownership_validates_pose_freshness(self) -> None:
        owned = normalize_navigation_map_snapshot(_external_snapshot())
        owned["source_frame"] = {"lag_known": False, "detail": {"n": 1}}
        adapter = ObservationActionAdapter()

        adapter.publish_navigation_map_snapshot(owned, take_ownership=True)
        owned["source_frame"]["detail"]["n"] = 99
        published = adapter.navigation_map_snapshot()
        assert published is not None
        self.assertEqual(published["pose_version"], "frame-a")
        self.assertEqual(published["pose_observed_sequence"], 0)
        self.assertEqual(published["pose_age_observations"], 0)
        self.assertEqual(published["source_frame"]["detail"]["n"], 1)

        unknown = dict(owned)
        unknown["pose_source_sequence_known"] = False
        unknown["pose_source_observation_sequence"] = 999
        adapter.publish_navigation_map_snapshot(unknown, take_ownership=True)
        published_unknown = adapter.navigation_map_snapshot()
        assert published_unknown is not None
        self.assertFalse(published_unknown["pose_source_sequence_known"])
        self.assertNotIn(
            "pose_source_observation_sequence", published_unknown
        )

        invalid = dict(owned)
        invalid.pop("pose_version")
        with self.assertRaisesRegex(ValueError, "pose_version"):
            ObservationActionAdapter().publish_navigation_map_snapshot(
                invalid,
                take_ownership=True,
            )

    def test_episode_change_clears_before_accepting_a_new_map(self) -> None:
        adapter = ObservationActionAdapter()
        self.assertFalse(adapter.begin_navigation_map_episode("episode-a"))
        adapter.publish_navigation_map_snapshot(_external_snapshot())
        self.assertIsNotNone(adapter.navigation_map_snapshot())

        self.assertTrue(adapter.begin_navigation_map_episode("episode-b"))
        self.assertIsNone(adapter.navigation_map_snapshot())
        with self.assertRaisesRegex(ValueError, "active episode"):
            adapter.publish_navigation_map_snapshot(_external_snapshot())

        replacement = _external_snapshot()
        replacement["episode_id"] = "episode-b"
        replacement["map_version"] = "map-b"
        adapter.publish_navigation_map_snapshot(replacement)
        current = adapter.navigation_map_snapshot()
        assert current is not None
        self.assertEqual(current["episode_id"], "episode-b")

    def test_policy_map_tick_publishes_to_adapter(self) -> None:
        mapper = SimpleNamespace(
            disabled=False,
            last_error="",
            _state_lock=threading.RLock(),
            _latest=SimpleNamespace(
                frame_id=1,
                loop_count=0,
                occupancy=np.zeros((2, 3), dtype=np.int8),
                low_obstacles=np.zeros((2, 3), dtype=np.uint8),
                high_obstacles=np.zeros((2, 3), dtype=np.uint8),
                x_min_m=-0.1,
                y_min_m=-0.2,
                cell_size_m=0.1,
                current_pose=SimpleNamespace(
                    x_m=0.0, y_m=0.0, yaw_rad=0.0
                ),
                tracking_ok=True,
                recovery_hold=False,
                mapping_active=True,
                localized=False,
                visual_localized=False,
                geometric_localized=False,
            ),
            _places_version=0,
            _start_place=None,
            _places=[],
            odom_tick=mock.Mock(),
            map_tick=mock.Mock(),
        )
        adapter = ObservationActionAdapter()
        adapter.update({"task_id": "task"})
        world = SimpleNamespace(
            episode_id=lambda: "runtime-episode",
            policy_local_base_pose=lambda: np.asarray([0.1, 0.2, 0.3]),
        )
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=mock.Mock())
        runtime._spatial_live_mapper = mapper

        runtime._spatial_map_frame(0.05)

        mapper.odom_tick.assert_called_once_with(world, 0.05)
        mapper.map_tick.assert_called_once_with(world)
        snapshot = adapter.navigation_map_snapshot()
        assert snapshot is not None
        self.assertEqual(snapshot["episode_id"], "runtime-episode")
        self.assertEqual(snapshot["observation_sequence"], 1)
        self.assertEqual(snapshot["policy_local_pose"]["x"], 0.1)

    def test_bridge_source_does_not_import_runtime_or_simulator(self) -> None:
        source = (
            Path(__file__).with_name("navigation_map_bridge.py")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn("import behavior_interface", source)
        self.assertNotIn("from behavior_interface", source)
        self.assertNotIn("omnigibson", source.lower())

    def test_layer_shape_mismatch_is_rejected(self) -> None:
        snapshot = _external_snapshot()
        snapshot["layers"] = {"wall": np.zeros((3, 2), dtype=np.bool_)}
        with self.assertRaisesRegex(ValueError, "shape"):
            normalize_navigation_map_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
