from __future__ import annotations

import importlib
from io import BytesIO
import math
import unittest

import numpy as np
from PIL import Image

from behavior_interface import spatial_map
from behavior_interface.rtabmap_slam import live as rtab_live
from behavior_interface.rtabmap_slam.official import SE2Pose
from behavior_interface.rtabmap_slam.protocol import MapResult, PoseRecord
from behavior_interface.rtabmap_slam.render import AGENT_RGBA
from behavior_interface_eval_test import navigation_route_overlay as overlay


def _red_mask(image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)[:, :, :3].astype(np.int16)
    return (
        (rgb[:, :, 0] > 205)
        & (rgb[:, :, 1] > 70)
        & (rgb[:, :, 1] < 195)
        & (rgb[:, :, 2] > 70)
        & (rgb[:, :, 2] < 200)
        & (rgb[:, :, 0] > rgb[:, :, 1] + 35)
    )


def _rtab_result() -> MapResult:
    size = 100
    occupancy = np.full((size, size), -1, dtype=np.int8)
    occupancy[10:90, 10:90] = 0
    occupancy[10:90, 10] = 100
    low = (occupancy == 100).astype(np.uint8)
    pose = SE2Pose(0.0, 0.0, 0.0)
    return MapResult(
        frame_id=7,
        tracking_ok=True,
        map_updated=True,
        loop_closed=False,
        occupancy=occupancy,
        low_obstacles=low,
        high_obstacles=np.zeros_like(low),
        x_min_m=-2.5,
        y_min_m=-2.5,
        cell_size_m=0.05,
        current_pose=pose,
        node_count=7,
        loop_count=0,
        inliers=100,
        features=200,
        poses=(PoseRecord(1, 0.0, 0.0, 0.0),),
        native_pose=pose,
    )


class RouteOverlayStateTest(unittest.TestCase):
    def setUp(self) -> None:
        overlay.clear_all_routes()

    def tearDown(self) -> None:
        overlay.clear_all_routes()

    def test_publish_advance_and_clear_remove_only_walked_prefix(self) -> None:
        revision = overlay.publish_route(
            "agent-a",
            "plan-1",
            ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0)),
            current_xy_m=(0.2, 0.0),
        )
        first = overlay.get_route_snapshot("agent-a")
        self.assertEqual(first.revision, revision)
        self.assertEqual(
            first.points_xy_m,
            ((0.2, 0.0), (1.0, 0.0), (2.0, 1.0)),
        )

        overlay.advance_route(
            "agent-a", "plan-1", (0.7, 0.0), next_waypoint_index=1
        )
        middle = overlay.get_route_snapshot("agent-a")
        self.assertEqual(
            middle.points_xy_m,
            ((0.7, 0.0), (1.0, 0.0), (2.0, 1.0)),
        )
        overlay.advance_route(
            "agent-a", "plan-1", (1.2, 0.2), next_waypoint_index=2
        )
        final_segment = overlay.get_route_snapshot("agent-a")
        self.assertEqual(
            final_segment.points_xy_m,
            ((1.2, 0.2), (2.0, 1.0)),
        )

        stale_revision = overlay.clear_route("agent-a", "old-plan")
        self.assertEqual(stale_revision, final_segment.revision)
        self.assertIsNotNone(overlay.get_route_snapshot("agent-a"))
        overlay.clear_route("agent-a", "plan-1")
        self.assertIsNone(overlay.get_route_snapshot("agent-a"))

    def test_display_fallback_is_only_used_for_one_unambiguous_route(self) -> None:
        overlay.publish_route("agent-a", "a", ((0.0, 0.0), (1.0, 0.0)))
        fallback = overlay.get_route_for_display("default")
        self.assertEqual(fallback.session_id, "agent-a")
        overlay.publish_route("agent-b", "b", ((0.0, 0.0), (0.0, 1.0)))
        self.assertIsNone(overlay.get_route_for_display("default"))
        self.assertEqual(
            overlay.get_route_for_display("agent-a").plan_id,
            "a",
        )

    def test_stale_plan_cannot_advance_replacement_route(self) -> None:
        overlay.publish_route("agent-a", "old", ((0.0, 0.0), (1.0, 0.0)))
        overlay.publish_route("agent-a", "new", ((0.0, 0.0), (0.0, 2.0)))
        before = overlay.get_route_snapshot("agent-a")
        overlay.advance_route(
            "agent-a", "old", (0.8, 0.0), next_waypoint_index=1
        )
        self.assertEqual(overlay.get_route_snapshot("agent-a"), before)


class RouteOverlayRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        overlay.clear_all_routes()
        spatial_map.reset_all_maps()
        token = overlay.capture_renderer_hook_state()
        self.addCleanup(overlay.restore_renderer_hook_state, token)
        overlay.install_renderer_hooks()

    def tearDown(self) -> None:
        overlay.clear_all_routes()
        spatial_map.reset_all_maps()

    def _ego(self):
        ego = spatial_map.get_map("route-render")
        ego.ensure_start(tool="unit_test")
        # A symmetric known floor keeps the automatic view stable while the
        # route prefix changes.
        values = np.arange(-2.0, 2.01, 0.10)
        xx, yy = np.meshgrid(values, values)
        for _ in range(3):
            ego.grid.add_floor(xx.ravel(), yy.ravel())
        ego.grid.frames = 1
        return ego

    def test_egomap_route_cache_updates_trim_and_clear(self) -> None:
        ego = self._ego()
        blank, blank_version = spatial_map.map_snapshot_png(
            ego.session_id, size=400
        )
        overlay.publish_route(
            ego.session_id,
            "plan-1",
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        )
        planned, planned_version = spatial_map.map_snapshot_png(
            ego.session_id, size=400
        )
        self.assertNotEqual(planned_version, blank_version)
        planned_red = _red_mask(Image.open(BytesIO(planned)).convert("RGBA"))
        self.assertGreater(int(np.count_nonzero(planned_red)), 30)

        overlay.advance_route(
            ego.session_id,
            "plan-1",
            (1.0, 0.0),
            next_waypoint_index=2,
        )
        trimmed, trimmed_version = spatial_map.map_snapshot_png(
            ego.session_id, size=400
        )
        self.assertNotEqual(trimmed_version, planned_version)
        trimmed_red = _red_mask(Image.open(BytesIO(trimmed)).convert("RGBA"))
        self.assertLess(
            int(np.count_nonzero(trimmed_red)),
            int(np.count_nonzero(planned_red)) * 0.72,
        )

        overlay.clear_route(ego.session_id, "plan-1")
        cleared, cleared_version = spatial_map.map_snapshot_png(
            ego.session_id, size=400
        )
        self.assertNotEqual(cleared_version, trimmed_version)
        self.assertEqual(cleared, blank)

    def test_rtab_live_snapshot_tracks_route_revision_and_preserves_agent(self) -> None:
        mapper = rtab_live.LiveMapper(client_factory=lambda: None)
        self.addCleanup(mapper.close)
        mapper._latest = _rtab_result()
        mapper._session_id = "rtab-route"
        blank, blank_version = mapper.map_snapshot_png(size=400, span_m=5.0)
        overlay.publish_route(
            "rtab-route",
            "plan-1",
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        )
        planned, planned_version = mapper.map_snapshot_png(size=400, span_m=5.0)
        self.assertNotEqual(planned_version, blank_version)
        rgba = np.asarray(Image.open(BytesIO(planned)).convert("RGBA"))
        self.assertGreater(int(np.count_nonzero(_red_mask(rgba))), 30)
        center = (400 - 1) // 2
        np.testing.assert_array_equal(
            rgba[center, center, :3],
            np.asarray(AGENT_RGBA[:3], dtype=np.uint8),
        )

        overlay.advance_route(
            "rtab-route", "plan-1", (1.0, 0.0), next_waypoint_index=2
        )
        trimmed, trimmed_version = mapper.map_snapshot_png(
            size=400, span_m=5.0
        )
        self.assertNotEqual(trimmed_version, planned_version)
        self.assertLess(
            int(np.count_nonzero(_red_mask(Image.open(BytesIO(trimmed))))),
            int(np.count_nonzero(_red_mask(rgba))),
        )
        overlay.clear_route("rtab-route", "plan-1")
        cleared, _version = mapper.map_snapshot_png(size=400, span_m=5.0)
        self.assertEqual(cleared, blank)

    def test_rtab_native_line_gets_a_metric_buffer_in_the_same_auto_view(self) -> None:
        from behavior_interface.rtabmap_slam import render as rtab_render

        mapper = rtab_live.LiveMapper(client_factory=lambda: None)
        self.addCleanup(mapper.close)
        mapper._latest = _rtab_result()
        mapper._session_id = "rtab-sweep"
        points = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.0))
        overlay.publish_route("rtab-sweep", "plan-sweep", points)
        png, version = mapper.map_snapshot_png(size=400)
        image = np.asarray(Image.open(BytesIO(png)).convert("RGBA"))
        pose, span = rtab_render._auto_map_view(
            mapper._latest, yaw_rad=0, extra_points_xy_m=points
        )
        # The route extends beyond the mapped area; both layers must use its
        # native auto-fit view, not a second robot-centered transform.
        x, y = rtab_render._to_pixel(3.5, -0.25, pose, 400, span)
        red, green, blue = image[int(round(y)), int(round(x)), :3]
        self.assertGreater(int(red), int(green) + 15)
        self.assertIn("navigation-sweep:0.4", version)
        self.assertEqual(mapper._latest.node_count, 7)


class RouteOverlayHotInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        # Navigation execution tests intentionally leave the process-wide
        # renderer hook installed.  This test exercises installation from the
        # native baseline, so establish that baseline explicitly instead of
        # depending on pytest module order.
        overlay.uninstall_renderer_hooks()
        overlay.clear_all_routes()

    def tearDown(self) -> None:
        overlay.uninstall_renderer_hooks()
        overlay.clear_all_routes()

    def test_hot_install_does_not_replace_any_live_map_state(self) -> None:
        spatial_map.reset_all_maps()
        ego = spatial_map.get_map("identity")
        maps_id = id(spatial_map._MAPS)
        ego_id = id(ego)
        old_singleton = rtab_live._SINGLETON
        sentinel = object()
        rtab_live._SINGLETON = sentinel
        functions = (
            spatial_map.render_minimap_rgba,
            spatial_map.map_version,
            rtab_live.LiveMapper.map_snapshot_png,
        )
        try:
            for function in functions:
                if hasattr(function, "_navigation_route_overlay_native"):
                    delattr(function, "_navigation_route_overlay_native")
            token = overlay.capture_renderer_hook_state()
            report = overlay.install_renderer_hooks()
            self.assertTrue(report["ok"])
            self.assertTrue(report["state_identity_preserved"])
            self.assertEqual(id(spatial_map._MAPS), maps_id)
            self.assertEqual(id(spatial_map.get_map("identity")), ego_id)
            self.assertIs(rtab_live._SINGLETON, sentinel)
            self.assertEqual(len(report["installed"]), 3)

            # A second module generation replaces, rather than stacks,
            # wrappers, and its opaque old-class token restores gen1 exactly.
            generation_one = spatial_map.render_minimap_rgba
            generation_token = overlay.capture_renderer_hook_state()
            overlay.publish_route(
                "identity", "survives-reload", ((0.0, 0.0), (1.0, 0.0))
            )
            importlib.reload(overlay)
            self.assertEqual(
                overlay.get_route_snapshot("identity").plan_id,
                "survives-reload",
            )
            second = overlay.install_renderer_hooks()
            self.assertEqual(len(second["installed"]), 3)
            self.assertIs(
                spatial_map.render_minimap_rgba
                ._navigation_route_overlay_original,
                functions[0],
            )
            overlay.restore_renderer_hook_state(generation_token)
            self.assertIs(spatial_map.render_minimap_rgba, generation_one)
            rollback = overlay.restore_renderer_hook_state(token)
            self.assertTrue(rollback["state_identity_preserved"])
            self.assertIs(spatial_map.render_minimap_rgba, functions[0])
            self.assertIs(spatial_map.map_version, functions[1])
            self.assertIs(
                rtab_live.LiveMapper.map_snapshot_png,
                functions[2],
            )
        finally:
            overlay.uninstall_renderer_hooks()
            rtab_live._SINGLETON = old_singleton
            for function in functions:
                function._navigation_route_overlay_native = True


if __name__ == "__main__":
    unittest.main()
