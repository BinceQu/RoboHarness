from __future__ import annotations

import base64
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from embodied_claude_code.catalog import ToolCatalog, _input_schema
from embodied_claude_code.config import Settings
from embodied_claude_code.coordinates import (
    VLM_IMAGE_COORDINATE_CONTRACT, arguments_to_interface, interface_to_pixel,
    pixel_to_interface, response_to_pixels, schema_uses_image_coordinates,
)
from embodied_claude_code.errors import ToolPolicyError
from embodied_claude_code.media import Media, MediaExtractor
from embodied_claude_code.profile import ToolProfile
from embodied_claude_code.qwen_bridge import BridgeConfig, translate_messages_request
from embodied_claude_code.server import create_mcp_server
from embodied_claude_code.service import EmbodiedService
from embodied_claude_code.skills import discover_task_skills
from fake_behavior import FakeBehaviorClient, PNG_720, png_image


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "tests/fixtures/interface_tools_15064.json").read_text())
POINT_TOOLS = {
    "read_depth", "move_chassis_to_floor_point",
    "move_chassis_to_directly_facing_surface", "spin_to_facing_point",
    "move_point_to_point", "plan_eef_translation_to_uvd_point",
    "move_to_reach_point", "measure_shoulder_distance",
    "plan_grasp_point_filter", "plan_grasp_point_filter_rgbd",
    "plan_grasp_point_filter_rgbd_lite", "plan_press_point", "mark_on_map",
    "cut_object", "track_object_distance",
}


def point_arguments(name: str, image_id: str) -> dict:
    args = {"image_id": image_id, "u": 345, "v": 548}
    if name == "move_chassis_to_directly_facing_surface":
        args = {"image_id": image_id, "points": [
            {"u": 345, "v": 548}, {"u": 400, "v": 500}, {"u": 500, "v": 400},
        ]}
    if name in {"move_point_to_point", "plan_grasp_point_filter_rgbd", "cut_object", "track_object_distance"}:
        args = {"image_id": image_id, "points": [
            {"u": 345, "v": 548}, {"u": 719, "v": 0},
        ]}
        for index, point in enumerate(args["points"]):
            if name in {"cut_object", "track_object_distance"}:
                point["name"] = ("cutting_tool_point", "target_object_point")[index]
            else:
                point["plan_arm"] = "any" if name == "move_point_to_point" else ("left", "right")[index]
    if name == "mark_on_map":
        args["name"] = "radio"
    if name == "plan_eef_translation_to_uvd_point":
        args.update(depth=0.63, arm="right")
    return args


class PixelClient(FakeBehaviorClient):
    def __init__(self):
        super().__init__()
        self.tools = deepcopy(CATALOG["tools"])
        self.frame = PNG_720
        self.feed = "head"
        self.declared = (720, 720)
        self.sequence = 0
        self.reply = None

    def post_json(self, path, payload):
        self.requests.append({"path": path, "body": deepcopy(payload)})
        if path == "/api/v2/capture_head_camera":
            self.sequence += 1
            return {"ok": True, "image_id": f"img_{self.sequence:04d}",
                    "feed": self.feed, "image_width": self.declared[0],
                    "image_height": self.declared[1],
                    "rgb_main": "data:image/png;base64," + base64.b64encode(self.frame).decode()}
        return deepcopy(self.reply) if self.reply is not None else {"ok": True, "received": deepcopy(payload)}


class PixelMathTests(unittest.TestCase):
    def test_every_pixel_round_trips_through_unchanged_interface_exactly(self):
        wire = [pixel_to_interface(pixel) for pixel in range(720)]
        self.assertEqual(len(set(wire)), 720)
        for pixel, uv in enumerate(wire):
            # Independent copy of the interface's published inverse formula.
            self.assertEqual(round(uv / 1000 * (720 - 1)), pixel)
            self.assertEqual(interface_to_pixel(uv), pixel)
        self.assertEqual((wire[0], wire[-1]), (0, 1000))

    def test_radio_session_three_clicks_remain_the_exact_selected_pixels(self):
        for u, v in ((345, 548), (375, 500), (368, 497)):
            wire = arguments_to_interface({"image_id": "test", "u": u, "v": v})
            self.assertEqual([round(wire[axis] / 1000 * 719) for axis in ("u", "v")], [u, v])

    def test_invalid_pixels_fail_closed_even_without_json_schema(self):
        for bad in (-1, 720, 1000, True, False, 1.0, 1.1, "345", None, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ToolPolicyError):
                pixel_to_interface(bad)
        with self.assertRaises(ToolPolicyError):
            arguments_to_interface({"points": [{"u": 30}]})

    def test_response_inverse_covers_interface_public_aliases_without_scaling_xyz(self):
        wire = {"coordinate_system": "qwen3vl_relative_0_1000",
                "coordinate_range": {"u": [0, 1000], "v": [0, 1000]},
                "points": [{"name": "p", "u": 480, "v": 762, "depth_m": 0.63}],
                "center_u": 1000, "center_v": 500,
                "qwen_u": 480, "qwen_v": 762,
                "uv_bbox": [0, 0, 1000, 1000],
                "corners_uv": [{"u": 480, "v": 762}],
                "pixel_uv": [345, 548], "hit_pixel": [345, 549],
                "target_xyz_m": [0.9, 0.33, -0.004],
                "target_robot_m": [0.9, 0.33, -0.004],
                "nearby_object_warning": "nearest (391,526) 0.12 m; chassis blocked"}
        aliases = ("uv", "input_uv", "pick_uv", "point_uv", "projected_uv",
                   "projected_uv_clipped", "resolved_uv", "vlm_uv", "reproj_uv",
                   "contact_uv", "grasp_point_uv", "relative_uv", "qwen_uv")
        for key in aliases:
            wire[key] = [480, 762]
        original = deepcopy(wire)
        pixels = response_to_pixels(wire)
        self.assertEqual(wire, original)
        for key in aliases:
            self.assertEqual(pixels["input_pixel_uv" if key == "relative_uv" else key], [345, 548], key)
        self.assertNotIn("relative_uv", pixels)
        self.assertEqual(pixels["points"], [{"name": "p", "u": 345, "v": 548, "depth_m": 0.63}])
        self.assertEqual(pixels["corners_uv"], [{"u": 345, "v": 548}])
        self.assertEqual(pixels["uv_bbox"], [0, 0, 719, 719])
        self.assertEqual(pixels["center_u"], 719)
        self.assertEqual(pixels["center_v"], 360)
        for key in ("pixel_uv", "hit_pixel", "target_xyz_m", "target_robot_m"):
            self.assertEqual(pixels[key], wire[key])
        self.assertEqual(pixels["coordinate_range"], {"u": [0, 719], "v": [0, 719]})
        self.assertIn("0.12 m", pixels["nearby_object_warning"])
        self.assertNotIn("391,526", pixels["nearby_object_warning"])
        self.assertEqual(response_to_pixels(pixels), pixels)
        metric = {"coordinate_system": "robot_base", "xyz": [0.3, 0.4, 0.1]}
        self.assertEqual(response_to_pixels(metric), metric)


class PixelServiceTests(unittest.TestCase):
    def setUp(self):
        # Exercise excluded legacy coordinate schemas without changing policy.
        self.policy_patch = patch("embodied_claude_code.profile.MCP_EXCLUDED_TOOLS", frozenset())
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)
        self.fake = PixelClient()
        self.service = EmbodiedService(Settings(session_id="pixel720-test", record=False),
                                      client=self.fake, profile=ToolProfile(deny_tools=frozenset()))
        self.catalog = self.service.prepare_episode(session_id="pixel720-test", record=False)

    def capture(self):
        result = self.service.call(tool_name="capture_head_camera", arguments={})
        return result.data["response"]["image_id"]

    def test_every_live_coordinate_tool_converts_at_the_service_boundary(self):
        self.assertEqual({name for name, spec in self.catalog.tools.items()
                          if schema_uses_image_coordinates(spec.input_schema)}, POINT_TOOLS)
        raw_by_name = {tool["name"]: tool for tool in CATALOG["tools"]}
        for name in sorted(POINT_TOOLS):
            with self.subTest(name=name):
                image_id = self.capture()
                args = point_arguments(name, image_id)
                original = deepcopy(args)
                result = self.service.call(tool_name=name, arguments=args)
                self.assertFalse(result.is_error)
                self.assertEqual(args, original)
                wire = self.fake.requests[-1]["body"]
                expected = arguments_to_interface(args)
                spec = self.catalog.get(name)
                if name == "plan_eef_translation_to_uvd_point":
                    self.assertTrue(spec.adapter_image_id)
                    expected.pop("image_id")
                expected.update(spec.fixed_arguments, session_id="pixel720-test")
                self.assertEqual(wire, expected)
                self.assertEqual(self.fake.requests[-1]["path"], spec.endpoint)
                # The exact original wire schema also accepts the transformed call.
                schema = _input_schema(raw_by_name[name], hidden={"session_id", *spec.fixed_arguments})
                Draft202012Validator(schema).validate({k: v for k, v in wire.items()
                                                      if k != "session_id" and k not in spec.fixed_arguments})
                received = result.data["response"]["received"]
                for key in ("u", "v", "points", "depth", "arm"):
                    if key in args:
                        self.assertEqual(received[key], args[key])

    def test_every_tool_rejects_unknown_image_before_any_http_post(self):
        for name in POINT_TOOLS:
            with self.subTest(name=name):
                before = len(self.fake.requests)
                with self.assertRaises(ToolPolicyError):
                    self.service.call(tool_name=name, arguments=point_arguments(name, "never-seen"))
                self.assertEqual(len(self.fake.requests), before)

    def test_all_schemas_and_metadata_have_pixel_units_including_nested_points(self):
        for name, spec in self.catalog.tools.items():
            with self.subTest(name=name):
                serialized = json.dumps(spec.public())
                self.assertNotIn("0..1000", serialized)
                self.assertNotIn("relative_0_1000", serialized)
                if name not in POINT_TOOLS:
                    continue
                args = point_arguments(name, "test")
                spec.validate_arguments(args)
                if "points" in args:
                    args["points"][-1]["v"] = 720
                else:
                    args["v"] = 720
                with self.assertRaises(ToolPolicyError):
                    spec.validate_arguments(args)

    def test_click_rejects_wrong_view_and_actual_source_or_declared_dimensions(self):
        for size, feed, declared in (
            ((1, 1), "head", (720, 720)), ((480, 480), "head", (480, 480)),
            ((720, 480), "head", (720, 480)), ((1440, 1440), "head", (720, 720)),
            ((720, 720), "left_wrist", (720, 720)),
            ((720, 720), "unknown", (720, 720)), ((720, 720), "head", (1440, 1440)),
        ):
            with self.subTest(size=size, feed=feed, declared=declared):
                self.fake.frame = png_image(*size)
                self.fake.feed = feed
                self.fake.declared = declared
                image_id = self.capture()
                before = len(self.fake.requests)
                with self.assertRaises(ToolPolicyError):
                    self.service.call(tool_name="mark_on_map", arguments=point_arguments("mark_on_map", image_id))
                self.assertEqual(len(self.fake.requests), before)

    def test_stale_head_and_imageless_motion_require_new_capture(self):
        first = self.capture()
        second = self.capture()
        with self.assertRaises(ToolPolicyError):
            self.service.call(tool_name="mark_on_map", arguments=point_arguments("mark_on_map", first))
        self.service.call(tool_name="adjust_chassis", arguments={"forward": 0.01})
        with self.assertRaises(ToolPolicyError):
            self.service.call(tool_name="mark_on_map", arguments=point_arguments("mark_on_map", second))
        fresh = self.capture()
        self.service.call(tool_name="mark_on_map", arguments=point_arguments("mark_on_map", fresh))

    def test_name_only_map_and_object_distance_do_not_require_click_image(self):
        for name, args in (("mark_on_map", {"name": "here"}),
                           ("measure_shoulder_distance", {"object_name": "radio"})):
            result = self.service.call(tool_name=name, arguments=args)
            self.assertFalse(result.is_error)
            self.assertEqual(self.fake.requests[-1]["body"], {**args, "session_id": "pixel720-test"})

    def test_partial_point_and_missing_id_never_fall_back_to_interface_defaults(self):
        image_id = self.capture()
        for args in ({"name": "radio", "u": 3, "v": 4},
                     {"name": "radio", "image_id": image_id, "u": 3},
                     {"name": "radio", "image_id": image_id, "v": 4}):
            before = len(self.fake.requests)
            with self.assertRaises(ToolPolicyError):
                self.service.call(tool_name="mark_on_map", arguments=args)
            self.assertEqual(len(self.fake.requests), before)

    def test_nested_observation_binds_the_image_actually_selected(self):
        self.fake.reply = {"ok": True, "observation": {
            "ok": True, "image_id": "nested-head", "feed": "head",
            "rgb_main": "data:image/png;base64," + base64.b64encode(PNG_720).decode()}}
        result = self.service.call(tool_name="adjust_chassis", arguments={"forward": 0.01})
        self.assertTrue(result.data["image_geometry"]["clickable"])
        self.assertEqual(result.data["image_geometry"]["image_id"], "nested-head")
        self.service._assert_click_image("nested-head")

    def test_new_head_id_cannot_hide_a_nested_wrist_view(self):
        self.fake.reply = {"ok": True, "image_id": "head-label", "feed": "head",
                           "observation": {"image_id": "head-label", "feed": "left_wrist",
                           "rgb_main": "data:image/png;base64," + base64.b64encode(PNG_720).decode()}}
        result = self.service.call(tool_name="adjust_chassis", arguments={"forward": 0.01})
        self.assertFalse(result.data["image_geometry"]["clickable"])
        with self.assertRaises(ToolPolicyError):
            self.service._assert_click_image("head-label")

    def test_persistent_track_uv_is_never_an_unbound_click_suggestion(self):
        self.fake.memory["raw"]["tracked_object_distances"] = {
            "radio": {"u": 500, "v": 730, "source_image_id": "old-image",
                      "depth_m": 0.75, "xyz_in_robot_base_coord_m": [0.9, 0.33, 0.04]}}
        result = self.service.call(tool_name="mark_on_map", arguments={"name": "here"})
        track = result.data["persistent_tracking"]["tracks"]["radio"]
        self.assertNotIn("u", track)
        self.assertNotIn("v", track)
        self.assertEqual(track["xyz_in_robot_base_coord_m"], [0.9, 0.33, 0.04])

    def test_no_numeric_guessing_and_no_coordinate_conversion_in_bridge(self):
        payload = {"system": VLM_IMAGE_COORDINATE_CONTRACT, "messages": [
            {"role": "user", "content": "Select the radio"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "p1",
             "name": "mcp__behavior-v2__mark_on_map",
             "input": {"image_id": "img_0001", "name": "radio", "u": 345, "v": 548}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "p1", "content": "ok"}]}],
            "tools": [{"name": "mcp__behavior-v2__mark_on_map",
                       "input_schema": self.catalog.get("mark_on_map").input_schema}],
            "max_tokens": 32}
        translated = translate_messages_request(payload, BridgeConfig(upstream_url="http://127.0.0.1:9/v1/chat/completions", model="Qwen3.8-Flash-Next-FP8"))
        call = next(message for message in translated["messages"] if message.get("tool_calls"))["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"])["u"], 345)
        self.assertEqual(json.loads(call["function"]["arguments"])["v"], 548)
        self.assertEqual(translated["tools"][0]["function"]["parameters"]["properties"]["u"]["maximum"], 719)


class PixelPromptTests(unittest.TestCase):
    def test_advertised_skills_and_session_hook_use_original_pixels(self):
        paths = [ROOT / 'skills' / doc.name / 'SKILL.md' for doc in discover_task_skills(ROOT)]
        paths.append(ROOT / 'skills/behavior-v2-baseline/SKILL.md')
        self.assertGreater(len(paths), 1)
        for path in paths + [ROOT / "prompts/session-context.md"]:
            with self.subTest(path=path):
                text = path.read_text()
                self.assertIn("EVERY", text)
                self.assertIn("720 x 720", text)
                self.assertIn("719", text)
                self.assertIn("Do not normalize", text)
                self.assertNotIn("1000", text)


class PixelMediaTests(unittest.TestCase):
    def test_old_nested_frame_does_not_outrank_current_display_image(self):
        extractor = MediaExtractor(FakeBehaviorClient(), max_bytes=1_000_000, max_images=1)
        old = png_image(480, 480)
        selection = extractor.extract({"image_id": "current", "feed": "head",
            "rgb_main": "data:image/png;base64," + base64.b64encode(PNG_720).decode(),
            "old_observation": {"image_id": "old", "feed": "head",
                "rgb_path": "data:image/png;base64," + base64.b64encode(old).decode()}},
            tool_name="set_arm_to_grasp_position")
        self.assertEqual(selection.selected[0].data, PNG_720)
        self.assertEqual(selection.selected[0].source_image_id, "current")
        self.assertTrue(selection.warnings)

    def test_jpeg_transcoding_preserves_original_geometry_and_binding(self):
        extractor = MediaExtractor(FakeBehaviorClient(), max_bytes=1_000_000, max_images=1)
        original = Media.create(label="rgb_main", mime_type="image/png",
            data=png_image(1440, 1440), source_image_id="original", source_feed="head")
        bounded = extractor._bound_for_model(original)
        self.assertEqual((bounded.width, bounded.height), (720, 720))
        self.assertEqual((bounded.source_width, bounded.source_height), (1440, 1440))
        self.assertEqual(bounded.source_image_id, "original")
        self.assertEqual(bounded.source_feed, "head")


@unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP SDK is absent")
class NativeMCPPixelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        runtime = tempfile.TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        env = patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime.name})
        env.start()
        self.addCleanup(env.stop)
        monitor = patch("embodied_claude_code.skills.publish_loaded_skills", return_value=True)
        monitor.start()
        self.addCleanup(monitor.stop)

    async def test_all_advertised_point_tools_and_all_skill_loads_through_native_mcp(self):
        from mcp.client import Client
        from mcp.types import TextContent
        fake = PixelClient()
        service = EmbodiedService(Settings(session_id="all-tools-pixel-test", record=False), client=fake)
        async with Client(create_mcp_server(service), mode="legacy") as client:
            for name in sorted(POINT_TOOLS - set(ToolProfile.baseline().deny_tools)):
                with self.subTest(name=name):
                    await client.call_tool("capture_head_camera", {})
                    image_id = f"img_{fake.sequence:04d}"
                    args = point_arguments(name, image_id)
                    result = await client.call_tool(name, args)
                    self.assertFalse(result.is_error)
                    wire = fake.requests[-1]["body"]
                    if "points" in wire:
                        self.assertEqual(wire["points"][0]["u"], 480)
                    else:
                        self.assertEqual((wire["u"], wire["v"]), (480, 762))
            advertised = {doc.name for doc in discover_task_skills(ROOT)}
            for path in (ROOT / "skills").glob("*/SKILL.md"):
                result = await client.call_tool("activate_skill", {"name": path.parent.name})
                text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
                if path.parent.name == "behavior-v2-baseline":
                    self.assertTrue(result.is_error)
                    self.assertIn("already active", text)
                    continue
                if path.parent.name not in advertised:
                    self.assertTrue(result.is_error)
                    self.assertIn("Unknown task Skill", text)
                    continue
                self.assertFalse(result.is_error)
                self.assertIn("720 x 720", text)
                self.assertIn("0..719", text)
                self.assertNotIn("1000", text)

    async def test_native_mcp_passes_real_pixels_through_and_preserves_failure_images(self):
        from mcp.client import Client
        from mcp.types import ImageContent, TextContent
        fake = PixelClient()
        service = EmbodiedService(Settings(session_id="native-pixel-test", record=False), client=fake)
        async with Client(create_mcp_server(service), mode="legacy") as client:
            capture = await client.call_tool("capture_head_camera", {})
            caption = next(block.text for block in capture.content if isinstance(block, TextContent) and block.text.startswith("image "))
            self.assertIn("clickable=true", caption)
            self.assertIn("width=720 height=720", caption)
            result = await client.call_tool("mark_on_map", {"name": "radio", "image_id": "img_0001", "u": 345, "v": 548})
            self.assertFalse(result.is_error)
            self.assertEqual([fake.requests[-1]["body"][key] for key in ("u", "v")], [480, 762])
            before = len(fake.requests)
            rejected = await client.call_tool("mark_on_map", {"name": "radio", "image_id": "img_0001", "u": 720, "v": 548})
            self.assertTrue(rejected.is_error)
            self.assertEqual(len(fake.requests), before)
            fake.reply = {"ok": False, "image_id": "img_0002", "feed": "head",
                          "error": "not reachable", "rgb_main": "data:image/png;base64," + base64.b64encode(PNG_720).decode()}
            failed = await client.call_tool("plan_grasp_point_filter_rgbd_lite", {"image_id": "img_0001", "u": 345, "v": 548})
            self.assertFalse(failed.is_error)
            self.assertTrue(any(isinstance(block, ImageContent) for block in failed.content))
            text = "\n".join(block.text for block in failed.content if isinstance(block, TextContent))
            self.assertIn("action_status=failed", text)
            self.assertIn("image_id=img_0002", text)
            self.assertIn("0..719", text)
            self.assertNotIn("0..1000", text)


if __name__ == "__main__":
    unittest.main()
