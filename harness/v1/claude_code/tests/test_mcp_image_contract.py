from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from typing import Any

from embodied_claude_code.config import Settings
from embodied_claude_code.coordinates import latest_image_grounding_reminder
from embodied_claude_code.media import MediaExtractor
from embodied_claude_code.server import (
    MODEL_STRUCTURED_MAX_TEXT_CHARS,
    create_mcp_server,
)
from embodied_claude_code.service import EmbodiedService
from fake_behavior import PNG_720


PNG_A = PNG_720
PNG_B = PNG_A + b"distinct-overlay"
LARGE_PNG_A = PNG_A + b"A" * (550_000 - len(PNG_A))
LARGE_PNG_B = PNG_A + b"B" * (550_000 - len(PNG_A))


def data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class FakeBehaviorClient:
    def __init__(self) -> None:
        self.memory_timeouts: list[float] = []
        self.track_called = False
        self.tools = [
            {
                "name": "capture_head_camera",
                "endpoint": "/api/v2/capture_head_camera",
                "args": [],
                "coordinate_system": "qwen3vl_relative_0_1000",
                "coordinate_range": {"u": [0, 1000], "v": [0, 1000]},
                "desc": "Capture head RGB.",
            },
            {
                "name": "measure_shoulder_distance",
                "endpoint": "/api/v2/measure_shoulder_distance",
                "args": [],
                "desc": "Measure shoulder distance.",
            },
            {
                "name": "adjust_chassis",
                "endpoint": "/api/v2/adjust_chassis",
                "args": [
                    {
                        "name": "forward",
                        "type": "number",
                        "required": False,
                        "default": 0,
                    }
                ],
                "desc": "Adjust the mobile base.",
            },
            {
                "name": "move_to_reach_point",
                "endpoint": "/api/v2/move_to_reach_point",
                "args": [
                    {
                        "name": "image_id",
                        "widget": "image",
                        "required": True,
                    },
                    {
                        "name": "u",
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "required": True,
                    },
                    {
                        "name": "v",
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "required": True,
                    },
                ],
                "desc": "Move the robot into reach of an image point.",
            },
            {
                "name": "track_object_distance",
                "endpoint": "/api/v2/track_object_distance",
                "args": [],
                "desc": "Track named image points and measure depth.",
            },
        ]

    def get_memory(self, *, timeout_s: float) -> Any:
        self.memory_timeouts.append(timeout_s)
        if self.track_called:
            depths = {
                "rim_upper": 0.66,
                "rim_mid": 0.69,
                "rim_lower": 0.77,
                "rim_outer": 0.68,
                "aperture": 0.91,
                "panel_ref": 0.67,
            }
            tracks = {
                name: {
                    "camera": "head",
                    "confidence": 0.9,
                    "depth_m": depth,
                    "observation_sequence": 186,
                    "source_image_id": "img-track-source",
                    "source_session_id": "image-contract",
                    "status": "observed",
                    "track_id": f"distance_{index:03d}",
                    "u": 420.0 + index,
                    "v": 510.0 + index,
                    "xyz_in_robot_base_coord_m": [depth, 0.01 * index, 0.4],
                }
                for index, (name, depth) in enumerate(depths.items(), 1)
            }
        else:
            tracks = {
                "held_object": {
                    "camera": "head",
                    "confidence": 0.91,
                    "depth_m": 0.58,
                    "observation_sequence": 44,
                    "source_image_id": "img-grounding",
                    "source_observation_sequence": 12,
                    "source_session_id": "image-contract",
                    "status": "observed",
                    "track_id": "distance_point_001",
                    "u": 505.0,
                    "v": 768.0,
                    "xyz_in_robot_base_coord_m": [0.61, -0.01, 0.13],
                }
            }
        return {
            "raw": {
                "tracked_object_distance_tracking": {
                    "camera": "head",
                    "coordinate_system": "qwen3vl_relative_0_1000",
                    "depth_unit": "m",
                    "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                    "xyz_in_robot_base_coord_frame": "current_robot_base",
                },
                "tracked_object_distances": tracks,
            }
        }

    def get_json(self, path: str) -> Any:
        if path == "/api/state":
            return {"task": "image-contract-test"}
        if path == "/api/v2/tools":
            return {"tool_version": "v2", "tools": self.tools}
        raise AssertionError(path)

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        if path == "/api/v2/capture_head_camera":
            return {
                "ok": True,
                "image_id": "img-contract",
                "image_width": 720,
                "image_height": 720,
                "plan_id": "plan-contract",
                "plan_arm": "left",
                "recommended_arm": "left",
                "rgb_main": data_url(LARGE_PNG_B),
                "rgb_overlay_path": data_url(LARGE_PNG_B),
                "rgb_path": data_url(LARGE_PNG_A),
                "diagnostic_payload": "D" * 20_000,
                "received_session_id": payload["session_id"],
            }
        if path == "/api/v2/measure_shoulder_distance":
            return {
                "ok": True,
                "distance_m": 0.42,
                "received_session_id": payload["session_id"],
            }
        if path == "/api/v2/adjust_chassis":
            return {
                "ok": True,
                "image_id": "img-action",
                "rgb_main": data_url(PNG_A),
                "requested": {
                    "forward_m": -0.05,
                    "translation_m": 0.0,
                    "spin_deg": 0.0,
                },
                "actual": {
                    "forward_m": -0.0460003,
                    "translation_m": 0.0,
                    "spin_deg": 0.0,
                },
                "linear_target_reached": True,
                "obstacle_limited": False,
                "robot": {
                    "frame": "local_command_odometry",
                    "motion_epoch": 437,
                    "base_pose": {
                        "pos": [-0.549994, 0.295578, 0.0],
                        "quat": [0.0, 0.0, 0.98, -0.19],
                        "yaw_deg": -158.0586,
                    },
                    "base_qvel": [0.0, 0.0, 0.0],
                    "eef_left": {
                        "frame": "local_command_odometry",
                        "pos": [-0.8, 0.0, 0.9],
                        "quat": [0.0, 0.0, 0.0, 1.0],
                    },
                    "gripper_left_qpos": [0.04, 0.04],
                    "arm_left_qpos": list(range(8)),
                },
                "memory": {"instruction": "must not reach the model"},
                "scene_graph": {"objects": ["must not reach the model"]},
                "diagnostic_payload": "D" * 20_000,
            }
        if path == "/api/v2/move_to_reach_point":
            return {
                "ok": True,
                "image_id": "img-reach",
                "rgb_main": data_url(PNG_A),
                "shoulder_distance_estimate": {
                    "left_m": 0.6840154658860365,
                    "right_m": 0.6872391287866355,
                    "minimum_m": 0.6840154658860365,
                    "threshold_m": 0.7,
                    "ok": True,
                },
            }
        if path == "/api/v2/track_object_distance":
            self.track_called = True
            broken_inline_image = (
                "data:image/png;base64," + ("A" * 1_048_576) + "=="
            )
            assert len(broken_inline_image) == 1_048_600
            return {
                "ok": True,
                "image_id": "img-track-result",
                "rgb_main": broken_inline_image,
                "replay_frames": [
                    {"frame": frame, "diagnostic": "R" * 1024}
                    for frame in range(186)
                ],
            }
        raise AssertionError(path)

    def get_media(self, reference: str, max_bytes: int) -> tuple[bytes, str]:
        raise AssertionError(reference)


class NoNetworkClient:
    def get_media(self, reference: str, max_bytes: int) -> tuple[bytes, str]:
        raise AssertionError(reference)


class ImageExtractorContractTests(unittest.TestCase):
    def make_extractor(self, **overrides: Any) -> MediaExtractor:
        options = {
            "max_bytes": 1024 * 1024,
            "max_images": 4,
        }
        options.update(overrides)
        return MediaExtractor(NoNetworkClient(), **options)

    def test_extractor_counts_candidates_and_selects_one_raw_image(self) -> None:
        selection = self.make_extractor().extract(
            {"rgb_main": data_url(PNG_B), "rgb_path": data_url(PNG_A)}
        )
        self.assertEqual(selection.warnings, [])
        self.assertEqual(selection.rest_candidate_count, 2)
        self.assertEqual(len(selection.selected), 1)
        self.assertEqual(selection.selected[0].label, "rgb_path")
        self.assertEqual(selection.selected[0].role, "raw_rgb")
        self.assertEqual(selection.selected[0].data, PNG_A)

    def test_extractor_output_is_independent_of_json_key_order(self) -> None:
        extractor = self.make_extractor()
        first = extractor.extract(
            {
                "rgb_main": data_url(PNG_B),
                "rgb_overlay_path": data_url(PNG_B),
                "rgb_path": data_url(PNG_A),
            },
            tool_name="move_chassis_to_floor_point",
        )
        second = extractor.extract(
            {
                "rgb_path": data_url(PNG_A),
                "rgb_overlay_path": data_url(PNG_B),
                "rgb_main": data_url(PNG_B),
            },
            tool_name="move_chassis_to_floor_point",
        )
        self.assertEqual(
            [(item.label, item.role, item.sha256) for item in first.selected],
            [(item.label, item.role, item.sha256) for item in second.selected],
        )
        self.assertEqual(first.rest_candidate_count, 2)
        self.assertEqual(second.rest_candidate_count, 2)
        self.assertEqual(first.selected[0].label, "rgb_overlay_path")
        self.assertEqual(first.selected[0].role, "path_overlay")

    def test_selection_is_deterministic_for_representative_roles(self) -> None:
        raw_and_overlay = {
            "rgb_path": data_url(PNG_A),
            "rgb_overlay_path": data_url(PNG_B),
        }
        cases = (
            (
                "capture_head_camera",
                raw_and_overlay,
                "rgb_overlay_path",
                "path_overlay",
            ),
            (
                "set_arm_to_grasp_position",
                {**raw_and_overlay, "feed": "head"},
                "rgb_path",
                "raw_rgb",
            ),
            (
                "adjust_left_eef_pose_in_head_frame",
                {**raw_and_overlay, "feed": "head"},
                "rgb_path",
                "raw_rgb",
            ),
            (
                "capture_right_wrist_camera",
                {**raw_and_overlay, "feed": "right_wrist_camera"},
                "rgb_overlay_path",
                "grasp_volume_overlay",
            ),
            (
                "plan_grasp_point_filter_rgbd_lite",
                {
                    **raw_and_overlay,
                    "marked_image_url": data_url(PNG_B),
                },
                "marked_image_url",
                "planning_overlay",
            ),
            (
                "plan_press_point",
                {
                    "rgb_path": data_url(PNG_A),
                    "marked_image_url": data_url(PNG_B),
                },
                "marked_image_url",
                "planning_overlay",
            ),
            (
                "track_object_distance",
                {
                    "rgb_path": data_url(PNG_A),
                    "rgb_overlay_path": data_url(PNG_B),
                    "marked_image_url": data_url(PNG_A + b"stale-mark"),
                },
                "rgb_overlay_path",
                "path_overlay",
            ),
        )
        for tool_name, payload, expected_label, expected_role in cases:
            with self.subTest(tool_name=tool_name):
                selection = self.make_extractor().extract(
                    payload, tool_name=tool_name
                )
                self.assertEqual(len(selection.selected), 1)
                self.assertEqual(selection.selected[0].label, expected_label)
                self.assertEqual(selection.selected[0].role, expected_role)

    def test_actual_head_fallback_overrides_planner_and_wrist_tool_defaults(
        self,
    ) -> None:
        payload = {
            "ok": False,
            "feed": "head",
            "rgb_path": data_url(PNG_A),
            "rgb_overlay_path": data_url(PNG_B),
            "marked_image_url": data_url(PNG_A + b"stale-red"),
        }
        for tool_name in (
            "plan_grasp_point_filter_rgbd_lite",
            "plan_press_point",
            "exec_plan_pose",
        ):
            with self.subTest(tool_name=tool_name):
                selection = self.make_extractor().extract(
                    payload, tool_name=tool_name
                )
                self.assertEqual(selection.selected[0].label, "rgb_overlay_path")
                self.assertEqual(selection.selected[0].role, "path_overlay")

    @unittest.skipUnless(
        Path("/usr/bin/convert").is_file(), "ImageMagick convert is unavailable"
    )
    def test_oversized_image_is_transcoded_below_the_hard_limit(self) -> None:
        selection = self.make_extractor().extract(
            {"rgb_path": data_url(LARGE_PNG_A)},
            tool_name="capture_head_camera",
        )
        self.assertEqual(selection.warnings, [])
        self.assertEqual(len(selection.selected), 1)
        image = selection.selected[0]
        self.assertEqual(image.mime_type, "image/jpeg")
        self.assertLessEqual(len(image.data), 256 * 1024)
        self.assertTrue(image.data.startswith(b"\xff\xd8\xff"))

    @unittest.skipUnless(
        Path("/usr/bin/convert").is_file(), "ImageMagick convert is unavailable"
    )
    def test_transcoding_preserves_a_720px_source_frame(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/convert",
                "-size",
                "720x720",
                "gradient:#101820-#f2aa4c",
                "png:-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10,
        )
        padded_png = completed.stdout + b"P" * (300_000 - len(completed.stdout))
        selection = self.make_extractor(
            max_model_bytes=256 * 1024,
            jpeg_quality=95,
        ).extract(
            {"rgb_path": data_url(padded_png)},
            tool_name="capture_head_camera",
        )

        self.assertEqual(selection.warnings, [])
        self.assertEqual(len(selection.selected), 1)
        image = selection.selected[0]
        self.assertEqual((image.width, image.height), (720, 720))
        self.assertLessEqual(len(image.data), 256 * 1024)

    def test_converter_failure_omits_oversized_image_without_raw_fallback(
        self,
    ) -> None:
        selection = self.make_extractor(
            converter_path="/definitely/missing/convert"
        ).extract({"rgb_path": data_url(LARGE_PNG_A)})
        self.assertEqual(selection.rest_candidate_count, 1)
        self.assertEqual(selection.selected, [])
        self.assertEqual(len(selection.warnings), 1)
        self.assertIn("image omitted", selection.warnings[0])


@unittest.skipUnless(importlib.util.find_spec("mcp") is not None, "MCP SDK missing")
class MCPImageContractTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, record_root: Path) -> Any:
        from mcp.client import Client

        fake = FakeBehaviorClient()
        service = EmbodiedService(
            Settings(
                base_url="http://127.0.0.1:5011",
                http_timeout_s=2,
                record_root=record_root,
                session_id="image-contract",
                record=False,
            ),
            client=fake,
        )
        return Client(create_mcp_server(service), mode="legacy")

    def test_hard_cap_keeps_24_worst_case_results_below_10_mb(self) -> None:
        encoded = base64.b64encode(b"X" * (256 * 1024)).decode("ascii")
        result = {
            "content": [
                {"type": "text", "text": "T" * 8192},
                {
                    "type": "image",
                    "data": encoded,
                    "mimeType": "image/jpeg",
                },
            ]
        }
        serialized_result = json.dumps(result, separators=(",", ":"))
        serialized_history = "[" + ",".join([serialized_result] * 24) + "]"

        self.assertLess(len(serialized_result.encode("utf-8")), 1_048_576)
        self.assertLess(len(serialized_history.encode("utf-8")), 10_000_000)

    async def test_one_native_image_has_an_adjacent_machine_visible_caption(
        self,
    ) -> None:
        from mcp.types import ImageContent, TextContent

        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                result = await asyncio.wait_for(
                    client.call_tool("capture_head_camera", {}), timeout=5
                )
        image_positions = [
            index
            for index, block in enumerate(result.content)
            if isinstance(block, ImageContent)
        ]
        self.assertEqual(len(image_positions), 1)
        for position in image_positions:
            self.assertGreater(position, 0)
            caption = result.content[position - 1]
            self.assertIsInstance(caption, TextContent)
            self.assertRegex(caption.text, r"^image label=rgb_overlay_path ")
            self.assertIn("image_id=img-contract", caption.text)
            self.assertIn(
                "coordinate_system=image_pixels_720x720",
                caption.text,
            )
            self.assertIn("coordinate_canvas=720x720", caption.text)
            self.assertIn("origin=top_left", caption.text)
            self.assertIn("bottom_right=719,719", caption.text)
            self.assertIn(
                "coordinate_values=original_image_pixels_0_719", caption.text
            )
            self.assertIn("width=720", caption.text)
            self.assertIn("height=720", caption.text)
            self.assertIn("clickable=true", caption.text)
            self.assertLess(len(caption.text), 512)
        captions = [result.content[position - 1].text for position in image_positions]
        self.assertIn("role=path_overlay", captions[0])
        self.assertIn("usage=read_blue_path_and_underlying_scene", captions[0])

    async def test_image_response_uses_native_content_without_structured_content(
        self,
    ) -> None:
        from mcp.types import ImageContent, TextContent

        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                result = await asyncio.wait_for(
                    client.call_tool("capture_head_camera", {}), timeout=5
                )
        text = "\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        images = [
            block for block in result.content if isinstance(block, ImageContent)
        ]
        # Bound response data independently of the mandatory grounding prompt.
        reminder = latest_image_grounding_reminder("img-contract")
        self.assertEqual(text.count(reminder), 1)
        self.assertLess(len(text) - len(reminder), 2048)
        self.assertNotIn("D" * 1000, text)
        self.assertIsNone(result.structured_content)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mime_type, "image/jpeg")
        self.assertLessEqual(len(base64.b64decode(images[0].data)), 256 * 1024)
        self.assertFalse(
            any(
                block.text.startswith('{"content":')
                for block in result.content
                if isinstance(block, TextContent)
            )
        )
        self.assertIn('action_evidence={"image_id":"img-contract"', text)
        self.assertIn('"plan_id":"plan-contract"', text)
        self.assertIn('"plan_arm":"left"', text)
        self.assertIn('"recommended_arm":"left"', text)
        tracking_blocks = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
            and block.text.startswith("persistent_tracking=")
        ]
        self.assertEqual(len(tracking_blocks), 1)
        tracking = json.loads(
            tracking_blocks[0].removeprefix("persistent_tracking=")
        )
        self.assertEqual(
            tracking["tracks"]["held_object"]["xyz_in_robot_base_coord_m"],
            [0.61, -0.01, 0.13],
        )
        self.assertEqual(
            tracking["coordinate_system"],
            "image_pixels_720x720",
        )
        self.assertEqual(tracking["depth_unit"], "m")
        self.assertEqual(
            tracking["xyz_in_robot_base_coord_axes"], "x_forward_y_left_z_up"
        )
        self.assertEqual(
            tracking["xyz_in_robot_base_coord_frame"], "current_robot_base"
        )
        self.assertEqual(
            tracking["tracks"]["held_object"]["source_session_id"],
            "image-contract",
        )
        self.assertEqual(
            tracking["tracks"]["held_object"]["track_id"],
            "distance_point_001",
        )
        summary = next(
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        )
        self.assertIn("rest_media_candidates=2", summary)
        self.assertIn("selected_media=1", summary)
        self.assertIn("emitted_images=1", summary)

        serialized = json.dumps(result.model_dump(by_alias=True), sort_keys=True)
        self.assertLess(len(serialized.encode("utf-8")), 1_048_576)
        self.assertNotIn(base64.b64encode(LARGE_PNG_A).decode("ascii"), serialized)
        self.assertNotIn(base64.b64encode(LARGE_PNG_B).decode("ascii"), serialized)

        history = "[" + ",".join([serialized] * 24) + "]"
        self.assertLess(len(history.encode("utf-8")), 10_000_000)

    async def test_media_action_exposes_compact_telemetry_and_native_image(
        self,
    ) -> None:
        from mcp.types import ImageContent, TextContent

        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                result = await asyncio.wait_for(
                    client.call_tool("adjust_chassis", {"forward": -0.05}),
                    timeout=5,
                )

        images = [
            block for block in result.content if isinstance(block, ImageContent)
        ]
        self.assertEqual(len(images), 1)
        self.assertEqual(base64.b64decode(images[0].data), PNG_A)
        self.assertIsNone(result.structured_content)

        evidence_blocks = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
            and block.text.startswith("action_evidence=")
        ]
        self.assertEqual(len(evidence_blocks), 1)
        self.assertLessEqual(len(evidence_blocks[0]), 4096 + len("action_evidence="))
        evidence = json.loads(evidence_blocks[0].removeprefix("action_evidence="))
        self.assertEqual(evidence["tool_name"], "adjust_chassis")
        self.assertEqual(evidence["requested"]["forward_m"], -0.05)
        self.assertEqual(evidence["actual"]["forward_m"], -0.0460003)
        self.assertTrue(evidence["linear_target_reached"])
        self.assertFalse(evidence["obstacle_limited"])
        self.assertEqual(
            evidence["robot"]["base_pose"]["pos"],
            [-0.549994, 0.295578, 0.0],
        )
        serialized = evidence_blocks[0]
        for forbidden_key in (
            "memory",
            "scene_graph",
            "diagnostic_payload",
            "rgb_main",
            "arm_left_qpos",
            "quat",
        ):
            self.assertNotIn(f'"{forbidden_key}":', serialized)

    async def test_move_to_reach_exposes_shoulder_distances_with_image(self) -> None:
        from mcp.types import TextContent

        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                await client.call_tool("capture_head_camera", {})
                result = await asyncio.wait_for(
                    client.call_tool(
                        "move_to_reach_point",
                        {"image_id": "img-contract", "u": 495, "v": 438},
                    ),
                    timeout=5,
                )

        self.assertIsNone(result.structured_content)
        evidence_text = next(
            block.text
            for block in result.content
            if isinstance(block, TextContent)
            and block.text.startswith("action_evidence=")
        )
        evidence = json.loads(evidence_text.removeprefix("action_evidence="))
        shoulder = evidence["shoulder_distance_estimate"]
        self.assertEqual(shoulder["left_m"], 0.6840154658860365)
        self.assertEqual(shoulder["right_m"], 0.6872391287866355)
        self.assertEqual(shoulder["minimum_m"], 0.6840154658860365)
        self.assertEqual(shoulder["threshold_m"], 0.7)
        self.assertTrue(shoulder["ok"])

    async def test_non_image_response_keeps_structured_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                result = await asyncio.wait_for(
                    client.call_tool("measure_shoulder_distance", {}), timeout=5
                )
        self.assertIsNotNone(result.structured_content)
        response = result.structured_content["response"]
        self.assertEqual(response["distance_m"], 0.42)
        self.assertEqual(
            result.structured_content["persistent_tracking"]["tracks"][
                "held_object"
            ]["observation_sequence"],
            44,
        )

    async def test_broken_track_image_cannot_fall_back_to_raw_replay(self) -> None:
        from mcp.types import ImageContent

        with tempfile.TemporaryDirectory() as directory:
            async with self.make_client(Path(directory)) as client:
                result = await asyncio.wait_for(
                    client.call_tool("track_object_distance", {}), timeout=5
                )

        self.assertFalse(result.is_error)
        self.assertFalse(
            any(isinstance(block, ImageContent) for block in result.content)
        )
        self.assertIsNotNone(result.structured_content)
        structured = result.structured_content
        self.assertTrue(structured["response"]["ok"])
        tracks = structured["persistent_tracking"]["tracks"]
        self.assertEqual(
            {name: track["depth_m"] for name, track in tracks.items()},
            {
                "aperture": 0.91,
                "panel_ref": 0.67,
                "rim_lower": 0.77,
                "rim_mid": 0.69,
                "rim_outer": 0.68,
                "rim_upper": 0.66,
            },
        )

        structured_json = json.dumps(
            structured, ensure_ascii=True, separators=(",", ":")
        )
        self.assertLessEqual(
            len(structured_json), MODEL_STRUCTURED_MAX_TEXT_CHARS
        )
        serialized = json.dumps(result.model_dump(by_alias=True), sort_keys=True)
        self.assertLess(len(serialized.encode("utf-8")), 16 * 1024)
        self.assertNotIn("replay_frames", serialized)
        self.assertNotIn("A" * 1024, serialized)
        history = "[" + ",".join([serialized] * 24) + "]"
        self.assertLess(len(history.encode("utf-8")), 400_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
