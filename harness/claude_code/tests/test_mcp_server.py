from __future__ import annotations

import base64
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code.config import Settings
from embodied_claude_code.coordinates import (
    LATEST_IMAGE_REMINDER_PREFIX,
    latest_image_grounding_reminder,
)
from embodied_claude_code.errors import EmbodiedError, RemoteAPIError
from embodied_claude_code.profile import MCP_EXCLUDED_TOOLS
from embodied_claude_code.server import (
    MODEL_ERROR_MAX_TEXT_CHARS,
    PERSISTENT_TRACKING_MAX_TEXT_CHARS,
    _action_evidence_text,
    _nearby_object_warning_text,
    _model_error_payload,
    _persistent_tracking_text,
    create_mcp_server,
)
from embodied_claude_code.service import EmbodiedService, ToolResult

from fake_behavior import FakeBehaviorClient, PNG_1X1, PNG_720


MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


class ActionEvidenceTests(unittest.TestCase):
    def test_measurement_geometry_is_model_visible_without_media_paths(self) -> None:
        result = ToolResult(
            summary="measure_shoulder_distance completed.",
            data={
                "tool_name": "measure_shoulder_distance",
                "response": {
                    "ok": True,
                    "image_id": "img-0607",
                    "target_robot_m": [0.7673, -0.3948, 1.0861],
                    "right_shoulder_to_object_m": 0.5164,
                    "left_shoulder_to_object_m": 0.7323,
                    "point_observation": {
                        "depth_m": 0.3491,
                        "pixel_uv": [706, 266],
                        "relative_uv": [982, 370],
                        "source": "current_depth_and_cam_rel_pose",
                    },
                    "recommended_pitch": {
                        "pitch_delta_deg": -15.96,
                        "target_aligned_robot_m": [0.8629, 0.0, 1.0861],
                    },
                    "rgb_path": "/tmp/private/raw.png",
                    "depth_path": "/tmp/private/depth.npy",
                },
            },
        )

        text = _action_evidence_text(result)
        evidence = json.loads(text.removeprefix("action_evidence="))

        self.assertEqual(
            evidence["target_robot_m"], [0.7673, -0.3948, 1.0861]
        )
        self.assertEqual(evidence["right_shoulder_to_object_m"], 0.5164)
        self.assertEqual(evidence["point_observation"]["pixel_uv"], [706, 266])
        self.assertEqual(
            evidence["recommended_pitch"]["target_aligned_robot_m"],
            [0.8629, 0.0, 1.0861],
        )
        self.assertNotIn("rgb_path", evidence)
        self.assertNotIn("depth_path", evidence)

    def test_nearby_object_warning_is_model_visible(self) -> None:
        result = ToolResult(
            summary="capture_head_camera completed.",
            data={
                "tool_name": "capture_head_camera",
                "response": {
                    "ok": True,
                    "image_id": "img_0028",
                    "nearby_object_warning": (
                        "Some kind of object is on the chassis-forward path: nearest point 1 (360,770), distance 0.80m"
                    ),
                    "base_path_overlay": {"pixels": "P" * 20_000},
                },
            },
        )

        text = _action_evidence_text(result)
        evidence = json.loads(text.removeprefix("action_evidence="))
        self.assertEqual(
            evidence["nearby_object_warning"],
            "Some kind of object is on the chassis-forward path: nearest point 1 (360,770), distance 0.80m",
        )
        self.assertNotIn("base_path_overlay", evidence)
        self.assertNotIn("eef_near_0.1m", evidence)
        self.assertEqual(
            _nearby_object_warning_text(result),
            "距离机器人近的物体预警="
            "Some kind of object is on the chassis-forward path: nearest point 1 (360,770), distance 0.80m",
        )

    def test_nearby_object_warning_ignores_eef_field(self) -> None:
        result = ToolResult(
            summary="capture_head_camera completed.",
            data={
                "tool_name": "capture_head_camera",
                "response": {
                    "ok": True,
                    "image_id": "img_0028",
                    "chassis_forward_2m": (
                        "Some kind of object is on the chassis-forward path: nearest point 1 (360,770), distance 0.80m"
                    ),
                    "eef_near_0.1m": (
                        "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m"
                    ),
                },
            },
        )
        evidence = json.loads(
            _action_evidence_text(result).removeprefix("action_evidence=")
        )
        self.assertNotIn("eef_near_0.1m", evidence)
        self.assertEqual(
            _nearby_object_warning_text(result),
            "距离机器人近的物体预警="
            "Some kind of object is on the chassis-forward path: nearest point 1 (360,770), distance 0.80m",
        )

    def test_nearby_object_warning_omitted_when_corridor_clear(self) -> None:
        result = ToolResult(
            summary="capture_head_camera completed.",
            data={
                "tool_name": "capture_head_camera",
                "response": {
                    "ok": True,
                    "image_id": "img_0028",
                    "chassis_forward_2m": (
                        "No object on the chassis-forward path within 2m"
                    ),
                    "eef_near_0.1m": (
                        "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m"
                    ),
                    "nearby_object_warning": (
                        "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m"
                    ),
                },
            },
        )
        self.assertIsNone(_nearby_object_warning_text(result))

    def test_nearby_object_warning_line_is_omitted_when_absent(self) -> None:
        result = ToolResult(
            summary="capture_left_wrist_camera completed.",
            data={
                "tool_name": "capture_left_wrist_camera",
                "response": {"ok": True, "image_id": "img_w1"},
            },
        )
        self.assertIsNone(_nearby_object_warning_text(result))

    def test_persistent_tracking_media_text_is_bounded(self) -> None:
        result = ToolResult(
            summary="capture_head_camera completed.",
            data={
                "persistent_tracking": {
                    "available": True,
                    "tracks": {
                        f"track-{index}": {
                            "status": "observed-" + ("x" * 3000),
                            "xyz_in_robot_base_coord_m": [0.5, 0.0, 0.2],
                        }
                        for index in range(8)
                    },
                }
            },
        )

        text = _persistent_tracking_text(result)
        payload = text.removeprefix("persistent_tracking=")
        tracking = json.loads(payload)

        self.assertLessEqual(len(payload), PERSISTENT_TRACKING_MAX_TEXT_CHARS)
        self.assertTrue(tracking["model_text_tracks_truncated"])
        self.assertEqual(tracking["total_track_count"], 8)


class ModelErrorContractTests(unittest.TestCase):
    def test_error_hard_limit_survives_expansive_unicode(self) -> None:
        payload = _model_error_payload(
            EmbodiedError(
                "custom_error",
                "\U0001f600" * 10_000,
                details={"context": "\U0001f600" * 10_000},
            ),
            tool_name="custom_tool",
        )
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        self.assertLessEqual(len(serialized), MODEL_ERROR_MAX_TEXT_CHARS)

    def test_unstructured_remote_body_is_omitted_with_size_metadata(self) -> None:
        raw_response = json.dumps(
            {
                "base_path_overlay": {"pixels": "P" * 20_000},
                "rgb": "data:image/png;base64," + ("R" * 40_000),
                "memory": {"replay_frames": ["M" * 10_000]},
            }
        )
        error = RemoteAPIError(
            "BEHAVIOR API request failed.",
            status=504,
            response=raw_response,
        )

        payload = _model_error_payload(error, tool_name="reset_body")
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

        self.assertLessEqual(len(serialized), MODEL_ERROR_MAX_TEXT_CHARS)
        self.assertEqual(payload["error"]["tool_name"], "reset_body")
        self.assertEqual(payload["error"]["details"]["http_status"], 504)
        omitted = payload["error"]["details"]["response_omitted"]
        self.assertTrue(omitted["omitted"])
        self.assertEqual(omitted["size"], len(raw_response))
        for forbidden in (
            "base_path_overlay",
            "data:image/png;base64",
            "memory",
            "replay_frames",
            "R" * 1024,
        ):
            self.assertNotIn(forbidden, serialized)

    def test_structured_remote_error_keeps_only_bounded_action_evidence(self) -> None:
        response = {
            "ok": False,
            "error": "operation timed out",
            "tool": "adjust_left_eef_pose_in_head_frame",
            "action_steps": 28,
            "achieved_qpos": [0.45, -0.82, 0.0, 0.0],
            "base_path_overlay": {"pixels": "P" * 20_000},
            "rgb": "data:image/png;base64," + ("R" * 40_000),
            "memory": {"replay_frames": ["M" * 10_000]},
            "scene_graph": {"objects": ["S" * 10_000]},
            "diagnostic_payload": "D" * 10_000,
        }
        payload = _model_error_payload(
            RemoteAPIError("operation timed out", status=504, response=response),
            tool_name="adjust_left_eef_pose_in_head_frame",
        )
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

        self.assertLessEqual(len(serialized), MODEL_ERROR_MAX_TEXT_CHARS)
        details = payload["error"]["details"]
        self.assertTrue(details["response_sanitized"])
        self.assertEqual(details["response"]["action_steps"], 28)
        self.assertEqual(details["response"]["achieved_qpos"], [0.45, -0.82, 0.0, 0.0])
        self.assertEqual(details["response"]["tool"], response["tool"])
        for forbidden in (
            "base_path_overlay",
            "data:image/png;base64",
            "memory",
            "scene_graph",
            "diagnostic_payload",
            "R" * 1024,
        ):
            self.assertNotIn(forbidden, serialized)


@unittest.skipUnless(MCP_AVAILABLE, "MCP SDK is not installed")
class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        runtime = tempfile.TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        env = patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime.name})
        env.start()
        self.addCleanup(env.stop)
        monitor = patch("embodied_claude_code.skills.publish_loaded_skills", return_value=True)
        monitor.start()
        self.addCleanup(monitor.stop)

    async def test_latest_image_survives_action_failure_without_hiding_failure(self) -> None:
        from mcp.client import Client
        from mcp.types import ImageContent, TextContent

        class ImageFailureBehaviorClient(FakeBehaviorClient):
            def post_json(self, path: str, payload: dict) -> object:
                response = super().post_json(path, payload)
                if path == "/api/v2/capture_head_camera":
                    response["image_id"] = f"img_{len(self.requests):04d}"
                elif path == "/api/v2/adjust_chassis":
                    response = {"ok": False, "error": "obstacle_limited"}
                    if payload.get("forward") == 0.1:
                        encoded = base64.b64encode(PNG_720).decode("ascii")
                        response.update(
                            image_id=f"img_{len(self.requests):04d}",
                            rgb_main=f"data:image/png;base64,{encoded}",
                        )
                return response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=root,
                    record=True,
                    session_id="mcp-latest-image-test",
                    record_label="unit",
                ),
                client=ImageFailureBehaviorClient(),
            )
            client = Client(create_mcp_server(service), mode="legacy")
            await asyncio.wait_for(client.__aenter__(), timeout=5)
            try:
                for name, arguments, image_id, failed in (
                    ("capture_head_camera", {}, "img_0001", False),
                    ("adjust_chassis", {"forward": 0.1}, "img_0002", True),
                    ("capture_head_camera", {}, "img_0003", False),
                ):
                    with self.subTest(image_id=image_id):
                        result = await asyncio.wait_for(
                            client.call_tool(name, arguments), timeout=5
                        )
                        self.assertFalse(result.is_error)
                        self.assertIsNone(result.structured_content)
                        images = [
                            item for item in result.content
                            if isinstance(item, ImageContent)
                        ]
                        self.assertEqual(len(images), 1)
                        self.assertEqual(base64.b64decode(images[0].data), PNG_720)
                        self.assertIsInstance(result.content[-2], ImageContent)
                        self.assertEqual(
                            result.content[-1].text,
                            latest_image_grounding_reminder(image_id),
                        )
                        texts = [
                            item.text for item in result.content
                            if isinstance(item, TextContent)
                        ]
                        self.assertEqual(
                            sum(text.startswith(LATEST_IMAGE_REMINDER_PREFIX)
                                for text in texts),
                            1,
                        )
                        self.assertEqual("action_status=failed" in texts[0], failed)
                        evidence = json.loads(next(
                            text.removeprefix("action_evidence=")
                            for text in texts if text.startswith("action_evidence=")
                        ))
                        self.assertEqual(evidence["image_id"], image_id)
                        self.assertEqual(evidence["ok"], not failed)
                        if failed:
                            self.assertEqual(evidence["error"], "obstacle_limited")

                without_image = await asyncio.wait_for(
                    client.call_tool("adjust_chassis", {"forward": 0.2}), timeout=5
                )
                self.assertTrue(without_image.is_error)
                self.assertFalse(without_image.structured_content["response"]["ok"])
                self.assertFalse(any(
                    isinstance(item, ImageContent) for item in without_image.content
                ))
                self.assertNotIn(
                    LATEST_IMAGE_REMINDER_PREFIX,
                    json.dumps(without_image.model_dump(by_alias=True)),
                )
            finally:
                await asyncio.wait_for(client.__aexit__(None, None, None), timeout=5)

            run_dir = next(root.iterdir())
            turns = [
                json.loads(line)
                for line in (run_dir / "turns.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [turn["tool_result"]["is_error"] for turn in turns],
                [False, True, False, True],
            )
            failed_result = turns[1]["tool_result"]
            self.assertFalse(failed_result["data"]["response"]["ok"])
            self.assertEqual(failed_result["data"]["response"]["image_id"], "img_0002")
            self.assertEqual(len(failed_result["media"]), 1)

    async def test_remote_error_is_bounded_at_mcp_boundary_only(self) -> None:
        from mcp.client import Client

        raw_response = json.dumps(
            {
                "base_path_overlay": {"pixels": "P" * 20_000},
                "rgb": "data:image/png;base64," + ("R" * 40_000),
                "memory": {"replay_frames": ["M" * 10_000]},
            }
        )

        class FailingBehaviorClient(FakeBehaviorClient):
            def post_json(self, path: str, payload: dict) -> object:
                if path == "/api/v2/adjust_chassis":
                    raise RemoteAPIError(
                        "BEHAVIOR API request failed.",
                        status=504,
                        response=raw_response,
                    )
                return super().post_json(path, payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=root,
                    record=True,
                    session_id="mcp-error-test",
                    record_label="unit",
                ),
                client=FailingBehaviorClient(),
            )
            client = Client(create_mcp_server(service), mode="legacy")
            await asyncio.wait_for(client.__aenter__(), timeout=5)
            try:
                result = await asyncio.wait_for(
                    client.call_tool("adjust_chassis", {"forward": 0.1}), timeout=5
                )
            finally:
                await asyncio.wait_for(
                    client.__aexit__(None, None, None), timeout=5
                )

            self.assertTrue(result.is_error)
            model_result = json.dumps(
                result.model_dump(by_alias=True),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            self.assertLess(len(model_result), 10 * 1024)
            self.assertEqual(
                result.structured_content["error"]["details"]["http_status"],
                504,
            )
            self.assertEqual(
                result.structured_content["error"]["tool_name"], "adjust_chassis"
            )
            for forbidden in (
                "base_path_overlay",
                "data:image/png;base64",
                "memory",
                "replay_frames",
                "R" * 1024,
            ):
                self.assertNotIn(forbidden, model_result)

            run_dir = next(root.iterdir())
            recorded = (run_dir / "turns.jsonl").read_text(encoding="utf-8")
            self.assertIn("base_path_overlay", recorded)
            self.assertIn("replay_frames", recorded)
            self.assertIn("[inline image omitted: image/png]", recorded)

    async def test_catalog_is_mirrored_and_direct_calls_are_recorded(self) -> None:
        from mcp.client import Client
        from mcp.types import ImageContent, TextContent

        fake = FakeBehaviorClient()
        fake.tools.append(
            {
                "name": "measure_shoulder_distance",
                "endpoint": "/api/v2/measure_shoulder_distance",
                "args": [
                    {"name": "object_name", "widget": "text", "required": False},
                    {"name": "image_id", "widget": "image", "required": False},
                    {
                        "name": "u",
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "required": False,
                    },
                    {
                        "name": "v",
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "required": False,
                    },
                ],
                "one_of": [
                    {"fields": ["object_name"]},
                    {"fields": ["image_id", "u", "v"]},
                ],
                "desc": "Measure shoulder distance to a named or image target.",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=root,
                    record=True,
                    session_id="mcp-test",
                    record_label="unit",
                ),
                client=fake,
            )
            server = create_mcp_server(service)
            client = Client(server, mode="legacy")
            await asyncio.wait_for(client.__aenter__(), timeout=5)
            try:
                listed = await asyncio.wait_for(client.list_tools(), timeout=5)
                interface_tools = {tool["name"] for tool in fake.tools}
                mcp_tools = {tool.name for tool in listed.tools}
                self.assertTrue(
                    MCP_EXCLUDED_TOOLS.issubset(
                        interface_tools
                    )
                )
                self.assertEqual(
                    mcp_tools,
                    (interface_tools - MCP_EXCLUDED_TOOLS)
                    | {
                        "mark_on_map",
                        "move_chassis_to_directly_facing_surface",
                        "activate_skill",
                        "deactivate_skill",
                    },
                )
                self.assertTrue(MCP_EXCLUDED_TOOLS.isdisjoint(mcp_tools))
                self.assertIn("plan_grasp_point_filter_rgbd_lite", mcp_tools)
                self.assertIn("mark_on_map", mcp_tools)
                self.assertIn("move_chassis_to_directly_facing_surface", mcp_tools)
                self.assertIn("activate_skill", mcp_tools)

                schemas = {tool.name: tool.input_schema for tool in listed.tools}
                listed_by_name = {tool.name: tool for tool in listed.tools}
                self.assertIn(
                    "compact live `persistent_tracking` state",
                    listed_by_name["adjust_chassis"].description,
                )
                self.assertTrue(
                    listed_by_name["capture_head_camera"].annotations.read_only_hint
                )
                self.assertFalse(
                    listed_by_name["adjust_chassis"].annotations.read_only_hint
                )
                self.assertTrue(
                    listed_by_name["adjust_chassis"].annotations.destructive_hint
                )
                plan_schema = schemas["plan_grasp_point_filter_rgbd_lite"]
                self.assertEqual(
                    plan_schema["required"], ["image_id", "u", "v"]
                )
                self.assertEqual(
                    plan_schema["properties"]["plan_arm"]["enum"],
                    ["any", "left", "right"],
                )
                self.assertIn(
                    "original-image pixel",
                    plan_schema["properties"]["u"]["description"],
                )
                self.assertIn(
                    "0 is the top edge",
                    plan_schema["properties"]["v"]["description"],
                )
                self.assertIn(
                    "actual resolution must be 720 x 720",
                    listed_by_name[
                        "plan_grasp_point_filter_rgbd_lite"
                    ].description,
                )
                self.assertNotIn(
                    "actual resolution must be 720 x 720",
                    listed_by_name["adjust_chassis"].description,
                )
                self.assertNotIn("mode", plan_schema["properties"])
                self.assertNotIn("session_id", plan_schema["properties"])

                mark_schema = schemas["mark_on_map"]
                self.assertEqual(mark_schema["required"], ["name"])
                self.assertNotIn("anyOf", mark_schema)
                self.assertIn(
                    "`name` OR `name` + `image_id` + `u` + `v`",
                    mark_schema["description"],
                )
                self.assertIn(
                    "original-image pixel",
                    mark_schema["properties"]["u"]["description"],
                )
                marked = await asyncio.wait_for(
                    client.call_tool("mark_on_map", {"name": "target_box"}),
                    timeout=5,
                )
                self.assertFalse(marked.is_error)
                self.assertEqual(
                    fake.requests[-1],
                    {
                        "path": "/api/v2/mark_on_map",
                        "body": {"name": "target_box", "session_id": "mcp-test"},
                    },
                )
                captured = await asyncio.wait_for(
                    client.call_tool("capture_head_camera", {}), timeout=5
                )
                marked_image_point = await asyncio.wait_for(
                    client.call_tool(
                        "mark_on_map",
                        {
                            "name": "drawer_handle",
                            "image_id": "img-1",
                            "u": 425,
                            "v": 610,
                        },
                    ),
                    timeout=5,
                )
                self.assertFalse(marked_image_point.is_error)
                self.assertEqual(
                    fake.requests[-1],
                    {
                        "path": "/api/v2/mark_on_map",
                        "body": {
                            "name": "drawer_handle",
                            "image_id": "img-1",
                            "u": round(425 * 1000 / 719),
                            "v": round(610 * 1000 / 719),
                            "session_id": "mcp-test",
                        },
                    },
                )

                measure_schema = schemas["measure_shoulder_distance"]
                self.assertNotIn("anyOf", measure_schema)
                self.assertIn(
                    "`object_name` OR `image_id` + `u` + `v`",
                    measure_schema["description"],
                )
                self.assertIn(
                    "`object_name` OR `image_id` + `u` + `v`",
                    listed_by_name["measure_shoulder_distance"].description,
                )
                invalid_measure = await asyncio.wait_for(
                    client.call_tool("measure_shoulder_distance", {}), timeout=5
                )
                self.assertTrue(invalid_measure.is_error)

                self.assertFalse(captured.is_error)
                images = [
                    content
                    for content in captured.content
                    if isinstance(content, ImageContent)
                ]
                self.assertEqual(len(images), 1)
                self.assertEqual(base64.b64decode(images[0].data), PNG_720)
                texts = [
                    content.text
                    for content in captured.content
                    if isinstance(content, TextContent)
                ]
                self.assertEqual(len(texts), 6)
                self.assertTrue(any(t.startswith('rollout_budget=') for t in texts))
                self.assertEqual(texts[-1], latest_image_grounding_reminder("img-1"))
                self.assertNotIn("CallToolResult", "\n".join(texts))
                self.assertNotIn(
                    base64.b64encode(PNG_1X1).decode("ascii"),
                    "\n".join(texts),
                )
                evidence = next(
                    text for text in texts if text.startswith("action_evidence=")
                )
                self.assertIn('"image_id":"img-1"', evidence)
                tracking = next(
                    text for text in texts if text.startswith("persistent_tracking=")
                )
                self.assertIn('"available":true', tracking)
                caption = next(text for text in texts if text.startswith("image "))
                self.assertRegex(caption, r"^image label=rgb_main ")
                self.assertIn("role=path_overlay", caption)
                self.assertIn("image_id=img-1", caption)

                request_count = len(fake.requests)
                unexpected = await asyncio.wait_for(
                    client.call_tool(
                        "capture_head_camera", {"unexpected": "value"}
                    ),
                    timeout=5,
                )
                self.assertTrue(unexpected.is_error)
                self.assertEqual(len(fake.requests), request_count)

                planned = await asyncio.wait_for(
                    client.call_tool(
                        "plan_grasp_point_filter_rgbd_lite",
                        {"image_id": "img-1", "u": 400, "v": 500},
                    ),
                    timeout=5,
                )
                self.assertFalse(planned.is_error)
                self.assertEqual(
                    fake.requests[-1]["body"]["mode"],
                    "grasp_point_filter_rgbd_lite",
                )
                self.assertEqual(
                    fake.requests[-1]["body"]["session_id"], "mcp-test"
                )

                invalid = await asyncio.wait_for(
                    client.call_tool(
                        "plan_grasp_point_filter_rgbd_lite",
                        {"image_id": "img-1", "u": 1001, "v": 500},
                    ),
                    timeout=5,
                )
                self.assertTrue(invalid.is_error)

                listed_skills = await asyncio.wait_for(
                    client.call_tool("activate_skill", {}), timeout=5
                )
                self.assertFalse(listed_skills.is_error)
                listed_payload = listed_skills.structured_content
                self.assertEqual(listed_payload["mode"], "list")
                self.assertIn(
                    "pick-up-object",
                    [item["name"] for item in listed_payload["skills"]],
                )

                loaded = await asyncio.wait_for(
                    client.call_tool(
                        "activate_skill", {"name": "pick-up-object"}
                    ),
                    timeout=5,
                )
                self.assertFalse(loaded.is_error)
                loaded_text = "\n".join(
                    content.text
                    for content in loaded.content
                    if isinstance(content, TextContent)
                )
                self.assertIn('<activated_skill name="pick-up-object">', loaded_text)
                self.assertIn("set_arm_to_grasp_position", loaded_text)
                before_requests = list(fake.requests)
                exited = await client.call_tool("deactivate_skill", {"name": "pick-up-object"})
                self.assertFalse(exited.is_error)
                self.assertIsNone(exited.structured_content["active_skill"])
                self.assertEqual(exited.structured_content["active_skills"], ["behavior-v2-baseline"])
                self.assertEqual(fake.requests, before_requests, "deactivation must not actuate the robot")
                inactive = await client.call_tool("activate_skill", {})
                self.assertIsNone(inactive.structured_content["active_skill"])
                bad = await client.call_tool("deactivate_skill", {"name": "pick-up-object", "session_id": "other"})
                self.assertTrue(bad.is_error)
                missing = await client.call_tool("deactivate_skill", {})
                self.assertTrue(missing.is_error)
            finally:
                await asyncio.wait_for(
                    client.__aexit__(None, None, None), timeout=5
                )

            run_dirs = list(root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            manifest = json.loads(
                (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
            )
            turns = [
                json.loads(line)
                for line in (run_dirs[0] / "turns.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["turn_count"], 6)
            self.assertEqual(
                [turn["tool_call"]["name"] for turn in turns],
                [
                    "mark_on_map",
                    "capture_head_camera",
                    "mark_on_map",
                    "measure_shoulder_distance",
                    "plan_grasp_point_filter_rgbd_lite",
                    "plan_grasp_point_filter_rgbd_lite",
                ],
            )


if __name__ == "__main__":
    unittest.main()
