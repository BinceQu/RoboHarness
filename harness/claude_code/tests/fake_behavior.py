from __future__ import annotations

import base64
import struct
import zlib
from typing import Any


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def png_image(width: int = 720, height: int = 720) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\x80\x90\xa0" * width) * height))
            + chunk(b"IEND", b""))


PNG_720 = png_image()


class FakeBehaviorClient:
    """In-memory RestClient test double; it never opens a socket."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.memory_timeouts: list[float] = []
        self.memory: dict[str, Any] = {
            "raw": {
                "tracked_object_distance_tracking": {
                    "camera": "head",
                    "coordinate_system": "qwen3vl_relative_0_1000",
                    "depth_unit": "m",
                    "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                    "xyz_in_robot_base_coord_frame": "current_robot_base",
                },
                "tracked_object_distances": {},
            }
        }
        self.state = {
            "task": "unit_test_task",
            "scene": "unit_test_scene",
            "goals": {"complete": False},
        }
        self.tools = [
            {
                "name": "capture_head_camera",
                "endpoint": "/api/v2/capture_head_camera",
                "args": [],
                "desc": "Capture the head RGB camera.",
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
                    },
                    {
                        "name": "translation",
                        "type": "number",
                        "required": False,
                        "default": 0,
                    },
                    {
                        "name": "spin",
                        "type": "number",
                        "required": False,
                        "default": 0,
                    },
                ],
                "desc": "Adjust the mobile base.",
            },
            {
                "name": "plan_grasp_point_filter",
                "endpoint": "/api/v2/plan",
                "mode": "grasp_point_filter",
                "args": [],
                "desc": "Legacy grasp-point planner.",
            },
            {
                "name": "plan_grasp_point_filter_rgbd",
                "endpoint": "/api/v2/plan",
                "mode": "grasp_point_filter_rgbd",
                "args": [],
                "desc": "Full RGBD grasp-point planner.",
            },
            {
                "name": "read_depth",
                "endpoint": "/api/v2/read_depth",
                "args": [],
                "desc": "Read depth at an image point.",
            },
            {
                "name": "move_point_to_point",
                "endpoint": "/api/v2/move_point_to_point",
                "args": [],
                "desc": "Move the end-effector between two image points.",
            },
            {
                "name": "plan_eef_translation_to_uvd_point",
                "endpoint": "/api/v2/plan_eef_translation_to_uvd_point",
                "args": [],
                "desc": "Plan end-effector translation to a UVD point.",
            },
            {
                "name": "adjust_plan_pose",
                "endpoint": "/api/v2/adjust_plan_pose",
                "args": [],
                "desc": "Adjust an existing move plan.",
            },
            {
                "name": "plan_press_point",
                "endpoint": "/api/v2/plan_press_point",
                "args": [],
                "desc": "Plan a press at an image point.",
            },
            {
                "name": "cut_object",
                "endpoint": "/api/v2/cut_object",
                "args": [],
                "desc": "Cut along two image points.",
            },
            {
                "name": "plan_grasp_point_filter_rgbd_lite",
                "endpoint": "/api/v2/plan",
                "mode": "grasp_point_filter_rgbd_lite",
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
                    {
                        "name": "plan_arm",
                        "widget": "select",
                        "options": ["any", "left", "right"],
                        "default": "any",
                        "required": False,
                    },
                ],
                "desc": "Plan a grasp from an RGB point.",
            },
        ]

    def get_json(self, path: str) -> Any:
        if path == "/api/state":
            return dict(self.state)
        if path == "/api/v2/tools":
            return {"tool_version": "v2", "tools": list(self.tools)}
        raise AssertionError(f"Unexpected GET path: {path}")

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        self.requests.append({"path": path, "body": dict(payload)})
        if path == "/api/v2/capture_head_camera":
            encoded = base64.b64encode(PNG_720).decode("ascii")
            return {
                "ok": True,
                "image_id": "img-1",
                "feed": "head",
                "rgb_main": f"data:image/png;base64,{encoded}",
            }
        if path in {"/api/v2/adjust_chassis", "/api/v2/plan"}:
            return {"ok": True, "received": dict(payload)}
        if path == "/api/v2/mark_on_map":
            return {
                "ok": True,
                "tool": "mark_on_map",
                "marked": payload["name"],
                "received": dict(payload),
            }
        raise AssertionError(f"Unexpected POST path: {path}")

    def get_memory(self, *, timeout_s: float) -> Any:
        self.memory_timeouts.append(timeout_s)
        return self.memory

    def get_media(self, reference: str, max_bytes: int) -> tuple[bytes, str]:
        raise AssertionError(f"Unexpected media URL: {reference}")
