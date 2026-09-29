"""Registry metadata owned entirely by the official_v2 tool package."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .contract import MOVE_TRACKED_POINT_ORDER_DESCRIPTION, MOVE_TRACKED_POINT_QUICK_COUNTS


_HEAD_ADJUST_TOOLS = frozenset(
    {
        "adjust_left_eef_pose_in_head_frame",
        "adjust_right_eef_pose_in_head_frame",
    }
)
_HEAD_ADJUST_PARAM_METADATA: dict[str, dict[str, Any]] = {
    "x": {
        "description": (
            "Robot-base-frame EEF translation delta in meters; +X is "
            "chassis-forward. Use short +X increments for chassis-horizontal "
            "pushing. Use only with y/z, never with "
            "forward/leftward/upward."
        ),
        "unit": "m",
        "coordinate_frame": "robot_base",
        "positive_direction": "chassis_forward",
        "exclusive_group": "robot_base_xyz",
        "mutually_exclusive_with": "head_camera_forward_leftward_upward",
    },
    "y": {
        "description": (
            "Robot-base-frame EEF translation delta in meters; +Y is "
            "chassis-left. It has the same positive-left sign as leftward, "
            "but y is base-fixed and is not rotated by the head camera pose. "
            "Use only with x/z."
        ),
        "unit": "m",
        "coordinate_frame": "robot_base",
        "positive_direction": "chassis_left",
        "exclusive_group": "robot_base_xyz",
        "mutually_exclusive_with": "head_camera_forward_leftward_upward",
    },
    "z": {
        "description": (
            "Robot-base-frame EEF translation delta in meters; +Z is "
            "chassis-up. Use only with x/y, never with "
            "forward/leftward/upward."
        ),
        "unit": "m",
        "coordinate_frame": "robot_base",
        "positive_direction": "chassis_up",
        "exclusive_group": "robot_base_xyz",
        "mutually_exclusive_with": "head_camera_forward_leftward_upward",
    },
    "forward": {
        "description": (
            "Starting-head-camera-frame translation delta in meters; positive "
            "moves along the camera viewing direction, including its vertical "
            "component when the camera is pitched. It is not chassis-horizontal "
            "forward. Use only with "
            "leftward/upward, never with x/y/z."
        ),
        "unit": "m",
        "coordinate_frame": "starting_head_camera",
        "positive_direction": "camera_forward",
        "exclusive_group": "head_camera_forward_leftward_upward",
        "mutually_exclusive_with": "robot_base_xyz",
    },
    "leftward": {
        "description": (
            "Starting-head-camera-frame translation delta in meters; positive "
            "moves toward camera-image left. It has the same positive-left sign "
            "as base y, but follows the camera orientation. Use only with "
            "forward/upward."
        ),
        "unit": "m",
        "coordinate_frame": "starting_head_camera",
        "positive_direction": "camera_left",
        "exclusive_group": "head_camera_forward_leftward_upward",
        "mutually_exclusive_with": "robot_base_xyz",
    },
    "upward": {
        "description": (
            "Starting-head-camera-frame translation delta in meters; positive "
            "moves toward camera-image up. Use only with forward/leftward, "
            "never with x/y/z."
        ),
        "unit": "m",
        "coordinate_frame": "starting_head_camera",
        "positive_direction": "camera_up",
        "exclusive_group": "head_camera_forward_leftward_upward",
        "mutually_exclusive_with": "robot_base_xyz",
    },
}


@dataclass
class ToolSpec:
    """Minimal Behavior Interface-compatible public tool specification."""

    name: str
    fn: Callable
    description: str = ""
    params: list[dict[str, Any]] = field(default_factory=list)


def tool_spec(name: str, fn: Callable, description: str) -> ToolSpec:
    params: list[dict[str, Any]] = []
    for index, (param_name, param) in enumerate(
        inspect.signature(fn).parameters.items()
    ):
        if index == 0:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # 对外目录必须用 JSON Schema 类型名，不能直接写 Python 注解（str/int）。
        _py_to_json = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        raw_annotation = (
            "any"
            if param.annotation is inspect.Signature.empty
            else getattr(param.annotation, "__name__", str(param.annotation))
        )
        annotation = _py_to_json.get(
            raw_annotation,
            raw_annotation
            if raw_annotation
            in {"string", "integer", "number", "boolean", "object", "array", "any"}
            else "any",
        )
        metadata = (
            _HEAD_ADJUST_PARAM_METADATA.get(param_name, {})
            if name in _HEAD_ADJUST_TOOLS
            else {}
        )
        if name == "move_tracked_point":
            quick_types = list(MOVE_TRACKED_POINT_QUICK_COUNTS)
            if param_name == "points":
                metadata = {
                    "description": MOVE_TRACKED_POINT_ORDER_DESCRIPTION,
                    "tool_description": description,
                    "quick_presets": [
                        {
                            "type": quick_type,
                            "on_hand_count": on[0] if len(on) == 1 else list(on),
                            "off_hand_count": off[0] if len(off) == 1 else list(off),
                        }
                        for quick_type, (on, off) in MOVE_TRACKED_POINT_QUICK_COUNTS.items()
                    ],
                }
            elif param_name == "quick_constraint":
                metadata = {"enum_types": quick_types, "description": MOVE_TRACKED_POINT_ORDER_DESCRIPTION}
            elif param_name == "quick_constraints":
                metadata = {"description": MOVE_TRACKED_POINT_ORDER_DESCRIPTION, "items": {
                    "type": "object",
                    "required": ["type", "on_hand_points", "off_hand_points"],
                    "properties": {
                        "type": {"type": "string", "enum": quick_types},
                        "on_hand_points": {"type": "array", "items": {"type": "string"}},
                        "off_hand_points": {"type": "array", "items": {"type": "string"}},
                        "axial_mode": {
                            "type": "string",
                            "enum": ["ordered", "ordered_containment", "segment_overlap", "line_only"],
                        },
                        "axial_point_order": {"type": "array", "items": {"type": "string"}},
                    },
                }}
            elif param_name == "inequalities":
                metadata = {
                    "description": (
                        "Strict affine variable inequalities solved jointly with "
                        "XYZ and geometry constraints; for example a-b > 0."
                    ),
                    "max_items": 6,
                    "items": {
                        "type": "object",
                        "required": ["lhs", "op", "rhs"],
                        "additionalProperties": False,
                        "properties": {
                            "lhs": {
                                "type": "string",
                                "description": "Variable or safe affine function.",
                            },
                            "op": {"type": "string", "enum": [">", "<"]},
                            "rhs": {"type": "number"},
                        },
                    },
                }
        elif name == "move_chassis_to_directly_facing_surface":
            if param_name == "session_id":
                metadata = {
                    "widget": "hidden",
                    "description": "Policy session identifier supplied by the caller.",
                }
            elif param_name == "image_id":
                metadata = {
                    "widget": "image",
                    "description": (
                        "Frozen head-camera RGB-D image returned by "
                        "capture_head_camera."
                    ),
                }
            elif param_name == "points":
                metadata = {
                    "type": "array",
                    "widget": "multi_uv_arm",
                    "min_points": 3,
                    "max_points": 3,
                    "arm_options": ["any"],
                    "description": (
                        "Exactly three non-collinear surface pixels in Qwen3-VL "
                        "relative image coordinates 0..1000. Point order does "
                        "not choose the normal sign; the frozen head-camera "
                        "viewing direction does. Final chassis yaw faces the "
                        "opposite horizontal projection of that normal."
                    ),
                    "items": {
                        "type": "object",
                        "required": ["u", "v"],
                        "additionalProperties": False,
                        "properties": {
                            "u": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1000,
                            },
                            "v": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1000,
                            },
                        },
                    },
                }
            elif param_name == "nav_timeout_s":
                metadata = {
                    "widget": "number",
                    "unit": "s",
                    "description": "Total timeout shared by translation and rotation.",
                }
            elif param_name == "pos_tol_m":
                metadata = {
                    "widget": "number",
                    "unit": "m",
                    "description": "Allowed base-centre XY arrival error.",
                }
        params.append(
            {
                "name": param_name,
                "type": annotation,
                "default": (
                    None
                    if param.default is inspect.Signature.empty
                    else param.default
                ),
                "required": param.default is inspect.Signature.empty,
                **metadata,
            }
        )
    return ToolSpec(
        name=str(name),
        fn=fn,
        description=str(description),
        params=params,
    )
