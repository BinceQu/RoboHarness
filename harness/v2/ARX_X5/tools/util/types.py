"""Registry metadata owned entirely by the official_v2 tool package."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


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
            "chassis-forward. Use only with y/z, never with "
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
            "moves along the camera viewing direction. Use only with "
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
        annotation = (
            "any"
            if param.annotation is inspect.Signature.empty
            else getattr(param.annotation, "__name__", str(param.annotation))
        )
        metadata = (
            _HEAD_ADJUST_PARAM_METADATA.get(param_name, {})
            if name in _HEAD_ADJUST_TOOLS
            else {}
        )
        params.append(
            {
                "name": param_name,
                "type": annotation,
                "default": (
                    None if param.default is inspect.Signature.empty else param.default
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
