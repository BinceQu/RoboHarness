"""Model-facing image-coordinate contract for BEHAVIOR tools."""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any

from .errors import ToolPolicyError


VLM_IMAGE_COORDINATE_SYSTEM = "image_pixels_720x720"
VLM_IMAGE_COORDINATE_CANVAS_SIZE = 720
PIXEL_MAX = VLM_IMAGE_COORDINATE_CANVAS_SIZE - 1
INTERFACE_COORDINATE_MAX = 1000
VLM_IMAGE_COORDINATE_CONTRACT = (
    "BEHAVIOR image-coordinate contract: the latest image in the conversation "
    "represents the current observed state; older images are historical context. "
    "Select points only in the latest "
    "clickable head-camera image, whose actual resolution must be 720 x 720. "
    "Send integer original-image pixel indices: u is the column, v is the row, "
    "top-left is (0,0), bottom-right is (719,719). Do not normalize, rescale, "
    "or use coordinates from a crop. Before EVERY point selection, inspect the "
    "latest attached head image, re-locate the physical target, and derive u/v "
    "from that exact image_id only. Never reuse locations from older images, "
    "earlier reasoning, reference photos, or tracking text. After motion, "
    "re-locate the target; do not shift, rotate, or scale an older point, "
    "bounding box, or object-part layout to guess a new click. Wrist images are "
    "observation-only and cannot be clicked. Numbers printed in overlays or "
    "path/obstacle telemetry are annotations, not target points."
)
VLM_IMAGE_COORDINATE_REMINDER = (
    "The latest image in the conversation represents the current observed state; "
    "older images are historical context. "
    "For a clickable 720 x 720 head image, read integer original-image pixels "
    "u,v in 0..719 directly. Do not normalize or rescale. Inspect the latest "
    "image again before EVERY selection; ignore overlay-label numbers and "
    "older target locations. After motion, re-locate the target; do not shift, "
    "rotate, or scale an older point, bounding box, or object-part layout to "
    "guess a new click. Wrist and reference images cannot be clicked."
)
LATEST_IMAGE_REMINDER_PREFIX = "Latest-image grounding requirement: "


def latest_image_grounding_reminder(image_id: str) -> str:
    if not image_id:
        return (
            LATEST_IMAGE_REMINDER_PREFIX
            + "This attached image has no usable image_id; obtain a fresh image "
            "with an image_id before selecting any point."
        )
    return (
        LATEST_IMAGE_REMINDER_PREFIX
        + "Before selecting any point, inspect the latest image actually attached "
        f"to this tool result (image_id={image_id}), re-locate the target in that "
        "image, and derive u/v from that image only; never copy coordinates or "
        "locations from older images or earlier reasoning. "
        "A newer image supersedes this observation. "
        + VLM_IMAGE_COORDINATE_REMINDER
    )


_U_DESCRIPTION = (
    "Integer original-image pixel column in the latest 720 x 720 head image: "
    "0 is the left edge and 719 is the right edge. Do not normalize or rescale."
)
_V_DESCRIPTION = (
    "Integer original-image pixel row in the latest 720 x 720 head image: "
    "0 is the top edge and 719 is the bottom edge. Do not normalize or rescale."
)
_IMAGE_ID_DESCRIPTION = (
    "The latest clickable head-camera image_id, actually 720 x 720; "
    "re-ground the point when image_id changes. Wrist images are not clickable."
)


def schema_uses_image_coordinates(schema: Any) -> bool:
    """Return whether a JSON schema contains a paired u/v point."""
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and {"u", "v"}.issubset(properties):
            return True
        return any(schema_uses_image_coordinates(value) for value in schema.values())
    if isinstance(schema, list):
        return any(schema_uses_image_coordinates(value) for value in schema)
    return False


def with_image_coordinate_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool schema and make every image-coordinate field unambiguous."""
    adapted = deepcopy(schema)
    if not schema_uses_image_coordinates(adapted):
        return adapted

    _annotate_uv_pairs(adapted)
    _annotate_named_property(adapted, "image_id", _IMAGE_ID_DESCRIPTION)
    return adapted


def _annotate_uv_pairs(schema: Any) -> None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and {"u", "v"}.issubset(properties):
            for axis, description in (("u", _U_DESCRIPTION), ("v", _V_DESCRIPTION)):
                # Replace upstream unit descriptions and constraints, not append
                # contradictory pixel instructions to a normalized schema.
                properties[axis] = {
                    "type": "integer", "minimum": 0, "maximum": PIXEL_MAX,
                    "description": description,
                }
        for value in schema.values():
            _annotate_uv_pairs(value)
    elif isinstance(schema, list):
        for value in schema:
            _annotate_uv_pairs(value)


def _annotate_named_property(schema: Any, name: str, description: str) -> None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and name in properties:
            properties[name]["description"] = description
        for value in schema.values():
            _annotate_named_property(value, name, description)
    elif isinstance(schema, list):
        for value in schema:
            _annotate_named_property(value, name, description)


def contains_uv(value: Any) -> bool:
    if isinstance(value, dict):
        return bool({"u", "v"} & value.keys()) or any(
            contains_uv(item) for item in value.values()
        )
    return isinstance(value, list) and any(contains_uv(item) for item in value)


def pixel_to_interface(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= PIXEL_MAX:
        raise ToolPolicyError("Image u/v must be integer original-image pixels in 0..719.")
    return round(value * INTERFACE_COORDINATE_MAX / PIXEL_MAX)


def interface_to_pixel(value: Any, size: int = 720) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= INTERFACE_COORDINATE_MAX):
        return None
    return round(value * (size - 1) / INTERFACE_COORDINATE_MAX)


def arguments_to_interface(value: Any) -> Any:
    """Convert flat and nested point objects once, preserving names/order/XYZ."""
    if isinstance(value, dict):
        out = {key: arguments_to_interface(item) for key, item in value.items()}
        if {"u", "v"} & value.keys():
            if not {"u", "v"}.issubset(value):
                raise ToolPolicyError("Every image point requires both u and v.")
            out.update(u=pixel_to_interface(value["u"]), v=pixel_to_interface(value["v"]))
        return out
    if isinstance(value, list):
        return [arguments_to_interface(item) for item in value]
    return value


def pixel_coordinate_text(text: str) -> str:
    """Translate the interface's unit labels without changing other numbers."""
    text = re.sub(r"(?:Qwen3-VL\s*)?(?:relative (?:image )?coordinates?|相对坐标)\s*0\s*\.\.\s*1000",
                  "original-image pixels 0..719", text, flags=re.IGNORECASE)
    text = re.sub(r"relative_image_coordinates_0_1000|qwen3vl_relative_0_1000",
                  VLM_IMAGE_COORDINATE_SYSTEM, text)
    return text


# Inverse of behavior_interface.coordinate_contract's public field aliases.
# Native diagnostic pixel_uv/hit_pixel and all XYZ/metric arrays stay unchanged.
_RELATIVE_PAIRS = frozenset({
    "uv", "input_uv", "pick_uv", "point_uv", "projected_uv",
    "projected_uv_clipped", "resolved_uv", "vlm_uv", "reproj_uv",
    "contact_uv", "grasp_point_uv", "corners_uv", "relative_uv", "qwen_uv",
})
_COORDINATE_TAGS = frozenset({"coordinate_system", "coord_system", "qwen_coord_system"})
_RANGE_KEYS = frozenset({"coordinate_range", "uv_range", "qwen_uv_range"})


def response_to_pixels(value: Any, *, width: int = 720, height: int = 720,
                       parent_key: str = "") -> Any:
    """Translate structured public image coordinates, never metric geometry."""
    if isinstance(value, dict):
        if any(value.get(key) == VLM_IMAGE_COORDINATE_SYSTEM for key in _COORDINATE_TAGS):
            return deepcopy(value)
        out = {}
        paired = {"u", "v"}.issubset(value)
        for key, item in value.items():
            if key in _COORDINATE_TAGS:
                is_image_unit = isinstance(item, str) and "1000" in item and (
                    "relative" in item.lower() or "qwen" in item.lower()
                )
                out[key] = ((VLM_IMAGE_COORDINATE_SYSTEM if (width, height) == (720, 720)
                             else "observation_only_image_pixels") if is_image_unit else item)
            elif key in _RANGE_KEYS:
                out[key] = {"u": [0, width - 1], "v": [0, height - 1]}
            elif (paired and key in {"u", "v"}) or key in {"center_u", "center_v", "qwen_u", "qwen_v"}:
                out[key] = interface_to_pixel(item, width if key.endswith("u") else height)
            else:
                out_key = "input_pixel_uv" if key == "relative_uv" else key
                out[out_key] = response_to_pixels(item, width=width, height=height, parent_key=key)
        return out
    if isinstance(value, (list, tuple)):
        count = 4 if parent_key == "uv_bbox" else 2 if parent_key in _RELATIVE_PAIRS else 0
        if count and len(value) >= count and all(
            item is None or isinstance(item, (int, float)) for item in value[:count]
        ):
            return [interface_to_pixel(item, width if i % 2 == 0 else height)
                    for i, item in enumerate(value[:count])] + list(value[count:])
        return [response_to_pixels(item, width=width, height=height, parent_key=parent_key)
                for item in value]
    if isinstance(value, str):
        if parent_key == "nearby_object_warning":
            # This upstream prose has no machine-readable coordinate unit.
            # Retain the safety distances without presenting guessed UV units.
            value = re.sub(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)",
                           "(image point omitted; not a target)", value)
        return pixel_coordinate_text(value)
    return value
