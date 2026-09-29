from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import re
from typing import Any

from jsonschema import Draft202012Validator

from .errors import EmbodiedError, ToolPolicyError
from .coordinates import (
    VLM_IMAGE_COORDINATE_SYSTEM, pixel_coordinate_text, schema_uses_image_coordinates,
    with_image_coordinate_descriptions,
)
from .profile import ToolProfile


TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ENDPOINT_RE = re.compile(r"^/api/v2/[a-z0-9_]+$")


# Some production tool profiles omit the spatial-map entries from
# /api/v2/tools even though the stable endpoint remains part of the interface.
# Keep this one Skill-required contract available without changing the server.
MARK_ON_MAP_FALLBACK = {
    "name": "mark_on_map",
    "endpoint": "/api/v2/mark_on_map",
    "args": [
        {
            "name": "name",
            "widget": "text",
            "required": True,
            "placeholder": "A short place name, such as target_box.",
        },
        {
            "name": "image_id",
            "widget": "image",
            "required": False,
            "placeholder": "A capture_head_camera image; omit to mark the robot position.",
        },
        {
            "name": "u",
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "required": False,
            "unit": "Qwen3-VL relative coordinate 0..1000",
        },
        {
            "name": "v",
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "required": False,
            "unit": "Qwen3-VL relative coordinate 0..1000",
        },
    ],
    "one_of": [
        {"fields": ["name"], "label": "Mark the robot's current position."},
        {
            "fields": ["name", "image_id", "u", "v"],
            "label": "Mark a point selected in a head-camera image.",
        },
    ],
    "pick_like": "mark_object",
    "desc": (
        "Mark a named position on the persistent minimap. Supply only `name` "
        "to mark the robot's current position, or supply `name`, `image_id`, "
        "`u`, and `v` to project a point from a capture_head_camera image. "
        "The result reports every named place relative to the current chassis."
    ),
}


# 接口目录有时还没登这个新底盘对准工具；MCP 侧补上，避免模型看不见。
SURFACE_FACING_FALLBACK = {
    "name": "move_chassis_to_directly_facing_surface",
    "endpoint": "/api/v2/move_chassis_to_directly_facing_surface",
    "args": [
        {
            "name": "image_id",
            "type": "string",
            "widget": "image",
            "required": True,
        },
        {
            "name": "points",
            "type": "array",
            "widget": "multi_uv_arm",
            "required": True,
            "min_points": 3,
            "max_points": 3,
            "arm_options": ["any"],
            "items": {
                "type": "object",
                "required": ["u", "v"],
                "additionalProperties": False,
                "properties": {
                    "u": {"type": "number", "minimum": 0, "maximum": 1000},
                    "v": {"type": "number", "minimum": 0, "maximum": 1000},
                },
            },
        },
        {
            "name": "nav_timeout_s",
            "type": "number",
            "widget": "number",
            "required": False,
            "default": 120.0,
            "unit": "s",
        },
        {
            "name": "pos_tol_m",
            "type": "number",
            "widget": "number",
            "required": False,
            "default": 0.04,
            "unit": "m",
        },
    ],
    "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
    "desc": (
        "在同一张冻结 head RGB-D 图上选择恰好三个不共线的表面点。"
        "工具反投影三点并取三角形中心；法向符号自动选择为与冻结 head "
        "相机视线成钝角的一侧。底盘中心先平移到中心沿该法向 0.8m 后投影"
        "到地面的 XY 点，再原地旋转，使机器人前向与该法向的水平投影平行"
        "且方向相反。"
        "三点顺序不会翻转最终法向；执行只使用 official action，并由后续 "
        "base_qvel proprioception 验证。"
    ),
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    endpoint: str
    metadata: dict[str, Any]
    fixed_arguments: dict[str, Any]
    input_schema: dict[str, Any]
    adapter_image_id: bool = False

    def public(self) -> dict[str, Any]:
        return dict(self.metadata)

    @property
    def description(self) -> str:
        return str(
            self.metadata.get("desc")
            or self.metadata.get("description")
            or f"Call BEHAVIOR v2 tool {self.name}."
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        errors = sorted(
            Draft202012Validator(self.input_schema).iter_errors(arguments),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.absolute_path)
            message = error.message
            if self.name == "move_tracked_point" and (
                path.startswith("points")
                or "additional propert" in message.lower()
            ):
                message += (
                    " Each points[] item allows only name, role, and "
                    "target_xyz_m. Put x/y/z inside target_xyz_m."
                )
            raise ToolPolicyError(
                f"Invalid arguments for {self.name}: {message}",
                details={"path": path, "schema": self.input_schema},
            )


@dataclass(frozen=True)
class ToolCatalog:
    tool_version: str
    tools: dict[str, ToolSpec]

    @classmethod
    def from_payload(
        cls, payload: Any, profile: ToolProfile
    ) -> "ToolCatalog":
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise EmbodiedError(
                "invalid_tool_catalog",
                "/api/v2/tools returned an invalid catalog.",
                details=payload,
            )
        catalog_items = list(payload["tools"])
        if not any(
            isinstance(item, dict) and item.get("name") == "mark_on_map"
            for item in catalog_items
        ):
            catalog_items.append(MARK_ON_MAP_FALLBACK)
        if not any(
            isinstance(item, dict)
            and item.get("name") == "move_chassis_to_directly_facing_surface"
            for item in catalog_items
        ):
            catalog_items.append(SURFACE_FACING_FALLBACK)
        tools: dict[str, ToolSpec] = {}
        for item in catalog_items:
            if not isinstance(item, dict):
                raise EmbodiedError(
                    "invalid_tool_catalog", "Tool catalog entries must be objects."
                )
            name = str(item.get("name") or "")
            endpoint = str(item.get("endpoint") or "")
            if not TOOL_NAME_RE.fullmatch(name) or not ENDPOINT_RE.fullmatch(endpoint):
                raise EmbodiedError(
                    "invalid_tool_catalog",
                    "Tool catalog contains an invalid name or endpoint.",
                    details={"name": name, "endpoint": endpoint},
                )
            if name in tools:
                raise EmbodiedError(
                    "invalid_tool_catalog", f"Duplicate tool in catalog: {name}"
                )
            if not profile.allows(name):
                continue
            fixed = dict(profile.fixed_arguments.get(name, {}))
            if "mode" in item:
                fixed["mode"] = item["mode"]
            wire_schema = _input_schema(item, hidden=set(fixed) | {"session_id"})
            input_schema = with_image_coordinate_descriptions(wire_schema)
            adapter_image_id = (
                schema_uses_image_coordinates(input_schema)
                and "image_id" not in input_schema["properties"]
            )
            if adapter_image_id:
                input_schema["properties"]["image_id"] = {
                    "type": "string",
                    "description": "The latest clickable 720 x 720 head image; required by the MCP pixel guard.",
                }
                input_schema.setdefault("required", []).append("image_id")
            try:
                Draft202012Validator.check_schema(input_schema)
            except Exception as exc:
                raise EmbodiedError(
                    "invalid_tool_catalog",
                    f"Tool {name!r} produced an invalid MCP input schema.",
                    details=str(exc),
                ) from exc
            tools[name] = ToolSpec(
                name=name,
                endpoint=endpoint,
                metadata=_pixel_metadata(item, input_schema),
                fixed_arguments=fixed,
                input_schema=input_schema,
                adapter_image_id=adapter_image_id,
            )
        return cls(tool_version=str(payload.get("tool_version") or ""), tools=tools)

    def get(self, name: str) -> ToolSpec:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise ToolPolicyError(
                f"Tool is not advertised or is blocked by the active profile: {name}",
                details={"available_tools": sorted(self.tools)},
            ) from exc

    def public(self) -> dict[str, Any]:
        return {
            "tool_version": self.tool_version,
            "tools": [self.tools[name].public() for name in sorted(self.tools)],
        }


def _pixel_metadata(item: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    metadata = deepcopy(item)
    if "coordinate_system" in metadata:
        metadata["coordinate_system"] = VLM_IMAGE_COORDINATE_SYSTEM
    if "coordinate_range" in metadata:
        metadata["coordinate_range"] = {"u": [0, 719], "v": [0, 719]}
    if not schema_uses_image_coordinates(schema):
        return metadata
    for key in ("desc", "description"):
        if isinstance(metadata.get(key), str):
            metadata[key] = pixel_coordinate_text(metadata[key])
    args = metadata.get("args", [])
    for arg in args:
        if not isinstance(arg, dict):
            continue
        field = schema["properties"].get(arg.get("name"), {})
        if arg.get("name") in {"u", "v", "image_id", "points"}:
            for key in ("unit", "placeholder", "coordinate_system"):
                arg.pop(key, None)
            arg.update(field)
    if "image_id" in schema["properties"] and not any(
        isinstance(arg, dict) and arg.get("name") == "image_id" for arg in args
    ):
        args.append({"name": "image_id", "required": True, **schema["properties"]["image_id"]})
    metadata["args"] = args
    return metadata


# 接口目录有时把 Python 注解名（str/int）直接写进 type，JSON Schema 不认。
_PYTHON_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "tuple": "array",
}


_JSON_SIMPLE_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)


def _json_schema_type(value_type: str | None) -> str | None:
    if not value_type or value_type == "any":
        return None
    mapped = _PYTHON_JSON_TYPES.get(value_type, value_type)
    if mapped in _JSON_SIMPLE_TYPES:
        return mapped
    # Optional/Union/泛型注解不能当 JSON Schema type，丢掉以免 MCP 起不来。
    return None


_WIDGET_TYPES = {
    "image": "string",
    "number": "number",
    "plan": "string",
    "select": "string",
    "text": "string",
    "uv": "integer",
}


def _input_schema(
    metadata: dict[str, Any], *, hidden: set[str]
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    args = metadata.get("args", [])
    if not isinstance(args, list):
        args = []
    for raw_arg in args:
        if isinstance(raw_arg, str):
            name = raw_arg.strip()
            if name and name not in hidden:
                properties[name] = {}
            continue
        if not isinstance(raw_arg, dict):
            continue
        name = str(raw_arg.get("name") or "").strip()
        if not TOOL_NAME_RE.fullmatch(name) or name in hidden:
            continue
        properties[name] = _argument_schema(raw_arg)
        if raw_arg.get("required") is True:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required

    alternatives = metadata.get("one_of")
    if isinstance(alternatives, list):
        any_of = []
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            fields = alternative.get("fields")
            if not isinstance(fields, list):
                continue
            names = [
                str(field)
                for field in fields
                if str(field) in properties
            ]
            if names:
                any_of.append({"required": names})
        if any_of:
            schema["anyOf"] = any_of
    return schema


def _argument_schema(argument: dict[str, Any]) -> dict[str, Any]:
    widget = str(argument.get("widget") or "")
    explicit_type = argument.get("type")
    items = argument.get("items") if isinstance(argument.get("items"), dict) else None
    # Official /api/v2/tools often labels complex arrays as type=any
    # (Python GenericAlias annotations). Keep items so MCP still shows
    # name/role/target_xyz_m instead of an untyped blob.
    if isinstance(explicit_type, str) and explicit_type not in {"", "any"}:
        value_type = explicit_type
    elif widget == "multi_uv_arm":
        value_type = "array"
    elif widget == "tracked_target_points" or items is not None:
        value_type = "array"
    else:
        value_type = _WIDGET_TYPES.get(widget)
    if (not value_type or value_type == "any") and items is not None:
        value_type = "array"

    schema: dict[str, Any] = {}
    json_type = _json_schema_type(value_type)
    if json_type:
        schema["type"] = json_type
        value_type = json_type
    options = argument.get("options")
    if isinstance(options, list):
        schema["enum"] = options
        if "type" not in schema and options:
            first = options[0]
            if isinstance(first, bool):
                schema["type"] = "boolean"
            elif isinstance(first, int):
                schema["type"] = "integer"
            elif isinstance(first, float):
                schema["type"] = "number"
            elif isinstance(first, str):
                schema["type"] = "string"
    if "default" in argument:
        schema["default"] = argument["default"]
    for source, target in (
        ("minimum", "minimum"),
        ("maximum", "maximum"),
        ("min_points", "minItems"),
        ("max_points", "maxItems"),
    ):
        if source in argument:
            schema[target] = argument[source]

    if value_type == "array":
        schema["items"] = (
            dict(items)
            if items is not None
            else _multi_uv_arm_item_schema(argument)
        )

    description = _argument_description(argument)
    if widget == "tracked_target_points":
        prefix = (
            "Each points[] item allows only name, role, and target_xyz_m. "
            "Put x/y/z inside target_xyz_m; do not put x/y/z on the point root."
        )
        description = f"{prefix} {description}".strip() if description else prefix
    if description:
        schema["description"] = description
    return schema


def _multi_uv_arm_item_schema(argument: dict[str, Any]) -> dict[str, Any]:
    arm_options = argument.get("arm_options")
    arm_schema: dict[str, Any] = {"type": "string"}
    if isinstance(arm_options, list) and arm_options:
        arm_schema["enum"] = arm_options
        arm_schema["default"] = arm_options[0]
    return {
        "type": "object",
        "properties": {
            "u": {"type": "integer", "minimum": 0, "maximum": 1000},
            "v": {"type": "integer", "minimum": 0, "maximum": 1000},
            "plan_arm": arm_schema,
        },
        "required": ["u", "v"],
        "additionalProperties": False,
    }


def _argument_description(argument: dict[str, Any]) -> str:
    parts = []
    for key in ("description", "unit", "placeholder", "coordinate_system"):
        value = argument.get(key)
        if value and str(value) not in parts:
            parts.append(str(value))
    return "; ".join(parts)
