from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from jsonschema import Draft202012Validator

from .errors import EmbodiedError, ToolPolicyError
from .profile import ToolProfile


TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ENDPOINT_RE = re.compile(r"^/api/v2/[a-z0-9_]+$")


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
            raise ToolPolicyError(
                f"Invalid arguments for {self.name}: {error.message}",
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
            input_schema = _input_schema(item, hidden=set(fixed) | {"session_id"})
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
                metadata=dict(item),
                fixed_arguments=fixed,
                input_schema=input_schema,
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
    if isinstance(explicit_type, str):
        value_type = explicit_type
    elif widget == "multi_uv_arm":
        value_type = "array"
    else:
        value_type = _WIDGET_TYPES.get(widget)

    schema: dict[str, Any] = {}
    if value_type:
        schema["type"] = value_type
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
        items = argument.get("items")
        schema["items"] = (
            dict(items)
            if isinstance(items, dict)
            else _multi_uv_arm_item_schema(argument)
        )

    description = _argument_description(argument)
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
