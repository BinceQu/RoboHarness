from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ToolPolicyError


PROFILE_SCHEMA = "embodied_codex.profile.v1"
# 这些接口工具仍存在于 /api/v2/tools，但不注册进 MCP，模型不可见、不可调。
MCP_EXCLUDED_TOOLS = frozenset(
    {
        "plan_grasp_point_filter",
        "plan_grasp_point_filter_rgbd",
        "plan_eef_translation_to_uvd_point",
        "read_depth",
        "move_point_to_point",
    }
)


def _string_set(value: Any, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"Profile field {field_name} must be a string list.")
    return frozenset(item.strip() for item in value if item.strip())


@dataclass(frozen=True)
class ToolProfile:
    name: str = "baseline"
    description: str = "Allow advertised v2 tools except MCP exclusions."
    allow_tools: frozenset[str] = field(
        default_factory=lambda: frozenset({"*"})
    )
    deny_tools: frozenset[str] = field(
        default_factory=lambda: MCP_EXCLUDED_TOOLS
    )
    fixed_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def baseline(cls) -> "ToolProfile":
        return cls()

    @classmethod
    def load(cls, path: Path | None) -> "ToolProfile":
        if path is None:
            return cls.baseline()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(
                f"Unable to read profile: {path}", details=str(exc)
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Profile is not valid JSON: {path}", details=str(exc)
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("Profile root must be an object.")
        if payload.get("schema_version") != PROFILE_SCHEMA:
            raise ConfigurationError(
                f"Profile schema_version must be {PROFILE_SCHEMA}."
            )
        fixed = payload.get("fixed_arguments", {})
        if not isinstance(fixed, dict) or not all(
            isinstance(name, str) and isinstance(arguments, dict)
            for name, arguments in fixed.items()
        ):
            raise ConfigurationError(
                "Profile fixed_arguments must map tool names to objects."
            )
        return cls(
            name=str(payload.get("name") or "unnamed"),
            description=str(payload.get("description") or ""),
            allow_tools=_string_set(payload.get("allow_tools", ["*"]), "allow_tools"),
            deny_tools=(
                _string_set(payload.get("deny_tools", []), "deny_tools")
                | MCP_EXCLUDED_TOOLS
            ),
            fixed_arguments={str(name): dict(arguments) for name, arguments in fixed.items()},
        )

    def allows(self, tool_name: str) -> bool:
        return (
            tool_name not in MCP_EXCLUDED_TOOLS
            and tool_name not in self.deny_tools
            and ("*" in self.allow_tools or tool_name in self.allow_tools)
        )

    def apply_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(arguments)
        for key, value in self.fixed_arguments.get(tool_name, {}).items():
            if key in result and result[key] != value:
                raise ToolPolicyError(
                    f"Argument {key!r} is fixed by profile {self.name!r}.",
                    details={"tool_name": tool_name, "expected": value},
                )
            result[key] = value
        return result

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allow_tools": sorted(self.allow_tools),
            "deny_tools": sorted(self.deny_tools | MCP_EXCLUDED_TOOLS),
            "fixed_arguments": self.fixed_arguments,
        }
