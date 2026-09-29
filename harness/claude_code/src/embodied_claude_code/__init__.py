"""Embodied Claude Code plugin and BEHAVIOR MCP adapter."""
from __future__ import annotations

from typing import Any


__all__ = ["EmbodiedService", "Settings", "ToolResult"]
__version__ = "0.4.0"


def __getattr__(name: str) -> Any:
    """Keep the stdlib-only Qwen bridge usable before MCP deps are imported."""
    if name == "Settings":
        from .config import Settings

        return Settings
    if name in {"EmbodiedService", "ToolResult"}:
        from .service import EmbodiedService, ToolResult

        return {"EmbodiedService": EmbodiedService, "ToolResult": ToolResult}[name]
    raise AttributeError(name)
