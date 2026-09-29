"""Embodied Codex baseline MCP adapter."""

from .config import Settings
from .service import EmbodiedService, ToolResult

__all__ = ["EmbodiedService", "Settings", "ToolResult"]
__version__ = "0.2.1"
