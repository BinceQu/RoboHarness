"""Tool-version selection for the interface.

The implementation modules still live under ``behavior_interface.skills`` for
backwards compatibility with existing imports.  This package describes which
registered tools are visible for each interface profile.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Optional, Sequence


VALID_TOOL_VERSIONS = ("v0", "v1", "v1_shortcut", "v2", "v3")
DEFAULT_TOOL_VERSION = "v2"
ENV_TOOL_VERSION = "INTERFACE_TOOL_VERSION"


def active_tool_version() -> str:
    ver = os.environ.get(ENV_TOOL_VERSION, DEFAULT_TOOL_VERSION).strip().lower()
    return ver if ver in VALID_TOOL_VERSIONS else DEFAULT_TOOL_VERSION


def load_tool_profile(version: Optional[str] = None) -> ModuleType:
    ver = (version or active_tool_version()).strip().lower()
    if ver not in VALID_TOOL_VERSIONS:
        ver = DEFAULT_TOOL_VERSION
    return importlib.import_module(f"behavior_interface.tool.{ver}")


def active_v2_tool_names(version: Optional[str] = None) -> Optional[Sequence[str]]:
    profile = load_tool_profile(version)
    names = getattr(profile, "V2_TOOL_NAMES", None)
    if names is None:
        return None
    return tuple(str(x) for x in names)
