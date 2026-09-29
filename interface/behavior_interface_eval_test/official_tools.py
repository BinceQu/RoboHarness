"""Compatibility exports for the isolated official_v1 tool profile.

New code should import ``behavior_interface_eval_test.tool.official_v1``.
This module remains so existing test scripts and external callers do not break.
"""

from __future__ import annotations

from behavior_interface_eval_test.tool.official_v1 import (
    TOOL_CAPABILITIES,
    OfficialToolBoundaryError,
    build_registry,
    capability_report,
    validate_submission,
)


StrictToolBoundaryError = OfficialToolBoundaryError


def install_strict_tool_registry(registry, adapter) -> None:
    """Replace a supplied registry with independent official_v1 specs."""
    registry.clear()
    registry.update(build_registry(adapter))


__all__ = [
    "StrictToolBoundaryError",
    "TOOL_CAPABILITIES",
    "capability_report",
    "install_strict_tool_registry",
    "validate_submission",
]
