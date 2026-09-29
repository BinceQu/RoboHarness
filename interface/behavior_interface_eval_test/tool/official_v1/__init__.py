"""Evaluator-only tool profile for the isolated Behavior Interface."""

from .capabilities import (
    ALLOWED_OBSERVATIONS,
    PUBLIC_SKILLS,
    TOOL_CAPABILITIES,
    TOOL_VERSION,
    OfficialToolBoundaryError,
    capability_report,
    validate_submission,
)


def build_registry(adapter):
    from .registry import build_registry as _build_registry

    return _build_registry(adapter)


def install_profile(skills_module, adapter):
    from .registry import install_profile as _install_profile

    return _install_profile(skills_module, adapter)

__all__ = [
    "ALLOWED_OBSERVATIONS",
    "PUBLIC_SKILLS",
    "TOOL_CAPABILITIES",
    "TOOL_VERSION",
    "OfficialToolBoundaryError",
    "build_registry",
    "capability_report",
    "install_profile",
    "validate_submission",
]
