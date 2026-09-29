"""Official evaluator profile for the public Interface v2 tools."""

from .capabilities import (
    ALLOWED_OBSERVATIONS,
    PUBLIC_TOOLS,
    TOOL_CAPABILITIES,
    TOOL_VERSION,
    WRIST_ROLL_TEST_TOOL_ENABLED,
    OfficialToolBoundaryError,
    capability_report,
    validate_submission,
)
from .dispatch import translate_submission
from .contract import MOVE_TRACKED_POINT_QUICK_MAX_POINTS
from .tracked_point_constraints_local import (
    evaluate_quick_constraint,
    quick_constraint_relation,
)


def build_registry(adapter):
    from .registry import build_registry as _build_registry

    return _build_registry(adapter)


def install_profile(skills_module, adapter):
    from .registry import install_profile as _install_profile

    return _install_profile(skills_module, adapter)


def ensure_profile_installed(skills_module, registry):
    from .registry import ensure_profile_installed as _ensure_profile_installed

    return _ensure_profile_installed(skills_module, registry)


__all__ = [
    "ALLOWED_OBSERVATIONS",
    "PUBLIC_TOOLS",
    "TOOL_CAPABILITIES",
    "TOOL_VERSION",
    "WRIST_ROLL_TEST_TOOL_ENABLED",
    "OfficialToolBoundaryError",
    "build_registry",
    "capability_report",
    "ensure_profile_installed",
    "install_profile",
    "MOVE_TRACKED_POINT_QUICK_MAX_POINTS",
    "evaluate_quick_constraint",
    "quick_constraint_relation",
    "translate_submission",
    "validate_submission",
]
