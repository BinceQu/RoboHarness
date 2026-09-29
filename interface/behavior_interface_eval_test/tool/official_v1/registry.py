"""Independent official_v1 registry; no legacy skill callable is reused."""

from __future__ import annotations

from typing import Any

from .capabilities import (
    PUBLIC_SKILLS,
    TOOL_CAPABILITIES,
    TOOL_VERSION,
    capability_report,
    validate_submission,
)
from .capture import capture_from_evaluator
from .motion import (
    blocked_tool,
    face_to_point,
    mesure_shoulder_distance,
    move_base_to_point,
    move_eef,
    move_in_robot_coord,
    reset_body,
    set_arm_to_grasp_position,
)
from .types import ToolSpec, tool_spec


_DESCRIPTIONS = {
    "capture": "Capture evaluator head RGB/depth without privileged modalities.",
    "capture_left_wrist_camera": "Capture evaluator left-wrist RGB/depth.",
    "capture_right_wrist_camera": "Capture evaluator right-wrist RGB/depth.",
    "move_in_robot_coord": "Action-only base/trunk motion in the robot frame.",
    "face_to_point": "Spin the base to center a relative head-image point.",
    "move_eef": "Gripper-only official action adapter.",
    "move_base_to_point": "Depth-only floor click followed by base spin and forward actions.",
    "mesure_shoulder_distance": "Measure static-model shoulder distance to a clicked depth point.",
    "set_arm_to_grasp_position": "Interpolate observed arm qpos to the fixed R1Pro grasp-prep target.",
    "reset_body": "Interpolate observed trunk qpos to an upright or requested pitch target.",
}


def build_registry(adapter) -> dict[str, ToolSpec]:
    def capture_head(ctx, session_id: str, **_kwargs):
        yield from capture_from_evaluator(
            ctx,
            adapter,
            session_id=session_id,
            role="head",
        )

    def capture_left(ctx, session_id: str, **_kwargs):
        yield from capture_from_evaluator(
            ctx,
            adapter,
            session_id=session_id,
            role="left_wrist",
        )

    def capture_right(ctx, session_id: str, **_kwargs):
        yield from capture_from_evaluator(
            ctx,
            adapter,
            session_id=session_id,
            role="right_wrist",
        )

    functions: dict[str, Any] = {
        "capture": capture_head,
        "capture_left_wrist_camera": capture_left,
        "capture_right_wrist_camera": capture_right,
        "move_in_robot_coord": move_in_robot_coord,
        "face_to_point": face_to_point,
        "move_eef": move_eef,
        "move_base_to_point": move_base_to_point,
        "mesure_shoulder_distance": mesure_shoulder_distance,
        "set_arm_to_grasp_position": set_arm_to_grasp_position,
        "reset_body": reset_body,
    }
    registry: dict[str, ToolSpec] = {}
    for name in PUBLIC_SKILLS:
        fn = functions.get(name)
        if fn is None:
            capability = TOOL_CAPABILITIES[name]
            fn = blocked_tool(name, str(capability["reason"]))
        registry[name] = tool_spec(
            name,
            fn,
            _DESCRIPTIONS.get(
                name,
                str(
                    TOOL_CAPABILITIES[name].get("required_rewrite")
                    or TOOL_CAPABILITIES[name].get("reason")
                    or name
                ),
            ),
        )
    return registry


def install_profile(skills_module, adapter) -> dict[str, ToolSpec]:
    registry = build_registry(adapter)
    skills_module.SKILL_REGISTRY = registry
    skills_module.PUBLIC_SKILLS = frozenset(PUBLIC_SKILLS)
    skills_module.TOOL_VERSION = TOOL_VERSION
    return registry


def profile_report() -> dict[str, Any]:
    return capability_report()


__all__ = [
    "build_registry",
    "install_profile",
    "profile_report",
    "validate_submission",
]
