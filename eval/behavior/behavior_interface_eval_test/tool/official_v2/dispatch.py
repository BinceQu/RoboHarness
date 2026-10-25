"""Translate existing v2 HTTP wrapper calls to the current public tool names."""

from __future__ import annotations

from typing import Any

from .capabilities import (
    PUBLIC_TOOLS,
    OfficialToolBoundaryError,
)


_DIRECT_LEGACY_TO_PUBLIC = {
    "capture": "capture_head_camera",
    "capture_left_wrist_camera": "capture_left_wrist_camera",
    "capture_right_wrist_camera": "capture_right_wrist_camera",
    "move_base_to_point": "move_chassis_to_floor_point",
    "face_to_point": "spin_to_facing_point",
    "adjust_left_eef_pose_in_head_frame": "adjust_left_eef_pose_in_head_frame",
    "adjust_right_eef_pose_in_head_frame": "adjust_right_eef_pose_in_head_frame",
    "adjust_left_eef_pose_in_wrist_frame": "adjust_left_eef_pose_in_wrist_frame",
    "adjust_right_eef_pose_in_wrist_frame": "adjust_right_eef_pose_in_wrist_frame",
    "plan_move_eef": "plan_eef_translation_to_uvd_point",
    "adjust_plan_pose": "adjust_plan_pose",
    "move_to_point_v2": "move_to_reach_point",
    "move_to_point_v3": "move_to_reach_point",
    "mesure_shoulder_distance": "measure_shoulder_distance",
    "plan_eef_rgbd_batch": "plan_grasp_point_filter_rgbd",
    "plan_eef_rgbd_lite": "plan_grasp_point_filter_rgbd_lite",
    "exec_eef_pose_v2": "exec_plan_pose",
    "set_arm_to_grasp_position": "set_arm_to_grasp_position",
    "reset_body": "reset_body",
    "move_tracked_points": "move_tracked_point",
}


def _nonzero(args: dict[str, Any], key: str) -> bool:
    try:
        return abs(float(args.get(key, 0.0))) > 1e-12
    except (TypeError, ValueError):
        return False


def translate_submission(
    name: str,
    args: dict[str, Any],
    *,
    public_hint: str = "",
) -> tuple[str, dict[str, Any]]:
    """Return a public v2 tool name and public argument shape."""
    legacy_name = str(name)
    normalized = dict(args or {})
    hint = str(public_hint or "").strip()

    if legacy_name in PUBLIC_TOOLS:
        return legacy_name, normalized

    if legacy_name == "move_in_robot_coord":
        if hint in ("adjust_chassis", "adjust_pitch", "adjust_height"):
            public_name = hint
        elif _nonzero(normalized, "pitch"):
            public_name = "adjust_pitch"
        elif _nonzero(normalized, "upward"):
            public_name = "adjust_height"
        else:
            public_name = "adjust_chassis"

        if public_name == "adjust_pitch":
            return public_name, {
                **normalized,
                "degree": normalized.get("pitch", normalized.get("degree", 0.0)),
            }
        if public_name == "adjust_height":
            return public_name, {
                **normalized,
                "upward": normalized.get("upward", 0.0),
            }
        return public_name, {
            **normalized,
            "forward": normalized.get("forward", 0.0),
            "translation": normalized.get(
                "translation",
                normalized.get("leftward", 0.0),
            ),
            "spin": normalized.get("spin", 0.0),
        }

    if legacy_name == "move_eef":
        moving = any(
            _nonzero(normalized, key)
            for key in ("upward", "forward", "leftward")
        )
        has_uvd = any(
            normalized.get(key) is not None
            and str(normalized.get(key)).strip() != ""
            for key in ("u", "v", "depth")
        )
        gripper = str(normalized.get("gripper", "keep")).strip().lower()
        if moving or has_uvd or gripper not in ("open", "close"):
            raise OfficialToolBoundaryError(
                "legacy move_eef has no public official_v2 equivalent except "
                "zero-motion open_gripper or close_gripper"
            )
        public_name = "open_gripper" if gripper == "open" else "close_gripper"
        return public_name, {
            "arm": normalized.get("arm", "right"),
            "session_id": normalized.get("session_id", ""),
        }

    if legacy_name == "plan_eef_v2":
        mode = str(normalized.get("mode", "")).strip()
        if mode == "grasp_point_filter":
            return "plan_grasp_point_filter", normalized
        if mode == "press_point":
            return "plan_press_point", normalized
        raise OfficialToolBoundaryError(
            f"legacy plan mode {mode!r} is not in the public official_v2 surface"
        )

    public_name = _DIRECT_LEGACY_TO_PUBLIC.get(legacy_name)
    if public_name is None:
        raise OfficialToolBoundaryError(
            f"legacy skill {legacy_name!r} has no public official_v2 tool mapping"
        )
    return public_name, normalized


__all__ = ["translate_submission"]
