import base64
from contextlib import asynccontextmanager
import inspect
import json
import math
import re
from typing import Any, Callable

from . import __version__
from .coordinates import (
    VLM_IMAGE_COORDINATE_CONTRACT,
    VLM_IMAGE_COORDINATE_SYSTEM,
    latest_image_grounding_reminder,
    response_to_pixels,
    schema_uses_image_coordinates,
    with_image_coordinate_descriptions,
)
from .errors import EmbodiedError
from .service import EmbodiedService, ToolResult
from .rollout_budget import normalize_budget, unavailable_budget
from .skills import activate_skill as load_task_skill
from .skills import deactivate_skill as leave_task_skill


SERVER_INSTRUCTIONS = f"""Operate the robot only through the BEHAVIOR v2 tools exposed by this MCP server. The MCP surface is loaded directly from the active interface `/api/v2/tools` catalog, with the stable `mark_on_map` contract restored when a profile omits it, plus `activate_skill` and `deactivate_skill`. `activate_skill` is this session's skill tool: omit `name` to list task Skills, or pass one Skill name to load that SKILL.md. Call `deactivate_skill` with its name to finish, cancel, or return to the baseline and parent task. Historical Skill text does not make it active; follow the latest lifecycle state. Use one operation at a time and treat returned RGB and structured results as current evidence. {VLM_IMAGE_COORDINATE_CONTRACT} Every direct tool result includes a compact `persistent_tracking` snapshot read after that operation (live `track_object_distance` points plus `mark_on_map` places); image-bearing results expose the same snapshot in a `persistent_tracking=...` text block. The top-right of each head image is a live SLAM map being built this episode: blue is the movement trail, straight up is the current heading. Use `mark_on_map` to mark an object or place; marked spots show on that map and as text in `persistent_tracking`. Use the map to tell direction. Never use shell or direct HTTP to bypass this adapter, and never start, restart, stop, or reconfigure the BEHAVIOR service. The adapter owns session_id, serializes calls, and records each model-visible call under its direct interface tool name. Task strategy comes only from the user or an activated Skill."""

ACTIVATE_SKILL_DESCRIPTION = (
    "Load exactly one task Skill the native way. Omit `name` to list available "
    "Skills (name and description only). Pass one Skill name to load that "
    "SKILL.md body and make it the only active task Skill. Do not activate "
    "behavior-v2-baseline; it is already loaded. After success and required "
    "post-actions, or to cancel or hand back control, call deactivate_skill."
)

DEACTIVATE_SKILL_DESCRIPTION = (
    "Deactivate the named current task Skill and return to the baseline and parent "
    "task. You may finish, cancel, or hand back control without user confirmation. "
    "This does not establish success, complete the parent task, change the robot, "
    "or delete conversation history. The baseline cannot be deactivated."
)

PERSISTENT_TRACKING_TOOL_DESCRIPTION = (
    "The result includes compact live `persistent_tracking` state read after this "
    "operation; `available=false` is fail-open tracker telemetry and does not "
    "change the operation's own status."
)
PERSISTENT_TRACKING_MAX_TEXT_CHARS = 4096
ROLLOUT_BUDGET_DESCRIPTION = (
    'Every tool reply includes rollout_budget: used_ticks, total_ticks, '
    'remaining_ticks, and used_fraction=used_ticks/total_ticks, sampled from '
    'the latest evaluator observation. Check episode_id before comparing '
    'counters; available=false means unknown, not zero. These are simulation '
    'steps, not wall-clock time, tokens, or a live goal score.'
)
SERVER_INSTRUCTIONS += ' ' + ROLLOUT_BUDGET_DESCRIPTION


def _rollout_budget_text(budget: Any) -> str:
    return 'rollout_budget=' + json.dumps(
        normalize_budget(budget), ensure_ascii=True, separators=(',', ':'), sort_keys=True
    )


def _skill_budget(provider: Callable[[], dict[str, Any]] | None) -> dict[str, Any]:
    try:
        return normalize_budget(provider()) if provider else unavailable_budget()
    except Exception:
        # Optional telemetry must never replace a skill's success/failure.
        return unavailable_budget()

NEARBY_OBJECT_WARNING_MAX_CHARS = 512
NEARBY_OBJECT_WARNING_LABEL = "距离机器人近的物体预警"
_CHASSIS_FORWARD_2M_CLEAR = "No object on the chassis-forward path within 2m"

READ_ONLY_TOOLS = {
    "capture_head_camera",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "measure_shoulder_distance",
}

CAPTION_VALUE_RE = re.compile(r"[^A-Za-z0-9._:/,+-]+")

ACTION_EVIDENCE_SCALAR_FIELDS = (
    "ok",
    "error",
    "failure_reason",
    "image_id",
    "feed",
    "arm",
    "plan_id",
    "plan_arm",
    "recommended_arm",
    "target_reached",
    "position_reached",
    "reachable",
    "linear_target_reached",
    "translation_ok",
    "near_target_ok",
    "obstacle_limited",
    "obstacle_stop_reason",
    "nearby_object_warning",
    "recovery_attempted",
    "recovery_ok",
    "recovery_trigger",
    "require_linear_target",
    "rgbd_motion_guard_used",
    "action_steps",
    "settled_steps",
    "pos_ok",
    "ori_ok",
    "j8_ok",
    "pos_err_mm",
    "ori_err_deg",
    "degree",
    "spin_deg",
    "forward_m",
    "ray_angle_deg",
    "clicked_bearing_deg",
    "motion_bearing_deg",
    "target_distance_m",
    "commanded_base_displacement_m",
    "effective_pos_tol_m",
    "requested_pos_tol_m",
    "linear_commanded_distance_m",
    "forward_commanded_distance_m",
    "linear_remaining_m",
    "linear_cross_track_m",
    "linear_tolerance_m",
    "commanded_gripper_action",
    "close_keepalive_active",
    "control_mode",
    "surface",
    "verification",
    "landing_verification",
    "execution_steps",
    "base_horizontal_to_object_m",
    "shoulder_mid_horizontal_to_object_m",
    "left_shoulder_to_object_m",
    "right_shoulder_to_object_m",
    "current_shoulder_line_pitch_deg",
    "target_source",
    "depth_m",
    "source",
    "unit",
    "grasp_confirmed",
    "actuation_detected",
    "timed_out",
    "http_status",
    "retryable",
    "exit_capture_attached",
    "failure_stage",
)

ACTION_EVIDENCE_VECTOR_FIELDS = (
    "eef_before",
    "eef_after",
    "eef_target",
    "finger_qpos",
    "gripper_qpos_before",
    "gripper_qpos_after",
    "start_trunk_q",
    "target_trunk_q",
    "point_local_m",
    "point_robot_m_at_start",
    "front_reference_robot_xy_m",
    "recovery_motion_vector",
    "target_robot_m",
    "shoulder_mid_robot_m",
    "left_shoulder_robot_m",
    "right_shoulder_robot_m",
    "target_aligned_robot_m",
    "relative_uv",
    "pixel_uv",
    "input_pixel_uv",
    "point_robot_m",
    "xyz_in_robot_base_coord_m",
    "robot_base_xyz_m",
)

ACTION_EVIDENCE_MOTION_FIELDS = (
    "forward_m",
    "translation_m",
    "spin_deg",
    "forward",
    "leftward",
    "upward",
    "pitch",
    "roll",
    "yaw",
    "x",
    "y",
    "z",
)

ACTION_EVIDENCE_MOTION_OBJECTS = (
    "requested",
    "actual",
    "translation_m",
    "rotation_deg",
    "delta_camera_m",
    "delta_head_camera_m",
    "delta_robot_m",
    "delta_world_m",
)

ACTION_EVIDENCE_NESTED_FIELDS = (
    "point_observation",
    "recommended_pitch",
    "response",
    "capture_error",
    "shoulder_distance_estimate",
    "execution",
)

ACTION_EVIDENCE_AUTO_EXCLUDED_FIELDS = frozenset(
    {
        "base_path_overlay",
        "camera",
        "depth",
        "diagnostic_payload",
        "eef_near_0.1m",
        "grasp_zone_overlay",
        "image_height",
        "image_width",
        "memory",
        "rgb",
        "rgb_main",
        "robot",
        "track_object_distance_binding",
    }
)

ACTION_EVIDENCE_EXCLUDED_FIELD_TOKENS = (
    "diagnostic",
    "debug",
    "payload",
    "replay",
    "scene_graph",
    "trace",
)

ACTION_EVIDENCE_MAX_STRING_CHARS = 512
ACTION_EVIDENCE_MAX_VECTOR_ITEMS = 16
ACTION_EVIDENCE_MAX_TEXT_CHARS = 4096
ACTION_EVIDENCE_PROTECTED_MAX_TEXT_CHARS = 2048
MODEL_STRUCTURED_MAX_TEXT_CHARS = 10 * 1024
MODEL_ERROR_MAX_TEXT_CHARS = 4096
MODEL_ERROR_MAX_MESSAGE_CHARS = 512
MODEL_ERROR_FALLBACK_MESSAGE_CHARS = 160


def _caption_value(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return CAPTION_VALUE_RE.sub("_", text)[:160]


def _evidence_scalar(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:ACTION_EVIDENCE_MAX_STRING_CHARS]
    return None


def _evidence_vector(value: Any) -> list[Any] | None:
    if not isinstance(value, (list, tuple)):
        return None
    compact: list[Any] = []
    for item in value[:ACTION_EVIDENCE_MAX_VECTOR_ITEMS]:
        scalar = _evidence_scalar(item)
        if scalar is None and item is not None:
            return None
        compact.append(scalar)
    return compact


def _selected_scalars(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    selected: dict[str, Any] = {}
    for field in fields:
        value = _evidence_scalar(source.get(field))
        if value is not None:
            selected[field] = value
    return selected


def _selected_vectors(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    selected: dict[str, Any] = {}
    for field in fields:
        value = _evidence_vector(source.get(field))
        if value is not None:
            selected[field] = value
    return selected


def _is_model_evidence_field(field: str) -> bool:
    lowered = field.lower()
    return (
        lowered not in ACTION_EVIDENCE_AUTO_EXCLUDED_FIELDS
        and not any(
            token in lowered for token in ACTION_EVIDENCE_EXCLUDED_FIELD_TOKENS
        )
        and not lowered.endswith("_path")
        and not lowered.endswith("_url")
    )


def _compact_mapping(value: Any, *, nested: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for raw_field, child in value.items():
        field = str(raw_field)
        if not _is_model_evidence_field(field):
            continue
        scalar = _evidence_scalar(child)
        if scalar is not None:
            compact[field] = scalar
            continue
        vector = _evidence_vector(child)
        if vector is not None:
            compact[field] = vector
            continue
        if nested and isinstance(child, dict):
            nested_value = _compact_mapping(child)
            if nested_value:
                compact[field] = nested_value
    return compact


def _bounded_exact_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    if len(payload) > ACTION_EVIDENCE_PROTECTED_MAX_TEXT_CHARS:
        return None
    decoded = json.loads(payload)
    return decoded if isinstance(decoded, dict) else None


def _protected_action_evidence(
    tool_name: str, response: dict[str, Any]
) -> dict[str, Any]:
    field = {
        "plan_grasp_point_filter_rgbd_lite": "selected_pose_ik",
        "move_to_reach_point": "shoulder_distance_estimate",
    }.get(tool_name)
    if field is None:
        return {}
    value = _bounded_exact_mapping(response.get(field))
    return {field: value} if value is not None else {}


def _filtered_robot_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    robot = _selected_scalars(value, ("frame", "motion_epoch"))
    robot.update(
        _selected_vectors(
            value,
            (
                "base_qvel",
                "gripper_left_qpos",
                "gripper_right_qpos",
                "trunk_qpos",
            ),
        )
    )

    base_pose = value.get("base_pose")
    if isinstance(base_pose, dict):
        pose = _selected_scalars(base_pose, ("yaw_deg",))
        pose.update(_selected_vectors(base_pose, ("pos",)))
        if pose:
            robot["base_pose"] = pose

    for field in ("eef_left", "eef_right"):
        eef = value.get(field)
        if not isinstance(eef, dict):
            continue
        pose = _selected_scalars(eef, ("frame",))
        pose.update(_selected_vectors(eef, ("pos",)))
        if pose:
            robot[field] = pose
    return robot


def _capture_quality_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    quality = _selected_scalars(
        value,
        (
            "attempts",
            "discarded_bad_frames",
            "minimum_valid_depth_ratio",
        ),
    )
    accepted = _selected_scalars(
        value.get("accepted_frame"),
        (
            "image_id",
            "valid_depth_pixel_count",
            "total_pixel_count",
            "valid_depth_ratio",
        ),
    )
    if accepted:
        quality["accepted_frame"] = accepted
    return quality


def _action_evidence(result: ToolResult) -> dict[str, Any]:
    response = result.data.get("response")
    response = response if isinstance(response, dict) else {}
    evidence: dict[str, Any] = {
        "tool_name": str(result.data.get("tool_name") or "unknown")
    }
    evidence.update(_selected_scalars(response, ACTION_EVIDENCE_SCALAR_FIELDS))
    evidence.update(_selected_vectors(response, ACTION_EVIDENCE_VECTOR_FIELDS))

    # Preserve future scalar and short-vector result fields without exposing
    # media, file paths, or bulky observation payloads. The explicit fields
    # above remain the bounded fallback contract.
    for field, value in _compact_mapping(response).items():
        evidence.setdefault(field, value)

    for field in ACTION_EVIDENCE_NESTED_FIELDS:
        nested_value = _compact_mapping(response.get(field), nested=True)
        if nested_value:
            evidence[field] = nested_value

    for field in ACTION_EVIDENCE_MOTION_OBJECTS:
        motion = _selected_scalars(
            response.get(field), ACTION_EVIDENCE_MOTION_FIELDS
        )
        if motion:
            evidence[field] = motion

    robot = _filtered_robot_state(response.get("robot"))
    if robot:
        evidence["robot"] = robot

    capture_quality = _capture_quality_evidence(result.data.get("capture_quality"))
    if capture_quality:
        evidence["capture_quality"] = capture_quality

    evidence.update(
        _protected_action_evidence(evidence["tool_name"], response)
    )

    warnings = result.data.get("warnings")
    if isinstance(warnings, list):
        compact_warnings = [
            warning
            for item in warnings[:4]
            if (warning := _evidence_scalar(item)) is not None
        ]
        if compact_warnings:
            evidence["warnings"] = compact_warnings
    _sanitize_model_nearby_warning(evidence)
    return evidence


def _action_evidence_text(result: ToolResult) -> str:
    evidence = _action_evidence(result)
    payload = json.dumps(
        evidence, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    if len(payload) > ACTION_EVIDENCE_MAX_TEXT_CHARS:
        for field in ("execution", "robot", "capture_quality"):
            evidence.pop(field, None)
        payload = json.dumps(
            evidence, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    if len(payload) > ACTION_EVIDENCE_MAX_TEXT_CHARS:
        response = result.data.get("response")
        response = response if isinstance(response, dict) else {}
        evidence = {
            "tool_name": str(result.data.get("tool_name") or "unknown"),
            "evidence_truncated": True,
        }
        evidence.update(
            _selected_scalars(response, ACTION_EVIDENCE_SCALAR_FIELDS)
        )
        evidence.update(
            _selected_vectors(response, ACTION_EVIDENCE_VECTOR_FIELDS)
        )
        for field in ACTION_EVIDENCE_NESTED_FIELDS[:-1]:
            nested_value = _compact_mapping(response.get(field), nested=True)
            if nested_value:
                evidence[field] = nested_value
        evidence.update(
            _protected_action_evidence(evidence["tool_name"], response)
        )
        _sanitize_model_nearby_warning(evidence)
        payload = json.dumps(
            evidence, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    if len(payload) > ACTION_EVIDENCE_MAX_TEXT_CHARS:
        raise ValueError("Core action evidence exceeded its bounded MCP contract")
    return f"action_evidence={payload}"


def _persistent_tracking_text(result: ToolResult) -> str:
    tracking = result.data.get("persistent_tracking")
    if not isinstance(tracking, dict):
        tracking = {"available": False, "warning": "Tracking snapshot is missing."}

    payload = json.dumps(
        tracking, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    if len(payload) > PERSISTENT_TRACKING_MAX_TEXT_CHARS:
        raw_tracks = tracking.get("tracks")
        raw_tracks = raw_tracks if isinstance(raw_tracks, dict) else {}
        bounded = {key: value for key, value in tracking.items() if key != "tracks"}
        bounded["tracks"] = {}
        bounded["model_text_tracks_truncated"] = True
        bounded.setdefault("total_track_count", len(raw_tracks))
        for name, track in sorted(raw_tracks.items(), key=lambda item: str(item[0])):
            candidate_tracks = dict(bounded["tracks"])
            candidate_tracks[str(name)] = track
            candidate = dict(bounded)
            candidate["tracks"] = candidate_tracks
            candidate_payload = json.dumps(
                candidate,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            if len(candidate_payload) > PERSISTENT_TRACKING_MAX_TEXT_CHARS:
                continue
            bounded = candidate
        payload = json.dumps(
            bounded, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    if len(payload) > PERSISTENT_TRACKING_MAX_TEXT_CHARS:
        raise ValueError("Persistent tracking exceeded its bounded MCP text contract")
    return f"persistent_tracking={payload}"


_EEF_NEAR_WARNING = re.compile(
    r"(?:非加持物体)?object near (?:left|right) eef：[^|]*",
    flags=re.IGNORECASE,
)


def _strip_eef_near_from_warning(text: str) -> str:
    """模型侧预警只留底盘走廊；手边 eef 句先丢掉。"""
    cleaned = _EEF_NEAR_WARNING.sub("", text)
    parts: list[str] = []
    for part in cleaned.split("|"):
        item = part.strip()
        if not item:
            continue
        lowered = item.lower()
        if "object near" in lowered and "eef" in lowered:
            continue
        if item.startswith("非加持物体"):
            continue
        parts.append(item)
    return " | ".join(parts)


def _compose_nearby_object_warning(chassis: object, eef: object = "") -> str:
    """只合成成熟的底盘走廊预警；eef 参数保留但不采用。"""
    del eef
    chassis_text = chassis.strip() if isinstance(chassis, str) else ""
    if chassis_text and chassis_text != _CHASSIS_FORWARD_2M_CLEAR:
        return chassis_text
    return ""


def _sanitize_model_nearby_warning(evidence: dict[str, Any]) -> None:
    """证据里也不给模型看 eef 近距字段。"""
    evidence.pop("eef_near_0.1m", None)
    warning = evidence.get("nearby_object_warning")
    if not isinstance(warning, str):
        return
    cleaned = _strip_eef_near_from_warning(warning)
    if cleaned:
        evidence["nearby_object_warning"] = cleaned
    else:
        evidence.pop("nearby_object_warning", None)


def _nearby_object_warning_text(result: ToolResult) -> str | None:
    """头图给模型的近物预警一行，不画进 RGB。只报底盘。"""
    response = result.data.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("nearby_object_warning")
    if isinstance(raw, str):
        sentence = _strip_eef_near_from_warning(raw)
    else:
        sentence = _compose_nearby_object_warning(
            response.get("chassis_forward_2m"),
        )
    text = sentence.strip()
    if not text or len(text) > NEARBY_OBJECT_WARNING_MAX_CHARS:
        return None
    return f"{NEARBY_OBJECT_WARNING_LABEL}={text}"


def _bounded_text_payload(text: str, prefix: str) -> dict[str, Any]:
    if not text.startswith(prefix):
        raise ValueError(f"Expected bounded MCP payload prefix {prefix!r}")
    value = json.loads(text.removeprefix(prefix))
    if not isinstance(value, dict):
        raise ValueError("Bounded MCP payload must be an object")
    return value


def _model_structured_content(result: ToolResult) -> dict[str, Any]:
    """Return the bounded no-image fallback exposed to the MCP client."""
    evidence = _bounded_text_payload(
        _action_evidence_text(result), "action_evidence="
    )
    tracking = _bounded_text_payload(
        _persistent_tracking_text(result), "persistent_tracking="
    )

    tool_name = str(evidence.pop("tool_name", "unknown"))
    warnings = evidence.pop("warnings", None)
    capture_quality = evidence.pop("capture_quality", None)
    structured: dict[str, Any] = {
        "tool_name": tool_name,
        "response": evidence,
        "persistent_tracking": tracking,
        'rollout_budget': normalize_budget(result.data.get('rollout_budget')),
    }
    if warnings is not None:
        structured["warnings"] = warnings
    if capture_quality is not None:
        structured["capture_quality"] = capture_quality
    if result.rest_media_candidate_count or result.media:
        structured["media_counts"] = {
            "rest_candidates": result.rest_media_candidate_count,
            "selected_for_model": len(result.media),
        }

    payload = json.dumps(
        structured, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    if len(payload) > MODEL_STRUCTURED_MAX_TEXT_CHARS:
        raise ValueError("Structured result exceeded its bounded MCP contract")
    return structured


def _omitted_error_response(value: Any, *, reason: str) -> dict[str, Any]:
    omitted: dict[str, Any] = {
        "omitted": True,
        "reason": reason,
        "type": type(value).__name__,
    }
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict)):
        omitted["size"] = len(value)
    return omitted


def _model_error_response(value: Any, *, tool_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "response_omitted": _omitted_error_response(
                value, reason="non_object_remote_response"
            )
        }

    result = ToolResult(
        summary=f"{tool_name} failed.",
        data={"tool_name": tool_name, "response": response_to_pixels(value)},
        is_error=True,
    )
    try:
        evidence = _bounded_text_payload(
            _action_evidence_text(result), "action_evidence="
        )
    except ValueError:
        return {
            "response_omitted": _omitted_error_response(
                value, reason="response_evidence_exceeded_contract"
            )
        }
    evidence.pop("tool_name", None)
    return {"response": evidence, "response_sanitized": True}


def _model_error_payload(
    exc: Exception, *, tool_name: str = "unknown"
) -> dict[str, Any]:
    """Build the only error representation allowed across the MCP boundary."""
    bounded_tool_name = str(tool_name or "unknown")[:160]
    if isinstance(exc, EmbodiedError):
        error: dict[str, Any] = {
            "code": str(exc.code)[:160],
            "message": str(exc.message)[:MODEL_ERROR_MAX_MESSAGE_CHARS],
            "retryable": bool(exc.retryable),
            "tool_name": bounded_tool_name,
        }
        raw_details = exc.details
    else:
        error = {
            "code": "adapter_error",
            "message": str(exc)[:MODEL_ERROR_MAX_MESSAGE_CHARS],
            "retryable": False,
            "tool_name": bounded_tool_name,
        }
        raw_details = None

    details: dict[str, Any] = {}
    if isinstance(raw_details, dict):
        details.update(
            _compact_mapping(
                {
                    key: value
                    for key, value in raw_details.items()
                    if str(key) != "response"
                },
                nested=True,
            )
        )
        if "response" in raw_details:
            details.update(
                _model_error_response(
                    raw_details["response"], tool_name=bounded_tool_name
                )
            )
    elif raw_details is not None:
        context = _evidence_scalar(raw_details)
        if context is not None:
            details["context"] = context
    if details:
        error["details"] = details

    payload = {"error": error}
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    if len(serialized) > MODEL_ERROR_MAX_TEXT_CHARS:
        minimal_details: dict[str, Any] = {
            "additional_details_omitted": True
        }
        http_status = details.get("http_status")
        if isinstance(http_status, (int, float)) and not isinstance(
            http_status, bool
        ):
            minimal_details["http_status"] = http_status
        response_omitted = details.get("response_omitted")
        if response_omitted is None and "response" in details:
            response_omitted = _omitted_error_response(
                details["response"], reason="bounded_model_error"
            )
        if response_omitted is not None:
            minimal_details["response_omitted"] = response_omitted
        error["details"] = minimal_details
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    if len(serialized) > MODEL_ERROR_MAX_TEXT_CHARS:
        fallback_details: dict[str, Any] = {
            "additional_details_omitted": True
        }
        http_status = details.get("http_status")
        if isinstance(http_status, (int, float)) and not isinstance(
            http_status, bool
        ):
            fallback_details["http_status"] = http_status
        payload = {
            "error": {
                "code": _caption_value(error.get("code"), "adapter_error"),
                "message": str(error.get("message") or "Adapter error")[:
                    MODEL_ERROR_FALLBACK_MESSAGE_CHARS
                ],
                "retryable": bool(error.get("retryable")),
                "tool_name": _caption_value(
                    error.get("tool_name"), "unknown"
                ),
                "details": fallback_details,
            }
        }
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    if len(serialized) > MODEL_ERROR_MAX_TEXT_CHARS:
        raise AssertionError("Minimal MCP error result exceeded its hard contract")
    return payload


def _result_summary(result: ToolResult, *, emitted_images: int = 0) -> str:
    response = result.data.get("response")
    details: list[str] = [result.summary]
    if result.is_error:
        details.append("action_status=failed")
    if isinstance(response, dict):
        for key in ("image_id", "feed", "ok", "error", "failure_reason"):
            value = response.get(key)
            if value is not None and value != "":
                details.append(f"{key}={_caption_value(value)}")
    warnings = result.data.get("warnings")
    if isinstance(warnings, list) and warnings:
        details.append(f"warnings={len(warnings)}")
    if result.rest_media_candidate_count or result.media or emitted_images:
        details.extend(
            (
                f"rest_media_candidates={result.rest_media_candidate_count}",
                f"selected_media={len(result.media)}",
                f"emitted_images={emitted_images}",
            )
        )
    return " ".join(details)


def _image_caption(result: ToolResult, media: Any, _catalog: Any) -> str:
    response = result.data.get("response")
    response = response if isinstance(response, dict) else {}
    geometry = result.data.get("image_geometry", {})
    clickable = geometry.get("clickable") is True
    coordinate_system = VLM_IMAGE_COORDINATE_SYSTEM if clickable else "observation_only"
    usage = {
        "raw_rgb": "ground_physical_objects_in_this_raw_image_only",
        "path_overlay": (
            "read_blue_path_and_underlying_scene_ignore_overlay_text_as_coordinates"
        ),
        "grasp_volume_overlay": "read_grasp_volume_from_this_overlay_only",
        "planning_overlay": "read_planning_marks_from_this_overlay_only",
        "tracking_overlay": "read_tracking_marks_from_this_overlay_only",
        "marked_overlay": "read_marks_from_this_overlay_only",
        "auxiliary_overlay": "read_overlay_annotations_only",
        "display_rgb": "display_fallback_check_role_before_grounding",
    }.get(media.role, "check_role_before_grounding")
    coordinate_details = (
        "coordinate_canvas=720x720 origin=top_left bottom_right=719,719 "
        "coordinate_values=original_image_pixels_0_719 " if clickable else
        "point_selection=forbidden_capture_a_fresh_720x720_head_image "
    )
    return (
        f"image label={_caption_value(media.label)} "
        f"role={_caption_value(media.role)} "
        f"image_id={_caption_value(geometry.get('image_id') or response.get('image_id'))} "
        f"coordinate_system={_caption_value(coordinate_system)} "
        f"width={media.width} height={media.height} "
        f"clickable={str(clickable).lower()} "
        f"{coordinate_details}"
        "coordinate_binding=this_emitted_image_only "
        f"usage={usage}"
    )


def create_mcp_server(service: EmbodiedService | None = None) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.tools import Tool
        from mcp.types import (
            CallToolResult,
            ImageContent,
            TextContent,
            ToolAnnotations,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The MCP SDK is not installed. Install this package with its dependencies."
        ) from exc

    embodied = service or EmbodiedService()
    catalog = embodied.prepare_episode(
        session_id=embodied.settings.session_id,
        record=embodied.settings.record,
        label=embodied.settings.record_label,
    )
    @asynccontextmanager
    async def lifespan(_: Any):
        try:
            yield None
        finally:
            embodied.finish_episode(
                outcome="unknown", note="MCP client session closed."
            )

    def as_mcp(result: ToolResult) -> Any:
        if 'rollout_budget' not in result.data:
            result.data['rollout_budget'] = embodied.read_rollout_budget()
        emitted_images = len(result.media)
        response = result.data.get("response")
        response = response if isinstance(response, dict) else {}
        content: list[Any] = [
            TextContent(
                type="text",
                text=_result_summary(result, emitted_images=emitted_images),
            )
        ]
        if result.media:
            content.append(
                TextContent(type="text", text=_action_evidence_text(result))
            )
            content.append(
                TextContent(type="text", text=_persistent_tracking_text(result))
            )
        content.append(TextContent(type='text', text=_rollout_budget_text(
            result.data.get('rollout_budget'))))
        warning = _nearby_object_warning_text(result)
        if warning:
            content.append(TextContent(type="text", text=warning))
        for media in result.media:
            content.append(
                TextContent(
                    type="text",
                    text=_image_caption(result, media, catalog),
                )
            )
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(media.data).decode("ascii"),
                    mime_type=media.mime_type,
                )
            )
            content.append(
                TextContent(
                    type="text",
                    text=latest_image_grounding_reminder(
                        _caption_value(result.data.get("image_geometry", {}).get("image_id")
                                       or response.get("image_id"), "")
                    ),
                )
            )
        return CallToolResult(
            content=content,
            # Keep native images instead of a duplicate structured REST payload.
            structured_content=(
                None if result.media else _model_structured_content(result)
            ),
            # Claude Code discards images in isError results. Deliver observation
            # updates normally; action_status and the recording retain failure.
            is_error=result.is_error and not result.media,
        )

    def error_result(tool_name: str, exc: Exception) -> Any:
        payload = _model_error_payload(exc, tool_name=tool_name)
        payload['rollout_budget'] = embodied.read_rollout_budget()
        return CallToolResult(
            content=[
                TextContent(type='text', text=json.dumps(
                    payload['error'], ensure_ascii=True, separators=(',', ':'), sort_keys=True)),
                TextContent(type='text', text=_rollout_budget_text(payload['rollout_budget'])),
            ],
            structured_content=payload, is_error=True,
        )

    def invoke(tool_name: str, operation: Callable[[], ToolResult]) -> Any:
        try:
            return as_mcp(operation())
        except Exception as exc:
            return error_result(tool_name, exc)

    class BudgetMCPServer(MCPServer):
        async def call_tool(self, name, arguments, context=None):
            try:
                return await super().call_tool(name, arguments, context)
            except Exception as exc:
                # SDK argument validation and unknown tools fail before invoke.
                return error_result(name, exc)

    tools = []
    for spec in catalog.tools.values():
        function = _direct_tool(spec.name, spec.input_schema, invoke, embodied)
        claude_schema, argument_requirements = _claude_input_schema(
            spec.name, spec.input_schema
        )
        description = (
            f"{spec.description.rstrip()} {PERSISTENT_TRACKING_TOOL_DESCRIPTION} "
            'The result also includes rollout_budget simulation-step usage.'
        )
        if schema_uses_image_coordinates(spec.input_schema):
            description += " " + VLM_IMAGE_COORDINATE_CONTRACT
        if argument_requirements:
            description += " " + argument_requirements
        tool = Tool.from_function(
            function,
            name=spec.name,
            description=description,
            annotations=ToolAnnotations(
                readOnlyHint=spec.name in READ_ONLY_TOOLS,
                destructiveHint=spec.name not in READ_ONLY_TOOLS,
                openWorldHint=False,
            ),
            structured_output=False,
        )
        tool.parameters = claude_schema
        tool.fn_metadata.arg_model.model_config["extra"] = "forbid"
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        tools.append(tool)

    session_id = embodied._require_episode().session_id
    tools.append(_activate_skill_mcp_tool(session_id, embodied.settings.base_url, embodied.read_rollout_budget))
    tools.append(_deactivate_skill_mcp_tool(session_id, embodied.settings.base_url, embodied.read_rollout_budget))

    return BudgetMCPServer(
        "embodied-claude-code-behavior-v2",
        title="Embodied Claude Code BEHAVIOR v2",
        description="Direct, recordable mirror of the BEHAVIOR v2 REST tool catalog.",
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        tools=tools,
    )


def _claude_input_schema(
    tool_name: str, input_schema: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Move generated top-level anyOf constraints out of Claude's tool schema.

    The Anthropic API rejects an otherwise valid object tool schema when it has
    a top-level ``anyOf``. ToolCatalog retains and enforces the original schema
    at call time; this MCP-facing copy describes the accepted combinations so
    the tool remains visible without weakening runtime validation.
    """
    schema = with_image_coordinate_descriptions(input_schema)
    alternatives = schema.pop("anyOf", None)
    if alternatives is None:
        return schema, ""
    if not isinstance(alternatives, list) or not alternatives:
        raise EmbodiedError(
            "invalid_tool_catalog",
            f"Tool {tool_name!r} has an unsupported empty anyOf constraint.",
        )

    combinations: list[tuple[str, ...]] = []
    for alternative in alternatives:
        if not isinstance(alternative, dict) or set(alternative) != {"required"}:
            raise EmbodiedError(
                "invalid_tool_catalog",
                f"Tool {tool_name!r} has an unsupported anyOf constraint.",
            )
        required = alternative.get("required")
        if not isinstance(required, list) or not required or not all(
            isinstance(name, str) and name for name in required
        ):
            raise EmbodiedError(
                "invalid_tool_catalog",
                f"Tool {tool_name!r} has an invalid anyOf requirement.",
            )
        combinations.append(tuple(required))

    rendered = [" + ".join(f"`{name}`" for name in names) for names in combinations]
    requirement = "Required arguments: supply " + " OR ".join(rendered) + "."
    root_description = str(schema.get("description") or "").strip()
    schema["description"] = " ".join(
        part for part in (root_description, requirement) if part
    )
    return schema, requirement


def _activate_skill_mcp_tool(session_id: str = "", base_url: str = "",
                             budget_provider: Callable[[], dict[str, Any]] | None = None) -> Any:
    from mcp.server.mcpserver.tools import Tool
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    async def activate_skill(name: str | None = None) -> CallToolResult:
        try:
            payload = load_task_skill(name or "", session_id=session_id, base_url=base_url)
        except Exception as exc:
            payload = {'ok': False, **_model_error_payload(exc, tool_name='activate_skill')}
        payload = {**payload, 'rollout_budget': _skill_budget(budget_provider)}
        text = json.dumps(payload, ensure_ascii=True, indent=2)
        if payload.get("mode") == "activated" and payload.get("body"):
            text = (
                f'<activated_skill name="{payload["name"]}">\n'
                + str(payload["body"]).strip()
                + "\n</activated_skill>"
                + "\n" + payload["instruction"]
                + "\n" + payload["state_notice"]
            )
        return CallToolResult(
            content=[TextContent(type="text", text=text),
                     TextContent(type='text', text=_rollout_budget_text(payload['rollout_budget']))],
            structured_content=payload,
            is_error=not bool(payload.get("ok")),
        )

    tool = Tool.from_function(
        activate_skill,
        name="activate_skill",
        description=ACTIVATE_SKILL_DESCRIPTION,
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    tool.parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Task Skill name such as pick-up-object. Omit to list "
                    "available Skills."
                ),
            }
        },
        "additionalProperties": False,
    }
    tool.fn_metadata.arg_model.model_config["extra"] = "forbid"
    tool.fn_metadata.arg_model.model_rebuild(force=True)
    return tool


def _deactivate_skill_mcp_tool(session_id: str = "", base_url: str = "",
                               budget_provider: Callable[[], dict[str, Any]] | None = None) -> Any:
    from mcp.server.mcpserver.tools import Tool
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    async def deactivate_skill(name: str, reason: str = "") -> CallToolResult:
        try:
            payload = leave_task_skill(name, reason, session_id=session_id, base_url=base_url)
        except Exception as exc:
            payload = {'ok': False, **_model_error_payload(exc, tool_name='deactivate_skill')}
        payload = {**payload, 'rollout_budget': _skill_budget(budget_provider)}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=True)),
                     TextContent(type='text', text=_rollout_budget_text(payload['rollout_budget']))],
            structured_content=payload,
            is_error=not bool(payload.get("ok")),
        )

    tool = Tool.from_function(
        deactivate_skill, name="deactivate_skill", description=DEACTIVATE_SKILL_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                    idempotentHint=True, openWorldHint=False),
        structured_output=False,
    )
    tool.parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the current task Skill to deactivate."},
            "reason": {"type": "string", "description": "Optional reason for completion, cancellation, or handoff."},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    tool.fn_metadata.arg_model.model_config["extra"] = "forbid"
    tool.fn_metadata.arg_model.model_rebuild(force=True)
    return tool


def _direct_tool(
    tool_name: str,
    input_schema: dict[str, Any],
    invoke: Callable[[str, Callable[[], ToolResult]], Any],
    service: EmbodiedService,
) -> Callable[..., Any]:
    from mcp.types import CallToolResult

    async def call(**kwargs: Any) -> CallToolResult:
        arguments = {key: value for key, value in kwargs.items() if value is not None}
        return invoke(
            tool_name,
            lambda: service.call(tool_name=tool_name, arguments=arguments)
        )

    parameters = []
    required = set(input_schema.get("required", []))
    properties = input_schema.get("properties", {})
    for name in properties:
        if name in required:
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Any,
                )
            )
    for name in properties:
        if name not in required:
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Any,
                )
            )
    call.__name__ = tool_name
    call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=CallToolResult,
    )
    return call


def main() -> None:
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
