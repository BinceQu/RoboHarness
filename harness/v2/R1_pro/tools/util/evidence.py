"""Pinned baseline evidence selection; pure functions, no tool transport."""
from __future__ import annotations
import json
import math
import re
from typing import Any

ToolResult = Any

PERSISTENT_TRACKING_MAX_TEXT_CHARS = 4096


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


PERSISTENT_TRACKING_MAX_TRACKS = 8


PERSISTENT_TRACKING_MAX_STRING_CHARS = 160


PERSISTENT_TRACKING_METADATA_FIELDS = (
    "camera",
    "coordinate_system",
    "depth_unit",
    "xyz_in_robot_base_coord_axes",
    "xyz_in_robot_base_coord_frame",
)


PERSISTENT_TRACKING_TEXT_FIELDS = (
    "camera",
    "episode_id",
    "source_image_id",
    "source_session_id",
    "status",
    "track_id",
)


PERSISTENT_TRACKING_BOOL_FIELDS = (
    "identity_verified",
    "lost",
    "valid",
)


PERSISTENT_TRACKING_NUMBER_FIELDS = (
    "confidence",
    "depth_m",
    "observation_sequence",
    "source_observation_sequence",
    "u",
    "v",
)


def _tracking_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:PERSISTENT_TRACKING_MAX_STRING_CHARS] if text else None


def _tracking_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tracking_xyz(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    coordinates = [_tracking_number(item) for item in value]
    if any(item is None for item in coordinates):
        return None
    return [item for item in coordinates if item is not None]


def _compact_persistent_tracking(memory: Any) -> dict[str, Any]:
    """Select only live point identity, binding, status, and coordinates."""
    if not isinstance(memory, dict):
        raise ValueError("/api/memory response must be an object")
    raw = memory.get("raw", memory)
    if not isinstance(raw, dict):
        raise ValueError("/api/memory raw field must be an object")

    compact: dict[str, Any] = {"available": True}
    metadata = raw.get("tracked_object_distance_tracking")
    if isinstance(metadata, dict):
        for field in PERSISTENT_TRACKING_METADATA_FIELDS:
            value = _tracking_text(metadata.get(field))
            if value is not None:
                compact[field] = value

    raw_tracks = raw.get("tracked_object_distances")
    if raw_tracks is None:
        raw_tracks = {}
    if not isinstance(raw_tracks, dict):
        raise ValueError("tracked_object_distances must be an object")

    tracks: dict[str, Any] = {}
    named_tracks = [
        (name, track)
        for name, track in raw_tracks.items()
        if isinstance(name, str) and isinstance(track, dict)
    ]
    named_tracks.sort(key=lambda item: item[0])
    for raw_name, track in named_tracks[:PERSISTENT_TRACKING_MAX_TRACKS]:
        name = _tracking_text(raw_name)
        if name is None:
            continue
        selected: dict[str, Any] = {}
        for field in PERSISTENT_TRACKING_TEXT_FIELDS:
            value = _tracking_text(track.get(field))
            if value is not None:
                selected[field] = value
        for field in PERSISTENT_TRACKING_BOOL_FIELDS:
            value = track.get(field)
            if isinstance(value, bool):
                selected[field] = value
        for field in PERSISTENT_TRACKING_NUMBER_FIELDS:
            value = _tracking_number(track.get(field))
            if value is not None:
                selected[field] = value
        xyz = _tracking_xyz(track.get("xyz_in_robot_base_coord_m"))
        if xyz is not None:
            selected["xyz_in_robot_base_coord_m"] = xyz
        tracks[name] = selected

    compact["tracks"] = tracks
    if len(named_tracks) > PERSISTENT_TRACKING_MAX_TRACKS:
        compact["total_track_count"] = len(named_tracks)
        compact["tracks_truncated"] = True

    # 地图地点和 track 点走同一条 persistent_tracking 通道
    compact["map_marks"] = _compact_map_marks(raw.get("marked_places"))
    return compact


PERSISTENT_MAP_MARK_MAX = 16


def _compact_map_marks(raw_marks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_marks, list):
        return []
    marks: list[dict[str, Any]] = []
    for item in raw_marks[:PERSISTENT_MAP_MARK_MAX]:
        if not isinstance(item, dict):
            continue
        name = _tracking_text(item.get("name"))
        if name is None:
            continue
        mark: dict[str, Any] = {"name": name}
        direction = _tracking_text(item.get("direction"))
        if direction is not None:
            mark["direction"] = direction
        for field in ("range_m", "spin_deg_to_face"):
            value = _tracking_number(item.get(field))
            if value is not None:
                mark[field] = value
        marks.append(mark)
    return marks
