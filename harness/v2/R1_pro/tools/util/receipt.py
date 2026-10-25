"""Model-visible results contain measured evidence, not a generic `ok` flag.

The CLI/hook presentation is smaller than the immutable result.json. Physical
measurements, errors and nonempty tracks retain their values. Camera guidance
is short and explicit; static audit/transport metadata stays in the full file.
"""
import json
from pathlib import Path


def _relative_path(value, workspace):
    if isinstance(value, str) and workspace is not None:
        try:
            return str(Path(value).relative_to(workspace))
        except ValueError:
            pass  # Paths outside this workspace must remain resolvable.
    return value


def image_instruction(media, latest_head):
    """One explicit coordinate/role instruction for a known image kind."""
    feed, role = media.get("feed"), media.get("role")
    image_id = media.get("image_id", "")
    if feed == "head" and role in {"raw_rgb", "path_overlay"} and (
            media.get("width"), media.get("height")) == (720, 720):
        if not image_id:
            return "No usable head image ID; capture a new head image before clicking."
        return (f"Head {image_id}: 720 x 720. Re-locate in this newest image; click in ORIGINAL PIXELS: "
                "u=column, v=row; top-left (0,0), bottom-right (719,719), center (360,360). "
                "Ignore old points and overlay labels.")
    if role == "planning_overlay":
        return f"Plan preview only; not clickable. Click latest head {latest_head}."
    if feed in {"left_wrist", "right_wrist"}:
        return f"{feed} {image_id}: observation only; click the latest head image {latest_head}."
    if feed == "proprioception":
        return "Offline posture render; no scene/held objects and not clickable. Refresh on demand."
    return media.get("caption")  # Unknown views keep their original semantics.


def _compact_metadata(result):
    response = result.get("response")
    if isinstance(response, dict):
        for key in ("tool", "tool_name"):
            if response.get(key) == result.get("tool_name"):
                response.pop(key, None)
        # Only known boilerplate values are omitted. Unexpected diagnostics
        # remain visible, and no physical completion/guard field is selected out.
        boilerplate = {
            "tool_version": "official_v2",
            "global_base_pose_truth": False,
            "dynamic_collision_truth": False,
            "forbidden_observation_reads": [],
            "forbidden_modalities_omitted": ["segmentation", "normal", "object_pose",
                                             "scene_graph", "BDDL_live_state", "simulator_mesh"],
            "observation_contract": ["base_qvel_proprioception", "action_history", "static_r1pro_action_limits"],
        }
        for key, value in boilerplate.items():
            if response.get(key) == value:
                response.pop(key, None)
        # Explanations of implementation/provenance belong in the full record
        # and readable tool source. Never drop a measured value, error, policy
        # failure, or unknown field by a suffix/wildcard rule.
        if result.get("record"):
            prose = {
                "adjust_chassis": {"verification", "landing_verification", "linear_control"},
                "close_gripper": {"bilateral_effort_policy", "precontact_effort_policy",
                    "contact_evidence_policy", "confirmation_source", "actuation_evidence_source",
                    "implementation", "control_mode"},
                "open_gripper": {"implementation", "control_mode", "actuation_evidence_source"},
                "set_arm_to_grasp_position": {"terminal_hold_source", "stationarity_velocity_sources"},
                "track_object_distance": {"deletion_policy", "replacement_policy", "frame_digest",
                    "frame_binding", "frame_alignment", "memory_key", "source"},
                "exec_plan_pose": {"trajectory_digest"},
                "plan_grasp_point_filter_rgbd_lite": {"trajectory_digest"},
            }
            for key in prose.get(result.get("tool_name"), ()):
                if isinstance(response.get(key), str):
                    response.pop(key)
            # These are audit labels shared by the known pose tools, not the
            # measured translation, rotation, IK outcome or contact evidence.
            if result.get("tool_name") in {
                    "adjust_left_eef_pose_in_head_frame", "adjust_right_eef_pose_in_head_frame",
                    "adjust_left_eef_pose_in_wrist_frame", "adjust_right_eef_pose_in_wrist_frame"}:
                for key in ("build", "kinematics_source"):
                    if isinstance(response.get(key), str):
                        response.pop(key)
    tracking = result.get("persistent_tracking")
    known_tracking = {"available", "camera", "coordinate_system", "depth_unit", "tracks", "map_marks",
                      "xyz_in_robot_base_coord_axes", "xyz_in_robot_base_coord_frame"}
    if (isinstance(tracking, dict) and tracking.get("available") is True
            and tracking.get("tracks") == {} and not tracking.get("map_marks")
            and set(tracking) <= known_tracking):
        result["persistent_tracking"] = {"available": True, "tracks": {}}
    budget = result.get("rollout_budget")
    if isinstance(budget, dict):
        for key, value in (("source", "official_evaluator_episode_steps"),
                           ("sampling", "latest_evaluator_observation")):
            if budget.get(key) == value:
                budget.pop(key, None)


def model_receipt(receipt, workspace=None):
    """A stateless projection of decision evidence for CLI and hooks.

    No delta encoding: each result remains interpretable after compaction.
    Do not round numbers, truncate errors/arrays, or prune physical evidence.
    The saved receipt retains absolute paths and hashes used to verify pixels.
    """
    result = without_ok(receipt)
    if workspace is None and isinstance(result.get("record"), str):
        record = Path(result["record"])
        if record.is_absolute() and record.parent.parent.name == "tool_call_history":
            workspace = record.parent.parent.parent
    if workspace is not None:
        workspace = Path(workspace)
    result.pop("duration_s", None)
    _compact_metadata(result)
    if "observation_instruction" in result:
        if result.get("images"):
            result["observation_instruction"] = "If no pixels are attached, open images[].path with native Read."
        else:
            result.pop("observation_instruction")
    for key in ("record", "perception", "skill_path"):
        if key in result:
            result[key] = _relative_path(result[key], workspace)
    for media in result.get("images", []):
        # Hash/MIME validation happens on the full saved receipt, not this text.
        for key in ("sha256", "bytes", "mime_type"):
            media.pop(key, None)
        for dimension in ("width", "height"):
            source = "source_" + dimension
            if source in media and media[source] == media.get(dimension):
                media.pop(source)
        if "path" in media:
            media["path"] = _relative_path(media["path"], workspace)
        caption = image_instruction(media, result.get("latest_clickable_image_id", ""))
        if caption is not None:
            media["caption"] = caption
    return result


def model_receipt_text(receipt, workspace=None):
    return json.dumps(model_receipt(receipt, workspace), ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def without_ok(value):
    """Copy a result, removing `ok` at every depth without changing other facts.

    Internal controllers may still use that flag for their own branching.
    Apply this only after control/image selection, at the public boundary.
    """
    if isinstance(value, dict):
        return {key: without_ok(item) for key, item in value.items() if key != "ok"}
    if isinstance(value, (list, tuple)):
        return [without_ok(item) for item in value]
    return value


def tool_error(receipt):
    """Read explicit errors; support older receipts in operator checks.

    No error is NOT evidence of reaching a pose, completing a displacement,
    making contact or grasping an object. Check those measurements separately.
    """
    if not isinstance(receipt, dict):
        return "Invalid tool receipt"
    response = receipt.get("response")
    response = response if isinstance(response, dict) else {}
    error = receipt.get("error") or response.get("error")
    if error:
        return str(error)
    if receipt.get("ok") is False or response.get("ok") is False:
        return "Legacy tool reported failure; inspect its recorded evidence"
    return None


def camera_observation_ready(receipt):
    """Startup needs actual camera evidence, not merely absence of an error."""
    return (isinstance(receipt, dict) and not tool_error(receipt)
            and receipt.get("tool_name") in {
                "capture_head_camera", "capture_left_wrist_camera", "capture_right_wrist_camera"}
            and bool(receipt.get("images")) and bool(receipt.get("perception"))
            and bool(receipt.get("latest_clickable_image_id")))
