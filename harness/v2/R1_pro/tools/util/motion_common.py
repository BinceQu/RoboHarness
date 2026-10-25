"""Shared robot functions used by: adjust_chassis, adjust_height, adjust_left_eef_pose_in_head_frame, adjust_left_eef_pose_in_wrist_frame, adjust_pitch, adjust_right_eef_pose_in_head_frame, adjust_right_eef_pose_in_wrist_frame, capture_head_camera, capture_left_wrist_camera, capture_right_wrist_camera, close_gripper, exec_plan_pose, measure_shoulder_distance, move_chassis_to_directly_facing_surface, move_chassis_to_floor_point, move_to_reach_point, move_tracked_point, open_gripper, plan_grasp_point_filter_rgbd_lite, plan_press_point, reset_body, set_arm_to_grasp_position, spin_to_facing_point, track_object_distance, tracked_point_planning_worker.

All commands and geometry execute locally; no macro RPC or external project.
"""
from __future__ import annotations

from dataclasses import dataclass
from tools.util.action_constants import ADJUST_CHASSIS_BASE_MAX_ANG_RADPS
from tools.util.action_constants import ADJUST_CHASSIS_BASE_MAX_LIN_MPS
from tools.util.action_constants import ADJUST_CHASSIS_FORWARD_TOL_M
from tools.util.action_constants import ADJUST_CHASSIS_LINEAR_DECEL_MPS2
from tools.util.action_constants import ADJUST_CHASSIS_LINEAR_FINAL_SPEED_MPS
from tools.util.action_constants import ADJUST_CHASSIS_LINEAR_NEAR_SPEED_MPS
from tools.util.action_constants import ADJUST_CHASSIS_MIN_QVEL_MPS
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_INITIAL_MPS
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_MARGIN_M
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_MAX_M
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_MAX_MPS
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_MIN_M
from tools.util.action_constants import ADJUST_CHASSIS_RETREAT_STALL_STEPS
from tools.util.action_constants import ADJUST_CHASSIS_SETTLE_SPEED_MPS
from tools.util.action_constants import ADJUST_CHASSIS_SETTLE_STEPS
from tools.util.action_constants import ADJUST_CHASSIS_SETTLE_YAW_RATE_RADPS
from tools.util.action_constants import ADJUST_CHASSIS_SPIN_GAIN
from tools.util.action_constants import ADJUST_CHASSIS_STALL_RATIO
from tools.util.action_constants import ADJUST_CHASSIS_STALL_STEPS
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_GOOD_PROGRESS_RATIO
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_GOOD_TRACKING_ERROR_RAD
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_MAX_SCALE
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_MIN_SCALE
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_NEAR_ORIENTATION_DEG
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_NEAR_POSITION_M
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_POOR_PROGRESS_RATIO
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_POOR_TRACKING_ERROR_RAD
from tools.util.action_constants import ADJUST_EEF_ADAPTIVE_SCALE_INCREMENT
from tools.util.action_constants import ADJUST_EEF_BASE_DISTURBANCE_STEPS
from tools.util.action_constants import ADJUST_EEF_BASE_IMMEDIATE_LINEAR_MPS
from tools.util.action_constants import ADJUST_EEF_BASE_IMMEDIATE_YAW_RAD_S
from tools.util.action_constants import ADJUST_EEF_BASE_LINEAR_DISTURBANCE_MPS
from tools.util.action_constants import ADJUST_EEF_BASE_TRANSLATION_LIMIT_M
from tools.util.action_constants import ADJUST_EEF_BASE_YAW_DISTURBANCE_RAD_S
from tools.util.action_constants import ADJUST_EEF_BASE_YAW_LIMIT_RAD
from tools.util.action_constants import ADJUST_EEF_FAILURE_SETTLE_STEPS
from tools.util.action_constants import ADJUST_EEF_GROSS_PREFLIGHT_MAX_NORMALIZED_ERROR
from tools.util.action_constants import ADJUST_EEF_GUIDED_JOINT_TRACKING_MULTIPLIER
from tools.util.action_constants import ADJUST_EEF_GUIDED_RELINEARIZATION_MULTIPLIER
from tools.util.action_constants import ADJUST_EEF_GUIDED_STEP_SCALE
from tools.util.action_constants import ADJUST_EEF_LOCAL_BUILD
from tools.util.action_constants import ADJUST_EEF_MAX_TRACKING_ERROR_RAD
from tools.util.action_constants import ADJUST_EEF_MIN_COMMAND_RAD
from tools.util.action_constants import ADJUST_EEF_MIN_PROGRESS_RATIO
from tools.util.action_constants import ADJUST_EEF_NEAR_ZERO_COMMAND_RAD
from tools.util.action_constants import ADJUST_EEF_NEAR_ZERO_COMMAND_STEPS
from tools.util.action_constants import ADJUST_EEF_ORIENTATION_DIVERGENCE_MARGIN_DEG
from tools.util.action_constants import ADJUST_EEF_ORIENTATION_DIVERGENCE_TOL_MULTIPLIER
from tools.util.action_constants import ADJUST_EEF_POSE_PROGRESS_EPS
from tools.util.action_constants import ADJUST_EEF_POSE_STALL_STEPS
from tools.util.action_constants import ADJUST_EEF_RELINEARIZATION_MULTIPLIER
from tools.util.action_constants import ADJUST_EEF_SETTLED_QVEL_RAD_S
from tools.util.action_constants import ADJUST_EEF_SETTLE_MAX_STEPS
from tools.util.action_constants import ADJUST_EEF_SETTLE_MIN_STEPS
from tools.util.action_constants import ADJUST_EEF_STALL_STEPS
from tools.util.action_constants import ADJUST_EEF_SUCCESS_SETTLE_MAX_STEPS
from tools.util.action_constants import ADJUST_EEF_TRUNK_DRIFT_LIMIT_RAD
from tools.util.action_constants import ADJUST_HEIGHT_REQUIRED_STABLE_STEPS
from tools.util.action_constants import ADJUST_HEIGHT_STABLE_QVEL_RAD_S
from tools.util.action_constants import ADJUST_HEIGHT_TRACKING_STALL_STEPS
from tools.util.action_constants import ADJUST_HEIGHT_WAYPOINT_ADVANCE_TOLERANCE_RAD
from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import DEFAULT_TRAJECTORY_STEP_RAD
from tools.util.action_constants import EXEC_FINAL_PAIR_ORI_TOL_DEG
from tools.util.action_constants import EXEC_FINAL_PAIR_POS_TOL_M
from tools.util.action_constants import EXEC_SAFE_BACK_MAX_M
from tools.util.action_constants import EXEC_SAFE_BACK_MIN_M
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA_VERSION
from tools.util.action_constants import LOCKED_J8_CAPTURE_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_INDEX
from tools.util.action_constants import LOCKED_J8_SERIALIZED_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_VALUE_RAD
from tools.util.action_constants import MAX_PLAN_RECORD_BYTES
from tools.util.action_constants import MAX_TRAJECTORY_WAYPOINTS
from tools.util.action_constants import MOVE_POINT_TO_POINT_MAX_TRANSLATION_M
from tools.util.action_constants import MOVE_POINT_TO_POINT_STALL_POSITION_DELTA_RAD
from tools.util.action_constants import MOVE_POINT_TO_POINT_STALL_QVEL_RAD_S
from tools.util.action_constants import MOVE_TRACKED_OVERLAP_FALLBACK_POLICY
from tools.util.action_constants import MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_ANGLE_OFFSETS_DEG
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_BASE_COUNT
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_ANGLE_OFFSETS_DEG
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_BASE_COUNT
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_MAX_IK_TARGETS
from tools.util.action_constants import MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_MAX_IK_TARGETS
from tools.util.action_constants import NAVIGATE_TO_LINEAR_DECEL_MPS2
from tools.util.action_constants import NAVIGATE_TO_SPIN_GAIN
from tools.util.action_constants import TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
from tools.util.action_constants import TRAJECTORY_STEP_VALIDATION_EPS_RAD
from tools.util.action_constants import TRUNK_LIMITS
from tools.util.action_constants import _ARM_AXES
from tools.util.action_constants import _ARM_BASE_ORIGINS
from tools.util.action_constants import _ARM_JOINT_ORIGINS
from tools.util.action_constants import _ARM_LIMITS
from tools.util.action_constants import _CAPTURE_TRACE_ENABLED
from tools.util.action_constants import _PLAN_ID_RE
from tools.util.action_constants import _SESSION_ID_RE
from tools.util.action_constants import _SESSION_LOCK
from tools.util.geometry.capabilities import TOOL_VERSION
from tools.util.geometry.contract import ACTION_DIM
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.contract import MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS
from tools.util.geometry.contract import PROPRIO_SLICES
from tools.util.geometry.contract import ROBOT_PROFILE
from tools.util.geometry.contract import coordinate_collinear_relations
from tools.util.geometry.dynamic_point_tracker import TrackerFrame
from tools.util.geometry.dynamic_point_tracker import transform_from_pose
from tools.util.geometry.eef_adjustment_local import ADJUST_POSE_MAX_JOINT_STEP_RAD
from tools.util.geometry.eef_adjustment_local import MAX_ORIENTATION_STEP_RAD
from tools.util.geometry.eef_adjustment_local import MAX_TRANSLATION_STEP_M
from tools.util.geometry.eef_adjustment_local import WRIST_MAX_JOINT_STEP_RAD
from tools.util.geometry.eef_adjustment_local import bounded_pose_goal_step
from tools.util.geometry.eef_adjustment_local import bounded_pose_tangent_step
from tools.util.geometry.eef_adjustment_local import camera_frame_pose_target
from tools.util.geometry.eef_adjustment_local import local_robot_state
from tools.util.geometry.eef_adjustment_local import locked_arm_vector
from tools.util.geometry.eef_adjustment_local import numerical_eef_jacobian
from tools.util.geometry.eef_adjustment_local import orientation_error_deg
from tools.util.geometry.eef_adjustment_local import orientation_error_vector
from tools.util.geometry.eef_adjustment_local import robot_base_frame_pose_target
from tools.util.geometry.eef_adjustment_local import solve_j567_orientation_target
from tools.util.geometry.eef_adjustment_local import solve_pose_target
from tools.util.geometry.grasp_kinematics_local import LocalRobotState
from tools.util.geometry.grasp_kinematics_local import arm_joint_limits as _local_arm_joint_limits
from tools.util.geometry.grasp_kinematics_local import eef_pose as _local_eef_pose
from tools.util.geometry.grasp_prediction_audit_local import record_compliant_grasp_prediction
from tools.util.geometry.rgbd_grasp_lite import LITE_POSE_OVERLAP_FILTER_BUILD
from tools.util.geometry.rgbd_grasp_lite import evaluate_pose_overlap_volumes_lite as _evaluate_rgbd_lite_pose_overlaps
from tools.util.geometry.rgbd_grasp_planner import INFLATED_OVERLAP_MAX_CM3
from tools.util.geometry.rgbd_grasp_planner import ORIGINAL_OVERLAP_MAX_CM3
from tools.util.geometry.rgbd_projective_occupancy import ProjectiveOccupancyContractError
from tools.util.geometry.tracked_object_distance import head_camera_intrinsics
from tools.util.geometry.tracked_object_distance import tracker_frame_content_digest
from tools.util.geometry.tracked_point_constraints_local import evaluate_relations as _evaluate_tracked_relations
from tools.util.geometry.tracked_point_constraints_local import geometry_rank as _tracked_geometry_rank
from tools.util.geometry.tracked_point_constraints_local import quick_constraint_relation as _quick_constraint_relation
from tools.util.geometry.tracked_point_motion_local import ENDPOINT_PATH_FALLBACK_ORIENTATION_REGULARIZATION_WEIGHT
from tools.util.geometry.tracked_point_motion_local import PlanningDeadlineExceeded
from tools.util.geometry.tracked_point_motion_local import plan_cartesian_trajectory as _plan_tracked_cartesian_trajectory
from tools.util.geometry.tracked_point_motion_local import plan_endpoint as _plan_tracked_endpoint
from tools.util.geometry.tracked_point_motion_local import plan_joint_space_trajectory as _plan_tracked_joint_space_trajectory
from tools.util.geometry.tracked_point_motion_local import shared_line_target_pose_family as _shared_line_target_pose_family
from tools.util.geometry.tracked_point_planning_worker import run_plan_worker as _run_tracked_plan_worker
from tools.util.geometry.trunk_vertical_lift_local import TRUNK_LUT_HOLD_ACTIONS
from tools.util.geometry.trunk_vertical_lift_local import TRUNK_LUT_MAX_EXEC_WAYPOINTS
from tools.util.geometry.trunk_vertical_lift_local import chest_height_lut_frame_m
from tools.util.geometry.trunk_vertical_lift_local import plan_absolute_height_trajectory
from tools.util.geometry.trunk_vertical_lift_local import sample_lut_execution_waypoints
from tools.util.geometry.trunk_vertical_lift_local import torso_theta_z_deg
from tools.util.settings import plan_key
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence
import base64
import cv2
import hashlib
import hmac
import json
import math
import numpy as np
import os
import re
import threading
import time


def _capture_trace(stage: str, **fields: Any) -> None:
    if not _CAPTURE_TRACE_ENABLED:
        return
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "thread": threading.get_native_id(),
        "module": "tools",
        "stage": str(stage),
        **fields,
    }
    try:
        with open(
            f"/tmp/official_capture_trace_{os.getpid()}.jsonl",
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass



@dataclass
class _FrozenCapture:
    session_id: str
    image_id: str
    role: str
    depth: np.ndarray
    camera: dict[str, Any]
    rgb: np.ndarray | None = None
    robot: dict[str, Any] | None = None
    evaluator_sequence: int | None = None



def _check_cancelled(ctx, where: str) -> None:
    fn = getattr(ctx, "raise_if_cancelled", None)
    if callable(fn):
        fn(where)



def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)



def _set_result(ctx, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    _capture_trace(
        "set_result_start",
        tool=tool_name,
        keys=len(payload) if isinstance(payload, Mapping) else -1,
        image_id=str(payload.get("image_id") or "") if isinstance(payload, Mapping) else "",
        rgb_chars=(
            len(str(payload.get("rgb_main") or ""))
            if isinstance(payload, Mapping)
            else 0
        ),
    )
    result = dict(payload)
    if (
        str(result.get("image_id") or "").strip()
        and any(result.get(key) for key in ("rgb_main", "rgb_main_path", "rgb_path"))
        and "track_object_distance_binding" not in result
    ):
        feed = str(result.get("feed") or "").strip().lower()
        reason = (
            "unsupported_camera_role"
            if feed and feed not in ("head", "main")
            else "binding_not_registered"
        )
        result["track_object_distance_binding"] = {
            "ok": False,
            "trackable": False,
            "reason": reason,
            "image_id": str(result.get("image_id") or "").strip(),
            "error": (
                "track_object_distance accepts only an exactly bound frozen "
                "head RGB-D capture"
            ),
        }
    result["tool"] = str(tool_name)
    result["tool_version"] = TOOL_VERSION
    # Tool diagnostics may use +/-inf internally to rank unavailable or
    # unobserved candidates.  Public results cross a strict JSON boundary,
    # where non-finite numbers are invalid and break browser JSON.parse().
    result = _json_ready(result)
    record_compliant_grasp_prediction(tool_name, result)
    # Returning the normalized payload is useful to the detached capture
    # waiter.  Existing callers ignore the return value, so this is a strictly
    # additive extension of the old result callback contract.
    installed = ctx.set_result(result)
    _capture_trace(
        "set_result_done",
        tool=tool_name,
        elapsed_ms=0.0,
        installed_mapping=isinstance(installed, Mapping),
    )
    if isinstance(installed, Mapping):
        return dict(installed)
    return result



def _runs_root() -> str:
    return os.environ["ROBOT_RECORDINGS"]



def _run_dir(session_id: str) -> str:
    return os.path.join(_runs_root(), _validated_session_id(session_id))



def _images_dir(session_id: str) -> str:
    return os.path.join(_run_dir(session_id), "images")



def _plans_dir(session_id: str) -> str:
    return os.path.join(_run_dir(session_id), "plans")



def _validated_session_id(session_id: str) -> str:
    session = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(session):
        raise ValueError(
            "session_id must contain only letters, digits, '_', '-', or '.'"
        )
    return session



def _validated_plan_id(plan_id: str) -> str:
    plan = str(plan_id or "").strip()
    if not _PLAN_ID_RE.fullmatch(plan):
        raise ValueError("plan_id must match plan_<letters-or-digits>")
    return plan



def _plan_record_path(session_id: str, plan_id: str) -> str:
    session = _validated_session_id(session_id)
    plan = _validated_plan_id(plan_id)
    root = os.path.realpath(_plans_dir(session))
    path = os.path.abspath(os.path.join(root, f"{plan}.json"))
    if os.path.commonpath((root, path)) != root:
        raise ValueError("plan path escapes the policy-owned plans directory")
    return path



def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")



def _trajectory_digest(trajectory: dict[str, Any]) -> str:
    unsigned = dict(trajectory)
    unsigned.pop("integrity", None)
    return hmac.new(
        plan_key(),
        _canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()



def _write_plan_record(path: str, record: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    encoded = _canonical_json_bytes(_json_ready(record))
    if not 1 <= len(encoded) <= MAX_PLAN_RECORD_BYTES:
        raise ValueError(
            "plan record size must be in "
            f"1..{MAX_PLAN_RECORD_BYTES} bytes (encoded size: {len(encoded)} bytes)"
        )
    try:
        with open(temp_path, "xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)



def _session_meta_path(session_id: str) -> str:
    return os.path.join(_run_dir(session_id), "session.json")



def _load_session_meta(session_id: str) -> dict[str, Any]:
    path = _session_meta_path(session_id)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    return {
        "session_id": str(session_id),
        "image_counter": 0,
        "plan_counter": 0,
    }



def _save_session_meta(session_id: str, meta: dict[str, Any]) -> None:
    os.makedirs(_run_dir(session_id), exist_ok=True)
    with open(_session_meta_path(session_id), "w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, ensure_ascii=False)



def _ensure_session(session_id: str) -> None:
    with _SESSION_LOCK:
        os.makedirs(_images_dir(session_id), exist_ok=True)
        os.makedirs(_plans_dir(session_id), exist_ok=True)
        _save_session_meta(session_id, _load_session_meta(session_id))



def _next_image_id(session_id: str) -> str:
    with _SESSION_LOCK:
        meta = _load_session_meta(session_id)
        meta["image_counter"] = int(meta.get("image_counter", 0)) + 1
        _save_session_meta(session_id, meta)
        return f"img_{meta['image_counter']:04d}"



def _next_plan_id(session_id: str) -> str:
    with _SESSION_LOCK:
        meta = _load_session_meta(session_id)
        meta["plan_counter"] = int(meta.get("plan_counter", 0)) + 1
        _save_session_meta(session_id, meta)
        return f"plan_{meta['plan_counter']:04d}"



def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def _arm_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != ARM_DOF:
        raise ValueError(
            f"{label} must contain {ARM_DOF} arm values, got {vector.size}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector



def _j8_locked_copy(value: Any, *, label: str) -> np.ndarray:
    vector = _arm_vector(value, label=label).copy()
    if LOCKED_J8_INDEX is not None:
        vector[LOCKED_J8_INDEX] = LOCKED_J8_VALUE_RAD
    return vector



def _require_serialized_j8_zero(value: Any, *, label: str) -> np.ndarray:
    vector = _arm_vector(value, label=label)
    if (
        LOCKED_J8_INDEX is not None
        and abs(float(vector[LOCKED_J8_INDEX]) - LOCKED_J8_VALUE_RAD)
        > LOCKED_J8_SERIALIZED_TOLERANCE_RAD
    ):
        raise ValueError(
            f"{label} contains non-zero locked J8; re-plan after episode initialization"
        )
    return vector



def _normalize_arm_q_to_joint_limits(
    value: Any,
    *,
    arm: str,
    label: str,
    point: str,
    recover_observed_overshoot: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """把关节角收到本地位，并拒绝规划点上的真实越界。

    recover_observed_overshoot=True 只给 evaluator 实时观测值使用：观测值
    可能因控制残差轻微越过静态限位，但 action 仍必须严格位于限位内。
    规划得到的 safe_q / final_q 必须继续严格拒绝真实越界。
    """
    q = _arm_vector(value, label=label).copy()
    limits = _ARM_LIMITS[arm][:ARM_DOF]
    below = limits[:, 0] - q
    above = q - limits[:, 1]
    violation = np.maximum(np.maximum(below, above), 0.0)
    if (
        bool(np.any(violation > TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD))
        and not recover_observed_overshoot
    ):
        joint_index = int(np.argmax(violation))
        raise ValueError(
            f"{label} exceeds local arm joint limits: arm={arm} "
            f"J{joint_index + 1}={q[joint_index]:.12g}rad outside "
            f"[{limits[joint_index, 0]:.12g},"
            f"{limits[joint_index, 1]:.12g}] by "
            f"{violation[joint_index]:.12g}rad; numeric tolerance is "
            f"{TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD:.12g}rad"
        )

    clipped = np.clip(q, limits[:, 0], limits[:, 1])
    normalizations = []
    for joint_index in np.flatnonzero(clipped != q):
        index = int(joint_index)
        item = {
            "point": str(point),
            "joint": f"J{index + 1}",
            "before_rad": float(q[index]),
            "after_rad": float(clipped[index]),
            "correction_rad": float(clipped[index] - q[index]),
        }
        if (
            recover_observed_overshoot
            and float(violation[index])
            > TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
        ):
            item["recovered_observed_overshoot"] = True
        normalizations.append(item)
    return clipped, normalizations



def _normalize_trunk_q_to_joint_limits(
    value: Any,
    *,
    label: str,
    point: str,
    recover_observed_overshoot: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Normalize the four trunk joints without accepting an invalid plan.

    Evaluator proprioception can contain float32-scale residuals at a static
    limit, including the locked fourth trunk joint. Planned values only get
    the shared numeric serialization tolerance; live observations may be
    recovered by callers after they enforce their bounded-action limit.
    """

    try:
        q = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain 4 trunk values") from exc
    if q.size != 4:
        raise ValueError(f"{label} must contain 4 trunk values, got {q.size}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{label} contains non-finite values")
    q = q.copy()
    limits = np.asarray(TRUNK_LIMITS, dtype=np.float64).reshape(4, 2)
    below = limits[:, 0] - q
    above = q - limits[:, 1]
    violation = np.maximum(np.maximum(below, above), 0.0)
    if (
        bool(np.any(violation > TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD))
        and not recover_observed_overshoot
    ):
        joint_index = int(np.argmax(violation))
        raise ValueError(
            f"{label} exceeds local trunk joint limits: "
            f"J{joint_index + 1}={q[joint_index]:.12g}rad outside "
            f"[{limits[joint_index, 0]:.12g},"
            f"{limits[joint_index, 1]:.12g}] by "
            f"{violation[joint_index]:.12g}rad; numeric tolerance is "
            f"{TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD:.12g}rad"
        )

    clipped = np.clip(q, limits[:, 0], limits[:, 1])
    normalizations = []
    for joint_index in np.flatnonzero(clipped != q):
        index = int(joint_index)
        item = {
            "point": str(point),
            "joint": f"J{index + 1}",
            "before_rad": float(q[index]),
            "after_rad": float(clipped[index]),
            "correction_rad": float(clipped[index] - q[index]),
        }
        if (
            recover_observed_overshoot
            and float(violation[index])
            > TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
        ):
            item["recovered_observed_overshoot"] = True
        normalizations.append(item)
    return clipped, normalizations



def _require_signed_trunk_q(
    value: Any,
    *,
    label: str,
    point: str,
) -> np.ndarray:
    """签名轨迹里的躯干只收 1e-6 数值残差，真越界仍拒绝。"""

    trunk, _normalizations = _normalize_trunk_q_to_joint_limits(
        value,
        label=label,
        point=point,
        recover_observed_overshoot=False,
    )
    return trunk



def _require_episode_initialized(ctx, operation: str) -> None:
    initialized = getattr(ctx.world, "episode_initialized", None)
    if callable(initialized) and not bool(initialized()):
        raise ValueError(
            f"episode grasp-prep initialization is incomplete; {operation} is blocked"
        )



def _interpolate_joint_segment(
    start_q: np.ndarray,
    target_q: np.ndarray,
    *,
    max_step_rad: float = DEFAULT_TRAJECTORY_STEP_RAD,
) -> list[list[float]]:
    start = _arm_vector(start_q, label="segment start")
    target = _arm_vector(target_q, label="segment target")
    max_step = float(max_step_rad)
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step_rad must be positive and finite")
    distance = float(np.linalg.norm(target - start, ord=np.inf))
    # Smoothstep's maximum derivative is 1.5, so its midpoint samples move
    # faster than linear interpolation.
    steps = max(1, int(math.ceil(1.5 * distance / max_step)))
    return [
        (
            start
            + _smoothstep(index / steps) * (target - start)
        ).astype(float).tolist()
        for index in range(1, steps + 1)
    ]



def _compile_exec_arm_segments(
    start_q: Sequence[float],
    safe_q: Sequence[float],
    final_q: Sequence[float],
) -> list[dict[str, Any]]:
    """Build the one canonical arm-only current/safe/final exec trajectory."""

    start = _require_serialized_j8_zero(start_q, label="exec segment start_q")
    safe = _require_serialized_j8_zero(safe_q, label="exec segment safe_q")
    final = _require_serialized_j8_zero(final_q, label="exec segment final_q")
    segment_specs = (
        ("tool_roll_reset", start, start),
        ("current_to_safe", start, safe),
        ("safe_to_final", safe, final),
    )
    segments: list[dict[str, Any]] = []
    for name, segment_start, segment_target in segment_specs:
        segments.append(
            {
                "name": name,
                "waypoints": _interpolate_joint_segment(
                    segment_start,
                    segment_target,
                ),
                "endpoint_q": segment_target.astype(float).tolist(),
            }
        )
    return segments



def _plan_start_local_state(start_state: dict[str, Any]) -> LocalRobotState:
    arm_qpos = dict(start_state.get("arm_qpos") or {})
    gripper_qpos = dict(start_state.get("gripper_qpos") or {})
    return local_robot_state(
        trunk_q=start_state.get("trunk_qpos"),
        arm_left_q=arm_qpos.get("left"),
        arm_right_q=arm_qpos.get("right"),
        gripper_left_q=gripper_qpos.get("left"),
        gripper_right_q=gripper_qpos.get("right"),
    )



def _validate_exec_final_q_for_arm(
    *,
    trajectory: dict[str, Any],
    arm: str,
    raw_q: Any,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify a stored arm branch reaches the plan's immutable final target."""
    requested_arm = str(arm).strip().lower()
    planned_arm = str(trajectory["active_arm"])
    if requested_arm not in ("left", "right"):
        raise ValueError("requested execution arm must be left or right")
    candidate_q = _require_serialized_j8_zero(
        raw_q,
        label=f"{source} {requested_arm} final_q",
    )
    candidate_q, normalizations = _normalize_arm_q_to_joint_limits(
        candidate_q,
        arm=requested_arm,
        label=f"{source} {requested_arm} final_q",
        point="exec_final_q",
    )
    planned_q = _require_serialized_j8_zero(
        trajectory["segments"][2]["endpoint_q"],
        label=f"signed {planned_arm} final endpoint_q",
    )
    state = _plan_start_local_state(dict(trajectory["start_state"]))
    planned_position, planned_quaternion = _local_eef_pose(
        state,
        planned_arm,
        planned_q,
    )
    candidate_position, candidate_quaternion = _local_eef_pose(
        state,
        requested_arm,
        candidate_q,
    )
    pair_position_error = float(
        np.linalg.norm(candidate_position - planned_position)
    )
    pair_orientation_error = float(
        orientation_error_deg(candidate_quaternion, planned_quaternion)
    )
    if (
        pair_position_error > EXEC_FINAL_PAIR_POS_TOL_M
        or pair_orientation_error > EXEC_FINAL_PAIR_ORI_TOL_DEG
    ):
        raise ValueError(
            f"stored {requested_arm} final_q does not reach the planned final "
            f"EEF pose: pair position error {pair_position_error:.4f}m "
            f"(limit {EXEC_FINAL_PAIR_POS_TOL_M:.4f}m), orientation error "
            f"{pair_orientation_error:.3f}deg "
            f"(limit {EXEC_FINAL_PAIR_ORI_TOL_DEG:.3f}deg)"
        )

    return candidate_q, {
        "source": str(source),
        "planned_arm": planned_arm,
        "execution_arm": requested_arm,
        "arm_switched": requested_arm != planned_arm,
        "pair_pos_err_m": pair_position_error,
        "pair_ori_err_deg": pair_orientation_error,
        "pair_pos_tol_m": EXEC_FINAL_PAIR_POS_TOL_M,
        "pair_ori_tol_deg": EXEC_FINAL_PAIR_ORI_TOL_DEG,
        "target_comparison_frame": "submission_local_robot_base",
        "joint_limit_normalizations": normalizations,
        "submission_local_fk_checked": True,
        "j8_locked": LOCKED_J8_INDEX is not None,
    }



def _reject_duplicate_json_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload



def _load_plan_trajectory(
    session_id: str,
    plan_id: str,
    *,
    include_record: bool = False,
):
    path = _plan_record_path(session_id, plan_id)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        stat = os.fstat(descriptor)
        if stat.st_size <= 0 or stat.st_size > MAX_PLAN_RECORD_BYTES:
            raise ValueError(
                f"plan record size must be in 1..{MAX_PLAN_RECORD_BYTES} bytes"
            )
        raw = os.read(descriptor, stat.st_size + 1)
    finally:
        os.close(descriptor)
    if len(raw) != stat.st_size:
        raise ValueError("plan record changed while it was being read")
    record = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value!r}")
        ),
    )
    if not isinstance(record, dict):
        raise ValueError("plan record must be a JSON object")
    if str(record.get("plan_id")) != str(plan_id):
        raise ValueError("plan record plan_id does not match the request")
    if str(record.get("session_id")) != str(session_id):
        raise ValueError("plan record session_id does not match the request")
    record_tool = str(record.get("tool") or "")
    supported_source_tools = {
        "plan_grasp_point_filter_rgbd_lite",
        "move_tracked_point",
    }
    if record_tool not in supported_source_tools:
        raise ValueError(
            "plan record was not created by a supported official_v2 planner"
        )

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError(
            "plan record has no immutable joint trajectory; re-plan with official_v2"
        )
    expected_scalars = {
        "schema": JOINT_TRAJECTORY_SCHEMA,
        "schema_version": JOINT_TRAJECTORY_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "session_id": str(session_id),
        "tool_version": TOOL_VERSION,
        "source_tool": record_tool,
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": int(ARM_DOF),
        "action_dim": int(ACTION_DIM),
    }
    for key, expected in expected_scalars.items():
        if trajectory.get(key) != expected:
            raise ValueError(
                f"trajectory {key}={trajectory.get(key)!r} "
                f"does not match {expected!r}"
            )
    active_arm = str(trajectory.get("active_arm", "")).strip().lower()
    if active_arm not in ("left", "right"):
        raise ValueError("trajectory active_arm is invalid")
    if not str(trajectory.get("source_image_id", "")).strip():
        raise ValueError("trajectory source_image_id is missing")
    expected_locked_joints = (
        {"J8": LOCKED_J8_VALUE_RAD}
        if LOCKED_J8_INDEX is not None
        else {}
    )
    if trajectory.get("locked_arm_joints") != expected_locked_joints:
        raise ValueError("trajectory locked-arm-joint contract is invalid")

    integrity = trajectory.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "hmac-sha256"
    ):
        raise ValueError("trajectory integrity metadata is invalid")
    expected_digest = _trajectory_digest(trajectory)
    if not hmac.compare_digest(
        str(integrity.get("digest", "")),
        expected_digest,
    ):
        raise ValueError("trajectory digest mismatch")

    if record_tool == "move_tracked_point":
        move_source = record.get("move_tracked_trajectory")
        if not isinstance(move_source, dict):
            raise ValueError("move-tracked source trajectory is missing; re-plan")
        source_digest = str((move_source.get("integrity") or {}).get("digest", ""))
        bound_digest = str((trajectory.get("safety_validation") or {}).get(
            "source_move_trajectory_digest", ""
        ))
        if not (hmac.compare_digest(source_digest, _trajectory_digest(move_source))
                and hmac.compare_digest(source_digest, bound_digest)):
            raise ValueError("move-tracked source trajectory digest mismatch")
        _validate_move_tracked_endpoint_order(move_source["target"])

    safety = trajectory.get("safety_validation")
    if not isinstance(safety, dict):
        raise ValueError("trajectory safety_validation is missing")
    required_checks = (
        "joint_limits_checked",
        "local_fk_finite_checked",
        "safe_final_pair_validated_by_planner",
    )
    if not all(safety.get(key) is True for key in required_checks):
        raise ValueError("trajectory planner safety validation is incomplete")
    if safety.get("runtime_may_recompute_kinematics") is not False:
        raise ValueError("trajectory permits forbidden runtime kinematics")

    start_state = trajectory.get("start_state")
    if not isinstance(start_state, dict):
        raise ValueError("trajectory start_state is missing")
    motion_epoch = start_state.get("motion_epoch")
    if not isinstance(motion_epoch, int) or motion_epoch < 0:
        raise ValueError("trajectory motion_epoch is invalid")
    if not str(start_state.get("episode_id", "")).strip():
        raise ValueError("trajectory episode_id is invalid")
    arm_qpos = start_state.get("arm_qpos")
    arm_qvel = start_state.get("arm_qvel")
    if not isinstance(arm_qpos, dict) or not isinstance(arm_qvel, dict):
        raise ValueError("trajectory start arm proprioception is invalid")
    for side in ("left", "right"):
        _require_serialized_j8_zero(
            arm_qpos.get(side),
            label=f"start {side} arm_qpos",
        )
        _arm_vector(arm_qvel.get(side), label=f"start {side} arm_qvel")
    gripper_qpos = start_state.get("gripper_qpos")
    if not isinstance(gripper_qpos, dict):
        raise ValueError("trajectory start gripper proprioception is invalid")
    for side in ("left", "right"):
        gripper = np.asarray(
            gripper_qpos.get(side),
            dtype=np.float64,
        ).reshape(-1)
        if gripper.size == 0 or not np.all(np.isfinite(gripper)):
            raise ValueError(
                f"trajectory start {side} gripper_qpos is invalid"
            )
    trunk = np.asarray(start_state.get("trunk_qpos"), dtype=np.float64).reshape(-1)
    if trunk.size != 4 or not np.all(np.isfinite(trunk)):
        raise ValueError("trajectory start trunk_qpos is invalid")

    parameters = trajectory.get("parameters")
    tracking = trajectory.get("tracking")
    if not isinstance(parameters, dict) or not isinstance(tracking, dict):
        raise ValueError("trajectory parameters or tracking metadata is missing")
    back_m = float(parameters.get("back_m"))
    if not math.isfinite(back_m) or back_m < 0.0:
        raise ValueError("trajectory back_m is invalid")
    if not isinstance(parameters.get("reset_tool_roll_at_start"), bool):
        raise ValueError("trajectory tool-roll parameter is invalid")
    for key in (
        "start_arm_tolerance_rad",
        "start_trunk_tolerance_rad",
        "start_gripper_tolerance_rad",
        "stationary_qvel_tolerance_rad_s",
        "settle_tolerance_rad",
        "max_tracking_error_rad",
        "waypoint_timeout_s",
        "stall_steps",
    ):
        value = float(tracking.get(key))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"trajectory tracking field {key} is invalid")
    raw_trunk_assisted = tracking.get("trunk_assisted", False)
    if not isinstance(raw_trunk_assisted, bool):
        raise ValueError("trajectory trunk-assisted flag is invalid")
    if raw_trunk_assisted:
        raise ValueError(
            "exec_plan_pose accepts arm-only plans; re-plan this legacy "
            "whole-body move-tracked target"
        )
    # 已签名 start trunk 只收 1e-6 残差；不要改磁盘 JSON，HMAC 仍对原字节。
    trunk = _require_signed_trunk_q(
        trunk,
        label="trajectory start trunk_qpos",
        point="signed_start_trunk",
    )
    if trajectory.get("final_trunk_q") is not None:
        raise ValueError("arm-only trajectory contains a final trunk target")

    segments = trajectory.get("segments")
    expected_names = (
        "tool_roll_reset",
        "current_to_safe",
        "safe_to_final",
    )
    if (
        not isinstance(segments, list)
        or len(segments) != len(expected_names)
        or not all(isinstance(segment, dict) for segment in segments)
        or tuple(str(segment.get("name", "")) for segment in segments)
        != expected_names
    ):
        raise ValueError("trajectory segment order is invalid")
    waypoint_count = 0
    limits = _ARM_LIMITS[active_arm][:ARM_DOF]
    previous_q = _require_serialized_j8_zero(
        start_state["arm_qpos"][active_arm],
        label="trajectory start arm_qpos",
    )
    for segment in segments:
        waypoints = segment.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError(f"trajectory segment {segment['name']} is empty")
        endpoint_q = _require_serialized_j8_zero(
            segment.get("endpoint_q"),
            label=f"trajectory {segment['name']} endpoint_q",
        )
        if (
            segment.get("trunk_waypoints") is not None
            or segment.get("endpoint_trunk_q") is not None
        ):
            raise ValueError("arm-only trajectory segment contains trunk motion")
        for raw_q in waypoints:
            q = _require_serialized_j8_zero(
                raw_q,
                label=f"trajectory {segment['name']} waypoint",
            )
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                raise ValueError("trajectory waypoint exceeds arm joint limits")
            if (
                float(np.linalg.norm(q - previous_q, ord=np.inf))
                > DEFAULT_TRAJECTORY_STEP_RAD
                + TRAJECTORY_STEP_VALIDATION_EPS_RAD
            ):
                raise ValueError("trajectory waypoint step exceeds the compiled limit")
            previous_q = q
            waypoint_count += 1
        if not np.allclose(
            previous_q,
            endpoint_q,
            rtol=0.0,
            atol=LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
        ):
            raise ValueError(
                f"trajectory {segment['name']} endpoint does not match its last waypoint"
            )
    if waypoint_count > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("trajectory contains too many waypoints")
    final_q_by_arm = trajectory.get("final_q_by_arm")
    if final_q_by_arm is not None:
        if (
            not isinstance(final_q_by_arm, dict)
            or active_arm not in final_q_by_arm
            or any(side not in ("left", "right") for side in final_q_by_arm)
        ):
            raise ValueError("trajectory final_q_by_arm contract is invalid")
        active_final_q = _require_serialized_j8_zero(
            final_q_by_arm[active_arm],
            label=f"signed {active_arm} final_q_by_arm",
        )
        if not np.allclose(
            active_final_q,
            trajectory["segments"][2]["endpoint_q"],
            rtol=0.0,
            atol=LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
        ):
            raise ValueError(
                "trajectory active final_q_by_arm does not match final endpoint"
            )
        for side, raw_q in final_q_by_arm.items():
            if side == active_arm:
                continue
            validated_q, _validation = _validate_exec_final_q_for_arm(
                trajectory=trajectory,
                arm=side,
                raw_q=raw_q,
                source="signed_final_q_by_arm",
            )
            if not np.allclose(
                validated_q,
                raw_q,
                rtol=0.0,
                atol=LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
            ):
                raise ValueError(
                    f"trajectory {side} final_q_by_arm requires normalization"
                )
    if include_record:
        return trajectory, path, record
    return trajectory, path



def _exec_safe_back_m_candidates(
    requested_back_m: float,
    *,
    fallback_back_m: float | None = None,
) -> list[float]:
    """Match v2's stored-final safe retreat search order."""
    requested = float(
        np.clip(
            float(requested_back_m),
            EXEC_SAFE_BACK_MIN_M,
            EXEC_SAFE_BACK_MAX_M,
        )
    )
    values = [requested]
    requested_cm = int(round(requested * 100.0))
    values.extend(
        centimetres / 100.0
        for centimetres in sorted(
            range(5, int(EXEC_SAFE_BACK_MAX_M * 100.0) + 1),
            key=lambda value: (
                abs(value - requested_cm),
                value > requested_cm,
                abs(value),
            ),
        )
    )
    if fallback_back_m is not None and math.isfinite(float(fallback_back_m)):
        values.append(
            float(
                np.clip(
                    float(fallback_back_m),
                    EXEC_SAFE_BACK_MIN_M,
                    EXEC_SAFE_BACK_MAX_M,
                )
            )
        )
    result: list[float] = []
    for value in values:
        if all(abs(float(value) - old) > 1e-4 for old in result):
            result.append(float(value))
    return result



def _image_path(session_id: str, image_id: str, suffix: str) -> str:
    return os.path.join(_images_dir(session_id), f"{image_id}{suffix}")



def _image_meta_path(session_id: str, image_id: str) -> str:
    return _image_path(session_id, image_id, ".meta.json")



def _save_image_meta(
    session_id: str,
    image_id: str,
    meta: dict[str, Any],
) -> None:
    with open(
        _image_meta_path(session_id, image_id),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(meta, stream, indent=2, ensure_ascii=False)



def _load_image_meta(session_id: str, image_id: str) -> dict[str, Any]:
    path = _image_meta_path(session_id, image_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"image metadata does not exist: {session_id}/{image_id}"
        )
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)



def _policy_owned_capture_artifact(
    session_id: str,
    artifact: Any,
) -> str:
    name = str(artifact or "").strip()
    if not name:
        raise ValueError("capture has no depth_linear artifact")
    image_dir = os.path.realpath(_images_dir(session_id))
    candidate = os.path.realpath(
        name if os.path.isabs(name) else os.path.join(image_dir, name)
    )
    try:
        inside_image_dir = os.path.commonpath([image_dir, candidate]) == image_dir
    except ValueError:
        inside_image_dir = False
    if not inside_image_dir:
        raise ValueError(
            "depth_linear artifact is outside policy-owned capture storage"
        )
    return candidate



def _file_to_data_url(path: str, mime: str = "image/png") -> str | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as stream:
        encoded = base64.b64encode(stream.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"



def _camera_intrinsics(role: str, width: int, height: int) -> dict[str, Any]:
    if role == "head":
        fx = fy = 306.0 * float(width) / 720.0
    else:
        fx = fy = 388.6639 * float(width) / 480.0
    focal_length = 17.0
    horizontal_aperture = focal_length * float(width) / fx
    return {
        "image_width": int(width),
        "image_height": int(height),
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
        "fx": float(fx),
        "fy": float(fy),
        # The evaluator contract publishes K with principal point exactly at
        # (width / 2, height / 2), e.g. (360, 360) for the 720px head image.
        "cx": float(width) * 0.5,
        "cy": float(height) * 0.5,
    }



def _depth_visual(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(arr) & (arr > 0.0)
    visual = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(arr[valid], [2.0, 98.0])
        if float(hi) <= float(lo):
            hi = float(lo) + 1e-6
        scaled = 255.0 * (np.clip(arr, lo, hi) - lo) / (hi - lo)
        visual[valid] = np.asarray(255.0 - scaled[valid], dtype=np.uint8)
    return cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)



def _quat_to_mat_xyzw(quat) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )



def _axis_rotation(axis, angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    x, y, z = vector
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )



def _rotation_vector(rotation) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-9:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        )
    sine = math.sin(angle)
    if abs(sine) <= 1e-8:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = math.copysign(axis[0], matrix[2, 1] - matrix[1, 2])
        axis[1] = math.copysign(axis[1], matrix[0, 2] - matrix[2, 0])
        axis[2] = math.copysign(axis[2], matrix[1, 0] - matrix[0, 1])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return axis * angle



def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))



def _r1pro_arm_fk_robot(
    arm: str,
    trunk_q,
    arm_q,
) -> tuple[np.ndarray, np.ndarray]:
    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("arm must be left or right")
    q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    if q.size not in (7, 8):
        raise ValueError(f"arm qpos must have 7 or 8 values, got {q.size}")
    position, rotation = _r1pro_chest_pose_robot(trunk_q)
    position = position + rotation @ _ARM_BASE_ORIGINS[side]
    origins = _ARM_JOINT_ORIGINS[side]
    for index in range(q.size):
        position = position + rotation @ origins[index]
        rotation = rotation @ _axis_rotation(_ARM_AXES[index], float(q[index]))
    if q.size == 7:
        position = position + rotation @ origins[7]
    return position, rotation



def _reach_observation_sequence(ctx) -> int | None:
    adapter = getattr(ctx.world, "_official_adapter", None)
    metadata = getattr(adapter, "observation_metadata", None)
    if callable(metadata):
        try:
            return int(metadata()[0])
        except (IndexError, TypeError, ValueError):
            return None
    status = getattr(adapter, "status", None)
    if callable(status):
        try:
            value = status().get("sequence")
            return None if value is None else int(value)
        except (AttributeError, TypeError, ValueError):
            return None
    return None



def _relative_uv_to_pixel(
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    u_f = float(u)
    v_f = float(v)
    if not (math.isfinite(u_f) and math.isfinite(v_f)):
        raise ValueError("u/v must be finite")
    if not (0.0 <= u_f <= 1000.0 and 0.0 <= v_f <= 1000.0):
        raise ValueError("u/v must use relative coordinates 0..1000")
    px = int(round(u_f / 1000.0 * max(0, int(width) - 1)))
    py = int(round(v_f / 1000.0 * max(0, int(height) - 1)))
    return px, py



def _depth_value(
    depth: np.ndarray,
    px: int,
    py: int,
    radius: int = 2,
) -> float:
    arr = np.asarray(depth, dtype=np.float64).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {arr.shape}")
    height, width = arr.shape
    x0, x1 = max(0, px - radius), min(width, px + radius + 1)
    y0, y1 = max(0, py - radius), min(height, py + radius + 1)
    patch = arr[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 1e-5)]
    if valid.size == 0:
        raise ValueError(f"no valid depth near pixel ({px}, {py})")
    return float(np.median(valid))



def _unproject_pixel(
    *,
    px: int,
    py: int,
    depth_m: float,
    camera: dict[str, Any],
) -> np.ndarray:
    z = float(depth_m)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"invalid forward depth {depth_m!r}")
    fx = float(camera["fx"])
    fy = float(camera.get("fy", fx))
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    point_camera = np.array(
        [
            (float(px) - cx) / fx * z,
            -(float(py) - cy) / fy * z,
            -z,
        ],
        dtype=np.float64,
    )
    camera_pos = np.asarray(camera["pos"], dtype=np.float64).reshape(3)
    return camera_pos + _quat_to_mat_xyzw(camera["quat"]) @ point_camera



def _load_frozen_capture(
    session_id: str,
    image_id: str,
    *,
    role: str = "head",
) -> _FrozenCapture:
    session = str(session_id or "").strip()
    image = str(image_id or "").strip()
    meta = _load_image_meta(session, image)
    camera = dict(meta.get("camera") or {})
    if not camera:
        raise ValueError(f"capture {image!r} has no camera metadata")
    modalities = meta.get("modalities") or {}
    if not isinstance(modalities, dict):
        raise ValueError(f"capture {image!r} modalities metadata is invalid")
    depth_name = str(modalities.get("depth_linear") or "")
    if not depth_name:
        raise ValueError(f"capture {image!r} has no depth_linear artifact")
    depth_path = _policy_owned_capture_artifact(session, depth_name)
    depth = np.asarray(
        np.load(depth_path, allow_pickle=False),
        dtype=np.float32,
    ).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"capture depth must be 2-D, got {depth.shape}")
    rgb = None
    rgb_name = str((meta.get("rgb") or {}).get(str(role)) or "")
    if rgb_name:
        rgb_path = _policy_owned_capture_artifact(session, rgb_name)
        if os.path.isfile(rgb_path):
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    return _FrozenCapture(
        session_id=session,
        image_id=image,
        role=str(role),
        depth=depth,
        camera=camera,
        rgb=rgb,
        robot=dict(meta.get("robot") or {}),
        evaluator_sequence=(
            int(meta["evaluator_sequence"])
            if isinstance(meta.get("evaluator_sequence"), int)
            else None
        ),
    )



def _point_from_relative_uv(
    capture: _FrozenCapture,
    u: float,
    v: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = capture.depth.shape[:2]
    px, py = _relative_uv_to_pixel(u, v, width, height)
    depth_m = _depth_value(capture.depth, px, py)
    point = _unproject_pixel(
        px=px,
        py=py,
        depth_m=depth_m,
        camera=capture.camera,
    )
    return point, {
        "relative_uv": [float(u), float(v)],
        "pixel_uv": [int(px), int(py)],
        "depth_m": float(depth_m),
        "frame": str(
            capture.camera.get("frame", "local_command_odometry")
        ),
    }



def _point_in_robot_frame(world, point_local) -> np.ndarray:
    point = np.asarray(point_local, dtype=np.float64).reshape(3)
    pose = world.robot_pose()
    base = np.asarray(pose.pos, dtype=np.float64).reshape(3)
    delta = point - base
    yaw = float(pose.yaw)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            cos_y * delta[0] + sin_y * delta[1],
            -sin_y * delta[0] + cos_y * delta[1],
            delta[2],
        ],
        dtype=np.float64,
    )



def _r1pro_chest_pose_robot(
    trunk_q,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact R1Pro torso_link4 FK in the robot base frame."""
    q1, q2, q3, q4 = np.asarray(trunk_q, dtype=np.float64).reshape(4)

    def rot_y(angle: float) -> np.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a],
            ],
            dtype=np.float64,
        )

    def rot_z(angle: float) -> np.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    position = np.array([-0.079032, 0.0, 0.34265], dtype=np.float64)
    rotation = rot_y(q1)
    position += rotation @ np.array([0.0, 0.0, 0.400])
    rotation = rotation @ rot_y(q2)
    position += rotation @ np.array([0.0, 0.0001005, 0.300])
    rotation = rotation @ rot_y(-q3)
    position += rotation @ np.array([0.0, -0.00010015, 0.09962])
    rotation = rotation @ rot_z(q4)
    return position, rotation



def _r1pro_shoulder_positions_robot(trunk_q) -> dict[str, np.ndarray]:
    chest, rotation = _r1pro_chest_pose_robot(trunk_q)
    return {
        "left": chest
        + rotation @ np.array([-0.00048618, 0.170734, 0.30302]),
        "right": chest
        + rotation @ np.array([-0.00048706, -0.170736, 0.30302]),
    }



def _shoulder_distances_robot(
    target_robot: np.ndarray,
    trunk_qpos: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    target = np.asarray(target_robot, dtype=np.float64).reshape(3)
    shoulders = _r1pro_shoulder_positions_robot(trunk_qpos)
    distances = {
        side: float(np.linalg.norm(target - shoulder))
        for side, shoulder in shoulders.items()
    }
    return shoulders, distances



def _load_tracker_bound_capture(
    session_id: str,
    image_id: str,
) -> dict[str, Any]:
    """Load and hash the exact frozen head frame consumed by the tracker."""

    capture = _load_frozen_capture(session_id, image_id, role="head")
    if capture.rgb is None:
        raise ValueError("frozen head capture has no RGB artifact")
    if capture.rgb.shape[:2] != capture.depth.shape:
        raise ValueError(
            "frozen head RGB and depth_linear resolutions do not match"
        )
    if capture.evaluator_sequence is None:
        raise ValueError("frozen head capture has no evaluator sequence")
    capture_robot = dict(capture.robot or {})
    episode_id = str(capture_robot.get("episode_id") or "").strip()
    if not episode_id:
        raise ValueError("frozen head capture has no evaluator episode identifier")
    relative_pose = dict(
        (capture.camera or {}).get("robot_relative_pose") or {}
    )
    if not relative_pose:
        raise ValueError(
            "frozen head capture has no camera-relative pose for frame binding"
        )
    camera_to_robot_base = transform_from_pose(
        relative_pose.get("pos"),
        relative_pose.get("quat"),
    )
    frame_digest = tracker_frame_content_digest(
        TrackerFrame(
            rgb=capture.rgb,
            depth_linear=capture.depth,
            intrinsics=head_camera_intrinsics(
                capture.depth.shape[1],
                capture.depth.shape[0],
            ),
            camera_to_robot_base=camera_to_robot_base,
            camera_to_policy=np.eye(4, dtype=np.float64),
            sequence=int(capture.evaluator_sequence),
            timestamp_s=0.0,
            episode_id=episode_id,
            camera_role="head",
            color_order="bgr",
        )
    )
    return {
        "capture": capture,
        "episode_id": episode_id,
        "frame_digest": frame_digest,
    }



def _yield_adjust_chassis_controller(
    ctx,
    forward: float = 0.0,
    translation: float = 0.0,
    spin: float = 0.0,
    timeout_s: float = 120.0,
    _linear_tolerance_m: float | None = None,
    _require_linear_target: bool = False,
    _result_tool_name: str = "adjust_chassis",
    _result_extra: dict[str, Any] | None = None,
    _commit_result: bool = True,
    _emit_terminal_hold: bool = True,
    _allow_reverse_recovery: bool = True,
    _skip_linear_settle_for_spin: bool = False,
    _skip_linear_settle: bool = False,
    _spin_settle_steps: int | None = None,
    _componentwise_linear_limit: bool = False,
    _linear_stall_absolute_only: bool = False,
    _linear_body_frame_feedback: bool = False,
    _coupled_linear_spin: bool = False,
    _coupled_heading_tolerance_deg: float = 2.0,
    **kwargs,
):
    """Track one robot-frame XY vector, then spin, using official actions."""
    vmax = max(
        0.08,
        min(
            abs(float(kwargs.get("vmax", 0.5))),
            ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
        ),
    )
    linear_deceleration_mps2 = max(
        ADJUST_CHASSIS_LINEAR_DECEL_MPS2,
        min(
            abs(
                float(
                    kwargs.get(
                        "_linear_deceleration_mps2",
                        ADJUST_CHASSIS_LINEAR_DECEL_MPS2,
                    )
                )
            ),
            NAVIGATE_TO_LINEAR_DECEL_MPS2,
        ),
    )
    linear_near_speed_mps = max(
        ADJUST_CHASSIS_LINEAR_NEAR_SPEED_MPS,
        min(
            abs(
                float(
                    kwargs.get(
                        "_linear_near_speed_mps",
                        ADJUST_CHASSIS_LINEAR_NEAR_SPEED_MPS,
                    )
                )
            ),
            vmax,
        ),
    )
    linear_final_speed_mps = max(
        ADJUST_CHASSIS_LINEAR_FINAL_SPEED_MPS,
        min(
            abs(
                float(
                    kwargs.get(
                        "_linear_final_speed_mps",
                        ADJUST_CHASSIS_LINEAR_FINAL_SPEED_MPS,
                    )
                )
            ),
            linear_near_speed_mps,
        ),
    )
    spin_gain = max(
        ADJUST_CHASSIS_SPIN_GAIN,
        min(
            abs(float(kwargs.get("_spin_gain", ADJUST_CHASSIS_SPIN_GAIN))),
            NAVIGATE_TO_SPIN_GAIN,
        ),
    )
    spin_settle_steps = (
        ADJUST_CHASSIS_SETTLE_STEPS
        if _spin_settle_steps is None
        else max(
            1,
            min(ADJUST_CHASSIS_SETTLE_STEPS, int(_spin_settle_steps)),
        )
    )
    wmax = max(0.10, min(abs(float(kwargs.get("wmax", 1.0))), 1.0))
    total_budget = max(1, int(max(0.1, float(timeout_s)) * CONTROL_HZ))
    linear_tolerance_m = (
        ADJUST_CHASSIS_FORWARD_TOL_M
        if _linear_tolerance_m is None
        else max(ADJUST_CHASSIS_FORWARD_TOL_M, float(_linear_tolerance_m))
    )
    dt = 1.0 / float(CONTROL_HZ)
    used_steps = 0
    adapter = getattr(ctx.world, "_official_adapter", None)

    def _sequence() -> int | None:
        if adapter is None:
            return None
        try:
            return int(adapter.status().get("sequence"))
        except (AttributeError, TypeError, ValueError):
            return None

    def _qvel() -> np.ndarray:
        try:
            value = np.asarray(
                ctx.world.base_qvel(), dtype=np.float64
            ).reshape(3)
        except (TypeError, ValueError):
            return np.zeros(3, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            return np.zeros(3, dtype=np.float64)
        return value

    def _action(
        vx_mps: float = 0.0,
        vy_mps: float = 0.0,
        wz_radps: float = 0.0,
    ):
        # Official actions are normalized controller inputs. The static R1Pro
        # output limits are part of the submitted robot contract, not a live
        # simulator observation.
        return ctx.world.set_base_velocity(
            float(np.clip(
                float(vx_mps) / ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
                -1.0,
                1.0,
            )),
            float(np.clip(
                float(vy_mps) / ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
                -1.0,
                1.0,
            )),
            float(np.clip(
                float(wz_radps) / ADJUST_CHASSIS_BASE_MAX_ANG_RADPS,
                -1.0,
                1.0,
            )),
        )

    def _fresh(sequence: int | None, previous: int | None) -> bool:
        return bool(
            adapter is None
            or (
                sequence is not None
                and previous is not None
                and sequence > previous
            )
        )

    direction_sign = 1.0 if float(forward) >= 0.0 else -1.0
    translation_sign = 1.0 if float(translation) >= 0.0 else -1.0
    target_xy = np.asarray(
        [float(forward), float(translation)], dtype=np.float64
    )
    target_distance = float(np.linalg.norm(target_xy))
    target_unit = (
        target_xy / target_distance
        if target_distance > 1e-9
        else np.zeros(2, dtype=np.float64)
    )
    position_xy = np.zeros(2, dtype=np.float64)
    last_linear_cmd = np.zeros(2, dtype=np.float64)
    reverse_escape_request = bool(
        float(forward) < -1e-9 and abs(float(translation)) <= 1e-9
    )
    recovery_direction = (
        target_unit.copy()
        if reverse_escape_request
        else -target_unit
    )
    recovery_motion_sign = float(recovery_direction[0])
    progress = 0.0
    translation_progress = 0.0
    last_healthy_progress = 0.0
    last_forward_cmd = 0.0
    last_translation_cmd = 0.0
    stall_steps = 0
    last_sequence = _sequence()
    obstacle_limited = False
    obstacle_reason = None
    recovery_attempted = False
    recovery_ok = True
    commanded_progress = 0.0
    forward_commanded_progress = 0.0
    linear_observed_xy = np.zeros(2, dtype=np.float64)
    linear_observed_along_mps = 0.0
    linear_stall_threshold_mps = 0.0
    low_speed_start_progress = 0.0
    linear_yaw_rad = 0.0
    coupled_spin_target_rad = math.radians(float(spin))
    coupled_linear_spin = bool(
        _coupled_linear_spin
        and _linear_body_frame_feedback
        and target_distance > 1.0e-9
        and abs(coupled_spin_target_rad) > 1.0e-9
    )
    coupled_heading_tolerance_rad = math.radians(
        max(0.5, min(5.0, abs(float(_coupled_heading_tolerance_deg))))
    )
    coupled_last_w = 0.0
    coupled_spin_stall_steps = 0
    coupled_max_schedule_error_rad = 0.0

    while target_distance > 1e-9 and used_steps < total_budget:
        _check_cancelled(ctx, "adjust chassis linear")
        sequence = _sequence()
        fresh_observation = _fresh(sequence, last_sequence)
        last_speed = float(np.linalg.norm(last_linear_cmd))
        if fresh_observation and (
            last_speed > 1e-9 or abs(coupled_last_w) > 1e-9
        ):
            observed_velocity = _qvel()
            velocity_xy = observed_velocity[:2]
            previous_progress = progress
            linear_observed_xy[:] = velocity_xy
            if _linear_body_frame_feedback:
                delta_yaw = float(observed_velocity[2]) * dt
                middle_yaw = linear_yaw_rad + delta_yaw / 2.0
                c, s = math.cos(middle_yaw), math.sin(middle_yaw)
                velocity_xy = np.array([c * velocity_xy[0] - s * velocity_xy[1],
                                        s * velocity_xy[0] + c * velocity_xy[1]])
                linear_yaw_rad += delta_yaw
            position_xy[:] = position_xy + velocity_xy * dt
            progress = max(0.0, float(np.dot(position_xy, target_unit)))
            translation_progress = max(
                0.0,
                translation_sign * float(position_xy[1]),
            )
            if last_speed > 1e-9:
                observed = float(
                    np.dot(velocity_xy, last_linear_cmd / last_speed)
                )
                threshold = max(
                    ADJUST_CHASSIS_MIN_QVEL_MPS,
                    0.0
                    if _linear_stall_absolute_only
                    else ADJUST_CHASSIS_STALL_RATIO * last_speed,
                )
                linear_observed_along_mps = observed
                linear_stall_threshold_mps = threshold
                if observed >= threshold:
                    stall_steps = 0
                    last_healthy_progress = progress
                else:
                    if stall_steps == 0:
                        low_speed_start_progress = previous_progress
                    stall_steps += 1
            if coupled_linear_spin and abs(coupled_last_w) > 1e-9:
                observed_spin = math.copysign(1.0, coupled_last_w) * float(
                    observed_velocity[2]
                )
                coupled_spin_stall_steps = (
                    coupled_spin_stall_steps + 1
                    if observed_spin < max(0.01, 0.20 * abs(coupled_last_w))
                    else 0
                )
        if sequence is not None:
            last_sequence = sequence
        remaining_xy = target_xy - position_xy
        remaining = float(np.linalg.norm(remaining_xy))
        coupled_final_error = _normalize_angle(
            coupled_spin_target_rad - linear_yaw_rad
        )
        if remaining <= linear_tolerance_m and (
            not coupled_linear_spin
            or abs(coupled_final_error) <= math.radians(0.7)
        ):
            break
        if stall_steps >= ADJUST_CHASSIS_STALL_STEPS:
            if abs(float(forward)) > 1e-9:
                obstacle_limited = True
                obstacle_reason = "base_qvel_stall"
                recovery_attempted = True
            else:
                recovery_ok = False
                obstacle_reason = "translation_qvel_stall"
            break
        if coupled_spin_stall_steps >= ADJUST_CHASSIS_STALL_STEPS:
            recovery_ok = False
            obstacle_reason = "coupled_spin_qvel_stall"
            break

        c, s = math.cos(linear_yaw_rad), math.sin(linear_yaw_rad)
        body_remaining = np.array([c * remaining_xy[0] + s * remaining_xy[1],
                                   -s * remaining_xy[0] + c * remaining_xy[1]])
        speed = 0.0
        if remaining > linear_tolerance_m:
            directional_vmax = vmax
            if _componentwise_linear_limit:
                direction_max = float(np.max(np.abs(body_remaining / remaining)))
                if direction_max > 1e-12:
                    directional_vmax = vmax / direction_max
            speed = min(
                directional_vmax,
                max(
                    0.05,
                    math.sqrt(2.0 * linear_deceleration_mps2 * remaining),
                ),
            )
            if remaining <= 0.08:
                speed = min(speed, linear_near_speed_mps)
            if remaining <= 0.03:
                speed = min(speed, linear_final_speed_mps)

        current_fraction = float(np.clip(progress / target_distance, 0.0, 1.0))
        schedule_error = _normalize_angle(
            coupled_spin_target_rad * current_fraction - linear_yaw_rad
        )
        coupled_max_schedule_error_rad = max(
            coupled_max_schedule_error_rad, abs(schedule_error)
        )
        if coupled_linear_spin and abs(schedule_error) > coupled_heading_tolerance_rad:
            # Hold centre progress while yaw catches the already-certified SE(2)
            # interpolation.  The error band is part of the signed trajectory
            # contract and remains under the depth tracking reserve.
            speed = 0.0
        if speed > 1.0e-9:
            last_linear_cmd[:] = remaining_xy * (speed / remaining)
            body_command = body_remaining * (speed / remaining)
        else:
            last_linear_cmd[:] = 0.0
            body_command = np.zeros(2, dtype=np.float64)
        last_forward_cmd = float(body_command[0])
        last_translation_cmd = float(body_command[1])
        coupled_last_w = 0.0
        if coupled_linear_spin:
            if remaining <= linear_tolerance_m:
                desired_yaw = coupled_spin_target_rad
                feedforward_w = 0.0
            else:
                desired_yaw = coupled_spin_target_rad * current_fraction
                feedforward_w = coupled_spin_target_rad * speed / target_distance
            coupled_last_w = float(np.clip(
                feedforward_w
                + NAVIGATE_TO_SPIN_GAIN
                * _normalize_angle(desired_yaw - linear_yaw_rad),
                -wmax,
                wmax,
            ))
        yield _action(
            vx_mps=last_forward_cmd,
            vy_mps=last_translation_cmd,
            wz_radps=coupled_last_w,
        )
        used_steps += 1
        commanded_progress += speed * dt
        forward_commanded_progress += abs(last_forward_cmd) * dt

    linear_remaining = float(np.linalg.norm(target_xy - position_xy))
    linear_target_reached = bool(
        linear_remaining <= linear_tolerance_m
    )
    if (
        target_distance > 1e-9
        and not obstacle_limited
        and not linear_target_reached
        and recovery_ok
    ):
        recovery_ok = False
        obstacle_reason = (
            "forward_timeout"
            if abs(float(translation)) <= 1e-9
            else (
                "translation_timeout"
                if abs(float(forward)) <= 1e-9
                else "linear_timeout"
            )
        )

    retreat_progress = 0.0
    retreat_target = 0.0
    retreat_speed = ADJUST_CHASSIS_RETREAT_INITIAL_MPS
    retreat_speed_max = 0.0
    retreat_escalations = 0
    recovery_trigger = None
    settled_steps = 0

    def _base_qvel_is_settled() -> bool:
        velocity = _qvel()
        return bool(
            math.hypot(float(velocity[0]), float(velocity[1]))
            <= ADJUST_CHASSIS_SETTLE_SPEED_MPS
            and abs(float(velocity[2]))
            <= ADJUST_CHASSIS_SETTLE_YAW_RATE_RADPS
        )

    settled_before_recovery = False
    if obstacle_limited and recovery_ok:
        requested_recovery = bool(recovery_attempted)
        consecutive = 0
        settle_budget = min(
            total_budget - used_steps,
            2 * ADJUST_CHASSIS_SETTLE_STEPS,
        )
        for _ in range(max(0, settle_budget)):
            yield _action()
            used_steps += 1
            consecutive = (
                consecutive + 1 if _base_qvel_is_settled() else 0
            )
            settled_steps = max(settled_steps, consecutive)
            if consecutive >= ADJUST_CHASSIS_SETTLE_STEPS:
                settled_before_recovery = True
                break
        if settled_before_recovery:
            if requested_recovery and reverse_escape_request:
                settled_before_recovery = False
                recovery_attempted = True
                recovery_trigger = "reverse_qvel_stall"
            else:
                # A repeat request at an already stable forward boundary is
                # idempotent.
                recovery_attempted = False
        else:
            recovery_attempted = True
            recovery_trigger = (
                "base_qvel_stall"
                if requested_recovery
                else "post_stop_unsettled"
            )
            recovery_ok = True

    reverse_stall_steps = 0
    reverse_last_cmd = 0.0

    if recovery_attempted and not _allow_reverse_recovery:
        recovery_attempted = False
        recovery_trigger = "reverse_recovery_disabled_for_signed_path"
        recovery_ok = False

    if recovery_attempted and recovery_ok:
        for _ in range(6):
            if used_steps >= total_budget:
                break
            yield _action()
            used_steps += 1
        overlap = max(0.0, progress - last_healthy_progress)
        retreat_target = min(
            ADJUST_CHASSIS_RETREAT_MAX_M,
            max(
                (
                    0.05
                    if recovery_trigger == "post_stop_unsettled"
                    else ADJUST_CHASSIS_RETREAT_MIN_M
                ),
                (
                    float(np.linalg.norm(target_xy - position_xy))
                    if recovery_trigger == "reverse_qvel_stall"
                    else overlap + ADJUST_CHASSIS_RETREAT_MARGIN_M
                ),
            ),
        )
        while recovery_ok:
            reverse_stall_steps = 0
            reverse_last_cmd = 0.0
            local_last_sequence = _sequence()
            while (
                used_steps < total_budget
                and retreat_progress < retreat_target
            ):
                _check_cancelled(ctx, "adjust chassis grounded recovery")
                sequence = _sequence()
                fresh_observation = _fresh(
                    sequence,
                    local_last_sequence,
                )
                if fresh_observation and reverse_last_cmd > 0.0:
                    velocity_xy = _qvel()[:2]
                    observed = max(
                        0.0,
                        float(np.dot(velocity_xy, recovery_direction)),
                    )
                    retreat_progress += observed * dt
                    position_xy[:] = position_xy + velocity_xy * dt
                    progress = max(
                        0.0,
                        float(np.dot(position_xy, target_unit)),
                    )
                    translation_progress = max(
                        0.0,
                        translation_sign * float(position_xy[1]),
                    )
                    threshold = max(
                        ADJUST_CHASSIS_MIN_QVEL_MPS,
                        ADJUST_CHASSIS_STALL_RATIO * reverse_last_cmd,
                    )
                    reverse_stall_steps = (
                        0
                        if observed >= threshold
                        else reverse_stall_steps + 1
                    )
                    if (
                        reverse_stall_steps
                        >= ADJUST_CHASSIS_RETREAT_STALL_STEPS
                    ):
                        if retreat_speed >= ADJUST_CHASSIS_RETREAT_MAX_MPS - 1e-9:
                            # A blocked retreat cannot be repaired by holding
                            # the maximum command for the rest of the rollout.
                            # Stop after a full fresh-observation stall window
                            # at maximum speed, then use the failure hold below.
                            recovery_ok = False
                            obstacle_reason = "recovery_qvel_stall"
                            break
                        retreat_speed = min(
                            ADJUST_CHASSIS_RETREAT_MAX_MPS,
                            max(
                                retreat_speed * 1.65,
                                retreat_speed + 0.06,
                            ),
                        )
                        retreat_escalations += 1
                        reverse_stall_steps = 0
                if sequence is not None:
                    local_last_sequence = sequence
                if retreat_progress >= retreat_target:
                    break
                retreat_speed_max = max(retreat_speed_max, retreat_speed)
                reverse_last_cmd = retreat_speed
                recovery_cmd = recovery_direction * retreat_speed
                yield _action(
                    vx_mps=float(recovery_cmd[0]),
                    vy_mps=float(recovery_cmd[1]),
                )
                used_steps += 1
            recovery_ok = bool(
                recovery_ok and retreat_progress + 0.002 >= retreat_target
            )
            if not recovery_ok:
                break
            if (
                recovery_trigger == "reverse_qvel_stall"
                and float(np.linalg.norm(target_xy - position_xy))
                <= linear_tolerance_m
            ):
                obstacle_limited = False
                obstacle_reason = None

            consecutive = 0
            settled_after_recovery = False
            settle_budget = min(
                total_budget - used_steps,
                2 * ADJUST_CHASSIS_SETTLE_STEPS,
            )
            for _ in range(max(0, settle_budget)):
                yield _action()
                used_steps += 1
                consecutive = (
                    consecutive + 1 if _base_qvel_is_settled() else 0
                )
                settled_steps = max(settled_steps, consecutive)
                if consecutive >= ADJUST_CHASSIS_SETTLE_STEPS:
                    settled_after_recovery = True
                    break
            if settled_after_recovery:
                recovery_ok = True
                break
            if (
                retreat_target >= ADJUST_CHASSIS_RETREAT_MAX_M - 1e-9
                or used_steps >= total_budget
            ):
                recovery_ok = False
                break
            retreat_target = min(
                ADJUST_CHASSIS_RETREAT_MAX_M,
                max(retreat_target + 0.04, retreat_progress + 0.025),
            )

    elif (
        recovery_ok
        and not settled_before_recovery
        and not (
            _skip_linear_settle
            and target_distance > 1.0e-9
            and linear_target_reached
            and abs(float(spin)) <= 1.0e-9
        )
        and not (
            _skip_linear_settle_for_spin
            and target_distance <= 1.0e-9
            and abs(float(spin)) > 1.0e-9
        )
    ):
        # Normal target completion and a zero-motion request both need an
        # explicit dynamic settle before success.
        consecutive = 0
        recovery_ok = False
        settle_budget = min(
            total_budget - used_steps,
            2 * ADJUST_CHASSIS_SETTLE_STEPS,
        )
        for _ in range(max(0, settle_budget)):
            yield _action()
            used_steps += 1
            consecutive = (
                consecutive + 1 if _base_qvel_is_settled() else 0
            )
            settled_steps = max(settled_steps, consecutive)
            if consecutive >= ADJUST_CHASSIS_SETTLE_STEPS:
                recovery_ok = True
                break

    if not recovery_ok:
        for _ in range(3):
            if used_steps >= total_budget:
                break
            yield _action()
            used_steps += 1

    progress = max(0.0, float(np.dot(position_xy, target_unit)))
    translation_progress = max(
        0.0,
        translation_sign * float(position_xy[1]),
    )
    linear_remaining = float(np.linalg.norm(target_xy - position_xy))
    linear_target_reached = bool(
        linear_remaining <= linear_tolerance_m
    )
    translation_ok = bool(
        recovery_ok
        and (
            abs(float(translation)) <= 1e-9
            or linear_target_reached
            or obstacle_limited
        )
    )

    spin_target = abs(math.radians(float(spin)))
    spin_sign = 1.0 if float(spin) >= 0.0 else -1.0
    spin_progress = (
        max(0.0, spin_sign * linear_yaw_rad)
        if coupled_linear_spin
        else 0.0
    )
    spin_stall_steps = 0
    spin_last_cmd = 0.0
    last_sequence = _sequence()
    spin_ok = bool(
        translation_ok
        and (
            not coupled_linear_spin
            or abs(_normalize_angle(coupled_spin_target_rad - linear_yaw_rad))
            <= math.radians(0.7)
        )
    )
    while (
        spin_ok
        and not coupled_linear_spin
        and spin_target > 1e-9
        and used_steps < total_budget
    ):
        _check_cancelled(ctx, "adjust chassis spin")
        sequence = _sequence()
        fresh_observation = _fresh(sequence, last_sequence)
        if fresh_observation and abs(spin_last_cmd) > 1e-9:
            observed = spin_sign * float(_qvel()[2])
            spin_progress = max(0.0, spin_progress + observed * dt)
            spin_stall_steps = (
                spin_stall_steps + 1
                if observed < max(0.01, 0.20 * abs(spin_last_cmd))
                else 0
            )
        if sequence is not None:
            last_sequence = sequence
        if spin_progress + math.radians(0.7) >= spin_target:
            break
        if spin_stall_steps >= ADJUST_CHASSIS_STALL_STEPS:
            spin_ok = False
            obstacle_reason = obstacle_reason or "spin_qvel_stall"
            break
        remaining = max(0.0, spin_target - spin_progress)
        spin_last_cmd = spin_sign * min(
            wmax,
            max(0.10, spin_gain * remaining),
        )
        yield _action(wz_radps=spin_last_cmd)
        used_steps += 1

    if spin_ok and not coupled_linear_spin and spin_target > 1e-9:
        spin_ok = spin_progress + math.radians(0.7) >= spin_target
        if spin_ok:
            consecutive = 0
            spin_ok = False
            settle_budget = min(
                total_budget - used_steps,
                2 * spin_settle_steps,
            )
            for _ in range(max(0, settle_budget)):
                yield _action()
                used_steps += 1
                consecutive = (
                    consecutive + 1 if _base_qvel_is_settled() else 0
                )
                settled_steps = max(settled_steps, consecutive)
                if consecutive >= spin_settle_steps:
                    spin_ok = True
                    break

    linear_cross_track = (
        float(
            np.dot(
                position_xy,
                np.asarray([-target_unit[1], target_unit[0]]),
            )
        )
        if target_distance > 1e-9
        else 0.0
    )
    near_target_ok = bool(linear_target_reached)
    ok = bool(
        recovery_ok
        and translation_ok
        and spin_ok
        and (near_target_ok or not _require_linear_target)
    )
    timed_out = bool(not ok and used_steps >= total_budget)
    payload = {
        "ok": ok,
        "error": (
            None
            if ok
            else (
                "chassis did not reach the requested XY tolerance from allowed proprioception"
                if _require_linear_target and not near_target_ok
                else "chassis could not recover and settle from allowed proprioception"
            )
        ),
        "requested": {
            "forward_m": float(forward),
            "translation_m": float(translation),
            "spin_deg": float(spin),
        },
        "actual": {
            "forward_m": float(position_xy[0]),
            "translation_m": float(position_xy[1]),
            "spin_deg": float(spin_sign * math.degrees(spin_progress)),
        },
        "translation_ok": bool(translation_ok),
        "linear_target_reached": bool(linear_target_reached),
        "near_target_ok": near_target_ok,
        "require_linear_target": bool(_require_linear_target),
        "linear_remaining_m": float(linear_remaining),
        "linear_tolerance_m": float(linear_tolerance_m),
        "linear_cross_track_m": float(linear_cross_track),
        "linear_control": "simultaneous_robot_frame_xy",
        "linear_speed_limit_mps": float(vmax),
        "linear_yaw_drift_deg": math.degrees(linear_yaw_rad),
        "coupled_linear_spin": bool(coupled_linear_spin),
        "coupled_schedule_error_max_deg": math.degrees(
            coupled_max_schedule_error_rad
        ),
        "linear_tracking": {
            "source": "evaluator_base_qvel",
            "stall_policy": "absolute_progress" if _linear_stall_absolute_only else "command_ratio",
            "observed_xy_mps": linear_observed_xy.tolist(),
            "commanded_xy_mps": [last_forward_cmd, last_translation_cmd],
            "observed_along_command_mps": float(linear_observed_along_mps),
            "stall_threshold_mps": float(linear_stall_threshold_mps),
            "consecutive_low_speed_observations": int(stall_steps),
            "low_speed_progress_m": (
                max(0.0, float(progress - low_speed_start_progress)) if stall_steps else 0.0
            ),
        },
        "obstacle_limited": bool(obstacle_limited),
        "obstacle_stop_reason": obstacle_reason,
        "recovery_attempted": bool(recovery_attempted),
        "recovery_trigger": recovery_trigger,
        "recovery_motion_sign": recovery_motion_sign,
        "recovery_motion_vector": recovery_direction.astype(float).tolist(),
        "recovery_ok": bool(recovery_ok),
        "retreat_m": float(retreat_progress),
        "retreat_target_m": float(retreat_target),
        "retreat_speed_max_mps": float(retreat_speed_max),
        "retreat_speed_escalations": int(retreat_escalations),
        "settled_steps": int(settled_steps),
        "spin_settle_steps_required": int(spin_settle_steps),
        "action_steps": int(used_steps),
        "timed_out": timed_out,
        "failure_stage": "timeout" if timed_out else (None if ok else "tracking"),
        "verification": "direct_evaluator_base_qvel_integration",
        "landing_verification": (
            "deferred_to_navigation_causal_hold"
            if _skip_linear_settle and linear_target_reached
            else "base_qvel_dynamic_settle"
        ),
        "observation_contract": [
            "base_qvel_proprioception",
            "action_history",
            "static_r1pro_action_limits",
        ],
        "forbidden_observation_reads": [],
        "forward_commanded_distance_m": float(forward_commanded_progress),
        "linear_commanded_distance_m": float(commanded_progress),
        "rgbd_motion_guard_used": False,
        "dynamic_collision_truth": False,
        "global_base_pose_truth": False,
    }
    if _result_extra:
        payload = {**dict(_result_extra), **payload}
    if _commit_result:
        _set_result(ctx, str(_result_tool_name), payload)
    if _emit_terminal_hold:
        yield ctx.world.hold_action()
    return payload



def _yield_adjust_height_motion(
    ctx,
    *,
    upward: float,
    timeout_s: float,
    tolerance: float,
    stage_prefix: str,
):
    """Execute the test-owned absolute v2 torso LUT through official actions."""
    timeout = float(timeout_s)
    tolerance = float(tolerance)
    if not math.isfinite(timeout) or timeout <= 0.0:
        return {
            "ok": False,
            "error": "timeout_s must be a positive finite value",
            "execution": "rejected_before_action",
            "direct_simulator_mutation": False,
        }
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        return {
            "ok": False,
            "error": "tol must be a positive finite value",
            "execution": "rejected_before_action",
            "direct_simulator_mutation": False,
        }

    start_observed = np.asarray(
        ctx.world.trunk_qpos(), dtype=np.float64
    ).reshape(4)
    try:
        full_waypoints, plan = plan_absolute_height_trajectory(
            start_observed,
            float(upward),
        )
        waypoints, waypoint_sampling = sample_lut_execution_waypoints(
            full_waypoints,
            plan["phases_used"],
            max_waypoints=TRUNK_LUT_MAX_EXEC_WAYPOINTS,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "execution": "rejected_before_action",
            "direct_simulator_mutation": False,
        }

    target = np.asarray(waypoints[-1], dtype=np.float64).reshape(4)
    target[3] = 0.0
    for side in ("left", "right"):
        ctx.world.set_arm_pin_qpos(side, ctx.world.arm_qpos_list(side))

    timeout_steps = max(1, int(math.ceil(timeout * CONTROL_HZ)))
    deadline = time.monotonic() + timeout
    action_steps = 0
    lut_action_steps = 0
    waypoint_hold_counts: list[int] = []
    waypoint_max_tracking_error = 0.0
    waypoint_advance_tolerance = min(
        tolerance,
        ADJUST_HEIGHT_WAYPOINT_ADVANCE_TOLERANCE_RAD,
    )
    timed_out = False
    for waypoint_index, waypoint in enumerate(waypoints):
        command = np.asarray(waypoint, dtype=np.float64).reshape(4).copy()
        command[3] = 0.0
        if action_steps >= timeout_steps or time.monotonic() >= deadline:
            timed_out = True
            break
        _check_cancelled(
            ctx,
            f"{stage_prefix} sampled LUT waypoint {waypoint_index}",
        )
        yield ctx.world.make_action_trunk_locked(command.tolist())
        action_steps += 1
        lut_action_steps += 1
        waypoint_hold_counts.append(1)

        observed_waypoint = np.asarray(
            ctx.world.trunk_qpos(), dtype=np.float64
        ).reshape(4)
        waypoint_error = float(
            np.max(np.abs(command - observed_waypoint))
        )
        if math.isfinite(waypoint_error):
            waypoint_max_tracking_error = max(
                waypoint_max_tracking_error,
                waypoint_error,
            )

    stable_steps = 0
    settle_steps = 0
    stall_steps = 0
    stalled = False
    best_error = float("inf")
    max_observed_error = 0.0
    if not timed_out:
        while action_steps < timeout_steps:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            _check_cancelled(ctx, f"{stage_prefix} final proprio settle")
            yield ctx.world.make_action_trunk_locked(target.tolist())
            action_steps += 1
            settle_steps += 1

            observed = np.asarray(
                ctx.world.trunk_qpos(), dtype=np.float64
            ).reshape(4)
            error = float(np.max(np.abs(target - observed)))
            max_observed_error = max(max_observed_error, error)
            adapter = getattr(ctx.world, "_official_adapter", None)
            proprio = None if adapter is None else adapter.proprio_vector()
            if proprio is None:
                qvel_inf = float("inf")
            else:
                qvel = np.asarray(
                    proprio[PROPRIO_SLICES["trunk_qvel"]],
                    dtype=np.float64,
                ).reshape(4)
                qvel_inf = float(np.linalg.norm(qvel, ord=np.inf))

            if error <= tolerance:
                stall_steps = 0
                if qvel_inf <= ADJUST_HEIGHT_STABLE_QVEL_RAD_S:
                    stable_steps += 1
                    if stable_steps >= ADJUST_HEIGHT_REQUIRED_STABLE_STEPS:
                        break
                else:
                    stable_steps = 0
            else:
                stable_steps = 0
                if error + 1e-4 < best_error:
                    best_error = error
                    stall_steps = 0
                else:
                    stall_steps += 1
            best_error = min(best_error, error)
            if stall_steps >= ADJUST_HEIGHT_TRACKING_STALL_STEPS:
                stalled = True
                break

    if action_steps >= timeout_steps and (
        stable_steps < ADJUST_HEIGHT_REQUIRED_STABLE_STEPS
    ):
        timed_out = True

    achieved = np.asarray(
        ctx.world.trunk_qpos(), dtype=np.float64
    ).reshape(4)
    joint_error = target - achieved
    max_error = float(np.max(np.abs(joint_error)))
    z_start = chest_height_lut_frame_m(start_observed)
    z_end = chest_height_lut_frame_m(achieved)
    z_target = float(plan["z_tgt_m"])
    z_predicted_target = float(plan["z_predicted_m"])
    z_error = float(z_end - z_predicted_target)
    z_requested_error = float(z_end - z_target)
    height_tolerance = max(0.02, tolerance)
    actual_upward = float(z_end - z_start)
    direction = str(plan["direction"])
    opposite_direction = bool(
        (direction == "down" and actual_upward > 1e-3)
        or (direction == "up" and actual_upward < -1e-3)
    )
    direction_verified = bool(
        (direction == "down" and actual_upward < -1e-3)
        or (direction == "up" and actual_upward > 1e-3)
        or (direction == "hold" and abs(actual_upward) <= height_tolerance)
    )
    requested_height_converged = bool(abs(z_requested_error) <= height_tolerance)
    # An off-table posture can map both endpoints to the same nearest LUT
    # row. The planner then holds the measured pose, even when an unsaturated
    # nonzero height request remains unfulfilled. Tracking that hold perfectly
    # is not success. Keep zero requests, small quantized moves and explicit
    # workspace saturation behavior unchanged; do not invent a connecting move.
    unfulfilled_hold = bool(
        direction == "hold"
        and plan["requested_direction"] != "hold"
        and not plan["saturated_hold"]
        and not requested_height_converged
    )
    height_converged = bool(abs(z_error) <= height_tolerance and not unfulfilled_hold)
    theta_start = torso_theta_z_deg(start_observed)
    theta_end = torso_theta_z_deg(achieved)
    converged = bool(
        not timed_out
        and not stalled
        and max_error <= tolerance
        and stable_steps >= ADJUST_HEIGHT_REQUIRED_STABLE_STEPS
        and direction_verified
        and height_converged
    )
    if converged:
        ctx.world.set_trunk_pin_qpos(target.tolist())
    else:
        stopped_target = np.clip(
            achieved,
            TRUNK_LIMITS[:, 0],
            TRUNK_LIMITS[:, 1],
        )
        stopped_target[3] = 0.0
        ctx.world.set_trunk_pin_qpos(stopped_target.tolist())

    return {
            "ok": converged,
            "upward_m": float(upward),
            "workspace_saturated": bool(plan["z_clamped"]),
            "effective_planned_upward_m": float(plan["predicted_upward_m"]),
            "start_trunk_q": start_observed.astype(float).tolist(),
            "target_trunk_q": target.astype(float).tolist(),
            "achieved_qpos": achieved.astype(float).tolist(),
            "error_rad": joint_error.astype(float).tolist(),
            "max_abs_error_rad": max_error,
            "max_observed_error_rad": max_observed_error,
            "tolerance_rad": tolerance,
            "timeout_s": timeout,
            "timed_out": bool(timed_out),
            "tracking_stalled": bool(stalled),
            "stall_steps": int(stall_steps),
            "stall_limit": int(ADJUST_HEIGHT_TRACKING_STALL_STEPS),
            "stable_steps": int(stable_steps),
            "required_stable_steps": int(
                ADJUST_HEIGHT_REQUIRED_STABLE_STEPS
            ),
            "settle_steps": int(settle_steps),
            "action_steps": int(action_steps),
            "lut_hold_actions": 1,
            "lut_hold_actions_min": 1,
            "lut_hold_actions_max": 1,
            "source_v2_lut_hold_actions": int(TRUNK_LUT_HOLD_ACTIONS),
            "lut_hold_policy": "single_action_per_sampled_lut_waypoint",
            "lut_action_steps": int(lut_action_steps),
            "fixed_hold_equivalent_lut_action_steps": int(
                len(waypoints) * TRUNK_LUT_HOLD_ACTIONS
            ),
            "saved_lut_action_steps": int(
                len(waypoints) * TRUNK_LUT_HOLD_ACTIONS - lut_action_steps
            ),
            "waypoint_hold_counts": list(waypoint_hold_counts),
            "waypoint_early_advance_count": 0,
            "full_lut_waypoint_count": int(len(full_waypoints)),
            "execution_lut_waypoint_count": int(len(waypoints)),
            "skipped_lut_waypoint_count": int(
                len(full_waypoints) - len(waypoints)
            ),
            "full_fixed_hold_equivalent_lut_action_steps": int(
                len(full_waypoints) * TRUNK_LUT_HOLD_ACTIONS
            ),
            "saved_vs_full_fixed_hold_action_steps": int(
                len(full_waypoints) * TRUNK_LUT_HOLD_ACTIONS
                - lut_action_steps
            ),
            "waypoint_sampling": waypoint_sampling,
            "waypoint_advance_tolerance_rad": float(
                waypoint_advance_tolerance
            ),
            "waypoint_max_tracking_error_rad": float(
                waypoint_max_tracking_error
            ),
            "z_start_m": float(z_start),
            "z_end_m": float(z_end),
            "actual_upward_m": actual_upward,
            "z_target_m": z_target,
            "z_error_m": z_error,
            "z_requested_error_m": z_requested_error,
            "height_tolerance_m": height_tolerance,
            "height_converged": height_converged,
            "requested_height_converged": requested_height_converged,
            "unfulfilled_hold": unfulfilled_hold,
            "direction_verified": direction_verified,
            "opposite_direction": opposite_direction,
            "z_predicted_target_m": z_predicted_target,
            "z_model_tracking_error_m": float(
                z_end - z_predicted_target
            ),
            "theta_z_start_deg": float(theta_start),
            "theta_z_end_deg": float(theta_end),
            "theta_z_delta_deg": float(theta_end - theta_start),
            "model": plan["model"],
            "planner": plan,
            "execution": "official_action_two_phase_absolute_lut",
            "verification": "evaluator_proprio_and_submission_local_fk",
            "trunk_yaw_locked": bool(target[3] == 0.0),
            "arms_grippers": "pinned through official action targets",
            "direct_simulator_mutation": False,
            **(
                {}
                if converged
                else {
                    "error": (
                        "adjust_height could not realize the requested height from this posture: "
                        "the LUT selected a hold, so no height change was commanded"
                        if unfulfilled_hold
                        else
                        "adjust_height timed out before the LUT target converged"
                        if timed_out
                        else (
                            "adjust_height proprioceptive tracking stalled"
                            if stalled
                            else (
                                (
                                    "adjust_height moved opposite to the requested direction"
                                    if opposite_direction
                                    else (
                                        "adjust_height made no measurable progress "
                                        "in the requested direction"
                                    )
                                )
                                if not direction_verified
                                else (
                                    "adjust_height did not reach the requested LUT height"
                                    if not height_converged
                                    else "adjust_height did not converge to the LUT target"
                                )
                            )
                        )
                    )
                }
            ),
    }



def _move_point_is_cancellation(exc: BaseException) -> bool:
    return any(
        cls.__name__ == "SkillCancelled"
        for cls in type(exc).__mro__
    )



@dataclass
class _AdjustSafetyMonitor:
    trunk_start: np.ndarray
    base_xy_start: np.ndarray
    base_yaw_start: float
    proprio_stall_steps: int = 0
    tracking_error_steps: int = 0
    base_disturbance_steps: int = 0
    max_tracking_error_rad: float = 0.0
    min_progress_ratio: float = 1.0
    max_base_linear_speed_mps: float = 0.0
    max_base_yaw_rate_rad_s: float = 0.0
    max_base_translation_m: float = 0.0
    max_base_yaw_drift_rad: float = 0.0
    max_trunk_drift_rad: float = 0.0



def _adjust_angle_delta(current: float, start: float) -> float:
    delta = float(current) - float(start)
    return math.atan2(math.sin(delta), math.cos(delta))



def _adjust_safety_monitor(ctx, initial: dict[str, Any]) -> _AdjustSafetyMonitor:
    pose = ctx.world.robot_pose()
    return _AdjustSafetyMonitor(
        trunk_start=np.asarray(initial["trunk"], dtype=np.float64)[:3].copy(),
        base_xy_start=np.asarray(pose.pos, dtype=np.float64)[:2].copy(),
        base_yaw_start=float(pose.yaw),
    )



def _adjust_safety_report(monitor: _AdjustSafetyMonitor) -> dict[str, Any]:
    return {
        "fresh_observation_stall_limit": int(ADJUST_EEF_STALL_STEPS),
        "proprio_stall_steps": int(monitor.proprio_stall_steps),
        "tracking_error_steps": int(monitor.tracking_error_steps),
        "tracking_error_is_diagnostic_only": True,
        "base_disturbance_steps": int(monitor.base_disturbance_steps),
        "min_command_progress_ratio": float(monitor.min_progress_ratio),
        "max_tracking_error_inf_rad": float(monitor.max_tracking_error_rad),
        "max_base_linear_speed_mps": float(monitor.max_base_linear_speed_mps),
        "max_base_yaw_rate_rad_s": float(monitor.max_base_yaw_rate_rad_s),
        "max_base_translation_m": float(monitor.max_base_translation_m),
        "max_base_yaw_drift_rad": float(monitor.max_base_yaw_drift_rad),
        "max_trunk_q123_drift_inf_rad": float(monitor.max_trunk_drift_rad),
    }



def _adjust_next_step_scale(
    current_scale: float,
    safety_observation: dict[str, Any],
    *,
    near_target: bool,
) -> float:
    """Increase command reach only after one well-tracked official action."""

    scale = float(
        np.clip(
            current_scale,
            ADJUST_EEF_ADAPTIVE_MIN_SCALE,
            ADJUST_EEF_ADAPTIVE_MAX_SCALE,
        )
    )
    if near_target:
        return min(scale, 1.0)
    progress_ratio = float(
        safety_observation.get("command_progress_ratio", 0.0)
    )
    tracking_error = float(
        safety_observation.get("tracking_error_inf_rad", float("inf"))
    )
    if (
        progress_ratio >= ADJUST_EEF_ADAPTIVE_GOOD_PROGRESS_RATIO
        and tracking_error <= ADJUST_EEF_ADAPTIVE_GOOD_TRACKING_ERROR_RAD
    ):
        return min(
            ADJUST_EEF_ADAPTIVE_MAX_SCALE,
            scale + ADJUST_EEF_ADAPTIVE_SCALE_INCREMENT,
        )
    if (
        progress_ratio < ADJUST_EEF_ADAPTIVE_POOR_PROGRESS_RATIO
        or tracking_error > ADJUST_EEF_ADAPTIVE_POOR_TRACKING_ERROR_RAD
    ):
        return max(ADJUST_EEF_ADAPTIVE_MIN_SCALE, 0.5 * scale)
    return scale



def _adjust_settle_steps(
    measured: dict[str, Any],
    *,
    target_reached: bool,
    skip_when_reached: bool,
) -> int:
    """Choose a bounded measured-hold tail from evaluator proprioception."""

    if not target_reached:
        return ADJUST_EEF_FAILURE_SETTLE_STEPS
    if skip_when_reached:
        return 0
    qvel = np.asarray(measured.get("qvel", []), dtype=np.float64).reshape(-1)
    if qvel.size >= 7 and np.all(np.isfinite(qvel[:7])):
        if float(np.linalg.norm(qvel[:7], ord=np.inf)) <= ADJUST_EEF_SETTLED_QVEL_RAD_S:
            return 0
    return ADJUST_EEF_SUCCESS_SETTLE_MAX_STEPS



def _adjust_safety_after_action(
    ctx,
    monitor: _AdjustSafetyMonitor,
    *,
    before: dict[str, Any],
    command: np.ndarray,
    after: dict[str, Any],
    base_translation_limit_m: float = ADJUST_EEF_BASE_TRANSLATION_LIMIT_M,
    base_yaw_limit_rad: float = ADJUST_EEF_BASE_YAW_LIMIT_RAD,
    proprio_stall_requires_stationary: bool = False,
    expected_trunk: Sequence[float] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    command_delta = np.asarray(command[:7], dtype=np.float64) - np.asarray(
        before["q"][:7],
        dtype=np.float64,
    )
    actual_delta = np.asarray(after["q"][:7], dtype=np.float64) - np.asarray(
        before["q"][:7],
        dtype=np.float64,
    )
    command_inf = float(np.linalg.norm(command_delta, ord=np.inf))
    progress_ratio = 1.0
    if command_inf >= ADJUST_EEF_MIN_COMMAND_RAD:
        denominator = float(command_delta @ command_delta)
        progress_ratio = float(actual_delta @ command_delta) / max(
            denominator,
            1e-12,
        )
        monitor.min_progress_ratio = min(
            monitor.min_progress_ratio,
            progress_ratio,
        )
        low_progress = bool(progress_ratio < ADJUST_EEF_MIN_PROGRESS_RATIO)
        if proprio_stall_requires_stationary:
            observed_joint_delta = float(
                np.linalg.norm(actual_delta, ord=np.inf)
            )
            observed_arm_qvel = float(
                np.linalg.norm(
                    np.asarray(after.get("qvel", []), dtype=np.float64)[:7],
                    ord=np.inf,
                )
            )
            low_progress = bool(
                low_progress
                and observed_joint_delta
                <= MOVE_POINT_TO_POINT_STALL_POSITION_DELTA_RAD
                and observed_arm_qvel
                <= MOVE_POINT_TO_POINT_STALL_QVEL_RAD_S
            )
        monitor.proprio_stall_steps = (
            monitor.proprio_stall_steps + 1 if low_progress else 0
        )
    else:
        monitor.proprio_stall_steps = 0

    tracking_error = float(
        np.linalg.norm(after["q"][:7] - command[:7], ord=np.inf)
    )
    monitor.max_tracking_error_rad = max(
        monitor.max_tracking_error_rad,
        tracking_error,
    )
    monitor.tracking_error_steps = (
        monitor.tracking_error_steps + 1
        if tracking_error > ADJUST_EEF_MAX_TRACKING_ERROR_RAD
        else 0
    )
    # Match v2's finite-impedance loop: command tracking lag is telemetry, not
    # a failure. Every action is a fresh measured-q tangent target, so a joint
    # need not settle to the previous one-frame target before the loop advances.
    # Directional progress below the floor is handled separately by the true
    # proprio_stall guard.

    base_qvel = np.asarray(after["base_qvel"], dtype=np.float64).reshape(3)
    base_linear_speed = float(np.linalg.norm(base_qvel[:2]))
    base_yaw_rate = abs(float(base_qvel[2]))
    monitor.max_base_linear_speed_mps = max(
        monitor.max_base_linear_speed_mps,
        base_linear_speed,
    )
    monitor.max_base_yaw_rate_rad_s = max(
        monitor.max_base_yaw_rate_rad_s,
        base_yaw_rate,
    )
    base_velocity_disturbed = bool(
        base_linear_speed > ADJUST_EEF_BASE_LINEAR_DISTURBANCE_MPS
        or base_yaw_rate > ADJUST_EEF_BASE_YAW_DISTURBANCE_RAD_S
    )
    monitor.base_disturbance_steps = (
        monitor.base_disturbance_steps + 1
        if base_velocity_disturbed
        else 0
    )

    pose = ctx.world.robot_pose()
    base_translation = float(
        np.linalg.norm(
            np.asarray(pose.pos, dtype=np.float64)[:2]
            - monitor.base_xy_start
        )
    )
    base_yaw_drift = abs(
        _adjust_angle_delta(float(pose.yaw), monitor.base_yaw_start)
    )
    monitor.max_base_translation_m = max(
        monitor.max_base_translation_m,
        base_translation,
    )
    monitor.max_base_yaw_drift_rad = max(
        monitor.max_base_yaw_drift_rad,
        base_yaw_drift,
    )

    trunk_drift = float(
        np.linalg.norm(
            np.asarray(after["trunk"], dtype=np.float64)[:3]
            - monitor.trunk_start,
            ord=np.inf,
        )
    )
    monitor.max_trunk_drift_rad = max(
        monitor.max_trunk_drift_rad,
        trunk_drift,
    )
    trunk_tracking_error = 0.0
    if expected_trunk is not None:
        expected_trunk_array = np.asarray(
            expected_trunk, dtype=np.float64
        ).reshape(4)
        if not np.all(np.isfinite(expected_trunk_array)):
            raise ValueError("expected trunk command is non-finite")
        trunk_tracking_error = float(
            np.linalg.norm(
                np.asarray(after["trunk"], dtype=np.float64).reshape(4)
                - expected_trunk_array,
                ord=np.inf,
            )
        )

    reason = None
    if (
        base_linear_speed > ADJUST_EEF_BASE_IMMEDIATE_LINEAR_MPS
        or base_yaw_rate > ADJUST_EEF_BASE_IMMEDIATE_YAW_RAD_S
    ):
        reason = "base_disturbance"
    elif (
        base_translation > float(base_translation_limit_m)
        or base_yaw_drift > float(base_yaw_limit_rad)
    ):
        reason = "base_odometry_disturbance"
    elif (
        expected_trunk is None
        and trunk_drift > ADJUST_EEF_TRUNK_DRIFT_LIMIT_RAD
    ):
        reason = "trunk_disturbance"
    elif monitor.base_disturbance_steps >= ADJUST_EEF_BASE_DISTURBANCE_STEPS:
        reason = "base_disturbance"
    elif monitor.proprio_stall_steps >= ADJUST_EEF_STALL_STEPS:
        reason = "proprio_stall"

    return reason, {
        "command_inf_rad": command_inf,
        "command_progress_ratio": progress_ratio,
        "tracking_error_inf_rad": tracking_error,
        "proprio_stall_requires_stationary": bool(
            proprio_stall_requires_stationary
        ),
        "base_linear_speed_mps": base_linear_speed,
        "base_yaw_rate_rad_s": base_yaw_rate,
        "base_translation_m": base_translation,
        "base_translation_limit_m": float(base_translation_limit_m),
        "base_yaw_drift_rad": base_yaw_drift,
        "base_yaw_limit_rad": float(base_yaw_limit_rad),
        "trunk_q123_drift_inf_rad": trunk_drift,
        "trunk_motion_commanded": bool(expected_trunk is not None),
        "trunk_tracking_error_inf_rad": float(trunk_tracking_error),
    }



def _adjust_adapter(ctx):
    adapter = getattr(ctx.world, "_official_adapter", None)
    if adapter is None:
        raise RuntimeError("official evaluator observation adapter is unavailable")
    snapshot = adapter.snapshot()
    if int(snapshot.sequence) <= 0 or float(snapshot.received_ts) <= 0.0:
        raise RuntimeError("no evaluator observation is available")
    proprio = adapter.proprio_vector()
    if proprio is None:
        raise RuntimeError("evaluator proprioception is unavailable")
    return adapter, int(snapshot.sequence)



def _adjust_camera_pose(ctx, role: str) -> tuple[dict[str, Any], int, Any]:
    adapter, sequence = _adjust_adapter(ctx)
    camera = dict(adapter.camera_relative_poses().get(str(role)) or {})
    if camera.get("pos") is None or camera.get("quat") is None:
        raise RuntimeError(
            f"evaluator {role} camera-relative pose is unavailable"
        )
    position = np.asarray(camera["pos"], dtype=np.float64).reshape(3)
    quaternion = np.asarray(camera["quat"], dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError(f"evaluator {role} camera-relative pose is invalid")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-9:
        raise RuntimeError(f"evaluator {role} camera quaternion is invalid")
    return {
        "pos": position,
        "quat": quaternion / quaternion_norm,
        "frame": "robot_base",
        "source": "evaluator_cam_rel_poses",
    }, sequence, adapter



def _move_tracked_camera_pose_payload(
    position: Sequence[float] | None,
    quaternion: Sequence[float] | None,
    *,
    label: str = "move-tracked head camera pose",
) -> dict[str, Any] | None:
    """Validate and canonicalize a camera pose persisted with a faceto plan.

    A camera-relative pose is policy observation data, not simulator state.  Keep
    the signed representation small and deterministic so a plan reload uses the
    exact same optical axis that was used to solve its endpoint.
    """

    if position is None and quaternion is None:
        return None
    if position is None or quaternion is None:
        raise ValueError(f"{label} requires both position and quaternion")
    try:
        position_array = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion_array = np.asarray(quaternion, dtype=np.float64).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain a 3-vector and a 4-vector") from exc
    if not np.all(np.isfinite(position_array)) or not np.all(
        np.isfinite(quaternion_array)
    ):
        raise ValueError(f"{label} contains a non-finite value")
    norm = float(np.linalg.norm(quaternion_array))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} quaternion is degenerate")
    return {
        "position_robot_base_m": position_array.astype(float).tolist(),
        "quaternion_xyzw": (quaternion_array / norm).astype(float).tolist(),
        "frame": "robot_base",
        "source": "frozen_evaluator_cam_rel_poses",
    }



def _adjust_gripper_vector(value: Any) -> np.ndarray:
    raw = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    if raw.size < 2:
        raw = np.pad(raw, (0, 2 - raw.size))
    raw = raw[:2]
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("evaluator gripper proprioception is invalid")
    return raw



def _adjust_kinematic_observation(ctx, arm: str) -> dict[str, Any]:
    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("arm must be left or right")
    trunk = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(-1)
    if trunk.size != 4 or not np.all(np.isfinite(trunk)):
        raise RuntimeError("evaluator trunk proprioception is invalid")
    raw_left = np.asarray(
        ctx.world.arm_qpos_list("left"),
        dtype=np.float64,
    ).reshape(-1)
    raw_right = np.asarray(
        ctx.world.arm_qpos_list("right"),
        dtype=np.float64,
    ).reshape(-1)
    raw_selected_qvel = np.asarray(
        ctx.world.arm_qvel_list(side),
        dtype=np.float64,
    ).reshape(-1)
    if (
        raw_selected_qvel.size < ARM_DOF
        or not np.all(np.isfinite(raw_selected_qvel[:ARM_DOF]))
    ):
        raise RuntimeError(f"evaluator {side} arm qvel is invalid")
    base_qvel = np.asarray(
        ctx.world.base_qvel(),
        dtype=np.float64,
    ).reshape(-1)
    if base_qvel.size < 3 or not np.all(np.isfinite(base_qvel[:3])):
        raise RuntimeError("evaluator base qvel is invalid")
    left = locked_arm_vector(raw_left, label="left arm proprioception")
    right = locked_arm_vector(raw_right, label="right arm proprioception")
    gripper_left = _adjust_gripper_vector(
        ctx.world.gripper_qpos_list("left")
    )
    gripper_right = _adjust_gripper_vector(
        ctx.world.gripper_qpos_list("right")
    )
    state = local_robot_state(
        trunk_q=trunk,
        arm_left_q=left,
        arm_right_q=right,
        gripper_left_q=gripper_left,
        gripper_right_q=gripper_right,
    )
    selected = left if side == "left" else right
    raw_selected = raw_left if side == "left" else raw_right
    position, quaternion = _local_eef_pose(state, side, selected)
    observed_j8 = (
        float(raw_selected[7])
        if ARM_DOF == 8 and raw_selected.size >= 8
        else 0.0
    )
    return {
        "state": state,
        "trunk": trunk,
        "left": left,
        "right": right,
        "q": selected.copy(),
        "raw_q": raw_selected.copy(),
        "qvel": raw_selected_qvel[:ARM_DOF].copy(),
        "base_qvel": base_qvel[:3].copy(),
        "position": np.asarray(position, dtype=np.float64).reshape(3),
        "quaternion": np.asarray(quaternion, dtype=np.float64).reshape(4),
        "observed_j8": observed_j8,
        "gripper": gripper_left if side == "left" else gripper_right,
    }



def _adjust_observation_advanced(adapter, previous_sequence: int) -> tuple[bool, int]:
    current_sequence, _received_ts = adapter.observation_metadata()
    return int(current_sequence) > int(previous_sequence), int(current_sequence)



def _adjust_action(ctx, arm: str, q_target: np.ndarray) -> np.ndarray:
    command = locked_arm_vector(q_target, label=f"{arm} action target")
    return ctx.world.make_action(**{f"arm_{arm}": command.tolist()})



def _adjust_latch_measured_hold(
    ctx,
    *,
    arm: str,
    adapter,
    observation_sequence: int,
    measured: dict[str, Any],
    settle_steps: int,
):
    sequence = int(observation_sequence)
    observation = measured
    steps = 0
    observation_advanced = True
    bounded_settle_steps = int(
        np.clip(
            int(settle_steps),
            ADJUST_EEF_SETTLE_MIN_STEPS,
            ADJUST_EEF_SETTLE_MAX_STEPS,
        )
    )
    for _ in range(bounded_settle_steps):
        hold_target = np.asarray(observation["q"], dtype=np.float64).copy()
        yield _adjust_action(ctx, arm, hold_target)
        steps += 1
        advanced, current_sequence = _adjust_observation_advanced(
            adapter,
            sequence,
        )
        if not advanced:
            observation_advanced = False
            break
        sequence = current_sequence
        observation = _adjust_kinematic_observation(ctx, arm)

    # The outer tool emits one final hold action after this helper returns.
    # Refreshing the pin here makes that action use the newest measured pose.
    ctx.world.set_arm_pin_qpos(arm, observation["q"])
    return {
        "steps": int(steps),
        "observation_advanced": bool(observation_advanced),
        "last_observation_sequence": int(sequence),
        "observation": observation,
    }



def _execute_local_pose_closed_loop(
    ctx,
    *,
    arm: str,
    adapter,
    observation_sequence: int,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
    stage_name: str,
    deadline_monotonic: float | None = None,
    skip_success_settle: bool = False,
):
    deadline = (
        float(deadline_monotonic)
        if deadline_monotonic is not None
        else time.monotonic() + max(1.0, float(timeout_s))
    )
    initial = _adjust_kinematic_observation(ctx, arm)
    safety_monitor = _adjust_safety_monitor(ctx, initial)
    q_start = initial["q"].copy()
    start_position = initial["position"].copy()
    start_quaternion = initial["quaternion"].copy()
    lower, upper = _local_arm_joint_limits(initial["state"], arm)
    initial_position_error = float(
        np.linalg.norm(np.asarray(target_position) - start_position)
    )
    initial_orientation_error = orientation_error_deg(
        start_quaternion,
        target_quaternion,
    )
    target_q, target_ik = solve_pose_target(
        initial["state"],
        arm,
        q_start,
        target_position=target_position,
        target_quaternion=target_quaternion,
        joint_indices=range(7),
        pos_tol_m=max(0.010, float(pos_tol)),
        ori_tol_deg=max(3.0, float(ori_tol_deg)),
    )
    position_scale = max(0.010, float(pos_tol))
    orientation_scale = max(3.0, float(ori_tol_deg))
    initial_ik_score = (
        initial_position_error / position_scale
        + initial_orientation_error / orientation_scale
    )
    initial_ik_max_score = max(
        initial_position_error / position_scale,
        initial_orientation_error / orientation_scale,
    )
    target_ik_position_error = float(
        target_ik.get("pos_err_m", float("inf"))
    )
    target_ik_orientation_error = float(
        target_ik.get("ori_err_deg", float("inf"))
    )
    candidate_ik_score = (
        target_ik_position_error / position_scale
        + target_ik_orientation_error / orientation_scale
    )
    candidate_ik_max_score = max(
        target_ik_position_error / position_scale,
        target_ik_orientation_error / orientation_scale,
    )
    # Preserve the v2 contract: a preflight solution is guidance only after it
    # satisfies both pose tolerances. A partial least-squares solution can pull
    # the tangent loop toward a position/orientation compromise near limits.
    guidance_accepted = bool(
        np.all(np.isfinite(target_q[:7]))
        and bool(target_ik.get("ok", False))
    )
    target_ik["guidance_accepted"] = guidance_accepted
    target_ik["initial_normalized_error"] = float(initial_ik_score)
    target_ik["initial_normalized_max_error"] = float(initial_ik_max_score)
    target_ik["candidate_normalized_error"] = float(candidate_ik_score)
    target_ik["candidate_normalized_max_error"] = float(
        candidate_ik_max_score
    )
    gross_preflight_rejection = bool(
        not guidance_accepted
        and (
            not math.isfinite(candidate_ik_max_score)
            or candidate_ik_max_score
            > ADJUST_EEF_GROSS_PREFLIGHT_MAX_NORMALIZED_ERROR
        )
    )
    target_ik["motion_rejected"] = gross_preflight_rejection
    target_ik["gross_rejection_normalized_error_limit"] = float(
        ADJUST_EEF_GROSS_PREFLIGHT_MAX_NORMALIZED_ERROR
    )
    joint_goal = target_q[:7] if guidance_accepted else None
    nominal_position_steps = max(
        1,
        int(math.ceil(initial_position_error / MAX_TRANSLATION_STEP_M)),
    )
    nominal_orientation_steps = max(
        1,
        int(math.ceil(math.radians(initial_orientation_error) / 0.06)),
    )
    relinearization_multiplier = (
        ADJUST_EEF_GUIDED_RELINEARIZATION_MULTIPLIER
        if guidance_accepted
        else ADJUST_EEF_RELINEARIZATION_MULTIPLIER
    )
    guided_joint_nominal_steps = (
        int(
            math.ceil(
                float(np.linalg.norm(target_q[:7] - q_start[:7], ord=np.inf))
                / (
                    ADJUST_POSE_MAX_JOINT_STEP_RAD
                    * ADJUST_EEF_GUIDED_STEP_SCALE
                )
            )
        )
        if guidance_accepted
        else 0
    )
    guided_joint_tracking_steps = (
        ADJUST_EEF_GUIDED_JOINT_TRACKING_MULTIPLIER
        * guided_joint_nominal_steps
    )
    nominal_execution_steps = min(
        max(1, int(max_steps)),
        max(
            8,
            relinearization_multiplier
            * max(nominal_position_steps, nominal_orientation_steps),
            guided_joint_tracking_steps,
        ),
    )
    # A mildly imperfect terminal solve still uses the observation-relinearized
    # tangent fallback. A grossly invalid solve must not spend dozens of
    # official actions chasing an unreachable pose and exciting the arm near
    # its limits.
    execution_steps = (
        0 if gross_preflight_rejection else nominal_execution_steps
    )
    previous_dq = None
    adaptive_step_scale = 1.0
    max_adaptive_step_scale = 1.0
    min_adaptive_step_scale = 1.0
    action_steps = 0
    max_observed_j8 = abs(float(initial["observed_j8"]))
    reason = (
        "terminal_ik_grossly_unreachable"
        if gross_preflight_rejection
        else "max_steps"
    )
    last_step: dict[str, Any] = {}
    last_observed = initial
    sequence = int(observation_sequence)
    best_pose_score = max(
        initial_position_error / position_scale,
        initial_orientation_error / orientation_scale,
    )
    pose_stall_steps = 0
    near_zero_command_steps = 0
    guided_steps = 0
    tangent_steps = 0
    guided_fallbacks = 0
    max_observed_orientation_error_deg = float(initial_orientation_error)
    orientation_divergence_limit_deg = max(
        ADJUST_EEF_ORIENTATION_DIVERGENCE_MARGIN_DEG,
        initial_orientation_error
        + ADJUST_EEF_ORIENTATION_DIVERGENCE_MARGIN_DEG,
        ADJUST_EEF_ORIENTATION_DIVERGENCE_TOL_MULTIPLIER
        * orientation_scale,
    )

    for step in range(execution_steps):
        _check_cancelled(ctx, f"{stage_name} step {step + 1}")
        if time.monotonic() >= deadline:
            reason = "timeout"
            break
        observed = _adjust_kinematic_observation(ctx, arm)
        last_observed = observed
        max_observed_j8 = max(
            max_observed_j8,
            abs(float(observed["observed_j8"])),
        )
        q_current = observed["q"]
        position_error_vector = np.asarray(
            target_position,
            dtype=np.float64,
        ) - observed["position"]
        position_error = float(np.linalg.norm(position_error_vector))
        orientation_vector = orientation_error_vector(
            observed["quaternion"],
            target_quaternion,
        )
        orientation_error_value = math.degrees(
            float(np.linalg.norm(orientation_vector))
        )
        max_observed_orientation_error_deg = max(
            max_observed_orientation_error_deg,
            orientation_error_value,
        )
        if (
            position_error <= max(0.001, float(pos_tol))
            and orientation_error_value <= max(0.1, float(ori_tol_deg))
        ):
            reason = "target_converged"
            break

        max_adaptive_step_scale = max(
            max_adaptive_step_scale,
            adaptive_step_scale,
        )
        min_adaptive_step_scale = min(
            min_adaptive_step_scale,
            adaptive_step_scale,
        )
        try:
            delta_q = np.zeros(7, dtype=np.float64)
            solver_report: dict[str, Any] = {}
            if joint_goal is not None:
                delta_q, solver_report = bounded_pose_goal_step(
                    observed["state"],
                    arm,
                    q_current,
                    target_q,
                    target_position=target_position,
                    target_quaternion=target_quaternion,
                    pos_tol_m=max(0.001, float(pos_tol)),
                    ori_tol_deg=max(0.1, float(ori_tol_deg)),
                    max_joint_step_rad=(
                        ADJUST_POSE_MAX_JOINT_STEP_RAD
                        * ADJUST_EEF_GUIDED_STEP_SCALE
                    ),
                    max_translation_step_m=(
                        MAX_TRANSLATION_STEP_M * ADJUST_EEF_GUIDED_STEP_SCALE
                    ),
                    max_orientation_step_rad=(
                        MAX_ORIENTATION_STEP_RAD * ADJUST_EEF_GUIDED_STEP_SCALE
                    ),
                    previous_dq=previous_dq,
                )
                if bool(solver_report.get("guided_goal_step_accepted")):
                    guided_steps += 1
                    max_adaptive_step_scale = max(
                        max_adaptive_step_scale,
                        ADJUST_EEF_GUIDED_STEP_SCALE,
                    )
                else:
                    guided_fallbacks += 1
            if not bool(solver_report.get("guided_goal_step_accepted")):
                jacobian, _position, _quaternion = numerical_eef_jacobian(
                    observed["state"],
                    arm,
                    q_current,
                    joint_indices=range(7),
                )
                delta_q, solver_report = bounded_pose_tangent_step(
                    jacobian,
                    position_error_vector,
                    orientation_vector,
                    q_current[:7],
                    q_lower=lower[:7],
                    q_upper=upper[:7],
                    previous_dq=previous_dq,
                    joint_goal=joint_goal,
                    position_satisfied=position_error
                    <= max(0.001, float(pos_tol)),
                    max_joint_step_rad=(
                        ADJUST_POSE_MAX_JOINT_STEP_RAD * adaptive_step_scale
                    ),
                    max_translation_step_m=(
                        MAX_TRANSLATION_STEP_M * adaptive_step_scale
                    ),
                    max_orientation_step_rad=(
                        MAX_ORIENTATION_STEP_RAD * adaptive_step_scale
                    ),
                )
                tangent_steps += 1
        except Exception as exc:
            reason = f"local_solver_error:{type(exc).__name__}"
            last_step = {"error": str(exc)}
            break

        command = q_current.copy()
        command[:7] = np.clip(
            q_current[:7] + delta_q,
            lower[:7],
            upper[:7],
        )
        if ARM_DOF == 8:
            command[7] = 0.0
        last_step = {
            **solver_report,
            "index": int(step + 1),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error_value,
            "adaptive_step_scale": float(adaptive_step_scale),
            "command_q_rad": command.tolist(),
        }
        yield _adjust_action(ctx, arm, command)
        action_steps += 1

        advanced, current_sequence = _adjust_observation_advanced(adapter, sequence)
        if not advanced:
            reason = "observation_not_advanced"
            break
        sequence = current_sequence
        after_action = _adjust_kinematic_observation(ctx, arm)
        last_observed = after_action
        safety_reason, safety_observation = _adjust_safety_after_action(
            ctx,
            safety_monitor,
            before=observed,
            command=command,
            after=after_action,
        )
        last_step["safety_observation"] = safety_observation
        if safety_reason is not None:
            reason = safety_reason
            break
        after_position_error = float(
            np.linalg.norm(
                np.asarray(target_position, dtype=np.float64)
                - after_action["position"]
            )
        )
        after_orientation_error = orientation_error_deg(
            after_action["quaternion"],
            target_quaternion,
        )
        max_observed_orientation_error_deg = max(
            max_observed_orientation_error_deg,
            after_orientation_error,
        )
        if (
            not guidance_accepted
            and after_orientation_error > orientation_divergence_limit_deg
        ):
            reason = "orientation_error_diverged"
            last_step["orientation_divergence_limit_deg"] = float(
                orientation_divergence_limit_deg
            )
            break
        after_pose_score = max(
            after_position_error / position_scale,
            after_orientation_error / orientation_scale,
        )
        if (
            after_position_error <= max(0.001, float(pos_tol))
            and after_orientation_error <= max(0.1, float(ori_tol_deg))
        ):
            reason = "target_converged"
            best_pose_score = min(best_pose_score, after_pose_score)
            break
        if after_pose_score < best_pose_score - ADJUST_EEF_POSE_PROGRESS_EPS:
            best_pose_score = after_pose_score
            pose_stall_steps = 0
        else:
            pose_stall_steps += 1
        command_inf = float(safety_observation["command_inf_rad"])
        near_zero_command_steps = (
            near_zero_command_steps + 1
            if command_inf <= ADJUST_EEF_NEAR_ZERO_COMMAND_RAD
            else 0
        )
        if near_zero_command_steps >= ADJUST_EEF_NEAR_ZERO_COMMAND_STEPS:
            reason = "near_zero_command_stall"
            break
        if pose_stall_steps >= ADJUST_EEF_POSE_STALL_STEPS:
            reason = "pose_error_stall"
            break
        adaptive_step_scale = _adjust_next_step_scale(
            adaptive_step_scale,
            safety_observation,
            near_target=bool(
                after_position_error
                <= max(
                    ADJUST_EEF_ADAPTIVE_NEAR_POSITION_M,
                    2.0 * float(pos_tol),
                )
                and after_orientation_error
                <= max(
                    ADJUST_EEF_ADAPTIVE_NEAR_ORIENTATION_DEG,
                    2.0 * float(ori_tol_deg),
                )
            ),
        )
        previous_dq = delta_q.copy()
    else:
        if not gross_preflight_rejection:
            reason = "fixed_budget_complete"

    pre_settle_position_error = float(
        np.linalg.norm(
            np.asarray(target_position, dtype=np.float64)
            - last_observed["position"]
        )
    )
    pre_settle_orientation_error = orientation_error_deg(
        last_observed["quaternion"],
        target_quaternion,
    )
    target_reached_before_settle = bool(
        pre_settle_position_error <= max(0.001, float(pos_tol))
        and pre_settle_orientation_error <= max(0.1, float(ori_tol_deg))
    )
    requested_settle_steps = _adjust_settle_steps(
        last_observed,
        target_reached=target_reached_before_settle,
        skip_when_reached=bool(skip_success_settle),
    )
    hold_report = yield from _adjust_latch_measured_hold(
        ctx,
        arm=arm,
        adapter=adapter,
        observation_sequence=sequence,
        measured=last_observed,
        settle_steps=requested_settle_steps,
    )
    settle_steps = int(hold_report["steps"])
    sequence = int(hold_report["last_observation_sequence"])

    final = _adjust_kinematic_observation(ctx, arm)
    final_position_error = float(
        np.linalg.norm(np.asarray(target_position) - final["position"])
    )
    final_orientation_error = orientation_error_deg(
        final["quaternion"],
        target_quaternion,
    )
    max_observed_orientation_error_deg = max(
        max_observed_orientation_error_deg,
        final_orientation_error,
    )
    converged = bool(
        final_position_error <= max(0.001, float(pos_tol))
        and final_orientation_error <= max(0.1, float(ori_tol_deg))
    )
    requested_delta = np.asarray(target_position, dtype=np.float64) - start_position
    actual_delta = final["position"] - start_position
    requested_distance = float(np.linalg.norm(requested_delta))
    actual_distance = float(np.linalg.norm(actual_delta))
    if requested_distance > 1e-9:
        requested_direction = requested_delta / requested_distance
        projected_progress = float(actual_delta @ requested_direction)
        lateral_delta = actual_delta - projected_progress * requested_direction
        lateral_error = float(np.linalg.norm(lateral_delta))
        progress_ratio = float(
            np.clip(projected_progress / requested_distance, 0.0, 1.0)
        )
    else:
        projected_progress = 0.0
        lateral_error = actual_distance
        progress_ratio = 1.0
    loop_exit_reason = reason
    if converged and reason in {
        "max_steps",
        "fixed_budget_complete",
        "timeout",
        "near_zero_command_stall",
        "pose_error_stall",
    }:
        reason = "target_converged_after_settle"
    command_completed = bool(
        converged
        and reason
        in {
            "target_converged",
            "target_converged_after_settle",
        }
    )
    return {
        "ok": bool(command_completed),
        "command_completed": bool(command_completed),
        "converged": converged,
        "reason": reason,
        "loop_exit_reason": loop_exit_reason,
        "controller": "observation_relinearized_static_urdf_tangent_ik_7dof",
        "observation_action_loop": True,
        "reference_frame": "robot_base",
        "action_dim": int(ACTION_DIM),
        "action_only": True,
        "j8_participates": False,
        "j8_target_rad": 0.0,
        "max_observed_j8_abs_rad": float(max_observed_j8),
        "target_ik": target_ik,
        "steps_planned": int(execution_steps),
        "nominal_steps_planned": int(nominal_execution_steps),
        "gross_preflight_rejection": bool(gross_preflight_rejection),
        "gross_preflight_normalized_error_limit": float(
            ADJUST_EEF_GROSS_PREFLIGHT_MAX_NORMALIZED_ERROR
        ),
        "relinearization_multiplier": int(relinearization_multiplier),
        "fallback_relinearization_multiplier": int(
            ADJUST_EEF_RELINEARIZATION_MULTIPLIER
        ),
        "guided_relinearization_multiplier": int(
            ADJUST_EEF_GUIDED_RELINEARIZATION_MULTIPLIER
        ),
        "guided_joint_nominal_steps": int(guided_joint_nominal_steps),
        "guided_joint_tracking_multiplier": int(
            ADJUST_EEF_GUIDED_JOINT_TRACKING_MULTIPLIER
        ),
        "guided_joint_tracking_steps": int(guided_joint_tracking_steps),
        "guided_step_scale": float(ADJUST_EEF_GUIDED_STEP_SCALE),
        "guided_steps_executed": int(guided_steps),
        "guided_fallback_steps": int(guided_fallbacks),
        "tangent_steps_executed": int(tangent_steps),
        "pose_stall_steps": int(pose_stall_steps),
        "pose_stall_limit": int(ADJUST_EEF_POSE_STALL_STEPS),
        "orientation_divergence_guard_active": bool(
            not guidance_accepted
        ),
        "orientation_divergence_limit_deg": float(
            orientation_divergence_limit_deg
        ),
        "max_observed_orientation_error_deg": float(
            max_observed_orientation_error_deg
        ),
        "near_zero_command_steps": int(near_zero_command_steps),
        "near_zero_command_limit": int(ADJUST_EEF_NEAR_ZERO_COMMAND_STEPS),
        "best_normalized_max_pose_error": float(best_pose_score),
        "steps_executed": int(action_steps),
        "settle_steps_executed": int(settle_steps),
        "settle_steps_requested": int(requested_settle_steps),
        "target_reached_before_settle": bool(target_reached_before_settle),
        "measured_hold_observation_advanced": bool(
            hold_report["observation_advanced"]
        ),
        "max_tracking_error_inf_rad": float(
            safety_monitor.max_tracking_error_rad
        ),
        "adaptive_step_scale_max": float(max_adaptive_step_scale),
        "adaptive_step_scale_min": float(min_adaptive_step_scale),
        "adaptive_step_scale_limit": float(ADJUST_EEF_ADAPTIVE_MAX_SCALE),
        "safety": _adjust_safety_report(safety_monitor),
        "q_start_rad": q_start.tolist(),
        "q_final_rad": final["q"].tolist(),
        "eef_start_robot_m": start_position.tolist(),
        "eef_final_robot_m": final["position"].tolist(),
        "requested_distance_m": requested_distance,
        "actual_distance_m": actual_distance,
        "actual_delta_robot_m": actual_delta.tolist(),
        "projected_progress_m": projected_progress,
        "lateral_error_m": lateral_error,
        "progress_ratio": progress_ratio,
        "final_position_error_m": final_position_error,
        "final_orientation_error_deg": final_orientation_error,
        "last_step": last_step,
        "last_observation_sequence": int(sequence),
    }



def _execute_local_j567_closed_loop(
    ctx,
    *,
    arm: str,
    adapter,
    observation_sequence: int,
    target_quaternion: np.ndarray,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
    stage_name: str,
    deadline_monotonic: float | None = None,
    skip_success_settle: bool = False,
):
    deadline = (
        float(deadline_monotonic)
        if deadline_monotonic is not None
        else time.monotonic() + max(1.0, float(timeout_s))
    )
    initial = _adjust_kinematic_observation(ctx, arm)
    safety_monitor = _adjust_safety_monitor(ctx, initial)
    q_start = initial["q"].copy()
    if time.monotonic() >= deadline:
        return {
            "ok": False,
            "command_completed": False,
            "reason": "timeout_before_orientation_solve",
            "orientation_solve": {
                "ok": False,
                "skipped": True,
                "reason": "shared tool timeout exhausted",
                "fixed_joint_numbers": [1, 2, 3, 4],
                "commanded_joint_numbers": [5, 6, 7],
                "j8_participates": False,
            },
            "action_only": True,
            "controller": "observation_tracked_static_urdf_j567_target",
            "controller_steps": 0,
            "settle_steps": 0,
            "settle_steps_requested": 0,
            "fixed_q1234_rad": q_start[:4].tolist(),
            "commanded_joint_numbers": [5, 6, 7],
            "j8_participates": False,
            "j8_target_rad": 0.0,
            "last_observation_sequence": int(observation_sequence),
        }
    target_q, solve_report = solve_j567_orientation_target(
        initial["state"],
        arm,
        q_start,
        target_quaternion,
        ori_tol_deg=float(ori_tol_deg),
    )
    if not solve_report.get("ok", False):
        return {
            "ok": False,
            "reason": "j567_target_unreachable",
            "orientation_solve": solve_report,
            "controller_steps": 0,
            "fixed_q1234_rad": q_start[:4].tolist(),
            "commanded_joint_numbers": [5, 6, 7],
            "j8_participates": False,
            "last_observation_sequence": int(observation_sequence),
        }

    fixed_q1234 = q_start[:4].copy()
    tolerance_rad = math.radians(max(0.25, min(float(ori_tol_deg), 1.0)))
    sequence = int(observation_sequence)
    steps = 0
    adaptive_step_scale = 1.0
    max_adaptive_step_scale = 1.0
    min_adaptive_step_scale = 1.0
    reason = "max_steps"
    max_j1234_drift = 0.0
    max_observed_j8 = abs(float(initial["observed_j8"]))
    last_observed = initial
    for step in range(min(max(1, int(max_steps)), 120)):
        _check_cancelled(ctx, f"{stage_name} step {step + 1}")
        if time.monotonic() >= deadline:
            reason = "timeout"
            break
        observed = _adjust_kinematic_observation(ctx, arm)
        last_observed = observed
        max_observed_j8 = max(
            max_observed_j8,
            abs(float(observed["observed_j8"])),
        )
        error = target_q[4:7] - observed["q"][4:7]
        max_j1234_drift = max(
            max_j1234_drift,
            float(np.linalg.norm(observed["q"][:4] - fixed_q1234, ord=np.inf)),
        )
        if float(np.max(np.abs(error))) <= tolerance_rad:
            reason = "target_converged"
            break
        max_adaptive_step_scale = max(
            max_adaptive_step_scale,
            adaptive_step_scale,
        )
        min_adaptive_step_scale = min(
            min_adaptive_step_scale,
            adaptive_step_scale,
        )
        command = observed["q"].copy()
        command[:4] = fixed_q1234
        command[4:7] += np.clip(
            error,
            -WRIST_MAX_JOINT_STEP_RAD * adaptive_step_scale,
            WRIST_MAX_JOINT_STEP_RAD * adaptive_step_scale,
        )
        if ARM_DOF == 8:
            command[7] = 0.0
        yield _adjust_action(ctx, arm, command)
        steps += 1
        advanced, current_sequence = _adjust_observation_advanced(adapter, sequence)
        if not advanced:
            reason = "observation_not_advanced"
            break
        sequence = current_sequence
        after_action = _adjust_kinematic_observation(ctx, arm)
        last_observed = after_action
        safety_reason, safety_observation = _adjust_safety_after_action(
            ctx,
            safety_monitor,
            before=observed,
            command=command,
            after=after_action,
        )
        if safety_reason is not None:
            reason = safety_reason
            break
        remaining_joint_error = float(
            np.linalg.norm(
                target_q[4:7] - after_action["q"][4:7],
                ord=np.inf,
            )
        )
        adaptive_step_scale = _adjust_next_step_scale(
            adaptive_step_scale,
            safety_observation,
            near_target=bool(remaining_joint_error <= 2.0 * tolerance_rad),
        )
    else:
        reason = "fixed_budget_complete"

    target_reached_before_settle = bool(
        float(
            np.linalg.norm(
                target_q[4:7] - last_observed["q"][4:7],
                ord=np.inf,
            )
        )
        <= tolerance_rad
    )
    requested_settle_steps = _adjust_settle_steps(
        last_observed,
        target_reached=target_reached_before_settle,
        skip_when_reached=bool(skip_success_settle),
    )
    hold_report = yield from _adjust_latch_measured_hold(
        ctx,
        arm=arm,
        adapter=adapter,
        observation_sequence=sequence,
        measured=last_observed,
        settle_steps=requested_settle_steps,
    )
    settle_steps = int(hold_report["steps"])
    sequence = int(hold_report["last_observation_sequence"])

    final = _adjust_kinematic_observation(ctx, arm)
    joint_error_deg = np.degrees(target_q - final["q"])
    max_j1234_drift = max(
        max_j1234_drift,
        float(np.linalg.norm(final["q"][:4] - fixed_q1234, ord=np.inf)),
    )
    converged = bool(
        np.max(np.abs(joint_error_deg[4:7]))
        <= max(0.25, float(ori_tol_deg))
    )
    if converged and reason in {"max_steps", "fixed_budget_complete"}:
        reason = "target_converged_after_settle"
    command_completed = bool(
        converged
        and reason
        in {
            "target_converged",
            "target_converged_after_settle",
        }
    )
    return {
        "ok": command_completed,
        "command_completed": command_completed,
        "reason": reason,
        "orientation_solve": solve_report,
        "action_only": True,
        "controller": "observation_tracked_static_urdf_j567_target",
        "controller_steps": int(steps),
        "settle_steps": int(settle_steps),
        "settle_steps_requested": int(requested_settle_steps),
        "target_reached_before_settle": bool(target_reached_before_settle),
        "measured_hold_observation_advanced": bool(
            hold_report["observation_advanced"]
        ),
        "safety": _adjust_safety_report(safety_monitor),
        "adaptive_step_scale_max": float(max_adaptive_step_scale),
        "adaptive_step_scale_min": float(min_adaptive_step_scale),
        "adaptive_step_scale_limit": float(ADJUST_EEF_ADAPTIVE_MAX_SCALE),
        "qpos_start_deg": np.degrees(q_start).round(4).tolist(),
        "qpos_target_deg": np.degrees(target_q).round(4).tolist(),
        "qpos_after_deg": np.degrees(final["q"]).round(4).tolist(),
        "final_error_deg": joint_error_deg.round(4).tolist(),
        "fixed_q1234_rad": fixed_q1234.tolist(),
        "commanded_joint_numbers": [5, 6, 7],
        "max_measured_j1234_drift_rad": float(max_j1234_drift),
        "j8_participates": False,
        "j8_target_rad": 0.0,
        "max_observed_j8_abs_rad": float(max_observed_j8),
        "last_observation_sequence": int(sequence),
    }



def _validate_adjust_values(
    *,
    translation: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
) -> np.ndarray:
    values = np.asarray(
        [
            *translation,
            roll,
            pitch,
            yaw,
            pos_tol,
            ori_tol_deg,
            max_steps,
            timeout_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("all EEF adjustment values must be finite")
    if float(pos_tol) <= 0.0:
        raise ValueError("pos_tol must be positive")
    if float(ori_tol_deg) <= 0.0:
        raise ValueError("ori_tol_deg must be positive")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be positive")
    return values



def _head_translation_request(
    *,
    x: float | None,
    y: float | None,
    z: float | None,
    forward: float | None,
    leftward: float | None,
    upward: float | None,
) -> dict[str, Any]:
    base_raw = {"x": x, "y": y, "z": z}
    camera_raw = {
        "forward": forward,
        "leftward": leftward,
        "upward": upward,
    }
    base_present = [key for key, value in base_raw.items() if value is not None]
    camera_present = [
        key for key, value in camera_raw.items() if value is not None
    ]
    if base_present and camera_present:
        raise ValueError(
            "head-frame translation parameter families are mutually exclusive: "
            "use only x/y/z (robot base frame) or only "
            "forward/leftward/upward (head camera frame); got base keys "
            f"{base_present} and camera keys {camera_present}"
        )

    if base_present:
        requested = {
            key: float(0.0 if value is None else value)
            for key, value in base_raw.items()
        }
        return {
            "mode": "robot_base_xyz",
            "frame": "robot_base",
            "provided": base_present,
            "requested": requested,
            "vector": (requested["x"], requested["y"], requested["z"]),
        }

    requested = {
        key: float(0.0 if value is None else value)
        for key, value in camera_raw.items()
    }
    return {
        "mode": "head_camera_forward_leftward_upward",
        "frame": "head_camera",
        "provided": camera_present,
        "requested": requested,
        "vector": (
            requested["forward"],
            requested["leftward"],
            requested["upward"],
        ),
    }



def _adjust_eef_pose_in_head_frame_local(
    ctx,
    *,
    arm: str,
    tool_name: str,
    forward: float | None,
    upward: float | None,
    leftward: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
    roll: float,
    pitch: float,
    yaw: float,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
):
    try:
        translation = _head_translation_request(
            x=x,
            y=y,
            z=z,
            forward=forward,
            leftward=leftward,
            upward=upward,
        )
        values = _validate_adjust_values(
            translation=translation["vector"],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            pos_tol=pos_tol,
            ori_tol_deg=ori_tol_deg,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        camera, sequence, adapter = _adjust_camera_pose(ctx, "head")
        initial = _adjust_kinematic_observation(ctx, arm)
        if ARM_DOF == 8 and abs(float(initial["observed_j8"])) > LOCKED_J8_CAPTURE_TOLERANCE_RAD:
            raise RuntimeError(
                f"{arm} J8 is not locked at 0rad in evaluator proprioception"
            )
        if translation["frame"] == "robot_base":
            target = robot_base_frame_pose_target(
                eef_position=initial["position"],
                eef_quaternion=initial["quaternion"],
                camera_quaternion=camera["quat"],
                x=translation["requested"]["x"],
                y=translation["requested"]["y"],
                z=translation["requested"]["z"],
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )
        else:
            target = camera_frame_pose_target(
                eef_position=initial["position"],
                eef_quaternion=initial["quaternion"],
                camera_quaternion=camera["quat"],
                forward=translation["requested"]["forward"],
                leftward=translation["requested"]["leftward"],
                upward=translation["requested"]["upward"],
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        _set_result(ctx, tool_name, {"ok": False, "error": str(exc)})
        yield ctx.world.hold_action()
        return

    move_position = bool(np.linalg.norm(values[:3]) > 1e-8)
    move_orientation = bool(np.any(np.abs(values[3:6]) > 1e-9))
    if not move_position and not move_orientation:
        _set_result(
            ctx,
            tool_name,
            {
                "ok": True,
                "build": ADJUST_EEF_LOCAL_BUILD,
                "arm": arm,
                "feed": "head",
                "note": "all deltas are zero; no action",
                "translation_input_mode": translation["mode"],
                "translation_frame": translation["frame"],
                "translation_m": translation["requested"],
                "j8_participates": False,
                "j8_target_rad": 0.0,
                "camera_pose_source": "evaluator_cam_rel_poses",
                "finger_qpos": initial["gripper"].tolist(),
            },
        )
        yield ctx.world.hold_action()
        return

    pose_report = yield from _execute_local_pose_closed_loop(
        ctx,
        arm=arm,
        adapter=adapter,
        observation_sequence=sequence,
        target_position=target.position,
        target_quaternion=target.quaternion,
        pos_tol=float(pos_tol),
        ori_tol_deg=float(ori_tol_deg),
        max_steps=int(max_steps),
        timeout_s=float(timeout_s),
        stage_name=f"{tool_name}.pose",
    )
    final = _adjust_kinematic_observation(ctx, arm)
    position_error = float(np.linalg.norm(final["position"] - target.position))
    orientation_error = orientation_error_deg(
        final["quaternion"],
        target.quaternion,
    )
    position_ok = bool(
        not move_position or position_error <= max(0.001, float(pos_tol))
    )
    orientation_limit = max(float(ori_tol_deg) * 2.0, 6.0)
    orientation_ok = bool(orientation_error <= orientation_limit)
    j8_ok = bool(
        ARM_DOF != 8
        or abs(float(final["observed_j8"])) <= LOCKED_J8_CAPTURE_TOLERANCE_RAD
    )
    overall_ok = bool(
        pose_report.get("ok", False)
        and position_ok
        and orientation_ok
        and j8_ok
    )
    failure_reason = None
    if not overall_ok:
        reason = str(pose_report.get("reason", "pose_controller_incomplete"))
        steps_executed = int(pose_report.get("steps_executed", 0))
        steps_planned = int(pose_report.get("steps_planned", 0))
        if reason == "timeout":
            reason_text = (
                "pose control timed out after "
                f"{steps_executed}/{steps_planned} fresh observation/action steps"
            )
        elif reason == "terminal_ik_grossly_unreachable":
            reason_text = (
                "pose control rejected before arm motion because the terminal "
                "pose is grossly unreachable in the submission-local model"
            )
        else:
            reason_text = f"pose control stopped with reason={reason}"
        details = [reason_text]
        target_ik = pose_report.get("target_ik") or {}
        if not bool(target_ik.get("ok", False)):
            details.append(
                "terminal-pose local IK preflight did not converge "
                f"(position residual {float(target_ik.get('pos_err_m', float('nan'))) * 1000.0:.1f}mm, "
                f"orientation residual {float(target_ik.get('ori_err_deg', float('nan'))):.2f}deg)"
            )
        details.append(
            f"measured requested-direction progress "
            f"{float(pose_report.get('projected_progress_m', 0.0)):.3f}m/"
            f"{float(pose_report.get('requested_distance_m', 0.0)):.3f}m"
        )
        if not position_ok:
            details.append(f"final position error {position_error * 1000.0:.1f}mm")
        if not orientation_ok:
            details.append(
                f"final orientation error {orientation_error:.2f}deg exceeds "
                f"{orientation_limit:.2f}deg"
            )
        if not j8_ok:
            details.append("observed J8 is outside its 0rad lock tolerance")
        failure_reason = "; ".join(details)
    _set_result(
        ctx,
        tool_name,
        {
            "ok": overall_ok,
            "build": ADJUST_EEF_LOCAL_BUILD,
            "arm": arm,
            "feed": "head",
            "translation_input_mode": translation["mode"],
            "translation_frame": translation["frame"],
            "translation_parameters_present": translation["provided"],
            "translation_m": translation["requested"],
            "delta_head_camera_m": target.delta_camera.round(6).tolist(),
            "delta_robot_base_m": target.delta_robot.round(6).tolist(),
            "rotation_deg": {
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
            "rpy_deg": {
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
            "convention": {
                "translation": (
                    "robot base +X chassis-forward, +Y chassis-left, +Z chassis-up"
                    if translation["frame"] == "robot_base"
                    else "head camera -Z/-X/+Y for forward/leftward/upward"
                ),
                "translation_parameter_families": (
                    "x/y/z and forward/leftward/upward are mutually exclusive"
                ),
                "rotation": "local gripper roll/pitch/yaw; positive pitch is fingertips-up",
            },
            "rpy_convention": {
                "translation": (
                    "x/y/z are robot-base-frame translation deltas"
                    if translation["frame"] == "robot_base"
                    else "forward/leftward/upward are head-camera-frame translation deltas"
                ),
                "rotation": "roll/pitch/yaw are local gripper axes",
                "positive_pitch": "positive pitch is nose-up / fingertips-up",
            },
            "camera_before": {
                "pos": camera["pos"].round(6).tolist(),
                "quat": camera["quat"].round(6).tolist(),
                "frame": "robot_base",
                "source": "evaluator_cam_rel_poses",
            },
            "delta_camera_m": target.delta_camera.round(6).tolist(),
            "delta_robot_m": target.delta_robot.round(6).tolist(),
            "delta_world_m": target.delta_robot.round(6).tolist(),
            "eef_before": initial["position"].round(6).tolist(),
            "eef_target": target.position.round(6).tolist(),
            "eef_after": final["position"].round(6).tolist(),
            "quat_before": initial["quaternion"].round(6).tolist(),
            "quat_target": target.quaternion.round(6).tolist(),
            "quat_after": final["quaternion"].round(6).tolist(),
            "kinematics_source": "submission_local_static_urdf_fk",
            "pos_err_mm": round(position_error * 1000.0, 3),
            "ori_err_deg": round(orientation_error, 3),
            "pos_ok": position_ok,
            "ori_ok": orientation_ok,
            "target_reached": bool(position_ok and orientation_ok),
            "pose_control": pose_report,
            "failure_reason": failure_reason,
            "external_object_motion_checked": False,
            "external_object_motion_evidence": (
                "requires a subsequent evaluator RGB-D observation"
            ),
            "finger_qpos": final["gripper"].tolist(),
            "j8_participates": False,
            "j8_target_rad": 0.0,
            "j8_ok": j8_ok,
            "observation_contract": [
                "evaluator_proprioception",
                "evaluator_head_cam_rel_poses",
                "submission_local_static_robot_model",
            ],
            "simulator_fk_ik_used": False,
            "simulator_eef_pose_used": False,
            "direct_simulator_mutation": False,
            "error": failure_reason,
        },
    )
    yield ctx.world.hold_action()



def _adjust_eef_pose_in_wrist_frame_local(
    ctx,
    *,
    arm: str,
    tool_name: str,
    forward: float,
    leftward: float,
    upward: float,
    roll: float,
    pitch: float,
    yaw: float,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
):
    try:
        values = _validate_adjust_values(
            translation=(forward, leftward, upward),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            pos_tol=pos_tol,
            ori_tol_deg=ori_tol_deg,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        camera, sequence, adapter = _adjust_camera_pose(ctx, f"{arm}_wrist")
        initial = _adjust_kinematic_observation(ctx, arm)
        if ARM_DOF == 8 and abs(float(initial["observed_j8"])) > LOCKED_J8_CAPTURE_TOLERANCE_RAD:
            raise RuntimeError(
                f"{arm} J8 is not locked at 0rad in evaluator proprioception"
            )
        target = camera_frame_pose_target(
            eef_position=initial["position"],
            eef_quaternion=initial["quaternion"],
            camera_quaternion=camera["quat"],
            forward=float(forward),
            leftward=float(leftward),
            upward=float(upward),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        _set_result(ctx, tool_name, {"ok": False, "error": str(exc)})
        yield ctx.world.hold_action()
        return

    move_position = bool(np.linalg.norm(values[:3]) > 1e-8)
    move_orientation = bool(np.any(np.abs(values[3:6]) > 1e-9))
    joint_mapping = {
        "fixed": [f"{arm}_arm_joint{index}" for index in range(1, 5)],
        "solved": [f"{arm}_arm_joint{index}" for index in range(5, 8)],
        "locked": [f"{arm}_arm_joint8"] if ARM_DOF == 8 else [],
    }
    if not move_position and not move_orientation:
        _set_result(
            ctx,
            tool_name,
            {
                "ok": True,
                "build": ADJUST_EEF_LOCAL_BUILD,
                "arm": arm,
                "feed": f"{arm}_wrist",
                "note": "all deltas are zero; no action",
                "joint_mapping": joint_mapping,
                "j8_participates": False,
                "j8_target_rad": 0.0,
                "camera_pose_source": "evaluator_cam_rel_poses",
            },
        )
        yield ctx.world.hold_action()
        return

    translation_report: dict[str, Any] = {
        "ok": True,
        "converged": True,
        "reason": "no_translation",
        "steps_executed": 0,
        "j8_participates": False,
    }
    control_deadline = time.monotonic() + max(1.0, float(timeout_s))
    if move_position:
        translation_report = yield from _execute_local_pose_closed_loop(
            ctx,
            arm=arm,
            adapter=adapter,
            observation_sequence=sequence,
            target_position=target.position,
            target_quaternion=initial["quaternion"],
            pos_tol=float(pos_tol),
            ori_tol_deg=float(ori_tol_deg),
            max_steps=int(max_steps),
            timeout_s=float(timeout_s),
            stage_name=f"{tool_name}.translation",
            deadline_monotonic=control_deadline,
            skip_success_settle=bool(move_orientation),
        )
        sequence = int(
            translation_report.get("last_observation_sequence", sequence)
        )

    after_translation = _adjust_kinematic_observation(ctx, arm)
    translation_position_error = float(
        np.linalg.norm(after_translation["position"] - target.position)
    )
    translation_orientation_error = orientation_error_deg(
        after_translation["quaternion"],
        initial["quaternion"],
    )
    translation_orientation_limit = max(float(ori_tol_deg) * 2.0, 6.0)
    translation_position_ok = bool(
        not move_position
        or translation_position_error <= max(0.001, float(pos_tol))
    )
    translation_orientation_ok = bool(
        not move_position
        or move_orientation
        or translation_orientation_error <= translation_orientation_limit
    )

    q_before_rotation = after_translation["q"].copy()
    orientation_motion: dict[str, Any] = {
        "ok": True,
        "reason": "no_rotation",
        "controller_steps": 0,
        "fixed_q1234_rad": q_before_rotation[:4].tolist(),
        "commanded_joint_numbers": [5, 6, 7],
        "j8_participates": False,
        "last_observation_sequence": int(sequence),
    }
    if move_orientation:
        orientation_motion = yield from _execute_local_j567_closed_loop(
            ctx,
            arm=arm,
            adapter=adapter,
            observation_sequence=sequence,
            target_quaternion=target.quaternion,
            ori_tol_deg=float(ori_tol_deg),
            max_steps=int(max_steps),
            timeout_s=float(timeout_s),
            stage_name=f"{tool_name}.orientation",
            deadline_monotonic=control_deadline,
        )

    final = _adjust_kinematic_observation(ctx, arm)
    final_position_error = float(
        np.linalg.norm(final["position"] - target.position)
    )
    final_orientation_error = orientation_error_deg(
        final["quaternion"],
        target.quaternion,
    )
    final_position_ok = bool(
        not move_position
        or final_position_error <= max(0.001, float(pos_tol))
    )
    final_orientation_limit = (
        max(0.25, float(ori_tol_deg))
        if move_orientation
        else translation_orientation_limit
    )
    final_orientation_ok = bool(
        final_orientation_error <= final_orientation_limit
    )
    j1234_drift = float(
        np.linalg.norm(final["q"][:4] - q_before_rotation[:4], ord=np.inf)
    )
    j1234_held = bool(
        not move_orientation
        or j1234_drift <= math.radians(max(0.25, float(ori_tol_deg)))
    )
    j8_ok = bool(
        ARM_DOF != 8
        or abs(float(final["observed_j8"])) <= LOCKED_J8_CAPTURE_TOLERANCE_RAD
    )
    overall_ok = bool(
        (not move_position or translation_report.get("ok", False))
        and translation_position_ok
        and translation_orientation_ok
        and orientation_motion.get("ok", False)
        and final_position_ok
        and final_orientation_ok
        and j1234_held
        and j8_ok
    )
    orientation_solve = dict(
        orientation_motion.get("orientation_solve")
        or {
            "ok": True,
            "skipped": True,
            "reason": "no_rotation",
            "fixed_joint_numbers": [1, 2, 3, 4],
            "commanded_joint_numbers": [5, 6, 7],
            "j8_participates": False,
        }
    )
    tool_roll_motion = {
        "ok": j8_ok,
        "active": bool(ARM_DOF == 8),
        "joint": f"{arm}_arm_joint8" if ARM_DOF == 8 else None,
        "qpos_before_deg": round(math.degrees(initial["observed_j8"]), 4),
        "qpos_target_deg": 0.0,
        "qpos_after_deg": round(math.degrees(final["observed_j8"]), 4),
        "locked_after_motion": j8_ok,
        "lock_target_deg": 0.0,
        "j8_participates": False,
        "note": "J8 is fixed by the official robot profile and excluded from EEF control",
    }
    _set_result(
        ctx,
        tool_name,
        {
            "ok": overall_ok,
            "build": ADJUST_EEF_LOCAL_BUILD,
            "robot_dof": int(ARM_DOF),
            "arm": arm,
            "feed": f"{arm}_wrist",
            "translation_m": {
                "forward": float(forward),
                "leftward": float(leftward),
                "upward": float(upward),
            },
            "rotation_deg": {
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
            "rotation_mode": "local gripper RPY solved with fixed-J1234 J567 IK",
            "shared_timeout_budget_s": float(timeout_s),
            "translation_and_orientation_share_timeout": True,
            "joint_mapping": joint_mapping,
            "orientation_solve": orientation_solve,
            "convention": {
                "positive_leftward": "image left / camera -X",
                "translation_stage_orientation": (
                    "the initial EEF orientation is held during translation; "
                    "the final orientation is applied afterward with fixed J1-J4"
                ),
                "rotation_mode": "local gripper RPY target solved with J5-J7 only",
                "solver_joint_numbers": [5, 6, 7],
                "fixed_joint_numbers": [1, 2, 3, 4],
                "j8_participates": False,
            },
            "camera_before": {
                "pos": camera["pos"].round(6).tolist(),
                "quat": camera["quat"].round(6).tolist(),
                "frame": "robot_base",
                "source": "evaluator_cam_rel_poses",
            },
            "wrist_camera_before": {
                "pos": camera["pos"].round(6).tolist(),
                "quat": camera["quat"].round(6).tolist(),
            },
            "delta_camera_m": target.delta_camera.round(6).tolist(),
            "delta_robot_m": target.delta_robot.round(6).tolist(),
            "eef_before": {
                "pos": initial["position"].round(6).tolist(),
                "quat": initial["quaternion"].round(6).tolist(),
                "source": "submission_local_static_urdf_fk",
            },
            "eef_target": {
                "pos": target.position.round(6).tolist(),
                "quat": target.quaternion.round(6).tolist(),
            },
            "eef_after_translation": {
                "pos": after_translation["position"].round(6).tolist(),
                "quat": after_translation["quaternion"].round(6).tolist(),
            },
            "eef_after": {
                "pos": final["position"].round(6).tolist(),
                "quat": final["quaternion"].round(6).tolist(),
                "source": "submission_local_static_urdf_fk",
            },
            "translation_stage_pos_err_mm": round(
                translation_position_error * 1000.0,
                3,
            ),
            "translation_stage_ori_err_deg": round(
                translation_orientation_error,
                3,
            ),
            "pos_err_mm": round(final_position_error * 1000.0, 3),
            "ori_err_deg": round(final_orientation_error, 3),
            "translation_stage_pos_ok": translation_position_ok,
            "translation_stage_ori_ok": translation_orientation_ok,
            "pos_ok": final_position_ok,
            "ori_ok": final_orientation_ok,
            "target_reached": bool(final_position_ok and final_orientation_ok),
            "translation_control": translation_report,
            "translation_command_ok": bool(
                not move_position or translation_report.get("ok", False)
            ),
            "wrist_joint_motion": {
                "arm_joints": orientation_motion,
                "tool_roll_j8": tool_roll_motion,
                "body_lock": {
                    "active": True,
                    "mode": "official action targets hold base, trunk, and inactive arm",
                    "direct_root_lock": False,
                },
            },
            "tool_roll_lock": tool_roll_motion,
            "legacy_7dof_precondition": {
                "required": bool(ARM_DOF == 8),
                "reset": False,
                "replacement": "episode initialization and profile-level J8=0 action lock",
                "direct_joint_write": False,
            },
            "arm_qpos_before_rotation_deg": np.degrees(
                q_before_rotation
            ).round(4).tolist(),
            "arm_qpos_after_all_deg": np.degrees(final["q"]).round(4).tolist(),
            "arm_qpos_delta_after_all_deg": np.degrees(
                final["q"] - q_before_rotation
            ).round(5).tolist(),
            "j1234_command_policy": "fixed at post-translation observed values for every rotation action",
            "j1234_measured_drift_inf_rad": j1234_drift,
            "j1234_held": j1234_held,
            "j567_only_orientation_control": True,
            "j8_participates": False,
            "j8_target_rad": 0.0,
            "j8_ok": j8_ok,
            "finger_qpos": final["gripper"].tolist(),
            "kinematic_snap": {
                "snapped": False,
                "disabled": True,
                "reason": "motion is observation-tracked official action output only",
            },
            "observation_contract": [
                "evaluator_proprioception",
                f"evaluator_{arm}_wrist_cam_rel_poses",
                "submission_local_static_robot_model",
            ],
            "simulator_fk_ik_used": False,
            "simulator_eef_pose_used": False,
            "direct_simulator_mutation": False,
            "error": None if overall_ok else "local observation/action wrist adjustment did not converge",
        },
    )
    yield ctx.world.hold_action()



def _move_tracked_strict_point_matrix(
    raw: Any,
    *,
    expected_count: int,
    label: str,
) -> np.ndarray:
    """Validate a live/trajectory point bundle without flattening it.

    Point boundaries are part of the tracked-point contract.  A flattened
    ``3*N`` buffer must therefore be rejected instead of being silently
    reshaped into a plausible marker matrix.
    """

    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nested N x 3 matrix") from exc
    if matrix.ndim != 2 or matrix.shape != (int(expected_count), 3):
        raise ValueError(
            f"{label} must be a nested N x 3 matrix with shape "
            f"({int(expected_count)}, 3)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain finite coordinates")
    return matrix.copy()



def _move_tracked_quick_constraint_specs(
    *,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return one canonical list without changing the legacy single-group wire form."""

    if quick_constraint is not None and quick_constraints is not None:
        raise ValueError("quick_constraint and quick_constraints are mutually exclusive")
    if quick_constraint is not None:
        if not isinstance(quick_constraint, Mapping):
            raise ValueError("quick_constraint must be an object")
        return [dict(quick_constraint)]
    if quick_constraints is None:
        return []
    if isinstance(quick_constraints, (str, bytes)) or not isinstance(
        quick_constraints, Sequence
    ):
        raise ValueError("quick_constraints must be an array")
    if not 1 <= len(quick_constraints) <= MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS:
        raise ValueError(
            "quick_constraints must contain one to "
            f"{MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS} groups"
        )
    result: list[dict[str, Any]] = []
    for index, constraint in enumerate(quick_constraints):
        if not isinstance(constraint, Mapping):
            raise ValueError(f"quick_constraints[{index}] must be an object")
        result.append(dict(constraint))
    return result



def _move_tracked_quick_group_points(
    quick_constraint: Mapping[str, Any],
    *,
    controlled_names: Sequence[str],
    controlled_points: np.ndarray,
    reference_names: Sequence[str],
    reference_points: np.ndarray | None,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray | None]:
    """Select one group's ordered point bundles from the complete frozen binding."""

    controlled_order = [str(name) for name in controlled_names]
    reference_order = [str(name) for name in reference_names]
    controlled = _move_tracked_strict_point_matrix(
        controlled_points,
        expected_count=len(controlled_order),
        label="quick-constraint controlled points",
    )
    if reference_order:
        references = _move_tracked_strict_point_matrix(
            reference_points,
            expected_count=len(reference_order),
            label="quick-constraint off-hand points",
        )
    else:
        references = None
        if reference_points is not None and np.asarray(reference_points).size:
            raise ValueError("quick-constraint reference points have no bound names")
    controlled_index = {name: index for index, name in enumerate(controlled_order)}
    reference_index = {name: index for index, name in enumerate(reference_order)}
    raw_on_names = quick_constraint.get("on_hand_points")
    raw_off_names = quick_constraint.get("off_hand_points")
    if not isinstance(raw_on_names, (list, tuple)) or not isinstance(
        raw_off_names, (list, tuple)
    ):
        raise ValueError("quick constraint group role bindings are missing")
    on_names = [str(name) for name in raw_on_names]
    off_names = [str(name) for name in raw_off_names]
    if any(name not in controlled_index for name in on_names):
        raise ValueError("quick constraint group references a non-on-hand point")
    if any(name not in reference_index for name in off_names):
        raise ValueError("quick constraint group references a non-off-hand point")
    on_points = np.asarray(
        [controlled[controlled_index[name]] for name in on_names],
        dtype=np.float64,
    )
    off_points = (
        np.asarray(
            [references[reference_index[name]] for name in off_names],
            dtype=np.float64,
        )
        if off_names
        else None
    )
    return on_names, on_points, off_names, off_points



def _move_tracked_filter_endpoint_overlap_batch(
    candidate_records: Sequence[tuple[dict[str, Any], str]],
    *,
    overlap_filter_context: Mapping[str, Any],
    stage: str,
) -> tuple[
    list[tuple[dict[str, Any], str]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Batch-filter one endpoint frontier with the exact Lite overlap gates."""

    records = list(candidate_records)
    if not records:
        return [], {
            "ok": True,
            "build": LITE_POSE_OVERLAP_FILTER_BUILD,
            "stage": str(stage),
            "pose_count": 0,
            "passing_pose_count": 0,
            "rejected_pose_count": 0,
            "poses": [],
        }, []
    poses = []
    for candidate_index, (candidate, _preference) in enumerate(records):
        try:
            position = np.asarray(
                candidate["final_eef_position_m"], dtype=np.float64
            ).reshape(3)
            quaternion = np.asarray(
                candidate["final_eef_quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"move-tracked endpoint candidate {candidate_index} has no "
                "finite EEF pose for overlap filtering"
            ) from exc
        if not np.all(np.isfinite(position)) or not np.all(
            np.isfinite(quaternion)
        ):
            raise ValueError(
                f"move-tracked endpoint candidate {candidate_index} has a "
                "non-finite EEF pose for overlap filtering"
            )
        poses.append(
            {
                "eef_pos": position.copy(),
                "quat": quaternion.copy(),
            }
        )

    try:
        evaluation = _evaluate_rgbd_lite_pose_overlaps(
            session=dict(overlap_filter_context["session"]),
            camera_pos=overlap_filter_context["camera_pos"],
            camera_quat_xyzw=overlap_filter_context["camera_quat_xyzw"],
            focal_length=float(overlap_filter_context["focal_length"]),
            horizontal_aperture=float(
                overlap_filter_context["horizontal_aperture"]
            ),
            poses=poses,
            **({
                "check_cancelled": overlap_filter_context["_check_cancelled"],
                "progress_callback": overlap_filter_context.get("_progress_callback"),
            } if "_check_cancelled" in overlap_filter_context else {}),
        )
    except ProjectiveOccupancyContractError:
        # A batch can exceed the dense-grid contract solely because its poses
        # are far apart.  The frontier wrapper splits that spatial envelope
        # and retries without changing the per-pose collision calculation.
        raise
    except Exception as exc:
        raise RuntimeError(
            "move-tracked frozen RGB-D overlap evaluation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(evaluation, Mapping) or not bool(evaluation.get("ok")):
        raise RuntimeError("move-tracked frozen RGB-D overlap evaluation failed")
    pose_reports = evaluation.get("poses")
    if not isinstance(pose_reports, list) or len(pose_reports) != len(records):
        raise RuntimeError(
            "move-tracked frozen RGB-D overlap evaluation returned the wrong "
            "candidate count"
        )
    try:
        original_threshold = float(
            evaluation["original_overlap_threshold_cm3"]
        )
        inflated_threshold = float(
            evaluation["inflated_overlap_threshold_cm3"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "move-tracked frozen RGB-D overlap thresholds are missing"
        ) from exc
    if (
        not math.isclose(
            original_threshold,
            float(ORIGINAL_OVERLAP_MAX_CM3),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            inflated_threshold,
            float(INFLATED_OVERLAP_MAX_CM3),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or evaluation.get("original_overlap_comparison") != "<="
        or evaluation.get("inflated_overlap_comparison") != "<"
    ):
        raise RuntimeError(
            "move-tracked overlap evaluator does not match the Lite thresholds"
        )

    accepted: list[tuple[dict[str, Any], str]] = []
    rejected: list[dict[str, Any]] = []
    decorated_reports: list[dict[str, Any]] = []
    for candidate_index, ((candidate, preference), raw_report) in enumerate(
        zip(records, pose_reports)
    ):
        if not isinstance(raw_report, Mapping):
            raise RuntimeError(
                "move-tracked frozen RGB-D overlap pose report is invalid"
            )
        report = {
            **dict(raw_report),
            "stage": str(stage),
            "candidate_index_in_batch": int(candidate_index),
            "endpoint_preference": str(preference),
            "endpoint_score_rank": int(
                (candidate.get("selected_candidate") or {}).get(
                    "endpoint_score_rank", candidate_index
                )
            ),
        }
        try:
            report_pose_index = int(report["pose_index"])
            report_position = np.asarray(
                report["eef_position_robot_base_m"], dtype=np.float64
            ).reshape(3)
            report_quaternion = np.asarray(
                report["eef_quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "move-tracked frozen RGB-D overlap pose binding is invalid"
            ) from exc
        requested_position = np.asarray(
            poses[candidate_index]["eef_pos"], dtype=np.float64
        ).reshape(3)
        requested_quaternion = np.asarray(
            poses[candidate_index]["quat"], dtype=np.float64
        ).reshape(4)
        report_quaternion_norm = float(np.linalg.norm(report_quaternion))
        requested_quaternion_norm = float(np.linalg.norm(requested_quaternion))
        if (
            report_pose_index != candidate_index
            or not np.all(np.isfinite(report_position))
            or not np.all(np.isfinite(report_quaternion))
            or report_quaternion_norm <= 1.0e-12
            or requested_quaternion_norm <= 1.0e-12
            or not np.allclose(
                report_position,
                requested_position,
                rtol=0.0,
                atol=1.0e-9,
            )
            or abs(
                float(
                    np.dot(
                        report_quaternion / report_quaternion_norm,
                        requested_quaternion / requested_quaternion_norm,
                    )
                )
            )
            < 1.0 - 1.0e-10
        ):
            raise RuntimeError(
                "move-tracked frozen RGB-D overlap result does not match its "
                "endpoint candidate"
            )
        overlap_volume = report.get("overlap_vol_cm3")
        inflated_volume = report.get("inflated_overlap_vol_cm3")
        if overlap_volume is None or not math.isfinite(float(overlap_volume)):
            raise RuntimeError(
                "move-tracked original overlap volume is missing or non-finite"
            )
        if bool(report.get("original_overlap_ok")):
            if inflated_volume is None or not math.isfinite(
                float(inflated_volume)
            ):
                raise RuntimeError(
                    "move-tracked inflated overlap volume is missing or non-finite"
                )
        expected_original_ok = bool(
            float(overlap_volume) <= ORIGINAL_OVERLAP_MAX_CM3 + 1.0e-12
        )
        if bool(report.get("original_overlap_ok")) != expected_original_ok:
            raise RuntimeError(
                "move-tracked original overlap report is inconsistent with "
                "the Lite threshold"
            )
        if not expected_original_ok:
            if (
                inflated_volume is not None
                or bool(report.get("inflated_overlap_ok"))
                or not bool(report.get("inflated_query_skipped"))
            ):
                raise RuntimeError(
                    "move-tracked rejected original overlap did not preserve "
                    "the equivalent inflated-query short circuit"
                )
            expected_inflated_ok = False
        else:
            expected_inflated_ok = bool(
                inflated_volume is not None
                and float(inflated_volume) < INFLATED_OVERLAP_MAX_CM3
            )
            if (
                bool(report.get("inflated_overlap_ok"))
                != expected_inflated_ok
                or bool(report.get("inflated_query_skipped"))
            ):
                raise RuntimeError(
                    "move-tracked inflated overlap report is inconsistent "
                    "with the Lite threshold"
                )
        expected_ok = bool(
            expected_original_ok and expected_inflated_ok
        )
        if bool(report.get("ok")) != expected_ok:
            raise RuntimeError(
                "move-tracked overlap pose report is inconsistent with Lite thresholds"
            )
        candidate["overlap_volume_filter"] = _json_ready(report)
        decorated_reports.append(_json_ready(report))
        if expected_ok:
            accepted.append((candidate, preference))
        else:
            rejected.append(
                {
                    "endpoint_score_rank": int(report["endpoint_score_rank"]),
                    "path_candidate_rank": -1,
                    "endpoint_preference": str(preference),
                    "path_mode": "frozen_rgbd_overlap_filter",
                    "reason": str(report.get("reason") or "overlap_rejected"),
                    "overlap_vol_cm3": float(overlap_volume),
                    "inflated_overlap_vol_cm3": (
                        None
                        if inflated_volume is None
                        else float(inflated_volume)
                    ),
                    "error": (
                        "endpoint rejected by frozen RGB-D Lite overlap filter "
                        f"(original={float(overlap_volume):.6f}cm3 "
                        f"<={ORIGINAL_OVERLAP_MAX_CM3:.6f}cm3, inflated="
                        + (
                            "not queried after original rejection"
                            if inflated_volume is None
                            else f"{float(inflated_volume):.6f}cm3 "
                            f"<{INFLATED_OVERLAP_MAX_CM3:.6f}cm3"
                        )
                        + ")"
                    ),
                }
            )
    expected_passing_indices = [
        index
        for index, report in enumerate(decorated_reports)
        if bool(report.get("ok"))
    ]
    try:
        reported_pose_count = int(evaluation["pose_count"])
        reported_passing_count = int(evaluation["passing_pose_count"])
        reported_rejected_count = int(evaluation["rejected_pose_count"])
        reported_passing_indices = [
            int(index) for index in evaluation["passing_pose_indices"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "move-tracked frozen RGB-D overlap batch summary is invalid"
        ) from exc
    if (
        reported_pose_count != len(records)
        or reported_passing_count != len(expected_passing_indices)
        or reported_rejected_count != len(records) - len(expected_passing_indices)
        or reported_passing_indices != expected_passing_indices
    ):
        raise RuntimeError(
            "move-tracked frozen RGB-D overlap batch summary is inconsistent"
        )
    batch_report = {
        **{key: value for key, value in dict(evaluation).items() if key != "poses"},
        "stage": str(stage),
        "poses": decorated_reports,
    }
    return accepted, _json_ready(batch_report), rejected



def _move_tracked_filter_endpoint_overlap_frontier(
    candidate_records: Sequence[tuple[dict[str, Any], str]],
    *,
    overlap_filter_context: Mapping[str, Any],
    stage: str,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[
    list[tuple[dict[str, Any], str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Filter a frontier, spatially splitting only oversized occupancy grids.

    The occupancy lattice is aligned to the robot-base voxel grid, so a pose
    receives the same overlap count in a smaller spatial batch.  Splitting is
    therefore result-equivalent to one large batch while avoiding a batch
    envelope that exceeds the occupancy implementation's bounded allocation.
    """

    records = list(candidate_records)
    accepted: list[tuple[dict[str, Any], str]] = []
    rejected: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    stats = {
        "batch_attempt_count": 0,
        "successful_batch_count": 0,
        "adaptive_split_count": 0,
        "evaluation_unavailable_count": 0,
        "maximum_attempted_grid_voxels": None,
        "grid_voxel_limit": None,
    }

    def cancellation_gate() -> None:
        if check_cancelled is not None:
            check_cancelled()

    def grid_contract_numbers(
        exc: ProjectiveOccupancyContractError,
    ) -> tuple[int | None, int | None]:
        match = re.search(
            r"local occupancy grid has\s+(\d+)\s+voxels,\s+limit is\s+(\d+)",
            str(exc),
        )
        if match is None:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def record_failed_attempt(
        *,
        indexed_records: Sequence[tuple[int, tuple[dict[str, Any], str]]],
        split_depth: int,
        exc: ProjectiveOccupancyContractError,
        action: str,
    ) -> None:
        total_voxels, voxel_limit = grid_contract_numbers(exc)
        if total_voxels is not None:
            previous = stats["maximum_attempted_grid_voxels"]
            stats["maximum_attempted_grid_voxels"] = (
                total_voxels
                if previous is None
                else max(int(previous), total_voxels)
            )
        if voxel_limit is not None:
            stats["grid_voxel_limit"] = voxel_limit
        batch_reports.append(
            {
                "ok": False,
                "build": LITE_POSE_OVERLAP_FILTER_BUILD,
                "stage": str(stage),
                "pose_count": int(len(indexed_records)),
                "passing_pose_count": 0,
                "rejected_pose_count": (
                    1 if action == "reject_single_candidate" else 0
                ),
                "poses": [],
                "evaluation_available": False,
                "failure_scope": "occupancy_batch_envelope",
                "adaptive_action": str(action),
                "adaptive_split_depth": int(split_depth),
                "frontier_candidate_indices": [
                    int(index) for index, _record in indexed_records
                ],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "requested_grid_voxels": total_voxels,
                "grid_voxel_limit": voxel_limit,
                "original_overlap_threshold_cm3": float(
                    ORIGINAL_OVERLAP_MAX_CM3
                ),
                "original_overlap_comparison": "<=",
                "inflated_overlap_threshold_cm3": float(
                    INFLATED_OVERLAP_MAX_CM3
                ),
                "inflated_overlap_comparison": "<",
                "segmentation_used": False,
                "simulator_mesh_used": False,
                "direct_simulator_mutation": False,
            }
        )

    def split_spatially(
        indexed_records: Sequence[tuple[int, tuple[dict[str, Any], str]]],
    ) -> tuple[
        list[tuple[int, tuple[dict[str, Any], str]]],
        list[tuple[int, tuple[dict[str, Any], str]]],
    ]:
        items = list(indexed_records)
        positions = np.stack(
            [
                np.asarray(
                    record[0]["final_eef_position_m"], dtype=np.float64
                ).reshape(3)
                for _index, record in items
            ],
            axis=0,
        )
        axis = int(np.argmax(np.ptp(positions, axis=0)))
        ordered = sorted(
            items,
            key=lambda item: (
                float(
                    np.asarray(
                        item[1][0]["final_eef_position_m"],
                        dtype=np.float64,
                    ).reshape(3)[axis]
                ),
                int(item[0]),
            ),
        )
        midpoint = len(ordered) // 2
        return ordered[:midpoint], ordered[midpoint:]

    def evaluate_subset(
        indexed_records: Sequence[tuple[int, tuple[dict[str, Any], str]]],
        *,
        split_depth: int,
    ) -> None:
        cancellation_gate()
        items = list(indexed_records)
        stats["batch_attempt_count"] += 1
        try:
            local_accepted, batch_report, local_rejected = (
                _move_tracked_filter_endpoint_overlap_batch(
                    [record for _index, record in items],
                    overlap_filter_context=overlap_filter_context,
                    stage=stage,
                )
            )
        except ProjectiveOccupancyContractError as exc:
            cancellation_gate()
            total_voxels, voxel_limit = grid_contract_numbers(exc)
            if total_voxels is None or voxel_limit is None:
                raise RuntimeError(
                    "move-tracked frozen RGB-D overlap evaluation reported a "
                    f"non-grid occupancy contract failure: {exc}"
                ) from exc
            if len(items) > 1:
                stats["adaptive_split_count"] += 1
                record_failed_attempt(
                    indexed_records=items,
                    split_depth=split_depth,
                    exc=exc,
                    action="split_spatial_batch",
                )
                left, right = split_spatially(items)
                evaluate_subset(left, split_depth=split_depth + 1)
                evaluate_subset(right, split_depth=split_depth + 1)
                return

            frontier_index, (candidate, preference) = items[0]
            stats["evaluation_unavailable_count"] += 1
            record_failed_attempt(
                indexed_records=items,
                split_depth=split_depth,
                exc=exc,
                action="reject_single_candidate",
            )
            total_voxels, voxel_limit = grid_contract_numbers(exc)
            candidate_report = {
                "ok": False,
                "stage": str(stage),
                "candidate_index_in_frontier": int(frontier_index),
                "endpoint_preference": str(preference),
                "endpoint_score_rank": int(
                    (candidate.get("selected_candidate") or {}).get(
                        "endpoint_score_rank", frontier_index
                    )
                ),
                "reason": "occupancy_grid_contract_exceeded",
                "evaluation_available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "requested_grid_voxels": total_voxels,
                "grid_voxel_limit": voxel_limit,
                "original_overlap_threshold_cm3": float(
                    ORIGINAL_OVERLAP_MAX_CM3
                ),
                "inflated_overlap_threshold_cm3": float(
                    INFLATED_OVERLAP_MAX_CM3
                ),
            }
            candidate["overlap_volume_filter"] = _json_ready(candidate_report)
            rejected.append(
                {
                    "endpoint_score_rank": int(
                        candidate_report["endpoint_score_rank"]
                    ),
                    "path_candidate_rank": -1,
                    "endpoint_preference": str(preference),
                    "path_mode": "frozen_rgbd_overlap_filter",
                    "reason": "overlap_evaluation_grid_limit",
                    "overlap_vol_cm3": None,
                    "inflated_overlap_vol_cm3": None,
                    "overlap_evaluation_available": False,
                    "requested_grid_voxels": total_voxels,
                    "grid_voxel_limit": voxel_limit,
                    "error": (
                        "endpoint rejected because its frozen RGB-D overlap "
                        "could not be evaluated within the bounded occupancy "
                        f"grid: {exc}"
                    ),
                }
            )
            return

        stats["successful_batch_count"] += 1
        frontier_indices = [int(index) for index, _record in items]
        batch_report = {
            **dict(batch_report),
            "evaluation_available": True,
            "adaptive_split_depth": int(split_depth),
            "frontier_candidate_indices": frontier_indices,
        }
        for local_index, (_frontier_index, (candidate, _preference)) in enumerate(
            items
        ):
            report = candidate.get("overlap_volume_filter")
            if isinstance(report, Mapping):
                candidate["overlap_volume_filter"] = {
                    **dict(report),
                    "candidate_index_in_frontier": frontier_indices[local_index],
                    "adaptive_split_depth": int(split_depth),
                }
            pose_reports = batch_report.get("poses")
            if isinstance(pose_reports, list) and local_index < len(pose_reports):
                pose_reports[local_index] = {
                    **dict(pose_reports[local_index]),
                    "candidate_index_in_frontier": frontier_indices[local_index],
                    "adaptive_split_depth": int(split_depth),
                }
        batch_reports.append(_json_ready(batch_report))
        accepted.extend(local_accepted)
        rejected.extend(local_rejected)
        cancellation_gate()

    if records:
        evaluate_subset(list(enumerate(records)), split_depth=0)
    stats.update(
        {
            "candidate_count": int(len(records)),
            "passing_candidate_count": int(len(accepted)),
            "rejected_candidate_count": int(len(rejected)),
        }
    )
    return accepted, batch_reports, rejected, _json_ready(stats)



def _move_tracked_best_accuracy_records(
    records: Sequence[tuple[dict[str, Any], str]],
    *,
    reference_records: Sequence[tuple[dict[str, Any], str]] | None = None,
) -> list[tuple[dict[str, Any], str]]:
    """Keep accuracy authoritative even when another pose passes collision gates."""

    def accuracy(candidate: Mapping[str, Any]) -> tuple[bool, float]:
        report = candidate.get("selected_candidate") or {}
        precise = bool(report.get("precise", candidate.get("selected_is_precise", False)))
        try:
            cost = float(report.get("accuracy_cost_mm_plus_deg", math.inf))
        except (TypeError, ValueError):
            cost = math.inf
        return precise, cost if math.isfinite(cost) and cost >= 0.0 else math.inf

    reference = list(records if reference_records is None else reference_records)
    has_precise = any(accuracy(candidate)[0] for candidate, _ in reference)
    best_cost = min((accuracy(candidate)[1] for candidate, _ in reference), default=math.inf)
    return [
        (candidate, preference) for candidate, preference in records
        if (accuracy(candidate)[0] if has_precise else (
            math.isfinite(best_cost)
            and math.isclose(accuracy(candidate)[1], best_cost, rel_tol=0.0, abs_tol=1.0e-9)
        ))
    ]



def _move_tracked_overlap_fallback_records(
    records: Sequence[tuple[dict[str, Any], str]],
) -> list[tuple[dict[str, Any], str]]:
    """Rank evaluated IK endpoints without relaxing their accuracy class."""

    eligible = []
    for candidate, preference in records:
        report = candidate.get("overlap_volume_filter")
        accuracy = candidate.get("selected_candidate")
        if (
            "q_final" not in candidate
            or not isinstance(report, Mapping)
            or not isinstance(accuracy, Mapping)
            or bool(report.get("ok"))
            or report.get("evaluation_available") is False
        ):
            continue
        try:
            original = float(report["overlap_vol_cm3"])
            raw_inflated = report.get("inflated_overlap_vol_cm3")
            inflated = None if raw_inflated is None else float(raw_inflated)
            cost = float(accuracy["accuracy_cost_mm_plus_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(original)
            or original < 0.0
            or not math.isfinite(cost)
            or cost < 0.0
            or (inflated is not None and (not math.isfinite(inflated) or inflated < 0.0))
            or (inflated is None and not bool(report.get("inflated_query_skipped")))
        ):
            continue
        precise = bool(
            accuracy.get("precise", candidate.get("selected_is_precise", False))
        )
        eligible.append((candidate, preference, precise, cost, original, inflated))
    if not eligible:
        return []
    has_precise = any(item[2] for item in eligible)
    minimum_cost = min(item[3] for item in eligible)
    frontier = [
        item for item in eligible
        if (item[2] if has_precise else math.isclose(
            item[3], minimum_cost, rel_tol=0.0, abs_tol=1.0e-9
        ))
    ]
    # Preserve accuracy first, then choose the least-overlapping and
    # least-moving fixed-trunk arm endpoint within that accuracy frontier.
    frontier.sort(key=lambda item: (
        item[4],
        math.inf if item[5] is None else item[5],
        item[3],
        float(item[0]["selected_candidate"].get("motion_score", math.inf)),
    ))
    result = []
    for candidate, preference, _precise, cost, _original, _inflated in frontier:
        result.append((
            {
                **candidate,
                "overlap_volume_filter": {
                    **candidate["overlap_volume_filter"],
                    "fallback_selection": {
                        "policy": MOVE_TRACKED_OVERLAP_FALLBACK_POLICY,
                        "accuracy_class": "precise" if has_precise else "minimum_error",
                        "accuracy_cost_mm_plus_deg": cost,
                        "accuracy_frontier_count": len(frontier),
                        "overlap_order": "original_then_inflated_cm3",
                    },
                },
            },
            preference,
        ))
    return result



def _move_tracked_plan_frozen(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: np.ndarray,
    source_points: np.ndarray,
    target_points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    off_hand_points: np.ndarray | None = None,
    off_hand_target_points: Sequence[Mapping[str, Any]] | None = None,
    on_hand_names: Sequence[str] | None = None,
    off_hand_names: Sequence[str] | None = None,
    head_camera_position_robot_base_m: Sequence[float] | None = None,
    head_camera_quaternion_xyzw: Sequence[float] | None = None,
    pos_tol_m: float,
    ori_tol_deg: float,
    max_steps: int,
    overlap_filter_context: Mapping[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
    planning_deadline_monotonic: float | None = None,
    hard_deadline_monotonic: float | None = None,
    isolate_process: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plan from frozen policy-owned data without touching the live world."""

    if isolate_process:
        worker_kwargs = {
            key: value for key, value in locals().items()
            if key not in {"isolate_process", "cancel_event", "progress_callback", "hard_deadline_monotonic"}
        }
        return _run_tracked_plan_worker(
            worker_kwargs, cancel_event=cancel_event,
            hard_deadline_monotonic=hard_deadline_monotonic,
            progress_callback=progress_callback,
        )
    if hard_deadline_monotonic is not None and not math.isfinite(float(hard_deadline_monotonic)):
        raise ValueError("frozen planning hard deadline must be finite")
    phase_started = time.monotonic()
    phase_history: list[dict[str, Any]] = []

    def planning_stage(name: str, **details: Any) -> None:
        now = time.monotonic()
        if phase_history:
            phase_history[-1]["elapsed_s"] = now - phase_history[-1]["started_monotonic"]
        phase_history.append({"stage": name, "started_monotonic": now, "elapsed_s": 0.0})
        if progress_callback is not None:
            progress_callback({
                "stage": name, "elapsed_s": now - phase_started,
                "stages": [dict(item) for item in phase_history], **_json_ready(details),
            })

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("frozen tracked-point planning was cancelled")
        if hard_deadline_monotonic is not None and time.monotonic() >= hard_deadline_monotonic:
            raise PlanningDeadlineExceeded("frozen planning hard deadline exhausted")

    check_cancelled()
    planning_stage("input_validation")
    if overlap_filter_context is not None:
        overlap_filter_context = {
            **overlap_filter_context,
            "_check_cancelled": check_cancelled,
            "_progress_callback": lambda name: planning_stage("overlap." + name),
        }
    planning_q_start, start_limit_recovery = _normalize_arm_q_to_joint_limits(
        q_start,
        arm=arm,
        label=f"move-tracked observed {arm} planning start qpos",
        point="observed_start",
        recover_observed_overshoot=True,
    )
    max_start_limit_recovery = max(
        (
            abs(float(item["correction_rad"]))
            for item in start_limit_recovery
        ),
        default=0.0,
    )
    if max_start_limit_recovery > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD:
        raise ValueError(
            "move-tracked observed planning start exceeds the recoverable "
            f"joint-limit residual ({max_start_limit_recovery:.12g}rad > "
            f"{MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD:.12g}rad)"
        )
    # A quick preset is lowered to the same local relation solver used by the
    # coordinate mode.  It is appended to, rather than substituted for, the
    # caller's XYZ and typed-relation constraints.
    planning_target_points = list(target_points)
    planning_relations = list(relations or [])
    quick_metadata: dict[str, Any] = {}
    quick_specs = _move_tracked_quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    faceto_camera_pose = _move_tracked_camera_pose_payload(
        head_camera_position_robot_base_m,
        head_camera_quaternion_xyzw,
    )
    if faceto_requested and faceto_camera_pose is None:
        raise ValueError(
            "faceto/reverse_faceto planning requires the frozen head-camera pose"
        )
    if quick_specs:
        controlled_names = (
            [str(name).strip() for name in on_hand_names]
            if on_hand_names is not None
            else [str(item.get("name") or "").strip() for item in planning_target_points]
        )
        if off_hand_names is not None:
            reference_names = [str(name).strip() for name in off_hand_names]
        else:
            # Preserve the legacy private-call shape where the frozen reference
            # matrix was supplied beside one quick constraint but its names were
            # implicit. For plural groups the deterministic first-use order is
            # the only unambiguous extension, and shared names remain one column.
            reference_names = []
            for quick_spec in quick_specs:
                for raw_name in quick_spec.get("off_hand_points", []):
                    name = str(raw_name).strip()
                    if name and name not in reference_names:
                        reference_names.append(name)
        if not controlled_names:
            raise ValueError("quick constraints require controlled point names")
        if [str(item.get("name")) for item in planning_target_points] != controlled_names:
            raise ValueError("quick constraint target order does not match on-hand points")
        controlled_source = np.asarray(source_points, dtype=np.float64)
        reference_source = (
            None
            if off_hand_points is None
            else np.asarray(off_hand_points, dtype=np.float64)
        )
        quick_relations: list[dict[str, Any]] = []
        for quick_spec in quick_specs:
            group_on_names, group_on_points, _group_off_names, group_off_points = (
                _move_tracked_quick_group_points(
                    quick_spec,
                    controlled_names=controlled_names,
                    controlled_points=controlled_source,
                    reference_names=reference_names,
                    reference_points=reference_source,
                )
            )
            quick_relation = _quick_constraint_relation(
                quick_spec,
                on_hand_names=group_on_names,
                on_hand_points_robot_base_m=group_on_points,
                off_hand_points_robot_base_m=group_off_points,
                camera_position_robot_base_m=head_camera_position_robot_base_m,
                camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            )
            planning_relations.append(quick_relation)
            quick_relations.append(quick_relation)
        quick_metadata = {
            "quick_constraints": _json_ready(quick_specs),
            "off_hand_source_points_robot_base_m": (
                [] if reference_source is None else reference_source.astype(float).tolist()
            ),
            "on_hand_point_names": list(controlled_names),
            "off_hand_point_names": list(reference_names),
            "quick_planning_relations": _json_ready(quick_relations),
        }
        if quick_constraint is not None:
            quick_metadata.update(
                {
                    "quick_constraint": _json_ready(dict(quick_constraint)),
                    "quick_planning_relation": _json_ready(quick_relations[0]),
                }
            )
    if faceto_camera_pose is not None:
        quick_metadata["faceto_camera_pose"] = faceto_camera_pose
    planning_metadata = {
        "planning_target_points": _json_ready(planning_target_points),
        "planning_relations": _json_ready(planning_relations),
        "planning_inequalities": _json_ready(list(inequalities or [])),
        "off_hand_target_points": _json_ready(list(off_hand_target_points or [])),
        **quick_metadata,
    }
    endpoint_kwargs = {
        "state": state,
        "arm": arm,
        "q_start": planning_q_start,
        "source_points_robot_base_m": source_points,
        "target_points": planning_target_points,
        "relations": planning_relations,
        "inequalities": list(inequalities or []),
        "fixed_points_robot_base_m": off_hand_points,
        "fixed_target_points": (
            list(off_hand_target_points or [])
            if off_hand_points is not None
            else None
        ),
        "pos_tol_m": pos_tol_m,
        "ori_tol_deg": ori_tol_deg,
        "cancel_requested": (
            cancel_event.is_set if cancel_event is not None else None
        ),
        "planning_deadline_monotonic": planning_deadline_monotonic,
    }
    planning_stage("endpoint_ik")
    endpoint_solution = _plan_tracked_endpoint(**endpoint_kwargs)
    check_cancelled()
    planning_budget_exhausted = bool(
        endpoint_solution.get("planning_budget_exhausted", False)
    )
    endpoint_candidates = list(
        endpoint_solution.pop(
            "_ranked_endpoint_candidates",
            [endpoint_solution],
        )
    )
    planning_stage("endpoint_candidates_ready", candidate_count=len(endpoint_candidates),
                   best_constraints=endpoint_solution.get("constraints"),
                   best_is_precise=endpoint_solution.get("selected_is_precise"))
    path_failures: list[dict[str, Any]] = []
    endpoint_solver_failures: list[dict[str, Any]] = []
    overlap_filter_batches: list[dict[str, Any]] = []
    overlap_filter_candidate_count = 0
    overlap_filter_passing_count = 0
    overlap_filter_endpoint_passing_count = 0
    overlap_filter_rejected_count = 0
    overlap_filter_evaluation_unavailable_count = 0
    overlap_filter_batch_attempt_count = 0
    overlap_filter_successful_batch_count = 0
    overlap_filter_adaptive_split_count = 0
    overlap_filter_maximum_attempted_grid_voxels: int | None = None
    overlap_filter_grid_voxel_limit: int | None = None
    overlap_rejected_endpoint_centers: list[dict[str, Any]] = []
    overlap_rejected_endpoint_records: list[tuple[dict[str, Any], str]] = []
    endpoint_accuracy_pool: list[tuple[dict[str, Any], str]] = []

    def filter_endpoint_records(
        records: Sequence[tuple[dict[str, Any], str]],
        *,
        stage: str,
        record_individual_rejections: bool = True,
    ) -> list[tuple[dict[str, Any], str]]:
        nonlocal overlap_filter_candidate_count
        nonlocal overlap_filter_passing_count
        nonlocal overlap_filter_endpoint_passing_count
        nonlocal overlap_filter_rejected_count
        nonlocal overlap_filter_evaluation_unavailable_count
        nonlocal overlap_filter_batch_attempt_count
        nonlocal overlap_filter_successful_batch_count
        nonlocal overlap_filter_adaptive_split_count
        nonlocal overlap_filter_maximum_attempted_grid_voxels
        nonlocal overlap_filter_grid_voxel_limit
        local_records = list(records)
        if overlap_filter_context is None:
            return local_records
        endpoint_accuracy_pool.extend(
            record for record in local_records if "q_final" in record[0]
        )
        check_cancelled()
        planning_stage("overlap_frontier", frontier=stage, candidate_count=len(local_records))
        accepted, batch_reports, rejected, adaptive_report = (
            _move_tracked_filter_endpoint_overlap_frontier(
                local_records,
                overlap_filter_context=overlap_filter_context,
                stage=stage,
                check_cancelled=check_cancelled,
            )
        )
        check_cancelled()
        overlap_filter_batches.extend(batch_reports)
        overlap_filter_candidate_count += len(local_records)
        overlap_filter_passing_count += len(accepted)
        overlap_filter_endpoint_passing_count += sum(
            "q_final" in candidate for candidate, _preference in accepted
        )
        overlap_filter_rejected_count += len(rejected)
        overlap_filter_evaluation_unavailable_count += int(
            adaptive_report["evaluation_unavailable_count"]
        )
        overlap_filter_batch_attempt_count += int(
            adaptive_report["batch_attempt_count"]
        )
        overlap_filter_successful_batch_count += int(
            adaptive_report["successful_batch_count"]
        )
        overlap_filter_adaptive_split_count += int(
            adaptive_report["adaptive_split_count"]
        )
        local_max_grid = adaptive_report.get(
            "maximum_attempted_grid_voxels"
        )
        if local_max_grid is not None:
            overlap_filter_maximum_attempted_grid_voxels = max(
                int(local_max_grid),
                int(overlap_filter_maximum_attempted_grid_voxels or 0),
            )
        local_grid_limit = adaptive_report.get("grid_voxel_limit")
        if local_grid_limit is not None:
            overlap_filter_grid_voxel_limit = int(local_grid_limit)
        if not str(stage).startswith("overlap_aware_pose_family_prefilter"):
            for candidate, preference in local_records:
                overlap_report = candidate.get("overlap_volume_filter")
                if (
                    "q_final" in candidate
                    and isinstance(overlap_report, Mapping)
                    and not bool(overlap_report.get("ok"))
                ):
                    overlap_rejected_endpoint_centers.append(candidate)
                    overlap_rejected_endpoint_records.append((candidate, preference))
        if record_individual_rejections:
            path_failures.extend(rejected)
        planning_stage("overlap_frontier_complete", frontier=stage,
                       passing_count=len(accepted), rejected_count=len(rejected),
                       overlap_batches=batch_reports)
        if not str(stage).startswith("overlap_aware_pose_family_prefilter"):
            return _move_tracked_best_accuracy_records(
                accepted, reference_records=endpoint_accuracy_pool
            )
        return accepted

    def overlap_filter_report(
        selected_candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if overlap_filter_context is None:
            return None
        selected = (
            selected_candidate.get("overlap_volume_filter")
            if isinstance(selected_candidate, Mapping)
            else None
        )
        fallback_selection = (
            selected.get("fallback_selection") if isinstance(selected, Mapping) else None
        )
        return _json_ready(
            {
                "ok": bool(isinstance(selected, Mapping) and selected.get("ok")),
                "enabled": True,
                "required_for_execution_mode": "plan",
                "build": LITE_POSE_OVERLAP_FILTER_BUILD,
                "source_planner": "plan_grasp_point_filter_rgbd_lite",
                "candidate_count_evaluated": int(overlap_filter_candidate_count),
                "candidate_count_passing": int(overlap_filter_passing_count),
                "endpoint_candidate_count_passing": int(overlap_filter_endpoint_passing_count),
                "candidate_count_rejected": int(overlap_filter_rejected_count),
                "candidate_count_evaluation_unavailable": int(
                    overlap_filter_evaluation_unavailable_count
                ),
                "batch_count": int(len(overlap_filter_batches)),
                "batch_attempt_count": int(overlap_filter_batch_attempt_count),
                "successful_batch_count": int(
                    overlap_filter_successful_batch_count
                ),
                "adaptive_split_count": int(
                    overlap_filter_adaptive_split_count
                ),
                "maximum_attempted_grid_voxels": (
                    overlap_filter_maximum_attempted_grid_voxels
                ),
                "grid_voxel_limit": overlap_filter_grid_voxel_limit,
                "original_overlap_threshold_cm3": float(
                    ORIGINAL_OVERLAP_MAX_CM3
                ),
                "original_overlap_comparison": "<=",
                "inflated_overlap_threshold_cm3": float(
                    INFLATED_OVERLAP_MAX_CM3
                ),
                "inflated_overlap_comparison": "<",
                "selected_pose": selected,
                "fallback_applied": bool(fallback_selection),
                "fallback_selection": fallback_selection,
                "warning": (
                    "overlap_thresholds_exceeded_minimum_overlap_fallback"
                    if fallback_selection else None
                ),
                "batches": overlap_filter_batches,
                "grasp_opening_volume_queried": False,
                "filter_order": (
                    "original_overlap_all_then_inflated_overlap_original_passers"
                ),
                "filter_result_equivalent_to_full_conjunction": True,
                "final_selection_order_preserved": not bool(fallback_selection),
                "segmentation_used": False,
                "simulator_mesh_used": False,
                "direct_simulator_mutation": False,
            }
        )

    def endpoint_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
        report = candidate.get("selected_candidate")
        report = report if isinstance(report, Mapping) else {}
        precise = bool(
            report.get("precise", candidate.get("selected_is_precise", False))
        )
        accuracy = float(report.get("accuracy_cost_mm_plus_deg", math.inf))
        motion = float(report.get("motion_score", math.inf))
        position_mm = float(report.get("position_error_mm", math.inf))
        orientation_deg = float(
            report.get("selection_orientation_error_deg", math.inf)
        )
        if precise:
            return (0.0, motion, accuracy, position_mm, orientation_deg)
        return (1.0, accuracy, position_mm, orientation_deg, motion)

    overlap_pose_family_seen_rotations: list[np.ndarray] = []

    def generate_overlap_aware_pose_family(
        *,
        search_stage: str,
        base_count: int,
        angular_offsets_deg: Sequence[float],
        max_ik_targets: int,
    ) -> list[tuple[dict[str, Any], str]]:
        """Search a real constraint null space after ordinary poses collide."""

        if overlap_filter_context is None or not overlap_rejected_endpoint_centers:
            return []
        if sum(
            str(relation.get("type") or "").strip().lower()
            == "line_through_point"
            for relation in planning_relations
            if isinstance(relation, Mapping)
        ) < 2:
            return []

        def center_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
            report = candidate.get("overlap_volume_filter")
            report = report if isinstance(report, Mapping) else {}
            original_ok = bool(report.get("original_overlap_ok"))
            try:
                original = float(report.get("overlap_vol_cm3", math.inf))
            except (TypeError, ValueError):
                original = math.inf
            try:
                inflated = float(
                    report.get("inflated_overlap_vol_cm3", math.inf)
                )
            except (TypeError, ValueError):
                inflated = math.inf
            collision_cost = inflated if original_ok else original
            return (
                0.0 if original_ok else 1.0,
                collision_cost,
                *endpoint_rank(candidate),
            )

        centers: list[dict[str, Any]] = []
        center_rotations: list[np.ndarray] = []
        for candidate in sorted(
            overlap_rejected_endpoint_centers,
            key=center_rank,
        ):
            try:
                quaternion = np.asarray(
                    candidate["final_eef_quaternion_xyzw"], dtype=np.float64
                ).reshape(4)
                rotation = _quat_to_mat_xyzw(quaternion)
            except (KeyError, TypeError, ValueError):
                continue
            if any(
                float(np.linalg.norm(rotation - old, ord="fro")) <= 1.0e-4
                for old in center_rotations
            ):
                continue
            centers.append(candidate)
            center_rotations.append(rotation)
            if len(centers) >= int(base_count):
                break
        if not centers:
            return []
        start_eef_position, start_eef_quaternion = _local_eef_pose(
            state,
            arm,
            planning_q_start,
        )
        family = _shared_line_target_pose_family(
            source_points_robot_base_m=source_points,
            point_names=[str(item["name"]) for item in planning_target_points],
            relations=[relation for relation in planning_relations
                       if relation.get("type") != "ordered_collinear"],
            start_eef_position_m=start_eef_position,
            start_eef_quaternion_xyzw=start_eef_quaternion,
            base_target_eef_quaternions_xyzw=[
                candidate["final_eef_quaternion_xyzw"] for candidate in centers
            ],
            angular_offsets_deg=angular_offsets_deg,
        )
        family_summary = {
            key: value for key, value in family.items() if key != "poses"
        }
        family_metadata = planning_metadata.setdefault(
            "overlap_aware_pose_family",
            {
                "search_order": "coarse_then_fine_then_legacy_fallbacks",
                "search_stages": [],
                "pose_count": 0,
                "unique_pose_count": 0,
                "overlap_prefilter_passing_count": 0,
                "ik_candidate_count": 0,
                "post_ik_overlap_passing_count": 0,
            },
        )
        stage_metadata = {
            **family_summary,
            "search_stage": str(search_stage),
            "base_count_limit": int(base_count),
            "ik_target_limit": int(max_ik_targets),
        }
        family_metadata["search_stages"].append(stage_metadata)
        family_metadata["pose_count"] = int(
            family_metadata.get("pose_count", 0)
        ) + int(family.get("pose_count", 0))
        if not bool(family.get("ok")):
            endpoint_solver_failures.append(
                {
                    "endpoint_preference": (
                        f"overlap_aware_pose_family_{search_stage}"
                    ),
                    "reason": str(
                        family.get("reason") or "pose_family_unavailable"
                    ),
                    "pose_family": _json_ready(family_summary),
                }
            )
            return []

        pose_records: list[tuple[dict[str, Any], str]] = []
        duplicate_pose_count = 0
        for pose_index, raw_pose in enumerate(family.get("poses", [])):
            pose = dict(raw_pose)
            pose_rotation = _quat_to_mat_xyzw(
                np.asarray(
                    pose["final_eef_quaternion_xyzw"], dtype=np.float64
                ).reshape(4)
            )
            if any(
                float(np.linalg.norm(pose_rotation - old, ord="fro"))
                <= 1.0e-9
                for old in overlap_pose_family_seen_rotations
            ):
                duplicate_pose_count += 1
                continue
            overlap_pose_family_seen_rotations.append(pose_rotation)
            pose["selected_candidate"] = {
                "endpoint_score_rank": int(pose_index),
                "candidate_origin": "overlap_aware_pose_family_prefilter",
                "pose_family_search_stage": str(search_stage),
            }
            pose_records.append(
                (pose, "overlap_aware_shared_line_pose_family_prefilter")
            )
        stage_metadata["duplicate_pose_count"] = int(duplicate_pose_count)
        stage_metadata["unique_pose_count"] = int(len(pose_records))
        family_metadata["unique_pose_count"] = int(
            family_metadata.get("unique_pose_count", 0)
        ) + len(pose_records)
        prefiltered = filter_endpoint_records(
            pose_records,
            stage=f"overlap_aware_pose_family_prefilter_{search_stage}",
            record_individual_rejections=False,
        )

        start_position = np.asarray(start_eef_position, dtype=np.float64)
        start_quaternion = np.asarray(start_eef_quaternion, dtype=np.float64)

        def prefiltered_pose_rank(
            record: tuple[dict[str, Any], str],
        ) -> tuple[float, ...]:
            pose = record[0]
            target_position = np.asarray(
                pose["final_eef_position_m"], dtype=np.float64
            )
            target_quaternion = np.asarray(
                pose["final_eef_quaternion_xyzw"], dtype=np.float64
            )
            target_quaternion /= max(float(np.linalg.norm(target_quaternion)), 1.0e-12)
            normalized_start = start_quaternion / max(
                float(np.linalg.norm(start_quaternion)), 1.0e-12
            )
            orientation_delta = 2.0 * math.acos(
                float(
                    np.clip(
                        abs(float(target_quaternion @ normalized_start)),
                        -1.0,
                        1.0,
                    )
                )
            )
            overlap_report = pose.get("overlap_volume_filter")
            overlap_report = (
                overlap_report if isinstance(overlap_report, Mapping) else {}
            )
            return (
                float(np.linalg.norm(target_position - start_position))
                + 0.05 * orientation_delta,
                float(
                    overlap_report.get("inflated_overlap_vol_cm3", math.inf)
                ),
                int(pose.get("base_pose_index", 0)),
                *[
                    abs(float(value))
                    for value in pose.get("local_euler_xyz_deg", [0.0, 0.0, 0.0])
                ],
            )

        prefiltered = sorted(prefiltered, key=prefiltered_pose_rank)
        total_prefilter_passing_count = len(prefiltered)
        prefiltered = prefiltered[: int(max_ik_targets)]
        stage_metadata.update(
            {
                "overlap_prefilter_passing_count": int(
                    total_prefilter_passing_count
                ),
                "ik_target_count": int(len(prefiltered)),
                "overlap_prefilter_applied_before_ik": True,
            }
        )
        family_metadata["overlap_prefilter_passing_count"] = int(
            family_metadata.get("overlap_prefilter_passing_count", 0)
        ) + int(total_prefilter_passing_count)
        family_metadata["overlap_prefilter_applied_before_ik"] = True
        if not prefiltered:
            path_failures.append(
                {
                    "endpoint_score_rank": -1,
                    "path_candidate_rank": -1,
                    "endpoint_preference": "overlap_aware_pose_family",
                    "path_mode": "frozen_rgbd_overlap_filter",
                    "reason": "all_pose_family_targets_overlap",
                    "search_stage": str(search_stage),
                    "error": (
                        "all analytically equivalent shared-line poses failed "
                        "the frozen RGB-D Lite overlap filter"
                    ),
                }
            )
            return []
        pose_targets = []
        for pose, _preference in prefiltered:
            pose_targets.append(
                {
                    **pose,
                    "pose_overlap_prefilter": pose.get(
                        "overlap_volume_filter"
                    ),
                }
            )
        try:
            solved = _plan_tracked_endpoint(
                **endpoint_kwargs,
                eef_pose_targets=pose_targets,
                pose_targets_only=True,
                max_endpoint_candidates=(
                    int(max_ik_targets)
                ),
            )
            solved_candidates = list(
                solved.pop("_ranked_endpoint_candidates", [solved])
            )
        except PlanningDeadlineExceeded as exc:
            endpoint_solver_failures.append(
                {
                    "endpoint_preference": (
                        f"overlap_aware_pose_family_{search_stage}"
                    ),
                    "reason": "planning_budget_exhausted",
                    "error": str(exc),
                }
            )
            return []
        except (RuntimeError, TypeError, ValueError) as exc:
            endpoint_solver_failures.append(
                {
                    "endpoint_preference": (
                        f"overlap_aware_pose_family_{search_stage}"
                    ),
                    "reason": "pose_family_ik_unavailable",
                    "error": str(exc),
                }
            )
            return []
        stage_metadata.update(
            {
                "ik_candidate_count": int(len(solved_candidates)),
                "post_ik_overlap_recheck": True,
            }
        )
        family_metadata["ik_candidate_count"] = int(
            family_metadata.get("ik_candidate_count", 0)
        ) + len(solved_candidates)
        family_metadata["post_ik_overlap_recheck"] = True
        accepted = filter_endpoint_records(
            [
                (candidate, "overlap_aware_shared_line_pose_family")
                for candidate in solved_candidates
            ],
            stage=f"overlap_aware_pose_family_post_ik_{search_stage}",
        )
        stage_metadata["post_ik_overlap_passing_count"] = int(len(accepted))
        family_metadata["post_ik_overlap_passing_count"] = int(
            family_metadata.get("post_ik_overlap_passing_count", 0)
        ) + len(accepted)
        return sorted(accepted, key=lambda item: endpoint_rank(item[0]))

    candidate_records: list[tuple[dict[str, Any], str]] = [
        (candidate, "nearest_tracked_point_transform")
        for candidate in endpoint_candidates
    ]
    shared_line_pose_family_search_enabled = bool(
        overlap_filter_context is not None
        and sum(
            str(relation.get("type") or "").strip().lower()
            == "line_through_point"
            for relation in planning_relations
            if isinstance(relation, Mapping)
        )
        >= 2
    )
    # move_tracked_point plans are arm-only. The trunk is part of the frozen
    # robot frame and must never become an extra endpoint degree of freedom.
    # This keeps their exec_plan_pose behavior identical to grasp plans:
    # current arm -> safe arm pose -> final arm pose, with the observed trunk
    # pinned throughout.
    candidate_records = filter_endpoint_records(
        candidate_records,
        stage="primary_endpoint_frontier",
    )
    candidate_queue = sorted(
        candidate_records,
        key=lambda item: endpoint_rank(item[0]),
    )
    tested_endpoint_q: list[np.ndarray] = []
    tested_endpoint_candidates: list[tuple[dict[str, Any], str, int]] = []
    attempted_joint_fallback_q: list[np.ndarray] = []

    def joint_fallback_result(
        candidate: dict[str, Any],
        endpoint_preference: str,
        path_candidate_rank: int,
    ) -> dict[str, Any]:
        """Build the bounded joint route for one already-ranked endpoint."""

        check_cancelled()
        planning_stage("joint_path_fallback")
        if bool(candidate.get("trunk_assisted", False)):
            raise ValueError(
                "arm-only joint fallback cannot discard a trunk-assisted endpoint"
            )
        candidate_q = np.asarray(candidate["q_final"], dtype=np.float64)
        attempted_joint_fallback_q.append(candidate_q.copy())
        translation_m = float(
            np.linalg.norm(
                candidate["final_eef_position_m"]
                - candidate["start_eef_position_m"]
            )
        )
        if translation_m > MOVE_POINT_TO_POINT_MAX_TRANSLATION_M:
            raise ValueError(
                f"planned EEF translation {translation_m:.3f}m exceeds "
                f"{MOVE_POINT_TO_POINT_MAX_TRANSLATION_M:.3f}m"
            )
        fallback_path = _plan_tracked_joint_space_trajectory(
            state=state,
            arm=arm,
            q_start=candidate["q_start"],
            q_final=candidate["q_final"],
            max_joint_step_rad=MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
            max_waypoints=max(1, MAX_TRAJECTORY_WAYPOINTS),
            tracked_anchors_eef=candidate.get("anchors_eef_m"),
            cancel_requested=(
                cancel_event.is_set if cancel_event is not None else None
            ),
        )
        fallback_path = {
            **fallback_path,
            "planning_budget_exhausted": bool(planning_budget_exhausted),
            "observed_start_joint_limit_recovery": _json_ready(
                start_limit_recovery
            ),
            "observed_start_max_joint_limit_recovery_rad": float(
                max_start_limit_recovery
            ),
            "planning_start_qpos": planning_q_start.astype(float).tolist(),
        }
        selected_overlap_report = overlap_filter_report(candidate)
        if selected_overlap_report is not None:
            fallback_path["frozen_rgbd_overlap_filter"] = selected_overlap_report
        candidate["path_candidate_rank"] = int(path_candidate_rank)
        candidate["path_endpoint_preference"] = str(endpoint_preference)
        candidate["path_candidates_tested"] = len(tested_endpoint_q)
        candidate["path_candidate_failures"] = list(path_failures)
        candidate["endpoint_solver_failures"] = list(endpoint_solver_failures)
        selected_candidate = {**candidate, **planning_metadata}
        planning_stage("complete", selected_constraints=candidate.get("constraints"))
        return {
            "ok": True,
            "endpoint": selected_candidate,
            "path_plan": fallback_path,
            "path_failures": path_failures,
            **planning_metadata,
        }
    # Keep a second endpoint frontier for every multi-point request.  The
    # fallback is no longer a special case for a two-point line: with three or
    # four markers it can select a different proper-rotation/IK branch while
    # preserving the same complete constraint set.
    # A second endpoint frontier is useful only when the captured bundle has
    # an orientation-bearing geometry rank.  Rank 0 is an effective single
    # anchor and already has its EEF orientation explicitly regularized; using
    # the axial-rotation fallback there would introduce an unnecessary pose
    # family.  This keeps the fallback driven by geometry, not by a hardcoded
    # one/two-point branch.
    fallback_generated = _tracked_geometry_rank(source_points) == 0
    overlap_pose_family_stages = (
        [
            {
                "search_stage": "coarse",
                "base_count": (
                    MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_BASE_COUNT
                ),
                "angular_offsets_deg": (
                    MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_ANGLE_OFFSETS_DEG
                ),
                "max_ik_targets": (
                    MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_COARSE_MAX_IK_TARGETS
                ),
            },
            {
                "search_stage": "fine",
                "base_count": MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_BASE_COUNT,
                "angular_offsets_deg": (
                    MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_ANGLE_OFFSETS_DEG
                ),
                "max_ik_targets": (
                    MOVE_TRACKED_POINT_OVERLAP_POSE_FAMILY_MAX_IK_TARGETS
                ),
            },
        ]
        if shared_line_pose_family_search_enabled
        else []
    )
    overlap_pose_family_stage_index = 0
    overlap_fallback_generated = overlap_filter_context is None
    while (
        candidate_queue
        or not fallback_generated
        or overlap_pose_family_stage_index < len(overlap_pose_family_stages)
        or not overlap_fallback_generated
    ):
        check_cancelled()
        if not candidate_queue:
            if overlap_pose_family_stage_index < len(
                overlap_pose_family_stages
            ):
                pose_family_stage = overlap_pose_family_stages[
                    overlap_pose_family_stage_index
                ]
                overlap_pose_family_stage_index += 1
                candidate_queue.extend(
                    generate_overlap_aware_pose_family(**pose_family_stage)
                )
                continue
            if fallback_generated:
                if not overlap_fallback_generated:
                    overlap_fallback_generated = True
                    # A less accurate passing endpoint must not suppress the
                    # minimum-overlap fallback in the best accuracy class.
                    candidate_queue.extend(_move_tracked_overlap_fallback_records(
                        _move_tracked_best_accuracy_records(
                            overlap_rejected_endpoint_records,
                            reference_records=endpoint_accuracy_pool,
                        )
                    ))
                    if candidate_queue:
                        continue
                break
            # The nearest endpoint transform can sit on an IK branch whose
            # straight Cartesian continuation is blocked by a local
            # singularity.  Generate a second, geometry-independent frontier
            # and subject it to the exact same path checks.
            fallback_generated = True
            try:
                fallback_solution = _plan_tracked_endpoint(
                    **endpoint_kwargs,
                    eef_orientation_regularization_weight=(
                        ENDPOINT_PATH_FALLBACK_ORIENTATION_REGULARIZATION_WEIGHT
                    ),
                )
            except PlanningDeadlineExceeded as fallback_error:
                planning_budget_exhausted = True
                path_failures.append(
                    {
                        "endpoint_score_rank": len(tested_endpoint_q),
                        "path_candidate_rank": len(tested_endpoint_q),
                        "endpoint_preference": (
                            "minimum_orientation_path_fallback"
                        ),
                        "path_mode": "planning_budget_exhausted",
                        "error": str(fallback_error),
                    }
                )
                continue
            except (RuntimeError, TypeError, ValueError) as fallback_error:
                path_failures.append(
                    {
                        "endpoint_score_rank": len(tested_endpoint_q),
                        "path_candidate_rank": len(tested_endpoint_q),
                        "endpoint_preference": (
                            "minimum_orientation_path_fallback"
                        ),
                        "error": f"fallback endpoint solve failed: {fallback_error}",
                    }
                )
                continue
            fallback_candidates = list(
                fallback_solution.pop(
                    "_ranked_endpoint_candidates",
                    [fallback_solution],
                )
            )
            fallback_records = filter_endpoint_records(
                [
                    (candidate, "minimum_orientation_path_fallback")
                    for candidate in fallback_candidates
                ],
                stage="minimum_orientation_path_fallback_frontier",
            )
            candidate_queue.extend(fallback_records)
            continue

        candidate, endpoint_preference = candidate_queue.pop(0)
        candidate_q = np.asarray(candidate["q_final"], dtype=np.float64)
        tested_endpoint_q.append(candidate_q)
        endpoint_score_rank = int(
            candidate.get("selected_candidate", {}).get(
                "endpoint_score_rank",
                len(tested_endpoint_q) - 1,
            )
        )
        path_candidate_rank = len(tested_endpoint_q) - 1
        tested_endpoint_candidates.append(
            (candidate, endpoint_preference, path_candidate_rank)
        )
        try:
            planning_stage("cartesian_path", endpoint_score_rank=endpoint_score_rank)
            translation_m = float(
                np.linalg.norm(
                    candidate["final_eef_position_m"]
                    - candidate["start_eef_position_m"]
                )
            )
            if translation_m > MOVE_POINT_TO_POINT_MAX_TRANSLATION_M:
                raise ValueError(
                    f"planned EEF translation {translation_m:.3f}m exceeds "
                    f"{MOVE_POINT_TO_POINT_MAX_TRANSLATION_M:.3f}m"
                )
            if bool(candidate.get("trunk_assisted", False)):
                raise ValueError("move-tracked endpoint unexpectedly requests trunk motion")
            path_plan = _plan_tracked_cartesian_trajectory(
                state=state,
                arm=arm,
                q_start=candidate["q_start"],
                q_final=candidate["q_final"],
                pos_tol_m=pos_tol_m,
                ori_tol_deg=ori_tol_deg,
                max_joint_step_rad=MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
                max_waypoints=max(
                    1,
                    # ``max_steps`` is an online action budget. Keep the dense,
                    # locally verified path for direct execution; plan mode is
                    # recompiled into the shared grasp-style safe/final path.
                    MAX_TRAJECTORY_WAYPOINTS,
                ),
                tracked_anchors_eef=candidate.get("anchors_eef_m"),
                cancel_requested=(
                    cancel_event.is_set if cancel_event is not None else None
                ),
                planning_deadline_monotonic=planning_deadline_monotonic,
            )
            path_plan = {
                **path_plan,
                "planning_budget_exhausted": bool(planning_budget_exhausted),
                "observed_start_joint_limit_recovery": _json_ready(
                    start_limit_recovery
                ),
                "observed_start_max_joint_limit_recovery_rad": float(
                    max_start_limit_recovery
                ),
                "planning_start_qpos": planning_q_start.astype(float).tolist(),
            }
            selected_overlap_report = overlap_filter_report(candidate)
            if selected_overlap_report is not None:
                path_plan["frozen_rgbd_overlap_filter"] = selected_overlap_report
            check_cancelled()
        except PlanningDeadlineExceeded as candidate_error:
            # The endpoint has already passed the complete constraint gate.
            # Stop sampling immediately and use the bounded joint route below;
            # continuing to another Cartesian density would only consume the
            # execution budget without changing the requested pose.
            planning_budget_exhausted = True
            path_failures.append(
                {
                    "endpoint_score_rank": endpoint_score_rank,
                    "path_candidate_rank": path_candidate_rank,
                    "endpoint_preference": endpoint_preference,
                    "path_mode": "planning_budget_exhausted",
                    "error": str(candidate_error),
                }
            )
            try:
                return joint_fallback_result(
                    candidate,
                    endpoint_preference,
                    path_candidate_rank,
                )
            except (RuntimeError, TypeError, ValueError) as fallback_error:
                path_failures.append(
                    {
                        "endpoint_score_rank": endpoint_score_rank,
                        "path_candidate_rank": path_candidate_rank,
                        "endpoint_preference": endpoint_preference,
                        "path_mode": "joint_space_fallback",
                        "error": str(fallback_error),
                    }
                )
            break
        except (RuntimeError, TypeError, ValueError) as candidate_error:
            path_failures.append(
                {
                    "endpoint_score_rank": endpoint_score_rank,
                    "path_candidate_rank": path_candidate_rank,
                    "endpoint_preference": endpoint_preference,
                    "error": str(candidate_error),
                }
            )
            # Endpoint ranking is authoritative: after the preferred
            # candidate has passed every target constraint, try its bounded
            # same-branch joint route before accepting a less accurate or
            # farther endpoint merely because that endpoint has a Cartesian
            # route.  Cartesian-success cases never enter this branch.
            try:
                return joint_fallback_result(
                    candidate,
                    endpoint_preference,
                    path_candidate_rank,
                )
            except (RuntimeError, TypeError, ValueError) as fallback_error:
                path_failures.append(
                    {
                        "endpoint_score_rank": endpoint_score_rank,
                        "path_candidate_rank": path_candidate_rank,
                        "endpoint_preference": endpoint_preference,
                        "path_mode": "joint_space_fallback",
                        "error": str(fallback_error),
                    }
                )
            continue
        candidate["path_candidate_rank"] = path_candidate_rank
        candidate["path_endpoint_preference"] = endpoint_preference
        candidate["path_candidates_tested"] = len(tested_endpoint_q)
        candidate["path_candidate_failures"] = list(path_failures)
        candidate["endpoint_solver_failures"] = list(endpoint_solver_failures)
        candidate = {**candidate, **planning_metadata}
        planning_stage("complete", selected_constraints=candidate.get("constraints"))
        return {
            "ok": True,
            "endpoint": candidate,
            "path_plan": path_plan,
            "path_failures": path_failures,
            **planning_metadata,
        }

    # This compatibility pass covers candidates reached through a terminal
    # planning-deadline break. Ordinary Cartesian failures already attempted
    # their same-ranked joint route above, before considering the next
    # endpoint.
    for candidate, endpoint_preference, path_candidate_rank in tested_endpoint_candidates:
        check_cancelled()
        candidate_q = np.asarray(candidate["q_final"], dtype=np.float64)
        if any(
            np.allclose(candidate_q, old, atol=1.0e-10, rtol=0.0)
            for old in attempted_joint_fallback_q
        ):
            continue
        try:
            return joint_fallback_result(
                candidate,
                endpoint_preference,
                path_candidate_rank,
            )
        except (RuntimeError, TypeError, ValueError) as fallback_error:
            path_failures.append(
                {
                    "endpoint_score_rank": int(
                        candidate.get("selected_candidate", {}).get(
                            "endpoint_score_rank", path_candidate_rank
                        )
                    ),
                    "path_candidate_rank": int(path_candidate_rank),
                    "endpoint_preference": endpoint_preference,
                    "path_mode": "joint_space_fallback",
                    "error": str(fallback_error),
                }
            )

    failure_summary = "; ".join(
        f"rank {item['path_candidate_rank']}: {item['error']}"
        for item in path_failures
    )
    failed_overlap_report = overlap_filter_report()
    overlap_only_failure = bool(
        failed_overlap_report is not None
        and overlap_filter_candidate_count > 0
        and overlap_filter_endpoint_passing_count == 0
        and not tested_endpoint_q
    )
    return {
        "ok": False,
        "error": (
            (
                "no endpoint candidate passes the frozen RGB-D Lite overlap filters"
                if overlap_only_failure
                else "no endpoint candidate has a valid same-branch Cartesian path"
            )
            + (f": {failure_summary}" if failure_summary else "")
        ),
        "path_failures": path_failures,
        "endpoint_solver_failures": endpoint_solver_failures,
        "planning_budget_exhausted": bool(planning_budget_exhausted),
        **(
            {"frozen_rgbd_overlap_filter": failed_overlap_report}
            if failed_overlap_report is not None
            else {}
        ),
    }



def _validate_move_tracked_endpoint_order(target: Mapping[str, Any]) -> None:
    """Recheck spatial order from signed coordinates, not cached success flags."""

    relations = [dict(item) for item in target.get("relations", [])
                 if item.get("type") == "ordered_collinear"]
    for relation in coordinate_collinear_relations(target.get("requested_points", [])):
        if relation not in relations:
            relations.append(relation)
    if not relations:
        return
    names = list(target["on_hand_point_names"]) + list(target["off_hand_point_names"])
    points = list(target["resolved_points_robot_base_m"]) + list(
        target.get("off_hand_source_points_robot_base_m") or []
    )
    report = _evaluate_tracked_relations(points, relations, point_names=names)
    if not report["axial_order_satisfied"]:
        raise ValueError("move-tracked endpoint violates the requested collinear axial order; re-plan")
