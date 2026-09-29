"""Machine-readable evaluator-only feasibility audit for all v2 skills."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from behavior_interface_eval_test.robot_contract import ACTION_DIM


TOOL_VERSION = "official_v1"
ALLOWED_OBSERVATIONS = (
    "task_id",
    "need_new_action",
    "*::rgb",
    "*::depth_linear",
    "*::proprio",
    "*::cam_rel_poses",
)

PUBLIC_SKILLS = (
    "capture",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "move_base_to_point",
    "move_eef",
    "adjust_left_eef_pose_in_head_frame",
    "adjust_right_eef_pose_in_head_frame",
    "adjust_left_eef_pose_in_wrist_frame",
    "adjust_right_eef_pose_in_wrist_frame",
    "plan_move_eef",
    "move_in_robot_coord",
    "face_to_point",
    "adjust_plan_pose",
    "move_to_point_v2",
    "move_to_point_v3",
    "mesure_shoulder_distance",
    "plan_eef_v2",
    "exec_eef_pose_v2",
    "set_arm_to_grasp_position",
    "reset_body",
    "diag_reset_object_diff",
    "diag_safe_back_scan",
    "diag_depth_mesh_overlap",
)


class OfficialToolBoundaryError(ValueError):
    """Raised before a tool could cross the evaluator observation/action boundary."""


def _implemented(
    status: str,
    implementation: str,
    *,
    semantics: str = "exact",
    constraints: str = "",
    observations: tuple[str, ...] = (),
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": status,
        "feasibility": "implemented_now",
        "legacy_semantics": semantics,
        "implementation": implementation,
        "observations": list(observations),
        "output": "official R1Pro action[23] or observation-derived artifact",
    }
    if constraints:
        out["constraints"] = constraints
    return out


def _portable(
    reason: str,
    requirements: str,
    *,
    semantics: str = "requires_rewrite",
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "feasibility": "portable",
        "legacy_semantics": semantics,
        "reason": reason,
        "required_rewrite": requirements,
        "allowed_basis": (
            "RGB-D reconstruction, evaluator proprioception, camera-relative "
            "poses, policy-owned state, and a submission-local static robot model"
        ),
    }


TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "capture": _implemented(
        "supported",
        "save evaluator head RGB/depth and camera-relative pose",
        semantics="limited",
        constraints="segmentation, normals, object truth, and live BDDL state are omitted",
        observations=("*::rgb", "*::depth_linear", "*::proprio", "*::cam_rel_poses"),
    ),
    "capture_left_wrist_camera": _implemented(
        "supported",
        "save evaluator left-wrist RGB/depth and camera-relative pose",
        semantics="limited",
        constraints="legacy segmentation-derived gripper overlays are omitted",
        observations=("*::rgb", "*::depth_linear", "*::cam_rel_poses"),
    ),
    "capture_right_wrist_camera": _implemented(
        "supported",
        "save evaluator right-wrist RGB/depth and camera-relative pose",
        semantics="limited",
        constraints="legacy segmentation-derived gripper overlays are omitted",
        observations=("*::rgb", "*::depth_linear", "*::cam_rel_poses"),
    ),
    "move_in_robot_coord": _implemented(
        "conditional",
        "base velocity actions plus absolute trunk joint targets",
        semantics="limited",
        constraints=(
            "base distance/yaw use command odometry because official proprioception "
            "does not expose global base pose; upward uses a conservative analytic torso model"
        ),
        observations=("*::proprio",),
    ),
    "face_to_point": _implemented(
        "supported",
        "head pinhole intrinsics followed by an action-only base spin",
        observations=("*::rgb", "*::cam_rel_poses"),
    ),
    "move_eef": _implemented(
        "conditional",
        "left/right gripper open or close action",
        semantics="partial",
        constraints=(
            "gripper-only is implemented; Cartesian EEF translation requires the "
            "portable local-kinematics rewrite described below"
        ),
        observations=("*::proprio",),
    ),
    "move_base_to_point": _implemented(
        "conditional",
        "clicked evaluator depth point, RGB-D-only floor test, then spin/forward actions",
        semantics="limited",
        constraints=(
            "uses a frozen depth surface normal and command odometry; it has no "
            "simulator floor category, global localization, or dynamic collision truth"
        ),
        observations=("*::depth_linear", "*::cam_rel_poses", "*::proprio"),
    ),
    "mesure_shoulder_distance": _implemented(
        "conditional",
        "clicked depth point and static R1Pro torso/shoulder geometry",
        semantics="limited",
        constraints=(
            "image_id+u+v mode only; object_name would require policy-side visual "
            "recognition and cannot be resolved from BDDL/object handles"
        ),
        observations=("*::depth_linear", "*::cam_rel_poses", "*::proprio"),
    ),
    "set_arm_to_grasp_position": _implemented(
        "conditional",
        "full-profile arm targets with J8 locked at zero and proprio convergence checks",
        semantics="limited",
        constraints=(
            "uses the fixed R1Pro grasp-prep joint target; no simulator IK, "
            "collision query, direct joint write, or teleport fallback"
        ),
        observations=("*::proprio",),
    ),
    "reset_body": _implemented(
        "conditional",
        (
            "absolute trunk targets with evaluator-proprio closed-loop "
            "settling to the official R1Pro upright posture"
        ),
        semantics="limited",
        constraints=(
            "base, arms, and grippers are action-pinned; keep_ori_arm is blocked "
            "until policy-local arm kinematics are implemented"
        ),
        observations=("*::proprio",),
    ),
    "plan_move_eef": _portable(
        "legacy planner reads simulator camera objects and simulator-backed IK/FK",
        "local R1Pro FK/Jacobian/IK, RGB-D collision volume, and plan-record schema",
    ),
    "adjust_plan_pose": _portable(
        "legacy plan validation depends on simulator/world-frame IK state",
        "policy-local plan records, camera-frame transforms, IK, and RGB-D collision checks",
    ),
    "move_to_point_v2": _portable(
        "legacy navigation/reach pipeline queries scene geometry and simulator link poses",
        "RGB-D local map, policy localization, shoulder model, and closed-loop base/trunk controller",
    ),
    "move_to_point_v3": _portable(
        "legacy navigation/reach pipeline queries scene geometry and simulator link poses",
        "RGB-D local map, policy localization, shoulder model, and closed-loop base/trunk controller",
    ),
    "plan_eef_v2": _portable(
        "legacy planner consumes segmentation, object handles, simulator IK, and collision geometry",
        "RGB-D-only target perception, local robot kinematics, and reconstructed collision geometry",
    ),
    "exec_eef_pose_v2": _portable(
        "legacy executor uses simulator FK/IK, object-drift truth, and collision state",
        "stored policy-local joint plans plus proprioceptive closed-loop action execution",
    ),
    "adjust_left_eef_pose_in_head_frame": _portable(
        "legacy implementation uses simulator-backed IK/FK",
        "head camera transform plus local differential IK and proprioceptive execution",
    ),
    "adjust_right_eef_pose_in_head_frame": _portable(
        "legacy implementation uses simulator-backed IK/FK",
        "head camera transform plus local differential IK and proprioceptive execution",
    ),
    "adjust_left_eef_pose_in_wrist_frame": _portable(
        "legacy implementation reads simulator camera and robot kinematics",
        "left-wrist camera transform plus local differential IK and proprioceptive execution",
    ),
    "adjust_right_eef_pose_in_wrist_frame": _portable(
        "legacy implementation reads simulator camera and robot kinematics",
        "right-wrist camera transform plus local differential IK and proprioceptive execution",
    ),
    "diag_safe_back_scan": _portable(
        "legacy diagnostic mutates robot joints while probing simulator IK and collision state",
        "pure policy-local IK/path scan against an RGB-D reconstructed collision volume",
    ),
    "diag_depth_mesh_overlap": {
        "status": "blocked",
        "feasibility": "replacement_only",
        "legacy_semantics": "not_preservable",
        "reason": (
            "the legacy metric compares frozen RGB-D reconstruction against "
            "simulator mesh/collision truth, which is outside the observation allowlist"
        ),
        "allowed_replacement": (
            "RGB-D self-consistency, temporal reprojection, multi-view fusion, "
            "or held-out depth error without simulator mesh truth"
        ),
    },
    "diag_reset_object_diff": {
        "status": "blocked",
        "feasibility": "forbidden",
        "legacy_semantics": "not_preservable",
        "reason": (
            "its purpose is to enumerate, perturb, reset, and compare simulator "
            "objects and evaluator-owned reset baselines"
        ),
        "owner": "OmniGibson Evaluator test infrastructure, not the policy Interface",
    },
}


def capability_report() -> dict[str, Any]:
    tools = deepcopy(TOOL_CAPABILITIES)
    runtime_counts = {
        status: sum(1 for item in tools.values() if item["status"] == status)
        for status in ("supported", "conditional", "blocked")
    }
    feasibility_counts = {
        status: sum(1 for item in tools.values() if item["feasibility"] == status)
        for status in (
            "implemented_now",
            "portable",
            "replacement_only",
            "forbidden",
        )
    }
    return {
        "mode": "official-strict",
        "tool_version": TOOL_VERSION,
        "allowed_observations": list(ALLOWED_OBSERVATIONS),
        "policy": (
            "The Interface receives evaluator observations only and returns one "
            f"official {ACTION_DIM}-D action. It has no simulator, scene, object, "
            "or BDDL handle."
        ),
        "counts": runtime_counts,
        "feasibility_counts": feasibility_counts,
        "tools": tools,
    }


def _has_value(args: dict[str, Any], key: str) -> bool:
    value = args.get(key)
    return value is not None and str(value).strip() != ""


def _finite_float(args: dict[str, Any], key: str, default: float = 0.0) -> float:
    import math

    try:
        value = float(args.get(key, default))
    except (TypeError, ValueError) as exc:
        raise OfficialToolBoundaryError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise OfficialToolBoundaryError(f"{key} must be finite")
    return value


def validate_submission(name: str, args: dict[str, Any]) -> dict[str, Any]:
    name = str(name)
    normalized = dict(args or {})
    capability = TOOL_CAPABILITIES.get(name)
    if capability is None:
        raise OfficialToolBoundaryError(
            f"tool {name!r} has no {TOOL_VERSION} boundary audit"
        )
    if capability["status"] == "blocked":
        raise OfficialToolBoundaryError(
            f"{TOOL_VERSION} blocks {name}: {capability['reason']}"
        )

    if name == "move_in_robot_coord":
        limits = {
            "forward": 5.0,
            "spin": 360.0,
            "pitch": 90.0,
            "upward": 0.45,
        }
        for key, limit in limits.items():
            value = _finite_float(normalized, key)
            if abs(value) > limit:
                raise OfficialToolBoundaryError(
                    f"{key}={value} exceeds official_v1 limit +/-{limit}"
                )
        normalized["nav_guard"] = False

    if name == "move_eef":
        moving = any(
            abs(_finite_float(normalized, key)) > 1e-9
            for key in ("upward", "forward", "leftward")
        )
        if moving or any(_has_value(normalized, key) for key in ("u", "v", "depth")):
            raise OfficialToolBoundaryError(
                "official_v1 move_eef currently supports gripper-only actions; "
                "Cartesian translation is portable but not implemented"
            )
        gripper = str(normalized.get("gripper", "keep")).strip().lower()
        if gripper not in ("open", "close", "keep"):
            raise OfficialToolBoundaryError(
                "official_v1 move_eef requires gripper='open', 'close', or 'keep'"
            )

    if name in ("capture", "capture_left_wrist_camera", "capture_right_wrist_camera"):
        if not str(normalized.get("session_id", "")).strip():
            raise OfficialToolBoundaryError("session_id is required")

    if name in ("move_base_to_point", "mesure_shoulder_distance"):
        if name == "mesure_shoulder_distance" and str(
            normalized.get("object_name", "")
        ).strip():
            raise OfficialToolBoundaryError(
                "official_v1 shoulder measurement does not resolve object_name; "
                "use image_id + u + v from evaluator RGB-D"
            )
        missing = [
            key
            for key in ("session_id", "image_id", "u", "v")
            if not _has_value(normalized, key)
        ]
        if missing:
            raise OfficialToolBoundaryError(
                f"{name} requires {', '.join(missing)}"
            )

    if name == "set_arm_to_grasp_position":
        arm = str(normalized.get("arm", "right")).strip().lower()
        if arm not in ("left", "right", "both"):
            raise OfficialToolBoundaryError("arm must be left, right, or both")
        gripper = str(normalized.get("gripper", "keep")).strip().lower()
        if gripper not in ("keep", "open"):
            raise OfficialToolBoundaryError("gripper must be keep or open")

    if name == "reset_body":
        keep_ori = str(normalized.get("keep_ori_arm", "none")).strip().lower()
        if keep_ori not in ("", "none", "false", "0", "no"):
            raise OfficialToolBoundaryError(
                "official_v1 reset_body cannot preserve world EEF orientation "
                "until policy-local arm kinematics are implemented"
            )
        pitch_deg = _finite_float(normalized, "pitch_deg")
        if abs(pitch_deg) > 60.0:
            raise OfficialToolBoundaryError("reset_body pitch_deg exceeds +/-60")

    return normalized


assert set(TOOL_CAPABILITIES) == set(PUBLIC_SKILLS)
