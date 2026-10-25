"""RoboDojo boundary audit for the public Interface v2 tools."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from tools.util.contract import (
    ACTION_DIM,
    ARM_DOF,
    validate_move_point_to_point_args,
    validate_move_tracked_point_args,
    validate_read_depth_args,
    validate_track_object_distance_args,
)


TOOL_VERSION = "official_v2"
OFFICIAL_ACTION_LABEL = f"RoboDojo dual-X5 action dict[{ACTION_DIM} scalars]"
ALLOWED_OBSERVATIONS = (
    "instruction",
    "vision.*.color",
    "vision.*.depth",
    "vision.*.intrinsic_matrix",
    "vision.*.extrinsic_matrix",
    "state.*_arm_joint_state",
    "state.*_ee_joint_state",
    "state.*_ee_pose",
)

PUBLIC_TOOLS = (
    "capture_head_camera",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "read_depth",
    "track_object_distance",
    "move_chassis_to_floor_point",
    "adjust_chassis",
    "adjust_pitch",
    "adjust_height",
    "spin_to_facing_point",
    "open_gripper",
    "close_gripper",
    "adjust_left_eef_pose_in_head_frame",
    "adjust_right_eef_pose_in_head_frame",
    "adjust_left_eef_pose_in_wrist_frame",
    "adjust_right_eef_pose_in_wrist_frame",
    "move_point_to_point",
    "move_tracked_point",
    "plan_eef_translation_to_uvd_point",
    "move_to_reach_point",
    "measure_shoulder_distance",
    "plan_grasp_point_filter_rgbd_lite",
    "plan_press_point",
    "exec_plan_pose",
    "set_arm_to_grasp_position",
    "reset_body",
)

# Retired names are kept only for explicit boundary errors.  They are not
# part of the registry, capability report, or UI metadata.
REMOVED_TOOLS = frozenset(
    {
        "plan_grasp_point_filter",
        "plan_grasp_point_filter_rgbd",
    }
)


class OfficialToolBoundaryError(ValueError):
    """Raised before a public tool can cross the evaluator boundary."""


def _implemented(
    status: str,
    implementation: str,
    *,
    semantics: str,
    constraints: str = "",
    observations: tuple[str, ...] = (),
    legacy_violations: tuple[str, ...] = (),
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "status": status,
        "compliance": "compliant" if status == "supported" else "conditional",
        "feasibility": "implemented_now",
        "semantics": semantics,
        "implementation": implementation,
        "observations": list(observations),
        "output": f"{OFFICIAL_ACTION_LABEL} or observation-derived artifact",
        "legacy_violations_removed": list(legacy_violations),
    }
    if constraints:
        item["constraints"] = constraints
    return item


def _blocked(
    current_violations: tuple[str, ...],
    required_rewrite: str,
    *,
    semantics: str = "portable_with_rewrite",
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "compliance": "noncompliant_current_implementation",
        "feasibility": "portable",
        "semantics": semantics,
        "current_violations": list(current_violations),
        "required_rewrite": required_rewrite,
        "allowed_basis": (
            "evaluator RGB-D, evaluator proprioception, evaluator camera-relative "
            "poses, policy-owned state, and a submission-local static official X5 model"
        ),
    }


# Retained as a porting audit of the BEHAVIOR implementation. The public
# capability report below is rebuilt from the RoboDojo implementation details.
_LEGACY_PORT_AUDIT: dict[str, dict[str, Any]] = {
    "capture_head_camera": _implemented(
        "supported",
        (
            "persist evaluator head RGB/depth and camera-relative pose, then "
            "project the frozen v2 metric base-path HUD from policy-local pose"
        ),
        semantics=("observation-equivalent capture with the frozen v2 base-path HUD"),
        constraints=(
            "the fixed yellow 0.10m margin rails are visual HUD geometry, not "
            "collision truth; "
            "segmentation, simulator normals, object identity, object pose, "
            "live BDDL state, and simulator meshes are omitted"
        ),
        observations=("*::rgb", "*::depth_linear", "*::proprio", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy capture requested seg_instance_id and normal",
            "legacy capture read task-relevant objects and global robot/link poses",
        ),
    ),
    "capture_left_wrist_camera": _implemented(
        "conditional",
        (
            "persist evaluator left-wrist RGB/depth and camera-relative pose, "
            "reconstruct visible points, and mark points inside the local "
            "two-finger opening red"
        ),
        semantics="RGB-D capture with a local red grasp-volume overlay",
        constraints=(
            "the red mask uses a policy-local official X5 dynamic finger mesh "
            "depth match to reject self surfaces; it does not use simulator "
            "instance segmentation, robot IDs, or semantic object truth"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "legacy overlay required seg_instance_id and simulator robot instance IDs",
            "legacy capture read live EEF and gripper state through the simulator world",
        ),
    ),
    "capture_right_wrist_camera": _implemented(
        "conditional",
        (
            "persist evaluator right-wrist RGB/depth and camera-relative pose, "
            "reconstruct visible points, and mark points inside the local "
            "two-finger opening red"
        ),
        semantics="RGB-D capture with a local red grasp-volume overlay",
        constraints=(
            "the red mask uses a policy-local official X5 dynamic finger mesh "
            "depth match to reject self surfaces; it does not use simulator "
            "instance segmentation, robot IDs, or semantic object truth"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "legacy overlay required seg_instance_id and simulator robot instance IDs",
            "legacy capture read live EEF and gripper state through the simulator world",
        ),
    ),
    "read_depth": _implemented(
        "supported",
        (
            "load the immutable depth_linear artifact belonging to the selected "
            "frozen evaluator capture and read exactly one UV pixel"
        ),
        semantics="exact frozen-capture depth lookup in meters",
        constraints=(
            "u/v use the public 0..1000 coordinate frame; invalid or non-positive "
            "depth and RGB/depth resolution mismatch are reported as failures; "
            "no neighborhood interpolation or live observation mixing is used"
        ),
        observations=("*::depth_linear",),
        legacy_violations=(
            "no simulator sensor, render helper, segmentation, or scene state is read",
        ),
    ),
    "track_object_distance": _implemented(
        "conditional",
        (
            "atomically mark multiple named points in the exact frozen head "
            "RGB-D capture identified by session_id and image_id, then update "
            "measured depths and current XYZ from each subsequent evaluator "
            "observation in policy memory"
        ),
        semantics=(
            "live named surface-point depth and XYZ in current robot base coord tracking "
            "in meters"
        ),
        constraints=(
            "names are model annotations rather than verified simulator object "
            "identities; every UV batch must carry the image_id of the displayed "
            "frozen capture, whose tracker RGB-D/camera geometry is SHA-256 "
            "verified; registration requires the capture to match the current "
            "observation sequence, and missing, modified, evicted, stale, or "
            "cross-episode frames fail without falling back to latest-frame UV; XYZ "
            "is derived only from synchronized evaluator depth and camera-relative "
            "pose; an unobserved point has no published depth or XYZ, an off-screen "
            "or lost point is retired, and predictions are never published as "
            "measurements"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "no segmentation, object ID, object pose, world pose, or simulator state is read",
        ),
    ),
    "move_chassis_to_floor_point": _implemented(
        "conditional",
        (
            "unproject frozen evaluator depth, test local slope/height, then "
            "move the submitted static base-front reference to that point"
        ),
        semantics=(
            "yaw-preserving simultaneous forward/left translation with "
            "evaluator-proprioceptive arrival tracking; the clicked target "
            "references the midpoint of the blue path-start/front edge"
        ),
        constraints=(
            "no floor category, object AABB, global base pose, path-planner collision "
            "truth, or simulator arrival truth; base progress integrates evaluator base_qvel"
        ),
        observations=("*::depth_linear", "*::cam_rel_poses", "*::proprio"),
        legacy_violations=(
            "legacy implementation iterated task.object_scope and scene objects",
            "legacy implementation used floor categories and simulator AABBs",
            "legacy implementation read global robot pose and simulator collision state",
        ),
    ),
    "adjust_chassis": _implemented(
        "conditional",
        "move along one robot-frame forward/left vector, then rotate in place",
        semantics=(
            "simultaneous forward/lateral proprioceptive tracking with a "
            "bounded vector speed and adaptive grounded recovery"
        ),
        constraints=(
            "progress integrates evaluator base_qvel and action history; "
            "settling uses base_qvel; no global pose, contact, "
            "segmentation, scene graph, object state, or simulator collision truth"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy navigation guard queried simulator geometry and collision truth",
            "legacy completion checked global base pose",
        ),
    ),
    "adjust_pitch": _implemented(
        "supported",
        "interpolate the observed trunk pitch joint through official actions",
        semantics="pitch-only trunk action",
        observations=("*::proprio",),
        legacy_violations=(
            "legacy implementation could read simulator joint/controller objects",
        ),
    ),
    "adjust_height": _implemented(
        "conditional",
        "derive a conservative trunk target with a local analytic torso model and emit actions",
        semantics="vertical torso adjustment within the local model workspace",
        constraints=(
            "no simulator FK or exact chest world height; requests outside the "
            "conservative analytic workspace are rejected"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy vertical lookup and verification used simulator kinematics",
            "legacy implementation could inspect world-frame chest height",
        ),
    ),
    "spin_to_facing_point": _implemented(
        "supported",
        "convert a relative head-image u coordinate to a yaw action",
        semantics="horizontal image centering",
        observations=("*::rgb", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy implementation could verify final facing from global robot pose",
        ),
    ),
    "open_gripper": _implemented(
        "supported",
        "emit the official open command for the selected gripper",
        semantics="gripper-only action",
        observations=("*::proprio",),
        legacy_violations=(
            "legacy helpers could access gripper joint objects directly",
        ),
    ),
    "close_gripper": _implemented(
        "supported",
        (
            "command symmetric X5 finger closure, detect target-band or "
            "stable non-limit blockage from legal proprioception, and keep "
            "maximum closing preload latched across later robot actions"
        ),
        semantics="gripper-only action",
        constraints=(
            "RoboDojo provides no assisted-grasp attachment or contact truth; "
            "retention is produced only by the official X5 actuator and PhysX "
            "contact/friction, so the tool verifies proprioceptive blockage "
            "without claiming a rigid grasp constraint"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy helpers could access gripper joint objects or grasp state directly",
        ),
    ),
    "adjust_left_eef_pose_in_head_frame": _implemented(
        "conditional",
        (
            "adjust the left EEF with exactly one translation family: x/y/z are "
            "meter deltas in the current robot base frame (+X chassis-forward, "
            "+Y chassis-left, +Z chassis-up), while forward/leftward/upward are "
            "meter deltas in the starting head-camera frame; never mix the two "
            "families. Compute FK/Jacobian/IK from current proprio and the "
            f"submission URDF, then close the loop through official action[{ACTION_DIM}] targets"
        ),
        semantics="mutually exclusive robot-base XYZ or head-camera translation plus local-gripper RPY",
        constraints=(
            "omitted axes in the selected family are zero; y and leftward share "
            "the positive-left sign, but y is base-fixed while leftward is rotated "
            "from the head camera; uses joint limits and proprioceptive "
            "stall/tracking guards; no hidden contact state or simulator scene "
            "collision truth is available"
        ),
        observations=("*::proprio", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy implementation read simulator-backed EEF pose and Jacobians",
            "legacy target IK and controller state came from the live simulator",
        ),
    ),
    "adjust_right_eef_pose_in_head_frame": _implemented(
        "conditional",
        (
            "adjust the right EEF with exactly one translation family: x/y/z are "
            "meter deltas in the current robot base frame (+X chassis-forward, "
            "+Y chassis-left, +Z chassis-up), while forward/leftward/upward are "
            "meter deltas in the starting head-camera frame; never mix the two "
            "families. Compute FK/Jacobian/IK from current proprio and the "
            f"submission URDF, then close the loop through official action[{ACTION_DIM}] targets"
        ),
        semantics="mutually exclusive robot-base XYZ or head-camera translation plus local-gripper RPY",
        constraints=(
            "omitted axes in the selected family are zero; y and leftward share "
            "the positive-left sign, but y is base-fixed while leftward is rotated "
            "from the head camera; uses joint limits and proprioceptive "
            "stall/tracking guards; no hidden contact state or simulator scene "
            "collision truth is available"
        ),
        observations=("*::proprio", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy implementation read simulator-backed EEF pose and Jacobians",
            "legacy target IK and controller state came from the live simulator",
        ),
    ),
    "adjust_left_eef_pose_in_wrist_frame": _implemented(
        "conditional",
        (
            "freeze the evaluator left-wrist camera-relative frame, translate with "
            "submission-local 7DOF tangent IK, then hold J1-J4 and solve local "
            f"gripper RPY with J5-J7 through official action[{ACTION_DIM}] targets"
        ),
        semantics="wrist-camera translation followed by fixed-J1234 J567 orientation",
        constraints=(
            "J8 is hard-locked at 0rad and excluded from FK, IK, and optimization; "
            "scene interaction is detected only as proprioceptive tracking/stall"
        ),
        observations=("*::proprio", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy implementation read the live simulator wrist sensor world pose",
            "legacy implementation read simulator joints, EEF pose, and Jacobians",
            "legacy implementation pinned or directly reset arm/tool-roll state",
        ),
    ),
    "adjust_right_eef_pose_in_wrist_frame": _implemented(
        "conditional",
        (
            "freeze the evaluator right-wrist camera-relative frame, translate with "
            "submission-local 7DOF tangent IK, then hold J1-J4 and solve local "
            f"gripper RPY with J5-J7 through official action[{ACTION_DIM}] targets"
        ),
        semantics="wrist-camera translation followed by fixed-J1234 J567 orientation",
        constraints=(
            "J8 is hard-locked at 0rad and excluded from FK, IK, and optimization; "
            "scene interaction is detected only as proprioceptive tracking/stall"
        ),
        observations=("*::proprio", "*::cam_rel_poses"),
        legacy_violations=(
            "legacy implementation read the live simulator wrist sensor world pose",
            "legacy implementation read simulator joints, EEF pose, and Jacobians",
            "legacy implementation pinned or directly reset arm/tool-roll state",
        ),
    ),
    "move_point_to_point": _implemented(
        "conditional",
        (
            "unproject both clicks from one frozen evaluator head RGB-D capture, "
            "select the nearest EEF with capture-time proprioception and static FK, "
            f"then track the translated EEF pose through action[{ACTION_DIM}] targets"
        ),
        semantics=(
            "rigid source-point transfer to target plus configurable robot-base +Z offset"
        ),
        constraints=(
            "above_target_point_m defaults to 0m; the grasp check is only a stable "
            "intermediate-aperture proprioceptive heuristic, and collision/stall coverage "
            "contains no simulator or hidden object truth"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "legacy arm selection read simulator-backed EEF and gripper state",
            "legacy motion read simulator FK, EEF pose, and Jacobians",
            "legacy execution could consume simulator collision or reachability truth",
        ),
    ),
    "plan_eef_translation_to_uvd_point": _blocked(
        (
            "reads simulator camera objects and world-frame EEF pose",
            "uses simulator-backed IK/FK for line-path validation",
            "may use simulator collision geometry",
        ),
        "unproject UVD from a frozen evaluator capture, solve the straight Cartesian "
        "path with local X5 IK, and collision-check only against RGB-D reconstruction",
    ),
    "move_to_reach_point": _implemented(
        "conditional",
        (
            "unproject the clicked evaluator depth point, preserve the legacy "
            "pre-lift/XY/yaw/q3/fallback state machine, and execute it with "
            "RGB-D guards plus submission-local torso and arm kinematics"
        ),
        semantics="clicked-point chord-sphere reach positioning",
        constraints=(
            "base completion is estimated in policy-local odometry; obstacle checks "
            "cover observed RGB-D geometry only; target reacquisition uses RGB optical "
            "flow plus current depth when available; keep_ori_arm fixes J1-J4 and J8 "
            "and solves J5-J7 with the submission-local model"
        ),
        observations=("*::rgb", "*::depth_linear", "*::cam_rel_poses", "*::proprio"),
        legacy_violations=(
            "legacy implementation queried scene geometry and simulator link poses",
            "legacy implementation used global base pose and path-planner truth",
            "legacy keep_ori_arm used simulator FK and direct joint mutation",
            "legacy result exposed world-frame target and arrival truth",
        ),
    ),
    "measure_shoulder_distance": _implemented(
        "conditional",
        "unproject a clicked evaluator depth point and measure from static X5 shoulders",
        semantics="clicked-point mode only",
        constraints=(
            "object_name mode is rejected because resolving a BDDL object and its "
            "AABB center requires privileged object identity and geometry"
        ),
        observations=("*::depth_linear", "*::cam_rel_poses", "*::proprio"),
        legacy_violations=(
            "legacy object_name mode resolved task objects and read simulator AABBs",
            "legacy shoulder positions could come from simulator link poses",
        ),
    ),
    "plan_grasp_point_filter_rgbd_lite": _implemented(
        "conditional",
        (
            "select the camera-nearest visible surface point inside a click-centered "
            "30 mm sphere, reconstruct the frozen evaluator RGB-D view, preserve the "
            "legacy 1120-pose sampling and lite stage/ranking policy, and solve "
            "retained "
            "safe/final targets with observation-calibrated policy-local X5 DLS IK/FK; "
            "render the selected official X5 pose as a red depth-occluded overlay "
            "and return BEHAVIOR-compatible front/side/top local grasp diagnostics"
        ),
        semantics="local-surface RGB-D grasp planning with unchanged lite pose policy",
        constraints=(
            "collision coverage is limited to geometry visible or completed from the "
            "frozen depth view; IK and open-gripper geometry use the official X5A "
            "chain frozen into the submission plus capture-time proprioception; "
            "no simulator fallback is available"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "legacy implementation read world.robot, EEF links, robot USD, and joint objects",
            "legacy final verification temporarily wrote robot joint positions",
            "legacy helper imports mixed the production planner call stack into test",
        ),
    ),
    "plan_press_point": _blocked(
        (
            "uses simulator GPU IK and simulator EEF link definitions",
            "probes FK by setting robot joint positions and then restoring them",
            "reads simulator-backed current EEF pose",
        ),
        "keep depth-derived point/normal sampling, but solve and verify safe/press "
        "trajectories with local X5 FK/IK and RGB-D collision geometry",
    ),
    "exec_plan_pose": _implemented(
        "conditional",
        (
            "load a policy-owned immutable joint trajectory, validate its digest, "
            "profile, episode, and motion epoch; use the planned arm by default or "
            "select another arm whose stored final_q reaches the same target under "
            "submission-local FK; resolve that arm's requested safe retreat with "
            "submission-local cuRobo/URDF, then execute only absolute official "
            "actions with proprioceptive tracking"
        ),
        semantics="from-current action-only execution of a signed plan pose",
        constraints=(
            "the current lite planner signs every locally verified stored final_q arm "
            "branch and compiles the recommended arm's safe trajectory; execution "
            "always rebuilds current-to-safe from evaluator proprioception; an arm "
            "switch or mismatched back_m is tested in v2 candidate order against the "
            "selected stored final_q with submission-local IK/FK, while that final "
            "endpoint remains immutable; a measured start just outside submission-local "
            "limits is permitted only along a monotonic recovery to safe_q; full "
            "swept-volume self-collision and frozen-RGB-D validation remain incomplete"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy execution recomputed simulator IK and read simulator collision state",
            "legacy execution read object drift, object motion, and grasp truth",
            "legacy execution could reset tool roll through direct joint mutation",
        ),
    ),
    "set_arm_to_grasp_position": _implemented(
        "conditional",
        (
            f"retrace policy-recorded evaluator observations for the {ARM_DOF}-DOF "
            "X5 arm, then servo each observed joint to the fixed grasp-prep target "
            "and verify convergence from fresh evaluator proprioception"
        ),
        semantics="legal fixed joint-space replacement",
        constraints=(
            "a previously executed path is reused only when its observation history "
            "contains a neutral anchor; otherwise the controller uses bounded direct "
            "feedback and reports each residual stalled joint"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy implementation used simulator IK and direct joint-set fallbacks",
            "legacy collision validation used simulator geometry",
        ),
    ),
    "reset_body": _implemented(
        "conditional",
        "interpolate observed trunk qpos to the upright target through official actions",
        semantics="body reset with arms/grippers action-pinned",
        constraints=(
            "keep_ori_arm must be none; preserving world EEF orientation requires "
            "policy-local arm FK/IK"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy reset directly set joint positions and robot pose",
            "legacy keep_ori_arm compensation used simulator kinematics",
        ),
    ),
}


_UNSUPPORTED_EMBODIMENT_TOOLS = {
    "move_chassis_to_floor_point",
    "adjust_chassis",
    "adjust_pitch",
    "adjust_height",
    "spin_to_facing_point",
    "reset_body",
}

_ROBODOJO_TOOL_DETAILS: dict[str, tuple[str, str, str, str]] = {
    "capture_head_camera": (
        "supported",
        "freeze evaluator head RGB-D, camera matrices, and observed dual-X5 state",
        "head-camera capture with immutable policy-owned artifacts",
        "segmentation, reward, object identity, and simulator state are omitted",
    ),
    "capture_left_wrist_camera": (
        "supported",
        (
            "freeze evaluator left-wrist RGB-D, register the physical finger "
            "separation against the official X5 mesh, reject depth-matched "
            "finger surfaces, and mark the remaining dynamic opening red"
        ),
        "left-wrist RGB-D with a registered X5 grasp-volume overlay and three views",
        (
            "uses evaluator RGB-D/camera matrices/proprio and a policy-local "
            "official X5 model; no segmentation, simulator state, or handles"
        ),
    ),
    "capture_right_wrist_camera": (
        "supported",
        (
            "freeze evaluator right-wrist RGB-D, register the physical finger "
            "separation against the official X5 mesh, reject depth-matched "
            "finger surfaces, and mark the remaining dynamic opening red"
        ),
        "right-wrist RGB-D with a registered X5 grasp-volume overlay and three views",
        (
            "uses evaluator RGB-D/camera matrices/proprio and a policy-local "
            "official X5 model; no segmentation, simulator state, or handles"
        ),
    ),
    "read_depth": (
        "supported",
        "verify and read one pixel from an immutable evaluator depth artifact",
        "frozen-capture metric depth lookup",
        "u/v remain in the public 0..1000 coordinate frame",
    ),
    "track_object_distance": (
        "conditional",
        "track agent-named RGB points with policy-local optical flow and current depth",
        "visible surface-point tracking rather than simulator object tracking",
        "names are annotations; identity, occluded position, and object pose are unknown",
    ),
    "move_tracked_point": (
        "conditional",
        (
            "consume one to six active live tracked RGB-D points labeled on-hand "
            "or frozen off-hand, optionally use predefined physical X5 fingertip "
            "points and lower one or more of six quick presets into "
            "the same numeric/symbolic/affine relation model, select the nearest "
            "observed X5 EEF, sign the joint trajectory, and either replay it "
            "through evaluator actions or return a frozen-head red X5 preview "
            "for exec_plan_pose"
        ),
        "live tracked rigid-point motion with BEHAVIOR v2 constraint semantics",
        (
            "all on-hand names assert one rigid object while off-hand names are "
            "frozen observed references; success requires synchronized subsequent "
            "proprioception, live RGB-D tracking, target and quick constraints, "
            "off-hand stability, and EEF-anchor agreement; no object, grasp, "
            "contact, segmentation, or collision truth"
        ),
    ),
    "move_chassis_to_floor_point": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has a fixed base",
    ),
    "adjust_chassis": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has a fixed base",
    ),
    "adjust_pitch": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has no articulated torso",
    ),
    "adjust_height": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has no articulated torso",
    ),
    "spin_to_facing_point": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has a fixed base",
    ),
    "open_gripper": (
        "supported",
        "hold fixed dual-X5 arm targets; command full opening, settle, then check wrist depth",
        "normalized position command proxy within 0.005, plus 10 fresh stable control steps",
        "reliable partial-opening depth vetoes success; unavailable depth is explicit; finite control-time budget",
    ),
    "close_gripper": (
        "supported",
        "repeat complete dual-X5 joint targets, then sustain closing preload for 1.0s",
        "normalized gripper position control",
        "non-limit blockage must remain stable; collapse toward the lower limit is rejected",
    ),
    "adjust_left_eef_pose_in_head_frame": (
        "conditional",
        "convert head/base deltas and close the loop on fresh evaluator EEF observations",
        "left EEF delta motion",
        "no collision or contact oracle is available",
    ),
    "adjust_right_eef_pose_in_head_frame": (
        "conditional",
        "convert head/base deltas and close the loop on fresh evaluator EEF observations",
        "right EEF delta motion",
        "no collision or contact oracle is available",
    ),
    "adjust_left_eef_pose_in_wrist_frame": (
        "conditional",
        "convert left-wrist deltas and close the loop on fresh evaluator EEF observations",
        "left wrist-relative EEF motion",
        "no collision or contact oracle is available",
    ),
    "adjust_right_eef_pose_in_wrist_frame": (
        "conditional",
        "convert right-wrist deltas and close the loop on fresh evaluator EEF observations",
        "right wrist-relative EEF motion",
        "no collision or contact oracle is available",
    ),
    "move_point_to_point": (
        "conditional",
        "translate the nearer EEF between frozen RGB-D clicks with observed feedback",
        "point-to-point EEF transfer",
        "uses a conservative reach bound but no collision oracle",
    ),
    "plan_eef_translation_to_uvd_point": (
        "conditional",
        "compile a frozen RGB-D point into a signed policy-owned EEF plan",
        "single-waypoint UVD translation plan",
        "plan validity is geometric and does not imply collision safety",
    ),
    "move_to_reach_point": (
        "conditional",
        "measure fixed-base reachability and decline navigation when unreachable",
        "clicked-point reach check",
        "dual ARX-X5 cannot move its base closer to the point",
    ),
    "measure_shoulder_distance": (
        "supported",
        "measure a frozen RGB-D point from static dual-X5 shoulder origins",
        "clicked-point shoulder distance",
        "object_name identity mode is rejected",
    ),
    "plan_grasp_point_filter_rgbd_lite": (
        "conditional",
        "select the camera-nearest surface point in a 30 mm sphere, then score "
        "1120 official-X5 poses with 3 mm overlap/grasp volumes and paired "
        "safe/final local IK; return matching front/side/top grasp diagnostics",
        "BEHAVIOR-aligned lightweight RGB-D grasp planning for the X5 configuration",
        "red preview and three views use frozen depth plus the official mesh; no identity, simulator collision oracle, or evaluator IK is used",
    ),
    "plan_press_point": (
        "conditional",
        "build signed safe/press waypoints from a clicked RGB-D surface normal",
        "RGB-D press approximation",
        "no contact truth or collision oracle is used",
    ),
    "exec_plan_pose": (
        "conditional",
        "verify the signed plan and close the loop on each evaluator waypoint",
        "signed EEF plan execution",
        "observed endpoint error is reported; collision safety is not claimed",
    ),
    "set_arm_to_grasp_position": (
        "supported",
        "retrace the observed outgoing path, then servo X5 joints to the evaluator's canonical forward-facing neutral pose",
        "dual-X5 action-only grasp preparation",
        "history contains only policy actions and evaluator observations; each joint command is independently bounded, blocked joints are named, and unselected channels hold",
    ),
    "reset_body": (
        "conditional",
        "API-compatible structured unsupported_embodiment result",
        "public call preserved with no physical effect",
        "dual ARX-X5 has no mobile base or articulated torso",
    ),
}


def _robodojo_capability(name: str) -> dict[str, Any]:
    status, implementation, semantics, constraints = _ROBODOJO_TOOL_DETAILS[name]
    return _implemented(
        status,
        implementation,
        semantics=semantics,
        constraints=constraints,
        observations=ALLOWED_OBSERVATIONS,
    )


TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    name: _robodojo_capability(name) for name in PUBLIC_TOOLS
}


def capability_report() -> dict[str, Any]:
    tools = deepcopy(TOOL_CAPABILITIES)
    counts = {
        status: sum(1 for item in tools.values() if item["status"] == status)
        for status in ("supported", "conditional", "blocked")
    }
    feasibility_counts = {
        status: sum(1 for item in tools.values() if item["feasibility"] == status)
        for status in ("implemented_now", "portable")
    }
    return {
        "mode": "official-strict",
        "tool_version": TOOL_VERSION,
        "registry_layer": "Interface v2 public tools",
        "allowed_observations": list(ALLOWED_OBSERVATIONS),
        "policy": (
            "The Interface receives RoboDojo evaluator observations only and "
            f"returns a dual-X5 action dict ({ACTION_DIM} joint/gripper scalars). "
            "It has no simulator, reward, scene, object, or task handle."
        ),
        "counts": counts,
        "feasibility_counts": feasibility_counts,
        "tools": tools,
    }


def _has_value(args: dict[str, Any], key: str) -> bool:
    value = args.get(key)
    return value is not None and str(value).strip() != ""


def _finite_float(
    args: dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    try:
        value = float(args.get(key, default))
    except (TypeError, ValueError) as exc:
        raise OfficialToolBoundaryError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise OfficialToolBoundaryError(f"{key} must be finite")
    return value


def _require(args: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if not _has_value(args, key)]
    if missing:
        raise OfficialToolBoundaryError(
            f"missing required arguments: {', '.join(missing)}"
        )


def _validate_arm(
    args: dict[str, Any],
    *,
    default: str = "right",
    allow_both: bool = False,
) -> str:
    arm = str(args.get("arm", default)).strip().lower()
    allowed = {"left", "right"}
    if allow_both:
        allowed.add("both")
    if arm not in allowed:
        raise OfficialToolBoundaryError(
            f"arm must be one of {', '.join(sorted(allowed))}"
        )
    return arm


def validate_submission(name: str, args: dict[str, Any]) -> dict[str, Any]:
    name = str(name)
    normalized = dict(args or {})
    capability = TOOL_CAPABILITIES.get(name)
    if capability is None:
        raise OfficialToolBoundaryError(
            f"tool {name!r} is not in the public Interface v2 tool surface"
        )
    if capability["status"] == "blocked":
        violations = "; ".join(capability["current_violations"])
        raise OfficialToolBoundaryError(f"{TOOL_VERSION} blocks {name}: {violations}")

    if name in (
        "capture_head_camera",
        "capture_left_wrist_camera",
        "capture_right_wrist_camera",
    ):
        _require(normalized, "session_id")

    if name == "read_depth":
        try:
            normalized = validate_read_depth_args(normalized)
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc

    if name == "track_object_distance":
        try:
            normalized = validate_track_object_distance_args(normalized)
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc


    if name == "move_point_to_point":
        try:
            normalized = validate_move_point_to_point_args(normalized)
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc

    if name == "move_tracked_point":
        try:
            normalized = validate_move_tracked_point_args(normalized)
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc

    if name == "move_chassis_to_floor_point":
        _require(normalized, "session_id", "image_id", "u", "v")
        for key in ("u", "v"):
            value = _finite_float(normalized, key)
            if value < 0.0 or value > 1000.0:
                raise OfficialToolBoundaryError(f"{key} must be in 0..1000")

    if name == "adjust_chassis":
        # Large forward values intentionally mean "advance until the observed
        # reachable limit". The timeout remains the hard execution budget.
        limits = {"forward": 50.0, "translation": 50.0, "spin": 360.0}
        for key, limit in limits.items():
            value = _finite_float(normalized, key)
            if abs(value) > limit:
                raise OfficialToolBoundaryError(
                    f"{key}={value} exceeds official_v2 limit +/-{limit}"
                )

    if name == "adjust_pitch":
        degree = _finite_float(
            normalized,
            "degree",
            normalized.get("pitch", 0.0),
        )
        if abs(degree) > 90.0:
            raise OfficialToolBoundaryError("degree exceeds +/-90")
        normalized["degree"] = degree

    if name == "adjust_height":
        upward = _finite_float(normalized, "upward")
        if abs(upward) > 0.45:
            raise OfficialToolBoundaryError("upward exceeds +/-0.45m")

    if name == "spin_to_facing_point":
        _require(normalized, "u", "v")
        for key in ("u", "v"):
            value = _finite_float(normalized, key)
            if value < 0.0 or value > 1000.0:
                raise OfficialToolBoundaryError(f"{key} must be in 0..1000")

    if name in ("open_gripper", "close_gripper"):
        _validate_arm(normalized)
        if name == "close_gripper":
            timeout_s = _finite_float(normalized, "timeout_s", 3.5)
            if not 0.3 <= timeout_s <= 10.0:
                raise OfficialToolBoundaryError("timeout_s must be in 0.3..10.0s")
            normalized["timeout_s"] = timeout_s

    if name in (
        "adjust_left_eef_pose_in_head_frame",
        "adjust_right_eef_pose_in_head_frame",
        "adjust_left_eef_pose_in_wrist_frame",
        "adjust_right_eef_pose_in_wrist_frame",
    ):
        head_adjust = name in (
            "adjust_left_eef_pose_in_head_frame",
            "adjust_right_eef_pose_in_head_frame",
        )
        if head_adjust:
            base_keys = ("x", "y", "z")
            camera_keys = ("forward", "leftward", "upward")
            common_keys = (
                "roll",
                "pitch",
                "yaw",
                "pos_tol",
                "ori_tol_deg",
                "max_steps",
                "timeout_s",
            )
            unexpected = sorted(
                set(normalized) - set(base_keys + camera_keys + common_keys)
            )
            if unexpected:
                raise OfficialToolBoundaryError(
                    "unsupported arguments: "
                    + ", ".join(str(key) for key in unexpected)
                )
            supplied_base = [key for key in base_keys if _has_value(normalized, key)]
            supplied_camera = [
                key for key in camera_keys if _has_value(normalized, key)
            ]
            if supplied_base and supplied_camera:
                raise OfficialToolBoundaryError(
                    "head-frame translation parameter families are mutually "
                    "exclusive: use only x/y/z (robot base frame) or only "
                    "forward/leftward/upward (head camera frame); got base keys "
                    f"{supplied_base} and camera keys {supplied_camera}"
                )
            for key in base_keys + camera_keys:
                if _has_value(normalized, key):
                    normalized[key] = _finite_float(normalized, key)
                else:
                    normalized.pop(key, None)
        else:
            for key in ("forward", "upward", "leftward"):
                normalized[key] = _finite_float(normalized, key, 0.0)

        for key in ("roll", "pitch", "yaw"):
            normalized[key] = _finite_float(normalized, key, 0.0)
        pos_tol = _finite_float(normalized, "pos_tol", 0.012)
        ori_tol_deg = _finite_float(normalized, "ori_tol_deg", 3.0)
        max_steps_value = _finite_float(normalized, "max_steps", 240.0)
        timeout_s = _finite_float(normalized, "timeout_s", 60.0)
        if not 0.001 <= pos_tol <= 0.10:
            raise OfficialToolBoundaryError("pos_tol must be in 0.001..0.10m")
        if not 0.1 <= ori_tol_deg <= 45.0:
            raise OfficialToolBoundaryError("ori_tol_deg must be in 0.1..45 degrees")
        if not max_steps_value.is_integer() or not 1 <= max_steps_value <= 2000:
            raise OfficialToolBoundaryError("max_steps must be an integer in 1..2000")
        if not 0.1 <= timeout_s <= 600.0:
            raise OfficialToolBoundaryError("timeout_s must be in 0.1..600s")
        normalized["pos_tol"] = pos_tol
        normalized["ori_tol_deg"] = ori_tol_deg
        normalized["max_steps"] = int(max_steps_value)
        normalized["timeout_s"] = timeout_s

    if name == "move_to_reach_point":
        _require(normalized, "session_id", "image_id", "u", "v")
        for key in ("u", "v"):
            value = _finite_float(normalized, key)
            if value < 0.0 or value > 1000.0:
                raise OfficialToolBoundaryError(f"{key} must be in 0..1000")
        reach = _finite_float(normalized, "reach")
        if reach > 0.01 and not 0.35 <= reach <= 0.90:
            raise OfficialToolBoundaryError("reach must be 0/default or in 0.35..0.90m")
        nav_timeout_s = _finite_float(
            normalized,
            "nav_timeout_s",
            120.0,
        )
        if nav_timeout_s <= 0.0 or nav_timeout_s > 600.0:
            raise OfficialToolBoundaryError("nav_timeout_s must be in (0, 600]")
        keep_ori = str(normalized.get("keep_ori_arm", "none")).strip().lower()
        if keep_ori in ("", "false", "0", "no"):
            keep_ori = "none"
        if keep_ori not in ("none", "left", "right", "both"):
            raise OfficialToolBoundaryError(
                "keep_ori_arm must be none, left, right, or both"
            )
        normalized["reach"] = reach
        normalized["nav_timeout_s"] = nav_timeout_s
        normalized["keep_ori_arm"] = keep_ori

    if name == "measure_shoulder_distance":
        if str(normalized.get("object_name", "")).strip():
            raise OfficialToolBoundaryError(
                "official_v2 measure_shoulder_distance cannot resolve object_name "
                "or simulator AABB; use image_id + u + v"
            )
        _require(normalized, "session_id", "image_id", "u", "v")

    if name == "plan_grasp_point_filter_rgbd_lite":
        _require(normalized, "session_id", "image_id", "u", "v")
        for key in ("u", "v"):
            value = _finite_float(normalized, key)
            if value < 0.0 or value > 1000.0:
                raise OfficialToolBoundaryError(f"{key} must be in 0..1000")
        plan_arm = str(normalized.get("plan_arm", "any")).strip().lower()
        if plan_arm not in ("left", "right", "any"):
            raise OfficialToolBoundaryError("plan_arm must be left, right, or any")
        normalized["plan_arm"] = plan_arm
        normalized["seed"] = int(_finite_float(normalized, "seed", 42))

    if name == "exec_plan_pose":
        _require(normalized, "session_id", "plan_id")
        session_id = str(normalized["session_id"]).strip()
        plan_id = str(normalized["plan_id"]).strip()
        if not session_id or not plan_id:
            raise OfficialToolBoundaryError("session_id and plan_id must be non-empty")
        arm = str(normalized.get("arm", "")).strip().lower()
        if arm and arm not in ("left", "right"):
            raise OfficialToolBoundaryError("arm must be left or right")
        normalized["arm"] = arm
        if normalized.get("back_m") not in (None, ""):
            back_m = _finite_float(normalized, "back_m")
            if back_m != 0.0 and not 0.02 <= back_m <= 0.50:
                raise OfficialToolBoundaryError("back_m must be 0 or in 0.02..0.50m")
            normalized["back_m"] = back_m
        for key in ("stop_after_safe", "reset_tool_roll_at_start"):
            if key in normalized:
                value = normalized[key]
                normalized[key] = (
                    str(value).strip().lower() in ("1", "true", "yes", "on")
                    if isinstance(value, str)
                    else bool(value)
                )

    if name == "set_arm_to_grasp_position":
        _validate_arm(normalized, allow_both=True)
        gripper = str(normalized.get("gripper", "keep")).strip().lower()
        if gripper not in ("keep", "open"):
            raise OfficialToolBoundaryError("gripper must be keep or open")
        normalized["gripper"] = gripper
        keep_ori = str(normalized.get("keep_ori_arm", "none")).strip().lower()
        if keep_ori in ("", "false", "0", "no"):
            keep_ori = "none"
        if keep_ori != "none":
            raise OfficialToolBoundaryError(
                "official_v2 fixed joint-space grasp prep requires keep_ori_arm=none"
            )
        normalized["keep_ori_arm"] = keep_ori
        timeout_s = _finite_float(normalized, "timeout_s", 45.0)
        max_step = _finite_float(normalized, "max_dq_per_step", 0.30)
        tolerance = _finite_float(normalized, "tol", 0.08)
        if not 0.1 <= timeout_s <= 180.0:
            raise OfficialToolBoundaryError("timeout_s must be in 0.1..180")
        if not 0.0 < max_step <= 1.0:
            raise OfficialToolBoundaryError("max_dq_per_step must be in 0..1rad")
        if not 0.0 < tolerance <= 0.5:
            raise OfficialToolBoundaryError("tol must be in 0..0.5rad")
        normalized["timeout_s"] = timeout_s
        normalized["max_dq_per_step"] = max_step
        normalized["tol"] = tolerance

    if name == "reset_body":
        keep_ori = str(normalized.get("keep_ori_arm", "none")).strip().lower()
        if keep_ori not in ("", "none", "false", "0", "no"):
            raise OfficialToolBoundaryError(
                "official_v2 reset_body requires keep_ori_arm=none until "
                "policy-local arm kinematics are implemented"
            )
        pitch_deg = _finite_float(normalized, "pitch_deg")
        if abs(pitch_deg) > 60.0:
            raise OfficialToolBoundaryError("pitch_deg exceeds +/-60")
        normalized["keep_ori_arm"] = "none"

    return normalized


assert tuple(TOOL_CAPABILITIES) == PUBLIC_TOOLS
