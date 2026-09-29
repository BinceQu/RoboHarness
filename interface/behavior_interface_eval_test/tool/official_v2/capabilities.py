"""Evaluator-boundary audit for the public Interface v2 tools."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any

from .contract import (
    ACTION_DIM,
    ARM_DOF,
    MOVE_TRACKED_POINT_ORDER_DESCRIPTION,
    validate_cut_object_args,
    validate_move_chassis_to_directly_facing_surface_args,
    validate_navigate_to_args,
    validate_move_tracked_point_args,
    validate_move_point_to_point_args,
    validate_read_depth_args,
    validate_track_object_distance_args,
)


TOOL_VERSION = "official_v2"
OFFICIAL_ACTION_LABEL = f"official R1Pro action[{ACTION_DIM}]"
WRIST_ROLL_TEST_TOOL_ENABLED = (
    ARM_DOF == 8
    and os.environ.get(
        "BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL",
        "0",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)
ALLOWED_OBSERVATIONS = (
    "task_id",
    "need_new_action",
    "*::rgb",
    "*::depth_linear",
    "*::proprio",
    "*::cam_rel_poses",
)

PUBLIC_TOOLS = (
    "capture_head_camera",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "read_depth",
    "track_object_distance",
    "cut_object",
    "move_chassis_to_floor_point",
    "move_chassis_to_directly_facing_surface",
    "navigate_to",
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
    "adjust_plan_pose",
    "move_to_reach_point",
    "measure_shoulder_distance",
    "plan_grasp_point_filter",
    "plan_grasp_point_filter_rgbd",
    "plan_grasp_point_filter_rgbd_lite",
    "plan_press_point",
    "exec_plan_pose",
    "set_arm_to_grasp_position",
    "reset_body",
) + (("control_wrist_roll",) if WRIST_ROLL_TEST_TOOL_ENABLED else ())


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
            "poses, policy-owned state, and a submission-local static R1Pro model"
        ),
    }


TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "capture_head_camera": _implemented(
        "supported",
        (
            "persist evaluator head RGB/depth and camera-relative pose, then "
            "project the frozen v2 metric base-path HUD from policy-local pose"
        ),
        semantics=(
            "observation-equivalent capture with the frozen v2 base-path HUD"
        ),
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
            "the red mask does not use simulator instance segmentation or robot IDs; "
            "unlike the legacy overlay, it cannot semantically remove robot pixels "
            "that also satisfy the local opening-volume test"
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
            "the red mask does not use simulator instance segmentation or robot IDs; "
            "unlike the legacy overlay, it cannot semantically remove robot pixels "
            "that also satisfy the local opening-volume test"
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
        "supported",
        (
            "atomically mark multiple named points in the exact frozen head "
            "RGB-D capture identified by session_id and image_id, replay every "
            "subsequent evaluator observation in order, then update measured "
            "depths and current XYZ in robot base coord in policy memory"
        ),
        semantics=(
            "live named surface-point depth and XYZ in current robot base coord tracking "
            "in meters"
        ),
        constraints=(
            "names are model annotations rather than verified simulator object "
            "identities; every UV batch must carry the image_id of the displayed "
            "frozen capture, whose tracker RGB-D/camera geometry is SHA-256 "
            "verified; replay inputs are retained losslessly under explicit "
            "frame and byte budgets with independent per-session binding quotas; "
            "each successful call is a complete replacement of the previously "
            "registered named set (old names and rigid-pair/prior state are not "
            "merged into the new selection); a valid public reselection clears "
            "the old set before validation of the new capture and leaves the "
            "set empty if the new request fails; "
            "missing, modified, evicted, cross-session, cross-episode, or "
            "non-consecutive replay frames fail without falling back to the "
            "latest-frame UV; XYZ "
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
    "cut_object": _implemented(
        "conditional",
        (
            "bind cutting_tool_point and target_object_point to one exact frozen "
            "head RGB-D capture, track both points through every evaluator "
            "observation, and translate the nearest eligible held-tool EEF until "
            "their live robot-base XYZ measurements coincide"
        ),
        semantics=(
            "single cutting-tool touch attempt verified by live tracked RGB-D "
            "point coincidence"
        ),
        constraints=(
            "the cutting tool must already be held; arm selection uses only EEF "
            "proximity and proprioceptive gripper hold evidence, not grasp truth; "
            "EEF orientation and both gripper states are preserved; success means "
            "the two selected surface points are within pos_tol for consecutive "
            "fresh observations and does not assert simulator contact, a Cut BDDL "
            "predicate, or task success; tracking loss, stale frame binding, episode "
            "change, proprioceptive stall, timeout, cancellation, or nonzero J8 "
            "fails closed"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "no contact truth, grasp truth, object identity, segmentation, object pose, or simulator state is read",
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
    "move_chassis_to_directly_facing_surface": _implemented(
        "conditional",
        (
            "backproject exactly three points from one frozen evaluator head "
            "RGB-D capture, orient their triangle normal away from the camera "
            "view direction, translate the chassis base centre to the normal "
            "0.8m standoff projected onto the ground plane, then rotate in "
            "place to face opposite the normal's horizontal projection"
        ),
        semantics=(
            "camera-disambiguated three-point surface standoff followed by "
            "direct robot-frame XY translation and in-place yaw alignment"
        ),
        constraints=(
            "requires a nondegenerate triangle on a surface whose normal has "
            "a usable horizontal component; the frozen surface is assumed "
            "stationary while capture-time geometry is rebased through "
            "policy-local action odometry; no collision-free route is asserted"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "no scene geometry, object pose, segmentation, global base pose, simulator collision state, or simulator path planner is read",
        ),
    ),
    "navigate_to": _implemented(
        "conditional",
        (
            "resolve an existing named mark in the current policy-owned map, "
            "plan a footprint-inflated high-clearance polyline, and track its "
            "bounded segments through official base actions"
        ),
        semantics=(
            "map-backend-neutral named-place navigation; execute the signed polyline "
            "one segment at a time, align once when segment heading error exceeds "
            "45 degrees, and replan after SLAM cross-track error exceeds 0.30m"
        ),
        constraints=(
            "accepts named marks only, never arbitrary x/y; consumes a defensive "
            "navigation-map snapshot derived from evaluator observations; unknown "
            "space is blocked and walls are inflated by the signed robot-contract "
            "base footprint; arm posture and J8 do not gate chassis planning; no "
            "simulator pose, collision, scene, object, or path-planner truth"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
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
        "supported",
        (
            "match the observed fold branch, replay the validated two-phase "
            "1 cm R1Pro torso LUT to an exact cached target, and verify "
            "proprioceptive convergence"
        ),
        semantics="v2-compatible phase1/phase2 vertical torso adjustment",
        constraints=(
            "finite requests outside the static LUT workspace saturate at the "
            "nearest highest or lowest row; torso yaw remains locked at zero"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy runtime verification used simulator kinematics",
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
            "independently seek with 0.5N per finger, linearly ramp a single "
            "finger blocked for four post-command qpos observations to 2N "
            "without requiring prior travel, reset it on renewed motion, then "
            "synchronously ramp bilateral contact from 1N to 5N across the "
            "existing 12-frame assisted-grasp confirmation window"
        ),
        semantics="gripper-only action",
        constraints=(
            "the policy never creates a simulator constraint directly; the "
            "official evaluator still requires two-finger contact, a gripper "
            "ray hit, and its continuous 0.3s grasp window; constraint truth "
            "is not an observation"
        ),
        observations=("*::proprio", "*::depth"),
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
            "families. For a chassis-horizontal push, use short +x increments; "
            "camera forward follows the pitched viewing axis and can move the "
            "EEF vertically. Compute FK/Jacobian/IK from current proprio and the "
            f"submission URDF, then close the loop through official action[{ACTION_DIM}] targets"
        ),
        semantics="mutually exclusive robot-base XYZ or head-camera translation plus local-gripper RPY",
        constraints=(
            "omitted axes in the selected family are zero; y and leftward share "
            "the positive-left sign, but y is base-fixed while leftward is rotated "
            "from the head camera; uses joint limits and proprioceptive "
            "stall/tracking guards; no hidden contact state or simulator scene "
            "collision truth is available; success means the requested EEF pose "
            "converged, while external-object motion requires a subsequent RGB-D observation"
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
            "families. For a chassis-horizontal push, use short +x increments; "
            "camera forward follows the pitched viewing axis and can move the "
            "EEF vertically. Compute FK/Jacobian/IK from current proprio and the "
            f"submission URDF, then close the loop through official action[{ACTION_DIM}] targets"
        ),
        semantics="mutually exclusive robot-base XYZ or head-camera translation plus local-gripper RPY",
        constraints=(
            "omitted axes in the selected family are zero; y and leftward share "
            "the positive-left sign, but y is base-fixed while leftward is rotated "
            "from the head camera; uses joint limits and proprioceptive "
            "stall/tracking guards; no hidden contact state or simulator scene "
            "collision truth is available; success means the requested EEF pose "
            "converged, while external-object motion requires a subsequent RGB-D observation"
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
    "move_tracked_point": _implemented(
        "conditional",
        (
            "consume one to six point references, each role-labeled on-hand or "
            "frozen off-hand; ordinary names resolve through already-registered "
            "live RGB-D tracking, while left_finger_tip, right_finger_tip, and "
            "gripper_slide_center resolve directly from synchronized evaluator "
            "proprioception and submission-local FK with no tracker registration; "
            "the two fingertip names are the centers of the narrow physical front "
            "contact caps at max +Z_EEF on the exact red-overlay mesh, and follow "
            "the live finger-prismatic joints; "
            "optionally add one or more "
            "named groups of the eight typed quick presets and solve every group "
            "simultaneously; select the nearest submission-local EEF, and either replay a signed "
            f"action[{ACTION_DIM}] trajectory (default exec mode) or return a frozen-head red "
            "final-EEF overlay plus a signed exec_plan_pose-compatible plan without motion; "
            "the signed exec plan uses the same fixed-trunk arm-only current-to-safe and "
            "safe-to-final segment compiler as the RGB-D Lite grasp planner; "
            "plan mode batch-filters every endpoint frontier with the exact Lite "
            "gripper-overlap gates (original <=0.15cm3 and inflated <1.6cm3); "
            "when no IK endpoint passes, return the minimum-overlap endpoint from "
            "the precise candidates (otherwise the minimum-error candidates), "
            "with an explicit overlap-exceeded warning; "
            "target x/y/z accepts finite numbers, "
            "variable names such as x, y, za, arbitrary affine expressions such as "
            "z+0.08, 2*z, (a+b)/2, or a-b, and ? for an unconstrained coordinate; repeated "
            "variables impose the declared relation; strict affine variable "
            "inequalities such as a-b > 0 select an open half-space and are "
            "solved jointly with a one-millimetre numerical interior margin; typed plane/normal/vector "
            "relations are supported for on-hand points; all on-hand points must "
            "name the same rigid object; touch, flatwise, plane-parallel, collinear, "
            "line-vertical-to-plane, vertical-to-ground, faceto, and reverse-faceto presets are solved jointly with each other "
            "and with XYZ constraints; groups may share explicitly named points. "
            "Faceto/reverse-faceto require three ordered on-hand surface points; their projected image winding is respectively clockwise/counterclockwise and their outward normal is aligned with the frozen head-camera optical axis (or its opposite). "
            + MOVE_TRACKED_POINT_ORDER_DESCRIPTION + " Explicit segment_overlap, "
            "ordered_containment and line_only compatibility semantics remain available."
        ),
        semantics=(
            "nearest-reachable N-point rigid tracked pose planning or action-only execution with affine and typed geometric constraints"
        ),
        constraints=(
            "ordinary point names must already be active in track_object_distance; "
            "the three reserved EEF point names are on-hand-only and need no visual "
            "registration; all on-hand "
            "names must belong to one rigid object as a policy assertion; affine "
            "coordinate expressions, inequalities, and typed relations are parsed without execution; "
            "off-hand points are frozen observation "
            "references and are never sent to IK; success uses the maximum "
            "per-point/relation/inequality error from subsequent proprioception and live RGB-D tracking; "
            "transient active-arm, gripper, or on-hand tracking disturbances continue the "
            "immutable path and are recoverable, while terminal RGB-D identity, constraints, "
            "rigid geometry, safety state, and J8 remain strict; plan mode signs the exact "
            "planned endpoint and path for exec_plan_pose, renders only evaluator RGB-D, "
            "pins the live observed trunk throughout execution, and contains no trunk IK, "
            "trunk waypoint, or whole-body execution branch; "
            "and short-circuits inflated overlap after an original-overlap rejection without "
            "querying grasp-opening volume; "
            "minimum-overlap fallback preserves the constraint accuracy class and "
            "signed path checks, but does not claim that collision filters passed; "
            "online sparse execution is "
            "admitted only as a monotone, joint-corridor-checked subsequence of the signed path"
        ),
        observations=(
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ),
        legacy_violations=(
            "no simulator object identity, pose, mesh, contact, grasp, link pose, or IK is read",
        ),
    ),
    "plan_eef_translation_to_uvd_point": _blocked(
        (
            "reads simulator camera objects and world-frame EEF pose",
            "uses simulator-backed IK/FK for line-path validation",
            "may use simulator collision geometry",
        ),
        "unproject UVD from a frozen evaluator capture, solve the straight Cartesian "
        "path with local R1Pro IK, and collision-check only against RGB-D reconstruction",
    ),
    "adjust_plan_pose": _blocked(
        (
            "reads a live simulator camera pose",
            "calls simulator render for preview",
            "validates the changed pose with simulator-backed IK/world state",
        ),
        "store plans entirely in policy-owned records, apply camera-frame pose math, "
        "rerun local IK/RGB-D collision checks, and render previews from frozen RGB-D",
    ),
    "move_to_reach_point": _implemented(
        "conditional",
        (
            "unproject the clicked evaluator depth point, preserve the legacy "
            "pre-lift/XY/yaw/q3/fallback state machine, execute pre-lift with "
            "the same submission-local absolute torso LUT as adjust_height, "
            "and use submission-local torso and arm kinematics"
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
        "unproject a clicked evaluator depth point and measure from static R1Pro shoulders",
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
    "plan_grasp_point_filter": _blocked(
        (
            "binds a clicked segmentation instance to a simulator object",
            "reads target object mesh, pose, AABB, and scene collision geometry",
            "uses simulator-backed FK/IK and reachability filtering",
        ),
        "segment or cluster the target from evaluator RGB-D without simulator IDs, "
        "reconstruct target/scene geometry from depth, and use local R1Pro IK and meshes",
    ),
    "plan_grasp_point_filter_rgbd": _blocked(
        (
            "RGB-D reconstruction is legal, but the current planner still reads world.robot",
            "temporarily calls robot.set_joint_positions while probing FK",
            "uses simulator EEF links, robot USD data, and simulator-backed IK validation",
        ),
        "retain the RGB-D reconstruction and candidate sampling, but replace all robot/world "
        "queries and joint probing with a submission-local R1Pro FK/IK/collision model",
    ),
    "plan_grasp_point_filter_rgbd_lite": _implemented(
        "conditional",
        (
            "reconstruct the frozen evaluator RGB-D view, preserve the legacy "
            "1120-pose sampling and lite stage/ranking policy, and solve retained "
            "safe/final targets with submission-local cuRobo plus URDF FK; render "
            "the selected pose as a red, depth-occluded local gripper overlay"
        ),
        semantics="clicked RGB-D grasp planning with unchanged lite candidate policy",
        constraints=(
            "collision coverage is limited to geometry visible or completed from the "
            "frozen depth view; IK uses the submitted r1pro_8dof_hf250 URDF and "
            "capture-time proprioception; no simulator fallback is available"
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
        "trajectories with local R1Pro FK/IK and RGB-D collision geometry",
    ),
    "exec_plan_pose": _implemented(
        "conditional",
        (
            "load a policy-owned immutable joint trajectory produced by the RGB-D lite "
            "grasp planner or move_tracked_point plan mode, validate its digest, "
            "profile, episode, and motion epoch; use the planned arm by default or "
            "select another arm whose stored final_q reaches the same target under "
            "submission-local FK; resolve that arm's requested safe retreat with "
            "submission-local cuRobo/URDF, then execute only absolute official "
            "arm actions with proprioceptive tracking; grasp and move-tracked plans "
            "share the same current-to-safe then safe-to-final compiler and the live "
            "observed trunk remains pinned"
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
            "swept-volume self-collision and frozen-RGB-D validation remain incomplete; "
            "legacy plans containing trunk targets or trunk waypoints are rejected"
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
            f"feedback-govern observed {ARM_DOF}-DOF arm qpos toward a fixed "
            "R1Pro grasp-prep target, lock J8 at zero, and require stationary "
            "position convergence from evaluator qpos/qvel"
        ),
        semantics="legal fixed joint-space replacement",
        constraints=(
            "timeout_s is a soft deadline that may receive at most two bounded "
            "extensions only while evaluator proprioception shows progress; arm "
            "commands wait for tracking catch-up, carrying uses a smaller step, and "
            "failure/cancellation latches the latest measured pose instead of the "
            "unreached target. keep_ori_arm must be none because the public "
            "description's straight EEF-line IK and orientation-preserving path are "
            "not part of this fixed joint-space replacement"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy implementation used simulator IK and direct joint-set fallbacks",
            "legacy collision validation used simulator geometry",
        ),
    ),
    "reset_body": _implemented(
        "conditional",
        (
            "closed-loop trunk steps to the upright target through official actions; "
            "selected arms follow measured J1-J4 with best-effort local-FK position "
            "correction while J5-J7 preserve the evaluator-observed entry EEF "
            "orientation; arm tracking, IK, and pose quality are advisory and never "
            "slow, stop, or fail the independently controlled trunk reset"
        ),
        semantics="body reset with optional action-only arm software compliance",
        constraints=(
            "keep_ori_arm may be none, left, right, or both; this is software "
            "compliance over position targets because the official action contract "
            "does not expose impedance or torque mode; exact EEF position can be "
            "kinematically incompatible with the upright trunk target, so position is "
            "best-effort while orientation is verified from evaluator proprioception; "
            "5 deg and 25 deg are diagnostic thresholds only, and all keep-orientation "
            "initialization, observation, planning, and action failures degrade to "
            "warnings while the full trunk command continues"
        ),
        observations=("*::proprio",),
        legacy_violations=(
            "legacy reset directly set joint positions and robot pose",
            "legacy keep_ori_arm compensation used simulator kinematics",
        ),
    ),
}

if WRIST_ROLL_TEST_TOOL_ENABLED:
    TOOL_CAPABILITIES["control_wrist_roll"] = _implemented(
        "supported",
        (
            "test-only closed-loop control of the selected arm's independent J8 "
            "wrist-roll joint through official position actions"
        ),
        semantics=(
            "relative, absolute, or reset-to-zero J8 motion with a persistent "
            "post-motion position lock"
        ),
        constraints=(
            "available only when BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL=1 on "
            "the 8DOF test interface; J1-J7 and the unselected arm are fixed at "
            "their entry targets throughout the command"
        ),
        observations=("*::proprio",),
    )


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
            "The Interface receives evaluator observations only and returns one "
            f"official {ACTION_DIM}-D action. It has no simulator, scene, object, "
            "mesh, or BDDL handle."
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
        raise OfficialToolBoundaryError(
            f"{TOOL_VERSION} blocks {name}: {violations}"
        )

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

    if name == "cut_object":
        try:
            normalized = validate_cut_object_args(normalized)
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

    if name == "move_chassis_to_directly_facing_surface":
        try:
            normalized = validate_move_chassis_to_directly_facing_surface_args(
                normalized
            )
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc

    if name == "navigate_to":
        try:
            normalized = validate_navigate_to_args(normalized)
        except ValueError as exc:
            raise OfficialToolBoundaryError(str(exc)) from exc

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
        if isinstance(normalized.get("upward"), bool):
            raise OfficialToolBoundaryError("upward must be numeric")
        upward = _finite_float(normalized, "upward")
        # The submission-local torso LUT defines the physical workspace.  A
        # finite request outside it intentionally saturates at the nearest row.
        normalized["upward"] = upward

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
                raise OfficialToolBoundaryError(
                    "timeout_s must be in 0.3..10.0s"
                )
            normalized["timeout_s"] = timeout_s

    if name == "control_wrist_roll":
        unexpected = sorted(
            set(normalized) - {"arm", "mode", "angle_deg", "timeout_s"}
        )
        if unexpected:
            raise OfficialToolBoundaryError(
                "unsupported arguments: "
                + ", ".join(str(key) for key in unexpected)
            )
        if ARM_DOF != 8:
            raise OfficialToolBoundaryError(
                "control_wrist_roll requires the 8DOF R1Pro test profile"
            )
        normalized["arm"] = _validate_arm(normalized)
        mode = str(normalized.get("mode", "relative")).strip().lower()
        if mode not in ("relative", "absolute", "reset"):
            raise OfficialToolBoundaryError(
                "mode must be relative, absolute, or reset"
            )
        angle_deg = _finite_float(normalized, "angle_deg", 0.0)
        if abs(angle_deg) > 360.0:
            raise OfficialToolBoundaryError("angle_deg exceeds +/-360 degrees")
        timeout_s = _finite_float(normalized, "timeout_s", 15.0)
        if not 0.2 <= timeout_s <= 60.0:
            raise OfficialToolBoundaryError("timeout_s must be in 0.2..60.0s")
        normalized["mode"] = mode
        normalized["angle_deg"] = 0.0 if mode == "reset" else angle_deg
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
            raise OfficialToolBoundaryError(
                "ori_tol_deg must be in 0.1..45 degrees"
            )
        if not max_steps_value.is_integer() or not 1 <= max_steps_value <= 2000:
            raise OfficialToolBoundaryError(
                "max_steps must be an integer in 1..2000"
            )
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
            raise OfficialToolBoundaryError(
                "reach must be 0/default or in 0.35..0.90m"
            )
        nav_timeout_s = _finite_float(
            normalized,
            "nav_timeout_s",
            120.0,
        )
        if nav_timeout_s <= 0.0 or nav_timeout_s > 600.0:
            raise OfficialToolBoundaryError(
                "nav_timeout_s must be in (0, 600]"
            )
        keep_ori = str(
            normalized.get("keep_ori_arm", "none")
        ).strip().lower()
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
        plan_arm = str(
            normalized.get("plan_arm", "any")
        ).strip().lower()
        if plan_arm not in ("left", "right", "any"):
            raise OfficialToolBoundaryError(
                "plan_arm must be left, right, or any"
            )
        normalized["plan_arm"] = plan_arm
        normalized["seed"] = int(_finite_float(normalized, "seed", 42))

    if name == "exec_plan_pose":
        _require(normalized, "session_id", "plan_id")
        session_id = str(normalized["session_id"]).strip()
        plan_id = str(normalized["plan_id"]).strip()
        if not session_id or not plan_id:
            raise OfficialToolBoundaryError(
                "session_id and plan_id must be non-empty"
            )
        arm = str(normalized.get("arm", "")).strip().lower()
        if arm and arm not in ("left", "right"):
            raise OfficialToolBoundaryError("arm must be left or right")
        normalized["arm"] = arm
        if normalized.get("back_m") not in (None, ""):
            back_m = _finite_float(normalized, "back_m")
            if back_m != 0.0 and not 0.02 <= back_m <= 0.50:
                raise OfficialToolBoundaryError(
                    "back_m must be 0 or in 0.02..0.50m"
                )
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
                "official_v2 fixed joint-space grasp prep requires "
                "keep_ori_arm=none"
            )
        normalized["keep_ori_arm"] = keep_ori
        timeout_s = _finite_float(normalized, "timeout_s", 15.0)
        max_step = _finite_float(normalized, "max_dq_per_step", 0.30)
        tolerance = _finite_float(normalized, "tol", 0.08)
        if not 0.1 <= timeout_s <= 180.0:
            raise OfficialToolBoundaryError("timeout_s must be in 0.1..180")
        if not 0.0 < max_step <= 1.0:
            raise OfficialToolBoundaryError(
                "max_dq_per_step must be in 0..1rad"
            )
        if not 0.0 < tolerance <= 0.5:
            raise OfficialToolBoundaryError("tol must be in 0..0.5rad")
        normalized["timeout_s"] = timeout_s
        normalized["max_dq_per_step"] = max_step
        normalized["tol"] = tolerance

    if name == "reset_body":
        keep_ori = str(normalized.get("keep_ori_arm", "none")).strip().lower()
        if keep_ori in ("", "false", "0", "no"):
            keep_ori = "none"
        if keep_ori not in ("none", "left", "right", "both"):
            raise OfficialToolBoundaryError(
                "keep_ori_arm must be none, left, right, or both"
            )
        pitch_deg = _finite_float(normalized, "pitch_deg")
        if abs(pitch_deg) > 60.0:
            raise OfficialToolBoundaryError("pitch_deg exceeds +/-60")
        normalized["keep_ori_arm"] = keep_ori

    return normalized


assert tuple(TOOL_CAPABILITIES) == PUBLIC_TOOLS
