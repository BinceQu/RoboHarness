"""move_tracked_point

Plan or move one to six point references. Ordinary names must be active in track_object_distance. The on-hand names left_finger_tip, right_finger_tip, and gripper_slide_center instead use the selected gripper's live evaluator proprioception and submission-local FK and do not need tracking. Each finger-tip name is the center of that finger's narrow physical front contact cap at max +Z_EEF on the exact red-overlay mesh, and moves with the observed finger-prismatic joint. execution_mode defaults to exec; plan returns the frozen-head red final-EEF overlay and a signed arm-only plan_id executable by exec_plan_pose without moving. It uses the same current-to-safe then safe-to-final compiler as an RGB-D Lite grasp plan and pins the live trunk throughout execution. Mark each point on-hand or frozen off-hand, optionally select touch, flatwise, plane-parallel, collinear, line-vertical-to-plane, vertical-to-ground, faceto, or reverse-faceto, and optionally enter target x/y/z as a finite number, a variable identifier, or an affine expression such as z+0.08, 2*z, (a+b)/2, or a-b. Reusing a variable imposes that relation across any point and axis; an identifier used once leaves that degree of freedom for the nearest reachable solution. Optional strict inequalities compare a variable or affine function with a finite constant using > or <; for example a-b > 0 selects the signed solution with a above b. Inequalities are solved jointly and rechecked against the final live tracked points. All on-hand points must belong to the same rigid object. Quick geometry and XYZ constraints are solved jointly. Four-point collinear uses the submitted row order as the directed axial order (for example 1,3,4,2). Vertical-to-ground uses two on-hand points for equal X/Y, or three on-hand points for a plane whose unit normal has Z=0, with no fixed heading. Faceto and reverse-faceto require exactly three ordered on-hand surface points and no off-hand points. The displayed row order 1,2,3 is preserved: faceto requires those points to wind clockwise in the frozen head-camera image, while reverse-faceto requires counterclockwise winding. Their outward triangle normal is aligned as closely as possible to the frozen head-camera optical-forward direction (or its opposite for reverse-faceto); the reported angle is the acute angle. Optional typed common_plane, oriented_plane_normal, and align_vector relations are checked by the local solver. The nearest EEF is selected from live local FK, and success requires subsequent proprioception and live tracked-point convergence. Collinear point rows specify spatial axial order, not execution order. Preserve each group's exact row order, including interleaved on/off-hand points; 1,3,2 means point 3 lies between 1 and 2. The canonical reading direction is robot-base left-to-right (-Y) for a Y-dominant line, top-to-bottom (-Z) for a Z-dominant line, and near-to-far (+X) for an X-dominant line; ties prefer Y, then Z, then X. It does not depend on camera viewpoint. This also applies when numeric/affine XYZ constraints imply three or more collinear points. Reversing rows requests the opposite physical order. Order violations cannot be returned as successful approximate or overlap-fallback plans.

How this tool works:
Read named on-hand and scene points, build the requested geometric constraints, solve a hand pose and joint trajectory, then execute with feedback. Check tracked-point geometry after movement; stale/missing observations invalidate the plan.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace as dataclass_replace
from tools.util.action_constants import ADJUST_EEF_STALL_STEPS
from tools.util.action_constants import DEFAULT_START_ARM_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_START_GRIPPER_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_START_TRUNK_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_TRACKING_STALL_STEPS
from tools.util.action_constants import DEFAULT_TRAJECTORY_STEP_RAD
from tools.util.action_constants import DEFAULT_WAYPOINT_MAX_TRACKING_ERROR_RAD
from tools.util.action_constants import DEFAULT_WAYPOINT_SETTLE_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_WAYPOINT_TIMEOUT_S
from tools.util.action_constants import EXEC_FINAL_PLAN_ORI_TOL_DEG
from tools.util.action_constants import EXEC_FINAL_PLAN_POS_TOL_M
from tools.util.action_constants import EXEC_SAFE_FINAL_BRANCH_GAP_RAD
from tools.util.action_constants import EXEC_SAFE_ORI_TOL_DEG
from tools.util.action_constants import EXEC_SAFE_POS_TOL_M
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA_VERSION
from tools.util.action_constants import LOCKED_J8_CAPTURE_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_INDEX
from tools.util.action_constants import LOCKED_J8_SERIALIZED_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_VALUE_RAD
from tools.util.action_constants import MAX_PLAN_RECORD_BYTES
from tools.util.action_constants import MAX_TRAJECTORY_WAYPOINTS
from tools.util.action_constants import MOVE_POINT_TO_POINT_STALL_POSITION_DELTA_RAD
from tools.util.action_constants import MOVE_POINT_TO_POINT_STALL_QVEL_RAD_S
from tools.util.action_constants import MOVE_POINT_TO_POINT_START_ARM_TOLERANCE_RAD
from tools.util.action_constants import MOVE_POINT_TO_POINT_START_GRIPPER_TOLERANCE_RAD
from tools.util.action_constants import MOVE_TRACKED_OVERLAP_FALLBACK_POLICY
from tools.util.action_constants import MOVE_TRACKED_POINT_GRIPPER_OPEN_HOLD_THRESHOLD_M
from tools.util.action_constants import MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N
from tools.util.action_constants import MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
from tools.util.action_constants import TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
from tools.util.action_constants import TRAJECTORY_STEP_VALIDATION_EPS_RAD
from tools.util.action_constants import _ARM_LIMITS
from tools.util.camera_frames import _capture_robot_state
from tools.util.geometry.capabilities import TOOL_VERSION
from tools.util.geometry.contract import ACTION_DIM
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.contract import MOVE_TRACKED_POINT_MAX_POINTS
from tools.util.geometry.contract import PROPRIO_SLICES
from tools.util.geometry.contract import ROBOT_PROFILE
from tools.util.geometry.contract import validate_move_tracked_point_args
from tools.util.geometry.dynamic_point_tracker import transform_from_pose
from tools.util.geometry.eef_adjustment_local import local_robot_state
from tools.util.geometry.eef_adjustment_local import locked_arm_vector
from tools.util.geometry.eef_adjustment_local import orientation_error_deg
from tools.util.geometry.eef_adjustment_local import solve_pose_target
from tools.util.geometry.grasp_kinematics_local import LocalRobotState
from tools.util.geometry.grasp_kinematics_local import arm_joint_limits as _local_arm_joint_limits
from tools.util.geometry.grasp_kinematics_local import eef_pose as _local_eef_pose
from tools.util.geometry.predefined_eef_points_local import is_predefined_eef_point_name
from tools.util.geometry.predefined_eef_points_local import partition_predefined_eef_point_names
from tools.util.geometry.predefined_eef_points_local import predefined_eef_point_local_m
from tools.util.geometry.predefined_eef_points_local import predefined_eef_points_robot_base_m
from tools.util.geometry.rgbd_grasp_lite import LITE_POSE_OVERLAP_FILTER_BUILD
from tools.util.geometry.rgbd_grasp_planner import INFLATED_OVERLAP_MAX_CM3
from tools.util.geometry.rgbd_grasp_planner import ORIGINAL_OVERLAP_MAX_CM3
from tools.util.geometry.tracked_object_distance import TrackObjectDistanceBindingError
from tools.util.geometry.tracked_object_distance import head_camera_intrinsics
from tools.util.geometry.tracked_point_constraints_local import evaluate_quick_constraint as _evaluate_quick_constraint
from tools.util.geometry.tracked_point_constraints_local import geometry_report_difference as _geometry_report_difference
from tools.util.geometry.tracked_point_constraints_local import quick_constraint_relation as _quick_constraint_relation
from tools.util.geometry.tracked_point_constraints_local import rigid_geometry_report as _tracked_rigid_geometry_report
from tools.util.geometry.tracked_point_execution_local import schedule_execution_waypoints as _schedule_tracked_execution_waypoints
from tools.util.geometry.tracked_point_motion_local import PlanningDeadlineExceeded
from tools.util.geometry.tracked_point_motion_local import evaluate_rigid_pair_consistency as _tracked_rigid_pair_consistency
from tools.util.geometry.tracked_point_motion_local import evaluate_target_constraints as _tracked_target_constraints
from tools.util.geometry.tracked_point_motion_local import rigid_pair_distance_tolerance as _tracked_rigid_pair_tolerance
from tools.util.geometry.visualization_local import render_plan_gripper_overlay
from tools.util.motion_common import _FrozenCapture
from tools.util.motion_common import _adjust_action
from tools.util.motion_common import _adjust_adapter
from tools.util.motion_common import _adjust_camera_pose
from tools.util.motion_common import _adjust_kinematic_observation
from tools.util.motion_common import _adjust_observation_advanced
from tools.util.motion_common import _adjust_safety_after_action
from tools.util.motion_common import _adjust_safety_monitor
from tools.util.motion_common import _adjust_safety_report
from tools.util.motion_common import _arm_vector
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _canonical_json_bytes
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _compile_exec_arm_segments
from tools.util.motion_common import _ensure_session
from tools.util.motion_common import _exec_safe_back_m_candidates
from tools.util.motion_common import _file_to_data_url
from tools.util.motion_common import _image_path
from tools.util.motion_common import _images_dir
from tools.util.motion_common import _json_ready
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _load_image_meta
from tools.util.motion_common import _load_plan_trajectory
from tools.util.motion_common import _move_point_is_cancellation
from tools.util.motion_common import _move_tracked_camera_pose_payload
from tools.util.motion_common import _move_tracked_plan_frozen
from tools.util.motion_common import _move_tracked_quick_constraint_specs
from tools.util.motion_common import _move_tracked_quick_group_points
from tools.util.motion_common import _move_tracked_strict_point_matrix
from tools.util.motion_common import _next_image_id
from tools.util.motion_common import _next_plan_id
from tools.util.motion_common import _normalize_arm_q_to_joint_limits
from tools.util.motion_common import _normalize_trunk_q_to_joint_limits
from tools.util.motion_common import _plan_record_path
from tools.util.motion_common import _plan_start_local_state
from tools.util.motion_common import _policy_owned_capture_artifact
from tools.util.motion_common import _quat_to_mat_xyzw
from tools.util.motion_common import _reject_duplicate_json_keys
from tools.util.motion_common import _require_episode_initialized
from tools.util.motion_common import _require_serialized_j8_zero
from tools.util.motion_common import _save_image_meta
from tools.util.motion_common import _set_result
from tools.util.motion_common import _trajectory_digest
from tools.util.motion_common import _validate_move_tracked_endpoint_order
from tools.util.motion_common import _validated_session_id
from tools.util.motion_common import _write_plan_record
from tools.util.trajectory_execution import _move_point_gripper_action_overrides
from typing import Any
from typing import Mapping
from typing import Sequence
import concurrent.futures
import cv2
import hashlib
import hmac
import json
import math
import numpy as np
import os
import secrets
import threading
import time
from tools.util.tool import main

NAME = 'move_tracked_point'
SCHEMA = {'type': 'object',
 'properties': {'execution_mode': {'type': 'string',
                                   'enum': ['exec', 'plan'],
                                   'default': 'exec',
                                   'description': 'exec moves immediately; plan returns a red '
                                                  'final-EEF overlay and an arm-only plan_id for '
                                                  'exec_plan_pose without moving; execution uses '
                                                  'current-to-safe then safe-to-final with the '
                                                  'live trunk pinned'},
                'points': {'type': 'array',
                           'minItems': 1,
                           'maxItems': 6,
                           'items': {'additionalProperties': False,
                                     'properties': {'name': {'description': 'An active tracked '
                                                                            'name, or on-hand '
                                                                            'predefined EEF point '
                                                                            'left_finger_tip, '
                                                                            'right_finger_tip, or '
                                                                            'gripper_slide_center '
                                                                            '(no tracking '
                                                                            'required). The '
                                                                            'finger-tip names '
                                                                            'denote the narrow '
                                                                            'physical front '
                                                                            'contact-cap centers '
                                                                            'at max +Z_EEF on the '
                                                                            'exact red '
                                                                            'plan-overlay mesh.',
                                                             'type': 'string'},
                                                    'role': {'enum': ['on_hand', 'off_hand'],
                                                             'type': 'string'},
                                                    'target_xyz_m': {'oneOf': [{'additionalProperties': False,
                                                                                'properties': {'x': {'oneOf': [{'type': 'number'},
                                                                                                               {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                'type': 'string'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'},
                                                                                                                               'free': {'const': True},
                                                                                                                               'var': {'pattern': '^[A-Za-z][A-Za-z0-9_]*$',
                                                                                                                                       'type': 'string'}},
                                                                                                                'required': ['var'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'}},
                                                                                                                'required': ['expr'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'free': {'const': True}},
                                                                                                                'required': ['free'],
                                                                                                                'type': 'object'}]},
                                                                                               'y': {'oneOf': [{'type': 'number'},
                                                                                                               {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                'type': 'string'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'},
                                                                                                                               'free': {'const': True},
                                                                                                                               'var': {'pattern': '^[A-Za-z][A-Za-z0-9_]*$',
                                                                                                                                       'type': 'string'}},
                                                                                                                'required': ['var'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'}},
                                                                                                                'required': ['expr'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'free': {'const': True}},
                                                                                                                'required': ['free'],
                                                                                                                'type': 'object'}]},
                                                                                               'z': {'oneOf': [{'type': 'number'},
                                                                                                               {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                'type': 'string'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'},
                                                                                                                               'free': {'const': True},
                                                                                                                               'var': {'pattern': '^[A-Za-z][A-Za-z0-9_]*$',
                                                                                                                                       'type': 'string'}},
                                                                                                                'required': ['var'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                                        'type': 'string'}},
                                                                                                                'required': ['expr'],
                                                                                                                'type': 'object'},
                                                                                                               {'additionalProperties': False,
                                                                                                                'properties': {'free': {'const': True}},
                                                                                                                'required': ['free'],
                                                                                                                'type': 'object'}]}},
                                                                                'required': ['x',
                                                                                             'y',
                                                                                             'z'],
                                                                                'type': 'object'},
                                                                               {'items': {'oneOf': [{'type': 'number'},
                                                                                                    {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                     'type': 'string'},
                                                                                                    {'additionalProperties': False,
                                                                                                     'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                             'type': 'string'},
                                                                                                                    'free': {'const': True},
                                                                                                                    'var': {'pattern': '^[A-Za-z][A-Za-z0-9_]*$',
                                                                                                                            'type': 'string'}},
                                                                                                     'required': ['var'],
                                                                                                     'type': 'object'},
                                                                                                    {'additionalProperties': False,
                                                                                                     'properties': {'expr': {'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                                                                             'type': 'string'}},
                                                                                                     'required': ['expr'],
                                                                                                     'type': 'object'},
                                                                                                    {'additionalProperties': False,
                                                                                                     'properties': {'free': {'const': True}},
                                                                                                     'required': ['free'],
                                                                                                     'type': 'object'}]},
                                                                                'maxItems': 3,
                                                                                'minItems': 3,
                                                                                'type': 'array'}]}},
                                     'required': ['name'],
                                     'type': 'object'},
                           'description': 'Each points[] item allows only name, role, and '
                                          'target_xyz_m. Put x/y/z inside target_xyz_m; do not put '
                                          'x/y/z on the point root.'},
                'relations': {'type': 'array',
                              'default': [],
                              'items': {'properties': {'direction_robot_base': {'items': {'type': 'number'},
                                                                                'type': 'array'},
                                                       'mode': {'enum': ['same',
                                                                         'opposite',
                                                                         'parallel'],
                                                                'type': 'string'},
                                                       'normal_robot_base': {'items': {'type': 'number'},
                                                                             'type': 'array'},
                                                       'offset_m': {},
                                                       'point_names': {'items': {'type': 'string'},
                                                                       'type': 'array'},
                                                       'type': {'enum': ['common_plane',
                                                                         'oriented_plane_normal',
                                                                         'align_vector',
                                                                         'vertical_to_ground'],
                                                                'type': 'string'}},
                                        'required': ['type', 'point_names'],
                                        'type': 'object'}},
                'inequalities': {'type': 'array',
                                 'default': [],
                                 'items': {'additionalProperties': False,
                                           'properties': {'lhs': {'description': 'Variable or '
                                                                                 'affine function, '
                                                                                 'for example a-b, '
                                                                                 '2*a-b, or '
                                                                                 '(a+b)/2',
                                                                  'pattern': '^(?:\\?|[A-Za-z0-9_().+*/\\-\\s]{1,256})$',
                                                                  'type': 'string'},
                                                          'op': {'enum': ['>', '<'],
                                                                 'type': 'string'},
                                                          'rhs': {'description': 'Finite constant '
                                                                                 'in metres.',
                                                                  'type': 'number'}},
                                           'required': ['lhs', 'op', 'rhs'],
                                           'type': 'object'},
                                 'description': 'Optional strict affine variable inequalities '
                                                'solved jointly with every XYZ and geometry '
                                                'constraint. The left side uses the same safe '
                                                'affine grammar as target coordinates, the '
                                                'operator is > or <, and the right side is a '
                                                'finite constant. For example, a-b > 0 selects the '
                                                'solution where a is above b.'},
                'quick_constraint': {'type': 'object',
                                     'description': 'Optional role-based geometry preset solved '
                                                    'jointly with all supplied XYZ and typed '
                                                    'relations'},
                'quick_constraints': {'type': 'array',
                                      'items': {'properties': {'axial_mode': {'enum': ['ordered_containment',
                                                                                       'segment_overlap',
                                                                                       'line_only'],
                                                                              'type': 'string'},
                                                               'axial_point_order': {'items': {'type': 'string'},
                                                                                     'type': 'array'},
                                                               'off_hand_points': {'items': {'type': 'string'},
                                                                                   'type': 'array'},
                                                               'on_hand_points': {'items': {'type': 'string'},
                                                                                  'type': 'array'},
                                                               'type': {'enum': ['touch',
                                                                                 'flatwise',
                                                                                 'plane_parallel',
                                                                                 'collinear',
                                                                                 'line_vertical_to_plane',
                                                                                 'vertical_to_ground',
                                                                                 'faceto',
                                                                                 'reverse_faceto'],
                                                                        'type': 'string'}},
                                                'required': ['type',
                                                             'on_hand_points',
                                                             'off_hand_points'],
                                                'type': 'object'},
                                      'description': 'Optional named preset groups solved '
                                                     'simultaneously with all supplied XYZ and '
                                                     'typed relations'},
                'pos_tol': {'type': 'number', 'default': 0.03, 'description': 'm'},
                'ori_tol_deg': {'type': 'number', 'default': 20.0, 'description': 'deg'},
                'max_steps': {'type': 'integer', 'default': 360},
                'timeout_s': {'type': 'number', 'default': 90.0, 'description': 's'}},
 'additionalProperties': False,
 'required': ['points']}


def actions(
    robot,
    points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    pos_tol: float = 0.03,
    ori_tol_deg: float = 20.0,
    max_steps: int = 360,
    timeout_s: float = 90.0,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    execution_mode: str = "exec",
    inequalities: Sequence[Mapping[str, Any]] | None = None,
):
    """Move or plan tracked/predefined points with spatial collinear order.

    The on-hand names ``left_finger_tip``, ``right_finger_tip``, and
    ``gripper_slide_center`` bind directly to the selected gripper geometry
    using synchronized evaluator proprioception and local FK. They must not be
    registered with ``track_object_distance``.

    Collinear rows are the required axial sequence: 1,3,2 places 3 between 1
    and 2. This includes coordinate-implied lines and interleaved on/off-hand
    rows. Robot-base reading direction is left-to-right (-Y), top-to-bottom
    (-Z), or near-to-far (+X), using the largest axis component (ties Y,Z,X).
    Camera viewpoint does not change the order. Reversed or shuffled endpoints
    cannot be returned as successful approximate/overlap-fallback plans.

    The ``faceto`` and ``reverse_faceto`` camera pose is frozen from the same
    evaluator observation as the tracked points and is reused for planning,
    signed-plan reload, and live validation.

    ``inequalities`` contains strict affine variable constraints. Each item is
    ``{"lhs": <variable-or-affine-expression>, "op": ">"|"<", "rhs":
    <finite-number>}``; for example ``a-b > 0`` selects the signed vertical
    ordering when two point rows use Z variables ``a`` and ``b``.

    Plan mode compiles exactly the same fixed-trunk, arm-only
    ``current -> safe -> final`` joint segments as the RGB-D Lite grasp
    planner. ``exec_plan_pose`` pins the observed trunk and never creates or
    replays a move-tracked trunk target.
    """

    lease_state: dict[str, Any] = {}
    motion_retention_lease_state: dict[str, Any] = {}
    implementation = _move_tracked_point_impl(
        robot,
        points,
        relations=relations,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
        mode=mode,
        execution_mode=execution_mode,
        inequalities=inequalities,
        _eef_prior_lease_state=lease_state,
        _motion_retention_lease_state=motion_retention_lease_state,
    )
    try:
        while True:
            try:
                yield next(implementation)
            except StopIteration:
                break
    finally:
        implementation.close()
        manager = motion_retention_lease_state.get("_manager") or getattr(
            robot, "_official_tracked_object_distances", None
        )
        _move_tracked_release_motion_retention(
            manager,
            motion_retention_lease_state,
        )
        lease_id = str(lease_state.get("lease_id") or "").strip()
        release = getattr(
            manager,
            "deactivate_on_hand_eef_observation_prior",
            None,
        )
        if lease_id and callable(release):
            try:
                release(lease_id)
            except Exception:
                pass


MOVE_TRACKED_PLAN_INLINE_DIAGNOSTIC_BYTES = 16 * 1024



_OFFICIAL_RELOAD_GENERATION = int(
    globals().get("_OFFICIAL_RELOAD_GENERATION", 0)
)



_OFFICIAL_STACK_DIGEST = globals().get("_OFFICIAL_STACK_DIGEST")



EXEC_SAFE_BACK_DEFAULT_M = 0.10



def _freeze_adapter_camera_array(
    adapter: Any,
    value: Any,
    *,
    dtype: Any | None = None,
    squeeze: bool = False,
) -> np.ndarray:
    """Retain official adapter-owned camera arrays without a second copy.

    The production adapter replaces its complete observation snapshot instead
    of mutating arrays in place. Unknown/test adapters retain the defensive
    copy used by the historical capture path.
    """

    array = np.asarray(value, dtype=dtype)
    if squeeze:
        array = array.squeeze()
    if bool(getattr(adapter, "_official_camera_arrays_owned", False)):
        return array
    return array.copy()



MOVE_POINT_TO_POINT_BASE_TRANSLATION_LIMIT_M = 0.005



MOVE_POINT_TO_POINT_BASE_YAW_LIMIT_RAD = 0.005



MOVE_POINT_TO_POINT_START_TRUNK_TOLERANCE_RAD = 0.02



MOVE_POINT_TO_POINT_START_ARM_QVEL_RAD_S = 0.05



MOVE_POINT_TO_POINT_START_TRUNK_QVEL_RAD_S = 0.05



MOVE_POINT_TO_POINT_START_BASE_QVEL = 0.01



MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S = 0.03



MOVE_POINT_TO_POINT_STABLE_STEPS = 2



MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD = 0.012



MOVE_POINT_TO_POINT_WAYPOINT_TOLERANCE_RAD = 0.018



MOVE_POINT_TO_POINT_FINAL_LIMIT_PROXIMITY_RAD = 0.005



MOVE_POINT_TO_POINT_FINAL_LIMIT_POSITION_STEP_RAD = 0.002



def _move_point_capture_kinematics(
    capture: _FrozenCapture,
) -> dict[str, Any]:
    robot = dict(capture.robot or {})
    trunk = np.asarray(robot.get("trunk_qpos"), dtype=np.float64).reshape(-1)
    if trunk.size != 4 or not np.all(np.isfinite(trunk)):
        raise ValueError("capture trunk_qpos must contain four finite values")
    arm_q: dict[str, np.ndarray] = {}
    arm_qvel: dict[str, np.ndarray] = {}
    gripper_q: dict[str, np.ndarray] = {}
    gripper_qvel: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        raw_q = np.asarray(
            robot.get(f"arm_{side}_qpos"),
            dtype=np.float64,
        ).reshape(-1)
        if raw_q.size != ARM_DOF or not np.all(np.isfinite(raw_q)):
            raise ValueError(
                f"capture {side} arm_qpos must contain {ARM_DOF} finite values"
            )
        if (
            ARM_DOF == 8
            and abs(float(raw_q[7])) > LOCKED_J8_CAPTURE_TOLERANCE_RAD
        ):
            raise ValueError(
                f"capture {side} J8={raw_q[7]:.6f}rad is outside the lock tolerance"
            )
        arm_q[side] = locked_arm_vector(
            raw_q,
            label=f"capture {side} arm_qpos",
        )
        raw_qvel = np.asarray(
            robot.get(f"arm_{side}_qvel"),
            dtype=np.float64,
        ).reshape(-1)
        if raw_qvel.size != ARM_DOF or not np.all(np.isfinite(raw_qvel)):
            raise ValueError(
                f"capture {side} arm_qvel must contain {ARM_DOF} finite values"
            )
        arm_qvel[side] = raw_qvel.copy()
        gripper_q[side] = np.asarray(
            robot.get(f"gripper_{side}_qpos"),
            dtype=np.float64,
        ).reshape(-1)
        gripper_qvel[side] = np.asarray(
            robot.get(f"gripper_{side}_qvel"),
            dtype=np.float64,
        ).reshape(-1)
    state = local_robot_state(
        trunk_q=trunk,
        arm_left_q=arm_q["left"],
        arm_right_q=arm_q["right"],
        gripper_left_q=gripper_q["left"],
        gripper_right_q=gripper_q["right"],
    )
    eef: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        position, quaternion = _local_eef_pose(state, side, arm_q[side])
        eef[side] = {
            "position": np.asarray(position, dtype=np.float64).reshape(3),
            "quaternion": np.asarray(quaternion, dtype=np.float64).reshape(4),
        }
    motion_epoch = robot.get("motion_epoch")
    if not isinstance(motion_epoch, int) or motion_epoch < 0:
        raise ValueError("capture has no valid policy motion_epoch")
    episode_id = str(robot.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("capture has no valid evaluator episode_id")
    return {
        "state": state,
        "trunk": trunk,
        "arm_q": arm_q,
        "arm_qvel": arm_qvel,
        "gripper_q": gripper_q,
        "gripper_qvel": gripper_qvel,
        "eef": eef,
        "motion_epoch": int(motion_epoch),
        "episode_id": episode_id,
    }



def _move_point_start_state_validation(
    ctx,
    capture_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if str(ctx.world.episode_id()) != str(capture_state["episode_id"]):
        return False, {"reason": "episode_changed"}
    if int(ctx.world.motion_epoch()) != int(capture_state["motion_epoch"]):
        return False, {"reason": "motion_epoch_changed"}
    adapter, sequence = _adjust_adapter(ctx)

    def invalid_observation(reason: str, error: str) -> tuple[bool, dict[str, Any]]:
        return False, {
            "reason": str(reason),
            "error": str(error),
            "observation_sequence": int(sequence),
            "adapter": adapter,
        }

    current = {
        "trunk": np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(-1),
        "trunk_qvel": np.asarray(
            adapter.proprio_vector()[PROPRIO_SLICES["trunk_qvel"]],
            dtype=np.float64,
        ).reshape(-1),
        "base_qvel": np.asarray(ctx.world.base_qvel(), dtype=np.float64).reshape(-1),
    }
    if current["trunk"].size != 4 or not np.all(np.isfinite(current["trunk"])):
        return invalid_observation(
            "invalid_start_state",
            "evaluator trunk proprioception is invalid",
        )
    if current["trunk_qvel"].size != 4 or not np.all(
        np.isfinite(current["trunk_qvel"])
    ):
        return invalid_observation(
            "invalid_start_state",
            "evaluator trunk velocity proprioception is invalid",
        )
    if current["base_qvel"].size < 3 or not np.all(
        np.isfinite(current["base_qvel"][:3])
    ):
        return invalid_observation(
            "invalid_start_state",
            "evaluator base velocity proprioception is invalid",
        )
    errors: dict[str, float] = {
        "trunk_inf_rad": float(
            np.linalg.norm(current["trunk"] - capture_state["trunk"], ord=np.inf)
        ),
        "trunk_qvel_inf_rad_s": float(
            np.linalg.norm(current["trunk_qvel"], ord=np.inf)
        ),
        "base_qvel_inf": float(
            np.linalg.norm(current["base_qvel"][:3], ord=np.inf)
        ),
    }
    for side in ("left", "right"):
        raw_q = np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64)
        if raw_q.size != ARM_DOF or not np.all(np.isfinite(raw_q)):
            return invalid_observation(
                "invalid_start_state",
                f"evaluator {side} arm position proprioception is invalid",
            )
        if (
            ARM_DOF == 8
            and raw_q.size >= 8
            and abs(float(raw_q[7])) > LOCKED_J8_CAPTURE_TOLERANCE_RAD
        ):
            return invalid_observation(
                f"{side}_j8_not_locked",
                f"evaluator {side} J8 is outside the locked-zero tolerance",
            )
        arm_q = locked_arm_vector(raw_q, label=f"current {side} arm_qpos")
        arm_qvel = np.asarray(ctx.world.arm_qvel_list(side), dtype=np.float64)
        gripper_q = np.asarray(ctx.world.gripper_qpos_list(side), dtype=np.float64)
        if arm_qvel.size != ARM_DOF or not np.all(np.isfinite(arm_qvel)):
            return invalid_observation(
                "invalid_start_state",
                f"evaluator {side} arm velocity proprioception is invalid",
            )
        if gripper_q.size != 2 or not np.all(np.isfinite(gripper_q)):
            return invalid_observation(
                "invalid_start_state",
                f"evaluator {side} gripper position proprioception is invalid",
            )
        # J8 is a contract-locked joint.  Its observed velocity can contain
        # residual controller telemetry that cannot be changed by an official
        # action, so it must not make the controllable J1-J7 start gate fail.
        arm_qvel_for_motion = (
            arm_qvel[:7] if ARM_DOF == 8 else arm_qvel
        )
        errors[f"{side}_arm_inf_rad"] = float(
            np.linalg.norm(arm_q - capture_state["arm_q"][side], ord=np.inf)
        )
        errors[f"{side}_arm_qvel_inf_rad_s"] = float(
            np.linalg.norm(arm_qvel_for_motion, ord=np.inf)
        )
        if ARM_DOF == 8:
            errors[f"{side}_j8_qvel_abs_rad_s"] = abs(float(arm_qvel[7]))
        errors[f"{side}_gripper_inf_rad"] = float(
            np.linalg.norm(
                gripper_q - capture_state["gripper_q"][side],
                ord=np.inf,
            )
        )
    if not all(math.isfinite(value) for value in errors.values()):
        return False, {
            "reason": "nonfinite_start_state",
            "errors": errors,
            "observation_sequence": int(sequence),
            "adapter": adapter,
        }
    stale = bool(
        errors["trunk_inf_rad"] > MOVE_POINT_TO_POINT_START_TRUNK_TOLERANCE_RAD
        or errors["trunk_qvel_inf_rad_s"]
        > MOVE_POINT_TO_POINT_START_TRUNK_QVEL_RAD_S
        or errors["base_qvel_inf"] > MOVE_POINT_TO_POINT_START_BASE_QVEL
        or any(
            errors[f"{side}_arm_inf_rad"]
            > MOVE_POINT_TO_POINT_START_ARM_TOLERANCE_RAD
            or errors[f"{side}_arm_qvel_inf_rad_s"]
            > MOVE_POINT_TO_POINT_START_ARM_QVEL_RAD_S
            or errors[f"{side}_gripper_inf_rad"]
            > MOVE_POINT_TO_POINT_START_GRIPPER_TOLERANCE_RAD
            for side in ("left", "right")
        )
    )
    return (not stale), {
        "reason": "start_state_matches_capture" if not stale else "stale_capture",
        "errors": errors,
        "observation_sequence": int(sequence),
        "adapter": adapter,
    }



def _move_point_waypoint_stalled_sample(
    *,
    previous_error: float,
    error: float,
    before_q: np.ndarray,
    after_q: np.ndarray,
    after_qvel: np.ndarray,
) -> tuple[bool, float, float]:
    observed_joint_delta = float(
        np.linalg.norm(
            np.asarray(after_q, dtype=np.float64)[:7]
            - np.asarray(before_q, dtype=np.float64)[:7],
            ord=np.inf,
        )
    )
    observed_arm_qvel = float(
        np.linalg.norm(np.asarray(after_qvel, dtype=np.float64)[:7], ord=np.inf)
    )
    stalled = bool(
        math.isfinite(float(previous_error))
        and float(previous_error) - float(error) <= 1e-4
        and observed_joint_delta
        <= MOVE_POINT_TO_POINT_STALL_POSITION_DELTA_RAD
        and observed_arm_qvel <= MOVE_POINT_TO_POINT_STALL_QVEL_RAD_S
    )
    return stalled, observed_joint_delta, observed_arm_qvel



def _move_point_execute_trajectory(
    ctx,
    *,
    trajectory: dict[str, Any],
    capture_state: dict[str, Any],
    adapter,
    observation_sequence: int,
    timeout_s: float,
    max_steps: int,
    step_monitor=None,
    operation_name: str = "move_point_to_point",
    allow_start_rebase: bool = False,
    allow_gripper_recovery: bool = False,
    continue_during_gripper_disturbance: bool = False,
    recover_active_arm_disturbance: bool = False,
    execution_waypoints: Sequence[Sequence[float]] | None = None,
    execution_max_step_rad: float | None = None,
    tracker_reacquisition_max_steps: int | None = None,
):
    arm = str(trajectory["active_arm"])
    inactive_arm = "right" if arm == "left" else "left"
    expected_episode_id = str(trajectory["start_state"]["episode_id"])
    tracking = dict(trajectory["tracking"])
    waypoint_tolerance = float(tracking["waypoint_tolerance_rad"])
    max_tracking_error = float(tracking["max_tracking_error_rad"])
    execution_mode = str(
        tracking.get("execution_mode", "settled_waypoint_replay")
    )
    if execution_mode not in {
        "settled_waypoint_replay",
        "continuous_waypoint_stream",
    }:
        raise ValueError(f"unsupported trajectory execution mode {execution_mode!r}")
    continuous_stream = execution_mode == "continuous_waypoint_stream"
    if tracker_reacquisition_max_steps is not None:
        if (
            isinstance(tracker_reacquisition_max_steps, bool)
            or int(tracker_reacquisition_max_steps)
            != tracker_reacquisition_max_steps
            or int(tracker_reacquisition_max_steps) < 1
        ):
            raise ValueError(
                "tracker_reacquisition_max_steps must be a positive integer"
            )
        tracker_reacquisition_max_steps = int(
            tracker_reacquisition_max_steps
        )
    raw_trunk_assisted = tracking.get("trunk_assisted", False)
    if not isinstance(raw_trunk_assisted, bool):
        raise ValueError("trajectory trunk-assisted flag is invalid")
    if raw_trunk_assisted:
        raise ValueError(
            f"{operation_name} does not execute whole-body trajectories"
        )
    sequence = int(observation_sequence)
    initial = _adjust_kinematic_observation(ctx, arm)
    initial_trunk = np.asarray(initial["trunk"], dtype=np.float64).reshape(4)
    signed_waypoints = list(trajectory["waypoints"])
    signed_waypoint_count = len(signed_waypoints)
    if trajectory.get("trunk_waypoints") is not None:
        raise ValueError("arm-only trajectory contains trunk waypoints")
    if execution_waypoints is None:
        execution_waypoints_list = signed_waypoints
    else:
        execution_waypoints_list = []
        for waypoint_index, raw_waypoint in enumerate(execution_waypoints):
            execution_waypoints_list.append(
                _require_serialized_j8_zero(
                    raw_waypoint,
                    label=f"{operation_name} execution schedule waypoint {waypoint_index + 1}",
                )
            )
        if not execution_waypoints_list:
            raise ValueError("execution schedule produced no waypoints")
    bridge_step = (
        float(execution_max_step_rad)
        if execution_max_step_rad is not None
        else float(
            tracking.get(
                "max_joint_step_rad",
                MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD,
            )
        )
    )
    if not math.isfinite(bridge_step) or bridge_step <= 0.0:
        raise ValueError("execution_max_step_rad must be positive and finite")
    execution_bridge_waypoint_count = 0
    if allow_start_rebase:
        execution_waypoints_list, execution_bridge_waypoint_count = (
            _move_tracked_execution_bridge(
                initial["q"],
                execution_waypoints_list,
                arm=arm,
                max_step_rad=bridge_step,
            )
        )
    initial_inactive_q = np.asarray(
        ctx.world.arm_qpos_list(inactive_arm),
        dtype=np.float64,
    ).reshape(ARM_DOF)
    ctx.world.set_trunk_pin_qpos(initial_trunk)
    ctx.world.set_arm_pin_qpos(arm, initial["q"])
    ctx.world.set_arm_pin_qpos(inactive_arm, initial_inactive_q)
    safety_monitor = _adjust_safety_monitor(ctx, initial)
    started = time.monotonic()
    action_steps = 0
    completed_waypoints = 0
    max_observed_tracking_error = 0.0
    max_inactive_arm_drift = 0.0
    max_gripper_drift = 0.0
    final_gripper_drift = 0.0
    max_observed_j8 = abs(float(initial["observed_j8"]))
    tracking_recovery_steps = 0
    tracking_recovery_events: list[dict[str, Any]] = []
    tracker_reacquisition_hold_q: np.ndarray | None = None
    tracker_reacquisition_hold_steps = 0
    tracker_reacquisition_events: list[dict[str, Any]] = []
    gripper_recovery_hold_q: np.ndarray | None = None
    gripper_recovery_fingers: dict[str, list[int]] = {}
    gripper_recovery_steps = 0
    gripper_recovery_streak = 0
    gripper_recovery_stable_steps = 0
    gripper_recovery_events: list[dict[str, Any]] = []
    gripper_recovery_samples: list[dict[str, Any]] = []
    gripper_disturbance_active = False
    gripper_disturbance_steps = 0
    reason = "trajectory_complete"
    last_observed = initial
    last_gripper_report: dict[str, Any] = {}
    last_waypoint: dict[str, Any] = {}
    last_step_monitor: dict[str, Any] = {}

    for waypoint_index, raw_target in enumerate(execution_waypoints_list):
        target = _require_serialized_j8_zero(
            raw_target,
            label=f"{operation_name} waypoint {waypoint_index + 1}",
        )
        previous_error = float("inf")
        waypoint_stall_steps = 0
        while True:
            _check_cancelled(
                ctx,
                f"{operation_name} waypoint {waypoint_index + 1}",
            )
            if str(ctx.world.episode_id()) != expected_episode_id:
                reason = "episode_changed"
                break
            if time.monotonic() - started >= float(timeout_s):
                reason = "timeout"
                break
            if action_steps >= int(max_steps):
                reason = "max_steps"
                break
            before = _adjust_kinematic_observation(ctx, arm)
            gripper_overrides, last_gripper_report = (
                _move_point_gripper_action_overrides(
                    ctx,
                    capture_state,
                    # Preserve the existing local servo on the fast path.
                    # The outward-only command is enabled only after a fresh
                    # proprio sample has crossed the soft recovery boundary.
                    enable_open_recovery=bool(
                        allow_gripper_recovery
                        and (
                            continue_during_gripper_disturbance
                            or gripper_recovery_hold_q is not None
                        )
                    ),
                )
            )
            command_is_reacquisition_hold = (
                tracker_reacquisition_hold_q is not None
            )
            if (
                command_is_reacquisition_hold
                and tracker_reacquisition_max_steps is not None
                and tracker_reacquisition_hold_steps
                >= tracker_reacquisition_max_steps
            ):
                tracker_reacquisition_events.append(
                    {
                        "event": "reacquisition_exhausted",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence),
                        "hold_steps": int(tracker_reacquisition_hold_steps),
                        "max_hold_steps": int(
                            tracker_reacquisition_max_steps
                        ),
                    }
                )
                reason = "tracked_point_reacquisition_failed"
                break
            command_is_gripper_recovery_hold = bool(
                not command_is_reacquisition_hold
                and gripper_recovery_hold_q is not None
            )
            command = (
                tracker_reacquisition_hold_q.copy()
                if command_is_reacquisition_hold
                else (
                    gripper_recovery_hold_q.copy()
                    if command_is_gripper_recovery_hold
                    else target
                )
            )
            action_overrides = {
                f"arm_{arm}": command.astype(float).tolist(),
                **gripper_overrides,
            }
            action = ctx.world.make_action(**action_overrides)
            if command_is_reacquisition_hold:
                tracker_reacquisition_hold_steps += 1
            yield action
            action_steps += 1
            advanced, current_sequence = _adjust_observation_advanced(
                adapter,
                sequence,
            )
            if not advanced:
                reason = "observation_not_advanced"
                break
            sequence = current_sequence
            if str(ctx.world.episode_id()) != expected_episode_id:
                reason = "episode_changed"
                break
            after = _adjust_kinematic_observation(ctx, arm)
            last_observed = after
            max_observed_j8 = max(
                max_observed_j8,
                abs(float(after["observed_j8"])),
            )
            arm_error = float(np.linalg.norm(command - after["q"], ord=np.inf))
            error = arm_error
            (
                stalled_sample,
                observed_joint_delta,
                observed_arm_qvel,
            ) = _move_point_waypoint_stalled_sample(
                previous_error=previous_error,
                error=error,
                before_q=before["q"],
                after_q=after["q"],
                after_qvel=after["qvel"],
            )
            max_observed_tracking_error = max(
                max_observed_tracking_error,
                error,
            )
            inactive_q = np.asarray(
                ctx.world.arm_qpos_list(inactive_arm),
                dtype=np.float64,
            ).reshape(ARM_DOF)
            inactive_drift = float(
                np.linalg.norm(inactive_q - initial_inactive_q, ord=np.inf)
            )
            max_inactive_arm_drift = max(
                max_inactive_arm_drift,
                inactive_drift,
            )
            gripper_observation: dict[str, dict[str, Any]] = {}
            for side in ("left", "right"):
                observed_gripper_q = np.asarray(
                    ctx.world.gripper_qpos_list(side),
                    dtype=np.float64,
                ).reshape(2)
                observed_gripper_qvel = np.asarray(
                    ctx.world.gripper_qvel_list(side),
                    dtype=np.float64,
                ).reshape(2)
                gripper_error = (
                    capture_state["gripper_q"][side] - observed_gripper_q
                )
                gripper_observation[side] = {
                    "capture_qpos_m": capture_state["gripper_q"][side]
                    .astype(float)
                    .tolist(),
                    "observed_qpos_m": observed_gripper_q.astype(float).tolist(),
                    "observed_qvel_mps": observed_gripper_qvel.astype(float).tolist(),
                    "error_m": gripper_error.astype(float).tolist(),
                    "drift_inf_m": float(
                        np.linalg.norm(gripper_error, ord=np.inf)
                    ),
                    "close_keepalive_active": bool(
                        getattr(
                            ctx.world,
                            "gripper_close_keepalive_active",
                            lambda _arm: False,
                        )(side)
                    ),
                    "effort_controlled": bool(
                        getattr(
                            ctx.world,
                            "gripper_uses_effort",
                            lambda _arm: False,
                        )(side)
                    ),
                }
            current_gripper_drift = max(
                float(item["drift_inf_m"])
                for item in gripper_observation.values()
            )
            max_gripper_drift = max(
                max_gripper_drift,
                current_gripper_drift,
            )
            final_gripper_drift = current_gripper_drift
            safety_reason, safety_observation = _adjust_safety_after_action(
                ctx,
                safety_monitor,
                before=before,
                command=command,
                after=after,
                base_translation_limit_m=(
                    MOVE_POINT_TO_POINT_BASE_TRANSLATION_LIMIT_M
                ),
                base_yaw_limit_rad=MOVE_POINT_TO_POINT_BASE_YAW_LIMIT_RAD,
                proprio_stall_requires_stationary=True,
                expected_trunk=None,
            )
            last_waypoint = {
                "index": int(waypoint_index + 1),
                "command_role": (
                    "tracker_reacquisition_hold"
                    if command_is_reacquisition_hold
                    else (
                        "gripper_recovery_hold"
                        if command_is_gripper_recovery_hold
                        else "trajectory_waypoint"
                    )
                ),
                "tracking_error_inf_rad": error,
                "arm_tracking_error_inf_rad": arm_error,
                "observed_joint_delta_inf_rad": observed_joint_delta,
                "observed_arm_qvel_inf_rad_s": observed_arm_qvel,
                "inactive_arm_drift_inf_rad": inactive_drift,
                "gripper_drift_inf_m": current_gripper_drift,
                "gripper_observation": gripper_observation,
                "safety_observation": safety_observation,
            }
            monitor_requests_reacquisition_hold = False
            if callable(step_monitor):
                monitor_reason, last_step_monitor = step_monitor(
                    after,
                    int(sequence),
                )
                if monitor_reason is not None:
                    reason = str(monitor_reason)
                    break
                monitor_requests_reacquisition_hold = bool(
                    last_step_monitor.get("temporarily_unobserved", False)
                )
            if (
                safety_reason == "proprio_stall"
                and recover_active_arm_disturbance
                and error > max_tracking_error
            ):
                # A finite active-arm displacement can be an external tug, not
                # a controller failure.  Reissuing the immutable target is the
                # recovery; timeout and max_steps remain the hard bound.
                safety_reason = None
                safety_monitor.proprio_stall_steps = 0
            if safety_reason is not None:
                reason = safety_reason
                break
            if inactive_drift > MOVE_POINT_TO_POINT_START_ARM_TOLERANCE_RAD:
                reason = "inactive_arm_drift"
                break
            if continue_during_gripper_disturbance:
                disturbance_now = bool(
                    current_gripper_drift
                    > MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M
                )
                if disturbance_now:
                    gripper_disturbance_steps += 1
                    gripper_recovery_steps += 1
                    gripper_recovery_samples.append(
                        {
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "drift_inf_m": float(current_gripper_drift),
                            "gripper_observation": gripper_observation,
                            "arm_trajectory_continued": True,
                        }
                    )
                    if not gripper_disturbance_active:
                        gripper_recovery_events.append(
                            {
                                "event": "concurrent_recovery_started",
                                "waypoint_index": int(waypoint_index + 1),
                                "observation_sequence": int(sequence),
                                "drift_inf_m": float(current_gripper_drift),
                                "soft_limit_m": float(
                                    MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M
                                ),
                                "hard_limit_m": float(
                                    MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M
                                ),
                                "arm_trajectory_continued": True,
                            }
                        )
                    gripper_disturbance_active = True
                elif gripper_disturbance_active and (
                    current_gripper_drift
                    <= MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M
                ):
                    gripper_recovery_events.append(
                        {
                            "event": "concurrent_recovery_observed",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "drift_inf_m": float(current_gripper_drift),
                            "disturbance_steps": int(gripper_disturbance_steps),
                            "arm_trajectory_continued": True,
                        }
                    )
                    gripper_disturbance_active = False
            elif (
                current_gripper_drift
                > MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M
            ):
                gripper_recovery_events.append(
                    {
                        "event": "hard_limit_exceeded",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence),
                        "drift_inf_m": float(current_gripper_drift),
                        "hard_limit_m": float(
                            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M
                        ),
                        "gripper_observation": gripper_observation,
                    }
                )
                reason = "gripper_drift"
                break
            if (
                not continue_during_gripper_disturbance
                and gripper_recovery_hold_q is not None
            ):
                recovery_supported = bool(
                    gripper_recovery_fingers
                    and all(
                        bool(gripper_observation[side]["effort_controlled"])
                        and not bool(
                            gripper_observation[side]["close_keepalive_active"]
                        )
                        for side in gripper_recovery_fingers
                    )
                )
                if not recovery_supported:
                    gripper_recovery_events.append(
                        {
                            "event": "hard_limit_or_mode_failure",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "drift_inf_m": float(current_gripper_drift),
                            "gripper_observation": gripper_observation,
                        }
                    )
                    reason = "gripper_drift"
                    break
                gripper_recovery_steps += 1
                gripper_recovery_streak += 1
                recovery_drift = max(
                    abs(float(gripper_observation[side]["error_m"][index]))
                    for side, indices in gripper_recovery_fingers.items()
                    for index in indices
                )
                max_gripper_qvel = max(
                    abs(
                        float(
                            gripper_observation[side]["observed_qvel_mps"][
                                index
                            ]
                        )
                    )
                    for side, indices in gripper_recovery_fingers.items()
                    for index in indices
                )
                gripper_recovery_samples.append(
                    {
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence),
                        "recovery_streak": int(gripper_recovery_streak),
                        "drift_inf_m": float(recovery_drift),
                        "overall_drift_inf_m": float(current_gripper_drift),
                        "max_qvel_mps": float(max_gripper_qvel),
                        "recovery_fingers": {
                            side: list(indices)
                            for side, indices in gripper_recovery_fingers.items()
                        },
                        "qpos_m": {
                            side: list(item["observed_qpos_m"])
                            for side, item in gripper_observation.items()
                        },
                        "qvel_mps": {
                            side: list(item["observed_qvel_mps"])
                            for side, item in gripper_observation.items()
                        },
                        "command_n": {
                            side: list(
                                last_gripper_report.get(side, {}).get(
                                    "command", []
                                )
                            )
                            for side in ("left", "right")
                        },
                    }
                )
                recovery_sample_stable = bool(
                    recovery_drift
                    <= MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M
                    and max_gripper_qvel
                    <= MOVE_TRACKED_POINT_GRIPPER_RECOVERY_QVEL_MPS
                )
                gripper_recovery_stable_steps = (
                    gripper_recovery_stable_steps + 1
                    if recovery_sample_stable
                    else 0
                )
                if (
                    gripper_recovery_stable_steps
                    >= MOVE_TRACKED_POINT_GRIPPER_RECOVERY_STABLE_STEPS
                ):
                    gripper_recovery_events.append(
                        {
                            "event": "recovered",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "drift_inf_m": float(recovery_drift),
                            "overall_drift_inf_m": float(
                                current_gripper_drift
                            ),
                            "max_qvel_mps": float(max_gripper_qvel),
                            "recovery_streak": int(gripper_recovery_streak),
                            "stable_steps": int(
                                gripper_recovery_stable_steps
                            ),
                        }
                    )
                    gripper_recovery_hold_q = None
                    gripper_recovery_fingers = {}
                    gripper_recovery_streak = 0
                    gripper_recovery_stable_steps = 0
                    previous_error = float("inf")
                    waypoint_stall_steps = 0
                    # The arm was deliberately held while the fingers
                    # recovered. Re-issue this same immutable waypoint.
                    continue
                if (
                    gripper_recovery_streak
                    >= MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS
                ):
                    gripper_recovery_events.append(
                        {
                            "event": "recovery_exhausted",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "drift_inf_m": float(recovery_drift),
                            "overall_drift_inf_m": float(
                                current_gripper_drift
                            ),
                            "max_qvel_mps": float(max_gripper_qvel),
                            "recovery_streak": int(gripper_recovery_streak),
                        }
                    )
                    reason = "gripper_drift"
                    break
                previous_error = float("inf")
                waypoint_stall_steps = 0
                continue
            recoverable_fingers: dict[str, list[int]] = {}
            if allow_gripper_recovery and not continue_during_gripper_disturbance:
                for side, item in gripper_observation.items():
                    if not bool(item["effort_controlled"]) or bool(
                        item["close_keepalive_active"]
                    ):
                        continue
                    capture_q = np.asarray(
                        item["capture_qpos_m"], dtype=np.float64
                    )
                    error_q = np.asarray(item["error_m"], dtype=np.float64)
                    indices = np.flatnonzero(
                        np.logical_and(
                            capture_q
                            >= MOVE_TRACKED_POINT_GRIPPER_OPEN_HOLD_THRESHOLD_M,
                            error_q
                            > MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M,
                        )
                    ).astype(int).tolist()
                    if indices:
                        recoverable_fingers[side] = indices
            if recoverable_fingers:
                gripper_recovery_hold_q = np.asarray(
                    after["q"], dtype=np.float64
                ).reshape(ARM_DOF).copy()
                gripper_recovery_fingers = {
                    side: list(indices)
                    for side, indices in recoverable_fingers.items()
                }
                gripper_recovery_streak = 0
                gripper_recovery_stable_steps = 0
                gripper_recovery_events.append(
                    {
                        "event": "hold_started",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence),
                        "recovery_fingers": {
                            side: list(indices)
                            for side, indices in gripper_recovery_fingers.items()
                        },
                        "drift_inf_m": float(current_gripper_drift),
                        "soft_limit_m": float(
                            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M
                        ),
                        "release_limit_m": float(
                            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M
                        ),
                        "hard_limit_m": float(
                            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M
                        ),
                        "gripper_observation": gripper_observation,
                        "hold_q_rad": gripper_recovery_hold_q.astype(
                            float
                        ).tolist(),
                    }
                )
                previous_error = float("inf")
                waypoint_stall_steps = 0
                continue
            if ARM_DOF == 8:
                raw_inactive = np.asarray(
                    ctx.world.arm_qpos_list(inactive_arm),
                    dtype=np.float64,
                ).reshape(ARM_DOF)
                if (
                    abs(float(raw_inactive[7]))
                    > LOCKED_J8_CAPTURE_TOLERANCE_RAD
                ):
                    reason = "inactive_arm_j8_not_locked"
                    break
            if error > max_tracking_error:
                moving_toward_target = bool(
                    math.isfinite(observed_joint_delta)
                    and observed_joint_delta > 1.0e-5
                ) or bool(
                    math.isfinite(observed_arm_qvel)
                    and observed_arm_qvel > 1.0e-3
                )
                if recover_active_arm_disturbance:
                    tracking_recovery_steps += 1
                    tracking_recovery_events.append(
                        {
                            "waypoint_index": int(waypoint_index + 1),
                            "attempt": int(tracking_recovery_steps),
                            "tracking_error_inf_rad": float(error),
                            "observed_joint_delta_inf_rad": float(
                                observed_joint_delta
                            ),
                            "observed_arm_qvel_inf_rad_s": float(
                                observed_arm_qvel
                            ),
                            "reason": "external_disturbance_target_reissued",
                            "bounded_by": "timeout_and_max_steps",
                        }
                    )
                    previous_error = error
                    continue
                if (
                    continuous_stream
                    and moving_toward_target
                    and error <= MOVE_TRACKED_POINT_STREAM_RECOVERY_ERROR_RAD
                    and tracking_recovery_steps
                    < MOVE_TRACKED_POINT_STREAM_MAX_RECOVERY_STEPS
                ):
                    tracking_recovery_steps += 1
                    tracking_recovery_events.append(
                        {
                            "waypoint_index": int(waypoint_index + 1),
                            "attempt": int(tracking_recovery_steps),
                            "tracking_error_inf_rad": float(error),
                            "observed_joint_delta_inf_rad": float(
                                observed_joint_delta
                            ),
                            "observed_arm_qvel_inf_rad_s": float(
                                observed_arm_qvel
                            ),
                            "reason": "moving_toward_target_retry",
                        }
                    )
                    # Re-send the same command once.  This is an execution
                    # recovery only; the signed waypoint order and endpoint
                    # remain unchanged.  A hard retry bound prevents a slow
                    # controller from consuming the entire evaluator budget.
                    previous_error = error
                    continue
                reason = "tracking_error"
                break
            tracking_recovery_steps = 0
            if monitor_requests_reacquisition_hold:
                if tracker_reacquisition_hold_q is None:
                    tracker_reacquisition_hold_q = np.asarray(
                        after["q"], dtype=np.float64
                    ).reshape(ARM_DOF).copy()
                    tracker_reacquisition_events.append(
                        {
                            "event": "hold_started",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence),
                            "hold_q_rad": tracker_reacquisition_hold_q.astype(
                                float
                            ).tolist(),
                        }
                    )
                previous_error = float("inf")
                waypoint_stall_steps = 0
                continue
            if command_is_reacquisition_hold:
                tracker_reacquisition_events.append(
                    {
                        "event": "observation_reacquired",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence),
                        "hold_steps": int(tracker_reacquisition_hold_steps),
                    }
                )
                tracker_reacquisition_hold_q = None
                previous_error = float("inf")
                waypoint_stall_steps = 0
                # Resume this same immutable waypoint on the next evaluator step.
                continue
            if continuous_stream:
                # This waypoint is one sample of a globally time-parameterized
                # trajectory, not a pose at which the robot should stop.  One
                # fresh evaluator observation validates its tracking corridor;
                # the final target receives the ordinary stable-pose check.
                completed_waypoints += 1
                break
            if error <= waypoint_tolerance:
                completed_waypoints += 1
                break
            # A finite-impedance controller can briefly overshoot a close
            # waypoint while the arm is still moving quickly.  Error
            # monotonicity alone therefore cannot establish a proprioceptive
            # stall.  Require both negligible measured displacement and low
            # measured velocity so transient inertia remains executable while
            # a genuinely stationary controller still fails closed.
            waypoint_stall_steps = (
                waypoint_stall_steps + 1 if stalled_sample else 0
            )
            previous_error = error
            if waypoint_stall_steps >= ADJUST_EEF_STALL_STEPS:
                reason = "proprio_stall"
                break
        if reason != "trajectory_complete":
            break

    completed = bool(
        reason == "trajectory_complete"
        and completed_waypoints == len(execution_waypoints_list)
    )
    ctx.world.set_arm_pin_qpos(arm, last_observed["q"])
    return {
        "ok": completed,
        "reason": reason,
        "controller": (
            "immutable_signed_joint_trajectory_stream"
            if continuous_stream
            else "immutable_signed_joint_waypoint_replay"
        ),
        "execution_mode": execution_mode,
        "waypoint_completion_semantics": (
            "one_official_action_per_waypoint_with_proprio_tracking_corridor"
            if continuous_stream
            else "hold_until_waypoint_tolerance"
        ),
        "observation_action_loop": True,
        "steps_executed": int(action_steps),
        "waypoint_count": int(len(execution_waypoints_list)),
        "signed_waypoint_count": int(signed_waypoint_count),
        "execution_schedule_applied": bool(execution_waypoints is not None),
        "execution_max_step_rad": float(bridge_step),
        "execution_bridge_waypoint_count": int(execution_bridge_waypoint_count),
        "execution_start_rebased": bool(allow_start_rebase),
        "trunk_motion": False,
        "waypoints_completed": int(completed_waypoints),
        "waypoint_tolerance_rad": waypoint_tolerance,
        "max_tracking_error_limit_rad": max_tracking_error,
        "max_observed_tracking_error_rad": float(
            max_observed_tracking_error
        ),
        "tracking_recovery_steps": int(
            sum(
                int(item.get("attempt", 0))
                for item in tracking_recovery_events
            )
        ),
        "tracking_recovery_events": tracking_recovery_events,
        "active_arm_disturbance_recovery_enabled": bool(
            recover_active_arm_disturbance
        ),
        "tracker_reacquisition_hold_steps": int(
            tracker_reacquisition_hold_steps
        ),
        "tracker_reacquisition_max_steps": tracker_reacquisition_max_steps,
        "tracker_reacquisition_events": tracker_reacquisition_events,
        "gripper_recovery_enabled": bool(allow_gripper_recovery),
        "gripper_recovery_concurrent_with_arm": bool(
            allow_gripper_recovery and continue_during_gripper_disturbance
        ),
        "gripper_recovery_steps": int(gripper_recovery_steps),
        "gripper_recovery_events": gripper_recovery_events,
        "gripper_recovery_samples": gripper_recovery_samples,
        "gripper_recovery_active_at_exit": bool(
            gripper_recovery_hold_q is not None or gripper_disturbance_active
        ),
        "gripper_recovery_fingers_active_at_exit": {
            side: list(indices)
            for side, indices in gripper_recovery_fingers.items()
        },
        "gripper_recovery_soft_limit_m": float(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M
        ),
        "gripper_recovery_release_m": float(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M
        ),
        "gripper_recovery_hard_limit_m": float(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M
        ),
        "gripper_recovery_max_steps": int(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS
        ),
        "gripper_recovery_required_stable_steps": int(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_STABLE_STEPS
        ),
        "gripper_recovery_qvel_threshold_mps": float(
            MOVE_TRACKED_POINT_GRIPPER_RECOVERY_QVEL_MPS
        ),
        "gripper_recovery_effort_n": float(
            MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N
        ),
        "gripper_recovery_target_policy": (
            "concurrent_servo_without_arm_interruption_final_goal_revalidated"
            if continue_during_gripper_disturbance
            else "near_open_inward_drift_only_intermediate_and_close_latch_unchanged"
        ),
        "max_inactive_arm_drift_inf_rad": float(max_inactive_arm_drift),
        "max_gripper_drift_inf_m": float(max_gripper_drift),
        "final_gripper_drift_inf_m": float(final_gripper_drift),
        "max_observed_j8_abs_rad": float(max_observed_j8),
        "gripper_command": last_gripper_report,
        "safety": _adjust_safety_report(safety_monitor),
        "last_waypoint": last_waypoint,
        "last_step_monitor": last_step_monitor,
        "last_observation_sequence": int(sequence),
    }



def _move_tracked_head_camera_pose(
    ctx,
    *,
    expected_sequence: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the frozen head camera pose used by camera-facing presets."""

    camera, sequence, _adapter = _adjust_camera_pose(ctx, "head")
    if int(sequence) != int(expected_sequence):
        raise RuntimeError(
            "head camera pose is not synchronized with the tracked-point observation"
        )
    position = np.asarray(camera["pos"], dtype=np.float64).reshape(3)
    quaternion = np.asarray(camera["quat"], dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("head camera pose contains a non-finite value")
    return position.copy(), quaternion.copy()



def _move_tracked_execution_bridge(
    initial_q: np.ndarray,
    signed_waypoints: list[Any],
    *,
    arm: str,
    max_step_rad: float = MOVE_POINT_TO_POINT_MAX_JOINT_STEP_RAD,
) -> tuple[list[np.ndarray], int]:
    """Prepend bounded joint interpolation when handoff q differs from plan.

    The signed trajectory is never rewritten.  This returns an execution-only
    sequence whose first command starts at the latest evaluator proprioception
    and whose every adjacent joint step remains within the action contract.
    """

    if not math.isfinite(float(max_step_rad)) or float(max_step_rad) <= 0.0:
        raise ValueError("execution bridge max_step_rad must be positive and finite")
    current = locked_arm_vector(initial_q, label=f"{arm} handoff qpos").copy()
    limits = _ARM_LIMITS[arm][:ARM_DOF]
    if not np.all(np.isfinite(current)):
        raise ValueError("handoff qpos contains a non-finite value")

    execution: list[np.ndarray] = []
    bridge_count = 0
    # Proprioception is measured state, not a planned action.  A controller can
    # report a small residual just beyond a URDF limit after an otherwise valid
    # trajectory.  Recover it with one bounded, in-limit command before replaying
    # the signed trajectory.  Larger discrepancies still fail closed instead of
    # hiding a stale/wrong robot state.
    lower_violation = np.maximum(limits[:, 0] - current, 0.0)
    upper_violation = np.maximum(current - limits[:, 1], 0.0)
    limit_violation = np.maximum(lower_violation, upper_violation)
    max_limit_violation = float(np.max(limit_violation))
    if max_limit_violation > 0.0:
        if max_limit_violation > float(max_step_rad):
            joint_index = int(np.argmax(limit_violation))
            raise ValueError(
                "handoff qpos exceeds the recoverable local joint-limit "
                f"residual: arm={arm} J{joint_index + 1}="
                f"{current[joint_index]:.12g}rad outside "
                f"[{limits[joint_index, 0]:.12g},"
                f"{limits[joint_index, 1]:.12g}] by "
                f"{max_limit_violation:.12g}rad; max recoverable residual is "
                f"{float(max_step_rad):.12g}rad"
            )
        current = np.clip(current, limits[:, 0], limits[:, 1])
        current = _require_serialized_j8_zero(
            current,
            label=f"{arm} handoff joint-limit recovery waypoint",
        )
        execution.append(current.astype(np.float64, copy=False))
        bridge_count += 1

    for waypoint_index, raw_target in enumerate(signed_waypoints):
        target = _require_serialized_j8_zero(
            raw_target,
            label=f"move_tracked execution waypoint {waypoint_index + 1}",
        )
        delta = target - current
        max_delta = float(np.linalg.norm(delta, ord=np.inf))
        # The execution scheduler accepts a transition at ``cap +`` its
        # serialized lock tolerance.  Recomputing that same subtraction here
        # can differ by a few ULPs; a strict ceil then turns a valid one-step
        # transition into two commands.  Across a compressed path this used to
        # manufacture dozens of unbudgeted bridge actions and stop a healthy
        # trajectory near its endpoint.  Ignore only numerical-scale excess;
        # any physically meaningful excess is still split below.
        effective_delta = max(
            0.0,
            max_delta
            - MOVE_TRACKED_POINT_EXECUTION_STEP_NUMERIC_TOLERANCE_RAD,
        )
        segment_steps = max(
            1,
            int(math.ceil(effective_delta / float(max_step_rad))),
        )
        for step_index in range(1, segment_steps + 1):
            point = current + (delta * (float(step_index) / float(segment_steps)))
            point = _require_serialized_j8_zero(
                point,
                label=f"move_tracked execution bridge waypoint {len(execution) + 1}",
            )
            if np.any(point < limits[:, 0]) or np.any(point > limits[:, 1]):
                raise ValueError("execution bridge waypoint exceeds local joint limits")
            execution.append(point.astype(np.float64, copy=False))
        if segment_steps > 1:
            bridge_count += segment_steps - 1
        current = target.copy()
    if not execution:
        raise ValueError("signed trajectory has no execution waypoints")
    return execution, int(bridge_count)



def _move_tracked_prepare_execution_schedule(
    *,
    trajectory: dict[str, Any],
    state: LocalRobotState,
    arm: str,
    start_q: Sequence[float],
    max_steps: int,
    available_time_s: float,
    observation_dt_s: float | None,
    visual_tracked_point_count: int | None = None,
) -> dict[str, Any]:
    """Derive a bounded execution view of a signed local trajectory.

    Planning remains immutable and is still loaded/verified from disk.  This
    helper only chooses a monotone subsequence for the current evaluator
    budget; it never edits the signed trajectory or queries simulator state.
    """

    tracking = trajectory.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("move-tracked trajectory tracking metadata is missing")
    if tracking.get("trunk_assisted") is True:
        raise ValueError(
            "move_tracked_point does not schedule whole-body trajectories"
        )
    scheduled_start_q, start_limit_recovery = _normalize_arm_q_to_joint_limits(
        start_q,
        arm=arm,
        label="move-tracked execution scheduling start qpos",
        point="execution_schedule_start",
        recover_observed_overshoot=True,
    )
    base_step = float(
        tracking.get("max_joint_step_rad", MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD)
    )
    # A measured start just outside a static local limit needs one explicit,
    # in-limit recovery command before the signed path.  Reserve that command
    # before scheduling rather than discovering it after the budget is full.
    start_bridge_reserve = 1 if start_limit_recovery else 0
    scheduled_max_steps = int(max_steps) - int(start_bridge_reserve)
    scheduled_available_time_s = float(available_time_s)
    if observation_dt_s is not None and start_bridge_reserve:
        scheduled_available_time_s -= float(observation_dt_s)
    if scheduled_max_steps < 1 or scheduled_available_time_s <= 0.0:
        return {
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": (
                "remaining evaluator budget cannot accommodate the observed "
                "joint-limit recovery and one trajectory action"
            ),
            "execution_start_joint_limit_recovery": _json_ready(
                start_limit_recovery
            ),
            "execution_bridge_reserved_waypoints": int(start_bridge_reserve),
            "budget_steps": int(max(0, max_steps)),
        }
    schedule_kwargs = {
        "start_q": scheduled_start_q,
        "max_steps": int(scheduled_max_steps),
        "available_time_s": float(scheduled_available_time_s),
        "observation_dt_s": observation_dt_s,
        "max_joint_step_rad": base_step,
        "preferred_max_steps": (
            MOVE_TRACKED_POINT_EXECUTION_PREFERRED_MAX_STEPS
        ),
        "max_joint_corridor_deviation_rad": (
            MOVE_TRACKED_POINT_EXECUTION_MAX_JOINT_CORRIDOR_DEVIATION_RAD
        ),
        "safety_time_steps": MOVE_TRACKED_POINT_EXECUTION_SAFETY_TIME_STEPS,
        "j8_index": LOCKED_J8_INDEX,
        "j8_value": LOCKED_J8_VALUE_RAD,
        "lock_tolerance": LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
    }
    relaxed_execution = False
    visual_speedup_cap = (
        MOVE_TRACKED_POINT_NPOINT_VISUAL_MAX_SPEEDUP_FACTOR
        if visual_tracked_point_count is not None
        and int(visual_tracked_point_count)
        >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
        else MOVE_TRACKED_POINT_EXECUTION_MAX_SPEEDUP_FACTOR
    )
    report = _schedule_tracked_execution_waypoints(
        trajectory.get("waypoints", []),
        max_speedup_factor=visual_speedup_cap,
        **schedule_kwargs,
    )
    if (
        not bool(report.get("ok"))
        and report.get("reason") == "execution_budget_infeasible"
        and visual_speedup_cap
        == MOVE_TRACKED_POINT_EXECUTION_MAX_SPEEDUP_FACTOR
    ):
        # Preserve the bounded scheduler byte-for-byte whenever it succeeds.
        # Only a path that cannot fit under that generic cap gets a second,
        # path-derived pass.  The signed trajectory already proved adjacent
        # same-branch IK continuity; this pass merely drops redundant temporal
        # samples and is revalidated by local FK below.
        bounded_report = {
            key: value
            for key, value in report.items()
            if key
            in {
                "reason",
                "error",
                "budget_steps",
                "time_budget_steps",
                "input_waypoint_count",
                "required_waypoints_at_max_speedup",
            }
        }
        relaxed_report = _schedule_tracked_execution_waypoints(
            trajectory.get("waypoints", []),
            max_speedup_factor=None,
            **schedule_kwargs,
        )
        relaxed_report["bounded_safety_attempt"] = bounded_report
        report = relaxed_report
        relaxed_execution = bool(report.get("ok"))
    if not bool(report.get("ok")):
        report["visual_tracked_point_count"] = visual_tracked_point_count
        report["visual_speedup_cap"] = float(visual_speedup_cap)
        return report

    schedule_drops_signed_samples = int(
        report.get("dropped_waypoint_count", 0)
    ) > 0
    if relaxed_execution or schedule_drops_signed_samples:
        # Every compressed schedule is admissible only for a trajectory whose
        # signed source already proved the IK branch and global path ordering.
        # This applies to both the bounded fast path and the last-resort
        # relaxed pass: either may omit samples, but neither may manufacture a
        # new branch or silently stop before the signed endpoint.
        planner = trajectory.get("planner")
        if not isinstance(planner, dict):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "compressed execution requires signed planner metadata",
            }
        if (
            planner.get("same_branch_checked") is not True
            or planner.get("monotonic_progress_checked") is not True
            or planner.get("internal_stop_count") != 0
        ):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": (
                    "compressed execution requires a signed same-branch, "
                    "monotone trajectory"
                ),
            }
        signed_waypoints = trajectory.get("waypoints")
        selected_indices = report.get("selected_indices")
        selected_waypoints = report.get("waypoints")
        endpoint_index_valid = False
        if isinstance(selected_indices, list) and selected_indices:
            try:
                endpoint_index_valid = (
                    int(selected_indices[-1]) == len(signed_waypoints) - 1
                    if isinstance(signed_waypoints, list)
                    else False
                )
            except (TypeError, ValueError):
                endpoint_index_valid = False
        if (
            not isinstance(signed_waypoints, list)
            or not signed_waypoints
            or not isinstance(selected_indices, list)
            or not isinstance(selected_waypoints, list)
            or len(selected_indices) != len(selected_waypoints)
            or not selected_indices
            or not endpoint_index_valid
        ):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "compressed execution must retain the signed endpoint",
            }
        previous_index = -1
        previous_q = scheduled_start_q.copy()
        local_lower, local_upper = _local_arm_joint_limits(state, arm)
        transition_limits = np.column_stack(
            (
                np.asarray(local_lower, dtype=np.float64).reshape(-1),
                np.asarray(local_upper, dtype=np.float64).reshape(-1),
            )
        )[:ARM_DOF]
        transition_samples = 0
        transition_max_step = 0.0
        for selected_index, raw_q in zip(selected_indices, selected_waypoints):
            try:
                source_index = int(selected_index)
            except (TypeError, ValueError):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution contains a non-integer source index",
                }
            if source_index < 0 or source_index >= len(signed_waypoints):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution source index is out of range",
                }
            if source_index <= previous_index:
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution source indices are not increasing",
                }
            target_q = _require_serialized_j8_zero(
                raw_q,
                label=f"relaxed execution transition {source_index}",
            )
            transition_step = float(
                np.linalg.norm(target_q - previous_q, ord=np.inf)
            )
            if not math.isfinite(transition_step):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution transition is non-finite",
                }
            transition_max_step = max(transition_max_step, transition_step)
            sample_count = max(
                1,
                min(
                    32,
                    int(
                        math.ceil(
                            transition_step
                            / max(base_step, 1.0e-6)
                        )
                    ),
                ),
            )
            for sample_index in range(1, sample_count + 1):
                fraction = float(sample_index) / float(sample_count)
                q_sample = _require_serialized_j8_zero(
                    previous_q
                    + fraction * (target_q - previous_q),
                    label=(
                        "relaxed execution local-FK sample "
                        f"{source_index}:{sample_index}"
                    ),
                )
                if np.any(q_sample < transition_limits[:, 0]) or np.any(
                    q_sample > transition_limits[:, 1]
                ):
                    return {
                        **report,
                        "ok": False,
                        "reason": "execution_schedule_invalid",
                        "error": "relaxed execution transition exceeds local limits",
                    }
                position, quaternion = _local_eef_pose(state, arm, q_sample)
                if not np.all(np.isfinite(position)) or not np.all(
                    np.isfinite(quaternion)
                ):
                    return {
                        **report,
                        "ok": False,
                        "reason": "execution_schedule_invalid",
                        "error": "relaxed execution local FK transition is non-finite",
                    }
                transition_samples += 1
            previous_q = target_q.copy()
            previous_index = source_index
        report.update(
            {
                "execution_start_joint_limit_recovery": _json_ready(
                    start_limit_recovery
                ),
                "same_branch_inherited_from_signed_path": True,
                "transition_local_fk_finite_checked": True,
                "transition_local_fk_sample_count": int(transition_samples),
                "transition_max_joint_step_rad": float(transition_max_step),
                "execution_path_validation": (
                    "signed_same_branch_with_local_fk_transition_sampling"
                ),
            }
        )

    # A schedule is still policy-owned data.  Revalidate every selected state
    # against the local robot model before an action can be emitted.  This
    # catches malformed/reordered schedule output and keeps the fallback
    # independent of simulator collision or pose APIs.
    local_lower, local_upper = _local_arm_joint_limits(state, arm)
    limits = np.column_stack(
        (
            np.asarray(local_lower, dtype=np.float64).reshape(-1),
            np.asarray(local_upper, dtype=np.float64).reshape(-1),
        )
    )[:ARM_DOF]
    checked: list[list[float]] = []
    previous = np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF)
    max_step = 0.0
    for index, raw_q in enumerate(report.get("waypoints", [])):
        q = _require_serialized_j8_zero(
            raw_q,
            label=f"move-tracked execution schedule waypoint {index + 1}",
        )
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule waypoint exceeds local joint limits",
            }
        step = float(np.linalg.norm(q - previous, ord=np.inf))
        if not math.isfinite(step):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule contains a non-finite joint step",
            }
        max_step = max(max_step, step)
        position, quaternion = _local_eef_pose(state, arm, q)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule local FK returned non-finite pose",
            }
        checked.append(q.astype(float).tolist())
        previous = q
    if not checked:
        return {
            **report,
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": "execution schedule contains no waypoints",
        }
    execution_max_step = max(
        float(report.get("max_joint_step_rad", base_step)),
        float(report.get("actual_max_joint_step_rad", 0.0)),
        float(max_step),
    )
    try:
        predicted_execution, predicted_bridge_count = (
            _move_tracked_execution_bridge(
                np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF),
                checked,
                arm=arm,
                max_step_rad=execution_max_step,
            )
        )
    except (TypeError, ValueError) as exc:
        return {
            **report,
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": f"execution bridge preflight failed: {exc}",
        }
    selected_budget = int(report.get("budget_steps", scheduled_max_steps))
    total_budget = selected_budget + int(start_bridge_reserve)
    predicted_count = len(predicted_execution)
    if predicted_count > total_budget or predicted_count > int(max_steps):
        return {
            **report,
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": (
                "bridge-expanded execution trajectory exceeds the remaining "
                "action/time budget"
            ),
            "execution_start_joint_limit_recovery": _json_ready(
                start_limit_recovery
            ),
            "execution_bridge_reserved_waypoints": int(start_bridge_reserve),
            "predicted_execution_bridge_waypoint_count": int(
                predicted_bridge_count
            ),
            "predicted_execution_waypoint_count": int(predicted_count),
            "execution_budget_steps_total": int(total_budget),
            "execution_budget_includes_bridge": True,
            "execution_max_joint_step_rad": float(execution_max_step),
        }
    return {
        **report,
        "waypoints": checked,
        "actual_max_joint_step_rad": float(max_step),
        "execution_max_joint_step_rad": float(execution_max_step),
        "execution_step_numeric_tolerance_rad": float(
            MOVE_TRACKED_POINT_EXECUTION_STEP_NUMERIC_TOLERANCE_RAD
        ),
        "execution_start_joint_limit_recovery": _json_ready(
            start_limit_recovery
        ),
        "execution_bridge_reserved_waypoints": int(start_bridge_reserve),
        "predicted_execution_bridge_waypoint_count": int(
            predicted_bridge_count
        ),
        "predicted_execution_waypoint_count": int(predicted_count),
        "execution_budget_steps_total": int(total_budget),
        "execution_budget_includes_bridge": True,
        "local_fk_finite_checked": True,
        "visual_tracked_point_count": visual_tracked_point_count,
        "visual_speedup_cap": float(visual_speedup_cap),
    }



def _move_tracked_execution_cadence_estimate(
    *,
    start_validation: Mapping[str, Any] | None,
    preplanning_source_sync: Mapping[str, Any] | None,
    planner: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Estimate a conservative evaluator period from this request only."""

    samples: list[dict[str, Any]] = []

    def add_sample(raw_value: Any, source: str) -> None:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value < 1.0e-4 or value > 5.0:
            return
        samples.append({"source": str(source), "observation_dt_s": value})

    def add_validation(
        raw_report: Mapping[str, Any] | None,
        source_prefix: str,
    ) -> None:
        if not isinstance(raw_report, Mapping):
            return
        raw_samples = raw_report.get("handoff_observation_dt_samples_s")
        if isinstance(raw_samples, Sequence) and not isinstance(
            raw_samples, (str, bytes, bytearray)
        ):
            for index, value in enumerate(raw_samples):
                add_sample(value, f"{source_prefix}.handoff_window[{index}]")
        add_sample(
            raw_report.get("last_handoff_observation_dt_s"),
            f"{source_prefix}.last_handoff",
        )
        assessment = raw_report.get("handoff_assessment")
        if isinstance(assessment, Mapping):
            add_sample(
                assessment.get("observation_dt_s"),
                f"{source_prefix}.handoff_assessment",
            )

    add_validation(preplanning_source_sync, "preplanning_source_sync")
    add_validation(start_validation, "execution_start_validation")
    if isinstance(planner, Mapping):
        try:
            planning_elapsed_s = float(planner.get("planning_elapsed_s"))
            planning_hold_steps = int(planner.get("planning_hold_steps"))
        except (TypeError, ValueError):
            planning_elapsed_s = float("nan")
            planning_hold_steps = 0
        if (
            math.isfinite(planning_elapsed_s)
            and planning_elapsed_s > 0.0
            and planning_hold_steps > 0
        ):
            add_sample(
                planning_elapsed_s / float(planning_hold_steps),
                "planner.hold_action_average",
            )

    if not samples:
        return {
            "ok": False,
            "reason": "no_valid_observation_period_samples",
            "observation_dt_s": None,
            "max_observed_observation_dt_s": None,
            "safety_factor": float(
                MOVE_TRACKED_POINT_EXECUTION_CADENCE_SAFETY_FACTOR
            ),
            "samples": [],
        }
    max_observed = max(float(item["observation_dt_s"]) for item in samples)
    conservative = min(
        5.0,
        max_observed
        * float(MOVE_TRACKED_POINT_EXECUTION_CADENCE_SAFETY_FACTOR),
    )
    return {
        "ok": True,
        "reason": "slowest_request_local_period_with_jitter_reserve",
        "observation_dt_s": float(conservative),
        "max_observed_observation_dt_s": float(max_observed),
        "safety_factor": float(
            MOVE_TRACKED_POINT_EXECUTION_CADENCE_SAFETY_FACTOR
        ),
        "samples": _json_ready(samples),
    }



MOVE_TRACKED_POINT_LOCAL_BUILD = (
    "official_v2_live_tracked_symbolic_rigid_trajectory_v78_arm_only_safe_exec"
)



MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA = (
    "official_v2_move_tracked_point_trajectory"
)



MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION = 4



MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M = 0.05



MOVE_TRACKED_POINT_START_EEF_ANCHOR_ERROR_M = 0.03



MOVE_TRACKED_POINT_REGISTRATION_EEF_INLIER_M = 0.008



MOVE_TRACKED_POINT_REGISTRATION_EEF_MAX_RECOVERY_M = 0.030



MOVE_TRACKED_POINT_REGISTRATION_REACQUIRE_MAX_STEPS = 16



MOVE_TRACKED_POINT_REGISTRATION_REACQUIRE_MAX_S = 8.0



MOVE_TRACKED_POINT_STREAM_TRACKING_ERROR_RAD = 0.12



MOVE_TRACKED_POINT_MAX_LIVE_PROGRESS_REGRESSION_M = 0.006



MOVE_TRACKED_POINT_START_STABLE_STEPS = 3



MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS = 90



MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_STABLE_STEPS = 2



MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS = 6



MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_S = 20.0



MOVE_TRACKED_POINT_RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M = 0.003



MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS = frozenset(
    {
        "rigid_pair_rgbd_correction_exceeded",
        "measured_pair_collapsed",
        "rigid_pair_point_unobserved",
        "rigid_pair_projection_requires_fresh_confirmation",
    }
)



MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_FRESH_RETRY_REASONS = frozenset(
    {
        "rigid_pair_rgbd_correction_exceeded",
        "measured_pair_collapsed",
        "rigid_pair_point_unobserved",
        "joint_rgbd_observation_unavailable",
        "pair_candidate_unobserved",
        "rigid_pair_distance_mismatch",
        "rigid_pair_identity_ambiguous",
        "individual_candidate_gate_rejected",
        "required_observation_prior_unavailable",
        "rigid_pair_projection_requires_fresh_confirmation",
    }
)



MOVE_TRACKED_POINT_START_HANDOFF_ARM_TOLERANCE_RAD = 0.12



MOVE_TRACKED_POINT_START_HANDOFF_TRUNK_TOLERANCE_RAD = 0.08



MOVE_TRACKED_POINT_START_HANDOFF_GRIPPER_TOLERANCE_RAD = 0.02



MOVE_TRACKED_POINT_OFF_HAND_REFERENCE_TOLERANCE_M = 0.020



MOVE_TRACKED_POINT_START_HANDOFF_QVEL_HARD_RAD_S = 2.50



MOVE_TRACKED_POINT_START_HANDOFF_J8_QVEL_HARD_RAD_S = 5.00



MOVE_TRACKED_POINT_START_HANDOFF_TRUNK_QVEL_HARD_RAD_S = 0.50



MOVE_TRACKED_POINT_START_HANDOFF_BASE_QVEL_HARD = 0.20



MOVE_TRACKED_POINT_START_HANDOFF_POSITION_STEP_RAD = 0.02



MOVE_TRACKED_POINT_START_ARM_QVEL_RAD_S = 0.15



MOVE_TRACKED_POINT_FINAL_MAX_STABILITY_STEPS = 90



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_SOFT_LIMIT_M = 0.006



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_RELEASE_M = 0.003



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_HARD_LIMIT_M = 0.010



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_MAX_STEPS = 8



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_STABLE_STEPS = 2



MOVE_TRACKED_POINT_GRIPPER_RECOVERY_QVEL_MPS = 0.02



MOVE_TRACKED_POINT_EXECUTION_MAX_SPEEDUP_FACTOR = 12.0



MOVE_TRACKED_POINT_EXECUTION_PREFERRED_MAX_STEPS = 8



MOVE_TRACKED_POINT_EXECUTION_MAX_JOINT_CORRIDOR_DEVIATION_RAD = 0.03



MOVE_TRACKED_POINT_EXECUTION_SAFETY_TIME_STEPS = 2



MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS = 3



MOVE_TRACKED_POINT_NPOINT_VISUAL_MAX_SPEEDUP_FACTOR = 5.0



MOVE_TRACKED_POINT_NPOINT_REACQUISITION_MAX_HOLD_STEPS = 12



MOVE_TRACKED_POINT_EXECUTION_STEP_NUMERIC_TOLERANCE_RAD = 1.0e-8



MOVE_TRACKED_POINT_EXECUTION_CADENCE_SAFETY_FACTOR = 1.25



MOVE_TRACKED_POINT_SUCCESSOR_RETENTION_EXTRA_S = 180.0



MOVE_TRACKED_POINT_STREAM_RECOVERY_ERROR_RAD = 0.22



MOVE_TRACKED_POINT_STREAM_MAX_RECOVERY_STEPS = 3



MOVE_TRACKED_POINT_FINAL_RAW_QVEL_HARD_RAD_S = 0.15



MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S = 0.50



MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS = 3



MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD = 0.002



MOVE_TRACKED_POINT_FINAL_COHERENCE_MAX_DT_S = 5.0



MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RESIDUAL_RAD = 0.02



MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RAW_QVEL_HARD_RAD_S = 0.50



MOVE_TRACKED_POINT_FINAL_LOAD_STABLE_STEPS = 3



MOVE_TRACKED_POINT_FINAL_LOAD_MIN_RESIDUAL_RAD = 0.004



MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUNDS = 3



MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD = 0.06



MOVE_TRACKED_POINT_FINAL_LOAD_MAX_TOTAL_RAD = 0.12



MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_POINTS = 2



MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ROUNDS = 2



MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ANCHOR_REBASE_M = 0.030



MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M = 0.050



MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_TRANSLATION_M = 0.080



MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_PLANNING_S = 12.0



MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_EXECUTION_RESERVE_S = 4.0



MOVE_TRACKED_POINT_FINAL_OCCLUSION_MIN_DEPTH_MARGIN_M = 0.002



MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_RADIUS_PX = 2



MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M = 0.008



MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M = 0.006



MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS = 12



MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_ROUNDS = 2



MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M = 0.030



MOVE_TRACKED_POINT_POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S = 4.0



MOVE_TRACKED_POINT_PLANNING_FRACTION = 0.60



MOVE_TRACKED_POINT_MIN_EXECUTION_RESERVE_S = 0.05



def _move_tracked_planning_budget(timeout_s: float) -> float:
    """Return a bounded local-planning budget inside the total timeout."""

    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("move-tracked timeout_s must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("move-tracked timeout_s must be finite and positive")
    reserve = min(
        max(float(MOVE_TRACKED_POINT_MIN_EXECUTION_RESERVE_S), 0.01),
        0.25 * timeout,
    )
    budget = min(
        float(MOVE_TRACKED_POINT_PLANNING_FRACTION) * timeout,
        timeout - reserve,
    )
    return max(0.01, min(timeout, budget))



def _move_tracked_trim_visual_correction_to_first_feasible(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    endpoint: Mapping[str, Any],
    path_plan: Mapping[str, Any],
    target_points: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None,
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None,
    fixed_target_points: Sequence[Mapping[str, Any]] | None,
    pos_tol_m: float,
    ori_tol_deg: float,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stop a feedback correction at its first fully feasible FK sample.

    An underconstrained endpoint solver may return a distant member of the
    feasible pose family even when only a millimetric correction is needed.
    The locally verified path already provides a deterministic ordering from
    the observed pose to that endpoint.  Re-evaluating the immutable public
    constraints at every FK sample yields the nearest feasible prefix without
    adding a request-specific analytic branch or querying simulator state.
    """

    dense_waypoints_raw = path_plan.get("waypoints")
    if not isinstance(dense_waypoints_raw, list) or not dense_waypoints_raw:
        raise ValueError("final visual correction path is empty")
    controlled_targets = [dict(item) for item in target_points]
    fixed_targets = (
        None
        if fixed_points_robot_base_m is None
        else [dict(item) for item in (fixed_target_points or [])]
    )
    anchors = _move_tracked_strict_point_matrix(
        endpoint.get("anchors_eef_m"),
        expected_count=len(controlled_targets),
        label="final visual correction EEF anchors",
    )
    start_q = _require_serialized_j8_zero(
        q_start,
        label=f"{arm} final visual correction trim start q",
    )
    start_position, start_quaternion = _local_eef_pose(
        state, arm, start_q
    )
    start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
    start_quaternion = np.asarray(
        start_quaternion, dtype=np.float64
    ).reshape(4)

    selected: dict[str, Any] | None = None
    validated_waypoints: list[np.ndarray] = []
    for waypoint_index, raw_q in enumerate(dense_waypoints_raw):
        q = _require_serialized_j8_zero(
            raw_q,
            label=(
                f"{arm} final visual correction feasibility waypoint "
                f"{waypoint_index + 1}"
            ),
        )
        validated_waypoints.append(q)
        position, quaternion = _local_eef_pose(state, arm, q)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
        points = position[None, :] + (
            _quat_to_mat_xyzw(quaternion) @ anchors.T
        ).T
        constraints = _tracked_target_constraints(
            points,
            controlled_targets,
            tolerance_m=float(pos_tol_m),
            relations=[dict(item) for item in (relations or [])],
            inequalities=[dict(item) for item in (inequalities or [])],
            orientation_tolerance_deg=float(ori_tol_deg),
            fixed_points_robot_base_m=fixed_points_robot_base_m,
            fixed_target_points=fixed_targets,
        )
        if bool(constraints.get("ok")):
            selected = {
                "index": int(waypoint_index),
                "q": q,
                "position": position,
                "quaternion": quaternion,
                "points": points,
                "constraints": constraints,
            }
            break
    if selected is None:
        raise ValueError(
            "final visual correction path contains no waypoint satisfying "
            "the immutable goal"
        )

    selected_index = int(selected["index"])
    retained_waypoints = validated_waypoints[: selected_index + 1]
    selected_q = np.asarray(selected["q"], dtype=np.float64).reshape(ARM_DOF)
    selected_position = np.asarray(
        selected["position"], dtype=np.float64
    ).reshape(3)
    selected_quaternion = np.asarray(
        selected["quaternion"], dtype=np.float64
    ).reshape(4)
    selected_points = np.asarray(
        selected["points"], dtype=np.float64
    ).reshape(len(controlled_targets), 3)
    joint_path = np.vstack([start_q, *retained_waypoints])
    joint_deltas = np.diff(joint_path, axis=0)
    actual_max_joint_step = float(
        np.max(np.abs(joint_deltas)) if joint_deltas.size else 0.0
    )
    actual_joint_path_length = float(
        np.sum(np.max(np.abs(joint_deltas), axis=1))
        if joint_deltas.size
        else 0.0
    )
    original_endpoint_position = np.asarray(
        endpoint.get("final_eef_position_m"), dtype=np.float64
    ).reshape(3)
    original_translation = float(
        np.linalg.norm(original_endpoint_position - start_position)
    )
    selected_translation = float(
        np.linalg.norm(selected_position - start_position)
    )

    trimmed_endpoint = dict(endpoint)
    trimmed_endpoint.update(
        {
            "q_final": selected_q,
            "resolved_points_robot_base_m": selected_points,
            "final_eef_position_m": selected_position,
            "final_eef_quaternion_xyzw": selected_quaternion,
            "constraints": selected["constraints"],
            "visual_correction_first_feasible_waypoint_index": selected_index,
            "visual_correction_original_endpoint_q": _json_ready(
                endpoint.get("q_final")
            ),
        }
    )
    trimmed_path = dict(path_plan)
    trimmed_path.update(
        {
            "waypoints": [q.astype(float).tolist() for q in retained_waypoints],
            "joint_waypoint_count": int(len(retained_waypoints)),
            "translation_m": selected_translation,
            "orientation_deg": float(
                orientation_error_deg(start_quaternion, selected_quaternion)
            ),
            "actual_max_joint_step_rad": actual_max_joint_step,
            "joint_path_length_inf_rad": actual_joint_path_length,
            "visual_correction_prefix_trimmed": bool(
                selected_index + 1 < len(dense_waypoints_raw)
            ),
            "visual_correction_original_dense_waypoint_count": int(
                len(dense_waypoints_raw)
            ),
            "visual_correction_first_feasible_waypoint_index": selected_index,
        }
    )
    return {
        "endpoint": trimmed_endpoint,
        "path_plan": trimmed_path,
        "report": {
            "applied": bool(selected_index + 1 < len(dense_waypoints_raw)),
            "original_dense_waypoint_count": int(len(dense_waypoints_raw)),
            "first_feasible_waypoint_index": selected_index,
            "retained_dense_waypoint_count": int(len(retained_waypoints)),
            "discarded_dense_waypoint_count": int(
                len(dense_waypoints_raw) - len(retained_waypoints)
            ),
            "original_endpoint_translation_m": original_translation,
            "selected_endpoint_translation_m": selected_translation,
            "selected_endpoint_orientation_deg": float(
                orientation_error_deg(start_quaternion, selected_quaternion)
            ),
            "selected_constraints": _json_ready(selected["constraints"]),
            "selected_points_robot_base_m": selected_points.astype(float).tolist(),
            "selection_rule": "first_complete_constraint_feasible_fk_waypoint",
        },
    }



def _move_tracked_common_result() -> dict[str, Any]:
    return {
        "reload_generation": int(_OFFICIAL_RELOAD_GENERATION),
        "official_v2_stack_digest": _OFFICIAL_STACK_DIGEST,
        "execution": "official_action_only",
        "observation_contract": [
            "current_evaluator_head_rgb",
            "current_evaluator_depth_linear",
            "current_evaluator_camera_relative_pose",
            "same_observation_evaluator_proprioception",
            "subsequent_evaluator_proprioception",
            "submission_local_static_robot_model",
            "policy_owned_tracked_point_state",
        ],
        "direct_simulator_mutation": False,
        "global_pose_available": False,
        "scene_geometry_truth_available": False,
        "collision_truth_available": False,
        "contact_truth_used": False,
        "grasp_truth_used": False,
        "segmentation_used": False,
        "object_identity_verified": False,
    }



def _move_tracked_failure(
    ctx,
    *,
    stage: str,
    error: str,
    args: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    _set_result(
        ctx,
        "move_tracked_point",
        {
            **_move_tracked_common_result(),
            "ok": False,
            "error": str(error),
            "failure_stage": str(stage),
            "request": _json_ready(dict(args or {})),
            **_json_ready(dict(details or {})),
        },
    )



def _move_tracked_live_snapshot(
    manager,
    names: list[str],
    *,
    episode_id: str,
) -> dict[str, Any]:
    snapshot_fn = getattr(manager, "observed_active_points_snapshot", None)
    if not callable(snapshot_fn):
        raise RuntimeError(
            "official dynamic distance tracker does not expose active atomic snapshots"
        )
    try:
        snapshot = snapshot_fn(names, episode_id=episode_id)
    except Exception as exc:
        # Tracker implementations are policy-owned observation adapters.  A
        # transient adapter/provider failure must become an observation result
        # that callers can handle (and, during execution, a structured
        # tracking failure), rather than escaping through the generator and
        # bypassing the normal measured-hold path.
        return {
            "ok": False,
            "reason": "tracked_point_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "entries": {},
            "unavailable": {
                str(name): "observation_provider_error" for name in names
            },
            "observation_sequence": None,
            "episode_id": str(episode_id),
        }
    if not isinstance(snapshot, dict):
        return {
            "ok": False,
            "reason": "tracked_point_invalid",
            "error": "active atomic snapshot must be an object",
            "entries": {},
            "unavailable": {
                str(name): "invalid_snapshot" for name in names
            },
            "observation_sequence": None,
            "episode_id": str(episode_id),
        }
    return snapshot



def _move_tracked_visual_point_count(manager, names: Sequence[str]) -> int:
    count = getattr(manager, "visual_point_count", None)
    if callable(count):
        return int(count(names))
    return len(names)



class _MoveTrackedPointSourceManager:
    """Merge visual tracker rows with live submission-local EEF points.

    The wrapper deliberately preserves the tracker snapshot interface used by
    the mature move executor. Predefined rows are recomputed from the same
    evaluator proprioception sequence on every call, so they never need a
    visual registration and never become a hidden observation channel.
    """

    def __init__(
        self,
        ctx,
        tracker,
        *,
        predefined_names: Sequence[str],
        episode_id: str,
        arm: str = "right",
    ) -> None:
        self._ctx = ctx
        self._tracker = tracker
        self._predefined_names = frozenset(str(name) for name in predefined_names)
        self._episode_id = str(episode_id)
        self._arm = "right"
        self.set_arm(arm)
        token = hashlib.sha256(self._episode_id.encode("utf-8")).hexdigest()[:16]
        self._session_id = f"eef_points_{token}"
        self._image_id = "current_proprio_fk"
        self._synthetic_leases: set[str] = set()

    @property
    def predefined_names(self) -> frozenset[str]:
        return self._predefined_names

    def set_arm(self, arm: str) -> None:
        side = str(arm).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("predefined EEF point arm must be left or right")
        self._arm = side

    def visual_names(self, names: Sequence[str]) -> list[str]:
        return [
            str(name)
            for name in names
            if str(name) not in self._predefined_names
        ]

    def visual_point_count(self, names: Sequence[str]) -> int:
        return len(self.visual_names(names))

    def observed_active_points_snapshot(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        requested = [str(name) for name in names]
        if str(episode_id) != self._episode_id:
            return {
                "ok": False,
                "reason": "episode_changed",
                "entries": {},
                "unavailable": {name: "episode_changed" for name in requested},
                "observation_sequence": None,
                "episode_id": str(episode_id),
            }
        adapter, sequence = _adjust_adapter(self._ctx)
        visual_names = self.visual_names(requested)
        if visual_names:
            if self._tracker is None:
                return {
                    "ok": False,
                    "reason": "tracked_point_provider_unavailable",
                    "entries": {},
                    "unavailable": {
                        name: "tracked_point_provider_unavailable"
                        for name in visual_names
                    },
                    "observation_sequence": sequence,
                    "episode_id": self._episode_id,
                }
            base = self._tracker.observed_active_points_snapshot(
                visual_names,
                episode_id=self._episode_id,
            )
            if not isinstance(base, Mapping):
                return {
                    "ok": False,
                    "reason": "tracked_point_invalid",
                    "entries": {},
                    "unavailable": {name: "invalid_snapshot" for name in visual_names},
                    "observation_sequence": sequence,
                    "episode_id": self._episode_id,
                }
            base = dict(base)
            session_id = str(base.get("session_id") or "").strip()
            image_id = str(base.get("image_id") or "").strip()
            if session_id:
                self._session_id = session_id
            if image_id:
                self._image_id = image_id
        else:
            base = {
                "ok": True,
                "reason": None,
                "entries": {},
                "unavailable": {},
                "observation_sequence": sequence,
                "episode_id": self._episode_id,
            }

        entries = dict(base.get("entries") or {})
        predefined_requested = [
            name for name in requested if name in self._predefined_names
        ]
        if predefined_requested:
            kinematics = _adjust_kinematic_observation(self._ctx, self._arm)
            points = predefined_eef_points_robot_base_m(
                predefined_requested,
                eef_position_robot_base_m=kinematics["position"],
                eef_quaternion_xyzw=kinematics["quaternion"],
                gripper_q_m=kinematics["gripper"],
            )
            for name in predefined_requested:
                entries[name] = {
                    "name": name,
                    "status": "observed",
                    "xyz_in_robot_base_coord_m": points[name].astype(float).tolist(),
                    "xyz_source": (
                        "submission_local_static_eef_geometry_and_current_"
                        "evaluator_proprio_fk"
                    ),
                    "measurement_source": (
                        "submission_local_static_eef_geometry_and_current_"
                        "evaluator_proprio_fk"
                    ),
                    "point_source_kind": "predefined_eef_point",
                    "predefined_eef_point": True,
                    "predefined_eef_local_m": predefined_eef_point_local_m(
                        name, kinematics["gripper"]
                    ).astype(float).tolist(),
                    "arm": self._arm,
                    "observation_sequence": int(sequence),
                    "episode_id": self._episode_id,
                }
        sequence_after, _received_ts = adapter.observation_metadata()
        if int(sequence_after) != int(sequence):
            return {
                "ok": False,
                "reason": "tracker_observation_mismatch",
                "entries": {},
                "unavailable": {
                    name: "observation_changed_while_resolving_predefined_eef_point"
                    for name in requested
                },
                "observation_sequence": int(sequence_after),
                "episode_id": self._episode_id,
            }

        unavailable = dict(base.get("unavailable") or {})
        missing = [name for name in requested if name not in entries]
        for name in missing:
            unavailable.setdefault(name, "tracked_point_unavailable")
        base_ok = bool(base.get("ok"))
        try:
            base_sequence = int(base.get("observation_sequence", sequence))
        except (TypeError, ValueError):
            base_sequence = -1
        synchronized = base_sequence == int(sequence)
        ok = bool(base_ok and synchronized and not missing)
        return {
            **base,
            "ok": ok,
            "reason": (
                None
                if ok
                else str(
                    base.get("reason")
                    or ("tracker_observation_mismatch" if not synchronized else "tracked_point_unavailable")
                )
            ),
            "entries": {name: entries[name] for name in requested if name in entries},
            "unavailable": unavailable,
            "requested_names": requested,
            "observation_sequence": int(sequence),
            "episode_id": self._episode_id,
            "session_id": str(base.get("session_id") or self._session_id),
            "image_id": str(base.get("image_id") or self._image_id),
            "measurement_source": (
                "hybrid_visual_rgbd_and_submission_local_eef_fk"
                if visual_names and predefined_requested
                else "submission_local_static_eef_geometry_and_current_evaluator_proprio_fk"
                if predefined_requested
                else base.get("measurement_source")
            ),
            "predefined_eef_point_names": predefined_requested,
            "visual_tracked_point_names": visual_names,
            "predefined_eef_arm": self._arm if predefined_requested else None,
            "predictions_published": False,
        }

    def begin_motion_retention(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        visual_names = self.visual_names(names)
        if visual_names:
            if self._tracker is None:
                return {"ok": False, "reason": "tracker_unavailable", "lease_id": None}
            return self._tracker.begin_motion_retention(
                visual_names,
                episode_id=episode_id,
                timeout_s=timeout_s,
            )
        lease_id = f"predefined_eef_{secrets.token_hex(8)}"
        self._synthetic_leases.add(lease_id)
        return {
            "ok": True,
            "lease_id": lease_id,
            "names": [],
            "predefined_eef_point_names": [str(name) for name in names],
            "duration_s": float(timeout_s),
            "reason": "no_visual_tracking_required",
        }

    def end_motion_retention(self, lease_id: str) -> dict[str, Any]:
        token = str(lease_id)
        if token in self._synthetic_leases:
            self._synthetic_leases.remove(token)
            return {
                "ok": True,
                "released": True,
                "lease_id": token,
                "reason": "predefined_eef_lease_released",
            }
        if self._tracker is None:
            return {"ok": False, "released": False, "lease_id": token, "reason": "tracker_unavailable"}
        return self._tracker.end_motion_retention(token)

    def activate_rigid_points(
        self, names: Sequence[str], *, episode_id: str
    ) -> dict[str, Any]:
        visual_names = self.visual_names(names)
        if len(visual_names) < 2:
            return {
                "ok": True,
                "active": False,
                "reason": "predefined_eef_points_are_exact_local_fk_anchors",
                "requested_names": [str(name) for name in names],
                "activated_names": [],
                "visual_tracked_point_names": visual_names,
            }
        activate_all = getattr(self._tracker, "activate_rigid_points", None)
        if callable(activate_all):
            return activate_all(visual_names, episode_id=episode_id)
        if len(visual_names) == 2:
            activate_pair = getattr(self._tracker, "activate_rigid_pair", None)
            if callable(activate_pair):
                return activate_pair(visual_names, episode_id=episode_id)
        return {
            "ok": True,
            "active": False,
            "reason": "tracker_has_no_n_point_fusion; local_geometry_validation_only",
            "requested_names": [str(name) for name in names],
            "activated_names": [],
            "visual_tracked_point_names": visual_names,
        }

    def activate_rigid_pair(
        self, names: Sequence[str], *, episode_id: str
    ) -> dict[str, Any]:
        return self.activate_rigid_points(names, episode_id=episode_id)

    def activate_on_hand_eef_observation_prior(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
        session_id: str,
        image_id: str,
        arm: str,
        anchors_eef_m: Any,
    ) -> dict[str, Any]:
        requested = [str(name) for name in names]
        visual_names = self.visual_names(requested)
        if not visual_names:
            return {
                "ok": True,
                "active": False,
                "reason": "predefined_eef_points_use_direct_local_fk",
                "names": requested,
                "activation_observation_sequence": _adjust_adapter(self._ctx)[1],
            }
        anchors = np.asarray(anchors_eef_m, dtype=np.float64)
        indices = [requested.index(name) for name in visual_names]
        activate = getattr(
            self._tracker, "activate_on_hand_eef_observation_prior", None
        )
        if not callable(activate):
            return {
                "ok": True,
                "active": False,
                "reason": "tracker_provider_has_no_eef_observation_prior",
                "names": visual_names,
            }
        return activate(
            visual_names,
            episode_id=episode_id,
            session_id=session_id,
            image_id=image_id,
            arm=arm,
            anchors_eef_m=anchors[indices],
        )

    def status(self) -> dict[str, Any]:
        status = getattr(self._tracker, "status", None)
        return status() if callable(status) else {"active_rigid_pair": None}

    def deactivate_rigid_pair(self, *args, **kwargs):
        deactivate = getattr(self._tracker, "deactivate_rigid_pair", None)
        if callable(deactivate):
            return deactivate(*args, **kwargs)
        return {"ok": True, "active": False, "reason": "no_visual_rigid_pair"}

    def deactivate_on_hand_eef_observation_prior(self, *args, **kwargs):
        deactivate = getattr(
            self._tracker, "deactivate_on_hand_eef_observation_prior", None
        )
        if callable(deactivate):
            return deactivate(*args, **kwargs)
        return {"ok": True, "active": False, "reason": "no_visual_eef_prior"}

    def __getattr__(self, name: str):
        tracker = object.__getattribute__(self, "_tracker")
        if tracker is None:
            raise AttributeError(name)
        return getattr(tracker, name)



def _move_tracked_select_predefined_eef_arm(
    *,
    points: Sequence[Mapping[str, Any]],
    roles_by_name: Mapping[str, str],
    observations: Mapping[str, Mapping[str, Any]],
    visual_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Select which gripper owns generic predefined point names."""

    predefined_names = [
        str(point["name"])
        for point in points
        if is_predefined_eef_point_name(point.get("name"))
    ]
    dynamic_on_hand = [
        str(point["name"])
        for point in points
        if roles_by_name.get(str(point["name"])) == "on_hand"
        and not is_predefined_eef_point_name(point.get("name"))
        and str(point["name"]) in visual_entries
    ]
    fixed_coordinates: list[tuple[str, int, float]] = []
    for point in points:
        name = str(point["name"])
        if name not in predefined_names:
            continue
        target = point.get("target_xyz_m")
        if not isinstance(target, Mapping):
            continue
        for axis_index, axis in enumerate(("x", "y", "z")):
            value = target.get(axis)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fixed_coordinates.append((name, axis_index, float(value)))
    off_hand_points = []
    for point in points:
        name = str(point["name"])
        if roles_by_name.get(name) != "off_hand" or name not in visual_entries:
            continue
        try:
            xyz = np.asarray(
                visual_entries[name]["xyz_in_robot_base_coord_m"], dtype=np.float64
            ).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(xyz)):
            off_hand_points.append(xyz)

    score_by_arm: dict[str, float] = {}
    basis = "deterministic_right_default"
    point_positions_by_arm: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        observation = observations[side]
        point_positions_by_arm[side] = predefined_eef_points_robot_base_m(
            predefined_names,
            eef_position_robot_base_m=observation["position"],
            eef_quaternion_xyzw=observation["quaternion"],
            gripper_q_m=observation["gripper"],
        )
        if dynamic_on_hand:
            basis = "nearest_to_visual_on_hand_points"
            score = float(np.mean([
                np.linalg.norm(
                    np.asarray(visual_entries[name]["xyz_in_robot_base_coord_m"], dtype=np.float64)
                    - np.asarray(observation["position"], dtype=np.float64)
                )
                for name in dynamic_on_hand
            ]))
        elif fixed_coordinates:
            basis = "minimum_fixed_coordinate_displacement"
            score = float(np.mean([
                abs(point_positions_by_arm[side][name][axis] - target)
                for name, axis, target in fixed_coordinates
            ]))
        elif off_hand_points:
            basis = "nearest_to_off_hand_reference_geometry"
            predefined_centroid = np.mean(
                np.vstack(list(point_positions_by_arm[side].values())), axis=0
            )
            score = float(np.min(
                np.linalg.norm(np.vstack(off_hand_points) - predefined_centroid, axis=1)
            ))
        else:
            score = 0.0 if side == "right" else 1.0
        score_by_arm[side] = score
    arm = min(("left", "right"), key=lambda side: (score_by_arm[side], side != "right"))
    return arm, {
        "selected_arm": arm,
        "selection_basis": basis,
        "score_by_arm": score_by_arm,
        "predefined_eef_point_names": predefined_names,
        "visual_on_hand_point_names": dynamic_on_hand,
        "off_hand_reference_count": len(off_hand_points),
    }



def _move_tracked_occluded_on_hand_evidence(
    live: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    """Validate that a missing on-hand marker is behind a nearer RGB-D layer."""

    point_name = str(name)
    fusion = live.get("rigid_pair_fusion")
    if not isinstance(fusion, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "reason": "rigid_pair_occlusion_report_unavailable",
        }
    visual = fusion.get("visual_tracker_pair")
    if not isinstance(visual, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "reason": "visual_pair_occlusion_report_unavailable",
        }
    names = [str(value) for value in fusion.get("names", [])]
    track_ids = [str(value) for value in visual.get("track_ids", [])]
    if point_name not in names or len(names) != len(track_ids):
        return {
            "ok": False,
            "name": point_name,
            "reason": "occluded_point_not_bound_to_rigid_pair",
        }
    track_id = track_ids[names.index(point_name)]
    per_point = visual.get("per_point")
    detail = per_point.get(track_id) if isinstance(per_point, Mapping) else None
    if not isinstance(detail, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "track_id": track_id,
            "reason": "occluded_point_diagnostics_unavailable",
        }

    candidates: list[dict[str, Any]] = []
    for raw_candidate in detail.get("rejected_candidates_before_acceptance", []):
        if not isinstance(raw_candidate, Mapping):
            continue
        if str(raw_candidate.get("reason") or "") != "eef_prior_depth_layer_mismatch":
            continue
        try:
            candidate_depth = float(raw_candidate["candidate_depth_m"])
            predicted_depth = float(raw_candidate["predicted_depth_m"])
            pixel_error = float(raw_candidate["pixel_error_px"])
            pixel_limit = float(raw_candidate["pixel_error_limit_px"])
        except (KeyError, TypeError, ValueError):
            continue
        values = (candidate_depth, predicted_depth, pixel_error, pixel_limit)
        if not all(math.isfinite(value) for value in values):
            continue
        depth_margin = predicted_depth - candidate_depth
        candidate = {
            "candidate_method": str(raw_candidate.get("candidate_method") or ""),
            "candidate_status": str(raw_candidate.get("candidate_status") or ""),
            "candidate_depth_m": candidate_depth,
            "predicted_depth_m": predicted_depth,
            "foreground_depth_margin_m": depth_margin,
            "pixel_error_px": pixel_error,
            "pixel_error_limit_px": pixel_limit,
            "ok": bool(
                depth_margin
                >= MOVE_TRACKED_POINT_FINAL_OCCLUSION_MIN_DEPTH_MARGIN_M
                and pixel_error <= pixel_limit
            ),
        }
        candidates.append(candidate)
    accepted = next((item for item in candidates if item["ok"]), None)
    return {
        "ok": accepted is not None,
        "name": point_name,
        "track_id": track_id,
        "status": str(detail.get("status") or ""),
        "reason": (
            None
            if accepted is not None
            else "nearer_rgbd_occluder_not_observed_at_predicted_pixel"
        ),
        "accepted_candidate": accepted,
        "candidate_count": len(candidates),
        "prediction_values_published_as_observation": False,
    }



def _move_tracked_eef_anchor_depth_reobservation(
    adapter,
    predicted_points_robot_base_m: Any,
    *,
    current_sequence: int,
) -> dict[str, Any]:
    """Measure EEF-rigid anchors at their current projected RGB-D pixels.

    Local FK supplies only the expected pixel and depth layer.  Every returned
    XYZ is independently back-projected from the current evaluator
    ``depth_linear`` sample.  The complete bundle is accepted atomically so a
    successful point cannot be mixed with a stale feature-track mate.
    """

    report: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "observation_sequence": int(current_sequence),
        "camera_role": "head",
        "source": (
            "current_evaluator_depth_linear_at_local_fk_projected_anchor"
        ),
        "prediction_values_published_as_observation": False,
        "authoritative_for_terminal_success": False,
        "reacquisition_diagnostic_only": True,
        "per_point": [],
    }
    if adapter is None:
        report["reason"] = "official_evaluator_adapter_unavailable"
        return report
    try:
        expected = _move_tracked_strict_point_matrix(
            predicted_points_robot_base_m,
            expected_count=len(predicted_points_robot_base_m),
            label="EEF anchor depth reobservation predictions",
        )
        sequence_before = int(adapter.status().get("sequence", -1))
        rgb = adapter.camera_frame("head")
        depth = adapter.camera_depth_frame("head")
        relative_pose = dict(adapter.camera_relative_poses().get("head") or {})
        sequence_after = int(adapter.status().get("sequence", -1))
    except (AttributeError, TypeError, ValueError) as exc:
        report.update(
            {
                "reason": "evaluator_rgbd_snapshot_unavailable",
                "error": str(exc),
            }
        )
        return report
    report.update(
        {
            "sequence_before": sequence_before,
            "sequence_after": sequence_after,
        }
    )
    if not (
        sequence_before == int(current_sequence)
        and sequence_after == int(current_sequence)
    ):
        report["reason"] = "evaluator_rgbd_sequence_mismatch"
        return report
    if rgb is None or depth is None:
        report["reason"] = "evaluator_head_rgbd_unavailable"
        return report
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth, dtype=np.float64).squeeze()
    if (
        rgb_array.ndim < 2
        or depth_array.ndim != 2
        or tuple(rgb_array.shape[:2]) != tuple(depth_array.shape)
    ):
        report["reason"] = "evaluator_head_rgbd_shape_mismatch"
        return report
    if "pos" not in relative_pose or "quat" not in relative_pose:
        report["reason"] = "evaluator_head_camera_relative_pose_unavailable"
        return report

    height, width = depth_array.shape
    intrinsics = head_camera_intrinsics(width, height)
    try:
        camera_to_robot = transform_from_pose(
            relative_pose["pos"], relative_pose["quat"]
        )
        robot_to_camera = np.linalg.inv(camera_to_robot)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        report.update(
            {
                "reason": "evaluator_head_camera_relative_pose_invalid",
                "error": str(exc),
            }
        )
        return report

    measured_points: list[np.ndarray] = []
    radius = int(MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_RADIUS_PX)
    for point_index, predicted in enumerate(expected):
        homogeneous = np.append(predicted, 1.0)
        point_camera = (robot_to_camera @ homogeneous)[:3]
        predicted_depth = -float(point_camera[2])
        point_report: dict[str, Any] = {
            "index": int(point_index),
            "predicted_point_robot_base_m": predicted.astype(float).tolist(),
            "predicted_depth_m": predicted_depth,
            "ok": False,
        }
        if not math.isfinite(predicted_depth) or predicted_depth <= 1.0e-6:
            point_report["reason"] = "predicted_anchor_behind_camera"
            report["per_point"].append(point_report)
            continue
        predicted_pixel = np.asarray(
            [
                intrinsics.cx
                + intrinsics.fx * float(point_camera[0]) / predicted_depth,
                intrinsics.cy
                - intrinsics.fy * float(point_camera[1]) / predicted_depth,
            ],
            dtype=np.float64,
        )
        point_report["predicted_pixel"] = predicted_pixel.astype(float).tolist()
        center_x = int(round(float(predicted_pixel[0])))
        center_y = int(round(float(predicted_pixel[1])))
        if not (0 <= center_x < width and 0 <= center_y < height):
            point_report["reason"] = "predicted_anchor_out_of_view"
            report["per_point"].append(point_report)
            continue

        candidates: list[tuple[float, float, np.ndarray, int, int, float]] = []
        for pixel_y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for pixel_x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                depth_m = float(depth_array[pixel_y, pixel_x])
                if not math.isfinite(depth_m) or depth_m <= 1.0e-6:
                    continue
                camera_point = np.asarray(
                    [
                        (float(pixel_x) - intrinsics.cx)
                        / intrinsics.fx
                        * depth_m,
                        -(float(pixel_y) - intrinsics.cy)
                        / intrinsics.fy
                        * depth_m,
                        -depth_m,
                    ],
                    dtype=np.float64,
                )
                measured = (camera_to_robot @ np.append(camera_point, 1.0))[:3]
                if not np.all(np.isfinite(measured)):
                    continue
                depth_error = abs(depth_m - predicted_depth)
                point_error = float(np.linalg.norm(measured - predicted))
                pixel_error = float(
                    np.linalg.norm(
                        np.asarray([pixel_x, pixel_y], dtype=np.float64)
                        - predicted_pixel
                    )
                )
                candidates.append(
                    (
                        point_error,
                        depth_error,
                        measured,
                        pixel_x,
                        pixel_y,
                        pixel_error,
                    )
                )
        if not candidates:
            point_report["reason"] = "no_valid_depth_near_predicted_anchor"
            report["per_point"].append(point_report)
            continue
        candidate = min(candidates, key=lambda item: (item[0], item[1], item[5]))
        point_error, depth_error, measured, pixel_x, pixel_y, pixel_error = candidate
        point_report.update(
            {
                "measured_pixel": [int(pixel_x), int(pixel_y)],
                "measured_depth_m": float(depth_array[pixel_y, pixel_x]),
                "measured_point_robot_base_m": measured.astype(float).tolist(),
                "pixel_error_px": pixel_error,
                "depth_error_m": depth_error,
                "point_error_m": point_error,
                "max_depth_error_m": float(
                    MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M
                ),
                "max_point_error_m": float(
                    MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M
                ),
                "ok": bool(
                    depth_error
                    <= MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M
                    and point_error
                    <= MOVE_TRACKED_POINT_FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M
                ),
            }
        )
        if not point_report["ok"]:
            point_report["reason"] = "predicted_anchor_depth_surface_mismatch"
        else:
            point_report["reason"] = None
            measured_points.append(measured)
        report["per_point"].append(point_report)

    if len(measured_points) != len(expected):
        report["reason"] = "complete_eef_anchor_depth_bundle_not_observed"
        return report
    report.update(
        {
            "ok": True,
            "reason": None,
            "points_robot_base_m": np.vstack(measured_points),
            "point_count": int(len(measured_points)),
            "patch_radius_px": radius,
        }
    )
    return report



def _move_tracked_final_observation_bundle(
    live: Mapping[str, Any],
    *,
    controlled_names: Sequence[str],
    fixed_names: Sequence[str],
    current_observation: Mapping[str, Any],
    anchors_eef: np.ndarray,
    current_sequence: int,
    anchor_tolerance_m: float,
    pos_tol_m: float,
    adapter=None,
) -> dict[str, Any]:
    """Build a synchronized final bundle without treating FK as RGB-D truth.

    A two-marker on-hand bundle can naturally self-occlude at contact.  When
    exactly one marker remains measured and the rejected RGB-D candidate is a
    demonstrably nearer foreground layer, complete only the hidden marker from
    its frozen EEF anchor and current evaluator proprioception.  If a complete
    feature-tracked bundle instead drifts outside the strict anchor envelope,
    FK may nominate pixels for a depth diagnostic, but an unlabelled surface
    at those pixels is never promoted to marker identity or terminal success.
    Fixed/off-hand points are never completed or reobserved this way.
    """

    controlled = [str(name) for name in controlled_names]
    fixed = [str(name) for name in fixed_names]
    predefined_controlled = [
        name for name in controlled if is_predefined_eef_point_name(name)
    ]
    report: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "mode": "unavailable",
        "controlled_names": controlled,
        "fixed_names": fixed,
        "measured_controlled_names": [],
        "predefined_eef_controlled_names": predefined_controlled,
        "fk_completed_controlled_names": [],
        "depth_reobserved_controlled_names": [],
        "eef_anchor_depth_reobservation": {
            "ok": False,
            "applicable": False,
            "reason": "not_requested",
            "prediction_values_published_as_observation": False,
            "authoritative_for_terminal_success": False,
            "reacquisition_diagnostic_only": True,
        },
        "point_sources": {},
        "prediction_values_published_as_observation": False,
    }
    try:
        tracker_sequence = int(live.get("observation_sequence", -1))
    except (TypeError, ValueError):
        tracker_sequence = -1
    report["tracker_observation_sequence"] = tracker_sequence
    if tracker_sequence != int(current_sequence):
        report["reason"] = "tracker_observation_mismatch"
        return report

    entries = live.get("entries")
    if not isinstance(entries, Mapping):
        report["reason"] = "tracked_point_invalid"
        return report
    try:
        position = np.asarray(
            current_observation["position"], dtype=np.float64
        ).reshape(3)
        quaternion = np.asarray(
            current_observation["quaternion"], dtype=np.float64
        ).reshape(4)
        anchors = _move_tracked_strict_point_matrix(
            anchors_eef,
            expected_count=len(controlled),
            label="final EEF anchor bundle",
        )
        predicted = position[None, :] + (
            _quat_to_mat_xyzw(quaternion) @ anchors.T
        ).T
    except (KeyError, TypeError, ValueError) as exc:
        report["reason"] = f"final_fk_bundle_invalid: {exc}"
        return report
    if not np.all(np.isfinite(predicted)):
        report["reason"] = "final_fk_bundle_invalid"
        return report
    visible_anchor_limit = min(
        float(anchor_tolerance_m),
        max(0.012, float(pos_tol_m)),
    )

    measured: dict[str, np.ndarray] = {}
    invalid_names: list[str] = []
    for name in controlled + fixed:
        entry = entries.get(name)
        if entry is None:
            continue
        try:
            xyz = np.asarray(
                entry["xyz_in_robot_base_coord_m"], dtype=np.float64
            ).reshape(3)
        except (KeyError, TypeError, ValueError):
            invalid_names.append(name)
            continue
        if not np.all(np.isfinite(xyz)):
            invalid_names.append(name)
            continue
        measured[name] = xyz
    if invalid_names:
        report.update(
            {
                "reason": "tracked_point_invalid",
                "invalid_names": invalid_names,
            }
        )
        return report

    missing_fixed = [name for name in fixed if name not in measured]
    if missing_fixed:
        report.update(
            {
                "reason": str(live.get("reason") or "tracked_point_unavailable"),
                "missing_fixed_names": missing_fixed,
            }
        )
        return report
    missing_controlled = [name for name in controlled if name not in measured]
    measured_controlled = [name for name in controlled if name in measured]
    report["measured_controlled_names"] = measured_controlled
    depth_reobserved_names: list[str] = []
    preemptive_depth_reobservation: dict[str, Any] | None = None

    # When every feature identity is lost, depth near the FK prediction can
    # help the tracker choose where to search next, but it cannot establish
    # that the observed surface belongs to either original marker.  Keep the
    # synchronized depth measurement in the report as reacquisition evidence;
    # never promote it to terminal point truth.
    if (
        missing_controlled
        and len(missing_controlled) == len(controlled)
        and not measured_controlled
        and str(live.get("reason") or "") == "tracked_point_unavailable"
        and adapter is not None
    ):
        preemptive_depth_reobservation = (
            _move_tracked_eef_anchor_depth_reobservation(
                adapter,
                predicted,
                current_sequence=int(current_sequence),
            )
        )
        preemptive_depth_reobservation["applicable"] = True
        preemptive_depth_reobservation["trigger"] = (
            "complete_controlled_tracker_bundle_unavailable"
        )
        report["eef_anchor_depth_reobservation"] = (
            preemptive_depth_reobservation
        )

    if not missing_controlled:
        if not bool(live.get("ok")) and not depth_reobserved_names:
            report["reason"] = str(
                live.get("reason") or "tracked_point_unavailable"
            )
            return report
        tracked_points = np.vstack([measured[name] for name in controlled])
        tracked_anchor_errors = np.linalg.norm(tracked_points - predicted, axis=1)
        anchor_error_key = (
            "depth_reobserved_anchor_error_m"
            if depth_reobserved_names
            else "raw_tracker_anchor_error_m"
        )
        report[anchor_error_key] = {
            name: float(tracked_anchor_errors[index])
            for index, name in enumerate(controlled)
        }
        depth_reobservation = (
            preemptive_depth_reobservation
            if preemptive_depth_reobservation is not None
            else {
                "ok": False,
                "applicable": False,
                "reason": "tracker_bundle_within_strict_anchor_envelope",
                "prediction_values_published_as_observation": False,
            }
        )
        if (
            not depth_reobserved_names
            and tracked_anchor_errors.size
            and float(np.max(tracked_anchor_errors)) > visible_anchor_limit
        ):
            depth_reobservation = _move_tracked_eef_anchor_depth_reobservation(
                adapter,
                predicted,
                current_sequence=int(current_sequence),
            )
            depth_reobservation["applicable"] = True
        report["eef_anchor_depth_reobservation"] = depth_reobservation
        report["depth_reobserved_controlled_names"] = depth_reobserved_names
        mode = (
            "live_predefined_eef_fk_and_rgbd_bundle"
            if predefined_controlled
            else "all_controlled_points_measured_rgbd"
        )
        occlusion_reports: dict[str, Any] = {}
    else:
        unavailable = live.get("unavailable")
        unavailable_names = (
            {str(name) for name in unavailable}
            if isinstance(unavailable, Mapping)
            else set()
        )
        if (
            len(controlled) != 2
            or len(missing_controlled) != 1
            or len(measured_controlled) != 1
            or unavailable_names != set(missing_controlled)
            or str(live.get("reason") or "") != "tracked_point_unavailable"
        ):
            report.update(
                {
                    "reason": str(
                        live.get("reason") or "tracked_point_unavailable"
                    ),
                    "missing_controlled_names": missing_controlled,
                }
            )
            return report
        occlusion_reports = {
            name: _move_tracked_occluded_on_hand_evidence(live, name)
            for name in missing_controlled
        }
        if not all(item.get("ok") for item in occlusion_reports.values()):
            report.update(
                {
                    "reason": "on_hand_point_unavailable_without_occlusion_evidence",
                    "missing_controlled_names": missing_controlled,
                    "occlusion_evidence": occlusion_reports,
                }
            )
            return report
        mode = "partial_rgbd_with_occluded_on_hand_fk_completion"

    points = predicted.copy()
    for index, name in enumerate(controlled):
        if name in measured:
            points[index] = measured[name]
            entry = entries.get(name)
            if isinstance(entry, Mapping) and bool(
                entry.get("predefined_eef_point")
            ):
                report["point_sources"][name] = (
                    "submission_local_static_eef_geometry_and_current_"
                    "evaluator_proprio_fk"
                )
            else:
                report["point_sources"][name] = (
                    "current_evaluator_depth_linear_at_local_fk_projected_anchor"
                    if name in depth_reobserved_names
                    else "current_evaluator_rgbd_and_camera_relative_pose"
                )
        else:
            report["point_sources"][name] = (
                "frozen_rgbd_eef_anchor_and_current_evaluator_proprio_local_fk"
            )
    fixed_points = (
        np.vstack([measured[name] for name in fixed])
        if fixed
        else np.empty((0, 3), dtype=np.float64)
    )
    for name in fixed:
        report["point_sources"][name] = (
            "current_evaluator_rgbd_and_camera_relative_pose"
        )
    visible_indices = [controlled.index(name) for name in measured_controlled]
    visible_anchor_errors = np.linalg.norm(
        points[visible_indices] - predicted[visible_indices], axis=1
    )
    if (
        missing_controlled
        and (
            not visible_anchor_errors.size
            or float(np.max(visible_anchor_errors)) > visible_anchor_limit
        )
    ):
        report.update(
            {
                "reason": "visible_on_hand_point_not_rigid_with_selected_eef",
                "visible_anchor_tolerance_m": visible_anchor_limit,
                "visible_anchor_error_m": {
                    name: float(visible_anchor_errors[index])
                    for index, name in enumerate(measured_controlled)
                },
                "occlusion_evidence": occlusion_reports,
            }
        )
        return report

    report.update(
        {
            "ok": True,
            "reason": None,
            "mode": mode,
            "points": points,
            "fixed_points": fixed_points,
            "predicted_points": predicted,
            "fk_completed_controlled_names": missing_controlled,
            "occlusion_evidence": occlusion_reports,
            "visible_anchor_tolerance_m": visible_anchor_limit,
            "visible_anchor_error_m": {
                name: float(visible_anchor_errors[index])
                for index, name in enumerate(measured_controlled)
            },
        }
    )
    return report



def _move_tracked_release_motion_retention(
    manager,
    lease_state: dict[str, Any],
) -> dict[str, Any]:
    """Release a request-scoped tracker lease exactly once."""

    cached = lease_state.get("release")
    if isinstance(cached, dict):
        release = dict(cached)
    else:
        lease_id = str(lease_state.get("lease_id") or "").strip()
        if not lease_id:
            release = {
                "ok": True,
                "released": False,
                "lease_id": None,
                "reason": "no_motion_retention_lease",
            }
        else:
            end_retention = getattr(manager, "end_motion_retention", None)
            if not callable(end_retention):
                release = {
                    "ok": False,
                    "released": False,
                    "lease_id": lease_id,
                    "reason": "motion_retention_release_unavailable",
                }
            else:
                try:
                    raw_release = end_retention(lease_id)
                    if not isinstance(raw_release, dict):
                        raise TypeError("motion retention release must return an object")
                    release = dict(raw_release)
                except Exception as exc:  # diagnostics; the bounded lease also expires
                    release = {
                        "ok": False,
                        "released": False,
                        "lease_id": lease_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        lease_state["release"] = dict(release)

    # The process may still hold the pre-reload public wrapper function.  That
    # wrapper always calls this helper from ``finally``, so co-locating the
    # rigid-pair lease here makes cancellation/failure cleanup hot-load safe.
    pair_lease = lease_state.get("_rigid_pair_lease_state")
    if (
        isinstance(pair_lease, dict)
        and bool(pair_lease.get("active"))
        and not bool(pair_lease.get("preserve_after_success"))
    ):
        if manager is None:
            cleanup = {
                "ok": False,
                "active": None,
                "reason": "rigid_pair_manager_unavailable_during_cleanup",
            }
        else:
            cleanup = _move_tracked_deactivate_rigid_pair(
                manager,
                reason="move_tracked_point_request_did_not_complete_successfully",
                required=False,
            )
        pair_lease["active"] = False
        pair_lease["cleanup"] = _json_ready(cleanup)
    return dict(release)



def _move_tracked_preserve_motion_retention_for_successor(
    lease_state: dict[str, Any],
) -> dict[str, Any]:
    """Keep only tracker identity state across a bounded successful handoff."""

    cached = lease_state.get("release")
    if isinstance(cached, dict):
        return dict(cached)
    lease_id = str(lease_state.get("lease_id") or "").strip()
    lease = lease_state.get("lease")
    if not lease_id or not isinstance(lease, Mapping):
        return {
            "ok": False,
            "released": False,
            "lease_id": lease_id or None,
            "reason": "successor_motion_retention_lease_missing",
        }
    release = {
        "ok": True,
        "released": False,
        "lease_id": lease_id,
        "reason": "bounded_successor_identity_retention",
        "duration_s": float(lease.get("duration_s", 0.0)),
        "coordinates_published_while_unobserved": False,
        "expires_without_successor": True,
        "superseded_by_next_matching_request": True,
    }
    lease_state["release"] = dict(release)
    return release



def _move_tracked_reference_snapshot_check(
    manager,
    names: Sequence[str],
    reference_points: np.ndarray,
    *,
    episode_id: str,
    expected_sequence: int | None = None,
    tolerance_m: float = 0.020,
) -> dict[str, Any]:
    """Check frozen off-hand references without treating them as anchors."""

    names = [str(name) for name in names]
    if not names:
        return {"ok": True, "applicable": False, "max_drift_m": 0.0}
    live = _move_tracked_live_snapshot(manager, names, episode_id=episode_id)
    if not bool(live.get("ok")):
        return {
            "ok": False,
            "applicable": True,
            "reason": str(live.get("reason") or "off_hand_points_unavailable"),
            "error": str(live.get("error") or live.get("reason") or "off-hand observation unavailable"),
            "max_drift_m": float("inf"),
        }
    if expected_sequence is not None and int(live.get("observation_sequence", -1)) != int(expected_sequence):
        return {
            "ok": False,
            "applicable": True,
            "reason": "off_hand_observation_mismatch",
            "error": "off-hand points are not synchronized with proprioception",
            "observation_sequence": live.get("observation_sequence"),
            "max_drift_m": float("inf"),
        }
    try:
        measured = _move_tracked_strict_point_matrix(
            [live["entries"][name]["xyz_in_robot_base_coord_m"] for name in names],
            expected_count=len(names),
            label="off-hand reference point data",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "applicable": True,
            "reason": "off_hand_points_invalid",
            "error": str(exc),
            "max_drift_m": float("inf"),
        }
    report = _move_tracked_off_hand_reference_consensus(
        names,
        measured,
        reference_points,
        tolerance_m=tolerance_m,
        allow_visual_outlier_suppression=False,
    )
    report["observation_sequence"] = live.get("observation_sequence")
    return report



def _move_tracked_off_hand_reference_consensus(
    names: Sequence[str],
    measured_points: Any,
    reference_points: Any,
    *,
    tolerance_m: float,
    allow_visual_outlier_suppression: bool,
) -> dict[str, Any]:
    """Classify off-hand motion without accepting a visual association jump.

    Off-hand rows are an explicit policy assertion that their scene points do
    not move.  Their frozen coordinates define the immutable geometric goal;
    later RGB-D coordinates only verify that the scene references stayed put.
    During execution an occluded marker can nevertheless be associated with
    an unrelated surface.  Such a jump is suppressible only when a stable
    reference-point consensus remains and the measured bundle also violates
    its frozen rigid geometry.  A coherent rigid translation or rotation
    therefore still fails closed, as do single-reference requests.
    """

    ordered_names = [str(name) for name in names]
    if not ordered_names:
        return {
            "ok": True,
            "applicable": False,
            "all_points_verified": True,
            "tracking_outlier_suppressed": False,
            "constraint_reference_source": "not_applicable",
            "max_drift_m": 0.0,
        }
    measured = _move_tracked_strict_point_matrix(
        measured_points,
        expected_count=len(ordered_names),
        label="off-hand measured points",
    )
    reference = _move_tracked_strict_point_matrix(
        reference_points,
        expected_count=len(ordered_names),
        label="frozen off-hand reference points",
    )
    tolerance = max(1.0e-6, float(tolerance_m))
    if not math.isfinite(tolerance):
        raise ValueError("off-hand reference tolerance must be finite")

    errors = np.linalg.norm(measured - reference, axis=1)
    stable_mask = errors <= tolerance
    stable_names = [
        name for index, name in enumerate(ordered_names) if stable_mask[index]
    ]
    outlier_names = [
        name for index, name in enumerate(ordered_names) if not stable_mask[index]
    ]
    max_drift = float(np.max(errors)) if errors.size else 0.0

    pair_errors: dict[str, float] = {}
    for first in range(len(ordered_names)):
        for second in range(first + 1, len(ordered_names)):
            source_distance = float(
                np.linalg.norm(reference[second] - reference[first])
            )
            measured_distance = float(
                np.linalg.norm(measured[second] - measured[first])
            )
            pair_errors[
                f"{ordered_names[first]}::{ordered_names[second]}"
            ] = abs(measured_distance - source_distance)
    max_pair_error = max(pair_errors.values(), default=0.0)
    rigid_geometry_consistent = bool(max_pair_error <= tolerance)

    # Two references are the minimum needed to distinguish a discontinuous
    # association from coherent scene motion.  For larger bundles require a
    # strict majority of stable peers; the two-point preset necessarily uses
    # its one remaining point as the consensus witness.
    required_stable_count = (
        1 if len(ordered_names) == 2 else len(ordered_names) // 2 + 1
    )
    suppress_outliers = bool(
        allow_visual_outlier_suppression
        and len(ordered_names) >= 2
        and outlier_names
        and len(stable_names) >= required_stable_count
        and not rigid_geometry_consistent
    )
    all_verified = not outlier_names
    ok = bool(all_verified or suppress_outliers)
    reason = (
        None
        if all_verified
        else (
            "off_hand_visual_tracking_outlier_suppressed"
            if suppress_outliers
            else "off_hand_points_moved"
        )
    )
    return {
        "ok": ok,
        "applicable": True,
        "reason": reason,
        "error": (
            None
            if ok
            else (
                f"off-hand reference drift {max_drift:.6f}m exceeds "
                f"{tolerance:.6f}m without a suppressible visual-outlier consensus"
            )
        ),
        "tolerance_m": tolerance,
        "max_drift_m": max_drift,
        "point_drift_m": {
            name: float(errors[index])
            for index, name in enumerate(ordered_names)
        },
        "stable_names": stable_names,
        "outlier_names": outlier_names,
        "stable_count": int(len(stable_names)),
        "required_stable_count": int(required_stable_count),
        "all_points_verified": bool(all_verified),
        "tracking_outlier_suppressed": bool(suppress_outliers),
        "constraint_reference_source": (
            "frozen_off_hand_reference_due_visual_outlier"
            if suppress_outliers
            else "frozen_off_hand_reference_verified_by_live_rgbd"
        ),
        "rigid_geometry_consistent": rigid_geometry_consistent,
        "max_pair_distance_error_m": float(max_pair_error),
        "pair_distance_error_m": pair_errors,
        "coherent_rigid_reference_motion_accepted": False,
    }



def _move_tracked_activate_rigid_pair(
    manager,
    names: list[str],
    *,
    episode_id: str,
    allow_observation_failure: bool = False,
) -> dict[str, Any]:
    if len(names) < 2:
        return {
            "ok": True,
            "active": False,
            "reason": "single_point_request",
            "requested_names": list(names),
            "all_points_same_rigid_object_policy_assertion": False,
        }
    # A tracker that supports N-point fusion may provide an explicit endpoint.
    # Prefer it, because it can register one immutable bundle atomically.
    activate_all = getattr(manager, "activate_rigid_points", None)
    if callable(activate_all):
        report = activate_all(list(names), episode_id=episode_id)
        if not isinstance(report, dict) or not bool(report.get("ok")):
            reason = (
                report.get("reason")
                if isinstance(report, dict)
                else "invalid_rigid_points_fusion_result"
            )
            failed_report = {
                **(report if isinstance(report, dict) else {}),
                "ok": False,
                "reason": reason,
                "requested_names": list(names),
                "activated_names": list(names),
                "all_points_same_rigid_object_policy_assertion": True,
            }
            if allow_observation_failure:
                return failed_report
            raise ValueError(f"tracked rigid point bundle is inconsistent: {reason}")
        return {
            **report,
            "requested_names": list(names),
            "activated_names": list(names),
            "all_points_same_rigid_object_policy_assertion": True,
        }

    # The legacy endpoint is *strictly* a pair operation.  Do not activate the
    # first two names and then pretend that the remaining markers share the
    # same tracker state: that would make a four-point request silently use a
    # mixed representation.  For N > 2 the atomic snapshot and the local
    # pair/triangle/tetrahedron checks below are the authority until the
    # tracker grows an explicit N-point fusion API.
    if len(names) > 2:
        return {
            "ok": True,
            "active": False,
            "requested_names": list(names),
            "activated_names": [],
            "activation_scope": "n_point_local_geometry_only",
            "all_points_same_rigid_object_policy_assertion": True,
            "reason": "tracker_has_no_n_point_fusion; all requested points are checked atomically",
        }

    activate = getattr(manager, "activate_rigid_pair", None)
    if not callable(activate):
        raise RuntimeError(
            "official dynamic distance tracker does not expose registered "
            "rigid-pair or rigid-point-bundle fusion"
        )
    activation_names = list(names[:2])
    report = activate(activation_names, episode_id=episode_id)
    if not isinstance(report, dict) or not bool(report.get("ok")):
        reason = (
            report.get("reason")
            if isinstance(report, dict)
            else "invalid_rigid_pair_fusion_result"
        )
        failed_report = {
            **(report if isinstance(report, dict) else {}),
            "ok": False,
            "reason": reason,
            "requested_names": list(names),
            "activated_names": activation_names,
            "activation_scope": "two_point_rigid_pair",
            "all_points_same_rigid_object_policy_assertion": True,
        }
        if allow_observation_failure:
            return failed_report
        raise ValueError(f"tracked rigid pair is inconsistent: {reason}")
    return {
        **report,
        "requested_names": list(names),
        "activated_names": activation_names,
        "activation_scope": "two_point_rigid_pair",
        "all_points_same_rigid_object_policy_assertion": True,
    }



def _move_tracked_rigid_pair_activation_summary(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep activation evidence useful without copying large tracker payloads."""

    raw = report if isinstance(report, Mapping) else {}
    keys = (
        "ok",
        "reason",
        "names",
        "requested_names",
        "activated_names",
        "observation_sequence",
        "registration_observation_sequence",
        "reference_distance_m",
        "measured_distance_m",
        "fused_distance_m",
        "distance_error_m",
        "point_correction_m",
        "max_point_correction_m",
        "correction_limit_m",
        "fusion_applied",
        "tracker_estimator_ok",
        "direct_rgbd_projection_accepted_for_move_source",
        "move_source_max_point_correction_m",
        "fresh_observation_confirmation_required",
    )
    summary = {key: raw.get(key) for key in keys if key in raw}
    visual = raw.get("visual_tracker_pair")
    if isinstance(visual, Mapping):
        summary["visual_tracker_pair"] = {
            key: visual.get(key)
            for key in (
                "ok",
                "reason",
                "observation_sequence",
                "track_ids",
                "identity_margin_px",
                "pair_distance_error_m",
                "per_point",
            )
            if key in visual
        }
    return _json_ready(summary)



def _move_tracked_gate_rigid_pair_source_projection(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reject a large pair projection as an immediate manipulation source.

    ``project_registered_rigid_pair`` may legitimately return an estimator-level
    success after moving each current RGB-D point by as much as 35 mm.  That is
    useful for keeping a tracker alive, but freezing those corrected coordinates
    as EEF anchors creates a self-consistent plan for points that were never
    observed.  Preserve the estimator report, but require fresh joint-tracker
    confirmation whenever its pointwise correction exceeds 3 mm.
    """

    gated = dict(report) if isinstance(report, Mapping) else {}
    estimator_ok = bool(gated.get("ok"))
    gated["tracker_estimator_ok"] = estimator_ok
    gated["move_source_max_point_correction_m"] = float(
        MOVE_TRACKED_POINT_RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M
    )
    if not estimator_ok:
        gated["direct_rgbd_projection_accepted_for_move_source"] = False
        return gated

    raw_correction = gated.get("max_point_correction_m")
    if raw_correction is None:
        # Compatibility for test-local/custom tracker providers that publish
        # direct EEF-rigid coordinates rather than a registration projection.
        gated["direct_rgbd_projection_accepted_for_move_source"] = True
        return gated
    try:
        correction = float(raw_correction)
    except (TypeError, ValueError):
        correction = float("inf")
    accepted = bool(
        math.isfinite(correction)
        and correction
        <= MOVE_TRACKED_POINT_RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M + 1e-12
    )
    gated["direct_rgbd_projection_accepted_for_move_source"] = accepted
    if accepted:
        return gated

    gated.update(
        {
            "ok": False,
            "reason": "rigid_pair_projection_requires_fresh_confirmation",
            "fresh_observation_confirmation_required": True,
        }
    )
    return gated



def _move_tracked_registered_points_snapshot(
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
) -> dict[str, Any]:
    """Read the policy-owned frozen RGB-D registration for a named pair."""

    requested = [str(name) for name in names]
    snapshot_fn = getattr(manager, "registered_points_snapshot", None)
    if not callable(snapshot_fn):
        return {
            "ok": True,
            "available": False,
            "reason": "tracker_provider_has_no_registration_snapshot",
            "requested_names": requested,
        }
    snapshot = snapshot_fn(requested, episode_id=str(episode_id))
    if not isinstance(snapshot, Mapping):
        raise ValueError("registered point snapshot must be an object")
    if not bool(snapshot.get("ok")):
        raise ValueError(
            "registered point snapshot is unavailable: "
            f"{snapshot.get('reason') or 'unknown'}"
        )
    entries = snapshot.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("registered point snapshot has no entries")
    points = _move_tracked_strict_point_matrix(
        [
            entries[name]["xyz_in_robot_base_coord_m"]
            for name in requested
        ],
        expected_count=len(requested),
        label="registered tracked points",
    )
    session_id = str(snapshot.get("session_id") or "").strip()
    image_id = str(snapshot.get("image_id") or "").strip()
    snapshot_episode = str(snapshot.get("episode_id") or "").strip()
    if not session_id or not image_id:
        raise ValueError("registered point snapshot has no capture binding")
    if snapshot_episode != str(episode_id):
        raise ValueError("registered point snapshot episode changed")
    try:
        registration_sequence = int(
            snapshot["registration_observation_sequence"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "registered point snapshot has no observation sequence"
        ) from exc
    return {
        "ok": True,
        "available": True,
        "reason": None,
        "requested_names": requested,
        "points_robot_base_m": points,
        "session_id": session_id,
        "image_id": image_id,
        "episode_id": snapshot_episode,
        "registration_observation_sequence": registration_sequence,
        "measurement_source": str(
            snapshot.get("measurement_source")
            or "policy_owned_frozen_head_rgbd_registration"
        ),
        "predictions_published": False,
    }



def _move_tracked_reconcile_registered_on_hand_source(
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
    current_capture_state: Mapping[str, Any],
    live_points_robot_base_m: Any,
) -> dict[str, Any]:
    """Recover a bounded N-point tracker outlier from frozen EEF anchors.

    The returned model points are planning inputs, never observations.  A
    strict majority of current RGB-D markers must independently agree with the
    registration-time anchors transformed by current evaluator proprioception
    and submission-local FK.  This keeps the normal live-RGB-D path unchanged,
    rejects an incoherent bundle, and prevents one feature association from
    deforming the rigid object used by endpoint/overlap candidate search.
    """

    requested = [str(name) for name in names]
    live = _move_tracked_strict_point_matrix(
        live_points_robot_base_m,
        expected_count=len(requested),
        label="live on-hand tracked points",
    )
    if len(requested) < 3:
        return {
            "ok": True,
            "applicable": False,
            "applied": False,
            "reason": "registration_eef_source_recovery_requires_three_points",
            "names": requested,
            "source_points_robot_base_m": live.astype(float).tolist(),
            "predictions_published_as_observation": False,
        }

    registration = _move_tracked_registered_points_snapshot(
        manager,
        requested,
        episode_id=episode_id,
    )
    if not bool(registration.get("available")):
        return {
            "ok": True,
            "applicable": False,
            "applied": False,
            "reason": registration.get("reason"),
            "names": requested,
            "source_points_robot_base_m": live.astype(float).tolist(),
            "registration_snapshot": _json_ready(registration),
            "predictions_published_as_observation": False,
        }

    capture = _load_frozen_capture(
        registration["session_id"],
        registration["image_id"],
    )
    registration_state = _move_point_capture_kinematics(capture)
    if str(registration_state["episode_id"]) != str(episode_id):
        raise ValueError("registration capture episode changed")
    if str(current_capture_state.get("episode_id") or "") != str(episode_id):
        raise ValueError("current proprioception episode changed")
    registration_sequence = int(
        registration["registration_observation_sequence"]
    )
    if (
        capture.evaluator_sequence is None
        or int(capture.evaluator_sequence) != registration_sequence
    ):
        raise ValueError(
            "registration RGB-D and capture-time proprioception are not synchronized"
        )

    registered = _move_tracked_strict_point_matrix(
        registration["points_robot_base_m"],
        expected_count=len(requested),
        label="registered on-hand tracked points",
    )
    registration_distances = {
        side: float(
            np.mean(
                np.linalg.norm(
                    registered
                    - np.asarray(
                        registration_state["eef"][side]["position"],
                        dtype=np.float64,
                    ).reshape(1, 3),
                    axis=1,
                )
            )
        )
        for side in ("left", "right")
    }
    arm = min(("left", "right"), key=lambda side: registration_distances[side])
    registration_eef_position = np.asarray(
        registration_state["eef"][arm]["position"], dtype=np.float64
    ).reshape(3)
    registration_eef_rotation = _quat_to_mat_xyzw(
        registration_state["eef"][arm]["quaternion"]
    )
    anchors = (
        registration_eef_rotation.T
        @ (registered - registration_eef_position[None, :]).T
    ).T
    current_eef_position = np.asarray(
        current_capture_state["eef"][arm]["position"], dtype=np.float64
    ).reshape(3)
    current_eef_rotation = _quat_to_mat_xyzw(
        current_capture_state["eef"][arm]["quaternion"]
    )
    expected = current_eef_position[None, :] + (
        current_eef_rotation @ anchors.T
    ).T
    errors = np.linalg.norm(live - expected, axis=1)
    inlier_mask = errors <= (
        MOVE_TRACKED_POINT_REGISTRATION_EEF_INLIER_M + 1e-12
    )
    inlier_count = int(np.count_nonzero(inlier_mask))
    required_inliers = max(2, len(requested) // 2 + 1)
    maximum_error = float(np.max(errors)) if errors.size else 0.0
    outlier_names = [
        requested[index]
        for index in range(len(requested))
        if not bool(inlier_mask[index])
    ]
    inlier_names = [
        requested[index]
        for index in range(len(requested))
        if bool(inlier_mask[index])
    ]

    current_gripper = np.asarray(
        current_capture_state.get("gripper_q", {}).get(arm),
        dtype=np.float64,
    ).reshape(-1)
    registration_gripper = np.asarray(
        registration_state["gripper_q"][arm], dtype=np.float64
    ).reshape(-1)
    gripper_delta = (
        float(np.max(np.abs(current_gripper - registration_gripper)))
        if current_gripper.shape == registration_gripper.shape
        and current_gripper.size
        and np.all(np.isfinite(current_gripper))
        else None
    )
    common = {
        "applicable": True,
        "names": requested,
        "selected_arm": arm,
        "registration_eef_mean_distance_m": registration_distances,
        "registration_observation_sequence": registration_sequence,
        "registration_measurement_source": registration["measurement_source"],
        "anchors_eef_m": anchors.astype(float).tolist(),
        "expected_points_robot_base_m": expected.astype(float).tolist(),
        "live_points_robot_base_m": live.astype(float).tolist(),
        "per_point_live_to_registered_eef_error_m": {
            requested[index]: float(errors[index])
            for index in range(len(requested))
        },
        "max_live_to_registered_eef_error_m": maximum_error,
        "inlier_tolerance_m": float(
            MOVE_TRACKED_POINT_REGISTRATION_EEF_INLIER_M
        ),
        "max_recovery_error_m": float(
            MOVE_TRACKED_POINT_REGISTRATION_EEF_MAX_RECOVERY_M
        ),
        "inlier_names": inlier_names,
        "outlier_names": outlier_names,
        "inlier_count": inlier_count,
        "required_inlier_count": required_inliers,
        "registration_to_current_gripper_delta_inf_m": gripper_delta,
        "planning_source_is_model_derived": bool(outlier_names),
        "predictions_published_as_observation": False,
    }
    if not outlier_names:
        return {
            **common,
            "ok": True,
            "applied": False,
            "reason": "all_live_points_agree_with_registered_eef_anchors",
            "source_points_robot_base_m": live.astype(float).tolist(),
            "planning_source_is_model_derived": False,
        }
    if inlier_count < required_inliers:
        return {
            **common,
            "ok": False,
            "applied": False,
            "reason": "registered_eef_anchor_consensus_insufficient",
            "source_points_robot_base_m": live.astype(float).tolist(),
        }
    if maximum_error > (
        MOVE_TRACKED_POINT_REGISTRATION_EEF_MAX_RECOVERY_M + 1e-12
    ):
        return {
            **common,
            "ok": False,
            "applied": False,
            "reason": "registered_eef_anchor_recovery_exceeded",
            "source_points_robot_base_m": live.astype(float).tolist(),
        }
    return {
        **common,
        "ok": True,
        "applied": True,
        "reason": "bounded_tracker_outlier_recovered_from_registered_eef_anchors",
        "source_points_robot_base_m": expected.astype(float).tolist(),
        "recovered_point_names": outlier_names,
        "planning_source_is_model_derived": True,
    }



def _move_tracked_bootstrap_on_hand_eef_prior(
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
    current_capture_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an EEF candidate gate from registration RGB-D and local FK.

    The predicted points are supplied only to the dynamic tracker's candidate
    gate.  They never replace a current evaluator RGB-D measurement.
    """

    requested = [str(name) for name in names]
    if (
        not 1 <= len(requested) <= MOVE_TRACKED_POINT_MAX_POINTS
        or any(not name for name in requested)
        or len(set(requested)) != len(requested)
    ):
        return {
            "ok": False,
            "active": False,
            "reason": "registration_eef_bootstrap_point_names_invalid",
            "names": requested,
        }
    registration = _move_tracked_registered_points_snapshot(
        manager,
        requested,
        episode_id=episode_id,
    )
    if not bool(registration.get("available")):
        return {
            "ok": True,
            "active": False,
            "reason": registration.get("reason"),
            "names": requested,
            "registration_snapshot": _json_ready(registration),
        }

    capture = _load_frozen_capture(
        registration["session_id"],
        registration["image_id"],
    )
    capture_state = _move_point_capture_kinematics(capture)
    if str(capture_state["episode_id"]) != str(episode_id):
        raise ValueError("registration capture episode changed")
    capture_sequence = capture.evaluator_sequence
    registration_sequence = int(
        registration["registration_observation_sequence"]
    )
    if capture_sequence is None or int(capture_sequence) != registration_sequence:
        raise ValueError(
            "registration RGB-D and capture-time proprioception are not synchronized"
        )

    points = np.asarray(
        registration["points_robot_base_m"], dtype=np.float64
    ).reshape(len(requested), 3)
    distances = {
        side: float(
            np.mean(
                np.linalg.norm(
                    points
                    - np.asarray(
                        capture_state["eef"][side]["position"],
                        dtype=np.float64,
                    ).reshape(1, 3),
                    axis=1,
                )
            )
        )
        for side in ("left", "right")
    }
    arm = min(("left", "right"), key=lambda side: distances[side])
    eef_position = np.asarray(
        capture_state["eef"][arm]["position"], dtype=np.float64
    ).reshape(3)
    eef_rotation = _quat_to_mat_xyzw(
        capture_state["eef"][arm]["quaternion"]
    )
    anchors = (eef_rotation.T @ (points - eef_position[None, :]).T).T
    current_gripper = np.asarray(
        current_capture_state.get("gripper_q", {}).get(arm),
        dtype=np.float64,
    ).reshape(-1)
    registration_gripper = np.asarray(
        capture_state["gripper_q"][arm], dtype=np.float64
    ).reshape(-1)
    gripper_delta = (
        float(np.max(np.abs(current_gripper - registration_gripper)))
        if current_gripper.shape == registration_gripper.shape
        and current_gripper.size
        and np.all(np.isfinite(current_gripper))
        else None
    )
    # Activate only after every capture-derived input has been validated.  A
    # malformed diagnostic field must not leave an EEF-prior lease behind.
    prior = _move_tracked_activate_on_hand_eef_prior(
        manager,
        requested,
        episode_id=episode_id,
        session_id=registration["session_id"],
        image_id=registration["image_id"],
        arm=arm,
        anchors_eef_m=anchors,
    )
    return {
        **prior,
        "bootstrap": "registration_rgbd_capture_proprio_local_fk",
        "selected_arm": arm,
        "eef_mean_distance_m": distances,
        "anchors_eef_m": anchors.astype(float).tolist(),
        "registration_observation_sequence": registration_sequence,
        "registration_measurement_source": registration["measurement_source"],
        "registration_to_current_gripper_delta_inf_m": gripper_delta,
        "prediction_values_published_as_observation": False,
    }



def _move_tracked_reacquire_registered_bundle(
    ctx,
    manager,
    *,
    names: Sequence[str],
    controlled_names: Sequence[str],
    episode_id: str,
    session_id: str,
    image_id: str,
    observation_sequence: int,
    hold_state: Mapping[str, Any],
    initial_report: Mapping[str, Any],
    lease_state: dict[str, Any],
    deadline_monotonic: float,
):
    """Recover a failed plan source using two fresh, independently measured frames.

    Registration/local FK supplies a candidate search gate, not replacement
    coordinates. All on-hand and reference points are re-frozen together only
    after every on-hand RGB-D measurement agrees with that gate.
    """
    report = {
        "ok": False,
        "recovered": False,
        "initial_failure": _json_ready(initial_report),
        "history": [],
        "hold_steps": 0,
        "predictions_published_as_observation": False,
    }
    _check_cancelled(ctx, "move_tracked_point registered bundle reacquisition")
    deadline = min(
        float(deadline_monotonic),
        time.monotonic() + MOVE_TRACKED_POINT_REGISTRATION_REACQUIRE_MAX_S,
    )
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        return {**report, "reason": "registered_bundle_reacquisition_timeout"}
    prior = _move_tracked_bootstrap_on_hand_eef_prior(
        manager, controlled_names, episode_id=episode_id,
        current_capture_state=hold_state,
    )
    report["observation_prior"] = _json_ready(prior)
    if not bool(prior.get("ok")) or not bool(prior.get("active")):
        return {**report, "reason": "registered_bundle_reacquisition_prior_unavailable"}
    # The public generator owns this lease even if cancelled at the next yield.
    lease_state.update(prior)
    previous_sequence = max(
        int(observation_sequence), int(prior["activation_observation_sequence"])
    )
    stable_frames = 0
    for step in range(MOVE_TRACKED_POINT_REGISTRATION_REACQUIRE_MAX_STEPS):
        _check_cancelled(ctx, "move_tracked_point registered bundle reacquisition")
        if time.monotonic() >= deadline:
            break
        yield _move_tracked_planning_hold_action(ctx, dict(hold_state))
        report["hold_steps"] = step + 1
        _check_cancelled(ctx, "move_tracked_point registered bundle reacquisition")
        if time.monotonic() >= deadline:
            break
        if str(ctx.world.episode_id()) != str(episode_id):
            return {**report, "reason": "episode_changed"}
        adapter, sequence = _adjust_adapter(ctx)
        if int(sequence) <= previous_sequence:
            return {**report, "reason": "registered_bundle_observation_not_advanced"}
        previous_sequence = int(sequence)
        live = _move_tracked_live_snapshot(manager, list(names), episode_id=episode_id)
        if not bool(live.get("ok")):
            report["history"].append({
                "sequence": sequence, "reason": live.get("reason"),
                "unavailable": _json_ready(live.get("unavailable")),
            })
            stable_frames = 0
            continue
        if (
            str(live.get("episode_id") or "") != str(episode_id)
            or str(live.get("session_id") or "") != session_id
            or str(live.get("image_id") or "") != image_id
        ):
            return {**report, "reason": "registered_bundle_binding_changed"}
        if int(live.get("observation_sequence", -1)) != int(sequence):
            return {**report, "reason": "registered_bundle_observation_mismatch"}
        observations = {
            side: _adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        capture_state = _move_tracked_capture_state(ctx, observations)
        sequence_after, _received_ts = adapter.observation_metadata()
        if int(sequence_after) != int(sequence):
            return {**report, "reason": "registered_bundle_observation_mismatch"}
        matrix = _move_tracked_strict_point_matrix(
            [live["entries"][name]["xyz_in_robot_base_coord_m"] for name in names],
            expected_count=len(names), label="reacquired tracked source points",
        )
        source_by_name = {name: matrix[index].copy() for index, name in enumerate(names)}
        source_points = np.asarray([source_by_name[name] for name in controlled_names])
        reconciliation = _move_tracked_reconcile_registered_on_hand_source(
            manager, controlled_names, episode_id=episode_id,
            current_capture_state=capture_state, live_points_robot_base_m=source_points,
        )
        # A bounded model-derived repair is not fresh visual confirmation.
        visually_confirmed = bool(
            reconciliation.get("ok") and reconciliation.get("applicable")
            and not reconciliation.get("applied")
            and not reconciliation.get("outlier_names")
        )
        stable_frames = stable_frames + 1 if visually_confirmed else 0
        report["history"].append({
            "sequence": sequence, "reason": reconciliation.get("reason"),
            "max_error_m": reconciliation.get("max_live_to_registered_eef_error_m"),
            "stable_frames": stable_frames,
        })
        report["final_reconciliation"] = _json_ready(reconciliation)
        if stable_frames >= 2:
            report.update(ok=True, recovered=True, reason=None)
            reconciliation["fresh_rgbd_reacquisition"] = report
            return {
                "ok": True, "report": report, "adapter": adapter,
                "observation_sequence": int(sequence), "live": live,
                "observations": observations, "capture_state": capture_state,
                "source_by_name": source_by_name, "source_points": source_points,
                "reconciliation": reconciliation,
            }
    return {**report, "reason": "registered_bundle_reacquisition_timeout"}



def _move_tracked_rigid_pair_activation_error(
    report: Mapping[str, Any],
) -> str:
    reason = str(report.get("reason") or "rigid_pair_activation_recovery_failed")
    last = report.get("last_pair_report")
    first = report.get("first_failure")
    diagnostic_sources = [
        source for source in (last, first) if isinstance(source, Mapping)
    ]
    if not diagnostic_sources:
        return f"tracked rigid pair activation failed: {reason}"

    diagnostics: list[str] = []
    for key, label in (
        ("reference_distance_m", "reference"),
        ("measured_distance_m", "measured"),
        ("distance_error_m", "distance_error"),
        ("max_point_correction_m", "required_point_correction"),
        ("correction_limit_m", "correction_limit"),
    ):
        for source in diagnostic_sources:
            try:
                number = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                diagnostics.append(f"{label}={number:.6f}m")
                break
    suffix = f" ({', '.join(diagnostics)})" if diagnostics else ""
    return f"tracked rigid pair activation failed: {reason}{suffix}"



def _move_tracked_deactivate_rigid_pair(
    manager,
    *,
    reason: str,
    required: bool,
) -> dict[str, Any]:
    """End request-scoped pair fusion while preserving named registrations."""

    deactivate = getattr(manager, "deactivate_rigid_pair", None)
    if not callable(deactivate):
        # A running interface may own an instance created before the public
        # deactivation method was hot-loaded.  Its existing request-local
        # retirement primitive is still safe to invoke under the manager lock;
        # this compatibility path can be removed after a full process restart.
        retire_locked = getattr(manager, "_retire_active_rigid_pair_locked", None)
        manager_lock = getattr(manager, "_lock", None)
        if callable(retire_locked) and manager_lock is not None:
            try:
                with manager_lock:
                    active_pair = getattr(manager, "_active_rigid_pair", None)
                    active_names = (
                        [str(name) for name in active_pair.get("names", ())]
                        if isinstance(active_pair, Mapping)
                        else []
                    )
                    retired = retire_locked(reason=str(reason))
                    # Pre-v6 manager instances leave pair-projected XYZ in the
                    # public rows after retirement.  Reverse that projection
                    # from the same observation's saved raw RGB-D coordinates;
                    # if they are unavailable, fail closed instead of leaking a
                    # stale fused coordinate into the next request.
                    restored_names: list[str] = []
                    unavailable_names: list[str] = []
                    unchanged_names: list[str] = []
                    entries = getattr(manager, "_entries", None)
                    temporarily_unobserved = getattr(
                        manager, "_temporarily_unobserved", None
                    )
                    if isinstance(entries, dict):
                        for name in active_names:
                            entry = entries.get(name)
                            if not isinstance(entry, dict):
                                continue
                            raw_present = "raw_xyz_in_robot_base_coord_m" in entry
                            raw_xyz = entry.pop(
                                "raw_xyz_in_robot_base_coord_m", None
                            )
                            raw_source = entry.pop("raw_xyz_source", None)
                            xyz_source = str(entry.get("xyz_source") or "")
                            fused_source = "registration_rigid_pair" in xyz_source
                            entry.pop("rigid_pair_fusion", None)
                            raw_array = np.empty(0, dtype=np.float64)
                            if raw_present:
                                try:
                                    raw_array = np.asarray(
                                        raw_xyz, dtype=np.float64
                                    ).reshape(3)
                                except (TypeError, ValueError):
                                    raw_array = np.empty(0, dtype=np.float64)
                            valid_raw = bool(
                                raw_array.shape == (3,)
                                and np.all(np.isfinite(raw_array))
                                and "registration_rigid_pair"
                                not in str(raw_source or "")
                            )
                            observed = str(entry.get("status") or "") == "observed"
                            if observed and valid_raw:
                                entry["xyz_in_robot_base_coord_m"] = (
                                    raw_array.astype(float).tolist()
                                )
                                entry["xyz_source"] = str(
                                    raw_source
                                    or "current_evaluator_depth_linear_and_cam_rel_poses"
                                )
                                if isinstance(temporarily_unobserved, dict):
                                    temporarily_unobserved.pop(name, None)
                                restored_names.append(name)
                                continue
                            if observed and not raw_present and not fused_source:
                                unchanged_names.append(name)
                                continue
                            entry.update(
                                {
                                    "depth_m": None,
                                    "u": None,
                                    "v": None,
                                    "xyz_in_robot_base_coord_m": None,
                                    "status": "temporarily_unobserved",
                                    "depth_source": None,
                                    "xyz_source": None,
                                    "tracker_status": (
                                        "rigid_pair_raw_rgbd_unavailable"
                                    ),
                                }
                            )
                            if isinstance(temporarily_unobserved, dict):
                                temporary = dict(
                                    temporarily_unobserved.get(name) or {}
                                )
                                temporary.update(
                                    {
                                        "name": name,
                                        "status": (
                                            "rigid_pair_raw_rgbd_unavailable"
                                        ),
                                        "missed_steps": max(
                                            1,
                                            int(temporary.get("missed_steps", 0)),
                                        ),
                                        "observation_sequence": int(
                                            entry.get("observation_sequence", -1)
                                        ),
                                    }
                                )
                                temporarily_unobserved[name] = temporary
                            unavailable_names.append(name)
                    entry_retirement = {
                        "names": active_names,
                        "restored_raw_names": restored_names,
                        "temporarily_unobserved_names": unavailable_names,
                        "unchanged_raw_names": unchanged_names,
                    }
                return {
                    "ok": True,
                    "active": False,
                    "reason": str(reason),
                    "hot_reload_compatibility_path": True,
                    "retired_report": _json_ready(retired),
                    "entry_retirement": _json_ready(entry_retirement),
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "active": None,
                    "reason": "rigid_pair_deactivation_failed",
                    "requested_reason": str(reason),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {
            "ok": not required,
            "active": None,
            "reason": "tracker_provider_has_no_rigid_pair_deactivation",
            "requested_reason": str(reason),
        }
    try:
        try:
            report = deactivate(reason=str(reason))
        except TypeError:
            report = deactivate()
    except Exception as exc:
        return {
            "ok": False,
            "active": None,
            "reason": "rigid_pair_deactivation_failed",
            "requested_reason": str(reason),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(report, Mapping):
        return {
            "ok": False,
            "active": None,
            "reason": "invalid_rigid_pair_deactivation_result",
            "requested_reason": str(reason),
        }
    return _json_ready(dict(report))



def _move_tracked_clear_incompatible_rigid_pair(
    manager,
    names: Sequence[str],
) -> dict[str, Any]:
    """Prevent an earlier pair request from governing a different request."""

    requested = [str(name) for name in names]
    status = getattr(manager, "status", None)
    if not callable(status):
        return {
            "ok": True,
            "active": None,
            "reason": "tracker_provider_has_no_status",
        }
    try:
        snapshot = status()
    except Exception as exc:
        raise RuntimeError(
            f"could not inspect prior rigid-pair state: {type(exc).__name__}: {exc}"
        ) from exc
    active = (
        snapshot.get("active_rigid_pair")
        if isinstance(snapshot, Mapping)
        else None
    )
    if not isinstance(active, Mapping):
        return {
            "ok": True,
            "active": False,
            "reason": "no_prior_rigid_pair",
        }
    active_names = [str(name) for name in active.get("names", ())]
    if len(requested) == 2 and active_names == requested:
        return {
            "ok": True,
            "active": True,
            "reason": "current_request_will_reactivate_exact_pair",
            "names": active_names,
        }
    report = _move_tracked_deactivate_rigid_pair(
        manager,
        reason="rigid_pair_not_part_of_current_controlled_point_set",
        required=True,
    )
    if not bool(report.get("ok")):
        raise RuntimeError(
            "could not retire a rigid pair left by a previous move request: "
            f"{report.get('reason') or 'unknown'}"
        )
    return report



def _move_tracked_confirm_fresh_rigid_pair_activation(
    ctx,
    manager,
    names: list[str],
    *,
    episode_id: str,
    adapter,
    observation_sequence: int,
    initial_report: Mapping[str, Any],
    hold_state: Mapping[str, Any],
    deadline_monotonic: float | None = None,
):
    """Confirm a provisional pair activation on strictly newer RGB-D frames.

    ``activate_rigid_pair`` switches the tracker into joint mode, but its return
    value is computed from coordinates published by the preceding independent
    trackers.  A mismatch in that one old frame is therefore not motion
    evidence.  Hold the complete robot action while the already-active joint
    tracker consumes fresh evaluator observations; never retry against the same
    sequence and never relax the geometric correction limit.
    """

    initial = dict(initial_report) if isinstance(initial_report, Mapping) else {}
    initial_reason = str(initial.get("reason") or "")
    base = {
        "required": not bool(initial.get("ok")),
        "recovered": False,
        "attempt_count": 1,
        "recovery_hold_steps": 0,
        "required_stable_steps": int(
            MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_STABLE_STEPS
        ),
        "max_wait_steps": int(
            MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS
        ),
        "observation_sequence_start": int(observation_sequence),
        "observation_sequence_end": int(observation_sequence),
        "first_failure": (
            None
            if bool(initial.get("ok"))
            else _move_tracked_rigid_pair_activation_summary(initial)
        ),
        "last_pair_report": _move_tracked_rigid_pair_activation_summary(initial),
        "history": [],
        "correction_limit_unchanged": True,
    }
    if bool(initial.get("ok")):
        return {
            **base,
            "ok": True,
            "reason": None,
            "confirmation_source": "initial_activation_fast_path",
        }
    if initial_reason not in (
        MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS
    ):
        return {
            **base,
            "ok": False,
            "reason": initial_reason or "invalid_rigid_pair_fusion_result",
            "error": "tracked rigid pair activation is not recoverable from a new observation",
        }

    stable_steps = 0
    previous_sequence = int(observation_sequence)
    started = time.monotonic()
    local_deadline = started + MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_S
    if deadline_monotonic is not None:
        requested_deadline = float(deadline_monotonic)
        if not math.isfinite(requested_deadline):
            raise ValueError("rigid-pair activation deadline must be finite")
        local_deadline = min(local_deadline, requested_deadline)
    last_pair_report: dict[str, Any] = dict(initial)
    history: list[dict[str, Any]] = []
    for wait_step in range(
        1, MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS + 1
    ):
        _check_cancelled(ctx, "move_tracked_point rigid-pair activation recovery")
        if time.monotonic() >= local_deadline:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_recovery_timeout",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step - 1),
                "observation_sequence_end": int(previous_sequence),
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    last_pair_report
                ),
                "history": history,
            }

        # The yield is the synchronization boundary: the policy emits exactly
        # one official full hold action and resumes only on the evaluator's next
        # action request.  No simulator state is queried or mutated here.
        yield _move_tracked_planning_hold_action(ctx, dict(hold_state))
        _check_cancelled(ctx, "move_tracked_point rigid-pair activation recovery")
        if time.monotonic() >= local_deadline:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_recovery_timeout",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(previous_sequence),
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    last_pair_report
                ),
                "history": history,
            }
        current_adapter, current_sequence = _adjust_adapter(ctx)
        advanced, observed_sequence = _adjust_observation_advanced(
            current_adapter,
            previous_sequence,
        )
        if not advanced:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_observation_not_advanced",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(observed_sequence),
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    last_pair_report
                ),
                "history": history,
            }
        adapter = current_adapter
        current_sequence = int(current_sequence)
        if str(ctx.world.episode_id()) != str(episode_id):
            return {
                **base,
                "ok": False,
                "reason": "episode_changed",
                "attempt_count": int(2 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": current_sequence,
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    last_pair_report
                ),
                "history": history,
            }

        live = _move_tracked_live_snapshot(
            manager,
            names,
            episode_id=episode_id,
        )
        pair_report = live.get("rigid_pair_fusion")
        if not isinstance(pair_report, Mapping):
            pair_report = {
                "ok": False,
                "reason": str(
                    live.get("reason") or "rigid_pair_measurement_unavailable"
                ),
                "observation_sequence": live.get("observation_sequence"),
            }
        pair_report = _move_tracked_gate_rigid_pair_source_projection(pair_report)
        last_pair_report = dict(pair_report)
        live_sequence = live.get("observation_sequence")
        synchronized = (
            live_sequence is not None
            and int(live_sequence) == current_sequence
        )
        frame_ok = bool(
            live.get("ok")
            and pair_report.get("ok")
            and synchronized
            and str(live.get("episode_id") or "") == str(episode_id)
        )
        frame_reason = (
            None
            if frame_ok
            else str(
                pair_report.get("reason")
                or live.get("reason")
                or (
                    "rigid_pair_activation_observation_not_synchronized"
                    if not synchronized
                    else "rigid_pair_measurement_invalid"
                )
            )
        )
        stable_steps = stable_steps + 1 if frame_ok else 0
        history.append(
            {
                "wait_step": int(wait_step),
                "observation_sequence": current_sequence,
                "live_observation_sequence": live_sequence,
                "synchronized": bool(synchronized),
                "frame_ok": bool(frame_ok),
                "reason": frame_reason,
                "stable_steps": int(stable_steps),
                "pair_report": _move_tracked_rigid_pair_activation_summary(
                    pair_report
                ),
            }
        )
        previous_sequence = current_sequence
        if frame_ok and stable_steps >= (
            MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_STABLE_STEPS
        ):
            return {
                **base,
                "ok": True,
                "reason": None,
                "recovered": True,
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": current_sequence,
                "stable_steps": int(stable_steps),
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    pair_report
                ),
                "history": history,
                "confirmation_source": (
                    "strictly_new_joint_rgbd_rigid_pair_observations"
                ),
            }
        if not frame_ok and frame_reason not in (
            MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_FRESH_RETRY_REASONS
        ):
            return {
                **base,
                "ok": False,
                "reason": frame_reason,
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": current_sequence,
                "last_pair_report": _move_tracked_rigid_pair_activation_summary(
                    pair_report
                ),
                "history": history,
            }

    return {
        **base,
        "ok": False,
        "reason": "rigid_pair_activation_recovery_exhausted",
        "attempt_count": int(1 + len(history)),
        "recovery_hold_steps": int(len(history)),
        "observation_sequence_end": int(previous_sequence),
        "last_failure_reason": str(
            last_pair_report.get("reason") or "rigid_pair_measurement_invalid"
        ),
        "last_pair_report": _move_tracked_rigid_pair_activation_summary(
            last_pair_report
        ),
        "history": history,
    }



def _move_tracked_activate_on_hand_eef_prior(
    manager,
    names: list[str],
    *,
    episode_id: str,
    session_id: str,
    image_id: str,
    arm: str,
    anchors_eef_m: Any,
) -> dict[str, Any]:
    requested = [str(name or "").strip() for name in names]
    if (
        not 1 <= len(requested) <= MOVE_TRACKED_POINT_MAX_POINTS
        or any(not name for name in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError(
            "on-hand EEF observation prior requires one to six unique names"
        )
    anchors = _move_tracked_strict_point_matrix(
        anchors_eef_m,
        expected_count=len(requested),
        label="on-hand EEF observation prior anchors",
    )
    activate = getattr(
        manager,
        "activate_on_hand_eef_observation_prior",
        None,
    )
    if not callable(activate):
        return {
            "ok": True,
            "active": False,
            "reason": "tracker_provider_has_no_eef_observation_prior",
            "names": requested,
        }
    report = activate(
        requested,
        episode_id=episode_id,
        session_id=session_id,
        image_id=image_id,
        arm=arm,
        anchors_eef_m=anchors,
    )
    if not isinstance(report, dict) or not bool(report.get("ok")):
        reason = (
            report.get("reason")
            if isinstance(report, dict)
            else "invalid_on_hand_eef_observation_prior_result"
        )
        raise ValueError(f"could not activate on-hand RGB-D gate: {reason}")
    return {**report, "active": bool(report.get("active", True))}



def _move_tracked_capture_state(
    ctx,
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trunk = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(4)
    arm_q = {
        side: locked_arm_vector(
            observations[side]["q"],
            label=f"move-tracked {side} start qpos",
        )
        for side in ("left", "right")
    }
    arm_qvel = {
        side: np.asarray(observations[side]["qvel"], dtype=np.float64).reshape(
            ARM_DOF
        )
        for side in ("left", "right")
    }
    gripper_q = {
        side: np.asarray(
            ctx.world.gripper_qpos_list(side), dtype=np.float64
        ).reshape(2)
        for side in ("left", "right")
    }
    gripper_qvel = {
        side: np.asarray(
            ctx.world.gripper_qvel_list(side), dtype=np.float64
        ).reshape(2)
        for side in ("left", "right")
    }
    if not all(
        np.all(np.isfinite(value))
        for mapping in (arm_q, arm_qvel, gripper_q, gripper_qvel)
        for value in mapping.values()
    ):
        raise ValueError("current evaluator proprioception contains non-finite values")
    return {
        "state": observations["left"]["state"],
        "trunk": trunk,
        "arm_q": arm_q,
        "arm_qvel": arm_qvel,
        "gripper_q": gripper_q,
        "gripper_qvel": gripper_qvel,
        "eef": {
            side: {
                "position": np.asarray(
                    observations[side]["position"], dtype=np.float64
                ).reshape(3),
                "quaternion": np.asarray(
                    observations[side]["quaternion"], dtype=np.float64
                ).reshape(4),
            }
            for side in ("left", "right")
        },
        "motion_epoch": int(ctx.world.motion_epoch()),
        "episode_id": str(ctx.world.episode_id()),
    }



def _move_tracked_prepare_j8_lock(ctx) -> None:
    """Reset the policy-owned J8 pins before freezing a tracked-point plan."""

    if ARM_DOF != 8:
        return
    set_pin = getattr(ctx.world, "set_tool_roll_pin_qpos", None)
    end_motion = getattr(ctx.world, "end_tool_roll_motion", None)
    if not callable(set_pin) or not callable(end_motion):
        raise RuntimeError("8DOF world does not expose the J8 lock contract")
    for side in ("left", "right"):
        # A prior wrist-roll tool may have temporarily enabled J8.  Revoke
        # that local authorization and make every subsequent official action
        # carry the contract value zero.
        end_motion(side)
        set_pin(side, LOCKED_J8_VALUE_RAD)



def _move_tracked_planning_targets(
    points: Sequence[Mapping[str, Any]],
    point_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return role-free target records, filling omitted XYZ with free axes."""

    by_name = {
        str(point.get("name") or "").strip(): point
        for point in points
        if isinstance(point, Mapping)
    }
    result: list[dict[str, Any]] = []
    for raw_name in point_names:
        name = str(raw_name).strip()
        point = by_name.get(name)
        if point is None:
            raise ValueError(f"move-tracked target point {name!r} is missing")
        target = point.get("target_xyz_m")
        if target is None:
            target = {axis: {"free": True} for axis in ("x", "y", "z")}
        result.append({"name": name, "target_xyz_m": _json_ready(target)})
    return result



def _move_tracked_evaluate_quick_constraints(
    controlled_points: np.ndarray,
    reference_points: np.ndarray | None,
    quick_constraints: Sequence[Mapping[str, Any]],
    *,
    controlled_names: Sequence[str],
    reference_names: Sequence[str],
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate every named preset group and aggregate with max-error semantics."""

    specs = _move_tracked_quick_constraint_specs(
        quick_constraints=quick_constraints
    )
    reports: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        on_names, on_points, off_names, off_points = (
            _move_tracked_quick_group_points(
                spec,
                controlled_names=controlled_names,
                controlled_points=controlled_points,
                reference_names=reference_names,
                reference_points=reference_points,
            )
        )
        report = _evaluate_quick_constraint(
            on_points,
            off_points,
            spec,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_deg=orientation_tolerance_deg,
            camera_position_robot_base_m=camera_position_robot_base_m,
            camera_quaternion_xyzw=camera_quaternion_xyzw,
        )
        reports.append(
            {
                **_json_ready(report),
                "group_index": int(index),
                "quick_constraint": _json_ready(spec),
                "on_hand_point_names": on_names,
                "off_hand_point_names": off_names,
            }
        )
    if len(reports) == 1:
        return {
            **reports[0],
            "group_count": 1,
            "groups": reports,
        }
    errors_m = [float(report.get("max_error_m", 0.0)) for report in reports]
    errors_deg = [float(report.get("max_error_deg", 0.0)) for report in reports]
    first_failure = next((report for report in reports if not report.get("ok")), None)
    return {
        "ok": bool(reports and all(report.get("ok") for report in reports)),
        "applicable": bool(reports),
        "type": "quick_constraint_group_set",
        "group_count": int(len(reports)),
        "groups": reports,
        "max_error_m": max(errors_m, default=0.0),
        "max_error_deg": max(errors_deg, default=0.0),
        "error": (
            None
            if first_failure is None
            else f"quick_constraint_group_{first_failure['group_index'] + 1}: "
            f"{first_failure.get('error') or 'constraint_not_satisfied'}"
        ),
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
    }



def _move_tracked_freeze_plan_capture(
    ctx, *, session_id: str, registration_image_id: str,
    observation_sequence: int,
) -> dict[str, Any]:
    """Freeze the scoring/preview frame without rebinding existing markers."""
    adapter = ctx.world._official_adapter
    if int(adapter.observation_metadata()[0]) != int(observation_sequence):
        raise ValueError("plan RGB-D observation changed before freezing")
    rgb = adapter.camera_frame("head")
    depth = adapter.camera_depth_frame("head")
    relative = deepcopy(adapter.camera_relative_poses().get("head"))
    robot = _capture_robot_state(ctx)
    if rgb is None or depth is None or relative is None:
        raise ValueError("plan requires current head RGB-D and camera-relative pose")
    rgb = _freeze_adapter_camera_array(adapter, rgb)
    depth = _freeze_adapter_camera_array(
        adapter,
        depth,
        dtype=np.float32,
        squeeze=True,
    )
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("plan RGB and depth_linear resolutions do not match")
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise ValueError("plan depth_linear has no valid positive samples")
    position = np.asarray(relative.get("pos"), dtype=np.float64).reshape(3)
    quaternion = np.asarray(relative.get("quat"), dtype=np.float64).reshape(4)
    if not (np.isfinite(position).all() and np.isfinite(quaternion).all()
            and float(np.linalg.norm(quaternion)) > 1e-12):
        raise ValueError("plan camera-relative pose is invalid")
    if int(adapter.observation_metadata()[0]) != int(observation_sequence):
        raise ValueError("plan RGB-D and proprio changed while freezing")
    _ensure_session(session_id)
    image_id = _next_image_id(session_id)
    rgb_path = _image_path(session_id, image_id, ".png")
    depth_path = _image_path(session_id, image_id, ".depth.npy")
    if not cv2.imwrite(rgb_path, rgb):
        raise ValueError("could not persist frozen plan RGB")
    np.save(depth_path, depth, allow_pickle=False)
    height, width = depth.shape
    camera = {
        **_camera_intrinsics("head", width, height),
        "pos": position.tolist(), "quat": quaternion.tolist(),
        "frame": "robot_base", "pose_available": True,
        "source": "official_evaluator_cam_rel_poses",
        "robot_relative_pose": {**relative, "frame": "robot_base"},
    }
    manager = getattr(ctx.world, "_official_tracked_object_distances", None)
    binding = {"ok": False, "trackable": False, "reason": "tracker_unavailable"}
    register_capture = getattr(manager, "register_capture_deferred", None)
    if not callable(register_capture):
        register_capture = getattr(manager, "register_capture", None)
    if callable(register_capture):
        try:
            binding = register_capture(
                session_id=session_id, image_id=image_id,
                observation_sequence=observation_sequence,
                episode_id=robot["episode_id"], image_shape=(height, width),
            )
        except (TrackObjectDistanceBindingError, ValueError) as exc:
            binding = {"ok": False, "trackable": False, "reason": str(exc)}
    _save_image_meta(session_id, image_id, {
        "session_id": session_id, "image_id": image_id,
        "rgb": {"head": os.path.basename(rgb_path)},
        "modalities": {"depth_linear": os.path.basename(depth_path)},
        "camera": camera, "robot": robot,
        "evaluator_sequence": int(observation_sequence),
        "observation_policy": "official_allowlist",
        "track_object_distance_binding": binding,
        "source": "move_tracked_point_current_plan_freeze",
        "registration_image_id": registration_image_id,
    })
    return {
        "image_id": image_id,
        "registration_image_id": registration_image_id,
        "observation_sequence": int(observation_sequence),
        "all_inputs_same_observation": True,
        "track_object_distance_binding": binding,
    }



def _move_tracked_plan_overlap_context(
    session_id: str,
    image_id: str,
) -> dict[str, Any]:
    """Resolve the frozen policy-owned inputs for plan-only collision scoring."""

    session = _validated_session_id(session_id)
    image = str(image_id or "").strip()
    meta = _load_image_meta(session, image)
    if not isinstance(meta, Mapping):
        raise ValueError("move-tracked frozen capture metadata must be an object")
    camera = meta.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("move-tracked plan overlap filter requires camera metadata")
    relative_pose = camera.get("robot_relative_pose")
    if not isinstance(relative_pose, Mapping):
        raise ValueError(
            "move-tracked plan overlap filter requires the frozen evaluator "
            "camera-relative pose"
        )
    try:
        camera_position = np.asarray(
            relative_pose.get("pos"), dtype=np.float64
        ).reshape(3)
        camera_quaternion = np.asarray(
            relative_pose.get("quat"), dtype=np.float64
        ).reshape(4)
        focal_length = float(camera["focal_length"])
        horizontal_aperture = float(camera["horizontal_aperture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "move-tracked plan overlap camera calibration is incomplete"
        ) from exc
    if not (
        np.all(np.isfinite(camera_position))
        and np.all(np.isfinite(camera_quaternion))
        and float(np.linalg.norm(camera_quaternion)) > 1.0e-12
        and math.isfinite(focal_length)
        and focal_length > 0.0
        and math.isfinite(horizontal_aperture)
        and horizontal_aperture > 0.0
    ):
        raise ValueError("move-tracked plan overlap camera calibration is invalid")

    modalities = meta.get("modalities")
    rgb_artifacts = meta.get("rgb")
    if not isinstance(modalities, Mapping) or not isinstance(
        rgb_artifacts, Mapping
    ):
        raise ValueError(
            "move-tracked plan overlap filter requires frozen RGB-D artifacts"
        )
    depth_path = _policy_owned_capture_artifact(
        session, modalities.get("depth_linear")
    )
    rgb_path = _policy_owned_capture_artifact(
        session, rgb_artifacts.get("head")
    )
    if not os.path.isfile(depth_path) or not os.path.isfile(rgb_path):
        raise FileNotFoundError(
            "move-tracked plan overlap frozen RGB-D artifact is unavailable"
        )
    return {
        "session": {
            "session_id": session,
            "image_id": image,
            "init_dir": _images_dir(session),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "tool_label": "move_tracked_point_plan_overlap_filter",
            "observation_source": "official_evaluator_frozen_rgbd_proprio",
        },
        "camera_pos": camera_position.astype(float).tolist(),
        "camera_quat_xyzw": camera_quaternion.astype(float).tolist(),
        "focal_length": focal_length,
        "horizontal_aperture": horizontal_aperture,
        "camera_frame": "robot_base",
        "session_id": session,
        "image_id": image,
    }



def _move_tracked_planning_hold_action(
    ctx,
    capture_state: dict[str, Any],
) -> np.ndarray:
    """Hold every controlled body part while frozen local planning runs."""

    ctx.world.set_trunk_pin_qpos(capture_state["trunk"].astype(float).tolist())
    for side in ("left", "right"):
        ctx.world.set_arm_pin_qpos(
            side,
            capture_state["arm_q"][side].astype(float).tolist(),
        )
    gripper_overrides, _report = _move_point_gripper_action_overrides(
        ctx,
        capture_state,
    )
    return ctx.world.make_action(**gripper_overrides)



def _move_tracked_start_handoff_assessment(
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Classify a finite, position-compatible post-plan handoff.

    A strict stationary sample is ideal, but it is not a prerequisite for a
    safe handoff: the evaluator can report residual joint velocity while the
    measured positions are still inside the frozen-start envelope.  In that
    case execution owns a fixed qpos brake target and the normal trajectory
    tracker/final low-velocity check remains authoritative.
    """

    errors = validation.get("errors")
    if not isinstance(errors, dict):
        return {
            "ok": False,
            "reason": "invalid_start_state",
            "error": "start-state validation returned no numeric errors",
            "mode": "rejected",
            "velocity_threshold_satisfied": False,
        }

    def value(name: str) -> float:
        try:
            return float(errors.get(name, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    required_names = [
        "trunk_inf_rad",
        "trunk_qvel_inf_rad_s",
        "base_qvel_inf",
    ]
    for side in ("left", "right"):
        required_names.extend(
            [
                f"{side}_arm_inf_rad",
                f"{side}_arm_qvel_inf_rad_s",
                f"{side}_gripper_inf_rad",
            ]
        )
    numeric = {name: value(name) for name in required_names}
    # Keep the locked wrist-roll velocity visible in diagnostics.  It has an
    # independent hard safety bound, but is deliberately excluded from the
    # controllable-arm stationary threshold above.
    if ARM_DOF == 8:
        for side in ("left", "right"):
            numeric[f"{side}_j8_qvel_abs_rad_s"] = value(
                f"{side}_j8_qvel_abs_rad_s"
            )
    if not all(math.isfinite(item) for item in numeric.values()):
        return {
            "ok": False,
            "reason": "nonfinite_start_state",
            "error": "start-state proprioception contains a non-finite value",
            "mode": "rejected",
            "velocity_threshold_satisfied": False,
            "errors": numeric,
        }

    position_limits = {
        "trunk_inf_rad": MOVE_TRACKED_POINT_START_HANDOFF_TRUNK_TOLERANCE_RAD,
    }
    for side in ("left", "right"):
        position_limits[f"{side}_arm_inf_rad"] = (
            MOVE_TRACKED_POINT_START_HANDOFF_ARM_TOLERANCE_RAD
        )
        position_limits[f"{side}_gripper_inf_rad"] = (
            MOVE_TRACKED_POINT_START_HANDOFF_GRIPPER_TOLERANCE_RAD
        )
    for name, limit in position_limits.items():
        if numeric[name] > float(limit):
            return {
                "ok": False,
                "reason": "start_position_drift",
                "error": (
                    f"{name}={numeric[name]:.6f} exceeds bounded handoff "
                    f"limit {float(limit):.6f}"
                ),
                "mode": "rejected",
                "velocity_threshold_satisfied": False,
                "errors": numeric,
                "position_limit": float(limit),
                "position_error_name": name,
            }

    hard_velocity_limits = {
        "trunk_qvel_inf_rad_s": MOVE_TRACKED_POINT_START_HANDOFF_TRUNK_QVEL_HARD_RAD_S,
        "base_qvel_inf": MOVE_TRACKED_POINT_START_HANDOFF_BASE_QVEL_HARD,
    }
    for side in ("left", "right"):
        hard_velocity_limits[f"{side}_arm_qvel_inf_rad_s"] = (
            MOVE_TRACKED_POINT_START_HANDOFF_QVEL_HARD_RAD_S
        )
        if ARM_DOF == 8:
            hard_velocity_limits[f"{side}_j8_qvel_abs_rad_s"] = (
                MOVE_TRACKED_POINT_START_HANDOFF_J8_QVEL_HARD_RAD_S
            )
    for name, limit in hard_velocity_limits.items():
        if numeric[name] > float(limit):
            return {
                "ok": False,
                "reason": "start_velocity_hard_limit",
                "error": (
                    f"{name}={numeric[name]:.6f} exceeds bounded handoff "
                    f"limit {float(limit):.6f}"
                ),
                "mode": "rejected",
                "velocity_threshold_satisfied": False,
                "errors": numeric,
                "velocity_limit": float(limit),
                "velocity_error_name": name,
            }

    strict_velocity_ok = bool(
        numeric["trunk_qvel_inf_rad_s"] <= MOVE_POINT_TO_POINT_START_TRUNK_QVEL_RAD_S
        and numeric["base_qvel_inf"] <= MOVE_POINT_TO_POINT_START_BASE_QVEL
        and all(
            numeric[f"{side}_arm_qvel_inf_rad_s"]
            <= MOVE_TRACKED_POINT_START_ARM_QVEL_RAD_S
            for side in ("left", "right")
        )
    )
    return {
        "ok": True,
        "reason": "stationary_threshold" if strict_velocity_ok else "bounded_handoff",
        "error": None,
        "mode": "stationary_threshold" if strict_velocity_ok else "bounded_handoff",
        "velocity_threshold_satisfied": strict_velocity_ok,
        "errors": numeric,
        "position_limits": {
            name: float(limit) for name, limit in position_limits.items()
        },
        "velocity_hard_limits": {
            name: float(limit) for name, limit in hard_velocity_limits.items()
        },
    }



def _move_tracked_handoff_position_snapshot(ctx) -> np.ndarray:
    """Read evaluator-controlled joint positions for handoff motion evidence.

    Gripper coordinates are deliberately excluded because they are expressed
    in metres while trunk and arm coordinates are radians.  Gripper position
    compatibility is still enforced by ``_move_point_start_state_validation``.
    """

    values = [
        np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(-1),
    ]
    for side in ("left", "right"):
        values.append(
            locked_arm_vector(
                ctx.world.arm_qpos_list(side),
                label=f"handoff {side} arm qpos",
            )
        )
    vector = np.concatenate(values).astype(np.float64, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ValueError("handoff proprioception contains a non-finite position")
    return vector



def _move_tracked_start_settle_action(
    ctx,
    capture_state: dict[str, Any],
    validation: dict[str, Any],
    *,
    brake_targets: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Brake at one latched measured qpos while the frozen start is nearby.

    The target is deliberately latched on the first compatible sample.  A
    target recomputed from every subsequent observation follows the moving
    joint and cannot dissipate its velocity.
    """

    if brake_targets is None:
        brake_targets = {}

    errors = validation.get("errors")
    positions_match = bool(
        isinstance(errors, dict)
        and math.isfinite(float(errors.get("trunk_inf_rad", float("inf"))))
        and float(errors["trunk_inf_rad"])
        <= MOVE_TRACKED_POINT_START_HANDOFF_TRUNK_TOLERANCE_RAD
        and all(
            math.isfinite(float(errors.get(f"{side}_arm_inf_rad", float("inf"))))
            and float(errors[f"{side}_arm_inf_rad"])
            <= MOVE_TRACKED_POINT_START_HANDOFF_ARM_TOLERANCE_RAD
            and math.isfinite(
                float(errors.get(f"{side}_gripper_inf_rad", float("inf")))
            )
            and float(errors[f"{side}_gripper_inf_rad"])
            <= MOVE_TRACKED_POINT_START_HANDOFF_GRIPPER_TOLERANCE_RAD
            for side in ("left", "right")
        )
    )
    if not positions_match and not {
        "trunk",
        "left",
        "right",
    }.issubset(brake_targets):
        return _move_tracked_planning_hold_action(ctx, capture_state), "frozen_capture"

    if not {"trunk", "left", "right"}.issubset(brake_targets):
        measured_trunk = np.asarray(
            ctx.world.trunk_qpos(), dtype=np.float64
        ).reshape(-1)
        if not np.all(np.isfinite(measured_trunk)):
            return _move_tracked_planning_hold_action(ctx, capture_state), "frozen_capture"
        brake_targets["trunk"] = measured_trunk.copy()
        for side in ("left", "right"):
            brake_targets[side] = locked_arm_vector(
                ctx.world.arm_qpos_list(side),
                label=f"start-settle measured {side} arm qpos",
            ).copy()

    ctx.world.set_trunk_pin_qpos(
        np.asarray(brake_targets["trunk"], dtype=np.float64).tolist()
    )
    for side in ("left", "right"):
        ctx.world.set_arm_pin_qpos(
            side,
            np.asarray(brake_targets[side], dtype=np.float64).tolist(),
        )
    gripper_overrides, _report = _move_point_gripper_action_overrides(
        ctx,
        capture_state,
    )
    return ctx.world.make_action(**gripper_overrides), "measured_qpos_brake"



def _move_tracked_wait_for_stable_start(
    ctx,
    *,
    capture_state: dict[str, Any],
    manager,
    point_names: list[str],
    source_points: np.ndarray,
    episode_id: str,
    session_id: str,
    image_id: str,
    pos_tol_m: float,
    deadline_monotonic: float,
    source_drift_tolerance_m: float | None = None,
    minimum_tracker_sequence_exclusive: int | None = None,
    rigid_eef_arm: str | None = None,
    rigid_anchors_eef: Any | None = None,
    rigid_anchor_tolerance_m: float | None = None,
    rigid_pair_distance_tolerance_m: float | None = None,
    accept_rigid_rgbd_refinement: bool = True,
):
    """Require synchronized, stationary proprio and tracker samples before motion."""

    stable_steps = 0
    handoff_steps = 0
    settle_steps = 0
    observation_sequence_start: int | None = None
    observation_sequence_end: int | None = None
    max_errors: dict[str, float] = {}
    hold_modes: dict[str, int] = {}
    last_validation: dict[str, Any] = {}
    last_handoff_assessment: dict[str, Any] = {}
    last_source_drift_m = 0.0
    max_handoff_position_step_rad = 0.0
    last_handoff_position_step_rad = float("inf")
    last_handoff_observation_dt_s: float | None = None
    handoff_observation_dt_samples_s: list[float] = []
    last_handoff_derived_velocity_rad_s = float("inf")
    last_handoff_velocity_evidence = "pending"
    last_source_point_drift_m: dict[str, float] = {}
    last_live_points_robot_base_m: np.ndarray | None = None
    last_eef_anchor_error_m: float | None = None
    last_eef_anchor_point_error_m: dict[str, float] = {}
    last_rigid_geometry_consistency: dict[str, Any] = {
        "ok": True,
        "applicable": False,
    }
    raw_source_drift_accepted_as_rgbd_refinement = False
    previous_handoff_positions: np.ndarray | None = None
    previous_handoff_received_ts: float | None = None
    brake_targets: dict[str, np.ndarray] = {}
    tracker_wait_steps = 0
    tracker_wait_streak = 0
    max_tracker_wait_streak = 0
    tracker_wait_reasons: dict[str, int] = {}
    last_tracker_wait_reason: str | None = None
    last_tracker_snapshot: dict[str, Any] = {}
    source_drift_limit_m = (
        max(0.02, 2.0 * float(pos_tol_m))
        if source_drift_tolerance_m is None
        else float(source_drift_tolerance_m)
    )
    if not math.isfinite(source_drift_limit_m) or source_drift_limit_m <= 0.0:
        raise ValueError("source drift tolerance must be positive and finite")
    rigid_handoff_enabled = bool(
        rigid_eef_arm is not None or rigid_anchors_eef is not None
    )
    normalized_rigid_arm: str | None = None
    normalized_rigid_anchors: np.ndarray | None = None
    normalized_anchor_tolerance: float | None = None
    normalized_pair_tolerance: float | None = None
    if rigid_handoff_enabled:
        normalized_rigid_arm = str(rigid_eef_arm or "").strip().lower()
        if normalized_rigid_arm not in ("left", "right"):
            raise ValueError("rigid handoff EEF arm must be left or right")
        if rigid_anchors_eef is None:
            raise ValueError("rigid handoff EEF anchors are required")
        normalized_rigid_anchors = _move_tracked_strict_point_matrix(
            rigid_anchors_eef,
            expected_count=len(point_names),
            label="rigid handoff EEF anchors",
        )
        normalized_anchor_tolerance = float(
            MOVE_TRACKED_POINT_START_EEF_ANCHOR_ERROR_M
            if rigid_anchor_tolerance_m is None
            else rigid_anchor_tolerance_m
        )
        if (
            not math.isfinite(normalized_anchor_tolerance)
            or normalized_anchor_tolerance <= 0.0
            or normalized_anchor_tolerance > MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M
        ):
            raise ValueError(
                "rigid handoff anchor tolerance must be positive, finite, and "
                f"at most {MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M:.3f}m"
            )
        normalized_pair_tolerance = float(
            _tracked_rigid_pair_tolerance(pos_tol_m)
            if rigid_pair_distance_tolerance_m is None
            else rigid_pair_distance_tolerance_m
        )
        if not math.isfinite(normalized_pair_tolerance) or normalized_pair_tolerance <= 0.0:
            raise ValueError(
                "rigid handoff pair-distance tolerance must be positive and finite"
            )

    def report(reason: str, *, ok: bool, error: str | None = None) -> dict[str, Any]:
        payload = {
            "ok": bool(ok),
            "reason": str(reason),
            "error": error,
            "stable_steps": int(stable_steps),
            "required_stable_steps": int(MOVE_TRACKED_POINT_START_STABLE_STEPS),
            "handoff_steps": int(handoff_steps),
            "stationary_velocity_required_for_execution": True,
            "settle_steps": int(settle_steps),
            "max_settle_steps": int(MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS),
            "observation_sequence_start": observation_sequence_start,
            "observation_sequence_end": observation_sequence_end,
            "max_observed_errors": dict(max_errors),
            "last_source_drift_m": float(last_source_drift_m),
            "last_source_point_drift_m": dict(last_source_point_drift_m),
            "source_drift_tolerance_m": float(source_drift_limit_m),
            "source_drift_validation_mode": (
                "live_eef_rigid_anchor_and_pair_geometry"
                if rigid_handoff_enabled
                else "absolute_frozen_xyz"
            ),
            "raw_source_drift_accepted_as_rgbd_refinement": bool(
                raw_source_drift_accepted_as_rgbd_refinement
            ),
            "rigid_rgbd_refinement_allowed": bool(
                accept_rigid_rgbd_refinement
            ),
            "last_live_points_robot_base_m": (
                None
                if last_live_points_robot_base_m is None
                else last_live_points_robot_base_m.astype(float).tolist()
            ),
            "frozen_eef_anchors_preserved": bool(rigid_handoff_enabled),
            "last_eef_anchor_error_m": last_eef_anchor_error_m,
            "last_eef_anchor_point_error_m": dict(
                last_eef_anchor_point_error_m
            ),
            "eef_anchor_tolerance_m": normalized_anchor_tolerance,
            "last_rigid_geometry_consistency": _json_ready(
                last_rigid_geometry_consistency
            ),
            "hold_modes": dict(hold_modes),
            "tracker_wait_steps": int(tracker_wait_steps),
            "tracker_wait_reason_counts": dict(tracker_wait_reasons),
            "max_consecutive_tracker_wait_steps": int(
                max_tracker_wait_streak
            ),
            "last_tracker_wait_reason": last_tracker_wait_reason,
            "last_tracker_snapshot": _json_ready(last_tracker_snapshot),
            "stale_tracker_coordinates_used_while_waiting": False,
            "last_validation": dict(last_validation),
            "handoff_assessment": dict(last_handoff_assessment),
            "handoff_mode": str(
                last_handoff_assessment.get("mode") or "rejected"
            ),
            "velocity_threshold_satisfied": bool(
                last_handoff_assessment.get("velocity_threshold_satisfied", False)
            ),
            "max_handoff_position_step_rad": float(
                max_handoff_position_step_rad
            ),
            "last_handoff_position_step_rad": (
                None
                if not math.isfinite(last_handoff_position_step_rad)
                else float(last_handoff_position_step_rad)
            ),
            "last_handoff_observation_dt_s": (
                None
                if last_handoff_observation_dt_s is None
                else float(last_handoff_observation_dt_s)
            ),
            "handoff_observation_dt_samples_s": [
                float(value) for value in handoff_observation_dt_samples_s
            ],
            "max_handoff_observation_dt_s": (
                None
                if not handoff_observation_dt_samples_s
                else float(max(handoff_observation_dt_samples_s))
            ),
            "last_handoff_derived_velocity_rad_s": (
                None
                if not math.isfinite(last_handoff_derived_velocity_rad_s)
                else float(last_handoff_derived_velocity_rad_s)
            ),
            "handoff_velocity_evidence": str(
                last_handoff_velocity_evidence
            ),
            "handoff_position_step_limit_rad": float(
                MOVE_TRACKED_POINT_START_HANDOFF_POSITION_STEP_RAD
            ),
            "handoff_brake_targets": {
                key: np.asarray(value, dtype=np.float64).astype(float).tolist()
                for key, value in brake_targets.items()
            },
        }
        if "adapter" in last_validation:
            payload["adapter"] = last_validation["adapter"]
            payload["last_validation"] = {
                key: value
                for key, value in last_validation.items()
                if key != "adapter"
            }
        return payload

    while True:
        _check_cancelled(ctx, "move_tracked_point start-state settling")
        if time.monotonic() >= float(deadline_monotonic):
            return report(
                "timeout",
                ok=False,
                error="move_tracked_point timed out while waiting for a stable start",
            )

        start_ok, validation = _move_point_start_state_validation(
            ctx,
            capture_state,
        )
        last_validation = dict(validation)
        current_sequence = validation.get("observation_sequence")
        if current_sequence is not None:
            current_sequence = int(current_sequence)
            if observation_sequence_start is None:
                observation_sequence_start = current_sequence
            observation_sequence_end = current_sequence
        for key, value in (validation.get("errors") or {}).items():
            numeric = float(value)
            if math.isfinite(numeric):
                max_errors[key] = max(max_errors.get(key, 0.0), numeric)

        validation_reason = str(validation.get("reason") or "invalid_start_state")
        if validation_reason not in ("start_state_matches_capture", "stale_capture"):
            return report(validation_reason, ok=False, error=validation_reason)

        handoff_assessment = _move_tracked_start_handoff_assessment(validation)
        last_handoff_assessment = dict(handoff_assessment)
        retryable_hard_velocity = bool(
            not handoff_assessment.get("ok")
            and handoff_assessment.get("reason") == "start_velocity_hard_limit"
            and not str(
                handoff_assessment.get("velocity_error_name") or ""
            ).endswith("_j8_qvel_abs_rad_s")
        )
        if not bool(handoff_assessment.get("ok")) and not retryable_hard_velocity:
            return report(
                str(handoff_assessment.get("reason") or "invalid_start_state"),
                ok=False,
                error=str(
                    handoff_assessment.get("error")
                    or handoff_assessment.get("reason")
                    or "bounded start-state handoff rejected"
                ),
            )

        try:
            current_handoff_positions = _move_tracked_handoff_position_snapshot(ctx)
        except (RuntimeError, TypeError, ValueError) as exc:
            return report(
                "nonfinite_start_state",
                ok=False,
                error=f"handoff proprioception is invalid: {exc}",
            )
        metadata_sequence, current_received_ts_raw = (
            validation["adapter"].observation_metadata()
        )
        current_received_ts: float | None = (
            float(current_received_ts_raw)
            if current_sequence is not None
            and int(metadata_sequence) == int(current_sequence)
            and math.isfinite(float(current_received_ts_raw))
            else None
        )
        last_handoff_position_step_rad = float("inf")
        last_handoff_observation_dt_s = None
        last_handoff_derived_velocity_rad_s = float("inf")
        if previous_handoff_positions is not None:
            last_handoff_position_step_rad = float(
                np.linalg.norm(
                    current_handoff_positions - previous_handoff_positions,
                    ord=np.inf,
                )
            )
            max_handoff_position_step_rad = max(
                max_handoff_position_step_rad,
                last_handoff_position_step_rad,
            )
            if not math.isfinite(last_handoff_position_step_rad):
                return report(
                    "nonfinite_start_state",
                    ok=False,
                    error="handoff proprioception position delta is non-finite",
                )
            observation_dt_s = (
                None
                if previous_handoff_received_ts is None
                or current_received_ts is None
                else float(current_received_ts - previous_handoff_received_ts)
            )
            if (
                observation_dt_s is not None
                and math.isfinite(observation_dt_s)
                and 1.0e-4 <= observation_dt_s <= 5.0
            ):
                last_handoff_observation_dt_s = float(observation_dt_s)
                handoff_observation_dt_samples_s.append(float(observation_dt_s))
                last_handoff_derived_velocity_rad_s = float(
                    last_handoff_position_step_rad / observation_dt_s
                )
        previous_handoff_positions = current_handoff_positions
        previous_handoff_received_ts = current_received_ts

        current_live = _move_tracked_live_snapshot(
            manager,
            point_names,
            episode_id=episode_id,
        )
        last_tracker_snapshot = {
            key: current_live.get(key)
            for key in (
                "ok",
                "reason",
                "error",
                "unavailable",
                "observation_sequence",
                "episode_id",
                "session_id",
                "image_id",
            )
            if key in current_live
        }
        tracker_wait_reason: str | None = None
        if not bool(current_live.get("ok")):
            tracker_reason = str(
                current_live.get("reason") or "tracked_point_unavailable"
            )
            unavailable = current_live.get("unavailable")
            unavailable_reasons = (
                [str(value) for value in unavailable.values()]
                if isinstance(unavailable, dict)
                else []
            )
            retryable_unavailable = bool(
                tracker_reason == "no_synchronized_frame"
                or (
                    tracker_reason == "tracked_point_unavailable"
                    and unavailable_reasons
                    and all(
                        reason
                        in {"point_unobserved", "stale_tracking_measurement"}
                        for reason in unavailable_reasons
                    )
                )
            )
            if retryable_unavailable:
                tracker_wait_reason = tracker_reason
            else:
                return report(
                    tracker_reason,
                    ok=False,
                    error=f"tracked start state unavailable: {tracker_reason}",
                )
        elif current_sequence is None or int(
            current_live.get("observation_sequence", -1)
        ) != int(current_sequence):
            tracker_wait_reason = "tracker_observation_mismatch"
        elif (
            minimum_tracker_sequence_exclusive is not None
            and int(current_live.get("observation_sequence", -1))
            <= int(minimum_tracker_sequence_exclusive)
        ):
            tracker_wait_reason = "awaiting_post_prior_rgbd_observation"
        elif (
            str(current_live.get("session_id")) != str(session_id)
            or str(current_live.get("image_id")) != str(image_id)
        ):
            return report(
                "tracker_binding_replaced",
                ok=False,
                error="tracked-point registration was replaced after planning",
            )

        if tracker_wait_reason is not None:
            tracker_wait_steps += 1
            tracker_wait_streak += 1
            max_tracker_wait_streak = max(
                max_tracker_wait_streak,
                tracker_wait_streak,
            )
            last_tracker_wait_reason = tracker_wait_reason
            tracker_wait_reasons[tracker_wait_reason] = (
                tracker_wait_reasons.get(tracker_wait_reason, 0) + 1
            )
            stable_steps = 0
            last_handoff_assessment.update(
                {
                    "mode": "waiting_for_synchronized_tracker",
                    "reason": tracker_wait_reason,
                }
            )
            if settle_steps >= MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS:
                return report(
                    "tracked_start_not_synchronized",
                    ok=False,
                    error=(
                        "tracked start state did not recover a synchronized "
                        f"observation; last reason: {tracker_wait_reason}"
                    ),
                )

            previous_sequence = int(current_sequence)
            action, hold_mode = _move_tracked_start_settle_action(
                ctx,
                capture_state,
                validation,
                brake_targets=brake_targets,
            )
            hold_modes[hold_mode] = hold_modes.get(hold_mode, 0) + 1
            yield action
            settle_steps += 1
            advanced, advanced_sequence = _adjust_observation_advanced(
                validation["adapter"],
                previous_sequence,
            )
            if not advanced:
                observation_sequence_end = int(advanced_sequence)
                return report(
                    "observation_not_advanced",
                    ok=False,
                    error=(
                        "official evaluator observation/action loop did not "
                        "advance while waiting for synchronized tracking"
                    ),
                )
            continue

        tracker_wait_streak = 0
        try:
            current_points = _move_tracked_strict_point_matrix(
                [
                    current_live["entries"][name][
                        "xyz_in_robot_base_coord_m"
                    ]
                    for name in point_names
                ],
                expected_count=len(point_names),
                label="tracked start point data",
            )
        except (KeyError, TypeError, ValueError) as exc:
            return report(
                "tracked_point_invalid",
                ok=False,
                error=f"tracked start point data is invalid: {exc}",
            )
        if not np.all(np.isfinite(current_points)):
            return report(
                "tracked_point_invalid",
                ok=False,
                error="tracked start point data contains non-finite XYZ",
            )
        last_live_points_robot_base_m = current_points.copy()
        source_drift_errors = np.linalg.norm(current_points - source_points, axis=1)
        last_source_drift_m = float(np.max(source_drift_errors))
        last_source_point_drift_m = {
            str(name): float(source_drift_errors[index])
            for index, name in enumerate(point_names)
        }
        if rigid_handoff_enabled:
            try:
                rigid_observation = _adjust_kinematic_observation(
                    ctx, str(normalized_rigid_arm)
                )
                predicted_points = np.asarray(
                    rigid_observation["position"], dtype=np.float64
                )[None, :] + (
                    _quat_to_mat_xyzw(rigid_observation["quaternion"])
                    @ np.asarray(normalized_rigid_anchors, dtype=np.float64).T
                ).T
                anchor_errors = np.linalg.norm(
                    current_points - predicted_points, axis=1
                )
                last_eef_anchor_error_m = float(np.max(anchor_errors))
                last_eef_anchor_point_error_m = {
                    str(name): float(anchor_errors[index])
                    for index, name in enumerate(point_names)
                }
                last_rigid_geometry_consistency = _tracked_rigid_pair_consistency(
                    current_points,
                    source_points,
                    tolerance_m=float(normalized_pair_tolerance),
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                return report(
                    "invalid_rigid_handoff_observation",
                    ok=False,
                    error=f"could not validate live EEF rigid anchors: {exc}",
                )
            if last_eef_anchor_error_m > float(normalized_anchor_tolerance):
                return report(
                    "tracked_points_not_rigid_with_selected_eef",
                    ok=False,
                    error=(
                        "tracked on-hand point differs from its live EEF anchor by "
                        f"{last_eef_anchor_error_m:.4f}m (limit "
                        f"{float(normalized_anchor_tolerance):.4f}m)"
                    ),
                )
            if not bool(last_rigid_geometry_consistency.get("ok")):
                return report(
                    "tracked_rigid_geometry_changed_during_planning",
                    ok=False,
                    error=(
                        "tracked on-hand point geometry changed after endpoint "
                        "planning"
                    ),
                )
            raw_source_drift_accepted_as_rgbd_refinement = bool(
                last_source_drift_m > source_drift_limit_m
            )
            if (
                raw_source_drift_accepted_as_rgbd_refinement
                and not bool(accept_rigid_rgbd_refinement)
            ):
                return report(
                    "tracked_points_moved_after_source_sync",
                    ok=False,
                    error=(
                        "tracked on-hand points changed "
                        f"{last_source_drift_m:.4f}m after the synchronized "
                        "planning source was frozen; the stale trajectory was "
                        "not executed"
                    ),
                )
        elif last_source_drift_m > source_drift_limit_m:
            return report(
                "tracked_points_moved_during_planning",
                ok=False,
                error=(
                    f"tracked points moved {last_source_drift_m:.4f}m "
                    "after endpoint planning"
                ),
            )

        handoff_steps = handoff_steps + 1
        errors = handoff_assessment.get("errors") or {}
        base_strict_velocity_ok = bool(
            math.isfinite(float(errors.get("base_qvel_inf", float("inf"))))
            and float(errors["base_qvel_inf"])
            <= MOVE_POINT_TO_POINT_START_BASE_QVEL
        )
        controlled_raw_strict_velocity_ok = bool(
            math.isfinite(
                float(errors.get("trunk_qvel_inf_rad_s", float("inf")))
            )
            and float(errors["trunk_qvel_inf_rad_s"])
            <= MOVE_POINT_TO_POINT_START_TRUNK_QVEL_RAD_S
            and all(
                math.isfinite(
                    float(
                        errors.get(
                            f"{side}_arm_qvel_inf_rad_s", float("inf")
                        )
                    )
                )
                and float(errors[f"{side}_arm_qvel_inf_rad_s"])
                <= MOVE_TRACKED_POINT_START_ARM_QVEL_RAD_S
                for side in ("left", "right")
            )
        )
        controlled_derived_velocity_ok = bool(
            last_handoff_observation_dt_s is not None
            and last_handoff_derived_velocity_rad_s
            <= MOVE_TRACKED_POINT_START_ARM_QVEL_RAD_S
        )
        position_step_ok = bool(
            previous_handoff_positions is not None
            and math.isfinite(last_handoff_position_step_rad)
            and last_handoff_position_step_rad
            <= MOVE_TRACKED_POINT_START_HANDOFF_POSITION_STEP_RAD
        )
        effective_velocity_ok = bool(
            base_strict_velocity_ok
            and (
                controlled_raw_strict_velocity_ok
                or controlled_derived_velocity_ok
            )
        )
        if controlled_raw_strict_velocity_ok:
            last_handoff_velocity_evidence = "raw_qvel"
        elif controlled_derived_velocity_ok:
            last_handoff_velocity_evidence = (
                "consecutive_position_delta_over_observation_dt"
            )
        else:
            last_handoff_velocity_evidence = "not_yet_stable"
        if bool(handoff_assessment.get("ok")):
            if not position_step_ok or not effective_velocity_ok:
                last_handoff_assessment["mode"] = "braking"
                last_handoff_assessment["reason"] = "not_yet_stable"
            elif controlled_raw_strict_velocity_ok:
                last_handoff_assessment["mode"] = "stationary_threshold"
                last_handoff_assessment["reason"] = "stationary_threshold"
            else:
                last_handoff_assessment["mode"] = (
                    "position_derived_velocity"
                )
                last_handoff_assessment["reason"] = (
                    "position_derived_velocity"
                )
        last_handoff_assessment.update(
            {
                "raw_velocity_threshold_satisfied": bool(
                    controlled_raw_strict_velocity_ok
                ),
                "derived_velocity_threshold_satisfied": bool(
                    controlled_derived_velocity_ok
                ),
                "base_velocity_threshold_satisfied": bool(
                    base_strict_velocity_ok
                ),
                "velocity_threshold_satisfied": bool(effective_velocity_ok),
                "velocity_evidence": str(last_handoff_velocity_evidence),
                "position_step_satisfied": bool(position_step_ok),
                "position_step_rad": (
                    None
                    if not math.isfinite(last_handoff_position_step_rad)
                    else float(last_handoff_position_step_rad)
                ),
                "observation_dt_s": last_handoff_observation_dt_s,
                "position_derived_velocity_rad_s": (
                    None
                    if not math.isfinite(
                        last_handoff_derived_velocity_rad_s
                    )
                    else float(last_handoff_derived_velocity_rad_s)
                ),
            }
        )
        stationary_sample = bool(
            handoff_assessment.get("ok")
            and effective_velocity_ok
            and position_step_ok
        )
        # Position compatibility uses the wider signed-plan handoff envelope;
        # velocity compatibility never does.  The execution bridge may start
        # from a nearby settled qpos, but a moving arm is always braked first.
        stable_steps = stable_steps + 1 if stationary_sample else 0
        if stable_steps >= MOVE_TRACKED_POINT_START_STABLE_STEPS:
            return report("stable_start_verified", ok=True)
        if settle_steps >= MOVE_TRACKED_POINT_START_MAX_SETTLE_STEPS:
            return report(
                "start_state_not_stable",
                ok=False,
                error=(
                    "start-state proprioception did not remain below velocity "
                    "and position thresholds"
                ),
            )

        previous_sequence = int(current_sequence)
        action, hold_mode = _move_tracked_start_settle_action(
            ctx,
            capture_state,
            validation,
            brake_targets=brake_targets,
        )
        hold_modes[hold_mode] = hold_modes.get(hold_mode, 0) + 1
        yield action
        settle_steps += 1
        advanced, advanced_sequence = _adjust_observation_advanced(
            validation["adapter"],
            previous_sequence,
        )
        if not advanced:
            observation_sequence_end = int(advanced_sequence)
            return report(
                "observation_not_advanced",
                ok=False,
                error=(
                    "official evaluator observation/action loop did not advance "
                    "during start-state settling"
                ),
            )



def _move_tracked_coherent_stability_velocity(
    *,
    arm: str,
    previous_q: np.ndarray,
    current_q: np.ndarray,
    current_qvel: np.ndarray,
    observation_dt_s: float | None,
    previous_incoherence_streak: np.ndarray,
    goal_satisfied: bool,
    hold_q: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fuse bounded qvel with consecutive observation-derived velocity."""

    q = np.asarray(current_q, dtype=np.float64).reshape(ARM_DOF)
    previous = np.asarray(previous_q, dtype=np.float64).reshape(ARM_DOF)
    qvel = np.asarray(current_qvel, dtype=np.float64).reshape(ARM_DOF)
    hold = (
        q.copy()
        if hold_q is None
        else np.asarray(hold_q, dtype=np.float64).reshape(ARM_DOF)
    )
    streak = np.asarray(
        previous_incoherence_streak, dtype=np.int64
    ).reshape(ARM_DOF)
    if not all(
        np.all(np.isfinite(value)) for value in (q, previous, qvel, hold)
    ):
        raise ValueError("final stability proprioception contains non-finite values")
    if np.any(streak < 0):
        raise ValueError("final qvel incoherence streak cannot be negative")

    limits = _ARM_LIMITS[str(arm)][:ARM_DOF]
    position_delta = np.abs(q - previous)
    distance_to_limit = np.minimum(
        np.abs(q - limits[:, 0]),
        np.abs(q - limits[:, 1]),
    )
    limit_chatter = np.logical_and(
        distance_to_limit <= MOVE_POINT_TO_POINT_FINAL_LIMIT_PROXIMITY_RAD,
        position_delta <= MOVE_POINT_TO_POINT_FINAL_LIMIT_POSITION_STEP_RAD,
    )
    if LOCKED_J8_INDEX is not None:
        limit_chatter[LOCKED_J8_INDEX] = False
    effective_qvel = qvel.copy()
    effective_qvel[limit_chatter] = 0.0

    dt = float("nan") if observation_dt_s is None else float(observation_dt_s)
    dt_valid = bool(
        math.isfinite(dt)
        and 1.0e-4 <= dt <= MOVE_TRACKED_POINT_FINAL_COHERENCE_MAX_DT_S
    )
    derived_qvel = np.full(ARM_DOF, float("inf"), dtype=np.float64)
    if dt_valid:
        derived_qvel = position_delta / dt
    bounded_qvel_coherence_candidate = np.logical_and.reduce(
        (
            np.abs(effective_qvel)
            > MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
            np.abs(effective_qvel)
            <= MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S,
            derived_qvel <= MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
            position_delta
            <= MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD,
        )
    )
    command_at_lower = np.abs(hold - limits[:, 0]) <= 1.0e-6
    command_at_upper = np.abs(hold - limits[:, 1]) <= 1.0e-6
    command_limit_direction = np.logical_or(
        np.logical_and.reduce(
            (
                command_at_lower,
                np.abs(q - limits[:, 0])
                <= MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RESIDUAL_RAD,
                qvel < 0.0,
            )
        ),
        np.logical_and.reduce(
            (
                command_at_upper,
                np.abs(q - limits[:, 1])
                <= MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RESIDUAL_RAD,
                qvel > 0.0,
            )
        ),
    )
    command_limit_coherence_candidate = np.logical_and.reduce(
        (
            command_limit_direction,
            np.abs(effective_qvel)
            > MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
            np.abs(effective_qvel)
            <= MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RAW_QVEL_HARD_RAD_S,
            derived_qvel <= MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
            position_delta
            <= MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD,
        )
    )
    coherence_candidate = np.logical_or(
        bounded_qvel_coherence_candidate,
        command_limit_coherence_candidate,
    )
    # Joint stationarity and visual goal satisfaction are independent pieces
    # of evidence.  A one-frame RGB-D constraint fluctuation must not erase a
    # valid run of position-derived velocity samples.  The caller still gates
    # every stable sample on ``goal_satisfied`` below, so accumulating this
    # streak cannot by itself report success.
    if LOCKED_J8_INDEX is not None:
        coherence_candidate[LOCKED_J8_INDEX] = False
    next_streak = np.where(coherence_candidate, streak + 1, 0)
    coherence_noise = np.logical_and(
        coherence_candidate,
        next_streak >= MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS,
    )
    command_limit_coherence_noise = np.logical_and(
        command_limit_coherence_candidate,
        next_streak >= MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS,
    )
    effective_qvel[coherence_noise] = derived_qvel[coherence_noise]

    active = slice(0, min(7, ARM_DOF))
    raw_qvel_inf = float(np.linalg.norm(qvel[active], ord=np.inf))
    effective_qvel_inf = float(
        np.linalg.norm(effective_qvel[active], ord=np.inf)
    )
    position_step_inf = float(
        np.linalg.norm(position_delta[active], ord=np.inf)
    )
    raw_hard_ok_by_joint = np.logical_or.reduce(
        (
            np.abs(qvel)
            <= MOVE_TRACKED_POINT_FINAL_RAW_QVEL_HARD_RAD_S,
            np.logical_and(
                limit_chatter,
                np.abs(qvel)
                <= MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RAW_QVEL_HARD_RAD_S,
            ),
            coherence_noise,
        )
    )
    return {
        "raw_qvel_inf_rad_s": raw_qvel_inf,
        "effective_qvel_inf_rad_s": effective_qvel_inf,
        "position_step_inf_rad": position_step_inf,
        "raw_velocity_hard_ok": bool(np.all(raw_hard_ok_by_joint[active])),
        "raw_qvel_by_joint_rad_s": qvel.astype(float).tolist(),
        "effective_qvel_by_joint_rad_s": (
            effective_qvel.astype(float).tolist()
        ),
        "position_delta_by_joint_rad": position_delta.astype(float).tolist(),
        "derived_qvel_by_joint_rad_s": (
            derived_qvel.astype(float).tolist()
            if dt_valid
            else [None] * ARM_DOF
        ),
        "observation_dt_s": float(dt) if dt_valid else None,
        "filtered_limit_velocity_joints": [
            f"J{index + 1}"
            for index in np.flatnonzero(limit_chatter[active])
        ],
        "coherence_filtered_velocity_joints": [
            f"J{index + 1}"
            for index in np.flatnonzero(coherence_noise[active])
        ],
        "command_limit_coherence_filtered_velocity_joints": [
            f"J{index + 1}"
            for index in np.flatnonzero(command_limit_coherence_noise[active])
        ],
        "qvel_incoherence_streak_by_joint": next_streak.astype(int).tolist(),
    }



def _move_tracked_verify_stable_target(
    ctx,
    *,
    arm: str,
    adapter,
    observation_sequence: int,
    manager,
    point_names: list[str],
    target_points: list[dict[str, Any]],
    fixed_point_names: Sequence[str] | None = None,
    fixed_target_points: Sequence[Mapping[str, Any]] | None = None,
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
    anchors_eef: np.ndarray,
    anchor_tolerance_m: float,
    pair_distance_tolerance_m: float,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    target_q: np.ndarray,
    pos_tol_m: float,
    ori_tol_deg: float,
    timeout_s: float,
    max_steps: int,
    expected_episode_id: str,
    fixed_reference_points: np.ndarray | None = None,
):
    """Verify the immutable tracked goal with bounded endpoint feedback.

    Joint load compensation remains the first recovery path.  If the planned
    EEF endpoint has converged but a coherent multi-point RGB-D bundle carries
    a small rigid offset, the verifier may re-freeze that observed bundle and
    plan a signed local Cartesian correction against the same public goal.
    """

    sequence = int(observation_sequence)
    # ``None`` means that the request has no quick-preset groups.  The
    # canonicalizer intentionally rejects an explicitly empty array on the
    # public wire, so normalize the internal optional value before calling it.
    quick_specs = (
        _move_tracked_quick_constraint_specs(
            quick_constraints=quick_constraints,
        )
        if quick_constraints
        else []
    )
    # The final verifier historically enforced ordinary coordinate/typed
    # relations through ``constraints``.  Only faceto variants need an extra
    # camera-image winding gate here; keeping the legacy quick groups out of
    # this additional gate avoids changing timing/acceptance of old plans.
    faceto_specs = [
        spec
        for spec in quick_specs
        if str(spec.get("type") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in {"faceto", "reverse_faceto"}
    ]
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    if faceto_requested and _move_tracked_camera_pose_payload(
        camera_position_robot_base_m,
        camera_quaternion_xyzw,
    ) is None:
        raise ValueError(
            "faceto/reverse_faceto final verification requires the frozen head-camera pose"
        )
    planned_hold_q = _require_serialized_j8_zero(
        target_q,
        label=f"{arm} move-tracked final target q",
    )
    planned_hold_q, planned_hold_limit_normalizations = (
        _normalize_arm_q_to_joint_limits(
            planned_hold_q,
            arm=arm,
            label=f"{arm} move-tracked final target q",
            point="planned_final_hold",
        )
    )
    original_planned_hold_q = planned_hold_q.copy()
    anchor_array = np.asarray(anchors_eef, dtype=np.float64).reshape(
        len(point_names), 3
    )
    fixed_names = [str(name) for name in (fixed_point_names or [])]
    fixed_targets = [dict(item) for item in (fixed_target_points or [])]
    if len(fixed_names) != len(fixed_targets):
        raise ValueError("fixed tracked point names and targets do not match")
    frozen_fixed_points = None
    if fixed_names and fixed_reference_points is not None:
        frozen_fixed_points = _move_tracked_strict_point_matrix(
            fixed_reference_points,
            expected_count=len(fixed_names),
            label="frozen fixed tracked goal points",
        )
    elif not fixed_names and fixed_reference_points is not None:
        raise ValueError("fixed reference points were supplied without names")
    all_goal_names = list(point_names) + fixed_names
    target_position_array = np.asarray(
        target_position, dtype=np.float64
    ).reshape(3)
    target_quaternion_array = np.asarray(
        target_quaternion, dtype=np.float64
    ).reshape(4)
    observation = _adjust_kinematic_observation(ctx, arm)
    previous_q = observation["q"].copy()
    metadata_sequence, metadata_received_ts = adapter.observation_metadata()
    previous_received_ts: float | None = (
        float(metadata_received_ts)
        if int(metadata_sequence) == int(sequence)
        and math.isfinite(float(metadata_received_ts))
        else None
    )
    qvel_incoherence_streak = np.zeros(ARM_DOF, dtype=np.int64)
    latched_hold_q: np.ndarray | None = None
    latch_limit_normalizations: list[dict[str, Any]] = []
    latch_sequence: int | None = None
    goal_hold_active = False
    goal_latch_brake_active = False
    goal_latch_brake_events: list[dict[str, Any]] = []
    command_hold_q = planned_hold_q.copy()
    load_compensation_waypoints: list[np.ndarray] = []
    load_compensation_rounds: list[dict[str, Any]] = []
    visual_correction_rounds: list[dict[str, Any]] = []
    visual_correction_failure: dict[str, Any] | None = None
    visual_tracking_pause_active = False
    visual_tracking_loss_steps = 0
    visual_tracking_inflight_waypoint: np.ndarray | None = None
    visual_tracking_recovery_events: list[dict[str, Any]] = []
    pending_command_kind: str | None = None
    load_stationary_steps = 0
    stable_steps = 0
    stability_trace: list[dict[str, Any]] = []
    verification_steps = 0
    started = time.monotonic()
    reason = "final_goal_not_reached"
    final_raw_qvel_inf = float("inf")
    final_qvel_inf = float("inf")
    final_position_step_inf = float("inf")
    filtered_limit_velocity_joints: list[str] = []
    coherence_filtered_velocity_joints: list[str] = []
    final_velocity_report: dict[str, Any] = {}
    last_report: dict[str, Any] = {
        "ok": False,
        "reason": "not_sampled",
        "goal_satisfied": False,
    }

    def synchronized_goal_report(
        current_observation: dict[str, Any],
        current_sequence: int,
    ) -> dict[str, Any]:
        live = _move_tracked_live_snapshot(
            manager,
            all_goal_names,
            episode_id=expected_episode_id,
        )
        report: dict[str, Any] = {
            "ok": False,
            "reason": None,
            "observation_sequence": int(current_sequence),
            "tracker_observation_sequence": live.get("observation_sequence"),
            "goal_satisfied": False,
        }
        bundle = _move_tracked_final_observation_bundle(
            live,
            controlled_names=point_names,
            fixed_names=fixed_names,
            current_observation=current_observation,
            anchors_eef=anchor_array,
            current_sequence=current_sequence,
            anchor_tolerance_m=anchor_tolerance_m,
            pos_tol_m=pos_tol_m,
            adapter=adapter,
        )
        report["on_hand_observation_completion"] = _json_ready(bundle)
        if not bool(bundle.get("ok")):
            report["reason"] = str(
                bundle.get("reason")
                or live.get("reason")
                or "tracked_point_unavailable"
            )
            return report
        points = np.asarray(bundle["points"], dtype=np.float64)
        fixed_points = (
            np.asarray(bundle["fixed_points"], dtype=np.float64)
            if fixed_names
            else None
        )
        predicted = np.asarray(bundle["predicted_points"], dtype=np.float64)
        anchor_errors = np.linalg.norm(points - predicted, axis=1)
        pair_consistency = _tracked_rigid_pair_consistency(
            points,
            anchor_array,
            tolerance_m=float(pair_distance_tolerance_m),
        )
        fixed_points_for_constraints = fixed_points
        off_hand_reference = {"ok": True, "applicable": False}
        if fixed_names and frozen_fixed_points is not None:
            off_hand_reference = _move_tracked_off_hand_reference_consensus(
                fixed_names,
                fixed_points,
                frozen_fixed_points,
                tolerance_m=MOVE_TRACKED_POINT_OFF_HAND_REFERENCE_TOLERANCE_M,
                allow_visual_outlier_suppression=True,
            )
            if not bool(off_hand_reference.get("ok")):
                report.update(
                    {
                        "reason": str(
                            off_hand_reference.get("reason")
                            or "off_hand_points_moved"
                        ),
                        "off_hand_reference": off_hand_reference,
                        "fixed_points_robot_base_m": {
                            name: fixed_points[index].astype(float).tolist()
                            for index, name in enumerate(fixed_names)
                        },
                    }
                )
                return report
            # The frozen off-hand coordinates are the requested target.  The
            # live coordinates above are mandatory evidence that those scene
            # points have not moved, but must not redefine the target on every
            # noisy RGB-D observation (especially for short reference lines).
            fixed_points_for_constraints = frozen_fixed_points
        constraints = _tracked_target_constraints(
            points,
            target_points,
            tolerance_m=float(pos_tol_m),
            relations=list(relations or []),
            inequalities=list(inequalities or []),
            orientation_tolerance_deg=float(ori_tol_deg),
            fixed_points_robot_base_m=fixed_points_for_constraints,
            fixed_target_points=(fixed_targets if fixed_names else None),
        )
        quick_report = (
            _move_tracked_evaluate_quick_constraints(
                points,
                fixed_points_for_constraints,
                faceto_specs,
                controlled_names=point_names,
                reference_names=fixed_names,
                position_tolerance_m=float(pos_tol_m),
                orientation_tolerance_deg=float(ori_tol_deg),
                camera_position_robot_base_m=camera_position_robot_base_m,
                camera_quaternion_xyzw=camera_quaternion_xyzw,
            )
            if faceto_specs
            else {
                "ok": True,
                "applicable": False,
                "group_count": 0,
                "groups": [],
            }
        )
        position_error = float(
            np.linalg.norm(
                target_position_array - current_observation["position"]
            )
        )
        orientation_error = orientation_error_deg(
            current_observation["quaternion"],
            target_quaternion_array,
        )
        anchor_ok = bool(
            float(np.max(anchor_errors)) <= float(anchor_tolerance_m)
        )
        pose_ok = bool(
            position_error <= max(0.001, float(pos_tol_m))
            and orientation_error <= max(0.1, float(ori_tol_deg))
        )
        goal_satisfied = bool(
            constraints.get("ok")
            and quick_report.get("ok")
            and pair_consistency.get("ok")
            and anchor_ok
            and pose_ok
        )
        report.update(
            {
                "ok": True,
                "reason": None,
                "goal_satisfied": goal_satisfied,
                "pose_satisfied": pose_ok,
                "anchor_satisfied": anchor_ok,
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "max_anchor_error_m": float(np.max(anchor_errors)),
                "point_anchor_error_m": {
                    name: float(anchor_errors[index])
                    for index, name in enumerate(point_names)
                },
                "rigid_pair_consistency": pair_consistency,
                "points_robot_base_m": {
                    name: points[index].astype(float).tolist()
                    for index, name in enumerate(point_names)
                },
                "fixed_points_robot_base_m": {
                    name: fixed_points[index].astype(float).tolist()
                    for index, name in enumerate(fixed_names)
                }
                if fixed_names
                else {},
                "constraint_fixed_points_robot_base_m": {
                    name: fixed_points_for_constraints[index]
                    .astype(float)
                    .tolist()
                    for index, name in enumerate(fixed_names)
                }
                if fixed_names
                else {},
                "off_hand_reference": off_hand_reference,
                "constraints": constraints,
                "quick_constraint": quick_report,
                "on_hand_observation_completion": _json_ready(bundle),
            }
        )
        return report

    def normalized_observed_latch(
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        normalized, corrections = _normalize_arm_q_to_joint_limits(
            current_q,
            arm=arm,
            label=f"{arm} move-tracked observed final hold q",
            point="observed_goal_latch",
            recover_observed_overshoot=True,
        )
        maximum = max(
            (
                abs(float(item["correction_rad"]))
                for item in corrections
            ),
            default=0.0,
        )
        if maximum > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD:
            raise ValueError(
                "move-tracked observed final hold exceeds the recoverable "
                f"joint-limit residual ({maximum:.12g}rad)"
            )
        return (
            _require_serialized_j8_zero(
                normalized,
                label=f"{arm} move-tracked normalized final hold q",
            ),
            corrections,
        )

    def engage_goal_latch_brake(
        current_observation: Mapping[str, Any],
        current_sequence: int,
        *,
        apply_brake: bool,
        trigger: str,
    ) -> None:
        """Latch goal evidence and brake only when the arm is still moving."""

        nonlocal command_hold_q
        nonlocal goal_hold_active
        nonlocal goal_latch_brake_active
        nonlocal latch_limit_normalizations
        nonlocal latch_sequence
        nonlocal latched_hold_q
        nonlocal load_compensation_waypoints
        nonlocal load_stationary_steps
        nonlocal pending_command_kind
        nonlocal stable_steps
        nonlocal visual_tracking_inflight_waypoint
        nonlocal visual_tracking_loss_steps
        nonlocal visual_tracking_pause_active

        previous_command = command_hold_q.copy()
        latched_hold_q, latch_limit_normalizations = normalized_observed_latch(
            np.asarray(current_observation["q"], dtype=np.float64)
        )
        if apply_brake:
            command_hold_q = latched_hold_q.copy()
        latch_sequence = int(current_sequence)
        goal_hold_active = True
        goal_latch_brake_active = bool(apply_brake)
        load_compensation_waypoints = []
        pending_command_kind = None
        visual_tracking_inflight_waypoint = None
        visual_tracking_pause_active = False
        visual_tracking_loss_steps = 0
        load_stationary_steps = 0
        stable_steps = 0
        qvel_incoherence_streak[:] = 0
        goal_latch_brake_events.append(
            {
                "event": len(goal_latch_brake_events) + 1,
                "observation_sequence": int(current_sequence),
                "trigger": str(trigger),
                "brake_applied": bool(apply_brake),
                "previous_command_q": previous_command.astype(float).tolist(),
                "measured_q": np.asarray(
                    current_observation["q"], dtype=np.float64
                )
                .reshape(ARM_DOF)
                .astype(float)
                .tolist(),
                "brake_command_q": command_hold_q.astype(float).tolist(),
                "previous_command_residual_inf_rad": float(
                    np.linalg.norm(previous_command - command_hold_q, ord=np.inf)
                ),
                "observed_qvel_inf_rad_s": float(
                    np.linalg.norm(
                        np.asarray(
                            current_observation["qvel"], dtype=np.float64
                        ).reshape(ARM_DOF)[: min(7, ARM_DOF)],
                        ord=np.inf,
                    )
                ),
                "joint_limit_normalizations": _json_ready(
                    latch_limit_normalizations
                ),
            }
        )

    def schedule_load_compensation(
        current_observation: dict[str, Any],
        current_goal_report: dict[str, Any],
    ) -> bool:
        nonlocal command_hold_q
        nonlocal load_compensation_waypoints
        nonlocal load_stationary_steps
        nonlocal pending_command_kind

        if load_compensation_waypoints:
            load_stationary_steps = 0
            return False
        if bool(current_goal_report.get("goal_satisfied")):
            load_stationary_steps = 0
            return False

        current_q = np.asarray(
            current_observation["q"], dtype=np.float64
        ).reshape(ARM_DOF)
        current_qvel = np.asarray(
            current_observation["qvel"], dtype=np.float64
        ).reshape(ARM_DOF)
        active = slice(0, min(7, ARM_DOF))
        position_step = float(
            np.linalg.norm(current_q - previous_q, ord=np.inf)
        )
        raw_qvel_inf = float(
            np.linalg.norm(current_qvel[active], ord=np.inf)
        )
        stationary = bool(
            position_step
            <= MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD
            and raw_qvel_inf
            <= MOVE_TRACKED_POINT_FINAL_RAW_QVEL_HARD_RAD_S
        )
        load_stationary_steps = (
            load_stationary_steps + 1 if stationary else 0
        )
        if (
            load_stationary_steps
            < MOVE_TRACKED_POINT_FINAL_LOAD_STABLE_STEPS
            or len(load_compensation_rounds)
            >= MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUNDS
        ):
            return False

        residual = planned_hold_q - current_q
        if LOCKED_J8_INDEX is not None:
            residual[LOCKED_J8_INDEX] = 0.0
        residual_inf = float(
            np.linalg.norm(residual[active], ord=np.inf)
        )
        if residual_inf < MOVE_TRACKED_POINT_FINAL_LOAD_MIN_RESIDUAL_RAD:
            return False

        increment = np.clip(
            residual,
            -MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD,
            MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD,
        )
        limits = _ARM_LIMITS[str(arm)][:ARM_DOF]
        total_lower = np.maximum(
            limits[:, 0],
            planned_hold_q - MOVE_TRACKED_POINT_FINAL_LOAD_MAX_TOTAL_RAD,
        )
        total_upper = np.minimum(
            limits[:, 1],
            planned_hold_q + MOVE_TRACKED_POINT_FINAL_LOAD_MAX_TOTAL_RAD,
        )
        candidate = np.clip(
            command_hold_q + increment,
            total_lower,
            total_upper,
        )
        if LOCKED_J8_INDEX is not None:
            candidate[LOCKED_J8_INDEX] = LOCKED_J8_VALUE_RAD
        candidate = _require_serialized_j8_zero(
            candidate,
            label=f"{arm} move-tracked final load compensation target",
        )
        command_delta = candidate - command_hold_q
        command_delta_inf = float(
            np.linalg.norm(command_delta, ord=np.inf)
        )
        if command_delta_inf <= 1.0e-9:
            return False
        waypoint_count = max(
            1,
            int(
                math.ceil(
                    command_delta_inf / MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
                )
            ),
        )
        command_start = command_hold_q.copy()
        load_compensation_waypoints = [
            _require_serialized_j8_zero(
                command_start
                + (candidate - command_start)
                * (waypoint_index / waypoint_count),
                label=(
                    f"{arm} move-tracked final load compensation waypoint "
                    f"{waypoint_index}"
                ),
            )
            for waypoint_index in range(1, waypoint_count + 1)
        ]
        load_compensation_rounds.append(
            {
                "round": len(load_compensation_rounds) + 1,
                "observed_q": current_q.astype(float).tolist(),
                "planned_q_final": planned_hold_q.astype(float).tolist(),
                "command_start_q": command_start.astype(float).tolist(),
                "command_target_q": candidate.astype(float).tolist(),
                "joint_residual_rad": residual.astype(float).tolist(),
                "joint_residual_inf_rad": residual_inf,
                "command_delta_inf_rad": command_delta_inf,
                "command_waypoint_count": int(waypoint_count),
                "source": "evaluator_proprio_static_load_tracking",
                "cartesian_replan_used": False,
            }
        )
        pending_command_kind = "joint_load_compensation"
        load_stationary_steps = 0
        return True

    def schedule_visual_endpoint_correction(
        current_observation: dict[str, Any],
        current_goal_report: dict[str, Any],
    ) -> str:
        """Replan once from a coherent live point bundle at the reached EEF.

        Returns ``scheduled``, ``not_applicable``, or ``failed``.  A failure is
        terminal because repeatedly solving the same frozen input would only
        consume evaluator time without adding observation evidence.
        """

        nonlocal anchor_array
        nonlocal command_hold_q
        nonlocal latched_hold_q
        nonlocal latch_sequence
        nonlocal load_compensation_waypoints
        nonlocal load_stationary_steps
        nonlocal pending_command_kind
        nonlocal planned_hold_q
        nonlocal target_position_array
        nonlocal target_quaternion_array
        nonlocal visual_correction_failure

        if (
            load_compensation_waypoints
            or bool(current_goal_report.get("goal_satisfied"))
            or len(point_names) < MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_POINTS
            or len(visual_correction_rounds)
            >= MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ROUNDS
            or load_stationary_steps
            < MOVE_TRACKED_POINT_FINAL_LOAD_STABLE_STEPS
        ):
            return "not_applicable"
        if not bool(current_goal_report.get("ok")):
            return "not_applicable"
        if not bool(current_goal_report.get("pose_satisfied")):
            return "not_applicable"

        constraints = current_goal_report.get("constraints")
        rigid_report = current_goal_report.get("rigid_pair_consistency")
        off_hand_report = current_goal_report.get("off_hand_reference")
        if not isinstance(constraints, dict) or bool(constraints.get("ok")):
            return "not_applicable"
        if (
            not isinstance(rigid_report, dict)
            or not bool(rigid_report.get("applicable"))
            or not bool(rigid_report.get("ok"))
        ):
            return "not_applicable"
        if fixed_names and (
            not isinstance(off_hand_report, dict)
            or not bool(off_hand_report.get("ok"))
        ):
            return "not_applicable"

        try:
            maximum_anchor_error = float(
                current_goal_report.get("max_anchor_error_m", float("nan"))
            )
            maximum_constraint_error = float(
                constraints.get("max_constraint_error_m", float("nan"))
            )
        except (TypeError, ValueError):
            return "not_applicable"
        constraint_rebase_limit = min(
            MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M,
            max(0.030, 3.0 * float(pos_tol_m)),
        )
        if (
            not math.isfinite(maximum_anchor_error)
            or maximum_anchor_error
            > min(
                float(anchor_tolerance_m),
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ANCHOR_REBASE_M,
            )
            or not math.isfinite(maximum_constraint_error)
            or maximum_constraint_error > constraint_rebase_limit
        ):
            return "not_applicable"

        remaining_before_plan_s = float(timeout_s) - (
            time.monotonic() - started
        )
        remaining_action_steps = (
            step_budget
            - verification_steps
            - MOVE_POINT_TO_POINT_STABLE_STEPS
        )
        if (
            remaining_before_plan_s
            <= MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_EXECUTION_RESERVE_S
            or remaining_action_steps < 1
        ):
            visual_correction_failure = {
                "reason": "insufficient_correction_budget",
                "error": (
                    "no bounded evaluator time/action budget remains for "
                    "final visual correction"
                ),
                "remaining_time_s": max(0.0, remaining_before_plan_s),
                "remaining_action_steps": max(0, remaining_action_steps),
            }
            return "failed"

        try:
            source_points = _move_tracked_strict_point_matrix(
                [
                    current_goal_report["points_robot_base_m"][name]
                    for name in point_names
                ],
                expected_count=len(point_names),
                label="final visual correction source points",
            )
            fixed_points = (
                _move_tracked_strict_point_matrix(
                    [
                        current_goal_report[
                            "constraint_fixed_points_robot_base_m"
                        ][name]
                        for name in fixed_names
                    ],
                    expected_count=len(fixed_names),
                    label="final visual correction fixed points",
                )
                if fixed_names
                else None
            )
            correction_observations = {
                side: (
                    current_observation
                    if side == arm
                    else _adjust_kinematic_observation(ctx, side)
                )
                for side in ("left", "right")
            }
            correction_capture = _move_tracked_capture_state(
                ctx, correction_observations
            )
            correction_q_start = _require_serialized_j8_zero(
                current_observation["q"],
                label=f"{arm} final visual correction start q",
            )
            planning_budget_s = min(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_PLANNING_S,
                remaining_before_plan_s
                - MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_EXECUTION_RESERVE_S,
            )
            _check_cancelled(ctx, "move_tracked_point final visual planning")
            correction_plan_kwargs = {
                "state": correction_capture["state"],
                "arm": arm,
                "q_start": correction_q_start,
                "source_points": source_points,
                "target_points": [dict(item) for item in target_points],
                "relations": [dict(item) for item in (relations or [])],
                "quick_constraint": None,
                "off_hand_points": fixed_points,
                "off_hand_target_points": [dict(item) for item in fixed_targets],
                "on_hand_names": list(point_names),
                "pos_tol_m": float(pos_tol_m),
                "ori_tol_deg": float(ori_tol_deg),
                "max_steps": max(1, remaining_action_steps),
                "planning_deadline_monotonic": (
                    time.monotonic() + max(0.01, planning_budget_s)
                ),
            }
            if inequalities:
                correction_plan_kwargs["inequalities"] = [
                    dict(item) for item in inequalities
                ]
            # Keep the old correction-call signature byte-for-byte for plans
            # without faceto.  Several downstream integrations monkeypatch
            # this helper, and no-camera plans do not need the new arguments.
            if faceto_requested:
                correction_plan_kwargs.update(
                    {
                        "quick_constraints": list(quick_specs),
                        "off_hand_names": list(fixed_names),
                        "camera_position_robot_base_m": camera_position_robot_base_m,
                        "camera_quaternion_xyzw": camera_quaternion_xyzw,
                    }
                )
            correction_plan = _move_tracked_plan_frozen(**correction_plan_kwargs)
            _check_cancelled(ctx, "move_tracked_point final visual planning")
            if not bool(correction_plan.get("ok")):
                raise ValueError(
                    str(
                        correction_plan.get("error")
                        or "no valid final visual correction path"
                    )
                )

            original_endpoint = correction_plan["endpoint"]
            original_path_plan = correction_plan["path_plan"]
            trimmed_correction = (
                _move_tracked_trim_visual_correction_to_first_feasible(
                    state=correction_capture["state"],
                    arm=arm,
                    q_start=correction_q_start,
                    endpoint=original_endpoint,
                    path_plan=original_path_plan,
                    target_points=[dict(item) for item in target_points],
                    relations=[dict(item) for item in (relations or [])],
                    inequalities=[dict(item) for item in (inequalities or [])],
                    fixed_points_robot_base_m=fixed_points,
                    fixed_target_points=[dict(item) for item in fixed_targets],
                    pos_tol_m=float(pos_tol_m),
                    ori_tol_deg=float(ori_tol_deg),
                )
            )
            endpoint = trimmed_correction["endpoint"]
            path_plan = trimmed_correction["path_plan"]
            correction_prefix_trim = trimmed_correction["report"]
            endpoint_q = _require_serialized_j8_zero(
                endpoint["q_final"],
                label=f"{arm} final visual correction endpoint q",
            )
            endpoint_position = np.asarray(
                endpoint["final_eef_position_m"], dtype=np.float64
            ).reshape(3)
            endpoint_quaternion = np.asarray(
                endpoint["final_eef_quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
            endpoint_anchors = _move_tracked_strict_point_matrix(
                endpoint["anchors_eef_m"],
                expected_count=len(point_names),
                label="final visual correction EEF anchors",
            )
            correction_translation = float(
                np.linalg.norm(
                    endpoint_position
                    - np.asarray(current_observation["position"], dtype=np.float64)
                )
            )
            if (
                not math.isfinite(correction_translation)
                or correction_translation
                > MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_TRANSLATION_M
            ):
                raise ValueError(
                    "final visual correction EEF translation exceeds the "
                    f"bounded {MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_TRANSLATION_M:.3f}m envelope"
                )
            if not bool(endpoint.get("constraints", {}).get("ok")):
                raise ValueError(
                    "final visual correction endpoint does not satisfy the immutable goal"
                )

            dense_waypoints = list(path_plan.get("waypoints") or [])
            remaining_after_plan_s = float(timeout_s) - (
                time.monotonic() - started
            )
            remaining_action_steps = (
                step_budget
                - verification_steps
                - MOVE_POINT_TO_POINT_STABLE_STEPS
            )
            if remaining_after_plan_s <= 0.0 or remaining_action_steps < 1:
                raise ValueError(
                    "final visual correction planning consumed its execution budget"
                )
            correction_trajectory = {
                "tracking": {
                    "max_joint_step_rad": float(
                        path_plan.get(
                            "max_joint_step_rad",
                            MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
                        )
                    )
                },
                "waypoints": dense_waypoints,
                "planner": {
                    "same_branch_checked": bool(
                        path_plan.get("same_branch_checked")
                    ),
                    "monotonic_progress_checked": bool(
                        path_plan.get("monotonic_progress_checked")
                    ),
                    "internal_stop_count": int(
                        path_plan.get("internal_stop_count", 0)
                    ),
                },
            }
            execution_schedule = _move_tracked_prepare_execution_schedule(
                trajectory=correction_trajectory,
                state=correction_capture["state"],
                arm=arm,
                start_q=correction_q_start,
                max_steps=max(1, remaining_action_steps),
                available_time_s=max(0.01, remaining_after_plan_s),
                observation_dt_s=None,
                visual_tracked_point_count=_move_tracked_visual_point_count(
                    manager, point_names
                ),
            )
            if not bool(execution_schedule.get("ok")):
                raise ValueError(
                    str(
                        execution_schedule.get("error")
                        or "final visual correction cannot fit execution budget"
                    )
                )
            scheduled_waypoints = [
                _require_serialized_j8_zero(
                    raw_q,
                    label=(
                        f"{arm} final visual correction waypoint {index + 1}"
                    ),
                )
                for index, raw_q in enumerate(
                    execution_schedule.get("waypoints") or []
                )
            ]
            if not scheduled_waypoints:
                raise ValueError("final visual correction path is empty")

            correction_record = _json_ready(
                {
                    "schema": (
                        "official_v2_move_tracked_point_final_visual_correction"
                    ),
                    "schema_version": 1,
                    "round": len(visual_correction_rounds) + 1,
                    "arm": arm,
                    "episode_id": expected_episode_id,
                    "source_observation_sequence": int(
                        current_goal_report["observation_sequence"]
                    ),
                    "source_points_robot_base_m": source_points,
                    "fixed_points_robot_base_m": fixed_points,
                    "target_points": target_points,
                    "relations": list(relations or []),
                    "inequalities": list(inequalities or []),
                    "q_start": correction_q_start,
                    "q_final": endpoint_q,
                    "anchors_eef_m": endpoint_anchors,
                    "eef_position_robot_base_m": endpoint_position,
                    "eef_quaternion_xyzw": endpoint_quaternion,
                    "dense_waypoint_count": len(dense_waypoints),
                    "waypoints": scheduled_waypoints,
                    "execution_schedule": execution_schedule,
                }
            )
            correction_record["integrity"] = {
                "algorithm": "hmac-sha256",
                "digest": _trajectory_digest(correction_record),
            }
            if not hmac.compare_digest(
                str(correction_record["integrity"]["digest"]),
                _trajectory_digest(correction_record),
            ):
                raise ValueError("final visual correction integrity check failed")
        except Exception as exc:
            if _move_point_is_cancellation(exc):
                raise
            visual_correction_failure = {
                "reason": (
                    "planning_timeout"
                    if isinstance(exc, PlanningDeadlineExceeded)
                    else "planning_failed"
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "source_observation_sequence": int(
                    current_goal_report.get("observation_sequence", -1)
                ),
                "source_max_anchor_error_m": maximum_anchor_error,
                "source_max_constraint_error_m": maximum_constraint_error,
            }
            return "failed"

        planned_hold_q = endpoint_q.copy()
        target_position_array = endpoint_position.copy()
        target_quaternion_array = endpoint_quaternion.copy()
        anchor_array = endpoint_anchors.copy()
        load_compensation_waypoints = [
            np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
            for raw_q in correction_record["waypoints"]
        ]
        pending_command_kind = "visual_endpoint_correction"
        latched_hold_q = None
        latch_sequence = None
        load_stationary_steps = 0
        visual_correction_rounds.append(
            {
                "round": len(visual_correction_rounds) + 1,
                "source_observation_sequence": int(
                    current_goal_report["observation_sequence"]
                ),
                "source_max_anchor_error_m": maximum_anchor_error,
                "source_max_constraint_error_m": maximum_constraint_error,
                "constraint_rebase_limit_m": float(constraint_rebase_limit),
                "eef_translation_m": correction_translation,
                "original_solver_eef_translation_m": float(
                    correction_prefix_trim["original_endpoint_translation_m"]
                ),
                "dense_waypoint_count": len(dense_waypoints),
                "execution_waypoint_count": len(load_compensation_waypoints),
                "trajectory_digest": correction_record["integrity"]["digest"],
                "trajectory_schema": correction_record["schema"],
                "trajectory_schema_version": correction_record["schema_version"],
                "source": "synchronized_evaluator_rgbd_and_proprio_local_replan",
                "cartesian_replan_used": True,
                "execution_schedule": _json_ready(execution_schedule),
                "resolved_endpoint_constraints": _json_ready(
                    endpoint.get("constraints")
                ),
                "first_feasible_prefix": _json_ready(
                    correction_prefix_trim
                ),
            }
        )
        return "scheduled"

    last_report = synchronized_goal_report(observation, sequence)
    if bool(last_report.get("goal_satisfied")):
        initial_qvel_inf = float(
            np.linalg.norm(
                np.asarray(observation["qvel"], dtype=np.float64).reshape(
                    ARM_DOF
                )[: min(7, ARM_DOF)],
                ord=np.inf,
            )
        )
        engage_goal_latch_brake(
            observation,
            sequence,
            apply_brake=(
                initial_qvel_inf > MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S
            ),
            trigger="initial_goal_observation",
        )

    step_budget = max(
        1,
        min(int(max_steps), int(MOVE_TRACKED_POINT_FINAL_MAX_STABILITY_STEPS)),
    )
    while verification_steps < step_budget:
        _check_cancelled(
            ctx,
            f"move_tracked_point final stability step {verification_steps + 1}",
        )
        if str(ctx.world.episode_id()) != str(expected_episode_id):
            reason = "episode_changed"
            break
        if time.monotonic() - started >= max(0.1, float(timeout_s)):
            reason = "timeout"
            break

        if load_compensation_waypoints and not visual_tracking_pause_active:
            command_hold_q = load_compensation_waypoints.pop(0).copy()
            if pending_command_kind == "visual_endpoint_correction":
                visual_tracking_inflight_waypoint = command_hold_q.copy()
            stable_steps = 0
        hold_q = command_hold_q
        yield _adjust_action(ctx, arm, hold_q)
        verification_steps += 1
        advanced, current_sequence = _adjust_observation_advanced(
            adapter, sequence
        )
        if not advanced:
            reason = "observation_not_advanced"
            break
        sequence = int(current_sequence)
        if str(ctx.world.episode_id()) != str(expected_episode_id):
            reason = "episode_changed"
            break

        observation = _adjust_kinematic_observation(ctx, arm)
        metadata_sequence, current_received_ts_raw = (
            adapter.observation_metadata()
        )
        current_received_ts: float | None = (
            float(current_received_ts_raw)
            if int(metadata_sequence) == int(sequence)
            and math.isfinite(float(current_received_ts_raw))
            else None
        )
        last_report = synchronized_goal_report(observation, sequence)
        rigid_report = last_report.get("rigid_pair_consistency")
        npoint_rigid_geometry_inconsistent = bool(
            _move_tracked_visual_point_count(manager, point_names)
            >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
            and bool(last_report.get("ok"))
            and isinstance(rigid_report, dict)
            and bool(rigid_report.get("applicable"))
            and not bool(rigid_report.get("ok"))
        )
        npoint_visual_evidence_unavailable = bool(
            _move_tracked_visual_point_count(manager, point_names)
            >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
            and not bool(last_report.get("ok"))
        )
        visual_tracker_block_reason = (
            "tracked_point_rigid_geometry_inconsistent"
            if npoint_rigid_geometry_inconsistent
            else str(
                last_report.get("reason")
                or "tracked_point_visual_evidence_unavailable"
            )
        )
        visual_tracker_blocked = bool(
            (
                pending_command_kind == "visual_endpoint_correction"
                and not bool(last_report.get("ok"))
            )
            or npoint_visual_evidence_unavailable
            or npoint_rigid_geometry_inconsistent
        )
        if visual_tracker_blocked:
            if not visual_tracking_pause_active:
                if visual_tracking_inflight_waypoint is not None:
                    load_compensation_waypoints.insert(
                        0, visual_tracking_inflight_waypoint.copy()
                    )
                    visual_tracking_inflight_waypoint = None
                try:
                    pause_q, pause_normalizations = (
                        _normalize_arm_q_to_joint_limits(
                            observation["q"],
                            arm=arm,
                            label=(
                                f"{arm} final visual correction tracking-loss hold q"
                            ),
                            point="final_visual_tracking_loss_hold",
                            recover_observed_overshoot=True,
                        )
                    )
                    command_hold_q = _require_serialized_j8_zero(
                        pause_q,
                        label=(
                            f"{arm} final visual correction tracking-loss hold q"
                        ),
                    )
                except (TypeError, ValueError) as exc:
                    visual_correction_failure = {
                        "reason": "tracking_loss_hold_invalid",
                        "error": (
                            "cannot form a finite measured hold while final "
                            f"visual tracking is unavailable: {exc}"
                        ),
                        "observation_sequence": int(sequence),
                    }
                    reason = "final_visual_correction_tracking_lost"
                    break
                visual_tracking_pause_active = True
                visual_tracking_recovery_events.append(
                    {
                        "event": len(visual_tracking_recovery_events) + 1,
                        "start_observation_sequence": int(sequence),
                        "start_reason": visual_tracker_block_reason,
                        "measured_hold_q": command_hold_q.astype(float).tolist(),
                        "joint_limit_normalizations": _json_ready(
                            pause_normalizations
                        ),
                        "requeued_inflight_waypoint": bool(
                            load_compensation_waypoints
                        ),
                        "recovered": False,
                    }
                )
            visual_tracking_loss_steps += 1
            if visual_tracking_recovery_events:
                visual_tracking_recovery_events[-1].update(
                    {
                        "wait_steps": int(visual_tracking_loss_steps),
                        "last_observation_sequence": int(sequence),
                        "last_reason": visual_tracker_block_reason,
                    }
                )
            if (
                visual_tracking_loss_steps
                >= MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS
            ):
                rigid_geometry_failure = bool(
                    visual_tracker_block_reason
                    == "tracked_point_rigid_geometry_inconsistent"
                )
                visual_correction_failure = {
                    "reason": "tracked_point_reacquisition_failed",
                    "error": (
                        (
                            "final tracked-point rigid geometry remained "
                            "inconsistent for "
                            f"{MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS} "
                            "official observation steps while holding measured "
                            "joints"
                        )
                        if rigid_geometry_failure
                        else (
                            "final tracked-point visual evidence did not "
                            "reacquire within "
                            f"{MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS} "
                            "official observation steps while holding measured "
                            "joints"
                        )
                    ),
                    "observation_sequence": int(sequence),
                    "last_tracker_reason": visual_tracker_block_reason,
                    "recovery_steps": int(visual_tracking_loss_steps),
                }
                reason = "final_visual_correction_tracking_lost"
                break
        elif visual_tracking_pause_active:
            visual_tracking_pause_active = False
            visual_tracking_loss_steps = 0
            visual_tracking_inflight_waypoint = None
            if visual_tracking_recovery_events:
                visual_tracking_recovery_events[-1].update(
                    {
                        "recovered": True,
                        "recovered_observation_sequence": int(sequence),
                    }
                )
        if bool(last_report.get("goal_satisfied")):
            # The first satisfying sample in each contiguous goal window takes
            # over the controller with a measured-q brake.  Keeping the old
            # planned/load-compensation setpoint here can drive a loaded arm
            # back out of an already valid tracked-point goal indefinitely.
            goal_qvel_inf = float(
                np.linalg.norm(
                    np.asarray(observation["qvel"], dtype=np.float64).reshape(
                        ARM_DOF
                    )[: min(7, ARM_DOF)],
                    ord=np.inf,
                )
            )
            brake_required = bool(
                goal_qvel_inf > MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S
            )
            if not goal_hold_active:
                engage_goal_latch_brake(
                    observation,
                    sequence,
                    apply_brake=brake_required,
                    trigger="goal_reentered_after_correction",
                )
            elif brake_required and not goal_latch_brake_active:
                engage_goal_latch_brake(
                    observation,
                    sequence,
                    apply_brake=True,
                    trigger="goal_velocity_requires_measured_q_brake",
                )
            load_compensation_waypoints = []
            pending_command_kind = None
            visual_tracking_inflight_waypoint = None
            visual_tracking_pause_active = False
            visual_tracking_loss_steps = 0
            load_stationary_steps = 0
        elif not visual_tracker_blocked:
            goal_hold_active = False
            goal_latch_brake_active = False
            scheduled_compensation = schedule_load_compensation(
                observation,
                last_report,
            )
            if scheduled_compensation:
                stable_steps = 0
            else:
                visual_status = schedule_visual_endpoint_correction(
                    observation,
                    last_report,
                )
                if visual_status == "scheduled":
                    visual_tracking_inflight_waypoint = None
                    visual_tracking_pause_active = False
                    visual_tracking_loss_steps = 0
                    stable_steps = 0
                elif visual_status == "failed":
                    reason = "final_visual_correction_failed"
                    break

        observation_dt_s = (
            None
            if previous_received_ts is None or current_received_ts is None
            else float(current_received_ts - previous_received_ts)
        )
        final_velocity_report = _move_tracked_coherent_stability_velocity(
            arm=arm,
            previous_q=previous_q,
            current_q=observation["q"],
            current_qvel=observation["qvel"],
            observation_dt_s=observation_dt_s,
            previous_incoherence_streak=qvel_incoherence_streak,
            goal_satisfied=bool(
                latched_hold_q is not None
                and last_report.get("goal_satisfied")
            ),
            hold_q=command_hold_q,
        )
        qvel_incoherence_streak = np.asarray(
            final_velocity_report["qvel_incoherence_streak_by_joint"],
            dtype=np.int64,
        )
        final_raw_qvel_inf = float(
            final_velocity_report["raw_qvel_inf_rad_s"]
        )
        final_qvel_inf = float(
            final_velocity_report["effective_qvel_inf_rad_s"]
        )
        final_position_step_inf = float(
            final_velocity_report["position_step_inf_rad"]
        )
        filtered_limit_velocity_joints = list(
            final_velocity_report["filtered_limit_velocity_joints"]
        )
        coherence_filtered_velocity_joints = list(
            final_velocity_report["coherence_filtered_velocity_joints"]
        )
        previous_q = observation["q"].copy()
        previous_received_ts = current_received_ts

        sample_stable = bool(
            latched_hold_q is not None
            and last_report.get("goal_satisfied")
            and not load_compensation_waypoints
            and final_qvel_inf <= MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S
            and bool(final_velocity_report.get("raw_velocity_hard_ok"))
            and abs(float(observation["observed_j8"]))
            <= LOCKED_J8_CAPTURE_TOLERANCE_RAD
        )
        stable_steps = stable_steps + 1 if sample_stable else 0
        stability_trace.append(
            {
                "observation_sequence": int(sequence),
                "goal_satisfied": bool(last_report.get("goal_satisfied")),
                "goal_reason": last_report.get("reason"),
                "pending_command_kind": pending_command_kind,
                "stable_sample": bool(sample_stable),
                "stable_steps": int(stable_steps),
                "raw_qvel_inf_rad_s": float(final_raw_qvel_inf),
                "effective_qvel_inf_rad_s": float(final_qvel_inf),
                "position_step_inf_rad": float(final_position_step_inf),
                "raw_velocity_hard_ok": bool(
                    final_velocity_report.get("raw_velocity_hard_ok")
                ),
                "coherence_filtered_velocity_joints": list(
                    coherence_filtered_velocity_joints
                ),
                "command_limit_coherence_filtered_velocity_joints": list(
                    final_velocity_report.get(
                        "command_limit_coherence_filtered_velocity_joints", []
                    )
                ),
                "qvel_incoherence_streak_by_joint": list(
                    final_velocity_report.get(
                        "qvel_incoherence_streak_by_joint", []
                    )
                ),
            }
        )
        if len(stability_trace) > 16:
            del stability_trace[:-16]
        if stable_steps >= MOVE_POINT_TO_POINT_STABLE_STEPS:
            reason = "stable_tracked_target_converged"
            break

    if stable_steps < MOVE_POINT_TO_POINT_STABLE_STEPS and reason == "final_goal_not_reached":
        if latched_hold_q is not None and bool(last_report.get("goal_satisfied")):
            reason = "final_velocity_not_stable"
        elif last_report.get("reason"):
            reason = str(last_report["reason"])

    final_hold_q = command_hold_q
    ctx.world.set_arm_pin_qpos(arm, final_hold_q)
    return {
        "ok": bool(stable_steps >= MOVE_POINT_TO_POINT_STABLE_STEPS),
        "reason": reason,
        "error": (
            None
            if stable_steps >= MOVE_POINT_TO_POINT_STABLE_STEPS
            else (
                str(visual_correction_failure.get("error"))
                if reason
                in {
                    "final_visual_correction_failed",
                    "final_visual_correction_tracking_lost",
                }
                and isinstance(visual_correction_failure, dict)
                else (
                    "final live tracked-point goal was never observed"
                    if latched_hold_q is None
                    else (
                        "final tracked-point goal was observed but arm velocity "
                        "did not settle"
                    )
                )
            )
        ),
        "required_stable_steps": int(MOVE_POINT_TO_POINT_STABLE_STEPS),
        "stable_steps": int(stable_steps),
        "verification_steps": int(verification_steps),
        "verification_step_budget": int(step_budget),
        "arm_qvel_threshold_rad_s": float(
            MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S
        ),
        "raw_arm_qvel_hard_threshold_rad_s": float(
            MOVE_TRACKED_POINT_FINAL_RAW_QVEL_HARD_RAD_S
        ),
        "incoherent_raw_qvel_hard_threshold_rad_s": float(
            MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S
        ),
        "command_limit_residual_rad": float(
            MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RESIDUAL_RAD
        ),
        "command_limit_raw_qvel_hard_rad_s": float(
            MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RAW_QVEL_HARD_RAD_S
        ),
        "qvel_coherence_required_steps": int(
            MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
        ),
        "qvel_coherence_position_step_rad": float(
            MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD
        ),
        "final_arm_qvel_inf_rad_s": float(final_qvel_inf),
        "final_raw_arm_qvel_inf_rad_s": float(final_raw_qvel_inf),
        "final_position_step_inf_rad": float(final_position_step_inf),
        "filtered_limit_velocity_joints": list(
            filtered_limit_velocity_joints
        ),
        "coherence_filtered_velocity_joints": list(
            coherence_filtered_velocity_joints
        ),
        "final_velocity_report": _json_ready(final_velocity_report),
        "recent_stability_trace": _json_ready(stability_trace),
        "goal_latched": latched_hold_q is not None,
        "goal_latch_brake_active": bool(goal_latch_brake_active),
        "goal_latch_observation_sequence": latch_sequence,
        "goal_latch_brake_policy": (
            "preserve_static_load_offset_but_hold_measured_q_while_moving"
        ),
        "goal_latch_brake_events": _json_ready(goal_latch_brake_events),
        "goal_latch_brake_event_count": int(len(goal_latch_brake_events)),
        "planned_hold_joint_limit_normalizations": _json_ready(
            planned_hold_limit_normalizations
        ),
        "latched_hold_joint_limit_normalizations": _json_ready(
            latch_limit_normalizations
        ),
        "initial_planned_target_q": original_planned_hold_q.astype(float).tolist(),
        "planned_target_q": planned_hold_q.astype(float).tolist(),
        "hold_target_q": final_hold_q.astype(float).tolist(),
        "final_load_compensation_rounds": _json_ready(
            load_compensation_rounds
        ),
        "final_load_compensation_round_count": int(
            len(load_compensation_rounds)
        ),
        "final_load_compensation_pending_waypoints": int(
            len(load_compensation_waypoints)
        ),
        "pending_command_kind": pending_command_kind,
        "final_visual_tracking_pause_active": bool(
            visual_tracking_pause_active
        ),
        "final_visual_tracking_loss_steps": int(
            visual_tracking_loss_steps
        ),
        "final_visual_tracking_recovery_events": _json_ready(
            visual_tracking_recovery_events
        ),
        "final_load_compensation": {
            "stationary_steps_required": int(
                MOVE_TRACKED_POINT_FINAL_LOAD_STABLE_STEPS
            ),
            "minimum_residual_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MIN_RESIDUAL_RAD
            ),
            "maximum_rounds": int(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUNDS
            ),
            "maximum_round_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD
            ),
            "maximum_total_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_TOTAL_RAD
            ),
            "cartesian_replan_used": False,
        },
        "final_visual_correction_rounds": _json_ready(
            visual_correction_rounds
        ),
        "final_visual_correction_round_count": int(
            len(visual_correction_rounds)
        ),
        "final_visual_correction_failure": _json_ready(
            visual_correction_failure
        ),
        "final_visual_correction": {
            "minimum_controlled_points": int(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_POINTS
            ),
            "maximum_rounds": int(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ROUNDS
            ),
            "maximum_anchor_rebase_m": float(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_ANCHOR_REBASE_M
            ),
            "maximum_constraint_error_m": float(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M
            ),
            "maximum_eef_translation_m": float(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_TRANSLATION_M
            ),
            "maximum_planning_s": float(
                MOVE_TRACKED_POINT_FINAL_VISUAL_MAX_PLANNING_S
            ),
            "tracking_reacquire_max_steps": int(
                MOVE_TRACKED_POINT_FINAL_VISUAL_REACQUIRE_MAX_STEPS
            ),
            "trajectory_integrity": "hmac-sha256",
            "simulator_state_used": False,
        },
        "last_observation_sequence": int(sequence),
        "last_live_goal_report": last_report,
    }



def _move_tracked_compact_overlap_audit(
    raw_report: Any,
) -> Any:
    """Bound persisted overlap diagnostics without changing decision evidence.

    RGB-D Lite reconstruction reports contain large topology component tables.
    Those tables explain how a local mesh was built but are not consumed by the
    move-plan loader or by ``exec_plan_pose``. Persist their exact digest and
    byte count while retaining the overlap thresholds, per-pose measurements,
    aggregate counts, and selection evidence used by execution validation.
    """

    report = _json_ready(raw_report)
    if not isinstance(report, Mapping):
        return report
    compact = dict(report)
    raw_batches = compact.get("batches")
    if not isinstance(raw_batches, list):
        return compact
    compact_batches: list[Any] = []
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, Mapping):
            compact_batches.append(raw_batch)
            continue
        batch = dict(raw_batch)
        for field in ("reconstruction", "occupancy", "inflation"):
            diagnostic = batch.get(field)
            if diagnostic is None:
                continue
            diagnostic_bytes = _canonical_json_bytes(diagnostic)
            if len(diagnostic_bytes) <= MOVE_TRACKED_PLAN_INLINE_DIAGNOSTIC_BYTES:
                continue
            batch[field] = {
                "plan_record_compacted": True,
                "reason": "non_execution_overlap_diagnostic",
                "canonical_size_bytes": int(len(diagnostic_bytes)),
                "sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
            }
        compact_batches.append(batch)
    compact["batches"] = compact_batches
    return compact



def _move_tracked_compact_planner_audits(
    raw_planner: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the signed, execution-complete bounded planner audit."""

    planner = dict(_json_ready(raw_planner))
    if "frozen_rgbd_overlap_filter" in planner:
        planner["frozen_rgbd_overlap_filter"] = (
            _move_tracked_compact_overlap_audit(
                planner["frozen_rgbd_overlap_filter"]
            )
        )
    diagnostics = planner.get("planning_diagnostics")
    if isinstance(diagnostics, Mapping):
        diagnostics = dict(diagnostics)
        if "overlap_batches" in diagnostics:
            diagnostics["overlap_batches"] = (
                _move_tracked_compact_overlap_audit(
                    {"batches": diagnostics["overlap_batches"]}
                ).get("batches", [])
            )
        planner["planning_diagnostics"] = diagnostics
    return planner



def _move_tracked_build_trajectory(
    *,
    plan_id: str,
    session_id: str,
    image_id: str,
    observation_sequence: int,
    args: dict[str, Any],
    capture_state: dict[str, Any],
    arm: str,
    source_names: list[str],
    point_a_name: str | None,
    relations: Sequence[Mapping[str, Any]] | None = None,
    source_points: np.ndarray,
    endpoint: dict[str, Any],
    path_plan: dict[str, Any],
    controlled_names: Sequence[str] | None = None,
    reference_names: Sequence[str] | None = None,
    reference_points: np.ndarray | None = None,
    reference_target_points: Sequence[Mapping[str, Any]] | None = None,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    planning_target_points: Sequence[Mapping[str, Any]] | None = None,
    planning_relations: Sequence[Mapping[str, Any]] | None = None,
    head_camera_position_robot_base_m: Sequence[float] | None = None,
    head_camera_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    def strict_point_matrix(raw: Any, *, label: str) -> np.ndarray:
        try:
            matrix = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an N x 3 array") from exc
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise ValueError(f"{label} must be an N x 3 array")
        if not 1 <= matrix.shape[0] <= MOVE_TRACKED_POINT_MAX_POINTS:
            raise ValueError(
                f"{label} must contain one to {MOVE_TRACKED_POINT_MAX_POINTS} points"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{label} must contain finite coordinates")
        return matrix.copy()

    quick_specs = _move_tracked_quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    quick_mode = bool(quick_specs)
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
            "faceto/reverse_faceto trajectory requires the frozen head-camera pose"
        )
    source_points = strict_point_matrix(source_points, label="move-tracked source points")
    controlled_names = [
        str(name) for name in (controlled_names if controlled_names is not None else source_names)
    ]
    reference_names = [str(name) for name in (reference_names or [])]
    if len(controlled_names) != len(source_points):
        raise ValueError("move-tracked controlled names do not match source points")
    if set(controlled_names) & set(reference_names):
        raise ValueError("move-tracked controlled and reference names overlap")
    if reference_names:
        if reference_points is None:
            raise ValueError("move-tracked off-hand reference points are missing")
        reference_points = strict_point_matrix(
            reference_points, label="move-tracked off-hand reference points"
        )
        if reference_points.shape != (len(reference_names), 3):
            raise ValueError("move-tracked off-hand reference points do not match names")
    elif reference_points is not None:
        raise ValueError("move-tracked reference points supplied without names")
    if not 1 <= len(source_points) <= MOVE_TRACKED_POINT_MAX_POINTS:
        raise ValueError(
            "move-tracked trajectory requires one to six controlled source points"
        )
    if (
        len(source_names) != len(controlled_names) + len(reference_names)
        or len(set(source_names)) != len(source_names)
        or set(source_names) != set(controlled_names) | set(reference_names)
    ):
        raise ValueError("move-tracked complete point binding is invalid")
    resolved_points = strict_point_matrix(
        endpoint["resolved_points_robot_base_m"],
        label="move-tracked endpoint resolved points",
    )
    if resolved_points.shape != source_points.shape:
        raise ValueError("move-tracked endpoint resolved points do not match source points")
    anchors = strict_point_matrix(
        endpoint["anchors_eef_m"],
        label="move-tracked endpoint anchors",
    )
    if anchors.shape != source_points.shape:
        raise ValueError("move-tracked endpoint anchors do not match source points")
    relation_payload = [
        dict(item)
        for item in (
            planning_relations
            if planning_relations is not None
            else (relations or args.get("relations") or [])
        )
    ]
    inequality_payload = [
        dict(item) for item in endpoint.get(
            "inequalities", args.get("inequalities") or []
        )
    ]
    if _json_ready(inequality_payload) != _json_ready(
        args.get("inequalities") or []
    ):
        raise ValueError(
            "move-tracked endpoint inequalities do not match the canonical request"
        )
    geometry_report = _tracked_rigid_geometry_report(
        source_points,
        resolved_points,
        tolerance_m=float(args["pos_tol"]),
    )
    raw_waypoints = list(path_plan["waypoints"])
    waypoints: list[list[float]] = []
    waypoint_limit_normalizations: list[dict[str, Any]] = []
    for waypoint_index, raw_waypoint in enumerate(raw_waypoints):
        normalized, corrections = _normalize_arm_q_to_joint_limits(
            raw_waypoint,
            arm=arm,
            label=f"move-tracked planned waypoint {waypoint_index + 1}",
            point=f"waypoint_{waypoint_index + 1}",
        )
        waypoints.append(normalized.astype(float).tolist())
        waypoint_limit_normalizations.extend(corrections)
    normalized_endpoint_q, endpoint_limit_normalizations = (
        _normalize_arm_q_to_joint_limits(
            endpoint["q_final"],
            arm=arm,
            label="move-tracked planned endpoint q",
            point="endpoint",
        )
    )
    if bool(endpoint.get("trunk_assisted", False)):
        raise ValueError(
            "move-tracked endpoint requested forbidden trunk motion; "
            "the arm-only planner must produce the endpoint"
        )
    # The signed plan may contain more samples than the requested online
    # action budget.  It is still bounded by MAX_TRAJECTORY_WAYPOINTS and is
    # compressed by the execution-only scheduler after start-state validation.
    # Rejecting it here would prevent the scheduler from solving exactly the
    # low-budget case it was designed to handle.
    if len(waypoints) > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("move-tracked trajectory contains too many waypoints")
    trajectory: dict[str, Any] = {
        "schema": MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA,
        "schema_version": MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "session_id": str(session_id),
        "tool_version": TOOL_VERSION,
        "source_tool": "move_tracked_point",
        "source_image_id": str(image_id),
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": int(ARM_DOF),
        "action_dim": int(ACTION_DIM),
        "active_arm": str(arm),
        "locked_arm_joints": (
            {"J8": LOCKED_J8_VALUE_RAD}
            if LOCKED_J8_INDEX is not None
            else {}
        ),
        "tracking_binding": {
            "session_id": str(session_id),
            "image_id": str(image_id),
            "point_names": list(controlled_names),
            "point_count": int(len(controlled_names)),
            "point_a_name": (
                str(point_a_name) if point_a_name is not None else None
            ),
            "observation_sequence": int(observation_sequence),
            "frame": "current_robot_base",
            "all_points_same_rigid_object_policy_assertion": (
                (not quick_mode) and len(controlled_names) >= 2
            ),
            # Kept for loading older diagnostic consumers; it is no longer
            # used as the execution decision for N-point requests.
            "two_points_same_rigid_object_policy_assertion": len(source_names) == 2,
            "controlled_point_names": list(controlled_names),
            "reference_point_names": list(reference_names),
            "tracking_point_names": list(source_names),
            "tracking_point_count": int(len(source_names)),
            "predefined_eef_point_names": [
                name for name in source_names
                if is_predefined_eef_point_name(name)
            ],
            "visual_tracked_point_names": [
                name for name in source_names
                if not is_predefined_eef_point_name(name)
            ],
            "predefined_eef_arm": (
                str(arm)
                if any(is_predefined_eef_point_name(name) for name in source_names)
                else None
            ),
            "quick_constraint_mode": bool(quick_mode),
        },
        "start_state": {
            "episode_id": str(capture_state["episode_id"]),
            "motion_epoch": int(capture_state["motion_epoch"]),
            "trunk_qpos": capture_state["trunk"].astype(float).tolist(),
            "arm_qpos": {
                side: capture_state["arm_q"][side].astype(float).tolist()
                for side in ("left", "right")
            },
            "arm_qvel": {
                side: capture_state["arm_qvel"][side].astype(float).tolist()
                for side in ("left", "right")
            },
            "gripper_qpos": {
                side: capture_state["gripper_q"][side].astype(float).tolist()
                for side in ("left", "right")
            },
            "gripper_qvel": {
                side: capture_state["gripper_qvel"][side].astype(float).tolist()
                for side in ("left", "right")
            },
        },
        "parameters": _json_ready(args),
        "target": {
            # Keep the normalized request beside the solved arrays.  The
            # loader validates this declarative input again, so a future
            # producer cannot accidentally sign a trajectory whose relation
            # metadata no longer describes the requested coordinates.
            "requested_points": _json_ready(args["points"]),
            "planning_target_points": _json_ready(
                list(planning_target_points)
                if planning_target_points is not None
                else args["points"]
            ),
            "source_points_robot_base_m": source_points.astype(float).tolist(),
            "resolved_points_robot_base_m": resolved_points.astype(float).tolist(),
            "point_count": int(len(source_points)),
            "point_names": list(controlled_names),
            "anchors_eef_m": anchors.astype(float).tolist(),
            "eef_position_robot_base_m": endpoint[
                "final_eef_position_m"
            ].astype(float).tolist(),
            "eef_quaternion_xyzw": endpoint[
                "final_eef_quaternion_xyzw"
            ].astype(float).tolist(),
            "endpoint_q": normalized_endpoint_q.astype(float).tolist(),
            "rigid_transform": endpoint["rigid_transform"],
            "constraints": _json_ready(endpoint["constraints"]),
            "relations": _json_ready(relation_payload),
            "requested_relations": _json_ready(args.get("relations") or []),
            "inequalities": _json_ready(inequality_payload),
            "requested_inequalities": _json_ready(
                args.get("inequalities") or []
            ),
            "quick_constraint": _json_ready(quick_constraint) if quick_mode else None,
            "quick_constraints": (
                _json_ready(quick_specs) if quick_constraints is not None else None
            ),
            "faceto_camera_pose": _json_ready(faceto_camera_pose),
            "on_hand_point_names": list(controlled_names),
            "off_hand_point_names": list(reference_names),
            "predefined_eef_point_names": [
                name for name in controlled_names
                if is_predefined_eef_point_name(name)
            ],
            "off_hand_source_points_robot_base_m": (
                None
                if reference_points is None
                else reference_points.astype(float).tolist()
            ),
            "off_hand_target_points": _json_ready(
                list(reference_target_points or [])
            ),
            "geometry_rank": int(endpoint.get("geometry_rank", geometry_report["source_rank"])),
            "geometry_preflight": _json_ready(endpoint.get("geometry_preflight", {})),
            "geometry_report": _json_ready(geometry_report),
            "kabsch_seed": _json_ready(endpoint.get("kabsch_seed")),
            "overlap_volume_filter": _json_ready(
                endpoint.get("overlap_volume_filter")
            ),
        },
        "tracking": {
            "execution_mode": "continuous_waypoint_stream",
            "point_count": int(len(controlled_names)),
            "point_names": list(controlled_names),
            # Keep the complete tracker binding in the signed tracking block
            # as well as ``tracking_binding``.  The loader validates both
            # copies before allowing an action, so a role list cannot be
            # silently dropped or reordered between planning and execution.
            "tracking_point_names": list(source_names),
            "tracking_point_count": int(len(source_names)),
            "predefined_eef_point_names": [
                name for name in source_names
                if is_predefined_eef_point_name(name)
            ],
            "visual_tracked_point_names": [
                name for name in source_names
                if not is_predefined_eef_point_name(name)
            ],
            "predefined_eef_arm": (
                str(arm)
                if any(is_predefined_eef_point_name(name) for name in source_names)
                else None
            ),
            "all_points_same_rigid_object_policy_assertion": (
                (not quick_mode) and len(controlled_names) >= 2
            ),
            "max_joint_step_rad": MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
            "waypoint_tolerance_rad": MOVE_POINT_TO_POINT_WAYPOINT_TOLERANCE_RAD,
            "max_tracking_error_rad": (
                MOVE_TRACKED_POINT_STREAM_TRACKING_ERROR_RAD
            ),
            "stall_steps": int(ADJUST_EEF_STALL_STEPS),
            "final_arm_qvel_threshold_rad_s": MOVE_POINT_TO_POINT_FINAL_ARM_QVEL_RAD_S,
            "final_raw_arm_qvel_hard_threshold_rad_s": (
                MOVE_TRACKED_POINT_FINAL_RAW_QVEL_HARD_RAD_S
            ),
            "final_incoherent_raw_qvel_hard_threshold_rad_s": (
                MOVE_TRACKED_POINT_FINAL_INCOHERENT_RAW_QVEL_HARD_RAD_S
            ),
            "final_qvel_coherence_steps": int(
                MOVE_TRACKED_POINT_FINAL_QVEL_COHERENCE_STEPS
            ),
            "final_qvel_coherence_position_step_rad": (
                MOVE_TRACKED_POINT_FINAL_COHERENCE_POSITION_STEP_RAD
            ),
            "final_qvel_coherence_max_dt_s": (
                MOVE_TRACKED_POINT_FINAL_COHERENCE_MAX_DT_S
            ),
            "final_command_limit_residual_rad": (
                MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RESIDUAL_RAD
            ),
            "final_command_limit_raw_qvel_hard_rad_s": (
                MOVE_TRACKED_POINT_FINAL_COMMAND_LIMIT_RAW_QVEL_HARD_RAD_S
            ),
            "final_load_stable_steps": int(
                MOVE_TRACKED_POINT_FINAL_LOAD_STABLE_STEPS
            ),
            "final_load_min_residual_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MIN_RESIDUAL_RAD
            ),
            "final_load_max_rounds": int(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUNDS
            ),
            "final_load_max_round_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_ROUND_RAD
            ),
            "final_load_max_total_rad": float(
                MOVE_TRACKED_POINT_FINAL_LOAD_MAX_TOTAL_RAD
            ),
            "stable_steps": int(MOVE_POINT_TO_POINT_STABLE_STEPS),
            "rigid_anchor_tolerance_m": max(
                MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M,
                4.0 * float(args["pos_tol"]),
            ),
            "rigid_pair_distance_tolerance_m": (
                _tracked_rigid_pair_tolerance(float(args["pos_tol"]))
            ),
            "max_live_progress_regression_m": (
                MOVE_TRACKED_POINT_MAX_LIVE_PROGRESS_REGRESSION_M
            ),
            "live_point_chord_regression_is_diagnostic": True,
            "controlled_point_count": int(len(controlled_names)),
            "controlled_point_names": list(controlled_names),
            "reference_point_count": int(len(reference_names)),
            "reference_point_names": list(reference_names),
            "quick_constraint_mode": bool(quick_mode),
        },
        "waypoints": waypoints,
        "planner": {
            **_move_tracked_compact_planner_audits(path_plan),
            "endpoint_solver": str(endpoint["solver"]),
            "endpoint_selected_candidate": _json_ready(
                endpoint["selected_candidate"]
            ),
            "endpoint_feasible_candidate_count": int(
                endpoint["feasible_candidate_count"]
            ),
            "endpoint_path_candidate_rank": int(
                endpoint.get("path_candidate_rank", 0)
            ),
            "endpoint_path_candidates_tested": int(
                endpoint.get("path_candidates_tested", 1)
            ),
            "endpoint_path_candidate_failures": _json_ready(
                endpoint.get("path_candidate_failures", [])
            ),
            "object_truth_used": False,
            "waypoint_joint_limit_normalizations": _json_ready(
                waypoint_limit_normalizations
            ),
            "endpoint_joint_limit_normalizations": _json_ready(
                endpoint_limit_normalizations
            ),
        },
    }
    trajectory["planner"].pop("waypoints", None)
    trajectory["planner"].pop("trunk_waypoints", None)
    trajectory["integrity"] = {
        "algorithm": "hmac-sha256",
        "digest": _trajectory_digest(trajectory),
    }
    return trajectory



def _move_tracked_write_plan(trajectory: dict[str, Any]) -> str:
    path = _plan_record_path(
        str(trajectory["session_id"]), str(trajectory["plan_id"])
    )
    _write_plan_record(
        path,
        {
            "ok": True,
            "tool": "move_tracked_point",
            "tool_version": TOOL_VERSION,
            "session_id": trajectory["session_id"],
            "plan_id": trajectory["plan_id"],
            "trajectory": trajectory,
        },
    )
    return path



def _move_tracked_load_plan(
    session_id: str,
    plan_id: str,
) -> tuple[dict[str, Any], str]:
    path = _plan_record_path(session_id, plan_id)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        stat = os.fstat(descriptor)
        if stat.st_size <= 0 or stat.st_size > MAX_PLAN_RECORD_BYTES:
            raise ValueError("move-tracked plan record has an invalid size")
        raw = os.read(descriptor, stat.st_size + 1)
    finally:
        os.close(descriptor)
    if len(raw) != stat.st_size:
        raise ValueError("move-tracked plan record changed while it was read")
    record = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value!r}")
        ),
    )
    if not isinstance(record, dict) or record.get("tool") != "move_tracked_point":
        raise ValueError("move-tracked plan record has the wrong source tool")
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("move-tracked plan record has no trajectory")
    expected = {
        "schema": MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA,
        "schema_version": MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "session_id": str(session_id),
        "tool_version": TOOL_VERSION,
        "source_tool": "move_tracked_point",
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": int(ARM_DOF),
        "action_dim": int(ACTION_DIM),
    }
    for key, value in expected.items():
        if trajectory.get(key) != value:
            raise ValueError(
                f"move-tracked trajectory {key} does not match the active contract"
            )
    integrity = trajectory.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "hmac-sha256"
        or not hmac.compare_digest(
            str(integrity.get("digest", "")),
            _trajectory_digest(trajectory),
        )
    ):
        raise ValueError("move-tracked trajectory digest mismatch")
    arm = str(trajectory.get("active_arm", ""))
    if arm not in ("left", "right"):
        raise ValueError("move-tracked trajectory active arm is invalid")
    expected_locked = (
        {"J8": LOCKED_J8_VALUE_RAD}
        if LOCKED_J8_INDEX is not None
        else {}
    )
    if trajectory.get("locked_arm_joints") != expected_locked:
        raise ValueError("move-tracked trajectory locked-joint contract is invalid")
    start_state = trajectory.get("start_state")
    binding = trajectory.get("tracking_binding")
    target = trajectory.get("target")
    planner = trajectory.get("planner")
    tracking = trajectory.get("tracking")
    if not all(
        isinstance(value, dict)
        for value in (start_state, binding, target, planner, tracking)
    ):
        raise ValueError("move-tracked trajectory metadata is incomplete")
    requested_quick = target.get("quick_constraint")
    requested_quick_groups = target.get("quick_constraints")
    quick_specs = _move_tracked_quick_constraint_specs(
        quick_constraint=requested_quick,
        quick_constraints=requested_quick_groups,
    )
    quick_mode = bool(quick_specs)
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    faceto_camera_position: np.ndarray | None = None
    faceto_camera_quaternion: np.ndarray | None = None
    if faceto_requested:
        raw_faceto_camera = target.get("faceto_camera_pose")
        if not isinstance(raw_faceto_camera, Mapping):
            raise ValueError(
                "move-tracked faceto trajectory is missing its frozen head-camera pose"
            )
        try:
            faceto_camera_position = np.asarray(
                raw_faceto_camera["position_robot_base_m"], dtype=np.float64
            ).reshape(3)
            faceto_camera_quaternion = np.asarray(
                raw_faceto_camera["quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "move-tracked faceto trajectory camera pose is invalid"
            ) from exc
        canonical_camera = _move_tracked_camera_pose_payload(
            faceto_camera_position,
            faceto_camera_quaternion,
        )
        if _json_ready(canonical_camera) != _json_ready(dict(raw_faceto_camera)):
            raise ValueError(
                "move-tracked faceto trajectory camera pose is not canonical"
            )

    # The point bundle is part of the signed plan, not an incidental diagnostic
    # field.  Validate its cardinality, order, shapes, and relation references
    # before any action can be emitted.  This closes the old two-point-only
    # loader gap while retaining ``point_a_name`` as a backwards-compatible
    # diagnostic field.
    def strict_point_count(
        raw: Any,
        *,
        label: str,
        maximum: int = MOVE_TRACKED_POINT_MAX_POINTS,
    ) -> int:
        # JSON booleans compare equal to integers in Python, and ``int(1.5)``
        # silently truncates.  Neither is a valid signed trajectory cardinality.
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{label} must be an integer")
        count = int(raw)
        if not 1 <= count <= int(maximum):
            raise ValueError(f"{label} must be in 1..{int(maximum)}")
        return count

    def strict_bool(raw: Any, *, label: str) -> bool:
        # Do not let JSON strings/integers pass through ``bool(...)``.  These
        # flags are part of the signed N-point contract and a malformed value
        # must fail closed instead of changing meaning during loading.
        if not isinstance(raw, bool):
            raise ValueError(f"{label} must be a boolean")
        return bool(raw)

    raw_binding_names = binding.get("point_names")
    if not isinstance(raw_binding_names, (list, tuple)):
        raise ValueError("move-tracked trajectory point_names must be an array")
    point_count = strict_point_count(
        binding.get("point_count"),
        label="move-tracked trajectory point_count",
    )
    if any(
        not isinstance(name, str)
        or not name
        or name != name.strip()
        for name in raw_binding_names
    ):
        raise ValueError("move-tracked trajectory point names must be non-empty strings")
    point_names = list(raw_binding_names)
    if len(point_names) != point_count:
        raise ValueError("move-tracked trajectory point count does not match names")
    if len(set(point_names)) != point_count or any(not name for name in point_names):
        raise ValueError("move-tracked trajectory point names are not unique")
    point_a_name = binding.get("point_a_name")
    if point_a_name is not None and (
        not isinstance(point_a_name, str) or point_a_name not in point_names
    ):
        raise ValueError("move-tracked trajectory point_a_name is not bound")
    if strict_bool(
        binding.get("all_points_same_rigid_object_policy_assertion"),
        label="move-tracked trajectory all-point rigid-object assertion",
    ) != ((not quick_mode) and point_count >= 2):
        raise ValueError("move-tracked trajectory all-point rigid-object assertion is invalid")
    raw_target_names = target.get("point_names")
    if not isinstance(raw_target_names, list):
        raise ValueError("move-tracked trajectory target point_names must be an array")
    target_count = strict_point_count(
        target.get("point_count"),
        label="move-tracked trajectory target point_count",
    )
    if target_count != point_count or raw_target_names != point_names:
        raise ValueError("move-tracked trajectory target point binding does not match")
    tracking_names = tracking.get("point_names")
    if not isinstance(tracking_names, list) or tracking_names != point_names:
        raise ValueError("move-tracked trajectory tracking point names do not match")
    tracking_count = strict_point_count(
        tracking.get("point_count"),
        label="move-tracked trajectory tracking point_count",
    )
    if tracking_count != point_count:
        raise ValueError("move-tracked trajectory tracking point count is invalid")
    if strict_bool(
        tracking.get("all_points_same_rigid_object_policy_assertion"),
        label="move-tracked trajectory tracking rigid-object assertion",
    ) != ((not quick_mode) and point_count >= 2):
        raise ValueError("move-tracked trajectory tracking rigid-object assertion is invalid")

    # The legacy point fields remain the controlled rigid bundle consumed by
    # FK/path code.  Explicit role fields carry the complete 1..6-point tracker
    # binding for both coordinate-only and quick-constraint requests.
    raw_controlled = binding.get("controlled_point_names")
    raw_references = binding.get("reference_point_names")
    raw_all = binding.get("tracking_point_names")
    if not isinstance(raw_controlled, list) or raw_controlled != point_names:
        raise ValueError("move-tracked controlled point binding is invalid")
    if not isinstance(raw_references, list):
        raise ValueError("move-tracked reference point binding is invalid")
    if not isinstance(raw_all, list):
        raise ValueError("move-tracked complete tracking point names are missing")
    role_reference_names = [str(name) for name in raw_references]
    role_tracking_names = [str(name) for name in raw_all]
    if not 1 <= len(role_tracking_names) <= MOVE_TRACKED_POINT_MAX_POINTS:
        raise ValueError("move-tracked complete tracking point count is invalid")
    if (
        any(not name or name != name.strip() for name in role_tracking_names)
        or len(set(role_tracking_names)) != len(role_tracking_names)
    ):
        raise ValueError("move-tracked complete tracking point names are invalid")
    if set(role_tracking_names) != set(point_names) | set(role_reference_names):
        raise ValueError("move-tracked complete tracking binding does not match role names")
    if set(point_names) & set(role_reference_names):
        raise ValueError("move-tracked controlled/reference names overlap")
    for container, label in (
        (binding, "binding"),
        (tracking, "tracking"),
    ):
        raw_tracking_count = container.get("tracking_point_count")
        if isinstance(raw_tracking_count, bool) or not isinstance(raw_tracking_count, int):
            raise ValueError(f"move-tracked {label} tracking point count is invalid")
        if int(raw_tracking_count) != len(role_tracking_names):
            raise ValueError(
                f"move-tracked {label} tracking point count does not match names"
            )
        if container.get("tracking_point_names") != role_tracking_names:
            raise ValueError(
                f"move-tracked {label} complete tracking metadata is invalid"
            )
        if strict_bool(
            container.get("quick_constraint_mode"),
            label=f"move-tracked {label} quick-constraint flag",
        ) != quick_mode:
            raise ValueError(
                f"move-tracked {label} quick-constraint flag is inconsistent"
            )
        expected_predefined_names = [
            name for name in role_tracking_names
            if is_predefined_eef_point_name(name)
        ]
        expected_visual_names = [
            name for name in role_tracking_names
            if not is_predefined_eef_point_name(name)
        ]
        if (
            "predefined_eef_point_names" in container
            and container.get("predefined_eef_point_names")
            != expected_predefined_names
        ):
            raise ValueError(
                f"move-tracked {label} predefined EEF point binding is invalid"
            )
        if (
            "visual_tracked_point_names" in container
            and container.get("visual_tracked_point_names")
            != expected_visual_names
        ):
            raise ValueError(
                f"move-tracked {label} visual point binding is invalid"
            )
        if "predefined_eef_arm" in container:
            expected_predefined_arm = arm if expected_predefined_names else None
            if container.get("predefined_eef_arm") != expected_predefined_arm:
                raise ValueError(
                    f"move-tracked {label} predefined EEF arm binding is invalid"
                )
    if tracking.get("controlled_point_names") != point_names:
        raise ValueError("move-tracked tracking controlled point names are invalid")
    if tracking.get("reference_point_names") != role_reference_names:
        raise ValueError("move-tracked tracking reference point names are invalid")
    if tracking.get("controlled_point_count") != len(point_names):
        raise ValueError("move-tracked tracking controlled point count is invalid")
    if tracking.get("reference_point_count") != len(role_reference_names):
        raise ValueError("move-tracked tracking reference point count is invalid")

    def point_matrix(field: str) -> np.ndarray:
        try:
            matrix = np.asarray(target[field], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"move-tracked trajectory target {field} must have shape ({point_count}, 3)"
            ) from exc
        if matrix.shape != (point_count, 3):
            raise ValueError(
                f"move-tracked trajectory target {field} must have shape ({point_count}, 3)"
            )
        matrix = matrix.copy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"move-tracked trajectory target {field} contains non-finite values")
        return matrix

    source_matrix = point_matrix("source_points_robot_base_m")
    resolved_matrix = point_matrix("resolved_points_robot_base_m")
    try:
        anchor_matrix = np.asarray(target["anchors_eef_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"move-tracked trajectory anchors must have shape ({point_count}, 3)"
        ) from exc
    if anchor_matrix.shape != (point_count, 3):
        raise ValueError(
            f"move-tracked trajectory anchors must have shape ({point_count}, 3)"
        )
    anchor_matrix = anchor_matrix.copy()
    if not np.all(np.isfinite(anchor_matrix)):
        raise ValueError("move-tracked trajectory anchors contain non-finite values")
    try:
        endpoint_position = np.asarray(
            target["eef_position_robot_base_m"], dtype=np.float64
        ).reshape(3)
        endpoint_quaternion = np.asarray(
            target["eef_quaternion_xyzw"], dtype=np.float64
        ).reshape(4)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("move-tracked trajectory endpoint EEF pose is invalid") from exc
    if not np.all(np.isfinite(endpoint_position)) or not np.all(np.isfinite(endpoint_quaternion)):
        raise ValueError("move-tracked trajectory endpoint EEF pose is non-finite")
    resolved_from_pose = endpoint_position[None, :] + (
        _quat_to_mat_xyzw(endpoint_quaternion) @ anchor_matrix.T
    ).T
    if not np.allclose(resolved_matrix, resolved_from_pose, atol=1.0e-7, rtol=0.0):
        raise ValueError("move-tracked trajectory resolved points do not match endpoint anchors")
    relations = target.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("move-tracked trajectory relations must be an array")
    for relation_index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"move-tracked trajectory relation {relation_index} is invalid")
        relation_names = relation.get("point_names")
        if not isinstance(relation_names, list) or any(
            str(name) not in (role_tracking_names if relation.get("type") == "ordered_collinear" else point_names)
            for name in relation_names
        ):
            raise ValueError(
                f"move-tracked trajectory relation {relation_index} references an unbound point"
            )
    requested_points = target.get("requested_points")
    if requested_points is None:
        requested_points = (trajectory.get("parameters") or {}).get("points")
    if requested_points is None:
        raise ValueError("move-tracked trajectory requested point expressions are missing")
    if requested_quick is not None and not isinstance(requested_quick, dict):
        raise ValueError("move-tracked quick constraint metadata is invalid")
    if requested_quick_groups is not None and not isinstance(
        requested_quick_groups, list
    ):
        raise ValueError("move-tracked quick constraint groups metadata is invalid")
    if target.get("on_hand_point_names") != point_names:
        raise ValueError("move-tracked on-hand names do not match controlled binding")
    if target.get("off_hand_point_names") != role_reference_names:
        raise ValueError("move-tracked off-hand names do not match reference binding")
    offhand_raw = target.get("off_hand_source_points_robot_base_m")
    offhand_matrix: np.ndarray | None = None
    if role_reference_names:
        try:
            offhand_matrix = np.asarray(offhand_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("move-tracked off-hand source points are invalid") from exc
        if offhand_matrix.shape != (len(role_reference_names), 3) or not np.all(
            np.isfinite(offhand_matrix)
        ):
            raise ValueError("move-tracked off-hand source points are invalid")
        offhand_matrix = offhand_matrix.copy()
    elif offhand_raw is not None:
        raise ValueError("move-tracked off-hand source points exist without references")
    requested_relations = target.get("requested_relations", [])
    if not isinstance(requested_relations, list):
        raise ValueError("move-tracked requested relations must be an array")
    requested_inequalities = target.get("requested_inequalities", [])
    if not isinstance(requested_inequalities, list):
        raise ValueError("move-tracked requested inequalities must be an array")
    request_payload: dict[str, Any] = {
        "points": requested_points,
        "relations": requested_relations,
        "inequalities": requested_inequalities,
    }
    if quick_mode:
        request_payload["mode"] = "quick_constraint"
        if requested_quick_groups is not None:
            request_payload["quick_constraints"] = requested_quick_groups
        else:
            request_payload["quick_constraint"] = requested_quick
    try:
        normalized_request = validate_move_tracked_point_args(request_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"move-tracked trajectory point expressions are invalid: {exc}"
        ) from exc
    if _json_ready(normalized_request["points"]) != _json_ready(requested_points):
        raise ValueError("move-tracked trajectory point expressions are not canonical")
    requested_names = [
        str(point.get("name") or "").strip()
        for point in requested_points
        if isinstance(point, dict)
    ]
    if requested_names != role_tracking_names:
        raise ValueError(
            "move-tracked requested point names do not match complete tracking binding"
        )
    normalized_roles = {
        str(point["name"]): str(point.get("role") or "on_hand")
        for point in normalized_request["points"]
    }
    expected_controlled_names = [
        name for name in requested_names if normalized_roles[name] == "on_hand"
    ]
    expected_reference_names = [
        name for name in requested_names if normalized_roles[name] == "off_hand"
    ]
    if expected_controlled_names != point_names:
        raise ValueError("move-tracked requested on-hand roles do not match binding")
    if expected_reference_names != role_reference_names:
        raise ValueError("move-tracked requested off-hand roles do not match binding")
    if quick_mode:
        normalized_specs = _move_tracked_quick_constraint_specs(
            quick_constraint=normalized_request.get("quick_constraint"),
            quick_constraints=normalized_request.get("quick_constraints"),
        )
        if _json_ready(normalized_specs) != _json_ready(quick_specs):
            raise ValueError("move-tracked quick constraint groups are not canonical")
    expected_planning_targets = _move_tracked_planning_targets(
        normalized_request["points"], point_names
    )
    planning_target_points = target.get("planning_target_points")
    if _json_ready(planning_target_points) != _json_ready(expected_planning_targets):
        raise ValueError("move-tracked planning target coordinates are inconsistent")
    expected_reference_targets = _move_tracked_planning_targets(
        normalized_request["points"], role_reference_names
    )
    if _json_ready(target.get("off_hand_target_points", [])) != _json_ready(
        expected_reference_targets
    ):
        raise ValueError("move-tracked off-hand target coordinates are inconsistent")
    if _json_ready(normalized_request["relations"]) != _json_ready(requested_relations):
        raise ValueError("move-tracked requested relations are not canonical")
    if _json_ready(normalized_request["inequalities"]) != _json_ready(
        requested_inequalities
    ):
        raise ValueError("move-tracked requested inequalities are not canonical")
    planning_inequalities = target.get("inequalities", [])
    if _json_ready(planning_inequalities) != _json_ready(
        normalized_request["inequalities"]
    ):
        raise ValueError("move-tracked planning inequalities are inconsistent")
    expected_planning_relations = list(normalized_request["relations"])
    if quick_mode:
        # Re-derive the preset relation from the signed frozen references.  The
        # resulting relation is solved jointly with every supplied XYZ/typed
        # relation and cannot be swapped independently in a signed plan.
        for quick_spec in quick_specs:
            (
                group_on_names,
                _group_on_points,
                _group_off_names,
                group_off_points,
            ) = _move_tracked_quick_group_points(
                quick_spec,
                controlled_names=point_names,
                controlled_points=source_matrix,
                reference_names=role_reference_names,
                reference_points=offhand_matrix,
            )
            expected_planning_relations.append(
                _quick_constraint_relation(
                    quick_spec,
                    on_hand_names=group_on_names,
                    on_hand_points_robot_base_m=_group_on_points,
                    off_hand_points_robot_base_m=group_off_points,
                    camera_position_robot_base_m=faceto_camera_position,
                    camera_quaternion_xyzw=faceto_camera_quaternion,
                )
            )
    if _json_ready(relations) != _json_ready(expected_planning_relations):
        raise ValueError("move-tracked planning relations are inconsistent")
    endpoint_constraint_report = _tracked_target_constraints(
        resolved_matrix,
        expected_planning_targets,
        tolerance_m=float((trajectory.get("parameters") or {}).get("pos_tol", 0.03)),
        relations=expected_planning_relations,
        inequalities=normalized_request["inequalities"],
        orientation_tolerance_deg=float(
            (trajectory.get("parameters") or {}).get("ori_tol_deg", 20.0)
        ),
        fixed_points_robot_base_m=offhand_matrix,
        fixed_target_points=(
            expected_reference_targets if role_reference_names else None
        ),
    )
    if not endpoint_constraint_report.get("inequality_constraints_ok", True):
        raise ValueError(
            "move-tracked signed endpoint violates its strict inequality constraints"
        )
    _validate_move_tracked_endpoint_order(target)
    geometry_report = target.get("geometry_report")
    if not isinstance(geometry_report, dict) or geometry_report.get("point_count") != point_count:
        raise ValueError("move-tracked trajectory N-point geometry report is missing")
    # Recompute the invariant report from the signed arrays so a stale report
    # cannot be used to bypass the N-point execution checks.
    recomputed_geometry = _tracked_rigid_geometry_report(
        source_matrix,
        resolved_matrix,
        tolerance_m=float(tracking.get("rigid_pair_distance_tolerance_m", 0.012)),
    )
    if int(recomputed_geometry.get("point_count", -1)) != point_count:
        raise ValueError("move-tracked trajectory geometry cardinality mismatch")
    raw_geometry_rank = target.get("geometry_rank")
    if (
        isinstance(raw_geometry_rank, bool)
        or not isinstance(raw_geometry_rank, int)
        or int(raw_geometry_rank) != int(recomputed_geometry.get("source_rank", -1))
    ):
        raise ValueError(
            "move-tracked trajectory geometry rank does not match the signed point bundle"
        )
    geometry_mismatch = _geometry_report_difference(
        geometry_report,
        recomputed_geometry,
        atol=1.0e-9,
    )
    if geometry_mismatch:
        raise ValueError(f"move-tracked trajectory {geometry_mismatch}")
    try:
        endpoint_q = _require_serialized_j8_zero(
            target["endpoint_q"], label="move-tracked trajectory endpoint q"
        )
        endpoint_q = np.asarray(endpoint_q, dtype=np.float64).reshape(ARM_DOF)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("move-tracked trajectory endpoint q is invalid") from exc
    if not np.all(np.isfinite(endpoint_q)):
        raise ValueError("move-tracked trajectory endpoint q contains non-finite values")
    target_trunk_assisted = strict_bool(
        target.get("trunk_assisted", False),
        label="move-tracked target trunk-assisted flag",
    )
    tracking_trunk_assisted = strict_bool(
        tracking.get("trunk_assisted", False),
        label="move-tracked tracking trunk-assisted flag",
    )
    if tracking_trunk_assisted != target_trunk_assisted:
        raise ValueError("move-tracked trunk-assisted flags are inconsistent")
    if target_trunk_assisted:
        raise ValueError(
            "move-tracked plans are arm-only; re-plan this legacy whole-body target"
        )
    if (
        trajectory.get("trunk_waypoints") is not None
        or target.get("endpoint_trunk_q") is not None
    ):
        raise ValueError("move-tracked arm-only trajectory contains trunk motion")
    if planner.get("joint_limits_checked") is not True:
        raise ValueError("move-tracked trajectory joint-limit validation is missing")
    if planner.get("local_fk_finite_checked") is not True:
        raise ValueError("move-tracked trajectory local-FK validation is missing")
    if planner.get("monotonic_progress_checked") is not True:
        raise ValueError("move-tracked trajectory monotonic-progress validation is missing")
    if planner.get("internal_stop_count") != 0:
        raise ValueError("move-tracked trajectory contains an internal timing stop")
    if tracking.get("execution_mode") != "continuous_waypoint_stream":
        raise ValueError("move-tracked trajectory execution mode is invalid")
    try:
        pair_distance_tolerance = float(
            tracking["rigid_pair_distance_tolerance_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "move-tracked trajectory pair-distance tolerance is invalid"
        ) from exc
    if (
        not math.isfinite(pair_distance_tolerance)
        or not 0.012 <= pair_distance_tolerance <= 0.020
    ):
        raise ValueError(
            "move-tracked trajectory pair-distance tolerance is out of bounds"
        )
    if planner.get("simulator_collision_checked") is not False:
        raise ValueError("move-tracked trajectory contains forbidden collision truth")
    for side in ("left", "right"):
        _require_serialized_j8_zero(
            (start_state.get("arm_qpos") or {}).get(side),
            label=f"move-tracked start {side} qpos",
        )
    waypoints = trajectory.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("move-tracked trajectory has no waypoints")
    if len(waypoints) > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("move-tracked trajectory contains too many waypoints")
    limits = _ARM_LIMITS[arm][:ARM_DOF]
    observed_start = _require_serialized_j8_zero(
        start_state["arm_qpos"][arm], label="move-tracked trajectory start qpos"
    )
    previous, start_limit_recovery = _normalize_arm_q_to_joint_limits(
        observed_start,
        arm=arm,
        label="move-tracked trajectory observed start qpos",
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
    if (
        max_start_limit_recovery
        > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
        + TRAJECTORY_STEP_VALIDATION_EPS_RAD
    ):
        raise ValueError(
            "move-tracked trajectory start exceeds the recoverable "
            "joint-limit residual"
        )
    _start_trunk, trunk_start_limit_recovery = (
        _normalize_trunk_q_to_joint_limits(
            start_state.get("trunk_qpos"),
            label="move-tracked trajectory observed start trunk qpos",
            point="observed_start_trunk",
            recover_observed_overshoot=True,
        )
    )
    max_trunk_start_limit_recovery = max(
        (
            abs(float(item["correction_rad"]))
            for item in trunk_start_limit_recovery
        ),
        default=0.0,
    )
    if (
        max_trunk_start_limit_recovery
        > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
        + TRAJECTORY_STEP_VALIDATION_EPS_RAD
    ):
        raise ValueError(
            "move-tracked trajectory start trunk exceeds the recoverable "
            "joint-limit residual"
        )
    for raw_q in waypoints:
        q = _require_serialized_j8_zero(
            raw_q, label="move-tracked trajectory waypoint"
        )
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("move-tracked trajectory waypoint exceeds joint limits")
        if (
            float(np.linalg.norm(q - previous, ord=np.inf))
            > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
            + TRAJECTORY_STEP_VALIDATION_EPS_RAD
        ):
            raise ValueError("move-tracked trajectory waypoint step is too large")
        previous = q
    return trajectory, path



def _move_tracked_plan_safe_q(
    *,
    state: LocalRobotState,
    arm: str,
    final_q: Sequence[float],
    requested_back_m: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Solve the same fixed-trunk pre-pose used by grasp-plan execution."""

    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("move-tracked planned safe arm is invalid")
    endpoint_q = _require_serialized_j8_zero(
        final_q,
        label=f"move-tracked {side} planned final q",
    )
    final_position, final_quaternion = _local_eef_pose(
        state,
        side,
        endpoint_q,
    )
    approach = _quat_to_mat_xyzw(final_quaternion)[:, 2]
    approach_norm = float(np.linalg.norm(approach))
    if not math.isfinite(approach_norm) or approach_norm <= 1.0e-12:
        raise ValueError("move-tracked final EEF approach axis is invalid")
    approach = approach / approach_norm

    attempts: list[dict[str, Any]] = []
    # Match the grasp/exec retreat convention, while bounding local work. A
    # nearby final-q warm start normally solves the requested 10 cm candidate;
    # progressively shorter retreats preserve useful pre-pose behavior near a
    # reach boundary without changing the final endpoint.
    candidates = _exec_safe_back_m_candidates(float(requested_back_m))[:12]
    for candidate_index, back_m in enumerate(candidates):
        target_position = final_position - float(back_m) * approach
        try:
            safe_q, ik_report = solve_pose_target(
                state,
                side,
                endpoint_q,
                target_position=target_position,
                target_quaternion=final_quaternion,
                joint_indices=range(7),
                pos_tol_m=EXEC_SAFE_POS_TOL_M,
                ori_tol_deg=EXEC_SAFE_ORI_TOL_DEG,
                max_iterations=160,
            )
            safe_q, normalizations = _normalize_arm_q_to_joint_limits(
                safe_q,
                arm=side,
                label=f"move-tracked planned safe candidate {candidate_index}",
                point="safe_q",
            )
            actual_position, actual_quaternion = _local_eef_pose(
                state,
                side,
                safe_q,
            )
            position_error = float(
                np.linalg.norm(actual_position - target_position)
            )
            orientation_error = float(
                orientation_error_deg(actual_quaternion, final_quaternion)
            )
            branch_gap = float(
                np.linalg.norm(safe_q - endpoint_q, ord=np.inf)
            )
            accepted = bool(
                ik_report.get("ok")
                and position_error <= EXEC_SAFE_POS_TOL_M
                and orientation_error <= EXEC_SAFE_ORI_TOL_DEG
                and branch_gap <= EXEC_SAFE_FINAL_BRANCH_GAP_RAD
            )
            attempt = {
                "candidate_index": int(candidate_index),
                "back_m": float(back_m),
                "ok": accepted,
                "pos_err_m": position_error,
                "ori_err_deg": orientation_error,
                "safe_to_final_joint_gap_rad": branch_gap,
                "joint_limit_normalizations": normalizations,
                "ik": _json_ready(ik_report),
            }
            attempts.append(attempt)
            if accepted:
                return safe_q, float(back_m), {
                    "ok": True,
                    "requested_back_m": float(requested_back_m),
                    "active_back_m": float(back_m),
                    "back_m_adapted": abs(
                        float(back_m) - float(requested_back_m)
                    )
                    > 1.0e-4,
                    "solver": "submission_local_static_urdf",
                    "safe_q": safe_q.astype(float).tolist(),
                    "target_eef_position_m": target_position.astype(float).tolist(),
                    "target_eef_quaternion_xyzw": np.asarray(
                        final_quaternion, dtype=np.float64
                    ).astype(float).tolist(),
                    "safe_to_final_interpolation": "linear_joint_same_as_grasp_plan",
                    "attempts": attempts,
                    "j8_locked": LOCKED_J8_INDEX is not None,
                    "trunk_motion": False,
                }
        except (RuntimeError, TypeError, ValueError) as exc:
            attempts.append(
                {
                    "candidate_index": int(candidate_index),
                    "back_m": float(back_m),
                    "ok": False,
                    "error": str(exc),
                }
            )
    attempted = ", ".join(f"{value:.2f}" for value in candidates)
    raise ValueError(
        "move-tracked endpoint has no fixed-trunk arm safe pose; "
        f"tried back_m=[{attempted}]"
    )



def _compile_move_tracked_exec_plan_trajectory(
    move_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a validated move plan into the immutable exec_plan_pose schema."""

    source = dict(move_trajectory)
    active_arm = str(source.get("active_arm") or "").strip().lower()
    if active_arm not in ("left", "right"):
        raise ValueError("move-tracked execution plan has an invalid active arm")
    start_state_source = source.get("start_state")
    target = source.get("target")
    if not isinstance(start_state_source, Mapping) or not isinstance(
        target, Mapping
    ):
        raise ValueError("move-tracked execution plan metadata is incomplete")
    _validate_move_tracked_endpoint_order(target)
    source_tracking = source.get("tracking")
    if (
        bool(target.get("trunk_assisted", False))
        or (
            isinstance(source_tracking, Mapping)
            and bool(source_tracking.get("trunk_assisted", False))
        )
        or source.get("trunk_waypoints") is not None
        or target.get("endpoint_trunk_q") is not None
    ):
        raise ValueError(
            "move-tracked plan contains legacy whole-body motion; re-plan with "
            "the arm-only planner before exec_plan_pose"
        )

    source_parameters = source.get("parameters")
    source_parameters = (
        source_parameters if isinstance(source_parameters, Mapping) else {}
    )
    plan_only_source = str(
        source_parameters.get("execution_mode") or ""
    ).strip().lower() == "plan"
    planner_source = source.get("planner")
    planner_source = planner_source if isinstance(planner_source, Mapping) else {}
    overlap_filter = planner_source.get("frozen_rgbd_overlap_filter")
    selected_overlap = target.get("overlap_volume_filter")
    frozen_rgbd_collision_checked = False
    frozen_rgbd_collision_evaluated = False
    overlap_fallback_applied = False
    if plan_only_source:
        if not isinstance(overlap_filter, Mapping):
            raise ValueError(
                "move-tracked plan endpoint has no successful frozen RGB-D "
                "Lite overlap filter"
            )
        overlap_fallback_applied = bool(overlap_filter.get("fallback_applied"))
        if not bool(overlap_filter.get("ok")) and not overlap_fallback_applied:
            raise ValueError(
                "move-tracked plan endpoint has no successful frozen RGB-D "
                "Lite overlap filter"
            )
        if not isinstance(selected_overlap, Mapping) or (
            not bool(selected_overlap.get("ok")) and not overlap_fallback_applied
        ):
            raise ValueError(
                "move-tracked selected endpoint did not pass the frozen RGB-D "
                "Lite overlap filter"
            )
        try:
            original_overlap = float(selected_overlap["overlap_vol_cm3"])
            raw_inflated_overlap = selected_overlap["inflated_overlap_vol_cm3"]
            inflated_overlap = (
                None if raw_inflated_overlap is None else float(raw_inflated_overlap)
            )
            original_threshold = float(
                overlap_filter["original_overlap_threshold_cm3"]
            )
            inflated_threshold = float(
                overlap_filter["inflated_overlap_threshold_cm3"]
            )
            overlap_position = np.asarray(
                selected_overlap["eef_position_robot_base_m"],
                dtype=np.float64,
            ).reshape(3)
            overlap_quaternion = np.asarray(
                selected_overlap["eef_quaternion_xyzw"],
                dtype=np.float64,
            ).reshape(4)
            target_position = np.asarray(
                target["eef_position_robot_base_m"], dtype=np.float64
            ).reshape(3)
            target_quaternion = np.asarray(
                target["eef_quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "move-tracked selected overlap report is incomplete"
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (
                original_overlap,
                original_threshold,
                inflated_threshold,
            )
        ) or original_overlap < 0.0 or (
            inflated_overlap is not None
            and (not math.isfinite(inflated_overlap) or inflated_overlap < 0.0)
        ):
            raise ValueError("move-tracked selected overlap report is non-finite")
        overlap_quaternion_norm = float(np.linalg.norm(overlap_quaternion))
        target_quaternion_norm = float(np.linalg.norm(target_quaternion))
        if (
            not np.all(np.isfinite(overlap_position))
            or not np.all(np.isfinite(overlap_quaternion))
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(target_quaternion))
            or overlap_quaternion_norm <= 1.0e-12
            or target_quaternion_norm <= 1.0e-12
            or not np.allclose(
                overlap_position,
                target_position,
                rtol=0.0,
                atol=1.0e-9,
            )
            or abs(
                float(
                    np.dot(
                        overlap_quaternion / overlap_quaternion_norm,
                        target_quaternion / target_quaternion_norm,
                    )
                )
            )
            < 1.0 - 1.0e-10
        ):
            raise ValueError(
                "move-tracked selected overlap report belongs to a different "
                "EEF endpoint"
            )
        if (
            not math.isclose(
                original_threshold,
                ORIGINAL_OVERLAP_MAX_CM3,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                inflated_threshold,
                INFLATED_OVERLAP_MAX_CM3,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or overlap_filter.get("original_overlap_comparison") != "<="
            or overlap_filter.get("inflated_overlap_comparison") != "<"
            or overlap_filter.get("build") != LITE_POSE_OVERLAP_FILTER_BUILD
            or overlap_filter.get("source_planner")
            != "plan_grasp_point_filter_rgbd_lite"
            or not bool(
                overlap_filter.get("filter_result_equivalent_to_full_conjunction")
            )
            or bool(overlap_filter.get("grasp_opening_volume_queried"))
        ):
            raise ValueError(
                "move-tracked selected endpoint violates the frozen RGB-D "
                "Lite overlap thresholds"
            )
        original_ok = original_overlap <= ORIGINAL_OVERLAP_MAX_CM3 + 1.0e-12
        inflated_ok = (
            inflated_overlap is not None and inflated_overlap < INFLATED_OVERLAP_MAX_CM3
        )
        thresholds_ok = original_ok and inflated_ok
        if (
            bool(selected_overlap.get("original_overlap_ok")) != original_ok
            or bool(selected_overlap.get("inflated_overlap_ok")) != inflated_ok
            or bool(selected_overlap.get("ok")) != thresholds_ok
            or bool(overlap_filter.get("ok")) != thresholds_ok
            or bool(selected_overlap.get("inflated_query_skipped")) == original_ok
            or (original_ok and inflated_overlap is None)
            or (not original_ok and inflated_overlap is not None)
        ):
            raise ValueError("move-tracked selected overlap threshold evidence is inconsistent")
        if overlap_fallback_applied:
            selection = selected_overlap.get("fallback_selection")
            accuracy = planner_source.get("endpoint_selected_candidate")
            if not isinstance(selection, Mapping) or not isinstance(accuracy, Mapping):
                raise ValueError("move-tracked overlap fallback has no accuracy evidence")
            try:
                cost = float(selection["accuracy_cost_mm_plus_deg"])
                endpoint_cost = float(accuracy["accuracy_cost_mm_plus_deg"])
                frontier_count = int(selection["accuracy_frontier_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("move-tracked overlap fallback accuracy evidence is invalid") from exc
            # ``endpoint_candidate_count_passing`` is cumulative over every
            # endpoint frontier evaluated by the frozen planner.  A passing
            # endpoint can therefore be non-zero even when the final endpoint
            # comes from the overlap fallback: the passing endpoint may have
            # been rejected later because its Cartesian/joint path was not
            # executable.  The old check required this count to be exactly
            # zero, which incorrectly rejected that legitimate fallback path
            # (and was the source of the opaque "evidence is inconsistent"
            # error).  Validate the actual count invariants instead of
            # conflating "overlap-passing" with "eventually selected".
            endpoint_passing_count = overlap_filter.get(
                "endpoint_candidate_count_passing"
            )
            evaluated_count = overlap_filter.get("candidate_count_evaluated")
            passing_count = overlap_filter.get("candidate_count_passing")
            counts_valid = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    endpoint_passing_count,
                    evaluated_count,
                    passing_count,
                )
            )
            if counts_valid:
                counts_valid = bool(
                    endpoint_passing_count <= passing_count <= evaluated_count
                )
            if (
                thresholds_ok
                or selection.get("policy") != MOVE_TRACKED_OVERLAP_FALLBACK_POLICY
                or selection.get("overlap_order") != "original_then_inflated_cm3"
                or selection.get("accuracy_class") != (
                    "precise" if bool(accuracy.get("precise")) else "minimum_error"
                )
                or not math.isfinite(cost)
                or cost < 0.0
                or not math.isfinite(endpoint_cost)
                or not math.isclose(cost, endpoint_cost, rel_tol=0.0, abs_tol=1.0e-9)
                or frontier_count < 1
                or not counts_valid
                or _json_ready(overlap_filter.get("fallback_selection"))
                != _json_ready(selection)
                or _json_ready(overlap_filter.get("selected_pose"))
                != _json_ready(selected_overlap)
            ):
                raise ValueError("move-tracked overlap fallback evidence is inconsistent")
        elif not thresholds_ok or selected_overlap.get("fallback_selection"):
            raise ValueError(
                "move-tracked selected endpoint violates the frozen RGB-D "
                "Lite overlap thresholds"
            )
        frozen_rgbd_collision_evaluated = True
        frozen_rgbd_collision_checked = bool(thresholds_ok)

    arm_qpos = dict(start_state_source.get("arm_qpos") or {})
    arm_qvel = dict(start_state_source.get("arm_qvel") or {})
    gripper_qpos = dict(start_state_source.get("gripper_qpos") or {})
    start_arms = {
        side: _require_serialized_j8_zero(
            arm_qpos.get(side),
            label=f"move-tracked exec start {side} qpos",
        )
        for side in ("left", "right")
    }
    normalized_start_q, start_normalizations = (
        _normalize_arm_q_to_joint_limits(
            start_arms[active_arm],
            arm=active_arm,
            label=f"move-tracked exec {active_arm} start qpos",
            point="start_q",
            recover_observed_overshoot=True,
        )
    )
    maximum_start_correction = max(
        (
            abs(float(item["correction_rad"]))
            for item in start_normalizations
        ),
        default=0.0,
    )
    if (
        maximum_start_correction
        > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
        + TRAJECTORY_STEP_VALIDATION_EPS_RAD
    ):
        raise ValueError(
            "move-tracked exec start arm exceeds the recoverable "
            "joint-limit residual"
        )
    start_arms[active_arm] = normalized_start_q
    final_q, final_normalizations = _normalize_arm_q_to_joint_limits(
        _require_serialized_j8_zero(
            target.get("endpoint_q"),
            label="move-tracked exec final q",
        ),
        arm=active_arm,
        label=f"move-tracked exec {active_arm} final q",
        point="final_q",
    )
    start_trunk, trunk_start_normalizations = (
        _normalize_trunk_q_to_joint_limits(
            start_state_source.get("trunk_qpos"),
            label="move-tracked exec observed start trunk q",
            point="start_trunk",
            recover_observed_overshoot=True,
        )
    )
    maximum_trunk_start_correction = max(
        (
            abs(float(item["correction_rad"]))
            for item in trunk_start_normalizations
        ),
        default=0.0,
    )
    if (
        maximum_trunk_start_correction
        > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD
        + TRAJECTORY_STEP_VALIDATION_EPS_RAD
    ):
        raise ValueError(
            "move-tracked exec start trunk exceeds the recoverable "
            "joint-limit residual"
        )
    if source.get("trunk_waypoints") is not None or target.get("endpoint_trunk_q") is not None:
        raise ValueError("move-tracked arm-only execution plan contains trunk motion")

    raw_waypoints = source.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("move-tracked execution plan has no signed waypoints")
    limits = _ARM_LIMITS[active_arm][:ARM_DOF]
    previous_q = normalized_start_q
    waypoints: list[list[float]] = []
    local_state = dataclass_replace(
        _plan_start_local_state(dict(start_state_source)),
        trunk_q=start_trunk.copy(),
    )
    for waypoint_index, raw_q in enumerate(raw_waypoints):
        q = _require_serialized_j8_zero(
            raw_q,
            label=f"move-tracked exec waypoint {waypoint_index + 1}",
        )
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError(
                "move-tracked exec waypoint exceeds local arm joint limits"
            )
        step = float(np.linalg.norm(q - previous_q, ord=np.inf))
        if step > DEFAULT_TRAJECTORY_STEP_RAD + TRAJECTORY_STEP_VALIDATION_EPS_RAD:
            raise ValueError(
                "move-tracked exec waypoint exceeds exec_plan_pose step limit"
            )
        position, quaternion = _local_eef_pose(local_state, active_arm, q)
        if not np.all(np.isfinite(position)) or not np.all(
            np.isfinite(quaternion)
        ):
            raise ValueError("move-tracked exec waypoint local FK is non-finite")
        waypoints.append(q.astype(float).tolist())
        previous_q = q
    if not np.allclose(
        previous_q,
        final_q,
        rtol=0.0,
        atol=LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
    ):
        raise ValueError(
            "move-tracked signed path does not end at its planned final q"
        )

    planned_position = np.asarray(
        target.get("eef_position_robot_base_m"), dtype=np.float64
    ).reshape(3)
    planned_quaternion = np.asarray(
        target.get("eef_quaternion_xyzw"), dtype=np.float64
    ).reshape(4)
    final_position, final_quaternion = _local_eef_pose(
        local_state, active_arm, final_q
    )
    final_position_error = float(
        np.linalg.norm(np.asarray(final_position) - planned_position)
    )
    final_orientation_error = float(
        orientation_error_deg(final_quaternion, planned_quaternion)
    )
    if (
        final_position_error > EXEC_FINAL_PLAN_POS_TOL_M
        or final_orientation_error > EXEC_FINAL_PLAN_ORI_TOL_DEG
    ):
        raise ValueError(
            "move-tracked final q does not reproduce its planned EEF pose"
        )

    start_state = {
        "motion_epoch": int(start_state_source["motion_epoch"]),
        "episode_id": str(start_state_source["episode_id"]),
        "arm_qpos": {
            side: start_arms[side].astype(float).tolist()
            for side in ("left", "right")
        },
        "arm_qvel": {
            side: _arm_vector(
                arm_qvel.get(side),
                label=f"move-tracked exec start {side} qvel",
            ).astype(float).tolist()
            for side in ("left", "right")
        },
        "trunk_qpos": start_trunk.astype(float).tolist(),
        "gripper_qpos": {
            side: np.asarray(
                gripper_qpos.get(side), dtype=np.float64
            ).reshape(-1).astype(float).tolist()
            for side in ("left", "right")
        },
    }
    for value in (
        np.asarray(start_state["trunk_qpos"], dtype=np.float64),
        *(
            np.asarray(start_state["gripper_qpos"][side], dtype=np.float64)
            for side in ("left", "right")
        ),
    ):
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("move-tracked exec start state is non-finite")

    safe_q, active_back_m, planned_safe = _move_tracked_plan_safe_q(
        state=local_state,
        arm=active_arm,
        final_q=final_q,
        requested_back_m=EXEC_SAFE_BACK_DEFAULT_M,
    )
    segments = _compile_exec_arm_segments(normalized_start_q, safe_q, final_q)
    waypoint_count = sum(len(segment["waypoints"]) for segment in segments)
    if waypoint_count > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("move-tracked exec trajectory contains too many waypoints")

    trajectory: dict[str, Any] = {
        "schema": JOINT_TRAJECTORY_SCHEMA,
        "schema_version": JOINT_TRAJECTORY_SCHEMA_VERSION,
        "plan_id": str(source["plan_id"]),
        "session_id": str(source["session_id"]),
        "tool_version": TOOL_VERSION,
        "source_tool": "move_tracked_point",
        "source_image_id": str(source["source_image_id"]),
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": int(ARM_DOF),
        "action_dim": int(ACTION_DIM),
        "active_arm": active_arm,
        "final_q_by_arm": {active_arm: final_q.astype(float).tolist()},
        "locked_arm_joints": (
            {"J8": LOCKED_J8_VALUE_RAD}
            if LOCKED_J8_INDEX is not None
            else {}
        ),
        "start_state": start_state,
        "parameters": {
            "back_m": float(active_back_m),
            "reset_tool_roll_at_start": True,
        },
        "tracking": {
            "start_arm_tolerance_rad": DEFAULT_START_ARM_TOLERANCE_RAD,
            "start_trunk_tolerance_rad": DEFAULT_START_TRUNK_TOLERANCE_RAD,
            "start_gripper_tolerance_rad": DEFAULT_START_GRIPPER_TOLERANCE_RAD,
            "stationary_qvel_tolerance_rad_s": 0.35,
            "settle_tolerance_rad": DEFAULT_WAYPOINT_SETTLE_TOLERANCE_RAD,
            "max_tracking_error_rad": DEFAULT_WAYPOINT_MAX_TRACKING_ERROR_RAD,
            "waypoint_timeout_s": DEFAULT_WAYPOINT_TIMEOUT_S,
            "stall_steps": DEFAULT_TRACKING_STALL_STEPS,
        },
        "segments": segments,
        "safety_validation": {
            "status": "conditional",
            "joint_limits_checked": True,
            "joint_limit_numeric_tolerance_rad": (
                TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
            ),
            "joint_limit_normalizations": _json_ready(
                [
                    *start_normalizations,
                    *final_normalizations,
                    *trunk_start_normalizations,
                ]
            ),
            "captured_start_joint_limits_enforced": False,
            "captured_start_recovered_to_local_limits": bool(
                start_normalizations or trunk_start_normalizations
            ),
            "local_fk_finite_checked": True,
            "safe_final_pair_validated_by_planner": True,
            "frozen_rgbd_collision_checked": bool(
                frozen_rgbd_collision_checked
            ),
            "frozen_rgbd_collision_evaluated": frozen_rgbd_collision_evaluated,
            "overlap_fallback_applied": overlap_fallback_applied,
            "frozen_rgbd_overlap_filter": (
                _json_ready(overlap_filter)
                if frozen_rgbd_collision_evaluated
                else None
            ),
            "self_collision_checked": False,
            "runtime_may_recompute_kinematics": False,
            "source_move_trajectory_digest": str(
                (source.get("integrity") or {}).get("digest") or ""
            ),
            "planned_safe": _json_ready(planned_safe),
        },
        "final_q_validation_by_arm": {
            active_arm: {
                "source": "signed_move_tracked_trajectory_endpoint",
                "planned_arm": active_arm,
                "execution_arm": active_arm,
                "arm_switched": False,
                "pair_pos_err_m": final_position_error,
                "pair_ori_err_deg": final_orientation_error,
                "submission_local_fk_checked": True,
                "j8_locked": LOCKED_J8_INDEX is not None,
            }
        },
    }
    trajectory["integrity"] = {
        "algorithm": "hmac-sha256",
        "digest": _trajectory_digest(trajectory),
    }
    return trajectory



def _move_tracked_plan_preview_result(
    *,
    move_trajectory: dict[str, Any],
    plan_path: str,
) -> dict[str, Any]:
    """Persist and expose the exact planned endpoint without moving the arm."""

    session_id = str(move_trajectory["session_id"])
    frozen_scene = move_trajectory.get("planner", {}).get("frozen_plan_capture") or {}
    image_id = str(frozen_scene.get("image_id") or move_trajectory["source_image_id"])
    plan_id = str(move_trajectory["plan_id"])
    arm = str(move_trajectory["active_arm"])
    target = dict(move_trajectory["target"])
    capture = _load_frozen_capture(session_id, image_id, role="head")
    if frozen_scene and (
        capture.evaluator_sequence != frozen_scene["observation_sequence"]
        or capture.evaluator_sequence != move_trajectory["tracking_binding"]["observation_sequence"]
    ):
        raise ValueError("plan preview RGB-D does not match the frozen planning observation")
    rgb_path = _image_path(session_id, image_id, ".png")
    overlay_path = _image_path(
        session_id,
        image_id,
        f".{plan_id}.gripper.png",
    )
    eef_pose = {
        "pos": _json_ready(target["eef_position_robot_base_m"]),
        "quat": _json_ready(target["eef_quaternion_xyzw"]),
    }
    camera_relative_pose = capture.camera.get("robot_relative_pose")
    if not isinstance(camera_relative_pose, Mapping):
        raise ValueError(
            "move-tracked plan preview requires the frozen evaluator "
            "camera-relative pose"
        )
    camera_position = np.asarray(
        camera_relative_pose.get("pos"),
        dtype=np.float64,
    ).reshape(3)
    camera_quaternion = np.asarray(
        camera_relative_pose.get("quat"),
        dtype=np.float64,
    ).reshape(4)
    if not (
        np.isfinite(camera_position).all()
        and np.isfinite(camera_quaternion).all()
        and float(np.linalg.norm(camera_quaternion)) > 1e-8
    ):
        raise ValueError(
            "move-tracked plan preview camera-relative pose is invalid"
        )
    preview_camera = {
        **capture.camera,
        "pos": camera_position.astype(float).tolist(),
        "quat": camera_quaternion.astype(float).tolist(),
        "frame": "robot_base",
        "source": "frozen_evaluator_camera_relative_pose",
    }
    gripper_visualization = render_plan_gripper_overlay(
        rgb_path=rgb_path,
        depth_linear=capture.depth,
        camera=preview_camera,
        eef_pose=eef_pose,
        output_path=overlay_path,
        gripper_qpos=move_trajectory["start_state"]["gripper_qpos"][arm],
    )
    gripper_visualization = {
        **dict(gripper_visualization),
        "camera_frame": "robot_base",
        "eef_pose_frame": "robot_base",
    }
    exec_trajectory = _compile_move_tracked_exec_plan_trajectory(
        move_trajectory
    )
    fallback_applied = bool(
        exec_trajectory["safety_validation"]["overlap_fallback_applied"]
    )
    overlap_selection_status = {
        "overlap_fallback_applied": fallback_applied,
        "overlap_thresholds_satisfied": not fallback_applied,
        "warnings": (
            ["overlap thresholds exceeded; minimum-overlap endpoint selected from "
             "the best-accuracy candidates; collision-free motion is not guaranteed"]
            if fallback_applied else []
        ),
    }
    payload = {
        **overlap_selection_status,
        "recommended_arm": arm,
        "eef_pose": eef_pose,
        "selected_pose_ik_q": {arm: target["endpoint_q"]},
        "target_constraints": target.get("constraints"),
        "resolved_target_points_robot_base_m": target.get(
            "resolved_points_robot_base_m"
        ),
        "quick_constraint": target.get("quick_constraint"),
        "quick_constraints": target.get("quick_constraints"),
        "inequalities": target.get("inequalities", []),
        "predefined_eef_point_names": target.get(
            "predefined_eef_point_names", []
        ),
        "overlap_volume_filter": target.get("overlap_volume_filter"),
        "overlap_vol_cm3": (
            (target.get("overlap_volume_filter") or {}).get(
                "overlap_vol_cm3"
            )
        ),
        "inflated_overlap_vol_cm3": (
            (target.get("overlap_volume_filter") or {}).get(
                "inflated_overlap_vol_cm3"
            )
        ),
        "source_move_trajectory_digest": move_trajectory["integrity"][
            "digest"
        ],
        "gripper_visualization": gripper_visualization,
        "frozen_plan_capture": frozen_scene,
    }
    record = _json_ready(
        {
            "ok": True,
            "plan_id": plan_id,
            "tool": "move_tracked_point",
            "tool_version": TOOL_VERSION,
            "session_id": session_id,
            "image_id": image_id,
            "execution_mode": "plan",
            "plan_arm": arm,
            "robot_model": {
                "profile": ROBOT_PROFILE,
                "arm_dof": int(ARM_DOF),
                "kinematics": "submission_local_urdf_curobo",
            },
            "visualization": gripper_visualization,
            "trajectory": exec_trajectory,
            "move_tracked_trajectory": move_trajectory,
            "payload": payload,
        }
    )
    _write_plan_record(plan_path, record)
    loaded_exec, loaded_path, _loaded_record = _load_plan_trajectory(
        session_id,
        plan_id,
        include_record=True,
    )
    if loaded_path != plan_path or loaded_exec["integrity"]["digest"] != (
        exec_trajectory["integrity"]["digest"]
    ):
        raise ValueError("move-tracked exec plan failed immutable reload validation")

    visualization_ok = bool(gripper_visualization.get("ok"))
    result = {
        **_move_tracked_common_result(),
        **overlap_selection_status,
        "ok": True,
        "error": None,
        "failure_stage": None,
        "build": MOVE_TRACKED_POINT_LOCAL_BUILD,
        "execution_mode": "plan",
        "execution": "plan_only_no_robot_motion",
        "session_id": session_id,
        "image_id": image_id,
        "plan_id": plan_id,
        "plan_record_path": plan_path,
        "trajectory_schema": exec_trajectory["schema"],
        "trajectory_schema_version": exec_trajectory["schema_version"],
        "trajectory_digest": exec_trajectory["integrity"]["digest"],
        "trajectory_waypoint_count": sum(
            len(segment["waypoints"])
            for segment in exec_trajectory["segments"]
        ),
        "move_trajectory_schema": move_trajectory["schema"],
        "move_trajectory_schema_version": move_trajectory["schema_version"],
        "move_trajectory_digest": move_trajectory["integrity"]["digest"],
        "move_trajectory_waypoint_count": len(move_trajectory["waypoints"]),
        "robot_profile": ROBOT_PROFILE,
        "arm": arm,
        "recommended_arm": arm,
        "eef_pose": eef_pose,
        "on_hand_point_names": target.get("on_hand_point_names"),
        "off_hand_point_names": target.get("off_hand_point_names"),
        "predefined_eef_point_names": target.get(
            "predefined_eef_point_names", []
        ),
        "quick_constraint": target.get("quick_constraint"),
        "quick_constraints": target.get("quick_constraints"),
        "overlap_volume_filter": target.get("overlap_volume_filter"),
        "overlap_vol_cm3": (
            (target.get("overlap_volume_filter") or {}).get(
                "overlap_vol_cm3"
            )
        ),
        "inflated_overlap_vol_cm3": (
            (target.get("overlap_volume_filter") or {}).get(
                "inflated_overlap_vol_cm3"
            )
        ),
        "resolved_variables": (target.get("constraints") or {}).get(
            "resolved_variables", {}
        ),
        "target_constraints": target.get("constraints"),
        "per_inequality_errors": (
            (target.get("constraints") or {}).get("inequalities", [])
        ),
        "max_inequality_violation_m": (
            (target.get("constraints") or {}).get(
                "max_inequality_violation_m", 0.0
            )
        ),
        "minimum_inequality_clearance_m": (
            (target.get("constraints") or {}).get(
                "minimum_inequality_clearance_m"
            )
        ),
        "resolved_target_points_robot_base_m": target.get(
            "resolved_points_robot_base_m"
        ),
        "endpoint_solver": move_trajectory["planner"].get(
            "endpoint_solver"
        ),
        "planner": move_trajectory["planner"],
        "frozen_plan_capture": frozen_scene,
        "track_object_distance_binding": frozen_scene.get("track_object_distance_binding"),
        "gripper_visualization": gripper_visualization,
        "render_image": (
            _file_to_data_url(overlay_path) if visualization_ok else None
        ),
        "render_image_path": overlay_path if visualization_ok else None,
        "rgb_overlay_path": overlay_path if visualization_ok else None,
        "marked_image_url": overlay_path if visualization_ok else None,
        "plan_preview_overlay": (
            "red_gripper_on_frozen_head" if visualization_ok else None
        ),
        "observation_contract": [
            "frozen_evaluator_rgb",
            "frozen_evaluator_depth_linear",
            "frozen_evaluator_camera_relative_pose",
            "capture_time_evaluator_proprioception",
            "submission_local_static_robot_model",
            "policy_owned_tracked_point_state",
        ],
        "segmentation_used": False,
        "object_identity_used": False,
        "simulator_mesh_used": False,
        "simulator_fk_ik_used": False,
        "direct_simulator_mutation": False,
    }
    return _json_ready(result)



def _move_tracked_point_impl(
    ctx,
    points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    pos_tol: float = 0.03,
    ori_tol_deg: float = 20.0,
    max_steps: int = 360,
    timeout_s: float = 90.0,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    execution_mode: str = "exec",
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    _eef_prior_lease_state: dict[str, Any] | None = None,
    _motion_retention_lease_state: dict[str, Any] | None = None,
):
    """Move visual or predefined EEF points with coordinates or a preset.

    Each target x/y/z accepts a finite number, a variable-name string, any
    affine expression such as ``z+0.08``, ``2*z``, ``(a+b)/2``, or ``a-b``,
    or ``?`` for an explicitly unconstrained coordinate. Repeated variables
    and expressions impose the declared relations; a variable that occurs
    once leaves that degree of freedom for the nearest reachable solution.
    ``inequalities`` adds strict affine variable half-spaces using ``lhs``,
    ``op`` (``>`` or ``<``), and a finite numeric ``rhs``. For example,
    ``{"lhs": "a-b", "op": ">", "rhs": 0}`` selects the solution with
    ``a`` above ``b``. These constraints are solved with the endpoint and
    verified again from the final live tracked points.
    Every row may be on-hand or off-hand and may omit XYZ entirely. One or more
    named groups may use the presets: touch, flatwise, plane-parallel,
    collinear, line-vertical-to-plane, vertical-to-ground, faceto, and
    reverse-faceto. Faceto requires exactly three ordered on-hand surface
    points whose projected order in the frozen head-camera image is clockwise;
    reverse-faceto uses counterclockwise order and the opposite camera-facing
    normal. All preset residuals and supplied
    XYZ/typed relations are solved jointly; groups may share named points, and
    off-hand points remain frozen references that are never sent to IK as
    movable anchors. ``left_finger_tip``, ``right_finger_tip``, and
    ``gripper_slide_center`` are reserved on-hand names resolved from current
    evaluator proprioception plus submission-local FK; they require no visual
    registration. ``execution_mode=plan`` stores the same
    endpoint and signed path for exec_plan_pose and returns a frozen-head red
    EEF overlay without executing the motion; ``exec`` remains the default.
    """

    request_started = time.monotonic()
    raw_args = {
        "points": points,
        "relations": list(relations or []),
        "inequalities": list(inequalities or []),
        "pos_tol": pos_tol,
        "ori_tol_deg": ori_tol_deg,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
        "execution_mode": execution_mode,
    }
    # ``validate_move_tracked_point_args`` emits the canonical mode marker for
    # role-based presets.  Keep it as an explicit optional keyword so the
    # official server can pass the normalized request through unchanged while
    # preserving positional compatibility for legacy coordinate callers.
    if mode is not None:
        raw_args["mode"] = mode
    if quick_constraint is not None:
        raw_args["quick_constraint"] = quick_constraint
        # Preserve an explicitly supplied selector so the shared contract can
        # reject a concrete preset that conflicts with the nested constraint.
        # Only synthesize the generic marker when the caller omitted ``mode``;
        # this keeps direct generator calls as strict as the HTTP boundary.
        if mode is None:
            raw_args["mode"] = "quick_constraint"
    if quick_constraints is not None:
        raw_args["quick_constraints"] = list(quick_constraints)
        if mode is None:
            raw_args["mode"] = "quick_constraint"
    # Lock J8 before *any* validation branch can emit a hold action.  This
    # includes malformed requests and missing-tracker failures, not only the
    # normal planning path.
    _move_tracked_prepare_j8_lock(ctx)
    try:
        args = validate_move_tracked_point_args(raw_args)
    except (TypeError, ValueError) as exc:
        _move_tracked_failure(
            ctx,
            stage="input validation",
            error=str(exc),
            args=raw_args,
        )
        yield ctx.world.hold_action()
        return
    request_deadline_monotonic = request_started + float(args["timeout_s"])

    real_manager = getattr(ctx.world, "_official_tracked_object_distances", None)
    manager = real_manager

    motion_retention_lease_state = (
        _motion_retention_lease_state
        if _motion_retention_lease_state is not None
        else {}
    )
    motion_retention: dict[str, Any] = {}
    motion_retention_release: dict[str, Any] = {}
    rigid_pair_activation: dict[str, Any] = {}
    rigid_pair_activation_confirmation: dict[str, Any] = {}
    rigid_pair_bootstrap_eef_prior: dict[str, Any] = {
        "ok": True,
        "active": False,
        "reason": "not_required",
    }
    registered_on_hand_source_reconciliation: dict[str, Any] = {
        "ok": True,
        "applicable": False,
        "applied": False,
        "reason": "not_attempted",
    }
    rigid_pair_activation_hold_steps = 0
    activation_hold_state: dict[str, Any] = {}
    prior_rigid_pair_cleanup: dict[str, Any] = {}
    failed_rigid_pair_cleanup: dict[str, Any] = {}
    rigid_pair_activation_provisional = False
    rigid_pair_lease_state = motion_retention_lease_state.setdefault(
        "_rigid_pair_lease_state", {}
    )

    names = [str(point["name"]) for point in args["points"]]
    quick_spec = args.get("quick_constraint")
    quick_specs = _move_tracked_quick_constraint_specs(
        quick_constraint=quick_spec,
        quick_constraints=args.get("quick_constraints"),
    )
    quick_mode = bool(quick_specs)
    roles_by_name = {
        str(point["name"]): str(point.get("role") or "on_hand")
        for point in args["points"]
    }
    controlled_names = [name for name in names if roles_by_name[name] == "on_hand"]
    reference_names = [name for name in names if roles_by_name[name] == "off_hand"]
    predefined_names, visual_names = partition_predefined_eef_point_names(names)
    predefined_name_set = set(predefined_names)
    visual_controlled_names = [
        name for name in controlled_names if name not in predefined_name_set
    ]
    visual_reference_names = [
        name for name in reference_names if name not in predefined_name_set
    ]
    predefined_arm_selection: dict[str, Any] = {
        "selected_arm": None,
        "selection_basis": "not_applicable",
        "score_by_arm": {},
        "predefined_eef_point_names": list(predefined_names),
    }
    preselected_arm: str | None = None
    try:
        _require_episode_initialized(ctx, "move_tracked_point")
        episode_id = str(ctx.world.episode_id())
        if visual_names and real_manager is None:
            raise RuntimeError(
                "official dynamic distance tracker is unavailable for visual point(s): "
                + ", ".join(visual_names)
            )
        manager = _MoveTrackedPointSourceManager(
            ctx,
            real_manager,
            predefined_names=predefined_names,
            episode_id=episode_id,
        )
        motion_retention_lease_state["_manager"] = manager
        motion_retention = manager.begin_motion_retention(
            names,
            episode_id=episode_id,
            timeout_s=min(
                660.0,
                float(args["timeout_s"])
                + MOVE_TRACKED_POINT_SUCCESSOR_RETENTION_EXTRA_S,
            ),
        )
        motion_retention_lease_id = str(
            motion_retention.get("lease_id") or ""
        ).strip()
        if not bool(motion_retention.get("ok")) or not motion_retention_lease_id:
            raise ValueError("tracker request retention lease was not created")
        motion_retention_lease_state.update(
            {
                "lease_id": motion_retention_lease_id,
                "lease": dict(motion_retention),
            }
        )
        adapter, observation_sequence = _adjust_adapter(ctx)
        activation_observations = {
            side: _adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        activation_sequence_after, _received_ts = adapter.observation_metadata()
        if int(activation_sequence_after) != observation_sequence:
            raise ValueError(
                "evaluator observation changed while latching rigid-pair activation hold"
            )
        activation_hold_state = _move_tracked_capture_state(
            ctx,
            activation_observations,
        )
        # Set every limb pin to the just-observed state before the tracker can
        # ask us to exchange recovery actions.  This prevents a previous
        # tool's uncompleted command target from moving the robot while the
        # joint tracker is acquiring its first fresh frames.
        _move_tracked_planning_hold_action(ctx, activation_hold_state)
        prior_rigid_pair_cleanup = _move_tracked_clear_incompatible_rigid_pair(
            manager,
            visual_controlled_names,
        )
        if len(visual_controlled_names) == 2:
            # The real tracker commits joint-pair mode before it validates the
            # old independently tracked entries.  Mark the request lease first
            # so exceptions and generator cancellation also retire that state.
            rigid_pair_lease_state.update(
                {
                    "active": True,
                    "preserve_after_success": False,
                    "names": list(visual_controlled_names),
                    "episode_id": episode_id,
                }
            )
        rigid_pair_activation = _move_tracked_gate_rigid_pair_source_projection(
            _move_tracked_activate_rigid_pair(
                manager,
                visual_controlled_names,
                episode_id=episode_id,
                allow_observation_failure=True,
            )
        )
        activation_reason = str(rigid_pair_activation.get("reason") or "")
        if (
            len(visual_controlled_names) == 2
            and not bool(rigid_pair_activation.get("ok"))
            and activation_reason
            in MOVE_TRACKED_POINT_RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS
        ):
            try:
                rigid_pair_bootstrap_eef_prior = (
                    _move_tracked_bootstrap_on_hand_eef_prior(
                        manager,
                        visual_controlled_names,
                        episode_id=episode_id,
                        current_capture_state=activation_hold_state,
                    )
                )
            except Exception as exc:
                # The fresh joint tracker was already able to recover some
                # legacy cases without an EEF gate.  Keep that fallback rather
                # than turning missing old capture metadata into a regression.
                rigid_pair_bootstrap_eef_prior = {
                    "ok": False,
                    "active": False,
                    "reason": "registration_eef_bootstrap_unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "prediction_values_published_as_observation": False,
                }
            if (
                _eef_prior_lease_state is not None
                and bool(rigid_pair_bootstrap_eef_prior.get("active"))
            ):
                _eef_prior_lease_state.update(
                    rigid_pair_bootstrap_eef_prior
                )
        rigid_pair_activation_provisional = bool(
            len(visual_controlled_names) == 2
            and not rigid_pair_activation.get("ok")
        )
        try:
            rigid_pair_activation_confirmation = (
                yield from _move_tracked_confirm_fresh_rigid_pair_activation(
                    ctx,
                    manager,
                    visual_controlled_names,
                    episode_id=episode_id,
                    adapter=adapter,
                    observation_sequence=observation_sequence,
                    initial_report=rigid_pair_activation,
                    hold_state=activation_hold_state,
                    deadline_monotonic=request_deadline_monotonic,
                )
            )
        finally:
            # ``GeneratorExit`` and cancellation can arrive while the helper is
            # suspended at its observation/action boundary.  Retire only a
            # failed provisional activation; a clean or freshly confirmed pair
            # retains the established success-path semantics.
            if (
                rigid_pair_activation_provisional
                and not rigid_pair_activation_confirmation.get("ok")
                and not failed_rigid_pair_cleanup
            ):
                cleanup_reason = (
                    "rigid_pair_activation_recovery_failed"
                    if rigid_pair_activation_confirmation
                    else "rigid_pair_activation_recovery_interrupted"
                )
                failed_rigid_pair_cleanup = _move_tracked_deactivate_rigid_pair(
                    manager,
                    reason=cleanup_reason,
                    required=False,
                )
                rigid_pair_lease_state["active"] = False
                rigid_pair_lease_state["cleanup"] = _json_ready(
                    failed_rigid_pair_cleanup
                )
        rigid_pair_activation_hold_steps = int(
            rigid_pair_activation_confirmation.get("recovery_hold_steps", 0)
        )
        if not bool(rigid_pair_activation_confirmation.get("ok")):
            raise ValueError(
                _move_tracked_rigid_pair_activation_error(
                    rigid_pair_activation_confirmation
                )
            )
        rigid_pair_activation_provisional = False
        if bool(rigid_pair_activation_confirmation.get("recovered")):
            initial_activation_report = _move_tracked_rigid_pair_activation_summary(
                rigid_pair_activation
            )
            confirmed_pair_report = rigid_pair_activation_confirmation.get(
                "last_pair_report"
            )
            if not isinstance(confirmed_pair_report, Mapping):
                confirmed_pair_report = {}
            rigid_pair_activation = {
                **rigid_pair_activation,
                **dict(confirmed_pair_report),
                "ok": True,
                "reason": None,
                "initial_activation_report": initial_activation_report,
                "fresh_observation_confirmation": _json_ready(
                    rigid_pair_activation_confirmation
                ),
            }
            adapter, observation_sequence = _adjust_adapter(ctx)
            accepted_sequence = int(
                rigid_pair_activation_confirmation.get(
                    "observation_sequence_end", -1
                )
            )
            if observation_sequence != accepted_sequence:
                raise ValueError(
                    "evaluator observation changed after rigid-pair activation confirmation"
                )
        else:
            rigid_pair_activation = {
                **rigid_pair_activation,
                "fresh_observation_confirmation": _json_ready(
                    rigid_pair_activation_confirmation
                ),
            }
        live = _move_tracked_live_snapshot(
            manager,
            names,
            episode_id=episode_id,
        )
        if not bool(live.get("ok")):
            raise ValueError(
                "tracked points are unavailable: "
                f"{live.get('reason') or 'unknown'}"
            )
        if int(live.get("observation_sequence", -1)) != observation_sequence:
            raise ValueError("tracked points are not synchronized to current proprioception")
        session_id = str(live.get("session_id") or "").strip()
        image_id = str(live.get("image_id") or "").strip()
        if not session_id or not image_id:
            raise ValueError("active tracker binding has no session_id or image_id")
        observations = {
            side: _adjust_kinematic_observation(ctx, side)
            for side in ("left", "right")
        }
        sequence_after, _received_ts = adapter.observation_metadata()
        if int(sequence_after) != observation_sequence:
            raise ValueError("evaluator observation changed while freezing tracked points")
        if predefined_names:
            preselected_arm, predefined_arm_selection = (
                _move_tracked_select_predefined_eef_arm(
                    points=args["points"],
                    roles_by_name=roles_by_name,
                    observations=observations,
                    visual_entries=live.get("entries") or {},
                )
            )
            manager.set_arm(preselected_arm)
            rebound_live = _move_tracked_live_snapshot(
                manager,
                names,
                episode_id=episode_id,
            )
            if not bool(rebound_live.get("ok")):
                raise ValueError(
                    "predefined EEF points could not be rebound to the selected arm: "
                    f"{rebound_live.get('reason') or 'unknown'}"
                )
            if int(rebound_live.get("observation_sequence", -1)) != observation_sequence:
                raise ValueError(
                    "predefined EEF point arm selection changed the frozen observation"
                )
            if (
                str(rebound_live.get("session_id") or "") != session_id
                or str(rebound_live.get("image_id") or "") != image_id
            ):
                raise ValueError(
                    "predefined EEF point arm selection changed the tracker binding"
                )
            live = rebound_live
        capture_state = _move_tracked_capture_state(ctx, observations)
        source_matrix = _move_tracked_strict_point_matrix(
            [
                live["entries"][name]["xyz_in_robot_base_coord_m"]
                for name in names
            ],
            expected_count=len(names),
            label="tracked source point data",
        )
        source_by_name = {
            name: source_matrix[index].copy()
            for index, name in enumerate(names)
        }
        source_points = np.asarray(
            [source_by_name[name] for name in controlled_names],
            dtype=np.float64,
        )
        if visual_controlled_names:
            visual_controlled_points = np.asarray(
                [source_by_name[name] for name in visual_controlled_names],
                dtype=np.float64,
            )
            registered_on_hand_source_reconciliation = (
                _move_tracked_reconcile_registered_on_hand_source(
                    manager,
                    visual_controlled_names,
                    episode_id=episode_id,
                    current_capture_state=capture_state,
                    live_points_robot_base_m=visual_controlled_points,
                )
            )
        else:
            registered_on_hand_source_reconciliation = {
                "ok": True,
                "applicable": False,
                "applied": False,
                "reason": "all_on_hand_points_are_predefined_eef_geometry",
                "names": [],
                "source_points_robot_base_m": [],
                "predictions_published_as_observation": False,
            }
        if (
            args["execution_mode"] == "plan"
            and not registered_on_hand_source_reconciliation.get("ok")
            and registered_on_hand_source_reconciliation.get("reason") in {
                "registered_eef_anchor_consensus_insufficient",
                "registered_eef_anchor_recovery_exceeded",
            }
        ):
            reacquisition = yield from _move_tracked_reacquire_registered_bundle(
                ctx, manager, names=names, controlled_names=visual_controlled_names,
                episode_id=episode_id, session_id=session_id, image_id=image_id,
                observation_sequence=observation_sequence, hold_state=capture_state,
                initial_report=registered_on_hand_source_reconciliation,
                lease_state=(_eef_prior_lease_state if _eef_prior_lease_state is not None else {}),
                deadline_monotonic=request_deadline_monotonic,
            )
            if bool(reacquisition.get("ok")):
                adapter = reacquisition["adapter"]
                observation_sequence = reacquisition["observation_sequence"]
                live = reacquisition["live"]
                observations = reacquisition["observations"]
                capture_state = reacquisition["capture_state"]
                source_by_name = reacquisition["source_by_name"]
                source_points = np.asarray(
                    [source_by_name[name] for name in controlled_names],
                    dtype=np.float64,
                )
                registered_on_hand_source_reconciliation = reacquisition["reconciliation"]
            else:
                registered_on_hand_source_reconciliation = {
                    **registered_on_hand_source_reconciliation,
                    "reason": reacquisition["reason"],
                    "fresh_rgbd_reacquisition": _json_ready(reacquisition),
                }
        if not bool(registered_on_hand_source_reconciliation.get("ok")):
            reason = str(
                registered_on_hand_source_reconciliation.get("reason")
                or "registered_on_hand_source_reconciliation_failed"
            )
            raise ValueError(
                "on-hand tracked points are inconsistent with their "
                f"registration-time EEF anchors: {reason}"
            )
        if bool(registered_on_hand_source_reconciliation.get("applied")):
            source_points = _move_tracked_strict_point_matrix(
                registered_on_hand_source_reconciliation[
                    "source_points_robot_base_m"
                ],
                expected_count=len(visual_controlled_names),
                label="registered-EEF-reconciled on-hand planning source",
            )
            for index, name in enumerate(visual_controlled_names):
                source_by_name[name] = source_points[index].copy()
            source_points = np.asarray(
                [source_by_name[name] for name in controlled_names],
                dtype=np.float64,
            )
        reference_points = np.asarray(
            [source_by_name[name] for name in reference_names],
            dtype=np.float64,
        ) if reference_names else None
    except Exception as exc:
        activation_failure_reason = str(
            rigid_pair_activation_confirmation.get("reason") or ""
        )
        observation_failure_stage = (
            "cancellation"
            if _move_point_is_cancellation(exc)
            else (
                "timeout"
                if activation_failure_reason == "rigid_pair_activation_recovery_timeout"
                or registered_on_hand_source_reconciliation.get("reason")
                == "registered_bundle_reacquisition_timeout"
                else "observation validation"
            )
        )
        _move_tracked_failure(
            ctx,
            stage=observation_failure_stage,
            error=str(exc),
            args=args,
            details={
                "rigid_pair_activation": _json_ready(rigid_pair_activation),
                "rigid_pair_activation_confirmation": _json_ready(
                    rigid_pair_activation_confirmation
                ),
                "rigid_pair_activation_hold_steps": int(
                    rigid_pair_activation_hold_steps
                ),
                "prior_rigid_pair_cleanup": _json_ready(
                    prior_rigid_pair_cleanup
                ),
                "failed_rigid_pair_cleanup": _json_ready(
                    failed_rigid_pair_cleanup
                ),
                "rigid_pair_bootstrap_eef_observation_prior": _json_ready(
                    rigid_pair_bootstrap_eef_prior
                ),
                "registered_on_hand_source_reconciliation": _json_ready(
                    registered_on_hand_source_reconciliation
                ),
            },
        )
        yield (
            _move_tracked_planning_hold_action(ctx, activation_hold_state)
            if activation_hold_state
            else ctx.world.hold_action()
        )
        return

    initial_observation_sequence = int(observation_sequence)
    # The public timeout covers the full request, including pair activation.
    planning_started = request_started
    planning_budget_s = _move_tracked_planning_budget(args["timeout_s"])
    planning_deadline_monotonic = planning_started + planning_budget_s
    planning_hold_steps = 0
    planning_resume_without_observation_count = 0
    planning_sequence = int(observation_sequence)
    planning_failure_stage = "planning"
    planner_executor = None
    planner_future = None
    planner_cancel_event = threading.Event()
    planning_progress: dict[str, Any] = {}
    isolated_plan_worker = args["execution_mode"] == "plan"
    # Leave a bounded handoff window for signing and rendering the preview.
    planning_hard_deadline = planning_started + float(args["timeout_s"]) - min(
        5.0, 0.1 * float(args["timeout_s"])
    )
    path_failures: list[dict[str, Any]] = []
    plan_overlap_filter_context: dict[str, Any] | None = None
    frozen_plan_capture: dict[str, Any] | None = None
    plan_overlap_filter_report: dict[str, Any] | None = None
    # Camera-facing presets are solved against the exact head-camera pose that
    # belongs to the frozen tracked-point observation.  Coordinate/legacy
    # requests leave these optional values unset and retain their old path.
    head_camera_position_robot_base_m: np.ndarray | None = None
    head_camera_quaternion_xyzw: np.ndarray | None = None
    preplanning_on_hand_eef_prior: dict[str, Any] = {
        "ok": True,
        "active": False,
        "reason": "not_attempted",
    }
    preplanning_source_sync: dict[str, Any] = {
        "ok": True,
        "applicable": False,
        "reason": "eef_observation_prior_not_active",
    }
    try:
        distances = {
            side: float(
                np.mean(
                    [
                        np.linalg.norm(
                            source_by_name[name]
                            - observations[side]["position"]
                        )
                        for name in controlled_names
                    ]
                )
            )
            for side in ("left", "right")
        }
        reconciled_arm = str(
            registered_on_hand_source_reconciliation.get("selected_arm") or ""
        )
        arm = (
            preselected_arm
            if preselected_arm in {"left", "right"}
            else reconciled_arm
            if bool(registered_on_hand_source_reconciliation.get("applied"))
            and reconciled_arm in {"left", "right"}
            else min(("left", "right"), key=lambda side: distances[side])
        )
        manager.set_arm(arm)
        # Preserve the policy's point order.  It is part of the constraint
        # semantics: swapping ``head`` and ``tail`` would silently invert an
        # affine relation such as ``tail.x = head.x + d``.  The nearest
        # point is still computed and reported as the geometric anchor, but it
        # never changes source/target pairing.
        ordered_indices = list(range(len(names)))
        ordered_args = {
            **args,
            "points": [args["points"][index] for index in ordered_indices],
        }
        ordered_names = [names[index] for index in ordered_indices]
        # ``ordered_names`` is the complete tracker retention order.  The
        # planner and rigid-anchor monitor use only the controlled subset;
        # off-hand names remain frozen references checked independently.
        ordered_controlled_names = list(controlled_names)
        ordered_reference_names = list(reference_names)
        nearest_point_index = min(
            range(len(ordered_controlled_names)),
            key=lambda index: float(
                np.linalg.norm(
                    source_by_name[ordered_controlled_names[index]]
                    - observations[arm]["position"]
                )
            ),
        )
        nearest_point_name = ordered_controlled_names[nearest_point_index]
        source_points = np.asarray(
            [source_by_name[name] for name in ordered_controlled_names],
            dtype=np.float64,
        )
        reference_points = (
            np.asarray(
                [source_by_name[name] for name in ordered_reference_names],
                dtype=np.float64,
            )
            if ordered_reference_names
            else None
        )
        reference_target_points = _move_tracked_planning_targets(
            ordered_args["points"], ordered_reference_names
        ) if ordered_reference_names else []
        reference_name_set = set(ordered_reference_names)
        for relation in ordered_args.get("relations") or []:
            if (relation.get("type") != "ordered_collinear"
                    and reference_name_set.intersection(relation.get("point_names") or [])):
                raise ValueError(
                    "typed relations may reference only on-hand points; use a "
                    "role-based quick preset for on-hand/off-hand geometry"
                )

        # The RGB-D tracker can legitimately refine registered on-hand points
        # once the selected EEF prior is enabled.  That refinement must happen
        # before the immutable endpoint and trajectory are solved.  Previously
        # it happened after planning; the handoff gate accepted the new points
        # as EEF-rigid, then executed a trajectory built from stale XYZ.  The
        # resulting endpoint miss was approximately the accepted refinement and
        # repeated calls appeared to inch toward the goal.
        if args["execution_mode"] != "plan":
            initial_eef_position = np.asarray(
                capture_state["eef"][arm]["position"], dtype=np.float64
            ).reshape(3)
            initial_eef_rotation = _quat_to_mat_xyzw(
                capture_state["eef"][arm]["quaternion"]
            )
            preplanning_anchors_eef = (
                initial_eef_rotation.T
                @ (source_points - initial_eef_position[None, :]).T
            ).T
            preplanning_on_hand_eef_prior = (
                _move_tracked_activate_on_hand_eef_prior(
                    manager,
                    ordered_controlled_names,
                    episode_id=episode_id,
                    session_id=session_id,
                    image_id=image_id,
                    arm=arm,
                    anchors_eef_m=preplanning_anchors_eef,
                )
            )
            if _eef_prior_lease_state is not None:
                _eef_prior_lease_state.update(preplanning_on_hand_eef_prior)

            if bool(preplanning_on_hand_eef_prior.get("active")):
                source_sync_generator = _move_tracked_wait_for_stable_start(
                    ctx,
                    capture_state=capture_state,
                    manager=manager,
                    point_names=ordered_controlled_names,
                    source_points=source_points,
                    episode_id=episode_id,
                    session_id=session_id,
                    image_id=image_id,
                    pos_tol_m=float(args["pos_tol"]),
                    source_drift_tolerance_m=max(
                        0.012, float(args["pos_tol"])
                    ),
                    minimum_tracker_sequence_exclusive=int(
                        preplanning_on_hand_eef_prior[
                            "activation_observation_sequence"
                        ]
                    ),
                    rigid_eef_arm=arm,
                    rigid_anchors_eef=preplanning_anchors_eef,
                    rigid_anchor_tolerance_m=min(
                        MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M,
                        float(
                            preplanning_on_hand_eef_prior.get(
                                "max_point_error_m",
                                MOVE_TRACKED_POINT_START_EEF_ANCHOR_ERROR_M,
                            )
                        ),
                    ),
                    rigid_pair_distance_tolerance_m=(
                        _tracked_rigid_pair_tolerance(float(args["pos_tol"]))
                    ),
                    accept_rigid_rgbd_refinement=True,
                    deadline_monotonic=(
                        planning_started + float(args["timeout_s"])
                    ),
                )
                while True:
                    try:
                        yield next(source_sync_generator)
                        planning_hold_steps += 1
                    except StopIteration as stopped:
                        preplanning_source_sync = stopped.value
                        break
                if not bool(preplanning_source_sync.get("ok")):
                    sync_reason = str(
                        preplanning_source_sync.get("reason")
                        or "preplanning_source_sync_failed"
                    )
                    planning_failure_stage = (
                        "timeout" if sync_reason == "timeout" else "start-state validation"
                    )
                    raise RuntimeError(
                        str(
                            preplanning_source_sync.get("error")
                            or "could not synchronize tracked points before planning"
                        )
                    )

                synchronized_sequence = int(
                    preplanning_source_sync["observation_sequence_end"]
                )
                synchronized_live = _move_tracked_live_snapshot(
                    manager,
                    ordered_names,
                    episode_id=episode_id,
                )
                if not bool(synchronized_live.get("ok")):
                    raise RuntimeError(
                        "tracked points became unavailable while freezing the "
                        "post-prior planning source: "
                        f"{synchronized_live.get('reason') or 'unknown'}"
                    )
                if int(
                    synchronized_live.get("observation_sequence", -1)
                ) != synchronized_sequence:
                    raise RuntimeError(
                        "post-prior tracked points are not synchronized to "
                        "the accepted proprioception"
                    )
                synchronized_observations = {
                    side: _adjust_kinematic_observation(ctx, side)
                    for side in ("left", "right")
                }
                synchronized_adapter = preplanning_source_sync.get("adapter")
                if synchronized_adapter is None:
                    raise RuntimeError(
                        "post-prior source synchronization returned no evaluator adapter"
                    )
                synchronized_sequence_after, _received_ts = (
                    synchronized_adapter.observation_metadata()
                )
                if int(synchronized_sequence_after) != synchronized_sequence:
                    raise RuntimeError(
                        "evaluator observation changed while freezing the "
                        "post-prior planning source"
                    )
                synchronized_matrix = _move_tracked_strict_point_matrix(
                    [
                        synchronized_live["entries"][name][
                            "xyz_in_robot_base_coord_m"
                        ]
                        for name in ordered_names
                    ],
                    expected_count=len(ordered_names),
                    label="post-prior synchronized tracked source point data",
                )
                source_by_name = {
                    name: synchronized_matrix[index].copy()
                    for index, name in enumerate(ordered_names)
                }
                source_points = np.asarray(
                    [source_by_name[name] for name in ordered_controlled_names],
                    dtype=np.float64,
                )
                reference_points = (
                    np.asarray(
                        [
                            source_by_name[name]
                            for name in ordered_reference_names
                        ],
                        dtype=np.float64,
                    )
                    if ordered_reference_names
                    else None
                )
                capture_state = _move_tracked_capture_state(
                    ctx, synchronized_observations
                )
                observation_sequence = synchronized_sequence
                planning_sequence = synchronized_sequence
                distances = {
                    side: float(
                        np.mean(
                            [
                                np.linalg.norm(
                                    source_by_name[name]
                                    - synchronized_observations[side]["position"]
                                )
                                for name in ordered_controlled_names
                            ]
                        )
                    )
                    for side in ("left", "right")
                }
                nearest_point_index = min(
                    range(len(ordered_controlled_names)),
                    key=lambda index: float(
                        np.linalg.norm(
                            source_by_name[ordered_controlled_names[index]]
                            - synchronized_observations[arm]["position"]
                        )
                    ),
                )
                nearest_point_name = ordered_controlled_names[
                    nearest_point_index
                ]
                preplanning_source_sync = {
                    key: value
                    for key, value in preplanning_source_sync.items()
                    if key != "adapter"
                }
                preplanning_source_sync.update(
                    {
                        "applicable": True,
                        "planning_source_rebased": True,
                        "initial_observation_sequence": int(
                            initial_observation_sequence
                        ),
                        "planning_observation_sequence": int(
                            observation_sequence
                        ),
                        "planning_source_points_robot_base_m": (
                            source_points.astype(float).tolist()
                        ),
                        "all_points_frozen_from_one_observation": True,
                    }
                )
        elif args["execution_mode"] == "plan":
            # A plan-only request already froze points, proprioception, and
            # local FK from one evaluator sequence above. Requiring another
            # observation after enabling the execution-only EEF prior can
            # lose an otherwise valid marker at an image boundary and emits
            # needless hold actions. Exec mode retains the stricter post-prior
            # re-freeze and its subsequent live execution validation.
            preplanning_on_hand_eef_prior = {
                "ok": True,
                "active": False,
                "reason": "plan_mode_uses_current_synchronized_tracker_frame",
            }
            preplanning_source_sync = {
                "ok": True,
                "applicable": True,
                "reason": "plan_mode_current_synchronized_tracker_frame",
                "planning_source_rebased": False,
                "initial_observation_sequence": int(
                    initial_observation_sequence
                ),
                "planning_observation_sequence": int(observation_sequence),
                "planning_source_points_robot_base_m": (
                    source_points.astype(float).tolist()
                ),
                "all_points_frozen_from_one_observation": True,
                "post_prior_refreeze_required_for_direct_exec_mode": True,
            }
        elif bool(registered_on_hand_source_reconciliation.get("applied")):
            preplanning_source_sync = {
                "ok": True,
                "applicable": True,
                "reason": "registered_eef_anchor_outlier_recovery",
                "planning_source_rebased": True,
                "initial_observation_sequence": int(
                    initial_observation_sequence
                ),
                "planning_observation_sequence": int(observation_sequence),
                "planning_source_points_robot_base_m": (
                    source_points.astype(float).tolist()
                ),
                "live_validation_points_from_one_observation": True,
                "planning_source_uses_registration_rgbd_and_current_proprio": True,
                "predictions_published_as_observation": False,
            }
        if args["execution_mode"] == "plan":
            frozen_plan_capture = _move_tracked_freeze_plan_capture(
                ctx, session_id=session_id, registration_image_id=image_id,
                observation_sequence=observation_sequence,
            )
            plan_overlap_filter_context = _move_tracked_plan_overlap_context(
                session_id,
                frozen_plan_capture["image_id"],
            )
        if any(
            str(spec.get("type") or "").strip().lower()
            in {"faceto", "reverse_faceto"}
            for spec in quick_specs
        ):
            head_camera_position_robot_base_m, head_camera_quaternion_xyzw = (
                _move_tracked_head_camera_pose(
                    ctx,
                    expected_sequence=int(observation_sequence),
                )
            )
        planner_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="official-v2-move-tracked",
        )
        planner_future = planner_executor.submit(
            _move_tracked_plan_frozen,
            state=capture_state["state"],
            arm=arm,
            q_start=capture_state["arm_q"][arm].copy(),
            source_points=source_points.copy(),
            target_points=_move_tracked_planning_targets(
                ordered_args["points"], ordered_controlled_names
            ),
            relations=list(ordered_args.get("relations") or []),
            inequalities=list(ordered_args.get("inequalities") or []),
            quick_constraint=quick_spec,
            quick_constraints=args.get("quick_constraints"),
            off_hand_points=(
                None if reference_points is None else reference_points.copy()
            ),
            off_hand_target_points=reference_target_points,
            on_hand_names=ordered_controlled_names,
            off_hand_names=ordered_reference_names,
            head_camera_position_robot_base_m=head_camera_position_robot_base_m,
            head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            pos_tol_m=float(args["pos_tol"]),
            ori_tol_deg=float(args["ori_tol_deg"]),
            max_steps=int(args["max_steps"]),
            overlap_filter_context=plan_overlap_filter_context,
            cancel_event=planner_cancel_event,
            planning_deadline_monotonic=planning_deadline_monotonic,
            **({
                "isolate_process": True,
                "hard_deadline_monotonic": planning_hard_deadline,
                "progress_callback": lambda report: planning_progress.update({"report": report}),
            } if isolated_plan_worker else {}),
        )
        planning_result = None
        while planning_result is None:
            _check_cancelled(ctx, "move_tracked_point frozen local planning")
            if str(ctx.world.episode_id()) != episode_id:
                planning_failure_stage = "start-state validation"
                raise RuntimeError("episode_changed_during_planning")
            planning_elapsed_s = time.monotonic() - planning_started
            if planning_elapsed_s >= float(args["timeout_s"]) and not planner_future.done():
                planning_failure_stage = "timeout"
                raise RuntimeError("frozen local planning timed out")
            try:
                planning_result = planner_future.result(timeout=0.005)
            except concurrent.futures.TimeoutError:
                yield _move_tracked_planning_hold_action(ctx, capture_state)
                planning_hold_steps += 1
                advanced, current_sequence = _adjust_observation_advanced(
                    adapter,
                    planning_sequence,
                )
                if advanced:
                    planning_sequence = int(current_sequence)
                else:
                    planning_resume_without_observation_count += 1

        path_failures = list(planning_result.get("path_failures", []))
        raw_overlap_filter_report = planning_result.get(
            "frozen_rgbd_overlap_filter"
        )
        if isinstance(raw_overlap_filter_report, Mapping):
            plan_overlap_filter_report = _json_ready(raw_overlap_filter_report)
        if not bool(planning_result.get("ok")):
            raise ValueError(
                str(planning_result.get("error") or "tracked-point planning failed")
            )
        endpoint = planning_result["endpoint"]
        planning_elapsed_s = time.monotonic() - planning_started
        path_plan = {
            **planning_result["path_plan"],
            "planning_observation_action_loop": True,
            "planning_hold_steps": int(planning_hold_steps),
            "planning_resume_without_observation_count": int(
                planning_resume_without_observation_count
            ),
            "planning_elapsed_s": float(planning_elapsed_s),
            "planning_budget_s": float(planning_budget_s),
            "planning_deadline_monotonic": float(planning_deadline_monotonic),
            "planning_diagnostics": _json_ready(planning_progress.get("report", {})),
            "frozen_plan_capture": _json_ready(frozen_plan_capture),
            "planning_observation_sequence_start": int(observation_sequence),
            "planning_observation_sequence_end": int(planning_sequence),
            "registered_on_hand_source_reconciliation": _json_ready(
                registered_on_hand_source_reconciliation
            ),
        }
        plan_id = _next_plan_id(session_id)
        trajectory = _move_tracked_build_trajectory(
            plan_id=plan_id,
            session_id=session_id,
            image_id=image_id,
            observation_sequence=observation_sequence,
            args=ordered_args,
            capture_state=capture_state,
            arm=arm,
            source_names=ordered_names,
            point_a_name=nearest_point_name,
            relations=list(ordered_args.get("relations") or []),
            source_points=source_points,
            endpoint=endpoint,
            path_plan=path_plan,
            controlled_names=ordered_controlled_names,
            reference_names=ordered_reference_names,
            reference_points=reference_points,
            reference_target_points=planning_result.get(
                "off_hand_target_points", reference_target_points
            ),
            quick_constraint=quick_spec,
            quick_constraints=args.get("quick_constraints"),
            planning_target_points=planning_result.get(
                "planning_target_points"
            ),
            planning_relations=planning_result.get("planning_relations"),
            head_camera_position_robot_base_m=head_camera_position_robot_base_m,
            head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
        )
        plan_path = _move_tracked_write_plan(trajectory)
        trajectory, plan_path = _move_tracked_load_plan(session_id, plan_id)
        if args["execution_mode"] == "plan":
            plan_result = _move_tracked_plan_preview_result(
                move_trajectory=trajectory,
                plan_path=plan_path,
            )
            plan_result.update(
                {
                    "preplanning_on_hand_eef_observation_prior": _json_ready(
                        preplanning_on_hand_eef_prior
                    ),
                    "preplanning_source_sync": _json_ready(
                        preplanning_source_sync
                    ),
                    "rigid_pair_activation": _json_ready(
                        rigid_pair_activation
                    ),
                    "rigid_pair_activation_confirmation": _json_ready(
                        rigid_pair_activation_confirmation
                    ),
                    "rigid_pair_bootstrap_eef_observation_prior": _json_ready(
                        rigid_pair_bootstrap_eef_prior
                    ),
                    "registered_on_hand_source_reconciliation": _json_ready(
                        registered_on_hand_source_reconciliation
                    ),
                }
            )
            _set_result(ctx, "move_tracked_point", plan_result)
            yield _move_tracked_planning_hold_action(ctx, capture_state)
            return
        planned_anchors_eef = _move_tracked_strict_point_matrix(
            trajectory["target"]["anchors_eef_m"],
            expected_count=len(ordered_controlled_names),
            label="move-tracked trajectory anchors",
        )
        on_hand_eef_prior = _move_tracked_activate_on_hand_eef_prior(
            manager,
            ordered_controlled_names,
            episode_id=episode_id,
            session_id=session_id,
            image_id=image_id,
            arm=arm,
            anchors_eef_m=planned_anchors_eef,
        )
        if _eef_prior_lease_state is not None:
            _eef_prior_lease_state.update(on_hand_eef_prior)
    except Exception as exc:
        planner_cancel_event.set()
        if planner_future is not None:
            planner_future.cancel()
        if _move_point_is_cancellation(exc):
            planning_failure_stage = "cancellation"
        elif isinstance(exc, PlanningDeadlineExceeded):
            planning_failure_stage = "timeout"
        _move_tracked_failure(
            ctx,
            stage=planning_failure_stage,
            error=str(exc),
            args=args,
            details={
                "robot_profile": ROBOT_PROFILE,
                "selected_arm": locals().get("arm"),
                "eef_mean_distance_m": locals().get("distances"),
                "endpoint_path_candidate_failures": path_failures,
                "frozen_rgbd_overlap_filter": _json_ready(
                    plan_overlap_filter_report
                ),
                "preplanning_on_hand_eef_observation_prior": _json_ready(
                    preplanning_on_hand_eef_prior
                ),
                "preplanning_source_sync": _json_ready(
                    {
                        key: value
                        for key, value in preplanning_source_sync.items()
                        if key != "adapter"
                    }
                ),
                "planning_observation_action_loop": True,
                "planning_budget_s": float(planning_budget_s),
                "planning_diagnostics": _json_ready(planning_progress.get("report", {})),
                "planning_hold_steps": int(planning_hold_steps),
                "planning_resume_without_observation_count": int(
                    planning_resume_without_observation_count
                ),
                "planning_observation_sequence_start": int(
                    observation_sequence
                ),
                "planning_observation_sequence_end": int(planning_sequence),
                **(
                    {"motion": {"ok": False, "reason": "timeout"}}
                    if planning_failure_stage == "timeout"
                    else {}
                ),
            },
        )
        yield ctx.world.hold_action()
        return
    finally:
        if planner_executor is not None:
            planner_cancel_event.set()
            planner_executor.shutdown(wait=isolated_plan_worker, cancel_futures=True)

    start_deadline = planning_started + float(args["timeout_s"])
    post_planning_source_replans: list[dict[str, Any]] = []
    while True:
        try:
            stable_start_generator = _move_tracked_wait_for_stable_start(
                ctx,
                capture_state=capture_state,
                manager=manager,
                point_names=ordered_controlled_names,
                source_points=source_points,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                pos_tol_m=float(args["pos_tol"]),
                source_drift_tolerance_m=(
                    max(0.012, float(args["pos_tol"]))
                    if bool(on_hand_eef_prior.get("active"))
                    else None
                ),
                minimum_tracker_sequence_exclusive=(
                    int(on_hand_eef_prior["activation_observation_sequence"])
                    if bool(on_hand_eef_prior.get("active"))
                    else None
                ),
                rigid_eef_arm=(
                    arm if bool(on_hand_eef_prior.get("active")) else None
                ),
                rigid_anchors_eef=(
                    planned_anchors_eef
                    if bool(on_hand_eef_prior.get("active"))
                    else None
                ),
                rigid_anchor_tolerance_m=(
                    min(
                        MOVE_TRACKED_POINT_MAX_ANCHOR_ERROR_M,
                        float(
                            on_hand_eef_prior.get(
                                "max_point_error_m",
                                MOVE_TRACKED_POINT_START_EEF_ANCHOR_ERROR_M,
                            )
                        ),
                    )
                    if bool(on_hand_eef_prior.get("active"))
                    else None
                ),
                rigid_pair_distance_tolerance_m=(
                    _tracked_rigid_pair_tolerance(float(args["pos_tol"]))
                    if bool(on_hand_eef_prior.get("active"))
                    else None
                ),
                accept_rigid_rgbd_refinement=False,
                deadline_monotonic=start_deadline,
            )
            while True:
                try:
                    yield next(stable_start_generator)
                except StopIteration as stopped:
                    start_validation = stopped.value
                    break
        except Exception as exc:
            if _move_point_is_cancellation(exc):
                start_validation = {
                    "ok": False,
                    "reason": "cancellation",
                    "error": str(exc) or "move_tracked_point was cancelled",
                }
            else:
                # A malformed/vanished evaluator sample must stay inside the
                # official tool result contract.  Letting a generator
                # exception escape here discards the handoff diagnostics.
                start_validation = {
                    "ok": False,
                    "reason": "start_state_exception",
                    "error": (
                        "move_tracked_point start-state validation failed: "
                        f"{exc}"
                    ),
                    "exception_type": type(exc).__name__,
                }

        if bool(start_validation.get("ok")):
            break
        start_reason = str(
            start_validation.get("reason") or "start_state_not_stable"
        )
        can_rebase = bool(
            start_reason == "tracked_points_moved_after_source_sync"
            and bool(on_hand_eef_prior.get("active"))
            and len(ordered_controlled_names)
            >= MOVE_TRACKED_POINT_FINAL_VISUAL_MIN_POINTS
            and len(post_planning_source_replans)
            < MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_ROUNDS
        )
        if not can_rebase:
            break

        old_plan_id = str(plan_id)
        old_plan_digest = str(trajectory["integrity"]["digest"])
        replan_round = len(post_planning_source_replans) + 1
        replan_diagnostic: dict[str, Any] = {
            "round": int(replan_round),
            "trigger": start_reason,
            "trigger_source_drift_m": start_validation.get(
                "last_source_drift_m"
            ),
            "trigger_eef_anchor_error_m": start_validation.get(
                "last_eef_anchor_error_m"
            ),
            "discarded_plan_id": old_plan_id,
            "discarded_trajectory_digest": old_plan_digest,
            "discarded_plan_executed": False,
        }
        try:
            rebase_drift = float(
                start_validation.get("last_source_drift_m", float("inf"))
            )
            rebase_anchor_error = float(
                start_validation.get("last_eef_anchor_error_m", float("inf"))
            )
            rebase_geometry = start_validation.get(
                "last_rigid_geometry_consistency"
            )
            if (
                not math.isfinite(rebase_drift)
                or rebase_drift
                > MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M
            ):
                raise RuntimeError(
                    "post-plan RGB-D refinement exceeds the bounded rebase "
                    f"envelope ({rebase_drift:.4f}m > "
                    f"{MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M:.4f}m)"
                )
            if (
                not math.isfinite(rebase_anchor_error)
                or rebase_anchor_error
                > MOVE_TRACKED_POINT_POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M
            ):
                raise RuntimeError(
                    "post-plan RGB-D bundle is not close enough to the live "
                    "EEF for a synchronized rebase"
                )
            if not isinstance(rebase_geometry, Mapping) or not bool(
                rebase_geometry.get("ok")
            ):
                raise RuntimeError(
                    "post-plan RGB-D bundle changed rigid geometry"
                )
            remaining_s = float(start_deadline - time.monotonic())
            if (
                remaining_s
                <= MOVE_TRACKED_POINT_POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S
            ):
                raise RuntimeError(
                    "not enough tool timeout remains for a synchronized "
                    "post-plan rebase"
                )

            rebase_sequence = int(
                start_validation.get("observation_sequence_end", -1)
            )
            rebase_adapter = start_validation.get("adapter")
            if rebase_adapter is None or rebase_sequence < 0:
                raise RuntimeError(
                    "post-plan rebase has no synchronized evaluator adapter"
                )
            rebase_live = _move_tracked_live_snapshot(
                manager,
                ordered_names,
                episode_id=episode_id,
            )
            if not bool(rebase_live.get("ok")):
                raise RuntimeError(
                    "tracked points became unavailable during post-plan "
                    f"rebase: {rebase_live.get('reason') or 'unknown'}"
                )
            if int(rebase_live.get("observation_sequence", -1)) != rebase_sequence:
                raise RuntimeError(
                    "post-plan rebase points are not synchronized to "
                    "proprioception"
                )
            if (
                str(rebase_live.get("session_id")) != session_id
                or str(rebase_live.get("image_id")) != image_id
            ):
                raise RuntimeError(
                    "tracked-point registration changed during post-plan rebase"
                )
            rebase_observations = {
                side: _adjust_kinematic_observation(ctx, side)
                for side in ("left", "right")
            }
            rebase_sequence_after, _received_ts = (
                rebase_adapter.observation_metadata()
            )
            if int(rebase_sequence_after) != rebase_sequence:
                raise RuntimeError(
                    "evaluator observation changed while freezing the "
                    "post-plan rebase"
                )
            rebase_matrix = _move_tracked_strict_point_matrix(
                [
                    rebase_live["entries"][name][
                        "xyz_in_robot_base_coord_m"
                    ]
                    for name in ordered_names
                ],
                expected_count=len(ordered_names),
                label="post-plan synchronized tracked source point data",
            )
            rebase_by_name = {
                name: rebase_matrix[index].copy()
                for index, name in enumerate(ordered_names)
            }
            rebase_source_points = np.asarray(
                [rebase_by_name[name] for name in ordered_controlled_names],
                dtype=np.float64,
            )
            gate_live_points = _move_tracked_strict_point_matrix(
                start_validation.get("last_live_points_robot_base_m"),
                expected_count=len(ordered_controlled_names),
                label="post-plan handoff point data",
            )
            if float(
                np.max(
                    np.linalg.norm(
                        rebase_source_points - gate_live_points,
                        axis=1,
                    )
                )
            ) > 1.0e-6:
                raise RuntimeError(
                    "post-plan rebase point bundle changed after the stable "
                    "handoff sample"
                )

            rebase_reference_points = None
            rebase_reference_validation = {
                "ok": True,
                "applicable": False,
            }
            if ordered_reference_names:
                if reference_points is None:
                    raise RuntimeError(
                        "post-plan rebase has no frozen off-hand references"
                    )
                rebase_reference_points = np.asarray(
                    [rebase_by_name[name] for name in ordered_reference_names],
                    dtype=np.float64,
                )
                rebase_reference_validation = (
                    _move_tracked_off_hand_reference_consensus(
                        ordered_reference_names,
                        rebase_reference_points,
                        reference_points,
                        tolerance_m=(
                            MOVE_TRACKED_POINT_OFF_HAND_REFERENCE_TOLERANCE_M
                        ),
                        allow_visual_outlier_suppression=False,
                    )
                )
                if not bool(rebase_reference_validation.get("ok")):
                    raise RuntimeError(
                        str(
                            rebase_reference_validation.get("error")
                            or "off-hand reference points moved before post-plan rebase"
                        )
                    )

            capture_state = _move_tracked_capture_state(
                ctx,
                rebase_observations,
            )
            source_by_name = rebase_by_name
            source_points = rebase_source_points
            reference_points = rebase_reference_points
            observation_sequence = rebase_sequence
            planning_sequence = rebase_sequence
            observations = rebase_observations
            distances = {
                side: float(
                    np.mean(
                        [
                            np.linalg.norm(
                                source_by_name[name]
                                - observations[side]["position"]
                            )
                            for name in ordered_controlled_names
                        ]
                    )
                )
                for side in ("left", "right")
            }
            nearest_point_index = min(
                range(len(ordered_controlled_names)),
                key=lambda index: float(
                    np.linalg.norm(
                        source_by_name[ordered_controlled_names[index]]
                        - observations[arm]["position"]
                    )
                ),
            )
            nearest_point_name = ordered_controlled_names[
                nearest_point_index
            ]
            rebase_eef_position = np.asarray(
                capture_state["eef"][arm]["position"], dtype=np.float64
            ).reshape(3)
            rebase_eef_rotation = _quat_to_mat_xyzw(
                capture_state["eef"][arm]["quaternion"]
            )
            rebase_anchors_eef = (
                rebase_eef_rotation.T
                @ (source_points - rebase_eef_position[None, :]).T
            ).T
            on_hand_eef_prior = _move_tracked_activate_on_hand_eef_prior(
                manager,
                ordered_controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=rebase_anchors_eef,
            )
            if _eef_prior_lease_state is not None:
                _eef_prior_lease_state.update(on_hand_eef_prior)

            round_deadline = min(
                float(planning_deadline_monotonic),
                float(
                    start_deadline
                    - MOVE_TRACKED_POINT_POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S
                ),
            )
            if time.monotonic() >= round_deadline:
                raise PlanningDeadlineExceeded(
                    "post-plan synchronized replan has no remaining planning budget"
                )
            if any(
                str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
                in {"faceto", "reverse_faceto"}
                for spec in quick_specs
            ):
                # Rebase planning uses a fresh frozen point/proprio frame.  Keep
                # the camera optical axis from that exact evaluator sequence so
                # the signed relation and later live winding check cannot drift
                # apart during a long handoff.
                head_camera_position_robot_base_m, head_camera_quaternion_xyzw = (
                    _move_tracked_head_camera_pose(
                        ctx,
                        expected_sequence=int(rebase_sequence),
                    )
                )
            replan_cancel_event = threading.Event()
            replan_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="official-v2-move-tracked-rebase",
            )
            replan_future = replan_executor.submit(
                _move_tracked_plan_frozen,
                state=capture_state["state"],
                arm=arm,
                q_start=capture_state["arm_q"][arm].copy(),
                source_points=source_points.copy(),
                target_points=_move_tracked_planning_targets(
                    ordered_args["points"], ordered_controlled_names
                ),
                relations=list(ordered_args.get("relations") or []),
                inequalities=list(ordered_args.get("inequalities") or []),
                quick_constraint=quick_spec,
                quick_constraints=args.get("quick_constraints"),
                off_hand_points=(
                    None
                    if reference_points is None
                    else reference_points.copy()
                ),
                off_hand_target_points=reference_target_points,
                on_hand_names=ordered_controlled_names,
                off_hand_names=ordered_reference_names,
                head_camera_position_robot_base_m=head_camera_position_robot_base_m,
                head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
                pos_tol_m=float(args["pos_tol"]),
                ori_tol_deg=float(args["ori_tol_deg"]),
                max_steps=int(args["max_steps"]),
                cancel_event=replan_cancel_event,
                planning_deadline_monotonic=round_deadline,
            )
            replan_started = time.monotonic()
            replanning_result = None
            replan_sequence_end = int(rebase_sequence)
            try:
                while replanning_result is None:
                    _check_cancelled(
                        ctx,
                        "move_tracked_point post-plan synchronized replanning",
                    )
                    if str(ctx.world.episode_id()) != episode_id:
                        raise RuntimeError(
                            "episode_changed_during_post_plan_replanning"
                        )
                    if time.monotonic() >= round_deadline:
                        raise PlanningDeadlineExceeded(
                            "post-plan synchronized local replanning timed out"
                        )
                    try:
                        replanning_result = replan_future.result(timeout=0.005)
                    except concurrent.futures.TimeoutError:
                        yield _move_tracked_planning_hold_action(
                            ctx,
                            capture_state,
                        )
                        planning_hold_steps += 1
                        advanced, current_sequence = (
                            _adjust_observation_advanced(
                                rebase_adapter,
                                replan_sequence_end,
                            )
                        )
                        if advanced:
                            replan_sequence_end = int(current_sequence)
                            planning_sequence = int(current_sequence)
                        else:
                            planning_resume_without_observation_count += 1
            finally:
                if replanning_result is None:
                    replan_cancel_event.set()
                    replan_future.cancel()
                replan_executor.shutdown(wait=False, cancel_futures=True)

            path_failures = list(
                replanning_result.get("path_failures", [])
            )
            if not bool(replanning_result.get("ok")):
                raise RuntimeError(
                    str(
                        replanning_result.get("error")
                        or "post-plan synchronized replanning failed"
                    )
                )
            endpoint = replanning_result["endpoint"]
            planning_elapsed_s = time.monotonic() - planning_started
            path_plan = {
                **replanning_result["path_plan"],
                "planning_observation_action_loop": True,
                "planning_hold_steps": int(planning_hold_steps),
                "planning_resume_without_observation_count": int(
                    planning_resume_without_observation_count
                ),
                "planning_elapsed_s": float(planning_elapsed_s),
                "planning_budget_s": float(planning_budget_s),
                "planning_deadline_monotonic": float(round_deadline),
                "planning_observation_sequence_start": int(
                    rebase_sequence
                ),
                "planning_observation_sequence_end": int(
                    replan_sequence_end
                ),
                "post_plan_source_rebase_round": int(replan_round),
            }
            plan_id = _next_plan_id(session_id)
            trajectory = _move_tracked_build_trajectory(
                plan_id=plan_id,
                session_id=session_id,
                image_id=image_id,
                observation_sequence=observation_sequence,
                args=ordered_args,
                capture_state=capture_state,
                arm=arm,
                source_names=ordered_names,
                point_a_name=nearest_point_name,
                relations=list(ordered_args.get("relations") or []),
                source_points=source_points,
                endpoint=endpoint,
                path_plan=path_plan,
                controlled_names=ordered_controlled_names,
                reference_names=ordered_reference_names,
                reference_points=reference_points,
                reference_target_points=replanning_result.get(
                    "off_hand_target_points", reference_target_points
                ),
                quick_constraint=quick_spec,
                quick_constraints=args.get("quick_constraints"),
                planning_target_points=replanning_result.get(
                    "planning_target_points"
                ),
                planning_relations=replanning_result.get(
                    "planning_relations"
                ),
                head_camera_position_robot_base_m=head_camera_position_robot_base_m,
                head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            )
            plan_path = _move_tracked_write_plan(trajectory)
            trajectory, plan_path = _move_tracked_load_plan(
                session_id,
                plan_id,
            )
            planned_anchors_eef = _move_tracked_strict_point_matrix(
                trajectory["target"]["anchors_eef_m"],
                expected_count=len(ordered_controlled_names),
                label="post-plan rebased trajectory anchors",
            )
            on_hand_eef_prior = _move_tracked_activate_on_hand_eef_prior(
                manager,
                ordered_controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=planned_anchors_eef,
            )
            if _eef_prior_lease_state is not None:
                _eef_prior_lease_state.update(on_hand_eef_prior)
            replan_diagnostic.update(
                {
                    "ok": True,
                    "source_observation_sequence": int(rebase_sequence),
                    "source_points_robot_base_m": (
                        source_points.astype(float).tolist()
                    ),
                    "off_hand_reference_validation": _json_ready(
                        rebase_reference_validation
                    ),
                    "planning_elapsed_s": float(
                        time.monotonic() - replan_started
                    ),
                    "new_plan_id": str(plan_id),
                    "new_trajectory_digest": str(
                        trajectory["integrity"]["digest"]
                    ),
                    "new_plan_schema_validated": True,
                    "new_plan_j8_locked_zero": True,
                }
            )
            post_planning_source_replans.append(replan_diagnostic)
            continue
        except Exception as exc:
            replan_diagnostic.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            post_planning_source_replans.append(replan_diagnostic)
            start_validation = {
                **start_validation,
                "ok": False,
                "reason": "post_planning_source_replan_failed",
                "error": (
                    "tracked points changed coherently after planning, but "
                    f"the synchronized replacement plan failed: {exc}"
                ),
            }
            break
    start_ok = bool(start_validation.get("ok"))
    if not start_ok:
        start_reason = str(
            start_validation.get("reason") or "start_state_not_stable"
        )
        start_failure_stage = (
            "timeout"
            if start_reason == "timeout"
            else "cancellation"
            if start_reason == "cancellation"
            else "start-state validation"
        )
        _move_tracked_failure(
            ctx,
            stage=start_failure_stage,
            error=str(
                start_validation.get("error")
                or start_validation.get("reason")
                or "move-tracked start state is not stable"
            ),
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": trajectory["integrity"]["digest"],
                "start_state_validation": {
                    key: value
                    for key, value in start_validation.items()
                    if key != "adapter"
                },
                "preplanning_source_sync": _json_ready(
                    preplanning_source_sync
                ),
                "post_planning_source_replans": _json_ready(
                    post_planning_source_replans
                ),
                "post_planning_source_replan_count": int(
                    len(post_planning_source_replans)
                ),
            },
        )
        yield ctx.world.hold_action()
        return
    off_hand_start_validation = {"ok": True, "applicable": False}
    if ordered_reference_names and reference_points is not None:
        off_hand_start_validation = _move_tracked_reference_snapshot_check(
            manager,
            ordered_reference_names,
            reference_points,
            episode_id=episode_id,
            expected_sequence=int(start_validation.get("observation_sequence_end", -1)),
            tolerance_m=max(0.020, 2.0 * float(args["pos_tol"])),
        )
        if not bool(off_hand_start_validation.get("ok")):
            _move_tracked_failure(
                ctx,
                stage="start-state validation",
                error=str(
                    off_hand_start_validation.get("error")
                    or "off-hand reference points moved before execution"
                ),
                args=args,
                details={
                    "plan_id": plan_id,
                    "trajectory_digest": trajectory["integrity"]["digest"],
                    "off_hand_start_validation": _json_ready(off_hand_start_validation),
                },
            )
            yield ctx.world.hold_action()
            return
    start_adapter = start_validation["adapter"]
    start_sequence = int(start_validation["observation_sequence_end"])

    try:
        anchors_eef = planned_anchors_eef.copy()
        resolved_points = _move_tracked_strict_point_matrix(
            trajectory["target"]["resolved_points_robot_base_m"],
            expected_count=len(ordered_controlled_names),
            label="move-tracked trajectory resolved points",
        )
    except ValueError as exc:
        _move_tracked_failure(
            ctx,
            stage="planning",
            error=str(exc),
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": trajectory["integrity"]["digest"],
            },
        )
        yield ctx.world.hold_action()
        return
    anchor_tolerance = float(
        trajectory["tracking"]["rigid_anchor_tolerance_m"]
    )
    pair_distance_tolerance = float(
        trajectory["tracking"]["rigid_pair_distance_tolerance_m"]
    )
    point_displacements = resolved_points - source_points
    point_distances = np.linalg.norm(point_displacements, axis=1)
    point_directions = np.zeros_like(point_displacements)
    moving_points = point_distances > max(1e-6, 0.25 * float(args["pos_tol"]))
    point_directions[moving_points] = (
        point_displacements[moving_points]
        / point_distances[moving_points, None]
    )
    progress_high_water = np.zeros(len(ordered_controlled_names), dtype=np.float64)
    max_progress_regression = np.zeros(len(ordered_controlled_names), dtype=np.float64)
    live_progress_regression_limit = float(
        trajectory["tracking"]["max_live_progress_regression_m"]
    )
    tracking_continuity = {
        "observed_steps": 0,
        "temporarily_unobserved_steps": 0,
        "current_unobserved_streak": 0,
        "max_unobserved_streak": 0,
        "reacquisition_count": 0,
        "last_unobserved_reason": None,
        "stale_coordinates_used": False,
        "trajectory_changed_for_reacquisition": False,
        "off_hand_tracking_outlier_steps": 0,
        "off_hand_tracking_outlier_names": [],
        "off_hand_reference_substitution_used": False,
        "controlled_anchor_disturbance_steps": 0,
        "controlled_rigid_geometry_disturbance_steps": 0,
        "max_controlled_anchor_error_m": 0.0,
        "max_controlled_pair_distance_error_m": 0.0,
            "execution_pauses_for_visual_reacquisition": bool(
                len(visual_controlled_names)
                >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
            ),
        "visual_reacquisition_max_hold_steps": int(
            MOVE_TRACKED_POINT_NPOINT_REACQUISITION_MAX_HOLD_STEPS
        ),
    }
    pause_for_visual_reacquisition = bool(
        len(visual_controlled_names)
        >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
    )

    def step_monitor(after: dict[str, Any], sequence: int):
        current_live = _move_tracked_live_snapshot(
            manager, ordered_names, episode_id=episode_id
        )
        report = {
            "observation_sequence": int(sequence),
            "anchor_tolerance_m": anchor_tolerance,
            "pair_distance_tolerance_m": pair_distance_tolerance,
        }
        if not bool(current_live.get("ok")):
            tracker_reason = str(
                current_live.get("reason") or "tracked_point_unavailable"
            )
            tracking_continuity["temporarily_unobserved_steps"] += 1
            tracking_continuity["current_unobserved_streak"] += 1
            tracking_continuity["max_unobserved_streak"] = max(
                int(tracking_continuity["max_unobserved_streak"]),
                int(tracking_continuity["current_unobserved_streak"]),
            )
            tracking_continuity["last_unobserved_reason"] = tracker_reason
            report.update(
                {
                    "tracker_reason": tracker_reason,
                    "rigid_pair_fusion": _json_ready(
                        current_live.get("rigid_pair_fusion") or {}
                    ),
                    "temporarily_unobserved": pause_for_visual_reacquisition,
                    "intermediate_visual_measurement_unavailable": True,
                    "trajectory_continues_from_immutable_plan": bool(
                        not pause_for_visual_reacquisition
                    ),
                    "trajectory_paused_for_reacquisition": bool(
                        pause_for_visual_reacquisition
                    ),
                    "final_reacquisition_required": True,
                }
            )
            # No coordinates are consumed on this sample. Proprioception and
            # local FK continue validating the precomputed trajectory; final
            # success remains impossible until all points are observed again.
            return None, report
        if int(current_live.get("observation_sequence", -1)) != int(sequence):
            report["tracker_sequence"] = current_live.get("observation_sequence")
            tracking_continuity["temporarily_unobserved_steps"] += 1
            tracking_continuity["current_unobserved_streak"] += 1
            tracking_continuity["max_unobserved_streak"] = max(
                int(tracking_continuity["max_unobserved_streak"]),
                int(tracking_continuity["current_unobserved_streak"]),
            )
            tracking_continuity[
                "last_unobserved_reason"
            ] = "tracker_observation_mismatch"
            report.update(
                {
                    "tracker_reason": "tracker_observation_mismatch",
                    "rigid_pair_fusion": _json_ready(
                        current_live.get("rigid_pair_fusion") or {}
                    ),
                    "temporarily_unobserved": pause_for_visual_reacquisition,
                    "intermediate_visual_measurement_unavailable": True,
                    "trajectory_continues_from_immutable_plan": bool(
                        not pause_for_visual_reacquisition
                    ),
                    "trajectory_paused_for_reacquisition": bool(
                        pause_for_visual_reacquisition
                    ),
                    "final_reacquisition_required": True,
                }
            )
            return None, report
        if int(tracking_continuity["current_unobserved_streak"]) > 0:
            tracking_continuity["reacquisition_count"] += 1
        tracking_continuity["current_unobserved_streak"] = 0
        tracking_continuity["observed_steps"] += 1
        reference_measured = None
        reference_for_constraints = None
        reference_drift_report = {"ok": True, "applicable": False}
        if ordered_reference_names:
            try:
                reference_measured = _move_tracked_strict_point_matrix(
                    [
                        current_live["entries"][name]["xyz_in_robot_base_coord_m"]
                        for name in ordered_reference_names
                    ],
                    expected_count=len(ordered_reference_names),
                    label="off-hand execution point data",
                )
                reference_limit = (
                    MOVE_TRACKED_POINT_OFF_HAND_REFERENCE_TOLERANCE_M
                )
                reference_drift_report = (
                    _move_tracked_off_hand_reference_consensus(
                        ordered_reference_names,
                        reference_measured,
                        reference_points,
                        tolerance_m=reference_limit,
                        allow_visual_outlier_suppression=True,
                    )
                )
                # Off-hand RGB-D remains authoritative for motion/drift
                # validation.  Geometry is evaluated against the immutable
                # coordinates frozen at request start.
                reference_for_constraints = reference_points
                if bool(
                    reference_drift_report.get("tracking_outlier_suppressed")
                ):
                    tracking_continuity["off_hand_tracking_outlier_steps"] += 1
                    tracking_continuity[
                        "off_hand_reference_substitution_used"
                    ] = True
                    known_outliers = set(
                        tracking_continuity["off_hand_tracking_outlier_names"]
                    )
                    known_outliers.update(
                        str(name)
                        for name in reference_drift_report.get(
                            "outlier_names", []
                        )
                    )
                    tracking_continuity[
                        "off_hand_tracking_outlier_names"
                    ] = sorted(known_outliers)
            except (KeyError, TypeError, ValueError) as exc:
                reference_drift_report = {
                    "ok": False,
                    "applicable": True,
                    "reason": "off_hand_points_invalid",
                    "error": str(exc),
                    "max_drift_m": float("inf"),
                }
            if not bool(reference_drift_report.get("ok")):
                report.update({"off_hand_reference": reference_drift_report})
                return "off_hand_points_moved", report
        try:
            measured = _move_tracked_strict_point_matrix(
                [
                    current_live["entries"][name]["xyz_in_robot_base_coord_m"]
                    for name in ordered_controlled_names
                ],
                expected_count=len(ordered_controlled_names),
                label="tracked execution point data",
            )
        except (KeyError, TypeError, ValueError) as exc:
            tracking_continuity["temporarily_unobserved_steps"] += 1
            tracking_continuity["current_unobserved_streak"] += 1
            tracking_continuity["max_unobserved_streak"] = max(
                int(tracking_continuity["max_unobserved_streak"]),
                int(tracking_continuity["current_unobserved_streak"]),
            )
            tracking_continuity["last_unobserved_reason"] = (
                "tracked_point_invalid"
            )
            report.update(
                {
                    "tracker_reason": "tracked_point_invalid",
                    "error": str(exc),
                    "temporarily_unobserved": pause_for_visual_reacquisition,
                    "intermediate_visual_measurement_unavailable": True,
                    "trajectory_continues_from_immutable_plan": bool(
                        not pause_for_visual_reacquisition
                    ),
                    "trajectory_paused_for_reacquisition": bool(
                        pause_for_visual_reacquisition
                    ),
                    "final_reacquisition_required": True,
                }
            )
            return None, report
        predicted = np.asarray(after["position"])[None, :] + (
            _quat_to_mat_xyzw(after["quaternion"]) @ anchors_eef.T
        ).T
        errors = np.linalg.norm(measured - predicted, axis=1)
        pair_consistency = _tracked_rigid_pair_consistency(
            measured,
            source_points,
            tolerance_m=pair_distance_tolerance,
        )
        maximum_anchor_error = float(np.max(errors))
        pair_distance_error = float(
            pair_consistency.get(
                "max_pair_distance_error_m",
                pair_consistency.get("distance_error_m", 0.0),
            )
        )
        if not math.isfinite(pair_distance_error):
            # The complete pair report retains the invalid diagnostic.  Do
            # not publish JSON Infinity through this aggregate counter.
            pair_distance_error = 0.0
        tracking_continuity["max_controlled_anchor_error_m"] = max(
            float(tracking_continuity["max_controlled_anchor_error_m"]),
            maximum_anchor_error,
        )
        tracking_continuity["max_controlled_pair_distance_error_m"] = max(
            float(
                tracking_continuity[
                    "max_controlled_pair_distance_error_m"
                ]
            ),
            pair_distance_error,
        )
        anchor_disturbed = bool(maximum_anchor_error > anchor_tolerance)
        rigid_geometry_disturbed = bool(not pair_consistency.get("ok"))
        if anchor_disturbed:
            tracking_continuity["controlled_anchor_disturbance_steps"] += 1
        if rigid_geometry_disturbed:
            tracking_continuity[
                "controlled_rigid_geometry_disturbance_steps"
            ] += 1
        progress = np.sum(
            (measured - source_points) * point_directions,
            axis=1,
        )
        regressions = np.where(
            moving_points,
            np.maximum(0.0, progress_high_water - progress),
            0.0,
        )
        max_progress_regression[:] = np.maximum(
            max_progress_regression,
            regressions,
        )
        progress_high_water[:] = np.maximum(progress_high_water, progress)
        report.update(
            {
                "max_anchor_error_m": maximum_anchor_error,
                "point_anchor_error_m": {
                    name: float(errors[index])
                    for index, name in enumerate(ordered_controlled_names)
                },
                "point_progress_m": {
                    name: float(progress[index])
                    for index, name in enumerate(ordered_controlled_names)
                },
                "point_progress_regression_m": {
                    name: float(regressions[index])
                    for index, name in enumerate(ordered_controlled_names)
                },
                "max_progress_regression_m": float(
                    np.max(max_progress_regression)
                ),
                "rigid_pair_consistency": pair_consistency,
                "progress_regression_limit_m": (
                    live_progress_regression_limit
                ),
                "off_hand_reference": reference_drift_report,
                "intermediate_anchor_disturbance": anchor_disturbed,
                "intermediate_rigid_geometry_disturbance": (
                    rigid_geometry_disturbed
                ),
                "trajectory_continues_from_immutable_plan": True,
                "terminal_revalidation_required": bool(
                    anchor_disturbed or rigid_geometry_disturbed
                ),
            }
        )
        if pause_for_visual_reacquisition and rigid_geometry_disturbed:
            tracker_reason = "tracked_point_rigid_geometry_inconsistent"
            tracking_continuity["temporarily_unobserved_steps"] += 1
            tracking_continuity["current_unobserved_streak"] += 1
            tracking_continuity["max_unobserved_streak"] = max(
                int(tracking_continuity["max_unobserved_streak"]),
                int(tracking_continuity["current_unobserved_streak"]),
            )
            tracking_continuity["last_unobserved_reason"] = tracker_reason
            report.update(
                {
                    "tracker_reason": tracker_reason,
                    "temporarily_unobserved": True,
                    "intermediate_visual_measurement_unavailable": False,
                    "intermediate_visual_geometry_inconsistent": True,
                    "trajectory_continues_from_immutable_plan": False,
                    "trajectory_paused_for_reacquisition": True,
                    "final_reacquisition_required": True,
                }
            )
            # Direct RGB-D coordinates that break the frozen point distances
            # cannot all retain their marker identities. Hold measured joints
            # and wait for a coherent fresh bundle before advancing.
            return None, report
        if bool(reference_drift_report.get("tracking_outlier_suppressed")):
            report.update(
                {
                    "off_hand_reference_substitution_used": True,
                    "trajectory_continues_from_immutable_plan": True,
                    "temporarily_unobserved": False,
                }
            )
        # A visual/anchor disturbance does not invalidate the signed arm path.
        # Keep moving toward the planned endpoint and let the synchronized
        # terminal verifier either confirm the public goal or perform its
        # bounded visual replan.  Stopping here made a transient tug or noisy
        # RGB-D sample permanently consume an otherwise valid invocation.
        if quick_mode:
            quick_report = _move_tracked_evaluate_quick_constraints(
                measured,
                reference_for_constraints,
                quick_specs,
                controlled_names=ordered_controlled_names,
                reference_names=ordered_reference_names,
                position_tolerance_m=float(args["pos_tol"]),
                orientation_tolerance_deg=float(args["ori_tol_deg"]),
                camera_position_robot_base_m=head_camera_position_robot_base_m,
                camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            )
            report["quick_constraint"] = quick_report
            # The preset describes the *terminal* orientation.  A Cartesian
            # or joint-space path is expected to violate that orientation at
            # every intermediate fraction while the held object rotates into
            # alignment.  Treating the transient angle as a hard tracking
            # failure stops a valid path on its first waypoint (and was the
            # reason the original 2+3 smoke case aborted at ~87 degrees).
            # Keep the live report for diagnostics; the final stability gate
            # and post-execution snapshot remain authoritative for success.
        # A rigid point offset from the EEF traces an arc during SLERP, so its
        # projection onto the endpoint chord may briefly decrease on a fully
        # one-way trajectory.  Execution monotonicity is guaranteed by the
        # immutable global path parameter (one action per waypoint); live point
        # observations remain mandatory here for rigid-anchor and final target
        # validation, while chord regression is diagnostic only.
        return None, report

    # The signed plan is intentionally dense enough for geometric validation.
    # Before the first motion action, derive an execution-only schedule from
    # the *current* handoff qpos and the observed evaluator period.  This
    # avoids discovering an impossible wall-clock budget halfway through a
    # valid trajectory (the old behavior that produced a timeout after a
    # deterministic prefix of the path).
    execution_schedule_report: dict[str, Any] = {
        "ok": True,
        "mode": "legacy_identity",
        "reason": "schedule_not_needed",
        "input_waypoint_count": int(len(trajectory.get("waypoints", []))),
        "selected_waypoint_count": int(len(trajectory.get("waypoints", []))),
        "dropped_waypoint_count": 0,
        "waypoints": trajectory.get("waypoints", []),
    }
    execution_started = time.monotonic()
    execution_timeout_s = float(args["timeout_s"]) - (
        execution_started - planning_started
    )
    if execution_timeout_s <= 0.0:
        _move_tracked_failure(
            ctx,
            stage="timeout",
            error="move_tracked_point timed out before trajectory execution",
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": trajectory["integrity"]["digest"],
                "planning_elapsed_s": float(
                    execution_started - planning_started
                ),
                "motion": {"ok": False, "reason": "timeout"},
            },
        )
        yield ctx.world.hold_action()
        return
    execution_max_steps = max(
        0,
        int(args["max_steps"]) - MOVE_POINT_TO_POINT_STABLE_STEPS,
    )
    execution_cadence = _move_tracked_execution_cadence_estimate(
        start_validation=start_validation,
        preplanning_source_sync=preplanning_source_sync,
        planner=trajectory.get("planner"),
    )
    raw_observation_dt = execution_cadence.get("observation_dt_s")
    observation_dt_s = (
        float(raw_observation_dt)
        if raw_observation_dt is not None
        else None
    )
    # Reserve the final stable observations here.  The pure scheduler adds
    # the two-period uncertainty margin of its own.  The executor still
    # receives the full remaining timeout; this is a preflight guard, not a
    # shortened runtime deadline.
    schedule_available_time_s = float(execution_timeout_s)
    if observation_dt_s is not None:
        schedule_available_time_s -= float(
            MOVE_POINT_TO_POINT_STABLE_STEPS
            * observation_dt_s
        )
    try:
        if schedule_available_time_s <= 0.0:
            execution_schedule_report = {
                "ok": False,
                "reason": "execution_budget_infeasible",
                "error": "no evaluator time remains after the final-stability reserve",
            }
        else:
            current_execution_observation = _adjust_kinematic_observation(ctx, arm)
            execution_schedule_report = _move_tracked_prepare_execution_schedule(
                trajectory=trajectory,
                state=capture_state["state"],
                arm=arm,
                start_q=current_execution_observation["q"],
                max_steps=execution_max_steps,
                available_time_s=schedule_available_time_s,
                observation_dt_s=observation_dt_s,
                visual_tracked_point_count=len(visual_controlled_names),
            )
            execution_schedule_report["cadence_estimate"] = _json_ready(
                execution_cadence
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        execution_schedule_report = {
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": str(exc),
        }
    if not bool(execution_schedule_report.get("ok")):
        schedule_reason = str(
            execution_schedule_report.get("reason") or "execution_schedule_invalid"
        )
        _move_tracked_failure(
            ctx,
            stage="timeout" if schedule_reason == "execution_budget_infeasible" else "planning",
            error=str(
                execution_schedule_report.get("error")
                or "could not construct a bounded execution schedule"
            ),
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": trajectory["integrity"]["digest"],
                "execution_schedule": _json_ready(execution_schedule_report),
                "planning_elapsed_s": float(execution_started - planning_started),
            },
        )
        yield ctx.world.hold_action()
        return
    scheduled_waypoints = (
        None
        if execution_schedule_report.get("mode") == "legacy_identity"
        else list(execution_schedule_report.get("waypoints", []))
    )
    scheduled_max_step = (
        None
        if scheduled_waypoints is None
        else float(
            execution_schedule_report.get(
                "execution_max_joint_step_rad",
                max(
                    float(
                        execution_schedule_report.get(
                            "max_joint_step_rad",
                            MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD,
                        )
                    ),
                    float(
                        execution_schedule_report.get(
                            "actual_max_joint_step_rad",
                            0.0,
                        )
                    ),
                ),
            )
        )
    )
    try:
        motion_generator = _move_point_execute_trajectory(
            ctx,
            trajectory=trajectory,
            capture_state=capture_state,
            adapter=start_adapter,
            observation_sequence=start_sequence,
            timeout_s=execution_timeout_s,
            max_steps=args["max_steps"] - MOVE_POINT_TO_POINT_STABLE_STEPS,
            step_monitor=step_monitor,
            operation_name="move_tracked_point",
            allow_start_rebase=True,
            allow_gripper_recovery=True,
            continue_during_gripper_disturbance=True,
            recover_active_arm_disturbance=True,
            execution_waypoints=scheduled_waypoints,
            execution_max_step_rad=scheduled_max_step,
            tracker_reacquisition_max_steps=(
                MOVE_TRACKED_POINT_NPOINT_REACQUISITION_MAX_HOLD_STEPS
                if len(ordered_controlled_names)
                >= MOVE_TRACKED_POINT_NPOINT_VISUAL_MIN_POINTS
                else None
            ),
        )
        while True:
            try:
                yield next(motion_generator)
            except StopIteration as stopped:
                motion = stopped.value
                break
        motion["tracking_continuity"] = _json_ready(tracking_continuity)
        motion["execution_schedule"] = _json_ready(execution_schedule_report)
        remaining_timeout_s = execution_timeout_s - (
            time.monotonic() - execution_started
        )
        remaining_action_steps = max(
            0,
            int(args["max_steps"]) - int(motion.get("steps_executed", 0)),
        )
        if not bool(motion.get("ok")):
            stability = {
                "ok": False,
                "reason": "trajectory_not_complete",
                "error": (
                    "trajectory execution failed before final stability: "
                    f"{motion.get('reason') or 'tracking_failed'}"
                ),
                "stable_steps": 0,
                "verification_steps": 0,
            }
        elif remaining_timeout_s <= 0.0:
            stability = {
                "ok": False,
                "reason": "timeout",
                "error": "move_tracked_point timed out before final stability",
                "stable_steps": 0,
                "verification_steps": 0,
            }
        elif remaining_action_steps < MOVE_POINT_TO_POINT_STABLE_STEPS:
            stability = {
                "ok": False,
                "reason": "max_steps",
                "error": "no action budget remains for final stability",
                "stable_steps": 0,
                "verification_steps": 0,
            }
        else:
            target_quick_groups = trajectory["target"].get("quick_constraints")
            if target_quick_groups is None:
                target_quick = trajectory["target"].get("quick_constraint")
                target_quick_groups = (
                    [target_quick]
                    if isinstance(target_quick, Mapping)
                    else []
                )
            target_faceto_camera = trajectory["target"].get(
                "faceto_camera_pose"
            )
            target_camera_position = (
                target_faceto_camera.get("position_robot_base_m")
                if isinstance(target_faceto_camera, Mapping)
                else None
            )
            target_camera_quaternion = (
                target_faceto_camera.get("quaternion_xyzw")
                if isinstance(target_faceto_camera, Mapping)
                else None
            )
            stability_generator = _move_tracked_verify_stable_target(
                ctx,
                arm=arm,
                adapter=start_adapter,
                observation_sequence=int(motion["last_observation_sequence"]),
                manager=manager,
                point_names=ordered_controlled_names,
                target_points=list(
                    trajectory["target"].get("planning_target_points") or []
                ),
                fixed_point_names=ordered_reference_names,
                fixed_target_points=list(
                    trajectory["target"].get("off_hand_target_points") or []
                ),
                relations=list(trajectory["target"].get("relations") or []),
                inequalities=list(
                    trajectory["target"].get("inequalities") or []
                ),
                quick_constraints=(
                    list(target_quick_groups)
                    if target_quick_groups
                    else None
                ),
                camera_position_robot_base_m=target_camera_position,
                camera_quaternion_xyzw=target_camera_quaternion,
                anchors_eef=anchors_eef,
                anchor_tolerance_m=anchor_tolerance,
                pair_distance_tolerance_m=pair_distance_tolerance,
                target_position=np.asarray(
                    trajectory["target"]["eef_position_robot_base_m"],
                    dtype=np.float64,
                ),
                target_quaternion=np.asarray(
                    trajectory["target"]["eef_quaternion_xyzw"],
                    dtype=np.float64,
                ),
                target_q=np.asarray(
                    trajectory["waypoints"][-1], dtype=np.float64
                ),
                pos_tol_m=args["pos_tol"],
                ori_tol_deg=args["ori_tol_deg"],
                timeout_s=remaining_timeout_s,
                max_steps=remaining_action_steps,
                expected_episode_id=episode_id,
                fixed_reference_points=reference_points,
            )
            while True:
                try:
                    yield next(stability_generator)
                except StopIteration as stopped:
                    stability = stopped.value
                    break
    except Exception as exc:
        if not _move_point_is_cancellation(exc):
            raise
        observed = _adjust_kinematic_observation(ctx, arm)
        cancellation_hold_q, cancellation_limit_normalizations = (
            _normalize_arm_q_to_joint_limits(
                observed["q"],
                arm=arm,
                label=f"{arm} move-tracked cancellation hold q",
                point="cancellation_hold",
                recover_observed_overshoot=True,
            )
        )
        cancellation_hold_q = _require_serialized_j8_zero(
            cancellation_hold_q,
            label=f"{arm} move-tracked cancellation hold q",
        )
        maximum_cancellation_correction = max(
            (
                abs(float(item["correction_rad"]))
                for item in cancellation_limit_normalizations
            ),
            default=0.0,
        )
        if maximum_cancellation_correction > MOVE_TRACKED_POINT_MAX_JOINT_STEP_RAD:
            raise RuntimeError(
                "move-tracked cancellation observation exceeds the "
                "recoverable joint-limit residual"
        )
        ctx.world.set_arm_pin_qpos(arm, cancellation_hold_q)
        _move_tracked_failure(
            ctx,
            stage="cancellation",
            error=str(exc) or "move_tracked_point was cancelled",
            args=args,
            details={
                "arm": arm,
                "plan_id": plan_id,
                "trajectory_digest": trajectory["integrity"]["digest"],
                "cancellation_hold_source": "subsequent_proprioception_measured_arm_qpos",
                "cancellation_hold_q": cancellation_hold_q.astype(float).tolist(),
                "cancellation_joint_limit_normalizations": _json_ready(
                    cancellation_limit_normalizations
                ),
            },
        )
        cancellation_overrides = {
            f"arm_{arm}": cancellation_hold_q.astype(float).tolist()
        }
        yield ctx.world.make_action(**cancellation_overrides)
        return
    final_observed = _adjust_kinematic_observation(ctx, arm)
    final_live = _move_tracked_live_snapshot(
        manager, ordered_names, episode_id=episode_id
    )
    final_sequence, _received_ts = start_adapter.observation_metadata()
    final_observation_bundle = _move_tracked_final_observation_bundle(
        final_live,
        controlled_names=ordered_controlled_names,
        fixed_names=ordered_reference_names,
        current_observation=final_observed,
        anchors_eef=anchors_eef,
        current_sequence=int(final_sequence),
        anchor_tolerance_m=anchor_tolerance,
        pos_tol_m=float(args["pos_tol"]),
        adapter=start_adapter,
    )
    final_tracking_ok = bool(final_observation_bundle.get("ok"))
    final_reference_points_for_constraints = None
    if final_tracking_ok:
        final_points = np.asarray(
            final_observation_bundle["points"], dtype=np.float64
        )
        final_reference_points = (
            np.asarray(
                final_observation_bundle["fixed_points"], dtype=np.float64
            )
            if ordered_reference_names
            else None
        )
        final_all_points = np.vstack(
            [
                final_points,
                final_reference_points
                if final_reference_points is not None
                else np.empty((0, 3), dtype=np.float64),
            ]
        )
        predicted_points = np.asarray(
            final_observation_bundle["predicted_points"], dtype=np.float64
        )
        final_anchor_errors = np.linalg.norm(final_points - predicted_points, axis=1)
        final_anchor_ok = bool(float(np.max(final_anchor_errors)) <= anchor_tolerance)
        final_pair_consistency = _tracked_rigid_pair_consistency(
            final_points,
            source_points,
            tolerance_m=pair_distance_tolerance,
        )
        try:
            if ordered_reference_names:
                reference_limit = (
                    MOVE_TRACKED_POINT_OFF_HAND_REFERENCE_TOLERANCE_M
                )
                final_reference_validation = (
                    _move_tracked_off_hand_reference_consensus(
                        ordered_reference_names,
                        final_reference_points,
                        reference_points,
                        tolerance_m=reference_limit,
                        allow_visual_outlier_suppression=True,
                    )
                )
                # Validate the live scene references above, then evaluate the
                # terminal goal against the same frozen target used by
                # planning and stable-goal verification.
                final_reference_points_for_constraints = reference_points
            else:
                final_reference_validation = {"ok": True, "applicable": False}
                final_reference_points_for_constraints = None
            base_constraints = _tracked_target_constraints(
                final_points,
                list(trajectory["target"].get("planning_target_points") or []),
                tolerance_m=args["pos_tol"],
                relations=list(trajectory["target"].get("relations") or []),
                inequalities=list(
                    trajectory["target"].get("inequalities") or []
                ),
                orientation_tolerance_deg=args["ori_tol_deg"],
                fixed_points_robot_base_m=final_reference_points_for_constraints,
                fixed_target_points=(
                    list(trajectory["target"].get("off_hand_target_points") or [])
                    if ordered_reference_names
                    else None
                ),
            )
            quick_report = (
                _move_tracked_evaluate_quick_constraints(
                    final_points,
                    final_reference_points_for_constraints,
                    quick_specs,
                    controlled_names=ordered_controlled_names,
                    reference_names=ordered_reference_names,
                    position_tolerance_m=float(args["pos_tol"]),
                    orientation_tolerance_deg=float(args["ori_tol_deg"]),
                    camera_position_robot_base_m=head_camera_position_robot_base_m,
                    camera_quaternion_xyzw=head_camera_quaternion_xyzw,
                )
                if quick_mode
                else {"ok": True, "applicable": False}
            )
            final_constraints = {
                **base_constraints,
                "ok": bool(
                    base_constraints.get("ok")
                    and quick_report.get("ok")
                    and final_reference_validation.get("ok")
                ),
                "quick_constraint": quick_report,
                "off_hand_reference": final_reference_validation,
                "error": (
                    None
                    if base_constraints.get("ok")
                    and quick_report.get("ok")
                    and final_reference_validation.get("ok")
                    else str(
                        quick_report.get("error")
                        or (
                            "off-hand reference points moved"
                            if not final_reference_validation.get("ok")
                            else "combined coordinate and geometry constraints failed"
                        )
                    )
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            final_tracking_ok = False
            final_reference_validation = {
                "ok": False,
                "applicable": bool(ordered_reference_names),
                "reason": f"tracked_constraint_validation_failed: {exc}",
            }
            final_constraints = {
                "ok": False,
                "error": f"tracked_constraint_validation_failed: {exc}",
                "relations": [],
                "quick_constraint": {
                    "ok": False,
                    "error": f"tracked_constraint_validation_failed: {exc}",
                },
                "off_hand_reference": final_reference_validation,
            }
    else:
        final_all_points = np.empty((0, 3), dtype=np.float64)
        final_reference_validation = {
            "ok": False if ordered_reference_names else True,
            "applicable": bool(ordered_reference_names),
            "reason": (
                "final_tracker_observation_unavailable"
                if ordered_reference_names
                else None
            ),
        }
        final_points = np.empty((0, 3), dtype=np.float64)
        final_anchor_errors = np.full(
            len(ordered_controlled_names), float("inf"), dtype=np.float64
        )
        final_anchor_ok = False
        final_pair_consistency = {
            "applicable": len(ordered_controlled_names) >= 2,
            "ok": False,
            "tolerance_m": pair_distance_tolerance,
            "error": str(
                final_observation_bundle.get("reason")
                or final_live.get("reason")
                or "tracker_observation_mismatch"
            ),
        }
        final_constraints = {
            "ok": False,
            "error": str(
                final_observation_bundle.get("reason")
                or final_live.get("reason")
                or "tracker_observation_mismatch"
            ),
        }
    final_live_points_report = (
        {
            name: final_all_points[index].astype(float).tolist()
            for index, name in enumerate(ordered_names)
        }
        if final_tracking_ok
        else {}
    )
    per_point_constraint_errors: dict[str, float] = {name: 0.0 for name in ordered_names}
    for item in list(final_constraints.get("fixed_coordinates", [])) + list(
        final_constraints.get("affine_constraints", [])
    ):
        point_name = str(item.get("point", ""))
        raw_error = item.get("error_m", item.get("residual_m", 0.0))
        try:
            error_value = abs(float(raw_error))
        except (TypeError, ValueError):
            error_value = float("inf")
        if point_name in per_point_constraint_errors:
            per_point_constraint_errors[point_name] = max(
                per_point_constraint_errors[point_name], error_value
            )
    per_relation_errors = list(final_constraints.get("relations", []))
    per_inequality_errors = list(final_constraints.get("inequalities", []))
    max_orientation_error_deg = float(
        final_constraints.get("max_relation_error_deg", 0.0)
    )
    last_goal_report = stability.get("last_live_goal_report", {})
    if isinstance(last_goal_report, dict):
        try:
            max_orientation_error_deg = max(
                max_orientation_error_deg,
                float(last_goal_report.get("orientation_error_deg", 0.0)),
            )
        except (TypeError, ValueError):
            pass
    max_anchor_error = float(np.max(final_anchor_errors)) if final_anchor_errors.size else float("inf")
    pair_distance_drift = float(
        final_pair_consistency.get(
            "max_pair_distance_error_m",
            final_pair_consistency.get("distance_error_m", float("inf")),
        )
    )
    post_execution_track_validation = {
        "ok": bool(
            final_tracking_ok
            and final_constraints.get("ok")
            and final_pair_consistency.get("ok")
            and final_reference_validation.get("ok", True)
        ),
        "required_for_success": True,
        "source": (
            "subsequent_evaluator_rgbd_and_proprio_local_fk_for_explicit_occlusion"
            if final_observation_bundle.get("fk_completed_controlled_names")
            else "subsequent_evaluator_observation"
        ),
        "episode_id": episode_id,
        "observation_sequence": int(final_sequence),
        "tracker_observation_sequence": final_live.get("observation_sequence"),
        "synchronized_with_final_proprioception": final_tracking_ok,
        "input_constraints": args["points"],
        "points_robot_base_m": (
            final_live_points_report
        ),
        "point_sources": _json_ready(
            final_observation_bundle.get("point_sources", {})
        ),
        "measured_on_hand_point_names": list(
            final_observation_bundle.get("measured_controlled_names", [])
        ),
        "fk_completed_occluded_on_hand_point_names": list(
            final_observation_bundle.get("fk_completed_controlled_names", [])
        ),
        "on_hand_observation_completion": _json_ready(
            final_observation_bundle
        ),
        "prediction_values_published_as_observation": False,
        "off_hand_reference": _json_ready(final_reference_validation),
        "off_hand_constraint_reference_points_robot_base_m": (
            {
                name: final_reference_points_for_constraints[index]
                .astype(float)
                .tolist()
                for index, name in enumerate(ordered_reference_names)
            }
            if final_tracking_ok
            and ordered_reference_names
            and final_reference_points_for_constraints is not None
            else {}
        ),
        "tolerance_m": float(args["pos_tol"]),
        "max_constraint_error_m": final_constraints.get(
            "max_constraint_error_m"
        ),
        "constraint_report": final_constraints,
        "rigid_pair_consistency": final_pair_consistency,
        "per_point_constraint_errors": per_point_constraint_errors,
        "per_relation_errors": per_relation_errors,
        "per_inequality_errors": per_inequality_errors,
        "max_anchor_error_m": max_anchor_error,
        "pair_distance_drift_m": pair_distance_drift,
        "max_kabsch_bundle_residual_m": float(
            final_pair_consistency.get("max_kabsch_bundle_residual_m", float("inf"))
        ),
    }
    overall_ok = bool(
        motion.get("ok")
        and stability.get("ok")
        and post_execution_track_validation["ok"]
        and final_anchor_ok
    )
    motion_reason = str(motion.get("reason") or "tracking_failed")
    stability_reason = str(stability.get("reason") or "tracking_failed")
    failure_stage = None
    result_error = None
    if not overall_ok:
        failure_stage = (
            "timeout"
            if "timeout" in {motion_reason, stability_reason}
            else "tracking"
        )
        if not bool(motion.get("ok")):
            result_error = f"trajectory execution failed: {motion_reason}"
        elif not bool(stability.get("ok")):
            result_error = str(
                stability.get("error")
                or f"final stability failed: {stability_reason}"
            )
        elif not bool(post_execution_track_validation.get("ok")):
            if ordered_reference_names and not bool(
                final_reference_validation.get("ok")
            ):
                result_error = str(
                    final_reference_validation.get("error")
                    or "off-hand reference points moved during execution"
                )
            elif not bool(final_pair_consistency.get("ok")):
                result_error = (
                    "final tracked rigid geometry changed: "
                    f"max_pair_error={float(final_pair_consistency.get('max_pair_distance_error_m', final_pair_consistency.get('distance_error_m', float('inf')))):.4f}m "
                    f"max_kabsch_bundle_error={float(final_pair_consistency.get('max_kabsch_bundle_residual_m', float('inf'))):.4f}m"
                )
            else:
                result_error = str(
                    final_constraints.get("error")
                    or "final live tracked-point constraints were not satisfied"
                )
        else:
            result_error = "final tracked points were not rigid with the selected EEF"
    if overall_ok:
        if len(visual_controlled_names) == 2:
            rigid_pair_lease_state["preserve_after_success"] = True
        motion_retention_release = (
            _move_tracked_preserve_motion_retention_for_successor(
                motion_retention_lease_state
            )
        )
    else:
        motion_retention_release = _move_tracked_release_motion_retention(
            manager,
            motion_retention_lease_state,
        )
    _set_result(
        ctx,
        "move_tracked_point",
        {
            **_move_tracked_common_result(),
            "ok": overall_ok,
            "error": result_error,
            "failure_stage": failure_stage,
            "build": MOVE_TRACKED_POINT_LOCAL_BUILD,
            "execution_mode": "exec",
            "session_id": session_id,
            "image_id": image_id,
            "plan_id": plan_id,
            "plan_record_path": plan_path,
            "trajectory_schema": trajectory["schema"],
            "trajectory_schema_version": trajectory["schema_version"],
            "trajectory_digest": trajectory["integrity"]["digest"],
            "trajectory_waypoint_count": len(trajectory["waypoints"]),
            "execution_waypoint_count": int(
                execution_schedule_report.get(
                    "selected_waypoint_count",
                    len(trajectory["waypoints"]),
                )
            ),
            "execution_schedule": _json_ready(execution_schedule_report),
            "robot_profile": ROBOT_PROFILE,
            "arm": arm,
            "eef_mean_distance_m": distances,
            "input_point_order": names,
            "internal_point_order": ordered_names,
            "point_count": int(len(ordered_names)),
            "point_a_selection": (
                "nearest_to_selected_eef" if len(ordered_names) >= 2 else "single_point"
            ),
            "nearest_point_name": (
                nearest_point_name if ordered_names else None
            ),
            "all_points_same_rigid_object_required": (
                (not quick_mode) and len(ordered_controlled_names) >= 2
            ),
            # Legacy result key retained for clients that only understand the
            # original pair contract; it is informational for N > 2.
            "two_points_same_rigid_object_required": (
                (not quick_mode) and len(ordered_controlled_names) == 2
            ),
            "quick_constraint": _json_ready(quick_spec) if quick_mode else None,
            "quick_constraints": _json_ready(quick_specs) if quick_mode else [],
            "inequalities": _json_ready(
                trajectory["target"].get("inequalities") or []
            ),
            "on_hand_point_names": list(ordered_controlled_names),
            "off_hand_point_names": list(ordered_reference_names),
            "predefined_eef_point_names": list(predefined_names),
            "visual_tracked_point_names": list(visual_names),
            "predefined_eef_arm_selection": _json_ready(
                predefined_arm_selection
            ),
            "off_hand_start_validation": _json_ready(off_hand_start_validation),
            "off_hand_final_validation": _json_ready(final_reference_validation),
            "rigid_pair_activation": _json_ready(rigid_pair_activation),
            "rigid_pair_activation_confirmation": _json_ready(
                rigid_pair_activation_confirmation
            ),
            "rigid_pair_activation_hold_steps": int(
                rigid_pair_activation_hold_steps
            ),
            "prior_rigid_pair_cleanup": _json_ready(
                prior_rigid_pair_cleanup
            ),
            "failed_rigid_pair_cleanup": _json_ready(
                failed_rigid_pair_cleanup
            ),
            "rigid_pair_bootstrap_eef_observation_prior": _json_ready(
                rigid_pair_bootstrap_eef_prior
            ),
            "on_hand_eef_observation_prior": _json_ready(
                on_hand_eef_prior
            ),
            "preplanning_on_hand_eef_observation_prior": _json_ready(
                preplanning_on_hand_eef_prior
            ),
            "preplanning_source_sync": _json_ready(
                preplanning_source_sync
            ),
            "post_planning_source_replans": _json_ready(
                post_planning_source_replans
            ),
            "post_planning_source_replan_count": int(
                len(post_planning_source_replans)
            ),
            "frozen_observation": {
                "episode_id": episode_id,
                "motion_epoch": capture_state["motion_epoch"],
                "observation_sequence": observation_sequence,
                "initial_observation_sequence": initial_observation_sequence,
                "tracker_and_proprio_same_observation": True,
                "post_eef_prior_source_synchronized": bool(
                    preplanning_source_sync.get("planning_source_rebased", False)
                ),
            },
            "source_points_robot_base_m": {
                name: source_by_name[name].astype(float).tolist()
                for name in names
            },
            "resolved_variables": trajectory["target"]["constraints"].get(
                "resolved_variables", {}
            ),
            "target_constraints": trajectory["target"]["constraints"],
            "endpoint_solver": trajectory["planner"]["endpoint_solver"],
            "planner": trajectory["planner"],
            "motion": motion,
            "tracking_continuity": _json_ready(tracking_continuity),
            "motion_tracking_retention": {
                "lease": _json_ready(motion_retention),
                "release": _json_ready(motion_retention_release),
            },
            "stability": stability,
            "final_live_tracking": _json_ready(final_live),
            "final_on_hand_observation_completion": _json_ready(
                final_observation_bundle
            ),
            "final_point_sources": _json_ready(
                final_observation_bundle.get("point_sources", {})
            ),
            "final_constraints": _json_ready(final_constraints),
            "final_rigid_pair_consistency": _json_ready(
                final_pair_consistency
            ),
            "post_execution_track_validation": _json_ready(
                post_execution_track_validation
            ),
            "final_anchor_ok": final_anchor_ok,
            "final_anchor_error_m": {
                name: float(final_anchor_errors[index])
                for index, name in enumerate(ordered_controlled_names)
            },
            "final_anchor_tolerance_m": anchor_tolerance,
            "final_live_points_robot_base_m": final_live_points_report,
            "resolved_target_points_robot_base_m": trajectory["target"][
                "resolved_points_robot_base_m"
            ],
            "per_point_constraint_errors": per_point_constraint_errors,
            "per_relation_errors": per_relation_errors,
            "per_inequality_errors": per_inequality_errors,
            "max_inequality_violation_m": final_constraints.get(
                "max_inequality_violation_m", float("inf")
            ),
            "minimum_inequality_clearance_m": final_constraints.get(
                "minimum_inequality_clearance_m"
            ),
            "max_constraint_error_m": final_constraints.get(
                "max_constraint_error_m", float("inf")
            ),
            "max_orientation_error_deg": max_orientation_error_deg,
            "max_anchor_error_m": max_anchor_error,
            "pair_distance_drift_m": pair_distance_drift,
            "max_kabsch_bundle_residual_m": float(
                final_pair_consistency.get(
                    "max_kabsch_bundle_residual_m", float("inf")
                )
            ),
            "geometry_rank": final_pair_consistency.get("geometry_rank"),
            "start_state_validation": {
                key: value
                for key, value in start_validation.items()
                if key != "adapter"
            },
            "collision_checked": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
