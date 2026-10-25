from __future__ import annotations

import math
import numpy as np
from tools.util.capabilities import PUBLIC_TOOLS


GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP = 0.12

GRASP_PREP_DEFAULT_TIMEOUT_S = 45.0

GRASP_PREP_MAX_DQ_PER_STEP = 0.25

GRASP_PREP_MAX_ACTION_STEPS = 96

GRASP_PREP_STALL_LIMIT = 10

GRASP_PREP_PROGRESS_EPS_RAD = 5.0e-4

GRASP_PREP_MIN_OBSERVED_MOTION_RAD = 1.0e-4

GRASP_PREP_RETRACE_ANCHOR_TOLERANCE_RAD = 0.12

GRASP_PREP_RETRACE_WAYPOINT_TOLERANCE_RAD = 0.055

GRASP_PREP_RETRACE_MAX_WAYPOINTS = 56

GRASP_PREP_CAPTURE_BUILD = "x5_neutral_public_feedback_stationary_exit_v1"

GRIPPER_POSITION_TOLERANCE = 0.15

GRIPPER_OPEN_POSITION_TOLERANCE = 0.005

GRIPPER_OPEN_CONTROL_HZ = 25.0

GRIPPER_OPEN_CONFIRMATION_STEPS = 10

GRIPPER_OPEN_SETTLE_STEP = 0.003

GRIPPER_OPEN_STALL_STEPS = 10

GRIPPER_OPEN_MAX_ACTION_STEPS = 250

GRIPPER_OPEN_DEPTH_TOLERANCE_M = 0.002

GRIPPER_OPEN_DEPTH_CONFIRMATION_SAMPLES = 2

GRIPPER_OPEN_DEPTH_RECHECK_STEPS = 4

GRIPPER_CLOSE_CONTROL_HZ = 25.0

GRIPPER_CLOSE_CONFIRMATION_S = 1.00

GRIPPER_CLOSE_CONFIRMATION_STEPS = int(
    math.ceil(GRIPPER_CLOSE_CONTROL_HZ * GRIPPER_CLOSE_CONFIRMATION_S)
)

GRIPPER_CLOSE_SETTLE_WINDOW_STEPS = 4

GRIPPER_CLOSE_SETTLE_SPAN = 0.003

GRIPPER_CLOSE_BLOCKAGE_MAX_POSITION = 0.75

GRIPPER_CLOSE_REBOUND_TOLERANCE = 0.02

GRIPPER_CLOSE_COLLAPSE_TOLERANCE = 0.01

GRIPPER_CLOSE_MAX_ACTION_STEPS = 96

GRIPPER_CLOSE_EVIDENCE_BUILD = "x5_close_stable_entry_progress_v2"

EEF_MAX_ACTION_STEPS = 32

EEF_MAX_JOINT_DELTA_PER_ACTION = 0.18

EEF_ADJUST_MAX_ACTION_STEPS = 240

EEF_ADJUST_MAX_CARTESIAN_STEP_M = 0.012

EEF_ADJUST_MAX_ORIENTATION_STEP_DEG = math.degrees(0.06)

EEF_ADJUST_MAX_JOINT_DELTA_PER_ACTION = 0.028

EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD = 0.20

EEF_JOINT_TRACKING_BLOCKED_RAD = 0.04

CONTROL_STALL_LIMIT = 4

EXEC_JOINT_SETTLE_TOLERANCE_RAD = 0.012

EXEC_JOINT_TRACKING_BLOCKED_RAD = 0.018

EXEC_MAX_ACTION_STEPS = 64

EXEC_TOTAL_ACTION_BUDGET = 50

EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET = 100_000

EXEC_MOVE_TRACKED_SCHEDULE_BUDGET = 40

EXEC_MOVE_TRACKED_PREFERRED_WAYPOINTS = 8

EXEC_STALL_LIMIT = 8

EXEC_MIN_PROGRESS_SCORE = 0.01

EXEC_MIN_OBSERVED_JOINT_STEP_RAD = 1.0e-4

EXEC_MIN_OBSERVED_TRANSLATION_STEP_M = 2.0e-5

EXEC_MIN_OBSERVED_ORIENTATION_STEP_DEG = 0.01

EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD = 0.002

EXEC_FINAL_STABLE_STEPS = 3

EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M = 0.001

EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG = 0.5

EXEC_LOAD_STABLE_STEPS = 3

EXEC_LOAD_MIN_RESIDUAL_RAD = 0.004

EXEC_LOAD_MAX_ROUNDS = 3

EXEC_LOAD_MAX_ROUND_RAD = 0.06

EXEC_LOAD_MAX_TOTAL_RAD = 0.12

EXEC_CLEARANCE_BRIDGE_CARTESIAN_STEP_M = 0.035

EXEC_CLEARANCE_BRIDGE_ORIENTATION_STEP_DEG = 10.0

EXEC_CLEARANCE_BRIDGE_JOINT_STEP_RAD = 0.16

EXEC_CLEARANCE_BRIDGE_LIFT_M = 0.10

EXEC_CLEARANCE_BRIDGE_INITIAL_RISE_M = 0.04

EXEC_CLEARANCE_BRIDGE_ROTATION_CLEARANCE_M = 0.05

EXEC_CLEARANCE_BRIDGE_Z_TOLERANCE_M = 0.005

EXEC_CLEARANCE_BRIDGE_MAX_WAYPOINTS = 40

X5_DEFAULT_REACH_M = 0.82

X5_SHOULDERS = {
    "left": np.array([-0.30, -0.45, 0.765], dtype=np.float64),
    "right": np.array([0.30, -0.45, 0.765], dtype=np.float64),
}

X5_GRASP_READY = {
    # RoboDojo initializes dual X5 at this forward-facing, collision-free pose.
    "left": np.zeros(6, dtype=np.float64),
    "right": np.zeros(6, dtype=np.float64),
}


TRACKING_CAMERA_ROLES = frozenset({"head", "left_wrist", "right_wrist"})

_COMPAT_PLAN_MODES = frozenset(
    {
        "grasp_point",
        "grasp_obj",
        "grasp_point_filter",
        "grasp_point_filter_rgbd_lite",
        "press",
        "press_point",
        "place",
        "push_left",
        "push_right",
        "push_forward",
        "push_backward",
        "push_up",
        "push_down",
        "open",
        "close",
    }
)

_COMPAT_PUSH_OPTICAL_DIRECTIONS = {
    "push_left": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "push_right": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "push_forward": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "push_backward": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "push_up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "push_down": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}

_EXIT_CAPTURE_HEAD = "head"

_EXIT_CAPTURE_LEFT_WRIST = "left_wrist"

_EXIT_CAPTURE_RIGHT_WRIST = "right_wrist"

_EXIT_CAPTURE_REQUEST_ARM_WRIST = "request_arm_wrist"

_EXIT_CAPTURE_RESULT_ARM_WRIST = "result_arm_wrist"

EXIT_CAPTURE_ROLE_POLICIES = {name: _EXIT_CAPTURE_HEAD for name in PUBLIC_TOOLS}
EXIT_CAPTURE_ROLE_POLICIES.update(
    {
        "capture_left_wrist_camera": _EXIT_CAPTURE_LEFT_WRIST,
        "capture_right_wrist_camera": _EXIT_CAPTURE_RIGHT_WRIST,
        "adjust_left_eef_pose_in_head_frame": _EXIT_CAPTURE_HEAD,
        "adjust_right_eef_pose_in_head_frame": _EXIT_CAPTURE_HEAD,
        "adjust_left_eef_pose_in_wrist_frame": _EXIT_CAPTURE_LEFT_WRIST,
        "adjust_right_eef_pose_in_wrist_frame": _EXIT_CAPTURE_RIGHT_WRIST,
        "open_gripper": _EXIT_CAPTURE_REQUEST_ARM_WRIST,
        "close_gripper": _EXIT_CAPTURE_REQUEST_ARM_WRIST,
        "exec_plan_pose": _EXIT_CAPTURE_RESULT_ARM_WRIST,
        "joint_command": _EXIT_CAPTURE_HEAD,
    }
)
if set(EXIT_CAPTURE_ROLE_POLICIES) != set(PUBLIC_TOOLS) | {"joint_command"}:
    raise RuntimeError("exit capture policy must cover every robot tool")

_EXIT_CAPTURE_PROMOTED_FIELDS = (
    "image_id",
    "rgb_main",
    "rgb_main_path",
    "rgb_overlay_path",
    "rgb_path",
    "image_width",
    "image_height",
    "camera",
    "robot",
    "base_path_overlay",
    "grasp_zone_overlay",
    "red_mask_path",
    "grasp_volume_three_views",
    "grasp_volume_three_views_path",
    "grasp_volume_three_views_metadata",
    "modalities",
    "track_object_distance_binding",
    "memory",
    "memory_text",
    "feed",
)
