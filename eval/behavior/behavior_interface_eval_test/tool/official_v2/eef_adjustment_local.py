"""Submission-local kinematics for observation-driven EEF adjustments.

The functions in this module consume only evaluator proprioception and camera
poses supplied by the caller.  They use the frozen R1Pro URDF shipped beside
this file; there is no simulator, scene, link, controller, or robot handle.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .contract import ARM_DOF
from .grasp_geometry_local import quat_to_mat_xyzw
from .grasp_kinematics_local import (
    LocalRobotState,
    arm_joint_limits,
    eef_pose,
)


MAX_TRANSLATION_STEP_M = 0.012
MAX_ORIENTATION_STEP_RAD = 0.06
MAX_JOINT_STEP_RAD = 0.012
ADJUST_POSE_MAX_JOINT_STEP_RAD = 0.028
MAX_JOINT_STEP_CHANGE_RAD = 0.010
WRIST_MAX_JOINT_STEP_RAD = 0.02
POSE_IK_MIN_FUNCTION_EVALUATIONS = 80
POSE_IK_FUNCTION_EVALUATIONS_PER_ITERATION = 2
POSE_IK_NUMERIC_TOLERANCE = 1e-6
DLS_LAMBDA = 0.05
ORIENTATION_WEIGHT_MAX_SCALE = 12.0
TARGET_IK_GUIDANCE_WEIGHT = 0.85
JOINT_REGULARIZATION_WEIGHTS = np.array(
    [1.00, 1.00, 0.90, 0.80, 0.60, 0.55, 0.55],
    dtype=np.float64,
)

_RY_PI_MAT = np.diag([-1.0, 1.0, -1.0])
_PALM_T_EEF = np.array([0.0, 0.0, -0.06], dtype=np.float64)
_RIGHT_REALSENSE_ORIGIN_GRIPPER = np.array(
    [0.05051, 0.0028934, 0.0051317],
    dtype=np.float64,
)
_GRIPPER_ROLL_AXIS_EEF = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class PoseTarget:
    position: np.ndarray
    quaternion: np.ndarray
    delta_camera: np.ndarray
    delta_robot: np.ndarray


def locked_arm_vector(raw: Any, *, label: str) -> np.ndarray:
    """Validate an observed arm vector and remove J8 from local kinematics."""
    value = np.asarray(raw, dtype=np.float64).reshape(-1)
    if value.size != ARM_DOF:
        raise ValueError(f"{label} must have {ARM_DOF} values, got {value.size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains non-finite values")
    value = value.copy()
    if ARM_DOF == 8:
        value[7] = 0.0
    return value


def local_robot_state(
    *,
    trunk_q: Sequence[float],
    arm_left_q: Sequence[float],
    arm_right_q: Sequence[float],
    gripper_left_q: Sequence[float] = (0.0, 0.0),
    gripper_right_q: Sequence[float] = (0.0, 0.0),
) -> LocalRobotState:
    """Build a robot-base-frame state from evaluator proprioception."""
    return LocalRobotState.from_capture(
        {
            "base_pose": {
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "trunk_qpos": np.asarray(trunk_q, dtype=np.float64).reshape(4),
            "arm_left_qpos": locked_arm_vector(
                arm_left_q,
                label="left arm proprioception",
            ),
            "arm_right_qpos": locked_arm_vector(
                arm_right_q,
                label="right arm proprioception",
            ),
            "gripper_left_qpos": np.asarray(
                gripper_left_q,
                dtype=np.float64,
            ).reshape(-1)[:2],
            "gripper_right_qpos": np.asarray(
                gripper_right_q,
                dtype=np.float64,
            ).reshape(-1)[:2],
        },
        arm_dof=ARM_DOF,
    )


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion is invalid")
    return value / norm


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    lx, ly, lz, lw = normalize_quaternion(left)
    rx, ry, rz, rw = normalize_quaternion(right)
    return normalize_quaternion(
        np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


def _axis_angle_quaternion(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or abs(float(angle_rad)) <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    vector /= norm
    half = 0.5 * float(angle_rad)
    return np.concatenate(
        [vector * math.sin(half), np.array([math.cos(half)], dtype=np.float64)]
    )


def gripper_axes_in_eef() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen v2 roll, pitch, and yaw axes in the EEF frame."""
    roll_axis = _GRIPPER_ROLL_AXIS_EEF.copy()
    wrist_camera_eef = (
        _RY_PI_MAT @ _RIGHT_REALSENSE_ORIGIN_GRIPPER + _PALM_T_EEF
    )
    yaw_axis = wrist_camera_eef - roll_axis * float(
        wrist_camera_eef @ roll_axis
    )
    if float(np.linalg.norm(yaw_axis)) <= 1e-12:
        yaw_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    yaw_axis /= float(np.linalg.norm(yaw_axis))
    pitch_axis = np.cross(yaw_axis, roll_axis)
    pitch_axis /= max(float(np.linalg.norm(pitch_axis)), 1e-12)
    yaw_axis = np.cross(roll_axis, pitch_axis)
    yaw_axis /= max(float(np.linalg.norm(yaw_axis)), 1e-12)
    return roll_axis, pitch_axis, yaw_axis


def local_gripper_rpy_delta_quaternion(
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Reproduce the v2 local-gripper RPY convention."""
    roll_axis, pitch_axis, yaw_axis = gripper_axes_in_eef()
    q_roll = _axis_angle_quaternion(roll_axis, math.radians(float(roll_deg)))
    # Positive pitch is fingertips-up, opposite the right-hand rule here.
    q_pitch = _axis_angle_quaternion(
        pitch_axis,
        -math.radians(float(pitch_deg)),
    )
    q_yaw = _axis_angle_quaternion(yaw_axis, math.radians(float(yaw_deg)))
    return quaternion_multiply(quaternion_multiply(q_roll, q_pitch), q_yaw)


def camera_frame_pose_target(
    *,
    eef_position: Sequence[float],
    eef_quaternion: Sequence[float],
    camera_quaternion: Sequence[float],
    forward: float,
    leftward: float,
    upward: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> PoseTarget:
    """Apply v2 camera translation and local-gripper rotation semantics."""
    delta_camera = np.array(
        [-float(leftward), float(upward), -float(forward)],
        dtype=np.float64,
    )
    delta_robot = quat_to_mat_xyzw(
        normalize_quaternion(camera_quaternion)
    ) @ delta_camera
    delta_quaternion = local_gripper_rpy_delta_quaternion(
        roll_deg=float(roll),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
    )
    return PoseTarget(
        position=np.asarray(eef_position, dtype=np.float64).reshape(3)
        + delta_robot,
        quaternion=quaternion_multiply(eef_quaternion, delta_quaternion),
        delta_camera=delta_camera,
        delta_robot=delta_robot,
    )


def robot_base_frame_pose_target(
    *,
    eef_position: Sequence[float],
    eef_quaternion: Sequence[float],
    camera_quaternion: Sequence[float],
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> PoseTarget:
    """Apply robot-base XYZ translation and local-gripper rotation semantics.

    The robot-base axes match ``track_object_distance``: +X is chassis-forward,
    +Y is chassis-left, and +Z is chassis-up.  The camera quaternion is used
    only to report the equivalent camera-frame delta; it does not rotate the
    requested robot-base displacement.
    """

    delta_robot = np.array([float(x), float(y), float(z)], dtype=np.float64)
    camera_rotation_robot = quat_to_mat_xyzw(
        normalize_quaternion(camera_quaternion)
    )
    delta_camera = camera_rotation_robot.T @ delta_robot
    delta_quaternion = local_gripper_rpy_delta_quaternion(
        roll_deg=float(roll),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
    )
    return PoseTarget(
        position=np.asarray(eef_position, dtype=np.float64).reshape(3)
        + delta_robot,
        quaternion=quaternion_multiply(eef_quaternion, delta_quaternion),
        delta_camera=delta_camera,
        delta_robot=delta_robot,
    )


def rotation_vector(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle <= 1e-9:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) > 1e-7:
        return skew * (angle / (2.0 * sine))
    # Stable pi-angle branch.
    symmetric = 0.5 * (matrix + np.eye(3, dtype=np.float64))
    axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
    largest = int(np.argmax(axis))
    if axis[largest] > 1e-7:
        for index in range(3):
            if index != largest:
                axis[index] = symmetric[largest, index] / axis[largest]
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis * angle


def orientation_error_vector(
    current_quaternion: Sequence[float],
    target_quaternion: Sequence[float],
) -> np.ndarray:
    current = quat_to_mat_xyzw(normalize_quaternion(current_quaternion))
    target = quat_to_mat_xyzw(normalize_quaternion(target_quaternion))
    return rotation_vector(target @ current.T)


def orientation_error_deg(
    current_quaternion: Sequence[float],
    target_quaternion: Sequence[float],
) -> float:
    return math.degrees(
        float(
            np.linalg.norm(
                orientation_error_vector(current_quaternion, target_quaternion)
            )
        )
    )


def numerical_eef_jacobian(
    state: LocalRobotState,
    arm: str,
    q_arm: Sequence[float],
    *,
    joint_indices: Iterable[int] = range(7),
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a robot-base-frame geometric Jacobian from the frozen URDF."""
    indices = tuple(int(index) for index in joint_indices)
    q = locked_arm_vector(q_arm, label=f"{arm} arm qpos")
    position, quaternion = eef_pose(state, arm, q)
    rotation = quat_to_mat_xyzw(quaternion)
    jacobian = np.zeros((6, len(indices)), dtype=np.float64)
    for column, joint_index in enumerate(indices):
        probe = q.copy()
        probe[joint_index] += float(epsilon)
        probe_position, probe_quaternion = eef_pose(state, arm, probe)
        probe_rotation = quat_to_mat_xyzw(probe_quaternion)
        jacobian[:3, column] = (
            np.asarray(probe_position, dtype=np.float64) - position
        ) / float(epsilon)
        jacobian[3:, column] = rotation_vector(
            probe_rotation @ rotation.T
        ) / float(epsilon)
    return jacobian, position, quaternion


def _damped_step(jacobian: np.ndarray, task: np.ndarray, damping: float) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    rhs = np.asarray(task, dtype=np.float64).reshape(matrix.shape[0])
    lhs = matrix @ matrix.T + float(damping) ** 2 * np.eye(matrix.shape[0])
    try:
        return matrix.T @ np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def solve_pose_target(
    state: LocalRobotState,
    arm: str,
    q_seed: Sequence[float],
    *,
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
    joint_indices: Iterable[int] = range(7),
    pos_tol_m: float = 0.010,
    ori_tol_deg: float = 3.0,
    max_iterations: int = 96,
    check_requested: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a nearby pose using only the frozen submission model.

    ``check_requested`` is an optional cooperative cancellation/deadline hook
    used by long-running live planners.  It is invoked before the solve and
    before every residual evaluation; callers that omit it retain the legacy
    solver behavior.
    """
    from scipy.optimize import least_squares

    if callable(check_requested):
        check_requested()
    indices = tuple(int(index) for index in joint_indices)
    seed = locked_arm_vector(q_seed, label=f"{arm} IK seed")
    lower, upper = arm_joint_limits(state, arm)
    target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_quat = normalize_quaternion(target_quaternion)
    position_scale_m = max(1e-6, float(pos_tol_m))
    orientation_scale_rad = max(1e-6, math.radians(float(ori_tol_deg)))
    orientation_weight_m_per_rad = position_scale_m / orientation_scale_rad

    def expand(selected_values: Sequence[float]) -> np.ndarray:
        candidate = seed.copy()
        candidate[list(indices)] = np.asarray(
            selected_values,
            dtype=np.float64,
        ).reshape(len(indices))
        if ARM_DOF == 8:
            candidate[7] = 0.0
        return candidate

    def residual(selected_values: Sequence[float]) -> np.ndarray:
        if callable(check_requested):
            check_requested()
        candidate = expand(selected_values)
        position, quaternion = eef_pose(state, arm, candidate)
        return np.concatenate(
            [
                target_pos - position,
                orientation_weight_m_per_rad
                * orientation_error_vector(quaternion, target_quat),
            ]
        )

    selected_lower = lower[list(indices)]
    selected_upper = upper[list(indices)]
    selected_seed = np.clip(
        seed[list(indices)],
        selected_lower,
        selected_upper,
    )
    max_function_evaluations = max(
        POSE_IK_MIN_FUNCTION_EVALUATIONS,
        int(max_iterations) * POSE_IK_FUNCTION_EVALUATIONS_PER_ITERATION,
    )
    solved = least_squares(
        residual,
        selected_seed,
        bounds=(selected_lower, selected_upper),
        xtol=POSE_IK_NUMERIC_TOLERANCE,
        ftol=POSE_IK_NUMERIC_TOLERANCE,
        gtol=POSE_IK_NUMERIC_TOLERANCE,
        max_nfev=max_function_evaluations,
    )
    q = expand(solved.x)

    final_position, final_quaternion = eef_pose(state, arm, q)
    final_pos_error = float(np.linalg.norm(target_pos - final_position))
    final_ori_error = orientation_error_deg(final_quaternion, target_quat)
    return q, {
        "ok": bool(
            final_pos_error <= float(pos_tol_m)
            and final_ori_error <= float(ori_tol_deg)
        ),
        "solver": "submission_local_static_urdf_tolerance_normalized_least_squares",
        "solver_success": bool(solved.success),
        "solver_status": int(solved.status),
        "solver_message": str(solved.message),
        "iterations": int(solved.nfev),
        "max_function_evaluations": int(max_function_evaluations),
        "numeric_tolerance": float(POSE_IK_NUMERIC_TOLERANCE),
        "position_residual_scale_m": float(position_scale_m),
        "orientation_residual_scale_rad": float(orientation_scale_rad),
        "orientation_residual_weight_m_per_rad": float(
            orientation_weight_m_per_rad
        ),
        "residual_weighting": "equal_error_at_requested_tolerances",
        "pos_err_m": final_pos_error,
        "ori_err_deg": final_ori_error,
        "joint_numbers": [index + 1 for index in indices],
        "j8_participates": False,
    }


def bounded_pose_goal_step(
    state: LocalRobotState,
    arm: str,
    q_current: Sequence[float],
    q_goal: Sequence[float],
    *,
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
    pos_tol_m: float,
    ori_tol_deg: float,
    max_joint_step_rad: float,
    max_translation_step_m: float,
    max_orientation_step_rad: float,
    previous_dq: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Take a locally verified bounded step toward an accepted terminal IK."""

    current = locked_arm_vector(q_current, label=f"{arm} guided-step qpos")
    goal = locked_arm_vector(q_goal, label=f"{arm} guided-step goal")
    lower, upper = arm_joint_limits(state, arm)
    target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_quat = normalize_quaternion(target_quaternion)
    position_scale = max(1e-6, float(pos_tol_m))
    orientation_scale = max(1e-6, float(ori_tol_deg))

    current_position, current_quaternion = eef_pose(state, arm, current)
    current_position_error = float(np.linalg.norm(target_pos - current_position))
    current_orientation_error = orientation_error_deg(
        current_quaternion,
        target_quat,
    )
    current_score = max(
        current_position_error / position_scale,
        current_orientation_error / orientation_scale,
    )

    delta = np.clip(goal[:7], lower[:7], upper[:7]) - current[:7]
    delta_inf = float(np.linalg.norm(delta, ord=np.inf))
    if delta_inf > float(max_joint_step_rad):
        delta *= float(max_joint_step_rad) / max(delta_inf, 1e-12)

    previous = (
        None
        if previous_dq is None
        else np.asarray(previous_dq, dtype=np.float64).reshape(7)
    )
    if previous is not None:
        reversing = (
            delta * previous <= 0.0
        ) & (np.abs(delta - previous) > MAX_JOINT_STEP_CHANGE_RAD)
        delta[reversing] = np.clip(
            delta[reversing],
            previous[reversing] - MAX_JOINT_STEP_CHANGE_RAD,
            previous[reversing] + MAX_JOINT_STEP_CHANGE_RAD,
        )

    accepted: dict[str, Any] | None = None
    accepted_delta = np.zeros(7, dtype=np.float64)
    for backtrack_scale in (1.0, 0.5, 0.25, 0.125):
        trial_delta = float(backtrack_scale) * delta
        trial = current.copy()
        trial[:7] = np.clip(
            current[:7] + trial_delta,
            lower[:7],
            upper[:7],
        )
        if ARM_DOF == 8:
            trial[7] = 0.0
        trial_position, trial_quaternion = eef_pose(state, arm, trial)
        translation_step = float(
            np.linalg.norm(trial_position - current_position)
        )
        orientation_step = float(
            np.linalg.norm(
                orientation_error_vector(current_quaternion, trial_quaternion)
            )
        )
        position_error = float(np.linalg.norm(target_pos - trial_position))
        orientation_error_value = orientation_error_deg(
            trial_quaternion,
            target_quat,
        )
        score = max(
            position_error / position_scale,
            orientation_error_value / orientation_scale,
        )
        reaches_target = bool(
            position_error <= position_scale
            and orientation_error_value <= orientation_scale
        )
        bounded = bool(
            translation_step <= float(max_translation_step_m) + 1e-9
            and orientation_step <= float(max_orientation_step_rad) + 1e-9
        )
        improves = bool(score < current_score - 1e-4 or reaches_target)
        if bounded and improves:
            accepted_delta = trial[:7] - current[:7]
            accepted = {
                "backtrack_scale": float(backtrack_scale),
                "predicted_translation_step_m": translation_step,
                "predicted_orientation_step_rad": orientation_step,
                "predicted_position_error_m": position_error,
                "predicted_orientation_error_deg": orientation_error_value,
                "predicted_normalized_max_error": score,
            }
            break

    report = {
        "dof_count": 7,
        "solver_joint_numbers": list(range(1, 8)),
        "j8_participates": False,
        "pose_task": "bounded_static_fk_verified_terminal_ik_guidance",
        "target_ik_guidance_used": True,
        "guided_goal_step_accepted": accepted is not None,
        "current_normalized_max_error": float(current_score),
        "command_dq_inf_rad": float(
            np.linalg.norm(accepted_delta, ord=np.inf)
        ),
        "max_dq_rad": float(max_joint_step_rad),
        "max_translation_step_m": float(max_translation_step_m),
        "max_orientation_step_rad": float(max_orientation_step_rad),
    }
    if accepted is not None:
        report.update(accepted)
    return accepted_delta, report


def solve_j567_orientation_target(
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    target_quaternion: Sequence[float],
    *,
    ori_tol_deg: float,
    allow_global_search: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Port the current v2 J567 solver onto submission-local static FK."""
    start = locked_arm_vector(q_start, label=f"{arm} wrist IK seed")
    target_quat = normalize_quaternion(target_quaternion)
    target_position, _ = eef_pose(state, arm, start)
    solve_tol = min(1.0, max(0.20, float(ori_tol_deg) / 3.0))

    lower, upper = arm_joint_limits(state, arm)
    fixed_q1234 = np.clip(start[:4], lower[:4], upper[:4])
    limit_lower = np.asarray(lower[4:7], dtype=np.float64).copy()
    limit_upper = np.asarray(upper[4:7], dtype=np.float64).copy()
    seed_j567 = np.clip(start[4:7], limit_lower, limit_upper)

    def append_seed(seeds: list[np.ndarray], raw: Sequence[float]) -> None:
        candidate = np.clip(
            np.asarray(raw, dtype=np.float64).reshape(3),
            limit_lower,
            limit_upper,
        )
        if any(
            float(np.linalg.norm(candidate - old, ord=np.inf)) < 1e-5
            for old in seeds
        ):
            return
        seeds.append(candidate)

    def prefer_candidate(
        error_deg: float,
        seed_gap: float,
        best_error_deg: float,
        best_seed_gap: float,
    ) -> bool:
        candidate_ok = bool(error_deg <= solve_tol)
        incumbent_ok = bool(best_error_deg <= solve_tol)
        if candidate_ok != incumbent_ok:
            return candidate_ok
        if candidate_ok:
            return seed_gap < best_seed_gap - 1e-9
        return bool(
            error_deg < best_error_deg - 1e-8
            or (
                abs(error_deg - best_error_deg) <= 1e-8
                and seed_gap < best_seed_gap
            )
        )

    def solve_attempt(
        *,
        max_steps: int,
        max_dq_rad: float,
        global_search: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        best_j567 = seed_j567.copy()
        best_quaternion: np.ndarray | None = None
        best_error_deg = float("inf")
        best_seed_gap = float("inf")
        best_rank = 0
        probe_count = 0
        exception_text: str | None = None

        def expand(j567: Sequence[float]) -> np.ndarray:
            candidate = start.copy()
            candidate[:4] = fixed_q1234
            candidate[4:7] = np.clip(
                np.asarray(j567, dtype=np.float64).reshape(3),
                limit_lower,
                limit_upper,
            )
            if ARM_DOF == 8:
                candidate[7] = 0.0
            return candidate

        def probe(
            j567: Sequence[float],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            nonlocal probe_count
            q_probe = expand(j567)
            wrist = q_probe[4:7].copy()
            _position, quaternion = eef_pose(state, arm, q_probe)
            quaternion = normalize_quaternion(quaternion)
            omega = orientation_error_vector(quaternion, target_quat)
            error_deg = math.degrees(float(np.linalg.norm(omega)))
            probe_count += 1
            return wrist, quaternion, omega, error_deg

        def consider(
            wrist: Sequence[float],
            quaternion: Sequence[float],
            error_deg: float,
            rank: int,
        ) -> None:
            nonlocal best_j567
            nonlocal best_quaternion
            nonlocal best_error_deg
            nonlocal best_seed_gap
            nonlocal best_rank
            candidate = np.asarray(wrist, dtype=np.float64).reshape(3)
            seed_gap = float(
                np.linalg.norm(candidate - seed_j567, ord=np.inf)
            )
            if prefer_candidate(
                error_deg,
                seed_gap,
                best_error_deg,
                best_seed_gap,
            ):
                best_j567 = candidate.copy()
                best_quaternion = normalize_quaternion(quaternion)
                best_error_deg = float(error_deg)
                best_seed_gap = seed_gap
                best_rank = int(rank)

        seeds: list[np.ndarray] = []
        append_seed(seeds, seed_j567)
        if global_search:
            midpoint = (limit_lower + limit_upper) * 0.5
            append_seed(seeds, midpoint)
            append_seed(seeds, np.zeros(3, dtype=np.float64))
            for axis in range(3):
                for fraction in (0.15, 0.50, 0.85):
                    candidate = seed_j567.copy()
                    candidate[axis] = limit_lower[axis] + fraction * (
                        limit_upper[axis] - limit_lower[axis]
                    )
                    append_seed(seeds, candidate)
            for middle_fraction in (0.15, 0.50, 0.85):
                for edge_fraction in (0.15, 0.85):
                    candidate = midpoint.copy()
                    candidate[0] = limit_lower[0] + edge_fraction * (
                        limit_upper[0] - limit_lower[0]
                    )
                    candidate[1] = limit_lower[1] + middle_fraction * (
                        limit_upper[1] - limit_lower[1]
                    )
                    candidate[2] = limit_lower[2] + (
                        1.0 - edge_fraction
                    ) * (limit_upper[2] - limit_lower[2])
                    append_seed(seeds, candidate)

        try:
            solved = False
            for initial_wrist in seeds:
                wrist = initial_wrist.copy()
                no_improve = 0
                for _ in range(max(1, int(max_steps))):
                    wrist, quaternion, omega, error_deg = probe(wrist)
                    q_probe = expand(wrist)
                    jacobian, _position, _quaternion = numerical_eef_jacobian(
                        state,
                        arm,
                        q_probe,
                        joint_indices=(4, 5, 6),
                    )
                    angular_jacobian = jacobian[3:6, :]
                    rank = int(
                        np.linalg.matrix_rank(angular_jacobian, tol=1e-6)
                    )
                    consider(wrist, quaternion, error_deg, rank)
                    if error_deg <= solve_tol:
                        solved = True
                        break

                    damping = 0.025 if error_deg > 3.0 else 0.050
                    delta = _damped_step(
                        angular_jacobian,
                        omega,
                        damping,
                    )
                    delta_norm = float(np.linalg.norm(delta, ord=np.inf))
                    if delta_norm > float(max_dq_rad):
                        delta *= float(max_dq_rad) / (delta_norm + 1e-12)

                    next_wrist: np.ndarray | None = None
                    next_error = error_deg
                    for scale in (1.0, 0.5, 0.25):
                        candidate = np.clip(
                            wrist + float(scale) * delta,
                            limit_lower,
                            limit_upper,
                        )
                        (
                            candidate,
                            candidate_quaternion,
                            _candidate_omega,
                            candidate_error,
                        ) = probe(candidate)
                        consider(
                            candidate,
                            candidate_quaternion,
                            candidate_error,
                            rank,
                        )
                        if candidate_error < next_error - 1e-7:
                            next_wrist = candidate.copy()
                            next_error = candidate_error
                    if next_wrist is None:
                        no_improve += 1
                        if no_improve >= 3:
                            break
                    else:
                        wrist = next_wrist
                        no_improve = 0
                if solved:
                    break

            if best_error_deg > solve_tol and global_search:
                try:
                    from scipy.optimize import least_squares

                    refine_seeds = [best_j567.copy(), *seeds]
                    for initial_wrist in refine_seeds:
                        result = least_squares(
                            lambda wrist: probe(wrist)[2],
                            np.clip(initial_wrist, limit_lower, limit_upper),
                            bounds=(limit_lower, limit_upper),
                            xtol=1e-8,
                            ftol=1e-8,
                            gtol=1e-8,
                            max_nfev=max(40, int(max_steps)),
                        )
                        wrist, quaternion, _omega, error_deg = probe(result.x)
                        q_probe = expand(wrist)
                        jacobian, _position, _quaternion = numerical_eef_jacobian(
                            state,
                            arm,
                            q_probe,
                            joint_indices=(4, 5, 6),
                        )
                        rank = int(
                            np.linalg.matrix_rank(jacobian[3:6, :], tol=1e-6)
                        )
                        consider(wrist, quaternion, error_deg, rank)
                        if error_deg <= solve_tol:
                            break
                except Exception as exc:
                    exception_text = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"

        if best_quaternion is None:
            _position, best_quaternion = eef_pose(state, arm, start)
            best_quaternion = normalize_quaternion(best_quaternion)
            best_error_deg = orientation_error_deg(
                best_quaternion,
                target_quat,
            )
        ok = bool(best_error_deg <= solve_tol)
        q_best = expand(best_j567)
        return q_best, {
            "ok": ok,
            "arm": str(arm),
            "fixed_q1234_rad": fixed_q1234.tolist(),
            "j567_rad": best_j567.tolist(),
            "j567_deg": np.degrees(best_j567).tolist(),
            "target_quat_xyzw": target_quat.tolist(),
            "achieved_quat_xyzw": best_quaternion.tolist(),
            "ori_err_deg": float(best_error_deg),
            "residual_deg": float(best_error_deg),
            "ori_tol_deg": float(solve_tol),
            "jacobian_rank": int(best_rank),
            "probe_count": int(probe_count),
            "only_optimized_joints": [5, 6, 7],
            "joint_limits_j567_rad": {
                "lower": limit_lower.tolist(),
                "upper": limit_upper.tolist(),
            },
            "wrist_step_limit_rad": None,
            "search_bounds_j567_rad": {
                "lower": limit_lower.tolist(),
                "upper": limit_upper.tolist(),
            },
            "global_search": bool(global_search),
            "error": (
                None
                if ok
                else (
                    exception_text
                    or "orientation_unreachable_with_fixed_j1234_and_j567_limits"
                )
            ),
        }

    best_q, best_report = solve_attempt(
        max_steps=24,
        max_dq_rad=0.10,
        global_search=False,
    )
    attempts = 1
    used_global_search = False
    if not best_report["ok"] and allow_global_search:
        best_q, best_report = solve_attempt(
            max_steps=72,
            max_dq_rad=0.10,
            global_search=True,
        )
        attempts = 2
        used_global_search = True

    # Position is intentionally not constrained in this phase, matching v2.
    final_position, _ = eef_pose(state, arm, best_q)
    return best_q, {
        **best_report,
        "solver": "submission_local_static_urdf_j567_v29",
        "attempts": int(attempts),
        "used_global_search": bool(used_global_search),
        "fixed_joint_numbers": [1, 2, 3, 4],
        "commanded_joint_numbers": [5, 6, 7],
        "j8_participates": False,
        "position_unconstrained": True,
        "position_after_solve_robot_m": np.asarray(
            final_position,
            dtype=np.float64,
        ).tolist(),
        "reference_position_robot_m": np.asarray(
            target_position,
            dtype=np.float64,
        ).tolist(),
    }


def bounded_pose_tangent_step(
    jacobian: np.ndarray,
    translation_error: Sequence[float],
    orientation_error: Sequence[float],
    q_current: Sequence[float],
    *,
    q_lower: Sequence[float],
    q_upper: Sequence[float],
    previous_dq: Sequence[float] | None = None,
    joint_goal: Sequence[float] | None = None,
    position_satisfied: bool = False,
    max_joint_step_rad: float = MAX_JOINT_STEP_RAD,
    max_translation_step_m: float = MAX_TRANSLATION_STEP_M,
    max_orientation_step_rad: float = MAX_ORIENTATION_STEP_RAD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One bounded v2-style near-priority translation/orientation step."""
    from scipy.optimize import Bounds, lsq_linear, minimize

    matrix = np.asarray(jacobian, dtype=np.float64).reshape(6, 7)
    dx = np.asarray(translation_error, dtype=np.float64).reshape(3)
    omega = np.asarray(orientation_error, dtype=np.float64).reshape(3)
    q = np.asarray(q_current, dtype=np.float64).reshape(7)
    lower_abs = np.asarray(q_lower, dtype=np.float64).reshape(7)
    upper_abs = np.asarray(q_upper, dtype=np.float64).reshape(7)
    previous = (
        None
        if previous_dq is None
        else np.asarray(previous_dq, dtype=np.float64).reshape(7)
    )
    goal = (
        None
        if joint_goal is None
        else np.asarray(joint_goal, dtype=np.float64).reshape(7)
    )
    max_joint_step = float(max_joint_step_rad)
    if not math.isfinite(max_joint_step) or max_joint_step <= 0.0:
        raise ValueError("max_joint_step_rad must be positive and finite")
    max_translation_step = float(max_translation_step_m)
    if not math.isfinite(max_translation_step) or max_translation_step <= 0.0:
        raise ValueError("max_translation_step_m must be positive and finite")
    max_orientation_step = float(max_orientation_step_rad)
    if not math.isfinite(max_orientation_step) or max_orientation_step <= 0.0:
        raise ValueError("max_orientation_step_rad must be positive and finite")

    dx_norm = float(np.linalg.norm(dx))
    if dx_norm > max_translation_step:
        dx *= max_translation_step / max(dx_norm, 1e-12)
    omega_norm = float(np.linalg.norm(omega))
    orientation_weight = 1.0
    orientation_saturated = omega_norm > max_orientation_step
    if orientation_saturated:
        omega *= max_orientation_step / max(omega_norm, 1e-12)
        orientation_weight *= min(
            ORIENTATION_WEIGHT_MAX_SCALE,
            omega_norm / max_orientation_step,
        )

    lower_dq = np.maximum(-max_joint_step, lower_abs - q)
    upper_dq = np.minimum(max_joint_step, upper_abs - q)
    below = q < lower_abs
    above = q > upper_abs
    lower_dq[below] = upper_dq[below] = np.minimum(
        max_joint_step,
        lower_abs[below] - q[below],
    )
    lower_dq[above] = upper_dq[above] = -np.minimum(
        max_joint_step,
        q[above] - upper_abs[above],
    )

    def solve_bounded(
        solve_matrix: np.ndarray,
        solve_rhs: np.ndarray,
    ) -> tuple[np.ndarray, int, str]:
        fixed = (upper_dq - lower_dq) <= 1e-10
        result = np.zeros(7, dtype=np.float64)
        if np.any(fixed):
            result[fixed] = 0.5 * (lower_dq[fixed] + upper_dq[fixed])
        free = ~fixed
        status = 0
        message = "all joints fixed"
        if np.any(free):
            free_rhs = solve_rhs - solve_matrix[:, fixed] @ result[fixed]
            solved = lsq_linear(
                solve_matrix[:, free],
                free_rhs,
                bounds=(lower_dq[free], upper_dq[free]),
                method="trf",
                lsq_solver="exact",
                tol=1e-10,
                max_iter=100,
            )
            result[free] = solved.x
            status = int(solved.status)
            message = str(solved.message)
        return result, status, message

    regularizer = np.diag(JOINT_REGULARIZATION_WEIGHTS)
    translation_regularization = DLS_LAMBDA * 0.05
    primary_matrix = np.vstack(
        [matrix[:3], translation_regularization * regularizer]
    )
    primary_rhs = np.concatenate([dx, np.zeros(7, dtype=np.float64)])
    primary_dq, primary_status, primary_message = solve_bounded(
        primary_matrix,
        primary_rhs,
    )
    primary_translation = matrix[:3] @ primary_dq

    base_slack = min(0.0015, 0.25 * dx_norm)
    if position_satisfied:
        translation_slack = 0.003
    elif orientation_saturated:
        translation_slack = min(
            0.006,
            base_slack * omega_norm / max_orientation_step,
        )
    else:
        translation_slack = base_slack

    goal_step = None
    if goal is not None and np.all(np.isfinite(goal)):
        goal_step = np.clip(goal, lower_abs, upper_abs) - q
        goal_inf = float(np.linalg.norm(goal_step, ord=np.inf))
        if goal_inf > max_joint_step:
            goal_step *= max_joint_step / max(goal_inf, 1e-12)
        goal_step = np.clip(goal_step, lower_dq, upper_dq)

    def objective(candidate: np.ndarray) -> float:
        value = np.asarray(candidate, dtype=np.float64).reshape(7)
        orientation_residual = matrix[3:] @ value - omega
        regularized = regularizer @ value
        cost = orientation_weight**2 * float(
            orientation_residual @ orientation_residual
        ) + DLS_LAMBDA**2 * float(regularized @ regularized)
        if goal_step is not None:
            goal_residual = value - goal_step
            cost += TARGET_IK_GUIDANCE_WEIGHT**2 * float(
                goal_residual @ goal_residual
            )
        return cost

    def retain_translation(candidate: np.ndarray) -> float:
        retained = matrix[:3] @ np.asarray(candidate) - primary_translation
        return translation_slack**2 - float(retained @ retained)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step.*",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._slsqp_py",
        )
        secondary = minimize(
            objective,
            primary_dq,
            method="SLSQP",
            bounds=Bounds(lower_dq, upper_dq),
            constraints={"type": "ineq", "fun": retain_translation},
            options={"ftol": 1e-12, "maxiter": 80, "disp": False},
        )
    if bool(secondary.success) and retain_translation(secondary.x) >= -1e-8:
        delta = np.asarray(secondary.x, dtype=np.float64)
    else:
        delta = primary_dq

    continuity_limited = np.zeros(7, dtype=bool)
    if previous is not None:
        reversing = (
            delta * previous <= 0.0
        ) & (np.abs(delta - previous) > MAX_JOINT_STEP_CHANGE_RAD)
        adjusted = delta.copy()
        adjusted[reversing] = np.clip(
            adjusted[reversing],
            previous[reversing] - MAX_JOINT_STEP_CHANGE_RAD,
            previous[reversing] + MAX_JOINT_STEP_CHANGE_RAD,
        )
        continuity_limited = ~np.isclose(adjusted, delta, atol=1e-10)
        delta = adjusted
    delta = np.clip(delta, lower_dq, upper_dq)

    return delta, {
        "dof_count": 7,
        "solver_joint_numbers": list(range(1, 8)),
        "j8_participates": False,
        "pose_task": "bounded_compliant_6d_fixed_target_tangent",
        "task_priority": "near_optimal_translation_with_orientation_retention",
        "target_ik_guidance_used": goal_step is not None,
        "orientation_error_norm_rad": omega_norm,
        "orientation_step_saturated": orientation_saturated,
        "orientation_weight_scale": orientation_weight,
        "translation_residual_slack_m": float(translation_slack),
        "position_satisfied": bool(position_satisfied),
        "command_dq_inf_rad": float(np.linalg.norm(delta, ord=np.inf)),
        "max_dq_rad": max_joint_step,
        "max_translation_step_m": max_translation_step,
        "max_orientation_step_rad": max_orientation_step,
        "predicted_translation_m": (matrix[:3] @ delta).tolist(),
        "predicted_orientation_rad": (matrix[3:] @ delta).tolist(),
        "continuity_limited_joint_numbers": (
            np.flatnonzero(continuity_limited).astype(int) + 1
        ).tolist(),
        "primary_solver_success": bool(primary_status >= 0),
        "primary_solver_status": int(primary_status),
        "primary_solver_message": primary_message,
        "secondary_success": bool(secondary.success),
    }


__all__ = [
    "ADJUST_POSE_MAX_JOINT_STEP_RAD",
    "MAX_JOINT_STEP_RAD",
    "MAX_TRANSLATION_STEP_M",
    "MAX_ORIENTATION_STEP_RAD",
    "POSE_IK_FUNCTION_EVALUATIONS_PER_ITERATION",
    "POSE_IK_MIN_FUNCTION_EVALUATIONS",
    "POSE_IK_NUMERIC_TOLERANCE",
    "PoseTarget",
    "WRIST_MAX_JOINT_STEP_RAD",
    "bounded_pose_goal_step",
    "bounded_pose_tangent_step",
    "camera_frame_pose_target",
    "local_robot_state",
    "locked_arm_vector",
    "numerical_eef_jacobian",
    "orientation_error_deg",
    "orientation_error_vector",
    "robot_base_frame_pose_target",
    "solve_j567_orientation_target",
    "solve_pose_target",
]
