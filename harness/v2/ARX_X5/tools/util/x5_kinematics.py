"""Policy-local kinematics for RoboDojo's six-axis X5 arm.

The constants below are the serial link geometry of the public X5A robot.
Runtime calibration uses only the evaluator's joint and EEF observations, so
the controller does not need simulator objects, task state, or evaluator IK.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


X5_JOINT_ORIGIN_XYZ = np.array(
    [
        [0.0, 0.0, 0.0605],
        [0.0200, 0.0, 0.0400],
        [-0.2640, 0.0, 0.0],
        [0.2450, 0.0, -0.0560],
        [0.06775, 0.0005, -0.0865],
        [0.02895, 0.0, 0.0865],
    ],
    dtype=np.float64,
)
X5_JOINT_ORIGIN_RPY = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.1416, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-3.1416, 0.0, 0.0],
    ],
    dtype=np.float64,
)
X5_JOINT_AXES = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
X5_JOINT_LOWER = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -3.14])
X5_JOINT_UPPER = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 3.14])


@dataclass(frozen=True)
class X5IKResult:
    joint_target: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_deg: float
    initial_position_error_m: float
    initial_orientation_error_deg: float
    calibration_translation_m: np.ndarray


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: Any) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -(value[:3, :3].T @ value[:3, 3])
    return result


def quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-10:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = value / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        value = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
                * 2.0
            )
            value = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
                * 2.0
            )
            value = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
                * 2.0
            )
            value = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = float(np.linalg.norm(value))
    if norm <= 1e-10:
        raise ValueError("rotation matrix produced a zero quaternion")
    value /= norm
    if value[0] < 0.0:
        value = -value
    return value


def pose_to_transform(pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.size != 7 or not np.all(np.isfinite(value)):
        raise ValueError("pose must contain seven finite values")
    return _rigid_transform(quaternion_to_matrix(value[3:]), value[:3])


def transform_to_pose(transform: Any) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate((value[:3, 3], matrix_to_quaternion(value[:3, :3])))


def _forward_with_axes(joints: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.size != 6 or not np.all(np.isfinite(q)):
        raise ValueError("X5 joints must contain six finite values")
    transform = np.eye(4, dtype=np.float64)
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    for index in range(6):
        transform = transform @ _rigid_transform(
            _rpy_matrix(X5_JOINT_ORIGIN_RPY[index]),
            X5_JOINT_ORIGIN_XYZ[index],
        )
        origins.append(transform[:3, 3].copy())
        axes.append(transform[:3, :3] @ X5_JOINT_AXES[index])
        transform = transform @ _rigid_transform(
            _axis_angle_matrix(X5_JOINT_AXES[index], float(q[index])),
            np.zeros(3),
        )
    return transform, np.asarray(origins), np.asarray(axes)


def x5_forward_kinematics(joints: Any) -> np.ndarray:
    return _forward_with_axes(joints)[0]


def x5_spatial_jacobian(joints: Any) -> np.ndarray:
    end_transform, origins, axes = _forward_with_axes(joints)
    end_position = end_transform[:3, 3]
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for index in range(6):
        jacobian[:3, index] = np.cross(axes[index], end_position - origins[index])
        jacobian[3:, index] = axes[index]
    return jacobian


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    quaternion = matrix_to_quaternion(rotation)
    vector = quaternion[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    return vector * (angle / vector_norm)


def _pose_error(
    current: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = target[:3, 3] - current[:3, 3]
    rotation = _rotation_vector(target[:3, :3] @ current[:3, :3].T)
    return position, rotation


def calibrated_pose_for_joints(
    reference_joints: Any,
    reference_observed_pose: Any,
    query_joints: Any,
) -> np.ndarray:
    reference_model = x5_forward_kinematics(reference_joints)
    world_from_model = pose_to_transform(reference_observed_pose) @ invert_transform(
        reference_model
    )
    return transform_to_pose(world_from_model @ x5_forward_kinematics(query_joints))


def solve_calibrated_x5_ik(
    current_joints: Any,
    current_observed_pose: Any,
    target_observed_pose: Any,
    *,
    initial_joints: Any | None = None,
    max_iterations: int = 120,
    position_tolerance_m: float = 0.0005,
    orientation_tolerance_deg: float = 0.15,
    damping: float = 0.015,
    orientation_weight_m_per_rad: float = 0.20,
) -> X5IKResult:
    """Solve a target expressed in the evaluator observation frame.

    The world-to-model transform is recalibrated from the current allowed
    proprioceptive sample. This absorbs the arm root pose and small simulator
    convention differences without consulting simulator internals.
    """

    current_q = np.asarray(current_joints, dtype=np.float64).reshape(-1)
    if current_q.size != 6 or not np.all(np.isfinite(current_q)):
        raise ValueError("current X5 joints must contain six finite values")
    observed = pose_to_transform(current_observed_pose)
    target_world = pose_to_transform(target_observed_pose)
    model_at_observation = x5_forward_kinematics(current_q)
    world_from_model = observed @ invert_transform(model_at_observation)
    model_target = invert_transform(world_from_model) @ target_world

    initial_position, initial_rotation = _pose_error(model_at_observation, model_target)
    if initial_joints is None:
        q = current_q.copy()
    else:
        q = np.asarray(initial_joints, dtype=np.float64).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise ValueError("initial X5 joints must contain six finite values")
        q = np.clip(q, X5_JOINT_LOWER, X5_JOINT_UPPER)
    best_q = q.copy()
    best_score = math.inf
    completed_iterations = 0
    converged = False

    for iteration in range(max(1, int(max_iterations))):
        completed_iterations = iteration + 1
        model_pose = x5_forward_kinematics(q)
        position_error, rotation_error = _pose_error(model_pose, model_target)
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(rotation_error))
        score = position_norm + orientation_weight_m_per_rad * orientation_norm
        if score < best_score:
            best_score = score
            best_q = q.copy()
        if (
            position_norm <= position_tolerance_m
            and math.degrees(orientation_norm) <= orientation_tolerance_deg
        ):
            converged = True
            break

        jacobian = x5_spatial_jacobian(q)
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:] *= orientation_weight_m_per_rad
        weighted_error = np.concatenate(
            (position_error, rotation_error * orientation_weight_m_per_rad)
        )
        normal = weighted_jacobian.T @ weighted_jacobian
        normal += (float(damping) ** 2) * np.eye(6)
        step = np.linalg.solve(normal, weighted_jacobian.T @ weighted_error)
        largest = float(np.max(np.abs(step)))
        if largest > 0.18:
            step *= 0.18 / largest
        if float(np.linalg.norm(step)) <= 1e-8:
            break

        accepted = False
        scale = 1.0
        for _ in range(10):
            candidate = np.clip(
                q + scale * step,
                X5_JOINT_LOWER,
                X5_JOINT_UPPER,
            )
            candidate_pose = x5_forward_kinematics(candidate)
            candidate_position, candidate_rotation = _pose_error(
                candidate_pose, model_target
            )
            candidate_score = float(np.linalg.norm(candidate_position)) + (
                orientation_weight_m_per_rad * float(np.linalg.norm(candidate_rotation))
            )
            if candidate_score < score - 1e-10:
                q = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    final_pose = x5_forward_kinematics(best_q)
    final_position, final_rotation = _pose_error(final_pose, model_target)
    final_position_norm = float(np.linalg.norm(final_position))
    final_orientation_deg = math.degrees(float(np.linalg.norm(final_rotation)))
    converged = bool(
        converged
        or (
            final_position_norm <= position_tolerance_m
            and final_orientation_deg <= orientation_tolerance_deg
        )
    )
    return X5IKResult(
        joint_target=best_q,
        converged=converged,
        iterations=completed_iterations,
        position_error_m=final_position_norm,
        orientation_error_deg=final_orientation_deg,
        initial_position_error_m=float(np.linalg.norm(initial_position)),
        initial_orientation_error_deg=math.degrees(
            float(np.linalg.norm(initial_rotation))
        ),
        calibration_translation_m=world_from_model[:3, 3].copy(),
    )


__all__ = [
    "X5IKResult",
    "calibrated_pose_for_joints",
    "invert_transform",
    "matrix_to_quaternion",
    "pose_to_transform",
    "quaternion_to_matrix",
    "solve_calibrated_x5_ik",
    "transform_to_pose",
    "x5_forward_kinematics",
    "x5_spatial_jacobian",
]
