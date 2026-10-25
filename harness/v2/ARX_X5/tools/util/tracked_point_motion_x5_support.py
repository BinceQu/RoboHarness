"""Dual-X5 backend for the BEHAVIOR v2 tracked-point planner.

The public planner operates in a BEHAVIOR-compatible robot-base frame and
uses xyzw quaternions internally. RoboDojo evaluator observations use the
fixed environment frame and wxyz quaternions, so the live tool freezes and
converts those poses before constructing :class:`LocalRobotState`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from tools.util.x5_kinematics import (
    X5_JOINT_LOWER,
    X5_JOINT_UPPER,
    calibrated_pose_for_joints,
    matrix_to_quaternion,
    quaternion_to_matrix,
    solve_calibrated_x5_ik,
)
from tools.util.contract import ARM_DOF


def normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def quat_to_mat_xyzw(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(value)
    return quaternion_to_matrix([w, x, y, z])


def mat_to_quat_xyzw(rotation: Any) -> np.ndarray:
    w, x, y, z = matrix_to_quaternion(rotation)
    return np.asarray([x, y, z, w], dtype=np.float64)


def _rotation_vector(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quaternion = mat_to_quat_xyzw(matrix)
    vector = quaternion[:3]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    angle = 2.0 * math.atan2(norm, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector * (angle / norm)


def orientation_error_vector(
    current_quaternion: Sequence[float],
    target_quaternion: Sequence[float],
) -> np.ndarray:
    current = quat_to_mat_xyzw(current_quaternion)
    target = quat_to_mat_xyzw(target_quaternion)
    return _rotation_vector(target @ current.T)


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


@dataclass(frozen=True)
class LocalRobotState:
    """Frozen dual-X5 proprioception used by the submission-local planner."""

    arm_dof: int
    arm_left_q: np.ndarray
    arm_right_q: np.ndarray
    eef_left_pose_base_wxyz: np.ndarray
    eef_right_pose_base_wxyz: np.ndarray

    @classmethod
    def from_capture(
        cls,
        robot: dict[str, Any],
        *,
        arm_dof: int = ARM_DOF,
    ) -> "LocalRobotState":
        if int(arm_dof) != ARM_DOF:
            raise ValueError(f"RoboDojo X5 requires arm_dof={ARM_DOF}")

        def vector(key: str, size: int) -> np.ndarray:
            result = np.asarray(robot.get(key), dtype=np.float64).reshape(-1)
            if result.size != size or not np.all(np.isfinite(result)):
                raise ValueError(f"{key} must contain {size} finite values")
            return result.copy()

        return cls(
            arm_dof=ARM_DOF,
            arm_left_q=vector("arm_left_qpos", ARM_DOF),
            arm_right_q=vector("arm_right_qpos", ARM_DOF),
            eef_left_pose_base_wxyz=vector("eef_left_pose_base_wxyz", 7),
            eef_right_pose_base_wxyz=vector("eef_right_pose_base_wxyz", 7),
        )

    def reference(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        side = str(arm).strip().lower()
        if side == "left":
            return self.arm_left_q.copy(), self.eef_left_pose_base_wxyz.copy()
        if side == "right":
            return self.arm_right_q.copy(), self.eef_right_pose_base_wxyz.copy()
        raise ValueError(f"bad arm {arm!r}")


def arm_joint_limits(
    state: LocalRobotState,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    state.reference(arm)
    return X5_JOINT_LOWER.copy(), X5_JOINT_UPPER.copy()


def eef_pose(
    state: LocalRobotState,
    arm: str,
    q_arm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(q_arm, dtype=np.float64).reshape(-1)
    if query.size != ARM_DOF or not np.all(np.isfinite(query)):
        raise ValueError(f"{arm} X5 q must contain {ARM_DOF} finite values")
    reference_q, reference_pose = state.reference(arm)
    pose = calibrated_pose_for_joints(reference_q, reference_pose, query)
    return pose[:3].copy(), np.asarray(
        [pose[4], pose[5], pose[6], pose[3]], dtype=np.float64
    )


def solve_pose_target(
    state: LocalRobotState,
    arm: str,
    q_seed: Sequence[float],
    *,
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
    joint_indices: Iterable[int] = range(ARM_DOF),
    pos_tol_m: float = 0.010,
    ori_tol_deg: float = 3.0,
    max_iterations: int = 160,
    check_requested: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if callable(check_requested):
        check_requested()
    indices = tuple(int(index) for index in joint_indices)
    if indices != tuple(range(ARM_DOF)):
        raise ValueError("X5 tracked-point IK requires all six arm joints")
    seed = np.asarray(q_seed, dtype=np.float64).reshape(ARM_DOF)
    target_xyzw = normalize_quaternion(target_quaternion)
    target_wxyz = np.asarray([target_xyzw[3], *target_xyzw[:3]], dtype=np.float64)
    target_pose = np.concatenate(
        [np.asarray(target_position, dtype=np.float64).reshape(3), target_wxyz]
    )
    reference_q, reference_pose = state.reference(arm)
    solved = solve_calibrated_x5_ik(
        reference_q,
        reference_pose,
        target_pose,
        initial_joints=seed,
        max_iterations=max_iterations,
        position_tolerance_m=float(pos_tol_m),
        orientation_tolerance_deg=float(ori_tol_deg),
        damping=0.012,
    )
    if callable(check_requested):
        check_requested()
    return solved.joint_target.copy(), {
        "ok": bool(solved.converged),
        "solver": "submission_local_calibrated_x5_damped_ik",
        "iterations": int(solved.iterations),
        "pos_err_m": float(solved.position_error_m),
        "ori_err_deg": float(solved.orientation_error_deg),
    }


def least_squares(
    function: Callable[[np.ndarray], Sequence[float]],
    x0: Sequence[float],
    *,
    bounds: tuple[Sequence[float], Sequence[float]],
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 120,
) -> SimpleNamespace:
    """Small bounded damped least-squares backend with SciPy-like results.

    RoboDojo's deployment environment does not ship SciPy. The BEHAVIOR
    planner only needs deterministic bounded residual minimization, so this
    keeps its multi-start and candidate logic intact without adding a runtime
    dependency or evaluator-side IK call.
    """

    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    x = np.clip(np.asarray(x0, dtype=np.float64).reshape(-1), lower, upper)
    if x.size != lower.size or lower.size != upper.size:
        raise ValueError("least_squares bounds do not match x0")
    if np.any(lower > upper) or not np.all(np.isfinite(x)):
        raise ValueError("least_squares received invalid bounds or seed")
    evaluation_limit = max(1, int(max_nfev))
    nfev = 0

    def evaluate(candidate: np.ndarray) -> np.ndarray:
        nonlocal nfev
        if nfev >= evaluation_limit:
            raise StopIteration
        value = np.asarray(function(candidate), dtype=np.float64).reshape(-1)
        nfev += 1
        if not np.all(np.isfinite(value)):
            raise ValueError("least_squares residual contains non-finite values")
        return value

    residual = evaluate(x)
    cost = 0.5 * float(residual @ residual)
    best_x = x.copy()
    best_residual = residual.copy()
    best_cost = cost
    damping = 1e-3
    success = False
    status = 0
    message = "maximum function evaluations reached"

    while nfev < evaluation_limit:
        if residual.size == 0 or float(np.max(np.abs(residual))) <= float(gtol):
            success = True
            status = 1
            message = "residual tolerance reached"
            break
        jacobian = np.zeros((residual.size, x.size), dtype=np.float64)
        complete = True
        for column in range(x.size):
            epsilon = 2e-5 * max(1.0, abs(float(x[column])))
            probe = x.copy()
            probe[column] = min(float(upper[column]), float(x[column]) + epsilon)
            delta = float(probe[column] - x[column])
            if delta <= 1e-12:
                probe[column] = max(float(lower[column]), float(x[column]) - epsilon)
                delta = float(probe[column] - x[column])
            if abs(delta) <= 1e-12:
                continue
            try:
                probe_residual = evaluate(probe)
            except StopIteration:
                complete = False
                break
            jacobian[:, column] = (probe_residual - residual) / delta
        if not complete:
            break

        gradient = jacobian.T @ residual
        if float(np.max(np.abs(gradient))) <= float(gtol):
            success = True
            status = 1
            message = "gradient tolerance reached"
            break
        normal = jacobian.T @ jacobian + damping * np.eye(x.size)
        try:
            step = -np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(normal, gradient, rcond=None)[0]
        largest = float(np.max(np.abs(step))) if step.size else 0.0
        if largest > 0.30:
            step *= 0.30 / largest
        if float(np.linalg.norm(step, ord=np.inf)) <= float(xtol):
            success = True
            status = 3
            message = "step tolerance reached"
            break

        accepted = False
        scale = 1.0
        previous_cost = cost
        for _ in range(10):
            candidate = np.clip(x + scale * step, lower, upper)
            if float(np.linalg.norm(candidate - x, ord=np.inf)) <= 1e-12:
                scale *= 0.5
                continue
            try:
                candidate_residual = evaluate(candidate)
            except StopIteration:
                break
            candidate_cost = 0.5 * float(candidate_residual @ candidate_residual)
            if candidate_cost < cost - 1e-14:
                x = candidate
                residual = candidate_residual
                cost = candidate_cost
                damping = max(1e-8, damping * 0.4)
                accepted = True
                if cost < best_cost:
                    best_x = x.copy()
                    best_residual = residual.copy()
                    best_cost = cost
                break
            scale *= 0.5
        if not accepted:
            damping = min(1e8, damping * 10.0)
            if damping >= 1e7:
                status = 2
                message = "no improving bounded step"
                break
        elif abs(previous_cost - cost) <= float(ftol) * max(1.0, previous_cost):
            success = True
            status = 2
            message = "cost tolerance reached"
            break

    return SimpleNamespace(
        x=best_x,
        fun=best_residual,
        cost=float(best_cost),
        success=bool(success),
        status=int(status),
        message=message,
        nfev=int(nfev),
    )


def minimize(
    function: Callable[[np.ndarray], float],
    x0: Sequence[float],
    *,
    method: str | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    constraints: Sequence[dict[str, Any]] = (),
    options: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Deterministic bounded pattern-search subset of ``scipy.minimize``.

    official_v2 uses minimization only as a secondary accuracy/motion
    refinement after bounded least-squares has produced valid IK basins.  The
    RoboDojo interface environment intentionally has no SciPy dependency, so
    this backend preserves that refinement and its inequality gates without
    introducing an evaluator-side solver.
    """

    del method
    start = np.asarray(x0, dtype=np.float64).reshape(-1)
    if bounds is None:
        lower = np.full(start.size, -np.inf, dtype=np.float64)
        upper = np.full(start.size, np.inf, dtype=np.float64)
    else:
        bound_array = np.asarray(list(bounds), dtype=np.float64)
        if bound_array.shape != (start.size, 2):
            raise ValueError("minimize bounds do not match x0")
        lower, upper = bound_array[:, 0], bound_array[:, 1]
    if np.any(lower > upper) or not np.all(np.isfinite(start)):
        raise ValueError("minimize received invalid bounds or seed")
    current = np.clip(start, lower, upper)
    settings = dict(options or {})
    max_evaluations = max(
        1,
        int(settings.get("maxfev", settings.get("maxiter", 120))),
    )
    x_tolerance = max(
        1.0e-10,
        float(settings.get("xatol", settings.get("ftol", 1.0e-7))),
    )
    span = np.where(
        np.isfinite(upper - lower),
        np.maximum(upper - lower, 1.0e-3),
        1.0,
    )
    step = np.minimum(0.05 * span, 0.10)
    step = np.maximum(step, 1.0e-3)
    nfev = 0

    def evaluate(candidate: np.ndarray) -> tuple[float, float, bool]:
        nonlocal nfev
        if nfev >= max_evaluations:
            raise StopIteration
        raw = float(function(candidate))
        nfev += 1
        if not math.isfinite(raw):
            raw = 1.0e12
        violation = 0.0
        feasible = True
        for record in constraints or ():
            if str(record.get("type") or "ineq") != "ineq":
                raise ValueError("only inequality minimize constraints are supported")
            values = np.asarray(record["fun"](candidate), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                violation = math.inf
                feasible = False
                break
            negative = np.minimum(values, 0.0)
            violation += float(negative @ negative)
            feasible = bool(feasible and np.all(values >= -1.0e-9))
        penalized = raw + (1.0e9 * violation if math.isfinite(violation) else 1.0e15)
        return raw, penalized, feasible

    raw_value, value, feasible = evaluate(current)
    best = current.copy()
    best_raw = raw_value
    best_value = value
    best_feasible = feasible
    status = 0
    message = "maximum function evaluations reached"
    while nfev < max_evaluations:
        improved = False
        for index in range(current.size):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[index] = np.clip(
                    candidate[index] + direction * step[index],
                    lower[index],
                    upper[index],
                )
                if abs(float(candidate[index] - current[index])) <= 1.0e-14:
                    continue
                try:
                    candidate_raw, candidate_value, candidate_feasible = evaluate(
                        candidate
                    )
                except StopIteration:
                    break
                better = candidate_value < value - 1.0e-12
                if candidate_feasible and not feasible:
                    better = True
                if better:
                    current = candidate
                    raw_value = candidate_raw
                    value = candidate_value
                    feasible = candidate_feasible
                    improved = True
                if (candidate_feasible and not best_feasible) or (
                    candidate_feasible == best_feasible and candidate_value < best_value
                ):
                    best = candidate.copy()
                    best_raw = candidate_raw
                    best_value = candidate_value
                    best_feasible = candidate_feasible
            if nfev >= max_evaluations:
                break
        if not improved:
            step *= 0.5
            if float(np.max(step)) <= x_tolerance:
                status = 1
                message = "pattern-search step tolerance reached"
                break
    success = bool(best_feasible and math.isfinite(best_raw))
    if success and status == 0:
        message = "bounded pattern-search evaluation limit reached"
    return SimpleNamespace(
        x=best,
        fun=float(best_raw),
        success=success,
        status=int(status),
        message=message,
        nfev=int(nfev),
    )


__all__ = [
    "LocalRobotState",
    "arm_joint_limits",
    "eef_pose",
    "least_squares",
    "mat_to_quat_xyzw",
    "minimize",
    "normalize_quaternion",
    "orientation_error_deg",
    "orientation_error_vector",
    "quat_to_mat_xyzw",
    "solve_pose_target",
]
