"""Pure execution-side helpers for tracked-point trajectory replay.

This module deliberately has no evaluator, simulator, scene, or robot-handle
dependencies.  It validates and deterministically subsamples an already
verified local joint trajectory for either a hard evaluator budget or an
explicit execution-latency target.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _waypoint_matrix(raw: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a copied finite ``N x D`` waypoint matrix."""

    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution waypoints must be a finite N x D array") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("execution waypoints must be a non-empty N x D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("execution waypoints must contain only finite values")
    return matrix.copy()


def _locked_matrix(
    matrix: np.ndarray,
    *,
    j8_index: int | None,
    j8_value: float,
    tolerance: float,
) -> np.ndarray:
    """Validate the locked wrist coordinate without silently changing it."""

    result = matrix.copy()
    if j8_index is None:
        return result
    index = int(j8_index)
    if index < 0 or index >= result.shape[1]:
        raise ValueError("locked joint index is outside the waypoint dimension")
    expected = float(j8_value)
    if not math.isfinite(expected):
        raise ValueError("locked joint value must be finite")
    values = result[:, index]
    if np.any(np.abs(values - expected) > float(tolerance)):
        raise ValueError("execution trajectory violates the locked joint contract")
    # Keep the exact contract value in the returned copy.  This is a
    # validation-only normalization; a non-zero input has already been
    # rejected above.
    result[:, index] = expected
    return result


def _select_with_step_cap(
    matrix: np.ndarray,
    *,
    start: np.ndarray,
    cap: float,
    tolerance: float,
    mandatory_indices: Sequence[int] = (),
    corridor_tolerance: float | None = None,
) -> tuple[np.ndarray, list[int], float, dict[str, Any]] | None:
    """Greedily retain the furthest reachable original sample at each step.

    The source path is ordered.  Choosing the furthest later index whose
    direct joint displacement is within ``cap`` gives a deterministic,
    monotone subsequence and never reorders or invents a waypoint.  Optional
    mandatory indices split the search into intervals, and the joint-space
    corridor prevents a shortcut from cutting across a bend in the signed
    dense path.
    """

    if not math.isfinite(float(cap)) or float(cap) <= 0.0:
        return None
    count = int(matrix.shape[0])
    current = np.asarray(start, dtype=np.float64).reshape(-1)
    if current.shape[0] != matrix.shape[1] or not np.all(np.isfinite(current)):
        raise ValueError("execution start q has the wrong finite dimension")
    selected: list[int] = []
    cursor = -1
    maximum_step = 0.0
    corridor_max_deviation = 0.0
    corridor_max_regression = 0.0
    corridor_transition_count = 0
    boundaries = sorted(
        set(int(index) for index in mandatory_indices) | {count - 1}
    )
    progress_tolerance = max(1.0e-6, float(tolerance))
    for boundary in boundaries:
        if boundary <= cursor:
            continue
        while cursor < boundary:
            furthest: int | None = None
            furthest_step = 0.0
            furthest_corridor: dict[str, float] | None = None
            for index in range(cursor + 1, boundary + 1):
                step = float(np.linalg.norm(matrix[index] - current, ord=np.inf))
                if not math.isfinite(step):
                    return None
                if step > float(cap) + float(tolerance):
                    continue
                corridor = _joint_corridor_report(
                    current,
                    matrix[index],
                    matrix[cursor + 1 : index + 1],
                    tolerance=float(tolerance),
                )
                if corridor_tolerance is not None and (
                    corridor["max_deviation_rad"]
                    > float(corridor_tolerance) + float(tolerance)
                    or corridor["max_progress_regression"]
                    > progress_tolerance
                    or corridor["min_progress"] < -progress_tolerance
                    or corridor["max_progress"] > 1.0 + progress_tolerance
                ):
                    continue
                furthest = index
                furthest_step = step
                furthest_corridor = corridor
            if furthest is None or furthest_corridor is None:
                return None
            selected.append(int(furthest))
            maximum_step = max(maximum_step, furthest_step)
            corridor_max_deviation = max(
                corridor_max_deviation,
                float(furthest_corridor["max_deviation_rad"]),
            )
            corridor_max_regression = max(
                corridor_max_regression,
                float(furthest_corridor["max_progress_regression"]),
            )
            corridor_transition_count += 1
            current = matrix[furthest].copy()
            cursor = furthest
    return (
        matrix[np.asarray(selected, dtype=np.int64)].copy(),
        selected,
        maximum_step,
        {
            "checked": corridor_tolerance is not None,
            "tolerance_rad": corridor_tolerance,
            "max_deviation_rad": float(corridor_max_deviation),
            "max_progress_regression": float(corridor_max_regression),
            "transition_count": int(corridor_transition_count),
        },
    )


def _joint_corridor_report(
    start: np.ndarray,
    target: np.ndarray,
    dense_samples: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float]:
    """Measure a proposed command chord against its signed dense-path span."""

    samples = np.asarray(dense_samples, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1] != start.shape[0]
        or samples.shape[0] < 1
    ):
        raise ValueError("joint corridor samples must be a non-empty N x D array")
    delta = np.asarray(target, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    squared_length = float(np.dot(delta, delta))
    if squared_length <= float(tolerance) ** 2:
        progress = np.zeros(samples.shape[0], dtype=np.float64)
        deviations = np.linalg.norm(samples - start[None, :], ord=np.inf, axis=1)
    else:
        progress = (samples - start[None, :]) @ delta / squared_length
        projections = start[None, :] + np.clip(progress, 0.0, 1.0)[:, None] * delta
        deviations = np.linalg.norm(samples - projections, ord=np.inf, axis=1)
    progress_deltas = np.diff(progress)
    max_regression = (
        max(0.0, float(-np.min(progress_deltas)))
        if progress_deltas.size
        else 0.0
    )
    return {
        "max_deviation_rad": float(np.max(deviations)),
        "max_progress_regression": float(max_regression),
        "min_progress": float(np.min(progress)),
        "max_progress": float(np.max(progress)),
    }


def _finite_positive(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def schedule_execution_waypoints(
    waypoints: Sequence[Sequence[float]],
    *,
    start_q: Sequence[float] | None = None,
    max_steps: int,
    available_time_s: float | None = None,
    observation_dt_s: float | None = None,
    max_joint_step_rad: float,
    max_speedup_factor: float | None = 2.5,
    preferred_max_steps: int | None = None,
    target_duration_s: float | None = None,
    mandatory_indices: Sequence[int] | None = None,
    max_joint_corridor_deviation_rad: float | None = None,
    safety_time_steps: int = 2,
    j8_index: int | None = None,
    j8_value: float = 0.0,
    lock_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Build a bounded monotone execution subsequence.

    The dense input is assumed to have already passed local Cartesian/IK
    validation.  With no preferred latency inputs the legacy behavior is
    preserved: the input is returned byte-for-byte (apart from an exact
    locked-joint value) whenever it fits the hard action/time budget.

    ``preferred_max_steps`` and ``target_duration_s`` opt into certified
    latency compression even when the dense path fits the hard budget.
    Compression retains every ``mandatory_indices`` entry and, when requested,
    admits a shortcut only if all skipped signed samples stay within the
    joint-space chord corridor.  A preferred budget is soft: if it cannot be
    met inside the requested safety cap, the safest available monotone
    subsequence is returned and explicitly marked as over target.
    """

    if isinstance(max_steps, bool) or int(max_steps) != max_steps or int(max_steps) < 1:
        raise ValueError("max_steps must be a positive integer")
    max_steps = int(max_steps)
    base_step = _finite_positive(max_joint_step_rad, label="max_joint_step_rad")
    assert base_step is not None
    speedup = _finite_positive(max_speedup_factor, label="max_speedup_factor")
    if speedup is not None and speedup < 1.0:
        raise ValueError("max_speedup_factor must be at least one")
    if isinstance(safety_time_steps, bool) or int(safety_time_steps) != safety_time_steps:
        raise ValueError("safety_time_steps must be an integer")
    safety_time_steps = max(0, int(safety_time_steps))
    matrix = _locked_matrix(
        _waypoint_matrix(waypoints),
        j8_index=j8_index,
        j8_value=j8_value,
        tolerance=lock_tolerance,
    )
    if preferred_max_steps is not None:
        if (
            isinstance(preferred_max_steps, bool)
            or int(preferred_max_steps) != preferred_max_steps
            or int(preferred_max_steps) < 1
        ):
            raise ValueError("preferred_max_steps must be a positive integer")
        preferred_max_steps = int(preferred_max_steps)
    target_duration = _finite_positive(
        target_duration_s,
        label="target_duration_s",
    )
    corridor_tolerance = _finite_positive(
        max_joint_corridor_deviation_rad,
        label="max_joint_corridor_deviation_rad",
    )
    normalized_mandatory: list[int] = []
    if mandatory_indices is not None:
        for raw_index in mandatory_indices:
            if isinstance(raw_index, bool) or int(raw_index) != raw_index:
                raise ValueError("mandatory_indices must contain integers")
            index = int(raw_index)
            if index < 0 or index >= matrix.shape[0]:
                raise ValueError("mandatory waypoint index is out of range")
            normalized_mandatory.append(index)
    normalized_mandatory = sorted(
        set(normalized_mandatory) | {int(matrix.shape[0] - 1)}
    )
    if start_q is None:
        start = matrix[0].copy()
    else:
        start = np.asarray(start_q, dtype=np.float64).reshape(-1)
        if start.shape[0] != matrix.shape[1] or not np.all(np.isfinite(start)):
            raise ValueError("start_q must be a finite vector matching waypoints")
        if j8_index is not None and abs(float(start[int(j8_index)]) - float(j8_value)) > lock_tolerance:
            raise ValueError("start_q violates the locked joint contract")

    dt = _finite_positive(observation_dt_s, label="observation_dt_s")
    available = _finite_positive(available_time_s, label="available_time_s")
    if target_duration is not None and dt is None:
        raise ValueError("target_duration_s requires observation_dt_s")
    time_budget_steps: int | None = None
    if dt is not None and available is not None:
        usable = float(available) - float(safety_time_steps) * float(dt)
        time_budget_steps = int(math.floor(usable / float(dt) + 1.0e-9))
    budget_steps = max_steps if time_budget_steps is None else min(max_steps, time_budget_steps)
    if budget_steps < 1:
        return {
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": "remaining evaluator budget cannot accommodate one action",
            "budget_steps": int(budget_steps),
            "time_budget_steps": time_budget_steps,
            "input_waypoint_count": int(matrix.shape[0]),
            "selected_waypoint_count": 0,
            "max_joint_step_rad": float(base_step),
        }

    preferred_time_steps: int | None = None
    if target_duration is not None and dt is not None:
        preferred_time_steps = max(
            1,
            int(math.floor(target_duration / dt + 1.0e-9)) - safety_time_steps,
        )
    preferred_candidates = [int(budget_steps)]
    if preferred_max_steps is not None:
        preferred_candidates.append(int(preferred_max_steps))
    if preferred_time_steps is not None:
        preferred_candidates.append(int(preferred_time_steps))
    preferred_budget_steps = max(1, min(preferred_candidates))

    input_steps = int(matrix.shape[0])
    original_deltas = np.vstack((matrix[0:1] - start, np.diff(matrix, axis=0)))
    original_max_step = float(np.max(np.linalg.norm(original_deltas, axis=1)))
    latency_compression_requested = bool(
        preferred_max_steps is not None or target_duration is not None
    )
    if (
        input_steps <= budget_steps
        and original_max_step <= base_step + lock_tolerance
        and (
            not latency_compression_requested
            or input_steps <= preferred_budget_steps
        )
    ):
        selected = matrix.copy()
        return {
            "ok": True,
            "mode": "legacy_identity",
            "reason": "original_trajectory_within_budget",
            "waypoints": selected.astype(float).tolist(),
            "selected_indices": list(range(input_steps)),
            "dropped_waypoint_count": 0,
            "input_waypoint_count": input_steps,
            "selected_waypoint_count": input_steps,
            "budget_steps": int(budget_steps),
            "preferred_budget_steps": int(preferred_budget_steps),
            "preferred_budget_met": True,
            "preferred_time_steps": preferred_time_steps,
            "time_budget_steps": time_budget_steps,
            "max_joint_step_rad": float(base_step),
            "actual_max_joint_step_rad": float(original_max_step),
            "speedup_factor": 1.0,
            "estimated_duration_s": None if dt is None else float(input_steps * dt),
            "mandatory_indices": normalized_mandatory,
            "mandatory_indices_retained": True,
            "joint_corridor": {
                "checked": False,
                "tolerance_rad": corridor_tolerance,
                "max_deviation_rad": 0.0,
                "max_progress_regression": 0.0,
                "transition_count": int(input_steps),
            },
        }

    if speedup is None:
        # The signed path has already been checked knot-by-knot for branch
        # continuity.  A relaxed execution pass may therefore skip samples
        # without inventing a new joint state; the smallest cap that can reach
        # the endpoint is the path-derived bound below.  The caller still
        # rechecks every selected state with local FK before emitting an action.
        path_displacements = np.linalg.norm(
            matrix - start[None, :],
            ord=np.inf,
            axis=1,
        )
        maximum_cap = max(
            float(base_step),
            float(np.max(path_displacements)),
        )
    else:
        maximum_cap = float(base_step) * float(speedup)
    max_selection = _select_with_step_cap(
        matrix,
        start=start,
        cap=maximum_cap,
        tolerance=lock_tolerance,
        mandatory_indices=normalized_mandatory,
        corridor_tolerance=corridor_tolerance,
    )
    if max_selection is None or len(max_selection[1]) > budget_steps:
        required = None if max_selection is None else len(max_selection[1])
        return {
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": (
                "verified trajectory cannot fit the remaining action/time budget "
                + (
                    f"within the {speedup:g}x joint-step safety cap"
                    if speedup is not None
                    else "using the path-derived joint-step cap"
                )
            ),
            "budget_steps": int(budget_steps),
            "time_budget_steps": time_budget_steps,
            "input_waypoint_count": input_steps,
            "selected_waypoint_count": 0 if required is None else int(required),
            "required_waypoints_at_max_speedup": required,
            "max_joint_step_rad": float(maximum_cap),
            "base_max_joint_step_rad": float(base_step),
        }

    preferred_budget_met = len(max_selection[1]) <= preferred_budget_steps
    selection_budget = (
        preferred_budget_steps if preferred_budget_met else budget_steps
    )
    if preferred_budget_met:
        low = float(base_step)
        high = float(maximum_cap)
        for _ in range(32):
            middle = 0.5 * (low + high)
            trial = _select_with_step_cap(
                matrix,
                start=start,
                cap=middle,
                tolerance=lock_tolerance,
                mandatory_indices=normalized_mandatory,
                corridor_tolerance=corridor_tolerance,
            )
            if trial is not None and len(trial[1]) <= selection_budget:
                high = middle
            else:
                low = middle
        selected_result = _select_with_step_cap(
            matrix,
            start=start,
            cap=high + lock_tolerance,
            tolerance=lock_tolerance,
            mandatory_indices=normalized_mandatory,
            corridor_tolerance=corridor_tolerance,
        )
    else:
        high = float(maximum_cap)
        selected_result = max_selection
    cap_search_fallback = False
    if selected_result is None or len(selected_result[1]) > budget_steps:
        # The greedy count is normally monotone in the cap.  A path that loops
        # within the corridor can violate that assumption at a binary-search
        # boundary, though, so retain the already-proved maximum-cap schedule
        # instead of turning a safe legacy execution into a false failure.
        selected_result = max_selection
        high = float(maximum_cap)
        cap_search_fallback = True
    selected_matrix, indices, actual_max, corridor_report = selected_result
    mandatory_retained = all(index in indices for index in normalized_mandatory)
    if not mandatory_retained:
        return {
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": "execution schedule dropped a mandatory signed waypoint",
            "mandatory_indices": normalized_mandatory,
            "selected_indices": [int(index) for index in indices],
        }
    return {
        "ok": True,
        "mode": (
            "adaptive_monotone_subsample"
            if speedup is not None
            else "adaptive_monotone_subsample_relaxed"
        ),
        "reason": (
            "preferred_latency_budget_met"
            if preferred_budget_met and latency_compression_requested
            else (
                "preferred_latency_budget_unmet_safe_fallback"
                if latency_compression_requested
                else "original_trajectory_exceeds_remaining_budget"
            )
        ),
        "waypoints": selected_matrix.astype(float).tolist(),
        "selected_indices": [int(index) for index in indices],
        "dropped_waypoint_count": int(input_steps - len(indices)),
        "input_waypoint_count": input_steps,
        "selected_waypoint_count": int(len(indices)),
        "budget_steps": int(budget_steps),
        "preferred_budget_steps": int(preferred_budget_steps),
        "preferred_budget_met": bool(
            len(indices) <= preferred_budget_steps
        ),
        "preferred_time_steps": preferred_time_steps,
        "time_budget_steps": time_budget_steps,
        "max_joint_step_rad": float(high),
        "actual_max_joint_step_rad": float(actual_max),
        "base_max_joint_step_rad": float(base_step),
        "speedup_factor": float(high / base_step),
        "speedup_cap": (
            None if speedup is None else float(speedup)
        ),
        "estimated_duration_s": None if dt is None else float(len(indices) * dt),
        "mandatory_indices": normalized_mandatory,
        "mandatory_indices_retained": bool(mandatory_retained),
        "joint_corridor": corridor_report,
        "certified_sparse_execution": bool(
            mandatory_retained
            and (
                corridor_tolerance is None
                or bool(corridor_report.get("checked"))
            )
        ),
        "cap_search_fallback_used": bool(cap_search_fallback),
    }


__all__ = ["schedule_execution_waypoints"]
