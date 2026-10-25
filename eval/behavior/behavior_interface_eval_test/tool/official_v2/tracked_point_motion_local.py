"""Submission-local endpoint and path planning for named tracked points."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contract import (
    ARM_DOF,
    MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M,
    normalize_tracked_target_coordinate,
    normalize_tracked_inequalities,
    parse_tracked_affine_expression,
)
from .eef_adjustment_local import (
    normalize_quaternion,
    orientation_error_deg,
    orientation_error_vector,
    solve_pose_target,
)
from .grasp_geometry_local import mat_to_quat_xyzw, quat_to_mat_xyzw
from .grasp_kinematics_local import (
    LocalRobotState,
    arm_joint_limits,
    eef_pose,
)
from .tracked_point_constraints_local import (
    COLLINEAR_POSITION_TOLERANCE_M,
    COLLINEAR_RELATION_TYPES,
    MAX_TRACKED_POINTS,
    ORDERED_COLLINEAR_INNER_MARGIN_M,
    _normalize_relation_records,
    geometry_preflight,
    geometry_rank,
    kabsch_rigid_transform,
    resolve_fully_numeric_target_points,
    evaluate_relations,
    relation_residual_blocks,
    relation_position_tolerance_m,
    rigid_geometry_report,
    signed_volume_status,
)
from .trunk_vertical_lift_local import TRUNK_JOINT_LIMITS


AXES = ("x", "y", "z")


def _strict_points(
    raw: Sequence[Sequence[float]],
    *,
    label: str,
    max_points: int = MAX_TRACKED_POINTS,
) -> np.ndarray:
    """Parse a structural N x 3 point matrix without flattening it."""

    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an N x 3 array") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must be an N x 3 array")
    if not 1 <= points.shape[0] <= int(max_points):
        raise ValueError(f"{label} must contain one to {int(max_points)} points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must contain finite coordinates")
    return points.copy()
# Try a coarse, geometry-independent Cartesian continuation first.  Local FK
# verifies the complete interpolated path, and deterministic finer densities
# remain available when the coarse continuation cannot stay on one IK branch.
DEFAULT_CARTESIAN_TRANSLATION_STEP_M = 0.050
DEFAULT_CARTESIAN_ORIENTATION_STEP_DEG = 10.0
CARTESIAN_SAMPLING_DENSITIES = (
    (DEFAULT_CARTESIAN_TRANSLATION_STEP_M, DEFAULT_CARTESIAN_ORIENTATION_STEP_DEG),
    (0.030, 6.0),
    (0.015, 3.0),
)
# Refining the same Cartesian curve is useful for a marginal interpolation
# error.  It cannot justify repeating expensive IK work when the locally
# verified joint interpolation misses that curve by several times the allowed
# corridor; the caller already has a bounded same-branch joint fallback for
# exactly that case.
CARTESIAN_REFINEMENT_MAX_DEVIATION_RATIO = 2.0
DEFAULT_MAX_BRANCH_JUMP_RAD = 1.20
# A previous accepted knot is already on the desired IK branch.  A short
# continuation solve is therefore a safe first attempt for live planning;
# the historical multi-seed solve remains the deterministic fallback when the
# continuation is not locally valid.
FAST_CARTESIAN_CONTINUATION_MAX_ITERATIONS = 64
# Give the observation-driven joint controller enough samples to shed speed
# before the final pose.  This is one global monotone envelope; it does not
# introduce internal stops or online waypoint retries.
GLOBAL_TIME_RAMP_FRACTION = 0.25
MAX_CARTESIAN_PROGRESS_REGRESSION_M = 0.001


class _CartesianPathDeviation(ValueError):
    """A measured FK path deviation with a numeric acceptance limit."""

    def __init__(self, message: str, *, measured: float, limit: float) -> None:
        super().__init__(message)
        self.measured = float(measured)
        self.limit = float(limit)

    @property
    def ratio(self) -> float:
        return self.measured / max(self.limit, 1.0e-12)
# These are resource bounds for the model-independent symbolic IK search.  They
# deliberately do not depend on point names, coordinate labels, or submitted
# constants.  A few independent endpoint branches are retained so Cartesian
# path planning can choose a branch that remains continuous from the capture.
GENERIC_PRIMARY_SEED_COUNT = 24
# Keep several independently seeded branches for Cartesian path validation.
# Underconstrained transforms can give a locally close endpoint a poor path,
# while a broader joint-space seed remains continuous.  This is a fixed
# resource bound, not a target-pattern selector.
MAX_GENERIC_ENDPOINT_CANDIDATES = 8
# A pathological unreachable target must not make a request spend unbounded
# time exploring every bounded joint seed.  The budget is shared by all
# residual evaluations, independent of coordinate names and values.
GENERIC_MAX_FUNCTION_EVALUATIONS = 8000
# If the feasibility search cannot enter the precise 3 mm / 5 degree class,
# spend a small deterministic budget on the exact public best-effort metric.
# This is deliberately separate from the least-squares budget: least-squares
# supplies robust basins, while this final pass optimizes the requested
# nonsmooth lexicographic fallback directly.
BEST_EFFORT_REFINEMENT_SEED_COUNT = 3
BEST_EFFORT_REFINEMENT_MAX_FUNCTION_EVALUATIONS_PER_SEED = 600
RELATION_PRECISE_MOTION_REFINEMENT_SEED_COUNT = 3
RELATION_PRECISE_MOTION_REFINEMENT_MAX_ITERATIONS_PER_SEED = 120
NOOP_RESIDUAL_M = 1e-6
# The point equations are often underdetermined (one shared coordinate leaves
# six joint-space degrees of freedom).  Anchor every generic solve to the
# capture posture so scipy cannot reduce an already negligible geometric
# residual by drifting along that null space.  The geometric residual is
# normalized by the public tolerance; this dimensionless weight is therefore
# small enough to preserve reachable equalities while still selecting the
# minimum-motion local solution.
ENDPOINT_JOINT_REGULARIZATION_WEIGHT = 0.02
# At every geometry rank, weakly prefer tracked points near their captured
# values after satisfying the submitted equations.  This all-rank term is a
# tie-break, not the rank-zero nearest-free-coordinate guarantee below.
# Point deltas are in metres while equality residuals are divided by the public
# tolerance, so a unit weight cannot materially trade away a valid equation.
ENDPOINT_TRACKED_POINT_REGULARIZATION_WEIGHT = 1.0
# Rank-zero tracked geometry has no point-set orientation with which to remove
# translational null-space drift.  Penalize only the affine constraint model's
# mathematically free point-coordinate directions at the same normalized scale
# as the submitted equations.  This is representation-independent: ``?`` and
# a variable used once induce the same null space, while a fully numeric target
# has a zero-dimensional free projector.
ENDPOINT_NEAREST_FREE_COORDINATE_WEIGHT = 1.0
# A rank-1 (collinear) bundle leaves rotation about its connecting line free.
# Prefer the smallest EEF rotation in that remaining degree of freedom so the
# Cartesian continuation does not add an arbitrary wrist twist.
ENDPOINT_EEF_ORIENTATION_REGULARIZATION_WEIGHT = 0.02
# If the nearest endpoint has no continuous Cartesian IK continuation, retry
# the same equations with a stronger preference for the minimum-angle member
# of the free axial-rotation family.  This is a fixed generic fallback, not a
# selector based on point names, variable names, or coordinate values.
ENDPOINT_PATH_FALLBACK_ORIENTATION_REGULARIZATION_WEIGHT = 0.20
# Endpoint search uses at most 3 mm / 5 degrees for its general precision
# class.  Collinear relations use a dedicated 1 mm position frontier without
# tightening unrelated XYZ or geometric constraints.
# Motion is a secondary preference only inside the resulting class.  If no
# such solution exists, every finite IK result remains a best-effort candidate
# and is ordered by millimetres plus degrees.
ENDPOINT_PRECISE_POSITION_TOLERANCE_M = 0.003
ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG = 5.0
CANDIDATE_BEST_EFFORT_SCORE_OFFSET = 1.0e6
CANDIDATE_JOINT_MOTION_WEIGHT = 0.02
# A commanded endpoint needs room to decelerate before a static joint stop.
# New motion from an interior posture therefore stays at least this far from
# either limit.  A capture already inside that margin remains admissible, but
# the generic null-space objective still prefers moving it back inward.
ENDPOINT_NEW_LIMIT_SAFETY_MARGIN_RAD = 0.08
ENDPOINT_LIMIT_AVOIDANCE_RESIDUAL_WEIGHT = 0.03
# The public contract checks each independent coordinate relation with an
# L-infinity tolerance.  A high even power makes the local least-squares
# objective approximate that same norm instead of averaging a violation across
# several axes (which can incorrectly reject an otherwise feasible endpoint).
POSITION_RESIDUAL_NORM_POWER = 4
POSITION_RESIDUAL_FALLBACK_POWER = 8
POSITION_RESIDUAL_PRIMARY_POWER = 2

# Whole-body fallback is entered only when the existing fixed-trunk arm solve
# cannot reach the strict per-constraint position / 5 degree precision class.
# Collinearity uses 1 mm while other position relations retain 3 mm. Geometry and
# minimum motion are intentionally separate: each nonlinear solve below sees
# only the submitted geometric equalities, and motion is used only to rank
# candidates that have already been evaluated by the authoritative checker.
WHOLE_BODY_MAX_FUNCTION_EVALUATIONS = 6000
WHOLE_BODY_DIRECT_MAX_FUNCTION_EVALUATIONS = 240
WHOLE_BODY_POSE_MAX_FUNCTION_EVALUATIONS = 96
WHOLE_BODY_LINE_ROLL_SAMPLE_DEG = 30.0
# A whole-body endpoint can have a genuine rigid-pose null space.  Keep a
# wider, still bounded frontier than the ordinary minimum-motion arm solver so
# downstream RGB-D scoring can inspect alternate proper-rotation branches.
WHOLE_BODY_ENDPOINT_FRONTIER_CANDIDATES = 64
WHOLE_BODY_POSE_FAMILY_ANGLE_OFFSETS_DEG = (
    0.0,
    30.0,
    -30.0,
    60.0,
    -60.0,
    90.0,
    -90.0,
    120.0,
    -120.0,
    150.0,
    -150.0,
    180.0,
)
WHOLE_BODY_POSE_FAMILY_MAX_FUNCTION_EVALUATIONS = 72
# ``least_squares`` endpoints are accepted after the ordinary 3 mm/5 degree
# precision check.  Their coordinates can therefore differ from the exact
# affine pose-family equations by a few floating-point/solver micrometres.
# Do not use an exact-zero test when expanding a family: it would discard a
# valid wrist-flipped member merely because the centre solve stopped at (for
# example) 10 micrometres.  This is only a *construction* tolerance; every
# generated IK result is still run through ``evaluate_target_constraints`` and
# the normal precision ranking before it can be selected.
WHOLE_BODY_POSE_FAMILY_NUMERICAL_EQUATION_TOLERANCE_M = 1.0e-4
# Plan-mode RGB-D overlap is evaluated after endpoint accuracy.  A mathematically
# exact pose can still occupy a handful of environment voxels because the
# frozen depth lattice is quantized.  Search a small, deterministic precision
# shell around exact poses so that a nearby pose which remains inside the
# strict 3 mm / 5 degree class can be evaluated before best-effort fallback.
# The shell is opt-in from the caller and is never used by exec-mode planning.
_WHOLE_BODY_PRECISION_SHELL_CARDINAL_OFFSETS_MM = tuple(
    tuple(
        sign * magnitude if component == axis else 0.0
        for component in range(3)
    )
    for magnitude in (3.0, 2.0, 1.0)
    for axis in range(3)
    for sign in (1.0, -1.0)
)
_WHOLE_BODY_PRECISION_SHELL_FACE_DIAGONAL_OFFSETS_MM = tuple(
    tuple(
        0.0
        if component == zero_axis
        else 4.0 * (first_sign if component == first_axis else second_sign)
        for component in range(3)
    )
    for zero_axis in range(3)
    for first_axis, second_axis in (
        ((zero_axis + 1) % 3, (zero_axis + 2) % 3),
    )
    for first_sign in (1.0, -1.0)
    for second_sign in (1.0, -1.0)
)
WHOLE_BODY_PRECISION_SHELL_TRANSLATION_OFFSETS_MM = (
    _WHOLE_BODY_PRECISION_SHELL_CARDINAL_OFFSETS_MM
    + _WHOLE_BODY_PRECISION_SHELL_FACE_DIAGONAL_OFFSETS_MM
)
# Keep rotation samples after all cardinal translations.  They are useful for
# collision alternatives, but translations are cheaper and cover the common
# voxel-boundary case first.
WHOLE_BODY_PRECISION_SHELL_ROTATION_OFFSETS_DEG = tuple(
    (axis, sign * magnitude)
    for magnitude in (1.0, 2.0, 3.0)
    for axis in range(3)
    for sign in (1.0, -1.0)
)
WHOLE_BODY_PRECISION_SHELL_MAX_POSES = 36
WHOLE_BODY_PRECISION_SHELL_MAX_FUNCTION_EVALUATIONS = 12


def _constraints_numerically_satisfied(
    report: Mapping[str, Any],
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
) -> bool:
    """Return whether a complete constraint report is effectively a no-op.

    ``ok`` is deliberately insufficient here: it means the request is inside
    the public acceptance tolerance, not that the captured pose already
    satisfies every requested relation.  In particular, a vector or plane
    relation can have zero positional residual while still requiring a finite
    orientation change.
    """

    if not isinstance(report, Mapping) or not bool(report.get("ok")):
        return False
    try:
        position_tolerance = float(pos_tol_m)
        orientation_tolerance = float(ori_tol_deg)
        position_error = float(report.get("max_constraint_error_m", math.inf))
        relation_error = float(report.get("max_relation_error_rad", 0.0))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
        or not math.isfinite(position_error)
        or position_error < 0.0
        or not math.isfinite(relation_error)
        or relation_error < 0.0
    ):
        return False
    numerical_position_limit = max(
        NOOP_RESIDUAL_M,
        1.0e-3 * position_tolerance,
    )
    numerical_relation_limit = max(
        1.0e-8,
        1.0e-3 * math.radians(orientation_tolerance),
    )
    if (
        position_error > numerical_position_limit
        or relation_error > numerical_relation_limit
    ):
        return False
    relation_blocks = report.get("relations", [])
    if relation_blocks is None:
        relation_blocks = []
    if not isinstance(relation_blocks, (list, tuple)):
        return False
    return not any(
        isinstance(block, Mapping) and bool(block.get("degenerate"))
        for block in relation_blocks
    )


class _SolverBudgetExceeded(RuntimeError):
    """Internal stop signal for an exhausted model-independent search budget."""


class PlanningDeadlineExceeded(RuntimeError):
    """Cooperative soft deadline for local planning.

    A deadline is deliberately distinct from cancellation: a valid endpoint
    that was found before the budget expired may still be routed through the
    verified joint-space fallback, whereas cancellation always aborts the
    request.  Callers must therefore catch this exception explicitly.
    """


def _normalize_planning_deadline(
    deadline_monotonic: float | None,
) -> float | None:
    if deadline_monotonic is None:
        return None
    try:
        value = float(deadline_monotonic)
    except (TypeError, ValueError) as exc:
        raise ValueError("planning deadline must be a finite monotonic timestamp") from exc
    if not math.isfinite(value):
        raise ValueError("planning deadline must be a finite monotonic timestamp")
    return value


def _check_planning_deadline(
    deadline_monotonic: float | None,
    *,
    message: str = "frozen tracked-point planning budget exhausted",
) -> None:
    if (
        deadline_monotonic is not None
        and time.monotonic() >= float(deadline_monotonic)
    ):
        raise PlanningDeadlineExceeded(message)


def _locked_q(raw: Sequence[float], *, label: str) -> np.ndarray:
    q = np.asarray(raw, dtype=np.float64).reshape(-1)
    if q.size != ARM_DOF or not np.all(np.isfinite(q)):
        raise ValueError(f"{label} must contain {ARM_DOF} finite values")
    q = q.copy()
    if ARM_DOF == 8:
        if abs(float(q[7])) > 1e-6:
            raise ValueError(f"{label} J8 must be zero")
        q[7] = 0.0
    return q


def _point_positions(
    state: LocalRobotState,
    arm: str,
    q: Sequence[float],
    anchors_eef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position, quaternion = eef_pose(state, arm, q)
    rotation = quat_to_mat_xyzw(quaternion)
    points = np.asarray(position, dtype=np.float64)[None, :] + (
        rotation @ np.asarray(anchors_eef, dtype=np.float64).T
    ).T
    return points, np.asarray(position), normalize_quaternion(quaternion)


def rigid_anchors_from_points(
    eef_position: Sequence[float],
    eef_quaternion: Sequence[float],
    points_robot_base_m: Sequence[Sequence[float]],
) -> np.ndarray:
    points = _strict_points(
        points_robot_base_m,
        label="tracked source points",
    )
    position = np.asarray(eef_position, dtype=np.float64).reshape(3)
    rotation = quat_to_mat_xyzw(eef_quaternion)
    return (rotation.T @ (points - position).T).T




@dataclass(frozen=True)
class _AffineForm:
    """One target coordinate as ``constant + sum(coeff * variable)``."""

    constant: float = 0.0
    coefficients: tuple[tuple[str, float], ...] = ()
    free: bool = False
    text: str | None = None

    def coefficient_map(self) -> dict[str, float]:
        return {str(name): float(value) for name, value in self.coefficients}

    def evaluate(self, values: Mapping[str, float]) -> float | None:
        if self.free:
            return None
        return float(
            self.constant
            + sum(
                float(coefficient) * float(values.get(name, 0.0))
                for name, coefficient in self.coefficients
            )
        )


def _parse_affine_text(raw: str, *, label: str) -> _AffineForm:
    text = str(raw).strip()
    if text == "?":
        return _AffineForm(free=True, text="?")
    constant, coefficients, compact = parse_tracked_affine_expression(
        text,
        label=label,
    )
    if not coefficients:
        return _AffineForm(constant=float(constant), text=compact)
    return _AffineForm(
        constant=float(constant),
        coefficients=tuple(sorted(coefficients.items())),
        text=compact,
    )


def _affine_form(raw: Any, *, label: str) -> _AffineForm:
    """Convert a validated/public coordinate into the planner's affine form."""

    normalized = normalize_tracked_target_coordinate(raw, label=label)
    if isinstance(normalized, Mapping):
        if normalized.get("free") is True and set(normalized) == {"free"}:
            return _AffineForm(free=True, text="?")
        if set(normalized) == {"var"}:
            name = str(normalized["var"])
            return _AffineForm(coefficients=((name, 1.0),), text=name)
        if set(normalized) == {"expr"}:
            return _parse_affine_text(
                str(normalized["expr"]),
                label=label,
            )
        raise ValueError(f"{label} has an unsupported coordinate form")
    return _AffineForm(constant=float(normalized), text=str(float(normalized)))


def _normalize_target_points(
    target_points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize planner input independently of the HTTP/API validator.

    The planner is also used directly by test-local benchmarks.  Keeping this
    small normalization step here means those callers get exactly the same
    number/variable/affine/free semantics as the public tool, rather than
    accidentally treating a symbolic string as a Python float.
    """

    if not isinstance(target_points, (list, tuple)):
        raise ValueError("target_points must contain one to six point objects")
    if not 1 <= len(target_points) <= 6:
        raise ValueError("target_points must contain one to six point objects")
    normalized_points: list[dict[str, Any]] = []
    names: set[str] = set()
    axes = AXES
    for index, point in enumerate(target_points):
        if not isinstance(point, Mapping):
            raise ValueError(f"target_points[{index}] must be an object")
        extra_fields = sorted(set(point) - {"name", "target_xyz_m"})
        if extra_fields:
            raise ValueError(
                f"target_points[{index}] has unsupported fields: "
                + ", ".join(str(field) for field in extra_fields)
            )
        raw_name = point.get("name")
        if not isinstance(raw_name, str):
            raise ValueError(
                f"target_points[{index}].name must be a non-empty string"
            )
        name = raw_name.strip()
        if not name:
            raise ValueError(f"target_points[{index}].name is required")
        if len(name) > 128:
            raise ValueError(f"target_points[{index}].name is too long")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError(
                f"target_points[{index}].name contains control characters"
            )
        if name in names:
            raise ValueError(f"duplicate target point name {name!r}")
        names.add(name)
        target = point.get("target_xyz_m")
        if isinstance(target, Mapping):
            if set(target) != set(axes):
                raise ValueError(
                    f"target_points[{index}].target_xyz_m must contain exactly x, y, z"
                )
            raw_coordinates = [target[axis] for axis in axes]
        elif isinstance(target, (list, tuple)) and len(target) == 3:
            raw_coordinates = list(target)
        else:
            raise ValueError(
                f"target_points[{index}].target_xyz_m must be an xyz object or length-3 list"
            )
        normalized_points.append(
            {
                "name": name,
                "target_xyz_m": {
                    axis: normalize_tracked_target_coordinate(
                        raw,
                        label=f"target_points[{index}].target_xyz_m.{axis}",
                    )
                    for axis, raw in zip(axes, raw_coordinates)
                },
            }
        )
    return normalized_points


@dataclass(frozen=True)
class _AffineConstraintRow:
    point_index: int
    axis_index: int
    form: _AffineForm


class _AffineConstraintModel:
    """Eliminate free coordinate variables before local joint optimization.

    For every constrained coordinate we have ``p = c + A v``.  Given a local
    FK point set, the least-squares ``v`` is eliminated analytically and the
    remaining projection residual is the complete set of geometric equality
    constraints.  This handles fixed values, repeated variables, affine
    offsets, and arbitrary combinations with one code path.
    """

    def __init__(self, target_points: Sequence[dict[str, Any]]) -> None:
        self.point_count = len(target_points)
        rows: list[_AffineConstraintRow] = []
        variables: set[str] = set()
        free_rows: list[tuple[int, int]] = []
        for point_index, point in enumerate(target_points):
            target = point.get("target_xyz_m")
            if isinstance(target, Mapping):
                target_values = target
            elif isinstance(target, (list, tuple)) and len(target) == 3:
                target_values = dict(zip(AXES, target))
            else:
                raise ValueError("target_xyz_m must be an xyz mapping or length-3 list")
            for axis_index, axis in enumerate(AXES):
                form = _affine_form(
                    target_values[axis],
                    label=(
                        f"points[{point_index}].target_xyz_m.{axis}"
                    ),
                )
                if form.free:
                    free_rows.append((point_index, axis_index))
                    continue
                rows.append(_AffineConstraintRow(point_index, axis_index, form))
                variables.update(name for name, _coefficient in form.coefficients)
        self.rows = tuple(rows)
        self.free_rows = tuple(free_rows)
        self.variable_names = tuple(sorted(variables))
        self._variable_index = {
            name: index for index, name in enumerate(self.variable_names)
        }
        self.constants = np.asarray(
            [row.form.constant for row in self.rows],
            dtype=np.float64,
        )
        matrix = np.zeros(
            (len(self.rows), len(self.variable_names)),
            dtype=np.float64,
        )
        for row_index, row in enumerate(self.rows):
            for name, coefficient in row.form.coefficients:
                matrix[row_index, self._variable_index[name]] = float(coefficient)
        self.matrix = matrix
        # Variable values are eliminated once per FK residual evaluation.  A
        # repeated ``lstsq`` factorization here used to dominate symbolic
        # three/four-point planning (the matrix is constant while only the
        # measured RHS changes).  Cache the equivalent minimum-norm linear
        # operator; this changes no constraint or acceptance semantics.
        if self.variable_names:
            # ``lstsq(..., rcond=None)`` uses machine precision scaled by the
            # larger matrix dimension.  Match that cutoff instead of relying
            # on ``pinv``'s version-dependent default so near-rank-deficient
            # affine inputs retain the old numerical behavior.
            lstsq_rcond = np.finfo(np.float64).eps * max(matrix.shape)
            self._variable_lstsq_operator = np.linalg.pinv(
                matrix,
                rcond=lstsq_rcond,
            )
        else:
            self._variable_lstsq_operator = np.zeros(
                (0, len(self.rows)),
                dtype=np.float64,
            )
        self.rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
        self.residual_dimension = max(0, len(self.rows) - self.rank)
        self._relation_matrix = self._build_relation_matrix()
        self.free_direction_projector = self._build_free_direction_projector()
        self.free_direction_dimension = int(
            np.linalg.matrix_rank(self.free_direction_projector)
        )

    def _build_relation_matrix(self) -> np.ndarray:
        """Build interpretable independent equations after variable elimination.

        A projection basis is numerically valid but splits a simple equality
        such as ``p0 == p1`` into two half-sized residuals.  Keeping a greedy
        independent set of variable rows lets each dependent row be expressed
        directly against that set, so the residual is in the same meters as
        the submitted constraint (``p1 - p0`` or ``p1 - p0 - offset``).
        """

        row_count = len(self.rows)
        if not row_count:
            return np.zeros((0, 0), dtype=np.float64)
        fixed_indices = [
            index
            for index, row in enumerate(self.rows)
            if not row.form.coefficients
        ]
        variable_indices = [
            index
            for index, row in enumerate(self.rows)
            if row.form.coefficients
        ]
        basis_indices: list[int] = []
        basis_matrix = np.zeros((0, len(self.variable_names)), dtype=np.float64)
        basis_rank = 0
        for index in variable_indices:
            candidate = np.vstack([basis_matrix, self.matrix[index]])
            candidate_rank = int(np.linalg.matrix_rank(candidate))
            if candidate_rank > basis_rank:
                basis_indices.append(index)
                basis_matrix = candidate
                basis_rank = candidate_rank
        dependent_indices = [
            index for index in variable_indices if index not in basis_indices
        ]
        relation_rows: list[np.ndarray] = []
        for index in fixed_indices:
            relation = np.zeros(row_count, dtype=np.float64)
            relation[index] = 1.0
            relation_rows.append(relation)
        if basis_indices:
            basis_coefficients = self.matrix[basis_indices]
            for index in dependent_indices:
                # Solve B.T * alpha = row_coefficients, then express this
                # dependent coordinate minus alpha times the basis coordinates.
                alpha, _residuals, _rank, _singular = np.linalg.lstsq(
                    basis_coefficients.T,
                    self.matrix[index],
                    rcond=None,
                )
                relation = np.zeros(row_count, dtype=np.float64)
                relation[index] = 1.0
                relation[basis_indices] -= np.asarray(alpha, dtype=np.float64)
                relation_rows.append(relation)
        return np.asarray(relation_rows, dtype=np.float64).reshape(
            len(relation_rows), row_count
        )

    def _build_free_direction_projector(self) -> np.ndarray:
        """Project point-coordinate motion onto the affine model's null space."""

        coordinate_count = 3 * self.point_count
        if not self._relation_matrix.size:
            return np.eye(coordinate_count, dtype=np.float64)
        jacobian = np.zeros(
            (self._relation_matrix.shape[0], coordinate_count),
            dtype=np.float64,
        )
        for row_index, row in enumerate(self.rows):
            coordinate_index = 3 * row.point_index + row.axis_index
            jacobian[:, coordinate_index] = self._relation_matrix[:, row_index]
        projector_rcond = np.finfo(np.float64).eps * max(jacobian.shape)
        constrained_projector = (
            np.linalg.pinv(jacobian, rcond=projector_rcond) @ jacobian
        )
        free_projector = np.eye(coordinate_count, dtype=np.float64) - constrained_projector
        # Suppress insignificant asymmetry/noise from the pseudoinverse so the
        # residual and reported dimension are deterministic across BLAS builds.
        free_projector = 0.5 * (free_projector + free_projector.T)
        free_projector[np.abs(free_projector) < 1.0e-12] = 0.0
        return free_projector

    @property
    def has_position_constraints(self) -> bool:
        return bool(self.residual_dimension)

    def _actual_values(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            values = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("tracked point positions must be an N x 3 array") from exc
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("tracked point positions must be an N x 3 array")
        if not np.all(np.isfinite(values)):
            raise ValueError("tracked point positions must be finite")
        if len(values) != self.point_count:
            raise ValueError("target/source point counts do not match")
        return np.asarray(
            [values[row.point_index, row.axis_index] for row in self.rows],
            dtype=np.float64,
        )

    def fit(
        self,
        points: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit affine variables and return physical per-coordinate corrections.

        ``rhs - A @ values`` is the displacement of each constrained
        coordinate from the nearest (least-squares) point in the submitted
        affine subspace.  It is deliberately returned instead of the reduced
        equation residual ``R @ rhs``: the latter changes magnitude when a
        caller rewrites an equivalent expression with a different coefficient
        scale (for example ``x`` versus ``2*x/2``).
        """
        actual = self._actual_values(points)
        if not len(self.rows):
            return np.empty((0,), dtype=np.float64), np.empty(
                (len(self.variable_names),), dtype=np.float64
            )
        rhs = actual - self.constants
        if self.variable_names:
            values = np.asarray(
                self._variable_lstsq_operator @ rhs,
                dtype=np.float64,
            ).reshape(-1)
        else:
            values = np.empty((0,), dtype=np.float64)
        coordinate_corrections = (
            rhs - self.matrix @ values
            if self.variable_names
            else rhs.copy()
        )
        if not np.all(np.isfinite(values)) or not np.all(
            np.isfinite(coordinate_corrections)
        ):
            raise ValueError("target constraint solve produced a non-finite value")
        return coordinate_corrections, values

    def equation_residuals(
        self,
        points: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return reduced algebraic residuals for diagnostics only.

        These values are intentionally never used as a physical acceptance
        metric because their scale depends on how an equivalent expression is
        written.
        """
        actual = self._actual_values(points)
        if not len(self.rows) or not self._relation_matrix.size:
            return np.empty((0,), dtype=np.float64)
        rhs = actual - self.constants
        residual = self._relation_matrix @ rhs
        if not np.all(np.isfinite(residual)):
            raise ValueError("target equation diagnostics are non-finite")
        return np.asarray(residual, dtype=np.float64).reshape(-1)

    def variable_values(self, values: Sequence[float]) -> dict[str, float]:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        return {
            name: float(vector[index])
            for index, name in enumerate(self.variable_names)
        }

    def target_value_map(
        self,
        values: Sequence[float],
    ) -> dict[tuple[int, int], float | None]:
        variable_values = self.variable_values(values)
        result: dict[tuple[int, int], float | None] = {
            row_key: None for row_key in self.free_rows
        }
        for row in self.rows:
            result[(row.point_index, row.axis_index)] = row.form.evaluate(
                variable_values
            )
        return result


def _build_constraint_model(
    target_points: Sequence[dict[str, Any]],
) -> _AffineConstraintModel:
    return _AffineConstraintModel(target_points)


def _inequality_report(
    inequalities: Sequence[Mapping[str, Any]],
    variable_values: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate canonical affine half-spaces at resolved coordinate values."""

    rows: list[dict[str, Any]] = []
    maximum_violation = 0.0
    minimum_clearance = math.inf
    for index, inequality in enumerate(inequalities):
        constant, coefficients, compact = parse_tracked_affine_expression(
            str(inequality["lhs"]),
            label=f"inequalities[{index}].lhs",
        )
        lhs_value = float(
            constant
            + sum(
                float(coefficient) * float(variable_values[name])
                for name, coefficient in coefficients.items()
            )
        )
        rhs_value = float(inequality["rhs"])
        operator = str(inequality["op"])
        signed_clearance = (
            lhs_value - rhs_value
            if operator == ">"
            else rhs_value - lhs_value
        )
        violation = max(
            0.0,
            float(MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M)
            - signed_clearance,
        )
        maximum_violation = max(maximum_violation, violation)
        minimum_clearance = min(minimum_clearance, signed_clearance)
        rows.append(
            {
                "lhs": compact,
                "op": operator,
                "rhs": rhs_value,
                "lhs_value_m": lhs_value,
                "signed_clearance_m": signed_clearance,
                "required_margin_m": float(
                    MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M
                ),
                "violation_m": violation,
                "ok": bool(violation <= 1.0e-12),
            }
        )
    return {
        "ok": all(bool(row["ok"]) for row in rows),
        "inequalities": rows,
        "max_inequality_violation_m": float(maximum_violation),
        "minimum_inequality_clearance_m": (
            None if not rows else float(minimum_clearance)
        ),
        "strict_inequality_margin_m": float(
            MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M
        ),
    }


def _inequality_violation_vector(
    inequalities: Sequence[Mapping[str, Any]],
    model: _AffineConstraintModel,
    fitted_values: Sequence[float],
) -> np.ndarray:
    report = _inequality_report(
        inequalities,
        model.variable_values(fitted_values),
    )
    return np.asarray(
        [float(row["violation_m"]) for row in report["inequalities"]],
        dtype=np.float64,
    )


def _target_slots(
    target_points: Sequence[dict[str, Any]],
) -> tuple[list[tuple[int, int, float]], dict[str, list[tuple[int, int]]]]:
    """Compatibility view of the affine model for older callers/tests."""

    model = _build_constraint_model(target_points)
    fixed: list[tuple[int, int, float]] = []
    variables: dict[str, list[tuple[int, int]]] = {}
    for row in model.rows:
        if not row.form.coefficients:
            fixed.append((row.point_index, row.axis_index, row.form.constant))
        else:
            for name, _coefficient in row.form.coefficients:
                variables.setdefault(name, []).append(
                    (row.point_index, row.axis_index)
                )
    return fixed, variables


def target_constraint_residuals(
    points_robot_base_m: Sequence[Sequence[float]],
    target_points: Sequence[dict[str, Any]],
) -> np.ndarray:
    model = _build_constraint_model(target_points)
    residuals, _values = model.fit(points_robot_base_m)
    return residuals


def evaluate_target_constraints(
    points_robot_base_m: Sequence[Sequence[float]],
    target_points: Sequence[dict[str, Any]],
    *,
    tolerance_m: float,
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    orientation_tolerance_deg: float = 5.0,
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    fixed_target_points: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    controlled_points = _strict_points(
        points_robot_base_m,
        label="tracked point positions",
    )
    controlled_target_points = _normalize_target_points(target_points)
    if controlled_points.shape[0] != len(controlled_target_points):
        raise ValueError("target/source point counts do not match")
    if (fixed_points_robot_base_m is None) != (fixed_target_points is None):
        raise ValueError(
            "fixed points and fixed target points must be supplied together"
        )
    fixed_points = np.empty((0, 3), dtype=np.float64)
    normalized_fixed_targets: list[dict[str, Any]] = []
    if fixed_points_robot_base_m is not None:
        fixed_points = _strict_points(
            fixed_points_robot_base_m,
            label="fixed tracked point positions",
        )
        normalized_fixed_targets = _normalize_target_points(
            fixed_target_points or []
        )
        if fixed_points.shape[0] != len(normalized_fixed_targets):
            raise ValueError("fixed target/source point counts do not match")
    if len(controlled_points) + len(fixed_points) > MAX_TRACKED_POINTS:
        raise ValueError(
            f"combined tracked constraints cannot exceed {MAX_TRACKED_POINTS} points"
        )
    controlled_names = [str(point["name"]) for point in controlled_target_points]
    fixed_names = [str(point["name"]) for point in normalized_fixed_targets]
    if set(controlled_names) & set(fixed_names):
        raise ValueError("controlled and fixed target point names overlap")
    points = (
        np.vstack([controlled_points, fixed_points])
        if len(fixed_points)
        else controlled_points
    )
    target_points = controlled_target_points + normalized_fixed_targets
    try:
        tolerance = float(tolerance_m)
        orientation_tolerance = float(orientation_tolerance_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("constraint tolerances must be numeric") from exc
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
    ):
        raise ValueError("constraint tolerances must be finite and positive")
    model = _build_constraint_model(target_points)
    canonical_inequalities = normalize_tracked_inequalities(
        inequalities or [],
        allowed_variables=set(model.variable_names),
    )
    residuals, fitted_values = model.fit(points)
    equation_residuals = model.equation_residuals(points)
    variable_values = model.variable_values(fitted_values)
    fixed_report: list[dict[str, Any]] = []
    affine_report: list[dict[str, Any]] = []
    for row in model.rows:
        point_index, axis_index, form = (
            row.point_index,
            row.axis_index,
            row.form,
        )
        actual = float(points[point_index, axis_index])
        target = form.evaluate(variable_values)
        if not form.coefficients:
            assert target is not None
            fixed_report.append(
                {
                    "point": str(target_points[point_index]["name"]),
                    "axis": AXES[axis_index],
                    "target_m": float(target),
                    "actual_m": actual,
                    "error_m": actual - float(target),
                }
            )
            continue
        affine_report.append(
            {
                "point": str(target_points[point_index]["name"]),
                "axis": AXES[axis_index],
                "expression": form.text,
                "actual_m": actual,
                "resolved_target_m": None if target is None else float(target),
                "residual_m": float(actual - target) if target is not None else 0.0,
            }
        )
    variable_report: dict[str, dict[str, Any]] = {}
    for name in model.variable_names:
        occurrences: list[dict[str, Any]] = []
        occurrence_errors: list[float] = []
        for row in model.rows:
            coefficient = dict(row.form.coefficients).get(name)
            if coefficient is None:
                continue
            actual = float(points[row.point_index, row.axis_index])
            expected = row.form.evaluate(variable_values)
            if expected is not None:
                occurrence_errors.append(abs(actual - expected))
            occurrences.append(
                {
                    "point": str(target_points[row.point_index]["name"]),
                    "axis": AXES[row.axis_index],
                    "actual_m": actual,
                    "coefficient": float(coefficient),
                    "constant_m": float(row.form.constant),
                    "expression": row.form.text,
                }
            )
        variable_report[name] = {
            "value_m": float(variable_values.get(name, 0.0)),
            "occurrence_count": len(occurrences),
            "max_equality_error_m": (
                float(max(occurrence_errors)) if occurrence_errors else 0.0
            ),
            "occurrences": occurrences,
        }
    residuals = np.asarray(residuals, dtype=np.float64)
    max_coordinate_error = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    target_values = model.target_value_map(fitted_values)
    resolved_target_points: dict[str, list[float | None]] = {}
    for index, point in enumerate(target_points):
        resolved_target_points[str(point["name"])] = [
            None
            if target_values.get((index, axis_index)) is None
            else float(target_values[(index, axis_index)])
            for axis_index in range(3)
        ]
    free_report = [
        {
            "point": str(target_points[point_index]["name"]),
            "axis": AXES[axis_index],
        }
        for point_index, axis_index in model.free_rows
    ]
    relation_report = evaluate_relations(
        points,
        relations,
        point_names=controlled_names + fixed_names,
        position_tolerance_m=tolerance,
        orientation_tolerance_deg=orientation_tolerance,
    )
    max_relation_position_error = float(
        relation_report.get("max_relation_error_m", 0.0)
    )
    inequality_report = _inequality_report(
        canonical_inequalities,
        variable_values,
    )
    max_inequality_violation = float(
        inequality_report["max_inequality_violation_m"]
    )
    max_error = max(
        max_coordinate_error,
        max_relation_position_error,
        max_inequality_violation,
    )
    return {
        "ok": bool(
            max_error <= tolerance
            and relation_report["ok"]
            and inequality_report["ok"]
        ),
        "axial_order_satisfied": relation_report.get("axial_order_satisfied", True),
        "tolerance_m": tolerance,
        "max_constraint_error_m": max_error,
        "max_coordinate_constraint_error_m": max_coordinate_error,
        "max_relation_error_m": max_relation_position_error,
        "max_collinear_error_m": relation_report.get("max_collinear_error_m"),
        "collinear_position_tolerance_m": relation_report.get(
            "collinear_position_tolerance_m"
        ),
        "collinear_constraints_ok": bool(
            relation_report.get("collinear_constraints_ok", True)
        ),
        "max_relation_error_rad": float(
            relation_report.get("max_relation_error_rad", 0.0)
        ),
        "max_relation_error_deg": float(
            relation_report.get("max_relation_error_deg", 0.0)
        ),
        "relations": relation_report.get("relations", []),
        "relation_constraints_ok": bool(relation_report.get("ok", True)),
        "inequality_constraints_ok": bool(inequality_report["ok"]),
        "inequalities": inequality_report["inequalities"],
        "max_inequality_violation_m": max_inequality_violation,
        "minimum_inequality_clearance_m": inequality_report[
            "minimum_inequality_clearance_m"
        ],
        "strict_inequality_margin_m": inequality_report[
            "strict_inequality_margin_m"
        ],
        "fixed_coordinates": fixed_report,
        "affine_constraints": affine_report,
        "variables": variable_report,
        "resolved_variables": variable_values,
        "free_coordinates": free_report,
        "constraint_row_count": int(len(model.rows)),
        "constraint_rank": int(model.rank),
        "constraint_residual_dimension": int(model.residual_dimension),
        "constraint_residuals_m": residuals.astype(float).tolist(),
        "affine_coordinate_corrections_m": residuals.astype(float).tolist(),
        "affine_equation_residuals_m": equation_residuals.astype(float).tolist(),
        "resolved_points_xyz_m": {
            str(point["name"]): points[index].astype(float).tolist()
            for index, point in enumerate(target_points)
        },
        "resolved_target_points_xyz_m": resolved_target_points,
        "controlled_point_count": int(len(controlled_points)),
        "fixed_point_count": int(len(fixed_points)),
    }


def rigid_pair_distance_tolerance(pos_tol_m: float) -> float:
    """Return the RGB-D tolerance for the N-point pair-distance invariant."""

    tolerance = float(pos_tol_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("pos_tol_m must be finite and positive")
    # Tracker depth has more noise than local FK, but identity validation must
    # not become arbitrarily loose when a caller requests a large goal
    # tolerance.  At the default 12 mm goal tolerance this permits 18 mm of
    # pair-distance noise and rejects the observed 30+ mm identity collapse.
    return float(min(0.020, max(0.012, 1.5 * tolerance)))


def evaluate_rigid_pair_consistency(
    measured_points_m: Sequence[Sequence[float]],
    reference_points_m: Sequence[Sequence[float]],
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Check all pairwise rigid invariants for one to six points.

    The historical function name is retained for API compatibility.  For
    three/four points the report is deliberately an N-point bundle, so a bad
    redundant marker cannot be hidden by averaging the other markers.
    """

    try:
        measured = np.asarray(measured_points_m, dtype=np.float64)
        reference = np.asarray(reference_points_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("measured and reference points must be N x 3 arrays") from exc
    if (
        measured.ndim != 2
        or reference.ndim != 2
        or measured.shape[1:] != (3,)
        or reference.shape[1:] != (3,)
    ):
        raise ValueError("measured and reference points must be N x 3 arrays")
    if measured.shape != reference.shape:
        raise ValueError("measured and reference point arrays must have the same shape")
    if measured.shape[0] < 1 or measured.shape[0] > 6:
        raise ValueError("rigid tracked-point consistency requires one to six points")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(reference)):
        raise ValueError("rigid tracked-point consistency requires finite points")
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rigid pair-distance tolerance must be finite and positive")
    report = rigid_geometry_report(reference, measured, tolerance_m=tolerance)
    # Depth noise is pointwise.  A pair distance can drift by roughly twice
    # that noise; retain the public tolerance as the lower bound while using
    # the same conservative allowance for every pair.
    pair_limit = tolerance
    pair_errors = [
        float(item["distance_error_m"])
        for item in report["pair_distances"]
    ]
    max_pair_error = max(pair_errors, default=0.0)
    volume_statuses: list[bool | None] = []
    for item in report["tetrahedron_volumes"]:
        volume_statuses.append(
            signed_volume_status(
                float(item["source_signed_volume_m3"]),
                float(item["target_signed_volume_m3"]),
                reference,
                measured,
                tolerance,
            )
        )
    volume_sign_ok = all(status is not False for status in volume_statuses)
    volume_sign_ambiguous = any(status is None for status in volume_statuses)
    kabsch = kabsch_rigid_transform(reference, measured)
    kabsch_errors = np.asarray(kabsch["per_point_error_m"], dtype=np.float64)
    max_kabsch_error = float(kabsch["max_error_m"])
    kabsch_limit = float(tolerance)
    kabsch_ok = bool(max_kabsch_error <= kabsch_limit)
    ok = bool(max_pair_error <= pair_limit and kabsch_ok and volume_sign_ok)
    # Keep legacy scalar fields for older one/two-point consumers while the
    # complete N-point reports remain authoritative.
    first_pair = report["pair_distances"][0] if report["pair_distances"] else None
    return {
        "applicable": bool(len(measured) >= 2),
        "ok": ok,
        "tolerance_m": tolerance,
        "pair_distance_tolerance_m": float(pair_limit),
        "reference_distance_m": (
            None if first_pair is None else first_pair["source_distance_m"]
        ),
        "measured_distance_m": (
            None if first_pair is None else first_pair["target_distance_m"]
        ),
        "distance_error_m": (
            0.0 if first_pair is None else first_pair["distance_error_m"]
        ),
        "max_pair_distance_error_m": float(max_pair_error),
        "kabsch_bundle_residual_m": kabsch_errors.astype(float).tolist(),
        "max_kabsch_bundle_residual_m": max_kabsch_error,
        "kabsch_bundle_tolerance_m": kabsch_limit,
        "kabsch_bundle_ok": kabsch_ok,
        "pair_distances": report["pair_distances"],
        "triangle_areas": report["triangle_areas"],
        "tetrahedron_volumes": report["tetrahedron_volumes"],
        "geometry_rank": report["target_rank"],
        "reference_geometry_rank": report["source_rank"],
        "reference_pair": report["reference_pair"],
        "reference_triangle": report["reference_triangle"],
        "improper_geometry": not volume_sign_ok,
        "volume_sign_ambiguous": volume_sign_ambiguous,
    }


def evaluate_rigid_geometry_consistency(
    measured_points_m: Sequence[Sequence[float]],
    reference_points_m: Sequence[Sequence[float]],
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Explicit N-point alias used by new callers and trajectory reports."""

    return evaluate_rigid_pair_consistency(
        measured_points_m,
        reference_points_m,
        tolerance_m=tolerance_m,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    unit /= max(float(np.linalg.norm(unit)), 1e-12)
    skew = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine > 1e-9:
        return _axis_rotation(cross / sine, math.atan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3, dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
    axis = np.cross(source, basis)
    return _axis_rotation(axis, math.pi)


def shared_line_target_pose_family(
    *,
    source_points_robot_base_m: Sequence[Sequence[float]],
    point_names: Sequence[str],
    relations: Sequence[Mapping[str, Any]],
    start_eef_position_m: Sequence[float],
    start_eef_quaternion_xyzw: Sequence[float],
    base_target_eef_quaternions_xyzw: Sequence[Sequence[float]],
    angular_offsets_deg: Sequence[float] = (-20.0, -10.0, 0.0, 10.0, 20.0),
) -> dict[str, Any]:
    """Enumerate the free rigid-pose family for concurrent line-point goals.

    Two nonparallel source lines constrained to pass through the same fixed
    target point determine a source pivot but leave the complete rigid
    rotation free.  Mapping that pivot to the fixed target therefore preserves
    the globally best line residual for every proper rotation.  This helper
    samples that genuine null space around already-found endpoint orientations;
    it never changes or approximates the submitted constraints.
    """

    source = _strict_points(
        source_points_robot_base_m,
        label="shared-line source positions",
    )
    names = [str(name).strip() for name in point_names]
    if len(names) != len(source) or any(not name for name in names):
        raise ValueError("shared-line point names do not match source positions")
    if len(set(names)) != len(names):
        raise ValueError("shared-line point names must be unique")
    canonical_relations = _normalize_relation_records(relations, names)
    name_to_index = {name: index for index, name in enumerate(names)}
    line_specs: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    target_points: list[np.ndarray] = []
    for relation in canonical_relations:
        if str(relation.get("type") or "") != "line_through_point":
            continue
        relation_names = [str(name) for name in relation.get("point_names", [])]
        if len(relation_names) != 2 or any(
            name not in name_to_index for name in relation_names
        ):
            continue
        first = source[name_to_index[relation_names[0]]]
        second = source[name_to_index[relation_names[1]]]
        axis = second - first
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-9:
            return {
                "ok": False,
                "reason": "shared_line_source_axis_degenerate",
                "poses": [],
            }
        target = np.asarray(
            relation.get("target_point_robot_base_m"), dtype=np.float64
        )
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("shared-line target point must be finite length 3")
        line_specs.append((first.copy(), axis / axis_norm, relation_names))
        target_points.append(target.copy())
    if len(line_specs) < 2:
        return {
            "ok": False,
            "reason": "shared_line_pose_family_requires_two_lines",
            "poses": [],
        }
    target_matrix = np.stack(target_points)
    target_point = np.mean(target_matrix, axis=0)
    target_spread = float(
        np.max(np.linalg.norm(target_matrix - target_point, axis=1))
    )
    if target_spread > 1.0e-6:
        return {
            "ok": False,
            "reason": "shared_line_targets_are_not_common",
            "target_point_spread_m": target_spread,
            "poses": [],
        }

    identity = np.eye(3, dtype=np.float64)
    projectors = [identity - np.outer(direction, direction) for _, direction, _ in line_specs]
    normal_matrix = np.sum(projectors, axis=0)
    normal_rhs = np.sum(
        [projector @ first for projector, (first, _direction, _names) in zip(projectors, line_specs)],
        axis=0,
    )
    singular_values = np.linalg.svd(normal_matrix, compute_uv=False)
    if singular_values[-1] <= 1.0e-6:
        return {
            "ok": False,
            "reason": "shared_line_source_axes_are_parallel",
            "normal_matrix_singular_values": singular_values.astype(float).tolist(),
            "poses": [],
        }
    source_pivot = np.linalg.solve(normal_matrix, normal_rhs)
    source_line_errors = np.asarray(
        [
            np.linalg.norm(projector @ (source_pivot - first))
            for projector, (first, _direction, _names) in zip(projectors, line_specs)
        ],
        dtype=np.float64,
    )

    start_position = np.asarray(start_eef_position_m, dtype=np.float64)
    start_quaternion = normalize_quaternion(start_eef_quaternion_xyzw)
    if start_position.shape != (3,) or not np.all(np.isfinite(start_position)):
        raise ValueError("shared-line start EEF position must be finite length 3")
    start_rotation = quat_to_mat_xyzw(start_quaternion)
    offset_values = sorted({float(value) for value in angular_offsets_deg})
    if (
        not offset_values
        or any(not math.isfinite(value) for value in offset_values)
        or any(abs(value) > 90.0 for value in offset_values)
    ):
        raise ValueError("shared-line angular offsets must be finite and within 90 degrees")
    offset_triples = [
        (x_deg, y_deg, z_deg)
        for x_deg in offset_values
        for y_deg in offset_values
        for z_deg in offset_values
    ]
    offset_triples.sort(
        key=lambda value: (
            max(abs(component) for component in value),
            sum(component * component for component in value),
            sum(abs(component) for component in value),
            value,
        )
    )

    poses: list[dict[str, Any]] = []
    seen_rotations: list[np.ndarray] = []
    for base_index, raw_quaternion in enumerate(
        base_target_eef_quaternions_xyzw
    ):
        base_quaternion = normalize_quaternion(raw_quaternion)
        base_rigid_rotation = quat_to_mat_xyzw(base_quaternion) @ start_rotation.T
        for x_deg, y_deg, z_deg in offset_triples:
            # scipy Rotation.from_euler("xyz") uses this extrinsic matrix
            # product.  Keeping it local avoids adding a dependency to this
            # otherwise analytic pose-family generator.
            delta_rotation = (
                _axis_rotation(np.asarray([0.0, 0.0, 1.0]), math.radians(z_deg))
                @ _axis_rotation(np.asarray([0.0, 1.0, 0.0]), math.radians(y_deg))
                @ _axis_rotation(np.asarray([1.0, 0.0, 0.0]), math.radians(x_deg))
            )
            rigid_rotation = delta_rotation @ base_rigid_rotation
            if any(
                float(np.linalg.norm(rigid_rotation - existing, ord="fro")) <= 1.0e-9
                for existing in seen_rotations
            ):
                continue
            seen_rotations.append(rigid_rotation.copy())
            target_eef_position = target_point + rigid_rotation @ (
                start_position - source_pivot
            )
            target_eef_quaternion = mat_to_quat_xyzw(
                rigid_rotation @ start_rotation
            )
            poses.append(
                {
                    "final_eef_position_m": target_eef_position.astype(float).tolist(),
                    "final_eef_quaternion_xyzw": target_eef_quaternion.astype(float).tolist(),
                    "pose_family": "shared_line_common_target_rigid_null_space",
                    "base_pose_index": int(base_index),
                    "local_euler_xyz_deg": [
                        float(x_deg),
                        float(y_deg),
                        float(z_deg),
                    ],
                    "invariant_max_line_error_m": float(
                        np.max(source_line_errors)
                    ),
                }
            )
    return {
        "ok": bool(poses),
        "reason": None if poses else "shared_line_pose_family_empty",
        "pose_family": "shared_line_common_target_rigid_null_space",
        "line_count": int(len(line_specs)),
        "line_point_names": [item[2] for item in line_specs],
        "source_pivot_robot_base_m": source_pivot.astype(float).tolist(),
        "target_pivot_robot_base_m": target_point.astype(float).tolist(),
        "target_point_spread_m": target_spread,
        "source_line_error_m": source_line_errors.astype(float).tolist(),
        "invariant_max_line_error_m": float(np.max(source_line_errors)),
        "normal_matrix_singular_values": singular_values.astype(float).tolist(),
        "base_pose_count": int(len(base_target_eef_quaternions_xyzw)),
        "angular_offsets_deg": [float(value) for value in offset_values],
        "pose_count": int(len(poses)),
        "poses": poses,
    }


def _candidate_score(
    q: np.ndarray,
    q_start: np.ndarray,
    current_points: np.ndarray,
    predicted_points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the secondary minimum-motion score for one endpoint.

    Joint-limit safety is enforced by the endpoint control bounds.  It must
    not be mixed into this score: doing so can make a distant posture appear
    to move less merely because it has more limit margin.
    """

    point_motion = float(np.mean(np.sum((predicted_points - current_points) ** 2, axis=1)))
    joint_span = np.maximum(upper[:7] - lower[:7], 0.25)
    joint_motion = float(np.mean(((q[:7] - q_start[:7]) / joint_span) ** 2))
    return point_motion + CANDIDATE_JOINT_MOTION_WEIGHT * joint_motion


def _candidate_accuracy(
    constraints: Mapping[str, Any],
    *,
    preserved_eef_orientation_error_deg: float,
    precise_position_tolerance_m: float = (
        ENDPOINT_PRECISE_POSITION_TOLERANCE_M
    ),
    precise_orientation_tolerance_deg: float = (
        ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
    ),
) -> dict[str, Any]:
    """Return the accuracy-first endpoint selection metrics.

    Position is expressed in millimetres and angular relations in degrees so
    the best-effort scalar has the exact units and ordering exposed by the
    public diagnostics.  A single-anchor request additionally preserves its
    capture-time EEF orientation; multi-point orientation comes from explicit
    relation geometry such as line or plane alignment.
    """

    position_error_m = float(
        constraints.get("max_constraint_error_m", math.inf)
    )
    coordinate_error_m = float(
        constraints.get("max_coordinate_constraint_error_m", position_error_m)
    )
    relation_blocks = constraints.get("relations", [])
    if not isinstance(relation_blocks, (list, tuple)):
        relation_blocks = []
    position_precision_ratios = [
        coordinate_error_m / max(float(precise_position_tolerance_m), 1.0e-12)
    ]
    inequality_violation_m = float(
        constraints.get("max_inequality_violation_m", 0.0)
    )
    position_precision_ratios.append(
        inequality_violation_m
        / max(float(precise_position_tolerance_m), 1.0e-12)
    )
    strictest_relation_tolerance = float(precise_position_tolerance_m)
    collinear_relation_present = False
    for block in relation_blocks:
        if not isinstance(block, Mapping) or block.get("kind") != "position":
            continue
        relation_type = str(block.get("type") or "").strip().lower()
        relation_tolerance = relation_position_tolerance_m(
            relation_type, precise_position_tolerance_m
        )
        relation_error = float(block.get("max_error", math.inf))
        position_precision_ratios.append(
            relation_error / max(relation_tolerance, 1.0e-12)
        )
        strictest_relation_tolerance = min(
            strictest_relation_tolerance, relation_tolerance
        )
        collinear_relation_present = bool(
            collinear_relation_present
            or relation_type in COLLINEAR_RELATION_TYPES
        )
    position_precision_ratio = max(position_precision_ratios, default=math.inf)
    relation_orientation_error_deg = float(
        constraints.get("max_relation_error_deg", 0.0)
    )
    orientation_error_deg_value = max(
        float(preserved_eef_orientation_error_deg),
        relation_orientation_error_deg,
    )
    if (
        not math.isfinite(position_error_m)
        or position_error_m < 0.0
        or not math.isfinite(position_precision_ratio)
        or position_precision_ratio < 0.0
        or not math.isfinite(orientation_error_deg_value)
        or orientation_error_deg_value < 0.0
    ):
        return {
            "position_error_m": math.inf,
            "position_error_mm": math.inf,
            "orientation_error_deg": math.inf,
            "accuracy_cost_mm_plus_deg": math.inf,
            "position_precision_ratio": math.inf,
            "position_precision_satisfied": False,
            "collinear_relation_present": bool(collinear_relation_present),
            "collinear_position_tolerance_m": (
                COLLINEAR_POSITION_TOLERANCE_M
                if collinear_relation_present
                else None
            ),
            "precise": False,
        }
    position_error_mm = 1000.0 * position_error_m
    precise = bool(
        position_precision_ratio <= 1.0
        and orientation_error_deg_value
        <= float(precise_orientation_tolerance_deg)
        and constraints.get("axial_order_satisfied", True)
        and constraints.get("inequality_constraints_ok", True)
    )
    return {
        "position_error_m": position_error_m,
        "position_error_mm": position_error_mm,
        "orientation_error_deg": orientation_error_deg_value,
        "accuracy_cost_mm_plus_deg": (
            position_error_mm + orientation_error_deg_value
        ),
        "position_precision_ratio": float(position_precision_ratio),
        "position_precision_satisfied": bool(position_precision_ratio <= 1.0),
        "precise_coordinate_tolerance_m": float(precise_position_tolerance_m),
        "strictest_relation_position_tolerance_m": float(
            strictest_relation_tolerance
        ),
        "collinear_relation_present": bool(collinear_relation_present),
        "collinear_position_tolerance_m": (
            float(
                relation_position_tolerance_m(
                    "line_coincident", precise_position_tolerance_m
                )
            )
            if collinear_relation_present
            else None
        ),
        "precise": precise,
    }


def _precision_class_label(accuracy: Mapping[str, Any]) -> str:
    if not bool(accuracy.get("precise")):
        return "best_effort"
    if bool(accuracy.get("collinear_relation_present")):
        return "precise_collinear_1mm_other_3mm_5deg"
    return "precise_3mm_5deg"


def _endpoint_selection_policy(relations: Sequence[Mapping[str, Any]]) -> str:
    has_collinear = any(
        str(relation.get("type") or "").strip().lower()
        in COLLINEAR_RELATION_TYPES
        for relation in relations
    )
    prefix = (
        "precise_collinear_1mm_other_3mm_5deg"
        if has_collinear
        else "precise_3mm_5deg"
    )
    return f"{prefix}_then_minimum_motion_else_minimum_position_mm_plus_orientation_deg"


def _candidate_rank_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    """Order precise endpoints by motion and all others by absolute accuracy."""

    precise = bool(report.get("precise"))
    accuracy = float(report.get("accuracy_cost_mm_plus_deg", math.inf))
    motion = float(report.get("motion_score", math.inf))
    position_mm = float(report.get("position_error_mm", math.inf))
    orientation_deg_value = float(
        report.get("selection_orientation_error_deg", math.inf)
    )
    if precise:
        return (0.0, motion, accuracy, position_mm, orientation_deg_value)
    return (1.0, accuracy, position_mm, orientation_deg_value, motion)


def _whole_body_endpoint_frontier(
    candidates: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
    ],
    *,
    primary_limit: int = MAX_GENERIC_ENDPOINT_CANDIDATES,
    frontier_limit: int = WHOLE_BODY_ENDPOINT_FRONTIER_CANDIDATES,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    """Retain the ordinary endpoint frontier plus distinct pose-family members.

    Endpoint ranking still determines the first (legacy) candidates.  A
    symbolic rigid bundle can, however, have a second proper EEF orientation
    with the same exact point constraints.  Such a member may be the only one
    that passes the RGB-D overlap gate, so it must not be discarded merely
    because its joint-motion score is larger than the first eight endpoints.
    The extra frontier is bounded and only recognizes the planner's explicit
    ``whole_body_affine_pose_family`` marker.
    """

    if isinstance(primary_limit, bool) or int(primary_limit) < 1:
        raise ValueError("whole-body primary endpoint limit must be positive")
    if isinstance(frontier_limit, bool) or int(frontier_limit) < int(primary_limit):
        raise ValueError(
            "whole-body endpoint frontier limit must cover the primary limit"
        )
    ordered = sorted(candidates, key=lambda item: _candidate_rank_key(item[3]))
    result = list(ordered[: int(primary_limit)])
    if len(result) >= int(frontier_limit):
        return result[: int(frontier_limit)]

    def q_signature(item: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(item[0], dtype=np.float64), decimals=7).tolist())

    seen = {q_signature(item) for item in result}
    family = [
        item
        for item in ordered[int(primary_limit) :]
        if str(item[3].get("mode") or "")
        in {
            "whole_body_affine_pose_family",
            "whole_body_precision_shell",
        }
    ]
    # Keep the half-turn and small-angle members deterministic while retaining
    # all distinct IK branches for a given angle when they are available.
    family.sort(
        key=lambda item: (
            abs(float(item[3].get("pose_family_angle_deg", 0.0)))
            == 180.0,
            abs(float(item[3].get("pose_family_angle_deg", 0.0))),
            0
            if str(item[3].get("mode") or "")
            == "whole_body_affine_pose_family"
            else 1,
            int(item[3].get("precision_shell_offset_index", 0)),
            int(item[3].get("pose_family_base_index", 0)),
            int(item[3].get("pose_family_seed_index", 0)),
            _candidate_rank_key(item[3]),
        )
    )
    for item in family:
        signature = q_signature(item)
        if signature in seen:
            continue
        result.append(item)
        seen.add(signature)
        if len(result) >= int(frontier_limit):
            break
    return result


def _endpoint_control_bounds(
    q_start: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reserve a limit buffer without invalidating the observed start pose."""

    start = np.asarray(q_start, dtype=np.float64).reshape(7)
    lo = np.asarray(lower, dtype=np.float64).reshape(7)
    hi = np.asarray(upper, dtype=np.float64).reshape(7)
    span = hi - lo
    if (
        not all(np.all(np.isfinite(value)) for value in (start, lo, hi))
        or np.any(span <= 0.0)
    ):
        raise ValueError("tracked-point endpoint joint bounds are invalid")
    if np.any(start < lo - 1e-6) or np.any(start > hi + 1e-6):
        raise ValueError("tracked-point endpoint start q exceeds local joint limits")

    margin = np.minimum(
        ENDPOINT_NEW_LIMIT_SAFETY_MARGIN_RAD,
        0.2 * span,
    )
    control_lower = lo + margin
    control_upper = hi - margin
    # The evaluator may begin at a configured/static limit.  Do not make that
    # observation infeasible, while preventing an interior joint from newly
    # entering the buffered region during this motion.  This is a robot safety
    # boundary, not a target-error acceptance threshold.
    control_lower[start < control_lower] = lo[start < control_lower]
    control_upper[start > control_upper] = hi[start > control_upper]
    return control_lower, control_upper, margin


def _generic_joint_seeds(
    q_start: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> list[np.ndarray]:
    """Return deterministic, model-independent seeds for the local IK solve.

    A symbolic target can leave any subset of the rigid transform's degrees of
    freedom unconstrained.  The seed set therefore spans the observed pose,
    the bounded midpoint, and signed perturbations in joint space; it does not
    inspect point names, coordinate labels, or particular test values.
    """

    start = np.asarray(q_start, dtype=np.float64).reshape(7)
    lo = np.asarray(lower, dtype=np.float64).reshape(7)
    hi = np.asarray(upper, dtype=np.float64).reshape(7)
    span = np.maximum(hi - lo, 1e-6)
    usable_span = np.minimum(span, 1.0)
    patterns: list[np.ndarray] = [np.zeros(7, dtype=np.float64)]
    # First cover each joint independently.  This makes the local solve able
    # to leave six joints at the captured posture while entering a different
    # elbow/wrist basin, without encoding any task-specific posture.
    for index in range(7):
        direction = np.zeros(7, dtype=np.float64)
        direction[index] = 1.0
        patterns.extend((direction, -direction))
    # Add deterministic mixed directions generated from binary masks.  The
    # sequence is fixed by the local model and never inspects target values.
    for mask in range(1, 32):
        patterns.append(
            np.asarray(
                [1.0 if (mask >> (index % 5)) & 1 else -1.0 for index in range(7)],
                dtype=np.float64,
            )
        )
    seeds: list[np.ndarray] = []

    def append(seed: np.ndarray) -> None:
        candidate = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
        if not any(
            float(np.linalg.norm(candidate - existing, ord=np.inf)) <= 1e-9
            for existing in seeds
        ):
            seeds.append(candidate)

    append(start)
    # Explore every signed single-joint direction near the capture before
    # distant postures.  This escapes stationary first-order residuals while
    # retaining the nearest feasible rigid transform.
    local_axis_patterns = patterns[1 : 1 + 2 * len(start)]
    mixed_patterns = patterns[1 + 2 * len(start) :]
    for pattern in local_axis_patterns:
        append(start + 0.12 * pattern * usable_span)

    # Still include global anchors in the primary frontier.  Some large rigid
    # rotations have no continuous endpoint in the capture's local basin.
    # Their placement immediately after the local axis coverage ensures both
    # classes reach candidate ranking before the primary resource bound.
    append(0.5 * (lo + hi))
    append(lo + 0.25 * span)
    append(hi - 0.25 * span)
    append(2.0 * 0.5 * (lo + hi) - start)
    for pattern in mixed_patterns:
        append(start + 0.12 * pattern * usable_span)
    for amplitude in (0.35, 0.65):
        for pattern in patterns[1:]:
            append(start + amplitude * pattern * usable_span)
    return seeds


def _affine_pose_family_targets(
    *,
    source_points_robot_base_m: Sequence[Sequence[float]],
    anchors_eef_m: Sequence[Sequence[float]],
    base_rotation: Sequence[Sequence[float]],
    base_position: Sequence[float],
    target_points: Sequence[Mapping[str, Any]],
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    fixed_target_points: Sequence[Mapping[str, Any]] | None = None,
    angular_offsets_deg: Sequence[float] = WHOLE_BODY_POSE_FAMILY_ANGLE_OFFSETS_DEG,
    axis_world: Sequence[float] | None = None,
    equation_tolerance_m: float = WHOLE_BODY_POSE_FAMILY_NUMERICAL_EQUATION_TOLERANCE_M,
) -> list[dict[str, Any]]:
    """Construct exact affine-constraint EEF pose-family members.

    Symbolic XYZ requests can constrain a rigid bundle's line direction while
    leaving roll about that line free.  A joint-space optimizer started at one
    member tends to remain in that basin, so this helper explicitly enumerates
    the genuine SE(3) null space.  Translation and affine variables are solved
    together for each sampled proper rotation; no target-specific names or
    constants are inspected.  Inconsistent full-pose targets simply produce no
    member at that angle.
    """

    source = _strict_points(
        source_points_robot_base_m,
        label="affine pose-family source points",
    )
    anchors = np.asarray(anchors_eef_m, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape != source.shape:
        raise ValueError("affine pose-family anchors must match source points")
    rotation = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    position = np.asarray(base_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(position)):
        raise ValueError("affine pose-family base pose must be finite")
    try:
        equation_tolerance = float(equation_tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "affine pose-family equation tolerance must be finite and positive"
        ) from exc
    if not math.isfinite(equation_tolerance) or equation_tolerance <= 0.0:
        raise ValueError(
            "affine pose-family equation tolerance must be finite and positive"
        )
    normalized_targets = _normalize_target_points(target_points)
    if len(normalized_targets) != len(source):
        raise ValueError("affine pose-family target/source counts do not match")
    fixed = np.empty((0, 3), dtype=np.float64)
    normalized_fixed: list[dict[str, Any]] = []
    if fixed_points_robot_base_m is not None:
        fixed = _strict_points(
            fixed_points_robot_base_m,
            label="affine pose-family fixed points",
        )
        normalized_fixed = _normalize_target_points(fixed_target_points or [])
        if len(fixed) != len(normalized_fixed):
            raise ValueError("affine pose-family fixed target/source counts do not match")
    model = _build_constraint_model(normalized_targets + normalized_fixed)
    if not model.rows:
        return []

    pair: tuple[int, int] | None = None
    pair_distance = -1.0
    pair_shared_coordinate_count = 0
    if axis_world is None:
        parsed_target_forms = [
            tuple(
                _affine_form(
                    point["target_xyz_m"][axis_name],
                    label=(
                        f"affine pose-family target_points[{point_index}]"
                        f".target_xyz_m.{axis_name}"
                    ),
                )
                for axis_name in AXES
            )
            for point_index, point in enumerate(normalized_targets)
        ]

        def same_constrained_form(first: _AffineForm, second: _AffineForm) -> bool:
            # Two ``?`` coordinates are independently free, not equal.  Every
            # other identical affine form fixes that component of the pair's
            # target displacement to zero.
            return bool(
                not first.free
                and not second.free
                and abs(float(first.constant) - float(second.constant)) <= 1.0e-12
                and first.coefficients == second.coefficients
            )

        pair_records: list[tuple[int, float, int, int]] = []
        for first in range(len(source)):
            for second in range(first + 1, len(source)):
                distance = float(np.linalg.norm(source[second] - source[first]))
                shared_coordinate_count = sum(
                    same_constrained_form(
                        parsed_target_forms[first][axis_index],
                        parsed_target_forms[second][axis_index],
                    )
                    for axis_index in range(3)
                )
                pair_records.append(
                    (int(shared_coordinate_count), distance, first, second)
                )
        # A pair sharing two target coordinates defines the physical line
        # about which a disconnected wrist flip can preserve the submitted
        # equations.  Prefer that semantic axis even when a third marker is
        # farther away.  Without such a line, preserve the historical longest
        # pair choice for numerical stability.
        constrained_pairs = [item for item in pair_records if item[0] >= 2]
        selectable_pairs = constrained_pairs if constrained_pairs else pair_records
        if selectable_pairs:
            pair_shared_coordinate_count, pair_distance, first, second = max(
                selectable_pairs,
                key=lambda item: (item[0], item[1], -item[2], -item[3]),
            )
            pair = (int(first), int(second))
        if pair is None or pair_distance <= 1.0e-9:
            return []
        # The pose family rotates EEF-frame anchors.  Applying the EEF
        # rotation to the already world-frame source points would apply the
        # captured transform twice and can erase a valid 180 degree member.
        anchor_axis = anchors[pair[1]] - anchors[pair[0]]
        anchor_distance = float(np.linalg.norm(anchor_axis))
        if anchor_distance <= 1.0e-9:
            return []
        axis = rotation @ (anchor_axis / anchor_distance)
    else:
        axis = np.asarray(axis_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 1.0e-9:
        return []
    axis = axis / axis_norm

    try:
        # Keep the supplied deterministic preference order.  The public list
        # intentionally places the half-turn early enough for collision
        # evaluation to see a wrist-flipped alternative promptly.
        offsets: list[float] = []
        for raw_value in angular_offsets_deg:
            value = float(raw_value)
            if value not in offsets:
                offsets.append(value)
    except (TypeError, ValueError):
        raise ValueError("affine pose-family angular offsets must be numeric")
    if not offsets or any(not math.isfinite(value) for value in offsets):
        raise ValueError("affine pose-family angular offsets must be finite")
    # Keep the helper bounded even when a direct benchmark supplies a large
    # custom list.  The public planner uses the deterministic list above.
    offsets = offsets[:32]

    variable_count = len(model.variable_names)
    rows = model.rows
    result: list[dict[str, Any]] = []
    seen_rotations: list[np.ndarray] = []
    for angle_deg in offsets:
        delta = _axis_rotation(axis, math.radians(angle_deg))
        candidate_rotation = delta @ rotation
        if any(
            float(np.linalg.norm(candidate_rotation - old, ord="fro")) <= 1.0e-9
            for old in seen_rotations
        ):
            continue
        seen_rotations.append(candidate_rotation.copy())
        local_offsets = (candidate_rotation @ anchors.T).T
        equation_matrix = np.zeros((len(rows), 3 + variable_count), dtype=np.float64)
        equation_rhs = np.zeros(len(rows), dtype=np.float64)
        for row_index, row in enumerate(rows):
            point_index = int(row.point_index)
            axis_index = int(row.axis_index)
            controlled = point_index < len(source)
            if controlled:
                equation_matrix[row_index, axis_index] = 1.0
                equation_rhs[row_index] = float(
                    row.form.constant - local_offsets[point_index, axis_index]
                )
            else:
                fixed_index = point_index - len(source)
                equation_rhs[row_index] = float(
                    row.form.constant - fixed[fixed_index, axis_index]
                )
            for name, coefficient in row.form.coefficients:
                equation_matrix[
                    row_index,
                    3 + model._variable_index[name],
                ] = -float(coefficient)
        try:
            solution, _residuals, _rank, _singular = np.linalg.lstsq(
                equation_matrix,
                equation_rhs,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue
        # Choose the translation-nearest member of an underdetermined affine
        # solution while preserving the equations exactly.  This only affects
        # null-space coordinates; it never relaxes a submitted relation.
        singular_values = np.linalg.svd(equation_matrix, compute_uv=False)
        rank = int(
            np.sum(
                singular_values
                > np.finfo(np.float64).eps
                * max(equation_matrix.shape)
                * max(float(singular_values[0]) if len(singular_values) else 1.0, 1.0)
            )
        )
        _u, _s, vh = np.linalg.svd(equation_matrix, full_matrices=True)
        if rank < vh.shape[0]:
            null_space = vh[rank:].T
            if null_space.size:
                correction = np.linalg.lstsq(
                    null_space[:3], position - solution[:3], rcond=None
                )[0]
                solution = solution + null_space @ correction
        equation_error = float(
            np.max(np.abs(equation_matrix @ solution - equation_rhs))
        )
        if not math.isfinite(equation_error) or equation_error > equation_tolerance:
            continue
        result.append(
            {
                "final_eef_position_m": solution[:3].astype(float).tolist(),
                "final_eef_quaternion_xyzw": mat_to_quat_xyzw(
                    candidate_rotation
                ).astype(float).tolist(),
                "pose_family": "affine_constraint_rigid_null_space",
                "pose_family_axis_world": axis.astype(float).tolist(),
                "pose_family_axis_source_pair": (
                    [int(pair[0]), int(pair[1])] if pair is not None else None
                ),
                "pose_family_axis_shared_coordinate_count": int(
                    pair_shared_coordinate_count
                ),
                "pose_family_angle_deg": float(angle_deg),
                "affine_equation_error_m": equation_error,
                "affine_equation_tolerance_m": equation_tolerance,
            }
        )
    return result


def plan_endpoint(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    source_points_robot_base_m: Sequence[Sequence[float]],
    target_points: Sequence[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    fixed_target_points: Sequence[dict[str, Any]] | None = None,
    pos_tol_m: float,
    ori_tol_deg: float,
    eef_orientation_regularization_weight: float = (
        ENDPOINT_EEF_ORIENTATION_REGULARIZATION_WEIGHT
    ),
    eef_pose_targets: Sequence[Mapping[str, Any]] | None = None,
    pose_targets_only: bool = False,
    max_endpoint_candidates: int = MAX_GENERIC_ENDPOINT_CANDIDATES,
    cancel_requested: Callable[[], bool] | None = None,
    planning_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Solve the nearest reachable rigid endpoint from numeric/symbolic targets."""

    from scipy.optimize import least_squares, minimize

    planning_deadline_monotonic = _normalize_planning_deadline(
        planning_deadline_monotonic
    )

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")
        _check_planning_deadline(planning_deadline_monotonic)

    check_cancelled()
    solver_check = (
        check_cancelled
        if planning_deadline_monotonic is not None
        or callable(cancel_requested)
        else None
    )
    orientation_regularization_weight = float(
        eef_orientation_regularization_weight
    )
    if (
        not math.isfinite(orientation_regularization_weight)
        or orientation_regularization_weight < 0.0
    ):
        raise ValueError(
            "EEF orientation regularization weight must be finite and nonnegative"
        )
    if isinstance(max_endpoint_candidates, bool) or not isinstance(
        max_endpoint_candidates, int
    ):
        raise ValueError("max_endpoint_candidates must be an integer")
    if not 1 <= int(max_endpoint_candidates) <= 128:
        raise ValueError("max_endpoint_candidates must be between 1 and 128")
    pose_target_records: list[dict[str, Any]] = []
    for target_index, raw_target in enumerate(eef_pose_targets or []):
        if not isinstance(raw_target, Mapping):
            raise ValueError(f"EEF pose target {target_index} must be an object")
        try:
            target_position = np.asarray(
                raw_target["final_eef_position_m"], dtype=np.float64
            ).reshape(3)
            target_quaternion = normalize_quaternion(
                raw_target["final_eef_quaternion_xyzw"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"EEF pose target {target_index} is incomplete"
            ) from exc
        if not np.all(np.isfinite(target_position)) or not np.all(
            np.isfinite(target_quaternion)
        ):
            raise ValueError(f"EEF pose target {target_index} must be finite")
        pose_target_records.append(
            {
                **dict(raw_target),
                "final_eef_position_m": target_position,
                "final_eef_quaternion_xyzw": target_quaternion,
            }
        )
    if pose_targets_only and not pose_target_records:
        raise ValueError("pose_targets_only requires at least one EEF pose target")
    q0 = _locked_q(q_start, label="tracked-point start q")
    source = _strict_points(
        source_points_robot_base_m,
        label="tracked-point source positions",
    )
    target_points = _normalize_target_points(target_points)
    # Canonicalize typed relations once at the local planner boundary.  The
    # same canonical records are then used for residuals, endpoint metadata,
    # and trajectory signing; direct benchmark callers therefore cannot take
    # a more permissive path than the HTTP contract.
    if len(target_points) != len(source):
        raise ValueError("tracked-point endpoint requires one to six paired points")
    if not np.all(np.isfinite(source)):
        raise ValueError("tracked-point source positions must be finite")
    if (fixed_points_robot_base_m is None) != (fixed_target_points is None):
        raise ValueError(
            "fixed points and fixed target points must be supplied together"
        )
    fixed_points = np.empty((0, 3), dtype=np.float64)
    normalized_fixed_targets: list[dict[str, Any]] = []
    if fixed_points_robot_base_m is not None:
        fixed_points = _strict_points(
            fixed_points_robot_base_m,
            label="tracked-point fixed reference positions",
        )
        normalized_fixed_targets = _normalize_target_points(
            fixed_target_points or []
        )
        if len(fixed_points) != len(normalized_fixed_targets):
            raise ValueError(
                "tracked-point fixed target/source point counts do not match"
            )
    if len(source) + len(fixed_points) > MAX_TRACKED_POINTS:
        raise ValueError(
            f"tracked-point endpoint cannot exceed {MAX_TRACKED_POINTS} total points"
        )
    controlled_names = [str(point["name"]) for point in target_points]
    fixed_names = [str(point["name"]) for point in normalized_fixed_targets]
    if set(controlled_names) & set(fixed_names):
        raise ValueError("tracked-point controlled/fixed names overlap")
    affine_target_points = target_points + normalized_fixed_targets
    relations = _normalize_relation_records(relations, controlled_names + fixed_names)
    constraint_model = _build_constraint_model(affine_target_points)
    inequalities = normalize_tracked_inequalities(
        inequalities or [],
        allowed_variables=set(constraint_model.variable_names),
    )

    def affine_points(points: np.ndarray) -> np.ndarray:
        return (
            np.vstack([points, fixed_points])
            if len(fixed_points)
            else points
        )

    def evaluate_constraints(points: np.ndarray) -> dict[str, Any]:
        return evaluate_target_constraints(
            points,
            target_points,
            tolerance_m=pos_tol_m,
            relations=relations,
            inequalities=inequalities,
            orientation_tolerance_deg=ori_tol_deg,
            fixed_points_robot_base_m=(
                fixed_points if len(fixed_points) else None
            ),
            fixed_target_points=(
                normalized_fixed_targets if len(fixed_points) else None
            ),
        )
    source_geometry_rank = geometry_rank(source)
    # The number of submitted markers is a structural property, not a pose
    # freedom classification.  Repeated/near-coincident markers are one
    # effective anchor (rank 0), a collinear set leaves the axial rotation
    # free (rank 1), and planar/spatial sets carry their corresponding rigid
    # orientation information (rank 2/3).  All endpoint regularization below
    # follows this rank rather than special-casing one or two list entries.
    single_anchor_geometry = source_geometry_rank == 0

    start_position, start_quaternion = eef_pose(state, arm, q0)
    start_rotation = quat_to_mat_xyzw(start_quaternion)
    anchors = rigid_anchors_from_points(start_position, start_quaternion, source)
    lower, upper = arm_joint_limits(state, arm)
    control_lower, control_upper, control_margin = _endpoint_control_bounds(
        q0[:7], lower[:7], upper[:7]
    )
    regularization_span = np.maximum(upper[:7] - lower[:7], 0.25)
    nearest_free_coordinate_projector = (
        constraint_model.free_direction_projector
        if single_anchor_geometry and not relations and not inequalities
        else np.zeros_like(constraint_model.free_direction_projector)
    )
    nearest_free_coordinate_dimension = (
        int(constraint_model.free_direction_dimension)
        if single_anchor_geometry and not relations and not inequalities
        else 0
    )
    affine_source = affine_points(source).reshape(-1)
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
    solver_evaluations = 0
    best_effort_refinement_evaluations = 0
    best_effort_refinement_attempts: list[dict[str, Any]] = []
    precise_motion_refinement_evaluations = 0
    precise_motion_refinement_attempts: list[dict[str, Any]] = []
    planning_budget_exhausted = False

    numeric_target_points = resolve_fully_numeric_target_points(target_points)
    geometry_report = geometry_preflight(
        source,
        numeric_target_points,
        tolerance_m=ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
    )

    # A rigid bundle preserves every pair separation.  Check only structural facts that
    # are independent of the submitted point names/values before nonlinear IK:
    # fixed affine displacement components cannot require a norm longer than
    # the source bundle.  Coupled/free components continue through the generic
    # solver below.  The bound is deliberately conservative: each endpoint is
    # allowed the public point tolerance, so a relative displacement may move
    # by at most twice that amount.  Crucially, this decision depends only on
    # the equations' fixed components, never on whether the caller spelled a
    # coordinate as a number, identifier, or affine object.
    def expand(selected: Sequence[float]) -> np.ndarray:
        q = q0.copy()
        q[:7] = np.asarray(selected, dtype=np.float64).reshape(7)
        if ARM_DOF == 8:
            q[7] = 0.0
        return q

    def add_candidate(q_raw: Sequence[float], report: dict[str, Any]) -> None:
        q = _locked_q(q_raw, label="tracked-point endpoint candidate")
        q[:7] = np.clip(q[:7], control_lower, control_upper)
        points, _position, quaternion = _point_positions(state, arm, q, anchors)
        constraints = evaluate_constraints(points)
        orientation_error = (
            orientation_error_deg(quaternion, start_quaternion)
            if single_anchor_geometry
            else 0.0
        )
        accuracy = _candidate_accuracy(
            constraints,
            preserved_eef_orientation_error_deg=float(orientation_error),
        )
        static_margins = np.minimum(
            q[:7] - lower[:7], upper[:7] - q[:7]
        )
        start_static_margins = np.minimum(
            q0[:7] - lower[:7], upper[:7] - q0[:7]
        )
        safety_shortfall = np.maximum(
            0.0, control_margin - static_margins
        )
        start_safety_shortfall = np.maximum(
            0.0, control_margin - start_static_margins
        )
        motion_score = _candidate_score(
            q,
            q0,
            source,
            points,
            lower,
            upper,
        )
        candidate_report = {
            **report,
            "constraints": constraints,
            # Keep the historical field as the capture-orientation metric for
            # compatibility.  The selection metric below also includes every
            # explicit line/plane/vector relation angle.
            "orientation_error_deg": float(orientation_error),
            "selection_orientation_error_deg": float(
                accuracy["orientation_error_deg"]
            ),
            "position_error_m": float(accuracy["position_error_m"]),
            "position_error_mm": float(accuracy["position_error_mm"]),
            "accuracy_cost_mm_plus_deg": float(
                accuracy["accuracy_cost_mm_plus_deg"]
            ),
            "precise": bool(accuracy["precise"]),
            "precision_class": _precision_class_label(accuracy),
            "precise_position_tolerance_m": float(
                ENDPOINT_PRECISE_POSITION_TOLERANCE_M
            ),
            "collinear_position_tolerance_m": accuracy.get(
                "collinear_position_tolerance_m"
            ),
            "position_precision_ratio": float(
                accuracy["position_precision_ratio"]
            ),
            "precise_orientation_tolerance_deg": float(
                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
            ),
            "meets_requested_tolerance": bool(
                constraints.get("ok")
                and orientation_error <= float(ori_tol_deg)
            ),
            "motion_score": float(motion_score),
            "joint_limit_safety_margin_rad": (
                control_margin.astype(float).tolist()
            ),
            "joint_static_margin_rad": (
                static_margins.astype(float).tolist()
            ),
            "joint_limit_safety_shortfall_rad": (
                safety_shortfall.astype(float).tolist()
            ),
            "joint_limit_safety_worsening_rad": np.maximum(
                0.0,
                safety_shortfall - start_safety_shortfall,
            )
            .astype(float)
            .tolist(),
            "control_lower_bound_rad": (
                control_lower.astype(float).tolist()
            ),
            "control_upper_bound_rad": (
                control_upper.astype(float).tolist()
            ),
            "score": float(
                motion_score
                if accuracy["precise"]
                else CANDIDATE_BEST_EFFORT_SCORE_OFFSET
                + float(accuracy["accuracy_cost_mm_plus_deg"])
            ),
        }
        for index, (existing_q, existing_report) in enumerate(candidates):
            if float(np.linalg.norm(q[:7] - existing_q[:7], ord=np.inf)) > 1e-5:
                continue
            if _candidate_rank_key(candidate_report) < _candidate_rank_key(
                existing_report
            ):
                candidates[index] = (q, candidate_report)
            return
        candidates.append((q, candidate_report))

    # ``residual_dimension`` is the number of independent geometric equalities
    # left after eliminating all free/affine coordinate variables.  Numeric,
    # symbolic, affine, and free coordinates all use the same solver below;
    # coordinate names, ordering, and particular constants never select a
    # different planning or execution path.
    has_position_constraints = bool(
        constraint_model.has_position_constraints or relations or inequalities
    )
    constraint_noop = False
    initial_constraints = evaluate_constraints(source)
    # A public tolerance is an acceptance bound, not permission to skip a
    # requested movement.  Only an already-satisfied equation (to numeric
    # precision) may take the no-motion path.  This check is representation
    # agnostic: numeric, variable, affine, and free coordinates use it alike.
    # A positional residual of zero is not enough to skip the solve: an
    # oriented-plane or vector relation may still be violated.  Require the
    # complete constraint report to pass before taking the no-motion path.
    if not pose_targets_only:
        if _constraints_numerically_satisfied(
            initial_constraints,
            pos_tol_m=pos_tol_m,
            ori_tol_deg=ori_tol_deg,
        ):
            add_candidate(
                q0,
                {"mode": "constraints_already_satisfied_no_motion"},
            )
            constraint_noop = True
        else:
            # A finite capture pose is the deterministic lower bound for a
            # best-effort request.  It also guarantees that a cooperative planning
            # deadline returns the best endpoint found so far instead of turning a
            # solvable approximation into a hard planning failure.
            add_candidate(
                q0,
                {
                    "mode": "joint_space_affine_constraint_ik",
                    "candidate_origin": "capture_pose_best_effort_baseline",
                },
            )

    # A proper Kabsch transform is an analytic seed for fully numeric 3/4
    # point targets.  It improves convergence but is never the acceptance
    # authority: the same local FK and affine/relation residuals still decide
    # whether the candidate is usable.
    kabsch_seed_report: dict[str, Any] | None = None
    # Kabsch needs at least three corresponding samples for an informative
    # bundle seed.  This is an analytic acceleration only; geometry rank and
    # the unified residual remain the acceptance authority.
    if (
        numeric_target_points is not None
        and source.shape[0] >= 3
        and not constraint_noop
        and not pose_targets_only
    ):
        try:
            check_cancelled()
            kabsch = kabsch_rigid_transform(source, numeric_target_points)
            kabsch_seed_report = {
                "max_rigid_fit_error_m": float(kabsch["max_error_m"]),
                "source_rank": int(kabsch["source_rank"]),
                "target_rank": int(kabsch["target_rank"]),
                "reflection_detected": bool(kabsch["reflection_detected"]),
            }
            # A reflected four-point target was already rejected by the
            # geometry preflight.  For rank-deficient sets the proper Kabsch
            # orientation is one deterministic member of the free family.
            target_rotation = kabsch["rotation_matrix"] @ start_rotation
            target_position = (
                kabsch["rotation_matrix"] @ np.asarray(start_position)
                + kabsch["translation_robot_base_m"]
            )
            kabsch_q, kabsch_ik = solve_pose_target(
                state,
                arm,
                q0,
                target_position=target_position,
                target_quaternion=mat_to_quat_xyzw(target_rotation),
                joint_indices=range(7),
                pos_tol_m=min(0.008, max(0.002, float(pos_tol_m))),
                ori_tol_deg=min(5.0, max(1.0, float(ori_tol_deg))),
                max_iterations=220,
                check_requested=solver_check,
            )
            check_cancelled()
            add_candidate(
                kabsch_q,
                {
                    "mode": "kabsch_proper_rigid_pose_seed",
                    "kabsch": kabsch_seed_report,
                    "pose_ik": kabsch_ik,
                },
            )
        except PlanningDeadlineExceeded:
            # Kabsch is an acceleration seed only.  A candidate found by an
            # earlier stage remains usable; otherwise the caller receives the
            # explicit deadline and can report a planning-budget failure.
            planning_budget_exhausted = True
        except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            kabsch_seed_report = {
                "ok": False,
                "error": str(exc),
            }

    pose_target_failures: list[dict[str, Any]] = []
    pose_target_solve_count = 0
    for pose_target_index, pose_target in enumerate(pose_target_records):
        if planning_budget_exhausted:
            break
        try:
            check_cancelled()
            pose_target_solve_count += 1
            pose_q, pose_ik = solve_pose_target(
                state,
                arm,
                q0,
                target_position=pose_target["final_eef_position_m"],
                target_quaternion=pose_target["final_eef_quaternion_xyzw"],
                joint_indices=range(7),
                pos_tol_m=min(0.003, max(0.001, float(pos_tol_m))),
                ori_tol_deg=min(3.0, max(1.0, float(ori_tol_deg))),
                max_iterations=160,
                check_requested=solver_check,
            )
            check_cancelled()
            add_candidate(
                pose_q,
                {
                    "mode": "explicit_eef_pose_target_frontier",
                    "pose_target_index": int(pose_target_index),
                    "pose_target": {
                        key: value
                        for key, value in pose_target.items()
                        if key
                        not in {
                            "final_eef_position_m",
                            "final_eef_quaternion_xyzw",
                        }
                    },
                    "pose_ik": pose_ik,
                },
            )
        except PlanningDeadlineExceeded:
            planning_budget_exhausted = True
            pose_target_failures.append(
                {
                    "pose_target_index": int(pose_target_index),
                    "reason": "planning_budget_exhausted",
                }
            )
            break
        except (RuntimeError, TypeError, ValueError) as exc:
            pose_target_failures.append(
                {
                    "pose_target_index": int(pose_target_index),
                    "error": str(exc),
                }
            )

    def feasibility_residual(
        selected: Sequence[float],
        norm_power: float = POSITION_RESIDUAL_NORM_POWER,
    ) -> np.ndarray:
        nonlocal solver_evaluations
        solver_evaluations += 1
        if solver_evaluations > GENERIC_MAX_FUNCTION_EVALUATIONS:
            raise _SolverBudgetExceeded(
                "tracked-point local IK evaluation budget exhausted"
            )
        check_cancelled()
        q = expand(selected)
        points, _position, quaternion = _point_positions(state, arm, q, anchors)
        position_residual, fitted_values = constraint_model.fit(
            affine_points(points)
        )
        position_scaled = position_residual / max(
            ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
            1e-6,
        )
        # Minimize an even high-order norm while retaining the sign for a
        # useful finite-difference direction.  Clipping only protects the
        # objective from overflow for wildly unreachable numeric inputs; the
        # unmodified residual is always used for acceptance/reporting.
        position_scaled = np.clip(position_scaled, -1e3, 1e3)
        position_objective = np.sign(position_scaled) * np.power(
            np.abs(position_scaled), float(norm_power) / 2.0
        )
        residual = list(position_objective)
        inequality_values = _inequality_violation_vector(
            inequalities,
            constraint_model,
            fitted_values,
        )
        if inequality_values.size:
            inequality_scaled = np.clip(
                inequality_values
                / max(ENDPOINT_PRECISE_POSITION_TOLERANCE_M, 1.0e-6),
                0.0,
                1.0e3,
            )
            residual.extend(
                np.power(
                    inequality_scaled,
                    float(norm_power) / 2.0,
                )
            )
        relation_blocks = relation_residual_blocks(
            affine_points(points),
            relations,
            point_names=controlled_names + fixed_names,
            _canonical=True,
        )
        for block in relation_blocks:
            relation_values = np.asarray(block["residual"], dtype=np.float64)
            scale = (
                max(
                    math.radians(
                        ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
                    ),
                    1.0e-6,
                )
                if block["kind"] == "orientation"
                else max(
                    relation_position_tolerance_m(
                        block.get("type"),
                        ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
                    ),
                    1.0e-6,
                )
            )
            relation_scaled = np.clip(relation_values / scale, -1.0e3, 1.0e3)
            residual.extend(
                np.sign(relation_scaled)
                * np.power(np.abs(relation_scaled), float(norm_power) / 2.0)
            )
            if block.get("orientation_residual") is not None:
                line_orientation = np.asarray(
                    block["orientation_residual"], dtype=np.float64
                ).reshape(-1)
                line_orientation_scaled = np.clip(
                    line_orientation
                    / max(
                        math.sin(
                            math.radians(
                                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
                            )
                        ),
                        1.0e-6,
                    ),
                    -1.0e3,
                    1.0e3,
                )
                residual.extend(
                    np.sign(line_orientation_scaled)
                    * np.power(
                        np.abs(line_orientation_scaled),
                        float(norm_power) / 2.0,
                    )
                )
        if single_anchor_geometry:
            orientation_scaled = orientation_error_vector(
                quaternion, start_quaternion
            ) / max(
                math.radians(ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG),
                1e-6,
            )
            orientation_scaled = np.clip(orientation_scaled, -1e3, 1e3)
            residual.extend(
                np.sign(orientation_scaled)
                * np.power(np.abs(orientation_scaled), 2.0)
            )
        residual.extend(
            ENDPOINT_TRACKED_POINT_REGULARIZATION_WEIGHT
            * (points - source).reshape(-1)
        )
        if nearest_free_coordinate_dimension:
            free_coordinate_motion = nearest_free_coordinate_projector @ (
                affine_points(points).reshape(-1) - affine_source
            )
            residual.extend(
                ENDPOINT_NEAREST_FREE_COORDINATE_WEIGHT
                * free_coordinate_motion
                / max(ENDPOINT_PRECISE_POSITION_TOLERANCE_M, 1e-6)
            )
        if not single_anchor_geometry and source_geometry_rank <= 1:
            residual.extend(
                orientation_regularization_weight
                * orientation_error_vector(quaternion, start_quaternion)
            )
        # These equations can constrain as little as one scalar while the arm
        # has seven active joints.  A capture-relative Tikhonov term gives the
        # null space a unique minimum-motion solution.  It is deliberately
        # representation-independent: numeric, symbolic, affine, and free
        # coordinates all contribute through the same eliminated constraint
        # residual above.
        residual.extend(
            ENDPOINT_JOINT_REGULARIZATION_WEIGHT
            * (q[:7] - q0[:7])
            / regularization_span
        )
        static_margins = np.minimum(
            q[:7] - lower[:7], upper[:7] - q[:7]
        )
        residual.extend(
            ENDPOINT_LIMIT_AVOIDANCE_RESIDUAL_WEIGHT
            * np.maximum(0.0, control_margin - static_margins)
            / np.maximum(control_margin, 1e-6)
        )
        return np.asarray(residual, dtype=np.float64)

    if (
        has_position_constraints
        and not constraint_noop
        and not planning_budget_exhausted
        and not pose_targets_only
    ):
        # The affine model has already removed coordinates that are free
        # variables, so the residual contains only equalities the rigid body
        # must satisfy.  Candidate ranking below selects the solution nearest
        # the frozen posture after this solve has established geometric
        # feasibility.  This exact same block handles every coordinate form.
        seeds = _generic_joint_seeds(
            q0[:7], control_lower, control_upper
        )
        best_constraint_error = float("inf")
        solver_errors: list[str] = []
        budget_exhausted = False
        deadline_exhausted = False

        def solve_seed(
            seed: np.ndarray,
            *,
            max_nfev: int,
            pass_name: str,
            seed_index: int,
            norm_power: float,
        ) -> tuple[np.ndarray | None, float | None, bool]:
            """Run one bounded local solve and return its endpoint and accuracy."""

            nonlocal best_constraint_error, budget_exhausted, deadline_exhausted
            try:
                check_cancelled()
                solved = least_squares(
                    lambda selected: feasibility_residual(
                        selected,
                        norm_power,
                    ),
                    seed,
                    bounds=(control_lower, control_upper),
                    # The public contract is millimetre-scale.  Tighter
                    # tolerances only spend iterations polishing a floating
                    # point residual which is later checked against
                    # ``pos_tol_m``.
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8,
                    max_nfev=max_nfev,
                )
                check_cancelled()
            except _SolverBudgetExceeded:
                budget_exhausted = True
                return None, None, False
            except PlanningDeadlineExceeded:
                budget_exhausted = True
                deadline_exhausted = True
                return None, None, False
            except (RuntimeError, ValueError, TypeError) as exc:
                check_cancelled()
                solver_errors.append(f"{pass_name}[{seed_index}]: {exc}")
                return None, None, False
            stage_a_q = expand(solved.x)
            stage_a_points, _stage_a_position, stage_a_quaternion = _point_positions(
                state,
                arm,
                stage_a_q,
                anchors,
            )
            stage_a_constraints = evaluate_constraints(stage_a_points)
            stage_a_error = float(stage_a_constraints["max_constraint_error_m"])
            best_constraint_error = min(best_constraint_error, stage_a_error)
            stage_a_orientation = (
                orientation_error_deg(stage_a_quaternion, start_quaternion)
                if single_anchor_geometry
                else 0.0
            )
            add_candidate(
                stage_a_q,
                {
                    "mode": "joint_space_affine_constraint_ik",
                    "solver_pass": pass_name,
                    "seed_index": int(seed_index),
                    "solver_success": bool(solved.success),
                    "solver_status": int(solved.status),
                    "solver_message": str(solved.message),
                    "function_evaluations": int(solved.nfev),
                    "position_residual_norm_power": float(norm_power),
                    "constraint_residual_dimension": int(
                        constraint_model.residual_dimension
                    ),
                    "constraint_error_m": stage_a_error,
                    "orientation_error_deg": float(stage_a_orientation),
                    "capture_joint_regularization_weight": float(
                        ENDPOINT_JOINT_REGULARIZATION_WEIGHT
                    ),
                    "tracked_point_regularization_weight": float(
                        ENDPOINT_TRACKED_POINT_REGULARIZATION_WEIGHT
                    ),
                    "nearest_free_coordinate_weight": float(
                        ENDPOINT_NEAREST_FREE_COORDINATE_WEIGHT
                        if nearest_free_coordinate_dimension
                        else 0.0
                    ),
                    "nearest_free_coordinate_dimension": int(
                        nearest_free_coordinate_dimension
                    ),
                    "eef_orientation_regularization_weight": float(
                        orientation_regularization_weight
                    ),
                },
            )
            # ``stage_a_error`` is position-only for historical diagnostics.
            # Feasibility must use the complete report so an oriented-plane or
            # vector relation cannot be treated as solved merely because all
            # affine coordinates happen to fit.  The final candidate gate
            # below performs the same check; keeping it here also stops the
            # refinement loop at the first genuinely feasible stage.
            stage_a_accuracy = _candidate_accuracy(
                stage_a_constraints,
                preserved_eef_orientation_error_deg=float(
                    stage_a_orientation
                ),
            )
            return stage_a_q, stage_a_error, bool(stage_a_accuracy["precise"])

        def solve_seed_sequence(
            seed: np.ndarray,
            *,
            max_nfev: int,
            pass_name: str,
            seed_index: int,
            include_refinement: bool,
        ) -> tuple[float | None, bool]:
            """Run the same residual stages for every coordinate representation."""

            stages = [POSITION_RESIDUAL_PRIMARY_POWER]
            if include_refinement:
                stages.extend(
                    (
                        POSITION_RESIDUAL_NORM_POWER,
                        POSITION_RESIDUAL_FALLBACK_POWER,
                    )
                )
            last_error: float | None = None
            last_precise = False
            current_seed = np.asarray(seed, dtype=np.float64).reshape(7)
            for stage_index, norm_power in enumerate(stages):
                if budget_exhausted:
                    break
                stage_name = (
                    pass_name
                    if stage_index == 0
                    else f"{pass_name}_refine{stage_index}"
                )
                solved_q, last_error, last_precise = solve_seed(
                    current_seed,
                    max_nfev=max_nfev,
                    pass_name=stage_name,
                    seed_index=seed_index,
                    norm_power=norm_power,
                )
                # Refinement is a continuation of the preceding solve, not an
                # independent restart from the original seed.  Restarting used
                # to discard the better basin just found and spend the finite
                # evaluation budget rediscovering it.
                if solved_q is not None:
                    current_seed = np.asarray(solved_q[:7], dtype=np.float64)
                if last_precise:
                    break
            return last_error, last_precise

        # A solve started at the capture is both the best nearest-posture
        # candidate and the cheapest way to handle ordinary constraints.  Only
        # when that basin cannot meet the strict equation residual do we pay
        # for model-independent multi-start exploration.
        local_error, local_precise = solve_seed_sequence(
            q0[:7],
            max_nfev=120,
            pass_name="local",
            seed_index=0,
            include_refinement=True,
        )
        passes = ()
        if not local_precise and not deadline_exhausted:
            # The first pass is intentionally bounded.  If no endpoint
            # satisfies the public tolerance, a second pass uses the
            # remaining generic seeds and a larger budget before declaring the
            # request infeasible.  No pass is selected by names or constants.
            passes = (
                (seeds[1:GENERIC_PRIMARY_SEED_COUNT], 180, "primary"),
                (seeds[GENERIC_PRIMARY_SEED_COUNT:], 500, "fallback"),
            )
        for pass_seeds, max_nfev, pass_name in passes:
                if budget_exhausted:
                    break
                if not pass_seeds:
                    continue
                for seed_index, seed in enumerate(pass_seeds, start=1):
                    stage_error, stage_precise = solve_seed_sequence(
                        seed,
                        max_nfev=max_nfev,
                        pass_name=pass_name,
                        seed_index=seed_index,
                        include_refinement=pass_name == "fallback",
                    )
                    if budget_exhausted:
                        break
                # Always finish the fixed primary frontier before ranking.
                # It contains both local-axis and global-anchor coverage; an
                # early candidate count cannot prove Cartesian path quality.
                if pass_name == "primary" and any(
                    bool(report.get("precise"))
                    for _candidate_q, report in candidates
                ):
                    break

        # Preserve the distinction between the generic evaluation cap and a
        # caller-supplied wall-clock budget in endpoint diagnostics.
        planning_budget_exhausted = bool(
            planning_budget_exhausted or deadline_exhausted
        )

        if not candidates:
            detail = (
                f"; solver errors: {' | '.join(solver_errors[:3])}"
                if solver_errors
                else ""
            )
            if budget_exhausted:
                detail += (
                    "; model-independent solver budget exhausted after "
                    f"{int(solver_evaluations)} evaluations"
                )
            # The caller turns this into a structured planning failure.  Keep
            # the measured best residual so an infeasible request is distinct
            # from an observation or execution failure.
            if deadline_exhausted or planning_budget_exhausted:
                raise PlanningDeadlineExceeded(
                    "tracked-point endpoint planning budget exhausted before a valid candidate"
                )
            raise ValueError(
                "no submission-local IK endpoint satisfies the tracked-point "
                f"constraints (best residual {best_constraint_error:.6f}m, "
                f"tolerance {float(pos_tol_m):.6f}m){detail}"
            )

    # Least-squares is the robust feasibility engine, but its smooth squared
    # residual is not the same ordering requested for a best-effort result.
    # Only when no 3 mm / 5 degree endpoint exists in the sampled frontier,
    # refine the most accurate basins against the authoritative scalar:
    #
    #     maximum position error [mm] + maximum orientation error [degree]
    #
    # Motion is intentionally absent here.  If any evaluation enters the
    # precise class, ``add_candidate`` and the final rank key switch back to
    # minimum-motion ordering automatically.
    if (
        has_position_constraints
        and candidates
        and not any(bool(report.get("precise")) for _q, report in candidates)
        and not planning_budget_exhausted
        and not pose_targets_only
    ):
        refinement_seeds = sorted(
            candidates,
            key=lambda item: _candidate_rank_key(item[1]),
        )[:BEST_EFFORT_REFINEMENT_SEED_COUNT]
        for refinement_index, (seed_q, seed_report) in enumerate(refinement_seeds):
            best_precise_q: np.ndarray | None = None
            best_precise_motion = math.inf

            def best_effort_objective(selected: Sequence[float]) -> float:
                nonlocal best_effort_refinement_evaluations
                nonlocal best_precise_q, best_precise_motion
                best_effort_refinement_evaluations += 1
                check_cancelled()
                q = expand(selected)
                points, _position, quaternion = _point_positions(
                    state, arm, q, anchors
                )
                constraints = evaluate_constraints(points)
                orientation_error = (
                    orientation_error_deg(quaternion, start_quaternion)
                    if single_anchor_geometry
                    else 0.0
                )
                accuracy = _candidate_accuracy(
                    constraints,
                    preserved_eef_orientation_error_deg=float(
                        orientation_error
                    ),
                )
                cost = float(accuracy["accuracy_cost_mm_plus_deg"])
                if bool(accuracy["precise"]):
                    motion = _candidate_score(
                        q,
                        q0,
                        source,
                        points,
                        lower,
                        upper,
                    )
                    if motion < best_precise_motion:
                        best_precise_motion = float(motion)
                        best_precise_q = q.copy()
                return cost if math.isfinite(cost) else 1.0e12

            try:
                refined = minimize(
                    best_effort_objective,
                    np.asarray(seed_q[:7], dtype=np.float64),
                    method="Nelder-Mead",
                    bounds=list(zip(control_lower, control_upper)),
                    options={
                        "maxiter": int(
                            BEST_EFFORT_REFINEMENT_MAX_FUNCTION_EVALUATIONS_PER_SEED
                        ),
                        "maxfev": int(
                            BEST_EFFORT_REFINEMENT_MAX_FUNCTION_EVALUATIONS_PER_SEED
                        ),
                        "xatol": 1.0e-7,
                        "fatol": 1.0e-7,
                    },
                )
                refined_q = expand(refined.x)
                attempt = {
                    "seed_rank": int(refinement_index),
                    "seed_accuracy_cost_mm_plus_deg": float(
                        seed_report.get("accuracy_cost_mm_plus_deg", math.inf)
                    ),
                    "solver": "bounded_nelder_mead_exact_mm_plus_deg",
                    "solver_success": bool(refined.success),
                    "solver_status": int(refined.status),
                    "solver_message": str(refined.message),
                    "function_evaluations": int(refined.nfev),
                    "objective_cost_mm_plus_deg": float(refined.fun),
                }
                best_effort_refinement_attempts.append(dict(attempt))
                add_candidate(
                    refined_q,
                    {
                        "mode": "joint_space_affine_constraint_ik",
                        "refinement_mode": "best_effort_accuracy_refinement",
                        **attempt,
                    },
                )
                if best_precise_q is not None:
                    add_candidate(
                        best_precise_q,
                        {
                            "mode": "joint_space_affine_constraint_ik",
                            "refinement_mode": (
                                "best_effort_refinement_precise_frontier"
                            ),
                            **attempt,
                        },
                    )
            except PlanningDeadlineExceeded:
                planning_budget_exhausted = True
                best_effort_refinement_attempts.append(
                    {
                        "seed_rank": int(refinement_index),
                        "solver": "bounded_nelder_mead_exact_mm_plus_deg",
                        "deadline_exhausted": True,
                    }
                )
                break
            except (RuntimeError, TypeError, ValueError) as exc:
                best_effort_refinement_attempts.append(
                    {
                        "seed_rank": int(refinement_index),
                        "solver": "bounded_nelder_mead_exact_mm_plus_deg",
                        "error": str(exc),
                    }
                )

    # A relation-only request describes a geometric equivalence class rather
    # than an exact Cartesian target.  Once the generic solver has entered the
    # fixed per-constraint position / 5 degree precision class, refine motion
    # while remaining in that class.  Collinearity uses 1 mm while unrelated
    # position constraints retain 3 mm.  Coordinate-bearing requests skip it:
    # moving a numeric/affine target back to the tolerance boundary would
    # reduce accuracy and regress the established coordinate semantics.
    if (
        relations
        and not constraint_model.has_position_constraints
        and not inequalities
        and candidates
        and not planning_budget_exhausted
        and not pose_targets_only
    ):
        precise_seeds = [
            item
            for item in sorted(candidates, key=lambda item: _candidate_rank_key(item[1]))
            if bool(item[1].get("precise"))
        ][:RELATION_PRECISE_MOTION_REFINEMENT_SEED_COUNT]
        for refinement_index, (seed_q, seed_report) in enumerate(precise_seeds):
            cached_x: np.ndarray | None = None
            cached_value: dict[str, Any] | None = None
            best_precise_q = np.asarray(seed_q, dtype=np.float64).copy()
            best_precise_motion = float(seed_report.get("motion_score", math.inf))

            def precise_evaluation(selected: Sequence[float]) -> dict[str, Any]:
                nonlocal cached_x, cached_value
                nonlocal precise_motion_refinement_evaluations
                nonlocal best_precise_q, best_precise_motion
                selected_array = np.asarray(selected, dtype=np.float64).reshape(7)
                if (
                    cached_x is not None
                    and np.array_equal(selected_array, cached_x)
                    and cached_value is not None
                ):
                    return cached_value
                precise_motion_refinement_evaluations += 1
                check_cancelled()
                q = expand(selected_array)
                points, _position, quaternion = _point_positions(
                    state, arm, q, anchors
                )
                constraints = evaluate_constraints(points)
                orientation_error = (
                    orientation_error_deg(quaternion, start_quaternion)
                    if single_anchor_geometry
                    else 0.0
                )
                accuracy = _candidate_accuracy(
                    constraints,
                    preserved_eef_orientation_error_deg=float(orientation_error),
                )
                motion = _candidate_score(
                    q,
                    q0,
                    source,
                    points,
                    lower,
                    upper,
                )
                value = {
                    "q": q,
                    "accuracy": accuracy,
                    "motion_score": float(motion),
                }
                if bool(accuracy["precise"]) and motion < best_precise_motion:
                    best_precise_q = q.copy()
                    best_precise_motion = float(motion)
                cached_x = selected_array.copy()
                cached_value = value
                return value

            try:
                refined = minimize(
                    lambda selected: float(
                        precise_evaluation(selected)["motion_score"]
                    ),
                    np.asarray(seed_q[:7], dtype=np.float64),
                    method="SLSQP",
                    bounds=list(zip(control_lower, control_upper)),
                    constraints=(
                        {
                            "type": "ineq",
                            "fun": lambda selected: (
                                1.0
                                - float(
                                    precise_evaluation(selected)["accuracy"][
                                        "position_precision_ratio"
                                    ]
                                )
                            ),
                        },
                        {
                            "type": "ineq",
                            "fun": lambda selected: (
                                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
                                - float(
                                    precise_evaluation(selected)["accuracy"][
                                        "orientation_error_deg"
                                    ]
                                )
                            ),
                        },
                    ),
                    options={
                        "maxiter": int(
                            RELATION_PRECISE_MOTION_REFINEMENT_MAX_ITERATIONS_PER_SEED
                        ),
                        "ftol": 1.0e-10,
                    },
                )
                final_value = precise_evaluation(refined.x)
                attempt = {
                    "seed_rank": int(refinement_index),
                    "seed_motion_score": float(
                        seed_report.get("motion_score", math.inf)
                    ),
                    "solver": (
                        "bounded_slsqp_relation_motion_inside_"
                        + (
                            "collinear_1mm_other_3mm_5deg"
                            if any(
                                str(item.get("type") or "").strip().lower()
                                in COLLINEAR_RELATION_TYPES
                                for item in relations
                            )
                            else "3mm_5deg"
                        )
                    ),
                    "solver_success": bool(refined.success),
                    "solver_status": int(refined.status),
                    "solver_message": str(refined.message),
                    "iterations": int(refined.nit),
                    "function_evaluations": int(refined.nfev),
                    "objective_motion_score": float(refined.fun),
                }
                precise_motion_refinement_attempts.append(dict(attempt))
                if bool(final_value["accuracy"]["precise"]):
                    add_candidate(
                        final_value["q"],
                        {
                            "mode": "joint_space_affine_constraint_ik",
                            "refinement_mode": (
                                "relation_precise_minimum_motion_refinement"
                            ),
                            **attempt,
                        },
                    )
                add_candidate(
                    best_precise_q,
                    {
                        "mode": "joint_space_affine_constraint_ik",
                        "refinement_mode": (
                            "relation_precise_minimum_motion_frontier"
                        ),
                        **attempt,
                    },
                )
            except PlanningDeadlineExceeded:
                planning_budget_exhausted = True
                precise_motion_refinement_attempts.append(
                    {
                        "seed_rank": int(refinement_index),
                        "solver": "bounded_slsqp_relation_motion_inside_precision_frontier",
                        "deadline_exhausted": True,
                    }
                )
                break
            except (RuntimeError, TypeError, ValueError) as exc:
                precise_motion_refinement_attempts.append(
                    {
                        "seed_rank": int(refinement_index),
                        "solver": "bounded_slsqp_relation_motion_inside_precision_frontier",
                        "error": str(exc),
                    }
                )

    if not has_position_constraints and not pose_targets_only:
        add_candidate(q0, {"mode": "all_coordinates_free_no_motion"})
    if not candidates:
        if planning_budget_exhausted:
            raise PlanningDeadlineExceeded(
                "tracked-point endpoint planning budget exhausted before a valid candidate"
            )
        raise ValueError(
            "no submission-local IK endpoint satisfies the tracked-point target constraints"
        )

    ordered_candidates = [
        item
        for item in candidates
        if item[1]["constraints"].get("axial_order_satisfied", True)
        and item[1]["constraints"].get("inequality_constraints_ok", True)
    ]
    if not ordered_candidates:
        if inequalities:
            raise ValueError(
                "no IK endpoint satisfies the requested strict inequality constraints"
            )
        raise ValueError("no IK endpoint satisfies the requested collinear axial order")
    ranked = sorted(
        ordered_candidates,
        key=lambda item: _candidate_rank_key(item[1]),
    )[: int(max_endpoint_candidates)]
    precise_candidate_count = sum(
        1 for _candidate_q, report in candidates if bool(report.get("precise"))
    )
    requested_tolerance_candidate_count = sum(
        1
        for _candidate_q, report in candidates
        if bool(report.get("meets_requested_tolerance"))
    )

    def endpoint_result(
        q_final: np.ndarray,
        selected_report: dict[str, Any],
        rank: int,
    ) -> dict[str, Any]:
        final_points, final_position, final_quaternion = _point_positions(
            state, arm, q_final, anchors
        )
        final_constraints = evaluate_constraints(final_points)
        rigid_rotation = quat_to_mat_xyzw(final_quaternion) @ start_rotation.T
        rigid_translation = final_position - rigid_rotation @ start_position
        return {
            "q_start": q0,
            "q_final": q_final,
            "anchors_eef_m": anchors,
            "source_points_robot_base_m": source,
            "fixed_points_robot_base_m": fixed_points,
            "fixed_target_points": normalized_fixed_targets,
            "resolved_points_robot_base_m": final_points,
            "geometry_rank": int(source_geometry_rank),
            "geometry_preflight": geometry_report,
            "kabsch_seed": kabsch_seed_report,
            "planning_budget_exhausted": bool(planning_budget_exhausted),
            "pose_target_solve_count": int(pose_target_solve_count),
            "pose_target_failure_count": int(len(pose_target_failures)),
            "pose_target_failures": list(pose_target_failures),
            "solver_evaluations": int(solver_evaluations),
            "best_effort_refinement_evaluations": int(
                best_effort_refinement_evaluations
            ),
            "best_effort_refinement_attempts": list(
                best_effort_refinement_attempts
            ),
            "precise_motion_refinement_evaluations": int(
                precise_motion_refinement_evaluations
            ),
            "precise_motion_refinement_attempts": list(
                precise_motion_refinement_attempts
            ),
            "selection_policy": _endpoint_selection_policy(relations),
            "inequalities": [dict(item) for item in inequalities],
            "precise_position_tolerance_m": float(
                ENDPOINT_PRECISE_POSITION_TOLERANCE_M
            ),
            "collinear_position_tolerance_m": (
                COLLINEAR_POSITION_TOLERANCE_M
                if any(
                    str(item.get("type") or "").strip().lower()
                    in COLLINEAR_RELATION_TYPES
                    for item in relations
                )
                else None
            ),
            "precise_orientation_tolerance_deg": float(
                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
            ),
            "precise_candidate_count": int(precise_candidate_count),
            "requested_tolerance_candidate_count": int(
                requested_tolerance_candidate_count
            ),
            "selected_is_precise": bool(selected_report.get("precise")),
            "selected_meets_requested_tolerance": bool(
                selected_report.get("meets_requested_tolerance")
            ),
            "nearest_free_coordinate_weight": float(
                ENDPOINT_NEAREST_FREE_COORDINATE_WEIGHT
                if nearest_free_coordinate_dimension
                else 0.0
            ),
            "nearest_free_coordinate_dimension": int(
                nearest_free_coordinate_dimension
            ),
            "relations": list(relations),
            "start_eef_position_m": np.asarray(start_position),
            "start_eef_quaternion_xyzw": normalize_quaternion(start_quaternion),
            "final_eef_position_m": final_position,
            "final_eef_quaternion_xyzw": final_quaternion,
            "rigid_transform": {
                "rotation_quaternion_xyzw": mat_to_quat_xyzw(rigid_rotation)
                .astype(float)
                .tolist(),
                "translation_robot_base_m": rigid_translation.astype(float).tolist(),
            },
            "constraints": final_constraints,
            "selected_candidate": {**selected_report, "endpoint_score_rank": int(rank)},
            "feasible_candidate_count": int(len(ranked)),
            "candidate_count": int(len(candidates)),
            "solver": (
                "submission_local_joint_endpoint_with_symbolic_coordinate_constraints"
            ),
            "j8_participates": False,
        }

    ranked_results = [
        endpoint_result(q_final, report, rank)
        for rank, (q_final, report) in enumerate(ranked)
    ]
    selected = dict(ranked_results[0])
    selected["_ranked_endpoint_candidates"] = ranked_results
    return selected


def plan_endpoint_with_trunk_assist(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    source_points_robot_base_m: Sequence[Sequence[float]],
    target_points: Sequence[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    fixed_target_points: Sequence[dict[str, Any]] | None = None,
    pos_tol_m: float,
    ori_tol_deg: float,
    pose_family_search_enabled: bool = True,
    precision_shell_search_enabled: bool = False,
    cancel_requested: Callable[[], bool] | None = None,
    planning_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Solve a tracked endpoint with the three movable trunk joints and one arm.

    This is a precision fallback for :func:`plan_endpoint`.  Its nonlinear
    objectives contain only geometric equations.  Capture-relative movement
    never competes with those equations; it is evaluated afterwards and is a
    tie-break only among endpoints in the same precision class.
    """

    from scipy.optimize import least_squares

    planning_deadline_monotonic = _normalize_planning_deadline(
        planning_deadline_monotonic
    )

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")
        _check_planning_deadline(planning_deadline_monotonic)

    check_cancelled()
    if arm not in ("left", "right"):
        raise ValueError("tracked-point whole-body arm must be left or right")
    q0 = _locked_q(q_start, label="whole-body tracked-point start q")
    raw_trunk_q = getattr(state, "trunk_q", None)
    if raw_trunk_q is None:
        raise ValueError(
            "whole-body tracked-point planning requires a local state with trunk_q"
        )
    trunk0 = np.asarray(raw_trunk_q, dtype=np.float64).reshape(4).copy()
    if not np.all(np.isfinite(trunk0)):
        raise ValueError("whole-body tracked-point start trunk is non-finite")
    if abs(float(trunk0[3])) > 1.0e-6:
        raise ValueError("whole-body tracked-point trunk J4 must be zero")
    trunk0[3] = 0.0
    trunk_limits = np.asarray(TRUNK_JOINT_LIMITS, dtype=np.float64).reshape(4, 2)
    if np.any(trunk0 < trunk_limits[:, 0] - 1.0e-6) or np.any(
        trunk0 > trunk_limits[:, 1] + 1.0e-6
    ):
        raise ValueError("whole-body tracked-point start trunk exceeds local limits")
    trunk0 = np.clip(trunk0, trunk_limits[:, 0], trunk_limits[:, 1])
    trunk0[3] = 0.0

    source = _strict_points(
        source_points_robot_base_m,
        label="whole-body tracked-point source positions",
    )
    normalized_targets = _normalize_target_points(target_points)
    if len(normalized_targets) != len(source):
        raise ValueError("whole-body tracked-point target/source counts do not match")
    controlled_names = [str(point["name"]) for point in normalized_targets]
    if (fixed_points_robot_base_m is None) != (fixed_target_points is None):
        raise ValueError(
            "whole-body fixed points and targets must be supplied together"
        )
    fixed_points = np.empty((0, 3), dtype=np.float64)
    normalized_fixed_targets: list[dict[str, Any]] = []
    if fixed_points_robot_base_m is not None:
        fixed_points = _strict_points(
            fixed_points_robot_base_m,
            label="whole-body fixed tracked-point positions",
        )
        normalized_fixed_targets = _normalize_target_points(
            fixed_target_points or []
        )
        if len(fixed_points) != len(normalized_fixed_targets):
            raise ValueError("whole-body fixed target/source counts do not match")
    if len(source) + len(fixed_points) > MAX_TRACKED_POINTS:
        raise ValueError(
            f"whole-body tracked constraints cannot exceed {MAX_TRACKED_POINTS} points"
        )
    fixed_names = [str(point["name"]) for point in normalized_fixed_targets]
    if set(controlled_names) & set(fixed_names):
        raise ValueError("whole-body controlled/fixed point names overlap")
    canonical_relations = _normalize_relation_records(relations, controlled_names + fixed_names)

    def affine_points(points: np.ndarray) -> np.ndarray:
        return np.vstack((points, fixed_points)) if len(fixed_points) else points

    constraint_model = _build_constraint_model(
        normalized_targets + normalized_fixed_targets
    )
    canonical_inequalities = normalize_tracked_inequalities(
        inequalities or [],
        allowed_variables=set(constraint_model.variable_names),
    )
    source_geometry_rank = geometry_rank(source)
    single_anchor_geometry = source_geometry_rank == 0
    start_position, start_quaternion = eef_pose(state, arm, q0)
    start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
    start_quaternion = normalize_quaternion(start_quaternion)
    start_rotation = quat_to_mat_xyzw(start_quaternion)
    anchors = rigid_anchors_from_points(start_position, start_quaternion, source)
    arm_lower, arm_upper = arm_joint_limits(state, arm)
    arm_lower = np.asarray(arm_lower, dtype=np.float64).reshape(-1)
    arm_upper = np.asarray(arm_upper, dtype=np.float64).reshape(-1)
    arm_control_lower, arm_control_upper, arm_control_margin = (
        _endpoint_control_bounds(
            q0[:7], arm_lower[:7], arm_upper[:7]
        )
    )
    full_lower = np.concatenate((trunk_limits[:3, 0], arm_control_lower))
    full_upper = np.concatenate((trunk_limits[:3, 1], arm_control_upper))
    full_start = np.concatenate((trunk0[:3], q0[:7]))
    full_span = np.maximum(full_upper - full_lower, 0.10)
    geometry_report = geometry_preflight(
        source,
        resolve_fully_numeric_target_points(normalized_targets),
        tolerance_m=ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
    )

    def state_and_q(selected: Sequence[float]) -> tuple[LocalRobotState, np.ndarray, np.ndarray]:
        values = np.asarray(selected, dtype=np.float64).reshape(10)
        trunk = trunk0.copy()
        trunk[:3] = values[:3]
        trunk[3] = 0.0
        q = q0.copy()
        q[:7] = values[3:]
        if ARM_DOF == 8:
            q[7] = 0.0
        if arm == "left":
            candidate_state = replace(state, trunk_q=trunk, arm_left_q=q)
        else:
            candidate_state = replace(state, trunk_q=trunk, arm_right_q=q)
        return candidate_state, q, trunk

    def point_positions(selected: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_state, q, _trunk = state_and_q(selected)
        return _point_positions(candidate_state, arm, q, anchors)

    def evaluate_constraints(points: np.ndarray) -> dict[str, Any]:
        return evaluate_target_constraints(
            points,
            normalized_targets,
            tolerance_m=pos_tol_m,
            relations=canonical_relations,
            inequalities=canonical_inequalities,
            orientation_tolerance_deg=ori_tol_deg,
            fixed_points_robot_base_m=(fixed_points if len(fixed_points) else None),
            fixed_target_points=(
                normalized_fixed_targets if len(fixed_points) else None
            ),
        )

    solver_evaluations = 0

    def geometry_residual(selected: Sequence[float]) -> np.ndarray:
        nonlocal solver_evaluations
        solver_evaluations += 1
        if solver_evaluations > WHOLE_BODY_MAX_FUNCTION_EVALUATIONS:
            raise _SolverBudgetExceeded(
                "whole-body tracked-point IK evaluation budget exhausted"
            )
        check_cancelled()
        points, _position, quaternion = point_positions(selected)
        affine_residual, fitted_values = constraint_model.fit(
            affine_points(points)
        )
        residual: list[float] = list(
            np.asarray(affine_residual, dtype=np.float64)
            / max(ENDPOINT_PRECISE_POSITION_TOLERANCE_M, 1.0e-6)
        )
        inequality_values = _inequality_violation_vector(
            canonical_inequalities,
            constraint_model,
            fitted_values,
        )
        if inequality_values.size:
            residual.extend(
                inequality_values
                / max(ENDPOINT_PRECISE_POSITION_TOLERANCE_M, 1.0e-6)
            )
        for block in relation_residual_blocks(
            affine_points(points),
            canonical_relations,
            point_names=controlled_names + fixed_names,
            _canonical=True,
        ):
            scale = (
                max(
                    math.radians(ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG),
                    1.0e-6,
                )
                if block["kind"] == "orientation"
                else max(
                    relation_position_tolerance_m(
                        block.get("type"),
                        ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
                    ),
                    1.0e-6,
                )
            )
            residual.extend(
                np.asarray(block["residual"], dtype=np.float64).reshape(-1)
                / scale
            )
            if block.get("orientation_residual") is not None:
                residual.extend(
                    np.asarray(
                        block["orientation_residual"], dtype=np.float64
                    ).reshape(-1)
                    / max(
                        math.sin(
                            math.radians(
                                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
                            )
                        ),
                        1.0e-6,
                    )
                )
        if single_anchor_geometry:
            residual.extend(
                orientation_error_vector(quaternion, start_quaternion)
                / max(
                    math.radians(ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG),
                    1.0e-6,
                )
            )
        if not residual:
            residual.append(0.0)
        return np.asarray(residual, dtype=np.float64)

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []

    def whole_body_motion_score(
        selected: np.ndarray,
        resolved_points: np.ndarray,
    ) -> float:
        point_motion = float(
            np.mean(np.sum((resolved_points - source) ** 2, axis=1))
        )
        normalized_joint_motion = float(
            np.mean(((selected - full_start) / full_span) ** 2)
        )
        return point_motion + CANDIDATE_JOINT_MOTION_WEIGHT * normalized_joint_motion

    def add_candidate(selected_raw: Sequence[float], report: Mapping[str, Any]) -> None:
        selected = np.asarray(selected_raw, dtype=np.float64).reshape(10).copy()
        if not np.all(np.isfinite(selected)):
            return
        selected = np.clip(selected, full_lower, full_upper)
        candidate_state, q, trunk = state_and_q(selected)
        points, _position, quaternion = _point_positions(
            candidate_state, arm, q, anchors
        )
        if not all(
            np.all(np.isfinite(value))
            for value in (points, _position, quaternion, q, trunk)
        ):
            return
        constraints = evaluate_constraints(points)
        preserved_orientation_error = (
            orientation_error_deg(quaternion, start_quaternion)
            if single_anchor_geometry
            else 0.0
        )
        accuracy = _candidate_accuracy(
            constraints,
            preserved_eef_orientation_error_deg=preserved_orientation_error,
        )
        candidate_report = {
            **dict(report),
            "constraints": constraints,
            "orientation_error_deg": float(preserved_orientation_error),
            "selection_orientation_error_deg": float(
                accuracy["orientation_error_deg"]
            ),
            "position_error_m": float(accuracy["position_error_m"]),
            "position_error_mm": float(accuracy["position_error_mm"]),
            "accuracy_cost_mm_plus_deg": float(
                accuracy["accuracy_cost_mm_plus_deg"]
            ),
            "precise": bool(accuracy["precise"]),
            "precision_class": _precision_class_label(accuracy),
            "precise_position_tolerance_m": float(
                ENDPOINT_PRECISE_POSITION_TOLERANCE_M
            ),
            "collinear_position_tolerance_m": accuracy.get(
                "collinear_position_tolerance_m"
            ),
            "position_precision_ratio": float(
                accuracy["position_precision_ratio"]
            ),
            "precise_orientation_tolerance_deg": float(
                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
            ),
            "meets_requested_tolerance": bool(
                constraints.get("ok")
                and preserved_orientation_error <= float(ori_tol_deg)
            ),
            "motion_score": float(whole_body_motion_score(selected, points)),
            "trunk_motion_inf_rad": float(
                np.linalg.norm(trunk[:3] - trunk0[:3], ord=np.inf)
            ),
            "arm_motion_inf_rad": float(
                np.linalg.norm(q[:7] - q0[:7], ord=np.inf)
            ),
            "geometry_then_motion_separated": True,
        }
        duplicate_index = next(
            (
                index
                for index, (old_selected, _old_q, _old_trunk, _old_report)
                in enumerate(candidates)
                if np.linalg.norm(old_selected - selected, ord=np.inf) <= 1.0e-6
            ),
            None,
        )
        value = (selected, q.copy(), trunk.copy(), candidate_report)
        if duplicate_index is None:
            candidates.append(value)
        elif _candidate_rank_key(candidate_report) < _candidate_rank_key(
            candidates[duplicate_index][3]
        ):
            candidates[duplicate_index] = value

    def solve_geometry_seed(
        seed: Sequence[float],
        *,
        mode: str,
        max_nfev: int,
        extra_report: Mapping[str, Any] | None = None,
    ) -> np.ndarray | None:
        check_cancelled()
        try:
            solved = least_squares(
                geometry_residual,
                np.clip(np.asarray(seed, dtype=np.float64), full_lower, full_upper),
                bounds=(full_lower, full_upper),
                max_nfev=int(max_nfev),
                xtol=1.0e-9,
                ftol=1.0e-9,
                gtol=1.0e-9,
            )
        except (_SolverBudgetExceeded, PlanningDeadlineExceeded):
            raise
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return None
        report = {
            "mode": mode,
            "solver_success": bool(solved.success),
            "solver_status": int(solved.status),
            "solver_message": str(solved.message),
            "function_evaluations": int(solved.nfev),
            **dict(extra_report or {}),
        }
        add_candidate(solved.x, report)
        return np.asarray(solved.x, dtype=np.float64)

    pose_family_base_signatures: set[tuple[float, ...]] = set()

    def solve_affine_pose_target(
        pose: Mapping[str, Any],
        *,
        seed: Sequence[float],
        base_index: int,
        seed_index: int,
        mode: str = "whole_body_affine_pose_family",
        max_nfev: int = WHOLE_BODY_POSE_FAMILY_MAX_FUNCTION_EVALUATIONS,
        extra_report: Mapping[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Solve one analytically generated EEF pose with local whole-body IK.

        The affine pose-family generator only supplies exact geometric pose
        targets.  Reachability is still decided by the same bounded local FK
        solve as every other endpoint; no simulator IK or collision state is
        consulted here.
        """

        try:
            target_position = np.asarray(
                pose["final_eef_position_m"], dtype=np.float64
            ).reshape(3)
            target_quaternion = normalize_quaternion(
                pose["final_eef_quaternion_xyzw"]
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(target_position)) or not np.all(
            np.isfinite(target_quaternion)
        ):
            return None

        def pose_residual(selected: Sequence[float]) -> np.ndarray:
            nonlocal solver_evaluations
            solver_evaluations += 1
            if solver_evaluations > WHOLE_BODY_MAX_FUNCTION_EVALUATIONS:
                raise _SolverBudgetExceeded(
                    "whole-body tracked-point IK evaluation budget exhausted"
                )
            check_cancelled()
            candidate_state, q, _trunk = state_and_q(selected)
            position, quaternion = eef_pose(candidate_state, arm, q)
            return np.concatenate(
                (
                    (np.asarray(position, dtype=np.float64) - target_position)
                    / max(ENDPOINT_PRECISE_POSITION_TOLERANCE_M, 1.0e-6),
                    orientation_error_vector(
                        quaternion, target_quaternion
                    )
                    / max(
                        math.radians(ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG),
                        1.0e-6,
                    ),
                )
            )

        try:
            solved = least_squares(
                pose_residual,
                np.clip(np.asarray(seed, dtype=np.float64), full_lower, full_upper),
                bounds=(full_lower, full_upper),
                max_nfev=int(max_nfev),
                xtol=1.0e-8,
                ftol=1.0e-8,
                gtol=1.0e-8,
            )
        except (_SolverBudgetExceeded, PlanningDeadlineExceeded):
            raise
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return None

        add_candidate(
            solved.x,
            {
                "mode": str(mode),
                "solver_success": bool(solved.success),
                "solver_status": int(solved.status),
                "solver_message": str(solved.message),
                "function_evaluations": int(solved.nfev),
                "pose_family_angle_deg": float(
                    pose.get("pose_family_angle_deg", 0.0)
                ),
                "pose_family_axis_world": list(
                    pose.get("pose_family_axis_world", [])
                ),
                "pose_family_base_index": int(base_index),
                "pose_family_seed_index": int(seed_index),
                "affine_equation_error_m": float(
                    pose.get("affine_equation_error_m", 0.0)
                ),
                **dict(extra_report or {}),
            },
        )
        return np.asarray(solved.x, dtype=np.float64)

    def run_affine_pose_family() -> None:
        """Add reachable members of every precise affine pose family.

        A direct endpoint can be exact while still representing only one
        member of a rigid-pose null space.  Generate the family from distinct
        precise EEF poses, then let local IK and the downstream RGB-D filter
        choose the physically useful member.  The bounded signature set keeps
        repeated calls (before/after the generic frontier) idempotent.
        """

        if source_geometry_rank <= 0:
            return
        precise_bases = [
            item
            for item in sorted(
                candidates,
                key=lambda item: _candidate_rank_key(item[3]),
            )
            if bool(item[3].get("precise"))
        ]
        for base_index, (selected, _q, _trunk, _report) in enumerate(
            precise_bases[:4]
        ):
            check_cancelled()
            candidate_state, candidate_q, _candidate_trunk = state_and_q(selected)
            base_position, base_quaternion = eef_pose(
                candidate_state, arm, candidate_q
            )
            base_position = np.asarray(base_position, dtype=np.float64).reshape(3)
            base_quaternion = normalize_quaternion(base_quaternion)
            signature = tuple(
                np.round(
                    np.concatenate((base_position, base_quaternion)),
                    decimals=7,
                ).tolist()
            )
            if signature in pose_family_base_signatures:
                continue
            pose_family_base_signatures.add(signature)
            pose_family_poses = _affine_pose_family_targets(
                source_points_robot_base_m=source,
                anchors_eef_m=anchors,
                base_rotation=quat_to_mat_xyzw(base_quaternion),
                base_position=base_position,
                target_points=normalized_targets,
                fixed_points_robot_base_m=(
                    fixed_points if len(fixed_points) else None
                ),
                fixed_target_points=(
                    normalized_fixed_targets if len(fixed_points) else None
                ),
            )
            for pose in pose_family_poses:
                # The base member is already present.  Solving it again only
                # burns the shared evaluation budget and cannot add a new
                # endpoint, while every non-zero family member is useful to
                # the overlap frontier.
                if abs(float(pose.get("pose_family_angle_deg", 0.0))) <= 1.0e-9:
                    continue
                seeds = [np.asarray(selected, dtype=np.float64)]
                # A wrist flip can be in a different local basin.  Give it a
                # capture seed and one deterministic arm seed after the
                # continuation seed, but keep the per-pose search bounded.
                seeds.append(full_start.copy())
                generic = _generic_joint_seeds(
                    q0[:7], arm_control_lower, arm_control_upper
                )
                if len(generic) > 1:
                    seeds.append(np.concatenate((trunk0[:3], generic[1])))
                best_result: np.ndarray | None = None
                for seed_index, seed in enumerate(seeds):
                    result = solve_affine_pose_target(
                        pose,
                        seed=seed,
                        base_index=base_index,
                        seed_index=seed_index,
                    )
                    if result is not None:
                        best_result = result
                        candidate_state, candidate_q, _candidate_trunk = state_and_q(
                            result
                        )
                        points, _position, _quaternion = _point_positions(
                            candidate_state, arm, candidate_q, anchors
                        )
                        if bool(
                            _candidate_accuracy(
                                evaluate_constraints(points),
                                preserved_eef_orientation_error_deg=0.0,
                            ).get("precise")
                        ):
                            break
                if best_result is None:
                    continue

    precision_shell_target_signatures: set[tuple[float, ...]] = set()
    precision_shell_base_signatures: set[tuple[float, ...]] = set()
    precision_shell_targets_generated = 0

    def run_precision_pose_shell() -> None:
        """Add bounded near-exact EEF alternatives for plan-only overlap scoring.

        RGB-D occupancy is quantized.  A pose that is exactly optimal for the
        symbolic equations can therefore overlap a voxel while a nearby pose
        remains inside the strict precision class and is collision-free.  This
        search is deliberately a candidate generator, not a relaxed acceptance
        rule: every result still passes ``add_candidate`` and is ranked by the
        ordinary precision policy.  It is enabled only by the plan-mode
        caller, and its finite shell is never used to alter exec-mode behavior.
        """

        nonlocal precision_shell_targets_generated
        if not precision_shell_search_enabled:
            return
        if precision_shell_targets_generated >= WHOLE_BODY_PRECISION_SHELL_MAX_POSES:
            return
        # Prefer analytically enumerated proper-pose-family members.  If the
        # family is unavailable, ordinary precise endpoints are still valid
        # shell centres.  Do not use shell candidates as new centres.
        def base_priority(item: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]) -> tuple[float, ...]:
            report = item[3]
            mode = str(report.get("mode") or "")
            if mode == "whole_body_affine_pose_family":
                priority = 0.0
            elif mode == "whole_body_exact_line_pose_family":
                priority = 1.0
            else:
                priority = 2.0
            return (priority, *_candidate_rank_key(report))

        precise_bases = [
            item
            for item in sorted(candidates, key=base_priority)
            if bool(item[3].get("precise"))
            and str(item[3].get("mode") or "") != "whole_body_precision_shell"
        ]
        if not precise_bases:
            return
        # A shell around several distinct IK branches is useful for overlap,
        # but the total target cap keeps the added work bounded.  Two centres
        # cover the usual nearest and flipped/trunk-assisted branches while
        # leaving enough evaluations for every cardinal translation.
        for base_index, (selected, _q, _trunk, report) in enumerate(precise_bases[:2]):
            check_cancelled()
            candidate_state, candidate_q, _candidate_trunk = state_and_q(selected)
            base_position, base_quaternion = eef_pose(
                candidate_state, arm, candidate_q
            )
            base_position = np.asarray(base_position, dtype=np.float64).reshape(3)
            base_quaternion = normalize_quaternion(base_quaternion)
            base_signature = tuple(
                np.round(
                    np.concatenate((base_position, base_quaternion)), decimals=7
                ).tolist()
            )
            if base_signature in precision_shell_base_signatures:
                continue
            precision_shell_base_signatures.add(base_signature)
            base_rotation = quat_to_mat_xyzw(base_quaternion)

            offsets: list[tuple[str, np.ndarray, float, int]] = []
            for offset_index, raw_offset in enumerate(
                WHOLE_BODY_PRECISION_SHELL_TRANSLATION_OFFSETS_MM
            ):
                offsets.append(
                    (
                        "translation",
                        np.asarray(raw_offset, dtype=np.float64).reshape(3) * 1.0e-3,
                        0.0,
                        int(offset_index),
                    )
                )
            rotation_offset_start = len(offsets)
            for rotation_index, (axis_index, angle_deg) in enumerate(
                WHOLE_BODY_PRECISION_SHELL_ROTATION_OFFSETS_DEG
            ):
                axis = np.zeros(3, dtype=np.float64)
                axis[int(axis_index)] = 1.0
                offsets.append(
                    (
                        "rotation",
                        axis,
                        float(angle_deg),
                        int(rotation_offset_start + rotation_index),
                    )
                )

            for offset_kind, offset_value, angle_deg, offset_index in offsets:
                if precision_shell_targets_generated >= WHOLE_BODY_PRECISION_SHELL_MAX_POSES:
                    return
                check_cancelled()
                if offset_kind == "translation":
                    target_position = base_position + offset_value
                    target_rotation = base_rotation
                else:
                    target_position = base_position.copy()
                    target_rotation = _axis_rotation(
                        offset_value, math.radians(float(angle_deg))
                    ) @ base_rotation
                target_quaternion = mat_to_quat_xyzw(target_rotation)
                target_signature = tuple(
                    np.round(
                        np.concatenate((target_position, target_quaternion)),
                        decimals=7,
                    ).tolist()
                )
                if target_signature in precision_shell_target_signatures:
                    continue
                precision_shell_target_signatures.add(target_signature)
                precision_shell_targets_generated += 1
                pose = {
                    "final_eef_position_m": target_position.astype(float).tolist(),
                    "final_eef_quaternion_xyzw": target_quaternion.astype(float).tolist(),
                    "pose_family": "precision_pose_shell",
                    "pose_family_base_index": int(base_index),
                    "precision_shell_offset_kind": str(offset_kind),
                    "precision_shell_offset_index": int(offset_index),
                    "precision_shell_translation_m": (
                        offset_value.astype(float).tolist()
                        if offset_kind == "translation"
                        else [0.0, 0.0, 0.0]
                    ),
                    "precision_shell_rotation_deg": (
                        float(angle_deg) if offset_kind == "rotation" else 0.0
                    ),
                    "affine_equation_error_m": 0.0,
                }
                # The centre branch is the best initial basin.  Only a
                # non-precise result receives a bounded capture-seed retry;
                # this keeps the shell cheap while preserving a second branch
                # for a difficult local pose.
                shell_extra_report = {
                    "precision_shell_offset_kind": str(offset_kind),
                    "precision_shell_offset_index": int(offset_index),
                    "precision_shell_translation_m": pose[
                        "precision_shell_translation_m"
                    ],
                    "precision_shell_rotation_deg": pose[
                        "precision_shell_rotation_deg"
                    ],
                }
                try:
                    result = solve_affine_pose_target(
                        pose,
                        seed=np.asarray(selected, dtype=np.float64),
                        base_index=base_index,
                        seed_index=0,
                        mode="whole_body_precision_shell",
                        max_nfev=WHOLE_BODY_PRECISION_SHELL_MAX_FUNCTION_EVALUATIONS,
                        extra_report=shell_extra_report,
                    )
                except (_SolverBudgetExceeded, PlanningDeadlineExceeded):
                    return
                if result is None:
                    continue
                solved_state, solved_q, _solved_trunk = state_and_q(result)
                solved_points, _solved_position, _solved_quaternion = _point_positions(
                    solved_state, arm, solved_q, anchors
                )
                shell_precise = bool(
                    _candidate_accuracy(
                        evaluate_constraints(solved_points),
                        preserved_eef_orientation_error_deg=0.0,
                    ).get("precise")
                )
                if not shell_precise:
                    try:
                        solve_affine_pose_target(
                            pose,
                            seed=full_start.copy(),
                            base_index=base_index,
                            seed_index=1,
                            mode="whole_body_precision_shell",
                            max_nfev=WHOLE_BODY_PRECISION_SHELL_MAX_FUNCTION_EVALUATIONS,
                            extra_report=shell_extra_report,
                        )
                    except (_SolverBudgetExceeded, PlanningDeadlineExceeded):
                        return

    # Preserve a finite best-effort result even when every nonlinear solve is
    # interrupted by a deadline.  It cannot outrank a subsequently found
    # precise endpoint because precision is the first rank component.
    add_candidate(
        full_start,
        {
            "mode": "whole_body_capture_seed",
            "solver_success": True,
            "function_evaluations": 0,
        },
    )
    planning_budget_exhausted = False
    try:
        solve_geometry_seed(
            full_start,
            mode="whole_body_geometry_only_direct",
            max_nfev=WHOLE_BODY_DIRECT_MAX_FUNCTION_EVALUATIONS,
        )
        # The direct geometry solve often finds an exact endpoint quickly,
        # but that endpoint can be the high-overlap member of an otherwise
        # reachable rigid-pose family.  Populate alternate proper rotations
        # before the generic joint frontier so overlap scoring can see them.
        if pose_family_search_enabled:
            run_affine_pose_family()
        if precision_shell_search_enabled and any(
            str(item[3].get("mode") or "")
            == "whole_body_affine_pose_family"
            and bool(item[3].get("precise"))
            for item in candidates
        ):
            run_precision_pose_shell()

        # Two tracked points constrained to a scene line have one free roll
        # DOF.  Generate exact rigid transforms for both line directions and
        # a deterministic roll grid, then solve each target EEF pose with the
        # movable trunk and arm.  This covers the full geometry family instead
        # of hoping a generic joint seed discovers its reachable member.
        line_specs: list[tuple[int, int, np.ndarray, np.ndarray, str]] = []
        name_to_index = {name: index for index, name in enumerate(controlled_names)}
        for relation in canonical_relations:
            relation_type = str(relation.get("type") or "")
            names = list(relation.get("point_names") or [])
            if relation_type not in {
                "line_coincident",
                "line_segment_overlap",
                "line_segment_contains",
            }:
                continue
            if len(names) != 2 or any(name not in name_to_index for name in names):
                continue
            if relation_type in {"line_segment_overlap", "line_segment_contains"}:
                line_start = np.asarray(
                    relation["segment_start_robot_base_m"], dtype=np.float64
                ).reshape(3)
                line_end = np.asarray(
                    relation["segment_end_robot_base_m"], dtype=np.float64
                ).reshape(3)
            else:
                line_start = np.asarray(
                    relation["line_point_robot_base_m"], dtype=np.float64
                ).reshape(3)
                line_end = line_start + np.asarray(
                    relation["line_direction_robot_base"], dtype=np.float64
                ).reshape(3)
            if float(np.linalg.norm(line_end - line_start)) <= 1.0e-9:
                continue
            line_specs.append(
                (
                    name_to_index[names[0]],
                    name_to_index[names[1]],
                    line_start,
                    line_end,
                    relation_type,
                )
            )

        analytic_results: list[tuple[np.ndarray, int, float]] = []
        roll_step_rad = math.radians(WHOLE_BODY_LINE_ROLL_SAMPLE_DEG)

        def solve_line_pose(
            spec: tuple[int, int, np.ndarray, np.ndarray, str],
            *,
            sign: int,
            roll_rad: float,
            seed: Sequence[float],
            refinement: bool,
        ) -> np.ndarray | None:
            index_a, index_b, line_start, line_end, relation_type = spec
            source_axis = source[index_b] - source[index_a]
            source_length = float(np.linalg.norm(source_axis))
            reference_delta = line_end - line_start
            reference_length = float(np.linalg.norm(reference_delta))
            if source_length <= 1.0e-9 or reference_length <= 1.0e-9:
                return None
            source_direction = source_axis / source_length
            reference_direction = reference_delta / reference_length
            desired_direction = float(sign) * reference_direction
            rigid_rotation = _axis_rotation(
                desired_direction, float(roll_rad)
            ) @ _align_vectors(source_direction, desired_direction)
            signed_offsets = np.asarray((0.0, float(sign) * source_length))
            source_pair = source[[index_a, index_b]]
            unconstrained_alpha = float(
                np.mean(
                    [
                        np.dot(source_pair[index] - line_start, reference_direction)
                        - signed_offsets[index]
                        for index in range(2)
                    ]
                )
            )
            if relation_type == "line_segment_contains":
                if sign < 0:
                    return None
                required_span = (
                    reference_length + 2.0 * ORDERED_COLLINEAR_INNER_MARGIN_M
                )
                if source_length >= required_span:
                    alpha = float(
                        np.clip(
                            unconstrained_alpha,
                            reference_length
                            + ORDERED_COLLINEAR_INNER_MARGIN_M
                            - source_length,
                            -ORDERED_COLLINEAR_INNER_MARGIN_M,
                        )
                    )
                else:
                    # The requested strict interior margin is impossible when
                    # the controlled rigid segment is too short.  Center it on
                    # the reference for the minimum-max-error best-effort seed.
                    alpha = 0.5 * (reference_length - source_length)
            elif relation_type == "line_segment_overlap":
                alpha_bounds = (
                    (-source_length, reference_length)
                    if sign > 0
                    else (0.0, reference_length + source_length)
                )
                alpha = float(np.clip(unconstrained_alpha, *alpha_bounds))
            else:
                alpha = unconstrained_alpha
            rigid_translation = (
                line_start
                + alpha * reference_direction
                - rigid_rotation @ source[index_a]
            )
            target_eef_position = (
                rigid_rotation @ start_position + rigid_translation
            )
            target_eef_quaternion = mat_to_quat_xyzw(
                rigid_rotation @ start_rotation
            )

            def pose_residual(selected: Sequence[float]) -> np.ndarray:
                nonlocal solver_evaluations
                solver_evaluations += 1
                if solver_evaluations > WHOLE_BODY_MAX_FUNCTION_EVALUATIONS:
                    raise _SolverBudgetExceeded(
                        "whole-body tracked-point IK evaluation budget exhausted"
                    )
                check_cancelled()
                candidate_state, q, _trunk = state_and_q(selected)
                position, quaternion = eef_pose(candidate_state, arm, q)
                return np.concatenate(
                    (
                        (np.asarray(position) - target_eef_position)
                        / relation_position_tolerance_m(
                            relation_type,
                            ENDPOINT_PRECISE_POSITION_TOLERANCE_M,
                        ),
                        orientation_error_vector(
                            quaternion, target_eef_quaternion
                        )
                        / math.radians(ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG),
                    )
                )

            solved = least_squares(
                pose_residual,
                np.clip(np.asarray(seed, dtype=np.float64), full_lower, full_upper),
                bounds=(full_lower, full_upper),
                max_nfev=WHOLE_BODY_POSE_MAX_FUNCTION_EVALUATIONS,
                xtol=1.0e-8,
                ftol=1.0e-8,
                gtol=1.0e-8,
            )
            candidate_state, candidate_q, _candidate_trunk = state_and_q(solved.x)
            solved_position, solved_quaternion = eef_pose(
                candidate_state, arm, candidate_q
            )
            add_candidate(
                solved.x,
                {
                    "mode": "whole_body_exact_line_pose_family",
                    "solver_success": bool(solved.success),
                    "solver_status": int(solved.status),
                    "solver_message": str(solved.message),
                    "function_evaluations": int(solved.nfev),
                    "line_direction_sign": int(sign),
                    "line_roll_deg": float(math.degrees(roll_rad)),
                    "line_alpha_m": float(alpha),
                    "line_roll_refinement": bool(refinement),
                    "pose_position_error_m": float(
                        np.linalg.norm(
                            np.asarray(solved_position) - target_eef_position
                        )
                    ),
                    "pose_orientation_error_deg": float(
                        orientation_error_deg(
                            solved_quaternion, target_eef_quaternion
                        )
                    ),
                },
            )
            return np.asarray(solved.x, dtype=np.float64)

        for spec in line_specs[:1]:
            allowed_signs = (1,) if spec[4] == "line_segment_contains" else (1, -1)
            for sign in allowed_signs:
                warm_seed = full_start.copy()
                for roll_deg in np.arange(-180.0, 180.0, WHOLE_BODY_LINE_ROLL_SAMPLE_DEG):
                    solved_x = solve_line_pose(
                        spec,
                        sign=sign,
                        roll_rad=math.radians(float(roll_deg)),
                        seed=warm_seed,
                        refinement=False,
                    )
                    if solved_x is not None:
                        warm_seed = solved_x
                        analytic_results.append((solved_x, sign, math.radians(float(roll_deg))))

            for sign in allowed_signs:
                precise_for_sign = [
                    item
                    for item in candidates
                    if item[3].get("mode") == "whole_body_exact_line_pose_family"
                    and int(item[3].get("line_direction_sign", 0)) == sign
                    and bool(item[3].get("precise"))
                ]
                if not precise_for_sign:
                    continue
                best = min(precise_for_sign, key=lambda item: _candidate_rank_key(item[3]))
                center = math.radians(float(best[3]["line_roll_deg"]))
                for direction in (-1.0, 1.0):
                    solve_line_pose(
                        spec,
                        sign=sign,
                        roll_rad=center + direction * 0.5 * roll_step_rad,
                        seed=best[0],
                        refinement=True,
                    )

        # A broad deterministic frontier is needed for plan-mode overlap
        # selection, where a precise endpoint can still be the wrong wrist
        # branch.  Exec mode keeps the historical early-stop behavior when a
        # precise endpoint is already available; its fallback still explores
        # this frontier when no precise endpoint was found.
        run_generic_frontier = bool(
            pose_family_search_enabled
            or not any(bool(item[3].get("precise")) for item in candidates)
        )
        if run_generic_frontier:
            seeds: list[np.ndarray] = []
            for trunk_index in range(3):
                for direction in (-1.0, 1.0):
                    seed = full_start.copy()
                    seed[trunk_index] += direction * 0.12 * full_span[trunk_index]
                    seeds.append(np.clip(seed, full_lower, full_upper))
            for arm_seed in _generic_joint_seeds(
                q0[:7], arm_control_lower, arm_control_upper
            )[1:7]:
                seeds.append(np.concatenate((trunk0[:3], arm_seed)))
            for seed_index, seed in enumerate(seeds):
                solve_geometry_seed(
                    seed,
                    mode="whole_body_geometry_only_multiseed",
                    max_nfev=WHOLE_BODY_DIRECT_MAX_FUNCTION_EVALUATIONS,
                    extra_report={"seed_index": int(seed_index)},
                )
            # If the direct endpoint was not precise, a generic seed may have
            # discovered a precise pose with a different orientation.  Run the
            # same family expansion once more; signatures make this idempotent
            # for the common direct-precise case.
            if pose_family_search_enabled:
                run_affine_pose_family()
            if precision_shell_search_enabled:
                run_precision_pose_shell()
    except (PlanningDeadlineExceeded, _SolverBudgetExceeded):
        planning_budget_exhausted = True

    if not candidates:
        if planning_budget_exhausted:
            raise PlanningDeadlineExceeded(
                "whole-body tracked-point endpoint planning budget exhausted"
            )
        raise ValueError("whole-body tracked-point endpoint solver produced no candidate")
    ordered_candidates = [
        item
        for item in candidates
        if item[3]["constraints"].get("axial_order_satisfied", True)
        and item[3]["constraints"].get("inequality_constraints_ok", True)
    ]
    if not ordered_candidates:
        if canonical_inequalities:
            raise ValueError(
                "no whole-body IK endpoint satisfies the requested strict "
                "inequality constraints"
            )
        raise ValueError("no whole-body IK endpoint satisfies the requested collinear axial order")
    ranked = _whole_body_endpoint_frontier(ordered_candidates)
    precise_candidate_count = sum(
        1 for _selected, _q, _trunk, report in candidates if report["precise"]
    )
    requested_tolerance_candidate_count = sum(
        1
        for _selected, _q, _trunk, report in candidates
        if report["meets_requested_tolerance"]
    )

    def endpoint_result(
        selected: np.ndarray,
        q_final: np.ndarray,
        trunk_final: np.ndarray,
        selected_report: dict[str, Any],
        rank: int,
    ) -> dict[str, Any]:
        final_state, _q, _trunk = state_and_q(selected)
        final_points, final_position, final_quaternion = _point_positions(
            final_state, arm, q_final, anchors
        )
        final_constraints = evaluate_constraints(final_points)
        rigid_rotation = quat_to_mat_xyzw(final_quaternion) @ start_rotation.T
        rigid_translation = final_position - rigid_rotation @ start_position
        return {
            "q_start": q0.copy(),
            "q_final": q_final.copy(),
            "trunk_q_start": trunk0.copy(),
            "trunk_q_final": trunk_final.copy(),
            "trunk_assisted": True,
            "anchors_eef_m": anchors.copy(),
            "source_points_robot_base_m": source.copy(),
            "fixed_points_robot_base_m": fixed_points.copy(),
            "fixed_target_points": list(normalized_fixed_targets),
            "resolved_points_robot_base_m": final_points,
            "geometry_rank": int(source_geometry_rank),
            "geometry_preflight": geometry_report,
            "kabsch_seed": None,
            "planning_budget_exhausted": bool(planning_budget_exhausted),
            "solver_evaluations": int(solver_evaluations),
            "best_effort_refinement_evaluations": 0,
            "best_effort_refinement_attempts": [],
            "precise_motion_refinement_evaluations": 0,
            "precise_motion_refinement_attempts": [],
            "selection_policy": _endpoint_selection_policy(canonical_relations),
            "inequalities": [dict(item) for item in canonical_inequalities],
            "precise_position_tolerance_m": float(
                ENDPOINT_PRECISE_POSITION_TOLERANCE_M
            ),
            "collinear_position_tolerance_m": (
                COLLINEAR_POSITION_TOLERANCE_M
                if any(
                    str(item.get("type") or "").strip().lower()
                    in COLLINEAR_RELATION_TYPES
                    for item in canonical_relations
                )
                else None
            ),
            "precise_orientation_tolerance_deg": float(
                ENDPOINT_PRECISE_ORIENTATION_TOLERANCE_DEG
            ),
            "precise_candidate_count": int(precise_candidate_count),
            "requested_tolerance_candidate_count": int(
                requested_tolerance_candidate_count
            ),
            "selected_is_precise": bool(selected_report["precise"]),
            "selected_meets_requested_tolerance": bool(
                selected_report["meets_requested_tolerance"]
            ),
            "nearest_free_coordinate_weight": 0.0,
            "nearest_free_coordinate_dimension": 0,
            "relations": list(canonical_relations),
            "start_eef_position_m": start_position.copy(),
            "start_eef_quaternion_xyzw": start_quaternion.copy(),
            "final_eef_position_m": final_position,
            "final_eef_quaternion_xyzw": final_quaternion,
            "rigid_transform": {
                "rotation_quaternion_xyzw": mat_to_quat_xyzw(rigid_rotation)
                .astype(float)
                .tolist(),
                "translation_robot_base_m": rigid_translation.astype(float).tolist(),
            },
            "constraints": final_constraints,
            "selected_candidate": {
                **selected_report,
                "endpoint_score_rank": int(rank),
            },
            "feasible_candidate_count": int(len(ranked)),
            "candidate_count": int(len(candidates)),
            "solver": (
                "submission_local_whole_body_geometry_first_trunk_arm_endpoint"
            ),
            "j8_participates": False,
            "trunk_j4_participates": False,
            "pose_family_search_enabled": bool(pose_family_search_enabled),
            "precision_shell_search_enabled": bool(precision_shell_search_enabled),
            "precision_shell_targets_generated": int(
                precision_shell_targets_generated
            ),
            "geometry_then_motion_separated": True,
            "arm_joint_limit_safety_margin_rad": arm_control_margin.astype(float).tolist(),
        }

    ranked_results = [
        endpoint_result(selected, q, trunk, report, rank)
        for rank, (selected, q, trunk, report) in enumerate(ranked)
    ]
    selected_result = dict(ranked_results[0])
    selected_result["_ranked_endpoint_candidates"] = ranked_results
    return selected_result


def quaternion_slerp(
    start: Sequence[float],
    target: Sequence[float],
    fraction: float,
) -> np.ndarray:
    q0 = normalize_quaternion(start)
    q1 = normalize_quaternion(target)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    s = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - s) * q0 + s * q1)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - s) * angle) / sine * q0
        + math.sin(s * angle) / sine * q1
    )


def _trapezoidal_path_progress(fraction: float) -> float:
    """Map wall-clock fraction to one globally eased path fraction."""

    value = float(np.clip(fraction, 0.0, 1.0))
    ramp = float(GLOBAL_TIME_RAMP_FRACTION)
    if not 0.0 < ramp < 0.5:
        raise ValueError("global trajectory ramp fraction must be in (0, 0.5)")
    peak_speed = 1.0 / (1.0 - ramp)
    if value < ramp:
        return 0.5 * peak_speed * value * value / ramp
    if value <= 1.0 - ramp:
        return peak_speed * (value - 0.5 * ramp)
    remaining = 1.0 - value
    return 1.0 - 0.5 * peak_speed * remaining * remaining / ramp


def _global_time_parameterize_knots(
    knots: Sequence[tuple[float, np.ndarray]],
    *,
    max_joint_step_rad: float,
    max_waypoints: int,
    check_cancelled: Callable[[], None],
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    """Resample an IK polyline with one acceleration/deceleration envelope."""

    fractions = np.asarray([float(item[0]) for item in knots], dtype=np.float64)
    joint_knots = np.asarray([item[1] for item in knots], dtype=np.float64)
    if joint_knots.ndim != 2 or joint_knots.shape[0] < 2:
        raise ValueError("Cartesian path must contain start and final IK knots")
    segment_lengths = np.max(np.abs(np.diff(joint_knots, axis=0)), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return [joint_knots[-1].copy()], [1.0], {
            "joint_path_length_inf_rad": 0.0,
            "global_time_ramp_fraction": float(GLOBAL_TIME_RAMP_FRACTION),
            "minimum_path_progress_increment_rad": 0.0,
            "internal_stop_count": 0,
        }

    ramp = float(GLOBAL_TIME_RAMP_FRACTION)
    peak_speed = 1.0 / (1.0 - ramp)
    step_count = max(
        1,
        int(
            math.ceil(
                (peak_speed + 0.01)
                * total_length
                / max(float(max_joint_step_rad), 1e-6)
            )
        ),
    )
    if step_count > int(max_waypoints):
        raise ValueError(
            "globally time-parameterized Cartesian trajectory requires "
            f"{step_count} waypoints, exceeding max_waypoints={int(max_waypoints)}"
        )

    waypoints: list[np.ndarray] = []
    path_fractions: list[float] = []
    path_distances: list[float] = []
    for step in range(1, step_count + 1):
        check_cancelled()
        time_fraction = step / step_count
        path_distance = _trapezoidal_path_progress(time_fraction) * total_length
        segment = min(
            len(segment_lengths) - 1,
            max(0, int(np.searchsorted(cumulative, path_distance, side="right") - 1)),
        )
        segment_length = float(segment_lengths[segment])
        local_fraction = (
            1.0
            if segment_length <= 1e-12
            else float(
                np.clip(
                    (path_distance - float(cumulative[segment])) / segment_length,
                    0.0,
                    1.0,
                )
            )
        )
        q = (
            (1.0 - local_fraction) * joint_knots[segment]
            + local_fraction * joint_knots[segment + 1]
        )
        if ARM_DOF == 8:
            q[7] = 0.0
        cartesian_fraction = float(
            (1.0 - local_fraction) * fractions[segment]
            + local_fraction * fractions[segment + 1]
        )
        waypoints.append(q)
        path_fractions.append(cartesian_fraction)
        path_distances.append(path_distance)

    waypoints[-1] = joint_knots[-1].copy()
    path_fractions[-1] = 1.0
    path_distances[-1] = total_length
    previous = joint_knots[0]
    maximum_step = 0.0
    for q in waypoints:
        maximum_step = max(
            maximum_step,
            float(np.linalg.norm(q - previous, ord=np.inf)),
        )
        previous = q
    if maximum_step > float(max_joint_step_rad) + 1e-9:
        raise ValueError(
            "global time parameterization exceeds max joint step: "
            f"{maximum_step:.6f}rad > {float(max_joint_step_rad):.6f}rad"
        )
    increments = np.diff(np.asarray([0.0, *path_distances], dtype=np.float64))
    internal_stops = int(np.sum(increments[1:-1] <= 1e-12))
    return waypoints, path_fractions, {
        "joint_path_length_inf_rad": total_length,
        "global_time_ramp_fraction": ramp,
        "minimum_path_progress_increment_rad": float(np.min(increments)),
        "internal_stop_count": internal_stops,
        "actual_max_joint_step_rad": maximum_step,
    }


def _plan_cartesian_trajectory_once(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    q_final: Sequence[float],
    pos_tol_m: float,
    ori_tol_deg: float,
    max_joint_step_rad: float,
    max_waypoints: int,
    translation_step_m: float,
    orientation_step_deg: float,
    tracked_anchors_eef: Sequence[Sequence[float]] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    planning_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Solve one straight-position/SLERP sampling of a local IK branch."""

    planning_deadline_monotonic = _normalize_planning_deadline(
        planning_deadline_monotonic
    )

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")
        _check_planning_deadline(planning_deadline_monotonic)

    check_cancelled()
    solver_check = (
        check_cancelled
        if planning_deadline_monotonic is not None
        or callable(cancel_requested)
        else None
    )
    orientation_step_deg = float(orientation_step_deg)
    if not math.isfinite(orientation_step_deg) or orientation_step_deg <= 0.0:
        raise ValueError("Cartesian orientation sampling step must be positive")
    translation_step_m = float(translation_step_m)
    if not math.isfinite(translation_step_m) or translation_step_m <= 0.0:
        raise ValueError("Cartesian translation sampling step must be positive")

    q0 = _locked_q(q_start, label="Cartesian path start q")
    q_goal = _locked_q(q_final, label="Cartesian path final q")
    start_position, start_quaternion = eef_pose(state, arm, q0)
    final_position, final_quaternion = eef_pose(state, arm, q_goal)
    translation = float(np.linalg.norm(np.asarray(final_position) - np.asarray(start_position)))
    orientation = orientation_error_deg(start_quaternion, final_quaternion)
    position_path_limit = max(0.025, 2.5 * float(pos_tol_m))
    orientation_path_limit = max(8.0, 2.0 * float(ori_tol_deg))
    knot_count = max(
        1,
        int(math.ceil(translation / translation_step_m)),
        int(math.ceil(orientation / orientation_step_deg)),
    )
    knots: list[tuple[float, np.ndarray]] = [(0.0, q0)]
    ik_reports: list[dict[str, Any]] = []
    dropped_knots: list[dict[str, Any]] = []
    endpoint_branch_jump_exceeded = False
    fast_continuation_attempt_count = 0
    fast_continuation_success_count = 0
    legacy_ik_call_count = 0
    previous = q0
    for index in range(1, knot_count + 1):
        check_cancelled()
        fraction = index / knot_count
        target_position = (
            (1.0 - fraction) * np.asarray(start_position)
            + fraction * np.asarray(final_position)
        )
        target_quaternion = quaternion_slerp(
            start_quaternion, final_quaternion, fraction
        )
        _seed_index = 0
        if index == knot_count:
            solved_q = q_goal.copy()
            report = {
                "ok": True,
                "solver": "signed_endpoint",
                "pos_err_m": 0.0,
                "ori_err_deg": 0.0,
            }
        else:
            # The signed endpoint already identifies the desired IK branch.
            # Predict this knot on the direct start-to-end joint homotopy, then
            # project that prediction onto the exact Cartesian line/SLERP pose.
            # Seeding every solve only from the preceding knot leaves the
            # redundant seventh joint unconstrained and can enter a slow or
            # discontinuous local basin during large rotations.
            branch_seed = (
                (1.0 - fraction) * q0
                + fraction * q_goal
            )
            strict_pos_tol = min(0.006, max(0.002, float(pos_tol_m)))
            strict_ori_tol = min(3.0, max(1.0, float(ori_tol_deg)))
            candidate_solutions: list[tuple[np.ndarray, dict[str, Any], int]] = []
            fast_continuation_accepted = False
            # In the live evaluator path the preceding knot is already a
            # verified point on the selected IK branch.  Try that continuation
            # first with a deliberately bounded solve.  This is an
            # acceleration only: a miss falls through to the unchanged
            # multi-seed retry below, and a branch jump is never accepted.
            if planning_deadline_monotonic is not None:
                fast_continuation_attempt_count += 1
                try:
                    fast_q, fast_report = solve_pose_target(
                        state,
                        arm,
                        previous,
                        target_position=target_position,
                        target_quaternion=target_quaternion,
                        joint_indices=range(7),
                        pos_tol_m=strict_pos_tol,
                        ori_tol_deg=strict_ori_tol,
                        max_iterations=FAST_CARTESIAN_CONTINUATION_MAX_ITERATIONS,
                        check_requested=solver_check,
                    )
                    fast_q = _locked_q(
                        fast_q,
                        label=f"Cartesian knot {index} continuation",
                    )
                    fast_jump = float(
                        np.linalg.norm(fast_q[:7] - previous[:7], ord=np.inf)
                    )
                    if bool((fast_report or {}).get("ok")) and fast_jump <= DEFAULT_MAX_BRANCH_JUMP_RAD:
                        candidate_solutions.append(
                            (
                                fast_q,
                                {
                                    **dict(fast_report or {}),
                                    "solver": str(
                                        (fast_report or {}).get(
                                            "solver", "local_pose_ik_continuation"
                                        )
                                    ),
                                    "fast_continuation": True,
                                    "branch_jump_inf_rad": fast_jump,
                                },
                                -1,
                            )
                        )
                        fast_continuation_accepted = True
                        fast_continuation_success_count += 1
                except PlanningDeadlineExceeded:
                    raise
                except (RuntimeError, TypeError, ValueError):
                    # The legacy retries below carry the established public
                    # tolerance projection behavior for a failed fast solve.
                    pass
            # A single numerical IK miss is a bad sample, not proof that the
            # endpoint or the whole branch is impossible.  Retry from the
            # signed homotopy, the last accepted knot, and the endpoint.  The
            # first successful solve remains the legacy path, so existing
            # successful cases receive identical commands and diagnostics.
            seed_candidates = (
                []
                if fast_continuation_accepted
                else [branch_seed, previous, q_goal]
            )
            seen_seeds: list[np.ndarray] = []
            for seed_index, seed in enumerate(seed_candidates):
                seed_array = np.asarray(seed, dtype=np.float64).reshape(ARM_DOF)
                if any(np.allclose(seed_array, old, atol=1.0e-12, rtol=0.0) for old in seen_seeds):
                    continue
                seen_seeds.append(seed_array.copy())
                try:
                    legacy_ik_call_count += 1
                    candidate_q, candidate_report = solve_pose_target(
                        state,
                        arm,
                        seed_array,
                        target_position=target_position,
                        target_quaternion=target_quaternion,
                        joint_indices=range(7),
                        pos_tol_m=strict_pos_tol,
                        ori_tol_deg=strict_ori_tol,
                        max_iterations=160,
                        check_requested=solver_check,
                    )
                except PlanningDeadlineExceeded:
                    # ``solve_pose_target`` invokes the cooperative hook from
                    # inside its residual function.  Preserve the deadline
                    # signal so the caller switches to the verified endpoint
                    # joint-space fallback immediately.
                    raise
                except (RuntimeError, TypeError, ValueError) as exc:
                    candidate_solutions.append(
                        (
                            np.full(ARM_DOF, np.nan, dtype=np.float64),
                            {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "solver": "local_pose_ik_exception",
                            },
                            seed_index,
                        )
                    )
                    continue
                try:
                    candidate_q = _locked_q(
                        candidate_q,
                        label=f"Cartesian knot {index} retry {seed_index}",
                    )
                except (TypeError, ValueError) as exc:
                    candidate_solutions.append(
                        (
                            np.full(ARM_DOF, np.nan, dtype=np.float64),
                            {
                                "ok": False,
                                "error": str(exc),
                                "solver": "local_pose_ik_invalid_output",
                            },
                            seed_index,
                        )
                    )
                    continue
                candidate_solutions.append(
                    (candidate_q, dict(candidate_report or {}), seed_index)
                )
                # The historical selector always preferred seed 0 when it
                # was valid and continuous.  Stop evaluating redundant seeds
                # in exactly that case; all other cases retain the full retry
                # frontier and ranking behavior.
                if seed_index == 0 and bool((candidate_report or {}).get("ok")):
                    candidate_jump = float(
                        np.linalg.norm(candidate_q[:7] - previous[:7], ord=np.inf)
                    )
                    if candidate_jump <= DEFAULT_MAX_BRANCH_JUMP_RAD:
                        break

            valid_solutions = [
                item for item in candidate_solutions if bool(item[1].get("ok"))
            ]
            if valid_solutions:
                # Preserve the old first-solve choice whenever it is valid;
                # retries only repair an actual failure or a discontinuous
                # branch.  Sorting by branch distance then error gives a
                # deterministic same-branch recovery for the latter case.
                valid_solutions.sort(
                    key=lambda item: (
                        float(np.linalg.norm(item[0][:7] - previous[:7], ord=np.inf)),
                        float(item[1].get("pos_err_m", float("inf"))),
                        float(item[1].get("ori_err_deg", float("inf"))),
                        int(item[2]),
                    )
                )
                # The signed homotopy result is preferred if it is already a
                # valid, continuous solution.  This preserves legacy output.
                preferred = next(
                    (
                        item
                        for item in valid_solutions
                        if int(item[2]) == 0
                        and float(np.linalg.norm(item[0][:7] - previous[:7], ord=np.inf))
                        <= DEFAULT_MAX_BRANCH_JUMP_RAD
                    ),
                    valid_solutions[0],
                )
                solved_q, report, _seed_index = preferred
            else:
                # Keep the previous public-tolerance acceptance rule, but use
                # the best finite failed report rather than averaging errors.
                failed_reports = [
                    item
                    for item in candidate_solutions
                    if math.isfinite(float(item[1].get("pos_err_m", float("nan"))))
                    and math.isfinite(float(item[1].get("ori_err_deg", float("nan"))))
                ]
                best_failed = min(
                    failed_reports,
                    key=lambda item: (
                        float(item[1].get("pos_err_m", float("inf"))) / max(position_path_limit, 1.0e-9)
                        + float(item[1].get("ori_err_deg", float("inf"))) / max(orientation_path_limit, 1.0e-9),
                    ),
                    default=None,
                )
                if best_failed is not None:
                    solved_q, report, _seed_index = best_failed
                    pos_error = float(report.get("pos_err_m", float("nan")))
                    ori_error = float(report.get("ori_err_deg", float("nan")))
                    within_path_tolerance = bool(
                        math.isfinite(pos_error)
                        and math.isfinite(ori_error)
                        and pos_error <= position_path_limit
                        and ori_error <= orientation_path_limit
                    )
                    if within_path_tolerance:
                        # A finite projection inside the public corridor can
                        # still be used as the knot, exactly as before.
                        report = {
                            **report,
                            "ok": True,
                            "accepted_by_public_tolerance": True,
                            "accepted_by_path_tolerance": True,
                            "strict_pos_tol_m": strict_pos_tol,
                            "strict_ori_tol_deg": strict_ori_tol,
                            "public_pos_tol_m": float(pos_tol_m),
                            "public_ori_tol_deg": float(ori_tol_deg),
                            "path_pos_tol_m": float(position_path_limit),
                            "path_ori_tol_deg": float(orientation_path_limit),
                        }
                    else:
                        solved_q = np.full(ARM_DOF, np.nan, dtype=np.float64)
                if not np.all(np.isfinite(solved_q)):
                    dropped_knots.append(
                        {
                            "fraction": float(fraction),
                            "reason": "ik_not_converged",
                            "pos_err_m": (
                                None
                                if best_failed is None
                                else float(best_failed[1].get("pos_err_m", float("nan")))
                            ),
                            "ori_err_deg": (
                                None
                                if best_failed is None
                                else float(best_failed[1].get("ori_err_deg", float("nan")))
                            ),
                            "attempt_count": int(len(candidate_solutions)),
                        }
                    )
                    continue
        solved_q = _locked_q(solved_q, label=f"Cartesian knot {index}")
        branch_jump = float(np.linalg.norm(solved_q[:7] - previous[:7], ord=np.inf))
        if branch_jump > DEFAULT_MAX_BRANCH_JUMP_RAD and index != knot_count:
            # A discontinuous sample is omitted and the next valid sample is
            # connected by the global joint interpolator.  This keeps the
            # branch-jump failure local; if no continuous samples remain the
            # caller still has the explicit joint-space fallback.
            dropped_knots.append(
                {
                    "fraction": float(fraction),
                    "reason": "branch_jump",
                    "branch_jump_inf_rad": float(branch_jump),
                    "threshold_rad": float(DEFAULT_MAX_BRANCH_JUMP_RAD),
                }
            )
            continue
        if branch_jump > DEFAULT_MAX_BRANCH_JUMP_RAD:
            # The signed endpoint is mandatory: dropping it would leave a
            # trajectory that ends at an intermediate pose while its metadata
            # still claims the requested endpoint.  Keep it and let the full
            # local FK/path-deviation validation decide whether the large
            # same-branch transition is geometrically continuous.  If that
            # validation fails, the caller tries another endpoint candidate or
            # the explicit joint-space fallback.
            endpoint_branch_jump_exceeded = True
        knots.append((fraction, solved_q))
        ik_reports.append(
            {
                "fraction": float(fraction),
                "branch_jump_inf_rad": branch_jump,
                "branch_jump_exceeded": bool(
                    branch_jump > DEFAULT_MAX_BRANCH_JUMP_RAD
                ),
                "pos_err_m": float(report.get("pos_err_m", 0.0)),
                "ori_err_deg": float(report.get("ori_err_deg", 0.0)),
                "solver": str(report.get("solver", "unknown")),
                "seed": (
                    "previous_knot_fast_continuation"
                    if int(_seed_index) < 0
                    else "signed_endpoint_joint_homotopy"
                ),
                "accepted_by_public_tolerance": bool(
                    report.get("accepted_by_public_tolerance", False)
                ),
            }
        )
        previous = solved_q

    waypoint_arrays, path_fractions, time_report = _global_time_parameterize_knots(
        knots,
        max_joint_step_rad=max_joint_step_rad,
        max_waypoints=max_waypoints,
        check_cancelled=check_cancelled,
    )
    waypoints: list[list[float]] = []
    max_path_position_error = 0.0
    max_path_orientation_error = 0.0
    start_position_array = np.asarray(start_position, dtype=np.float64)
    final_position_array = np.asarray(final_position, dtype=np.float64)
    eef_displacement = final_position_array - start_position_array
    eef_distance = float(np.linalg.norm(eef_displacement))
    eef_direction = (
        eef_displacement / eef_distance
        if eef_distance > 1e-12
        else np.zeros(3, dtype=np.float64)
    )
    eef_progress_high_water = 0.0
    max_eef_progress_regression = 0.0

    anchors = None
    source_tracked_points = None
    final_tracked_points = None
    tracked_directions = None
    tracked_distances = None
    tracked_progress_high_water = None
    max_tracked_progress_regressions = None
    if tracked_anchors_eef is not None:
        try:
            anchors = np.asarray(tracked_anchors_eef, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tracked EEF anchors must be a finite nested N x 3 matrix"
            ) from exc
        if (
            anchors.ndim != 2
            or anchors.shape[1] != 3
            or not 1 <= anchors.shape[0] <= 6
            or not np.all(np.isfinite(anchors))
        ):
            raise ValueError(
                "tracked EEF anchors must be a finite nested N x 3 matrix"
            )
        anchors = anchors.copy()
        source_tracked_points = start_position_array[None, :] + (
            quat_to_mat_xyzw(start_quaternion) @ anchors.T
        ).T
        final_tracked_points = final_position_array[None, :] + (
            quat_to_mat_xyzw(final_quaternion) @ anchors.T
        ).T
        tracked_displacements = final_tracked_points - source_tracked_points
        tracked_distances = np.linalg.norm(tracked_displacements, axis=1)
        tracked_directions = np.zeros_like(tracked_displacements)
        moving = tracked_distances > 1e-12
        tracked_directions[moving] = (
            tracked_displacements[moving] / tracked_distances[moving, None]
        )
        tracked_progress_high_water = np.zeros(anchors.shape[0], dtype=np.float64)
        max_tracked_progress_regressions = np.zeros(
            anchors.shape[0], dtype=np.float64
        )

    for q, fraction in zip(waypoint_arrays, path_fractions):
        check_cancelled()
        position, quaternion = eef_pose(state, arm, q)
        position_array = np.asarray(position, dtype=np.float64)
        expected_position = (
            (1.0 - fraction) * start_position_array
            + fraction * final_position_array
        )
        expected_quaternion = quaternion_slerp(
            start_quaternion, final_quaternion, fraction
        )
        max_path_position_error = max(
            max_path_position_error,
            float(np.linalg.norm(position_array - expected_position)),
        )
        max_path_orientation_error = max(
            max_path_orientation_error,
            orientation_error_deg(quaternion, expected_quaternion),
        )
        if eef_distance > 1e-12:
            progress = float((position_array - start_position_array) @ eef_direction)
            max_eef_progress_regression = max(
                max_eef_progress_regression,
                eef_progress_high_water - progress,
            )
            eef_progress_high_water = max(eef_progress_high_water, progress)
        if anchors is not None:
            tracked_points = position_array[None, :] + (
                quat_to_mat_xyzw(quaternion) @ anchors.T
            ).T
            tracked_progress = np.sum(
                (tracked_points - source_tracked_points) * tracked_directions,
                axis=1,
            )
            regressions = tracked_progress_high_water - tracked_progress
            max_tracked_progress_regressions = np.maximum(
                max_tracked_progress_regressions,
                regressions,
            )
            tracked_progress_high_water = np.maximum(
                tracked_progress_high_water,
                tracked_progress,
            )
        waypoints.append(q.astype(float).tolist())

    if not waypoints:
        waypoints = [q0.astype(float).tolist()]
    if max_path_position_error > position_path_limit:
        raise _CartesianPathDeviation(
            "joint interpolation leaves the verified Cartesian path by "
            f"{max_path_position_error:.4f}m",
            measured=max_path_position_error,
            limit=position_path_limit,
        )
    if max_path_orientation_error > orientation_path_limit:
        raise _CartesianPathDeviation(
            "joint interpolation leaves the verified orientation path by "
            f"{max_path_orientation_error:.2f}deg",
            measured=max_path_orientation_error,
            limit=orientation_path_limit,
        )
    eef_progress_regression_limit = max(
        MAX_CARTESIAN_PROGRESS_REGRESSION_M,
        2.0 * float(max_path_position_error),
    )
    eef_progress_check_applicable = bool(
        eef_distance > 2.0 * eef_progress_regression_limit
    )
    if (
        eef_progress_check_applicable
        and max_eef_progress_regression > eef_progress_regression_limit
    ):
        raise ValueError(
            "Cartesian EEF path regresses by "
            f"{max_eef_progress_regression:.4f}m instead of moving monotonically"
        )
    max_tracked_progress_regression = float(
        np.max(max_tracked_progress_regressions)
        if max_tracked_progress_regressions is not None
        else 0.0
    )
    # A point offset from the EEF follows a curved arc during quaternion SLERP.
    # Its projection on the start-to-finish chord can legitimately decrease
    # even though the SE(3) path parameter and every command advance exactly
    # once.  Keep that chord value as a diagnostic; rejecting it would discard
    # the nearest rigid transform and select a remote endpoint simply because
    # a physical rotation is not a straight point translation.

    joint_path = np.vstack([q0, np.asarray(waypoints, dtype=np.float64)])[:, :7]
    joint_deltas = np.diff(joint_path, axis=0)
    joint_direction_reversals = []
    for joint_index in range(joint_deltas.shape[1]):
        meaningful = joint_deltas[:, joint_index]
        meaningful = meaningful[np.abs(meaningful) > 1e-6]
        joint_direction_reversals.append(
            int(np.sum(meaningful[:-1] * meaningful[1:] < 0.0))
            if meaningful.size > 1
            else 0
        )
    return {
        "waypoints": waypoints,
        "cartesian_knot_count": int(len(knots) - 1),
        "joint_waypoint_count": int(len(waypoints)),
        "translation_m": translation,
        "orientation_deg": float(orientation),
        "translation_step_m": translation_step_m,
        "orientation_step_deg": orientation_step_deg,
        "max_path_position_error_m": float(max_path_position_error),
        "max_path_orientation_error_deg": float(max_path_orientation_error),
        "max_joint_step_rad": float(max_joint_step_rad),
        "interpolation": "robot_base_position_linear_and_quaternion_slerp",
        "time_parameterization": "single_global_trapezoidal_joint_path_speed",
        **time_report,
        "monotonic_progress_checked": True,
        "monotonic_progress_definition": (
            "strictly_increasing_global_path_parameter_and_linear_eef_translation"
        ),
        "max_eef_progress_regression_m": float(max_eef_progress_regression),
        "eef_progress_regression_limit_m": float(
            eef_progress_regression_limit
        ),
        "eef_chord_progress_check_applicable": eef_progress_check_applicable,
        "max_tracked_point_progress_regression_m": (
            max_tracked_progress_regression
        ),
        "tracked_point_chord_regression_is_diagnostic": True,
        "tracked_point_progress_regression_m": (
            []
            if max_tracked_progress_regressions is None
            else max_tracked_progress_regressions.astype(float).tolist()
        ),
        "joint_direction_reversal_count": joint_direction_reversals,
        "ik_reports": ik_reports,
        "fast_continuation_attempt_count": int(fast_continuation_attempt_count),
        "fast_continuation_success_count": int(fast_continuation_success_count),
        "legacy_ik_call_count": int(legacy_ik_call_count),
        "dropped_knots": dropped_knots,
        "dropped_knot_count": int(len(dropped_knots)),
        "endpoint_branch_jump_exceeded": bool(endpoint_branch_jump_exceeded),
        "same_branch_checked": True,
        "joint_limits_checked": True,
        "local_fk_finite_checked": True,
        "j8_locked_zero": ARM_DOF == 8,
        "simulator_collision_checked": False,
        "frozen_rgbd_collision_checked": False,
    }


def plan_whole_body_joint_trajectory(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    q_final: Sequence[float],
    trunk_q_start: Sequence[float],
    trunk_q_final: Sequence[float],
    max_joint_step_rad: float,
    max_waypoints: int,
    tracked_anchors_eef: Sequence[Sequence[float]] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Plan one locally verified trunk-plus-arm joint interpolation.

    The endpoint solver may use the three movable trunk joints to reach a
    precise tracked-point transform.  This routine preserves that exact
    whole-body endpoint while producing a single synchronized path parameter;
    no pose is re-solved during execution.
    """

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")

    check_cancelled()
    if arm not in ("left", "right"):
        raise ValueError("whole-body joint path arm must be left or right")
    if not math.isfinite(float(max_joint_step_rad)) or float(max_joint_step_rad) <= 0.0:
        raise ValueError("whole-body joint path max_joint_step_rad must be positive")
    if (
        isinstance(max_waypoints, bool)
        or int(max_waypoints) != max_waypoints
        or int(max_waypoints) < 1
    ):
        raise ValueError("whole-body joint path max_waypoints must be a positive integer")

    q0 = _locked_q(q_start, label="whole-body joint path start arm q")
    q1 = _locked_q(q_final, label="whole-body joint path final arm q")
    trunk0 = np.asarray(trunk_q_start, dtype=np.float64).reshape(4).copy()
    trunk1 = np.asarray(trunk_q_final, dtype=np.float64).reshape(4).copy()
    if not np.all(np.isfinite(trunk0)) or not np.all(np.isfinite(trunk1)):
        raise ValueError("whole-body joint path trunk q contains non-finite values")
    if abs(float(trunk0[3])) > 1.0e-6 or abs(float(trunk1[3])) > 1.0e-6:
        raise ValueError("whole-body joint path trunk J4 must remain zero")
    trunk0[3] = 0.0
    trunk1[3] = 0.0

    arm_lower, arm_upper = arm_joint_limits(state, arm)
    arm_lower = np.asarray(arm_lower, dtype=np.float64).reshape(-1)
    arm_upper = np.asarray(arm_upper, dtype=np.float64).reshape(-1)
    if arm_lower.size < ARM_DOF or arm_upper.size < ARM_DOF:
        raise ValueError("whole-body joint path local arm limits are invalid")
    trunk_limits = np.asarray(TRUNK_JOINT_LIMITS, dtype=np.float64).reshape(4, 2)
    if np.any(q0 < arm_lower[:ARM_DOF]) or np.any(q0 > arm_upper[:ARM_DOF]):
        raise ValueError("whole-body joint path start arm q exceeds local limits")
    if np.any(q1 < arm_lower[:ARM_DOF]) or np.any(q1 > arm_upper[:ARM_DOF]):
        raise ValueError("whole-body joint path final arm q exceeds local limits")
    if np.any(trunk0 < trunk_limits[:, 0] - 1.0e-6) or np.any(
        trunk0 > trunk_limits[:, 1] + 1.0e-6
    ):
        raise ValueError("whole-body joint path start trunk q exceeds local limits")
    if np.any(trunk1 < trunk_limits[:, 0] - 1.0e-6) or np.any(
        trunk1 > trunk_limits[:, 1] + 1.0e-6
    ):
        raise ValueError("whole-body joint path final trunk q exceeds local limits")
    trunk0 = np.clip(trunk0, trunk_limits[:, 0], trunk_limits[:, 1])
    trunk1 = np.clip(trunk1, trunk_limits[:, 0], trunk_limits[:, 1])
    trunk0[3] = 0.0
    trunk1[3] = 0.0

    arm_delta = q1 - q0
    trunk_delta = trunk1 - trunk0
    maximum_delta = max(
        float(np.linalg.norm(arm_delta, ord=np.inf)),
        float(np.linalg.norm(trunk_delta, ord=np.inf)),
    )
    segment_count = max(
        1,
        int(math.ceil(maximum_delta / float(max_joint_step_rad))),
    )
    if segment_count > int(max_waypoints):
        raise ValueError(
            "whole-body joint path requires "
            f"{segment_count} waypoints, exceeding max_waypoints={int(max_waypoints)}"
        )

    anchors = None
    if tracked_anchors_eef is not None:
        anchors = _strict_points(
            tracked_anchors_eef,
            label="whole-body joint path tracked EEF anchors",
            max_points=MAX_TRACKED_POINTS,
        )

    def state_for(trunk: np.ndarray, q: np.ndarray) -> LocalRobotState:
        if arm == "left":
            return replace(state, trunk_q=trunk, arm_left_q=q)
        return replace(state, trunk_q=trunk, arm_right_q=q)

    start_state = state_for(trunk0, q0)
    final_state = state_for(trunk1, q1)
    start_position, start_quaternion = eef_pose(start_state, arm, q0)
    final_position, final_quaternion = eef_pose(final_state, arm, q1)
    if not all(
        np.all(np.isfinite(value))
        for value in (
            start_position,
            start_quaternion,
            final_position,
            final_quaternion,
        )
    ):
        raise ValueError("whole-body joint path endpoint FK is non-finite")

    waypoints: list[list[float]] = []
    trunk_waypoints: list[list[float]] = []
    previous_q = q0.copy()
    previous_trunk = trunk0.copy()
    maximum_step = 0.0
    max_fk_anchor_error = 0.0
    combined_path = [np.concatenate((trunk0[:3], q0[:7]))]
    for step_index in range(1, segment_count + 1):
        check_cancelled()
        fraction = float(step_index) / float(segment_count)
        q = _locked_q(
            q0 + fraction * arm_delta,
            label=f"whole-body joint path arm waypoint {step_index}",
        )
        trunk = trunk0 + fraction * trunk_delta
        trunk[3] = 0.0
        if np.any(q < arm_lower[:ARM_DOF]) or np.any(q > arm_upper[:ARM_DOF]):
            raise ValueError("whole-body joint path arm waypoint exceeds local limits")
        if np.any(trunk < trunk_limits[:, 0]) or np.any(trunk > trunk_limits[:, 1]):
            raise ValueError("whole-body joint path trunk waypoint exceeds local limits")
        waypoint_state = state_for(trunk, q)
        position, quaternion = eef_pose(waypoint_state, arm, q)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("whole-body joint path waypoint FK is non-finite")
        if anchors is not None:
            point_positions = np.asarray(position)[None, :] + (
                quat_to_mat_xyzw(quaternion) @ anchors.T
            ).T
            if not np.all(np.isfinite(point_positions)):
                raise ValueError("whole-body joint path tracked projection is non-finite")
            expected = (1.0 - fraction) * (
                np.asarray(start_position)[None, :]
                + (quat_to_mat_xyzw(start_quaternion) @ anchors.T).T
            ) + fraction * (
                np.asarray(final_position)[None, :]
                + (quat_to_mat_xyzw(final_quaternion) @ anchors.T).T
            )
            max_fk_anchor_error = max(
                max_fk_anchor_error,
                float(np.max(np.linalg.norm(point_positions - expected, axis=1))),
            )
        maximum_step = max(
            maximum_step,
            float(np.linalg.norm(q - previous_q, ord=np.inf)),
            float(np.linalg.norm(trunk - previous_trunk, ord=np.inf)),
        )
        waypoints.append(q.astype(float).tolist())
        trunk_waypoints.append(trunk.astype(float).tolist())
        combined_path.append(np.concatenate((trunk[:3], q[:7])))
        previous_q = q
        previous_trunk = trunk

    combined_matrix = np.asarray(combined_path, dtype=np.float64)
    deltas = np.diff(combined_matrix, axis=0)
    reversals: list[int] = []
    for joint_index in range(deltas.shape[1]):
        values = deltas[:, joint_index]
        values = values[np.abs(values) > 1.0e-8]
        reversals.append(
            int(np.sum(values[:-1] * values[1:] < 0.0))
            if values.size > 1
            else 0
        )
    return {
        "waypoints": waypoints,
        "trunk_waypoints": trunk_waypoints,
        "trunk_q_start": trunk0.astype(float).tolist(),
        "trunk_q_final": trunk1.astype(float).tolist(),
        "trunk_assisted": True,
        "cartesian_knot_count": 0,
        "joint_waypoint_count": int(len(waypoints)),
        "translation_m": float(
            np.linalg.norm(np.asarray(final_position) - np.asarray(start_position))
        ),
        "orientation_deg": float(
            orientation_error_deg(start_quaternion, final_quaternion)
        ),
        "translation_step_m": None,
        "orientation_step_deg": None,
        "max_path_position_error_m": None,
        "max_path_orientation_error_deg": None,
        "max_joint_step_rad": float(max_joint_step_rad),
        "actual_max_joint_step_rad": float(maximum_step),
        "joint_path_length_inf_rad": float(
            np.sum(np.max(np.abs(deltas), axis=1))
        ),
        "global_time_ramp_fraction": None,
        "minimum_path_progress_increment_rad": float(
            np.min(np.max(np.abs(deltas), axis=1))
        ),
        "internal_stop_count": 0,
        "interpolation": "synchronized_trunk_and_arm_joint_space_linear",
        "time_parameterization": "bounded_whole_body_joint_linear_fallback",
        "path_mode": "whole_body_joint_space_fallback",
        "fallback_reason": "precise_endpoint_requires_trunk_assist",
        "cartesian_path_verified": False,
        "monotonic_progress_checked": True,
        "monotonic_progress_definition": (
            "strictly_increasing_shared_trunk_arm_path_parameter"
        ),
        "joint_direction_reversal_count": reversals,
        "same_branch_checked": True,
        "joint_limits_checked": True,
        "local_fk_finite_checked": True,
        "local_fk_cartesian_deviation_m": float(max_fk_anchor_error),
        "j8_locked_zero": ARM_DOF == 8,
        "trunk_j4_locked_zero": True,
        "simulator_collision_checked": False,
        "frozen_rgbd_collision_checked": False,
        "ik_reports": [],
        "dropped_knots": [],
        "dropped_knot_count": 0,
    }


def plan_joint_space_trajectory(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    q_final: Sequence[float],
    max_joint_step_rad: float,
    max_waypoints: int,
    tracked_anchors_eef: Sequence[Sequence[float]] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Plan a bounded local joint interpolation as a Cartesian fallback.

    This is deliberately a *fallback*, not an alternate endpoint solver.  It
    is called only after the endpoint has passed the complete tracked-point
    constraint check and the straight position/SLERP continuation has failed.
    Every interpolated state is checked with submission-local FK, joint
    limits, finite values, and the J8 lock.  No simulator collision or IK API
    is consulted.
    """

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")

    check_cancelled()
    if not math.isfinite(float(max_joint_step_rad)) or float(max_joint_step_rad) <= 0.0:
        raise ValueError("joint fallback max_joint_step_rad must be positive")
    if isinstance(max_waypoints, bool) or int(max_waypoints) != max_waypoints or int(max_waypoints) < 1:
        raise ValueError("joint fallback max_waypoints must be a positive integer")
    q0 = _locked_q(q_start, label="joint fallback start q")
    q1 = _locked_q(q_final, label="joint fallback final q")
    lower_limits, upper_limits = arm_joint_limits(state, arm)
    lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if lower_limits.shape[0] < ARM_DOF or upper_limits.shape[0] < ARM_DOF:
        raise ValueError("joint fallback local joint limits are invalid")
    if np.any(q0 < lower_limits[:ARM_DOF]) or np.any(q0 > upper_limits[:ARM_DOF]):
        raise ValueError("joint fallback start q exceeds local limits")
    if np.any(q1 < lower_limits[:ARM_DOF]) or np.any(q1 > upper_limits[:ARM_DOF]):
        raise ValueError("joint fallback final q exceeds local limits")
    delta = q1 - q0
    segment_count = max(
        1,
        int(math.ceil(float(np.linalg.norm(delta, ord=np.inf)) / float(max_joint_step_rad))),
    )
    if segment_count > int(max_waypoints):
        raise ValueError(
            "joint fallback requires "
            f"{segment_count} waypoints, exceeding max_waypoints={int(max_waypoints)}"
        )

    waypoints: list[list[float]] = []
    maximum_step = 0.0
    previous = q0.copy()
    anchors = None
    if tracked_anchors_eef is not None:
        anchors = _strict_points(
            tracked_anchors_eef,
            label="joint fallback tracked EEF anchors",
            max_points=MAX_TRACKED_POINTS,
        )
    max_fk_anchor_error = 0.0
    start_position, start_quaternion = eef_pose(state, arm, q0)
    final_position, final_quaternion = eef_pose(state, arm, q1)
    if not np.all(np.isfinite(start_position)) or not np.all(np.isfinite(final_position)):
        raise ValueError("joint fallback local FK returned non-finite EEF position")
    for step_index in range(1, segment_count + 1):
        check_cancelled()
        fraction = float(step_index) / float(segment_count)
        q = _locked_q(
            q0 + fraction * delta,
            label=f"joint fallback waypoint {step_index}",
        )
        if np.any(q < lower_limits[:ARM_DOF]) or np.any(q > upper_limits[:ARM_DOF]):
            raise ValueError("joint fallback waypoint exceeds local limits")
        position, quaternion = eef_pose(state, arm, q)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("joint fallback local FK returned a non-finite waypoint")
        if anchors is not None:
            point_positions = np.asarray(position)[None, :] + (
                quat_to_mat_xyzw(quaternion) @ anchors.T
            ).T
            if not np.all(np.isfinite(point_positions)):
                raise ValueError("joint fallback tracked point projection is non-finite")
            # This is an internal diagnostic only.  The signed endpoint and
            # live tracked-point monitor remain the authoritative constraints.
            expected = (1.0 - fraction) * (
                np.asarray(start_position)[None, :] + (
                    quat_to_mat_xyzw(start_quaternion) @ anchors.T
                ).T
            ) + fraction * (
                np.asarray(final_position)[None, :] + (
                    quat_to_mat_xyzw(final_quaternion) @ anchors.T
                ).T
            )
            max_fk_anchor_error = max(
                max_fk_anchor_error,
                float(np.max(np.linalg.norm(point_positions - expected, axis=1))),
            )
        maximum_step = max(
            maximum_step,
            float(np.linalg.norm(q - previous, ord=np.inf)),
        )
        waypoints.append(q.astype(float).tolist())
        previous = q
    if not waypoints:
        raise ValueError("joint fallback produced no waypoints")
    joint_path = np.vstack((q0, np.asarray(waypoints, dtype=np.float64)))[:, :7]
    deltas = np.diff(joint_path, axis=0)
    reversals: list[int] = []
    for joint_index in range(deltas.shape[1]):
        values = deltas[:, joint_index]
        values = values[np.abs(values) > 1.0e-8]
        reversals.append(
            int(np.sum(values[:-1] * values[1:] < 0.0))
            if values.size > 1
            else 0
        )
    return {
        "waypoints": waypoints,
        "cartesian_knot_count": 0,
        "joint_waypoint_count": int(len(waypoints)),
        "translation_m": float(np.linalg.norm(np.asarray(final_position) - np.asarray(start_position))),
        "orientation_deg": float(orientation_error_deg(start_quaternion, final_quaternion)),
        "translation_step_m": None,
        "orientation_step_deg": None,
        "max_path_position_error_m": None,
        "max_path_orientation_error_deg": None,
        "max_joint_step_rad": float(max_joint_step_rad),
        "actual_max_joint_step_rad": float(maximum_step),
        "joint_path_length_inf_rad": float(np.sum(np.max(np.abs(deltas), axis=1))),
        "global_time_ramp_fraction": None,
        "minimum_path_progress_increment_rad": float(
            np.min(np.max(np.abs(deltas), axis=1))
        ),
        "internal_stop_count": 0,
        "interpolation": "local_joint_space_linear_fallback",
        "time_parameterization": "bounded_joint_linear_fallback",
        "path_mode": "joint_space_fallback",
        "fallback_reason": "straight_cartesian_path_unavailable",
        "cartesian_path_verified": False,
        "monotonic_progress_checked": True,
        "monotonic_progress_definition": "strictly_increasing_joint_path_parameter",
        "joint_direction_reversal_count": reversals,
        "same_branch_checked": True,
        "joint_limits_checked": True,
        "local_fk_finite_checked": True,
        "local_fk_cartesian_deviation_m": float(max_fk_anchor_error),
        "j8_locked_zero": ARM_DOF == 8,
        "simulator_collision_checked": False,
        "frozen_rgbd_collision_checked": False,
        "ik_reports": [],
        "dropped_knots": [],
        "dropped_knot_count": 0,
    }


def plan_cartesian_trajectory(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    q_final: Sequence[float],
    pos_tol_m: float,
    ori_tol_deg: float,
    max_joint_step_rad: float,
    max_waypoints: int,
    tracked_anchors_eef: Sequence[Sequence[float]] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    planning_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Plan a verified Cartesian path under the caller's waypoint budget.

    Every attempt represents the same linear-position/SLERP path.  A fixed
    coarse-to-fine density sequence is used for every request, independent of
    point names, coordinate expressions, and target values.
    """

    planning_deadline_monotonic = _normalize_planning_deadline(
        planning_deadline_monotonic
    )

    def check_cancelled() -> None:
        if callable(cancel_requested) and bool(cancel_requested()):
            raise RuntimeError("frozen tracked-point planning was cancelled")
        _check_planning_deadline(planning_deadline_monotonic)

    check_cancelled()
    if int(max_waypoints) < 1:
        raise ValueError("max_waypoints must be positive")

    q0 = _locked_q(q_start, label="Cartesian path start q")
    q_goal = _locked_q(q_final, label="Cartesian path final q")
    direct_waypoints = int(
        math.ceil(
            float(np.linalg.norm(q_goal - q0, ord=np.inf))
            / max(float(max_joint_step_rad), 1e-6)
        )
    )
    if direct_waypoints > int(max_waypoints):
        raise ValueError(
            "Cartesian endpoint joint displacement alone requires at least "
            f"{direct_waypoints} waypoints, exceeding max_waypoints="
            f"{int(max_waypoints)}"
        )

    failures: list[str] = []
    densities = [
        (float(translation_step), float(orientation_step))
        for translation_step, orientation_step in CARTESIAN_SAMPLING_DENSITIES
    ]
    for attempt_index, (translation_step, orientation_step) in enumerate(densities):
        check_cancelled()
        try:
            path = _plan_cartesian_trajectory_once(
                state=state,
                arm=arm,
                q_start=q_start,
                q_final=q_final,
                pos_tol_m=pos_tol_m,
                ori_tol_deg=ori_tol_deg,
                max_joint_step_rad=max_joint_step_rad,
                max_waypoints=max_waypoints,
                translation_step_m=translation_step,
                orientation_step_deg=orientation_step,
                tracked_anchors_eef=tracked_anchors_eef,
                cancel_requested=cancel_requested,
                planning_deadline_monotonic=planning_deadline_monotonic,
            )
            path["sampling_attempt"] = int(attempt_index)
            path["sampling_steps_tried_deg"] = [
                float(value[1])
                for value in densities[: attempt_index + 1]
            ]
            path["sampling_translation_steps_tried_m"] = [
                float(value[0])
                for value in densities[: attempt_index + 1]
            ]
            return path
        except PlanningDeadlineExceeded:
            # A live caller can still use an already validated endpoint via
            # its joint-space fallback.  Do not spend the remaining budget on
            # finer sampling attempts.
            raise
        except _CartesianPathDeviation as exc:
            check_cancelled()
            failures.append(
                f"{translation_step:g}m/{orientation_step:g}deg: {exc}"
            )
            if exc.ratio >= CARTESIAN_REFINEMENT_MAX_DEVIATION_RATIO:
                break
        except (RuntimeError, TypeError, ValueError) as exc:
            check_cancelled()
            failures.append(
                f"{translation_step:g}m/{orientation_step:g}deg: {exc}"
            )

    # Keep the original detailed failure visible to the caller while making it
    # clear that every representation-independent sampling attempt was tried.
    detail = "; ".join(failures)
    raise ValueError(
        "no verified Cartesian trajectory fits the evaluator waypoint budget "
        f"after uniform sampling attempts ({detail})"
    )


__all__ = [
    "PlanningDeadlineExceeded",
    "evaluate_rigid_pair_consistency",
    "evaluate_rigid_geometry_consistency",
    "evaluate_target_constraints",
    "plan_joint_space_trajectory",
    "plan_whole_body_joint_trajectory",
    "plan_cartesian_trajectory",
    "plan_endpoint",
    "plan_endpoint_with_trunk_assist",
    "quaternion_slerp",
    "rigid_anchors_from_points",
    "rigid_pair_distance_tolerance",
    "target_constraint_residuals",
]
