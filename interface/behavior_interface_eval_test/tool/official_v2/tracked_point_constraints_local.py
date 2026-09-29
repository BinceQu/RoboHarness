"""Pure submission-local geometry constraints for tracked-point motion.

The module intentionally has no evaluator/world imports.  It operates on
frozen robot-base point arrays and declarative relation records only.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    _validate_move_tracked_relations,
    normalize_tracked_target_coordinate,
    parse_tracked_affine_expression,
)


MAX_TRACKED_POINTS = 6
GEOMETRY_RANK_TOLERANCE_M = 1.0e-7
GEOMETRY_DEGENERATE_AREA_M2 = 1.0e-10
GEOMETRY_DEGENERATE_VOLUME_M3 = 1.0e-12
# Collinearity is used to place a held axis against a scene axis.  Its visual
# endpoint error must stay substantially tighter than the caller's general XYZ
# tolerance, which may legitimately be centimetre-scale for other constraints.
COLLINEAR_POSITION_TOLERANCE_M = 0.001
COLLINEAR_RELATION_TYPES = frozenset(
    {
        "line_through_point",
        "line_coincident",
        "line_segment_overlap",
        "line_segment_contains",
        "ordered_collinear",
    }
)
# Ordered collinearity is categorical: the fixed segment must remain between
# the two controlled markers, not merely within the caller's broad positional
# tolerance.  Planning six millimetres inside either endpoint leaves three
# millimetres of physical containment even after the endpoint enters the
# dedicated 1 mm collinearity frontier.
ORDERED_COLLINEAR_INNER_MARGIN_M = 0.006
ORDERED_COLLINEAR_LIVE_BOUNDARY_TOLERANCE_M = 0.001


def relation_position_tolerance_m(
    relation_type: Any,
    requested_tolerance_m: float,
) -> float:
    """Return the effective position tolerance for one typed relation."""

    try:
        requested = float(requested_tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("relation position tolerance must be numeric") from exc
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("relation position tolerance must be finite and positive")
    normalized_type = str(relation_type or "").strip().lower()
    if normalized_type in COLLINEAR_RELATION_TYPES:
        return float(min(requested, COLLINEAR_POSITION_TOLERANCE_M))
    return requested


def _points(raw: Sequence[Sequence[float]], *, label: str) -> np.ndarray:
    try:
        value = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an N x 3 array") from exc
    # Keep the public geometry contract structural.  In particular, accepting
    # a flattened 3*N buffer here would make a tampered trajectory look like a
    # valid point bundle and would lose the marker boundaries used by all
    # pair/triangle/tetrahedron checks.
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{label} must be an N x 3 array")
    value = value.copy()
    if not 1 <= value.shape[0] <= MAX_TRACKED_POINTS:
        raise ValueError(
            f"{label} must contain one to {MAX_TRACKED_POINTS} points"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain finite coordinates")
    return value


def geometry_rank(
    points_robot_base_m: Sequence[Sequence[float]],
    *,
    tolerance_m: float = GEOMETRY_RANK_TOLERANCE_M,
) -> int:
    """Return the centered affine rank (0, 1, 2, or 3) of a point set."""

    points = _points(points_robot_base_m, label="points")
    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry rank tolerance must be numeric") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("geometry rank tolerance must be finite and nonnegative")
    if points.shape[0] <= 1:
        return 0
    centered = points - np.mean(points, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    # ``tolerance_m`` is already expressed in the same robot-base units as
    # the points.  Multiplying it by the object span makes a small marker set
    # appear artificially higher-rank and makes the report depend on an
    # unrelated scale factor.
    threshold = max(tolerance, 1.0e-12)
    return int(np.sum(singular > threshold))


def pair_distance_matrix(points_robot_base_m: Sequence[Sequence[float]]) -> np.ndarray:
    points = _points(points_robot_base_m, label="points")
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def triangle_area_m2(
    points_robot_base_m: Sequence[Sequence[float]],
    indices: Sequence[int] = (0, 1, 2),
) -> float:
    points = _points(points_robot_base_m, label="points")
    if len(indices) != 3 or any(int(i) < 0 or int(i) >= len(points) for i in indices):
        raise ValueError("triangle indices are invalid")
    a, b, c = (points[int(i)] for i in indices)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def tetrahedron_signed_volume_m3(
    points_robot_base_m: Sequence[Sequence[float]],
    indices: Sequence[int] = (0, 1, 2, 3),
) -> float:
    points = _points(points_robot_base_m, label="points")
    if len(indices) != 4 or any(int(i) < 0 or int(i) >= len(points) for i in indices):
        raise ValueError("tetrahedron indices are invalid")
    a, b, c, d = (points[int(i)] for i in indices)
    return float(np.dot(b - a, np.cross(c - a, d - a)) / 6.0)


def _longest_pair(points: np.ndarray) -> tuple[int, int] | None:
    if len(points) < 2:
        return None
    pairs = list(itertools.combinations(range(len(points)), 2))
    return max(pairs, key=lambda pair: float(np.linalg.norm(points[pair[1]] - points[pair[0]])))


def _largest_triangle(points: np.ndarray) -> tuple[int, int, int] | None:
    if len(points) < 3:
        return None
    triangles = list(itertools.combinations(range(len(points)), 3))
    return max(triangles, key=lambda tri: triangle_area_m2(points, tri))


def rigid_geometry_report(
    source_points_robot_base_m: Sequence[Sequence[float]],
    target_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    *,
    tolerance_m: float = 0.012,
) -> dict[str, Any]:
    """Describe N-point rigid invariants without assuming a point count."""

    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry report tolerance must be numeric") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("geometry report tolerance must be finite and nonnegative")
    source = _points(source_points_robot_base_m, label="source points")
    target = None
    if target_points_robot_base_m is not None:
        target = _points(target_points_robot_base_m, label="target points")
        if target.shape != source.shape:
            raise ValueError("source and target point arrays must have the same shape")
    source_pairs = pair_distance_matrix(source)
    target_pairs = None if target is None else pair_distance_matrix(target)
    pair_records: list[dict[str, Any]] = []
    max_pair_error = 0.0
    for i, j in itertools.combinations(range(len(source)), 2):
        source_distance = float(source_pairs[i, j])
        target_distance = None if target_pairs is None else float(target_pairs[i, j])
        error = 0.0 if target_distance is None else abs(target_distance - source_distance)
        max_pair_error = max(max_pair_error, error)
        pair_records.append(
            {
                "indices": [int(i), int(j)],
                "source_distance_m": source_distance,
                "target_distance_m": target_distance,
                "distance_error_m": float(error),
            }
        )

    triangle_records: list[dict[str, Any]] = []
    max_triangle_error = 0.0
    if len(source) >= 3:
        for tri in itertools.combinations(range(len(source)), 3):
            source_area = triangle_area_m2(source, tri)
            target_area = None if target is None else triangle_area_m2(target, tri)
            error = 0.0 if target_area is None else abs(target_area - source_area)
            max_triangle_error = max(max_triangle_error, error)
            triangle_records.append(
                {
                    "indices": list(map(int, tri)),
                    "source_area_m2": float(source_area),
                    "target_area_m2": None if target_area is None else float(target_area),
                    "area_error_m2": float(error),
                }
            )

    volume_records: list[dict[str, Any]] = []
    max_volume_error = 0.0
    if len(source) >= 4:
        for tetra in itertools.combinations(range(len(source)), 4):
            source_volume = tetrahedron_signed_volume_m3(source, tetra)
            target_volume = None if target is None else tetrahedron_signed_volume_m3(target, tetra)
            error = 0.0 if target_volume is None else abs(abs(target_volume) - abs(source_volume))
            max_volume_error = max(max_volume_error, error)
            volume_records.append(
                {
                    "indices": list(map(int, tetra)),
                    "source_signed_volume_m3": float(source_volume),
                    "target_signed_volume_m3": None if target_volume is None else float(target_volume),
                    "absolute_volume_error_m3": float(error),
                    "orientation_preserved": (
                        None
                        if target_volume is None
                        else bool(source_volume == 0.0 or target_volume == 0.0 or source_volume * target_volume > 0.0)
                    ),
                }
            )
    kabsch_residuals: list[float] | None = None
    max_kabsch_residual = 0.0
    if target is not None:
        # Kabsch is a bundle-level check: pair distances can each be within a
        # loose depth tolerance while no single proper rigid transform fits all
        # markers.  Keep this diagnostic in the signed report so loaders and
        # live monitoring can recompute exactly the same quantity.
        kabsch = kabsch_rigid_transform(source, target)
        kabsch_residuals = np.asarray(
            kabsch["per_point_error_m"], dtype=np.float64
        ).astype(float).tolist()
        max_kabsch_residual = float(kabsch["max_error_m"])
    longest_pair = _longest_pair(source)
    largest_triangle = _largest_triangle(source)
    return {
        "point_count": int(len(source)),
        "source_rank": geometry_rank(source),
        "target_rank": None if target is None else geometry_rank(target),
        "pair_distances": pair_records,
        "triangle_areas": triangle_records,
        "tetrahedron_volumes": volume_records,
        "max_pair_distance_error_m": float(max_pair_error),
        "max_triangle_area_error_m2": float(max_triangle_error),
        "max_tetrahedron_volume_error_m3": float(max_volume_error),
        "kabsch_bundle_residual_m": kabsch_residuals,
        "max_kabsch_bundle_residual_m": float(max_kabsch_residual),
        "reference_pair": None if longest_pair is None else list(map(int, longest_pair)),
        "reference_triangle": None if largest_triangle is None else list(map(int, largest_triangle)),
        "tolerance_m": tolerance,
    }


def geometry_report_difference(
    reported: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    atol: float = 1.0e-9,
) -> str | None:
    """Return a precise mismatch reason for two invariant reports.

    Trajectory records are signed, but their diagnostic report is still
    validated from the signed point arrays.  This helper intentionally
    compares only geometry invariants (the tolerance used to *format* a
    report is not an invariant and may differ between planner and loader).
    """

    try:
        absolute_tolerance = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry report comparison tolerance is invalid") from exc
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("geometry report comparison tolerance is invalid")
    if not isinstance(reported, Mapping) or not isinstance(expected, Mapping):
        return "geometry report is not an object"

    def numeric_equal(left: Any, right: Any) -> bool:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and abs(left_value - right_value) <= absolute_tolerance
        )

    for key in ("point_count", "source_rank", "target_rank", "reference_pair", "reference_triangle"):
        if reported.get(key) != expected.get(key):
            return f"geometry report {key} is inconsistent"

    scalar_fields = (
        "max_pair_distance_error_m",
        "max_triangle_area_error_m2",
        "max_tetrahedron_volume_error_m3",
        "max_kabsch_bundle_residual_m",
    )
    for key in scalar_fields:
        if not numeric_equal(reported.get(key), expected.get(key)):
            return f"geometry report {key} is inconsistent"

    def compare_records(
        key: str,
        numeric_fields: Sequence[str],
        boolean_fields: Sequence[str] = (),
    ) -> str | None:
        actual_records = reported.get(key)
        expected_records = expected.get(key)
        if not isinstance(actual_records, list) or not isinstance(expected_records, list):
            return f"geometry report {key} is invalid"
        if len(actual_records) != len(expected_records):
            return f"geometry report {key} count is inconsistent"
        for index, (actual, expected_record) in enumerate(
            zip(actual_records, expected_records)
        ):
            if not isinstance(actual, Mapping) or not isinstance(expected_record, Mapping):
                return f"geometry report {key}[{index}] is invalid"
            if actual.get("indices") != expected_record.get("indices"):
                return f"geometry report {key}[{index}] indices are inconsistent"
            for field in numeric_fields:
                if actual.get(field) is None or expected_record.get(field) is None:
                    if actual.get(field) is not None or expected_record.get(field) is not None:
                        return f"geometry report {key}[{index}].{field} is inconsistent"
                elif not numeric_equal(actual.get(field), expected_record.get(field)):
                    return f"geometry report {key}[{index}].{field} is inconsistent"
            for field in boolean_fields:
                if actual.get(field) != expected_record.get(field):
                    return f"geometry report {key}[{index}].{field} is inconsistent"
        return None

    for key, fields, booleans in (
        (
            "pair_distances",
            ("source_distance_m", "target_distance_m", "distance_error_m"),
            (),
        ),
        (
            "triangle_areas",
            ("source_area_m2", "target_area_m2", "area_error_m2"),
            (),
        ),
        (
            "tetrahedron_volumes",
            (
                "source_signed_volume_m3",
                "target_signed_volume_m3",
                "absolute_volume_error_m3",
            ),
            ("orientation_preserved",),
        ),
    ):
        mismatch = compare_records(key, fields, booleans)
        if mismatch:
            return mismatch
    reported_kabsch = reported.get("kabsch_bundle_residual_m")
    expected_kabsch = expected.get("kabsch_bundle_residual_m")
    if reported_kabsch is None or expected_kabsch is None:
        if reported_kabsch is not expected_kabsch:
            return "geometry report kabsch_bundle_residual_m is inconsistent"
    elif (
        not isinstance(reported_kabsch, list)
        or not isinstance(expected_kabsch, list)
        or len(reported_kabsch) != len(expected_kabsch)
        or any(
            not numeric_equal(left, right)
            for left, right in zip(reported_kabsch, expected_kabsch)
        )
    ):
        return "geometry report kabsch_bundle_residual_m is inconsistent"
    return None


def resolve_fully_numeric_target_points(
    target_points: Sequence[Mapping[str, Any]],
) -> np.ndarray | None:
    """Return target coordinates only when every coordinate is a literal."""

    result: list[list[float]] = []
    for point in target_points:
        target = point.get("target_xyz_m")
        if isinstance(target, Mapping):
            values = [target.get(axis) for axis in ("x", "y", "z")]
        elif isinstance(target, (list, tuple)) and len(target) == 3:
            values = list(target)
        else:
            return None
        row: list[float] = []
        for value in values:
            if isinstance(value, bool) or isinstance(value, Mapping):
                return None
            if isinstance(value, str):
                try:
                    number = float(value.strip())
                except (TypeError, ValueError):
                    return None
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
            if not math.isfinite(number):
                return None
            row.append(number)
        result.append(row)
    return _points(result, label="numeric target points")


def kabsch_rigid_transform(
    source_points_robot_base_m: Sequence[Sequence[float]],
    target_points_robot_base_m: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compute the nearest proper rigid transform mapping source to target."""

    source = _points(source_points_robot_base_m, label="source points")
    target = _points(target_points_robot_base_m, label="target points")
    if source.shape != target.shape:
        raise ValueError("source and target point arrays must have the same shape")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    source_rank = geometry_rank(source)
    target_rank = geometry_rank(target)
    if source.shape[0] == 1 or np.linalg.norm(centered_source) <= 1.0e-12:
        rotation = np.eye(3, dtype=np.float64)
        reflected = False
        raw_determinant = 1.0
    else:
        covariance = centered_source.T @ centered_target
        u, _singular, vt = np.linalg.svd(covariance)
        raw_rotation = vt.T @ u.T
        raw_determinant = float(np.linalg.det(raw_rotation))
        corrected_vt = vt.copy()
        if np.linalg.det(raw_rotation) < 0.0:
            corrected_vt[-1, :] *= -1.0
        rotation = corrected_vt.T @ u.T
        # For rank-deficient (one-dimensional or planar) markers the null
        # basis has an arbitrary sign, so the raw SVD determinant is not
        # evidence of an improper physical transform.  Handedness is
        # observable only when both sets span 3-D; four non-coplanar points
        # are preflighted separately before this seed is used.
        reflected = bool(
            raw_determinant < 0.0 and source_rank >= 3 and target_rank >= 3
        )
    translation = target_center - rotation @ source_center
    predicted = (rotation @ source.T).T + translation
    errors = np.linalg.norm(predicted - target, axis=1)
    return {
        "rotation_matrix": rotation,
        "translation_robot_base_m": translation,
        "predicted_points_robot_base_m": predicted,
        "per_point_error_m": errors,
        "max_error_m": float(np.max(errors)) if errors.size else 0.0,
        "proper_rotation": bool(np.linalg.det(rotation) > 0.0),
        "raw_rotation_determinant": raw_determinant,
        "reflection_detected": bool(reflected),
        "source_rank": source_rank,
        "target_rank": target_rank,
    }


def geometry_preflight(
    source_points_robot_base_m: Sequence[Sequence[float]],
    target_points_robot_base_m: Sequence[Sequence[float]] | None,
    *,
    tolerance_m: float = 0.012,
) -> dict[str, Any]:
    """Conservatively reject only provably non-rigid fixed targets."""

    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry preflight tolerance must be numeric") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("geometry preflight tolerance must be finite and nonnegative")
    source = _points(source_points_robot_base_m, label="source points")
    report = rigid_geometry_report(source, target_points_robot_base_m, tolerance_m=tolerance)
    if target_points_robot_base_m is None:
        return {"ok": True, "checked": False, "reason": None, **report}
    target = _points(target_points_robot_base_m, label="target points")
    point_error_bound = 2.0 * math.sqrt(3.0) * tolerance
    failures: list[str] = []
    for pair in report["pair_distances"]:
        if float(pair["distance_error_m"]) > point_error_bound + 1.0e-9:
            failures.append(
                "pair "
                f"{pair['indices']} distance changes by {pair['distance_error_m']:.6f}m"
            )
    source_rank = int(report["source_rank"])
    target_rank = int(report["target_rank"])
    # Rank labels are diagnostic only here.  A pointwise tolerance can make a
    # very shallow tetrahedron effectively coplanar while it is still a valid
    # target, so rejecting on rank mismatch alone would violate the
    # conservative-preflight contract.  Pair bounds above remain a strict
    # lower bound for distance-changing requests.
    if len(source) >= 4 and source_rank == 3 and target_rank == 3:
        source_volume = tetrahedron_signed_volume_m3(source)
        target_volume = tetrahedron_signed_volume_m3(target)
        # Bound the largest signed-volume change induced by moving each target
        # point by at most sqrt(3)*tolerance (the public coordinate box).  A
        # sign change outside that bound cannot be repaired by any proper
        # rotation, so it is a genuine improper/mirror request.  Near-zero
        # volumes are intentionally left to the nonlinear solver.
        point_radius = math.sqrt(3.0) * tolerance
        edge_a = float(np.linalg.norm(target[1] - target[0]))
        edge_b = float(np.linalg.norm(target[2] - target[0]))
        edge_c = float(np.linalg.norm(target[3] - target[0]))
        edge_bound = 2.0 * point_radius
        determinant_change_bound = (
            edge_bound * (edge_b + edge_bound) * (edge_c + edge_bound)
            + edge_a * edge_bound * (edge_c + edge_bound)
            + edge_a * edge_b * edge_bound
        ) / 6.0
        sign_reversed = source_volume * target_volume < 0.0
        # Fully numeric coordinates are an explicit correspondence, not an
        # interval.  If all pair distances agree to floating-point precision,
        # an opposite tetrahedron sign is an unambiguous mirror even when the
        # caller chose a comparatively wide public position tolerance.  For
        # noisy/approximate targets retain the conservative uncertainty bound.
        exact_numeric_mirror = (
            sign_reversed
            and max(
                (float(item["distance_error_m"]) for item in report["pair_distances"]),
                default=0.0,
            )
            <= 1.0e-9
            and abs(source_volume) > GEOMETRY_DEGENERATE_VOLUME_M3
            and abs(target_volume) > GEOMETRY_DEGENERATE_VOLUME_M3
        )
        bounded_mirror = (
            sign_reversed
            and abs(target_volume)
            > max(GEOMETRY_DEGENERATE_VOLUME_M3, determinant_change_bound)
        )
        if exact_numeric_mirror or bounded_mirror:
            failures.append("target four-point tetrahedron is an improper mirror transform")
    return {
        "ok": not failures,
        "checked": True,
        "reason": "; ".join(failures) if failures else None,
        "pointwise_distance_error_bound_m": float(point_error_bound),
        **report,
    }


def signed_volume_status(
    source_volume: float,
    target_volume: float,
    source_points: np.ndarray,
    target_points: np.ndarray,
    tolerance_m: float,
) -> bool | None:
    """Classify tetrahedron handedness, retaining an ambiguity band.

    ``None`` means that the observed volume is close enough to coplanar that
    depth noise of the requested tolerance could change its sign.  Treating
    that case as a failure would reject valid nearly-planar four-marker
    bundles; only a provable sign reversal is an improper rigid transform.
    """

    if source_volume == 0.0 or target_volume == 0.0:
        return None
    if source_volume * target_volume > 0.0:
        return True
    point_radius = math.sqrt(3.0) * float(tolerance_m)
    edge_bound = 2.0 * point_radius
    # Use the three edges from the first vertex.  Their largest values give a
    # conservative determinant perturbation bound for either point set.
    source_edges = [
        float(np.linalg.norm(source_points[index] - source_points[0]))
        for index in (1, 2, 3)
    ]
    target_edges = [
        float(np.linalg.norm(target_points[index] - target_points[0]))
        for index in (1, 2, 3)
    ]
    edge_a = max(source_edges[0], target_edges[0])
    edge_b = max(source_edges[1], target_edges[1])
    edge_c = max(source_edges[2], target_edges[2])
    determinant_change_bound = (
        edge_bound * (edge_b + edge_bound) * (edge_c + edge_bound)
        + edge_a * edge_bound * (edge_c + edge_bound)
        + edge_a * edge_b * edge_bound
    ) / 6.0
    if (
        abs(source_volume) > max(GEOMETRY_DEGENERATE_VOLUME_M3, determinant_change_bound)
        and abs(target_volume) > max(GEOMETRY_DEGENERATE_VOLUME_M3, determinant_change_bound)
    ):
        return False
    return None


def _unit_vector(raw: Sequence[float], *, label: str) -> np.ndarray:
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and non-zero") from exc
    if vector.ndim != 1 or vector.shape != (3,):
        raise ValueError(f"{label} must be a length-3 vector")
    vector = vector.copy()
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError(f"{label} must be finite and non-zero")
    return vector / norm


def _line_segment_overlap_metrics(
    controlled_segment: np.ndarray,
    reference_start: np.ndarray,
    reference_end: np.ndarray,
) -> dict[str, Any]:
    """Measure line coincidence and finite-interval overlap in one frame."""

    reference_axis = reference_end - reference_start
    reference_length = float(np.linalg.norm(reference_axis))
    if reference_length <= 1.0e-12:
        raise ValueError("reference segment endpoints must be distinct")
    controlled_axis = controlled_segment[1] - controlled_segment[0]
    controlled_length = float(np.linalg.norm(controlled_axis))
    direction = reference_axis / reference_length
    if controlled_length <= 1.0e-12:
        alignment_error = math.pi
        alignment_residual = np.asarray([math.pi], dtype=np.float64)
    else:
        controlled_direction = controlled_axis / controlled_length
        alignment_error = math.acos(
            float(
                np.clip(
                    abs(float(controlled_direction @ direction)),
                    -1.0,
                    1.0,
                )
            )
        )
        # The relation is unoriented, so both parallel and anti-parallel are
        # exact.  The cross product provides a smooth vector residual near
        # either solution while ``alignment_error`` remains the authoritative
        # angular metric used for reporting and candidate ordering.
        alignment_residual = np.cross(controlled_direction, direction)
    deltas = controlled_segment - reference_start[None, :]
    projections = deltas @ direction
    perpendicular = deltas - projections[:, None] * direction[None, :]
    distances = np.linalg.norm(perpendicular, axis=1)
    controlled_interval = (
        float(np.min(projections)),
        float(np.max(projections)),
    )
    reference_interval = (0.0, reference_length)
    intersection_low = max(controlled_interval[0], reference_interval[0])
    intersection_high = min(controlled_interval[1], reference_interval[1])
    overlap = max(0.0, intersection_high - intersection_low)
    gap = max(0.0, intersection_low - intersection_high)
    return {
        "direction": direction,
        "perpendicular": perpendicular,
        "point_distances": distances,
        "controlled_projections": projections,
        "controlled_interval": controlled_interval,
        "reference_interval": reference_interval,
        "segment_gap_m": float(gap),
        "segment_overlap_m": float(overlap),
        "controlled_length_m": controlled_length,
        "reference_length_m": reference_length,
        "degenerate": controlled_length <= 1.0e-12,
        "orientation_error_rad": float(alignment_error),
        "orientation_error_deg": float(math.degrees(alignment_error)),
        "orientation_residual": alignment_residual.astype(float).tolist(),
    }


def _line_segment_contains_metrics(
    controlled_segment: np.ndarray,
    reference_start: np.ndarray,
    reference_end: np.ndarray,
) -> dict[str, Any]:
    """Measure ordered containment ``control[0] -> ref[0] -> ref[1] -> control[1]``."""

    metrics = _line_segment_overlap_metrics(
        controlled_segment,
        reference_start,
        reference_end,
    )
    reference_direction = np.asarray(metrics["direction"], dtype=np.float64)
    controlled_axis = controlled_segment[1] - controlled_segment[0]
    controlled_length = float(metrics["controlled_length_m"])
    reference_length = float(metrics["reference_length_m"])
    projections = np.asarray(metrics["controlled_projections"], dtype=np.float64)
    if controlled_length <= 1.0e-12:
        directed_dot = -1.0
        orientation_error = math.pi
        orientation_residual = np.full(3, math.pi, dtype=np.float64)
    else:
        controlled_direction = controlled_axis / controlled_length
        directed_dot = float(
            np.clip(controlled_direction @ reference_direction, -1.0, 1.0)
        )
        orientation_error = math.acos(directed_dot)
        # Unlike ordinary overlap, point order is authoritative.  The vector
        # difference is zero only for the same direction; a cross product alone
        # would incorrectly have a zero residual at the anti-parallel solution.
        orientation_residual = controlled_direction - reference_direction
    start_margin = float(-projections[0])
    end_margin = float(projections[1] - reference_length)
    start_violation = max(0.0, -start_margin)
    end_violation = max(0.0, -end_margin)
    containment_violation = max(start_violation, end_violation)
    start_margin_shortfall = max(
        0.0, ORDERED_COLLINEAR_INNER_MARGIN_M - start_margin
    )
    end_margin_shortfall = max(
        0.0, ORDERED_COLLINEAR_INNER_MARGIN_M - end_margin
    )
    margin_shortfall = max(start_margin_shortfall, end_margin_shortfall)
    return {
        **metrics,
        "controlled_interval": (
            float(projections[0]),
            float(projections[1]),
        ),
        "directed_axis_dot": float(directed_dot),
        "orientation_error_rad": float(orientation_error),
        "orientation_error_deg": float(math.degrees(orientation_error)),
        "orientation_residual": orientation_residual.astype(float).tolist(),
        "required_inner_margin_m": float(ORDERED_COLLINEAR_INNER_MARGIN_M),
        "start_containment_margin_m": start_margin,
        "end_containment_margin_m": end_margin,
        "start_containment_violation_m": float(start_violation),
        "end_containment_violation_m": float(end_violation),
        "containment_violation_m": float(containment_violation),
        "start_margin_shortfall_m": float(start_margin_shortfall),
        "end_margin_shortfall_m": float(end_margin_shortfall),
        "containment_margin_shortfall_m": float(margin_shortfall),
        "reference_inside_controlled": bool(containment_violation <= 1.0e-9),
        "ordered_containment_margin_satisfied": bool(margin_shortfall <= 1.0e-9),
    }


def _relation_mode(raw: Any, *, label: str) -> str:
    mode = str(raw if raw is not None else "same").strip().lower()
    mode = {
        "same_direction": "same",
        "aligned": "same",
        "opposite_direction": "opposite",
        "either": "parallel",
        "undirected": "parallel",
    }.get(mode, mode)
    if mode not in {"same", "opposite", "parallel"}:
        raise ValueError(f"{label} must be same, opposite, or parallel")
    return mode


def _normalize_relation_records(
    relations: Sequence[Mapping[str, Any]] | None,
    point_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Apply the public typed-relation schema to direct local API callers.

    The HTTP boundary normally canonicalizes relations before they reach the
    planner.  This module is also intentionally usable by interface-free
    benchmarks, so it must not silently accept conflicting aliases or fields
    that the public contract would reject.  Keeping one canonicalizer here
    makes those callers observe exactly the same relation vocabulary.
    """

    if relations is None:
        return []
    if not isinstance(relations, (list, tuple)):
        raise ValueError("relations must be an array")
    try:
        normalized = _validate_move_tracked_relations(
            relations,
            point_names=set(point_names),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return [dict(item) for item in normalized]


def _vertical_to_ground_metrics(points: np.ndarray) -> dict[str, Any]:
    """Measure a vertical axis or plane against submission-local robot-base Z."""

    if len(points) == 2:
        delta = points[1] - points[0]
        length = float(np.linalg.norm(delta))
        if not np.all(np.isfinite(delta)) or not math.isfinite(length):
            raise ValueError("vertical_to_ground axis geometry is non-finite")
        degenerate = length <= 1.0e-12
        angle = (
            math.pi if degenerate
            else math.atan2(float(np.linalg.norm(delta[:2])), abs(float(delta[2])))
        )
        return {
            "kind": "position",
            "residual": np.full(2, 1.0e3) if degenerate else delta[:2].copy(),
            "max_error": 1.0e3 if degenerate else float(np.max(np.abs(delta[:2]))),
            "orientation_error_rad": angle,
            "orientation_residual": (
                np.ones(2) if degenerate else delta[:2] / length
            ),
            "degenerate": degenerate,
            "xy_difference_m": delta[:2].astype(float).tolist(),
            "axis_length_m": length,
        }
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    magnitude = float(np.linalg.norm(cross))
    if not np.all(np.isfinite(cross)) or not math.isfinite(magnitude):
        raise ValueError("vertical_to_ground plane geometry is non-finite")
    degenerate = magnitude <= GEOMETRY_DEGENERATE_AREA_M2
    normal = np.zeros(3) if degenerate else cross / magnitude
    # Signed elevation stays smooth at the horizontal-normal solution, and
    # reversing the triangle order changes only the residual sign.
    elevation = (
        math.pi if degenerate
        else math.atan2(float(normal[2]), float(np.linalg.norm(normal[:2])))
    )
    return {
        "kind": "orientation",
        "residual": np.asarray([elevation]),
        "max_error": abs(elevation),
        "degenerate": degenerate,
        "observed_normal_robot_base": None if degenerate else normal.astype(float).tolist(),
        "normal_z": None if degenerate else float(normal[2]),
        "on_hand_plane_area_m2": 0.5 * magnitude,
    }


def ordered_collinear_metrics(points: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Measure a spatial row order in a deterministic robot-base reading direction."""

    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    degenerate = not math.isfinite(length) or length <= 1.0e-9
    direction = np.zeros(3) if degenerate else delta / length
    axis = max((1, 2, 0), key=lambda index: abs(float(direction[index])))
    sign = 1.0 if axis == 0 else -1.0
    if direction[axis] * sign < 0.0:
        direction = -direction
    offsets = points - points[0]
    projections = offsets @ direction
    gaps = np.diff(projections)
    perpendicular = offsets - projections[:, None] * direction
    distances = np.linalg.norm(perpendicular, axis=1)
    # Keep a small positive gap for a unique order, independent of pos_tol.
    shortfalls = np.maximum(1.0e-5 - gaps, 0.0)
    order_ok = bool(not degenerate and np.all(gaps > 0.0))
    return {
        "kind": "position",
        "residual": (np.full(4 * len(points) - 1, 1.0e3) if degenerate else
                     np.concatenate((perpendicular.ravel(), shortfalls))),
        "max_error": 1.0e3 if degenerate else float(max(np.max(distances), np.max(shortfalls))),
        "degenerate": degenerate,
        "order_satisfied": order_ok,
        "requested_axial_order": list(names),
        "observed_axial_order": [names[i] for i in np.argsort(projections, kind="stable")],
        "axial_coordinates_m": projections.astype(float).tolist(),
        "adjacent_axial_gaps_m": gaps.astype(float).tolist(),
        "line_error_m": float(np.max(distances)),
        "order_violation_m": float(max(0.0, -np.min(gaps))),
        "reading_axis": ("x_near_to_far", "y_left_to_right", "z_top_to_bottom")[axis],
        "reading_direction_robot_base": direction.astype(float).tolist(),
    }


def relation_residual_blocks(
    points_robot_base_m: Sequence[Sequence[float]],
    relations: Sequence[Mapping[str, Any]] | None,
    *,
    point_names: Sequence[str] | None = None,
    _canonical: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate typed geometric relations as scalar/vector residual blocks."""

    points = _points(points_robot_base_m, label="relation points")
    if point_names is None:
        if relations:
            raise ValueError("point_names are required when relations are supplied")
        return []
    if any(not isinstance(name, str) for name in point_names):
        raise ValueError("point_names must contain only non-empty strings")
    names = [name.strip() for name in point_names]
    if len(names) != len(points):
        raise ValueError("point_names count does not match relation points")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("point_names must be unique and non-empty")
    if not relations:
        return []
    relation_records = (
        [dict(item) for item in relations]
        if _canonical
        else _normalize_relation_records(relations, names)
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    blocks: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(relation_records):
        if not isinstance(relation, Mapping):
            raise ValueError(f"relation {relation_index} must be an object")
        relation_type = str(relation.get("type") or "").strip().lower()
        raw_names = relation.get("point_names") or []
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError(
                f"relation {relation_index} point_names must be an array"
            )
        if any(not isinstance(name, str) for name in raw_names):
            raise ValueError(
                f"relation {relation_index} point_names must contain only strings"
            )
        raw_names = [name.strip() for name in raw_names]
        if len(set(raw_names)) != len(raw_names) or any(not name for name in raw_names):
            raise ValueError(
                f"relation {relation_index} point_names must be unique and non-empty"
            )
        required_count: int | tuple[int, int]
        if relation_type == "common_plane":
            required_count = (3, 4)
        elif relation_type == "vertical_to_ground":
            required_count = (2, 3)
        elif relation_type == "ordered_collinear":
            required_count = (2, 6)
        elif relation_type == "oriented_plane_normal":
            required_count = 3
        elif relation_type == "align_vector":
            required_count = 2
        elif relation_type == "point_at_position":
            required_count = 1
        elif relation_type in {
            "line_through_point",
            "line_coincident",
            "line_segment_overlap",
            "line_segment_contains",
        }:
            required_count = 2
        else:
            raise ValueError(f"unsupported relation type {relation_type!r}")
        if isinstance(required_count, tuple):
            if len(raw_names) < required_count[0] or len(raw_names) > required_count[1]:
                raise ValueError(
                    f"relation {relation_index} {relation_type} requires "
                    f"{required_count[0]} or {required_count[1]} points"
                )
        elif len(raw_names) != required_count:
            raise ValueError(
                f"relation {relation_index} {relation_type} requires "
                f"exactly {required_count} points"
            )
        indices: list[int] = []
        for name in raw_names:
            key = str(name)
            if key not in name_to_index:
                raise ValueError(f"relation {relation_index} references unknown point {key!r}")
            indices.append(name_to_index[key])
        selected = points[indices]
        if relation_type == "ordered_collinear":
            blocks.append({
                "index": int(relation_index), "type": relation_type,
                "point_names": raw_names,
                **ordered_collinear_metrics(selected, raw_names),
            })
            continue
        if relation_type == "vertical_to_ground":
            blocks.append({
                "index": int(relation_index),
                "type": relation_type,
                "point_names": raw_names,
                **_vertical_to_ground_metrics(selected),
            })
            continue
        if relation_type == "point_at_position":
            target = np.asarray(
                relation.get("target_position_robot_base_m"), dtype=np.float64
            )
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError(
                    f"relation {relation_index} target position must be a finite length-3 vector"
                )
            residual = selected[0] - target
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "position",
                    "residual": residual.astype(float),
                    "max_error": float(np.linalg.norm(residual)),
                    "degenerate": False,
                }
            )
            continue
        if relation_type == "line_through_point":
            target = np.asarray(
                relation.get("target_point_robot_base_m"), dtype=np.float64
            )
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError(
                    f"relation {relation_index} target point must be a finite length-3 vector"
                )
            axis = selected[1] - selected[0]
            axis_norm = float(np.linalg.norm(axis))
            degenerate = axis_norm <= 1.0e-12
            if degenerate:
                residual = np.full(3, 1.0e3, dtype=np.float64)
                distance = 1.0e3
            else:
                direction = axis / axis_norm
                delta = target - selected[0]
                residual = delta - float(delta @ direction) * direction
                distance = float(np.linalg.norm(residual))
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "position",
                    "residual": residual.astype(float),
                    "max_error": distance,
                    "degenerate": degenerate,
                    "target_point_robot_base_m": target.astype(float).tolist(),
                }
            )
            continue
        if relation_type == "line_coincident":
            line_point = np.asarray(
                relation.get("line_point_robot_base_m"), dtype=np.float64
            )
            if line_point.shape != (3,) or not np.all(np.isfinite(line_point)):
                raise ValueError(
                    f"relation {relation_index} line point must be a finite length-3 vector"
                )
            direction = _unit_vector(
                relation.get("line_direction_robot_base"),
                label=f"relation {relation_index} line direction",
            )
            deltas = selected - line_point[None, :]
            perpendicular = deltas - (deltas @ direction)[:, None] * direction[None, :]
            distances = np.linalg.norm(perpendicular, axis=1)
            controlled_axis = selected[1] - selected[0]
            controlled_length = float(np.linalg.norm(controlled_axis))
            if controlled_length <= 1.0e-12:
                orientation_error = math.pi
                orientation_residual = np.asarray([math.pi], dtype=np.float64)
                degenerate = True
            else:
                controlled_direction = controlled_axis / controlled_length
                orientation_error = math.acos(
                    float(
                        np.clip(
                            abs(float(controlled_direction @ direction)),
                            -1.0,
                            1.0,
                        )
                    )
                )
                orientation_residual = np.cross(controlled_direction, direction)
                degenerate = False
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "position",
                    "residual": perpendicular.reshape(-1).astype(float),
                    "max_error": float(np.max(distances)),
                    "degenerate": degenerate,
                    "point_distance_to_line_m": distances.astype(float).tolist(),
                    "line_point_robot_base_m": line_point.astype(float).tolist(),
                    "line_direction_robot_base": direction.astype(float).tolist(),
                    "orientation_error_rad": float(orientation_error),
                    "orientation_error_deg": float(math.degrees(orientation_error)),
                    "orientation_residual": orientation_residual.astype(float).tolist(),
                }
            )
            continue
        if relation_type in {"line_segment_overlap", "line_segment_contains"}:
            segment_start = np.asarray(
                relation.get("segment_start_robot_base_m"), dtype=np.float64
            )
            segment_end = np.asarray(
                relation.get("segment_end_robot_base_m"), dtype=np.float64
            )
            if (
                segment_start.shape != (3,)
                or segment_end.shape != (3,)
                or not np.all(np.isfinite(segment_start))
                or not np.all(np.isfinite(segment_end))
            ):
                raise ValueError(
                    f"relation {relation_index} target segment must contain two finite length-3 vectors"
                )
            metrics = (
                _line_segment_contains_metrics(
                    selected,
                    segment_start,
                    segment_end,
                )
                if relation_type == "line_segment_contains"
                else _line_segment_overlap_metrics(
                    selected,
                    segment_start,
                    segment_end,
                )
            )
            if metrics["degenerate"]:
                residual_size = 8 if relation_type == "line_segment_contains" else 7
                residual = np.full(residual_size, 1.0e3, dtype=np.float64)
                maximum = 1.0e3
            else:
                axial_residual = (
                    np.asarray(
                        [
                            metrics["start_margin_shortfall_m"],
                            metrics["end_margin_shortfall_m"],
                        ],
                        dtype=np.float64,
                    )
                    if relation_type == "line_segment_contains"
                    else np.asarray([metrics["segment_gap_m"]], dtype=np.float64)
                )
                residual = np.concatenate(
                    [
                        np.asarray(metrics["perpendicular"], dtype=np.float64).reshape(-1),
                        axial_residual,
                    ]
                )
                maximum = max(
                    float(np.max(metrics["point_distances"])),
                    float(
                        metrics["containment_margin_shortfall_m"]
                        if relation_type == "line_segment_contains"
                        else metrics["segment_gap_m"]
                    ),
                )
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "position",
                    "residual": residual.astype(float),
                    "max_error": float(maximum),
                    "degenerate": bool(metrics["degenerate"]),
                    "point_distance_to_line_m": np.asarray(
                        metrics["point_distances"], dtype=np.float64
                    ).astype(float).tolist(),
                    "segment_gap_m": float(metrics["segment_gap_m"]),
                    "segment_overlap_m": float(metrics["segment_overlap_m"]),
                    "controlled_projection_interval_m": list(
                        map(float, metrics["controlled_interval"])
                    ),
                    "reference_projection_interval_m": list(
                        map(float, metrics["reference_interval"])
                    ),
                    "controlled_segment_length_m": float(
                        metrics["controlled_length_m"]
                    ),
                    "reference_segment_length_m": float(
                        metrics["reference_length_m"]
                    ),
                    "segment_start_robot_base_m": segment_start.astype(float).tolist(),
                    "segment_end_robot_base_m": segment_end.astype(float).tolist(),
                    "orientation_error_rad": float(
                        metrics["orientation_error_rad"]
                    ),
                    "orientation_error_deg": float(
                        metrics["orientation_error_deg"]
                    ),
                    "orientation_residual": list(
                        metrics["orientation_residual"]
                    ),
                    **(
                        {
                            "ordered_point_sequence": [
                                str(raw_names[0]),
                                "segment_start",
                                "segment_end",
                                str(raw_names[1]),
                            ],
                            "directed_axis_dot": float(metrics["directed_axis_dot"]),
                            "required_inner_margin_m": float(
                                metrics["required_inner_margin_m"]
                            ),
                            "start_containment_margin_m": float(
                                metrics["start_containment_margin_m"]
                            ),
                            "end_containment_margin_m": float(
                                metrics["end_containment_margin_m"]
                            ),
                            "start_containment_violation_m": float(
                                metrics["start_containment_violation_m"]
                            ),
                            "end_containment_violation_m": float(
                                metrics["end_containment_violation_m"]
                            ),
                            "containment_violation_m": float(
                                metrics["containment_violation_m"]
                            ),
                            "start_margin_shortfall_m": float(
                                metrics["start_margin_shortfall_m"]
                            ),
                            "end_margin_shortfall_m": float(
                                metrics["end_margin_shortfall_m"]
                            ),
                            "containment_margin_shortfall_m": float(
                                metrics["containment_margin_shortfall_m"]
                            ),
                            "reference_inside_controlled": bool(
                                metrics["reference_inside_controlled"]
                            ),
                            "ordered_containment_margin_satisfied": bool(
                                metrics["ordered_containment_margin_satisfied"]
                            ),
                        }
                        if relation_type == "line_segment_contains"
                        else {}
                    ),
                }
            )
            continue
        if relation_type == "common_plane":
            normal = _unit_vector(
                relation.get("normal_robot_base", relation.get("normal")),
                label=f"relation {relation_index} normal_robot_base",
            )
            projections = selected @ normal
            offset_raw = relation.get(
                "offset_m", relation.get("offset", {"free": True})
            )
            if isinstance(offset_raw, Mapping) and offset_raw.get("free") is True:
                if set(offset_raw) != {"free"}:
                    raise ValueError(
                        f"relation {relation_index} offset must be a finite number or {{'free': true}}"
                    )
                offset = float(np.mean(projections))
            else:
                if isinstance(offset_raw, bool):
                    raise ValueError(f"relation {relation_index} offset must be numeric")
                try:
                    offset = float(offset_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"relation {relation_index} offset must be numeric or free"
                    ) from exc
                if not math.isfinite(offset):
                    raise ValueError(f"relation {relation_index} offset must be finite")
            residual = projections - offset
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "position",
                    "residual": residual.astype(float),
                    "max_error": float(np.max(np.abs(residual))),
                    "offset_resolved_m": offset,
                }
            )
            continue
        if relation_type == "oriented_plane_normal":
            normal = _unit_vector(
                relation.get(
                    "normal_robot_base",
                    relation.get(
                        "target_normal_robot_base", relation.get("normal")
                    ),
                ),
                label=f"relation {relation_index} normal_robot_base",
            )
            mode = _relation_mode(
                relation.get("mode", relation.get("sense", "same")),
                label=f"relation {relation_index} mode",
            )
            cross = np.cross(selected[1] - selected[0], selected[2] - selected[0])
            cross_norm = float(np.linalg.norm(cross))
            if cross_norm <= GEOMETRY_DEGENERATE_AREA_M2:
                angle = math.pi
                degenerate = True
            else:
                observed = cross / cross_norm
                dot = float(np.clip(observed @ normal, -1.0, 1.0))
                if mode == "opposite":
                    dot = -dot
                elif mode == "parallel":
                    dot = abs(dot)
                angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
                degenerate = False
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "orientation",
                    "residual": np.asarray([angle], dtype=np.float64),
                    "max_error": float(angle),
                    "degenerate": degenerate,
                }
            )
            continue
        if relation_type == "align_vector":
            direction = _unit_vector(
                relation.get(
                    "direction_robot_base",
                    relation.get(
                        "target_direction_robot_base", relation.get("direction")
                    ),
                ),
                label=f"relation {relation_index} direction_robot_base",
            )
            mode = _relation_mode(
                relation.get("mode", relation.get("sense", "same")),
                label=f"relation {relation_index} mode",
            )
            vector = selected[1] - selected[0]
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm <= 1.0e-12:
                angle = math.pi
                degenerate = True
            else:
                observed = vector / vector_norm
                dot = float(np.clip(observed @ direction, -1.0, 1.0))
                if mode == "opposite":
                    dot = -dot
                elif mode == "parallel":
                    dot = abs(dot)
                angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
                degenerate = False
            blocks.append(
                {
                    "index": int(relation_index),
                    "type": relation_type,
                    "point_names": [str(name) for name in raw_names],
                    "kind": "orientation",
                    "residual": np.asarray([angle], dtype=np.float64),
                    "max_error": float(angle),
                    "degenerate": degenerate,
                }
            )
            continue
        raise ValueError(f"unsupported relation type {relation_type!r}")
    return blocks


def evaluate_relations(
    points_robot_base_m: Sequence[Sequence[float]],
    relations: Sequence[Mapping[str, Any]] | None,
    *,
    point_names: Sequence[str],
    position_tolerance_m: float = 0.012,
    orientation_tolerance_deg: float = 5.0,
) -> dict[str, Any]:
    try:
        position_tolerance = float(position_tolerance_m)
        orientation_tolerance = float(orientation_tolerance_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("relation tolerances must be numeric") from exc
    if (
        not math.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
    ):
        raise ValueError("relation tolerances must be finite and positive")
    blocks = relation_residual_blocks(
        points_robot_base_m,
        relations,
        point_names=point_names,
    )
    reported_blocks: list[dict[str, Any]] = []
    position_checks: list[bool] = []
    collinear_errors: list[float] = []
    collinear_tolerances: list[float] = []
    for block in blocks:
        reported = dict(block)
        if block["kind"] == "position":
            effective_tolerance = relation_position_tolerance_m(
                block.get("type"), position_tolerance
            )
            error = float(block["max_error"])
            within_tolerance = bool(error <= effective_tolerance)
            reported.update(
                {
                    "requested_position_tolerance_m": position_tolerance,
                    "position_tolerance_m": effective_tolerance,
                    "within_position_tolerance": within_tolerance,
                }
            )
            position_checks.append(within_tolerance)
            if str(block.get("type") or "").strip().lower() in COLLINEAR_RELATION_TYPES:
                collinear_errors.append(error)
                collinear_tolerances.append(effective_tolerance)
        reported_blocks.append(reported)
    position_errors = [
        float(block["max_error"])
        for block in blocks
        if block["kind"] == "position"
    ]
    orientation_errors = [
        float(block["max_error"])
        for block in blocks
        if block["kind"] == "orientation"
    ]
    orientation_errors.extend(
        float(block["orientation_error_rad"])
        for block in blocks
        if block.get("orientation_error_rad") is not None
    )
    max_position = max(position_errors, default=0.0)
    max_orientation = max(orientation_errors, default=0.0)
    position_ok = all(position_checks)
    return {
        "ok": bool(
            position_ok
            and max_orientation <= math.radians(orientation_tolerance)
            and not any(bool(block.get("degenerate")) for block in blocks)
            and all(block.get("order_satisfied", True) for block in blocks)
        ),
        "axial_order_satisfied": all(block.get("order_satisfied", True) for block in blocks),
        "relations": [
            {
                **{key: value for key, value in block.items() if key != "residual"},
                "residual": np.asarray(block["residual"], dtype=np.float64).astype(float).tolist(),
            }
            for block in reported_blocks
        ],
        "max_relation_error_m": float(max_position),
        "max_collinear_error_m": (
            float(max(collinear_errors)) if collinear_errors else None
        ),
        "collinear_position_tolerance_m": (
            float(min(collinear_tolerances)) if collinear_tolerances else None
        ),
        "collinear_constraints_ok": bool(
            all(
                error <= tolerance
                for error, tolerance in zip(collinear_errors, collinear_tolerances)
            )
        ),
        "position_constraints_ok": bool(position_ok),
        "max_relation_error_rad": float(max_orientation),
        "max_relation_error_deg": float(math.degrees(max_orientation)),
        "position_tolerance_m": position_tolerance,
        "orientation_tolerance_deg": orientation_tolerance,
    }


def _quick_matrix(
    raw: Sequence[Sequence[float]],
    *,
    count: int,
    label: str,
) -> np.ndarray:
    """Validate one role subset without relaxing the legacy four-point API."""

    try:
        value = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an N x 3 array") from exc
    if value.shape != (int(count), 3):
        raise ValueError(f"{label} must have shape ({int(count)}, 3)")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain finite coordinates")
    return value.copy()


def _quick_plane_normal(
    points: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, float]:
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    magnitude = float(np.linalg.norm(cross))
    if not math.isfinite(magnitude) or magnitude <= GEOMETRY_DEGENERATE_AREA_M2:
        raise ValueError(
            f"{label} is degenerate (triangle area {0.5 * magnitude:.12g}m^2)"
        )
    return cross / magnitude, 0.5 * magnitude


def _quick_quat_to_mat_xyzw(raw: Sequence[float]) -> np.ndarray:
    """Return a finite camera-to-robot rotation from an xyzw quaternion."""

    try:
        quat = np.asarray(raw, dtype=np.float64).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera quaternion must contain four finite values") from exc
    if not np.all(np.isfinite(quat)):
        raise ValueError("camera quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("camera quaternion is degenerate")
    x, y, z, w = (quat / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quick_camera_forward(
    camera_position_robot_base_m: Sequence[float] | None,
    camera_quaternion_xyzw: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a frozen camera pose and return position, rotation, and -Z ray."""

    if camera_position_robot_base_m is None or camera_quaternion_xyzw is None:
        raise ValueError(
            "faceto requires the frozen head-camera relative pose"
        )
    try:
        position = np.asarray(
            camera_position_robot_base_m, dtype=np.float64
        ).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "camera position must contain three finite values"
        ) from exc
    if not np.all(np.isfinite(position)):
        raise ValueError("camera position must contain three finite values")
    rotation = _quick_quat_to_mat_xyzw(camera_quaternion_xyzw)
    # The official RGB-D convention uses camera -Z as optical forward.
    forward = rotation @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-12 or not np.all(np.isfinite(forward)):
        raise ValueError("camera forward vector is degenerate")
    return position, rotation, forward / forward_norm


def _quick_projected_winding(
    points_robot_base_m: np.ndarray,
    *,
    camera_position_robot_base_m: Sequence[float] | None,
    camera_quaternion_xyzw: Sequence[float] | None,
) -> dict[str, Any]:
    """Project a triangle with the evaluator camera and classify its winding."""

    camera_position, rotation, _forward = _quick_camera_forward(
        camera_position_robot_base_m,
        camera_quaternion_xyzw,
    )
    camera_points = (points_robot_base_m - camera_position[None, :]) @ rotation
    depths = -camera_points[:, 2]
    if not np.all(np.isfinite(depths)) or np.any(depths <= 1.0e-9):
        raise ValueError("faceto triangle is not in front of the head camera")
    uv = np.column_stack(
        (
            camera_points[:, 0] / depths,
            -camera_points[:, 1] / depths,
        )
    )
    signed_double_area = float(
        sum(
            uv[index, 0] * uv[(index + 1) % 3, 1]
            - uv[(index + 1) % 3, 0] * uv[index, 1]
            for index in range(3)
        )
    )
    if not math.isfinite(signed_double_area) or abs(signed_double_area) <= 1.0e-12:
        raise ValueError("faceto triangle projection is degenerate in the head camera")
    # Image v grows downward.  Consequently positive shoelace area is a
    # clockwise traversal in the displayed head image.
    winding = "clockwise" if signed_double_area > 0.0 else "counterclockwise"
    return {
        "camera_points": camera_points,
        "depths": depths,
        "uv_normalized": uv,
        "projected_signed_double_area": signed_double_area,
        "projected_signed_area": 0.5 * signed_double_area,
        "projected_winding": winding,
    }


def _quick_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "touch": "touch",
        "point_touch": "touch",
        "point-to-point": "touch",
        "flatwise": "flatwise",
        "flat_wise": "flatwise",
        "plane_parallel_to_ground": "flatwise",
        "plane-parallel": "plane_parallel",
        "plane_parallel": "plane_parallel",
        "two_planes_parallel": "plane_parallel",
        "planes_parallel": "plane_parallel",
        "parallel_planes": "plane_parallel",
        "two_faces_parallel": "plane_parallel",
        "collinear": "collinear",
        "co_linear": "collinear",
        "line-vertical-to-plane": "line_vertical_to_plane",
        "line_vertical_to_plane": "line_vertical_to_plane",
        "axis_perpendicular_to_plane": "line_vertical_to_plane",
        "line_perpendicular_to_plane": "line_vertical_to_plane",
        "perpendicular_to_plane": "line_vertical_to_plane",
        "on_axis_perpendicular_to_off_plane": "line_vertical_to_plane",
        "vertical_to_ground": "vertical_to_ground",
        "vertical-to-ground": "vertical_to_ground",
        "vertical to ground": "vertical_to_ground",
        "faceto": "faceto",
        "face_to": "faceto",
        "face-to": "faceto",
        "face to": "faceto",
        "reverse_faceto": "reverse_faceto",
        "reverse_face_to": "reverse_faceto",
        "reverse-face-to": "reverse_faceto",
        "reverse face to": "reverse_faceto",
    }
    quick_type = aliases.get(value)
    if quick_type is None:
        raise ValueError("unsupported quick constraint type")
    return quick_type


def _validate_quick_unoriented(raw: Mapping[str, Any]) -> None:
    """Keep direct local/benchmark calls aligned with the public contract."""

    present = [
        key
        for key in ("normal_mode", "direction_mode", "mode", "sense")
        if key in raw
    ]
    if len(present) > 1:
        raise ValueError(
            "quick constraint orientation mode must use only one field"
        )
    if not present:
        return
    value = str(raw[present[0]]).strip().lower()
    if value not in {"parallel", "either", "undirected"}:
        raise ValueError(
            "quick constraints are unoriented; use parallel/either, not same/opposite"
        )


def _quick_collinear_axial_mode(
    quick_constraint: Mapping[str, Any],
    *,
    off_hand_point_count: int,
) -> str:
    axial_mode = str(
        quick_constraint.get(
            "axial_mode",
            "ordered",
        )
    ).strip().lower()
    axial_mode = {
        "overlap": "segment_overlap",
        "contact": "segment_overlap",
        "nearest": "segment_overlap",
        "contain": "ordered_containment",
        "contains": "ordered_containment",
        "reference_inside": "ordered_containment",
    }.get(axial_mode, axial_mode)
    if axial_mode not in {
        "ordered",
        "ordered_containment",
        "segment_overlap",
        "line_only",
    }:
        raise ValueError(
            "collinear axial_mode must be ordered_containment, segment_overlap, "
            "or line_only"
        )
    if off_hand_point_count != 2 and axial_mode not in {"line_only", "ordered"}:
        raise ValueError(
            f"collinear axial_mode {axial_mode} requires two off-hand points"
        )
    return axial_mode


def quick_constraint_relation(
    quick_constraint: Mapping[str, Any],
    *,
    on_hand_names: Sequence[str] | None = None,
    on_hand_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    off_hand_points_robot_base_m: Sequence[Sequence[float]] | None = None,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Convert a role-based preset into the existing local relation vocabulary.

    The returned relation is a *planning* relation.  For off-hand geometry its
    normal/direction is derived from the frozen RGB-D coordinates.  ``faceto``
    and ``reverse_faceto`` instead derive the target normal from the frozen
    head-camera optical axis and retain the submitted three-point winding for
    independent live validation.  No simulator object or pose is consulted.
    """

    if not isinstance(quick_constraint, Mapping):
        raise ValueError("quick_constraint must be an object")
    _validate_quick_unoriented(quick_constraint)
    quick_type = _quick_type(quick_constraint.get("type"))
    if "axial_mode" in quick_constraint and quick_type != "collinear":
        raise ValueError("quick constraint axial_mode is only valid for collinear")
    if on_hand_names is None:
        raw_names = quick_constraint.get("on_hand_points")
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError("quick_constraint.on_hand_points must be an array")
        on_names = [str(name).strip() for name in raw_names]
    else:
        on_names = [str(name).strip() for name in on_hand_names]
    if quick_type in {"faceto", "reverse_faceto"}:
        if len(on_names) != 3:
            raise ValueError(f"{quick_type} requires three on-hand names")
        if off_hand_points_robot_base_m is not None:
            _quick_matrix(
                off_hand_points_robot_base_m,
                count=0,
                label="off-hand points",
            )
        on = _quick_matrix(
            on_hand_points_robot_base_m,
            count=3,
            label="on-hand faceto source points",
        )
        _camera_position, _camera_rotation, camera_forward = _quick_camera_forward(
            camera_position_robot_base_m,
            camera_quaternion_xyzw,
        )
        # Validate the source triangle before producing a relation.  Its
        # orientation is supplied by the ordered rows; the solver will rotate
        # the rigid bundle so that the requested image winding is attainable.
        _quick_plane_normal(on, label="on-hand faceto source triangle")
        target_normal = (
            camera_forward
            if quick_type == "faceto"
            else -camera_forward
        )
        return {
            "type": "oriented_plane_normal",
            "point_names": on_names,
            "normal_robot_base": target_normal.astype(float).tolist(),
            "mode": "same",
        }
    if quick_type == "vertical_to_ground":
        if len(on_names) not in {2, 3}:
            raise ValueError("vertical_to_ground requires two or three on-hand names")
        if off_hand_points_robot_base_m is not None:
            _quick_matrix(off_hand_points_robot_base_m, count=0, label="off-hand points")
        return {"type": quick_type, "point_names": on_names}
    if quick_type == "touch":
        if len(on_names) != 1:
            raise ValueError("touch requires one on-hand name")
        off = _quick_matrix(
            off_hand_points_robot_base_m, count=1, label="off-hand reference points"
        )
        return {
            "type": "point_at_position",
            "point_names": on_names,
            "target_position_robot_base_m": off[0].astype(float).tolist(),
        }
    if quick_type == "flatwise":
        if len(on_names) != 3:
            raise ValueError("flatwise requires three on-hand names")
        return {
            "type": "oriented_plane_normal",
            "point_names": on_names,
            "normal_robot_base": [0.0, 0.0, 1.0],
            "mode": "parallel",
        }
    if quick_type == "collinear":
        if len(on_names) != 2:
            raise ValueError("collinear requires two on-hand names")
        raw_off = np.asarray(off_hand_points_robot_base_m, dtype=np.float64)
        if raw_off.shape not in {(1, 3), (2, 3)} or not np.isfinite(raw_off).all():
            raise ValueError("collinear requires one or two finite off-hand reference points")
        if _quick_collinear_axial_mode(quick_constraint, off_hand_point_count=len(raw_off)) == "ordered":
            off_names = list(quick_constraint.get("off_hand_points") or [])
            order = list(quick_constraint.get("axial_point_order") or [on_names[0], *off_names, on_names[1]])
            if len(order) != 2 + len(raw_off) or set(order) != set(on_names + off_names):
                raise ValueError("collinear axial point order does not match its points")
            return {"type": "ordered_collinear", "point_names": order}
        if raw_off.shape == (1, 3):
            _quick_collinear_axial_mode(
                quick_constraint,
                off_hand_point_count=1,
            )
            off = _quick_matrix(raw_off, count=1, label="off-hand reference points")
            return {
                "type": "line_through_point",
                "point_names": on_names,
                "target_point_robot_base_m": off[0].astype(float).tolist(),
            }
        off = _quick_matrix(raw_off, count=2, label="off-hand reference points")
        direction = off[1] - off[0]
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-12:
            raise ValueError("collinear off-hand line points are coincident")
        axial_mode = _quick_collinear_axial_mode(
            quick_constraint,
            off_hand_point_count=2,
        )
        if axial_mode == "line_only":
            return {
                "type": "line_coincident",
                "point_names": on_names,
                "line_point_robot_base_m": off[0].astype(float).tolist(),
                "line_direction_robot_base": (direction / length).astype(float).tolist(),
            }
        if axial_mode == "ordered_containment":
            return {
                "type": "line_segment_contains",
                "point_names": on_names,
                "segment_start_robot_base_m": off[0].astype(float).tolist(),
                "segment_end_robot_base_m": off[1].astype(float).tolist(),
            }
        return {
            "type": "line_segment_overlap",
            "point_names": on_names,
            "segment_start_robot_base_m": off[0].astype(float).tolist(),
            "segment_end_robot_base_m": off[1].astype(float).tolist(),
        }

    off = _quick_matrix(
        off_hand_points_robot_base_m, count=3, label="off-hand reference points"
    )
    normal, _area = _quick_plane_normal(off, label="off-hand reference plane")
    if quick_type == "plane_parallel":
        if len(on_names) != 3:
            raise ValueError("plane_parallel requires three on-hand names")
        return {
            "type": "oriented_plane_normal",
            "point_names": on_names,
            "normal_robot_base": normal.astype(float).tolist(),
            "mode": "parallel",
        }
    if len(on_names) != 2:
        raise ValueError(
            "line_vertical_to_plane requires two on-hand names"
        )
    return {
        "type": "align_vector",
        "point_names": on_names,
        "direction_robot_base": normal.astype(float).tolist(),
        "mode": "parallel",
    }


def evaluate_quick_constraint(
    on_hand_points_robot_base_m: Sequence[Sequence[float]],
    off_hand_points_robot_base_m: Sequence[Sequence[float]] | None,
    quick_constraint: Mapping[str, Any],
    *,
    position_tolerance_m: float = 0.012,
    orientation_tolerance_deg: float = 5.0,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate a role-based preset from two live/frozen point bundles.

    This evaluator is deliberately independent from the synthetic planning
    relation.  It catches a moving off-hand reference and reports the actual
    two normals (or the axis) used for the final acceptance decision.
    """

    if not isinstance(quick_constraint, Mapping):
        raise ValueError("quick_constraint must be an object")
    _validate_quick_unoriented(quick_constraint)
    quick_type = _quick_type(quick_constraint.get("type"))
    if "axial_mode" in quick_constraint and quick_type != "collinear":
        raise ValueError("quick constraint axial_mode is only valid for collinear")
    on_count = {
        "touch": 1,
        "flatwise": 3,
        "plane_parallel": 3,
        "collinear": 2,
        "line_vertical_to_plane": 2,
        "faceto": 3,
        "reverse_faceto": 3,
    }.get(quick_type)
    if quick_type == "vertical_to_ground":
        raw_on = np.asarray(on_hand_points_robot_base_m, dtype=np.float64)
        if raw_on.shape not in {(2, 3), (3, 3)}:
            raise ValueError("vertical_to_ground requires two or three on-hand points")
        on_count = len(raw_on)
        if off_hand_points_robot_base_m is not None:
            _quick_matrix(off_hand_points_robot_base_m, count=0, label="off-hand points")
    on = _quick_matrix(
        on_hand_points_robot_base_m,
        count=on_count,
        label="on-hand points",
    )
    try:
        position_tolerance = float(position_tolerance_m)
        orientation_tolerance = float(orientation_tolerance_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("quick constraint tolerances must be numeric") from exc
    if (
        not math.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
    ):
        raise ValueError("quick constraint tolerances must be finite and positive")

    if quick_type in {"faceto", "reverse_faceto"}:
        try:
            camera_position, _camera_rotation, camera_forward = _quick_camera_forward(
                camera_position_robot_base_m,
                camera_quaternion_xyzw,
            )
            observed_normal, triangle_area = _quick_plane_normal(
                on,
                label="on-hand faceto triangle",
            )
            projection = _quick_projected_winding(
                on,
                camera_position_robot_base_m=camera_position,
                camera_quaternion_xyzw=camera_quaternion_xyzw,
            )
        except ValueError as exc:
            message = str(exc)
            if "front" in message:
                error = "faceto_triangle_not_in_front_of_camera"
            elif "projection" in message:
                error = "faceto_projected_triangle_degenerate"
            elif "degenerate" in message:
                error = "faceto_triangle_degenerate"
            else:
                error = "faceto_camera_pose_unavailable"
            return {
                "ok": False,
                "type": quick_type,
                "error": error,
                "detail": message,
                "max_error_m": 0.0,
                "max_error_rad": math.pi,
                "max_error_deg": 180.0,
                "on_hand_points_robot_base_m": on.astype(float).tolist(),
                "off_hand_points_robot_base_m": [],
            }
        dot = float(np.clip(observed_normal @ camera_forward, -1.0, 1.0))
        acute_angle = float(math.acos(float(np.clip(abs(dot), -1.0, 1.0))))
        desired_winding = (
            "clockwise" if quick_type == "faceto" else "counterclockwise"
        )
        winding_ok = projection["projected_winding"] == desired_winding
        orientation_ok = acute_angle <= math.radians(orientation_tolerance)
        ok = bool(orientation_ok and winding_ok)
        return {
            "ok": ok,
            "type": quick_type,
            "mode": "camera_facing" if quick_type == "faceto" else "camera_away",
            "error": None if ok else (
                "faceto_winding_not_satisfied"
                if not winding_ok
                else "faceto_camera_angle_not_satisfied"
            ),
            "max_error_m": 0.0,
            "max_error_rad": acute_angle,
            "max_error_deg": float(math.degrees(acute_angle)),
            "acute_normal_camera_angle_deg": float(math.degrees(acute_angle)),
            "normal_camera_alignment_dot": float(abs(dot)),
            "reference_camera_forward_robot_base": camera_forward.astype(float).tolist(),
            "observed_normal_robot_base": observed_normal.astype(float).tolist(),
            "triangle_area_m2": float(triangle_area),
            "projected_winding": projection["projected_winding"],
            "desired_winding": desired_winding,
            "winding_ok": bool(winding_ok),
            "projected_signed_area": float(projection["projected_signed_area"]),
            "projected_uv_normalized": projection["uv_normalized"].astype(float).tolist(),
            "orientation_tolerance_deg": orientation_tolerance,
            "on_hand_points_robot_base_m": on.astype(float).tolist(),
            "off_hand_points_robot_base_m": [],
        }

    if quick_type == "vertical_to_ground":
        metrics = _vertical_to_ground_metrics(on)
        position_error = metrics["max_error"] if on_count == 2 else 0.0
        angle = metrics["orientation_error_rad"] if on_count == 2 else metrics["max_error"]
        ok = bool(
            not metrics["degenerate"]
            and position_error <= position_tolerance
            and angle <= math.radians(orientation_tolerance)
        )
        return {
            **{
                key: value for key, value in metrics.items()
                if key not in {"residual", "orientation_residual", "kind", "max_error"}
            },
            "ok": ok,
            "type": quick_type,
            "point_count": on_count,
            "error": (
                ("on_hand_axis_degenerate" if on_count == 2 else "on_hand_plane_degenerate")
                if metrics["degenerate"]
                else (None if ok else "vertical_to_ground_not_satisfied")
            ),
            "max_error_m": float(position_error),
            "max_error_rad": float(angle),
            "max_error_deg": math.degrees(angle),
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "reference_normal_robot_base": [0.0, 0.0, 1.0],
            "on_hand_points_robot_base_m": on.astype(float).tolist(),
            "off_hand_points_robot_base_m": [],
        }

    if quick_type == "touch":
        off = _quick_matrix(off_hand_points_robot_base_m, count=1, label="off-hand points")
        distance = float(np.linalg.norm(on[0] - off[0]))
        return {
            "ok": bool(distance <= position_tolerance),
            "type": quick_type,
            "error": None if distance <= position_tolerance else "touch_not_satisfied",
            "max_error_m": distance,
            "max_error_rad": 0.0,
            "max_error_deg": 0.0,
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "on_hand_points_robot_base_m": on.astype(float).tolist(),
            "off_hand_points_robot_base_m": off.astype(float).tolist(),
        }

    if quick_type == "collinear":
        collinear_position_tolerance = relation_position_tolerance_m(
            "line_coincident", position_tolerance
        )
        raw_off = np.asarray(off_hand_points_robot_base_m, dtype=np.float64)
        if raw_off.shape not in {(1, 3), (2, 3)} or not np.all(np.isfinite(raw_off)):
            raise ValueError("collinear off-hand points must have shape (1, 3) or (2, 3)")
        on_axis = on[1] - on[0]
        on_length = float(np.linalg.norm(on_axis))
        if on_length <= 1.0e-12:
            return {
                "ok": False, "type": quick_type, "error": "on_hand_axis_degenerate",
                "max_error_m": 1.0e3, "max_error_rad": 0.0, "max_error_deg": 0.0,
            }
        angular_error = 0.0
        if len(raw_off) == 1:
            direction = on_axis / on_length
            delta = raw_off[0] - on[0]
            distances = np.asarray(
                [np.linalg.norm(delta - float(delta @ direction) * direction)]
            )
        else:
            off_axis = raw_off[1] - raw_off[0]
            off_length = float(np.linalg.norm(off_axis))
            if off_length <= 1.0e-12:
                return {
                    "ok": False, "type": quick_type, "error": "off_hand_axis_degenerate",
                    "max_error_m": 1.0e3, "max_error_rad": 0.0, "max_error_deg": 0.0,
                }
            direction = off_axis / off_length
            directed_dot = float(
                np.clip((on_axis / on_length) @ direction, -1.0, 1.0)
            )
            deltas = on - raw_off[0]
            distances = np.linalg.norm(
                deltas - (deltas @ direction)[:, None] * direction[None, :], axis=1
            )
        axial_mode = _quick_collinear_axial_mode(
            quick_constraint,
            off_hand_point_count=len(raw_off),
        )
        if axial_mode == "ordered":
            on_names = list(quick_constraint["on_hand_points"])
            off_names = list(quick_constraint["off_hand_points"])
            order = list(quick_constraint.get("axial_point_order") or [on_names[0], *off_names, on_names[1]])
            if len(order) != len(on) + len(raw_off) or set(order) != set(on_names + off_names):
                raise ValueError("collinear axial point order does not match its points")
            by_name = dict(zip(on_names + off_names, np.vstack((on, raw_off))))
            metrics = ordered_collinear_metrics(np.asarray([by_name[name] for name in order]), order)
            segment_metrics = (_line_segment_contains_metrics(on, raw_off[0], raw_off[1])
                               if len(raw_off) == 2 else {})
            ok = bool(metrics["order_satisfied"] and metrics["max_error"] <= collinear_position_tolerance)
            return {
                **segment_metrics,
                **{key: value for key, value in metrics.items() if key != "residual"},
                "ok": ok, "type": quick_type,
                "error": None if ok else ("collinear_axial_order_not_satisfied" if not metrics["order_satisfied"] else "collinear_not_satisfied"),
                "axial_mode": axial_mode, "axial_point_order": order,
                "max_error_m": metrics["max_error"],
                "max_error_rad": float(segment_metrics.get("orientation_error_rad", 0.0)),
                "max_error_deg": float(segment_metrics.get("orientation_error_deg", 0.0)),
                "ordered_point_sequence": order,
                "position_tolerance_m": collinear_position_tolerance,
                "requested_position_tolerance_m": position_tolerance,
                "orientation_tolerance_deg": orientation_tolerance,
                "on_hand_points_robot_base_m": on.astype(float).tolist(),
                "off_hand_points_robot_base_m": raw_off.astype(float).tolist(),
            }
        segment_gap = 0.0
        segment_overlap = 0.0
        containment_violation = 0.0
        containment_margin_shortfall = 0.0
        required_inner_margin = None
        start_containment_margin = None
        end_containment_margin = None
        reference_inside_controlled = None
        ordered_point_sequence = None
        controlled_interval: list[float] | None = None
        reference_interval: list[float] | None = None
        if len(raw_off) == 2:
            metrics = (
                _line_segment_contains_metrics(on, raw_off[0], raw_off[1])
                if axial_mode == "ordered_containment"
                else _line_segment_overlap_metrics(on, raw_off[0], raw_off[1])
            )
            angular_error = float(metrics["orientation_error_rad"])
            segment_gap = float(metrics["segment_gap_m"])
            segment_overlap = float(metrics["segment_overlap_m"])
            controlled_interval = list(map(float, metrics["controlled_interval"]))
            reference_interval = list(map(float, metrics["reference_interval"]))
            if axial_mode == "ordered_containment":
                containment_violation = float(metrics["containment_violation_m"])
                containment_margin_shortfall = float(
                    metrics["containment_margin_shortfall_m"]
                )
                required_inner_margin = float(metrics["required_inner_margin_m"])
                start_containment_margin = float(
                    metrics["start_containment_margin_m"]
                )
                end_containment_margin = float(metrics["end_containment_margin_m"])
                reference_inside_controlled = bool(
                    metrics["reference_inside_controlled"]
                )
                ordered_point_sequence = list(
                    map(
                        str,
                        quick_constraint.get(
                            "axial_point_order",
                            [
                                quick_constraint["on_hand_points"][0],
                                quick_constraint["off_hand_points"][0],
                                quick_constraint["off_hand_points"][1],
                                quick_constraint["on_hand_points"][1],
                            ],
                        ),
                    )
                )
        line_error = float(np.max(distances))
        maximum = max(
            line_error,
            segment_gap if axial_mode == "segment_overlap" else 0.0,
            containment_margin_shortfall
            if axial_mode == "ordered_containment"
            else 0.0,
        )
        ok = bool(
            maximum <= collinear_position_tolerance
            and angular_error <= math.radians(orientation_tolerance)
            and (
                axial_mode != "ordered_containment"
                or containment_violation
                <= ORDERED_COLLINEAR_LIVE_BOUNDARY_TOLERANCE_M
            )
        )
        if not ok and line_error > collinear_position_tolerance:
            error = "collinear_not_satisfied"
        elif not ok and angular_error > math.radians(orientation_tolerance):
            error = "collinear_orientation_not_satisfied"
        elif not ok:
            error = (
                "collinear_reference_not_inside_on_hand_segment"
                if axial_mode == "ordered_containment"
                else "collinear_segments_do_not_overlap"
            )
        else:
            error = None
        return {
            "ok": bool(ok),
            "type": quick_type,
            "error": error,
            "max_error_m": maximum,
            "line_error_m": line_error,
            "point_distance_to_line_m": distances.astype(float).tolist(),
            "axial_mode": axial_mode,
            "segment_gap_m": segment_gap,
            "segment_overlap_m": segment_overlap,
            "containment_violation_m": containment_violation,
            "containment_margin_shortfall_m": containment_margin_shortfall,
            "required_inner_margin_m": required_inner_margin,
            "start_containment_margin_m": start_containment_margin,
            "end_containment_margin_m": end_containment_margin,
            "reference_inside_controlled": reference_inside_controlled,
            "ordered_point_sequence": ordered_point_sequence,
            "axial_point_order": list(
                map(str, quick_constraint.get("axial_point_order", []))
            ),
            "controlled_projection_interval_m": controlled_interval,
            "reference_projection_interval_m": reference_interval,
            "max_error_rad": float(angular_error),
            "max_error_deg": float(math.degrees(angular_error)),
            "position_tolerance_m": collinear_position_tolerance,
            "requested_position_tolerance_m": position_tolerance,
            "collinear_position_tolerance_m": collinear_position_tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "on_hand_points_robot_base_m": on.astype(float).tolist(),
            "off_hand_points_robot_base_m": raw_off.astype(float).tolist(),
        }

    if quick_type == "flatwise":
        off = np.empty((0, 3), dtype=np.float64)
        reference_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        reference_area = None
    else:
        off = _quick_matrix(off_hand_points_robot_base_m, count=3, label="off-hand points")
        try:
            reference_normal, reference_area = _quick_plane_normal(
                off, label="off-hand reference plane"
            )
        except ValueError as exc:
            return {
                "ok": False,
                "type": quick_type,
                "error": "off_hand_plane_degenerate",
                "detail": str(exc),
                "max_error_m": 0.0,
                "max_error_rad": math.pi,
                "max_error_deg": 180.0,
                "reference_plane_area_m2": None,
            }
    if quick_type in {"flatwise", "plane_parallel"}:
        try:
            observed_normal, observed_area = _quick_plane_normal(
                on, label="on-hand plane"
            )
        except ValueError as exc:
            return {
                "ok": False,
                "type": quick_type,
                "error": "on_hand_plane_degenerate",
                "detail": str(exc),
                "max_error_m": 0.0,
                "max_error_rad": math.pi,
                "max_error_deg": 180.0,
                "reference_plane_area_m2": (
                    None if reference_area is None else float(reference_area)
                ),
            }
    else:
        axis = on[1] - on[0]
        axis_length = float(np.linalg.norm(axis))
        if not math.isfinite(axis_length) or axis_length <= 1.0e-12:
            return {
                "ok": False,
                "type": quick_type,
                "error": "on_hand_axis_degenerate",
                "detail": "on-hand axis points are coincident",
                "max_error_m": 0.0,
                "max_error_rad": math.pi,
                "max_error_deg": 180.0,
                "reference_plane_area_m2": float(reference_area),
            }
        observed_normal = axis / axis_length
        observed_area = None
    dot = float(np.clip(observed_normal @ reference_normal, -1.0, 1.0))
    error_rad = float(math.acos(float(np.clip(abs(dot), -1.0, 1.0))))
    return {
        "ok": bool(error_rad <= math.radians(orientation_tolerance)),
        "type": quick_type,
        "mode": "parallel",
        "error": None if error_rad <= math.radians(orientation_tolerance) else "orientation_constraint_not_satisfied",
        "max_error_m": 0.0,
        "max_error_rad": error_rad,
        "max_error_deg": float(math.degrees(error_rad)),
        "orientation_tolerance_deg": orientation_tolerance,
        "reference_normal_robot_base": reference_normal.astype(float).tolist(),
        "observed_normal_robot_base": observed_normal.astype(float).tolist(),
        "reference_plane_area_m2": None if reference_area is None else float(reference_area),
        "on_hand_plane_area_m2": None if observed_area is None else float(observed_area),
        "off_hand_points_robot_base_m": off.astype(float).tolist(),
        "on_hand_points_robot_base_m": on.astype(float).tolist(),
    }


@dataclass(frozen=True)
class _AffineForm:
    """One safe coordinate form: constant plus affine variable terms."""

    constant: float = 0.0
    coefficients: tuple[tuple[str, float], ...] = ()
    free: bool = False
    text: str | None = None

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


def _affine_form(raw: Any, *, label: str) -> _AffineForm:
    normalized = normalize_tracked_target_coordinate(raw, label=label)
    if isinstance(normalized, Mapping):
        if normalized.get("free") is True and set(normalized) == {"free"}:
            return _AffineForm(free=True, text="?")
        if set(normalized) == {"var"}:
            return _AffineForm(
                coefficients=((str(normalized["var"]), 1.0),),
                text=str(normalized["var"]),
            )
        if set(normalized) == {"expr"}:
            constant, coefficients, compact = parse_tracked_affine_expression(
                str(normalized["expr"]),
                label=label,
            )
            return _AffineForm(
                constant=float(constant),
                coefficients=tuple(
                    sorted(
                        (str(name), float(value))
                        for name, value in coefficients.items()
                        if abs(float(value)) > 1.0e-15
                    )
                ),
                text=compact,
            )
        raise ValueError(f"{label} has an unsupported coordinate form")
    return _AffineForm(constant=float(normalized), text=str(float(normalized)))


def _target_forms(
    target_points: Sequence[Mapping[str, Any]],
    point_names: Sequence[str],
) -> tuple[tuple[_AffineForm, ...], ...]:
    if any(not isinstance(name, str) for name in point_names):
        raise ValueError("point_names must contain only non-empty strings")
    names = [name.strip() for name in point_names]
    if len(names) < 1 or len(names) > MAX_TRACKED_POINTS:
        raise ValueError(f"point_names must contain one to {MAX_TRACKED_POINTS} names")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("point_names must be unique and non-empty")
    if not isinstance(target_points, (list, tuple)):
        raise ValueError("target_points must be an array")
    if len(target_points) != len(names):
        raise ValueError("target_points count does not match point_names")
    result: list[tuple[_AffineForm, ...]] = []
    for point_index, point in enumerate(target_points):
        if not isinstance(point, Mapping):
            raise ValueError(f"target_points[{point_index}] must be an object")
        raw_name = point.get("name")
        if not isinstance(raw_name, str):
            raise ValueError(
                f"target_points[{point_index}].name must be a non-empty string"
            )
        if raw_name.strip() != names[point_index]:
            raise ValueError("target_points order/names do not match point_names")
        raw_target = point.get("target_xyz_m")
        if isinstance(raw_target, Mapping):
            if set(raw_target) != {"x", "y", "z"}:
                raise ValueError(
                    f"target_points[{point_index}].target_xyz_m must contain x, y, z"
                )
            coordinates = [raw_target[axis] for axis in ("x", "y", "z")]
        elif isinstance(raw_target, (list, tuple)) and len(raw_target) == 3:
            coordinates = list(raw_target)
        else:
            raise ValueError(
                f"target_points[{point_index}].target_xyz_m must contain three coordinates"
            )
        result.append(
            tuple(
                _affine_form(
                    value,
                    label=(
                        f"target_points[{point_index}].target_xyz_m.{axis}"
                    ),
                )
                for axis, value in zip(("x", "y", "z"), coordinates)
            )
        )
    return tuple(result)


class _AffineTargetModel:
    """Local affine-coordinate model shared by the compact constraint API."""

    def __init__(
        self,
        forms: tuple[tuple[_AffineForm, ...], ...],
    ) -> None:
        self.point_count = len(forms)
        rows: list[tuple[int, int, _AffineForm]] = []
        free_rows: list[tuple[int, int]] = []
        variables: set[str] = set()
        for point_index, point_forms in enumerate(forms):
            if len(point_forms) != 3:
                raise ValueError("each target point must contain three coordinates")
            for axis_index, form in enumerate(point_forms):
                if form.free:
                    free_rows.append((point_index, axis_index))
                else:
                    rows.append((point_index, axis_index, form))
                    variables.update(name for name, _ in form.coefficients)
        self.rows = tuple(rows)
        self.free_rows = tuple(free_rows)
        self.variable_names = tuple(sorted(variables))
        self._variable_index = {
            name: index for index, name in enumerate(self.variable_names)
        }
        self.constants = np.asarray(
            [row[2].constant for row in self.rows],
            dtype=np.float64,
        )
        self.matrix = np.zeros(
            (len(self.rows), len(self.variable_names)),
            dtype=np.float64,
        )
        for row_index, (_point, _axis, form) in enumerate(self.rows):
            for name, coefficient in form.coefficients:
                self.matrix[row_index, self._variable_index[name]] = float(coefficient)
        self.rank = int(np.linalg.matrix_rank(self.matrix)) if self.matrix.size else 0
        self.residual_dimension = max(0, len(self.rows) - self.rank)
        self._relation_matrix = self._build_relation_matrix()

    def _build_relation_matrix(self) -> np.ndarray:
        row_count = len(self.rows)
        if not row_count:
            return np.zeros((0, 0), dtype=np.float64)
        fixed_indices = [
            index
            for index, (_point, _axis, form) in enumerate(self.rows)
            if not form.coefficients
        ]
        variable_indices = [
            index
            for index, (_point, _axis, form) in enumerate(self.rows)
            if form.coefficients
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

    def fit(
        self,
        points_robot_base_m: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit variables and return physical per-coordinate corrections.

        The correction vector is measured in metres for the corresponding
        submitted coordinates.  Returning it in this form keeps acceptance
        invariant under algebraically equivalent affine expressions whose
        coefficient rows have different scales.
        """
        points = _points(points_robot_base_m, label="constraint points")
        if points.shape[0] != self.point_count:
            raise ValueError("point count does not match target constraint model")
        actual = np.asarray(
            [points[point, axis] for point, axis, _ in self.rows],
            dtype=np.float64,
        )
        if not len(self.rows):
            values = np.empty((0,), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), values, points
        rhs = actual - self.constants
        if self.variable_names:
            values, _residuals, _rank, _singular = np.linalg.lstsq(
                self.matrix,
                rhs,
                rcond=None,
            )
            values = np.asarray(values, dtype=np.float64).reshape(-1)
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
            raise ValueError("affine target resolution produced a non-finite value")
        return coordinate_corrections, values, points

    def equation_residuals(
        self,
        points_robot_base_m: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return reduced affine equations for diagnostics, never acceptance."""
        points = _points(points_robot_base_m, label="constraint points")
        if points.shape[0] != self.point_count:
            raise ValueError("point count does not match target constraint model")
        if not self.rows or not self._relation_matrix.size:
            return np.empty((0,), dtype=np.float64)
        actual = np.asarray(
            [points[point, axis] for point, axis, _ in self.rows],
            dtype=np.float64,
        )
        residual = self._relation_matrix @ (actual - self.constants)
        if not np.all(np.isfinite(residual)):
            raise ValueError("affine equation diagnostics are non-finite")
        return np.asarray(residual, dtype=np.float64).reshape(-1)

    def variable_values(self, values: Sequence[float]) -> dict[str, float]:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != len(self.variable_names):
            raise ValueError("affine variable vector has an invalid size")
        return {
            name: float(vector[index])
            for index, name in enumerate(self.variable_names)
        }


@dataclass(frozen=True)
class TrackedPointConstraintSet:
    """Unified affine-coordinate and typed-relation evaluator for N markers."""

    point_names: tuple[str, ...]
    relations: tuple[Mapping[str, Any], ...] = ()
    # Optional coordinate expressions make this class useful to interface-free
    # benchmarks as well as relation-only live checks.  The motion planner
    # keeps its richer diagnostic model, while this compact API can still
    # recover the least-squares value of every repeated affine variable.
    target_points: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate the public point binding at construction time.

        Constraint reports are keyed by marker name.  Coercing arbitrary
        objects to strings here would make a malformed binding (for example,
        ``1`` and ``"1"``) silently alias the same marker and could let a
        reordered trajectory pass a later report.  Keep this compact API
        strict in the same way as the signed trajectory loader.
        """

        raw_names = self.point_names
        if isinstance(raw_names, (str, bytes)) or not isinstance(
            raw_names, (list, tuple)
        ):
            raise ValueError("point_names must be a list or tuple of strings")
        if any(name != name.strip() for name in raw_names if isinstance(name, str)):
            # Keep the public contract's backwards-compatible trimming, but
            # canonicalize once at construction so relation references and
            # signed metadata use the same marker spelling thereafter.
            names = tuple(name.strip() if isinstance(name, str) else name for name in raw_names)
        else:
            names = tuple(raw_names)
        if not 1 <= len(names) <= MAX_TRACKED_POINTS:
            raise ValueError(
                f"point_names must contain one to {MAX_TRACKED_POINTS} names"
            )
        for index, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"point_names[{index}] must be a non-empty string"
                )
            if len(name) > 128:
                raise ValueError(f"point_names[{index}] is too long")
            if any(ord(char) < 32 or ord(char) == 127 for char in name):
                raise ValueError(
                    f"point_names[{index}] contains a control character"
                )
        if len(set(names)) != len(names):
            raise ValueError("point_names must be unique")
        object.__setattr__(self, "point_names", names)

        if self.relations is None:
            object.__setattr__(self, "relations", ())
        elif isinstance(self.relations, (list, tuple)):
            object.__setattr__(
                self,
                "relations",
                tuple(_normalize_relation_records(self.relations, names)),
            )
        else:
            raise ValueError("relations must be a list or tuple")

        if self.target_points is None:
            object.__setattr__(self, "target_points", ())
        elif isinstance(self.target_points, (list, tuple)):
            object.__setattr__(self, "target_points", tuple(self.target_points))
        else:
            raise ValueError("target_points must be a list or tuple")

    def residual_blocks(self, points_robot_base_m: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        points = _points(points_robot_base_m, label="constraint points")
        names = [name.strip() for name in self.point_names]
        if len(names) != len(points):
            raise ValueError("point_names count does not match constraint points")
        blocks: list[dict[str, Any]] = []
        if self.target_points:
            forms = _target_forms(self.target_points, names)
            model = _AffineTargetModel(forms)
            residual, values, actual_points = model.fit(points)
            equation_residuals = model.equation_residuals(points)
            variable_values = model.variable_values(values)
            coordinate_errors: list[dict[str, Any]] = []
            for point_index, axis_index, form in model.rows:
                target = form.evaluate(variable_values)
                assert target is not None
                actual = float(actual_points[point_index, axis_index])
                coordinate_errors.append(
                    {
                        "point": names[point_index],
                        "axis": ("x", "y", "z")[axis_index],
                        "actual_m": actual,
                        "target_m": float(target),
                        "error_m": float(actual - target),
                        "expression": form.text,
                    }
                )
            blocks.append(
                {
                    "type": "affine_coordinates",
                    "kind": "position",
                    "point_names": names,
                    "residual": residual.astype(float),
                    "max_error": (
                        float(np.max(np.abs(residual))) if residual.size else 0.0
                    ),
                    "coordinate_errors": coordinate_errors,
                    "resolved_variables": variable_values,
                    "constraint_rank": int(model.rank),
                    "constraint_residual_dimension": int(
                        model.residual_dimension
                    ),
                    "equation_residual": equation_residuals.astype(float).tolist(),
                }
            )
        blocks.extend(
            relation_residual_blocks(
                points,
                self.relations,
                point_names=names,
                _canonical=True,
            )
        )
        return blocks

    def evaluate(
        self,
        points_robot_base_m: Sequence[Sequence[float]],
        *,
        position_tolerance_m: float = 0.012,
        orientation_tolerance_deg: float = 5.0,
    ) -> dict[str, Any]:
        points = _points(points_robot_base_m, label="constraint points")
        names = [name.strip() for name in self.point_names]
        if len(names) != len(points):
            raise ValueError("point_names count does not match constraint points")
        blocks = self.residual_blocks(points)
        try:
            position_tolerance = float(position_tolerance_m)
            orientation_tolerance = float(orientation_tolerance_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("constraint tolerances must be numeric") from exc
        if (
            not math.isfinite(position_tolerance)
            or position_tolerance <= 0.0
            or not math.isfinite(orientation_tolerance)
            or orientation_tolerance <= 0.0
        ):
            raise ValueError("constraint tolerances must be finite and positive")
        coordinate_blocks = [
            block for block in blocks if block.get("type") == "affine_coordinates"
        ]
        relation_blocks = [
            block for block in blocks if block.get("type") != "affine_coordinates"
        ]
        coordinate_errors = [
            float(block.get("max_error", 0.0)) for block in coordinate_blocks
        ]
        relation_position_errors = [
            float(block.get("max_error", 0.0))
            for block in relation_blocks
            if block.get("kind") == "position"
        ]
        relation_orientation_errors = [
            float(block.get("max_error", 0.0))
            for block in relation_blocks
            if block.get("kind") == "orientation"
        ]
        relation_orientation_errors.extend(
            float(block["orientation_error_rad"])
            for block in relation_blocks
            if block.get("orientation_error_rad") is not None
        )
        max_coordinate = max(coordinate_errors, default=0.0)
        max_relation_position = max(relation_position_errors, default=0.0)
        max_relation_orientation = max(relation_orientation_errors, default=0.0)
        reported_relation_blocks: list[dict[str, Any]] = []
        relation_position_checks: list[bool] = []
        collinear_errors: list[float] = []
        collinear_tolerances: list[float] = []
        for block in relation_blocks:
            reported = dict(block)
            if block.get("kind") == "position":
                effective_tolerance = relation_position_tolerance_m(
                    block.get("type"), position_tolerance
                )
                error = float(block.get("max_error", 0.0))
                within_tolerance = bool(error <= effective_tolerance)
                reported.update(
                    {
                        "requested_position_tolerance_m": position_tolerance,
                        "position_tolerance_m": effective_tolerance,
                        "within_position_tolerance": within_tolerance,
                    }
                )
                relation_position_checks.append(within_tolerance)
                if (
                    str(block.get("type") or "").strip().lower()
                    in COLLINEAR_RELATION_TYPES
                ):
                    collinear_errors.append(error)
                    collinear_tolerances.append(effective_tolerance)
            reported_relation_blocks.append(reported)
        relation_ok = bool(
            all(relation_position_checks)
            and max_relation_orientation <= math.radians(orientation_tolerance)
            and not any(bool(block.get("degenerate")) for block in relation_blocks)
        )
        resolved_variables: dict[str, float] = {}
        for block in coordinate_blocks:
            resolved_variables.update(
                {
                    str(name): float(value)
                    for name, value in (block.get("resolved_variables") or {}).items()
                }
            )
        max_position = max(max_coordinate, max_relation_position)
        return {
            "ok": bool(max_position <= position_tolerance and relation_ok),
            "point_count": int(len(points)),
            "point_names": names,
            "max_constraint_error_m": float(max_position),
            "max_coordinate_constraint_error_m": float(max_coordinate),
            "max_relation_error_m": float(max_relation_position),
            "max_collinear_error_m": (
                float(max(collinear_errors)) if collinear_errors else None
            ),
            "collinear_position_tolerance_m": (
                float(min(collinear_tolerances)) if collinear_tolerances else None
            ),
            "collinear_constraints_ok": bool(
                all(
                    error <= tolerance
                    for error, tolerance in zip(
                        collinear_errors, collinear_tolerances
                    )
                )
            ),
            "max_relation_error_rad": float(max_relation_orientation),
            "max_relation_error_deg": float(math.degrees(max_relation_orientation)),
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "affine_equation_residuals_m": [
                np.asarray(block.get("equation_residual", []), dtype=np.float64)
                .astype(float)
                .tolist()
                for block in coordinate_blocks
            ],
            "residual_blocks": [
                {
                    **{
                        key: value
                        for key, value in block.items()
                        if key != "residual"
                    },
                    "residual": np.asarray(
                        block["residual"], dtype=np.float64
                    ).astype(float).tolist(),
                }
                for block in coordinate_blocks + reported_relation_blocks
            ],
            "relations": [
                {
                    **{
                        key: value
                        for key, value in block.items()
                        if key != "residual"
                    },
                    "residual": np.asarray(
                        block["residual"], dtype=np.float64
                    ).astype(float).tolist(),
                }
                for block in reported_relation_blocks
            ],
            "resolved_variables": resolved_variables,
        }

    def resolve_variables(
        self,
        points_robot_base_m: Sequence[Sequence[float]],
        target_points: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Resolve affine coordinate variables from a measured point set.

        A variable that occurs only once is returned at the measured value; a
        repeated variable is the least-squares value shared by all of its
        occurrences.  Free ``?`` coordinates deliberately do not create a
        variable entry because they have no declared identity.
        """

        points = _points(points_robot_base_m, label="points")
        specs = list(self.target_points if target_points is None else target_points)
        if not specs:
            return {}
        names = list(self.point_names)
        if len(names) != len(points) or len(specs) != len(names):
            raise ValueError("point_names count does not match points")
        model = _AffineTargetModel(_target_forms(specs, names))
        _residual, values, _points_value = model.fit(points)
        return model.variable_values(values)

    def geometry_preflight(
        self,
        source_points_robot_base_m: Sequence[Sequence[float]],
        target_points_robot_base_m: Sequence[Sequence[float]] | None = None,
        *,
        tolerance_m: float = 0.012,
    ) -> dict[str, Any]:
        source = _points(source_points_robot_base_m, label="source points")
        if len(self.point_names) != len(source):
            raise ValueError("point_names count does not match source points")
        normalized_names = [name.strip() for name in self.point_names]
        if len(set(normalized_names)) != len(source) or any(
            not name for name in normalized_names
        ):
            raise ValueError("point_names must be unique and non-empty")
        if target_points_robot_base_m is not None:
            target = _points(target_points_robot_base_m, label="target points")
            if target.shape != source.shape:
                raise ValueError("source and target point arrays must have the same shape")
        return geometry_preflight(
            source,
            target_points_robot_base_m,
            tolerance_m=tolerance_m,
        )


__all__ = [
    "COLLINEAR_POSITION_TOLERANCE_M",
    "COLLINEAR_RELATION_TYPES",
    "GEOMETRY_DEGENERATE_AREA_M2",
    "GEOMETRY_DEGENERATE_VOLUME_M3",
    "GEOMETRY_RANK_TOLERANCE_M",
    "MAX_TRACKED_POINTS",
    "ORDERED_COLLINEAR_INNER_MARGIN_M",
    "ORDERED_COLLINEAR_LIVE_BOUNDARY_TOLERANCE_M",
    "TrackedPointConstraintSet",
    "evaluate_relations",
    "evaluate_quick_constraint",
    "geometry_preflight",
    "geometry_report_difference",
    "geometry_rank",
    "kabsch_rigid_transform",
    "pair_distance_matrix",
    "relation_residual_blocks",
    "relation_position_tolerance_m",
    "quick_constraint_relation",
    "resolve_fully_numeric_target_points",
    "rigid_geometry_report",
    "signed_volume_status",
    "tetrahedron_signed_volume_m3",
    "triangle_area_m2",
]
