"""Robot dimensions and public argument contracts for official_v2 tools."""

from __future__ import annotations

import ast
import itertools
import math
import re
from fractions import Fraction
from typing import Any, Mapping

from ...robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    ROBOT_PROFILE,
)
from .predefined_eef_points_local import (
    PREDEFINED_EEF_POINT_NAMES,
    is_predefined_eef_point_name,
)


READ_DEPTH_COORDINATE_MIN = 0.0
READ_DEPTH_COORDINATE_MAX = 1000.0
TRACK_OBJECT_DISTANCE_MAX_POINTS = 32
MOVE_POINT_TO_POINT_MAX_ABOVE_M = 0.75
MOVE_POINT_TO_POINT_MAX_STEPS = 2000
MOVE_POINT_TO_POINT_MAX_TIMEOUT_S = 600.0
MOVE_CHASSIS_SURFACE_POINT_COUNT = 3
MOVE_CHASSIS_SURFACE_MAX_TIMEOUT_S = 600.0
# One unified request may contain movable markers attached to the held object
# and frozen scene-reference markers.  Coordinate and preset constraints share
# this bound and may be combined in the same request.
MOVE_TRACKED_POINT_MAX_POINTS = 6
MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS = 6
MOVE_TRACKED_POINT_MAX_INEQUALITIES = 6
# A strict inequality has no closed boundary. Keep solved endpoints at least
# 1 mm inside the requested half-space so later RGB-D/proprio validation does
# not accept an equality-boundary pose as ``>`` or ``<`` due to numeric noise.
MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M = 0.001
MOVE_TRACKED_POINT_QUICK_MAX_POINTS = 6
MOVE_TRACKED_POINT_QUICK_COUNTS = {
    "touch": ((1,), (1,)),
    "flatwise": ((3,), (0,)),
    "plane_parallel": ((3,), (3,)),
    "collinear": ((2,), (1, 2)),
    "line_vertical_to_plane": ((2,), (3,)),
    "vertical_to_ground": ((2, 3), (0,)),
    # The triangle is an ordered, on-hand surface marker bundle.  Its
    # projected winding disambiguates the two signs of the camera-facing
    # normal; no off-hand registration is needed.
    "faceto": ((3,), (0,)),
    "reverse_faceto": ((3,), (0,)),
}
MOVE_TRACKED_POINT_ORDER_DESCRIPTION = (
    "Collinear point rows specify spatial axial order, not execution order. "
    "Preserve each group's exact row order, including interleaved on/off-hand points; "
    "1,3,2 means point 3 lies between 1 and 2. The canonical reading direction is "
    "robot-base left-to-right (-Y) for a Y-dominant line, top-to-bottom (-Z) for "
    "a Z-dominant line, and near-to-far (+X) for an X-dominant line; ties prefer "
    "Y, then Z, then X. It does not depend on camera viewpoint. This also applies "
    "when numeric/affine XYZ constraints imply three or more collinear points. "
    "Reversing rows requests the opposite physical order. Order violations cannot "
    "be returned as successful approximate or overlap-fallback plans."
)
MOVE_TRACKED_POINT_MAX_STEPS = 2000
MOVE_TRACKED_POINT_MAX_TIMEOUT_S = 600.0
CUT_OBJECT_POINT_NAMES = (
    "cutting_tool_point",
    "target_object_point",
)
CUT_OBJECT_MAX_STEPS = 2000
CUT_OBJECT_MAX_TIMEOUT_S = 600.0
OFFICIAL_SESSION_ID_MAX_LEN = 64
NAVIGATE_TO_NAME_MAX_LEN = 128
# The executor derives a conservative budget from the signed route length.
# Keep the public ceiling above the long-route floor so a slow official
# evaluator can finish without the HTTP wrapper cancelling a healthy run.
NAVIGATE_TO_MAX_TIMEOUT_S = 3600.0
NAVIGATE_TO_MIN_ARRIVAL_TOLERANCE_M = 0.05
NAVIGATE_TO_MAX_ARRIVAL_TOLERANCE_M = 1.0
_READ_DEPTH_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_READ_DEPTH_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TRACKED_TARGET_VARIABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TRACKED_TARGET_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
_TRACKED_TARGET_TERM_RE = re.compile(
    r"(?P<sign>[+-]?)(?:(?P<number>"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r")|(?P<variable>[A-Za-z][A-Za-z0-9_]{0,63}))"
)

# Coordinate expressions are declarative data.  Keep the grammar deliberately
# finite and parse it with ``ast`` only as a syntax tree; the tree is never
# passed to Python's evaluator.  The resulting form is affine, which is the
# complete class of coordinate relations that can be imposed on a rigid point
# pair without introducing an additional physical state variable.
_TRACKED_TARGET_EXPRESSION_MAX_LEN = 256
_TRACKED_TARGET_EXPRESSION_MAX_NODES = 96
_TRACKED_TARGET_EXPRESSION_MAX_DEPTH = 24


def _affine_expression_node(
    node: ast.AST,
    *,
    label: str,
    depth: int = 0,
) -> tuple[float, dict[str, float]]:
    """Return ``constant, coefficients`` for one safe affine AST node."""

    if depth > _TRACKED_TARGET_EXPRESSION_MAX_DEPTH:
        raise ValueError(f"{label} expression is too deeply nested")

    if isinstance(node, ast.Constant):
        # ``bool`` is an ``int`` subclass but is not a coordinate literal.
        if isinstance(node.value, bool) or not isinstance(
            node.value,
            (int, float),
        ):
            raise ValueError(
                f"{label} must use only finite numbers and variable names"
            )
        try:
            value = float(node.value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value, {}

    if isinstance(node, ast.Name):
        name = str(node.id)
        if not _TRACKED_TARGET_VARIABLE_RE.fullmatch(name):
            raise ValueError(
                f"{label} variable names must start with a letter and contain "
                "only letters, digits, or underscores"
            )
        return 0.0, {name: 1.0}

    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        constant, coefficients = _affine_expression_node(
            node.operand,
            label=label,
            depth=depth + 1,
        )
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return sign * constant, {
            name: sign * value for name, value in coefficients.items()
        }

    if not isinstance(node, ast.BinOp):
        raise ValueError(
            f"{label} must be a number, identifier, affine expression, or '?'"
        )

    left_constant, left_coefficients = _affine_expression_node(
        node.left,
        label=label,
        depth=depth + 1,
    )
    right_constant, right_coefficients = _affine_expression_node(
        node.right,
        label=label,
        depth=depth + 1,
    )

    if isinstance(node.op, ast.Add):
        sign = 1.0
    elif isinstance(node.op, ast.Sub):
        sign = -1.0
    else:
        sign = 0.0

    if sign:
        coefficients = dict(left_coefficients)
        for name, value in right_coefficients.items():
            coefficients[name] = coefficients.get(name, 0.0) + sign * value
        return (
            left_constant + sign * right_constant,
            coefficients,
        )

    if isinstance(node.op, ast.Mult):
        if left_coefficients and right_coefficients:
            raise ValueError(
                f"{label} must be affine; multiplication of two variables "
                "is not a rigid-coordinate relation"
            )
        if left_coefficients:
            scale = right_constant
            return (
                left_constant * scale,
                {name: value * scale for name, value in left_coefficients.items()},
            )
        scale = left_constant
        return (
            right_constant * scale,
            {name: value * scale for name, value in right_coefficients.items()},
        )

    if isinstance(node.op, ast.Div):
        if right_coefficients:
            raise ValueError(
                f"{label} must be affine; a variable denominator is not supported"
            )
        if abs(right_constant) <= 1e-15:
            raise ValueError(f"{label} divides by zero")
        scale = 1.0 / right_constant
        return (
            left_constant * scale,
            {name: value * scale for name, value in left_coefficients.items()},
        )

    if isinstance(node.op, ast.Pow):
        if right_coefficients:
            raise ValueError(
                f"{label} must use a numeric exponent"
            )
        exponent = right_constant
        if not math.isfinite(exponent):
            raise ValueError(f"{label} exponent must be finite")
        if left_coefficients:
            # x**1 and x**0 remain affine.  Any other power is nonlinear.
            if abs(exponent - 1.0) <= 1e-15:
                return left_constant, left_coefficients
            if abs(exponent) <= 1e-15:
                return 1.0, {}
            raise ValueError(
                f"{label} must be affine; a variable power other than 0 or 1 "
                "is not supported"
            )
        # Constant powers are not needed for coordinate relations, but a
        # bounded subset is harmless. Reject huge exponents before Python
        # attempts to construct an enormous intermediate integer.
        if abs(exponent) > 1024.0:
            raise ValueError(f"{label} constant exponent is too large")
        try:
            value = float(left_constant**exponent)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{label} contains an invalid power") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite power")
        return value, {}

    raise ValueError(
        f"{label} supports only +, -, *, /, **, parentheses, and variables"
    )


def parse_tracked_affine_expression(
    raw: str,
    *,
    label: str = "tracked coordinate",
) -> tuple[float, dict[str, float], str]:
    """Parse a user coordinate expression without executing user code.

    The returned tuple is ``(constant, coefficients, compact_text)`` for
    ``constant + sum(coefficients[name] * name)``. It is shared by the public
    contract and the local planner so the UI/API and IK cannot disagree about
    a relation. A separate ``?`` wildcard is handled by the caller. The
    accepted operators are the affine subset of ``+``, ``-``, scalar ``*``,
    scalar ``/``, and parentheses; nonlinear variable products and powers are
    rejected explicitly.
    """

    text = str(raw).strip()
    if not text:
        raise ValueError(f"{label} is empty")
    compact = re.sub(r"\s+", "", text)
    if not compact:
        raise ValueError(f"{label} is empty")
    if len(compact) > _TRACKED_TARGET_EXPRESSION_MAX_LEN:
        raise ValueError(
            f"{label} expression is too long (maximum "
            f"{_TRACKED_TARGET_EXPRESSION_MAX_LEN} characters)"
        )
    try:
        parsed = ast.parse(compact, mode="eval")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{label} must be a number, identifier, affine expression, or '?'"
        ) from exc
    nodes = list(ast.walk(parsed))
    if len(nodes) > _TRACKED_TARGET_EXPRESSION_MAX_NODES:
        raise ValueError(
            f"{label} expression contains too many terms "
            f"(maximum {_TRACKED_TARGET_EXPRESSION_MAX_NODES})"
        )
    constant, coefficients = _affine_expression_node(
        parsed.body,
        label=label,
    )
    coefficients = {
        str(name): float(value)
        for name, value in coefficients.items()
        if abs(float(value)) > 1e-15
    }
    if not math.isfinite(constant) or any(
        not math.isfinite(value) for value in coefficients.values()
    ):
        raise ValueError(f"{label} must contain only finite values")
    return float(constant), coefficients, compact


def normalize_tracked_inequalities(
    raw_inequalities: Any,
    *,
    allowed_variables: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate safe affine half-space constraints over coordinate variables.

    Public rows use ``{"lhs": "a-b", "op": ">", "rhs": 0}``. The left
    side follows the same non-executing affine grammar as point coordinates;
    the right side is deliberately a finite scalar so meaning and units stay
    unambiguous. When supplied, ``allowed_variables`` binds every identifier
    to a variable declared by a point coordinate.
    """

    if raw_inequalities is None:
        return []
    if not isinstance(raw_inequalities, (list, tuple)):
        raise ValueError("inequalities must be an array")
    if len(raw_inequalities) > MOVE_TRACKED_POINT_MAX_INEQUALITIES:
        raise ValueError(
            "inequalities cannot contain more than "
            f"{MOVE_TRACKED_POINT_MAX_INEQUALITIES} entries"
        )
    allowed = None if allowed_variables is None else set(allowed_variables)
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_inequalities):
        label = f"inequalities[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object")
        extra = sorted(set(raw) - {"lhs", "op", "rhs"})
        if extra:
            raise ValueError(
                f"{label} has unsupported fields: "
                + ", ".join(str(field) for field in extra)
            )
        missing = sorted({"lhs", "op", "rhs"} - set(raw))
        if missing:
            raise ValueError(
                f"{label} is missing fields: "
                + ", ".join(str(field) for field in missing)
            )
        lhs = raw.get("lhs")
        if not isinstance(lhs, str):
            raise ValueError(f"{label}.lhs must be an affine expression string")
        _constant, coefficients, compact = parse_tracked_affine_expression(
            lhs,
            label=f"{label}.lhs",
        )
        if not coefficients:
            raise ValueError(f"{label}.lhs must reference at least one variable")
        if allowed is not None:
            undeclared = sorted(set(coefficients) - allowed)
            if undeclared:
                raise ValueError(
                    f"{label}.lhs references undeclared coordinate variable(s): "
                    + ", ".join(undeclared)
                )
        op = raw.get("op")
        if not isinstance(op, str) or op.strip() not in {">", "<"}:
            raise ValueError(f"{label}.op must be > or <")
        raw_rhs = raw.get("rhs")
        if isinstance(raw_rhs, bool):
            raise ValueError(f"{label}.rhs must be numeric")
        try:
            rhs = float(raw_rhs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.rhs must be numeric") from exc
        if not math.isfinite(rhs):
            raise ValueError(f"{label}.rhs must be finite")
        result.append({"lhs": compact, "op": op.strip(), "rhs": rhs})
    return result


def _parse_tracked_target_text(raw: str, *, label: str) -> float | dict[str, Any]:
    """Normalize one tracked coordinate into a safe affine expression.

    A coordinate is either a finite number, a free variable, any affine
    expression of variables and constants (for example ``x+offset``,
    ``2*x``, or ``(x+y)/2``), or ``?`` for an explicitly unconstrained
    coordinate. The parser is deliberately small and never evaluates user
    text as Python.
    """

    text = str(raw).strip()
    if not text:
        raise ValueError(f"{label} must be a finite number, constraint, or '?'")
    if text == "?":
        return {"free": True}
    constant, coefficients, compact = parse_tracked_affine_expression(
        text,
        label=label,
    )
    if not coefficients:
        return float(constant)
    if len(coefficients) == 1 and abs(constant) <= 1e-15:
        name, coefficient = next(iter(coefficients.items()))
        if abs(coefficient - 1.0) <= 1e-15:
            return {"var": name}
    return {"expr": compact}


def _normalize_tracked_target_coordinate(raw: Any, *, label: str) -> float | dict[str, Any]:
    """Validate and normalize a numeric, variable, affine, or free coordinate."""

    if isinstance(raw, Mapping):
        keys = set(raw)
        if keys == {"var"}:
            raw_variable = raw.get("var")
            # JSON booleans are not coordinate identifiers.  Converting
            # ``true`` to the string ``True`` would otherwise make malformed
            # input look like a legitimate variable name and could disagree
            # with clients that preserve JSON types.
            if not isinstance(raw_variable, str):
                raise ValueError(f"{label}.var must be a string")
            variable = raw_variable.strip()
            if not _TRACKED_TARGET_VARIABLE_RE.fullmatch(variable):
                raise ValueError(
                    f"{label}.var must start with a letter and contain only "
                    "letters, digits, or underscores"
                )
            return {"var": variable}
        if keys == {"expr"}:
            expression = raw.get("expr")
            if not isinstance(expression, str):
                raise ValueError(f"{label}.expr must be a string")
            normalized = _parse_tracked_target_text(expression, label=f"{label}.expr")
            if isinstance(normalized, dict) and "free" in normalized:
                raise ValueError(f"{label}.expr cannot be the '?' wildcard")
            return normalized
        if keys == {"free"} and raw.get("free") is True:
            return {"free": True}
        raise ValueError(
            f"{label} must contain exactly var, expr, or free=true"
        )
    if raw is None or isinstance(raw, bool):
        raise ValueError(
            f"{label} must be a finite number, constraint, or '?'"
        )
    if isinstance(raw, str):
        return _parse_tracked_target_text(raw, label=label)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite number, constraint, or '?'"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def normalize_tracked_target_coordinate(
    raw: Any,
    *,
    label: str = "tracked coordinate",
) -> float | dict[str, Any]:
    """Public test-local normalizer shared by the planner and API contract."""

    return _normalize_tracked_target_coordinate(raw, label=label)


def coordinate_collinear_relations(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find maximal lines implied by affine coordinates, never by a sampled pose.

    A triple is guaranteed collinear exactly when every coefficient of its
    quadratic cross-product polynomial vanishes. Independent '?' coordinates
    receive distinct symbols. Rational arithmetic avoids tolerance-based guesses.
    """

    if len(points) < 3:
        return []
    forms = []
    for index, point in enumerate(points):
        raw = point.get("target_xyz_m") or {axis: "?" for axis in ("x", "y", "z")}
        values = [raw[axis] for axis in ("x", "y", "z")] if isinstance(raw, Mapping) else raw
        row = []
        for axis, value in enumerate(values):
            normalized = _normalize_tracked_target_coordinate(value, label="line coordinate")
            if isinstance(normalized, Mapping):
                if normalized.get("free"):
                    constant, coefficients = 0.0, {f"?{index}:{axis}": 1.0}
                elif "var" in normalized:
                    constant, coefficients = 0.0, {normalized["var"]: 1.0}
                else:
                    constant, coefficients, _ = parse_tracked_affine_expression(normalized["expr"])
            else:
                constant, coefficients = normalized, {}
            row.append({key: Fraction(str(val)) for key, val in {"": constant, **coefficients}.items() if val})
        forms.append(row)

    def difference(first, second):
        return [{key: a.get(key, 0) - b.get(key, 0) for key in a.keys() | b.keys()}
                for a, b in zip(first, second)]

    def cross_is_zero(a, b):
        for i, j in ((1, 2), (2, 0), (0, 1)):
            polynomial = {}
            for left, right, sign in ((a[i], b[j], 1), (a[j], b[i], -1)):
                for x, vx in left.items():
                    for y, vy in right.items():
                        key = tuple(sorted((x, y)))
                        polynomial[key] = polynomial.get(key, 0) + sign * vx * vy
            if any(polynomial.values()):
                return False
        return True

    triples = {
        triple: cross_is_zero(difference(forms[triple[1]], forms[triple[0]]),
                              difference(forms[triple[2]], forms[triple[0]]))
        for triple in itertools.combinations(range(len(points)), 3)
    }
    groups = []
    for size in range(len(points), 2, -1):
        for indices in itertools.combinations(range(len(points)), size):
            if any(set(indices) <= set(group) for group in groups):
                continue
            if all(triples[triple] for triple in itertools.combinations(indices, 3)):
                groups.append(indices)
    return [{"type": "ordered_collinear", "point_names": [points[i]["name"] for i in group]}
            for group in groups]


def _validate_official_session_id(session_id: Any) -> str:
    """先报超长，再报非法字符，避免 65+ 合法字符被误报成 unsupported characters。"""
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if len(sid) > OFFICIAL_SESSION_ID_MAX_LEN:
        raise ValueError(
            f"session_id is too long ({len(sid)} chars; official max is "
            f"{OFFICIAL_SESSION_ID_MAX_LEN})"
        )
    if not _READ_DEPTH_SESSION_RE.fullmatch(sid):
        raise ValueError("session_id contains unsupported characters")
    return sid


def validate_navigate_to_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate navigation to an existing named mark in the current map."""

    normalized = dict(args or {})
    allowed = {
        "name",
        "session_id",
        "timeout_s",
        "arrival_tolerance_m",
    }
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "unsupported arguments: "
            + ", ".join(str(key) for key in unexpected)
        )

    raw_name = normalized.get("name")
    if not isinstance(raw_name, str):
        raise ValueError("name must be a string")
    name = raw_name.strip()
    if not name:
        raise ValueError("name is required")
    if len(name) > NAVIGATE_TO_NAME_MAX_LEN:
        raise ValueError(
            f"name is too long ({len(name)} chars; official max is "
            f"{NAVIGATE_TO_NAME_MAX_LEN})"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError("name contains control characters")

    session_id = str(normalized.get("session_id") or "").strip()
    if session_id:
        session_id = _validate_official_session_id(session_id)

    def finite_number(key: str, default: float) -> float:
        raw = normalized.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    timeout_s = finite_number("timeout_s", 180.0)
    arrival_tolerance_m = finite_number("arrival_tolerance_m", 0.25)
    if not 0.1 <= timeout_s <= NAVIGATE_TO_MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_s must be in 0.1..{NAVIGATE_TO_MAX_TIMEOUT_S:g}s"
        )
    if not (
        NAVIGATE_TO_MIN_ARRIVAL_TOLERANCE_M
        <= arrival_tolerance_m
        <= NAVIGATE_TO_MAX_ARRIVAL_TOLERANCE_M
    ):
        raise ValueError(
            "arrival_tolerance_m must be in "
            f"{NAVIGATE_TO_MIN_ARRIVAL_TOLERANCE_M:g}.."
            f"{NAVIGATE_TO_MAX_ARRIVAL_TOLERANCE_M:g}m"
        )

    return {
        "name": name,
        "session_id": session_id,
        "timeout_s": timeout_s,
        "arrival_tolerance_m": arrival_tolerance_m,
    }


def validate_read_depth_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public frozen-capture depth lookup contract."""
    normalized = dict(args or {})
    session_id = _validate_official_session_id(normalized.get("session_id"))
    image_id = str(normalized.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("image_id is required")
    if not _READ_DEPTH_IMAGE_RE.fullmatch(image_id):
        raise ValueError("image_id contains unsupported characters")

    for key in ("u", "v"):
        raw = normalized.get(key)
        if raw is None or isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if not READ_DEPTH_COORDINATE_MIN <= value <= READ_DEPTH_COORDINATE_MAX:
            raise ValueError(f"{key} must be in 0..1000")
        normalized[key] = value

    normalized["session_id"] = session_id
    normalized["image_id"] = image_id
    return normalized


def validate_track_object_distance_args(
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate named points bound to one frozen head-camera capture."""

    normalized = dict(args or {})
    unexpected = sorted(
        set(normalized) - {"session_id", "image_id", "points"}
    )
    if unexpected:
        raise ValueError(
            "unsupported arguments: " + ", ".join(str(key) for key in unexpected)
        )
    session_id = _validate_official_session_id(normalized.get("session_id"))
    image_id = str(normalized.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("image_id is required")
    if not _READ_DEPTH_IMAGE_RE.fullmatch(image_id):
        raise ValueError("image_id contains unsupported characters")

    points = normalized.get("points")
    if not isinstance(points, list):
        raise ValueError("points must be a non-empty list")
    if not points:
        raise ValueError("points must contain at least one point")
    if len(points) > TRACK_OBJECT_DISTANCE_MAX_POINTS:
        raise ValueError(
            "points cannot contain more than "
            f"{TRACK_OBJECT_DISTANCE_MAX_POINTS} entries"
        )

    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError(f"points[{index}] must be an object")
        extra_fields = sorted(set(point) - {"name", "u", "v"})
        if extra_fields:
            raise ValueError(
                f"points[{index}] has unsupported fields: "
                + ", ".join(str(key) for key in extra_fields)
            )
        name = str(point.get("name") or "").strip()
        if not name:
            raise ValueError(f"points[{index}].name is required")
        if len(name) > 128:
            raise ValueError(f"points[{index}].name is too long")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError(f"points[{index}].name contains control characters")
        if name in names:
            raise ValueError(f"duplicate point name {name!r}")
        names.add(name)

        coordinates: dict[str, float] = {}
        for key in ("u", "v"):
            raw = point.get(key)
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"points[{index}].{key} must be numeric")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"points[{index}].{key} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"points[{index}].{key} must be finite")
            if not READ_DEPTH_COORDINATE_MIN <= value <= READ_DEPTH_COORDINATE_MAX:
                raise ValueError(f"points[{index}].{key} must be in 0..1000")
            coordinates[key] = value
        validated.append({"name": name, **coordinates})

    return {
        "session_id": session_id,
        "image_id": image_id,
        "points": validated,
    }


def validate_move_chassis_to_directly_facing_surface_args(
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen-head three-point surface-facing request."""

    normalized = dict(args or {})
    allowed = {
        "session_id",
        "image_id",
        "points",
        "nav_timeout_s",
        "pos_tol_m",
    }
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "unsupported arguments: "
            + ", ".join(str(key) for key in unexpected)
        )

    session_id = _validate_official_session_id(normalized.get("session_id"))
    image_id = str(normalized.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("image_id is required")
    if not _READ_DEPTH_IMAGE_RE.fullmatch(image_id):
        raise ValueError("image_id contains unsupported characters")

    raw_points = normalized.get("points")
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) != (
        MOVE_CHASSIS_SURFACE_POINT_COUNT
    ):
        raise ValueError(
            "points must contain exactly three surface points"
        )
    points: list[dict[str, float]] = []
    for index, point in enumerate(raw_points):
        if isinstance(point, Mapping):
            extra = sorted(set(point) - {"u", "v"})
            if extra:
                raise ValueError(
                    f"points[{index}] has unsupported fields: "
                    + ", ".join(str(key) for key in extra)
                )
            raw_uv = (point.get("u"), point.get("v"))
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            raw_uv = (point[0], point[1])
        else:
            raise ValueError(f"points[{index}] must contain u and v")
        coordinates: dict[str, float] = {}
        for key, raw in zip(("u", "v"), raw_uv):
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"points[{index}].{key} must be numeric")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"points[{index}].{key} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"points[{index}].{key} must be finite")
            if not READ_DEPTH_COORDINATE_MIN <= value <= READ_DEPTH_COORDINATE_MAX:
                raise ValueError(f"points[{index}].{key} must be in 0..1000")
            coordinates[key] = value
        points.append(coordinates)

    def finite_number(key: str, default: float) -> float:
        raw = normalized.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    nav_timeout_s = finite_number("nav_timeout_s", 120.0)
    pos_tol_m = finite_number("pos_tol_m", 0.04)
    if not 0.1 <= nav_timeout_s <= MOVE_CHASSIS_SURFACE_MAX_TIMEOUT_S:
        raise ValueError(
            "nav_timeout_s must be in "
            f"0.1..{MOVE_CHASSIS_SURFACE_MAX_TIMEOUT_S:g}s"
        )
    if not 0.004 <= pos_tol_m <= 0.20:
        raise ValueError("pos_tol_m must be in 0.004..0.20m")

    return {
        "session_id": session_id,
        "image_id": image_id,
        "points": points,
        "nav_timeout_s": nav_timeout_s,
        "pos_tol_m": pos_tol_m,
    }


def validate_move_point_to_point_args(
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen RGB-D point-transfer public contract."""

    normalized = dict(args or {})
    allowed = {
        "session_id",
        "image_id",
        "points",
        "above_target_point_m",
        "pos_tol",
        "ori_tol_deg",
        "max_steps",
        "timeout_s",
    }
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "unsupported arguments: "
            + ", ".join(str(key) for key in unexpected)
        )

    session_id = _validate_official_session_id(normalized.get("session_id"))
    image_id = str(normalized.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("image_id is required")
    if not _READ_DEPTH_IMAGE_RE.fullmatch(image_id):
        raise ValueError("image_id contains unsupported characters")

    points = normalized.get("points")
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        raise ValueError("points must contain exactly two points")
    validated_points: list[dict[str, float]] = []
    for index, point in enumerate(points):
        if isinstance(point, Mapping):
            extra_fields = sorted(set(point) - {"u", "v"})
            if extra_fields:
                raise ValueError(
                    f"points[{index}] has unsupported fields: "
                    + ", ".join(str(key) for key in extra_fields)
                )
            raw_coordinates = (point.get("u"), point.get("v"))
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            raw_coordinates = (point[0], point[1])
        else:
            raise ValueError(f"points[{index}] must contain u and v")
        coordinates: dict[str, float] = {}
        for key, raw in zip(("u", "v"), raw_coordinates):
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"points[{index}].{key} must be numeric")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"points[{index}].{key} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"points[{index}].{key} must be finite")
            if not READ_DEPTH_COORDINATE_MIN <= value <= READ_DEPTH_COORDINATE_MAX:
                raise ValueError(f"points[{index}].{key} must be in 0..1000")
            coordinates[key] = value
        validated_points.append(coordinates)

    def finite_number(key: str, default: float) -> float:
        raw = normalized.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    above = finite_number("above_target_point_m", 0.0)
    pos_tol = finite_number("pos_tol", 0.012)
    ori_tol_deg = finite_number("ori_tol_deg", 5.0)
    max_steps_value = finite_number("max_steps", 360.0)
    timeout_s = finite_number("timeout_s", 90.0)
    if not 0.0 <= above <= MOVE_POINT_TO_POINT_MAX_ABOVE_M:
        raise ValueError(
            "above_target_point_m must be in "
            f"0..{MOVE_POINT_TO_POINT_MAX_ABOVE_M}m"
        )
    if not 0.001 <= pos_tol <= 0.10:
        raise ValueError("pos_tol must be in 0.001..0.10m")
    if not 0.1 <= ori_tol_deg <= 45.0:
        raise ValueError("ori_tol_deg must be in 0.1..45 degrees")
    if (
        not max_steps_value.is_integer()
        or not 1 <= max_steps_value <= MOVE_POINT_TO_POINT_MAX_STEPS
    ):
        raise ValueError(
            "max_steps must be an integer in "
            f"1..{MOVE_POINT_TO_POINT_MAX_STEPS}"
        )
    if not 0.1 <= timeout_s <= MOVE_POINT_TO_POINT_MAX_TIMEOUT_S:
        raise ValueError(
            "timeout_s must be in "
            f"0.1..{MOVE_POINT_TO_POINT_MAX_TIMEOUT_S:g}s"
        )

    return {
        "session_id": session_id,
        "image_id": image_id,
        "points": validated_points,
        "above_target_point_m": above,
        "pos_tol": pos_tol,
        "ori_tol_deg": ori_tol_deg,
        "max_steps": int(max_steps_value),
        "timeout_s": timeout_s,
    }


def _finite_relation_vector(
    raw: Any,
    *,
    label: str,
    allow_zero: bool = False,
) -> list[float]:
    """Validate a relation direction/normal without importing geometry code."""

    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label} must be a length-3 numeric vector")
    values: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool):
            raise ValueError(f"{label}[{index}] must be numeric")
        try:
            value = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] must be finite")
        values.append(value)
    if not allow_zero and math.sqrt(sum(value * value for value in values)) <= 1.0e-12:
        raise ValueError(f"{label} must be non-zero")
    return values


def _validate_move_tracked_relations(
    raw_relations: Any,
    *,
    point_names: set[str],
) -> list[dict[str, Any]]:
    """Normalize the small, typed geometry-relation vocabulary.

    Relation data is deliberately declarative.  The local planner evaluates
    these blocks; no expression or callback supplied by a policy is executed.
    """

    if raw_relations is None:
        return []
    if not isinstance(raw_relations, (list, tuple)):
        raise ValueError("relations must be an array")
    if len(raw_relations) > 16:
        raise ValueError("relations cannot contain more than 16 entries")
    normalized: list[dict[str, Any]] = []
    missing_alias = object()

    def one_alias(
        raw: Mapping[str, Any],
        aliases: tuple[str, ...],
        *,
        label: str,
    ) -> Any:
        """Resolve one spelling and reject ambiguous duplicate spellings."""

        present = [name for name in aliases if name in raw]
        if len(present) > 1:
            raise ValueError(
                f"{label} must use only one of: " + ", ".join(aliases)
            )
        return raw[present[0]] if present else missing_alias

    for index, raw in enumerate(raw_relations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"relations[{index}] must be an object")
        relation_type = str(raw.get("type") or "").strip().lower()
        if relation_type == "coplanar":
            raise ValueError(
                "coplanar is not a motion constraint; use common_plane with a normal"
            )
        if relation_type not in {
            "common_plane",
            "oriented_plane_normal",
            "align_vector",
            "point_at_position",
            "line_through_point",
            "line_coincident",
            "line_segment_overlap",
            "line_segment_contains",
            "vertical_to_ground",
            "ordered_collinear",
        }:
            raise ValueError(
                f"relations[{index}].type must be common_plane, "
                "oriented_plane_normal, align_vector, point_at_position, "
                "line_through_point, line_coincident, line_segment_overlap, "
                "line_segment_contains, vertical_to_ground, or ordered_collinear"
            )
        names = raw.get("point_names")
        if not isinstance(names, (list, tuple)):
            raise ValueError(f"relations[{index}].point_names must be an array")
        # Marker bindings are identifiers, not arbitrary JSON values.  Do not
        # coerce an integer (or another object) to text here: doing so can
        # alias a malformed request with a legitimate marker and makes the
        # HTTP validator disagree with the signed-trajectory loader.
        if any(not isinstance(name, str) for name in names):
            raise ValueError(
                f"relations[{index}].point_names entries must be non-empty strings"
            )
        relation_names = [name.strip() for name in names]
        if len(set(relation_names)) != len(relation_names) or any(
            not name for name in relation_names
        ):
            raise ValueError(f"relations[{index}].point_names must be unique")
        if any(
            len(name) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            for name in relation_names
        ):
            raise ValueError(
                f"relations[{index}].point_names contains an invalid name"
            )
        missing = sorted(set(relation_names) - set(point_names))
        if missing:
            raise ValueError(
                f"relations[{index}] references unknown points: {', '.join(missing)}"
            )
        if relation_type == "common_plane" and not 3 <= len(relation_names) <= 4:
            raise ValueError("common_plane requires 3 or 4 point names")
        if relation_type == "oriented_plane_normal" and len(relation_names) != 3:
            raise ValueError(
                "oriented_plane_normal requires exactly 3 points (ordered)"
            )
        if relation_type == "align_vector" and len(relation_names) != 2:
            raise ValueError("align_vector requires exactly 2 ordered points")
        if relation_type == "vertical_to_ground" and len(relation_names) not in {2, 3}:
            raise ValueError("vertical_to_ground requires 2 or 3 points")
        if relation_type == "ordered_collinear" and not 2 <= len(relation_names) <= 6:
            raise ValueError("ordered_collinear requires 2 to 6 ordered points")
        if relation_type == "point_at_position" and len(relation_names) != 1:
            raise ValueError("point_at_position requires exactly 1 point")
        if relation_type in {
            "line_through_point",
            "line_coincident",
            "line_segment_overlap",
            "line_segment_contains",
        } and len(relation_names) != 2:
            raise ValueError(f"{relation_type} requires exactly 2 ordered points")

        allowed_relation_fields = {"type", "point_names"}
        if relation_type == "common_plane":
            allowed_relation_fields.update(
                {"normal_robot_base", "normal", "offset_m", "offset"}
            )
        elif relation_type == "oriented_plane_normal":
            allowed_relation_fields.update(
                {
                    "normal_robot_base",
                    "target_normal_robot_base",
                    "normal",
                    "mode",
                    "sense",
                }
            )
        elif relation_type == "align_vector":
            allowed_relation_fields.update(
                {
                    "direction_robot_base",
                    "target_direction_robot_base",
                    "direction",
                    "mode",
                    "sense",
                }
            )
        elif relation_type == "point_at_position":
            allowed_relation_fields.update(
                {"target_position_robot_base_m", "target_position", "position"}
            )
        elif relation_type == "line_through_point":
            allowed_relation_fields.update(
                {"target_point_robot_base_m", "target_point", "point"}
            )
        elif relation_type == "line_coincident":
            allowed_relation_fields.update(
                {
                    "line_point_robot_base_m",
                    "line_point",
                    "point",
                    "line_direction_robot_base",
                    "line_direction",
                    "direction",
                }
            )
        elif relation_type in {"line_segment_overlap", "line_segment_contains"}:
            allowed_relation_fields.update(
                {
                    "segment_start_robot_base_m",
                    "segment_start",
                    "start",
                    "segment_end_robot_base_m",
                    "segment_end",
                    "end",
                }
            )
        unexpected_relation_fields = sorted(
            set(raw) - allowed_relation_fields
        )
        if unexpected_relation_fields:
            raise ValueError(
                f"relations[{index}] has unsupported fields: "
                + ", ".join(str(field) for field in unexpected_relation_fields)
            )

        if relation_type in {"vertical_to_ground", "ordered_collinear"}:
            normalized.append({"type": relation_type, "point_names": relation_names})
            continue

        if relation_type == "common_plane":
            vector_raw = one_alias(
                raw,
                ("normal_robot_base", "normal"),
                label=f"relations[{index}].normal_robot_base",
            )
            normal = _finite_relation_vector(
                vector_raw,
                label=f"relations[{index}].normal_robot_base",
            )
            offset_raw = one_alias(
                raw,
                ("offset_m", "offset"),
                label=f"relations[{index}].offset_m",
            )
            if offset_raw is missing_alias:
                offset_raw = {"free": True}
            if isinstance(offset_raw, Mapping):
                if set(offset_raw) != {"free"} or offset_raw.get("free") is not True:
                    raise ValueError(
                        f"relations[{index}].offset_m must be a finite number or {{'free': true}}"
                    )
                offset: float | dict[str, bool] = {"free": True}
            else:
                if isinstance(offset_raw, bool):
                    raise ValueError(f"relations[{index}].offset_m must be numeric")
                try:
                    offset_value = float(offset_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"relations[{index}].offset_m must be numeric or free"
                    ) from exc
                if not math.isfinite(offset_value):
                    raise ValueError(f"relations[{index}].offset_m must be finite")
                offset = offset_value
            normalized.append(
                {
                    "type": relation_type,
                    "point_names": relation_names,
                    "normal_robot_base": normal,
                    "offset_m": offset,
                }
            )
            continue

        if relation_type in {
            "point_at_position", "line_through_point", "line_coincident"
        }:
            if relation_type == "point_at_position":
                point_raw = one_alias(
                    raw,
                    ("target_position_robot_base_m", "target_position", "position"),
                    label=f"relations[{index}].target_position_robot_base_m",
                )
                point_key = "target_position_robot_base_m"
            elif relation_type == "line_through_point":
                point_raw = one_alias(
                    raw,
                    ("target_point_robot_base_m", "target_point", "point"),
                    label=f"relations[{index}].target_point_robot_base_m",
                )
                point_key = "target_point_robot_base_m"
            else:
                point_raw = one_alias(
                    raw,
                    ("line_point_robot_base_m", "line_point", "point"),
                    label=f"relations[{index}].line_point_robot_base_m",
                )
                point_key = "line_point_robot_base_m"
            point_value = _finite_relation_vector(
                point_raw,
                label=f"relations[{index}].{point_key}",
                allow_zero=True,
            )
            record = {
                "type": relation_type,
                "point_names": relation_names,
                point_key: point_value,
            }
            if relation_type == "line_coincident":
                direction_raw = one_alias(
                    raw,
                    (
                        "line_direction_robot_base", "line_direction", "direction"
                    ),
                    label=f"relations[{index}].line_direction_robot_base",
                )
                record["line_direction_robot_base"] = _finite_relation_vector(
                    direction_raw,
                    label=f"relations[{index}].line_direction_robot_base",
                )
            normalized.append(record)
            continue

        if relation_type in {"line_segment_overlap", "line_segment_contains"}:
            start_raw = one_alias(
                raw,
                ("segment_start_robot_base_m", "segment_start", "start"),
                label=f"relations[{index}].segment_start_robot_base_m",
            )
            end_raw = one_alias(
                raw,
                ("segment_end_robot_base_m", "segment_end", "end"),
                label=f"relations[{index}].segment_end_robot_base_m",
            )
            start = _finite_relation_vector(
                start_raw,
                label=f"relations[{index}].segment_start_robot_base_m",
                allow_zero=True,
            )
            end = _finite_relation_vector(
                end_raw,
                label=f"relations[{index}].segment_end_robot_base_m",
                allow_zero=True,
            )
            if math.sqrt(
                sum(
                    (end_value - start_value) ** 2
                    for start_value, end_value in zip(start, end)
                )
            ) <= 1.0e-12:
                raise ValueError(
                    f"relations[{index}] target segment endpoints must be distinct"
                )
            normalized.append(
                {
                    "type": relation_type,
                    "point_names": relation_names,
                    "segment_start_robot_base_m": start,
                    "segment_end_robot_base_m": end,
                }
            )
            continue

        vector_aliases = (
            ("normal_robot_base", "target_normal_robot_base", "normal")
            if relation_type == "oriented_plane_normal"
            else (
                "direction_robot_base",
                "target_direction_robot_base",
                "direction",
            )
        )
        vector_raw = one_alias(
            raw,
            vector_aliases,
            label=(
                f"relations[{index}].normal_robot_base"
                if relation_type == "oriented_plane_normal"
                else f"relations[{index}].direction_robot_base"
            ),
        )
        vector_label = (
            "normal_robot_base"
            if relation_type == "oriented_plane_normal"
            else "direction_robot_base"
        )
        vector = _finite_relation_vector(
            vector_raw,
            label=f"relations[{index}].{vector_label}",
        )
        mode_raw = one_alias(
            raw,
            ("mode", "sense"),
            label=f"relations[{index}].mode",
        )
        mode = str(
            "same" if mode_raw is missing_alias else mode_raw
        ).strip().lower()
        mode_aliases = {
            "same_direction": "same",
            "same": "same",
            "aligned": "same",
            "opposite_direction": "opposite",
            "opposite": "opposite",
            "either": "parallel",
            "parallel": "parallel",
            "undirected": "parallel",
        }
        mode = mode_aliases.get(mode, mode)
        if mode not in {"same", "opposite", "parallel"}:
            raise ValueError(
                f"relations[{index}].mode must be same, opposite, or parallel"
            )
        normalized.append(
            {
                "type": relation_type,
                "point_names": relation_names,
                (
                    "normal_robot_base"
                    if relation_type == "oriented_plane_normal"
                    else "direction_robot_base"
                ): vector,
                "mode": mode,
            }
        )
    return normalized


# Quick constraints are intentionally a small, typed vocabulary.  They add a
# geometry residual to the same affine-coordinate solve; they never replace or
# discard coordinates supplied on the point rows.
_MOVE_TRACKED_QUICK_TYPES = {
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
_MOVE_TRACKED_QUICK_ROLES = {
    "on_hand": "on_hand",
    "onhand": "on_hand",
    "on-hand": "on_hand",
    "controlled": "on_hand",
    "held": "on_hand",
    "off_hand": "off_hand",
    "offhand": "off_hand",
    "off-hand": "off_hand",
    "fixed": "off_hand",
    "reference": "off_hand",
}


def _validate_move_tracked_quick_constraint(
    raw_constraint: Any,
    *,
    point_names: set[str],
    point_roles: Mapping[str, str],
    label: str = "quick_constraint",
    require_all_points: bool = True,
) -> dict[str, Any]:
    """Validate and canonicalize one role-based geometry preset.

    ``on_hand_points`` are attached to the selected EEF and become the only
    variables in the local pose solve.  ``off_hand_points`` are read from the
    same frozen tracker observation and are checked for drift during
    execution; they never become movable targets.
    """

    if not isinstance(raw_constraint, Mapping):
        raise ValueError(f"{label} must be an object")
    allowed = {
        "type",
        "on_hand_points",
        "off_hand_points",
        "axial_point_order",
        "axial_mode",
        "normal_mode",
        "direction_mode",
        "mode",
        "sense",
    }
    unexpected = sorted(set(raw_constraint) - allowed)
    if unexpected:
        raise ValueError(
            f"{label} has unsupported fields: "
            + ", ".join(str(item) for item in unexpected)
        )
    raw_type = str(raw_constraint.get("type") or "").strip().lower()
    quick_type = _MOVE_TRACKED_QUICK_TYPES.get(raw_type)
    if quick_type is None:
        raise ValueError(
            f"{label}.type must be touch, flatwise, plane_parallel, "
            "collinear, line_vertical_to_plane, vertical_to_ground, faceto, "
            "or reverse_faceto"
        )

    def names_field(key: str, role: str) -> list[str]:
        # Explicit arrays are canonical on the wire, but compact callers may
        # omit them when every submitted point has an explicit role.  Derive
        # the names in submission order so the resulting normal direction is
        # deterministic and still fully bound to the request.
        if key not in raw_constraint or raw_constraint.get(key) is None:
            return [
                name
                for name, point_role in point_roles.items()
                if point_role == role
            ]
        raw_names = raw_constraint.get(key)
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError(f"{label}.{key} must be an array")
        result: list[str] = []
        for index, raw_name in enumerate(raw_names):
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    f"{label}.{key}[{index}] must be a non-empty string"
                )
            name = raw_name.strip()
            if name not in point_names:
                raise ValueError(
                    f"{label}.{key} references unknown point {name!r}"
                )
            result.append(name)
        if len(set(result)) != len(result):
            raise ValueError(f"{label}.{key} must contain unique names")
        return result

    on_names = names_field("on_hand_points", "on_hand")
    off_names = names_field("off_hand_points", "off_hand")
    if set(on_names) & set(off_names):
        raise ValueError(
            f"{label} on_hand_points and off_hand_points must be disjoint"
        )
    expected_on, expected_off = MOVE_TRACKED_POINT_QUICK_COUNTS[quick_type]
    if len(on_names) not in expected_on or len(off_names) not in expected_off:
        on_text = " or ".join(str(value) for value in sorted(expected_on))
        off_text = " or ".join(str(value) for value in sorted(expected_off))
        raise ValueError(
            f"{quick_type} requires exactly {on_text} on-hand and "
            f"{off_text} off-hand points"
        )
    if require_all_points and set(on_names) | set(off_names) != set(point_names):
        raise ValueError(
            f"{label} point lists must contain every submitted point exactly once"
        )
    for name in on_names:
        if point_roles.get(name) != "on_hand":
            raise ValueError(
                f"point {name!r} must have role on_hand in {label}"
            )
    for name in off_names:
        if point_roles.get(name) != "off_hand":
            raise ValueError(
                f"point {name!r} must have role off_hand in {label}"
            )

    # The camera-facing presets use the submitted 1,2,3 order twice: it is
    # the winding order in the frozen head image and the corresponding signed
    # triangle normal.  Do not allow an explicit name list to silently reorder
    # the UI rows, otherwise a caller could satisfy the normal relation while
    # violating the requested clockwise/counterclockwise semantics.
    if require_all_points and quick_type in {"faceto", "reverse_faceto"}:
        submitted_on_names = [
            name for name, point_role in point_roles.items()
            if point_role == "on_hand"
        ]
        if on_names != submitted_on_names or off_names:
            raise ValueError(
                f"{quick_type} on_hand_points must preserve the three submitted "
                "on-hand rows and cannot include off-hand points"
            )

    # Plane/vector presets are deliberately unoriented.  Legacy callers may
    # still send an explicit undirected spelling, but same/opposite is rejected.
    # Four-point ordered collinear is the exception: its row order defines the
    # directed axial sequence and is validated below.
    present = [
        key
        for key in ("normal_mode", "direction_mode", "mode", "sense")
        if key in raw_constraint
    ]
    if len(present) > 1:
        raise ValueError(
            f"{label} orientation mode must use only one field"
        )
    mode = str(raw_constraint[present[0]] if present else "parallel").strip().lower()
    if mode not in {"either", "parallel", "undirected"}:
        raise ValueError(
            "quick constraints are unoriented; use parallel/either, not same/opposite"
        )
    result = {
        "type": quick_type,
        "on_hand_points": on_names,
        "off_hand_points": off_names,
    }
    if "axial_mode" in raw_constraint and quick_type != "collinear":
        raise ValueError(f"{label}.axial_mode is only valid for collinear")
    if quick_type == "collinear":
        submitted_names = list(point_roles)
        submitted_on_names = [
            name for name in submitted_names if point_roles[name] == "on_hand"
        ]
        submitted_off_names = [
            name for name in submitted_names if point_roles[name] == "off_hand"
        ]
        if require_all_points and (
            on_names != submitted_on_names or off_names != submitted_off_names
        ):
            raise ValueError(
                "collinear on_hand_points and off_hand_points must preserve "
                "their point-row order"
            )
        raw_axial_order = raw_constraint.get("axial_point_order")
        if raw_axial_order is not None:
            if not isinstance(raw_axial_order, (list, tuple)):
                raise ValueError(
                    f"{label}.axial_point_order must be an array"
                )
            supplied_axial_order = [str(name).strip() for name in raw_axial_order]
            if (len(supplied_axial_order) != len(on_names) + len(off_names)
                    or set(supplied_axial_order) != set(on_names) | set(off_names)
                    or (require_all_points and supplied_axial_order != submitted_names)):
                raise ValueError(
                    f"{label}.axial_point_order must exactly match the "
                    "constraint group's axial point order"
                )
        axial_mode = str(
            raw_constraint.get(
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
                f"{label}.axial_mode must be ordered, ordered_containment, "
                "segment_overlap, or line_only"
            )
        if len(off_names) != 2 and axial_mode not in {"line_only", "ordered"}:
            raise ValueError(
                f"{label}.axial_mode {axial_mode} requires two off-hand points"
            )
        if axial_mode == "ordered_containment":
            required_order = [
                on_names[0], off_names[0], off_names[1], on_names[1]
            ]
            actual_order = (
                submitted_names
                if require_all_points
                else (
                    [str(name).strip() for name in raw_axial_order]
                    if raw_axial_order is not None
                    else required_order
                )
            )
            if actual_order != required_order:
                raise ValueError(
                    "four-point collinear ordered_containment requires point rows "
                    "in axial order on_hand[0] -> off_hand[0] -> off_hand[1] -> "
                    "on_hand[1] (for example 1, 3, 4, 2)"
                )
        result["axial_mode"] = axial_mode
        result["axial_point_order"] = (
            supplied_axial_order if raw_axial_order is not None else
            [name for name in submitted_names if name in set(on_names) | set(off_names)]
        )
    return result


def validate_move_tracked_point_args(
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one unified 1..6 point coordinate/preset request."""

    normalized = dict(args or {})
    allowed = {
        "points", "relations", "inequalities", "mode", "quick_constraint", "quick_constraints",
        "execution_mode",
        "pos_tol", "ori_tol_deg", "max_steps", "timeout_s",
    }
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "unsupported arguments: " + ", ".join(str(key) for key in unexpected)
        )

    missing = object()
    quick_raw = normalized.get("quick_constraint", missing)
    if quick_raw is None:
        quick_raw = missing
    quick_groups_raw = normalized.get("quick_constraints", missing)
    if quick_groups_raw is None:
        quick_groups_raw = missing
    # An explicitly serialized empty plural field carries the same meaning as
    # an omitted optional field.  The UI omits it, while replay/MCP clients
    # often emit ``quick_constraints: []``; accepting both forms keeps the
    # official and development routes semantically identical.  Non-empty
    # arrays still take the strict named-group validation path below.
    if (
        quick_groups_raw is not missing
        and isinstance(quick_groups_raw, (list, tuple))
        and not quick_groups_raw
    ):
        quick_groups_raw = missing
    if quick_raw is not missing and quick_groups_raw is not missing:
        raise ValueError("quick_constraint and quick_constraints are mutually exclusive")
    mode_raw = normalized.get("mode", missing)
    mode_text = "" if mode_raw in (missing, None) else str(mode_raw).strip().lower()
    concrete_mode = _MOVE_TRACKED_QUICK_TYPES.get(mode_text)
    if quick_raw is missing and quick_groups_raw is missing and concrete_mode is not None:
        quick_raw = {"type": mode_text}
    has_quick = quick_raw is not missing or quick_groups_raw is not missing
    if mode_text in {"quick", "quick_constraint", "preset"} and not has_quick:
        raise ValueError("mode quick_constraint requires quick_constraint or quick_constraints")
    if has_quick and mode_text in {
        "coordinate", "coordinates", "legacy", "none"
    }:
        raise ValueError("quick constraints conflict with coordinate-only mode")
    allowed_modes = {
        "", "coordinate", "coordinates", "legacy", "none", "quick",
        "quick_constraint", "preset", *_MOVE_TRACKED_QUICK_TYPES,
    }
    if mode_text not in allowed_modes:
        raise ValueError("mode must be coordinate or a supported quick constraint")

    execution_mode_raw = normalized.get("execution_mode", "exec")
    if not isinstance(execution_mode_raw, str):
        raise ValueError("execution_mode must be exec or plan")
    execution_mode = execution_mode_raw.strip().lower()
    if execution_mode not in {"exec", "plan"}:
        raise ValueError("execution_mode must be exec or plan")

    points = normalized.get("points")
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError(
            f"points must contain one to {MOVE_TRACKED_POINT_MAX_POINTS} tracked points"
        )
    if len(points) > MOVE_TRACKED_POINT_MAX_POINTS:
        raise ValueError(
            f"points cannot contain more than {MOVE_TRACKED_POINT_MAX_POINTS} entries"
        )

    names: set[str] = set()
    roles: dict[str, str] = {}
    validated_points: list[dict[str, Any]] = []
    axes = ("x", "y", "z")
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError(f"points[{index}] must be an object")
        extra_fields = sorted(set(point) - {"name", "role", "target_xyz_m"})
        if extra_fields:
            raise ValueError(
                f"points[{index}] has unsupported fields: "
                + ", ".join(str(key) for key in extra_fields)
            )
        raw_name = point.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"points[{index}].name must be a non-empty string")
        name = raw_name.strip()
        if len(name) > 128:
            raise ValueError(f"points[{index}].name is too long")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError(f"points[{index}].name contains control characters")
        if name in names:
            raise ValueError(f"duplicate point name {name!r}")

        raw_role = point.get("role", "on_hand")
        if not isinstance(raw_role, str):
            raise ValueError(f"points[{index}].role must be on_hand or off_hand")
        role = _MOVE_TRACKED_QUICK_ROLES.get(raw_role.strip().lower())
        if role is None:
            raise ValueError(f"points[{index}].role must be on_hand or off_hand")
        if role == "off_hand" and is_predefined_eef_point_name(name):
            raise ValueError(
                f"points[{index}].name {name!r} is predefined EEF geometry and "
                "must use role on_hand"
            )

        # Preserve the canonical representation used by legacy coordinate-only
        # callers.  ``role`` is emitted when the caller supplied it; omitted
        # roles still mean ``on_hand`` everywhere in the unified implementation.
        canonical_point: dict[str, Any] = {"name": name}
        if "role" in point:
            canonical_point["role"] = role
        if "target_xyz_m" in point and point.get("target_xyz_m") is not None:
            target = point.get("target_xyz_m")
            if isinstance(target, Mapping):
                if set(target) != set(axes):
                    raise ValueError(
                        f"points[{index}].target_xyz_m must contain exactly x, y, z"
                    )
                raw_coordinates = [target[axis] for axis in axes]
            elif isinstance(target, (list, tuple)) and len(target) == 3:
                raw_coordinates = list(target)
            else:
                raise ValueError(
                    f"points[{index}].target_xyz_m must be an xyz object or length-3 list"
                )
            canonical_point["target_xyz_m"] = {
                axis: _normalize_tracked_target_coordinate(
                    raw,
                    label=f"points[{index}].target_xyz_m.{axis}",
                )
                for axis, raw in zip(axes, raw_coordinates)
            }
        names.add(name)
        roles[name] = role
        validated_points.append(canonical_point)

    if not any(role == "on_hand" for role in roles.values()):
        raise ValueError("move_tracked_point requires at least one on-hand point")

    coordinate_variables: set[str] = set()
    for point_index, point in enumerate(validated_points):
        target = point.get("target_xyz_m")
        if not isinstance(target, Mapping):
            continue
        for axis in axes:
            coordinate = target[axis]
            if not isinstance(coordinate, Mapping):
                continue
            if set(coordinate) == {"var"}:
                coordinate_variables.add(str(coordinate["var"]))
            elif set(coordinate) == {"expr"}:
                _constant, coefficients, _compact = parse_tracked_affine_expression(
                    str(coordinate["expr"]),
                    label=f"points[{point_index}].target_xyz_m.{axis}",
                )
                coordinate_variables.update(coefficients)
    inequalities = normalize_tracked_inequalities(
        normalized.get("inequalities", []),
        allowed_variables=coordinate_variables,
    )

    relations = _validate_move_tracked_relations(
        normalized.get("relations", []), point_names=names
    )
    for inferred in coordinate_collinear_relations(validated_points):
        if inferred not in relations:
            relations.append(inferred)
    quick = None
    quick_groups: list[dict[str, Any]] | None = None
    if quick_raw is not missing:
        quick = _validate_move_tracked_quick_constraint(
            quick_raw, point_names=names, point_roles=roles
        )
    elif quick_groups_raw is not missing:
        if not isinstance(quick_groups_raw, (list, tuple)) or not quick_groups_raw:
            raise ValueError("quick_constraints must be a non-empty array")
        if len(quick_groups_raw) > MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS:
            raise ValueError(
                "quick_constraints cannot contain more than "
                f"{MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS} entries"
            )
        quick_groups = [
            _validate_move_tracked_quick_constraint(
                raw_group,
                point_names=names,
                point_roles=roles,
                label=f"quick_constraints[{index}]",
                require_all_points=False,
            )
            for index, raw_group in enumerate(quick_groups_raw)
        ]
    if concrete_mode is not None:
        candidate_types = (
            [quick["type"]]
            if quick is not None
            else [group["type"] for group in (quick_groups or [])]
        )
        if any(group_type != concrete_mode for group_type in candidate_types):
            raise ValueError("mode preset conflicts with quick constraint type")

    def finite_number(key: str, default: float) -> float:
        raw = normalized.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    pos_tol = finite_number("pos_tol", 0.03)
    ori_tol_deg = finite_number("ori_tol_deg", 20.0)
    max_steps_value = finite_number("max_steps", 360.0)
    timeout_s = finite_number("timeout_s", 90.0)
    if not 0.001 <= pos_tol <= 0.10:
        raise ValueError("pos_tol must be in 0.001..0.10m")
    if not 0.1 <= ori_tol_deg <= 45.0:
        raise ValueError("ori_tol_deg must be in 0.1..45 degrees")
    if (
        not max_steps_value.is_integer()
        or not 1 <= max_steps_value <= MOVE_TRACKED_POINT_MAX_STEPS
    ):
        raise ValueError(
            "max_steps must be an integer in "
            f"1..{MOVE_TRACKED_POINT_MAX_STEPS}"
        )
    if not 0.1 <= timeout_s <= MOVE_TRACKED_POINT_MAX_TIMEOUT_S:
        raise ValueError(
            "timeout_s must be in "
            f"0.1..{MOVE_TRACKED_POINT_MAX_TIMEOUT_S:g}s"
        )
    result = {
        "points": validated_points,
        "relations": relations,
        "inequalities": inequalities,
        "execution_mode": execution_mode,
        "pos_tol": pos_tol,
        "ori_tol_deg": ori_tol_deg,
        "max_steps": int(max_steps_value),
        "timeout_s": timeout_s,
    }
    if quick is not None:
        result["mode"] = "quick_constraint"
        result["quick_constraint"] = quick
    elif quick_groups is not None:
        result["mode"] = "quick_constraint"
        result["quick_constraints"] = quick_groups
    return result


def validate_cut_object_args(
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the live tracked two-point cutting-touch contract."""

    normalized = dict(args or {})
    allowed = {
        "session_id",
        "image_id",
        "points",
        "pos_tol",
        "ori_tol_deg",
        "max_steps",
        "timeout_s",
    }
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "unsupported arguments: "
            + ", ".join(str(key) for key in unexpected)
        )

    tracked = validate_track_object_distance_args(
        {
            "session_id": normalized.get("session_id"),
            "image_id": normalized.get("image_id"),
            "points": normalized.get("points"),
        }
    )
    if len(tracked["points"]) != 2:
        raise ValueError("points must contain exactly two named points")
    by_name = {point["name"]: point for point in tracked["points"]}
    required_names = set(CUT_OBJECT_POINT_NAMES)
    if set(by_name) != required_names:
        raise ValueError(
            "points must be named exactly cutting_tool_point and "
            "target_object_point"
        )

    def finite_number(key: str, default: float) -> float:
        raw = normalized.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    pos_tol = finite_number("pos_tol", 0.012)
    ori_tol_deg = finite_number("ori_tol_deg", 5.0)
    max_steps_value = finite_number("max_steps", 360.0)
    timeout_s = finite_number("timeout_s", 90.0)
    if not 0.001 <= pos_tol <= 0.10:
        raise ValueError("pos_tol must be in 0.001..0.10m")
    if not 0.1 <= ori_tol_deg <= 45.0:
        raise ValueError("ori_tol_deg must be in 0.1..45 degrees")
    if (
        not max_steps_value.is_integer()
        or not 1 <= max_steps_value <= CUT_OBJECT_MAX_STEPS
    ):
        raise ValueError(
            f"max_steps must be an integer in 1..{CUT_OBJECT_MAX_STEPS}"
        )
    if not 0.1 <= timeout_s <= CUT_OBJECT_MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_s must be in 0.1..{CUT_OBJECT_MAX_TIMEOUT_S:g}s"
        )

    return {
        "session_id": tracked["session_id"],
        "image_id": tracked["image_id"],
        "points": [by_name[name] for name in CUT_OBJECT_POINT_NAMES],
        "pos_tol": pos_tol,
        "ori_tol_deg": ori_tol_deg,
        "max_steps": int(max_steps_value),
        "timeout_s": timeout_s,
    }


__all__ = [
    "ACTION_DIM",
    "ACTION_SLICES",
    "ARM_DOF",
    "CUT_OBJECT_MAX_STEPS",
    "CUT_OBJECT_MAX_TIMEOUT_S",
    "CUT_OBJECT_POINT_NAMES",
    "PROPRIO_DIM",
    "PROPRIO_SLICES",
    "READ_DEPTH_COORDINATE_MAX",
    "READ_DEPTH_COORDINATE_MIN",
    "MOVE_CHASSIS_SURFACE_MAX_TIMEOUT_S",
    "MOVE_CHASSIS_SURFACE_POINT_COUNT",
    "MOVE_POINT_TO_POINT_MAX_ABOVE_M",
    "MOVE_POINT_TO_POINT_MAX_STEPS",
    "MOVE_POINT_TO_POINT_MAX_TIMEOUT_S",
    "MOVE_TRACKED_POINT_MAX_POINTS",
    "MOVE_TRACKED_POINT_MAX_INEQUALITIES",
    "MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS",
    "MOVE_TRACKED_POINT_QUICK_MAX_POINTS",
    "MOVE_TRACKED_POINT_MAX_STEPS",
    "MOVE_TRACKED_POINT_MAX_TIMEOUT_S",
    "MOVE_TRACKED_POINT_STRICT_INEQUALITY_MARGIN_M",
    "NAVIGATE_TO_MAX_ARRIVAL_TOLERANCE_M",
    "NAVIGATE_TO_MAX_TIMEOUT_S",
    "NAVIGATE_TO_MIN_ARRIVAL_TOLERANCE_M",
    "NAVIGATE_TO_NAME_MAX_LEN",
    "OFFICIAL_SESSION_ID_MAX_LEN",
    "ROBOT_PROFILE",
    "TRACK_OBJECT_DISTANCE_MAX_POINTS",
    "parse_tracked_affine_expression",
    "normalize_tracked_inequalities",
    "normalize_tracked_target_coordinate",
    "validate_cut_object_args",
    "validate_move_chassis_to_directly_facing_surface_args",
    "validate_navigate_to_args",
    "validate_read_depth_args",
    "validate_move_tracked_point_args",
    "validate_move_point_to_point_args",
    "validate_track_object_distance_args",
]
