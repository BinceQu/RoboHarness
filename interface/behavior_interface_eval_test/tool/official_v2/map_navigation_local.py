"""Map-backend-neutral, evaluator-safe 2D clearance path planning.

The planner consumes a plain snapshot published by a policy-owned mapping
backend.  It has no simulator, scene, robot-state, or RTAB-Map dependency.
Unknown cells are deliberately non-traversable. An optional mapper-provided
history may promote only the area physically swept by the base footprint and
may guide a route when a degraded map invents free-space shortcuts. Fresh
occupied cells remain protected except for a separately certified, bounded
movable-leaf recovery.
"""

from __future__ import annotations

import bisect
import heapq
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NAVIGATION_SNAPSHOT_SCHEMA = "official_v2_navigation_map"
NAVIGATION_PLAN_SCHEMA = "official_v2_clearance_path"

# Standalone defaults; navigate_to passes the active robot-contract base radius.
DEFAULT_ROBOT_RADIUS_M = 0.42
DEFAULT_SAFETY_MARGIN_M = 0.08
START_EGRESS_SAFETY_MARGIN_M = 0.02
DEFAULT_CLEARANCE_WEIGHT = 2.0
DEFAULT_ROUTE_SWEEP_RADIUS_M = 0.40
SWEEP_OVERLAP_WEIGHT = 12.0
DEFAULT_ARRIVAL_TOLERANCE_M = 0.25
GOAL_TRACKING_RESERVE_M = 0.08
# The executor monitors every action against the continuously certified line
# segment.  Six metres matches the simplifier lookahead, so path_xy_m preserves
# actual polyline corners instead of inserting artificial stop points.
DEFAULT_MAX_SEGMENT_M = 6.0
EXECUTION_CERTIFICATE_RESERVE_M = 0.30
PLANNING_RESOLUTION_M = 0.10
SIMPLIFY_MAX_LOOKAHEAD_M = 6.0
# A clearance-aware A* path may contain staircase corners whose direct chord
# has the same bottleneck clearance but a slightly larger integrated soft
# cost.  A small bounded stretch removes those execution stops without
# turning the hard footprint margin into a soft preference.
SIMPLIFY_MAX_CLEARANCE_COST_STRETCH = 1.12
# A point-to-point segment certificate is not enough for an in-place heading
# change. The static base radius encloses the footprint at every yaw. Certify
# that full disk plus a tracking reserve at each real corner, not additional
# translated copies of the disk which would inflate the footprint twice.
TURN_MIN_ANGLE_DEG = 25.0
TURNING_CLEARANCE_RESERVE_M = 0.12
TURNING_POCKET_SEARCH_M = 0.60
TURNING_POCKET_MAX_CANDIDATES = 192
MAX_GRID_CELLS = 2_000_000
MAX_EDGE_REPLAN_PASSES = 32
# A narrow doorway can require the rectangular base to translate while it
# changes yaw.  One refined-grid cell (normally 2.5 cm) is too short to bridge
# the two fixed-yaw configuration-space components.  Candidate centre motions
# remain bounded and every accepted primitive is certified by the continuous
# polygon sweep below; this is not an obstacle or unknown-space relaxation.
PARTIAL_TURN_MAX_TRANSLATION_M = 0.30
PARTIAL_TURN_DIRECTION_COUNT = 16
PARTIAL_TURN_DISTANCE_LEVELS = 6
PARTIAL_TURN_CANDIDATES_PER_OFFSET = 2
PARTIAL_TURN_CANDIDATES_PER_DISTANCE = 4
MAX_TRAVERSED_PATH_POINTS = 250_000
TRAVERSED_PATH_MAX_LINK_M = 0.75
TRAVERSED_OVERRIDE_MAX_SEGMENT_M = 0.50
TRAVERSED_DIRECT_MAX_SEGMENT_M = 1.00
TRAVERSED_GRAPH_MERGE_RESOLUTION_M = 0.05
# Revisit joins are topological edges, not a generic proximity shortcut.  A
# 20 cm join can bridge the two sides of a thin wall or a doorway jamb after
# several laps have been fused into the same map.  Bucket coalescing already
# joins sub-cell repeats; this smaller bound only absorbs normal centreline
# jitter while keeping nearby, non-connected passages distinct.
TRAVERSED_GRAPH_REVISIT_RADIUS_M = 0.075
TRAVERSED_GRAPH_ENDPOINT_NEAREST_SLACK_M = 0.075
MAX_TRAVERSED_GRAPH_NODES = 40_000
# Static-map overlap is a secondary preference once a previous traversal has
# certified the corridor.  Capping it keeps the selected history route within
# 1.35x of the geometric shortest route instead of accepting multi-lap detours
# to avoid stale occupied pixels.
TRAVERSED_GRAPH_MAX_OVERLAP_PENALTY = 0.35
TRAVERSED_PREFERENCE_DETOUR_RATIO = 1.50
TRAVERSED_PREFERENCE_MIN_SAVING_M = 0.50
TRAVERSED_PREFERENCE_MAX_ENDPOINT_M = 0.30
# A history-guided shortcut may smooth localization jitter, but must remain in
# a narrow tube around space where a base centre was actually observed.  This
# prevents a missing wall in the current raster from turning an L-shaped taught
# route into a diagonal chord through untraversed space.
TRAVERSED_GUIDANCE_TUBE_M = 0.12
TRAVERSED_SELECTED_ROUTE_TUBE_M = 0.06
TRAVERSED_CENTERLINE_WEIGHT = 3.0
TRAVERSED_GUIDANCE_MIN_UNSUPPORTED_M = 0.30
TRAVERSED_GUIDANCE_MIN_UNSUPPORTED_RATIO = 0.05
# Keep the planner's endpoint classification identical to the executor's
# signed-path tolerance.  Raster cell centres can differ from an exact pose by
# a few tens of microns after a pose-graph update; that is quantization, not an
# egress segment.
PLAN_ENDPOINT_TOLERANCE_M = 1.0e-4
DEPTH_POINT_MARGIN_M = 0.02
DEPTH_TRACKING_RESERVE_M = 0.02
# A second, explicitly signed retry may spend 1 cm of the tracking reserve
# after the strict 4 cm depth margin proves disconnected.  It retains the
# full 2 cm measurement allowance and is still re-checked against fresh depth
# every five policy ticks during low-speed execution.
DEPTH_TIGHT_TRACKING_RESERVE_M = 0.01
DEPTH_STRICT_MARGIN_POLICY = "strict_depth_margin_v1"
DEPTH_TIGHT_MARGIN_POLICY = "bounded_tight_passage_margin_v1"
DEPTH_REFINEMENT_SCHEMA = "oriented_base_depth_sweep_v1"
STATIC_FOOTPRINT_SCHEMA = "oriented_base_static_grid_sweep_v1"
DEPTH_EGRESS_GAP_M = .01
TRAVERSED_THIN_BARRIER_SCHEMA = "traversed_thin_vertical_barrier_v1"

# A traversed-barrier override is limited to one geometrically certified
# movable leaf.  It is never a blanket exception for a walked corridor, so
# separate jambs, long walls, and unrelated current-depth obstacles remain.
THIN_BARRIER_CLUSTER_VOXEL_M = .04
THIN_BARRIER_CLUSTER_LINK_M = .12
THIN_BARRIER_MIN_POINTS = 24
# Robust 2/98-percentile endpoints shorten a partly occluded half-metre leaf.
# Keep this below that nominal width; the independent vertical-plane, mapped-
# free, endpoint-topology and continuous-history proofs remain mandatory.
THIN_BARRIER_MIN_TANGENT_SPAN_M = .45
THIN_BARRIER_MAX_TANGENT_SPAN_M = 1.75
# A wider plane is eligible only for the stricter free-end path below.  This
# covers a fully observed wide door/partition without letting a reference path
# turn an equally long wall into a direct-crossing exception.
THIN_BARRIER_MAX_FREE_END_TANGENT_SPAN_M = 2.05
THIN_BARRIER_MAX_NORMAL_SPAN_M = .10
THIN_BARRIER_MIN_VERTICAL_SPAN_M = .60
THIN_BARRIER_MIN_HIGH_Z_M = .85
THIN_BARRIER_MAX_LOW_Z_M = .45
# Near a leaf, the pitched head camera can lose the panel bottom below the
# image while still observing more than a metre of its vertical face.  That
# is not equivalent to seeing a chassis-height obstacle.  Permit the missing
# lower extent only after a proprioceptively verified base stall at this same
# plane and with multiple plane inliers on the lower image boundary.
THIN_BARRIER_CENSORED_MIN_VERTICAL_SPAN_M = .80
THIN_BARRIER_CENSORED_MIN_HIGH_Z_M = 1.45
THIN_BARRIER_CENSORED_MAX_LOW_Z_M = .95
THIN_BARRIER_CENSORED_MIN_BOUNDARY_POINTS = 8
THIN_BARRIER_CENSORED_MAX_STALL_PLANE_DISTANCE_M = .67
THIN_BARRIER_CENSORED_MAX_STALL_CENTROID_DISTANCE_M = 1.35
# At very close range a pitched camera can crop the panel below its physical
# bottom without putting any inliers on the image boundary (the robot or an
# occluder owns those pixels).  This second recovery is intentionally much
# stricter than ordinary boundary censoring: it is usable only after a real
# base stall, on a nearly ideal vertical plane in previously mapped free
# space, and the selected route must still go around the currently observed
# free endpoint.
THIN_BARRIER_STALL_HISTORY_MIN_VERTICAL_SPAN_M = 1.00
THIN_BARRIER_STALL_HISTORY_MIN_HIGH_Z_M = 1.60
THIN_BARRIER_STALL_HISTORY_MAX_LOW_Z_M = .70
THIN_BARRIER_STALL_HISTORY_MAX_PLANE_STD_M = .015
THIN_BARRIER_STALL_HISTORY_MIN_ROBUST_INLIER_RATIO = .75
THIN_BARRIER_STALL_HISTORY_MIN_THIN_SLAB_RATIO = .95
THIN_BARRIER_STALL_HISTORY_MAX_STALL_PLANE_DISTANCE_M = .45
THIN_BARRIER_STALL_HISTORY_MAX_STALL_CENTROID_DISTANCE_M = .75
THIN_BARRIER_STALL_HISTORY_MIN_MAP_FREE_RATIO = .90
THIN_BARRIER_STALL_HISTORY_MAX_MAP_WALL_RATIO = .02
THIN_BARRIER_MAX_PLANE_STD_M = .035
THIN_BARRIER_MIN_ROUTE_NORMAL_DOT = .65
THIN_BARRIER_HISTORY_SIDE_M = .65
THIN_BARRIER_HISTORY_MAX_SIDE_M = 1.50
THIN_BARRIER_HISTORY_SIDE_STEP_M = .05
THIN_BARRIER_HISTORY_MIN_NORMAL_PROGRESS_M = .25
THIN_BARRIER_HISTORY_MAX_CROSSING_ERROR_M = .20
THIN_BARRIER_MAX_ROUTE_DISTANCE_M = 2.25
THIN_BARRIER_ROUTE_CONTACT_PAD_M = .10
# A historical centreline is only an association witness: the executable
# route is planned again against current depth.  Allow a slightly wider
# association band for that witness so a 0.42 m base which previously cleared
# a leaf by normal navigation reserve is not mistaken for an unrelated route.
THIN_BARRIER_HISTORY_ROUTE_CONTACT_PAD_M = .16
THIN_BARRIER_FREE_END_CONTACT_M = .20
# A direct plane crossing is safe to reinterpret as an ajar movable leaf only
# at one of its observed ends.  Crossing the middle may instead be a closed
# door, glass panel, or thin furniture and must remain blocked.
THIN_BARRIER_DIRECT_MAX_ENDPOINT_DISTANCE_M = .20
THIN_BARRIER_FREE_END_MAX_CROSSING_OFFSET_M = 1.00
THIN_BARRIER_ANCHOR_MAX_DISTANCE_M = .36
THIN_BARRIER_FREE_END_MIN_DISTANCE_M = .55
# A depth edge or a few returns from the leaf itself can sit just beyond the
# robust 2/98-percentile endpoint.  A previously traversed sweep may identify
# that end with a weaker absolute gap, but only when the opposite end is much
# more strongly attached and the plane has the width of a normal door leaf.
THIN_BARRIER_HISTORY_FREE_END_MIN_DISTANCE_M = .16
THIN_BARRIER_HISTORY_ENDPOINT_MARGIN_M = .12
THIN_BARRIER_HISTORY_ENDPOINT_RATIO = 2.0
THIN_BARRIER_HISTORY_MAX_ASYMMETRIC_SPAN_M = 1.40
# Endpoint attachment is a horizontal-geometry claim.  Multiple depth pixels
# on one vertical ray are one XY observation and cannot, by themselves, prove
# that a leaf is attached to a jamb.
THIN_BARRIER_ENDPOINT_CONTEXT_MIN_CELLS = 3
# If neither endpoint has supported attachment context, recovery is still
# possible for a bounded leaf lying almost entirely in mapped free space.  The
# 4.5 cm robust-fit core is deliberately narrow; perspective and the thickness
# of a real leaf can put many valid returns just outside it.  For this fallback
# require the *whole raw component* to remain inside a 10 cm slab instead.
THIN_BARRIER_CLEAR_END_MAX_SLAB_RESIDUAL_M = .10
THIN_BARRIER_CLEAR_END_MIN_SLAB_RATIO = .90
THIN_BARRIER_CLEAR_END_MIN_FREE_RATIO = .90
THIN_BARRIER_CLEAR_END_MAX_WALL_RATIO = .02
# A route through the current panel interior is only evidence of a changed
# door pose when the live plane lands in cells the persistent map had already
# observed as free, rather than on its static wall layer.
THIN_BARRIER_CHANGED_POSE_MIN_FREE_CELLS = 6
THIN_BARRIER_CHANGED_POSE_MIN_FREE_RATIO = .55
THIN_BARRIER_CHANGED_POSE_MAX_WALL_RATIO = .10
THIN_BARRIER_MIN_CONTACT_SPAN_M = .30
THIN_BARRIER_OVERRIDE_TANGENT_PAD_M = .03
# Endpoint extents use robust quantiles, while a depth image still contains a
# sparse inlier tail beyond that estimate.  Keep this one-sided and bounded so
# those leaf returns cannot falsely close the historically crossed free end.
THIN_BARRIER_FREE_END_TAIL_PAD_M = .23
THIN_BARRIER_ROBUST_RESIDUAL_M = .045
THIN_BARRIER_ROBUST_MIN_INLIER_RATIO = .45
# A close door leaf can be XY-connected to a perpendicular jamb or wall.  The
# ordinary iterative PCA deliberately keeps its original consensus rule.  A
# verified-stall retry may additionally extract one dominant vertical
# subplane with a bounded deterministic fit; the map/history gates below are
# still required before that subplane can authorize motion.
THIN_BARRIER_CONNECTED_SUBPLANE_MIN_INLIER_RATIO = .60
THIN_BARRIER_CONNECTED_SUBPLANE_MIN_RAW_SLAB_RATIO = .60
THIN_BARRIER_CONNECTED_SUBPLANE_MIN_SEED_SPAN_M = .35
THIN_BARRIER_CONNECTED_SUBPLANE_MAX_VOXEL_SEEDS = 96
# Only the free-end portion of an ajar leaf may be ignored.  The extra swept
# allowance covers map/depth registration error while the separately enforced
# centreline preference keeps the base on the historically traversed side of
# the tip.  The anchored remainder remains a live obstacle.
THIN_BARRIER_FREE_END_OVERRIDE_CLEARANCE_PAD_M = .25
THIN_BARRIER_PREFERRED_PATH_TUBE_M = .16
THIN_BARRIER_PREFERRED_PATH_RAMP_M = .24
THIN_BARRIER_PREFERRED_PATH_WEIGHT = 24.0
THIN_BARRIER_PREFERRED_PATH_SIMPLIFY_SLACK_M = .03
# Once a changed leaf has been associated with a taught traversal, a soft
# route preference is insufficient: leaving the local cost region can be
# cheaper than using the observed free end.  Cut the private configuration
# space at the current leaf plane and leave one bounded centre-path gate at
# the newly constructed free-end sweep.  This only removes route candidates;
# it never clears live or persistent collision geometry.
THIN_BARRIER_PREFERRED_GATE_MIN_HALF_WIDTH_M = .18
THIN_BARRIER_PREFERRED_GATE_MAX_HALF_WIDTH_M = .30
THIN_BARRIER_PREFERRED_GATE_FENCE_HALF_THICKNESS_M = .06
THIN_BARRIER_CONTACT_SPEED_MPS = .15
DEPTH_REFINEMENT_LAZY_EDGE_REPLANS = 12
DEPTH_REFINEMENT_ROUTE_CROP_PAD_M = 2.0


class NavigationPlanError(ValueError):
    """Raised when a map snapshot cannot produce a safe navigation plan."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NavigationPlanError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise NavigationPlanError(f"{label} must be finite")
    return result


def _pair(value: Any, label: str) -> tuple[float, float]:
    try:
        items = list(value)
    except TypeError as exc:
        raise NavigationPlanError(f"{label} must contain x and y") from exc
    if len(items) != 2:
        raise NavigationPlanError(f"{label} must contain exactly x and y")
    return _finite(items[0], f"{label}.x"), _finite(items[1], f"{label}.y")


def _snapshot_geometry(
    snapshot: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    raw = snapshot.get("occupancy")
    if raw is None:
        raise NavigationPlanError("navigation map has no occupancy grid")
    occupancy = np.asarray(raw)
    if occupancy.ndim != 2 or not occupancy.size:
        raise NavigationPlanError("occupancy must be a non-empty 2D grid")
    if occupancy.size > MAX_GRID_CELLS:
        raise NavigationPlanError(
            f"occupancy has {occupancy.size} cells; limit is {MAX_GRID_CELLS}"
        )
    resolution = _finite(
        snapshot.get("resolution_m", snapshot.get("resolution")),
        "resolution",
    )
    if resolution <= 0.0 or resolution > 1.0:
        raise NavigationPlanError("resolution must be in (0, 1] meters")
    origin = _pair(
        snapshot.get("origin_xy_m", snapshot.get("origin")),
        "origin",
    )

    obstacle_value = snapshot.get("obstacle_mask")
    free_value = snapshot.get("free_mask")
    if obstacle_value is not None or free_value is not None:
        if obstacle_value is None or free_value is None:
            raise NavigationPlanError(
                "obstacle_mask and free_mask must be published together"
            )
        obstacle = np.asarray(obstacle_value, dtype=bool)
        free = np.asarray(free_value, dtype=bool)
        if obstacle.shape != occupancy.shape or free.shape != occupancy.shape:
            raise NavigationPlanError("navigation map masks have different shapes")
        free = free & ~obstacle
        return obstacle, free, resolution, origin

    if occupancy.dtype == np.bool_:
        obstacle = occupancy.astype(bool, copy=True)
        return obstacle, ~obstacle, resolution, origin

    try:
        values = occupancy.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise NavigationPlanError("occupancy values must be numeric") from exc
    finite = np.isfinite(values)
    if not np.any(finite):
        raise NavigationPlanError("occupancy contains no finite cells")
    finite_values = values[finite]
    probability_scale = (
        float(finite_values.min()) >= 0.0
        and float(finite_values.max()) <= 1.0
        and not np.issubdtype(occupancy.dtype, np.integer)
    )
    if probability_scale:
        obstacle = finite & (values >= 0.65)
        free = finite & (values <= 0.35)
    else:
        # ROS / RTAB-Map convention: -1 unknown, 0 free, 100 occupied.
        obstacle = finite & (values >= 50.0)
        free = finite & (values >= 0.0) & (values < 50.0)
    return obstacle, free & ~obstacle, resolution, origin


def _pose(snapshot: Mapping[str, Any]) -> tuple[float, float, float]:
    pose = snapshot.get("pose")
    if not isinstance(pose, Mapping):
        raise NavigationPlanError("navigation map has no current pose")
    x = _finite(pose.get("x_m", pose.get("x")), "pose.x")
    y = _finite(pose.get("y_m", pose.get("y")), "pose.y")
    if pose.get("yaw_rad") is not None:
        yaw = _finite(pose.get("yaw_rad"), "pose.yaw_rad")
    else:
        yaw = math.radians(
            _finite(pose.get("yaw_deg", 0.0), "pose.yaw_deg")
        )
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw))


def _goal(snapshot: Mapping[str, Any], name: str) -> tuple[float, float, int]:
    label = str(name or "").strip()
    if not label:
        raise NavigationPlanError("marked place name is required")
    places = snapshot.get("places")
    if not isinstance(places, Sequence) or isinstance(places, (str, bytes)):
        raise NavigationPlanError("navigation map has no marked places")
    matches: list[tuple[float, float]] = []
    for item in places:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("name") or "").strip() != label:
            continue
        matches.append(
            (
                _finite(item.get("x_m", item.get("x")), f"place {label}.x"),
                _finite(item.get("y_m", item.get("y")), f"place {label}.y"),
            )
        )
    if not matches:
        available = sorted(
            {
                str(item.get("name") or "").strip()
                for item in places
                if isinstance(item, Mapping) and str(item.get("name") or "").strip()
            }
        )
        suffix = "" if not available else f"; available: {', '.join(available)}"
        raise NavigationPlanError(f"marked place {label!r} was not found{suffix}")
    x, y = matches[-1]
    return x, y, len(matches)


def _traversed_paths(
    snapshot: Mapping[str, Any],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    value = snapshot.get("traversed_paths_xy_m")
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise NavigationPlanError(
            "traversed_paths_xy_m must be a sequence of paths"
        )
    paths: list[tuple[tuple[float, float], ...]] = []
    point_count = 0
    for path_index, raw_path in enumerate(value):
        if not isinstance(raw_path, Sequence) or isinstance(
            raw_path, (str, bytes)
        ):
            raise NavigationPlanError(
                f"traversed path {path_index} must be a sequence of points"
            )
        path: list[tuple[float, float]] = []
        for point_index, raw_point in enumerate(raw_path):
            point = _pair(
                raw_point,
                f"traversed path {path_index} point {point_index}",
            )
            if path and math.dist(path[-1], point) <= 1.0e-9:
                continue
            path.append(point)
            point_count += 1
            if point_count > MAX_TRAVERSED_PATH_POINTS:
                raise NavigationPlanError(
                    "traversed path evidence exceeds the point limit"
                )
        if path:
            paths.append(tuple(path))
    return tuple(paths)


def _distance_to_traversed_paths(
    point: tuple[float, float],
    paths: Sequence[Sequence[tuple[float, float]]],
) -> float | None:
    """Return the exact 2-D distance to the nearest recorded path segment."""

    nearest = math.inf
    for path in paths:
        if len(path) == 1:
            nearest = min(nearest, math.dist(point, path[0]))
            continue
        for start, end in zip(path[:-1], path[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length_squared = dx * dx + dy * dy
            if length_squared <= 1.0e-18:
                distance = math.dist(point, start)
            else:
                projection = (
                    (point[0] - start[0]) * dx
                    + (point[1] - start[1]) * dy
                ) / length_squared
                projection = min(1.0, max(0.0, projection))
                nearest_x = start[0] + projection * dx
                nearest_y = start[1] + projection * dy
                distance = math.hypot(
                    point[0] - nearest_x,
                    point[1] - nearest_y,
                )
            nearest = min(nearest, distance)
    return None if not math.isfinite(nearest) else float(nearest)


def _continuous_traversed_components(
    paths: Sequence[Sequence[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    components: list[list[tuple[float, float]]] = []
    for path in paths:
        current: list[tuple[float, float]] = []
        for point in path:
            if current and (
                math.dist(current[-1], point)
                > TRAVERSED_PATH_MAX_LINK_M + 1.0e-9
            ):
                components.append(current)
                current = []
            current.append(point)
        if current:
            components.append(current)
    return components


def _project_onto_polyline(
    point: tuple[float, float],
    path: Sequence[tuple[float, float]],
) -> tuple[float, float, tuple[float, float], list[float]]:
    cumulative = [0.0]
    for start, end in zip(path[:-1], path[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    if len(path) == 1:
        return math.dist(point, path[0]), 0.0, path[0], cumulative
    best_distance = math.inf
    best_arc = 0.0
    best_point = path[0]
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1.0e-18:
            projection = 0.0
        else:
            projection = (
                (point[0] - start[0]) * dx
                + (point[1] - start[1]) * dy
            ) / length_squared
            projection = min(1.0, max(0.0, projection))
        projected = (
            start[0] + projection * dx,
            start[1] + projection * dy,
        )
        distance = math.dist(point, projected)
        arc = cumulative[index] + projection * math.sqrt(length_squared)
        if (distance, arc) < (best_distance, best_arc):
            best_distance = distance
            best_arc = arc
            best_point = projected
    return float(best_distance), float(best_arc), best_point, cumulative


def _traversed_route_between(
    paths: Sequence[Sequence[tuple[float, float]]],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_endpoint_distance_m: float,
    max_goal_endpoint_distance_m: float | None = None,
) -> tuple[list[tuple[float, float]], float, float] | None:
    """Extract the shortest time-continuous route between two projections.

    A repeatedly visited place has several equally plausible projections on
    one recorded trajectory.  Selecting only the first nearest projection can
    replay an entire old lap, while rasterizing all laps can connect opposite
    sides of a wall.  Enumerate every admissible projection and choose the
    shortest sub-arc without discarding the trajectory's temporal topology.
    """

    goal_limit = (
        max_endpoint_distance_m
        if max_goal_endpoint_distance_m is None
        else max_goal_endpoint_distance_m
    )
    best: tuple[
        tuple[float, float, float, int, float, float],
        list[tuple[float, float]],
        float,
        float,
    ] | None = None

    def projections(
        point: tuple[float, float],
        path: Sequence[tuple[float, float]],
        cumulative: Sequence[float],
        limit: float,
    ) -> list[tuple[float, float, tuple[float, float]]]:
        if len(path) == 1:
            distance = math.dist(point, path[0])
            return [(distance, 0.0, path[0])] if distance <= limit + 1e-9 else []
        candidates: list[tuple[float, float, tuple[float, float]]] = []
        for index, (segment_start, segment_end) in enumerate(
            zip(path[:-1], path[1:])
        ):
            dx = segment_end[0] - segment_start[0]
            dy = segment_end[1] - segment_start[1]
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-18:
                fraction = 0.0
                segment_length = 0.0
            else:
                fraction = min(
                    1.0,
                    max(
                        0.0,
                        (
                            (point[0] - segment_start[0]) * dx
                            + (point[1] - segment_start[1]) * dy
                        )
                        / length_squared,
                    ),
                )
                segment_length = math.sqrt(length_squared)
            projected = (
                segment_start[0] + fraction * dx,
                segment_start[1] + fraction * dy,
            )
            distance = math.dist(point, projected)
            if distance <= limit + 1e-9:
                candidates.append(
                    (
                        float(distance),
                        float(cumulative[index] + fraction * segment_length),
                        projected,
                    )
                )
        return candidates

    for path_index, path in enumerate(_continuous_traversed_components(paths)):
        cumulative = [0.0]
        for segment_start, segment_end in zip(path[:-1], path[1:]):
            cumulative.append(
                cumulative[-1] + math.dist(segment_start, segment_end)
            )
        start_candidates = projections(
            start_xy, path, cumulative, max_endpoint_distance_m
        )
        goal_candidates = projections(goal_xy, path, cumulative, goal_limit)
        if not start_candidates or not goal_candidates:
            continue
        # Endpoint tolerances express how far localization/marking may lie
        # from history, not permission to attach to every nearby branch. In a
        # doorway, two distinct traversals can both fall inside 30 cm when the
        # wall raster is missing. Keep only projections close to the nearest
        # branch while retaining a small allowance for trail jitter.
        nearest_start_distance = min(
            candidate[0] for candidate in start_candidates
        )
        nearest_goal_distance = min(candidate[0] for candidate in goal_candidates)
        start_candidates = [
            candidate
            for candidate in start_candidates
            if candidate[0]
            <= nearest_start_distance
            + TRAVERSED_GRAPH_ENDPOINT_NEAREST_SLACK_M
            + 1.0e-9
        ]
        goal_candidates = [
            candidate
            for candidate in goal_candidates
            if candidate[0]
            <= nearest_goal_distance
            + TRAVERSED_GRAPH_ENDPOINT_NEAREST_SLACK_M
            + 1.0e-9
        ]
        goal_candidates.sort(key=lambda candidate: candidate[1])
        goal_arcs = [candidate[1] for candidate in goal_candidates]
        prefix_best: list[int] = []
        best_index = 0
        for candidate_index, (distance, arc, _point) in enumerate(
            goal_candidates
        ):
            if (
                distance - arc,
                -arc,
                distance,
                candidate_index,
            ) < (
                goal_candidates[best_index][0] - goal_candidates[best_index][1],
                -goal_candidates[best_index][1],
                goal_candidates[best_index][0],
                best_index,
            ):
                best_index = candidate_index
            prefix_best.append(best_index)
        suffix_best = [0] * len(goal_candidates)
        best_index = len(goal_candidates) - 1
        for candidate_index in range(len(goal_candidates) - 1, -1, -1):
            distance, arc, _point = goal_candidates[candidate_index]
            if (
                distance + arc,
                arc,
                distance,
                candidate_index,
            ) < (
                goal_candidates[best_index][0] + goal_candidates[best_index][1],
                goal_candidates[best_index][1],
                goal_candidates[best_index][0],
                best_index,
            ):
                best_index = candidate_index
            suffix_best[candidate_index] = best_index
        for start_distance, start_arc, start_point in start_candidates:
            split = bisect.bisect_right(goal_arcs, start_arc)
            candidate_indices: list[int] = []
            if split:
                candidate_indices.append(prefix_best[split - 1])
            if split < len(goal_candidates):
                candidate_indices.append(suffix_best[split])
            for candidate_index in dict.fromkeys(candidate_indices):
                goal_distance, goal_arc, goal_point = goal_candidates[
                    candidate_index
                ]
                route_length = abs(goal_arc - start_arc)
                score = (
                    route_length + start_distance + goal_distance,
                    route_length,
                    start_distance + goal_distance,
                    path_index,
                    start_arc,
                    goal_arc,
                )
                if best is not None and score >= best[0]:
                    continue
                low_arc = min(start_arc, goal_arc)
                high_arc = max(start_arc, goal_arc)
                low_point = start_point if start_arc <= goal_arc else goal_point
                high_point = goal_point if start_arc <= goal_arc else start_point
                route = [low_point]
                route.extend(
                    vertex
                    for vertex, arc in zip(path[1:-1], cumulative[1:-1])
                    if low_arc + 1.0e-9 < arc < high_arc - 1.0e-9
                )
                route.append(high_point)
                if start_arc > goal_arc:
                    route.reverse()
                deduplicated = [route[0]]
                for vertex in route[1:]:
                    if math.dist(deduplicated[-1], vertex) > 1.0e-9:
                        deduplicated.append(vertex)
                best = (
                    score,
                    deduplicated,
                    float(start_distance),
                    float(goal_distance),
                )
    if best is None:
        return None
    return best[1], best[2], best[3]


def _point_at_polyline_arc(
    path: Sequence[tuple[float, float]],
    cumulative: Sequence[float],
    arc_m: float,
) -> tuple[float, float]:
    """Interpolate one point on a polyline at a clamped arc length."""

    if not path:
        raise NavigationPlanError("cannot sample an empty traversed path")
    target = min(max(0.0, float(arc_m)), float(cumulative[-1]))
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        low = float(cumulative[index])
        high = float(cumulative[index + 1])
        if target > high + 1.0e-12:
            continue
        fraction = 0.0 if high <= low else (target - low) / (high - low)
        return (
            float(start[0] + fraction * (end[0] - start[0])),
            float(start[1] + fraction * (end[1] - start[1])),
        )
    return float(path[-1][0]), float(path[-1][1])


def _thin_barrier_history_witness(
    paths: Sequence[Sequence[tuple[float, float]]],
    crossing_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_limit_m: float,
) -> dict[str, Any] | None:
    """Prove that one continuous recorded base path crossed both barrier sides."""

    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for path_index, path in enumerate(_continuous_traversed_components(paths)):
        if len(path) < 2:
            continue
        _distance, projected_arc, projected, cumulative = _project_onto_polyline(
            (float(crossing_xy[0]), float(crossing_xy[1])), path
        )
        total = float(cumulative[-1])
        if (
            projected_arc < THIN_BARRIER_HISTORY_SIDE_M
            or total - projected_arc < THIN_BARRIER_HISTORY_SIDE_M
        ):
            continue
        projected_xy = np.asarray(projected, dtype=np.float64)
        crossing_error = float(np.linalg.norm(projected_xy - crossing_xy))
        projected_tangent = float((projected_xy - crossing_xy) @ tangent_xy)
        if crossing_error > THIN_BARRIER_HISTORY_MAX_CROSSING_ERROR_M:
            continue

        def side_samples(direction: float, available_m: float) -> list[tuple]:
            maximum = min(THIN_BARRIER_HISTORY_MAX_SIDE_M, available_m)
            offsets = np.arange(
                THIN_BARRIER_HISTORY_SIDE_M,
                maximum + .5 * THIN_BARRIER_HISTORY_SIDE_STEP_M,
                THIN_BARRIER_HISTORY_SIDE_STEP_M,
            )
            if not len(offsets) or offsets[-1] < maximum - 1.0e-9:
                offsets = np.append(offsets, maximum)
            result = []
            for offset in offsets:
                point = np.asarray(
                    _point_at_polyline_arc(
                        path,
                        cumulative,
                        projected_arc + direction * float(offset),
                    ),
                    dtype=np.float64,
                )
                result.append((
                    float(offset),
                    point,
                    float((point - crossing_xy) @ normal_xy),
                    float((point - crossing_xy) @ tangent_xy),
                ))
            return result

        before_samples = side_samples(-1.0, projected_arc)
        after_samples = side_samples(1.0, total - projected_arc)
        for before_offset, before, before_normal, before_tangent in before_samples:
            for after_offset, after, after_normal, after_tangent in after_samples:
                normal_progress = min(abs(before_normal), abs(after_normal))
                max_tangent = max(
                    abs(before_tangent),
                    abs(after_tangent),
                    abs(projected_tangent),
                )
                if (
                    before_normal * after_normal >= 0.0
                    or normal_progress
                    < THIN_BARRIER_HISTORY_MIN_NORMAL_PROGRESS_M
                    or max_tangent > tangent_limit_m + 1.0e-9
                ):
                    continue
                witness = {
                    "path_index": int(path_index),
                    "projected_arc_m": float(projected_arc),
                    "path_length_m": total,
                    "crossing_error_m": crossing_error,
                    "projected_xy_m": projected_xy.astype(float).tolist(),
                    "before_xy_m": before.astype(float).tolist(),
                    "after_xy_m": after.astype(float).tolist(),
                    "before_normal_m": before_normal,
                    "after_normal_m": after_normal,
                    "before_arc_offset_m": before_offset,
                    "after_arc_offset_m": after_offset,
                    "maximum_tangent_offset_m": max_tangent,
                    "tangent_limit_m": float(tangent_limit_m),
                }
                score = (
                    crossing_error,
                    before_offset + after_offset,
                    max_tangent,
                    -normal_progress,
                )
                if best is None or score < best[0]:
                    best = score, witness
    return None if best is None else best[1]


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _project_point_to_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, float]:
    delta = end - start
    length_squared = float(delta @ delta)
    fraction = (
        0.0
        if length_squared <= 1.0e-18
        else float(np.clip((point - start) @ delta / length_squared, 0.0, 1.0))
    )
    return start + fraction * delta, fraction


def _closest_segment_points(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Return exact closest points and fractions for two planar segments."""

    first_delta = first_end - first_start
    second_delta = second_end - second_start
    denominator = _cross_2d(first_delta, second_delta)
    if abs(denominator) > 1.0e-12:
        offset = second_start - first_start
        first_fraction = _cross_2d(offset, second_delta) / denominator
        second_fraction = _cross_2d(offset, first_delta) / denominator
        if (
            -1.0e-12 <= first_fraction <= 1.0 + 1.0e-12
            and -1.0e-12 <= second_fraction <= 1.0 + 1.0e-12
        ):
            first_fraction = float(np.clip(first_fraction, 0.0, 1.0))
            second_fraction = float(np.clip(second_fraction, 0.0, 1.0))
            crossing = first_start + first_fraction * first_delta
            return 0.0, crossing, crossing.copy(), first_fraction, second_fraction

    candidates: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    projected, fraction = _project_point_to_segment(
        first_start, second_start, second_end
    )
    candidates.append(
        (
            float(np.linalg.norm(first_start - projected)),
            first_start,
            projected,
            0.0,
            fraction,
        )
    )
    projected, fraction = _project_point_to_segment(
        first_end, second_start, second_end
    )
    candidates.append(
        (
            float(np.linalg.norm(first_end - projected)),
            first_end,
            projected,
            1.0,
            fraction,
        )
    )
    projected, fraction = _project_point_to_segment(
        second_start, first_start, first_end
    )
    candidates.append(
        (
            float(np.linalg.norm(second_start - projected)),
            projected,
            second_start,
            fraction,
            0.0,
        )
    )
    projected, fraction = _project_point_to_segment(
        second_end, first_start, first_end
    )
    candidates.append(
        (
            float(np.linalg.norm(second_end - projected)),
            projected,
            second_end,
            fraction,
            1.0,
        )
    )
    return min(candidates, key=lambda item: (item[0], item[3], item[4]))


def _thin_barrier_free_end_route(
    path_xy_m: Sequence[Sequence[float]],
    centroid_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_min_m: float,
    tangent_max_m: float,
    *,
    free_is_low: bool,
    robot_radius_m: float,
) -> dict[str, Any] | None:
    """Prove that a route sweeps a free leaf end, then exits on that side."""

    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    panel_start = centroid_xy + tangent_min_m * tangent_xy
    panel_end = centroid_xy + tangent_max_m * tangent_xy
    panel_span = float(tangent_max_m - tangent_min_m)
    if panel_span <= 1.0e-9:
        return None

    contact: tuple[tuple[float, float], dict[str, Any]] | None = None
    cumulative = 0.0
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        length = float(np.linalg.norm(end - start))
        if length <= 1.0e-9:
            continue
        distance, route_point, barrier_point, route_fraction, panel_fraction = (
            _closest_segment_points(start, end, panel_start, panel_end)
        )
        route_arc = cumulative + route_fraction * length
        free_end_distance = panel_span * (
            panel_fraction if free_is_low else 1.0 - panel_fraction
        )
        if (
            route_arc <= THIN_BARRIER_MAX_ROUTE_DISTANCE_M
            and distance <= robot_radius_m + THIN_BARRIER_ROUTE_CONTACT_PAD_M
            and free_end_distance <= THIN_BARRIER_FREE_END_CONTACT_M
        ):
            candidate = {
                "contact_xy_m": route_point.astype(float).tolist(),
                "barrier_contact_xy_m": barrier_point.astype(float).tolist(),
                "contact_distance_m": float(distance),
                "contact_arc_m": float(route_arc),
                "contact_segment_index": int(segment_index),
                "free_end_distance_m": float(free_end_distance),
            }
            score = (route_arc, distance)
            if contact is None or score < contact[0]:
                contact = score, candidate
        cumulative += length
    if contact is None:
        return None

    cumulative = 0.0
    plane_crossing: tuple[tuple[float, float], dict[str, Any]] | None = None
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        start_normal = float((start - centroid_xy) @ normal_xy)
        end_normal = float((end - centroid_xy) @ normal_xy)
        denominator = end_normal - start_normal
        if start_normal * end_normal > 0.0 or abs(denominator) <= 1.0e-12:
            cumulative += length
            continue
        fraction = float(np.clip(-start_normal / denominator, 0.0, 1.0))
        crossing = start + fraction * delta
        route_arc = cumulative + fraction * length
        tangent_value = float((crossing - centroid_xy) @ tangent_xy)
        outside_offset = (
            tangent_min_m - tangent_value
            if free_is_low
            else tangent_value - tangent_max_m
        )
        if (
            route_arc + 1.0e-9 < float(contact[1]["contact_arc_m"])
            or route_arc > THIN_BARRIER_MAX_ROUTE_DISTANCE_M
            or outside_offset < -1.0e-9
            or outside_offset > THIN_BARRIER_FREE_END_MAX_CROSSING_OFFSET_M
        ):
            cumulative += length
            continue
        oriented_normal = normal_xy.copy()
        direction = delta / length
        if float(direction @ oriented_normal) < 0.0:
            oriented_normal *= -1.0
        candidate = {
            "plane_crossing_xy_m": crossing.astype(float).tolist(),
            "plane_crossing_arc_m": float(route_arc),
            "plane_crossing_segment_index": int(segment_index),
            "plane_crossing_tangent_offset_m": float(outside_offset),
            "route_normal_alignment": abs(float(direction @ normal_xy)),
            "travel_normal_xy": oriented_normal.astype(float).tolist(),
        }
        score = (route_arc, outside_offset)
        if plane_crossing is None or score < plane_crossing[0]:
            plane_crossing = score, candidate
        cumulative += length
    if plane_crossing is None:
        return None
    result = dict(contact[1])
    result.update(plane_crossing[1])
    result["mode"] = "free_endpoint_swept_contact"
    result["route_source"] = "reference_plan"
    return result


def _thin_barrier_historical_free_end_route(
    path_xy_m: Sequence[Sequence[float]],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    centroid_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_min_m: float,
    tangent_max_m: float,
    *,
    free_is_low: bool,
    robot_radius_m: float,
) -> dict[str, Any] | None:
    """Find a query-relevant taught crossing around an observed leaf end.

    A long trajectory can revisit a doorway many times, so its arc length from
    the current projection is not a useful measure of how near the doorway is
    now.  We instead require the current query endpoints to lie on opposite
    sides, the taught segment to travel in the query direction, and the
    crossing itself to be spatially near the current pose.
    """

    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    query_normal = np.asarray(normal_xy, dtype=np.float64).copy()
    start_normal = float((start_xy - centroid_xy) @ query_normal)
    goal_normal = float((goal_xy - centroid_xy) @ query_normal)
    if start_normal * goal_normal >= 0.0:
        return None
    if goal_normal < start_normal:
        query_normal *= -1.0

    panel_endpoint = centroid_xy + (
        tangent_min_m if free_is_low else tangent_max_m
    ) * tangent_xy
    cumulative = 0.0
    best: tuple[tuple[float, float, float, int], dict[str, Any]] | None = None
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        alignment = float(direction @ query_normal)
        segment_start_normal = float((start - centroid_xy) @ query_normal)
        segment_end_normal = float((end - centroid_xy) @ query_normal)
        denominator = segment_end_normal - segment_start_normal
        if (
            alignment < .25
            or segment_start_normal > 0.0
            or segment_end_normal < 0.0
            or abs(denominator) <= 1.0e-12
        ):
            cumulative += length
            continue
        fraction = float(np.clip(-segment_start_normal / denominator, 0.0, 1.0))
        crossing = start + fraction * delta
        tangent_value = float((crossing - centroid_xy) @ tangent_xy)
        outside_offset = (
            tangent_min_m - tangent_value
            if free_is_low
            else tangent_value - tangent_max_m
        )
        approach_distance = float(np.linalg.norm(crossing - start_xy))
        if (
            outside_offset < -1.0e-9
            or outside_offset
            > min(
                THIN_BARRIER_FREE_END_MAX_CROSSING_OFFSET_M,
                robot_radius_m + THIN_BARRIER_HISTORY_ROUTE_CONTACT_PAD_M,
            )
            or approach_distance > THIN_BARRIER_MAX_ROUTE_DISTANCE_M
        ):
            cumulative += length
            continue
        history_arc = cumulative + fraction * length
        candidate = {
            "contact_xy_m": crossing.astype(float).tolist(),
            "barrier_contact_xy_m": panel_endpoint.astype(float).tolist(),
            "contact_distance_m": float(outside_offset),
            # This is the current query's spatial approach distance.  The
            # (possibly much longer) recorded trajectory arc is kept separately.
            "contact_arc_m": approach_distance,
            "contact_segment_index": int(segment_index),
            "free_end_distance_m": 0.0,
            "plane_crossing_xy_m": crossing.astype(float).tolist(),
            "plane_crossing_arc_m": approach_distance,
            "plane_crossing_segment_index": int(segment_index),
            "plane_crossing_tangent_offset_m": float(outside_offset),
            "route_normal_alignment": alignment,
            "travel_normal_xy": query_normal.astype(float).tolist(),
            "history_route_arc_m": float(history_arc),
            "route_source": "traversed_history",
            "mode": "free_endpoint_swept_contact",
        }
        score = (
            approach_distance,
            float(outside_offset),
            -alignment,
            int(segment_index),
        )
        if best is None or score < best[0]:
            best = score, candidate
        cumulative += length
    return None if best is None else best[1]


def _thin_barrier_changed_pose_route(
    path_xy_m: Sequence[Sequence[float]],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    centroid_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_min_m: float,
    tangent_max_m: float,
    *,
    free_is_low: bool,
    robot_radius_m: float,
) -> dict[str, Any] | None:
    """Route around a free leaf end after the leaf changed its pose.

    A taught trajectory can cross the *current* door plane through its middle
    when the leaf was previously open at another angle.  The old trajectory is
    evidence that the two sides form one passage, not a template for the
    leaf's current geometry.  Require the historical swept footprint to contact
    the current panel, then construct a new local sweep outside the currently
    observed free endpoint.  The final oriented-footprint planner still has to
    certify this new sweep.
    """

    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    query_normal = np.asarray(normal_xy, dtype=np.float64).copy()
    start_normal = float((start_xy - centroid_xy) @ query_normal)
    goal_normal = float((goal_xy - centroid_xy) @ query_normal)
    if start_normal * goal_normal >= 0.0:
        return None
    if goal_normal < start_normal:
        query_normal *= -1.0

    cumulative = 0.0
    best: tuple[
        tuple[float, float, float, float, int], dict[str, Any]
    ] | None = None
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        alignment = float(direction @ query_normal)
        segment_start_normal = float((start - centroid_xy) @ query_normal)
        segment_end_normal = float((end - centroid_xy) @ query_normal)
        denominator = segment_end_normal - segment_start_normal
        if (
            alignment < .25
            or segment_start_normal > 0.0
            or segment_end_normal < 0.0
            or abs(denominator) <= 1.0e-12
        ):
            cumulative += length
            continue
        fraction = float(np.clip(-segment_start_normal / denominator, 0.0, 1.0))
        historical_crossing = start + fraction * delta
        historical_tangent = float(
            (historical_crossing - centroid_xy) @ tangent_xy
        )
        approach_distance = float(
            np.linalg.norm(historical_crossing - start_xy)
        )
        # The leaf may have rotated since the taught traversal.  In that case
        # the old footprint can pass just beyond either *current* endpoint;
        # requiring its centreline to pierce the current panel rejects the
        # exact case this recovery is meant to handle.  A swept-footprint
        # contact is sufficient here because the component must independently
        # be a thin vertical plane in mapped free space, one endpoint must have
        # absolute free/anchor topology, and the final current-depth planner
        # still certifies the newly constructed detour around that free end.
        historical_panel_distance = max(
            tangent_min_m - historical_tangent,
            historical_tangent - tangent_max_m,
            0.0,
        )
        if (
            historical_panel_distance
            > robot_radius_m + THIN_BARRIER_ROUTE_CONTACT_PAD_M
            or approach_distance > THIN_BARRIER_MAX_ROUTE_DISTANCE_M
        ):
            cumulative += length
            continue

        free_tangent = tangent_min_m if free_is_low else tangent_max_m
        outward_sign = -1.0 if free_is_low else 1.0
        detour_offset = min(
            THIN_BARRIER_FREE_END_MAX_CROSSING_OFFSET_M,
            robot_radius_m + THIN_BARRIER_ROUTE_CONTACT_PAD_M,
        )
        panel_endpoint = centroid_xy + free_tangent * tangent_xy
        detour_crossing = (
            panel_endpoint + outward_sign * detour_offset * tangent_xy
        )
        detour_approach = float(np.linalg.norm(detour_crossing - start_xy))
        if detour_approach > THIN_BARRIER_MAX_ROUTE_DISTANCE_M:
            cumulative += length
            continue
        history_arc = cumulative + fraction * length
        preferred = [
            (detour_crossing - THIN_BARRIER_HISTORY_SIDE_M * query_normal)
            .astype(float)
            .tolist(),
            detour_crossing.astype(float).tolist(),
            (detour_crossing + THIN_BARRIER_HISTORY_SIDE_M * query_normal)
            .astype(float)
            .tolist(),
        ]
        candidate = {
            "contact_xy_m": detour_crossing.astype(float).tolist(),
            "barrier_contact_xy_m": panel_endpoint.astype(float).tolist(),
            "contact_distance_m": float(detour_offset),
            "contact_arc_m": detour_approach,
            "contact_segment_index": int(segment_index),
            "free_end_distance_m": 0.0,
            "plane_crossing_xy_m": detour_crossing.astype(float).tolist(),
            "plane_crossing_arc_m": detour_approach,
            "plane_crossing_segment_index": int(segment_index),
            "plane_crossing_tangent_offset_m": float(detour_offset),
            "route_normal_alignment": alignment,
            "travel_normal_xy": query_normal.astype(float).tolist(),
            "history_route_arc_m": float(history_arc),
            "history_witness_xy_m": historical_crossing.astype(float).tolist(),
            "history_panel_crossing_xy_m": (
                historical_crossing.astype(float).tolist()
            ),
            "history_panel_crossing_tangent_m": historical_tangent,
            "history_panel_crossing_distance_m": float(
                historical_panel_distance
            ),
            "preferred_path_xy_m": preferred,
            "route_source": "traversed_history_changed_leaf_pose",
            "mode": "free_endpoint_swept_contact",
        }
        score = (
            detour_approach,
            historical_panel_distance,
            abs(historical_tangent - free_tangent),
            -alignment,
            int(segment_index),
        )
        if best is None or score < best[0]:
            best = score, candidate
        cumulative += length
    return None if best is None else best[1]


def _thin_barrier_reference_route_with_history_connectivity(
    reference_route: Mapping[str, Any] | None,
    path_xy_m: Sequence[Sequence[float]],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    centroid_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_min_m: float,
    tangent_max_m: float,
    *,
    free_is_low: bool,
    start_distance_m: float,
    goal_distance_m: float,
) -> dict[str, Any] | None:
    """Bind a current free-end route to an older, connected plane crossing.

    A door can rotate far enough that the old centreline no longer passes near
    either endpoint of the current leaf. The old route still proves passage
    connectivity, while the current map-only route identifies which observed
    endpoint is free. The caller separately requires mapped-free leaf evidence,
    clear endpoint context, a local history witness, and a fresh footprint plan.
    """

    if reference_route is None:
        return None
    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    if (
        not math.isfinite(float(start_distance_m))
        or not math.isfinite(float(goal_distance_m))
        or start_distance_m < 0.0
        or goal_distance_m < 0.0
        or start_distance_m > TRAVERSED_PREFERENCE_MAX_ENDPOINT_M
        or goal_distance_m > TRAVERSED_PREFERENCE_MAX_ENDPOINT_M
    ):
        return None

    query_normal = np.asarray(normal_xy, dtype=np.float64).copy()
    start_normal = float((start_xy - centroid_xy) @ query_normal)
    goal_normal = float((goal_xy - centroid_xy) @ query_normal)
    if start_normal * goal_normal >= 0.0:
        return None
    if goal_normal < start_normal:
        query_normal *= -1.0

    cumulative = 0.0
    best: tuple[tuple[float, float, float, int], dict[str, Any]] | None = None
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        alignment = float(direction @ query_normal)
        segment_start_normal = float((start - centroid_xy) @ query_normal)
        segment_end_normal = float((end - centroid_xy) @ query_normal)
        denominator = segment_end_normal - segment_start_normal
        if (
            alignment < .25
            or segment_start_normal > 0.0
            or segment_end_normal < 0.0
            or abs(denominator) <= 1.0e-12
        ):
            cumulative += length
            continue
        fraction = float(np.clip(-segment_start_normal / denominator, 0.0, 1.0))
        crossing = start + fraction * delta
        tangent_value = float((crossing - centroid_xy) @ tangent_xy)
        outside_offset = (
            tangent_min_m - tangent_value
            if free_is_low
            else tangent_value - tangent_max_m
        )
        approach_distance = float(np.linalg.norm(crossing - start_xy))
        if (
            outside_offset < -1.0e-9
            or outside_offset > THIN_BARRIER_FREE_END_MAX_CROSSING_OFFSET_M
            or approach_distance > THIN_BARRIER_MAX_ROUTE_DISTANCE_M
        ):
            cumulative += length
            continue
        history_arc = cumulative + fraction * length
        route = dict(reference_route)
        route.update({
            "route_source": "reference_plan_with_traversed_connectivity",
            "history_route_arc_m": float(history_arc),
            "history_witness_xy_m": crossing.astype(float).tolist(),
            "history_connectivity_crossing_xy_m": (
                crossing.astype(float).tolist()
            ),
            "history_connectivity_crossing_tangent_offset_m": float(
                outside_offset
            ),
            "history_connectivity_start_distance_m": float(start_distance_m),
            "history_connectivity_goal_distance_m": float(goal_distance_m),
        })
        score = (
            approach_distance,
            outside_offset,
            -alignment,
            int(segment_index),
        )
        if best is None or score < best[0]:
            best = score, route
        cumulative += length
    return None if best is None else best[1]


def _thin_barrier_robust_inlier_mask(component: np.ndarray) -> np.ndarray:
    """Extract one dominant vertical plane from a connected depth component.

    Door leaves commonly touch a jamb in the XY connectivity graph.  Iterative
    orthogonal trimming separates the dense planar leaf without using a scene
    axis or a task-specific pose.  A weak or small consensus is discarded.
    """

    points = np.asarray(component, dtype=np.float64).reshape(-1, 3)
    result = np.ones(len(points), dtype=bool)
    minimum = max(
        THIN_BARRIER_MIN_POINTS,
        int(math.ceil(THIN_BARRIER_ROBUST_MIN_INLIER_RATIO * len(points))),
    )
    if len(points) < minimum:
        return result
    for _ in range(8):
        selected = points[result]
        covariance = np.cov(selected.T)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return np.ones(len(points), dtype=bool)
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        centroid = np.median(selected, axis=0)
        residual = np.abs((points - centroid) @ eigenvectors[:, 0])
        updated = residual <= THIN_BARRIER_ROBUST_RESIDUAL_M
        if int(np.count_nonzero(updated)) < minimum:
            return np.ones(len(points), dtype=bool)
        if np.array_equal(updated, result):
            break
        result = updated
    return result


def _thin_barrier_vertical_plane_geometry(masked_points: np.ndarray) -> bool:
    points = np.asarray(masked_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < THIN_BARRIER_MIN_POINTS:
        return False
    covariance = np.cov(points.T)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return False
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return bool(
        math.sqrt(max(0.0, float(eigenvalues[0])))
        <= THIN_BARRIER_MAX_PLANE_STD_M
        and float(eigenvalues[0])
        <= .12 * max(float(eigenvalues[1]), 1.0e-9)
        and abs(float(eigenvectors[2, 0])) <= .25
    )


def _thin_barrier_connected_subplane_mask(
    component: np.ndarray,
) -> np.ndarray | None:
    """Extract a dominant vertical subplane from one XY-connected component.

    Candidate lines come from at most 96 occupied XY voxel centres, so the
    work is bounded independently of depth-image pixel count.  Raw point
    multiplicity supplies the consensus score after candidate generation.
    """

    points = np.asarray(component, dtype=np.float64).reshape(-1, 3)
    minimum = max(
        THIN_BARRIER_MIN_POINTS,
        int(math.ceil(
            THIN_BARRIER_CONNECTED_SUBPLANE_MIN_INLIER_RATIO * len(points)
        )),
    )
    if len(points) < minimum:
        return None
    voxel_keys = np.floor(
        points[:, :2] / THIN_BARRIER_CLUSTER_VOXEL_M
    ).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    if len(unique_keys) < 2:
        return None
    voxel_centres = np.zeros((len(unique_keys), 2), dtype=np.float64)
    voxel_counts = np.bincount(inverse, minlength=len(unique_keys)).astype(
        np.float64
    )
    np.add.at(voxel_centres, inverse, points[:, :2])
    voxel_centres /= voxel_counts[:, None]
    if len(voxel_centres) > THIN_BARRIER_CONNECTED_SUBPLANE_MAX_VOXEL_SEEDS:
        seed_indices = np.unique(np.linspace(
            0,
            len(voxel_centres) - 1,
            THIN_BARRIER_CONNECTED_SUBPLANE_MAX_VOXEL_SEEDS,
        ).astype(np.int64))
    else:
        seed_indices = np.arange(len(voxel_centres), dtype=np.int64)

    best: tuple[tuple[float, float, float, int, int], np.ndarray, np.ndarray] | None = None
    for offset, first_index in enumerate(seed_indices[:-1]):
        first = voxel_centres[first_index]
        for second_index in seed_indices[offset + 1:]:
            delta = voxel_centres[second_index] - first
            separation = float(np.linalg.norm(delta))
            if separation < THIN_BARRIER_CONNECTED_SUBPLANE_MIN_SEED_SPAN_M:
                continue
            normal = np.asarray([-delta[1], delta[0]], dtype=np.float64)
            normal /= separation
            residual = np.abs((voxel_centres - first) @ normal)
            voxel_inliers = residual <= THIN_BARRIER_ROBUST_RESIDUAL_M
            consensus = int(np.sum(voxel_counts[voxel_inliers]))
            if consensus < minimum:
                continue
            weighted_residual = float(
                np.average(residual[voxel_inliers], weights=voxel_counts[voxel_inliers])
            )
            tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
            tangent_values = voxel_centres[voxel_inliers] @ tangent
            tangent_span = float(np.ptp(tangent_values))
            score = (
                float(consensus),
                tangent_span,
                -weighted_residual,
                -int(first_index),
                -int(second_index),
            )
            if best is None or score > best[0]:
                best = score, first.copy(), normal
    if best is None:
        return None

    _score, origin_xy, normal_xy = best
    result = (
        np.abs((points[:, :2] - origin_xy) @ normal_xy)
        <= THIN_BARRIER_ROBUST_RESIDUAL_M
    )
    for _ in range(6):
        selected = points[result]
        if len(selected) < minimum:
            return None
        covariance = np.cov(selected.T)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return None
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        if abs(float(normal[2])) > .25:
            return None
        centroid = np.median(selected, axis=0)
        updated = (
            np.abs((points - centroid) @ normal)
            <= THIN_BARRIER_ROBUST_RESIDUAL_M
        )
        if int(np.count_nonzero(updated)) < minimum:
            return None
        if np.array_equal(updated, result):
            break
        result = updated
    return result if _thin_barrier_vertical_plane_geometry(points[result]) else None


def _thin_barrier_robust_fit(
    component: np.ndarray, *, allow_connected_subplane: bool,
) -> tuple[np.ndarray, str]:
    """Return the ordinary fit or a tightly gated connected-plane fallback."""

    points = np.asarray(component, dtype=np.float64).reshape(-1, 3)
    ordinary = _thin_barrier_robust_inlier_mask(points)
    if (
        _thin_barrier_vertical_plane_geometry(points[ordinary])
        or not allow_connected_subplane
    ):
        return ordinary, "iterative_orthogonal_pca"
    connected = _thin_barrier_connected_subplane_mask(points)
    if connected is None:
        return ordinary, "iterative_orthogonal_pca"
    return connected, "bounded_xy_vertical_plane_consensus"


def _thin_barrier_endpoint_context(
    context_xy_m: np.ndarray,
    endpoints_xy_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure endpoint context and its independent horizontal support.

    Plane fitting intentionally removes jamb and leaf-edge returns.  Those
    points remain useful endpoint context, but RGB-D supplies many heights for
    one image column.  Count connected XY voxels rather than raw 3-D returns so
    one noisy column cannot reverse the inferred free and attached ends.
    """

    points = np.asarray(context_xy_m, dtype=np.float64).reshape(-1, 2)
    endpoints = np.asarray(endpoints_xy_m, dtype=np.float64).reshape(-1, 2)
    empty_distances = np.full(len(endpoints), math.inf, dtype=np.float64)
    empty_support = np.zeros(len(endpoints), dtype=np.int64)
    if not len(points) or not np.all(np.isfinite(points)):
        return {
            "nearest_distance_m": empty_distances,
            "nearest_support_cells": empty_support,
            "supported_distance_m": empty_distances.copy(),
            "supported_support_cells": empty_support.copy(),
        }

    voxel_keys = np.floor(
        points / THIN_BARRIER_CLUSTER_VOXEL_M
    ).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_centres = np.zeros((len(unique_keys), 2), dtype=np.float64)
    voxel_counts = np.bincount(inverse, minlength=len(unique_keys)).astype(
        np.float64
    )
    np.add.at(voxel_centres, inverse, points)
    voxel_centres /= voxel_counts[:, None]

    voxel_tree = cKDTree(voxel_centres)
    neighbour_lists = voxel_tree.query_ball_point(
        voxel_centres, r=THIN_BARRIER_CLUSTER_LINK_M
    )
    edge_count = sum(len(items) for items in neighbour_lists)
    owners = np.empty(edge_count, dtype=np.int64)
    neighbours = np.empty(edge_count, dtype=np.int64)
    offset = 0
    for owner, items in enumerate(neighbour_lists):
        count = len(items)
        owners[offset : offset + count] = owner
        neighbours[offset : offset + count] = items
        offset += count
    graph = coo_matrix(
        (np.ones(edge_count, dtype=np.uint8), (owners, neighbours)),
        shape=(len(voxel_centres), len(voxel_centres)),
    )
    _component_count, voxel_labels = connected_components(
        graph, directed=False
    )
    component_sizes = np.bincount(voxel_labels)
    point_support = component_sizes[voxel_labels[inverse]]

    point_tree = cKDTree(points)
    nearest_distance, nearest_index = point_tree.query(endpoints, k=1)
    nearest_support = point_support[np.asarray(nearest_index, dtype=np.int64)]
    supported = (
        point_support >= THIN_BARRIER_ENDPOINT_CONTEXT_MIN_CELLS
    )
    supported_distance = empty_distances.copy()
    supported_support = empty_support.copy()
    if np.any(supported):
        supported_indices = np.flatnonzero(supported)
        supported_tree = cKDTree(points[supported])
        supported_distance, local_index = supported_tree.query(endpoints, k=1)
        selected = supported_indices[np.asarray(local_index, dtype=np.int64)]
        supported_support = point_support[selected].astype(np.int64)
    return {
        "nearest_distance_m": np.asarray(nearest_distance, dtype=np.float64),
        "nearest_support_cells": np.asarray(nearest_support, dtype=np.int64),
        "supported_distance_m": np.asarray(
            supported_distance, dtype=np.float64
        ),
        "supported_support_cells": np.asarray(
            supported_support, dtype=np.int64
        ),
    }


def _thin_barrier_endpoint_topology_mode(
    free_distance_m: float,
    anchor_distance_m: float,
    tangent_span_m: float,
    route_source: str,
) -> str | None:
    """Validate one free/anchored endpoint assignment.

    The absolute-gap rule remains the default.  The asymmetric rule exists for
    a common RGB-D failure mode: robust fitting trims the leaf edge, then those
    same edge returns become the nearest "external" context.  Only a
    query-relevant, previously traversed sweep may use this weaker rule.
    """

    values = (free_distance_m, anchor_distance_m, tangent_span_m)
    if not all(math.isfinite(float(value)) and value >= 0.0 for value in values):
        return None
    if anchor_distance_m > THIN_BARRIER_ANCHOR_MAX_DISTANCE_M:
        return None
    if free_distance_m >= THIN_BARRIER_FREE_END_MIN_DISTANCE_M:
        return "absolute_clearance"
    if (
        route_source != "traversed_history"
        or tangent_span_m > THIN_BARRIER_HISTORY_MAX_ASYMMETRIC_SPAN_M
        or free_distance_m < THIN_BARRIER_HISTORY_FREE_END_MIN_DISTANCE_M
        or free_distance_m - anchor_distance_m
        < THIN_BARRIER_HISTORY_ENDPOINT_MARGIN_M
        or free_distance_m
        < THIN_BARRIER_HISTORY_ENDPOINT_RATIO * anchor_distance_m
    ):
        return None
    return "history_asymmetric_context"


def _thin_barrier_changed_pose_map_evidence(
    snapshot: Mapping[str, Any], component_xy_m: np.ndarray
) -> dict[str, Any]:
    """Measure whether a live leaf occupies previously mapped free cells."""

    rejected = {
        "qualified": False,
        "component_cell_count": 0,
        "mapped_free_cell_count": 0,
        "mapped_wall_cell_count": 0,
        "mapped_free_ratio": 0.0,
        "mapped_wall_ratio": 1.0,
    }
    try:
        obstacle, free, resolution, origin = _snapshot_geometry(snapshot)
        wall = np.asarray(
            snapshot.get("wall_mask", np.zeros_like(obstacle)), dtype=bool
        )
        points = np.asarray(component_xy_m, dtype=np.float64).reshape(-1, 2)
    except (KeyError, TypeError, ValueError, NavigationPlanError):
        return rejected
    if wall.shape != obstacle.shape or not len(points):
        return rejected
    cells = np.floor((points - np.asarray(origin)) / resolution).astype(np.int64)
    valid = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < obstacle.shape[1])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < obstacle.shape[0])
    )
    if not np.any(valid):
        return rejected
    cells = np.unique(cells[valid], axis=0)
    rows, columns = cells[:, 1], cells[:, 0]
    count = int(len(cells))
    free_count = int(np.count_nonzero(free[rows, columns]))
    wall_count = int(np.count_nonzero(wall[rows, columns]))
    free_ratio = float(free_count / count)
    wall_ratio = float(wall_count / count)
    return {
        "qualified": bool(
            free_count >= THIN_BARRIER_CHANGED_POSE_MIN_FREE_CELLS
            and free_ratio >= THIN_BARRIER_CHANGED_POSE_MIN_FREE_RATIO
            and wall_ratio <= THIN_BARRIER_CHANGED_POSE_MAX_WALL_RATIO
        ),
        "component_cell_count": count,
        "mapped_free_cell_count": free_count,
        "mapped_wall_cell_count": wall_count,
        "mapped_free_ratio": free_ratio,
        "mapped_wall_ratio": wall_ratio,
    }


def _thin_barrier_route_crossing(
    path_xy_m: Sequence[Sequence[float]],
    centroid_xy: np.ndarray,
    normal_xy: np.ndarray,
    tangent_xy: np.ndarray,
    tangent_min_m: float,
    tangent_max_m: float,
) -> dict[str, Any] | None:
    """Find a near-future reference-path segment crossing a candidate plane."""

    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    cumulative = 0.0
    best: tuple[tuple[float, float, int], dict[str, Any]] | None = None
    tangent_pad = .08
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        alignment = abs(float(direction @ normal_xy))
        if alignment < THIN_BARRIER_MIN_ROUTE_NORMAL_DOT:
            cumulative += length
            continue
        start_normal = float((start - centroid_xy) @ normal_xy)
        end_normal = float((end - centroid_xy) @ normal_xy)
        denominator = end_normal - start_normal
        fraction = (
            float(np.clip(-start_normal / denominator, 0.0, 1.0))
            if abs(denominator) > 1.0e-12
            else 0.0
        )
        crossing = start + fraction * delta
        plane_error = abs(float((crossing - centroid_xy) @ normal_xy))
        tangent_value = float((crossing - centroid_xy) @ tangent_xy)
        endpoint_distance = min(
            abs(tangent_value - tangent_min_m),
            abs(tangent_value - tangent_max_m),
        )
        route_arc = cumulative + fraction * length
        if (
            plane_error > THIN_BARRIER_MAX_NORMAL_SPAN_M
            or tangent_value < tangent_min_m - tangent_pad
            or tangent_value > tangent_max_m + tangent_pad
            or endpoint_distance > THIN_BARRIER_DIRECT_MAX_ENDPOINT_DISTANCE_M
            or route_arc > THIN_BARRIER_MAX_ROUTE_DISTANCE_M
        ):
            cumulative += length
            continue
        oriented_normal = normal_xy.copy()
        if float(direction @ oriented_normal) < 0.0:
            oriented_normal *= -1.0
        candidate = {
            "crossing_xy_m": crossing.astype(float).tolist(),
            "reference_segment_index": int(segment_index),
            "route_arc_m": float(route_arc),
            "route_normal_alignment": float(alignment),
            "route_endpoint_distance_m": float(endpoint_distance),
            "travel_normal_xy": oriented_normal.astype(float).tolist(),
            "route_source": "reference_plan",
        }
        score = (route_arc, plane_error, segment_index)
        if best is None or score < best[0]:
            best = score, candidate
        cumulative += length
    return None if best is None else best[1]


def detect_traversed_thin_barrier(
    snapshot: Mapping[str, Any],
    reference_plan: Mapping[str, Any],
    points_robot_xyz: np.ndarray,
    *,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
    lower_image_boundary_mask: np.ndarray | None = None,
    verified_stall_xy_m: Sequence[float] | None = None,
    allow_censored_low_extent: bool = False,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Recognize a bounded movable leaf across a previously swept passage.

    This high-specificity recovery requires a dense, bounded, thin vertical
    plane; a near-future path crossing close to the plane normal; and one
    continuous historical base trajectory with observations on both sides.
    The result is private planning evidence, not a mapper-grid mutation.
    """

    def note(stage: str, **details: Any) -> None:
        if diagnostics is not None:
            diagnostics.append({"stage": stage, **details})

    if reference_plan.get("ok") is not True:
        note(
            "reference_rejected",
            plan_ok=False,
        )
        return None
    try:
        radius = _finite(robot_radius_m, "robot_radius_m")
        pose_x, pose_y, pose_yaw = _pose(snapshot)
        paths = _traversed_paths(snapshot)
        reference_path = reference_plan["path_xy_m"]
        query_start_xy = np.asarray(
            _pair(reference_path[0], "reference path start"), dtype=np.float64
        )
        query_goal_xy = np.asarray(
            _pair(reference_path[-1], "reference path goal"), dtype=np.float64
        )
        points = np.asarray(points_robot_xyz, dtype=np.float64).reshape(-1, 3)
        if lower_image_boundary_mask is None:
            lower_boundary = np.zeros(len(points), dtype=bool)
        else:
            lower_boundary = np.asarray(
                lower_image_boundary_mask, dtype=bool
            ).reshape(-1)
            if len(lower_boundary) != len(points):
                raise NavigationPlanError(
                    "lower image-boundary mask must match depth points"
                )
        stall_xy = (
            None
            if verified_stall_xy_m is None
            else np.asarray(
                _pair(verified_stall_xy_m, "verified stall"),
                dtype=np.float64,
            )
        )
    except (KeyError, TypeError, ValueError, NavigationPlanError) as exc:
        note("input_rejected", error=str(exc))
        return None
    if radius <= 0.0 or not paths or len(points) < THIN_BARRIER_MIN_POINTS:
        note(
            "input_rejected",
            robot_radius_m=float(radius),
            traversed_path_count=len(paths),
            point_count=int(len(points)),
        )
        return None
    history_route = _traversed_route_between(
        paths,
        tuple(query_start_xy),
        tuple(query_goal_xy),
        TRAVERSED_PREFERENCE_MAX_ENDPOINT_M,
    )
    note(
        "history_route",
        planner_used_traversed_evidence=bool(
            reference_plan.get("traversed_route_evidence_used", False)
        ),
        continuous_history_route_available=history_route is not None,
        local_barrier_history_required=True,
        start_distance_m=(None if history_route is None else history_route[1]),
        goal_distance_m=(None if history_route is None else history_route[2]),
    )
    local_history_paths: list[Sequence[Sequence[float]]] = []
    if history_route is not None:
        local_history_paths.append(history_route[0])
    # Door connectivity is local evidence. A previous traversal need not end
    # at today's marked destination, and a passage crossed in the opposite
    # direction is equally valid. Spatial plane association and the separate
    # before/after witness below keep unrelated trajectory portions out.
    for continuous_path in _continuous_traversed_components(paths):
        local_history_paths.append(continuous_path)
        local_history_paths.append(list(reversed(continuous_path)))
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    lower_boundary = lower_boundary[finite]
    original_indices = np.flatnonzero(finite)
    nearby = (
        np.linalg.norm(points[:, :2], axis=1)
        <= THIN_BARRIER_MAX_ROUTE_DISTANCE_M + radius
    )
    points = points[nearby]
    lower_boundary = lower_boundary[nearby]
    original_indices = original_indices[nearby]
    if len(points) < THIN_BARRIER_MIN_POINTS:
        note("range_rejected", nearby_point_count=int(len(points)))
        return None

    # Raw depth contains many returns with the same horizontal coordinate but
    # different heights.  A point-level k-nearest graph spends all neighbours
    # within those dense vertical columns and fragments one physical leaf into
    # narrow strips.  Collapse only for connectivity; all geometry below still
    # uses the original 3-D returns.
    voxel_keys = np.floor(
        points[:, :2] / THIN_BARRIER_CLUSTER_VOXEL_M
    ).astype(np.int64)
    _unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_centres = np.zeros((len(_unique_keys), 2), dtype=np.float64)
    voxel_counts = np.bincount(inverse, minlength=len(_unique_keys)).astype(
        np.float64
    )
    np.add.at(voxel_centres, inverse, points[:, :2])
    voxel_centres /= voxel_counts[:, None]
    tree = cKDTree(voxel_centres)
    neighbour_lists = tree.query_ball_point(
        voxel_centres, r=THIN_BARRIER_CLUSTER_LINK_M
    )
    edge_count = sum(len(items) for items in neighbour_lists)
    owners = np.empty(edge_count, dtype=np.int64)
    neighbours = np.empty(edge_count, dtype=np.int64)
    offset = 0
    for owner, items in enumerate(neighbour_lists):
        count = len(items)
        owners[offset : offset + count] = owner
        neighbours[offset : offset + count] = items
        offset += count
    graph = coo_matrix(
        (np.ones(edge_count, dtype=np.uint8), (owners, neighbours)),
        shape=(len(voxel_centres), len(voxel_centres)),
    )
    component_count, voxel_labels = connected_components(graph, directed=False)
    labels = voxel_labels[inverse]
    rotation = np.array(
        [[math.cos(pose_yaw), -math.sin(pose_yaw)],
         [math.sin(pose_yaw), math.cos(pose_yaw)]],
        dtype=np.float64,
    )
    points_map_xy = points[:, :2] @ rotation.T + [pose_x, pose_y]
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    note(
        "components",
        nearby_point_count=int(len(points)),
        component_count=int(component_count),
    )

    for label in range(component_count):
        raw_component_mask = labels == label
        raw_component = points[raw_component_mask]
        if len(raw_component) < THIN_BARRIER_MIN_POINTS:
            note(
                "component_rejected",
                component_label=int(label),
                reason="point_count",
                point_count=int(len(raw_component)),
            )
            continue
        local_inliers, robust_fit_method = _thin_barrier_robust_fit(
            raw_component,
            allow_connected_subplane=bool(
                allow_censored_low_extent
                and stall_xy is not None
                and history_route is not None
            ),
        )
        raw_indices = np.flatnonzero(raw_component_mask)
        component_mask = np.zeros(len(points), dtype=bool)
        component_mask[raw_indices[local_inliers]] = True
        component = points[component_mask]
        robust_inlier_ratio = float(len(component) / len(raw_component))
        covariance = np.cov(component.T)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            note(
                "component_rejected",
                component_label=int(label),
                reason="covariance",
                point_count=int(len(component)),
            )
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        plane_std = math.sqrt(max(0.0, float(eigenvalues[0])))
        normal_robot = eigenvectors[:, 0]
        raw_plane_residual = np.abs(
            (raw_component - np.median(component, axis=0)) @ normal_robot
        )
        raw_thin_slab_ratio = float(np.mean(
            raw_plane_residual
            <= THIN_BARRIER_CLEAR_END_MAX_SLAB_RESIDUAL_M
        ))
        if (
            plane_std > THIN_BARRIER_MAX_PLANE_STD_M
            or float(eigenvalues[0]) > .12 * max(float(eigenvalues[1]), 1.0e-9)
            or abs(float(normal_robot[2])) > .25
        ):
            note(
                "component_rejected",
                component_label=int(label),
                reason="plane_geometry",
                point_count=int(len(component)),
                raw_component_point_count=int(len(raw_component)),
                robust_inlier_ratio=robust_inlier_ratio,
                robust_fit_method=robust_fit_method,
                plane_std_m=float(plane_std),
                planar_eigen_ratio=float(
                    float(eigenvalues[0])
                    / max(float(eigenvalues[1]), 1.0e-9)
                ),
                vertical_normal_abs=float(abs(float(normal_robot[2]))),
            )
            continue
        normal_xy_robot = normal_robot[:2]
        normal_norm = float(np.linalg.norm(normal_xy_robot))
        if normal_norm <= 1.0e-9:
            note(
                "component_rejected",
                component_label=int(label),
                reason="horizontal_plane_normal",
                point_count=int(len(component)),
            )
            continue
        normal_xy = normal_xy_robot / normal_norm @ rotation.T
        tangent_xy = np.asarray([-normal_xy[1], normal_xy[0]], dtype=np.float64)
        component_map = points_map_xy[component_mask]
        centroid_xy = np.median(component_map, axis=0)
        tangent_coordinate = (component_map - centroid_xy) @ tangent_xy
        normal_coordinate = (component_map - centroid_xy) @ normal_xy
        tangent_low, tangent_high = np.quantile(tangent_coordinate, [.02, .98])
        normal_low, normal_high = np.quantile(normal_coordinate, [.02, .98])
        z_low, z_high = np.quantile(component[:, 2], [.02, .98])
        tangent_span = float(tangent_high - tangent_low)
        normal_span = float(normal_high - normal_low)
        vertical_span = float(z_high - z_low)
        low_point_count = int(np.count_nonzero(component[:, 2] <= .50))
        component_lower_boundary = lower_boundary[component_mask]
        lower_boundary_point_count = int(np.count_nonzero(
            component_lower_boundary
        ))
        lower_boundary_min_z = (
            float(np.min(component[component_lower_boundary, 2]))
            if lower_boundary_point_count
            else math.inf
        )
        stall_plane_distance = (
            abs(float((stall_xy - centroid_xy) @ normal_xy))
            if stall_xy is not None
            else math.inf
        )
        stall_centroid_distance = (
            float(np.linalg.norm(stall_xy - centroid_xy))
            if stall_xy is not None
            else math.inf
        )
        observed_low_extent = bool(
            float(z_low) <= THIN_BARRIER_MAX_LOW_Z_M
            and low_point_count >= 8
        )
        censored_low_extent = bool(
            allow_censored_low_extent
            and stall_xy is not None
            and vertical_span
            >= THIN_BARRIER_CENSORED_MIN_VERTICAL_SPAN_M
            and float(z_high) >= THIN_BARRIER_CENSORED_MIN_HIGH_Z_M
            and float(z_low) <= THIN_BARRIER_CENSORED_MAX_LOW_Z_M
            and lower_boundary_point_count
            >= THIN_BARRIER_CENSORED_MIN_BOUNDARY_POINTS
            and lower_boundary_min_z <= float(z_low) + .05
            and stall_plane_distance
            <= THIN_BARRIER_CENSORED_MAX_STALL_PLANE_DISTANCE_M
            and stall_centroid_distance
            <= THIN_BARRIER_CENSORED_MAX_STALL_CENTROID_DISTANCE_M
        )
        stall_history_censored_low_extent = bool(
            allow_censored_low_extent
            and stall_xy is not None
            and history_route is not None
            and not observed_low_extent
            and not censored_low_extent
            and vertical_span
            >= THIN_BARRIER_STALL_HISTORY_MIN_VERTICAL_SPAN_M
            and float(z_high) >= THIN_BARRIER_STALL_HISTORY_MIN_HIGH_Z_M
            and float(z_low) <= THIN_BARRIER_STALL_HISTORY_MAX_LOW_Z_M
            and plane_std <= THIN_BARRIER_STALL_HISTORY_MAX_PLANE_STD_M
            and (
                (
                    robust_fit_method == "iterative_orthogonal_pca"
                    and robust_inlier_ratio
                    >= THIN_BARRIER_STALL_HISTORY_MIN_ROBUST_INLIER_RATIO
                    and raw_thin_slab_ratio
                    >= THIN_BARRIER_STALL_HISTORY_MIN_THIN_SLAB_RATIO
                )
                or (
                    robust_fit_method
                    == "bounded_xy_vertical_plane_consensus"
                    and robust_inlier_ratio
                    >= THIN_BARRIER_CONNECTED_SUBPLANE_MIN_INLIER_RATIO
                    and raw_thin_slab_ratio
                    >= THIN_BARRIER_CONNECTED_SUBPLANE_MIN_RAW_SLAB_RATIO
                )
            )
            and stall_plane_distance
            <= THIN_BARRIER_STALL_HISTORY_MAX_STALL_PLANE_DISTANCE_M
            and stall_centroid_distance
            <= THIN_BARRIER_STALL_HISTORY_MAX_STALL_CENTROID_DISTANCE_M
        )
        if (
            tangent_span < THIN_BARRIER_MIN_TANGENT_SPAN_M
            or tangent_span > THIN_BARRIER_MAX_FREE_END_TANGENT_SPAN_M
            or normal_span > THIN_BARRIER_MAX_NORMAL_SPAN_M
            or vertical_span < THIN_BARRIER_MIN_VERTICAL_SPAN_M
            or float(z_high) < THIN_BARRIER_MIN_HIGH_Z_M
            or not (
                observed_low_extent
                or censored_low_extent
                or stall_history_censored_low_extent
            )
        ):
            note(
                "component_rejected",
                component_label=int(label),
                reason="door_extent",
                point_count=int(len(component)),
                raw_component_point_count=int(len(raw_component)),
                robust_inlier_ratio=robust_inlier_ratio,
                robust_fit_method=robust_fit_method,
                tangent_span_m=tangent_span,
                normal_span_m=normal_span,
                vertical_span_m=vertical_span,
                z_low_m=float(z_low),
                z_high_m=float(z_high),
                low_point_count=low_point_count,
                observed_low_extent=observed_low_extent,
                censored_low_extent=censored_low_extent,
                stall_history_censored_low_extent=(
                    stall_history_censored_low_extent
                ),
                lower_image_boundary_point_count=lower_boundary_point_count,
                lower_image_boundary_min_z_m=(
                    None
                    if not math.isfinite(lower_boundary_min_z)
                    else lower_boundary_min_z
                ),
                verified_stall_plane_distance_m=(
                    None
                    if not math.isfinite(stall_plane_distance)
                    else stall_plane_distance
                ),
                verified_stall_centroid_distance_m=(
                    None
                    if not math.isfinite(stall_centroid_distance)
                    else stall_centroid_distance
                ),
                plane_std_m=float(plane_std),
            )
            continue
        changed_pose_map_evidence = _thin_barrier_changed_pose_map_evidence(
            snapshot, component_map
        )
        stall_history_map_evidence = bool(
            changed_pose_map_evidence["qualified"]
            and changed_pose_map_evidence["mapped_free_ratio"]
            >= THIN_BARRIER_STALL_HISTORY_MIN_MAP_FREE_RATIO
            and changed_pose_map_evidence["mapped_wall_ratio"]
            <= THIN_BARRIER_STALL_HISTORY_MAX_MAP_WALL_RATIO
        )
        if (
            stall_history_censored_low_extent
            and not stall_history_map_evidence
        ):
            note(
                "component_rejected",
                component_label=int(label),
                reason="stall_history_map_evidence",
                changed_pose_map_evidence=changed_pose_map_evidence,
            )
            continue
        route = _thin_barrier_route_crossing(
            reference_path,
            centroid_xy,
            normal_xy,
            tangent_xy,
            float(tangent_low),
            float(tangent_high),
        )
        if stall_history_censored_low_extent:
            # This cropped observation may authorize only a newly constructed
            # route around today's free endpoint, never an exception through
            # the middle of the observed plane.
            route = None
        if (
            route is not None
            and tangent_span > THIN_BARRIER_MAX_TANGENT_SPAN_M
        ):
            # Extended-width planes must establish an anchored end, a free end
            # and a taught sweep around that end.  Do not authorize them from
            # proximity to a reference-polyline endpoint alone.
            route = None
        route_mode = "direct_plane_crossing"
        anchor_metadata: dict[str, Any] = {}
        if route is None:
            other_points = points_map_xy[~component_mask]
            if not len(other_points):
                note(
                    "component_rejected",
                    component_label=int(label),
                    reason="no_anchor_context",
                )
                continue
            low_endpoint = centroid_xy + float(tangent_low) * tangent_xy
            high_endpoint = centroid_xy + float(tangent_high) * tangent_xy
            endpoint_context = _thin_barrier_endpoint_context(
                other_points, np.stack((low_endpoint, high_endpoint))
            )
            # Robust-plane outliers can be real jamb returns, so the strict
            # anchored-end modes continue to inspect them.  They cannot prove
            # that *both* ends are clear: leaf thickness and depth-edge noise
            # are part of the same raw connected component.  The clear-end
            # fallback therefore measures only independent XY components.
            independent_endpoint_context = _thin_barrier_endpoint_context(
                points_map_xy[~raw_component_mask],
                np.stack((low_endpoint, high_endpoint)),
            )
            low_anchor_distance, high_anchor_distance = endpoint_context[
                "nearest_distance_m"
            ]
            low_anchor_support, high_anchor_support = endpoint_context[
                "nearest_support_cells"
            ]
            low_supported_distance, high_supported_distance = endpoint_context[
                "supported_distance_m"
            ]
            low_supported_support, high_supported_support = endpoint_context[
                "supported_support_cells"
            ]
            (
                low_independent_distance,
                high_independent_distance,
            ) = independent_endpoint_context["supported_distance_m"]
            (
                low_independent_support,
                high_independent_support,
            ) = independent_endpoint_context["supported_support_cells"]
            (
                low_independent_nearest_distance,
                high_independent_nearest_distance,
            ) = independent_endpoint_context["nearest_distance_m"]
            (
                low_independent_nearest_support,
                high_independent_nearest_support,
            ) = independent_endpoint_context["nearest_support_cells"]
            endpoint_options = []
            endpoint_diagnostics = []
            for (
                free_is_low,
                free_distance,
                anchor_distance,
                free_context_support,
                anchor_context_support,
                supported_free_distance,
                supported_anchor_distance,
                supported_free_support,
                supported_anchor_support,
                independent_free_distance,
                independent_anchor_distance,
                independent_free_support,
                independent_anchor_support,
                independent_free_nearest_distance,
                independent_anchor_nearest_distance,
                independent_free_nearest_support,
                independent_anchor_nearest_support,
            ) in (
                (
                    True,
                    float(low_anchor_distance),
                    float(high_anchor_distance),
                    int(low_anchor_support),
                    int(high_anchor_support),
                    float(low_supported_distance),
                    float(high_supported_distance),
                    int(low_supported_support),
                    int(high_supported_support),
                    float(low_independent_distance),
                    float(high_independent_distance),
                    int(low_independent_support),
                    int(high_independent_support),
                    float(low_independent_nearest_distance),
                    float(high_independent_nearest_distance),
                    int(low_independent_nearest_support),
                    int(high_independent_nearest_support),
                ),
                (
                    False,
                    float(high_anchor_distance),
                    float(low_anchor_distance),
                    int(high_anchor_support),
                    int(low_anchor_support),
                    float(high_supported_distance),
                    float(low_supported_distance),
                    int(high_supported_support),
                    int(low_supported_support),
                    float(high_independent_distance),
                    float(low_independent_distance),
                    int(high_independent_support),
                    int(low_independent_support),
                    float(high_independent_nearest_distance),
                    float(low_independent_nearest_distance),
                    int(high_independent_nearest_support),
                    int(low_independent_nearest_support),
                ),
            ):
                reference_route = _thin_barrier_free_end_route(
                    reference_path,
                    centroid_xy,
                    normal_xy,
                    tangent_xy,
                    float(tangent_low),
                    float(tangent_high),
                    free_is_low=free_is_low,
                    robot_radius_m=radius,
                )
                historical_route = None
                changed_pose_route = None
                connectivity_route = None
                for local_history_path in local_history_paths:
                    candidate_historical = (
                        _thin_barrier_historical_free_end_route(
                            local_history_path,
                            query_start_xy,
                            query_goal_xy,
                            centroid_xy,
                            normal_xy,
                            tangent_xy,
                            float(tangent_low),
                            float(tangent_high),
                            free_is_low=free_is_low,
                            robot_radius_m=radius,
                        )
                    )
                    if (
                        candidate_historical is not None
                        and (
                            historical_route is None
                            or (
                                float(candidate_historical["contact_arc_m"]),
                                -float(candidate_historical[
                                    "route_normal_alignment"
                                ]),
                            )
                            < (
                                float(historical_route["contact_arc_m"]),
                                -float(historical_route[
                                    "route_normal_alignment"
                                ]),
                            )
                        )
                    ):
                        historical_route = candidate_historical
                    if changed_pose_map_evidence["qualified"]:
                        candidate_changed = _thin_barrier_changed_pose_route(
                            local_history_path,
                            query_start_xy,
                            query_goal_xy,
                            centroid_xy,
                            normal_xy,
                            tangent_xy,
                            float(tangent_low),
                            float(tangent_high),
                            free_is_low=free_is_low,
                            robot_radius_m=radius,
                        )
                        if (
                            candidate_changed is not None
                            and (
                                changed_pose_route is None
                                or (
                                    float(candidate_changed["contact_arc_m"]),
                                    -float(candidate_changed[
                                        "route_normal_alignment"
                                    ]),
                                )
                                < (
                                    float(changed_pose_route["contact_arc_m"]),
                                    -float(changed_pose_route[
                                        "route_normal_alignment"
                                    ]),
                                )
                            )
                        ):
                            changed_pose_route = candidate_changed
                if history_route is not None:
                    # Full query-to-goal connectivity remains a stronger
                    # optional proof for binding a current reference route.
                    # Local history above is sufficient for the leaf itself.
                    if changed_pose_map_evidence["qualified"]:
                        connectivity_route = (
                            _thin_barrier_reference_route_with_history_connectivity(
                                reference_route,
                                history_route[0],
                                query_start_xy,
                                query_goal_xy,
                                centroid_xy,
                                normal_xy,
                                tangent_xy,
                                float(tangent_low),
                                float(tangent_high),
                                free_is_low=free_is_low,
                                start_distance_m=history_route[1],
                                goal_distance_m=history_route[2],
                            )
                        )
                # Endpoint topology is a horizontal-structure claim.  The
                # raw nearest return is useful diagnostics, but a one- or
                # two-cell depth-edge fragment must not close an otherwise
                # supported free end.  Use the nearest *supported component*
                # consistently for both ends; this is the same evidence used
                # to decide whether an anchor exists at all.
                selected_free_distance = supported_free_distance
                selected_anchor_distance = supported_anchor_distance
                selected_free_support = supported_free_support
                selected_anchor_support = supported_anchor_support
                endpoint_support_required = (
                    2
                    if (
                        censored_low_extent
                        or stall_history_censored_low_extent
                    )
                    else THIN_BARRIER_ENDPOINT_CONTEXT_MIN_CELLS
                )
                if (
                    (
                        censored_low_extent
                        or stall_history_censored_low_extent
                    )
                    and selected_free_support < endpoint_support_required
                    and free_context_support >= endpoint_support_required
                ):
                    selected_free_distance = free_distance
                    selected_free_support = free_context_support
                if (
                    (
                        censored_low_extent
                        or stall_history_censored_low_extent
                    )
                    and selected_anchor_support < endpoint_support_required
                    and anchor_context_support >= endpoint_support_required
                ):
                    selected_anchor_distance = anchor_distance
                    selected_anchor_support = anchor_context_support
                # An unsupported fragment trimmed from this same raw leaf can
                # be a depth edge.  An independent live-depth component is a
                # different object, however, and remains a conservative free-
                # end blocker even when it spans fewer than three XY cells.
                if (
                    independent_free_nearest_distance
                    < selected_free_distance
                ):
                    selected_free_distance = (
                        independent_free_nearest_distance
                    )
                    selected_free_support = independent_free_nearest_support
                anchor_is_supported = bool(
                    selected_anchor_support
                    >= endpoint_support_required
                    and math.isfinite(selected_anchor_distance)
                )
                strong_topology = (
                    _thin_barrier_endpoint_topology_mode(
                        selected_free_distance,
                        selected_anchor_distance,
                        tangent_span,
                        "reference_plan",
                    )
                    if anchor_is_supported
                    else None
                )
                candidate_route = reference_route or historical_route
                # A taught centreline is connectivity evidence, not proof that
                # the base still clears a leaf in its current pose.  When the
                # mapped-free evidence supports a changed leaf and the old
                # centreline would put the circular footprint into today's
                # panel, execute the freshly constructed free-end detour.
                # This applies before the first collision as well as on a
                # verified-stall retry.
                candidate_contacts_current_leaf = bool(
                    candidate_route is not None
                    and float(candidate_route["contact_distance_m"])
                    < radius + THIN_BARRIER_ROUTE_CONTACT_PAD_M - 1.0e-9
                )
                candidate_is_taught_contact = bool(
                    candidate_route is not None
                    and candidate_route.get("route_source")
                    == "traversed_history"
                    and candidate_contacts_current_leaf
                )
                if (
                    strong_topology == "absolute_clearance"
                    and changed_pose_route is not None
                    and (
                        candidate_route is None
                        or candidate_is_taught_contact
                        or (
                            stall_xy is not None
                            and candidate_contacts_current_leaf
                        )
                    )
                ):
                    candidate_route = changed_pose_route
                topology_mode = None
                if candidate_route is not None and strong_topology is not None:
                    topology_mode = strong_topology
                elif historical_route is not None and anchor_is_supported:
                    topology_mode = _thin_barrier_endpoint_topology_mode(
                        selected_free_distance,
                        selected_anchor_distance,
                        tangent_span,
                        "traversed_history",
                    )
                # Prefer the route certified by the current static map.  An
                # older traversal proves connectivity, but a moved leaf can
                # put that old centreline beyond today's mapped-free opening.
                # The executable route is rebuilt against current depth in
                # either case; this ordering selects the reachable gate.
                clear_end_route = (
                    changed_pose_route
                    if (
                        stall_history_censored_low_extent
                        and changed_pose_route is not None
                    )
                    else (
                        connectivity_route
                        or historical_route
                        or changed_pose_route
                    )
                )
                clear_end_history = bool(
                    topology_mode is None
                    and clear_end_route is not None
                    and tangent_span
                    <= THIN_BARRIER_HISTORY_MAX_ASYMMETRIC_SPAN_M
                    and raw_thin_slab_ratio
                    >= THIN_BARRIER_CLEAR_END_MIN_SLAB_RATIO
                    and math.isfinite(independent_free_distance)
                    and math.isfinite(independent_anchor_distance)
                    and independent_free_distance
                    >= THIN_BARRIER_FREE_END_MIN_DISTANCE_M
                    and independent_anchor_distance
                    >= THIN_BARRIER_FREE_END_MIN_DISTANCE_M
                    and independent_free_support
                    >= THIN_BARRIER_ENDPOINT_CONTEXT_MIN_CELLS
                    and independent_anchor_support
                    >= THIN_BARRIER_ENDPOINT_CONTEXT_MIN_CELLS
                    and changed_pose_map_evidence["qualified"]
                    and changed_pose_map_evidence["mapped_free_ratio"]
                    >= THIN_BARRIER_CLEAR_END_MIN_FREE_RATIO
                    and changed_pose_map_evidence["mapped_wall_ratio"]
                    <= THIN_BARRIER_CLEAR_END_MAX_WALL_RATIO
                )
                if clear_end_history:
                    topology_mode = "history_mapped_free_clear_ends"
                    candidate_route = clear_end_route
                    selected_free_distance = independent_free_distance
                    selected_anchor_distance = independent_anchor_distance
                    selected_free_support = independent_free_support
                    selected_anchor_support = independent_anchor_support
                endpoint_diagnostics.append({
                    "free_is_low": bool(free_is_low),
                    "free_distance_m": float(free_distance),
                    "anchor_distance_m": float(anchor_distance),
                    "free_context_support_cells": int(free_context_support),
                    "anchor_context_support_cells": int(anchor_context_support),
                    "supported_free_distance_m": float(
                        supported_free_distance
                    ),
                    "supported_anchor_distance_m": float(
                        supported_anchor_distance
                    ),
                    "independent_free_distance_m": float(
                        independent_free_distance
                    ),
                    "independent_anchor_distance_m": float(
                        independent_anchor_distance
                    ),
                    "independent_free_nearest_distance_m": float(
                        independent_free_nearest_distance
                    ),
                    "independent_anchor_nearest_distance_m": float(
                        independent_anchor_nearest_distance
                    ),
                    "independent_free_support_cells": int(
                        independent_free_support
                    ),
                    "independent_anchor_support_cells": int(
                        independent_anchor_support
                    ),
                    "raw_thin_slab_ratio": raw_thin_slab_ratio,
                    "strong_topology": strong_topology,
                    "clear_end_history": clear_end_history,
                    "reference_free_end_route": reference_route is not None,
                    "historical_free_end_route": historical_route is not None,
                    "historical_changed_pose_route": changed_pose_route is not None,
                    "reference_history_connectivity_route": (
                        connectivity_route is not None
                    ),
                    "changed_pose_map_evidence": changed_pose_map_evidence,
                })
                if topology_mode is None:
                    continue
                if topology_mode == "history_asymmetric_context":
                    candidate_route = historical_route
                endpoint_options.append(
                    (
                        {
                            "absolute_clearance": 0,
                            "history_asymmetric_context": 1,
                            "history_mapped_free_clear_ends": 2,
                        }[topology_mode],
                        0
                        if str(candidate_route["route_source"]) in {
                            "reference_plan",
                            "reference_plan_with_traversed_connectivity",
                        }
                        else 1,
                        float(candidate_route["contact_arc_m"]),
                        free_is_low,
                        selected_free_distance,
                        selected_anchor_distance,
                        selected_free_support,
                        selected_anchor_support,
                        topology_mode,
                        candidate_route,
                    )
                )
            if not endpoint_options:
                note(
                    "component_rejected",
                    component_label=int(label),
                    reason="endpoint_topology",
                    point_count=int(len(component)),
                    raw_component_point_count=int(len(raw_component)),
                    robust_inlier_ratio=robust_inlier_ratio,
                    tangent_span_m=tangent_span,
                    centroid_xy_m=centroid_xy.astype(float).tolist(),
                    normal_xy=normal_xy.astype(float).tolist(),
                    tangent_xy=tangent_xy.astype(float).tolist(),
                    low_endpoint_xy_m=low_endpoint.astype(float).tolist(),
                    high_endpoint_xy_m=high_endpoint.astype(float).tolist(),
                    query_start_normal_m=float(
                        (query_start_xy - centroid_xy) @ normal_xy
                    ),
                    query_goal_normal_m=float(
                        (query_goal_xy - centroid_xy) @ normal_xy
                    ),
                    low_endpoint_neighbour_distance_m=float(low_anchor_distance),
                    high_endpoint_neighbour_distance_m=float(high_anchor_distance),
                    endpoint_candidates=endpoint_diagnostics,
                )
                continue
            (
                _topology_rank,
                _route_source_rank,
                _route_rank,
                free_is_low,
                free_distance,
                anchor_distance,
                free_context_support,
                anchor_context_support,
                endpoint_topology_mode,
                route,
            ) = min(endpoint_options, key=lambda item: item[:2])
            if free_is_low:
                free_endpoint = low_endpoint
                anchor_endpoint = high_endpoint
            else:
                free_endpoint = high_endpoint
                anchor_endpoint = low_endpoint
            if (
                stall_history_censored_low_extent
                and (
                    str(route.get("route_source"))
                    != "traversed_history_changed_leaf_pose"
                    or endpoint_topology_mode not in {
                        "absolute_clearance",
                        "history_mapped_free_clear_ends",
                    }
                )
            ):
                note(
                    "component_rejected",
                    component_label=int(label),
                    reason="stall_history_free_end_route",
                    route_source=str(route.get("route_source")),
                    endpoint_topology_mode=endpoint_topology_mode,
                )
                continue
            route_mode = str(route["mode"])
            anchor_metadata = {
                "free_endpoint_xy_m": free_endpoint.astype(float).tolist(),
                "anchor_endpoint_xy_m": anchor_endpoint.astype(float).tolist(),
                "free_endpoint_neighbour_distance_m": free_distance,
                "anchor_endpoint_neighbour_distance_m": anchor_distance,
                "free_endpoint_neighbour_support_cells": int(
                    free_context_support
                ),
                "anchor_endpoint_neighbour_support_cells": int(
                    anchor_context_support
                ),
                "endpoint_topology_mode": endpoint_topology_mode,
                "endpoint_context_source": (
                    "independent_components"
                    if endpoint_topology_mode
                    == "history_mapped_free_clear_ends"
                    else "all_non_plane_points"
                ),
                "route_contact_xy_m": list(route["contact_xy_m"]),
                "barrier_contact_xy_m": list(route["barrier_contact_xy_m"]),
                "route_contact_distance_m": float(route["contact_distance_m"]),
                "route_plane_crossing_xy_m": list(
                    route["plane_crossing_xy_m"]
                ),
                "route_plane_crossing_arc_m": float(
                    route["plane_crossing_arc_m"]
                ),
                "route_plane_crossing_tangent_offset_m": float(
                    route["plane_crossing_tangent_offset_m"]
                ),
                "route_source": str(route["route_source"]),
            }
            if "history_route_arc_m" in route:
                anchor_metadata["history_route_arc_m"] = float(
                    route["history_route_arc_m"]
                )
            if endpoint_topology_mode == "history_mapped_free_clear_ends":
                anchor_metadata["changed_pose_map_evidence"] = dict(
                    changed_pose_map_evidence
                )
            if "history_panel_crossing_xy_m" in route:
                anchor_metadata["history_panel_crossing_xy_m"] = list(
                    route["history_panel_crossing_xy_m"]
                )
                anchor_metadata["history_panel_crossing_tangent_m"] = float(
                    route["history_panel_crossing_tangent_m"]
                )
                anchor_metadata["history_panel_crossing_distance_m"] = float(
                    route.get("history_panel_crossing_distance_m", 0.0)
                )
                anchor_metadata["changed_pose_map_evidence"] = dict(
                    changed_pose_map_evidence
                )
            for key in (
                "history_connectivity_crossing_xy_m",
                "history_connectivity_crossing_tangent_offset_m",
                "history_connectivity_start_distance_m",
                "history_connectivity_goal_distance_m",
            ):
                if key in route:
                    value = route[key]
                    anchor_metadata[key] = (
                        list(value)
                        if key.endswith("_xy_m")
                        else float(value)
                    )
            route_crossing_xy = np.asarray(
                route["plane_crossing_xy_m"], dtype=np.float64
            )
            barrier_origin_xy = free_endpoint.copy()
            route_arc_m = float(route["contact_arc_m"])
            reference_segment_index = int(route["contact_segment_index"])
        else:
            route_crossing_xy = np.asarray(
                route["crossing_xy_m"], dtype=np.float64
            )
            barrier_origin_xy = route_crossing_xy.copy()
            route_arc_m = float(route["route_arc_m"])
            reference_segment_index = int(route["reference_segment_index"])
            anchor_metadata = {
                "route_endpoint_distance_m": float(
                    route["route_endpoint_distance_m"]
                ),
                "route_source": str(route["route_source"]),
            }
        travel_normal = np.asarray(route["travel_normal_xy"], dtype=np.float64)
        if float(travel_normal @ normal_xy) < 0.0:
            normal_xy *= -1.0
            tangent_xy *= -1.0
            tangent_low, tangent_high = -tangent_high, -tangent_low
        if "history_panel_crossing_xy_m" in anchor_metadata:
            # Canonicalize signed tangent metadata after the possible basis
            # flip above.  Reusing the pre-flip scalar made an otherwise valid
            # certificate depend on the arbitrary PCA eigenvector sign.
            historical_crossing = np.asarray(
                anchor_metadata["history_panel_crossing_xy_m"],
                dtype=np.float64,
            )
            historical_tangent = float(
                (historical_crossing - centroid_xy) @ tangent_xy
            )
            anchor_metadata["history_panel_crossing_tangent_m"] = (
                historical_tangent
            )
            anchor_metadata["history_panel_crossing_distance_m"] = float(max(
                float(tangent_low) - historical_tangent,
                historical_tangent - float(tangent_high),
                0.0,
            ))
        history_witness_xy = np.asarray(
            route.get("history_witness_xy_m", route_crossing_xy),
            dtype=np.float64,
        )
        history_tangent_limit = radius + .15
        if route.get("route_source") == "traversed_history_changed_leaf_pose":
            history_witness_tangent = float(
                (history_witness_xy - centroid_xy) @ tangent_xy
            )
            history_tangent_limit += max(
                abs(float(tangent_low) - history_witness_tangent),
                abs(float(tangent_high) - history_witness_tangent),
            )
        history = _thin_barrier_history_witness(
            paths,
            history_witness_xy,
            normal_xy,
            tangent_xy,
            history_tangent_limit,
        )
        if history is None:
            note(
                "component_rejected",
                component_label=int(label),
                reason="history_witness",
                route_mode=route_mode,
                tangent_span_m=tangent_span,
                normal_span_m=normal_span,
            )
            continue
        preferred_path_xy_m = None
        if route_mode == "free_endpoint_swept_contact":
            free_tangent = float((free_endpoint - centroid_xy) @ tangent_xy)
            anchor_tangent = float((anchor_endpoint - centroid_xy) @ tangent_xy)
            free_is_low = free_tangent < anchor_tangent
            anchor_metadata["free_is_low"] = bool(free_is_low)
            if "history_connectivity_crossing_xy_m" in anchor_metadata:
                connectivity_crossing = np.asarray(
                    anchor_metadata["history_connectivity_crossing_xy_m"],
                    dtype=np.float64,
                )
                connectivity_tangent = float(
                    (connectivity_crossing - centroid_xy) @ tangent_xy
                )
                anchor_metadata[
                    "history_connectivity_crossing_tangent_offset_m"
                ] = float(
                    float(tangent_low) - connectivity_tangent
                    if free_is_low
                    else connectivity_tangent - float(tangent_high)
                )
            preferred = route.get("preferred_path_xy_m") or [
                list(history["before_xy_m"]),
                list(history["projected_xy_m"]),
                list(history["after_xy_m"]),
            ]
            preferred = [list(_pair(point, "preferred leaf route")) for point in preferred]
            if (
                (np.asarray(preferred[-1]) - route_crossing_xy) @ normal_xy
                < (np.asarray(preferred[0]) - route_crossing_xy) @ normal_xy
            ):
                preferred.reverse()
            preferred_path_xy_m = preferred

        # The live layer remains authoritative except for the portion of the
        # certified movable leaf needed by the taught base sweep.  In free-end
        # mode, retaining the anchored remainder also forces the planner to go
        # around the observed tip instead of shortcutting through the panel.
        tangent_at_origin = float(
            (barrier_origin_xy - centroid_xy) @ tangent_xy
        )
        relative_tangent_low = float(tangent_low - tangent_at_origin)
        relative_tangent_high = float(tangent_high - tangent_at_origin)
        if route_mode == "free_endpoint_swept_contact":
            inward = max(
                THIN_BARRIER_MIN_CONTACT_SPAN_M,
                radius + THIN_BARRIER_FREE_END_OVERRIDE_CLEARANCE_PAD_M,
            )
            if free_is_low:
                override_low = (
                    relative_tangent_low - THIN_BARRIER_FREE_END_TAIL_PAD_M
                )
                override_high = min(
                    relative_tangent_high + THIN_BARRIER_OVERRIDE_TANGENT_PAD_M,
                    inward,
                )
            else:
                override_low = max(
                    relative_tangent_low - THIN_BARRIER_OVERRIDE_TANGENT_PAD_M,
                    -inward,
                )
                override_high = (
                    relative_tangent_high + THIN_BARRIER_FREE_END_TAIL_PAD_M
                )
        else:
            override_low = (
                relative_tangent_low - THIN_BARRIER_OVERRIDE_TANGENT_PAD_M
            )
            override_high = (
                relative_tangent_high + THIN_BARRIER_OVERRIDE_TANGENT_PAD_M
            )
        if override_high - override_low < THIN_BARRIER_MIN_CONTACT_SPAN_M:
            note(
                "component_rejected",
                component_label=int(label),
                reason="override_width",
                override_width_m=float(override_high - override_low),
                required_width_m=THIN_BARRIER_MIN_CONTACT_SPAN_M,
                route_mode=route_mode,
                tangent_span_m=tangent_span,
                normal_span_m=normal_span,
                centroid_xy_m=centroid_xy.astype(float).tolist(),
                crossing_xy_m=barrier_origin_xy.astype(float).tolist(),
                normal_xy=normal_xy.astype(float).tolist(),
                tangent_xy=tangent_xy.astype(float).tolist(),
                tangent_min_m=float(tangent_low),
                tangent_max_m=float(tangent_high),
                history=history,
            )
            continue
        certificate = {
            "schema": TRAVERSED_THIN_BARRIER_SCHEMA,
            "mode": route_mode,
            "crossing_xy_m": barrier_origin_xy.astype(float).tolist(),
            "centroid_xy_m": centroid_xy.astype(float).tolist(),
            "normal_xy": normal_xy.astype(float).tolist(),
            "tangent_xy": tangent_xy.astype(float).tolist(),
            "travel_normal_xy": normal_xy.astype(float).tolist(),
            "normal_span_m": normal_span,
            "tangent_span_m": tangent_span,
            "vertical_span_m": vertical_span,
            "z_low_m": float(z_low),
            "z_high_m": float(z_high),
            "plane_std_m": plane_std,
            "component_point_count": int(len(component)),
            "raw_component_point_count": int(len(raw_component)),
            "robust_inlier_ratio": robust_inlier_ratio,
            "raw_thin_slab_ratio": raw_thin_slab_ratio,
            "robust_fit_method": robust_fit_method,
            "low_point_count": low_point_count,
            "low_extent_evidence": (
                "observed_low_points"
                if observed_low_extent
                else (
                    "verified_stall_lower_image_boundary_censoring"
                    if censored_low_extent
                    else "verified_stall_history_mapped_free_censoring"
                )
            ),
            "lower_image_boundary_point_count": lower_boundary_point_count,
            "lower_image_boundary_min_z_m": (
                lower_boundary_min_z
                if math.isfinite(lower_boundary_min_z)
                else -1.0
            ),
            "verified_stall_xy_m": (
                stall_xy.astype(float).tolist()
                if (
                    censored_low_extent
                    or stall_history_censored_low_extent
                ) and stall_xy is not None
                else []
            ),
            "verified_stall_plane_distance_m": (
                stall_plane_distance
                if censored_low_extent or stall_history_censored_low_extent
                else -1.0
            ),
            "verified_stall_centroid_distance_m": (
                stall_centroid_distance
                if censored_low_extent or stall_history_censored_low_extent
                else -1.0
            ),
            "route_arc_m": route_arc_m,
            "route_normal_alignment": float(route["route_normal_alignment"]),
            "reference_segment_index": reference_segment_index,
            "history": history,
            "override_tangent_min_m": override_low,
            "override_tangent_max_m": override_high,
            "override_normal_half_width_m": float(
                min(.16, max(.12, .5 * normal_span + .06))
            ),
            "runtime_normal_half_width_m": .55,
            "runtime_approach_limit_m": .75,
            "runtime_exit_limit_m": .65,
            "max_speed_mps": THIN_BARRIER_CONTACT_SPEED_MPS,
            "segment_indices": [],
            **anchor_metadata,
        }
        score = (
            route_arc_m,
            plane_std,
            -float(route["route_normal_alignment"]),
        )
        candidate = {
            "certificate": certificate,
            "component_point_indices": original_indices[component_mask].copy(),
            "component_points_xy_m": component_map.copy(),
            # Detection intentionally has a longer range and uses the full
            # vertical return, while the ordinary chassis costmap keeps only
            # nearby low-layer points.  Carry the observed raw component so a
            # recovery plan cannot recognize a door and then silently omit its
            # anchored remainder from collision geometry.
            "raw_component_points_xy_m": points_map_xy[
                raw_component_mask
            ].copy(),
            "robot_radius_m": radius,
        }
        if preferred_path_xy_m is not None:
            candidate["preferred_path_xy_m"] = preferred_path_xy_m
        if best is None or score < best[0]:
            best = score, candidate
        note(
            "component_accepted",
            component_label=int(label),
            route_mode=route_mode,
            point_count=int(len(component)),
            tangent_span_m=tangent_span,
            normal_span_m=normal_span,
            vertical_span_m=vertical_span,
            route_arc_m=route_arc_m,
        )
    note("result", accepted=best is not None)
    return None if best is None else best[1]


def apply_traversed_thin_barrier_override(
    snapshot: Mapping[str, Any], detection: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a private planner snapshot with only the certified leaf removed."""

    certificate = detection.get("certificate")
    if (
        not isinstance(certificate, Mapping)
        or certificate.get("schema") != TRAVERSED_THIN_BARRIER_SCHEMA
    ):
        raise NavigationPlanError("traversed thin-barrier certificate is invalid")
    obstacle, free, resolution, origin = _snapshot_geometry(snapshot)
    crossing = np.asarray(certificate["crossing_xy_m"], dtype=np.float64)
    normal = np.asarray(certificate["normal_xy"], dtype=np.float64)
    tangent = np.asarray(certificate["tangent_xy"], dtype=np.float64)
    rows, columns = np.indices(obstacle.shape)
    centres = np.stack(
        (
            origin[0] + (columns + .5) * resolution,
            origin[1] + (rows + .5) * resolution,
        ),
        axis=-1,
    )
    relative = centres - crossing
    cell_pad = resolution / math.sqrt(2.0)
    corridor = (
        np.abs(relative @ normal)
        <= float(certificate["override_normal_half_width_m"]) + cell_pad
    )
    corridor &= (
        relative @ tangent
        >= float(certificate["override_tangent_min_m"]) - cell_pad
    ) & (
        relative @ tangent
        <= float(certificate["override_tangent_max_m"]) + cell_pad
    )
    result = dict(snapshot)
    result["obstacle_mask"] = np.asarray(obstacle & ~corridor, dtype=bool)
    result["free_mask"] = np.asarray(free | corridor, dtype=bool)

    raw = snapshot.get("navigation_depth_points_xy_m")
    retained = None
    removed_count = 0
    if raw is not None:
        raw_points = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        raw_relative = raw_points - crossing
        removable = (
            np.abs(raw_relative @ normal)
            <= float(certificate["override_normal_half_width_m"]) + resolution
        )
        removable &= (
            raw_relative @ tangent
            >= float(certificate["override_tangent_min_m"]) - resolution
        ) & (
            raw_relative @ tangent
            <= float(certificate["override_tangent_max_m"]) + resolution
        )
        retained = raw_points[~removable]
        removed_count = int(np.count_nonzero(removable))
    observed_raw = detection.get("raw_component_points_xy_m")
    observed_removed_count = 0
    injected_count = 0
    if observed_raw is not None:
        observed_points = np.asarray(observed_raw, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(observed_points)):
            raise NavigationPlanError(
                "traversed thin-barrier observed component is invalid"
            )
        observed_relative = observed_points - crossing
        observed_removable = (
            np.abs(observed_relative @ normal)
            <= float(certificate["override_normal_half_width_m"]) + resolution
        )
        observed_removable &= (
            observed_relative @ tangent
            >= float(certificate["override_tangent_min_m"]) - resolution
        ) & (
            observed_relative @ tangent
            <= float(certificate["override_tangent_max_m"]) + resolution
        )
        observed_removed_count = int(np.count_nonzero(observed_removable))
        observed_retained = observed_points[~observed_removable]
        if len(observed_retained):
            # One RGB-D surface contributes many heights at almost identical
            # XY.  Keep centimetre geometry, then add only portions absent from
            # the ordinary (shorter-horizon) planning layer.
            merge_bin_m = max(.005, min(.01, .25 * resolution))
            _keys, unique_indices = np.unique(
                np.floor(observed_retained / merge_bin_m).astype(np.int64),
                axis=0,
                return_index=True,
            )
            observed_retained = observed_retained[
                np.sort(unique_indices)
            ]
            if retained is not None and len(retained):
                absent = (
                    cKDTree(retained).query(observed_retained, k=1)[0]
                    > merge_bin_m
                )
                observed_retained = observed_retained[absent]
            injected_count = int(len(observed_retained))
            retained = (
                observed_retained
                if retained is None
                else np.concatenate((retained, observed_retained), axis=0)
            )
    if retained is not None:
        result["navigation_depth_points_xy_m"] = retained
    protected = snapshot.get("navigation_obstacle_mask")
    if protected is not None:
        protected = np.asarray(protected, dtype=bool).copy()
        protected[corridor] = False
        if retained is not None and len(retained):
            cells = np.floor((retained - origin) / resolution).astype(np.int64)
            valid = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < obstacle.shape[1])
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < obstacle.shape[0])
            )
            protected[cells[valid, 1], cells[valid, 0]] = True
        result["navigation_obstacle_mask"] = protected
    result["traversed_thin_barrier_override_cells"] = int(
        np.count_nonzero(corridor)
    )
    result["traversed_thin_barrier_removed_depth_points"] = removed_count
    result[
        "traversed_thin_barrier_observed_removed_depth_points"
    ] = observed_removed_count
    result[
        "traversed_thin_barrier_injected_depth_points"
    ] = injected_count
    preferred_path = detection.get("preferred_path_xy_m")
    if preferred_path is not None:
        result["navigation_preferred_path_xy_m"] = [
            list(_pair(point, "preferred traversed-barrier path point"))
            for point in preferred_path
        ]
        result["navigation_preferred_path_context"] = {
            "crossing_xy_m": list(certificate["crossing_xy_m"]),
            "normal_xy": list(certificate["normal_xy"]),
            "tangent_xy": list(certificate["tangent_xy"]),
            "override_tangent_min_m": float(
                certificate["override_tangent_min_m"]
            ),
            "override_tangent_max_m": float(
                certificate["override_tangent_max_m"]
            ),
            "free_is_low": bool(certificate["free_is_low"]),
            "robot_radius_m": float(
                detection.get("robot_radius_m", DEFAULT_ROBOT_RADIUS_M)
            ),
        }
    return result


def restore_traversed_thin_barrier_observation(
    snapshot: Mapping[str, Any], detection: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a preferred-route snapshot with the complete leaf restored.

    Recovery may find a route around the currently observed free endpoint
    whose base envelope never contacts the leaf. Such a route needs no runtime
    exception, but it must be proved against the recognition layer as well as
    the shorter-horizon chassis layer. Start with the same signed gate, then
    restore every original map cell and observed component point before the
    ordinary depth refiner runs again.
    """

    result = apply_traversed_thin_barrier_override(snapshot, detection)
    obstacle, free, resolution, origin = _snapshot_geometry(snapshot)
    result["obstacle_mask"] = np.asarray(obstacle, dtype=bool).copy()
    result["free_mask"] = np.asarray(free, dtype=bool).copy()

    point_sets: list[np.ndarray] = []
    for label, value in (
        (
            "navigation depth points",
            snapshot.get("navigation_depth_points_xy_m"),
        ),
        (
            "observed barrier points",
            detection.get("raw_component_points_xy_m"),
        ),
    ):
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            raise NavigationPlanError(f"{label} are invalid")
        if len(points):
            point_sets.append(points)
    if not point_sets:
        raise NavigationPlanError(
            "traversed thin-barrier observation is unavailable"
        )
    points = np.concatenate(point_sets, axis=0)
    result["navigation_depth_points_xy_m"] = points

    protected_value = snapshot.get("navigation_obstacle_mask")
    if protected_value is None:
        protected = np.zeros(obstacle.shape, dtype=bool)
    else:
        protected = np.asarray(protected_value, dtype=bool).copy()
        if protected.shape != obstacle.shape:
            raise NavigationPlanError(
                "navigation_obstacle_mask has a different shape"
            )
    cells = np.floor((points - origin) / resolution).astype(np.int64)
    valid = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < obstacle.shape[1])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < obstacle.shape[0])
    )
    protected[cells[valid, 1], cells[valid, 0]] = True
    result["navigation_obstacle_mask"] = protected
    result["traversed_thin_barrier_full_observation_point_count"] = int(
        len(points)
    )
    return result


def bind_traversed_thin_barrier_segments(
    certificate: Mapping[str, Any],
    path_xy_m: Sequence[Sequence[float]],
    *,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
) -> dict[str, Any] | None:
    """Bind a geometric barrier certificate to the executable path segments."""

    if certificate.get("schema") != TRAVERSED_THIN_BARRIER_SCHEMA:
        return None
    try:
        path = np.asarray(path_xy_m, dtype=np.float64)
        crossing = np.asarray(certificate["crossing_xy_m"], dtype=np.float64)
        normal = np.asarray(certificate["normal_xy"], dtype=np.float64)
        tangent = np.asarray(certificate["tangent_xy"], dtype=np.float64)
        radius = float(robot_radius_m)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or len(path) < 2
        or not np.all(np.isfinite(path))
        or not np.all(np.isfinite(crossing))
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(tangent))
        or radius <= 0.0
    ):
        return None
    if certificate.get("mode") == "free_endpoint_swept_contact":
        try:
            centroid = np.asarray(certificate["centroid_xy_m"], dtype=np.float64)
            free_endpoint = np.asarray(
                certificate["free_endpoint_xy_m"], dtype=np.float64
            )
            anchor_endpoint = np.asarray(
                certificate["anchor_endpoint_xy_m"], dtype=np.float64
            )
            free_is_low = bool(certificate["free_is_low"])
            endpoint_tangent = np.asarray(
                [
                    float((free_endpoint - centroid) @ tangent),
                    float((anchor_endpoint - centroid) @ tangent),
                ]
            )
        except (KeyError, TypeError, ValueError):
            return None
        free_end_route = _thin_barrier_free_end_route(
            path,
            centroid,
            normal,
            tangent,
            float(endpoint_tangent.min()),
            float(endpoint_tangent.max()),
            free_is_low=free_is_low,
            robot_radius_m=radius,
        )
        if free_end_route is None:
            return None
        if certificate.get("route_source") == (
            "traversed_history_changed_leaf_pose"
        ):
            panel_start = centroid + float(endpoint_tangent.min()) * tangent
            panel_end = centroid + float(endpoint_tangent.max()) * tangent
            minimum_panel_clearance = min(
                _closest_segment_points(
                    start, end, panel_start, panel_end
                )[0]
                for start, end in zip(path[:-1], path[1:])
                if float(np.linalg.norm(end - start)) > 1.0e-9
            )
            required_panel_clearance = (
                radius + DEPTH_POINT_MARGIN_M + DEPTH_TRACKING_RESERVE_M
            )
            if minimum_panel_clearance + 1.0e-9 < required_panel_clearance:
                # The historical sweep proves topology only.  A path that
                # shortcuts into a leaf at its new pose must fall through to
                # full-observation replanning; it cannot receive a runtime
                # obstacle-ignore certificate.
                return None
    indices: list[int] = []
    crossing_segments: list[int] = []
    barrier_start = crossing + float(
        certificate["override_tangent_min_m"]
    ) * tangent
    barrier_end = crossing + float(
        certificate["override_tangent_max_m"]
    ) * tangent
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        start_relative = start - crossing
        end_relative = end - crossing
        normal_range = sorted(
            [float(start_relative @ normal), float(end_relative @ normal)]
        )
        tangent_range = sorted(
            [float(start_relative @ tangent), float(end_relative @ tangent)]
        )
        normal_overlap = bool(
            normal_range[0] <= radius + .08
            and normal_range[1] >= -radius - .08
        )
        tangent_overlap = bool(
            tangent_range[0]
            <= float(certificate["override_tangent_max_m"]) + radius
            and tangent_range[1]
            >= float(certificate["override_tangent_min_m"]) - radius
        )
        contact_distance = _closest_segment_points(
            start, end, barrier_start, barrier_end
        )[0]
        direct_mode = certificate.get("mode") == "direct_plane_crossing"
        direction_valid = bool(
            not direct_mode
            or float(direction @ normal) >= THIN_BARRIER_MIN_ROUTE_NORMAL_DOT
        )
        if (
            direction_valid
            and normal_overlap
            and tangent_overlap
            and contact_distance <= radius + THIN_BARRIER_ROUTE_CONTACT_PAD_M
        ):
            indices.append(index)
        if normal_range[0] <= 0.0 <= normal_range[1] and tangent_overlap:
            crossing_segments.append(index)
    if not indices or (
        certificate.get("mode") == "direct_plane_crossing"
        and not crossing_segments
    ):
        return None
    result = dict(certificate)
    result["segment_indices"] = indices
    result["crossing_segment_indices"] = crossing_segments
    result["contact_segment_indices"] = list(indices)
    result["robot_radius_m"] = radius
    return result


def _coarsen(
    obstacle: np.ndarray,
    free: np.ndarray,
    resolution: float,
    target_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    factor = max(
        1, int(math.ceil(target_resolution_m / resolution - 1e-9))
    )
    if factor == 1:
        return obstacle.copy(), free.copy(), resolution, factor
    height, width = obstacle.shape
    out_h = int(math.ceil(height / factor))
    out_w = int(math.ceil(width / factor))
    padded_h, padded_w = out_h * factor, out_w * factor
    padded_obstacle = np.ones((padded_h, padded_w), dtype=bool)
    padded_free = np.zeros((padded_h, padded_w), dtype=bool)
    padded_valid = np.zeros((padded_h, padded_w), dtype=bool)
    padded_obstacle[:height, :width] = obstacle
    padded_free[:height, :width] = free
    padded_valid[:height, :width] = True
    obstacle_blocks = padded_obstacle.reshape(
        out_h, factor, out_w, factor
    ).any(axis=(1, 3))
    free_count = padded_free.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))
    valid_count = padded_valid.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))
    # Unknown space is blocked at every resolution. A coarse cell is free only
    # when every contributing source cell is explicitly observed free.
    free_blocks = (
        ~obstacle_blocks
        & (valid_count > 0)
        & (free_count == valid_count)
    )
    return obstacle_blocks, free_blocks, resolution * factor, factor


def _cell(
    xy: tuple[float, float],
    origin: tuple[float, float],
    resolution: float,
) -> tuple[int, int]:
    column = int(math.floor((xy[0] - origin[0]) / resolution))
    row = int(math.floor((xy[1] - origin[1]) / resolution))
    return row, column


def _xy(
    cell: tuple[int, int],
    origin: tuple[float, float],
    resolution: float,
) -> tuple[float, float]:
    row, column = cell
    return (
        origin[0] + (float(column) + 0.5) * resolution,
        origin[1] + (float(row) + 0.5) * resolution,
    )


def _inside(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _nearest_cell_xy(
    mask: np.ndarray,
    desired_xy: tuple[float, float],
    origin: tuple[float, float],
    resolution: float,
    max_distance_m: float,
) -> tuple[tuple[int, int] | None, float]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None, math.inf
    x_m = origin[0] + (columns.astype(np.float64) + 0.5) * resolution
    y_m = origin[1] + (rows.astype(np.float64) + 0.5) * resolution
    dx = x_m - float(desired_xy[0])
    dy = y_m - float(desired_xy[1])
    distances_sq = dx * dx + dy * dy
    index = int(np.argmin(distances_sq))
    distance = math.sqrt(float(distances_sq[index]))
    if distance > max_distance_m + 1e-9:
        return None, distance
    return (int(rows[index]), int(columns[index])), distance


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _step_allowed(
    traversable: np.ndarray,
    row: int,
    column: int,
    drow: int,
    dcolumn: int,
) -> bool:
    nr, nc = row + drow, column + dcolumn
    if not _inside(traversable.shape, (nr, nc)) or not traversable[nr, nc]:
        return False
    if drow and dcolumn:
        # A diagonal cannot squeeze through two cells touching at a corner.
        return bool(
            traversable[row + drow, column]
            and traversable[row, column + dcolumn]
        )
    return True


def _reachable(traversable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    if not _inside(traversable.shape, start) or not traversable[start]:
        return np.zeros(traversable.shape, dtype=bool)
    # A legal diagonal step requires both adjacent cardinal cells, so the
    # no-corner-cutting graph has the same connected components as a 4-neighbor
    # grid. SciPy labels that graph in native code and avoids a full Python
    # traversal of large maps.
    labels, _ = ndimage.label(
        traversable,
        structure=np.asarray(
            ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
        ),
    )
    start_label = int(labels[start])
    return labels == start_label


def _astar(
    traversable: np.ndarray,
    clearance_m: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution: float,
    minimum_clearance_m: float,
    clearance_weight: float,
    forbidden_edges: set[tuple[int, int]] | None = None,
    progress_check: Callable[[], None] | None = None,
    overlap_cost: np.ndarray | None = None,
    edge_check: Callable[[tuple[int, int], tuple[int, int]], bool] | None = None,
) -> tuple[list[tuple[int, int]], int]:
    if progress_check is not None:
        progress_check()
    if start == goal:
        return [start], 0
    height, width = traversable.shape
    total = height * width
    start_index = start[0] * width + start[1]
    goal_index = goal[0] * width + goal[1]
    costs = np.full(total, np.inf, dtype=np.float64)
    parents = np.full(total, -1, dtype=np.int64)
    closed = np.zeros(total, dtype=bool)
    costs[start_index] = 0.0

    def heuristic(row: int, column: int) -> float:
        return math.hypot(goal[0] - row, goal[1] - column) * resolution

    queue: list[tuple[float, float, int, int]] = [
        (heuristic(*start), 0.0, start[0], start[1])
    ]
    expanded = 0
    decay_m = max(0.20, minimum_clearance_m * 0.75)
    while queue:
        _, current_cost, row, column = heapq.heappop(queue)
        index = row * width + column
        if closed[index] or current_cost > costs[index] + 1e-12:
            continue
        closed[index] = True
        expanded += 1
        if progress_check is not None and expanded % 256 == 0:
            progress_check()
        if index == goal_index:
            break
        for drow, dcolumn, step_scale in _NEIGHBORS:
            nr, nc = row + drow, column + dcolumn
            if edge_check is None:
                if not _step_allowed(traversable, row, column, drow, dcolumn):
                    continue
            elif not (0 <= nr < height and 0 <= nc < width and traversable[nr, nc]):
                continue
            neighbor_index = nr * width + nc
            edge_key = (
                min(index, neighbor_index),
                max(index, neighbor_index),
            )
            if forbidden_edges is not None and edge_key in forbidden_edges:
                continue
            if closed[neighbor_index]:
                continue
            if edge_check is not None and not edge_check((row, column), (nr, nc)):
                continue
            clearance = float(clearance_m[nr, nc])
            wall_penalty = clearance_weight * math.exp(
                -max(0.0, clearance - minimum_clearance_m) / decay_m
            )
            sweep_penalty = (
                0.0 if overlap_cost is None else float(overlap_cost[nr, nc])
            )
            step_cost = resolution * step_scale * (
                1.0 + wall_penalty + sweep_penalty
            )
            candidate = current_cost + step_cost
            if candidate + 1e-12 >= costs[neighbor_index]:
                continue
            costs[neighbor_index] = candidate
            parents[neighbor_index] = index
            heapq.heappush(
                queue,
                (candidate + heuristic(nr, nc), candidate, nr, nc),
            )
    if not closed[goal_index]:
        raise NavigationPlanError(
            "marked place is not connected by observed free space"
        )
    path: list[tuple[int, int]] = []
    current = goal_index
    while current >= 0:
        path.append((current // width, current % width))
        if current == start_index:
            break
        current = int(parents[current])
    if not path or path[-1] != start:
        raise NavigationPlanError("planner parent chain is incomplete")
    path.reverse()
    return path, expanded


def _astar_oriented_layers(
    layers: Sequence[Mapping[str, Any]],
    start_states: Sequence[tuple[int, tuple[int, int], float]],
    goal_layers: set[int],
    goal: tuple[int, int],
    turnable: np.ndarray,
    resolution: float,
    clearance_weight: float,
    *,
    forbidden_translation_edges: set[tuple[int, int, int]] | None = None,
    forbidden_rotation_edges: set[tuple[int, int, int, int]] | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[list[tuple[int, int, int]], int]:
    """Search an SE(2) lattice whose yaw changes only in full-turn pockets.

    Each layer is an exact fixed-yaw configuration-space mask.  Translation
    stays in one layer; a yaw transition is possible only where the complete
    base circumcircle is clear in both the map and live depth.  The returned
    discrete edges still need continuous polygon-sweep certification.
    """

    if not layers or not start_states or not goal_layers:
        raise NavigationPlanError("oriented route has no valid endpoint state")
    height, width = np.asarray(layers[0]["allowed"]).shape
    layer_count = len(layers)
    plane_size = height * width
    total = layer_count * plane_size
    costs = np.full(total, np.inf, dtype=np.float64)
    parents = np.full(total, -1, dtype=np.int64)
    closed = np.zeros(total, dtype=bool)
    queue: list[tuple[float, float, int]] = []

    def state_index(layer_index: int, row: int, column: int) -> int:
        return layer_index * plane_size + row * width + column

    def heuristic(row: int, column: int) -> float:
        return math.hypot(goal[0] - row, goal[1] - column) * resolution

    for layer_index, cell, initial_cost in start_states:
        row, column = cell
        if not np.asarray(layers[layer_index]["allowed"])[row, column]:
            continue
        index = state_index(layer_index, row, column)
        if initial_cost + 1.0e-12 >= costs[index]:
            continue
        costs[index] = float(initial_cost)
        heapq.heappush(
            queue,
            (float(initial_cost) + heuristic(row, column), float(initial_cost), index),
        )

    expanded = 0
    goal_index: int | None = None
    translation_forbidden = forbidden_translation_edges or set()
    rotation_forbidden = forbidden_rotation_edges or set()
    while queue:
        _, current_cost, index = heapq.heappop(queue)
        if closed[index] or current_cost > costs[index] + 1.0e-12:
            continue
        closed[index] = True
        expanded += 1
        if progress_check is not None and expanded % 256 == 0:
            progress_check()
        layer_index, spatial_index = divmod(index, plane_size)
        row, column = divmod(spatial_index, width)
        if (row, column) == goal and layer_index in goal_layers:
            goal_index = index
            break

        layer = layers[layer_index]
        allowed = np.asarray(layer["allowed"])
        local_clearance = np.asarray(layer["configuration_clearance_m"])
        decay_m = 0.15
        for drow, dcolumn, step_scale in _NEIGHBORS:
            if not _step_allowed(allowed, row, column, drow, dcolumn):
                continue
            nr, nc = row + drow, column + dcolumn
            next_spatial = nr * width + nc
            edge = (
                layer_index,
                min(spatial_index, next_spatial),
                max(spatial_index, next_spatial),
            )
            if edge in translation_forbidden:
                continue
            neighbor = state_index(layer_index, nr, nc)
            if closed[neighbor]:
                continue
            spare = max(0.0, float(local_clearance[nr, nc]) - resolution)
            wall_penalty = clearance_weight * math.exp(-spare / decay_m)
            candidate = current_cost + resolution * step_scale * (1.0 + wall_penalty)
            if candidate + 1.0e-12 >= costs[neighbor]:
                continue
            costs[neighbor] = candidate
            parents[neighbor] = index
            heapq.heappush(
                queue,
                (candidate + heuristic(nr, nc), candidate, neighbor),
            )

        if not turnable[row, column]:
            continue
        current_yaw = float(layer["yaw"])
        for next_layer, candidate_layer in enumerate(layers):
            if next_layer == layer_index:
                continue
            rotation_edge = (
                spatial_index,
                min(layer_index, next_layer),
                max(layer_index, next_layer),
            )
            if rotation_edge in rotation_forbidden:
                continue
            if not np.asarray(candidate_layer["allowed"])[row, column]:
                continue
            neighbor = state_index(next_layer, row, column)
            if closed[neighbor]:
                continue
            delta = abs(
                (float(candidate_layer["yaw"]) - current_yaw + math.pi)
                % (2.0 * math.pi)
                - math.pi
            )
            # A positive turn cost avoids gratuitous yaw changes while keeping
            # distance, not the arbitrary layer index, as the primary metric.
            candidate = current_cost + 0.10 + 0.05 * delta
            if candidate + 1.0e-12 >= costs[neighbor]:
                continue
            costs[neighbor] = candidate
            parents[neighbor] = index
            heapq.heappush(
                queue,
                (candidate + heuristic(row, column), candidate, neighbor),
            )

    if goal_index is None:
        raise NavigationPlanError(
            "marked place is not connected in the oriented configuration space"
        )
    path: list[tuple[int, int, int]] = []
    current = goal_index
    while current >= 0:
        layer_index, spatial_index = divmod(current, plane_size)
        row, column = divmod(spatial_index, width)
        path.append((layer_index, row, column))
        parent = int(parents[current])
        if parent < 0:
            break
        current = parent
    path.reverse()
    return path, expanded


def _component_oriented_path(
    layers: Sequence[Mapping[str, Any]],
    start_states: Sequence[tuple[int, tuple[int, int], float]],
    goal_layers: set[int],
    goal: tuple[int, int],
    rotation_links: Sequence[
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ]
    ],
    resolution: float,
    clearance_weight: float,
    *,
    forbidden_translation_edges: set[tuple[int, int, int]] | None = None,
    forbidden_rotation_edges: set[tuple[int, int, int]] | None = None,
    progress_check: Callable[[], None] | None = None,
    overlap_cost: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int]], int]:
    """Plan through fixed-yaw components joined by certified turn pockets."""

    structure = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
    )
    height, width = np.asarray(layers[0]["allowed"]).shape
    plane_size = height * width
    component_labels: list[np.ndarray] = []
    for layer in layers:
        labels, _ = ndimage.label(np.asarray(layer["allowed"]), structure=structure)
        component_labels.append(labels)
    start_by_node: dict[tuple[int, int], list[tuple[float, tuple[int, int]]]] = {}
    for layer_index, cell, cost in start_states:
        component = int(component_labels[layer_index][cell])
        if component:
            start_by_node.setdefault((layer_index, component), []).append(
                (float(cost), cell)
            )
    goal_nodes = {
        (layer_index, int(component_labels[layer_index][goal]))
        for layer_index in goal_layers
        if int(component_labels[layer_index][goal])
    }
    if not start_by_node or not goal_nodes:
        raise NavigationPlanError("oriented route has no endpoint component")

    adjacency: dict[
        tuple[int, int],
        list[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ],
    ] = {}
    for first, second, start_cell, end_cell in rotation_links:
        adjacency.setdefault(first, []).append((second, start_cell, end_cell))
        adjacency.setdefault(second, []).append((first, end_cell, start_cell))

    # Breadth-first search minimizes the number of in-place orientation
    # changes. Translation length is optimized by the subsequent 2-D A* legs.
    queue = list(start_by_node)
    parent: dict[
        tuple[int, int],
        tuple[
            tuple[int, int], tuple[int, int], tuple[int, int]
        ] | None,
    ] = {
        node: None for node in queue
    }
    selected_goal: tuple[int, int] | None = None
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if node in goal_nodes:
            selected_goal = node
            break
        for next_node, start_cell, end_cell in adjacency.get(node, ()):
            if next_node in parent:
                continue
            parent[next_node] = (node, start_cell, end_cell)
            queue.append(next_node)
    if selected_goal is None:
        raise NavigationPlanError(
            "oriented components are not joined by a certified partial turn"
        )

    nodes = [selected_goal]
    transitions: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while parent[nodes[-1]] is not None:
        previous, start_cell, end_cell = parent[nodes[-1]]
        transitions.append((start_cell, end_cell))
        nodes.append(previous)
    nodes.reverse()
    transitions.reverse()

    initial_node = nodes[0]
    first_target = transitions[0][0] if transitions else goal
    initial_options = start_by_node[initial_node]
    _, current_cell = min(
        initial_options,
        key=lambda item: (
            item[0]
            + resolution * math.dist(item[1], first_target),
            item[1],
        ),
    )
    state_path: list[tuple[int, int, int]] = [
        (initial_node[0], current_cell[0], current_cell[1])
    ]
    expanded_total = 0
    translation_forbidden = forbidden_translation_edges or set()
    rotation_forbidden = forbidden_rotation_edges or set()
    for leg_index, node in enumerate(nodes):
        layer_index = node[0]
        target_cell = (
            transitions[leg_index][0]
            if leg_index < len(transitions)
            else goal
        )
        layer_forbidden = {
            (first, second)
            for edge_layer, first, second in translation_forbidden
            if edge_layer == layer_index
        }
        grid_path, expanded = _astar(
            np.asarray(layers[layer_index]["allowed"]),
            np.asarray(layers[layer_index]["configuration_clearance_m"]),
            current_cell,
            target_cell,
            resolution,
            resolution,
            clearance_weight,
            forbidden_edges=layer_forbidden,
            progress_check=progress_check,
            overlap_cost=overlap_cost,
        )
        expanded_total += expanded
        state_path.extend(
            (layer_index, row, column) for row, column in grid_path[1:]
        )
        current_cell = target_cell
        if leg_index >= len(transitions):
            continue
        next_layer = nodes[leg_index + 1][0]
        next_cell = transitions[leg_index][1]
        spatial = current_cell[0] * width + current_cell[1]
        next_spatial = next_cell[0] * width + next_cell[1]
        rotation_edge = (
            min(spatial, next_spatial),
            max(spatial, next_spatial),
            min(layer_index, next_layer),
            max(layer_index, next_layer),
        )
        if rotation_edge in rotation_forbidden:
            raise NavigationPlanError("selected turn-pocket cell was rejected")
        state_path.append((next_layer, next_cell[0], next_cell[1]))
        current_cell = next_cell
    return state_path, expanded_total


def _line_cells(
    traversable: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    samples = max(abs(dr), abs(dc)) * 2 + 1
    previous = start
    cells = [start]
    for index in range(samples + 1):
        alpha = float(index) / float(max(1, samples))
        cell = (
            int(round(start[0] + alpha * dr)),
            int(round(start[1] + alpha * dc)),
        )
        if not _inside(traversable.shape, cell) or not traversable[cell]:
            return None
        step = (cell[0] - previous[0], cell[1] - previous[1])
        if step != (0, 0):
            # Dense sampling limits each transition to one cell.
            if abs(step[0]) > 1 or abs(step[1]) > 1:
                return None
            if not _step_allowed(
                traversable, previous[0], previous[1], step[0], step[1]
            ):
                return None
            cells.append(cell)
        previous = cell
    return cells if previous == end else None


def _line_is_clear(
    traversable: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    return _line_cells(traversable, start, end) is not None


def _path_cost(
    path: Sequence[tuple[int, int]],
    clearance_m: np.ndarray,
    resolution: float,
    minimum_clearance_m: float,
    clearance_weight: float,
    overlap_cost: np.ndarray | None = None,
) -> float:
    decay_m = max(0.20, minimum_clearance_m * 0.75)
    result = 0.0
    for start, end in zip(path[:-1], path[1:]):
        step_scale = math.hypot(end[0] - start[0], end[1] - start[1])
        clearance = float(clearance_m[end])
        wall_penalty = clearance_weight * math.exp(
            -max(0.0, clearance - minimum_clearance_m) / decay_m
        )
        sweep_penalty = 0.0 if overlap_cost is None else float(overlap_cost[end])
        result += resolution * step_scale * (1.0 + wall_penalty + sweep_penalty)
    return result


def _path_overlap_cost(path, overlap_cost, resolution: float) -> float:
    if overlap_cost is None:
        return 0.0
    return sum(
        math.dist(start, end) * resolution * float(overlap_cost[end])
        for start, end in zip(path[:-1], path[1:])
    )


def _metric_overlap_cost(start, end, overlap_cost, origin, resolution) -> float:
    if overlap_cost is None:
        return 0.0
    length = math.dist(start, end)
    samples = max(1, int(math.ceil(length / (0.5 * resolution))))
    alpha = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    columns = np.floor(
        (start[0] + alpha * (end[0] - start[0]) - origin[0]) / resolution
    ).astype(np.int64)
    rows = np.floor(
        (start[1] + alpha * (end[1] - start[1]) - origin[1]) / resolution
    ).astype(np.int64)
    if (np.any(rows < 0) or np.any(columns < 0)
            or np.any(rows >= overlap_cost.shape[0])
            or np.any(columns >= overlap_cost.shape[1])):
        return math.inf
    return length * float(np.mean(overlap_cost[rows, columns]))


def _preferred_path_cost(
    snapshot: Mapping[str, Any],
    shape: tuple[int, int],
    origin: tuple[float, float],
    resolution: float,
    *,
    robot_radius_m: float,
) -> np.ndarray | None:
    """Rasterize a local cost that preserves a certified taught door sweep.

    The cost exists only around the one detected movable leaf.  It does not
    turn arbitrary historic travel into free space and it cannot authorize a
    route by itself; collision checks and the signed leaf-end binding remain
    mandatory after planning.
    """

    raw_path = snapshot.get("navigation_preferred_path_xy_m")
    raw_context = snapshot.get("navigation_preferred_path_context")
    if raw_path is None or not isinstance(raw_context, Mapping):
        return None
    try:
        path = np.asarray(raw_path, dtype=np.float64).reshape(-1, 2)
        crossing = np.asarray(
            raw_context["crossing_xy_m"], dtype=np.float64
        ).reshape(2)
        normal = np.asarray(
            raw_context["normal_xy"], dtype=np.float64
        ).reshape(2)
        tangent = np.asarray(
            raw_context["tangent_xy"], dtype=np.float64
        ).reshape(2)
        tangent_low = float(raw_context["override_tangent_min_m"])
        tangent_high = float(raw_context["override_tangent_max_m"])
        free_is_low = raw_context["free_is_low"]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(path) < 2
        or not np.all(np.isfinite(path))
        or not np.all(np.isfinite(crossing))
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(tangent))
        or not math.isfinite(tangent_low)
        or not math.isfinite(tangent_high)
        or tangent_low >= tangent_high
        or not isinstance(free_is_low, (bool, np.bool_))
    ):
        return None
    normal_norm = float(np.linalg.norm(normal))
    tangent_norm = float(np.linalg.norm(tangent))
    if normal_norm <= 1.0e-9 or tangent_norm <= 1.0e-9:
        return None
    normal /= normal_norm
    tangent /= tangent_norm

    dense = np.asarray(
        _subdivide(path.tolist(), max(.02, .5 * resolution)),
        dtype=np.float64,
    )
    rows, columns = np.indices(shape)
    centres = np.stack(
        (
            origin[0] + (columns + .5) * resolution,
            origin[1] + (rows + .5) * resolution,
        ),
        axis=-1,
    )
    relative = centres - crossing
    normal_coordinate = relative @ normal
    tangent_coordinate = relative @ tangent
    preferred_tangent = (path - crossing) @ tangent
    tangent_pad = robot_radius_m + .15
    local = (
        np.abs(normal_coordinate)
        <= THIN_BARRIER_HISTORY_SIDE_M + robot_radius_m + .10
    )
    local &= tangent_coordinate >= min(
        tangent_low, float(preferred_tangent.min())
    ) - tangent_pad
    local &= tangent_coordinate <= max(
        tangent_high, float(preferred_tangent.max())
    ) + tangent_pad
    if not np.any(local):
        return None

    distances = np.full(shape, np.inf, dtype=np.float64)
    distances[local] = cKDTree(dense).query(centres[local], k=1)[0]
    scaled = np.clip(
        (distances - THIN_BARRIER_PREFERRED_PATH_TUBE_M)
        / THIN_BARRIER_PREFERRED_PATH_RAMP_M,
        0.0,
        1.0,
    )
    cost = np.zeros(shape, dtype=np.float64)
    cost[local] = THIN_BARRIER_PREFERRED_PATH_WEIGHT * scaled[local]
    # The generated preferred polyline crosses the current leaf plane at its
    # free-end detour.  Make that crossing topologically mandatory inside this
    # private recovery plan.  An infinite plane strip spans the bounded search
    # raster, with only a centre-path opening around the taught/current gate;
    # start and goal were already proven to lie on opposite sides.  Static-map
    # and live-depth configuration masks remain authoritative inside the gate.
    preferred_normal = (path - crossing) @ normal
    gate_index = int(np.argmin(np.abs(preferred_normal)))
    gate_tangent = float(preferred_tangent[gate_index])
    gate_half_width = min(
        THIN_BARRIER_PREFERRED_GATE_MAX_HALF_WIDTH_M,
        max(
            THIN_BARRIER_PREFERRED_GATE_MIN_HALF_WIDTH_M,
            .5 * robot_radius_m + .06,
        ),
    )
    gate_fence = np.abs(normal_coordinate) <= max(
        THIN_BARRIER_PREFERRED_GATE_FENCE_HALF_THICKNESS_M,
        2.0 * resolution,
    )
    gate_opening = (
        np.abs(tangent_coordinate - gate_tangent)
        <= gate_half_width + .5 * resolution
    )
    cost[gate_fence & ~gate_opening] = np.inf
    # Removing the movable leaf from collision geometry lets the base envelope
    # overlap it while pushing around the taught free end.  It must not let the
    # base centre teleport through the leaf itself.  An infinite centreline
    # cost also participates in component construction below, so orientation
    # transitions cannot silently pin the route to the middle of the panel.
    if free_is_low:
        panel_tangent = (tangent_coordinate >= 0.0) & (
            tangent_coordinate <= max(0.0, tangent_high)
        )
    else:
        panel_tangent = (tangent_coordinate <= 0.0) & (
            tangent_coordinate >= min(0.0, tangent_low)
        )
    panel_centreline = panel_tangent & (
        np.abs(normal_coordinate) <= max(.04, 1.5 * resolution)
    )
    cost[panel_centreline] = np.inf
    return cost


def _shortcut_preserves_preference(
    start: Sequence[float],
    end: Sequence[float],
    original_segments: Sequence[
        tuple[Sequence[float], Sequence[float]]
    ],
    overlap_cost: np.ndarray | None,
    origin: tuple[float, float],
    resolution: float,
) -> bool:
    """Reject a simplification that cuts away from a taught local sweep."""

    if overlap_cost is None:
        return True
    direct = _metric_overlap_cost(
        start, end, overlap_cost, origin, resolution
    )
    original = sum(
        _metric_overlap_cost(
            segment_start,
            segment_end,
            overlap_cost,
            origin,
            resolution,
        )
        for segment_start, segment_end in original_segments
    )
    return bool(
        math.isfinite(direct)
        and math.isfinite(original)
        and direct
        <= original + THIN_BARRIER_PREFERRED_PATH_SIMPLIFY_SLACK_M
    )


def _segment_to_rectangles_distance(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    x_min: np.ndarray,
    y_min: np.ndarray,
    x_max: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Return exact distances from one segment to axis-aligned rectangles."""

    ax, ay = start_xy
    bx, by = end_xy
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy

    def point_to_rect(px: float, py: float) -> np.ndarray:
        dx = np.maximum(np.maximum(x_min - px, 0.0), px - x_max)
        dy = np.maximum(np.maximum(y_min - py, 0.0), py - y_max)
        return np.hypot(dx, dy)

    distances = np.minimum(point_to_rect(ax, ay), point_to_rect(bx, by))
    if length_sq <= 1e-18:
        return distances

    for corner_x, corner_y in (
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ):
        projection = np.clip(
            ((corner_x - ax) * vx + (corner_y - ay) * vy) / length_sq,
            0.0,
            1.0,
        )
        projected_x = ax + projection * vx
        projected_y = ay + projection * vy
        distances = np.minimum(
            distances,
            np.hypot(corner_x - projected_x, corner_y - projected_y),
        )

    if abs(vx) <= 1e-15:
        x_valid = (ax >= x_min) & (ax <= x_max)
        x_enter = np.full(x_min.shape, -np.inf, dtype=np.float64)
        x_exit = np.full(x_min.shape, np.inf, dtype=np.float64)
    else:
        tx0 = (x_min - ax) / vx
        tx1 = (x_max - ax) / vx
        x_enter = np.minimum(tx0, tx1)
        x_exit = np.maximum(tx0, tx1)
        x_valid = np.ones(x_min.shape, dtype=bool)
    if abs(vy) <= 1e-15:
        y_valid = (ay >= y_min) & (ay <= y_max)
        y_enter = np.full(y_min.shape, -np.inf, dtype=np.float64)
        y_exit = np.full(y_min.shape, np.inf, dtype=np.float64)
    else:
        ty0 = (y_min - ay) / vy
        ty1 = (y_max - ay) / vy
        y_enter = np.minimum(ty0, ty1)
        y_exit = np.maximum(ty0, ty1)
        y_valid = np.ones(y_min.shape, dtype=bool)
    intersects = (
        x_valid
        & y_valid
        & (np.maximum.reduce((x_enter, y_enter, np.zeros(x_min.shape)))
           <= np.minimum.reduce((x_exit, y_exit, np.ones(x_min.shape))))
    )
    distances[intersects] = 0.0
    return distances


def _swept_footprint_mask(
    shape: tuple[int, int],
    paths: Sequence[Sequence[tuple[float, float]]],
    origin: tuple[float, float],
    resolution: float,
    radius_m: float,
    *,
    max_link_m: float = math.inf,
) -> tuple[np.ndarray, int]:
    """Rasterize the union of segment capsules, with round joins and caps.

    Each closed grid-cell square is tested against the continuous segment,
    not just its center. Unioning the capsules counts shared corners once.
    """

    height, width = shape
    map_x_min, map_y_min = origin
    evidence = np.zeros(shape, dtype=np.bool_)
    segment_count = 0
    for path in paths:
        if not path:
            continue
        segments = list(zip(path[:-1], path[1:]))
        if not segments:
            segments = [(path[0], path[0])]
        for start_xy, end_xy in segments:
            if math.dist(start_xy, end_xy) > max_link_m + 1.0e-9:
                continue
            x0 = min(start_xy[0], end_xy[0]) - radius_m
            x1 = max(start_xy[0], end_xy[0]) + radius_m
            y0 = min(start_xy[1], end_xy[1]) - radius_m
            y1 = max(start_xy[1], end_xy[1]) + radius_m
            column0 = max(
                0, int(math.floor((x0 - map_x_min) / resolution)) - 1
            )
            column1 = min(
                width - 1, int(math.floor((x1 - map_x_min) / resolution)) + 1
            )
            row0 = max(
                0, int(math.floor((y0 - map_y_min) / resolution)) - 1
            )
            row1 = min(
                height - 1, int(math.floor((y1 - map_y_min) / resolution)) + 1
            )
            if row0 > row1 or column0 > column1:
                continue
            rows, columns = np.mgrid[
                row0 : row1 + 1,
                column0 : column1 + 1,
            ]
            rect_x_min = map_x_min + columns.astype(np.float64) * resolution
            rect_y_min = map_y_min + rows.astype(np.float64) * resolution
            distances = _segment_to_rectangles_distance(
                start_xy,
                end_xy,
                rect_x_min,
                rect_y_min,
                rect_x_min + resolution,
                rect_y_min + resolution,
            )
            evidence[row0 : row1 + 1, column0 : column1 + 1] |= (
                distances <= radius_m + 1.0e-9
            )
            segment_count += 1
    return evidence, segment_count


def polyline_swept_mask(
    path_xy_m: Sequence[Sequence[float]],
    shape: tuple[int, int],
    origin_xy_m: tuple[float, float],
    resolution_m: float,
    radius_m: float = DEFAULT_ROUTE_SWEEP_RADIUS_M,
) -> np.ndarray:
    """Return cells touched by a metric, round-buffered planned polyline."""

    radius = _finite(radius_m, "sweep radius")
    resolution = _finite(resolution_m, "sweep resolution")
    if radius < 0.0 or radius > 1.5 or resolution <= 0.0:
        raise NavigationPlanError("invalid sweep radius or resolution")
    if len(shape) != 2 or min(shape) <= 0 or math.prod(shape) > MAX_GRID_CELLS:
        raise NavigationPlanError("invalid sweep grid shape")
    points = [_pair(point, "sweep point") for point in path_xy_m]
    origin = _pair(origin_xy_m, "sweep origin")
    mask, _ = _swept_footprint_mask(shape, [points], origin, resolution, radius)
    return mask


def _sweep_overlap_cost(blocked: np.ndarray, resolution: float) -> np.ndarray:
    """Penalize the fraction of the 0.4m disk touching the original map.

    Only evidence-override searches need this soft cost. Their hard mask still
    retains all fresh depth obstacles; old grey/black cells are not free of cost
    merely because a recorded traversal made them eligible for reconsideration.
    """

    radius = DEFAULT_ROUTE_SWEEP_RADIUS_M
    extent = int(math.ceil(radius / resolution + 0.5))
    offsets = np.arange(-extent, extent + 1, dtype=np.float64)
    edge_distances = np.maximum(np.abs(offsets) - 0.5, 0.0) * resolution
    # Row prefix sums evaluate the disk kernel in O(radius/cell_size) work per
    # cell, without a quadratic convolution kernel or a BLAS thread pool.
    height, width = blocked.shape
    padded = np.pad(blocked, extent, constant_values=True)
    prefix = np.pad(
        np.cumsum(padded, axis=1, dtype=np.int32), ((0, 0), (1, 0))
    )
    counts = np.zeros(blocked.shape, dtype=np.float32)
    kernel_cells = 0
    for row_offset, dy in enumerate(edge_distances):
        columns = np.flatnonzero(np.hypot(dy, edge_distances) <= radius + 1.0e-9)
        if not columns.size:
            continue
        left, right = int(columns[0]), int(columns[-1]) + 1
        counts += (
            prefix[row_offset:row_offset + height, right:right + width]
            - prefix[row_offset:row_offset + height, left:left + width]
        )
        kernel_cells += right - left
    return counts * (SWEEP_OVERLAP_WEIGHT / kernel_cells)


def route_sweep_audit(
    snapshot: Mapping[str, Any], path_xy_m: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Audit original map cells, without erasing conflicts via walked history."""

    obstacle, free, resolution, origin = _snapshot_geometry(snapshot)
    swept = polyline_swept_mask(path_xy_m, obstacle.shape, origin, resolution)
    unknown = ~free & ~obstacle
    protected = _optional_navigation_obstacle_mask(
        snapshot, "navigation_obstacle_mask", obstacle.shape
    )
    execution_stalls = _optional_navigation_obstacle_mask(
        snapshot, "navigation_stall_obstacle_mask", obstacle.shape
    )
    cells = int(np.count_nonzero(swept))
    occupied_cells = int(np.count_nonzero(swept & obstacle))
    unknown_cells = int(np.count_nonzero(swept & unknown))
    depth_cells = int(np.count_nonzero(swept & protected))
    stall_cells = int(np.count_nonzero(swept & execution_stalls))
    return {
        "radius_m": DEFAULT_ROUTE_SWEEP_RADIUS_M,
        "join_style": "round",
        "cap_style": "round",
        "swept_cells": cells,
        "occupied_overlap_cells": occupied_cells,
        "unknown_overlap_cells": unknown_cells,
        "live_depth_overlap_cells": depth_cells,
        "execution_stall_overlap_cells": stall_cells,
        "map_overlap_fraction": (occupied_cells + unknown_cells) / max(1, cells),
        "map_overlap_area_upper_bound_m2": (
            (occupied_cells + unknown_cells) * resolution * resolution
        ),
        "area_metric": "union_of_intersected_closed_grid_cells",
        "resolution_m": resolution,
    }


def _promote_traversed_footprint_free(
    obstacle: np.ndarray,
    free: np.ndarray,
    paths: Sequence[Sequence[tuple[float, float]]],
    origin: tuple[float, float],
    resolution: float,
    radius_m: float,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Promote cells intersected by a continuous, previously swept footprint."""

    evidence, segment_count = _swept_footprint_mask(
        obstacle.shape, paths, origin, resolution, radius_m,
        max_link_m=TRAVERSED_PATH_MAX_LINK_M,
    )
    promoted = evidence & ~obstacle & ~free
    return (
        free | (evidence & ~obstacle),
        int(np.count_nonzero(promoted)),
        segment_count,
        evidence,
    )


def _audit_traversed_segment_clearance(
    paths: Sequence[Sequence[tuple[float, float]]],
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
) -> tuple[int, int, float | None]:
    """Audit recorded centerline segments against the current planning grid."""

    safe_count = 0
    unsafe_count = 0
    minimum = math.inf
    for path in paths:
        for start_xy, end_xy in zip(path[:-1], path[1:]):
            if math.dist(start_xy, end_xy) > TRAVERSED_PATH_MAX_LINK_M + 1.0e-9:
                continue
            safe, certificate = _segment_clearance_certificate(
                start_xy,
                end_xy,
                blocked,
                clearance,
                origin,
                resolution,
                required_clearance_m,
            )
            minimum = min(minimum, certificate)
            if safe:
                safe_count += 1
            else:
                unsafe_count += 1
    return (
        safe_count,
        unsafe_count,
        None if not math.isfinite(minimum) else float(minimum),
    )


def _traversed_graph_route_between(
    paths: Sequence[Sequence[tuple[float, float]]],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_endpoint_distance_m: float,
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
    progress_check: Callable[[], None] | None,
    overlap_cost: np.ndarray | None = None,
    max_goal_endpoint_distance_m: float | None = None,
    centerline_support: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], float, float] | None:
    """Find a shortest certified route through the prior-traversal graph."""

    goal_limit = (
        max_endpoint_distance_m
        if max_goal_endpoint_distance_m is None
        else max_goal_endpoint_distance_m
    )
    if centerline_support is not None and centerline_support.shape != blocked.shape:
        raise NavigationPlanError(
            "traversed centerline support has a different shape"
        )
    merge_resolution = max(resolution, TRAVERSED_GRAPH_MERGE_RESOLUTION_M)
    node_by_bucket: dict[tuple[int, int], int] = {}
    buckets: list[tuple[int, int]] = []
    nodes: list[tuple[float, float]] = []
    component_nodes: list[list[int]] = []
    for component in _continuous_traversed_components(paths):
        node_path: list[int] = []
        for point in component:
            bucket = (
                int(math.floor((point[0] - origin[0]) / merge_resolution)),
                int(math.floor((point[1] - origin[1]) / merge_resolution)),
            )
            node_index = node_by_bucket.get(bucket)
            if node_index is None:
                node_index = len(nodes)
                if node_index >= MAX_TRAVERSED_GRAPH_NODES:
                    return None
                node_by_bucket[bucket] = node_index
                buckets.append(bucket)
                nodes.append(point)
            if not node_path or node_path[-1] != node_index:
                node_path.append(node_index)
        if node_path:
            component_nodes.append(node_path)
    if not nodes:
        return None
    adjacency: list[dict[int, float]] = [dict() for _ in nodes]

    def bounded_overlap(
        start: tuple[float, float], end: tuple[float, float]
    ) -> float:
        distance = math.dist(start, end)
        return min(
            _metric_overlap_cost(
                start, end, overlap_cost, origin, resolution
            ),
            distance * TRAVERSED_GRAPH_MAX_OVERLAP_PENALTY,
        )

    def add_certified_edge(first: int, second: int) -> None:
        if first == second or second in adjacency[first]:
            return
        if centerline_support is not None and not _metric_segment_within_mask(
            nodes[first], nodes[second], centerline_support, origin, resolution
        ):
            return
        safe, _ = _segment_clearance_certificate(
            nodes[first],
            nodes[second],
            blocked,
            clearance,
            origin,
            resolution,
            required_clearance_m,
        )
        if not safe:
            return
        distance = math.dist(nodes[first], nodes[second])
        edge_cost = distance + bounded_overlap(nodes[first], nodes[second])
        adjacency[first][second] = edge_cost
        adjacency[second][first] = edge_cost

    for node_path in component_nodes:
        for first, second in zip(node_path[:-1], node_path[1:]):
            add_certified_edge(first, second)
    bucket_radius = int(
        math.ceil(TRAVERSED_GRAPH_REVISIT_RADIUS_M / merge_resolution)
    )
    for first, (bucket_x, bucket_y) in enumerate(buckets):
        for offset_x in range(-bucket_radius, bucket_radius + 1):
            for offset_y in range(-bucket_radius, bucket_radius + 1):
                second = node_by_bucket.get(
                    (bucket_x + offset_x, bucket_y + offset_y)
                )
                if second is None or second <= first:
                    continue
                if (
                    math.dist(nodes[first], nodes[second])
                    <= TRAVERSED_GRAPH_REVISIT_RADIUS_M + 1.0e-9
                ):
                    add_certified_edge(first, second)
        if progress_check is not None and first % 256 == 0:
            progress_check()

    costs = [math.inf] * len(nodes)
    parents = [-1] * len(nodes)
    queue: list[tuple[float, int]] = []
    start_distances = [math.dist(start_xy, point) for point in nodes]
    goal_distances = [math.dist(goal_xy, point) for point in nodes]
    eligible_start_distances = [
        distance
        for distance in start_distances
        if distance <= max_endpoint_distance_m + 1.0e-9
    ]
    eligible_goal_distances = [
        distance
        for distance in goal_distances
        if distance <= goal_limit + 1.0e-9
    ]
    if not eligible_start_distances or not eligible_goal_distances:
        return None
    start_attachment_limit = min(
        max_endpoint_distance_m,
        min(eligible_start_distances)
        + TRAVERSED_GRAPH_ENDPOINT_NEAREST_SLACK_M,
    )
    goal_attachment_limit = min(
        goal_limit,
        min(eligible_goal_distances) + TRAVERSED_GRAPH_ENDPOINT_NEAREST_SLACK_M,
    )
    for node_index, distance in enumerate(start_distances):
        if distance > start_attachment_limit + 1.0e-9:
            continue
        safe, _ = _segment_clearance_certificate(
            start_xy,
            nodes[node_index],
            blocked,
            clearance,
            origin,
            resolution,
            required_clearance_m,
        )
        if safe:
            cost = distance + bounded_overlap(start_xy, nodes[node_index])
            costs[node_index] = cost
            heapq.heappush(queue, (cost, node_index))
    targets = {
        node_index
        for node_index, distance in enumerate(goal_distances)
        if distance <= goal_attachment_limit + 1.0e-9
    }
    if not queue or not targets:
        return None
    best_target: int | None = None
    best_total = math.inf
    expanded = 0
    while queue:
        cost, node_index = heapq.heappop(queue)
        if cost > costs[node_index] + 1.0e-12:
            continue
        if cost > best_total + 1.0e-12:
            break
        if node_index in targets:
            total = (
                cost
                + goal_distances[node_index]
                + bounded_overlap(nodes[node_index], goal_xy)
            )
            if total < best_total:
                best_total = total
                best_target = node_index
        for neighbor, edge_cost in adjacency[node_index].items():
            candidate = cost + edge_cost
            if candidate + 1.0e-12 >= costs[neighbor]:
                continue
            costs[neighbor] = candidate
            parents[neighbor] = node_index
            heapq.heappush(queue, (candidate, neighbor))
        expanded += 1
        if progress_check is not None and expanded % 256 == 0:
            progress_check()
    if best_target is None:
        return None
    route_indices: list[int] = []
    node_index = best_target
    while node_index >= 0:
        route_indices.append(node_index)
        node_index = parents[node_index]
    route_indices.reverse()
    return (
        [nodes[index] for index in route_indices],
        float(start_distances[route_indices[0]]),
        float(goal_distances[route_indices[-1]]),
    )


def _traversed_raster_route_between(
    centerline_support: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_start_distance_m: float,
    max_goal_distance_m: float,
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
    progress_check: Callable[[], None] | None,
    overlap_cost: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], float, float] | None:
    """Find the shortest safe route in the union of prior centreline tubes.

    Unlike a temporal trajectory, this raster has no notion of which lap was
    driven first.  Revisits therefore join naturally, while A* cannot replay a
    long chronological loop merely to reach an earlier occurrence of a place.
    """

    if centerline_support.shape != blocked.shape:
        raise NavigationPlanError(
            "traversed centerline support has a different shape"
        )
    route_mask = (
        centerline_support
        & ~blocked
        & (clearance + 1.0e-9 >= required_clearance_m)
    )
    start, start_distance = _nearest_cell_xy(
        route_mask,
        start_xy,
        origin,
        resolution,
        max_start_distance_m,
    )
    if start is None:
        return None
    reachable = _reachable(route_mask, start)
    goal, goal_distance = _nearest_cell_xy(
        reachable,
        goal_xy,
        origin,
        resolution,
        max_goal_distance_m,
    )
    if goal is None:
        return None
    # The old map is useful as a tie-breaker, but traversal evidence has
    # already certified this corridor.  A bounded per-metre penalty prevents
    # a few stale pixels from making a many-lap route look cheaper.
    bounded_overlap = (
        None
        if overlap_cost is None
        else np.minimum(
            overlap_cost,
            TRAVERSED_GRAPH_MAX_OVERLAP_PENALTY,
        )
    )
    # The support mask is the union of noisy/repeated traversals. A shortest
    # path alone tends to ride one edge of that union, leaving almost no
    # tracking allowance at doorways. Distance to the support boundary gives
    # a scene-independent medial preference; the hard route mask above still
    # enforces current obstacles, unknown space, and footprint clearance.
    support_clearance = ndimage.distance_transform_edt(
        np.pad(centerline_support, 1, constant_values=False)
    )[1:-1, 1:-1].astype(np.float64) * resolution
    try:
        grid_path, _ = _astar(
            route_mask,
            support_clearance,
            start,
            goal,
            resolution,
            0.0,
            TRAVERSED_CENTERLINE_WEIGHT,
            progress_check=progress_check,
            overlap_cost=bounded_overlap,
        )
    except NavigationPlanError:
        return None
    return (
        [_xy(cell, origin, resolution) for cell in grid_path],
        float(start_distance),
        float(goal_distance),
    )


def _snapshot_wall_mask(
    snapshot: Mapping[str, Any], shape: tuple[int, int]
) -> np.ndarray:
    """Return the optional structural-wall layer without changing planning."""

    value = snapshot.get("wall_mask")
    if value is None:
        layers = snapshot.get("layers")
        value = layers.get("wall") if isinstance(layers, Mapping) else None
    if value is None:
        return np.zeros(shape, dtype=np.bool_)
    wall = np.asarray(value, dtype=np.bool_)
    if wall.shape != shape:
        raise NavigationPlanError("wall_mask has a different shape")
    return wall


def _optional_navigation_obstacle_mask(
    snapshot: Mapping[str, Any],
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Validate one private, non-SLAM obstacle layer."""

    value = snapshot.get(key)
    if value is None:
        return np.zeros(shape, dtype=np.bool_)
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape != shape:
        raise NavigationPlanError(f"{key} has a different shape")
    return mask


def _with_execution_stall_obstacles(
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply invocation-local collision evidence after history overrides."""

    obstacle, free, _resolution, _origin = _snapshot_geometry(snapshot)
    stalls = _optional_navigation_obstacle_mask(
        snapshot, "navigation_stall_obstacle_mask", obstacle.shape
    )
    if not np.any(stalls):
        return snapshot
    result = dict(snapshot)
    result["obstacle_mask"] = np.asarray(obstacle | stalls, dtype=np.bool_)
    result["free_mask"] = np.asarray(free & ~stalls, dtype=np.bool_)
    return result


def _replay_reference_traversed_evidence(
    snapshot: Mapping[str, Any], reference_plan: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    """Rebuild the map-only plan's private traversed-space interpretation.

    ``plan_clearance_path`` may prove that stale occupancy is inconsistent
    with a continuous history of the base physically sweeping that space.
    Exact live-depth refinement must use that same static-map interpretation;
    otherwise it silently restores the stale cells and rejects the route
    before testing the current depth points. All masks are recomputed from
    the snapshot and checked against the reference plan's audit counters.
    """

    if not bool(reference_plan.get("traversed_route_evidence_used", False)):
        return _with_execution_stall_obstacles(snapshot), None

    trials = reference_plan.get("planning_evidence_trials")
    raw_index = reference_plan.get("planning_evidence_trial_index")
    if (
        not isinstance(trials, Sequence)
        or isinstance(trials, (str, bytes))
        or isinstance(raw_index, (bool, np.bool_))
    ):
        raise NavigationPlanError(
            "reference plan has invalid traversed-evidence selection"
        )
    try:
        trial_index = int(raw_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NavigationPlanError(
            "reference plan has invalid traversed-evidence selection"
        ) from exc
    if trial_index < 0 or trial_index >= len(trials):
        raise NavigationPlanError(
            "reference plan has invalid traversed-evidence selection"
        )
    mode = str(trials[trial_index])
    allowed_modes = {
        "traversed_base_footprint",
        "traversed_nonwall_obstacle_override",
        "traversed_swept_space_override",
    }
    if mode not in allowed_modes:
        raise NavigationPlanError(
            "reference plan did not select an auditable traversed-evidence mode"
        )

    obstacle, free, resolution, origin = _snapshot_geometry(snapshot)
    paths = _traversed_paths(snapshot)
    radius = _finite(reference_plan.get("robot_radius_m"), "robot_radius_m")
    if radius < 0.0 or radius > 1.5 or not paths:
        raise NavigationPlanError(
            "reference plan's traversed-evidence source is unavailable"
        )
    source_free_cells = int(np.count_nonzero(free & ~obstacle))
    source_obstacle_cells = int(np.count_nonzero(obstacle))
    wall = _snapshot_wall_mask(snapshot, obstacle.shape) & obstacle
    (
        adjusted_free,
        promoted_cells,
        segment_count,
        evidence,
    ) = _promote_traversed_footprint_free(
        obstacle, free, paths, origin, resolution, radius
    )
    conflicting = evidence & obstacle
    if mode == "traversed_base_footprint":
        overridden = np.zeros(obstacle.shape, dtype=np.bool_)
    elif mode == "traversed_nonwall_obstacle_override":
        overridden = conflicting & ~wall
        adjusted_free = adjusted_free | (evidence & ~wall)
    else:
        overridden = conflicting
        adjusted_free = adjusted_free | evidence
    adjusted_obstacle = obstacle & ~overridden
    adjusted_free = adjusted_free & ~adjusted_obstacle

    audit = {
        "schema": "reference_traversed_evidence_v1",
        "mode": mode,
        "path_count": len(paths),
        "point_count": sum(len(path) for path in paths),
        "evidence_cells": int(np.count_nonzero(evidence)),
        "promoted_cells": int(promoted_cells),
        "segment_count": int(segment_count),
        "conflicting_obstacle_cells": int(np.count_nonzero(conflicting)),
        "conflicting_wall_cells": int(np.count_nonzero(conflicting & wall)),
        "overridden_nonwall_obstacle_cells": int(
            np.count_nonzero(overridden & ~wall)
        ),
        "overridden_wall_obstacle_cells": int(
            np.count_nonzero(overridden & wall)
        ),
        "source_free_cells": source_free_cells,
        "source_obstacle_cells": source_obstacle_cells,
    }
    reference_fields = {
        "traversed_route_path_count": "path_count",
        "traversed_route_point_count": "point_count",
        "traversed_route_evidence_cells": "evidence_cells",
        "traversed_route_promoted_cells": "promoted_cells",
        "traversed_route_segment_count": "segment_count",
        "traversed_route_conflicting_obstacle_cells": (
            "conflicting_obstacle_cells"
        ),
        "traversed_route_conflicting_wall_cells": "conflicting_wall_cells",
        "traversed_route_overridden_nonwall_obstacle_cells": (
            "overridden_nonwall_obstacle_cells"
        ),
        "traversed_route_overridden_wall_obstacle_cells": (
            "overridden_wall_obstacle_cells"
        ),
        "source_free_cells": "source_free_cells",
        "source_obstacle_cells": "source_obstacle_cells",
    }
    for reference_key, audit_key in reference_fields.items():
        raw_value = reference_plan.get(reference_key)
        if isinstance(raw_value, (bool, np.bool_)):
            raise NavigationPlanError(
                f"reference plan has invalid {reference_key} audit value"
            )
        try:
            reference_value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NavigationPlanError(
                f"reference plan has invalid {reference_key} audit value"
            ) from exc
        if reference_value != audit[audit_key]:
            raise NavigationPlanError(
                f"reference plan's {reference_key} audit does not match the map"
            )
    evidence_used = bool(
        audit["promoted_cells"]
        or audit["overridden_nonwall_obstacle_cells"]
        or audit["overridden_wall_obstacle_cells"]
    )
    if not evidence_used:
        raise NavigationPlanError(
            "reference plan claims traversed evidence without a map change"
        )

    private_snapshot = dict(snapshot)
    private_snapshot["obstacle_mask"] = np.asarray(
        adjusted_obstacle, dtype=np.bool_
    ).copy()
    private_snapshot["free_mask"] = np.asarray(
        adjusted_free, dtype=np.bool_
    ).copy()
    return _with_execution_stall_obstacles(private_snapshot), audit


def _segment_clearance_lower_bound(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    clearance_m: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
) -> float:
    """Certify a conservative continuous clearance from center samples."""

    length = math.dist(start_xy, end_xy)
    pieces = max(1, int(math.ceil(length / max(1e-9, 0.25 * resolution))))
    lower = math.inf
    for index in range(pieces + 1):
        alpha = float(index) / float(pieces)
        point = (
            start_xy[0] + alpha * (end_xy[0] - start_xy[0]),
            start_xy[1] + alpha * (end_xy[1] - start_xy[1]),
        )
        cell = _cell(point, origin, resolution)
        if not _inside(clearance_m.shape, cell):
            return 0.0
        center = _xy(cell, origin, resolution)
        lower = min(
            lower,
            float(clearance_m[cell]) - math.dist(point, center),
        )
    # The true distance-to-obstacle function is 1-Lipschitz. Every point on
    # the segment is at most half a sample interval from an audited point.
    return max(0.0, lower - 0.5 * length / float(pieces))


def _segment_clearance_certificate(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    blocked: np.ndarray,
    clearance_m: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_m: float,
    *,
    measure_reserve_m: float = 0.0,
) -> tuple[bool, float]:
    """Prove a continuous segment stays ``required_m`` from blocked space."""

    height, width = blocked.shape
    map_x_min, map_y_min = origin
    map_x_max = map_x_min + width * resolution
    map_y_max = map_y_min + height * resolution
    boundary_clearance = min(
        start_xy[0] - map_x_min,
        end_xy[0] - map_x_min,
        map_x_max - start_xy[0],
        map_x_max - end_xy[0],
        start_xy[1] - map_y_min,
        end_xy[1] - map_y_min,
        map_y_max - start_xy[1],
        map_y_max - end_xy[1],
    )
    if boundary_clearance <= 1e-12 or boundary_clearance + 1e-9 < required_m:
        return False, max(0.0, float(boundary_clearance))

    search = max(0.0, float(required_m)) + max(0.0, float(measure_reserve_m))
    x0 = min(start_xy[0], end_xy[0]) - search
    x1 = max(start_xy[0], end_xy[0]) + search
    y0 = min(start_xy[1], end_xy[1]) - search
    y1 = max(start_xy[1], end_xy[1]) + search
    # The search box is closed.  When x0/y0 lands exactly on a grid boundary,
    # the cell immediately below that boundary can still touch the segment.
    col0 = max(0, int(math.floor((x0 - map_x_min) / resolution)) - 1)
    col1 = min(width - 1, int(math.floor((x1 - map_x_min) / resolution)))
    row0 = max(0, int(math.floor((y0 - map_y_min) / resolution)) - 1)
    row1 = min(height - 1, int(math.floor((y1 - map_y_min) / resolution)))
    nearest = math.inf
    if row0 <= row1 and col0 <= col1:
        local_rows, local_columns = np.nonzero(
            blocked[row0 : row1 + 1, col0 : col1 + 1]
        )
        if local_rows.size:
            rows = local_rows.astype(np.float64) + row0
            columns = local_columns.astype(np.float64) + col0
            rect_x_min = map_x_min + columns * resolution
            rect_y_min = map_y_min + rows * resolution
            distances = _segment_to_rectangles_distance(
                start_xy,
                end_xy,
                rect_x_min,
                rect_y_min,
                rect_x_min + resolution,
                rect_y_min + resolution,
            )
            nearest = float(np.min(distances))
            if nearest <= 1e-12 or nearest + 1e-9 < required_m:
                return False, max(0.0, nearest)

    sampled_lower = _segment_clearance_lower_bound(
        start_xy,
        end_xy,
        clearance_m,
        origin,
        resolution,
    )
    # Rectangles outside the search box are at least `search` away. Measure
    # actual spare room for execution instead of rounding it down to the hard
    # radius, which would give the controller a zero-width tracking corridor.
    certified = max(min(search, nearest), sampled_lower)
    certified = min(certified, boundary_clearance)
    if math.isfinite(nearest):
        certified = min(certified, nearest)
    return True, max(0.0, float(certified))


def _simplify(
    path: Sequence[tuple[int, int]],
    traversable: np.ndarray,
    blocked: np.ndarray,
    clearance_m: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    minimum_clearance_m: float,
    clearance_weight: float,
    edge_clearances_m: Sequence[float],
    overlap_cost: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return list(path)
    prefix_cost = np.zeros(len(path), dtype=np.float64)
    prefix_overlap = np.zeros(len(path), dtype=np.float64)
    for index in range(1, len(path)):
        prefix_cost[index] = prefix_cost[index - 1] + _path_cost(
            path[index - 1 : index + 1],
            clearance_m,
            resolution,
            minimum_clearance_m,
            clearance_weight,
            overlap_cost,
        )
        prefix_overlap[index] = prefix_overlap[index - 1] + _path_overlap_cost(
            path[index - 1 : index + 1], overlap_cost, resolution
        )
    result = [path[0]]
    anchor = 0
    max_lookahead_cells = max(
        2, int(math.ceil(SIMPLIFY_MAX_LOOKAHEAD_M / resolution))
    )
    while anchor < len(path) - 1:
        candidate = min(len(path) - 1, anchor + max_lookahead_cells)
        while candidate > anchor + 1:
            shortcut = _line_cells(
                traversable, path[anchor], path[candidate]
            )
            if shortcut is not None:
                original_bottleneck = min(
                    edge_clearances_m[anchor:candidate],
                    default=minimum_clearance_m,
                )
                continuous_ok, _ = _segment_clearance_certificate(
                    _xy(path[anchor], origin, resolution),
                    _xy(path[candidate], origin, resolution),
                    blocked,
                    clearance_m,
                    origin,
                    resolution,
                    max(minimum_clearance_m, original_bottleneck),
                )
                shortcut_cost = _path_cost(
                    shortcut,
                    clearance_m,
                    resolution,
                    minimum_clearance_m,
                    clearance_weight,
                    overlap_cost,
                )
                original_cost = float(
                    prefix_cost[candidate] - prefix_cost[anchor]
                )
                # Preserve the A* clearance detour while allowing small
                # staircase-cost differences. The continuous certificate
                # above also preserves the original subpath bottleneck.
                if (
                    continuous_ok
                    and _path_overlap_cost(shortcut, overlap_cost, resolution)
                    <= prefix_overlap[candidate] - prefix_overlap[anchor] + 1e-9
                    and shortcut_cost
                    <= original_cost * SIMPLIFY_MAX_CLEARANCE_COST_STRETCH
                    + 1e-9
                ):
                    break
            candidate -= 1
        result.append(path[candidate])
        anchor = candidate
    return result


def _subdivide(
    points: Sequence[tuple[float, float]], max_segment_m: float
) -> list[tuple[float, float]]:
    if not points:
        return []
    output = [points[0]]
    for end in points[1:]:
        start = output[-1]
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        pieces = max(1, int(math.ceil(distance / max_segment_m)))
        for index in range(1, pieces + 1):
            alpha = float(index) / float(pieces)
            output.append(
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
            )
    return output


def _metric_segment_within_mask(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    allowed: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
) -> bool:
    """Check an exact metric centerline against a raster support mask."""

    start = _pair(start_xy, "supported segment start")
    end = _pair(end_xy, "supported segment end")
    start_cell = _cell(start, origin, resolution)
    end_cell = _cell(end, origin, resolution)
    if (
        not _inside(allowed.shape, start_cell)
        or not _inside(allowed.shape, end_cell)
        or not allowed[start_cell]
        or not allowed[end_cell]
        or _line_cells(allowed, start_cell, end_cell) is None
    ):
        return False
    distance = math.dist(start, end)
    samples = max(1, int(math.ceil(distance / (0.25 * resolution))))
    alpha = np.linspace(0.0, 1.0, samples + 1, dtype=np.float64)
    columns = np.floor(
        (start[0] + alpha * (end[0] - start[0]) - origin[0]) / resolution
    ).astype(np.int64)
    rows = np.floor(
        (start[1] + alpha * (end[1] - start[1]) - origin[1]) / resolution
    ).astype(np.int64)
    return bool(
        np.all(rows >= 0)
        and np.all(columns >= 0)
        and np.all(rows < allowed.shape[0])
        and np.all(columns < allowed.shape[1])
        and np.all(allowed[rows, columns])
    )


def _polyline_unsupported_length(
    points: Sequence[Sequence[float]],
    allowed: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
) -> float:
    """Measure route length whose centerline leaves a support raster."""

    route = [_pair(point, "comparison route point") for point in points]
    unsupported = 0.0
    for start, end in zip(route[:-1], route[1:]):
        length = math.dist(start, end)
        if length <= 1.0e-12:
            continue
        samples = max(1, int(math.ceil(length / (0.5 * resolution))))
        alpha = (np.arange(samples, dtype=np.float64) + 0.5) / samples
        columns = np.floor(
            (start[0] + alpha * (end[0] - start[0]) - origin[0]) / resolution
        ).astype(np.int64)
        rows = np.floor(
            (start[1] + alpha * (end[1] - start[1]) - origin[1]) / resolution
        ).astype(np.int64)
        inside = (
            (rows >= 0)
            & (columns >= 0)
            & (rows < allowed.shape[0])
            & (columns < allowed.shape[1])
        )
        supported = np.zeros(samples, dtype=np.bool_)
        supported[inside] = allowed[rows[inside], columns[inside]]
        unsupported += length * float(np.count_nonzero(~supported)) / samples
    return float(unsupported)


def _simplify_certified_metric_route(
    points: Sequence[tuple[float, float]],
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
    max_segment_m: float,
    progress_check: Callable[[], None] | None,
    overlap_cost: np.ndarray | None = None,
    centerline_support: np.ndarray | None = None,
) -> list[tuple[float, float]] | None:
    """Greedily shorten a taught route without leaving certified free space."""

    dense = _subdivide(points, max_segment_m)
    if len(dense) <= 1:
        return dense
    prefix_overlap = np.concatenate(([0.0], np.cumsum([
        _metric_overlap_cost(start, end, overlap_cost, origin, resolution)
        for start, end in zip(dense[:-1], dense[1:])
    ])))
    output = [dense[0]]
    index = 0
    checks = 0
    while index < len(dense) - 1:
        best_index: int | None = None
        traversed_length = 0.0
        for candidate_index in range(index + 1, len(dense)):
            traversed_length += math.dist(
                dense[candidate_index - 1], dense[candidate_index]
            )
            if traversed_length > max_segment_m + 1.0e-9:
                break
            direct_length = math.dist(dense[index], dense[candidate_index])
            if direct_length > max_segment_m + 1.0e-9:
                continue
            safe, _ = _segment_clearance_certificate(
                dense[index],
                dense[candidate_index],
                blocked,
                clearance,
                origin,
                resolution,
                required_clearance_m,
            )
            if safe and centerline_support is not None:
                safe = _metric_segment_within_mask(
                    dense[index],
                    dense[candidate_index],
                    centerline_support,
                    origin,
                    resolution,
                )
            checks += 1
            if progress_check is not None and checks % 128 == 0:
                progress_check()
            if safe and _metric_overlap_cost(
                dense[index], dense[candidate_index], overlap_cost, origin, resolution
            ) <= prefix_overlap[candidate_index] - prefix_overlap[index] + 1e-9:
                best_index = candidate_index
        if best_index is None:
            return None
        output.append(dense[best_index])
        index = best_index
    return output


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points[:-1], points[1:])
    )


def _turning_pocket_certificate(
    points: Sequence[tuple[float, float]],
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
) -> tuple[bool, list[float], int]:
    """Certify that every heading-change corner has room to rotate safely."""
    if len(points) < 3:
        return True, [], 0
    turn_clearances: list[float] = []
    turn_count = 0
    required = required_clearance_m + TURNING_CLEARANCE_RESERVE_M

    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        incoming_norm = math.hypot(*incoming)
        outgoing_norm = math.hypot(*outgoing)
        if incoming_norm <= 1.0e-6 or outgoing_norm <= 1.0e-6:
            continue
        cosine = max(
            -1.0,
            min(
                1.0,
                (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
                / (incoming_norm * outgoing_norm),
            ),
        )
        angle_deg = math.degrees(math.acos(cosine))
        if angle_deg < TURN_MIN_ANGLE_DEG:
            continue
        turn_count += 1
        # The zero-length segment tests the complete disk against closed
        # obstacle/unknown cell rectangles, including sub-cell offsets.
        safe, local_clearance = _segment_clearance_certificate(
            current, current, blocked, clearance, origin, resolution, required
        )
        turn_clearances.append(float(local_clearance))
        if not safe:
            return False, turn_clearances, turn_count
    return True, turn_clearances, turn_count


def _repair_turning_pockets(
    points: Sequence[tuple[float, float]],
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    required_clearance_m: float,
    protected_points: Sequence[tuple[float, float]] = (),
    progress_check: Callable[[], None] | None = None,
    overlap_cost: np.ndarray | None = None,
    centerline_support: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], int]:
    """Relocate tight corners, certifying both new edges and adjacent turns."""
    route = list(points)
    repaired = 0
    turn_required = required_clearance_m + TURNING_CLEARANCE_RESERVE_M

    def turn_safe(path, index):
        if index <= 0 or index >= len(path) - 1:
            return True
        return _turning_pocket_certificate(
            path[index - 1:index + 2], blocked, clearance, origin, resolution,
            required_clearance_m,
        )[0]

    for index in range(1, len(route) - 1):
        if turn_safe(route, index):
            continue
        current = route[index]
        if any(math.dist(current, point) <= 1.0e-9 for point in protected_points):
            continue
        row, column = _cell(current, origin, resolution)
        cell_radius = int(math.ceil(TURNING_POCKET_SEARCH_M / resolution))
        row0, row1 = max(0, row - cell_radius), min(blocked.shape[0], row + cell_radius + 1)
        col0, col1 = max(0, column - cell_radius), min(blocked.shape[1], column + cell_radius + 1)
        local = (~blocked[row0:row1, col0:col1]) & (
            clearance[row0:row1, col0:col1]
            >= turn_required - math.sqrt(2.0) * resolution
        )
        if centerline_support is not None:
            local &= centerline_support[row0:row1, col0:col1]
        rows, columns = np.nonzero(local)
        candidates = [_xy((int(r) + row0, int(c) + col0), origin, resolution)
                      for r, c in zip(rows, columns)]
        candidates = [point for point in candidates
                      if math.dist(point, current) <= TURNING_POCKET_SEARCH_M + 1.0e-9]
        previous, following = route[index - 1], route[index + 1]
        candidates.sort(key=lambda point: (
            math.dist(previous, point) + math.dist(point, following)
            + math.dist(current, point)
            + _metric_overlap_cost(previous, point, overlap_cost, origin, resolution)
            + _metric_overlap_cost(point, following, overlap_cost, origin, resolution),
            point,
        ))
        previous_was_safe = turn_safe(route, index - 1)
        following_was_safe = turn_safe(route, index + 1)
        for candidate_index, candidate in enumerate(candidates[:TURNING_POCKET_MAX_CANDIDATES]):
            if progress_check is not None and candidate_index % 32 == 0:
                progress_check()
            if min(math.dist(previous, candidate), math.dist(candidate, following)) <= resolution:
                continue
            if not _segment_clearance_certificate(
                candidate, candidate, blocked, clearance, origin, resolution,
                turn_required,
            )[0]:
                continue
            if not all(_segment_clearance_certificate(
                start, end, blocked, clearance, origin, resolution, required_clearance_m
            )[0] for start, end in ((previous, candidate), (candidate, following))):
                continue
            if centerline_support is not None and not all(
                _metric_segment_within_mask(
                    start,
                    end,
                    centerline_support,
                    origin,
                    resolution,
                )
                for start, end in ((previous, candidate), (candidate, following))
            ):
                continue
            trial = list(route)
            trial[index] = candidate
            # Moving a waypoint can sharpen the two neighboring corners.
            # Do not trade one missing turning pocket for another.
            if previous_was_safe and not turn_safe(trial, index - 1):
                continue
            if following_was_safe and not turn_safe(trial, index + 1):
                continue
            route = trial
            repaired += 1
            break
    return route, repaired


def _plan_clearance_path_once(
    snapshot: Mapping[str, Any],
    goal_name: str,
    *,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    safety_margin_fallbacks_m: Sequence[float] = (),
    clearance_weight: float = DEFAULT_CLEARANCE_WEIGHT,
    arrival_tolerance_m: float = DEFAULT_ARRIVAL_TOLERANCE_M,
    max_segment_m: float = DEFAULT_MAX_SEGMENT_M,
    optimize_goal_standoff: bool = False,
    progress_check: Callable[[], None] | None = None,
    planning_resolution_m: float = PLANNING_RESOLUTION_M,
    use_traversed_evidence: bool = False,
    override_traversed_nonwall_obstacles: bool = False,
    override_traversed_all_obstacles: bool = False,
    require_turning_pockets: bool = False,
    prefer_traversed_route: bool = False,
    traversed_reference_path_xy_m: Sequence[Sequence[float]] = (),
    committed_traversed_path_xy_m: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Plan a collision-checked, high-clearance polyline to a marked place.

    The returned coordinates stay in the map provider's local metric frame.
    The full cell path is retained for audit; execution should use
    ``path_xy_m`` and re-read the provider pose between bounded segments.
    """

    if progress_check is not None:
        progress_check()
    if not isinstance(snapshot, Mapping):
        raise NavigationPlanError("navigation map snapshot must be a mapping")
    radius = _finite(robot_radius_m, "robot_radius_m")
    margin = _finite(safety_margin_m, "safety_margin_m")
    weight = _finite(clearance_weight, "clearance_weight")
    tolerance = _finite(arrival_tolerance_m, "arrival_tolerance_m")
    max_segment = _finite(max_segment_m, "max_segment_m")
    if not isinstance(optimize_goal_standoff, bool):
        raise NavigationPlanError("optimize_goal_standoff must be a boolean")
    if not isinstance(prefer_traversed_route, bool):
        raise NavigationPlanError("prefer_traversed_route must be a boolean")
    committed_path = [
        _pair(point, "committed traversed route point")
        for point in committed_traversed_path_xy_m
    ]
    if committed_path and len(committed_path) < 2:
        raise NavigationPlanError(
            "committed traversed route must contain at least two points"
        )
    if radius < 0.0 or radius > 1.5:
        raise NavigationPlanError("robot_radius_m must be in [0, 1.5]")
    if margin < 0.0 or margin > 1.0:
        raise NavigationPlanError("safety_margin_m must be in [0, 1]")
    try:
        fallback_values = list(safety_margin_fallbacks_m)
    except TypeError as exc:
        raise NavigationPlanError(
            "safety_margin_fallbacks_m must be a sequence"
        ) from exc
    margin_trials = [margin]
    for index, raw_fallback in enumerate(fallback_values):
        fallback = _finite(
            raw_fallback, f"safety_margin_fallbacks_m[{index}]"
        )
        if fallback < 0.0 or fallback >= margin_trials[-1]:
            raise NavigationPlanError(
                "safety margin fallbacks must be non-negative and strictly "
                "decreasing"
            )
        margin_trials.append(fallback)
    if weight < 0.0 or weight > 100.0:
        raise NavigationPlanError("clearance_weight must be in [0, 100]")
    if tolerance <= 0.0 or tolerance > 2.0:
        raise NavigationPlanError("arrival_tolerance_m must be in (0, 2]")
    if max_segment <= 0.05 or max_segment > SIMPLIFY_MAX_LOOKAHEAD_M:
        raise NavigationPlanError(
            f"max_segment_m must be in (0.05, {SIMPLIFY_MAX_LOOKAHEAD_M:g}]"
        )
    lifecycle = snapshot.get("lifecycle")
    # A frame-level feature-match miss is advisory.  Explicit lifecycle holds
    # below are the mapper-owned authority for whether navigation must stop.
    lifecycle_recovery_hold = (
        lifecycle.get("recovery_hold")
        if isinstance(lifecycle, Mapping)
        else False
    )
    lifecycle_uncertain_hold = (
        lifecycle.get("uncertain_hold")
        if isinstance(lifecycle, Mapping)
        else False
    )
    if bool(snapshot.get("recovery_hold", False)) or bool(
        lifecycle_recovery_hold
    ):
        raise NavigationPlanError("map is relocalizing; navigation is held")
    if bool(snapshot.get("uncertain_hold", False)) or bool(
        lifecycle_uncertain_hold
    ):
        raise NavigationPlanError(
            "map localization is uncertain; navigation is held"
        )

    obstacle, free, source_resolution, origin = _snapshot_geometry(snapshot)
    original_blocked = ~free | obstacle
    pose_x, pose_y, pose_yaw = _pose(snapshot)
    goal_x, goal_y, duplicate_count = _goal(snapshot, goal_name)
    traversed_paths = _traversed_paths(snapshot)
    traversed_point_count = sum(len(path) for path in traversed_paths)
    traversed_start_distance_m = _distance_to_traversed_paths(
        (pose_x, pose_y), traversed_paths
    )
    traversed_goal_distance_m = _distance_to_traversed_paths(
        (goal_x, goal_y), traversed_paths
    )
    source_free_cells = int(np.count_nonzero(free & ~obstacle))
    source_obstacle_cells = int(np.count_nonzero(obstacle))
    wall = _snapshot_wall_mask(snapshot, obstacle.shape) & obstacle
    traversed_promoted_cells = 0
    traversed_evidence_cells = 0
    traversed_segment_count = 0
    traversed_conflicting_obstacle_cells = 0
    traversed_conflicting_wall_cells = 0
    traversed_overridden_nonwall_obstacle_cells = 0
    traversed_overridden_wall_obstacle_cells = 0
    traversed_guidance: np.ndarray | None = None
    traversed_guidance_selected = False
    traversed_raster_route_used = False
    traversed_reference_unsupported_length_m: float | None = None
    traversed_reference_unsupported_fraction: float | None = None
    if use_traversed_evidence and traversed_paths:
        (
            free,
            traversed_promoted_cells,
            traversed_segment_count,
            traversed_evidence,
        ) = (
            _promote_traversed_footprint_free(
                obstacle,
                free,
                traversed_paths,
                origin,
                source_resolution,
                radius,
            )
        )
        conflicting = traversed_evidence & obstacle
        traversed_evidence_cells = int(np.count_nonzero(traversed_evidence))
        traversed_conflicting_obstacle_cells = int(
            np.count_nonzero(conflicting)
        )
        traversed_conflicting_wall_cells = int(
            np.count_nonzero(conflicting & wall)
        )
        if (
            override_traversed_nonwall_obstacles
            or override_traversed_all_obstacles
        ):
            overridden = (
                conflicting
                if override_traversed_all_obstacles
                else conflicting & ~wall
            )
            traversed_overridden_nonwall_obstacle_cells = int(
                np.count_nonzero(overridden & ~wall)
            )
            traversed_overridden_wall_obstacle_cells = int(
                np.count_nonzero(overridden & wall)
            )
            obstacle = obstacle & ~overridden
            free = free | (
                traversed_evidence
                if override_traversed_all_obstacles
                else traversed_evidence & ~wall
            )
    # Fresh depth and collision evidence cannot be erased by an old walked
    # trail. Both are private planning inputs, never mutations of the SLAM map.
    protected = _optional_navigation_obstacle_mask(
        snapshot, "navigation_obstacle_mask", obstacle.shape
    )
    execution_stalls = _optional_navigation_obstacle_mask(
        snapshot, "navigation_stall_obstacle_mask", obstacle.shape
    )
    protected = protected | execution_stalls
    obstacle = obstacle | protected
    free = free & ~protected
    obstacle, free, resolution, downsample = _coarsen(
        obstacle, free, source_resolution, planning_resolution_m
    )
    overlap_cost = None
    if use_traversed_evidence:
        original_coarse, _, _, _ = _coarsen(
            original_blocked, ~original_blocked,
            source_resolution, planning_resolution_m,
        )
        overlap_cost = _sweep_overlap_cost(original_coarse, resolution)
    if progress_check is not None:
        progress_check()
    blocked = ~(free & ~obstacle)
    start_requested = _cell((pose_x, pose_y), origin, resolution)
    goal_requested = _cell((goal_x, goal_y), origin, resolution)
    if not _inside(free.shape, start_requested):
        raise NavigationPlanError("current map pose lies outside the occupancy grid")
    if not _inside(free.shape, goal_requested):
        raise NavigationPlanError("marked place lies outside the occupancy grid")

    # Distance to both obstacles and unknown cells. Padding makes the map edge
    # an obstacle even when a backend publishes free cells up to its boundary.
    free_for_distance = np.pad(free & ~obstacle, 1, constant_values=False)
    clearance = ndimage.distance_transform_edt(free_for_distance)[
        1:-1, 1:-1
    ].astype(np.float64) * resolution
    # EDT measures center-to-center. Occupied/unknown cells represent squares,
    # so subtract their circumradius to obtain a conservative lower bound to
    # the cell area. Subtracting only half a side is unsafe for diagonal cells.
    clearance = np.maximum(
        0.0,
        clearance - (math.sqrt(2.0) * 0.5 * resolution),
    )
    if prefer_traversed_route and traversed_paths:
        traversed_guidance, _ = _swept_footprint_mask(
            blocked.shape,
            traversed_paths,
            origin,
            resolution,
            TRAVERSED_GUIDANCE_TUBE_M,
            max_link_m=TRAVERSED_PATH_MAX_LINK_M,
        )
        if traversed_reference_path_xy_m:
            reference_path = [
                _pair(point, "traversed reference route point")
                for point in traversed_reference_path_xy_m
            ]
            reference_length = _polyline_length(reference_path)
            traversed_reference_unsupported_length_m = (
                _polyline_unsupported_length(
                    reference_path,
                    traversed_guidance,
                    origin,
                    resolution,
                )
            )
            traversed_reference_unsupported_fraction = (
                traversed_reference_unsupported_length_m / reference_length
                if reference_length > 1.0e-12
                else 0.0
            )
    if progress_check is not None:
        progress_check()

    requested_margin = margin
    selected: tuple[
        int,
        float,
        np.ndarray,
        tuple[int, int],
        tuple[int, int],
        float,
        float,
    ] | None = None
    nearest_failure = "no traversable cells"
    last_reachable: np.ndarray | None = None
    traversed_direct_route: list[tuple[float, float]] | None = None
    traversed_direct_start_safe_xy: tuple[float, float] | None = None
    traversed_graph_route_used = False
    committed_traversed_route_selected = False
    for margin_trial_index, effective_margin in enumerate(margin_trials):
        minimum_clearance = radius + effective_margin
        start_egress_clearance = radius + min(
            effective_margin, START_EGRESS_SAFETY_MARGIN_M
        )
        traversable = free & ~obstacle & (
            clearance + 1e-9 >= minimum_clearance
        )
        start_requested_center = _xy(start_requested, origin, resolution)
        start_center_offset = math.dist(
            (pose_x, pose_y), start_requested_center
        )
        start_exact_safe = bool(
            traversable[start_requested]
            and clearance[start_requested] + 1e-9
            >= minimum_clearance + start_center_offset
        )
        if start_exact_safe:
            start = start_requested
            start_snap_exact = 0.0
        else:
            start, start_snap_exact = _nearest_cell_xy(
                traversable,
                (pose_x, pose_y),
                origin,
                resolution,
                tolerance,
            )
        if start is None:
            nearest_failure = "current pose has no footprint-safe free cell nearby"
            continue
        if start_snap_exact > 1e-9:
            safe_entry, entry_clearance_m = _segment_clearance_certificate(
                (pose_x, pose_y),
                _xy(start, origin, resolution),
                blocked,
                clearance,
                origin,
                resolution,
                start_egress_clearance,
            )
            if not safe_entry:
                nearest_failure = (
                    "current pose cannot safely enter the footprint-safe grid "
                    f"(connector_clearance={entry_clearance_m:.3f}m, "
                    f"required={start_egress_clearance:.3f}m, "
                    f"snap={start_snap_exact:.3f}m)"
                )
                continue
        reachable = _reachable(traversable, start)
        last_reachable = reachable
        if progress_check is not None:
            progress_check()
        goal_requested_center = _xy(goal_requested, origin, resolution)
        goal_center_offset = math.dist(
            (goal_x, goal_y), goal_requested_center
        )
        goal_exact_safe = bool(
            reachable[goal_requested]
            and clearance[goal_requested] + 1e-9
            >= minimum_clearance + goal_center_offset
        )
        if goal_exact_safe:
            goal = goal_requested
            goal_snap_exact = 0.0
        else:
            goal, goal_snap_exact = _nearest_cell_xy(
                reachable,
                (goal_x, goal_y),
                origin,
                resolution,
                tolerance,
            )
        if goal is None:
            nearest_failure = (
                "marked place is not connected by footprint-safe free space"
            )
            continue
        if optimize_goal_standoff:
            # The public tool promises arrival within a radius, not contact
            # with the exact mark. Select the reachable safe cell closest to
            # the current pose while retaining an explicit final tracking
            # reserve. This shortens every scene generically and still leaves
            # the executor enough tolerance to verify the marked point.
            standoff_limit = max(0.0, tolerance - GOAL_TRACKING_RESERVE_M)
            rows, columns = np.nonzero(reachable & traversable)
            if rows.size:
                candidate_x = origin[0] + (
                    columns.astype(np.float64) + 0.5
                ) * resolution
                candidate_y = origin[1] + (
                    rows.astype(np.float64) + 0.5
                ) * resolution
                mark_dx = candidate_x - goal_x
                mark_dy = candidate_y - goal_y
                candidate_standoff = np.hypot(mark_dx, mark_dy)
                admissible = candidate_standoff <= standoff_limit + 1e-9
                if np.any(admissible):
                    start_distance = np.hypot(
                        candidate_x - pose_x,
                        candidate_y - pose_y,
                    )
                    indices = np.flatnonzero(admissible)
                    chosen = int(
                        indices[
                            np.lexsort(
                                (
                                    candidate_standoff[indices],
                                    start_distance[indices],
                                )
                            )[0]
                        ]
                    )
                    goal = (int(rows[chosen]), int(columns[chosen]))
                    goal_snap_exact = float(candidate_standoff[chosen])
        # The exact-cell checks above use the 1-Lipschitz clearance bound:
        # clearance at a point can be lower than its cell-center value by at
        # most the center offset. Unsafe exact endpoints use the nearest safe
        # center selected directly in continuous metric coordinates.
        if start_snap_exact > tolerance + 1e-9:
            nearest_failure = "current pose needs a snap beyond arrival tolerance"
            continue
        if goal_snap_exact > tolerance + 1e-9:
            nearest_failure = "marked place needs a snap beyond arrival tolerance"
            continue
        selected = (
            margin_trial_index,
            effective_margin,
            traversable,
            start,
            goal,
            start_snap_exact,
            goal_snap_exact,
        )
        break
    if (
        (selected is None or prefer_traversed_route)
        and override_traversed_all_obstacles
        and traversed_paths
    ):
        taught_margin_trial_index = len(margin_trials) - 1
        taught_effective_margin = margin_trials[taught_margin_trial_index]
        taught_required_clearance = radius + taught_effective_margin
        taught_traversable = free & ~obstacle & (
            clearance + 1.0e-9 >= taught_required_clearance
        )
        taught_start_limit = (
            min(tolerance, TRAVERSED_PREFERENCE_MAX_ENDPOINT_M)
            if prefer_traversed_route
            else tolerance
        )
        taught_goal_limit = (
            max(0.0, tolerance - GOAL_TRACKING_RESERVE_M)
            if prefer_traversed_route
            else tolerance
        )
        # Preserve recorded connectivity before considering the binary union
        # of every lap.  The latter is useful when history is fragmented, but
        # at a narrow doorway two unrelated trail branches can lie within one
        # guidance tube and manufacture a through-wall shortcut.
        if committed_path:
            # A commitment names one concrete geometric branch, not merely
            # the fact that some history should be used. Densification keeps
            # execution-sized segments from being split by the raw-history
            # continuity threshold before projecting onto the remaining arc.
            committed_dense = _subdivide(
                committed_path, min(TRAVERSED_OVERRIDE_MAX_SEGMENT_M, 0.5)
            )
            taught_route = _traversed_route_between(
                [committed_dense],
                (pose_x, pose_y),
                (goal_x, goal_y),
                taught_start_limit,
                max_goal_endpoint_distance_m=taught_goal_limit,
            )
            traversed_graph_route_used = taught_route is not None
        else:
            taught_route = _traversed_graph_route_between(
                traversed_paths,
                (pose_x, pose_y),
                (goal_x, goal_y),
                taught_start_limit,
                blocked,
                clearance,
                origin,
                resolution,
                taught_required_clearance,
                progress_check,
                overlap_cost,
                max_goal_endpoint_distance_m=taught_goal_limit,
                centerline_support=traversed_guidance,
            )
            traversed_graph_route_used = taught_route is not None
            if taught_route is None:
                taught_route = _traversed_route_between(
                    traversed_paths,
                    (pose_x, pose_y),
                    (goal_x, goal_y),
                    taught_start_limit,
                    max_goal_endpoint_distance_m=taught_goal_limit,
                )
        if taught_route is not None:
            # ``graph_used`` describes topology-preserving history, while the
            # raster flag is reserved for the last-resort union fallback.
            traversed_graph_route_used = bool(
                traversed_graph_route_used or not traversed_raster_route_used
            )

        def certify_taught_route(
            candidate: tuple[
                list[tuple[float, float]], float, float
            ] | None,
        ) -> tuple[list[tuple[float, float]], float, float] | None:
            if candidate is None:
                return None
            route_core, route_start_snap_m, route_goal_snap_m = candidate
            selected_route_support, _ = _swept_footprint_mask(
                blocked.shape,
                [route_core],
                origin,
                resolution,
                TRAVERSED_SELECTED_ROUTE_TUBE_M,
                max_link_m=TRAVERSED_PATH_MAX_LINK_M,
            )
            simplified_route = _simplify_certified_metric_route(
                route_core,
                blocked,
                clearance,
                origin,
                resolution,
                taught_required_clearance,
                max_segment,
                progress_check,
                overlap_cost,
                centerline_support=selected_route_support,
            )
            exact_start = (pose_x, pose_y)
            start_connector_ok = False
            if simplified_route:
                start_connector_ok, _ = _segment_clearance_certificate(
                    exact_start,
                    simplified_route[0],
                    blocked,
                    clearance,
                    origin,
                    resolution,
                    taught_required_clearance,
                )
            if not simplified_route or not start_connector_ok:
                return None
            return simplified_route, route_start_snap_m, route_goal_snap_m

        certified_taught_route = certify_taught_route(taught_route)
        # The temporal graph preserves which side of a wall was driven, but
        # pose corrections can leave two observations of the same corridor a
        # few centimetres apart.  In that case the graph may replay an entire
        # old lap to reach a point which is spatially nearby.  Also evaluate
        # the shortest route in the union of taught centreline tubes when the
        # robot footprint is at least as wide as that union's possible join.
        # This dimensional guard matters: for a tiny synthetic robot, two
        # nearby tracks may genuinely lie on opposite sides of a thin wall.
        # For the full base, overlapping tubes are contained in space its
        # previously observed swept footprint already covered.  Every raster
        # candidate is still checked against current map/depth obstacles and
        # continuous footprint clearance before it can replace the graph.
        raster_join_is_footprint_supported = bool(
            taught_required_clearance + 1.0e-9
            >= 2.0 * TRAVERSED_GUIDANCE_TUBE_M
        )
        if (
            not committed_path
            and (
                certified_taught_route is None
                or raster_join_is_footprint_supported
            )
        ):
            raster_route = (
                _traversed_raster_route_between(
                    traversed_guidance,
                    (pose_x, pose_y),
                    (goal_x, goal_y),
                    taught_start_limit,
                    taught_goal_limit,
                    blocked,
                    clearance,
                    origin,
                    resolution,
                    taught_required_clearance,
                    progress_check,
                    overlap_cost,
                )
                if traversed_guidance is not None
                else None
            )
            certified_raster_route = certify_taught_route(raster_route)
            graph_length = (
                _polyline_length(certified_taught_route[0])
                if certified_taught_route is not None
                else math.inf
            )
            raster_length = (
                _polyline_length(certified_raster_route[0])
                if certified_raster_route is not None
                else math.inf
            )
            if raster_length + 1.0e-9 < graph_length:
                certified_taught_route = certified_raster_route
                traversed_graph_route_used = False
                traversed_raster_route_used = True
        if certified_taught_route is not None:
            (
                simplified_route,
                route_start_snap_m,
                route_goal_snap_m,
            ) = certified_taught_route
            committed_traversed_route_selected = bool(committed_path)
            traversed_direct_start_safe_xy = simplified_route[0]
            # Keep the first taught safe point here.  The common endpoint
            # logic below can then collapse a short exact-pose connector
            # into the following segment when that full chord is safe.
            # Prepending now would hide that opportunity and manufacture
            # a centimetre-scale heading-change corner at every replan.
            traversed_direct_route = simplified_route
            traversed_guidance_selected = bool(prefer_traversed_route)
            selected = (
                taught_margin_trial_index,
                taught_effective_margin,
                taught_traversable,
                _cell(traversed_direct_start_safe_xy, origin, resolution),
                _cell(traversed_direct_route[-1], origin, resolution),
                route_start_snap_m,
                route_goal_snap_m,
            )
    if selected is None:
        traversed_centerline_cells: set[tuple[int, int]] = set()
        for traversed_path in traversed_paths:
            for point in traversed_path:
                cell = _cell(point, origin, resolution)
                if _inside(traversable.shape, cell):
                    traversed_centerline_cells.add(cell)
        traversed_centerline_traversable_cells = sum(
            bool(traversable[cell]) for cell in traversed_centerline_cells
        )
        traversed_centerline_reachable_cells = (
            sum(bool(last_reachable[cell]) for cell in traversed_centerline_cells)
            if last_reachable is not None
            else 0
        )
        (
            traversed_segment_safe_count,
            traversed_segment_unsafe_count,
            traversed_segment_minimum_clearance_m,
        ) = (
            _audit_traversed_segment_clearance(
                traversed_paths,
                blocked,
                clearance,
                origin,
                resolution,
                radius + margin_trials[-1],
            )
            if override_traversed_all_obstacles and traversed_paths
            else (0, 0, None)
        )
        return {
            "ok": False,
            "schema": NAVIGATION_PLAN_SCHEMA,
            "schema_version": 1,
            "failure_stage": "planning",
            "error": f"unreachable: {nearest_failure}",
            "goal_name": str(goal_name).strip(),
            "goal": {
                "name": str(goal_name).strip(),
                "requested_xy_m": [goal_x, goal_y],
                "planned_xy_m": None,
            },
            "start": {
                "requested_xy_m": [pose_x, pose_y],
                "planned_xy_m": None,
                "safe_grid_xy_m": None,
                "egress_certified": False,
            },
            "start_egress_certified": False,
            "start_safe_xy_m": None,
            "path_xy_m": [],
            "corner_path_xy_m": [],
            "grid_path": [],
            "map_backend": str(snapshot.get("backend") or "unknown"),
            "map_version": str(snapshot.get("map_version") or ""),
            "episode_id": str(snapshot.get("episode_id") or ""),
            "unknown_space_policy": "blocked",
            "traversed_route_evidence_used": bool(
                use_traversed_evidence
                and (
                    traversed_promoted_cells
                    or traversed_overridden_nonwall_obstacle_cells
                    or traversed_overridden_wall_obstacle_cells
                )
            ),
            "traversed_route_direct_replay": False,
            "traversed_route_graph_used": False,
            "traversed_route_raster_graph_used": False,
            "committed_traversed_route_requested": bool(committed_path),
            "committed_traversed_route_selected": False,
            "traversed_route_guidance_requested": bool(
                prefer_traversed_route
            ),
            "traversed_route_guidance_selected": False,
            "traversed_route_guidance_tube_m": float(
                TRAVERSED_GUIDANCE_TUBE_M
            ),
            "traversed_route_reference_unsupported_length_m": (
                traversed_reference_unsupported_length_m
            ),
            "traversed_route_reference_unsupported_fraction": (
                traversed_reference_unsupported_fraction
            ),
            "traversed_route_path_count": len(traversed_paths),
            "traversed_route_point_count": int(traversed_point_count),
            "traversed_route_evidence_cells": int(traversed_evidence_cells),
            "traversed_route_start_distance_m": traversed_start_distance_m,
            "traversed_route_goal_distance_m": traversed_goal_distance_m,
            "traversed_route_centerline_cell_count": len(
                traversed_centerline_cells
            ),
            "traversed_route_centerline_traversable_cells": int(
                traversed_centerline_traversable_cells
            ),
            "traversed_route_centerline_reachable_cells": int(
                traversed_centerline_reachable_cells
            ),
            "traversed_route_segment_safe_count": int(
                traversed_segment_safe_count
            ),
            "traversed_route_segment_unsafe_count": int(
                traversed_segment_unsafe_count
            ),
            "traversed_route_segment_minimum_clearance_m": (
                traversed_segment_minimum_clearance_m
            ),
            "traversed_route_promoted_cells": int(traversed_promoted_cells),
            "traversed_route_segment_count": int(traversed_segment_count),
            "traversed_route_conflicting_obstacle_cells": int(
                traversed_conflicting_obstacle_cells
            ),
            "traversed_route_conflicting_wall_cells": int(
                traversed_conflicting_wall_cells
            ),
            "traversed_route_overridden_nonwall_obstacle_cells": int(
                traversed_overridden_nonwall_obstacle_cells
            ),
            "traversed_route_overridden_wall_obstacle_cells": int(
                traversed_overridden_wall_obstacle_cells
            ),
            "source_free_cells": source_free_cells,
            "source_obstacle_cells": source_obstacle_cells,
        }

    (
        margin_trial_index,
        effective_margin,
        traversable,
        start,
        goal,
        start_snap_m,
        goal_snap_m,
    ) = selected
    if start_snap_m <= PLAN_ENDPOINT_TOLERANCE_M:
        start_snap_m = 0.0
    minimum_clearance = radius + effective_margin
    start_egress_clearance = radius + min(
        effective_margin, START_EGRESS_SAFETY_MARGIN_M
    )
    start_safe_xy = (
        (pose_x, pose_y)
        if start_snap_m <= PLAN_ENDPOINT_TOLERANCE_M
        else (
            traversed_direct_start_safe_xy
            if traversed_direct_start_safe_xy is not None
            else _xy(start, origin, resolution)
        )
    )
    expanded = 0
    if traversed_direct_route is not None:
        corner_xy = list(traversed_direct_route)
        grid_path = []
        for point in corner_xy:
            cell = _cell(point, origin, resolution)
            if not grid_path or grid_path[-1] != cell:
                grid_path.append(cell)
        corners = list(grid_path)
    else:
        forbidden_edges: set[tuple[int, int]] = set()
        edge_clearances: list[float] = []
        for _ in range(MAX_EDGE_REPLAN_PASSES):
            grid_path, pass_expanded = _astar(
                traversable,
                clearance,
                start,
                goal,
                resolution,
                minimum_clearance,
                weight,
                forbidden_edges,
                progress_check,
                overlap_cost,
            )
            expanded += pass_expanded
            edge_clearances = []
            unsafe_edges: list[tuple[int, int]] = []
            width = traversable.shape[1]
            for edge_start, edge_end in zip(grid_path[:-1], grid_path[1:]):
                safe_edge, certificate = _segment_clearance_certificate(
                    _xy(edge_start, origin, resolution),
                    _xy(edge_end, origin, resolution),
                    blocked,
                    clearance,
                    origin,
                    resolution,
                    minimum_clearance,
                )
                edge_clearances.append(certificate)
                if not safe_edge:
                    start_index = edge_start[0] * width + edge_start[1]
                    end_index = edge_end[0] * width + edge_end[1]
                    unsafe_edges.append(
                        (
                            min(start_index, end_index),
                            max(start_index, end_index),
                        )
                    )
            if not unsafe_edges:
                break
            forbidden_edges.update(unsafe_edges)
            if progress_check is not None:
                progress_check()
        else:
            raise NavigationPlanError(
                "planner could not certify continuous footprint clearance"
            )
        corners = _simplify(
            grid_path,
            traversable,
            blocked,
            clearance,
            resolution,
            origin,
            minimum_clearance,
            weight,
            edge_clearances,
            overlap_cost,
        )
        corner_xy = [_xy(cell, origin, resolution) for cell in corners]
    # Preserve a safe exact endpoint through a short center-to-point segment
    # inside its cell. Replacing the center directly would change the slope of
    # the whole first/last shortcut and could make that long segment enter a
    # different cell. Obstacle/image-picked marks otherwise remain reported
    # stand-off goals at a safe cell center.
    exact_start = (pose_x, pose_y)
    if math.dist(exact_start, corner_xy[0]) > 1e-9:
        collapsed_start = False
        if len(corner_xy) >= 2:
            grid_start = corner_xy[0]
            next_corner = corner_xy[1]
            start_ok, start_certificate = _segment_clearance_certificate(
                exact_start,
                grid_start,
                blocked,
                clearance,
                origin,
                resolution,
                minimum_clearance,
            )
            next_ok, next_certificate = _segment_clearance_certificate(
                grid_start,
                next_corner,
                blocked,
                clearance,
                origin,
                resolution,
                minimum_clearance,
            )
            preserved_clearance = max(
                minimum_clearance,
                min(start_certificate, next_certificate),
            )
            collapsed_start, _ = _segment_clearance_certificate(
                exact_start,
                next_corner,
                blocked,
                clearance,
                origin,
                resolution,
                preserved_clearance,
            )
            collapsed_start = bool(start_ok and next_ok and collapsed_start)
        if collapsed_start:
            corner_xy[0] = exact_start
            # A fully certified direct connector needs no grid-center egress.
            # Keeping a centimetre-long snap can invent a turning-pocket corner.
            start_safe_xy = exact_start
            start_snap_m = 0.0
        else:
            corner_xy.insert(0, exact_start)
    else:
        corner_xy[0] = exact_start
    if goal_snap_m <= 1e-9:
        exact_goal = (goal_x, goal_y)
        if math.dist(corner_xy[-1], exact_goal) > 1e-9:
            corner_xy.append(exact_goal)
        else:
            corner_xy[-1] = exact_goal
    turning_repair_count = 0
    if require_turning_pockets:
        corner_xy, turning_repair_count = _repair_turning_pockets(
            corner_xy, blocked, clearance, origin, resolution, minimum_clearance,
            protected_points=(start_safe_xy,), progress_check=progress_check,
            overlap_cost=overlap_cost,
            centerline_support=(
                traversed_guidance if traversed_guidance_selected else None
            ),
        )
        turning_ok, turning_clearances, turning_count = (
            _turning_pocket_certificate(
                corner_xy,
                blocked,
                clearance,
                origin,
                resolution,
                minimum_clearance,
            )
        )
        if not turning_ok:
            return {
                "ok": False,
                "schema": NAVIGATION_PLAN_SCHEMA,
                "schema_version": 1,
                "failure_stage": "planning",
                "error": (
                    "route has a heading-change corner without a certified "
                    "turning pocket"
                ),
                "turning_clearance_m": turning_clearances,
                "turning_count": int(turning_count),
                "turning_required_clearance_m": float(
                    minimum_clearance + TURNING_CLEARANCE_RESERVE_M
                ),
                "turning_candidate_path_xy_m": [list(point) for point in corner_xy],
                "required_clearance_m": float(minimum_clearance),
                "goal_name": str(goal_name).strip(),
                "map_backend": str(snapshot.get("backend") or "unknown"),
                "map_version": str(snapshot.get("map_version") or ""),
                "episode_id": str(snapshot.get("episode_id") or ""),
                "unknown_space_policy": "blocked",
            }
    else:
        turning_clearances = []
        turning_count = 0
    execution_max_segment = (
        min(max_segment, TRAVERSED_DIRECT_MAX_SEGMENT_M)
        if traversed_direct_route is not None
        else (
            min(max_segment, TRAVERSED_OVERRIDE_MAX_SEGMENT_M)
            if override_traversed_all_obstacles
            else max_segment
        )
    )
    execution_xy = _subdivide(corner_xy, execution_max_segment)
    start_safe_execution_index = next(
        (
            index
            for index, point in enumerate(execution_xy)
            if math.dist(point, start_safe_xy) <= 1.0e-9
        ),
        None,
    )
    if start_safe_execution_index is None:
        raise NavigationPlanError(
            "navigation polyline lost its certified start-safe waypoint"
        )
    start_egress_segment_count = (
        int(start_safe_execution_index)
        if start_snap_m > PLAN_ENDPOINT_TOLERANCE_M
        else 0
    )
    polyline_cells: list[tuple[int, int]] = []
    if traversed_direct_route is not None:
        polyline_cells = list(grid_path)
    else:
        for line_start, line_end in zip(corners[:-1], corners[1:]):
            segment_cells = _line_cells(traversable, line_start, line_end)
            if segment_cells is None:
                raise NavigationPlanError(
                    "simplified navigation path failed final collision validation"
                )
            if polyline_cells and segment_cells[0] == polyline_cells[-1]:
                segment_cells = segment_cells[1:]
            polyline_cells.extend(segment_cells)
    if not polyline_cells:
        polyline_cells = list(corners)
    continuous_clearances: list[float] = []
    continuous_lengths: list[float] = []
    # Audit the exact bounded segments consumed by the executor.  Keeping one
    # certificate per signed segment lets runtime localization corrections be
    # checked against a mathematically valid path tube without consulting the
    # map backend or silently drawing a new, uncertified connector.
    execution_segments = list(zip(execution_xy[:-1], execution_xy[1:]))
    audit_segments = execution_segments
    if not audit_segments:
        audit_segments = [(corner_xy[0], corner_xy[0])]
    for segment_index, (segment_start, segment_end) in enumerate(audit_segments):
        segment_required_clearance = (
            start_egress_clearance
            if segment_index < start_egress_segment_count
            else minimum_clearance
        )
        segment_ok, certificate = _segment_clearance_certificate(
            segment_start,
            segment_end,
            blocked,
            clearance,
            origin,
            resolution,
            segment_required_clearance,
            measure_reserve_m=EXECUTION_CERTIFICATE_RESERVE_M,
        )
        if not segment_ok:
            raise NavigationPlanError(
                "navigation polyline failed continuous footprint validation"
            )
        continuous_clearances.append(certificate)
        continuous_lengths.append(math.dist(segment_start, segment_end))
    minimum_execution_clearance = min(continuous_clearances)
    regular_clearances = continuous_clearances[start_egress_segment_count:]
    minimum_continuous_clearance = (
        min(regular_clearances)
        if regular_clearances
        else max(minimum_clearance, float(clearance[start]))
    )
    total_audit_length = sum(continuous_lengths)
    mean_continuous_clearance = (
        sum(
            clearance_value * segment_length
            for clearance_value, segment_length in zip(
                continuous_clearances, continuous_lengths
            )
        )
        / total_audit_length
        if total_audit_length > 1e-12
        else minimum_continuous_clearance
    )
    path_length = _polyline_length(execution_xy)
    direct_distance = math.hypot(goal_x - pose_x, goal_y - pose_y)

    return {
        "ok": True,
        "schema": NAVIGATION_PLAN_SCHEMA,
        "schema_version": 1,
        "goal_name": str(goal_name).strip(),
        "map_backend": str(snapshot.get("backend") or "unknown"),
        "map_version": str(snapshot.get("map_version") or ""),
        "episode_id": str(snapshot.get("episode_id") or ""),
        "session_id": str(snapshot.get("session_id") or ""),
        "start_pose": {
            "x_m": pose_x,
            "y_m": pose_y,
            "yaw_rad": pose_yaw,
            "yaw_deg": math.degrees(pose_yaw),
        },
        "marked_goal_xy_m": [goal_x, goal_y],
        "planned_goal_xy_m": list(corner_xy[-1]),
        "goal": {
            "name": str(goal_name).strip(),
            "requested_xy_m": [goal_x, goal_y],
            "planned_xy_m": list(corner_xy[-1]),
        },
        "start": {
            "requested_xy_m": [pose_x, pose_y],
            "planned_xy_m": list(corner_xy[0]),
            "safe_grid_xy_m": list(start_safe_xy),
            "egress_certified": True,
        },
        "start_egress_certified": True,
        "start_safe_xy_m": list(start_safe_xy),
        "snap": {
            "start": {
                "applied": bool(
                    start_snap_m > PLAN_ENDPOINT_TOLERANCE_M
                ),
                "distance_m": float(start_snap_m),
                "egress_certified": True,
            },
            "goal": {
                "applied": bool(goal_snap_m > 1e-9),
                "distance_m": float(goal_snap_m),
            },
        },
        "path_xy_m": [[float(x), float(y)] for x, y in execution_xy],
        "route_sweep": route_sweep_audit(snapshot, execution_xy),
        "sweep_overlap_weight": float(SWEEP_OVERLAP_WEIGHT),
        "path_segment_clearance_m": [
            float(value)
            for value in (
                continuous_clearances if execution_segments else []
            )
        ],
        "start_egress_segment_count": int(start_egress_segment_count),
        "start_egress_required_clearance_m": float(start_egress_clearance),
        "corner_path_xy_m": [[float(x), float(y)] for x, y in corner_xy],
        "grid_path": [[int(row), int(column)] for row, column in grid_path],
        "path_length_m": float(path_length),
        "direct_distance_m": float(direct_distance),
        "minimum_clearance_m": float(minimum_continuous_clearance),
        "minimum_execution_clearance_m": float(minimum_execution_clearance),
        "mean_clearance_m": float(mean_continuous_clearance),
        "required_clearance_m": float(minimum_clearance),
        "robot_radius_m": float(radius),
        "requested_safety_margin_m": float(requested_margin),
        "effective_safety_margin_m": float(effective_margin),
        "safety_margin_trials_m": [float(value) for value in margin_trials],
        "safety_margin_trial_index": int(margin_trial_index),
        "safety_margin_relaxed": bool(effective_margin + 1e-9 < requested_margin),
        "clearance_weight": float(weight),
        "arrival_tolerance_m": float(tolerance),
        "goal_standoff_m": float(goal_snap_m),
        "start_snap_m": float(start_snap_m),
        "goal_duplicate_count": int(duplicate_count),
        "source_resolution_m": float(source_resolution),
        "planning_resolution_m": float(resolution),
        "downsample_factor": int(downsample),
        "expanded_cells": int(expanded),
        "unknown_space_policy": "blocked",
        "free_space_evidence": (
            "depth_and_certified_traversed_polyline"
            if traversed_direct_route is not None
            else (
                "depth_and_traversed_swept_space_override"
                if traversed_overridden_wall_obstacle_cells
                else (
                    "depth_and_traversed_base_footprint_with_nonwall_override"
                    if traversed_overridden_nonwall_obstacle_cells
                    else (
                        "depth_and_traversed_base_footprint"
                        if traversed_promoted_cells
                        else "depth"
                    )
                )
            )
        ),
        "traversed_route_evidence_used": bool(
            traversed_promoted_cells
            or traversed_overridden_nonwall_obstacle_cells
            or traversed_overridden_wall_obstacle_cells
        ),
        "traversed_route_direct_replay": bool(
            traversed_direct_route is not None
        ),
        "traversed_route_graph_used": bool(traversed_graph_route_used),
        "traversed_route_raster_graph_used": bool(
            traversed_raster_route_used
        ),
        "committed_traversed_route_requested": bool(committed_path),
        "committed_traversed_route_selected": bool(
            committed_traversed_route_selected
        ),
        "traversed_route_guidance_requested": bool(prefer_traversed_route),
        "traversed_route_guidance_selected": bool(
            traversed_guidance_selected
        ),
        "traversed_route_guidance_tube_m": float(TRAVERSED_GUIDANCE_TUBE_M),
        "traversed_route_reference_unsupported_length_m": (
            traversed_reference_unsupported_length_m
        ),
        "traversed_route_reference_unsupported_fraction": (
            traversed_reference_unsupported_fraction
        ),
        "traversed_route_planned_goal_distance_m": (
            _distance_to_traversed_paths(corner_xy[-1], traversed_paths)
        ),
        "traversed_route_path_count": len(traversed_paths),
        "traversed_route_point_count": int(traversed_point_count),
        "traversed_route_evidence_cells": int(traversed_evidence_cells),
        "traversed_route_start_distance_m": traversed_start_distance_m,
        "traversed_route_goal_distance_m": traversed_goal_distance_m,
        "traversed_route_promoted_cells": int(traversed_promoted_cells),
        "traversed_route_segment_count": int(traversed_segment_count),
        "traversed_route_conflicting_obstacle_cells": int(
            traversed_conflicting_obstacle_cells
        ),
        "traversed_route_conflicting_wall_cells": int(
            traversed_conflicting_wall_cells
        ),
        "traversed_route_overridden_nonwall_obstacle_cells": int(
            traversed_overridden_nonwall_obstacle_cells
        ),
        "traversed_route_overridden_wall_obstacle_cells": int(
            traversed_overridden_wall_obstacle_cells
        ),
        "source_free_cells": source_free_cells,
        "source_obstacle_cells": source_obstacle_cells,
        "obstacle_policy": "occupied_and_unobserved_inflated_by_static_footprint",
        "planner": (
            "certified_traversed_polyline"
            if traversed_direct_route is not None
            else "8_connected_a_star_with_exponential_clearance_cost"
        ),
        "polyline_validation": (
            "continuous_clearance_certified_traversed_polyline"
            if traversed_direct_route is not None
            else "inflated_grid_dense_line_of_sight_no_corner_cutting"
        ),
        "clearance_metric": "certified_continuous_segment_lower_bound",
        "max_execution_segment_m": float(execution_max_segment),
        "turning_clearance_m": [float(value) for value in turning_clearances],
        "turning_count": int(turning_count),
        "turning_repair_count": int(turning_repair_count),
        "turning_clearance_reserve_m": float(TURNING_CLEARANCE_RESERVE_M),
    }


def plan_clearance_path(
    snapshot: Mapping[str, Any],
    goal_name: str,
    *,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    safety_margin_fallbacks_m: Sequence[float] = (),
    clearance_weight: float = DEFAULT_CLEARANCE_WEIGHT,
    arrival_tolerance_m: float = DEFAULT_ARRIVAL_TOLERANCE_M,
    max_segment_m: float = DEFAULT_MAX_SEGMENT_M,
    optimize_goal_standoff: bool = False,
    progress_check: Callable[[], None] | None = None,
    require_turning_pockets: bool = False,
    require_traversed_route: bool = False,
    committed_traversed_path_xy_m: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Plan coarsely first, then preserve narrow known-free passages.

    Ten-centimetre cells keep ordinary searches inexpensive.  Conservative
    coarsening can erase a real residential passage when any source cell in a
    coarse block is unknown or occupied, so a failed coarse search is retried
    at the backend's native resolution.  Both attempts keep unknown space
    blocked and use the same continuous footprint-clearance certification.

    ``require_traversed_route`` is an invocation-level continuation contract:
    once an executor has selected a certified taught route, a replan may not
    silently switch to unrelated free space exposed by an incomplete map.
    """

    if not isinstance(require_traversed_route, bool):
        raise NavigationPlanError("require_traversed_route must be a boolean")
    committed_path = [
        _pair(point, "committed traversed route point")
        for point in committed_traversed_path_xy_m
    ]
    if committed_path and not require_traversed_route:
        raise NavigationPlanError(
            "a committed traversed route requires require_traversed_route=True"
        )
    if committed_path and len(committed_path) < 2:
        raise NavigationPlanError(
            "committed traversed route must contain at least two points"
        )

    common = {
        "robot_radius_m": robot_radius_m,
        "safety_margin_m": safety_margin_m,
        "safety_margin_fallbacks_m": safety_margin_fallbacks_m,
        "clearance_weight": clearance_weight,
        "arrival_tolerance_m": arrival_tolerance_m,
        "max_segment_m": max_segment_m,
        "optimize_goal_standoff": optimize_goal_standoff,
        "progress_check": progress_check,
        "require_turning_pockets": require_turning_pockets,
    }
    coarse = _plan_clearance_path_once(
        snapshot,
        goal_name,
        planning_resolution_m=PLANNING_RESOLUTION_M,
        **common,
    )
    source_resolution = _finite(
        snapshot.get("resolution_m", snapshot.get("resolution")),
        "resolution",
    )
    resolutions = [
        float(max(source_resolution, PLANNING_RESOLUTION_M))
    ]
    selected = coarse
    selected_index = 0
    evidence_trials = ["depth"]
    selected_evidence_index = 0

    def source_resolution_trial_index() -> int:
        for index, value in enumerate(resolutions):
            if abs(value - source_resolution) <= 1.0e-9:
                return index
        resolutions.append(float(source_resolution))
        return len(resolutions) - 1

    if not bool(coarse.get("ok")) and source_resolution + 1.0e-9 < resolutions[0]:
        selected = _plan_clearance_path_once(
            snapshot,
            goal_name,
            planning_resolution_m=source_resolution,
            **common,
        )
        selected_index = source_resolution_trial_index()
    if not bool(selected.get("ok")) and _traversed_paths(snapshot):
        selected = _plan_clearance_path_once(
            snapshot,
            goal_name,
            planning_resolution_m=source_resolution,
            use_traversed_evidence=True,
            **common,
        )
        evidence_trials.append("traversed_base_footprint")
        selected_evidence_index = 1
        selected_index = source_resolution_trial_index()
    if not bool(selected.get("ok")) and _traversed_paths(snapshot):
        selected = _plan_clearance_path_once(
            snapshot,
            goal_name,
            planning_resolution_m=source_resolution,
            use_traversed_evidence=True,
            override_traversed_nonwall_obstacles=True,
            **common,
        )
        evidence_trials.append("traversed_nonwall_obstacle_override")
        selected_evidence_index = 2
        selected_index = source_resolution_trial_index()
    if not bool(selected.get("ok")) and _traversed_paths(snapshot):
        selected = _plan_clearance_path_once(
            snapshot,
            goal_name,
            planning_resolution_m=source_resolution,
            use_traversed_evidence=True,
            override_traversed_all_obstacles=True,
            **common,
        )
        evidence_trials.append("traversed_swept_space_override")
        selected_evidence_index = 3
        selected_index = source_resolution_trial_index()
    preference_evaluated = False
    preference_selected = False
    preference_reason = "not_evaluated"
    preference_failure_stage: str | None = None
    preference_failure_error: str | None = None
    preference_candidate_path_length_m: float | None = None
    committed_branch_reselection_evaluated = False
    committed_branch_reselection_selected = False
    ordinary_unsupported_length_m: float | None = None
    ordinary_unsupported_fraction: float | None = None
    ordinary_path_length_m = (
        float(selected.get("path_length_m", 0.0))
        if bool(selected.get("ok"))
        else None
    )
    traversed_paths_available = bool(_traversed_paths(snapshot))
    ordinary_preference_eligible = bool(
        bool(selected.get("ok"))
        and selected_evidence_index == 0
        and traversed_paths_available
        and selected.get("traversed_route_start_distance_m") is not None
        and selected.get("traversed_route_goal_distance_m") is not None
        and float(selected["traversed_route_start_distance_m"])
        <= min(arrival_tolerance_m, TRAVERSED_PREFERENCE_MAX_ENDPOINT_M)
        and float(selected["traversed_route_goal_distance_m"])
        <= max(0.0, arrival_tolerance_m - GOAL_TRACKING_RESERVE_M)
    )
    if require_traversed_route or ordinary_preference_eligible:
        preference_evaluated = True
        preferred = _plan_clearance_path_once(
            snapshot,
            goal_name,
            planning_resolution_m=source_resolution,
            use_traversed_evidence=True,
            override_traversed_all_obstacles=True,
            prefer_traversed_route=True,
            traversed_reference_path_xy_m=(
                committed_path or selected.get("path_xy_m", ())
            ),
            committed_traversed_path_xy_m=committed_path,
            **common,
        )
        # Keep the established evidence-stage label for persisted-plan and UI
        # compatibility. The guidance-specific audit fields distinguish this
        # proactive candidate from the legacy unreachable-route fallback.
        evidence_trials.append("traversed_swept_space_override")
        ordinary_unsupported_length_m = preferred.get(
            "traversed_route_reference_unsupported_length_m"
        )
        ordinary_unsupported_fraction = preferred.get(
            "traversed_route_reference_unsupported_fraction"
        )
        preferred_ok = bool(
            bool(preferred.get("ok"))
            and bool(preferred.get("traversed_route_guidance_selected"))
        )
        if committed_path and not preferred_ok:
            # A map correction or fresh obstacle can invalidate the remaining
            # geometry of one concrete committed branch.  The invocation must
            # remain on previously traversed space, but it need not fail when
            # another currently certified history branch exists.  Re-run the
            # history-only search without binding it to the stale polyline;
            # the executor will immediately commit the returned replacement.
            committed_branch_reselection_evaluated = True
            rebound = _plan_clearance_path_once(
                snapshot,
                goal_name,
                planning_resolution_m=source_resolution,
                use_traversed_evidence=True,
                override_traversed_all_obstacles=True,
                prefer_traversed_route=True,
                traversed_reference_path_xy_m=committed_path,
                committed_traversed_path_xy_m=(),
                **common,
            )
            rebound_ok = bool(
                bool(rebound.get("ok"))
                and bool(rebound.get("traversed_route_guidance_selected"))
            )
            if rebound_ok:
                preferred = rebound
                preferred_ok = True
                committed_branch_reselection_selected = True
        if bool(preferred.get("ok")):
            preference_candidate_path_length_m = float(
                preferred.get("path_length_m", 0.0)
            )
        if not preferred_ok:
            preference_failure_stage = str(
                preferred.get("failure_stage") or "planning"
            )
            preference_failure_error = str(
                preferred.get("error")
                or "history route failed continuous footprint certification"
            )
        materially_unsupported = bool(
            ordinary_unsupported_length_m is not None
            and ordinary_unsupported_fraction is not None
            and float(ordinary_unsupported_length_m)
            >= TRAVERSED_GUIDANCE_MIN_UNSUPPORTED_M
            and float(ordinary_unsupported_fraction)
            >= TRAVERSED_GUIDANCE_MIN_UNSUPPORTED_RATIO
        )
        history_is_shorter = bool(
            preferred_ok
            and float(preferred.get("path_length_m", math.inf))
            + TRAVERSED_PREFERENCE_MIN_SAVING_M
            <= float(selected.get("path_length_m", 0.0))
        )
        if preferred_ok and (
            require_traversed_route
            or materially_unsupported
            or history_is_shorter
        ):
            selected = preferred
            selected_evidence_index = len(evidence_trials) - 1
            selected_index = source_resolution_trial_index()
            preference_selected = True
            preference_reason = (
                "history_route_commitment"
                if require_traversed_route
                else "ordinary_route_leaves_traversed_corridor"
                if materially_unsupported
                else "history_route_is_shorter"
            )
        elif preferred_ok:
            preference_reason = "ordinary_route_remains_supported"
        elif require_traversed_route:
            selected = dict(preferred)
            selected["ok"] = False
            selected["failure_stage"] = "planning"
            selected["error"] = (
                "committed traversed route is not currently footprint-safe"
            )
            preference_reason = (
                "history_route_commitment_temporarily_blocked"
            )
        else:
            preference_reason = "history_route_failed_safety_validation"
    if require_traversed_route and not traversed_paths_available:
        selected = dict(selected)
        selected["ok"] = False
        selected["failure_stage"] = "planning"
        selected["error"] = "committed traversed route evidence is unavailable"
        preference_failure_stage = "planning"
        preference_failure_error = selected["error"]
        preference_reason = "history_route_commitment_evidence_unavailable"
    selected = dict(selected)
    selected["planning_resolution_trials_m"] = resolutions
    selected["planning_resolution_trial_index"] = int(selected_index)
    selected["planning_resolution_fallback_used"] = bool(selected_index)
    selected["planning_evidence_trials"] = evidence_trials
    selected["planning_evidence_trial_index"] = int(selected_evidence_index)
    selected["planning_evidence_fallback_used"] = bool(
        selected_evidence_index
    )
    selected["traversed_route_preference_evaluated"] = bool(
        preference_evaluated
    )
    selected["traversed_route_preference_selected"] = bool(
        preference_selected
    )
    selected["traversed_route_preference_reason"] = preference_reason
    selected["traversed_route_preference_failure_stage"] = (
        preference_failure_stage
    )
    selected["traversed_route_preference_failure_error"] = (
        preference_failure_error
    )
    selected["traversed_route_preference_candidate_path_length_m"] = (
        preference_candidate_path_length_m
    )
    selected["committed_traversed_branch_reselection_evaluated"] = bool(
        committed_branch_reselection_evaluated
    )
    selected["committed_traversed_branch_reselection_selected"] = bool(
        committed_branch_reselection_selected
    )
    # Preserve the caller's audit contract after a successful branch rebind;
    # the inner unbound search truthfully leaves ``..._selected`` false.
    selected["committed_traversed_route_requested"] = bool(committed_path)
    selected["ordinary_route_unsupported_by_history_m"] = (
        ordinary_unsupported_length_m
    )
    selected["ordinary_route_unsupported_by_history_fraction"] = (
        ordinary_unsupported_fraction
    )
    selected["ordinary_path_length_m"] = ordinary_path_length_m
    selected["traversed_route_commitment_requested"] = bool(
        require_traversed_route
    )
    return selected


def convex_point_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Euclidean distances to a filled, counterclockwise convex polygon."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    edges = np.roll(polygon, -1, axis=0) - polygon
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 2048):
        delta = points[start:start + 2048, None, :] - polygon
        cross = edges[:, 0] * delta[:, :, 1] - edges[:, 1] * delta[:, :, 0]
        fraction = np.clip(np.sum(delta * edges, axis=2) / np.sum(edges * edges, axis=1), 0, 1)
        distance = np.linalg.norm(delta - fraction[:, :, None] * edges, axis=2).min(axis=1)
        distance[np.all(cross >= -1e-12, axis=1)] = 0.0
        result[start:start + len(distance)] = distance
    return result


def convex_point_distance_lower_bounds(
    points: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """Cheap conservative lower bounds on distance to a convex polygon.

    For a counterclockwise convex polygon, every unit outward edge normal is a
    separating axis.  The maximum positive half-space violation cannot exceed
    Euclidean distance to the polygon.  Using this bound to construct a search
    mask can therefore reject extra near-corner centres, but it can never admit
    a colliding centre.  Selected motion primitives are still checked with
    ``convex_point_distances`` through the continuous sweep functions.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    normals, support = _convex_polygon_halfspaces(polygon)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 32768):
        violations = points[start : start + 32768] @ normals.T - support
        result[start : start + len(violations)] = np.maximum(
            0.0, np.max(violations, axis=1)
        )
    return result


def _convex_polygon_halfspaces(
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit outward normals and their support values."""

    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("convex polygon has a degenerate edge")
    signed_area = float(np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    ))
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    if signed_area < 0.0:
        normals *= -1.0
    return normals, np.sum(polygon * normals, axis=1)


def _prepare_cuda_depth_clearance(
    relative_points: np.ndarray,
    pair_point_radii: np.ndarray,
    owners: np.ndarray,
    owner_count: int,
    polygon: np.ndarray,
) -> dict[str, Any] | None:
    """Cache repeated fixed-yaw clearance inputs on CUDA when available."""

    if len(relative_points) < 250_000:
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
        normals, support = _convex_polygon_halfspaces(polygon)
        return {
            "torch": torch,
            "device": device,
            "points": torch.as_tensor(
                np.asarray(relative_points, dtype=np.float32), device=device
            ),
            "radii": torch.as_tensor(
                np.asarray(pair_point_radii, dtype=np.float32), device=device
            ),
            "owners": torch.as_tensor(
                np.asarray(owners, dtype=np.int64), device=device
            ),
            "normals": torch.as_tensor(
                np.asarray(normals, dtype=np.float32), device=device
            ),
            "support": torch.as_tensor(
                np.asarray(support, dtype=np.float32), device=device
            ),
            "owner_count": int(owner_count),
        }
    except (ImportError, RuntimeError, ValueError):
        return None


def _fixed_yaw_depth_clearance(
    relative_points: np.ndarray,
    pair_point_radii: np.ndarray,
    owners: np.ndarray,
    owner_count: int,
    polygon: np.ndarray,
    rotation: np.ndarray,
    cuda_backend: Mapping[str, Any] | None,
) -> np.ndarray:
    """Conservative per-centre clearance for fixed-yaw search layers."""

    if cuda_backend is not None:
        try:
            torch = cuda_backend["torch"]
            device = cuda_backend["device"]
            # Row vectors are transformed as p @ rotation before testing the
            # base-frame half spaces.  Keep the large pair table resident and
            # transfer only the per-heading 2xE matrix and final owner minima.
            axes = torch.as_tensor(
                np.asarray(rotation, dtype=np.float32), device=device
            ) @ cuda_backend["normals"].T
            result = torch.full(
                (int(cuda_backend["owner_count"]),),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            points = cuda_backend["points"]
            radii = cuda_backend["radii"]
            owner_ids = cuda_backend["owners"]
            support = cuda_backend["support"]
            for start in range(0, len(points), 262_144):
                stop = min(len(points), start + 262_144)
                violation = points[start:stop] @ axes - support
                # Float32 roundoff must not turn this lower bound into an upper
                # bound.  A 10um guard is far below the 20mm planning margin.
                lower = torch.clamp(
                    torch.max(violation, dim=1).values - 1.0e-5,
                    min=0.0,
                ) - radii[start:stop]
                result.scatter_reduce_(
                    0,
                    owner_ids[start:stop],
                    lower,
                    reduce="amin",
                    include_self=True,
                )
            return result.cpu().numpy().astype(np.float64, copy=False)
        except (KeyError, RuntimeError, TypeError, ValueError):
            # Resource pressure may make CUDA temporarily unavailable.  The
            # deterministic CPU lower-bound path has the same safety direction.
            pass
    local_clearance = np.full(owner_count, np.inf, dtype=np.float64)
    np.minimum.at(
        local_clearance,
        owners,
        convex_point_distance_lower_bounds(
            relative_points @ rotation, polygon
        )
        - pair_point_radii,
    )
    return local_clearance


def convex_sweep_distances(
    points: np.ndarray, polygon: np.ndarray, start_xy: Sequence[float],
    end_xy: Sequence[float], start_yaw: float, end_yaw: float,
) -> np.ndarray:
    """Conservative continuous translation/rotation sweep, not endpoint-only.

    Each angular interval is enclosed by its endpoint hull plus the maximum
    vertex arc sagitta. Translation is exact when yaw is unchanged.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    delta_yaw = (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi
    steps = max(1, int(math.ceil(abs(delta_yaw) / math.radians(5))))
    radius = float(np.linalg.norm(polygon, axis=1).max())
    sagitta = radius * (1 - math.cos(delta_yaw / steps / 2))
    start_xy, end_xy = np.asarray(start_xy), np.asarray(end_xy)
    distances = np.full(len(points), np.inf)
    previous = None
    for fraction in np.linspace(0, 1, steps + 1):
        yaw = start_yaw + delta_yaw * fraction
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        current = polygon @ rotation.T + start_xy + fraction * (end_xy - start_xy)
        if previous is not None:
            vertices = np.concatenate((previous, current))
            hull = vertices[ConvexHull(vertices).vertices]
            distances = np.minimum(distances, convex_point_distances(points, hull) - sagitta)
        previous = current
    return np.maximum(distances, 0.0)


def _convex_polygon_rectangle_distances(
    polygon: np.ndarray,
    x_min: np.ndarray,
    y_min: np.ndarray,
    x_max: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Return exact distances from a filled convex polygon to rectangles."""

    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    shape = np.broadcast_shapes(
        np.shape(x_min), np.shape(y_min), np.shape(x_max), np.shape(y_max)
    )
    x_min = np.broadcast_to(np.asarray(x_min, dtype=np.float64), shape).reshape(-1)
    y_min = np.broadcast_to(np.asarray(y_min, dtype=np.float64), shape).reshape(-1)
    x_max = np.broadcast_to(np.asarray(x_max, dtype=np.float64), shape).reshape(-1)
    y_max = np.broadcast_to(np.asarray(y_max, dtype=np.float64), shape).reshape(-1)
    if not len(x_min):
        return np.empty(shape, dtype=np.float64)
    signed_area = float(np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    ))
    if signed_area < 0.0:
        polygon = polygon[::-1]
    distances = np.full(len(x_min), np.inf, dtype=np.float64)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        distances = np.minimum(
            distances,
            _segment_to_rectangles_distance(
                tuple(start), tuple(end), x_min, y_min, x_max, y_max
            ),
        )
    corners = np.stack(
        (
            np.column_stack((x_min, y_min)),
            np.column_stack((x_min, y_max)),
            np.column_stack((x_max, y_min)),
            np.column_stack((x_max, y_max)),
        ),
        axis=1,
    )
    corner_distances = convex_point_distances(
        corners.reshape(-1, 2), polygon
    ).reshape(-1, 4)
    distances = np.minimum(distances, corner_distances.min(axis=1))
    return distances.reshape(shape)


def _oriented_static_invalid_mask(
    blocked: np.ndarray,
    polygon: np.ndarray,
    yaw: float,
    origin: tuple[float, float],
    resolution: float,
    margin_m: float,
) -> tuple[np.ndarray, int]:
    """Rasterize the exact fixed-yaw polygon configuration-space obstacle."""

    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    world_polygon = np.asarray(polygon, dtype=np.float64) @ rotation.T
    half_cell = 0.5 * resolution
    column_radius = int(math.ceil(
        (float(np.max(np.abs(world_polygon[:, 0]))) + margin_m + half_cell)
        / resolution
    )) + 1
    row_radius = int(math.ceil(
        (float(np.max(np.abs(world_polygon[:, 1]))) + margin_m + half_cell)
        / resolution
    )) + 1
    drows, dcolumns = np.mgrid[
        -row_radius : row_radius + 1,
        -column_radius : column_radius + 1,
    ]
    # The morphology offset is robot-centre minus obstacle-centre. Express the
    # obstacle squares in the robot frame before testing polygon clearance.
    obstacle_x = -dcolumns.astype(np.float64) * resolution
    obstacle_y = -drows.astype(np.float64) * resolution
    distances = _convex_polygon_rectangle_distances(
        world_polygon,
        obstacle_x - half_cell,
        obstacle_y - half_cell,
        obstacle_x + half_cell,
        obstacle_y + half_cell,
    )
    kernel = distances + 1.0e-12 < margin_m
    invalid = fftconvolve(
        np.asarray(blocked, dtype=np.float32),
        np.asarray(kernel, dtype=np.float32),
        mode="same",
    ) > 0.5

    height, width = blocked.shape
    x_centers = origin[0] + (np.arange(width, dtype=np.float64) + 0.5) * resolution
    y_centers = origin[1] + (np.arange(height, dtype=np.float64) + 0.5) * resolution
    map_x_max = origin[0] + width * resolution
    map_y_max = origin[1] + height * resolution
    valid_columns = (
        (x_centers + float(world_polygon[:, 0].min()) >= origin[0] + margin_m - 1.0e-12)
        & (x_centers + float(world_polygon[:, 0].max()) <= map_x_max - margin_m + 1.0e-12)
    )
    valid_rows = (
        (y_centers + float(world_polygon[:, 1].min()) >= origin[1] + margin_m - 1.0e-12)
        & (y_centers + float(world_polygon[:, 1].max()) <= map_y_max - margin_m + 1.0e-12)
    )
    invalid |= ~(valid_rows[:, None] & valid_columns[None, :])
    return invalid, int(np.count_nonzero(kernel))


def _oriented_static_sweep_clearance(
    polygon: np.ndarray,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    start_yaw: float,
    end_yaw: float,
    blocked: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    *,
    measure_limit_m: float,
) -> float:
    """Conservatively measure a continuous oriented sweep against grid cells."""

    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(2)
    end_xy = np.asarray(end_xy, dtype=np.float64).reshape(2)
    body_radius = float(np.linalg.norm(polygon, axis=1).max())
    search = max(0.0, float(measure_limit_m))
    height, width = blocked.shape
    map_x_max = origin[0] + width * resolution
    map_y_max = origin[1] + height * resolution
    lower = np.minimum(start_xy, end_xy) - body_radius - search - resolution
    upper = np.maximum(start_xy, end_xy) + body_radius + search + resolution
    column0 = max(0, int(math.floor((lower[0] - origin[0]) / resolution)) - 1)
    column1 = min(width - 1, int(math.floor((upper[0] - origin[0]) / resolution)) + 1)
    row0 = max(0, int(math.floor((lower[1] - origin[1]) / resolution)) - 1)
    row1 = min(height - 1, int(math.floor((upper[1] - origin[1]) / resolution)) + 1)
    local_rows, local_columns = np.nonzero(
        blocked[row0 : row1 + 1, column0 : column1 + 1]
    )
    rect_x_min = origin[0] + (local_columns + column0).astype(np.float64) * resolution
    rect_y_min = origin[1] + (local_rows + row0).astype(np.float64) * resolution

    delta_yaw = (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi
    steps = max(1, int(math.ceil(abs(delta_yaw) / math.radians(5.0))))
    sagitta = body_radius * (1.0 - math.cos(delta_yaw / steps / 2.0))
    clearance = search
    previous: np.ndarray | None = None
    for fraction in np.linspace(0.0, 1.0, steps + 1):
        yaw = start_yaw + delta_yaw * fraction
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        current = (
            polygon @ rotation.T
            + start_xy
            + fraction * (end_xy - start_xy)
        )
        if previous is None:
            previous = current
            continue
        if np.max(np.abs(current - previous)) <= 1.0e-15:
            hull = current
        else:
            vertices = np.concatenate((previous, current))
            hull = vertices[ConvexHull(vertices).vertices]
        boundary_gap = min(
            float(hull[:, 0].min()) - origin[0],
            map_x_max - float(hull[:, 0].max()),
            float(hull[:, 1].min()) - origin[1],
            map_y_max - float(hull[:, 1].max()),
        ) - sagitta
        clearance = min(clearance, boundary_gap)
        if len(local_rows):
            distances = _convex_polygon_rectangle_distances(
                hull,
                rect_x_min,
                rect_y_min,
                rect_x_min + resolution,
                rect_y_min + resolution,
            )
            clearance = min(clearance, float(distances.min()) - sagitta)
        previous = current
    return max(0.0, float(clearance))


def certify_fixed_yaw_local_translation(
    snapshot: Mapping[str, Any],
    reference_plan: Mapping[str, Any],
    polygon: np.ndarray,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    yaw: float,
    depth_points_xy_m: np.ndarray,
    *,
    depth_margin_m: float = DEPTH_POINT_MARGIN_M,
    static_margin_m: float = START_EGRESS_SAFETY_MARGIN_M,
) -> dict[str, Any] | None:
    """Certify a fixed-yaw translation that may separate from near depth.

    A generic clearance ball cannot prove an escape when the base starts near
    one obstacle: it subtracts the requested displacement in every direction,
    including directions that move away from that obstacle.  This certificate
    instead checks the complete convex footprint sweep against both the static
    navigation map and every current depth return.  A sub-margin start is
    allowed only when every affected return is non-approaching along the whole
    primitive, matching the planner's monotone depth-egress contract.
    """

    try:
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        end = np.asarray(end_xy, dtype=np.float64).reshape(2)
        points = np.asarray(depth_points_xy_m, dtype=np.float64).reshape(-1, 2)
        yaw = float(yaw)
        depth_margin_m = float(depth_margin_m)
        static_margin_m = float(static_margin_m)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        len(polygon) < 3
        or not len(points)
        or not np.all(np.isfinite(polygon))
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
        or not np.all(np.isfinite(points))
        or not math.isfinite(yaw)
        or not math.isfinite(depth_margin_m)
        or not math.isfinite(static_margin_m)
        or depth_margin_m < DEPTH_EGRESS_GAP_M
        or static_margin_m < 0.0
        or float(np.linalg.norm(end - start)) <= 1.0e-9
    ):
        return None

    try:
        planning_snapshot, evidence_audit = (
            _replay_reference_traversed_evidence(snapshot, reference_plan)
        )
        obstacle, free, resolution, origin = _snapshot_geometry(
            planning_snapshot
        )
    except (KeyError, NavigationPlanError, TypeError, ValueError):
        return None
    blocked = obstacle | ~free
    static_clearance_m = _oriented_static_sweep_clearance(
        polygon,
        start,
        end,
        yaw,
        yaw,
        blocked,
        origin,
        resolution,
        measure_limit_m=(
            static_margin_m + EXECUTION_CERTIFICATE_RESERVE_M
        ),
    )
    if static_clearance_m + 1.0e-9 < static_margin_m:
        return None

    start_gaps = convex_sweep_distances(
        points, polygon, start, start, yaw, yaw
    )
    sweep_gaps = convex_sweep_distances(
        points, polygon, start, end, yaw, yaw
    )
    initial_gap_m = float(start_gaps.min())
    if initial_gap_m + 1.0e-9 < DEPTH_EGRESS_GAP_M:
        return None
    initially_tight = start_gaps < depth_margin_m
    required_gaps = np.full(len(points), depth_margin_m, dtype=np.float64)
    required_gaps[initially_tight] = np.maximum(
        DEPTH_EGRESS_GAP_M,
        start_gaps[initially_tight] - 1.0e-9,
    )
    if np.any(sweep_gaps + 1.0e-9 < required_gaps):
        return None

    return {
        "schema": "official_v2_fixed_yaw_local_translation_v1",
        "start_xy_m": start.astype(float).tolist(),
        "end_xy_m": end.astype(float).tolist(),
        "fixed_yaw_rad": float(yaw),
        "translation_m": float(np.linalg.norm(end - start)),
        "static_clearance_m": float(static_clearance_m),
        "static_margin_m": float(static_margin_m),
        "initial_depth_clearance_m": initial_gap_m,
        "sweep_depth_clearance_m": float(sweep_gaps.min()),
        "depth_margin_m": float(depth_margin_m),
        "monotone_egress": bool(np.any(initially_tight)),
        "tight_point_count": int(np.count_nonzero(initially_tight)),
        "depth_point_count": int(len(points)),
        "traversed_evidence": evidence_audit,
    }


def compact_depth_disks(points: np.ndarray, bin_m: float = .01) -> tuple[np.ndarray, np.ndarray]:
    """Compress nearby XY samples into enclosing disks, without dropping evidence."""
    _, inverse, counts = np.unique(np.floor(points / bin_m), axis=0, return_inverse=True, return_counts=True)
    centers = np.zeros((len(counts), 2), dtype=np.float64)
    np.add.at(centers, inverse, points)
    centers /= counts[:, None]
    radii = np.zeros(len(counts))
    np.maximum.at(radii, inverse, np.linalg.norm(points - centers[inverse], axis=1))
    return centers, radii


def _oriented_heading_candidates(
    snapshot: Mapping[str, Any],
    map_only_plan: Mapping[str, Any],
    pose_yaw: float,
    depth_proposals: Sequence[float],
) -> tuple[list[float], int]:
    """Return yaw hypotheses and the route-relative coarse-ladder size."""

    start_xy = tuple(_pose(snapshot)[:2])
    goal_xy = tuple(map_only_plan["planned_goal_xy_m"])
    route_targets: list[float] = []
    map_targets: list[float] = []

    # A prior physical crossing is the strongest available doorway-direction
    # cue.  Use only the ends of the route relevant to this query; long loops in
    # the trail must not dominate the candidate order.
    taught = _traversed_route_between(
        _traversed_paths(snapshot), start_xy, goal_xy, max_endpoint_distance_m=.50
    )
    if taught is not None:
        route = taught[0]
        route_segments: list[tuple[float, float, float]] = []
        cumulative = 0.0
        total_length = _polyline_length(route)
        for start, end in zip(route[:-1], route[1:]):
            length = math.dist(start, end)
            if length <= .02 or length > TRAVERSED_PATH_MAX_LINK_M + 1.0e-9:
                cumulative += length
                continue
            midpoint_arc = cumulative + .5 * length
            endpoint_distance = min(midpoint_arc, max(0.0, total_length - midpoint_arc))
            route_segments.append((endpoint_distance, -length, math.atan2(
                end[1] - start[1], end[0] - start[0]
            )))
            cumulative += length
        route_segments.sort()
        for _, _, heading in route_segments[:8]:
            route_targets.append(heading)

    # The current map route supplies useful corridor tangents even when there
    # is no prior traversal reaching both endpoints.
    map_path = [tuple(point) for point in map_only_plan.get("path_xy_m", ())]
    for start, end in zip(map_path[:-1], map_path[1:]):
        if math.dist(start, end) > .02:
            heading = math.atan2(end[1] - start[1], end[0] - start[0])
            map_targets.append(heading)

    primary_targets = [*route_targets, *map_targets, *depth_proposals[:8]]

    candidates: list[float] = [pose_yaw]
    # Build one shared fifteen-degree ladder toward the taught and map-route
    # headings before considering raw-depth PCA alternatives. This reaches a
    # previously traversed doorway tangent after a handful of layers and keeps
    # unrelated depth edges from interleaving duplicate work. The ladder is
    # relative to the observed yaw, never a room or world-axis special case.
    preferred_targets = [*route_targets, *map_targets]
    if not preferred_targets:
        preferred_targets = list(depth_proposals[:8])
    preferred_deltas = [
        (target - pose_yaw + math.pi) % (2.0 * math.pi) - math.pi
        for target in preferred_targets
    ]
    max_step = max(
        (
            int(math.ceil(abs(delta) / math.radians(15.0)))
            for delta in preferred_deltas
        ),
        default=0,
    )
    for step in range(1, max_step + 1):
        travelled = step * math.radians(15.0)
        for sign in (1.0, -1.0):
            if any(
                math.copysign(1.0, delta) == sign
                and abs(delta) + 1.0e-9 >= travelled
                for delta in preferred_deltas
                if abs(delta) > 1.0e-9
            ):
                candidates.append(pose_yaw + sign * travelled)
    coarse_candidate_count = len(candidates)
    # If the coarse configuration spaces touch but a 15-degree sweep cannot
    # certify their transition, these nested refinements supply 7.5- and
    # 3.75-degree motion primitives without changing the scene model.
    for fractional_offset in (.5, .25, .75):
        for step in range(max_step):
            travelled = (step + fractional_offset) * math.radians(15.0)
            for sign in (1.0, -1.0):
                if any(
                    math.copysign(1.0, delta) == sign
                    and abs(delta) + 1.0e-9 >= travelled
                    for delta in preferred_deltas
                    if abs(delta) > 1.0e-9
                ):
                    candidates.append(pose_yaw + sign * travelled)
    candidates.extend(preferred_targets)

    # Depth-only headings remain a general fallback for a never-traversed
    # passage, after the cheaper high-confidence route evidence.
    for target in depth_proposals[:8]:
        delta = (target - pose_yaw + math.pi) % (2.0 * math.pi) - math.pi
        pieces = max(1, int(math.ceil(abs(delta) / math.radians(15.0))))
        candidates.extend(
            pose_yaw + delta * step / pieces for step in range(1, pieces + 1)
        )

    # Reversed body orientations remain useful in cul-de-sacs, but are lower
    # priority than the observed travel direction and its smooth transition.
    candidates.extend(target + math.pi for target in primary_targets)

    # Exhaustive fallback, expressed relative to the observed pose rather than
    # a room-specific world axis.  Thirty-degree coverage is tried before the
    # interleaved fifteen-degree hypotheses.
    candidates.extend(
        pose_yaw + sign * step * math.pi / 6.0
        for step in range(1, 7)
        for sign in (1, -1)
    )
    candidates.extend(
        pose_yaw + sign * (step + .5) * math.pi / 6.0
        for step in range(6)
        for sign in (1, -1)
    )

    result: list[float] = []
    coarse_result_count = 0
    for candidate_index, candidate in enumerate(candidates):
        yaw = (float(candidate) + math.pi) % (2.0 * math.pi) - math.pi
        if any(
            abs((yaw - existing + math.pi) % (2.0 * math.pi) - math.pi)
            < math.radians(2.0)
            for existing in result
        ):
            continue
        result.append(yaw)
        if candidate_index < coarse_candidate_count:
            coarse_result_count += 1
    return result, coarse_result_count


def _plan_variable_orientation_route(
    layers: Sequence[Mapping[str, Any]],
    *,
    polygon: np.ndarray,
    points: np.ndarray,
    point_radii: np.ndarray,
    blocked: np.ndarray,
    clearance: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    start_xy: np.ndarray,
    start_cell: tuple[int, int],
    start_yaw: float,
    goal_xy: tuple[float, float],
    goal_cell: tuple[int, int],
    static_margin: float,
    depth_margin: float,
    max_segment_m: float,
    clearance_weight: float,
    progress_check: Callable[[], None] | None,
    overlap_cost: np.ndarray | None = None,
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Find and continuously certify a route that changes yaw in open space."""

    if len(layers) < 2:
        if diagnostic is not None:
            diagnostic["failure"] = "insufficient_orientation_layers"
        return None
    body_radius = float(np.linalg.norm(polygon, axis=1).max())
    start_depth_distances = (
        convex_sweep_distances(
            points, polygon, start_xy, start_xy, start_yaw, start_yaw
        )
        - point_radii
    )
    start_depth_gap = float(start_depth_distances.min())
    needs_egress = start_depth_gap < depth_margin

    def static_gap(
        start: Sequence[float],
        end: Sequence[float],
        yaw0: float,
        yaw1: float,
    ) -> float:
        return _oriented_static_sweep_clearance(
            polygon,
            start,
            end,
            yaw0,
            yaw1,
            blocked,
            origin,
            resolution,
            measure_limit_m=static_margin + EXECUTION_CERTIFICATE_RESERVE_M,
        )

    def depth_gap(
        start: Sequence[float],
        end: Sequence[float],
        yaw0: float,
        yaw1: float,
    ) -> float:
        distances = (
            convex_sweep_distances(points, polygon, start, end, yaw0, yaw1)
            - point_radii
        )
        return min(depth_margin + .10, float(distances.min()))

    def translation_ok(
        start: Sequence[float],
        end: Sequence[float],
        yaw: float,
        *,
        allow_start_egress: bool = False,
    ) -> tuple[bool, float, float]:
        map_gap = static_gap(start, end, yaw, yaw)
        depth_value = depth_gap(start, end, yaw, yaw)
        if map_gap + 1.0e-9 < static_margin:
            return False, map_gap, depth_value
        if depth_value + 1.0e-9 >= depth_margin:
            return True, map_gap, depth_value
        if not allow_start_egress or math.dist(start, start_xy) > 1.0e-8:
            return False, map_gap, depth_value
        end_distances = (
            convex_sweep_distances(points, polygon, end, end, yaw, yaw)
            - point_radii
        )
        swept_distances = (
            convex_sweep_distances(points, polygon, start, end, yaw, yaw)
            - point_radii
        )
        separating = bool(
            start_depth_gap >= DEPTH_EGRESS_GAP_M
            and np.all(
                swept_distances
                >= np.minimum(start_depth_distances, depth_margin) - 1.0e-9
            )
            and np.all(end_distances >= depth_margin - 1.0e-9)
        )
        return separating, map_gap, depth_value

    height, width = blocked.shape
    rows, columns = np.nonzero(~blocked)
    centres = np.column_stack(
        (
            origin[0] + (columns.astype(np.float64) + .5) * resolution,
            origin[1] + (rows.astype(np.float64) + .5) * resolution,
        )
    )
    point_centre_gap = np.full(blocked.shape, np.inf, dtype=np.float64)
    if len(centres):
        tree = cKDTree(points)
        neighbours = tree.query_ball_point(
            centres, body_radius + depth_margin + resolution + float(point_radii.max())
        )
        counts = np.fromiter((len(items) for items in neighbours), dtype=np.int64)
        if int(counts.sum()):
            owners = np.repeat(np.arange(len(neighbours)), counts)
            point_ids = np.fromiter(
                (item for items in neighbours for item in items),
                dtype=np.int64,
                count=int(counts.sum()),
            )
            gaps = (
                np.linalg.norm(points[point_ids] - centres[owners], axis=1)
                - point_radii[point_ids]
            )
            local = np.full(len(centres), np.inf, dtype=np.float64)
            np.minimum.at(local, owners, gaps)
            point_centre_gap[rows, columns] = local
    # This full-circumcircle gate is used only for in-place yaw changes.  It
    # does not inflate translations, which retain the tighter exact polygon.
    turnable = (
        (~blocked)
        & (clearance + 1.0e-9 >= body_radius + static_margin)
        & (point_centre_gap + 1.0e-9 >= body_radius + depth_margin)
    )
    if diagnostic is not None:
        diagnostic.update(
            orientation_layer_count=len(layers),
            turnable_cell_count=int(np.count_nonzero(turnable)),
        )

    start_states: list[tuple[int, tuple[int, int], float]] = []
    start_radius = max(2, int(math.ceil(.25 / resolution))) if needs_egress else 2
    for layer_index, layer in enumerate(layers):
        yaw = float(layer["yaw"])
        rotation_map_gap = static_gap(start_xy, start_xy, start_yaw, yaw)
        rotation_depth_gap = depth_gap(start_xy, start_xy, start_yaw, yaw)
        same_yaw = abs(
            (yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        ) <= 1.0e-8
        rotation_required = start_depth_gap - 1.0e-9 if needs_egress else depth_margin
        if (
            rotation_map_gap + 1.0e-9 < static_margin
            or rotation_depth_gap + 1.0e-9 < rotation_required
            or (needs_egress and not same_yaw)
        ):
            continue
        allowed = np.asarray(layer["allowed"])
        candidates: list[tuple[float, tuple[int, int]]] = []
        for row in range(max(0, start_cell[0] - start_radius), min(height, start_cell[0] + start_radius + 1)):
            for column in range(max(0, start_cell[1] - start_radius), min(width, start_cell[1] + start_radius + 1)):
                cell = (row, column)
                if not allowed[cell]:
                    continue
                xy = _xy(cell, origin, resolution)
                distance = math.dist(start_xy, xy)
                if distance > max(.25, resolution * 3) + 1.0e-9:
                    continue
                valid, _, _ = translation_ok(
                    start_xy,
                    xy,
                    yaw,
                    allow_start_egress=needs_egress,
                )
                if valid:
                    candidates.append((distance, cell))
        candidates.sort()
        turn_cost = .05 * abs(
            (yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        )
        start_states.extend(
            (layer_index, cell, distance + turn_cost)
            for distance, cell in candidates[:16]
        )

    goal_layers: set[int] = set()
    for layer_index, layer in enumerate(layers):
        if not np.asarray(layer["allowed"])[goal_cell]:
            continue
        valid, _, _ = translation_ok(
            _xy(goal_cell, origin, resolution), goal_xy, float(layer["yaw"])
        )
        if valid:
            goal_layers.add(layer_index)
    if diagnostic is not None:
        diagnostic.update(
            start_state_count=len(start_states),
            start_layer_count=len({item[0] for item in start_states}),
            goal_layer_count=len(goal_layers),
        )
    if not start_states or not goal_layers:
        if diagnostic is not None:
            diagnostic["failure"] = "no_valid_endpoint_state"
        return None

    structure = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
    )
    component_labels = [
        ndimage.label(np.asarray(layer["allowed"]), structure=structure)[0]
        for layer in layers
    ]

    def build_rotation_links(
        forbidden: set[tuple[int, int, int, int]],
    ) -> list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ]
    ]:
        links: list[
            tuple[
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
            ]
        ] = []
        partial_turn_audit: list[dict[str, Any]] = []
        ordered_layers = sorted(
            range(len(layers)),
            key=lambda index: (
                float(layers[index]["yaw"]) - start_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi,
        )
        layer_pairs = list(zip(ordered_layers[:-1], ordered_layers[1:]))
        linked_component_pairs: set[tuple[int, int, int, int]] = set()
        for first, second in layer_pairs:
            overlap = (
                np.asarray(layers[first]["allowed"])
                & np.asarray(layers[second]["allowed"])
            )
            overlap_rows, overlap_columns = np.nonzero(overlap)
            if not len(overlap_rows):
                continue
            pairs = np.column_stack(
                (
                    component_labels[first][overlap_rows, overlap_columns],
                    component_labels[second][overlap_rows, overlap_columns],
                )
            )
            unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
            for pair_index, (component0, component1) in enumerate(unique_pairs):
                if not component0 or not component1:
                    continue
                members = np.flatnonzero(inverse == pair_index)
                fully_turnable = members[
                    turnable[
                        overlap_rows[members], overlap_columns[members]
                    ]
                ]
                if len(fully_turnable):
                    if overlap_cost is None:
                        member_index = int(
                            fully_turnable[len(fully_turnable) // 2]
                        )
                    else:
                        member_index = min(
                            (int(item) for item in fully_turnable),
                            key=lambda item: (
                                float(
                                    overlap_cost[
                                        overlap_rows[item], overlap_columns[item]
                                    ]
                                ),
                                math.dist(
                                    start_xy,
                                    _xy(
                                        (
                                            int(overlap_rows[item]),
                                            int(overlap_columns[item]),
                                        ),
                                        origin,
                                        resolution,
                                    ),
                                )
                                + math.dist(
                                    goal_xy,
                                    _xy(
                                        (
                                            int(overlap_rows[item]),
                                            int(overlap_columns[item]),
                                        ),
                                        origin,
                                        resolution,
                                    ),
                                ),
                                int(overlap_rows[item]),
                                int(overlap_columns[item]),
                            ),
                        )
                    row = int(overlap_rows[member_index])
                    column = int(overlap_columns[member_index])
                    spatial = row * width + column
                    edge = (
                        spatial,
                        spatial,
                        min(first, second),
                        max(first, second),
                    )
                    if edge not in forbidden:
                        links.append(
                            (
                                (first, int(component0)),
                                (second, int(component1)),
                                (row, column),
                                (row, column),
                            )
                        )
                        linked_component_pairs.add(
                            (first, int(component0), second, int(component1))
                        )
                        continue
                endpoint_spare = np.minimum(
                    np.asarray(layers[first]["configuration_clearance_m"])[
                        overlap_rows[members], overlap_columns[members]
                    ],
                    np.asarray(layers[second]["configuration_clearance_m"])[
                        overlap_rows[members], overlap_columns[members]
                    ],
                )
                # Highest endpoint clearance is the best predictor of a clear
                # intermediate sweep. Add a deterministic spatial spread so a
                # long room does not test 96 adjacent cells at one end.
                if overlap_cost is None:
                    order = np.argsort(-endpoint_spare, kind="stable")
                else:
                    order = np.asarray(
                        sorted(
                            range(len(members)),
                            key=lambda item: (
                                -float(endpoint_spare[item]),
                                float(
                                    overlap_cost[
                                        overlap_rows[members[item]],
                                        overlap_columns[members[item]],
                                    ]
                                ),
                                int(overlap_rows[members[item]]),
                                int(overlap_columns[members[item]]),
                            ),
                        ),
                        dtype=np.int64,
                    )
                if len(order) > 12:
                    high = order[:8]
                    spread = order[
                        np.linspace(0, len(order) - 1, 4).astype(int)
                    ]
                    order = np.unique(np.concatenate((high, spread)))
                yaw0 = float(layers[first]["yaw"])
                yaw1 = float(layers[second]["yaw"])
                for member_index in members[order]:
                    row = int(overlap_rows[member_index])
                    column = int(overlap_columns[member_index])
                    spatial = row * width + column
                    edge = (
                        spatial,
                        spatial,
                        min(first, second),
                        max(first, second),
                    )
                    if edge in forbidden:
                        continue
                    xy = _xy((row, column), origin, resolution)
                    if (
                        static_gap(xy, xy, yaw0, yaw1) + 1.0e-9
                        < static_margin
                        or depth_gap(xy, xy, yaw0, yaw1) + 1.0e-9
                        < depth_margin
                    ):
                        continue
                    links.append(
                        (
                            (first, int(component0)),
                            (second, int(component1)),
                            (row, column),
                            (row, column),
                        )
                    )
                    linked_component_pairs.add(
                        (first, int(component0), second, int(component1))
                    )
                    break

        # Where neighboring fixed-yaw free spaces do not overlap at a safe
        # in-place turn, certify bounded coupled translation/rotation
        # primitives.  A one-cell transition is insufficient when the two
        # rectangular-footprint configuration spaces separate around a door
        # edge.  Probe a sparse, rotation-invariant set of metric offsets up to
        # 30 cm; endpoint membership is only a proposal and the exact continuous
        # static/depth sweeps below remain the authority.
        ordered_arc_offsets = [
            (drow, dcolumn) for drow, dcolumn, _ in _NEIGHBORS
        ]
        max_arc_cells = max(
            1, int(math.floor(PARTIAL_TURN_MAX_TRANSLATION_M / resolution + 1.0e-9))
        )
        distance_levels = np.unique(
            np.rint(
                np.linspace(
                    1,
                    max_arc_cells,
                    min(PARTIAL_TURN_DISTANCE_LEVELS, max_arc_cells),
                )
            ).astype(np.int64)
        )
        for radius_cells in distance_levels:
            for angle in np.linspace(
                0.0,
                2.0 * math.pi,
                PARTIAL_TURN_DIRECTION_COUNT,
                endpoint=False,
            ):
                offset = (
                    int(round(float(radius_cells) * math.sin(float(angle)))),
                    int(round(float(radius_cells) * math.cos(float(angle)))),
                )
                if offset != (0, 0) and offset not in ordered_arc_offsets:
                    ordered_arc_offsets.append(offset)
        for first, second in layer_pairs:
            yaw0 = float(layers[first]["yaw"])
            yaw1 = float(layers[second]["yaw"])
            yaw_delta = abs((yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi)
            if yaw_delta > math.radians(20.0) + 1.0e-9:
                continue
            allowed0 = np.asarray(layers[first]["allowed"])
            allowed1 = np.asarray(layers[second]["allowed"])
            layer_audit: dict[str, Any] = {
                "from_layer": int(first),
                "to_layer": int(second),
                "from_yaw_deg": math.degrees(yaw0),
                "to_yaw_deg": math.degrees(yaw1),
                "component_pair_count": 0,
                "candidate_sweep_count": 0,
                "static_rejection_count": 0,
                "depth_rejection_count": 0,
                "accepted_count": 0,
                "accepted_translation_m": [],
                "best_rejected_static_gap_m": None,
                "best_rejected_depth_gap_m": None,
            }
            candidates_by_components: dict[
                tuple[int, int, int, int],
                list[tuple[float, float, tuple[int, int], tuple[int, int]]],
            ] = {}
            for drow, dcolumn in ordered_arc_offsets:
                row0_start = max(0, -drow)
                row0_stop = min(height, height - drow)
                column0_start = max(0, -dcolumn)
                column0_stop = min(width, width - dcolumn)
                local = allowed0[
                    row0_start:row0_stop, column0_start:column0_stop
                ] & allowed1[
                    row0_start + drow:row0_stop + drow,
                    column0_start + dcolumn:column0_stop + dcolumn,
                ]
                local_rows, local_columns = np.nonzero(local)
                if not len(local_rows):
                    continue
                rows0 = local_rows + row0_start
                columns0 = local_columns + column0_start
                rows1 = rows0 + drow
                columns1 = columns0 + dcolumn
                components0 = component_labels[first][rows0, columns0]
                components1 = component_labels[second][rows1, columns1]
                endpoint_spare = np.minimum(
                    np.asarray(layers[first]["configuration_clearance_m"])[
                        rows0, columns0
                    ],
                    np.asarray(layers[second]["configuration_clearance_m"])[
                        rows1, columns1
                    ],
                )
                valid_components = (components0 > 0) & (components1 > 0)
                if not np.any(valid_components):
                    continue
                component_pairs = np.unique(
                    np.column_stack(
                        (components0[valid_components], components1[valid_components])
                    ),
                    axis=0,
                )
                for raw_component0, raw_component1 in component_pairs:
                    component0 = int(raw_component0)
                    component1 = int(raw_component1)
                    key = (first, component0, second, component1)
                    if key in linked_component_pairs:
                        continue
                    members = np.flatnonzero(
                        (components0 == component0) & (components1 == component1)
                    )
                    order = members[
                        np.argsort(-endpoint_spare[members], kind="stable")
                    ]
                    if len(order) > 8:
                        high = order[:6]
                        spread = order[
                            np.linspace(0, len(order) - 1, 2).astype(int)
                        ]
                        order = np.unique(np.concatenate((high, spread)))
                    bucket = candidates_by_components.setdefault(key, [])
                    for index in order[:PARTIAL_TURN_CANDIDATES_PER_OFFSET]:
                        bucket.append(
                            (
                                math.hypot(drow, dcolumn) * resolution,
                                float(endpoint_spare[index]),
                                (int(rows0[index]), int(columns0[index])),
                                (int(rows1[index]), int(columns1[index])),
                            )
                        )
            for key, candidates in candidates_by_components.items():
                layer_audit["component_pair_count"] += 1
                first_layer, component0, second_layer, component1 = key
                candidates.sort(
                    key=lambda item: (
                        -item[1],
                        0.0
                        if overlap_cost is None
                        else .5
                        * (
                            float(overlap_cost[item[2]])
                            + float(overlap_cost[item[3]])
                        ),
                        item[0],
                        item[2],
                        item[3],
                    )
                )
                # Preserve metric diversity.  Otherwise many high-clearance
                # one-cell proposals can consume the candidate budget even
                # though only a longer coupled sweep joins the components.
                selected: list[
                    tuple[float, float, tuple[int, int], tuple[int, int]]
                ] = []
                per_distance: dict[int, int] = {}
                for candidate in candidates:
                    distance_key = int(round(candidate[0] / resolution))
                    used = per_distance.get(distance_key, 0)
                    if used >= PARTIAL_TURN_CANDIDATES_PER_DISTANCE:
                        continue
                    per_distance[distance_key] = used + 1
                    selected.append(candidate)
                for _, _, cell0, cell1 in selected:
                    spatial0 = cell0[0] * width + cell0[1]
                    spatial1 = cell1[0] * width + cell1[1]
                    edge = (
                        min(spatial0, spatial1),
                        max(spatial0, spatial1),
                        min(first_layer, second_layer),
                        max(first_layer, second_layer),
                    )
                    if edge in forbidden:
                        continue
                    xy0 = _xy(cell0, origin, resolution)
                    xy1 = _xy(cell1, origin, resolution)
                    layer_audit["candidate_sweep_count"] += 1
                    map_gap = static_gap(xy0, xy1, yaw0, yaw1)
                    observed_gap = depth_gap(xy0, xy1, yaw0, yaw1)
                    if map_gap + 1.0e-9 < static_margin:
                        layer_audit["static_rejection_count"] += 1
                        prior = layer_audit["best_rejected_static_gap_m"]
                        layer_audit["best_rejected_static_gap_m"] = (
                            map_gap if prior is None else max(float(prior), map_gap)
                        )
                        continue
                    if observed_gap + 1.0e-9 < depth_margin:
                        layer_audit["depth_rejection_count"] += 1
                        prior = layer_audit["best_rejected_depth_gap_m"]
                        layer_audit["best_rejected_depth_gap_m"] = (
                            observed_gap
                            if prior is None
                            else max(float(prior), observed_gap)
                        )
                        continue
                    links.append(
                        (
                            (first_layer, component0),
                            (second_layer, component1),
                            cell0,
                            cell1,
                        )
                    )
                    linked_component_pairs.add(key)
                    layer_audit["accepted_count"] += 1
                    layer_audit["accepted_translation_m"].append(
                        math.dist(xy0, xy1)
                    )
                    break
            partial_turn_audit.append(layer_audit)
        if diagnostic is not None:
            diagnostic["partial_turn_audit"] = partial_turn_audit
        return links

    forbidden_translation_edges: set[tuple[int, int, int]] = set()
    forbidden_rotation_edges: set[tuple[int, int, int, int]] = set()
    state_path: list[tuple[int, int, int]] | None = None
    expanded_total = 0
    plane_size = height * width
    for _ in range(MAX_EDGE_REPLAN_PASSES):
        rotation_links_started = time.monotonic()
        rotation_links = build_rotation_links(forbidden_rotation_edges)
        if diagnostic is not None:
            diagnostic["rotation_link_seconds"] = (
                time.monotonic() - rotation_links_started
            )
            diagnostic["certified_partial_rotation_link_count"] = len(
                rotation_links
            )
            start_nodes = {
                (
                    int(layer_index),
                    int(component_labels[layer_index][cell]),
                )
                for layer_index, cell, _ in start_states
                if int(component_labels[layer_index][cell]) > 0
            }
            goal_nodes = {
                (int(layer_index), int(component_labels[layer_index][goal_cell]))
                for layer_index in goal_layers
                if int(component_labels[layer_index][goal_cell]) > 0
            }
            adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
            for first_node, second_node, _, _ in rotation_links:
                adjacency.setdefault(first_node, set()).add(second_node)
                adjacency.setdefault(second_node, set()).add(first_node)
            reachable = set(start_nodes)
            frontier = list(start_nodes)
            while frontier:
                node = frontier.pop()
                for neighbour in adjacency.get(node, ()):
                    if neighbour not in reachable:
                        reachable.add(neighbour)
                        frontier.append(neighbour)
            diagnostic["component_graph"] = {
                "start_nodes": [list(node) for node in sorted(start_nodes)],
                "goal_nodes": [list(node) for node in sorted(goal_nodes)],
                "reachable_nodes": [list(node) for node in sorted(reachable)],
                "reachable_goal": bool(reachable & goal_nodes),
                "links": [
                    {
                        "from_node": list(first_node),
                        "to_node": list(second_node),
                        "translation_m": math.dist(
                            _xy(first_cell, origin, resolution),
                            _xy(second_cell, origin, resolution),
                        ),
                    }
                    for first_node, second_node, first_cell, second_cell in rotation_links
                ],
            }
        component_search_started = time.monotonic()
        try:
            candidate, expanded = _component_oriented_path(
                layers,
                start_states,
                goal_layers,
                goal_cell,
                rotation_links,
                resolution,
                clearance_weight,
                forbidden_translation_edges=forbidden_translation_edges,
                forbidden_rotation_edges=forbidden_rotation_edges,
                progress_check=progress_check,
                overlap_cost=overlap_cost,
            )
        except NavigationPlanError as exc:
            if diagnostic is not None:
                diagnostic["component_search_seconds"] = (
                    time.monotonic() - component_search_started
                )
                diagnostic["failure"] = (
                    "state_lattice_disconnected_or_timed_out"
                )
                diagnostic["detail"] = str(exc)
            return None
        if diagnostic is not None:
            diagnostic["component_search_seconds"] = (
                time.monotonic() - component_search_started
            )
        expanded_total += expanded
        rejected = False
        for first, second in zip(candidate[:-1], candidate[1:]):
            layer0, row0, column0 = first
            layer1, row1, column1 = second
            spatial0 = row0 * width + column0
            spatial1 = row1 * width + column1
            xy0 = _xy((row0, column0), origin, resolution)
            xy1 = _xy((row1, column1), origin, resolution)
            if layer0 == layer1:
                valid, _, _ = translation_ok(
                    xy0, xy1, float(layers[layer0]["yaw"])
                )
                if valid:
                    continue
                forbidden_translation_edges.add(
                    (layer0, min(spatial0, spatial1), max(spatial0, spatial1))
                )
            else:
                yaw0 = float(layers[layer0]["yaw"])
                yaw1 = float(layers[layer1]["yaw"])
                if (
                    static_gap(xy0, xy1, yaw0, yaw1) + 1.0e-9 >= static_margin
                    and depth_gap(xy0, xy1, yaw0, yaw1) + 1.0e-9 >= depth_margin
                ):
                    continue
                forbidden_rotation_edges.add(
                    (
                        min(spatial0, spatial1),
                        max(spatial0, spatial1),
                        min(layer0, layer1),
                        max(layer0, layer1),
                    )
                )
            rejected = True
            break
        if not rejected:
            state_path = candidate
            break
    if state_path is None:
        if diagnostic is not None:
            diagnostic["failure"] = "continuous_edge_replan_exhausted"
        return None
    if diagnostic is not None:
        diagnostic["selected_layer_transitions"] = [
            {
                "from_yaw_deg": math.degrees(float(layers[first[0]]["yaw"])),
                "to_yaw_deg": math.degrees(float(layers[second[0]]["yaw"])),
                "from_xy_m": list(_xy((first[1], first[2]), origin, resolution)),
                "to_xy_m": list(_xy((second[1], second[2]), origin, resolution)),
                "preferred_cost": (
                    0.0
                    if overlap_cost is None
                    else float(overlap_cost[second[1], second[2]])
                ),
            }
            for first, second in zip(state_path[:-1], state_path[1:])
            if first[0] != second[0]
        ]
        diagnostic["selected_grid_preferred_cost_m"] = _path_overlap_cost(
            [(row, column) for _, row, column in state_path],
            overlap_cost,
            resolution,
        )

    # Convert the state path into fixed-heading translations and short,
    # continuously certified translation/rotation primitives. A same-cell
    # layer change remains an in-place spin before the following segment.
    elementary: list[
        tuple[
            tuple[float, float],
            tuple[float, float],
            float,
            float,
        ]
    ] = []
    first_layer, first_row, first_column = state_path[0]
    first_xy = _xy((first_row, first_column), origin, resolution)
    if math.dist(start_xy, first_xy) > 1.0e-8:
        first_yaw = float(layers[first_layer]["yaw"])
        elementary.append((tuple(start_xy), first_xy, first_yaw, first_yaw))
    active_layer = first_layer
    previous_xy = first_xy
    for layer_index, row, column in state_path[1:]:
        xy = _xy((row, column), origin, resolution)
        if math.dist(previous_xy, xy) <= 1.0e-9:
            active_layer = layer_index
            continue
        previous_yaw = float(layers[active_layer]["yaw"])
        next_yaw = float(layers[layer_index]["yaw"])
        elementary.append((previous_xy, xy, previous_yaw, next_yaw))
        previous_xy = xy
        active_layer = layer_index
    final_yaw = float(layers[active_layer]["yaw"])
    if math.dist(previous_xy, goal_xy) > 1.0e-8:
        elementary.append((previous_xy, goal_xy, final_yaw, final_yaw))
    if not elementary:
        return None

    # Greedily collapse each constant-yaw run, retaining exact continuous
    # certificates and the executor's segment-length bound.  Coupled turns stay
    # as their independently certified short primitives; the executor's arc-
    # aware waypoint policy prevents their yaw transition from being skipped.
    path: list[tuple[float, float]] = [elementary[0][0]]
    segment_start_yaws: list[float] = []
    segment_end_yaws: list[float] = []
    arc_segment_indices: list[int] = []
    index = 0
    while index < len(elementary):
        yaw0 = elementary[index][2]
        yaw1 = elementary[index][3]
        is_arc = abs(
            (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        ) > 1.0e-8
        if is_arc:
            end = elementary[index][1]
            if math.dist(path[-1], end) <= 1.0e-9:
                return None
            arc_segment_indices.append(len(segment_end_yaws))
            path.append(end)
            segment_start_yaws.append(yaw0)
            segment_end_yaws.append(yaw1)
            index += 1
            continue

        yaw = yaw0
        run_end = index
        while (
            run_end + 1 < len(elementary)
            and abs(
                (elementary[run_end + 1][2] - yaw + math.pi)
                % (2.0 * math.pi)
                - math.pi
            ) <= 1.0e-8
            and abs(
                (elementary[run_end + 1][3] - yaw + math.pi)
                % (2.0 * math.pi)
                - math.pi
            ) <= 1.0e-8
            and math.dist(elementary[run_end][1], elementary[run_end + 1][0])
            <= 1.0e-8
        ):
            run_end += 1
        cursor = index
        while cursor <= run_end:
            accepted = cursor
            for candidate_index in range(run_end, cursor - 1, -1):
                end = elementary[candidate_index][1]
                if math.dist(path[-1], end) > max_segment_m + 1.0e-9:
                    continue
                if not _shortcut_preserves_preference(
                    path[-1],
                    end,
                    [
                        (elementary[item][0], elementary[item][1])
                        for item in range(cursor, candidate_index + 1)
                    ],
                    overlap_cost,
                    origin,
                    resolution,
                ):
                    continue
                if not (
                    needs_egress and len(segment_end_yaws) == 0
                ) and _line_cells(
                    np.asarray(layers[active_layer]["allowed"]),
                    _cell(path[-1], origin, resolution),
                    _cell(end, origin, resolution),
                ) is None:
                    continue
                valid, _, _ = translation_ok(
                    path[-1],
                    end,
                    yaw,
                    allow_start_egress=(
                        needs_egress and len(segment_end_yaws) == 0
                    ),
                )
                if valid:
                    accepted = candidate_index
                    break
            end = elementary[accepted][1]
            if math.dist(path[-1], end) <= 1.0e-9:
                return None
            path.append(end)
            segment_start_yaws.append(yaw)
            segment_end_yaws.append(yaw)
            cursor = accepted + 1
        index = run_end + 1

    depth_segment_gaps: list[float] = []
    static_segment_gaps: list[float] = []
    depth_rotation_gaps: list[float] = []
    static_rotation_gaps: list[float] = []
    constrained_indices: list[int] = []
    previous_yaw = start_yaw
    for segment_index, (start, end, yaw0, yaw1) in enumerate(
        zip(
            path[:-1],
            path[1:],
            segment_start_yaws,
            segment_end_yaws,
        )
    ):
        rotation_depth = depth_gap(start, start, previous_yaw, yaw0)
        rotation_static = static_gap(start, start, previous_yaw, yaw0)
        rotation_required = (
            start_depth_gap - 1.0e-9
            if needs_egress and segment_index == 0
            else depth_margin
        )
        is_arc = segment_index in arc_segment_indices
        if is_arc:
            map_gap = static_gap(start, end, yaw0, yaw1)
            measured_depth_gap = depth_gap(start, end, yaw0, yaw1)
            valid = bool(
                map_gap + 1.0e-9 >= static_margin
                and measured_depth_gap + 1.0e-9 >= depth_margin
            )
        else:
            valid, map_gap, measured_depth_gap = translation_ok(
                start,
                end,
                yaw0,
                allow_start_egress=(needs_egress and segment_index == 0),
            )
        if (
            not valid
            or rotation_depth + 1.0e-9 < rotation_required
            or rotation_static + 1.0e-9 < static_margin
        ):
            return None
        chord_yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        chord_delta = (chord_yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        default_yaw = chord_yaw if abs(chord_delta) > math.radians(45.0) else previous_yaw
        if (
            is_arc
            or abs(
                (yaw0 - default_yaw + math.pi) % (2.0 * math.pi) - math.pi
            ) > math.radians(2.0)
            or abs(
                (yaw1 - default_yaw + math.pi) % (2.0 * math.pi) - math.pi
            ) > math.radians(2.0)
        ):
            constrained_indices.append(segment_index)
        depth_segment_gaps.append(float(measured_depth_gap))
        static_segment_gaps.append(float(map_gap))
        depth_rotation_gaps.append(float(rotation_depth))
        static_rotation_gaps.append(float(rotation_static))
        previous_yaw = yaw1

    grid_path: list[list[int]] = []
    for _, row, column in state_path:
        cell = [int(row), int(column)]
        if not grid_path or grid_path[-1] != cell:
            grid_path.append(cell)
    return {
        "path": path,
        "segment_yaws": segment_end_yaws,
        "segment_start_yaws": segment_start_yaws,
        "segment_end_yaws": segment_end_yaws,
        "arc_segment_indices": arc_segment_indices,
        "depth_segment_gaps": depth_segment_gaps,
        "static_segment_gaps": static_segment_gaps,
        "depth_rotation_gaps": depth_rotation_gaps,
        "static_rotation_gaps": static_rotation_gaps,
        "orientation_constrained_segment_indices": constrained_indices,
        "turning_count": int(sum(
            abs((current - previous + math.pi) % (2.0 * math.pi) - math.pi)
            > 1.0e-8
            for previous, current in zip(
                [start_yaw, *segment_end_yaws[:-1]], segment_start_yaws
            )
        ) + sum(
            abs((end - start + math.pi) % (2.0 * math.pi) - math.pi)
            > 1.0e-8
            for start, end in zip(segment_start_yaws, segment_end_yaws)
        )),
        "turnable_cell_count": int(np.count_nonzero(turnable)),
        "expanded_cells": int(expanded_total),
        "grid_path": grid_path,
        "needs_egress": bool(needs_egress),
        "start_depth_gap": start_depth_gap,
    }


def plan_depth_refined_path(
    snapshot: Mapping[str, Any], map_only_plan: Mapping[str, Any],
    polygon: np.ndarray, polygon_digest: str,
    *, progress_check: Callable[[], None] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    _lattice_resolution_m: float = .02501,
    _search_deadline: float | None = None,
    _tracking_reserve_m: float | None = None,
) -> dict[str, Any] | None:
    """Resolve a coarse-depth rejection with exact oriented base geometry.

    The static map and private live-depth layer both use the signed base
    polygon. Initial rotation, every translation, and every later yaw change
    are checked continuously; turns are allowed only in certified open pockets.
    The bounded fallback never edits the SLAM map or promotes unknown cells.
    """
    raw = snapshot.get("navigation_depth_points_xy_m")
    if raw is None or map_only_plan.get("ok") is not True:
        return None
    search_deadline = time.monotonic() + 45.0 if _search_deadline is None else _search_deadline
    tracking_reserve_m = (
        DEPTH_TRACKING_RESERVE_M
        if _tracking_reserve_m is None
        else float(_tracking_reserve_m)
    )
    if tracking_reserve_m not in (
        DEPTH_TRACKING_RESERVE_M,
        DEPTH_TIGHT_TRACKING_RESERVE_M,
    ):
        return None
    margin_policy = (
        DEPTH_STRICT_MARGIN_POLICY
        if tracking_reserve_m == DEPTH_TRACKING_RESERVE_M
        else DEPTH_TIGHT_MARGIN_POLICY
    )

    def check_progress():
        if progress_check is not None:
            progress_check()
        if time.monotonic() >= search_deadline:
            raise NavigationPlanError("oriented depth search exhausted its bounded CPU-time window")
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        return None
    original_point_count = len(points)
    points, point_radii = compact_depth_disks(points)
    max_point_radius = float(point_radii.max())
    if len(points) < 4:
        return None
    planning_snapshot, map_evidence_audit = (
        _replay_reference_traversed_evidence(snapshot, map_only_plan)
    )
    obstacle, free, resolution, origin = _snapshot_geometry(planning_snapshot)
    if resolution > 0.051:
        return None
    # Keep the actual wheel envelope inside the static-map circle certificate.
    radius = float(map_only_plan["robot_radius_m"])
    required = radius + START_EGRESS_SAFETY_MARGIN_M
    body_radius = float(np.linalg.norm(polygon, axis=1).max())
    if body_radius + 0.01 > required:
        return None
    # A 5cm centre lattice can miss the remaining offset in a tight doorway.
    # Subdivide occupied cells without changing their geometric boundary.
    fr, fc = np.nonzero(free)
    if not len(fr):
        return None
    padding = int(math.ceil(required / resolution)) + 2
    r0, r1 = max(0, int(fr.min())-padding), min(free.shape[0], int(fr.max())+padding+1)
    c0, c1 = max(0, int(fc.min())-padding), min(free.shape[1], int(fc.max())+padding+1)
    try:
        reference_path = np.asarray(
            map_only_plan["path_xy_m"], dtype=np.float64
        ).reshape(-1, 2)
    except (KeyError, TypeError, ValueError):
        return None
    if len(reference_path) < 2 or not np.all(np.isfinite(reference_path)):
        return None
    route_pad = DEPTH_REFINEMENT_ROUTE_CROP_PAD_M
    route_low = reference_path.min(axis=0) - route_pad
    route_high = reference_path.max(axis=0) + route_pad
    route_c0 = int(math.floor((route_low[0] - origin[0]) / resolution))
    route_c1 = int(math.ceil((route_high[0] - origin[0]) / resolution))
    route_r0 = int(math.floor((route_low[1] - origin[1]) / resolution))
    route_r1 = int(math.ceil((route_high[1] - origin[1]) / resolution))
    c0, c1 = max(c0, route_c0), min(c1, route_c1)
    r0, r1 = max(r0, route_r0), min(r1, route_r1)
    if r0 >= r1 or c0 >= c1:
        return None
    source_grid_shape = tuple(int(value) for value in free.shape)
    origin = (origin[0] + c0 * resolution, origin[1] + r0 * resolution)
    refinement_factor = max(1, int(math.ceil(resolution / _lattice_resolution_m)))
    if (r1-r0) * (c1-c0) * refinement_factor**2 > MAX_GRID_CELLS:
        return None
    free = np.repeat(np.repeat(free[r0:r1, c0:c1], refinement_factor, axis=0), refinement_factor, axis=1)
    obstacle = np.repeat(np.repeat(obstacle[r0:r1, c0:c1], refinement_factor, axis=0), refinement_factor, axis=1)
    resolution /= refinement_factor
    blocked = obstacle | ~free
    preferred_path_cost = _preferred_path_cost(
        planning_snapshot,
        free.shape,
        origin,
        resolution,
        robot_radius_m=radius,
    )
    clearance = np.maximum(0, ndimage.distance_transform_edt(~blocked) * resolution
                           - resolution / math.sqrt(2))
    rr, cc = np.indices(free.shape)
    boundary = np.minimum(np.minimum(rr + .5, free.shape[0] - rr - .5),
                          np.minimum(cc + .5, free.shape[1] - cc - .5)) * resolution
    clearance = np.minimum(clearance, boundary)
    circular_traversable = ~blocked & (clearance >= required)
    pose = _pose(snapshot)
    start_xy = np.asarray(pose[:2])
    goal_xy = tuple(map_only_plan["planned_goal_xy_m"])
    goal = _cell(goal_xy, origin, resolution)
    start = _cell(start_xy, origin, resolution)
    if not (0 <= goal[0] < free.shape[0] and 0 <= goal[1] < free.shape[1]
            and not blocked[goal]):
        return None
    margin = DEPTH_POINT_MARGIN_M + tracking_reserve_m
    static_margin = START_EGRESS_SAFETY_MARGIN_M
    rows, columns = np.nonzero(~blocked)
    centers = np.column_stack((origin[0] + (columns + .5) * resolution,
                               origin[1] + (rows + .5) * resolution))
    tree = cKDTree(points)
    nearby = tree.query(centers)[0] <= body_radius + margin + resolution * 2 + max_point_radius
    local_centers = centers[nearby]
    neighbors = tree.query_ball_point(local_centers, body_radius + margin + resolution * 2 + max_point_radius)
    counts = np.fromiter((len(items) for items in neighbors), dtype=int)
    owners = np.repeat(np.arange(len(neighbors)), counts)
    point_ids = np.asarray([item for items in neighbors for item in items], dtype=int)
    relative_points = points[point_ids] - local_centers[owners]
    pair_point_radii = point_radii[point_ids]
    cuda_depth_clearance = _prepare_cuda_depth_clearance(
        relative_points,
        pair_point_radii,
        owners,
        len(local_centers),
        polygon,
    )
    start_distances = convex_sweep_distances(
        points, polygon, start_xy, start_xy, pose[2], pose[2]
    ) - point_radii
    start_depth_gap = float(start_distances.min())
    needs_egress = start_depth_gap < margin

    expanded_total = 0
    # Door/partition normals are useful proposals, never geometric constraints.
    # Align the footprint's narrow caliper with each observed elongated cluster.
    neighbor_distances, neighbor_ids = tree.query(points, k=min(6, len(points)))
    links = neighbor_distances < .12
    owners_graph = np.broadcast_to(np.arange(len(points))[:, None], links.shape)
    graph = coo_matrix((np.ones(int(links.sum())), (owners_graph[links], neighbor_ids[links])),
                       shape=(len(points), len(points)))
    count, labels = connected_components(graph, directed=False)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    widths = np.ptp(polygon @ normals.T, axis=0)
    narrow = normals[int(widths.argmin())]
    narrow_angle = math.atan2(narrow[1], narrow[0])
    proposals = []
    for label in range(count):
        cluster = points[labels == label]
        if len(cluster) < 4:
            continue
        eigenvalues, axes = np.linalg.eigh(np.cov(cluster.T))
        axis = axes[:, -1]
        length = float(np.ptp(cluster @ axis))
        if length < .35 or eigenvalues[-1] < 8 * max(eigenvalues[0], 1e-8):
            continue
        normal_angle = math.atan2(axis[1], axis[0]) + math.pi / 2
        proposals.extend(normal_angle - narrow_angle + turn * math.pi / 2 for turn in range(4))
    proposals.sort(key=lambda angle: abs((angle - pose[2] + math.pi) % (2*math.pi)-math.pi))
    headings, coarse_heading_layer_count = _oriented_heading_candidates(
        planning_snapshot, map_only_plan, pose[2], proposals
    )
    checked_headings = set()
    orientation_layers: list[dict[str, Any]] = []
    orientation_union: np.ndarray | None = None
    # The coarse 15-degree ladder establishes the relevant angular span.  Add a
    # small route-relative batch of midpoint layers before the first expensive
    # component search, then continue in bounded batches only if necessary.
    # This reaches a narrow feasible band early without assuming a world axis or
    # any scene-specific doorway angle.
    refinement_batch_size = max(
        1, min(3, max(1, coarse_heading_layer_count - 1))
    )
    next_variable_attempt_layer_count = max(
        2, coarse_heading_layer_count + refinement_batch_size
    )

    def endpoint_component_count(mask):
        component_labels, _ = ndimage.label(
            mask,
            structure=np.asarray(
                ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
            ),
        )
        goal_label = int(component_labels[goal])
        if goal_label == 0:
            return 0
        radius_cells = max(2, int(math.ceil(.25 / resolution)))
        count = 0
        for row in range(
            max(0, start[0] - radius_cells),
            min(mask.shape[0], start[0] + radius_cells + 1),
        ):
            for column in range(
                max(0, start[1] - radius_cells),
                min(mask.shape[1], start[1] + radius_cells + 1),
            ):
                if (
                    component_labels[row, column] == goal_label
                    and math.dist(
                        start_xy, _xy((row, column), origin, resolution)
                    ) <= max(.25, resolution * 3)
                ):
                    count += 1
        return count

    for proposed_yaw in headings:
        layer_started = time.monotonic()
        if time.monotonic() >= search_deadline:
            return None
        if progress_check is not None:
            progress_check()
        yaw = (proposed_yaw + math.pi) % (2 * math.pi) - math.pi
        heading_key = round(yaw, 6)
        if heading_key in checked_headings:
            continue
        checked_headings.add(heading_key)
        static_rotation_clearance = _oriented_static_sweep_clearance(
            polygon,
            start_xy,
            start_xy,
            pose[2],
            yaw,
            blocked,
            origin,
            resolution,
            measure_limit_m=static_margin + EXECUTION_CERTIFICATE_RESERVE_M,
        )
        rotation_clearance = float((convex_sweep_distances(
            points, polygon, start_xy, start_xy, pose[2], yaw) - point_radii).min())
        audit = {"yaw_deg": math.degrees(yaw), "rotation_clearance_m": rotation_clearance,
                 "static_rotation_clearance_m": static_rotation_clearance,
                 "lattice_resolution_m": resolution,
                 "point_margin_m": DEPTH_POINT_MARGIN_M,
                 "tracking_reserve_m": tracking_reserve_m,
                 "planning_margin_m": margin,
                 "margin_policy": margin_policy,
                 "source_grid_shape": source_grid_shape,
                 "planning_grid_shape": tuple(int(value) for value in free.shape),
                 "route_crop_pad_m": route_pad,
                 "depth_pair_count": int(len(relative_points)),
                 "depth_clearance_backend": (
                     "cuda_halfspace_lower_bound"
                     if cuda_depth_clearance is not None
                     else "cpu_halfspace_lower_bound"
                 )}
        if diagnostics is not None:
            diagnostics.append(audit)
        rotation_required = start_depth_gap - 1.0e-9 if needs_egress else margin
        initial_rotation_ok = bool(
            static_rotation_clearance + 1.0e-9 >= static_margin
            and rotation_clearance + 1.0e-9 >= rotation_required
            and (
                not needs_egress
                or abs((yaw-pose[2]+math.pi) % (2*math.pi)-math.pi) <= 1e-8
            )
            and start_depth_gap >= DEPTH_EGRESS_GAP_M
        )
        audit["initial_rotation_ok"] = initial_rotation_ok
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        static_invalid, static_kernel_cells = _oriented_static_invalid_mask(
            blocked, polygon, yaw, origin, resolution, static_margin
        )
        static_traversable = ~static_invalid
        audit["static_configuration_kernel_cells"] = static_kernel_cells
        audit["circular_static_cells_recovered"] = int(np.count_nonzero(
            static_traversable & ~circular_traversable
        ))
        # Search-space construction needs a one-sided guarantee, not an exact
        # distance for every point/centre pair.  The half-space lower bound is
        # conservative and matrix-multiplication friendly; exact 45-vertex
        # polygon sweeps remain mandatory for every selected edge below.
        local_clearance = _fixed_yaw_depth_clearance(
            relative_points,
            pair_point_radii,
            owners,
            len(local_centers),
            polygon,
            rotation,
            cuda_depth_clearance,
        )
        depth_clearance = np.full(free.shape, np.inf)
        depth_clearance[rows[nearby], columns[nearby]] = local_clearance
        allowed = static_traversable & (depth_clearance >= margin)
        if preferred_path_cost is not None:
            allowed &= np.isfinite(preferred_path_cost)
        orientation_layers.append({
            "yaw": yaw,
            "allowed": allowed,
            "depth_clearance": depth_clearance,
            "configuration_clearance_m": (
                ndimage.distance_transform_edt(allowed) * resolution
            ),
            "initial_rotation_ok": initial_rotation_ok,
        })
        orientation_union = (
            allowed.copy()
            if orientation_union is None
            else orientation_union | allowed
        )
        if diagnostics is not None:
            audit["variable_union_start_candidate_count"] = (
                endpoint_component_count(orientation_union)
            )
            audit["layer_build_seconds"] = time.monotonic() - layer_started
        if (
            len(orientation_layers) >= 2
            and len(orientation_layers) >= next_variable_attempt_layer_count
            and endpoint_component_count(orientation_union) > 0
        ):
            variable_diagnostic: dict[str, Any] = {
                "mode": "variable_orientation_attempt",
                "trigger_yaw_deg": math.degrees(yaw),
            }
            variable_started = time.monotonic()
            variable = _plan_variable_orientation_route(
                orientation_layers,
                polygon=polygon,
                points=points,
                point_radii=point_radii,
                blocked=blocked,
                clearance=clearance,
                origin=origin,
                resolution=resolution,
                start_xy=start_xy,
                start_cell=start,
                start_yaw=pose[2],
                goal_xy=goal_xy,
                goal_cell=goal,
                static_margin=static_margin,
                depth_margin=margin,
                max_segment_m=float(map_only_plan["max_execution_segment_m"]),
                clearance_weight=float(map_only_plan["clearance_weight"]),
                progress_check=check_progress,
                overlap_cost=preferred_path_cost,
                diagnostic=variable_diagnostic,
            )
            variable_diagnostic["total_seconds"] = (
                time.monotonic() - variable_started
            )
            if diagnostics is not None:
                diagnostics.append(variable_diagnostic)
            if variable is not None:
                path = variable["path"]
                execution_yaws = variable["segment_yaws"]
                execution_start_yaws = variable["segment_start_yaws"]
                execution_end_yaws = variable["segment_end_yaws"]
                arc_segment_indices = variable["arc_segment_indices"]
                execution_depth_certificates = variable[
                    "depth_segment_gaps"
                ]
                execution_static_gaps = variable["static_segment_gaps"]
                execution_rotation_clearances = variable[
                    "depth_rotation_gaps"
                ]
                execution_static_rotation_clearances = variable[
                    "static_rotation_gaps"
                ]
                tight_margin_segment_indices = (
                    [
                        index
                        for index, gap in enumerate(
                            execution_depth_certificates
                        )
                        if gap + 1.0e-9
                        < DEPTH_POINT_MARGIN_M + DEPTH_TRACKING_RESERVE_M
                    ]
                    if margin_policy == DEPTH_TIGHT_MARGIN_POLICY
                    else []
                )
                constrained_segment_indices = sorted(set(
                    variable["orientation_constrained_segment_indices"]
                ).union(tight_margin_segment_indices))
                map_certificates = [
                    radius + gap for gap in execution_static_gaps
                ]
                trials = list(map_only_plan["safety_margin_trials_m"])
                if START_EGRESS_SAFETY_MARGIN_M not in trials:
                    return None
                result = dict(map_only_plan)
                result.update(
                    path_xy_m=[list(point) for point in path],
                    corner_path_xy_m=[list(point) for point in path],
                    grid_path=variable["grid_path"],
                    path_length_m=_polyline_length(path),
                    path_segment_clearance_m=map_certificates,
                    minimum_clearance_m=min(map_certificates),
                    minimum_execution_clearance_m=min(map_certificates),
                    mean_clearance_m=float(np.mean(map_certificates)),
                    required_clearance_m=required,
                    effective_safety_margin_m=START_EGRESS_SAFETY_MARGIN_M,
                    safety_margin_trial_index=trials.index(
                        START_EGRESS_SAFETY_MARGIN_M
                    ),
                    safety_margin_relaxed=True,
                    start_safe_xy_m=list(start_xy),
                    start_snap_m=0.0,
                    start_egress_segment_count=0,
                    start_egress_required_clearance_m=required,
                    expanded_cells=(
                        expanded_total + int(variable["expanded_cells"])
                    ),
                    planning_resolution_m=resolution,
                    downsample_factor=1,
                    turning_count=int(variable["turning_count"]),
                    turning_clearance_m=[],
                    turning_repair_count=0,
                    turning_clearance_reserve_m=0.0,
                    planner=(
                        "clearance_astar_variable_orientation_static_and_"
                        "live_depth"
                    ),
                    # The exact-depth search may leave the reference
                    # centreline.  It is independently certified, but it is
                    # no longer a direct replay of the taught polyline.
                    traversed_route_direct_replay=False,
                    polyline_validation=(
                        "continuous_convex_static_map_depth_and_rotation_sweeps"
                    ),
                    route_sweep=route_sweep_audit(planning_snapshot, path),
                    depth_refinement={
                        "schema": DEPTH_REFINEMENT_SCHEMA,
                        "polygon_sha256": polygon_digest,
                        "segment_yaw_rad": execution_yaws,
                        "segment_start_yaw_rad": execution_start_yaws,
                        "segment_end_yaw_rad": execution_end_yaws,
                        "arc_segment_indices": arc_segment_indices,
                        "point_margin_m": DEPTH_POINT_MARGIN_M,
                        "tracking_reserve_m": tracking_reserve_m,
                        "planning_margin_m": margin,
                        "margin_policy": margin_policy,
                        "rotation_clearance_m": min(
                            execution_rotation_clearances
                        ),
                        "segment_clearance_m": (
                            execution_depth_certificates
                        ),
                        "point_count": original_point_count,
                        "compressed_disk_count": len(points),
                        "max_disk_radius_m": max_point_radius,
                        "heading_tolerance_deg": 2.0,
                        "max_speed_mps": .35,
                        "heading_policy": (
                            "face_chord_over_45_unless_depth_constrained"
                        ),
                        "orientation_mode": (
                            "se2_lattice_certified_partial_turn_and_arc"
                        ),
                        "orientation_layer_count": len(orientation_layers),
                        "turnable_cell_count": int(
                            variable["turnable_cell_count"]
                        ),
                        "orientation_constrained_segment_indices": (
                            constrained_segment_indices
                        ),
                        "tight_margin_segment_indices": (
                            tight_margin_segment_indices
                        ),
                        "orientation_constrained_max_speed_mps": .15,
                        "refresh_depth_after_alignment_deg": 45.0,
                        "static_map": {
                            "schema": STATIC_FOOTPRINT_SCHEMA,
                            "polygon_sha256": polygon_digest,
                            "planning_margin_m": static_margin,
                            "unknown_space_blocked": True,
                            "grid_cell_obstacles_filled": True,
                            "segment_clearance_m": execution_static_gaps,
                            "rotation_clearance_m": min(
                                execution_static_rotation_clearances
                            ),
                            "equivalent_center_clearance_m": map_certificates,
                        },
                    },
                )
                if map_evidence_audit is not None:
                    result["depth_refinement"]["map_evidence"] = (
                        map_evidence_audit
                    )
                result["start"] = dict(
                    result["start"], safe_grid_xy_m=list(start_xy)
                )
                result["snap"] = dict(
                    result["snap"],
                    start={
                        "applied": False,
                        "distance_m": 0.0,
                        "egress_certified": True,
                    },
                )
                if variable["needs_egress"]:
                    result["depth_refinement"]["start_egress"] = {
                        "schema": "monotone_convex_depth_egress_v1",
                        "segment_count": 1,
                        "physical_gap_floor_m": DEPTH_EGRESS_GAP_M,
                        "initial_clearance_m": float(
                            variable["start_depth_gap"]
                        ),
                        "fixed_yaw_rad": execution_yaws[0],
                        "separating": True,
                    }
                return result
            next_variable_attempt_layer_count = (
                len(orientation_layers) + refinement_batch_size
            )
        if not initial_rotation_ok:
            audit["failure"] = "initial_rotation_clearance"
            continue
        if not allowed[goal]:
            audit["failure"] = "goal_not_oriented_safe"
            continue

        def static_gap_at_yaw(a, b, motion_yaw):
            return _oriented_static_sweep_clearance(
                polygon,
                a,
                b,
                motion_yaw,
                motion_yaw,
                blocked,
                origin,
                resolution,
                measure_limit_m=static_margin + EXECUTION_CERTIFICATE_RESERVE_M,
            )

        def map_certificate_at_yaw(a, b, motion_yaw):
            gap = static_gap_at_yaw(a, b, motion_yaw)
            return gap + 1.0e-9 >= static_margin, radius + gap

        def map_certificate(a, b):
            return map_certificate_at_yaw(a, b, yaw)

        def depth_certificate_at_yaw(
            a, b, motion_yaw, *, allow_start_egress=False,
        ):
            # Restrict exact hull tests to returns near this continuous segment.
            search_margin = margin + .10
            lower = np.minimum(a, b) - body_radius - search_margin - max_point_radius
            upper = np.maximum(a, b) + body_radius + search_margin + max_point_radius
            selected_mask = np.all((points >= lower) & (points <= upper), axis=1)
            selected = points[selected_mask]
            if not len(selected):
                return True, search_margin
            distances = convex_sweep_distances(
                selected, polygon, a, b, motion_yaw, motion_yaw
            ) - point_radii[selected_mask]
            value = min(search_margin, float(distances.min()))
            if value >= margin:
                return True, value
            if not allow_start_egress or math.dist(a, start_xy) > 1e-8:
                return False, value
            # Fixed-yaw convex translation: the complete sweep must never
            # reduce any existing shortfall, and must finish outside it.
            end_distances = convex_sweep_distances(
                selected, polygon, b, b, motion_yaw, motion_yaw
            ) - point_radii[selected_mask]
            baseline = start_distances[selected_mask]
            return bool(
                np.all(distances >= np.minimum(baseline, margin) - 1e-9)
                and np.all(end_distances >= margin)
            ), value

        def depth_certificate(a, b):
            return depth_certificate_at_yaw(
                a, b, yaw, allow_start_egress=needs_egress,
            )

        components, _ = ndimage.label(allowed, structure=np.ones((3, 3)))
        start_radius = max(2, int(math.ceil(.25 / resolution))) if needs_egress else 2
        starts = [(r, c) for r in range(max(0, start[0]-start_radius), min(free.shape[0], start[0]+start_radius+1))
                  for c in range(max(0, start[1]-start_radius), min(free.shape[1], start[1]+start_radius+1))
                  if allowed[r, c] and components[r, c] == components[goal]
                  and math.dist(start_xy, _xy((r, c), origin, resolution)) <= max(.25, resolution * 3)]
        starts.sort(key=lambda cell: math.dist(start_xy, _xy(cell, origin, resolution)))
        start_cell = None
        map_valid_count = 0
        depth_valid_count = 0
        joint_valid_count = 0
        best_map_gap = -math.inf
        best_depth_gap = -math.inf
        for cell in starts:
            cell_xy = _xy(cell, origin, resolution)
            map_valid, map_gap = map_certificate(start_xy, cell_xy)
            depth_valid, depth_gap = depth_certificate(start_xy, cell_xy)
            map_valid_count += int(map_valid)
            depth_valid_count += int(depth_valid)
            best_map_gap = max(best_map_gap, float(map_gap))
            best_depth_gap = max(best_depth_gap, float(depth_gap))
            if map_valid and depth_valid:
                joint_valid_count += 1
                if start_cell is None:
                    start_cell = cell
        goal_connector_ok, goal_connector_gap = depth_certificate(
            _xy(goal, origin, resolution), goal_xy
        )
        if start_cell is None or not goal_connector_ok:
            audit["failure"] = "endpoint_connector"
            audit["endpoint_connector"] = {
                "same_component_start_candidate_count": int(len(starts)),
                "map_valid_start_candidate_count": int(map_valid_count),
                "depth_valid_start_candidate_count": int(depth_valid_count),
                "joint_valid_start_candidate_count": int(joint_valid_count),
                "best_start_map_clearance_m": (
                    None if not math.isfinite(best_map_gap) else best_map_gap
                ),
                "best_start_depth_clearance_m": (
                    None if not math.isfinite(best_depth_gap) else best_depth_gap
                ),
                "goal_depth_connector_ok": bool(goal_connector_ok),
                "goal_depth_connector_clearance_m": float(goal_connector_gap),
            }
            continue
        if components[start_cell] != components[goal]:
            audit["failure"] = "disconnected_oriented_centres"
            continue

        def edge_check(a, b):
            length = resolution * math.hypot(a[0]-b[0], a[1]-b[1])
            axy, bxy = _xy(a, origin, resolution), _xy(b, origin, resolution)
            if (min(clearance[a], clearance[b]) < required + length / 2
                    and not map_certificate(axy, bxy)[0]):
                return False
            return (min(depth_clearance[a], depth_clearance[b]) >= margin + length / 2
                    or depth_certificate(axy, bxy)[0])

        # Pointwise spare room is a soft preference as well as a hard bound.
        soft_cost = 3.0 * np.exp(-np.maximum(depth_clearance - margin, 0) / .15)
        if preferred_path_cost is not None:
            soft_cost = soft_cost + preferred_path_cost
        # Continuous polygon sweeps are too expensive to run for every edge
        # expanded by A* (hundreds of thousands of Python calls on a typical
        # house map).  Search the already exact fixed-yaw configuration space,
        # certify only the resulting route, and lazily forbid any bad edge.
        # The execution polyline is certified again below, so this changes the
        # cost of finding candidates, not the collision standard.
        forbidden_edges: set[tuple[int, int]] = set()
        grid_path = None
        lazy_rejected_edges = 0
        lazy_search_rounds = 0
        try:
            for _round in range(DEPTH_REFINEMENT_LAZY_EDGE_REPLANS + 1):
                candidate_path, expanded = _astar(
                    allowed,
                    clearance,
                    start_cell,
                    goal,
                    resolution,
                    required,
                    float(map_only_plan["clearance_weight"]),
                    forbidden_edges=forbidden_edges,
                    progress_check=check_progress,
                    overlap_cost=soft_cost,
                )
                expanded_total += expanded
                lazy_search_rounds += 1
                rejected_this_round: set[tuple[int, int]] = set()
                for edge_start, edge_end in zip(
                    candidate_path[:-1], candidate_path[1:]
                ):
                    if edge_check(edge_start, edge_end):
                        continue
                    start_index = edge_start[0] * allowed.shape[1] + edge_start[1]
                    end_index = edge_end[0] * allowed.shape[1] + edge_end[1]
                    rejected_this_round.add(
                        (min(start_index, end_index), max(start_index, end_index))
                    )
                if not rejected_this_round:
                    grid_path = candidate_path
                    break
                forbidden_edges.update(rejected_this_round)
                lazy_rejected_edges += len(rejected_this_round)
                check_progress()
            if grid_path is None:
                raise NavigationPlanError(
                    "oriented route retained uncertified edges after bounded lazy replanning"
                )
        except NavigationPlanError as exc:
            audit["failure"] = "depth_connectivity"
            audit["depth_connectivity_error"] = str(exc)
            audit["lazy_search_rounds"] = lazy_search_rounds
            audit["lazy_rejected_edge_count"] = lazy_rejected_edges
            continue
        audit["lazy_search_rounds"] = lazy_search_rounds
        audit["lazy_rejected_edge_count"] = lazy_rejected_edges
        candidate_entries = (
            [(tuple(start_xy), start_cell)]
            + [(_xy(cell, origin, resolution), cell) for cell in grid_path]
            + [(goal_xy, goal)]
        )
        candidate_entries = [
            entry
            for entry_index, entry in enumerate(candidate_entries)
            if (
                entry_index == 0
                or math.dist(entry[0], candidate_entries[entry_index - 1][0])
                > 1e-7
            )
        ]
        candidates = [entry[0] for entry in candidate_entries]
        candidate_cells = [entry[1] for entry in candidate_entries]
        path = [candidates[0]]
        map_certificates, depth_certificates = [], []
        index = 0
        shortcut_prefilter_rejections = 0
        shortcut_continuous_checks = 0
        while index < len(candidates) - 1:
            accepted = None
            for end_index in range(len(candidates)-1, index, -1):
                end = candidates[end_index]
                if math.dist(path[-1], end) > float(map_only_plan["max_execution_segment_m"]):
                    continue
                if not _shortcut_preserves_preference(
                    path[-1],
                    end,
                    list(
                        zip(
                            candidates[index:end_index],
                            candidates[index + 1:end_index + 1],
                        )
                    ),
                    preferred_path_cost,
                    origin,
                    resolution,
                ):
                    continue
                # The fixed-yaw configuration-space mask already contains the
                # exact polygon, filled map cells, live-depth disks, and both
                # safety margins at every lattice centre. Use it only as a
                # cheap rejection filter before the continuous metric proof.
                # A raster false negative merely retains more A* waypoints;
                # every accepted shortcut still passes both sweeps below.
                if (
                    not (needs_egress and index == 0)
                    and _line_cells(
                        allowed,
                        candidate_cells[index],
                        candidate_cells[end_index],
                    )
                    is None
                ):
                    shortcut_prefilter_rejections += 1
                    continue
                shortcut_continuous_checks += 1
                valid_map, map_gap = map_certificate(path[-1], end)
                if not valid_map:
                    continue
                valid_depth, depth_gap = depth_certificate(path[-1], end)
                if valid_depth:
                    accepted = end_index, end, map_gap, depth_gap
                    break
            if accepted is None:
                break
            index, end, map_gap, depth_gap = accepted
            path.append(end)
            map_certificates.append(map_gap)
            depth_certificates.append(depth_gap)
        if index != len(candidates) - 1 or not map_certificates:
            audit["failure"] = "continuous_map_or_depth_certificate"
            continue
        audit["shortcut_prefilter_rejection_count"] = (
            shortcut_prefilter_rejections
        )
        audit["shortcut_continuous_check_count"] = shortcut_continuous_checks
        # Keep the camera and chassis substantially facing each executed
        # line. If the next chord differs by more than 45 degrees, certify an
        # in-place spin to the chord bearing before translating. This retains
        # limited holonomic motion while preventing long blind side/reverse
        # drives through a passage.
        execution_yaws: list[float] = []
        execution_depth_certificates: list[float] = []
        execution_rotation_clearances: list[float] = []
        execution_static_gaps: list[float] = []
        execution_static_rotation_clearances: list[float] = []
        orientation_constrained_segments: list[int] = []
        previous_yaw = pose[2]
        execution_valid = True
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(path[:-1], path[1:])
        ):
            chord_yaw = math.atan2(
                segment_end[1] - segment_start[1],
                segment_end[0] - segment_start[0],
            )
            chord_delta = (
                chord_yaw - previous_yaw + math.pi
            ) % (2 * math.pi) - math.pi
            if needs_egress and segment_index == 0:
                motion_yaw = pose[2]
            elif abs(chord_delta) > math.radians(45.0) + 1.0e-12:
                motion_yaw = chord_yaw
            else:
                motion_yaw = previous_yaw
            rotation_gap = float((convex_sweep_distances(
                points,
                polygon,
                segment_start,
                segment_start,
                previous_yaw,
                motion_yaw,
            ) - point_radii).min())
            static_rotation_gap = _oriented_static_sweep_clearance(
                polygon,
                segment_start,
                segment_start,
                previous_yaw,
                motion_yaw,
                blocked,
                origin,
                resolution,
                measure_limit_m=(
                    static_margin + EXECUTION_CERTIFICATE_RESERVE_M
                ),
            )
            rotation_required = (
                float(start_distances.min()) - 1.0e-9
                if needs_egress and segment_index == 0
                else margin
            )
            segment_ok, segment_gap = depth_certificate_at_yaw(
                segment_start,
                segment_end,
                motion_yaw,
                allow_start_egress=(needs_egress and segment_index == 0),
            )
            static_segment_gap = static_gap_at_yaw(
                segment_start, segment_end, motion_yaw
            )
            static_segment_ok = (
                static_segment_gap + 1.0e-9 >= static_margin
            )
            orientation_constrained = False
            if (
                (
                    rotation_gap < rotation_required
                    or static_rotation_gap + 1.0e-9 < static_margin
                    or not segment_ok
                    or not static_segment_ok
                )
                and not (needs_egress and segment_index == 0)
                and abs((yaw - motion_yaw + math.pi) % (2 * math.pi) - math.pi)
                > 1.0e-8
            ):
                # A narrow passage can physically require the base's narrow
                # caliper to remain across the line of travel. Preserve that
                # exact orientation only when both its rotation and complete
                # translation sweep are certified by current depth.
                fallback_rotation_gap = float((convex_sweep_distances(
                    points,
                    polygon,
                    segment_start,
                    segment_start,
                    previous_yaw,
                    yaw,
                ) - point_radii).min())
                fallback_static_rotation_gap = _oriented_static_sweep_clearance(
                    polygon,
                    segment_start,
                    segment_start,
                    previous_yaw,
                    yaw,
                    blocked,
                    origin,
                    resolution,
                    measure_limit_m=(
                        static_margin + EXECUTION_CERTIFICATE_RESERVE_M
                    ),
                )
                fallback_ok, fallback_gap = depth_certificate_at_yaw(
                    segment_start, segment_end, yaw,
                )
                fallback_static_segment_gap = static_gap_at_yaw(
                    segment_start, segment_end, yaw
                )
                if (
                    fallback_rotation_gap >= margin
                    and fallback_static_rotation_gap + 1.0e-9 >= static_margin
                    and fallback_ok
                    and fallback_static_segment_gap + 1.0e-9 >= static_margin
                ):
                    motion_yaw = yaw
                    rotation_gap = fallback_rotation_gap
                    static_rotation_gap = fallback_static_rotation_gap
                    segment_gap = fallback_gap
                    static_segment_gap = fallback_static_segment_gap
                    segment_ok = True
                    static_segment_ok = True
                    orientation_constrained = True
            if (
                rotation_gap < rotation_required
                or static_rotation_gap + 1.0e-9 < static_margin
                or not segment_ok
                or not static_segment_ok
            ):
                execution_valid = False
                audit["execution_heading_failure"] = {
                    "segment_index": int(segment_index),
                    "chord_yaw_deg": math.degrees(chord_yaw),
                    "previous_yaw_deg": math.degrees(previous_yaw),
                    "motion_yaw_deg": math.degrees(motion_yaw),
                    "rotation_clearance_m": rotation_gap,
                    "rotation_required_m": rotation_required,
                    "segment_clearance_m": float(segment_gap),
                    "segment_required_m": margin,
                    "static_rotation_clearance_m": float(static_rotation_gap),
                    "static_rotation_required_m": static_margin,
                    "static_segment_clearance_m": float(static_segment_gap),
                    "static_segment_required_m": static_margin,
                }
                break
            execution_yaws.append(float(motion_yaw))
            execution_depth_certificates.append(float(segment_gap))
            execution_rotation_clearances.append(rotation_gap)
            execution_static_gaps.append(float(static_segment_gap))
            execution_static_rotation_clearances.append(
                float(static_rotation_gap)
            )
            if orientation_constrained:
                orientation_constrained_segments.append(segment_index)
            previous_yaw = motion_yaw
        if not execution_valid:
            audit["failure"] = "heading_aligned_depth_certificate"
            continue
        map_certificates = [
            radius + gap for gap in execution_static_gaps
        ]
        tight_margin_segment_indices = (
            [
                index
                for index, gap in enumerate(execution_depth_certificates)
                if gap + 1.0e-9
                < DEPTH_POINT_MARGIN_M + DEPTH_TRACKING_RESERVE_M
            ]
            if margin_policy == DEPTH_TIGHT_MARGIN_POLICY
            else []
        )
        orientation_constrained_segments = sorted(set(
            orientation_constrained_segments
        ).union(tight_margin_segment_indices))
        trials = list(map_only_plan["safety_margin_trials_m"])
        if START_EGRESS_SAFETY_MARGIN_M not in trials:
            continue
        result = dict(map_only_plan)
        result.update(
            path_xy_m=[list(point) for point in path], corner_path_xy_m=[list(point) for point in path],
            grid_path=[list(cell) for cell in grid_path], path_length_m=_polyline_length(path),
            path_segment_clearance_m=map_certificates,
            minimum_clearance_m=min(map_certificates), minimum_execution_clearance_m=min(map_certificates),
            mean_clearance_m=float(np.mean(map_certificates)), required_clearance_m=required,
            effective_safety_margin_m=START_EGRESS_SAFETY_MARGIN_M,
            safety_margin_trial_index=trials.index(START_EGRESS_SAFETY_MARGIN_M), safety_margin_relaxed=True,
            start_safe_xy_m=list(start_xy), start_snap_m=0.0, start_egress_segment_count=0,
            start_egress_required_clearance_m=required, expanded_cells=expanded_total,
            planning_resolution_m=resolution, downsample_factor=1,
            turning_count=int(sum(
                abs((current - previous + math.pi) % (2 * math.pi) - math.pi)
                > 1.0e-8
                for previous, current in zip(
                    [pose[2], *execution_yaws[:-1]], execution_yaws
                )
            )),
            turning_clearance_m=[], turning_repair_count=0,
            turning_clearance_reserve_m=0.0,
            planner="clearance_astar_oriented_static_and_live_depth",
            # Do not inherit the reference plan's motion-frame contract.  The
            # depth search is free to choose another certified path.
            traversed_route_direct_replay=False,
            polyline_validation="continuous_convex_static_map_and_depth_sweeps",
            route_sweep=route_sweep_audit(planning_snapshot, path),
            depth_refinement={
                "schema": DEPTH_REFINEMENT_SCHEMA, "polygon_sha256": polygon_digest,
                "segment_yaw_rad": execution_yaws,
                "point_margin_m": DEPTH_POINT_MARGIN_M, "tracking_reserve_m": tracking_reserve_m,
                "planning_margin_m": margin,
                "margin_policy": margin_policy,
                "rotation_clearance_m": min(execution_rotation_clearances),
                "segment_clearance_m": execution_depth_certificates,
                "point_count": original_point_count,
                "compressed_disk_count": len(points), "max_disk_radius_m": max_point_radius,
                "heading_tolerance_deg": 2.0, "max_speed_mps": .35,
                "heading_policy": "face_chord_over_45_unless_depth_constrained",
                "orientation_constrained_segment_indices": (
                    orientation_constrained_segments
                ),
                "tight_margin_segment_indices": (
                    tight_margin_segment_indices
                ),
                "orientation_constrained_max_speed_mps": .15,
                "refresh_depth_after_alignment_deg": 45.0,
                "static_map": {
                    "schema": STATIC_FOOTPRINT_SCHEMA,
                    "polygon_sha256": polygon_digest,
                    "planning_margin_m": static_margin,
                    "unknown_space_blocked": True,
                    "grid_cell_obstacles_filled": True,
                    "segment_clearance_m": execution_static_gaps,
                    "rotation_clearance_m": min(
                        execution_static_rotation_clearances
                    ),
                    "equivalent_center_clearance_m": map_certificates,
                },
            },
        )
        if map_evidence_audit is not None:
            result["depth_refinement"]["map_evidence"] = map_evidence_audit
        result["start"] = dict(result["start"], safe_grid_xy_m=list(start_xy))
        result["snap"] = dict(result["snap"], start={"applied": False, "distance_m": 0.0, "egress_certified": True})
        if needs_egress:
            result["depth_refinement"]["start_egress"] = {
                "schema": "monotone_convex_depth_egress_v1", "segment_count": 1,
                "physical_gap_floor_m": DEPTH_EGRESS_GAP_M,
                "initial_clearance_m": float(start_distances.min()),
                "fixed_yaw_rad": execution_yaws[0], "separating": True,
            }
        return result
    if diagnostics is not None and orientation_layers:
        diagnostics.append({
            "mode": "variable_orientation_union",
            "heading_count": len(orientation_layers),
            "start_candidate_count": endpoint_component_count(np.logical_or.reduce([
                layer["allowed"] for layer in orientation_layers
            ])),
        })
    if _lattice_resolution_m > .011 and time.monotonic() < search_deadline:
        return plan_depth_refined_path(
            snapshot, map_only_plan, polygon, polygon_digest,
            progress_check=progress_check, diagnostics=diagnostics,
            _lattice_resolution_m=.01001, _search_deadline=search_deadline,
        )
    return None


__all__ = [
    "DEFAULT_ARRIVAL_TOLERANCE_M",
    "DEFAULT_CLEARANCE_WEIGHT",
    "DEFAULT_MAX_SEGMENT_M",
    "DEFAULT_ROBOT_RADIUS_M",
    "DEFAULT_SAFETY_MARGIN_M",
    "NAVIGATION_PLAN_SCHEMA",
    "NAVIGATION_SNAPSHOT_SCHEMA",
    "STATIC_FOOTPRINT_SCHEMA",
    "TRAVERSED_THIN_BARRIER_SCHEMA",
    "NavigationPlanError",
    "apply_traversed_thin_barrier_override",
    "bind_traversed_thin_barrier_segments",
    "detect_traversed_thin_barrier",
    "plan_clearance_path",
    "restore_traversed_thin_barrier_observation",
]
