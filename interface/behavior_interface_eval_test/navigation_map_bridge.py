"""Backend-neutral, evaluator-safe navigation map snapshots.

The bridge deliberately has no dependency on ``behavior_interface``.  A map
backend may implement :meth:`export_navigation_map_snapshot`, or the policy
runtime may pass one of the two existing live mapper objects.  Compatibility
with those objects is implemented by reading their already-produced NumPy
state through a small duck-typed boundary; no simulator or scene API is used.

Coordinates follow one contract throughout this module:

* array row is map ``y`` and column is map ``x``;
* the centre of cell ``(row, column)`` is
  ``origin + (column + .5, row + .5) * resolution``;
* occupancy is ``-1`` unknown, ``0`` observed free, ``100`` blocked;
* pose and place coordinates are absolute in the snapshot's map frame.

``pose_observed_*`` records when this bridge first saw a producer pose
revision.  It is deliberately not described as the backend's capture time:
the existing asynchronous mapper interfaces do not expose that timestamp.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import math
import threading
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


NAVIGATION_MAP_SCHEMA = "behavior.official.navigation_map.v1"
NAVIGATION_MAP_SCHEMA_VERSION = 1
ORDERED_POSE_PRODUCER_SCHEMA = (
    "behavior.official.navigation_map.ordered_pose_producer.v1"
)
UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(100)
MAX_TRAVERSED_PATH_POINTS = 250_000
RTAB_TRAIL_MAX_LINK_M = 0.75


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _strict_bool(value: Any, field: str, *, default: bool) -> bool:
    """Normalize safety state without treating non-empty strings as true."""

    if value is None:
        return bool(default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be boolean")
    return bool(value)


def _read_only_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or not array.size:
        raise ValueError("navigation map layers must be non-empty 2-D arrays")
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"navigation map layer shape {array.shape} does not match {shape}"
        )
    # A write-disabled owning ndarray can make itself writable again.  Use an
    # immutable bytes buffer as the ultimate owner so neither a published view
    # nor anything reachable through its ``base`` chain can re-enable writes.
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.dtype(dtype)
    ).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _is_irreversibly_read_only(value: Any) -> bool:
    """Return whether an ndarray ultimately points at an immutable buffer."""

    if not isinstance(value, np.ndarray) or value.flags.writeable:
        return False
    owner: Any = value
    seen: set[int] = set()
    while isinstance(owner, np.ndarray):
        if owner.flags.writeable or id(owner) in seen:
            return False
        seen.add(id(owner))
        owner = owner.base
        if owner is None:
            return False
    return isinstance(owner, bytes)


def _optional_mask(
    layers: Mapping[str, Any],
    names: tuple[str, ...],
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    for name in names:
        value = layers.get(name)
        if value is not None:
            array = np.asarray(value, dtype=np.bool_)
            if array.shape != shape:
                raise ValueError(
                    f"navigation map layer {name!r} shape {array.shape} "
                    f"does not match {shape}"
                )
            return array
    return None


def _normalise_traversed_paths(
    value: Any,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return immutable, finite map-frame paths supplied by the mapper."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("traversed_paths_xy_m must be a sequence of paths")
    paths: list[tuple[tuple[float, float], ...]] = []
    point_count = 0
    for path_index, raw_path in enumerate(value):
        if not isinstance(raw_path, Sequence) or isinstance(
            raw_path, (str, bytes)
        ):
            raise ValueError(
                f"traversed path {path_index} must be a sequence of points"
            )
        path: list[tuple[float, float]] = []
        for point_index, raw_point in enumerate(raw_path):
            try:
                point = np.asarray(raw_point, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"traversed path {path_index} point {point_index} is invalid"
                ) from exc
            if point.size != 2 or not np.all(np.isfinite(point)):
                raise ValueError(
                    f"traversed path {path_index} point {point_index} must "
                    "contain finite x and y"
                )
            xy = (float(point[0]), float(point[1]))
            if path and math.dist(path[-1], xy) <= 1.0e-9:
                continue
            path.append(xy)
            point_count += 1
            if point_count > MAX_TRAVERSED_PATH_POINTS:
                raise ValueError("traversed path evidence exceeds the point limit")
        if path:
            paths.append(tuple(path))
    return tuple(paths)


def _rtab_traversed_paths(mapper: Any, result: Any) -> tuple[
    tuple[tuple[float, float], ...], ...
]:
    """Read the mapper's already-resolved blue trail without importing it."""

    display_result = getattr(mapper, "_display_result", None)
    if not callable(display_result):
        return ()
    try:
        displayed = display_result(False)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ()
    records = tuple(getattr(displayed, "poses", ())) if displayed else ()
    if not records:
        return ()
    resolution = float(getattr(result, "cell_size_m", 0.05))
    max_link_m = max(RTAB_TRAIL_MAX_LINK_M, 8.0 * resolution)
    paths: list[tuple[tuple[float, float], ...]] = []
    current: list[tuple[float, float]] = []
    current_cell: Optional[tuple[int, int]] = None
    previous: Optional[tuple[float, float]] = None
    for record in records:
        point = (
            float(getattr(record, "x_m")),
            float(getattr(record, "y_m")),
        )
        if not all(math.isfinite(value) for value in point):
            continue
        if previous is not None and math.dist(previous, point) > max_link_m:
            if current:
                paths.append(tuple(current))
            current = []
            current_cell = None
        cell = (
            int(math.floor(point[1] / resolution)),
            int(math.floor(point[0] / resolution)),
        )
        if not current:
            current.append(point)
        elif cell == current_cell:
            current[-1] = point
        else:
            current.append(point)
        current_cell = cell
        previous = point
    if current:
        paths.append(tuple(current))
    return tuple(paths)


def _rtab_graph_poses(
    result: Any,
) -> Optional[tuple[tuple[int, float, float, float], ...]]:
    """Return an exact, deterministic RTAB pose-graph snapshot."""

    records = getattr(result, "poses", None)
    if records is None:
        return None
    graph: list[tuple[int, float, float, float]] = []
    try:
        for record in records:
            node_id = int(getattr(record, "node_id"))
            values = (
                float(getattr(record, "x_m")),
                float(getattr(record, "y_m")),
                float(getattr(record, "yaw_rad")),
            )
            if node_id <= 0 or not all(math.isfinite(value) for value in values):
                continue
            graph.append((node_id, *values))
    except (AttributeError, TypeError, ValueError):
        return None
    graph.sort(key=lambda item: item[0])
    return tuple(graph)


def _rtab_resolve_anchored_trail_point(
    item: Any,
    graph: Mapping[int, tuple[float, float, float]],
) -> Optional[tuple[float, float]]:
    """Resolve one duck-typed LiveMapper trail item without core imports."""

    try:
        anchor_node_id = getattr(item, "anchor_node_id")
        local_pose = getattr(item, "local_pose")
        fallback_pose = getattr(item, "fallback_pose")
        anchor = graph.get(anchor_node_id)
        if anchor is None:
            point = (
                float(getattr(fallback_pose, "x_m")),
                float(getattr(fallback_pose, "y_m")),
            )
        else:
            anchor_x, anchor_y, anchor_yaw = anchor
            local_x = float(getattr(local_pose, "x_m"))
            local_y = float(getattr(local_pose, "y_m"))
            cosine = math.cos(anchor_yaw)
            sine = math.sin(anchor_yaw)
            point = (
                anchor_x + cosine * local_x - sine * local_y,
                anchor_y + sine * local_x + cosine * local_y,
            )
    except (AttributeError, TypeError, ValueError):
        return None
    return point if all(math.isfinite(value) for value in point) else None


def _rtab_append_trail_point(
    paths: list[list[tuple[float, float]]],
    point: tuple[float, float],
    *,
    resolution: float,
    max_link_m: float,
) -> None:
    previous = paths[-1][-1] if paths and paths[-1] else None
    if previous is not None and math.dist(previous, point) > max_link_m:
        paths.append([])
        previous = None
    if not paths:
        paths.append([])
    current = paths[-1]
    cell = (
        int(math.floor(point[1] / resolution)),
        int(math.floor(point[0] / resolution)),
    )
    previous_cell = (
        (
            int(math.floor(previous[1] / resolution)),
            int(math.floor(previous[0] / resolution)),
        )
        if previous is not None
        else None
    )
    if not current:
        current.append(point)
    elif cell == previous_cell:
        current[-1] = point
    else:
        current.append(point)


def _rtab_traversed_paths_incremental(
    mapper: Any,
    result: Any,
    cache: Optional[Mapping[str, Any]],
) -> tuple[
    tuple[tuple[tuple[float, float], ...], ...],
    Optional[dict[str, Any]],
]:
    """Resolve only appended trail poses while the pose graph is unchanged."""

    trail = getattr(mapper, "_trail", None)
    graph_signature = _rtab_graph_poses(result)
    if not isinstance(trail, list) or graph_signature is None:
        return _rtab_traversed_paths(mapper, result), None
    resolution = float(getattr(result, "cell_size_m", 0.05))
    if not math.isfinite(resolution) or resolution <= 0.0:
        return _rtab_traversed_paths(mapper, result), None
    graph = {
        node_id: (x_m, y_m, yaw_rad)
        for node_id, x_m, y_m, yaw_rad in graph_signature
    }
    reusable = bool(
        cache is not None
        and cache.get("mapper") is mapper
        and cache.get("trail") is trail
        and cache.get("graph_signature") == graph_signature
        and cache.get("resolution") == resolution
        and isinstance(cache.get("processed"), int)
        and 0 <= int(cache["processed"]) <= len(trail)
        and isinstance(cache.get("paths"), list)
    )
    if reusable:
        processed = int(cache["processed"])
        paths = cache["paths"]
    else:
        processed = 0
        paths = []
    max_link_m = max(RTAB_TRAIL_MAX_LINK_M, 8.0 * resolution)
    for item in trail[processed:]:
        point = _rtab_resolve_anchored_trail_point(item, graph)
        if point is None:
            # Unknown trail layouts are compatibility-provider territory.
            # Fall back to its own resolver instead of publishing partial data.
            return _rtab_traversed_paths(mapper, result), None
        _rtab_append_trail_point(
            paths,
            point,
            resolution=resolution,
            max_link_m=max_link_m,
        )
    updated_cache = {
        "mapper": mapper,
        "trail": trail,
        "graph_signature": graph_signature,
        "resolution": resolution,
        "processed": len(trail),
        "paths": paths,
    }
    return (
        tuple(tuple(path) for path in paths if path),
        updated_cache,
    )


def _normalise_grid(
    raw_occupancy: Any,
    raw_layers: Optional[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(raw_occupancy)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("occupancy must be a non-empty 2-D array")
    shape = (int(raw.shape[0]), int(raw.shape[1]))
    layers = dict(raw_layers or {})

    occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
    if raw.dtype == np.bool_:
        occupancy[~raw] = FREE
        occupancy[raw] = OCCUPIED
    else:
        finite = np.isfinite(raw)
        occupancy[finite & (raw == 0)] = FREE
        # ROS-style occupancy uses 1..100.  Values below 50 remain unknown so
        # uncertain evidence never silently becomes traversable.
        occupancy[finite & (raw >= 50)] = OCCUPIED

    explicit_free = _optional_mask(
        layers,
        ("free", "traversable", "walkable"),
        shape,
    )
    low = _optional_mask(
        layers,
        ("low_obstacle", "low_obstacles", "base_obstacle"),
        shape,
    )
    high = _optional_mask(
        layers,
        ("high_obstacle", "high_obstacles", "overhead"),
        shape,
    )
    obstacle = _optional_mask(
        layers,
        ("obstacle", "obstacles", "blocked"),
        shape,
    )
    wall = _optional_mask(layers, ("wall", "walls"), shape)

    low = np.zeros(shape, dtype=np.bool_) if low is None else low
    high = np.zeros(shape, dtype=np.bool_) if high is None else high
    obstacle = np.zeros(shape, dtype=np.bool_) if obstacle is None else obstacle
    wall = np.zeros(shape, dtype=np.bool_) if wall is None else wall
    blocked = (occupancy == OCCUPIED) | low | obstacle | wall

    # A high-only return may be a table top or another structure above the
    # chassis.  It is not labelled as a base obstacle, but it also must not
    # promote a cell to known free without corroborating low-height clearance.
    occupancy[(occupancy == FREE) & high] = UNKNOWN
    if explicit_free is not None:
        occupancy[(occupancy == FREE) & ~explicit_free] = UNKNOWN
        occupancy[explicit_free & ~blocked & ~high] = FREE
    occupancy[blocked] = OCCUPIED

    canonical_layers = {
        "free": occupancy == FREE,
        "obstacle": occupancy == OCCUPIED,
        "wall": wall & (occupancy == OCCUPIED),
        "low_obstacle": low,
        "high_obstacle": high,
    }
    frozen_occupancy = _read_only_array(occupancy, dtype=np.int8)
    frozen_layers = {
        name: _read_only_array(value, dtype=np.bool_, shape=shape)
        for name, value in canonical_layers.items()
    }
    return frozen_occupancy, frozen_layers


def _normalise_pose(value: Mapping[str, Any]) -> dict[str, Any]:
    pose = dict(value or {})
    if "yaw_deg" in pose:
        yaw_deg = _finite_float(pose["yaw_deg"], "pose.yaw_deg")
    elif "yaw_rad" in pose:
        yaw_deg = math.degrees(
            _finite_float(pose["yaw_rad"], "pose.yaw_rad")
        )
    else:
        raise ValueError("pose requires yaw_deg or yaw_rad")
    return {
        "x": _finite_float(pose.get("x"), "pose.x"),
        "y": _finite_float(pose.get("y"), "pose.y"),
        "yaw_deg": yaw_deg,
        "frame": str(pose.get("frame") or "map").strip(),
        "global_confident": _strict_bool(
            pose.get("global_confident"),
            "pose.global_confident",
            default=False,
        ),
        "source": str(pose.get("source") or "map_backend"),
    }


def _normalise_policy_pose(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        pose = dict(value)
        array = np.asarray(
            [pose.get("x"), pose.get("y"), pose.get("yaw_rad")],
            dtype=np.float64,
        )
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        raise ValueError("policy_local_pose must contain finite x, y, yaw_rad")
    return {
        "x": float(array[0]),
        "y": float(array[1]),
        "yaw_rad": float(array[2]),
        "frame": "policy_local_odometry",
    }


def normalize_navigation_pose_freshness(
    snapshot: Mapping[str, Any],
    *,
    pose_observed_sequence: Optional[int] = None,
    pose_observed_ts: Optional[float] = None,
) -> dict[str, Any]:
    """Validate pose first-observation metadata and derive its current age."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("navigation map snapshot must be a mapping")
    payload = dict(snapshot)
    observation_sequence = int(payload.get("observation_sequence") or 0)
    if observation_sequence < 0:
        raise ValueError("observation_sequence must be non-negative")
    captured_ts = _finite_float(
        payload.get("captured_ts", time.time()), "captured_ts"
    )
    pose_version = str(payload.get("pose_version") or "").strip()
    if not pose_version:
        raise ValueError("navigation map snapshot requires pose_version")

    first_sequence_raw = (
        pose_observed_sequence
        if pose_observed_sequence is not None
        else payload.get("pose_observed_sequence")
    )
    first_sequence = (
        observation_sequence
        if first_sequence_raw is None
        else int(first_sequence_raw)
    )
    if first_sequence < 0 or first_sequence > observation_sequence:
        raise ValueError(
            "pose_observed_sequence must be between zero and the current "
            "observation_sequence"
        )

    first_ts_raw = (
        pose_observed_ts
        if pose_observed_ts is not None
        else payload.get("pose_observed_ts")
    )
    first_ts = (
        captured_ts
        if first_ts_raw is None
        else _finite_float(first_ts_raw, "pose_observed_ts")
    )
    if first_ts > captured_ts:
        raise ValueError("pose_observed_ts cannot be after captured_ts")
    return {
        "pose_version": pose_version,
        "pose_observed_sequence": first_sequence,
        "pose_observed_ts": first_ts,
        "pose_age_observations": observation_sequence - first_sequence,
        "pose_age_s": max(0.0, captured_ts - first_ts),
    }


def normalize_navigation_pose_source(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the optional exact evaluator observation behind the pose."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("navigation map snapshot must be a mapping")
    payload = dict(snapshot)
    lifecycle = dict(payload.get("lifecycle") or {})
    known_raw = payload.get(
        "pose_source_sequence_known",
        lifecycle.get("pose_source_sequence_known", False),
    )
    if not isinstance(known_raw, (bool, np.bool_)):
        raise ValueError("pose_source_sequence_known must be a boolean")
    known = bool(known_raw)
    output: dict[str, Any] = {"pose_source_sequence_known": known}
    if not known:
        return output
    source_sequence = payload.get("pose_source_observation_sequence")
    if (
        isinstance(source_sequence, (bool, np.bool_))
        or not isinstance(source_sequence, (int, np.integer))
    ):
        raise ValueError(
            "pose_source_observation_sequence must be an integer when known"
        )
    source_sequence = int(source_sequence)
    observation_sequence = int(payload.get("observation_sequence") or 0)
    if source_sequence < 0 or source_sequence > observation_sequence:
        raise ValueError(
            "pose_source_observation_sequence must be between zero and the "
            "current observation_sequence"
        )
    output["pose_source_observation_sequence"] = source_sequence
    return output


def normalize_ordered_pose_producer(
    snapshot: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Validate optional FIFO producer progress used for causal pose waits.

    A processed frame covers every earlier enqueued frame only when the
    producer promises FIFO execution and its published pose belongs to the
    latest processed frame.  The RTAB live adapter can prove that relationship
    while holding both of the worker's state locks.
    """

    if not isinstance(snapshot, Mapping):
        raise TypeError("navigation map snapshot must be a mapping")
    source_frame = snapshot.get("source_frame")
    if source_frame is None:
        return None
    if not isinstance(source_frame, Mapping):
        raise ValueError("source_frame must be a mapping")
    raw = source_frame.get("ordered_pose_producer")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("ordered_pose_producer must be a mapping")
    if str(raw.get("schema") or "") != ORDERED_POSE_PRODUCER_SCHEMA:
        raise ValueError("ordered_pose_producer has an unsupported schema")

    values: dict[str, int] = {}
    for field in (
        "enqueued_frame_count",
        "processed_frame_count",
        "pose_frame_id",
        "last_enqueued_observation_sequence",
    ):
        value = raw.get(field)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"ordered_pose_producer.{field} must be non-negative")
        values[field] = int(value)

    enqueued = values["enqueued_frame_count"]
    processed = values["processed_frame_count"]
    pose_frame = values["pose_frame_id"]
    source_sequence = values["last_enqueued_observation_sequence"]
    observation_sequence = int(snapshot.get("observation_sequence") or 0)
    if processed > enqueued:
        raise ValueError(
            "ordered_pose_producer processed count exceeds enqueued count"
        )
    if pose_frame != processed:
        raise ValueError(
            "ordered_pose_producer pose frame is not the latest processed frame"
        )
    if source_sequence > observation_sequence:
        raise ValueError(
            "ordered_pose_producer source sequence exceeds the current observation"
        )
    return {
        "schema": ORDERED_POSE_PRODUCER_SCHEMA,
        **values,
    }


def _normalise_places(value: Any) -> list[dict[str, Any]]:
    places: list[dict[str, Any]] = []
    for raw in list(value or []):
        if not isinstance(raw, Mapping):
            raise ValueError("every navigation map place must be a mapping")
        item = dict(raw)
        name = str(item.get("name") or item.get("label") or "").strip()
        if not name:
            raise ValueError("navigation map places require a name")
        place = {
            "name": name,
            "x": _finite_float(item.get("x"), f"place {name}.x"),
            "y": _finite_float(item.get("y"), f"place {name}.y"),
        }
        for key in ("source", "image_id", "kind"):
            if item.get(key) not in (None, ""):
                place[key] = str(item[key])
        if item.get("count") is not None:
            place["count"] = int(item["count"])
        places.append(place)
    return places


def _normalise_observation_anchors(value: Any) -> list[dict[str, Any]]:
    """Validate sparse, evaluator-observation poses in the map frame.

    These are correspondence records, not an additional pose source.  The
    matching policy-owned odometry pose remains in the saved observation
    metadata and is joined by ``image_id`` only when navigation needs it.
    """

    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in list(value or []):
        if not isinstance(raw, Mapping):
            raise ValueError("every navigation observation anchor must be a mapping")
        item = dict(raw)
        image_id = str(item.get("image_id") or "").strip()
        session_id = str(item.get("session_id") or "").strip()
        if not image_id or not session_id:
            raise ValueError(
                "navigation observation anchors require session_id and image_id"
            )
        identity = (session_id, image_id)
        if identity in seen:
            raise ValueError("navigation observation anchors must be unique")
        seen.add(identity)
        anchor = {
            "session_id": session_id,
            "image_id": image_id,
            "x": _finite_float(item.get("x"), f"anchor {image_id}.x"),
            "y": _finite_float(item.get("y"), f"anchor {image_id}.y"),
            "yaw_rad": _finite_float(
                item.get("yaw_rad"), f"anchor {image_id}.yaw_rad"
            ),
        }
        source_sequence = item.get("source_sequence")
        if source_sequence is not None:
            if (
                isinstance(source_sequence, (bool, np.bool_))
                or not isinstance(source_sequence, (int, np.integer))
                or int(source_sequence) < 0
            ):
                raise ValueError(
                    f"anchor {image_id}.source_sequence must be non-negative"
                )
            anchor["source_sequence"] = int(source_sequence)
        anchors.append(anchor)
    anchors.sort(
        key=lambda item: (
            item.get("source_sequence", 2**63 - 1),
            item["session_id"],
            item["image_id"],
        )
    )
    return anchors


def normalize_navigation_map_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and take an immutable copy of a provider snapshot."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("navigation map snapshot must be a mapping")
    payload = dict(snapshot)
    occupancy, layers = _normalise_grid(
        payload.get("occupancy"),
        payload.get("layers"),
    )
    origin = np.asarray(payload.get("origin"), dtype=np.float64).reshape(-1)
    if origin.size != 2 or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain finite x_min_m and y_min_m")
    resolution = _finite_float(payload.get("resolution"), "resolution")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    pose = _normalise_pose(payload.get("pose") or {})
    lifecycle = deepcopy(dict(payload.get("lifecycle") or {}))
    pose_confident = _strict_bool(
        lifecycle.get("pose_confident"),
        "lifecycle.pose_confident",
        default=pose["global_confident"],
    )
    pose["global_confident"] = pose_confident
    lifecycle["pose_confident"] = pose_confident
    pose_source = normalize_navigation_pose_source(payload)
    lifecycle["pose_source_sequence_known"] = pose_source[
        "pose_source_sequence_known"
    ]
    lifecycle["source_lag_known"] = pose_source[
        "pose_source_sequence_known"
    ]

    episode_id = str(payload.get("episode_id") or "").strip()
    map_version = str(payload.get("map_version") or "").strip()
    if not episode_id:
        raise ValueError("navigation map snapshot requires episode_id")
    if not map_version:
        raise ValueError("navigation map snapshot requires map_version")

    observation_sequence = int(payload.get("observation_sequence") or 0)
    captured_ts = _finite_float(
        payload.get("captured_ts", time.time()), "captured_ts"
    )
    freshness_payload = dict(payload)
    freshness_payload["observation_sequence"] = observation_sequence
    freshness_payload["captured_ts"] = captured_ts
    pose_freshness = normalize_navigation_pose_freshness(freshness_payload)

    frame = str(payload.get("frame") or pose["frame"]).strip()
    if not frame:
        raise ValueError("navigation map frame must be non-empty")
    if frame != pose["frame"]:
        raise ValueError(
            "navigation map frame must match pose.frame "
            f"({frame!r} != {pose['frame']!r})"
        )

    output: dict[str, Any] = {
        "schema": NAVIGATION_MAP_SCHEMA,
        "schema_version": NAVIGATION_MAP_SCHEMA_VERSION,
        "episode_id": episode_id,
        # map_epoch is the stable token.  map_version may change whenever
        # occupancy or landmarks change and must never be used as an episode
        # identity by a long-running navigation action.
        "map_epoch": str(payload.get("map_epoch") or episode_id),
        "map_version": map_version,
        "frame_version": str(
            payload.get("frame_version") or map_version
        ),
        "backend": str(payload.get("backend") or "external"),
        "build": str(payload.get("build") or ""),
        "frame": frame,
        "observation_sequence": observation_sequence,
        "captured_ts": captured_ts,
        "origin": [float(origin[0]), float(origin[1])],
        "resolution": resolution,
        "shape": [int(occupancy.shape[0]), int(occupancy.shape[1])],
        "occupancy": occupancy,
        "layers": layers,
        "free_mask": layers["free"],
        "obstacle_mask": layers["obstacle"],
        "wall_mask": layers["wall"],
        "pose": pose,
        "places": _normalise_places(payload.get("places")),
        "lifecycle": lifecycle,
        **pose_source,
        **pose_freshness,
    }
    output["traversed_paths_xy_m"] = _normalise_traversed_paths(
        payload.get("traversed_paths_xy_m")
    )
    output["observation_anchors"] = _normalise_observation_anchors(
        payload.get("observation_anchors")
    )
    policy_pose = _normalise_policy_pose(payload.get("policy_local_pose"))
    if policy_pose is not None:
        output["policy_local_pose"] = policy_pose
    for key in (
        "provider_map_epoch",
        "provider_map_version",
        "provider_frame_version",
        "provider_pose_version",
    ):
        if payload.get(key) is not None:
            output[key] = str(payload[key])
    if payload.get("source_frame") is not None:
        source_frame = payload["source_frame"]
        if not isinstance(source_frame, Mapping):
            raise ValueError("source_frame must be a mapping")
        output["source_frame"] = deepcopy(dict(source_frame))
        ordered_producer = normalize_ordered_pose_producer(payload)
        if ordered_producer is not None:
            output["source_frame"]["ordered_pose_producer"] = ordered_producer
    return output


def copy_navigation_map_snapshot(
    snapshot: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return a complete defensive copy, including every grid layer."""

    if snapshot is None:
        return None
    return normalize_navigation_map_snapshot(snapshot)


def view_navigation_map_snapshot(
    snapshot: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return new containers with read-only views of an immutable grid.

    This is the adapter hot-read path.  The adapter owns canonical arrays whose
    write flag is already disabled.  A view of such an array cannot enable its
    own write flag, while avoiding several full-grid copies per navigation
    segment.  Non-array values still receive defensive container copies.
    """

    if snapshot is None:
        return None
    payload = dict(snapshot)
    occupancy = payload.get("occupancy")
    layers = payload.get("layers")
    canonical = (
        payload.get("schema") == NAVIGATION_MAP_SCHEMA
        and payload.get("schema_version") == NAVIGATION_MAP_SCHEMA_VERSION
        and _is_irreversibly_read_only(occupancy)
        and isinstance(layers, Mapping)
        and {"free", "obstacle", "wall"}.issubset(layers)
        and all(_is_irreversibly_read_only(value) for value in layers.values())
    )
    if not canonical:
        payload = normalize_navigation_map_snapshot(payload)
        occupancy = payload["occupancy"]
        layers = payload["layers"]

    occupancy_view = occupancy.view()
    occupancy_view.setflags(write=False)
    layer_views: dict[str, np.ndarray] = {}
    for name, value in layers.items():
        layer_view = value.view()
        layer_view.setflags(write=False)
        layer_views[str(name)] = layer_view

    output = dict(payload)
    output["occupancy"] = occupancy_view
    output["layers"] = layer_views
    output["free_mask"] = layer_views["free"]
    output["obstacle_mask"] = layer_views["obstacle"]
    output["wall_mask"] = layer_views["wall"]
    output["origin"] = list(payload.get("origin") or ())
    output["shape"] = list(payload.get("shape") or ())
    output["pose"] = dict(payload.get("pose") or {})
    output["places"] = [
        dict(item) for item in list(payload.get("places") or ())
    ]
    output["observation_anchors"] = [
        dict(item) for item in list(payload.get("observation_anchors") or ())
    ]
    output["lifecycle"] = deepcopy(dict(payload.get("lifecycle") or {}))
    if payload.get("policy_local_pose") is not None:
        output["policy_local_pose"] = dict(payload["policy_local_pose"])
    if payload.get("source_frame") is not None:
        output["source_frame"] = deepcopy(dict(payload["source_frame"]))
    return output


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _place_dict(value: Any) -> dict[str, Any]:
    return {
        "name": str(
            getattr(value, "name", "") or getattr(value, "label", "")
        ),
        "x": float(getattr(value, "x_m", getattr(value, "x", 0.0))),
        "y": float(getattr(value, "y_m", getattr(value, "y", 0.0))),
        "source": str(getattr(value, "source", "")),
        "image_id": str(getattr(value, "image_id", "")),
        "kind": str(getattr(value, "kind", "")),
        "count": int(getattr(value, "count", 1)),
    }


def _rtab_observation_anchors(mapper: Any, result: Any) -> list[dict[str, Any]]:
    """Resolve saved head-camera observations through the latest pose graph."""

    image_poses = getattr(mapper, "_image_poses", None)
    resolve = getattr(mapper, "_resolve_anchored_pose", None)
    session_id = str(getattr(mapper, "_session_id", "") or "").strip()
    if not isinstance(image_poses, Mapping) or not callable(resolve) or not session_id:
        return []
    anchors: list[dict[str, Any]] = []
    for image_id, value in tuple(image_poses.items()):
        key = str(image_id or "").strip()
        if not key:
            continue
        try:
            pose = resolve(
                result,
                getattr(value, "anchor_node_id"),
                getattr(value, "local_pose"),
                getattr(value, "fallback_pose"),
            )
            anchor = {
                "session_id": session_id,
                "image_id": key,
                "x": float(getattr(pose, "x_m")),
                "y": float(getattr(pose, "y_m")),
                "yaw_rad": float(getattr(pose, "yaw_rad")),
            }
            source_sequence = getattr(value, "source_sequence", None)
            if source_sequence is not None:
                anchor["source_sequence"] = int(source_sequence)
            anchors.append(anchor)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
    return _normalise_observation_anchors(anchors)


def _rtab_snapshot_raw(
    mapper: Any,
    *,
    cached_key: Optional[tuple[Any, ...]],
    trail_cache: Optional[Mapping[str, Any]],
    episode_id: str,
    map_epoch: str,
    observation_sequence: int,
    captured_ts: float,
    policy_local_pose: Any,
) -> tuple[
    Optional[dict[str, Any]],
    Optional[tuple[Any, ...]],
    Optional[dict[str, Any]],
]:
    input_lock = getattr(mapper, "_input_lock", None)
    state_lock = getattr(mapper, "_state_lock", None)
    with ExitStack() as stack:
        # LiveMapper writes enqueue sequence/count under _input_lock and the
        # accepted result/count under _state_lock.  This order is compatible
        # with map_tick, which releases _input_lock before touching state.
        input_locked = hasattr(input_lock, "__enter__")
        state_locked = hasattr(state_lock, "__enter__")
        if input_locked:
            stack.enter_context(input_lock)
        if state_lock is not input_lock and state_locked:
            stack.enter_context(state_lock)
        result = getattr(mapper, "_latest", None)
        if result is None:
            return None, None, None
        frame_id = int(getattr(result, "frame_id"))
        places_version = int(getattr(mapper, "_places_version", 0))
        loop_count = int(getattr(result, "loop_count", 0))
        node_count = int(getattr(result, "node_count", 0))
        map_updated = bool(getattr(result, "map_updated", True))
        trail_revision = len(getattr(mapper, "_trail", ()))
        image_pose_revision = len(getattr(mapper, "_image_poses", ()))
        same_stream = bool(
            cached_key is not None
            and len(cached_key) == 8
            and cached_key[0] == "rtabmap"
            and cached_key[1] == map_epoch
        )
        counters_changed = bool(
            not same_stream
            or int(cached_key[3]) != loop_count
            or int(cached_key[4]) != node_count
            or int(cached_key[5]) != places_version
            or int(cached_key[6]) != trail_revision
            or int(cached_key[7]) != image_pose_revision
        )
        geometry_frame = (
            frame_id
            if map_updated or counters_changed
            else int(cached_key[2])
        )
        map_version = (
            f"{map_epoch}|rtabmap-geometry:{geometry_frame}:{node_count}:"
            f"{loop_count}:{places_version}:{trail_revision}:{image_pose_revision}"
        )
        resolve_places = getattr(mapper, "_resolved_places", None)
        if callable(resolve_places):
            places = [
                _place_dict(item)
                for item in resolve_places(result, include_start=True)
            ]
        else:
            places = []
            start = getattr(mapper, "_start_place", None)
            if start is not None:
                places.append(_place_dict(start))
            places.extend(
                _place_dict(item)
                for item in tuple(getattr(mapper, "_places", ()))
            )
        pose = getattr(result, "current_pose")
        tracking_ok = bool(getattr(result, "tracking_ok", True))
        recovery_hold = bool(getattr(result, "recovery_hold", False))
        uncertain_hold = bool(getattr(result, "uncertain_hold", False))
        soft_localization = bool(
            getattr(result, "soft_localization", False)
        )
        read_only_match = bool(getattr(result, "read_only_match", False))
        pose_confident = tracking_ok and not recovery_hold and not uncertain_hold
        now_monotonic = time.monotonic()
        worker_started = getattr(
            mapper, "_worker_started_monotonic_s", None
        )
        worker_busy_s = (
            0.0
            if worker_started is None
            else max(0.0, now_monotonic - float(worker_started))
        )
        stall_warning_s = float(
            getattr(mapper, "worker_stall_warning_s", float("inf"))
        )
        worker_timed_out = bool(getattr(mapper, "worker_timed_out", False))
        worker_stalled = bool(
            worker_timed_out
            or (worker_started is not None and worker_busy_s >= stall_warning_s)
        )
        worker_disabled = bool(getattr(mapper, "disabled", False))
        frames_enqueued = getattr(mapper, "frames_enqueued", None)
        frames_processed = getattr(mapper, "frames_processed", None)
        last_evaluator_sequence = getattr(mapper, "_last_sequence", None)
        legacy_source_sequence_known = bool(
            input_locked
            and state_locked
            and isinstance(frames_enqueued, (int, np.integer))
            and isinstance(frames_processed, (int, np.integer))
            and not isinstance(frames_enqueued, (bool, np.bool_))
            and not isinstance(frames_processed, (bool, np.bool_))
            and int(frames_enqueued) > 0
            and int(frames_enqueued) == int(frames_processed)
            and int(frames_processed) == frame_id
            and isinstance(last_evaluator_sequence, (int, np.integer))
            and not isinstance(last_evaluator_sequence, (bool, np.bool_))
            and int(last_evaluator_sequence) >= 0
            and int(last_evaluator_sequence) <= observation_sequence
        )
        ordered_producer_known = bool(
            input_locked
            and state_locked
            and isinstance(frames_enqueued, (int, np.integer))
            and isinstance(frames_processed, (int, np.integer))
            and not isinstance(frames_enqueued, (bool, np.bool_))
            and not isinstance(frames_processed, (bool, np.bool_))
            and int(frames_enqueued) >= int(frames_processed) >= 0
            and int(frames_processed) == frame_id
            and isinstance(last_evaluator_sequence, (int, np.integer))
            and not isinstance(last_evaluator_sequence, (bool, np.bool_))
            and 0 <= int(last_evaluator_sequence) <= observation_sequence
        )
        # The accepted pose and its sequence are published under _state_lock.
        # A pending newer frame does not make the accepted frame's time unknown.
        processed_sequence = getattr(mapper, "_last_processed_sequence", None)
        processed_sequence_known = bool(
            ordered_producer_known
            and int(frames_processed) > 0
            and isinstance(processed_sequence, (int, np.integer))
            and not isinstance(processed_sequence, (bool, np.bool_))
            and 0 <= int(processed_sequence) <= int(last_evaluator_sequence)
        )
        source_sequence_known = bool(
            processed_sequence_known
            or (
                not hasattr(mapper, "_last_processed_sequence")
                and legacy_source_sequence_known
            )
        )
        cache_key = (
            "rtabmap",
            map_epoch,
            geometry_frame,
            loop_count,
            node_count,
            places_version,
            trail_revision,
            image_pose_revision,
        )
        frame_version = f"{map_epoch}|rtabmap-frame:{frame_id}"
        raw = {
            "episode_id": episode_id,
            "map_epoch": map_epoch,
            "map_version": map_version,
            "frame_version": frame_version,
            "pose_version": frame_version,
            "backend": _type_name(mapper),
            "build": str(getattr(mapper, "BUILD", "")),
            "frame": "rtabmap_contact_aware_q",
            "observation_sequence": observation_sequence,
            "captured_ts": captured_ts,
            "pose": {
                "x": float(getattr(pose, "x_m")),
                "y": float(getattr(pose, "y_m")),
                "yaw_rad": float(getattr(pose, "yaw_rad")),
                "frame": "rtabmap_contact_aware_q",
                "global_confident": pose_confident,
                "source": "rtabmap_current_pose",
            },
            "policy_local_pose": policy_local_pose,
            "source_frame": {
                "backend_frame_id": frame_id,
                "lag_known": source_sequence_known,
            },
            "pose_source_sequence_known": source_sequence_known,
            "places": places,
            "lifecycle": {
                "pose_confident": pose_confident,
                "tracking_ok": tracking_ok,
                "recovery_hold": recovery_hold,
                "uncertain_hold": uncertain_hold,
                "soft_localization": soft_localization,
                "read_only_match": read_only_match,
                "source_lag_known": source_sequence_known,
                "pose_source_sequence_known": source_sequence_known,
                "mapping_active": bool(
                    getattr(result, "mapping_active", True)
                ),
                "map_updated_this_frame": map_updated,
                "node_count": node_count,
                "loop_count": loop_count,
                "localized_this_frame": bool(
                    getattr(result, "localized", False)
                ),
                "visual_localized_this_frame": bool(
                    getattr(result, "visual_localized", False)
                ),
                "geometric_localized_this_frame": bool(
                    getattr(result, "geometric_localized", False)
                ),
                "worker_busy": worker_started is not None,
                "worker_busy_s": worker_busy_s,
                "worker_stalled": worker_stalled,
                "worker_timed_out": worker_timed_out,
                "worker_healthy": not worker_disabled and not worker_stalled,
            },
        }
        if ordered_producer_known:
            raw["source_frame"]["ordered_pose_producer"] = {
                "schema": ORDERED_POSE_PRODUCER_SCHEMA,
                "enqueued_frame_count": int(frames_enqueued),
                "processed_frame_count": int(frames_processed),
                "pose_frame_id": frame_id,
                "last_enqueued_observation_sequence": int(
                    last_evaluator_sequence
                ),
            }
        if source_sequence_known:
            source_sequence = int(
                processed_sequence
                if processed_sequence_known
                else last_evaluator_sequence
            )
            raw["pose_source_observation_sequence"] = source_sequence
            raw["source_frame"].update(
                {
                    "evaluator_sequence": source_sequence,
                    "lag_observations": observation_sequence - source_sequence,
                }
            )
        if cache_key != cached_key:
            # Resolving RTAB's anchored trail is O(total episode ticks).  Do
            # it once per producer revision, not once per evaluator tick.
            # Cached snapshots retain the immutable path tuple while only
            # pose/freshness metadata advances between RTAB worker results.
            (
                raw["traversed_paths_xy_m"],
                trail_cache,
            ) = _rtab_traversed_paths_incremental(
                mapper,
                result,
                trail_cache,
            )
            raw["observation_anchors"] = _rtab_observation_anchors(
                mapper, result
            )
            raw_occupancy = np.asarray(getattr(result, "occupancy"))
            raw_low = np.asarray(getattr(result, "low_obstacles"))
            raw.update(
                {
                    "origin": [
                        float(getattr(result, "x_min_m")),
                        float(getattr(result, "y_min_m")),
                    ],
                    "resolution": float(getattr(result, "cell_size_m")),
                    "occupancy": raw_occupancy,
                    "layers": {
                        "low_obstacle": raw_low,
                        "high_obstacle": np.asarray(
                            getattr(result, "high_obstacles")
                        ),
                        "wall": (
                            np.asarray(raw_low, dtype=np.bool_)
                            & (raw_occupancy >= 65)
                        ),
                    },
                }
            )
        return raw, cache_key, trail_cache


def _legacy_snapshot_raw(
    ego: Any,
    *,
    cached_key: Optional[tuple[Any, ...]],
    episode_id: str,
    map_epoch: str,
    grid_generation: int,
    observation_sequence: int,
    captured_ts: float,
    policy_local_pose: Any,
) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, ...]]]:
    grid = getattr(ego, "grid", None)
    if grid is None:
        return None, None
    places = [_place_dict(item) for item in tuple(getattr(ego, "places", ()))]
    start = dict(getattr(ego, "landmarks", {}) or {}).get("start")
    if start is not None:
        places.insert(0, _place_dict(start))
    grid_revision = int(getattr(grid, "_rev", getattr(grid, "frames", 0)))
    place_revision = tuple(
        (
            item["name"],
            round(float(item["x"]), 4),
            round(float(item["y"]), 4),
            int(item.get("count", 1)),
        )
        for item in places
    )
    mapping_state = str(getattr(ego, "mapping_state", "mapping"))
    localization_state = str(
        getattr(ego, "localization_state", mapping_state)
    )
    pose_confident = bool(getattr(ego, "initialized", False)) and (
        mapping_state == "mapping" or localization_state == "localized"
    )
    place_digest = hashlib.blake2b(
        repr(place_revision).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    map_version = (
        f"{map_epoch}|egomap-grid:{grid_generation}:"
        f"{grid_revision}:{place_digest}"
    )
    cache_key = (
        "egomap",
        map_epoch,
        id(ego),
        id(grid),
        grid_generation,
        grid_revision,
        place_revision,
    )
    pose_x = float(getattr(ego, "x", 0.0))
    pose_y = float(getattr(ego, "y", 0.0))
    pose_yaw_deg = float(getattr(ego, "yaw_deg", 0.0))
    frame_version = (
        f"{map_epoch}|egomap-pose:"
        f"{int(getattr(ego, 'update_count', 0))}:"
        f"{pose_x.hex()}:{pose_y.hex()}:{pose_yaw_deg.hex()}"
    )
    raw = {
        "episode_id": episode_id,
        "map_epoch": map_epoch,
        "map_version": map_version,
        "frame_version": frame_version,
        "pose_version": frame_version,
        "backend": _type_name(ego),
        "build": str(getattr(ego, "BUILD", "")),
        "frame": "episode_start_odometry",
        "observation_sequence": observation_sequence,
        "captured_ts": captured_ts,
        "pose": {
            "x": pose_x,
            "y": pose_y,
            "yaw_deg": pose_yaw_deg,
            "frame": "episode_start_odometry",
            "global_confident": pose_confident,
            "source": "egomap_live_pose",
        },
        "policy_local_pose": policy_local_pose,
        "source_frame": {
            "backend_update_count": int(getattr(ego, "update_count", 0)),
            "evaluator_sequence": observation_sequence,
            "lag_observations": 0,
            "lag_known": True,
        },
        "pose_source_sequence_known": True,
        "pose_source_observation_sequence": observation_sequence,
        "places": places,
        "lifecycle": {
            "pose_confident": pose_confident,
            "tracking_ok": pose_confident,
            "recovery_hold": not pose_confident,
            "mapping_active": mapping_state == "mapping",
            "mapping_state": mapping_state,
            "localization_state": localization_state,
            "source_lag_known": True,
            "pose_source_sequence_known": True,
        },
    }
    if cache_key != cached_key:
        free = np.asarray(grid.observed_free_mask(), dtype=np.bool_)
        obstacle = np.asarray(
            grid.observed_occupied_mask(), dtype=np.bool_
        )
        wall = np.asarray(grid.observed_wall_mask(), dtype=np.bool_)
        overhead_provider = getattr(grid, "observed_overhead_mask", None)
        high = (
            np.asarray(overhead_provider(), dtype=np.bool_)
            if callable(overhead_provider)
            else np.zeros_like(free)
        )
        occupancy = np.full(free.shape, UNKNOWN, dtype=np.int8)
        occupancy[free] = FREE
        occupancy[obstacle] = OCCUPIED
        raw.update(
            {
                "origin": [
                    -float(getattr(grid, "half_span_m")),
                    -float(getattr(grid, "half_span_m")),
                ],
                "resolution": float(getattr(grid, "resolution_m")),
                "occupancy": occupancy,
                "layers": {
                    "free": free,
                    "obstacle": obstacle,
                    "wall": wall,
                    "low_obstacle": obstacle,
                    "high_obstacle": high,
                },
            }
        )
    return raw, cache_key


def _external_provider_header(
    snapshot: Any,
) -> tuple[dict[str, Any], str, str, str]:
    """Validate the versioned plugin boundary before evaluator metadata wins."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("external navigation map snapshot must be a mapping")
    payload = dict(snapshot)
    if payload.get("schema") != NAVIGATION_MAP_SCHEMA:
        raise ValueError(
            "external navigation map provider requires schema "
            f"{NAVIGATION_MAP_SCHEMA!r}"
        )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, (bool, np.bool_))
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != NAVIGATION_MAP_SCHEMA_VERSION
    ):
        raise ValueError(
            "external navigation map provider requires schema_version "
            f"{NAVIGATION_MAP_SCHEMA_VERSION}"
        )
    provider_epoch = str(payload.get("map_epoch") or "").strip()
    provider_version = str(payload.get("map_version") or "").strip()
    if not provider_epoch:
        raise ValueError("external navigation map provider requires map_epoch")
    if not provider_version:
        raise ValueError("external navigation map provider requires map_version")
    provider_frame_version = str(
        payload.get("frame_version") or ""
    ).strip()
    provider_pose_version = str(payload.get("pose_version") or "").strip()
    if not provider_frame_version:
        raise ValueError(
            "external navigation map provider requires frame_version"
        )
    if not provider_pose_version:
        raise ValueError(
            "external navigation map provider requires pose_version"
        )
    if provider_pose_version != provider_frame_version:
        raise ValueError(
            "external navigation map provider pose_version must match "
            "frame_version"
        )
    pose = payload.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("external navigation map provider requires pose")
    frame = str(payload.get("frame") or "").strip()
    pose_frame = str(pose.get("frame") or "").strip()
    if not frame or not pose_frame:
        raise ValueError(
            "external navigation map provider requires frame and pose.frame"
        )
    if frame != pose_frame:
        raise ValueError(
            "external navigation map provider frame must match pose.frame "
            f"({frame!r} != {pose_frame!r})"
        )
    return payload, provider_epoch, provider_version, provider_frame_version


def _validate_stable_external_snapshot(
    payload: Mapping[str, Any],
    cached: Mapping[str, Any],
) -> None:
    """Cheaply enforce fields that must stay fixed under one map version.

    Array contents are governed by the provider's version contract.  Inspecting
    every cell here would defeat the stable-version cache, but dimensions and
    coordinate metadata are inexpensive to validate on every publication.
    """

    occupancy = np.asarray(payload.get("occupancy"))
    cached_shape = tuple(int(value) for value in cached.get("shape") or ())
    if occupancy.ndim != 2 or not occupancy.size:
        raise ValueError("occupancy must be a non-empty 2-D array")
    if occupancy.shape != cached_shape:
        raise ValueError(
            "external provider changed occupancy shape without changing "
            "map_version"
        )
    declared_shape = payload.get("shape")
    if declared_shape is not None:
        shape_values = np.asarray(declared_shape).reshape(-1)
        if (
            shape_values.size != 2
            or tuple(int(v) for v in shape_values) != cached_shape
        ):
            raise ValueError(
                "external provider shape does not match occupancy"
            )
    layers = payload.get("layers")
    if layers is not None:
        if not isinstance(layers, Mapping):
            raise ValueError("navigation map layers must be a mapping")
        for name, value in layers.items():
            if np.asarray(value).shape != cached_shape:
                raise ValueError(
                    f"navigation map layer {str(name)!r} shape does not "
                    f"match {cached_shape}"
                )

    origin = np.asarray(payload.get("origin"), dtype=np.float64).reshape(-1)
    if origin.size != 2 or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain finite x_min_m and y_min_m")
    if not np.array_equal(origin, np.asarray(cached["origin"])):
        raise ValueError(
            "external provider changed origin without changing map_version"
        )
    resolution = _finite_float(payload.get("resolution"), "resolution")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if resolution != float(cached["resolution"]):
        raise ValueError(
            "external provider changed resolution without changing map_version"
        )
    if str(payload.get("frame") or "").strip() != str(cached["frame"]):
        raise ValueError(
            "external provider changed frame without changing map_epoch"
        )
    if _normalise_places(payload.get("places")) != list(cached["places"]):
        raise ValueError(
            "external provider changed places without changing map_version"
        )


def _refresh_cached_snapshot(
    cached: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh per-frame fields while retaining cached immutable geometry."""

    refreshed = dict(cached)
    for key in ("episode_id", "map_epoch", "map_version", "frame_version"):
        refreshed[key] = str(raw.get(key) or "")
    refreshed["backend"] = str(
        raw.get("backend") or cached.get("backend") or "external"
    )
    refreshed["build"] = str(raw.get("build") or cached.get("build") or "")
    refreshed["observation_sequence"] = int(raw["observation_sequence"])
    refreshed["captured_ts"] = _finite_float(
        raw["captured_ts"], "captured_ts"
    )
    pose_freshness = normalize_navigation_pose_freshness(raw)

    pose = _normalise_pose(raw.get("pose") or {})
    frame = str(raw.get("frame") or pose["frame"]).strip()
    if frame != pose["frame"]:
        raise ValueError(
            "navigation map frame must match pose.frame "
            f"({frame!r} != {pose['frame']!r})"
        )
    if pose_freshness["pose_version"] == cached.get("pose_version"):
        cached_pose = dict(cached.get("pose") or {})
        current_revision = (
            pose["x"],
            pose["y"],
            pose["yaw_deg"],
            pose["frame"],
        )
        cached_revision = (
            cached_pose.get("x"),
            cached_pose.get("y"),
            cached_pose.get("yaw_deg"),
            cached_pose.get("frame"),
        )
        if current_revision != cached_revision:
            raise ValueError(
                "map provider changed pose without changing pose_version"
            )
    refreshed["frame"] = frame
    lifecycle = deepcopy(dict(raw.get("lifecycle") or {}))
    pose_confident = _strict_bool(
        lifecycle.get("pose_confident"),
        "lifecycle.pose_confident",
        default=pose["global_confident"],
    )
    pose["global_confident"] = pose_confident
    lifecycle["pose_confident"] = pose_confident
    pose_source = normalize_navigation_pose_source(raw)
    lifecycle["pose_source_sequence_known"] = pose_source[
        "pose_source_sequence_known"
    ]
    lifecycle["source_lag_known"] = pose_source[
        "pose_source_sequence_known"
    ]
    refreshed["pose"] = pose
    refreshed["lifecycle"] = lifecycle
    if not pose_source["pose_source_sequence_known"]:
        refreshed.pop("pose_source_observation_sequence", None)
    refreshed.update(pose_source)
    refreshed.update(pose_freshness)

    policy_pose = _normalise_policy_pose(raw.get("policy_local_pose"))
    if policy_pose is None:
        refreshed.pop("policy_local_pose", None)
    else:
        refreshed["policy_local_pose"] = policy_pose
    source_frame = raw.get("source_frame")
    if source_frame is None:
        refreshed.pop("source_frame", None)
    elif not isinstance(source_frame, Mapping):
        raise ValueError("source_frame must be a mapping")
    else:
        refreshed["source_frame"] = deepcopy(dict(source_frame))
        ordered_producer = normalize_ordered_pose_producer(raw)
        if ordered_producer is not None:
            refreshed["source_frame"]["ordered_pose_producer"] = (
                ordered_producer
            )
    if "traversed_paths_xy_m" in raw:
        refreshed["traversed_paths_xy_m"] = _normalise_traversed_paths(
            raw.get("traversed_paths_xy_m")
        )
    if "observation_anchors" in raw:
        refreshed["observation_anchors"] = _normalise_observation_anchors(
            raw.get("observation_anchors")
        )
    for key in (
        "provider_map_epoch",
        "provider_map_version",
        "provider_frame_version",
        "provider_pose_version",
    ):
        if raw.get(key) is None:
            refreshed.pop(key, None)
        else:
            refreshed[key] = str(raw[key])
    return refreshed


class NavigationMapBridge:
    """Capture standard snapshots while reusing unchanged immutable grids."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache_key: Optional[tuple[Any, ...]] = None
        self._cached_snapshot: Optional[dict[str, Any]] = None
        self._stream_generation = 0
        self._stream_episode: Optional[str] = None
        self._stream_kind: Optional[str] = None
        self._stream_provider: Any = None
        self._stream_aux: Any = None
        self._stream_provider_epoch: Optional[str] = None
        self._legacy_grid: Any = None
        self._legacy_grid_generation = 0
        self._pose_version: Optional[str] = None
        self._pose_observed_sequence: Optional[int] = None
        self._pose_observed_ts: Optional[float] = None
        self._rtab_trail_cache: Optional[dict[str, Any]] = None

    def _clear_stream_state(self) -> None:
        self._cache_key = None
        self._cached_snapshot = None
        self._stream_episode = None
        self._stream_kind = None
        self._stream_provider = None
        self._stream_aux = None
        self._stream_provider_epoch = None
        self._legacy_grid = None
        self._legacy_grid_generation = 0
        self._pose_version = None
        self._pose_observed_sequence = None
        self._pose_observed_ts = None
        self._rtab_trail_cache = None

    def _activate_stream(
        self,
        provider: Any,
        *,
        episode_id: str,
        kind: str,
        aux: Any = None,
        provider_epoch: Optional[str] = None,
    ) -> int:
        changed = bool(
            self._stream_episode != episode_id
            or self._stream_kind != kind
            or self._stream_provider is not provider
            or self._stream_aux is not aux
            or self._stream_provider_epoch != provider_epoch
        )
        if changed:
            self._stream_generation += 1
            self._cache_key = None
            self._cached_snapshot = None
            self._legacy_grid = None
            self._legacy_grid_generation = 0
            self._pose_version = None
            self._pose_observed_sequence = None
            self._pose_observed_ts = None
            self._rtab_trail_cache = None
            self._stream_episode = episode_id
            self._stream_kind = kind
            # Strong references make ``is`` meaningful even if Python later
            # reuses an object id after a mapper restart.
            self._stream_provider = provider
            self._stream_aux = aux
            self._stream_provider_epoch = provider_epoch
        return self._stream_generation

    @staticmethod
    def _map_epoch(
        episode_id: str,
        stream_generation: int,
        provider_epoch: Optional[str] = None,
    ) -> str:
        epoch = f"{episode_id}:navigation-stream:{stream_generation}"
        if provider_epoch is not None:
            epoch = f"{epoch}:provider:{provider_epoch}"
        return epoch

    def _stamp_pose_freshness(self, raw: dict[str, Any]) -> None:
        """Record bridge first-seen time for an opaque producer revision."""

        pose_version = str(raw.get("pose_version") or "").strip()
        if not pose_version:
            raise ValueError("map provider requires a non-empty pose_version")
        observation_sequence = int(raw.get("observation_sequence") or 0)
        captured_ts = _finite_float(raw.get("captured_ts"), "captured_ts")
        if pose_version != self._pose_version:
            self._pose_version = pose_version
            self._pose_observed_sequence = observation_sequence
            self._pose_observed_ts = captured_ts
        raw["pose_observed_sequence"] = self._pose_observed_sequence
        raw["pose_observed_ts"] = self._pose_observed_ts
        raw.update(normalize_navigation_pose_freshness(raw))

    def reset(self) -> None:
        with self._lock:
            # Keep the counter monotonic so a stale action cannot mistake a
            # post-reset mapper for the stream it planned against.
            self._clear_stream_state()

    def capture(
        self,
        provider: Any,
        *,
        episode_id: str,
        observation_sequence: int,
        policy_local_pose: Any = None,
        captured_ts: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Capture one provider state without importing its implementation.

        New backends should expose ``export_navigation_map_snapshot()`` and
        return the canonical mapping fields documented at module level.  The
        RTAB-Map and EgoMap branches are compatibility adapters for the two
        live mappers that predate that provider protocol.
        """

        if provider is None:
            return None
        episode = str(episode_id or "").strip()
        if not episode:
            raise ValueError("episode_id is required for navigation maps")
        sequence = int(observation_sequence)
        timestamp = float(time.time() if captured_ts is None else captured_ts)
        with self._lock:
            exporter = getattr(
                provider, "export_navigation_map_snapshot", None
            )
            if callable(exporter):
                exported = exporter()
                if exported is None:
                    return None
                (
                    raw,
                    provider_epoch,
                    provider_version,
                    provider_frame_version,
                ) = _external_provider_header(exported)
                generation = self._activate_stream(
                    provider,
                    episode_id=episode,
                    kind="external",
                    provider_epoch=provider_epoch,
                )
                map_epoch = self._map_epoch(
                    episode, generation, provider_epoch
                )
                raw["provider_map_epoch"] = provider_epoch
                raw["provider_map_version"] = provider_version
                raw["provider_frame_version"] = provider_frame_version
                raw["provider_pose_version"] = provider_frame_version
                raw["episode_id"] = episode
                raw["map_epoch"] = map_epoch
                raw["map_version"] = (
                    f"{map_epoch}|provider-map:{provider_version}"
                )
                raw["frame_version"] = (
                    f"{map_epoch}|provider-frame:{provider_frame_version}"
                )
                raw["pose_version"] = raw["frame_version"]
                raw["observation_sequence"] = sequence
                raw["captured_ts"] = timestamp
                raw["policy_local_pose"] = policy_local_pose
                # Plugin providers may bind their pose to an evaluator
                # observation explicitly.  Missing metadata remains unknown;
                # a claimed source is range/type checked against this capture.
                provider_pose_source = normalize_navigation_pose_source(raw)
                raw.update(provider_pose_source)
                if not provider_pose_source["pose_source_sequence_known"]:
                    raw.pop("pose_source_observation_sequence", None)
                source_frame = raw.get("source_frame")
                if source_frame is not None and not isinstance(
                    source_frame, Mapping
                ):
                    raise ValueError("source_frame must be a mapping")
                raw["source_frame"] = {
                    **deepcopy(dict(source_frame or {})),
                    "provider_frame_version": provider_frame_version,
                    "lag_known": provider_pose_source[
                        "pose_source_sequence_known"
                    ],
                }
                if provider_pose_source["pose_source_sequence_known"]:
                    source_sequence = provider_pose_source[
                        "pose_source_observation_sequence"
                    ]
                    raw["source_frame"].update(
                        {
                            "evaluator_sequence": source_sequence,
                            "lag_observations": sequence - source_sequence,
                        }
                    )
                lifecycle = dict(raw.get("lifecycle") or {})
                lifecycle["pose_source_sequence_known"] = (
                    provider_pose_source["pose_source_sequence_known"]
                )
                lifecycle["source_lag_known"] = provider_pose_source[
                    "pose_source_sequence_known"
                ]
                raw["lifecycle"] = lifecycle
                self._stamp_pose_freshness(raw)
                cache_key = ("external", map_epoch, provider_version)
                cached = self._cached_snapshot
                if cache_key == self._cache_key and cached is not None:
                    _validate_stable_external_snapshot(raw, cached)
                    refreshed = _refresh_cached_snapshot(cached, raw)
                    self._cached_snapshot = refreshed
                    return refreshed
                snapshot = normalize_navigation_map_snapshot(raw)
                self._cache_key = cache_key
                self._cached_snapshot = snapshot
                return snapshot

            if hasattr(provider, "_latest"):
                generation = self._activate_stream(
                    provider,
                    episode_id=episode,
                    kind="rtabmap",
                )
                map_epoch = self._map_epoch(episode, generation)
                raw, cache_key, trail_cache = _rtab_snapshot_raw(
                    provider,
                    cached_key=self._cache_key,
                    trail_cache=getattr(self, "_rtab_trail_cache", None),
                    episode_id=episode,
                    map_epoch=map_epoch,
                    observation_sequence=sequence,
                    captured_ts=timestamp,
                    policy_local_pose=policy_local_pose,
                )
                self._rtab_trail_cache = trail_cache
            else:
                ego_provider = getattr(provider, "_ego", None)
                if not callable(ego_provider):
                    return None
                ego = ego_provider()
                if ego is None or getattr(ego, "grid", None) is None:
                    return None
                generation = self._activate_stream(
                    provider,
                    episode_id=episode,
                    kind="egomap",
                    aux=ego,
                )
                grid = ego.grid
                if self._legacy_grid is not grid:
                    self._legacy_grid = grid
                    self._legacy_grid_generation += 1
                map_epoch = self._map_epoch(episode, generation)
                raw, cache_key = _legacy_snapshot_raw(
                    ego,
                    cached_key=self._cache_key,
                    episode_id=episode,
                    map_epoch=map_epoch,
                    grid_generation=self._legacy_grid_generation,
                    observation_sequence=sequence,
                    captured_ts=timestamp,
                    policy_local_pose=policy_local_pose,
                )
            if raw is None:
                return None
            self._stamp_pose_freshness(raw)

            cached = self._cached_snapshot
            if cache_key is not None and cache_key == self._cache_key and cached:
                # Geometry and landmarks are unchanged.  Preserve their
                # immutable arrays, but always publish the current pose,
                # lifecycle, sequence and policy-local odometry metadata.
                refreshed = _refresh_cached_snapshot(cached, raw)
                self._cached_snapshot = refreshed
                return refreshed

            snapshot = normalize_navigation_map_snapshot(raw)
            self._cache_key = cache_key
            self._cached_snapshot = snapshot
            return snapshot


def capture_navigation_map_snapshot(
    provider: Any,
    *,
    episode_id: str,
    observation_sequence: int,
    policy_local_pose: Any = None,
    captured_ts: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Stateless convenience wrapper for tests and one-shot providers."""

    return NavigationMapBridge().capture(
        provider,
        episode_id=episode_id,
        observation_sequence=observation_sequence,
        policy_local_pose=policy_local_pose,
        captured_ts=captured_ts,
    )


__all__ = [
    "FREE",
    "NAVIGATION_MAP_SCHEMA",
    "NAVIGATION_MAP_SCHEMA_VERSION",
    "NavigationMapBridge",
    "OCCUPIED",
    "UNKNOWN",
    "capture_navigation_map_snapshot",
    "copy_navigation_map_snapshot",
    "normalize_navigation_map_snapshot",
    "normalize_navigation_pose_freshness",
    "normalize_navigation_pose_source",
    "view_navigation_map_snapshot",
]
