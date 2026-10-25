"""Thread-safe transient navigation route state for minimap renderers.

This module contains no planner or map-backend logic.  ``navigate_to`` publishes
map-frame points, while any minimap renderer can consume the immutable snapshot.
The plan id is used as a compare-and-swap token so a finishing old controller
cannot erase a newer route for the same session.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import replace
from functools import wraps
from io import BytesIO
import math
import os
import sys
import threading
from typing import Any, Iterable, Optional, Sequence

import numpy as np


SCHEMA = "behavior.navigation_route_overlay.v1"
MAX_ROUTE_POINTS = 4096
MAX_ABS_COORDINATE_M = 1.0e6
ROUTE_SWEEP_RADIUS_M = 0.40

# The route overlay is installed into the long-lived interface process.  Keep
# its RTAB snapshot path bounded even when that process has not been restarted
# after a live.py change.  This limit applies to PNG rendering only; the
# mapper's anchored trail and the navigation bridge retain every source tick.
try:
    _display_trail_limit = int(
        os.environ.get("BEHAVIOR_RTABMAP_DISPLAY_TRAIL_MAX_POINTS", "8192")
    )
except (TypeError, ValueError):
    _display_trail_limit = 8192
DISPLAY_TRAIL_MAX_POINTS = max(1024, min(_display_trail_limit, 65536))


@dataclass(frozen=True)
class RouteOverlaySnapshot:
    schema: str
    session_id: str
    plan_id: str
    points_xy_m: tuple[tuple[float, float], ...]
    revision: int


@dataclass(frozen=True)
class _RouteRecord:
    plan_id: str
    path_xy_m: tuple[tuple[float, float], ...]
    points_xy_m: tuple[tuple[float, float], ...]
    next_waypoint_index: int
    revision: int


@dataclass(frozen=True)
class RendererHookToken:
    """Opaque exact function-pointer snapshot for hot-load rollback."""

    spatial_module: Any
    spatial_render_minimap_rgba: Any
    spatial_map_version: Any
    live_module: Any
    live_mapper_class: Any
    live_map_snapshot_png: Any
    schema: str = "behavior.navigation_route_hook_token.v1"


# Preserve state if a development reload explicitly reloads this module.  The
# public functions are rebound, but a route already visible in the UI is not
# lost between two controller ticks.
try:  # pragma: no branch - only false on the first import
    _LOCK
except NameError:
    _LOCK = threading.RLock()
    _ROUTES: dict[str, _RouteRecord] = {}
    _REVISION = 0


def _session_key(session_id: Any) -> str:
    key = str(session_id or "").strip()
    if not key:
        raise ValueError("navigation route overlay session_id is required")
    if len(key) > 256:
        raise ValueError("navigation route overlay session_id is too long")
    return key


def _plan_key(plan_id: Any) -> str:
    key = str(plan_id or "").strip()
    if not key:
        raise ValueError("navigation route overlay plan_id is required")
    if len(key) > 256:
        raise ValueError("navigation route overlay plan_id is too long")
    return key


def _point(value: Sequence[Any], *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element coordinate")
    try:
        x_m = float(value[0])
        y_m = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite coordinates") from exc
    if (
        not math.isfinite(x_m)
        or not math.isfinite(y_m)
        or abs(x_m) > MAX_ABS_COORDINATE_M
        or abs(y_m) > MAX_ABS_COORDINATE_M
    ):
        raise ValueError(f"{label} must contain bounded finite coordinates")
    return x_m, y_m


def _path(points_xy_m: Iterable[Sequence[Any]]) -> tuple[tuple[float, float], ...]:
    try:
        raw = tuple(points_xy_m)
    except TypeError as exc:
        raise ValueError("navigation route points must be iterable") from exc
    if not raw:
        raise ValueError("navigation route must contain at least one point")
    if len(raw) > MAX_ROUTE_POINTS:
        raise ValueError("navigation route has too many points")
    points = tuple(
        _point(value, label=f"points_xy_m[{index}]")
        for index, value in enumerate(raw)
    )
    return points


def _next_revision() -> int:
    global _REVISION
    _REVISION += 1
    return int(_REVISION)


def publish_route(
    session_id: str,
    plan_id: str,
    path_xy_m: Iterable[Sequence[Any]],
    *,
    current_xy_m: Optional[Sequence[Any]] = None,
) -> int:
    """Publish a complete newly planned route and return its revision."""

    session = _session_key(session_id)
    plan = _plan_key(plan_id)
    path = _path(path_xy_m)
    current = path[0] if current_xy_m is None else _point(
        current_xy_m, label="current_xy_m"
    )
    remaining = (current,) + path[1:]
    with _LOCK:
        revision = _next_revision()
        _ROUTES[session] = _RouteRecord(
            plan_id=plan,
            path_xy_m=path,
            points_xy_m=remaining,
            next_waypoint_index=1,
            revision=revision,
        )
    return revision


def advance_route(
    session_id: str,
    plan_id: str,
    current_xy_m: Sequence[Any],
    *,
    next_waypoint_index: int,
) -> int:
    """Trim the walked prefix while retaining the current active segment.

    ``next_waypoint_index`` names the point in the originally published path
    that the controller is currently approaching.  The visible route starts at
    the latest SLAM pose and continues from that waypoint, so every already
    travelled portion disappears without guessing at self-intersecting paths.
    A stale plan id is ignored and returns the current session revision.
    """

    session = _session_key(session_id)
    plan = _plan_key(plan_id)
    current = _point(current_xy_m, label="current_xy_m")
    if isinstance(next_waypoint_index, bool):
        raise ValueError("next_waypoint_index must be an integer")
    try:
        target_index = int(next_waypoint_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("next_waypoint_index must be an integer") from exc
    if target_index != next_waypoint_index:
        raise ValueError("next_waypoint_index must be an integer")

    with _LOCK:
        record = _ROUTES.get(session)
        if record is None:
            return 0
        if record.plan_id != plan:
            return int(record.revision)
        if target_index < record.next_waypoint_index:
            return int(record.revision)
        if target_index < 1 or target_index >= len(record.path_xy_m):
            raise ValueError("next_waypoint_index is outside the route")
        remaining = (current,) + record.path_xy_m[target_index:]
        # A sub-pixel pose update still needs a revision: PNG caches otherwise
        # keep showing the walked prefix until the next map frame arrives.
        revision = _next_revision()
        _ROUTES[session] = _RouteRecord(
            plan_id=record.plan_id,
            path_xy_m=record.path_xy_m,
            points_xy_m=remaining,
            next_waypoint_index=target_index,
            revision=revision,
        )
        return revision


def get_route_snapshot(session_id: str) -> Optional[RouteOverlaySnapshot]:
    """Return an immutable copy of the current route for one session."""

    try:
        session = _session_key(session_id)
    except ValueError:
        return None
    with _LOCK:
        record = _ROUTES.get(session)
        if record is None:
            return None
        return RouteOverlaySnapshot(
            schema=SCHEMA,
            session_id=session,
            plan_id=str(record.plan_id),
            points_xy_m=tuple(record.points_xy_m),
            revision=int(record.revision),
        )


def get_route_for_display(session_id: str) -> Optional[RouteOverlaySnapshot]:
    """Resolve a renderer session, with a safe single-route UI fallback.

    The map starts before an agent session exists, and a UI poll may still ask
    for ``default`` after ``navigate_to`` has an evaluator session id.  Falling
    back is unambiguous only while exactly one route exists in this one-robot
    process.  Multiple sessions never leak into one another.
    """

    exact = get_route_snapshot(session_id)
    if exact is not None:
        return exact
    with _LOCK:
        if len(_ROUTES) != 1:
            return None
        session, record = next(iter(_ROUTES.items()))
        return RouteOverlaySnapshot(
            schema=SCHEMA,
            session_id=session,
            plan_id=str(record.plan_id),
            points_xy_m=tuple(record.points_xy_m),
            revision=int(record.revision),
        )


def route_version(session_id: str) -> int:
    snapshot = get_route_snapshot(session_id)
    return 0 if snapshot is None else int(snapshot.revision)


def clear_route(session_id: str, plan_id: Optional[str] = None) -> int:
    """Clear one route, conditionally when ``plan_id`` is supplied."""

    session = _session_key(session_id)
    plan = None if plan_id is None else _plan_key(plan_id)
    with _LOCK:
        record = _ROUTES.get(session)
        if record is None:
            return 0
        if plan is not None and record.plan_id != plan:
            return int(record.revision)
        _ROUTES.pop(session, None)
        return _next_revision()


def clear_all_routes() -> int:
    """Clear transient routes at an evaluator episode boundary."""

    with _LOCK:
        if not _ROUTES:
            return int(_REVISION)
        _ROUTES.clear()
        return _next_revision()


def _composite_route_rgba(image: Any, points_xy_m, to_pixel, *, width: int):
    """Draw a metric round buffer, preserving blue trail/marks above it."""

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("minimap renderer returned an invalid image")
    source_rgba = (
        source.copy()
        if source.shape[2] == 4
        else np.concatenate(
            [
                source,
                np.full(source.shape[:2] + (1,), 255, dtype=np.uint8),
            ],
            axis=2,
        )
    )
    base = Image.fromarray(source_rgba, mode="RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    pixels = [to_pixel(float(x_m), float(y_m)) for x_m, y_m in points_xy_m]
    if len(pixels) < 2:
        return source.copy()
    first_x, first_y = points_xy_m[0]
    offset_pixel = to_pixel(float(first_x) + ROUTE_SWEEP_RADIUS_M, float(first_y))
    radius_px = math.dist(pixels[0], offset_pixel)
    # The polyline is unchanged. Its footprint is a union of rectangles and
    # endpoint disks, not a square-cornered stroke or an arc shortcut to drive.
    sweep_fill = (238, 126, 132, 76)
    for start, end in zip(pixels[:-1], pixels[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            continue
        nx, ny = -dy * radius_px / length, dx * radius_px / length
        draw.polygon(
            [
                (start[0] + nx, start[1] + ny),
                (end[0] + nx, end[1] + ny),
                (end[0] - nx, end[1] - ny),
                (start[0] - nx, start[1] - ny),
            ],
            fill=sweep_fill,
        )
    for x_px, y_px in pixels:
        draw.ellipse(
            (x_px - radius_px, y_px - radius_px,
             x_px + radius_px, y_px + radius_px),
            fill=sweep_fill,
        )
    draw.line(
        pixels,
        fill=(238, 126, 132, 224),
        width=max(3, int(width)),
        joint="curve",
    )
    composed = np.asarray(Image.alpha_composite(base, overlay)).copy()

    # Runtime hooks draw over an already-rendered old-process image.  Restore
    # saturated blue vector pixels (and a two-pixel halo) so the robot, marks
    # and actual travelled trail retain the same priority as the native path.
    red = source_rgba[:, :, 0].astype(np.int16)
    green = source_rgba[:, :, 1].astype(np.int16)
    blue = source_rgba[:, :, 2].astype(np.int16)
    protected = (blue > 110) & (green < 155) & (blue > red + 35)
    protected = np.asarray(
        Image.fromarray(protected.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(5)
        )
    ) > 0
    composed[protected] = source_rgba[protected]
    return composed if source.shape[2] == 4 else composed[:, :, :3]


def _renderer_identity_snapshot() -> dict[str, Any]:
    spatial = sys.modules.get("behavior_interface.spatial_map")
    live = sys.modules.get("behavior_interface.rtabmap_slam.live")
    maps = getattr(spatial, "_MAPS", None) if spatial is not None else None
    singleton = getattr(live, "_SINGLETON", None) if live is not None else None
    return {
        "spatial_module_loaded": spatial is not None,
        "rtab_live_module_loaded": live is not None,
        "egomap_container_id": None if maps is None else id(maps),
        "egomap_instance_ids": (
            {}
            if not isinstance(maps, dict)
            else {str(key): id(value) for key, value in maps.items()}
        ),
        "rtab_singleton_id": None if singleton is None else id(singleton),
    }


def capture_renderer_hook_state() -> RendererHookToken:
    """Capture exact hook targets before a transactional hot installation."""

    spatial = sys.modules.get("behavior_interface.spatial_map")
    live = sys.modules.get("behavior_interface.rtabmap_slam.live")
    with ExitStack() as locks:
        for lock in (
            getattr(spatial, "_MAPS_LOCK", None),
            getattr(live, "_SINGLETON_LOCK", None),
        ):
            if hasattr(lock, "__enter__"):
                locks.enter_context(lock)
        live_mapper = (
            getattr(live, "LiveMapper", None) if live is not None else None
        )
        return RendererHookToken(
            spatial_module=spatial,
            spatial_render_minimap_rgba=(
                getattr(spatial, "render_minimap_rgba", None)
                if spatial is not None
                else None
            ),
            spatial_map_version=(
                getattr(spatial, "map_version", None)
                if spatial is not None
                else None
            ),
            live_module=live,
            live_mapper_class=live_mapper,
            live_map_snapshot_png=(
                getattr(live_mapper, "map_snapshot_png", None)
                if live_mapper is not None
                else None
            ),
        )


def restore_renderer_hook_state(token: RendererHookToken) -> dict[str, Any]:
    """Restore the exact pre-transaction methods captured in ``token``."""

    if getattr(token, "schema", None) != "behavior.navigation_route_hook_token.v1":
        raise TypeError("renderer hook rollback token is invalid")
    restored = []
    spatial = sys.modules.get("behavior_interface.spatial_map")
    live = sys.modules.get("behavior_interface.rtabmap_slam.live")
    with ExitStack() as locks:
        for lock in (
            getattr(spatial, "_MAPS_LOCK", None),
            getattr(live, "_SINGLETON_LOCK", None),
        ):
            if hasattr(lock, "__enter__"):
                locks.enter_context(lock)
        before = _renderer_identity_snapshot()
        if spatial is not token.spatial_module or live is not token.live_module:
            raise RuntimeError(
                "stateful map module identity changed before hook rollback"
            )
        if spatial is not None:
            spatial.render_minimap_rgba = token.spatial_render_minimap_rgba
            spatial.map_version = token.spatial_map_version
            restored.extend(
                [
                    "behavior_interface.spatial_map.render_minimap_rgba",
                    "behavior_interface.spatial_map.map_version",
                ]
            )
        if token.live_mapper_class is not None:
            if getattr(live, "LiveMapper", None) is not token.live_mapper_class:
                raise RuntimeError(
                    "LiveMapper class identity changed before hook rollback"
                )
            token.live_mapper_class.map_snapshot_png = token.live_map_snapshot_png
            restored.append(
                "behavior_interface.rtabmap_slam.live.LiveMapper.map_snapshot_png"
            )
        after = _renderer_identity_snapshot()
        if before != after:
            # Function replacement must not alter any entry represented in this
            # identity report; fail visibly if a concurrent episode reset did.
            raise RuntimeError("live map identity changed during hook rollback")
    return {
        "ok": True,
        "restored": restored,
        "state_identity_preserved": True,
    }


def renderer_hooks_installed() -> tuple[str, ...]:
    """List currently monkey-patched targets (native integration is omitted)."""

    targets = []
    spatial = sys.modules.get("behavior_interface.spatial_map")
    for owner, name in (
        (spatial, "render_minimap_rgba"),
        (spatial, "map_version"),
    ):
        value = getattr(owner, name, None) if owner is not None else None
        if bool(getattr(value, "_navigation_route_overlay_hook", False)):
            targets.append(f"behavior_interface.spatial_map.{name}")
    live = sys.modules.get("behavior_interface.rtabmap_slam.live")
    live_mapper = getattr(live, "LiveMapper", None) if live is not None else None
    value = getattr(live_mapper, "map_snapshot_png", None)
    if bool(getattr(value, "_navigation_route_overlay_hook", False)):
        targets.append(
            "behavior_interface.rtabmap_slam.live.LiveMapper.map_snapshot_png"
        )
    return tuple(targets)


def _original_hook_target(value: Any) -> Any:
    return getattr(value, "_navigation_route_overlay_original", value)


def _display_trail_items(
    trail: Any,
    limit: int,
) -> Iterable[Any]:
    """Select a bounded deterministic view of an RTAB anchored trail."""

    try:
        count = len(trail)
    except (TypeError, AttributeError):
        return ()
    if count <= limit:
        return iter(trail)
    if limit <= 1:
        return (trail[count - 1],)
    return (
        trail[round(index * (count - 1) / (limit - 1))]
        for index in range(limit)
    )


def _fast_live_display_result(
    mapper: Any,
    live_module: Any,
    heading_up: bool,
) -> Any:
    """Resolve only the points needed by a raster snapshot.

    Older interface processes may still hold the pre-boundary ``live.py``
    method.  Keeping this compatibility path here lets a transactional tool
    reload remove its O(total-trail) request stall without replacing the live
    mapper object or its worker.  ``heading_up`` is intentionally accepted to
    mirror the native helper; pose resolution itself is orientation-neutral.
    """

    del heading_up
    state_lock = getattr(mapper, "_state_lock", None)
    lock_context = (
        state_lock if hasattr(state_lock, "__enter__") else _NullContext()
    )
    with lock_context:
        result = getattr(mapper, "_latest", None)
        if result is None:
            return None
        pose_type = getattr(live_module, "SE2Pose")
        graph = {
            item.node_id: pose_type(item.x_m, item.y_m, item.yaw_rad)
            for item in result.poses
        }
        trail = getattr(mapper, "_trail", ())
        selected = _display_trail_items(trail, DISPLAY_TRAIL_MAX_POINTS)
        pose_records = getattr(live_module, "PoseRecord")
        resolved = []
        for item in selected:
            anchor = graph.get(item.anchor_node_id)
            pose = (
                anchor.compose(item.local_pose)
                if anchor is not None
                else item.fallback_pose
            )
            resolved.append(
                pose_records(
                    len(resolved) + 1,
                    pose.x_m,
                    pose.y_m,
                    pose.yaw_rad,
                )
            )
        return replace(result, poses=tuple(resolved))


def _fast_live_snapshot_png(
    mapper: Any,
    live_module: Any,
    *,
    heading_up: bool,
    size: int,
    span_m: Optional[float],
) -> tuple[bytes, str, Any, Any]:
    """Render a bounded RTAB snapshot and return its result/route metadata."""

    state_lock = getattr(mapper, "_state_lock", None)
    lock_context = (
        state_lock if hasattr(state_lock, "__enter__") else _NullContext()
    )
    with lock_context:
        native_result = getattr(mapper, "_latest", None)
        if native_result is None:
            raise RuntimeError(
                "RTAB-Map has not processed its first official RGB-D frame"
            )
        result = _fast_live_display_result(mapper, live_module, heading_up)
        if result is None:
            raise RuntimeError(
                "RTAB-Map has not processed its first official RGB-D frame"
            )
        resolver = getattr(mapper, "_resolved_places", None)
        if callable(resolver):
            places = tuple(
                (item.x_m, item.y_m, item.name)
                for item in resolver(native_result)
            )
        else:
            places = ()
        route = get_route_for_display(getattr(mapper, "_session_id", ""))
        route_revision = 0 if route is None else int(route.revision)
        view_span = None if span_m is None else float(span_m)
        key = (
            result.frame_id,
            bool(heading_up),
            int(size),
            view_span,
            int(getattr(mapper, "_places_version", 0)),
            route_revision,
        )
        cache = getattr(mapper, "_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(mapper, "_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached[0], cached[1], result, route

    image = live_module.render_heading_up(
        result,
        size_px=int(size),
        span_m=view_span,
        heading_up=bool(heading_up),
        places=places,
        route_xy_m=(None if route is None else route.points_xy_m),
    )
    output = BytesIO()
    image.save(output, format="PNG")
    payload = output.getvalue()
    version = (
        f"{getattr(live_module, 'BUILD', '')}:{result.frame_id}:"
        f"{result.loop_count}:{int(getattr(mapper, '_places_version', 0))}:"
        f"{route_revision}:{int(bool(heading_up))}"
    )
    with lock_context:
        cache[key] = (payload, version)
    return payload, version, result, route


def _install_egomap_hooks(
    module: Any,
    changed: list[tuple[Any, str, Any]],
) -> list[str]:
    installed = []

    current_render = getattr(module, "render_minimap_rgba", None)
    if callable(current_render) and not bool(
        getattr(current_render, "_navigation_route_sweep_native", False)
    ):
        original_render = _original_hook_target(current_render)

        @wraps(original_render)
        def route_render(ego, *args, **kwargs):
            rendered = original_render(ego, *args, **kwargs)
            route = get_route_for_display(getattr(ego, "session_id", ""))
            if route is None or len(route.points_xy_m) < 2:
                return rendered
            size = int(np.asarray(rendered).shape[0])
            heading_up = bool(kwargs.get("heading_up", True))
            range_m = kwargs.get("range_m")
            center_x, center_y, span = module._view_window(
                ego, range_m=range_m, heading_up=heading_up
            )
            scale = (size * module.VIEW_FILL) / (2.0 * span)

            def to_pixel(x_m: float, y_m: float):
                up, left = module._view_axes(
                    ego,
                    x_m,
                    y_m,
                    center_x,
                    center_y,
                    heading_up,
                )
                return size * 0.5 - left * scale, size * 0.5 - up * scale

            return _composite_route_rgba(
                rendered,
                route.points_xy_m,
                to_pixel,
                width=max(3, int(round(size / 150.0))),
            )

        route_render._navigation_route_overlay_hook = True
        route_render._navigation_route_overlay_original = original_render
        changed.append((module, "render_minimap_rgba", current_render))
        module.render_minimap_rgba = route_render
        installed.append("behavior_interface.spatial_map.render_minimap_rgba")

    current_version = getattr(module, "map_version", None)
    if callable(current_version) and not bool(
        getattr(current_version, "_navigation_route_overlay_native", False)
    ):
        original_version = _original_hook_target(current_version)

        @wraps(original_version)
        def route_versioned(ego):
            base = str(original_version(ego))
            route = get_route_for_display(getattr(ego, "session_id", ""))
            revision = 0 if route is None else int(route.revision)
            return f"{base}|navigation-route:{revision}"

        route_versioned._navigation_route_overlay_hook = True
        route_versioned._navigation_route_overlay_original = original_version
        changed.append((module, "map_version", current_version))
        module.map_version = route_versioned
        installed.append("behavior_interface.spatial_map.map_version")
    return installed


def _install_rtab_hook(module: Any, changed: list[tuple[Any, str, Any]]) -> list[str]:
    live_mapper = getattr(module, "LiveMapper", None)
    if live_mapper is None:
        return []
    current = getattr(live_mapper, "map_snapshot_png", None)
    if not callable(current) or bool(
        getattr(current, "_navigation_route_sweep_native", False)
    ):
        return []
    original = _original_hook_target(current)

    @wraps(original)
    def route_snapshot_png(self, *args, **kwargs):
        # LiveMapper's public signature is keyword-only.  Preserve the native
        # error/argument behavior for any legacy positional caller.
        if args:
            return original(self, *args, **kwargs)
        live_module = module
        payload, version, result, route = _fast_live_snapshot_png(
            self,
            live_module,
            heading_up=bool(kwargs.get("heading_up", True)),
            size=int(
                kwargs.get(
                    "size",
                    getattr(live_module, "DEFAULT_MAP_SIZE_PX", 490),
                )
            ),
            span_m=kwargs.get(
                "span_m",
                getattr(live_module, "DEFAULT_MAP_SPAN_M", None),
            ),
        )
        if route is None or len(route.points_xy_m) < 2 or result is None:
            return payload, version

        from PIL import Image

        render_module = sys.modules.get("behavior_interface.rtabmap_slam.render")
        if render_module is None:
            return payload, version
        size = int(kwargs.get("size", getattr(module, "DEFAULT_MAP_SIZE_PX", 490)))
        span_m = kwargs.get("span_m", getattr(module, "DEFAULT_MAP_SPAN_M", None))
        heading_up = bool(kwargs.get("heading_up", True))
        view_yaw = result.current_pose.yaw_rad if heading_up else 0.0
        if span_m is None:
            view_options = (
                {"extra_points_xy_m": route.points_xy_m}
                if getattr(original, "_navigation_route_overlay_native", False)
                else {}
            )
            view_pose, render_span_m = render_module._auto_map_view(
                result, yaw_rad=view_yaw, **view_options
            )
        else:
            view_pose = module.SE2Pose(
                result.current_pose.x_m,
                result.current_pose.y_m,
                view_yaw,
            )
            render_span_m = float(span_m)
        with Image.open(BytesIO(payload)) as encoded:
            base = np.asarray(encoded.convert("RGBA"))

        def to_pixel(x_m: float, y_m: float):
            return render_module._to_pixel(
                x_m, y_m, view_pose, size, render_span_m
            )

        rendered = _composite_route_rgba(
            base,
            route.points_xy_m,
            to_pixel,
            width=max(3, int(round(size / 150.0))),
        )
        output = BytesIO()
        Image.fromarray(rendered, mode="RGBA").save(output, format="PNG")
        return output.getvalue(), f"{version}|navigation-sweep:0.4:{route.revision}"

    route_snapshot_png._navigation_route_overlay_hook = True
    route_snapshot_png._navigation_route_overlay_original = original
    changed.append((live_mapper, "map_snapshot_png", current))
    live_mapper.map_snapshot_png = route_snapshot_png
    return ["behavior_interface.rtabmap_slam.live.LiveMapper.map_snapshot_png"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def install_renderer_hooks() -> dict[str, Any]:
    """Hot-install route drawing without reloading either stateful map module."""

    with ExitStack() as locks:
        spatial = sys.modules.get("behavior_interface.spatial_map")
        live = sys.modules.get("behavior_interface.rtabmap_slam.live")
        spatial_lock = getattr(spatial, "_MAPS_LOCK", None)
        singleton_lock = getattr(live, "_SINGLETON_LOCK", None)
        if hasattr(spatial_lock, "__enter__"):
            locks.enter_context(spatial_lock)
        if hasattr(singleton_lock, "__enter__"):
            locks.enter_context(singleton_lock)

        before = _renderer_identity_snapshot()
        changed: list[tuple[Any, str, Any]] = []
        installed: list[str] = []
        try:
            if spatial is not None:
                installed.extend(_install_egomap_hooks(spatial, changed))
            if live is not None:
                installed.extend(_install_rtab_hook(live, changed))
            after = _renderer_identity_snapshot()
            identity_fields = (
                "egomap_container_id",
                "egomap_instance_ids",
                "rtab_singleton_id",
            )
            if any(before[field] != after[field] for field in identity_fields):
                raise RuntimeError("route hook installation changed live map identity")
        except Exception:
            for owner, name, previous in reversed(changed):
                setattr(owner, name, previous)
            raise
    return {
        "ok": True,
        "installed": installed,
        "native_or_not_loaded": not bool(installed),
        "state_identity_preserved": True,
        "before": before,
        "after": after,
    }


def uninstall_renderer_hooks() -> list[str]:
    """Restore functions replaced by :func:`install_renderer_hooks`."""

    restored = []
    spatial = sys.modules.get("behavior_interface.spatial_map")
    for owner, name in (
        (spatial, "render_minimap_rgba"),
        (spatial, "map_version"),
    ):
        current = getattr(owner, name, None) if owner is not None else None
        if bool(getattr(current, "_navigation_route_overlay_hook", False)):
            setattr(owner, name, current._navigation_route_overlay_original)
            restored.append(f"behavior_interface.spatial_map.{name}")
    live = sys.modules.get("behavior_interface.rtabmap_slam.live")
    live_mapper = getattr(live, "LiveMapper", None) if live is not None else None
    current = (
        getattr(live_mapper, "map_snapshot_png", None)
        if live_mapper is not None
        else None
    )
    if bool(getattr(current, "_navigation_route_overlay_hook", False)):
        live_mapper.map_snapshot_png = current._navigation_route_overlay_original
        restored.append(
            "behavior_interface.rtabmap_slam.live.LiveMapper.map_snapshot_png"
        )
    return restored


__all__ = [
    "RendererHookToken",
    "RouteOverlaySnapshot",
    "SCHEMA",
    "advance_route",
    "capture_renderer_hook_state",
    "clear_all_routes",
    "clear_route",
    "get_route_for_display",
    "get_route_snapshot",
    "install_renderer_hooks",
    "publish_route",
    "renderer_hooks_installed",
    "restore_renderer_hook_state",
    "route_version",
    "uninstall_renderer_hooks",
]
