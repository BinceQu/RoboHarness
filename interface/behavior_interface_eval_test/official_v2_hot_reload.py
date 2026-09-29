"""Transactional hot reload for the evaluator-only official-v2 tool stack.

The coordinator lives outside ``tool.official_v2`` so it can reload that
package without replacing the code currently performing the transaction.
No evaluator action is emitted here and all policy-owned runtime objects keep
their identity across a successful reload.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


_PACKAGE = "behavior_interface_eval_test.tool.official_v2"
_LIVE_MODULE = "behavior_interface_eval_test.live_move_tracked_point_test"
_NAVIGATION_MAP_BRIDGE_MODULE = (
    "behavior_interface_eval_test.navigation_map_bridge"
)
_NAVIGATION_ROUTE_OVERLAY_MODULE = (
    "behavior_interface_eval_test.navigation_route_overlay"
)

# Dependencies precede their consumers. The base-overlay/visualization cycle
# is intentionally loaded twice so both modules finish with fresh imports.
_RELOAD_MODULE_NAMES = (
    _NAVIGATION_MAP_BRIDGE_MODULE,
    f"{_PACKAGE}.predefined_eef_points_local",
    f"{_PACKAGE}.contract",
    f"{_PACKAGE}.capabilities",
    f"{_PACKAGE}.dispatch",
    f"{_PACKAGE}.depth_mesh_reconstruction",
    f"{_PACKAGE}.dynamic_point_tracker",
    f"{_PACKAGE}.eef_near_sam2_local",
    f"{_PACKAGE}.grasp_geometry_local",
    f"{_PACKAGE}.grasp_kinematics_local",
    f"{_PACKAGE}.grasp_prediction_audit_local",
    f"{_PACKAGE}.map_navigation_local",
    f"{_PACKAGE}.navigation_footprint_local",
    _NAVIGATION_ROUTE_OVERLAY_MODULE,
    f"{_PACKAGE}.mesh_occupancy",
    f"{_PACKAGE}.rgbd_projective_occupancy",
    f"{_PACKAGE}.rgbd_only_mesh_reconstruction",
    f"{_PACKAGE}.rgbd_scene_components",
    f"{_PACKAGE}.rgbd_scene_mesh_v8",
    f"{_PACKAGE}.rgbd_cuda_mesh_ops",
    f"{_PACKAGE}.rgbd_cuda_support_plane",
    f"{_PACKAGE}.rgbd_scene_mesh_v12",
    f"{_PACKAGE}.rgbd_thin_structure_v13",
    f"{_PACKAGE}.rgbd_scene_mesh_v13",
    f"{_PACKAGE}.rgbd_scene_mesh_v52",
    f"{_PACKAGE}.rgbd_scene_mesh_v53",
    f"{_PACKAGE}.rgbd_warp_occupancy",
    f"{_PACKAGE}.rgbd_grasp_planner",
    f"{_PACKAGE}.rgbd_grasp_lite",
    f"{_PACKAGE}.task_memory",
    f"{_PACKAGE}.trunk_vertical_lift_local",
    f"{_PACKAGE}.eef_adjustment_local",
    f"{_PACKAGE}.surface_facing_local",
    f"{_PACKAGE}.tracked_point_constraints_local",
    f"{_PACKAGE}.tracked_point_motion_local",
    f"{_PACKAGE}.tracked_point_execution_local",
    f"{_PACKAGE}.tracked_point_planning_worker",
    f"{_PACKAGE}.tracked_object_distance",
    f"{_PACKAGE}.base_path_overlay_local",
    f"{_PACKAGE}.visualization_local",
    f"{_PACKAGE}.base_path_overlay_local",
    f"{_PACKAGE}.visualization_local",
    f"{_PACKAGE}.human_track_object_distance_ui",
    f"{_PACKAGE}.types",
    f"{_PACKAGE}.tools",
    f"{_PACKAGE}.registry",
    _PACKAGE,
    _LIVE_MODULE,
)

# Reload must never import an unused optional feature merely because its source
# exists. Some official deployments intentionally omit dependencies for those
# features. Core protocol/motion modules are mandatory; optional modules are
# included only when the running process already imported them.
_MANDATORY_MODULE_NAMES = frozenset(
    {
        f"{_PACKAGE}.predefined_eef_points_local",
        f"{_PACKAGE}.contract",
        f"{_PACKAGE}.capabilities",
        f"{_PACKAGE}.dispatch",
        f"{_PACKAGE}.dynamic_point_tracker",
        f"{_PACKAGE}.grasp_geometry_local",
        f"{_PACKAGE}.grasp_kinematics_local",
        f"{_PACKAGE}.map_navigation_local",
        f"{_PACKAGE}.navigation_footprint_local",
        f"{_PACKAGE}.eef_adjustment_local",
        f"{_PACKAGE}.surface_facing_local",
        f"{_PACKAGE}.tracked_point_constraints_local",
        f"{_PACKAGE}.tracked_point_motion_local",
        f"{_PACKAGE}.tracked_point_execution_local",
        f"{_PACKAGE}.tracked_point_planning_worker",
        f"{_PACKAGE}.tracked_object_distance",
        f"{_PACKAGE}.human_track_object_distance_ui",
        f"{_PACKAGE}.task_memory",
        f"{_PACKAGE}.visualization_local",
        f"{_PACKAGE}.types",
        f"{_PACKAGE}.tools",
        f"{_PACKAGE}.registry",
        _PACKAGE,
        _LIVE_MODULE,
        _NAVIGATION_MAP_BRIDGE_MODULE,
        _NAVIGATION_ROUTE_OVERLAY_MODULE,
    }
)

_PERSISTENT_GLOBALS = {
    f"{_PACKAGE}.tools": ("_PLAN_INTEGRITY_KEY", "_SESSION_LOCK"),
    f"{_PACKAGE}.tracked_object_distance": (
        "_REPLAY_COMPRESSION_EXECUTOR",
    ),
    f"{_PACKAGE}.grasp_kinematics_local": (
        "_PERSISTENT_IK_WORKERS",
        "_PERSISTENT_IK_WORKERS_LOCK",
    ),
    f"{_PACKAGE}.rgbd_grasp_lite": (
        "_LITE_RENDER_EXECUTOR",
        "_LITE_RENDER_FUTURES",
        "_LITE_RENDER_EXECUTOR_LOCK",
    ),
    f"{_PACKAGE}.grasp_prediction_audit_local": (
        "_writers",
        "_writers_lock",
    ),
    _NAVIGATION_ROUTE_OVERLAY_MODULE: ("_LOCK", "_ROUTES", "_REVISION"),
}

_BACKGROUND_DRAIN_TIMEOUT_S = 15.0
_CAPTURE_WAIT_BRIDGE_ATTR = "_official_capture_wait_bridge"


def _enable_capture_runtime_markers(runtime: Any) -> bool:
    """Opt an already-created official runtime into detached capture.

    Older policy processes can load this coordinator through the live harness
    without reconstructing their world.  Only the observation-backed official
    world satisfies all three ownership markers below; simulator/unit worlds
    therefore retain their historical synchronous capture contract.
    """

    server = getattr(runtime, "server", None)
    world = getattr(server, "world", None)
    if world is None:
        return False
    if getattr(world, "_official_adapter", None) is None:
        return False
    if not callable(getattr(world, "_official_request_commit_guard", None)):
        return False
    if getattr(world, "_official_tracked_object_distances", None) is None:
        return False
    try:
        tools = importlib.import_module(f"{_PACKAGE}.tools")
    except Exception:
        return False
    # Do not turn on the marker until the currently committed module exposes
    # the resolver/cancellation API.  This keeps the first bootstrap request
    # compatible with an old tools generation; the following transaction then
    # enables the complete detached path atomically.
    if not all(
        callable(getattr(tools, name, None))
        for name in (
            "wait_for_capture_artifact",
            "cancel_capture_artifact",
            "capture_artifact_pending_count",
        )
    ):
        return False
    setattr(world, "_official_async_capture_artifacts", True)
    setattr(world, "_official_detached_capture_results", True)
    return True

_POLICY_REBINDS = {
    "MOVE_TRACKED_POINT_ORDER_DESCRIPTION": (
        "contract",
        "MOVE_TRACKED_POINT_ORDER_DESCRIPTION",
    ),
    "NAVIGATION_MAP_SCHEMA": (
        "navigation_map_bridge",
        "NAVIGATION_MAP_SCHEMA",
    ),
    "NAVIGATION_MAP_SCHEMA_VERSION": (
        "navigation_map_bridge",
        "NAVIGATION_MAP_SCHEMA_VERSION",
    ),
    "NavigationMapBridge": (
        "navigation_map_bridge",
        "NavigationMapBridge",
    ),
    "copy_navigation_map_snapshot": (
        "navigation_map_bridge",
        "copy_navigation_map_snapshot",
    ),
    "normalize_navigation_pose_freshness": (
        "navigation_map_bridge",
        "normalize_navigation_pose_freshness",
    ),
    "normalize_navigation_pose_source": (
        "navigation_map_bridge",
        "normalize_navigation_pose_source",
    ),
    "view_navigation_map_snapshot": (
        "navigation_map_bridge",
        "view_navigation_map_snapshot",
    ),
    "validate_move_tracked_point_args": ("contract", "validate_move_tracked_point_args"),
    "validate_navigate_to_args": ("contract", "validate_navigate_to_args"),
    "TrackedObjectDistanceMemory": ("tracked_object_distance", "TrackedObjectDistanceMemory"),
    "tracker_rgb_as_uint8": ("tracked_object_distance", "tracker_rgb_as_uint8"),
    "tracker_frame_from_allowed_observation": (
        "tracked_object_distance",
        "tracker_frame_from_allowed_observation",
    ),
    "PUBLIC_TOOLS": ("capabilities", "PUBLIC_TOOLS"),
    "OFFICIAL_TOOL_VERSION": ("capabilities", "TOOL_VERSION"),
    "WRIST_ROLL_TEST_TOOL_ENABLED": (
        "capabilities",
        "WRIST_ROLL_TEST_TOOL_ENABLED",
    ),
    "capability_report": ("capabilities", "capability_report"),
    "translate_submission": ("dispatch", "translate_submission"),
    "validate_submission": ("capabilities", "validate_submission"),
    "ensure_profile_installed": ("registry", "ensure_profile_installed"),
    "install_profile": ("registry", "install_profile"),
    "ADJUST_EEF_LOCAL_BUILD": ("tools", "ADJUST_EEF_LOCAL_BUILD"),
    "ADJUST_HEIGHT_IMPLEMENTATION_VERSION": (
        "tools",
        "ADJUST_HEIGHT_IMPLEMENTATION_VERSION",
    ),
    "CONTROL_HZ": ("tools", "CONTROL_HZ"),
    "GRIPPER_CLOSE_DEFAULT_TIMEOUT_S": (
        "tools",
        "GRIPPER_CLOSE_DEFAULT_TIMEOUT_S",
    ),
    "MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION": (
        "tools",
        "MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION",
    ),
    "RESET_BODY_IMPLEMENTATION_VERSION": (
        "tools",
        "RESET_BODY_IMPLEMENTATION_VERSION",
    ),
    "_camera_intrinsics": ("tools", "_camera_intrinsics"),
    "_json_ready": ("tools", "_json_ready"),
    "grasp_prep_hard_timeout_s": ("tools", "grasp_prep_hard_timeout_s"),
    "render_head_path_overlay_frame": (
        "visualization_local",
        "render_head_path_overlay_frame",
    ),
}

# These imports are consumed only while constructing a brand-new policy
# runtime or Flask application. Existing runtime objects/routes are migrated
# explicitly by this coordinator, so rebinding these names would not update a
# live closure and would give a false impression of coverage.
_POLICY_STARTUP_ONLY_IMPORTS = frozenset(
    {"install_official_task_memory", "install_track_object_distance_human_ui"}
)

_MISSING = object()
_TRANSACTION_STATE = threading.local()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: Any) -> str:
    def ready(item: Any) -> Any:
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, bytes):
            return {"bytes_hex": item.hex()}
        if isinstance(item, Mapping):
            return {
                str(key): ready(inner)
                for key, inner in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (list, tuple)):
            return [ready(inner) for inner in item]
        if isinstance(item, (set, frozenset)):
            return sorted((ready(inner) for inner in item), key=repr)
        tolist = getattr(item, "tolist", None)
        if callable(tolist):
            return ready(tolist())
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            fields = dict(values)
            # Runtime migrations may add this diagnostic field to track
            # objects created by an older module generation.  Its empty
            # default carries no observation state and is therefore
            # semantically identical to the field being absent.
            if fields.get("last_observation_rejections", _MISSING) == ():
                fields.pop("last_observation_rejections", None)
            deferred_component_fields = (
                "pending_component_depth_roi",
                "pending_component_origin_px",
                "pending_component_anchor_px",
                "pending_component_depth_m",
            )
            if (
                item.__class__.__name__ == "_TrackState"
                and all(
                    fields.get(name) is None
                    for name in deferred_component_fields
                )
            ):
                # Adding empty lazy-cache slots to a legacy live track does not
                # change its observation state. Once populated, the cached ROI
                # remains part of the digest and must survive future reloads.
                for name in deferred_component_fields:
                    fields.pop(name, None)
            return {
                "class": f"{item.__class__.__module__}.{item.__class__.__name__}",
                "fields": ready(fields),
            }
        return repr(item)

    encoded = json.dumps(
        ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _module_path(module: ModuleType) -> Path:
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not origin or not os.path.isfile(origin):
        raise RuntimeError(f"hot-reload module has no source file: {module.__name__}")
    return Path(origin)


def _source_record(module: ModuleType) -> dict[str, Any]:
    path = _module_path(module)
    return _path_source_record(path, name=module.__name__)


def _path_source_record(path: Path, *, name: str) -> dict[str, Any]:
    source = path.read_bytes()
    compile(source, str(path), "exec")
    return {
        "name": str(name),
        "path": str(path),
        "digest": _sha256_bytes(source),
    }


def _package_source_tree(
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    """Hash every Python source file in the official-v2 package.

    A file does not need to be imported yet to affect the stack identity. This
    prevents a newly added lazy helper from falling outside the generation
    receipt and makes source changes auditable before that helper's first use.
    """

    package_root = _module_path(_module_alias(modules, "contract")).parent.resolve()
    loaded_paths: dict[Path, str] = {}
    loaded_candidates = dict(modules)
    loaded_candidates.update(
        {
            str(name): module
            for name, module in tuple(sys.modules.items())
            if (
                str(name) == _PACKAGE or str(name).startswith(f"{_PACKAGE}.")
            )
            and isinstance(module, ModuleType)
        }
    )
    for name, module in loaded_candidates.items():
        try:
            loaded_paths[_module_path(module).resolve()] = str(name)
        except RuntimeError:
            continue

    worker_path = Path(
        str(getattr(_module_alias(modules, "grasp_kinematics_local"), "_WORKER_PATH"))
    ).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*.py")):
        resolved = path.resolve()
        record = _path_source_record(
            resolved,
            name=str(resolved.relative_to(package_root)),
        )
        if resolved == worker_path:
            execution = "subprocess"
        elif resolved in loaded_paths:
            execution = "in_process"
        else:
            execution = "lazy"
        files.append(
            {
                "relative_path": record["name"],
                "digest": record["digest"],
                "execution": execution,
                "module": loaded_paths.get(resolved),
            }
        )
    return {
        "root": str(package_root),
        "digest": _canonical_digest(
            [(item["relative_path"], item["digest"]) for item in files]
        ),
        "files": files,
    }


def _source_tree_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    before_files = {
        str(item.get("relative_path")): str(item.get("digest"))
        for item in list(before.get("files") or [])
    }
    after_files = {
        str(item.get("relative_path")): str(item.get("digest"))
        for item in list(after.get("files") or [])
    }
    result: list[str] = []
    for path in sorted(set(before_files) | set(after_files)):
        if path not in before_files:
            result.append(f"added:{path}")
        elif path not in after_files:
            result.append(f"removed:{path}")
        elif before_files[path] != after_files[path]:
            result.append(f"changed:{path}")
    return result


def _package_asset_tree(modules: Mapping[str, ModuleType]) -> dict[str, Any]:
    """Hash policy-shipped robot/UI assets used by official-v2 helpers."""

    package_root = _module_path(_module_alias(modules, "contract")).parent.resolve()
    asset_root = package_root / "assets"
    files: list[dict[str, str]] = []
    if asset_root.is_dir():
        for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": str(path.relative_to(package_root)),
                    "digest": _sha256_bytes(path.read_bytes()),
                }
            )
    return {
        "root": str(asset_root),
        "digest": _canonical_digest(
            [(item["relative_path"], item["digest"]) for item in files]
        ),
        "files": files,
    }


def _module_source_dependencies(
    module: ModuleType,
    selected_names: frozenset[str],
) -> set[str]:
    """Return statically visible selected-module dependencies for ordering."""

    try:
        tree = ast.parse(_module_path(module).read_bytes())
    except (OSError, SyntaxError, RuntimeError):
        return set()
    package_name = str(getattr(module, "__package__", "") or _PACKAGE)
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = str(alias.name)
                if imported in selected_names:
                    dependencies.add(imported)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * int(node.level) + str(node.module or "")
                try:
                    imported = importlib.util.resolve_name(relative, package_name)
                except (ImportError, ValueError):
                    continue
            else:
                imported = str(node.module or "")
            if imported in selected_names:
                dependencies.add(imported)
            for alias in node.names:
                candidate = f"{imported}.{alias.name}" if imported else str(alias.name)
                if candidate in selected_names:
                    dependencies.add(candidate)
    dependencies.discard(module.__name__)
    return dependencies


def _reload_order(modules: Mapping[str, ModuleType]) -> tuple[str, ...]:
    """Topologically order known and future loaded official-v2 helpers.

    The explicit manifest remains a stable priority for modules whose imports
    are intentionally dynamic. Source-visible dependencies determine the
    actual ordering. Strongly connected components are executed twice, which
    preserves the existing overlay/visualization cycle behavior without a
    hard-coded special case.
    """

    names = frozenset(modules)
    explicit = tuple(dict.fromkeys(_RELOAD_MODULE_NAMES))
    explicit_priority = {name: index for index, name in enumerate(explicit)}
    default_priority = len(explicit)

    dependencies = {
        name: _module_source_dependencies(module, names)
        for name, module in modules.items()
    }

    # Tarjan SCC keeps import cycles atomic in the condensation graph.
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def priority(name: str) -> tuple[int, str]:
        return (explicit_priority.get(name, default_priority), name)

    def visit(name: str) -> None:
        nonlocal index
        indices[name] = index
        lowlinks[name] = index
        index += 1
        stack.append(name)
        on_stack.add(name)
        for dependency in sorted(dependencies[name], key=priority):
            if dependency not in indices:
                visit(dependency)
                lowlinks[name] = min(lowlinks[name], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[name] = min(lowlinks[name], indices[dependency])
        if lowlinks[name] != indices[name]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == name:
                break
        components.append(tuple(sorted(component, key=priority)))

    for name in sorted(names, key=priority):
        if name not in indices:
            visit(name)

    component_by_name = {
        name: component_index
        for component_index, component in enumerate(components)
        for name in component
    }
    outgoing: dict[int, set[int]] = {index: set() for index in range(len(components))}
    indegree = {index: 0 for index in range(len(components))}
    for consumer, imported_names in dependencies.items():
        consumer_component = component_by_name[consumer]
        for dependency in imported_names:
            dependency_component = component_by_name[dependency]
            if dependency_component == consumer_component:
                continue
            if consumer_component not in outgoing[dependency_component]:
                outgoing[dependency_component].add(consumer_component)
                indegree[consumer_component] += 1

    def component_priority(component_index: int) -> tuple[int, str]:
        members = components[component_index]
        return min(priority(name) for name in members)

    ready = sorted(
        (item for item, degree in indegree.items() if degree == 0),
        key=component_priority,
    )
    ordered_components: list[int] = []
    while ready:
        current = ready.pop(0)
        ordered_components.append(current)
        for consumer in sorted(outgoing[current], key=component_priority):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
                ready.sort(key=component_priority)
    if len(ordered_components) != len(components):  # pragma: no cover - SCC invariant
        raise RuntimeError("official-v2 reload dependency graph is inconsistent")

    explicit_counts = {
        name: _RELOAD_MODULE_NAMES.count(name) for name in names
    }
    result: list[str] = []
    for component_index in ordered_components:
        component = components[component_index]
        repeat = max(
            2 if len(component) > 1 else 1,
            max((explicit_counts.get(name, 1) for name in component), default=1),
        )
        for _ in range(repeat):
            result.extend(component)
    return tuple(result)


def _loaded_official_v2_modules(
    package_root: Path,
) -> dict[str, ModuleType]:
    """Discover future helpers already resident in the policy process."""

    result: dict[str, ModuleType] = {}
    prefix = f"{_PACKAGE}."
    for name, candidate in tuple(sys.modules.items()):
        if not str(name).startswith(prefix) or not isinstance(candidate, ModuleType):
            continue
        try:
            path = _module_path(candidate).resolve()
            path.relative_to(package_root)
        except (RuntimeError, ValueError):
            continue
        if path.name == "ik_filter_worker.py":
            continue
        result[str(name)] = candidate
    return result


def _ik_worker_source_record(modules: Mapping[str, ModuleType]) -> dict[str, Any]:
    kinematics = _module_alias(modules, "grasp_kinematics_local")
    path = Path(str(getattr(kinematics, "_WORKER_PATH")))
    return _path_source_record(path, name=f"{_PACKAGE}.ik_filter_worker")


def _invalidate_ik_workers(modules: Mapping[str, ModuleType]) -> dict[str, Any]:
    """Close old-code subprocesses at the final, non-rollback commit point."""

    kinematics = _module_alias(modules, "grasp_kinematics_local")
    workers = kinematics._PERSISTENT_IK_WORKERS
    worker_lock = kinematics._PERSISTENT_IK_WORKERS_LOCK
    with worker_lock:
        stale_workers = list(workers.values())
        workers.clear()
    errors: list[str] = []
    for worker in stale_workers:
        try:
            worker.close()
        except Exception as exc:  # pragma: no cover - defensive cleanup only
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "invalidated_count": len(stale_workers),
        "close_errors": errors,
    }


def _wait_background_futures(
    futures: list[Any],
    *,
    label: str,
    timeout_s: float = _BACKGROUND_DRAIN_TIMEOUT_S,
) -> None:
    """Wait for already-submitted old-generation work before module mutation."""

    deadline = time.monotonic() + float(timeout_s)
    for future in futures:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError(f"timed out draining {label} before hot reload")
        try:
            future.result(timeout=remaining)
        except TimeoutError as exc:
            raise RuntimeError(
                f"timed out draining {label} before hot reload"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{label} failed before hot reload: {type(exc).__name__}: {exc}"
            ) from exc


def _drain_background_tasks(
    runtime: Any,
    modules: Mapping[str, ModuleType],
) -> dict[str, dict[str, int]]:
    """Finish module-owned futures while their original code is still intact."""

    manager = getattr(runtime, "tracked_object_distances", None)
    replay_pending = list(getattr(manager, "_pending_replay", ()) or ())
    _wait_background_futures(
        [item.future for item in replay_pending],
        label="tracked RGB-D replay compression",
    )
    replay_drain = getattr(manager, "_drain_ready_replay_locked", None)
    if callable(replay_drain):
        replay_drain()
    replay_remaining = len(getattr(manager, "_pending_replay", ()) or ())
    if replay_remaining:
        raise RuntimeError(
            "tracked RGB-D replay compression did not drain before hot reload"
        )

    lite = modules.get(f"{_PACKAGE}.rgbd_grasp_lite")
    lite_pending: list[Any] = []
    lite_remaining = 0
    if lite is not None:
        lock = getattr(lite, "_LITE_RENDER_EXECUTOR_LOCK", None)
        with lock if lock is not None else nullcontext():
            futures = getattr(lite, "_LITE_RENDER_FUTURES", None)
            if isinstance(futures, list):
                lite_pending = list(futures)
        _wait_background_futures(
            lite_pending,
            label="RGB-D lite rendering",
        )
        drained_ids = {id(item) for item in lite_pending}
        with lock if lock is not None else nullcontext():
            futures = getattr(lite, "_LITE_RENDER_FUTURES", None)
            if isinstance(futures, list):
                futures[:] = [item for item in futures if id(item) not in drained_ids]
                lite_remaining = len(futures)
        if lite_remaining:
            raise RuntimeError(
                "RGB-D lite rendering accepted new work during hot reload"
            )

    return {
        "replay_compression": {
            "drained_count": len(replay_pending),
            "pending_count_after": replay_remaining,
        },
        "lite_render": {
            "drained_count": len(lite_pending),
            "pending_count_after": lite_remaining,
        },
    }


def _close_prediction_audit_writers(
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    """Close old-generation raw file descriptors at the final commit point."""

    audit = modules.get(f"{_PACKAGE}.grasp_prediction_audit_local")
    close_all = getattr(audit, "close_prediction_audit_writers", None)
    if not callable(close_all):
        return {"closed_count": 0, "close_errors": []}
    report = dict(close_all() or {})
    return {
        "closed_count": int(report.get("closed_count", 0)),
        "close_errors": [str(item) for item in report.get("close_errors", [])],
    }


def _observation_sequence(runtime: Any) -> int | None:
    provider = getattr(getattr(runtime, "adapter", None), "observation_metadata", None)
    if not callable(provider):
        return None
    try:
        metadata = provider()
    except Exception:
        return None
    if isinstance(metadata, Mapping):
        raw = metadata.get("sequence", metadata.get("observation_sequence"))
    elif isinstance(metadata, (tuple, list)) and metadata:
        raw = metadata[0]
    else:
        raw = None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _episode_id(runtime: Any) -> str | None:
    world = getattr(getattr(runtime, "server", None), "world", None)
    provider = getattr(world, "episode_id", None)
    try:
        value = provider() if callable(provider) else provider
    except Exception:
        return None
    return None if value is None else str(value)


def _tracked_names(manager: Any) -> list[str]:
    entries = getattr(manager, "_entries", None)
    if isinstance(entries, Mapping):
        return [str(name) for name in entries]
    names = getattr(manager, "active_names", None)
    if isinstance(names, (tuple, list)):
        return [str(name) for name in names]
    status = getattr(manager, "status", None)
    if callable(status):
        try:
            report = status()
            raw = report.get("names", report.get("registered_names", []))
            return [str(name) for name in list(raw or [])]
        except Exception:
            pass
    return []


def _tracked_state_digest(manager: Any) -> str | None:
    if manager is None:
        return None
    state = {
        name: deepcopy(getattr(manager, name, None))
        for name in (
            "_entries",
            "_registration_entries",
            "_name_by_track_id",
            "_capture_bindings",
            "_registration_binding",
            "_active_rigid_pair",
            "_active_rigid_pair_report",
            "_active_on_hand_eef_prior",
            "_motion_retention_leases",
        )
    }
    inner = getattr(manager, "_tracker", None)
    inner_defaults = {
        "_episode_id": None,
        "_frame": None,
        "_tracks": {},
        "_active_rigid_pair": None,
        "_last_rigid_pair_report": {},
        "_next_track_number": 1,
    }
    state["inner"] = {
        name: deepcopy(getattr(inner, name, default))
        for name, default in inner_defaults.items()
    }
    return _canonical_digest(state)


def _task_memory_state_digest(memory: Any) -> str | None:
    if memory is None:
        return None
    public_fields = getattr(memory, "public_fields", None)
    if callable(public_fields):
        try:
            return _canonical_digest(public_fields())
        except Exception:
            pass
    return _canonical_digest(getattr(memory, "__dict__", memory))


def _array_state(value: Any) -> dict[str, Any] | None:
    """Return a compact identity/content record for a NumPy-like array."""

    if value is None or not all(
        hasattr(value, name) for name in ("shape", "dtype", "tobytes")
    ):
        return None
    try:
        raw = value.tobytes(order="C")
        shape = [int(item) for item in tuple(value.shape)]
    except Exception:
        return None
    return {
        "identity": id(value),
        "shape": shape,
        "dtype": str(value.dtype),
        "digest": _sha256_bytes(raw),
    }


def _grid_state(grid: Any) -> dict[str, Any] | None:
    if grid is None:
        return None
    arrays = {
        str(name): record
        for name, value in vars(grid).items()
        if not str(name).startswith("_")
        if (record := _array_state(value)) is not None
    }
    scalars = {
        str(name): value
        for name, value in vars(grid).items()
        if not str(name).startswith("_")
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        "identity": id(grid),
        "arrays": arrays,
        "scalars": scalars,
    }


def _egomap_state() -> dict[str, Any]:
    """Fingerprint resident EgoMaps without importing or reloading production."""

    module = sys.modules.get("behavior_interface.spatial_map")
    maps = getattr(module, "_MAPS", None)
    if not isinstance(maps, Mapping):
        return {
            "maps_identity": None,
            "map_object_identities": [],
            "content_digest": None,
        }
    records: list[dict[str, Any]] = []
    for session_id, ego in sorted(maps.items(), key=lambda item: str(item[0])):
        submaps = []
        for submap in list(getattr(ego, "submaps", ()) or ()):
            submaps.append(
                {
                    "identity": id(submap),
                    "serial": getattr(submap, "serial", None),
                    "origin": [
                        getattr(submap, "origin_x", None),
                        getattr(submap, "origin_y", None),
                        getattr(submap, "origin_yaw_deg", None),
                    ],
                    "finished": getattr(submap, "finished", None),
                    "grid": _grid_state(getattr(submap, "grid", None)),
                    "wall_dir_hist": _array_state(
                        getattr(submap, "wall_dir_hist", None)
                    ),
                }
            )
        records.append(
            {
                "session_id": str(session_id),
                "identity": id(ego),
                "pose": [
                    getattr(ego, "x", None),
                    getattr(ego, "y", None),
                    getattr(ego, "yaw_deg", None),
                ],
                "initialized": getattr(ego, "initialized", None),
                "update_count": getattr(ego, "update_count", None),
                "mapping_state": getattr(ego, "mapping_state", None),
                "grid": _grid_state(getattr(ego, "grid", None)),
                "wall_dir_hist": _array_state(
                    getattr(ego, "wall_dir_hist", None)
                ),
                "submaps": submaps,
                "vectors_digest": _canonical_digest(
                    {
                        "trail": getattr(ego, "trail", ()),
                        "trajectory_samples": getattr(
                            ego, "trajectory_samples", ()
                        ),
                        "landmarks": getattr(ego, "landmarks", {}),
                        "events": getattr(ego, "events", ()),
                        "places": getattr(ego, "places", ()),
                        "pose_edges": getattr(ego, "pose_edges", ()),
                    }
                ),
            }
        )
    return {
        "maps_identity": id(maps),
        "map_object_identities": [item["identity"] for item in records],
        "content_digest": _canonical_digest(records),
    }


def _navigation_snapshot_state(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {
            "snapshot_identity": None,
            "occupancy_identity": None,
            "occupancy_digest": None,
        }
    occupancy = _array_state(snapshot.get("occupancy"))
    return {
        "snapshot_identity": id(snapshot),
        "occupancy_identity": (
            None if occupancy is None else occupancy["identity"]
        ),
        "occupancy_digest": (
            None if occupancy is None else occupancy["digest"]
        ),
    }


def _navigation_provider_state(provider: Any) -> dict[str, Any]:
    latest = getattr(provider, "_latest", None)
    if latest is None:
        return {
            "identity": None if provider is None else id(provider),
            "latest_identity": None,
            "content_digest": None,
        }
    arrays = {
        name: _array_state(getattr(latest, name, None))
        for name in ("occupancy", "low_obstacles", "high_obstacles")
    }
    content = {
        "frame_id": getattr(latest, "frame_id", None),
        "map_updated": getattr(latest, "map_updated", None),
        "node_count": getattr(latest, "node_count", None),
        "loop_count": getattr(latest, "loop_count", None),
        "current_pose": getattr(latest, "current_pose", None),
        "arrays": arrays,
    }
    return {
        "identity": id(provider),
        "latest_identity": id(latest),
        "content_digest": _canonical_digest(content),
    }


def _navigation_state_identity(runtime: Any) -> dict[str, Any]:
    adapter = getattr(runtime, "adapter", None)
    bridge = getattr(runtime, "_navigation_map_bridge", None)
    provider = getattr(runtime, "_spatial_live_mapper", None)
    adapter_map = getattr(adapter, "_navigation_map", None)
    bridge_snapshot = getattr(bridge, "_cached_snapshot", None)
    adapter_state = _navigation_snapshot_state(adapter_map)
    bridge_state = _navigation_snapshot_state(bridge_snapshot)
    ego_state = _egomap_state()
    provider_state = _navigation_provider_state(provider)
    return {
        "navigation_bridge_identity": None if bridge is None else id(bridge),
        "navigation_provider_identity": provider_state["identity"],
        "navigation_provider_latest_identity": provider_state["latest_identity"],
        "navigation_provider_content_digest": provider_state["content_digest"],
        "navigation_adapter_map_identity": adapter_state["snapshot_identity"],
        "navigation_adapter_occupancy_identity": adapter_state[
            "occupancy_identity"
        ],
        "navigation_adapter_occupancy_digest": adapter_state[
            "occupancy_digest"
        ],
        "navigation_bridge_snapshot_identity": bridge_state["snapshot_identity"],
        "navigation_bridge_occupancy_identity": bridge_state[
            "occupancy_identity"
        ],
        "navigation_bridge_occupancy_digest": bridge_state["occupancy_digest"],
        "navigation_egomaps_container_identity": ego_state["maps_identity"],
        "navigation_egomap_identities": ego_state["map_object_identities"],
        "navigation_egomap_content_digest": ego_state["content_digest"],
    }


def _navigation_state_locks(runtime: Any) -> tuple[Any, ...]:
    """Locks which make map-state fingerprints stable during a transaction."""

    adapter = getattr(runtime, "adapter", None)
    bridge = getattr(runtime, "_navigation_map_bridge", None)
    provider = getattr(runtime, "_spatial_live_mapper", None)
    candidates = (
        getattr(adapter, "_lock", None),
        getattr(bridge, "_lock", None),
        getattr(provider, "_state_lock", None),
    )
    result: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        if not hasattr(candidate, "__enter__"):
            continue
        seen.add(id(candidate))
        result.append(candidate)
    return tuple(result)


def _state_identity(runtime: Any) -> dict[str, Any]:
    manager = getattr(runtime, "tracked_object_distances", None)
    world = getattr(getattr(runtime, "server", None), "world", None)
    adapter = getattr(runtime, "adapter", None)
    task_memory = getattr(runtime, "task_memory", None)
    bindings = getattr(manager, "_capture_bindings", None)
    last_action = getattr(adapter, "_last_action", None)
    if hasattr(last_action, "tolist"):
        last_action = last_action.tolist()
    epoch_provider = getattr(world, "motion_epoch", None)
    try:
        motion_epoch = (
            epoch_provider()
            if callable(epoch_provider)
            else getattr(world, "_motion_epoch", None)
        )
    except Exception:
        motion_epoch = None
    state = {
        "episode_id": _episode_id(runtime),
        "observation_sequence": _observation_sequence(runtime),
        "policy_motion_epoch": motion_epoch,
        "last_action": deepcopy(last_action),
        "manager_identity": None if manager is None else id(manager),
        "tracked_names": _tracked_names(manager),
        "tracked_state_digest": _tracked_state_digest(manager),
        "capture_bindings": 0 if bindings is None else len(bindings),
        "task_memory_identity": None if task_memory is None else id(task_memory),
        "task_memory_digest": _task_memory_state_digest(task_memory),
    }
    state.update(_navigation_state_identity(runtime))
    return state


def _module_alias(modules: Mapping[str, ModuleType], basename: str) -> ModuleType:
    if basename == "navigation_map_bridge":
        return modules[_NAVIGATION_MAP_BRIDGE_MODULE]
    return modules[f"{_PACKAGE}.{basename}"]


def _builds(modules: Mapping[str, ModuleType]) -> dict[str, Any]:
    tools = _module_alias(modules, "tools")
    tracker = _module_alias(modules, "tracked_object_distance")
    dynamic = _module_alias(modules, "dynamic_point_tracker")
    live = modules[_LIVE_MODULE]
    return {
        "move_tracked_point": str(
            getattr(tools, "MOVE_TRACKED_POINT_LOCAL_BUILD", "unknown")
        ),
        "live_runner": str(getattr(live, "BUILD", "unknown")),
        "contract": _source_record(_module_alias(modules, "contract"))["digest"],
        "capabilities": _source_record(_module_alias(modules, "capabilities"))["digest"],
        "tracker": str(
            getattr(tracker, "TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD", "unknown")
        ),
        "dynamic_tracker": str(
            getattr(dynamic, "DYNAMIC_POINT_TRACKER_BUILD", "unknown")
        ),
        "ui": _canonical_digest(_asset_revision(modules)),
    }


def _trajectory_identity(modules: Mapping[str, ModuleType]) -> dict[str, Any]:
    tools = _module_alias(modules, "tools")
    return {
        "schema": str(
            getattr(tools, "MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA", "unknown")
        ),
        "version": int(
            getattr(tools, "MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION", 0)
        ),
    }


def _navigation_reload_report(
    modules: Mapping[str, ModuleType],
    registry: Any,
    *,
    runtime: Any,
    bridge_class_before: type | None,
    bridge_class_rebound: bool,
    callback_ids_before: Mapping[str, int],
    previous_receipt: Any,
    reload_names: tuple[str, ...],
    renderer_hooks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit that the published navigation stack is one coherent generation."""

    tools = _module_alias(modules, "tools")
    planner = _module_alias(modules, "map_navigation_local")
    footprint = _module_alias(modules, "navigation_footprint_local")
    bridge_module = modules[_NAVIGATION_MAP_BRIDGE_MODULE]
    overlay = modules[_NAVIGATION_ROUTE_OVERLAY_MODULE]
    policy = sys.modules.get(
        "behavior_interface_eval_test.official_policy_interface"
    )
    bridge = getattr(runtime, "_navigation_map_bridge", None)
    bridge_class_after = None if bridge is None else bridge.__class__
    bridge_target_class = bridge_module.NavigationMapBridge
    policy_bridge_names = (
        "NAVIGATION_MAP_SCHEMA",
        "NAVIGATION_MAP_SCHEMA_VERSION",
        "NavigationMapBridge",
        "copy_navigation_map_snapshot",
        "normalize_navigation_pose_freshness",
        "normalize_navigation_pose_source",
        "view_navigation_map_snapshot",
    )
    bindings = {
        "planner": (
            getattr(tools, "plan_clearance_path", None)
            is getattr(planner, "plan_clearance_path", None)
        ),
        "footprint": (
            getattr(tools, "base_navigation_footprint_envelope", None)
            is getattr(footprint, "base_navigation_footprint_envelope", None)
        ),
        "legacy_footprint_binding": (
            getattr(tools, "whole_body_footprint_envelope", None)
            is getattr(footprint, "whole_body_footprint_envelope", None)
        ),
        "registry": (
            getattr(registry.get("navigate_to"), "fn", None)
            is getattr(tools, "navigate_to", None)
        ),
        "policy_bridge": bool(
            policy is not None
            and all(
                getattr(policy, name, _MISSING)
                is getattr(bridge_module, name, _MISSING)
                for name in policy_bridge_names
            )
        ),
        "runtime_bridge": (
            bridge is None
            or not _is_navigation_bridge_class(bridge_class_before)
            or bridge_class_after is bridge_target_class
        ),
    }
    if not all(bindings.values()):
        stale = ", ".join(
            name for name, current in bindings.items() if not current
        )
        raise RuntimeError(
            "hot-reload navigation binding retained stale code: " + stale
        )

    source_modules = {
        "contract": _module_alias(modules, "contract"),
        "map_bridge": bridge_module,
        "planner": planner,
        "footprint": footprint,
        "executor": tools,
        "route_overlay": overlay,
    }
    source_digests = {
        label: _source_record(module)["digest"]
        for label, module in source_modules.items()
    }
    previous_navigation = (
        dict(previous_receipt.get("navigation") or {})
        if isinstance(previous_receipt, Mapping)
        else {}
    )
    previous_sources = dict(
        previous_navigation.get("source_digests") or {}
    )
    reloaded_names = set(reload_names)
    tools_index = reload_names.index(f"{_PACKAGE}.tools")
    dependency_names = (
        _NAVIGATION_MAP_BRIDGE_MODULE,
        f"{_PACKAGE}.map_navigation_local",
        f"{_PACKAGE}.navigation_footprint_local",
        _NAVIGATION_ROUTE_OVERLAY_MODULE,
    )
    callback_after = _registry_callback_ids(registry).get("navigate_to")
    callback_before = callback_ids_before.get("navigate_to")
    return {
        "source_digests": source_digests,
        "source_changed_since_previous_commit": {
            label: previous_sources.get(label) != digest
            for label, digest in source_digests.items()
        },
        "reloaded_modules": [module.__name__ for module in source_modules.values()],
        "dependencies_before_executor": all(
            name in reloaded_names and reload_names.index(name) < tools_index
            for name in dependency_names
        ),
        "bindings_current": bindings,
        "callback_id_before": callback_before,
        "callback_id_after": callback_after,
        "callback_replaced": (
            callback_before is None or callback_after != callback_before
        ),
        "bridge": {
            "instance_id": None if bridge is None else id(bridge),
            "class_id_before": (
                None if bridge_class_before is None else id(bridge_class_before)
            ),
            "class_id_after": (
                None if bridge_class_after is None else id(bridge_class_after)
            ),
            "class_rebound": bool(bridge_class_rebound),
            "class_current": bindings["runtime_bridge"],
        },
        "renderer_hooks": dict(renderer_hooks or {}),
    }


def _asset_revision(modules: Mapping[str, ModuleType]) -> dict[str, str]:
    ui = _module_alias(modules, "human_track_object_distance_ui")
    root = Path(getattr(ui, "_ASSET_DIR"))
    result: dict[str, str] = {}
    for name in (
        "track_object_distance_human_ui.js",
        "track_object_distance_human_ui.css",
        "move_tracked_point_human_ui.js",
        "move_tracked_point_human_ui.css",
    ):
        path = root / name
        result[name] = _sha256_bytes(path.read_bytes())
    return result


def _metadata_revision(modules: Mapping[str, ModuleType]) -> str:
    contract = _module_alias(modules, "contract")
    capabilities = _module_alias(modules, "capabilities")
    return _canonical_digest(
        {
            "public_tools": list(getattr(capabilities, "PUBLIC_TOOLS", ())),
            "capabilities": {
                name: getattr(capabilities, "TOOL_CAPABILITIES", {}).get(
                    name, {}
                )
                for name in ("move_tracked_point", "navigate_to")
            },
            "max_points": getattr(contract, "MOVE_TRACKED_POINT_MAX_POINTS", None),
            "quick_max_points": getattr(
                contract, "MOVE_TRACKED_POINT_QUICK_MAX_POINTS", None
            ),
            "max_steps": getattr(contract, "MOVE_TRACKED_POINT_MAX_STEPS", None),
            "max_timeout_s": getattr(
                contract, "MOVE_TRACKED_POINT_MAX_TIMEOUT_S", None
            ),
        }
    )


def _registry_callback_ids(registry: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if isinstance(registry, Mapping):
        for name, spec in registry.items():
            callback = getattr(spec, "fn", None)
            result[str(name)] = id(callback)
    return result


def _ensure_runtime_fields(runtime: Any) -> None:
    if not hasattr(runtime, "_official_reload_lock"):
        runtime._official_reload_lock = threading.RLock()
    if not hasattr(runtime, "_official_reload_generation"):
        runtime._official_reload_generation = 0
    if not hasattr(runtime, "_official_reload_receipt"):
        runtime._official_reload_receipt = None


def _ensure_submission_barrier(runtime: Any) -> None:
    """Linearize submissions for a process created before reload support."""

    server = getattr(runtime, "server", None)
    submit = getattr(server, "submit_skill", None)
    if callable(submit) and not bool(
        getattr(server, "_official_reload_submit_barrier", False)
    ):
        def submit_with_reload_barrier(*args: Any, **kwargs: Any):
            with runtime._official_reload_lock:
                return submit(*args, **kwargs)

        server.submit_skill = submit_with_reload_barrier
        server._official_reload_submit_barrier = True


def _check_idle(runtime: Any) -> None:
    server = runtime.server
    if getattr(server, "current_job", None) is not None:
        raise RuntimeError("cannot reload while a normal skill is running")
    pending = getattr(server, "skill_queue", None)
    if pending is not None:
        try:
            if not pending.empty():
                raise RuntimeError("cannot reload while a normal skill is queued")
        except AttributeError:
            pass
    # Detached captures have released ``current_job`` already, but their
    # immutable media worker still owns the result token.  Reloading the tools
    # module at that point would retire its executor and strand the HTTP waiter.
    tools_module = sys.modules.get(f"{_PACKAGE}.tools")
    pending_capture_count = getattr(
        tools_module,
        "capture_artifact_pending_count",
        None,
    )
    if callable(pending_capture_count):
        try:
            if int(pending_capture_count()) > 0:
                raise RuntimeError(
                    "cannot reload while a detached capture artifact is running"
                )
        except RuntimeError:
            raise
        except Exception:
            # A diagnostic-only query must not make an otherwise valid reload
            # fail when an old tools generation lacks the helper.
            pass
    job = getattr(runtime, "_live_test_job", None)
    if job is None:
        return
    state = str(job.status().get("state") or "")
    if state not in {"done", "failed", "cancelled"}:
        raise RuntimeError("cannot reload while a live test is running")
    future = getattr(job, "_future", None)
    if future is not None and not future.done():
        raise RuntimeError(
            "cannot reload while the live test planner thread is still stopping"
        )


def reload_live_test_harness(
    runtime: Any,
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    reload_module: Callable[[ModuleType], ModuleType] = importlib.reload,
    invalidate_caches: Callable[[], None] = importlib.invalidate_caches,
) -> ModuleType:
    """Return the harness bound to the current committed generation.

    Before the first transaction this retains the legacy one-module bootstrap
    needed by an already-running process. After generation 1, only the full
    transaction may execute new harness bytes; per-case dev starts merely
    verify and reuse the committed module.
    """

    _ensure_runtime_fields(runtime)
    with runtime._official_reload_lock:
        with _runtime_lock(runtime, "_live_test_lock"):
            job = getattr(runtime, "_live_test_job", None)
            if job is not None:
                state = str(job.status().get("state") or "")
                if state not in {"done", "failed", "cancelled"}:
                    raise RuntimeError(
                        "cannot reload live test module while a live test is running"
                    )
                future = getattr(job, "_future", None)
                if future is not None and not future.done():
                    raise RuntimeError(
                        "cannot reload live test module while its planner thread "
                        "is still stopping"
                    )

            invalidate_caches()
            module_name = str(
                getattr(runtime, "_live_test_module_name", _LIVE_MODULE)
            )
            module = import_module(module_name)
            generation = int(getattr(runtime, "_official_reload_generation", 0))
            receipt = getattr(runtime, "_official_reload_receipt", None)
            if generation > 0 and isinstance(receipt, Mapping):
                expected_digest = next(
                    (
                        str(item.get("new_file_digest"))
                        for item in list(receipt.get("modules") or [])
                        if str(item.get("name")) == module_name
                    ),
                    None,
                )
                current_digest = _source_record(module)["digest"]
                if not expected_digest or current_digest != expected_digest:
                    raise RuntimeError(
                        "live test harness source differs from the committed "
                        "official-v2 generation; run the transactional reload first"
                    )
                runtime._live_test_module_build = str(
                    getattr(module, "BUILD", "unknown")
                )
                return module

            previous_build = getattr(runtime, "_live_test_module_build", None)
            snapshot = dict(module.__dict__)
            try:
                module = _reload_one(module, reload_module)
                runtime._live_test_module_build = str(
                    getattr(module, "BUILD", "unknown")
                )
                return module
            except Exception:
                module.__dict__.clear()
                module.__dict__.update(snapshot)
                runtime._live_test_module_build = previous_build
                raise


class _InPlaceStateSnapshot:
    """Snapshot mutable tracker state while retaining every object identity."""

    _STATE_OBJECT_NAMES = frozenset(
        {
            "TrackedObjectDistanceMemory",
            "DynamicPointTracker",
            "_TrackState",
            "OfficialTaskMemory",
        }
    )

    def __init__(self, root: Any) -> None:
        self._memo: dict[int, dict[str, Any]] = {}
        self._root = self._capture(root, force_object=True)

    @staticmethod
    def _is_state_object(value: Any) -> bool:
        cls = value.__class__
        return (
            cls.__name__ in _InPlaceStateSnapshot._STATE_OBJECT_NAMES
            and cls.__module__.startswith(_PACKAGE)
        )

    @staticmethod
    def _is_numpy_array(value: Any) -> bool:
        cls = value.__class__
        return cls.__module__.startswith("numpy") and hasattr(value, "copy")

    def _capture(self, value: Any, *, force_object: bool = False) -> dict[str, Any]:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return {"kind": "ref", "value": value}

        identity = id(value)
        existing = self._memo.get(identity)
        if existing is not None:
            return existing

        if self._is_numpy_array(value):
            node = {"kind": "array", "value": value, "saved": value.copy()}
            self._memo[identity] = node
            return node
        if isinstance(value, dict):
            node = {"kind": "dict", "value": value, "items": []}
            self._memo[identity] = node
            node["items"] = [
                (key, self._capture(inner)) for key, inner in value.items()
            ]
            return node
        if isinstance(value, list):
            node = {"kind": "list", "value": value, "items": []}
            self._memo[identity] = node
            node["items"] = [self._capture(inner) for inner in value]
            return node
        if isinstance(value, deque):
            node = {
                "kind": "deque",
                "value": value,
                "maxlen": value.maxlen,
                "items": [],
            }
            self._memo[identity] = node
            node["items"] = [self._capture(inner) for inner in value]
            return node
        if isinstance(value, set):
            node = {"kind": "set", "value": value, "items": []}
            self._memo[identity] = node
            node["items"] = [self._capture(inner) for inner in value]
            return node
        if isinstance(value, tuple):
            node = {"kind": "tuple", "value": value, "items": []}
            self._memo[identity] = node
            node["items"] = [self._capture(inner) for inner in value]
            return node
        values = getattr(value, "__dict__", None)
        if isinstance(values, dict) and (force_object or self._is_state_object(value)):
            node = {"kind": "object", "value": value, "attrs": {}}
            self._memo[identity] = node
            node["attrs"] = {
                name: self._capture(inner) for name, inner in values.items()
            }
            return node
        return {"kind": "ref", "value": value}

    def restore(self) -> Any:
        restored: set[int] = set()

        def apply(node: dict[str, Any]) -> Any:
            kind = node["kind"]
            value = node["value"]
            if kind == "ref":
                return value
            identity = id(node)
            if identity in restored:
                return value
            restored.add(identity)
            if kind == "array":
                saved = node["saved"]
                if getattr(value, "shape", None) != getattr(saved, "shape", None):
                    raise RuntimeError("cannot restore resized tracker array in place")
                value[...] = saved
            elif kind == "dict":
                value.clear()
                for key, child in node["items"]:
                    value[key] = apply(child)
            elif kind == "list":
                value[:] = [apply(child) for child in node["items"]]
            elif kind == "deque":
                value.clear()
                value.extend(apply(child) for child in node["items"])
            elif kind == "set":
                value.clear()
                value.update(apply(child) for child in node["items"])
            elif kind == "tuple":
                for child in node["items"]:
                    apply(child)
            elif kind == "object":
                attrs = value.__dict__
                attrs.clear()
                attrs.update(
                    {name: apply(child) for name, child in node["attrs"].items()}
                )
            return value

        return apply(self._root)


def _runtime_lock(runtime: Any, name: str):
    value = getattr(runtime, name, None)
    return value if value is not None else nullcontext()


def _server_lock(runtime: Any):
    value = getattr(getattr(runtime, "server", None), "skill_lock", None)
    return value if value is not None else nullcontext()


def _manager_lock(runtime: Any):
    manager = getattr(runtime, "tracked_object_distances", None)
    value = getattr(manager, "_lock", None)
    return value if value is not None else nullcontext()


def _restore_modules(
    module_snapshots: Mapping[ModuleType, dict[str, Any]],
) -> None:
    for module, values in reversed(tuple(module_snapshots.items())):
        module.__dict__.clear()
        module.__dict__.update(values)


def _reload_one(
    module: ModuleType,
    reload_module: Callable[[ModuleType], ModuleType],
) -> ModuleType:
    # Do not delegate the production path to importlib.reload(): timestamp
    # based pyc validation can execute stale bytecode when an edit keeps the
    # same file size and mtime second. Execute the already validated source
    # directly into the existing module object so references held by callers
    # remain valid while every function/class binding is refreshed.
    if not isinstance(module, ModuleType):
        return reload_module(module)
    if reload_module is importlib.reload or (
        getattr(reload_module, "__module__", "") == "importlib"
        and getattr(reload_module, "__name__", "") == "reload"
    ):
        path = _module_path(module)
        source = path.read_bytes()
        code = compile(source, str(path), "exec")
        keep = {
            name: module.__dict__[name]
            for name in (
                "__name__",
                "__loader__",
                "__package__",
                "__spec__",
                "__path__",
                "__file__",
                "__cached__",
                "__builtins__",
            )
            if name in module.__dict__
        }
        # Stateful policy helpers deliberately guard their module globals with
        # ``if name not in globals()``.  Preserve those values while executing
        # the validated source so readers never observe a transient empty
        # route/session between reload and the later commit restoration.
        for name in _PERSISTENT_GLOBALS.get(module.__name__, ()):
            if name in module.__dict__:
                keep[name] = module.__dict__[name]
        module.__dict__.clear()
        module.__dict__.update(keep)
        exec(code, module.__dict__)
        return module
    # Injected reload functions are used by rollback and route-contract tests.
    return reload_module(module)


def _migrate_tracker(runtime: Any, modules: Mapping[str, ModuleType]) -> None:
    manager = getattr(runtime, "tracked_object_distances", None)
    if manager is None:
        return
    tracked_module = _module_alias(modules, "tracked_object_distance")
    dynamic_module = _module_alias(modules, "dynamic_point_tracker")
    target_manager_class = tracked_module.TrackedObjectDistanceMemory
    target_tracker_class = dynamic_module.DynamicPointTracker
    if (
        manager.__class__.__module__ == target_manager_class.__module__
        and manager.__class__.__name__ == target_manager_class.__name__
    ):
        manager.__class__ = target_manager_class
    inner = getattr(manager, "_tracker", None)
    if inner is not None and (
        inner.__class__.__module__ == target_tracker_class.__module__
        and inner.__class__.__name__ == target_tracker_class.__name__
    ):
        inner.__class__ = target_tracker_class
    factory = getattr(manager, "_tracker_factory", None)
    if (
        getattr(factory, "__module__", None) == target_tracker_class.__module__
        and getattr(factory, "__name__", None) == target_tracker_class.__name__
    ):
        manager._tracker_factory = target_tracker_class
    # Migrations may add fields required by the new class, but may not alter
    # any valid policy-owned observation. The transaction snapshots the full
    # objects first and the state guard below rolls back a destructive or
    # incompatible migration.
    upgrade_inner = getattr(inner, "upgrade_runtime_state", None)
    if callable(upgrade_inner):
        upgrade_inner()
    upgrade_manager = getattr(manager, "upgrade_runtime_state", None)
    if callable(upgrade_manager):
        upgrade_manager()


def _migrate_task_memory(runtime: Any, modules: Mapping[str, ModuleType]) -> None:
    memory = getattr(runtime, "task_memory", None)
    if memory is None:
        return
    task_module = _module_alias(modules, "task_memory")
    target_class = task_module.OfficialTaskMemory
    if (
        memory.__class__.__module__ == target_class.__module__
        and memory.__class__.__name__ == target_class.__name__
    ):
        object.__setattr__(memory, "__class__", target_class)
    upgrade = getattr(memory, "upgrade_runtime_state", None)
    if callable(upgrade):
        upgrade()
    if memory.__class__ is target_class:
        task_module.bind_official_task_memory(
            runtime.server,
            memory,
            dynamic_memory=getattr(runtime, "tracked_object_distances", None),
        )


def _is_navigation_bridge_class(value: Any) -> bool:
    return bool(
        value is not None
        and getattr(value, "__module__", None) == _NAVIGATION_MAP_BRIDGE_MODULE
        and getattr(value, "__name__", None) == "NavigationMapBridge"
    )


def _migrate_navigation_bridge(
    runtime: Any,
    modules: Mapping[str, ModuleType],
) -> bool:
    """Move the policy-owned bridge onto fresh methods without replacing it."""

    bridge = getattr(runtime, "_navigation_map_bridge", None)
    if bridge is None or not _is_navigation_bridge_class(bridge.__class__):
        return False
    target_class = modules[_NAVIGATION_MAP_BRIDGE_MODULE].NavigationMapBridge
    if bridge.__class__ is target_class:
        return False
    # NavigationMapBridge is an unslotted Python class. An incompatible future
    # layout raises here and aborts the transaction instead of discarding its
    # live cache or silently retaining old capture semantics.
    bridge.__class__ = target_class
    return True


def _rebind_policy_globals(modules: Mapping[str, ModuleType]) -> tuple[ModuleType, dict[str, Any]]:
    policy = importlib.import_module(
        "behavior_interface_eval_test.official_policy_interface"
    )
    previous = {name: policy.__dict__.get(name, _MISSING) for name in _POLICY_REBINDS}
    for policy_name, (basename, source_name) in _POLICY_REBINDS.items():
        policy.__dict__[policy_name] = getattr(
            _module_alias(modules, basename), source_name
        )
    return policy, previous


def _restore_policy_globals(policy: ModuleType, previous: Mapping[str, Any]) -> None:
    for name, value in previous.items():
        if value is _MISSING:
            policy.__dict__.pop(name, None)
        else:
            policy.__dict__[name] = value


def _make_move_tracked_point_view(runtime: Any, policy: ModuleType):
    """Build a request handler whose schema is resolved at call time."""

    from flask import jsonify, request

    def execute_request():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, Mapping):
            return jsonify(
                {
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "input validation",
                    "error": "request body must be a JSON object",
                }
            ), 400
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            return jsonify(
                {
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "input validation",
                    "error": "session_id is required",
                }
            ), 400
        raw_args = {
            str(key): value
            for key, value in body.items()
            if str(key) != "session_id"
        }
        request_id = None
        try:
            # Validate and enqueue under one generation lock. Once queued, the
            # coordinator's idle check prevents a commit until this exact job
            # is terminal, while the HTTP wait itself does not block a reload
            # request from returning a deterministic conflict.
            with runtime._official_reload_lock:
                args = policy.validate_submission("move_tracked_point", raw_args)
                request_stack = runtime.official_v2_reload_status()
                wait_timeout_s = min(
                    630.0,
                    max(120.0, float(args.get("timeout_s", 90.0)) + 30.0),
                )
                request_id = runtime.server.submit_skill(
                    "move_tracked_point", args
                )
        except (TypeError, ValueError) as exc:
            return jsonify(
                {
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "input validation",
                    "error": str(exc),
                }
            ), 400
        except Exception as exc:
            return jsonify(
                {
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "planning",
                    "error": str(exc),
                }
            ), 400

        def respond(result: Any, status_code: int):
            if not isinstance(result, Mapping):
                return jsonify(
                    {
                        "ok": False,
                        "tool": "move_tracked_point",
                        "failure_stage": "planning",
                        "error": "official server returned a non-object result",
                    }
                ), 400
            payload = dict(result)
            payload.setdefault("tool", "move_tracked_point")
            payload.setdefault(
                "execution_mode", str(args.get("execution_mode", "exec"))
            )
            payload.setdefault(
                "reload_generation", int(request_stack.get("generation", 0))
            )
            payload.setdefault(
                "official_v2_stack_digest", request_stack.get("stack_digest")
            )
            if payload.get("ok") is True:
                if str(args.get("execution_mode", "exec")) == "plan":
                    policy._expose_successful_plan_preview(payload)
                else:
                    policy._attach_official_head_capture(
                        runtime,
                        payload,
                        session_id=session_id,
                        timeout_s=policy._FAILURE_EXIT_CAPTURE_TIMEOUT_S,
                    )
            return jsonify(policy._json_ready(payload)), status_code

        try:
            result = runtime.server.wait_for_skill_result(
                "move_tracked_point",
                timeout_s=wait_timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            if callable(cancel_request) and request_id:
                cancel_request(request_id)
            terminal = None
            if request_id:
                try:
                    terminal = runtime.server.wait_for_skill_result(
                        "move_tracked_point",
                        timeout_s=1.0,
                        poll_s=0.01,
                        request_id=request_id,
                    )
                except Exception:
                    terminal = None
            if terminal is not None:
                acknowledge = getattr(runtime, "acknowledge_public_job_terminal", None)
                if callable(acknowledge):
                    acknowledge(request_id)
            if isinstance(terminal, Mapping) and terminal.get("ok"):
                return respond(terminal, 200)
            payload = {
                "ok": False,
                "tool": "move_tracked_point",
                "failure_stage": "timeout",
                "error": str(exc),
            }
            if terminal is not None:
                payload["terminal_result"] = terminal
            return jsonify(policy._json_ready(payload)), 504
        except Exception as exc:
            return jsonify(
                {
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "planning",
                    "error": str(exc),
                }
            ), 400
        return respond(
            result,
            200 if isinstance(result, Mapping) and result.get("ok") else 400,
        )

    def official_move_tracked_point():
        return execute_request()

    return official_move_tracked_point


def _make_tools_metadata_view(runtime: Any, base_view: Callable[..., Any]):
    """Overlay current move metadata without nesting wrappers per reload."""

    from flask import current_app, jsonify

    def current_tools():
        with runtime._official_reload_lock:
            response = current_app.make_response(base_view())
            payload = response.get_json(silent=True) or {}
            tools = list(payload.get("tools") or [])
            previous = next(
                (dict(item) for item in tools if item.get("name") == "move_tracked_point"),
                {"name": "move_tracked_point", "endpoint": "/api/v2/move_tracked_point"},
            )
            tools = [item for item in tools if item.get("name") != "move_tracked_point"]
            spec = runtime.tool_registry.get("move_tracked_point")
            params = list(getattr(spec, "params", ()) or ())
            existing_args = [dict(item) for item in list(previous.get("args") or [])]
            old_by_name = {
                str(item.get("name")): item
                for item in existing_args
                if str(item.get("name") or "")
            }
            # The fresh sealed registry is authoritative for the executable
            # signature.  Keep rich UI-only fields from the previous static
            # description, but overwrite every overlapping contract field and
            # drop parameters which no longer exist.
            existing_args = []
            for parameter in params:
                fresh = dict(parameter)
                merged = dict(old_by_name.get(str(fresh.get("name")), {}))
                merged.update(fresh)
                existing_args.append(merged)
            contract = importlib.import_module(f"{_PACKAGE}.contract")
            for item in existing_args:
                if str(item.get("name")) == "points":
                    item["min_points"] = 1
                    item["max_points"] = int(
                        getattr(contract, "MOVE_TRACKED_POINT_MAX_POINTS", 6)
                    )
                    item["quick_max_points"] = int(
                        getattr(contract, "MOVE_TRACKED_POINT_QUICK_MAX_POINTS", 6)
                    )
            previous["args"] = existing_args
            previous["desc"] = str(getattr(spec, "description", "") or previous.get("desc", ""))
            status = runtime.official_v2_reload_status()
            previous["metadata_revision"] = (status.get("ui") or {}).get(
                "metadata_revision"
            )
            insertion = next(
                (
                    index + 1
                    for index, item in enumerate(tools)
                    if item.get("name") == "track_object_distance"
                ),
                len(tools),
            )
            tools.insert(insertion, previous)
            payload["tools"] = tools
            payload["official_v2_reload"] = {
                "generation": int(status.get("generation", 0)),
                "stack_digest": status.get("stack_digest"),
                "metadata_revision": (status.get("ui") or {}).get("metadata_revision"),
            }
            return jsonify(payload)

    return current_tools


def install_capture_wait_bridge(runtime: Any) -> bool:
    """Make an already-created server resolve detached capture tokens.

    A full process restart installs the strict policy wait closure directly.
    A transactional tool reload, however, intentionally leaves the policy
    coordinator and its existing server object in place.  Wrapping that
    instance once keeps both generations compatible: old waiters get the
    completed artifact, while a new strict waiter (which already resolves the
    token) passes through without a second wait.
    """

    server = getattr(runtime, "server", None)
    previous = getattr(server, "wait_for_skill_result", None)
    if not callable(previous):
        return False
    # Keep the bridge implementation next to the token registry.  This is
    # important during a transactional reload: the coordinator may still be
    # an older module object while ``tools`` has just been replaced, and the
    # current registry is the only place that can resolve its tokens.
    try:
        tools = importlib.import_module(f"{_PACKAGE}.tools")
        installer = getattr(tools, "_install_capture_wait_bridge", None)
    except Exception:
        installer = None
    if callable(installer):
        return bool(installer(server))
    if bool(getattr(previous, _CAPTURE_WAIT_BRIDGE_ATTR, False)):
        return True

    def wait_for_skill_result_with_capture(
        skill_name: str,
        timeout_s: float = 180.0,
        poll_s: float = 0.25,
        request_id: str | None = None,
    ):
        started = time.monotonic()
        result = previous(
            skill_name,
            timeout_s=timeout_s,
            poll_s=poll_s,
            request_id=request_id,
        )
        if not isinstance(result, Mapping):
            return result
        token = str(result.get("_official_capture_pending_token") or "").strip()
        if not token:
            return result
        tools = importlib.import_module(f"{_PACKAGE}.tools")
        resolver = getattr(tools, "wait_for_capture_artifact", None)
        if not callable(resolver):
            raise RuntimeError("official capture result resolver is unavailable")
        try:
            remaining = max(
                0.0,
                float(timeout_s) - (time.monotonic() - started),
            )
            final = resolver(token, timeout_s=remaining)
        except TimeoutError:
            cancel = getattr(tools, "cancel_capture_artifact", None)
            if callable(cancel):
                cancel(token)
            raise
        if isinstance(final, Mapping):
            final = dict(final)
            final.pop("_official_capture_pending_token", None)
        return final

    wait_for_skill_result_with_capture.__name__ = (
        "wait_for_skill_result_with_capture"
    )
    setattr(wait_for_skill_result_with_capture, _CAPTURE_WAIT_BRIDGE_ATTR, True)
    setattr(
        wait_for_skill_result_with_capture,
        "_official_capture_wait_bridge_base",
        previous,
    )
    server.wait_for_skill_result = wait_for_skill_result_with_capture
    return True


def _install_http_bindings(runtime: Any, policy: ModuleType) -> None:
    install_capture_wait_bridge(runtime)
    _enable_capture_runtime_markers(runtime)
    app = getattr(runtime, "_official_http_app", None)
    if app is None:
        return
    app.view_functions["official_move_tracked_point"] = (
        _make_move_tracked_point_view(runtime, policy)
    )
    base = getattr(runtime, "_official_v2_tools_base_view", None)
    if base is None:
        base = app.view_functions.get("api_v2_tools")
        runtime._official_v2_tools_base_view = base
    if base is not None:
        app.view_functions["api_v2_tools"] = _make_tools_metadata_view(runtime, base)


def reload_official_v2_tool_stack(
    runtime: Any,
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    reload_module: Callable[[ModuleType], ModuleType] = importlib.reload,
    invalidate_caches: Callable[[], None] = importlib.invalidate_caches,
) -> dict[str, Any]:
    """Reload and atomically publish one coherent official-v2 generation."""

    _ensure_runtime_fields(runtime)
    with ExitStack() as locks:
        locks.enter_context(runtime._official_reload_lock)
        locks.enter_context(_runtime_lock(runtime, "_step_lock"))
        locks.enter_context(_server_lock(runtime))
        locks.enter_context(_runtime_lock(runtime, "_live_test_lock"))
        locks.enter_context(_manager_lock(runtime))
        for state_lock in _navigation_state_locks(runtime):
            locks.enter_context(state_lock)
        _check_idle(runtime)

        if not hasattr(runtime, "_official_skills_module"):
            runtime._official_skills_module = import_module(
                "behavior_interface.skills"
            )

        invalidate_caches()
        manifest_names = tuple(
            name
            for name in _RELOAD_MODULE_NAMES
            if name in _MANDATORY_MODULE_NAMES or name in sys.modules
        )
        unique_names = tuple(dict.fromkeys(manifest_names))
        modules = {name: import_module(name) for name in unique_names}
        package_root = _module_path(
            _module_alias(modules, "contract")
        ).parent.resolve()
        for name, module in _loaded_official_v2_modules(package_root).items():
            modules.setdefault(name, module)
        overlay_lock = getattr(
            modules[_NAVIGATION_ROUTE_OVERLAY_MODULE], "_LOCK", None
        )
        if overlay_lock is not None:
            locks.enter_context(overlay_lock)
        reload_names = _reload_order(modules)
        preflight = {
            name: _source_record(module) for name, module in modules.items()
        }
        worker_preflight = _ik_worker_source_record(modules)
        asset_preflight = _asset_revision(modules)
        source_tree_preflight = _package_source_tree(modules)
        asset_tree_preflight = _package_asset_tree(modules)

        for name, module in modules.items():
            if _source_record(module)["digest"] != preflight[name]["digest"]:
                raise RuntimeError(f"hot-reload source changed during preflight: {name}")

        background_drains = _drain_background_tasks(runtime, modules)

        old_registry = getattr(runtime, "tool_registry", None)
        skills = runtime._official_skills_module
        skills_state = {
            name: getattr(skills, name, _MISSING)
            for name in ("SKILL_REGISTRY", "PUBLIC_SKILLS", "TOOL_VERSION")
        }
        old_generation = int(runtime._official_reload_generation)
        old_receipt = deepcopy(runtime._official_reload_receipt)
        old_live_build = getattr(runtime, "_live_test_module_build", None)
        old_world_runtime = {}
        world = getattr(getattr(runtime, "server", None), "world", None)
        if world is not None:
            for name in (
                "_official_reload_generation",
                "_official_stack_digest",
                "_official_async_capture_artifacts",
                "_official_detached_capture_results",
            ):
                old_world_runtime[name] = getattr(world, name, _MISSING)

        manager = getattr(runtime, "tracked_object_distances", None)
        inner_tracker = getattr(manager, "_tracker", None)
        task_memory = getattr(runtime, "task_memory", None)
        navigation_bridge = getattr(runtime, "_navigation_map_bridge", None)
        manager_class = None if manager is None else manager.__class__
        inner_class = None if inner_tracker is None else inner_tracker.__class__
        task_memory_class = None if task_memory is None else task_memory.__class__
        navigation_bridge_class = (
            None if navigation_bridge is None else navigation_bridge.__class__
        )
        tracker_state_snapshot = (
            None if manager is None else _InPlaceStateSnapshot(manager)
        )
        task_memory_snapshot = (
            None if task_memory is None else _InPlaceStateSnapshot(task_memory)
        )
        memory_callback_state = {
            name: getattr(runtime.server, name, _MISSING)
            for name in ("get_memory", "get_memory_text", "get_memory_summary")
        }
        state_before = _state_identity(runtime)
        callback_ids_before = _registry_callback_ids(old_registry)
        module_snapshots = {module: dict(module.__dict__) for module in modules.values()}
        persistent = {
            (module_name, global_name): modules[module_name].__dict__.get(
                global_name, _MISSING
            )
            for module_name, names in _PERSISTENT_GLOBALS.items()
            if module_name in modules
            for global_name in names
        }
        policy = None
        policy_state = None
        http_app = getattr(runtime, "_official_http_app", None)
        http_views = (
            None if http_app is None else dict(http_app.view_functions)
        )
        old_tools_base_view = getattr(
            runtime, "_official_v2_tools_base_view", _MISSING
        )

        previous_transaction_active = bool(
            getattr(_TRANSACTION_STATE, "active", False)
        )
        _TRANSACTION_STATE.active = True
        migration_started = False
        renderer_hook_token = None
        renderer_hooks_installed = False
        renderer_hook_report: dict[str, Any] = {}
        navigation_bridge_class_rebound = False
        try:
            reload_reports: list[dict[str, Any]] = []
            for name in reload_names:
                module = modules[name]
                _reload_one(module, reload_module)
                current = _source_record(module)
                if current["digest"] != preflight[name]["digest"]:
                    raise RuntimeError(f"hot-reload source changed while loading: {name}")
                reload_reports.append(
                    {
                        "name": name,
                        "old_file_digest": preflight[name]["digest"],
                        "new_file_digest": current["digest"],
                    }
                )

            # A module checked early in the graph can be edited while later
            # modules are loading. Recheck the complete source and UI asset
            # set at the commit boundary so one generation always corresponds
            # to one immutable set of bytes.
            for name, module in modules.items():
                if _source_record(module)["digest"] != preflight[name]["digest"]:
                    raise RuntimeError(
                        f"hot-reload source changed before commit: {name}"
                    )
            if _asset_revision(modules) != asset_preflight:
                raise RuntimeError("hot-reload UI assets changed before commit")
            source_tree = _package_source_tree(modules)
            if source_tree["digest"] != source_tree_preflight["digest"]:
                changed = _source_tree_changes(source_tree_preflight, source_tree)
                suffix = "" if not changed else ": " + ", ".join(changed)
                raise RuntimeError(
                    "hot-reload package source tree changed before commit" + suffix
                )
            asset_tree = _package_asset_tree(modules)
            if asset_tree["digest"] != asset_tree_preflight["digest"]:
                raise RuntimeError(
                    "hot-reload package asset tree changed before commit"
                )

            for (module_name, global_name), value in persistent.items():
                if value is not _MISSING:
                    modules[module_name].__dict__[global_name] = value

            overlay_module = modules[_NAVIGATION_ROUTE_OVERLAY_MODULE]
            renderer_hook_token = overlay_module.capture_renderer_hook_state()

            registry_module = _module_alias(modules, "registry")
            entries = registry_module.build_registry(runtime.adapter)
            new_registry = registry_module.OfficialToolRegistry(entries)
            if tuple(new_registry) != tuple(
                _module_alias(modules, "capabilities").PUBLIC_TOOLS
            ):
                raise RuntimeError("reloaded official-v2 registry surface is inconsistent")

            migration_started = True
            _migrate_tracker(runtime, modules)
            _migrate_task_memory(runtime, modules)
            navigation_bridge_class_rebound = _migrate_navigation_bridge(
                runtime,
                modules,
            )
            state_after_migration = _state_identity(runtime)
            for key in state_before:
                if state_after_migration[key] != state_before[key]:
                    raise RuntimeError(
                        f"hot reload changed policy-owned runtime state: {key}"
                    )

            policy, policy_state = _rebind_policy_globals(modules)
            runtime.tool_registry = new_registry
            registry_module.ensure_profile_installed(skills, new_registry)
            _install_http_bindings(runtime, policy)
            renderer_hook_report = dict(
                overlay_module.install_renderer_hooks() or {}
            )
            renderer_hooks_installed = True

            generation = old_generation + 1
            builds = _builds(modules)
            trajectory = _trajectory_identity(modules)
            asset_revision = _asset_revision(modules)
            metadata_revision = _metadata_revision(modules)
            worker_source = _ik_worker_source_record(modules)
            if (
                worker_source["path"] == worker_preflight["path"]
                and worker_source["digest"] != worker_preflight["digest"]
            ):
                raise RuntimeError(
                    "hot-reload source changed before commit: "
                    f"{worker_source['name']}"
                )
            if _path_source_record(
                Path(worker_source["path"]), name=worker_source["name"]
            )["digest"] != worker_source["digest"]:
                raise RuntimeError(
                    "hot-reload source changed while loading: "
                    f"{worker_source['name']}"
                )
            stack_digest = _canonical_digest(
                {
                    "modules": [
                        (item["name"], item["new_file_digest"])
                        for item in reload_reports
                    ],
                    "builds": builds,
                    "trajectory": trajectory,
                    "metadata_revision": metadata_revision,
                    "asset_revision": asset_revision,
                    "source_tree": source_tree["digest"],
                    "asset_tree": asset_tree["digest"],
                    "ik_filter_worker": worker_source,
                }
            )
            tools_module = _module_alias(modules, "tools")
            live_module = modules[_LIVE_MODULE]
            tools_module._OFFICIAL_RELOAD_GENERATION = generation
            tools_module._OFFICIAL_STACK_DIGEST = stack_digest
            live_module._OFFICIAL_RELOAD_GENERATION = generation
            live_module._OFFICIAL_STACK_DIGEST = stack_digest
            runtime._live_test_module_build = builds["live_runner"]
            if world is not None:
                world._official_reload_generation = generation
                world._official_stack_digest = stack_digest

            state_after = _state_identity(runtime)
            for key in state_before:
                if state_after[key] != state_before[key]:
                    raise RuntimeError(
                        "hot reload changed policy-owned runtime state after "
                        f"publication: {key}"
                    )
            preservation = {
                key: state_after.get(key) == state_before.get(key)
                for key in state_before
            }
            preservation.update(
                {
                    "plan_integrity_key": (
                        tools_module.__dict__.get("_PLAN_INTEGRITY_KEY")
                        is persistent.get((f"{_PACKAGE}.tools", "_PLAN_INTEGRITY_KEY"))
                    ),
                    "session_lock": (
                        tools_module.__dict__.get("_SESSION_LOCK")
                        is persistent.get((f"{_PACKAGE}.tools", "_SESSION_LOCK"))
                    ),
                    "replay_executor": (
                        _module_alias(modules, "tracked_object_distance").__dict__.get(
                            "_REPLAY_COMPRESSION_EXECUTOR"
                        )
                        is persistent.get(
                            (
                                f"{_PACKAGE}.tracked_object_distance",
                                "_REPLAY_COMPRESSION_EXECUTOR",
                            )
                        )
                    ),
                    "ik_workers": (
                        _module_alias(modules, "grasp_kinematics_local").__dict__.get(
                            "_PERSISTENT_IK_WORKERS"
                        )
                        is persistent.get(
                            (
                                f"{_PACKAGE}.grasp_kinematics_local",
                                "_PERSISTENT_IK_WORKERS",
                            )
                        )
                    ),
                    "ik_workers_lock": (
                        _module_alias(modules, "grasp_kinematics_local").__dict__.get(
                            "_PERSISTENT_IK_WORKERS_LOCK"
                        )
                        is persistent.get(
                            (
                                f"{_PACKAGE}.grasp_kinematics_local",
                                "_PERSISTENT_IK_WORKERS_LOCK",
                            )
                        )
                    ),
                }
            )
            navigation = _navigation_reload_report(
                modules,
                new_registry,
                runtime=runtime,
                bridge_class_before=navigation_bridge_class,
                bridge_class_rebound=navigation_bridge_class_rebound,
                callback_ids_before=callback_ids_before,
                previous_receipt=old_receipt,
                reload_names=reload_names,
                renderer_hooks=renderer_hook_report,
            )
            receipt = {
                "ok": True,
                "generation": generation,
                "stack_digest": stack_digest,
                "rollback": False,
                "modules": reload_reports,
                "source_tree": source_tree,
                "asset_tree": asset_tree,
                "builds": builds,
                "trajectory": trajectory,
                "navigation": navigation,
                "registry": {
                    "public_tools": list(new_registry),
                    "callback_ids_before": callback_ids_before,
                    "callback_ids_after": _registry_callback_ids(new_registry),
                    "installed": skills.SKILL_REGISTRY is new_registry,
                },
                "state_preserved": preservation,
                "state_snapshot": state_after,
                "ui": {
                    "metadata_revision": metadata_revision,
                    "asset_revision": asset_revision,
                },
                "workers": {
                    "ik_filter": {
                        "source_digest": worker_source["digest"],
                        "source_path": worker_source["path"],
                        "previous_source_digest": worker_preflight["digest"],
                        "invalidated_count": 0,
                        "close_errors": [],
                    },
                    "replay_compression": background_drains[
                        "replay_compression"
                    ],
                    "lite_render": background_drains["lite_render"],
                    "prediction_audit": {
                        "closed_count": 0,
                        "close_errors": [],
                    },
                },
            }
            audit_cleanup = _close_prediction_audit_writers(modules)
            receipt["workers"]["prediction_audit"].update(audit_cleanup)
            worker_cleanup = _invalidate_ik_workers(modules)
            receipt["workers"]["ik_filter"].update(worker_cleanup)
            runtime._official_reload_generation = generation
            runtime._official_reload_receipt = deepcopy(receipt)
            return receipt
        except Exception as exc:
            renderer_rollback_error = None
            if renderer_hooks_installed and renderer_hook_token is not None:
                try:
                    modules[
                        _NAVIGATION_ROUTE_OVERLAY_MODULE
                    ].restore_renderer_hook_state(renderer_hook_token)
                except Exception as rollback_exc:  # pragma: no cover - fatal guard
                    renderer_rollback_error = rollback_exc
            _restore_modules(module_snapshots)
            if (
                navigation_bridge is not None
                and navigation_bridge_class is not None
                and navigation_bridge.__class__ is not navigation_bridge_class
            ):
                navigation_bridge.__class__ = navigation_bridge_class
            if manager is not None:
                manager.__class__ = manager_class
                if migration_started and tracker_state_snapshot is not None:
                    tracker_state_snapshot.restore()
            if inner_tracker is not None:
                inner_tracker.__class__ = inner_class
            if task_memory is not None:
                object.__setattr__(task_memory, "__class__", task_memory_class)
                if migration_started and task_memory_snapshot is not None:
                    task_memory_snapshot.restore()
            for name, value in memory_callback_state.items():
                if value is _MISSING:
                    try:
                        delattr(runtime.server, name)
                    except AttributeError:
                        pass
                else:
                    setattr(runtime.server, name, value)
            if policy is not None and policy_state is not None:
                _restore_policy_globals(policy, policy_state)
            runtime.tool_registry = old_registry
            for name, value in skills_state.items():
                if value is _MISSING:
                    try:
                        delattr(skills, name)
                    except AttributeError:
                        pass
                else:
                    setattr(skills, name, value)
            runtime._official_reload_generation = old_generation
            runtime._official_reload_receipt = old_receipt
            runtime._live_test_module_build = old_live_build
            if http_app is not None and http_views is not None:
                http_app.view_functions.clear()
                http_app.view_functions.update(http_views)
            if old_tools_base_view is _MISSING:
                try:
                    delattr(runtime, "_official_v2_tools_base_view")
                except AttributeError:
                    pass
            else:
                runtime._official_v2_tools_base_view = old_tools_base_view
            if world is not None:
                for name, value in old_world_runtime.items():
                    if value is _MISSING:
                        try:
                            delattr(world, name)
                        except AttributeError:
                            pass
                    else:
                        setattr(world, name, value)
            if renderer_rollback_error is not None:
                raise RuntimeError(
                    "official-v2 reload failed and renderer hook rollback also "
                    f"failed: {type(renderer_rollback_error).__name__}: "
                    f"{renderer_rollback_error}"
                ) from exc
            raise
        finally:
            _TRANSACTION_STATE.active = previous_transaction_active


def official_v2_reload_status(runtime: Any) -> dict[str, Any]:
    """Return the last committed generation without mutating the stack."""

    _ensure_runtime_fields(runtime)
    with runtime._official_reload_lock:
        receipt = runtime._official_reload_receipt
        if isinstance(receipt, Mapping):
            return deepcopy(dict(receipt))
        registry = getattr(runtime, "tool_registry", {})
        state_snapshot = _state_identity(runtime)
        return {
            "ok": True,
            "generation": int(runtime._official_reload_generation),
            "stack_digest": None,
            "rollback": False,
            "modules": [],
            "source_tree": {"root": None, "digest": None, "files": []},
            "asset_tree": {"root": None, "digest": None, "files": []},
            "builds": {},
            "trajectory": {},
            "navigation": {
                "source_digests": {},
                "source_changed_since_previous_commit": {},
                "reloaded_modules": [],
                "dependencies_before_executor": False,
                "bindings_current": {},
                "callback_id_before": _registry_callback_ids(registry).get(
                    "navigate_to"
                ),
                "callback_id_after": _registry_callback_ids(registry).get(
                    "navigate_to"
                ),
                "callback_replaced": False,
                "bridge": {
                    "instance_id": (
                        None
                        if getattr(runtime, "_navigation_map_bridge", None) is None
                        else id(runtime._navigation_map_bridge)
                    ),
                    "class_id_before": None,
                    "class_id_after": None,
                    "class_rebound": False,
                    "class_current": False,
                },
                "renderer_hooks": {},
            },
            "registry": {
                "public_tools": list(registry) if isinstance(registry, Mapping) else [],
                "callback_ids_before": _registry_callback_ids(registry),
                "callback_ids_after": _registry_callback_ids(registry),
                "installed": getattr(
                    getattr(runtime, "_official_skills_module", None),
                    "SKILL_REGISTRY",
                    None,
                )
                is registry,
            },
            "state_preserved": {key: True for key in state_snapshot},
            "state_snapshot": state_snapshot,
            "ui": {"metadata_revision": None, "asset_revision": {}},
            "workers": {
                "ik_filter": {
                    "source_digest": None,
                    "source_path": None,
                    "previous_source_digest": None,
                    "invalidated_count": 0,
                    "close_errors": [],
                },
                "replay_compression": {
                    "drained_count": 0,
                    "pending_count_after": 0,
                },
                "lite_render": {
                    "drained_count": 0,
                    "pending_count_after": 0,
                },
                "prediction_audit": {
                    "closed_count": 0,
                    "close_errors": [],
                },
            },
        }


def bootstrap_runtime_class(runtime_class: type) -> dict[str, Any]:
    """Install the transaction methods on an already-created runtime class.

    This method changes code bindings only. It neither imports/reloads the tool
    graph nor touches an evaluator observation, so it is safe to invoke from
    the tail of the legacy live-module reload that is already holding the live
    lock.
    """

    if bool(getattr(runtime_class, "_official_v2_reload_bootstrap", False)):
        return {"ok": True, "already_installed": True}

    # These methods mutate the live-job lifecycle. An old process already has
    # their bytecode in memory, so wrap the existing methods here as part of
    # the one-request bootstrap rather than requiring a policy-process restart.
    for method_name in (
        "start_live_move_tracked_point_test",
        "cancel_live_test",
        "release_terminal_live_test",
        "reset",
    ):
        previous_method = getattr(runtime_class, method_name, None)
        if not callable(previous_method):
            continue

        def linearized(self, *args: Any, __method=previous_method, **kwargs: Any):
            coordinator = importlib.import_module(__name__)
            coordinator._ensure_runtime_fields(self)
            with self._official_reload_lock:
                return __method(self, *args, **kwargs)

        linearized.__name__ = str(method_name)
        linearized.__qualname__ = f"{runtime_class.__name__}.{method_name}"
        setattr(runtime_class, method_name, linearized)

    def full_reload(self):
        coordinator = importlib.import_module(__name__)
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )
        return coordinator.reload_official_v2_tool_stack(
            self,
            import_module=policy.importlib.import_module,
            reload_module=policy.importlib.reload,
            invalidate_caches=policy.importlib.invalidate_caches,
        )

    def reload_status(self):
        coordinator = importlib.import_module(__name__)
        return coordinator.official_v2_reload_status(self)

    def harness_reload(self):
        return reload_live_test_harness(
            self,
            import_module=importlib.import_module,
            reload_module=importlib.reload,
            invalidate_caches=importlib.invalidate_caches,
        )

    runtime_class.reload_official_v2_tool_stack = full_reload
    runtime_class.official_v2_reload_status = reload_status
    runtime_class.reload_live_test_module = harness_reload
    runtime_class._official_v2_reload_bootstrap = True
    return {"ok": True, "already_installed": False}


def install_legacy_runtime_bootstrap() -> bool:
    """Upgrade an already-running pre-transaction runtime without restart."""

    try:
        from flask import current_app, has_request_context
        from behavior_interface_eval_test import official_policy_interface as policy
    except Exception:
        return False

    bootstrap_runtime_class(policy.OfficialPolicyRuntime)
    if bool(getattr(_TRANSACTION_STATE, "active", False)):
        return False
    if not has_request_context():
        return False

    # Replace the existing POST callback so the second call returns the full
    # receipt even though Flask cannot add new routes after serving requests.
    app = current_app._get_current_object()
    old_view = app.view_functions.get("dev_live_reload")
    runtime = None
    for cell in getattr(old_view, "__closure__", ()) or ():
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if candidate.__class__ is policy.OfficialPolicyRuntime or (
            hasattr(candidate, "_live_test_lock")
            and hasattr(candidate, "server")
            and hasattr(candidate, "tool_registry")
        ):
            runtime = candidate
            break
    if runtime is not None:
        from flask import jsonify

        runtime._official_http_app = app
        _ensure_runtime_fields(runtime)
        _ensure_submission_barrier(runtime)
        _install_http_bindings(runtime, policy)

        def transactional_reload_view():
            try:
                return jsonify(runtime.reload_official_v2_tool_stack())
            except RuntimeError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 409
            except Exception as exc:
                return jsonify(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                ), 500

        app.view_functions["dev_live_reload"] = transactional_reload_view
        old_status_view = app.view_functions.get("dev_live_status")
        if old_status_view is not None:
            def transactional_status_view():
                payload = dict(runtime.live_test_status() or {})
                payload["reload_receipt"] = runtime.official_v2_reload_status()
                return jsonify(payload)

            app.view_functions["dev_live_status"] = transactional_status_view
    return True


__all__ = [
    "bootstrap_runtime_class",
    "install_capture_wait_bridge",
    "install_legacy_runtime_bootstrap",
    "official_v2_reload_status",
    "reload_live_test_harness",
    "reload_official_v2_tool_stack",
]
