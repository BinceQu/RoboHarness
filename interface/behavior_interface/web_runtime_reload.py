"""Install selected web.py changes into a running Flask app without restart."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
from types import ModuleType
from typing import Any


_V2_ENDPOINTS_TO_REPLACE = (
    "api_skills_version",
    "api_v2_read_depth",
    "api_v2_adjust_chassis",
    "api_v2_adjust_left_eef_pose_in_head_frame",
    "api_v2_adjust_right_eef_pose_in_head_frame",
    "api_v2_adjust_left_eef_pose_in_wrist_frame",
    "api_v2_adjust_right_eef_pose_in_wrist_frame",
    "api_v2_move_point_to_point",
    "api_v2_move_eef",
    "api_v2_plan_move_eef",
    "api_v2_plan_move_eef_to_point",
    "api_v2_plan",
    "api_v2_adjust_plan_pose",
    "api_v2_set_arm_to_grasp_position",
    "api_v2_reset_body",
    "api_v2_manipulate_add_vector_to_point",
    "api_v2_manipulate_move_vector_to_vector",
)


def _load_current_web_snapshot() -> ModuleType:
    web_path = os.path.join(os.path.dirname(__file__), "web.py")
    module_name = f"behavior_interface._web_hot_snapshot_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, web_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load web module snapshot from {web_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _running_server(app) -> Any:
    api_skill = app.view_functions.get("api_skill")
    if api_skill is None:
        raise RuntimeError("running Flask app has no api_skill endpoint")
    closure = inspect.getclosurevars(api_skill)
    server = closure.nonlocals.get("server")
    if server is None:
        raise RuntimeError("cannot recover BehaviorInterface from api_skill closure")
    return server


def hot_install_v2_web_changes() -> dict[str, Any]:
    """Patch meter-aware v2 metadata and handlers into the current Flask app."""
    from flask import current_app, has_app_context

    if not has_app_context():
        return {"installed": False, "reason": "no Flask app context"}

    live_app = current_app._get_current_object()
    live_web = sys.modules.get("behavior_interface.web")
    if live_web is None:
        raise RuntimeError("behavior_interface.web is not loaded")

    server = _running_server(live_app)
    snapshot = _load_current_web_snapshot()
    fresh_app = snapshot.build_app(server)

    pinned_actions_reinstalled = False
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(server.world)
        pinned_actions_reinstalled = bool(
            getattr(server.world, "_codex_pinned_actions_v11", False)
        )
    except Exception:
        pinned_actions_reinstalled = False

    # Existing /api/v2/tools handlers resolve V2_TOOLS through the live
    # behavior_interface.web module globals, so replacing this list is enough
    # to expose newly added schemas without reloading that module.
    live_web.V2_TOOLS = list(snapshot.V2_TOOLS)

    installed_routes: list[str] = []
    replaced_handlers: list[str] = []
    live_rule_endpoints = {rule.endpoint for rule in live_app.url_map.iter_rules()}
    fresh_rules = {rule.endpoint: rule for rule in fresh_app.url_map.iter_rules()}

    for endpoint in _V2_ENDPOINTS_TO_REPLACE:
        handler = fresh_app.view_functions.get(endpoint)
        if handler is None:
            raise RuntimeError(f"fresh web app is missing endpoint {endpoint}")
        live_app.view_functions[endpoint] = handler
        replaced_handlers.append(endpoint)

        if endpoint not in live_rule_endpoints:
            rule = fresh_rules.get(endpoint)
            if rule is None:
                raise RuntimeError(f"fresh web app is missing route for {endpoint}")
            live_app.url_map.add(rule.empty())
            live_rule_endpoints.add(endpoint)
            installed_routes.append(rule.rule)

    return {
        "installed": True,
        "metadata_count": len(live_web.V2_TOOLS),
        "pinned_actions_reinstalled": pinned_actions_reinstalled,
        "installed_routes": installed_routes,
        "replaced_handlers": replaced_handlers,
    }
