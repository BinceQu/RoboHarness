"""Reject stale / cross-port actions before monitor or simulator side effects."""
from __future__ import annotations

from pathlib import Path

from .catalog import mapped_task_for_port
from .isolation import SID_RE, process_running, read_json


def install_session_guard(app, port: int, guard_path: str = "") -> None:
    from flask import jsonify, request

    port = int(port)
    path = Path(guard_path) if guard_path else None
    app.extensions["official_session_guard"] = {"enabled": path is not None, "port": port}

    def reject(message: str):
        return jsonify(ok=False, code="official_session_rejected", error=message), 409

    def check_session():
        # cache=True preserves the body for monitor hooks which call get_data().
        body = request.get_json(silent=True)
        body = body if isinstance(body, dict) else {}
        ids = {str(value) for value in (body.get("session_id"), request.args.get("session_id"),
                                       request.headers.get("X-Behavior-Session-ID")) if value}
        if len(ids) > 1:
            return reject("请求中的 session_id 不一致")
        sid = next(iter(ids), "")
        match = SID_RE.fullmatch(sid)
        if match and (int(match[2]) != port or int(match[1]) != mapped_task_for_port(port)):
            return reject("session 与本口 task/port 不匹配")
        if path is None or request.method in {"GET", "HEAD", "OPTIONS"}:
            return None
        active = read_json(path)
        if (not match or active.get("port") != port
                or not active.get("session_id") or sid != active["session_id"]):
            return reject("本口只接受当前 harness 授权的 session；旧 session 已失效")
        pid, start = active.get("owner_pid"), active.get("owner_start")
        if not isinstance(pid, int) or not start or not process_running(pid, str(start)):
            return reject("本口 harness owner 已退出，禁止继续执行动作")
        # The evaluator / harness own episode selection. Even an active model
        # cannot reset into an instance outside this run's explicit plan.
        if not (request.path.startswith("/api/v2/")
                or request.path.startswith("/api/agent_monitor")):
            return reject("评测期间只允许本局工具和监控请求；切局由 harness 管理")
        return None

    # build_app has already registered monitor hooks that mutate session state.
    # Rejection must run first, not merely before the endpoint handler.
    app.before_request_funcs.setdefault(None, []).insert(0, check_session)
