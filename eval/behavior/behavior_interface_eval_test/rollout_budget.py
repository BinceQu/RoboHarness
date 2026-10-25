"""Operational rollout counters only: no goal truth or simulator object state."""
from __future__ import annotations

from numbers import Integral
import re
from typing import Any

ROLLOUT_BUDGET_KEY = '__behavior_rollout_budget__'
SOURCES = {'official_evaluator_episode_steps', 'resident_active_steps'}


def unavailable(reason: str) -> dict[str, Any]:
    return {'available': False, 'used_ticks': None, 'total_ticks': None,
            'remaining_ticks': None, 'used_fraction': None, 'reason': reason}


def counter(value: Any) -> int | None:
    try:
        if hasattr(value, 'item'):
            value = value.item()
    except (ValueError, TypeError, RuntimeError):
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**53:
        return None
    return int(value)


def budget_snapshot(used: Any, total: Any, *, episode_id: str,
                    instance_id: Any, source: str) -> dict[str, Any]:
    used, total = counter(used), counter(total)
    if used is None or total is None or total <= 0:
        return unavailable('evaluator_counter_unavailable')
    if (not isinstance(episode_id, str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', episode_id)
            or not isinstance(source, str) or source not in SOURCES):
        return unavailable('invalid_rollout_identity')
    return {'available': True, 'used_ticks': used, 'total_ticks': total,
            'remaining_ticks': max(0, total - used), 'used_fraction': used / total,
            'episode_id': episode_id, 'instance_id': counter(instance_id),
            'source': source, 'sampling': 'latest_evaluator_observation'}


def sanitize_budget(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get('available') is not True:
        return unavailable('evaluator_budget_not_published')
    return budget_snapshot(value.get('used_ticks'), value.get('total_ticks'),
                           episode_id=value.get('episode_id'),
                           instance_id=value.get('instance_id'), source=value.get('source'))


def install_rollout_budget_routes(app: Any, runtime: Any) -> None:
    """Expose cached counters without requesting an action or simulator step."""
    from flask import jsonify

    def read_rollout_budget():
        try:
            connected, _ = runtime.evaluator_connections.snapshot()
            if not connected:
                return unavailable('evaluator_disconnected')
            return runtime.adapter.rollout_budget()
        except Exception:
            # Optional telemetry must not break the existing memory endpoint.
            return unavailable('budget_read_failed')

    runtime.server.rollout_budget_snapshot = read_rollout_budget

    @app.get('/api/rollout_budget')
    def api_rollout_budget():
        response = jsonify(read_rollout_budget())
        response.headers['Cache-Control'] = 'no-store'
        return response
