"""Bounded, allowlisted model-visible simulation budget telemetry."""
from __future__ import annotations

import math
import re
from typing import Any


def unavailable_budget(reason: str = 'budget_not_available') -> dict[str, Any]:
    return {'available': False, 'used_ticks': None, 'total_ticks': None,
            'remaining_ticks': None, 'used_fraction': None, 'reason': reason}


def normalize_budget(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get('available') is not True:
        return unavailable_budget()
    used, total = value.get('used_ticks'), value.get('total_ticks')
    if (type(used) is not int or type(total) is not int
            or not 0 <= used <= 2**53 or not 0 < total <= 2**53):
        return unavailable_budget('invalid_evaluator_counters')
    episode = value.get('episode_id')
    source = value.get('source')
    if (not isinstance(episode, str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', episode)
            or not isinstance(source, str)
            or source not in {'official_evaluator_episode_steps', 'resident_active_steps'}):
        return unavailable_budget('invalid_rollout_identity')
    result = {'available': True, 'used_ticks': used, 'total_ticks': total,
              'remaining_ticks': max(0, total - used), 'used_fraction': used / total,
              'episode_id': episode, 'source': source,
              'sampling': 'latest_evaluator_observation'}
    for key in ('instance_id', 'observation_sequence'):
        number = value.get(key)
        if type(number) is int and 0 <= number <= 2**53:
            result[key] = number
    age = value.get('observation_age_s')
    if type(age) in (int, float) and math.isfinite(age) and age >= 0:
        result['observation_age_s'] = round(age, 3)
    return result
