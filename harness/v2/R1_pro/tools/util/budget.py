"""Only operational counters from the evaluator; never goal-state telemetry."""
from numbers import Integral
import re


def unavailable(reason):
    return {"available": False, "used_ticks": None, "total_ticks": None,
            "remaining_ticks": None, "used_fraction": None, "reason": reason}


def counter(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**53:
        return None
    return int(value)


def sanitize_budget(value):
    if not isinstance(value, dict) or value.get("available") is not True:
        return unavailable("evaluator_budget_not_published")
    used, total = counter(value.get("used_ticks")), counter(value.get("total_ticks"))
    if used is None or total is None or total <= 0:
        return unavailable("evaluator_counter_unavailable")
    episode, source = value.get("episode_id"), value.get("source")
    if (not isinstance(episode, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", episode)
        or source not in {"official_evaluator_episode_steps", "resident_active_steps"}):
        return unavailable("invalid_rollout_identity")
    return {"available": True, "used_ticks": used, "total_ticks": total,
        "remaining_ticks": max(0, total-used), "used_fraction": used / total,
        "episode_id": episode, "instance_id": counter(value.get("instance_id")),
        "source": source, "sampling": "latest_evaluator_observation"}
