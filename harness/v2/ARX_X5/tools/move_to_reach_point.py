from __future__ import annotations

import numpy as np

from tools.util.motion_constants import X5_DEFAULT_REACH_M, X5_SHOULDERS
from tools.util.motion_common import (
    _current_snapshot,
    _finite_number,
    _point,
    _set_result,
)


def move_to_reach_point(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    reach: float = 0.0,
    nav_timeout_s: float = 120.0,
    keep_ori_arm: str = "none",
    **kwargs,
):
    del kwargs
    try:
        snapshot = _current_snapshot(ctx, ctx.snapshots.load(session_id, image_id))
        target, point_meta = _point(snapshot, u, v)
        reach_m = (
            X5_DEFAULT_REACH_M
            if float(reach) <= 0.01
            else _finite_number(reach, "reach")
        )
        distances = {
            side: float(np.linalg.norm(target - X5_SHOULDERS[side]))
            for side in ("left", "right")
        }
        reachable = min(distances.values()) <= reach_m
        timeout = _finite_number(nav_timeout_s, "nav_timeout_s")
        if timeout <= 0.0:
            raise ValueError("nav_timeout_s must be positive")
        result = {
            "ok": reachable,
            "error": None
            if reachable
            else "fixed-base X5 cannot move closer to the point",
            "code": None if reachable else "unsupported_embodiment",
            "target": point_meta,
            "target_robodojo_env_m": target.tolist(),
            "shoulder_distance_m": distances,
            "reach_m": reach_m,
            "already_reachable": reachable,
            "base_motion_m": [0.0, 0.0, 0.0],
            "keep_ori_arm": str(keep_ori_arm),
            "action_emitted": False,
        }
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc), "action_emitted": False}
    _set_result(ctx, "move_to_reach_point", result)
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("move_to_reach_point", move_to_reach_point)
