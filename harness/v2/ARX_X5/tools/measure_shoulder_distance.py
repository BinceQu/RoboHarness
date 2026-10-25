from __future__ import annotations

import numpy as np

from tools.util.motion_constants import X5_SHOULDERS
from tools.util.motion_common import _current_snapshot, _point, _set_result


def measure_shoulder_distance(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    object_name: str = "",
    **kwargs,
):
    del kwargs
    try:
        if str(object_name or "").strip():
            raise ValueError("object_name is unavailable; select an RGB-D point")
        snapshot = _current_snapshot(ctx, ctx.snapshots.load(session_id, image_id))
        target, point_meta = _point(snapshot, u, v)
        distances = {
            side: float(np.linalg.norm(target - X5_SHOULDERS[side]))
            for side in ("left", "right")
        }
        result = {
            "ok": True,
            "target_source": "frozen_evaluator_depth_click",
            "target_robodojo_env_m": target.tolist(),
            "point_observation": point_meta,
            "left_shoulder_robodojo_env_m": X5_SHOULDERS["left"].tolist(),
            "right_shoulder_robodojo_env_m": X5_SHOULDERS["right"].tolist(),
            "left_shoulder_to_object_m": distances["left"],
            "right_shoulder_to_object_m": distances["right"],
            "recommended_arm": min(distances, key=distances.get),
            "static_model": "RoboDojo dual_x5.yml root transforms",
        }
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    _set_result(ctx, "measure_shoulder_distance", result)
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("measure_shoulder_distance", measure_shoulder_distance)
