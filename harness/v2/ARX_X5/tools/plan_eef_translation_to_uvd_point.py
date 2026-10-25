from __future__ import annotations

import numpy as np

from tools.util.motion_common import (
    _head_snapshot,
    _save_pose_plan,
    _select_arm,
    _set_result,
    _target_from_click,
)


def plan_eef_translation_to_uvd_point(ctx, **kwargs):
    try:
        snapshot = _head_snapshot(ctx, kwargs)
        target, point_meta = _target_from_click(
            snapshot,
            kwargs.get("u"),
            kwargs.get("v"),
            kwargs.get("depth"),
        )
        arm = _select_arm(ctx, target, kwargs.get("arm", "right"))
        pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        pose[:3] = target
        result = _save_pose_plan(
            ctx,
            tool_name="plan_eef_translation_to_uvd_point",
            snapshot=snapshot,
            arm=arm,
            waypoints=[{"label": "target", "pose": pose.tolist()}],
            parameters={"point": point_meta, "pos_tol": kwargs.get("pos_tol", 0.012)},
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    _set_result(ctx, "plan_eef_translation_to_uvd_point", result)
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("plan_eef_translation_to_uvd_point", plan_eef_translation_to_uvd_point)
