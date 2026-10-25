from __future__ import annotations

from typing import Any
import numpy as np
from tools.util.contract import (
    validate_move_point_to_point_args,
)

from tools.util.motion_constants import X5_SHOULDERS
from tools.util.motion_common import (
    _current_snapshot,
    _drive_eef_target,
    _point,
    _select_arm,
    _set_result,
)


def move_point_to_point(
    ctx,
    session_id: str,
    image_id: str,
    points: list[Any],
    above_target_point_m: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 5.0,
    max_steps: int = 360,
    timeout_s: float = 90.0,
):
    try:
        arguments = validate_move_point_to_point_args(
            {
                "session_id": session_id,
                "image_id": image_id,
                "points": points,
                "above_target_point_m": above_target_point_m,
                "pos_tol": pos_tol,
                "ori_tol_deg": ori_tol_deg,
                "max_steps": max_steps,
                "timeout_s": timeout_s,
            }
        )
        snapshot = _current_snapshot(
            ctx,
            ctx.snapshots.load(arguments["session_id"], arguments["image_id"]),
        )
        source, source_meta = _point(snapshot, **arguments["points"][0])
        target, target_meta = _point(snapshot, **arguments["points"][1])
        arm = _select_arm(ctx, source)
        start_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        target_pose = start_pose.copy()
        delta = target - source
        target_pose[:3] += delta
        target_pose[2] += arguments["above_target_point_m"]
        shoulder_distance = float(np.linalg.norm(target_pose[:3] - X5_SHOULDERS[arm]))
        if shoulder_distance > 1.05:
            raise ValueError(
                "target EEF pose is outside the conservative X5 reach bound"
            )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        _set_result(
            ctx,
            "move_point_to_point",
            {"ok": False, "error": str(exc), "image_id": str(image_id or "")},
        )
        yield ctx.world.hold_action()
        return

    tracking = yield from _drive_eef_target(
        ctx,
        arm=arm,
        target=target_pose,
        pos_tolerance=arguments["pos_tol"],
        ori_tolerance_deg=arguments["ori_tol_deg"],
        max_steps=arguments["max_steps"],
        timeout_s=arguments["timeout_s"],
    )
    observed = np.asarray(tracking["observed_pose"], dtype=np.float64)
    position_error = float(tracking["position_error_m"])
    orientation_error = float(tracking["orientation_error_deg"])
    ok = bool(tracking["converged"])
    _set_result(
        ctx,
        "move_point_to_point",
        {
            "ok": ok,
            "error": None if ok else "observed EEF pose did not reach target",
            "arm": arm,
            "source_point": source_meta,
            "target_point": target_meta,
            "delta_robodojo_env_m": delta.tolist(),
            "target_eef_pose": target_pose.tolist(),
            "observed_eef_pose": observed.tolist(),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "above_target_point_m": arguments["above_target_point_m"],
            "action_steps": tracking["action_steps"],
            "elapsed_s": tracking["elapsed_s"],
            "stop_reason": tracking["stop_reason"],
            "tracking_trace": tracking["trace"],
            "max_joint_tracking_error_rad": tracking["max_joint_tracking_error_rad"],
            "joint_tracking_blocked_threshold_rad": tracking[
                "joint_tracking_blocked_threshold_rad"
            ],
            "collision_checked": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("move_point_to_point", move_point_to_point)
