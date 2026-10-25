"""Shared robot functions used by: move_tracked_point.

All commands and geometry execute locally; no macro RPC or external project.
"""
from __future__ import annotations

from tools.util.action_constants import MOVE_POINT_TO_POINT_GRIPPER_SERVO_KD_N_PER_MPS
from tools.util.action_constants import MOVE_POINT_TO_POINT_GRIPPER_SERVO_KP_N_PER_M
from tools.util.action_constants import MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N
from tools.util.action_constants import MOVE_TRACKED_POINT_GRIPPER_OPEN_HOLD_THRESHOLD_M
from tools.util.action_constants import MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_DEADBAND_M
from tools.util.action_constants import MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N
from typing import Any
import numpy as np


def _move_point_gripper_action_overrides(
    ctx,
    capture_state: dict[str, Any],
    *,
    enable_open_recovery: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overrides: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for side in ("left", "right"):
        close_latched = bool(
            getattr(
                ctx.world,
                "gripper_close_keepalive_active",
                lambda _arm: False,
            )(side)
        )
        if close_latched:
            report[side] = {
                "mode": "verified_close_keepalive",
                "capture_qpos": capture_state["gripper_q"][side].tolist(),
            }
            continue
        if bool(
            getattr(
                ctx.world,
                "gripper_uses_effort",
                lambda _arm: False,
            )(side)
        ):
            current_q = np.asarray(
                ctx.world.gripper_qpos_list(side),
                dtype=np.float64,
            ).reshape(2)
            current_qvel = np.asarray(
                ctx.world.gripper_qvel_list(side),
                dtype=np.float64,
            ).reshape(2)
            error = capture_state["gripper_q"][side] - current_q
            effort = np.clip(
                MOVE_POINT_TO_POINT_GRIPPER_SERVO_KP_N_PER_M * error
                - MOVE_POINT_TO_POINT_GRIPPER_SERVO_KD_N_PER_MPS
                * current_qvel,
                -MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N,
                MOVE_POINT_TO_POINT_GRIPPER_SERVO_MAX_N,
            )
            outward_only = (
                capture_state["gripper_q"][side]
                >= MOVE_TRACKED_POINT_GRIPPER_OPEN_HOLD_THRESHOLD_M
            )
            boosted_open = np.zeros(2, dtype=bool)
            # Negative effort is the evaluator's assisted-close intent.  A
            # near-open capture target must never receive close intent merely
            # because its measured qvel points outward.  This clamp is part of
            # the ordinary hold and adds no evaluator step on the fast path.
            effort[outward_only] = np.maximum(effort[outward_only], 0.0)
            if enable_open_recovery:
                boosted_open = np.logical_and(
                    outward_only,
                    error > MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_DEADBAND_M,
                )
                effort[boosted_open] = (
                    MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N
                )
            overrides[f"gripper_effort_{side}"] = effort.astype(float).tolist()
            report[side] = {
                "mode": "capture_qpos_local_effort_servo",
                "capture_qpos": capture_state["gripper_q"][side].tolist(),
                "observed_qpos": current_q.astype(float).tolist(),
                "observed_qvel": current_qvel.astype(float).tolist(),
                "error_m": error.astype(float).tolist(),
                "command": effort.astype(float).tolist(),
                "outward_only_fingers": np.flatnonzero(outward_only)
                .astype(int)
                .tolist(),
                "boosted_open_fingers": np.flatnonzero(boosted_open)
                .astype(int)
                .tolist(),
                "outward_recovery_effort_n": float(
                    MOVE_TRACKED_POINT_GRIPPER_OPEN_RECOVERY_EFFORT_N
                ),
            }
        else:
            mean_q = float(np.mean(capture_state["gripper_q"][side]))
            command = float(np.clip(2.0 * mean_q / 0.05 - 1.0, -1.0, 1.0))
            overrides[f"gripper_{side}"] = [command]
            report[side] = {
                "mode": "capture_qpos_static_limit_position_command",
                "capture_qpos": capture_state["gripper_q"][side].tolist(),
                "command": [command],
            }
    return overrides, report

