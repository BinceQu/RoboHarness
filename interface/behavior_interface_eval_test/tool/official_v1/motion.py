"""Action-only tools implemented without simulator handles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .capture import camera_intrinsics
from .geometry import (
    load_frozen_capture,
    point_from_relative_uv,
    point_in_robot_frame,
    r1pro_shoulder_positions_robot,
    surface_normal_from_relative_uv,
)


CONTROL_HZ = 30.0
GRASP_PREP_Q = np.array(
    [
        0.0,
        0.0,
        0.0,
        -2.0943951024,
        0.0,
        -1.0471975512,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
R1PRO_UPRIGHT_TRUNK_Q = np.array(
    [0.45, -0.4, 0.0, 0.0],
    dtype=np.float64,
)
TRUNK_LIMITS = np.array(
    [
        [-1.1345, 1.8326],
        [-2.7900, 2.5300],
        [-1.8326, 1.5708],
        [-3.0500, 3.0500],
    ],
    dtype=np.float64,
)


def _check_cancelled(ctx, where: str) -> None:
    fn = getattr(ctx, "raise_if_cancelled", None)
    if callable(fn):
        fn(where)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _interpolate_controller(
    ctx,
    *,
    controller: str,
    start,
    target,
    max_step: float,
    min_steps: int = 2,
):
    start_q = np.asarray(start, dtype=np.float64).reshape(-1)
    target_q = np.asarray(target, dtype=np.float64).reshape(-1)
    gap = float(np.linalg.norm(target_q - start_q, ord=np.inf))
    steps = max(int(min_steps), int(math.ceil(gap / max(float(max_step), 1e-4))))
    for index in range(1, steps + 1):
        _check_cancelled(ctx, f"{controller} interpolation")
        alpha = _smoothstep(index / steps)
        command = start_q + alpha * (target_q - start_q)
        yield ctx.world.make_action(**{controller: command.tolist()})


def _drive_base(
    ctx,
    *,
    vx: float = 0.0,
    wz: float = 0.0,
    duration_s: float,
    timeout_s: float,
):
    duration = min(max(0.0, float(duration_s)), max(0.0, float(timeout_s)))
    exact_steps = duration * CONTROL_HZ
    full_steps = int(math.floor(exact_steps + 1e-9))
    remainder_scale = float(np.clip(exact_steps - full_steps, 0.0, 1.0))
    for _ in range(full_steps):
        _check_cancelled(ctx, "base action")
        yield ctx.world.set_base_velocity(float(vx), 0.0, float(wz))
    if remainder_scale > 1e-9:
        _check_cancelled(ctx, "base action remainder")
        yield ctx.world.set_base_velocity(
            float(vx) * remainder_scale,
            0.0,
            float(wz) * remainder_scale,
        )
    for _ in range(3):
        yield ctx.world.set_base_velocity(0.0, 0.0, 0.0)
    steps = full_steps + int(remainder_scale > 1e-9)
    return {
        "steps": steps,
        "duration_s": duration,
        "control_span_s": steps / CONTROL_HZ,
        "vx_mps": float(vx),
        "wz_radps": float(wz),
        "last_step_scale": remainder_scale if steps > full_steps else 1.0,
    }


def _drive_spin(
    ctx,
    spin_deg: float,
    *,
    wmax: float,
    timeout_s: float,
):
    radians = math.radians(float(spin_deg))
    speed = max(0.10, min(abs(float(wmax)), 1.0))
    command = math.copysign(speed, radians) if abs(radians) > 1e-9 else 0.0
    stats = yield from _drive_base(
        ctx,
        wz=command,
        duration_s=abs(radians) / speed if speed > 0.0 else 0.0,
        timeout_s=timeout_s,
    )
    stats.update(
        {
            "requested_spin_deg": float(spin_deg),
            "command_odometry_only": True,
        }
    )
    return stats


def _drive_forward(
    ctx,
    distance_m: float,
    *,
    vmax: float,
    timeout_s: float,
):
    distance = float(distance_m)
    speed = max(0.08, min(abs(float(vmax)), 0.5))
    command = math.copysign(speed, distance) if abs(distance) > 1e-9 else 0.0
    stats = yield from _drive_base(
        ctx,
        vx=command,
        duration_s=abs(distance) / speed if speed > 0.0 else 0.0,
        timeout_s=timeout_s,
    )
    stats.update(
        {
            "requested_forward_m": distance,
            "command_odometry_only": True,
        }
    )
    return stats


def _trunk_upward_target(q_start: np.ndarray, upward_m: float) -> np.ndarray:
    q1, q2, q3, q4 = np.asarray(q_start, dtype=np.float64).reshape(4)
    a12 = q1 + q2
    pitch = a12 - q3
    current_z = (
        0.343
        + 0.400 * math.cos(q1)
        + 0.300 * math.cos(a12)
        + 0.100 * math.cos(pitch)
    )
    target_z = current_z + float(upward_m)
    rhs = (target_z - 0.643 - 0.100 * math.cos(q3)) / 0.400
    if rhs < -1.0 or rhs > 1.0:
        raise ValueError(
            f"requested upward target chest_z={target_z:.3f}m is unreachable"
        )
    q1_abs = math.acos(float(np.clip(rhs, -1.0, 1.0)))
    sign = -1.0 if q1 < -0.05 else 1.0
    q1_target = sign * q1_abs
    q1_limit = abs(TRUNK_LIMITS[0, 0]) if sign < 0.0 else TRUNK_LIMITS[0, 1]
    if abs(q1_target) > q1_limit + 1e-9:
        raise ValueError(
            f"requested upward target requires q1={q1_target:.3f}rad outside limits"
        )
    target = np.array([q1_target, -q1_target, q3, q4], dtype=np.float64)
    return np.clip(target, TRUNK_LIMITS[:, 0], TRUNK_LIMITS[:, 1])


def move_in_robot_coord(
    ctx,
    forward: float = 0.0,
    spin: float = 0.0,
    pitch: float = 0.0,
    upward: float = 0.0,
    vmax: float = 0.5,
    wmax: float = 1.0,
    k_ang: float = 2.0,
    yaw_tol_deg: float = 3.0,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 5.0,
    trunk_max_step_rad: float = 0.10,
    trunk_timeout_s: float = 30.0,
    timeout_s: float = 120.0,
    nav_guard: bool = False,
    extra_inflate: float = 0.04,
):
    del k_ang, yaw_tol_deg, z_tol, theta_z_tol_deg, trunk_timeout_s, extra_inflate
    report: dict[str, Any] = {
        "ok": True,
        "tool": "move_in_robot_coord",
        "tool_version": "official_v1",
        "requested": {
            "forward_m": float(forward),
            "spin_deg": float(spin),
            "pitch_deg": float(pitch),
            "upward_m": float(upward),
        },
        "nav_guard_requested": bool(nav_guard),
        "nav_guard_available": False,
        "verification": "proprio for joints; command odometry for base",
    }

    q_start = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(4)
    if abs(float(upward)) > 1e-9:
        try:
            upward_target = _trunk_upward_target(q_start, float(upward))
        except ValueError as exc:
            report.update({"ok": False, "phase": "upward", "error": str(exc)})
            ctx.set_result(report)
            yield ctx.world.hold_action()
            return
        yield from _interpolate_controller(
            ctx,
            controller="trunk",
            start=q_start,
            target=upward_target,
            max_step=float(trunk_max_step_rad),
        )
        report["upward_target_q"] = upward_target.astype(float).tolist()
        q_start = upward_target

    if abs(float(pitch)) > 1e-9:
        pitch_target = q_start.copy()
        pitch_target[2] = float(
            np.clip(
                pitch_target[2] + math.radians(float(pitch)),
                TRUNK_LIMITS[2, 0],
                TRUNK_LIMITS[2, 1],
            )
        )
        if abs(
            pitch_target[2] - q_start[2] - math.radians(float(pitch))
        ) > 1e-6:
            report.update(
                {
                    "ok": False,
                    "phase": "pitch",
                    "error": "pitch target exceeds R1Pro torso_joint3 limits",
                }
            )
            ctx.set_result(report)
            yield ctx.world.hold_action()
            return
        yield from _interpolate_controller(
            ctx,
            controller="trunk",
            start=q_start,
            target=pitch_target,
            max_step=float(trunk_max_step_rad),
        )
        report["pitch_target_q"] = pitch_target.astype(float).tolist()

    remaining = max(0.0, float(timeout_s))
    if abs(float(forward)) > 1e-9:
        forward_stats = yield from _drive_forward(
            ctx,
            float(forward),
            vmax=float(vmax),
            timeout_s=remaining,
        )
        report["forward_execution"] = forward_stats
        remaining = max(0.0, remaining - float(forward_stats["duration_s"]))
    if abs(float(spin)) > 1e-9:
        report["spin_execution"] = yield from _drive_spin(
            ctx,
            float(spin),
            wmax=float(wmax),
            timeout_s=remaining,
        )

    report["trunk_q_commanded"] = (
        ctx.world.trunk_pin_qpos_list() or ctx.world.trunk_qpos().astype(float).tolist()
    )
    ctx.set_result(report)
    yield ctx.world.hold_action()


def face_to_point(
    ctx,
    u: float,
    v: float,
    image_id: str = "",
    max_abs_spin: float = 60.0,
    min_abs_spin: float = 0.25,
    timeout_s: float = 150.0,
    session_id: str = "",
):
    del v
    intrinsics = camera_intrinsics("head", 720, 720)
    source = "official_head_nominal"
    if str(session_id or "").strip() and str(image_id or "").strip():
        try:
            capture = load_frozen_capture(session_id, image_id)
            intrinsics = capture.camera
            source = "capture_meta"
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            pass
    width = int(intrinsics["image_width"])
    px = float(u) / 1000.0 * max(0, width - 1)
    fx = float(intrinsics["fx"])
    cx = float(intrinsics["cx"])
    ray_angle_deg = math.degrees(math.atan((px - cx) / fx))
    spin_deg = -ray_angle_deg
    limit = abs(float(max_abs_spin))
    if limit > 0.0:
        spin_deg = float(np.clip(spin_deg, -limit, limit))
    if abs(spin_deg) < abs(float(min_abs_spin)):
        spin_deg = 0.0
    stats = yield from _drive_spin(
        ctx,
        spin_deg,
        wmax=1.0,
        timeout_s=float(timeout_s),
    )
    ctx.set_result(
        {
            "ok": True,
            "tool": "face_to_point",
            "tool_version": "official_v1",
            "input_u_relative": float(u),
            "input_u_px": px,
            "ray_angle_deg": ray_angle_deg,
            "spin_deg": spin_deg,
            "intrinsics_source": source,
            "execution": stats,
        }
    )
    yield ctx.world.hold_action()


def move_eef(
    ctx,
    upward: float = 0.0,
    forward: float = 0.0,
    leftward: float = 0.0,
    u=None,
    v=None,
    depth=None,
    gripper: str = "keep",
    arm: str = "right",
    **_kwargs,
):
    del upward, forward, leftward, u, v, depth
    arm_eff = str(arm).strip().lower()
    if arm_eff not in ("left", "right"):
        raise ValueError("arm must be left or right")
    mode = str(gripper).strip().lower()
    if mode not in ("open", "close", "keep"):
        ctx.set_result(
            {
                "ok": False,
                "tool": "move_eef",
                "tool_version": "official_v1",
                "arm": arm_eff,
                "error": "gripper must be open, close, or keep",
            }
        )
        yield ctx.world.hold_action()
        return
    before_qpos = ctx.world.gripper_qpos_list(arm_eff)
    if mode == "keep":
        ctx.set_result(
            {
                "ok": True,
                "tool": "move_eef",
                "tool_version": "official_v1",
                "arm": arm_eff,
                "gripper_requested": mode,
                "gripper_qpos_before": before_qpos,
                "gripper_qpos_after": ctx.world.gripper_qpos_list(arm_eff),
                "implementation": "official_action_profile_gripper_hold",
            }
        )
        yield ctx.world.hold_action()
        return
    command = 1.0 if mode == "open" else -1.0
    frames = 6 if command > 0.0 else 9
    for _ in range(frames):
        _check_cancelled(ctx, "gripper action")
        yield ctx.world.make_action(**{f"gripper_{arm_eff}": [command]})
    ctx.set_result(
        {
            "ok": True,
            "tool": "move_eef",
            "tool_version": "official_v1",
            "arm": arm_eff,
            "gripper_requested": mode,
            "gripper_qpos_before": before_qpos,
            "gripper_qpos_after": ctx.world.gripper_qpos_list(arm_eff),
            "implementation": "official_action_profile_gripper",
        }
    )
    yield ctx.world.hold_action()


def _settle_trunk_target(
    ctx,
    *,
    target: np.ndarray,
    tolerance_rad: float,
    timeout_s: float,
):
    target_q = np.asarray(target, dtype=np.float64).reshape(4)
    tolerance = max(1e-4, float(tolerance_rad))
    max_steps = max(
        3,
        min(
            int(math.ceil(max(0.1, float(timeout_s)) * CONTROL_HZ)),
            90,
        ),
    )
    stable = 0
    steps = 0
    for _ in range(max_steps):
        _check_cancelled(ctx, "trunk closed-loop settle")
        yield ctx.world.make_action_trunk_locked(target_q.tolist())
        steps += 1
        observed = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(4)
        error = target_q - observed
        if float(np.max(np.abs(error))) <= tolerance:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0

    achieved = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(4)
    error = target_q - achieved
    max_error = float(np.max(np.abs(error)))
    return {
        "converged": max_error <= tolerance,
        "settle_steps": int(steps),
        "tolerance_rad": tolerance,
        "max_abs_error_rad": max_error,
        "error_rad": error.astype(float).tolist(),
        "achieved_qpos": achieved.astype(float).tolist(),
        "final_command_qpos": target_q.astype(float).tolist(),
    }


def reset_body(
    ctx,
    timeout_s: float = 45.0,
    trunk_max_step: float = 0.06,
    shoulder_iters: int = 4,
    keep_ori_arm: str = "none",
    pitch_deg: float = 0.0,
    tolerance_rad: float = 0.035,
):
    del shoulder_iters, keep_ori_arm
    start = np.asarray(ctx.world.trunk_qpos(), dtype=np.float64).reshape(4)
    target = R1PRO_UPRIGHT_TRUNK_Q.copy()
    target[2] = -math.radians(float(pitch_deg))
    target = np.clip(target, TRUNK_LIMITS[:, 0], TRUNK_LIMITS[:, 1])
    yield from _interpolate_controller(
        ctx,
        controller="trunk",
        start=start,
        target=target,
        max_step=max(0.01, float(trunk_max_step)),
    )
    settle = yield from _settle_trunk_target(
        ctx,
        target=target,
        tolerance_rad=float(tolerance_rad),
        timeout_s=float(timeout_s),
    )
    result = {
        "ok": bool(settle["converged"]),
        "tool": "reset_body",
        "tool_version": "official_v1",
        "target_trunk_q": target.astype(float).tolist(),
        "arms_grippers": "pinned through action targets",
        "direct_simulator_mutation": False,
        **settle,
    }
    if not result["ok"]:
        result["error"] = (
            "trunk did not reach the requested reset tolerance using "
            "evaluator-proprio closed-loop action control"
        )
    ctx.set_result(result)
    yield ctx.world.make_action_trunk_locked(
        settle["final_command_qpos"]
    )


def set_arm_to_grasp_position(
    ctx,
    arm: str = "right",
    gripper: str | None = None,
    open_gripper: bool | None = None,
    max_dq_per_step: float = 0.30,
    tol: float = 0.08,
    timeout_s: float = 15.0,
    force_jointspace: bool = False,
):
    del force_jointspace
    arm_eff = str(arm).strip().lower()
    arms = ["left", "right"] if arm_eff == "both" else [arm_eff]
    if any(side not in ("left", "right") for side in arms):
        ctx.set_result(
            {
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "tool_version": "official_v1",
                "error": "arm must be left, right, or both",
            }
        )
        yield ctx.world.hold_action()
        return
    tolerance = float(tol)
    timeout = float(timeout_s)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be a positive finite value")
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be a positive finite value")
    mode = str(gripper or "keep").strip().lower()
    if open_gripper is True:
        mode = "open"
    starts: dict[str, np.ndarray] = {}
    for side in arms:
        starts[side] = np.asarray(
            ctx.world.arm_qpos_list(side),
            dtype=np.float64,
        ).reshape(-1)
    arm_dof = int(starts[arms[0]].size)
    if arm_dof > GRASP_PREP_Q.size or any(
        start.size != arm_dof for start in starts.values()
    ):
        raise ValueError("active arm dimensions do not match the grasp-prep target")
    target = GRASP_PREP_Q[:arm_dof].copy()
    max_step = max(0.03, min(float(max_dq_per_step), 0.30))
    interpolation_steps = max(
        2,
        int(
            math.ceil(
                max(
                    float(np.linalg.norm(target - start, ord=np.inf))
                    for start in starts.values()
                )
                / max_step
            )
        ),
    )
    timeout_steps = max(1, int(math.ceil(timeout * CONTROL_HZ)))
    action_steps = 0
    for index in range(1, interpolation_steps + 1):
        if action_steps >= timeout_steps:
            break
        _check_cancelled(ctx, "bilateral grasp preparation")
        alpha = _smoothstep(index / interpolation_steps)
        yield ctx.world.make_action(
            **{
                f"arm_{side}": (
                    starts[side] + alpha * (target - starts[side])
                ).tolist()
                for side in arms
            }
        )
        action_steps += 1

    stable_steps = 0
    while action_steps < timeout_steps:
        achieved = {
            side: np.asarray(
                ctx.world.arm_qpos_list(side),
                dtype=np.float64,
            ).reshape(arm_dof)
            for side in arms
        }
        errors = {side: target - achieved[side] for side in arms}
        if max(float(np.max(np.abs(error))) for error in errors.values()) <= tolerance:
            stable_steps += 1
            if stable_steps >= 3:
                break
        else:
            stable_steps = 0
        _check_cancelled(ctx, "bilateral grasp-prep convergence")
        yield ctx.world.make_action(
            **{f"arm_{side}": target.tolist() for side in arms}
        )
        action_steps += 1

    achieved = {
        side: np.asarray(
            ctx.world.arm_qpos_list(side),
            dtype=np.float64,
        ).reshape(arm_dof)
        for side in arms
    }
    errors = {side: target - achieved[side] for side in arms}
    max_error = max(float(np.max(np.abs(error))) for error in errors.values())
    converged = bool(max_error <= tolerance and stable_steps >= 3)
    if converged:
        for side in arms:
            ctx.world.set_arm_pin_qpos(side, target)
        if mode == "open":
            for _ in range(4):
                yield ctx.world.make_action(
                    **{f"gripper_{side}": [1.0] for side in arms}
                )
    ctx.set_result(
        {
            "ok": converged,
            "tool": "set_arm_to_grasp_position",
            "tool_version": "official_v1",
            "arms": arms,
            "target_qpos": {
                side: target.astype(float).tolist() for side in arms
            },
            "achieved_qpos": {
                side: achieved[side].astype(float).tolist() for side in arms
            },
            "error_rad": {
                side: errors[side].astype(float).tolist() for side in arms
            },
            "converged": converged,
            "tolerance_rad": tolerance,
            "timeout_s": timeout,
            "action_steps": int(action_steps),
            "stable_steps": int(stable_steps),
            "max_abs_error_rad": float(max_error),
            "gripper": mode,
            "execution": "observed_qpos_to_absolute_joint_targets",
            "collision_checked": False,
            "direct_simulator_mutation": False,
            **(
                {}
                if converged
                else {
                    "error": (
                        "arms did not reach the grasp-prep tolerance from "
                        "evaluator proprioception before timeout"
                    )
                }
            ),
        }
    )
    yield ctx.world.hold_action()


def move_base_to_point(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    nav_timeout_s: float = 120.0,
    ground_tol_m: float = 0.04,
    pos_tol_m: float = 0.12,
    max_forward_m=None,
):
    capture = load_frozen_capture(session_id, image_id)
    point_local, point_meta = point_from_relative_uv(capture, u, v)
    point_robot = point_in_robot_frame(ctx.world, point_local)
    normal, normal_meta = surface_normal_from_relative_uv(capture, u, v)
    max_slope_deg = 35.0
    normal_ok = abs(float(normal[2])) >= math.cos(math.radians(max_slope_deg))
    height_limit = max(0.20, 5.0 * abs(float(ground_tol_m)))
    height_ok = float(point_robot[2]) <= height_limit
    if not normal_ok or not height_ok:
        ctx.set_result(
            {
                "ok": False,
                "tool": "move_base_to_point",
                "tool_version": "official_v1",
                "error": "clicked RGB-D surface does not pass the observation-only floor test",
                "point_robot_m": point_robot.astype(float).tolist(),
                "surface": normal_meta,
                "max_ground_height_robot_m": height_limit,
                "floor_test": {
                    "normal_ok": normal_ok,
                    "height_ok": height_ok,
                    "source": "frozen_depth_only",
                },
            }
        )
        yield ctx.world.hold_action()
        return

    horizontal = float(np.linalg.norm(point_robot[:2]))
    spin_deg = math.degrees(math.atan2(point_robot[1], point_robot[0]))
    forward_m = max(0.0, horizontal - max(0.0, float(pos_tol_m)))
    if max_forward_m is not None:
        forward_m = min(forward_m, max(0.0, float(max_forward_m)))
    spin_stats = yield from _drive_spin(
        ctx,
        spin_deg,
        wmax=0.8,
        timeout_s=float(nav_timeout_s),
    )
    remaining = max(0.0, float(nav_timeout_s) - float(spin_stats["duration_s"]))
    forward_stats = yield from _drive_forward(
        ctx,
        forward_m,
        vmax=0.35,
        timeout_s=remaining,
    )
    ctx.set_result(
        {
            "ok": True,
            "tool": "move_base_to_point",
            "tool_version": "official_v1",
            "image_id": str(image_id),
            "point_local_m": point_local.astype(float).tolist(),
            "point_robot_m_at_start": point_robot.astype(float).tolist(),
            "point_observation": point_meta,
            "surface": normal_meta,
            "floor_test": {
                "ok": True,
                "source": "frozen_depth_only",
                "max_slope_deg": max_slope_deg,
                "max_ground_height_robot_m": height_limit,
            },
            "spin_deg": spin_deg,
            "forward_m": forward_m,
            "spin_execution": spin_stats,
            "forward_execution": forward_stats,
            "arrival_verification": "command_odometry_only",
            "dynamic_collision_truth": False,
        }
    )
    yield ctx.world.hold_action()


def mesure_shoulder_distance(
    ctx,
    session_id: str = "",
    object_name: str = "",
    image_id: str = "",
    u: int | None = None,
    v: int | None = None,
):
    del object_name
    capture = load_frozen_capture(session_id, image_id)
    point_local, point_meta = point_from_relative_uv(capture, float(u), float(v))
    target_robot = point_in_robot_frame(ctx.world, point_local)
    shoulders = r1pro_shoulder_positions_robot(ctx.world.trunk_qpos())
    distances = {
        side: float(np.linalg.norm(target_robot - shoulder))
        for side, shoulder in shoulders.items()
    }
    ctx.set_result(
        {
            "ok": True,
            "tool": "mesure_shoulder_distance",
            "tool_version": "official_v1",
            "target_source": "evaluator_depth_click",
            "target_robot_m": target_robot.astype(float).tolist(),
            "point_observation": point_meta,
            "left_shoulder_robot_m": shoulders["left"].astype(float).tolist(),
            "right_shoulder_robot_m": shoulders["right"].astype(float).tolist(),
            "left_shoulder_to_object_m": distances["left"],
            "right_shoulder_to_object_m": distances["right"],
            "static_model": "R1Pro URDF torso and shoulder joint origins",
        }
    )
    yield ctx.world.hold_action()


def blocked_tool(name: str, reason: str):
    def run(ctx, **_kwargs):
        ctx.set_result(
            {
                "ok": False,
                "tool": name,
                "tool_version": "official_v1",
                "error": reason,
            }
        )
        yield ctx.world.hold_action()

    run.__name__ = str(name)
    return run
