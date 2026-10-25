"""
move_to(x, y, z=None, theta_x_deg=None, theta_z_deg=None)
move(dx, dy, dz, dthetax, dthetaz)

胸口位姿（chest = torso_link4）5D 控制接口：
  x, y, z           胸口在世界系的位置
  theta_x_deg       胸口 forward 与 +X 夹角（度，水平 yaw）
  theta_z_deg       胸口 forward 与 +Z 夹角（度，0=朝天 / 90=水平 / 180=朝地）

move_to(x, y, z=None, theta_x_deg=None, theta_z_deg=None)：
  绝对目标。z / theta_x_deg / theta_z_deg 都默认 None = 保留当前。
  典型：
    move_to(x=7.5, y=-0.1)                                  → 仅底盘平移
    move_to(x=7.5, y=-0.1, theta_x_deg=-90)                 → 平移 + 转身朝物体
    move_to(x=7.5, y=-0.1, z=1.0, theta_z_deg=120)          → 平移 + 弯腰俯视抓地面

move(dx, dy, dz, dthetax, dthetaz)：
  相对增量。所有参数默认 0，按需填写。

R1Pro trunk 物理约束（极其重要 —— 不正确控制会摔倒）：
  trunk 4 关节链：base -> t1(+Y) -> t2(+Y) -> t3(-Y) -> t4(+Z)
  几何（base frame）：
    chest_x = -0.079 - 0.4·sin(q1) - 0.3·sin(q1+q2) - 0.1·sin(q1+q2-q3)
    chest_z =  0.343 + 0.4·cos(q1) + 0.3·cos(q1+q2) + 0.1·cos(q1+q2-q3)
    chest_pitch = q1 + q2 - q3        （绕 base +Y，正 = 抬头）
    chest_yaw   = q4                  （绕 base +Z）

  **协同约束**：升降时必须 q1 + q2 = 0（即 q2 = -q1），否则胸口会前后偏移
  导致重心出 base footprint → 摔倒。chest_z 仅靠 q1 单独控制；chest_pitch
  仅靠 q3 单独控制（保持 q1+q2=0）。
    chest_z(q1 协同) = 0.443 + 0.4·cos(q1) + 0.1·cos(pitch_target)
    范围：q1 ∈ [-1.13, 1.13]  →  chest_z(base frame) ∈ [0.61, 1.143]
          + base_z (~0.05m) →  chest_z(world) ∈ [0.66, 1.20] m

实现策略：
  阶段 1（trunk）：dz / dthetaz 用解析 IK，q1=-q2 协同升降 + q3 控 pitch
  阶段 2（base 平移）：dx, dy 用差速 _drive_to_target 走（带 A* 避障）
  阶段 3（base yaw）：dthetax 用原地旋转补
"""

from __future__ import annotations

import importlib
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import behavior_interface.skills.base_forward_observation_guard as _forward_guard_mod

    # 该模块不在 RELOAD_MODULES 里：热加载 move_to 时 sys.modules 仍持有旧对象，
    # 下面函数内的 from ... import 会继续拿到旧的限速/净空实现。必须手动刷新。
    _forward_guard_mod = importlib.reload(_forward_guard_mod)
except (PermissionError, OSError):
    _forward_guard_mod = None  # admin-only file in NFS; ignore at startup time

from behavior_interface.skills import register_skill


def _challenge_action_only() -> bool:
    mode = str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE")
        or os.environ.get("INTERFACE_CHALLENGE_MODE")
        or ""
    ).strip().lower()
    return mode in {"train", "public_test", "hidden_test"}

_TRUNK_Q_LIMITS = (
    (-1.1345, 1.8326),   # torso_joint1
    (-2.7925, 2.5307),   # torso_joint2
    (-1.8326, 1.5708),   # torso_joint3
    (-3.0543, 3.0543),   # torso_joint4
)


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

# 移动到位后、返回相机 observation 前的物理稳定等待（秒）
MOVE_SETTLE_S = 1.0


def yield_move_settle(world, seconds: float = MOVE_SETTLE_S):
    """hold 仿真步直至 wall-clock 达到 seconds，让机体晃动衰减后再 capture。"""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        yield world.hold_action()


def _norm_angle(a: float) -> float:
    """把角度归一化到 [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _norm_angle_deg(a_deg: float) -> float:
    """把角度归一化到 (-180, 180]."""
    while a_deg > 180.0:
        a_deg -= 360.0
    while a_deg <= -180.0:
        a_deg += 360.0
    return a_deg


# ─────────────────────────────────────────────────────────────────────────────
# 底盘差速 drive-to-target（沿用旧逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _deadline_expired(deadline_mono: float | None) -> bool:
    return deadline_mono is not None and time.time() >= deadline_mono


def _set_nav_seg_fail(ctx, reason: str) -> None:
    setattr(ctx, "_nav_seg_fail", reason)


def _nav_seg_fail(ctx) -> str:
    return str(getattr(ctx, "_nav_seg_fail", "") or "")


def _refresh_nav_scene_graph(ctx, log_prefix: str = "move_to") -> bool:
    """执行前刷新 Scene Graph（含铰链门板 AABB），避免 5s 缓存过期。"""
    world = ctx.world
    try:
        sg = world.build_scene_graph(robot_radius=0.25)
        if sg is not None:
            world.current_scene_graph = sg
            n_obs = len(sg.free_region.obstacles)
            ctx.log(f"{log_prefix} scene_graph 已刷新 obstacles={n_obs}")
            return True
    except Exception as e:
        ctx.log(f"{log_prefix} scene_graph 刷新失败: {e}")
    return False


def _nav_point_free(
    ctx, x: float, y: float, extra_inflate: float,
) -> Tuple[bool, Optional[str]]:
    sg = getattr(ctx.world, "current_scene_graph", None)
    if sg is None:
        return True, None
    from behavior_interface.scene_graph import is_point_free
    return is_point_free(sg.free_region, x, y, extra_inflate=extra_inflate)


def _drive_to_target(ctx, x: float, y: float,
                     pos_tol: float, align_tol_deg: float,
                     vmax: float, wmax: float, k_lin: float, k_ang: float,
                     timeout_s: float, log_prefix: str,
                     deadline_mono: float | None = None,
                     extra_inflate: float = 0.0,
                     nav_guard: bool = True):
    """全向趋近路径点：机体系 vx/vy 同时工作，不禁止进入障碍膨胀区，不靠前瞻硬停。"""
    world = ctx.world
    trunk_hold_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    if hasattr(world, "set_trunk_pin_qpos"):
        world.set_trunk_pin_qpos(trunk_hold_q.tolist())
    t_start = time.time()
    stuck_x = stuck_y = None
    stuck_t = t_start
    _ = align_tol_deg  # 全向模式下不再「先对准再直走」
    while True:
        if _deadline_expired(deadline_mono):
            ctx.log(f"{log_prefix} DEADLINE reached")
            yield world.set_base_velocity(0.0, 0.0, 0.0)
            _set_nav_seg_fail(ctx, "deadline")
            return False
        if time.time() - t_start > timeout_s:
            ctx.log(f"{log_prefix} TIMEOUT after {timeout_s:.1f}s")
            yield world.set_base_velocity(0.0, 0.0, 0.0)
            _set_nav_seg_fail(ctx, "timeout")
            return False

        pose = world.robot_pose()
        xr, yr = float(pose.pos[0]), float(pose.pos[1])
        yaw = float(pose.yaw)

        dx, dy = x - xr, y - yr
        dist = math.hypot(dx, dy)

        now = time.time()
        if stuck_x is None:
            stuck_x, stuck_y, stuck_t = xr, yr, now
        elif nav_guard and now - stuck_t >= _STUCK_WINDOW_S:
            moved_xy = math.hypot(xr - stuck_x, yr - stuck_y)
            if moved_xy < _STUCK_XY_EPS_M:
                ctx.log(
                    f"{log_prefix} STUCK at ({xr:.2f},{yr:.2f}) "
                    f"{now - stuck_t:.1f}s 内 xy 仅 {moved_xy:.4f}m "
                    f"(<{_STUCK_XY_EPS_M}m)"
                )
                yield world.set_base_velocity(0.0, 0.0, 0.0)
                _set_nav_seg_fail(ctx, "stuck")
                return False
            stuck_x, stuck_y, stuck_t = xr, yr, now
        if dist < pos_tol:
            ctx.log(f"{log_prefix} DONE pos=({xr:.2f},{yr:.2f}) dist={dist:.3f}")
            for _ in range(3):
                yield world.set_base_velocity(0.0, 0.0, 0.0)
            _set_nav_seg_fail(ctx, "ok")
            return True

        # 世界系误差 → 机体系：可横移穿过冰箱/微波炉缝隙，无需原地对准
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        err_f = cos_y * dx + sin_y * dy
        err_l = -sin_y * dx + cos_y * dy
        vx = max(-vmax, min(vmax, k_lin * err_f))
        vy = max(-vmax, min(vmax, k_lin * err_l))
        vm = math.hypot(vx, vy)
        if vm > vmax:
            vx, vy = vx * vmax / vm, vy * vmax / vm

        target_yaw = math.atan2(dy, dx)
        d_yaw = _norm_angle(target_yaw - yaw)
        wz = max(-wmax, min(wmax, 0.35 * k_ang * d_yaw))

        ctx.set_status(
            f"holonomic vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} dist={dist:.2f}"
        )
        yield world.set_base_velocity(vx, vy, wz)
        if hasattr(world, "set_trunk_pin_qpos"):
            world.set_trunk_pin_qpos(trunk_hold_q.tolist())


# 段内卡住（如烤箱门突然打开）后倒车距离，再刷新 SG 并重规划
_STUCK_BACKUP_M = 0.32
_STUCK_MAX_RECOVERIES = 4
# 窗口内 xy 位移低于此值（米）才判 STUCK——极低阈值，避免误报
_STUCK_XY_EPS_M = 0.005
_STUCK_WINDOW_S = 5.0
_BODY_MIN_LIN_VEL = 0.12
_BODY_CLOSE_IN_MAX_M = 0.12
_BODY_STUCK_MIN_COMMANDS = 8
_BODY_STUCK_WINDOW_S = 0.80
_BODY_COLLISION_BRAKE_MIN_S = 0.30
_BODY_COLLISION_SETTLE_TIMEOUT_S = 14.0
_BODY_COLLISION_RETREAT_SPEED_MPS = 0.08
_BODY_COLLISION_RETREAT_MAX_SPEED_MPS = 0.32
_BODY_COLLISION_RETREAT_STALL_S = 0.60
_BODY_COLLISION_RETREAT_PROGRESS_RATIO = 0.20
_BODY_COLLISION_RETREAT_SPEED_GROWTH = 1.65
_BODY_COLLISION_RETREAT_SPEED_STEP_MPS = 0.06
_BODY_COLLISION_RETREAT_MAX_M = 0.06
# 骑上台面基座/门槛时轮子悬空，退到常规上限也落不下来，零速干等只会超时失败。
# 此时逐级放宽退让上限继续退，直到重新压实三点支撑。
_BODY_COLLISION_RETREAT_HARD_MAX_M = 0.25
_BODY_COLLISION_SETTLE_WAIT_S = 0.60
_BODY_POST_STOP_VERIFY_S = 1.20
_BODY_POST_STOP_SOFT_RISE_M = 0.010
_BODY_POST_STOP_SOFT_TILT_RISE_DEG = 2.0
_BODY_GROUND_REF_CACHE_MAX_DELTA_M = 0.15
# 地面参考单次允许的上调幅度：够覆盖地毯/地板接缝，又不足以把台面基座当地面。
_BODY_GROUND_REF_RISE_MAX_M = 0.015
_BODY_BOUNDARY_CREEP_SPEED_MPS = 0.05
_BODY_BOUNDARY_CREEP_STEP_M = 0.015
_BODY_BOUNDARY_REFINE_EPS_M = 0.004
_BODY_BOUNDARY_MAX_REFINES = 24
# 边界逼近专用的“便宜步”参数：厘米级试探不需要 1.2s 验地，退让只要够脱困。
# 否则每步固定开销 ~9s、退让 2cm 远大于步长 0.5cm，会在边界前反复来回。
_BODY_BOUNDARY_PROBE_VERIFY_S = 0.30
_BODY_BOUNDARY_PROBE_RETREAT_MAX_M = 0.015
# 回到历史稳定点的路径是走过且站稳过的，可用较快速度一段吃掉，无需逐厘米蠕行。
_BODY_BOUNDARY_RESUME_SPEED_MPS = 0.10
_BODY_BOUNDARY_RESUME_VERIFY_S = 0.20
# 连续这么多次试探都没能把稳定点推进 _BODY_BOUNDARY_MIN_GAIN_M，即认定已到边界。
_BODY_BOUNDARY_STALL_LIMIT = 2
_BODY_BOUNDARY_MIN_GAIN_M = 0.003
_BODY_BOUNDARY_HYSTERESIS_RESETS = 4
# 贴着障碍「低速磨」：指令还在给、实际几乎不动，既不满足 no_progress 也走不完，
# 只能磨到超时（实测 0.015m 磨了 15–33s）。用实测/指令速度比提前判边界。
_BODY_LOW_SPEED_RATIO = 0.35
_BODY_LOW_SPEED_STALL_S = 1.20
# 近距离段落的贴地观察时长：1.2s 在重场景要 9s 墙钟，边界处不值得。
_BODY_NEAR_VERIFY_S = 0.50
_BODY_NEAR_VERIFY_CLEARANCE_M = 0.10

# Public adjust_chassis must also run inside the official evaluator, where the
# only base feedback is proprioceptive qvel. These bounds deliberately do not
# depend on simulator pose, contact, segmentation, or scene geometry.
_CHASSIS_OBS_STALL_S = 0.65
_CHASSIS_OBS_STALL_RATIO = 0.22
_CHASSIS_OBS_MIN_SPEED_MPS = 0.012
_CHASSIS_OBS_RETREAT_INITIAL_MPS = 0.08
_CHASSIS_OBS_RETREAT_MAX_MPS = 0.32
_CHASSIS_OBS_RETREAT_STALL_S = 0.50
_CHASSIS_OBS_RETREAT_MIN_M = 0.025
_CHASSIS_OBS_RETREAT_MAX_M = 0.16
_CHASSIS_OBS_RETREAT_MARGIN_M = 0.015
_CHASSIS_OBS_SETTLE_S = 0.50
_CHASSIS_OBS_SETTLE_SPEED_MPS = 0.015
_CHASSIS_OBS_SETTLE_YAW_RATE_RADPS = 0.02
_CHASSIS_OBS_DEPTH_SAMPLE_S = 0.10
_CHASSIS_OBS_STOP_GAP_M = 0.004
_CHASSIS_OBS_ACTION_DISTANCE_FACTOR = 1.60
_CHASSIS_OBS_BLIND_PROBE_MPS = 0.02
# Challenge 2026 contract constants. Keeping these local avoids consulting the
# live simulator/controller object from the observation-only policy path.
_CHASSIS_OBS_CONTROL_HZ = 30.0
_CHASSIS_OBS_BASE_MAX_LIN_MPS = 0.75
_CHASSIS_OBS_BASE_MAX_ANG_RADPS = 1.0


def _recover_nav_stuck(
    ctx,
    *,
    backup_m: float = _STUCK_BACKUP_M,
    inflate_boost: float = 0.0,
    deadline_mono: float | None = None,
    log_prefix: str = "move_to",
    timeout_s: float = 20.0,
) -> bool:
    """物理卡住后：沿机头倒车 → 刷新 SG → 重规划（inflate 可逐次加大）。"""
    world = ctx.world
    dist = abs(float(backup_m))
    backup_deadline = None if _deadline_expired(deadline_mono) else deadline_mono
    ctx.log(
        f"{log_prefix} STUCK 脱困 #{int(getattr(ctx, '_nav_recover_n', 0))}: "
        f"倒车 {dist:.2f}m，刷新障碍图重规划 inflate+={inflate_boost:.2f}m"
    )
    ok = yield from _drive_body_forward(
        ctx,
        -dist,
        vmax=0.35,
        tol=0.08,
        timeout_s=timeout_s,
        log_prefix=f"{log_prefix}[backup]",
        nav_guard=False,
        deadline_mono=backup_deadline,
    )
    for _ in range(4):
        yield world.set_base_velocity(0.0, 0.0, 0.0)
    _refresh_nav_scene_graph(ctx, log_prefix)
    return bool(ok)


def _drive_body_forward_segment(
    ctx,
    distance_m: float,
    *,
    translation_m: float = 0.0,
    world_direction_xy: Optional[np.ndarray] = None,
    vmax: float = 0.5,
    tol: float = 0.06,
    timeout_s: float = 120.0,
    log_prefix: str = "move_robot",
    nav_guard: bool = True,
    extra_inflate: float = 0.0,
    deadline_mono: float | None = None,
    lookahead_m: float = 0.28,
    post_stop_verify_s: float | None = None,
    post_stop_early_exit: bool = False,
    retreat_max_m: float | None = None,
):
    """沿一个固定平面方向移动，使用实体底盘贴地保护。

    ``distance_m`` / ``translation_m`` 分别是起始本体坐标系的前向和
    左向位移。两者非零时每个 action 同时输出 ``vx`` / ``vy``，不会拆成
    先前进再横移。外层分段控制可用 ``world_direction_xy`` 锁定整条路径的
    世界方向，避免分段间因轻微 yaw 漂移改变路线。

    ``post_stop_verify_s`` / ``retreat_max_m`` 供边界逼近使用：厘米级试探步不需要
    1.2s 验地，退让也只要够脱困，否则每步固定开销会远大于步长本身。

    ``post_stop_early_exit`` 只给「开阔中途段」用：确认贴地后立刻继续行驶，
    不把剩余观察窗口空转完（见 ``_verify_post_stop_grounded``）。
    """
    from behavior_interface.skills.move_to_object_v2 import (
        _BASE_AIRBORNE_RISE_STOP_M,
        _BASE_AIRBORNE_RISE_VZ_GATE_M,
        _BASE_AIRBORNE_TILT_RISE_STOP_DEG,
        _BASE_AIRBORNE_UP_VEL_STOP_MPS,
        _BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M,
        _BASE_COLLISION_CONTACT_IMPULSE_HARD,
        _BASE_COLLISION_CONTACT_PERSIST_SAMPLES,
        _BASE_GROUNDED_MAX_ABS_VZ_MPS,
        _BASE_GROUNDED_MAX_PLANAR_SPEED_MPS,
        _BASE_GROUNDED_MAX_YAW_RATE_DPS,
        _BASE_GROUNDED_MIN_CONTACT_LINKS,
        _BASE_GROUNDED_MAX_TILT_RISE_DEG,
        _BASE_GROUNDED_SETTLE_CONFIRM_SAMPLES,
        _BASE_GROUNDED_STABLE_SAMPLES,
        _BASE_GROUNDED_Z_MARGIN_M,
        _BASE_GROUND_MONITOR_HZ,
        _BASE_MAX_LIN_ACCEL,
        _BASE_MIN_RESUME_SPEED_FACTOR,
        _BASE_RESUME_SPEED_FACTOR,
        _base_ground_contact_link_count,
        _base_motion_collision_hits,
        _base_yaw_rad,
        _norm_angle_rad,
        _base_physical_velocity_to_action,
        _base_physics_dt,
        _base_tilt_deg,
        _ramp_base_cmd,
    )

    world = ctx.world
    requested_local_xy = np.asarray(
        [float(distance_m), float(translation_m)], dtype=np.float64
    )
    target_abs = float(np.linalg.norm(requested_local_xy))
    if target_abs < 1e-6:
        return True
    verify_s = (
        _BODY_POST_STOP_VERIFY_S
        if post_stop_verify_s is None
        else max(0.05, float(post_stop_verify_s))
    )
    default_retreat_max_m = (
        _BODY_COLLISION_RETREAT_MAX_M
        if retreat_max_m is None
        else max(0.004, float(retreat_max_m))
    )
    trunk_hold_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    if hasattr(world, "set_trunk_pin_qpos"):
        world.set_trunk_pin_qpos(trunk_hold_q.tolist())
    requested_unit_local = requested_local_xy / target_abs
    # Short close-in moves are intentionally only a few centimeters. The old
    # fixed 6 cm tolerance accepted them before issuing any velocity command.
    # Require at least 60% of the requested displacement while preserving the
    # existing absolute tolerance for normal navigation distances.
    effective_tol = min(max(0.0, float(tol)), target_abs * 0.40)
    pose0 = world.robot_pose()
    x0, y0 = float(pose0.pos[0]), float(pose0.pos[1])
    yaw0 = float(pose0.yaw)
    if world_direction_xy is None:
        cos_y0, sin_y0 = math.cos(yaw0), math.sin(yaw0)
        heading0 = np.asarray(
            [
                cos_y0 * requested_unit_local[0]
                - sin_y0 * requested_unit_local[1],
                sin_y0 * requested_unit_local[0]
                + cos_y0 * requested_unit_local[1],
            ],
            dtype=np.float64,
        )
    else:
        heading0 = np.asarray(world_direction_xy, dtype=np.float64).reshape(2)
        heading_norm = float(np.linalg.norm(heading0))
        if not math.isfinite(heading_norm) or heading_norm <= 1e-9:
            raise ValueError("world_direction_xy must be a finite non-zero vector")
        heading0 = heading0 / heading_norm
    target_world_xy = np.asarray([x0, y0], dtype=np.float64) + heading0 * target_abs
    primary_axis = 0 if abs(float(requested_unit_local[0])) >= 1e-9 else 1
    if abs(float(requested_unit_local[0])) > 1e-9 and abs(
        float(requested_unit_local[1])
    ) > 1e-9:
        motion_label = "diagonal"
    elif primary_axis == 0:
        motion_label = "forward" if requested_unit_local[0] > 0.0 else "reverse"
    else:
        motion_label = "left" if requested_unit_local[1] > 0.0 else "right"
    z0 = float(pose0.pos[2])
    t_start = time.time()
    command_steps = 0
    physics_steps = 0
    physics_dt = _base_physics_dt(world)
    monitor_steps = max(
        1, int(round(1.0 / (_BASE_GROUND_MONITOR_HZ * physics_dt)))
    )
    last_monitor_step = -monitor_steps
    last_monitor_z = z0
    cached_ground_z = getattr(world, "_codex_base_ground_z_ref", None)
    try:
        cached_ground_z = float(cached_ground_z)
    except (TypeError, ValueError):
        cached_ground_z = None
    if (
        cached_ground_z is not None
        and math.isfinite(cached_ground_z)
        and z0 >= cached_ground_z - _BASE_GROUNDED_Z_MARGIN_M
        and z0 - cached_ground_z <= _BODY_GROUND_REF_CACHE_MAX_DELTA_M
    ):
        ground_z_ref = min(z0, cached_ground_z)
    else:
        ground_z_ref = z0
    z_min = z0
    z_max = z0
    max_ground_rise = 0.0
    max_abs_vz = 0.0
    try:
        initial_tilt_deg = _base_tilt_deg(pose0.quat)
    except Exception:
        initial_tilt_deg = 0.0
    max_tilt_deg = initial_tilt_deg
    max_tilt_rise_deg = 0.0
    recovering_ground = False
    grounded_stable_samples = 0
    ground_recoveries = 0
    speed_scale = 1.0
    last_physical_cmd = [0.0, 0.0, 0.0]
    contact_streak = 0
    no_progress_s = 0.0
    progress_ref_x = x0
    progress_ref_y = y0
    low_speed_s = 0.0
    speed_ref_x = x0
    speed_ref_y = y0
    collision_guard_reason = None
    collision_ground_settled = None
    collision_ground_contacts = None
    collision_retreat_m = 0.0
    collision_recovery_phase = None
    collision_retreat_sign = None
    collision_retreat_speed_max_mps = 0.0
    collision_retreat_speed_escalations = 0
    collision_retreat_qvel_max_mps = 0.0
    obstacle_limited = False
    obstacle_stop_reason = None
    safe_traveled_m = 0.0
    post_stop_verified = False
    post_stop_steps = 0
    verify_budget_steps = max(
        _BASE_GROUNDED_STABLE_SAMPLES,
        int(math.ceil(verify_s / physics_dt)),
    )
    last_stable_progress_m = 0.0
    first_unstable_progress_m = None
    odom_progress_m = 0.0
    # 诊断用：base_qvel 平面速度模长积分（与坐标系约定无关），
    # 与位姿位移对比即可判断 odom 分量取法是否正确。
    odom_mag_m = 0.0
    max_cmd_speed = 0.0

    def _world_vector_to_local(pose_value, world_vector: np.ndarray) -> np.ndarray:
        yaw_value = float(pose_value.yaw)
        cos_y, sin_y = math.cos(yaw_value), math.sin(yaw_value)
        world_vector = np.asarray(world_vector, dtype=np.float64).reshape(2)
        return np.asarray(
            [
                cos_y * world_vector[0] + sin_y * world_vector[1],
                -sin_y * world_vector[0] + cos_y * world_vector[1],
            ],
            dtype=np.float64,
        )

    def _world_axis_to_local(pose_value, axis_sign: float = 1.0) -> np.ndarray:
        return _world_vector_to_local(
            pose_value, float(axis_sign) * heading0
        )

    def _signed_primary_component(local_axis: np.ndarray) -> float:
        value = float(np.asarray(local_axis, dtype=np.float64)[primary_axis])
        return 1.0 if value >= 0.0 else -1.0

    def _requested_progress(pose_value) -> float:
        delta = np.asarray(
            [
                float(pose_value.pos[0]) - x0,
                float(pose_value.pos[1]) - y0,
            ],
            dtype=np.float64,
        )
        return float(np.dot(delta, heading0))

    def _positive_progress(pose_value) -> float:
        return max(0.0, _requested_progress(pose_value))

    def _zero_action():
        return world.set_base_velocity(
            *_base_physical_velocity_to_action(world, [0.0, 0.0, 0.0])
        )

    def _note_stable_progress(pose_value) -> None:
        nonlocal last_stable_progress_m
        last_stable_progress_m = max(
            last_stable_progress_m, _positive_progress(pose_value)
        )

    def _store_ground_report(*, safety_abort: bool = False) -> None:
        setattr(ctx, "_body_ground_guard_report", {
            "monitor_hz": float(_BASE_GROUND_MONITOR_HZ),
            "ground_recoveries": int(ground_recoveries),
            "recovering_ground": bool(recovering_ground),
            "safety_abort": bool(safety_abort),
            "speed_scale": round(float(speed_scale), 4),
            "z0_m": round(z0, 4),
            "z_min_m": round(z_min, 4),
            "z_max_m": round(z_max, 4),
            "max_ground_rise_m": round(max_ground_rise, 4),
            "max_abs_vz_mps": round(max_abs_vz, 3),
            "initial_tilt_deg": round(initial_tilt_deg, 2),
            "max_tilt_deg": round(max_tilt_deg, 2),
            "max_tilt_rise_deg": round(max_tilt_rise_deg, 2),
            "physics_steps": int(physics_steps),
            "ground_z_ref_m": round(float(ground_z_ref), 4),
            "cached_ground_z_m": (
                round(float(cached_ground_z), 4)
                if cached_ground_z is not None else None
            ),
            "collision_guard_reason": collision_guard_reason,
            "collision_ground_settled": collision_ground_settled,
            "collision_ground_contacts": collision_ground_contacts,
            "collision_retreat_m": round(float(collision_retreat_m), 4),
            "collision_recovery_phase": collision_recovery_phase,
            "collision_retreat_sign": collision_retreat_sign,
            "collision_retreat_speed_max_mps": round(
                float(collision_retreat_speed_max_mps), 3
            ),
            "collision_retreat_speed_escalations": int(
                collision_retreat_speed_escalations
            ),
            "collision_retreat_qvel_max_mps": round(
                float(collision_retreat_qvel_max_mps), 3
            ),
            "obstacle_limited": bool(obstacle_limited),
            "obstacle_stop_reason": obstacle_stop_reason,
            "requested_m": round(float(target_abs), 4),
            "requested_forward_m": round(float(distance_m), 4),
            "requested_translation_m": round(float(translation_m), 4),
            "safe_traveled_m": round(float(safe_traveled_m), 4),
            "safe_forward_m": round(
                float(safe_traveled_m * requested_unit_local[0]), 4
            ),
            "safe_translation_m": round(
                float(safe_traveled_m * requested_unit_local[1]), 4
            ),
            "linear_control": "simultaneous_robot_frame_xy",
            "last_stable_progress_m": round(float(last_stable_progress_m), 4),
            "first_unstable_progress_m": (
                None
                if first_unstable_progress_m is None
                else round(float(first_unstable_progress_m), 4)
            ),
            "odom_progress_m": round(float(odom_progress_m), 4),
            "post_stop_verified": bool(post_stop_verified),
            "post_stop_steps": int(post_stop_steps),
        })

    def _cache_ground_z(
        z_value: float, *, contacts: Optional[int] = None
    ) -> None:
        """更新地面高度参考。

        地面参考若只取历史最低值，机器人走上地毯/地板接缝后就永远相对偏高，
        会被离地守护当成「离地」而在通畅处硬停。因此允许向上跟随真实地面，
        但只在确认三点支撑时、且单次上调不超过 _BODY_GROUND_REF_RISE_MAX_M，
        否则「骑在障碍上」会被当成新地面，离地保护就失效了。
        """
        nonlocal cached_ground_z, ground_z_ref
        z_value = float(z_value)
        if not math.isfinite(z_value):
            return
        prev_ref = float(ground_z_ref)
        rise = z_value - prev_ref
        if rise > 0.0:
            if (
                contacts is None
                or int(contacts) < _BASE_GROUNDED_MIN_CONTACT_LINKS
                or rise > _BODY_GROUND_REF_RISE_MAX_M
            ):
                return
            ground_z_ref = z_value
            if rise >= 0.0005:
                ctx.log(
                    f"{log_prefix} [地面参考] 上调 "
                    f"{prev_ref:.4f}->{z_value:.4f}m "
                    f"(+{rise * 1000.0:.1f}mm) contacts={contacts}"
                )
        else:
            ground_z_ref = z_value
        cached_ground_z = float(ground_z_ref)
        try:
            world._codex_base_ground_z_ref = float(cached_ground_z)
        except Exception:
            pass

    def _brake_until_grounded(
        trigger_reason: str,
        *,
        retreat_sign: Optional[float] = None,
        retreat_max_m: Optional[float] = None,
    ):
        if retreat_max_m is None:
            retreat_max_m = default_retreat_max_m
        nonlocal physics_steps, z_min, z_max, max_ground_rise
        nonlocal max_abs_vz, max_tilt_deg, max_tilt_rise_deg
        nonlocal collision_ground_settled, collision_ground_contacts
        nonlocal collision_retreat_m, collision_recovery_phase
        nonlocal collision_retreat_sign
        nonlocal collision_retreat_speed_max_mps
        nonlocal collision_retreat_speed_escalations
        nonlocal collision_retreat_qvel_max_mps
        recovery_axis_sign = (
            -1.0 if retreat_sign is None
            else (1.0 if float(retreat_sign) >= 0.0 else -1.0)
        )
        retreat_limit = max(0.0, float(retreat_max_m))
        min_steps = max(
            1, int(math.ceil(_BODY_COLLISION_BRAKE_MIN_S / physics_dt))
        )
        timeout_steps = max(
            min_steps,
            int(math.ceil(_BODY_COLLISION_SETTLE_TIMEOUT_S / physics_dt)),
        )
        stable_samples = 0
        retreat_speed_mps = _BODY_COLLISION_RETREAT_SPEED_MPS
        retreat_stall_s = 0.0
        settle_wait_steps = 0
        settle_wait_limit = max(
            1, int(math.ceil(_BODY_COLLISION_SETTLE_WAIT_S / physics_dt))
        )
        phase = "brake"
        collision_recovery_phase = phase
        prev_pose = world.robot_pose()
        retreat_local_axis = _world_axis_to_local(
            prev_pose, recovery_axis_sign
        )
        collision_retreat_sign = _signed_primary_component(retreat_local_axis)
        prev_z = float(prev_pose.pos[2])
        retreat_start_xy = np.asarray(
            prev_pose.pos[:2], dtype=np.float64
        ).reshape(2)
        prev_xy = retreat_start_xy.copy()
        try:
            prev_yaw = _base_yaw_rad(prev_pose.quat)
        except Exception:
            prev_yaw = None
        max_planar_speed = 0.0
        max_yaw_rate_dps = 0.0
        last_contacts = _base_ground_contact_link_count(world)
        for settle_step in range(1, timeout_steps + 1):
            physics_steps += 1
            issued_retreat = phase == "retreat"
            if phase == "retreat":
                recovery_physical_cmd = [
                    float(retreat_local_axis[0]) * retreat_speed_mps,
                    float(retreat_local_axis[1]) * retreat_speed_mps,
                    0.0,
                ]
                collision_retreat_speed_max_mps = max(
                    collision_retreat_speed_max_mps,
                    retreat_speed_mps,
                )
                yield world.set_base_velocity(
                    *_base_physical_velocity_to_action(
                        world, recovery_physical_cmd
                    )
                )
            else:
                yield _zero_action()
            pose_now = world.robot_pose()
            retreat_local_axis = _world_axis_to_local(
                pose_now, recovery_axis_sign
            )
            z_settle = float(pose_now.pos[2])
            z_min = min(z_min, z_settle)
            z_max = max(z_max, z_settle)
            ground_z_ref = min(
                float(getattr(world, "_codex_base_ground_z_ref", z_settle)),
                z_settle,
            ) if hasattr(world, "_codex_base_ground_z_ref") else min(
                float(z0), z_settle
            )
            vz_settle = (z_settle - prev_z) / max(physics_dt, 1e-6)
            cur_xy = np.asarray(pose_now.pos[:2], dtype=np.float64).reshape(2)
            planar_speed = float(
                np.linalg.norm(cur_xy - prev_xy)
            ) / max(physics_dt, 1e-6)
            observed_retreat_speed = None
            if issued_retreat:
                try:
                    base_qvel = np.asarray(
                        world.base_qvel(), dtype=np.float64
                    ).reshape(3)
                    candidate = float(
                        np.dot(base_qvel[:2], retreat_local_axis)
                    )
                    if math.isfinite(candidate):
                        observed_retreat_speed = max(0.0, candidate)
                        collision_retreat_qvel_max_mps = max(
                            collision_retreat_qvel_max_mps,
                            observed_retreat_speed,
                        )
                except Exception:
                    observed_retreat_speed = None
            try:
                cur_yaw = _base_yaw_rad(pose_now.quat)
            except Exception:
                cur_yaw = None
            if prev_yaw is None or cur_yaw is None:
                yaw_rate_dps = 0.0
            else:
                yaw_rate_dps = abs(
                    math.degrees(_norm_angle_rad(cur_yaw - prev_yaw))
                ) / max(physics_dt, 1e-6)
            try:
                tilt_settle = _base_tilt_deg(pose_now.quat)
            except Exception:
                tilt_settle = initial_tilt_deg
            tilt_rise_settle = max(0.0, tilt_settle - initial_tilt_deg)
            max_ground_rise = max(
                max_ground_rise, z_settle - float(ground_z_ref)
            )
            max_abs_vz = max(max_abs_vz, abs(vz_settle))
            if phase != "retreat":
                max_planar_speed = max(max_planar_speed, planar_speed)
                max_yaw_rate_dps = max(max_yaw_rate_dps, yaw_rate_dps)
            max_tilt_deg = max(max_tilt_deg, tilt_settle)
            max_tilt_rise_deg = max(
                max_tilt_rise_deg, tilt_rise_settle
            )
            last_contacts = _base_ground_contact_link_count(world)
            z_grounded = bool(
                z_settle <= float(ground_z_ref) + _BASE_GROUNDED_Z_MARGIN_M
            )
            contact_grounded = bool(
                last_contacts is not None
                and last_contacts >= _BASE_GROUNDED_MIN_CONTACT_LINKS
            )
            surface_grounded = bool(
                z_grounded
                and (
                    contact_grounded
                    if last_contacts is not None
                    else True
                )
            )
            # 水平静止只在零速阶段才有意义；retreat 阶段是主动倒车。
            planar_settled = bool(
                phase == "retreat"
                or (
                    planar_speed <= _BASE_GROUNDED_MAX_PLANAR_SPEED_MPS
                    and yaw_rate_dps <= _BASE_GROUNDED_MAX_YAW_RATE_DPS
                )
            )
            dynamics_stable = bool(
                abs(vz_settle) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                and tilt_rise_settle <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
                and planar_settled
            )
            if phase == "brake":
                if settle_step >= min_steps and surface_grounded:
                    stable_samples = (
                        stable_samples + 1 if dynamics_stable else 0
                    )
                elif settle_step >= min_steps:
                    phase = "retreat"
                    collision_recovery_phase = phase
                    stable_samples = 0
                    retreat_stall_s = 0.0
                    ctx.log(
                        f"{log_prefix} [底盘碰撞守护] 零速无法落地，"
                        "开始反向低速退让 "
                        f"v=({retreat_local_axis[0] * _BODY_COLLISION_RETREAT_SPEED_MPS:+.2f},"
                        f"{retreat_local_axis[1] * _BODY_COLLISION_RETREAT_SPEED_MPS:+.2f})m/s "
                        f"max={retreat_limit:.2f}m"
                    )
            elif phase == "retreat":
                current_xy = np.asarray(
                    pose_now.pos[:2], dtype=np.float64
                ).reshape(2)
                collision_retreat_m = max(
                    collision_retreat_m,
                    float(np.linalg.norm(current_xy - retreat_start_xy)),
                )
                if surface_grounded:
                    phase = "settle"
                    collision_recovery_phase = phase
                    stable_samples = 0
                    ctx.log(
                        f"{log_prefix} [底盘碰撞守护] 反向退让检测到落地，"
                        f"立即零速 retreat={collision_retreat_m:.3f}m "
                        f"z={z_settle:.4f} contacts={last_contacts}"
                    )
                elif (
                    collision_retreat_m
                    >= retreat_limit
                ):
                    phase = "settle"
                    collision_recovery_phase = phase
                    stable_samples = 0
                    ctx.log(
                        f"{log_prefix} [底盘碰撞守护] 反向退让达到上限，"
                        f"立即零速 retreat={collision_retreat_m:.3f}m"
                    )
                elif issued_retreat:
                    # 5020 的失败不是检测太晚，而是固定 0.08m/s 在轮子受压时
                    # 14 秒只退了 8mm。用 evaluator 允许的 base_qvel 判断退让
                    # 是否真正发生；无进展时逐级增力，但始终受 0.32m/s 硬上限约束。
                    motion_evidence = (
                        planar_speed
                        if observed_retreat_speed is None
                        else observed_retreat_speed
                    )
                    min_expected = max(
                        _CHASSIS_OBS_MIN_SPEED_MPS,
                        _BODY_COLLISION_RETREAT_PROGRESS_RATIO
                        * retreat_speed_mps,
                    )
                    retreat_stall_s = (
                        retreat_stall_s + physics_dt
                        if motion_evidence < min_expected
                        else 0.0
                    )
                    if (
                        retreat_stall_s >= _BODY_COLLISION_RETREAT_STALL_S
                        and retreat_speed_mps
                        < _BODY_COLLISION_RETREAT_MAX_SPEED_MPS - 1e-9
                    ):
                        old_speed = retreat_speed_mps
                        retreat_speed_mps = min(
                            _BODY_COLLISION_RETREAT_MAX_SPEED_MPS,
                            max(
                                old_speed
                                * _BODY_COLLISION_RETREAT_SPEED_GROWTH,
                                old_speed
                                + _BODY_COLLISION_RETREAT_SPEED_STEP_MPS,
                            ),
                        )
                        collision_retreat_speed_escalations += 1
                        retreat_stall_s = 0.0
                        ctx.log(
                            f"{log_prefix} [底盘碰撞守护] 反向退让无进展，"
                            f"自适应增速 {old_speed:.2f}->{retreat_speed_mps:.2f}m/s "
                            f"qvel={observed_retreat_speed} "
                            f"planar={planar_speed:.3f}m/s "
                            f"retreat={collision_retreat_m:.3f}/{retreat_limit:.3f}m"
                        )
            else:
                stable_samples = (
                    stable_samples + 1
                    if surface_grounded and dynamics_stable else 0
                )
                # 零速等不到落地说明轮子还骑在障碍上，放宽上限继续退而不是干等超时。
                if not surface_grounded:
                    settle_wait_steps += 1
                    if (
                        settle_wait_steps >= settle_wait_limit
                        and retreat_limit < _BODY_COLLISION_RETREAT_HARD_MAX_M
                    ):
                        retreat_limit = min(
                            _BODY_COLLISION_RETREAT_HARD_MAX_M,
                            retreat_limit * 2.0 + 0.02,
                        )
                        settle_wait_steps = 0
                        phase = "retreat"
                        collision_recovery_phase = phase
                        retreat_stall_s = 0.0
                        ctx.log(
                            f"{log_prefix} [底盘碰撞守护] 零速仍未落地，"
                            f"放宽退让上限继续退 max={retreat_limit:.3f}m "
                            f"retreat={collision_retreat_m:.3f}m "
                            f"z={z_settle:.4f} contacts={last_contacts}"
                        )
                else:
                    settle_wait_steps = 0
            # 退让过说明真撞上了，残余挤压释放得慢，确认窗口要拉长；
            # 没撞过的普通刹车沿用原来的 3 采样，不增加日常开销。
            required_stable = (
                _BASE_GROUNDED_SETTLE_CONFIRM_SAMPLES
                if (collision_retreat_m > 0.0 or phase == "settle")
                else _BASE_GROUNDED_STABLE_SAMPLES
            )
            if stable_samples >= required_stable:
                collision_ground_settled = True
                collision_ground_contacts = last_contacts
                collision_recovery_phase = "grounded"
                _cache_ground_z(z_settle, contacts=last_contacts)
                try:
                    world._codex_base_collision_escape_axis = None
                    world._codex_base_collision_escape_sign = None
                except Exception:
                    pass
                ctx.log(
                    f"{log_prefix} [底盘碰撞守护] 刹车/退让后已稳定贴地 "
                    f"reason={trigger_reason} steps={settle_step} "
                    f"retreat={collision_retreat_m:.3f}m "
                    f"z={z_settle:.4f} vz={vz_settle:+.3f}m/s "
                    f"tilt={tilt_settle:.1f}° ground_contacts={last_contacts} "
                    f"planar={planar_speed:.4f}m/s yaw_rate={yaw_rate_dps:.2f}°/s "
                    f"confirm={stable_samples}/{required_stable}"
                )
                return True
            prev_z = z_settle
            prev_xy = cur_xy
            if cur_yaw is not None:
                prev_yaw = cur_yaw
        collision_ground_settled = False
        collision_ground_contacts = last_contacts
        collision_recovery_phase = "timeout"
        ctx.log(
            f"{log_prefix} [底盘碰撞守护] 刹车/退让后落地超时 "
            f"reason={trigger_reason} timeout={_BODY_COLLISION_SETTLE_TIMEOUT_S:.1f}s "
            f"retreat={collision_retreat_m:.3f}m "
            f"z={prev_z:.4f} ground_z={ground_z_ref:.4f} "
            f"ground_contacts={last_contacts} "
            f"max_planar={max_planar_speed:.4f}m/s "
            f"max_yaw_rate={max_yaw_rate_dps:.2f}°/s"
        )
        return False

    def _finish_obstacle_limited(reason: str) -> bool:
        nonlocal obstacle_limited, obstacle_stop_reason
        nonlocal safe_traveled_m, post_stop_verified
        pose_safe = world.robot_pose()
        # 如实报告退让后的当前贴地里程；外层用 last_stable/first_unstable 做逼近。
        safe_traveled_m = max(0.0, _requested_progress(pose_safe))
        _note_stable_progress(pose_safe)
        obstacle_limited = True
        obstacle_stop_reason = str(reason)
        post_stop_verified = True
        _store_ground_report()
        _set_nav_seg_fail(ctx, "obstacle_limited")
        ctx.log(
            f"{log_prefix} body MAX_SAFE_DISTANCE "
            f"requested={target_abs:.3f}m safe={safe_traveled_m:.3f}m "
            f"last_stable={last_stable_progress_m:.3f}m "
            f"first_unstable="
            f"{None if first_unstable_progress_m is None else round(first_unstable_progress_m, 3)} "
            f"reason={reason} retreat={collision_retreat_m:.3f}m "
            "final_grounded=True "
            f"[速度] cmd_max={max_cmd_speed:.3f} "
            f"avg={safe_traveled_m / max(1e-6, command_steps * physics_dt):.3f} "
            f"odom_mag="
            f"{odom_mag_m / max(1e-6, command_steps * physics_dt):.3f}m/s"
        )
        return True

    def _verify_post_stop_grounded():
        nonlocal physics_steps, z_min, z_max, max_ground_rise
        nonlocal max_abs_vz, max_tilt_deg, max_tilt_rise_deg
        nonlocal collision_guard_reason, post_stop_steps
        nonlocal post_stop_verified, collision_ground_settled
        nonlocal collision_ground_contacts, last_physical_cmd
        nonlocal first_unstable_progress_m
        verify_steps = verify_budget_steps
        stable_samples = 0
        max_stable_streak = 0
        stop_contact_streak = 0
        prev_pose = world.robot_pose()
        prev_z = float(prev_pose.pos[2])
        last_contacts = _base_ground_contact_link_count(world)
        ctx.log(
            f"{log_prefix} body 达到距离阈值，零速观察贴地 "
            f"verify={verify_s:.2f}s "
            f"ground_z={ground_z_ref:.4f}"
        )
        for verify_step in range(1, verify_steps + 1):
            physics_steps += 1
            post_stop_steps = verify_step
            last_physical_cmd = [0.0, 0.0, 0.0]
            yield _zero_action()
            pose_now = world.robot_pose()
            z_now = float(pose_now.pos[2])
            z_min = min(z_min, z_now)
            z_max = max(z_max, z_now)
            vz_now = (z_now - prev_z) / max(physics_dt, 1e-6)
            try:
                tilt_now = _base_tilt_deg(pose_now.quat)
            except Exception:
                tilt_now = initial_tilt_deg
            tilt_rise_now = max(0.0, tilt_now - initial_tilt_deg)
            ground_rise_now = z_now - float(ground_z_ref)
            max_ground_rise = max(max_ground_rise, ground_rise_now)
            max_abs_vz = max(max_abs_vz, abs(vz_now))
            max_tilt_deg = max(max_tilt_deg, tilt_now)
            max_tilt_rise_deg = max(
                max_tilt_rise_deg, tilt_rise_now
            )
            last_contacts = _base_ground_contact_link_count(world)
            collision_hits = (
                _base_motion_collision_hits(world) if nav_guard else []
            )
            stop_contact_streak = (
                stop_contact_streak + 1 if collision_hits else 0
            )
            contact_triggered = bool(
                collision_hits
                and (
                    float(collision_hits[0]["impulse"])
                    >= _BASE_COLLISION_CONTACT_IMPULSE_HARD
                    or stop_contact_streak
                    >= _BASE_COLLISION_CONTACT_PERSIST_SAMPLES
                )
            )
            contact_support_low = bool(
                last_contacts is not None
                and last_contacts < _BASE_GROUNDED_MIN_CONTACT_LINKS
            )
            delayed_lift = bool(
                ground_rise_now >= _BASE_AIRBORNE_RISE_STOP_M
                or (
                    ground_rise_now >= _BODY_POST_STOP_SOFT_RISE_M
                    and (
                        vz_now > 0.02
                        or tilt_rise_now
                        >= _BODY_POST_STOP_SOFT_TILT_RISE_DEG
                        or contact_support_low
                    )
                )
            )
            # Contact with a movable obstacle (notably a hinged door) is not a
            # grounding failure.  Only delayed lift / support loss requires a
            # rollback here; a grounded contact may remain at the requested
            # endpoint and is handled by measured progress in the drive loop.
            if nav_guard and delayed_lift:
                collision_guard_reason = "post_stop_airborne"
                first_unstable_progress_m = (
                    _positive_progress(pose_now)
                    if first_unstable_progress_m is None
                    else min(
                        first_unstable_progress_m,
                        _positive_progress(pose_now),
                    )
                )
                try:
                    world._codex_base_collision_escape_axis = -1.0
                    world._codex_base_collision_escape_sign = float(
                        _signed_primary_component(
                            _world_axis_to_local(pose_now, -1.0)
                        )
                    )
                except Exception:
                    pass
                ctx.log(
                    f"{log_prefix} [停止后贴地守护] 检测到延迟离地，"
                    "反向退让到稳定贴地 "
                    f"reason={collision_guard_reason} "
                    f"rise={ground_rise_now:+.4f}m "
                    f"vz={vz_now:+.3f}m/s "
                    f"tilt_rise={tilt_rise_now:+.1f}° "
                    f"ground_contacts={last_contacts}"
                )
                settled = yield from _brake_until_grounded(
                    collision_guard_reason,
                    retreat_sign=-1.0,
                )
                if settled:
                    post_stop_verified = True
                    return "obstacle"
                return "timeout"

            z_grounded = bool(
                z_now <= float(ground_z_ref) + _BASE_GROUNDED_Z_MARGIN_M
            )
            contact_grounded = bool(
                last_contacts is not None
                and last_contacts >= _BASE_GROUNDED_MIN_CONTACT_LINKS
            )
            surface_grounded = bool(
                z_grounded
                and (
                    contact_grounded
                    if last_contacts is not None
                    else True
                )
            )
            dynamics_stable = bool(
                abs(vz_now) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                and tilt_rise_now <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
            )
            stable_samples = (
                stable_samples + 1
                if surface_grounded and dynamics_stable else 0
            )
            max_stable_streak = max(max_stable_streak, stable_samples)
            prev_z = z_now
            if (
                post_stop_early_exit
                and max_stable_streak >= _BASE_GROUNDED_STABLE_SAMPLES
            ):
                # 开阔中途段：贴地已确认，窗口剩下的步数纯属空转（每步墙钟
                # 成本是仿真步长的 2~3 倍）。万一之后才出现延迟抬升，下一段
                # 一起步 20Hz 行驶守护就会接管；末段不走这条路径。
                break

        collision_ground_contacts = last_contacts
        # 只要观察期内曾连续稳定贴地即算通过：只看窗口结尾连续 N 步太脆弱，
        # 停车残余微动会把已经贴地的状态误判成未贴地，从而白退让、提前收工。
        # 观察期内真出现抬升，上面的 delayed_lift 分支已经会立刻接管。
        if max_stable_streak >= _BASE_GROUNDED_STABLE_SAMPLES:
            collision_ground_settled = True
            post_stop_verified = True
            _cache_ground_z(prev_z, contacts=last_contacts)
            return "grounded"

        if nav_guard:
            collision_guard_reason = "post_stop_not_grounded"
            ctx.log(
                f"{log_prefix} [停止后贴地守护] 观察期结束仍未稳定贴地，"
                "反向退让 "
                f"z={prev_z:.4f} ground_z={ground_z_ref:.4f} "
                f"ground_contacts={last_contacts}"
            )
            settled = yield from _brake_until_grounded(
                collision_guard_reason,
                retreat_sign=-1.0,
            )
            if settled:
                post_stop_verified = True
                return "obstacle"
        return "timeout"

    initial_ground_contacts = _base_ground_contact_link_count(world)
    if (
        initial_ground_contacts is not None
        and initial_ground_contacts >= _BASE_GROUNDED_MIN_CONTACT_LINKS
    ):
        _cache_ground_z(z0, contacts=initial_ground_contacts)
    preexisting_airborne = bool(
        z0 - ground_z_ref >= _BASE_AIRBORNE_RISE_STOP_M
        or (
            initial_ground_contacts is not None
            and initial_ground_contacts < _BASE_GROUNDED_MIN_CONTACT_LINKS
            and z0 - ground_z_ref >= _BASE_GROUNDED_Z_MARGIN_M
        )
    )
    stored_escape_axis = getattr(
        world, "_codex_base_collision_escape_axis", None
    )
    try:
        stored_escape_axis = (
            1.0 if float(stored_escape_axis) >= 0.0 else -1.0
        )
    except (TypeError, ValueError):
        stored_escape_axis = None

    _store_ground_report()
    ctx.log(
        f"{log_prefix} body {motion_label} "
        f"local=({distance_m:+.3f},{translation_m:+.3f})m "
        f"length={target_abs:.3f}m from ({x0:.2f},{y0:.2f}) "
        f"ground_guard={_BASE_GROUND_MONITOR_HZ:.0f}Hz"
    )
    while True:
        if _deadline_expired(deadline_mono):
            ctx.log(f"{log_prefix} body DEADLINE")
            yield _zero_action()
            _store_ground_report(safety_abort=recovering_ground)
            _set_nav_seg_fail(ctx, "deadline")
            return False
        if time.time() - t_start > timeout_s:
            reason = "ground_recovery_timeout" if recovering_ground else "timeout"
            ctx.log(
                f"{log_prefix} body TIMEOUT recovering_ground={recovering_ground} "
                f"[Z监控] z0={z0:.4f} zmin={z_min:.4f} zmax={z_max:.4f}"
            )
            yield _zero_action()
            _store_ground_report(safety_abort=recovering_ground)
            _set_nav_seg_fail(ctx, reason)
            return False

        pose = world.robot_pose()
        xr, yr = float(pose.pos[0]), float(pose.pos[1])
        z_now = float(pose.pos[2])
        z_min = min(z_min, z_now)
        z_max = max(z_max, z_now)
        just_recovered = False
        if physics_steps % monitor_steps == 0:
            sample_steps = max(1, physics_steps - last_monitor_step)
            sample_dt = max(physics_dt, sample_steps * physics_dt)
            vz = (z_now - last_monitor_z) / sample_dt
            try:
                tilt_deg = _base_tilt_deg(pose.quat)
            except Exception:
                tilt_deg = 0.0
            tilt_rise_deg = max(0.0, tilt_deg - initial_tilt_deg)
            ground_z_ref = min(ground_z_ref, z_now)
            ground_rise = z_now - ground_z_ref
            z_rise_from_start = z_now - z0
            max_ground_rise = max(max_ground_rise, ground_rise)
            max_abs_vz = max(max_abs_vz, abs(vz))
            max_tilt_deg = max(max_tilt_deg, tilt_deg)
            max_tilt_rise_deg = max(max_tilt_rise_deg, tilt_rise_deg)

            if recovering_ground:
                grounded_now = bool(
                    z_now <= ground_z_ref + _BASE_GROUNDED_Z_MARGIN_M
                    and abs(vz) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                    and tilt_rise_deg <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
                )
                grounded_stable_samples = (
                    grounded_stable_samples + 1 if grounded_now else 0
                )
                if grounded_stable_samples >= _BASE_GROUNDED_STABLE_SAMPLES:
                    recovering_ground = False
                    grounded_stable_samples = 0
                    last_physical_cmd = [0.0, 0.0, 0.0]
                    speed_scale = max(
                        _BASE_MIN_RESUME_SPEED_FACTOR,
                        _BASE_RESUME_SPEED_FACTOR ** ground_recoveries,
                    )
                    just_recovered = True
                    _cache_ground_z(
                        z_now,
                        contacts=_base_ground_contact_link_count(world),
                    )
                    _note_stable_progress(pose)
                    ctx.log(
                        f"{log_prefix} [离地守护] 已稳定落地，resume body "
                        f"recovery={ground_recoveries} z={z_now:.4f} "
                        f"vz={vz:+.3f}m/s tilt={tilt_deg:.1f}° "
                        f"tilt_rise={tilt_rise_deg:+.1f}° "
                        f"speed_scale={speed_scale:.2f}"
                    )
            else:
                if (
                    ground_rise <= _BASE_GROUNDED_Z_MARGIN_M
                    and abs(vz) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                    and tilt_rise_deg <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
                ):
                    _note_stable_progress(pose)
                stop_reasons = []
                if z_rise_from_start >= _BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M:
                    stop_reasons.append(
                        f"z_from_start={z_rise_from_start:.3f}"
                        f">={_BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M:.3f}"
                    )
                if ground_rise >= _BASE_AIRBORNE_RISE_STOP_M:
                    stop_reasons.append(
                        f"rise={ground_rise:.3f}>={_BASE_AIRBORNE_RISE_STOP_M:.3f}"
                    )
                if (
                    ground_rise >= _BASE_AIRBORNE_RISE_VZ_GATE_M
                    and vz >= _BASE_AIRBORNE_UP_VEL_STOP_MPS
                ):
                    stop_reasons.append(
                        f"rise={ground_rise:.3f},vz={vz:+.3f}"
                    )
                if tilt_rise_deg >= _BASE_AIRBORNE_TILT_RISE_STOP_DEG:
                    stop_reasons.append(
                        f"tilt_rise={tilt_rise_deg:.1f}"
                        f">={_BASE_AIRBORNE_TILT_RISE_STOP_DEG:.1f}"
                    )
                if stop_reasons:
                    if nav_guard:
                        ground_recoveries += 1
                        last_physical_cmd = [0.0, 0.0, 0.0]
                        preexisting_recovery = bool(
                            preexisting_airborne and command_steps == 0
                        )
                        collision_guard_reason = (
                            "preexisting_airborne_recovery"
                            if preexisting_recovery
                            else "airborne_or_tilt"
                        )
                        recovery_axis_sign = (
                            stored_escape_axis
                            if preexisting_recovery
                            and stored_escape_axis is not None
                            else (1.0 if preexisting_recovery else -1.0)
                        )
                        if not preexisting_recovery:
                            first_unstable_progress_m = (
                                _positive_progress(pose)
                                if first_unstable_progress_m is None
                                else min(
                                    first_unstable_progress_m,
                                    _positive_progress(pose),
                                )
                            )
                            try:
                                world._codex_base_collision_escape_axis = float(
                                    recovery_axis_sign
                                )
                                world._codex_base_collision_escape_sign = (
                                    float(
                                        _signed_primary_component(
                                            _world_axis_to_local(
                                                pose, recovery_axis_sign
                                            )
                                        )
                                    )
                                )
                            except Exception:
                                pass
                        recovery_local_axis = _world_axis_to_local(
                            pose, recovery_axis_sign
                        )
                        ctx.log(
                            f"{log_prefix} [离地守护] 20Hz检测触发"
                            + (
                                "预存离地恢复，允许请求方向脱困 "
                                if preexisting_recovery else
                                "碰撞急停，本次直线动作禁止原方向 resume "
                            )
                            + f"reason={';'.join(stop_reasons)} "
                            f"recovery_axis={recovery_axis_sign:+.0f} "
                            f"recovery_v=({recovery_local_axis[0]:+.2f},"
                            f"{recovery_local_axis[1]:+.2f}) "
                            f"z={z_now:.4f} ground_z={ground_z_ref:.4f} "
                            f"vz={vz:+.3f}m/s tilt={tilt_deg:.1f}° "
                            f"tilt0={initial_tilt_deg:.1f}° "
                            f"tilt_rise={tilt_rise_deg:+.1f}° "
                            f"last_stable={last_stable_progress_m:.3f}m"
                        )
                        settled = yield from _brake_until_grounded(
                            collision_guard_reason,
                            retreat_sign=recovery_axis_sign,
                            retreat_max_m=(
                                min(
                                    max(
                                        default_retreat_max_m,
                                        target_abs,
                                    ),
                                    0.12,
                                )
                                if preexisting_recovery
                                else default_retreat_max_m
                            ),
                        )
                        if (
                            preexisting_recovery
                            and settled
                            and recovery_axis_sign > 0.0
                        ):
                            preexisting_airborne = False
                            progress_ref_x, progress_ref_y = xr, yr
                            no_progress_s = 0.0
                            contact_streak = 0
                            last_physical_cmd = [0.0, 0.0, 0.0]
                            just_recovered = True
                            ctx.log(
                                f"{log_prefix} [离地守护] 预存离地已恢复，"
                                "继续完成用户请求的同向退让 "
                                f"requested_local=({requested_unit_local[0]:+.2f},"
                                f"{requested_unit_local[1]:+.2f}) "
                                f"recovered_retreat={collision_retreat_m:.3f}m"
                            )
                        elif settled:
                            return _finish_obstacle_limited(
                                collision_guard_reason
                            )
                        else:
                            _store_ground_report(safety_abort=True)
                            _set_nav_seg_fail(
                                ctx,
                                "stuck"
                                if settled else "ground_recovery_timeout",
                            )
                            return False
                    else:
                        recovering_ground = True
                        grounded_stable_samples = 0
                        ground_recoveries += 1
                        last_physical_cmd = [0.0, 0.0, 0.0]
                        ctx.log(
                            f"{log_prefix} [离地守护] 20Hz检测触发急停 "
                            f"recovery={ground_recoveries} "
                            f"reason={';'.join(stop_reasons)} "
                            f"z={z_now:.4f} ground_z={ground_z_ref:.4f} "
                            f"vz={vz:+.3f}m/s tilt={tilt_deg:.1f}° "
                            f"tilt0={initial_tilt_deg:.1f}° "
                            f"tilt_rise={tilt_rise_deg:+.1f}°"
                        )

            if nav_guard and not recovering_ground:
                collision_hits = _base_motion_collision_hits(world)
                contact_streak = contact_streak + 1 if collision_hits else 0
                max_impulse = (
                    float(collision_hits[0]["impulse"])
                    if collision_hits else 0.0
                )
                command_active = bool(
                    command_steps > 0
                    and math.hypot(
                        float(last_physical_cmd[0]),
                        float(last_physical_cmd[1]),
                    ) > 1e-3
                )
                contact_triggered = bool(
                    command_active
                    and collision_hits
                    and (
                        max_impulse >= _BASE_COLLISION_CONTACT_IMPULSE_HARD
                        or contact_streak
                        >= _BASE_COLLISION_CONTACT_PERSIST_SAMPLES
                    )
                )
                # 近距低速时不再用 _BODY_MIN_LIN_VEL 门槛，否则 3cm/s 蠕行永远不判卡住。
                cmd_speed = math.hypot(
                    float(last_physical_cmd[0]),
                    float(last_physical_cmd[1]),
                )
                if command_active and cmd_speed >= 0.02:
                    progress = math.hypot(
                        xr - progress_ref_x, yr - progress_ref_y
                    )
                    if progress >= _STUCK_XY_EPS_M:
                        progress_ref_x, progress_ref_y = xr, yr
                        no_progress_s = 0.0
                    else:
                        no_progress_s += sample_dt
                else:
                    progress_ref_x, progress_ref_y = xr, yr
                    no_progress_s = 0.0
                # 低速磨：实测里程持续远低于指令速度，说明已在推障碍，别再磨下去。
                if command_active and cmd_speed >= 0.005:
                    actual_m = math.hypot(xr - speed_ref_x, yr - speed_ref_y)
                    if actual_m < _BODY_LOW_SPEED_RATIO * cmd_speed * sample_dt:
                        low_speed_s += sample_dt
                    else:
                        low_speed_s = 0.0
                else:
                    low_speed_s = 0.0
                speed_ref_x, speed_ref_y = xr, yr
                stuck_triggered = bool(
                    command_steps >= _BODY_STUCK_MIN_COMMANDS
                    and no_progress_s >= _BODY_STUCK_WINDOW_S
                )
                low_speed_triggered = bool(
                    command_steps >= _BODY_STUCK_MIN_COMMANDS
                    and low_speed_s >= _BODY_LOW_SPEED_STALL_S
                )
                # External contact is expected while opening a door.  It may
                # cap the achieved speed, but it is not itself a stop reason.
                # Continue applying the requested vector while the base stays
                # grounded; only measured lack of progress or the independent
                # lift/tilt guard can end the probe.
                if stuck_triggered or low_speed_triggered:
                    collision_guard_reason = (
                        "no_progress"
                        if stuck_triggered
                        else "low_speed_stall"
                    )
                    first_unstable_progress_m = (
                        _positive_progress(pose)
                        if first_unstable_progress_m is None
                        else min(
                            first_unstable_progress_m,
                            _positive_progress(pose),
                        )
                    )
                    top_hit = collision_hits[0] if collision_hits else None
                    ctx.log(
                        f"{log_prefix} [底盘碰撞守护] 20Hz急停 "
                        f"reason={collision_guard_reason} "
                        f"traveled={_positive_progress(pose):.3f}m "
                        f"cmd=({last_physical_cmd[0]:+.2f},"
                        f"{last_physical_cmd[1]:+.2f}) "
                        f"no_progress={no_progress_s:.2f}s "
                        f"low_speed={low_speed_s:.2f}s "
                        f"contact_seen={contact_triggered} contact={top_hit}"
                    )
                    last_physical_cmd = [0.0, 0.0, 0.0]
                    try:
                        world._codex_base_collision_escape_axis = -1.0
                        world._codex_base_collision_escape_sign = float(
                            _signed_primary_component(
                                _world_axis_to_local(pose, -1.0)
                            )
                        )
                    except Exception:
                        pass
                    settled = yield from _brake_until_grounded(
                        collision_guard_reason,
                        retreat_sign=-1.0,
                    )
                    if settled:
                        return _finish_obstacle_limited(
                            collision_guard_reason
                        )
                    _store_ground_report(safety_abort=True)
                    _set_nav_seg_fail(ctx, "ground_recovery_timeout")
                    return False

            last_monitor_z = z_now
            last_monitor_step = physics_steps

        if recovering_ground or just_recovered:
            ctx.set_status(
                f"body ground recovery #{ground_recoveries}"
                if recovering_ground
                else f"body resume after recovery #{ground_recoveries}"
            )
            physics_steps += 1
            yield _zero_action()
            _store_ground_report()
            continue

        traveled = _positive_progress(pose)
        safe_traveled_m = traveled
        try:
            base_qv = np.asarray(world.base_qvel(), dtype=np.float64).reshape(3)
            yaw_now = float(pose.yaw)
            cos_y, sin_y = math.cos(yaw_now), math.sin(yaw_now)
            qvel_world = np.asarray(
                [
                    cos_y * base_qv[0] - sin_y * base_qv[1],
                    sin_y * base_qv[0] + cos_y * base_qv[1],
                ],
                dtype=np.float64,
            )
            odom_progress_m += float(
                np.dot(qvel_world, heading0)
            ) * float(physics_dt)
            odom_mag_m += float(
                math.hypot(float(base_qv[0]), float(base_qv[1]))
            ) * float(physics_dt)
        except Exception:
            pass
        current_xy = np.asarray(pose.pos[:2], dtype=np.float64).reshape(2)
        remaining_world_xy = target_world_xy - current_xy
        target_error_m = float(np.linalg.norm(remaining_world_xy))
        cross_track_m = float(
            np.dot(
                current_xy - np.asarray([x0, y0], dtype=np.float64),
                np.asarray([-heading0[1], heading0[0]], dtype=np.float64),
            )
        )
        if target_error_m <= effective_tol:
            if nav_guard:
                post_stop_status = yield from _verify_post_stop_grounded()
                if post_stop_status == "obstacle":
                    return _finish_obstacle_limited(
                        collision_guard_reason or "post_stop_obstacle"
                    )
                if post_stop_status != "grounded":
                    _store_ground_report(safety_abort=True)
                    _set_nav_seg_fail(ctx, "ground_recovery_timeout")
                    return False
            else:
                for _ in range(3):
                    physics_steps += 1
                    yield _zero_action()
                post_stop_verified = True
                post_stop_steps = 3
            pose_done = world.robot_pose()
            safe_traveled_m = _positive_progress(pose_done)
            _note_stable_progress(pose_done)
            drive_s = max(1e-6, command_steps * physics_dt)
            ctx.log(
                f"{log_prefix} body DONE traveled={safe_traveled_m:.3f}m "
                f"target={target_abs:.3f}m tol={effective_tol:.3f}m "
                f"pos=({float(pose_done.pos[0]):.2f},"
                f"{float(pose_done.pos[1]):.2f}) "
                f"ground_recoveries={ground_recoveries} "
                f"post_stop_verified={post_stop_verified} "
                f"verify_steps={post_stop_steps}/{verify_budget_steps}"
                f"{'(提前)' if post_stop_early_exit else ''} "
                f"[速度] cmd_max={max_cmd_speed:.3f} "
                f"avg={safe_traveled_m / drive_s:.3f} "
                f"odom_axis={odom_progress_m / drive_s:+.3f} "
                f"odom_mag={odom_mag_m / drive_s:.3f}m/s "
                f"drive_s={drive_s:.2f}"
            )
            _store_ground_report()
            _set_nav_seg_fail(ctx, "ok")
            return True

        rem = target_error_m
        # 剩余越短速度越低，避免 0.1m 满速撞击；但衰减律不能用 0.9*rem：
        # 那是指数收敛，0.55m 的减速段要跑 5s，而按 0.35m/s² 制动只需 1.8s。
        # 底盘实际制动能力是 _BASE_MAX_LIN_ACCEL(2.4)，这里的 0.35 已有近 7 倍
        # 裕度，下面的分档硬限速再兜底一层。
        # vmax 仍是硬上限，蠕行段传入的低 vmax 不会被这里抬高。
        speed = min(
            max(0.0, float(vmax)),
            max(0.05, math.sqrt(2.0 * 0.35 * rem)),
        )
        if rem <= 0.08:
            speed = min(speed, 0.10)
        if rem <= 0.04:
            speed = min(speed, 0.07)
        if rem <= 0.015:
            speed = min(speed, 0.05)
        if target_error_m > 1e-9:
            desired_world_unit = remaining_world_xy / target_error_m
            desired_local_unit = _world_vector_to_local(
                pose, desired_world_unit
            )
        else:
            desired_local_unit = np.zeros(2, dtype=np.float64)
        desired_physical_cmd = [
            float(desired_local_unit[0]) * speed * speed_scale,
            float(desired_local_unit[1]) * speed * speed_scale,
            0.0,
        ]
        physical_cmd = _ramp_base_cmd(
            last_physical_cmd,
            desired_physical_cmd,
            dt_s=physics_dt,
            max_lin_accel=_BASE_MAX_LIN_ACCEL,
            max_ang_accel=0.0,
        )
        last_physical_cmd = physical_cmd
        max_cmd_speed = max(
            max_cmd_speed,
            math.hypot(float(physical_cmd[0]), float(physical_cmd[1])),
        )
        action_cmd = _base_physical_velocity_to_action(world, physical_cmd)
        ctx.set_status(
            f"body {motion_label} vx={physical_cmd[0]:+.2f} "
            f"vy={physical_cmd[1]:+.2f} "
            f"traveled={traveled:.2f}/{target_abs:.2f} "
            f"cross={cross_track_m:+.3f}m"
        )
        command_steps += 1
        physics_steps += 1
        yield world.set_base_velocity(*action_cmd)
        if hasattr(world, "set_trunk_pin_qpos"):
            world.set_trunk_pin_qpos(trunk_hold_q.tolist())


# 单段上限只是「RGB-D 复查间隔」，真正的硬约束是 safe_advance（净空实测值）。
# 取 0.35 会让开阔区每 0.35m 就多付一次验地 + 渲染 + 加减速；重场景下这些固定
# 开销比行驶本身还贵，而 0.8m 仍在一帧深度能可靠覆盖的范围内。
_BODY_OBS_OPEN_LOOP_CHUNK_M = 0.80
_BODY_OBS_STOP_GAP_M = 0.003
_BODY_OBS_MAX_SEGMENTS = 64
# 净空充足的中途段不是边界，1.2s 验地纯属固定浪费；缩到 0.5s 仍有 10 个 20Hz 采样。
_BODY_OPEN_VERIFY_CLEARANCE_M = 0.35
_BODY_OPEN_VERIFY_S = 0.50


def _read_forward_observation(ctx, *, log_prefix: str) -> Dict[str, Any]:
    """Read an evaluator-contract-compatible head-depth clearance sample."""
    world = ctx.world
    try:
        adapter = getattr(world, "_official_adapter", None)
        if adapter is not None:
            depth = adapter.camera_depth_frames().get("head")
            camera_relative = adapter.camera_relative_poses().get("head")
            if depth is None or camera_relative is None:
                return {
                    "ok": False,
                    "error": "official head depth/cam_rel_pose unavailable",
                }
            intrinsics = {
                "focal_length": 17.0,
                "horizontal_aperture": 40.0,
            }
            source = "official_evaluator_observation"
        else:
            import importlib

            from behavior_interface.skills.reach_point_pitch_recovery import (
                head_camera_pose_robot_from_proprio,
            )

            # 热加载偶发让 head_capture 停留在旧模块对象上；这里强制取最新符号。
            head_capture = importlib.import_module(
                "behavior_interface.head_capture"
            )
            if not hasattr(head_capture, "head_factory_mount_pose"):
                head_capture = importlib.reload(head_capture)
            move_to_object_v2 = importlib.import_module(
                "behavior_interface.skills.move_to_object_v2"
            )
            read_rgbd = getattr(
                move_to_object_v2, "_read_current_head_rgbd_for_recovery", None
            )
            if read_rgbd is None:
                return {
                    "ok": False,
                    "error": "head RGB-D reader unavailable after reload",
                }
            current = read_rgbd(
                ctx, world, f"{log_prefix}.forward_guard"
            )
            if not current.get("ok"):
                return {
                    "ok": False,
                    "error": current.get("error", "head RGB-D unavailable"),
                }
            depth = current["depth"]
            camera_relative = head_camera_pose_robot_from_proprio(
                np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4),
                camera_parent_pos=current["camera_parent_pos"],
                camera_parent_quat=current["camera_parent_quat"],
            )
            intrinsics = current["intrinsics"]
            source = "standalone_allowed_observation_adapter"

        from behavior_interface.skills.base_forward_observation_guard import (
            fit_floor_plane_from_depth,
            forward_clearance_from_depth,
        )

        clearance = forward_clearance_from_depth(
            depth,
            camera_relative_pose=camera_relative,
            focal_length=float(intrinsics["focal_length"]),
            horizontal_aperture=float(intrinsics["horizontal_aperture"]),
        )
        floor = fit_floor_plane_from_depth(
            depth,
            camera_relative_pose=camera_relative,
            focal_length=float(intrinsics["focal_length"]),
            horizontal_aperture=float(intrinsics["horizontal_aperture"]),
        )
        return {
            # A synchronized depth frame is valid even when no supported
            # obstacle cluster is visible. Conflating those states produced
            # the misleading 5020 log "frame unavailable error=None" and
            # disabled the observation guard exactly at close range.
            "ok": True,
            "source": source,
            "clearance_available": bool(clearance.get("ok")),
            "clearance": clearance,
            "floor": floor,
            "observation_contract": [
                "head_depth_linear",
                "head_cam_rel_pose",
                "trunk_proprioception",
                "static_r1pro_geometry",
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": f"forward observation failed: {exc}"}


def _yield_adjust_chassis_observation_only(
    ctx,
    *,
    forward: float,
    spin: float,
    vmax: float,
    wmax: float,
    timeout_s: float,
    translation: float = 0.0,
    log_prefix: str = "adjust_chassis",
):
    """Execute chassis motion using only official-evaluator observations.

    Forward motion fuses robot-frame ``base_qvel`` with physically bounded
    changes in head depth. Camera-relative pose and bounded action history provide a
    conservative forward stop; when depth cannot see a close obstacle, a
    proprioceptive stall triggers an adaptive reverse recovery. No global pose,
    contacts, segmentation, scene graph, object state, or simulator collision
    query is read here.
    """
    from behavior_interface.skills.base_forward_observation_guard import (
        floor_plane_delta,
    )
    world = ctx.world
    try:
        control_hz = float(
            getattr(world, "control_hz", _CHASSIS_OBS_CONTROL_HZ)
        )
    except (TypeError, ValueError):
        control_hz = _CHASSIS_OBS_CONTROL_HZ
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        control_hz = _CHASSIS_OBS_CONTROL_HZ
    dt = 1.0 / control_hz
    max_steps = max(1, int(math.ceil(max(0.1, float(timeout_s)) / dt)))
    used_steps = 0
    forward_target = abs(float(forward))
    forward_sign = 1.0 if float(forward) >= 0.0 else -1.0
    translation_target = abs(float(translation))
    translation_sign = 1.0 if float(translation) >= 0.0 else -1.0
    linear_target_xy = np.asarray(
        [float(forward), float(translation)], dtype=np.float64
    )
    linear_target_distance = float(np.linalg.norm(linear_target_xy))
    linear_target_unit = (
        linear_target_xy / linear_target_distance
        if linear_target_distance > 1e-9
        else np.zeros(2, dtype=np.float64)
    )
    linear_position_xy = np.zeros(2, dtype=np.float64)
    linear_last_cmd = np.zeros(2, dtype=np.float64)
    reverse_escape_request = bool(
        float(forward) < -1e-6 and translation_target <= 1e-6
    )
    recovery_direction_xy = (
        linear_target_unit.copy()
        if reverse_escape_request
        else -linear_target_unit
    )
    recovery_motion_sign = float(recovery_direction_xy[0])
    vmax = max(0.05, min(abs(float(vmax)), 0.50))
    wmax = max(0.10, min(abs(float(wmax)), 1.00))
    forward_progress = 0.0
    translation_progress = 0.0
    last_healthy_progress = 0.0
    forward_last_cmd = 0.0
    translation_last_cmd = 0.0
    forward_stall_steps = 0
    forward_motion_confirmed = False
    forward_stall_limit = max(3, int(math.ceil(_CHASSIS_OBS_STALL_S / dt)))
    sample_interval_steps = max(
        1, int(math.ceil(_CHASSIS_OBS_DEPTH_SAMPLE_S / dt))
    )
    next_sample_step = 0
    sample_progress = 0.0
    sample_commanded_distance = 0.0
    sampled_clearance = None
    depth_anchor_clearance = None
    depth_anchor_progress = 0.0
    depth_anchor_step = None
    depth_anchor_commanded_distance = 0.0
    depth_fused_progress = 0.0
    depth_motion_updates = 0
    forward_commanded_distance = 0.0
    floor_baseline = None
    floor_final = None
    floor_delta_report = None
    observation_samples: List[Dict[str, Any]] = []
    obstacle_limited = False
    obstacle_reason = None
    recovery_attempted = False
    recovery_ok = True
    retreat_progress = 0.0
    retreat_target = 0.0
    retreat_speed = _CHASSIS_OBS_RETREAT_INITIAL_MPS
    retreat_speed_max = 0.0
    retreat_escalations = 0
    post_stop_stable = False
    post_stop_stable_steps = 0
    recovery_trigger = None

    def _sync_linear_progress() -> float:
        nonlocal forward_progress, translation_progress
        forward_progress = (
            max(0.0, forward_sign * float(linear_position_xy[0]))
            if forward_target > 1e-9
            else 0.0
        )
        translation_progress = (
            max(0.0, translation_sign * float(linear_position_xy[1]))
            if translation_target > 1e-9
            else 0.0
        )
        return max(
            0.0,
            float(np.dot(linear_position_xy, linear_target_unit)),
        )

    def _action(
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
    ):
        return world.set_base_velocity(
            float(np.clip(
                float(vx) / _CHASSIS_OBS_BASE_MAX_LIN_MPS,
                -1.0,
                1.0,
            )),
            float(np.clip(
                float(vy) / _CHASSIS_OBS_BASE_MAX_LIN_MPS,
                -1.0,
                1.0,
            )),
            float(np.clip(
                float(wz) / _CHASSIS_OBS_BASE_MAX_ANG_RADPS,
                -1.0,
                1.0,
            )),
        )

    def _qvel() -> np.ndarray:
        try:
            value = np.asarray(world.base_qvel(), dtype=np.float64).reshape(3)
            if np.all(np.isfinite(value)):
                return value
        except Exception:
            pass
        return np.zeros(3, dtype=np.float64)

    def _sample_forward_observation() -> None:
        nonlocal sampled_clearance, sample_progress, floor_baseline
        nonlocal depth_anchor_clearance, depth_anchor_progress
        nonlocal depth_anchor_step, depth_fused_progress
        nonlocal depth_anchor_commanded_distance
        nonlocal depth_motion_updates, forward_progress
        nonlocal last_healthy_progress, forward_stall_steps
        nonlocal sample_commanded_distance
        nonlocal forward_motion_confirmed
        observation = _read_forward_observation(ctx, log_prefix=log_prefix)
        clearance = dict(observation.get("clearance") or {})
        floor = dict(observation.get("floor") or {})
        value = clearance.get("clearance_m")
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        depth_motion_accepted = False
        clearance_drop = None
        max_plausible_drop = None
        depth_progress_candidate = None
        if value is not None and math.isfinite(value):
            value = max(0.0, value)
            if (
                depth_anchor_clearance is not None
                and depth_anchor_step is not None
            ):
                clearance_drop = depth_anchor_clearance - value
                # A changing nearest cluster can make depth jump. Only fuse a
                # cumulative decrease that bounded action history could have
                # produced since the stable depth anchor. Using an
                # anchor instead of adjacent frames tolerates quantization
                # jitter without accepting an unbounded cluster switch.
                max_plausible_drop = max(
                    0.02,
                    (
                        forward_commanded_distance
                        - depth_anchor_commanded_distance
                    )
                    * _CHASSIS_OBS_ACTION_DISTANCE_FACTOR
                    + 0.01,
                )
                if 0.001 <= clearance_drop <= max_plausible_drop:
                    depth_progress_candidate = max(
                        0.0,
                        depth_anchor_progress + clearance_drop,
                    )
                    depth_fused_progress = max(
                        depth_fused_progress,
                        depth_progress_candidate,
                    )
                    if depth_progress_candidate > forward_progress + 1e-6:
                        forward_progress = depth_progress_candidate
                        linear_position_xy[0] = (
                            forward_sign * depth_progress_candidate
                        )
                        last_healthy_progress = max(
                            last_healthy_progress,
                            _sync_linear_progress(),
                        )
                        forward_stall_steps = 0
                        forward_motion_confirmed = True
                        depth_motion_updates += 1
                        depth_motion_accepted = True
            else:
                depth_anchor_clearance = value
                depth_anchor_progress = float(forward_progress)
                depth_anchor_step = int(used_steps)
                depth_anchor_commanded_distance = float(
                    forward_commanded_distance
                )
            sampled_clearance = value
            sample_progress = float(forward_progress)
            sample_commanded_distance = float(forward_commanded_distance)
        if floor_baseline is None and floor.get("ok"):
            floor_baseline = floor
        observation_samples.append({
            "ok": bool(observation.get("ok")),
            "source": observation.get("source"),
            "error": observation.get("error"),
            "clearance_available": value is not None,
            "clearance_m": value,
            "clearance_drop_m": clearance_drop,
            "max_plausible_clearance_drop_m": max_plausible_drop,
            "depth_progress_candidate_m": depth_progress_candidate,
            "depth_motion_accepted": depth_motion_accepted,
            "floor_ok": bool(floor.get("ok")),
        })

    def _settle_zero() -> bool:
        nonlocal used_steps, post_stop_stable, post_stop_stable_steps
        required = max(3, int(math.ceil(_CHASSIS_OBS_SETTLE_S / dt)))
        stable = 0
        budget = max(required, 2 * required)
        for _ in range(budget):
            if used_steps >= max_steps:
                break
            yield _action()
            used_steps += 1
            velocity = _qvel()
            is_stable = bool(
                math.hypot(float(velocity[0]), float(velocity[1]))
                <= _CHASSIS_OBS_SETTLE_SPEED_MPS
                and abs(float(velocity[2]))
                <= _CHASSIS_OBS_SETTLE_YAW_RATE_RADPS
            )
            stable = stable + 1 if is_stable else 0
            post_stop_stable_steps = max(post_stop_stable_steps, stable)
            if stable >= required:
                post_stop_stable = True
                return True
        post_stop_stable = False
        return False

    def _settle_with_floor_check() -> bool:
        nonlocal floor_final, floor_delta_report, post_stop_stable
        qvel_stable = bool((yield from _settle_zero()))
        if not qvel_stable or floor_baseline is None:
            return qvel_stable
        final_observation = _read_forward_observation(
            ctx, log_prefix=f"{log_prefix}.settle"
        )
        floor_final = dict(final_observation.get("floor") or {})
        floor_delta_report = floor_plane_delta(floor_final, floor_baseline)
        # An unavailable floor plane is not promoted to privileged truth; qvel
        # remains the fallback. When both allowed depth planes are valid, an
        # unstable delta must veto the landing decision.
        if floor_delta_report.get("ok") and not floor_delta_report.get(
            "stable", False
        ):
            post_stop_stable = False
            return False
        return True

    # Clear residual velocity before taking the baseline depth/floor sample.
    for _ in range(3):
        if used_steps >= max_steps:
            break
        yield _action()
        used_steps += 1

    if forward_target > 1e-6 and forward_sign > 0.0:
        _sample_forward_observation()
        next_sample_step = used_steps + sample_interval_steps

    while linear_target_distance > 1e-6 and used_steps < max_steps:
        velocity = _qvel()
        last_speed = float(np.linalg.norm(linear_last_cmd))
        if last_speed > 1e-6:
            linear_position_xy[:] = (
                linear_position_xy + velocity[:2] * dt
            )
            path_progress = _sync_linear_progress()
            observed_path_speed = float(
                np.dot(velocity[:2], linear_last_cmd / last_speed)
            )
            healthy_threshold = max(
                _CHASSIS_OBS_MIN_SPEED_MPS,
                _CHASSIS_OBS_STALL_RATIO * last_speed,
            )
            if observed_path_speed >= healthy_threshold:
                last_healthy_progress = path_progress
                forward_stall_steps = 0
                if forward_sign * float(velocity[0]) > 0.0:
                    forward_motion_confirmed = True
            else:
                forward_stall_steps += 1

        remaining_xy = linear_target_xy - linear_position_xy
        remaining = float(np.linalg.norm(remaining_xy))
        if remaining <= 0.004:
            break

        estimated_clearance = None
        if sampled_clearance is not None:
            observed_delta = max(0.0, forward_progress - sample_progress)
            commanded_delta_upper_bound = (
                max(
                    0.0,
                    forward_commanded_distance
                    - sample_commanded_distance,
                )
                * _CHASSIS_OBS_ACTION_DISTANCE_FACTOR
            )
            estimated_clearance = max(
                0.0,
                sampled_clearance
                - max(observed_delta, commanded_delta_upper_bound),
            )
            if (
                forward_sign > 0.0
                and estimated_clearance <= _CHASSIS_OBS_STOP_GAP_M
            ):
                obstacle_limited = True
                obstacle_reason = "rgbd_clearance_limit"
                break

        if forward_stall_steps >= forward_stall_limit:
            if forward_target > 1e-6:
                obstacle_limited = True
                obstacle_reason = "base_qvel_stall"
                recovery_attempted = True
            else:
                recovery_ok = False
                obstacle_reason = "translation_qvel_stall"
            break

        if (
            forward_target > 1e-6
            and forward_sign > 0.0
            and used_steps >= next_sample_step
        ):
            _sample_forward_observation()
            next_sample_step = used_steps + sample_interval_steps
            continue

        speed = min(vmax, max(0.05, math.sqrt(2.0 * 0.35 * remaining)))
        if remaining <= 0.08:
            speed = min(speed, 0.10)
        if remaining <= 0.03:
            speed = min(speed, 0.06)
        if estimated_clearance is not None and forward_sign > 0.0:
            margin = max(
                0.0, estimated_clearance - _CHASSIS_OBS_STOP_GAP_M
            )
            speed = min(speed, math.sqrt(max(0.0, 2.0 * 0.30 * margin)))
            if margin <= 0.08:
                speed = min(speed, 0.06)
        elif (
            forward_target > 1e-6
            and forward_sign > 0.0
            and not forward_motion_confirmed
        ):
            # With neither a supported depth cluster nor measured base motion,
            # only apply a low-energy probe. This avoids reloading an already
            # stable close boundary while still allowing qvel-confirmed open
            # space to release the cap on the next observation.
            speed = min(speed, _CHASSIS_OBS_BLIND_PROBE_MPS)
        if speed <= 0.01:
            obstacle_limited = True
            obstacle_reason = "rgbd_clearance_limit"
            break

        linear_last_cmd[:] = remaining_xy * (speed / remaining)
        forward_last_cmd = float(linear_last_cmd[0])
        translation_last_cmd = float(linear_last_cmd[1])
        ctx.set_status(
            "adjust chassis observation-only "
            f"vx={forward_last_cmd:+.2f} vy={translation_last_cmd:+.2f} "
            f"remaining={remaining:.3f}/{linear_target_distance:.3f}m"
        )
        yield _action(vx=forward_last_cmd, vy=translation_last_cmd)
        used_steps += 1
        forward_commanded_distance += max(0.0, forward_last_cmd) * dt

    linear_remaining_m = float(
        np.linalg.norm(linear_target_xy - linear_position_xy)
    )
    linear_target_reached = linear_remaining_m <= 0.004
    if linear_target_distance > 1e-6 and not obstacle_limited:
        if not linear_target_reached and recovery_ok:
            recovery_ok = False
            obstacle_reason = (
                "forward_timeout"
                if translation_target <= 1e-6
                else (
                    "translation_timeout"
                    if forward_target <= 1e-6
                    else "linear_timeout"
                )
            )

    # A depth stop normally needs no retreat. If zero velocity cannot settle,
    # however, the base is already dynamically loaded and must enter the same
    # adaptive unload path as a qvel stall. This was the second 5020 failure:
    # RGB-D stopped at the right limit, then returned without ever recovering.
    settled_before_recovery = False
    if obstacle_limited and recovery_ok:
        requested_recovery = bool(recovery_attempted)
        settled_before_recovery = bool(
            (yield from _settle_with_floor_check())
        )
        if not settled_before_recovery:
            recovery_attempted = True
            recovery_trigger = (
                "base_qvel_stall"
                if requested_recovery
                else "post_stop_unsettled_after_rgbd_limit"
            )
            # _settle_zero reports failure, but recovery is still possible.
            recovery_ok = True
        elif requested_recovery and reverse_escape_request:
            # Rearward commands are themselves the conservative escape motion.
            # A stable zero qvel can be static friction from a front load, not
            # proof of a rear boundary (there is no privileged rear contact).
            settled_before_recovery = False
            recovery_attempted = True
            recovery_trigger = "reverse_qvel_stall"
        else:
            # A stable qvel stall is the maximum reachable boundary, not a
            # reason to retreat. This keeps repeated large requests idempotent.
            recovery_attempted = False

    # Reverse far enough to unload the base. If the initial command cannot
    # move the compressed wheels, increase it in bounded stages. If one unload
    # distance is insufficient to settle, extend it incrementally instead of
    # returning in a suspended state.
    retreat_last_cmd = 0.0
    retreat_stall_steps = 0
    retreat_stall_limit = max(
        2, int(math.ceil(_CHASSIS_OBS_RETREAT_STALL_S / dt))
    )
    observed_reverse_once = False

    def _retreat_to_current_target():
        nonlocal used_steps
        nonlocal retreat_progress
        nonlocal retreat_speed, retreat_speed_max, retreat_escalations
        nonlocal retreat_last_cmd, retreat_stall_steps
        nonlocal observed_reverse_once
        retreat_last_cmd = 0.0
        retreat_stall_steps = 0
        while used_steps < max_steps and retreat_progress < retreat_target:
            velocity = _qvel()
            if retreat_last_cmd > 0.0:
                reverse_speed = max(
                    0.0,
                    float(np.dot(velocity[:2], recovery_direction_xy)),
                )
                retreat_progress += reverse_speed * dt
                linear_position_xy[:] = (
                    linear_position_xy + velocity[:2] * dt
                )
                _sync_linear_progress()
                expected = max(
                    _CHASSIS_OBS_MIN_SPEED_MPS,
                    _CHASSIS_OBS_STALL_RATIO * retreat_last_cmd,
                )
                if reverse_speed >= expected:
                    retreat_stall_steps = 0
                    observed_reverse_once = True
                else:
                    retreat_stall_steps += 1
                if (
                    retreat_stall_steps >= retreat_stall_limit
                    and retreat_speed
                    < _CHASSIS_OBS_RETREAT_MAX_MPS - 1e-9
                ):
                    old_speed = retreat_speed
                    retreat_speed = min(
                        _CHASSIS_OBS_RETREAT_MAX_MPS,
                        max(old_speed * 1.65, old_speed + 0.06),
                    )
                    retreat_escalations += 1
                    retreat_stall_steps = 0
                    ctx.log(
                        f"{log_prefix} [observation-only recovery] "
                        f"reverse qvel stalled; speed "
                        f"{old_speed:.2f}->{retreat_speed:.2f}m/s"
                    )
            if retreat_progress >= retreat_target:
                break
            retreat_speed_max = max(retreat_speed_max, retreat_speed)
            retreat_last_cmd = retreat_speed
            recovery_cmd = recovery_direction_xy * retreat_speed
            ctx.set_status(
                "adjust chassis recovery "
                f"vx={recovery_cmd[0]:+.2f} vy={recovery_cmd[1]:+.2f} "
                f"retreat={retreat_progress:.3f}/{retreat_target:.3f}m"
            )
            yield _action(vx=recovery_cmd[0], vy=recovery_cmd[1])
            used_steps += 1

    if recovery_attempted and recovery_ok:
        for _ in range(max(2, int(math.ceil(0.25 / dt)))):
            if used_steps >= max_steps:
                break
            yield _action()
            used_steps += 1
        path_progress = _sync_linear_progress()
        overlap = max(0.0, path_progress - last_healthy_progress)
        retreat_target = min(
            _CHASSIS_OBS_RETREAT_MAX_M,
            max(
                (
                    0.05
                    if recovery_trigger
                    == "post_stop_unsettled_after_rgbd_limit"
                    else _CHASSIS_OBS_RETREAT_MIN_M
                ),
                (
                    float(
                        np.linalg.norm(
                            linear_target_xy - linear_position_xy
                        )
                    )
                    if recovery_trigger == "reverse_qvel_stall"
                    else overlap + _CHASSIS_OBS_RETREAT_MARGIN_M
                ),
            ),
        )
        yield from _retreat_to_current_target()
        recovery_ok = bool(
            retreat_progress + 0.002 >= retreat_target
        )
        if (
            recovery_ok
            and recovery_trigger == "reverse_qvel_stall"
            and float(
                np.linalg.norm(linear_target_xy - linear_position_xy)
            ) <= 0.004
        ):
            obstacle_limited = False
            obstacle_reason = None

    if recovery_ok and not settled_before_recovery:
        recovery_ok = bool((yield from _settle_with_floor_check()))
        while (
            not recovery_ok
            and recovery_attempted
            and retreat_target < _CHASSIS_OBS_RETREAT_MAX_M - 1e-9
            and used_steps < max_steps
        ):
            old_target = retreat_target
            retreat_target = min(
                _CHASSIS_OBS_RETREAT_MAX_M,
                max(retreat_target + 0.04, retreat_progress + 0.025),
            )
            ctx.log(
                f"{log_prefix} [observation-only recovery] "
                f"zero-speed settle still unstable; extend retreat "
                f"{old_target:.3f}->{retreat_target:.3f}m"
            )
            recovery_ok = True
            yield from _retreat_to_current_target()
            recovery_ok = bool(
                retreat_progress + 0.002 >= retreat_target
            )
            if recovery_ok:
                recovery_ok = bool((yield from _settle_with_floor_check()))

    if not recovery_ok:
        # Even on failure, leave the last actions at zero rather than returning
        # while a recovery velocity remains latched.
        for _ in range(3):
            if used_steps >= max_steps:
                break
            yield _action()
            used_steps += 1

    if floor_baseline is not None:
        final_observation = _read_forward_observation(
            ctx, log_prefix=f"{log_prefix}.final"
        )
        floor_final = dict(final_observation.get("floor") or {})
        floor_delta_report = floor_plane_delta(floor_final, floor_baseline)

    _sync_linear_progress()
    linear_remaining_m = float(
        np.linalg.norm(linear_target_xy - linear_position_xy)
    )
    linear_target_reached = linear_remaining_m <= 0.004
    translation_ok = bool(
        recovery_ok
        and (
            translation_target <= 1e-6
            or linear_target_reached
            or obstacle_limited
        )
    )

    spin_target = abs(math.radians(float(spin)))
    spin_sign = 1.0 if float(spin) >= 0.0 else -1.0
    spin_progress = 0.0
    spin_last_cmd = 0.0
    spin_stall_steps = 0
    spin_ok = translation_ok
    while spin_ok and spin_target > 1e-6 and used_steps < max_steps:
        velocity = _qvel()
        if abs(spin_last_cmd) > 1e-6:
            observed_rate = spin_sign * float(velocity[2])
            spin_progress = max(0.0, spin_progress + observed_rate * dt)
            if observed_rate >= max(0.01, 0.20 * abs(spin_last_cmd)):
                spin_stall_steps = 0
            else:
                spin_stall_steps += 1
        remaining = max(0.0, spin_target - spin_progress)
        if remaining <= math.radians(0.7):
            break
        if spin_stall_steps >= forward_stall_limit:
            spin_ok = False
            obstacle_reason = obstacle_reason or "spin_qvel_stall"
            break
        rate = min(wmax, max(0.10, 1.8 * remaining))
        spin_last_cmd = spin_sign * rate
        yield _action(wz=spin_last_cmd)
        used_steps += 1
    if spin_ok and spin_target > 1e-6:
        spin_ok = spin_progress + math.radians(0.7) >= spin_target
        if spin_ok:
            spin_ok = bool((yield from _settle_with_floor_check()))

    linear_cross_track_m = (
        float(
            np.dot(
                linear_position_xy,
                np.asarray(
                    [-linear_target_unit[1], linear_target_unit[0]],
                    dtype=np.float64,
                ),
            )
        )
        if linear_target_distance > 1e-9
        else 0.0
    )
    ok = bool(recovery_ok and translation_ok and spin_ok)
    report = {
        "ok": ok,
        "forward": float(forward),
        "translation": float(translation),
        "spin": float(spin),
        "forward_ok": bool(recovery_ok),
        "forward_actual_m": round(float(linear_position_xy[0]), 4),
        "forward_obstacle_limited": bool(obstacle_limited),
        "translation_ok": bool(translation_ok),
        "translation_actual_m": round(float(linear_position_xy[1]), 4),
        "linear_target_reached": bool(linear_target_reached),
        "linear_remaining_m": round(float(linear_remaining_m), 4),
        "linear_cross_track_m": round(linear_cross_track_m, 4),
        "linear_control": "simultaneous_robot_frame_xy",
        "linear_speed_limit_mps": float(vmax),
        "spin_actual_deg": round(math.degrees(spin_progress) * spin_sign, 3),
        "observation_only": True,
        "observation_contract": [
            "head_depth_linear",
            "head_cam_rel_pose",
            "base_qvel_proprioception",
            "trunk_proprioception",
            "action_history",
            "static_r1pro_geometry",
        ],
        "forbidden_observation_reads": [],
        "ground_guard": {
            "observation_only": True,
            "requested_m": round(forward_target, 4),
            "safe_traveled_m": round(float(forward_progress), 4),
            "last_healthy_progress_m": round(
                float(last_healthy_progress), 4
            ),
            "depth_fused_progress_m": round(
                float(depth_fused_progress), 4
            ),
            "depth_motion_updates": int(depth_motion_updates),
            "forward_motion_confirmed": bool(forward_motion_confirmed),
            "forward_commanded_distance_m": round(
                float(forward_commanded_distance), 4
            ),
            "action_distance_upper_bound_m": round(
                float(
                    forward_commanded_distance
                    * _CHASSIS_OBS_ACTION_DISTANCE_FACTOR
                ),
                4,
            ),
            "obstacle_limited": bool(obstacle_limited),
            "obstacle_stop_reason": obstacle_reason,
            "recovery_attempted": bool(recovery_attempted),
            "recovery_trigger": recovery_trigger,
            "recovery_motion_sign": recovery_motion_sign,
            "recovery_motion_vector": recovery_direction_xy.astype(
                float
            ).tolist(),
            "collision_retreat_m": round(float(retreat_progress), 4),
            "collision_retreat_target_m": round(float(retreat_target), 4),
            "collision_retreat_speed_max_mps": round(
                float(retreat_speed_max), 3
            ),
            "collision_retreat_speed_escalations": int(
                retreat_escalations
            ),
            "collision_ground_settled": bool(post_stop_stable),
            "post_stop_verified": bool(post_stop_stable),
            "post_stop_stable_steps": int(post_stop_stable_steps),
            "landing_verification": (
                "base_qvel_dynamic_settle_plus_rgbd_floor_delta"
                if floor_delta_report is not None
                else "base_qvel_dynamic_settle"
            ),
            "floor_plane_delta": floor_delta_report,
            "observation_samples": observation_samples,
            "action_steps": int(used_steps),
            "safety_abort": not ok,
        },
    }
    if not ok:
        if not recovery_ok:
            if forward_target <= 1e-6 and translation_target > 1e-6:
                report["phase"] = "translation"
                report["error"] = "底盘横移未到位、速度停滞或停止后未稳定"
            else:
                report["phase"] = "linear"
                report["error"] = "底盘二维移动未到位或自适应退让后未稳定"
        elif not translation_ok:
            report["phase"] = "translation"
            report["error"] = "底盘横移未到位、速度停滞或停止后未稳定"
        else:
            report["phase"] = "spin"
            report["error"] = "底盘原地转向未到位或本体速度停滞"
    ctx.log(
        f"{log_prefix} observation-only done ok={ok} "
        f"requested=({float(forward):+.3f},{float(translation):+.3f})m "
        f"actual=({linear_position_xy[0]:+.3f},"
        f"{linear_position_xy[1]:+.3f})m "
        f"remaining={linear_remaining_m:.3f}m "
        f"obstacle_limited={obstacle_limited} reason={obstacle_reason} "
        f"retreat={retreat_progress:.3f}m speed_max={retreat_speed_max:.2f}m/s "
        f"settled={post_stop_stable}"
    )
    ctx.set_result(report)
    return


def _drive_body_forward(
    ctx,
    distance_m: float,
    *,
    translation_m: float = 0.0,
    vmax: float = 0.5,
    tol: float = 0.06,
    timeout_s: float = 120.0,
    log_prefix: str = "move_robot",
    nav_guard: bool = True,
    extra_inflate: float = 0.0,
    deadline_mono: float | None = None,
    lookahead_m: float = 0.28,
):
    """V2 grounded boundary refinement for a robot-frame XY displacement.

    Pure forward motion additionally uses the existing head RGB-D sampling for
    approach speed. Lateral and diagonal motion use the same physical
    lift/tilt/support guard without pretending that a forward camera clearance
    is a lateral clearance measurement.
    """
    requested_local_xy = np.asarray(
        [float(distance_m), float(translation_m)], dtype=np.float64
    )
    requested = float(np.linalg.norm(requested_local_xy))
    if requested < 1e-6:
        return True
    requested_unit_local = requested_local_xy / requested
    world = ctx.world
    is_runtime_world = bool(
        getattr(world, "robot", None) is not None
        or getattr(world, "_official_adapter", None) is not None
    )
    if not is_runtime_world:
        return (
            yield from _drive_body_forward_segment(
                ctx,
                distance_m,
                translation_m=translation_m,
                vmax=vmax,
                tol=tol,
                timeout_s=timeout_s,
                log_prefix=log_prefix,
                nav_guard=nav_guard,
                extra_inflate=extra_inflate,
                deadline_mono=deadline_mono,
                lookahead_m=lookahead_m,
            )
        )
    if (
        float(distance_m) < 0.0
        and abs(float(translation_m)) <= 1e-9
    ) or not nav_guard:
        return (
            yield from _drive_body_forward_segment(
                ctx,
                distance_m,
                translation_m=translation_m,
                vmax=vmax,
                tol=tol,
                timeout_s=timeout_s,
                log_prefix=log_prefix,
                nav_guard=nav_guard,
                extra_inflate=extra_inflate,
                deadline_mono=deadline_mono,
                lookahead_m=lookahead_m,
            )
        )

    from behavior_interface.skills.base_forward_observation_guard import (
        FORWARD_MIN_REFINE_M,
        approach_speed_mps,
    )

    wrapper_start = time.time()
    pose_anchor = world.robot_pose()
    x0 = float(pose_anchor.pos[0])
    y0 = float(pose_anchor.pos[1])
    yaw0 = float(pose_anchor.yaw)
    cos_y0, sin_y0 = math.cos(yaw0), math.sin(yaw0)
    heading0 = np.asarray(
        [
            cos_y0 * requested_unit_local[0]
            - sin_y0 * requested_unit_local[1],
            sin_y0 * requested_unit_local[0]
            + cos_y0 * requested_unit_local[1],
        ],
        dtype=np.float64,
    )
    use_forward_observation = bool(
        float(distance_m) > 0.0 and abs(float(translation_m)) <= 1e-9
    )

    def _wrapper_progress() -> float:
        pose_now = world.robot_pose()
        delta = np.asarray(
            [
                float(pose_now.pos[0]) - x0,
                float(pose_now.pos[1]) - y0,
            ],
            dtype=np.float64,
        )
        return float(np.dot(delta, heading0))

    def _wrapper_positive_progress() -> float:
        return max(0.0, _wrapper_progress())

    segment_reports: List[Dict[str, Any]] = []
    observation_samples: List[Dict[str, Any]] = []
    collision_refinements = 0
    obstacle_limited = False
    obstacle_reason = None
    last_report: Dict[str, Any] = {}
    last_stable_m = 0.0
    # verified_stable_m 只记「整段走完并通过零速验地」的位置，即真正能停住的点。
    # last_stable_m 来自运动中的贴地采样，那时还有动量，停在那儿往往会重新抬起，
    # 拿它当回补目标会一直留 1–2cm 空隙，且让收敛判据过早成立。
    verified_stable_m = 0.0
    first_unstable_m = None
    refine_phase = False
    boundary_stall = 0
    hysteresis_resets = 0

    for segment_index in range(_BODY_OBS_MAX_SEGMENTS):
        if _deadline_expired(deadline_mono) or (
            time.time() - wrapper_start > float(timeout_s)
        ):
            if segment_index > 0:
                # 已经完成过段（机器人此刻贴地），超时就如实交付已达成果，
                # 不要把逼近超时当成整个动作失败。
                obstacle_limited = True
                obstacle_reason = obstacle_reason or "boundary_deadline"
                break
            _set_nav_seg_fail(ctx, "deadline")
            return False
        # 缺口/回补必须用有符号里程：退让可能把机器人送到起点后方，
        # 若用截断到 0 的进度算缺口，回补量会少算，反复调用就会净后退。
        current_signed_m = _wrapper_progress()
        current_m = max(0.0, current_signed_m)
        remaining = max(0.0, requested - current_m)
        if remaining <= max(0.001, min(float(tol), requested * 0.05)):
            last_stable_m = max(last_stable_m, current_m)
            break

        # 收敛只看 [verified_stable, first_unstable] 区间宽度，不看当前位置：
        # 触发后的退让会让 current 远落后于稳定点，若拿 current 做判据，
        # 每次退让都会重新拉开差距，机器人只能在边界前反复来回。
        if first_unstable_m is not None:
            if first_unstable_m <= verified_stable_m:
                # 已验证能停住的位置反而被记成不稳：说明那次触发是保守/噪声，
                # 该上界作废，重新探索；靠 boundary_stall 判真正到顶，别在这儿收工，
                # 否则柔性接触边界会白留几厘米。重置次数有限，避免来回抖动。
                if hysteresis_resets < _BODY_BOUNDARY_HYSTERESIS_RESETS:
                    hysteresis_resets += 1
                    first_unstable_m = None
                    ctx.log(
                        f"{log_prefix} [最大可达点逼近] 上界与已验证稳定点矛盾，"
                        f"作废上界继续逼近 reset={hysteresis_resets} "
                        f"verified_stable={verified_stable_m:.4f}m"
                    )
                else:
                    obstacle_limited = True
                    obstacle_reason = obstacle_reason or "boundary_hysteresis"
                    break
            elif (
                first_unstable_m - verified_stable_m
            ) <= _BODY_BOUNDARY_REFINE_EPS_M:
                obstacle_limited = True
                obstacle_reason = obstacle_reason or "boundary_converged"
                break
        if boundary_stall >= _BODY_BOUNDARY_STALL_LIMIT:
            obstacle_limited = True
            obstacle_reason = obstacle_reason or "boundary_stalled"
            break

        resume_gap_m = max(0.0, verified_stable_m - current_signed_m)
        observation = None
        clearance_m = None
        # 只有开阔中途段允许提前结束验地；probe/resume/近障/末段都保持完整窗口。
        segment_verify_early_exit = False
        if resume_gap_m > _BODY_BOUNDARY_REFINE_EPS_M:
            # 退让留下的缺口：这段刚刚站稳过，属已知安全区，直接快速回补。
            # 跳过 RGB-D 与长验地，把「退 2cm 要蠕行 4 步」压成一步。
            segment_mode = "resume"
            # 回补是把已丢失的里程走回来，不受 remaining（按正向进度算）限制。
            segment_target_m = resume_gap_m
            segment_vmax = _BODY_BOUNDARY_RESUME_SPEED_MPS
            segment_verify_s = _BODY_BOUNDARY_RESUME_VERIFY_S
            segment_retreat_max_m = _BODY_BOUNDARY_PROBE_RETREAT_MAX_M
            ctx.log(
                f"{log_prefix} [最大可达点逼近] 快速回补已验证稳定点 "
                f"gap={segment_target_m:.4f}m progress={current_m:.4f}m "
                f"verified_stable={verified_stable_m:.4f}m"
            )
        elif refine_phase:
            # 未知区二分：步长由 (verified_stable, first_unstable) 区间决定。
            # 此阶段深度净空已无信息增益（读数只在噪声内抖动），故不再读 RGB-D。
            gap_m = (
                max(0.0, float(first_unstable_m) - verified_stable_m)
                if first_unstable_m is not None
                else _BODY_BOUNDARY_CREEP_STEP_M
            )
            segment_mode = "probe"
            segment_target_m = min(
                remaining,
                _BODY_BOUNDARY_CREEP_STEP_M,
                max(_BODY_BOUNDARY_REFINE_EPS_M, 0.5 * gap_m),
            )
            segment_vmax = _BODY_BOUNDARY_CREEP_SPEED_MPS
            segment_verify_s = _BODY_BOUNDARY_PROBE_VERIFY_S
            segment_retreat_max_m = _BODY_BOUNDARY_PROBE_RETREAT_MAX_M
            ctx.log(
                f"{log_prefix} [最大可达点逼近] 二分试探 "
                f"step={segment_target_m:.4f}m gap={gap_m:.4f}m "
                f"verified_stable={verified_stable_m:.4f}m "
                f"first_unstable="
                f"{'-' if first_unstable_m is None else round(float(first_unstable_m), 4)}"
            )
        else:
            segment_mode = "observe"
            segment_verify_s = None
            segment_retreat_max_m = None
            if use_forward_observation:
                observation = _read_forward_observation(
                    ctx, log_prefix=log_prefix
                )
                observation_samples.append({
                    "ok": bool(observation.get("ok")),
                    "source": observation.get("source"),
                    "error": observation.get("error"),
                    "clearance": observation.get("clearance"),
                    "floor": observation.get("floor"),
                })
            else:
                observation = None
                safe_advance_m = None
                segment_mode = "physical_vector"
                if segment_index == 0:
                    ctx.log(
                        f"{log_prefix} [二维实体守护] local="
                        f"({distance_m:+.3f},{translation_m:+.3f})m，"
                        "前向深度不作为横向净空，使用位姿进度与贴地边界搜索"
                    )
            if (
                observation is not None
                and observation.get("ok")
                and observation.get("clearance", {}).get("clearance_m")
                is not None
            ):
                clearance_m = float(observation["clearance"]["clearance_m"])
                safe_advance_m = max(0.0, clearance_m - _BODY_OBS_STOP_GAP_M)
                ctx.log(
                    f"{log_prefix} [RGB-D前向守护] sample={segment_index + 1} "
                    f"clearance={clearance_m:.4f}m "
                    f"safe_advance={safe_advance_m:.4f}m "
                    f"progress={current_m:.4f}m remaining={remaining:.3f}m"
                )
            elif observation is not None:
                safe_advance_m = None
                # 这里有两种完全不同的成因，混在一条日志里会把排查带偏：
                # 帧真的读不到，和帧正常但贴得太近、障碍已落到视野下沿之外。
                if observation.get("ok"):
                    detail = "帧正常但未检出前方障碍簇（多为过近超出视野下沿）"
                else:
                    detail = f"帧不可用 error={observation.get('error')}"
                ctx.log(
                    f"{log_prefix} [RGB-D前向守护] {detail}，"
                    f"改用物理边界搜索 progress={current_m:.4f}m"
                )

            if safe_advance_m is not None:
                # RGB-D 只限速/限步长，不单独硬停；过近时转入物理二分试探。
                if safe_advance_m <= FORWARD_MIN_REFINE_M:
                    refine_phase = True
                    if first_unstable_m is None:
                        first_unstable_m = current_m + max(
                            0.02, float(clearance_m or 0.02)
                        )
                    segment_mode = "probe"
                    segment_target_m = min(
                        remaining, _BODY_BOUNDARY_CREEP_STEP_M
                    )
                    segment_vmax = _BODY_BOUNDARY_CREEP_SPEED_MPS
                    segment_verify_s = _BODY_BOUNDARY_PROBE_VERIFY_S
                    segment_retreat_max_m = (
                        _BODY_BOUNDARY_PROBE_RETREAT_MAX_M
                    )
                else:
                    segment_target_m = min(
                        remaining,
                        _BODY_OBS_OPEN_LOOP_CHUNK_M,
                        max(safe_advance_m, _BODY_BOUNDARY_CREEP_STEP_M),
                    )
                    if safe_advance_m <= _BODY_NEAR_VERIFY_CLEARANCE_M:
                        # 深度净空在贴近踢脚/台面基座时会明显高估（看不到的斜面），
                        # 近距限制单步长度，让首次触发更轻，退让量随之更小。
                        segment_target_m = min(segment_target_m, 0.03)
                        segment_verify_s = _BODY_NEAR_VERIFY_S
                        segment_retreat_max_m = (
                            _BODY_BOUNDARY_PROBE_RETREAT_MAX_M
                        )
                    elif safe_advance_m >= _BODY_OPEN_VERIFY_CLEARANCE_M:
                        segment_verify_s = _BODY_OPEN_VERIFY_S
                        # 末段走完就退出 skill，之后是无人监管的自由物理步进，
                        # 必须看满窗口；中途段后面还有守护接手，确认即走。
                        segment_verify_early_exit = bool(
                            segment_target_m < remaining - _BODY_OBS_STOP_GAP_M
                        )
                    segment_vmax = approach_speed_mps(
                        requested_vmax_mps=float(vmax),
                        remaining_request_m=segment_target_m,
                        clearance_m=clearance_m,
                        stop_gap_m=_BODY_OBS_STOP_GAP_M,
                    )
            else:
                # 没有净空读数时是盲走，不能享受制动律带来的提速：这一段本来就
                # 指望「撞到再退」找边界，撞击速度必须保持在原有量级。
                segment_target_m = min(remaining, 0.22)
                segment_vmax = approach_speed_mps(
                    requested_vmax_mps=min(float(vmax), 0.20),
                    remaining_request_m=segment_target_m,
                    clearance_m=None,
                    stop_gap_m=_BODY_OBS_STOP_GAP_M,
                )

        if segment_target_m <= 1e-4 or segment_vmax <= 1e-6:
            obstacle_limited = True
            obstacle_reason = obstacle_reason or "boundary_step_too_small"
            break

        segment_tol = min(
            0.006,
            max(0.001, 0.15 * float(segment_target_m)),
        )
        progress_before = current_signed_m
        verified_before = verified_stable_m
        ok = yield from _drive_body_forward_segment(
            ctx,
            float(requested_unit_local[0]) * segment_target_m,
            translation_m=(
                float(requested_unit_local[1]) * segment_target_m
            ),
            world_direction_xy=heading0,
            vmax=segment_vmax,
            tol=segment_tol,
            timeout_s=max(
                1.0, float(timeout_s) - (time.time() - wrapper_start)
            ),
            log_prefix=f"{log_prefix}[segment {segment_index + 1}]",
            nav_guard=True,
            extra_inflate=extra_inflate,
            deadline_mono=deadline_mono,
            lookahead_m=lookahead_m,
            post_stop_verify_s=segment_verify_s,
            post_stop_early_exit=segment_verify_early_exit,
            retreat_max_m=segment_retreat_max_m,
        )
        last_report = dict(
            getattr(ctx, "_body_ground_guard_report", None) or {}
        )
        segment_reports.append(last_report)
        if not ok:
            return False

        current_signed_m = _wrapper_progress()
        current_m = max(0.0, current_signed_m)
        seg_last_stable = last_report.get("last_stable_progress_m")
        if seg_last_stable is not None:
            last_stable_m = max(
                last_stable_m,
                progress_before + max(0.0, float(seg_last_stable)),
            )
        last_stable_m = max(last_stable_m, current_m)
        # 段一旦正常返回，机器人此刻必然已通过贴地验证（走完验地，或触发后退让落地），
        # 所以当前位置就是「能停住」的硬证据。触发过守护不代表这里站不住。
        verified_stable_m = max(verified_stable_m, current_m)

        if last_report.get("obstacle_limited"):
            collision_refinements += 1
            obstacle_limited = True
            obstacle_reason = str(
                last_report.get("obstacle_stop_reason")
                or last_report.get("collision_guard_reason")
                or "physical_guard"
            )
            retreat_m = float(last_report.get("collision_retreat_m", 0.0) or 0.0)
            seg_first_unstable = last_report.get("first_unstable_progress_m")
            if seg_first_unstable is not None:
                trigger_m = progress_before + max(
                    0.0, float(seg_first_unstable)
                )
            else:
                trigger_m = current_signed_m + max(0.0, retreat_m)
            first_unstable_m = (
                trigger_m
                if first_unstable_m is None
                else min(first_unstable_m, trigger_m)
            )
            refine_phase = True
            ctx.log(
                f"{log_prefix} [最大可达点逼近] "
                f"refinement={collision_refinements} mode={segment_mode} "
                f"current={current_m:.4f}m "
                f"verified_stable={verified_stable_m:.4f}m "
                f"last_stable={last_stable_m:.4f}m "
                f"first_unstable={first_unstable_m:.4f}m "
                f"retreat={retreat_m:.4f}m"
            )
            if collision_refinements >= _BODY_BOUNDARY_MAX_REFINES:
                break

        if segment_mode == "probe":
            # 试探没能把「能停住的位置」往前推，说明已经贴在边界上，别再重复撞。
            if verified_stable_m - verified_before < _BODY_BOUNDARY_MIN_GAIN_M:
                boundary_stall += 1
            else:
                boundary_stall = 0

        if last_report.get("obstacle_limited"):
            continue

        gained = current_signed_m - progress_before
        # 段控制器按 tol 提前完成时，wrapper 进度常刚好差一个浮点误差；
        # 多留 2mm，避免把成功段误判成 stuck。
        if (
            segment_mode == "observe"
            and gained + segment_tol + 0.002 < segment_target_m
        ):
            _set_nav_seg_fail(ctx, "stuck")
            return False

    final_signed = _wrapper_progress()
    final_progress = max(0.0, final_signed)
    last_stable_m = max(last_stable_m, final_progress)
    verified_stable_m = max(verified_stable_m, final_progress)
    # 若仍明显落后于已验证能停住的位置，再给一次回补把空隙吃掉。
    if (
        obstacle_limited
        and verified_stable_m > final_signed + _BODY_BOUNDARY_REFINE_EPS_M
        and collision_refinements < _BODY_BOUNDARY_MAX_REFINES + 2
    ):
        # 只回补到已验证能停住的位置，不再越界试探：回补目标若取运动中的贴地点，
        # 这一步自己就会再撞一次，白花时间还留下空隙。
        # 缺口按有符号里程算：退到起点后方时也要把这段负位移一起补回来。
        gap_back = min(
            _BODY_COLLISION_RETREAT_MAX_M,
            verified_stable_m - final_signed,
        )
        ctx.log(
            f"{log_prefix} [最大可达点逼近] 退让留缝，回补 "
            f"gap={gap_back:.4f}m toward verified_stable={verified_stable_m:.4f}m"
        )
        ok = yield from _drive_body_forward_segment(
            ctx,
            float(requested_unit_local[0]) * gap_back,
            translation_m=float(requested_unit_local[1]) * gap_back,
            world_direction_xy=heading0,
            vmax=_BODY_BOUNDARY_RESUME_SPEED_MPS,
            tol=0.002,
            timeout_s=max(
                1.0, float(timeout_s) - (time.time() - wrapper_start)
            ),
            log_prefix=f"{log_prefix}[gap_recover]",
            nav_guard=True,
            extra_inflate=extra_inflate,
            deadline_mono=deadline_mono,
            lookahead_m=lookahead_m,
            post_stop_verify_s=_BODY_BOUNDARY_RESUME_VERIFY_S,
            retreat_max_m=_BODY_BOUNDARY_PROBE_RETREAT_MAX_M,
        )
        last_report = dict(
            getattr(ctx, "_body_ground_guard_report", None) or {}
        )
        segment_reports.append(last_report)
        if not ok:
            return False
        final_signed = _wrapper_progress()
        final_progress = max(0.0, final_signed)
        if last_report.get("obstacle_limited"):
            obstacle_reason = str(
                last_report.get("obstacle_stop_reason")
                or obstacle_reason
                or "gap_recover_guard"
            )

    final_report = dict(last_report)
    final_report.update({
        "requested_m": round(requested, 4),
        "requested_forward_m": round(float(distance_m), 4),
        "requested_translation_m": round(float(translation_m), 4),
        "safe_traveled_m": round(final_progress, 4),
        "safe_forward_m": round(
            float(final_progress * requested_unit_local[0]), 4
        ),
        "safe_translation_m": round(
            float(final_progress * requested_unit_local[1]), 4
        ),
        "linear_control": "simultaneous_robot_frame_xy",
        "last_stable_progress_m": round(verified_stable_m, 4),
        "motion_last_stable_progress_m": round(last_stable_m, 4),
        "first_unstable_progress_m": (
            None
            if first_unstable_m is None
            else round(float(first_unstable_m), 4)
        ),
        "obstacle_limited": bool(obstacle_limited),
        "obstacle_stop_reason": obstacle_reason,
        "observation_guard": bool(use_forward_observation),
        "physical_ground_guard": True,
        "verification": "simulator_pose_and_ground_support_v2_non_test",
        "official_evaluator_compatible": False,
        "observation_samples": observation_samples,
        "segments": segment_reports,
        "collision_refinements": int(collision_refinements),
        "post_stop_verified": True,
        "boundary_search": True,
    })
    setattr(ctx, "_body_ground_guard_report", final_report)
    if obstacle_limited:
        _set_nav_seg_fail(ctx, "obstacle_limited")
        ctx.log(
            f"{log_prefix} body MAX_REACHABLE_DISTANCE "
            f"requested={requested:.3f}m reached={final_progress:.4f}m "
            f"verified_stable={verified_stable_m:.4f}m "
            f"motion_last_stable={last_stable_m:.4f}m "
            f"reason={obstacle_reason} final_grounded=True"
        )
    else:
        _set_nav_seg_fail(ctx, "ok")
    return True


def _rotate_to_target_yaw(ctx, target_yaw_rad: float,
                          yaw_tol_deg: float, wmax: float, k_ang: float,
                          timeout_s: float, log_prefix: str,
                          deadline_mono: float | None = None):
    world = ctx.world
    trunk_hold_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    if hasattr(world, "set_trunk_pin_qpos"):
        world.set_trunk_pin_qpos(trunk_hold_q.tolist())
    t_start = time.time()
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    try:
        world._codex_fast_motion_no_obs = True
    except Exception:
        pass
    try:
        while True:
            if _deadline_expired(deadline_mono):
                ctx.log(f"{log_prefix} rotate DEADLINE reached")
                yield world.set_base_velocity(0.0, 0.0, 0.0)
                return False
            if time.time() - t_start > timeout_s:
                ctx.log(f"{log_prefix} rotate TIMEOUT after {timeout_s:.1f}s")
                yield world.set_base_velocity(0.0, 0.0, 0.0)
                return False
            pose = world.robot_pose()
            yaw = float(pose.yaw)
            err = _norm_angle(target_yaw_rad - yaw)
            if abs(err) < math.radians(yaw_tol_deg):
                ctx.log(f"{log_prefix} rotate DONE yaw={math.degrees(yaw):+.1f}° err={math.degrees(err):+.1f}°")
                for _ in range(3):
                    yield world.set_base_velocity(0.0, 0.0, 0.0)
                return True
            wz = max(-wmax, min(wmax, k_ang * err))
            ctx.set_status(f"rotate err={math.degrees(err):+.1f}°")
            yield world.set_base_velocity(0.0, 0.0, wz)
            if hasattr(world, "set_trunk_pin_qpos"):
                world.set_trunk_pin_qpos(trunk_hold_q.tolist())
    finally:
        try:
            world._codex_fast_motion_no_obs = old_no_obs
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Scene Graph 避障：A* 或直连
# ─────────────────────────────────────────────────────────────────────────────

def _plan_or_direct(ctx, sx: float, sy: float, gx: float, gy: float,
                    avoid_obstacles: bool,
                    extra_inflate: float,
                    log_prefix: str,
                    snap_goal: bool = True) -> Tuple[List[Tuple[float, float]], str]:
    if not avoid_obstacles:
        return [(sx, sy), (gx, gy)], "no_check"
    sg = getattr(ctx.world, "current_scene_graph", None)
    if sg is None:
        ctx.log(f"{log_prefix} 无 Scene Graph（首次构建未完成？）退化为直连")
        return [(sx, sy), (gx, gy)], "no_sg"
    from behavior_interface.scene_graph import plan_path, is_point_free
    ok_start, sblk = is_point_free(sg.free_region, sx, sy, extra_inflate=extra_inflate)
    ok_goal, gblk = is_point_free(sg.free_region, gx, gy, extra_inflate=extra_inflate)
    if not ok_start:
        ctx.log(f"{log_prefix} 起点 ({sx:.2f},{sy:.2f}) 被障碍 [{sblk}] 覆盖；会尝试在邻近找 free cell")
    if not ok_goal:
        if snap_goal:
            ctx.log(
                f"{log_prefix} 目标 ({gx:.2f},{gy:.2f}) 被障碍 [{gblk}] 挡住，"
                "将尝试规划至邻近 free cell"
            )
        else:
            ctx.log(
                f"{log_prefix} 目标 ({gx:.2f},{gy:.2f}) 被障碍 [{gblk}] 挡住，"
                "固定目标不重规划，闭环驱动直至超时"
            )
    wps, status = plan_path(
        sg.free_region, (sx, sy), (gx, gy),
        resolution=0.15, extra_inflate=extra_inflate, snap_goal=snap_goal,
    )
    if len(wps) >= 2:
        path_txt = " → ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in wps)
        ctx.log(
            f"{log_prefix} A* {status} inflate={extra_inflate:.2f} "
            f"({len(wps)} pts): {path_txt}"
        )
    return wps, status


# ─────────────────────────────────────────────────────────────────────────────
# Trunk 协同 IK：q1=-q2 升降 + q3 控 pitch（避免重心偏移摔倒）
# ─────────────────────────────────────────────────────────────────────────────
#
# R1Pro trunk URDF：
#   t1 +Y origin (-0.079, 0, 0.343) [-1.1345, 1.8326] rad
#   t2 +Y origin ( 0, 0, 0.400)     [-2.79, 2.53] rad
#   t3 -Y origin ( 0, 0, 0.300)     [-1.83, 1.57] rad
#   t4 +Z origin ( 0, 0, 0.100)     [-3.05, 3.05] rad
#
# 几何（base frame，所有 q=0 时 chest 站直朝上）：
#   chest_x = -0.079 - 0.4·sin(q1) - 0.3·sin(q1+q2) - 0.1·sin(q1+q2-q3)
#   chest_z =  0.343 + 0.4·cos(q1) + 0.3·cos(q1+q2) + 0.1·cos(q1+q2-q3)
#   chest_pitch = q1 + q2 - q3     （正 = 抬头）
#   chest_yaw   = q4
#
# 协同约束：q2 = -q1（让 q1+q2=0），这样 chest_x 偏移最小，不会摔倒。
# 此时：
#   chest_z(base) = 0.443 + 0.4·cos(q1) + 0.1·cos(-q3) = 0.443 + 0.4·cos(q1) + 0.1·cos(q3)
#   chest_pitch = -q3
#
# 解析 IK：
#   1) q3* = -pitch_target
#   2) q1* = ±acos((z_target_base - 0.443 - 0.1·cos(q3*)) / 0.4)
#      取符号同当前 q1（连续性）；q1=0 是奇异点，从 q1=0 启动时取 +0.1 rad 起步
#   3) q2* = -q1*
#   4) q4* = chest yaw 目标（由 base yaw 控，trunk 不用）
#
# chest world z = base_link.z(world) + chest_z(base frame)
# base_link.z 实测约 0.052 m（轮子 + base）

_TRUNK_T1_OFFSET_Z = 0.343
_TRUNK_T2_LEN      = 0.400
_TRUNK_T3_LEN      = 0.300
_TRUNK_T4_LEN      = 0.100
_TRUNK_BASE_OFFSET = _TRUNK_T1_OFFSET_Z  # 0.343

# q1 关节限位（r1pro.urdf torso_joint1，与 eval_utils.JOINT_RANGE 一致）
_Q1_MIN = -1.1345
_Q1_MAX = +1.8326
# torso_joint3（离胸廓最近）；与 r1pro.urdf / eval_utils.JOINT_RANGE 一致
_Q3_MIN = -1.8326
_Q3_MAX = +1.5708
# 快速模式抽稀路点时允许的单步腰关节位移上限；与直驱闭环的 trunk_max_step_rad 同量级
_TRUNK_FAST_MAX_STEP_RAD = 0.08

# 纯 q3 俯仰允许的胸口 θz 区间（与 move_to_object 俯身上限一致）：
# 60°≈仰头约 30°，165°≈大幅俯身。
_Q3_PITCH_THETA_Z_MIN_DEG = 60.0
_Q3_PITCH_THETA_Z_MAX_DEG = 165.0

# 俯仰末段减速：|err|≥5° 全步长，近目标线性缩至 ≥0.35°/步
_PITCH_DECEL_ERR_DEG = 5.0
_PITCH_STEP_MIN_RAD = math.radians(1.5)
_PITCH_SETTLE_FRAMES = 1  # move_to_object 普通模式需要少帧快速收敛


def _pitch_step_lim_rad(err_theta_z_deg: float, pitch_lim: float) -> float:
    """按剩余 θz 误差缩放俯仰步长，近目标减速。"""
    err_abs = abs(float(err_theta_z_deg))
    if err_abs >= _PITCH_DECEL_ERR_DEG:
        return float(pitch_lim)
    frac = max(err_abs / _PITCH_DECEL_ERR_DEG, 0.12)
    return max(_PITCH_STEP_MIN_RAD, float(pitch_lim) * frac)


def _chest_z_base_from_q(q1: float, q3: float) -> float:
    """假设 q2 = -q1（协同约束），算 chest z 在 base frame 的高度。"""
    return (_TRUNK_BASE_OFFSET
            + _TRUNK_T2_LEN * math.cos(q1)
            + _TRUNK_T3_LEN                 # cos(q1+q2)=cos(0)=1
            + _TRUNK_T4_LEN * math.cos(q3)) # q1+q2-q3 = -q3, cos 对称


_CHEST_Z_MIN = 0.66
_CHEST_Z_MAX = 1.15
_THETA_Z_MIN_DEG = 70.0
_THETA_Z_MAX_DEG = 118.0


def _base_link_z_world(world) -> float:
    try:
        return float(world.robot.get_position_orientation()[0][2])
    except Exception:
        return 0.05


def _solve_q1_q2_for_z(
    target_chest_z_base: float,
    q3_fixed: float,
    prev_q1_sign: float,
) -> Optional[Tuple[float, float]]:
    """固定 q3，仅反解 q1/q2 协同升降。不可达返回 None。"""
    rhs = (
        target_chest_z_base - 0.643 - _TRUNK_T4_LEN * math.cos(q3_fixed)
    ) / _TRUNK_T2_LEN
    if rhs < -1.0 - 1e-8 or rhs > 1.0 + 1e-8:
        return None
    rhs = max(-1.0, min(1.0, rhs))
    q1_abs = min(math.acos(rhs), _Q1_MAX)
    sign = 1.0 if prev_q1_sign >= 0 else -1.0
    q1 = sign * q1_abs
    return q1, -q1


def _feasible_trunk_z_only(
    chest: dict,
    curr_q: np.ndarray,
    target_z_world: float,
    base_z: float,
    *,
    z_tol: float = 0.012,
) -> Tuple[bool, str]:
    """仅升降：俯仰(θz)保持不变。"""
    if target_z_world < _CHEST_Z_MIN - 1e-3 or target_z_world > _CHEST_Z_MAX + 1e-3:
        return False, (
            f"胸口 z={target_z_world:.3f}m 超出范围 "
            f"[{_CHEST_Z_MIN:.2f},{_CHEST_Z_MAX:.2f}]m"
        )
    prev_sign = 1.0 if float(curr_q[0]) >= 0 else -1.0
    sol = _solve_q1_q2_for_z(target_z_world - base_z, float(curr_q[2]), prev_sign)
    if sol is None:
        return False, (
            f"upward={target_z_world - chest['z']:+.3f}m 在保持当前俯仰 θz="
            f"{chest['theta_z_deg']:.1f}° 时不可达（不能用俯仰补偿升降）"
        )
    q1, _ = sol
    z_ach = base_z + _chest_z_base_from_q(q1, float(curr_q[2]))
    if abs(z_ach - target_z_world) > z_tol:
        return False, f"胸口 z 目标 {target_z_world:.3f}m 不可达"
    return True, ""


def _feasible_trunk_pitch_only(
    chest: dict,
    curr_q: np.ndarray,
    target_theta_z_deg: float,
    base_z: float,
    *,
    z_tol: float = 0.012,
) -> Tuple[bool, str]:
    """仅俯仰：胸口 z 保持不变。"""
    if target_theta_z_deg < _THETA_Z_MIN_DEG - 1e-3 or target_theta_z_deg > _THETA_Z_MAX_DEG + 1e-3:
        return False, (
            f"pitch 目标 θz={target_theta_z_deg:.1f}° 超出范围 "
            f"[{_THETA_Z_MIN_DEG:.0f},{_THETA_Z_MAX_DEG:.0f}]°"
        )
    target_pitch_rad = math.radians(90.0 - target_theta_z_deg)
    if target_pitch_rad < _Q3_MIN - 1e-6 or target_pitch_rad > _Q3_MAX + 1e-6:
        return False, f"俯仰目标超出 trunk q3 关节限位"
    q3_t = max(_Q3_MIN, min(_Q3_MAX, target_pitch_rad))
    z_would = base_z + _chest_z_base_from_q(float(curr_q[0]), q3_t)
    if abs(z_would - chest["z"]) > z_tol:
        return False, (
            f"pitch 在保持 z={chest['z']:.3f}m 时不可达（q3 会改变胸口高度 "
            f"Δz≈{abs(z_would - chest['z']) * 100:.1f}cm，不能用 upward 补偿）"
        )
    return True, ""


def _feasible_trunk_q3_delta(
    curr_q: np.ndarray,
    pitch_deg: float,
    *,
    curr_theta_z_deg: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """纯 q3 俯仰：pitch>0 仰头 → q3 增大。

    约束：
    1) q3 关节限位；
    2) 目标姿态胸口 θz ∈ [60°, 165°]（只动 q3，用 FK 预测）。
    不再用 |pitch|≤45 这种与姿态无关的硬卡。
    """
    from behavior_interface.trunk_vertical_lift import (
        fk_torso_link4_forward,
        fk_torso_link4_theta_z_deg,
    )

    q1 = float(curr_q[0])
    q2 = float(curr_q[1])
    q3_curr = float(curr_q[2])
    q3_tgt = q3_curr + math.radians(float(pitch_deg))
    if q3_tgt < _Q3_MIN - 1e-6:
        return False, (
            f"pitch={float(pitch_deg):+.1f}° 使 q3={math.degrees(q3_tgt):.1f}° "
            f"低于限位 {math.degrees(_Q3_MIN):.1f}°"
        ), q3_tgt
    if q3_tgt > _Q3_MAX + 1e-6:
        return False, (
            f"pitch={float(pitch_deg):+.1f}° 使 q3={math.degrees(q3_tgt):.1f}° "
            f"高于限位 {math.degrees(_Q3_MAX):.1f}°"
        ), q3_tgt

    fwd = fk_torso_link4_forward(q1, q2, float(q3_tgt))
    # fx<0 表示过折（真实俯角>180°），acos 读数是镜像假象，一律拒绝
    if float(fwd[0]) < 0.0:
        tz_now = (
            float(curr_theta_z_deg)
            if curr_theta_z_deg is not None
            else float(fk_torso_link4_theta_z_deg(q1, q2, q3_curr))
        )
        return False, (
            f"pitch={float(pitch_deg):+.1f}° 会使躯干过折（胸口朝后下），"
            f"当前 θz≈{tz_now:.1f}°，允许区间 "
            f"[{_Q3_PITCH_THETA_Z_MIN_DEG:.0f},{_Q3_PITCH_THETA_Z_MAX_DEG:.0f}]°"
        ), q3_tgt

    tz_tgt = float(fk_torso_link4_theta_z_deg(q1, q2, float(q3_tgt)))
    if tz_tgt < _Q3_PITCH_THETA_Z_MIN_DEG - 1e-3:
        tz_now = (
            float(curr_theta_z_deg)
            if curr_theta_z_deg is not None
            else float(fk_torso_link4_theta_z_deg(q1, q2, q3_curr))
        )
        return False, (
            f"pitch={float(pitch_deg):+.1f}° 使 θz={tz_tgt:.1f}° "
            f"低于仰头下限 {_Q3_PITCH_THETA_Z_MIN_DEG:.0f}°"
            f"（当前 θz={tz_now:.1f}°）"
        ), q3_tgt
    if tz_tgt > _Q3_PITCH_THETA_Z_MAX_DEG + 1e-3:
        tz_now = (
            float(curr_theta_z_deg)
            if curr_theta_z_deg is not None
            else float(fk_torso_link4_theta_z_deg(q1, q2, q3_curr))
        )
        return False, (
            f"pitch={float(pitch_deg):+.1f}° 使 θz={tz_tgt:.1f}° "
            f"超过俯身上限 {_Q3_PITCH_THETA_Z_MAX_DEG:.0f}°"
            f"（当前 θz={tz_now:.1f}°）"
        ), q3_tgt
    return True, "", q3_tgt


def _yield_trunk_q3_pitch(
    ctx,
    world,
    pitch_deg: float,
    *,
    max_step_rad: float = 0.10,
    timeout_s: float = 30.0,
    q_tol_rad: float = 0.02,
    n_hold: int = 3,
    log_prefix: str = "move_robot/pitch",
) -> dict:
    """锁定 q1/q2/q4，仅驱动 q3 变化 pitch_deg（度）。"""
    q_act = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    q_locked = np.array([q_act[0], q_act[1], q_act[3]], dtype=np.float64)
    q3_tgt = float(np.clip(
        q_act[2] + math.radians(float(pitch_deg)),
        _Q3_MIN,
        _Q3_MAX,
    ))
    stats = {
        "mode": "q3_only",
        "pitch_deg": round(float(pitch_deg), 3),
        "q3_start_rad": round(float(q_act[2]), 4),
        "q3_target_rad": round(q3_tgt, 4),
        "q1_locked": round(float(q_act[0]), 4),
        "q2_locked": round(float(q_act[1]), 4),
    }
    if abs(q3_tgt - q_act[2]) <= float(q_tol_rad):
        stats["reached"] = True
        stats["steps"] = 0
        return stats

    # 进度按仿真步计、超时按墙钟计会在低 fps 下误杀可达 pitch。
    # 改为：步数预算 = ceil(|Δ|/step)×2 + 余量；墙钟只留宽松安全上限。
    delta0 = abs(q3_tgt - float(q_act[2]))
    step_budget = max(8, int(math.ceil(delta0 / max(1e-6, float(max_step_rad))) * 2) + 4)
    wall_cap_s = max(float(timeout_s), 180.0)
    stats["step_budget"] = int(step_budget)
    stats["wall_cap_s"] = round(float(wall_cap_s), 1)

    t0 = time.time()
    steps = 0
    in_tol_n = 0
    while steps < step_budget and (time.time() - t0) < wall_cap_s:
        q_act = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        err = q3_tgt - float(q_act[2])
        if abs(err) <= float(q_tol_rad):
            in_tol_n += 1
            if in_tol_n >= _PITCH_SETTLE_FRAMES:
                q_hold = [
                    float(q_locked[0]),
                    float(q_locked[1]),
                    q3_tgt,
                    float(q_locked[2]),
                ]
                for _ in range(max(1, int(n_hold))):
                    yield world.make_action_trunk_locked(q_hold)
                chest = world.chest_pose()
                q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
                stats.update({
                    "reached": True,
                    "steps": steps,
                    "q1_end_rad": round(float(q_fin[0]), 4),
                    "q2_end_rad": round(float(q_fin[1]), 4),
                    "q3_end_rad": round(float(q_fin[2]), 4),
                    "q4_end_rad": round(float(q_fin[3]), 4),
                    "locked_q_drift_rad": round(
                        float(max(
                            abs(float(q_fin[0]) - float(q_locked[0])),
                            abs(float(q_fin[1]) - float(q_locked[1])),
                            abs(float(q_fin[3]) - float(q_locked[2])),
                        )),
                        4,
                    ),
                    "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
                    "chest_z": round(float(chest["z"]), 4),
                })
                ctx.log(
                    f"{log_prefix} q3 DONE "
                    f"q3 {stats['q3_start_rad']:+.3f}→{q3_tgt:+.3f}rad "
                    f"θz={chest['theta_z_deg']:.1f}°"
                )
                return stats
        else:
            in_tol_n = 0

        step = float(np.clip(err, -max_step_rad, max_step_rad))
        q_cmd = [
            float(q_locked[0]),
            float(q_locked[1]),
            float(q_act[2]) + step,
            float(q_locked[2]),
        ]
        for _ in range(6):
            yield world.make_action_trunk_locked(q_cmd)
        steps += 1
        if steps % 4 == 0:
            chest = world.chest_pose()
            ctx.set_status(
                f"pitch q3 {steps}/{step_budget} "
                f"q3={q_act[2]:+.2f}→{q3_tgt:+.2f} "
                f"θz={chest['theta_z_deg']:.1f}°"
            )

    # 接近目标时直接运动学吸附，避免 0.02rad 容差压线误报 TIMEOUT
    q_act = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    err_final = q3_tgt - float(q_act[2])
    if abs(err_final) <= max(float(q_tol_rad) * 2.5, 0.05):
        q_hold = [
            float(q_locked[0]),
            float(q_locked[1]),
            q3_tgt,
            float(q_locked[2]),
        ]
        for _ in range(max(2, int(n_hold))):
            yield world.make_action_trunk_locked(q_hold)
        chest = world.chest_pose()
        q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        stats.update({
            "reached": True,
            "steps": steps,
            "snapped": True,
            "q3_end_rad": round(float(q_fin[2]), 4),
            "q3_err_rad": round(q3_tgt - float(q_fin[2]), 4),
            "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
            "chest_z": round(float(chest["z"]), 4),
        })
        ctx.log(
            f"{log_prefix} q3 SNAP "
            f"q3 {stats['q3_start_rad']:+.3f}→{q3_tgt:+.3f}rad "
            f"θz={chest['theta_z_deg']:.1f}°"
        )
        return stats

    stats.update({
        "reached": False,
        "steps": steps,
        "q3_end_rad": round(float(q_act[2]), 4),
        "q3_err_rad": round(err_final, 4),
    })
    ctx.log(f"{log_prefix} q3 TIMEOUT err={stats.get('q3_err_rad')}rad steps={steps}/{step_budget}")
    return stats


class _PitchKeepOriController:
    """Track EEF orientation with J5-J7 during the final trunk pitch only."""

    def __init__(
        self,
        ctx,
        world,
        keep_ori_arm: str | None,
        *,
        log_prefix: str,
    ) -> None:
        from behavior_interface.skills.reset_body import _EefSmoothActiveBalancer

        self.ctx = ctx
        self.world = world
        self.log_prefix = str(log_prefix)
        self._balancer = _EefSmoothActiveBalancer(
            ctx,
            world,
            keep_ori_arm,
            log_prefix=self.log_prefix,
        )

    @property
    def active(self) -> bool:
        return self._balancer.active

    @property
    def arms(self) -> Set[str]:
        return set(self._balancer.arms)

    @property
    def requested_arms(self) -> Set[str]:
        return set(self._balancer.requested_arms)

    @property
    def arm_q(self) -> Dict[str, np.ndarray]:
        return self._balancer.arm_q

    def make_action(self, trunk_q):
        from behavior_interface.skills.eef import _assert_legacy_7dof_motion_ready
        from behavior_interface.skills.reset_body import _set_trunk_and_arms_direct

        trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
        arm_map = self._balancer.solve_for_trunk(trunk)
        if not _challenge_action_only():
            try:
                for arm in sorted(self.arms):
                    _assert_legacy_7dof_motion_ready(self.world, arm)
                _set_trunk_and_arms_direct(self.world, trunk, arm_map)
                self._balancer.apply_payload_hold()
            except Exception as exc:
                self.ctx.log(
                    f"{self.log_prefix} eef_balance 同步设置失败: "
                    f"{type(exc).__name__}: {exc}"
                )
        overrides: Dict[str, Any] = {"trunk": trunk.tolist()}
        for arm, q_cmd in arm_map.items():
            overrides[f"arm_{arm}"] = q_cmd.tolist()
        return self.world.make_action(**overrides)

    def set_direct(self, trunk_q) -> None:
        if _challenge_action_only():
            return
        if not self.active:
            _kinematic_set_trunk_q(self.world, trunk_q)
            return
        from behavior_interface.skills.reset_body import _set_trunk_and_arms_direct

        trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
        arm_map = self._balancer.solve_for_trunk(trunk)
        _set_trunk_and_arms_direct(
            self.world,
            trunk,
            arm_map,
        )
        self._balancer.apply_payload_hold()

    def record(self) -> None:
        self._balancer.after_step()

    def report(self) -> dict:
        return self._balancer.report(
            scope="final_trunk_pitch_after_base_yaw",
        )


def _yield_pitch_keep_ori_action(
    world,
    trunk_q,
    keep_ori: _PitchKeepOriController | None,
    *,
    direct_after: bool = False,
):
    if keep_ori is not None and keep_ori.active:
        action = keep_ori.make_action(trunk_q)
    else:
        action = world.make_action_trunk_locked(trunk_q)
    yield action
    if keep_ori is not None and keep_ori.active:
        if direct_after:
            keep_ori.set_direct(trunk_q)
        keep_ori.record()


def _yield_trunk_theta_z_pitch_closed_loop(
    ctx,
    world,
    *,
    theta_z_tgt: float,
    max_step_rad: float = 0.08,
    theta_z_tol_deg: float = 1.5,
    timeout_s: float = 60.0,
    log_prefix: str = "trunk/pitch",
    pitch_keep_ori: _PitchKeepOriController | None = None,
) -> dict:
    """纯 q3 俯仰，每步按实际 θz 误差闭环（近目标减速 + 过冲回退）。"""
    from behavior_interface.trunk_vertical_lift import q3_pitch_step_dir

    stats: dict = {
        "mode": "theta_z_closed_loop_q3",
        "theta_z_target_deg": round(float(theta_z_tgt), 2),
    }
    t0 = time.time()
    steps = 0
    pitch_settle_n = 0
    prev_err_tz: float | None = None

    while time.time() - t0 < float(timeout_s):
        chest = world.chest_pose()
        q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        err_tz = _norm_angle_deg(float(theta_z_tgt) - float(chest["theta_z_deg"]))

        sign_flip = (
            prev_err_tz is not None
            and prev_err_tz * err_tz < -1e-6
        )
        in_tol = abs(err_tz) <= float(theta_z_tol_deg)
        if in_tol and not sign_flip:
            pitch_settle_n += 1
            if pitch_settle_n >= _PITCH_SETTLE_FRAMES:
                hold_q = q.tolist()
                for _ in range(3):
                    yield from _yield_pitch_keep_ori_action(
                        world, hold_q, pitch_keep_ori,
                    )
                chest_f = world.chest_pose()
                stats.update({
                    "reached": True,
                    "steps": steps,
                    "theta_z_deg": round(float(chest_f["theta_z_deg"]), 2),
                    "theta_z_err_deg": round(err_tz, 2),
                })
                ctx.log(
                    f"{log_prefix} θz DONE "
                    f"{chest['theta_z_deg']:.1f}°→{theta_z_tgt:.1f}° "
                    f"err={err_tz:+.2f}° steps={steps}"
                )
                return stats
        else:
            pitch_settle_n = 0

        step_dir = q3_pitch_step_dir(
            float(q[0]), float(q[1]), float(q[2]), float(theta_z_tgt),
        )
        if step_dir == 0:
            stats.update({
                "reached": in_tol,
                "steps": steps,
                "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
                "theta_z_err_deg": round(err_tz, 2),
                "q3_saturated": True,
            })
            if in_tol:
                ctx.log(
                    f"{log_prefix} θz 到位(q3饱和) "
                    f"θz={chest['theta_z_deg']:.1f}° err={err_tz:+.2f}°"
                )
            else:
                ctx.log(
                    f"{log_prefix} q3 饱和未达 θz={theta_z_tgt:.1f}° "
                    f"当前={chest['theta_z_deg']:.1f}° err={err_tz:+.1f}°"
                )
            return stats

        step_lim = _pitch_step_lim_rad(err_tz, max_step_rad)
        dq3 = float(step_dir) * step_lim
        q3_next = float(np.clip(q[2] + dq3, _Q3_MIN, _Q3_MAX))
        q_cmd = [float(q[0]), float(q[1]), q3_next, float(q[3])]
        yield from _yield_pitch_keep_ori_action(
            world, q_cmd, pitch_keep_ori,
        )
        steps += 1
        prev_err_tz = err_tz
        if steps % 4 == 0:
            ctx.set_status(
                f"pitch θz {steps} "
                f"θz={chest['theta_z_deg']:.1f}°→{theta_z_tgt:.1f}° "
                f"err={err_tz:+.1f}°"
            )

    chest = world.chest_pose()
    err_tz = _norm_angle_deg(float(theta_z_tgt) - float(chest["theta_z_deg"]))
    stats.update({
        "reached": False,
        "steps": steps,
        "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
        "theta_z_err_deg": round(err_tz, 2),
    })
    ctx.log(
        f"{log_prefix} θz TIMEOUT err={err_tz:+.1f}° "
        f"θz={chest['theta_z_deg']:.1f}° tgt={theta_z_tgt:.1f}°"
    )
    return stats


def _yield_trunk_to_qpos_closed_loop(
    ctx,
    world,
    *,
    target_q,
    max_step_rad: float = 0.08,
    q_tol_rad: float = 0.025,
    timeout_s: float = 60.0,
    n_hold: int = 3,
    log_prefix: str = "trunk/qpos",
) -> dict:
    """按绝对 trunk q 执行；用于 move_to* 的低位 q3 cap + q1 repair 俯仰。"""
    tgt = np.asarray(target_q, dtype=np.float64).reshape(4)
    lohi = _TRUNK_Q_LIMITS
    for i, (lo, hi) in enumerate(lohi):
        tgt[i] = float(np.clip(tgt[i], lo, hi))

    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    stats: dict = {
        "mode": "trunk_qpos_closed_loop",
        "q_start": [round(float(x), 4) for x in q0.tolist()],
        "q_target": [round(float(x), 4) for x in tgt.tolist()],
    }
    t0 = time.time()
    steps = 0
    in_tol_n = 0
    while time.time() - t0 < float(timeout_s):
        q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        err = tgt - q
        err_inf = float(np.max(np.abs(err)))
        if err_inf <= float(q_tol_rad):
            in_tol_n += 1
            if in_tol_n >= _PITCH_SETTLE_FRAMES:
                for _ in range(max(1, int(n_hold))):
                    yield world.make_action_trunk_locked(tgt.tolist())
                q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
                chest = world.chest_pose()
                stats.update({
                    "reached": True,
                    "steps": steps,
                    "q_end": [round(float(x), 4) for x in q_fin.tolist()],
                    "q_err_inf_rad": round(float(np.max(np.abs(tgt - q_fin))), 4),
                    "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
                    "chest_z": round(float(chest["z"]), 4),
                })
                ctx.log(
                    f"{log_prefix} qpos DONE "
                    f"q={[round(float(x), 3) for x in q_fin.tolist()]} "
                    f"θz={chest['theta_z_deg']:.1f}° z={chest['z']:.3f}"
                )
                return stats
        else:
            in_tol_n = 0

        step = np.clip(err, -float(max_step_rad), float(max_step_rad))
        q_cmd = q + step
        for _ in range(2):
            yield world.make_action_trunk_locked(q_cmd.tolist())
        steps += 1
        if steps % 4 == 0:
            chest = world.chest_pose()
            ctx.set_status(
                f"trunk qpos {steps} err={err_inf:.3f} "
                f"θz={chest['theta_z_deg']:.1f}°"
            )

    q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    chest = world.chest_pose()
    stats.update({
        "reached": False,
        "steps": steps,
        "q_end": [round(float(x), 4) for x in q_fin.tolist()],
        "q_err_inf_rad": round(float(np.max(np.abs(tgt - q_fin))), 4),
        "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
        "chest_z": round(float(chest["z"]), 4),
    })
    ctx.log(
        f"{log_prefix} qpos TIMEOUT err={stats['q_err_inf_rad']}rad "
        f"θz={chest['theta_z_deg']:.1f}°"
    )
    return stats


def _solve_trunk_q_analytic(target_chest_z_base: float,
                             target_pitch_rad: float,
                             prev_q1_sign: float) -> Tuple[float, float, float]:
    """解析反解：给定 chest_z(base frame) 和 chest pitch（rad），返回 (q1, q2, q3)。
    q2 = -q1 协同约束，q1 取与 prev_q1_sign 同号的解保持连续性。

    若目标超出物理范围，clip 到能达到的最近值。
    """
    # 推导：q1+q2=0 协同下，chest_pitch_geom = q1+q2-q3 = -q3
    # chest forward = R_y(-q3)·(1,0,0) = (cos q3, 0, sin q3)
    # → fz = sin q3，theta_z = acos(sin q3) = 90° - q3(deg)
    # → q3 = 90° - theta_z = pitch_target_deg（target_pitch_rad 已是这个差）
    q3 = max(_Q3_MIN, min(_Q3_MAX, +target_pitch_rad))
    # chest_z(base) = 0.343 + 0.4·cos(q1) + 0.3 + 0.1·cos(q3)
    # 所以 cos(q1) = (z - 0.643 - 0.1·cos(q3)) / 0.4
    rhs = (target_chest_z_base - 0.643 - _TRUNK_T4_LEN * math.cos(q3)) / _TRUNK_T2_LEN
    rhs = max(-1.0, min(1.0, rhs))  # clip → cos(q1) ∈ [-1, 1]
    q1_abs = math.acos(rhs)  # ∈ [0, π]，clip 到 URDF q1 上限
    q1_abs = min(q1_abs, _Q1_MAX)
    # 取符号：保持连续性。prev_q1_sign 为 0 时默认 +
    sign = 1.0 if prev_q1_sign >= 0 else -1.0
    q1 = sign * q1_abs
    q2 = -q1
    return q1, q2, q3


def _trunk_ik_step(curr_chest, target_chest, curr_q,
                   base_link_z: float,
                   max_step_rad: float = 0.10,
                   max_pitch_step_rad: float | None = None,
                   ik_mode: str = "full"):
    """协同 IK 单步：求 (dq1, dq2, dq3, dq4) 让 chest 朝 (target_z, target_pitch) 走一步。

    Args:
        curr_chest: dict {x,y,z,theta_x_deg,theta_z_deg}（world frame）
        target_chest: dict 同上（z 和 theta_z_deg 是控制目标）
        curr_q: shape (4,) 当前 trunk_qpos = [q1, q2, q3, q4]
        base_link_z: 当前 base_link 在 world 的 z（用 chest_world - chest_base 算）

    Returns:
        delta np.ndarray shape (4,)（弧度），传给 trunk JointController (use_delta_commands)

    ik_mode:
      full — 同时追 z 与俯仰（旧行为）
      z_only — 仅 q1/q2 升降，q3 不动
      pitch_only — 仅 q3 俯仰，q1/q2 不动
      pitch_q1_only — 仅 q1 俯仰（q2/q3 锁定）
      pitch_q3_then_q1 — q3 未达限位先俯 q3；q3 饱和后再俯 q1
    """
    pitch_lim = max_pitch_step_rad if max_pitch_step_rad is not None else max_step_rad
    err_tz_deg = _norm_angle_deg(
        float(target_chest["theta_z_deg"]) - float(curr_chest["theta_z_deg"])
    )
    step_lim = _pitch_step_lim_rad(err_tz_deg, pitch_lim)

    if ik_mode == "z_only":
        target_z_base = target_chest["z"] - base_link_z
        prev_q1_sign = 1.0 if curr_q[0] >= 0 else -1.0
        sol = _solve_q1_q2_for_z(target_z_base, float(curr_q[2]), prev_q1_sign)
        if sol is None:
            return np.zeros(4, dtype=np.float64)
        q1_t, q2_t = sol
        dq1 = float(np.clip(q1_t - curr_q[0], -max_step_rad, max_step_rad))
        dq2 = float(np.clip(q2_t - curr_q[1], -max_step_rad, max_step_rad))
        if abs(curr_q[0] + curr_q[1]) < 0.10:
            dq2 = -dq1
        return np.array([dq1, dq2, 0.0, 0.0], dtype=np.float64)

    if ik_mode == "pitch_only":
        target_pitch_rad = math.radians(90.0 - target_chest["theta_z_deg"])
        q3_t = max(_Q3_MIN, min(_Q3_MAX, target_pitch_rad))
        dq3 = float(np.clip(q3_t - curr_q[2], -step_lim, step_lim))
        return np.array([0.0, 0.0, dq3, 0.0], dtype=np.float64)

    if ik_mode == "pitch_q1_only":
        # 锁定 q2/q3，用 r1pro.urdf FK 反解 q1（与俯仰规划 clamp 一致）
        from behavior_interface.trunk_vertical_lift import solve_q1_for_theta_z_pitch_q1_only

        q1_t = solve_q1_for_theta_z_pitch_q1_only(
            float(target_chest["theta_z_deg"]),
            float(curr_q[1]),
            float(curr_q[2]),
            prefer_q1=float(curr_q[0]),
        )
        if q1_t is None:
            return np.zeros(4, dtype=np.float64)
        q1_t = max(_Q1_MIN, min(_Q1_MAX, q1_t))
        dq1 = float(np.clip(q1_t - curr_q[0], -step_lim, step_lim))
        return np.array([dq1, 0.0, 0.0, 0.0], dtype=np.float64)

    if ik_mode == "pitch_q3_then_q1":
        from behavior_interface.trunk_vertical_lift import (
            q3_pitch_step_dir,
            solve_q1_for_theta_z_pitch_q1_only,
        )

        target_tz = float(target_chest["theta_z_deg"])
        q1, q2, q3 = float(curr_q[0]), float(curr_q[1]), float(curr_q[2])
        step_dir = q3_pitch_step_dir(q1, q2, q3, target_tz)
        if step_dir > 0:
            dq3 = float(min(step_lim, _Q3_MAX - q3))
            return np.array([0.0, 0.0, dq3, 0.0], dtype=np.float64)
        if step_dir < 0:
            dq3 = float(-min(step_lim, q3 - _Q3_MIN))
            return np.array([0.0, 0.0, dq3, 0.0], dtype=np.float64)

        q1_t = solve_q1_for_theta_z_pitch_q1_only(
            target_tz, q2, q3, prefer_q1=q1,
        )
        if q1_t is None:
            return np.zeros(4, dtype=np.float64)
        q1_t = max(_Q1_MIN, min(_Q1_MAX, q1_t))
        dq1 = float(np.clip(q1_t - q1, -step_lim, step_lim))
        return np.array([dq1, 0.0, 0.0, 0.0], dtype=np.float64)

    # 1. 目标转 base frame（full）
    # chest world z = base_link world z + chest z (base frame)
    target_z_base = target_chest["z"] - base_link_z
    # chest forward = R_y(pitch_rad)·(1,0,0) = (cos p, 0, -sin p)
    # theta_z = acos(fz) = acos(-sin p) → sin(p) = -cos(theta_z_rad)
    # 当 theta_z=90° (水平) → p=0；theta_z<90° (抬头) → p>0；theta_z>90° (低头) → p<0
    # 即 pitch_rad ≈ (90° - theta_z_deg) / 57.3
    target_pitch_rad = math.radians(90.0 - target_chest["theta_z_deg"])

    # 2. 解析反解目标 (q1, q2, q3)
    prev_q1_sign = 1.0 if curr_q[0] >= 0 else -1.0
    if abs(curr_q[0]) < 0.05:
        # 站直附近，根据 z_err 方向选择 bend 方向：z_target > z_curr → 想抬高，但 q1=0 已经最高
        # 所以需要 bend 才能升降 —— 任选 + 方向（向前折叠）
        z_err = target_chest["z"] - curr_chest["z"]
        prev_q1_sign = 1.0
    q1_t, q2_t, q3_t = _solve_trunk_q_analytic(target_z_base, target_pitch_rad, prev_q1_sign)

    # 3. delta = q_target - q_curr，每个 joint 限速
    dq1 = q1_t - curr_q[0]
    dq2 = q2_t - curr_q[1]
    dq3 = q3_t - curr_q[2]
    dq4 = 0.0

    delta = np.array([dq1, dq2, dq3, dq4])
    # 升降 q1/q2 与俯仰 q3 分档限速（俯仰过快易晃）
    delta[0] = float(np.clip(dq1, -max_step_rad, max_step_rad))
    delta[1] = float(np.clip(dq2, -max_step_rad, max_step_rad))
    delta[2] = float(np.clip(dq3, -pitch_lim, pitch_lim))
    # 强制 dq2 = -dq1（维持协同约束，避免数值误差累积让 q1+q2 偏离 0）
    # 但只在 |q1+q2| 很小时强制（避免初始偏离时强制摆动）
    if abs(curr_q[0] + curr_q[1]) < 0.10:
        delta[1] = -delta[0]
    return delta


def _trunk_to_target(ctx, target_chest: dict,
                     z_tol: float, theta_z_tol_deg: float,
                     max_step_rad: float, timeout_s: float, log_prefix: str,
                     deadline_mono: float | None = None,
                     max_pitch_step_rad: float | None = None,
                     ik_mode: str = "full"):
    """循环驱动 trunk 让 chest 接近 target_chest (只考虑 z 和 theta_z_deg)。
    底盘保持静止；只发 trunk delta。
    """
    world = ctx.world
    if world.dry_run:
        ctx.log(f"{log_prefix} dry_run 跳过 trunk IK")
        return True
    try:
        _ = world.controller_action_idx("trunk")
    except Exception:
        ctx.log(f"{log_prefix} robot 无 trunk controller，跳过 dz/dthetaz")
        return True

    t_start = time.time()
    last_log = 0.0
    pitch_settle_n = 0
    prev_err_tz_deg: float | None = None
    while True:
        if _deadline_expired(deadline_mono):
            ctx.log(f"{log_prefix} trunk DEADLINE reached")
            return False
        if time.time() - t_start > timeout_s:
            ctx.log(f"{log_prefix} trunk IK TIMEOUT after {timeout_s:.1f}s")
            return False

        chest = world.chest_pose()
        err_z = target_chest["z"] - chest["z"]
        err_theta_z_deg = _norm_angle_deg(target_chest["theta_z_deg"] - chest["theta_z_deg"])

        if ik_mode == "z_only":
            done = abs(err_z) < z_tol
        elif ik_mode in ("pitch_only", "pitch_q1_only", "pitch_q3_then_q1"):
            in_tol = abs(err_theta_z_deg) < theta_z_tol_deg
            sign_flipped = (
                prev_err_tz_deg is not None
                and prev_err_tz_deg * err_theta_z_deg < -1e-6
            )
            if in_tol and not sign_flipped:
                pitch_settle_n += 1
            else:
                pitch_settle_n = 0
            done = in_tol and pitch_settle_n >= _PITCH_SETTLE_FRAMES
        else:
            done = abs(err_z) < z_tol and abs(err_theta_z_deg) < theta_z_tol_deg

        if done:
            ctx.log(
                f"{log_prefix} trunk DONE z={chest['z']:.3f} (err={err_z:+.3f}) "
                f"theta_z={chest['theta_z_deg']:.1f}° (err={err_theta_z_deg:+.1f}°)"
            )
            # absolute mode：发当前 qpos 保持
            hold_q = world.trunk_qpos().tolist()
            for _ in range(3):
                yield world.make_action_trunk_locked(hold_q)
            return True

        curr_q = world.trunk_qpos()
        # 估 base_link world z：用 robot.get_position_orientation()[0].z（base_link 是 root）
        base_z_world = _base_link_z_world(world)
        dq = _trunk_ik_step(
            chest, target_chest, curr_q, base_link_z=base_z_world,
            max_step_rad=max_step_rad,
            max_pitch_step_rad=max_pitch_step_rad,
            ik_mode=ik_mode,
        )

        if time.time() - last_log > 0.5:
            ctx.set_status(
                f"trunk z={chest['z']:.3f}→{target_chest['z']:.3f} "
                f"θz={chest['theta_z_deg']:.0f}°→{target_chest['theta_z_deg']:.0f}° "
                f"q=({curr_q[0]:+.2f},{curr_q[1]:+.2f},{curr_q[2]:+.2f}) "
                f"dq=({dq[0]:+.3f},{dq[1]:+.3f},{dq[2]:+.3f})"
            )
            last_log = time.time()

        # absolute mode：发 curr_q + dq（绝对目标）
        q_target = (np.asarray(curr_q, dtype=np.float64) + dq).tolist()
        yield world.make_action_trunk_locked(q_target)
        prev_err_tz_deg = err_theta_z_deg


# ─────────────────────────────────────────────────────────────────────────────
# move_to: 底盘移动到绝对 (x, y)
# ─────────────────────────────────────────────────────────────────────────────

@register_skill(
    "move_to",
    description=(
        "胸口位姿绝对值移动：x, y 必填（底盘平移目标）；"
        "z, theta_x_deg, theta_z_deg 可选（缺省保留当前胸口姿态）。"
        "z 控升降（trunk 协同），theta_x_deg 控朝向（底盘 yaw），"
        "theta_z_deg 控俯仰（trunk q3）。"
    ),
)
def move_to(
    ctx,
    x: float,
    y: float,
    z: float = None,
    theta_x_deg: float = None,
    theta_z_deg: float = None,
    exec_order: str = "trunk_xy_yaw",
    avoid_obstacles: bool = True,
    pos_tol: float = 0.15,
    yaw_tol_deg: float = 3.0,
    align_tol_deg: float = 25.0,
    vmax: float = 0.6,
    wmax: float = 1.0,
    k_lin: float = 0.8,
    k_ang: float = 2.0,
    extra_inflate: float = 0.0,
    z_tol: float = 0.04,
    theta_z_tol_deg: float = 5.0,
    trunk_max_step_rad: float = 0.10,
    trunk_max_pitch_step_rad: float | None = None,
    trunk_ik_mode: str = "full",
    trunk_timeout_s: float = 30.0,
    timeout_s: float = 120.0,
    waypoints: list | None = None,
    snap_goal: bool = True,
    deadline_mono: float | None = None,
    settle_after: bool = True,
):
    """绝对 5D 胸口位姿。

    示例：
      move_to(x=7.5, y=-0.1)                                  仅底盘平移
      move_to(x=7.5, y=-0.1, theta_x_deg=-90)                 平移 + 转向
      move_to(x=7.5, y=-0.1, z=0.95, theta_z_deg=110)         平移 + 弯腰俯视抓矮物
      move_to(x=7.5, y=-0.1, z=1.18, theta_z_deg=80)          平移 + 抬头仰视抓高物

    执行顺序（exec_order）：
      trunk_xy_yaw（默认）：trunk → 平移 xy → 旋转 yaw
      xy_yaw_trunk：平移 xy → 旋转 yaw → trunk（move_to_object / move_to_point）
      pitch_xy_yaw：trunk 俯仰 → 平移 xy → 旋转 yaw（遗留，点地面易死锁）
    """
    from behavior_interface.skills.eef import _freeze_world_limb_pins
    _freeze_world_limb_pins(ctx.world)

    order = (exec_order or "trunk_xy_yaw").strip().lower()
    if order not in ("trunk_xy_yaw", "xy_yaw_trunk", "pitch_xy_yaw"):
        order = "trunk_xy_yaw"

    def _remaining_timeout() -> float:
        if deadline_mono is not None:
            return max(1.0, deadline_mono - time.time())
        return timeout_s

    report = {
        "xy_ok": True,
        "yaw_ok": True,
        "trunk_ok": True,
        "path_status": None,
        "waypoints_n": 0,
        "snap_goal": bool(snap_goal),
        "stuck_recoveries": 0,
    }

    def _stage_trunk():
        if z is None and theta_z_deg is None:
            return True
        chest0 = ctx.world.chest_pose()
        ik_mode = (trunk_ik_mode or "full").strip().lower()
        if ik_mode not in ("full", "z_only", "pitch_only", "pitch_q1_only", "pitch_q3_then_q1"):
            ik_mode = "full"
        if ik_mode in ("pitch_only", "pitch_q1_only", "pitch_q3_then_q1"):
            z_exec = chest0["z"]
        else:
            z_exec = z if z is not None else chest0["z"]
        target_chest = {
            "x": chest0["x"],
            "y": chest0["y"],
            "z": z_exec,
            "theta_x_deg": chest0["theta_x_deg"],
            "theta_z_deg": theta_z_deg if theta_z_deg is not None else chest0["theta_z_deg"],
        }
        ctx.log(
            f"move_to stage-trunk ik={ik_mode}: z={chest0['z']:.3f}→{target_chest['z']:.3f} "
            f"theta_z={chest0['theta_z_deg']:.1f}°→{target_chest['theta_z_deg']:.1f}°"
        )
        return (yield from _trunk_to_target(
            ctx, target_chest, z_tol, theta_z_tol_deg,
            trunk_max_step_rad, min(trunk_timeout_s, _remaining_timeout()), "move_to",
            deadline_mono=deadline_mono,
            max_pitch_step_rad=trunk_max_pitch_step_rad,
            ik_mode=ik_mode,
        ))

    def _stage_xy():
        n_recover = 0

        def _should_recover() -> bool:
            fail = _nav_seg_fail(ctx)
            if fail == "stuck":
                return True
            if fail == "timeout":
                return True
            return False

        def _replan_inflate() -> float:
            return float(extra_inflate) + n_recover * 0.06

        def _drive_waypoints(wps: list, status: str) -> bool:
            ctx.log(
                f"move_to stage-xy: goal=({x:.2f},{y:.2f}) "
                f"status={status} waypoints={len(wps)} snap_goal={snap_goal} "
                f"recoveries={n_recover}"
            )
            for i, (wx, wy) in enumerate(wps[1:], 1):
                ctx.log(f"move_to seg {i}/{len(wps)-1} -> ({wx:.2f},{wy:.2f})")
                seg_tol = pos_tol if i == len(wps) - 1 else max(pos_tol, 0.25)
                seg_ok = yield from _drive_to_target(
                    ctx, wx, wy, seg_tol, align_tol_deg, vmax, wmax,
                    k_lin, k_ang, _remaining_timeout(), f"move_to[{i}]",
                    deadline_mono=deadline_mono,
                    extra_inflate=extra_inflate,
                    nav_guard=avoid_obstacles,
                )
                if not seg_ok:
                    ctx.log(
                        f"move_to seg {i} 失败 reason={_nav_seg_fail(ctx)}"
                    )
                    return False
            return True

        def _after_seg_fail() -> bool:
            nonlocal n_recover
            if not avoid_obstacles or not _should_recover():
                return False
            if n_recover >= _STUCK_MAX_RECOVERIES:
                return False
            n_recover += 1
            setattr(ctx, "_nav_recover_n", n_recover)
            yield from _recover_nav_stuck(
                ctx,
                inflate_boost=_replan_inflate() - float(extra_inflate),
                deadline_mono=deadline_mono,
                log_prefix="move_to",
            )
            return True

        if waypoints is not None:
            wps_fixed = list(waypoints)
            report["path_status"] = "fixed"
            report["waypoints_n"] = len(wps_fixed)
            while True:
                seg_all_ok = yield from _drive_waypoints(wps_fixed, "fixed")
                if seg_all_ok:
                    report["stuck_recoveries"] = n_recover
                    return True
                if not (yield from _after_seg_fail()):
                    report["stuck_recoveries"] = n_recover
                    return False
                pose = ctx.world.robot_pose()
                sx, sy = float(pose.pos[0]), float(pose.pos[1])
                wps_fixed, status = _plan_or_direct(
                    ctx, sx, sy, x, y, avoid_obstacles, _replan_inflate(), "move_to",
                    snap_goal=snap_goal,
                )
                report["path_status"] = status
                report["waypoints_n"] = len(wps_fixed)

        while True:
            if avoid_obstacles:
                _refresh_nav_scene_graph(ctx, "move_to")
            pose = ctx.world.robot_pose()
            sx, sy = float(pose.pos[0]), float(pose.pos[1])
            repl_inflate = _replan_inflate()
            wps, status = _plan_or_direct(
                ctx, sx, sy, x, y, avoid_obstacles, repl_inflate, "move_to",
                snap_goal=snap_goal,
            )
            report["path_status"] = status
            report["waypoints_n"] = len(wps)
            seg_all_ok = yield from _drive_waypoints(wps, status)
            if seg_all_ok:
                report["stuck_recoveries"] = n_recover
                return True
            if not (yield from _after_seg_fail()):
                report["stuck_recoveries"] = n_recover
                return False

    def _stage_yaw():
        if theta_x_deg is None:
            return True
        target_yaw = math.radians(theta_x_deg)
        pose_now = ctx.world.robot_pose()
        cur_yaw_deg = math.degrees(pose_now.yaw)
        ctx.log(
            f"move_to stage-yaw: {cur_yaw_deg:+.1f}° → {theta_x_deg:+.1f}°"
        )
        return (yield from _rotate_to_target_yaw(
            ctx, target_yaw, yaw_tol_deg, wmax, k_ang, _remaining_timeout(), "move_to",
            deadline_mono=deadline_mono,
        ))

    if order == "xy_yaw_trunk":
        ctx.log("move_to exec_order=xy_yaw_trunk (先平移→再转向→最后俯仰/升降)")
        report["xy_ok"] = bool((yield from _stage_xy()))
        if report["xy_ok"]:
            report["yaw_ok"] = bool((yield from _stage_yaw()))
            report["trunk_ok"] = bool((yield from _stage_trunk()))
        else:
            ctx.log("move_to xy 未到位，跳过 yaw/trunk")
            report["yaw_ok"] = False
            report["trunk_ok"] = False
    elif order == "pitch_xy_yaw":
        ctx.log("move_to exec_order=pitch_xy_yaw (先俯仰→再平移 xy→最后 yaw)")
        report["trunk_ok"] = bool((yield from _stage_trunk()))
        if report["trunk_ok"]:
            report["xy_ok"] = bool((yield from _stage_xy()))
            if report["xy_ok"]:
                report["yaw_ok"] = bool((yield from _stage_yaw()))
            else:
                report["yaw_ok"] = False
        else:
            ctx.log("move_to pitch 未到位，跳过 xy/yaw")
            report["xy_ok"] = False
            report["yaw_ok"] = False
    else:
        report["trunk_ok"] = bool((yield from _stage_trunk()))
        report["xy_ok"] = bool((yield from _stage_xy()))
        report["yaw_ok"] = bool((yield from _stage_yaw()))
    setattr(ctx, "_move_to_report", report)
    if settle_after:
        yield from yield_move_settle(ctx.world)


# ─────────────────────────────────────────────────────────────────────────────
# move_in_world_coord: 世界系 5D delta (dx, dy, dz, dthetax, dthetaz)
# ─────────────────────────────────────────────────────────────────────────────

@register_skill(
    "move_in_world_coord",
    description=(
        "世界系胸口 5D delta：dx, dy（世界 +X/+Y），dz（胸口高度 m），"
        "dthetax（水平 yaw 变化 deg），dthetaz（俯仰变化 deg）。"
        "默认 0；底盘段用 A* 避障，仅前进不倒车。"
    ),
)
def move_in_world_coord(
    ctx,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    dthetax: float = 0.0,
    dthetaz: float = 0.0,
    avoid_obstacles: bool = True,
    pos_tol: float = 0.15,
    yaw_tol_deg: float = 3.0,
    align_tol_deg: float = 25.0,
    vmax: float = 0.6,
    wmax: float = 1.0,
    k_lin: float = 0.8,
    k_ang: float = 2.0,
    extra_inflate: float = 0.0,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 5.0,
    trunk_max_step_rad: float = 0.10,
    trunk_timeout_s: float = 30.0,
    timeout_s: float = 120.0,
):
    """5D 增量移动：先 trunk，再 base 平移，再 base yaw。

    示例：
      move(dx=1.0)                  → 沿 +X 走 1m（带避障）
      move(dthetax=90)              → 原地左转 90°
      move(dthetaz=-20)             → 抬头 20°（forward 与 +Z 夹角 -20°）
      move(dz=-0.15, dthetaz=15)    → 弯腰 ~0.15m 同时低头 15°
      move(dx=1.0, dthetax=-45)     → 走 1m 再右转 45°
    """
    from behavior_interface.skills.eef import _freeze_world_limb_pins
    _freeze_world_limb_pins(ctx.world)

    # 当前胸口 pose（用于算目标）
    chest0 = ctx.world.chest_pose()
    _CHEST_Z_MIN, _CHEST_Z_MAX = 0.66, 1.15
    z_tgt = chest0["z"] + dz
    if abs(dz) > 0.35 or z_tgt > _CHEST_Z_MAX + 0.02 or z_tgt < _CHEST_Z_MIN - 0.02:
        z_clamped = float(np.clip(z_tgt, _CHEST_Z_MIN, _CHEST_Z_MAX))
        ctx.log(
            f"move 警告: dz={dz:+.2f}m 使目标胸口 z={z_tgt:.2f}m 超出物理范围 "
            f"[{_CHEST_Z_MIN},{_CHEST_Z_MAX}]，已钳位为 {z_clamped:.2f}m "
            f"（dz 单位是米，不是度）"
        )
        z_tgt = z_clamped
    target_chest = {
        "x": chest0["x"] + dx,
        "y": chest0["y"] + dy,
        "z": z_tgt,
        "theta_x_deg": _norm_angle_deg(chest0["theta_x_deg"] + dthetax),
        "theta_z_deg": max(70.0, min(118.0, chest0["theta_z_deg"] + dthetaz)),
    }
    ctx.log(
        f"move target chest: "
        f"({chest0['x']:.2f},{chest0['y']:.2f},{chest0['z']:.2f})"
        f"@({chest0['theta_x_deg']:+.1f}°,{chest0['theta_z_deg']:.1f}°) → "
        f"({target_chest['x']:.2f},{target_chest['y']:.2f},{target_chest['z']:.2f})"
        f"@({target_chest['theta_x_deg']:+.1f}°,{target_chest['theta_z_deg']:.1f}°)"
    )

    # ── 阶段 1：trunk（dz / dthetaz）────────────────────────────────────────
    if abs(dz) > 1e-6 or abs(dthetaz) > 1e-6:
        ctx.log(f"move stage-1 trunk: target z={target_chest['z']:.3f} theta_z={target_chest['theta_z_deg']:.1f}°")
        yield from _trunk_to_target(
            ctx, target_chest, z_tol, theta_z_tol_deg,
            trunk_max_step_rad, trunk_timeout_s, "move_world",
        )

    # ── 阶段 2：base 平移（dx, dy）──────────────────────────────────────────
    # 注意：trunk 弯腰会让胸口 xy 略偏移，但 dx/dy 仍按用户输入执行
    # （想要精确"胸口位置"控制需要 full IK，超出当前实现范围）
    pose = ctx.world.robot_pose()
    xr, yr = float(pose.pos[0]), float(pose.pos[1])
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        tx, ty = xr + dx, yr + dy
        waypoints, status = _plan_or_direct(
            ctx, xr, yr, tx, ty, avoid_obstacles, extra_inflate, "move_world",
        )
        ctx.log(
            f"move_world stage-2 translate delta=({dx:+.2f},{dy:+.2f}) -> ({tx:.2f},{ty:.2f}) "
            f"status={status} waypoints={len(waypoints)}"
        )
        for i, (wx, wy) in enumerate(waypoints[1:], 1):
            ctx.log(f"move_world seg {i}/{len(waypoints)-1} -> ({wx:.2f},{wy:.2f})")
            seg_tol = pos_tol if i == len(waypoints) - 1 else max(pos_tol, 0.25)
            yield from _drive_to_target(
                ctx, wx, wy, seg_tol, align_tol_deg, vmax, wmax,
                k_lin, k_ang, timeout_s, f"move_world[{i}]",
            )

    # ── 阶段 3：base yaw 修正（dthetax）────────────────────────────────────
    if abs(dthetax) > 1e-6:
        pose_now = ctx.world.robot_pose()
        current_yaw = float(pose_now.yaw)
        target_yaw = _norm_angle(current_yaw + math.radians(dthetax))
        ctx.log(
            f"move stage-3 rotate dthetax={dthetax:+.1f}° "
            f"({math.degrees(current_yaw):+.1f}° -> {math.degrees(target_yaw):+.1f}°)"
        )
        yield from _rotate_to_target_yaw(
            ctx, target_yaw, yaw_tol_deg, wmax, k_ang, timeout_s, "move_world",
        )

    yield from yield_move_settle(ctx.world)


# 垂直升降：开环播放规划路点（阶段2 含 dq1=dq2，不再现场闭环反解）
_VERT_P1_HOLD = 3
_VERT_P2_HOLD = 3
_VERT_P1_MAX_WAYPOINTS = 28
_VERT_P2_MAX_WAYPOINTS = 48


def _subsample_trunk_waypoints(
    waypoints: List[np.ndarray],
    max_pts: int,
) -> List[np.ndarray]:
    if len(waypoints) <= max(2, int(max_pts)):
        return list(waypoints)
    idx = np.linspace(0, len(waypoints) - 1, int(max_pts), dtype=int)
    out: List[np.ndarray] = []
    seen = -1
    for i in idx:
        if int(i) != seen:
            out.append(waypoints[int(i)])
            seen = int(i)
    if out[-1] is not waypoints[-1]:
        out.append(waypoints[-1])
    return out


def _phase_waypoint_counts(vmeta: dict, n_total: int) -> Tuple[int, int, int]:
    """从规划 meta 拆阶段1/2/3 路点数（阶段2/3 拼接时去掉重复首点）。"""
    p1_n = p2_n = p3_n = 0
    for ph in vmeta.get("phases", []):
        n_ph = int(ph.get("n_waypoints", 0))
        if ph.get("phase") == "sine_manifold":
            p1_n = n_ph
        elif ph.get("phase") == "q1q2_locked_q3":
            p2_n = max(0, n_ph - 1)
        elif ph.get("phase") == "q2q3_locked_q1":
            p3_n = max(0, n_ph - 1)
    if p1_n + p2_n + p3_n > n_total:
        p3_n = max(0, n_total - p1_n - p2_n)
    return p1_n, p2_n, p3_n


def _yield_trunk_waypoints_openloop(
    ctx,
    world,
    waypoints: List[np.ndarray],
    *,
    phase_tag: str,
    n_hold: int = 2,
    status_every: int = 4,
):
    """开环下发 trunk 路点（无 IK）。"""
    for i, q_wp in enumerate(waypoints):
        for _ in range(max(1, int(n_hold))):
            yield world.make_action_trunk_locked(np.asarray(q_wp, dtype=np.float64).tolist())
        if status_every > 0 and (i % status_every == 0 or i == len(waypoints) - 1):
            chest = world.chest_pose()
            q_act = world.trunk_qpos()
            ctx.set_status(
                f"{phase_tag} {i+1}/{len(waypoints)} "
                f"z={chest['z']:.3f} "
                f"q1={q_act[0]:+.2f} q2={q_act[1]:+.2f} q3={q_act[2]:+.2f}"
            )


def _clip_trunk_q(q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64).reshape(4).copy()
    for i, (lo, hi) in enumerate(_TRUNK_Q_LIMITS):
        q_arr[i] = float(np.clip(q_arr[i], lo, hi))
    return q_arr


def _relative_reverse_upward_waypoints(
    world,
    waypoints: List[np.ndarray],
    meta: dict,
) -> Tuple[List[np.ndarray], dict]:
    """把标准直立 LUT 的绝对 q 轨迹转换成相对当前 trunk 起点的 dq 轨迹。"""
    if not waypoints:
        return [], {"relative_lut": True, "n_waypoints": 0}

    q_start_actual = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    if meta.get("q_start") and len(meta["q_start"]) >= 4:
        q_start_lut = np.asarray(meta["q_start"], dtype=np.float64).reshape(4)
    else:
        q_start_lut = np.asarray(waypoints[0], dtype=np.float64).reshape(4)
    converted: List[np.ndarray] = []
    clipped = 0
    for q_lut in waypoints:
        q_lut_arr = np.asarray(q_lut, dtype=np.float64).reshape(4)
        q_rel = q_start_actual + (q_lut_arr - q_start_lut)
        q_rel[3] = q_start_actual[3]
        q_clip = _clip_trunk_q(q_rel)
        if float(np.max(np.abs(q_clip - q_rel))) > 1e-6:
            clipped += 1
        converted.append(q_clip)

    q_end = converted[-1]
    return converted, {
        "relative_lut": True,
        "q_actual_start": [round(float(x), 4) for x in q_start_actual[:4]],
        "q_lut_start": [round(float(x), 4) for x in q_start_lut[:4]],
        "q_lut_end": [round(float(x), 4) for x in np.asarray(waypoints[-1]).reshape(4)[:4]],
        "q_cmd_end": [round(float(x), 4) for x in q_end[:4]],
        "dq_lut_end": [
            round(float(x), 4)
            for x in (np.asarray(waypoints[-1], dtype=np.float64).reshape(4) - q_start_lut)[:4]
        ],
        "q2_start_rad": round(float(q_start_actual[1]), 4),
        "q2_end_cmd_rad": round(float(q_end[1]), 4),
        "q2_delta_cmd_rad": round(float(q_end[1] - q_start_actual[1]), 4),
        # Historical reverse-fold LUT validated in skills/test/reverse_vertical_z_ik_*.png:
        # descending folds q1 negative, q2 positive, q3 positive in this joint convention.
        "q2_reverse_fold": bool(float(q_end[1] - q_start_actual[1]) > 1e-4),
        "clipped_waypoints": int(clipped),
        "n_waypoints": len(converted),
    }


def _yield_trunk_settle_to_pose(
    ctx,
    world,
    q_target: np.ndarray,
    *,
    theta_z_deg: float = 90.0,
    q_tol_rad: float = 0.022,
    tz_tol_deg: float = 2.5,
    max_steps: int = 160,
    settle_frames: int = 6,
    n_hold: int = 2,
) -> dict:
    """闭环保持目标 trunk 角，直到关节与 θz 连续多帧进入容差（非惯性欠调）。"""
    from behavior_interface.trunk_vertical_lift import estimate_chest_z_world

    q_tgt = np.asarray(q_target, dtype=np.float64).reshape(4)
    in_tol_n = 0
    last: dict = {}
    for step in range(max(1, int(max_steps))):
        for _ in range(max(1, int(n_hold))):
            yield world.make_action_trunk_locked(q_tgt.tolist())
        q_act = world.trunk_qpos()
        chest = world.chest_pose()
        q_err = float(np.max(np.abs(q_act[:3] - q_tgt[:3])))
        tz_err = abs(float(chest["theta_z_deg"]) - float(theta_z_deg))
        z_err = abs(float(chest["z"]) - float(
            estimate_chest_z_world(q_tgt, _base_link_z_world(world))
        ))
        last = {
            "steps": step + 1,
            "q_err_rad": round(q_err, 4),
            "tz_err_deg": round(tz_err, 3),
            "z_err_m": round(z_err, 4),
            "q_act": [round(float(x), 4) for x in q_act[:3]],
            "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
            "chest_z": round(float(chest["z"]), 4),
        }
        if q_err <= float(q_tol_rad) and tz_err <= float(tz_tol_deg):
            in_tol_n += 1
            if in_tol_n >= int(settle_frames):
                last["reached"] = True
                return last
        else:
            in_tol_n = 0
        if step % 12 == 0:
            ctx.set_status(
                f"settle q_err={q_err:.3f} tz_err={tz_err:.1f}° "
                f"step {step+1}/{max_steps}"
            )
    last["reached"] = False
    ctx.log(
        f"[trunk_settle] 未完全到位: q_err={last.get('q_err_rad')} "
        f"tz_err={last.get('tz_err_deg')}°"
    )
    return last


def _yield_reverse_upward_trajectory(
    ctx,
    world,
    waypoints: List[np.ndarray],
    meta: dict,
    *,
    theta_z_deg: float = 90.0,
    n_hold: int = 3,
    settle_final: bool = True,
    require_theta_settle: bool = True,
    max_duration_s: float | None = None,
    max_fast_waypoints: int | None = None,
) -> dict:
    """沿反向 upward 查表路点开环行走，末点闭环到位。"""
    if not waypoints:
        return {"n_exec": 0, "reached": True}
    waypoints_exec = list(waypoints)
    fast_mode = max_duration_s is not None and float(max_duration_s) > 0.0
    n_cap_by_step = None
    if fast_mode:
        n_cap = max(2, int(max_fast_waypoints or 12))
        # 抽稀不能让单步关节位移超过步限：路点是开环下发的，单步给太大物理跟不上，
        # 下降就走不到底（实测落差 1.45m 时 48 点抽成 2 点，q2 指令 2.48rad 只跟上
        # 0.16rad，chest_z 只降了 4mm 便报「upward 末点未到位」）。故按总关节位移
        # 反推所需最少路点数，抬高抽稀下限；落差小时算出的下限低于 n_cap，行为不变。
        dq_total = float(
            np.max(
                np.abs(
                    np.asarray(waypoints[-1], dtype=np.float64).reshape(-1)
                    - np.asarray(waypoints[0], dtype=np.float64).reshape(-1)
                )
            )
        )
        # n 个路点之间只有 n-1 段，所以要 +1 才能保证每段都不超步限
        n_cap_by_step = int(math.ceil(dq_total / _TRUNK_FAST_MAX_STEP_RAD)) + 1
        n_cap = max(n_cap, min(len(waypoints_exec), n_cap_by_step))
        if len(waypoints_exec) > n_cap:
            waypoints_exec = _subsample_trunk_waypoints(waypoints_exec, n_cap)
        n_hold = 1
        settle_final = False

    stats = {
        "exec_mode": "reverse_upward_lut",
        "n_waypoints": len(waypoints),
        "n_waypoints_exec": len(waypoints_exec),
        "direction": meta.get("direction"),
        "z_tgt_m": meta.get("z_tgt_m"),
        "phases_used": meta.get("phases_used"),
    }
    if fast_mode:
        stats["fast_duration_limit_s"] = round(float(max_duration_s), 3)
        stats["fast_waypoint_subsampled"] = len(waypoints_exec) < len(waypoints)
        stats["fast_waypoints_min_by_step_limit"] = n_cap_by_step
        stats["fast_step_limit_rad"] = _TRUNK_FAST_MAX_STEP_RAD
    ctx.log(
        f"move_in_robot_coord 反向upward: {meta.get('direction')} "
        f"z {meta.get('z_curr_m')}→{meta.get('z_tgt_m')}m "
        f"(对齐 upward={meta.get('upward_snapped_m'):+.3f}m) "
        f"路点={len(waypoints_exec)}/{len(waypoints)} "
        f"phase边界={meta.get('z_boundary_m')}m"
        + (f" max={float(max_duration_s):.1f}s" if fast_mode else "")
    )
    if meta.get("snap_note"):
        ctx.log(f"move_in_robot_coord {meta['snap_note']}")
    if meta.get("clamp_note"):
        ctx.log(f"move_in_robot_coord 警告: {meta['clamp_note']}")

    t0 = time.time()
    deadline = t0 + float(max_duration_s) if fast_mode else None
    for i, q_wp in enumerate(waypoints_exec):
        is_last = i == len(waypoints_exec) - 1
        if deadline is not None and time.time() >= deadline and not is_last:
            stats["deadline_hit_before_last"] = True
            q_wp = waypoints_exec[-1]
            is_last = True
        if is_last and settle_final:
            settle = yield from _yield_trunk_settle_to_pose(
                ctx, world, q_wp,
                theta_z_deg=float(theta_z_deg),
                q_tol_rad=0.025,
                tz_tol_deg=3.0 if bool(require_theta_settle) else 180.0,
                max_steps=200,
                settle_frames=5,
                n_hold=max(2, int(n_hold)),
            )
            stats["final_settle"] = settle
            stats["require_theta_settle"] = bool(require_theta_settle)
            stats["reached"] = bool(settle.get("reached", False))
        else:
            phase_list = meta.get("phases_used", [0]) or [0]
            phase_idx = min(i, len(phase_list) - 1)
            yield from _yield_trunk_waypoints_openloop(
                ctx, world, [q_wp],
                phase_tag=f"rev_up p{phase_list[phase_idx]}",
                n_hold=max(1 if fast_mode else 2, int(n_hold)),
            )
        if fast_mode and is_last and not _challenge_action_only():
            try:
                robot = getattr(world, "robot", None)
                q_full = robot.get_joint_positions()
                q_new = q_full.clone() if hasattr(q_full, "clone") else np.asarray(q_full, dtype=np.float64).copy()
                tidx = np.asarray(robot.trunk_control_idx).astype(int)
                q_arr = np.asarray(q_wp, dtype=np.float64).reshape(len(tidx))
                for local_i, joint_i in enumerate(tidx):
                    q_new[int(joint_i)] = float(q_arr[int(local_i)])
                robot.set_joint_positions(q_new)
                try:
                    v_full = robot.get_joint_velocities()
                    v_new = v_full.clone() if hasattr(v_full, "clone") else np.asarray(v_full, dtype=np.float64).copy()
                    for joint_i in tidx:
                        v_new[int(joint_i)] = 0.0
                    robot.set_joint_velocities(v_new)
                except Exception:
                    pass
                if hasattr(world, "set_trunk_pin_qpos"):
                    world.set_trunk_pin_qpos(q_arr)
                stats["fast_final_direct_set"] = True
            except Exception as exc:
                stats["fast_final_direct_set_error"] = f"{type(exc).__name__}: {exc}"
            for _ in range(1):
                yield world.make_action_trunk_locked(np.asarray(q_wp, dtype=np.float64).reshape(4).tolist())
        if i % 3 == 0 or is_last:
            chest = world.chest_pose()
            ctx.set_status(
                f"rev_upward {i+1}/{len(waypoints_exec)} "
                f"z={chest['z']:.3f} θz={chest['theta_z_deg']:.1f}°"
            )
        if deadline is not None and time.time() >= deadline and is_last:
            break

    chest = world.chest_pose()
    stats["chest_z"] = round(float(chest["z"]), 4)
    z_tgt = float(meta.get("z_tgt_m", chest["z"]))
    stats["z_err"] = round(float(chest["z"]) - z_tgt, 4)
    stats["n_exec"] = len(waypoints_exec)
    if fast_mode:
        stats["elapsed_s"] = round(float(time.time() - t0), 3)
        stats["reached"] = abs(float(stats["z_err"])) <= 0.05
    else:
        stats.setdefault("reached", True)
    return stats


def yield_reverse_upward_relative_lut(
    ctx,
    world,
    *,
    upward_delta_m: float | None = None,
    chest_z_target: float | None = None,
    theta_z_deg: float = 90.0,
    z_tol: float = 0.05,
    prepare_arms: bool = False,
    log_prefix: str = "move_robot/upward",
    max_duration_s: float | None = None,
    max_fast_waypoints: int | None = None,
) -> dict:
    """公共升降执行器：用 1cm LUT 的相对 dq 轨迹驱动当前 trunk。

    upward_delta_m 是相对当前 chest z 的位移；chest_z_target 是绝对 chest z。
    LUT 本身仍按标准直立模型选高度格点，执行时转换为 q_current + (q_lut - q_lut_start)。
    """
    from behavior_interface.trunk_vertical_lift import (
        get_reverse_upward_combined_lut,
        plan_reverse_upward_trajectory_from_upward,
    )

    chest0 = world.chest_pose()
    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    base_z = _base_link_z_world(world)
    z_curr = float(chest0["z"])
    theta_z_hold_deg = float(chest0["theta_z_deg"])
    if chest_z_target is None:
        z_tgt_raw = z_curr + float(upward_delta_m or 0.0)
    else:
        z_tgt_raw = float(chest_z_target)
    report: dict = {
        "ok": True,
        "mode": "reverse_upward_relative_lut",
        "build": "relative_lut_reset_grasp_only_v1",
        "z_start_m": round(z_curr, 4),
        "z_target_raw_m": round(float(z_tgt_raw), 4),
        "upward_delta_cmd_m": round(float(z_tgt_raw - z_curr), 4),
        "theta_z_start_deg": round(float(chest0["theta_z_deg"]), 2),
        "q_start": [round(float(x), 4) for x in q0[:4]],
    }
    if abs(float(z_tgt_raw - z_curr)) <= max(0.004, float(z_tol) * 0.25):
        report["skipped"] = True
        report["reason"] = "target height near current"
        return report

    lut = get_reverse_upward_combined_lut(
        base_z,
        q4=float(q0[3]),
        theta_z_deg=float(theta_z_deg),
    )
    if not lut.get("ok"):
        report["ok"] = False
        report["error"] = lut.get("error", "反向 upward 查表失败")
        return report
    upward_for_planner = float(z_tgt_raw) - float(lut["z_upright_m"])
    waypoints_abs, vmeta = plan_reverse_upward_trajectory_from_upward(
        z_curr,
        upward_for_planner,
        base_z,
        q4=float(q0[3]),
        theta_z_deg=float(theta_z_deg),
    )
    report["reverse_upward"] = vmeta
    if vmeta.get("direction") == "hold" and not waypoints_abs:
        report["skipped"] = True
        report["reason"] = vmeta.get("note", "hold")
        return report
    if not vmeta.get("ok") or not waypoints_abs:
        report["ok"] = False
        report["error"] = vmeta.get("error", "反向 upward 规划失败")
        return report

    waypoints, rel_meta = _relative_reverse_upward_waypoints(world, waypoints_abs, vmeta)
    report["relative_lut"] = rel_meta
    ctx.log(
        f"{log_prefix} 相对LUT升降: z {z_curr:.3f}→{vmeta.get('z_tgt_m'):.3f}m "
        f"cmd_delta={float(z_tgt_raw - z_curr):+.3f}m "
        f"q2 {rel_meta['q2_start_rad']:+.3f}→{rel_meta['q2_end_cmd_rad']:+.3f} "
        f"({'legacy reverse-fold' if rel_meta['q2_reverse_fold'] else 'unfold/up'}) "
        f"路点={len(waypoints)}"
    )
    if rel_meta.get("clipped_waypoints"):
        ctx.log(
            f"{log_prefix} 警告: 相对LUT有 {rel_meta['clipped_waypoints']} 个路点触及关节限位"
        )

    exec_stats = yield from _yield_reverse_upward_trajectory(
        ctx,
        world,
        waypoints,
        vmeta,
        theta_z_deg=float(theta_z_hold_deg),
        n_hold=_VERT_P2_HOLD,
        require_theta_settle=False,
        max_duration_s=max_duration_s,
        max_fast_waypoints=max_fast_waypoints,
    )
    chest1 = world.chest_pose()
    q1 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    report["vertical_exec"] = exec_stats
    report["z_end_m"] = round(float(chest1["z"]), 4)
    report["theta_z_end_deg"] = round(float(chest1["theta_z_deg"]), 2)
    report["q_end"] = [round(float(x), 4) for x in q1[:4]]
    report["z_err_m"] = round(float(chest1["z"]) - float(vmeta.get("z_tgt_m", chest1["z"])), 4)
    report["theta_z_err_deg"] = round(
        _norm_angle_deg(float(theta_z_hold_deg) - float(chest1["theta_z_deg"])), 2,
    )
    report["q2_delta_actual_rad"] = round(float(q1[1] - q0[1]), 4)
    report["q2_reverse_fold_actual"] = bool(float(q1[1] - q0[1]) > 1e-4)
    if not exec_stats.get("reached", True):
        report["ok"] = False
        report["error"] = "upward 末点未到位"
    return report


def yield_trunk_deltaz_planned_pose(
    ctx,
    world,
    *,
    chest_z_tgt: float,
    theta_z_tgt: float,
    object_z: float | None = None,
    reach_m: float | None = None,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 1.5,
    trunk_max_step_rad: float = 0.08,
    trunk_timeout_s: float = 30.0,
    log_prefix: str = "move_object/deltaz",
) -> dict:
    """deltaz 末段：按离线模型 chest_z + theta_z 协同 full IK；附 FK 俯仰诊断。"""
    from behavior_interface.skills.base_chord_reach import reach_sphere_radius
    from behavior_interface.skills.move_to_object_geom import (
        read_shoulder_head_pose,
        solve_theta_z_ray_sphere,
    )
    from behavior_interface.trunk_vertical_lift import clamp_theta_z_pitch_q3_then_q1

    report: dict = {
        "ok": True,
        "mode": "deltaz_full_ik_offline_model",
        "theta_z_model_deg": round(float(theta_z_tgt), 2),
    }
    chest0 = world.chest_pose()
    z_err = float(chest_z_tgt) - float(chest0["z"])
    tz_err = _norm_angle_deg(float(theta_z_tgt) - float(chest0["theta_z_deg"]))
    if abs(z_err) <= z_tol and abs(tz_err) <= theta_z_tol_deg:
        report["already_at_target"] = True
        return report

    theta_z_exec = float(theta_z_tgt)
    if object_z is not None:
        R = float(reach_m) if reach_m is not None else reach_sphere_radius()
        pose0 = read_shoulder_head_pose(world)
        pitch_fk = solve_theta_z_ray_sphere(pose0, float(object_z), reach=R)
        tz_fk_geom = float(pitch_fk.get("theta_z_deg", theta_z_tgt))
        tz_fk, _ = clamp_theta_z_pitch_q3_then_q1(
            tz_fk_geom,
            float(pose0["trunk_q1"]),
            float(pose0["trunk_q2"]),
            float(pose0["trunk_q3"]),
            prefer_q1=float(pose0["trunk_q1"]),
            prefer_q3=float(pose0["trunk_q3"]),
        )
        report["pitch_fk_at_deltaz_start"] = {
            "theta_z_deg": round(tz_fk, 2),
            "err_z_m": pitch_fk.get("err_z_m"),
            "intersection_z": pitch_fk.get("intersection_z"),
        }
        delta = float(theta_z_tgt) - float(tz_fk)
        ctx.log(
            f"{log_prefix} 俯仰诊断: 离线θz={theta_z_tgt:.1f}° "
            f"FK@当前姿态={tz_fk:.1f}° Δ={delta:+.1f}° "
            f"err_z={pitch_fk.get('err_z_m')}m "
            f"交点z={pitch_fk.get('intersection_z')} "
            f"(执行仍用离线模型)"
        )

    target_chest = {
        "x": chest0["x"],
        "y": chest0["y"],
        "z": float(chest_z_tgt),
        "theta_x_deg": float(chest0["theta_x_deg"]),
        "theta_z_deg": float(theta_z_exec),
    }
    ctx.log(
        f"{log_prefix} deltaz 协同 z+θz: "
        f"z {chest0['z']:.3f}→{chest_z_tgt:.3f}m "
        f"θz {chest0['theta_z_deg']:.1f}°→{theta_z_exec:.1f}°"
    )
    ok = yield from _trunk_to_target(
        ctx, target_chest,
        float(z_tol), float(theta_z_tol_deg),
        float(trunk_max_step_rad), float(trunk_timeout_s),
        log_prefix,
        ik_mode="full",
    )
    report["trunk_ik_reached"] = bool(ok)
    if not ok:
        report["ok"] = False
        report["error"] = "deltaz 协同 IK 超时"
        return report

    chest_fin = world.chest_pose()
    report["theta_z_target_deg"] = round(float(theta_z_exec), 2)
    report["theta_z_exec_used_deg"] = round(float(theta_z_exec), 2)
    if object_z is not None:
        pose_fin = read_shoulder_head_pose(world)
        R = float(reach_m) if reach_m is not None else reach_sphere_radius()
        pitch_end = solve_theta_z_ray_sphere(pose_fin, float(object_z), reach=R)
        report["pitch_fk_after_deltaz"] = {
            "theta_z_deg": pitch_end.get("theta_z_deg"),
            "err_z_m": pitch_end.get("err_z_m"),
            "intersection_z": pitch_end.get("intersection_z"),
            "chest_theta_z_deg": round(float(chest_fin["theta_z_deg"]), 2),
        }
        ctx.log(
            f"{log_prefix} 俯仰实测: chest θz={chest_fin['theta_z_deg']:.1f}° "
            f"FK建议={pitch_end.get('theta_z_deg')}° "
            f"err_z={pitch_end.get('err_z_m')}m"
        )
    report["chest_z"] = round(float(chest_fin["z"]), 4)
    report["theta_z_deg"] = round(float(chest_fin["theta_z_deg"]), 2)
    report["z_err_m"] = round(float(chest_fin["z"]) - float(chest_z_tgt), 4)
    report["theta_z_err_deg"] = round(
        _norm_angle_deg(float(theta_z_exec) - float(chest_fin["theta_z_deg"])), 2,
    )
    return report


def _kinematic_set_trunk_q(world, q) -> bool:
    """运动学直接把 4 个躯干关节设到 q（清零速度 + pin 锁），保证严格到位、不漂移。"""
    q_arr = np.asarray(q, dtype=np.float64).reshape(4)
    if getattr(world, "dry_run", False):
        return True
    if _challenge_action_only():
        return False
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    try:
        q_full = robot.get_joint_positions()
        q_new = q_full.clone() if hasattr(q_full, "clone") else np.asarray(q_full, dtype=np.float64).copy()
        tidx_raw = robot.trunk_control_idx
        if hasattr(tidx_raw, "detach"):
            tidx_raw = tidx_raw.detach().cpu().numpy()
        tidx = np.asarray(tidx_raw, dtype=int).reshape(-1)[:4]
        if len(tidx) < 4:
            return False
        for local_i, joint_i in enumerate(tidx):
            q_new[int(joint_i)] = float(q_arr[int(local_i)])
        robot.set_joint_positions(q_new)
        try:
            vel = robot.get_joint_velocities()
            v_new = vel.clone() if hasattr(vel, "clone") else np.asarray(vel, dtype=np.float64).copy()
            for joint_i in tidx:
                v_new[int(joint_i)] = 0.0
            robot.set_joint_velocities(v_new)
        except Exception:
            pass
        if hasattr(world, "set_trunk_pin_qpos"):
            world.set_trunk_pin_qpos(q_arr.tolist())
        return True
    except Exception:
        return False


def _install_normal_trunk_hold(world, trunk_q) -> bool:
    """注册 normal 模式「躯干+双臂」保持器：step_once 每步调用，运动学重设关节到
    目标，防止位置控制器撑不住深俯角而下垂/塌低。由非 capture 的下一个 skill 清除。
    """
    if getattr(world, "dry_run", False):
        return False
    if _challenge_action_only():
        return False
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    try:
        import types

        q_trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
        # 解析躯干 + 双臂关节索引
        tidx_raw = robot.trunk_control_idx
        if hasattr(tidx_raw, "detach"):
            tidx_raw = tidx_raw.detach().cpu().numpy()
        tidx = [int(i) for i in np.asarray(tidx_raw, dtype=int).reshape(-1)[:4]]
        names = list(robot.joints.keys())
        arm_idx = []
        for arm in ("left", "right"):
            for k in range(7):
                nm = f"{arm}_arm_joint{k+1}"
                if nm in names:
                    arm_idx.append(int(names.index(nm)))
        # 注册时快照当前双臂关节角（保持 move 结束时的臂姿，不让其垂落）
        q_full0 = robot.get_joint_positions()
        q_full0 = q_full0.clone() if hasattr(q_full0, "clone") else np.asarray(q_full0, dtype=np.float64).copy()
        arm_snapshot = {int(j): float(q_full0[int(j)]) for j in arm_idx}
        hold_idx = [int(j) for j in tidx] + list(arm_snapshot.keys())

        def _normal_hold(self):
            r = getattr(self, "robot", None)
            if r is None:
                return None
            try:
                q_full = r.get_joint_positions()
                q_new = q_full.clone() if hasattr(q_full, "clone") else np.asarray(q_full, dtype=np.float64).copy()
                for local_i, joint_i in enumerate(tidx):
                    q_new[int(joint_i)] = float(q_trunk[int(local_i)])
                for joint_i, val in arm_snapshot.items():
                    q_new[int(joint_i)] = float(val)
                r.set_joint_positions(q_new)
                v0 = r.get_joint_velocities()
                v_new = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
                for joint_i in hold_idx:
                    v_new[int(joint_i)] = 0.0
                r.set_joint_velocities(v_new)
            except Exception:
                pass
            return None

        world.shortcut_post_step_stabilize_now = types.MethodType(_normal_hold, world)
        world._codex_normal_trunk_hold = True
        return True
    except Exception:
        return False


def _yield_trunk_direct_to_q(
    ctx,
    world,
    *,
    q_target,
    chest_z_tgt: float,
    theta_z_tgt: float,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 1.5,
    max_step_rad: float = 0.10,
    timeout_s: float = 45.0,
    settle_steps: int = 8,
    log_prefix: str = "trunk/direct",
    pitch_keep_ori: _PitchKeepOriController | None = None,
) -> dict:
    """把 4 个躯干关节一起步限闭环驱动到规划的 target_trunk_q，再把 pin 锁死到该 q。

    target_trunk_q 由俯仰求解器算出，是「同时满足规划 chest_z + θz」的耦合正确配置。
    相比「upward 升降 → q3 俯仰 → q1q2 微调升降」三段解耦闭环（chest_z 与 θz 在 4 连杆
    躯干上强耦合，三段会互相把对方顶离目标，最终中止留下塌低/θz 失锁），直驱到目标 q
    不会自相矛盾，能严格到位且不漂移、不塌低。
    """
    q_tgt = np.asarray(q_target, dtype=np.float64).reshape(4)
    t0 = time.time()
    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    chest0 = world.chest_pose()
    ctx.log(
        f"{log_prefix} 直驱 target_trunk_q "
        f"q={[round(float(x), 3) for x in q0]}→{[round(float(x), 3) for x in q_tgt]} "
        f"chest_z {chest0['z']:.3f}→{chest_z_tgt:.3f} "
        f"θz {chest0['theta_z_deg']:.1f}°→{theta_z_tgt:.1f}° step≤{max_step_rad:.3f}rad"
    )
    # 臂锁死修复：若没有任何 skill 设过 arm pin，limb_pin_kwargs 会回退成
    # 「每步重读当前臂角」当锁定目标——深俯身时重力把臂一点点拽下去，目标
    # 跟着走，形成棘轮式下垂/外翻（move 结束后保持器又把垂掉的臂姿钉死）。
    # 这里在直驱开始时把当前臂角快照成固定 pin，臂目标不再随动。
    try:
        for _arm in ("left", "right"):
            if world.arm_pin_qpos_list(_arm) is None:
                _aq = world.arm_qpos_list(_arm)
                if _aq is not None:
                    world.set_arm_pin_qpos(_arm, _aq)
                    ctx.log(f"{log_prefix} 臂pin为空，快照锁定 {_arm} 臂当前角")
    except Exception:
        pass
    step_lim = abs(float(max_step_rad))
    steps = 0
    last_log = 0.0
    prev_q_err = float("inf")
    stall_n = 0
    overshoot_n = 0
    converged = False
    while time.time() - t0 < float(timeout_s):
        q_now = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        dq = q_tgt - q_now
        q_err = float(np.max(np.abs(dq)))
        chest = world.chest_pose()
        z_err = float(chest_z_tgt) - float(chest["z"])
        tz_err = _norm_angle_deg(theta_z_tgt - chest["theta_z_deg"])
        now = time.time()
        if steps == 0 or now - last_log >= 1.0:
            ctx.log(
                f"{log_prefix} step={steps} q_err={q_err:.4f}rad "
                f"chest_z={chest['z']:.3f}(err={z_err:+.3f}) "
                f"θz={chest['theta_z_deg']:.1f}°(err={tz_err:+.1f}°)"
            )
            last_log = now
        # 关节已收敛即结束
        if q_err <= 0.01:
            converged = True
            break
        # 俯身过冲检测：q2 在限位/重力压塌时，q1/q2 物理上会往前趴，θz 会越过
        # 目标继续加深（观测到过 163.7°目标冲到 179°趴平）。一旦实测 θz 比目标
        # 深超过 6° 且持续 2 步，立即切运动学吸附纠正，不等失速计满 6 步。
        if tz_err <= -6.0:
            overshoot_n += 1
        else:
            overshoot_n = 0
        if overshoot_n >= 2:
            ctx.log(
                f"{log_prefix} 俯身过冲 @step{steps} θz={chest['theta_z_deg']:.1f}°"
                f"(超目标{-tz_err:.1f}°)，立即运动学吸附到 target_trunk_q"
            )
            break
        # 失速检测：物理被卡（关节到限位/被挡/力矩饱和），连续几步无进展就停物理，交给运动学吸附
        if prev_q_err - q_err < 5e-4:
            stall_n += 1
        else:
            stall_n = 0
        prev_q_err = q_err
        if stall_n >= 6:
            ctx.log(
                f"{log_prefix} 物理逼近失速 @step{steps} q_err={q_err:.4f}rad，"
                f"改用运动学吸附到 target_trunk_q"
            )
            break
        q_cmd = q_now.copy()
        for j in range(4):
            q_cmd[j] = float(q_now[j]) + max(-step_lim, min(step_lim, float(dq[j])))
        yield from _yield_pitch_keep_ori_action(
            world, q_cmd.tolist(), pitch_keep_ori,
        )
        steps += 1
        if steps % 4 == 0:
            ctx.set_status(
                f"trunk直驱 {steps} q_err={q_err:.3f} "
                f"z_err={z_err:+.3f} θz_err={tz_err:+.1f}"
            )

    # 保证严格到位：运动学吸附到 target_trunk_q（清零速度）+ pin 锁；
    # 物理顺利时吸附量≈0（无跳变），物理卡住时由吸附补足，绝不留塌低/漂移姿态。
    if not converged:
        if pitch_keep_ori is not None and pitch_keep_ori.active:
            pitch_keep_ori.set_direct(q_tgt)
        else:
            _kinematic_set_trunk_q(world, q_tgt)
    world.set_trunk_pin_qpos(q_tgt.tolist())
    for _ in range(max(1, int(settle_steps))):
        yield from _yield_pitch_keep_ori_action(
            world,
            q_tgt.tolist(),
            pitch_keep_ori,
            direct_after=True,
        )
        if pitch_keep_ori is None or not pitch_keep_ori.active:
            # 绝对位置控制器会在物理步里把关节往回拉，每帧重设以锁定最终姿态
            _kinematic_set_trunk_q(world, q_tgt)

    # 注册持续保持器：贯穿后续「稳定5s + capture + 空闲」每步重设躯干+双臂，
    # 防止控制器撑不住深俯角而下垂塌低；下一个非 capture skill 开始时自动清除。
    if _install_normal_trunk_hold(world, q_tgt):
        ctx.log(f"{log_prefix} 已注册躯干+双臂持续保持器（防空闲下垂塌低）")

    chest_fin = world.chest_pose()
    q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    z_err_fin = float(chest_z_tgt) - float(chest_fin["z"])
    tz_err_fin = _norm_angle_deg(theta_z_tgt - chest_fin["theta_z_deg"])
    q_err_fin = float(np.max(np.abs(q_tgt - q_fin)))
    # 规划 pose 本质由 target_trunk_q 定义；到位以「关节配置吻合」为主判据
    # （chest_z_tgt 为模型值，sim FK 与模型可能有出入，不作为硬失败条件）。
    reached_q = bool(q_err_fin <= 0.03)
    reached_pose = bool(abs(z_err_fin) <= z_tol and abs(tz_err_fin) <= theta_z_tol_deg)
    reached = bool(reached_q or reached_pose)
    ctx.log(
        f"{log_prefix} 完成 reached={reached}(q={reached_q},pose={reached_pose}) "
        f"steps={steps} q_err={q_err_fin:.4f}rad "
        f"chest_z={chest_fin['z']:.3f}(err={z_err_fin:+.3f}m) "
        f"θz={chest_fin['theta_z_deg']:.1f}°(err={tz_err_fin:+.1f}°) pin已锁"
    )
    return {
        "ok": reached,
        "reached": reached,
        "mode": "direct_target_trunk_q",
        "steps": int(steps),
        "chest_z": round(float(chest_fin["z"]), 4),
        "theta_z_deg": round(float(chest_fin["theta_z_deg"]), 2),
        "z_err_m": round(z_err_fin, 4),
        "theta_z_err_deg": round(tz_err_fin, 2),
        "theta_z_target_deg": round(float(theta_z_tgt), 2),
        "q_err_rad": round(q_err_fin, 5),
        "target_trunk_q": [round(float(x), 5) for x in q_tgt.tolist()],
    }


def yield_trunk_to_planned_pose(
    ctx,
    world,
    *,
    chest_z_tgt: float,
    theta_z_tgt: float,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 1.5,
    trunk_max_step_rad: float = 0.08,
    trunk_timeout_s: float = 30.0,
    log_prefix: str = "move_point/trunk",
    object_z: float | None = None,
    reach_m: float | None = None,
    upward_require_theta_settle: bool = True,
    target_trunk_q=None,
    keep_ori_arm: str = "none",
) -> dict:
    """对准 chest_z / theta_z：upward 查表 + θz 闭环纯 q3（move_to_point 等）。

    move_to_object 请用 yield_trunk_deltaz_planned_pose（真机 deltaz 段）。
    """
    from behavior_interface.trunk_vertical_lift import (
        get_reverse_upward_combined_lut,
        plan_reverse_upward_trajectory_from_upward,
    )

    from behavior_interface.skills.reset_body import _normalize_keep_ori_arm

    requested_keep_ori_arms = _normalize_keep_ori_arm(keep_ori_arm)
    pitch_keep_ori: _PitchKeepOriController | None = None
    report: dict = {
        "ok": True,
        "skipped_upward": False,
        "skipped_pitch": False,
        "keep_ori_scope": "final_trunk_pitch_after_base_yaw",
        "keep_ori_requested_arm": sorted(requested_keep_ori_arms),
        "keep_ori_arm": [],
        "keep_ori_ok": True,
    }

    def _start_pitch_keep_ori() -> _PitchKeepOriController | None:
        nonlocal pitch_keep_ori
        if pitch_keep_ori is None and requested_keep_ori_arms:
            pitch_keep_ori = _PitchKeepOriController(
                ctx,
                world,
                keep_ori_arm,
                log_prefix=log_prefix,
            )
        return pitch_keep_ori

    def _attach_pitch_keep_ori_report() -> None:
        if pitch_keep_ori is None:
            return
        keep_report = pitch_keep_ori.report()
        report.update(keep_report)
        if not keep_report.get("keep_ori_ok", False):
            report["ok"] = False
            report["error"] = (
                report.get("error")
                or "俯仰过程姿态跟踪失败或 EEF 平移出现冲击"
            )

    def _pitch_keep_ori_ready(
        keeper: _PitchKeepOriController | None,
    ) -> bool:
        if keeper is None:
            return True
        requested = set(getattr(keeper, "requested_arms", requested_keep_ori_arms))
        active = set(getattr(keeper, "arms", requested))
        if requested == active:
            return True
        _attach_pitch_keep_ori_report()
        report["ok"] = False
        report["error"] = "keep_ori_arm 初始化失败；为避免无补偿俯仰已中止"
        return False

    theta_z_offline = float(theta_z_tgt)
    report["theta_z_offline_deg"] = round(theta_z_offline, 2)

    # 优先：规划已给出耦合正确的 target_trunk_q（同时满足 chest_z+θz）。
    # 直接驱动到该 q 并锁死，避免解耦三段（upward→俯仰→微调升降）互相打架导致塌低/失锁。
    if target_trunk_q is not None:
        try:
            q_tgt_arr = np.asarray(target_trunk_q, dtype=np.float64).reshape(4)
            valid_q = bool(np.all(np.isfinite(q_tgt_arr)))
        except Exception:
            valid_q = False
        if valid_q:
            keeper = _start_pitch_keep_ori()
            if not _pitch_keep_ori_ready(keeper):
                return report
            direct = yield from _yield_trunk_direct_to_q(
                ctx, world,
                q_target=q_tgt_arr,
                chest_z_tgt=float(chest_z_tgt),
                theta_z_tgt=float(theta_z_tgt),
                z_tol=float(z_tol),
                theta_z_tol_deg=float(theta_z_tol_deg),
                max_step_rad=float(trunk_max_step_rad),
                timeout_s=float(trunk_timeout_s),
                log_prefix=log_prefix,
                pitch_keep_ori=keeper,
            )
            report.update({
                "mode": "direct_target_trunk_q",
                "direct_trunk": direct,
                "theta_z_target_deg": direct.get("theta_z_target_deg"),
                "chest_z": direct.get("chest_z"),
                "theta_z_deg": direct.get("theta_z_deg"),
                "z_err_m": direct.get("z_err_m"),
                "theta_z_err_deg": direct.get("theta_z_err_deg"),
            })
            report["ok"] = bool(direct.get("ok"))
            if not report["ok"]:
                report["error"] = "直驱 target_trunk_q 后 chest_z/θz 未达容差"
            _attach_pitch_keep_ori_report()
            return report

    chest0 = world.chest_pose()
    curr_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    base_z = _base_link_z_world(world)
    q4 = float(curr_q[3])

    z_err = float(chest_z_tgt) - float(chest0["z"])
    tz_err = _norm_angle_deg(theta_z_tgt - chest0["theta_z_deg"])

    if abs(z_err) <= z_tol and abs(tz_err) <= theta_z_tol_deg:
        report["already_at_target"] = True
        return report

    if abs(z_err) > z_tol:
        lut = get_reverse_upward_combined_lut(base_z, q4=q4)
        if not lut.get("ok"):
            report["ok"] = False
            report["error"] = lut.get("error", "反向 upward 查表失败")
            return report
        z_upright = float(lut["z_upright_m"])
        upward_m = float(chest_z_tgt) - z_upright
        z_curr = float(chest0["z"])
        waypoints, vmeta = plan_reverse_upward_trajectory_from_upward(
            z_curr,
            upward_m,
            base_z,
            q4=q4,
        )
        if vmeta.get("direction") == "hold" and not waypoints:
            ctx.log(
                f"{log_prefix} upward 已在目标格点 z≈{vmeta.get('z_tgt_m')}m"
            )
            report["reverse_upward"] = vmeta
        elif not vmeta.get("ok") or not waypoints:
            report["ok"] = False
            report["error"] = vmeta.get("error", "反向 upward 规划失败")
            if vmeta.get("phase2_limited_by"):
                report["error"] = (
                    f"{report['error']}（phase2 触限: {vmeta.get('phase2_limited_by')}）"
                )
            report["reverse_upward"] = vmeta
            return report
        else:
            ctx.log(
                f"{log_prefix} upward: z {z_curr:.3f}→{vmeta.get('z_tgt_m'):.3f}m "
                f"(对齐 upward={vmeta.get('upward_snapped_m'):+.3f}m) "
                f"路点={len(waypoints)}"
            )
            exec_stats = yield from _yield_reverse_upward_trajectory(
                ctx, world, waypoints, vmeta,
                theta_z_deg=90.0,
                n_hold=_VERT_P2_HOLD,
                settle_final=bool(upward_require_theta_settle),
                require_theta_settle=bool(upward_require_theta_settle),
            )
            report["reverse_upward"] = vmeta
            report["vertical_exec"] = exec_stats
            if not exec_stats.get("reached", True):
                report["ok"] = False
                report["error"] = "upward 末点未到位"
                return report

        chest0 = world.chest_pose()
        curr_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    else:
        report["skipped_upward"] = True

    report["theta_z_target_deg"] = round(float(theta_z_tgt), 2)

    tz_err = _norm_angle_deg(theta_z_tgt - chest0["theta_z_deg"])
    if abs(tz_err) > theta_z_tol_deg:
        label = "仰头" if tz_err < 0 else "俯身"
        ctx.log(
            f"{log_prefix} pitch {label}(θz闭环/q3): "
            f"θz {chest0['theta_z_deg']:.1f}°→{theta_z_tgt:.1f}° "
            f"Δ={tz_err:+.1f}°"
        )
        keeper = _start_pitch_keep_ori()
        if not _pitch_keep_ori_ready(keeper):
            return report
        pitch_stats = yield from _yield_trunk_theta_z_pitch_closed_loop(
            ctx, world,
            theta_z_tgt=float(theta_z_tgt),
            max_step_rad=trunk_max_step_rad,
            theta_z_tol_deg=theta_z_tol_deg,
            timeout_s=trunk_timeout_s,
            log_prefix=log_prefix,
            pitch_keep_ori=keeper,
        )
        report["pitch_q3"] = pitch_stats
        if not pitch_stats.get("reached"):
            report["ok"] = False
            report["error"] = "θz 俯仰未到位或超时"
            _attach_pitch_keep_ori_report()
            return report
    else:
        report["skipped_pitch"] = True

    chest_fin = world.chest_pose()
    z_err_fin = float(chest_z_tgt) - float(chest_fin["z"])
    if abs(z_err_fin) > z_tol:
        ctx.log(
            f"{log_prefix} pitch 后 chest_z 漂移 "
            f"当前={chest_fin['z']:.3f} 目标={chest_z_tgt:.3f} "
            f"err={z_err_fin:+.3f}m；保持 θz={theta_z_tgt:.1f}° 微调升降"
        )
        n_trim = yield from _yield_phase2_q1q2_closed_loop(
            ctx, world,
            z_target=float(chest_z_tgt),
            theta_z_hold_deg=float(theta_z_tgt),
            descending=(float(chest_fin["z"]) > float(chest_z_tgt)),
            z_tol=float(z_tol),
            theta_z_tol_deg=float(theta_z_tol_deg),
            max_steps=100,
            pitch_keep_ori=pitch_keep_ori,
        )
        report["z_trim_after_pitch_steps"] = int(n_trim)
        chest_fin = world.chest_pose()

    report["chest_z"] = round(chest_fin["z"], 4)
    report["theta_z_deg"] = round(chest_fin["theta_z_deg"], 2)
    report["z_err_m"] = round(float(chest_fin["z"]) - float(chest_z_tgt), 4)
    report["theta_z_err_deg"] = round(
        _norm_angle_deg(theta_z_tgt - chest_fin["theta_z_deg"]), 2,
    )
    _attach_pitch_keep_ori_report()
    return report


def _yield_cached_vertical_trajectory(
    ctx,
    world,
    waypoints: List[np.ndarray],
    vmeta: dict,
    *,
    z_tgt: float,
    z_tol: float,
    theta_z_tol_deg: float,
) -> dict:
    """开环执行 plan_vertical_lift_waypoints_v2_cached 的完整轨迹。"""
    p1_n, p2_n, p3_n = _phase_waypoint_counts(vmeta, len(waypoints))
    wp1 = _subsample_trunk_waypoints(waypoints[:p1_n], _VERT_P1_MAX_WAYPOINTS) if p1_n else []
    wp2_raw = waypoints[p1_n:p1_n + p2_n] if p2_n > 0 else []
    wp3_raw = waypoints[p1_n + p2_n:] if p3_n > 0 else []
    wp2 = _subsample_trunk_waypoints(wp2_raw, _VERT_P2_MAX_WAYPOINTS) if wp2_raw else []
    wp3 = _subsample_trunk_waypoints(wp3_raw, _VERT_P2_MAX_WAYPOINTS) if wp3_raw else []

    stats = {
        "exec_mode": "cached_openloop",
        "cached": bool(vmeta.get("cached")),
        "p1_planned": p1_n,
        "p2_planned": p2_n,
        "p3_planned": p3_n,
        "p1_exec": len(wp1),
        "p2_exec": len(wp2),
        "p3_exec": len(wp3),
    }
    ctx.log(
        f"move_in_robot_coord 开环轨迹: "
        f"cached={stats['cached']} "
        f"p1 {stats['p1_planned']}→{stats['p1_exec']} "
        f"p2 {stats['p2_planned']}→{stats['p2_exec']} "
        f"p3 {stats['p3_planned']}→{stats['p3_exec']} "
        f"(阶段2 |dq2|=|dq1| 阶段3 |dq2|=|dq3|)"
    )

    for i, q_wp in enumerate(wp1):
        yield from _yield_trunk_vertical_waypoint(
            ctx, world, q_wp,
            phase_tag="p1",
            theta_z_hold_deg=None,
            theta_z_tol_deg=theta_z_tol_deg,
            n_hold=_VERT_P1_HOLD,
        )
        if i % 4 == 0 or i == len(wp1) - 1:
            chest = world.chest_pose()
            ctx.set_status(
                f"vertical p1 {i+1}/{len(wp1)} "
                f"z={chest['z']:.3f} θz={chest['theta_z_deg']:.1f}°"
            )

    if wp2:
        p2_meta = next(
            (p for p in vmeta.get("phases", [])
             if p.get("phase") == "q1q2_locked_q3"),
            {},
        )
        ctx.log(
            f"move_in_robot_coord 阶段2 开环 |dq2|=|dq1| dq3=0 "
            f"dq1={p2_meta.get('dq1_rad', '?')} dq2={p2_meta.get('dq2_rad', '?')} "
            f"路点={len(wp2)}"
        )
        yield from _yield_trunk_waypoints_openloop(
            ctx, world, wp2, phase_tag="vertical p2", n_hold=_VERT_P2_HOLD,
        )

    if wp3:
        p3_meta = next(
            (p for p in vmeta.get("phases", [])
             if p.get("phase") == "q2q3_locked_q1"),
            {},
        )
        ctx.log(
            f"move_in_robot_coord 阶段3 开环 |dq2|=|dq3| dq1=0 "
            f"dq2={p3_meta.get('dq2_rad', '?')} dq3={p3_meta.get('dq3_rad', '?')} "
            f"路点={len(wp3)}"
        )
        yield from _yield_trunk_waypoints_openloop(
            ctx, world, wp3, phase_tag="vertical p3", n_hold=_VERT_P2_HOLD,
        )

    chest = world.chest_pose()
    stats["chest_z"] = round(chest["z"], 4)
    stats["z_err"] = round(chest["z"] - float(z_tgt), 4)
    return stats


def _yield_trunk_vertical_waypoint(
    ctx,
    world,
    q_cmd: np.ndarray,
    *,
    phase_tag: str,
    theta_z_hold_deg: Optional[float],
    theta_z_tol_deg: float = 5.0,
    n_hold: int = 4,
):
    """下发 trunk 路点；阶段2 闭环读 chest θz，必要时只调 q3 拉回垂直。"""
    from behavior_interface.trunk_vertical_lift import solve_q3_for_theta_z_holding_q12

    q_hold = np.asarray(q_cmd, dtype=np.float64).reshape(4)
    for _ in range(max(1, int(n_hold))):
        yield world.make_action_trunk_locked(q_hold.tolist())

    if theta_z_hold_deg is None:
        return

    for attempt in range(4):
        chest = world.chest_pose()
        tz_err = float(chest["theta_z_deg"]) - float(theta_z_hold_deg)
        if abs(tz_err) <= float(theta_z_tol_deg):
            return

        q_act = world.trunk_qpos()
        nq3 = solve_q3_for_theta_z_holding_q12(
            float(theta_z_hold_deg),
            float(q_act[0]),
            float(q_act[1]),
            prefer_q3=float(q_act[2]),
        )
        if nq3 is None:
            ctx.log(
                f"[vertical] θz 守护 #{attempt+1} 无解: "
                f"实测 {chest['theta_z_deg']:.1f}° 目标 {float(theta_z_hold_deg):.1f}°"
            )
            return

        q_fix = np.array([q_act[0], q_act[1], nq3, q_act[3]], dtype=np.float64)
        for _ in range(6):
            yield world.make_action_trunk_locked(q_fix.tolist())

    chest2 = world.chest_pose()
    if abs(float(chest2["theta_z_deg"]) - float(theta_z_hold_deg)) > float(theta_z_tol_deg) + 3.0:
        ctx.log(
            f"[vertical] 警告: θz={chest2['theta_z_deg']:.1f}° "
            f"偏离垂直 {float(theta_z_hold_deg):.1f}°"
        )


def _yield_phase2_q2_backward_exec(
    ctx,
    world,
    *,
    z_target: float,
    theta_z_hold_deg: float = 90.0,
    dq2_step_rad: float = -0.012,
    bias_q3_fold: bool = False,
    trunk_step_rad: float = 0.05,
    z_tol: float = 0.03,
    max_steps: int = 100,
    n_hold: int = _VERT_P2_HOLD,
) -> int:
    """阶段2 执行：q2 向后（减小），每步反解 q1/q3 保胸廓垂直。"""
    from behavior_interface.trunk_vertical_lift import (
        R1PRO_Q_LIMITS,
        solve_q1_q3_for_theta_z_holding_q2,
    )

    lo2, hi2 = R1PRO_Q_LIMITS[1]
    dq2 = -abs(float(dq2_step_rad))
    step_lim = abs(float(trunk_step_rad))
    n_done = 0

    for step_i in range(max_steps):
        chest = world.chest_pose()
        z_now = float(chest["z"])
        if z_now <= float(z_target) + float(z_tol):
            break

        q_act = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        nq2 = float(q_act[1]) + dq2
        if nq2 < lo2 + 1e-4:
            ctx.log(f"[vertical] 阶段2 q2 后摆触限 q2={nq2:.3f}")
            break

        nq2_cmd = float(q_act[1]) + max(
            -step_lim, min(step_lim, nq2 - float(q_act[1])),
        )
        sol = solve_q1_q3_for_theta_z_holding_q2(
            float(theta_z_hold_deg), nq2_cmd,
            prefer_q1=float(q_act[0]),
            prefer_q3=float(q_act[2]),
            bias_q3_fold=bias_q3_fold,
        )
        if sol is None:
            ctx.log(f"[vertical] 阶段2 q2={nq2_cmd:.3f} 无 q1/q3 保垂直解")
            break
        nq1, nq3 = sol
        q_cmd = np.array([nq1, nq2_cmd, nq3, q_act[3]], dtype=np.float64)
        for _ in range(max(1, int(n_hold))):
            yield world.make_action_trunk_locked(q_cmd.tolist())
        n_done += 1
        if step_i % 2 == 0:
            chest = world.chest_pose()
            q_now = world.trunk_qpos()
            ctx.set_status(
                f"vertical p2q2- {step_i+1} z={chest['z']:.3f} "
                f"q2={q_now[1]:+.3f} θz={chest['theta_z_deg']:.1f}°"
            )
    return n_done


def _yield_phase2_q1q2_locked_q3(
    ctx,
    world,
    *,
    z_target: float,
    dq_sign_hint: float = 1.0,
    dq_step_rad: float = 0.015,
    trunk_step_rad: float = 0.05,
    z_tol: float = 0.03,
    max_steps: int = 80,
    n_hold: int = _VERT_P2_HOLD,
    pitch_keep_ori: _PitchKeepOriController | None = None,
) -> int:
    """阶段2 补降：dq1=dq2 同向，**锁定 q3**（q3 饱和后无法再用 q3 保 θz）。"""
    from behavior_interface.trunk_vertical_lift import R1PRO_Q_LIMITS

    lo1, hi1 = R1PRO_Q_LIMITS[0]
    lo2, hi2 = R1PRO_Q_LIMITS[1]
    sign = 1.0 if float(dq_sign_hint) >= 0 else -1.0
    dq = sign * abs(float(dq_step_rad))
    n_done = 0

    for step_i in range(max_steps):
        chest = world.chest_pose()
        z_now = float(chest["z"])
        if z_now <= float(z_target) + float(z_tol):
            break

        q_act = world.trunk_qpos()
        tgt1 = float(q_act[0]) + dq
        tgt2 = float(q_act[1]) + dq
        if not (lo1 <= tgt1 <= hi1 and lo2 <= tgt2 <= hi2):
            ctx.log(
                f"[vertical] 阶段2 q1=q2 触限 @step{step_i}: "
                f"q1→{tgt1:.3f} q2→{tgt2:.3f}"
            )
            break

        step_lim = abs(float(trunk_step_rad))
        q_cmd = q_act.copy()
        q_cmd[0] = float(q_act[0]) + max(-step_lim, min(step_lim, tgt1 - float(q_act[0])))
        q_cmd[1] = float(q_act[1]) + max(-step_lim, min(step_lim, tgt2 - float(q_act[1])))
        # q3/q4 锁定：与阶段1 末段同向继续弯折
        for _ in range(max(1, int(n_hold))):
            yield from _yield_pitch_keep_ori_action(
                world, q_cmd.tolist(), pitch_keep_ori,
            )
        n_done += 1
        if step_i % 2 == 0:
            chest = world.chest_pose()
            q_now = world.trunk_qpos()
            ctx.set_status(
                f"vertical p2锁q3 {step_i+1} z={chest['z']:.3f} "
                f"q1={q_now[0]:+.3f} q2={q_now[1]:+.3f}"
            )
    return n_done


def _yield_phase2_q1q2_closed_loop(
    ctx,
    world,
    *,
    z_target: float,
    theta_z_hold_deg: float = 90.0,
    dq_sign_hint: float = 1.0,
    dq_step_rad: float = 0.008,
    trunk_step_rad: float = 0.028,
    theta_z_tol_deg: float = 4.0,
    theta_z_abort_deg: float = 15.0,
    max_steps: int = 150,
    z_tol: float = 0.03,
    descending: bool = True,
    pitch_keep_ori: _PitchKeepOriController | None = None,
) -> int:
    """阶段2 闭环：dq1=dq2，每步反解 q3 保持 θz≈90°（胸廓垂直），小步逼近防趴下。"""
    from behavior_interface.trunk_vertical_lift import (
        R1PRO_Q_LIMITS,
        solve_q3_for_theta_z_holding_q12,
    )

    lo1, hi1 = R1PRO_Q_LIMITS[0]
    lo2, hi2 = R1PRO_Q_LIMITS[1]
    lo3, hi3 = R1PRO_Q_LIMITS[2]
    sign = 1.0 if float(dq_sign_hint) >= 0 else -1.0
    dq = sign * abs(float(dq_step_rad))
    if not descending:
        dq = -dq
    n_done = 0
    theta_z_hold_deg = float(theta_z_hold_deg)

    for step_i in range(max_steps):
        chest = world.chest_pose()
        z_now = float(chest["z"])
        tz_now = float(chest["theta_z_deg"])
        if descending and z_now <= float(z_target) + float(z_tol):
            break
        if not descending and z_now >= float(z_target) - float(z_tol):
            break
        if descending and (tz_now - theta_z_hold_deg) > float(theta_z_abort_deg):
            ctx.log(
                f"[vertical] 阶段2中止 @step{step_i}: "
                f"θz={tz_now:.1f}° > hold+{theta_z_abort_deg}°，尝试回退半步"
            )
            q_act = world.trunk_qpos()
            q_back = np.array([
                float(q_act[0]) - 0.5 * dq,
                float(q_act[1]) - 0.5 * dq,
                float(q_act[2]),
                float(q_act[3]),
            ], dtype=np.float64)
            nq3b = solve_q3_for_theta_z_holding_q12(
                theta_z_hold_deg, q_back[0], q_back[1], prefer_q3=q_back[2],
            )
            if nq3b is not None:
                q_back[2] = nq3b
                for _ in range(20):
                    yield from _yield_pitch_keep_ori_action(
                        world, q_back.tolist(), pitch_keep_ori,
                    )
            break

        q_act = world.trunk_qpos()
        tgt1 = float(q_act[0]) + dq
        tgt2 = float(q_act[1]) + dq
        if not (lo1 <= tgt1 <= hi1 and lo2 <= tgt2 <= hi2):
            break
        nq3 = solve_q3_for_theta_z_holding_q12(
            theta_z_hold_deg, tgt1, tgt2, prefer_q3=float(q_act[2]),
        )
        if nq3 is None or not (lo3 <= nq3 <= hi3):
            ctx.log(
                f"[vertical] 阶段2闭环 step{step_i} q3 无解/越限 "
                f"(θz_hold={theta_z_hold_deg:.1f}°)，改锁定 q3 仅动 q1=q2"
            )
            n_locked = yield from _yield_phase2_q1q2_locked_q3(
                ctx, world,
                z_target=z_target,
                dq_sign_hint=dq_sign_hint,
                dq_step_rad=dq_step_rad,
                trunk_step_rad=trunk_step_rad,
                z_tol=z_tol,
                max_steps=max(20, max_steps - step_i),
                pitch_keep_ori=pitch_keep_ori,
            )
            return n_done + n_locked

        q_tgt = np.array([tgt1, tgt2, nq3, q_act[3]], dtype=np.float64)
        step_lim = abs(float(trunk_step_rad))
        q_cmd = q_act.copy()
        for j in range(3):
            q_cmd[j] = float(q_act[j]) + max(
                -step_lim, min(step_lim, float(q_tgt[j]) - float(q_act[j])),
            )

        for _ in range(24):
            yield from _yield_pitch_keep_ori_action(
                world, q_cmd.tolist(), pitch_keep_ori,
            )

        for _ in range(4):
            chest = world.chest_pose()
            if abs(float(chest["theta_z_deg"]) - theta_z_hold_deg) <= float(theta_z_tol_deg):
                break
            q_fix = world.trunk_qpos()
            nq3_fix = solve_q3_for_theta_z_holding_q12(
                theta_z_hold_deg,
                float(q_fix[0]),
                float(q_fix[1]),
                prefer_q3=float(q_fix[2]),
            )
            if nq3_fix is None:
                break
            q_fix = np.array(
                [q_fix[0], q_fix[1], nq3_fix, q_fix[3]], dtype=np.float64,
            )
            for _ in range(12):
                yield from _yield_pitch_keep_ori_action(
                    world, q_fix.tolist(), pitch_keep_ori,
                )

        n_done += 1
        if step_i % 2 == 0:
            chest = world.chest_pose()
            ctx.set_status(
                f"vertical p2闭环 {step_i+1} "
                f"z={chest['z']:.3f} θz={chest['theta_z_deg']:.1f}°"
            )
    return n_done


# ─────────────────────────────────────────────────────────────────────────────
# move_in_robot_coord: 机体系 forward / spin / pitch / upward（两阶段垂直升降）
# ─────────────────────────────────────────────────────────────────────────────

@register_skill(
    "diag_vertical_lift",
    description=(
        "正弦定理垂直升降诊断：扫描 C 可达范围；可选执行 upward 并逐点记录 "
        "theta_z / C / 胸口 z，验证上半身是否保持垂直。"
    ),
)
def diag_vertical_lift(
    ctx,
    upward: float = -0.1,
    run_motion: bool = True,
    n_steps: int = 15,
):
    """离线扫描 + 仿真实测垂直升降。"""
    from behavior_interface.trunk_vertical_lift import (
        estimate_chest_theta_z_deg,
        estimate_chest_z_world,
        plan_vertical_lift_waypoints,
        scan_sine_law_manifold,
        vertical_C_from_trunk_q,
    )

    world = ctx.world
    mf = scan_sine_law_manifold(refresh=True)
    base_z = _base_link_z_world(world)
    chest0 = world.chest_pose()
    q0 = world.trunk_qpos()
    C0 = vertical_C_from_trunk_q(q0)

    report: dict = {
        "ok": True,
        "manifold": mf,
        "start": {
            "chest_z": round(chest0["z"], 4),
            "theta_z_deg": round(chest0["theta_z_deg"], 2),
            "trunk_q": [round(float(x), 4) for x in q0[:3]],
            "C_m": round(C0, 4),
        },
    }
    ctx.log(
        f"[diag_vertical_lift] C 可达 [{mf.get('C_min_m')}, {mf.get('C_max_m')}] m "
        f"胸口 z(base) [{mf.get('z_chest_min_m')}, {mf.get('z_chest_max_m')}] m "
        f"当前 C={C0:.3f} theta_z={chest0['theta_z_deg']:.1f}°"
    )

    if abs(float(upward)) < 1e-6:
        ctx.set_result(report)
        yield world.hold_action()
        return

    z_tgt = float(chest0["z"]) + float(upward)
    waypoints, vmeta = plan_vertical_lift_waypoints(
        q0, float(chest0["z"]), z_tgt, base_link_z=base_z, n_steps=int(n_steps),
    )
    report["plan"] = vmeta
    if not vmeta.get("ok") or not waypoints:
        report["ok"] = False
        report["error"] = vmeta.get("error", "规划失败")
        ctx.log(f"[diag_vertical_lift] 规划失败: {report['error']}")
        ctx.set_result(report)
        yield world.hold_action()
        return

    trace: List[dict] = []
    if run_motion:
        for i, q_wp in enumerate(waypoints):
            q_hold = np.asarray(q_wp, dtype=np.float64).reshape(4)
            for _ in range(3):
                yield world.make_action_trunk_locked(q_hold.tolist())
            chest = world.chest_pose()
            q_act = world.trunk_qpos()
            C_act = vertical_C_from_trunk_q(q_act)
            tz_model = estimate_chest_theta_z_deg(q_act[0], q_act[1], q_act[2])
            trace.append({
                "i": i,
                "C_cmd": round(vertical_C_from_trunk_q(q_hold), 4),
                "C_act": round(C_act, 4),
                "theta_z_sim_deg": round(chest["theta_z_deg"], 2),
                "theta_z_model_deg": round(tz_model, 2),
                "chest_z_sim": round(chest["z"], 4),
                "q_act": [round(float(x), 4) for x in q_act[:3]],
            })
            if i % 3 == 0 or i == len(waypoints) - 1:
                ctx.set_status(
                    f"diag_vertical_lift {i+1}/{len(waypoints)} "
                    f"θz={chest['theta_z_deg']:.1f}° C={C_act:.3f}"
                )

    chest1 = world.chest_pose()
    tz_spread = max(t["theta_z_sim_deg"] for t in trace) - min(t["theta_z_sim_deg"] for t in trace) if trace else 0.0
    report["trace"] = trace
    report["end"] = {
        "chest_z": round(chest1["z"], 4),
        "theta_z_deg": round(chest1["theta_z_deg"], 2),
        "trunk_q": [round(float(x), 4) for x in world.trunk_qpos()[:3]],
        "C_m": round(vertical_C_from_trunk_q(world.trunk_qpos()), 4),
        "theta_z_spread_deg": round(tz_spread, 2),
        "vertical_ok": tz_spread <= 3.0,
    }
    ctx.log(
        f"[diag_vertical_lift] 完成 z {report['start']['chest_z']}→{report['end']['chest_z']} "
        f"θz spread={tz_spread:.2f}° vertical_ok={report['end']['vertical_ok']}"
    )
    ctx.set_result(report)
    yield world.hold_action()


@register_skill(
    "move_in_robot_coord",
    description=(
        "机体系增量：forward/translation 同帧合成为二维直线移动，随后 spin；"
        "upward 为反向垂直查表（z≥0.73m 正弦流形 phase1；"
        "z<0.73m 仅 θz=90° phase2）；pitch 为纯 q3 俯仰（锁定 q1/q2）；"
        "输入 upward 对齐最近 1cm 格点。"
    ),
)
def move_in_robot_coord(
    ctx,
    forward: float = 0.0,
    translation: float = 0.0,
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
    nav_guard: bool = True,
    extra_inflate: float = 0.04,
    observation_only_chassis: bool = False,
):
    """机体系移动：upward 沿 phase1/phase2 合并查表从当前高度走到目标 1cm 格点。"""
    world = ctx.world
    from behavior_interface.skills.eef import _freeze_world_limb_pins

    if observation_only_chassis:
        # Public adjust_chassis enters here before any legacy chest/global-pose
        # read. pitch/upward are separate public tools and are intentionally not
        # accepted by this evaluator-compliant chassis-only branch.
        if abs(float(pitch)) > 1e-9 or abs(float(upward)) > 1e-9:
            ctx.set_result({
                "ok": False,
                "error": "observation-only chassis branch does not accept pitch/upward",
                "observation_only": True,
            })
            yield world.hold_action()
            return
        yield from _yield_adjust_chassis_observation_only(
            ctx,
            forward=float(forward),
            spin=float(spin),
            vmax=float(vmax),
            wmax=float(wmax),
            timeout_s=float(timeout_s),
            translation=float(translation),
        )
        return

    _freeze_world_limb_pins(world)

    chest0 = world.chest_pose()
    base_z = _base_link_z_world(world)
    curr_q = world.trunk_qpos()
    pose_start = world.robot_pose()
    trunk_q_start = np.asarray(curr_q, dtype=np.float64).reshape(4)
    q4 = float(curr_q[3])
    report: dict = {
        "ok": True,
        "forward": float(forward),
        "spin": float(spin),
        "pitch": float(pitch),
        "upward": float(upward),
        "build": "reverse_upward_relative_lut_reset_grasp_only_v1",
        "robot_yaw_start_deg": round(math.degrees(float(pose_start.yaw)), 2),
        "trunk_q_start": [round(float(x), 4) for x in trunk_q_start],
    }

    def _abort(phase: str, reason: str):
        report["ok"] = False
        report["error"] = reason
        report["phase"] = phase
        ctx.log(f"move_in_robot_coord 不可达 [{phase}]: {reason}")
        ctx.set_result(dict(report))
        yield world.hold_action()

    if abs(upward) > 1e-6:
        up_report = yield from yield_reverse_upward_relative_lut(
            ctx,
            world,
            upward_delta_m=float(upward),
            theta_z_deg=90.0,
            z_tol=float(z_tol),
            log_prefix="move_robot/upward",
        )
        report["upward_exec"] = up_report
        report["reverse_upward"] = up_report.get("reverse_upward")
        report["vertical_exec"] = up_report.get("vertical_exec")
        report["relative_lut"] = up_report.get("relative_lut")
        if not up_report.get("ok", False):
            reason = up_report.get("error", "反向 upward 相对 LUT 执行失败")
            vmeta = up_report.get("reverse_upward") or {}
            if vmeta.get("phase2_limited_by"):
                reason = (
                    f"{reason}（phase2 触限: {vmeta.get('phase2_limited_by')}，"
                    f"z_min={vmeta.get('z_min_m')}m）"
                )
            yield from _abort("upward", reason)
            return

        chest0 = world.chest_pose()
        curr_q = world.trunk_qpos()
        base_z = _base_link_z_world(world)
        report["chest_z"] = round(chest0["z"], 4)
        report["theta_z_deg"] = round(chest0["theta_z_deg"], 2)
        vmeta = up_report.get("reverse_upward") or {}
        z_tgt = float(vmeta.get("z_tgt_m", chest0["z"]))
        if abs(chest0["z"] - z_tgt) > z_tol + 0.04:
            ctx.log(
                f"move_in_robot_coord upward 警告: 目标 z={z_tgt:.3f} "
                f"实际 z={chest0['z']:.3f} err={chest0['z']-z_tgt:+.3f}m"
            )

    if abs(pitch) > 1e-6:
        chest_now = world.chest_pose()
        ok_p, why, q3_tgt = _feasible_trunk_q3_delta(
            curr_q, float(pitch),
            curr_theta_z_deg=float(chest_now.get("theta_z_deg", 90.0)),
        )
        if not ok_p:
            yield from _abort("pitch", why)
            return
        label = "仰头" if float(pitch) > 0 else "俯身"
        tz_pred = None
        try:
            from behavior_interface.trunk_vertical_lift import fk_torso_link4_theta_z_deg
            tz_pred = float(fk_torso_link4_theta_z_deg(
                float(curr_q[0]), float(curr_q[1]), float(q3_tgt)))
        except Exception:
            pass
        ctx.log(
            f"move_in_robot_coord pitch {label}(纯q3): "
            f"q3 {curr_q[2]:+.3f}→{q3_tgt:+.3f}rad "
            f"({math.degrees(curr_q[2]):+.1f}°→{math.degrees(q3_tgt):+.1f}°) "
            f"θz {float(chest_now.get('theta_z_deg', 0)):.1f}°"
            + (f"→{tz_pred:.1f}°" if tz_pred is not None else "")
            + f" q1/q2 锁定"
        )
        pitch_stats = yield from _yield_trunk_q3_pitch(
            ctx, world, float(pitch),
            max_step_rad=trunk_max_step_rad,
            timeout_s=trunk_timeout_s,
            log_prefix="move_robot/pitch",
        )
        report["pitch_q3"] = pitch_stats
        if not pitch_stats.get("reached"):
            yield from _abort("pitch", "q3 俯仰未到位或超时")
            return
        chest0 = world.chest_pose()
        curr_q = world.trunk_qpos()
        report["theta_z_deg"] = round(chest0["theta_z_deg"], 2)
        report["chest_z"] = round(chest0["z"], 4)

    if math.hypot(float(forward), float(translation)) > 1e-6:
        if nav_guard:
            _refresh_nav_scene_graph(ctx, "move_robot")
        limb_hold = {}
        payload_hold = {}
        try:
            from behavior_interface.skills.move_to_object_v2 import (
                _fast_base_limb_drift,
                _payload_end_drift,
                _rounded_payload_drift,
                _snapshot_assisted_payload_hold,
                _snapshot_fast_base_limb_hold,
            )

            limb_hold = _snapshot_fast_base_limb_hold(world)
            payload_hold = _snapshot_assisted_payload_hold(world)
            payload_names = {
                arm: spec.get("name") for arm, spec in payload_hold.items()
            }
            ctx.log(
                "move_robot body action-only lock "
                f"arms={[name.removeprefix('arm_') for name in ('arm_left', 'arm_right') if name in limb_hold]} "
                f"payloads={payload_names}"
            )
        except Exception as exc:
            ctx.log(f"move_robot body WARN action-lock snapshot failed: {exc}")
        try:
            fwd_ok = yield from _drive_body_forward(
                ctx, float(forward),
                translation_m=float(translation),
                vmax=vmax,
                timeout_s=timeout_s,
                log_prefix="move_robot",
                nav_guard=nav_guard,
                extra_inflate=extra_inflate,
            )
        finally:
            try:
                limb_end = _fast_base_limb_drift(world, limb_hold)
                payload_end = _payload_end_drift(world, payload_hold)
                ctx.log(
                    "move_robot body action-only summary "
                    f"limb_end={{{', '.join(f'{name}: {float(err):.5f}' for name, err in limb_end.items())}}} "
                    f"payload_end={_rounded_payload_drift(payload_end)}"
                )
            except Exception as exc:
                ctx.log(f"move_robot body WARN action-only summary failed: {exc}")
        report["forward_ok"] = bool(fwd_ok)
        report["translation_ok"] = bool(fwd_ok)
        report["linear_control"] = "simultaneous_robot_frame_xy"
        report["ground_guard"] = dict(
            getattr(ctx, "_body_ground_guard_report", None) or {}
        )
        report["forward_obstacle_limited"] = bool(
            report["ground_guard"].get("obstacle_limited", False)
        )
        report["forward_actual_m"] = report["ground_guard"].get(
            "safe_forward_m"
        )
        report["translation_actual_m"] = report["ground_guard"].get(
            "safe_translation_m"
        )
        if not fwd_ok:
            forward_failure = _nav_seg_fail(ctx)
            report["forward_failure_reason"] = forward_failure
            if forward_failure == "stuck":
                error = "stuck或者碰撞"
            elif forward_failure == "ground_recovery_timeout":
                error = "离地急停后未能稳定落地"
            else:
                error = "底盘二维移动未到位或超时"
            failure_phase = (
                "translation" if abs(float(forward)) <= 1e-6 else "chassis"
            )
            yield from _abort(failure_phase, error)
            return

    if abs(spin) > 1e-6:
        pose_now = world.robot_pose()
        target_yaw = _norm_angle(float(pose_now.yaw) + math.radians(float(spin)))
        spin_ok = False
        try:
            from behavior_interface.skills.move_to_object_v2 import (
                _yield_base_xy_yaw_controller,
            )

            spin_stats = yield from _yield_base_xy_yaw_controller(
                ctx,
                world,
                bx=float(pose_now.pos[0]),
                by=float(pose_now.pos[1]),
                theta_x_deg=math.degrees(float(target_yaw)),
                timeout_s=float(timeout_s),
                log_tag="move_robot/spin",
            )
            spin_stats = dict(spin_stats or {})
            spin_stats["ok"] = bool(spin_stats.get("xy_ok")) and bool(
                spin_stats.get("yaw_ok")
            )
            spin_stats["mode"] = "base_velocity_action_only"
            spin_stats["action_only"] = True
            report["spin_exec"] = spin_stats
            spin_ok = bool(spin_stats.get("ok", False))
        except Exception as e:
            report["spin_controller_error"] = str(e)
            ctx.log(f"move_in_robot_coord spin 控制器异常，回退 velocity rotate: {e}")
            spin_ok = yield from _rotate_to_target_yaw(
                ctx, target_yaw, yaw_tol_deg, wmax, k_ang, timeout_s, "move_robot",
            )
        if not spin_ok:
            pose_fail = world.robot_pose()
            yaw_fail = float(pose_fail.yaw)
            report["robot_yaw_target_deg"] = round(math.degrees(float(target_yaw)), 2)
            report["robot_yaw_fail_deg"] = round(math.degrees(yaw_fail), 2)
            report["robot_yaw_err_deg"] = round(
                math.degrees(_norm_angle(float(target_yaw) - yaw_fail)), 2
            )
            spin_reason = str(
                (report.get("spin_exec") or {}).get("reason")
                or (report.get("spin_exec") or {}).get("status")
                or ""
            )
            if spin_reason == "stuck_or_collision":
                yield from _abort("spin", "stuck或者碰撞")
            else:
                yield from _abort("spin", "原地转向未到位或超时")
            return

    yield from yield_move_settle(world)
    chest_fin = world.chest_pose()
    pose_fin = world.robot_pose()
    trunk_q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    report["chest_z"] = round(chest_fin["z"], 4)
    report["theta_z_deg"] = round(chest_fin["theta_z_deg"], 2)
    report["robot_yaw_end_deg"] = round(math.degrees(float(pose_fin.yaw)), 2)
    report["robot_yaw_delta_deg"] = round(
        math.degrees(_norm_angle(float(pose_fin.yaw) - float(pose_start.yaw))), 2
    )
    report["trunk_q_end"] = [round(float(x), 4) for x in trunk_q_fin]
    report["trunk_q_delta"] = [
        round(float(trunk_q_fin[i] - trunk_q_start[i]), 4) for i in range(4)
    ]
    report["trunk_q_delta_abs_max_rad"] = round(
        float(np.max(np.abs(trunk_q_fin - trunk_q_start))), 4
    )
    ctx.set_result(report)


@register_skill(
    "diag_vertical_lift_v2",
    description=(
        "两阶段垂直升降诊断：测量阶段1+2 胸口 z 可达范围；"
        "可选执行最大下降再回升，记录各阶段 trace。"
    ),
)
def diag_vertical_lift_v2(
    ctx,
    run_motion: bool = True,
    test_upward: float = -0.55,
):
    """扫描 + 仿真实测两阶段垂直升降范围。"""
    from behavior_interface.trunk_vertical_lift import (
        CHEST_Z_MIN_M,
        estimate_chest_theta_z_deg,
        estimate_chest_z_world,
        measure_two_phase_vertical_range,
        plan_vertical_lift_waypoints_v2,
        vertical_C_from_trunk_q,
    )

    world = ctx.world
    base_z = _base_link_z_world(world)
    chest0 = world.chest_pose()
    q0 = world.trunk_qpos()

    range_report = measure_two_phase_vertical_range(q0, base_link_z=base_z)
    report: dict = {
        "ok": True,
        "range": range_report,
        "start": {
            "chest_z": round(chest0["z"], 4),
            "theta_z_deg": round(chest0["theta_z_deg"], 2),
            "trunk_q": [round(float(x), 4) for x in q0[:3]],
            "C_m": round(vertical_C_from_trunk_q(q0), 4),
        },
    }
    p1 = range_report.get("phase1_sine_manifold", {})
    p12 = range_report.get("phase1_plus_phase2", {})
    ctx.log(
        f"[diag_vertical_lift_v2] 阶段1 z∈{p1.get('z_chest_world_m')} "
        f"阶段1+2 最低 z={p12.get('z_chest_min_world_m')}m "
        f"直立总降幅={p12.get('total_drop_from_upright_m')}m"
    )

    if abs(float(test_upward)) < 1e-6 or not run_motion:
        ctx.set_result(report)
        yield world.hold_action()
        return

    z_tgt = max(
        float(range_report.get("phase1_plus_phase2", {}).get("z_chest_min_world_m", CHEST_Z_MIN_M)),
        float(chest0["z"]) + float(test_upward),
    )
    if z_tgt >= float(chest0["z"]) - 0.005:
        z_tgt = float(range_report.get("phase1_plus_phase2", {}).get(
            "z_chest_min_world_m", CHEST_Z_MIN_M,
        ))
        ctx.log(
            f"[diag_vertical_lift_v2] test_upward 裁剪后非下降，改为测最大下降 z→{z_tgt:.3f}m"
        )
    waypoints, vmeta = plan_vertical_lift_waypoints_v2(
        q0, float(chest0["z"]), z_tgt, base_link_z=base_z,
    )
    report["plan_down"] = vmeta
    if not vmeta.get("ok") or not waypoints:
        report["ok"] = False
        report["error"] = vmeta.get("error", "下降规划失败")
        ctx.set_result(report)
        yield world.hold_action()
        return

    trace: List[dict] = []
    p1_n = 0
    p2_meta: dict = {}
    for ph in vmeta.get("phases", []):
        if ph.get("phase") == "sine_manifold":
            p1_n = int(ph.get("n_waypoints", 0))
        if ph.get("phase") == "q1q2_coupled_theta_guard":
            p2_meta = ph

    wp1 = waypoints[:p1_n] if p1_n > 0 else []
    for i, q_wp in enumerate(wp1):
        yield from _yield_trunk_vertical_waypoint(
            ctx, world, q_wp, phase_tag="p1",
            theta_z_hold_deg=None, theta_z_tol_deg=5.0, n_hold=4,
        )
        chest = world.chest_pose()
        q_act = world.trunk_qpos()
        trace.append({
            "i": i, "phase": "p1",
            "C_act": round(vertical_C_from_trunk_q(q_act), 4),
            "theta_z_sim_deg": round(chest["theta_z_deg"], 2),
            "chest_z_sim": round(chest["z"], 4),
            "q_act": [round(float(x), 4) for x in q_act[:3]],
        })

    if p2_meta:
        n_p2 = yield from _yield_phase2_q1q2_closed_loop(
            ctx, world,
            z_target=z_tgt,
            theta_z_hold_deg=90.0,
            dq_sign_hint=float(p2_meta.get("dq_sign_hint", 1.0)),
            dq_step_rad=float(p2_meta.get("dq_rad", 0.012)),
            descending=True,
        )
        report["phase2_closed_loop_steps"] = n_p2
        chest = world.chest_pose()
        q_act = world.trunk_qpos()
        trace.append({
            "i": len(trace), "phase": "p2_end",
            "theta_z_sim_deg": round(chest["theta_z_deg"], 2),
            "chest_z_sim": round(chest["z"], 4),
            "q_act": [round(float(x), 4) for x in q_act[:3]],
        })

    chest_low = world.chest_pose()
    report["after_descent"] = {
        "chest_z": round(chest_low["z"], 4),
        "theta_z_deg": round(chest_low["theta_z_deg"], 2),
        "trunk_q": [round(float(x), 4) for x in world.trunk_qpos()[:3]],
    }
    report["trace_down"] = trace

    # 回升：下降的逆过程
    q_now = world.trunk_qpos()
    wp_up, vmeta_up = plan_vertical_lift_waypoints_v2(
        q_now, float(chest_low["z"]), float(chest0["z"]), base_link_z=base_z,
    )
    report["plan_up"] = vmeta_up
    if vmeta_up.get("ok") and wp_up:
        tz_up = vmeta_up.get("theta_z_hold_deg", chest0["theta_z_deg"])
        for i, q_wp in enumerate(wp_up):
            phase_tag = "p2" if i < max(len(wp_up) - 20, 0) else "p1"
            yield from _yield_trunk_vertical_waypoint(
                ctx, world, q_wp,
                phase_tag="p2" if phase_tag == "p2" else "p1",
                theta_z_hold_deg=tz_up,
                theta_z_tol_deg=5.0,
                n_hold=8,
            )
            if i % 5 == 0 or i == len(wp_up) - 1:
                chest = world.chest_pose()
                ctx.set_status(
                    f"diag_v2 ascent {i+1}/{len(wp_up)} z={chest['z']:.3f}"
                )

    chest1 = world.chest_pose()
    tz_spread = (
        max(t["theta_z_sim_deg"] for t in trace) - min(t["theta_z_sim_deg"] for t in trace)
        if trace else 0.0
    )
    report["end"] = {
        "chest_z": round(chest1["z"], 4),
        "theta_z_deg": round(chest1["theta_z_deg"], 2),
        "theta_z_spread_deg": round(tz_spread, 2),
        "measured_drop_m": round(float(chest0["z"]) - float(chest_low["z"]), 4),
        "measured_rise_back_err_m": round(float(chest1["z"]) - float(chest0["z"]), 4),
    }
    ctx.log(
        f"[diag_vertical_lift_v2] 实测降 {report['end']['measured_drop_m']}m "
        f"回升误差 {report['end']['measured_rise_back_err_m']:+.3f}m "
        f"θz spread={tz_spread:.1f}°"
    )
    ctx.set_result(report)
    yield world.hold_action()


@register_skill(
    "fold_deepest_phase2_q3",
    description=(
        "从当前姿态沿阶段2 q2向后下降；反解 q1/q3 保 θz≈90° 且优先折叠 q3；"
        "至 q2 后摆限位后拍 head 主视图保存。"
    ),
)
def fold_deepest_phase2_q3(
    ctx,
    z_target: float = 0.0,
    theta_z_hold_deg: float = 90.0,
    save_png: str = "behavior_interface/skills/test/fold_deepest_q3_priority.png",
):
    """q2 向后扫至限位，IK 优先折 q3，保存最深折姿 head 图。"""
    import os
    from behavior_interface.head_capture import capture_head_png
    from behavior_interface.trunk_vertical_lift import (
        plan_phase2_q2_backward_waypoints,
        vertical_C_from_trunk_q,
    )

    world = ctx.world
    base_z = _base_link_z_world(world)
    chest0 = world.chest_pose()
    q0 = world.trunk_qpos()

    waypoints, meta = plan_phase2_q2_backward_waypoints(
        q0,
        float(z_target),
        base_z,
        theta_z_hold_deg=float(theta_z_hold_deg),
        dq2_step_rad=-0.015,
        bias_q3_fold=True,
    )
    wp_exec = _subsample_trunk_waypoints(waypoints, max_pts=36)

    report: dict = {
        "ok": bool(meta.get("ok")),
        "plan": meta,
        "start": {
            "chest_z": round(float(chest0["z"]), 4),
            "theta_z_deg": round(float(chest0["theta_z_deg"]), 2),
            "trunk_q": [round(float(x), 4) for x in q0[:3]],
            "C_m": round(vertical_C_from_trunk_q(q0), 4),
        },
        "n_planned": len(waypoints),
        "n_exec": len(wp_exec),
        "save_png": save_png,
    }
    ctx.log(
        f"[fold_deepest_q3] z {chest0['z']:.3f}→目标{float(z_target):.3f}m "
        f"路点 {len(waypoints)}→{len(wp_exec)} "
        f"bias_q3_fold=True limited_by={meta.get('limited_by')}"
    )

    if not wp_exec:
        report["ok"] = False
        report["error"] = meta.get("limited_by", "无路点")
        ctx.set_result(report)
        yield world.hold_action()
        return

    for i, q_wp in enumerate(wp_exec):
        yield from _yield_trunk_vertical_waypoint(
            ctx, world, q_wp,
            phase_tag="p2q3",
            theta_z_hold_deg=None,
            theta_z_tol_deg=5.0,
            n_hold=2,
        )
        if i % 3 == 0 or i == len(wp_exec) - 1:
            chest = world.chest_pose()
            q_act = world.trunk_qpos()
            ctx.set_status(
                f"fold_q3 {i+1}/{len(wp_exec)} z={chest['z']:.3f} "
                f"q2={q_act[1]:+.3f} q3={q_act[2]:+.3f} "
                f"θz={chest['theta_z_deg']:.1f}°"
            )

    n_extra = yield from _yield_phase2_q2_backward_exec(
        ctx, world,
        z_target=float(z_target),
        theta_z_hold_deg=float(theta_z_hold_deg),
        dq2_step_rad=-0.012,
        bias_q3_fold=True,
        z_tol=0.02,
        max_steps=80,
        n_hold=2,
    )
    report["extra_closed_loop_steps"] = n_extra

    for _ in range(8):
        yield world.hold_action()

    chest_end = world.chest_pose()
    q_end = world.trunk_qpos()
    report["end"] = {
        "chest_z": round(float(chest_end["z"]), 4),
        "theta_z_deg": round(float(chest_end["theta_z_deg"]), 2),
        "trunk_q": [round(float(x), 4) for x in q_end[:3]],
        "C_m": round(vertical_C_from_trunk_q(q_end), 4),
        "drop_m": round(float(chest0["z"]) - float(chest_end["z"]), 4),
    }

    out_path = os.path.abspath(
        save_png if os.path.isabs(save_png)
        else os.path.join(os.getcwd(), save_png)
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap = capture_head_png(world, out_path, n_render=12)
    report["capture"] = cap or {"ok": False, "path": out_path}
    if cap:
        ctx.log(
            f"[fold_deepest_q3] 最深 z={chest_end['z']:.3f}m "
            f"q={[round(float(x),3) for x in q_end[:3]]} "
            f"已存 {out_path}"
        )
    else:
        ctx.log(f"[fold_deepest_q3] head 截图失败 path={out_path}")

    ctx.set_result(report)
    yield world.hold_action()


# 三阶段垂直折姿 GTA 截图视角（用户指定）
_VERTICAL_GTA_CAMERA = {
    "distance": 1.90,
    "height": 1.90,
    "look_z_offset": 1.00,
    "yaw_offset_deg": 75.0,
}


def _ensure_vertical_gta_camera(ctx) -> dict:
    fn = getattr(ctx, "_adjust_camera", None)
    if fn is None:
        return dict(_VERTICAL_GTA_CAMERA)
    return fn(overrides=dict(_VERTICAL_GTA_CAMERA))


def _capture_gta_view(ctx, world, fpath: str) -> bool:
    from behavior_interface.skills.tuck_trajectory import _capture_gta_png
    import os
    os.makedirs(os.path.dirname(os.path.abspath(fpath)) or ".", exist_ok=True)
    return bool(_capture_gta_png(world, fpath, ctx))


@register_skill(
    "diag_vertical_three_phase_gta",
    description=(
        "三阶段垂直折姿演示：阶段1 正弦流形→C_min；"
        "阶段2 |dq2|=|dq1| dq3=0 至 q1 限位；"
        "阶段3 |dq2|=|dq3| dq1=0 至 q3 限位；"
        "各阶段末 GTA 截图。"
    ),
)
def diag_vertical_three_phase_gta(
    ctx,
    reset_first: bool = True,
    out_dir: str = "behavior_interface/skills/test",
    dq_step_rad: float = 0.015,
    n_hold: int = 2,
    reverse_joint_dirs: bool = False,
):
    """直立复位后执行三阶段折姿，并在每阶段末保存 GTA 视角 PNG。

    reverse_joint_dirs=True：阶段1 fold_sign 取反，阶段2/3 关节步进方向全反（q2 回退下降）。
    """
    import os
    from behavior_interface.trunk_vertical_lift import (
        plan_phase2_q1q2_locked_q3_waypoints,
        plan_phase3_q2q3_locked_q1_waypoints,
        scan_sine_law_manifold,
        vertical_C_from_trunk_q,
        _sine_manifold_waypoints_between_C,
    )

    world = ctx.world
    base_z = _base_link_z_world(world)
    suffix = "_rev" if reverse_joint_dirs else ""
    out_base = os.path.abspath(
        out_dir if os.path.isabs(out_dir) else os.path.join(os.getcwd(), out_dir),
    )
    os.makedirs(out_base, exist_ok=True)
    shots = {
        "phase1": os.path.join(out_base, f"vertical_phase1_sine_end{suffix}.png"),
        "phase2": os.path.join(out_base, f"vertical_phase2_q1_limit{suffix}.png"),
        "phase3": os.path.join(out_base, f"vertical_phase3_q3_limit{suffix}.png"),
    }

    report: dict = {
        "ok": True,
        "reverse_joint_dirs": bool(reverse_joint_dirs),
        "camera": dict(_VERTICAL_GTA_CAMERA),
        "shots": shots,
    }

    if reset_first:
        from behavior_interface.skills.reset_body import reset_body
        ctx.log("[vertical_3phase] reset_body 直立复位…")
        yield from reset_body(ctx)
        for _ in range(6):
            yield world.hold_action()

    cam = _ensure_vertical_gta_camera(ctx)
    mode = "反向q2回退" if reverse_joint_dirs else "正向"
    ctx.log(
        f"[vertical_3phase/{mode}] GTA dist={cam.get('distance')}m h={cam.get('height')}m "
        f"tilt={cam.get('look_z_offset')}m yaw_off={cam.get('yaw_offset_deg')}°"
    )

    chest0 = world.chest_pose()
    q0 = world.trunk_qpos()
    C0 = vertical_C_from_trunk_q(q0)
    mf = scan_sine_law_manifold()
    C_min = float(mf["C_min_m"])
    report["start"] = {
        "chest_z": round(float(chest0["z"]), 4),
        "trunk_q": [round(float(x), 4) for x in q0[:3]],
        "C_m": round(C0, 4),
    }

    fold_sign = -1.0 if reverse_joint_dirs else 1.0
    n1 = 32 if reverse_joint_dirs else 20
    hold1 = max(int(n_hold), 4 if reverse_joint_dirs else int(n_hold))
    wp1, _ = _sine_manifold_waypoints_between_C(
        q0, C0, C_min, n_steps=n1, fold_sign=fold_sign,
    )
    if not wp1:
        report["ok"] = False
        report["error"] = "阶段1 正弦流形规划失败"
        ctx.set_result(report)
        yield world.hold_action()
        return

    ctx.log(
        f"[vertical_3phase] 阶段1 正弦流形({'反向fold' if reverse_joint_dirs else '正向'}) "
        f"C {C0:.3f}→{C_min:.3f} fold_sign={fold_sign:+.0f} 路点={len(wp1)}"
    )
    wp1_exec = _subsample_trunk_waypoints(wp1, max_pts=32 if reverse_joint_dirs else 24)
    yield from _yield_trunk_waypoints_openloop(
        ctx, world, wp1_exec, phase_tag="p1", n_hold=hold1,
    )
    for _ in range(6):
        yield world.hold_action()
    _ensure_vertical_gta_camera(ctx)
    ok1 = _capture_gta_view(ctx, world, shots["phase1"])
    q1_end = world.trunk_qpos()
    chest1 = world.chest_pose()
    report["after_phase1"] = {
        "chest_z": round(float(chest1["z"]), 4),
        "theta_z_deg": round(float(chest1["theta_z_deg"]), 2),
        "trunk_q": [round(float(x), 4) for x in q1_end[:3]],
        "C_m": round(vertical_C_from_trunk_q(q1_end), 4),
        "shot_ok": ok1,
    }
    ctx.log(
        f"[vertical_3phase] 阶段1 末 z={chest1['z']:.3f} "
        f"q={[round(float(x),3) for x in q1_end[:3]]} shot={shots['phase1']}"
    )

    wp2, m2 = plan_phase2_q1q2_locked_q3_waypoints(
        q1_end, base_z, z_target_world=None, dq_step_rad=float(dq_step_rad),
        reverse_joint_dirs=reverse_joint_dirs,
    )
    wp2_exec = _subsample_trunk_waypoints(
        wp2[1:] if len(wp2) > 1 else [], max_pts=48,
    )
    ctx.log(
        f"[vertical_3phase] 阶段2 |dq2|=|dq1| dq3=0 "
        f"路点={len(wp2_exec)} limited_by={m2.get('limited_by')}"
    )
    if wp2_exec:
        yield from _yield_trunk_waypoints_openloop(
            ctx, world, wp2_exec, phase_tag="p2", n_hold=n_hold,
        )
    for _ in range(6):
        yield world.hold_action()
    _ensure_vertical_gta_camera(ctx)
    ok2 = _capture_gta_view(ctx, world, shots["phase2"])
    q2_end = world.trunk_qpos()
    chest2 = world.chest_pose()
    report["after_phase2"] = {
        **m2,
        "chest_z": round(float(chest2["z"]), 4),
        "theta_z_deg": round(float(chest2["theta_z_deg"]), 2),
        "trunk_q": [round(float(x), 4) for x in q2_end[:3]],
        "shot_ok": ok2,
    }
    ctx.log(
        f"[vertical_3phase] 阶段2 末 z={chest2['z']:.3f} "
        f"q={[round(float(x),3) for x in q2_end[:3]]} shot={shots['phase2']}"
    )

    wp3, m3 = plan_phase3_q2q3_locked_q1_waypoints(
        q2_end, base_z, z_target_world=None, dq_step_rad=float(dq_step_rad),
        reverse_joint_dirs=reverse_joint_dirs,
    )
    wp3_exec = _subsample_trunk_waypoints(
        wp3[1:] if len(wp3) > 1 else [], max_pts=48,
    )
    ctx.log(
        f"[vertical_3phase] 阶段3 |dq2|=|dq3| dq1=0 "
        f"路点={len(wp3_exec)} limited_by={m3.get('limited_by')}"
    )
    if wp3_exec:
        yield from _yield_trunk_waypoints_openloop(
            ctx, world, wp3_exec, phase_tag="p3", n_hold=n_hold,
        )
    for _ in range(8):
        yield world.hold_action()
    _ensure_vertical_gta_camera(ctx)
    ok3 = _capture_gta_view(ctx, world, shots["phase3"])
    q3_end = world.trunk_qpos()
    chest3 = world.chest_pose()
    report["after_phase3"] = {
        **m3,
        "chest_z": round(float(chest3["z"]), 4),
        "theta_z_deg": round(float(chest3["theta_z_deg"]), 2),
        "trunk_q": [round(float(x), 4) for x in q3_end[:3]],
        "shot_ok": ok3,
    }
    report["end"] = {
        "total_drop_m": round(float(chest0["z"]) - float(chest3["z"]), 4),
        "all_shots_ok": bool(ok1 and ok2 and ok3),
    }
    ctx.log(
        f"[vertical_3phase] 阶段3 末 z={chest3['z']:.3f} "
        f"q={[round(float(x),3) for x in q3_end[:3]]} "
        f"总降={report['end']['total_drop_m']:.3f}m shot={shots['phase3']}"
    )
    if not (ok1 and ok2 and ok3):
        report["ok"] = False
        report["error"] = "部分 GTA 截图失败"
    ctx.set_result(report)
    yield world.hold_action()


def _annotate_gta_png_label(fpath: str, lines: List[str]) -> bool:
    """在 GTA 截图上叠加文字标签。"""
    try:
        import cv2
        img = cv2.imread(fpath)
        if img is None:
            return False
        for i, line in enumerate(lines):
            y = 42 + i * 46
            cv2.putText(
                img, str(line), (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (40, 255, 80), 3, cv2.LINE_AA,
            )
            cv2.putText(
                img, str(line), (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 1, cv2.LINE_AA,
            )
        cv2.imwrite(fpath, img)
        return True
    except Exception:
        return False


@register_skill(
    "diag_reverse_vertical_z_ik_gta",
    description=(
        "反向直立→正弦流形 C_min：每 1cm、fold_sign=-1 流形 + θz≡90° 反解 q；"
        "均匀 10 帧 GTA 并标注 upward。"
    ),
)
def diag_reverse_vertical_z_ik_gta(
    ctx,
    reset_first: bool = True,
    out_dir: str = "behavior_interface/skills/test/upward/phase1",
    dz_step_m: float = 0.01,
    theta_z_deg: float = 90.0,
    n_shots: int = 10,
    n_hold: int = 5,
    pick_row_index: Optional[int] = None,
):
    """反向下降 z-θz IK 查表 + 均匀 10 张 GTA 验证图。"""
    import json
    import os
    from behavior_interface.trunk_vertical_lift import (
        estimate_chest_z_world,
        fk_torso_link4_theta_z_deg,
        sample_reverse_vertical_z_ik_table,
        vertical_C_from_trunk_q,
    )

    world = ctx.world
    base_z = _base_link_z_world(world)
    out_base = os.path.abspath(
        out_dir if os.path.isabs(out_dir) else os.path.join(os.getcwd(), out_dir),
    )
    os.makedirs(out_base, exist_ok=True)
    table_path = os.path.join(out_base, "reverse_vertical_z_ik_table.json")

    report: dict = {
        "ok": True,
        "out_dir": out_base,
        "table_json": table_path,
        "theta_z_hold_deg": float(theta_z_deg),
        "dz_step_m": float(dz_step_m),
    }

    if reset_first:
        from behavior_interface.skills.reset_body import reset_body
        ctx.log("[rev_z_ik] reset_body 直立复位…")
        yield from reset_body(ctx)
        for _ in range(6):
            yield world.hold_action()

    q4 = float(world.trunk_qpos()[3])
    table = sample_reverse_vertical_z_ik_table(
        base_z,
        dz_step_m=float(dz_step_m),
        theta_z_deg=float(theta_z_deg),
        q4=q4,
    )
    report["table_meta"] = {
        k: table.get(k)
        for k in (
            "ok", "model", "z_start_m", "z_end_m", "z_end_manifold_Cmin_m",
            "z_end_phase2_openloop_m", "C_start_m", "C_min_m",
            "dz_step_m", "n_levels", "n_solved", "theta_z_hold_deg",
            "q_at_C_min_manifold", "q_phase2_end_openloop", "manifold_chain_n",
        )
    }
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    ok_rows = [r for r in table.get("samples", []) if r.get("ok")]
    if not ok_rows:
        report["ok"] = False
        report["error"] = "z-θz IK 表为空"
        ctx.set_result(report)
        yield world.hold_action()
        return

    n_pick = max(1, min(int(n_shots), len(ok_rows)))
    if pick_row_index is not None:
        k = int(pick_row_index)
        if k < 0 or k >= len(ok_rows):
            report["ok"] = False
            report["error"] = f"pick_row_index={k} 越界 (0..{len(ok_rows)-1})"
            ctx.set_result(report)
            yield world.hold_action()
            return
        pick_idx = np.array([k], dtype=int)
    else:
        pick_idx = np.linspace(0, len(ok_rows) - 1, n_pick, dtype=int)

    cam = _ensure_vertical_gta_camera(ctx)
    ctx.log(
        f"[rev_z_ik] z {table['z_start_m']:.3f}→{table['z_end_m']:.3f}m "
        f"步长={dz_step_m*100:.0f}cm 解出 {len(ok_rows)}/{table.get('n_levels')} "
        f"抽 {n_pick} 张 GTA"
    )

    shots: List[dict] = []
    z_start = float(table["z_start_m"])
    prev_row_k = 0
    all_reached = True
    for j, k_end in enumerate(pick_idx):
        k_end_i = int(k_end)
        for k in range(prev_row_k, k_end_i + 1):
            row_k = ok_rows[k]
            q_wp = np.array(row_k["trunk_q"], dtype=np.float64)
            is_photo_pose = (k == k_end_i)
            if is_photo_pose:
                settle = yield from _yield_trunk_settle_to_pose(
                    ctx, world, q_wp,
                    theta_z_deg=float(theta_z_deg),
                    q_tol_rad=0.022,
                    tz_tol_deg=2.5,
                    max_steps=180,
                    settle_frames=6,
                    n_hold=max(2, int(n_hold)),
                )
            else:
                yield from _yield_trunk_waypoints_openloop(
                    ctx, world, [q_wp], phase_tag=f"z{k}", n_hold=4,
                )
                settle = {"reached": True}
        prev_row_k = k_end_i
        row = ok_rows[k_end_i]

        _ensure_vertical_gta_camera(ctx)

        upward = float(row.get("upward_m", row["z_target_m"] - z_start))
        fname = (
            f"reverse_vertical_z_ik_up{upward:+.3f}m_"
            f"z{row['z_target_m']:.3f}.png"
        ).replace("+", "p")
        fpath = os.path.join(out_base, fname)
        ok_cap = _capture_gta_view(ctx, world, fpath)

        chest = world.chest_pose()
        q_act = world.trunk_qpos()
        reached = bool(settle.get("reached", False))
        if not reached:
            all_reached = False

        label = f"upward={upward:+.3f}m"
        label2 = (
            f"plan q=[{row['q1_rad']:+.2f},{row['q2_rad']:+.2f},{row['q3_rad']:+.2f}]"
        )
        label3 = (
            f"sim  q=[{q_act[0]:+.2f},{q_act[1]:+.2f},{q_act[2]:+.2f}] "
            f"tz={chest['theta_z_deg']:.1f}"
        )
        if ok_cap:
            _annotate_gta_png_label(fpath, [label, label2, label3])

        shots.append({
            "shot": fpath,
            "shot_ok": ok_cap,
            "pose_reached": reached,
            "settle": settle,
            "upward_m": round(upward, 4),
            "z_target_m": row["z_target_m"],
            "plan_q": row["trunk_q"][:3],
            "sim_q": [round(float(x), 4) for x in q_act[:3]],
            "sim_z": round(float(chest["z"]), 4),
            "sim_theta_z_deg": round(float(chest["theta_z_deg"]), 2),
            "label": label,
        })
        ctx.log(
            f"[rev_z_ik] {j+1}/{n_pick} {label} "
            f"reached={reached} sim_z={chest['z']:.3f} "
            f"θz={chest['theta_z_deg']:.1f}° "
            f"q_err={settle.get('q_err_rad', '?')}"
        )

    report["shots"] = shots
    report["picked_indices"] = [int(i) for i in pick_idx]
    report["all_poses_reached"] = all_reached
    report["all_shots_ok"] = all(s.get("shot_ok") for s in shots)
    if not report["all_shots_ok"]:
        report["ok"] = False
        report["error"] = "部分 GTA 截图失败"
    elif not all_reached:
        report["ok"] = False
        report["error"] = "部分截图姿态未收敛到规划关节/θz"
    ctx.set_result(report)
    yield world.hold_action()


def _capture_reverse_upward_z_rows(
    ctx,
    world,
    ok_rows: List[dict],
    *,
    out_dir: str,
    z_start: float,
    theta_z_deg: float,
    n_shots: int,
    n_hold: int,
    pick_row_index: Optional[int],
    phase_tag: str,
    fname_prefix: str = "",
) -> Tuple[List[dict], bool, int]:
    """沿查表行单调行走，均匀截图并闭环到位。返回 (shots, all_reached, last_row_k)。"""
    import os

    os.makedirs(out_dir, exist_ok=True)
    n_pick = max(1, min(int(n_shots), len(ok_rows)))
    if pick_row_index is not None:
        k = int(pick_row_index)
        if k < 0 or k >= len(ok_rows):
            return [], False, 0
        pick_idx = np.array([k], dtype=int)
    else:
        pick_idx = np.linspace(0, len(ok_rows) - 1, n_pick, dtype=int)

    shots: List[dict] = []
    prev_row_k = 0
    all_reached = True
    for j, k_end in enumerate(pick_idx):
        k_end_i = int(k_end)
        settle: dict = {"reached": True}
        for k in range(prev_row_k, k_end_i + 1):
            row_k = ok_rows[k]
            q_wp = np.array(row_k["trunk_q"], dtype=np.float64)
            if k == k_end_i:
                settle = yield from _yield_trunk_settle_to_pose(
                    ctx, world, q_wp,
                    theta_z_deg=float(theta_z_deg),
                    q_tol_rad=0.022,
                    tz_tol_deg=2.5,
                    max_steps=180,
                    settle_frames=6,
                    n_hold=max(2, int(n_hold)),
                )
            else:
                yield from _yield_trunk_waypoints_openloop(
                    ctx, world, [q_wp], phase_tag=f"{phase_tag}{k}", n_hold=4,
                )
        prev_row_k = k_end_i
        row = ok_rows[k_end_i]
        _ensure_vertical_gta_camera(ctx)

        upward = float(row.get("upward_m", row["z_target_m"] - z_start))
        fname = (
            f"{fname_prefix}up{upward:+.3f}m_z{row['z_target_m']:.3f}.png"
        ).replace("+", "p")
        fpath = os.path.join(out_dir, fname)
        ok_cap = _capture_gta_view(ctx, world, fpath)

        chest = world.chest_pose()
        q_act = world.trunk_qpos()
        reached = bool(settle.get("reached", False))
        if not reached:
            all_reached = False
        label = f"upward={upward:+.3f}m"
        label2 = (
            f"plan q=[{row['q1_rad']:+.2f},{row['q2_rad']:+.2f},{row['q3_rad']:+.2f}]"
        )
        label3 = (
            f"sim  q=[{q_act[0]:+.2f},{q_act[1]:+.2f},{q_act[2]:+.2f}] "
            f"tz={chest['theta_z_deg']:.1f}"
        )
        if ok_cap:
            _annotate_gta_png_label(fpath, [label, label2, label3])
        shots.append({
            "shot": fpath,
            "shot_ok": ok_cap,
            "pose_reached": reached,
            "settle": settle,
            "upward_m": round(upward, 4),
            "z_target_m": row["z_target_m"],
            "plan_q": row["trunk_q"][:3],
            "sim_q": [round(float(x), 4) for x in q_act[:3]],
            "sim_z": round(float(chest["z"]), 4),
            "sim_theta_z_deg": round(float(chest["theta_z_deg"]), 2),
            "label": label,
        })
        ctx.log(
            f"[{phase_tag}] {j+1}/{n_pick} {label} reached={reached} "
            f"z={chest['z']:.3f} θz={chest['theta_z_deg']:.1f}°"
        )
    return shots, all_reached, prev_row_k


@register_skill(
    "diag_reverse_upward_two_phase_gta",
    description=(
        "反向 upward 两阶段：phase1 正弦流形→C_min 截图 upward/phase1；"
        "phase2 仅 θz≈90° 探底截图 upward/phase2，报告触限关节。"
    ),
)
def diag_reverse_upward_two_phase_gta(
    ctx,
    reset_first: bool = True,
    base_out_dir: str = "behavior_interface/skills/test/upward",
    dz_step_m: float = 0.01,
    theta_z_deg: float = 90.0,
    n_shots_phase1: int = 10,
    n_shots_phase2: int = 10,
    n_hold: int = 5,
):
    """phase1 流形 + phase2 θz-only 连续演示与截图。"""
    import json
    import os
    from behavior_interface.trunk_vertical_lift import (
        reverse_sine_manifold_z_bounds,
        sample_phase2_reverse_theta90_z_table,
        sample_reverse_vertical_z_ik_table,
    )

    world = ctx.world
    base_z = _base_link_z_world(world)
    out_root = os.path.abspath(
        base_out_dir if os.path.isabs(base_out_dir)
        else os.path.join(os.getcwd(), base_out_dir),
    )
    dir_p1 = os.path.join(out_root, "phase1")
    dir_p2 = os.path.join(out_root, "phase2")
    os.makedirs(dir_p1, exist_ok=True)
    os.makedirs(dir_p2, exist_ok=True)

    report: dict = {
        "ok": True,
        "out_dirs": {"phase1": dir_p1, "phase2": dir_p2},
        "theta_z_hold_deg": float(theta_z_deg),
    }

    if reset_first:
        from behavior_interface.skills.reset_body import reset_body
        ctx.log("[upward_2phase] reset_body…")
        yield from reset_body(ctx)
        for _ in range(6):
            yield world.hold_action()

    q4 = float(world.trunk_qpos()[3])
    _ensure_vertical_gta_camera(ctx)

    table_p1 = sample_reverse_vertical_z_ik_table(
        base_z, dz_step_m=float(dz_step_m), theta_z_deg=float(theta_z_deg), q4=q4,
    )
    p1_json = os.path.join(dir_p1, "phase1_sine_manifold_table.json")
    with open(p1_json, "w", encoding="utf-8") as f:
        json.dump(table_p1, f, ensure_ascii=False, indent=2)
    report["phase1"] = {k: table_p1.get(k) for k in (
        "ok", "model", "z_start_m", "z_end_m", "n_solved", "n_levels",
        "q_at_C_min_manifold",
    )}
    report["phase1"]["table_json"] = p1_json

    ok_p1 = [r for r in table_p1.get("samples", []) if r.get("ok")]
    if not ok_p1:
        report["ok"] = False
        report["error"] = "阶段1 查表为空"
        ctx.set_result(report)
        yield world.hold_action()
        return

    z_upright = float(table_p1["z_start_m"])
    ctx.log(
        f"[upward_2phase] 阶段1 流形 z {table_p1['z_start_m']:.3f}→"
        f"{table_p1['z_end_m']:.3f}m"
    )
    shots_p1, reached_p1, last_k = yield from _capture_reverse_upward_z_rows(
        ctx, world, ok_p1,
        out_dir=dir_p1,
        z_start=z_upright,
        theta_z_deg=float(theta_z_deg),
        n_shots=int(n_shots_phase1),
        n_hold=int(n_hold),
        pick_row_index=None,
        phase_tag="p1_",
        fname_prefix="phase1_",
    )
    report["phase1"]["shots"] = shots_p1
    report["phase1"]["all_poses_reached"] = reached_p1

    for k in range(last_k, len(ok_p1)):
        q_wp = np.array(ok_p1[k]["trunk_q"], dtype=np.float64)
        if k == len(ok_p1) - 1:
            yield from _yield_trunk_settle_to_pose(
                ctx, world, q_wp, theta_z_deg=float(theta_z_deg),
                q_tol_rad=0.022, tz_tol_deg=2.5, max_steps=180, n_hold=max(2, int(n_hold)),
            )
        else:
            yield from _yield_trunk_waypoints_openloop(
                ctx, world, [q_wp], phase_tag=f"p1tail{k}", n_hold=4,
            )

    _, _, _, _, q_p1_end = reverse_sine_manifold_z_bounds(base_z, q4=q4)

    table_p2 = sample_phase2_reverse_theta90_z_table(
        q_p1_end, base_z,
        z_upright_world=z_upright,
        dz_step_m=float(dz_step_m),
        theta_z_deg=float(theta_z_deg),
    )
    p2_json = os.path.join(dir_p2, "phase2_theta90_table.json")
    with open(p2_json, "w", encoding="utf-8") as f:
        json.dump(table_p2, f, ensure_ascii=False, indent=2)

    limited = table_p2.get("limited_by")
    report["phase2"] = {
        "table_json": p2_json,
        "model": table_p2.get("model"),
        "z_start_m": table_p2.get("z_start_m"),
        "z_end_m": table_p2.get("z_end_m"),
        "z_lowest_m": table_p2.get("z_end_m"),
        "limited_by": limited,
        "limit_detail": table_p2.get("limit_detail"),
        "total_drop_from_upright_m": table_p2.get("total_drop_from_upright_m"),
        "phase2_extra_drop_m": table_p2.get("phase2_extra_drop_m"),
        "q_end": table_p2.get("phase2_plan", {}).get("q_end"),
        "theta_z_end_deg": table_p2.get("phase2_plan", {}).get("theta_z_end_deg"),
    }
    ctx.log(
        f"[upward_2phase] 阶段2 z {table_p2['z_start_m']:.3f}→{table_p2['z_end_m']:.3f}m "
        f"触限 **{limited}** 共降 {table_p2['total_drop_from_upright_m']:.3f}m"
    )

    ok_p2 = [r for r in table_p2.get("samples", []) if r.get("ok")]
    shots_p2, reached_p2, _ = yield from _capture_reverse_upward_z_rows(
        ctx, world, ok_p2,
        out_dir=dir_p2,
        z_start=z_upright,
        theta_z_deg=float(theta_z_deg),
        n_shots=int(n_shots_phase2),
        n_hold=int(n_hold),
        pick_row_index=None,
        phase_tag="p2_",
        fname_prefix="phase2_",
    )

    if ok_p2:
        row_deep = ok_p2[-1]
        q_deep = np.array(row_deep["trunk_q"], dtype=np.float64)
        yield from _yield_trunk_settle_to_pose(
            ctx, world, q_deep, theta_z_deg=float(theta_z_deg),
            q_tol_rad=0.022, tz_tol_deg=2.5, max_steps=200, n_hold=max(2, int(n_hold)),
        )
        _ensure_vertical_gta_camera(ctx)
        upward_d = float(row_deep.get("upward_m", 0))
        fdeep = os.path.join(
            dir_p2,
            f"phase2_deepest_{limited}_up{upward_d:+.3f}m_z{row_deep['z_target_m']:.3f}.png"
            .replace("+", "p"),
        )
        if _capture_gta_view(ctx, world, fdeep):
            chest = world.chest_pose()
            q_act = world.trunk_qpos()
            _annotate_gta_png_label(fdeep, [
                f"DEEPEST limited_by={limited}",
                f"upward={upward_d:+.3f}m",
                f"plan q=[{row_deep['q1_rad']:+.2f},{row_deep['q2_rad']:+.2f},{row_deep['q3_rad']:+.2f}]",
                f"sim q=[{q_act[0]:+.2f},{q_act[1]:+.2f},{q_act[2]:+.2f}] tz={chest['theta_z_deg']:.1f}",
            ])
            shots_p2.append({
                "shot": fdeep,
                "shot_ok": True,
                "deepest": True,
                "limited_by": limited,
                "upward_m": round(upward_d, 4),
                "z_target_m": row_deep["z_target_m"],
            })

    report["phase2"]["shots"] = shots_p2
    report["phase2"]["all_poses_reached"] = reached_p2
    if not (reached_p1 and reached_p2):
        report["ok"] = False
        report["error"] = "部分截图未到位"
    ctx.set_result(report)
    yield world.hold_action()
