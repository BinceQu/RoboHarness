"""
arm_reset(arm='right'|'left'|'both', mode='grasp'|'hang'|'ready', open_gripper=True)

把 R1Pro 手臂关节回到安全默认姿态（不动 base / trunk）。

mode 选项：
  'grasp' （默认）抓取预备位：肘腕折叠在身前，避免升降时手臂撑到底盘。
  'hang'  自然下垂 = scene reset 时的初始状态：所有 arm 关节归零。
  'ready' 就绪姿态（R1Pro untucked）：right [0,-1.57,0,-1.57,-1.57,0,0]，
          双手弯肘向前，方便接近操作台。

示例：
  arm_reset()                       # 默认 right + grasp prep + 开夹爪
  arm_reset(arm='left')
  arm_reset(arm='both')
  arm_reset(mode='hang')            # 显式恢复自然下垂
  arm_reset(mode='ready')           # 弯肘就绪
  arm_reset(open_gripper=False)     # 抓着东西换姿态时保持夹爪闭合
"""

from __future__ import annotations

import os
import time
import math

import numpy as np

from behavior_interface.skills import register_skill


# 自然下垂 = scene reset 初始状态 = 所有关节归零
_HANG_ARM_QPOS = np.zeros(7, dtype=np.float64)
_GRASP_PREP_ARM_QPOS = np.array(
    [0.0, 0.0, 0.0, -2.0943951024, 0.0, -1.0471975512, 0.0],
    dtype=np.float64,
)
_GRASP_PHASE1_ARM_QPOS = _GRASP_PREP_ARM_QPOS.copy()
_GRASP_PHASE1_ARM_QPOS[0] = 1.401

# 大臂/肩：j1-j3；肘腕 j4-j7 在抓取预备位保持 Phase1 角
_UPPER_ARM_DOF = (0, 1, 2)
_LOWER_ARM_DOF = (3, 4, 5, 6)
_GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP = 0.12
_GRASP_PREP_ACCEPT_TOL_RAD = 0.16
_GRASP_PREP_DIRECT_LOWER_TOL_RAD = 0.12
_GRASP_PREP_STAGNATION_S = 2.0
_GRASP_PREP_EEF_LINE_MIN_TIMEOUT_S = 6.0
_GRASP_PREP_EEF_LINE_MAX_TIMEOUT_S = 15.0
_GRASP_PREP_EEF_LINE_MAX_START_GAP_RAD = 0.75
# 普通 grasp-prep 是分段安全轨迹，不需要 keep_ori 的逐帧腕部追踪粒度。
# 两套速度档必须分开：曾把这里降到 0.04..0.10rad/frame，5026 的
# 1.79rad 首段因此被拆成 49 帧，并在 15s 墙钟预算内必然超时。
_GRASP_PREP_EEF_LINE_MAX_DQ_RAD = 0.45
_GRASP_PREP_JOINTSPACE_MAX_DQ_RAD = 0.45
_GRASP_PREP_FAST_MIN_DQ_RAD = 0.28
_GRASP_PREP_INTERMEDIATE_SETTLE_TOL_RAD = 0.05
_GRASP_PREP_FINAL_SETTLE_TOL_RAD = 0.035
_GRASP_PREP_RESIDUAL_CORRECT_TOL_RAD = 0.50
_GRASP_PREP_RESIDUAL_CORRECT_FRAMES = 24
_GRASP_PREP_SETTLE_STABLE_FRAMES = 2
_GRASP_PREP_INTERMEDIATE_SETTLE_MAX_FRAMES = 2
_GRASP_PREP_FINAL_SETTLE_MAX_FRAMES = 24
_GRASP_PREP_HIGH_J3_ESCAPE_RAD = 1.10
_GRASP_PREP_ELBOW_UNFOLD_ESCAPE_Q = -0.65
_GRASP_PREP_ESCAPE_MAX_DQ_RAD = 0.10
_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES = 4
_GRASP_PREP_SMOOTHSTEP_MAX_SLOPE = 1.5
_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG = 1.0
_GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG = 5.0
_GRASP_PREP_KEEP_ORI_ABORT_DEG = 25.0
_GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_P_GAIN = 0.75
_GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_GAIN_RAMP = 0.02
_GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_MAX_GAIN = 1.50
_GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD = 0.012
_GRASP_PREP_KEEP_ORI_FINAL_SETTLE_MAX_FRAMES = 80
_GRASP_PREP_KEEP_ORI_FINAL_STABLE_FRAMES = 2
_GRASP_PREP_KEEP_ORI_POS_TOL_M = 0.018
_GRASP_PREP_KEEP_ORI_WAYPOINT_STEP_M = 0.025
_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD = 0.10
# keep_ori 下 J1-J4 每帧运动量必须明显小于腕部单帧步长预算，否则腕部
# 无论如何都抵消不掉机体带来的姿态变化。这里不能复用 fast 路径的 0.28 下限。
_GRASP_PREP_KEEP_ORI_MIN_DQ_RAD = 0.03
_GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD = 0.25
# keep_ori 的插值粒度是普通路径的数倍细，帧数也是数倍，不能沿用普通路径的
# 时间预算（调用方常传 15s，那是按粗粒度播放估的）。
_GRASP_PREP_KEEP_ORI_MIN_TIMEOUT_S = 60.0
_GRASP_PREP_KEEP_ORI_MAX_TIMEOUT_S = 150.0

# 就绪姿态（R1Pro untucked_default_joint_pos 定义的弯肘向前）
_READY_ARM = {
    "left":  np.array([0.0, +1.57, 0.0, -1.57, +1.57, 0.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, -1.57, 0.0, -1.57, -1.57, 0.0, 0.0], dtype=np.float64),
}


def _grasp_prep_play_max_dq(
    requested: float,
    *,
    keep_ori: bool,
    force_jointspace: bool = False,
) -> float:
    """Resolve independent normal and keep-orientation playback speeds."""
    value = float(requested)
    if keep_ori:
        return max(
            _GRASP_PREP_KEEP_ORI_MIN_DQ_RAD,
            min(_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD, value),
        )
    cap = (
        _GRASP_PREP_JOINTSPACE_MAX_DQ_RAD
        if bool(force_jointspace)
        else _GRASP_PREP_EEF_LINE_MAX_DQ_RAD
    )
    return max(_GRASP_PREP_FAST_MIN_DQ_RAD, min(cap, value))


def _challenge_action_only_enabled() -> bool:
    return str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE", "")
    ).lower().strip() in {"train", "public_test", "hidden_test"}


def _get_arm_dof_idx(world, arm: str) -> np.ndarray:
    """返回 arm_left/right 在 robot.get_joint_positions() 中的索引。"""
    robot = world.robot
    names = list(robot.joints.keys())
    return np.array(
        [names.index(f"{arm}_arm_joint{i+1}") for i in range(7)], dtype=int
    )


def _arm_qpos(world, arm: str) -> np.ndarray:
    robot = world.robot
    qpos = robot.get_joint_positions()
    idx = _get_arm_dof_idx(world, arm)
    out = np.zeros(7, dtype=np.float64)
    for i, j in enumerate(idx):
        out[i] = float(qpos[int(j)])
    return out


# ±2.6 只是读不到 URDF 限位时的兜底值，不该再与真实限位取交集：R1Pro 的 J1
# 真实下限是 -4.451rad(-255°)、J2 真实上限是 +3.142rad(180°)，被这个通用值分别
# 砍掉 106° 和 31°。J1 曾因此顶死在 -2.6（一个纯人为的限位），导致终点 IK 在
# "保持姿态"前提下报 273mm 不可达，姿态只能靠牺牲自己去换位置。
# 改为以真实限位为准，仅对大行程关节保留自碰撞余量：J1 往负方向是手臂绕向身体
# 后侧，不放到机械极限。
_ARM_JOINT_FALLBACK_LIMIT = 2.6
_ARM_JOINT_SAFETY_LOWER = {1: -3.6}
_ARM_JOINT_SAFETY_UPPER = {2: 3.0}


def _arm_joint_limits(world, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite per-joint limits for one 7-DOF arm."""
    names = list(world.robot.joints.keys())
    lo = np.full(7, -_ARM_JOINT_FALLBACK_LIMIT, dtype=np.float64)
    hi = np.full(7, +_ARM_JOINT_FALLBACK_LIMIT, dtype=np.float64)
    for i in range(7):
        jn = f"{arm}_arm_joint{i+1}"
        if jn not in names:
            continue
        joint = world.robot.joints[jn]
        try:
            l = float(getattr(joint, "lower_limit"))
            u = float(getattr(joint, "upper_limit"))
        except Exception:
            continue
        if np.isfinite(l):
            lo[i] = l
        if np.isfinite(u):
            hi[i] = u
        safety_lower = _ARM_JOINT_SAFETY_LOWER.get(i + 1)
        if safety_lower is not None:
            lo[i] = max(lo[i], safety_lower)
        safety_upper = _ARM_JOINT_SAFETY_UPPER.get(i + 1)
        if safety_upper is not None:
            hi[i] = min(hi[i], safety_upper)
    hi = np.maximum(hi, lo + 1e-3)
    return lo, hi


def _make_legacy_arm_action(world, **overrides):
    """Submit an old 7DOF arm action through the per-frame J8=0 guard."""
    from behavior_interface.skills.eef import _make_legacy_7dof_action

    return _make_legacy_7dof_action(world, **overrides)


def _arm_reset_one(ctx, arm: str, target_qpos: np.ndarray,
                   open_gripper: bool,
                   max_dq_per_step: float, tol: float, timeout_s: float,
                   active_dof: tuple[int, ...] | None = None,
                   hold_target_dof: tuple[int, ...] | None = None,
                   log_prefix: str | None = None,
                   accept_tol: float | None = None,
                   stagnation_s: float = 2.0,
                   stagnation_epsilon: float = 0.003):
    """单臂关节空间插值：每步 dq=clip(target-cur, ±max_dq)，无需 IK/规划。"""
    active = tuple(active_dof) if active_dof is not None else tuple(range(7))
    hold_target = tuple(hold_target_dof) if hold_target_dof is not None else ()
    tag = log_prefix or f"arm_reset[{arm}]"
    world = ctx.world
    if world.dry_run:
        ctx.log(f"{tag} dry_run 跳过")
        return

    try:
        _ = world.controller_action_idx(f"arm_{arm}")
    except Exception:
        ctx.log(f"{tag} 找不到 arm_{arm} controller，跳过")
        return

    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{tag}.prepare"
    )
    t_start = time.time()
    last_log = 0.0
    last_file_log = 0.0
    best_err_inf = float("inf")
    best_err_t = t_start
    last_cur = None
    last_err_inf = float("inf")
    while True:
        if time.time() - t_start > timeout_s:
            cur_txt = "unknown" if last_cur is None else ",".join(
                f"{v:+.2f}" for v in last_cur
            )
            ctx.log(
                f"{tag} TIMEOUT after {timeout_s:.1f}s "
                f"final_err_inf={last_err_inf:.3f} cur=[{cur_txt}]"
            )
            return {"ok": False, "timeout": True, "err_inf": last_err_inf}

        try:
            cur = _arm_qpos(world, arm)
        except Exception as e:
            ctx.log(f"{tag} 读 qpos 失败: {e}")
            return

        err = target_qpos - cur
        err_inf = float(max(abs(float(err[i])) for i in active))
        last_cur = cur
        last_err_inf = err_inf
        now = time.time()
        if err_inf + float(stagnation_epsilon) < best_err_inf:
            best_err_inf = err_inf
            best_err_t = now
        accepted = (
            accept_tol is not None
            and err_inf <= float(accept_tol)
            and now - best_err_t >= float(stagnation_s)
        )
        stalled = (
            now - best_err_t >= max(2.0, float(stagnation_s))
            and not (accept_tol is not None and err_inf <= float(accept_tol))
        )
        if err_inf < tol or accepted:
            if accepted and err_inf >= tol:
                ctx.log(
                    f"{tag} ACCEPT plateau err_inf={err_inf:.4f} rad "
                    f"accept_tol={float(accept_tol):.3f} best={best_err_inf:.4f}"
                )
            else:
                ctx.log(f"{tag} DONE err_inf={err_inf:.4f} rad")
            hold_q = cur.tolist()
            for i in hold_target:
                hold_q[int(i)] = float(target_qpos[int(i)])
            for _ in range(5):
                yield _make_legacy_arm_action(
                    world,
                    **{
                        f"arm_{arm}": hold_q,
                        f"gripper_{arm}": _gripper_cmd(world, arm, open_gripper),
                    },
                )
            return {"ok": True, "accepted": accepted, "err_inf": err_inf}
        if stalled:
            ctx.log(
                f"{tag} STALL err_inf={err_inf:.4f} rad "
                f"best={best_err_inf:.4f} no_improve={now - best_err_t:.1f}s"
            )
            return {"ok": False, "stalled": True, "err_inf": err_inf}

        dq = np.zeros(7, dtype=np.float64)
        for i in active:
            dq[i] = float(np.clip(err[i], -max_dq_per_step, max_dq_per_step))
        if now - last_log > 0.5:
            ctx.set_status(f"{tag} err_inf={err_inf:.3f}rad")
            last_log = now
        if now - last_file_log > 2.0:
            ctx.log(
                f"{tag} step t={now-t_start:.1f}s err_inf={err_inf:.3f}rad "
                f"cur=[{','.join(f'{v:+.2f}' for v in cur)}] "
                f"tgt=[{','.join(f'{v:+.2f}' for v in target_qpos)}]"
            )
            last_file_log = now
        q_target = (cur + dq).tolist()
        for i in hold_target:
            q_target[int(i)] = float(target_qpos[int(i)])
        yield _make_legacy_arm_action(
            world,
            **{
                f"arm_{arm}": q_target,
                f"gripper_{arm}": _gripper_cmd(world, arm, open_gripper),
            },
        )


def _arms_move_parallel(
    ctx,
    targets: dict[str, np.ndarray],
    *,
    open_gripper: bool,
    max_dq_per_step: float,
    tol: float,
    timeout_s: float,
    active_dof: tuple[int, ...] | None = None,
    hold_target_dof: tuple[int, ...] | None = None,
    log_prefix: str = "arms_parallel",
    accept_tol: float | None = None,
    stagnation_s: float = 2.0,
    stagnation_epsilon: float = 0.003,
):
    """多臂同步到位：每步同一 action 里下发所有 arm。"""
    active = tuple(active_dof) if active_dof is not None else tuple(range(7))
    hold_target = tuple(hold_target_dof) if hold_target_dof is not None else ()
    world = ctx.world
    arms = [a for a in ("left", "right") if a in targets]
    if not arms:
        return
    if world.dry_run:
        ctx.log(f"{log_prefix} dry_run 跳过 arms={arms}")
        return

    for a in arms:
        try:
            _ = world.controller_action_idx(f"arm_{a}")
        except Exception:
            ctx.log(f"{log_prefix} 找不到 arm_{a} controller，跳过该臂")
            arms = [x for x in arms if x != a]
    if not arms:
        return

    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    for a in arms:
        _prepare_legacy_7dof_motion(
            world, a, ctx=ctx, stage_name=f"{log_prefix}.{a}.prepare"
        )
    t_start = time.time()
    last_log = 0.0
    last_file_log = 0.0
    logged_start = False
    best_err = float("inf")
    best_err_t = t_start
    last_err_map: dict[str, float] = {}

    def _step_err(cur: np.ndarray, tgt: np.ndarray) -> float:
        err = tgt - cur
        return float(max(abs(float(err[i])) for i in active))

    while True:
        if time.time() - t_start > timeout_s:
            ctx.log(
                f"{log_prefix} TIMEOUT after {timeout_s:.1f}s arms={arms} "
                + " ".join(f"{a}={e:.3f}" for a, e in last_err_map.items())
            )
            return {"ok": False, "timeout": True, "err_map": last_err_map}

        cur_map: dict[str, np.ndarray] = {}
        err_map: dict[str, float] = {}
        all_done = True
        for a in arms:
            try:
                cur = _arm_qpos(world, a)
            except Exception as e:
                ctx.log(f"{log_prefix}[{a}] 读 qpos 失败: {e}")
                return
            cur_map[a] = cur
            ei = _step_err(cur, targets[a])
            err_map[a] = ei
            if ei >= tol:
                all_done = False
        last_err_map = dict(err_map)

        if not logged_start:
            logged_start = True
            for a in arms:
                cur = cur_map[a]
                ctx.log(
                    f"{log_prefix}[{a}] start err_inf={err_map[a]:.3f} rad "
                    f"cur=[{','.join(f'{v:+.2f}' for v in cur)}] "
                    f"tgt=[{','.join(f'{v:+.2f}' for v in targets[a])}]"
                )

        now = time.time()
        max_err = max(err_map.values()) if err_map else 0.0
        if max_err + float(stagnation_epsilon) < best_err:
            best_err = max_err
            best_err_t = now
        accepted = (
            accept_tol is not None
            and max_err <= float(accept_tol)
            and now - best_err_t >= float(stagnation_s)
        )
        stalled = (
            now - best_err_t >= max(2.0, float(stagnation_s))
            and not (accept_tol is not None and max_err <= float(accept_tol))
        )

        if all_done or accepted:
            if accepted and not all_done:
                ctx.log(
                    f"{log_prefix} ACCEPT plateau arms={arms} "
                    + " ".join(f"{a}={err_map[a]:.4f}" for a in arms)
                    + f" accept_tol={float(accept_tol):.3f} best={best_err:.4f}"
                )
            else:
                ctx.log(
                    f"{log_prefix} DONE arms={arms} "
                    + " ".join(f"{a}={err_map[a]:.4f}" for a in arms)
                )
            hold_action = {}
            for a in arms:
                hold_q = cur_map[a].tolist()
                for i in hold_target:
                    hold_q[int(i)] = float(targets[a][int(i)])
                hold_action[f"arm_{a}"] = hold_q
                hold_action[f"gripper_{a}"] = _gripper_cmd(world, a, open_gripper)
            for _ in range(5):
                yield _make_legacy_arm_action(world, **hold_action)
            return {"ok": True, "accepted": accepted, "err_map": err_map}
        if stalled:
            ctx.log(
                f"{log_prefix} STALL arms={arms} "
                + " ".join(f"{a}={err_map[a]:.4f}" for a in arms)
                + f" best={best_err:.4f} no_improve={now - best_err_t:.1f}s"
            )
            return {"ok": False, "stalled": True, "err_map": err_map}

        action = {}
        status_parts = []
        for a in arms:
            cur = cur_map[a]
            err = targets[a] - cur
            dq = np.zeros(7, dtype=np.float64)
            for i in active:
                dq[i] = float(np.clip(err[i], -max_dq_per_step, max_dq_per_step))
            q_next = (cur + dq).tolist()
            for i in hold_target:
                q_next[int(i)] = float(targets[a][int(i)])
            action[f"arm_{a}"] = q_next
            action[f"gripper_{a}"] = _gripper_cmd(world, a, open_gripper)
            status_parts.append(f"{a}={err_map[a]:.3f}")

        if now - last_log > 0.5:
            ctx.set_status(f"{log_prefix} " + " ".join(status_parts))
            last_log = now
        if now - last_file_log > 2.0:
            ctx.log(f"{log_prefix} step t={now - t_start:.1f}s " + " ".join(status_parts))
            last_file_log = now
        yield _make_legacy_arm_action(world, **action)


def _force_set_arm_qpos(world, arm: str, target_qpos: np.ndarray) -> list[int]:
    """Diagnostic-only joint injection; forbidden during challenge execution."""
    if _challenge_action_only_enabled() and not getattr(world, "dry_run", False):
        raise RuntimeError(
            "direct arm qpos injection is disabled in challenge mode; "
            "use controller actions"
        )
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    _prepare_legacy_7dof_motion(world, arm, stage_name="force_set_arm_qpos")
    q = world.robot.get_joint_positions()
    try:
        q_new = q.clone()
    except Exception:
        q_new = np.asarray(q, dtype=np.float64).copy()
    idx = _get_arm_dof_idx(world, arm)
    target = np.asarray(target_qpos, dtype=np.float64).reshape(7)
    for local_i, joint_i in enumerate(idx):
        q_new[int(joint_i)] = float(target[int(local_i)])
    world.robot.set_joint_positions(q_new)
    try:
        v0 = world.robot.get_joint_velocities()
        v_new = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
        for joint_i in idx:
            v_new[int(joint_i)] = 0.0
        world.robot.set_joint_velocities(v_new)
    except Exception:
        pass
    try:
        world.set_arm_pin_qpos(arm, target)
    except Exception:
        pass
    return [int(j) for j in idx]


_FINGER_OPEN_FALLBACK_QPOS = 0.05


def _gripper_joint_targets(world, arm: str) -> list[tuple[int, str, float]]:
    """Return (joint_index, joint_name, open_qpos) for the arm finger joints."""
    if getattr(world, "dry_run", False):
        return []
    robot = world.robot
    names = list(robot.joints.keys())
    joint_names: list[str] = []
    try:
        joint_names.extend(list(getattr(robot, "finger_joint_names", {}).get(arm, [])))
    except Exception:
        pass
    joint_names.extend([
        f"{arm}_gripper_finger_joint1",
        f"{arm}_gripper_finger_joint2",
    ])
    try:
        for fj in getattr(robot, "finger_joints", {}).get(arm, []):
            jn = getattr(fj, "joint_name", None) or getattr(fj, "name", None)
            if jn:
                joint_names.append(str(jn))
    except Exception:
        pass

    out: list[tuple[int, str, float]] = []
    seen: set[str] = set()
    for jn in joint_names:
        jn = str(jn)
        if not jn or jn in seen or jn not in names:
            continue
        seen.add(jn)
        joint = robot.joints[jn]
        try:
            open_q = float(getattr(joint, "upper_limit"))
        except Exception:
            open_q = _FINGER_OPEN_FALLBACK_QPOS
        if not np.isfinite(open_q):
            open_q = _FINGER_OPEN_FALLBACK_QPOS
        out.append((int(names.index(jn)), jn, open_q))
    return out


def _gripper_open_override(world, arm: str) -> list[float]:
    """Controller command for a fully open gripper, with scalar fallback."""
    release_keepalive = getattr(
        world,
        "release_gripper_close_keepalive",
        None,
    )
    if callable(release_keepalive):
        release_keepalive(arm)
    targets = _gripper_joint_targets(world, arm)
    if targets:
        vals = [float(q_open) for _, _, q_open in targets]
        try:
            idx = world.controller_action_idx(f"gripper_{arm}")
            if len(idx) == len(vals):
                return vals
        except Exception:
            pass
    return [1.0]


def _gripper_cmd(world, arm: str, open_gripper: bool) -> list[float]:
    return _gripper_open_override(world, arm) if bool(open_gripper) else [-1.0]


def _force_open_gripper_qpos(world, arm: str) -> list[tuple[str, float]]:
    """Diagnostic-only finger injection; forbidden during challenge execution."""
    if getattr(world, "dry_run", False):
        return []
    if _challenge_action_only_enabled():
        raise RuntimeError(
            "direct gripper qpos injection is disabled in challenge mode; "
            "use gripper controller actions"
        )
    targets = _gripper_joint_targets(world, arm)
    if not targets:
        return []
    robot = world.robot
    q0 = robot.get_joint_positions()
    q = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
    for j_idx, _, q_open in targets:
        q[int(j_idx)] = float(q_open)
    robot.set_joint_positions(q)
    try:
        v0 = robot.get_joint_velocities()
        v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
        for j_idx, _, _ in targets:
            v[int(j_idx)] = 0.0
        robot.set_joint_velocities(v)
    except Exception:
        pass
    try:
        world.set_gripper_pin_qpos(arm, [float(q_open) for _, _, q_open in targets])
    except Exception:
        pass
    return [(jn, float(q_open)) for _, jn, q_open in targets]


def _gripper_qpos_map(world, arms: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for a in arms:
        vals = None
        if not getattr(world, "dry_run", False):
            try:
                robot = world.robot
                qpos = robot.get_joint_positions()
                vals = [float(qpos[j_idx]) for j_idx, _, _ in _gripper_joint_targets(world, a)]
            except Exception:
                vals = None
        if vals is None:
            try:
                vals = world.gripper_qpos_list(a)
            except Exception:
                vals = None
        out[a] = [round(float(x), 5) for x in (vals or [])]
    return out


def _yield_force_open_grippers(ctx, arms: list[str], *, frames: int, log_prefix: str):
    """Open grippers using repeated controller commands only."""
    world = ctx.world
    from behavior_interface.skills.eef import (
        _assert_legacy_7dof_motion_ready,
        _ensure_world_pinned_actions,
    )

    try:
        _ensure_world_pinned_actions(world)
    except Exception:
        pass
    arms = [a for a in ("left", "right") if a in set(arms)]
    if not arms:
        return {}
    for _ in range(max(1, int(frames))):
        action = {}
        for a in arms:
            action[f"gripper_{a}"] = _gripper_open_override(world, a)
            _assert_legacy_7dof_motion_ready(world, a)
        yield world.make_action(**action)
    qpos = _gripper_qpos_map(world, arms)
    ctx.log(f"{log_prefix} gripper controller-open qpos={qpos}")
    return qpos


def _snap_arms_to_targets(
    ctx,
    targets: dict[str, np.ndarray],
    *,
    open_gripper: bool,
    log_prefix: str,
    hold_frames: int = 4,
):
    """Compatibility wrapper that now converges through controller actions only."""
    world = ctx.world
    if not targets:
        return {"ok": True, "err_map": {}}
    arms = [a for a in ("left", "right") if a in targets]
    normalized = {
        a: np.asarray(targets[a], dtype=np.float64).reshape(7)
        for a in arms
    }
    ctx.log(f"{log_prefix} controller convergence arms={arms}")
    result = yield from _arms_move_parallel(
        ctx,
        normalized,
        open_gripper=open_gripper,
        max_dq_per_step=0.10,
        tol=0.05,
        timeout_s=max(8.0, float(hold_frames) * 1.5),
        log_prefix=f"{log_prefix}/action_only",
        accept_tol=None,
    )
    action = {}
    for a in arms:
        action[f"arm_{a}"] = normalized[a].tolist()
        action[f"gripper_{a}"] = _gripper_cmd(world, a, open_gripper)
    for _ in range(max(1, int(hold_frames))):
        yield _make_legacy_arm_action(world, **action)
    return result


@register_skill(
    "arm_reset",
    description=(
        "把手臂复位，base/trunk 保持不动。"
        "arm='right'|'left'|'both'（默认 right）。"
        "mode='grasp'（默认，抓取预备位，避免升降撑到底盘）"
        "|'hang'（自然下垂=所有关节归零，即 scene reset 初始状态）"
        "|'ready'（弯肘就绪姿态，方便抓取）。"
        "open_gripper=True 时同步张开夹爪（默认 True）。"
    ),
)
def arm_reset(
    ctx,
    arm: str = "right",
    mode: str = "grasp",
    open_gripper: bool = True,
    max_dq_per_step: float = 0.50,
    tol: float = 0.05,
    timeout_s: float = 20.0,
):
    """yield 每步 action 给 sim。"""
    arm = arm.lower().strip()
    if arm not in ("left", "right", "both"):
        ctx.log(f"arm_reset: arm 参数错误 '{arm}'，应为 left/right/both")
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        return

    mode = mode.lower().strip()
    # 兼容旧别名
    if mode in ("untucked", "tucked"):
        mode = "hang"
    if mode in ("grasp", "grasp_position", "grasp_prep"):
        yield from set_arm_to_grasp_position(
            ctx,
            arm=arm,
            open_gripper=open_gripper,
            max_dq_per_step=max(float(max_dq_per_step), _GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP),
            tol=max(float(tol), 0.08),
            timeout_s=max(float(timeout_s), 80.0),
        )
        return
    if mode not in ("hang", "ready"):
        ctx.log(f"arm_reset: mode 错误 '{mode}'，应为 grasp/hang/ready")
        ctx.set_result({"ok": False, "error": f"bad mode '{mode}'"})
        return

    arms_todo = ["left", "right"] if arm == "both" else [arm]
    ctx.log(f"arm_reset arms={arms_todo} mode={mode} open_gripper={open_gripper}")
    targets_used = {}
    for a in arms_todo:
        target = (_HANG_ARM_QPOS.copy() if mode == "hang"
                  else _READY_ARM[a].copy())
        targets_used[a] = target.tolist()
        ctx.log(
            f"arm_reset[{a}] target_qpos=[{','.join(f'{v:+.2f}' for v in target)}]"
        )
        yield from _arm_reset_one(
            ctx, a, target,
            open_gripper=open_gripper,
            max_dq_per_step=max_dq_per_step,
            tol=tol, timeout_s=timeout_s,
        )

    ctx.set_result({
        "ok": True,
        "arms_reset": arms_todo,
        "mode": mode,
        "open_gripper": open_gripper,
        "target_qpos": targets_used,
    })


def _grasp_prep_shoulder_hang_target_qpos(phase1_q: np.ndarray) -> np.ndarray:
    """在 Phase1 v2 肘腕角上冻结 j4-j7，仅将肩 j1-j3 对齐 hang。"""
    target = np.asarray(phase1_q, dtype=np.float64).reshape(7).copy()
    for i in _UPPER_ARM_DOF:
        target[i] = float(_HANG_ARM_QPOS[i])
    return target


def _phase1_v2_grasp_target_qpos(world, arm: str) -> np.ndarray:
    """固定 Phase1 中间位；肘腕与 reset grasp prep 完全一致，避免 IK 翻支。"""
    return _GRASP_PHASE1_ARM_QPOS.copy()


def _active_joint_err(
    q: np.ndarray,
    target: np.ndarray,
    active_dof: tuple[int, ...],
) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(7)
    tgt = np.asarray(target, dtype=np.float64).reshape(7)
    return float(max(abs(float(q[i]) - float(tgt[i])) for i in active_dof))


def _arms_phase1_elbow_short(
    world,
    arms: list[str],
    p1_map: dict[str, np.ndarray],
    *,
    tol: float,
) -> list[tuple[str, float]]:
    """返回肘腕 j4-j7 未达 Phase1 目标的臂及误差。"""
    bad: list[tuple[str, float]] = []
    for a in arms:
        try:
            q = _arm_qpos(world, a)
        except Exception:
            bad.append((a, float("inf")))
            continue
        err = _active_joint_err(q, p1_map[a], _LOWER_ARM_DOF)
        if err >= tol:
            bad.append((a, err))
    return bad


def _is_at_grasp_prep(
    q: np.ndarray,
    phase1_q: np.ndarray,
    final_q: np.ndarray,
    *,
    tol: float = 0.05,
) -> bool:
    """是否已在抓取预备终态：肩 hang + 肘腕与当前 Phase1 v2 一致。"""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    fq = np.asarray(final_q, dtype=np.float64).reshape(7)
    p1 = np.asarray(phase1_q, dtype=np.float64).reshape(7)
    if float(np.linalg.norm(q - fq, ord=np.inf)) >= tol:
        return False
    if float(np.linalg.norm(q[3:7] - p1[3:7], ord=np.inf)) >= tol:
        return False
    return True


def _grasp_prep_verify(
    world,
    arm: str,
    phase1_q: np.ndarray,
    final_q: np.ndarray,
    *,
    tol: float,
) -> dict:
    """读取当前关节并对比 grasp prep 目标。"""
    q = _arm_qpos(world, arm)
    fq = np.asarray(final_q, dtype=np.float64).reshape(7)
    err_final = float(np.linalg.norm(q - fq, ord=np.inf))
    ok = _is_at_grasp_prep(q, phase1_q, final_q, tol=tol)
    return {
        "ok": ok,
        "qpos": [round(float(x), 4) for x in q],
        "err_final_rad": round(err_final, 4),
        "target_qpos": [round(float(x), 4) for x in fq],
    }


def _normalize_grasp_gripper_mode(gripper=None, open_gripper=None) -> str:
    """Map legacy open_gripper and new gripper strings onto open/keep."""
    if gripper is None:
        if open_gripper is None:
            return "keep"
        return "open" if bool(open_gripper) else "keep"
    g = str(gripper).strip().lower()
    if g in ("open", "opened", "true", "1", "yes"):
        return "open"
    if g in ("keep", "hold", "lock", "locked", "false", "0", "no"):
        return "keep"
    return "keep"


def _current_gripper_qpos(world, arm: str) -> list[float] | None:
    """Read raw finger qpos for keep mode."""
    if getattr(world, "dry_run", False):
        return None
    try:
        vals = world.gripper_qpos_list(arm)
        if vals is not None:
            out = [float(x) for x in vals]
            if out:
                return out
    except Exception:
        pass
    try:
        robot = world.robot
        qpos = robot.get_joint_positions()
        out = [float(qpos[j_idx]) for j_idx, _, _ in _gripper_joint_targets(world, arm)]
        return out or None
    except Exception:
        return None


def _target_eef_pose_from_arm_qpos(world, arm: str, target_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probe FK for target_qpos without yielding a sim frame."""
    target = np.asarray(target_qpos, dtype=np.float64).reshape(7)
    saved = None
    saved_vel = None
    try:
        saved = world.robot.get_joint_positions()
        try:
            saved_vel = world.robot.get_joint_velocities()
        except Exception:
            saved_vel = None
        try:
            q = saved.clone()
        except Exception:
            q = np.asarray(saved, dtype=np.float64).copy()
        idx = _get_arm_dof_idx(world, arm)
        for local_i, joint_i in enumerate(idx):
            q[int(joint_i)] = float(target[int(local_i)])
        world.robot.set_joint_positions(q)
        eef = world.eef_pose(arm=arm)
        pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3).copy()
        quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4).copy()
        return pos, quat
    finally:
        if saved is not None:
            try:
                world.robot.set_joint_positions(saved)
            except Exception:
                pass
        if saved_vel is not None:
            try:
                world.robot.set_joint_velocities(saved_vel)
            except Exception:
                pass


def _quat_angle_deg_xyzw(q0, q1) -> float:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    n0 = float(np.linalg.norm(q0))
    n1 = float(np.linalg.norm(q1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    dot = abs(float(np.dot(q0 / n0, q1 / n1)))
    return float(math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))))


def _grasp_position_keep_ori_verify(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    pos_tol_m: float = _GRASP_PREP_KEEP_ORI_POS_TOL_M,
    ori_tol_deg: float = _GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG,
) -> dict:
    eef = world.eef_pose(arm=arm)
    pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
    target_pos_np = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat_np = np.asarray(target_quat, dtype=np.float64).reshape(4)
    pos_err = float(np.linalg.norm(pos - target_pos_np))
    ori_err = float(_quat_angle_deg_xyzw(quat, target_quat_np))
    return {
        "ok": bool(pos_err <= float(pos_tol_m) and ori_err <= float(ori_tol_deg)),
        "eef_pos": [round(float(x), 6) for x in pos],
        "eef_quat": [round(float(x), 6) for x in quat],
        "target_pos": [round(float(x), 6) for x in target_pos_np],
        "target_quat": [round(float(x), 6) for x in target_quat_np],
        "pos_err_m": round(pos_err, 6),
        "ori_err_deg": round(ori_err, 4),
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
    }


def _grasp_j1234_keep_ori_verify(
    world,
    arm: str,
    target_q1234,
    target_quat,
    *,
    joint_tol_rad: float,
    ori_tol_deg: float = _GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG,
) -> dict:
    q = _arm_qpos(world, arm)
    target_q1234_np = np.asarray(target_q1234, dtype=np.float64).reshape(4)
    eef = world.eef_pose(arm=arm)
    eef_pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    eef_quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
    target_quat_np = np.asarray(target_quat, dtype=np.float64).reshape(4)
    joint_err = float(
        np.linalg.norm(q[:4] - target_q1234_np, ord=np.inf)
    )
    ori_err = float(_quat_angle_deg_xyzw(eef_quat, target_quat_np))
    return {
        "ok": bool(
            joint_err <= float(joint_tol_rad)
            and ori_err <= float(ori_tol_deg)
        ),
        "qpos": [round(float(x), 5) for x in q],
        "j1234": [round(float(x), 5) for x in q[:4]],
        "j567": [round(float(x), 5) for x in q[4:7]],
        "target_j1234": [
            round(float(x), 5) for x in target_q1234_np
        ],
        "j1234_err_inf_rad": round(joint_err, 5),
        "joint_tol_rad": float(joint_tol_rad),
        "eef_pos": [round(float(x), 6) for x in eef_pos],
        "eef_quat": [round(float(x), 6) for x in eef_quat],
        "target_quat": [round(float(x), 6) for x in target_quat_np],
        "ori_err_deg": round(ori_err, 4),
        "ori_tol_deg": float(ori_tol_deg),
    }


def _solve_orientation_primary_position_q(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    seed_q,
    nominal_q,
    pos_tol_m: float = _GRASP_PREP_KEEP_ORI_POS_TOL_M,
    ori_tol_deg: float = _GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
    max_steps: int = 96,
    max_dq_per_step: float = 0.08,
) -> tuple[np.ndarray | None, float, float, int]:
    """Strict task-priority IK: orientation first, position in Jw nullspace."""
    from behavior_interface.skills.grasp import (
        _arm_qpos as _grasp_arm_qpos,
        _orientation_error_omega,
        _quat_to_mat,
        _read_jacobian_arm,
        _set_arm_qpos_direct,
    )
    from behavior_interface.skills.reset_body import (
        _orientation_primary_nullspace_dq,
    )

    target_pos_np = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat_np = np.asarray(target_quat, dtype=np.float64).reshape(4)
    target_quat_np /= max(float(np.linalg.norm(target_quat_np)), 1e-12)
    target_R = _quat_to_mat(target_quat_np)
    nominal = np.asarray(nominal_q, dtype=np.float64).reshape(7)
    lo, hi = _arm_joint_limits(world, arm)
    q = np.clip(np.asarray(seed_q, dtype=np.float64).reshape(7), lo, hi)
    robot = world.robot
    saved = None
    saved_vel = None
    best_q = q.copy()
    best_pos_err = float("inf")
    best_ori_err = float("inf")
    best_nominal_err = float("inf")
    min_rank = 3

    def _better(
        ori_err: float,
        pos_err: float,
        nominal_err: float,
        ref_ori_err: float,
        ref_pos_err: float,
        ref_nominal_err: float,
    ) -> bool:
        if ori_err > float(ori_tol_deg) or ref_ori_err > float(ori_tol_deg):
            if ori_err < ref_ori_err - 1e-7:
                return True
            if ref_ori_err < ori_err - 1e-7:
                return False
        if pos_err < ref_pos_err - 1e-6:
            return True
        if ref_pos_err < pos_err - 1e-6:
            return False
        if ori_err < ref_ori_err - 1e-5:
            return True
        if ref_ori_err < ori_err - 1e-5:
            return False
        return nominal_err < ref_nominal_err - 1e-8

    def _measure(q_probe: np.ndarray):
        _set_arm_qpos_direct(world, arm, q_probe)
        eef = world.eef_pose(arm=arm)
        pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
        cur_R = _quat_to_mat(quat)
        omega = _orientation_error_omega(target_R, cur_R)
        pos_err = float(np.linalg.norm(target_pos_np - pos))
        ori_err = float(math.degrees(np.linalg.norm(omega)))
        nominal_err = float(np.linalg.norm(q_probe - nominal, ord=np.inf))
        return pos, omega, pos_err, ori_err, nominal_err

    try:
        raw_saved = robot.get_joint_positions()
        try:
            saved = raw_saved.clone()
        except Exception:
            saved = np.asarray(raw_saved, dtype=np.float64).copy()
        try:
            raw_vel = robot.get_joint_velocities()
            try:
                saved_vel = raw_vel.clone()
            except Exception:
                saved_vel = np.asarray(raw_vel, dtype=np.float64).copy()
        except Exception:
            saved_vel = None

        no_improve = 0
        for _step in range(max(1, int(max_steps))):
            pos, omega, pos_err, ori_err, nominal_err = _measure(q)
            if _better(
                ori_err,
                pos_err,
                nominal_err,
                best_ori_err,
                best_pos_err,
                best_nominal_err,
            ):
                best_q = q.copy()
                best_pos_err = pos_err
                best_ori_err = ori_err
                best_nominal_err = nominal_err
            if (
                ori_err <= float(ori_tol_deg)
                and pos_err <= float(pos_tol_m)
            ):
                return q.copy(), pos_err, ori_err, int(min_rank)

            omega_norm = float(np.linalg.norm(omega))
            if omega_norm > 0.12:
                omega = omega * (0.12 / (omega_norm + 1e-12))
            dx = target_pos_np - pos
            dx_norm = float(np.linalg.norm(dx))
            if dx_norm > 0.025:
                dx = dx * (0.025 / (dx_norm + 1e-12))

            J, _ = _read_jacobian_arm(world, arm)
            q_cur = np.asarray(_grasp_arm_qpos(world, arm), dtype=np.float64).reshape(7)
            dq, dq_primary, rank = _orientation_primary_nullspace_dq(
                J,
                omega,
                dx,
                np.clip(nominal - q_cur, -0.12, 0.12),
                position_weight=1.0,
                joint_weight=0.001,
            )
            min_rank = min(min_rank, int(rank))

            candidates: list[np.ndarray] = []
            for dq_raw, scales in (
                (dq, (1.0, 0.5, 0.25)),
                (dq_primary, (1.0, 0.5)),
            ):
                dq_try = np.asarray(dq_raw, dtype=np.float64).reshape(7).copy()
                for i in range(7):
                    if (q_cur[i] >= hi[i] - 1e-4 and dq_try[i] > 0.0) or (
                        q_cur[i] <= lo[i] + 1e-4 and dq_try[i] < 0.0
                    ):
                        dq_try[i] = 0.0
                dq_inf = float(np.linalg.norm(dq_try, ord=np.inf))
                if dq_inf > float(max_dq_per_step):
                    dq_try *= float(max_dq_per_step) / (dq_inf + 1e-12)
                for scale in scales:
                    q_try = np.clip(q_cur + float(scale) * dq_try, lo, hi)
                    if float(np.linalg.norm(q_try - q_cur, ord=np.inf)) > 1e-8:
                        candidates.append(q_try)

            next_q = None
            next_pos_err = pos_err
            next_ori_err = ori_err
            next_nominal_err = nominal_err
            for q_try in candidates:
                _, _, pos_try, ori_try, nominal_try = _measure(q_try)
                if _better(
                    ori_try,
                    pos_try,
                    nominal_try,
                    next_ori_err,
                    next_pos_err,
                    next_nominal_err,
                ):
                    next_q = q_try.copy()
                    next_pos_err = pos_try
                    next_ori_err = ori_try
                    next_nominal_err = nominal_try
            if next_q is None:
                no_improve += 1
                if no_improve >= 4:
                    break
                q = best_q.copy()
                continue
            q = next_q
            no_improve = 0
    except Exception:
        return None, float("inf"), float("inf"), int(min_rank)
    finally:
        if saved is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass
        if saved_vel is not None:
            try:
                robot.set_joint_velocities(saved_vel)
            except Exception:
                pass

    if (
        best_ori_err <= float(ori_tol_deg)
        and best_pos_err <= float(pos_tol_m)
    ):
        return best_q.copy(), best_pos_err, best_ori_err, int(min_rank)
    return None, best_pos_err, best_ori_err, int(min_rank)


def _append_unique_seed(seeds: list[tuple[str, np.ndarray]], label: str, q) -> None:
    arr = np.asarray(q, dtype=np.float64).reshape(7).copy()
    for _, old in seeds:
        if float(np.linalg.norm(arr - old, ord=np.inf)) < 1e-4:
            return
    seeds.append((label, arr))


def _target_branch_limit_rad(t: float) -> float:
    """Tighten branch proximity as waypoints approach grasp-prep q."""
    tt = float(t)
    if tt < 0.62:
        return float("inf")
    u = float(np.clip((tt - 0.62) / 0.38, 0.0, 1.0))
    return float((1.25 * (1.0 - u)) + (0.18 * u))


def _adjacent_branch_limit_rad(t: float) -> float:
    tt = float(t)
    if tt < 0.75:
        return 0.95
    if tt < 0.92:
        return 0.78
    return 0.62


def _plan_grasp_prep_line_ik(
    ctx,
    arm: str,
    target_qpos: np.ndarray,
    *,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 5.0,
    keep_ori_quat=None,
) -> tuple[list[dict], dict]:
    """Plan current EEF -> grasp-prep EEF as branch-continuous IK anchors."""
    from behavior_interface.skills.eef import (
        _eef_solve_6d_dls_arm_q,
        _quat_normalize_xyzw,
        _quat_slerp_xyzw,
    )

    world = ctx.world
    target_q = np.asarray(target_qpos, dtype=np.float64).reshape(7).copy()
    start_eef = world.eef_pose(arm=arm)
    start_pos = np.asarray(start_eef["pos"], dtype=np.float64).reshape(3)
    start_quat = _quat_normalize_xyzw(start_eef["quat"])
    start_q = _arm_qpos(world, arm)
    target_pos, grasp_target_quat = _target_eef_pose_from_arm_qpos(world, arm, target_q)
    grasp_target_quat = _quat_normalize_xyzw(grasp_target_quat)
    keep_ori = keep_ori_quat is not None
    target_quat = (
        _quat_normalize_xyzw(keep_ori_quat)
        if keep_ori
        else grasp_target_quat
    )
    dist = float(np.linalg.norm(target_pos - start_pos))
    rot_deg = _quat_angle_deg_xyzw(start_quat, target_quat)
    base_n = max(
        12 if keep_ori else 10,
        int(math.ceil(
            dist / (
                _GRASP_PREP_KEEP_ORI_WAYPOINT_STEP_M
                if keep_ori
                else 0.040
            )
        )),
        int(math.ceil(rot_deg / 9.0)),
    )
    base_n = min(48 if keep_ori else 34, max(12 if keep_ori else 10, base_n))
    ts = [float(x) for x in np.linspace(1.0 / base_n, 1.0, base_n)]
    ts.extend([0.62, 0.70, 0.78, 0.86, 0.92, 0.96, 0.985, 1.0])
    ts = sorted({round(float(np.clip(t, 0.0, 1.0)), 5) for t in ts if t > 1e-6})

    anchors: list[dict] = []
    dropped: list[dict] = []
    last_q = start_q.copy()
    meta = {
        "start_pos": [round(float(x), 5) for x in start_pos],
        "target_pos": [round(float(x), 5) for x in target_pos],
        "target_quat": [round(float(x), 6) for x in target_quat],
        "dist_m": float(dist),
        "rot_deg": float(rot_deg),
        "waypoints": len(ts),
        "accepted": 0,
        "dropped": dropped,
        "keep_ori": bool(keep_ori),
        "keep_ori_policy": (
            "orientation_primary_position_nullspace"
            if keep_ori
            else "slerp_to_grasp_prep_orientation"
        ),
        "keep_ori_solve_tol_deg": (
            _GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG if keep_ori else None
        ),
        "target_qpos": [round(float(x), 5) for x in target_q],
        "grasp_target_quat": [round(float(x), 6) for x in grasp_target_quat],
        "final_target_gap_required_rad": None if keep_ori else 0.18,
        "orientation_rank_min": 3 if keep_ori else None,
    }
    ctx.log(
        f"set_arm_to_grasp_position[{arm}] offline branch IK "
        f"waypoints={len(ts)} dist={dist*100:.1f}cm rot={rot_deg:.1f}deg "
        f"keep_ori={keep_ori}"
    )

    for wi, t in enumerate(ts, start=1):
        is_final = bool(t >= 0.99999)
        wp_pos = start_pos + (target_pos - start_pos) * float(t)
        wp_quat = (
            target_quat.copy()
            if keep_ori
            else _quat_slerp_xyzw(start_quat, target_quat, float(t))
        )
        adj_limit = min(
            _GRASP_PREP_KEEP_ORI_MAX_DQ_RAD * 4.0,
            _adjacent_branch_limit_rad(float(t)),
        ) if keep_ori else _adjacent_branch_limit_rad(float(t))
        target_limit = (
            float("inf")
            if keep_ori
            else _target_branch_limit_rad(float(t))
        )
        if is_final and not keep_ori:
            q_sol = target_q.copy()
            pos_err = 0.0
            ori_err = 0.0
            adj_gap = float(np.linalg.norm(q_sol - last_q, ord=np.inf))
            if adj_gap > adj_limit:
                reason = "final_branch_gap"
                dropped.append({
                    "wp": wi,
                    "t": float(t),
                    "reason": reason,
                    "adj_gap_rad": round(adj_gap, 4),
                    "adj_limit_rad": round(adj_limit, 4),
                })
                meta["error"] = reason
                ctx.log(
                    f"set_arm_to_grasp_position[{arm}] wp {wi}/{len(ts)} FAIL "
                    f"final gap={adj_gap:.3f}>{adj_limit:.3f}rad"
                )
                return [], meta
            anchors.append({
                "wp": wi,
                "t": float(t),
                "q": q_sol,
                "pos": target_pos.copy(),
                "quat": target_quat.copy(),
                "pos_err": pos_err,
                "ori_err": ori_err,
                "joint_gap": adj_gap,
                "target_gap": 0.0,
                "is_final": True,
                "seed": "target_q_exact",
            })
            last_q = q_sol.copy()
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] wp {wi}/{len(ts)} keep final "
                f"gap={adj_gap:.3f}rad target_gap=0.000"
            )
            continue

        seeds: list[tuple[str, np.ndarray]] = []
        _append_unique_seed(seeds, "prev", last_q)
        _append_unique_seed(seeds, "linear_q", start_q + (target_q - start_q) * float(t))
        if not keep_ori and t >= 0.45:
            toward = last_q + (target_q - last_q) * min(0.65, max(0.25, (float(t) - 0.45) / 0.55))
            _append_unique_seed(seeds, "prev_to_target", toward)
        if not keep_ori and t >= 0.58:
            _append_unique_seed(seeds, "target_branch", target_q)

        candidates: list[dict] = []
        reject_notes: list[str] = []
        for label, seed in seeds:
            orientation_rank = None
            if keep_ori:
                q_try, solve_pos_err, solve_ori_err, orientation_rank = (
                    _solve_orientation_primary_position_q(
                        world,
                        arm,
                        wp_pos,
                        wp_quat,
                        seed_q=seed,
                        nominal_q=target_q,
                        pos_tol_m=max(
                            float(pos_tol),
                            _GRASP_PREP_KEEP_ORI_POS_TOL_M,
                        ),
                        ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
                        max_steps=120 if t >= 0.82 else 96,
                        max_dq_per_step=0.070,
                    )
                )
            else:
                wp_pos_tol = max(float(pos_tol), 0.026 if t >= 0.82 else 0.040)
                wp_ori_tol = max(float(ori_tol_deg), 10.0 if t >= 0.82 else 20.0)
                q_try, solve_pos_err, solve_ori_err = _eef_solve_6d_dls_arm_q(
                    world,
                    arm,
                    wp_pos,
                    wp_quat,
                    pos_tol=wp_pos_tol,
                    ori_tol_deg=wp_ori_tol,
                    max_steps=260 if t >= 0.82 else 210,
                    max_dq_per_step=0.060 if t >= 0.82 else 0.075,
                    max_dx_per_step=0.018 if t >= 0.82 else 0.026,
                    max_dw_per_step=0.080 if t >= 0.82 else 0.120,
                    ori_weight=0.70 if t >= 0.82 else 0.52,
                    lam=0.08,
                    seed_q=seed,
                )
            if q_try is None:
                reject_notes.append(f"{label}:ik_fail {solve_pos_err*1000:.0f}mm/{solve_ori_err:.1f}deg")
                continue
            adj_gap = float(np.linalg.norm(q_try - last_q, ord=np.inf))
            target_gap = float(np.linalg.norm(q_try - target_q, ord=np.inf))
            if adj_gap > adj_limit:
                reject_notes.append(f"{label}:adj {adj_gap:.2f}>{adj_limit:.2f}")
                continue
            if target_gap > target_limit:
                reject_notes.append(f"{label}:target {target_gap:.2f}>{target_limit:.2f}")
                continue
            near_weight = 0.2 + 3.0 * max(0.0, float(t) - 0.55)
            score = (
                adj_gap * 4.0
                + target_gap * near_weight
                + float(solve_pos_err) * 18.0
                + float(solve_ori_err) * (2.0 if keep_ori else 0.015)
            )
            candidates.append({
                "q": np.asarray(q_try, dtype=np.float64).reshape(7).copy(),
                "seed": label,
                "score": float(score),
                "pos_err": float(solve_pos_err),
                "ori_err": float(solve_ori_err),
                "joint_gap": adj_gap,
                "target_gap": target_gap,
                "orientation_rank": orientation_rank,
            })

        if not candidates:
            reason = "no_same_branch_ik"
            dropped.append({
                "wp": wi,
                "t": float(t),
                "reason": reason,
                "adj_limit_rad": round(adj_limit, 4),
                "target_limit_rad": None if math.isinf(target_limit) else round(target_limit, 4),
                "rejects": reject_notes[:8],
            })
            meta["error"] = reason
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] wp {wi}/{len(ts)} FAIL "
                f"t={t:.3f} rejects={'; '.join(reject_notes[:5])}"
            )
            return [], meta

        best = min(candidates, key=lambda c: c["score"])
        anchors.append({
            "wp": wi,
            "t": float(t),
            "q": best["q"],
            "pos": wp_pos.copy(),
            "quat": wp_quat.copy(),
            "pos_err": best["pos_err"],
            "ori_err": best["ori_err"],
            "joint_gap": best["joint_gap"],
            "target_gap": best["target_gap"],
            "is_final": bool(is_final),
            "seed": best["seed"],
            "orientation_rank": best.get("orientation_rank"),
        })
        if keep_ori and best.get("orientation_rank") is not None:
            meta["orientation_rank_min"] = min(
                int(meta["orientation_rank_min"]),
                int(best["orientation_rank"]),
            )
        last_q = best["q"].copy()
        ctx.log(
            f"set_arm_to_grasp_position[{arm}] wp {wi}/{len(ts)} keep "
            f"seed={best['seed']} pos={best['pos_err']*1000:.0f}mm "
            f"ori={best['ori_err']:.1f}deg adj={best['joint_gap']:.3f}rad "
            f"target_gap={best['target_gap']:.3f}rad"
        )

    meta["accepted"] = len(anchors)
    return anchors, meta


def _compress_grasp_prep_line_anchors(
    anchors: list[dict],
    *,
    max_play_anchors: int = 6,
) -> tuple[list[dict], dict]:
    """Keep the full IK branch solve, but play only evenly spaced key anchors."""
    n = len(anchors)
    meta = {
        "enabled": False,
        "from": n,
        "to": n,
        "t": [round(float(a.get("t", 0.0)), 4) for a in anchors],
    }
    if n <= max(2, int(max_play_anchors)):
        return anchors, meta
    keep = {
        int(round(x))
        for x in np.linspace(0, n - 1, max(2, int(max_play_anchors)))
    }
    keep.add(n - 1)
    out: list[dict] = []
    for i in sorted(keep):
        a = dict(anchors[int(i)])
        old_label = str(a.get("label") or a.get("seed") or f"wp{a.get('wp', i + 1)}")
        a["label"] = f"line_key_{i + 1}_of_{n}:{old_label}"
        # Non-final anchors are branch waypoints, not precision stops.  The
        # final verification below still enforces the requested grasp-prep q.
        if not bool(a.get("is_final")):
            a["settle_max_frames"] = 0
        out.append(a)
    meta = {
        "enabled": True,
        "from": n,
        "to": len(out),
        "indices_1based": [int(i) + 1 for i in sorted(keep)],
        "t": [round(float(a.get("t", 0.0)), 4) for a in out],
    }
    return out, meta


def _plan_grasp_prep_jointspace_fallback(
    ctx,
    arm: str,
    phase1_qpos: np.ndarray,
    final_qpos: np.ndarray,
    *,
    tol: float,
    reason: str,
) -> tuple[list[dict], dict]:
    """Fallback for FK-reachable poses when the Cartesian EEF line is not IK-feasible."""
    world = ctx.world
    start_q = _arm_qpos(world, arm)
    phase1_q = np.asarray(phase1_qpos, dtype=np.float64).reshape(7).copy()
    final_q = np.asarray(final_qpos, dtype=np.float64).reshape(7).copy()
    lo, hi = _arm_joint_limits(world, arm)
    anchors: list[dict] = []
    lower_err0 = _active_joint_err(start_q, final_q, _LOWER_ARM_DOF)
    min_gap = max(float(tol) * 0.35, 0.025)

    def _clipped(q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64).reshape(7), lo, hi)

    def _append(
        label: str,
        q_raw: np.ndarray,
        ref_q: np.ndarray,
        *,
        is_final: bool = False,
        max_dq: float | None = None,
        settle_max_frames: int | None = None,
        strong_servo: bool = False,
    ) -> np.ndarray:
        q = _clipped(q_raw)
        gap = float(np.linalg.norm(q - ref_q, ord=np.inf))
        if gap < min_gap and not is_final:
            return ref_q
        rec = {
            "q": q,
            "seed": label,
            "label": label,
            "joint_gap": gap,
            "is_final": bool(is_final),
        }
        if max_dq is not None:
            rec["max_dq"] = float(max_dq)
        if settle_max_frames is not None:
            rec["settle_max_frames"] = int(settle_max_frames)
        if strong_servo:
            rec["strong_servo"] = True
        anchors.append(rec)
        return q

    ref_q = start_q.copy()
    high_j3_escape = bool(abs(float(ref_q[2])) >= _GRASP_PREP_HIGH_J3_ESCAPE_RAD)

    # Stage the recovery so shoulder and elbow do not make large coupled moves.
    # This keeps the motion real (action playback) while avoiding the failed
    # phase1->final branch that can trap j3 high and j4 half folded.
    if high_j3_escape:
        wrist_q = ref_q.copy()
        wrist_q[4:7] = 0.0
        ref_q = _append(
            "jointspace_wrist_neutral_escape",
            wrist_q,
            ref_q,
            max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD,
            settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES,
            strong_servo=True,
        )

        elbow_escape_q = ref_q.copy()
        elbow_escape_q[3] = _GRASP_PREP_ELBOW_UNFOLD_ESCAPE_Q
        elbow_escape_q[4:7] = 0.0
        ref_q = _append(
            "jointspace_elbow_unfold_escape",
            elbow_escape_q,
            ref_q,
            max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD,
            settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES,
            strong_servo=True,
        )

        j1_escape_q = ref_q.copy()
        j1_escape_q[0] = phase1_q[0]
        ref_q = _append(
            "jointspace_j1_escape",
            j1_escape_q,
            ref_q,
            max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD,
            settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES,
            strong_servo=True,
        )

        shoulder_q = ref_q.copy()
        shoulder_q[1] = final_q[1]
        shoulder_q[2] = final_q[2]
        ref_q = _append(
            "jointspace_high_j3_release",
            shoulder_q,
            ref_q,
            max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD,
            settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES,
            strong_servo=True,
        )
    else:
        wrist_q = ref_q.copy()
        wrist_q[4:7] = final_q[4:7]
        ref_q = _append("jointspace_wrist_align", wrist_q, ref_q)

        shoulder_q = ref_q.copy()
        shoulder_q[1] = final_q[1]
        shoulder_q[2] = final_q[2]
        ref_q = _append("jointspace_shoulder_release", shoulder_q, ref_q)

    elbow_q = ref_q.copy()
    elbow_q[3] = final_q[3]
    elbow_q[4:7] = final_q[4:7]
    ref_q = _append(
        "jointspace_elbow_fold",
        elbow_q,
        ref_q,
        max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD if high_j3_escape else None,
        settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES if high_j3_escape else None,
        strong_servo=high_j3_escape,
    )

    j1_q = ref_q.copy()
    j1_q[0] = final_q[0]
    ref_q = _append(
        "jointspace_j1_center",
        j1_q,
        ref_q,
        max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD if high_j3_escape else None,
        settle_max_frames=_GRASP_PREP_ESCAPE_SETTLE_MAX_FRAMES if high_j3_escape else None,
        strong_servo=high_j3_escape,
    )

    final_q = _clipped(final_q)
    ref_q = _append(
        "jointspace_final",
        final_q,
        ref_q,
        is_final=True,
        max_dq=_GRASP_PREP_ESCAPE_MAX_DQ_RAD if high_j3_escape else None,
        strong_servo=high_j3_escape,
    )

    if len(anchors) == 1 and bool(anchors[0].get("is_final")):
        anchors[0]["label"] = "jointspace_direct_final"
        anchors[0]["seed"] = "jointspace_direct_final"

    meta = {
        "fallback": True,
        "fallback_reason": str(reason),
        "mode": "jointspace_staged_wrist_shoulder_elbow_final",
        "accepted": len(anchors),
        "lower_err0_rad": round(float(lower_err0), 4),
        "high_j3_escape": high_j3_escape,
        "labels": [str(a.get("label") or a.get("seed")) for a in anchors],
        "start_qpos": [round(float(x), 4) for x in start_q],
        "phase1_qpos": [round(float(x), 4) for x in phase1_q],
        "target_qpos": [round(float(x), 4) for x in final_q],
    }
    ctx.log(
        f"set_arm_to_grasp_position[{arm}] EEF-line IK failed ({reason}); "
        f"fallback to staged actual jointspace motion anchors={len(anchors)} "
        f"labels={meta['labels']} lower_err={lower_err0:.3f}rad"
    )
    return anchors, meta


def _grasp_anchor_frame_count(gap: float, max_dq_per_frame: float) -> int:
    """Size a smoothstep segment so no interpolation step exceeds max_dq."""
    gap = max(0.0, float(gap))
    max_dq = max(1e-6, float(max_dq_per_frame))
    return max(
        1,
        int(math.ceil(_GRASP_PREP_SMOOTHSTEP_MAX_SLOPE * gap / max_dq)),
    )


def _grasp_line_play_anchors(
    ctx,
    arm: str,
    anchors: list[dict],
    *,
    gripper_mode: str,
    max_dq_per_frame: float,
    timeout_s: float,
    final_tol: float = _GRASP_PREP_FINAL_SETTLE_TOL_RAD,
    keep_ori_quat=None,
    keep_ori_tracking_tol_deg: float = _GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG,
    keep_ori_abort_deg: float = _GRASP_PREP_KEEP_ORI_ABORT_DEG,
) -> dict:
    """Play IK anchors with joint-space interpolation and independent gripper lock."""
    world = ctx.world
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    _prepare_legacy_7dof_motion(
        world,
        arm,
        ctx=ctx,
        stage_name=f"set_arm_to_grasp_position.{arm}.prepare",
    )
    t0 = time.monotonic()
    keep_ori_target = None
    if keep_ori_quat is not None:
        keep_ori_target = np.asarray(
            keep_ori_quat,
            dtype=np.float64,
        ).reshape(4)
        keep_ori_target /= max(float(np.linalg.norm(keep_ori_target)), 1e-12)
        max_dq = max(
            0.015,
            min(_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD, float(max_dq_per_frame)),
        )
    else:
        max_dq = max(
            _GRASP_PREP_FAST_MIN_DQ_RAD,
            min(_GRASP_PREP_JOINTSPACE_MAX_DQ_RAD, float(max_dq_per_frame)),
        )
    keep_ori_path_max_err_deg = 0.0
    keep_ori_samples = 0

    def _observe_keep_ori() -> float:
        nonlocal keep_ori_path_max_err_deg, keep_ori_samples
        if keep_ori_target is None:
            return 0.0
        try:
            eef = world.eef_pose(arm=arm)
            err = float(
                _quat_angle_deg_xyzw(
                    np.asarray(eef["quat"], dtype=np.float64).reshape(4),
                    keep_ori_target,
                )
            )
        except Exception:
            err = float("inf")
        keep_ori_path_max_err_deg = max(keep_ori_path_max_err_deg, err)
        keep_ori_samples += 1
        return err

    def _keep_ori_report() -> dict:
        if keep_ori_target is None:
            return {}
        return {
            "keep_ori": True,
            "keep_ori_quat": [
                round(float(x), 6) for x in keep_ori_target
            ],
            "keep_ori_tracking_tol_deg": float(keep_ori_tracking_tol_deg),
            "keep_ori_abort_deg": float(keep_ori_abort_deg),
            "keep_ori_path_max_err_deg": (
                None
                if not np.isfinite(keep_ori_path_max_err_deg)
                else round(float(keep_ori_path_max_err_deg), 4)
            ),
            "keep_ori_samples": int(keep_ori_samples),
        }

    def _orientation_abort(err_deg: float, frames_total: int) -> dict | None:
        if keep_ori_target is None or err_deg <= float(keep_ori_abort_deg):
            return None
        ctx.log(
            f"set_arm_to_grasp_position[{arm}] keep_ori tracking abort "
            f"ori={err_deg:.2f}deg>{float(keep_ori_abort_deg):.2f}deg "
            f"frame={frames_total}"
        )
        return {
            "ok": False,
            "keep_ori_violation": True,
            "error": "keep_ori_tracking_abort",
            "ori_err_deg": round(float(err_deg), 4),
            "frames": int(frames_total),
            **_keep_ori_report(),
        }

    keep_qpos = _current_gripper_qpos(world, arm)
    if gripper_mode == "keep" and keep_qpos is not None:
        try:
            world.set_gripper_pin_qpos(arm, keep_qpos)
        except Exception:
            pass
    frames_total = 0
    for ai, anchor in enumerate(anchors, start=1):
        try:
            if ctx.is_cancelled():
                return {"ok": False, "cancelled": True, "frames": frames_total}
        except Exception:
            pass
        q1 = np.asarray(anchor["q"], dtype=np.float64).reshape(7)
        q0 = _arm_qpos(world, arm)
        gap = float(np.linalg.norm(q1 - q0, ord=np.inf))
        if gap < 0.010 and not bool(anchor.get("is_final")):
            label = str(anchor.get("label") or anchor.get("seed") or f"anchor{ai}")
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] skip anchor {ai}/{len(anchors)} "
                f"{label} gap={gap:.4f}rad"
            )
            continue
        anchor_max_dq = max_dq
        try:
            if anchor.get("max_dq") is not None:
                # Explicit escape-anchor caps are safety limits and may be
                # lower than the ordinary fast-path floor.
                anchor_max_dq = max(
                    1e-3,
                    min(max_dq, float(anchor["max_dq"])),
                )
        except Exception:
            anchor_max_dq = max_dq
        strong_servo = bool(anchor.get("strong_servo"))
        n_frames = _grasp_anchor_frame_count(gap, anchor_max_dq)
        label = str(anchor.get("label") or anchor.get("seed") or f"anchor{ai}")
        ctx.log(
            f"set_arm_to_grasp_position[{arm}] play anchor {ai}/{len(anchors)} "
            f"{label} gap={gap:.3f}rad frames={n_frames} "
            f"dq={anchor_max_dq:.3f} strong={strong_servo} gripper={gripper_mode}"
        )
        for fi in range(1, n_frames + 1):
            try:
                if ctx.is_cancelled():
                    return {
                        "ok": False,
                        "cancelled": True,
                        "frames": frames_total,
                        "anchor": ai,
                        "anchors_total": len(anchors),
                    }
            except Exception:
                pass
            if time.monotonic() - t0 > float(timeout_s):
                return {
                    "ok": False,
                    "timeout": True,
                    "frames": frames_total,
                    "anchor": ai,
                    "anchors_total": len(anchors),
                    "frame_in_anchor": fi,
                    "frames_in_anchor": n_frames,
                }
            s = fi / n_frames
            ss = s * s * (3.0 - 2.0 * s)
            q = q1 if strong_servo else q0 + (q1 - q0) * float(ss)
            action = {f"arm_{arm}": q.tolist()}
            if gripper_mode == "open":
                action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
            elif keep_qpos is not None:
                try:
                    world.set_gripper_pin_qpos(arm, keep_qpos)
                except Exception:
                    pass
                action[f"gripper_{arm}"] = keep_qpos
            yield _make_legacy_arm_action(world, **action)
            frames_total += 1
            ori_err_now = _observe_keep_ori()
            ori_abort = _orientation_abort(ori_err_now, frames_total)
            if ori_abort is not None:
                return ori_abort

        is_final_anchor = bool(anchor.get("is_final"))
        settle_tol = (
            max(0.018, float(final_tol))
            if is_final_anchor
            else _GRASP_PREP_INTERMEDIATE_SETTLE_TOL_RAD
        )
        settle_max_frames = (
            _GRASP_PREP_FINAL_SETTLE_MAX_FRAMES
            if is_final_anchor
            else _GRASP_PREP_INTERMEDIATE_SETTLE_MAX_FRAMES
        )
        try:
            if anchor.get("settle_max_frames") is not None:
                settle_max_frames = max(settle_max_frames, int(anchor["settle_max_frames"]))
        except Exception:
            pass
        stable = 0
        settle_frames = 0
        while stable < _GRASP_PREP_SETTLE_STABLE_FRAMES:
            try:
                if ctx.is_cancelled():
                    return {
                        "ok": False,
                        "cancelled": True,
                        "frames": frames_total,
                        "anchor": ai,
                        "anchors_total": len(anchors),
                    }
            except Exception:
                pass
            if time.monotonic() - t0 > float(timeout_s):
                try:
                    err_inf_now = float(np.linalg.norm(q1 - _arm_qpos(world, arm), ord=np.inf))
                except Exception:
                    err_inf_now = float("inf")
                return {
                    "ok": False,
                    "timeout": True,
                    "frames": frames_total,
                    "anchor": ai,
                    "anchors_total": len(anchors),
                    "settling": True,
                    "settle_frames": settle_frames,
                    "err_inf": round(float(err_inf_now), 5),
                }
            cur = _arm_qpos(world, arm)
            err = q1 - cur
            err_inf = float(np.linalg.norm(err, ord=np.inf))
            if err_inf <= settle_tol:
                stable += 1
            else:
                stable = 0
            if stable >= _GRASP_PREP_SETTLE_STABLE_FRAMES:
                break
            if settle_max_frames <= 0 and not is_final_anchor:
                break
            if settle_frames >= settle_max_frames:
                if is_final_anchor:
                    ctx.log(
                        f"set_arm_to_grasp_position[{arm}] final anchor settle limit "
                        f"frames={settle_frames} err={err_inf:.3f}rad; verify/fallback"
                    )
                    return {
                        "ok": False,
                        "timeout": False,
                        "settle_limit": True,
                        "frames": frames_total,
                        "anchor": ai,
                        "anchors_total": len(anchors),
                        "settle_frames": settle_frames,
                        "err_inf": round(float(err_inf), 5),
                    }
                ctx.log(
                    f"set_arm_to_grasp_position[{arm}] settle anchor {ai}/{len(anchors)} "
                    f"limit frames={settle_frames} err={err_inf:.3f}rad; continue"
                )
                break
            if strong_servo:
                q_next = q1.tolist()
            else:
                dq = np.clip(err, -anchor_max_dq, anchor_max_dq)
                q_next = (cur + dq).tolist()
            action = {f"arm_{arm}": q_next}
            if gripper_mode == "open":
                action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
            elif keep_qpos is not None:
                try:
                    world.set_gripper_pin_qpos(arm, keep_qpos)
                except Exception:
                    pass
                action[f"gripper_{arm}"] = keep_qpos
            yield _make_legacy_arm_action(world, **action)
            frames_total += 1
            settle_frames += 1
            ori_err_now = _observe_keep_ori()
            ori_abort = _orientation_abort(ori_err_now, frames_total)
            if ori_abort is not None:
                return ori_abort
        if settle_frames:
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] settle anchor {ai}/{len(anchors)} "
                f"frames={settle_frames} tol={settle_tol:.3f}rad"
            )

    final_q = np.asarray(anchors[-1]["q"], dtype=np.float64).reshape(7)
    try:
        world.set_arm_pin_qpos(arm, final_q)
    except Exception:
        pass
    final_tol = max(0.018, float(final_tol))
    try:
        cur_final = _arm_qpos(world, arm)
        final_err0 = float(np.linalg.norm(final_q - cur_final, ord=np.inf))
    except Exception:
        final_err0 = float("inf")
    if final_err0 <= final_tol:
        if gripper_mode == "open":
            world.set_gripper_pin_qpos(
                arm, _gripper_open_override(world, arm)
            )
        elif keep_qpos is not None:
            try:
                world.set_gripper_pin_qpos(arm, keep_qpos)
            except Exception:
                pass
        ctx.log(
            f"set_arm_to_grasp_position[{arm}] final already within tol "
            f"err={final_err0:.4f}rad tol={final_tol:.3f}rad; return OK"
        )
        return {
            "ok": True,
            "frames": frames_total,
            "final_settle_frames": 0,
            "final_err_inf": round(float(final_err0), 5),
            "final_settle_skipped": True,
            "keep_gripper_qpos": [round(float(x), 5) for x in (keep_qpos or [])],
            **_keep_ori_report(),
        }
    for _ in range(1):
        try:
            if ctx.is_cancelled():
                return {"ok": False, "cancelled": True, "frames": frames_total}
        except Exception:
            pass
        action = {f"arm_{arm}": final_q.tolist()}
        if gripper_mode == "open":
            action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
        elif keep_qpos is not None:
            try:
                world.set_gripper_pin_qpos(arm, keep_qpos)
            except Exception:
                pass
            action[f"gripper_{arm}"] = keep_qpos
        yield _make_legacy_arm_action(world, **action)
        frames_total += 1
        ori_err_now = _observe_keep_ori()
        ori_abort = _orientation_abort(ori_err_now, frames_total)
        if ori_abort is not None:
            return ori_abort
    final_stable = 0
    final_settle_frames = 0
    while final_stable < _GRASP_PREP_SETTLE_STABLE_FRAMES:
        try:
            if ctx.is_cancelled():
                return {"ok": False, "cancelled": True, "frames": frames_total}
        except Exception:
            pass
        if time.monotonic() - t0 > float(timeout_s):
            return {
                "ok": False,
                "timeout": True,
                "frames": frames_total,
                "final_settling": True,
                "final_settle_frames": final_settle_frames,
                "err_inf": round(float(err_inf), 5) if "err_inf" in locals() else None,
            }
        cur = _arm_qpos(world, arm)
        err = final_q - cur
        err_inf = float(np.linalg.norm(err, ord=np.inf))
        if err_inf <= final_tol:
            final_stable += 1
        else:
            final_stable = 0
        if final_stable >= _GRASP_PREP_SETTLE_STABLE_FRAMES:
            break
        if final_settle_frames >= _GRASP_PREP_FINAL_SETTLE_MAX_FRAMES:
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] final settle limit "
                f"frames={final_settle_frames} err={err_inf:.3f}rad; verify/fallback"
            )
            return {
                "ok": False,
                "timeout": False,
                "final_settle_limit": True,
                "frames": frames_total,
                "final_settle_frames": final_settle_frames,
                "err_inf": round(float(err_inf), 5),
            }
        dq = np.clip(err, -max_dq, max_dq)
        q_next = (cur + dq).tolist()
        action = {f"arm_{arm}": q_next}
        if gripper_mode == "open":
            action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
        elif keep_qpos is not None:
            try:
                world.set_gripper_pin_qpos(arm, keep_qpos)
            except Exception:
                pass
            action[f"gripper_{arm}"] = keep_qpos
        yield _make_legacy_arm_action(world, **action)
        frames_total += 1
        final_settle_frames += 1
        ori_err_now = _observe_keep_ori()
        ori_abort = _orientation_abort(ori_err_now, frames_total)
        if ori_abort is not None:
            return ori_abort
    if final_settle_frames:
        ctx.log(
            f"set_arm_to_grasp_position[{arm}] final settle "
            f"frames={final_settle_frames} tol={final_tol:.3f}rad"
        )
    return {
        "ok": True,
        "frames": frames_total,
        "final_settle_frames": final_settle_frames,
        "keep_gripper_qpos": [round(float(x), 5) for x in (keep_qpos or [])],
        **_keep_ori_report(),
    }


def _clamp_keep_ori_wrist_step(
    solution: np.ndarray,
    seed: np.ndarray,
    result: dict,
) -> np.ndarray:
    """把腕部指令限制在单帧步长内，超限时朝解的方向走满一步。

    以前超限直接判 branch_jump 并冻结腕部，J1-J4 却继续走，姿态误差只会越
    滚越大、下一帧更解不出来。改成走截断步后，剩余偏差由后续帧继续收敛。
    """
    sol = np.asarray(solution, dtype=np.float64).reshape(-1)
    seed_np = np.asarray(seed, dtype=np.float64).reshape(sol.shape)
    limit = _GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD
    delta = sol - seed_np
    raw_step = float(np.linalg.norm(delta, ord=np.inf))
    result["wrist_solution_step_inf_rad"] = raw_step
    if raw_step > limit:
        sol = seed_np + delta * (limit / raw_step)
        result["wrist_command_clamped"] = True
    else:
        result["wrist_command_clamped"] = False
    result["wrist_command_step_inf_rad"] = float(
        np.linalg.norm(sol - seed_np, ord=np.inf)
    )
    return sol


def _solve_grasp_keep_ori_j567_command(
    world,
    arm: str,
    q1234_command,
    target_rpy_deg,
    seed_j567,
) -> tuple[np.ndarray | None, dict]:
    """7DOF / 无独立 J8 时的 keep_ori 腕部求解（仅 J5-J7）。"""
    from behavior_interface.skills.wrist_j567_solver import (
        solve_wrist_j567_for_eef_rpy,
    )

    q1234 = np.asarray(q1234_command, dtype=np.float64).reshape(4)
    seed = np.asarray(seed_j567, dtype=np.float64).reshape(3)
    step_limit = _GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD
    # 第一遍只在 seed 的单帧步长邻域内找解，解天然连续。
    result = solve_wrist_j567_for_eef_rpy(
        world,
        arm,
        target_rpy_deg,
        fixed_q1234=q1234,
        seed_j567=seed,
        ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
        max_steps=24,
        max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
        global_search=False,
        wrist_step_limit_rad=step_limit,
    )
    result["global_search_fallback"] = False
    result["wrist_search_limited"] = True
    if not result.get("ok", False):
        # 邻域内无解才放开全关节域；此时解可能落在远处分支，下面按步长截断。
        result = solve_wrist_j567_for_eef_rpy(
            world,
            arm,
            target_rpy_deg,
            fixed_q1234=q1234,
            seed_j567=seed,
            ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
            max_steps=72,
            max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
            global_search=True,
        )
        result["global_search_fallback"] = True
        result["wrist_search_limited"] = False
    if not result.get("ok", False):
        return None, result

    j567 = np.asarray(result["j567_rad"], dtype=np.float64).reshape(3)
    j567 = _clamp_keep_ori_wrist_step(j567, seed, result)
    command = np.concatenate([q1234, j567]).astype(np.float64)
    return command, result


def _solve_grasp_keep_ori_j5678_command(
    world,
    arm: str,
    q1234_command,
    target_rpy_deg,
    seed_j5678,
) -> tuple[np.ndarray | None, float | None, dict]:
    """8DOF keep_ori：固定 J1-J4，用 J5-J8 追入口 RPY。

    返回 (arm_q7_command, j8_command, solve_result)。

    当前 keep_ori 不走这里：J8 一律锁 0，姿态只由 J5-J7 追。保留它是为了
    重新启用 J8 跟踪时可以直接接回播放器。
    """
    from behavior_interface.skills.wrist_j567_solver import (
        solve_wrist_j5678_for_eef_rpy,
    )

    q1234 = np.asarray(q1234_command, dtype=np.float64).reshape(4)
    seed = np.asarray(seed_j5678, dtype=np.float64).reshape(4)
    step_limit = _GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD
    # 第一遍只在 seed 的单帧步长邻域内找解。J5-J8 冗余，不收紧搜索域的话
    # 解会沿零空间漂移到远处分支，表现为腕部乱甩。
    result = solve_wrist_j5678_for_eef_rpy(
        world,
        arm,
        target_rpy_deg,
        fixed_q1234=q1234,
        seed_j5678=seed,
        ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
        max_steps=24,
        max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
        global_search=False,
        wrist_step_limit_rad=step_limit,
    )
    result["global_search_fallback"] = False
    result["wrist_search_limited"] = True
    if not result.get("ok", False):
        # 邻域内无解才放开全关节域；此时解可能落在远处分支，下面按步长截断。
        result = solve_wrist_j5678_for_eef_rpy(
            world,
            arm,
            target_rpy_deg,
            fixed_q1234=q1234,
            seed_j5678=seed,
            ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
            max_steps=72,
            max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
            global_search=True,
        )
        result["global_search_fallback"] = True
        result["wrist_search_limited"] = False
    if not result.get("ok", False):
        return None, None, result

    j5678 = np.asarray(result["j5678_rad"], dtype=np.float64).reshape(4)
    j5678 = _clamp_keep_ori_wrist_step(j5678, seed, result)
    command = np.concatenate([q1234, j5678[:3]]).astype(np.float64)
    return command, float(j5678[3]), result


def _grasp_line_play_anchors_j567_keep_ori(
    ctx,
    arm: str,
    anchors: list[dict],
    *,
    gripper_mode: str,
    max_dq_per_frame: float,
    timeout_s: float,
    target_rpy_deg,
    target_quat,
    final_tol: float = _GRASP_PREP_FINAL_SETTLE_TOL_RAD,
) -> dict:
    """Keep nominal J1-J4 untouched; J5-J7 追入口 RPY，J8 保持锁 0。"""
    world = ctx.world
    from behavior_interface.skills.eef import (
        _ensure_world_pinned_actions,
        _prepare_legacy_7dof_motion,
    )

    _ensure_world_pinned_actions(world)
    _prepare_legacy_7dof_motion(
        world,
        arm,
        ctx=ctx,
        stage_name=f"set_arm_to_grasp_position.{arm}.j567_keep_ori.prepare",
    )
    # J8 由 skill 入口通过 controller action 归零并锁定，这里不再下发它的指令。
    keep_ori_policy = "nominal_j1234_unchanged_j567_feedback_tracking"
    target_rpy = np.asarray(target_rpy_deg, dtype=np.float64).reshape(3)
    target_quat_np = np.asarray(target_quat, dtype=np.float64).reshape(4)
    target_quat_np /= max(float(np.linalg.norm(target_quat_np)), 1e-12)
    # J1-J4 插值粒度必须远小于腕部单帧步长预算，否则腕部追不上机体带来的
    # 姿态变化；这里不能沿用 fast 路径 0.28rad 的下限。
    max_dq = max(
        _GRASP_PREP_KEEP_ORI_MIN_DQ_RAD,
        min(_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD, float(max_dq_per_frame)),
    )
    keep_qpos = _current_gripper_qpos(world, arm)
    if gripper_mode == "keep" and keep_qpos is not None:
        try:
            world.set_gripper_pin_qpos(arm, keep_qpos)
        except Exception:
            pass
    t0 = time.monotonic()
    frames_total = 0
    solve_count = 0
    global_search_count = 0
    solve_max_err_deg = 0.0
    wrist_max_step_rad = 0.0
    wrist_clamped_frames = 0
    path_max_ori_err_deg = 0.0
    wrist_feedback_frames = 0
    wrist_feedback_max_ideal_gap_rad = 0.0
    wrist_feedback_max_compensation_rad = 0.0
    j1234_nominal_command_max_error_rad = 0.0
    solver_failures = 0
    trajectory_frames = 0
    final_settle_frames = 0
    last_solver_result: dict = {}
    last_command_q = _arm_qpos(world, arm)
    last_j567 = last_command_q[4:7].copy()
    arm_lo, arm_hi = _arm_joint_limits(world, arm)
    # 实测腕角可能因为之前的失控绕出限位（线上见过 J5 到 -10.27rad）。它会成为
    # 第一帧的指令种子，必须先钳回可下发范围，否则第一帧就发出越限指令。
    last_j567 = np.clip(last_j567, arm_lo[4:7], arm_hi[4:7])
    last_command_q[4:7] = last_j567
    nominal_baseline_q = last_command_q.copy()

    def _cancelled() -> bool:
        try:
            return bool(ctx.is_cancelled())
        except Exception:
            return False

    def _timed_out() -> bool:
        return bool(time.monotonic() - t0 > float(timeout_s))

    def _seed_wrist_command() -> np.ndarray:
        """求解种子取上一帧的腕部指令，而不是实测。

        实测滞后于指令，用实测当种子会把跟踪滞后误判成分支跳变，并让相邻帧
        的解在实测振荡下来回跳。
        """
        return np.asarray(last_j567, dtype=np.float64).reshape(3).copy()

    def _solve_command(
        measured_q1234,
        *,
        output_q1234,
        seed_wrist=None,
    ):
        nonlocal solve_count
        nonlocal global_search_count
        nonlocal solve_max_err_deg
        nonlocal wrist_max_step_rad
        nonlocal wrist_clamped_frames
        nonlocal last_solver_result
        nonlocal last_command_q
        nonlocal last_j567

        seed = (
            _seed_wrist_command()
            if seed_wrist is None
            else np.asarray(seed_wrist, dtype=np.float64).reshape(-1)[:3]
        )
        command, solve = _solve_grasp_keep_ori_j567_command(
            world,
            arm,
            measured_q1234,
            target_rpy,
            seed,
        )
        solve_count += 1
        if solve.get("global_search_fallback"):
            global_search_count += 1
        solve_err = float(solve.get("ori_err_deg", float("inf")))
        if np.isfinite(solve_err):
            solve_max_err_deg = max(solve_max_err_deg, solve_err)
        wrist_step = float(
            solve.get("wrist_command_step_inf_rad", float("inf"))
        )
        if np.isfinite(wrist_step):
            wrist_max_step_rad = max(wrist_max_step_rad, wrist_step)
        if solve.get("wrist_command_clamped"):
            wrist_clamped_frames += 1
        last_solver_result = dict(solve)
        if command is None:
            return None
        last_command_q = np.asarray(command, dtype=np.float64).reshape(7)
        last_command_q[:4] = np.asarray(
            output_q1234,
            dtype=np.float64,
        ).reshape(4)
        last_j567 = last_command_q[4:7].copy()
        return last_command_q.copy()

    def _apply_wrist_feedback(
        command_q: np.ndarray,
        measured_q: np.ndarray,
        *,
        frame_number: int,
    ) -> np.ndarray:
        nonlocal wrist_feedback_frames
        nonlocal wrist_feedback_max_ideal_gap_rad
        nonlocal wrist_feedback_max_compensation_rad
        nonlocal last_command_q
        nonlocal last_j567
        nonlocal last_solver_result

        command = np.asarray(command_q, dtype=np.float64).reshape(7).copy()
        measured = np.asarray(measured_q, dtype=np.float64).reshape(7)
        ideal = command[4:7].copy()
        measured_wrist = measured[4:7].copy()
        wrist_lo = arm_lo[4:7]
        wrist_hi = arm_hi[4:7]
        ideal_wrist_error = ideal - measured_wrist
        ideal_wrist_gap = float(
            np.linalg.norm(ideal_wrist_error, ord=np.inf)
        )
        gain = min(
            _GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_MAX_GAIN,
            _GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_P_GAIN
            + _GRASP_PREP_KEEP_ORI_WRIST_FEEDBACK_GAIN_RAMP
            * float(max(0, frame_number - 1)),
        )
        compensated = ideal + gain * ideal_wrist_error
        compensated = np.clip(compensated, wrist_lo, wrist_hi)
        compensated_step = np.clip(
            compensated - measured_wrist,
            -_GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD,
            _GRASP_PREP_KEEP_ORI_MAX_WRIST_COMMAND_STEP_RAD,
        )
        compensated = measured_wrist + compensated_step
        compensation = float(
            np.linalg.norm(compensated - ideal, ord=np.inf)
        )
        command[4:7] = compensated[:3]
        wrist_feedback_frames += 1
        wrist_feedback_max_ideal_gap_rad = max(
            wrist_feedback_max_ideal_gap_rad,
            ideal_wrist_gap,
        )
        wrist_feedback_max_compensation_rad = max(
            wrist_feedback_max_compensation_rad,
            compensation,
        )
        last_command_q = command.copy()
        last_j567 = command[4:7].copy()
        last_solver_result = {
            **last_solver_result,
            "controller_compensation_gain": round(float(gain), 4),
            "ideal_wrist_rad": [float(x) for x in ideal],
            "measured_wrist_rad": [float(x) for x in measured_wrist],
            "compensated_wrist_command_rad": [
                float(x) for x in compensated
            ],
            "ideal_wrist_gap_inf_rad": ideal_wrist_gap,
            "command_compensation_inf_rad": compensation,
        }
        return command

    def _make_action(command_q: np.ndarray) -> dict:
        # 不下发 tool_roll_*：J8 已锁 0，由 world 的 pin 机制保持。
        action = {f"arm_{arm}": command_q.tolist()}
        if gripper_mode == "open":
            action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
        elif keep_qpos is not None:
            try:
                world.set_gripper_pin_qpos(arm, keep_qpos)
            except Exception:
                pass
            action[f"gripper_{arm}"] = keep_qpos
        return action

    def _observe_orientation() -> float:
        nonlocal path_max_ori_err_deg
        try:
            quat = np.asarray(
                world.eef_pose(arm=arm)["quat"],
                dtype=np.float64,
            ).reshape(4)
            err = float(_quat_angle_deg_xyzw(quat, target_quat_np))
        except Exception:
            err = float("inf")
        path_max_ori_err_deg = max(path_max_ori_err_deg, err)
        return err

    def _report(ok: bool, **extra) -> dict:
        return {
            "ok": bool(ok),
            "frames": int(frames_total),
            "keep_ori": True,
            "keep_ori_policy": keep_ori_policy,
            "uses_j8": False,
            "j8_locked_rad": 0.0,
            "target_rpy_deg": [
                round(float(x), 5) for x in target_rpy
            ],
            "target_quat": [
                round(float(x), 6) for x in target_quat_np
            ],
            "j567_solve_count": int(solve_count),
            "j567_global_search_count": int(global_search_count),
            "j567_solve_max_err_deg": round(float(solve_max_err_deg), 4),
            "j567_wrist_max_step_rad": round(float(wrist_max_step_rad), 5),
            "wrist_command_clamped_frames": int(wrist_clamped_frames),
            "keep_ori_play_max_dq_rad": round(float(max_dq), 5),
            "keep_ori_path_max_err_deg": (
                None
                if not np.isfinite(path_max_ori_err_deg)
                else round(float(path_max_ori_err_deg), 4)
            ),
            "wrist_feedback_frames": int(wrist_feedback_frames),
            "wrist_feedback_max_ideal_gap_rad": round(
                float(wrist_feedback_max_ideal_gap_rad), 5
            ),
            "wrist_feedback_max_compensation_rad": round(
                float(wrist_feedback_max_compensation_rad), 5
            ),
            "j1234_nominal_command_max_error_rad": round(
                float(j1234_nominal_command_max_error_rad), 12
            ),
            "j567_solver_failures": int(solver_failures),
            "trajectory_frames": int(trajectory_frames),
            "final_settle_frames": int(final_settle_frames),
            "final_j1234_tol_rad": float(
                _GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD
            ),
            "final_q_command": [
                round(float(x), 6) for x in last_command_q
            ],
            "last_solver_result": last_solver_result,
            "keep_gripper_qpos": [
                round(float(x), 5) for x in (keep_qpos or [])
            ],
            **extra,
        }

    def _finish(report: dict) -> dict:
        return report

    try:
        for anchor_index, anchor in enumerate(anchors, start=1):
            if _cancelled():
                return _finish(_report(False, cancelled=True))
            nominal_target = np.asarray(
                anchor["q"],
                dtype=np.float64,
            ).reshape(7)
            # Immutable counterfactual baseline: this is the same full-joint
            # anchor interpolation that would be played without the wrist overlay.
            # Measured wrist feedback must never feed back into its timing or J1-J4.
            nominal_start = nominal_baseline_q.copy()
            gap = float(
                np.linalg.norm(nominal_target - nominal_start, ord=np.inf)
            )
            is_final = bool(anchor.get("is_final"))
            if gap < 0.010 and not is_final:
                continue
            strong_servo = bool(anchor.get("strong_servo"))
            anchor_max_dq = max_dq
            try:
                if anchor.get("max_dq") is not None:
                    anchor_max_dq = max(
                        _GRASP_PREP_KEEP_ORI_MIN_DQ_RAD,
                        min(max_dq, float(anchor["max_dq"])),
                    )
            except Exception:
                anchor_max_dq = max_dq
            frame_count = _grasp_anchor_frame_count(gap, anchor_max_dq)
            label = str(
                anchor.get("label")
                or anchor.get("seed")
                or f"anchor{anchor_index}"
            )
            ctx.log(
                f"set_arm_to_grasp_position[{arm}] keep_ori J1234 anchor "
                f"{anchor_index}/{len(anchors)} {label} gap={gap:.3f}rad "
                f"frames={frame_count} dq={anchor_max_dq:.3f} "
                f"wrist=J567 j8=locked0"
            )

            for frame_index in range(1, frame_count + 1):
                if _cancelled():
                    return _finish(_report(
                        False,
                        cancelled=True,
                        anchor=anchor_index,
                        frame_in_anchor=frame_index,
                    ))
                if _timed_out():
                    return _finish(_report(
                        False,
                        timeout=True,
                        anchor=anchor_index,
                        frame_in_anchor=frame_index,
                    ))
                s = float(frame_index) / float(frame_count)
                smooth = s * s * (3.0 - 2.0 * s)
                if strong_servo:
                    q1234_nominal = nominal_target[:4]
                else:
                    q1234_nominal = (
                        nominal_start[:4]
                        + (nominal_target[:4] - nominal_start[:4]) * smooth
                    )
                measured_q = _arm_qpos(world, arm)
                # 前馈：用本帧要发出的 nominal J1-J4 求解，让 (J1-J4, J5-J7)
                # 指令组合自洽。用实测 J1-J4 求解会解出配合"滞后机体"的腕角，
                # 配上前跑的 J1-J4 指令在物理上并不满足目标姿态。实测偏差由
                # 下面的 _apply_wrist_feedback 单独补偿。
                command_q = _solve_command(
                    q1234_nominal,
                    output_q1234=q1234_nominal,
                    seed_wrist=_seed_wrist_command(),
                )
                if command_q is None:
                    solver_failures += 1
                    if solver_failures <= 5 or solver_failures % 10 == 0:
                        ctx.log(
                            f"set_arm_to_grasp_position[{arm}] keep_ori J567 "
                            f"solve failed anchor={anchor_index} "
                            f"frame={frame_index}/{frame_count}; keep exact "
                            f"nominal J1234 and previous wrist "
                            f"result={last_solver_result}"
                        )
                    command_q = np.concatenate(
                        [q1234_nominal, last_j567]
                    ).astype(np.float64)
                else:
                    command_q = _apply_wrist_feedback(
                        command_q,
                        measured_q,
                        frame_number=frame_index,
                    )
                j1234_command_error = float(
                    np.linalg.norm(
                        command_q[:4] - q1234_nominal,
                        ord=np.inf,
                    )
                )
                j1234_nominal_command_max_error_rad = max(
                    j1234_nominal_command_max_error_rad,
                    j1234_command_error,
                )
                if j1234_command_error > 1e-10:
                    return _finish(_report(
                        False,
                        error="keep_ori_modified_nominal_j1234",
                        anchor=anchor_index,
                        frame_in_anchor=frame_index,
                        j1234_command_error_rad=j1234_command_error,
                    ))
                yield _make_legacy_arm_action(
                    world,
                    **_make_action(command_q),
                )
                frames_total += 1
                trajectory_frames += 1
                ori_err = _observe_orientation()
                if (
                    ori_err > _GRASP_PREP_KEEP_ORI_ABORT_DEG
                    and (
                        frame_index == 1
                        or frame_index == frame_count
                        or frame_index % 4 == 0
                    )
                ):
                    ctx.log(
                        f"set_arm_to_grasp_position[{arm}] keep_ori path "
                        f"ori={ori_err:.2f}deg; J1-J4 nominal path unchanged"
                    )

            nominal_baseline_q = nominal_target.copy()

            if not is_final:
                continue

            settle_limit = _GRASP_PREP_KEEP_ORI_FINAL_SETTLE_MAX_FRAMES
            stable = 0
            settle_tol = _GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD
            while stable < _GRASP_PREP_KEEP_ORI_FINAL_STABLE_FRAMES:
                if _cancelled():
                    return _finish(_report(
                        False, cancelled=True, final_settling=True
                    ))
                if _timed_out():
                    return _finish(_report(
                        False,
                        timeout=True,
                        final_settling=True,
                        final_settle_frames=final_settle_frames,
                    ))
                current = _arm_qpos(world, arm)
                if final_settle_frames >= settle_limit:
                    err_inf = float(
                        np.linalg.norm(
                            nominal_target[:4] - current[:4],
                            ord=np.inf,
                        )
                    )
                    ori_err = _observe_orientation()
                    return _finish(_report(
                        False,
                        error="keep_ori_final_settle_failed",
                        final_settle_frames=final_settle_frames,
                        j1234_err_inf_rad=round(err_inf, 5),
                        ori_err_deg=round(float(ori_err), 4),
                    ))
                command_q = _solve_command(
                    nominal_target[:4],
                    output_q1234=nominal_target[:4],
                    seed_wrist=_seed_wrist_command(),
                )
                if command_q is None:
                    solver_failures += 1
                    if solver_failures <= 3 or final_settle_frames % 10 == 0:
                        ctx.log(
                            f"set_arm_to_grasp_position[{arm}] keep_ori J567 "
                            f"settle solve failed "
                            f"frame={final_settle_frames + 1}; keep previous "
                            f"wrist result={last_solver_result}"
                        )
                    command_q = np.concatenate(
                        [nominal_target[:4], last_j567]
                    ).astype(np.float64)
                else:
                    command_q = _apply_wrist_feedback(
                        command_q,
                        current,
                        frame_number=final_settle_frames + 1,
                    )
                j1234_command_error = float(
                    np.linalg.norm(
                        command_q[:4] - nominal_target[:4],
                        ord=np.inf,
                    )
                )
                j1234_nominal_command_max_error_rad = max(
                    j1234_nominal_command_max_error_rad,
                    j1234_command_error,
                )
                if j1234_command_error > 1e-10:
                    return _finish(_report(
                        False,
                        error="keep_ori_modified_final_j1234",
                        final_settling=True,
                        j1234_command_error_rad=j1234_command_error,
                    ))
                yield _make_legacy_arm_action(
                    world,
                    **_make_action(command_q),
                )
                frames_total += 1
                final_settle_frames += 1
                current_after = _arm_qpos(world, arm)
                err_inf = float(
                    np.linalg.norm(
                        nominal_target[:4] - current_after[:4],
                        ord=np.inf,
                    )
                )
                ori_err = _observe_orientation()
                if (
                    err_inf <= settle_tol
                    and ori_err <= _GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG
                ):
                    stable += 1
                else:
                    stable = 0
                if final_settle_frames == 1 or final_settle_frames % 8 == 0:
                    ctx.log(
                        f"set_arm_to_grasp_position[{arm}] fixed grasp settle "
                        f"frame={final_settle_frames} "
                        f"j1234_err={err_inf:.4f}rad ori={ori_err:.2f}deg"
                    )

        final_target_q1234 = np.asarray(
            anchors[-1]["q"],
            dtype=np.float64,
        ).reshape(7)[:4]
        if float(
            np.linalg.norm(
                last_command_q[:4] - final_target_q1234,
                ord=np.inf,
            )
        ) > 1e-10:
            return _finish(_report(
                False,
                error="keep_ori_final_command_not_fixed_grasp_j1234",
                final_command_j1234=[
                    round(float(x), 8) for x in last_command_q[:4]
                ],
                expected_j1234=[
                    round(float(x), 8) for x in final_target_q1234
                ],
            ))

        try:
            world.set_arm_pin_qpos(arm, last_command_q)
        except Exception:
            pass
        final_verify = _grasp_j1234_keep_ori_verify(
            world,
            arm,
            final_target_q1234,
            target_quat_np,
            joint_tol_rad=_GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD,
        )
        if not final_verify["ok"]:
            return _finish(_report(
                False,
                error="keep_ori_final_verify_failed",
                final_verify=final_verify,
            ))
        return _finish(_report(True, final_verify=final_verify))
    except GeneratorExit:
        _finish({"ok": False})
        raise


def _grasp_prep_residual_correct(
    ctx,
    arm: str,
    phase1_q: np.ndarray,
    final_q: np.ndarray,
    *,
    gripper_mode: str,
    tol: float,
    deadline: float | None = None,
) -> dict:
    """Correct the last tiny controller residual after a real trajectory finished."""
    world = ctx.world
    v0 = _grasp_prep_verify(world, arm, phase1_q, final_q, tol=tol)
    if v0.get("ok"):
        return {"ok": True, "already_ok": True, "verify": v0}
    err0 = float(v0.get("err_final_rad", float("inf")))
    if not np.isfinite(err0) or err0 > _GRASP_PREP_RESIDUAL_CORRECT_TOL_RAD:
        return {
            "ok": False,
            "skipped": True,
            "reason": "residual_too_large",
            "err_final_rad": err0,
            "limit_rad": _GRASP_PREP_RESIDUAL_CORRECT_TOL_RAD,
            "verify": v0,
        }

    target = np.asarray(final_q, dtype=np.float64).reshape(7)
    ctx.log(
        f"set_arm_to_grasp_position[{arm}] residual controller settle "
        f"err={err0:.4f}rad <= {_GRASP_PREP_RESIDUAL_CORRECT_TOL_RAD:.3f}rad"
    )
    keep_qpos = _current_gripper_qpos(world, arm)
    for _ in range(max(1, int(_GRASP_PREP_RESIDUAL_CORRECT_FRAMES))):
        if deadline is not None and time.monotonic() >= float(deadline):
            return {
                "ok": False,
                "timeout": True,
                "reason": "trajectory_deadline_exhausted",
                "err_before_rad": err0,
                "verify": _grasp_prep_verify(
                    world, arm, phase1_q, target, tol=tol,
                ),
            }
        current = _arm_qpos(world, arm)
        err = target - current
        if float(np.linalg.norm(err, ord=np.inf)) <= max(0.018, float(tol)):
            break
        q_next = current + np.clip(err, -0.04, 0.04)
        action = {f"arm_{arm}": q_next.tolist()}
        if gripper_mode == "open":
            action[f"gripper_{arm}"] = _gripper_open_override(world, arm)
        elif keep_qpos is not None:
            try:
                world.set_gripper_pin_qpos(arm, keep_qpos)
            except Exception:
                pass
            action[f"gripper_{arm}"] = keep_qpos
        yield _make_legacy_arm_action(world, **action)

    v1 = _grasp_prep_verify(world, arm, phase1_q, target, tol=tol)
    return {
        "ok": bool(v1.get("ok")),
        "controller_settled": True,
        "err_before_rad": err0,
        "verify": v1,
    }


@register_skill(
    "set_arm_to_grasp_position_shortcut",
    description=(
        "抓取预备位 shortcut（关节 controller action，任意起点，无 IK/规划）："
        "①全臂→Phase1 v2（大臂后摆+肘腕折叠）；"
        "②肩 j1-j3→hang，肘腕 j4-j7 冻结。"
        "base/trunk 不动。arm='right'|'left'|'both'。"
    ),
)
def set_arm_to_grasp_position_shortcut(
    ctx,
    arm: str = "right",
    open_gripper: bool = True,
    max_dq_per_step: float = _GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP,
    tol: float = 0.08,
    timeout_s: float = 35.0,
):
    """任意关节起点 → grasp prep：两步关节插值，与旧版目标相同。"""
    arm = arm.lower().strip()
    if arm not in ("left", "right", "both"):
        ctx.log(f"set_arm_to_grasp_position: arm 参数错误 '{arm}'，应为 left/right/both")
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        return
    if not bool(open_gripper):
        ctx.log("set_arm_to_grasp_position: 强制 open_gripper=True（grasp prep 固定开爪）")
    open_gripper = True

    world = ctx.world
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
    except Exception:
        pass
    arms_todo = ["left", "right"] if arm == "both" else [arm]
    from behavior_interface.skills.eef import _reset_selected_tool_roll_to_zero

    for a in arms_todo:
        reset_report = yield from _reset_selected_tool_roll_to_zero(
            world,
            a,
            ctx=ctx,
            stage_name="set_arm_to_grasp_position_shortcut.start",
        )
        if not reset_report.get("ok", True):
            ctx.set_result({
                "ok": False,
                "error": f"{a} J8 controller reset did not converge",
                "tool_roll_reset": reset_report,
            })
            return
    timeout_total = max(8.0, float(timeout_s))
    tol_strict = float(tol)
    # accept_tol is only a motion-stage escape hatch for controller plateaus.
    # Final success is always verified with tol_strict.
    accept_tol = max(tol_strict, _GRASP_PREP_ACCEPT_TOL_RAD)
    phase1_timeout = max(3.0, min(6.0, timeout_total * 0.12))
    phase1b_timeout = max(2.0, min(4.0, timeout_total * 0.08))
    phase2_timeout = max(3.0, min(5.0, timeout_total * 0.12))
    polish_timeout = max(1.5, min(3.0, timeout_total * 0.06))
    phase1_targets: dict = {}
    final_targets: dict = {}
    frozen_used: dict = {}
    phase1_q_map: dict[str, np.ndarray] = {}
    final_q_map: dict[str, np.ndarray] = {}
    arms_active: list[str] = []
    arms_need_phase1: list[str] = []
    verify: dict[str, dict] = {}

    ctx.log(
        f"set_arm_to_grasp_position arms={arms_todo} "
        f"模式=关节两步直达(无规划) open_gripper={open_gripper} "
        f"max_dq={float(max_dq_per_step):.2f} tol={float(tol):.3f} "
        f"accept_tol={accept_tol:.3f}(motion-only) "
        f"timeouts=phase1:{phase1_timeout:.1f}s/phase1b:{phase1b_timeout:.1f}s/"
        f"phase2:{phase2_timeout:.1f}s/polish:{polish_timeout:.1f}s"
    )

    for a in arms_todo:
        try:
            phase1_q = _phase1_v2_grasp_target_qpos(world, a)
        except RuntimeError as exc:
            ctx.log(f"set_arm_to_grasp_position[{a}] {exc}")
            ctx.set_result({"ok": False, "error": str(exc), "arm": a})
            return

        final_q = _grasp_prep_shoulder_hang_target_qpos(phase1_q)
        phase1_q_map[a] = phase1_q
        final_q_map[a] = final_q
        phase1_targets[a] = phase1_q.tolist()
        final_targets[a] = final_q.tolist()
        frozen_used[a] = {
            f"j{i+1}": round(float(phase1_q[i]), 4) for i in range(3, 7)
        }

        try:
            q0 = _arm_qpos(world, a)
        except Exception as exc:
            ctx.log(f"set_arm_to_grasp_position[{a}] 读 qpos 失败: {exc}")
            ctx.set_result({"ok": False, "error": str(exc), "arm": a})
            return

        if _is_at_grasp_prep(q0, phase1_q, final_q, tol=tol_strict):
            try:
                world.set_arm_pin_qpos(a, final_q)
            except Exception:
                pass
            verify[a] = _grasp_prep_verify(world, a, phase1_q, final_q, tol=tol_strict)
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 已在 grasp prep，跳过 "
                f"q={verify[a]['qpos']}"
            )
            continue

        arms_active.append(a)
        lower_err0 = _active_joint_err(q0, final_q, _LOWER_ARM_DOF)
        need_phase1 = bool(lower_err0 >= max(tol_strict, _GRASP_PREP_DIRECT_LOWER_TOL_RAD))
        if need_phase1:
            arms_need_phase1.append(a)
        ctx.log(
            f"set_arm_to_grasp_position[{a}] grasp终态(肩垂直+肘腕折): "
            f"j1-j3=[0,0,0] j4-j7="
            f"[{','.join(f'{final_q[i]:+.3f}' for i in range(3, 7))}]"
        )
        ctx.log(
            f"set_arm_to_grasp_position[{a}] ①Phase1 v2 全臂 tgt="
            f"[{','.join(f'{phase1_q[i]:+.3f}' for i in range(7))}]"
        )
        ctx.log(
            f"set_arm_to_grasp_position[{a}] 当前 q="
            f"[{','.join(f'{q0[i]:+.2f}' for i in range(7))}] "
            f"err→final={float(np.linalg.norm(q0 - final_q, ord=np.inf)):.3f}rad "
            f"lower_err={lower_err0:.3f} phase1={'yes' if need_phase1 else 'skip'}"
        )

    if not arms_active:
        gripper_qpos = yield from _yield_force_open_grippers(
            ctx,
            arms_todo,
            frames=20,
            log_prefix="set_arm_to_grasp_position/already_prep",
        )
        all_ok = all(verify.get(a, {}).get("ok", True) for a in arms_todo)
        ctx.set_result({
            "ok": all_ok,
            "arms": arms_todo,
            "pose": "grasp_prep_shoulder_hang_elbow_wrist_folded",
            "open_gripper": open_gripper,
            "gripper_qpos": gripper_qpos,
            "phase1_target_qpos": phase1_targets,
            "target_qpos": final_targets,
            "verify": verify,
            "note": "all arms already at grasp prep",
            "tol_rad": tol_strict,
            "accept_tol_rad": accept_tol,
            "accept_tol_is_motion_only": True,
        })
        return

    snap_map_initial = {a: final_q_map[a] for a in arms_active}
    ctx.log(
        "set_arm_to_grasp_position controller-action grasp prep "
        f"arms={arms_active}"
    )
    yield from _snap_arms_to_targets(
        ctx, snap_map_initial,
        open_gripper=open_gripper,
        log_prefix="set_grasp/action_only",
        hold_frames=12,
    )
    gripper_qpos = yield from _yield_force_open_grippers(
        ctx,
        arms_todo,
        frames=12,
        log_prefix="set_arm_to_grasp_position/final",
    )

    all_ok = True
    for a in arms_todo:
        v = _grasp_prep_verify(
            world, a, phase1_q_map[a], final_q_map[a], tol=tol_strict,
        )
        verify[a] = v
        if not v["ok"]:
            all_ok = False
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 未到位 "
                f"err={v['err_final_rad']}rad q={v['qpos']}"
            )
        else:
            try:
                world.set_arm_pin_qpos(a, final_q_map[a])
            except Exception:
                pass
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 到位验证 OK "
                f"q={v['qpos']}"
            )

    ctx.set_result({
        "ok": all_ok,
        "arms": arms_todo,
        "pose": "grasp_prep_shoulder_hang_elbow_wrist_folded",
        "open_gripper": open_gripper,
        "gripper_qpos": gripper_qpos,
        "phase1_target_qpos": phase1_targets,
        "target_qpos": final_targets,
        "frozen_j4_j7": frozen_used,
        "shoulder_dof": [f"j{i+1}" for i in _UPPER_ARM_DOF],
        "verify": verify,
        "tol_rad": tol_strict,
        "accept_tol_rad": accept_tol,
        "accept_tol_is_motion_only": True,
        "execution": "controller_action_and_pin",
        "error": None if all_ok else "grasp prep 关节未到位",
    })
    return


@register_skill(
    "set_arm_to_grasp_position",
    description=(
        "抓取预备位（EEF 直线 IK + 同分支过滤 + 关节空间插值）："
        "从当前 EEF 到 grasp prep EEF 取多段中间点，逐点 IK，"
        "过滤相邻关节跳变和末端附近偏离目标关节分支的解。"
        "gripper='open'|'keep'（默认 keep）；keep_ori_arm 可让指定臂"
        "固定入口世界系 EEF RPY：J1-J4 走 grasp-prep，"
        "8DOF 用 J5-J8（7DOF 用 J5-J7）逐帧追姿态；base/trunk 不动。"
    ),
)
def set_arm_to_grasp_position(
    ctx,
    arm: str = "right",
    keep_ori_arm: str = "none",
    gripper: str | None = None,
    open_gripper: bool | None = None,
    max_dq_per_step: float = 0.30,
    tol: float = 0.08,
    timeout_s: float = 15.0,
    force_jointspace: bool = False,
):
    """Current EEF -> grasp prep via branch-filtered 6D IK anchors."""
    arm = arm.lower().strip()
    if arm not in ("left", "right", "both"):
        ctx.log(f"set_arm_to_grasp_position: arm 参数错误 '{arm}'，应为 left/right/both")
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        return

    gripper_mode = _normalize_grasp_gripper_mode(gripper, open_gripper)
    world = ctx.world
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
    except Exception:
        pass

    arms_todo = ["left", "right"] if arm == "both" else [arm]
    from behavior_interface.skills.reset_body import _normalize_keep_ori_arm

    requested_keep_ori_arms = _normalize_keep_ori_arm(keep_ori_arm)
    keep_ori_wrist_policy = "nominal_j1234_unchanged_j567_feedback_tracking"
    keep_ori_scope_text = "entry_rpy_tracked_by_j567_every_frame"
    invalid_keep_ori_arms = requested_keep_ori_arms - set(arms_todo)
    if invalid_keep_ori_arms:
        error = (
            f"keep_ori_arm={sorted(requested_keep_ori_arms)} 包含未执行手臂 "
            f"{sorted(invalid_keep_ori_arms)}；arm={arm}"
        )
        ctx.log(f"set_arm_to_grasp_position: {error}")
        ctx.set_result({
            "ok": False,
            "arm": arm,
            "keep_ori_requested_arm": sorted(requested_keep_ori_arms),
            "error": error,
        })
        return
    from behavior_interface.skills.eef import _reset_selected_tool_roll_to_zero

    # keep_ori 只用 J5-J7 追姿态，J8 通过 controller action 归零锁定。
    # 必须在下面快照入口 EEF
    # 姿态之前完成，否则 J567 会去追一个 J8≠0 时的姿态，凭空多出一段偏差。
    tool_roll_zero_reports: dict[str, dict] = {}
    for a in arms_todo:
        tool_roll_zero_reports[a] = yield from _reset_selected_tool_roll_to_zero(
            world,
            a,
            ctx=ctx,
            stage_name="set_arm_to_grasp_position.start",
        )
        if not tool_roll_zero_reports[a].get("ok", True):
            ctx.set_result({
                "ok": False,
                "error": f"{a} J8 controller reset did not converge",
                "tool_roll_zero": tool_roll_zero_reports,
            })
            return
    timeout_total = min(
        (
            _GRASP_PREP_KEEP_ORI_MAX_TIMEOUT_S
            if requested_keep_ori_arms
            else _GRASP_PREP_EEF_LINE_MAX_TIMEOUT_S
        ),
        max(
            (
                _GRASP_PREP_KEEP_ORI_MIN_TIMEOUT_S
                if requested_keep_ori_arms
                else _GRASP_PREP_EEF_LINE_MIN_TIMEOUT_S
            ),
            float(timeout_s),
        ),
    )
    tol_strict = float(tol)
    normal_max_dq_frame = _grasp_prep_play_max_dq(
        max_dq_per_step,
        keep_ori=False,
        force_jointspace=bool(force_jointspace),
    )
    keep_ori_max_dq_frame = _grasp_prep_play_max_dq(
        max_dq_per_step,
        keep_ori=True,
        force_jointspace=bool(force_jointspace),
    )
    phase1_targets: dict[str, list[float]] = {}
    final_targets: dict[str, list[float]] = {}
    frozen_used: dict[str, dict] = {}
    phase1_q_map: dict[str, np.ndarray] = {}
    final_q_map: dict[str, np.ndarray] = {}
    verify: dict[str, dict] = {}
    plans: dict[str, list[dict]] = {}
    planner_meta: dict[str, dict] = {}
    arms_active: list[str] = []
    keep_ori_pose_targets: dict[str, dict[str, np.ndarray]] = {}
    keep_ori_final_q_map: dict[str, np.ndarray] = {}
    keep_ori_preflight: dict[str, dict] = {}

    ctx.log(
        f"set_arm_to_grasp_position arms={arms_todo} "
        f"mode=EEF直线IK同分支过滤 gripper={gripper_mode} "
        f"keep_ori_arm={sorted(requested_keep_ori_arms)} "
        f"max_dq_normal={normal_max_dq_frame:.3f}rad "
        f"max_dq_keep_ori={keep_ori_max_dq_frame:.3f}rad "
        f"tol={tol_strict:.3f} "
        f"timeout={timeout_total:.1f}s force_jointspace={bool(force_jointspace)}"
    )

    for a in arms_todo:
        try:
            phase1_q = _phase1_v2_grasp_target_qpos(world, a)
            final_q = _grasp_prep_shoulder_hang_target_qpos(phase1_q)
            q0 = _arm_qpos(world, a)
        except Exception as exc:
            ctx.log(f"set_arm_to_grasp_position[{a}] 初始化失败: {exc}")
            ctx.set_result({"ok": False, "error": str(exc), "arm": a})
            return
        phase1_q_map[a] = phase1_q
        final_q_map[a] = final_q
        phase1_targets[a] = [float(x) for x in phase1_q]
        final_targets[a] = [float(x) for x in final_q]
        if a in requested_keep_ori_arms:
            try:
                start_eef = world.eef_pose(arm=a)
                start_pos = np.asarray(
                    start_eef["pos"],
                    dtype=np.float64,
                ).reshape(3)
                start_quat = np.asarray(
                    start_eef["quat"],
                    dtype=np.float64,
                ).reshape(4)
                start_quat /= max(float(np.linalg.norm(start_quat)), 1e-12)
                from behavior_interface.skills.wrist_j567_solver import (
                    quat_xyzw_to_rpy_deg,
                )

                start_rpy_deg = quat_xyzw_to_rpy_deg(start_quat)
                keep_ori_pose_targets[a] = {
                    "pos0": start_pos.copy(),
                    "quat": start_quat.copy(),
                    "rpy_deg": start_rpy_deg.copy(),
                }
                # J8 已锁 0，姿态可达性只能由 J5-J7 决定。
                from behavior_interface.skills.wrist_j567_solver import (
                    solve_wrist_j567_for_eef_rpy,
                )

                preflight = solve_wrist_j567_for_eef_rpy(
                    world,
                    a,
                    start_rpy_deg,
                    fixed_q1234=final_q[:4],
                    seed_j567=q0[4:7],
                    ori_tol_deg=_GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
                    max_steps=90,
                    max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
                    global_search=True,
                )
                preflight_label = "j567"
                preflight_deg_key = "j567_deg"
                policy_text = "nominal J1-J4 + framewise J5-J7 RPY tracking"
                execution_text = "nominal_j1234_unchanged_j567_feedback_tracking"
                fail_prefix = "keep_ori_final_j567_preflight_failed:"
                keep_ori_preflight[a] = preflight
                if not preflight.get("ok", False):
                    error = (
                        f"{fail_prefix}"
                        f"{preflight.get('error') or 'unreachable'}"
                    )
                    ctx.log(
                        f"set_arm_to_grasp_position[{a}] {error} "
                        f"ori_err={preflight.get('ori_err_deg')}deg"
                    )
                    ctx.set_result({
                        "ok": False,
                        "arm": a,
                        "keep_ori_requested_arm": sorted(
                            requested_keep_ori_arms
                        ),
                        "execution": execution_text,
                        "error": error,
                        "keep_ori_preflight": keep_ori_preflight,
                    })
                    return
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] keep_ori snapshot "
                    f"rpy_deg=[{','.join(f'{v:+.3f}' for v in start_rpy_deg)}] "
                    f"quat=[{','.join(f'{v:+.6f}' for v in start_quat)}] "
                    f"policy={policy_text} "
                    f"final_preflight_{preflight_label}_deg="
                    f"{[round(float(v), 2) for v in preflight[preflight_deg_key]]}"
                )
            except Exception as exc:
                error = f"keep_ori 初始化失败: {type(exc).__name__}: {exc}"
                ctx.log(f"set_arm_to_grasp_position[{a}] {error}")
                ctx.set_result({
                    "ok": False,
                    "arm": a,
                    "keep_ori_requested_arm": sorted(requested_keep_ori_arms),
                    "error": error,
                })
                return
        frozen_used[a] = {
            f"j{i+1}": round(float(phase1_q[i]), 4) for i in range(3, 7)
        }
        ctx.log(
            f"set_arm_to_grasp_position[{a}] target grasp-prep q="
            f"[{','.join(f'{v:+.3f}' for v in final_q)}] "
            f"current_gap={float(np.linalg.norm(q0 - final_q, ord=np.inf)):.3f}rad"
        )
        if a in requested_keep_ori_arms:
            keep_target = keep_ori_pose_targets[a]
            keep_verify = _grasp_j1234_keep_ori_verify(
                world,
                a,
                final_q[:4],
                keep_target["quat"],
                joint_tol_rad=_GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD,
            )
            if keep_verify["ok"]:
                try:
                    world.set_arm_pin_qpos(a, q0)
                except Exception:
                    pass
                keep_ori_final_q_map[a] = q0.copy()
                verify[a] = keep_verify
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] J1-J4 已在 grasp prep "
                    "且入口 RPY 满足，跳过轨迹"
                )
                continue
            ctx.log(
                f"set_arm_to_grasp_position[{a}] keep_ori 初始检查 "
                f"j1234_err={keep_verify['j1234_err_inf_rad']:.4f}rad "
                f"ori_err={keep_verify['ori_err_deg']:.2f}deg；继续运动"
            )
        elif _is_at_grasp_prep(q0, phase1_q, final_q, tol=tol_strict):
            try:
                world.set_arm_pin_qpos(a, final_q)
            except Exception:
                pass
            verify[a] = _grasp_prep_verify(
                world,
                a,
                phase1_q,
                final_q,
                tol=tol_strict,
            )
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 已在 grasp prep，跳过 IK path"
            )
            continue
        arms_active.append(a)

    if gripper_mode == "open":
        yield from _yield_force_open_grippers(
            ctx,
            arms_todo,
            frames=1,
            log_prefix="set_arm_to_grasp_position/open_start",
        )
    else:
        for a in arms_todo:
            qg = _current_gripper_qpos(world, a)
            if qg is not None:
                try:
                    world.set_gripper_pin_qpos(a, qg)
                except Exception:
                    pass
        ctx.log(
            "set_arm_to_grasp_position keep gripper qpos="
            f"{_gripper_qpos_map(world, arms_todo)}"
        )

    deadline = time.monotonic() + timeout_total
    for a in arms_active:
        start_gap = float(np.linalg.norm(_arm_qpos(world, a) - final_q_map[a], ord=np.inf))
        if bool(force_jointspace):
            anchors, meta = _plan_grasp_prep_jointspace_fallback(
                ctx,
                a,
                phase1_q_map[a],
                final_q_map[a],
                tol=tol_strict,
                reason="force_jointspace",
            )
        else:
            if start_gap > _GRASP_PREP_EEF_LINE_MAX_START_GAP_RAD:
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] start_gap={start_gap:.3f}rad "
                    f"> old_guard={_GRASP_PREP_EEF_LINE_MAX_START_GAP_RAD:.3f}rad; "
                    "still trying EEF-line IK as primary"
                )
            anchors, meta = _plan_grasp_prep_line_ik(
                ctx,
                a,
                final_q_map[a],
                pos_tol=0.012,
                ori_tol_deg=5.0,
            )
            if anchors:
                compress_meta = {
                    "enabled": False,
                    "from": len(anchors),
                    "to": len(anchors),
                    "reason": "physics_safe_full_cartesian_anchor_path",
                }
                meta["play_anchor_compression"] = compress_meta
                if a in requested_keep_ori_arms:
                    meta["keep_ori_overlay"] = keep_ori_wrist_policy
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] preserve all "
                    f"{len(anchors)} EEF-line IK anchors"
                )
        planner_meta[a] = meta
        if not anchors:
            fallback_reason = meta.get("error", "ik_plan_failed")
            anchors, fallback_meta = _plan_grasp_prep_jointspace_fallback(
                ctx,
                a,
                phase1_q_map[a],
                final_q_map[a],
                tol=tol_strict,
                reason=fallback_reason,
            )
            fallback_meta["eef_line_meta"] = meta
            if a in requested_keep_ori_arms:
                fallback_meta["keep_ori_overlay"] = keep_ori_wrist_policy
            planner_meta[a] = fallback_meta
        plans[a] = anchors

    play_meta: dict[str, dict] = {}
    for a in arms_active:
        remaining = max(0.0, deadline - time.monotonic())
        keep_target = keep_ori_pose_targets.get(a)
        play_max_dq = (
            keep_ori_max_dq_frame
            if keep_target is not None
            else normal_max_dq_frame
        )
        if keep_target is not None:
            res = yield from _grasp_line_play_anchors_j567_keep_ori(
                ctx,
                a,
                plans[a],
                gripper_mode=gripper_mode,
                max_dq_per_frame=play_max_dq,
                timeout_s=remaining,
                target_rpy_deg=keep_target["rpy_deg"],
                target_quat=keep_target["quat"],
                final_tol=tol_strict,
            )
            if res.get("final_q_command") is not None:
                keep_ori_final_q_map[a] = np.asarray(
                    res["final_q_command"],
                    dtype=np.float64,
                ).reshape(7)
        else:
            res = yield from _grasp_line_play_anchors(
                ctx,
                a,
                plans[a],
                gripper_mode=gripper_mode,
                max_dq_per_frame=play_max_dq,
                timeout_s=remaining,
                final_tol=tol_strict,
            )
        play_meta[a] = res
        if not res.get("ok", False):
            if a in requested_keep_ori_arms:
                error = str(
                    res.get("error")
                    or (
                        "keep_ori_trajectory_timeout"
                        if res.get("timeout")
                        else "keep_ori_trajectory_play_failed"
                    )
                )
                ctx.set_result({
                    "ok": False,
                    "arm": a,
                    "arms": arms_todo,
                    "gripper": gripper_mode,
                    "keep_ori_requested_arm": sorted(requested_keep_ori_arms),
                    "keep_ori_arm": sorted(requested_keep_ori_arms),
                    "keep_ori_scope": keep_ori_scope_text,
                    "execution": keep_ori_wrist_policy,
                    "error": error,
                    "play_meta": play_meta,
                    "planner_meta": planner_meta,
                    "target_qpos": final_targets,
                })
                return
            verify_after_first = _grasp_prep_verify(
                world, a, phase1_q_map[a], final_q_map[a], tol=tol_strict,
            )
            if verify_after_first.get("ok"):
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] first play reported {res}, "
                    f"but verify is OK; accept without fallback"
                )
                res = {
                    **res,
                    "ok": True,
                    "accepted_by_verify": True,
                    "verify": verify_after_first,
                }
                play_meta[a] = res
                continue
            retry_timeout = max(0.0, deadline - time.monotonic())
            if res.get("timeout") or retry_timeout <= 0.0:
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] trajectory budget exhausted; "
                    "skip doomed fallback retry"
                )
            else:
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] first trajectory play failed "
                    f"{res}; retry actual jointspace fallback from current q"
                )
                retry_anchors, retry_meta = _plan_grasp_prep_jointspace_fallback(
                    ctx,
                    a,
                    phase1_q_map[a],
                    final_q_map[a],
                    tol=tol_strict,
                    reason="trajectory_play_failed:False",
                )
                retry_meta["fallback_after_play_failed"] = True
                retry_meta["previous_planner_meta"] = planner_meta.get(a)
                planner_meta[a] = retry_meta
                retry_res = yield from _grasp_line_play_anchors(
                    ctx,
                    a,
                    retry_anchors,
                    gripper_mode=gripper_mode,
                    max_dq_per_frame=normal_max_dq_frame,
                    timeout_s=retry_timeout,
                    final_tol=tol_strict,
                )
                play_meta[a] = {
                    "first_attempt": res,
                    "retry": retry_res,
                    "ok": bool(retry_res.get("ok")),
                }
                res = retry_res
        if not res.get("ok", False) and not res.get("timeout"):
            correction = yield from _grasp_prep_residual_correct(
                ctx,
                a,
                phase1_q_map[a],
                final_q_map[a],
                gripper_mode=gripper_mode,
                tol=tol_strict,
                deadline=deadline,
            )
            if isinstance(play_meta.get(a), dict):
                play_meta[a]["residual_correction"] = correction
            else:
                play_meta[a] = {
                    "play": play_meta.get(a),
                    "residual_correction": correction,
                }
            if correction.get("ok", False):
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] accepted after residual "
                    f"correction verify={correction.get('verify')}"
                )
                res = {
                    **res,
                    "ok": True,
                    "accepted_by_residual_correction": True,
                    "residual_correction": correction,
                }
                play_meta[a]["ok"] = True
                continue
            if correction.get("timeout"):
                res = {
                    **res,
                    "timeout": True,
                    "residual_correction": correction,
                }
        if not res.get("ok", False):
            if res.get("timeout"):
                error = (
                    "trajectory_play_timeout "
                    f"anchor={res.get('anchor')}/{res.get('anchors_total')} "
                    f"frame={res.get('frame_in_anchor')}/{res.get('frames_in_anchor')} "
                    f"frames_total={res.get('frames')}"
                )
            elif res.get("cancelled"):
                error = "trajectory_play_cancelled"
            else:
                error = "trajectory_play_failed"
            ctx.set_result({
                "ok": False,
                "arm": a,
                "arms": arms_todo,
                "gripper": gripper_mode,
                "execution": (
                    "eef_line_ik_branch_filtered_joint_interpolation"
                    if not any(m.get("fallback") for m in planner_meta.values())
                    else "eef_line_ik_with_jointspace_fallback"
                ),
                "error": error,
                "play_meta": play_meta,
                "planner_meta": planner_meta,
                "target_qpos": final_targets,
            })
            return

    if not arms_active:
        # Keep/open gripper mode is already applied above.
        from behavior_interface.skills.eef import _assert_legacy_7dof_motion_ready

        for _ in range(1):
            for a in arms_todo:
                _assert_legacy_7dof_motion_ready(world, a)
            yield world.make_action()

    all_ok = True
    for a in arms_todo:
        if a in requested_keep_ori_arms:
            keep_target = keep_ori_pose_targets[a]
            v = _grasp_j1234_keep_ori_verify(
                world,
                a,
                final_q_map[a][:4],
                keep_target["quat"],
                joint_tol_rad=_GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD,
            )
            verify[a] = v
            if not v["ok"]:
                all_ok = False
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] keep_ori 未到位 "
                    f"j1234_err={v['j1234_err_inf_rad']:.4f}rad "
                    f"ori_err={v['ori_err_deg']:.2f}deg"
                )
            else:
                try:
                    pin_q = keep_ori_final_q_map.get(a)
                    if pin_q is None:
                        pin_q = _arm_qpos(world, a)
                    world.set_arm_pin_qpos(a, pin_q)
                except Exception:
                    pass
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] keep_ori 到位验证 OK "
                    f"j1234_err={v['j1234_err_inf_rad']:.4f}rad "
                    f"ori_err={v['ori_err_deg']:.2f}deg"
                )
            continue

        v = _grasp_prep_verify(
            world, a, phase1_q_map[a], final_q_map[a], tol=tol_strict,
        )
        verify[a] = v
        if not v["ok"]:
            correction = yield from _grasp_prep_residual_correct(
                ctx,
                a,
                phase1_q_map[a],
                final_q_map[a],
                gripper_mode=gripper_mode,
                tol=tol_strict,
                deadline=deadline,
            )
            if isinstance(play_meta.get(a), dict):
                play_meta[a]["post_verify_residual_correction"] = correction
            else:
                play_meta[a] = {
                    "post_verify_residual_correction": correction,
                }
            if correction.get("ok", False):
                v = correction.get("verify", v)
                verify[a] = v
                try:
                    world.set_arm_pin_qpos(a, final_q_map[a])
                except Exception:
                    pass
                ctx.log(
                    f"set_arm_to_grasp_position[{a}] 到位验证 OK "
                    f"after residual correction q={v['qpos']}"
                )
                continue
            all_ok = False
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 未到位 "
                f"err={v['err_final_rad']}rad q={v['qpos']}"
            )
        else:
            try:
                world.set_arm_pin_qpos(a, final_q_map[a])
            except Exception:
                pass
            ctx.log(
                f"set_arm_to_grasp_position[{a}] 到位验证 OK "
                f"q={v['qpos']}"
            )

    ctx.set_result({
        "ok": all_ok,
        "arms": arms_todo,
        "pose": (
            "grasp_prep_j1234_keep_entry_eef_rpy_with_j567"
            if requested_keep_ori_arms
            else "grasp_prep_shoulder_hang_elbow_wrist_folded"
        ),
        "gripper": gripper_mode,
        "open_gripper": gripper_mode == "open",
        "gripper_qpos": _gripper_qpos_map(world, arms_todo),
        "phase1_target_qpos": phase1_targets,
        "target_qpos": final_targets,
        "frozen_j4_j7": frozen_used,
        "shoulder_dof": [f"j{i+1}" for i in _UPPER_ARM_DOF],
        "verify": verify,
        "planner_meta": planner_meta,
        "play_meta": play_meta,
        "tol_rad": tol_strict,
        "keep_ori_requested_arm": sorted(requested_keep_ori_arms),
        "keep_ori_arm": sorted(requested_keep_ori_arms),
        "keep_ori_scope": (
            keep_ori_scope_text if requested_keep_ori_arms else "none"
        ),
        "keep_ori_ok": bool(
            all(
                bool(verify.get(a, {}).get("ok"))
                for a in requested_keep_ori_arms
            )
        ),
        "tool_roll_zeroed": {
            a: {
                "before_rad": round(float(r.get("before_rad", 0.0)), 6),
                "after_rad": round(float(r.get("after_rad", 0.0)), 6),
                "reset": bool(r.get("reset")),
            }
            for a, r in tool_roll_zero_reports.items()
            if r.get("required", True)
        },
        "keep_ori_tracking_tol_deg": _GRASP_PREP_KEEP_ORI_TRACK_TOL_DEG,
        "keep_ori_abort_deg": _GRASP_PREP_KEEP_ORI_ABORT_DEG,
        "keep_ori_final_j1234_tol_rad": (
            _GRASP_PREP_KEEP_ORI_FINAL_J1234_TOL_RAD
        ),
        "keep_ori_target_qpos": {
            a: [round(float(x), 6) for x in q]
            for a, q in keep_ori_final_q_map.items()
        },
        "keep_ori_preflight": keep_ori_preflight,
        "execution": (
            keep_ori_wrist_policy
            if requested_keep_ori_arms
            else (
                "eef_line_ik_branch_filtered_joint_interpolation"
                if not any(m.get("fallback") for m in planner_meta.values())
                else "eef_line_ik_with_jointspace_fallback"
            )
        ),
        "error": (
            None
            if all_ok
            else (
                "grasp prep J1-J4 或 keep_ori 姿态未到位"
                if requested_keep_ori_arms
                else "grasp prep 关节未到位"
            )
        ),
    })
    return


class _ChildSkillContext:
    """Capture a nested skill result while sharing the real world and log stream."""

    def __init__(self, parent):
        self.world = parent.world
        self._parent = parent
        self.result: dict | None = None

    def log(self, msg: str) -> None:
        self._parent.log(msg)

    def set_status(self, msg: str) -> None:
        try:
            self._parent.set_status(msg)
        except Exception:
            pass

    def set_result(self, payload: dict) -> None:
        self.result = dict(payload)

    def get_last_result(self, skill_name: str):
        try:
            return self._parent.get_last_result(skill_name)
        except Exception:
            return None

    def is_cancelled(self) -> bool:
        try:
            return bool(self._parent.is_cancelled())
        except Exception:
            return False


@register_skill(
    "diag_set_arm_to_grasp_position_random_fk",
    description=(
        "Diagnostic: reset/grasp prep 后随机采样 arm FK 关节姿态，"
        "逐帧实际调用 set_arm_to_grasp_position 恢复到 grasp prep。"
    ),
)
def diag_set_arm_to_grasp_position_random_fk(
    ctx,
    arm: str = "right",
    n: int = 100,
    seed: int = 20260620,
    gripper: str = "open",
    max_dq_per_step: float = _GRASP_PREP_JOINTSPACE_MAX_DQ_RAD,
    tol: float = 0.08,
    timeout_s: float = 900.0,
    qpos: list[float] | None = None,
):
    """Stress-test real motion recovery from random FK-reachable arm poses."""
    arm = arm.lower().strip()
    if arm not in ("left", "right", "both"):
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        return
    arms_todo = ["left", "right"] if arm == "both" else [arm]
    n_total = max(1, int(n))
    tol_strict = float(tol)
    max_dq_frame = max(0.018, min(_GRASP_PREP_JOINTSPACE_MAX_DQ_RAD, float(max_dq_per_step)))
    gripper_mode = _normalize_grasp_gripper_mode(gripper, None)
    rng = np.random.default_rng(int(seed))
    world = ctx.world
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
    except Exception:
        pass

    ctx.log(
        f"diag_set_arm_to_grasp_position_random_fk arms={arms_todo} "
        f"n={n_total} seed={int(seed)} gripper={gripper_mode} "
        f"max_dq={max_dq_frame:.3f} tol={tol_strict:.3f} "
        f"qpos_override={qpos is not None}"
    )
    t_start = time.time()
    samples: list[dict] = []
    failures: list[dict] = []
    target_maps: dict[str, dict] = {}
    for a in arms_todo:
        phase1_q = _phase1_v2_grasp_target_qpos(world, a)
        final_q = _grasp_prep_shoulder_hang_target_qpos(phase1_q)
        target_maps[a] = {"phase1": phase1_q, "final": final_q}

    qpos_override = None
    if qpos is not None:
        try:
            qpos_override = np.asarray(qpos, dtype=np.float64).reshape(7)
            n_total = 1
        except Exception as exc:
            ctx.set_result({"ok": False, "error": f"bad qpos override: {exc}"})
            return

    attempts: dict[str, int] = {a: 0 for a in arms_todo}
    skipped_unreachable: dict[str, int] = {a: 0 for a in arms_todo}

    def _play_frames_from_result(res: dict) -> int:
        play_frames = 0
        for pm in (res.get("play_meta") or {}).values():
            if isinstance(pm, dict):
                if isinstance(pm.get("retry"), dict):
                    play_frames += int(pm["retry"].get("frames", 0) or 0)
                play_frames += int(pm.get("frames", 0) or 0)
        return int(play_frames)

    def _sample_controller_friendly_q(a: str) -> np.ndarray:
        lo, hi = _arm_joint_limits(world, a)
        final_q = target_maps[a]["final"]
        width = np.maximum(hi - lo, 1e-3)
        # The default diagnostic tests controller-reachable FK poses, not
        # arbitrary teleported joint states.  Sample a broad neighborhood around
        # grasp prep, then require actual outbound motion to reach it.
        max_delta = np.minimum(0.82, width * 0.24)
        q = final_q + rng.uniform(-1.0, 1.0, size=7) * max_delta
        return np.clip(q, lo + 0.03 * width, hi - 0.03 * width)

    def _recover_current_pose(
        a: str,
        sample_i: int,
        q_start: np.ndarray,
        eef: dict,
        *,
        outbound_res: dict | None,
        direct_injection: bool,
    ):
        phase1_q = target_maps[a]["phase1"]
        final_q = target_maps[a]["final"]
        child_ctx = _ChildSkillContext(ctx)
        sample_timeout = max(80.0, float(timeout_s) - (time.time() - t_start))
        yield from set_arm_to_grasp_position(
            child_ctx,
            arm=a,
            gripper=gripper_mode,
            open_gripper=(gripper_mode == "open"),
            max_dq_per_step=max_dq_frame,
            tol=tol_strict,
            timeout_s=sample_timeout,
            force_jointspace=True,
        )
        res = child_ctx.result or {"ok": False, "error": "nested_set_arm_no_result"}
        verify = _grasp_prep_verify(world, a, phase1_q, final_q, tol=tol_strict)
        ok = bool(res.get("ok") and verify.get("ok"))
        rec = {
            "sample": int(sample_i),
            "arm": a,
            "ok": ok,
            "start_gap_rad": round(float(np.linalg.norm(q_start - final_q, ord=np.inf)), 4),
            "eef_pos": [round(float(x), 4) for x in eef["pos"]],
            "frames": _play_frames_from_result(res),
            "verify": verify,
            "execution": res.get("execution"),
            "planner_meta": res.get("planner_meta"),
            "nested_result_ok": bool(res.get("ok")),
            "nested_error": res.get("error"),
            "direct_injection": bool(direct_injection),
        }
        if outbound_res is not None:
            rec["outbound"] = {
                "ok": bool(outbound_res.get("ok")),
                "frames": int(outbound_res.get("frames", 0) or 0),
                "err_inf": outbound_res.get("err_inf"),
            }
        return rec, res

    if qpos_override is not None:
        for i in range(n_total):
            for a in arms_todo:
                if time.time() - t_start > float(timeout_s):
                    failures.append({"sample": i + 1, "arm": a, "error": "diag_timeout"})
                    ctx.set_result({
                        "ok": False,
                        "error": "diag_timeout",
                        "tested": len(samples),
                        "failures": failures[:20],
                        "n": n_total,
                        "qpos_override": True,
                    })
                    return
                lo, hi = _arm_joint_limits(world, a)
                q_rand = np.clip(qpos_override.copy(), lo, hi)
                _force_set_arm_qpos(world, a, q_rand)
                if gripper_mode == "open":
                    _force_open_gripper_qpos(world, a)
                elif (qg := _current_gripper_qpos(world, a)) is not None:
                    try:
                        world.set_gripper_pin_qpos(a, qg)
                    except Exception:
                        pass
                for _ in range(2):
                    action = {f"arm_{a}": q_rand.tolist()}
                    if gripper_mode == "open":
                        action[f"gripper_{a}"] = _gripper_open_override(world, a)
                    yield _make_legacy_arm_action(world, **action)
                eef = world.eef_pose(arm=a)
                rec, res = yield from _recover_current_pose(
                    a,
                    i + 1,
                    q_rand,
                    eef,
                    outbound_res=None,
                    direct_injection=True,
                )
                samples.append(rec)
                if not rec["ok"]:
                    failures.append(rec)
                    ctx.log(
                        f"diag_set_arm_to_grasp_position_random_fk FAIL "
                        f"sample={i+1} arm={a} direct_injection=True "
                        f"verify={rec['verify']} nested={res}"
                    )
                    ctx.set_result({
                        "ok": False,
                        "error": "qpos_override_recovery_failed",
                        "tested": len(samples),
                        "n": n_total,
                        "failures": failures[:20],
                        "samples_tail": samples[-10:],
                        "qpos_override": True,
                        "note": "qpos override is direct joint-state injection, used only to reproduce controller-stuck states.",
                    })
                    return
    else:
        # Start from grasp prep using the same actual-motion skill under test.
        for a in arms_todo:
            child_ctx = _ChildSkillContext(ctx)
            yield from set_arm_to_grasp_position(
                child_ctx,
                arm=a,
                gripper=gripper_mode,
                open_gripper=(gripper_mode == "open"),
                max_dq_per_step=max_dq_frame,
                tol=tol_strict,
                timeout_s=max(80.0, min(180.0, float(timeout_s))),
                force_jointspace=True,
            )
            init_res = child_ctx.result or {"ok": False, "error": "initial_set_arm_no_result"}
            verify0 = _grasp_prep_verify(
                world,
                a,
                target_maps[a]["phase1"],
                target_maps[a]["final"],
                tol=tol_strict,
            )
            if not (init_res.get("ok") and verify0.get("ok")):
                ctx.set_result({
                    "ok": False,
                    "error": "initial_grasp_prep_failed",
                    "arm": a,
                    "initial_result": init_res,
                    "verify": verify0,
                })
                return

        counts: dict[str, int] = {a: 0 for a in arms_todo}
        attempts_cap = max(20, n_total * 30)
        outbound_tol = max(tol_strict, 0.12)
        while any(counts[a] < n_total for a in arms_todo):
            for a in arms_todo:
                if counts[a] >= n_total:
                    continue
                if time.time() - t_start > float(timeout_s):
                    failures.append({"sample": counts[a] + 1, "arm": a, "error": "diag_timeout"})
                    ctx.set_result({
                        "ok": False,
                        "error": "diag_timeout",
                        "tested": len(samples),
                        "attempts": attempts,
                        "skipped_unreachable": skipped_unreachable,
                        "failures": failures[:20],
                        "n": n_total,
                    })
                    return
                attempts[a] += 1
                if attempts[a] > attempts_cap:
                    ctx.set_result({
                        "ok": False,
                        "error": "not_enough_controller_reachable_random_fk_samples",
                        "arm": a,
                        "tested": len(samples),
                        "attempts": attempts,
                        "skipped_unreachable": skipped_unreachable,
                        "n": n_total,
                        "samples_tail": samples[-10:],
                    })
                    return

                q_rand = _sample_controller_friendly_q(a)
                outbound_res = yield from _grasp_line_play_anchors(
                    ctx,
                    a,
                    [{
                        "q": q_rand,
                        "label": "diag_random_fk_outbound",
                        "seed": "diag_random_fk_outbound",
                        "is_final": True,
                        "strong_servo": True,
                    }],
                    gripper_mode=gripper_mode,
                    max_dq_per_frame=max_dq_frame,
                    timeout_s=max(20.0, min(60.0, float(timeout_s) - (time.time() - t_start))),
                    final_tol=outbound_tol,
                )
                q_reached = _arm_qpos(world, a)
                outbound_err = float(np.linalg.norm(q_reached - q_rand, ord=np.inf))
                if not outbound_res.get("ok") or outbound_err > outbound_tol:
                    skipped_unreachable[a] += 1
                    ctx.log(
                        f"diag_set_arm_to_grasp_position_random_fk skip_unreachable "
                        f"arm={a} attempt={attempts[a]} outbound_ok={bool(outbound_res.get('ok'))} "
                        f"err={outbound_err:.3f}rad tol={outbound_tol:.3f}"
                    )
                    child_ctx = _ChildSkillContext(ctx)
                    yield from set_arm_to_grasp_position(
                        child_ctx,
                        arm=a,
                        gripper=gripper_mode,
                        open_gripper=(gripper_mode == "open"),
                        max_dq_per_step=max_dq_frame,
                        tol=tol_strict,
                        timeout_s=max(80.0, float(timeout_s) - (time.time() - t_start)),
                        force_jointspace=True,
                    )
                    recovery_res = child_ctx.result or {"ok": False, "error": "skip_recovery_no_result"}
                    verify_skip = _grasp_prep_verify(
                        world,
                        a,
                        target_maps[a]["phase1"],
                        target_maps[a]["final"],
                        tol=tol_strict,
                    )
                    if not (recovery_res.get("ok") and verify_skip.get("ok")):
                        failures.append({
                            "sample": counts[a] + 1,
                            "arm": a,
                            "error": "recovery_after_unreachable_outbound_failed",
                            "outbound_err_rad": round(float(outbound_err), 4),
                            "outbound": outbound_res,
                            "recovery": recovery_res,
                            "verify": verify_skip,
                        })
                        ctx.set_result({
                            "ok": False,
                            "error": "recovery_after_unreachable_outbound_failed",
                            "tested": len(samples),
                            "attempts": attempts,
                            "skipped_unreachable": skipped_unreachable,
                            "failures": failures[:20],
                            "samples_tail": samples[-10:],
                        })
                        return
                    continue

                eef = world.eef_pose(arm=a)
                q_start = q_reached.copy()
                sample_i = counts[a] + 1
                rec, res = yield from _recover_current_pose(
                    a,
                    sample_i,
                    q_start,
                    eef,
                    outbound_res={**outbound_res, "err_inf": round(float(outbound_err), 5)},
                    direct_injection=False,
                )
                samples.append(rec)
                if not rec["ok"]:
                    failures.append(rec)
                    ctx.log(
                        f"diag_set_arm_to_grasp_position_random_fk FAIL "
                        f"sample={sample_i} arm={a} verify={rec['verify']} nested={res}"
                    )
                    ctx.set_result({
                        "ok": False,
                        "error": "controller_reachable_random_fk_recovery_failed",
                        "tested": len(samples),
                        "attempts": attempts,
                        "skipped_unreachable": skipped_unreachable,
                        "n": n_total,
                        "failures": failures[:20],
                        "samples_tail": samples[-10:],
                    })
                    return
                counts[a] += 1
                if counts[a] % 10 == 0 or counts[a] == 1:
                    ctx.log(
                        f"diag_set_arm_to_grasp_position_random_fk progress "
                        f"{counts[a]}/{n_total} arm={a} ok "
                        f"start_gap={rec['start_gap_rad']}rad "
                        f"attempts={attempts[a]} skipped={skipped_unreachable[a]}"
                    )

    ctx.set_result({
        "ok": True,
        "tested": len(samples),
        "n": n_total,
        "arms": arms_todo,
        "attempts": attempts,
        "skipped_unreachable": skipped_unreachable,
        "failures": failures,
        "samples_tail": samples[-10:],
        "elapsed_s": round(float(time.time() - t_start), 2),
        "execution": (
            "qpos_override_direct_injection_for_repro"
            if qpos_override is not None
            else "controller_reachable_random_fk_via_actual_outbound_and_set_arm_actual_recovery"
        ),
        "qpos_override": qpos_override is not None,
    })


@register_skill(
    "diag_keep_ori_wrist_preflight",
    description=(
        "Diagnostic（不运动）：用当前入口 EEF RPY + grasp-prep 固定 J1-J4，"
        "对比 J567 / J5678 姿态求解器能否解出。"
    ),
)
def diag_keep_ori_wrist_preflight(
    ctx,
    arm: str = "right",
    ori_tol_deg: float = _GRASP_PREP_KEEP_ORI_SOLVE_TOL_DEG,
):
    """FK-only keep_ori 终点可解性探针：J567 vs J5678。"""
    arm = str(arm).lower().strip()
    if arm not in ("left", "right"):
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        return
    world = ctx.world
    import importlib

    import behavior_interface.skills.wrist_j567_solver as _wrist_solver_mod

    _wrist_solver_mod = importlib.reload(_wrist_solver_mod)
    quat_xyzw_to_rpy_deg = _wrist_solver_mod.quat_xyzw_to_rpy_deg
    solve_wrist_j567_for_eef_rpy = _wrist_solver_mod.solve_wrist_j567_for_eef_rpy
    solve_wrist_j5678_for_eef_rpy = (
        _wrist_solver_mod.solve_wrist_j5678_for_eef_rpy
    )

    try:
        phase1_q = _phase1_v2_grasp_target_qpos(world, arm)
        final_q = _grasp_prep_shoulder_hang_target_qpos(phase1_q)
        q0 = _arm_qpos(world, arm)
        eef = world.eef_pose(arm=arm)
        start_quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
        start_quat /= max(float(np.linalg.norm(start_quat)), 1e-12)
        start_rpy = quat_xyzw_to_rpy_deg(start_quat)
        j8 = (
            float(world.tool_roll_qpos(arm))
            if bool(world.has_tool_roll(arm))
            else 0.0
        )
    except Exception as exc:
        ctx.set_result({
            "ok": False,
            "error": f"init_failed:{type(exc).__name__}:{exc}",
            "arm": arm,
        })
        return

    tol = float(ori_tol_deg)
    ctx.log(
        f"diag_keep_ori_wrist_preflight[{arm}] "
        f"rpy_deg=[{','.join(f'{v:+.3f}' for v in start_rpy)}] "
        f"fixed_q1234={[round(float(v), 4) for v in final_q[:4]]} "
        f"seed_j567={[round(float(v), 4) for v in q0[4:7]]} "
        f"j8={j8:.4f}rad tol={tol:.2f}deg"
    )

    j567 = solve_wrist_j567_for_eef_rpy(
        world,
        arm,
        start_rpy,
        fixed_q1234=final_q[:4],
        seed_j567=q0[4:7],
        ori_tol_deg=tol,
        max_steps=90,
        max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
        global_search=True,
    )
    j5678 = solve_wrist_j5678_for_eef_rpy(
        world,
        arm,
        start_rpy,
        fixed_q1234=final_q[:4],
        seed_j5678=np.concatenate([q0[4:7], [j8]]),
        ori_tol_deg=tol,
        max_steps=90,
        max_dq_rad=_GRASP_PREP_KEEP_ORI_MAX_DQ_RAD,
        global_search=True,
    )

    ctx.log(
        f"diag_keep_ori_wrist_preflight[{arm}] "
        f"j567 ok={j567.get('ok')} ori_err={j567.get('ori_err_deg')}deg "
        f"err={j567.get('error')}"
    )
    ctx.log(
        f"diag_keep_ori_wrist_preflight[{arm}] "
        f"j5678 ok={j5678.get('ok')} ori_err={j5678.get('ori_err_deg')}deg "
        f"err={j5678.get('error')} "
        f"j5678_deg={[round(float(v), 2) for v in (j5678.get('j5678_deg') or [])]}"
    )

    ctx.set_result({
        "ok": True,
        "arm": arm,
        "moved": False,
        "ori_tol_deg": tol,
        "entry_rpy_deg": [round(float(v), 4) for v in start_rpy],
        "entry_quat_xyzw": [round(float(v), 6) for v in start_quat],
        "fixed_q1234_rad": [round(float(v), 6) for v in final_q[:4]],
        "current_q_rad": [round(float(v), 6) for v in q0],
        "current_j8_rad": round(float(j8), 6),
        "j567": {
            "ok": bool(j567.get("ok")),
            "ori_err_deg": j567.get("ori_err_deg"),
            "error": j567.get("error"),
            "j567_deg": j567.get("j567_deg"),
            "achieved_rpy_deg": j567.get("achieved_rpy_deg"),
            "jacobian_rank": j567.get("jacobian_rank"),
            "probe_count": j567.get("probe_count"),
        },
        "j5678": {
            "ok": bool(j5678.get("ok")),
            "ori_err_deg": j5678.get("ori_err_deg"),
            "error": j5678.get("error"),
            "j5678_deg": j5678.get("j5678_deg"),
            "achieved_rpy_deg": j5678.get("achieved_rpy_deg"),
            "jacobian_rank": j5678.get("jacobian_rank"),
            "probe_count": j5678.get("probe_count"),
        },
        "j5678_helps": bool(j5678.get("ok")) and not bool(j567.get("ok")),
    })
    yield None
