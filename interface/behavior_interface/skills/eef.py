"""新一代 EEF skill：统一 grasp / open / close / push 接口。

设计：
    get_eef_pose(object_name, target, push_direction=None, arm='right', k=0)
        返回 candidates，每个 dict：
            {
              "id":   int,
              "target": "grasp"|"open"|"close"|"push",
              "label": str,
              "arm":   "right" / "left",
              "eef_target": {
                "pos":      [x, y, z],      # eef 目标 TCP 世界坐标
                "approach": [ax, ay, az],   # eef 接近方向（指向目标）
                "gripper_cmd": +1 (open) | -1 (close),
              },
              "next_eef_move": [dx, dy, dz],  # 到目标后下一步 eef 增量（世界系）
              "reachable": bool,
              "reach_reason": str,
              "score":   float,
              # extra metadata（meta-action 用）：
              "meta":    {...},  # 比如 joint 名 / hinge 几何 / push 方向
            }
        - target='grasp': 沿用顶部抓取（向下 approach），next_eef_move = (0,0,+0.15)
        - target='open':  自动算门把手 + hinge 弧线，next_eef_move = 开门后弧线终点 - 当前点
        - target='close': 类似 open 反向
        - target='push':  必须传 push_direction (世界向量)，eef 从相反方向贴 +
                         next_eef_move = push_direction 归一化 * push_dist

    execute_eef_pose(eef_id=0, push_direction=None)
        读 last_result get_eef_pose，取 candidates[eef_id] 执行：
        - pre-approach (eef_target.pos 沿 -approach 偏 15cm) + 张爪
        - contact (eef_target.pos) + 按 gripper_cmd 设置夹爪
        - "next_eef_move" 阶段：
            * grasp: 直接 lift
            * push:  直线推
            * open/close: 弧线插值跟随门把手轨迹
        - 验证物体状态变化（grasped / Open state / 物体位移）

兼容旧接口：
    grasp.py 里的 get_grasp_position / execute_grasp 现在转发到这里。
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill

# 复用 grasp.py 里的工具：物体解析、AABB、reachable 判断、IK 控制器、suggested base
from behavior_interface.skills.grasp import (
    _to_np,
    _resolve_object_handle,
    _aabb_of,
    _is_reachable,
    _ARM_MAX_REACH,
    _ARM_MIN_REACH,
    _sample_grasp_candidates,
    _farthest_point_sampling,
    _eef_goto_world,
    _hold_pose,
    _read_finger_qpos,
    _read_obj_z,
    _release_arm_to_home,
    _get_arm_dof_idx,
)


_SAFE_Q_RESIDUAL_CORRECT_TOL_RAD = 0.25
_SAFE_Q_RESIDUAL_CORRECT_FRAMES = 2
_LINE_Q_RESIDUAL_CORRECT_TOL_RAD = 0.16
_LINE_Q_RESIDUAL_CORRECT_FRAMES = 2
_MOVE_ONLY_LINE_Q_TRACK_TOL_RAD = 0.025
_MOVE_ONLY_LINE_HOLD_MAX_FRAMES = 14
_TO_SAFE_WAYPOINT_JOINT_TOL_RAD = 0.035
_TO_SAFE_WAYPOINT_HOLD_FRAMES = 3
_FINAL_STORED_Q_HOLD_FRAMES = 24
_FINAL_STORED_Q_MIN_HOLD_FRAMES = 8
_FINAL_STORED_Q_RETRY_HOLD_FRAMES = 36
_FINAL_STORED_Q_RETRY_MIN_HOLD_FRAMES = 16
_TOOL_ROLL_RESET_MAX_DQ_RAD = 0.08
_TOOL_ROLL_RESET_TOL_RAD = math.radians(0.5)
_TOOL_ROLL_RESET_MAX_STEPS = 64
_TOOL_ROLL_RESET_SETTLE_STEPS = 6
_GRIPPER_INITIAL_SEEK_FORCE_N = 0.1
_GRIPPER_SEEK_FORCE_N = 0.5
_GRIPPER_BLOCKED_HOLD_FORCE_N = 0.05
_GRIPPER_CONFIRM_FORCE_N = 0.1
_GRIPPER_CARRY_FORCE_N = 0.1
_GRIPPER_STALL_VEL_M_S = 0.005
_GRIPPER_STALL_MAX_OPENING_VEL_M_S = 0.001
_GRIPPER_MIN_TRAVEL_M = 0.0008
_GRIPPER_LOWER_LIMIT_MARGIN_M = 0.002
_GRIPPER_UPPER_LIMIT_M = 0.05
_GRIPPER_UPPER_CONTACT_MARGIN_M = 0.002
_GRIPPER_STALL_WINDOW_DRIFT_M = 0.0005
_GRIPPER_CONTACT_DRIFT_M = 0.002
_GRIPPER_SEEK_TIMEOUT_S = 3.25
_GRIPPER_SEEK_WALL_TIMEOUT_S = 12.0
_GRIPPER_STALL_CONFIRM_S = 0.08


def _challenge_action_only_enabled() -> bool:
    return str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE", "")
    ).lower().strip() in {"train", "public_test", "hidden_test"}


def _is_safe_q_residual_stage(stage_name: str) -> bool:
    return str(stage_name or "").startswith("to_safe")


def _is_to_safe_xyz_stage(stage_name: str) -> bool:
    return str(stage_name or "").startswith("to_safe_xyz_joint")


def _is_repairable_line_stage(stage_name: str) -> bool:
    stage = str(stage_name or "")
    return (
        stage.startswith("eef_pose_final_q_translate")
        or stage.startswith("move_only_line")
    )


def _suggest_base_pose(world, grasp, arm="right"):
    """惰性代理：每次调用时从 sys.modules 取最新版本的 grasp._suggest_base_pose。
    这样 hot-reload grasp.py 后无需重启即可生效。
    """
    import sys
    mod = sys.modules.get("behavior_interface.skills.grasp")
    if mod is None:
        import behavior_interface.skills.grasp as mod
    return mod._suggest_base_pose(world, grasp, arm)


def _finger_q_values(finger_qpos_str: str) -> list[float]:
    """解析 _read_finger_qpos 的 "[q1, q2]"。"""
    import re
    s = str(finger_qpos_str or "")
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def _gripper_fingers_closed(finger_qpos_str: str, *, thresh: float = 0.025) -> bool:
    """解析 _read_finger_qpos 的 "[q1, q2]"，两指均值 < thresh 视为已合爪。"""
    nums = _finger_q_values(finger_qpos_str)
    if len(nums) < 2:
        return False
    return 0.5 * (nums[0] + nums[1]) < float(thresh)


def _gripper_matches_cmd(finger_qpos_str: str, cmd: float) -> bool:
    """当前夹爪是否已经接近目标开合状态。"""
    nums = _finger_q_values(finger_qpos_str)
    if len(nums) < 2:
        return False
    mean_q = 0.5 * (nums[0] + nums[1])
    return mean_q >= 0.040 if float(cmd) > 0.0 else mean_q <= 0.025


def _gripper_cmd_override(gripper_cmd) -> Optional[list[float]]:
    """Normalize scalar open/close commands or explicit finger qpos to action override."""
    if gripper_cmd is None:
        return None
    try:
        arr = np.asarray(gripper_cmd, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.asarray([float(gripper_cmd)], dtype=np.float64)
    if arr.size <= 0:
        return None
    return [float(x) for x in arr.tolist()]


def _finger_joint_indices(world, arm: str) -> list[tuple[int, str]]:
    """Return real qpos indices for the gripper finger joints of one arm."""
    robot = getattr(world, "robot", None)
    if robot is None:
        return []
    names = list(getattr(robot, "joints", {}).keys())
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
    out: list[tuple[int, str]] = []
    seen: set[str] = set()
    for jn in joint_names:
        jn = str(jn)
        if not jn or jn in seen or jn not in names:
            continue
        seen.add(jn)
        out.append((int(names.index(jn)), jn))
    return out


def _apply_gripper_cmd_to_full_q(world, arm: str, q_full, gripper_cmd) -> Optional[list[float]]:
    vals = _gripper_cmd_override(gripper_cmd)
    if vals is None:
        return None
    robot = getattr(world, "robot", None)
    if robot is None:
        return vals
    targets = _finger_joint_indices(world, arm)
    if not targets:
        return vals
    if len(vals) == 1 and len(targets) > 1:
        cmd_val = float(vals[0])
        expanded: list[float] = []
        for _idx, jn in targets:
            try:
                joint = robot.joints[jn]
                lower = float(getattr(joint, "lower_limit"))
                upper = float(getattr(joint, "upper_limit"))
                expanded.append(upper if cmd_val > 0.0 else lower)
            except Exception:
                expanded.append(cmd_val)
        vals = expanded
    if len(vals) < len(targets):
        vals = vals + [float(vals[-1])] * (len(targets) - len(vals))
    applied: list[float] = []
    for (joint_i, _jn), val in zip(targets, vals):
        q_full[int(joint_i)] = float(val)
        applied.append(float(val))
    return applied


def _apply_pinned_limb_targets_to_full_q(world, q_full, *, moving_arm: str) -> None:
    """Keep the full-q action target anchored to pinned non-moving limbs."""
    robot = getattr(world, "robot", None)
    if robot is None:
        return
    try:
        raw_idx = getattr(robot, "trunk_control_idx", None)
        if raw_idx is not None and hasattr(world, "trunk_pin_qpos_list"):
            if hasattr(raw_idx, "detach"):
                raw_idx = raw_idx.detach().cpu().numpy()
            idx = np.asarray(raw_idx, dtype=int).reshape(-1)[:4]
            tq = np.asarray(world.trunk_pin_qpos_list(), dtype=np.float64).reshape(4)
            for local_i, joint_i in enumerate(idx):
                q_full[int(joint_i)] = float(tq[int(local_i)])
    except Exception:
        pass
    moving_arm = str(moving_arm or "").lower().strip()
    names = list(getattr(robot, "joints", {}).keys())
    for other in ("left", "right"):
        if other == moving_arm:
            continue
        try:
            aq = world.arm_pin_qpos_list(other) if hasattr(world, "arm_pin_qpos_list") else None
            if aq is None:
                continue
            aq_np = np.asarray(aq, dtype=np.float64).reshape(7)
            for local_i in range(7):
                jn = f"{other}_arm_joint{local_i + 1}"
                if jn in names:
                    q_full[int(names.index(jn))] = float(aq_np[int(local_i)])
        except Exception:
            pass
    for grip_arm in ("left", "right"):
        try:
            gq = world.gripper_pin_qpos_list(grip_arm) if hasattr(world, "gripper_pin_qpos_list") else None
            if gq is None:
                continue
            _apply_gripper_cmd_to_full_q(world, grip_arm, q_full, gq)
        except Exception:
            pass


def _make_arm_q_action(world, arm: str, q_arm, gripper_cmd=None):
    """Build a normal controller action from a full robot q target.

    `world.make_action(arm_left=...)` writes into the controller slice directly.
    For R1Pro execution we want the same path as OmniGibson's motion planner:
    construct a full joint-position target and let `robot.q_to_action` map it
    through the configured controllers.
    """
    _assert_legacy_7dof_motion_ready(world, arm)
    q_arm_np = np.asarray(q_arm, dtype=np.float64).reshape(7)
    robot = getattr(world, "robot", None)
    effort_gripper_active = any(_gripper_uses_effort(world, side) for side in ("left", "right"))
    if (
        robot is None
        or getattr(world, "dry_run", False)
        or not hasattr(robot, "q_to_action")
        or effort_gripper_active
    ):
        overrides = {f"arm_{arm}": q_arm_np.tolist()}
        grip_override = _gripper_cmd_override(gripper_cmd)
        if grip_override is not None:
            overrides[f"gripper_{arm}"] = grip_override
        return world.make_action(**overrides)
    try:
        q_full0 = robot.get_joint_positions()
        q_full = q_full0.clone() if hasattr(q_full0, "clone") else np.asarray(q_full0, dtype=np.float64).copy()
        _apply_pinned_limb_targets_to_full_q(world, q_full, moving_arm=arm)
        idx = _get_arm_dof_idx(world, arm)
        for local_i, joint_i in enumerate(idx):
            q_full[int(joint_i)] = float(q_arm_np[int(local_i)])
        grip_applied = _apply_gripper_cmd_to_full_q(world, arm, q_full, gripper_cmd)
        action = robot.q_to_action(q_full)
        try:
            world.set_arm_pin_qpos(arm, q_arm_np)
        except Exception:
            pass
        if grip_applied is not None:
            try:
                world.set_gripper_pin_qpos(arm, grip_applied)
            except Exception:
                pass
        return action
    except Exception:
        overrides = {f"arm_{arm}": q_arm_np.tolist()}
        grip_override = _gripper_cmd_override(gripper_cmd)
        if grip_override is not None:
            overrides[f"gripper_{arm}"] = grip_override
        return world.make_action(**overrides)


def _gripper_limit_cmd(world, arm: str, *, open_gripper: bool) -> list[float]:
    """Return explicit absolute finger joint targets when the controller exposes them."""
    if bool(open_gripper):
        release_keepalive = getattr(
            world,
            "release_gripper_close_keepalive",
            None,
        )
        if callable(release_keepalive):
            release_keepalive(arm)
    if getattr(world, "dry_run", False):
        return [1.0 if bool(open_gripper) else -1.0]
    robot = getattr(world, "robot", None)
    names = list(getattr(robot, "joints", {}).keys()) if robot is not None else []
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

    vals: list[float] = []
    seen: set[str] = set()
    for jn in joint_names:
        jn = str(jn)
        if not jn or jn in seen or jn not in names:
            continue
        seen.add(jn)
        joint = robot.joints[jn]
        try:
            lo = float(getattr(joint, "lower_limit"))
            up = float(getattr(joint, "upper_limit"))
            q = up if bool(open_gripper) else lo
        except Exception:
            q = 0.05 if bool(open_gripper) else 0.0
        if np.isfinite(q):
            vals.append(float(q))
    if vals:
        try:
            idx = world.controller_action_idx(f"gripper_{arm}")
            if len(idx) == len(vals):
                return vals
        except Exception:
            pass
    return [1.0 if bool(open_gripper) else -1.0]


def _current_gripper_qpos_cmd(world, arm: str) -> Optional[list[float]]:
    """Read current gripper finger qpos as an absolute-position hold command."""
    try:
        if world.gripper_uses_effort(arm) and world.gripper_pin_effort_list(arm) is not None:
            return None
    except Exception:
        pass
    nums = _finger_q_values(_read_finger_qpos(world, arm))
    if nums:
        return [float(x) for x in nums]
    try:
        vals = world.gripper_qpos_list(arm)
    except Exception:
        vals = None
    if vals is not None:
        cmd = _gripper_cmd_override(vals)
        if cmd:
            return cmd
    return None


def _gripper_uses_effort(world, arm: str) -> bool:
    try:
        return bool(world.gripper_uses_effort(arm))
    except Exception:
        robot = getattr(world, "robot", None)
        controller = getattr(robot, "controllers", {}).get(f"gripper_{arm}") if robot is not None else None
        motor_type = getattr(controller, "motor_type", getattr(controller, "_motor_type", None))
        return str(motor_type).lower() == "effort"


def _finger_qpos_qvel(world, arm: str) -> tuple[np.ndarray, np.ndarray]:
    qpos = world.gripper_qpos_list(arm)
    qvel = world.gripper_qvel_list(arm)
    if qpos is None or qvel is None:
        raise RuntimeError(f"cannot read {arm} gripper qpos/qvel")
    qpos_np = np.asarray(qpos, dtype=np.float64).reshape(-1)
    qvel_np = np.asarray(qvel, dtype=np.float64).reshape(-1)
    if qpos_np.size != 2 or qvel_np.size != 2:
        raise RuntimeError(
            f"{arm} effort gripper requires two fingers, got qpos={qpos_np.size} qvel={qvel_np.size}"
        )
    return qpos_np, qvel_np


def _finger_lower_limits(world, arm: str) -> np.ndarray:
    try:
        joints = list(world.robot.finger_joints.get(arm, []))
        values = np.asarray([float(j.lower_limit) for j in joints[:2]], dtype=np.float64)
        if values.size == 2 and np.all(np.isfinite(values)):
            return values
    except Exception:
        pass
    return np.zeros(2, dtype=np.float64)


def _action_dt_s(world) -> float:
    """Return simulated time advanced by one yielded skill action."""
    try:
        frequency = float(world.env.env_config["action_frequency"])
        if np.isfinite(frequency) and frequency > 0.0:
            return 1.0 / frequency
    except Exception:
        pass
    try:
        import omnigibson as og

        dt = float(og.sim.get_sim_step_dt())
        if np.isfinite(dt) and dt > 0.0:
            return dt
    except Exception:
        pass
    return 1.0 / 30.0


def _gripper_effort_hold_action(world, arm: str, arm_q_hold, effort) -> np.ndarray:
    effort_list = np.asarray(effort, dtype=np.float64).reshape(-1).tolist()
    latch_keepalive = getattr(world, "latch_gripper_close_keepalive", None)
    if callable(latch_keepalive):
        # Update the final-boundary guard before constructing the action. This
        # makes each close phase atomic and leaves no zero-effort frame if the
        # generator is cancelled immediately after yielding.
        latch_keepalive(arm, effort=effort_list)
    return world.make_action(
        **{
            f"arm_{arm}": np.asarray(arm_q_hold, dtype=np.float64).reshape(7).tolist(),
            f"gripper_effort_{arm}": effort_list,
        }
    )


def _object_root_pos_np(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    try:
        p, _ = obj.get_position_orientation()
        return np.asarray(_to_np(p), dtype=np.float64).reshape(3)
    except Exception:
        return None


def _aabb_np(entity) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if entity is None:
        return None
    try:
        lo, hi = entity.aabb
        return (
            np.asarray(_to_np(lo), dtype=np.float64).reshape(3),
            np.asarray(_to_np(hi), dtype=np.float64).reshape(3),
        )
    except Exception:
        return None


def _point_aabb_clearance(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return float(np.linalg.norm(outside))


def _collision_boundary_aabb_np(entity) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        pts = getattr(entity, "collision_boundary_points_world", None)
        if pts is None:
            return None
        pts_np = np.asarray(_to_np(pts), dtype=np.float64).reshape(-1, 3)
        if pts_np.size == 0:
            return None
        return np.min(pts_np, axis=0), np.max(pts_np, axis=0)
    except Exception:
        return None


def _segment_intersects_aabb(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    d = b - a
    t_min = 0.0
    t_max = 1.0
    for i in range(3):
        if abs(float(d[i])) < 1e-12:
            if a[i] < lo[i] or a[i] > hi[i]:
                return False
            continue
        inv = 1.0 / float(d[i])
        t1 = float((lo[i] - a[i]) * inv)
        t2 = float((hi[i] - a[i]) * inv)
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _segment_aabb_clearance(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    if _segment_intersects_aabb(a, b, lo, hi):
        return 0.0
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    # Diagnostic-quality minimum: sample densely along the short gripper rays.
    best = min(_point_aabb_clearance(a, lo, hi), _point_aabb_clearance(b, lo, hi))
    for t in np.linspace(0.0, 1.0, 41):
        p = a + (b - a) * float(t)
        best = min(best, _point_aabb_clearance(p, lo, hi))
    return float(best)


def _log_grasp_contact_geometry(ctx, world, arm: str, obj, *, label: str) -> None:
    """Exec-side contact diagnostics: object AABB and finger link centers/AABBs."""
    if ctx is None:
        return
    try:
        obj_ab = _aabb_np(obj)
        if obj_ab is None:
            ctx.log(f"  [contact-diag:{label}] object AABB unavailable")
            return
        obj_lo, obj_hi = obj_ab
        obj_ctr = 0.5 * (obj_lo + obj_hi)
        ctx.log(
            f"  [contact-diag:{label}] obj_aabb "
            f"x=[{obj_lo[0]:.3f},{obj_hi[0]:.3f}] "
            f"y=[{obj_lo[1]:.3f},{obj_hi[1]:.3f}] "
            f"z=[{obj_lo[2]:.3f},{obj_hi[2]:.3f}] "
            f"ctr=({obj_ctr[0]:.3f},{obj_ctr[1]:.3f},{obj_ctr[2]:.3f})"
        )
        for fln in world.robot.finger_link_names.get(arm, []):
            lk = world.robot.links.get(fln)
            if lk is None:
                continue
            p, _ = lk.get_position_orientation()
            fp = np.asarray(_to_np(p), dtype=np.float64).reshape(3)
            clear = _point_aabb_clearance(fp, obj_lo, obj_hi)
            link_ab = _aabb_np(lk)
            msg = (
                f"  [contact-diag:{label}] {fln} center="
                f"({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f}) "
                f"center_to_obj_aabb={clear*1000:.1f}mm"
            )
            if link_ab is not None:
                lo, hi = link_ab
                overlap = np.minimum(hi, obj_hi) - np.maximum(lo, obj_lo)
                ov = np.maximum(overlap, 0.0)
                ov_vol_cm3 = float(ov[0] * ov[1] * ov[2] * 1e6)
                msg += (
                    f" link_aabb_x=[{lo[0]:.3f},{hi[0]:.3f}]"
                    f" y=[{lo[1]:.3f},{hi[1]:.3f}]"
                    f" z=[{lo[2]:.3f},{hi[2]:.3f}]"
                    f" link_obj_overlap_cm3={ov_vol_cm3:.3f}"
                )
            coll_ab = _collision_boundary_aabb_np(lk)
            if coll_ab is not None:
                clo, chi = coll_ab
                coverlap = np.minimum(chi, obj_hi) - np.maximum(clo, obj_lo)
                cov = np.maximum(coverlap, 0.0)
                cov_vol_cm3 = float(cov[0] * cov[1] * cov[2] * 1e6)
                msg += (
                    f" coll_aabb_x=[{clo[0]:.3f},{chi[0]:.3f}]"
                    f" y=[{clo[1]:.3f},{chi[1]:.3f}]"
                    f" z=[{clo[2]:.3f},{chi[2]:.3f}]"
                    f" coll_obj_overlap_cm3={cov_vol_cm3:.3f}"
                )
            ctx.log(msg)
    except Exception as exc:
        ctx.log(f"  [contact-diag:{label}] failed: {type(exc).__name__}: {exc}")


def _finger_entity_overlap_stats(
    world,
    arm: str,
    entity,
    *,
    min_overlap_cm3: float = 1.0,
) -> dict:
    """两指 AABB 与任意 entity（物体或单个 link）的重叠统计。"""
    stats = {
        "ok": False,
        "finger_count": 0,
        "overlap_count": 0,
        "overlaps_cm3": [],
        "contact_pos": None,
        "error": None,
    }
    try:
        ent_ab = _collision_boundary_aabb_np(entity) or _aabb_np(entity)
        if ent_ab is None:
            stats["error"] = "entity_aabb_unavailable"
            return stats
        ent_lo, ent_hi = ent_ab
        centers = []
        overlaps = []
        for fln in world.robot.finger_link_names.get(arm, []):
            lk = world.robot.links.get(fln)
            if lk is None:
                continue
            stats["finger_count"] += 1
            link_ab = _collision_boundary_aabb_np(lk) or _aabb_np(lk)
            if link_ab is None:
                continue
            lo, hi = link_ab
            inter_lo = np.maximum(lo, ent_lo)
            inter_hi = np.minimum(hi, ent_hi)
            inter = np.maximum(inter_hi - inter_lo, 0.0)
            vol_cm3 = float(inter[0] * inter[1] * inter[2] * 1e6)
            overlaps.append(vol_cm3)
            if vol_cm3 > 0.0:
                stats["overlap_count"] += 1
                centers.append(0.5 * (inter_lo + inter_hi))
        stats["overlaps_cm3"] = overlaps
        if centers:
            stats["contact_pos"] = np.mean(np.asarray(centers, dtype=np.float64), axis=0)
        stats["ok"] = bool(
            stats["finger_count"] >= 2
            and stats["overlap_count"] >= 2
            and min(overlaps or [0.0]) >= float(min_overlap_cm3)
        )
        return stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats


def _finger_object_overlap_stats(world, arm: str, obj) -> dict:
    return _finger_entity_overlap_stats(world, arm, obj, min_overlap_cm3=1.0)


def _object_root_prim_path(obj) -> str:
    try:
        root = getattr(obj, "root_link", None)
        return str(getattr(root, "prim_path", "") or "")
    except Exception:
        return ""


def _object_non_root_links(obj) -> list:
    """fixed-base 铰接体上可抓的活动 link（门/抽屉），排除 root。"""
    out = []
    root_path = _object_root_prim_path(obj)
    try:
        links = list((getattr(obj, "links", None) or {}).values())
    except Exception:
        links = []
    for link in links:
        try:
            path = str(getattr(link, "prim_path", "") or "")
        except Exception:
            path = ""
        if not path:
            continue
        if root_path and path == root_path:
            continue
        out.append(link)
    return out


def _best_graspable_ag_link(world, arm: str, obj):
    """为 AG 选择最合适的约束 link。

    - fixed_base：必须选非 root 活动 link（柜门/抽屉），否则 OG 会拒绝
    - 可动物体：优先双指重叠最大的 link，否则退回 root
    """
    if obj is None:
        return None, None
    is_fixed = bool(getattr(obj, "fixed_base", False))
    # 薄门板允许更低的单指重叠体积阈值
    min_ov = 0.05 if is_fixed else 1.0
    ranked = []
    candidates = _object_non_root_links(obj) if is_fixed else list(
        (getattr(obj, "links", None) or {}).values()
    )
    if not candidates and not is_fixed:
        try:
            root = getattr(obj, "root_link", None)
        except Exception:
            root = None
        if root is not None:
            candidates = [root]
    for link in candidates:
        ov = _finger_entity_overlap_stats(
            world, arm, link, min_overlap_cm3=min_ov
        )
        overlaps = [float(x) for x in (ov.get("overlaps_cm3") or [])]
        if not overlaps or max(overlaps) <= 0.0:
            continue
        ranked.append((
            int(ov.get("overlap_count") or 0),
            float(sum(overlaps)),
            float(min(overlaps)),
            link,
            ov,
        ))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if ranked:
        return ranked[0][3], ranked[0][4]
    if is_fixed:
        return None, None
    try:
        root = getattr(obj, "root_link", None)
    except Exception:
        root = None
    return root, _finger_object_overlap_stats(world, arm, obj)


def _short_prim_path(path) -> str:
    s = str(path)
    parts = [p for p in s.split("/") if p]
    if len(parts) <= 3:
        return s
    return ".../" + "/".join(parts[-3:])


def _object_link_prim_paths(obj) -> set[str]:
    paths: set[str] = set()
    if obj is None:
        return paths
    try:
        prim = getattr(obj, "prim_path", None)
        if prim:
            paths.add(str(prim))
    except Exception:
        pass
    try:
        for link in getattr(obj, "links", {}).values():
            prim = getattr(link, "prim_path", None)
            if prim:
                paths.add(str(prim))
    except Exception:
        pass
    return paths


def _same_og_object(a, b) -> bool:
    if a is None or b is None:
        return False
    if a is b:
        return True
    for attr in ("prim_path", "name"):
        try:
            if getattr(a, attr, None) and getattr(a, attr, None) == getattr(b, attr, None):
                return True
        except Exception:
            pass
    return False


def _ag_obj_name(obj) -> str:
    if obj is None:
        return "None"
    for attr in ("name", "prim_path"):
        try:
            val = getattr(obj, attr, None)
            if val:
                return str(val)
        except Exception:
            pass
    return type(obj).__name__


def _ag_state_summary(world, arm: str) -> str:
    robot = getattr(world, "robot", None)
    if robot is None:
        return "robot=None"
    try:
        held = getattr(robot, "_ag_obj_in_hand", {}).get(arm)
    except Exception:
        held = None
    try:
        counter = getattr(robot, "_ag_grasp_counter", {}).get(arm)
    except Exception:
        counter = None
    try:
        release = getattr(robot, "_ag_release_counter", {}).get(arm)
    except Exception:
        release = None
    try:
        freeze = getattr(robot, "_ag_freeze_gripper", {}).get(arm)
    except Exception:
        freeze = None
    try:
        mode = getattr(robot, "grasping_mode", None)
    except Exception:
        mode = None
    return (
        f"mode={mode} held={_ag_obj_name(held)} "
        f"counter={counter} release={release} freeze={freeze}"
    )


def _ag_data_summary(ag_data) -> str:
    if ag_data is None:
        return "None"
    try:
        ag_obj, ag_link = ag_data[:2]
    except Exception:
        return str(ag_data)
    try:
        link_path = getattr(ag_link, "prim_path", None) or getattr(ag_link, "name", None)
    except Exception:
        link_path = None
    return f"{_ag_obj_name(ag_obj)} link={_short_prim_path(link_path)}"


def _ag_point_world(robot, gp) -> Optional[np.ndarray]:
    try:
        link = robot.links[str(gp.link_name)]
        lp, lq = link.get_position_orientation()
        local = np.asarray(_to_np(gp.position), dtype=np.float64).reshape(3)
        return np.asarray(_to_np(lp), dtype=np.float64).reshape(3) + _quat_rot(
            np.asarray(_to_np(lq), dtype=np.float64).reshape(4),
            local,
        )
    except Exception:
        return None


def _assisted_grasp_ray_aabb_stats(world, arm: str, obj) -> dict:
    stats = {
        "ok": False,
        "n_start": 0,
        "n_end": 0,
        "total": 0,
        "hits": 0,
        "best_dist_m": float("inf"),
        "best_start": None,
        "best_end": None,
        "best_i": None,
        "best_j": None,
        "best_len_m": None,
        "error": None,
    }
    robot = getattr(world, "robot", None)
    if robot is None or obj is None:
        stats["error"] = "robot_or_object_unavailable"
        return stats
    obj_ab = _aabb_np(obj)
    if obj_ab is None:
        stats["error"] = "object_aabb_unavailable"
        return stats
    obj_lo, obj_hi = obj_ab
    try:
        starts = list(robot.assisted_grasp_start_points.get(arm) or [])
        ends = list(robot.assisted_grasp_end_points.get(arm) or [])
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats
    start_world = [_ag_point_world(robot, gp) for gp in starts]
    end_world = [_ag_point_world(robot, gp) for gp in ends]
    start_world = [p for p in start_world if p is not None]
    end_world = [p for p in end_world if p is not None]
    stats["n_start"] = len(start_world)
    stats["n_end"] = len(end_world)
    if not start_world or not end_world:
        stats["error"] = f"no_usable_ag_points raw={len(starts)}/{len(ends)}"
        return stats
    best = None
    hit_count = 0
    for si, sp in enumerate(start_world):
        for ei, ep in enumerate(end_world):
            dist = _segment_aabb_clearance(sp, ep, obj_lo, obj_hi)
            hit = dist <= 1e-9
            hit_count += 1 if hit else 0
            length = float(np.linalg.norm(ep - sp))
            cand = (float(dist), si, ei, length, sp, ep)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        stats["error"] = "no_ray_pairs"
        return stats
    dist, si, ei, length, sp, ep = best
    stats.update({
        "ok": bool(hit_count > 0 or float(dist) <= 0.002),
        "total": int(len(start_world) * len(end_world)),
        "hits": int(hit_count),
        "best_dist_m": float(dist),
        "best_start": sp,
        "best_end": ep,
        "best_i": int(si),
        "best_j": int(ei),
        "best_len_m": float(length),
    })
    return stats


def _log_assisted_grasp_ray_geometry(ctx, world, arm: str, obj, *, label: str) -> dict:
    stats = _assisted_grasp_ray_aabb_stats(world, arm, obj)
    if ctx is None:
        return stats
    if stats.get("error"):
        ctx.log(f"  [ag-ray:{label}] unavailable: {stats.get('error')}")
        return stats
    sp = np.asarray(stats["best_start"], dtype=np.float64).reshape(3)
    ep = np.asarray(stats["best_end"], dtype=np.float64).reshape(3)
    ctx.log(
        f"  [ag-ray-aabb-diag:{label}] n_start={stats['n_start']} n_end={stats['n_end']} "
        f"hits={stats['hits']}/{stats['total']} diagnostic_only=1 "
        f"best_dist={stats['best_dist_m']*1000:.1f}mm "
        f"best={stats['best_i']}->{stats['best_j']} "
        f"len={stats['best_len_m']*1000:.1f}mm "
        f"start=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) "
        f"end=({ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f})"
    )
    return stats


def _ag_target_link(obj):
    if obj is None:
        return
    try:
        root = getattr(obj, "root_link", None)
        if root is not None:
            return root
    except Exception:
        pass
    try:
        links = list(getattr(obj, "links", {}).values())
        if links:
            return links[0]
    except Exception:
        pass
    return None


def _log_assisted_grasp_diag(ctx, world, arm: str, obj, *, label: str) -> dict:
    """Log OG assisted-grasp inputs: true finger contacts, grasp raycasts, and AG state."""
    out = {
        "contacts": set(),
        "robot_contact_links": {},
        "raycasts": set(),
        "ag_data": None,
    }
    if ctx is None:
        return out
    robot = getattr(world, "robot", None)
    if robot is None:
        ctx.log(f"  [ag-diag:{label}] robot unavailable")
        return out
    obj_paths = _object_link_prim_paths(obj)

    try:
        contacts, robot_contact_links = robot._find_gripper_contacts(arm=arm)
        contacts = {str(p) for p in contacts}
        robot_contact_links = {
            str(k): {str(vv) for vv in vals}
            for k, vals in dict(robot_contact_links).items()
        }
        out["contacts"] = contacts
        out["robot_contact_links"] = robot_contact_links
    except Exception as exc:
        contacts = set()
        robot_contact_links = {}
        ctx.log(f"  [ag-diag:{label}] _find_gripper_contacts failed: {type(exc).__name__}: {exc}")

    try:
        raycasts = {str(p) for p in robot._find_gripper_raycast_collisions(arm=arm)}
        out["raycasts"] = raycasts
    except Exception as exc:
        raycasts = set()
        ctx.log(f"  [ag-diag:{label}] _find_gripper_raycast_collisions failed: {type(exc).__name__}: {exc}")

    contact_obj = sorted([p for p in contacts if p in obj_paths])
    ray_obj = sorted([p for p in raycasts if p in obj_paths])
    both_obj = sorted(set(contact_obj).intersection(ray_obj))

    try:
        ag_data = robot._calculate_in_hand_object(arm=arm)
        out["ag_data"] = ag_data
    except Exception as exc:
        ag_data = None
        ctx.log(f"  [ag-diag:{label}] _calculate_in_hand_object failed: {type(exc).__name__}: {exc}")

    contact_preview = ",".join(_short_prim_path(p) for p in sorted(contacts)[:4])
    ray_preview = ",".join(_short_prim_path(p) for p in sorted(raycasts)[:4])
    contact_links_preview = []
    for p in contact_obj[:3]:
        links = sorted(robot_contact_links.get(p, []))
        contact_links_preview.append(
            f"{_short_prim_path(p)}<-{','.join(_short_prim_path(x) for x in links)}"
        )
    ctx.log(
        f"  [ag-diag:{label}] {_ag_state_summary(world, arm)} "
        f"contacts={len(contacts)} obj_contacts={len(contact_obj)} "
        f"raycasts={len(raycasts)} obj_raycasts={len(ray_obj)} "
        f"obj_both={len(both_obj)} ag_candidate={_ag_data_summary(ag_data)}"
    )
    if contact_preview or ray_preview or contact_links_preview:
        ctx.log(
            f"  [ag-diag:{label}] contact_preview=[{contact_preview}] "
            f"ray_preview=[{ray_preview}] "
            f"obj_contact_links=[{'; '.join(contact_links_preview)}]"
        )
    _log_assisted_grasp_ray_geometry(ctx, world, arm, obj, label=label)
    return out


def _current_assisted_grasp_object(world, arm: str):
    """Return the object held by a live core-assisted-grasp constraint."""
    robot = getattr(world, "robot", None)
    if robot is None:
        return None
    try:
        held = getattr(robot, "_ag_obj_in_hand", {}).get(arm)
    except Exception:
        held = None
    if held is None:
        return None
    constraints = getattr(robot, "_ag_obj_constraints", None)
    if isinstance(constraints, dict) and constraints.get(arm) is None:
        return None
    return held


def _official_grasp_active(world, arm: str) -> bool:
    """Read the public grasp proprioception state without inspecting scene state."""
    robot = getattr(world, "robot", None)
    is_grasping = getattr(robot, "is_grasping", None)
    if not callable(is_grasping):
        return False
    try:
        state = is_grasping(arm=arm)
    except TypeError:
        try:
            state = is_grasping(arm)
        except Exception:
            return False
    except Exception:
        return False
    name = str(getattr(state, "name", state)).upper().strip()
    if name == "TRUE":
        return True
    if name in {"FALSE", "UNKNOWN"}:
        return False
    try:
        return int(state) == 1
    except (TypeError, ValueError):
        return False


def _assisted_grasp_hold_frames(world, *, safety_frames: int = 2) -> int:
    """Return enough yielded close actions to cover the core assisted-grasp window."""
    window_s = 0.3
    try:
        from omnigibson.robots import manipulation_robot

        window_s = float(manipulation_robot.m.GRASP_WINDOW)
    except Exception:
        pass
    action_dt = _action_dt_s(world)
    return max(1, int(math.ceil(window_s / action_dt)) + max(0, int(safety_frames)))


def _sample_aabb_surface_points(lo, hi, *, n: int = 7) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    xs = np.linspace(float(lo[0]), float(hi[0]), max(2, int(n)))
    ys = np.linspace(float(lo[1]), float(hi[1]), max(2, int(n)))
    zs = np.linspace(float(lo[2]), float(hi[2]), max(2, int(n)))
    pts: list[list[float]] = []
    for x in (float(lo[0]), float(hi[0])):
        for y in ys:
            for z in zs:
                pts.append([x, float(y), float(z)])
    for y in (float(lo[1]), float(hi[1])):
        for x in xs:
            for z in zs:
                pts.append([float(x), y, float(z)])
    for z in (float(lo[2]), float(hi[2])):
        for x in xs:
            for y in ys:
                pts.append([float(x), float(y), z])
    return np.asarray(pts, dtype=np.float64)


def _grasp_close_qpos_from_object_width(
    obj,
    target_pos,
    target_quat,
    *,
    ctx=None,
) -> Optional[list[float]]:
    """Pick an absolute finger qpos that squeezes the object instead of full-closing.

    R1Pro gripper commands are absolute finger joint positions: q=0.05 is open,
    q=0.0 is fully closed.  For small rigid objects, commanding 0.0 sweeps the
    fingers through the entire gap and can push the object out before contact
    settles.  Estimate object width along the planned gripper Y axis and close
    only a few millimeters past that width.
    """
    if obj is None:
        return None
    try:
        from behavior_interface.skills.grasp import _quat_to_mat
        from behavior_interface.skills.plan_grasp_gripper_geom import (
            gap_contact_z_bounds,
            get_wedge_lut,
        )

        lo, hi = obj.aabb
        pts_w = _sample_aabb_surface_points(_to_np(lo), _to_np(hi), n=7)
        R = _quat_to_mat(_quat_normalize_xyzw(target_quat))
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        pts_e = (R.T @ (pts_w - target_pos).T).T
        z_tip, z_hi = gap_contact_z_bounds()
        lut = get_wedge_lut()
        z_vals = np.asarray(lut["z"], dtype=np.float64)
        xh = np.interp(
            np.clip(pts_e[:, 2], float(z_vals.min()), float(z_vals.max())),
            z_vals,
            np.asarray(lut["x_half_gap"], dtype=np.float64),
        )
        contact = (
            (pts_e[:, 2] >= float(z_tip) - 0.012)
            & (pts_e[:, 2] <= float(z_hi) + 0.012)
            & (np.abs(pts_e[:, 0]) <= xh + 0.035)
        )
        pts_c = pts_e[contact]
        if len(pts_c) < 6:
            pts_c = pts_e
        y_lo = float(np.percentile(pts_c[:, 1], 5))
        y_hi = float(np.percentile(pts_c[:, 1], 95))
        width_y = max(0.0, y_hi - y_lo)
        z_med = float(np.median(pts_c[:, 2]))
        z_med = float(np.clip(z_med, float(z_vals.min()), float(z_vals.max())))
        gap_open = float(np.interp(z_med, z_vals, np.asarray(lut["gap_width"], dtype=np.float64)))
        if not np.isfinite(gap_open) or gap_open <= 0.0:
            gap_open = 0.10
        # Close past the measured width enough to create contact force.  The
        # LUT gap is conservative for the R1Pro fingers, so leave a small
        # nonzero finger command but bias toward a real squeeze for cans.
        squeeze_m = 0.045
        min_closure_m = 0.018
        desired_gap = max(0.020, width_y - squeeze_m)
        desired_gap = min(desired_gap, max(0.020, gap_open - min_closure_m))
        q_open = 0.050
        q_target = q_open - max(0.0, gap_open - desired_gap) * 0.5
        q_target = float(np.clip(q_target, 0.006, q_open))
        if ctx is not None:
            ctx.log(
                "  [exec-grip] adaptive close target "
                f"width_y={width_y*1000:.1f}mm gap_open={gap_open*1000:.1f}mm "
                f"desired_gap={desired_gap*1000:.1f}mm q={q_target:.4f}"
            )
        return [q_target, q_target]
    except Exception as exc:
        if ctx is not None:
            ctx.log(f"  [exec-grip] adaptive close target failed: {type(exc).__name__}: {exc}")
        return None


def _effort_close_gripper_hold_wrist(
    world,
    arm: str,
    *,
    ctx=None,
    n_hold: int,
):
    """Run effort close without camera rendering and restore the prior mode."""
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    world._codex_fast_motion_no_obs = True
    try:
        return (yield from _effort_close_gripper_hold_wrist_impl(
            world,
            arm,
            ctx=ctx,
            n_hold=n_hold,
        ))
    finally:
        world._codex_fast_motion_no_obs = old_no_obs


def _effort_close_gripper_hold_wrist_impl(
    world,
    arm: str,
    *,
    ctx=None,
    n_hold: int,
):
    """Close gently, then require the official assisted-grasp state."""
    from behavior_interface.skills.grasp import _arm_qpos

    q_start, _ = _finger_qpos_qvel(world, arm)
    q_lower = _finger_lower_limits(world, arm)
    q_hold = _arm_qpos(world, arm).tolist()
    dt = _action_dt_s(world)
    stall_frames = max(4, int(math.ceil(_GRIPPER_STALL_CONFIRM_S / dt)))
    seek_frames = max(stall_frames + 1, int(math.ceil(_GRIPPER_SEEK_TIMEOUT_S / dt)))
    blocked = np.zeros(2, dtype=bool)
    moved = np.zeros(2, dtype=bool)
    stall_count = np.zeros(2, dtype=np.int64)
    stall_anchor_qpos = q_start.copy()
    contact_qpos = np.full(2, np.nan, dtype=np.float64)
    vel_ema = np.zeros(2, dtype=np.float64)
    last_q = q_start.copy()
    official_grasp = _official_grasp_active(world, arm)
    bilateral_stall = False
    stop_reason = "already_grasping" if official_grasp else "sim_timeout"
    seek_started = time.monotonic()
    seek_steps = 0
    contact_reset_count = 0
    max_contact_closure_drift_m = 0.0

    if ctx:
        ctx.log(
            "  [effort-grip] seek "
            f"force={_GRIPPER_INITIAL_SEEK_FORCE_N:.2f}-"
            f"{_GRIPPER_SEEK_FORCE_N:.2f}N "
            f"stall={stall_frames}frames timeout={seek_frames}frames/"
            f"{_GRIPPER_SEEK_WALL_TIMEOUT_S:.0f}s wall "
            f"action_dt={dt:.4f}s"
        )

    for step in range(seek_frames):
        if official_grasp or _official_grasp_active(world, arm):
            official_grasp = True
            stop_reason = "official_grasp"
            break
        if time.monotonic() - seek_started >= _GRIPPER_SEEK_WALL_TIMEOUT_S:
            stop_reason = "wall_timeout"
            break

        qpos, qvel = _finger_qpos_qvel(world, arm)
        vel_ema = qvel if step == 0 else 0.65 * vel_ema + 0.35 * qvel
        moved |= (q_start - qpos) >= _GRIPPER_MIN_TRAVEL_M
        moved |= vel_ema <= -_GRIPPER_STALL_VEL_M_S
        at_lower_limit = qpos <= q_lower + _GRIPPER_LOWER_LIMIT_MARGIN_M

        for finger_i in range(2):
            if blocked[finger_i]:
                closure_drift = max(
                    0.0,
                    float(contact_qpos[finger_i] - qpos[finger_i]),
                )
                max_contact_closure_drift_m = max(
                    max_contact_closure_drift_m,
                    closure_drift,
                )
                if (
                    closure_drift > _GRIPPER_CONTACT_DRIFT_M
                    or at_lower_limit[finger_i]
                ):
                    blocked[finger_i] = False
                    stall_count[finger_i] = 0
                    stall_anchor_qpos[finger_i] = qpos[finger_i]
                    contact_qpos[finger_i] = np.nan
                    contact_reset_count += 1
                continue

            other_i = 1 - finger_i
            observable_engagement = bool(
                moved[finger_i]
                or (
                    step >= stall_frames
                    and qpos[finger_i]
                    < _GRIPPER_UPPER_LIMIT_M
                    - _GRIPPER_UPPER_CONTACT_MARGIN_M
                )
                or (moved[other_i] and step >= stall_frames)
            )
            is_stalled = bool(
                observable_engagement
                and not at_lower_limit[finger_i]
                and float(vel_ema[finger_i]) <= _GRIPPER_STALL_MAX_OPENING_VEL_M_S
                and abs(float(qpos[finger_i] - last_q[finger_i])) < 0.0005
                and abs(
                    float(qpos[finger_i] - stall_anchor_qpos[finger_i])
                ) < _GRIPPER_STALL_WINDOW_DRIFT_M
            )
            if is_stalled:
                stall_count[finger_i] += 1
            else:
                stall_count[finger_i] = 0
                stall_anchor_qpos[finger_i] = qpos[finger_i]
            if stall_count[finger_i] >= stall_frames:
                blocked[finger_i] = True
                contact_qpos[finger_i] = qpos[finger_i]
                if ctx:
                    ctx.log(
                        f"  [effort-grip] finger{finger_i + 1} blocked "
                        f"q={qpos[finger_i]:.4f} v_ema={vel_ema[finger_i]:+.5f}"
                    )

        if bool(np.all(blocked)):
            bilateral_stall = True
            stop_reason = "bilateral_stall"
            seek_steps = step
            break
        if bool(np.all(at_lower_limit)):
            stop_reason = "finger_lower_limits"
            break

        effort = np.zeros(2, dtype=np.float64)
        for finger_i in range(2):
            if at_lower_limit[finger_i]:
                effort[finger_i] = 0.0
            elif blocked[finger_i]:
                effort[finger_i] = -_GRIPPER_BLOCKED_HOLD_FORCE_N
            else:
                seek_force = _GRIPPER_INITIAL_SEEK_FORCE_N
                if (
                    not moved[finger_i]
                    and qpos[finger_i] > q_lower[finger_i] + 0.04
                ):
                    seek_force = min(
                        _GRIPPER_SEEK_FORCE_N,
                        _GRIPPER_INITIAL_SEEK_FORCE_N
                        + (
                            _GRIPPER_SEEK_FORCE_N
                            - _GRIPPER_INITIAL_SEEK_FORCE_N
                        )
                        * min(1.0, step / max(1.0, 0.5 / dt)),
                    )
                effort[finger_i] = -seek_force
        last_q = qpos.copy()
        seek_steps = step + 1
        yield _gripper_effort_hold_action(world, arm, q_hold, effort)
    official_grasp = official_grasp or _official_grasp_active(world, arm)

    if not official_grasp and not bilateral_stall:
        qpos, qvel = _finger_qpos_qvel(world, arm)
        fallback_effort = [
            -_GRIPPER_INITIAL_SEEK_FORCE_N,
            -_GRIPPER_INITIAL_SEEK_FORCE_N,
        ]
        yield _gripper_effort_hold_action(world, arm, q_hold, fallback_effort)
        if ctx:
            ctx.log(
                "  [effort-grip] official grasp not established; retain low-force close "
                f"reason={stop_reason} steps={seek_steps} "
                f"q={qpos.round(5).tolist()} v={qvel.round(5).tolist()} "
                f"blocked={blocked.tolist()}"
            )
        return {
            "mode": "effort",
            "bilateral_stall": False,
            "official_grasp": False,
            "success": False,
            "reason": stop_reason,
            "seek_steps": seek_steps,
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            "blocked": blocked.tolist(),
            "fallback_effort_n": list(fallback_effort),
            "close_keepalive_latched": bool(
                getattr(
                    world,
                    "gripper_close_keepalive_active",
                    lambda _arm: False,
                )(arm)
            ),
        }

    hold_frames = max(int(n_hold), _assisted_grasp_hold_frames(world))
    hold_steps = 0
    if not official_grasp and ctx:
        q_contact, _ = _finger_qpos_qvel(world, arm)
        ctx.log(
            "  [effort-grip] bilateral stall confirmed; gentle official-window hold "
            f"q={q_contact.round(5).tolist()} "
            f"force={_GRIPPER_CONFIRM_FORCE_N:.2f}N "
            f"frames={hold_frames}"
        )
    while not official_grasp and hold_steps < hold_frames:
        q_confirm, _ = _finger_qpos_qvel(world, arm)
        closure_drift = np.maximum(contact_qpos - q_confirm, 0.0)
        max_contact_closure_drift_m = max(
            max_contact_closure_drift_m,
            float(np.nanmax(closure_drift)),
        )
        if bool(np.any(closure_drift > _GRIPPER_CONTACT_DRIFT_M)):
            contact_reset_count += 1
            contact_qpos = q_confirm.copy()
        yield _gripper_effort_hold_action(
            world,
            arm,
            q_hold,
            [-_GRIPPER_CONFIRM_FORCE_N, -_GRIPPER_CONFIRM_FORCE_N],
        )
        hold_steps += 1
        official_grasp = _official_grasp_active(world, arm)
    if official_grasp and stop_reason == "bilateral_stall":
        stop_reason = "official_grasp_during_hold"

    if not official_grasp:
        q_final, qvel_final = _finger_qpos_qvel(world, arm)
        fallback_effort = [
            -_GRIPPER_INITIAL_SEEK_FORCE_N,
            -_GRIPPER_INITIAL_SEEK_FORCE_N,
        ]
        yield _gripper_effort_hold_action(world, arm, q_hold, fallback_effort)
        if ctx:
            ctx.log(
                "  [effort-grip] bilateral contact did not become an official grasp; "
                "retain low-force close "
                f"q={q_final.round(5).tolist()} hold_steps={hold_steps}"
            )
        return {
            "mode": "effort",
            "bilateral_stall": True,
            "official_grasp": False,
            "success": False,
            "reason": "official_grasp_timeout",
            "seek_steps": seek_steps,
            "hold_steps": hold_steps,
            "qpos": q_final.tolist(),
            "qvel": qvel_final.tolist(),
            "blocked": blocked.tolist(),
            "contact_reset_count": contact_reset_count,
            "max_contact_closure_drift_m": max_contact_closure_drift_m,
            "fallback_effort_n": list(fallback_effort),
            "close_keepalive_latched": bool(
                getattr(
                    world,
                    "gripper_close_keepalive_active",
                    lambda _arm: False,
                )(arm)
            ),
        }

    carry_effort = [-_GRIPPER_CARRY_FORCE_N, -_GRIPPER_CARRY_FORCE_N]
    for _ in range(2):
        yield _gripper_effort_hold_action(
            world,
            arm,
            q_hold,
            carry_effort,
        )
    q_final, qvel_final = _finger_qpos_qvel(world, arm)
    official_grasp = _official_grasp_active(world, arm)
    if not official_grasp:
        return {
            "mode": "effort",
            "bilateral_stall": bool(bilateral_stall),
            "official_grasp": False,
            "success": False,
            "reason": "official_grasp_lost_during_carry",
            "qpos": q_final.tolist(),
            "qvel": qvel_final.tolist(),
            "blocked": blocked.tolist(),
            "contact_reset_count": contact_reset_count,
            "max_contact_closure_drift_m": max_contact_closure_drift_m,
            "carry_effort_n": list(carry_effort),
            "close_keepalive_latched": bool(
                getattr(
                    world,
                    "gripper_close_keepalive_active",
                    lambda _arm: False,
                )(arm)
            ),
        }
    if ctx:
        ctx.log(
            "  [effort-grip] official assisted grasp active; "
            f"carry={_GRIPPER_CARRY_FORCE_N:.2f}N reason={stop_reason}"
        )
    latch_keepalive = getattr(world, "latch_gripper_close_keepalive", None)
    if callable(latch_keepalive):
        latch_keepalive(arm, effort=carry_effort)
    return {
        "mode": "effort",
        "bilateral_stall": bool(bilateral_stall),
        "official_grasp": True,
        "success": True,
        "reason": stop_reason,
        "seek_steps": seek_steps,
        "hold_steps": hold_steps,
        "qpos": q_final.tolist(),
        "qvel": qvel_final.tolist(),
        "blocked": blocked.tolist(),
        "contact_reset_count": contact_reset_count,
        "max_contact_closure_drift_m": max_contact_closure_drift_m,
        "carry_effort_n": list(carry_effort),
        "close_keepalive_latched": bool(
            getattr(world, "gripper_close_keepalive_active", lambda _arm: False)(arm)
        ),
    }


def _slow_close_gripper_hold_wrist(
    world,
    arm: str,
    wrist_pos,
    wrist_quat,
    *,
    q_closed_cmd,
    ctx=None,
    n_ramp: int = 30,
    n_hold: int = 18,
):
    """Close the gripper gradually while holding the wrist.

    Effort controllers use bounded seeking, bilateral stall detection, and a
    gentle official-window hold. Position controllers retain the legacy ramp.
    """
    q_end_probe = _gripper_cmd_override(q_closed_cmd)
    q_now_probe = _finger_q_values(_read_finger_qpos(world, arm))
    close_requested = bool(q_end_probe and float(q_end_probe[0]) < 0.0)
    if q_end_probe and q_now_probe and len(q_end_probe) == len(q_now_probe):
        close_requested = float(np.mean(q_end_probe)) < float(np.mean(q_now_probe)) - 1e-5
    if _gripper_uses_effort(world, arm) and close_requested:
        return (yield from _effort_close_gripper_hold_wrist(
            world,
            arm,
            ctx=ctx,
            n_hold=n_hold,
        ))

    q_start = _current_gripper_qpos_cmd(world, arm)
    q_end = _gripper_cmd_override(q_closed_cmd)
    if not q_end:
        q_end = [-1.0]
    if not q_start or len(q_start) != len(q_end):
        q_start = _finger_q_values(_read_finger_qpos(world, arm))
    if not q_start or len(q_start) != len(q_end):
        q_start = [0.05] * len(q_end)
    q_start_arr = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_end_arr = np.asarray(q_end, dtype=np.float64).reshape(-1)
    if ctx:
        ctx.log(
            "  close gripper slow-ramp "
            f"start={[round(float(x), 4) for x in q_start_arr.tolist()]} "
            f"end={[round(float(x), 4) for x in q_end_arr.tolist()]} "
            f"frames={int(n_ramp)}+{int(n_hold)}"
        )

    if int(n_ramp) <= 1 and int(n_hold) <= 1:
        yield from _hold_pose(
            world,
            arm,
            wrist_pos,
            wrist_quat,
            n_frames=2,
            gripper_cmd=q_end_arr.tolist(),
        )
        return
    for i in range(max(1, int(n_ramp))):
        alpha = float(i + 1) / float(max(1, int(n_ramp)))
        q_cmd = (1.0 - alpha) * q_start_arr + alpha * q_end_arr
        yield from _hold_pose(
            world,
            arm,
            wrist_pos,
            wrist_quat,
            n_frames=1,
            gripper_cmd=q_cmd.tolist(),
        )
    if int(n_hold) > 0:
        yield from _hold_pose(
            world,
            arm,
            wrist_pos,
            wrist_quat,
            n_frames=int(n_hold),
            gripper_cmd=q_end_arr.tolist(),
        )
    official_grasp = _official_grasp_active(world, arm)
    if official_grasp:
        latch_keepalive = getattr(world, "latch_gripper_close_keepalive", None)
        if callable(latch_keepalive):
            latch_keepalive(arm)
    return {
        "mode": "position",
        "bilateral_stall": None,
        "official_grasp": bool(official_grasp),
        "success": bool(official_grasp),
        "close_keepalive_latched": bool(
            getattr(world, "gripper_close_keepalive_active", lambda _arm: False)(arm)
        ),
    }


def _ensure_world_pinned_actions(world) -> None:
    """Hot-reload 后给已有 WorldAPI 实例补上默认锁关节 action。"""
    if getattr(world, "dry_run", False) or getattr(world, "_codex_pinned_actions_v13", False):
        return
    if all(
        callable(getattr(world, name, None))
        for name in (
            "gripper_uses_effort",
            "set_gripper_pin_effort",
            "gripper_pin_effort_list",
            "latch_gripper_close_keepalive",
            "release_gripper_close_keepalive",
            "enforce_gripper_close_keepalive",
        )
    ):
        if not callable(getattr(world, "_codex_raw_make_action_unpinned", None)):
            class_raw_make = getattr(type(world), "make_action_unpinned", None)
            if callable(class_raw_make):
                world._codex_raw_make_action_unpinned = class_raw_make.__get__(
                    world,
                    type(world),
                )
        world._codex_pinned_actions_v10 = True
        world._codex_pinned_actions_v11 = True
        world._codex_pinned_actions_v12 = True
        world._codex_pinned_actions_v13 = True
        return
    if getattr(world, "_codex_pinned_actions_v11", False):
        return
    import types
    from behavior_interface.robot_variant import robot_has_tool_roll

    if not hasattr(world, "_arm_pin_qpos"):
        world._arm_pin_qpos = {}
    if not hasattr(world, "_trunk_pin_qpos"):
        world._trunk_pin_qpos = None
    if not hasattr(world, "_gripper_pin_qpos"):
        world._gripper_pin_qpos = {}
    if not hasattr(world, "_tool_roll_pin_qpos"):
        world._tool_roll_pin_qpos = {}
    if not hasattr(world, "_tool_roll_motion_enabled"):
        world._tool_roll_motion_enabled = set()

    raw_make = getattr(world, "_codex_raw_make_action_unpinned", None)
    if not callable(raw_make):
        class_raw_make = getattr(type(world), "make_action_unpinned", None)
        if callable(class_raw_make):
            raw_make = class_raw_make.__get__(world, type(world))
        else:
            raw_make = getattr(world, "make_action_unpinned", None)
            if not callable(raw_make):
                raw_make = getattr(world, "make_action")
        world._codex_raw_make_action_unpinned = raw_make

    def _set_trunk_pin_qpos(self, qpos=None):
        arr = self.trunk_qpos() if qpos is None else np.asarray(qpos, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        self._trunk_pin_qpos = [float(x) for x in arr[:4]]

    def _trunk_pin_qpos_list(self):
        if self._trunk_pin_qpos is None:
            self.set_trunk_pin_qpos()
        return list(self._trunk_pin_qpos or self.trunk_qpos().tolist())

    def _arm_qpos_list(self, arm: str):
        names = list(self.robot.joints.keys())
        idx = [names.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
        qpos = self.robot.get_joint_positions()
        return [float(qpos[int(i)]) for i in idx]

    def _set_arm_pin_qpos(self, arm: str, qpos):
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self._arm_pin_qpos[str(arm).lower().strip()] = [float(x) for x in arr[:7]]

    def _arm_pin_qpos_list(self, arm: str):
        pin = self._arm_pin_qpos.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def _has_tool_roll(self, arm=None):
        return robot_has_tool_roll(getattr(self, "robot", None), arm)

    def _tool_roll_qpos(self, arm: str):
        arm = str(arm).lower().strip()
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        names = list(self.robot.joints.keys())
        idx = names.index(f"{arm}_arm_joint8")
        return float(self.robot.get_joint_positions()[idx])

    def _tool_roll_joint_limits(self, arm: str):
        arm = str(arm).lower().strip()
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        joint = self.robot.joints[f"{arm}_arm_joint8"]
        return float(joint.lower_limit), float(joint.upper_limit)

    def _set_tool_roll_pin_qpos(self, arm: str, qpos):
        arm = str(arm).lower().strip()
        value = float(qpos)
        if not np.isfinite(value):
            raise ValueError(f"tool roll pin qpos for {arm} must be finite")
        self._tool_roll_pin_qpos[arm] = value

    def _tool_roll_pin_qpos_value(self, arm: str):
        return float(self._tool_roll_pin_qpos.get(str(arm).lower().strip(), 0.0))

    def _begin_tool_roll_motion(self, arm: str):
        self._tool_roll_motion_enabled.add(str(arm).lower().strip())

    def _end_tool_roll_motion(self, arm: str):
        self._tool_roll_motion_enabled.discard(str(arm).lower().strip())

    def _force_set_tool_roll_qpos(self, arm: str, qpos):
        arm = str(arm).lower().strip()
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        target = float(qpos)
        if _challenge_action_only_enabled():
            raise RuntimeError(
                "direct J8 qpos injection is disabled in challenge mode; "
                "use tool_roll controller actions"
            )
        names = list(self.robot.joints.keys())
        idx = names.index(f"{arm}_arm_joint8")
        q0 = self.robot.get_joint_positions()
        q = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        q[int(idx)] = target
        self.robot.set_joint_positions(q)
        try:
            v0 = self.robot.get_joint_velocities()
            v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
            v[int(idx)] = 0.0
            self.robot.set_joint_velocities(v)
        except Exception:
            pass
        self.set_tool_roll_pin_qpos(arm, target)

    def _reset_tool_roll_to_zero(self, arm: str):
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if not self.has_tool_roll(arm):
            return {
                "arm": arm,
                "required": False,
                "reset": False,
                "reason": "7dof robot has no independent J8",
            }
        before = float(self.tool_roll_qpos(arm))
        pin_before = float(self.tool_roll_pin_qpos(arm))
        reset = abs(before) > 1e-7 or abs(pin_before) > 1e-7
        if _challenge_action_only_enabled():
            if reset:
                raise RuntimeError(
                    "synchronous J8 reset is disabled in challenge mode; "
                    "use _reset_selected_tool_roll_to_zero with yield from"
                )
            self.end_tool_roll_motion(arm)
            self.set_tool_roll_pin_qpos(arm, 0.0)
            return {
                "arm": arm,
                "reset": False,
                "before_rad": before,
                "pin_before_rad": pin_before,
                "after_rad": before,
                "pin_after_rad": 0.0,
                "locked": True,
                "action_only": True,
            }
        self.end_tool_roll_motion(arm)
        self.force_set_tool_roll_qpos(arm, 0.0)
        after = float(self.tool_roll_qpos(arm))
        pin_after = float(self.tool_roll_pin_qpos(arm))
        if abs(after) > 1e-7 or abs(pin_after) > 1e-7:
            raise RuntimeError(
                f"{arm} J8 reset failed: q={after:.8f}rad pin={pin_after:.8f}rad"
            )
        return {
            "arm": arm,
            "reset": bool(reset),
            "before_rad": before,
            "pin_before_rad": pin_before,
            "after_rad": after,
            "pin_after_rad": pin_after,
            "locked": True,
        }

    def _set_gripper_pin_qpos(self, arm: str, qpos):
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self._gripper_pin_qpos[str(arm).lower().strip()] = [float(x) for x in arr]

    def _gripper_pin_qpos_list(self, arm: str):
        pin = self._gripper_pin_qpos.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def _clear_gripper_pin_qpos(self, arm=None):
        if arm is None:
            self._gripper_pin_qpos.clear()
            return
        self._gripper_pin_qpos.pop(str(arm).lower().strip(), None)

    def _gripper_qpos_list(self, arm: str):
        try:
            names = list(self.robot.joints.keys())
            qpos = self.robot.get_joint_positions()
            joint_names = []
            try:
                joint_names.extend(list(getattr(self.robot, "finger_joint_names", {}).get(arm, [])))
            except Exception:
                pass
            joint_names.extend([
                f"{arm}_gripper_finger_joint1",
                f"{arm}_gripper_finger_joint2",
            ])
            seen = set()
            out = []
            for jn in joint_names:
                jn = str(jn)
                if jn in seen or jn not in names:
                    continue
                seen.add(jn)
                out.append(float(qpos[names.index(jn)]))
            if out:
                return out
            # Fallback only for robots whose controller indices also index qpos.
            idx = self.controller_action_idx(f"gripper_{arm}")
            return [float(qpos[int(i)]) for i in idx]
        except Exception:
            return None

    def _limb_pin_kwargs(self):
        out = {"trunk": self.trunk_pin_qpos_list()}
        for a in ("left", "right"):
            try:
                self.controller_action_idx(f"arm_{a}")
                out[f"arm_{a}"] = self.arm_pin_qpos_list(a) or self.arm_qpos_list(a)
                grip = self.gripper_pin_qpos_list(a) or self.gripper_qpos_list(a)
                if grip is not None:
                    out[f"gripper_{a}"] = grip
            except Exception:
                pass
            try:
                self.controller_action_idx(f"tool_roll_{a}")
                out[f"tool_roll_{a}"] = [self.tool_roll_pin_qpos(a)]
            except Exception:
                pass
        return out

    def _make_action_unpinned(self, **overrides):
        overrides = dict(overrides)
        for a in ("left", "right"):
            ctrl = f"tool_roll_{a}"
            if ctrl not in overrides:
                try:
                    self.controller_action_idx(ctrl)
                    overrides[ctrl] = [self.tool_roll_pin_qpos(a)]
                except Exception:
                    pass
        for ctrl, vec in overrides.items():
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            if ctrl.startswith("tool_roll_") and arr.size == 1:
                arm = ctrl.split("tool_roll_", 1)[1]
                requested = float(arr[0])
                locked = self.tool_roll_pin_qpos(arm)
                if (
                    arm not in self._tool_roll_motion_enabled
                    and abs(requested - locked) > 1e-7
                ):
                    raise PermissionError(
                        f"{ctrl} is locked; only adjust_eef_pose_in_wrist_frame roll may move J8"
                    )
        action = raw_make(**overrides)
        for ctrl, vec in overrides.items():
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            if ctrl in ("arm_left", "arm_right"):
                if arr.size >= 7:
                    self._arm_pin_qpos[ctrl.split("_", 1)[1]] = [
                        float(x) for x in arr[:7]
                    ]
            elif ctrl == "trunk" and arr.size >= 4:
                self._trunk_pin_qpos = [float(x) for x in arr[:4]]
            elif ctrl.startswith("gripper_") and arr.size > 0:
                self._gripper_pin_qpos[ctrl.split("_", 1)[1]] = [float(x) for x in arr]
            elif ctrl.startswith("tool_roll_") and arr.size == 1:
                self.set_tool_roll_pin_qpos(
                    ctrl.split("tool_roll_", 1)[1],
                    float(arr[0]),
                )
        return action

    def _pinned_action(self, **overrides):
        kw = self.limb_pin_kwargs()
        kw["base"] = [0.0, 0.0, 0.0]
        kw.update(overrides)
        return self.make_action_unpinned(**kw)

    def _hold_action_pinned(self):
        return self.pinned_action()

    def _empty_action(self):
        return self.hold_action_pinned()

    def _set_base_velocity(self, vx: float, vy: float, wz: float):
        return self.pinned_action(base=[float(vx), float(vy), float(wz)])

    def _make_action_trunk_locked(self, trunk_q):
        return self.pinned_action(
            trunk=np.asarray(trunk_q, dtype=np.float64).reshape(-1).tolist()
        )

    world.set_trunk_pin_qpos = types.MethodType(_set_trunk_pin_qpos, world)
    world.trunk_pin_qpos_list = types.MethodType(_trunk_pin_qpos_list, world)
    world.set_trunk_pin_qpos()
    world.arm_qpos_list = types.MethodType(_arm_qpos_list, world)
    world.set_arm_pin_qpos = types.MethodType(_set_arm_pin_qpos, world)
    world.arm_pin_qpos_list = types.MethodType(_arm_pin_qpos_list, world)
    world.has_tool_roll = types.MethodType(_has_tool_roll, world)
    world.tool_roll_qpos = types.MethodType(_tool_roll_qpos, world)
    world.tool_roll_joint_limits = types.MethodType(_tool_roll_joint_limits, world)
    world.set_tool_roll_pin_qpos = types.MethodType(_set_tool_roll_pin_qpos, world)
    world.tool_roll_pin_qpos = types.MethodType(_tool_roll_pin_qpos_value, world)
    world.begin_tool_roll_motion = types.MethodType(_begin_tool_roll_motion, world)
    world.end_tool_roll_motion = types.MethodType(_end_tool_roll_motion, world)
    world.force_set_tool_roll_qpos = types.MethodType(_force_set_tool_roll_qpos, world)
    world.reset_tool_roll_to_zero = types.MethodType(_reset_tool_roll_to_zero, world)
    for _arm in ("left", "right"):
        if _arm not in world._tool_roll_pin_qpos:
            try:
                world.set_tool_roll_pin_qpos(_arm, world.tool_roll_qpos(_arm))
            except Exception:
                pass
    world.gripper_qpos_list = types.MethodType(_gripper_qpos_list, world)
    world.set_gripper_pin_qpos = types.MethodType(_set_gripper_pin_qpos, world)
    world.gripper_pin_qpos_list = types.MethodType(_gripper_pin_qpos_list, world)
    world.clear_gripper_pin_qpos = types.MethodType(_clear_gripper_pin_qpos, world)
    world.limb_pin_kwargs = types.MethodType(_limb_pin_kwargs, world)
    world.make_action_unpinned = types.MethodType(_make_action_unpinned, world)
    world.pinned_action = types.MethodType(_pinned_action, world)
    world.make_action = world.pinned_action
    world.hold_action_pinned = types.MethodType(_hold_action_pinned, world)
    world.hold_action = world.hold_action_pinned
    world.empty_action = types.MethodType(_empty_action, world)
    world.set_base_velocity = types.MethodType(_set_base_velocity, world)
    world.make_action_trunk_locked = types.MethodType(_make_action_trunk_locked, world)
    for _arm in ("left", "right"):
        if _arm not in world._gripper_pin_qpos:
            try:
                vals = world.gripper_qpos_list(_arm)
                if vals is not None:
                    world.set_gripper_pin_qpos(_arm, vals)
            except Exception:
                pass
    world._codex_pinned_actions_v10 = True
    world._codex_pinned_actions_v11 = True
    world._codex_pinned_actions_v12 = True


def _prepare_legacy_7dof_motion(
    world,
    arm: str,
    *,
    ctx=None,
    stage_name: str = "",
) -> dict:
    """Compatibility report only; preserve the current locked J8 target."""
    _ensure_world_pinned_actions(world)
    if not world.has_tool_roll(arm):
        return {
            "arm": str(arm).lower().strip(),
            "required": False,
            "reset": False,
            "locked": True,
            "reason": "7dof robot has no independent J8",
        }
    del ctx, stage_name
    qpos = float(world.tool_roll_qpos(arm))
    pin = float(world.tool_roll_pin_qpos(arm))
    return {
        "arm": str(arm).lower().strip(),
        "required": False,
        "reset": False,
        "before_rad": qpos,
        "pin_before_rad": pin,
        "after_rad": qpos,
        "pin_after_rad": pin,
        "locked": bool(
            abs(qpos - pin) <= math.radians(0.02)
            and str(arm).lower().strip() not in world._tool_roll_motion_enabled
        ),
        "reason": "J8 stays at its current pin; the 7DOF payload is unchanged",
    }


def _reset_selected_tool_roll_to_zero(
    world,
    arm: str,
    *,
    ctx=None,
    stage_name: str,
) -> dict:
    """Reset the selected arm's J8 through controller actions only."""
    _ensure_world_pinned_actions(world)
    arm = str(arm).lower().strip()
    if arm not in ("left", "right"):
        raise ValueError(f"bad arm '{arm}'")
    if not world.has_tool_roll(arm):
        return {
            "arm": arm,
            "required": False,
            "reset": False,
            "ok": True,
            "action_only": True,
            "reason": "7dof robot has no independent J8",
        }

    before = float(world.tool_roll_qpos(arm))
    pin_before = float(world.tool_roll_pin_qpos(arm))
    reset = bool(
        abs(before) > _TOOL_ROLL_RESET_TOL_RAD
        or abs(pin_before) > _TOOL_ROLL_RESET_TOL_RAD
    )
    steps = 0
    world.begin_tool_roll_motion(arm)
    try:
        for step in range(_TOOL_ROLL_RESET_MAX_STEPS):
            current = float(world.tool_roll_qpos(arm))
            error = -current
            if abs(error) <= _TOOL_ROLL_RESET_TOL_RAD:
                break
            command = current + float(np.clip(
                error,
                -_TOOL_ROLL_RESET_MAX_DQ_RAD,
                +_TOOL_ROLL_RESET_MAX_DQ_RAD,
            ))
            steps = step + 1
            yield world.make_action(**{f"tool_roll_{arm}": [command]})

        for _ in range(_TOOL_ROLL_RESET_SETTLE_STEPS):
            yield world.make_action(**{f"tool_roll_{arm}": [0.0]})
            steps += 1
    finally:
        after = float(world.tool_roll_qpos(arm))
        ok = bool(abs(after) <= _TOOL_ROLL_RESET_TOL_RAD)
        world.set_tool_roll_pin_qpos(arm, 0.0 if ok else after)
        world.end_tool_roll_motion(arm)

    pin_after = float(world.tool_roll_pin_qpos(arm))
    report = {
        "arm": arm,
        "reset": reset,
        "ok": ok,
        "action_only": True,
        "before_rad": before,
        "pin_before_rad": pin_before,
        "after_rad": after,
        "pin_after_rad": pin_after,
        "steps": int(steps),
        "locked": True,
        "tolerance_rad": _TOOL_ROLL_RESET_TOL_RAD,
    }
    if ctx is not None and reset:
        ctx.log(
            f"[{stage_name}] J8 controller reset {arm}: "
            f"{math.degrees(before):+.3f}deg "
            f"(pin {math.degrees(pin_before):+.3f}deg) -> "
            f"{math.degrees(after):+.3f}deg "
            f"steps={steps} ok={ok}"
        )
    return report


def _assert_legacy_7dof_motion_ready(world, arm: str) -> None:
    """Compatibility no-op; normal J1-J7 execution must remain unchanged."""
    del world, arm


def _make_legacy_7dof_action(world, **overrides):
    """Submit the original 7DOF action unchanged."""
    return world.make_action(**overrides)


def _freeze_world_limb_pins(world) -> dict:
    """Freeze current trunk / arm / finger joints as the pinned action targets."""
    _ensure_world_pinned_actions(world)
    out = {"arms": {}, "grippers": {}, "gripper_efforts": {}}
    if getattr(world, "dry_run", False):
        return out
    try:
        world.set_trunk_pin_qpos()
    except Exception:
        pass
    for arm in ("left", "right"):
        try:
            arm_q = world.arm_qpos_list(arm)
            world.set_arm_pin_qpos(arm, arm_q)
            out["arms"][arm] = [round(float(x), 5) for x in arm_q]
        except Exception:
            pass
        try:
            grip_q = world.gripper_qpos_list(arm)
            if grip_q is not None:
                out["grippers"][arm] = [round(float(x), 5) for x in grip_q]
            carry = world.gripper_pin_effort_list(arm)
            if carry is not None and any(float(x) < -1e-6 for x in carry):
                out["gripper_efforts"][arm] = [round(float(x), 5) for x in carry]
            elif grip_q is not None:
                world.set_gripper_pin_qpos(arm, grip_q)
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper：把 candidate 改成统一 eef_target 字段
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_grasp_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    """把旧 grasp.py 的 candidate 包成新 get_eef_pose 的格式。

    旧 candidate：{pos, approach, label, reachable, ...}
    新 candidate：{target='grasp', eef_target={pos, approach, gripper_cmd}, next_eef_move, ...}
    """
    return {
        "id": c.get("id"),
        "target": "grasp",
        "label": c.get("label"),
        "arm": c.get("arm"),
        "eef_target": {
            "pos": list(c["pos"]),
            "approach": list(c["approach"]),
            "gripper_cmd": -1.0,  # contact 后关爪
        },
        "next_eef_move": [0.0, 0.0, 0.15],  # 默认提起 15cm
        "reachable": c.get("reachable", False),
        "reach_reason": c.get("reach_reason", ""),
        "score": c.get("score", 0.5),
        "meta": {"kind": "topdown_grasp"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# OPEN / CLOSE：动态采样门 hinge 几何
# ─────────────────────────────────────────────────────────────────────────────

def _list_openable_joints(obj):
    """返回 [(joint, dir, child_link_name), ...]，找不到返回空列表。"""
    from omnigibson.object_states.open_state import Open, _get_relevant_joints
    if obj is None or Open not in obj.states:
        return []
    st = obj.states[Open]
    info = st.relevant_joints_info or _get_relevant_joints(obj)
    _, joints, dirs = info
    out = []
    for j, d in zip(joints, dirs):
        # 从 child path 提取 link 短名
        child_raw = None
        for attr in ("body1", "child", "child_link", "_body1_name", "child_name"):
            v = getattr(j, attr, None)
            if v is None:
                continue
            child_raw = v if isinstance(v, str) else getattr(v, "name", str(v))
            break
        child_short = None
        if child_raw is not None:
            child_short = child_raw.split("/")[-1]
            if child_short not in obj.links:
                for ln in obj.links.keys():
                    if ln in child_raw:
                        child_short = ln
                        break
        out.append((j, int(d), child_short))
    return out


def _sample_door_geometry_legacy_unsafe(
    ctx,
    obj,
    joint,
    joint_dir: int,
    child_link_name: str,
):
    """generator 版：临时把 hinge set 到 closed/open 两端，记录门面 link 世界 AABB，
    反推 hinge 中心、把手位置、门外法向。每次 set_pos 后 yield 几帧让 PhysX sync。

    yield 出 world.empty_action() 让 server 真正 step sim。
    通过 StopIteration.value 返回 geom dict（None 表示失败）。

    返回 dict（None 表示失败）：
      {
        "lower": float, "upper": float,
        "closed_q": float, "open_q": float,
        "cur_q": float,
        "closed_aabb": (lo, hi),  # numpy
        "open_aabb":   (lo, hi),
        "hinge_world": np.array([x,y,z]),     # 估计 hinge 旋转轴上某点
        "axis_world":  np.array([0,0,1]),     # 估计 (默认 Z 轴)
        "handle_world": np.array([x,y,z]),    # 当前 q 下的把手位置
        "handle_open_world":  np.array([x,y,z]),  # open_q 时把手位置
        "handle_closed_world":np.array([x,y,z]),  # closed_q 时把手位置
        "outward_normal_world": np.array([nx,ny,nz]),  # 把手朝外侧法向
      }
    """
    if _challenge_action_only_enabled():
        raise RuntimeError(
            "legacy hinge state sampling is disabled in challenge mode"
        )
    if obj is None or joint is None or child_link_name is None:
        return None
    link = obj.links.get(child_link_name)
    if link is None:
        ctx.log(f"  _sample_door_geometry: link {child_link_name} not found")
        return None

    lower = float(joint.lower_limit)
    upper = float(joint.upper_limit)
    cur_q = float(joint.get_state()[0])
    closed_q = lower if joint_dir == 1 else upper
    open_q = upper if joint_dir == 1 else lower

    world = ctx.world

    def _settle(n):
        # 直接 advance 物理，不发 action（这里我们是生成器外部）
        try:
            world.env.step(world.empty_action())
        except Exception:
            pass

    def _set_and_read(q_tgt: float, n_settle: int = 20):
        try:
            # 直接强制设位置 + 清速度
            joint.set_pos(q_tgt)
            # 一些 OG joint 没有 set_vel，忽略
        except Exception as e:
            ctx.log(f"    set_pos err: {e}")
            return None
        # OG 的 step 接口在 generator 外不好用 → 这里我们假设调用方已经 yield 过 settle frames
        return None

    # 因为 _sample_door_geometry 不是 generator，无法 yield action 来 settle。
    # 我们直接用 link.aabb（瞬时几何，set_pos 后 PhysX 会立刻 sync transform，
    # 但 fabric/render 可能差 1 帧；对计算几何足够）。
    geom: Dict[str, Any] = {
        "lower": lower, "upper": upper,
        "closed_q": closed_q, "open_q": open_q, "cur_q": cur_q,
    }

    # —— 读 close AABB + 精确计算 handle_closed ——
    try:
        joint.set_pos(closed_q)
    except Exception as e:
        ctx.log(f"  set_pos(closed) err: {e}")
        return None
    # yield 6 帧让 PhysX 真正 sync articulation transform
    for _ in range(6):
        yield world.empty_action()
    try:
        ll, hh = link.aabb
        geom["closed_aabb"] = (_to_np(ll), _to_np(hh))
        ctx.log(f"    closed aabb: {geom['closed_aabb'][0].tolist()} → {geom['closed_aabb'][1].tolist()}")
    except Exception as e:
        ctx.log(f"  closed aabb err: {e}")
        return None
    # ← 在 closed_q 状态下读 link 世界位姿，用于计算精确的 handle_closed
    # （必须在 set_pos(closed_q) + settle 后立刻读，否则恢复到 cur_q 后位姿会变）
    _link_pos_closed = None
    _link_quat_closed = None
    try:
        _lp, _lq = link.get_position_orientation()
        _link_pos_closed = _to_np(_lp).reshape(3)
        _link_quat_closed = _to_np(_lq).reshape(4)
    except Exception as _ep:
        ctx.log(f"  closed link pose err: {_ep}")

    # —— 读 open AABB ——
    try:
        joint.set_pos(open_q)
    except Exception as e:
        ctx.log(f"  set_pos(open) err: {e}")
        return None
    for _ in range(6):
        yield world.empty_action()
    try:
        ll, hh = link.aabb
        geom["open_aabb"] = (_to_np(ll), _to_np(hh))
        ctx.log(f"    open   aabb: {geom['open_aabb'][0].tolist()} → {geom['open_aabb'][1].tolist()}")
    except Exception as e:
        ctx.log(f"  open aabb err: {e}")
        return None

    # —— 恢复到原始 q ——
    try:
        joint.set_pos(cur_q)
    except Exception as e:
        ctx.log(f"  restore q err: {e}")
    for _ in range(3):
        yield world.empty_action()
    try:
        ll, hh = link.aabb
        geom["cur_aabb"] = (_to_np(ll), _to_np(hh))
    except Exception:
        geom["cur_aabb"] = geom["closed_aabb"]

    # —— 几何推断（正确做法：用 link 世界位姿 + joint axis） ——
    #
    # 关键发现（诊断验证）：
    # 1. link_0（门 link）的世界原点 = 铰链轴上某点（link origin IS the hinge）
    # 2. link.aabb 返回整个物体 AABB（包含 frame），不是门面板 AABB → 不可用
    # 3. 把手位置 = link_0 世界原点 + R(link_quat) × [metadata 门远端局部坐标]
    # 4. axis_world = link_0 quat 旋转 joint.axis → 应直接从 link_0 的世界 quat 获得
    #
    # 方案：
    # hinge_world  = link.get_position_orientation()[0]  ← 直接读取
    # axis_world   = _quat_rot(link_quat, joint_axis_local_vec)
    # handle_world = hinge_world + _quat_rot(link_quat, handle_in_link_local)
    #   其中 handle_in_link_local 从 metadata AABB 中取"门把手侧"的中点
    #
    # Step 1：读 joint axis
    axis_local = "Z"
    try:
        a = getattr(joint, "axis", None) or getattr(joint, "joint_axis", None)
        if isinstance(a, str):
            axis_local = a.upper()
    except Exception:
        pass
    axis_local_vec = {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]}.get(axis_local, [0,0,1])

    # Step 2：读 link_0（门面板）世界位姿 → hinge 精确世界坐标 + axis_world
    hinge_from_joint = None
    try:
        link_pos_w, link_quat_w = link.get_position_orientation()
        link_pos_w  = _to_np(link_pos_w).reshape(3)
        link_quat_w = _to_np(link_quat_w).reshape(4)   # (x,y,z,w)
        # link 原点 = 铰链位置（URDF/USD 约定：link origin at joint）
        hinge_from_joint = link_pos_w.copy()
        # axis in world
        axis_world = _quat_rot(link_quat_w, np.array(axis_local_vec, dtype=np.float64))
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-9)
        ctx.log(f"    [link] world_pos={link_pos_w.round(4).tolist()} "
                f"world_quat={link_quat_w.round(4).tolist()}")
        ctx.log(f"    [joint] axis={axis_local} → world={axis_world.round(3).tolist()}")

        # 读 metadata AABB 得到把手在 link 局部帧中的位置
        # metadata 中 link_bounding_boxes[link_name].collision.axis_aligned.{extent, transform}
        handle_local = None
        try:
            meta = getattr(obj, "metadata", None) or {}
            lb = meta.get("link_bounding_boxes", {}).get(child_link_name, {})
            aabb_info = lb.get("collision", {}).get("axis_aligned", {})
            bb_extent = np.asarray(aabb_info.get("extent", [0,0,0]), dtype=np.float64)
            bb_transform = np.asarray(aabb_info.get("transform", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), dtype=np.float64)
            # AABB 中心在 link 局部帧
            bb_center_local = bb_transform[:3, 3]
            # bb_extent = [thick, length, width] in SOME local axis order
            # 找把手侧：最远离局部原点的方向（hinge at origin → handle at far end）
            # 沿 axis_local_vec 方向的半 extent = bb_extent 中与 axis_local_vec 对应的分量
            axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis_local, 2)
            # 门长方向（非厚度、非轴方向）= 3 个 extent 中最大的、且不是 axis 方向的那个
            bb_ex = bb_extent.copy()
            bb_ex[axis_idx] = 0  # 去掉 axis 方向
            length_idx = int(np.argmax(bb_ex))  # 最大 extent = 门的长度方向
            # 把手 = 在 length_idx 方向上 bb_center_local + bb_extent[length_idx]/2（远端）
            handle_local = bb_center_local.copy()
            handle_local[length_idx] += bb_extent[length_idx] / 2.0
            # axis 方向取 center（把手在门高度中点）
            ctx.log(f"    [meta] bb_center_local={bb_center_local.round(4).tolist()} "
                    f"bb_extent={bb_extent.round(4).tolist()} "
                    f"axis_idx={axis_idx} length_idx={length_idx} "
                    f"handle_local={handle_local.round(4).tolist()}")
        except Exception as _em:
            ctx.log(f"    [meta] AABB read err: {_em}")

        if handle_local is not None:
            # 优先用 closed 状态下读到的 link pose 计算 handle_closed（精确）
            # 若没有（first call），fallback 到当前状态
            if _link_pos_closed is not None and _link_quat_closed is not None:
                handle_from_link = _link_pos_closed + _quat_rot(_link_quat_closed, handle_local)
                ctx.log(f"    [handle_true] from_link_frame(closed)={handle_from_link.round(4).tolist()}")
            else:
                handle_from_link = link_pos_w + _quat_rot(link_quat_w, handle_local)
                ctx.log(f"    [handle_true] from_link_frame(cur)={handle_from_link.round(4).tolist()}")
        else:
            handle_from_link = None

        # 读 joint stiffness/damping（了解是否有弹簧阻力）
        try:
            kp = float(joint.stiffness)
            kd = float(joint.damping)
            driven = getattr(joint, "driven", False)
            ctx.log(f"    [joint_drive] stiffness={kp:.1f} damping={kd:.1f} driven={driven}")
        except Exception as _ejd:
            ctx.log(f"    [joint_drive] err: {_ejd}")

        # 读 local_position_0/1 和 local_orientation_0（正确属性名）
        try:
            lp0 = np.asarray(joint.local_position_0.cpu().numpy(), dtype=np.float64)
            ctx.log(f"    [joint] local_position_0={lp0.round(4).tolist()}")
        except Exception as _e0:
            ctx.log(f"    [joint] local_position_0 err: {_e0}")
        try:
            lp1 = np.asarray(joint.local_position_1.cpu().numpy(), dtype=np.float64)
            ctx.log(f"    [joint] local_position_1={lp1.round(4).tolist()}")
        except Exception as _e1:
            ctx.log(f"    [joint] local_position_1 err: {_e1}")

    except Exception as _e2:
        ctx.log(f"    [link_pose] err: {_e2}")
        axis_world = np.array(axis_local_vec, dtype=np.float64)
    except Exception:
        axis_world = np.array(axis_local_vec, dtype=np.float64)
    geom["axis_world"] = axis_world
    geom["axis_local"] = axis_local  # 'X'/'Y'/'Z'

    # 门面 center @ closed/open
    cmin, cmax = geom["closed_aabb"]
    omin, omax = geom["open_aabb"]
    c_center = (cmin + cmax) / 2.0
    o_center = (omin + omax) / 2.0
    geom["closed_center"] = c_center
    geom["open_center"] = o_center

    # hinge 位置：close + open 时门 center 都在同样的弧线半径 r 上，hinge 在两 center
    # 的弦的中垂线上、且与轴垂直平面内。简化：取 closed corner 中离 open AABB 最近的那个 corner。
    # close AABB 8 个 corner
    corners = np.array([
        [cmin[0], cmin[1], cmin[2]],
        [cmin[0], cmin[1], cmax[2]],
        [cmin[0], cmax[1], cmin[2]],
        [cmin[0], cmax[1], cmax[2]],
        [cmax[0], cmin[1], cmin[2]],
        [cmax[0], cmin[1], cmax[2]],
        [cmax[0], cmax[1], cmin[2]],
        [cmax[0], cmax[1], cmax[2]],
    ])
    # hinge corner = open AABB 距离最近的 corner（因为它转动时几乎不动）
    o_center_xy = o_center.copy()
    # 在与 axis 垂直平面里找 closest，先把 corner 沿 axis 投到平面
    def _project_to_plane(p, n, p0):
        d = float(np.dot(p - p0, n))
        return p - d * n
    p0 = c_center  # 平面经过 c_center
    proj_corners = np.array([_project_to_plane(c, axis_world, p0) for c in corners])
    proj_o = _project_to_plane(o_center, axis_world, p0)
    dists = np.linalg.norm(proj_corners - proj_o, axis=1)
    hinge_idx = int(np.argmin(dists))
    hinge_world = corners[hinge_idx].copy()
    # 若 joint frame 数据可用，用更精确的 hinge_from_joint 替代 AABB 角点估计
    if hinge_from_joint is not None:
        ctx.log(f"    [hinge] AABB-corner={hinge_world.round(4).tolist()} "
                f"joint-frame={hinge_from_joint.round(4).tolist()} "
                f"diff={np.linalg.norm(hinge_from_joint-hinge_world)*1000:.1f}mm")
        hinge_world = hinge_from_joint  # 使用精确值
    geom["hinge_world"] = hinge_world
    geom["hinge_corner_idx"] = hinge_idx

    # 把手位置：优先用 link 世界位姿 + metadata AABB（handle_from_link）
    # 后备：AABB 角点估计（不准确但能用）
    handle_idx = int(np.argmax(dists))
    handle_aabb_corner = corners[handle_idx].copy()
    ctx.log(f"    [handle] AABB-corner={handle_aabb_corner.round(4).tolist()}")

    if handle_from_link is not None:
        # 精确把手位置（基于 link 世界位姿 + metadata 局部 AABB）
        handle_closed = handle_from_link
        ctx.log(f"    [handle] using link-frame handle (precise)")
    else:
        # 后备：AABB 角点，但沿 axis 方向校正到门板中心高度
        # （角点在 axis 方向上是极端值，实际把手在门板高度中点）
        handle_closed = handle_aabb_corner.copy()
        corner_proj   = float(np.dot(handle_closed, axis_world))
        center_proj   = float(np.dot(c_center,      axis_world))
        handle_closed += (center_proj - corner_proj) * axis_world
        ctx.log(f"    [handle] AABB corner axis-corrected to center-height="
                f"{handle_closed.round(4).tolist()}")

    geom["handle_closed_world"] = handle_closed

    # 开门后的把手位置 = handle_closed 绕 (hinge_world, axis_world) 转 (open_q - closed_q) 弧度
    delta_q = open_q - closed_q  # 注意正负
    handle_open = _rotate_around_axis(handle_closed, hinge_world, axis_world, delta_q)
    geom["handle_open_world"] = handle_open

    # 当前位置（与 cur_q 对应）
    cur_delta = cur_q - closed_q
    geom["handle_world"] = _rotate_around_axis(handle_closed, hinge_world, axis_world, cur_delta)

    # 门外法向（朝向机器人/物体前方）
    # 正确做法：link 的局部 X 轴 = 门面厚度方向（外法线在 closed 时朝外）
    # link_quat_w 已在上面读取（Step 2 中）
    outward = None
    try:
        # 门面外法向 = link_0 局部 -X 轴在世界中的方向（负号因为 metadata 显示 link_0 朝向是 -x→outward）
        # 诊断：打印 link local_x 在世界的方向
        _door_x_world = _quat_rot(link_quat_w, np.array([1.0, 0.0, 0.0]))
        _door_neg_x_world = _quat_rot(link_quat_w, np.array([-1.0, 0.0, 0.0]))
        ctx.log(f"    [door] local+X_world={_door_x_world.round(3).tolist()} "
                f"local-X_world={_door_neg_x_world.round(3).tolist()}")
        # 取与「handle→obj_center」方向较近的那个
        obj_pos_w = _to_np(obj.get_position_orientation()[0]).reshape(3)
        handle_to_center = obj_pos_w - handle_closed
        handle_to_center[2] = 0.0
        hn = np.linalg.norm(handle_to_center)
        if hn > 1e-6:
            handle_to_center /= hn
        # outward = 朝远离物体中心（与 handle_to_center 反向）
        candidate_outward = -handle_to_center
        # 检查 link local_x vs local_-x 哪个更接近 candidate_outward
        dot_pos = float(np.dot(_door_x_world, candidate_outward))
        dot_neg = float(np.dot(_door_neg_x_world, candidate_outward))
        if dot_pos >= dot_neg:
            outward = _door_x_world / (np.linalg.norm(_door_x_world) + 1e-9)
        else:
            outward = _door_neg_x_world / (np.linalg.norm(_door_neg_x_world) + 1e-9)
        ctx.log(f"    [door] outward_normal={outward.round(3).tolist()} "
                f"(dot+x={dot_pos:.3f} dot-x={dot_neg:.3f})")
    except Exception as _eo:
        ctx.log(f"    [door] outward err: {_eo}")
    if outward is None:
        # 后备：handle → obj center 的反方向
        try:
            obj_center_world = _to_np(obj.get_position_orientation()[0])
        except Exception:
            obj_center_world = (cmin + cmax) / 2.0
        outward = handle_closed - obj_center_world
        outward[2] = 0.0
        if np.linalg.norm(outward) < 1e-6:
            outward = np.array([1.0, 0.0, 0.0])
        outward = outward / np.linalg.norm(outward)
    geom["outward_normal_world"] = outward

    # generator return（StopIteration.value）
    return geom


def _sample_door_geometry(ctx, obj, joint, joint_dir: int, child_link_name: str):
    """Infer hinge and handle geometry without changing the articulation state."""
    world = ctx.world
    if False:
        yield world.empty_action()
    if obj is None or joint is None or child_link_name is None:
        return None
    link = obj.links.get(child_link_name)
    if link is None:
        ctx.log(f"  _sample_door_geometry: link {child_link_name} not found")
        return None

    lower = float(joint.lower_limit)
    upper = float(joint.upper_limit)
    cur_q = float(joint.get_state()[0])
    closed_q = lower if joint_dir == 1 else upper
    open_q = upper if joint_dir == 1 else lower

    axis_name = "Z"
    try:
        raw_axis = getattr(joint, "axis", None) or getattr(
            joint,
            "joint_axis",
            None,
        )
        candidate = str(raw_axis).split(".")[-1].upper()
        if candidate in ("X", "Y", "Z"):
            axis_name = candidate
    except Exception:
        pass
    axis_local = {
        "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }[axis_name]

    link_pos_raw, link_quat_raw = link.get_position_orientation()
    link_pos = _to_np(link_pos_raw).reshape(3)
    link_quat = _to_np(link_quat_raw).reshape(4)
    hinge_world = link_pos.copy()
    joint_quat = link_quat
    try:
        joint_pos_raw, joint_quat_raw = joint.get_position_orientation()
        hinge_world = _to_np(joint_pos_raw).reshape(3)
        joint_quat = _to_np(joint_quat_raw).reshape(4)
    except Exception:
        pass
    axis_world = _quat_rot(joint_quat, axis_local)
    axis_world /= float(np.linalg.norm(axis_world)) + 1e-9

    handle_current = None
    handle_local = None
    try:
        meta = getattr(obj, "metadata", None) or {}
        link_box = meta.get("link_bounding_boxes", {}).get(
            child_link_name,
            {},
        )
        box = link_box.get("collision", {}).get("axis_aligned", {})
        extent = np.asarray(box.get("extent"), dtype=np.float64).reshape(3)
        transform = np.asarray(box.get("transform"), dtype=np.float64).reshape(4, 4)
        center = transform[:3, 3].copy()
        axis_index = {"X": 0, "Y": 1, "Z": 2}[axis_name]
        length_candidates = [
            index for index in range(3) if index != axis_index
        ]
        length_index = max(
            length_candidates,
            key=lambda index: float(extent[index]),
        )
        endpoints = []
        for sign in (-1.0, 1.0):
            point = center.copy()
            point[length_index] += sign * 0.5 * float(extent[length_index])
            radial = point - float(point @ axis_local) * axis_local
            endpoints.append((float(np.linalg.norm(radial)), point))
        handle_local = max(endpoints, key=lambda item: item[0])[1]
        handle_current = link_pos + _quat_rot(link_quat, handle_local)
    except Exception as exc:
        ctx.log(
            f"    [door-geom] metadata handle unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    if handle_current is None:
        try:
            lo_raw, hi_raw = link.aabb
            lo = _to_np(lo_raw).reshape(3)
            hi = _to_np(hi_raw).reshape(3)
            corners = np.array([
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ], dtype=np.float64)
            radial = corners - hinge_world[None, :]
            radial -= (
                radial @ axis_world
            )[:, None] * axis_world[None, :]
            handle_current = corners[
                int(np.argmax(np.linalg.norm(radial, axis=1)))
            ]
        except Exception as exc:
            ctx.log(
                f"    [door-geom] AABB handle fallback failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    handle_closed = _rotate_around_axis(
        handle_current,
        hinge_world,
        axis_world,
        closed_q - cur_q,
    )
    handle_open = _rotate_around_axis(
        handle_current,
        hinge_world,
        axis_world,
        open_q - cur_q,
    )

    try:
        obj_pos_raw, _ = obj.get_position_orientation()
        obj_pos = _to_np(obj_pos_raw).reshape(3)
        outward_current = handle_current - obj_pos
    except Exception:
        outward_current = handle_current - hinge_world
    outward_current -= float(outward_current @ axis_world) * axis_world
    if float(np.linalg.norm(outward_current)) < 1e-6:
        local_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(local_normal @ axis_local)) > 0.9:
            local_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        outward_current = _quat_rot(link_quat, local_normal)
        outward_current -= float(outward_current @ axis_world) * axis_world
    outward_current /= float(np.linalg.norm(outward_current)) + 1e-9
    outward_closed_tip = _rotate_around_axis(
        hinge_world + outward_current,
        hinge_world,
        axis_world,
        closed_q - cur_q,
    )
    outward_closed = outward_closed_tip - hinge_world
    outward_closed /= float(np.linalg.norm(outward_closed)) + 1e-9

    cur_aabb = None
    try:
        lo_raw, hi_raw = link.aabb
        cur_aabb = (
            _to_np(lo_raw).reshape(3),
            _to_np(hi_raw).reshape(3),
        )
    except Exception:
        pass

    ctx.log(
        f"    [door-geom/action-only] q={cur_q:+.3f} "
        f"closed={closed_q:+.3f} open={open_q:+.3f} "
        f"hinge={hinge_world.round(4).tolist()} "
        f"handle={handle_current.round(4).tolist()}"
    )
    return {
        "lower": lower,
        "upper": upper,
        "closed_q": closed_q,
        "open_q": open_q,
        "cur_q": cur_q,
        "hinge_world": hinge_world,
        "axis_world": axis_world,
        "axis_local": axis_name,
        "handle_world": handle_current,
        "handle_closed_world": handle_closed,
        "handle_open_world": handle_open,
        "outward_normal_world": outward_closed,
        "cur_aabb": cur_aabb,
        "handle_local": (
            None if handle_local is None else handle_local.copy()
        ),
        "geometry_source": "current_joint_link_metadata_no_state_mutation",
        "action_only": True,
    }


def _quat_rot(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    u = np.array([qx, qy, qz], dtype=np.float64)
    s = qw
    v = np.asarray(v, dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → 四元数 (x,y,z,w)。Shepperd 数值稳定方法。"""
    m00, m01, m02 = float(R[0,0]), float(R[0,1]), float(R[0,2])
    m10, m11, m12 = float(R[1,0]), float(R[1,1]), float(R[1,2])
    m20, m21, m22 = float(R[2,0]), float(R[2,1]), float(R[2,2])
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _rotate_around_axis(p, anchor, axis, angle):
    """Rodrigues 公式：把 p 绕 (anchor, axis) 旋转 angle 弧度。"""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    n = np.asarray(axis, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-9)
    v = p - a
    c, s = math.cos(angle), math.sin(angle)
    rot = v * c + np.cross(n, v) * s + n * float(np.dot(n, v)) * (1.0 - c)
    return a + rot


def _sample_open_candidates(ctx, obj, opening: bool, arm: str):
    """generator 版：每个 hinge 调 _sample_door_geometry（其本身也 yield 帧给 sim）。
    通过 StopIteration.value 返回 candidates list。
    """
    cands: List[Dict[str, Any]] = []
    j_list = _list_openable_joints(obj)
    if not j_list:
        return cands

    for ji, (j, jdir, child) in enumerate(j_list):
        geom = yield from _sample_door_geometry(ctx, obj, j, jdir, child)
        if geom is None:
            continue
        hinge = np.asarray(geom["hinge_world"], dtype=np.float64)
        axis = np.asarray(geom["axis_world"], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        outward = np.asarray(geom["outward_normal_world"], dtype=np.float64)
        handle_closed = np.asarray(geom["handle_closed_world"], dtype=np.float64)
        handle_open = np.asarray(geom["handle_open_world"], dtype=np.float64)
        handle_now = np.asarray(geom["handle_world"], dtype=np.float64)
        closed_q = float(geom["closed_q"])
        open_q = float(geom["open_q"])

        # ① 使用 _sample_door_geometry 中精确计算的 handle_closed_world
        # （由 link.get_position_orientation() + metadata AABB 推算，不再用整体 AABB 角点）
        # handle_mid = 把手世界坐标（门关闭时）
        handle_mid = handle_closed.copy()
        ctx.log(f"  j{ji} [precise] handle_mid from link-frame "
                f"=({handle_mid[0]:.4f},{handle_mid[1]:.4f},{handle_mid[2]:.4f})")

        # edge_extent = 把手沿铰链轴方向的尺寸（从 metadata link_0 AABB 推算）
        # axis = z_world，link_0 在 z 方向的 extent = metadata AABB 的第三维（对应 local z）
        # metadata: link_0 collision AABB extent=[0.041, 0.370, 0.235]，axis=Z → extent[2]=0.235
        try:
            _meta = getattr(obj, "metadata", None) or {}
            _lb = _meta.get("link_bounding_boxes", {}).get(child, {})
            _bb_ex = np.asarray(
                _lb.get("collision", {}).get("axis_aligned", {}).get("extent", [0.235, 0.235, 0.235]),
                dtype=np.float64)
            _axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(
                (geom.get("axis_local", "Z") or "Z").upper(), 2)
            edge_extent = float(_bb_ex[_axis_idx])
        except Exception:
            edge_extent = 0.20  # 默认 20cm

        # ② 计算切向 tangent (next_eef_move 方向，q=closed 时)
        hh = handle_mid - hinge
        hh_perp = hh - (hh @ axis) * axis
        if np.linalg.norm(hh_perp) < 1e-6:
            ctx.log(f"  hinge 与 handle 共线，跳过 j{ji}")
            continue
        tangent = np.cross(axis, hh_perp)
        tn = np.linalg.norm(tangent)
        if tn < 1e-6:
            continue
        tangent = tangent / tn
        # 用 handle_open - handle_closed 校准方向：tangent 朝开门方向
        h_delta = handle_open - handle_closed
        if np.dot(tangent, h_delta) < 0:
            tangent = -tangent
        if not opening:
            tangent = -tangent  # close 时反向
        ctx.log(f"  j{ji} child={child} edge_extent={edge_extent:.3f}m "
                f"handle_mid=({handle_mid[0]:.3f},{handle_mid[1]:.3f},{handle_mid[2]:.3f}) "
                f"tangent=({tangent[0]:+.2f},{tangent[1]:+.2f},{tangent[2]:+.2f})")

        # ③ 多个 6-DOF candidate：approach 从"机器人→把手"方向（在铰链垂直平面内）
        # 这样 pre_pos = handle - 0.25*approach 落在可达的自由空间中（微波炉外部）
        z_world = np.array([0.0, 0.0, 1.0])
        alphas = [1.0, 0.8, 0.6]
        spreads = [(+1, "S+"), (-1, "S-")]
        # 把手高度偏移采样范围：最大0.08m，避免大型门（冰箱/柜门）采样到离把手过远的位置
        h_off_max = min(0.08, max(0.02, edge_extent * 0.25))
        heights = [(0.0, "mid"), (+h_off_max, "hi"), (-h_off_max, "lo")]
        arc_len = float(np.linalg.norm(hh_perp)) * abs(open_q - closed_q)
        next_eef_move = (tangent * arc_len * 0.5).tolist()

        # 用肩关节世界坐标计算接近方向（比底盘坐标更精确）
        try:
            sh = ctx.world.shoulder_pose(arm=arm)
            approach_origin = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
        except Exception:
            try:
                approach_origin = np.array(ctx.world.robot.get_position(), dtype=np.float64)
            except Exception:
                approach_origin = np.zeros(3)
        handle_center = handle_mid.copy()
        # approach_base = 肩关节 → 把手（去除铰链轴分量后归一化）
        # 不做 z=0 截断：肩部已在正确高度，保留三维方向
        r2h = handle_center - approach_origin
        r2h -= np.dot(r2h, axis) * axis      # 去铰链轴分量（保持垂直于轴）
        r2h_norm = np.linalg.norm(r2h)
        if r2h_norm > 0.01:
            approach_base = r2h / r2h_norm   # 肩部指向把手的单位向量
        else:
            approach_base = -tangent  # fallback
        ctx.log(f"  j{ji} shoulder={approach_origin.round(3).tolist()} "
                f"approach_base=({approach_base[0]:+.2f},{approach_base[1]:+.2f},{approach_base[2]:+.2f})")

        ctx.log(f"  j{ji} tangent=({tangent[0]:+.2f},{tangent[1]:+.2f},{tangent[2]:+.2f}) "
                f"arc_len={arc_len:.3f}m")

        for alpha in alphas:
            # alpha=1.0：纯水平接近；alpha<1：混入朝下分量（方便抓手柄顶端）
            approach_raw = alpha * approach_base + (1.0 - alpha) * (-z_world)
            ap_norm = np.linalg.norm(approach_raw)
            if ap_norm < 1e-6:
                continue
            approach_vec = approach_raw / ap_norm
            z_axis_eef = approach_vec.copy()

            for spread_sign, lab_s in spreads:
                y_raw = spread_sign * axis
                y_axis_eef = y_raw - np.dot(y_raw, z_axis_eef) * z_axis_eef
                ny = np.linalg.norm(y_axis_eef)
                if ny < 1e-3:
                    continue
                y_axis_eef = y_axis_eef / ny
                x_axis_eef = np.cross(y_axis_eef, z_axis_eef)
                R_mat = np.column_stack([x_axis_eef, y_axis_eef, z_axis_eef])
                target_quat = _mat_to_quat_xyzw(R_mat)

                # 评分组件（越大越好）：
                # (a) force_score: approach 与 tangent 同向度
                #     approach ≈ tangent 时，pre/cnt 和 arc 运动方向一致，利于施力
                force_score = float(np.dot(approach_vec, tangent))  # ∈ [-1, 1]
                # (b) ik_ease: 占位
                ik_ease = 0.5
                # (c) spread_align: spread 与 axis 平行度
                spread_align = float(abs(np.dot(spread_sign * axis, y_axis_eef)))
                score_total = (2.0 * max(0.0, force_score)
                               + 1.0 * ik_ease
                               + 1.0 * spread_align)

                for axis_off, lab_h in heights:
                    handle_pt = handle_mid + axis_off * axis
                    contact_pos = handle_pt.copy()

                    cands.append({
                        "id": None,
                        "target": "open" if opening else "close",
                        "label": (f"{'open' if opening else 'close'}@j{ji}"
                                  f"-{lab_h}-{lab_s}-a{int(alpha*10):02d}"),
                        "arm": arm,
                        "eef_target": {
                            "pos": contact_pos.tolist(),
                            "approach": approach_vec.tolist(),
                            "quat": target_quat.tolist(),
                            "gripper_cmd": -1.0,
                        },
                        "next_eef_move": next_eef_move,
                        "score": float(score_total),
                        "meta": {
                            "kind": "door_arc",
                            "joint_name": getattr(j, "name", "?"),
                            "joint_dir": int(jdir),
                            "joint_range": [geom["lower"], geom["upper"]],
                            "closed_q": closed_q,
                            "open_q": open_q,
                            "cur_q": geom["cur_q"],
                            "hinge_world": hinge.tolist(),
                            "axis_world": axis.tolist(),
                            "handle_mid_world": handle_mid.tolist(),
                            "handle_open_world": handle_open.tolist(),
                            "handle_closed_world": handle_closed.tolist(),
                            "outward_normal_world": outward.tolist(),
                            "tangent_closed_world": tangent.tolist(),
                            "edge_extent": edge_extent,
                            "axis_off": axis_off,
                            "spread_sign": spread_sign,
                            "alpha": alpha,
                            "target_quat": target_quat.tolist(),
                            "force_score": force_score,
                            "spread_align": spread_align,
                            "child_link": child,
                        },
                    })
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# PUSH：根据 push_direction (世界向量) 生成 eef 接触点候选
# ─────────────────────────────────────────────────────────────────────────────

def _sample_push_candidates(obj, push_dir_world: List[float], push_dist: float,
                            arm: str) -> List[Dict[str, Any]]:
    """根据 push_direction (世界 [dx,dy,dz])，在物体 AABB 上算 eef 接触点。

    设计：R1Pro 默认 eef 朝下（finger 朝 -z），手腕不旋转的情况下水平推物体最现实的
    做法是从**顶部**贴住物体，然后整个 eef 沿 push_direction 水平平移：finger 在物体
    顶面与物体接触面产生摩擦，再加上 finger 形成的物理"挡板"在 push_direction 反方向
    一侧挡住物体（finger 在 eef +y 方向延伸），形成水平推。

    所以 approach 设成 (0,0,-1)（top-down 贴上）；接触点选在物体顶面，**沿 -push_direction
    略偏**（让 finger 落在 push_direction 反方向一侧，便于挡住物体往 push_direction 推）。
    next_eef_move = push_direction * push_dist (世界系)。

    采样 5 个候选：顶面中心 + 4 个略偏的位置。
    """
    aabb = _aabb_of(obj)
    if aabb is None:
        return []
    lo, hi = aabb
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    n = np.asarray(push_dir_world, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        return []
    n = n / norm
    n_xy = np.array([n[0], n[1], 0.0])
    n_xy_norm = float(np.linalg.norm(n_xy))
    if n_xy_norm > 1e-6:
        n_xy = n_xy / n_xy_norm

    # 顶面 z：物体 AABB 顶 - 1cm（让 finger 略压入物体顶部）
    top_z = float(hi[2]) - 0.01
    # 接触点 = 顶面中心沿 -push_direction (xy) 偏 30% half（让 finger 在 push 反方向一侧）
    off_x = -n_xy[0] * 0.30 * float(half[0])
    off_y = -n_xy[1] * 0.30 * float(half[1])
    center_pos = np.array([center[0] + off_x, center[1] + off_y, top_z])

    cands: List[Dict[str, Any]] = []
    eef_approach = [0.0, 0.0, -1.0]
    next_move = (n * push_dist).tolist()

    # 候选 1：顶面 push 反方向偏 30% 中心
    cands.append({
        "target": "push", "label": "push@top_center", "arm": arm,
        "eef_target": {
            "pos": center_pos.tolist(),
            "approach": eef_approach,
            "gripper_cmd": -1.0,  # 关爪让 finger 形成一个挡块
        },
        "next_eef_move": next_move,
        "score": 1.0,
        "meta": {"kind": "push_topdown", "push_direction": n.tolist(),
                 "push_dist": push_dist, "n_xy": n_xy.tolist()},
    })
    # 候选 2-5：垂直 push 方向的 ±perpendicular 偏移（覆盖物体宽度）+ ±push 方向的偏移
    # 垂直方向 = (n_y, -n_x, 0)（在 xy 平面里旋转 90°）
    if n_xy_norm > 1e-6:
        perp = np.array([-n_xy[1], n_xy[0], 0.0])  # 逆时针 90°
        perp_off = 0.30 * np.linalg.norm([half[0], half[1]])
        side_offsets = [
            (+perp_off * perp,         "+perp"),
            (-perp_off * perp,         "-perp"),
            (-0.50 * np.array([n[0]*half[0], n[1]*half[1], 0]), "back"),  # 更靠后
            ( 0.10 * np.array([n[0]*half[0], n[1]*half[1], 0]), "front"), # 略前
        ]
        for off, lab in side_offsets:
            pos = center_pos + off
            cands.append({
                "target": "push", "label": f"push@{lab}", "arm": arm,
                "eef_target": {
                    "pos": pos.tolist(),
                    "approach": eef_approach,
                    "gripper_cmd": -1.0,
                },
                "next_eef_move": next_move,
                "score": 0.8,
                "meta": {"kind": "push_topdown", "push_direction": n.tolist(),
                         "push_dist": push_dist, "n_xy": n_xy.tolist()},
            })
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# Skill: get_eef_pose
# ─────────────────────────────────────────────────────────────────────────────

@register_skill(
    "get_eef_pose",
    description=(
        "统一 eef 目标生成：根据 target (grasp/open/close/push) 算 3-5 个候选 eef pose + "
        "下一步 eef 移动向量。push 必须传 push_direction（世界系 3 维向量）。"
        "返回 candidates；execute_eef_pose(eef_id=...) 执行某个。"
    ),
)
def get_eef_pose(
    ctx,
    object_name: str,
    target: str = "grasp",
    push_direction: Optional[list] = None,
    push_dist: float = 0.30,
    arm: str = "right",
    k: int = 0,
):
    """yield 至少一次 action。"""
    world = ctx.world

    target = target.lower().strip()
    if target not in ("grasp", "open", "close", "push"):
        ctx.set_result({"ok": False, "error": f"target must be grasp/open/close/push, got '{target}'"})
        yield world.empty_action()
        return

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        ctx.set_result({"ok": False, "error": f"object not found: {object_name}"})
        yield world.empty_action()
        return

    aabb = _aabb_of(obj)
    if aabb is None and target != "open" and target != "close":
        ctx.set_result({"ok": False, "error": f"no AABB for {obj.name}"})
        yield world.empty_action()
        return
    lo, hi = aabb if aabb is not None else (np.zeros(3), np.zeros(3))
    obj_center = (lo + hi) / 2.0

    # 实时打印物体状态 + 与机器人距离
    try:
        rp = world.robot_pose()
        sh = world.shoulder_pose(arm=arm)
        d_xy = math.hypot(obj_center[0]-float(rp.pos[0]), obj_center[1]-float(rp.pos[1]))
        d_sh = math.sqrt((obj_center[0]-sh["x"])**2 + (obj_center[1]-sh["y"])**2 + (obj_center[2]-sh["z"])**2)
        ctx.log(
            f"get_eef_pose obj={obj.name} target={target} arm={arm}  "
            f"obj_center=({obj_center[0]:.2f},{obj_center[1]:.2f},{obj_center[2]:.2f})  "
            f"base→obj_xy={d_xy:.2f}m  shoulder→obj_3d={d_sh:.2f}m  (max reach={_ARM_MAX_REACH}m)"
        )
    except Exception:
        ctx.log(f"get_eef_pose obj={obj.name} target={target} arm={arm}")

    # ── 生成候选 ──
    raw: List[Dict[str, Any]] = []
    if target == "grasp":
        # 复用 grasp.py 的 top-down 采样
        gs = _sample_grasp_candidates(lo, hi, n_top=5, n_side=0)
        for g in gs:
            g["arm"] = arm
            raw.append(_wrap_grasp_candidate(g))

    elif target == "push":
        if push_direction is None or len(push_direction) != 3:
            ctx.set_result({"ok": False, "error": "push needs push_direction=[dx,dy,dz] (world)"})
            yield world.empty_action(); return
        raw = _sample_push_candidates(obj, list(push_direction), push_dist, arm)

    elif target in ("open", "close"):
        raw = yield from _sample_open_candidates(ctx, obj, opening=(target == "open"), arm=arm)
        if not raw:
            ctx.set_result({"ok": False, "error": f"{obj.name} 不可 {target}（无 Open state 或 hinge）"})
            yield world.empty_action(); return

    # ── 评估 reachable + IK 易解性评分 ──
    # 读 current eef quat 估算 ori 旋转难度
    try:
        eef_now = world.eef_pose(arm=arm)
        cur_quat = np.asarray(eef_now["quat"], dtype=np.float64)
    except Exception:
        cur_quat = None

    for c in raw:
        cgrasp = {"pos": c["eef_target"]["pos"], "approach": c["eef_target"]["approach"]}
        ok, why = _is_reachable(world, cgrasp, arm=arm)
        c["reachable"] = bool(ok)
        c["reach_reason"] = why
        # IK 易解性：与 current eef quat 的角度（弧度），越小越易解
        target_q = c["eef_target"].get("quat")
        if target_q is not None and cur_quat is not None:
            tq = np.asarray(target_q, dtype=np.float64)
            # |q1·q2| ∈ [0,1]，1=同 quat, 0=180°
            dot = abs(float(np.dot(tq, cur_quat)))
            dot = min(1.0, dot)
            ori_diff_rad = 2.0 * math.acos(dot)  # quat dot 到角度
            c["meta"] = c.get("meta", {})
            c["meta"]["ori_diff_to_current_rad"] = ori_diff_rad
            # 把 IK 易解性加入总分：与 current 旋转 < 60° 给满分，> 180° 给 0
            ik_ease = max(0.0, 1.0 - ori_diff_rad / math.pi)
            c["score"] = c.get("score", 0.5) + 1.5 * ik_ease
        if not ok:
            c["score"] = c.get("score", 0.5) * 0.3  # unreachable 大幅降权

    # 排序：reachable 优先，再按总分
    raw.sort(key=lambda x: (not x["reachable"], -x.get("score", 0)))
    n_reach = sum(1 for c in raw if c["reachable"])

    if k <= 0:
        if n_reach >= 5: k = 5
        elif n_reach >= 3: k = n_reach
        else: k = max(3, n_reach)
        k = max(3, min(5, k))

    # 对 open/close/push 模式：直接按总分 Top-K（已编码物理 + IK 易解评分）
    # 对 grasp：保持 FPS 多样性（候选差异小）
    if target in ("open", "close", "push"):
        reach_pool = [c for c in raw if c["reachable"]]
        chosen = reach_pool[:k]
        if len(chosen) < k:
            chosen.extend([c for c in raw if not c["reachable"]][:k - len(chosen)])
    else:
        reach_pool = [c for c in raw if c["reachable"]]
        if len(reach_pool) >= k:
            proxy = [{"pos": c["eef_target"]["pos"],
                      "approach": c["eef_target"]["approach"]} for c in reach_pool]
            idxs = _farthest_point_sampling(proxy, k)
            chosen = [reach_pool[i] for i in idxs]
        else:
            chosen = list(reach_pool)
            remain = k - len(chosen)
            unreach = [c for c in raw if not c["reachable"]]
            if remain > 0 and unreach:
                if len(unreach) <= remain:
                    chosen.extend(unreach)
                else:
                    proxy = [{"pos": c["eef_target"]["pos"],
                              "approach": c["eef_target"]["approach"]} for c in unreach]
                    idxs = _farthest_point_sampling(proxy, remain)
                    chosen.extend([unreach[i] for i in idxs])

    for i, c in enumerate(chosen):
        c["id"] = i

    ctx.log(f"get_eef_pose: 采样 {len(raw)} -> reachable {n_reach} -> 选 {len(chosen)}")
    for c in chosen:
        et = c["eef_target"]
        nm = c["next_eef_move"]
        ctx.log(
            f"  [{c['id']}] target={c['target']:5s} {c['label']:18s} "
            f"pos=({et['pos'][0]:+.2f},{et['pos'][1]:+.2f},{et['pos'][2]:+.2f}) "
            f"app=({et['approach'][0]:+.2f},{et['approach'][1]:+.2f},{et['approach'][2]:+.2f}) "
            f"grip={et['gripper_cmd']:+.0f}  next_move=({nm[0]:+.2f},{nm[1]:+.2f},{nm[2]:+.2f}) "
            f"reach={c['reachable']} ({c['reach_reason']})"
        )

    payload = {
        "ok": True,
        "object": {
            "input": object_name,
            "resolved_name": getattr(obj, "name", object_name),
            "aabb_min": lo.tolist(),
            "aabb_max": hi.tolist(),
            "center": obj_center.tolist(),
        },
        "target": target,
        "arm": arm,
        "n_candidates_sampled": len(raw),
        "n_reachable": n_reach,
        "candidates": chosen,
    }
    if n_reach == 0 and chosen:
        # 建议 base pose
        anchor = {
            "pos": chosen[0]["eef_target"]["pos"],
            "approach": chosen[0]["eef_target"]["approach"],
        }
        bp = _suggest_base_pose(world, anchor, arm=arm)
        if bp is not None:
            payload["suggested_base_pose"] = bp
            ctx.log(
                f"  → 不可达，建议 move_to(x={bp['x']:.2f}, y={bp['y']:.2f}, z={bp['z']:.2f}, "
                f"theta_x_deg={bp['theta_x_deg']:.1f}, theta_z_deg={bp['theta_z_deg']:.1f})"
            )
    ctx.set_result(payload)
    yield world.empty_action()


# ─────────────────────────────────────────────────────────────────────────────
# Skill: execute_eef_pose
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_MAX_STEPS  = 150
# tuck 由 tuck_trajectory 模块固定路点播放（hang→胸前）
_STAGE_PRE_TOL    = 0.04
_STAGE_TGT_TOL    = 0.025
_OPEN_FR          = 8
_GRIP_FR          = 30


def _err_json(x) -> float | None:
    """运动误差写入 API result：inf/nan → None（JSON 安全）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _err_failed(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(v) or math.isinf(v)


def _pose_within_tol(pos_err, ori_err, pos_tol: float, ori_tol_deg: float) -> bool:
    if _err_failed(pos_err) or _err_failed(ori_err):
        return False
    return (
        float(pos_err) <= float(pos_tol)
        and float(ori_err) <= float(ori_tol_deg)
    )


_ARC_PER_STEP_TOL = 0.05
# grasp_obj：先竖直抬升、再从上方悬停下降（transit 阶段开 cuRobo 避障）
_GRASP_LIFT_CLEAR_M = 0.15
_GRASP_HOVER_ABOVE_M = 0.10
_GRASP_DESCENT_SEG_M = 0.04
# 两段式接近：预抓取点沿 approach 反向后退距离（接触段长度）
_GRASP_PREGRASP_BACK_M = 0.07
# 只有真正已经贴近 safe pose 时才跳过 tuck；较远时先整理到稳定起始分支。
_GRASP_TUCK_SKIP_SAFE_DIST_M = 0.25
# 顶抓下探前夹爪朝向与目标 grasp_ori 的最大允许夹角；超过则中止，避免横扫物体
_GRASP_ORI_GATE_DEG = 40.0
_GRASP_ORI_HARD_ABORT_DEG = 85.0
_EEF_SAFE_POS_TOL_M = 0.030
_EEF_SAFE_ORI_TOL_DEG = 10.0
_EEF_SAFE_ENTRY_POS_TOL_M = 0.030
_EEF_SAFE_REFINE_MAX_POS_M = 0.090
_EEF_FINAL_POS_TOL_M = 0.010
_EEF_FINAL_ORI_TOL_DEG = 3.0
_EEF_WAYPOINT_POS_STEP_M = 0.055
_EEF_WAYPOINT_ORI_STEP_DEG = 20.0
_EEF_WAYPOINT_MID_POS_TOL_M = 0.020
_EEF_WAYPOINT_MID_ORI_TOL_DEG = 15.0
_EEF_WAYPOINT_MAX_LINE_DEV_M = 0.025
_EEF_BRIDGE_FK_LINE_POS_TOL_M = 0.060
_EEF_REACH_PREFLIGHT_MARGIN_M = 0.18
_EEF_SAFE_BACK_MIN_M = 0.000
_EEF_SAFE_BACK_MAX_M = 0.500

from behavior_interface.skills.tuck_trajectory import CHEST_TUCK_ARM as _CHEST_TUCK_ARM  # noqa: E402

def _eef_ori_err_deg(world, arm: str, target_quat) -> float:
    """当前 eef 朝向与 target_quat 的夹角（度）。用于下探前的安全闸。"""
    from behavior_interface.skills.grasp import _quat_to_mat, _orientation_error_omega

    try:
        eef = world.eef_pose(arm=arm)
        R_cur = _quat_to_mat(np.asarray(eef["quat"], dtype=np.float64))
        R_tgt = _quat_to_mat(np.asarray(target_quat, dtype=np.float64))
        omega = _orientation_error_omega(R_tgt, R_cur)
        return float(np.degrees(np.linalg.norm(omega)))
    except Exception:
        return 0.0


def _eef_approach_err_deg(world, arm: str, target_quat) -> float:
    """夹爪指向轴(+Z)与目标 +Z 的夹角；忽略 wrist roll。"""
    from behavior_interface.skills.grasp import _quat_to_mat

    try:
        eef = world.eef_pose(arm=arm)
        R_cur = _quat_to_mat(np.asarray(eef["quat"], dtype=np.float64))
        R_tgt = _quat_to_mat(np.asarray(target_quat, dtype=np.float64))
        a_cur = np.asarray(R_cur[:, 2], dtype=np.float64)
        a_tgt = np.asarray(R_tgt[:, 2], dtype=np.float64)
        a_cur = a_cur / (np.linalg.norm(a_cur) + 1e-9)
        a_tgt = a_tgt / (np.linalg.norm(a_tgt) + 1e-9)
        return float(np.degrees(math.acos(float(np.clip(np.dot(a_cur, a_tgt), -1.0, 1.0)))))
    except Exception:
        return 0.0


def _quat_normalize_xyzw(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_slerp_xyzw(q0, q1, t: float) -> np.ndarray:
    q0 = _quat_normalize_xyzw(q0)
    q1 = _quat_normalize_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    tt = float(np.clip(t, 0.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize_xyzw(q0 + tt * (q1 - q0))
    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)
    theta = theta0 * tt
    s0 = math.sin(theta0 - theta) / (sin0 + 1e-12)
    s1 = math.sin(theta) / (sin0 + 1e-12)
    return _quat_normalize_xyzw(s0 * q0 + s1 * q1)


def _quat_mul_xyzw(q1, q2) -> np.ndarray:
    x1, y1, z1, w1 = [float(v) for v in _quat_normalize_xyzw(q1)]
    x2, y2, z2, w2 = [float(v) for v in _quat_normalize_xyzw(q2)]
    return _quat_normalize_xyzw(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64))


def _axis_angle_quat_xyzw(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / n
    half = float(angle_rad) * 0.5
    s = math.sin(half)
    return _quat_normalize_xyzw(np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half),
    ], dtype=np.float64))


def _roll_quat_about_local_approach(q, roll_rad: float) -> np.ndarray:
    # Right-multiply by a local +Z rotation.  This preserves the world approach
    # axis R[:, 2], changing only the gripper roll / front-back side.
    return _quat_mul_xyzw(
        _quat_normalize_xyzw(q),
        _axis_angle_quat_xyzw([0.0, 0.0, 1.0], float(roll_rad)),
    )


def _camera_side_normal_world(q) -> Optional[np.ndarray]:
    try:
        from behavior_interface.skills.grasp import _quat_to_mat
        from behavior_interface.skills.gripper_camera_face import wrist_camera_normal_eef

        n = _quat_to_mat(_quat_normalize_xyzw(q)) @ wrist_camera_normal_eef()
        n = np.asarray(n, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 1e-9 else None
    except Exception:
        return None


def _arm_adapt_camera_face_score(world, q, reference_cam=None) -> tuple[bool, float, float]:
    """Camera-face gate for cross-arm EEF pose adaptation.

    A 180 degree roll around approach is IK-friendly but puts the wrist camera on
    the opposite side of the gripper.  Cross-arm adaptation may roll around
    approach, but the camera side must stay in the same horizontal half-space as
    the planned pose and, when available, toward robot/chest forward.
    """
    try:
        from behavior_interface.skills.gripper_camera_face import robot_forward_horizontal

        cam = _camera_side_normal_world(q)
        if cam is None:
            return True, 0.0, 0.0
        fwd = robot_forward_horizontal(world)
        cam_xy = cam.copy()
        cam_xy[2] = 0.0
        cam_xy_n = float(np.linalg.norm(cam_xy))
        cam_xy = cam_xy / cam_xy_n if cam_xy_n > 1e-9 else cam_xy
        dot_forward = float(np.dot(cam_xy, fwd)) if fwd is not None else 0.0
        dot_ref = (
            float(np.dot(cam, reference_cam))
            if reference_cam is not None
            else 1.0
        )
        if reference_cam is None:
            no_vertical_flip = True
        else:
            ref_z = float(reference_cam[2])
            cam_z = float(cam[2])
            # The user-visible failure mode is the wrist camera moving to the
            # opposite upper/lower side of the gripper.  A horizontal hand swap
            # may change XY direction, but it must preserve this vertical side.
            no_vertical_flip = (abs(ref_z) < 0.06 or abs(cam_z) < 0.06 or ref_z * cam_z > 0.0)
        return bool(no_vertical_flip), dot_forward, dot_ref
    except Exception:
        return True, 0.0, 0.0


def _roll_to_best_camera_forward(world, q) -> tuple[np.ndarray, float, float]:
    """Preserve approach and choose the roll with wrist camera most forward."""
    try:
        from behavior_interface.skills.gripper_camera_face import (
            camera_normal_world_horizontal,
            robot_forward_horizontal,
        )

        fwd = robot_forward_horizontal(world)
        if fwd is None:
            return _quat_normalize_xyzw(q), 0.0, 0.0
        best_q = _quat_normalize_xyzw(q)
        best_roll = 0.0
        best_dot = -2.0
        for deg in range(-90, 91, 5):
            roll = math.radians(float(deg))
            qq = _roll_quat_about_local_approach(q, roll)
            cam = camera_normal_world_horizontal(qq)
            if cam is None:
                continue
            dot = float(np.dot(cam, fwd))
            if dot > best_dot:
                best_q = qq
                best_roll = roll
                best_dot = dot
        return best_q, best_roll, best_dot
    except Exception:
        return _quat_normalize_xyzw(q), 0.0, 0.0


def _left_right_mirror_quat(world, q) -> tuple[np.ndarray, dict]:
    """Mirror a gripper pose horizontally for the opposite arm.

    This is not a roll-180 camera flip.  The target approach axis (EEF +Z) is
    preserved, and the gripper side axis is reflected across the vertical plane
    spanned by world-up and the approach vector.  The reflected frame is then
    completed as a proper right-handed rotation.
    """
    from behavior_interface.skills.grasp import _quat_to_mat

    q0 = _quat_normalize_xyzw(q)
    R0 = _quat_to_mat(q0)
    x0 = np.asarray(R0[:, 0], dtype=np.float64)
    y0 = np.asarray(R0[:, 1], dtype=np.float64)
    z = np.asarray(R0[:, 2], dtype=np.float64)
    z = z / (float(np.linalg.norm(z)) + 1e-12)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mirror_normal = np.cross(up, z)
    if float(np.linalg.norm(mirror_normal)) < 1e-6:
        try:
            fwd = np.asarray(world.chest_pose().get("forward", [1.0, 0.0, 0.0]), dtype=np.float64)
            fwd[2] = 0.0
            mirror_normal = np.cross(fwd, z)
        except Exception:
            mirror_normal = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), z)
    if float(np.linalg.norm(mirror_normal)) < 1e-6:
        mirror_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mirror_normal = mirror_normal / (float(np.linalg.norm(mirror_normal)) + 1e-12)

    # Reflect the local X side axis across the vertical approach plane, then
    # re-orthogonalize against the preserved approach axis.
    x_m = x0 - 2.0 * mirror_normal * float(np.dot(x0, mirror_normal))
    x_m = x_m - z * float(np.dot(x_m, z))
    if float(np.linalg.norm(x_m)) < 1e-6:
        y_ref = y0 - 2.0 * mirror_normal * float(np.dot(y0, mirror_normal))
        x_m = np.cross(y_ref, z)
    x_m = x_m / (float(np.linalg.norm(x_m)) + 1e-12)
    y_m = np.cross(z, x_m)
    y_m = y_m / (float(np.linalg.norm(y_m)) + 1e-12)
    x_m = np.cross(y_m, z)
    x_m = x_m / (float(np.linalg.norm(x_m)) + 1e-12)
    Rm = np.column_stack([x_m, y_m, z])
    if float(np.linalg.det(Rm)) < 0.0:
        y_m = -y_m
        Rm = np.column_stack([x_m, y_m, z])
    qm = _mat_to_quat_xyzw(Rm)
    meta = {
        "mirror_normal_world": mirror_normal.tolist(),
        "approach_world": z.tolist(),
        "x_before": x0.tolist(),
        "y_before": y0.tolist(),
        "x_after": x_m.tolist(),
        "y_after": y_m.tolist(),
        "det_after": float(np.linalg.det(Rm)),
    }
    return _quat_normalize_xyzw(qm), meta


def _eef_pose_err(world, arm: str, target_pos, target_quat) -> tuple[float, float, float]:
    eef = world.eef_pose(arm=arm)
    pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    pos_err = float(np.linalg.norm(pos - np.asarray(target_pos, dtype=np.float64).reshape(3)))
    ori_err = _eef_ori_err_deg(world, arm, target_quat)
    app_err = _eef_approach_err_deg(world, arm, target_quat)
    return pos_err, float(ori_err), float(app_err)


def _eef_6d_ok(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    pos_tol: float,
    ori_tol_deg: float,
) -> bool:
    pos_err, ori_err, _app_err = _eef_pose_err(world, arm, target_pos, target_quat)
    return bool(pos_err <= float(pos_tol) and ori_err <= float(ori_tol_deg))


def _adapt_eef_quat_for_exec_arm(
    world,
    *,
    plan_arm: str,
    exec_arm: str,
    target_pos,
    target_quat,
    back_m: float,
    ctx=None,
) -> tuple[np.ndarray, dict, bool]:
    """Pick an execution-arm quaternion for a plan-arm 6D pose.

    A saved grasp pose is a physical gripper pose.  When switching arms, use a
    horizontal left/right mirror: preserve the approach axis, keep the wrist
    camera on the same upper/lower side, and swap the gripper side axis across
    the vertical plane through the approach.  A roll-180 candidate is kept only
    as a rejected diagnostic reference.
    """
    p_arm = str(plan_arm or "").lower().strip()
    e_arm = str(exec_arm or "").lower().strip()
    q0 = _quat_normalize_xyzw(target_quat)
    if p_arm not in ("left", "right") or e_arm not in ("left", "right") or p_arm == e_arm:
        return q0, {"applied": False, "reason": "same_or_unknown_arm", "plan_arm": p_arm, "exec_arm": e_arm}, False

    from behavior_interface.skills.grasp import _quat_to_mat

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    ref_cam = _camera_side_normal_world(q0)
    q_mirror, mirror_meta = _left_right_mirror_quat(world, q0)
    q_roll180 = _roll_quat_about_local_approach(q0, math.pi)
    candidates: list[tuple[str, np.ndarray, bool, dict]] = [
        ("left_right_mirror", q_mirror, True, mirror_meta),
        ("original_no_adapt_reference", q0, False, {}),
        ("roll180_wrong_reference", q_roll180, False, {}),
    ]
    probes: list[dict] = []
    for label, q, selectable, extra_meta in candidates:
        q = _quat_normalize_xyzw(q)
        camera_ok, camera_dot_forward, camera_dot_ref = _arm_adapt_camera_face_score(
            world, q, ref_cam
        )
        pointing = _quat_to_mat(q)[:, 2]
        npt = float(np.linalg.norm(pointing))
        pointing = pointing / npt if npt > 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        safe_pos = target_pos - float(back_m) * pointing
        q_safe, safe_pos_err, safe_ori_err = _eef_solve_6d_dls_arm_q(
            world,
            e_arm,
            safe_pos,
            q,
            pos_tol=_EEF_SAFE_POS_TOL_M,
            ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
            max_steps=240,
            max_dq_per_step=0.075,
            max_dx_per_step=0.022,
            max_dw_per_step=0.12,
            ori_weight=0.55,
            lam=0.08,
        )
        q_final, final_pos_err, final_ori_err = _eef_solve_6d_dls_arm_q(
            world,
            e_arm,
            target_pos,
            q,
            pos_tol=_EEF_FINAL_POS_TOL_M,
            ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
            max_steps=320,
            max_dq_per_step=0.070,
            max_dx_per_step=0.018,
            max_dw_per_step=0.09,
            ori_weight=0.70,
            lam=0.08,
            seed_q=q_safe,
        )
        safe_ok = (
            q_safe is not None
            and float(safe_pos_err) <= _EEF_SAFE_POS_TOL_M
            and float(safe_ori_err) <= _EEF_SAFE_ORI_TOL_DEG
        )
        final_ok = (
            q_final is not None
            and float(final_pos_err) <= _EEF_FINAL_POS_TOL_M
            and float(final_ori_err) <= _EEF_FINAL_ORI_TOL_DEG
        )
        ok = bool(camera_ok and safe_ok and final_ok)
        # Only the explicit left/right mirror is selectable.  References are
        # evaluated for the report but cannot silently replace the mirror.
        select_penalty = 0.0 if selectable else 100000.0
        face_penalty = 0.0 if camera_ok else 10000.0
        score = (
            select_penalty
            + face_penalty
            + (0.0 if ok else 1000.0)
            + 2.0 * float(final_pos_err)
            + 0.050 * math.radians(float(final_ori_err))
            + 1.0 * float(safe_pos_err)
            + 0.020 * math.radians(float(safe_ori_err))
        )
        probes.append({
            "label": label,
            "transform": "left_right_mirror" if selectable else "reference_only",
            "quat": [round(float(x), 6) for x in q.tolist()],
            "safe_pos_err_m": float(safe_pos_err),
            "safe_ori_err_deg": float(safe_ori_err),
            "safe_ok": bool(safe_ok),
            "final_pos_err_m": float(final_pos_err),
            "final_ori_err_deg": float(final_ori_err),
            "final_ok": bool(final_ok),
            "camera_face_ok": bool(camera_ok),
            "camera_dot_forward": float(camera_dot_forward),
            "camera_dot_plan": float(camera_dot_ref),
            "selectable": bool(selectable),
            "mirror_meta": extra_meta,
            "ok": ok,
            "score": float(score),
            "q": q,
        })

    if not probes:
        return q0, {
            "applied": False,
            "reason": "no_quat_candidates",
            "plan_arm": p_arm,
            "exec_arm": e_arm,
        }, False

    probes.sort(key=lambda x: float(x["score"]))
    best = probes[0]
    if ctx is not None:
        for pr in probes:
            ctx.log(
                "   [arm-adapt] "
                f"{p_arm}→{e_arm} {pr['label']} selectable={pr['selectable']} "
                f"camera_ok={pr['camera_face_ok']} "
                f"cam_dot_fwd={pr['camera_dot_forward']:+.2f} "
                f"cam_dot_plan={pr['camera_dot_plan']:+.2f} "
                f"safe=({pr['safe_pos_err_m']*1000:.1f}mm,{pr['safe_ori_err_deg']:.1f}°,"
                f"ok={pr['safe_ok']}) "
                f"final=({pr['final_pos_err_m']*1000:.1f}mm,{pr['final_ori_err_deg']:.1f}°,"
                f"ok={pr['final_ok']})"
            )
        ctx.log(
            "   [arm-adapt] selected "
            f"{best['label']} quat=({best['q'][0]:+.3f},{best['q'][1]:+.3f},"
            f"{best['q'][2]:+.3f},{best['q'][3]:+.3f}) "
            f"ok={best['ok']}"
        )
    meta = {
        "applied": True,
        "plan_arm": p_arm,
        "exec_arm": e_arm,
        "selected": {k: v for k, v in best.items() if k != "q"},
        "candidates": [
            {k: v for k, v in pr.items() if k != "q"}
            for pr in probes
        ],
    }
    # The selection was made with DLS/OG FK.  Force the direct exec path to use
    # the same solver family so a cuRobo near-orientation miss does not block
    # the already-validated arm-adapted candidate.
    return np.asarray(best["q"], dtype=np.float64).reshape(4), meta, True


def _eef_target_reach_preflight(
    world,
    arm: str,
    target_pos,
    *,
    label: str,
    ctx=None,
    limit_m: Optional[float] = None,
) -> tuple[bool, float, float]:
    """Fast reject targets obviously outside the current arm workspace."""
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    limit = float(_ARM_MAX_REACH + _EEF_REACH_PREFLIGHT_MARGIN_M if limit_m is None else limit_m)
    try:
        sh = world.shoulder_pose(arm=arm)
        shoulder = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
        d_sh = float(np.linalg.norm(target - shoulder))
    except Exception as e:
        if ctx is not None:
            ctx.log(f"  WARN reach preflight {label} 查询 shoulder 失败: {e}")
        return True, float("nan"), limit
    ok = d_sh <= limit
    if ctx is not None:
        status = "OK" if ok else "FAIL"
        ctx.log(
            f"  [eef] reach preflight {label}: "
            f"shoulder→target={d_sh:.3f}m limit={limit:.3f}m {status}"
        )
    return ok, d_sh, limit


def _eef_goto_6d_waypoints(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_total_steps: int = 240,
    waypoint_pos_step_m: float = _EEF_WAYPOINT_POS_STEP_M,
    waypoint_ori_step_deg: float = _EEF_WAYPOINT_ORI_STEP_DEG,
    max_dq_per_step: float = 0.045,
    max_dx_per_step: float = 0.018,
    max_dw_per_step: float = 0.10,
    ori_weight: float = 0.55,
    lam: float = 0.10,
    allow_contact_object=None,
) -> tuple[float, float]:
    """Follow an approximately straight Cartesian 6DoF path and hard-check pose."""
    from behavior_interface.skills.grasp import _eef_goto_world

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    eef0 = world.eef_pose(arm=arm)
    start_pos = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    start_quat = _quat_normalize_xyzw(eef0["quat"])
    dist = float(np.linalg.norm(target_pos - start_pos))
    # Quaternion angular distance in degrees.
    dot = abs(float(np.dot(start_quat, target_quat)))
    dot = float(np.clip(dot, -1.0, 1.0))
    ang_deg = float(math.degrees(2.0 * math.acos(dot)))
    n_pos = int(math.ceil(dist / max(1e-6, float(waypoint_pos_step_m))))
    n_ori = int(math.ceil(ang_deg / max(1e-6, float(waypoint_ori_step_deg))))
    n_wp = max(1, min(8, max(n_pos, n_ori)))
    steps_per_wp = max(10, int(math.ceil(int(max_total_steps) / max(1, n_wp))))

    last_pos_err = float("inf")
    last_ori_err = float("inf")
    ctx.log(
        f"    [{stage_name}] 6d path waypoints={n_wp} "
        f"dist={dist*100:.1f}cm rot={ang_deg:.1f}°"
    )
    for wi in range(1, n_wp + 1):
        t = wi / n_wp
        wp_pos = start_pos + (target_pos - start_pos) * t
        wp_quat = _quat_slerp_xyzw(start_quat, target_quat, t)
        err = yield from _eef_goto_world(
            world, arm, wp_pos,
            target_world_quat=wp_quat,
            max_steps=steps_per_wp,
            pos_tol=max(float(pos_tol), _EEF_WAYPOINT_MID_POS_TOL_M if wi < n_wp else float(pos_tol)),
            ori_tol=math.radians(max(float(ori_tol_deg), _EEF_WAYPOINT_MID_ORI_TOL_DEG if wi < n_wp else float(ori_tol_deg))),
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name=f"{stage_name}[{wi}/{n_wp}]",
            max_dq_per_step=max_dq_per_step,
            max_dx_per_step=max_dx_per_step,
            max_dw_per_step=max_dw_per_step,
            ori_weight=float(ori_weight),
            lam=lam,
            adaptive_ori=False,
            disable_stuck_check=True,
            allow_contact_object=allow_contact_object,
        )
        if _err_failed(err):
            break
        last_pos_err, last_ori_err, app_err = _eef_pose_err(world, arm, wp_pos, wp_quat)
        ctx.log(
            f"    [{stage_name}] wp {wi}/{n_wp} "
            f"pos={last_pos_err*1000:.0f}mm ori={last_ori_err:.1f}° approach={app_err:.1f}°"
        )
        wp_pos_tol = float(pos_tol) if wi == n_wp else max(
            float(pos_tol), _EEF_WAYPOINT_MID_POS_TOL_M
        )
        wp_ori_tol = float(ori_tol_deg) if wi == n_wp else max(
            float(ori_tol_deg), _EEF_WAYPOINT_MID_ORI_TOL_DEG
        )
        if last_pos_err > wp_pos_tol or last_ori_err > wp_ori_tol:
            ctx.log(
                f"    [{stage_name}] ABORT waypoint {wi}/{n_wp} 未跟上 6DoF："
                f"pos={last_pos_err*1000:.0f}mm>{wp_pos_tol*1000:.0f}mm "
                f"ori={last_ori_err:.1f}°>{wp_ori_tol:.1f}°"
            )
            break

    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(world, arm, target_pos, target_quat)
    if final_pos_err > float(pos_tol) and final_ori_err <= max(float(ori_tol_deg) * 1.5, 20.0):
        refine_err = yield from _eef_goto_world(
            world, arm, target_pos,
            target_world_quat=None,
            hold_world_quat=target_quat,
            max_steps=90,
            pos_tol=float(pos_tol),
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name=f"{stage_name}:pos_refine_hold_quat",
            max_dq_per_step=max_dq_per_step,
            max_dx_per_step=max_dx_per_step,
            max_dw_per_step=max_dw_per_step,
            lam=lam,
            disable_stuck_check=True,
            allow_contact_object=allow_contact_object,
        )
        if not _err_failed(refine_err):
            final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
                world, arm, target_pos, target_quat
            )
    ctx.log(
        f"    [{stage_name}] 6d final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}°"
    )
    return float(final_pos_err), float(final_ori_err)


def _safe_back_m_candidates(
    requested_back_m: float,
    *,
    has_final_q: bool = False,
    fallback_back_m: Optional[float] = None,
) -> list[float]:
    req = float(np.clip(float(requested_back_m), _EEF_SAFE_BACK_MIN_M, _EEF_SAFE_BACK_MAX_M))
    vals = [req]
    fallback = None
    if fallback_back_m is not None:
        fallback = float(fallback_back_m)
        if np.isfinite(fallback):
            fallback = float(np.clip(
                fallback,
                _EEF_SAFE_BACK_MIN_M,
                _EEF_SAFE_BACK_MAX_M,
            ))
        else:
            fallback = None
    if bool(has_final_q):
        # Filter-grasp execution uses the stored final branch and probes a
        # fixed safe retreat band. Prefer the exact requested retreat first,
        # then the nearest centimeter values. A stored planner retreat is only
        # a fallback candidate; it must not jump ahead of closer distances.
        cm0 = 5
        cm1 = int(math.floor(_EEF_SAFE_BACK_MAX_M * 100.0 + 1e-9))
        req_cm = int(round(req * 100.0))
        vals.extend([
            cm / 100.0
            for cm in sorted(
                range(cm0, cm1 + 1),
                key=lambda cm: (abs(cm - req_cm), cm > req_cm, abs(cm)),
            )
        ])
    else:
        vals.extend([
            req * 0.75,
            req * 1.25,
            req * 0.55,
            req * 1.55,
            0.06,
            0.05,
            0.045,
            0.04,
            0.035,
            0.03,
            0.025,
            0.02,
            0.015,
            0.01,
            0.0,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.22,
        ])
    if fallback is not None:
        vals.append(fallback)
    out: list[float] = []
    for v in vals:
        v = float(np.clip(float(v), _EEF_SAFE_BACK_MIN_M, _EEF_SAFE_BACK_MAX_M))
        if all(abs(v - old) > 1e-4 for old in out):
            out.append(v)
    return out


def _safe_plan_failure_message(stage_name: str, safe_plan_meta: dict) -> str:
    meta = safe_plan_meta if isinstance(safe_plan_meta, dict) else {}
    back = meta.get("back_m")
    back_s = "unknown"
    try:
        back_s = f"{float(back) * 100:.0f}cm"
    except (TypeError, ValueError):
        pass
    reason = str(meta.get("error") or meta.get("reason") or "unknown")
    path_meta = meta.get("to_safe_path") if isinstance(meta.get("to_safe_path"), dict) else {}
    path_error = str(path_meta.get("error") or "")
    accepted = path_meta.get("accepted")
    dropped = path_meta.get("dropped")
    if reason == "safe_endpoint_6d_ik_failed":
        detail = f"safe endpoint 6D IK 未解出（back={back_s}）"
    elif reason == "safe_final_branch_gap":
        detail = (
            f"safe endpoint 不在 final 同分支（back={back_s}, "
            f"gap={meta.get('safe_to_final_joint_gap_rad')}rad）"
        )
    elif path_error or reason in {"to_safe_xyz_waypoint_ik_failed", "waypoint_branch_gap"}:
        detail = (
            f"current→safe 离线 XYZ waypoint 同分支路径未解出"
            f"（back={back_s}, accepted={accepted}, error={path_error or reason}）"
        )
        if dropped:
            detail += f" dropped={dropped}"
    else:
        detail = f"safe 离线规划失败（back={back_s}, reason={reason}）"
    return f"{stage_name}: {detail}"


def _eef_plan_to_safe_xyz_anchor_path(
    world,
    arm: str,
    safe_pos,
    target_quat,
    *,
    safe_q: np.ndarray,
    final_q: Optional[np.ndarray],
    ctx,
    stage_name: str,
    max_joint_gap_rad: float,
    require_endpoint: bool = False,
) -> tuple[list[dict], dict]:
    """Plan current->safe using XYZ-only intermediate waypoints on one branch.

    A nonzero-back safe pose is a transit waypoint, not the task endpoint.  If
    that sampled endpoint cannot stay on the current branch, keep the preceding
    anchors and let the following safe->final segment bridge from the last valid
    anchor.  A zero-back safe pose is the true final target and remains required.
    """
    safe_pos = np.asarray(safe_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    safe_q = np.asarray(safe_q, dtype=np.float64).reshape(7)
    final_q_np = (
        np.asarray(final_q, dtype=np.float64).reshape(7)
        if final_q is not None else None
    )
    anchors, meta = _eef_plan_6d_joint_anchors(
        world,
        arm,
        safe_pos,
        target_quat,
        ctx=ctx,
        stage_name=stage_name,
        pos_tol=_EEF_SAFE_POS_TOL_M,
        ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
        waypoint_pos_step_m=0.025,
        max_joint_gap_rad=float(max_joint_gap_rad),
        max_anchor_drop_ratio=1.0,
        mid_pos_tol=_EEF_SAFE_POS_TOL_M,
        intermediate_pos_only=True,
        final_q=safe_q,
        final_seed_q=None,
        require_final_waypoint=bool(require_endpoint),
        retain_validated_final_q_across_gap=True,
    )
    meta = dict(meta)
    meta["ok"] = bool(anchors)
    meta["anchor_count"] = int(len(anchors))
    meta["endpoint_required"] = bool(require_endpoint)
    meta["endpoint_reached"] = bool(
        anchors and anchors[-1].get("is_final")
    )
    if final_q_np is not None:
        meta["safe_to_final_joint_gap_rad"] = float(np.linalg.norm(
            safe_q - final_q_np,
            ord=np.inf,
        ))
    if anchors:
        meta["max_anchor_joint_gap_rad"] = float(max(
            float(a.get("joint_gap", 0.0)) for a in anchors
        ))
        meta["final_anchor_pos_err_m"] = float(anchors[-1].get("pos_err", float("inf")))
        meta["final_anchor_ori_err_deg"] = float(anchors[-1].get("ori_err", float("inf")))
    return anchors, meta


def _adapt_safe_pose_back_m(
    world,
    arm: str,
    target_pos,
    target_quat,
    pointing,
    requested_back_m: float,
    *,
    ctx,
    stage_name: str,
    final_q: Optional[np.ndarray] = None,
    final_branch_gap_rad: float = 0.85,
    fallback_back_m: Optional[float] = None,
) -> tuple[np.ndarray, float, dict]:
    """Pick a safe-pose retreat distance whose endpoint and line are executable."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    pointing = np.asarray(pointing, dtype=np.float64).reshape(3)
    pointing = pointing / (float(np.linalg.norm(pointing)) + 1e-9)
    best_meta: dict = {"ok": False, "requested_back_m": float(requested_back_m)}
    best_score = float("inf")
    best_pos = target_pos - float(requested_back_m) * pointing
    best_back = float(requested_back_m)

    for bm in _safe_back_m_candidates(
        float(requested_back_m),
        has_final_q=final_q is not None,
        fallback_back_m=fallback_back_m,
    ):
        score = None
        safe_pos = target_pos - float(bm) * pointing
        reach_ok, d_sh, limit = _eef_target_reach_preflight(
            world, arm, safe_pos, label=f"{stage_name}_back{bm:.2f}", ctx=None,
        )
        if not reach_ok:
            meta = {
                "ok": False,
                "back_m": float(bm),
                "reason": "reach_preflight",
                "shoulder_dist_m": float(d_sh),
                "limit_m": float(limit),
            }
            score = float("inf")
        elif abs(float(bm)) <= 1e-6 and final_q is not None:
            from behavior_interface.skills.grasp import _set_arm_qpos_direct

            saved = None
            q_probe = np.asarray(final_q, dtype=np.float64).reshape(7)
            try:
                saved = world.robot.get_joint_positions().clone()
                _set_arm_qpos_direct(world, arm, q_probe)
                pos_err, ori_err, app_err = _eef_pose_err(
                    world, arm, safe_pos, target_quat
                )
            finally:
                if saved is not None:
                    try:
                        world.robot.set_joint_positions(saved)
                    except Exception:
                        pass
            anchors, path_meta = _eef_plan_to_safe_xyz_anchor_path(
                world,
                arm,
                safe_pos,
                target_quat,
                safe_q=q_probe,
                final_q=final_q,
                ctx=ctx,
                stage_name=f"{stage_name}_probe_back{bm:.2f}_xyz",
                max_joint_gap_rad=float(final_branch_gap_rad),
                require_endpoint=True,
            )
            meta = {
                "ok": bool(
                    pos_err <= _EEF_SAFE_POS_TOL_M
                    and ori_err <= _EEF_SAFE_ORI_TOL_DEG
                    and anchors
                ),
                "back_m": 0.0,
                "reason": "final_as_safe",
                "final_as_safe": True,
                "safe_q": q_probe.tolist(),
                "final_q_fk_pos_err_m": float(pos_err),
                "final_q_fk_ori_err_deg": float(ori_err),
                "final_q_fk_approach_err_deg": float(app_err),
                "to_safe_path": path_meta,
            }
            if anchors:
                meta["_to_safe_anchors"] = anchors
            if meta["ok"]:
                ctx.log(
                    f"    [{stage_name}] adapt safe back_m "
                    f"{float(requested_back_m)*100:.0f}cm→0cm "
                    f"(final-as-safe, FK={pos_err*1000:.1f}mm/{ori_err:.2f}°, "
                    f"xyz_anchors={len(anchors)})"
                )
            elif pos_err <= _EEF_SAFE_POS_TOL_M and ori_err <= _EEF_SAFE_ORI_TOL_DEG:
                ctx.log(
                    f"    [{stage_name}] reject back=0cm final-as-safe: "
                    f"current→final XYZ branch path failed "
                    f"accepted={path_meta.get('accepted')} "
                    f"error={path_meta.get('error')}"
                )
            if meta["ok"]:
                score = 0.0
            else:
                # If the full offline same-branch path fails, execution can
                # still use online XYZ tracking.  Prefer a real retreat near
                # the requested back distance over a tiny near-final safe pose.
                score = (
                    abs(float(bm) - float(requested_back_m))
                    + (0.25 if float(bm) < 0.045 else 0.0)
                )
        elif final_q is not None:
            seed_list = [
                ("final_branch", np.asarray(final_q, dtype=np.float64).reshape(7)),
                ("current", _arm_qpos_now(world, arm)),
            ]
            best = None
            solve_failures = []
            for seed_label, seed_q in seed_list:
                q_try, pos_try, ori_try = _eef_solve_6d_dls_arm_q(
                    world,
                    arm,
                    safe_pos,
                    target_quat,
                    pos_tol=_EEF_SAFE_POS_TOL_M,
                    ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
                    max_steps=320,
                    max_dq_per_step=0.065,
                    max_dx_per_step=0.020,
                    max_dw_per_step=0.10,
                    ori_weight=0.65,
                    lam=0.08,
                    seed_q=seed_q,
                    nominal_q=final_q,
                    nominal_weight=0.035,
                )
                if q_try is None:
                    solve_failures.append((seed_label, float(pos_try), float(ori_try)))
                    continue
                final_gap = float(np.linalg.norm(
                    np.asarray(q_try, dtype=np.float64).reshape(7)
                    - np.asarray(final_q, dtype=np.float64).reshape(7),
                    ord=np.inf,
                ))
                score_try = float(pos_try) + 0.002 * float(ori_try) + 0.010 * final_gap
                if best is None or score_try < best[0]:
                    best = (score_try, q_try.copy(), float(pos_try), float(ori_try), final_gap, seed_label)
            if best is not None:
                _, q_safe, pos_err, ori_err, final_gap, seed_label = best
                meta = {
                    "ok": False,
                    "back_m": float(bm),
                    "reason": "safe_endpoint_6d_ik",
                    "safe_q": q_safe.tolist(),
                    "safe_endpoint_pos_err_m": float(pos_err),
                    "safe_endpoint_ori_err_deg": float(ori_err),
                    "safe_to_final_joint_gap_rad": float(final_gap),
                    "seed": seed_label,
                }
                ctx.log(
                    f"    [{stage_name}] safe endpoint IK back={bm*100:.0f}cm "
                    f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° "
                    f"gap_to_final={final_gap:.3f}rad seed={seed_label}"
                )
                if final_gap > float(final_branch_gap_rad):
                    ctx.log(
                        f"    [{stage_name}] keep back={bm*100:.0f}cm "
                        f"FK-valid safe endpoint despite large safe→final gap: "
                        f"safe→final joint gap={final_gap:.3f}rad>"
                        f"{float(final_branch_gap_rad):.3f}rad; "
                        "bridge deleted Cartesian samples with bounded joint interpolation"
                    )
                    meta["safe_final_branch_gap_soft"] = True
                if (
                    float(bm) > 1e-4
                    and final_gap <= 1e-4
                    and float(pos_err) > 0.50 * _EEF_SAFE_POS_TOL_M
                ):
                    ctx.log(
                        f"    [{stage_name}] reject back={bm*100:.0f}cm "
                        "safe endpoint collapsed to final q "
                        f"(pos={pos_err*1000:.0f}mm); try smaller/zero back_m"
                    )
                    meta["reason"] = "safe_collapsed_to_final_q"
                    meta["error"] = "safe_collapsed_to_final_q"
                    score = (
                        abs(float(bm) - float(requested_back_m))
                        + 0.20
                        + float(pos_err)
                    )
                else:
                    anchors, path_meta = _eef_plan_to_safe_xyz_anchor_path(
                        world,
                        arm,
                        safe_pos,
                        target_quat,
                        safe_q=q_safe,
                        final_q=final_q,
                        ctx=ctx,
                        stage_name=f"{stage_name}_probe_back{bm:.2f}_xyz",
                        max_joint_gap_rad=float(final_branch_gap_rad),
                        require_endpoint=False,
                    )
                    meta["to_safe_path"] = path_meta
                    if anchors:
                        meta["ok"] = True
                        meta["_to_safe_anchors"] = anchors
                        meta["to_safe_anchor_count"] = int(len(anchors))
                        score = abs(float(bm) - float(requested_back_m)) + 0.01 * min(final_gap, 1.5)
                    else:
                        meta["reason"] = "to_safe_xyz_waypoint_ik_failed"
                        meta["error"] = path_meta.get("error") or "to_safe_xyz_waypoint_ik_failed"
                        ctx.log(
                            f"    [{stage_name}] reject back={bm*100:.0f}cm "
                            f"current→safe XYZ branch path failed "
                            f"accepted={path_meta.get('accepted')} "
                            f"error={path_meta.get('error')}"
                        )
                        score = (
                            abs(float(bm) - float(requested_back_m))
                            + 0.01 * min(float(final_gap), 1.5)
                            + (0.25 if float(bm) < 0.045 else 0.0)
                        )
            else:
                meta = {
                    "ok": False,
                    "back_m": float(bm),
                    "reason": "safe_endpoint_6d_ik_failed",
                }
                if solve_failures:
                    meta["endpoint_solve_attempts"] = [
                        {
                            "seed": seed_label,
                            "pos_err_m": float(pos_err),
                            "ori_err_deg": float(ori_err),
                        }
                        for seed_label, pos_err, ori_err in solve_failures
                    ]
                    fail_s = "; ".join(
                        f"{seed}:pos={pos_err*1000:.0f}mm ori={ori_err:.1f}°"
                        for seed, pos_err, ori_err in solve_failures
                    )
                else:
                    fail_s = "no seed attempts"
                ctx.log(
                    f"    [{stage_name}] reject back={bm*100:.0f}cm "
                    f"safe endpoint 6D IK failed ({fail_s})"
                )
        else:
            anchors, meta = _eef_plan_6d_joint_anchors(
                world,
                arm,
                safe_pos,
                target_quat,
                ctx=ctx,
                stage_name=f"{stage_name}_probe_back{bm:.2f}",
                pos_tol=_EEF_SAFE_POS_TOL_M,
                ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
                intermediate_pos_only=True,
                max_anchor_drop_ratio=0.0,
                final_seed_q=final_q,
            )
            meta = dict(meta)
            meta["back_m"] = float(bm)
            final_gap = None
            if anchors and final_q is not None:
                final_gap = float(np.linalg.norm(
                    np.asarray(final_q, dtype=np.float64).reshape(7)
                    - np.asarray(anchors[-1]["q"], dtype=np.float64).reshape(7),
                    ord=np.inf,
                ))
                meta["safe_to_final_joint_gap_rad"] = final_gap
                if final_gap > float(final_branch_gap_rad):
                    meta["ok"] = False
                    meta["error"] = "safe_final_branch_gap"
                    ctx.log(
                        f"    [{stage_name}] reject back={bm*100:.0f}cm "
                        f"safe→final gap={final_gap:.3f}rad>{float(final_branch_gap_rad):.3f}rad"
                    )
                else:
                    meta["ok"] = True
            else:
                meta["ok"] = bool(anchors)
            if anchors:
                meta["_to_safe_anchors"] = anchors
                meta["to_safe_anchor_count"] = int(len(anchors))
            score = (
                (abs(float(bm) - float(requested_back_m)) + 0.05 * float(final_gap or 0.0))
                if meta.get("ok")
                else float(meta.get("accepted", 0)) * -1.0
            )
        if meta.get("ok"):
            if abs(float(bm) - float(requested_back_m)) > 1e-4:
                ctx.log(
                    f"    [{stage_name}] adapt safe back_m "
                    f"{float(requested_back_m)*100:.0f}cm→{float(bm)*100:.0f}cm"
                )
            return safe_pos, float(bm), meta
        if score is None:
            score = (
                abs(float(bm) - float(requested_back_m))
                + 0.35
                + (0.25 if float(bm) < 0.045 else 0.0)
            )
        if score < best_score:
            best_score = float(score)
            best_pos = safe_pos
            best_back = float(bm)
            best_meta = meta

    ctx.log(
        f"    [{stage_name}] no safe back_m produced executable to-safe line; "
        f"best_back={best_back*100:.0f}cm reason={best_meta.get('error') or best_meta.get('reason')}"
    )
    return best_pos, best_back, best_meta


def _eef_goto_6d_curobo_ik(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_attempts: int = 24,
    timeout: float = 4.0,
    max_frames: int = 90,
    line_start_pos=None,
    line_target_pos=None,
    max_line_deviation_m: Optional[float] = None,
    line_deviation_margin_m: float = 0.0,
) -> tuple[float, float]:
    """Solve a strict 6DoF IK target, then play only arm joints toward it."""
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    from behavior_interface.skills.grasp import _get_curobo_mg
    from behavior_interface.gpu_diag import log_gpu_diag

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    line_start_arr = None
    line_target_arr = None
    line_limit = None
    if (
        max_line_deviation_m is not None
        and line_start_pos is not None
        and line_target_pos is not None
    ):
        try:
            line_start_arr = np.asarray(line_start_pos, dtype=np.float64).reshape(3)
            line_target_arr = np.asarray(line_target_pos, dtype=np.float64).reshape(3)
            line_limit = float(max_line_deviation_m) + max(0.0, float(line_deviation_margin_m))
        except Exception:
            line_start_arr = None
            line_target_arr = None
            line_limit = None
    try:
        log_gpu_diag(
            ctx.log,
            "curobo_ik.request",
            include_nvidia=False,
            extra={
                "stage": stage_name,
                "arm": arm,
                "target_pos": [round(float(x), 4) for x in target_pos.tolist()],
                "pos_tol_m": float(pos_tol),
                "ori_tol_deg": float(ori_tol_deg),
                "max_attempts": int(max_attempts),
                "timeout": float(timeout),
            },
        )
    except Exception:
        pass
    try:
        mg = _get_curobo_mg(world)
    except Exception as e:
        ctx.log(f"    [{stage_name}] curobo IK 初始化失败: {e}")
        try:
            log_gpu_diag(
                ctx.log,
                "curobo_ik.init_failed",
                extra={
                    "stage": stage_name,
                    "arm": arm,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
        except Exception:
            pass
        return float("inf"), float("inf")

    robot = world.robot
    bs = int(mg.batch_size)
    link_name = robot.eef_link_names[arm]
    tp = {
        link_name: th.stack([
            th.tensor(target_pos.tolist(), dtype=th.float32)
            for _ in range(bs)
        ])
    }
    tq = {
        link_name: th.stack([
            th.tensor(target_quat.tolist(), dtype=th.float32)
            for _ in range(bs)
        ])
    }
    for other in robot.arm_names:
        if other == arm:
            continue
        try:
            ep = world.eef_pose(arm=other)
            ol = robot.eef_link_names[other]
            tp[ol] = th.stack([
                th.tensor(list(ep["pos"]), dtype=th.float32)
                for _ in range(bs)
            ])
            tq[ol] = th.stack([
                th.tensor(list(ep["quat"]), dtype=th.float32)
                for _ in range(bs)
            ])
        except Exception:
            pass

    ctx.log(
        f"    [{stage_name}] curobo IK-only 6DoF → "
        f"({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
        f"quat=({target_quat[0]:+.3f},{target_quat[1]:+.3f},"
        f"{target_quat[2]:+.3f},{target_quat[3]:+.3f})"
    )
    try:
        res = mg.compute_trajectories(
            target_pos=tp,
            target_quat=tq,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=max(1, int(math.ceil(float(max_attempts) / max(1, bs)))),
            timeout=float(timeout),
            ik_fail_return=50,
            enable_finetune_trajopt=False,
            finetune_attempts=0,
            return_full_result=True,
            success_ratio=1.0 / bs,
            attached_obj=None,
            attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=True,
            ik_only=True,
            ik_world_collision_check=False,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
    except Exception as e:
        ctx.log(f"    [{stage_name}] curobo IK 异常: {type(e).__name__}: {e}")
        return float("inf"), float("inf")

    def _extract_arm_q_from_js(js_obj, batch_i: int):
        names_full = list(robot.joints.keys())
        try:
            arm_idx_full = [names_full.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
        except ValueError:
            return None, "missing_og_arm_joint"
        try:
            js_pos_full_local = js_obj.position
            js_names_local = list(js_obj.joint_names)
            if js_pos_full_local.dim() == 1:
                js_pos_local = js_pos_full_local.cpu().float()
            else:
                rows = js_pos_full_local.reshape(-1, int(js_pos_full_local.shape[-1]))
                bi = max(0, min(int(batch_i), int(rows.shape[0]) - 1))
                js_pos_local = rows[bi].cpu().float()
        except Exception as e:
            return None, f"read_js_failed:{type(e).__name__}:{e}"

        by_name = {str(n): i for i, n in enumerate(js_names_local)}
        direct = []
        for jn in (f"{arm}_arm_joint{i + 1}" for i in range(7)):
            idx = by_name.get(jn)
            if idx is None or idx >= len(js_pos_local):
                direct = []
                break
            direct.append(float(js_pos_local[idx].item()))
        if len(direct) == 7:
            return direct, f"direct_js_names n={len(js_names_local)}"

        # Some cuRobo IK JointState objects are in internal kinematics order and
        # do not carry OG joint names.  Reorder through the full robot joint
        # order, then slice the OG arm joints.
        try:
            import torch as _th
            from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as _CES

            q_cur_full = robot.get_joint_positions().cpu().float()
            q_target_full = q_cur_full.clone()
            src_names = list(js_names_local)
            dst_names = list(getattr(mg, "robot_joint_names", []) or [])
            if len(src_names) != int(js_pos_local.numel()) or not dst_names:
                ordered = type(js_obj)(
                    position=js_pos_local.reshape(1, -1).cuda(),
                    joint_names=src_names,
                ).get_ordered_joint_state(mg.mg[_CES.ARM].kinematics.joint_names)
                src_names = list(ordered.joint_names)
                js_pos_local = ordered.position.reshape(-1).cpu().float()
            src_map = {str(n): i for i, n in enumerate(src_names)}
            wrote = 0
            for jn in names_full:
                idx_src = src_map.get(jn)
                if idx_src is not None and idx_src < len(js_pos_local):
                    q_target_full[names_full.index(jn)] = js_pos_local[idx_src]
                    wrote += 1
            if wrote == 0 and len(dst_names) == int(js_pos_local.numel()):
                for i, jn in enumerate(dst_names):
                    if jn in names_full:
                        q_target_full[names_full.index(jn)] = js_pos_local[i]
                        wrote += 1
            if wrote == 0:
                return None, (
                    f"no_name_overlap js_first={src_names[:5]} "
                    f"mg_first={dst_names[:5]}"
                )
            return [float(q_target_full[i].item()) for i in arm_idx_full], (
                f"full_reorder wrote={wrote} js_first={src_names[:3]}"
            )
        except Exception as e:
            return None, f"full_reorder_failed:{type(e).__name__}:{e}"

    names_full = list(robot.joints.keys())
    try:
        arm_idx_full = [names_full.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
    except ValueError:
        ctx.log(f"    [{stage_name}] IK solution 无法定位 {arm} arm joint")
        return float("inf"), float("inf")

    results = res if isinstance(res, (list, tuple)) else [res]
    raw_candidates = []
    for r in results:
        pe = getattr(r, "position_error", None)
        re = getattr(r, "rotation_error", None)
        js = getattr(r, "js_solution", None)
        if pe is None or js is None:
            continue
        pe_pb = pe.max(dim=0).values if pe.dim() == 2 else pe
        re_pb = (
            re.max(dim=0).values
            if (re is not None and re.dim() == 2)
            else (re if re is not None else None)
        )
        s = getattr(r, "success", None)
        s_t = s if isinstance(s, th.Tensor) else None
        for b in range(pe_pb.numel()):
            pv = float(pe_pb.flatten()[b].item())
            rv = float(re_pb.flatten()[b].item()) if re_pb is not None else 0.0
            ok_flag = bool(s_t.flatten()[b].item()) if s_t is not None and b < s_t.numel() else False
            q_arm, q_source = _extract_arm_q_from_js(js, int(b))
            if q_arm is None:
                continue
            raw_candidates.append({
                "batch": int(b),
                "q": np.asarray(q_arm, dtype=np.float64).reshape(7),
                "source": q_source,
                "cu_pos": pv,
                "cu_rot_rad": rv,
                "cu_ok": ok_flag,
            })

    if not raw_candidates:
        ctx.log(f"    [{stage_name}] curobo IK 无可执行 joint 候选")
        return float("inf"), float("inf")

    saved_q = None
    checked = []
    try:
        saved_q = robot.get_joint_positions().clone()
        for cand in raw_candidates:
            q_probe = saved_q.clone()
            for i, j in enumerate(arm_idx_full):
                q_probe[int(j)] = float(cand["q"][i])
            robot.set_joint_positions(q_probe)
            og_pos_err, og_ori_err, og_app_err = _eef_pose_err(
                world, arm, target_pos, target_quat
            )
            q_delta = float(np.linalg.norm(
                cand["q"] - saved_q.cpu().numpy()[arm_idx_full],
                ord=np.inf,
            ))
            # Rank by the pose actually seen by OmniGibson, not by cuRobo's
            # internal FK error.  Full orientation is weighted more than the
            # approach-only gate here because this function is the strict 6DoF
            # path used for safe/final poses.
            score = (
                float(og_pos_err)
                + 0.040 * math.radians(float(og_ori_err))
                + 0.020 * math.radians(float(og_app_err))
                + 0.002 * q_delta
                + (0.0 if bool(cand["cu_ok"]) else 0.004)
            )
            cand.update({
                "og_pos": float(og_pos_err),
                "og_ori": float(og_ori_err),
                "og_app": float(og_app_err),
                "q_delta": q_delta,
                "score": float(score),
            })
            checked.append(cand)
    except Exception as e:
        ctx.log(f"    [{stage_name}] OG FK 复核 IK 候选失败: {type(e).__name__}: {e}")
        return float("inf"), float("inf")
    finally:
        if saved_q is not None:
            try:
                robot.set_joint_positions(saved_q)
            except Exception:
                pass

    if not checked:
        ctx.log(f"    [{stage_name}] curobo IK 候选无法通过 OG FK 复核")
        return float("inf"), float("inf")

    checked.sort(key=lambda c: float(c["score"]))
    for rank, cand in enumerate(checked[: min(3, len(checked))], start=1):
        ctx.log(
            f"    [{stage_name}] IK cand#{rank} batch={cand['batch']} "
            f"cu=({cand['cu_pos']*1000:.1f}mm,{math.degrees(cand['cu_rot_rad']):.1f}°,"
            f"ok={cand['cu_ok']}) "
            f"OG=({cand['og_pos']*1000:.1f}mm,{cand['og_ori']:.1f}°,"
            f"app={cand['og_app']:.1f}°) dq={cand['q_delta']:.3f}"
        )

    pass_gate = [
        cand for cand in checked
        if float(cand["og_pos"]) <= float(pos_tol)
        and float(cand["og_ori"]) <= float(ori_tol_deg)
    ]
    if pass_gate:
        best = pass_gate[0]
        if best is not checked[0]:
            ctx.log(
                f"    [{stage_name}] IK score-best fails strict gate; "
                f"using gated cand batch={best['batch']} "
                f"OG=({best['og_pos']*1000:.1f}mm,{best['og_ori']:.1f}°) "
                f"instead of batch={checked[0]['batch']} "
                f"OG=({checked[0]['og_pos']*1000:.1f}mm,{checked[0]['og_ori']:.1f}°)"
            )
    else:
        best = checked[0]
    q_arm_target = best["q"].tolist()
    q_source = best["source"]
    pos_ik_err = float(best["cu_pos"])
    rot_ik_deg = float(math.degrees(float(best["cu_rot_rad"])))
    og_pos_err = float(best["og_pos"])
    og_ori_err = float(best["og_ori"])
    og_app_err = float(best["og_app"])
    ok_flag = bool(best["cu_ok"])
    ctx.log(
        f"    [{stage_name}] curobo IK selected success={ok_flag} "
        f"cu_pos={pos_ik_err*1000:.1f}mm cu_rot={rot_ik_deg:.1f}° "
        f"OG_pos={og_pos_err*1000:.1f}mm OG_ori={og_ori_err:.1f}° "
        f"OG_app={og_app_err:.1f}° batch={best['batch']}"
    )
    if og_pos_err > float(pos_tol) or og_ori_err > float(ori_tol_deg):
        ctx.log(
            f"    [{stage_name}] curobo IK candidate rejected by OG FK strict gate: "
            f"pos={og_pos_err * 1000:.1f}mm>{float(pos_tol) * 1000:.1f}mm "
            f"or ori={og_ori_err:.1f}°>{float(ori_tol_deg):.1f}°; "
            f"no gated candidate among {len(checked)} checked solutions; "
            "not playing approximate q"
        )
        return float("inf"), float("inf")

    q0_arm = robot.get_joint_positions().cpu().float()[arm_idx_full]
    q1_arm = th.tensor(q_arm_target, dtype=q0_arm.dtype)
    max_delta = float(th.max(th.abs(q1_arm - q0_arm)).item())
    # Large wrist/shoulder reorientations need more controller-settle time than
    # the small-pose smoke tests.  cuRobo can return a strict FK-valid q, but if
    # we rush the absolute joint controller the real arm lags and the following
    # DLS fallback starts from a bad pose.
    if _is_safe_q_residual_stage(stage_name):
        n_frames = max(4, min(int(max_frames), int(math.ceil(max_delta / 0.120)) + 4))
    else:
        n_frames = max(8, min(int(max_frames), int(math.ceil(max_delta / 0.045)) + 10))
    ctx.log(
        f"    [{stage_name}] play IK arm joints frames={n_frames} "
        f"max_delta={max_delta:.3f}rad source={q_source}"
    )
    for fi in range(1, n_frames + 1):
        s = fi / n_frames
        ss = s * s * (3.0 - 2.0 * s)
        q_arm = (q0_arm + (q1_arm - q0_arm) * float(ss)).tolist()
        yield _make_arm_q_action(world, arm, [float(x) for x in q_arm], gripper_cmd)
        if line_limit is not None and line_start_arr is not None and line_target_arr is not None:
            try:
                eef_now = world.eef_pose(arm=arm)
                dev = _dist_point_to_segment(
                    np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                    line_start_arr,
                    line_target_arr,
                )
            except Exception:
                dev = 0.0
            if dev > float(line_limit):
                ctx.log(
                    f"    [{stage_name}] IK repair left line corridor frame={fi}/{n_frames} "
                    f"line_dev={dev*1000:.0f}mm>{float(line_limit)*1000:.0f}mm"
                )
                return float("inf"), float("inf")
    q1_arm_np = np.asarray(q1_arm.tolist(), dtype=np.float64).reshape(7)
    q1_arm_list = [float(x) for x in q1_arm_np.tolist()]
    if _is_safe_q_residual_stage(stage_name):
        settle_frames = max(2, min(8, int(math.ceil(max_delta / 0.120)) + 2))
    else:
        settle_frames = max(24, min(160, int(math.ceil(max_delta / 0.025)) + 24))
    q_track = float("inf")
    for si in range(settle_frames):
        q_track = float(np.linalg.norm(_arm_qpos_now(world, arm) - q1_arm_np, ord=np.inf))
        if q_track <= 0.018:
            break
        yield _make_arm_q_action(world, arm, q1_arm_list, gripper_cmd)
        if line_limit is not None and line_start_arr is not None and line_target_arr is not None:
            try:
                eef_now = world.eef_pose(arm=arm)
                dev = _dist_point_to_segment(
                    np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                    line_start_arr,
                    line_target_arr,
                )
            except Exception:
                dev = 0.0
            if dev > float(line_limit):
                ctx.log(
                    f"    [{stage_name}] IK repair settle left line corridor "
                    f"line_dev={dev*1000:.0f}mm>{float(line_limit)*1000:.0f}mm "
                    f"q_track={q_track:.3f}rad"
                )
                return float("inf"), float("inf")
    if q_track > 0.018:
        if q_track > 0.018:
            ctx.log(
                f"    [{stage_name}] IK target settle incomplete "
                f"q_track={q_track:.3f}rad after {settle_frames} hold frames"
            )

    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
        world, arm, target_pos, target_quat
    )
    ctx.log(
        f"    [{stage_name}] IK play final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}°"
    )
    return float(final_pos_err), float(final_ori_err)


def _eef_solve_6d_dls_arm_q(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int = 180,
    max_dq_per_step: float = 0.070,
    max_dx_per_step: float = 0.025,
    max_dw_per_step: float = 0.12,
    ori_weight: float = 0.55,
    lam: float = 0.08,
    seed_q: Optional[np.ndarray] = None,
    nominal_q: Optional[np.ndarray] = None,
    nominal_weight: float = 0.0,
    nominal_tol_rad: Optional[float] = None,
    min_steps: int = 0,
) -> tuple[Optional[np.ndarray], float, float]:
    """Side-effect-free kinematic DLS solve; returns arm q target then restores."""
    from behavior_interface.skills.grasp import (
        _arm_qpos,
        _get_arm_dof_idx,
        _dls_solve_dq,
        _orientation_error_omega,
        _quat_to_mat,
        _read_jacobian_arm,
        _set_arm_qpos_direct,
    )

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    target_R = _quat_to_mat(target_quat)
    q_nominal = (
        np.asarray(nominal_q, dtype=np.float64).reshape(7)
        if nominal_q is not None else None
    )
    saved = None
    q_solution: Optional[np.ndarray] = None
    best_ok_q: Optional[np.ndarray] = None
    best_ok_score = float("inf")
    best_ok_pos_err = float("inf")
    best_ok_ori_err = float("inf")
    final_pos_err = float("inf")
    final_ori_err = float("inf")
    try:
        saved = world.robot.get_joint_positions().clone()
        joint_names = list(world.robot.joints.keys())
        arm_idx = _get_arm_dof_idx(world, arm)
        q_lo = np.full(7, -np.inf, dtype=np.float64)
        q_hi = np.full(7, np.inf, dtype=np.float64)
        for i, jidx in enumerate(arm_idx):
            try:
                joint = world.robot.joints[joint_names[int(jidx)]]
                q_lo[i] = float(joint.lower_limit)
                q_hi[i] = float(joint.upper_limit)
            except Exception:
                pass
        if seed_q is not None:
            q_seed = np.asarray(seed_q, dtype=np.float64).reshape(7)
            _set_arm_qpos_direct(world, arm, np.clip(q_seed, q_lo, q_hi))
        limit_eps = 1e-4
        for _step in range(int(max_steps)):
            eef = world.eef_pose(arm=arm)
            epos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
            equat = _quat_normalize_xyzw(eef["quat"])
            cur_R = _quat_to_mat(equat)
            q_cur = np.asarray(_arm_qpos(world, arm), dtype=np.float64).reshape(7)
            dx = target_pos - epos
            pos_err = float(np.linalg.norm(dx))
            omega = _orientation_error_omega(target_R, cur_R)
            ori_err_rad = float(np.linalg.norm(omega))
            ori_err_deg = float(math.degrees(ori_err_rad))
            final_pos_err, final_ori_err = pos_err, ori_err_deg
            if pos_err <= float(pos_tol) and ori_err_deg <= float(ori_tol_deg):
                nominal_err = (
                    float(np.linalg.norm(q_cur - q_nominal, ord=np.inf))
                    if q_nominal is not None and float(nominal_weight) > 0.0
                    else 0.0
                )
                ok_score = float(pos_err) + 0.0005 * float(ori_err_deg) + 0.002 * nominal_err
                if ok_score < best_ok_score:
                    best_ok_q = q_cur.copy()
                    best_ok_score = ok_score
                    best_ok_pos_err = pos_err
                    best_ok_ori_err = ori_err_deg
                nominal_ok = (
                    nominal_tol_rad is None
                    or q_nominal is None
                    or float(nominal_weight) <= 0.0
                    or nominal_err <= float(nominal_tol_rad)
                )
                if _step >= int(min_steps) and nominal_ok:
                    q_solution = q_cur.copy()
                    break

            if pos_err > float(max_dx_per_step):
                dx = dx * (float(max_dx_per_step) / (pos_err + 1e-12))
            if ori_err_rad > float(max_dw_per_step):
                omega = omega * (float(max_dw_per_step) / (ori_err_rad + 1e-12))

            J, _ = _read_jacobian_arm(world, arm)
            # If orientation is already good, let position dominate the last centimeters.
            if float(ori_weight) <= 0.0:
                ow = 0.0
            else:
                ow = float(ori_weight) if ori_err_deg > max(3.0, float(ori_tol_deg) * 0.5) else 0.18
            J_scaled = np.vstack([J[:3, :], ow * J[3:6, :]])
            rhs = np.concatenate([dx, ow * omega])
            nw = float(nominal_weight)
            if q_nominal is not None and nw > 0.0:
                # Nullspace-ish branch bias: keep the hard task Cartesian, but
                # prefer the plan-filter final branch so the last safe waypoint
                # does not require a discontinuous wrist/elbow flip.
                q_to_nom = np.clip(q_nominal - q_cur, -0.18, 0.18)
                J_scaled = np.vstack([J_scaled, nw * np.eye(7)])
                rhs = np.concatenate([rhs, nw * q_to_nom])
            dq = _dls_solve_dq(J_scaled, rhs, lam=float(lam))
            # Respect true URDF / controller limits while solving. Direct
            # set_joint_positions can exceed limits; actions cannot, so an
            # unclamped probe produces fake IK solutions that never execute.
            for i in range(7):
                if (q_cur[i] >= q_hi[i] - limit_eps and dq[i] > 0.0) or (
                    q_cur[i] <= q_lo[i] + limit_eps and dq[i] < 0.0
                ):
                    dq[i] = 0.0
            dq_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_inf > float(max_dq_per_step):
                dq = dq * (float(max_dq_per_step) / (dq_inf + 1e-12))
            q_next = np.clip(q_cur + dq, q_lo, q_hi)
            _set_arm_qpos_direct(world, arm, q_next)

    except Exception:
        q_solution = None
        final_pos_err = float("inf")
        final_ori_err = float("inf")
    finally:
        if saved is not None:
            try:
                world.robot.set_joint_positions(saved)
            except Exception:
                pass
    if q_solution is None and best_ok_q is not None:
        q_solution = best_ok_q.copy()
        final_pos_err = best_ok_pos_err
        final_ori_err = best_ok_ori_err
    return q_solution, float(final_pos_err), float(final_ori_err)


def _arm_qpos_now(world, arm: str) -> np.ndarray:
    from behavior_interface.skills.grasp import _arm_qpos
    return np.asarray(_arm_qpos(world, arm), dtype=np.float64).reshape(7)


def _probe_arm_q_pose_err(world, arm: str, q_arm, target_pos, target_quat):
    """Temporarily FK-probe one arm q against a target pose, then restore state."""
    robot = world.robot
    saved = None
    saved_vel = None
    try:
        saved = robot.get_joint_positions().clone()
        try:
            saved_vel = robot.get_joint_velocities().clone()
        except Exception:
            saved_vel = None
        q_probe = saved.clone()
        idx = _get_arm_dof_idx(world, arm)
        q_arm_np = np.asarray(q_arm, dtype=np.float64).reshape(7)
        for local_i, joint_i in enumerate(idx):
            q_probe[int(joint_i)] = float(q_arm_np[int(local_i)])
        robot.set_joint_positions(q_probe)
        pos_err, ori_err, app_err = _eef_pose_err(world, arm, target_pos, target_quat)
        pose = world.eef_pose(arm=arm)
        return float(pos_err), float(ori_err), float(app_err), pose
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


def _play_arm_q_target(
    world,
    arm: str,
    q_target,
    *,
    gripper_cmd: Optional[float],
    max_frames: int,
    hold_frames: int = 4,
    min_hold_frames: int = 0,
    joint_tol: float = 0.010,
    ctx=None,
    stage_name: str = "",
    line_start_pos=None,
    line_target_pos=None,
    max_line_deviation_m: Optional[float] = None,
    allow_direct_residual_correction: bool = False,
):
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    q0 = _arm_qpos_now(world, arm)
    q1 = np.asarray(q_target, dtype=np.float64).reshape(7)
    max_delta = float(np.linalg.norm(q1 - q0, ord=np.inf))
    line_stage = (
        _is_safe_q_residual_stage(stage_name)
        or str(stage_name or "").startswith("eef_pose_final_q_translate")
        or str(stage_name or "").startswith("move_only_line")
        or str(stage_name or "").startswith("lift_fast_q")
    )
    if line_stage:
        stage_s = str(stage_name or "")
        if _is_safe_q_residual_stage(stage_name):
            q_step = 0.040
        elif stage_s.startswith("eef_pose_final_q_translate"):
            q_step = 0.080
        elif stage_s.startswith("move_only_line"):
            q_step = 0.070
        else:
            q_step = 0.120
        n_frames = max(2, min(int(max_frames), int(math.ceil(max_delta / q_step)) + 1))
        if ctx is not None:
            ctx.log(
                f"    [{stage_name}] Cartesian-line q interpolation "
                f"max_delta={max_delta:.3f}rad frames={n_frames}"
            )
    else:
        n_frames = max(3, min(int(max_frames), int(math.ceil(max_delta / 0.220)) + 2))
    for fi in range(1, n_frames + 1):
        s = fi / n_frames
        ss = s * s * (3.0 - 2.0 * s)
        q = q0 + (q1 - q0) * float(ss)
        yield _make_arm_q_action(world, arm, q.tolist(), gripper_cmd)
        if (
            max_line_deviation_m is not None
            and line_start_pos is not None
            and line_target_pos is not None
        ):
            try:
                eef_now = world.eef_pose(arm=arm)
                dev = _dist_point_to_segment(
                    np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                    np.asarray(line_start_pos, dtype=np.float64).reshape(3),
                    np.asarray(line_target_pos, dtype=np.float64).reshape(3),
                )
            except Exception:
                dev = 0.0
            if dev > float(max_line_deviation_m):
                if ctx is not None:
                    ctx.log(
                        f"    [{stage_name}] ABORT frame line deviation "
                        f"{dev*1000:.0f}mm>{float(max_line_deviation_m)*1000:.0f}mm"
                    )
                return None if _is_repairable_line_stage(stage_name) else False
    # Absolute-position arm controllers can report joint error before the link
    # pose has settled.  For Cartesian-line anchor playback, keep commanding the
    # anchor until q tracking is actually close; the planner already gave us the
    # desired branch, this is executor-side tracking robustness.
    hold_n = max(0, int(hold_frames))
    min_hold_n = min(hold_n, max(0, int(min_hold_frames)))
    if str(stage_name or "").startswith("move_only_line"):
        hold_n = max(hold_n, _MOVE_ONLY_LINE_HOLD_MAX_FRAMES)
        min_hold_n = min(hold_n, min_hold_n)
        joint_tol = min(float(joint_tol), _MOVE_ONLY_LINE_Q_TRACK_TOL_RAD)
    q_reached = False
    last_line_dev = None
    for hold_i in range(hold_n):
        yield _make_arm_q_action(world, arm, q1.tolist(), gripper_cmd)
        qerr_now = float(np.linalg.norm(_arm_qpos_now(world, arm) - q1, ord=np.inf))
        if (
            max_line_deviation_m is not None
            and line_start_pos is not None
            and line_target_pos is not None
        ):
            try:
                eef_now = world.eef_pose(arm=arm)
                last_line_dev = _dist_point_to_segment(
                    np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                    np.asarray(line_start_pos, dtype=np.float64).reshape(3),
                    np.asarray(line_target_pos, dtype=np.float64).reshape(3),
                )
            except Exception:
                last_line_dev = None
            if last_line_dev is not None and last_line_dev > float(max_line_deviation_m):
                if ctx is not None:
                    ctx.log(
                        f"    [{stage_name}] ABORT hold line deviation "
                        f"{last_line_dev*1000:.0f}mm>{float(max_line_deviation_m)*1000:.0f}mm "
                        f"q_track={qerr_now:.3f}rad"
                    )
                return None if _is_repairable_line_stage(stage_name) else False
        if hold_i + 1 >= min_hold_n and qerr_now <= float(joint_tol):
            q_reached = True
            break
    qerr = float(np.linalg.norm(_arm_qpos_now(world, arm) - q1, ord=np.inf))
    if qerr <= float(joint_tol):
        q_reached = True
    if allow_direct_residual_correction and ctx is not None:
        ctx.log(
            f"    [{stage_name}] direct residual correction disabled; "
            "controller tracking remains collision-checked"
        )
    if qerr > float(joint_tol):
        if ctx is not None:
            e = q1 - _arm_qpos_now(world, arm)
            e_str = "[" + ",".join(f"{v:+.3f}" for v in e) + "]"
            ctx.log(
                f"    [{stage_name}] joint target not reached "
                f"q_track={qerr:.3f}rad err={e_str}"
                + (
                    f" after hold={hold_n} tol={float(joint_tol):.3f}"
                    if (_is_to_safe_xyz_stage(stage_name) or _is_repairable_line_stage(stage_name)) else ""
                )
            )
    elif (_is_to_safe_xyz_stage(stage_name) or _is_repairable_line_stage(stage_name)) and ctx is not None and hold_n > 0:
        ctx.log(
            f"    [{stage_name}] joint target reached "
            f"q_track={qerr:.3f}rad hold={hold_n}"
        )
    return bool(q_reached)


def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _eef_split_joint_gap_fk_line_anchors(
    world,
    arm: str,
    q_start: np.ndarray,
    q_end: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    start_quat: np.ndarray,
    target_quat: np.ndarray,
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    mid_pos_tol: float,
    mid_ori_tol_deg: float,
    max_joint_gap_rad: float,
    intermediate_pos_only: bool,
    is_final_waypoint: bool,
) -> tuple[list[dict], Optional[dict]]:
    """Bridge a large adjacent-q gap only if dense FK stays on the Cartesian line."""
    from behavior_interface.skills.grasp import _set_arm_qpos_direct

    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    q_end = np.asarray(q_end, dtype=np.float64).reshape(7)
    gap = float(np.linalg.norm(q_end - q_start, ord=np.inf))
    n_split = max(2, int(math.ceil(gap / max(1e-6, float(max_joint_gap_rad) * 0.55))))
    out: list[dict] = []
    last_q = q_start.copy()
    for si in range(1, n_split + 1):
        alpha = float(si) / float(n_split)
        sub_t = float(t_start) + (float(t_end) - float(t_start)) * alpha
        sub_q = q_start + (q_end - q_start) * alpha
        sub_gap = float(np.linalg.norm(sub_q - last_q, ord=np.inf))
        sub_is_final = bool(is_final_waypoint and si == n_split)
        wp_pos = np.asarray(start_pos, dtype=np.float64).reshape(3) + (
            np.asarray(target_pos, dtype=np.float64).reshape(3)
            - np.asarray(start_pos, dtype=np.float64).reshape(3)
        ) * sub_t
        wp_quat = _quat_slerp_xyzw(start_quat, target_quat, sub_t)
        _set_arm_qpos_direct(world, arm, sub_q)
        pos_err, ori_err, _app_err = _eef_pose_err(world, arm, wp_pos, wp_quat)
        wp_pos_tol = float(pos_tol) if sub_is_final else float(mid_pos_tol)
        wp_ori_tol = float(ori_tol_deg) if sub_is_final else (
            180.0 if bool(intermediate_pos_only) else float(mid_ori_tol_deg)
        )
        check_ori = bool(sub_is_final or not intermediate_pos_only)
        if sub_gap > float(max_joint_gap_rad):
            return [], {
                "reason": "split_branch_gap",
                "t": round(sub_t, 4),
                "joint_gap_rad": round(sub_gap, 4),
                "limit_rad": round(float(max_joint_gap_rad), 4),
            }
        if pos_err > wp_pos_tol or (check_ori and ori_err > wp_ori_tol):
            return [], {
                "reason": "split_fk_line_error",
                "t": round(sub_t, 4),
                "pos_err_m": round(float(pos_err), 4),
                "pos_tol_m": round(float(wp_pos_tol), 4),
                "ori_err_deg": round(float(ori_err), 2),
                "ori_tol_deg": round(float(wp_ori_tol), 2),
                "check_ori": bool(check_ori),
            }
        out.append({
            "wp": None,
            "t": float(sub_t),
            "q": sub_q.copy(),
            "pos": wp_pos.copy(),
            "quat": wp_quat.copy(),
            "pos_err": float(pos_err),
            "ori_err": float(ori_err),
            "pos_tol": float(wp_pos_tol),
            "ori_tol": float(wp_ori_tol),
            "check_ori": bool(check_ori),
            "joint_gap": float(sub_gap),
            "is_final": bool(sub_is_final),
            "split_bridge": True,
        })
        last_q = sub_q.copy()
    ctx.log(
        f"    [{stage_name}] bridged joint gap {gap:.3f}rad with "
        f"{n_split} FK-checked sub-anchors"
    )
    return out, None


def _eef_plan_6d_joint_anchors(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    waypoint_pos_step_m: float = _EEF_WAYPOINT_POS_STEP_M,
    waypoint_ori_step_deg: float = _EEF_WAYPOINT_ORI_STEP_DEG,
    max_joint_gap_rad: float = 0.85,
    max_anchor_drop_ratio: float = 0.0,
    mid_pos_tol: Optional[float] = None,
    mid_ori_tol_deg: Optional[float] = None,
    intermediate_pos_only: bool = False,
    final_q: Optional[np.ndarray] = None,
    final_seed_q: Optional[np.ndarray] = None,
    bridge_pos_tol: Optional[float] = None,
    require_final_waypoint: bool = True,
    retain_validated_final_q_across_gap: bool = False,
) -> tuple[list[dict], dict]:
    """Offline solve Cartesian line waypoints and return joint anchors.

    The anchors are intended to keep the EEF on the Cartesian line.  Adjacent
    IK solutions with a large joint-space jump are treated as branch changes and
    are rejected so playback does not silently turn a Cartesian segment into a
    sweeping joint-space shortcut.
    """
    from behavior_interface.skills.grasp import _set_arm_qpos_direct

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    saved = None
    anchors: list[dict] = []
    dropped: list[dict] = []
    try:
        saved = world.robot.get_joint_positions().clone()
        eef0 = world.eef_pose(arm=arm)
        start_pos = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
        start_quat = _quat_normalize_xyzw(eef0["quat"])
        q_seed = _arm_qpos_now(world, arm)
        q_line_start = q_seed.copy()
        dist = float(np.linalg.norm(target_pos - start_pos))
        dot = abs(float(np.dot(start_quat, target_quat)))
        ang_deg = float(math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))))
        n_pos = int(math.ceil(dist / max(1e-6, float(waypoint_pos_step_m))))
        if bool(intermediate_pos_only):
            n_wp = max(1, min(24, n_pos))
        else:
            n_wp = max(1, min(24, max(
                n_pos,
                int(math.ceil(ang_deg / max(1e-6, float(waypoint_ori_step_deg)))),
            )))
        meta = {
            "start_pos": start_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "dist_m": dist,
            "rot_deg": ang_deg,
            "waypoints": n_wp,
            "dropped": dropped,
            "accepted": 0,
            "max_joint_gap_rad": float(max_joint_gap_rad),
            "intermediate_pos_only": bool(intermediate_pos_only),
            "uses_final_q": final_q is not None,
            "uses_final_seed_q": final_seed_q is not None,
            "require_final_waypoint": bool(require_final_waypoint),
            "retain_validated_final_q_across_gap": bool(
                retain_validated_final_q_across_gap
            ),
            "bypassed_branch_gaps": [],
        }
        mid_pos_tol_v = max(
            float(pos_tol),
            _EEF_WAYPOINT_MID_POS_TOL_M if mid_pos_tol is None else float(mid_pos_tol),
        )
        mid_ori_tol_v = max(
            float(ori_tol_deg),
            _EEF_WAYPOINT_MID_ORI_TOL_DEG if mid_ori_tol_deg is None else float(mid_ori_tol_deg),
        )
        meta["mid_pos_tol_m"] = float(mid_pos_tol_v)
        meta["mid_ori_tol_deg"] = float(mid_ori_tol_v)
        bridge_pos_tol_v = max(
            float(mid_pos_tol_v),
            _EEF_BRIDGE_FK_LINE_POS_TOL_M if bridge_pos_tol is None else float(bridge_pos_tol),
        )
        meta["bridge_pos_tol_m"] = float(bridge_pos_tol_v)
        ctx.log(
            f"    [{stage_name}] offline IK line waypoints={n_wp} "
            f"dist={dist*100:.1f}cm rot={ang_deg:.1f}°"
        )
        last_q = q_seed.copy()
        for wi in range(1, n_wp + 1):
            t = wi / n_wp
            wp_pos = start_pos + (target_pos - start_pos) * t
            wp_quat = _quat_slerp_xyzw(start_quat, target_quat, t)
            is_final = wi == n_wp
            wp_pos_tol = float(pos_tol) if is_final else mid_pos_tol_v
            wp_ori_tol = float(ori_tol_deg) if is_final else (
                180.0 if bool(intermediate_pos_only) else mid_ori_tol_v
            )
            if is_final and final_q is not None:
                q_probe = np.asarray(final_q, dtype=np.float64).reshape(7)
                _set_arm_qpos_direct(world, arm, q_probe)
                solve_pos_err, solve_ori_err, _app_err = _eef_pose_err(
                    world, arm, wp_pos, wp_quat
                )
                q_sol = q_probe.copy() if (
                    solve_pos_err <= wp_pos_tol and solve_ori_err <= wp_ori_tol
                ) else None
            else:
                seed_candidates: list[tuple[str, np.ndarray]] = [("prev", last_q)]
                q_nominal_seed = (
                    np.asarray(final_seed_q, dtype=np.float64).reshape(7)
                    if final_seed_q is not None
                    else None
                )
                if bool(intermediate_pos_only) and q_nominal_seed is not None:
                    q_interp_seed = (
                        (1.0 - float(t)) * q_line_start
                        + float(t) * q_nominal_seed
                    )
                    for label, seed in (
                        ("line_to_final", q_interp_seed),
                        ("final_branch", q_nominal_seed),
                    ):
                        if all(
                            float(np.linalg.norm(seed - old_seed, ord=np.inf)) > 1e-6
                            for _, old_seed in seed_candidates
                        ):
                            seed_candidates.append((label, seed))
                elif is_final and final_seed_q is not None:
                    q_final_seed = np.asarray(final_seed_q, dtype=np.float64).reshape(7)
                    if float(np.linalg.norm(q_final_seed - last_q, ord=np.inf)) > 1e-6:
                        seed_candidates.append(("final_branch", q_final_seed))
                q_sol = None
                solve_pos_err = float("inf")
                solve_ori_err = float("inf")
                solve_seed_label = "none"
                best_fail = (float("inf"), float("inf"), "none")
                best_q_score = float("inf")
                for seed_label, seed_q in seed_candidates:
                    has_nominal = bool(q_nominal_seed is not None)
                    nominal_weight = (
                        (
                            0.120 if is_final
                            else 0.050 + 0.080 * float(t)
                        )
                        if (intermediate_pos_only and has_nominal)
                        else 0.0
                    )
                    q_try, pos_try, ori_try = _eef_solve_6d_dls_arm_q(
                        world, arm, wp_pos, wp_quat,
                        pos_tol=wp_pos_tol,
                        ori_tol_deg=wp_ori_tol,
                        max_steps=240 if is_final else 160,
                        max_dq_per_step=0.085 if is_final else 0.095,
                        max_dx_per_step=0.030 if is_final else 0.040,
                        max_dw_per_step=0.08 if is_final else 0.10,
                        ori_weight=0.70 if is_final else (0.0 if bool(intermediate_pos_only) else 0.60),
                        lam=0.08,
                        seed_q=seed_q,
                        nominal_q=q_nominal_seed,
                        nominal_weight=nominal_weight,
                        nominal_tol_rad=float(max_joint_gap_rad) if is_final and has_nominal else None,
                        min_steps=8 if (intermediate_pos_only and has_nominal) else 0,
                    )
                    fail_score = float(pos_try) + 0.0025 * float(ori_try)
                    if fail_score < best_fail[0] + 0.0025 * best_fail[1]:
                        best_fail = (float(pos_try), float(ori_try), seed_label)
                    if q_try is None:
                        continue
                    gap_try = float(np.linalg.norm(q_try - last_q, ord=np.inf))
                    score_try = gap_try
                    if bool(intermediate_pos_only) and q_nominal_seed is not None:
                        target_gap = float(np.linalg.norm(q_try - q_nominal_seed, ord=np.inf))
                        score_try += (0.80 * (float(t) ** 1.7)) * target_gap
                        score_try += 0.010 * float(pos_try)
                        if gap_try > float(max_joint_gap_rad):
                            score_try += 10.0 + gap_try
                    if q_sol is None or score_try < best_q_score:
                        q_sol = q_try
                        solve_pos_err = float(pos_try)
                        solve_ori_err = float(ori_try)
                        solve_seed_label = seed_label
                        best_q_score = float(score_try)
                if q_sol is None:
                    solve_pos_err, solve_ori_err, solve_seed_label = best_fail
            if q_sol is None:
                dropped.append({
                    "wp": wi,
                    "t": round(float(t), 4),
                    "reason": "ik_fail",
                    "pos_err_m": round(float(solve_pos_err), 4),
                    "ori_err_deg": round(float(solve_ori_err), 2),
                    "seed": solve_seed_label if 'solve_seed_label' in locals() else None,
                })
                ctx.log(
                    f"    [{stage_name}] offline wp {wi}/{n_wp} drop ik_fail "
                    f"pos={solve_pos_err*1000:.0f}mm>{wp_pos_tol*1000:.0f}mm "
                    f"ori={solve_ori_err:.1f}°>{wp_ori_tol:.1f}°"
                    + (" final_q_fk" if is_final and final_q is not None else "")
                    + (f" seed={solve_seed_label}" if 'solve_seed_label' in locals() else "")
                )
                if is_final and bool(require_final_waypoint):
                    meta["accepted"] = len(anchors)
                    meta["error"] = "final_waypoint_ik_missing"
                    return [], meta
                _set_arm_qpos_direct(world, arm, last_q)
                continue
            gap = float(np.linalg.norm(q_sol - last_q, ord=np.inf))
            if math.isfinite(float(max_joint_gap_rad)) and gap > float(max_joint_gap_rad):
                prev_t = float(anchors[-1]["t"]) if anchors else 0.0
                bridge, bridge_err = _eef_split_joint_gap_fk_line_anchors(
                    world,
                    arm,
                    last_q,
                    q_sol,
                    t_start=prev_t,
                    t_end=float(t),
                    start_pos=start_pos,
                    target_pos=target_pos,
                    start_quat=start_quat,
                    target_quat=target_quat,
                    ctx=ctx,
                    stage_name=stage_name,
                    pos_tol=wp_pos_tol,
                    ori_tol_deg=wp_ori_tol,
                    mid_pos_tol=bridge_pos_tol_v,
                    mid_ori_tol_deg=mid_ori_tol_v,
                    max_joint_gap_rad=max_joint_gap_rad,
                    intermediate_pos_only=intermediate_pos_only,
                    is_final_waypoint=is_final,
                )
                if not bridge:
                    gap_report = {
                        "wp": wi,
                        "t": round(float(t), 4),
                        "reason": bridge_err.get("reason", "branch_gap") if bridge_err else "branch_gap",
                        "joint_gap_rad": round(float(gap), 4),
                        "limit_rad": round(float(max_joint_gap_rad), 4),
                        "is_final": bool(is_final),
                        "bridge_error": bridge_err,
                    }
                    retain_final_q = bool(
                        is_final
                        and final_q is not None
                        and retain_validated_final_q_across_gap
                    )
                    if retain_final_q:
                        meta["bypassed_branch_gaps"].append(gap_report)
                        anchors.append({
                            "wp": wi,
                            "t": float(t),
                            "q": q_sol.copy(),
                            "pos": wp_pos.copy(),
                            "quat": wp_quat.copy(),
                            "pos_err": float(solve_pos_err),
                            "ori_err": float(solve_ori_err),
                            "pos_tol": float(wp_pos_tol),
                            "ori_tol": float(wp_ori_tol),
                            "check_ori": True,
                            "joint_gap": gap,
                            "is_final": True,
                            "joint_interpolation_across_dropped_waypoints": True,
                            "bridge_error": bridge_err,
                        })
                        last_q = q_sol.copy()
                        _set_arm_qpos_direct(world, arm, last_q)
                        ctx.log(
                            f"    [{stage_name}] offline wp {wi}/{n_wp} keep "
                            "FK-validated final_q across deleted Cartesian samples "
                            f"gap={gap:.3f}rad>{float(max_joint_gap_rad):.3f}rad "
                            f"bridge={bridge_err}"
                        )
                        continue
                    dropped.append(gap_report)
                    ctx.log(
                        f"    [{stage_name}] offline wp {wi}/{n_wp} drop branch_gap "
                        f"gap={gap:.3f}rad>{float(max_joint_gap_rad):.3f}rad "
                        f"bridge={bridge_err}"
                    )
                    if is_final and bool(require_final_waypoint):
                        meta["accepted"] = len(anchors)
                        meta["error"] = "waypoint_branch_gap"
                        return [], meta
                    _set_arm_qpos_direct(world, arm, last_q)
                    continue
                for bi, sub_anchor in enumerate(bridge, start=1):
                    sub_anchor["wp"] = float(wi) - 1.0 + bi / max(1, len(bridge))
                    anchors.append(sub_anchor)
                last_q = q_sol.copy()
                _set_arm_qpos_direct(world, arm, last_q)
                ctx.log(
                    f"    [{stage_name}] offline wp {wi}/{n_wp} keep via bridge "
                    f"pos={solve_pos_err*1000:.0f}mm ori={solve_ori_err:.1f}° gap={gap:.3f}rad"
                )
                continue
            anchors.append({
                "wp": wi,
                "t": float(t),
                "q": q_sol.copy(),
                "pos": wp_pos.copy(),
                "quat": wp_quat.copy(),
                "pos_err": float(solve_pos_err),
                "ori_err": float(solve_ori_err),
                "pos_tol": float(wp_pos_tol),
                "ori_tol": float(wp_ori_tol),
                "check_ori": bool(is_final or not intermediate_pos_only),
                "joint_gap": gap,
                "is_final": bool(is_final),
            })
            last_q = q_sol.copy()
            _set_arm_qpos_direct(world, arm, last_q)
            ctx.log(
                f"    [{stage_name}] offline wp {wi}/{n_wp} keep "
                f"pos={solve_pos_err*1000:.0f}mm ori={solve_ori_err:.1f}° gap={gap:.3f}rad"
                + (
                    f" seed={solve_seed_label}"
                    if 'solve_seed_label' in locals() and is_final
                    else ""
                )
            )
        meta["accepted"] = len(anchors)
        meta["final_waypoint_accepted"] = bool(
            anchors and anchors[-1].get("is_final")
        )
        if bool(require_final_waypoint) and not meta["final_waypoint_accepted"]:
            meta["error"] = "final_waypoint_ik_missing"
            return [], meta
        drop_ratio = len(dropped) / max(1, n_wp)
        if drop_ratio > float(max_anchor_drop_ratio):
            meta["error"] = "too_many_waypoints_dropped"
            return [], meta
        return anchors, meta
    finally:
        if saved is not None:
            try:
                world.robot.set_joint_positions(saved)
            except Exception:
                pass


def _eef_plan_move_only_joint_anchors(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    final_q: Optional[np.ndarray],
    ctx,
    stage_name: str = "move_only_line_offline",
    reset_tool_roll_at_start: bool = False,
    back_m: float = 0.10,
) -> tuple[list[dict], dict]:
    """Build the exact same-branch Cartesian anchors used by move_only exec."""
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    robot = world.robot
    saved_q = None
    saved_vel = None
    tool_pin_before = None
    tool_motion_before = None
    tool_roll_reset = {
        "requested": bool(reset_tool_roll_at_start),
        "applied": False,
    }
    start_arm_q = None
    try:
        if bool(reset_tool_roll_at_start):
            try:
                if world.has_tool_roll(arm):
                    tool_pin_before = float(world.tool_roll_pin_qpos(arm))
                    tool_motion_before = set(world._tool_roll_motion_enabled)
                    before_rad = float(world.tool_roll_qpos(arm))
                    if abs(before_rad) > _TOOL_ROLL_RESET_TOL_RAD:
                        return [], {
                            "error": "tool_roll_not_reset_before_offline_plan",
                            "tool_roll_reset": {
                                **tool_roll_reset,
                                "before_rad": before_rad,
                                "pin_before_rad": tool_pin_before,
                                "required_action": (
                                    "run action-only J8 reset before building "
                                    "offline anchors"
                                ),
                            },
                        }
                    world.end_tool_roll_motion(arm)
                    world.set_tool_roll_pin_qpos(arm, 0.0)
                    tool_roll_reset.update({
                        "applied": False,
                        "verified": True,
                        "action_only": True,
                        "before_rad": before_rad,
                        "pin_before_rad": tool_pin_before,
                        "after_rad": float(world.tool_roll_qpos(arm)),
                    })
            except Exception as exc:
                return [], {
                    "error": "tool_roll_reset_failed",
                    "tool_roll_reset": {
                        **tool_roll_reset,
                        "error_detail": f"{type(exc).__name__}: {exc}",
                    },
                }

        start_arm_q = _arm_qpos_now(world, arm)
        anchors, meta = _eef_plan_6d_joint_anchors(
            world,
            arm,
            target_pos,
            target_quat,
            ctx=ctx,
            stage_name=stage_name,
            pos_tol=_EEF_FINAL_POS_TOL_M,
            ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
            waypoint_pos_step_m=0.035,
            waypoint_ori_step_deg=12.0,
            max_joint_gap_rad=0.55,
            intermediate_pos_only=False,
            final_q=(
                None
                if final_q is None
                else np.asarray(final_q, dtype=np.float64).reshape(7)
            ),
            mid_pos_tol=0.012,
            mid_ori_tol_deg=6.0,
        )
        meta["planner"] = "exec_plan_pose_move_only_same_branch_6d"
        meta["back_m_requested_m"] = float(back_m)
        meta["back_m_used"] = False
        meta["tool_roll_reset"] = tool_roll_reset
        meta["start_arm_q"] = np.asarray(
            start_arm_q, dtype=np.float64
        ).reshape(7).tolist()
        return anchors, meta
    finally:
        if saved_q is not None:
            try:
                robot.set_joint_positions(saved_q)
            except Exception:
                pass
        if saved_vel is not None:
            try:
                robot.set_joint_velocities(saved_vel)
            except Exception:
                pass
        if tool_pin_before is not None:
            try:
                world.set_tool_roll_pin_qpos(arm, tool_pin_before)
            except Exception:
                pass
        if tool_motion_before is not None:
            try:
                world._tool_roll_motion_enabled.clear()
                world._tool_roll_motion_enabled.update(tool_motion_before)
            except Exception:
                pass


def _eef_plan_final_q_fk_line_anchors(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    final_q: np.ndarray,
    waypoint_pos_step_m: float = _EEF_WAYPOINT_POS_STEP_M,
    max_joint_gap_rad: float = 0.85,
    mid_pos_tol: Optional[float] = None,
) -> tuple[list[dict], dict]:
    """Use the known final branch, but keep only paths whose FK tracks XYZ line."""
    from behavior_interface.skills.grasp import _set_arm_qpos_direct

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    q_final = np.asarray(final_q, dtype=np.float64).reshape(7)
    saved = None
    anchors: list[dict] = []
    dropped: list[dict] = []
    try:
        saved = world.robot.get_joint_positions().clone()
        eef0 = world.eef_pose(arm=arm)
        start_pos = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
        start_quat = _quat_normalize_xyzw(eef0["quat"])
        q0 = _arm_qpos_now(world, arm)
        dist = float(np.linalg.norm(target_pos - start_pos))
        max_delta = float(np.linalg.norm(q_final - q0, ord=np.inf))
        n_wp = max(2, min(40, max(
            int(math.ceil(dist / max(1e-6, float(waypoint_pos_step_m)))),
            int(math.ceil(max_delta / max(1e-6, float(max_joint_gap_rad) * 0.70))),
        )))
        mid_pos_tol_v = max(
            float(pos_tol),
            _EEF_WAYPOINT_MID_POS_TOL_M if mid_pos_tol is None else float(mid_pos_tol),
        )
        meta = {
            "fallback": "final_q_fk_line",
            "start_pos": start_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "dist_m": dist,
            "max_delta_rad": max_delta,
            "waypoints": n_wp,
            "accepted": 0,
            "dropped": dropped,
            "max_joint_gap_rad": float(max_joint_gap_rad),
            "mid_pos_tol_m": float(mid_pos_tol_v),
        }
        ctx.log(
            f"    [{stage_name}] final-q FK line validation waypoints={n_wp} "
            f"dist={dist*100:.1f}cm max_delta={max_delta:.3f}rad"
        )
        last_q = q0.copy()
        for wi in range(1, n_wp + 1):
            t = wi / n_wp
            wp_pos = start_pos + (target_pos - start_pos) * t
            wp_quat = _quat_slerp_xyzw(start_quat, target_quat, t)
            q = q0 + (q_final - q0) * float(t)
            gap = float(np.linalg.norm(q - last_q, ord=np.inf))
            if gap > float(max_joint_gap_rad):
                dropped.append({
                    "wp": wi,
                    "reason": "branch_gap",
                    "joint_gap_rad": round(gap, 4),
                })
                ctx.log(
                    f"    [{stage_name}] final-q FK wp {wi}/{n_wp} drop "
                    f"gap={gap:.3f}rad>{float(max_joint_gap_rad):.3f}rad"
                )
                meta["accepted"] = len(anchors)
                meta["error"] = "qline_branch_gap"
                return [], meta
            _set_arm_qpos_direct(world, arm, q)
            pos_err, ori_err, _app_err = _eef_pose_err(
                world, arm, wp_pos, wp_quat
            )
            is_final = wi == n_wp
            wp_pos_tol = float(pos_tol) if is_final else mid_pos_tol_v
            wp_ori_tol = float(ori_tol_deg) if is_final else 180.0
            if pos_err > wp_pos_tol or (is_final and ori_err > wp_ori_tol):
                dropped.append({
                    "wp": wi,
                    "reason": "fk_line_error",
                    "pos_err_m": round(float(pos_err), 4),
                    "ori_err_deg": round(float(ori_err), 2),
                    "is_final": bool(is_final),
                })
                ctx.log(
                    f"    [{stage_name}] final-q FK wp {wi}/{n_wp} drop "
                    f"pos={pos_err*1000:.0f}mm>{wp_pos_tol*1000:.0f}mm "
                    + (
                        f"ori={ori_err:.1f}°>{wp_ori_tol:.1f}°"
                        if is_final else
                        f"ori={ori_err:.1f}° (mid XYZ-only)"
                    )
                )
                meta["accepted"] = len(anchors)
                meta["error"] = "qline_fk_line_error"
                return [], meta
            anchors.append({
                "wp": wi,
                "t": float(t),
                "q": q.copy(),
                "pos": wp_pos.copy(),
                "quat": wp_quat.copy(),
                "pos_err": float(pos_err),
                "ori_err": float(ori_err),
                "pos_tol": float(wp_pos_tol),
                "ori_tol": float(wp_ori_tol),
                "check_ori": bool(is_final),
                "joint_gap": gap,
                "is_final": bool(is_final),
            })
            last_q = q.copy()
            ctx.log(
                f"    [{stage_name}] final-q FK wp {wi}/{n_wp} keep "
                f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° gap={gap:.3f}rad"
            )
        meta["accepted"] = len(anchors)
        return anchors, meta
    finally:
        if saved is not None:
            try:
                world.robot.set_joint_positions(saved)
            except Exception:
                pass


def _eef_play_joint_anchor_path(
    world,
    arm: str,
    anchors: list[dict],
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    target_pos,
    target_quat,
    pos_tol: float,
    ori_tol_deg: float,
    max_frames_per_anchor: int,
    max_line_deviation_m: float = _EEF_WAYPOINT_MAX_LINE_DEV_M,
    repair_line_waypoints: bool = False,
    validate_intermediate_anchors: bool = True,
    validate_final_anchor: bool = True,
) -> tuple[float, float]:
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    start_pose = world.eef_pose(arm=arm)
    start_pos = np.asarray(start_pose["pos"], dtype=np.float64).reshape(3)
    max_dev = 0.0
    for ai, anchor in enumerate(anchors, start=1):
        q_target = np.asarray(anchor["q"], dtype=np.float64).reshape(7)
        joint_bridge = bool(
            anchor.get("joint_interpolation_across_dropped_waypoints")
        )
        is_final_anchor = bool(anchor.get("is_final"))
        enforce_anchor_validation = bool(
            validate_intermediate_anchors
            or (is_final_anchor and validate_final_anchor)
        )
        disable_line_guard = bool(
            joint_bridge or not validate_intermediate_anchors
        )
        stage_s = str(stage_name or "")
        final_translate_stage = stage_s.startswith("eef_pose_final_q_translate")
        move_only_line_stage = stage_s.startswith("move_only_line")
        line_stage = (
            stage_s.startswith("to_safe")
            or final_translate_stage
            or move_only_line_stage
        )
        to_safe_xyz_stage = stage_s.startswith("to_safe_xyz_joint")
        play_line_limit = float(max_line_deviation_m)
        if bool(repair_line_waypoints) and (final_translate_stage or move_only_line_stage):
            play_line_limit += 0.020
        if joint_bridge:
            ctx.log(
                f"    [{stage_name}] anchor {ai}/{len(anchors)} deleted Cartesian "
                "samples are bridged by bounded-step joint interpolation to the "
                f"FK-validated final q, gap={float(anchor.get('joint_gap', 0.0)):.3f}rad"
            )
        played_ok = yield from _play_arm_q_target(
            world, arm, q_target,
            gripper_cmd=gripper_cmd,
            max_frames=max_frames_per_anchor,
            hold_frames=(
                _TO_SAFE_WAYPOINT_HOLD_FRAMES
                if to_safe_xyz_stage
                else (
                    (
                        _FINAL_STORED_Q_HOLD_FRAMES
                        if final_translate_stage and bool(anchor.get("is_final"))
                        else 2
                    )
                    if (final_translate_stage or move_only_line_stage)
                    else (0 if line_stage else (2 if anchor.get("is_final") else 1))
                )
            ),
            min_hold_frames=(
                _FINAL_STORED_Q_MIN_HOLD_FRAMES
                if final_translate_stage and bool(anchor.get("is_final"))
                else 0
            ),
            joint_tol=(
                _TO_SAFE_WAYPOINT_JOINT_TOL_RAD
                if to_safe_xyz_stage else 0.010
            ),
            ctx=ctx,
            stage_name=f"{stage_name}:anchor{ai}/{len(anchors)}",
            line_start_pos=None if disable_line_guard else start_pos,
            line_target_pos=None if disable_line_guard else target_pos,
            max_line_deviation_m=None if disable_line_guard else play_line_limit,
            allow_direct_residual_correction=False,
        )
        if not bool(played_ok):
            ctx.log(
                f"    [{stage_name}] offline anchor {ai}/{len(anchors)} "
                "controller residual is diagnostic-only; "
                "continue the scheduled joint trajectory"
            )
        pos_err, ori_err, app_err = _eef_pose_err(world, arm, anchor["pos"], anchor["quat"])
        eef_now = world.eef_pose(arm=arm)
        dev = _dist_point_to_segment(
            np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
            start_pos,
            target_pos,
        )
        max_dev = max(max_dev, float(dev))
        ctx.log(
            f"    [{stage_name}] anchor {ai}/{len(anchors)} actual "
            f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° approach={app_err:.1f}° "
            f"line_dev={dev*1000:.0f}mm"
        )
        line_limit = float(max_line_deviation_m)
        anchor_pos_tol = float(anchor.get("pos_tol", pos_tol))
        anchor_ori_tol = float(anchor.get("ori_tol", ori_tol_deg))
        check_ori = bool(anchor.get("check_ori", True))
        line_failed = bool(
            enforce_anchor_validation
            and not joint_bridge
            and dev > line_limit
        )
        pos_failed = bool(
            enforce_anchor_validation and pos_err > anchor_pos_tol
        )
        ori_failed = bool(
            enforce_anchor_validation
            and check_ori
            and ori_err > anchor_ori_tol
        )
        if not enforce_anchor_validation:
            if (
                dev > line_limit
                or pos_err > anchor_pos_tol
                or (check_ori and ori_err > anchor_ori_tol)
            ):
                ctx.log(
                    f"    [{stage_name}] offline joint anchor "
                    f"{ai}/{len(anchors)} runtime deviation is diagnostic-only; "
                    "continue to the next offline-valid joint anchor"
                )
            continue
        if line_failed or pos_failed or ori_failed:
            can_repair = (
                bool(repair_line_waypoints)
                and (final_translate_stage or move_only_line_stage)
            )
            if can_repair:
                reason_bits = []
                if line_failed:
                    reason_bits.append(f"line_dev={dev*1000:.0f}>{line_limit*1000:.0f}mm")
                if pos_failed:
                    reason_bits.append(f"pos={pos_err*1000:.0f}>{anchor_pos_tol*1000:.0f}mm")
                if ori_failed:
                    reason_bits.append(f"ori={ori_err:.1f}>{anchor_ori_tol:.1f}°")
                repair_tol = min(anchor_pos_tol, max(float(pos_tol), 0.006))
                repair_ori_tol = anchor_ori_tol if check_ori else max(float(ori_tol_deg), 12.0)
                ctx.log(
                    f"    [{stage_name}] repair anchor {ai}/{len(anchors)} back to line waypoint "
                    f"({'; '.join(reason_bits)}) tol={repair_tol*1000:.0f}mm/"
                    f"{repair_ori_tol:.1f}°"
                )
                final_stored_q_anchor = (
                    final_translate_stage and bool(anchor.get("is_final"))
                )
                if final_stored_q_anchor:
                    ctx.log(
                        f"    [{stage_name}] final anchor is the plan-time "
                        "OG-FK-validated q; settling the exact q again instead "
                        "of running Cartesian DLS repair"
                    )
                    retry_ok = yield from _play_arm_q_target(
                        world,
                        arm,
                        q_target,
                        gripper_cmd=gripper_cmd,
                        max_frames=max(
                            8, min(int(max_frames_per_anchor), 18)
                        ),
                        hold_frames=_FINAL_STORED_Q_RETRY_HOLD_FRAMES,
                        min_hold_frames=_FINAL_STORED_Q_RETRY_MIN_HOLD_FRAMES,
                        joint_tol=0.010,
                        ctx=ctx,
                        stage_name=(
                            f"{stage_name}:anchor{ai}/{len(anchors)}"
                            "_final_q_settle"
                        ),
                        line_start_pos=None if disable_line_guard else start_pos,
                        line_target_pos=None if disable_line_guard else target_pos,
                        max_line_deviation_m=None if disable_line_guard else play_line_limit,
                        allow_direct_residual_correction=False,
                    )
                    if retry_ok is False:
                        return float("inf"), float("inf")
                elif move_only_line_stage:
                    yield from _eef_snap_6d_line_anchor(
                        world,
                        arm,
                        anchor["pos"],
                        anchor["quat"],
                        gripper_cmd=gripper_cmd,
                        ctx=ctx,
                        stage_name=f"{stage_name}:anchor{ai}/{len(anchors)}_line_snap",
                        pos_tol=repair_tol,
                        ori_tol_deg=repair_ori_tol,
                        line_start_pos=start_pos,
                        line_target_pos=target_pos,
                        max_line_deviation_m=max_line_deviation_m,
                        line_deviation_margin_m=0.020,
                        nominal_q=q_target,
                        max_q_delta_rad=0.45,
                    )
                else:
                    _repair_pos_err, _repair_ori_err = yield from _eef_goto_6d_line_repair_dls(
                        world,
                        arm,
                        anchor["pos"],
                        anchor["quat"],
                        gripper_cmd=gripper_cmd,
                        ctx=ctx,
                        stage_name=f"{stage_name}:anchor{ai}/{len(anchors)}_line_repair",
                        pos_tol=repair_tol,
                        ori_tol_deg=repair_ori_tol,
                        max_steps=65,
                        max_dq_per_step=0.035,
                        max_dx_per_step=0.012,
                        max_dw_per_step=0.075,
                        ori_weight=0.65,
                        lam=0.11,
                        line_start_pos=start_pos,
                        line_target_pos=target_pos,
                        max_line_deviation_m=max_line_deviation_m,
                        line_deviation_margin_m=0.020,
                    )
                pos_err, ori_err, app_err = _eef_pose_err(world, arm, anchor["pos"], anchor["quat"])
                eef_now = world.eef_pose(arm=arm)
                dev = _dist_point_to_segment(
                    np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                    start_pos,
                    target_pos,
                )
                max_dev = max(max_dev, float(dev))
                line_failed = (not joint_bridge) and dev > line_limit
                pos_failed = pos_err > anchor_pos_tol
                ori_failed = check_ori and ori_err > anchor_ori_tol
                ctx.log(
                    f"    [{stage_name}] repair anchor {ai}/{len(anchors)} result "
                    f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° approach={app_err:.1f}° "
                    f"line_dev={dev*1000:.0f}mm"
                )
                if not (line_failed or pos_failed or ori_failed):
                    continue
            if line_failed:
                ctx.log(
                    f"    [{stage_name}] ABORT joint path line deviation "
                    f"{dev*1000:.0f}mm>{line_limit*1000:.0f}mm"
                )
                return float("inf"), float("inf")
            fail_bits = []
            if pos_failed:
                fail_bits.append(f"pos={pos_err*1000:.0f}mm>{anchor_pos_tol*1000:.0f}mm")
            if ori_failed:
                fail_bits.append(f"ori={ori_err:.1f}°>{anchor_ori_tol:.1f}°")
            if not fail_bits:
                fail_bits.append("unknown")
            ctx.log(
                f"    [{stage_name}] ABORT anchor {ai}/{len(anchors)} 未到直线 waypoint："
                f"{'; '.join(fail_bits)} "
                + (
                    ""
                    if check_ori else
                    f"ori={ori_err:.1f}° (mid waypoint XYZ-only)"
                )
            )
            return float("inf"), float("inf")
    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
        world, arm, target_pos, target_quat
    )
    ctx.log(
        f"    [{stage_name}] joint-anchor final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}° "
        f"max_line_dev={max_dev*1000:.0f}mm"
    )
    return float(final_pos_err), float(final_ori_err)


def _eef_goto_6d_kinematic_plan(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_frames_per_waypoint: int = 36,
    max_refine_attempts: int = 3,
    intermediate_pos_only: bool = False,
    final_q: Optional[np.ndarray] = None,
    final_seed_q: Optional[np.ndarray] = None,
    waypoint_pos_step_m: float = _EEF_WAYPOINT_POS_STEP_M,
    waypoint_ori_step_deg: float = _EEF_WAYPOINT_ORI_STEP_DEG,
    max_joint_gap_rad: float = 0.85,
    max_line_deviation_m: float = _EEF_WAYPOINT_MAX_LINE_DEV_M,
    repair_line_waypoints: bool = False,
    mid_pos_tol: Optional[float] = None,
    mid_ori_tol_deg: Optional[float] = None,
) -> tuple[float, float]:
    """Plan 6DoF Cartesian waypoints via kinematic DLS, then play arm q targets."""
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    anchors, anchor_meta = _eef_plan_6d_joint_anchors(
        world, arm, target_pos, target_quat,
        ctx=ctx,
        stage_name=f"{stage_name}_offline",
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        waypoint_pos_step_m=waypoint_pos_step_m,
        waypoint_ori_step_deg=waypoint_ori_step_deg,
        max_joint_gap_rad=max_joint_gap_rad,
        mid_pos_tol=mid_pos_tol,
        mid_ori_tol_deg=mid_ori_tol_deg,
        intermediate_pos_only=intermediate_pos_only,
        final_q=final_q,
        final_seed_q=final_seed_q,
    )
    if anchors:
        return (yield from _eef_play_joint_anchor_path(
            world, arm, anchors,
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name=f"{stage_name}_joint",
            target_pos=target_pos,
            target_quat=target_quat,
            pos_tol=pos_tol,
            ori_tol_deg=ori_tol_deg,
            max_frames_per_anchor=max_frames_per_waypoint,
            max_line_deviation_m=max_line_deviation_m,
            repair_line_waypoints=repair_line_waypoints,
        ))
    if bool(intermediate_pos_only) and final_q is not None:
        anchors, qline_meta = _eef_plan_final_q_fk_line_anchors(
            world, arm, target_pos, target_quat,
            ctx=ctx,
            stage_name=f"{stage_name}_qline",
            pos_tol=pos_tol,
            ori_tol_deg=ori_tol_deg,
            final_q=final_q,
            max_joint_gap_rad=0.85,
        )
        if anchors:
            ctx.log(
                f"    [{stage_name}] using final-q FK line fallback "
                f"accepted={qline_meta.get('accepted')}"
            )
            return (yield from _eef_play_joint_anchor_path(
                world, arm, anchors,
                gripper_cmd=gripper_cmd,
                ctx=ctx,
                stage_name=f"{stage_name}_qline_joint",
                target_pos=target_pos,
                target_quat=target_quat,
                pos_tol=pos_tol,
                ori_tol_deg=ori_tol_deg,
                max_frames_per_anchor=max_frames_per_waypoint,
            ))
        ctx.log(
            f"    [{stage_name}] final-q FK line fallback failed: "
            f"{qline_meta.get('error', 'unknown')} "
            f"accepted={qline_meta.get('accepted')} "
            f"dropped={len(qline_meta.get('dropped') or [])}"
        )
    ctx.log(
        f"    [{stage_name}] offline joint-anchor plan failed: "
        f"{anchor_meta.get('error', 'unknown')} "
        f"accepted={anchor_meta.get('accepted')} dropped={len(anchor_meta.get('dropped') or [])}; "
        "no curved fallback"
    )
    return float("inf"), float("inf")
    eef0 = world.eef_pose(arm=arm)
    start_pos = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    start_quat = _quat_normalize_xyzw(eef0["quat"])
    dist = float(np.linalg.norm(target_pos - start_pos))
    dot = abs(float(np.dot(start_quat, target_quat)))
    ang_deg = float(math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))))
    n_wp = max(1, min(6, max(
        int(math.ceil(dist / 0.080)),
        int(math.ceil(ang_deg / 28.0)),
    )))
    ctx.log(
        f"    [{stage_name}] kinematic 6d plan waypoints={n_wp} "
        f"dist={dist*100:.1f}cm rot={ang_deg:.1f}°"
    )
    final_pos_err = float("inf")
    final_ori_err = float("inf")
    for wi in range(1, n_wp + 1):
        t = wi / n_wp
        wp_pos = start_pos + (target_pos - start_pos) * t
        wp_quat = _quat_slerp_xyzw(start_quat, target_quat, t)
        wp_pos_tol = float(pos_tol) if wi == n_wp else max(float(pos_tol), 0.060)
        wp_ori_tol = float(ori_tol_deg) if wi == n_wp else max(float(ori_tol_deg), 35.0)
        attempts = max(1, int(max_refine_attempts if wi == n_wp else min(2, max_refine_attempts)))
        for ai in range(1, attempts + 1):
            q_target, solve_pos_err, solve_ori_err = _eef_solve_6d_dls_arm_q(
                world, arm, wp_pos, wp_quat,
                pos_tol=wp_pos_tol,
                ori_tol_deg=wp_ori_tol,
                max_steps=260 if wi == n_wp else 170,
                max_dq_per_step=0.055 if wi == n_wp else 0.065,
                max_dx_per_step=0.018 if wi == n_wp else 0.025,
                max_dw_per_step=0.08 if wi == n_wp else 0.12,
                ori_weight=0.65 if wi == n_wp else 0.50,
                lam=0.09,
            )
            retry_tag = f" retry={ai}/{attempts}" if attempts > 1 else ""
            ctx.log(
                f"    [{stage_name}] solve wp {wi}/{n_wp}{retry_tag} "
                f"pos={solve_pos_err*1000:.0f}mm ori={solve_ori_err:.1f}° "
                f"{'ok' if q_target is not None else 'fail'}"
            )
            if q_target is None:
                ctx.log(
                    f"    [{stage_name}] no executable IK for wp {wi}/{n_wp}: "
                    f"best pos={solve_pos_err*1000:.0f}mm ori={solve_ori_err:.1f}°"
                )
                return float(solve_pos_err), float(solve_ori_err)
            yield from _play_arm_q_target(
                world, arm, q_target,
                gripper_cmd=gripper_cmd,
                max_frames=max_frames_per_waypoint,
                hold_frames=6 if wi == n_wp else 3,
                ctx=ctx,
                stage_name=f"{stage_name}:wp{wi}/{n_wp}:retry{ai}",
            )
            final_pos_err, final_ori_err, app_err = _eef_pose_err(world, arm, wp_pos, wp_quat)
            q_track = float(np.linalg.norm(_arm_qpos_now(world, arm) - q_target, ord=np.inf))
            ctx.log(
                f"    [{stage_name}] actual wp {wi}/{n_wp}{retry_tag} "
                f"pos={final_pos_err*1000:.0f}mm ori={final_ori_err:.1f}° "
                f"approach={app_err:.1f}° q_track={q_track:.3f}rad"
            )
            if final_pos_err <= wp_pos_tol and final_ori_err <= wp_ori_tol:
                break
            if q_track <= 0.012:
                ctx.log(
                    f"    [{stage_name}] wp {wi}/{n_wp} reached joint target but "
                    f"6DoF still off; treat as constrained IK failure"
                )
                return float(final_pos_err), float(final_ori_err)
            if ai < attempts:
                ctx.log(
                    f"    [{stage_name}] refine wp {wi}/{n_wp}: "
                    f"pos={final_pos_err*1000:.0f}mm>{wp_pos_tol*1000:.0f}mm "
                    f"ori={final_ori_err:.1f}°>{wp_ori_tol:.1f}°，按当前真实姿态重算"
                )
        if final_pos_err > wp_pos_tol or final_ori_err > wp_ori_tol:
            return float(final_pos_err), float(final_ori_err)

    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(world, arm, target_pos, target_quat)
    ctx.log(
        f"    [{stage_name}] kinematic final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}°"
    )
    return float(final_pos_err), float(final_ori_err)


def _eef_play_stored_filter_q(
    world,
    arm: str,
    q_arm,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_frames: int = 140,
) -> tuple[float, float]:
    """Replay a plan-time OG-FK-validated IK q and verify the real EEF pose."""
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    try:
        q_target = np.asarray(q_arm, dtype=np.float64).reshape(7)
    except Exception:
        return float("inf"), float("inf")
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    ctx.log(
        f"    [{stage_name}] replay stored filter IK q arm={arm} "
        f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
    )
    q0 = _arm_qpos_now(world, arm)
    max_delta = float(np.linalg.norm(q_target - q0, ord=np.inf))
    n_frames = max(8, min(int(max_frames), int(math.ceil(max_delta / 0.045)) + 10))
    ctx.log(
        f"    [{stage_name}] play stored q frames={n_frames} "
        f"max_delta={max_delta:.3f}rad"
    )
    for fi in range(1, n_frames + 1):
        s = fi / n_frames
        ss = s * s * (3.0 - 2.0 * s)
        q = q0 + (q_target - q0) * float(ss)
        yield _make_arm_q_action(world, arm, q.tolist(), gripper_cmd)
    settle_frames = max(24, min(180, int(math.ceil(max_delta / 0.025)) + 30))
    q_track = float("inf")
    q_list = [float(x) for x in q_target.tolist()]
    for _ in range(settle_frames):
        q_track = float(np.linalg.norm(_arm_qpos_now(world, arm) - q_target, ord=np.inf))
        if q_track <= 0.018:
            break
        yield _make_arm_q_action(world, arm, q_list, gripper_cmd)
    if q_track > 0.018:
        ctx.log(
            f"    [{stage_name}] stored q settle incomplete "
            f"q_track={q_track:.3f}rad after {settle_frames} hold frames"
        )
    pos_err, ori_err, app_err = _eef_pose_err(world, arm, target_pos, target_quat)
    ctx.log(
        f"    [{stage_name}] stored-q actual "
        f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° approach={app_err:.1f}°"
    )
    if pos_err > float(pos_tol) or ori_err > float(ori_tol_deg):
        return float(pos_err), float(ori_err)
    return float(pos_err), float(ori_err)


def _eef_align_safe_pose_current_branch(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
) -> tuple[float, float]:
    """Loose 6D safe-pose alignment solved from the current safe-near branch."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    q_seed = _arm_qpos_now(world, arm)
    q_target, solve_pos_err, solve_ori_err = _eef_solve_6d_dls_arm_q(
        world,
        arm,
        target_pos,
        target_quat,
        pos_tol=_EEF_SAFE_POS_TOL_M,
        ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
        max_steps=180,
        max_dq_per_step=0.080,
        max_dx_per_step=0.030,
        max_dw_per_step=0.16,
        ori_weight=0.55,
        lam=0.09,
        seed_q=q_seed,
        nominal_q=q_seed,
        nominal_weight=0.010,
        nominal_tol_rad=0.95,
    )
    if q_target is None:
        ctx.log(
            f"    [{stage_name}] no current-branch safe 6D q "
            f"solve_pos={solve_pos_err*1000:.0f}mm solve_ori={solve_ori_err:.1f}°"
        )
        return float(solve_pos_err), float(solve_ori_err)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    max_delta = float(np.linalg.norm(q_target - q_seed, ord=np.inf))
    ctx.log(
        f"    [{stage_name}] current-branch safe 6D q "
        f"solve_pos={solve_pos_err*1000:.0f}mm solve_ori={solve_ori_err:.1f}° "
        f"max_delta={max_delta:.3f}rad"
    )
    yield from _play_arm_q_target(
        world,
        arm,
        q_target,
        gripper_cmd=gripper_cmd,
        max_frames=max(8, min(28, int(math.ceil(max_delta / 0.10)) + 6)),
        hold_frames=8,
        joint_tol=0.035,
        ctx=ctx,
        stage_name=stage_name,
    )
    pos_err, ori_err, app_err = _eef_pose_err(world, arm, target_pos, target_quat)
    ctx.log(
        f"    [{stage_name}] actual safe "
        f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}° approach={app_err:.1f}°"
    )
    return float(pos_err), float(ori_err)


def _eef_play_stored_final_q_translation(
    world,
    arm: str,
    q_arm,
    target_pos,
    target_quat,
    *,
    gripper_cmd: Optional[float],
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_line_deviation_m: float = 0.030,
    max_frames: int = 80,
) -> tuple[float, float]:
    """Play safe→final as Cartesian translation ending at plan-filter final q.

    The final grasp q was already OG-FK-validated by plan_filter.  This segment
    therefore solves only intermediate XYZ waypoints on the same branch and
    forces the last anchor to the stored final q; it never runs a new final IK.
    """
    try:
        q_target = np.asarray(q_arm, dtype=np.float64).reshape(7)
    except Exception:
        return float("inf"), float("inf")
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    start_pose = world.eef_pose(arm=arm)
    start_pos = np.asarray(start_pose["pos"], dtype=np.float64).reshape(3)
    ctx.log(
        f"    [{stage_name}] safe→final same-quat translation: "
        f"midpoints XYZ-only, final 6D "
        f"dist={float(np.linalg.norm(target_pos - start_pos))*100:.1f}cm "
        f"line_tol={float(max_line_deviation_m)*1000:.0f}mm"
    )
    probe_pos, probe_ori, probe_app, probe_pose = _probe_arm_q_pose_err(
        world, arm, q_target, target_pos, target_quat
    )
    ctx.log(
        f"    [{stage_name}] stored final-q current-state FK probe "
        f"pos={probe_pos*1000:.1f}mm ori={probe_ori:.2f}° "
        f"approach={probe_app:.2f}° "
        f"eef=({probe_pose['pos'][0]:.3f},{probe_pose['pos'][1]:.3f},{probe_pose['pos'][2]:.3f})"
    )
    if probe_pos > float(pos_tol) or probe_ori > float(ori_tol_deg):
        q_re, re_pos, re_ori = _eef_solve_6d_dls_arm_q(
            world,
            arm,
            target_pos,
            target_quat,
            pos_tol=float(pos_tol),
            ori_tol_deg=float(ori_tol_deg),
            max_steps=260,
            max_dq_per_step=0.055,
            max_dx_per_step=0.018,
            max_dw_per_step=0.090,
            ori_weight=0.65,
            lam=0.070,
            seed_q=q_target,
            nominal_q=q_target,
            nominal_weight=0.010,
            nominal_tol_rad=0.85,
            min_steps=10,
        )
        if q_re is None or re_pos > float(pos_tol) or re_ori > float(ori_tol_deg):
            ctx.log(
                f"    [{stage_name}] current-state final 6D re-solve failed "
                f"pos={re_pos*1000:.1f}mm ori={re_ori:.2f}°; "
                "not playing approximate final"
            )
            return float("inf"), float("inf")
        q_target = np.asarray(q_re, dtype=np.float64).reshape(7)
        probe_pos, probe_ori, probe_app, probe_pose = _probe_arm_q_pose_err(
            world, arm, q_target, target_pos, target_quat
        )
        ctx.log(
            f"    [{stage_name}] current-state final 6D re-solve OK "
            f"solve=({re_pos*1000:.1f}mm,{re_ori:.2f}°) "
            f"probe=({probe_pos*1000:.1f}mm,{probe_ori:.2f}°) "
            f"q_delta_from_stored={float(np.linalg.norm(q_target - np.asarray(q_arm, dtype=np.float64).reshape(7), ord=np.inf)):.3f}rad"
        )
    anchors, meta = _eef_plan_6d_joint_anchors(
        world,
        arm,
        target_pos,
        target_quat,
        ctx=ctx,
        stage_name=f"{stage_name}_offline",
        pos_tol=float(pos_tol),
        ori_tol_deg=float(ori_tol_deg),
        waypoint_pos_step_m=0.030,
        max_joint_gap_rad=0.55,
        max_anchor_drop_ratio=1.0,
        mid_pos_tol=max(0.018, float(pos_tol) * 1.8),
        intermediate_pos_only=True,
        final_q=q_target,
        final_seed_q=q_target,
        retain_validated_final_q_across_gap=True,
    )
    if not anchors:
        ctx.log(
            f"    [{stage_name}] ABORT no safe→final XYZ branch path "
            f"error={meta.get('error')} accepted={meta.get('accepted')}"
        )
        return float("inf"), float("inf")
    ctx.log(
        f"    [{stage_name}] play safe→final anchors n={len(anchors)} "
        f"accepted={meta.get('accepted')}"
    )
    played_err, played_ori = yield from _eef_play_joint_anchor_path(
        world,
        arm,
        anchors,
        gripper_cmd=gripper_cmd,
        ctx=ctx,
        stage_name=stage_name,
        target_pos=target_pos,
        target_quat=target_quat,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_frames_per_anchor=max(18, min(48, int(max_frames))),
        max_line_deviation_m=max_line_deviation_m,
        repair_line_waypoints=False,
        validate_intermediate_anchors=False,
    )
    if _err_failed(played_err):
        return float("inf"), float("inf")
    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
        world, arm, target_pos, target_quat
    )
    eef_now = world.eef_pose(arm=arm)
    line_dev = _dist_point_to_segment(
        np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
        start_pos,
        target_pos,
    )
    final_app_err = _eef_approach_err_deg(world, arm, target_quat)
    ctx.log(
        f"    [{stage_name}] final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}° "
        f"line_dev={line_dev*1000:.0f}mm"
    )
    if line_dev > float(max_line_deviation_m):
        ctx.log(
            f"    [{stage_name}] ABORT safe→final line deviation "
            f"{line_dev*1000:.0f}mm>{float(max_line_deviation_m)*1000:.0f}mm"
        )
        return float("inf"), float("inf")
    if final_pos_err > float(pos_tol) or final_ori_err > float(ori_tol_deg):
        return float(final_pos_err), float(final_ori_err)
    return float(final_pos_err), float(final_ori_err)


def _eef_fast_lift_from_final_q(
    world,
    arm: str,
    start_q,
    target_pos,
    lift_pos,
    target_quat,
    *,
    gripper_cmd,
    ctx,
) -> tuple[bool, float]:
    """Fast exec-only lift: solve one 3D/loose-orientation IK q and replay it."""
    q_seed = np.asarray(start_q, dtype=np.float64).reshape(7)
    lift_pos = np.asarray(lift_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    q_lift, pos_err, ori_err = _eef_solve_6d_dls_arm_q(
        world,
        arm,
        lift_pos,
        target_quat,
        pos_tol=0.030,
        ori_tol_deg=25.0,
        max_steps=140,
        max_dq_per_step=0.090,
        max_dx_per_step=0.030,
        max_dw_per_step=0.10,
        ori_weight=0.06,
        lam=0.10,
        seed_q=q_seed,
        nominal_q=q_seed,
        nominal_weight=0.020,
        nominal_tol_rad=1.10,
    )
    if q_lift is None:
        ctx.log(
            f"    [lift_fast_q] no fast lift q "
            f"pos={pos_err*1000:.0f}mm ori={ori_err:.1f}°"
        )
        return False, float("inf")
    q_lift = np.asarray(q_lift, dtype=np.float64).reshape(7)
    max_delta = float(np.linalg.norm(q_lift - q_seed, ord=np.inf))
    ctx.log(
        f"    [lift_fast_q] play lift q max_delta={max_delta:.3f}rad "
        f"solve_pos={pos_err*1000:.0f}mm solve_ori={ori_err:.1f}°"
    )
    yield from _play_arm_q_target(
        world,
        arm,
        q_lift,
        gripper_cmd=gripper_cmd,
        max_frames=10,
        hold_frames=1,
        joint_tol=0.030,
        ctx=ctx,
        stage_name="lift_fast_q",
    )
    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
        world, arm, lift_pos, target_quat
    )
    ctx.log(
        f"    [lift_fast_q] final pos={final_pos_err*1000:.0f}mm "
        f"ori={final_ori_err:.1f}° approach={final_app_err:.1f}°"
    )
    return bool(final_pos_err <= 0.060), float(final_pos_err)


def _eef_exec_short_tuck(world, arm: str, gripper_cmd, ctx) -> bool:
    """Exec-only deterministic tuck: move arm to the cached chest pose."""
    try:
        q_tuck = np.asarray(_CHEST_TUCK_ARM[str(arm).lower().strip()], dtype=np.float64).reshape(7)
    except Exception:
        return False
    q0 = _arm_qpos_now(world, arm)
    max_delta = float(np.linalg.norm(q_tuck - q0, ord=np.inf))
    ctx.log(
        f"  [tuck-short] exec fixed chest tuck arm={arm} "
        f"max_delta={max_delta:.3f}rad"
    )
    yield from _play_arm_q_target(
        world,
        arm,
        q_tuck,
        gripper_cmd=gripper_cmd,
        max_frames=max(24, min(90, int(math.ceil(max_delta / 0.050)) + 12)),
        hold_frames=6,
        ctx=ctx,
        stage_name="tuck_short",
    )
    q_track = float(np.linalg.norm(_arm_qpos_now(world, arm) - q_tuck, ord=np.inf))
    eef = world.eef_pose(arm=arm)
    ctx.log(
        f"  [tuck-short] done q_track={q_track:.3f}rad "
        f"eef=({eef['pos'][0]:.3f},{eef['pos'][1]:.3f},{eef['pos'][2]:.3f})"
    )
    return q_track <= 0.12


def _eef_goto_xyz_dls(
    world,
    arm: str,
    target_pos,
    *,
    gripper_cmd,
    ctx,
    stage_name: str,
    pos_tol: float,
    max_steps: int = 260,
    max_dq_per_step: float = 0.045,
    max_dx_per_step: float = 0.018,
    lam: float = 0.10,
    line_start_pos=None,
    line_target_pos=None,
    max_line_deviation_m: Optional[float] = None,
    line_deviation_margin_m: float = 0.0,
) -> float:
    """Online DLS that tracks only EEF XYZ; used for current→safe and line repair."""
    from behavior_interface.skills.grasp import _arm_qpos, _dls_solve_dq, _read_jacobian_arm

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    line_start_arr = None
    line_target_arr = None
    line_limit = None
    if (
        max_line_deviation_m is not None
        and line_start_pos is not None
        and line_target_pos is not None
    ):
        try:
            line_start_arr = np.asarray(line_start_pos, dtype=np.float64).reshape(3)
            line_target_arr = np.asarray(line_target_pos, dtype=np.float64).reshape(3)
            line_limit = float(max_line_deviation_m) + max(0.0, float(line_deviation_margin_m))
        except Exception:
            line_start_arr = None
            line_target_arr = None
            line_limit = None
    last_err = float("inf")
    last_eef_pos = None
    stuck_cnt = 0
    for step in range(int(max_steps)):
        eef = world.eef_pose(arm=arm)
        epos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
        dx = target_pos - epos
        err = float(np.linalg.norm(dx))
        last_err = err
        if line_limit is not None and line_start_arr is not None and line_target_arr is not None:
            dev = _dist_point_to_segment(epos, line_start_arr, line_target_arr)
            if dev > float(line_limit):
                ctx.log(
                    f"    [{stage_name}] xyz line repair left corridor step={step} "
                    f"line_dev={dev*1000:.0f}mm>{float(line_limit)*1000:.0f}mm "
                    f"err={err*1000:.0f}mm"
                )
                return float(last_err)
        if err <= float(pos_tol):
            ctx.log(f"    [{stage_name}] xyz converged step={step} err={err*1000:.0f}mm")
            return float(err)
        if err > float(max_dx_per_step):
            dx_cmd = dx * (float(max_dx_per_step) / (err + 1e-12))
        else:
            dx_cmd = dx
        try:
            J, _ = _read_jacobian_arm(world, arm)
            dq = _dls_solve_dq(J[:3, :], dx_cmd, lam=float(lam))
            dx_pred = J[:3, :] @ dq
            dq_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_inf > float(max_dq_per_step):
                dq = dq * (float(max_dq_per_step) / (dq_inf + 1e-12))
        except Exception as exc:
            ctx.log(f"    [{stage_name}] xyz jacobian err: {type(exc).__name__}: {exc}")
            return float(last_err)
        if step == 0 or step % 30 == 0:
            dq_str = "[" + ",".join(f"{v:+.3f}" for v in dq) + "]"
            ctx.log(
                f"    [{stage_name}] step={step:3d} eef=({epos[0]:.3f},{epos[1]:.3f},{epos[2]:.3f}) "
                f"err={err*1000:.0f}mm dq_inf={dq_inf:.4f} dq={dq_str} "
                f"dx_pred=({dx_pred[0]:+.3f},{dx_pred[1]:+.3f},{dx_pred[2]:+.3f})"
            )
        if last_eef_pos is not None:
            moved = float(np.linalg.norm(epos - last_eef_pos))
            stuck_cnt = stuck_cnt + 1 if moved < 0.0007 else 0
            if stuck_cnt >= 30:
                ctx.log(f"    [{stage_name}] xyz stuck step={step} err={err*1000:.0f}mm")
                return float(last_err)
        last_eef_pos = epos.copy()
        q_target = np.asarray(_arm_qpos(world, arm), dtype=np.float64).reshape(7) + dq
        yield _make_arm_q_action(world, arm, q_target.tolist(), gripper_cmd)
    ctx.log(f"    [{stage_name}] xyz timeout err={last_err*1000:.0f}mm")
    return float(last_err)


def _eef_goto_6d_line_repair_dls(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd,
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int = 90,
    max_dq_per_step: float = 0.040,
    max_dx_per_step: float = 0.012,
    max_dw_per_step: float = 0.075,
    pos_weight: float = 1.0,
    ori_weight: float = 0.70,
    lam: float = 0.10,
    line_start_pos=None,
    line_target_pos=None,
    max_line_deviation_m: Optional[float] = None,
    line_deviation_margin_m: float = 0.0,
) -> tuple[float, float]:
    """Online 6D repair for an already-planned line anchor.

    This is executor robustness, not re-planning: it nudges the real controller
    state back to the current Cartesian waypoint while staying inside the line
    corridor and preserving the planned orientation.
    """
    from behavior_interface.skills.grasp import (
        _arm_qpos,
        _dls_solve_dq,
        _orientation_error_omega,
        _quat_to_mat,
        _read_jacobian_arm,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    target_R = _quat_to_mat(target_quat)
    line_start_arr = None
    line_target_arr = None
    line_limit = None
    if (
        max_line_deviation_m is not None
        and line_start_pos is not None
        and line_target_pos is not None
    ):
        try:
            line_start_arr = np.asarray(line_start_pos, dtype=np.float64).reshape(3)
            line_target_arr = np.asarray(line_target_pos, dtype=np.float64).reshape(3)
            line_limit = float(max_line_deviation_m) + max(0.0, float(line_deviation_margin_m))
        except Exception:
            line_start_arr = None
            line_target_arr = None
            line_limit = None

    last_pos_err = float("inf")
    last_ori_err = float("inf")
    last_eef_pos = None
    stuck_cnt = 0
    for step in range(int(max_steps)):
        eef = world.eef_pose(arm=arm)
        epos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
        equat = _quat_normalize_xyzw(eef["quat"])
        cur_R = _quat_to_mat(equat)
        dx = target_pos - epos
        pos_err = float(np.linalg.norm(dx))
        omega = _orientation_error_omega(target_R, cur_R)
        ori_err_rad = float(np.linalg.norm(omega))
        ori_err_deg = float(math.degrees(ori_err_rad))
        last_pos_err = pos_err
        last_ori_err = ori_err_deg
        if line_limit is not None and line_start_arr is not None and line_target_arr is not None:
            dev = _dist_point_to_segment(epos, line_start_arr, line_target_arr)
            if dev > float(line_limit):
                ctx.log(
                    f"    [{stage_name}] 6d line repair left corridor step={step} "
                    f"line_dev={dev*1000:.0f}mm>{float(line_limit)*1000:.0f}mm "
                    f"pos={pos_err*1000:.0f}mm ori={ori_err_deg:.1f}°"
                )
                return float(last_pos_err), float(last_ori_err)
        if pos_err <= float(pos_tol) and ori_err_deg <= float(ori_tol_deg):
            ctx.log(
                f"    [{stage_name}] 6d converged step={step} "
                f"pos={pos_err*1000:.0f}mm ori={ori_err_deg:.1f}°"
            )
            return float(pos_err), float(ori_err_deg)
        dx_cmd = dx
        if pos_err > float(max_dx_per_step):
            dx_cmd = dx * (float(max_dx_per_step) / (pos_err + 1e-12))
        omega_cmd = omega
        if ori_err_rad > float(max_dw_per_step):
            omega_cmd = omega * (float(max_dw_per_step) / (ori_err_rad + 1e-12))
        try:
            J, _ = _read_jacobian_arm(world, arm)
            J_scaled = np.vstack([float(pos_weight) * J[:3, :], float(ori_weight) * J[3:6, :]])
            rhs = np.concatenate([float(pos_weight) * dx_cmd, float(ori_weight) * omega_cmd])
            dq = _dls_solve_dq(J_scaled, rhs, lam=float(lam))
            dx_pred = J[:3, :] @ dq
            dq_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_inf > float(max_dq_per_step):
                dq = dq * (float(max_dq_per_step) / (dq_inf + 1e-12))
        except Exception as exc:
            ctx.log(f"    [{stage_name}] 6d jacobian err: {type(exc).__name__}: {exc}")
            return float(last_pos_err), float(last_ori_err)
        if step == 0 or step % 20 == 0:
            dq_str = "[" + ",".join(f"{v:+.3f}" for v in dq) + "]"
            ctx.log(
                f"    [{stage_name}] step={step:3d} eef=({epos[0]:.3f},{epos[1]:.3f},{epos[2]:.3f}) "
                f"pos={pos_err*1000:.0f}mm ori={ori_err_deg:.1f}° "
                f"dq_inf={dq_inf:.4f} dq={dq_str} "
                f"dx_pred=({dx_pred[0]:+.3f},{dx_pred[1]:+.3f},{dx_pred[2]:+.3f})"
            )
        if last_eef_pos is not None:
            moved = float(np.linalg.norm(epos - last_eef_pos))
            stuck_cnt = stuck_cnt + 1 if moved < 0.0005 else 0
            if stuck_cnt >= 35:
                ctx.log(
                    f"    [{stage_name}] 6d stuck step={step} "
                    f"pos={pos_err*1000:.0f}mm ori={ori_err_deg:.1f}°"
                )
                return float(last_pos_err), float(last_ori_err)
        last_eef_pos = epos.copy()
        q_target = np.asarray(_arm_qpos(world, arm), dtype=np.float64).reshape(7) + dq
        yield _make_arm_q_action(world, arm, q_target.tolist(), gripper_cmd)
    ctx.log(
        f"    [{stage_name}] 6d timeout "
        f"pos={last_pos_err*1000:.0f}mm ori={last_ori_err:.1f}°"
    )
    return float(last_pos_err), float(last_ori_err)


def _eef_snap_6d_line_anchor(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    gripper_cmd,
    ctx,
    stage_name: str,
    pos_tol: float,
    ori_tol_deg: float,
    line_start_pos=None,
    line_target_pos=None,
    max_line_deviation_m: Optional[float] = None,
    line_deviation_margin_m: float = 0.010,
    nominal_q: Optional[np.ndarray] = None,
    max_q_delta_rad: float = 0.35,
) -> tuple[float, float, float, bool]:
    """Controller-play a same-branch correction for a line waypoint."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    q_before = _arm_qpos_now(world, arm)
    pos_before, ori_before, app_before = _eef_pose_err(world, arm, target_pos, target_quat)
    dev_before = 0.0
    line_limit = None
    if (
        max_line_deviation_m is not None
        and line_start_pos is not None
        and line_target_pos is not None
    ):
        line_limit = float(max_line_deviation_m) + max(0.0, float(line_deviation_margin_m))
        try:
            eef_now = world.eef_pose(arm=arm)
            dev_before = _dist_point_to_segment(
                np.asarray(eef_now["pos"], dtype=np.float64).reshape(3),
                np.asarray(line_start_pos, dtype=np.float64).reshape(3),
                np.asarray(line_target_pos, dtype=np.float64).reshape(3),
            )
        except Exception:
            dev_before = 0.0

    q_snap, solve_pos_err, solve_ori_err = _eef_solve_6d_dls_arm_q(
        world,
        arm,
        target_pos,
        target_quat,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=420,
        max_dq_per_step=0.060,
        max_dx_per_step=0.016,
        max_dw_per_step=0.085,
        ori_weight=0.85,
        lam=0.08,
        seed_q=q_before,
        nominal_q=nominal_q,
        nominal_weight=0.050 if nominal_q is not None else 0.0,
        nominal_tol_rad=0.55 if nominal_q is not None else None,
    )
    if q_snap is None:
        ctx.log(
            f"    [{stage_name}] snap no same-branch 6D q "
            f"solve_pos={solve_pos_err*1000:.0f}mm solve_ori={solve_ori_err:.1f}° "
            f"before_pos={pos_before*1000:.0f}mm before_ori={ori_before:.1f}°"
        )
        return float(pos_before), float(ori_before), float(dev_before), False
    q_snap = np.asarray(q_snap, dtype=np.float64).reshape(7)
    q_delta = float(np.linalg.norm(q_snap - q_before, ord=np.inf))
    if q_delta > float(max_q_delta_rad):
        ctx.log(
            f"    [{stage_name}] snap rejected q_delta={q_delta:.3f}rad>"
            f"{float(max_q_delta_rad):.3f}rad solve_pos={solve_pos_err*1000:.0f}mm "
            f"solve_ori={solve_ori_err:.1f}°"
        )
        return float(pos_before), float(ori_before), float(dev_before), False

    played_ok = yield from _play_arm_q_target(
        world,
        arm,
        q_snap,
        gripper_cmd=gripper_cmd,
        max_frames=max(8, int(math.ceil(q_delta / 0.04)) + 2),
        hold_frames=max(8, _LINE_Q_RESIDUAL_CORRECT_FRAMES),
        min_hold_frames=4,
        joint_tol=0.015,
        ctx=ctx,
        stage_name=f"{stage_name}_controller",
        line_start_pos=line_start_pos,
        line_target_pos=line_target_pos,
        max_line_deviation_m=line_limit,
        allow_direct_residual_correction=False,
    )
    if played_ok is not True:
        ctx.log(f"    [{stage_name}] controller correction did not settle")
        return float(pos_before), float(ori_before), float(dev_before), False

    pos_after, ori_after, app_after = _eef_pose_err(world, arm, target_pos, target_quat)
    dev_after = 0.0
    if line_limit is not None and line_start_pos is not None and line_target_pos is not None:
        try:
            eef_after = world.eef_pose(arm=arm)
            dev_after = _dist_point_to_segment(
                np.asarray(eef_after["pos"], dtype=np.float64).reshape(3),
                np.asarray(line_start_pos, dtype=np.float64).reshape(3),
                np.asarray(line_target_pos, dtype=np.float64).reshape(3),
            )
        except Exception:
            dev_after = 0.0
    before_score = float(pos_before) + 0.002 * float(ori_before) + 0.5 * max(0.0, float(dev_before) - float(line_limit or 0.0))
    after_score = float(pos_after) + 0.002 * float(ori_after) + 0.5 * max(0.0, float(dev_after) - float(line_limit or 0.0))
    corridor_ok = bool(line_limit is None or dev_after <= float(line_limit))
    improved = bool(after_score <= before_score + 1e-6)
    ok = bool(corridor_ok and improved)
    ctx.log(
        f"    [{stage_name}] snap result "
        f"pos={pos_before*1000:.0f}->{pos_after*1000:.0f}mm "
        f"ori={ori_before:.1f}->{ori_after:.1f}° "
        f"approach={app_before:.1f}->{app_after:.1f}° "
        f"line_dev={dev_before*1000:.0f}->{dev_after*1000:.0f}mm "
        f"q_delta={q_delta:.3f}rad ok={ok}"
    )
    if not ok:
        yield from _play_arm_q_target(
            world,
            arm,
            q_before,
            gripper_cmd=gripper_cmd,
            max_frames=max(8, int(math.ceil(q_delta / 0.04)) + 2),
            hold_frames=max(8, _LINE_Q_RESIDUAL_CORRECT_FRAMES),
            min_hold_frames=4,
            joint_tol=0.015,
            ctx=ctx,
            stage_name=f"{stage_name}_rollback",
            allow_direct_residual_correction=False,
        )
        return float(pos_before), float(ori_before), float(dev_before), False
    return float(pos_after), float(ori_after), float(dev_after), True


def _stored_filter_q_for_arm(cand: Dict[str, Any], arm: str) -> Optional[list]:
    """Return selected plan-filter IK q for arm, if the plan record carries it."""
    arm = str(arm or "").lower().strip()
    if arm not in ("left", "right"):
        return None
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    plan_audit = meta.get("plan_audit") if isinstance(meta.get("plan_audit"), dict) else {}
    pick = plan_audit.get("pick") if isinstance(plan_audit.get("pick"), dict) else {}
    locations = [
        cand,
        meta,
        meta.get("grip_fit") if isinstance(meta, dict) else None,
        plan_audit,
        pick,
    ]
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        info = loc.get("selected_pose_ik")
        if isinstance(info, dict):
            q = info.get(f"{arm}_q_arm") or info.get("q_arm")
            if q is not None:
                return q
        q_map = loc.get("selected_pose_ik_q")
        if isinstance(q_map, dict) and q_map.get(arm) is not None:
            return q_map.get(arm)
        ik_filter = loc.get("ik_filter")
        if isinstance(ik_filter, dict):
            arm_info = ik_filter.get(arm)
            if isinstance(arm_info, dict) and arm_info.get("q_arm") is not None:
                return arm_info.get("q_arm")
    return None


def _stored_planned_safe_for_arm(
    cand: Dict[str, Any],
    arm: str,
) -> Optional[Dict[str, Any]]:
    """Return the safe endpoint already paired with final IK during planning."""
    arm = str(arm or "").lower().strip()
    if arm not in ("left", "right") or not isinstance(cand, dict):
        return None
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    plan_audit = (
        meta.get("plan_audit")
        if isinstance(meta.get("plan_audit"), dict)
        else {}
    )
    pick = (
        plan_audit.get("pick")
        if isinstance(plan_audit.get("pick"), dict)
        else {}
    )
    locations = [
        cand,
        meta,
        meta.get("grip_fit") if isinstance(meta, dict) else None,
        plan_audit,
        pick,
    ]
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        planned = loc.get("planned_safe")
        if not isinstance(planned, dict):
            continue
        planned_arm = str(planned.get("arm") or arm).lower().strip()
        if (
            planned_arm == arm
            and planned.get("ok")
            and planned.get("safe_pos") is not None
            and planned.get("safe_q") is not None
        ):
            return dict(planned)
    return None


def _stored_planned_safe_matches_request(
    planned_safe: Optional[Dict[str, Any]],
    requested_back_m: float,
    *,
    tol_m: float = 1e-4,
) -> bool:
    """Whether a stored safe pair was planned for this execution request."""
    if not isinstance(planned_safe, dict):
        return False
    stored_requested = planned_safe.get(
        "requested_back_m",
        planned_safe.get("active_back_m"),
    )
    try:
        stored_requested_f = float(stored_requested)
        requested_f = float(requested_back_m)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(stored_requested_f)
        and np.isfinite(requested_f)
        and abs(stored_requested_f - requested_f) <= float(tol_m)
    )


def _candidate_requires_stored_final_q(cand: Dict[str, Any]) -> bool:
    """Filter planners already validated final IK; exec must not replace that q."""
    if not isinstance(cand, dict):
        return False
    mode_values = [
        cand.get("skill"),
        cand.get("mode"),
        cand.get("label"),
        (cand.get("meta") or {}).get("mode") if isinstance(cand.get("meta"), dict) else None,
        (cand.get("meta") or {}).get("skill") if isinstance(cand.get("meta"), dict) else None,
    ]
    text = " ".join(str(v or "") for v in mode_values).lower()
    return (
        ("plan_grasp" in text and "filter" in text)
        or "grasp_obj_filter" in text
        or "grasp_point_filter" in text
        or "press_point" in text
    )


def _stored_press_gpu_move_trajectory(
    cand: Dict[str, Any],
    arm: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the plan-time GPU current->safe->press path, if present."""
    if not isinstance(cand, dict):
        return None
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    plan_audit = (
        meta.get("plan_audit")
        if isinstance(meta.get("plan_audit"), dict)
        else {}
    )
    requested_arm = str(arm or "").lower().strip()
    for location in (cand, meta, plan_audit):
        if not isinstance(location, dict):
            continue
        trajectory = location.get("move_only_trajectory")
        if not isinstance(trajectory, dict):
            continue
        if trajectory.get("planner") != "gpu_ik_current_to_safe_to_press":
            continue
        trajectory_arm = str(trajectory.get("arm") or "").lower().strip()
        if requested_arm and trajectory_arm != requested_arm:
            continue
        segments = trajectory.get("segments")
        if not isinstance(segments, list) or len(segments) != 2:
            continue
        if [str(segment.get("name")) for segment in segments] != [
            "current_to_safe",
            "safe_to_press",
        ]:
            continue
        if any(not (segment.get("anchors") or []) for segment in segments):
            continue
        return trajectory
    return None


def _eef_play_stored_press_gpu_trajectory(
    world,
    arm: str,
    trajectory: Dict[str, Any],
    *,
    gripper_cmd,
    ctx,
    cand: Dict[str, Any],
    target_pos,
    target_quat,
    stop_after_safe: bool = False,
):
    """Replay the exact two GPU-IK segments accepted by plan_press_point."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(target_quat)
    segments = list(trajectory.get("segments") or [])
    back_m = float(trajectory.get("back_m", 0.10))
    try:
        start_q = np.asarray(
            trajectory.get("start_arm_q"), dtype=np.float64
        ).reshape(7)
    except Exception:
        return {
            "ok": False,
            "eef_id": cand.get("id"),
            "label": cand.get("label"),
            "target": cand.get("target", "move_eef"),
            "arm": arm,
            "gpu_ik_trajectory": True,
            "error": "plan_press_point GPU 轨迹缺少有效的 7-DOF 起点关节状态",
        }
    current_q = _arm_qpos_now(world, arm)
    start_gap = float(np.linalg.norm(current_q - start_q, ord=np.inf))
    if start_gap > 0.12:
        ctx.log(
            "  ABORT stored press GPU trajectory start state is stale: "
            f"gap={start_gap:.3f}rad>0.120rad"
        )
        return {
            "ok": False,
            "eef_id": cand.get("id"),
            "label": cand.get("label"),
            "target": cand.get("target", "move_eef"),
            "arm": arm,
            "gpu_ik_trajectory": True,
            "trajectory_start_joint_gap_rad": start_gap,
            "error": (
                "plan_press_point 轨迹起点已失效：当前手臂关节状态与规划时相差"
                f" {start_gap:.3f}rad > 0.120rad，请重新 plan"
            ),
        }

    final_segment = segments[1]
    stored_target_pos = np.asarray(
        final_segment["target_pos"], dtype=np.float64
    ).reshape(3)
    stored_target_quat = _quat_normalize_xyzw(final_segment["target_quat"])
    target_pos_mismatch = float(np.linalg.norm(stored_target_pos - target_pos))
    target_quat_dot = abs(float(stored_target_quat @ target_quat))
    target_ori_mismatch = math.degrees(
        2.0 * math.acos(float(np.clip(target_quat_dot, -1.0, 1.0)))
    )
    if target_pos_mismatch > 1e-5 or target_ori_mismatch > 1e-3:
        ctx.log(
            "  ABORT stored press GPU trajectory target mismatch: "
            f"pos={target_pos_mismatch * 1000.0:.3f}mm "
            f"ori={target_ori_mismatch:.4f}deg"
        )
        return {
            "ok": False,
            "eef_id": cand.get("id"),
            "label": cand.get("label"),
            "target": cand.get("target", "move_eef"),
            "arm": arm,
            "gpu_ik_trajectory": True,
            "error": "plan_press_point 存储轨迹与 candidate 末端位姿不一致",
        }

    ctx.log(
        "  [move_only] replay plan-time GPU trajectory "
        f"current→safe→press arm={arm} back={back_m * 100.0:.0f}cm "
        f"start_gap={start_gap:.3f}rad"
    )
    segment_reports = []
    for segment_index, segment in enumerate(segments):
        name = str(segment["name"])
        segment_target_pos = np.asarray(
            segment["target_pos"], dtype=np.float64
        ).reshape(3)
        segment_target_quat = _quat_normalize_xyzw(segment["target_quat"])
        anchors = list(segment.get("anchors") or [])
        pos_err, ori_err = yield from _eef_play_joint_anchor_path(
            world,
            arm,
            anchors,
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name=f"move_only_line_gpu_{name}",
            target_pos=segment_target_pos,
            target_quat=segment_target_quat,
            pos_tol=_EEF_FINAL_POS_TOL_M,
            ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
            max_frames_per_anchor=42,
            max_line_deviation_m=(
                0.035 if name == "current_to_safe" else 0.020
            ),
            repair_line_waypoints=False,
        )
        segment_ok = bool(
            math.isfinite(float(pos_err))
            and math.isfinite(float(ori_err))
            and float(pos_err) <= _EEF_FINAL_POS_TOL_M
            and float(ori_err) <= _EEF_FINAL_ORI_TOL_DEG
        )
        segment_reports.append({
            "name": name,
            "ok": segment_ok,
            "anchor_count": int(len(anchors)),
            "pos_err_m": _err_json(pos_err),
            "ori_err_deg": round(float(ori_err), 3),
        })
        if not segment_ok:
            ctx.log(
                f"  ABORT stored GPU segment {name} failed "
                f"pos={float(pos_err) * 1000.0:.1f}mm "
                f"ori={float(ori_err):.2f}deg"
            )
            return {
                "ok": False,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": cand.get("target", "move_eef"),
                "arm": arm,
                "pre_err": (
                    segment_reports[0]["pos_err_m"]
                    if segment_reports else None
                ),
                "cnt_err": None,
                "end_err": None,
                "final_pos_err_m": _err_json(pos_err),
                "final_ori_err_deg": round(float(ori_err), 2),
                "used_curobo": True,
                "gpu_ik_trajectory": True,
                "line_constraint_failed": True,
                "trajectory_segments": segment_reports,
                "error": (
                    f"plan_press_point GPU 轨迹回放失败：{name} 未满足 "
                    f"{_EEF_FINAL_POS_TOL_M * 1000.0:.0f}mm/"
                    f"{_EEF_FINAL_ORI_TOL_DEG:.0f}deg"
                ),
            }
        if segment_index == 0 and bool(stop_after_safe):
            safe_pos_err, safe_ori_err, safe_app_err = _eef_pose_err(
                world, arm, segment_target_pos, segment_target_quat
            )
            return {
                "ok": True,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": "safe_pose",
                "arm": arm,
                "pre_err": _err_json(safe_pos_err),
                "cnt_err": _err_json(safe_pos_err),
                "end_err": 0.0,
                "final_pos_err_m": _err_json(safe_pos_err),
                "final_ori_err_deg": round(float(safe_ori_err), 2),
                "final_approach_err_deg": round(float(safe_app_err), 2),
                "used_curobo": True,
                "gpu_ik_trajectory": True,
                "back_m": back_m,
                "trajectory_start_joint_gap_rad": start_gap,
                "trajectory_segments": segment_reports,
                "safe_pose": {
                    "target_pos": segment_target_pos.tolist(),
                    "target_quat": segment_target_quat.tolist(),
                    "back_m": back_m,
                },
            }

    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
        world, arm, target_pos, target_quat
    )
    final_ok = bool(
        final_pos_err <= _EEF_FINAL_POS_TOL_M
        and final_ori_err <= _EEF_FINAL_ORI_TOL_DEG
    )
    result = {
        "ok": final_ok,
        "eef_id": cand.get("id"),
        "label": cand.get("label"),
        "target": cand.get("target", "move_eef"),
        "arm": arm,
        "pre_err": segment_reports[0]["pos_err_m"],
        "cnt_err": _err_json(final_pos_err),
        "end_err": 0.0,
        "final_pos_err_m": _err_json(final_pos_err),
        "final_ori_err_deg": round(float(final_ori_err), 2),
        "final_approach_err_deg": round(float(final_app_err), 2),
        "used_curobo": True,
        "gpu_ik_trajectory": True,
        "back_m": back_m,
        "trajectory_start_joint_gap_rad": start_gap,
        "trajectory_segments": segment_reports,
    }
    if not final_ok:
        result["error"] = (
            "plan_press_point GPU 轨迹已回放，但最终 EEF 未满足 "
            f"{_EEF_FINAL_POS_TOL_M * 1000.0:.0f}mm/"
            f"{_EEF_FINAL_ORI_TOL_DEG:.0f}deg"
        )
    return result


def _eef_goto_world_approach(
    world,
    arm: str,
    target_world_pos,
    target_world_quat,
    *,
    gripper_cmd: Optional[float] = None,
    ctx=None,
    stage_name: str = "",
    max_steps: int = 220,
    pos_tol: float = 0.025,
    approach_tol_deg: float = _GRASP_ORI_GATE_DEG,
    max_dq_per_step: float = 0.04,
    max_dx_per_step: float = 0.018,
    max_dw_per_step: float = 0.08,
    pos_weight: float = 8.0,
    approach_weight: float = 0.8,
    lam: float = 0.12,
) -> float:
    """DLS move with hard position priority and approach-axis orientation only."""
    from behavior_interface.skills.grasp import (
        _arm_qpos,
        _dls_solve_dq,
        _quat_to_mat,
        _read_jacobian_arm,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_world_pos = np.asarray(target_world_pos, dtype=np.float64).reshape(3)
    R_tgt = _quat_to_mat(np.asarray(target_world_quat, dtype=np.float64).reshape(4))
    app_tgt = np.asarray(R_tgt[:, 2], dtype=np.float64)
    app_tgt = app_tgt / (np.linalg.norm(app_tgt) + 1e-9)
    last_err = float("inf")

    for step in range(int(max_steps)):
        eef = world.eef_pose(arm=arm)
        epos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
        equat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
        dx_world = target_world_pos - epos
        pos_err = float(np.linalg.norm(dx_world))
        last_err = pos_err

        R_cur = _quat_to_mat(equat)
        app_cur = np.asarray(R_cur[:, 2], dtype=np.float64)
        app_cur = app_cur / (np.linalg.norm(app_cur) + 1e-9)
        cross = np.cross(app_cur, app_tgt)
        sin_ang = float(np.linalg.norm(cross))
        cos_ang = float(np.clip(np.dot(app_cur, app_tgt), -1.0, 1.0))
        app_err_rad = float(math.atan2(sin_ang, cos_ang))
        app_err_deg = float(math.degrees(app_err_rad))

        if pos_err <= pos_tol and app_err_deg <= approach_tol_deg:
            if ctx is not None:
                ctx.log(
                    f"    [{stage_name}] converged step={step} "
                    f"pos_err={pos_err:.3f}m approach_err={app_err_deg:.1f}°"
                )
            return last_err

        if pos_err > max_dx_per_step:
            dx_cmd = dx_world * (max_dx_per_step / pos_err)
        else:
            dx_cmd = dx_world
        if sin_ang > 1e-8:
            omega = cross / sin_ang * min(app_err_rad, float(max_dw_per_step))
        else:
            omega = np.zeros(3, dtype=np.float64)

        try:
            J, _ = _read_jacobian_arm(world, arm)
            Jp = J[:3, :]
            Jo = J[3:6, :]
            J_scaled = np.vstack([pos_weight * Jp, approach_weight * Jo])
            rhs = np.concatenate([pos_weight * dx_cmd, approach_weight * omega])
            dq = _dls_solve_dq(J_scaled, rhs, lam=lam)
            dx_pred = Jp @ dq
            dq_norm_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_norm_inf > max_dq_per_step:
                dq = dq * (max_dq_per_step / dq_norm_inf)
        except Exception as e:
            if ctx is not None:
                ctx.log(f"    [{stage_name}] approach jacobian err: {e}")
            return last_err

        if ctx is not None and (step == 0 or step % 30 == 0):
            dq_str = "[" + ",".join(f"{v:+.3f}" for v in dq) + "]"
            ctx.log(
                f"    [{stage_name}] step={step:3d} eef=({epos[0]:.3f},"
                f"{epos[1]:.3f},{epos[2]:.3f}) err={pos_err:.3f}m "
                f"approach_err={app_err_deg:.1f}° dq_inf={dq_norm_inf:.4f} "
                f"dq={dq_str} dx_pred=({dx_pred[0]:+.3f},{dx_pred[1]:+.3f},{dx_pred[2]:+.3f})"
            )

        q_tgt_arm = _arm_qpos(world, arm) + dq
        yield _make_arm_q_action(world, arm, q_tgt_arm.tolist(), gripper_cmd)

    if ctx is not None:
        ctx.log(f"    [{stage_name}] timeout {max_steps} steps, final err={last_err:.3f}m")
    return last_err


def _top_down_grasp_quat(target_pos: np.ndarray, world) -> np.ndarray:
    """自上而下接近用的 EEF 四元数（approach 沿 -Z）。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _mat_to_quat_xyzw

    z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    try:
        bp = world.robot.get_position()
        to_robot = np.array([float(bp[0]) - target_pos[0], float(bp[1]) - target_pos[1], 0.0])
    except Exception:
        to_robot = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if float(np.linalg.norm(to_robot)) < 1e-3:
        y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        y = to_robot / (np.linalg.norm(to_robot) + 1e-9)
    x = np.cross(y, z)
    if float(np.linalg.norm(x)) < 1e-3:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    y /= np.linalg.norm(y) + 1e-9
    return _mat_to_quat_xyzw(np.column_stack([x, y, z]))


def _goto_eef_stage(
    world,
    arm: str,
    pos: np.ndarray,
    quat: np.ndarray,
    *,
    gripper_cmd: float,
    ctx,
    stage_name: str,
    use_curobo: bool,
    allow_contact: bool,
    pos_tol: float = _STAGE_PRE_TOL,
    lock_body: bool = False,
    lock_other_arm: bool = True,
) -> Tuple[float, bool]:
    """单段 EEF 移动；allow_contact=False 时启用 cuRobo 世界碰撞避障。"""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if use_curobo:
        from behavior_interface.skills.grasp import _eef_goto_world_curobo

        err = yield from _eef_goto_world_curobo(
            world, arm, pos, target_world_quat=quat.tolist(),
            gripper_cmd=gripper_cmd, ctx=ctx, stage_name=stage_name,
            max_attempts=36 if not allow_contact else 32,
            timeout=14.0 if not allow_contact else 10.0,
            allow_contact=allow_contact,
            lock_body=True,
            lock_other_arm=True,
        )
        if _curobo_move_failed(err):
            ctx.log(f"  WARN curobo {stage_name} 失败 → DLS IK（无避障）")
            err = yield from _eef_goto_world(
                world, arm, pos, target_world_quat=quat,
                max_steps=_STAGE_MAX_STEPS, pos_tol=pos_tol,
                gripper_cmd=gripper_cmd, ctx=ctx, stage_name=stage_name,
            )
            return float(err), False
        return float(err), True
    err = yield from _eef_goto_world(
        world, arm, pos, target_world_quat=quat,
        max_steps=_STAGE_MAX_STEPS, pos_tol=pos_tol,
        gripper_cmd=gripper_cmd, ctx=ctx, stage_name=stage_name,
    )
    return float(err), use_curobo


def _grasp_lift_topdown_approach(
    world,
    arm: str,
    target_pos: np.ndarray,
    grasp_ori: np.ndarray,
    pre_grip: float,
    ctx,
    use_curobo: bool,
    approach: Optional[np.ndarray] = None,
    target_obj=None,
    back_m: float = 0.10,
    candidate: Optional[Dict[str, Any]] = None,
):
    """
    简化避障（无 cuRobo 路径规划，靠"安全中转位形"实现避障效果）：

      Stage0 tuck：两阶段关节折线收胸前（垂直抬升→贴胸收拢，只动 arm）
      Stage1 to_safe：从胸前 DLS 直线到 safe_pos
        safe_pos = target_pos - back_m * pointing
        pointing = 夹爪指向（由 grasp quat 的局部 +Z 推出）
      Stage2 contact：从 safe_pos DLS 直线到真正抓取点（仅 back_m 短距离）

    全程直线 IK，不做任何路径避障；靠"先收胸前、再外伸"经过自由空间避开障碍。
    输入只需 eef pose(target_pos/grasp_ori) + next_move（由调用方处理），
    与用哪个 plan 函数无关。

    【exec_move 硬约束】全程**只允许动指定手臂的 7 个关节 + 夹爪**。
    **禁止**移动底盘(base)或腰部(trunk)；make_action 不得出现 base/trunk 覆盖。
    """
    from behavior_interface.skills.grasp import _eef_goto_world, _quat_to_mat

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    grasp_ori = np.asarray(grasp_ori, dtype=np.float64).reshape(4)

    # 夹爪指向：eef 局部 +Z（世界系）。safe = 沿指向反向后退 back_m，姿态与 eef 目标相同。
    pointing = _quat_to_mat(grasp_ori)[:, 2]
    npt = float(np.linalg.norm(pointing))
    pointing = pointing / npt if npt > 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    safe_pos = target_pos - float(back_m) * pointing

    ctx.log(
        f"  [grasp exec] 简化避障 tuck→safe→contact back={back_m*100:.0f}cm\n"
        f"    safe_pos=({safe_pos[0]:.3f},{safe_pos[1]:.3f},{safe_pos[2]:.3f}) "
        f"safe_quat=({grasp_ori[0]:+.3f},{grasp_ori[1]:+.3f},{grasp_ori[2]:+.3f},{grasp_ori[3]:+.3f})\n"
        f"    eef_pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
        f"(姿态同 safe_quat)\n"
        f"    pointing=({pointing[0]:+.2f},{pointing[1]:+.2f},{pointing[2]:+.2f})"
    )
    safe_reach_ok, safe_d_sh, safe_limit = _eef_target_reach_preflight(
        world, arm, safe_pos, label="safe_pose", ctx=ctx,
    )
    final_reach_ok, final_d_sh, final_limit = _eef_target_reach_preflight(
        world, arm, target_pos, label="eef_pose", ctx=ctx,
    )
    if not safe_reach_ok:
        ctx.log(
            f"  [grasp exec] requested safe_pose reach preflight failed "
            f"({safe_d_sh:.3f}m>{safe_limit:.3f}m); will try adaptive back_m"
        )
    if not final_reach_ok:
        ctx.log(
            f"  ABORT eef_pose 超出当前肩膀明显可达范围："
            f"{final_d_sh:.3f}m > {final_limit:.3f}m；请先 move_to_object 靠近后重新 plan/exec"
        )
        return float("inf"), float("inf"), False

    ctx.log("  [grasp exec] no tuck: current→safe uses XYZ-only midpoints, same-branch IK")
    return (yield from _eef_direct_safe_approach(
        world, arm, target_pos, grasp_ori, pre_grip, ctx,
        approach=approach, target_obj=target_obj, back_m=back_m,
        prefer_curobo=use_curobo,
        candidate=candidate,
    ))

def _eef_direct_safe_approach(
    world,
    arm: str,
    target_pos: np.ndarray,
    grasp_ori: np.ndarray,
    gripper_cmd: Optional[float],
    ctx,
    approach: Optional[np.ndarray] = None,
    target_obj=None,
    back_m: float = 0.10,
    prefer_curobo: bool = True,
    stop_after_safe: bool = False,
    candidate: Optional[Dict[str, Any]] = None,
):
    """exec_eef_pose 直连：当前 EEF → safe_pose → eef_pose。

    Filter grasp execution treats the plan-filter final q as the only final
    branch.  The safe endpoint is a relaxed 6D solve, intermediate waypoints
    may be XYZ-only same-branch IK, and the final grasp is verified as the
    stored plan-filter 6D pose.
    """
    from behavior_interface.skills.grasp import _eef_goto_world, _quat_to_mat

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    grasp_ori = np.asarray(grasp_ori, dtype=np.float64).reshape(4)
    # safe_pose：位置沿夹爪 +Z 反向后退 back_m，四元数与 eef 目标相同
    pointing = _quat_to_mat(grasp_ori)[:, 2]
    npt = float(np.linalg.norm(pointing))
    pointing = pointing / npt if npt > 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    safe_pos = target_pos - float(back_m) * pointing
    requested_back_m = float(back_m)
    stored_final_q = _stored_filter_q_for_arm(candidate or {}, arm)
    stored_final_q_np = (
        np.asarray(stored_final_q, dtype=np.float64).reshape(7)
        if stored_final_q is not None else None
    )
    stored_planned_safe = _stored_planned_safe_for_arm(
        candidate or {},
        arm,
    )
    use_stored_planned_safe = _stored_planned_safe_matches_request(
        stored_planned_safe,
        requested_back_m,
    )
    if stored_final_q_np is not None:
        ctx.log(
            f"    [to_safe] using stored plan-filter final q as same-branch seed "
            f"arm={arm}"
        )
    else:
        cand_keys = sorted((candidate or {}).keys()) if isinstance(candidate, dict) else []
        meta_keys = sorted(((candidate or {}).get("meta") or {}).keys()) if isinstance(candidate, dict) else []
        ctx.log(
            f"    [to_safe] no stored plan-filter final q for arm={arm}; "
            f"cand_keys={cand_keys} meta_keys={meta_keys}"
        )
        if _candidate_requires_stored_final_q(candidate or {}):
            ctx.log(
                "  ABORT filter grasp exec requires stored plan-filter final q; "
                "not running a fresh final IK branch"
            )
            return float("inf"), float("inf"), False
    if use_stored_planned_safe:
        safe_pos = np.asarray(
            stored_planned_safe["safe_pos"], dtype=np.float64
        ).reshape(3)
        back_m = float(
            stored_planned_safe.get(
                "active_back_m",
                stored_planned_safe.get("requested_back_m", requested_back_m),
            )
        )
        safe_plan_meta = dict(stored_planned_safe)
        safe_plan_meta["ok"] = True
        safe_plan_meta["source"] = "paired_safe_final_ik_filter"
        ctx.log(
            "    [to_safe] using planner-filtered same-arm safe+final IK pair "
            f"arm={arm} back={back_m*100:.0f}cm"
        )
    else:
        stored_fallback_back_m = None
        if stored_planned_safe is not None:
            try:
                stored_fallback_back_m = float(
                    stored_planned_safe.get(
                        "active_back_m",
                        stored_planned_safe.get("requested_back_m"),
                    )
                )
            except (TypeError, ValueError):
                stored_fallback_back_m = None
            if (
                stored_fallback_back_m is not None
                and not np.isfinite(stored_fallback_back_m)
            ):
                stored_fallback_back_m = None
            mismatch_log = (
                "    [to_safe] planner safe does not match exec request; "
                f"requested={requested_back_m*100:.1f}cm"
            )
            if stored_fallback_back_m is not None:
                mismatch_log += (
                    f" stored={stored_fallback_back_m*100:.1f}cm"
                    " (fallback after requested)"
                )
            ctx.log(mismatch_log + "; replanning requested back_m first")
        safe_pos, back_m, safe_plan_meta = _adapt_safe_pose_back_m(
            world,
            arm,
            target_pos,
            grasp_ori,
            pointing,
            requested_back_m,
            ctx=ctx,
            stage_name="to_safe",
            final_q=stored_final_q_np,
            fallback_back_m=stored_fallback_back_m,
        )
    to_safe_anchors = None
    if isinstance(safe_plan_meta, dict):
        to_safe_anchors = safe_plan_meta.pop("_to_safe_anchors", None)
    safe_q_from_meta = None
    if isinstance(safe_plan_meta, dict) and safe_plan_meta.get("safe_q") is not None:
        try:
            safe_q_from_meta = np.asarray(safe_plan_meta.get("safe_q"), dtype=np.float64).reshape(7)
        except Exception:
            safe_q_from_meta = None
    paired_safe_plan = bool(
        use_stored_planned_safe
        and stored_final_q_np is not None
        and safe_q_from_meta is not None
    )
    if paired_safe_plan and not to_safe_anchors:
        to_safe_anchors, paired_path_meta = _eef_plan_to_safe_xyz_anchor_path(
            world,
            arm,
            safe_pos,
            grasp_ori,
            safe_q=safe_q_from_meta,
            final_q=stored_final_q_np,
            ctx=ctx,
            stage_name="to_safe_paired_xyz",
            max_joint_gap_rad=0.85,
            require_endpoint=abs(float(back_m)) <= 1e-6,
        )
        safe_plan_meta["to_safe_path"] = paired_path_meta
        safe_plan_meta["to_safe_anchor_count"] = int(len(to_safe_anchors))
        if to_safe_anchors:
            safe_plan_meta["ok"] = True
            safe_plan_meta.pop("error", None)
        else:
            safe_plan_meta["ok"] = False
            safe_plan_meta["error"] = (
                paired_path_meta.get("error")
                or "to_safe_xyz_waypoint_ik_failed"
            )

    if (
        not (
            isinstance(safe_plan_meta, dict)
            and bool(safe_plan_meta.get("ok"))
            and to_safe_anchors
        )
    ):
        err_msg = _safe_plan_failure_message("to_safe", safe_plan_meta)
        ctx.log(f"  ABORT safe pose 离线规划失败，未执行手臂动作：{err_msg}")
        eef_now = world.eef_pose(arm=arm)
        try:
            pos_now = np.asarray(eef_now["pos"], dtype=np.float64).reshape(3)
            pos_err_now = float(np.linalg.norm(pos_now - target_pos))
            safe_pos_err_now = float(np.linalg.norm(pos_now - safe_pos))
        except Exception:
            pos_err_now = float("inf")
            safe_pos_err_now = float("inf")
        try:
            ori_now = _eef_ori_err_deg(world, arm, grasp_ori)
            approach_now = _eef_approach_err_deg(world, arm, grasp_ori)
        except Exception:
            ori_now = 999.0
            approach_now = 999.0
        return {
            "ok": False,
            "target": "safe_pose",
            "arm": arm,
            "pre_err": None,
            "cnt_err": None,
            "end_err": None,
            "final_pos_err_m": _err_json(pos_err_now),
            "final_ori_err_deg": round(float(ori_now), 2),
            "final_approach_err_deg": round(float(approach_now), 2),
            "safe_pose": {
                "target_pos": safe_pos.tolist(),
                "target_quat": grasp_ori.tolist(),
                "actual_pos": (
                    np.asarray(eef_now.get("pos"), dtype=np.float64).reshape(3).tolist()
                    if isinstance(eef_now, dict) and eef_now.get("pos") is not None
                    else None
                ),
                "safe_pos_err_m": _err_json(safe_pos_err_now),
                "eef_target_pos": target_pos.tolist(),
                "back_m": float(back_m),
                "requested_back_m": float(requested_back_m),
                "pointing": pointing.tolist(),
                "safe_plan_meta": safe_plan_meta,
            },
            "skip_next_move": True,
            "skip_grip_close": True,
            "used_curobo": False,
            "safe_plan_failed": True,
            "error": err_msg,
        }

    eef0 = world.eef_pose(arm=arm)
    ctx.log(
        f"  [exec_eef_pose] 直连 current→safe_pose→eef_pose "
        f"back={requested_back_m*100:.0f}cm active_back={back_m*100:.0f}cm\n"
        f"    eef0=({eef0['pos'][0]:.3f},{eef0['pos'][1]:.3f},{eef0['pos'][2]:.3f})\n"
        f"    safe_pos=({safe_pos[0]:.3f},{safe_pos[1]:.3f},{safe_pos[2]:.3f}) "
        f"safe_quat=({grasp_ori[0]:+.3f},{grasp_ori[1]:+.3f},{grasp_ori[2]:+.3f},{grasp_ori[3]:+.3f})\n"
        f"    eef_pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
        f"(姿态同 safe_quat)"
    )
    safe_reach_ok, safe_d_sh, safe_limit = _eef_target_reach_preflight(
        world, arm, safe_pos, label="safe_pose", ctx=ctx,
    )
    final_reach_ok, final_d_sh, final_limit = _eef_target_reach_preflight(
        world, arm, target_pos, label="eef_pose", ctx=ctx,
    )
    if not safe_reach_ok or not final_reach_ok:
        bad_label = "safe_pose" if not safe_reach_ok else "eef_pose"
        bad_d = safe_d_sh if not safe_reach_ok else final_d_sh
        bad_limit = safe_limit if not safe_reach_ok else final_limit
        msg = (
            f"{bad_label} 超出当前肩膀明显可达范围："
            f"{bad_d:.3f}m > {bad_limit:.3f}m；请先 move_to_object 靠近后重新 plan/exec"
        )
        ctx.log(f"  ABORT {msg}")
        if stop_after_safe:
            return {
                "ok": False,
                "target": "safe_pose",
                "arm": arm,
                "pre_err": None,
                "cnt_err": None,
                "final_pos_err_m": None,
                "final_ori_err_deg": None,
                "safe_pose": {
                    "safe_target_pos": safe_pos.tolist(),
                    "safe_actual_pos": np.asarray(eef0["pos"], dtype=np.float64).reshape(3).tolist(),
                    "safe_pos_err_m": None,
                    "safe_ori_err_deg": None,
                    "eef_target_pos": target_pos.tolist(),
                    "eef_target_quat": grasp_ori.tolist(),
                    "back_m": float(back_m),
                    "pointing": pointing.tolist(),
                    "reach_preflight": {
                        "safe_shoulder_dist_m": None if math.isnan(safe_d_sh) else float(safe_d_sh),
                        "final_shoulder_dist_m": None if math.isnan(final_d_sh) else float(final_d_sh),
                        "limit_m": float(max(safe_limit, final_limit)),
                    },
                },
                "skip_next_move": True,
                "skip_grip_close": True,
                "grasped": False,
                "used_curobo": False,
                "error": msg,
            }
        return float("inf"), float("inf"), False

    obj_pos0 = None
    if target_obj is not None:
        try:
            p, _ = target_obj.get_position_orientation()
            obj_pos0 = _to_np(p)
        except Exception:
            obj_pos0 = None

    safe_pos_err = float("inf")
    safe_ori_err = float("inf")
    used_curobo_safe = False
    ctx.log(
        "    [to_safe_3d] Cartesian waypoint IK: mid waypoints XYZ-only + same branch; "
        f"safe gate pos<={_EEF_SAFE_POS_TOL_M*1000:.0f}mm ori<={_EEF_SAFE_ORI_TOL_DEG:.0f}°"
    )
    safe_is_final = bool(
        safe_plan_meta.get("final_as_safe")
        or abs(float(back_m)) <= 1e-6
    )
    safe_endpoint_reached = bool(
        to_safe_anchors and to_safe_anchors[-1].get("is_final")
    )
    if to_safe_anchors:
        ctx.log(
            f"    [to_safe_3d] play offline XYZ-only same-branch anchors "
            f"n={len(to_safe_anchors)}"
        )
        safe_pos_err, safe_ori_err = yield from _eef_play_joint_anchor_path(
            world,
            arm,
            to_safe_anchors,
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name="to_safe_xyz_joint",
            target_pos=safe_pos,
            target_quat=grasp_ori,
            pos_tol=_EEF_SAFE_POS_TOL_M,
            ori_tol_deg=_EEF_SAFE_ORI_TOL_DEG,
            max_frames_per_anchor=12,
            max_line_deviation_m=max(_EEF_SAFE_POS_TOL_M, 0.045),
            validate_intermediate_anchors=False,
            validate_final_anchor=False,
        )
        if not _pose_within_tol(
            safe_pos_err,
            safe_ori_err,
            _EEF_SAFE_POS_TOL_M,
            _EEF_SAFE_ORI_TOL_DEG,
        ):
            if _err_failed(safe_pos_err) or _err_failed(safe_ori_err):
                ctx.log(
                    "    [to_safe_3d] offline joint-anchor playback failed"
                )
            else:
                ctx.log(
                    "    [to_safe_3d] offline joint-anchor playback completed; "
                    "runtime safe pose deviation is diagnostic-only"
                )
    safe_pose_err = safe_pos_err
    safe_err = float(safe_pos_err)
    ori_now = _eef_ori_err_deg(world, arm, grasp_ori)
    approach_now = _eef_approach_err_deg(world, arm, grasp_ori)
    eef_safe = world.eef_pose(arm=arm)
    eef_safe_pos = np.asarray(eef_safe["pos"], dtype=np.float64).reshape(3)
    safe_measured_err = float(np.linalg.norm(eef_safe_pos - safe_pos))
    obj_pos_safe = None
    if target_obj is not None:
        try:
            p, _ = target_obj.get_position_orientation()
            obj_pos_safe = _to_np(p)
        except Exception:
            obj_pos_safe = None
    obj_displ_safe = None
    obj_moved_safe = None
    if obj_pos0 is not None and obj_pos_safe is not None:
        obj_displ_arr = np.asarray(obj_pos_safe, dtype=np.float64) - np.asarray(obj_pos0, dtype=np.float64)
        obj_displ_safe = obj_displ_arr.tolist()
        obj_moved_safe = float(np.linalg.norm(obj_displ_arr))

    ctx.log(
        f"  [exec_eef_pose] to_safe_pose 完成 "
        f"pos_stage={float(safe_pos_err)*1000:.0f}mm "
        f"pose_stage={float(safe_pose_err)*1000:.0f}mm "
        f"measured={safe_measured_err*1000:.0f}mm "
        f"ori={ori_now:.0f}° approach={approach_now:.0f}°"
    )

    safe_pose_within_tolerance = (
        safe_measured_err <= _EEF_SAFE_ENTRY_POS_TOL_M
        and ori_now <= _EEF_SAFE_ORI_TOL_DEG
    )
    safe_trajectory_completed = bool(
        to_safe_anchors
        and np.isfinite(float(safe_pos_err))
        and np.isfinite(float(safe_ori_err))
    )
    safe_meta = {
        "safe_target_pos": safe_pos.tolist(),
        "safe_actual_pos": eef_safe_pos.tolist(),
        "safe_pos_err_m": safe_measured_err,
        "safe_pos_stage_err_m": float(safe_pos_err),
        "safe_pose_stage_err_m": float(safe_pose_err),
        "safe_ori_err_deg": round(float(ori_now), 2),
        "safe_approach_err_deg": round(float(approach_now), 2),
        "eef_target_pos": target_pos.tolist(),
        "eef_target_quat": grasp_ori.tolist(),
        "requested_back_m": float(requested_back_m),
        "back_m": float(back_m),
        "back_m_adapted": abs(float(back_m) - float(requested_back_m)) > 1e-4,
        "safe_plan_meta": safe_plan_meta,
        "safe_endpoint_reached": safe_endpoint_reached,
        "safe_trajectory_completed": safe_trajectory_completed,
        "safe_pose_within_tolerance": safe_pose_within_tolerance,
        "pointing": pointing.tolist(),
        "object_pos_before": obj_pos0.tolist() if obj_pos0 is not None else None,
        "object_pos_after": obj_pos_safe.tolist() if obj_pos_safe is not None else None,
        "object_displacement": obj_displ_safe,
        "object_moved_dist": obj_moved_safe,
    }
    if stop_after_safe:
        return {
            "ok": bool(safe_trajectory_completed),
            "target": "safe_pose",
            "arm": arm,
            "pre_err": float(safe_measured_err),
            "cnt_err": float(safe_measured_err),
            "final_pos_err_m": float(safe_measured_err),
            "final_ori_err_deg": round(float(ori_now), 2),
            "safe_pose": safe_meta,
            "skip_next_move": True,
            "skip_grip_close": True,
            "grasped": False,
            "used_curobo": bool(used_curobo_safe),
            **({} if safe_trajectory_completed else {
                "error": (
                    "safe pose 关节轨迹执行失败；"
                    f"运行时位置误差 {safe_measured_err * 1000:.0f}mm，"
                    f"姿态误差 {ori_now:.1f}°"
                )
            }),
        }
    if not safe_trajectory_completed:
        ctx.log(
            "  ABORT safe pose 关节轨迹未完整执行，不进入 eef_pose"
        )
        return float(safe_measured_err), float("inf"), False
    if not safe_pose_within_tolerance:
        ctx.log(
            f"  [exec_eef_pose] safe runtime pose deviation is diagnostic-only: "
            f"pos={safe_measured_err*1000:.0f}mm "
            f"ori={ori_now:.1f}° approach={approach_now:.1f}°；"
            "离线关节轨迹已完成，继续 safe→eef"
        )

    if safe_is_final:
        ctx.log(
            "  [exec_eef_pose] active_back=0cm: safe 已合并到 final pose，"
            "跳过 safe→eef 平移段"
        )
        cnt_err, cnt_ori_err, app_final = _eef_pose_err(
            world, arm, target_pos, grasp_ori
        )
        final_ok = (
            float(cnt_err) <= _EEF_FINAL_POS_TOL_M
            and float(cnt_ori_err) <= _EEF_FINAL_ORI_TOL_DEG
            and app_final <= _GRASP_ORI_GATE_DEG
        )
        if not final_ok:
            ctx.log(
                f"  ABORT eef_pose 未到位：pos={float(cnt_err)*1000:.0f}mm "
                f"ori={cnt_ori_err:.1f}° approach={app_final:.1f}°"
            )
            return float(safe_err), float("inf"), False
        ctx.log(
            f"  [exec_eef_pose] eef_pose 完成 "
            f"err={float(cnt_err)*1000:.0f}mm ori={cnt_ori_err:.0f}° "
            f"approach={app_final:.0f}°"
        )
        return float(safe_err), float(cnt_err), False

    ctx.log(
        "  [exec_eef_pose] safe→eef 平移段：same-quat translation，"
        "midpoints XYZ-only, final is stored plan-filter 6D q"
    )

    cnt_err = float("inf")
    cnt_ori_err = float("inf")
    cnt_app_after_kin = float("inf")

    if stored_final_q_np is not None:
        cnt_err, cnt_ori_err = yield from _eef_play_stored_final_q_translation(
            world,
            arm,
            stored_final_q_np,
            target_pos,
            grasp_ori,
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name="eef_pose_final_q_translate",
            pos_tol=_EEF_FINAL_POS_TOL_M,
            ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
            max_line_deviation_m=max(0.030, float(back_m) * 0.35),
            max_frames=80,
        )
        cnt_app_after_kin = _eef_approach_err_deg(world, arm, grasp_ori)
    else:
        cnt_err, cnt_ori_err = yield from _eef_goto_6d_kinematic_plan(
            world, arm, target_pos, grasp_ori,
            gripper_cmd=gripper_cmd,
            ctx=ctx,
            stage_name="eef_pose_6d",
            pos_tol=_EEF_FINAL_POS_TOL_M,
            ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
            max_frames_per_waypoint=16,
            max_refine_attempts=1,
            intermediate_pos_only=True,
            final_q=stored_final_q_np,
        )
        cnt_app_after_kin = _eef_approach_err_deg(world, arm, grasp_ori)
    cnt_kin_pos_ok = (
        not _err_failed(cnt_err)
        and float(cnt_err) <= _EEF_FINAL_POS_TOL_M
        and float(cnt_ori_err) <= _EEF_FINAL_ORI_TOL_DEG
        and cnt_app_after_kin <= _GRASP_ORI_GATE_DEG
    )

    if not cnt_kin_pos_ok and not _pose_within_tol(
        cnt_err, cnt_ori_err, _EEF_FINAL_POS_TOL_M, _EEF_FINAL_ORI_TOL_DEG
    ):
        ctx.log(
            f"    [eef_pose_6d] final strict 6D failed; "
            f"pos={float(cnt_err)*1000:.0f}mm ori={float(cnt_ori_err):.1f}° "
            "not using non-line fallback"
        )
    if _err_failed(cnt_err):
        ctx.log("  WARN contact 未收敛")
        cnt_err = _err_json(cnt_err) or 9.99
    ori_final = _eef_ori_err_deg(world, arm, grasp_ori)
    app_final = _eef_approach_err_deg(world, arm, grasp_ori)
    final_ok = (
        float(cnt_err) <= _EEF_FINAL_POS_TOL_M
        and float(ori_final) <= _EEF_FINAL_ORI_TOL_DEG
        and app_final <= _GRASP_ORI_GATE_DEG
    )
    if not final_ok:
        ctx.log(
            f"  ABORT eef_pose 未到位：pos={float(cnt_err)*1000:.0f}mm "
            f"ori={ori_final:.1f}° approach={app_final:.1f}°"
        )
        return float(safe_err), float("inf"), False
    ctx.log(
        f"  [exec_eef_pose] eef_pose 完成 "
        f"err={float(cnt_err)*1000:.0f}mm ori={ori_final:.0f}° "
        f"approach={app_final:.0f}°"
    )
    return float(safe_err), float(cnt_err), False


def _curobo_move_failed(err) -> bool:
    import math
    return err is None or (isinstance(err, (float, int)) and math.isinf(float(err)))


def _execute_one_eef(
    ctx,
    last,
    cand: Dict[str, Any],
    arm_arg: str,
    back_m: float = 0.10,
    *,
    skip_next_move: bool = False,
    skip_grip_close: bool = False,
    stop_after_safe: bool = False,
    lock_gripper_cmd=None,
):
    """单个 eef candidate 的执行（generator）。
    return 值通过 StopIteration.value 取回。

    【exec_move 硬约束】执行全程**只允许动指定手臂 + 夹爪**。
    **禁止**移动底盘(base)或腰部(trunk)。导航/避障靠手臂收胸前再外伸，不靠腰或底盘。
    """
    world = ctx.world
    arm_eff = arm_arg or cand.get("arm") or "right"
    _prepare_legacy_7dof_motion(
        world, arm_eff, ctx=ctx, stage_name="exec_eef.prepare"
    )
    target = cand.get("target", "grasp")
    et = cand["eef_target"]
    next_move = np.array(cand.get("next_eef_move", [0.0, 0.0, 0.0]), dtype=np.float64)
    target_pos = np.array(et["pos"], dtype=np.float64)
    target_quat = et.get("quat")
    approach_raw = et.get("approach")
    if approach_raw is None and target_quat is not None:
        try:
            from behavior_interface.skills.grasp import _quat_to_mat

            approach = np.asarray(
                _quat_to_mat(_quat_normalize_xyzw(target_quat))[:, 2],
                dtype=np.float64,
            )
            ctx.log("   approach missing; derived from target_quat local +Z")
        except Exception:
            approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif approach_raw is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ctx.log("   approach missing and no target_quat; fallback to world -Z")
    else:
        approach = np.array(approach_raw, dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm > 1e-9:
        approach = approach / norm
    raw_grip_after = et.get("gripper_cmd", cand.get("gripper_cmd"))
    try:
        grip_after = float(raw_grip_after) if raw_grip_after is not None else -1.0
    except (TypeError, ValueError):
        grip_after = -1.0

    # 解析 object（用于状态验证）
    obj = _resolve_object_handle(
        world, last["object"].get("resolved_name") or last["object"].get("input")
    )

    if skip_next_move or skip_grip_close:
        mode_tag = []
        if skip_next_move:
            mode_tag.append("无 next_move")
        if skip_grip_close:
            mode_tag.append("不合爪")
        ctx.log(
            f"── exec eef #{cand.get('id')} target={target} label={cand.get('label')} arm={arm_eff} "
            f"[pose-only: {', '.join(mode_tag)}]\n"
            f"   eef target pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
            f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f}) grip={grip_after:+.0f}"
        )
    else:
        ctx.log(
            f"── exec eef #{cand.get('id')} target={target} label={cand.get('label')} arm={arm_eff}\n"
            f"   eef target pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
            f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f}) grip={grip_after:+.0f}\n"
            f"   next_eef_move=({next_move[0]:+.3f},{next_move[1]:+.3f},{next_move[2]:+.3f})"
        )

    # 起点 eef 朝向
    eef0 = world.eef_pose(arm=arm_eff)
    eef0_quat = np.asarray(eef0["quat"], dtype=np.float64)
    # 若 candidate 显式给了目标 quat（例如 open/close 模式），用它做 6-DOF IK 目标
    # 否则保留 eef 当前朝向作为目标（grasp/push 老逻辑）
    force_dls_for_adapted_quat = False
    if target_quat is not None:
        grasp_ori = np.asarray(target_quat, dtype=np.float64)
        ctx.log(f"   target_quat=({grasp_ori[0]:+.3f},{grasp_ori[1]:+.3f},"
                f"{grasp_ori[2]:+.3f},{grasp_ori[3]:+.3f}) (from candidate)")
        plan_arm = str(cand.get("arm") or "").lower().strip()
        if plan_arm and arm_eff in ("left", "right") and plan_arm != arm_eff:
            # The stored EEF pose is already a world-frame 6D pose.  The
            # candidate arm is only provenance/default routing; changing the
            # quaternion here corrupts the requested pose.
            meta_for_adapt = cand.setdefault("meta", {})
            meta_for_adapt["exec_arm_quat_adaptation"] = {
                "applied": False,
                "reason": "disabled_world_frame_pose",
                "plan_arm": plan_arm,
                "exec_arm": arm_eff,
            }
            ctx.log(
                f"   [arm-adapt] disabled: keep world-frame target_quat unchanged "
                f"(plan_arm={plan_arm}, exec_arm={arm_eff})"
            )
    else:
        grasp_ori = eef0_quat
    ctx.log(f"   eef0=({eef0['pos'][0]:.3f},{eef0['pos'][1]:.3f},{eef0['pos'][2]:.3f})")

    exec_seq = (cand.get("meta") or {}).get("exec_sequence", "move_then_close")
    move_only = bool(target == "move_eef" or exec_seq == "move_only")
    # 初始夹爪与 exec_move 对齐；pose-only / move-only 默认锁住进入时指位。
    if target == "grasp":
        # Grasp execution must always keep the fingers open until the final
        # 6D EEF target has been reached and verified immediately before close.
        pre_grip = +1.0
    elif exec_seq == "close_then_move":
        pre_grip = -1.0
    else:
        pre_grip = +1.0 if target in ("grasp", "open") else -1.0
    if target == "close" and exec_seq != "close_then_move":
        pre_grip = +1.0
    grip_cmd_motion = pre_grip
    grip_q_before = _read_finger_qpos(world, arm_eff)
    locked_grip_override = _gripper_cmd_override(lock_gripper_cmd)
    if locked_grip_override is None and move_only:
        locked_grip_override = _current_gripper_qpos_cmd(world, arm_eff)
    if locked_grip_override is not None:
        pre_grip = locked_grip_override
        grip_cmd_motion = locked_grip_override
        grip_q_init = grip_q_before
        ctx.log(
            f"   finger qpos locked for pose/move-only: {grip_q_before}; "
            f"cmd={[round(float(x), 4) for x in locked_grip_override]} "
            "skip pre-grip/open-close"
        )
    else:
        pre_cmd = _gripper_limit_cmd(world, arm_eff, open_gripper=(float(pre_grip) > 0.0))
        open_frames = 6
        for _ in range(open_frames):
            yield world.make_action(**{f"gripper_{arm_eff}": pre_cmd})
        grip_q_init = _read_finger_qpos(world, arm_eff)
        ctx.log(
            f"   finger qpos before pre-grip: {grip_q_before}; "
            f"after: {grip_q_init} frames={open_frames}"
        )

    # 记录初始物体状态（用于验证）
    obj_z0 = _read_obj_z(obj)
    obj_pos0 = None
    open_val0 = None
    if obj is not None:
        try:
            p, _ = obj.get_position_orientation()
            obj_pos0 = _to_np(p)
        except Exception:
            obj_pos0 = None
        try:
            from omnigibson.object_states.open_state import Open
            if Open in obj.states:
                open_val0 = bool(obj.states[Open].get_value())
        except Exception:
            open_val0 = None

    # plan_eef_v2 / grasp_obj 带目标 quat：优先 cuRobo；OOM/失败时回退 DLS IK
    use_curobo = (et.get("quat") is not None) and (target in ("open", "close", "grasp"))
    if use_curobo:
        try:
            from behavior_interface.skills.grasp import _curobo_gpu_low_mem
            if _curobo_gpu_low_mem():
                ctx.log("  WARN GPU 显存紧张，仍尝试 cuRobo low_mem；失败则 DLS IK")
        except Exception:
            pass
    if force_dls_for_adapted_quat and use_curobo:
        ctx.log("  [arm-adapt] 使用 DLS/kinematic 执行已验证的左右手自适应 quaternion")
        use_curobo = False

    meta_exec = cand.get("meta") or {}
    # 两段式避障是 exec 层逻辑：只需 eef pose + next_move，自动后退算 safe pose 再两段避障，
    # 与用哪个 plan 函数（grasp_obj / grasp_point / …）无关。任何带朝向(quat)的 grasp 都走两段式。
    lift_topdown = (target == "grasp" and et.get("quat") is not None)
    _curobo_attempts = 32 if target == "grasp" else 20
    _curobo_timeout = 10.0 if target == "grasp" else 5.0

    pose_only_direct = bool(skip_next_move and skip_grip_close and lift_topdown)

    if move_only:
        stored_press_trajectory_any = _stored_press_gpu_move_trajectory(cand)
        if stored_press_trajectory_any is not None:
            stored_trajectory_arm = str(
                stored_press_trajectory_any.get("arm") or ""
            ).lower().strip()
            if stored_trajectory_arm != arm_eff:
                return {
                    "ok": False,
                    "eef_id": cand.get("id"),
                    "label": cand.get("label"),
                    "target": target,
                    "arm": arm_eff,
                    "gpu_ik_trajectory": True,
                    "error": (
                        "plan_press_point GPU 轨迹锁定 "
                        f"arm={stored_trajectory_arm}，拒绝用 arm={arm_eff} 执行"
                    ),
                }
            return (yield from _eef_play_stored_press_gpu_trajectory(
                world,
                arm_eff,
                stored_press_trajectory_any,
                gripper_cmd=grip_cmd_motion,
                ctx=ctx,
                cand=cand,
                target_pos=target_pos,
                target_quat=grasp_ori,
                stop_after_safe=bool(stop_after_safe),
            ))

        line_tol = 0.020
        stored_move_final_q = _stored_filter_q_for_arm(cand, arm_eff)
        stored_move_final_q_np = None
        if stored_move_final_q is not None:
            try:
                stored_move_final_q_np = np.asarray(
                    stored_move_final_q, dtype=np.float64
                ).reshape(7)
                ctx.log(
                    "  [move_only] using plan-validated final q "
                    f"arm={arm_eff}"
                )
            except Exception:
                stored_move_final_q_np = None
        if (
            stored_move_final_q_np is None
            and _candidate_requires_stored_final_q(cand)
        ):
            ctx.log(
                "  ABORT move_only strict IK plan requires its stored final q; "
                "not running a fresh final IK branch"
            )
            return {
                "ok": False,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": target,
                "arm": arm_eff,
                "error": (
                    f"move_only 严格 IK 计划缺少 arm={arm_eff} 的存储最终关节解，"
                    "拒绝重新求解或换手执行"
                ),
            }
        ctx.log(
            f"  [move_only] straight-line same-branch 6D anchors "
            f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
            f"line_tol={line_tol*1000:.0f}mm"
        )
        pre_err = 0.0
        move_anchors, move_anchor_meta = _eef_plan_move_only_joint_anchors(
            world,
            arm_eff,
            target_pos,
            grasp_ori,
            final_q=stored_move_final_q_np,
            ctx=ctx,
            stage_name="move_only_line_offline",
            back_m=float(back_m),
        )
        if move_anchors:
            cnt_err, cnt_ori_err = yield from _eef_play_joint_anchor_path(
                world,
                arm_eff,
                move_anchors,
                gripper_cmd=grip_cmd_motion,
                ctx=ctx,
                stage_name="move_only_line_joint",
                target_pos=target_pos,
                target_quat=grasp_ori,
                pos_tol=_EEF_FINAL_POS_TOL_M,
                ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
                max_frames_per_anchor=42,
                max_line_deviation_m=line_tol,
                repair_line_waypoints=True,
            )
        else:
            ctx.log(
                "    [move_only_line] offline joint-anchor plan failed: "
                f"{move_anchor_meta.get('error', 'unknown')} "
                f"accepted={move_anchor_meta.get('accepted', 0)} "
                f"dropped={len(move_anchor_meta.get('dropped') or [])}; "
                "no curved fallback"
            )
            cnt_err, cnt_ori_err = float("inf"), float("inf")
        if _err_failed(cnt_err):
            ctx.log(
                "  [move_only] straight-line anchor execution failed; "
                "trying executor straight-corridor final-pose repair"
            )
            try:
                repair_start_pose = world.eef_pose(arm=arm_eff)
                repair_start_pos = np.asarray(
                    repair_start_pose["pos"], dtype=np.float64
                ).reshape(3)
            except Exception:
                repair_start_pos = target_pos.copy()
            try:
                yield from _eef_snap_6d_line_anchor(
                    world,
                    arm_eff,
                    target_pos,
                    grasp_ori,
                    gripper_cmd=grip_cmd_motion,
                    ctx=ctx,
                    stage_name="move_only_final_snap_repair",
                    pos_tol=_EEF_FINAL_POS_TOL_M,
                    ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
                    line_start_pos=repair_start_pos,
                    line_target_pos=target_pos,
                    max_line_deviation_m=line_tol,
                    line_deviation_margin_m=0.020,
                    nominal_q=_arm_qpos_now(world, arm_eff),
                    max_q_delta_rad=0.80,
                )
            except Exception as snap_exc:
                ctx.log(
                    "  [move_only] final snap repair exception: "
                    f"{type(snap_exc).__name__}: {snap_exc}"
                )
            try:
                final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
                    world, arm_eff, target_pos, grasp_ori
                )
            except Exception:
                final_pos_err, final_ori_err, final_app_err = float("inf"), 999.0, 999.0
            if (
                float(final_pos_err) > _EEF_FINAL_POS_TOL_M
                or float(final_ori_err) > _EEF_FINAL_ORI_TOL_DEG
            ):
                ctx.log(
                    "  [move_only] DLS snap repair not enough; "
                    "trying cuRobo IK-only executor repair"
                )
                try:
                    yield from _eef_goto_6d_curobo_ik(
                        world,
                        arm_eff,
                        target_pos,
                        grasp_ori,
                        gripper_cmd=grip_cmd_motion,
                        ctx=ctx,
                        stage_name="move_only_final_curobo_repair",
                        pos_tol=_EEF_FINAL_POS_TOL_M,
                        ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
                        max_attempts=64,
                        timeout=8.0,
                        max_frames=120,
                        line_start_pos=repair_start_pos,
                        line_target_pos=target_pos,
                        max_line_deviation_m=line_tol,
                        line_deviation_margin_m=0.020,
                    )
                except Exception as cu_exc:
                    ctx.log(
                        "  [move_only] cuRobo final repair exception: "
                        f"{type(cu_exc).__name__}: {cu_exc}"
                    )
                try:
                    final_pos_err, final_ori_err, final_app_err = _eef_pose_err(
                        world, arm_eff, target_pos, grasp_ori
                    )
                except Exception:
                    final_pos_err, final_ori_err, final_app_err = float("inf"), 999.0, 999.0
            if (
                float(final_pos_err) <= _EEF_FINAL_POS_TOL_M
                and float(final_ori_err) <= _EEF_FINAL_ORI_TOL_DEG
            ):
                ctx.log(
                    "  [move_only] final snap repair recovered "
                    f"pos={float(final_pos_err)*1000:.1f}mm "
                    f"ori={float(final_ori_err):.2f}° "
                    f"approach={float(final_app_err):.2f}°"
                )
                cnt_err = float(final_pos_err)
                cnt_ori_err = float(final_ori_err)
            else:
                ctx.log(
                    "  [move_only] final snap repair did not meet final tolerance "
                    f"pos={float(final_pos_err)*1000:.1f}mm>{_EEF_FINAL_POS_TOL_M*1000:.0f}mm "
                    f"ori={float(final_ori_err):.2f}°>{_EEF_FINAL_ORI_TOL_DEG:.1f}°"
                )
                return {
                    "ok": False,
                    "eef_id": cand.get("id"),
                    "label": cand.get("label"),
                    "target": target,
                    "arm": arm_eff,
                    "pre_err": _err_json(pre_err),
                    "cnt_err": None,
                    "end_err": None,
                    "final_pos_err_m": _err_json(final_pos_err),
                    "final_ori_err_deg": round(float(final_ori_err), 2),
                    "final_approach_err_deg": round(float(final_app_err), 2),
                    "used_curobo": False,
                    "line_constraint_failed": True,
                    "error": (
                        "move_only 直线执行失败：无法沿当前点→目标点建立/回放同分支 IK anchors；"
                        "final snap repair 未满足最终误差"
                    ),
                }
        else:
            try:
                from behavior_interface.skills.move_eef import kinematic_snap_eef_arm

                snap_report = kinematic_snap_eef_arm(
                    ctx,
                    world,
                    arm_eff,
                    target_pos,
                    grasp_ori,
                    pos_tol=_EEF_FINAL_POS_TOL_M,
                    ori_tol_deg=_EEF_FINAL_ORI_TOL_DEG,
                )
                if isinstance(snap_report, dict):
                    ctx.log(f"  [move_only] final kinematic_snap={snap_report}")
            except Exception as snap_exc:
                ctx.log(f"  [move_only] kinematic_snap skipped: {type(snap_exc).__name__}: {snap_exc}")
        cnt_target_pos = target_pos
        use_curobo = False
    elif pose_only_direct:
        direct_result = yield from _eef_direct_safe_approach(
            world, arm_eff, target_pos, grasp_ori, pre_grip, ctx,
            approach=approach, target_obj=obj, back_m=back_m,
            prefer_curobo=use_curobo,
            stop_after_safe=stop_after_safe,
            candidate=cand,
        )
        if isinstance(direct_result, dict):
            direct_result.setdefault("eef_id", cand.get("id"))
            direct_result.setdefault("label", cand.get("label"))
            direct_result.setdefault("target", "safe_pose")
            direct_result.setdefault("arm", arm_eff)
            return direct_result
        pre_err, cnt_err, use_curobo = direct_result
        cnt_target_pos = target_pos
        use_curobo = False
    elif lift_topdown:
        approach_result = yield from _grasp_lift_topdown_approach(
            world, arm_eff, target_pos, grasp_ori, pre_grip, ctx, use_curobo,
            approach=approach, target_obj=obj, back_m=back_m, candidate=cand,
        )
        if isinstance(approach_result, dict):
            approach_result.setdefault("eef_id", cand.get("id"))
            approach_result.setdefault("label", cand.get("label"))
            approach_result.setdefault("target", target)
            approach_result.setdefault("arm", arm_eff)
            approach_result.setdefault("grasped", False)
            approach_result.setdefault("object_dz", 0.0)
            return approach_result
        pre_err, cnt_err, use_curobo = approach_result
        cnt_target_pos = target_pos
        import math as _math
        if _math.isinf(float(cnt_err)):
            ctx.log("  [grasp_obj exec] 接近/接触失败，跳过闭爪与上抬")
            return {
                "ok": False,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": target,
                "arm": arm_eff,
                "pre_err": float(pre_err),
                "cnt_err": float("inf"),
                "end_err": float("inf"),
                "final_pos_err_m": float("inf"),
                "used_curobo": bool(use_curobo),
                "grasped": False,
                "object_dz": 0.0,
                "error": "抓取失败：手臂无法避障到达预抓取/最终位姿（请 move_to_object 重新靠近）",
            }
    else:
        # ─── Stage A：pre-approach (沿 -approach 偏 15cm) ───
        pre_pos = target_pos - 0.25 * approach
        ctx.log(f"  → pre ({pre_pos[0]:.3f},{pre_pos[1]:.3f},{pre_pos[2]:.3f})")
        if use_curobo:
            from behavior_interface.skills.grasp import _eef_goto_world_curobo
            pre_err = yield from _eef_goto_world_curobo(
                world, arm_eff, pre_pos, target_world_quat=grasp_ori.tolist(),
                gripper_cmd=pre_grip, ctx=ctx, stage_name="pre",
                max_attempts=_curobo_attempts, timeout=_curobo_timeout,
                allow_contact=False,
                lock_body=True,
                lock_other_arm=True,
            )
            if _curobo_move_failed(pre_err):
                ctx.log("  WARN curobo pre 失败 → 回退 DLS IK")
                use_curobo = False
                pre_err = yield from _eef_goto_world(
                    world, arm_eff, pre_pos, target_world_quat=grasp_ori,
                    max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_PRE_TOL,
                    gripper_cmd=pre_grip, ctx=ctx, stage_name="pre",
                )
        else:
            pre_err = yield from _eef_goto_world(
                world, arm_eff, pre_pos, target_world_quat=grasp_ori,
                max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_PRE_TOL,
                gripper_cmd=pre_grip, ctx=ctx, stage_name="pre",
            )

        # ─── Stage B：contact ───
        cnt_target_pos = target_pos
        ctx.log(
            f"  → contact eef=({cnt_target_pos[0]:.3f},{cnt_target_pos[1]:.3f},"
            f"{cnt_target_pos[2]:.3f})"
        )
        if use_curobo:
            from behavior_interface.skills.grasp import _eef_goto_world_curobo
            cnt_err = yield from _eef_goto_world_curobo(
                world, arm_eff, cnt_target_pos, target_world_quat=grasp_ori.tolist(),
                gripper_cmd=pre_grip, ctx=ctx, stage_name="cnt",
                max_attempts=_curobo_attempts, timeout=_curobo_timeout,
                allow_contact=True,
                lock_body=True,
                lock_other_arm=True,
            )
            if _curobo_move_failed(cnt_err):
                ctx.log("  WARN curobo cnt 失败 → 回退 DLS IK")
                use_curobo = False
                cnt_err = yield from _eef_goto_world(
                    world, arm_eff, cnt_target_pos, target_world_quat=grasp_ori,
                    max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_TGT_TOL,
                    gripper_cmd=pre_grip, ctx=ctx, stage_name="cnt",
                )
        else:
            cnt_err = yield from _eef_goto_world(
                world, arm_eff, cnt_target_pos, target_world_quat=grasp_ori,
                max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_TGT_TOL,
                gripper_cmd=pre_grip, ctx=ctx, stage_name="cnt",
            )

    # ─── Stage C：设置目标 gripper_cmd（grasp / open / close 时都 push_in 让 finger 触到）───
    grip_q_before = _read_finger_qpos(world, arm_eff)
    if move_only:
        grip_after_cmd = grip_cmd_motion
    elif target in ("grasp", "close"):
        grip_after_cmd = _gripper_limit_cmd(world, arm_eff, open_gripper=False)
    elif target == "open":
        grip_after_cmd = _gripper_limit_cmd(world, arm_eff, open_gripper=True)
    else:
        grip_after_cmd = grip_after
    if target == "grasp":
        ctx.log(
            "  [exec-grip] use full close command for assisted grasp; "
            f"cmd={[round(float(x), 4) for x in _gripper_cmd_override(grip_after_cmd) or [float(grip_after_cmd)]]}"
        )

    # ── 诊断：测量 wrist / finger / handle 真实位置（详细版，含各 finger link 位置）──
    try:
        eef_pose_now = world.eef_pose(arm=arm_eff)
        wrist_pos = np.asarray(eef_pose_now["pos"], dtype=np.float64)
        finger_link_names_diag = world.robot.finger_link_names.get(arm_eff, [])
        finger_centers = []
        for fln in finger_link_names_diag:
            lk = world.robot.links.get(fln)
            if lk is not None:
                p, _ = lk.get_position_orientation()
                fp = np.asarray(_to_np(p), dtype=np.float64)
                finger_centers.append(fp)
                ctx.log(f"  [diag-finger] {fln}: world=({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f})"
                        f" wrist→f=({(fp-wrist_pos).round(3)})")
        finger_center_actual = np.mean(finger_centers, axis=0) if finger_centers else wrist_pos
        wrist_to_finger = finger_center_actual - wrist_pos
        finger_to_handle = target_pos - finger_center_actual
        ctx.log(f"  [diag] wrist@{wrist_pos.round(3)} finger@{finger_center_actual.round(3)} "
                f"handle@{target_pos.round(3)} wrist→finger={wrist_to_finger.round(3)} "
                f"finger→handle={finger_to_handle.round(3)} |fh|={np.linalg.norm(finger_to_handle)*1000:.0f}mm")
    except Exception as _e:
        ctx.log(f"  [diag] finger 位置查询失败: {_e}")

    if move_only:
        ctx.log("  [exec] move_only: keep gripper state; no open/close stage")
        yield from _hold_pose(
            world, arm_eff, cnt_target_pos, grasp_ori,
            n_frames=3, gripper_cmd=grip_cmd_motion,
        )
    elif skip_grip_close:
        ctx.log("  [exec] 保持预夹爪状态，不合爪（pose-only）")
        yield from _hold_pose(
            world, arm_eff, cnt_target_pos, grasp_ori,
            n_frames=3, gripper_cmd=grip_cmd_motion if grip_cmd_motion is not None else pre_grip,
        )
    elif target in ("grasp", "open", "close"):
        final_pos_preclose, final_ori_preclose, final_app_preclose = _eef_pose_err(
            world, arm_eff, cnt_target_pos, grasp_ori
        )
        preclose_final_ok = (
            float(final_pos_preclose) <= _EEF_FINAL_POS_TOL_M
            and float(final_ori_preclose) <= _EEF_FINAL_ORI_TOL_DEG
            and float(final_app_preclose) <= _GRASP_ORI_GATE_DEG
        )
        ctx.log(
            "  [pre-close final 6D gate] "
            f"pos={float(final_pos_preclose)*1000:.1f}mm "
            f"ori={float(final_ori_preclose):.2f}° "
            f"approach={float(final_app_preclose):.2f}° "
            f"ok={bool(preclose_final_ok)}"
        )
        if target == "grasp" and not preclose_final_ok:
            ctx.log("  ABORT final 6D 未到位，禁止合爪")
            return {
                "ok": False,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": target,
                "arm": arm_eff,
                "pre_err": _err_json(pre_err),
                "cnt_err": _err_json(final_pos_preclose),
                "end_err": None,
                "final_pos_err_m": _err_json(final_pos_preclose),
                "final_ori_err_deg": round(float(final_ori_preclose), 2),
                "final_approach_err_deg": round(float(final_app_preclose), 2),
                "used_curobo": bool(use_curobo),
                "grasped": False,
                "object_dz": 0.0,
                "error": (
                    f"抓取失败：final 6D pose 未到位，禁止合爪 "
                    f"(pos={float(final_pos_preclose)*1000:.1f}mm, "
                    f"ori={float(final_ori_preclose):.2f}°, "
                    f"approach={float(final_app_preclose):.2f}°)"
                ),
            }
        if target == "grasp" and float(cnt_err) > 0.08:
            ctx.log(
                f"  ABORT contact 残差={float(cnt_err)*1000:.0f}mm>80mm，"
                "跳过闭爪/上抬"
            )
            return {
                "ok": False,
                "eef_id": cand.get("id"),
                "label": cand.get("label"),
                "target": target,
                "arm": arm_eff,
                "pre_err": _err_json(pre_err),
                "cnt_err": _err_json(cnt_err),
                "end_err": None,
                "final_pos_err_m": _err_json(cnt_err),
                "used_curobo": bool(use_curobo),
                "grasped": False,
                "object_dz": 0.0,
                "error": (
                    f"抓取失败：接触点未到位（残差 {float(cnt_err)*1000:.0f}mm）"
                ),
            }
        obj_pos_pre_close = _object_root_pos_np(obj)
        if obj_pos0 is not None and obj_pos_pre_close is not None:
            pre_close_delta = obj_pos_pre_close - np.asarray(obj_pos0, dtype=np.float64)
            pre_close_drift = float(np.linalg.norm(pre_close_delta))
            ctx.log(
                "  [exec-drift] before close object_delta="
                f"({pre_close_delta[0]:+.3f},{pre_close_delta[1]:+.3f},{pre_close_delta[2]:+.3f})m "
                f"|Δ|={pre_close_drift*1000:.1f}mm"
            )
            if target == "grasp" and pre_close_drift > 0.015:
                ctx.log(
                    "  [exec-drift] object moved before close, but final EEF is at "
                    "plan pose; continue to close/lift and let grasp result decide"
                )
        # cnt 阶段已到 plan pose。闭爪时只锁腕部姿态，夹爪目标慢速 ramp，
        # 避免 absolute position gripper 一步打到下限造成小物体穿模。
        if target == "grasp":
            _log_grasp_contact_geometry(ctx, world, arm_eff, obj, label="before_close")
            _log_assisted_grasp_diag(ctx, world, arm_eff, obj, label="before_close")
        ctx.log("  close gripper slow-ramp (hold wrist)")
        close_report = yield from _slow_close_gripper_hold_wrist(
            world, arm_eff, cnt_target_pos, grasp_ori,
            q_closed_cmd=grip_after_cmd,
            ctx=ctx,
            n_ramp=6 if target == "grasp" else 12,
            n_hold=_assisted_grasp_hold_frames(world) if target == "grasp" else 8,
        )
        if (
            target == "grasp"
            and isinstance(close_report, dict)
            and close_report.get("mode") == "effort"
        ):
            # The effort pin owns the gripper from this point onward. Omitting
            # qpos overrides is what keeps the official AG constraint alive.
            grip_after_cmd = None
            if not close_report.get("success"):
                grip_q_failed = _read_finger_qpos(world, arm_eff)
                ctx.log("  ABORT effort close did not establish official assisted grasp; skip lift")
                return {
                    "ok": False,
                    "eef_id": cand.get("id"),
                    "label": cand.get("label"),
                    "target": target,
                    "arm": arm_eff,
                    "pre_err": _err_json(pre_err),
                    "cnt_err": _err_json(cnt_err),
                    "end_err": None,
                    "final_pos_err_m": _err_json(cnt_err),
                    "used_curobo": bool(use_curobo),
                    "grasped": False,
                    "object_dz": 0.0,
                    "finger_qpos": grip_q_failed,
                    "gripper_close": close_report,
                    "error": "抓取失败：低力闭爪未建立官方 assisted-grasp 连接，已停止试提",
                }
        obj_pos_post_close = _object_root_pos_np(obj)
        if obj_pos_pre_close is not None and obj_pos_post_close is not None:
            close_delta = obj_pos_post_close - obj_pos_pre_close
            close_delta_norm = float(np.linalg.norm(close_delta))
            ctx.log(
                "  [exec-drift] after close object_delta="
                f"({close_delta[0]:+.3f},{close_delta[1]:+.3f},{close_delta[2]:+.3f})m "
                f"|Δ|={close_delta_norm*1000:.1f}mm"
            )
        if target == "grasp":
            _log_grasp_contact_geometry(ctx, world, arm_eff, obj, label="after_close")
            _log_assisted_grasp_diag(ctx, world, arm_eff, obj, label="after_close")
            held = _current_assisted_grasp_object(world, arm_eff)
            ag_ready = held is not None and _same_og_object(held, obj)
            ctx.log(
                "  [exec-grip] core assisted grasp "
                f"held={_ag_obj_name(held)} target={_ag_obj_name(obj)} ok={bool(ag_ready)} "
                "source=physx_action_window"
            )
    else:
        yield from _hold_pose(world, arm_eff, target_pos, grasp_ori,
                              n_frames=_GRIP_FR, gripper_cmd=grip_after_cmd)
    grip_q_after = _read_finger_qpos(world, arm_eff)
    ctx.log(f"  finger qpos: before={grip_q_before} after={grip_q_after} (0=closed)")

    # ─── Stage D：next_eef_move（exec_eef_pose 跳过：仅胸前→到位→合爪）───
    end_err = 0.0  # place / 未知 target 无 Stage D 移动，预置以免 return 时未定义
    lift_goal_pos = np.asarray(cnt_target_pos, dtype=np.float64)
    if skip_next_move or move_only:
        ctx.log("  [exec] skip next_eef_move（move_only/无抬升/推拉/开门弧）")
    elif target == "grasp":
        if lift_topdown:
            # next_move 就是期望的抬升增量（如纯 +Z），直接叠加，不再沿 approach 侧偏
            end_pos = target_pos + next_move
            lift_allow_contact = True
        else:
            end_pos = target_pos + 0.03 * approach + next_move
            lift_allow_contact = False
        lift_goal_pos = end_pos
        ctx.log(f"  → lift {end_pos.tolist()}")
        if lift_topdown:
            # 两段式抓取：抓着物体沿 next_move 抬起（DLS 纯手臂 make_action，底盘/躯干绝不移动），
            # 用物体 z 增加判定抓取是否成功。
            stored_lift_seed = _stored_filter_q_for_arm(cand, arm_eff)
            lifted_fast = False
            if stored_lift_seed is not None:
                lifted_fast, end_err = yield from _eef_fast_lift_from_final_q(
                    world,
                    arm_eff,
                    stored_lift_seed,
                    target_pos,
                    end_pos,
                    grasp_ori,
                    gripper_cmd=grip_after_cmd,
                    ctx=ctx,
                )
            if not lifted_fast:
                end_err = yield from _eef_goto_world(
                    world, arm_eff, end_pos, target_world_quat=grasp_ori,
                    max_steps=40, pos_tol=_STAGE_TGT_TOL,
                    gripper_cmd=grip_after_cmd, ctx=ctx, stage_name="lift",
                    disable_stuck_check=True,
                    ori_weight=0.0, ori_tol=3.14,
                    max_dq_per_step=0.08,
                    max_dx_per_step=0.05,
                )
        elif use_curobo:
            from behavior_interface.skills.grasp import _eef_goto_world_curobo
            end_err = yield from _eef_goto_world_curobo(
                world, arm_eff, end_pos, target_world_quat=grasp_ori.tolist(),
                gripper_cmd=grip_after_cmd, ctx=ctx, stage_name="lift",
                max_attempts=24, timeout=10.0,
                allow_contact=lift_allow_contact,
                lock_body=True,
                lock_other_arm=True,
            )
        else:
            end_err = yield from _eef_goto_world(
                world, arm_eff, end_pos, target_world_quat=grasp_ori,
                max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_TGT_TOL,
                gripper_cmd=grip_after_cmd, ctx=ctx, stage_name="lift",
            )

    elif target == "push":
        end_pos = target_pos + next_move
        ctx.log(f"  → push {end_pos.tolist()}")
        # push 也用普通 IK，但允许较大容差 + stuck 不退出
        end_err = yield from _eef_goto_world(
            world, arm_eff, end_pos, target_world_quat=grasp_ori,
            max_steps=_STAGE_MAX_STEPS, pos_tol=0.05,
            gripper_cmd=grip_after_cmd, ctx=ctx, stage_name="push",
            disable_stuck_check=True,
        )

    elif not skip_next_move and target in ("open", "close"):
        # 弧线开门：先尝试 Jacobian 笛卡尔空间推动（必经过门面产生接触力），
        # 若物理接触无效，自动切换到运动学辅助（joint.set_pos + cuRobo IK 追踪）。
        from behavior_interface.skills.grasp import _eef_goto_world_curobo
        meta = cand.get("meta", {})
        joint_name = meta.get("joint_name")
        closed_q = float(meta.get("closed_q", 0.0))
        open_q = float(meta.get("open_q", 0.0))
        hinge = np.asarray(meta["hinge_world"], dtype=np.float64)
        axis = np.asarray(meta["axis_world"], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        handle_closed = np.asarray(meta["handle_closed_world"], dtype=np.float64)
        handle_mid_closed = np.asarray(
            meta.get("handle_mid_world", meta["handle_closed_world"]), dtype=np.float64)
        tangent_closed = np.asarray(meta["tangent_closed_world"], dtype=np.float64)
        spread_sign = int(meta.get("spread_sign", +1))
        axis_off = float(meta.get("axis_off", 0.0))
        cur_q_init = float(meta.get("cur_q", closed_q))
        end_q = open_q if target == "open" else closed_q

        # 找 joint 对象（只读 q，不写）
        joint_obj = None
        if obj is not None:
            for j_name, joint in obj.joints.items():
                if joint.name == joint_name:
                    joint_obj = joint
                    break

        # 诊断：joint_obj 是否找到
        if joint_obj is None:
            ctx.log(f"  [WARN] joint_obj not found (joint_name={joint_name!r}), "
                    f"real_q will stay at cur_q_init={cur_q_init:.3f}")
        else:
            ctx.log(f"  [OK] joint_obj found: {joint_obj.name!r}, cur_q={float(joint_obj.get_state()[0]):.4f}")

        # 诊断：打印 EEF link 的实际世界位姿和手指朝向，确认 EEF Z 轴方向
        try:
            eef_pose_arc = world.eef_pose(arm=arm_eff)
            wrist_now = np.asarray(eef_pose_arc["pos"], dtype=np.float64)
            wrist_quat_now = np.asarray(eef_pose_arc["quat"], dtype=np.float64)  # xyzw
            # 计算 EEF Z 轴在世界系的方向（R * [0,0,1]）
            eef_z_world = _quat_rot(wrist_quat_now, np.array([0, 0, 1.0]))
            eef_z_world /= (np.linalg.norm(eef_z_world) + 1e-9)
            # 手指链接位置
            f_links = world.robot.finger_link_names.get(arm_eff, [])
            f_positions = []
            for fln in f_links:
                lk = world.robot.links.get(fln)
                if lk:
                    fp, _ = lk.get_position_orientation()
                    f_positions.append(_to_np(fp).reshape(3))
            f_center = np.mean(f_positions, axis=0) if f_positions else wrist_now
            wrist2finger = f_center - wrist_now
            ctx.log(f"  [arc-diag] wrist={wrist_now.round(3)} eef_Z_world={eef_z_world.round(3)}")
            ctx.log(f"  [arc-diag] wrist→finger={wrist2finger.round(4)} "
                    f"handle_mid={handle_mid_closed.round(3)} "
                    f"wrist→handle={handle_mid_closed-wrist_now:.3f}")
            # 判断 EEF Z 是朝向把手还是远离把手
            wrist2handle = handle_mid_closed - wrist_now
            wrist2handle /= (np.linalg.norm(wrist2handle) + 1e-9)
            dot_z_toward_handle = float(np.dot(eef_z_world, wrist2handle))
            ctx.log(f"  [arc-diag] dot(eef_Z, wrist→handle)={dot_z_toward_handle:.3f} "
                    f"(>0 means Z points toward handle ✓; <0 means BACKWARDS ✗)")
        except Exception as _efd:
            ctx.log(f"  [arc-diag] err: {_efd}")

        # 弧半径
        hh = handle_mid_closed - hinge
        hh_perp = hh - np.dot(hh, axis) * axis
        arc_radius = float(np.linalg.norm(hh_perp))
        ctx.log(f"  → arc {target} from q={cur_q_init:.3f} to q={end_q:.3f} "
                f"arc_radius={arc_radius:.3f}m hinge={hinge.round(3)} handle={handle_mid_closed.round(3)}")

        # 每段推进角度（小幅推进，让物理跟得上）
        step_rad = 0.087   # 5°
        max_arc_steps = int(abs(end_q - cur_q_init) / step_rad) + 4
        max_arc_steps = max(max_arc_steps, 8)

        def _compute_target_quat_at(approach_qi: np.ndarray) -> np.ndarray:
            """根据当前 approach 方向（机器人→把手）重算 eef target quat。
            approach_qi = eef +Z（指向目标），finger_spread 沿 axis。"""
            z_axis_eef = approach_qi / (np.linalg.norm(approach_qi) + 1e-9)
            y_raw = spread_sign * axis
            y_axis_eef = y_raw - np.dot(y_raw, z_axis_eef) * z_axis_eef
            ny = np.linalg.norm(y_axis_eef)
            if ny < 1e-3:
                y_raw = np.array([0.0, 0.0, 1.0]) * spread_sign
                y_axis_eef = y_raw - np.dot(y_raw, z_axis_eef) * z_axis_eef
                ny = np.linalg.norm(y_axis_eef) + 1e-9
            y_axis_eef = y_axis_eef / ny
            x_axis_eef = np.cross(y_axis_eef, z_axis_eef)
            R = np.column_stack([x_axis_eef, y_axis_eef, z_axis_eef])
            return _mat_to_quat_xyzw(R)

        # 计算机器人位置（用于 arc 每步更新 approach 方向）
        try:
            _robot_pos_arc = np.array(ctx.world.robot.get_position(), dtype=np.float64)
        except Exception:
            _robot_pos_arc = np.zeros(3)

        eef_target_qi = target_pos.copy()
        last_err = 0.0
        prev_hinge_q = cur_q_init
        stuck_arc_cnt = 0

        for si in range(max_arc_steps):
            # 读真实 hinge q
            real_q = cur_q_init
            if joint_obj is not None:
                try:
                    real_q = float(joint_obj.get_state()[0])
                except Exception:
                    pass

            # 到目标？
            margin = 0.05  # 5°
            if target == "open" and real_q >= end_q - margin:
                ctx.log(f"    arc done at step={si} real_q={real_q:.3f} ≥ {end_q - margin:.3f}")
                break
            if target == "close" and real_q <= end_q + margin:
                ctx.log(f"    arc done at step={si} real_q={real_q:.3f} ≤ {end_q + margin:.3f}")
                break

            # 卡死检查（放宽：open/close 前几步可以 real_q 没动，靠前馈推进）
            q_delta = abs(real_q - prev_hinge_q)
            if si > 4 and q_delta < 0.003:
                stuck_arc_cnt += 1
                if stuck_arc_cnt >= 8:
                    ctx.log(f"    arc stuck at step={si} real_q={real_q:.3f}（门拉不动）")
                    break
            else:
                stuck_arc_cnt = 0
            prev_hinge_q = real_q

            # 前馈推进：不管门是否真的转动，每步都递增 step_rad 个角度
            # 这样手臂持续向前施力（PD 目标始终超前于当前门的实际状态）
            # 若门已实际打开超过前馈位置，则以 real_q + step_rad 为准
            dir_sign = 1.0 if end_q > cur_q_init else -1.0
            ff_q = cur_q_init + dir_sign * (si + 1) * step_rad   # 前馈目标
            reactive_q = real_q + dir_sign * step_rad             # 反应式目标
            if dir_sign > 0:
                next_q = min(max(ff_q, reactive_q), end_q)
            else:
                next_q = max(min(ff_q, reactive_q), end_q)

            # arc 阶段始终追踪实际把手中点（handle_mid_closed），不使用 axis_off 偏移
            # 采样阶段 axis_off 是为了采样可达候选，但 arc 执行时应始终指向把手本体
            handle_qi = _rotate_around_axis(handle_mid_closed,
                                            hinge, axis, next_q - closed_q)
            # 算 q 对应的切向（绕轴旋转）
            tangent_qi = _rotate_around_axis(tangent_closed, np.zeros(3),
                                             axis, next_q - closed_q)
            tangent_qi = tangent_qi / (np.linalg.norm(tangent_qi) + 1e-9)

            # approach_qi = 机器人→把手 方向（在铰链垂直平面内），与 pre/cnt 保持一致
            r2h_qi = handle_qi - _robot_pos_arc
            r2h_qi[2] = 0.0
            r2h_qi -= np.dot(r2h_qi, axis) * axis
            r2h_norm_qi = np.linalg.norm(r2h_qi)
            if r2h_norm_qi > 0.01:
                approach_qi = r2h_qi / r2h_norm_qi
            else:
                approach_qi = approach  # fallback 到 pre/cnt 的 approach

            # wrist 在 handle 前方 0.10m（handle - approach * 0.10）
            # 保证 wrist 始终在门面外侧（y > door_face_y）
            # 前馈推进时随着 ff_q 增大，wrist target 也随着门的旋转逐步深入门面
            # → 产生递增的接触推力让门打开
            # eef_link = fingertip center，wrist 直接在把手弧线位置，由碰撞产生持续推力
            # 前馈目标超前实际门位置，保证 eef 总在门"前方"（产生推力）
            _EEF_TO_FINGER_TIP_DIST_ARC = 0.0
            eef_target_qi = handle_qi - _EEF_TO_FINGER_TIP_DIST_ARC * approach_qi
            target_quat_qi = _compute_target_quat_at(approach_qi)

            ctx.log(f"    [arc{si}] ff_q={ff_q:.3f} real_q={real_q:.3f} "
                    f"wrist_target=({eef_target_qi[0]:.3f},{eef_target_qi[1]:.3f},{eef_target_qi[2]:.3f})"
                    f" [phys]")

            # reach 检查（保守 0.97）
            sh = world.shoulder_pose(arm=arm_eff)
            sh_arr = np.array([sh["x"], sh["y"], sh["z"]])
            d_sh = float(np.linalg.norm(eef_target_qi - sh_arr))
            if d_sh > _ARM_MAX_REACH * 0.97:
                ctx.log(f"    arc step={si}: eef target too far from shoulder "
                        f"({d_sh:.3f}m > {_ARM_MAX_REACH*0.97:.3f}m), stopping")
                break

            tquat_list = (target_quat_qi.tolist()
                          if hasattr(target_quat_qi, 'tolist') else list(target_quat_qi))

            # ── 纯物理弧线：Jacobian 小步长渐进推力（每步 ≤3cm）──────────────────
            # finger 夹住把手后，Jacobian 缓慢推进，通过接触力带动门旋转。
            # 比 cuRobo "跳跃式" IK 更能稳定传递夹持力。
            last_err = yield from _eef_goto_world(
                world, arm_eff, eef_target_qi,
                target_world_quat=tquat_list,
                max_steps=20, pos_tol=0.005,
                gripper_cmd=grip_after_cmd, ctx=ctx, stage_name=f"arc{si}",
                disable_stuck_check=True,
            )

        end_err = float(last_err)
        yield from _hold_pose(world, arm_eff, eef_target_qi, grasp_ori,
                              n_frames=12, gripper_cmd=grip_after_cmd)

    else:
        end_err = 0.0

    # settle；exec_eef_pose 跳过闭爪时，settle 也必须保持预夹爪状态。
    settle_grip_cmd = (
        grip_cmd_motion
        if (skip_grip_close or move_only) and grip_cmd_motion is not None
        else grip_after_cmd
    )
    settle_frames = 3 if (skip_grip_close or move_only) else (1 if target == "grasp" else 10)
    settle_pos = lift_goal_pos if target == "grasp" else target_pos
    yield from _hold_pose(world, arm_eff, settle_pos, grasp_ori,
                          n_frames=settle_frames,
                          gripper_cmd=settle_grip_cmd)

    try:
        eef_final = world.eef_pose(arm=arm_eff)
        final_ref = lift_goal_pos if target == "grasp" else target_pos
        final_pos_err = float(np.linalg.norm(
            np.asarray(eef_final["pos"], dtype=np.float64) - final_ref
        ))
        final_ori_err_deg = _eef_ori_err_deg(world, arm_eff, grasp_ori)
        final_approach_err_deg = _eef_approach_err_deg(world, arm_eff, grasp_ori)
    except Exception:
        try:
            final_pos_err = float(cnt_err)
        except (TypeError, ValueError):
            final_pos_err = 9.99
        if math.isnan(final_pos_err) or math.isinf(final_pos_err):
            final_pos_err = 9.99
        final_ori_err_deg = 999.0
        final_approach_err_deg = 999.0

    # ─── 结果验证 ───
    exec_ok = float(final_pos_err) < 0.06 and (
        float(final_approach_err_deg) < _GRASP_ORI_GATE_DEG
        or float(final_ori_err_deg) < _GRASP_ORI_GATE_DEG
    )
    result = {
        "ok": exec_ok,
        "eef_id": cand.get("id"),
        "label": cand.get("label"),
        "target": target,
        "arm": arm_eff,
        "pre_err": _err_json(pre_err),
        "cnt_err": _err_json(cnt_err),
        "end_err": _err_json(end_err),
        "final_pos_err_m": _err_json(final_pos_err),
        "final_ori_err_deg": round(float(final_ori_err_deg), 2),
        "final_approach_err_deg": round(float(final_approach_err_deg), 2),
        "used_curobo": bool(use_curobo),
    }
    if not exec_ok:
        pre_mm = (
            f"{float(pre_err) * 1000:.0f}"
            if pre_err is not None and math.isfinite(float(pre_err))
            else "?"
        )
        cnt_mm = (
            f"{float(cnt_err) * 1000:.0f}"
            if cnt_err is not None and math.isfinite(float(cnt_err))
            else "?"
        )
        if float(final_pos_err) >= 0.06:
            result["error"] = (
                f"执行未到位：末端距规划点 {float(final_pos_err) * 1000:.0f}mm "
                f"(pre={pre_mm} cnt={cnt_mm}mm)。"
                "请先 move_to_object 靠近物体再 plan/exec"
            )
        else:
            result["error"] = (
                f"执行未到位：夹爪指向偏差 {float(final_approach_err_deg):.1f}°"
                f"（完整姿态偏差 {float(final_ori_err_deg):.1f}°，阈值 "
                f"{_GRASP_ORI_GATE_DEG:.0f}°）"
            )
        ctx.log(f"  EXEC FAIL: {result['error']}")
    if target == "grasp":
        obj_z1 = _read_obj_z(obj)
        dz = (obj_z1 - obj_z0) if (obj_z0 is not None and obj_z1 is not None) else None
        obj_pos1 = None
        if obj is not None:
            try:
                p, _ = obj.get_position_orientation()
                obj_pos1 = _to_np(p)
            except Exception:
                obj_pos1 = None
        obj_displ = None
        obj_moved_dist = None
        if obj_pos0 is not None and obj_pos1 is not None:
            obj_displ_arr = np.asarray(obj_pos1, dtype=np.float64) - np.asarray(obj_pos0, dtype=np.float64)
            obj_displ = obj_displ_arr.tolist()
            obj_moved_dist = float(np.linalg.norm(obj_displ_arr))
        if skip_grip_close:
            grasped = False
        elif skip_next_move:
            grasped = _gripper_fingers_closed(grip_q_after)
            if grasped and float(cnt_err) < 0.04:
                exec_ok = True
                result.pop("error", None)
        else:
            grasped = bool(dz is not None and dz > 0.04)
            if grasped:
                exec_ok = True
                result.pop("error", None)
        finger_closed_empty = (
            not bool(skip_grip_close)
            and _gripper_fingers_closed(grip_q_after, thresh=0.006)
            and not bool(grasped)
        )
        suspected_tunnel = False
        if finger_closed_empty and obj_pos0 is not None and obj_pos1 is not None:
            # If the wrist was at the planned contact pose but the fingers
            # fully closed and the object was not lifted, treat this as an
            # execution/physics-contact failure, not a planning failure.
            suspected_tunnel = bool(float(cnt_err) <= 0.015)
        result.update({
            "object_z_before": obj_z0, "object_z_after": obj_z1, "object_dz": dz,
            "object_pos_before": obj_pos0.tolist() if obj_pos0 is not None else None,
            "object_pos_after": obj_pos1.tolist() if obj_pos1 is not None else None,
            "object_displacement": obj_displ,
            "object_moved_dist": obj_moved_dist,
            "grasped": grasped,
            "finger_closed_empty": finger_closed_empty,
            "suspected_gripper_tunneling": suspected_tunnel,
            "skip_next_move": bool(skip_next_move),
            "skip_grip_close": bool(skip_grip_close),
        })
        extra = ""
        if skip_grip_close:
            extra = " (pose-only: 仅到位)"
        elif skip_next_move:
            extra = " (pose-only: 以合爪判定)"
        tunnel_s = " suspected_gripper_tunneling=True" if suspected_tunnel else ""
        ctx.log(f"  result: dz={dz} grasped={grasped}{extra}{tunnel_s}")

    elif target == "push":
        obj_pos1 = None
        if obj is not None:
            try:
                p, _ = obj.get_position_orientation()
                obj_pos1 = _to_np(p)
            except Exception:
                obj_pos1 = None
        if obj_pos0 is not None and obj_pos1 is not None:
            displ = (obj_pos1 - obj_pos0).tolist()
            dist = float(np.linalg.norm(np.array(displ)))
            # "掉到地上" = z 下降很多
            fell = bool((obj_pos0[2] - obj_pos1[2]) > 0.2)
            result.update({
                "object_pos_before": obj_pos0.tolist(),
                "object_pos_after": obj_pos1.tolist(),
                "object_displacement": displ,
                "object_moved_dist": dist,
                "object_fell": fell,
                "pushed": bool(dist > 0.05 or fell),
            })
            ctx.log(f"  result: displacement={displ} dist={dist:.3f} fell={fell}")

    elif target in ("open", "close"):
        open_val1 = None
        if obj is not None:
            try:
                from omnigibson.object_states.open_state import Open
                if Open in obj.states:
                    open_val1 = bool(obj.states[Open].get_value())
            except Exception:
                open_val1 = None
        # 读 joint 实际角度
        cur_q_after = None
        try:
            for j_name, joint in obj.joints.items():
                if joint.name == cand["meta"]["joint_name"]:
                    cur_q_after = float(joint.get_state()[0])
                    break
        except Exception:
            pass
        success = False
        if target == "open" and open_val1 is True:
            success = True
        elif target == "close" and open_val1 is False:
            success = True
        result.update({
            "open_before": open_val0,
            "open_after": open_val1,
            "joint_q_after": cur_q_after,
            "success": success,
        })
        ctx.log(f"  result: open {open_val0}→{open_val1} q={cur_q_after} success={success}")

    result["ok"] = exec_ok
    return result


@register_skill(
    "execute_eef_pose",
    description=(
        "执行 get_eef_pose 返回的某个 candidate。eef_id=0 默认；eef_id=-1 = 遍历所有 "
        "reachable 候选；push 时若 push_direction 传入则覆盖 cache 的方向。"
    ),
)
def execute_eef_pose(
    ctx,
    eef_id: int = 0,
    arm: str = "",
    push_direction: Optional[list] = None,
    push_dist: Optional[float] = None,
):
    """yield 每步 action 到 sim。"""
    world = ctx.world

    # plan_eef / plan_grasp / vlm_scene_grasp 精化结果优先于纯几何 get_eef_pose
    last = (
        ctx.get_last_result("plan_eef")
        or ctx.get_last_result("plan_grasp")
        or ctx.get_last_result("vlm_scene_grasp")
        or ctx.get_last_result("get_eef_pose")
    )
    if last is None or not last.get("ok"):
        ctx.set_result({"ok": False, "error": "先执行 plan_eef / plan_grasp / get_eef_pose 获得候选"})
        yield world.empty_action()
        return

    cands = last["candidates"]
    target = last.get("target", "grasp")

    # push_direction / push_dist 覆盖
    if push_direction is not None and target == "push":
        n = np.asarray(push_direction, dtype=np.float64)
        if np.linalg.norm(n) > 1e-6:
            n = n / np.linalg.norm(n)
            dist = float(push_dist) if push_dist is not None else float(
                np.linalg.norm(np.array(cands[0]["next_eef_move"]))
            )
            for c in cands:
                c["next_eef_move"] = (n * dist).tolist()
                c["eef_target"]["approach"] = n.tolist()
                c["meta"]["push_direction"] = n.tolist()
                c["meta"]["push_dist"] = dist

    # ─── eef_id < 0：遍历所有 reachable ───
    if eef_id < 0:
        if _challenge_action_only_enabled():
            ctx.set_result({
                "ok": False,
                "error": (
                    "评测模式禁用 eef_id<0：该旧遍历模式会在多轮之间直接"
                    "回写物体/铰链状态。请逐个执行候选，并用正式 scene reset"
                    "开始下一轮。"
                ),
            })
            yield world.empty_action()
            return
        reach = [c for c in cands if c.get("reachable")]
        if not reach:
            ctx.set_result({"ok": False, "error": "无 reachable 候选"})
            yield world.empty_action(); return

        # 记录物体初始 pose（用于 reset 多轮）
        obj_name = (last["object"].get("input") or last["object"].get("resolved_name"))
        obj = _resolve_object_handle(world, obj_name)
        init_pos = init_quat = None
        if obj is not None:
            try:
                p, q = obj.get_position_orientation()
                init_pos, init_quat = _to_np(p).copy(), _to_np(q).copy()
            except Exception:
                pass

        # 对 open/close：每轮 reset 把 hinge 设回 closed_q 同时还原物体 pose
        results = []
        for round_i, c in enumerate(reach):
            ctx.log(f"\n[round {round_i+1}/{len(reach)}] eef #{c['id']} label={c['label']} target={c['target']}")
            if round_i > 0 and obj is not None:
                try:
                    import torch as _th
                    if init_pos is not None and init_quat is not None:
                        obj.set_position_orientation(
                            position=_th.tensor(init_pos, dtype=_th.float32),
                            orientation=_th.tensor(init_quat, dtype=_th.float32),
                        )
                except Exception as e:
                    ctx.log(f"  reset obj pose err: {e}")
                # hinge reset
                if target in ("open", "close"):
                    try:
                        meta = c.get("meta", {})
                        jname = meta.get("joint_name")
                        for j_name, joint in obj.joints.items():
                            if joint.name == jname:
                                joint.set_pos(float(meta.get("cur_q", meta.get("closed_q", 0.0))))
                                break
                    except Exception as e:
                        ctx.log(f"  reset hinge err: {e}")
                for _ in range(20):
                    yield world.empty_action()

            r = yield from _execute_one_eef(ctx, last, c, arm)
            results.append(r)

            # release arm 准备下一轮
            arm_eff = r["arm"]
            try:
                grasp_ori = np.asarray(world.eef_pose(arm=arm_eff)["quat"], dtype=np.float64)
                eef_now = np.asarray(world.eef_pose(arm=arm_eff)["pos"], dtype=np.float64)
                yield from _release_arm_to_home(
                    world, arm_eff, last_lift_pos=eef_now + np.array([0, 0, 0.05]),
                    grasp_ori=grasp_ori, ctx=ctx,
                )
            except Exception as e:
                ctx.log(f"  release err: {e}")

        n_success = sum(1 for r in results
                        if r.get("grasped") or r.get("pushed") or r.get("success"))
        ctx.log(f"\n=== 遍历完成: {n_success}/{len(results)} 成功 ===")
        ctx.set_result({
            "ok": True, "mode": "all_reachable", "target": target,
            "n_tried": len(results), "n_success": n_success,
            "per_eef": results,
        })
        yield world.empty_action()
        return

    # ─── 单个执行 ───
    if eef_id >= len(cands):
        ctx.set_result({"ok": False, "error": f"eef_id={eef_id} 越界 (共 {len(cands)})"})
        yield world.empty_action(); return

    cand = cands[eef_id]
    # 重判 reachable
    cgrasp = {"pos": cand["eef_target"]["pos"], "approach": cand["eef_target"]["approach"]}
    ok, why = _is_reachable(world, cgrasp, arm=arm or cand.get("arm") or "right")
    if not ok:
        bp = _suggest_base_pose(world, cgrasp, arm=arm or cand.get("arm") or "right")
        if bp is not None:
            next_action = (f"move_to(x={bp['x']:.2f}, y={bp['y']:.2f}, z={bp['z']:.2f}, "
                           f"theta_x_deg={bp['theta_x_deg']:.1f}, theta_z_deg={bp['theta_z_deg']:.1f})")
            ctx.log(f"  不 reachable: {why}; 建议 {next_action}")
            ctx.set_result({"ok": True, "reachable": False, "reason": why,
                            "suggested_base_pose": bp, "next_action": next_action,
                            "eef_used": cand})
        else:
            ctx.set_result({"ok": False, "error": f"不 reachable 且 free_region 无合适 base: {why}"})
        yield world.empty_action(); return

    r = yield from _execute_one_eef(ctx, last, cand, arm)
    ctx.set_result(r)
    for _ in range(3):
        yield world.hold_action()


@register_skill(
    "diag_q4_lock_smoke",
    description="诊断：只发夹爪 partial action，验证 base/trunk/arms 是否被 pinned lock",
)
def diag_q4_lock_smoke(ctx, arm: str = "left", frames: int = 20, gripper_cmd: float = 1.0):
    world = ctx.world
    _ensure_world_pinned_actions(world)
    if hasattr(world, "set_trunk_pin_qpos"):
        world.set_trunk_pin_qpos()
    arm_eff = str(arm or "left").strip().lower()
    if arm_eff not in ("left", "right"):
        arm_eff = "left"
    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    for _ in range(max(1, int(frames))):
        grip_override = _gripper_cmd_override(gripper_cmd)
        action = {} if grip_override is None else {f"gripper_{arm_eff}": grip_override}
        yield world.make_action(**action)
    q1 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    delta = q1 - q0
    ctx.log(
        f"[diag_q4_lock_smoke] arm={arm_eff} frames={int(frames)} "
        f"trunk0={np.round(q0, 5).tolist()} trunk1={np.round(q1, 5).tolist()} "
        f"dq={np.round(delta, 6).tolist()}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm_eff,
        "frames": int(frames),
        "trunk_q_start": q0.tolist(),
        "trunk_q_end": q1.tolist(),
        "trunk_q_delta": delta.tolist(),
        "q4_start": float(q0[3]),
        "q4_end": float(q1[3]),
        "q4_delta_rad": float(delta[3]),
        "trunk_delta_inf_rad": float(np.linalg.norm(delta, ord=np.inf)),
    })
    yield world.hold_action()


@register_skill(
    "diag_plan_final_q_fk",
    description="诊断：读取 plan 的 stored final arm q，在当前整机状态下做 OG FK 探针并恢复",
)
def diag_plan_final_q_fk(ctx, session_id: str = "web", plan_id: str = "", arm: str = ""):
    world = ctx.world
    import json
    from behavior_interface import agent_runs

    try:
        record = agent_runs.load_plan_record(session_id, plan_id)
    except Exception as exc:
        ctx.set_result({"ok": False, "error": f"load_plan_record failed: {exc}"})
        yield world.hold_action()
        return

    cand = record.get("candidate") or {}
    eef_target = cand.get("eef_target") or {}
    target_pos = np.asarray(eef_target.get("pos"), dtype=np.float64).reshape(3)
    target_quat = _quat_normalize_xyzw(eef_target.get("quat"))
    arm_eff = str(arm or record.get("arm") or cand.get("arm") or "left").lower().strip()
    if arm_eff not in ("left", "right"):
        arm_eff = "left"
    q_arm = _stored_filter_q_for_arm(cand, arm_eff)
    if q_arm is None:
        ctx.set_result({
            "ok": False,
            "error": f"no stored final q for arm={arm_eff}",
            "candidate_keys": sorted(cand.keys()),
            "meta_keys": sorted((cand.get("meta") or {}).keys()),
        })
        yield world.hold_action()
        return

    robot = world.robot
    names = list(robot.joints.keys())
    arm_idx = [names.index(f"{arm_eff}_arm_joint{i + 1}") for i in range(7)]
    saved = robot.get_joint_positions().clone()
    q_now = _arm_qpos_now(world, arm_eff)
    q_target = np.asarray(q_arm, dtype=np.float64).reshape(7)
    before = _eef_pose_err(world, arm_eff, target_pos, target_quat)
    trunk = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    try:
        q_probe = saved.clone()
        for k, jidx in enumerate(arm_idx):
            q_probe[int(jidx)] = float(q_target[k])
        robot.set_joint_positions(q_probe)
        probe = _eef_pose_err(world, arm_eff, target_pos, target_quat)
        probe_pose = world.eef_pose(arm=arm_eff)
    finally:
        try:
            robot.set_joint_positions(saved)
        except Exception:
            pass

    q_delta = q_target - q_now
    out = {
        "ok": True,
        "session_id": session_id,
        "plan_id": plan_id,
        "arm": arm_eff,
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "trunk_qpos": trunk.tolist(),
        "current_arm_q": q_now.tolist(),
        "stored_final_q": q_target.tolist(),
        "current_minus_stored_inf_rad": float(np.linalg.norm(q_now - q_target, ord=np.inf)),
        "current_minus_stored": q_delta.tolist(),
        "current_pose_err": {
            "pos_mm": float(before[0]) * 1000.0,
            "ori_deg": float(before[1]),
            "approach_deg": float(before[2]),
        },
        "probe_stored_q_pose_err": {
            "pos_mm": float(probe[0]) * 1000.0,
            "ori_deg": float(probe[1]),
            "approach_deg": float(probe[2]),
        },
        "probe_eef_pose": probe_pose,
        "record_arm": record.get("arm"),
        "candidate_arm": cand.get("arm"),
        "selected_pose_ik": (cand.get("meta") or {}).get("selected_pose_ik") or cand.get("selected_pose_ik"),
    }
    ctx.log("[diag_plan_final_q_fk] " + json.dumps({
        "arm": arm_eff,
        "q_inf": round(out["current_minus_stored_inf_rad"], 4),
        "cur_pos_mm": round(out["current_pose_err"]["pos_mm"], 1),
        "cur_ori_deg": round(out["current_pose_err"]["ori_deg"], 2),
        "probe_pos_mm": round(out["probe_stored_q_pose_err"]["pos_mm"], 1),
        "probe_ori_deg": round(out["probe_stored_q_pose_err"]["ori_deg"], 2),
        "trunk": [round(float(x), 4) for x in trunk.tolist()],
    }, ensure_ascii=False))
    ctx.set_result(out)
    yield world.hold_action()


@register_skill("test_joint_open",
                description="直接测试微波炉门关节是否可以物理移动（诊断用）")
def test_joint_open(ctx, object_name: str = "microwave", target_q: float = 0.5):
    """直接驱动关节到 target_q，验证 joint 是否可移动。"""
    import numpy as np
    world = ctx.world

    # 找 object
    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        ctx.log(f"[diag] object {object_name!r} not found")
        ctx.set_result({"ok": False, "error": "not found"})
        yield world.empty_action()
        return

    ctx.log(f"[diag] object={obj.name}, testing joint movability")

    for jname, joint in obj.joints.items():
        q0 = float(joint.get_state()[0])
        ctx.log(f"  joint {jname}: q0={q0:.4f} limits=[{joint.lower_limit:.3f}, {joint.upper_limit:.3f}]")

    # 等 2 帧让物理稳定
    yield world.empty_action()
    yield world.empty_action()

    # 直接 set_pos 所有 joint
    for jname, joint in obj.joints.items():
        q0 = float(joint.get_state()[0])
        if abs(joint.upper_limit - joint.lower_limit) < 0.01:
            ctx.log(f"  skip fixed joint {jname}")
            continue
        try:
            joint.set_pos(np.array([target_q]))
            ctx.log(f"  {jname}: set_pos({target_q:.3f}) OK (was {q0:.4f})")
        except Exception as e:
            ctx.log(f"  {jname}: set_pos error: {e}")

    # 等 10 帧观察效果
    for _ in range(10):
        yield world.empty_action()

    for jname, joint in obj.joints.items():
        q1 = float(joint.get_state()[0])
        ctx.log(f"  after 10 frames: {jname}: q={q1:.6f}")

    ctx.set_result({"ok": True})
