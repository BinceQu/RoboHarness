"""reset_body —— 腰部三关节复位为直立（q1=q2=q3=0，q4=0），底盘不动。

reset_body 只负责躯干复位。进入 skill 时快照左右臂和夹爪当前关节角，
之后每一帧显式下发同一组目标，避免复位腰部时手臂被 hang/补偿逻辑带着甩动。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.arm_reset import _arm_qpos

RESET_BODY_BUILD = "v11_monotonic_deadline_step_limited_j1234_unchanged_j567_keep_ori"

# 直立：与 scene reset / arm_reset hang 一致
_UPRIGHT_TRUNK = np.zeros(4, dtype=np.float64)
_TRUNK_TOL_RAD = 0.025
# 控制器尝试上限：超过则切运动学（控制器抬深俯躯干太慢）
_KINEMATIC_AFTER_S = 8.0
_SMOOTHSTEP_MAX_SLOPE = 1.5
# Active keep_ori path: preserve the entry J1-J4 command and solve only J5-J7
# against the entry world-frame EEF orientation. EEF translation is diagnostic.
_EEF_ORI_SOLVE_TOL_DEG = 1.0
_EEF_ORI_TRACK_OK_DEG = 5.0
_EEF_ORI_DAMPING = 0.035
_EEF_PASSIVE_DAMPING = 0.060
_EEF_PASSIVE_VELOCITY_WEIGHT = 1.0
_EEF_PASSIVE_ACCEL_WEIGHT = 1.5
_EEF_PASSIVE_JOINT_WEIGHT = 0.002
_EEF_IK_MAX_DQ_PER_STEP = 0.10
_EEF_WRIST_COMMAND_MAX_STEP_RAD = 0.25
_EEF_POS_MONITOR_MAX_STEP_M = 0.020
_EEF_POS_MONITOR_MAX_ACCEL_STEP_M = 0.020


def _challenge_action_only() -> bool:
    mode = str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE")
        or os.environ.get("INTERFACE_CHALLENGE_MODE")
        or ""
    ).strip().lower()
    return mode in {"train", "public_test", "hidden_test"}


def _trunk_step_toward_upright(
    q_cur: np.ndarray,
    *,
    max_step: float = 0.06,
) -> np.ndarray:
    """下一步 trunk 绝对目标；reset_body 中四个 trunk 关节各自回零。"""
    q_cur = np.asarray(q_cur, dtype=np.float64).reshape(4)
    err = _UPRIGHT_TRUNK - q_cur
    step = np.clip(err, -max_step, max_step)
    return q_cur + step


def _trunk_upright_reached(q: np.ndarray, *, tol: float = _TRUNK_TOL_RAD) -> bool:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return float(np.linalg.norm(q[:3] - _UPRIGHT_TRUNK[:3], ord=np.inf)) < tol


def _gripper_qpos(world, arm: str) -> Optional[List[float]]:
    fn = getattr(world, "gripper_qpos_list", None)
    if not callable(fn):
        return None
    try:
        q = fn(arm)
    except Exception:
        return None
    if q is None:
        return None
    return [float(x) for x in q]


def _normalize_keep_ori_arm(value: str | None) -> Set[str]:
    v = str(value or "none").strip().lower()
    if v in ("", "none", "false", "0", "no"):
        return set()
    if v in ("both", "all"):
        return {"left", "right"}
    out = {p.strip().lower() for p in v.replace(",", " ").split() if p.strip()}
    return {a for a in out if a in ("left", "right")}


def _quat_normalize(q) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return arr / n


def _quat_err_deg(q0, q1) -> float:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _eef_pose_np(world, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    pose = world.eef_pose(arm=arm)
    pos = np.asarray(pose["pos"], dtype=np.float64).reshape(3)
    quat = _quat_normalize(pose["quat"])
    return pos.copy(), quat.copy()


def _clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).reshape(-1).copy()
    norm = float(np.linalg.norm(out))
    if norm > float(max_norm) > 0.0:
        out *= float(max_norm) / (norm + 1e-12)
    return out


def _robot_joint_state_clone(robot):
    q = robot.get_joint_positions()
    try:
        return q.clone()
    except Exception:
        return np.asarray(q, dtype=np.float64).copy()


def _set_full_joint_positions(robot, q) -> None:
    robot.set_joint_positions(q)


def _set_trunk_and_arm_probe(world, trunk_q: np.ndarray, arm: str, arm_q: np.ndarray) -> None:
    from behavior_interface.skills.arm_reset import _get_arm_dof_idx

    robot = world.robot
    q = _robot_joint_state_clone(robot)
    trunk_idx_raw = robot.trunk_control_idx
    if hasattr(trunk_idx_raw, "detach"):
        trunk_idx_raw = trunk_idx_raw.detach().cpu().numpy()
    trunk_idx = np.asarray(trunk_idx_raw, dtype=int).reshape(-1)[:4]
    for i, j in enumerate(trunk_idx):
        q[int(j)] = float(np.asarray(trunk_q, dtype=np.float64).reshape(4)[i])
    arm_idx = _get_arm_dof_idx(world, arm)
    aq = np.asarray(arm_q, dtype=np.float64).reshape(7)
    for i, j in enumerate(arm_idx):
        q[int(j)] = float(aq[i])
    _set_full_joint_positions(robot, q)


def _set_trunk_and_arms_direct(
    world,
    trunk_q: np.ndarray,
    arm_q_map: Dict[str, np.ndarray],
) -> None:
    """Directly place trunk + explicit arm qpos at one synchronized waypoint."""
    from behavior_interface.skills.arm_reset import _get_arm_dof_idx

    if _challenge_action_only() and not getattr(world, "dry_run", False):
        try:
            world.set_trunk_pin_qpos(np.asarray(trunk_q, dtype=np.float64).reshape(4))
        except Exception:
            pass
        for arm, aq_raw in arm_q_map.items():
            try:
                world.set_arm_pin_qpos(arm, np.asarray(aq_raw, dtype=np.float64).reshape(7))
            except Exception:
                pass
        return

    robot = world.robot
    q = _robot_joint_state_clone(robot)
    touched: List[int] = []
    trunk_idx_raw = robot.trunk_control_idx
    if hasattr(trunk_idx_raw, "detach"):
        trunk_idx_raw = trunk_idx_raw.detach().cpu().numpy()
    trunk_idx = np.asarray(trunk_idx_raw, dtype=int).reshape(-1)[:4]
    tq = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    for i, j in enumerate(trunk_idx):
        q[int(j)] = float(tq[i])
        touched.append(int(j))
    for arm, aq_raw in arm_q_map.items():
        aq = np.asarray(aq_raw, dtype=np.float64).reshape(7)
        arm_idx = _get_arm_dof_idx(world, arm)
        for i, j in enumerate(arm_idx):
            q[int(j)] = float(aq[i])
            touched.append(int(j))
    robot.set_joint_positions(q)
    try:
        v = robot.get_joint_velocities()
        try:
            v_new = v.clone()
        except Exception:
            v_new = np.asarray(v, dtype=np.float64).copy()
        for j in touched:
            v_new[int(j)] = 0.0
        robot.set_joint_velocities(v_new)
    except Exception:
        pass
    try:
        world.set_trunk_pin_qpos(tq.tolist())
    except Exception:
        pass
    for arm, aq_raw in arm_q_map.items():
        try:
            world.set_arm_pin_qpos(arm, np.asarray(aq_raw, dtype=np.float64).reshape(7))
        except Exception:
            pass


def _robot_base_pose_np(world) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取底盘 link 世界位姿 (pos[3], quat_xyzw[4])，失败返回 None。"""
    robot = getattr(world, "robot", None)
    if robot is None:
        return None
    try:
        pos, quat = robot.get_position_orientation()
        pos = np.asarray(
            pos.detach().cpu().numpy() if hasattr(pos, "detach") else pos,
            dtype=np.float64,
        ).reshape(3)
        quat = np.asarray(
            quat.detach().cpu().numpy() if hasattr(quat, "detach") else quat,
            dtype=np.float64,
        ).reshape(4)
        return pos.copy(), quat.copy()
    except Exception:
        return None


def _hold_robot_base(world, base_pose: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
    """把底盘 link 钉回快照世界位姿并清零底盘线/角速度。

    运动学吸附躯干时若不锁底盘，突变的关节配置会被物理引擎当成高能/穿模状态，
    把整机掀飞甚至翻倒（实测 base z 0.005→0.403 翻转）。reset_body 不移动底盘，
    因此每帧把底盘钉死即可。
    """
    if (
        base_pose is None
        or getattr(world, "dry_run", False)
        or _challenge_action_only()
    ):
        return
    robot = getattr(world, "robot", None)
    if robot is None:
        return
    pos, quat = base_pose
    try:
        import torch as th

        position = th.tensor(np.asarray(pos, dtype=np.float64), dtype=th.float32)
        orientation = th.tensor(np.asarray(quat, dtype=np.float64), dtype=th.float32)
    except Exception:
        position, orientation = np.asarray(pos), np.asarray(quat)
    try:
        robot.set_position_orientation(position=position, orientation=orientation)
    except Exception:
        pass
    for setter_name in ("set_linear_velocity", "set_angular_velocity"):
        setter = getattr(robot, setter_name, None)
        if callable(setter):
            try:
                setter(np.zeros(3, dtype=np.float32))
            except Exception:
                pass


def _kinematic_drive_trunk_to_upright(
    ctx,
    world,
    arm_targets: Dict[str, np.ndarray],
    base_pose: Optional[Tuple[np.ndarray, np.ndarray]],
    *,
    n_frames: int = 18,
    max_step: Optional[float] = None,
    deadline: Optional[float] = None,
    legacy_7dof_arms: Optional[Set[str]] = None,
    eef_balancer=None,
) -> Generator[Any, None, Dict[str, Any]]:
    """逐帧渐进运动学插值把躯干开到直立，同时每帧锁住底盘。

    相比一步突跳 3rad（会掀飞底盘），渐进插值让物理在每帧只承受很小的构型变化，
    配合底盘钉位，保证机器人稳稳站直、底盘不离地不翻倒。
    """
    q_from = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    n = max(2, int(n_frames))
    if max_step is not None:
        step_limit = float(max_step)
        if not np.isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("max_step must be a positive finite value")
        max_gap = float(
            np.linalg.norm(_UPRIGHT_TRUNK - q_from, ord=np.inf)
        )
        n = max(
            n,
            int(np.ceil(_SMOOTHSTEP_MAX_SLOPE * max_gap / step_limit)),
        )
    guarded_arms = set(legacy_7dof_arms or ())
    from behavior_interface.skills.eef import _assert_legacy_7dof_motion_ready

    frames_played = 0
    for k in range(1, n + 1):
        if deadline is not None and time.monotonic() >= float(deadline):
            return {
                "completed": False,
                "timed_out": True,
                "frames": frames_played,
                "frames_planned": n,
            }
        s = k / n
        ss = s * s * (3.0 - 2.0 * s)
        q_k = q_from + (_UPRIGHT_TRUNK - q_from) * float(ss)
        for arm in guarded_arms:
            _assert_legacy_7dof_motion_ready(world, arm)
        arm_targets_k = {
            arm: np.asarray(q, dtype=np.float64).reshape(7).copy()
            for arm, q in arm_targets.items()
        }
        if eef_balancer is not None and eef_balancer.active:
            arm_targets_k.update(eef_balancer.solve_for_trunk(q_k))
        _set_trunk_and_arms_direct(world, q_k, arm_targets_k)
        if eef_balancer is not None:
            eef_balancer.apply_payload_hold()
        _hold_robot_base(world, base_pose)
        kw: Dict[str, Any] = {"trunk": q_k.tolist()}
        for a, q in arm_targets_k.items():
            kw[f"arm_{a}"] = np.asarray(q, dtype=np.float64).reshape(7).tolist()
        for arm in guarded_arms:
            _assert_legacy_7dof_motion_ready(world, arm)
        frames_played += 1
        yield world.make_action(**kw)
        _hold_robot_base(world, base_pose)
        if eef_balancer is not None:
            eef_balancer.after_step()
    return {
        "completed": True,
        "timed_out": False,
        "frames": frames_played,
        "frames_planned": n,
    }


def _arm_joint_limits(world, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    from behavior_interface.skills.arm_reset import _arm_joint_limits as _limits

    return _limits(world, arm)


def _orientation_primary_nullspace_dq(
    jacobian: np.ndarray,
    orientation_error: np.ndarray,
    passive_position_error: np.ndarray,
    q_to_nominal: np.ndarray,
    *,
    orientation_damping: float = _EEF_ORI_DAMPING,
    passive_damping: float = _EEF_PASSIVE_DAMPING,
    position_weight: float = 1.0,
    joint_weight: float = _EEF_PASSIVE_JOINT_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One strict task-priority differential IK step.

    Orientation is solved first. Translation and joint continuity are solved
    only in the exact SVD nullspace of the angular Jacobian, so they cannot
    trade EEF orientation for a smaller position change.
    """
    J = np.asarray(jacobian, dtype=np.float64).reshape(6, 7)
    Jv = J[:3, :]
    Jw = J[3:6, :]
    omega = np.asarray(orientation_error, dtype=np.float64).reshape(3)
    dx_passive = np.asarray(
        passive_position_error,
        dtype=np.float64,
    ).reshape(3)
    dq_nominal = np.asarray(q_to_nominal, dtype=np.float64).reshape(7)

    lam = max(1e-8, float(orientation_damping))
    dq_primary = Jw.T @ np.linalg.solve(
        Jw @ Jw.T + (lam * lam) * np.eye(3, dtype=np.float64),
        omega,
    )

    _, singular_values, vt = np.linalg.svd(Jw, full_matrices=True)
    if singular_values.size:
        rank_tol = max(
            1e-8,
            float(singular_values[0]) * max(Jw.shape) * 1e-6,
        )
        rank = int(np.count_nonzero(singular_values > rank_tol))
    else:
        rank = 0
    null_basis = vt[rank:, :].T
    if null_basis.shape[1] == 0:
        return dq_primary, dq_primary.copy(), rank

    blocks: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    pos_w = max(0.0, float(position_weight))
    if pos_w > 0.0:
        sqrt_w = float(np.sqrt(pos_w))
        blocks.append(sqrt_w * (Jv @ null_basis))
        targets.append(
            sqrt_w * (dx_passive - Jv @ dq_primary),
        )
    joint_w = max(0.0, float(joint_weight))
    if joint_w > 0.0:
        sqrt_w = float(np.sqrt(joint_w))
        blocks.append(sqrt_w * null_basis)
        targets.append(
            sqrt_w * (dq_nominal - dq_primary),
        )

    damp = max(0.0, float(passive_damping))
    if damp > 0.0:
        blocks.append(damp * np.eye(null_basis.shape[1], dtype=np.float64))
        targets.append(np.zeros(null_basis.shape[1], dtype=np.float64))

    if not blocks:
        return dq_primary, dq_primary.copy(), rank

    A = np.vstack(blocks)
    b = np.concatenate(targets)
    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    dq = dq_primary + null_basis @ z
    return dq, dq_primary, rank


def _orientation_primary_passive_arm_q(
    world,
    arm: str,
    *,
    trunk_q: np.ndarray,
    seed_q: np.ndarray,
    target_quat: np.ndarray,
    previous_pos: np.ndarray,
    previous_delta: np.ndarray,
    ori_tol_deg: float = _EEF_ORI_SOLVE_TOL_DEG,
    max_steps: int = 48,
    max_dq_per_step: float = _EEF_IK_MAX_DQ_PER_STEP,
    velocity_weight: float = _EEF_PASSIVE_VELOCITY_WEIGHT,
    accel_weight: float = _EEF_PASSIVE_ACCEL_WEIGHT,
    joint_weight: float = _EEF_PASSIVE_JOINT_WEIGHT,
) -> Tuple[np.ndarray, float, float, str, int]:
    """Solve exact-trunk arm compensation with orientation as the primary task.

    The secondary objective is the passive cost

        w_v ||delta_p||^2 + w_a ||delta_p - delta_p_prev||^2

    which is equivalent to a velocity damper plus acceleration smoothing. It
    has no fixed world-position setpoint and is applied only in Jw's nullspace.
    """
    from behavior_interface.skills.grasp import (
        _orientation_error_omega,
        _quat_to_mat,
        _read_jacobian_arm,
    )

    trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    q_seed = np.asarray(seed_q, dtype=np.float64).reshape(7)
    q_lo, q_hi = _arm_joint_limits(world, arm)
    q = np.clip(q_seed, q_lo, q_hi)
    target_R = _quat_to_mat(_quat_normalize(target_quat))
    pos_prev = np.asarray(previous_pos, dtype=np.float64).reshape(3)
    delta_prev = np.asarray(previous_delta, dtype=np.float64).reshape(3)
    vel_w = max(0.0, float(velocity_weight))
    acc_w = max(0.0, float(accel_weight))
    momentum_ratio = acc_w / max(vel_w + acc_w, 1e-12)
    passive_delta = momentum_ratio * delta_prev
    passive_target = pos_prev + passive_delta
    orientation_band_deg = 0.08

    best_q = q.copy()
    best_pos = pos_prev.copy()
    best_ori_err = float("inf")
    best_passive_cost = float("inf")
    min_rank = 3
    no_improve = 0

    def _measure(q_probe: np.ndarray):
        _set_trunk_and_arm_probe(world, trunk, arm, q_probe)
        pos, quat = _eef_pose_np(world, arm)
        cur_R = _quat_to_mat(quat)
        omega = _orientation_error_omega(target_R, cur_R)
        ori_err_deg = float(np.degrees(np.linalg.norm(omega)))
        delta = pos - pos_prev
        passive_cost = (
            vel_w * float(np.dot(delta, delta))
            + acc_w * float(np.dot(delta - delta_prev, delta - delta_prev))
            + max(0.0, float(joint_weight))
            * float(np.dot(q_probe - q_seed, q_probe - q_seed))
        )
        return pos, omega, ori_err_deg, passive_cost

    def _better(
        ori_err: float,
        passive_cost: float,
        ref_ori_err: float,
        ref_passive_cost: float,
    ) -> bool:
        if (
            ori_err > float(ori_tol_deg)
            or ref_ori_err > float(ori_tol_deg)
        ):
            if ori_err < ref_ori_err - 1e-7:
                return True
            if ref_ori_err < ori_err - 1e-7:
                return False
            return passive_cost < ref_passive_cost - 1e-12
        if ori_err < ref_ori_err - orientation_band_deg:
            return True
        if ref_ori_err < ori_err - orientation_band_deg:
            return False
        return passive_cost < ref_passive_cost - 1e-12

    for step_i in range(max(1, int(max_steps))):
        pos, omega, ori_err_deg, passive_cost = _measure(q)
        if _better(
            ori_err_deg,
            passive_cost,
            best_ori_err,
            best_passive_cost,
        ):
            best_q = q.copy()
            best_pos = pos.copy()
            best_ori_err = ori_err_deg
            best_passive_cost = passive_cost

        omega_step = _clip_vector_norm(omega, 0.14)
        J, _ = _read_jacobian_arm(world, arm)
        dq, dq_primary, rank = _orientation_primary_nullspace_dq(
            J,
            omega_step,
            passive_target - pos,
            q_seed - q,
            position_weight=max(vel_w + acc_w, 1e-6),
            joint_weight=float(joint_weight),
        )
        min_rank = min(min_rank, int(rank))

        candidates: List[np.ndarray] = []
        for dq_raw, scales in (
            (dq, (1.0, 0.5)),
            (dq_primary, (1.0,)),
        ):
            dq_try = np.asarray(dq_raw, dtype=np.float64).reshape(7).copy()
            for i in range(7):
                if (q[i] >= q_hi[i] - 1e-4 and dq_try[i] > 0.0) or (
                    q[i] <= q_lo[i] + 1e-4 and dq_try[i] < 0.0
                ):
                    dq_try[i] = 0.0
            dq_inf = float(np.linalg.norm(dq_try, ord=np.inf))
            if dq_inf > float(max_dq_per_step):
                dq_try *= float(max_dq_per_step) / (dq_inf + 1e-12)
            for scale in scales:
                q_try = np.clip(q + float(scale) * dq_try, q_lo, q_hi)
                if float(np.linalg.norm(q_try - q, ord=np.inf)) > 1e-8:
                    candidates.append(q_try)

        next_q: Optional[np.ndarray] = None
        next_ori = ori_err_deg
        next_cost = passive_cost
        for q_try in candidates:
            _, _, ori_try, cost_try = _measure(q_try)
            if _better(ori_try, cost_try, next_ori, next_cost):
                next_q = q_try.copy()
                next_ori = ori_try
                next_cost = cost_try

        if next_q is None:
            no_improve += 1
            if no_improve >= 3 or (
                ori_err_deg <= float(ori_tol_deg) and step_i >= 4
            ):
                break
            q = best_q.copy()
            continue

        q = next_q
        no_improve = 0

    final_pos, _, final_ori_err, final_passive_cost = _measure(best_q)
    if _better(
        final_ori_err,
        final_passive_cost,
        best_ori_err,
        best_passive_cost,
    ):
        best_pos = final_pos.copy()
        best_ori_err = final_ori_err
        best_passive_cost = final_passive_cost
    passive_residual_m = float(
        np.linalg.norm((best_pos - pos_prev) - passive_delta),
    )
    mode = (
        "orientation_primary"
        if best_ori_err <= float(ori_tol_deg)
        else "orientation_limited"
    )
    return (
        best_q.copy(),
        passive_residual_m,
        float(best_ori_err),
        mode,
        int(min_rank),
    )


def _solve_wrist_for_eef_ori_at_trunk(
    world,
    arm: str,
    *,
    trunk_q: np.ndarray,
    shoulder_elbow_q: np.ndarray,
    wrist_seed_q: np.ndarray,
    target_quat: np.ndarray,
    max_steps: int = 40,
    ori_tol_deg: float = 0.20,
) -> Tuple[np.ndarray, float, bool]:
    """Solve only q5-q7 so EEF orientation stays fixed for a probed trunk pose.

    The first four arm joints remain exactly at their entry snapshot.  The last
    three wrist joints are updated with an orientation-only DLS step using OG FK
    and the real EEF Jacobian, then all probed joint state is restored by the
    caller.
    """
    from behavior_interface.skills.grasp import (
        _dls_solve_dq,
        _orientation_error_omega,
        _quat_to_mat,
        _read_jacobian_arm,
    )

    target_q = _quat_normalize(target_quat)
    target_R = _quat_to_mat(target_q)
    lo, hi = _arm_joint_limits(world, arm)
    shoulder_elbow = np.asarray(shoulder_elbow_q, dtype=np.float64).reshape(4)
    seed = np.asarray(wrist_seed_q, dtype=np.float64).reshape(3)
    mid = (lo[4:7] + hi[4:7]) * 0.5
    wrist_seeds = [
        seed,
        np.array([seed[0], seed[1] + 0.45, seed[2]], dtype=np.float64),
        np.array([seed[0], seed[1] - 0.45, seed[2]], dtype=np.float64),
        np.array([seed[0], seed[1] + 0.90, seed[2]], dtype=np.float64),
        np.array([seed[0], seed[1] - 0.90, seed[2]], dtype=np.float64),
        mid,
        np.zeros(3, dtype=np.float64),
    ]
    unique_seeds: List[np.ndarray] = []
    for ws in wrist_seeds:
        ws = np.clip(np.asarray(ws, dtype=np.float64).reshape(3), lo[4:7], hi[4:7])
        if not any(float(np.linalg.norm(ws - old, ord=np.inf)) < 1e-5 for old in unique_seeds):
            unique_seeds.append(ws)
    for axis in range(3):
        for delta in (-1.20, -0.75, +0.75, +1.20):
            ws = seed.copy()
            ws[axis] += delta
            ws = np.clip(ws, lo[4:7], hi[4:7])
            if not any(float(np.linalg.norm(ws - old, ord=np.inf)) < 1e-5 for old in unique_seeds):
                unique_seeds.append(ws)
    best_q = np.concatenate([shoulder_elbow, np.clip(seed, lo[4:7], hi[4:7])])
    best_err = float("inf")
    best_wrist_gap = float("inf")
    ok = False

    for ws0 in unique_seeds:
        q = np.concatenate([shoulder_elbow, ws0]).astype(np.float64)
        q = np.clip(q, lo, hi)
        for _ in range(int(max_steps)):
            _set_trunk_and_arm_probe(world, trunk_q, arm, q)
            eef = world.eef_pose(arm=arm)
            cur_q = _quat_normalize(eef["quat"])
            cur_R = _quat_to_mat(cur_q)
            omega = _orientation_error_omega(target_R, cur_R)
            err_deg = float(np.degrees(np.linalg.norm(omega)))
            wrist_gap = float(np.linalg.norm(q[4:7] - seed, ord=np.inf))
            if (
                err_deg + 1e-7 < best_err
                or (abs(err_deg - best_err) <= 1e-7 and wrist_gap < best_wrist_gap)
            ):
                best_err = err_deg
                best_wrist_gap = wrist_gap
                best_q = q.copy()
            if err_deg <= float(ori_tol_deg):
                ok = True
                break
            J, _ = _read_jacobian_arm(world, arm)
            Jw = J[3:6, 4:7]
            lam = 0.030 if err_deg > 2.0 else 0.055
            dq_wrist = _dls_solve_dq(Jw, omega, lam=lam)
            dq_inf = float(np.linalg.norm(dq_wrist, ord=np.inf))
            limit = 0.16 if err_deg > 6.0 else 0.10
            if dq_inf > limit:
                dq_wrist = dq_wrist * (limit / (dq_inf + 1e-12))
            q_next = q.copy()
            q_next[4:7] = np.clip(q[4:7] + dq_wrist, lo[4:7], hi[4:7])
            if float(np.linalg.norm(q_next[4:7] - q[4:7], ord=np.inf)) < 1e-8:
                break
            q = q_next
        if ok:
            break

    if best_err > float(ori_tol_deg):
        try:
            from scipy.optimize import least_squares

            def _residual(ws_raw) -> np.ndarray:
                ws = np.clip(np.asarray(ws_raw, dtype=np.float64).reshape(3), lo[4:7], hi[4:7])
                q_probe = np.concatenate([shoulder_elbow, ws]).astype(np.float64)
                _set_trunk_and_arm_probe(world, trunk_q, arm, q_probe)
                eef = world.eef_pose(arm=arm)
                cur_R = _quat_to_mat(_quat_normalize(eef["quat"]))
                omega = _orientation_error_omega(target_R, cur_R)
                reg = 0.002 * (ws - seed)
                return np.concatenate([omega, reg]).astype(np.float64)

            refine_seeds: List[np.ndarray] = [best_q[4:7].copy()]
            refine_seeds.extend(unique_seeds)
            for a in (-1.1, 0.0, 1.1):
                for b in (-1.1, 0.0, 1.1):
                    ws = np.clip(seed + np.array([a, 0.0, b], dtype=np.float64), lo[4:7], hi[4:7])
                    if not any(float(np.linalg.norm(ws - old, ord=np.inf)) < 1e-5 for old in refine_seeds):
                        refine_seeds.append(ws)
            for ws0 in refine_seeds:
                ls = least_squares(
                    _residual,
                    np.clip(ws0, lo[4:7], hi[4:7]),
                    bounds=(lo[4:7], hi[4:7]),
                    xtol=1e-5,
                    ftol=1e-5,
                    gtol=1e-5,
                    max_nfev=45,
                )
                ws = np.clip(np.asarray(ls.x, dtype=np.float64).reshape(3), lo[4:7], hi[4:7])
                q_probe = np.concatenate([shoulder_elbow, ws]).astype(np.float64)
                _set_trunk_and_arm_probe(world, trunk_q, arm, q_probe)
                eef = world.eef_pose(arm=arm)
                cur_R = _quat_to_mat(_quat_normalize(eef["quat"]))
                omega = _orientation_error_omega(target_R, cur_R)
                err_deg = float(np.degrees(np.linalg.norm(omega)))
                wrist_gap = float(np.linalg.norm(ws - seed, ord=np.inf))
                if (
                    err_deg + 1e-7 < best_err
                    or (abs(err_deg - best_err) <= 1e-7 and wrist_gap < best_wrist_gap)
                ):
                    best_err = err_deg
                    best_wrist_gap = wrist_gap
                    best_q = q_probe.copy()
                if err_deg <= float(ori_tol_deg):
                    ok = True
                    break
        except Exception:
            pass

    return best_q[4:7].copy(), float(best_err), bool(ok)


class _EefSmoothActiveBalancer:
    """Track EEF orientation with J5-J7 only during prescribed trunk motion.

    Trunk targets and entry J1-J4 commands are immutable. This controller
    synchronously places trunk + arm before every physics step, so wrist IK is
    solved against the exact J1-J4 command that will be placed. Only J5-J7 may
    change; measured J5-J7 is used only as the continuity seed.
    """

    def __init__(
        self,
        ctx,
        world,
        keep_ori_arm: str | None,
        *,
        log_prefix: str,
        prepared: bool = False,
        arm_q0: Optional[Dict[str, np.ndarray]] = None,
        target_quat_by_arm: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ctx = ctx
        self.world = world
        self.log_prefix = str(log_prefix)
        self.requested_arms = _normalize_keep_ori_arm(keep_ori_arm)
        self.arms: Set[str] = set()
        self.arm_q0: Dict[str, np.ndarray] = {}
        self.arm_q: Dict[str, np.ndarray] = {}
        self.eef_pos0: Dict[str, np.ndarray] = {}
        self.eef_pos_last: Dict[str, np.ndarray] = {}
        self.eef_pos_after: Dict[str, np.ndarray] = {}
        self.eef_pos_command: Dict[str, np.ndarray] = {}
        self.eef_delta_last: Dict[str, np.ndarray] = {}
        self.quat_entry: Dict[str, np.ndarray] = {}
        self.quat0: Dict[str, np.ndarray] = {}
        self.solve_err_m: Dict[str, float] = {}
        self.solve_err_deg: Dict[str, float] = {}
        self.solve_max_err_m: Dict[str, float] = {}
        self.solve_max_err_deg: Dict[str, float] = {}
        self.path_max_step_m: Dict[str, float] = {}
        self.path_max_accel_step_m: Dict[str, float] = {}
        self.path_max_err_deg: Dict[str, float] = {}
        self.path_samples: Dict[str, int] = {}
        self.position_violation_count: Dict[str, int] = {}
        self.orientation_limited_count: Dict[str, int] = {}
        self.orientation_rank_min: Dict[str, int] = {}
        self.mode_counts: Dict[str, Dict[str, int]] = {}
        self.last_mode: Dict[str, str] = {}
        self.j567_solve_count: Dict[str, int] = {}
        self.j567_global_search_count: Dict[str, int] = {}
        self.j567_solver_failure_count: Dict[str, int] = {}
        self.j567_feedback_max_ideal_gap_rad: Dict[str, float] = {}
        self.j567_feedback_max_compensation_rad: Dict[str, float] = {}
        self.j1234_command_max_error_rad: Dict[str, float] = {}
        self.j1234_measured_max_drift_rad: Dict[str, float] = {}
        self.snapshot_errors: Dict[str, str] = {}
        self.payload_hold: Dict[str, Dict[str, Any]] = {}

        if not self.requested_arms or getattr(world, "dry_run", False):
            return

        from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

        supplied_q = arm_q0 or {}
        supplied_target_quat = target_quat_by_arm or {}
        for arm in sorted(self.requested_arms):
            try:
                if not prepared:
                    _prepare_legacy_7dof_motion(
                        world,
                        arm,
                        ctx=ctx,
                        stage_name=f"{self.log_prefix}.eef_balance.{arm}.prepare",
                    )
                q_source = supplied_q[arm] if arm in supplied_q else _arm_qpos(world, arm)
                q0 = np.asarray(q_source, dtype=np.float64).reshape(7).copy()
                pos0, quat_entry = _eef_pose_np(world, arm)
                quat0 = _quat_normalize(
                    supplied_target_quat.get(arm, quat_entry),
                )
                self.arms.add(arm)
                self.arm_q0[arm] = q0
                self.arm_q[arm] = q0.copy()
                self.eef_pos0[arm] = pos0
                self.eef_pos_last[arm] = pos0.copy()
                self.eef_pos_after[arm] = pos0.copy()
                self.eef_pos_command[arm] = pos0.copy()
                self.eef_delta_last[arm] = np.zeros(3, dtype=np.float64)
                self.quat_entry[arm] = quat_entry
                self.quat0[arm] = quat0
                self.solve_err_m[arm] = 0.0
                self.solve_err_deg[arm] = 0.0
                self.solve_max_err_m[arm] = 0.0
                self.solve_max_err_deg[arm] = 0.0
                self.path_max_step_m[arm] = 0.0
                self.path_max_accel_step_m[arm] = 0.0
                self.path_max_err_deg[arm] = 0.0
                self.path_samples[arm] = 0
                self.position_violation_count[arm] = 0
                self.orientation_limited_count[arm] = 0
                self.orientation_rank_min[arm] = 3
                self.mode_counts[arm] = {}
                self.last_mode[arm] = "snapshot"
                self.j567_solve_count[arm] = 0
                self.j567_global_search_count[arm] = 0
                self.j567_solver_failure_count[arm] = 0
                self.j567_feedback_max_ideal_gap_rad[arm] = 0.0
                self.j567_feedback_max_compensation_rad[arm] = 0.0
                self.j1234_command_max_error_rad[arm] = 0.0
                self.j1234_measured_max_drift_rad[arm] = 0.0
                world.set_arm_pin_qpos(arm, q0)
                ctx.log(
                    f"{self.log_prefix} eef_balance {arm} "
                    f"pos0=[{','.join(f'{v:+.4f}' for v in pos0)}] "
                    f"quat_entry=[{','.join(f'{v:+.6f}' for v in quat_entry)}] "
                    f"quat_target=[{','.join(f'{v:+.6f}' for v in quat0)}] "
                    "policy=J1-J4 unchanged; J5-J7 orientation tracking "
                    f"ori_tol={_EEF_ORI_SOLVE_TOL_DEG:.1f}deg "
                    "translation=monitor_only"
                )
            except Exception as exc:
                self.snapshot_errors[arm] = f"{type(exc).__name__}: {exc}"
                ctx.log(
                    f"{self.log_prefix} eef_balance {arm} 初始化失败: "
                    f"{self.snapshot_errors[arm]}"
                )

        held_map = getattr(getattr(world, "robot", None), "_ag_obj_in_hand", None)
        if isinstance(held_map, dict) and any(held_map.get(a) is not None for a in held_map):
            try:
                from behavior_interface.skills.move_to_object_v2 import (
                    _snapshot_assisted_payload_hold,
                )

                self.payload_hold = _snapshot_assisted_payload_hold(world)
            except Exception as exc:
                ctx.log(
                    f"{self.log_prefix} eef_balance payload snapshot failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    @property
    def active(self) -> bool:
        return bool(self.arms)

    def solve_for_trunk(self, trunk_q) -> Dict[str, np.ndarray]:
        if not self.active:
            return {}
        trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
        measured_q = {
            arm: _arm_qpos(self.world, arm).copy()
            for arm in sorted(self.arms)
        }

        saved_q = _robot_joint_state_clone(self.world.robot)
        saved_vel = None
        try:
            saved_vel = self.world.robot.get_joint_velocities()
            try:
                saved_vel = saved_vel.clone()
            except Exception:
                saved_vel = np.asarray(saved_vel, dtype=np.float64).copy()
        except Exception:
            saved_vel = None

        try:
            for arm in sorted(self.arms):
                measured = measured_q[arm]
                nominal_j1234 = self.arm_q0[arm][:4].copy()
                measured_j1234 = measured[:4].copy()
                measured_j567 = measured[4:7].copy()
                measured_j1234_drift = float(
                    np.linalg.norm(
                        measured_j1234 - nominal_j1234,
                        ord=np.inf,
                    )
                )
                self.j1234_measured_max_drift_rad[arm] = max(
                    self.j1234_measured_max_drift_rad[arm],
                    measured_j1234_drift,
                )
                try:
                    from behavior_interface.skills.wrist_j567_solver import (
                        solve_wrist_j567_for_eef_quat,
                    )

                    # The caller directly places q_cmd before stepping physics.
                    # Solve the FK problem for that exact immutable J1-J4
                    # command; measured J5-J7 remains the continuity seed.
                    _set_trunk_and_arm_probe(
                        self.world,
                        trunk,
                        arm,
                        np.concatenate([
                            nominal_j1234,
                            measured_j567,
                        ]),
                    )
                    solve = solve_wrist_j567_for_eef_quat(
                        self.world,
                        arm,
                        self.quat0[arm],
                        fixed_q1234=nominal_j1234,
                        seed_j567=measured_j567,
                        ori_tol_deg=_EEF_ORI_SOLVE_TOL_DEG,
                        max_steps=24,
                        max_dq_rad=_EEF_IK_MAX_DQ_PER_STEP,
                        global_search=False,
                    )
                    self.j567_solve_count[arm] += 1
                    if not solve.get("ok", False):
                        solve = solve_wrist_j567_for_eef_quat(
                            self.world,
                            arm,
                            self.quat0[arm],
                            fixed_q1234=nominal_j1234,
                            seed_j567=measured_j567,
                            ori_tol_deg=_EEF_ORI_SOLVE_TOL_DEG,
                            max_steps=72,
                            max_dq_rad=_EEF_IK_MAX_DQ_PER_STEP,
                            global_search=True,
                        )
                        self.j567_solve_count[arm] += 1
                        self.j567_global_search_count[arm] += 1

                    ori_err = float(
                        solve.get("ori_err_deg", float("inf"))
                    )
                    orientation_rank = int(
                        solve.get("jacobian_rank", 0)
                    )
                    if solve.get("ok", False):
                        ideal_j567 = np.asarray(
                            solve["j567_rad"],
                            dtype=np.float64,
                        ).reshape(3)
                        ideal_error = ideal_j567 - measured_j567
                        ideal_gap = float(
                            np.linalg.norm(ideal_error, ord=np.inf)
                        )
                        previous_command_j567 = self.arm_q[arm][4:7]
                        lo, hi = _arm_joint_limits(self.world, arm)
                        ideal_j567 = np.clip(
                            ideal_j567,
                            lo[4:7],
                            hi[4:7],
                        )
                        wrist_step = np.clip(
                            ideal_j567 - previous_command_j567,
                            -_EEF_WRIST_COMMAND_MAX_STEP_RAD,
                            _EEF_WRIST_COMMAND_MAX_STEP_RAD,
                        )
                        wrist_command = previous_command_j567 + wrist_step
                        compensation = float(
                            np.linalg.norm(
                                wrist_command - ideal_j567,
                                ord=np.inf,
                            )
                        )
                        q_cmd = np.concatenate([
                            nominal_j1234,
                            wrist_command,
                        ]).astype(np.float64)
                        mode = "j567_tracking"
                        self.j567_feedback_max_ideal_gap_rad[arm] = max(
                            self.j567_feedback_max_ideal_gap_rad[arm],
                            ideal_gap,
                        )
                        self.j567_feedback_max_compensation_rad[arm] = max(
                            self.j567_feedback_max_compensation_rad[arm],
                            compensation,
                        )
                    else:
                        self.j567_solver_failure_count[arm] += 1
                        q_cmd = np.concatenate([
                            nominal_j1234,
                            self.arm_q[arm][4:7],
                        ]).astype(np.float64)
                        mode = "j567_limited"
                except Exception as exc:
                    self.j567_solver_failure_count[arm] += 1
                    q_cmd = np.concatenate([
                        nominal_j1234,
                        self.arm_q[arm][4:7],
                    ]).astype(np.float64)
                    ori_err = float("inf")
                    mode = "j567_exception"
                    orientation_rank = 0
                    self.ctx.log(
                        f"{self.log_prefix} eef_balance {arm} solve failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                command_j1234_error = float(
                    np.linalg.norm(
                        q_cmd[:4] - nominal_j1234,
                        ord=np.inf,
                    )
                )
                self.j1234_command_max_error_rad[arm] = max(
                    self.j1234_command_max_error_rad[arm],
                    command_j1234_error,
                )
                if command_j1234_error > 1e-10:
                    self.ctx.log(
                        f"{self.log_prefix} eef_balance {arm} invariant "
                        f"violation J1-J4 command error="
                        f"{command_j1234_error:.9f}rad; forcing snapshot"
                    )
                    q_cmd[:4] = nominal_j1234

                _set_trunk_and_arm_probe(self.world, trunk, arm, q_cmd)
                predicted_pos, predicted_quat = _eef_pose_np(self.world, arm)
                predicted_delta = predicted_pos - self.eef_pos_last[arm]
                predicted_step = float(np.linalg.norm(predicted_delta))
                predicted_accel = float(
                    np.linalg.norm(
                        predicted_delta - self.eef_delta_last[arm],
                    )
                )
                predicted_ori_err = _quat_err_deg(
                    self.quat0[arm],
                    predicted_quat,
                )
                if (
                    mode != "j567_tracking"
                    or predicted_ori_err > _EEF_ORI_SOLVE_TOL_DEG
                ):
                    self.orientation_limited_count[arm] += 1
                self.orientation_rank_min[arm] = min(
                    int(self.orientation_rank_min.get(arm, 3)),
                    int(orientation_rank),
                )
                self.arm_q[arm] = np.asarray(
                    q_cmd,
                    dtype=np.float64,
                ).reshape(7).copy()
                self.eef_pos_command[arm] = predicted_pos.copy()
                self.solve_err_m[arm] = float(predicted_step)
                self.solve_err_deg[arm] = float(predicted_ori_err)
                self.solve_max_err_m[arm] = max(
                    self.solve_max_err_m[arm],
                    float(predicted_step),
                )
                self.solve_max_err_deg[arm] = max(
                    self.solve_max_err_deg[arm],
                    float(predicted_ori_err),
                )
                self.last_mode[arm] = mode
                counts = self.mode_counts[arm]
                counts[mode] = int(counts.get(mode, 0)) + 1
                if (
                    mode != "j567_tracking"
                    or predicted_step > _EEF_POS_MONITOR_MAX_STEP_M
                    or predicted_accel > _EEF_POS_MONITOR_MAX_ACCEL_STEP_M
                ):
                    self.ctx.log(
                        f"{self.log_prefix} eef_balance {arm} mode={mode} "
                        f"pred_step={predicted_step * 1000.0:.1f}mm "
                        f"pred_accel={predicted_accel * 1000.0:.1f}mm/frame² "
                        f"ori={predicted_ori_err:.2f}deg "
                        f"J567_rank={orientation_rank}"
                    )
        finally:
            try:
                _set_full_joint_positions(self.world.robot, saved_q)
            except Exception:
                pass
            if saved_vel is not None:
                try:
                    self.world.robot.set_joint_velocities(saved_vel)
                except Exception:
                    pass

        return {arm: self.arm_q[arm].copy() for arm in sorted(self.arms)}

    def apply_payload_hold(self) -> None:
        if not self.payload_hold:
            return
        try:
            from behavior_interface.skills.move_to_object_v2 import (
                _force_apply_assisted_payload_hold,
            )

            _force_apply_assisted_payload_hold(self.world, self.payload_hold)
        except Exception:
            pass

    def after_step(self) -> None:
        if not self.active:
            return
        self.apply_payload_hold()
        for arm in sorted(self.arms):
            try:
                pos, quat = _eef_pose_np(self.world, arm)
                delta = pos - self.eef_pos_last[arm]
                step_m = float(np.linalg.norm(delta))
                accel_step_m = float(
                    np.linalg.norm(delta - self.eef_delta_last[arm])
                )
                ori_err = _quat_err_deg(self.quat0[arm], quat)
                self.path_max_step_m[arm] = max(
                    self.path_max_step_m[arm], step_m,
                )
                self.path_max_accel_step_m[arm] = max(
                    self.path_max_accel_step_m[arm], accel_step_m,
                )
                self.path_max_err_deg[arm] = max(
                    self.path_max_err_deg[arm], ori_err,
                )
                self.path_samples[arm] += 1
                if (
                    step_m > _EEF_POS_MONITOR_MAX_STEP_M
                    or accel_step_m > _EEF_POS_MONITOR_MAX_ACCEL_STEP_M
                ):
                    self.position_violation_count[arm] += 1
                self.eef_pos_last[arm] = pos.copy()
                self.eef_pos_after[arm] = pos.copy()
                self.eef_delta_last[arm] = delta.copy()
                if self.path_samples[arm] == 1 or self.path_samples[arm] % 4 == 0:
                    self.ctx.log(
                        f"{self.log_prefix} eef_{arm}_pos="
                        f"[{','.join(f'{v:+.4f}' for v in pos)}] "
                        f"step={step_m * 1000.0:.1f}mm "
                        f"accel={accel_step_m * 1000.0:.1f}mm/frame² "
                        f"ori={ori_err:.2f}deg mode={self.last_mode[arm]}"
                    )
            except Exception:
                self.position_violation_count[arm] += 1
                self.path_max_step_m[arm] = float("inf")
                self.path_max_accel_step_m[arm] = float("inf")
                self.path_max_err_deg[arm] = float("inf")

    def status_text(self) -> str:
        parts: List[str] = []
        for arm in sorted(self.arms):
            pos = self.eef_pos_after.get(arm)
            pos_text = (
                f"[{','.join(f'{v:+.3f}' for v in pos)}]"
                if pos is not None else "?"
            )
            parts.append(
                f"eef_{arm}_pos={pos_text} "
                f"dmax={self.path_max_step_m.get(arm, float('nan')) * 1000.0:.1f}mm "
                f"ori={self.path_max_err_deg.get(arm, float('nan')):.1f}deg"
            )
        return " ".join(parts)

    @staticmethod
    def _finite_or_none(value: float):
        return round(float(value), 6) if np.isfinite(value) else None

    def report(self, *, scope: str) -> Dict[str, Any]:
        final_pos: Dict[str, List[float]] = {}
        final_quat: Dict[str, List[float]] = {}
        final_ori_err: Dict[str, float] = {}
        for arm in sorted(self.arms):
            try:
                pos, quat = _eef_pose_np(self.world, arm)
                final_pos[arm] = pos.round(6).tolist()
                final_quat[arm] = quat.round(6).tolist()
                final_ori_err[arm] = _quat_err_deg(self.quat0[arm], quat)
            except Exception:
                final_ori_err[arm] = float("inf")

        pos_smooth_ok = (
            self.requested_arms == self.arms
            and all(
                int(self.position_violation_count.get(arm, 0)) == 0
                and np.isfinite(self.path_max_step_m.get(arm, float("inf")))
                and self.path_max_step_m[arm] <= _EEF_POS_MONITOR_MAX_STEP_M
                and np.isfinite(
                    self.path_max_accel_step_m.get(arm, float("inf")),
                )
                and self.path_max_accel_step_m[arm]
                <= _EEF_POS_MONITOR_MAX_ACCEL_STEP_M
                for arm in self.requested_arms
            )
        )
        ori_tracking_ok = (
            self.requested_arms == self.arms
            and all(
                np.isfinite(final_ori_err.get(arm, float("inf")))
                and final_ori_err[arm] <= _EEF_ORI_TRACK_OK_DEG
                and np.isfinite(self.path_max_err_deg.get(arm, float("inf")))
                and self.path_max_err_deg[arm] <= _EEF_ORI_TRACK_OK_DEG
                for arm in self.requested_arms
            )
        )
        j1234_unchanged = (
            self.requested_arms == self.arms
            and all(
                float(self.j1234_command_max_error_rad.get(arm, float("inf")))
                <= 1e-10
                for arm in self.requested_arms
            )
        )
        j567_solver_ok = (
            self.requested_arms == self.arms
            and all(
                int(self.j567_solver_failure_count.get(arm, 0)) == 0
                for arm in self.requested_arms
            )
        )
        payload_end: Dict[str, Any] = {}
        payload_max: Dict[str, Any] = {}
        if self.payload_hold:
            try:
                from behavior_interface.skills.move_to_object_v2 import (
                    _payload_end_drift,
                    _rounded_payload_drift,
                )

                payload_end = _rounded_payload_drift(
                    _payload_end_drift(self.world, self.payload_hold),
                )
                payload_max = _rounded_payload_drift(
                    getattr(
                        self.world,
                        "_codex_motion_payload_max_drift",
                        None,
                    ),
                )
            except Exception:
                pass
        return {
            "keep_ori_scope": str(scope),
            "keep_ori_requested_arm": sorted(self.requested_arms),
            "keep_ori_arm": sorted(self.arms),
            "keep_ori_ok": bool(
                ori_tracking_ok and j1234_unchanged and j567_solver_ok
            ),
            "keep_ori_tracking_ok": bool(ori_tracking_ok),
            "eef_pos_smooth_ok": bool(pos_smooth_ok),
            "eef_balance_policy": "j1234_unchanged_j567_orientation_tracking",
            "eef_position_policy": "monitor_only_no_position_control",
            "j1234_command_unchanged": bool(j1234_unchanged),
            "keep_ori_solve_tol_deg": _EEF_ORI_SOLVE_TOL_DEG,
            "keep_ori_tracking_tol_deg": _EEF_ORI_TRACK_OK_DEG,
            "keep_ori_snapshot_errors": dict(self.snapshot_errors),
            "eef_pos_before": {
                arm: pos.round(6).tolist() for arm, pos in self.eef_pos0.items()
            },
            "eef_pos_after": final_pos,
            "eef_pos_path_max_step_m": {
                arm: self._finite_or_none(value)
                for arm, value in self.path_max_step_m.items()
            },
            "eef_pos_path_max_accel_step_m": {
                arm: self._finite_or_none(value)
                for arm, value in self.path_max_accel_step_m.items()
            },
            "eef_pos_violation_count": dict(self.position_violation_count),
            "keep_ori_limited_count": dict(self.orientation_limited_count),
            "keep_ori_jacobian_rank_min": dict(self.orientation_rank_min),
            "j567_solve_count": dict(self.j567_solve_count),
            "j567_global_search_count": dict(
                self.j567_global_search_count
            ),
            "j567_solver_failure_count": dict(
                self.j567_solver_failure_count
            ),
            "j567_feedback_max_ideal_gap_rad": {
                arm: self._finite_or_none(value)
                for arm, value in self.j567_feedback_max_ideal_gap_rad.items()
            },
            "j567_feedback_max_compensation_rad": {
                arm: self._finite_or_none(value)
                for arm, value in (
                    self.j567_feedback_max_compensation_rad.items()
                )
            },
            "j1234_command_max_error_rad": {
                arm: self._finite_or_none(value)
                for arm, value in self.j1234_command_max_error_rad.items()
            },
            "j1234_measured_max_drift_rad": {
                arm: self._finite_or_none(value)
                for arm, value in self.j1234_measured_max_drift_rad.items()
            },
            "eef_balance_mode_counts": {
                arm: dict(counts) for arm, counts in self.mode_counts.items()
            },
            "keep_ori_quat_before": {
                arm: quat.round(6).tolist()
                for arm, quat in self.quat_entry.items()
            },
            "keep_ori_target_quat": {
                arm: quat.round(6).tolist() for arm, quat in self.quat0.items()
            },
            "keep_ori_quat_after": final_quat,
            "keep_ori_err_deg": {
                arm: self._finite_or_none(value)
                for arm, value in final_ori_err.items()
            },
            "keep_ori_path_max_err_deg": {
                arm: self._finite_or_none(value)
                for arm, value in self.path_max_err_deg.items()
            },
            "keep_ori_solve_max_err_deg": {
                arm: self._finite_or_none(value)
                for arm, value in self.solve_max_err_deg.items()
            },
            "eef_pos_solve_max_err_m": {
                arm: self._finite_or_none(value)
                for arm, value in self.solve_max_err_m.items()
            },
            "eef_passive_residual_max_m": {
                arm: self._finite_or_none(value)
                for arm, value in self.solve_max_err_m.items()
            },
            "keep_ori_path_samples": dict(self.path_samples),
            "keep_ori_arm_qpos": {
                arm: q.round(5).tolist() for arm, q in self.arm_q.items()
            },
            "keep_ori_wrist_qpos": {
                arm: q[4:7].round(5).tolist()
                for arm, q in self.arm_q.items()
            },
            "payload_hold": {
                arm: spec.get("name") for arm, spec in self.payload_hold.items()
            },
            "payload_max_drift": payload_max,
            "payload_end_drift": payload_end,
        }


def _locked_limb_overrides(
    arm_hold: Dict[str, np.ndarray],
    gripper_hold: Dict[str, Optional[List[float]]],
    keep_ori_arms: Optional[Set[str]] = None,
    keep_ori_arm_q: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    keep_ori_arms = keep_ori_arms or set()
    keep_ori_arm_q = keep_ori_arm_q or {}
    out: Dict[str, Any] = {}
    for arm, q in arm_hold.items():
        q_cmd = keep_ori_arm_q.get(arm) if arm in keep_ori_arms else q
        out[f"arm_{arm}"] = np.asarray(q_cmd, dtype=np.float64).reshape(7).tolist()
        grip = gripper_hold.get(arm)
        if grip is not None:
            out[f"gripper_{arm}"] = list(grip)
    return out


def _arm_drift_report(world, arm_hold: Dict[str, np.ndarray]) -> Dict[str, float]:
    report: Dict[str, float] = {}
    for arm, q0 in arm_hold.items():
        try:
            cur = _arm_qpos(world, arm)
            report[arm] = float(np.linalg.norm(cur - q0, ord=np.inf))
        except Exception:
            report[arm] = float("nan")
    return report


def _arm_locked_drift_report(
    world,
    arm_hold: Dict[str, np.ndarray],
    keep_ori_arms: Set[str],
) -> Dict[str, float]:
    report: Dict[str, float] = {}
    for arm, q0 in arm_hold.items():
        try:
            cur = _arm_qpos(world, arm)
            if arm in keep_ori_arms:
                report[arm] = float(np.linalg.norm(cur[:4] - q0[:4], ord=np.inf))
            else:
                report[arm] = float(np.linalg.norm(cur - q0, ord=np.inf))
        except Exception:
            report[arm] = float("nan")
    return report


def _record_keep_ori_path_error(
    world,
    *,
    keep_ori_arms: Set[str],
    keep_ori_quat0: Dict[str, np.ndarray],
    keep_ori_path_max_err_deg: Dict[str, float],
    keep_ori_path_samples: Dict[str, int],
) -> None:
    for arm in sorted(keep_ori_arms):
        try:
            q_now = _quat_normalize(world.eef_pose(arm=arm)["quat"])
            path_err = _quat_err_deg(keep_ori_quat0[arm], q_now)
            keep_ori_path_max_err_deg[arm] = max(
                float(keep_ori_path_max_err_deg.get(arm, 0.0)),
                float(path_err),
            )
            keep_ori_path_samples[arm] = int(keep_ori_path_samples.get(arm, 0)) + 1
        except Exception:
            keep_ori_path_max_err_deg[arm] = float("inf")


def _yield_reset_body(
    ctx,
    *,
    timeout_s: float = 45.0,
    trunk_max_step: float = 0.06,
    shoulder_iters_per_trunk: int = 4,
    keep_ori_arm: str = "none",
    pitch_deg: float = 0.0,
    keep_ori_target_quat: Optional[Dict[str, Any]] = None,
) -> Generator:
    world = ctx.world
    if world.dry_run:
        ctx.log("[reset_body] dry_run 跳过")
        ctx.set_result({"ok": True, "dry_run": True, "build": RESET_BODY_BUILD})
        yield world.hold_action()
        return

    # 直接运动学设俯身：pitch_deg = 躯干前倾角(°)，0=直立；前倾 → q3 取负。
    # trunk_q=[0,0,q3,0]，q3=-radians(pitch_deg)，纯 q3、从直立出发，不折叠 q1/q2。
    if abs(float(pitch_deg)) > 1e-6:
        import math as _math
        q3 = -_math.radians(float(pitch_deg))
        target_trunk = np.array([0.0, 0.0, q3, 0.0], dtype=np.float64)
        base_pose = _robot_base_pose_np(world)
        requested_arms = _normalize_keep_ori_arm(keep_ori_arm)
        arm_targets: Dict[str, np.ndarray] = {}
        gripper_targets: Dict[str, Optional[List[float]]] = {}
        for _arm in ("left", "right"):
            try:
                arm_targets[_arm] = _arm_qpos(world, _arm).copy()
                gripper_targets[_arm] = _gripper_qpos(world, _arm)
            except Exception:
                pass
        balancer = _EefSmoothActiveBalancer(
            ctx,
            world,
            " ".join(sorted(requested_arms)),
            log_prefix="[reset_body/pitch]",
            arm_q0=arm_targets,
            target_quat_by_arm=keep_ori_target_quat,
        )
        if requested_arms != balancer.arms:
            report = balancer.report(scope="reset_body_kinematic_pitch")
            report.update({
                "ok": False,
                "mode": "kinematic_pitch",
                "build": RESET_BODY_BUILD,
                "pitch_deg": float(pitch_deg),
                "error": "keep_ori_arm 初始化失败；为避免无补偿 trunk 运动已中止",
            })
            ctx.set_result(report)
            yield world.hold_action()
            return
        for _ in range(40):
            arm_commands = {
                arm: q.copy() for arm, q in arm_targets.items()
            }
            arm_commands.update(balancer.solve_for_trunk(target_trunk))
            _set_trunk_and_arms_direct(world, target_trunk, arm_commands)
            balancer.apply_payload_hold()
            _hold_robot_base(world, base_pose)
            kw: Dict[str, Any] = {"trunk": target_trunk.tolist()}
            for arm, q in arm_commands.items():
                kw[f"arm_{arm}"] = q.tolist()
                grip = gripper_targets.get(arm)
                if grip is not None:
                    kw[f"gripper_{arm}"] = list(grip)
            yield world.make_action(**kw)
            _hold_robot_base(world, base_pose)
            balancer.after_step()
        tz = None
        try:
            tz = float(world.chest_pose().get("theta_z_deg"))
        except Exception:
            pass
        ctx.log(
            f"[reset_body] kinematic_pitch pitch_deg={float(pitch_deg):.1f} "
            f"q3={_math.degrees(q3):.1f}° chest_theta_z={tz}"
        )
        balance_report = balancer.report(scope="reset_body_kinematic_pitch")
        result = {
            "ok": bool(balance_report.get("keep_ori_ok", True)),
            "mode": "kinematic_pitch",
            "build": RESET_BODY_BUILD,
            "pitch_deg": float(pitch_deg),
            "trunk_q": [round(float(v), 4) for v in target_trunk],
            "chest_theta_z_deg": tz,
        }
        result.update(balance_report)
        result["ok"] = bool(balance_report.get("keep_ori_ok", True))
        ctx.set_result(result)
        return

    try:
        _ = world.controller_action_idx("trunk")
    except Exception as e:
        ctx.set_result({"ok": False, "error": f"无 trunk controller: {e}"})
        yield world.hold_action()
        return

    trunk0 = world.trunk_qpos().copy()
    requested_keep_ori_arms = _normalize_keep_ori_arm(keep_ori_arm)
    arm_hold: Dict[str, np.ndarray] = {}
    gripper_hold: Dict[str, Optional[List[float]]] = {}
    for arm in ("left", "right"):
        try:
            arm_hold[arm] = _arm_qpos(world, arm).copy()
            gripper_hold[arm] = _gripper_qpos(world, arm)
        except Exception as e:
            ctx.log(f"[reset_body] 警告: 读取 {arm} arm qpos 失败，无法显式锁该臂: {e}")

    balancer = _EefSmoothActiveBalancer(
        ctx,
        world,
        " ".join(sorted(requested_keep_ori_arms)),
        log_prefix="[reset_body]",
        prepared=False,
        arm_q0=arm_hold,
        target_quat_by_arm=keep_ori_target_quat,
    )
    keep_ori_arms = set(balancer.arms)
    if requested_keep_ori_arms != keep_ori_arms:
        report = balancer.report(scope="reset_body_trunk_motion")
        report.update({
            "ok": False,
            "build": RESET_BODY_BUILD,
            "trunk_before": trunk0.round(4).tolist(),
            "trunk_after": trunk0.round(4).tolist(),
            "error": "keep_ori_arm 初始化失败；为避免无补偿 trunk 运动已中止",
        })
        ctx.set_result(report)
        yield world.hold_action()
        return
    if keep_ori_arms:
        from behavior_interface.skills.eef import _assert_legacy_7dof_motion_ready

    ctx.log(
        f"[reset_body] BUILD={RESET_BODY_BUILD} trunk0="
        f"[{','.join(f'{v:+.3f}' for v in trunk0)}] "
        f"upright=[0,0,0,0] arms_locked={list(arm_hold.keys())} "
        f"keep_ori_arm={sorted(keep_ori_arms)} "
        f"shoulder_iters_ignored={shoulder_iters_per_trunk}"
    )
    for arm, q in arm_hold.items():
        ctx.log(f"[reset_body] lock {arm} arm q=[{','.join(f'{v:+.3f}' for v in q)}]")

    timeout_total = float(timeout_s)
    step_limit = float(trunk_max_step)
    if not np.isfinite(timeout_total) or timeout_total <= 0.0:
        ctx.set_result({
            "ok": False,
            "build": RESET_BODY_BUILD,
            "error": "timeout_s must be a positive finite value",
        })
        yield world.hold_action()
        return
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        ctx.set_result({
            "ok": False,
            "build": RESET_BODY_BUILD,
            "error": "trunk_max_step must be a positive finite value",
        })
        yield world.hold_action()
        return

    started_at = time.monotonic()
    deadline = started_at + timeout_total
    last_log = started_at
    trunk_done = False
    # 失速检测：躯干到直立(upright=[0,0,0,0]) 的 inf 距离若连续多步不再下降，
    # 说明位置控制器扭矩不足、抬不动深俯/塌陷躯干（重力 + 锁定双臂）。
    prev_err: Optional[float] = None
    stall = 0
    kinematic_snapped = False
    action_only_fallback_played = False
    fallback_playback: Optional[Dict[str, Any]] = None
    action_only = _challenge_action_only()
    # reset_body 不移动底盘：快照底盘世界位姿，运动学吸附躯干时每帧把它钉回去，
    # 避免突变构型把整机掀飞/翻倒。
    base_pose_hold = _robot_base_pose_np(world)

    def _arm_targets_for_snap() -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for a, q in arm_hold.items():
            out[a] = balancer.arm_q.get(a, q) if a in keep_ori_arms else q
        return out

    def _finish_timeout(q_trunk: np.ndarray, *, phase: str) -> None:
        elapsed_s = max(0.0, time.monotonic() - started_at)
        report = balancer.report(scope="reset_body_trunk_motion")
        report.update({
            "ok": False,
            "build": RESET_BODY_BUILD,
            "trunk_before": trunk0.round(4).tolist(),
            "trunk_after": np.asarray(q_trunk).round(4).tolist(),
            "kinematic_snapped": bool(kinematic_snapped),
            "action_only_fallback_played": bool(action_only_fallback_played),
            "fallback_playback": fallback_playback,
            "timed_out": True,
            "timeout_s": timeout_total,
            "elapsed_s": round(elapsed_s, 3),
            "timeout_phase": str(phase),
            "error": "reset_body timed out before trunk reached upright",
        })
        ctx.log(
            f"[reset_body] TIMEOUT phase={phase} elapsed={elapsed_s:.1f}s "
            f"trunk=[{','.join(f'{v:+.3f}' for v in np.asarray(q_trunk).reshape(4))}]"
        )
        ctx.set_result(report)

    while True:
        q_trunk = world.trunk_qpos()
        trunk_done = _trunk_upright_reached(q_trunk)

        # 当前到直立的 inf 误差（upright 全 0）
        cur_err = float(
            np.linalg.norm(np.asarray(q_trunk, dtype=np.float64).reshape(4), ord=np.inf)
        )
        if prev_err is not None and cur_err > prev_err - 1e-3:
            stall += 1
        else:
            stall = 0
        prev_err = cur_err

        now = time.monotonic()
        elapsed = now - started_at
        timed_out = now >= deadline
        if not trunk_done and timed_out:
            _finish_timeout(q_trunk, phase="controller")
            return
        # 软超时：控制器抬深俯躯干很慢（实测 ~45s 才接近直立），给它 _KINEMATIC_AFTER_S
        # 尝试，仍没到位就提前切运动学，既保留平滑控制器起步又不至于干等满硬超时。
        soft_switch = (elapsed > _KINEMATIC_AFTER_S)
        # 关键：非 shortcut 也必须有运动学兜底。控制器失速 / 软超时 / 硬超时，就逐帧运动学
        # 插值开到直立 [0,0,0,0]（连同锁定双臂、每帧锁底盘），保证任何塌陷/深俯姿态都能
        # 100% 复位到精确直立，而不是像旧版那样 TIMEOUT 失败、把机器人留在塌陷穿模态。
        if (
            not trunk_done
            and not kinematic_snapped
            and (not action_only_fallback_played or not action_only)
            and (stall >= 6 or soft_switch or timed_out)
        ):
            ctx.log(
                f"[reset_body] 控制器失速/软超时/超时（stall={stall} elapsed={elapsed:.1f}s "
                f"timed_out={timed_out} q_err={cur_err:.4f}rad），改用逐帧运动学插值开到直立 "
                f"[0,0,0,0]（锁底盘）"
            )
            fallback_playback = yield from _kinematic_drive_trunk_to_upright(
                ctx,
                world,
                _arm_targets_for_snap(),
                base_pose_hold,
                n_frames=18,
                max_step=step_limit,
                deadline=deadline,
                legacy_7dof_arms=keep_ori_arms,
                eef_balancer=balancer,
            )
            q_trunk = world.trunk_qpos()
            trunk_done = _trunk_upright_reached(q_trunk)
            if action_only:
                action_only_fallback_played = True
                stall = 0
            else:
                kinematic_snapped = bool(trunk_done)
            if not trunk_done and (
                bool((fallback_playback or {}).get("timed_out"))
                or time.monotonic() >= deadline
            ):
                _finish_timeout(q_trunk, phase="fallback")
                return

        if not trunk_done:
            q_trunk_tgt = _trunk_step_toward_upright(q_trunk, max_step=step_limit)
        else:
            q_trunk_tgt = _UPRIGHT_TRUNK.copy()

        if keep_ori_arms:
            balancer.solve_for_trunk(q_trunk_tgt)

        overrides: Dict[str, Any] = _locked_limb_overrides(
            arm_hold,
            gripper_hold,
            keep_ori_arms=keep_ori_arms,
            keep_ori_arm_q=balancer.arm_q,
        )
        overrides["trunk"] = q_trunk_tgt.tolist()
        if keep_ori_arms:
            try:
                for arm in sorted(keep_ori_arms):
                    _assert_legacy_7dof_motion_ready(world, arm)
                direct_arm_q = {
                    arm: np.asarray(overrides[f"arm_{arm}"], dtype=np.float64).reshape(7)
                    for arm in arm_hold
                    if f"arm_{arm}" in overrides
                }
                _set_trunk_and_arms_direct(world, q_trunk_tgt, direct_arm_q)
                balancer.apply_payload_hold()
            except Exception as e:
                ctx.log(f"[reset_body] direct keep_ori waypoint set failed: {type(e).__name__}: {e}")

        if keep_ori_arms:
            for arm in sorted(keep_ori_arms):
                _assert_legacy_7dof_motion_ready(world, arm)
        yield world.make_action(**overrides)
        balancer.after_step()

        now = time.monotonic()
        if now - last_log > 0.5:
            drift = _arm_drift_report(world, arm_hold)
            parts = [
                f"trunk=[{','.join(f'{v:+.2f}' for v in world.trunk_qpos()[:3])}]",
            ]
            for arm, d in drift.items():
                parts.append(f"{arm}_dq_inf={d:.4f}")
            if balancer.active:
                parts.append(balancer.status_text())
            ctx.set_status(f"reset_body {' '.join(parts)}")
            last_log = now

        if trunk_done:
            trunk_fin = world.trunk_qpos()
            settle_steps = 0
            for _ in range(8):
                if time.monotonic() >= deadline:
                    break
                if keep_ori_arms:
                    balancer.solve_for_trunk(_UPRIGHT_TRUNK)
                kw = _locked_limb_overrides(
                    arm_hold,
                    gripper_hold,
                    keep_ori_arms=keep_ori_arms,
                    keep_ori_arm_q=balancer.arm_q,
                )
                kw["trunk"] = _UPRIGHT_TRUNK.tolist()
                # 运动学吸附后，settle 期间每步再运动学重设 + 锁底盘，抵消控制器回拉，确保锁住直立
                if kinematic_snapped:
                    if keep_ori_arms:
                        for arm in sorted(keep_ori_arms):
                            _assert_legacy_7dof_motion_ready(world, arm)
                    _set_trunk_and_arms_direct(world, _UPRIGHT_TRUNK, _arm_targets_for_snap())
                    balancer.apply_payload_hold()
                    _hold_robot_base(world, base_pose_hold)
                if keep_ori_arms:
                    for arm in sorted(keep_ori_arms):
                        _assert_legacy_7dof_motion_ready(world, arm)
                yield world.make_action(**kw)
                settle_steps += 1
                if kinematic_snapped:
                    _hold_robot_base(world, base_pose_hold)
                balancer.after_step()
            arm_drift = _arm_drift_report(world, arm_hold)
            locked_drift: Dict[str, float] = {}
            for arm, q0 in arm_hold.items():
                try:
                    q_target = balancer.arm_q.get(arm, q0)
                    locked_drift[arm] = float(
                        np.linalg.norm(_arm_qpos(world, arm) - q_target, ord=np.inf)
                    )
                except Exception:
                    locked_drift[arm] = float("nan")
            finite_drifts = [d for d in locked_drift.values() if np.isfinite(d)]
            max_arm_drift = float(max(finite_drifts)) if finite_drifts else 0.0
            arm_ok = max_arm_drift < 0.08
            balance_report = balancer.report(scope="reset_body_trunk_motion")
            balance_ok = bool(
                balance_report.get("keep_ori_ok", not keep_ori_arms)
            )
            ctx.log(
                f"[reset_body] DONE trunk="
                f"[{','.join(f'{v:+.3f}' for v in trunk_fin)}] "
                f"arm_drift_inf_rad={{{', '.join(f'{a}: {d:.4f}' for a, d in arm_drift.items())}}} "
                f"eef_pos_max_step_m={balance_report.get('eef_pos_path_max_step_m')} "
                f"keep_ori_final_err_deg={balance_report.get('keep_ori_err_deg')} "
                f"keep_ori_path_max_err_deg={balance_report.get('keep_ori_path_max_err_deg')}"
            )
            result = {
                "ok": bool(arm_ok and balance_ok),
                "build": RESET_BODY_BUILD,
                "kinematic_snapped": bool(kinematic_snapped),
                "trunk_before": trunk0.round(4).tolist(),
                "trunk_after": trunk_fin.round(4).tolist(),
                "arms_locked": list(arm_hold.keys()),
                "arm_hold_qpos": {
                    arm: np.asarray(q, dtype=np.float64).round(4).tolist()
                    for arm, q in arm_hold.items()
                },
                "arm_drift_inf_rad": {
                    arm: round(float(d), 5) if np.isfinite(d) else None
                    for arm, d in arm_drift.items()
                },
                "locked_drift_inf_rad": {
                    arm: round(float(d), 5) if np.isfinite(d) else None
                    for arm, d in locked_drift.items()
                },
                "max_arm_drift_inf_rad": round(max_arm_drift, 5),
                "timed_out": False,
                "timeout_s": timeout_total,
                "elapsed_s": round(
                    max(0.0, time.monotonic() - started_at),
                    3,
                ),
                "settle_steps": int(settle_steps),
                "action_only_fallback_played": bool(
                    action_only_fallback_played
                ),
                "fallback_playback": fallback_playback,
                "error": (
                    None
                    if (arm_ok and balance_ok)
                    else "reset_body 后姿态跟踪失败、J1-J4 命令被修改、J567 求解失败或手臂漂移过大"
                ),
            }
            result.update(balance_report)
            result["ok"] = bool(arm_ok and balance_ok)
            ctx.set_result(result)
            return


@register_skill(
    "reset_body",
    description=(
        "腰部三关节复位直立（q1=q2=q3=0）；底盘不动。"
        "keep_ori_arm 指定的手臂保持入口 J1-J4 命令不变，"
        "仅用 J5-J7 追踪入口世界系 EEF 姿态；"
        "EEF 平移只监控、不参与控制；"
        "未指定手臂和夹爪保持进入 skill 时的当前关节角。"
    ),
)
def reset_body(
    ctx,
    timeout_s: float = 45.0,
    trunk_max_step: float = 0.06,
    shoulder_iters: int = 4,
    keep_ori_arm: str = "none",
    pitch_deg: float = 0.0,
    keep_ori_target_quat: Optional[Dict[str, Any]] = None,
) -> Generator:
    yield from _yield_reset_body(
        ctx,
        timeout_s=float(timeout_s),
        trunk_max_step=float(trunk_max_step),
        shoulder_iters_per_trunk=int(shoulder_iters),
        keep_ori_arm=str(keep_ori_arm),
        pitch_deg=float(pitch_deg),
        keep_ori_target_quat=keep_ori_target_quat,
    )
