"""move_eef：在 head 相机坐标系下平移 EEF（姿态不变），可选先开合夹爪。

相机系（与 head / USD 针孔一致）：
  +X = 画面右，+Y = 画面上，+Z = 朝向观察者（出屏）
  forward  = 沿视线进入场景 → 相机 -Z
  upward   = 相机 +Y
  leftward = 相机 -X

顺序：若夹爪目标与当前不同 → 先动夹爪（手臂关节保持）→ 再平移 EEF。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill

MOVE_EEF_BUILD = "v11_explicit_official_ag_close_latch"
_GRIPPER_SETTLE_STEPS = 6
_GRIPPER_CLOSE_SETTLE_STEPS = 9
_ARM_HOLD_STEPS = 20  # 结束/失败后锁定关节，避免控制器继续追上一帧 IK 目标
_FINGER_OPEN_THRESH = 0.035  # m，两指均值 ≥ 此视为已张开
_ORI_DRIFT_OK_DEG = 3.0


def _parse_gripper_target(gripper: str) -> Optional[float]:
    """open → +1，close → -1，空/keep → 不改变。"""
    if gripper is None:
        return None
    g = str(gripper).strip().lower()
    if g in ("", "keep", "none", "skip", "unchanged"):
        return None
    if g in ("open", "o"):
        return 1.0
    if g in ("close", "closed", "c"):
        return -1.0
    raise ValueError(f"gripper 必须是 open/close/keep，收到 {gripper!r}")


def _finger_qpos_pair(world, arm: str) -> Optional[Tuple[float, float]]:
    if getattr(world, "dry_run", False):
        return None
    try:
        robot = world.robot
        qpos = robot.get_joint_positions()
        names = list(robot.joints.keys())
        j1 = f"{arm}_gripper_finger_joint1"
        j2 = f"{arm}_gripper_finger_joint2"
        return float(qpos[names.index(j1)]), float(qpos[names.index(j2)])
    except Exception:
        return None


def _finger_mean_qpos(world, arm: str) -> Optional[float]:
    pair = _finger_qpos_pair(world, arm)
    if pair is None:
        return None
    return 0.5 * (pair[0] + pair[1])


def _gripper_is_open(world, arm: str) -> bool:
    pair = _finger_qpos_pair(world, arm)
    if pair is None:
        return True
    return min(pair) >= _FINGER_OPEN_THRESH


def _as_gripper_action(cmd) -> Optional[list[float]]:
    if cmd is None:
        return None
    try:
        arr = np.asarray(cmd, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.asarray([float(cmd)], dtype=np.float64)
    if arr.size <= 0:
        return None
    return [float(x) for x in arr.tolist()]


def _current_gripper_hold_cmd(world, arm: str):
    try:
        if world.gripper_uses_effort(arm) and world.gripper_pin_effort_list(arm) is not None:
            return None
    except Exception:
        pass
    try:
        vals = world.gripper_qpos_list(arm)
        if vals is not None:
            out = _as_gripper_action(vals)
            if out:
                return out
    except Exception:
        pass
    pair = _finger_qpos_pair(world, arm)
    if pair is not None:
        return [float(pair[0]), float(pair[1])]
    return 1.0 if _gripper_is_open(world, arm) else -1.0


def _head_cam_pose(world) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    from behavior_interface.head_capture import get_head_sensor

    head = get_head_sensor(world)
    if head is None:
        return None, None
    cp, cq = head.get_position_orientation()
    if hasattr(cp, "detach"):
        cp = cp.detach().cpu().numpy()
        cq = cq.detach().cpu().numpy()
    return (
        np.asarray(cp, dtype=np.float64).reshape(3),
        np.asarray(cq, dtype=np.float64).reshape(4),
    )


def _head_sensor(world):
    from behavior_interface.head_capture import get_head_sensor

    return get_head_sensor(world)


def _head_camera_meta(sensor) -> Tuple[int, int, float, float]:
    from behavior_interface.head_capture import head_camera_size

    w, h = head_camera_size(sensor)
    fl = float(getattr(sensor, "focal_length", 17.0))
    ha = float(getattr(sensor, "horizontal_aperture", 40.0))
    return int(w), int(h), fl, ha


def _has_value(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def camera_delta_to_world(
    cam_quat_xyzw: np.ndarray,
    *,
    upward_cm: float = 0.0,
    forward_cm: float = 0.0,
    leftward_cm: float = 0.0,
) -> np.ndarray:
    """相机系增量 (cm) → 世界系位移向量 (m)。"""
    from behavior_interface.skills.grasp import _quat_to_mat

    R = _quat_to_mat(np.asarray(cam_quat_xyzw, dtype=np.float64).reshape(4))
    left_m = float(leftward_cm) / 100.0
    up_m = float(upward_cm) / 100.0
    fwd_m = float(forward_cm) / 100.0
    # 相机系：left=-X, up=+Y, forward=-Z
    delta_cam = np.array([-left_m, up_m, -fwd_m], dtype=np.float64)
    return R @ delta_cam


def camera_delta_from_world(cam_quat_xyzw: np.ndarray, delta_world: np.ndarray) -> Dict[str, float]:
    """世界系位移向量 (m) → move_eef 使用的 head 相机系 cm 参数。"""
    from behavior_interface.skills.grasp import _quat_to_mat

    rot = _quat_to_mat(np.asarray(cam_quat_xyzw, dtype=np.float64).reshape(4))
    delta_cam = rot.T @ np.asarray(delta_world, dtype=np.float64).reshape(3)
    return {
        "upward": float(delta_cam[1] * 100.0),
        "forward": float(-delta_cam[2] * 100.0),
        "leftward": float(-delta_cam[0] * 100.0),
    }


def world_from_uv_depth(
    *,
    u: float,
    v: float,
    depth_m: float,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> np.ndarray:
    """Memory 口径反投影：u/v + head-camera forward z-depth(-z_cam) → world xyz."""
    from behavior_interface.skills.grasp import _quat_to_mat

    d = float(depth_m)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(f"depth 必须是正的 head-camera forward z-depth(m)，收到 {depth_m!r}")
    fx = float(fl) / float(ha) * float(w)
    fy = fx
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError(f"head 相机内参无效: fl={fl}, ha={ha}, w={w}, h={h}")
    x_cam = (float(u) - float(w) / 2.0) * d / fx
    y_cam = (float(h) / 2.0 - float(v)) * d / fy
    z_cam = -d
    p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    rot = _quat_to_mat(np.asarray(cam_quat, dtype=np.float64).reshape(4))
    return np.asarray(cam_pos, dtype=np.float64).reshape(3) + rot @ p_cam


def target_from_uv_depth(world, cam_pos: np.ndarray, cam_quat: np.ndarray, *, u: float, v: float, depth: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    head = _head_sensor(world)
    if head is None:
        raise ValueError("未找到 head 相机，无法用 u/v/depth 反解 EEF 目标")
    w, h, fl, ha = _head_camera_meta(head)
    pos = world_from_uv_depth(
        u=float(u),
        v=float(v),
        depth_m=float(depth),
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        w=w,
        h=h,
        fl=fl,
        ha=ha,
    )
    return pos, {
        "u": float(u),
        "v": float(v),
        "depth_m": float(depth),
        "image_width": int(w),
        "image_height": int(h),
        "focal_length": float(fl),
        "horizontal_aperture": float(ha),
    }


def _yield_gripper_only(world, arm: str, grip_cmd, *, steps: int = _GRIPPER_SETTLE_STEPS):
    """保持手臂关节，仅驱动夹爪。"""
    from behavior_interface.skills.arm_reset import _arm_qpos

    hold = _arm_qpos(world, arm).tolist()
    grip_action = _as_gripper_action(grip_cmd)
    if grip_action is None:
        return
    for _ in range(int(steps)):
        yield world.make_action(**{f"arm_{arm}": hold, f"gripper_{arm}": grip_action})


def _yield_arm_hold(
    world,
    arm: str,
    grip_cmd,
    *,
    steps: int = _ARM_HOLD_STEPS,
):
    """把当前关节角重复下发，冻结手臂（及夹爪命令）。"""
    from behavior_interface.skills.arm_reset import _arm_qpos

    hold = _arm_qpos(world, arm).tolist()
    kw: Dict[str, Any] = {f"arm_{arm}": hold}
    grip_action = _as_gripper_action(grip_cmd)
    if grip_action is not None:
        kw[f"gripper_{arm}"] = grip_action
    for _ in range(int(steps)):
        yield world.make_action(**kw)


def _quat_ori_err_deg(q_ref: np.ndarray, q_cur: np.ndarray) -> float:
    from behavior_interface.skills.grasp import _orientation_error_omega, _quat_to_mat

    R_ref = _quat_to_mat(np.asarray(q_ref, dtype=np.float64).reshape(4))
    R_cur = _quat_to_mat(np.asarray(q_cur, dtype=np.float64).reshape(4))
    omega = _orientation_error_omega(R_ref, R_cur)
    return float(np.degrees(np.linalg.norm(omega)))


def kinematic_snap_eef_arm(
    ctx,
    world,
    arm: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
) -> Dict[str, Any]:
    """Report terminal EEF error without teleporting robot joints.

    This helper is kept for API compatibility. Normal execution must converge
    through yielded controller actions so collision detection remains active.
    """
    report: Dict[str, Any] = {
        "snapped": False,
        "disabled": True,
        "reason": "physics_safe_no_joint_teleport",
    }
    if getattr(world, "dry_run", False):
        return report

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
    try:
        eef_now = world.eef_pose(arm=arm)
        pos_now = np.asarray(eef_now["pos"], dtype=np.float64).reshape(3)
        quat_now = np.asarray(eef_now["quat"], dtype=np.float64).reshape(4)
        cur_pos_err = float(np.linalg.norm(pos_now - target_pos))
        cur_ori_err = _quat_ori_err_deg(target_quat, quat_now)
    except Exception as exc:
        report["error"] = f"read eef failed: {type(exc).__name__}: {exc}"
        return report
    report["before_pos_err_mm"] = round(cur_pos_err * 1000.0, 2)
    report["before_ori_err_deg"] = round(cur_ori_err, 3)
    try:
        current_q = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7)
        world.set_arm_pin_qpos(arm, current_q)
        report["held_qpos"] = [round(float(x), 5) for x in current_q.tolist()]
    except Exception:
        pass
    if ctx is not None:
        ctx.log(
            f"[kinematic_snap] {arm} disabled: collision-safe controller result "
            f"pos={report['before_pos_err_mm']:.1f}mm "
            f"ori={report['before_ori_err_deg']:.2f}deg "
            f"tol={float(pos_tol)*1000:.0f}mm/{float(ori_tol_deg):.1f}deg"
        )
    return report


def _live_assisted_grasp_constraint(robot, arm: str) -> bool:
    constraints = getattr(robot, "_ag_obj_constraints", None)
    return isinstance(constraints, dict) and constraints.get(arm) is not None


def _fixed_base_articulated_grasp_link(robot, arm: str, held) -> Optional[str]:
    """Return the constrained movable link for a valid fixed-base articulation grasp."""
    if held is None or not bool(getattr(held, "fixed_base", False)):
        return None
    if not _live_assisted_grasp_constraint(robot, arm):
        return None
    params_map = getattr(robot, "_ag_obj_constraint_params", None)
    params = params_map.get(arm) if isinstance(params_map, dict) else None
    if not isinstance(params, dict) or params.get("joint_type") != "SphericalJoint":
        return None
    link_path = str(params.get("ag_link_prim_path") or "")
    root_link = getattr(held, "root_link", None)
    root_path = str(getattr(root_link, "prim_path", "") or "")
    if not link_path or not root_path or link_path == root_path:
        return None
    return link_path


def _observe_move_eef_assisted_grasp(ctx, world, arm: str) -> Optional[str]:
    """Report only the public grasp proprioception state."""
    try:
        from behavior_interface.skills.eef import _official_grasp_active
    except Exception as exc:
        ctx.log(f"move_eef [ag] helper import failed: {type(exc).__name__}: {exc}")
        return None
    if not _official_grasp_active(world, arm):
        ctx.log("move_eef [ag] public is_grasping=FALSE")
        return None
    ctx.log("move_eef [ag] public is_grasping=TRUE")
    return "official_assisted_grasp"


@register_skill(
    "move_eef",
    description=(
        "head 相机系 EEF 平移：upward/forward/leftward 单位 cm（可正负），"
        "也可同时输入 u/v/depth 作为目标 EEF 位置；"
        "gripper=open|close|keep；姿态四元数不变；夹爪先动再平移。"
    ),
)
def move_eef(
    ctx,
    upward: float = 0.0,
    forward: float = 0.0,
    leftward: float = 0.0,
    u: Optional[float] = None,
    v: Optional[float] = None,
    depth: Optional[float] = None,
    gripper: str = "keep",
    arm: str = "right",
    pos_tol: float = 0.012,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    """相机坐标系下 EEF 增量移动（仅平移，姿态锁定）。"""
    from behavior_interface.skills.eef import (
        _ensure_world_pinned_actions,
        _prepare_legacy_7dof_motion,
    )
    from behavior_interface.skills.grasp import _eef_goto_world, _read_finger_qpos

    world = ctx.world
    _ensure_world_pinned_actions(world)
    arm_eff = str(arm or "right").strip().lower()
    if arm_eff not in ("left", "right"):
        ctx.set_result({"ok": False, "error": f"arm 必须是 left/right，收到 {arm!r}"})
        yield world.hold_action()
        return

    try:
        grip_tgt = _parse_gripper_target(gripper)
    except ValueError as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    uv_depth_supplied = [_has_value(u), _has_value(v), _has_value(depth)]
    use_uv_depth = any(uv_depth_supplied)
    if use_uv_depth and not all(uv_depth_supplied):
        ctx.set_result({
            "ok": False,
            "error": "u, v, depth 必须同时提供；depth 单位 m，语义为 head-camera forward z-depth",
        })
        yield world.hold_action()
        return
    legacy_motion_requested = bool(
        use_uv_depth
        or abs(float(upward)) > 1e-9
        or abs(float(forward)) > 1e-9
        or abs(float(leftward)) > 1e-9
    )
    if legacy_motion_requested:
        _prepare_legacy_7dof_motion(
            world, arm_eff, ctx=ctx, stage_name="move_eef.prepare"
        )

    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系增量"})
        yield world.hold_action()
        return

    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    quat0 = np.asarray(eef0["quat"], dtype=np.float64).reshape(4)

    target_source = "camera_delta_cm"
    uv_depth_target = None
    if use_uv_depth:
        try:
            pos_tgt, uv_depth_target = target_from_uv_depth(
                world,
                cam_pos,
                cam_quat,
                u=float(u),
                v=float(v),
                depth=float(depth),
            )
        except Exception as e:
            ctx.set_result({"ok": False, "error": f"u/v/depth 反解 EEF 目标失败: {e}"})
            yield world.hold_action()
            return
        delta_world = pos_tgt - pos0
        delta_cam_actual = camera_delta_from_world(cam_quat, delta_world)
        upward_eff = float(delta_cam_actual["upward"])
        forward_eff = float(delta_cam_actual["forward"])
        leftward_eff = float(delta_cam_actual["leftward"])
        target_source = "uv_depth"
    else:
        upward_eff = float(upward)
        forward_eff = float(forward)
        leftward_eff = float(leftward)
        delta_world = camera_delta_to_world(
            cam_quat,
            upward_cm=upward_eff,
            forward_cm=forward_eff,
            leftward_cm=leftward_eff,
        )
        pos_tgt = pos0 + delta_world
    move_pos = float(np.linalg.norm(delta_world)) > 1e-6

    grip_changed = False
    grip_cmd_hold = None
    grip_ok = True
    close_report: Optional[Dict[str, Any]] = None
    assisted_grasp_object: Optional[str] = None
    if grip_tgt is not None:
        want_open = grip_tgt > 0
        grip_changed = True
        from behavior_interface.skills.eef import (
            _assisted_grasp_hold_frames,
            _gripper_limit_cmd,
            _slow_close_gripper_hold_wrist,
        )

        grip_cmd_hold = _gripper_limit_cmd(world, arm_eff, open_gripper=want_open)
        settle_steps = (
            _GRIPPER_SETTLE_STEPS if want_open else _GRIPPER_CLOSE_SETTLE_STEPS
        )
        ctx.log(
            f"move_eef [{arm_eff}] 显式夹爪命令 → {'open' if want_open else 'close'} "
            f"cmd={_as_gripper_action(grip_cmd_hold)} steps={settle_steps} "
            f"(finger≈{_read_finger_qpos(world, arm_eff)})"
        )
        if want_open:
            yield from _yield_gripper_only(world, arm_eff, grip_cmd_hold, steps=settle_steps)
        else:
            wrist = world.eef_pose(arm=arm_eff)
            wrist_pos = np.asarray(wrist["pos"], dtype=np.float64).reshape(3)
            wrist_quat = np.asarray(wrist["quat"], dtype=np.float64).reshape(4)
            ag_hold_frames = _assisted_grasp_hold_frames(world)
            close_report = yield from _slow_close_gripper_hold_wrist(
                world,
                arm_eff,
                wrist_pos,
                wrist_quat,
                q_closed_cmd=grip_cmd_hold,
                ctx=ctx,
                n_ramp=8,
                n_hold=ag_hold_frames,
            )
            if isinstance(close_report, dict) and close_report.get("mode") == "effort":
                # The world effort pin supplies the negative assisted-grasp
                # keepalive during any following EEF motion.
                grip_cmd_hold = None
            assisted_grasp_object = _observe_move_eef_assisted_grasp(
                ctx, world, arm_eff
            )
            grip_ok = bool(assisted_grasp_object)
            if grip_ok:
                latch_keepalive = getattr(
                    world,
                    "latch_gripper_close_keepalive",
                    None,
                )
                if callable(latch_keepalive):
                    carry_effort = (
                        close_report.get("carry_effort_n")
                        if isinstance(close_report, dict)
                        else None
                    )
                    latch_keepalive(arm_eff, effort=carry_effort)
            if grip_ok and assisted_grasp_object:
                ctx.log(
                    f"move_eef [ag] close 成功 held={assisted_grasp_object} "
                    f"hold_frames={ag_hold_frames} build={MOVE_EEF_BUILD}"
                )
            else:
                ctx.log(
                    f"move_eef [ag] close 失败，未建立官方连接 build={MOVE_EEF_BUILD} "
                    f"hold_frames={ag_hold_frames} finger={_read_finger_qpos(world, arm_eff)}"
                )
        ctx.log(
            f"move_eef [{arm_eff}] 夹爪命令完成 finger={_read_finger_qpos(world, arm_eff)}"
        )
    else:
        # 平移阶段保持当前 finger qpos，避免 closed 状态被误发成 [-1] 标量。
        grip_cmd_hold = _current_gripper_hold_cmd(world, arm_eff)

    snap_report: Optional[Dict[str, Any]] = None
    if move_pos and grip_ok:
        ctx.log(
            f"move_eef [{arm_eff}] source={target_source} cam Δ(cm) "
            f"up={upward_eff:+.1f} fwd={forward_eff:+.1f} "
            f"left={leftward_eff:+.1f} → world Δ(m)="
            f"({delta_world[0]:+.4f},{delta_world[1]:+.4f},{delta_world[2]:+.4f}) "
            f"pos {pos0.round(4).tolist()} → {pos_tgt.round(4).tolist()} "
            f"(quat 不变)"
        )
        # 完整 6D IK：位置追目标，同时把起始四元数作为硬姿态目标。
        err = yield from _eef_goto_world(
            world, arm_eff, pos_tgt,
            target_world_quat=quat0,
            hold_world_quat=None,
            max_steps=int(max_steps),
            pos_tol=float(pos_tol),
            ori_tol=np.radians(_ORI_DRIFT_OK_DEG),
            gripper_cmd=grip_cmd_hold,
            ctx=ctx,
            stage_name="move_eef",
            max_dx_per_step=0.06,
            max_dw_per_step=0.06,
            ori_weight=1.0,
            adaptive_ori=False,
        )
        # Record terminal error without bypassing collision-checked control.
        snap_report = kinematic_snap_eef_arm(
            ctx, world, arm_eff, pos_tgt, quat0,
            pos_tol=float(pos_tol), ori_tol_deg=_ORI_DRIFT_OK_DEG,
        )
        eef1 = world.eef_pose(arm=arm_eff)
        pos1 = np.asarray(eef1["pos"], dtype=np.float64).reshape(3)
        quat1 = np.asarray(eef1["quat"], dtype=np.float64).reshape(4)
        pos_err = float(np.linalg.norm(pos1 - pos_tgt))
        ori_drift_deg = _quat_ori_err_deg(quat0, quat1)
        pos_ok = pos_err < max(float(pos_tol) * 2.5, 0.02)
        ori_ok = ori_drift_deg <= _ORI_DRIFT_OK_DEG
        ok = pos_ok and ori_ok and grip_ok
        if not ok:
            ctx.log(
                f"move_eef WARN 未完全到位 pos_err={pos_err*1000:.1f}mm "
                f"ori_drift={ori_drift_deg:.1f}° pos_ok={pos_ok} ori_ok={ori_ok}"
            )
    elif move_pos:
        err = float("inf")
        pos1 = pos0.copy()
        pos_err = float(np.linalg.norm(pos0 - pos_tgt))
        pos_ok = False
        ori_ok = True
        ok = False
        ctx.log(
            f"move_eef [{arm_eff}] ABORT 平移：close 未建立官方 assisted-grasp 连接"
        )
    else:
        err = 0.0
        pos1 = pos0.copy()
        pos_err = 0.0
        pos_ok = True
        ori_ok = True
        ok = bool(grip_ok)
        ctx.log(
            f"move_eef [{arm_eff}] 仅夹爪变化，无平移 "
            f"source={target_source} cam Δ(cm) "
            f"up={upward_eff:+.1f} fwd={forward_eff:+.1f} left={leftward_eff:+.1f}"
        )

    quat_after = quat0
    ori_drift_out = 0.0
    if move_pos:
        eef_fin = world.eef_pose(arm=arm_eff)
        quat_after = np.asarray(eef_fin["quat"], dtype=np.float64).reshape(4)
        ori_drift_out = _quat_ori_err_deg(quat0, quat_after)

    out: Dict[str, Any] = {
        "ok": bool(ok),
        "build": MOVE_EEF_BUILD,
        "arm": arm_eff,
        "target_source": target_source,
        "uv_depth_target": uv_depth_target,
        "delta_cam_cm": {
            "upward": upward_eff,
            "forward": forward_eff,
            "leftward": leftward_eff,
        },
        "delta_world_m": delta_world.round(6).tolist(),
        "eef_before": pos0.round(5).tolist(),
        "eef_after": np.asarray(pos1).reshape(3).round(5).tolist(),
        "eef_target": pos_tgt.round(5).tolist(),
        "pos_err_mm": round(float(pos_err) * 1000.0, 2),
        "pos_ok": bool(pos_ok),
        "quat": quat0.round(6).tolist(),
        "quat_after": quat_after.round(6).tolist(),
        "ori_drift_deg": round(float(ori_drift_out), 2),
        "ori_drift_limit_deg": float(_ORI_DRIFT_OK_DEG),
        "ori_ok": bool(ori_ok),
        "gripper_requested": gripper,
        "gripper_changed": grip_changed,
        "gripper_ok": bool(grip_ok),
        "gripper_close": close_report,
        "gripper_hold_cmd": _as_gripper_action(grip_cmd_hold),
        "assisted_grasp_object": assisted_grasp_object,
        "gripper_keepalive": (
            world.gripper_keepalive_status().get(arm_eff)
            if callable(getattr(world, "gripper_keepalive_status", None))
            else None
        ),
        "finger_qpos": _read_finger_qpos(world, arm_eff),
        "cam_pos": cam_pos.round(5).tolist(),
        "kinematic_snap": snap_report,
    }
    if not grip_ok:
        out["error"] = "close gripper 未建立官方 assisted-grasp 连接"
    ctx.set_result(out)
    # 显式锁定关节若干帧，避免 IK 中断后控制器继续追最后一帧目标
    yield from _yield_arm_hold(world, arm_eff, grip_cmd_hold)
