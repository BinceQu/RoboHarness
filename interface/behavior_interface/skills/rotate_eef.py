"""rotate_eef：在 head 相机坐标系下旋转 EEF（位置不变）。

相机系（与 move_eef / head 针孔一致）：
  +X = 画面右，+Y = 画面上，+Z = 朝向观察者（出屏）
  forward  = 视线进入场景 → 相机 -Z
  upward   = 相机 +Y
  leftward = 相机 -X

旋转正方向（三轴统一）：
  正值 = 右手定则 = 拇指指向该轴**正方向**时，四指弯曲方向；
  等价于沿该轴正方向观察时的逆时针（CCW）。

  rotate_forward_deg：绕 forward 轴；沿视线看入场景时，画面 CCW 滚转。
  rotate_upward_deg：绕 upward 轴；从机器人头顶向下看时 CCW（末端朝画面右侧偏）。
  rotate_leftward_deg：绕 leftward 轴；沿画面左向观察时 CCW。

复合顺序（外旋、固定于命令时刻的相机系）：forward → upward → leftward。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_eef import (
    MOVE_EEF_BUILD,
    _gripper_is_open,
    _head_cam_pose,
    _quat_ori_err_deg,
    _yield_arm_hold,
    kinematic_snap_eef_arm,
)

ROTATE_EEF_BUILD = "v2_cam_axes_pos_hold_kinematic_snap"


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle_rad) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / n
    half = float(angle_rad) * 0.5
    s = math.sin(half)
    c = math.cos(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)


def camera_frame_delta_quat(
    cam_quat_xyzw: np.ndarray,
    *,
    rotate_forward_deg: float = 0.0,
    rotate_upward_deg: float = 0.0,
    rotate_leftward_deg: float = 0.0,
) -> np.ndarray:
    """命令时刻相机系下的三轴外旋 → 世界系增量四元数 (xyzw)。"""
    from behavior_interface.skills.grasp import _quat_mul, _quat_to_mat

    if (
        abs(rotate_forward_deg) < 1e-9
        and abs(rotate_upward_deg) < 1e-9
        and abs(rotate_leftward_deg) < 1e-9
    ):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    R_cam = _quat_to_mat(np.asarray(cam_quat_xyzw, dtype=np.float64).reshape(4))
    fwd_w = R_cam @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up_w = R_cam @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    left_w = R_cam @ np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    q_fwd = _axis_angle_to_quat(fwd_w, math.radians(float(rotate_forward_deg)))
    q_up = _axis_angle_to_quat(up_w, math.radians(float(rotate_upward_deg)))
    q_left = _axis_angle_to_quat(left_w, math.radians(float(rotate_leftward_deg)))
    return _quat_mul(q_left, _quat_mul(q_up, q_fwd))


@register_skill(
    "rotate_eef",
    description=(
        "head 相机系 EEF 旋转（deg）：rotate_forward/upward/leftward_deg；"
        "位置不变；正方向=右手定则/CCW（见模块文档）。"
    ),
)
def rotate_eef(
    ctx,
    rotate_forward_deg: float = 0.0,
    rotate_upward_deg: float = 0.0,
    rotate_leftward_deg: float = 0.0,
    arm: str = "right",
    ori_tol_deg: float = 3.0,
    pos_tol: float = 0.015,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    """相机坐标系下 EEF 增量旋转（位置锁定）。"""
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion
    from behavior_interface.skills.grasp import _eef_goto_world, _quat_mul, _read_finger_qpos

    world = ctx.world
    arm_eff = str(arm or "right").strip().lower()
    if arm_eff not in ("left", "right"):
        ctx.set_result({"ok": False, "error": f"arm 必须是 left/right，收到 {arm!r}"})
        yield world.hold_action()
        return

    rf = float(rotate_forward_deg)
    ru = float(rotate_upward_deg)
    rl = float(rotate_leftward_deg)
    if abs(rf) >= 1e-9 or abs(ru) >= 1e-9 or abs(rl) >= 1e-9:
        _prepare_legacy_7dof_motion(
            world, arm_eff, ctx=ctx, stage_name="rotate_eef.prepare"
        )

    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系旋转"})
        yield world.hold_action()
        return

    if abs(rf) < 1e-9 and abs(ru) < 1e-9 and abs(rl) < 1e-9:
        ctx.set_result({
            "ok": True,
            "build": ROTATE_EEF_BUILD,
            "arm": arm_eff,
            "note": "三轴旋转角均为 0，无动作",
        })
        yield world.hold_action()
        return

    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    quat0 = np.asarray(eef0["quat"], dtype=np.float64).reshape(4)

    q_delta = camera_frame_delta_quat(
        cam_quat,
        rotate_forward_deg=rf,
        rotate_upward_deg=ru,
        rotate_leftward_deg=rl,
    )
    quat_tgt = _quat_mul(q_delta, quat0)
    nq = float(np.linalg.norm(quat_tgt))
    if nq > 1e-9:
        quat_tgt = quat_tgt / nq

    grip_cmd_hold = 1.0 if _gripper_is_open(world, arm_eff) else -1.0

    ctx.log(
        f"rotate_eef [{arm_eff}] cam Δ(deg) fwd={rf:+.1f} up={ru:+.1f} left={rl:+.1f} "
        f"(+CCW/右手定则) pos 锁定 {pos0.round(4).tolist()}"
    )

    yield from _eef_goto_world(
        world, arm_eff, pos0,
        target_world_quat=quat_tgt,
        max_steps=int(max_steps),
        pos_tol=float(pos_tol),
        ori_tol=math.radians(float(ori_tol_deg)),
        gripper_cmd=grip_cmd_hold,
        ctx=ctx,
        stage_name="rotate_eef",
        max_dx_per_step=0.03,
        max_dw_per_step=0.12,
        ori_weight=1.0,
        adaptive_ori=False,
    )

    # 末态运动学吸附：把手臂精确钉到目标 EEF 姿态(quat_tgt)、位置锁 pos0
    snap_report = kinematic_snap_eef_arm(
        ctx, world, arm_eff, pos0, quat_tgt,
        pos_tol=float(pos_tol), ori_tol_deg=float(ori_tol_deg),
    )

    eef1 = world.eef_pose(arm=arm_eff)
    pos1 = np.asarray(eef1["pos"], dtype=np.float64).reshape(3)
    quat1 = np.asarray(eef1["quat"], dtype=np.float64).reshape(4)
    pos_err = float(np.linalg.norm(pos1 - pos0))
    ori_err = _quat_ori_err_deg(quat_tgt, quat1)
    ok = ori_err < max(float(ori_tol_deg) * 2.0, 6.0) and pos_err < max(float(pos_tol) * 3.0, 0.04)
    if not ok:
        ctx.log(
            f"rotate_eef WARN 未完全到位 pos_err={pos_err*1000:.1f}mm "
            f"ori_err={ori_err:.1f}° (tgt)"
        )

    ctx.set_result({
        "ok": bool(ok),
        "build": ROTATE_EEF_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "arm": arm_eff,
        "rotate_cam_deg": {
            "rotate_forward_deg": rf,
            "rotate_upward_deg": ru,
            "rotate_leftward_deg": rl,
        },
        "convention": (
            "positive = right-hand rule = CCW when looking along +axis "
            "(forward=into scene, upward=image up, leftward=image left)"
        ),
        "eef_pos_before": pos0.round(5).tolist(),
        "eef_pos_after": pos1.round(5).tolist(),
        "pos_drift_mm": round(pos_err * 1000.0, 2),
        "quat_before": quat0.round(6).tolist(),
        "quat_target": quat_tgt.round(6).tolist(),
        "quat_after": quat1.round(6).tolist(),
        "ori_err_deg": round(float(ori_err), 2),
        "finger_qpos": _read_finger_qpos(world, arm_eff),
        "cam_pos": cam_pos.round(5).tolist(),
        "kinematic_snap": snap_report,
    })
    yield from _yield_arm_hold(world, arm_eff, grip_cmd_hold)
