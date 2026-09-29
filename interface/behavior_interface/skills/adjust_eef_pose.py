"""Head-camera-frame EEF pose adjustment for the left and right arms."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.adjust_plan_pose import (
    _apply_local_delta_quat,
    _local_gripper_rpy_delta_quat,
)
from behavior_interface.skills.move_eef import (
    _current_gripper_hold_cmd,
    _head_cam_pose,
    _quat_ori_err_deg,
    _yield_arm_hold,
    camera_delta_to_world,
)

ADJUST_EEF_POSE_BUILD = "head_camera_fixed_lookahead_fast_settle_v17"


def _adjust_eef_pose_in_head_frame(
    ctx,
    *,
    arm: str,
    tool_name: str,
    forward: float = 0.0,
    upward: float = 0.0,
    leftward: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    from behavior_interface.skills.eef import (
        _ensure_world_pinned_actions,
        _prepare_legacy_7dof_motion,
    )
    from behavior_interface.skills.adjust_eef_pose_in_wrist_frame import (
        ADJUST_EEF_WRIST_FRAME_BUILD,
        _translate_eef_adaptive,
    )
    from behavior_interface.skills.grasp import _read_finger_qpos

    world = ctx.world
    _ensure_world_pinned_actions(world)
    arm_eff = str(arm).strip().lower()
    values = np.asarray(
        [forward, upward, leftward, roll, pitch, yaw],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        ctx.set_result({
            "ok": False,
            "tool": tool_name,
            "error": "all deltas must be finite",
        })
        yield world.hold_action()
        return
    move_pos_requested = any(
        abs(float(x)) > 1e-9 for x in (forward, upward, leftward)
    )
    move_ori_requested = any(
        abs(float(x)) > 1e-9 for x in (roll, pitch, yaw)
    )
    if move_pos_requested or move_ori_requested:
        _prepare_legacy_7dof_motion(
            world, arm_eff, ctx=ctx, stage_name=f"{tool_name}.prepare"
        )

    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({
            "ok": False,
            "tool": tool_name,
            "error": "未找到 head 相机，无法换算 EEF 增量",
        })
        yield world.hold_action()
        return

    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    quat0 = np.asarray(eef0["quat"], dtype=np.float64).reshape(4)

    delta_world = camera_delta_to_world(
        cam_quat,
        upward_cm=float(upward) * 100.0,
        forward_cm=float(forward) * 100.0,
        leftward_cm=float(leftward) * 100.0,
    )
    pos_tgt = pos0 + delta_world
    q_delta_local = _local_gripper_rpy_delta_quat(
        roll_deg=float(roll),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
    )
    quat_tgt = _apply_local_delta_quat(quat0, q_delta_local)
    grip_cmd_hold = _current_gripper_hold_cmd(world, arm_eff)

    move_pos = float(np.linalg.norm(delta_world)) > 1e-6
    move_ori = any(abs(float(x)) > 1e-9 for x in (roll, pitch, yaw))
    if not move_pos and not move_ori:
        ctx.set_result({
            "ok": True,
            "tool": tool_name,
            "build": ADJUST_EEF_POSE_BUILD,
            "arm": arm_eff,
            "note": "平移和旋转均为 0，无动作",
            "finger_qpos": _read_finger_qpos(world, arm_eff),
        })
        yield world.hold_action()
        return

    ctx.log(
        f"{tool_name} [{arm_eff}] build={ADJUST_EEF_POSE_BUILD} "
        f"adaptive_ref={ADJUST_EEF_WRIST_FRAME_BUILD} "
        f"head_cam_delta(m) fwd={float(forward):+.3f} "
        f"up={float(upward):+.3f} left={float(leftward):+.3f}; "
        f"local rpy(deg) roll={float(roll):+.1f} pitch={float(pitch):+.1f} yaw={float(yaw):+.1f}"
    )

    pose_report = yield from _translate_eef_adaptive(
        ctx,
        world,
        arm_eff,
        pos_tgt,
        quat_tgt,
        gripper_cmd=grip_cmd_hold,
        pos_tol=float(pos_tol),
        max_steps=int(max_steps),
        timeout_s=float(timeout_s),
        stage_name=f"{tool_name}.adaptive_pose",
        require_orientation=move_ori,
        orientation_tol_deg=float(ori_tol_deg),
    )
    eef1 = world.eef_pose(arm=arm_eff)
    pos1 = np.asarray(eef1["pos"], dtype=np.float64).reshape(3)
    quat1 = np.asarray(eef1["quat"], dtype=np.float64).reshape(4)
    pos_err = float(np.linalg.norm(pos1 - pos_tgt))
    ori_err = _quat_ori_err_deg(quat_tgt, quat1)
    pos_reached = bool(
        not move_pos or pos_err <= max(0.001, float(pos_tol))
    )
    # 纯平移请求的姿态目标就是起始姿态，漂移同样必须落在容差内。旧写法在
    # move_ori=False 时无条件放行，末端翻转 68° 也会被报成 target_reached=True。
    ori_tolerance_deg = max(float(ori_tol_deg) * 2.0, 6.0)
    ori_reached = bool(ori_err <= ori_tolerance_deg)
    overall_ok = bool(
        pose_report.get("ok", False)
        and pos_reached
        and ori_reached
    )
    failure_reason = None
    if not overall_ok:
        if not ori_reached and pos_reached:
            failure_reason = (
                f"orientation drifted {ori_err:.1f}deg "
                f"(limit {ori_tolerance_deg:.1f}deg) while position reached; "
                "target pose is likely unreachable without changing orientation"
            )
        elif not pos_reached and ori_reached:
            failure_reason = (
                f"position error {pos_err * 1000.0:.1f}mm exceeds tolerance"
            )
        elif not pos_reached and not ori_reached:
            failure_reason = (
                f"position error {pos_err * 1000.0:.1f}mm and orientation "
                f"drift {ori_err:.1f}deg both exceed tolerance"
            )
        else:
            failure_reason = str(
                pose_report.get("reason", "pose controller did not complete")
            )
    out: Dict[str, Any] = {
        "ok": overall_ok,
        "tool": tool_name,
        "build": ADJUST_EEF_POSE_BUILD,
        "adaptive_controller_build_ref": ADJUST_EEF_WRIST_FRAME_BUILD,
        "arm": arm_eff,
        "delta_head_camera_m": {
            "forward": float(forward),
            "upward": float(upward),
            "leftward": float(leftward),
        },
        "rpy_deg": {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
        "rpy_convention": {
            "translation": "forward/upward/leftward are head-camera-frame translations",
            "rotation": "roll/pitch/yaw are local gripper axes, same convention as adjust_plan_pose",
            "positive_pitch": "positive pitch is nose-up / fingertips-up",
        },
        "delta_world_m": delta_world.round(6).tolist(),
        "eef_before": pos0.round(5).tolist(),
        "eef_target": pos_tgt.round(5).tolist(),
        "eef_after": pos1.round(5).tolist(),
        "quat_before": quat0.round(6).tolist(),
        "quat_target": quat_tgt.round(6).tolist(),
        "quat_after": quat1.round(6).tolist(),
        "pos_err_mm": round(pos_err * 1000.0, 2),
        "ori_err_deg": round(float(ori_err), 2),
        "pos_ok": pos_reached,
        "ori_ok": ori_reached,
        "ori_tol_applied_deg": round(float(ori_tolerance_deg), 2),
        "target_reached": bool(pos_reached and ori_reached),
        "failure_reason": failure_reason,
        "pose_control": pose_report,
        "j8_participates": False,
        "finger_qpos": _read_finger_qpos(world, arm_eff),
        "kinematic_snap": {
            "snapped": False,
            "disabled": True,
            "reason": "head-frame adaptive pose control never uses kinematic snap",
        },
        "error": None if overall_ok else "EEF tangent command 未完整执行",
    }
    ctx.set_result(out)


@register_skill(
    "adjust_left_eef_pose_in_head_frame",
    description=(
        "调整左 EEF：forward/upward/leftward 为 head camera 坐标系米制平移；"
        "roll/pitch/yaw 为夹爪局部轴角度，与 adjust_plan_pose 一致。"
    ),
)
def adjust_left_eef_pose_in_head_frame(
    ctx,
    forward: float = 0.0,
    upward: float = 0.0,
    leftward: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    yield from _adjust_eef_pose_in_head_frame(
        ctx,
        arm="left",
        tool_name="adjust_left_eef_pose_in_head_frame",
        forward=forward,
        upward=upward,
        leftward=leftward,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
    )


@register_skill(
    "adjust_right_eef_pose_in_head_frame",
    description=(
        "调整右 EEF：forward/upward/leftward 为 head camera 坐标系米制平移；"
        "roll/pitch/yaw 为夹爪局部轴角度，与 adjust_plan_pose 一致。"
    ),
)
def adjust_right_eef_pose_in_head_frame(
    ctx,
    forward: float = 0.0,
    upward: float = 0.0,
    leftward: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    yield from _adjust_eef_pose_in_head_frame(
        ctx,
        arm="right",
        tool_name="adjust_right_eef_pose_in_head_frame",
        forward=forward,
        upward=upward,
        leftward=leftward,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
    )
