"""plan_move_eef：预览 move_eef 的目标 EEF 位置，不实际执行。

与 move_eef 使用同一套 head 相机系增量换算：
  forward  = 相机 -Z（进场景）
  upward   = 相机 +Y
  leftward = 相机 -X

也可以输入 u/v/depth；公开 u/v 使用 Qwen3-VL 0..1000 相对坐标，
注册表边界转换为 head 图像原生像素后，再与 head-camera forward z-depth
反解为 EEF 目标三维点。depth 的语义与 Memory 一致：depth = -z_cam，单位 m。

输出是一张当前 head RGB 上的红色夹爪叠影，红爪 pose = 预估 EEF pose。
"""

from __future__ import annotations

import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_eef import (
    MOVE_EEF_BUILD,
    camera_delta_from_world,
    _finger_mean_qpos,
    _gripper_is_open,
    _head_cam_pose,
    _parse_gripper_target,
    camera_delta_to_world,
    target_from_uv_depth,
)

PLAN_MOVE_EEF_BUILD = "v3_move_eef_6d_line_preflight_overlay"
_STOP_RESIDUAL_FRAC = 0.85


def _head_sensor(world):
    from behavior_interface.head_capture import get_head_sensor

    return get_head_sensor(world)


def _head_camera_meta(sensor) -> Tuple[int, int, float, float]:
    from behavior_interface.head_capture import head_camera_size

    w, h = head_camera_size(sensor)
    fl = float(getattr(sensor, "focal_length", 17.0))
    ha = float(getattr(sensor, "horizontal_aperture", 40.0))
    return int(w), int(h), fl, ha


def _save_head_rgb_depth_from_obs(head, out_rgb_path: str, out_depth_path: str) -> Tuple[bool, Optional[np.ndarray]]:
    from behavior_interface.skills.vlm_lawn_dual import _save_rgb, _save_depth

    try:
        obs, _info = head.get_obs()
    except Exception:
        return False, None
    ok = bool(_save_rgb(obs, out_rgb_path))
    depth = _save_depth(obs, out_depth_path, out_depth_path + ".png")
    return ok, depth


def _requested_gripper_state(world, arm: str, gripper: str) -> Tuple[Optional[float], str, bool]:
    """返回 (cmd, predicted_state, changed)，cmd 为 move_eef 会下发的 gripper 命令。"""
    grip_tgt = _parse_gripper_target(gripper)
    is_open = _gripper_is_open(world, arm)
    if grip_tgt is None:
        return (1.0 if is_open else -1.0), ("open" if is_open else "close"), False
    want_open = grip_tgt > 0.0
    return float(grip_tgt), ("open" if want_open else "close"), bool(want_open != is_open)


def _has_value(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _plan_move_eef_execution_preflight(
    world,
    arm: str,
    target_pos,
    target_quat,
    *,
    ctx,
) -> Dict[str, Any]:
    """Run the same strict move-only Cartesian-line IK gate used by exec."""
    from behavior_interface.skills.eef import _eef_plan_6d_joint_anchors

    anchors, meta = _eef_plan_6d_joint_anchors(
        world,
        arm,
        target_pos,
        target_quat,
        ctx=ctx,
        stage_name="plan_move_eef_line_preflight",
        pos_tol=0.010,
        ori_tol_deg=3.0,
        waypoint_pos_step_m=0.035,
        waypoint_ori_step_deg=12.0,
        max_joint_gap_rad=0.55,
        mid_pos_tol=0.012,
        mid_ori_tol_deg=6.0,
        intermediate_pos_only=False,
    )
    out: Dict[str, Any] = {
        "ok": bool(anchors),
        "anchor_count": int(len(anchors)),
        "line_tol_m": 0.020,
        "pos_tol_m": 0.010,
        "ori_tol_deg": 3.0,
        "meta": _json_ready(meta),
    }
    if anchors:
        final = anchors[-1]
        out["final_q"] = _json_ready(final.get("q"))
        out["final_pos_err_m"] = float(final.get("pos_err", 0.0))
        out["final_ori_err_deg"] = float(final.get("ori_err", 0.0))
    return out


def _drop_current_gripper_from_depth(
    scene_depth: Optional[np.ndarray],
    *,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Optional[np.ndarray]:
    """当前底图里有移动前夹爪；遮挡深度中挖掉它，避免挡住目标红爪预览。"""
    if scene_depth is None:
        return None
    try:
        import cv2
        from behavior_interface.skills.viz_gripper_overlay import rasterize_gripper_buffers

        buf = rasterize_gripper_buffers(
            eef_pos, eef_quat, cam_pos, cam_quat, w, h, fl, ha,
            base_color=(255, 255, 255),
        )
        if buf is None:
            return scene_depth
        _color, mask, _depth = buf
        sd = np.asarray(scene_depth, dtype=np.float32).copy()
        if sd.shape != mask.shape:
            return scene_depth
        kernel = np.ones((9, 9), dtype=np.uint8)
        m = cv2.dilate(mask, kernel, iterations=1) > 0
        sd[m] = np.inf
        return sd
    except Exception:
        return scene_depth


@register_skill(
    "plan_move_eef",
    description=(
        "规划并预览 move_eef：按 head 相机系 upward/forward/leftward(cm) "
        "估计目标 EEF，先验证同分支 6D 直线 IK，再在当前 head 图上叠加红色夹爪；"
        "不实际移动机器人。"
    ),
)
def plan_move_eef(
    ctx,
    session_id: str,
    upward: float = 0.0,
    forward: float = 0.0,
    leftward: float = 0.0,
    u: Optional[float] = None,
    v: Optional[float] = None,
    depth: Optional[float] = None,
    gripper: str = "keep",
    arm: str = "right",
    pos_tol: float = 0.012,
):
    """预览 move_eef 的目标 EEF pose，并返回红色夹爪叠影 head 视图。"""
    from behavior_interface import agent_runs
    from behavior_interface.head_capture import reset_head_sensor_to_mount
    from behavior_interface.skills.viz_gripper_overlay import render_gripper_overlay

    world = ctx.world
    arm_eff = str(arm or "right").strip().lower()
    if arm_eff not in ("left", "right"):
        ctx.set_result({"ok": False, "error": f"arm 必须是 left/right，收到 {arm!r}"})
        yield world.hold_action()
        return

    try:
        grip_cmd, grip_pred, grip_changed = _requested_gripper_state(world, arm_eff, gripper)
    except ValueError as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    head = _head_sensor(world)
    if head is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法生成预览"})
        yield world.hold_action()
        return

    # 与 capture 一致：只在 head sensor 脱节时恢复挂载。
    if reset_head_sensor_to_mount(world, ctx=ctx):
        yield world.hold_action()

    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系增量"})
        yield world.hold_action()
        return

    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    quat0 = np.asarray(eef0["quat"], dtype=np.float64).reshape(4)
    w_img, h_img, fl_m, ha_m = _head_camera_meta(head)

    uv_depth_supplied = [_has_value(u), _has_value(v), _has_value(depth)]
    use_uv_depth = any(uv_depth_supplied)
    if use_uv_depth and not all(uv_depth_supplied):
        ctx.set_result({
            "ok": False,
            "error": "u, v, depth 必须同时提供；depth 单位 m，语义为 head-camera forward z-depth",
        })
        yield world.hold_action()
        return

    target_source = "camera_delta_cm"
    uv_depth_target = None
    if use_uv_depth:
        try:
            u_f = float(u)
            v_f = float(v)
            depth_f = float(depth)
            pos_ideal, uv_depth_target = target_from_uv_depth(
                world,
                cam_pos,
                cam_quat,
                u=u_f,
                v=v_f,
                depth=depth_f,
            )
        except Exception as e:
            ctx.set_result({"ok": False, "error": f"u/v/depth 反解 EEF 目标失败: {e}"})
            yield world.hold_action()
            return
        delta_world = pos_ideal - pos0
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
        pos_ideal = pos0 + delta_world

    move_err0 = float(np.linalg.norm(delta_world))
    execution_preflight = _plan_move_eef_execution_preflight(
        world,
        arm_eff,
        pos_ideal,
        quat0,
        ctx=ctx,
    )
    if not execution_preflight.get("ok"):
        preflight_meta = execution_preflight.get("meta") or {}
        dropped = preflight_meta.get("dropped") or []
        last_drop = dropped[-1] if dropped else {}
        reason = preflight_meta.get("error") or last_drop.get("reason") or "ik_failed"
        ctx.set_result({
            "ok": False,
            "tool": "plan_move_eef",
            "error": (
                "目标 EEF 位姿没有可执行的同分支 6D 直线路径："
                f"{reason}；请缩短单次位移、调整目标点/手臂，"
                "或使用允许重新选择夹爪姿态的 grasp-point planner"
            ),
            "arm": arm_eff,
            "target_source": target_source,
            "uv_depth_target": uv_depth_target,
            "delta_world_m": delta_world.round(6).tolist(),
            "eef_before": pos0.round(6).tolist(),
            "target_eef_pose": {
                "pos": pos_ideal.round(6).tolist(),
                "quat": quat0.round(6).tolist(),
            },
            "execution_preflight": execution_preflight,
        })
        yield world.hold_action()
        return

    # move_eef/_eef_goto_world 会在 step0 先判断 err < pos_tol；小位移不会实际执行。
    converged_at_start = move_err0 < float(pos_tol)
    if converged_at_start:
        pos_preview = pos0.copy()
        preview_kind = "start_converged"
        stop_residual_m = move_err0
        preview_travel_m = 0.0
    else:
        # move_eef 不会追到数学目标，而是在 pos_err < pos_tol 时停下。
        # 预览用一个保守的停止残差估计，让红爪更接近执行后 head capture 中的位置。
        stop_residual_m = min(move_err0, max(0.0, float(pos_tol) * _STOP_RESIDUAL_FRAC))
        preview_travel_m = max(0.0, move_err0 - stop_residual_m)
        pos_preview = pos0 + delta_world * (preview_travel_m / max(move_err0, 1e-9))
        preview_kind = "estimated_stop_pose"

    agent_runs.ensure_session(session_id)
    plan_id = agent_runs.next_plan_id(session_id)
    base_rgb = agent_runs.plan_path(session_id, f"{plan_id}_plan_move_eef_base", ".png")
    base_depth = agent_runs.plan_path(session_id, f"{plan_id}_plan_move_eef_depth", ".npy")
    preview_png = agent_runs.plan_path(session_id, plan_id, ".png")

    # 稳定一两帧，但不下发任何手臂/夹爪动作。
    for _ in range(2):
        yield world.hold_action()
    try:
        import omnigibson as og

        from behavior_interface.skills.capture import _ensure_modalities

        if _ensure_modalities(head, ["depth_linear"]):
            yield world.hold_action()
            for _ in range(2):
                og.sim.render()
        for _ in range(2):
            og.sim.render()
    except Exception:
        pass

    rgb_ok, scene_depth = _save_head_rgb_depth_from_obs(head, base_rgb, base_depth)
    if not rgb_ok:
        ctx.set_result({"ok": False, "error": "head RGB 渲染失败，无法生成预览"})
        yield world.hold_action()
        return
    shutil.copy2(base_rgb, preview_png)

    scene_depth_for_overlay = _drop_current_gripper_from_depth(
        scene_depth,
        eef_pos=pos0,
        eef_quat=quat0,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        w=w_img,
        h=h_img,
        fl=fl_m,
        ha=ha_m,
    )
    overlay_ok = render_gripper_overlay(
        preview_png,
        eef_pos=pos_preview,
        eef_quat=quat0,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        w=w_img,
        h=h_img,
        fl=fl_m,
        ha=ha_m,
        alpha=0.78,
        base_color=(150, 18, 18),
        scene_depth=scene_depth_for_overlay,
    )
    if not overlay_ok:
        ctx.log("plan_move_eef WARN 红色夹爪叠影失败，返回未叠加底图")

    try:
        from behavior_interface.skills.grasp import _quat_to_mat

        approach = np.asarray(_quat_to_mat(np.asarray(quat0, dtype=np.float64).reshape(4))[:, 2], dtype=np.float64)
        approach = approach / (float(np.linalg.norm(approach)) + 1e-9)
    except Exception:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    record: Dict[str, Any] = {
        "plan_id": plan_id,
        "session_id": session_id,
        "skill": "plan_move_eef",
        "mode": "plan_move_eef",
        "arm": arm_eff,
        "build": PLAN_MOVE_EEF_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "target_source": target_source,
        "uv_depth_target": uv_depth_target,
        "delta_cam_cm": {
            "upward": upward_eff,
            "forward": forward_eff,
            "leftward": leftward_eff,
        },
        "delta_world_m": delta_world.round(6).tolist(),
        "eef_before": pos0.round(6).tolist(),
        "eef_pose": {
            "pos": pos_preview.round(6).tolist(),
            "quat": quat0.round(6).tolist(),
        },
        "ideal_eef_pose": {
            "pos": pos_ideal.round(6).tolist(),
            "quat": quat0.round(6).tolist(),
        },
        "target_eef_pose": {
            "pos": pos_ideal.round(6).tolist(),
            "quat": quat0.round(6).tolist(),
        },
        "move": {
            "tool": "move_eef",
            "arm": arm_eff,
            "upward": upward_eff,
            "forward": forward_eff,
            "leftward": leftward_eff,
            "gripper": str(gripper),
            "pos_tol": float(pos_tol),
        },
        "gripper_requested": str(gripper),
        "gripper_cmd": grip_cmd,
        "gripper_predicted": grip_pred,
        "gripper_changed": grip_changed,
        "pos_tol": float(pos_tol),
        "converged_at_start": bool(converged_at_start),
        "move_err0_m": round(move_err0, 6),
        "preview_kind": preview_kind,
        "stop_residual_m": round(float(stop_residual_m), 6),
        "preview_travel_m": round(float(preview_travel_m), 6),
        "finger_qpos_before": _finger_mean_qpos(world, arm_eff),
        "camera": {
            "pos": cam_pos.round(6).tolist(),
            "quat": cam_quat.round(6).tolist(),
            "image_width": w_img,
            "image_height": h_img,
            "focal_length": fl_m,
            "horizontal_aperture": ha_m,
        },
        "render_image_path": preview_png,
        "base_image_path": base_rgb,
        "overlay_ok": bool(overlay_ok),
        "execution_preflight": execution_preflight,
        "note": "preview only: robot state was not changed",
    }
    record["next_move"] = record["move"]
    record["move_eef_args"] = record["move"]
    record["candidate"] = {
        "arm": arm_eff,
        "target": "move_eef",
        "eef_target": {
            "pos": record["target_eef_pose"]["pos"],
            "quat": record["target_eef_pose"]["quat"],
            "approach": approach.round(6).tolist(),
            "gripper_cmd": grip_cmd,
        },
        "eef_pose": record["target_eef_pose"],
        "next_eef_move": [0.0, 0.0, 0.0],
        "gripper_cmd": grip_cmd,
        "reachable": True,
        "selected_pose_ik_q": {
            arm_eff: execution_preflight["final_q"],
        },
        "meta": {
            "source": "plan_move_eef",
            "target_source": target_source,
            "exec_sequence": "move_only",
            "selected_pose_ik_q": {
                arm_eff: execution_preflight["final_q"],
            },
            "move_only_preflight": execution_preflight,
        },
    }
    agent_runs.save_plan_record(session_id, plan_id, record)

    ctx.log(
        f"plan_move_eef [{arm_eff}] source={target_source} cam Δ(cm) up={upward_eff:+.1f} "
        f"fwd={forward_eff:+.1f} left={leftward_eff:+.1f} → "
        f"eef {pos0.round(4).tolist()} -> preview {pos_preview.round(4).tolist()} "
        f"(ideal {pos_ideal.round(4).tolist()}, {preview_kind}) overlay={overlay_ok}"
    )
    ctx.set_result({
        "ok": True,
        "tool": "plan_move_eef",
        "plan_id": plan_id,
        "build": PLAN_MOVE_EEF_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "arm": arm_eff,
        "target_source": target_source,
        "uv_depth_target": uv_depth_target,
        "delta_cam_cm": record["delta_cam_cm"],
        "delta_world_m": record["delta_world_m"],
        "eef_before": record["eef_before"],
        "eef_pose": record["eef_pose"],
        "ideal_eef_pose": record["ideal_eef_pose"],
        "target_eef_pose": record["target_eef_pose"],
        "move": record["move"],
        "next_move": record["move"],
        "move_eef_args": record["move"],
        "camera": record["camera"],
        "gripper_requested": str(gripper),
        "gripper_cmd": grip_cmd,
        "gripper_predicted": grip_pred,
        "gripper_changed": grip_changed,
        "pos_tol": float(pos_tol),
        "converged_at_start": bool(converged_at_start),
        "move_err0_m": round(move_err0, 6),
        "preview_kind": preview_kind,
        "stop_residual_m": round(float(stop_residual_m), 6),
        "preview_travel_m": round(float(preview_travel_m), 6),
        "render_image": agent_runs.file_to_data_url(preview_png),
        "render_image_path": preview_png,
        "base_image_path": base_rgb,
        "overlay_ok": bool(overlay_ok),
        "execution_preflight": execution_preflight,
    })
    yield world.hold_action()
