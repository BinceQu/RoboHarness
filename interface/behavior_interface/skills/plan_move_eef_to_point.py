"""plan_move_eef_to_point：点选 head 图上一点，规划 EEF 平移到其正上方。

输入 capture 的 image_id + Qwen3-VL 0..1000 相对坐标 (u,v)；注册表边界
先将其转换为原生像素，再按 plan_grasp_point 同口径反解冻结 head 图中的
3D hit，最后取 hit 世界 Z 轴正上方 upward cm 作为目标 EEF 中心。
返回可直接传给 move_eef 的相机系 move 参数，并生成与 plan_grasp_point 相同风格的
head 冻结图红夹爪叠影。
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_eef import MOVE_EEF_BUILD

PLAN_MOVE_EEF_TO_POINT_BUILD = "v1_point_world_z_to_move_eef"


def _camera_delta_from_world(cam_quat_xyzw: np.ndarray, delta_world: np.ndarray) -> Dict[str, float]:
    """世界系位移向量 (m) → move_eef 使用的 head 相机系 cm 参数。"""
    from behavior_interface.skills.grasp import _quat_to_mat

    rot = _quat_to_mat(np.asarray(cam_quat_xyzw, dtype=np.float64).reshape(4))
    delta_cam = rot.T @ np.asarray(delta_world, dtype=np.float64).reshape(3)
    return {
        "upward": float(delta_cam[1] * 100.0),
        "forward": float(-delta_cam[2] * 100.0),
        "leftward": float(-delta_cam[0] * 100.0),
    }


def _float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


@register_skill(
    "plan_move_eef_to_point",
    description=(
        "用 Qwen3-VL 0..1000 相对坐标点选 head 图，反解 3D hit，"
        "取 hit 世界 Z 正上方 upward cm "
        "作为 EEF 目标；返回 move_eef 参数和红色夹爪预览，不实际移动。"
    ),
)
def plan_move_eef_to_point(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    upward: float = 5.0,
    arm: str = "right",
    gripper: str = "keep",
    pos_tol: float = 0.012,
):
    from behavior_interface import agent_runs
    from behavior_interface.skills.plan_eef_core import _viz_grasp_obj_frozen_overlay
    from behavior_interface.skills.plan_eef_v2 import _build_session, _hit_at_uv
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    world = ctx.world
    arm_eff = str(arm or "right").strip().lower()
    if arm_eff not in ("left", "right"):
        ctx.set_result({"ok": False, "error": f"arm 必须是 left/right，收到 {arm!r}"})
        yield world.hold_action()
        return

    sid = (session_id or "").strip()
    img = (image_id or "").strip()
    if not sid or not img:
        ctx.set_result({"ok": False, "error": "需要 session_id / image_id"})
        yield world.hold_action()
        return

    try:
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(sid, img, arm_eff)
    except (FileNotFoundError, ValueError) as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    u_i = int(u)
    v_i = int(v)
    if u_i < 0 or v_i < 0:
        ctx.set_result({
            "ok": False,
            "error": "u, v 必须是 0..1000 的 Qwen3-VL 相对坐标",
        })
        yield world.hold_action()
        return

    ctx.log(
        f"[plan_move_eef_to_point] image={img} pixel=({u_i},{v_i}) "
        f"upward={float(upward):+.1f}cm cam=head res={w}x{h}"
    )

    yield world.hold_action()
    try:
        hit, hit_method = _hit_at_uv(session, u_i, v_i, cam_pos, cam_quat, w, h, fl, ha)
    except Exception as e:
        ctx.set_result({
            "ok": False,
            "error": f"反解 (u,v) 3D 点失败: {e}",
        })
        yield world.hold_action()
        return
    if hit is None:
        ctx.set_result({
            "ok": False,
            "error": (
                "无法在 head 图上反解 (u,v) 的 3D 点；请先 capture，"
                "并在有效 depth 区域点选。"
            ),
        })
        yield world.hold_action()
        return
    hit = np.asarray(hit, dtype=np.float64).reshape(3)
    target_pos = hit + np.array([0.0, 0.0, float(upward) / 100.0], dtype=np.float64)

    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = np.asarray(eef0["pos"], dtype=np.float64).reshape(3)
    quat0 = np.asarray(eef0["quat"], dtype=np.float64).reshape(4)
    delta_world = target_pos - pos0
    move_delta_cam = _camera_delta_from_world(cam_quat, delta_world)
    move = {
        "tool": "move_eef",
        "arm": arm_eff,
        "upward": move_delta_cam["upward"],
        "forward": move_delta_cam["forward"],
        "leftward": move_delta_cam["leftward"],
        "gripper": str(gripper),
        "pos_tol": float(pos_tol),
    }

    agent_runs.ensure_session(sid)
    plan_id = agent_runs.next_plan_id(sid)
    plan_png = agent_runs.plan_path(sid, plan_id, ".png")
    render_src = ""
    init_dir = session.get("init_dir", "")
    rgb_path = os.path.join(init_dir, "rgb.png")
    tmp_png = os.path.join(os.path.dirname(rgb_path), f"{plan_id}_plan_move_eef_to_point.png")

    contact_px = _world_to_pixel(cam_pos, cam_quat, hit, w, h, fl, ha) or (u_i, v_i)
    grip_fit = {
        "click_hit_world": hit.tolist(),
        "marker_pt": hit.tolist(),
        "anchor_marker_pt": target_pos.tolist(),
        "grasp_vol_cm3": 0.0,
        "overlap_vol_cm3": 0.0,
        "gripper_vol_cm3": 0.0,
        "overlap_frac": 0.0,
        "anchor_dist_mm": float(np.linalg.norm(target_pos - hit) * 1000.0),
    }
    overlay_ok = False
    if os.path.isfile(rgb_path):
        overlay_ok = _viz_grasp_obj_frozen_overlay(
            rgb_path, tmp_png,
            eef_pos=target_pos,
            eef_quat=quat0,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w_img=w,
            h_img=h,
            fl_m=fl,
            ha_m=ha,
            hit_world=hit,
            grip_fit=grip_fit,
            u=u_i,
            v=v_i,
            contact_u=int(contact_px[0]),
            contact_v=int(contact_px[1]),
            next_world=target_pos,
            tool_label="plan_move_eef_to_point",
            plan_mode="grasp_point",
            gap_center_world=target_pos,
        )
        if overlay_ok and os.path.isfile(tmp_png):
            shutil.copy2(tmp_png, plan_png)
            render_src = plan_png

    if not render_src and os.path.isfile(rgb_path):
        shutil.copy2(rgb_path, plan_png)
        render_src = plan_png

    move_norm_m = float(np.linalg.norm(delta_world))
    move_err0_m = move_norm_m
    converged_at_start = move_err0_m < float(pos_tol)

    record: Dict[str, Any] = {
        "plan_id": plan_id,
        "session_id": sid,
        "image_id": img,
        "skill": "plan_move_eef_to_point",
        "mode": "plan_move_eef_to_point",
        "build": PLAN_MOVE_EEF_TO_POINT_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "arm": arm_eff,
        "pixel": {"u": u_i, "v": v_i},
        "hit_world": hit.round(6).tolist(),
        "hit_method": hit_method,
        "point_upward_cm": float(upward),
        "eef_before": pos0.round(6).tolist(),
        "eef_pose": {
            "pos": target_pos.round(6).tolist(),
            "quat": quat0.round(6).tolist(),
        },
        "target_world": target_pos.round(6).tolist(),
        "delta_world_m": delta_world.round(6).tolist(),
        "delta_cam_cm": {
            "upward": round(move_delta_cam["upward"], 6),
            "forward": round(move_delta_cam["forward"], 6),
            "leftward": round(move_delta_cam["leftward"], 6),
        },
        "move": move,
        "next_move": move,
        "move_eef_args": move,
        "pos_tol": float(pos_tol),
        "converged_at_start": bool(converged_at_start),
        "move_err0_m": round(move_err0_m, 6),
        "camera": {
            "pos": np.asarray(cam_pos).round(6).tolist(),
            "quat": np.asarray(cam_quat).round(6).tolist(),
            "image_width": int(w),
            "image_height": int(h),
            "focal_length": float(fl),
            "horizontal_aperture": float(ha),
        },
        "grip_fit": grip_fit,
        "render_image_path": render_src,
        "overlay_ok": bool(overlay_ok),
        "note": "preview only: robot state was not changed",
    }
    agent_runs.save_plan_record(sid, plan_id, record)

    ctx.log(
        f"plan_move_eef_to_point [{arm_eff}] hit={hit.round(4).tolist()} "
        f"target={target_pos.round(4).tolist()} move_cam(cm)="
        f"up={move_delta_cam['upward']:+.2f} fwd={move_delta_cam['forward']:+.2f} "
        f"left={move_delta_cam['leftward']:+.2f} overlay={overlay_ok}"
    )
    ctx.set_result({
        "ok": True,
        "tool": "plan_move_eef_to_point",
        "plan_id": plan_id,
        "build": PLAN_MOVE_EEF_TO_POINT_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "arm": arm_eff,
        "image_id": img,
        "pixel": record["pixel"],
        "hit_world": record["hit_world"],
        "hit_method": hit_method,
        "point_upward_cm": float(upward),
        "eef_before": record["eef_before"],
        "eef_pose": record["eef_pose"],
        "target_world": record["target_world"],
        "delta_world_m": record["delta_world_m"],
        "delta_cam_cm": record["delta_cam_cm"],
        "move": move,
        "next_move": move,
        "move_eef_args": move,
        "pos_tol": float(pos_tol),
        "converged_at_start": bool(converged_at_start),
        "move_err0_m": round(move_err0_m, 6),
        "camera": record["camera"],
        "grip_fit": grip_fit,
        "render_image": agent_runs.file_to_data_url(render_src),
        "render_image_path": render_src,
        "overlay_ok": bool(overlay_ok),
    })
    yield world.hold_action()
