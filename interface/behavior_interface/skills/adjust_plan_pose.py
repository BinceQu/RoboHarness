"""adjust_plan_pose：调整已有 move 的 EEF 6D pose 并返回 head 相机红爪预览。

平移参数 forward / upward / leftward 与 move_eef 完全一致：
  forward  = head camera -Z（进场景）
  upward   = head camera +Y
  leftward = head camera -X
它们只改 eef_target.pos，和夹爪四元数无关。

旋转参数 roll / pitch / yaw 只作用在输入 move 的 eef_target.quat 上，和左右手无关。
它们以当前 move 的 EEF pose 为基准做局部旋转：
  roll  绕局部 gripper +X（夹爪中心 -> 爪尖）
  pitch 正值为 nose-up / fingertips-up
  yaw   绕局部 gripper +Z（夹爪中心 -> wrist camera），正值为 left turn
"""

from __future__ import annotations

import copy
import math
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.gripper_geometry_calibration import (
    R1PRO_GRIPPER_LINK_Z_EEF_M,
)
from behavior_interface.skills import register_skill
from behavior_interface.skills.move_eef import MOVE_EEF_BUILD, camera_delta_to_world
from behavior_interface.skills.plan_move_eef import (
    _drop_current_gripper_from_depth,
    _head_camera_meta,
    _head_sensor,
    _save_head_rgb_depth_from_obs,
)

ADJUST_PLAN_POSE_BUILD = "v3_no_arm_input_or_metadata_pose_only_preview"

# These are the same real-gripper axes used by
# test/adjust_plan_pose/gripper_true_mesh_rpy_axes_corrected.png, expressed in
# the EEF local frame used by gripper_trimesh_eef()/render_gripper_overlay().
_RY_PI_MAT = np.diag([-1.0, 1.0, -1.0])
_PALM_T_EEF = np.array(
    [0.0, 0.0, R1PRO_GRIPPER_LINK_Z_EEF_M],
    dtype=np.float64,
)
_RIGHT_REALSENSE_ORIGIN_GRIPPER = np.array([0.05051, 0.0028934, 0.0051317], dtype=np.float64)
_GRIPPER_ROLL_AXIS_EEF = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _gripper_axes_in_eef() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (roll_axis, pitch_axis, yaw_axis) in the EEF local frame."""
    roll_axis = _GRIPPER_ROLL_AXIS_EEF.copy()
    wrist_cam_eef = _RY_PI_MAT @ _RIGHT_REALSENSE_ORIGIN_GRIPPER + _PALM_T_EEF
    yaw_raw = wrist_cam_eef - roll_axis * float(np.dot(wrist_cam_eef, roll_axis))
    if float(np.linalg.norm(yaw_raw)) < 1e-9:
        yaw_raw = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    yaw_axis = yaw_raw / float(np.linalg.norm(yaw_raw))
    pitch_axis = np.cross(yaw_axis, roll_axis)
    pitch_axis = pitch_axis / float(np.linalg.norm(pitch_axis))
    yaw_axis = np.cross(roll_axis, pitch_axis)
    yaw_axis = yaw_axis / float(np.linalg.norm(yaw_axis))
    return roll_axis, pitch_axis, yaw_axis


def _as_dict(value: Any, *, name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{name} 必须是 dict")


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(float(angle_rad)) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / n
    half = float(angle_rad) * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64)


def _local_gripper_rpy_delta_quat(
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """局部夹爪 RPY 增量四元数。

    这里的 pitch 采用用户确认后的语义正方向：+pitch = nose-up /
    fingertips-up。因此它是绕局部 +Y 轴的负右手定则旋转。
    """
    from behavior_interface.skills.grasp import _quat_mul

    roll_axis, pitch_axis, yaw_axis = _gripper_axes_in_eef()
    q_roll = _axis_angle_quat(roll_axis, math.radians(float(roll_deg)))
    q_pitch = _axis_angle_quat(pitch_axis, -math.radians(float(pitch_deg)))
    q_yaw = _axis_angle_quat(yaw_axis, math.radians(float(yaw_deg)))
    # 局部内旋：按 roll -> pitch -> yaw 依次作用。
    return _quat_normalize(_quat_mul(_quat_mul(q_roll, q_pitch), q_yaw))


def _apply_local_delta_quat(quat_xyzw: np.ndarray, q_delta_local: np.ndarray) -> np.ndarray:
    from behavior_interface.skills.grasp import _quat_mul

    # 局部旋转需要右乘到当前 EEF 四元数。
    return _quat_normalize(_quat_mul(_quat_normalize(quat_xyzw), _quat_normalize(q_delta_local)))


def _extract_pose_from_move(move: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """从 plan record / move dict 中抽取 (pos, quat, eef_target_dict)。"""
    candidates = move.get("candidates")
    if isinstance(candidates, list) and candidates:
        cand0 = candidates[0] if isinstance(candidates[0], dict) else {}
        et = cand0.get("eef_target") if isinstance(cand0.get("eef_target"), dict) else None
        if et is not None and et.get("pos") is not None:
            pos = np.asarray(et["pos"], dtype=np.float64).reshape(3)
            quat = np.asarray(et.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
            return pos, _quat_normalize(quat), et

    cand = move.get("candidate") if isinstance(move.get("candidate"), dict) else {}
    et = cand.get("eef_target") if isinstance(cand.get("eef_target"), dict) else None
    if et is not None and et.get("pos") is not None:
        pos = np.asarray(et["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(et.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        return pos, _quat_normalize(quat), et

    et = move.get("eef_target") if isinstance(move.get("eef_target"), dict) else None
    if et is not None and et.get("pos") is not None:
        pos = np.asarray(et["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(et.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        return pos, _quat_normalize(quat), et

    ep = move.get("eef_pose") if isinstance(move.get("eef_pose"), dict) else None
    if ep is not None and ep.get("pos") is not None:
        pos = np.asarray(ep["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(ep.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        return pos, _quat_normalize(quat), ep

    if move.get("pos") is not None:
        pos = np.asarray(move["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(move.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        return pos, _quat_normalize(quat), move

    raise ValueError("move 中找不到 eef_target/eef_pose 的 pos/quat")


def _ensure_candidate(
    source: Dict[str, Any],
    *,
    pos: np.ndarray,
    quat: np.ndarray,
) -> Dict[str, Any]:
    """返回带 candidate.eef_target 的 record 副本，供 exec_* 直接消费。"""
    record = copy.deepcopy(source)
    cand = record.get("candidate") if isinstance(record.get("candidate"), dict) else None
    if cand is None:
        cands = record.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            cand = copy.deepcopy(cands[0])
        else:
            cand = {
                "id": 0,
                "target": record.get("target", "grasp"),
                "label": "adjust_plan_pose",
                "next_eef_move": record.get("next_eef_move_delta", record.get("next_eef_move", [0.0, 0.0, 0.0])),
                "reachable": True,
                "score": 1.0,
                "meta": {},
            }
    et = cand.get("eef_target") if isinstance(cand.get("eef_target"), dict) else {}
    et["pos"] = np.asarray(pos, dtype=np.float64).reshape(3).tolist()
    et["quat"] = _quat_normalize(quat).tolist()
    try:
        from behavior_interface.skills.grasp import _quat_to_mat

        et["approach"] = _quat_to_mat(_quat_normalize(quat))[:, 2].tolist()
    except Exception:
        et.setdefault("approach", [0.0, 0.0, 1.0])
    if "gripper_cmd" not in et:
        grip_hint = str(
            record.get("gripper_predicted")
            or record.get("gripper_requested")
            or record.get("gripper")
            or ""
        ).strip().lower()
        et["gripper_cmd"] = 1.0 if grip_hint == "open" else -1.0
    cand["eef_target"] = et
    cand.pop("arm", None)
    meta = cand.setdefault("meta", {})
    meta["adjust_plan_pose"] = True
    meta.setdefault("exec_sequence", meta.get("exec_sequence", "move_then_close"))
    record["candidate"] = cand
    record["candidates"] = [cand]
    record.pop("arm", None)
    return record


def _load_source_move(
    *,
    session_id: str,
    plan_id: Optional[str],
    move: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    if move is not None:
        return copy.deepcopy(_as_dict(move, name="move")), "inline_move"
    if not plan_id:
        raise ValueError("需要 plan_id 或 move")
    return agent_runs.load_plan_record(session_id, plan_id), str(plan_id)


def _render_adjusted_head_overlay(
    *,
    ctx,
    world,
    session_id: str,
    plan_id: str,
    pos_before: np.ndarray,
    quat_before: np.ndarray,
    pos_after: np.ndarray,
    quat_after: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> Tuple[str, str, bool]:
    from behavior_interface.head_capture import reset_head_sensor_to_mount
    from behavior_interface.skills.viz_gripper_overlay import render_gripper_overlay

    head = _head_sensor(world)
    if head is None:
        return "", "", False

    if reset_head_sensor_to_mount(world, ctx=ctx):
        yield world.hold_action()

    base_rgb = agent_runs.plan_path(session_id, f"{plan_id}_adjust_plan_pose_base", ".png")
    base_depth = agent_runs.plan_path(session_id, f"{plan_id}_adjust_plan_pose_depth", ".npy")
    preview_png = agent_runs.plan_path(session_id, plan_id, ".png")

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
        return "", base_rgb, False
    shutil.copy2(base_rgb, preview_png)

    w_img, h_img, fl_m, ha_m = _head_camera_meta(head)
    scene_depth_for_overlay = _drop_current_gripper_from_depth(
        scene_depth,
        eef_pos=pos_before,
        eef_quat=quat_before,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        w=w_img,
        h=h_img,
        fl=fl_m,
        ha=ha_m,
    )
    overlay_ok = render_gripper_overlay(
        preview_png,
        eef_pos=pos_after,
        eef_quat=quat_after,
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
    return preview_png, base_rgb, bool(overlay_ok)


@register_skill(
    "adjust_plan_pose",
    description=(
        "调整已有 move/eef_pose：forward/upward/leftward 按 head 相机系平移 pos；"
        "roll/pitch/yaw 按 move 的 EEF 局部夹爪轴调整 quat；返回新 move 和 head 红爪预览。"
    ),
)
def adjust_plan_pose(
    ctx,
    session_id: str,
    plan_id: str = "",
    move: Optional[Dict[str, Any]] = None,
    forward: float = 0.0,
    upward: float = 0.0,
    leftward: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
):
    """调整 move 的 EEF 6D pose，不执行机器人动作。"""
    world = ctx.world
    sid = (session_id or "").strip()
    if not sid:
        ctx.set_result({"ok": False, "error": "需要 session_id"})
        yield world.hold_action()
        return

    try:
        source, source_id = _load_source_move(
            session_id=sid,
            plan_id=(plan_id or "").strip() or None,
            move=move,
        )
        pos0, quat0, _et0 = _extract_pose_from_move(source)
    except Exception as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    cam_pos, cam_quat = None, None
    try:
        from behavior_interface.skills.move_eef import _head_cam_pose

        cam_pos, cam_quat = _head_cam_pose(world)
    except Exception:
        pass
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系平移"})
        yield world.hold_action()
        return

    delta_world = camera_delta_to_world(
        cam_quat,
        upward_cm=float(upward),
        forward_cm=float(forward),
        leftward_cm=float(leftward),
    )
    pos_adj = pos0 + delta_world

    q_delta_local = _local_gripper_rpy_delta_quat(
        roll_deg=float(roll),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
    )
    quat_adj = _apply_local_delta_quat(quat0, q_delta_local)

    agent_runs.ensure_session(sid)
    new_plan_id = agent_runs.next_plan_id(sid)
    record = _ensure_candidate(source, pos=pos_adj, quat=quat_adj)

    render_image_path = ""
    base_image_path = ""
    overlay_ok = False
    try:
        render_gen = _render_adjusted_head_overlay(
            ctx=ctx,
            world=world,
            session_id=sid,
            plan_id=new_plan_id,
            pos_before=pos0,
            quat_before=quat0,
            pos_after=pos_adj,
            quat_after=quat_adj,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
        )
        try:
            while True:
                yield next(render_gen)
        except StopIteration as done:
            render_image_path, base_image_path, overlay_ok = done.value
    except Exception as e:
        ctx.log(f"adjust_plan_pose WARN head 红爪预览失败: {e}")

    exec_eef_pose_args = {
        "tool": "exec_eef_pose",
        "session_id": sid,
        "plan_id": new_plan_id,
    }
    adjusted_move = {
        "plan_id": new_plan_id,
        "session_id": sid,
        "tool": "adjust_plan_pose",
        "source_plan_id": source_id,
        "eef_pose": {
            "pos": pos_adj.round(6).tolist(),
            "quat": quat_adj.round(6).tolist(),
        },
        "candidate": record["candidate"],
        "exec_eef_pose_args": exec_eef_pose_args,
    }

    record.update({
        "plan_id": new_plan_id,
        "session_id": sid,
        "skill": "adjust_plan_pose",
        "mode": "adjust_plan_pose",
        "build": ADJUST_PLAN_POSE_BUILD,
        "move_eef_build_ref": MOVE_EEF_BUILD,
        "source_plan_id": source_id,
        "source_skill": source.get("skill"),
        "eef_pose": {
            "pos": pos_adj.round(6).tolist(),
            "quat": quat_adj.round(6).tolist(),
        },
        "eef_pose_before": {
            "pos": pos0.round(6).tolist(),
            "quat": quat0.round(6).tolist(),
        },
        "delta_cam_cm": {
            "forward": float(forward),
            "upward": float(upward),
            "leftward": float(leftward),
        },
        "delta_world_m": delta_world.round(6).tolist(),
        "rpy_deg": {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
        "rpy_convention": {
            "translation": "forward/upward/leftward are head-camera-frame translations and only change eef_target.pos",
            "rotation": "roll/pitch/yaw are local to the input move eef_target.quat and are independent of left/right arm",
            "positive_roll": "from the fingertip side, positive roll follows the corrected tri-view red arc",
            "positive_pitch": "positive pitch is nose-up / fingertips-up",
            "positive_yaw": "positive yaw is a left turn around the wrist-camera axis",
        },
        "move": adjusted_move,
        "next_move": adjusted_move,
        "exec_eef_pose_args": exec_eef_pose_args,
        "move_eef_args": None,
        "render_image_path": render_image_path,
        "base_image_path": base_image_path,
        "overlay_ok": bool(overlay_ok),
        "camera": {
            "pos": np.asarray(cam_pos).round(6).tolist(),
            "quat": np.asarray(cam_quat).round(6).tolist(),
        },
        "note": "preview only: robot state was not changed",
    })
    # candidate 里也要确保保存的是最终 pose，而不是 source 里的旧 rounded 值。
    cand = record["candidate"]
    cand["eef_target"]["pos"] = pos_adj.tolist()
    cand["eef_target"]["quat"] = quat_adj.tolist()
    agent_runs.save_plan_record(sid, new_plan_id, record)

    ctx.log(
        f"adjust_plan_pose source={source_id} -> {new_plan_id} "
        f"camΔ(cm) fwd={float(forward):+.1f} up={float(upward):+.1f} left={float(leftward):+.1f} "
        f"rpy(deg) roll={float(roll):+.1f} pitch={float(pitch):+.1f} yaw={float(yaw):+.1f} "
        f"pos {pos0.round(4).tolist()} -> {pos_adj.round(4).tolist()} overlay={overlay_ok}"
    )
    ctx.set_result({
        "ok": True,
        "tool": "adjust_plan_pose",
        "plan_id": new_plan_id,
        "source_plan_id": source_id,
        "build": ADJUST_PLAN_POSE_BUILD,
        "eef_pose_before": record["eef_pose_before"],
        "eef_pose": record["eef_pose"],
        "delta_cam_cm": record["delta_cam_cm"],
        "delta_world_m": record["delta_world_m"],
        "rpy_deg": record["rpy_deg"],
        "rpy_convention": record["rpy_convention"],
        "move": adjusted_move,
        "next_move": adjusted_move,
        "exec_eef_pose_args": exec_eef_pose_args,
        "render_image": agent_runs.file_to_data_url(render_image_path),
        "render_image_path": render_image_path,
        "base_image_path": base_image_path,
        "overlay_ok": bool(overlay_ok),
    })
    yield world.hold_action()
