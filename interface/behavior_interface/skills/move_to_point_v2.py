"""move_to_point_v2 —— 点选 (u,v) 反解 3D 点作弦球球心 C。

与 move_to_object_v2 共用 _run_move_to_center（统一 升降→xy→spin/yaw→q3 移动管线）；
退出 capture 前稳定时长亦相同（_NAV_EXIT_SETTLE_S，当前 5s）。
唯一区别：C 来自光束/点云交点，而非物体 AABB 中心。
"""

from __future__ import annotations

import math
from typing import Any, Generator, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_to_object_v2 import (
    _MOVE_TO_OBJECT_BUILD,
    _NAV_TIMEOUT_S_DEFAULT,
    _aabb_around_point,
    _capture_head_snapshot,
    _hold,
    _run_move_to_center,
)

# 与 move_to_object / move_to_point_v3 同一套 point_pipeline
_MOVE_TO_POINT_BUILD = _MOVE_TO_OBJECT_BUILD

# 点选反投影用的是拍摄瞬间冻结的相机外参。机器人移动后再点同一张旧图，
# 反投影本身仍然正确，但解出的世界点相对「当前」机器人可能在侧后方，
# 于是底盘会转向一个用户在画面上完全看不出来的方位（极端情况直接掉头）。
_STALE_CAPTURE_SHIFT_M = 0.10
_STALE_CAPTURE_ROT_DEG = 5.0


def _capture_pose_drift(
    world: Any,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """返回拍摄时与当前 head 相机位姿的 (位移m, 转角deg)；读不到当前位姿时返回 None。"""
    try:
        from behavior_interface.skills.move_eef import _head_cam_pose

        now_pos, now_quat = _head_cam_pose(world)
    except Exception:
        return None
    if now_pos is None or now_quat is None:
        return None
    shift_m = float(np.linalg.norm(now_pos - np.asarray(cam_pos, dtype=np.float64)))
    q_ref = np.asarray(cam_quat, dtype=np.float64).reshape(4)
    q_now = np.asarray(now_quat, dtype=np.float64).reshape(4)
    n_ref = float(np.linalg.norm(q_ref))
    n_now = float(np.linalg.norm(q_now))
    if n_ref < 1e-9 or n_now < 1e-9:
        return None
    # 四元数 q 与 -q 表示同一旋转，取 |dot| 才是真实夹角。
    dot = abs(float(np.dot(q_ref / n_ref, q_now / n_now)))
    rot_deg = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
    return shift_m, rot_deg


def _sphere_center_from_uv(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """冻结 capture：光束 (u,v) + depth 点云 → 弦球球心 C（世界系）。"""
    from behavior_interface.skills.plan_eef_v2 import _build_session
    from behavior_interface.skills.plan_eef_core import _load_depth_seg
    from behavior_interface.skills.plan_grasp_gripper_fit import resolve_hit_on_surface
    from behavior_interface.skills.vlm_lawn_dual import _build_pointcloud

    try:
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(
            session_id, image_id, "right",
        )
    except (FileNotFoundError, ValueError) as e:
        return None, None, str(e)

    drift = _capture_pose_drift(getattr(ctx, "world", None), cam_pos, cam_quat)
    if drift is not None:
        shift_m, rot_deg = drift
        if shift_m > _STALE_CAPTURE_SHIFT_M or rot_deg > _STALE_CAPTURE_ROT_DEG:
            return None, None, (
                f"图像 {image_id} 已过期：拍摄后机器人已移动 {shift_m:.3f}m / "
                f"转过 {rot_deg:.1f}°"
                f"（阈值 {_STALE_CAPTURE_SHIFT_M:.2f}m / "
                f"{_STALE_CAPTURE_ROT_DEG:.0f}°）。"
                "在旧图上点选会把目标解到与当前画面不符的方位（可能在身后），"
                "请重新 capture 后再点选。"
            )
        ctx.log(
            f"[move_to_point_v2] 图像新鲜度 OK: 位移={shift_m:.3f}m "
            f"转角={rot_deg:.1f}°"
        )

    u_i, v_i = int(u), int(v)
    if u_i >= w - 8 or u_i <= 7:
        ctx.log(
            f"[move_to_point_v2] 警告: u={u_i} 贴近图像左右边缘 (宽={w})，"
            "反投影/命中可能不准，建议点选画面中部"
        )

    depth, _ = _load_depth_seg(session)
    pts = _build_pointcloud(depth, cam_pos, cam_quat, fl, ha)
    hit, method = resolve_hit_on_surface(
        u_i, v_i, depth, pts, cam_pos, cam_quat, w, h, fl, ha, gta=None,
    )
    if hit is None:
        return None, None, f"无法在 ({u_i},{v_i}) 反解 3D 点（depth/点云射线均无交点）"

    C = np.asarray(hit, dtype=np.float64).reshape(3)
    ctx.log(
        f"[move_to_point_v2] 弦球球心 C=hit({u_i},{v_i})="
        f"{C.round(3).tolist()} method={method} pcd_n={len(pts)}"
    )
    return C, method, None


@register_skill(
    "move_to_point_v2",
    description=(
        "点选 (u,v) 反解 3D 点作弦球球心；离线几何+执行与 move_to_object 相同；"
        "reach 弦球半径（0=0.60m）；nav_timeout_s 导航超时（默认120s）；"
        "keep_ori_arm 仅在底盘 spin 后的最终俯仰阶段生效，"
        "保持原 J1-J4 轨迹不变并逐帧仅用 J5-J7 追踪入口世界系 EEF 姿态；"
        "EEF 平移只监控。"
    ),
)
def move_to_point_v2(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    reach: float = 0.0,
    nav_timeout_s: float = _NAV_TIMEOUT_S_DEFAULT,
    keep_ori_arm: str = "none",
) -> Generator:
    world = ctx.world

    def _fail(msg: str) -> Generator:
        ctx.log(f"[move_to_point_v2] 中止: {msg}")
        yield from _hold(world, 4)
        head_live = _capture_head_snapshot(
            world,
            session_id=session_id,
            head_png_tag="move_to_point_head",
            log_tag="move_to_point_v2",
            ctx=ctx,
            suffix="failure",
        )
        ctx.set_result({
            "ok": False,
            "error": msg,
            "tool": "move_to_point",
            "visibility": head_live,
            "head_after_move_png": head_live.get("head_png"),
        })
        yield from _hold(world)
        return

    img = (image_id or "").strip()
    if not img:
        yield from _fail("需要 image_id（先 capture）")
        return
    if int(u) < 0 or int(v) < 0:
        yield from _fail("需要 head 图上点选 (u,v)")
        return

    C, hit_method, err = _sphere_center_from_uv(ctx, session_id, img, int(u), int(v))
    if err:
        yield from _fail(err)
        return
    assert C is not None

    # head 取景投影：点目标用 C 邻域小 AABB（物体版用整物体 AABB）
    lo, hi = _aabb_around_point(C, margin=0.03)

    yield from _run_move_to_center(
        ctx,
        session_id=session_id,
        C=C,
        lo=lo,
        hi=hi,
        reach=reach,
        tool="move_to_point",
        build_id=_MOVE_TO_POINT_BUILD,
        log_tag="move_to_point_v2",
        head_png_tag="move_to_point_head",
        result_extra={
            "image_id": img,
            "uv": [int(u), int(v)],
            "hit_world": [float(x) for x in C],
            "hit_method": hit_method,
            "reach_m": float(reach) if float(reach) > 0.01 else None,
            "sphere_center_source": "uv_hit",
            "exec_order": "lift_xy_spin_pitch",
            "trunk_mode": "q3_pitch_after_nav",
        },
        travel_hint="若仍不可达请重新点选或调整 reach",
        nav_timeout_s=nav_timeout_s,
        keep_ori_arm=keep_ori_arm,
    )
