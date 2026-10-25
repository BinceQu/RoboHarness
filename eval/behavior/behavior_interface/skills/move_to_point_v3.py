"""move_to_point_v3 —— 点选 3D 点作弦球球心。

真机顺序：
  ① 升降/预降
  ② 平移 xy
  ③ 底盘 spin/yaw
  ④ 纯 q3 俯仰对准规划 theta_z
"""

from __future__ import annotations

from typing import Generator

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
from behavior_interface.skills.move_to_point_v2 import _sphere_center_from_uv

_MOVE_TO_POINT_BUILD = _MOVE_TO_OBJECT_BUILD


@register_skill(
    "move_to_point_v3",
    description=(
        "点选 (u,v)→弦球球心；执行 ①升降 ②xy ③spin/yaw ④纯q3俯仰；"
        "支持点地面；reach 弦球半径（0=0.60m）；"
        "keep_ori_arm 仅在④最终俯仰阶段生效，保持原 J1-J4 轨迹不变，"
        "逐帧仅用 J5-J7 追踪入口世界系 EEF 姿态；EEF 平移只监控。"
    ),
)
def move_to_point_v3(
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
        ctx.log(f"[move_to_point_v3] 中止: {msg}")
        yield from _hold(world, 4)
        head_live = _capture_head_snapshot(
            world,
            session_id=session_id,
            head_png_tag="move_to_point_head",
            log_tag="move_to_point_v3",
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
        log_tag="move_to_point_v3",
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
