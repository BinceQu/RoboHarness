"""Measure current left/right shoulder distance to an object center or clicked 3D point."""

from __future__ import annotations

from typing import Generator

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_to_object_v2 import (
    REACH_SPHERE_R_M,
    _aabb,
    _hold,
    _reach_report_live,
    _resolve,
    _shoulder_distance_result,
)
from behavior_interface.skills.move_to_point_v2 import _sphere_center_from_uv


def _fmt_distance(x) -> str:
    try:
        return f"{float(x):.4f}m"
    except Exception:
        return "N/A"


@register_skill(
    "mesure_shoulder_distance",
    description=(
        "测量当前左右肩到目标的距离；object_name 使用物体 AABB 中心，"
        "否则使用 head 图点击点反解 3D。"
    ),
)
def mesure_shoulder_distance(
    ctx,
    session_id: str = "",
    object_name: str = "",
    image_id: str = "",
    u: Optional[int] = None,
    v: Optional[int] = None,
) -> Generator:
    world = ctx.world

    def _fail(msg: str):
        ctx.log(f"[mesure_shoulder_distance] 中止: {msg}")
        ctx.set_result({
            "ok": False,
            "tool": "mesure_shoulder_distance",
            "error": msg,
        })
        yield from _hold(world, 2)

    obj_name = (object_name or "").strip()
    img = (image_id or "").strip()
    center = None
    source = ""
    hit_method = None

    if obj_name:
        obj = _resolve(world, obj_name)
        if obj is None:
            yield from _fail(f"找不到物体 {obj_name!r}")
            return
        aabb = _aabb(obj)
        if aabb is None:
            yield from _fail(f"无法读取物体 {obj_name!r} 的 AABB")
            return
        lo, hi = aabb
        center = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) * 0.5
        source = "object_aabb_center"
        ctx.log(
            "[mesure_shoulder_distance] object_name 优先，"
            f"目标物体={obj_name} center={center.round(4).tolist()}"
        )
    else:
        if not img:
            yield from _fail("需要 object_name，或 image_id + u + v")
            return
        if int(u) < 0 or int(v) < 0:
            yield from _fail("点击模式需要有效的 u/v")
            return
        center, hit_method, err = _sphere_center_from_uv(ctx, session_id, img, int(u), int(v))
        if err:
            yield from _fail(err)
            return
        if center is None:
            yield from _fail("无法反解点击点 3D")
            return
        source = "uv_hit"
        ctx.log(
            "[mesure_shoulder_distance] click 反解目标 "
            f"image_id={img} uv=({int(u)},{int(v)}) "
            f"center={np.asarray(center).round(4).tolist()} method={hit_method}"
        )

    center = np.asarray(center, dtype=np.float64).reshape(3)
    dual = _reach_report_live(world, center)
    dist = _shoulder_distance_result(dual, center, reach_R_m=REACH_SPHERE_R_M)
    left_d = dist.get("left_shoulder_to_object_m")
    right_d = dist.get("right_shoulder_to_object_m")
    message = f"肩距: left={_fmt_distance(left_d)} right={_fmt_distance(right_d)}"
    ctx.log(f"[mesure_shoulder_distance] {message}")

    result = {
        "ok": True,
        "tool": "mesure_shoulder_distance",
        "message": message,
        "target_source": source,
        "target_center_world": [float(x) for x in center],
        **dist,
    }
    if obj_name:
        result["object_name"] = obj_name
    if img:
        result["image_id"] = img
    if not obj_name and int(u) >= 0 and int(v) >= 0:
        result["uv"] = [int(u), int(v)]
        result["hit_method"] = hit_method
    ctx.set_result(result)
    yield from _hold(world, 2)
