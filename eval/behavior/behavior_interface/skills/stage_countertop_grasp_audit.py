"""在厨房台面摆放 grasp audit 用物体（apple / easter_egg，保留场景内 popcorn）。"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Tuple

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.plan_eef_lawn_capture import _safe_set_pose, _try_disable_dynamics

# 相对爆米花袋的台面摆放（世界系 xy，米）
# 与爆米花袋拉开距离，便于 head 分物体取景
_EXTRA_PLACEMENTS: Tuple[Tuple[str, str, str, Tuple[float, float]], ...] = (
    ("apple", "agveuv", "apple_agveuv", (7.08, -0.78)),
    ("easter_egg", "rxwfse", "easter_egg_rxwfse", (7.92, -0.22)),
)


def _remove_by_name(world, name: str, ctx=None) -> None:
    for existing in list(world.env.scene.objects):
        if getattr(existing, "name", None) != name:
            continue
        try:
            world.env.scene.remove_object(existing)
            if ctx:
                ctx.log(f"  [stage_ct] 移除旧物体 {name}")
        except Exception as e:
            if ctx:
                ctx.log(f"  [stage_ct] 移除 {name} 失败: {e}")


def _place_dataset_on_z(
    world,
    ctx,
    *,
    category: str,
    model: str,
    obj_name: str,
    xy: Tuple[float, float],
    z_contact: float,
) -> Dict[str, Any]:
    import torch as th
    from omnigibson.objects import DatasetObject

    _remove_by_name(world, obj_name, ctx=ctx)
    for _ in range(2):
        yield world.empty_action()

    quat_t = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    pos_t = th.tensor([float(xy[0]), float(xy[1]), z_contact + 0.05], dtype=th.float32)
    obj = DatasetObject(
        name=obj_name,
        category=category,
        model=model,
        position=pos_t.tolist(),
        orientation=quat_t.tolist(),
    )
    world.env.scene.add_object(obj)
    _try_disable_dynamics(obj, ctx=ctx)
    for _ in range(4):
        _safe_set_pose(obj, pos_t, quat_t)
        yield world.empty_action()

    try:
        lo, hi = obj.aabb
        z_bottom = float(lo[2])
        z_target = float(z_contact - (z_bottom - float(pos_t[2])))
    except Exception:
        z_target = float(z_contact)
    pos_t = th.tensor([float(xy[0]), float(xy[1]), z_target], dtype=th.float32)
    for _ in range(6):
        _safe_set_pose(obj, pos_t, quat_t)
        yield world.empty_action()

    try:
        lo, hi = obj.aabb
        ctr = ((np.asarray(lo) + np.asarray(hi)) * 0.5).tolist()
    except Exception:
        ctr = pos_t.tolist()
    if ctx:
        ctx.log(
            f"  [stage_ct] {obj_name} ({category}/{model}) "
            f"pos={pos_t.tolist()} ctr={[round(x, 3) for x in ctr]}"
        )
    return {
        "object_name": obj_name,
        "tag": obj_name,
        "category": category,
        "model": model,
        "pos": [float(x) for x in pos_t.tolist()],
        "centroid": ctr,
    }


@register_skill(
    "stage_countertop_grasp_audit",
    description="在台面摆放 apple_agveuv、easter_egg_rxwfse（与场景 popcorn 同高）",
)
def stage_countertop_grasp_audit(ctx) -> Generator:
    from behavior_interface.skills.grasp import _aabb_of, _resolve_object_handle

    world = ctx.world
    if world.dry_run:
        ctx.set_result({
            "ok": True,
            "objects": [
                {"object_name": "popcorn__bag.n.01_1", "tag": "popcorn_bag"},
                {"object_name": "apple_agveuv", "tag": "apple_agveuv"},
                {"object_name": "easter_egg_rxwfse", "tag": "easter_egg_rxwfse"},
            ],
        })
        yield world.empty_action()
        return

    pop = _resolve_object_handle(world, "popcorn__bag.n.01_1")
    if pop is None:
        ctx.set_result({"ok": False, "error": "未找到 popcorn__bag.n.01_1"})
        yield world.empty_action()
        return

    ab = _aabb_of(pop)
    if ab is None:
        ctx.set_result({"ok": False, "error": "爆米花袋 AABB 不可用"})
        yield world.empty_action()
        return
    lo, hi = ab
    z_contact = float(lo[2])
    pop_ctr = ((lo + hi) * 0.5).tolist()
    ctx.log(
        f"  [stage_ct] popcorn AABB lo={lo.round(3).tolist()} "
        f"hi={hi.round(3).tolist()} z_contact={z_contact:.3f}"
    )

    placed: List[Dict[str, Any]] = []
    for category, model, obj_name, xy in _EXTRA_PLACEMENTS:
        info = yield from _place_dataset_on_z(
            world, ctx,
            category=category, model=model, obj_name=obj_name,
            xy=xy, z_contact=z_contact,
        )
        placed.append(info)

    ctx.set_result({
        "ok": True,
        "popcorn": {
            "object_name": "popcorn__bag.n.01_1",
            "tag": "popcorn_bag",
            "centroid": pop_ctr,
        },
        "objects": placed,
    })
    for _ in range(3):
        yield world.empty_action()
