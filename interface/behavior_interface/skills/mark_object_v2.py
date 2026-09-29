"""mark_object_v2 —— agent 工具 mark_object(img_id, u, v) 的仿真后端。

把图像上的点反投影到 3D，找到指向的物体，把该物体的"所有相关信息"链接成一个结构：
  - bddl 谓词用的 name（如 microwave.n.02_1）
  - base 世界位姿（pos/quat/yaw）
  - AABB（min/max/size）
  - 物体点云质心（世界系）
  - category / model / scene_name / prim_path / usd(mesh) 文件路径
落到 session 的 memory.json（结构化、可被程序链接）与 memory.md（人类可读），
并返回该物体唯一的 bddl name。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill

# 这些 category 太"大"（地板/墙/天花板/机器人本体），不作为 mark 目标
_SKIP_CATEGORIES = {"floors", "floor", "walls", "wall", "ceilings", "ceiling",
                     "lawn", "driveway", "agent"}


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _aabb(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        lo, hi = obj.aabb
        return _to_np(lo).reshape(-1), _to_np(hi).reshape(-1)
    except Exception:
        return None


def _iter_scope_objects(world):
    """遍历 BDDL object_scope：yield (bddl_name, obj)。"""
    try:
        task = world.env.task
        scope = task.object_scope or {}
    except Exception:
        scope = {}
    for bddl_name, ent in scope.items():
        obj = getattr(ent, "unwrapped", ent)
        if obj is None or not hasattr(obj, "get_position_orientation"):
            continue
        try:
            if not getattr(ent, "exists", True):
                continue
        except Exception:
            pass
        yield bddl_name, obj


def _scene_objects_iter(world):
    """遍历 scene 中已加载物体（含 stage 动态摆放的 DatasetObject）。"""
    try:
        for obj in world.env.scene.objects:
            if obj is None or not hasattr(obj, "get_position_orientation"):
                continue
            name = getattr(obj, "name", None)
            if not name:
                continue
            cat = (getattr(obj, "category", "") or "").lower()
            if cat in _SKIP_CATEGORIES:
                continue
            yield str(name), obj
    except Exception:
        return


def _find_object_at(world, hit: np.ndarray) -> Tuple[Optional[str], Any]:
    """点 hit 指向哪个物体：BDDL scope 优先，其次 scene.objects（stage 摆放）。"""
    contains: List[Tuple[float, str, Any]] = []
    nearest: List[Tuple[float, str, Any]] = []
    for bddl_name, obj in _iter_scope_objects(world):
        cat = (getattr(obj, "category", "") or "").lower()
        if cat in _SKIP_CATEGORIES:
            continue
        ab = _aabb(obj)
        if ab is None:
            continue
        lo, hi = ab
        center = (lo + hi) / 2.0
        vol = float(np.prod(np.maximum(hi - lo, 1e-3)))
        margin = 0.05
        if bool(np.all(hit >= lo - margin) and np.all(hit <= hi + margin)):
            contains.append((vol, bddl_name, obj))
        nearest.append((float(np.linalg.norm(hit - center)), bddl_name, obj))
    if contains:
        contains.sort(key=lambda t: t[0])  # 体积最小 = 最具体
        return contains[0][1], contains[0][2]
    if nearest:
        nearest.sort(key=lambda t: t[0])
        if nearest[0][0] < 0.6:  # 0.6m 内才认
            return nearest[0][1], nearest[0][2]

    # stage 动态物体可能不在 BDDL object_scope
    contains2: List[Tuple[float, str, Any]] = []
    nearest2: List[Tuple[float, str, Any]] = []
    for scene_name, obj in _scene_objects_iter(world):
        ab = _aabb(obj)
        if ab is None:
            continue
        lo, hi = ab
        center = (lo + hi) / 2.0
        vol = float(np.prod(np.maximum(hi - lo, 1e-3)))
        margin = 0.05
        if bool(np.all(hit >= lo - margin) and np.all(hit <= hi + margin)):
            contains2.append((vol, scene_name, obj))
        nearest2.append((float(np.linalg.norm(hit - center)), scene_name, obj))
    if contains2:
        contains2.sort(key=lambda t: t[0])
        return contains2[0][1], contains2[0][2]
    if nearest2:
        nearest2.sort(key=lambda t: t[0])
        if nearest2[0][0] < 0.35:
            return nearest2[0][1], nearest2[0][2]
    return None, None


def _object_usd_path(obj) -> Optional[str]:
    for attr in ("usd_path", "_usd_path"):
        p = getattr(obj, attr, None)
        if isinstance(p, str) and p:
            return p
    try:
        cat = getattr(obj, "category", None)
        mdl = getattr(obj, "model", None)
        if cat and mdl and hasattr(type(obj), "get_usd_path"):
            return type(obj).get_usd_path(category=cat, model=mdl)
    except Exception:
        pass
    return None


def _gather_object_info(obj, bddl_name: str, hit: np.ndarray,
                        pcd_centroid: Optional[np.ndarray]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"bddl_name": bddl_name}
    try:
        pos, quat = obj.get_position_orientation()
        pos = _to_np(pos).reshape(-1)
        quat = _to_np(quat).reshape(-1)
        import math
        x, y, z, w = [float(q) for q in quat[:4]]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        info["base_pose_world"] = {
            "pos": [float(v) for v in pos[:3]],
            "quat_xyzw": [float(v) for v in quat[:4]],
            "yaw_deg": round(yaw, 2),
        }
    except Exception:
        info["base_pose_world"] = None
    ab = _aabb(obj)
    if ab is not None:
        lo, hi = ab
        info["aabb"] = {
            "min": [float(v) for v in lo], "max": [float(v) for v in hi],
            "size": [float(v) for v in (hi - lo)],
            "center": [float(v) for v in (lo + hi) / 2.0],
        }
    info["category"] = getattr(obj, "category", None)
    info["model"] = getattr(obj, "model", None)
    info["scene_name"] = getattr(obj, "name", None)
    info["prim_path"] = getattr(obj, "prim_path", None)
    info["usd_path"] = _object_usd_path(obj)
    info["hit_world"] = [float(v) for v in np.asarray(hit)]
    if pcd_centroid is not None:
        info["pcd_centroid_world"] = [float(v) for v in np.asarray(pcd_centroid)]
    return info


def _write_memory(session_id: str, info: Dict[str, Any], image_id: str, u: int, v: int) -> str:
    """把 info 写入 session/memory.json（结构化链接）+ memory.md（人类可读）。"""
    import json

    run = agent_runs.run_dir(session_id)
    os.makedirs(run, exist_ok=True)
    mem_json = os.path.join(run, "memory.json")
    mem_md = os.path.join(run, "memory.md")

    mem: Dict[str, Any] = {"objects": {}, "marks": []}
    if os.path.isfile(mem_json):
        try:
            with open(mem_json, encoding="utf-8") as f:
                mem = json.load(f)
        except Exception:
            pass
    mem.setdefault("objects", {})
    mem.setdefault("marks", [])
    bddl = info["bddl_name"]
    mem["objects"][bddl] = info
    mem["marks"].append({"image_id": image_id, "u": int(u), "v": int(v),
                         "bddl_name": bddl, "t": time.time()})
    with open(mem_json, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

    # memory.md：整体重写（objects 段 + marks 段），保证可读且不重复
    lines = ["# Session Memory", "",
             f"_session: {session_id}_", "",
             "## Marked Objects", ""]
    for name, oi in mem["objects"].items():
        bp = oi.get("base_pose_world") or {}
        ab = oi.get("aabb") or {}
        lines += [
            f"### {name}",
            f"- category / model: `{oi.get('category')}` / `{oi.get('model')}`",
            f"- scene_name: `{oi.get('scene_name')}`",
            f"- base_pose_world: pos={bp.get('pos')} yaw={bp.get('yaw_deg')}°",
            f"- aabb: center={ab.get('center')} size={ab.get('size')}",
            f"- pcd_centroid_world: {oi.get('pcd_centroid_world')}",
            f"- prim_path: `{oi.get('prim_path')}`",
            f"- usd(mesh): `{oi.get('usd_path')}`",
            "",
        ]
    lines += ["## Mark Log", ""]
    for m in mem["marks"][-30:]:
        lines.append(f"- {m['image_id']} ({m['u']},{m['v']}) → **{m['bddl_name']}**")
    with open(mem_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return mem_md


@register_skill(
    "mark_object_v2",
    description=(
        "标记物体：把 image_id 上 (u,v) 反投影到 3D，找到指向的 BDDL 物体，"
        "将其 base 位姿/AABB/点云质心/category/model/usd 路径链接写入 session 的 "
        "memory.json + memory.md，返回该物体唯一的 bddl name。"
    ),
)
def mark_object_v2(
    ctx,
    session_id: str,
    image_id: str,
    u: Optional[int] = None,
    v: Optional[int] = None,
    object_name: str = "",
) -> Generator:
    from behavior_interface.skills.grasp import _resolve_object_handle
    from behavior_interface.skills.plan_eef_v2 import _build_session, _hit_at_uv
    from behavior_interface.skills.plan_eef_core import _load_depth_seg
    from behavior_interface.skills.plan_grasp_gripper_fit import build_object_pointcloud
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    world = ctx.world
    name = (object_name or "").strip()
    use_uv = u is not None and v is not None

    try:
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(session_id, image_id, "right")
    except (FileNotFoundError, ValueError) as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.empty_action()
        return

    yield world.empty_action()
    hit_method = "uv_ray"
    bddl_name: Optional[str] = None
    obj = None
    hit = None
    ab: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if name:
        obj = _resolve_object_handle(world, name)
        if obj is None:
            ctx.set_result({"ok": False, "error": f"mark_object 未找到物体: {name}"})
            yield world.empty_action()
            return
        ab = _aabb(obj)
        if ab is None:
            ctx.set_result({"ok": False, "error": f"{name} 无 AABB"})
            yield world.empty_action()
            return
        lo, hi = ab
        hit = (lo + hi) / 2.0
        bddl_name = name
        hit_method = "object_name_aabb"
        px = _world_to_pixel(cam_pos, cam_quat, hit, w, h, fl, ha)
        if px is not None:
            u, v = int(px[0]), int(px[1])
            use_uv = True
        elif use_uv:
            pass
        else:
            ctx.set_result({"ok": False, "error": f"{name} AABB 中心无法投影到 head 图"})
            yield world.empty_action()
            return
    elif use_uv:
        u, v = int(u), int(v)
        hit, hit_method = _hit_at_uv(session, u, v, cam_pos, cam_quat, w, h, fl, ha)
        if hit is None:
            ctx.set_result({"ok": False, "error": "mark_object 无法反解 (u,v) 处 3D 点"})
            yield world.empty_action()
            return
    else:
        ctx.set_result({"ok": False, "error": "需要 object_name 或 image_id + u + v"})
        yield world.empty_action()
        return

    # 物体点云质心（depth+seg；按 AABB 中心作 hit_ref 裁切）
    pcd_centroid = None
    crop_r = None
    if ab is not None:
        lo, hi = ab
        crop_r = float(max(0.12, min(0.28, np.linalg.norm(hi - lo) * 0.9)))
    try:
        depth, seg = _load_depth_seg(session)
        pts, _ = build_object_pointcloud(
            depth, seg, cam_pos, cam_quat, fl, ha, int(u), int(v),
            hit_ref=np.asarray(hit),
            max_radius_from_ref=crop_r,
        )
        if len(pts) >= 8:
            pcd_centroid = pts.mean(axis=0)
    except Exception:
        pass

    yield world.empty_action()
    if bddl_name is None or obj is None:
        bddl_name, obj = _find_object_at(world, np.asarray(hit))
    if bddl_name is None or obj is None:
        ctx.set_result({
            "ok": False,
            "error": "该点附近未找到物体",
            "hit_world": [float(x) for x in np.asarray(hit)],
        })
        yield world.empty_action()
        return

    info = _gather_object_info(obj, bddl_name, np.asarray(hit), pcd_centroid)
    mem_md = _write_memory(session_id, info, image_id, int(u), int(v))

    ctx.set_result({
        "ok": True,
        "tool": "mark_object",
        "bddl_name": bddl_name,
        "object": info,
        "memory_md": mem_md,
        "resolved_by": hit_method,
        "uv": [int(u), int(v)],
    })
    ctx.log(
        f"[mark_object_v2] ({u},{v}) method={hit_method} → {bddl_name} written to {mem_md}"
    )
    yield world.empty_action()
