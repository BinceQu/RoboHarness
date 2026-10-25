"""plan_eef 草坪悬浮 capture：与 nova_vlm_grasp_obj / vlm_lawn_dual 相同 staging。"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.skills.plan_grasp_core import new_session_id, save_session, session_dir
from behavior_interface.skills.vlm_grasp_verify import _capture, _move_cam
from behavior_interface.skills.vlm_lawn_dual import (
    GROUND_XY,
    GROUND_Z,
    VIEWS,
    _cam_positions,
    _save_depth,
    _save_rgb,
    _save_seg,
)

LAWN_VIEW_DEFAULT = "left_upper"
# 与 vlm_lawn_dual grasp_obj 一致：物体中心 + 虚拟肩膀偏移
VIRTUAL_SHOULDER_OFFSET = np.array([0.0, -0.65, 0.35], dtype=np.float64)


def _json_safe(obj: Any) -> Any:
    """将 door_meta 中的 ndarray / tuple 转为可 JSON 序列化类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _trim_door_meta(geom: Dict[str, Any]) -> Dict[str, Any]:
    """仅保留 plan_eef open/close 需要的铰链字段，避免 AABB 等大对象写坏 session。"""
    keep = (
        "mode", "hinge_world", "axis_world",
        "handle_closed_world", "handle_open_world", "handle_world",
        "outward_normal_world", "closed_q", "open_q", "cur_q",
        "tangent_closed_world", "kind",
    )
    return {k: _json_safe(geom[k]) for k in keep if k in geom}


def _cleanup_lawn_object(world, model: str, ctx=None) -> None:
    """移除 vlm_{model} 悬浮物体。"""
    _cleanup_all_lawn_objects(world, ctx, keep_model=None)


def _purge_all_vlm_usd_roots(ctx=None) -> None:
    """扫描 scene_0 下所有 vlm_* USD 根节点并删除（含 remove_object 失败的残留）。"""
    import omnigibson as og

    stage = og.sim.stage
    root = stage.GetPrimAtPath("/World/scene_0")
    if not root.IsValid():
        return
    for child in root.GetChildren():
        name = child.GetName()
        if name.startswith("vlm_"):
            _purge_vlm_usd_prim(name, ctx)


def _purge_vlm_usd_prim(obj_name: str, ctx=None) -> None:
    """remove_object 后若 USD prim 仍残留，会触发 base_link 重复 initialize。"""
    import omnigibson as og
    import omnigibson.lazy as lazy

    stage = og.sim.stage
    for root in (f"/World/scene_0/{obj_name}", f"/World/{obj_name}"):
        prim = stage.GetPrimAtPath(root)
        if prim.IsValid():
            try:
                stage.RemovePrim(lazy.pxr.Sdf.Path(root))
                if ctx:
                    ctx.log(f"  [plan_eef/lawn] USD 清除 {root}")
            except Exception as e:
                if ctx:
                    ctx.log(f"  [plan_eef/lawn] USD 清除 {root} 失败: {e}")


def _cleanup_all_lawn_objects(world, ctx=None, keep_model: Optional[str] = None) -> None:
    """清除场景中所有 vlm_* 悬浮物体，避免上一 case 残留。"""
    _purge_all_vlm_usd_roots(ctx)
    keep_name = f"vlm_{keep_model}" if keep_model else None
    removed_models: List[str] = []
    for existing in list(world.env.scene.objects):
        name = getattr(existing, "name", None) or ""
        if not name.startswith("vlm_"):
            continue
        if keep_name and name == keep_name:
            continue
        try:
            world.env.scene.remove_object(existing)
            if ctx:
                ctx.log(f"  [plan_eef/lawn] 清除残留 {name}")
            _purge_vlm_usd_prim(name, ctx)
        except Exception as e:
            if ctx:
                ctx.log(f"  [plan_eef/lawn] 清除 {name} 失败: {e}")
            _purge_vlm_usd_prim(name, ctx)


def _safe_set_pose(obj, pos, quat) -> None:
    """设置位姿；避免 kinematic_only 触发 OG 版本不兼容的 clear_kinematic_only_cache。"""
    try:
        obj.set_position_orientation(position=pos, orientation=quat)
    except AttributeError:
        try:
            obj.root_link.set_position_orientation(position=pos, orientation=quat)
        except Exception:
            pass
    except Exception:
        pass


def _try_disable_dynamics(obj, ctx=None) -> None:
    """尽量关闭重力/动力学（勿设 kinematic_only，会与 set_position_orientation 冲突）。"""
    try:
        if hasattr(obj, "fixed_base"):
            obj.fixed_base = True
    except Exception:
        pass
    try:
        obj.keep_still()
    except Exception:
        pass
    try:
        for _, link in (getattr(obj, "links", {}) or {}).items():
            try:
                link.disable_gravity = True
            except Exception:
                pass
    except Exception:
        pass
    if ctx:
        ctx.log("  [plan_eef/lawn] 已尝试冻结物体动力学")


def pin_lawn_object(world, session: Dict[str, Any], ctx=None) -> Optional[Any]:
    """按 session 记录位姿强制复位 lawn 物体（每步调用）。"""
    import torch as th

    name = session.get("lawn_obj_name")
    pos = session.get("lawn_fixed_pos")
    quat = session.get("lawn_fixed_quat")
    if not name or not pos or not quat:
        return None
    obj = None
    for existing in world.env.scene.objects:
        if getattr(existing, "name", None) == name:
            obj = existing
            break
    if obj is None:
        return None
    pt = th.tensor(pos, dtype=th.float32)
    qt = th.tensor(quat, dtype=th.float32)
    _safe_set_pose(obj, pt, qt)
    closed_q = session.get("lawn_closed_q")
    if closed_q is not None and session.get("grasp_mode") == "hinge":
        for j in getattr(obj, "joints", {}).values():
            try:
                j.set_pos(float(closed_q))
            except Exception:
                pass
    return obj


def yield_pin_lawn(world, session: Dict[str, Any], n: int = 4):
    """yield 若干步并每步钉住物体。"""
    for _ in range(n):
        pin_lawn_object(world, session)
        yield world.empty_action()
        pin_lawn_object(world, session)


def _hold_gen_safe(gen, hold_pairs):
    """与 vlm_grasp_verify._hold_gen 相同，但用 _safe_set_pose 钉住物体。"""
    result = None
    try:
        while True:
            action = next(gen)
            for obj, pt, qt in hold_pairs:
                _safe_set_pose(obj, pt, qt)
            yield action
    except StopIteration as e:
        result = e.value
    return result


def _fast_hinge_meta_lawn(
    ctx,
    world,
    obj,
    joint,
    joint_dir: int,
    child_link_name: str,
    obj_pos_t,
    obj_quat_t,
) -> Generator:
    """草坪测试用轻量铰链几何（少 yield，避免 _sample_door_geometry 超时）。"""
    from behavior_interface.skills.eef import _rotate_around_axis
    from behavior_interface.skills.vlm_grasp_verify import _to_np

    link = obj.links.get(child_link_name)
    if link is None:
        ctx.set_result({"ok": False, "error": f"缺少 link {child_link_name}"})
        yield world.empty_action()
        return

    lower = float(joint.lower_limit)
    upper = float(joint.upper_limit)
    closed_q = lower if joint_dir == 1 else upper
    open_q = upper if joint_dir == 1 else lower

    hold = [(obj, obj_pos_t, obj_quat_t)]

    def _body():
        yield world.empty_action()
        try:
            joint.set_pos(closed_q)
        except Exception:
            pass
        for _ in range(3):
            yield world.empty_action()
        yield world.empty_action()

        lo, hi = link.aabb
        lo_a, hi_a = _to_np(lo).reshape(3), _to_np(hi).reshape(3)
        handle_closed = (lo_a + hi_a) / 2.0

        yield world.empty_action()
        try:
            j_pos, j_quat = joint.get_position_orientation()
            hinge_world = _to_np(j_pos).reshape(3)
            from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat
            Rj = _quat_to_mat(_to_np(j_quat).reshape(4))
            axis_local = np.array([1.0, 0.0, 0.0] if joint_dir == 1 else [-1.0, 0.0, 0.0])
            axis_world = Rj @ axis_local
            axis_world /= np.linalg.norm(axis_world) + 1e-9
        except Exception:
            hinge_world = lo_a.copy()
            axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        delta_q = open_q - closed_q
        handle_open = _rotate_around_axis(handle_closed, hinge_world, axis_world, delta_q)

        outward = handle_closed - hinge_world
        outward[2] = 0.0
        on = float(np.linalg.norm(outward))
        if on < 1e-6:
            outward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            outward /= on

        return {
            "lower": lower, "upper": upper,
            "closed_q": closed_q, "open_q": open_q, "cur_q": closed_q,
            "hinge_world": hinge_world,
            "axis_world": axis_world,
            "handle_closed_world": handle_closed,
            "handle_open_world": handle_open,
            "outward_normal_world": outward,
            "mode": "hinge",
        }

    geom = yield from _hold_gen_safe(_body(), hold)
    return geom


def yield_capture_lawn(
    ctx,
    world,
    gta,
    *,
    category: str,
    model: str,
    view: str = LAWN_VIEW_DEFAULT,
    arm: str = "right",
) -> Generator:
    """
    草坪悬浮物体 + left_upper 视角 capture，yield 仿真步，最后 ctx.set_result。
    返回的 session 已写入磁盘；物体保留在场景中供后续 plan（open 需 gta）。
    """
    # generator 第一次 next() 在 physics step 内执行，必须先 yield
    yield world.empty_action()
    yield from _yield_capture_lawn_impl(
        ctx, world, gta, category=category, model=model, view=view, arm=arm,
    )


def _yield_capture_lawn_impl(
    ctx,
    world,
    gta,
    *,
    category: str,
    model: str,
    view: str = LAWN_VIEW_DEFAULT,
    arm: str = "right",
) -> Generator:
    import torch as th
    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from behavior_interface.skills.eef import _list_openable_joints
    from behavior_interface.skills.vlm_grasp_verify import _hold_gen

    category = (category or "").strip()
    model = (model or "").strip()
    if not category or not model:
        ctx.set_result({"ok": False, "error": "capture_lawn 需要 category 与 model"})
        yield world.empty_action()
        return

    view_name = (view or LAWN_VIEW_DEFAULT).strip()
    if view_name not in VIEWS:
        ctx.set_result({"ok": False, "error": f"草坪视角仅支持 {list(VIEWS)}，收到 {view_name!r}"})
        yield world.empty_action()
        return

    w_img = int(getattr(gta, "image_width", 1280))
    h_img = int(getattr(gta, "image_height", 720))
    fl = float(getattr(gta, "focal_length", 17.0))
    ha = float(getattr(gta, "horizontal_aperture", 20.995))

    yield world.empty_action()
    _purge_all_vlm_usd_roots(ctx)
    for _ in range(3):
        yield world.empty_action()

    obj_pos_t = th.tensor([GROUND_XY[0], GROUND_XY[1], GROUND_Z], dtype=th.float32)
    obj_quat_t = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    # 每次 capture 使用唯一名，避免 remove 后 USD prim 残留导致重复 initialize
    obj_name = f"vlm_{model}_{uuid.uuid4().hex[:6]}"
    for existing in list(world.env.scene.objects):
        if getattr(existing, "name", None) == obj_name:
            try:
                world.env.scene.remove_object(existing)
                if ctx:
                    ctx.log(f"  [plan_eef/lawn] 清除同名残留 {obj_name}")
            except Exception as e:
                if ctx:
                    ctx.log(f"  [plan_eef/lawn] 清除 {obj_name} 失败: {e}")
    for stale in list(world.env.scene.objects):
        sn = getattr(stale, "name", None) or ""
        if sn.startswith(f"vlm_{model}"):
            _purge_vlm_usd_prim(sn, ctx)
    for _ in range(2):
        yield world.empty_action()

    obj = DatasetObject(
        name=obj_name, category=category, model=model,
        position=obj_pos_t.tolist(), orientation=obj_quat_t.tolist(),
    )
    world.env.scene.add_object(obj)
    _try_disable_dynamics(obj, ctx)
    for _ in range(4):
        _safe_set_pose(obj, obj_pos_t, obj_quat_t)
        yield world.empty_action()

    yield world.empty_action()
    try:
        _aabb_lo, _aabb_hi = obj.aabb
        _z_bottom_offset = float(_aabb_lo[2]) - GROUND_Z
        _z_target = GROUND_Z - _z_bottom_offset
    except Exception:
        _z_target = GROUND_Z
    obj_pos_t = th.tensor([GROUND_XY[0], GROUND_XY[1], _z_target], dtype=th.float32)
    ctx.log(f"  [plan_eef/lawn] {category}/{model} z={float(_z_target):.3f}")

    for _ in range(3):
        _safe_set_pose(obj, obj_pos_t, obj_quat_t)
        yield world.empty_action()

    yield world.empty_action()
    j_list = _list_openable_joints(obj)
    _scene_closed_q = 0.0
    hinge_joint = None
    geom_mode = "hinge"
    meta: Dict[str, Any] = {}
    if j_list:
        j, jdir, child = j_list[0]
        hinge_joint = j
        geom = yield from _fast_hinge_meta_lawn(
            ctx, world, obj, j, jdir, child, obj_pos_t, obj_quat_t,
        )
        if geom is None:
            ctx.set_result({"ok": False, "error": "铰链几何采样失败"})
            yield world.empty_action()
            return
        _scene_closed_q = float(geom["closed_q"])
        try:
            j.set_pos(_scene_closed_q)
        except Exception:
            pass
        meta = _trim_door_meta(geom)
        meta["mode"] = "hinge"
        grasp_mode = "hinge"
    else:
        grasp_mode = "grasp"
        geom_mode = "grasp"
        yield world.empty_action()
        try:
            lo, hi = obj.aabb
            lo_a = np.asarray(lo, dtype=np.float64)
            hi_a = np.asarray(hi, dtype=np.float64)
        except Exception:
            lo_a = hi_a = np.zeros(3, dtype=np.float64)
        center = (lo_a + hi_a) / 2.0
        meta = {
            "mode": "grasp",
            "center_world": center.tolist(),
            "handle_closed_world": center.tolist(),
            "outward_normal_world": [0.0, 1.0, 0.0],
            "axis_world": [0.0, 0.0, 1.0],
        }

    outward = np.asarray(meta["outward_normal_world"], dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    handle_pos = np.asarray(
        meta.get("handle_closed_world", meta.get("center_world", [0, 0, GROUND_Z])),
        dtype=np.float64,
    )
    focus = handle_pos.copy()
    cams = _cam_positions(focus, outward)

    tag = f"{category}_{model}"
    sid = new_session_id(tag)
    sdir = session_dir(sid)
    init_dir = os.path.join(sdir, "init")
    os.makedirs(init_dir, exist_ok=True)

    added_modalities: List[str] = []
    for mod in ("depth_linear", "seg_instance_id"):
        if mod not in gta.modalities:
            try:
                gta.add_modality(mod)
                added_modalities.append(mod)
            except Exception:
                pass

    yield world.empty_action()
    cam_pos_v, cam_quat_v = _move_cam(gta, cams[view_name], focus)
    for _ in range(2):
        _safe_set_pose(obj, obj_pos_t, obj_quat_t)
        if hinge_joint is not None:
            try:
                hinge_joint.set_pos(_scene_closed_q)
            except Exception:
                pass
        og.sim.render()
    obs, info = gta.get_obs()
    _save_rgb(obs, os.path.join(init_dir, "rgb.png"))
    depth = _save_depth(
        obs, os.path.join(init_dir, "depth.npy"), os.path.join(init_dir, "depth_vis.png"),
    )
    seg = _save_seg(
        obs, info, os.path.join(init_dir, "seg.npy"), os.path.join(init_dir, "seg_vis.png"),
    )

    image_path = os.path.join(sdir, f"{view_name}.png")
    for _ in range(2):
        _safe_set_pose(obj, obj_pos_t, obj_quat_t)
        og.sim.render()
    _capture(gta, image_path, ctx)
    # 与 init rgb 一致，供后续四图复用
    import shutil
    shutil.copy2(os.path.join(init_dir, "rgb.png"), image_path)

    yield world.empty_action()
    try:
        lo, hi = obj.aabb
        lo_a = np.asarray(lo, dtype=np.float64)
        hi_a = np.asarray(hi, dtype=np.float64)
    except Exception:
        lo_a = hi_a = np.zeros(3, dtype=np.float64)
    obj_center = (lo_a + hi_a) / 2.0
    virtual_shoulder = (obj_center + VIRTUAL_SHOULDER_OFFSET).tolist()

    cam_meta = {
        "view": view_name,
        "cam_pos": cam_pos_v.tolist(),
        "cam_quat_xyzw": cam_quat_v.tolist(),
        "look_at": focus.tolist(),
        "image_width": w_img,
        "image_height": h_img,
        "focal_length": fl,
        "horizontal_aperture": ha,
        "outward": outward.tolist(),
        "handle_geom": handle_pos.tolist(),
        "ground_xyz": [GROUND_XY[0], GROUND_XY[1], GROUND_Z],
        "category": category,
        "model": model,
        "geom_mode": geom_mode,
        "staging": "lawn",
    }
    with open(os.path.join(init_dir, f"camera_meta_{view_name}.json"), "w", encoding="utf-8") as f:
        json.dump(cam_meta, f, indent=2)

    session = {
        "session_id": sid,
        "object_name": tag,
        "category": category,
        "model": model,
        "lawn_obj_name": obj_name,
        "staging": "lawn",
        "grasp_mode": grasp_mode,
        "view": view_name,
        "available_views": list(VIEWS),
        "arm": arm,
        "image_width": w_img,
        "image_height": h_img,
        "focal_length": fl,
        "horizontal_aperture": ha,
        "handle_pos": handle_pos.tolist(),
        "outward": outward.tolist(),
        "door_meta": meta,
        "virtual_shoulder": virtual_shoulder,
        "object_info": {
            "input": tag,
            "resolved_name": obj_name,
            "aabb_min": lo_a.tolist(),
            "aabb_max": hi_a.tolist(),
            "center": obj_center.tolist(),
        },
        "image_path": image_path,
        "init_dir": init_dir,
        "has_depth": depth is not None,
        "has_seg": seg is not None,
        "lawn_fixed_pos": obj_pos_t.detach().cpu().tolist(),
        "lawn_fixed_quat": obj_quat_t.detach().cpu().tolist(),
        "lawn_closed_q": _scene_closed_q,
    }
    save_session(sid, session)

    from behavior_interface.skills.plan_eef_core import ALL_MODES, allocate_image_id
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    iid = allocate_image_id(sid, {
        "object_name": tag,
        "category": category,
        "model": model,
        "view": view_name,
        "image_path": image_path,
        "grasp_mode": grasp_mode,
        "staging": "lawn",
    })

    cap_result: Dict[str, Any] = {
        "ok": True,
        "step": "capture",
        "staging": "lawn",
        "category": category,
        "model": model,
        "image_id": iid,
        "session_id": sid,
        "object_name": tag,
        "view": view_name,
        "image_path": image_path,
        "image_width": w_img,
        "image_height": h_img,
        "virtual_shoulder": virtual_shoulder,
        "grasp_mode": grasp_mode,
        "available_modes": sorted(ALL_MODES),
    }
    cpos = np.asarray(cam_pos_v, dtype=np.float64)
    cquat = np.asarray(cam_quat_v, dtype=np.float64)
    for label, pt in (("object_center", obj_center), ("handle", handle_pos)):
        px = _world_to_pixel(cpos, cquat, np.asarray(pt), w_img, h_img, fl, ha)
        if px:
            cap_result[f"{label}_uv"] = {"u": px[0], "v": px[1]}

    ctx.set_result(cap_result)
    ctx.log(f"  [plan_eef/lawn] capture → {iid} {tag} view={view_name}")
    yield world.empty_action()
