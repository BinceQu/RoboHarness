"""
plan_eef：VLM/用户 在已编号图像上指定点与 mode，返回 EEF pose + next 目标 + 渲染图。

Step:
  capture      — 拍摄物体视角，注册 image_id（img_0001 …）
  plan         — image_id + point(u,v) + mode → eef 规划
  list_images  — 列出已注册图像供 VLM 选择
"""

from __future__ import annotations

import json
import os
from typing import Generator, Optional

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.grasp import _aabb_of, _resolve_object_handle
from behavior_interface.skills.eef import _sample_open_candidates
from behavior_interface.skills.vlm_grasp_verify import _capture, _move_cam
from behavior_interface.skills.vlm_lawn_dual import _save_depth, _save_rgb, _save_seg
from behavior_interface.skills.plan_grasp_core import (
    PLAN_GRASP_VIEW,
    PLAN_GRASP_VIEWS,
    compute_cam_positions,
    new_session_id,
    normalize_click,
    normalize_view,
    save_session,
    session_dir,
)
from behavior_interface.skills.plan_grasp_core import load_session
from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel
from behavior_interface.skills.plan_eef_core import (
    ALL_MODES,
    allocate_image_id,
    list_registered_images,
    lookup_image,
    plan_eef_from_session,
    _load_cam_meta,
)


def _parse_point(u, v, image_id: str) -> tuple:
    """Resolve public u/v after the registry boundary converted them to pixels."""
    if u is None or v is None:
        raise ValueError("plan 需要 Qwen3-VL 0..1000 的 u,v")
    entry = lookup_image(image_id)
    session = load_session(entry["session_id"])
    w_s = int(session["image_width"])
    h_s = int(session["image_height"])
    return normalize_click(u, v, w_s, h_s), session


@register_skill(
    "plan_eef",
    description=(
        "EEF 规划：capture 注册 image_id；plan(image_id, u, v, mode) 返回 "
        "eef_pose、next_eef_move(世界坐标)、render 图；list_images 列出已拍图像。"
    ),
)
def plan_eef(
    ctx,
    step: str = "capture",
    object_name: str = "",
    image_id: str = "",
    u: float = None,
    v: float = None,
    mode: str = "grasp_point",
    view: str = PLAN_GRASP_VIEW,
    arm: str = "right",
    cam_dist: float = None,
    staging: str = "",
    category: str = "",
    model: str = "",
    cleanup_lawn: bool = True,
) -> Generator:
    import omnigibson as og

    world = ctx.world
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        ctx.set_result({"ok": False, "error": "gta_view 不可用"})
        yield world.empty_action()
        return

    w_img = int(getattr(gta, "image_width", 1280))
    h_img = int(getattr(gta, "image_height", 720))
    fl = float(getattr(gta, "focal_length", 17.0))
    ha = float(getattr(gta, "horizontal_aperture", 20.995))
    step = (step or "capture").lower().strip()

    # ── list_images ──
    if step == "list_images":
        images = list_registered_images()
        ctx.set_result({"ok": True, "step": "list_images", "images": images, "n": len(images)})
        yield world.empty_action()
        return

    # ── capture_lawn：草坪悬浮 + left_upper（同 nova_vlm_grasp_obj）──
    if step == "capture_lawn" or ((staging or "").lower() == "lawn" and step == "capture"):
        from behavior_interface.skills.plan_eef_lawn_capture import (
            LAWN_VIEW_DEFAULT,
            yield_capture_lawn,
        )

        cat = category.strip()
        mdl = model.strip()
        if not cat and object_name.strip():
            parts = object_name.strip().split("_", 1)
            cat, mdl = parts[0], parts[1] if len(parts) > 1 else ""
        if not cat or not mdl:
            ctx.set_result({"ok": False, "error": "capture_lawn 需要 category 与 model"})
            yield world.empty_action()
            return
        lawn_view = (view or LAWN_VIEW_DEFAULT).strip()
        if lawn_view in ("left_upper_front",):
            lawn_view = "left_upper"
        yield from yield_capture_lawn(
            ctx, world, gta, category=cat, model=mdl, view=lawn_view, arm=arm,
        )
        return

    # ── capture（场景内物体 + image_id）──
    if step == "capture":
        if not object_name.strip():
            ctx.set_result({"ok": False, "error": "capture 需要 object_name"})
            yield world.empty_action()
            return
        try:
            view_name = normalize_view(view)
        except ValueError as e:
            ctx.set_result({"ok": False, "error": str(e)})
            yield world.empty_action()
            return

        obj = _resolve_object_handle(world, object_name.strip())
        if obj is None:
            ctx.set_result({"ok": False, "error": f"未找到物体: {object_name}"})
            yield world.empty_action()
            return

        ctx.log(f"  [plan_eef] capture object={obj.name} view={view_name}")

        aabb = _aabb_of(obj)
        raw = yield from _sample_open_candidates(ctx, obj, opening=True, arm=arm)
        grasp_mode = "hinge"
        meta: dict = {}
        if raw:
            open_cands = [c for c in raw if c.get("meta", {}).get("kind") == "door_arc"]
            best_cand = open_cands[0] if open_cands else raw[0]
            meta = best_cand.get("meta", {})
            handle_pos = np.asarray(
                meta.get("handle_closed_world", best_cand["eef_target"]["pos"]),
                dtype=np.float64,
            )
            outward = np.asarray(
                meta.get("outward_normal_world", [0.0, -1.0, 0.0]),
                dtype=np.float64,
            )
        else:
            grasp_mode = "grasp"
            if aabb is None:
                ctx.set_result({"ok": False, "error": f"{obj.name} 无 AABB"})
                yield world.empty_action()
                return
            lo, hi = aabb
            handle_pos = ((lo + hi) / 2.0).astype(np.float64)
            outward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            meta = {
                "mode": "grasp",
                "center_world": handle_pos.tolist(),
                "handle_closed_world": handle_pos.tolist(),
            }

        outward /= np.linalg.norm(outward) + 1e-9
        if cam_dist is None:
            if aabb is not None:
                lo, hi = aabb
                diag = float(np.linalg.norm(hi - lo))
                cam_dist = float(np.clip(diag * 0.9 if grasp_mode == "grasp" else diag * 0.7, 0.4, 1.8))
            else:
                cam_dist = 0.8

        cam_positions, focus = compute_cam_positions(
            handle_pos, outward, float(cam_dist),
            door_meta=meta if grasp_mode == "hinge" else None,
        )

        sid = new_session_id(object_name.strip())
        sdir = session_dir(sid)
        init_dir = os.path.join(sdir, "init")
        os.makedirs(init_dir, exist_ok=True)

        for mod in ("depth_linear", "seg_instance_id"):
            if mod not in gta.modalities:
                try:
                    gta.add_modality(mod)
                except Exception:
                    pass

        cam_pos_v, cam_quat_v = _move_cam(gta, cam_positions[view_name], focus)
        for _ in range(12):
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
        for _ in range(4):
            og.sim.render()
        _capture(gta, image_path, ctx)

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
            "object_name": object_name.strip(),
        }
        with open(os.path.join(init_dir, f"camera_meta_{view_name}.json"), "w") as f:
            json.dump(cam_meta, f, indent=2)

        lo, hi = aabb if aabb is not None else (np.zeros(3), np.zeros(3))
        session = {
            "session_id": sid,
            "object_name": object_name.strip(),
            "grasp_mode": grasp_mode,
            "view": view_name,
            "available_views": list(PLAN_GRASP_VIEWS),
            "arm": arm,
            "image_width": w_img,
            "image_height": h_img,
            "focal_length": fl,
            "horizontal_aperture": ha,
            "handle_pos": handle_pos.tolist(),
            "outward": outward.tolist(),
            "door_meta": meta,
            "object_info": {
                "input": object_name.strip(),
                "resolved_name": getattr(obj, "name", object_name),
                "aabb_min": lo.tolist(),
                "aabb_max": hi.tolist(),
                "center": ((lo + hi) / 2.0).tolist(),
            },
            "image_path": image_path,
            "init_dir": init_dir,
            "has_depth": depth is not None,
            "has_seg": seg is not None,
        }
        save_session(sid, session)

        iid = allocate_image_id(sid, {
            "object_name": object_name.strip(),
            "view": view_name,
            "image_path": image_path,
            "grasp_mode": grasp_mode,
        })

        cap_result = {
            "ok": True,
            "step": "capture",
            "image_id": iid,
            "session_id": sid,
            "object_name": object_name.strip(),
            "view": view_name,
            "image_path": image_path,
            "image_width": w_img,
            "image_height": h_img,
            "available_modes": sorted(ALL_MODES),
        }
        try:
            cam_m = _load_cam_meta(session)
            cpos = np.asarray(cam_m["cam_pos"], dtype=np.float64)
            cquat = np.asarray(cam_m["cam_quat_xyzw"], dtype=np.float64)
            fl_c = float(cam_m.get("focal_length", fl))
            ha_c = float(cam_m.get("horizontal_aperture", ha))
            for label, pt in (
                ("object_center", ((lo + hi) / 2.0)),
                ("handle", handle_pos),
            ):
                px = _world_to_pixel(cpos, cquat, np.asarray(pt), w_img, h_img, fl_c, ha_c)
                if px:
                    cap_result[f"{label}_uv"] = {"u": px[0], "v": px[1]}
        except Exception:
            pass
        ctx.set_result(cap_result)
        ctx.log(f"  [plan_eef] capture → {iid}")
        yield world.empty_action()
        return

    # ── plan ──
    if step == "plan":
        if not image_id.strip():
            ctx.set_result({"ok": False, "error": "plan 需要 image_id"})
            yield world.empty_action()
            return
        try:
            (u_i, v_i), session = _parse_point(u, v, image_id.strip())
        except (KeyError, ValueError, FileNotFoundError) as e:
            ctx.set_result({"ok": False, "error": str(e)})
            yield world.empty_action()
            return

        if session.get("staging") == "lawn":
            from behavior_interface.skills.plan_eef_lawn_capture import yield_pin_lawn
            yield from yield_pin_lawn(world, session, n=1)

        yield world.empty_action()
        payload = plan_eef_from_session(
            session, u_i, v_i, mode,
            gta=gta, world=world, ctx=ctx,
            fast_plan=(session.get("staging") == "lawn"),
        )
        if not payload.get("ok"):
            ctx.set_result(payload)
            yield world.empty_action()
            return

        payload["image_id"] = image_id.strip()
        payload["image_path"] = session.get("image_path")
        save_session(session["session_id"], session)
        ctx.set_result(payload)
        yield world.empty_action()
        if session.get("staging") == "lawn" and cleanup_lawn:
            from behavior_interface.skills.plan_eef_lawn_capture import _purge_all_vlm_usd_roots
            yield world.empty_action()
            _purge_all_vlm_usd_roots(ctx)
            for _ in range(4):
                og.sim.render()
                yield world.empty_action()
        return

    ctx.set_result({
        "ok": False,
        "error": f"未知 step={step!r}，支持 capture / capture_lawn / plan / list_images",
    })
    yield world.empty_action()
