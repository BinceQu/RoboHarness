"""
plan_grasp：交互式抓取规划（用户点击 / VLM 共用后端）

Step  capture      — object + view → 照片 + session_id
Step  resolve_3d    — 2D 点 → 3D 击中点 + 仿真红球
Step  resolve_grasp — 由上次 3D 点 → EEF pose + 仿真夹爪
Step  resolve        — 兼容：一步完成 3d + grasp

视角（view）四选一：
  left_upper_front, left_upper_back, right_upper_front, right_upper_back
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
    load_session,
    new_session_id,
    normalize_click,
    normalize_view,
    resolve_grasp_from_session,
    resolve_pixel_to_3d,
    resolve_pixel_to_grasp,
    run_vlm_on_session,
    save_session,
    session_dir,
)


def _resolve_point(ctx, session, point_source, u, v):
    """解析 2D 点：用户点击或 VLM。"""
    ps = (point_source or "user").lower().strip()
    vlm_info = None
    if ps == "vlm":
        ctx.log(f"  [plan_grasp] VLM session={session['session_id']}")
        u_i, v_i, vlm_info = run_vlm_on_session(session)
        ctx.log(f"  [plan_grasp] VLM pixel=({u_i},{v_i})")
    elif ps == "user":
        if u is None or v is None:
            return None, None, "point_source=user 时需要 Qwen3-VL 0..1000 的 u, v"
        w_s = int(session["image_width"])
        h_s = int(session["image_height"])
        u_i, v_i = normalize_click(u, v, w_s, h_s)
        ctx.log(f"  [plan_grasp] user click=({u_i},{v_i})")
    else:
        return None, None, f"未知 point_source={point_source!r}"
    return (u_i, v_i, vlm_info), ps, None


@register_skill(
    "plan_grasp",
    description=(
        "交互抓取规划。"
        "step=capture + view(四视角)；"
        "step=resolve_3d → 3D 点+红球；"
        "step=resolve_grasp → 夹爪；"
        "step=resolve 兼容一步完成。"
    ),
)
def plan_grasp(
    ctx,
    step: str = "capture",
    object_name: str = "",
    session_id: str = "",
    point_source: str = "user",
    u: float = None,
    v: float = None,
    view: str = PLAN_GRASP_VIEW,
    arm: str = "right",
    cam_dist: float = None,
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

    # ══════════════════════════════════════════════════════════════════════════
    # capture
    # ══════════════════════════════════════════════════════════════════════════
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

        ctx.log(f"  [plan_grasp] capture object={obj.name} view={view_name}")

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
            ctx.log(f"  [plan_grasp] 无铰链 → grasp 模式")
            if aabb is None:
                ctx.set_result({"ok": False, "error": f"{obj.name} 无 AABB，无法 grasp 模式拍摄"})
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
            handle_pos, outward, float(cam_dist), door_meta=meta if grasp_mode == "hinge" else None,
        )
        if view_name not in cam_positions:
            ctx.set_result({"ok": False, "error": f"视角 {view_name} 无相机位置"})
            yield world.empty_action()
            return

        sid = new_session_id(object_name.strip())
        sdir = session_dir(sid)
        init_dir = os.path.join(sdir, "init")
        os.makedirs(init_dir, exist_ok=True)

        for mod in ("depth_linear", "seg_instance_id"):
            if mod not in gta.modalities:
                try:
                    gta.add_modality(mod)
                except Exception as e:
                    ctx.log(f"  [plan_grasp] 添加 {mod} 失败: {e}")

        cam_pos_v, cam_quat_v = _move_cam(gta, cam_positions[view_name], focus)
        for _ in range(12):
            og.sim.render()
        obs, info = gta.get_obs()
        _save_rgb(obs, os.path.join(init_dir, "rgb.png"))
        depth = _save_depth(
            obs,
            os.path.join(init_dir, "depth.npy"),
            os.path.join(init_dir, "depth_vis.png"),
        )
        seg = _save_seg(
            obs, info,
            os.path.join(init_dir, "seg.npy"),
            os.path.join(init_dir, "seg_vis.png"),
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
        object_info = {
            "input": object_name.strip(),
            "resolved_name": getattr(obj, "name", object_name),
            "aabb_min": lo.tolist(),
            "aabb_max": hi.tolist(),
            "center": ((lo + hi) / 2.0).tolist(),
        }

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
            "object_info": object_info,
            "image_path": image_path,
            "init_dir": init_dir,
            "has_depth": depth is not None,
            "has_seg": seg is not None,
        }
        save_session(sid, session)

        ctx.set_result({
            "ok": True,
            "step": "capture",
            "session_id": sid,
            "object_name": object_name.strip(),
            "grasp_mode": grasp_mode,
            "view": view_name,
            "available_views": list(PLAN_GRASP_VIEWS),
            "image_width": w_img,
            "image_height": h_img,
            "image_path": image_path,
            "image_url": f"/api/plan_grasp/session/{sid}/image",
            "handle_pos": handle_pos.tolist(),
            "has_depth": depth is not None,
        })
        ctx.log(f"  [plan_grasp] capture done view={view_name} session={sid}")
        yield world.empty_action()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # resolve_3d — 2D → 3D + 红球
    # ══════════════════════════════════════════════════════════════════════════
    if step == "resolve_3d":
        if not session_id.strip():
            ctx.set_result({"ok": False, "error": "resolve_3d 需要 session_id"})
            yield world.empty_action()
            return
        try:
            session = load_session(session_id.strip())
        except FileNotFoundError as e:
            ctx.set_result({"ok": False, "error": str(e)})
            yield world.empty_action()
            return

        pt_result, ps, err = _resolve_point(ctx, session, point_source, u, v)
        if err:
            ctx.set_result({"ok": False, "error": err})
            yield world.empty_action()
            return
        u_i, v_i, vlm_info = pt_result
        yield world.empty_action()

        payload = resolve_pixel_to_3d(
            gta=gta, session=session, u=u_i, v=v_i, ctx=ctx,
        )
        if not payload.get("ok"):
            ctx.set_result(payload)
            yield world.empty_action()
            return

        payload["point_source"] = ps
        payload["marked_image_url"] = f"/api/plan_grasp/session/{session_id}/marked"
        if vlm_info:
            payload["vlm"] = vlm_info
            with open(os.path.join(session_dir(session_id), "vlm_result.json"), "w") as f:
                json.dump(vlm_info, f, indent=2)

        session["last_3d"] = {
            "point_source": ps,
            "pixel": payload["pixel"],
            "hit_world": payload["hit_world"],
            "hit_method": payload.get("hit_method"),
            "object_pcd_path": payload.get("object_pcd_path"),
        }
        if payload.get("object_pcd_path"):
            session["object_pcd_path"] = payload["object_pcd_path"]
        save_session(session_id, session)
        ctx.set_result(payload)
        yield world.empty_action()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # resolve_grasp — 3D → EEF + 夹爪
    # ══════════════════════════════════════════════════════════════════════════
    if step == "resolve_grasp":
        if not session_id.strip():
            ctx.set_result({"ok": False, "error": "resolve_grasp 需要 session_id"})
            yield world.empty_action()
            return
        try:
            session = load_session(session_id.strip())
        except FileNotFoundError as e:
            ctx.set_result({"ok": False, "error": str(e)})
            yield world.empty_action()
            return

        yield world.empty_action()
        payload = resolve_grasp_from_session(session, gta=gta, world=world, ctx=ctx)
        if not payload.get("ok"):
            ctx.set_result(payload)
            yield world.empty_action()
            return

        session["last_grasp"] = {
            "grasp": payload["grasp"],
            "hit_world": payload.get("hit_world"),
        }
        save_session(session_id, session)
        ctx.set_result(payload)
        yield world.empty_action()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # resolve — 兼容一步完成
    # ══════════════════════════════════════════════════════════════════════════
    if step == "resolve":
        if not session_id.strip():
            ctx.set_result({"ok": False, "error": "resolve 需要 session_id"})
            yield world.empty_action()
            return
        try:
            session = load_session(session_id.strip())
        except FileNotFoundError as e:
            ctx.set_result({"ok": False, "error": str(e)})
            yield world.empty_action()
            return

        pt_result, ps, err = _resolve_point(ctx, session, point_source, u, v)
        if err:
            ctx.set_result({"ok": False, "error": err})
            yield world.empty_action()
            return
        u_i, v_i, vlm_info = pt_result
        yield world.empty_action()

        payload = resolve_pixel_to_grasp(
            gta=gta, world=world, session=session, u=u_i, v=v_i, ctx=ctx,
        )
        if not payload.get("ok"):
            ctx.set_result(payload)
            yield world.empty_action()
            return

        payload["point_source"] = ps
        payload["marked_image_url"] = f"/api/plan_grasp/session/{session_id}/marked"
        if vlm_info:
            payload["vlm"] = vlm_info

        session["last_3d"] = {
            "point_source": ps,
            "pixel": payload["pixel"],
            "hit_world": payload["hit_world"],
        }
        session["last_grasp"] = {"grasp": payload["grasp"]}
        save_session(session_id, session)
        ctx.set_result(payload)
        yield world.empty_action()
        return

    ctx.set_result({
        "ok": False,
        "error": f"未知 step={step!r}，支持 capture / resolve_3d / resolve_grasp / resolve",
    })
    yield world.empty_action()
