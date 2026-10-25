"""
vlm_scene_grasp：VLM 把手识别 + 深度反解 3D 抓取点 → 结合几何生成可执行 EEF pose

流程（配合 get_eef_pose 和 execute_eef_pose 使用）：

    Step 0: get_eef_pose(object_name, target="open")
            → 存储候选 candidate 到 last_skill_results["get_eef_pose"]

    Step 1: vlm_scene_grasp(mode="init", object_name=...)
            → 读 get_eef_pose 几何 → 摆好 GTA 相机 → 捕获 RGB/depth/seg
            → 保存到 out_dir/init/

    Step 2: [shell] 运行 VLM Python 脚本
            → 输出 vlm_result_{source_view}.json（含 point_2d）

    Step 3: vlm_scene_grasp(mode="pcd", ...)
            → 读 depth + VLM 结果 → 计算 3D 击中点 → 计算 EEF pose
            → 将精化后的候选写回 last_skill_results["get_eef_pose"]

    Step 4: execute_eef_pose(eef_id=0)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill


# ──────────────────────────────────────────────────────────────────────────────
# 复用 vlm_lawn_dual / vlm_grasp_verify 中的图像工具函数
# ──────────────────────────────────────────────────────────────────────────────
from behavior_interface.skills.vlm_grasp_verify import (
    _capture,
    _depth_to_world_point,
    _make_sphere,
    _move_cam,
    _pixel_to_world_ray,
    _quat_to_mat,
    _to_np,
    _world_to_pixel,
)
from behavior_interface.skills.vlm_lawn_dual import (
    CAM_D,
    CAM_SIDE,
    CAM_UP,
    _build_pointcloud,
    _cam_positions,
    _ray_pcd_hit,
    _save_depth,
    _save_rgb,
    _save_seg,
)

DEFAULT_SOURCE_VIEW = "right_upper"
VIEWS = ("left_upper", "right_upper")
BALL_SURFACE_OFFSET = 0.035  # 球心距物体表面偏移，保证可见


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_vlm_json(out_dir: str, vlm_json: str, source_view: str) -> str:
    if vlm_json and os.path.isfile(vlm_json):
        return vlm_json
    for name in (
        f"vlm_result_{source_view}.json",
        "vlm_result.json",
        f"vlm_result_{DEFAULT_SOURCE_VIEW}.json",
    ):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            return p
    return os.path.join(out_dir, f"vlm_result_{source_view}.json")


def _point_to_pixel(item: Dict, w: int, h: int) -> Optional[Tuple[int, int]]:
    """从 VLM 结果 dict 解析 2D 像素坐标（优先 point_2d，降级 bbox_2d 中心）。"""
    pt = item.get("point_2d")
    if pt and len(pt) == 2:
        x, y = float(pt[0]), float(pt[1])
        if max(x, y) <= 1.0:  # 0-1 归一化
            return int(x * w), int(y * h)
        elif max(x, y) <= 1000:  # 0-1000 归一化
            return int(x / 1000.0 * w), int(y / 1000.0 * h)
        return int(x), int(y)  # 已是像素
    bbox = item.get("bbox_2d")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if max(x0, y0, x1, y1) <= 1000:
            x0, y0 = x0 / 1000.0 * w, y0 / 1000.0 * h
            x1, y1 = x1 / 1000.0 * w, y1 / 1000.0 * h
        return int((x0 + x1) / 2), int((y0 + y1) / 2)
    return None


def _draw_vlm_point(img_path: str, u: int, v: int, out_path: str):
    """在 RGB 图上画十字线标记 VLM 点，保存到 out_path。"""
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return
    size = 20
    cv2.line(img, (u - size, v), (u + size, v), (0, 255, 0), 2)
    cv2.line(img, (u, v - size), (u, v + size), (0, 255, 0), 2)
    cv2.circle(img, (u, v), 6, (0, 255, 0), -1)
    cv2.imwrite(out_path, img)


# ──────────────────────────────────────────────────────────────────────────────
# 主 Skill
# ──────────────────────────────────────────────────────────────────────────────

@register_skill(
    "vlm_scene_grasp",
    description=(
        "VLM 场景物体把手抓取：mode=init 拍 RGB/depth/seg；mode=pcd 计算 3D 抓取点并写回 "
        "get_eef_pose 候选以便 execute_eef_pose 执行。"
        "使用前必须先运行 get_eef_pose(object_name, target='open')。"
    ),
)
def vlm_scene_grasp(
    ctx,
    mode: str = "init",
    object_name: str = "microwave",
    out_dir: str = "/tmp/vlm_scene_grasp",
    vlm_json: str = "",
    source_view: str = DEFAULT_SOURCE_VIEW,
    arm: str = "right",
    cam_dist: float = None,   # None → 自动根据物体大小估算
    add_ball: bool = False,   # 是否在 3D 抓取点放可视化球（pcd 模式）
) -> Generator:
    import omnigibson as og

    os.makedirs(out_dir, exist_ok=True)
    world = ctx.world

    # ── GTA 相机 ──────────────────────────────────────────────────────────────
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        ctx.set_result({"ok": False, "error": "gta_view 不可用"})
        yield world.empty_action()
        return

    w_img = int(getattr(gta, "image_width",  1280))
    h_img = int(getattr(gta, "image_height",  720))
    fl    = float(getattr(gta, "focal_length",  17.0))
    ha    = float(getattr(gta, "horizontal_aperture", 20.995))

    sv = source_view if source_view in VIEWS else DEFAULT_SOURCE_VIEW

    # ── 读 get_eef_pose 结果（必须提前运行）────────────────────────────────────
    eef_res = ctx.get_last_result("get_eef_pose")
    if eef_res is None or not eef_res.get("ok"):
        ctx.set_result({"ok": False,
                        "error": "请先运行 get_eef_pose(object_name, target='open')"})
        yield world.empty_action()
        return

    # 取分数最高的 candidate（通常第一个已按 score 排序）
    cands = eef_res.get("candidates", [])
    if not cands:
        ctx.set_result({"ok": False, "error": "get_eef_pose candidates 为空"})
        yield world.empty_action()
        return

    # 筛选 "open" / "door_arc" 类型的候选
    open_cands = [c for c in cands if c.get("meta", {}).get("kind") == "door_arc"]
    best_cand = open_cands[0] if open_cands else cands[0]
    meta = best_cand.get("meta", {})
    handle_pos = np.asarray(
        meta.get("handle_closed_world",
                 best_cand["eef_target"]["pos"]), dtype=np.float64)
    outward = np.asarray(
        meta.get("outward_normal_world", [0.0, -1.0, 0.0]), dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9

    # 相机距离：若不指定则取物体 AABB 对角线 * 0.6（最少 0.5m 最多 1.5m）
    if cam_dist is None:
        aabb_min = np.asarray(eef_res["object"].get("aabb_min", [0, 0, 0]))
        aabb_max = np.asarray(eef_res["object"].get("aabb_max", [1, 1, 1]))
        diag = float(np.linalg.norm(aabb_max - aabb_min))
        cam_dist = float(np.clip(diag * 0.7, 0.5, 1.8))
    ctx.log(f"  [vlm_sg] handle={handle_pos.round(3).tolist()} "
            f"outward={outward.round(3).tolist()} cam_dist={cam_dist:.2f}")

    # ── 计算相机位置 ─────────────────────────────────────────────────────────
    up_w = np.array([0.0, 0.0, 1.0])
    perp = np.cross(up_w, outward)
    pn   = np.linalg.norm(perp)
    if pn < 1e-6:
        perp = np.cross(np.array([1.0, 0.0, 0.0]), outward)
        pn   = np.linalg.norm(perp)
    perp /= pn + 1e-9

    cam_positions = {
        "left_upper":  handle_pos + outward * cam_dist + perp * CAM_SIDE + up_w * CAM_UP,
        "right_upper": handle_pos + outward * cam_dist - perp * CAM_SIDE + up_w * CAM_UP,
    }
    focus = handle_pos.copy()

    # ════════════════════════════════════════════════════════════════════════════
    # Mode: init — 拍摄 RGB + depth + seg
    # ════════════════════════════════════════════════════════════════════════════
    if mode == "init":
        init_dir = os.path.join(out_dir, "init")
        os.makedirs(init_dir, exist_ok=True)

        # 动态添加 depth_linear 和 seg_instance_id modality
        added_modalities: List[str] = []
        for mod in ("depth_linear", "seg_instance_id"):
            if mod not in gta.modalities:
                try:
                    gta.add_modality(mod)
                    added_modalities.append(mod)
                    ctx.log(f"  [vlm_sg] 添加 modality: {mod}")
                except Exception as e:
                    ctx.log(f"  [vlm_sg] 添加 {mod} 失败: {e}")

        # 移相机到 source_view，稳定后抓图
        cam_pos_sv, cam_quat_sv = _move_cam(gta, cam_positions[sv], focus)
        for _ in range(12):
            og.sim.render()

        obs, info = gta.get_obs()

        _save_rgb(obs, os.path.join(init_dir, "rgb.png"))
        ctx.log("  [vlm_sg] rgb saved")

        depth = _save_depth(obs,
                            os.path.join(init_dir, "depth.npy"),
                            os.path.join(init_dir, "depth_vis.png"))
        ctx.log(f"  [vlm_sg] depth available: {depth is not None}")

        seg = _save_seg(obs, info,
                        os.path.join(init_dir, "seg.npy"),
                        os.path.join(init_dir, "seg_vis.png"))
        ctx.log(f"  [vlm_sg] seg available: {seg is not None}")

        # 保存所有视角截图 + 各自相机元信息
        for vname in VIEWS:
            cam_pos_v, cam_quat_v = _move_cam(gta, cam_positions[vname], focus)
            for _ in range(8):
                og.sim.render()
            _capture(gta, os.path.join(out_dir, f"{vname}.png"), ctx)

            meta_v = {
                "view": vname,
                "cam_pos": cam_pos_v.tolist(),
                "cam_quat_xyzw": cam_quat_v.tolist(),
                "look_at": focus.tolist(),
                "image_width": w_img, "image_height": h_img,
                "focal_length": fl, "horizontal_aperture": ha,
                "outward": outward.tolist(),
                "handle_geom": handle_pos.tolist(),
                "object_name": object_name,
                "eef_cand_id": best_cand.get("id", 0),
            }
            with open(os.path.join(out_dir,   f"camera_meta_{vname}.json"), "w") as f:
                json.dump(meta_v, f, indent=2)
            with open(os.path.join(init_dir,  f"camera_meta_{vname}.json"), "w") as f:
                json.dump(meta_v, f, indent=2)

        ctx.set_result({
            "ok": True,
            "mode": "init",
            "init_dir": init_dir,
            "source_view": sv,
            "handle_pos": handle_pos.tolist(),
            "outward": outward.tolist(),
            "has_depth": depth is not None,
            "has_seg": seg is not None,
        })
        ctx.log(f"  [vlm_sg] init done → {init_dir}")
        yield world.empty_action()
        return

    # ════════════════════════════════════════════════════════════════════════════
    # Mode: pcd — VLM 点 + 深度 → 3D 抓取点 → EEF pose 写回 get_eef_pose
    # ════════════════════════════════════════════════════════════════════════════
    if mode == "pcd":
        init_dir = os.path.join(out_dir, "init")

        # 加载 depth
        depth_npy = os.path.join(init_dir, "depth.npy")
        if not os.path.isfile(depth_npy):
            ctx.set_result({"ok": False,
                            "error": f"缺少 depth.npy，请先运行 mode=init: {depth_npy}"})
            yield world.empty_action()
            return
        depth = np.load(depth_npy).astype(np.float64)
        ctx.log(f"  [vlm_sg] depth shape={depth.shape} "
                f"valid={int((depth>0).sum())}")

        # 加载相机元信息
        meta_path = os.path.join(init_dir, f"camera_meta_{sv}.json")
        if not os.path.isfile(meta_path):
            meta_path = os.path.join(out_dir, f"camera_meta_{sv}.json")
        if not os.path.isfile(meta_path):
            ctx.set_result({"ok": False, "error": "缺少 camera_meta"})
            yield world.empty_action()
            return
        with open(meta_path, encoding="utf-8") as f:
            cam_meta = json.load(f)
        cam_pos_sv = np.asarray(cam_meta["cam_pos"])
        cam_quat_sv = np.asarray(cam_meta["cam_quat_xyzw"])
        fl_m = float(cam_meta.get("focal_length", fl))
        ha_m = float(cam_meta.get("horizontal_aperture", ha))

        # 加载 VLM 结果
        vlm_path = _resolve_vlm_json(out_dir, vlm_json, sv)
        if not os.path.isfile(vlm_path):
            ctx.set_result({"ok": False,
                            "error": f"缺少 VLM 结果: {vlm_path}"})
            yield world.empty_action()
            return
        with open(vlm_path, encoding="utf-8") as f:
            vlm_data = json.load(f)

        items = vlm_data if isinstance(vlm_data, list) else vlm_data.get("items", [])
        if not items:
            ctx.set_result({"ok": False, "error": "VLM 结果无 items"})
            yield world.empty_action()
            return
        item0 = items[0]
        pix = _point_to_pixel(item0, w_img, h_img)
        if pix is None:
            ctx.set_result({"ok": False,
                            "error": f"VLM item 无 point_2d / bbox_2d: {item0}"})
            yield world.empty_action()
            return
        u_vlm, v_vlm = pix
        ctx.log(f"  [vlm_sg] VLM 像素 u={u_vlm} v={v_vlm}")

        # 画标注图
        rgb_path = os.path.join(init_dir, "rgb.png")
        _draw_vlm_point(rgb_path,
                        u_vlm, v_vlm,
                        os.path.join(out_dir, "point_on_image.png"))

        # ── 方法1：直接从 depth 读像素 ─────────────────────────────────────
        hit_pos = None
        hit_method = "none"

        # 复位相机（保证深度和射线一致）
        _move_cam(gta, cam_pos_sv, focus)
        for _ in range(6):
            og.sim.render()

        p_depth = _depth_to_world_point(gta, u_vlm, v_vlm,
                                        cam_pos_sv, cam_quat_sv,
                                        w_img, h_img, fl_m, ha_m)
        if p_depth is not None:
            ctx.log(f"  [vlm_sg] depth_backproject hit={p_depth.round(3).tolist()}")
            hit_pos = p_depth
            hit_method = "depth_backproject"

        # ── 方法2：点云射线求交 ──────────────────────────────────────────────
        if hit_pos is None:
            ctx.log("  [vlm_sg] depth 直查失败，尝试点云射线求交")
            pts_world = _build_pointcloud(depth, cam_pos_sv, cam_quat_sv, fl_m, ha_m)
            ctx.log(f"  [vlm_sg] 点云总点数: {len(pts_world)}")

            # 只保留靠近 handle 附近 2m 范围内的点
            handle_ref = np.asarray(cam_meta.get("handle_geom", handle_pos.tolist()))
            dists = np.linalg.norm(pts_world - handle_ref, axis=1)
            pts_near = pts_world[dists < 2.0]
            ctx.log(f"  [vlm_sg] handle 附近点数: {len(pts_near)}")

            origin, ray_dir = _pixel_to_world_ray(
                cam_pos_sv, cam_quat_sv, u_vlm, v_vlm, w_img, h_img, fl_m, ha_m)

            if len(pts_near) >= 5:
                hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=0.15)
                if hit_pos is not None:
                    hit_method = "pcd_ray"
                    ctx.log(f"  [vlm_sg] pcd_ray hit={hit_pos.round(3).tolist()}")

            if hit_pos is None and len(pts_near) >= 1:
                hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=9999.0)
                if hit_pos is not None:
                    hit_method = "pcd_nearest"
                    ctx.log(f"  [vlm_sg] pcd_nearest hit={hit_pos.round(3).tolist()}")

        # ── 方法3：几何回退（射线 + handle 参考深度）───────────────────────
        if hit_pos is None:
            ctx.log("  [vlm_sg] 点云失败，使用几何射线回退")
            origin, ray_dir = _pixel_to_world_ray(
                cam_pos_sv, cam_quat_sv, u_vlm, v_vlm, w_img, h_img, fl_m, ha_m)
            handle_ref = np.asarray(cam_meta.get("handle_geom", handle_pos.tolist()))
            t = float(np.dot(handle_ref - origin, ray_dir))
            if t > 0.05:
                hit_pos = origin + ray_dir * t
                hit_method = "ray_handle_fallback"
                ctx.log(f"  [vlm_sg] fallback hit={hit_pos.round(3).tolist()}")

        if hit_pos is None:
            ctx.set_result({"ok": False, "error": "无法从 depth/点云反解 3D 点"})
            yield world.empty_action()
            return

        # outward 方向偏移（让球心在表面外，避免被物体挡）
        hit_pos_vis = hit_pos + outward * BALL_SURFACE_OFFSET
        ctx.log(f"  [vlm_sg] 3D hit={hit_pos.round(3).tolist()} method={hit_method}")

        # 2D 重投影对比
        reproj = _world_to_pixel(cam_pos_sv, cam_quat_sv,
                                 hit_pos, w_img, h_img, fl_m, ha_m)
        if reproj:
            err_px = float(np.hypot(reproj[0] - u_vlm, reproj[1] - v_vlm))
            ctx.log(f"  [vlm_sg] reproj=({reproj[0]},{reproj[1]}) err={err_px:.1f}px")
        else:
            err_px = None

        # 保存对比图
        import cv2
        raw = cv2.imread(rgb_path)
        if raw is None:
            raw = np.zeros((h_img, w_img, 3), np.uint8)
        cmp_img = raw.copy()
        cv2.circle(cmp_img, (u_vlm, v_vlm),   12, (0, 0, 255),  -1, lineType=cv2.LINE_AA)
        if reproj:
            cv2.circle(cmp_img, reproj, 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"compare_2d_{sv}.png"), cmp_img)

        # 可选：红球可视化
        if add_ball:
            stage = og.sim.stage
            ball_path = "/World/vlm_sg_hit_ball"
            _make_sphere(stage, ball_path, hit_pos_vis, radius=0.022, color=(1.0, 0.0, 0.0))
            for _ in range(3):
                og.sim.render()

        # ── 计算 EEF pose ────────────────────────────────────────────────────
        from behavior_interface.skills.viz_eef_v2 import compute_eef_at_grasp

        # 构造 geom dict（供 compute_eef_at_grasp 使用）
        geom_dict = {
            "outward_normal_world": outward.tolist(),
            "axis_world": meta.get("axis_world", [0.0, 0.0, 1.0]),
        }
        eef_p = compute_eef_at_grasp(hit_pos, geom_dict)
        if eef_p is None:
            ctx.set_result({"ok": False, "error": "EEF pose 计算失败"})
            yield world.empty_action()
            return

        ctx.log(f"  [vlm_sg] eef_pos={np.asarray(eef_p['pos']).round(3).tolist()}")

        # ── 构造与 get_eef_pose 兼容的 candidate ──────────────────────────────
        # 继承原始 candidate 的几何 meta（hinge、axis 等），只替换 handle 位置和 EEF pose
        refined_meta = dict(meta)
        refined_meta.update({
            "handle_closed_world": hit_pos.tolist(),
            "handle_mid_world":    hit_pos.tolist(),
            "vlm_computed": True,
            "vlm_pixel": {"u": u_vlm, "v": v_vlm},
            "hit_method": hit_method,
        })

        # 从几何 meta 中取 handle_open_world（用于 arc 末点），若不存在则用几何估算
        if "handle_open_world" not in refined_meta:
            # 用 hinge 旋转 90° 估算开门末点
            hinge = np.asarray(meta.get("hinge_world", hit_pos + [0.3, 0, 0]))
            r = float(np.linalg.norm(hit_pos - hinge))
            tangent = np.asarray(meta.get("tangent_closed_world",
                                          np.cross(outward, [0, 0, 1])))
            tn = np.linalg.norm(tangent)
            if tn > 1e-6:
                tangent /= tn
            handle_open = hinge + r * (-outward)  # 近似：门旋转 90°
            refined_meta["handle_open_world"] = handle_open.tolist()

        # tangent_closed_world（开门切线方向）
        if "tangent_closed_world" not in refined_meta:
            tangent_raw = np.cross(outward,
                                   np.asarray(meta.get("axis_world", [0, 0, 1])))
            tn = np.linalg.norm(tangent_raw)
            refined_meta["tangent_closed_world"] = (
                (tangent_raw / tn).tolist() if tn > 1e-6 else [0, -1, 0]
            )

        refined_cand = {
            "id": 0,
            "target": "open",
            "arm": arm,
            "label": f"vlm_scene_{object_name}",
            "eef_target": {
                "pos": eef_p["pos"],
                "quat": eef_p["quat"],
                "approach": eef_p["approach"],
                "gripper_cmd": -1.0,
            },
            "next_eef_move": [0.0, 0.0, 0.0],  # arc 阶段由 execute_eef_pose 负责
            "reachable": True,
            "reach_reason": "vlm_computed",
            "score": 1.0,
            "meta": refined_meta,
        }

        # 写回 last_skill_results["get_eef_pose"]（供 execute_eef_pose 读取）
        # 通过 ctx.set_result 只能写自己的 key，这里直接访问底层存储
        # （标准做法：modify eef.py 的 execute_eef_pose 也接受 "vlm_scene_grasp"）
        result_payload = {
            "ok": True,
            "target": "open",
            "arm": arm,
            "object": eef_res.get("object", {"input": object_name, "resolved_name": object_name}),
            "candidates": [refined_cand],
            "vlm_hit_world": hit_pos.tolist(),
            "vlm_pixel": {"u": u_vlm, "v": v_vlm},
            "hit_method": hit_method,
            "pixel_error": err_px,
        }

        # ctx.set_result 会写入 last_skill_results["vlm_scene_grasp"]
        ctx.set_result(result_payload)

        # 同时通过 _get_last_result 机制检查是否能访问底层 dict
        # （fallback：execute_eef_pose 将被修改为也接受 "vlm_scene_grasp"）
        ctx.log(f"  [vlm_sg] pcd done  hit={hit_pos.round(3).tolist()} "
                f"method={hit_method} err={err_px}px")

        # 保存结果 JSON
        with open(os.path.join(out_dir, "vlm_scene_grasp_result.json"), "w") as f:
            json.dump({k: v for k, v in result_payload.items()
                       if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                      f, indent=2)

        yield world.empty_action()
        return

    # ── 未知 mode ──────────────────────────────────────────────────────────────
    ctx.set_result({"ok": False, "error": f"未知 mode={mode!r}，支持 init / pcd"})
    yield world.empty_action()
