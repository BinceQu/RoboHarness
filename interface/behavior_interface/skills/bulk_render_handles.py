"""
bulk_render_handles skill:
  在现有服务器场景中逐个加载 20 个铰链物体，
  放把手红球标记，用 GTA 相机捕获三视图，
  保存到 /tmp/real_handles/<cat>_<inst>_<view>.png

用法：
  POST /api/skill  {"name":"bulk_render_handles","args":{}}
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill

# ── 输出目录
OUT_DIR = "/tmp/real_handles"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 20 个铰链测试物体
OBJECTS = [
    ("microwave",      "abzvij"),
    ("microwave",      "hjjxmi"),
    ("microwave",      "bfbeeb"),
    ("microwave",      "ihxrvr"),
    ("fridge",         "dszchb"),
    ("fridge",         "dxwbae"),
    ("top_cabinet",    "dhguym"),
    ("top_cabinet",    "dmwxyl"),
    ("bottom_cabinet", "bamfsz"),
    ("bottom_cabinet", "bycegi"),
    ("wardrobe",       "brimns"),
    ("door",           "adejun"),
    ("dishwasher",     "cgugie"),
    ("oven",           "amblrk"),
    ("washer",         "dobgmu"),
    ("laptop",         "izydvb"),
    ("briefcase",      "fdpjtj"),
    ("desk",           "aduafr"),
    ("rice_cooker",    "qatgeb"),
    ("toaster_oven",   "ctrngh"),
]

# 物体悬浮在空中（z=2.5m），避开所有场景家具
# 每步通过 set_position_orientation 强制保持位置
FLOAT_POS  = [6.0, 0.0, 2.5]   # 悬浮坐标（机器人正上方 2.5m）
FLOAT_QUAT = [0.0, 0.0, 0.0, 1.0]  # 标准朝向（identity）


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 把手检测（同 mark_handle.py）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_handles(obj, robot_pos: np.ndarray) -> List[Dict]:
    from omnigibson.utils.constants import JointType

    results = []
    # 先收集 handle link 名（优先级高）
    handle_links = {ln for ln in obj.links
                    if ("handle" in ln.lower() or "knob" in ln.lower()) and ln != "base_link"}

    for jname, joint in obj.joints.items():
        if joint.joint_type != JointType.JOINT_REVOLUTE:
            continue
        link_name = joint.body1.split("/")[-1]
        link = obj.links.get(link_name)
        if link is None:
            continue

        lp, lq = link.get_position_orientation()
        link_pos  = _to_np(lp)
        link_quat = _to_np(lq)

        # bbox
        extent = center_in_link = None
        meta = getattr(obj, "metadata", None)
        if meta:
            bb = meta.get("link_bounding_boxes", {}).get(link_name, {})
            col = bb.get("collision", {}).get("axis_aligned", {})
            ext = col.get("extent")
            tf  = col.get("transform")
            if ext and tf:
                extent         = np.array(ext, dtype=np.float64)
                center_in_link = np.array(tf, dtype=np.float64)[:3, 3]

        if extent is None:
            try:
                lo, hi = link.aabb
                lo, hi = _to_np(lo), _to_np(hi)
                x, y, z, w = link_quat
                Rm = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                    [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
                ], dtype=np.float64)
                lo_l = Rm.T @ (lo - link_pos)
                hi_l = Rm.T @ (hi - link_pos)
                mn, mx = np.minimum(lo_l, hi_l), np.maximum(lo_l, hi_l)
                extent         = mx - mn
                center_in_link = (mx + mn) / 2.0
            except Exception:
                continue

        if extent is None:
            continue

        x, y, z, w = link_quat
        Rm = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ], dtype=np.float64)

        half_ext = extent / 2.0 + 1e-9
        norm_off = np.abs(center_in_link) / half_ext
        door_ax  = int(np.argmax(norm_off))
        thick_ax = int(np.argmin(extent))

        free_edge = center_in_link.copy()
        sd = np.sign(center_in_link[door_ax]) if center_in_link[door_ax] != 0 else 1.0
        free_edge[door_ax] = center_in_link[door_ax] + sd * extent[door_ax] / 2.0

        cands = []
        for fs in [+1.0, -1.0]:
            p = free_edge.copy()
            p[thick_ax] = center_in_link[thick_ax] + fs * extent[thick_ax] / 2.0
            pw = link_pos + Rm @ p
            cands.append((np.linalg.norm(pw - robot_pos), pw, fs))
        cands.sort(key=lambda t: t[0])
        _, front_world, fsgn = cands[0]

        approach = np.zeros(3)
        approach[thick_ax] = -fsgn
        approach_world = Rm @ approach

        handle_world = link_pos + Rm @ free_edge

        # 若有专属 handle link，直接用其 bbox 中心
        for h_ln in handle_links:
            h_link = obj.links[h_ln]
            hlp, _ = h_link.get_position_orientation()
            handle_world = _to_np(hlp)
            front_world  = handle_world.copy()
            break

        results.append({
            "joint": jname, "link": link_name,
            "handle": handle_world, "front": front_world,
            "approach": approach_world,
            "extent": extent, "door_ax": door_ax, "thick_ax": thick_ax,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# USD 红球
# ─────────────────────────────────────────────────────────────────────────────

def _place_sphere(pos, path, radius=0.04, color=(1.0, 0.0, 0.0)):
    import omnigibson as og
    import omnigibson.lazy as lazy
    stage = og.sim.stage
    sph = lazy.pxr.UsdGeom.Sphere.Define(stage, lazy.pxr.Sdf.Path(path))
    sph.GetRadiusAttr().Set(float(radius))
    xf = lazy.pxr.UsdGeom.Xformable(sph)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(lazy.pxr.Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    try:
        sph.GetDisplayColorAttr().Set(
            lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*color)]))
    except Exception:
        pass
    try:
        prim = stage.GetPrimAtPath(path)
        api = lazy.pxr.UsdPhysics.CollisionAPI.Apply(prim)
        api.GetCollisionEnabledAttr().Set(False)
    except Exception:
        pass


def _del_sphere(path):
    import omnigibson as og
    import omnigibson.lazy as lazy
    try:
        og.sim.stage.RemovePrim(lazy.pxr.Sdf.Path(path))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 相机朝向
# ─────────────────────────────────────────────────────────────────────────────

def _lookat_quat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    import torch as th
    fwd = np.asarray(target) - np.asarray(cam_pos)
    nf  = np.linalg.norm(fwd)
    if nf < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    fwd /= nf
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    z_cam = -fwd
    right = np.cross(up, z_cam); right /= max(np.linalg.norm(right), 1e-9)
    y_cam = np.cross(z_cam, right); y_cam /= max(np.linalg.norm(y_cam), 1e-9)
    R = np.column_stack([right, y_cam, z_cam])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw, qx, qy, qz = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw, qx, qy, qz = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw, qx, qy, qz = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw, qx, qy, qz = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def _move_cam(sensor, cam_pos, look_at):
    import torch as th
    q = _lookat_quat(np.asarray(cam_pos), np.asarray(look_at))
    sensor.set_position_orientation(
        position=th.tensor(cam_pos, dtype=th.float32),
        orientation=th.tensor(q, dtype=th.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 截图
# ─────────────────────────────────────────────────────────────────────────────

def _capture(sensor, fpath: str, ctx) -> bool:
    import cv2
    try:
        obs, _ = sensor.get_obs()
        rgb = obs.get("rgb")
        if rgb is None:
            ctx.log(f"  [cap] rgb=None → {fpath}")
            return False
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fpath, bgr)
        ctx.log(f"  [cap] OK {fpath} {bgr.shape}")
        return True
    except Exception as e:
        ctx.log(f"  [cap] error: {e}")
        return False


def _merge_views3(paths: Dict, out_path: str, title: str, ctx):
    import cv2
    view_names = ["front_diag", "side_diag", "top_down"]
    labels     = ["FRONT-DIAG", "SIDE-DIAG",  "TOP-DOWN"]
    imgs = []
    for vname, lbl in zip(view_names, labels):
        p = paths.get(vname, "")
        if p and os.path.exists(p):
            im = cv2.imread(p)
        else:
            im = np.zeros((480, 640, 3), dtype=np.uint8)
        im = cv2.resize(im, (640, 480))
        cv2.putText(im, lbl, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        imgs.append(im)
    # 图例：标注颜色含义
    legend = np.zeros((50, 640*3, 3), dtype=np.uint8)
    cv2.circle(legend, (30, 25), 12, (0, 128, 255), -1)   # BGR 橙 = (0,128,255)
    cv2.putText(legend, "橙色大球=自由边缘中心(free edge)", (50, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(legend, (560, 25), 8, (0, 0, 255), -1)     # BGR 红 = (0,0,255)
    cv2.putText(legend, "红色小球=前面把手点(grasp point)", (580, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row = np.hstack(imgs)
    bar = np.zeros((50, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2)
    final = np.vstack([bar, legend, row])
    cv2.imwrite(out_path, final)
    ctx.log(f"  [merge] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主 skill
# ─────────────────────────────────────────────────────────────────────────────

@register_skill("bulk_render_handles3",
                description="第三版：悬浮+大球+水平三角度渲染把手标记")
def bulk_render_handles(ctx):
    """
    在现有场景中依次加载 20 个铰链物体（漂浮在 z=5m 处），
    检测把手位置，放红球标记，GTA 相机三视图截图，保存合并图。
    """
    import omnigibson as og
    from omnigibson.objects import DatasetObject

    world = ctx.world
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        ctx.log("[bulk] 找不到 GTA 相机，退出")
        ctx.set_result({"ok": False, "error": "no gta_view sensor"})
        return

    ctx.log(f"[bulk] 开始批量渲染，共 {len(OBJECTS)} 个物体，输出到 {OUT_DIR}")
    all_results = []

    import torch as th

    for idx, (cat, inst) in enumerate(OBJECTS):
        label = f"{idx+1:02d}_{cat}_{inst}"
        ctx.log(f"\n[bulk] === [{idx+1}/{len(OBJECTS)}] {cat}/{inst} ===")

        obj_name = f"test_{inst}"
        obj = None
        sphere_paths = []

        try:
            # ── 1. 添加物体（悬浮在 z=2.5m 避开家具）
            fp = list(FLOAT_POS)
            obj = DatasetObject(
                name=obj_name,
                category=cat,
                model=inst,
                position=fp,
                orientation=FLOAT_QUAT,
            )
            world.env.scene.add_object(obj)
            ctx.log(f"  [bulk] {cat}/{inst} 已添加，初始位置={fp}")

            # 3 步初始化，每步强制重置位置（对抗重力）
            fp_t  = th.tensor(fp, dtype=th.float32)
            fq_t  = th.tensor(FLOAT_QUAT, dtype=th.float32)
            for _ in range(3):
                yield world.empty_action()
                obj.set_position_orientation(position=fp_t, orientation=fq_t)

            # ── 2. 读 AABB（现在物体应在悬浮位置）
            try:
                lo, hi = obj.aabb
                lo, hi = _to_np(lo), _to_np(hi)
            except Exception:
                lo = np.array(fp) - 0.5
                hi = np.array(fp) + 0.5
            obj_ctr = (lo + hi) / 2.0
            obj_ext = hi - lo
            dist    = float(max(np.max(obj_ext) * 1.8, 1.0))
            ctx.log(f"  [bulk] float ctr={obj_ctr.round(3)} ext={obj_ext.round(3)} dist={dist:.2f}")

            # ── 3. 计算把手（假设机器人在 y- 方向）
            robot_pos = obj_ctr + np.array([0.0, -5.0, 0.0])
            handles = _compute_handles(obj, robot_pos)
            ctx.log(f"  [bulk] 把手候选 {len(handles)} 个")

            # ── 4. 放标记球（大尺寸，保证画面可见）
            #   橙色大球 (r≈10cm)：自由边缘中心 (free edge center)
            #   红色中球 (r≈6cm) ：前面朝向点 (grasp point，最终抓取位)
            for i, h in enumerate(handles):
                rad_big   = max(float(max(obj_ext)) * 0.08, 0.10)
                rad_small = max(float(max(obj_ext)) * 0.05, 0.06)
                p1 = f"/World/bulk_free_{i}"
                _place_sphere(h["handle"], p1, rad_big,   (1.0, 0.5, 0.0))   # 橙色
                sphere_paths.append(p1)
                p2 = f"/World/bulk_front_{i}"
                _place_sphere(h["front"],  p2, rad_small, (1.0, 0.0, 0.0))   # 红色
                sphere_paths.append(p2)

            # 再强制 2 次重置位置（确保悬浮）+ 等标记生效
            for _ in range(2):
                yield world.empty_action()
                obj.set_position_orientation(position=fp_t, orientation=fq_t)

            # ── 5. 三视图截图
            # look_at 聚焦第一个把手（或物体中心）
            if handles:
                look_at = np.asarray(handles[0]["front"], dtype=np.float64)
            else:
                look_at = obj_ctr.copy()

            # 物体悬浮在 obj_ctr，三个水平方向相机（不穿墙）
            cam_dist = max(dist * 1.2, 1.5)
            views_def = {
                "front_diag": obj_ctr + np.array([cam_dist*0.7, -cam_dist*0.7,  cam_dist*0.4]),
                "side_diag":  obj_ctr + np.array([-cam_dist,    0.0,            cam_dist*0.4]),
                "top_down":   obj_ctr + np.array([0.0,          0.0,            cam_dist*1.5]),
            }

            saved_paths = {}
            for vname, cam_pos in views_def.items():
                _move_cam(gta, cam_pos, look_at)
                # 保持物体悬浮后再渲染
                obj.set_position_orientation(position=fp_t, orientation=fq_t)
                try:
                    og.sim.render()
                    og.sim.render()
                except Exception:
                    pass
                fpath = os.path.join(OUT_DIR, f"{label}_{vname}.png")
                ok = _capture(gta, fpath, ctx)
                if ok:
                    saved_paths[vname] = fpath

            # 合并三视图
            merged = os.path.join(OUT_DIR, f"{label}_merged.png")
            _merge_views3(saved_paths, merged,
                          f"{cat}/{inst}  handles={len(handles)}", ctx)

            all_results.append({
                "cat": cat, "inst": inst,
                "handles": len(handles),
                "images": saved_paths,
                "merged": merged,
            })

        except Exception as e:
            import traceback
            ctx.log(f"  [bulk] ERROR: {e}\n{traceback.format_exc()}")

        finally:
            # ── 6. 清理球标和物体
            for p in sphere_paths:
                _del_sphere(p)
            if obj is not None:
                try:
                    world.env.scene.remove_object(obj=obj)
                    ctx.log(f"  [bulk] removed {cat}/{inst}")
                except Exception as e:
                    ctx.log(f"  [bulk] remove failed: {e}")
            for _ in range(3):
                yield world.empty_action()

    ctx.log(f"\n[bulk] 全部完成，共处理 {len(all_results)}/{len(OBJECTS)} 个物体")
    ctx.set_result({
        "ok": True,
        "total": len(all_results),
        "results": all_results,
        "out_dir": OUT_DIR,
    })
