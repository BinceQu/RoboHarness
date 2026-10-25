"""
mark_handle skill：
  1. 找指定物体的 revolute joint 子 link（门板）
  2. 用 bbox 几何算出"把手候选点"（free edge × front face）
  3. 在场景中放红色球形标记
  4. 从 GTA 主视图、头部相机、右手腕相机分别截图保存
  5. 返回图像路径列表

用法：
  POST /api/skill  {"name":"mark_handle","args":{"object_name":"microwave"}}
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 把手位置计算核心（通用 revolute joint 版本）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_handle_candidates(obj, robot_pos: np.ndarray, ctx) -> List[Dict]:
    """
    对 obj 的每个 revolute joint：
      1. 取子 link 在世界系的 pose
      2. 从 native_link_bboxes 读 link frame 内的 bbox extent + center_offset
      3. 找 door_extent_axis = 距铰链 origin 最远的 bbox 轴（从铰链到自由边缘）
      4. free_edge_in_link = center + half_extent × sign（沿该轴）
      5. thickness_axis = extent 最小的轴 → 两个候选"前面/后面"
      6. 分别变换到世界系，取离机器人更近的作为前面（front face handle）
    返回候选列表，每个元素含：
      {
        "joint_name": str, "link_name": str,
        "handle_world": np.ndarray[3],       # 自由边缘中心，世界系
        "front_handle_world": np.ndarray[3], # 前面×自由边缘，世界系
        "approach_world": np.ndarray[3],     # 接近方向（单位向量，指向门面）
        "link_pos": np.ndarray[3],
        "link_quat": np.ndarray[4],
        "extent": list,
        "center_in_link": list,
        "door_extent_axis": int,
        "thickness_axis": int,
      }
    """
    import omnigibson as og
    from omnigibson.utils.constants import JointType

    results = []

    for jname, joint in obj.joints.items():
        if joint.joint_type != JointType.JOINT_REVOLUTE:
            continue
        # 子 link 名
        link_name = joint.body1.split("/")[-1]
        link = obj.links.get(link_name)
        if link is None:
            ctx.log(f"  [mark] joint={jname} link={link_name} → link not found, skip")
            continue

        # link 世界 pose
        lp, lq = link.get_position_orientation()
        link_pos = _to_np(lp)
        link_quat = _to_np(lq)

        # 方法1: 先用 metadata["link_bounding_boxes"] 获取 link frame 内 bbox
        # 方法2: 运行时 AABB → 转换到 link frame（回退）
        extent = None
        center_in_link = None

        # 尝试 metadata
        meta = getattr(obj, "metadata", None)
        if meta and isinstance(meta, dict):
            bb_map = meta.get("link_bounding_boxes", {})
            bb_link = bb_map.get(link_name, {})
            col = bb_link.get("collision", {}).get("axis_aligned", {})
            ext_raw = col.get("extent")
            tf_raw  = col.get("transform")
            if ext_raw is not None and tf_raw is not None:
                extent = np.array(ext_raw, dtype=np.float64)
                center_in_link = np.array(tf_raw, dtype=np.float64)[:3, 3]
                ctx.log(f"  [mark] {link_name}: bbox from metadata OK")

        if extent is None:
            # 回退：用运行时 AABB 转换到 link frame
            try:
                aabb_lo, aabb_hi = link.aabb
                lo = _to_np(aabb_lo)
                hi = _to_np(aabb_hi)
                # 转到 link frame
                x, y, z, w = link_quat
                R = np.array([
                    [1 - 2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
                    [2*(x*y + w*z),     1 - 2*(x*x+z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x+y*y)],
                ], dtype=np.float64)
                Rt = R.T
                lo_link = Rt @ (lo - link_pos)
                hi_link = Rt @ (hi - link_pos)
                # 取 min/max（世界 AABB 的角点在 link frame 不一定对齐）
                mn = np.minimum(lo_link, hi_link)
                mx = np.maximum(lo_link, hi_link)
                extent = mx - mn
                center_in_link = (mx + mn) / 2.0
                ctx.log(f"  [mark] {link_name}: bbox from runtime AABB "
                        f"extent={extent.round(3)} center={center_in_link.round(3)}")
            except Exception as e:
                ctx.log(f"  [mark] {link_name}: AABB 失败: {e}, skip")
                continue

        if extent is None or center_in_link is None:
            ctx.log(f"  [mark] {link_name}: 无法获取 bbox，skip")
            continue

        # 旋转矩阵 link→world（已在 AABB 回退中可能计算过，这里统一重算）
        x, y, z, w = link_quat
        R = np.array([
            [1 - 2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x+z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x+y*y)],
        ], dtype=np.float64)

        # ── 找 door_extent_axis：哪个轴让 bbox center 离 link origin 最远（归一化） ──
        # 归一化到 bbox half-extent，避免 tiny 轴有最大绝对偏移
        half_ext = extent / 2.0 + 1e-9
        normalized_offset = np.abs(center_in_link) / half_ext
        door_extent_axis = int(np.argmax(normalized_offset))
        thickness_axis   = int(np.argmin(extent))          # 最薄的轴 = 门板厚度

        ctx.log(f"  [mark] {link_name}: extent={extent.round(3)} "
                f"center_in_link={center_in_link.round(3)} "
                f"door_extent_axis={door_extent_axis} thickness_axis={thickness_axis}")

        # ── free edge center in link frame ──
        free_edge_in_link = center_in_link.copy()
        sign = np.sign(center_in_link[door_extent_axis])
        free_edge_in_link[door_extent_axis] = (
            center_in_link[door_extent_axis] + sign * extent[door_extent_axis] / 2.0
        )

        # ── 两个厚度轴候选"前/后面" ──
        front_cands = []
        for face_sign in [+1.0, -1.0]:
            p = free_edge_in_link.copy()
            p[thickness_axis] = (
                center_in_link[thickness_axis] + face_sign * extent[thickness_axis] / 2.0
            )
            p_world = link_pos + R @ p
            dist_to_robot = np.linalg.norm(p_world - robot_pos)
            front_cands.append((dist_to_robot, p_world, face_sign))

        front_cands.sort(key=lambda t: t[0])
        _, front_handle_world, front_sign = front_cands[0]   # 更近的那面 = 前面

        # 接近方向：从门面外部指向门面（单位向量）
        approach_in_link = np.zeros(3)
        approach_in_link[thickness_axis] = -front_sign   # 朝向门内
        approach_world = R @ approach_in_link

        # free edge center（不区分前后面）→ 世界系
        handle_world = link_pos + R @ free_edge_in_link

        ctx.log(f"  [mark] {link_name}: free_edge_in_link={free_edge_in_link.round(3)} "
                f"→ world={handle_world.round(3)}")
        ctx.log(f"  [mark] {link_name}: front_handle_world={front_handle_world.round(3)} "
                f"approach_world={approach_world.round(3)}")

        results.append({
            "joint_name": jname,
            "link_name": link_name,
            "handle_world": handle_world,
            "front_handle_world": front_handle_world,
            "approach_world": approach_world,
            "link_pos": link_pos,
            "link_quat": link_quat,
            "extent": extent.tolist(),
            "center_in_link": center_in_link.tolist(),
            "door_extent_axis": door_extent_axis,
            "thickness_axis": thickness_axis,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 红球放置（使用 USD API，轻量，不走 OmniGibson 对象系统）
# ─────────────────────────────────────────────────────────────────────────────

def _place_red_sphere(pos: np.ndarray, prim_path: str, radius: float = 0.025):
    """在 USD stage 上创建一个无碰撞的红色小球用于可视化。返回 prim。"""
    import omnigibson as og
    import omnigibson.lazy as lazy

    stage = og.sim.stage
    sphere = lazy.pxr.UsdGeom.Sphere.Define(stage, lazy.pxr.Sdf.Path(prim_path))
    sphere.GetRadiusAttr().Set(float(radius))
    # 位置
    xform = lazy.pxr.UsdGeom.Xformable(sphere)
    xform.ClearXformOpOrder()
    t_op = xform.AddTranslateOp()
    t_op.Set(lazy.pxr.Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    # 红色 displayColor（简单，不需要 shader）
    try:
        sphere.GetDisplayColorAttr().Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(1.0, 0.0, 0.0)]))
    except Exception:
        pass
    # 关闭碰撞（仅可视）
    try:
        prim = stage.GetPrimAtPath(prim_path)
        prim.SetMetadata("purpose", "guide")
        col_api = lazy.pxr.UsdPhysics.CollisionAPI.Apply(prim)
        col_api.GetCollisionEnabledAttr().Set(False)
    except Exception:
        pass
    return sphere


def _remove_prim(prim_path: str):
    """删除 USD prim。"""
    import omnigibson as og
    import omnigibson.lazy as lazy

    stage = og.sim.stage
    stage.RemovePrim(lazy.pxr.Sdf.Path(prim_path))


# ─────────────────────────────────────────────────────────────────────────────
# 截图工具
# ─────────────────────────────────────────────────────────────────────────────

def _capture_from_sensor(sensor, label: str, out_dir: str, ctx) -> Optional[str]:
    """从 VisionSensor 捕获 RGB 并保存为 PNG。返回文件路径或 None。"""
    import cv2
    try:
        obs, _ = sensor.get_obs()
        rgb = obs.get("rgb")
        if rgb is None:
            ctx.log(f"  [mark] {label}: rgb=None")
            return None
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        fname = os.path.join(out_dir, f"{label}.png")
        cv2.imwrite(fname, bgr)
        ctx.log(f"  [mark] saved {fname} shape={bgr.shape}")
        return fname
    except Exception as e:
        ctx.log(f"  [mark] {label} capture failed: {e}")
        return None


def _move_gta_camera(sensor, cam_pos: np.ndarray, look_at: np.ndarray):
    """将 GTA sensor 移到指定位置，朝向 look_at。"""
    import torch as th

    forward = look_at - cam_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return
    forward /= norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(world_up, -forward)
    right /= max(np.linalg.norm(right), 1e-9)
    cam_up = np.cross(-forward, right)
    cam_up /= max(np.linalg.norm(cam_up), 1e-9)
    R = np.column_stack([right, cam_up, -forward]).astype(np.float64)

    # R → xyzw quat
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)

    sensor.set_position_orientation(
        position=th.tensor(cam_pos, dtype=th.float32),
        orientation=th.tensor(quat_xyzw, dtype=th.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主 skill 生成器
# ─────────────────────────────────────────────────────────────────────────────

@register_skill("mark_handle", description="计算把手位置，放红球标记，多角度截图")
def mark_handle(ctx, object_name: str = "microwave"):
    """
    1. 找物体 → 算把手候选点
    2. 放红球标记
    3. 从 GTA（正面/侧面/俯视） + 头部 + 右手腕 相机截图
    4. 删除红球
    5. 返回文件路径列表
    """
    import omnigibson as og

    world = ctx.world
    ctx.log(f"[mark_handle] 开始，目标物体: {object_name}")

    # ── 1. 找物体 ──
    obj = None
    for o in world.env.scene.objects:
        nm = getattr(o, "name", "")
        cat = getattr(o, "category", "")
        if object_name.lower() in nm.lower() or object_name.lower() in cat.lower():
            obj = o
            break
    if obj is None:
        ctx.log(f"[mark_handle] 找不到物体 {object_name}")
        ctx.set_result({"ok": False, "error": f"object not found: {object_name}"})
        return
    ctx.log(f"[mark_handle] 找到物体: {obj.name}")

    # ── 2. 机器人位置（用于判断哪面是前面） ──
    rp, _ = world.robot.get_position_orientation()
    robot_pos = _to_np(rp)
    ctx.log(f"[mark_handle] robot_pos={robot_pos.round(3)}")

    # ── 3. 计算把手候选点 ──
    candidates = _compute_handle_candidates(obj, robot_pos, ctx)
    if not candidates:
        ctx.log("[mark_handle] 没有 revolute joint，退出")
        ctx.set_result({"ok": False, "error": "no revolute joints found"})
        return

    ctx.log(f"[mark_handle] 找到 {len(candidates)} 个关节把手候选")

    # ── 4. 创建红球标记 ──
    marker_paths = []
    for i, cand in enumerate(candidates):
        # 在 free edge 处放一个大球（橙/红色）
        p1 = f"/World/debug_handle_free_{i}"
        _place_red_sphere(cand["handle_world"], p1, radius=0.04)
        marker_paths.append(p1)
        ctx.log(f"  [mark] 红球 {p1} @ {cand['handle_world'].round(3)}")

        # 在 front_handle 处放一个小球（精确位置）
        p2 = f"/World/debug_handle_front_{i}"
        _place_red_sphere(cand["front_handle_world"], p2, radius=0.025)
        marker_paths.append(p2)
        ctx.log(f"  [mark] 小红球 {p2} @ {cand['front_handle_world'].round(3)}")

    # ── 5. 渲染几帧让标记生效 ──
    for _ in range(5):
        yield world.empty_action()   # 空动作 → sim step + 渲染

    # ── 6. 截图 ──
    out_dir = "/tmp/handle_vis"
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []

    # GTA 外部相机（多角度）
    gta_sensor = world.env._external_sensors.get("gta_view") if hasattr(world.env, "_external_sensors") else None

    if gta_sensor is None:
        ctx.log("  [mark] GTA sensor 不可用")
    else:
        # 主要看微波炉把手的几个角度
        # 取第一个候选的把手位置作为 look_at 目标
        look_at = candidates[0]["front_handle_world"]

        views = [
            ("front",   np.array([look_at[0] - 1.5, look_at[1],        look_at[2] + 0.5])),
            ("side_r",  np.array([look_at[0],        look_at[1] + 1.5,  look_at[2] + 0.5])),
            ("top",     np.array([look_at[0],        look_at[1],        look_at[2] + 2.0])),
            ("diag",    np.array([look_at[0] - 1.0,  look_at[1] + 1.0, look_at[2] + 1.0])),
        ]

        for vname, cam_pos in views:
            _move_gta_camera(gta_sensor, cam_pos, look_at)
            # 渲染 2 帧
            try:
                og.sim.render()
                og.sim.render()
            except Exception:
                pass
            yield world.empty_action()   # sim step
            fpath = _capture_from_sensor(gta_sensor, f"gta_{vname}", out_dir, ctx)
            if fpath:
                saved_files.append(fpath)

    # 机器人头部相机
    for sname, sensor in world.robot.sensors.items():
        if "rgb" not in getattr(sensor, "modalities", []):
            continue
        if "zed_link" in sname:
            fpath = _capture_from_sensor(sensor, "robot_head", out_dir, ctx)
            if fpath:
                saved_files.append(fpath)
        elif "right_realsense" in sname:
            fpath = _capture_from_sensor(sensor, "robot_right_wrist", out_dir, ctx)
            if fpath:
                saved_files.append(fpath)

    # ── 7. 删除红球 ──
    for p in marker_paths:
        try:
            _remove_prim(p)
        except Exception as e:
            ctx.log(f"  [mark] 删除 {p} 失败: {e}")

    ctx.log(f"[mark_handle] 完成，共保存 {len(saved_files)} 张截图: {saved_files}")
    ctx.set_result({
        "ok": True,
        "saved_files": saved_files,
        "candidates": [
            {
                "joint_name": c["joint_name"],
                "link_name": c["link_name"],
                "handle_world": c["handle_world"].tolist(),
                "front_handle_world": c["front_handle_world"].tolist(),
                "approach_world": c["approach_world"].tolist(),
                "extent": c["extent"],
                "door_extent_axis": c["door_extent_axis"],
                "thickness_axis": c["thickness_axis"],
            }
            for c in candidates
        ],
    })
