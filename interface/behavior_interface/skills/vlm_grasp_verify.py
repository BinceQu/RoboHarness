"""
vlm_grasp_verify skill：
  1. 渲染无夹爪的空白微波炉图 + 保存相机位姿
  2. （外部脚本 vlm_grasp_openrouter.py 打 VLM 红点）
  3. 读取 vlm_result.json 的像素 → 射线与物体求交 → 绿色球
  4. 将 3D 交点重投影到图像，与 VLM 红点对比误差

用法：
  # 仅渲染空白图
  POST /api/skill {"name":"vlm_grasp_verify","args":{"mode":"render","category":"microwave","model":"hjjxmi"}}

  # 射线验证（需先有 vlm_result.json）
  POST /api/skill {"name":"vlm_grasp_verify","args":{"mode":"verify","vlm_json":"/tmp/vlm_grasp_test/vlm_result.json"}}
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.head_capture import (
    HEAD_FOCAL_LENGTH,
    HEAD_HORIZONTAL_APERTURE,
)
from behavior_interface.skills import register_skill

OUT_DIR = "/tmp/vlm_grasp_test"
FLOAT_XY = [0.0, 0.0]
FLOAT_Z = 0.0  # 固定在地面，不悬浮
CAM_D, CAM_SIDE, CAM_UP = 0.8, 0.55, 0.35


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        return np.array([(R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s,
                         (R[1, 0]-R[0, 1])/s, 0.25*s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s,
                         (R[2, 1]-R[1, 2])/s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s,
                         (R[0, 2]-R[2, 0])/s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s,
                     (R[1, 0]-R[0, 1])/s])


def _lookat_quat(cam: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = target - cam
    fn = np.linalg.norm(fwd)
    if fn < 1e-6:
        return np.array([0., 0., 0., 1.])
    fwd /= fn
    up = np.array([0., 0., 1.])
    z_cam = -fwd
    right = np.cross(up, z_cam)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        up = np.array([0., 1., 0.])
        right = np.cross(up, z_cam)
        rn = np.linalg.norm(right)
    right /= rn
    y_cam = np.cross(z_cam, right)
    y_cam /= max(np.linalg.norm(y_cam), 1e-9)
    R = np.column_stack([right, y_cam, z_cam])
    return _mat_to_quat_xyzw(R)


def _pixel_to_world_ray(cam_pos, cam_quat_xyzw, u, v, w, h,
                        focal_length=HEAD_FOCAL_LENGTH,
                        horizontal_aperture=HEAD_HORIZONTAL_APERTURE):
    """USD 相机：局部 -Z 为视线；针孔模型。"""
    fx = focal_length / horizontal_aperture * w
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    x_c = (u - cx) / fx
    # 图像 v 向下增大，相机系 y 向上 → 取负
    y_c = -(v - cy) / fy
    # 视线方向（相机系，朝 -Z）
    d_cam = np.array([x_c, y_c, -1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam) + 1e-9
    R = _quat_to_mat(cam_quat_xyzw)
    d_world = R @ d_cam
    return cam_pos.astype(np.float64), d_world


def _depth_to_world_point(sensor, u: int, v: int, cam_pos, cam_quat_xyzw,
                          w: int, h: int, focal_length=HEAD_FOCAL_LENGTH,
                          horizontal_aperture=HEAD_HORIZONTAL_APERTURE) -> Optional[np.ndarray]:
    """用仿真 depth_linear 将像素反投影到世界坐标。"""
    import omnigibson as og
    for _ in range(3):
        og.sim.render()
    obs, _ = sensor.get_obs()
    depth = None
    for k, val in obs.items():
        if "depth_linear" in k.lower():
            depth = val
            break
    if depth is None:
        return None
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    ui = int(np.clip(u, 0, w - 1))
    vi = int(np.clip(v, 0, h - 1))
    d = float(depth[vi, ui])
    if not np.isfinite(d) or d <= 0.01 or d > 50.0:
        return None
    fx = focal_length / horizontal_aperture * w
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    x_c = (ui - cx) / fx * d
    y_c = -(vi - cy) / fy * d
    p_cam = np.array([x_c, y_c, -d], dtype=np.float64)
    R = _quat_to_mat(cam_quat_xyzw)
    return cam_pos + R @ p_cam


def _world_to_pixel(cam_pos, cam_quat_xyzw, p_world, w, h,
                    focal_length=HEAD_FOCAL_LENGTH,
                    horizontal_aperture=HEAD_HORIZONTAL_APERTURE):
    fx = focal_length / horizontal_aperture * w
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    R = _quat_to_mat(cam_quat_xyzw)
    p_cam = R.T @ (p_world - cam_pos)
    # 相机看 -Z
    if p_cam[2] >= -1e-6:
        return None
    u = fx * p_cam[0] / (-p_cam[2]) + cx
    v = cy - fy * p_cam[1] / (-p_cam[2])
    return int(round(u)), int(round(v))


def _move_cam(sensor, cam_pos, look_at):
    """与 mark_handle 相同 GTA 相机朝向，并返回传感器真实位姿。"""
    from behavior_interface.skills.mark_handle import _move_gta_camera
    _move_gta_camera(sensor, cam_pos, look_at)
    import torch as th
    p, q = sensor.get_position_orientation()
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
        q = q.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)


def _read_cam_meta(meta_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _capture(sensor, fpath, ctx) -> bool:
    import cv2
    import omnigibson as og
    try:
        for _ in range(6):
            og.sim.render()
        obs, _ = sensor.get_obs()
        rgb = None
        for k, v in obs.items():
            if "rgb" in k.lower():
                rgb = v
                break
        if rgb is None:
            return False
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        cv2.imwrite(fpath, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        ctx.log(f"  [cap] {e}")
        return False


def _make_sphere(stage, path, pos, radius=0.02, color=(0, 1, 0), opacity=0.95):
    import omnigibson.lazy as lazy
    sph = lazy.pxr.UsdGeom.Sphere.Define(stage, path)
    sph.GetRadiusAttr().Set(float(radius))
    xf = lazy.pxr.UsdGeom.Xformable(sph)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(lazy.pxr.Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    prim = sph.GetPrim()
    try:
        lazy.pxr.UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(
            lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*color)]))
    except Exception:
        pass
    try:
        attr = prim.GetAttribute("primvars:displayOpacity")
        if not attr.IsValid():
            attr = prim.CreateAttribute("primvars:displayOpacity",
                                        lazy.pxr.Sdf.ValueTypeNames.FloatArray)
        attr.Set(lazy.pxr.Vt.FloatArray([float(opacity)]))
    except Exception:
        pass


@register_skill("vlm_grasp_verify")
def vlm_grasp_verify(ctx, mode: str = "render",
                     category: str = "microwave", model: str = "hjjxmi",
                     vlm_json: str = "",
                     out_dir: str = "",
                     arm: str = "right") -> Generator:
    """
    mode=render: 空白微波炉 + 保存相机
    mode=verify: 读 VLM 像素 → 射线 → 绿球 + 重投影对比图
    """
    import torch as th
    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from behavior_interface.skills.eef import _sample_door_geometry, _list_openable_joints

    case_dir = out_dir.strip() or OUT_DIR
    os.makedirs(case_dir, exist_ok=True)
    world = ctx.world
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        ctx.set_result({"ok": False, "error": "gta_view 不可用"})
        yield world.empty_action()
        return

    obj = None
    viz_paths: List[str] = []

    try:
        if mode == "verify" and not vlm_json:
            vlm_json = os.path.join(case_dir, "vlm_result.json")
        if mode == "verify" and not os.path.isfile(vlm_json):
            ctx.set_result({"ok": False, "error": f"缺少 {vlm_json}，请先运行 vlm_grasp_openrouter.py"})
            yield world.empty_action()
            return

        # 加载物体
        obj_pos_t = th.tensor([FLOAT_XY[0], FLOAT_XY[1], FLOAT_Z], dtype=th.float32)
        obj_quat_t = th.tensor([0., 0., 0., 1.], dtype=th.float32)
        obj = DatasetObject(
            name=f"vlm_{model}", category=category, model=model,
            position=obj_pos_t.tolist(), orientation=obj_quat_t.tolist(),
        )
        world.env.scene.add_object(obj)
        for _ in range(8):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

        j_list = _list_openable_joints(obj)
        if not j_list:
            ctx.set_result({"ok": False, "error": "无可开关铰链"})
            yield world.empty_action()
            return
        j, jdir, child = j_list[0]
        hold = [(obj, obj_pos_t, obj_quat_t)]
        geom = yield from _hold_gen(_sample_door_geometry(ctx, obj, j, jdir, child), hold)
        if geom is None:
            ctx.set_result({"ok": False, "error": "几何失败"})
            yield world.empty_action()
            return

        outward = np.asarray(geom["outward_normal_world"], dtype=np.float64)
        outward /= np.linalg.norm(outward) + 1e-9
        handle = np.asarray(geom["handle_closed_world"], dtype=np.float64)
        up_w = np.array([0., 0., 1.])
        perp = np.cross(up_w, outward)
        perp /= np.linalg.norm(perp) + 1e-9
        focus = handle.copy()
        cam_pos = focus + outward * CAM_D + perp * CAM_SIDE + up_w * CAM_UP

        cam_pos, cam_quat = _move_cam(gta, cam_pos, focus)
        for _ in range(4):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

        clean_path = os.path.join(case_dir, "clean.png")
        meta_path = os.path.join(case_dir, "camera_meta.json")
        w_img = int(getattr(gta, "image_width", 1280))
        h_img = int(getattr(gta, "image_height", 720))
        fl = float(getattr(gta, "focal_length", 17.0))
        ha = float(getattr(gta, "horizontal_aperture", 20.995))

        # verify 时始终用当前物体重新截图，避免串 case
        if mode != "verify":
            if not _capture(gta, clean_path, ctx):
                ctx.set_result({"ok": False, "error": "截图失败"})
                yield world.empty_action()
                return
        # 截图后再读一次真实相机位姿（与像素对齐）
        cam_pos, cam_quat = _move_cam(gta, cam_pos, focus)
        for _ in range(3):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

        cam_meta = {
            "cam_pos": cam_pos.tolist(),
            "cam_quat_xyzw": cam_quat.tolist(),
            "look_at": focus.tolist(),
            "image_width": w_img,
            "image_height": h_img,
            "focal_length": fl,
            "horizontal_aperture": ha,
            "outward": outward.tolist(),
            "handle_geom": handle.tolist(),
            "obj_name": obj.name,
            "category": category,
            "model": model,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cam_meta, f, indent=2)

        ctx.log(f"  [vlm] clean={clean_path} cam={cam_pos.round(3).tolist()}")

        if mode == "render":
            ctx.set_result({
                "ok": True,
                "clean_png": clean_path,
                "camera_meta": meta_path,
                "next": f"python behavior_interface/tools/vlm_grasp_openrouter.py --image {clean_path} --out-dir {OUT_DIR}",
            })
            yield world.empty_action()
            return

        # ── verify 模式 ──
        with open(vlm_json, encoding="utf-8") as f:
            vlm = json.load(f)
        u_vlm = int(vlm["pixel"]["u"])
        v_vlm = int(vlm["pixel"]["v"])
        # 若 vlm 与 render 不是同一次截图，仍用当前相机（应一致）
        cam_pos = np.asarray(cam_meta["cam_pos"])
        cam_quat = np.asarray(cam_meta["cam_quat_xyzw"])
        w_img = int(cam_meta["image_width"])
        h_img = int(cam_meta["image_height"])
        fl = float(cam_meta["focal_length"])
        ha = float(cam_meta["horizontal_aperture"])

        hit_pos = None
        hit_method = None
        # 优先：深度反投影（与渲染像素严格一致）
        for _ in range(2):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
        p_depth = _depth_to_world_point(
            gta, u_vlm, v_vlm, cam_pos, cam_quat, w_img, h_img, fl, ha)
        if p_depth is not None:
            hit_pos = p_depth
            hit_method = "depth_linear"
            ctx.log(f"  [vlm] depth hit={hit_pos.round(4).tolist()}")

        if hit_pos is None:
            origin, direction = _pixel_to_world_ray(
                cam_pos, cam_quat, u_vlm, v_vlm, w_img, h_img, fl, ha)
            end = origin + direction * 8.0
            import omnigibson as og
            for _ in range(4):
                og.sim.render()
                yield world.empty_action()
                obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            from omnigibson.utils.sampling_utils import raytest
            hit = raytest(start_point=origin, end_point=end, only_closest=True)
            if hit.get("hit", False):
                hit_pos = _to_np(hit["position"]).reshape(3)
                hit_method = "raytest"
        if hit_pos is None:
            ctx.set_result({"ok": False, "error": "深度与射线均未得到有效 3D 点"})
            yield world.empty_action()
            return
        ctx.log(f"  [vlm] 3D method={hit_method} pos={hit_pos.round(4).tolist()}")
        handle_geom = np.asarray(cam_meta.get("handle_geom", [0, 0, 0]))
        ctx.log(f"  [vlm] handle_geom dist={np.linalg.norm(hit_pos - handle_geom)*1000:.1f}mm")

        stage = og.sim.stage
        green_path = f"/World/vlm_hit_green_{model}"
        _make_sphere(stage, green_path, hit_pos, radius=0.022, color=(0.0, 1.0, 0.0))
        viz_paths.append(green_path)
        # 几何把手参考：蓝色小球
        ref_path = f"/World/vlm_handle_ref_{model}"
        _make_sphere(stage, ref_path, handle_geom, radius=0.018, color=(0.2, 0.4, 1.0))
        viz_paths.append(ref_path)

        reproj = _world_to_pixel(cam_pos, cam_quat, hit_pos, w_img, h_img, fl, ha)
        err_px = None
        if reproj is not None:
            err_px = float(np.hypot(reproj[0] - u_vlm, reproj[1] - v_vlm))

        import cv2
        base = cv2.imread(vlm.get("marked_image", os.path.join(case_dir, "vlm_red_marked.png")))
        if base is None:
            base = cv2.imread(clean_path)
        compare = base.copy() if base is not None else np.zeros((h_img, w_img, 3), np.uint8)
        cv2.circle(compare, (u_vlm, v_vlm), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if reproj is not None:
            cv2.circle(compare, reproj, 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(compare, (u_vlm, v_vlm), reproj, (255, 255, 0), 1)
        compare_path = os.path.join(case_dir, "vlm_compare_2d.png")
        cv2.imwrite(compare_path, compare)

        verify_path = os.path.join(case_dir, "grasp_3d.png")
        for _ in range(2):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
        _capture(gta, verify_path, ctx)

        verify_result = {
            "ok": True,
            "model": vlm.get("model"),
            "hit_method": hit_method,
            "vlm_pixel": {"u": u_vlm, "v": v_vlm},
            "hit_world": hit_pos.tolist(),
            "handle_geom_world": handle_geom.tolist(),
            "handle_dist_mm": float(np.linalg.norm(hit_pos - handle_geom) * 1000),
            "reproj_pixel": {"u": reproj[0], "v": reproj[1]} if reproj else None,
            "pixel_error": err_px,
            "compare_png": compare_path,
            "grasp_3d_png": verify_path,
            "verify_capture_png": verify_path,
            "clean_png": clean_path,
            "out_dir": case_dir,
        }
        vpath = os.path.join(case_dir, "verify_result.json")
        with open(vpath, "w", encoding="utf-8") as f:
            json.dump(verify_result, f, indent=2)

        ctx.set_result(verify_result)
        ctx.log(f"  [vlm] 重投影误差={err_px:.1f}px" if err_px is not None else "  [vlm] 重投影失败")

    finally:
        for p in viz_paths:
            try:
                pr = og.sim.stage.GetPrimAtPath(p)
                if pr.IsValid():
                    og.sim.stage.RemovePrim(p)
            except Exception:
                pass
        if obj is not None:
            try:
                world.env.scene.remove_object(obj)
            except Exception:
                pass
        yield world.empty_action()


def _hold_gen(gen, hold_pairs):
    result = None
    try:
        while True:
            action = next(gen)
            for obj, pt, qt in hold_pairs:
                obj.set_position_orientation(position=pt, orientation=qt)
            yield action
    except StopIteration as e:
        result = e.value
    return result
