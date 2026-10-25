"""
vlm_lawn_dual：草坪场景 + 物体悬浮 (0,0,10m) + 左上/右上双视角

模式说明
--------
render     : 空白渲染双视角 RGB（旧流程，兼容）
init_render: 渲染 right_upper 的 RGB/depth/seg，保存到 {out_dir}/init/
pcd_grasp  : 读 init/ 重建点云 + VLM 射线求最近交点 → 红球渲染到 eef_point/
eef_pose   : 由 pcd_grasp 的 3D 点生成夹爪 pose → 渲染到 eef_pose/
grasp_3d   : 旧 depth+raytest 流程（兼容）
eef_viz    : 旧 eef 流程（兼容）
verify     : 旧单视角验证（兼容）
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface.camera_frames import rgb_array, scalar_image_array
from behavior_interface.skills import register_skill
from behavior_interface.skills.vlm_grasp_verify import (
    CAM_D,
    CAM_SIDE,
    CAM_UP,
    FLOAT_XY,
    FLOAT_Z as _FLOAT_Z_UNUSED,  # 不再悬浮，使用 GROUND_Z=0
    _capture,
    _depth_to_world_point,
    _hold_gen,
    _make_sphere,
    _move_cam,
    _pixel_to_world_ray,
    _quat_to_mat,
    _to_np,
    _world_to_pixel,
)

VIEWS = ("left_upper", "right_upper")
DEFAULT_SOURCE_VIEW = "right_upper"


# ──────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────

def _merge_dual(left_path: str, right_path: str, out_path: str, title: str):
    import cv2
    w, h = 960, 540
    imgs = []
    for p, lbl in [(left_path, "LEFT-UPPER"), (right_path, "RIGHT-UPPER")]:
        im = cv2.imread(p) if p and os.path.isfile(p) else np.zeros((h, w, 3), np.uint8)
        im = cv2.resize(im, (w, h))
        cv2.putText(im, lbl, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        imgs.append(im)
    bar = np.zeros((48, w * 2, 3), np.uint8)
    cv2.putText(bar, title[:120], (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imwrite(out_path, np.vstack([bar, np.hstack(imgs)]))


# 物体悬浮在 z=10m 处（高于场景屋顶，背景是天空，不被场景遮挡）
# 注意：这是在 house_double_floor_lower 场景中使用的 hack，
# 把对象放在场景几何体以上 10m 处，相机指向把手，背景干净。
GROUND_XY = (0.0, 0.0)
GROUND_Z = 10.0   # 悬浮高度（"地面"相对于场景的偏移）
FLOAT_Z = GROUND_Z  # 向后兼容旧引用


def _cam_positions(focus: np.ndarray, outward: np.ndarray) -> Dict[str, np.ndarray]:
    up_w = np.array([0.0, 0.0, 1.0])
    perp = np.cross(up_w, outward)
    pn = np.linalg.norm(perp)
    if pn < 1e-6:
        perp = np.cross(np.array([1.0, 0.0, 0.0]), outward)
        pn = np.linalg.norm(perp)
    perp /= pn + 1e-9
    return {
        "left_upper":  focus + outward * CAM_D + perp * CAM_SIDE + up_w * CAM_UP,
        "right_upper": focus + outward * CAM_D - perp * CAM_SIDE + up_w * CAM_UP,
    }


def _resolve_vlm_json(case_dir: str, vlm_json: str, source_view: str) -> str:
    if vlm_json and os.path.isfile(vlm_json):
        return vlm_json
    for name in (
        f"vlm_result_{source_view}.json",
        "vlm_result.json",
        f"vlm_result_{DEFAULT_SOURCE_VIEW}.json",
    ):
        p = os.path.join(case_dir, name)
        if os.path.isfile(p):
            return p
    return vlm_json or os.path.join(case_dir, f"vlm_result_{source_view}.json")


def _bbox_to_pixels(bbox: List, w: int, h: int) -> Tuple[int, int, int, int]:
    """VLM bbox_2d 为 0–1000 归一化坐标时转为像素 [x1,y1,x2,y2]。"""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if max(x0, y0, x1, y1) <= 1000:
        return (
            int(x0 / 1000.0 * w),
            int(y0 / 1000.0 * h),
            int(x1 / 1000.0 * w),
            int(y1 / 1000.0 * h),
        )
    return x0, y0, x1, y1


def _capture_obs(gta, ctx) -> Dict[str, Any]:
    """渲染若干帧后一次性读所有 obs。"""
    import omnigibson as og
    for _ in range(8):
        og.sim.render()
    obs, info = gta.get_obs()
    return obs, info


def _free_mib(path: str) -> float:
    """path 所在文件系统的剩余空间（MiB）；取不到时返回 -1。"""
    import shutil
    probe = os.path.dirname(os.path.abspath(path)) or "."
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free / (1024.0 * 1024.0)
    except OSError:
        return -1.0


_DEFAULT_HARD_RESERVE_MIB = 32 * 1024


def _configured_hard_reserve_mib(value: Optional[float]) -> float:
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return float(_DEFAULT_HARD_RESERVE_MIB)
    raw = os.environ.get(
        "BEHAVIOR_INTERFACE_DISK_HARD_RESERVE_MIB",
        str(_DEFAULT_HARD_RESERVE_MIB),
    )
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(_DEFAULT_HARD_RESERVE_MIB)


def _ensure_write_capacity(path: str, min_free_mib: Optional[float]) -> None:
    """硬保留水位下先回收过期会话；仍不足则在创建临时文件前拒绝写入。"""
    reserve_mib = _configured_hard_reserve_mib(min_free_mib)
    if reserve_mib <= 0:
        return
    before_mib = _free_mib(path)
    if before_mib < 0:
        raise OSError(
            errno.EIO,
            f"无法读取目标文件系统剩余空间，拒绝继续写入 {path}",
            path,
        )
    if before_mib >= reserve_mib:
        return

    gc_error = ""
    try:
        from behavior_interface import agent_runs

        raw_age_h = os.environ.get("BEHAVIOR_INTERFACE_RUNS_MAX_AGE_H", "48")
        try:
            max_age_h = max(0.0, float(raw_age_h))
        except ValueError:
            max_age_h = 48.0
        agent_runs.prune_all_stale_sessions(max_age_h)
    except Exception as exc:
        gc_error = f"; age GC 失败: {type(exc).__name__}: {exc}"

    after_mib = _free_mib(path)
    if after_mib < 0:
        raise OSError(
            errno.EIO,
            f"回收后仍无法读取目标文件系统剩余空间，拒绝继续写入 {path}",
            path,
        )
    if after_mib >= reserve_mib:
        return
    raise OSError(
        errno.ENOSPC,
        "磁盘剩余空间低于写入硬保留水位，"
        f"已尝试回收过期会话：{before_mib:.0f} -> {after_mib:.0f} MiB，"
        f"要求 >= {reserve_mib:.0f} MiB，放弃写入 {path}{gc_error}",
    )


def _unique_temp_path(path: str) -> str:
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or "."
    stem, ext = os.path.splitext(os.path.basename(target))
    fd, tmp = tempfile.mkstemp(
        prefix=f".{stem}.part.", suffix=ext, dir=parent
    )
    try:
        try:
            mode = os.stat(target).st_mode & 0o777
        except OSError:
            mode = 0o664
        os.fchmod(fd, mode)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    return tmp


def _atomic_write(
    path: str,
    writer,
    *,
    min_free_mib: Optional[float] = None,
) -> bool:
    """先写同目录临时文件再 os.replace 改名。

    直接写目标路径时磁盘写满会留下截断文件（线上出现过 depth/seg 的
    "518400 requested and 367584 written"），下游把坏数据当好数据读。改名是
    原子的，因此目标路径只有「不存在」和「完整」两种状态。

    @writer 收到临时路径，返回 False 表示未产出内容。临时文件后缀保持与目标
    一致，cv2 才能按扩展名判断编码格式。
    """
    _ensure_write_capacity(path, min_free_mib)
    tmp = ""
    try:
        tmp = _unique_temp_path(path)
        produced = writer(tmp)
        if produced is False or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            return False
        os.replace(tmp, path)
        tmp = ""
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                f"磁盘空间不足，放弃写入 {path}"
                f"（剩余 {_free_mib(path):.0f} MiB）",
            ) from exc
        raise
    finally:
        if tmp:
            _discard(tmp)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def save_npy_atomic(
    path: str,
    arr: np.ndarray,
    *,
    min_free_mib: Optional[float] = None,
) -> bool:
    """原子落盘 .npy。传文件对象而非路径，避免 np.save 追加扩展名。"""
    def _write(tmp: str) -> bool:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        return True
    return _atomic_write(path, _write, min_free_mib=min_free_mib)


def save_image_atomic(
    path: str,
    bgr: np.ndarray,
    *,
    min_free_mib: Optional[float] = None,
) -> bool:
    """原子落盘图片（入参已是 BGR）。"""
    import cv2
    return _atomic_write(
        path,
        lambda tmp: bool(cv2.imwrite(tmp, bgr)),
        min_free_mib=min_free_mib,
    )


def _save_rgb(obs: Dict, path: str) -> bool:
    import cv2
    arr = rgb_array(obs.get("rgb"))
    if arr is None:
        return False
    return save_image_atomic(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _save_depth(obs: Dict, path_npy: str, path_vis: str) -> Optional[np.ndarray]:
    import cv2
    depth = scalar_image_array(obs.get("depth_linear"), dtype=np.float32)
    if depth is None:
        return None
    if not save_npy_atomic(path_npy, depth):
        return None
    # 可视化：截到 [0, max_finite] 映射到 0-255
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if len(finite):
        vmax = float(np.percentile(finite, 99))
    else:
        vmax = 20.0
    vis = np.clip(depth / max(vmax, 0.1), 0.0, 1.0)
    vis = (vis * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    save_image_atomic(path_vis, vis_color)
    return depth


def _save_seg(obs: Dict, info: Dict, path_npy: str, path_vis: str) -> Optional[np.ndarray]:
    seg = obs.get("seg_instance_id")
    if seg is None:
        seg = obs.get("instance_id_segmentation")
    seg = scalar_image_array(seg, dtype=np.int32)
    if seg is None:
        return None
    if not save_npy_atomic(path_npy, seg):
        return None
    # 可视化：随机颜色
    rng = np.random.default_rng(42)
    ids = np.unique(seg)
    color_map = {int(i): rng.integers(0, 256, 3).tolist() for i in ids}
    vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for id_val, color in color_map.items():
        vis[seg == id_val] = color
    save_image_atomic(path_vis, vis)
    return seg


def _build_pointcloud(depth: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray,
                      fl: float, ha: float) -> np.ndarray:
    """由深度图重建世界坐标点云。返回 (N,3) 数组（仅有效像素）。"""
    h, w = depth.shape
    fx = fl / ha * w
    cx, cy = w / 2.0, h / 2.0
    vs_idx, us_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = depth.ravel()
    valid = np.isfinite(d) & (d > 0.05) & (d < 50.0)
    u = us_idx.ravel()[valid]
    v = vs_idx.ravel()[valid]
    d = d[valid]
    x_c = (u - cx) / fx * d
    y_c = -(v - cy) / fx * d
    z_c = -d
    pts_cam = np.stack([x_c, y_c, z_c], axis=1)  # (N,3)
    R = _quat_to_mat(cam_quat)
    pts_world = (R @ pts_cam.T).T + cam_pos
    return pts_world


def _ray_pcd_hit(pts: np.ndarray, origin: np.ndarray, ray_dir: np.ndarray,
                 max_perp: float = 0.15) -> Optional[np.ndarray]:
    """点云射线求交：在 perp < max_perp 的点中取 t 最小（最先被射线命中）的点。
    这等价于取摄像机射线方向上距摄像机最近的交点，避免选到门面内部点。
    """
    v = pts - origin  # (N,3)
    t = v @ ray_dir   # (N,) — t>0 表示在射线正方向
    front = t > 0.05
    if not front.any():
        return None
    pts_f, t_f = pts[front], t[front]
    proj = origin + t_f[:, None] * ray_dir
    perp = np.linalg.norm(pts_f - proj, axis=1)

    # 在满足垂直距离阈值的点中，取 t 最小（摄像机射线最先命中的前表面点）
    mask = perp < max_perp
    if not mask.any():
        # fallback：放宽阈值到 0.5m
        mask = perp < 0.5
        if not mask.any():
            return None
    # argmin t（而非 argmin perp），保证取前表面而非内部点
    t_masked = np.where(mask, t_f, np.inf)
    idx = int(np.argmin(t_masked))
    # 返回射线上的点（非点云顶点），重投影才能与点击像素重合
    t_best = float(t_f[idx])
    return (origin + t_best * ray_dir).astype(np.float64)


def _capture_view_png(gta, obj, obj_pos_t, obj_quat_t, out_dir, vname,
                      cams, focus, out_name, ctx,
                      cam_pos_override=None, focus_override=None) -> str:
    """渲染单视角 PNG。
    cam_pos_override: 若提供，使用保存的相机位置而非重新计算（保证与 init_render 完全一致）。
    focus_override:   若提供，使用保存的 look-at 点（保证相机朝向一致）。
    """
    import omnigibson as og
    cam_pos = cam_pos_override if cam_pos_override is not None else cams[vname]
    look_at  = focus_override  if focus_override  is not None else focus
    _move_cam(gta, np.asarray(cam_pos), np.asarray(look_at))
    obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
    for _ in range(6):
        og.sim.render()
    fpath = os.path.join(out_dir, out_name)
    if not _capture(gta, fpath, ctx):
        ctx.log(f"  [lawn] 截图失败 {out_name}")
    return fpath


# 旧 grasp_3d 流程（兼容）
def _gen_resolve_vlm_3d(gta, obj, obj_pos_t, obj_quat_t, world,
                         cam_meta, u_vlm, v_vlm, w_img, h_img, fl, ha,
                         handle_ref, ctx):
    import omnigibson as og
    cam_pos = np.asarray(cam_meta["cam_pos"])
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"])
    cam_pos, cam_quat = _move_cam(gta, cam_pos, np.asarray(cam_meta["look_at"]))
    for _ in range(6):
        yield world.empty_action()
        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
    for _ in range(2):
        yield world.empty_action()
        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
    p_depth = _depth_to_world_point(gta, u_vlm, v_vlm, cam_pos, cam_quat, w_img, h_img, fl, ha)
    if p_depth is not None:
        ctx.log(f"  [lawn] depth hit={p_depth.round(4).tolist()}")
        return p_depth, "depth_linear"
    origin, direction = _pixel_to_world_ray(cam_pos, cam_quat, u_vlm, v_vlm, w_img, h_img, fl, ha)
    end = origin + direction * 12.0
    for _ in range(6):
        og.sim.render()
        yield world.empty_action()
        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
    from omnigibson.utils.sampling_utils import raytest
    hit = raytest(start_point=origin, end_point=end, only_closest=True)
    if hit.get("hit", False):
        hp = _to_np(hit["position"]).reshape(3)
        ctx.log(f"  [lawn] raytest hit={hp.round(4).tolist()}")
        return hp, "raytest"
    t = float(np.dot(handle_ref - origin, direction))
    if t > 0.05:
        hp = origin + direction * t
        ctx.log(f"  [lawn] fallback ray_handle_depth t={t:.3f}")
        return hp, "ray_handle_depth"
    return None, None


# ──────────────────────────────────────────────
# 主 Skill
# ──────────────────────────────────────────────

@register_skill("vlm_lawn_dual")
def vlm_lawn_dual(ctx, mode: str = "render",
                  category: str = "microwave", model: str = "hjjxmi",
                  out_dir: str = "", view: str = "left_upper",
                  vlm_json: str = "", source_view: str = DEFAULT_SOURCE_VIEW,
                  src_init_dir: str = "") -> Generator:
    import torch as th
    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from behavior_interface.skills.eef import _list_openable_joints, _sample_door_geometry
    from behavior_interface.skills.viz_eef_v2 import _create_gripper, compute_eef_at_grasp

    case_dir = out_dir.strip() or "/tmp/vlm_lawn_case"
    os.makedirs(case_dir, exist_ok=True)
    world = ctx.world
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        ctx.set_result({"ok": False, "error": "gta_view 不可用"})
        yield world.empty_action()
        return

    obj = None
    viz_paths: List[str] = []
    added_modalities: List[str] = []

    try:
        obj_pos_t = th.tensor([GROUND_XY[0], GROUND_XY[1], GROUND_Z], dtype=th.float32)
        obj_quat_t = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)

        # 先尝试清除场景中可能残留的同名对象
        _obj_name = f"vlm_{model}"
        for existing in list(world.env.scene.objects):
            if getattr(existing, "name", None) == _obj_name:
                try:
                    world.env.scene.remove_object(existing)
                    ctx.log(f"  [lawn] 清除残留对象 {_obj_name}")
                except Exception as rm_e:
                    ctx.log(f"  [lawn] 清除残留对象失败: {rm_e}")

        obj = DatasetObject(
            name=_obj_name, category=category, model=model,
            position=obj_pos_t.tolist(), orientation=obj_quat_t.tolist(),
        )
        world.env.scene.add_object(obj)
        # 第一步：先跑 1 帧注册物体（同时固定位置防止重力落下），读取 AABB 确定底面偏移
        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
        yield world.empty_action()
        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)  # 物理步后重设
        try:
            _aabb_lo, _aabb_hi = obj.aabb
            ctx.log(f"  [lawn] AABB: lo={[round(float(x),3) for x in _aabb_lo]} hi={[round(float(x),3) for x in _aabb_hi]}")
            # 底面 z 相对于放置高度 GROUND_Z 的偏移（负=底面在 GROUND_Z 以下）
            _z_bottom_offset = float(_aabb_lo[2]) - GROUND_Z
            # 调整使底面恰好在 GROUND_Z 处
            _z_target = GROUND_Z - _z_bottom_offset
        except Exception as _aabb_err:
            ctx.log(f"  [lawn] 无法读取 AABB: {_aabb_err}，使用默认 FLOAT_Z")
            _z_target = GROUND_Z
        obj_pos_t = th.tensor([GROUND_XY[0], GROUND_XY[1], _z_target], dtype=th.float32)
        ctx.log(f"  [lawn] AABB z 调整: {GROUND_Z:.1f}→{_z_target:.3f}（底面对齐 z={GROUND_Z}）")
        # 第二步：再跑 7 帧让物体稳定在正确位置（每帧都固定位置）
        for _ in range(7):
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

        j_list = _list_openable_joints(obj)
        j = None
        jdir = None
        child = None
        geom_mode = "hinge"
        _scene_closed_q = 0.0

        if j_list:
            j, jdir, child = j_list[0]
            hold = [(obj, obj_pos_t, obj_quat_t)]
            geom = yield from _hold_gen(_sample_door_geometry(ctx, obj, j, jdir, child), hold)
            if geom is None:
                ctx.set_result({"ok": False, "error": "几何失败"})
                yield world.empty_action()
                return
            _scene_closed_q = float(geom["closed_q"])
            try:
                j.set_pos(_scene_closed_q)
            except Exception as _je:
                ctx.log(f"  [lawn] j.set_pos(closed_q) 失败: {_je}")
            geom["mode"] = "hinge"
        else:
            geom_mode = "grasp"
            ctx.log(f"  [lawn] 无铰链 → grasp 模式 ({category}/{model})")
            try:
                _aabb_lo, _aabb_hi = obj.aabb
                lo = np.asarray(_aabb_lo, dtype=np.float64)
                hi = np.asarray(_aabb_hi, dtype=np.float64)
            except Exception:
                lo = hi = np.zeros(3, dtype=np.float64)
            center = (lo + hi) / 2.0
            geom = {
                "mode": "grasp",
                "center_world": center.tolist(),
                "handle_closed_world": center.tolist(),
                "outward_normal_world": [0.0, 1.0, 0.0],
                "axis_world": [0.0, 0.0, 1.0],
            }

        obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

        outward = np.asarray(geom["outward_normal_world"], dtype=np.float64)
        outward /= np.linalg.norm(outward) + 1e-9
        handle = np.asarray(geom.get("handle_closed_world", geom.get("center_world", [0, 0, GROUND_Z])), dtype=np.float64)
        focus = handle.copy()
        cams = _cam_positions(focus, outward)
        w_img = int(getattr(gta, "image_width", 1280))
        h_img = int(getattr(gta, "image_height", 720))
        fl = float(getattr(gta, "focal_length", 17.0))
        ha = float(getattr(gta, "horizontal_aperture", 20.995))

        # ══════════════════════════════════════════
        # render：双视角空白 RGB（视角间不 yield）
        # ══════════════════════════════════════════
        if mode == "render":
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

            saved: Dict[str, str] = {}
            meta_paths: Dict[str, str] = {}
            for vname in VIEWS:
                cam_pos, cam_quat = _move_cam(gta, cams[vname], focus)
                obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
                for _ in range(6):
                    og.sim.render()
                fpath = os.path.join(case_dir, f"{vname}.png")
                if not _capture(gta, fpath, ctx):
                    ctx.set_result({"ok": False, "error": f"截图失败 {vname}"})
                    yield world.empty_action()
                    return
                saved[vname] = fpath
                meta = {
                    "view": vname, "cam_pos": cam_pos.tolist(),
                    "cam_quat_xyzw": cam_quat.tolist(), "look_at": focus.tolist(),
                    "image_width": w_img, "image_height": h_img,
                    "focal_length": fl, "horizontal_aperture": ha,
                    "outward": outward.tolist(), "handle_geom": handle.tolist(),
                    "ground_xyz": [GROUND_XY[0], GROUND_XY[1], GROUND_Z],
                    "closed_q": _scene_closed_q,
                    "category": category, "model": model,
                }
                mp = os.path.join(case_dir, f"camera_meta_{vname}.json")
                with open(mp, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                meta_paths[vname] = mp
                ctx.log(f"  [lawn] {vname} → {fpath}")

            merged = os.path.join(case_dir, "scene_merged.png")
            _merge_dual(saved["left_upper"], saved["right_upper"], merged,
                        f"{category}/{model}  ground z={GROUND_Z}")
            ctx.set_result({
                "ok": True, "out_dir": case_dir,
                "left_upper": saved["left_upper"], "right_upper": saved["right_upper"],
                "scene_merged": merged, "camera_meta": meta_paths,
            })
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # init_render：渲染 RGB + depth + seg 到 init/
        # ══════════════════════════════════════════
        if mode == "init_render":
            sv = source_view if source_view in VIEWS else DEFAULT_SOURCE_VIEW
            init_dir = os.path.join(case_dir, "init")
            os.makedirs(init_dir, exist_ok=True)

            # 动态添加 depth_linear 和 seg_instance_id
            for mod in ("depth_linear", "seg_instance_id"):
                if mod not in gta.modalities:
                    try:
                        gta.add_modality(mod)
                        added_modalities.append(mod)
                        ctx.log(f"  [lawn] 添加 modality: {mod}")
                    except Exception as e:
                        ctx.log(f"  [lawn] 添加 {mod} 失败: {e}")

            # 移相机到 source_view，稳定后抓图
            cam_pos_sv, cam_quat_sv = _move_cam(gta, cams[sv], focus)
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            for _ in range(10):
                og.sim.render()
            obs, info = gta.get_obs()

            # 保存 RGB
            _save_rgb(obs, os.path.join(init_dir, "rgb.png"))
            ctx.log("  [lawn] init rgb saved")

            # 保存 depth
            depth = _save_depth(obs,
                                os.path.join(init_dir, "depth.npy"),
                                os.path.join(init_dir, "depth_vis.png"))
            if depth is None:
                ctx.log("  [lawn] depth 不可用，尝试使用已知焦距重建")

            # 保存 seg
            seg = _save_seg(obs, info,
                            os.path.join(init_dir, "seg.npy"),
                            os.path.join(init_dir, "seg_vis.png"))
            ctx.log(f"  [lawn] seg available: {seg is not None}")

            # 保存相机元信息（含精确 focus 和 closed_q，保证 pcd_multi_grasp 渲染完全一致）
            meta = {
                "view": sv, "cam_pos": cam_pos_sv.tolist(),
                "cam_quat_xyzw": cam_quat_sv.tolist(), "look_at": focus.tolist(),
                "image_width": w_img, "image_height": h_img,
                "focal_length": fl, "horizontal_aperture": ha,
                "outward": outward.tolist(), "handle_geom": handle.tolist(),
                "ground_xyz": [GROUND_XY[0], GROUND_XY[1], GROUND_Z],
                "obj_pos": obj_pos_t.tolist(),
                "obj_quat": obj_quat_t.tolist(),
                "category": category, "model": model,
                "geom_mode": geom_mode,
                "focus": focus.tolist(),
                "closed_q": _scene_closed_q if geom_mode == "hinge" else None,
            }
            meta_path = os.path.join(init_dir, f"camera_meta_{sv}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # 同时也保存到 case_dir 供后续 mode 兼容，记录每个视角的精确相机位置
            for vname in VIEWS:
                cam_pos_v, cam_quat_v = _move_cam(gta, cams[vname], focus)
                obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
                for _ in range(6):
                    og.sim.render()
                _capture(gta, os.path.join(case_dir, f"{vname}.png"), ctx)
                m2 = dict(meta)
                m2.update({
                    "view": vname,
                    "cam_pos": cam_pos_v.tolist(),
                    "cam_quat_xyzw": cam_quat_v.tolist(),
                    # focus / obj_pos 对所有视角相同，已从 meta 继承
                })
                with open(os.path.join(case_dir, f"camera_meta_{vname}.json"), "w") as f:
                    json.dump(m2, f, indent=2)
                # 同步到 init_dir（方便 pcd_multi_grasp 统一从 init_dir 读取）
                with open(os.path.join(init_dir, f"camera_meta_{vname}.json"), "w") as f:
                    json.dump(m2, f, indent=2)

            ctx.set_result({
                "ok": True, "init_dir": init_dir,
                "has_depth": depth is not None,
                "has_seg": seg is not None,
            })
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # pcd_grasp：点云 + VLM 射线求交 → eef_point/
        # ══════════════════════════════════════════
        if mode == "pcd_grasp":
            sv = source_view if source_view in VIEWS else DEFAULT_SOURCE_VIEW
            init_dir = os.path.join(case_dir, "init")
            eef_point_dir = os.path.join(case_dir, "eef_point")
            os.makedirs(eef_point_dir, exist_ok=True)
            ctx.log(f"  [pcd] 开始 pcd_grasp sv={sv} init_dir={init_dir}")

            # 加载深度
            depth_npy = os.path.join(init_dir, "depth.npy")
            if not os.path.isfile(depth_npy):
                ctx.set_result({"ok": False, "error": f"缺少 {depth_npy}，请先运行 init_render"})
                yield world.empty_action()
                return
            ctx.log("  [pcd] 加载 depth.npy")
            depth = np.load(depth_npy).astype(np.float64)
            ctx.log(f"  [pcd] depth shape={depth.shape} min={float(depth[depth>0].min()) if (depth>0).any() else 0:.2f}")

            # 加载相机元信息
            meta_path = os.path.join(init_dir, f"camera_meta_{sv}.json")
            if not os.path.isfile(meta_path):
                meta_path = os.path.join(case_dir, f"camera_meta_{sv}.json")
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
            ctx.log(f"  [pcd] cam_pos={cam_pos_sv.round(3).tolist()}")

            # 加载 VLM 结果
            vlm_path = _resolve_vlm_json(case_dir, vlm_json, sv)
            if not os.path.isfile(vlm_path):
                ctx.set_result({"ok": False, "error": f"缺少 VLM 结果 {vlm_path}"})
                yield world.empty_action()
                return
            with open(vlm_path, encoding="utf-8") as f:
                vlm_data = json.load(f)
            u_vlm = int(vlm_data["pixel"]["u"])
            v_vlm = int(vlm_data["pixel"]["v"])
            ctx.log(f"  [pcd] VLM pixel=({u_vlm},{v_vlm})")

            # 重建点云
            ctx.log("  [pcd] 重建点云...")
            pts_world = _build_pointcloud(depth, cam_pos_sv, cam_quat_sv, fl_m, ha_m)
            ctx.log(f"  [pcd] 点云总点数: {len(pts_world)}")

            # 过滤：只保留靠近悬浮物体的点（z ≈ FLOAT_Z=10 附近 ±3m）
            z_mask = (pts_world[:, 2] > (GROUND_Z - 0.5)) & (pts_world[:, 2] < (GROUND_Z + 3.5))
            xy_mask = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2) < 3.0
            pts_near = pts_world[z_mask & xy_mask]
            ctx.log(f"  [pcd] 物体附近点数: {len(pts_near)}")

            # VLM 射线
            origin, ray_dir = _pixel_to_world_ray(
                cam_pos_sv, cam_quat_sv, u_vlm, v_vlm, w_img, h_img, fl_m, ha_m)
            ctx.log(f"  [pcd] ray origin={origin.round(3).tolist()} dir={ray_dir.round(3).tolist()}")

            hit_pos = None
            hit_method = "unknown"

            if len(pts_near) >= 5:
                hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=0.15)
                if hit_pos is not None:
                    hit_method = "pcd_ray_intersection"
                    ctx.log(f"  [pcd] hit={hit_pos.round(4).tolist()}")

            if hit_pos is None and len(pts_near) >= 1:
                hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=9999.0)
                if hit_pos is not None:
                    hit_method = "pcd_nearest_fallback"
                    ctx.log(f"  [pcd] pcd_nearest fallback={hit_pos.round(4).tolist()}")

            if hit_pos is None:
                handle_ref_fb = np.asarray(cam_meta.get("handle_geom", handle.tolist()))
                t_fb = float(np.dot(handle_ref_fb - origin, ray_dir))
                if t_fb > 0.05:
                    hit_pos = origin + ray_dir * t_fb
                    hit_method = "ray_handle_depth_fallback"
                    ctx.log(f"  [pcd] final fallback t={t_fb:.3f}")

            if hit_pos is None:
                ctx.set_result({"ok": False, "error": "无法从点云或几何反解 3D 点"})
                yield world.empty_action()
                return

            handle_ref = np.asarray(cam_meta.get("handle_geom", handle.tolist()))
            handle_dist_mm = float(np.linalg.norm(hit_pos - handle_ref) * 1000)
            ctx.log(f"  [pcd] hit={hit_pos.round(3).tolist()} hd={handle_dist_mm:.0f}mm {hit_method}")

            # 红球
            stage = og.sim.stage
            red_path = f"/World/vlm_pcd_red_{model}"
            _make_sphere(stage, red_path, hit_pos, radius=0.024, color=(1.0, 0.0, 0.0))
            viz_paths.append(red_path)

            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

            # 渲染双视角到 eef_point/
            ctx.log("  [pcd] 开始渲染双视角...")
            saved_views: Dict[str, str] = {}
            for vname in VIEWS:
                saved_views[vname] = _capture_view_png(
                    gta, obj, obj_pos_t, obj_quat_t,
                    eef_point_dir, vname, cams, focus, f"{vname}.png", ctx)

            # 2D 对比图
            import cv2
            reproj = _world_to_pixel(cam_pos_sv, cam_quat_sv, hit_pos, w_img, h_img, fl_m, ha_m)
            err_px = float(np.hypot(reproj[0] - u_vlm, reproj[1] - v_vlm)) if reproj else None
            raw_img = cv2.imread(os.path.join(case_dir, f"{sv}.png"))
            compare = raw_img.copy() if raw_img is not None else np.zeros((h_img, w_img, 3), np.uint8)
            cv2.circle(compare, (u_vlm, v_vlm), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            if reproj:
                cv2.circle(compare, reproj, 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(eef_point_dir, f"compare_2d_{sv}.png"), compare)

            result = {
                "ok": True, "mode": "pcd_grasp",
                "hit_world": hit_pos.tolist(), "hit_method": hit_method,
                "handle_geom_world": handle_ref.tolist(),
                "handle_dist_mm": handle_dist_mm,
                "pixel_error": err_px,
                "vlm_pixel": {"u": u_vlm, "v": v_vlm},
                "pcd_size": int(len(pts_near)),
                "eef_point_dir": eef_point_dir,
                "views": saved_views,
            }
            with open(os.path.join(case_dir, "pcd_grasp_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            ctx.set_result(result)
            ctx.log(f"  [pcd] pcd_grasp done → {eef_point_dir}")
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # eef_pose：从 pcd_grasp_result 生成夹爪 → eef_pose/
        # ══════════════════════════════════════════
        if mode == "eef_pose":
            eef_pose_dir = os.path.join(case_dir, "eef_pose")
            os.makedirs(eef_pose_dir, exist_ok=True)

            # 读 pcd_grasp_result 获得 3D 点
            res_path = os.path.join(case_dir, "pcd_grasp_result.json")
            if not os.path.isfile(res_path):
                ctx.set_result({"ok": False, "error": f"缺少 {res_path}，请先运行 pcd_grasp"})
                yield world.empty_action()
                return
            with open(res_path, encoding="utf-8") as f:
                pcd_res = json.load(f)
            hit_pos = np.asarray(pcd_res["hit_world"])

            # 生成 EEF pose
            eef_p = compute_eef_at_grasp(hit_pos, geom)
            if eef_p is None:
                ctx.set_result({"ok": False, "error": "EEF pose 计算失败"})
                yield world.empty_action()
                return

            # 放红球 + 夹爪
            stage = og.sim.stage
            red_path = f"/World/vlm_eef_red_{model}"
            _make_sphere(stage, red_path, hit_pos, radius=0.020, color=(1.0, 0.0, 0.0))
            viz_paths.append(red_path)

            grip_paths = _create_gripper(
                stage, f"/World/vlm_eef_grip_{model}",
                np.asarray(eef_p["pos"]), np.asarray(eef_p["quat"]),
                outward, (1.0, 0.0, 0.0), opacity=0.92)
            viz_paths.extend(grip_paths)

            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)

            # 渲染双视角到 eef_pose/
            saved_views: Dict[str, str] = {}
            for vname in VIEWS:
                saved_views[vname] = _capture_view_png(
                    gta, obj, obj_pos_t, obj_quat_t,
                    eef_pose_dir, vname, cams, focus, f"{vname}.png", ctx)

            result = {
                "ok": True, "mode": "eef_pose",
                "eef_pose": eef_p,
                "hit_world": hit_pos.tolist(),
                "eef_pose_dir": eef_pose_dir,
            }
            with open(os.path.join(case_dir, "eef_pose_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            ctx.set_result(result)
            ctx.log(f"  [lawn] eef_pose done → {eef_pose_dir}")
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # pcd_multi_grasp / pcd_grasp_object
        # ══════════════════════════════════════════
        if mode in ("pcd_multi_grasp", "pcd_grasp_object"):
            _grasp_obj_mode = mode == "pcd_grasp_object"
            _result_name = "pcd_grasp_object_result.json" if _grasp_obj_mode else "pcd_multi_grasp_result.json"
            import cv2, shutil
            import omnigibson.lazy as lazy

            sv = source_view if source_view in VIEWS else DEFAULT_SOURCE_VIEW
            N_SAMPLES = 1   # VLM 直接给点，无需多采样
            SEED = 42
            COLORS_3D = [
                (1.0, 0.0, 0.0),   # 0 红
                (0.0, 1.0, 0.0),   # 1 绿
                (0.0, 0.5, 1.0),   # 2 蓝
                (1.0, 0.85, 0.0),  # 3 黄
                (1.0, 0.0, 1.0),   # 4 紫
            ]
            COLORS_2D_BGR = [
                (0, 0, 255),       # 红
                (0, 255, 0),       # 绿
                (255, 128, 0),     # 蓝
                (0, 210, 255),     # 黄
                (255, 0, 255),     # 紫
            ]

            # ── 确定 init_dir（可由外部指定，否则用 case_dir/init）──
            eff_init_dir = src_init_dir.strip() if src_init_dir.strip() else os.path.join(case_dir, "init")
            depth_npy = os.path.join(eff_init_dir, "depth.npy")
            if not os.path.isfile(depth_npy):
                ctx.set_result({"ok": False, "error": f"缺少 {depth_npy}，请先运行 init_render"})
                yield world.empty_action()
                return

            depth = np.load(depth_npy).astype(np.float64)
            meta_path = os.path.join(eff_init_dir, f"camera_meta_{sv}.json")
            if not os.path.isfile(meta_path):
                meta_path = os.path.join(case_dir, f"camera_meta_{sv}.json")
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
            # ── 从 camera_meta 恢复精确 focus，确保渲染与 init_render 完全一致 ──
            focus_saved = np.asarray(
                cam_meta.get("focus", cam_meta.get("handle_geom", focus.tolist()))
            )
            obj_pos_render = th.tensor(
                cam_meta.get("obj_pos", obj_pos_t.tolist()), dtype=th.float32
            )
            obj_quat_render = th.tensor(
                cam_meta.get("obj_quat", obj_quat_t.tolist()), dtype=th.float32
            )
            # 加载所有视角的保存相机位置（key: vname → np.array cam_pos）
            # 搜索顺序：init_dir → init_dir 的父目录（staging）→ case_dir
            saved_cam_pos: Dict[str, np.ndarray] = {sv: cam_pos_sv}
            _staging_dir = os.path.dirname(eff_init_dir)
            for _vn in VIEWS:
                if _vn == sv:
                    continue
                for _mdir in [eff_init_dir, _staging_dir, case_dir]:
                    _mp = os.path.join(_mdir, f"camera_meta_{_vn}.json")
                    if os.path.isfile(_mp):
                        with open(_mp) as _mf:
                            _mm = json.load(_mf)
                        saved_cam_pos[_vn] = np.asarray(_mm["cam_pos"])
                        ctx.log(f"  [multi] 加载 {_vn} 相机位置 from {_mp}")
                        break
            ctx.log(f"  [multi] init_dir={eff_init_dir} depth={depth.shape} "
                    f"focus_saved={focus_saved.round(3).tolist()} "
                    f"saved_cams={list(saved_cam_pos.keys())}")

            # ── 恢复关节到 closed_q（与 init_render 完全一致的场景状态）──
            # 公共代码已在 _sample_door_geometry 后 set_pos(closed_q)，
            # 但 _sample_door_geometry 内部的 yield 可能让物理再次移动门。
            # 在这里再次显式锁定，确保 pcd 渲染与 init 场景状态一致。
            _pcd_closed_q = float(cam_meta.get("closed_q") or _scene_closed_q)
            _pcd_geom_mode = cam_meta.get("geom_mode", geom_mode)
            if j is not None and _pcd_geom_mode == "hinge":
                try:
                    j.set_pos(_pcd_closed_q)
                    obj.set_position_orientation(position=obj_pos_render, orientation=obj_quat_render)
                except Exception as _re:
                    ctx.log(f"  [multi] 关节复位失败: {_re}")
            else:
                obj.set_position_orientation(position=obj_pos_render, orientation=obj_quat_render)
            # 纯渲染刷新（不推进物理）
            for _ in range(6):
                og.sim.render()
            ctx.log(f"  [multi] geom_mode={_pcd_geom_mode} closed_q={_pcd_closed_q:.4f}，场景已就绪")

            # ── VLM：拿 bbox ──
            vlm_path = _resolve_vlm_json(case_dir, vlm_json, sv)
            if not os.path.isfile(vlm_path):
                ctx.set_result({"ok": False, "error": f"缺少 {vlm_path}"})
                yield world.empty_action()
                return
            with open(vlm_path, encoding="utf-8") as f:
                vlm_data = json.load(f)
            items = vlm_data.get("vlm_items", [])
            if not items:
                ctx.set_result({"ok": False, "error": "VLM 无检测 item"})
                yield world.empty_action()
                return
            # VLM 直接输出 point_2d（优先）或退而使用 bbox_2d 中心
            item0 = items[0]
            if "point_2d" in item0:
                pt_norm = item0["point_2d"]
                pu = int(np.clip(int(pt_norm[0]) / 1000.0 * w_img, 0, w_img - 1))
                pv = int(np.clip(int(pt_norm[1]) / 1000.0 * h_img, 0, h_img - 1))
                ctx.log(f"  [multi] point_2d={pt_norm} → px=({pu},{pv})")
            elif "bbox_2d" in item0:
                x1b, y1b, x2b, y2b = _bbox_to_pixels(item0["bbox_2d"], w_img, h_img)
                pu = int((x1b + x2b) / 2)
                pv = int((y1b + y2b) / 2)
                ctx.log(f"  [multi] bbox_2d fallback → center px=({pu},{pv})")
            else:
                pu, pv = w_img // 2, h_img // 2
                ctx.log(f"  [multi] 无有效 VLM 点，使用图像中心 ({pu},{pv})")
            sample_us = [pu]
            sample_vs = [pv]
            ctx.log(f"  [multi] 直接使用 VLM 点: ({pu},{pv})"
                    + (" [grasp_object: 仅用于 seg]" if _grasp_obj_mode else ""))

            # ── 加载 seg.npy，用分割掩码过滤物体像素 ──
            # 对于关节体（如洗碗机），门(door link)和机身(body link)是不同 instance ID。
            # 因此同时查询图像中心区域 + VLM bbox 区域，将所有频次足够的 ID 合并为物体掩码。
            seg_npy = os.path.join(eff_init_dir, "seg.npy")
            obj_mask_2d = None
            obj_id_used = None
            obj_ids_used: List[int] = []
            if os.path.isfile(seg_npy):
                seg_raw = np.load(seg_npy).astype(np.int32)
                if seg_raw.ndim == 3:
                    seg_raw = seg_raw[..., 0]
                cy_img, cx_img = seg_raw.shape[0] // 2, seg_raw.shape[1] // 2
                ph, pw = seg_raw.shape[0] // 3, seg_raw.shape[1] // 3
                # 中心区域
                center_crop = seg_raw[cy_img - ph:cy_img + ph, cx_img - pw:cx_img + pw]
                # VLM 点周围窗口（±80px，覆盖把手 link）
                win = 80
                pt_crop = seg_raw[
                    max(0, pv - win):min(seg_raw.shape[0], pv + win),
                    max(0, pu - win):min(seg_raw.shape[1], pu + win)
                ]
                combined_ids = np.concatenate([center_crop.ravel(), pt_crop.ravel()])
                ids, counts = np.unique(combined_ids, return_counts=True)
                nonzero = ids != 0
                if nonzero.any():
                    ids_nz = ids[nonzero]
                    counts_nz = counts[nonzero]
                    # 取频次 ≥ 最高频次 5% 的所有 ID（同时覆盖机身和关节门）
                    threshold = max(counts_nz.max() * 0.05, 10)
                    obj_ids_used = [int(i) for i in ids_nz[counts_nz >= threshold]]
                    obj_id_used = int(ids_nz[np.argmax(counts_nz)])  # 主 ID（仅供日志）
                    obj_mask_2d = np.isin(seg_raw, obj_ids_used)
                ctx.log(f"  [multi] seg obj_ids={obj_ids_used} obj_px={obj_mask_2d.sum() if obj_mask_2d is not None else 0}")
            else:
                ctx.log("  [multi] 无 seg.npy，使用全场景深度图")

            # ── 重建点云（seg 掩码过滤，与 gripper_fit 一致）──
            seg_raw = None
            if os.path.isfile(seg_npy):
                seg_raw = np.load(seg_npy).astype(np.int32)
                if seg_raw.ndim == 3:
                    seg_raw = seg_raw[..., 0]
            if obj_mask_2d is not None:
                depth_for_pcd = depth.copy()
                depth_for_pcd[~obj_mask_2d] = 0.0
            else:
                depth_for_pcd = depth
            pts_world = _build_pointcloud(depth_for_pcd, cam_pos_sv, cam_quat_sv, fl_m, ha_m)
            _obj_z = float(focus_saved[2])
            z_mask = (pts_world[:, 2] > (_obj_z - 2.0)) & (pts_world[:, 2] < (_obj_z + 2.0))
            xy_mask = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2) < 3.0
            pts_near = pts_world[z_mask & xy_mask]

            from behavior_interface.skills.plan_grasp_gripper_fit import (
                VIZ_BALL_RADIUS,
                build_object_pointcloud,
                resolve_hit_on_surface,
            )
            pts_obj, _ = build_object_pointcloud(
                depth, seg_raw, cam_pos_sv, cam_quat_sv, fl_m, ha_m, pu, pv,
                hit_ref=focus_saved,
                z_band=(_obj_z - 2.0, _obj_z + 2.0),
            )
            pts_for_fit = pts_obj if len(pts_obj) >= 8 else pts_near
            ctx.log(f"  [multi] 物体点云 fit={len(pts_for_fit)} near={len(pts_near)}")

            # grasp_object：肩膀位置用于可达性过滤
            _shoulder_pos = None
            if _grasp_obj_mode:
                for _arm in ("right", "left"):
                    try:
                        _sh = world.shoulder_pose(arm=_arm)
                        _shoulder_pos = np.array([_sh["x"], _sh["y"], _sh["z"]], dtype=np.float64)
                        ctx.log(f"  [grasp_obj] shoulder({_arm})={_shoulder_pos.round(3).tolist()}")
                        break
                    except Exception as _se:
                        ctx.log(f"  [grasp_obj] shoulder({_arm}) 失败: {_se}")
                if _shoulder_pos is None:
                    _obj_c = pts_for_fit.mean(axis=0) if len(pts_for_fit) else focus_saved
                    _shoulder_pos = _obj_c + np.array([0.0, -0.65, 0.35], dtype=np.float64)
                    ctx.log(f"  [grasp_obj] 虚拟肩膀={_shoulder_pos.round(3).tolist()}")

            hit_positions: List[Optional[np.ndarray]] = []
            hit_methods: List[str] = []
            eef_poses: List[Optional[Dict]] = []
            for si in range(N_SAMPLES):
                u_s, v_s = sample_us[si], sample_vs[si]
                if _grasp_obj_mode:
                    from behavior_interface.skills.plan_grasp_object import (
                        compute_eef_from_pcd_grasp_object,
                    )
                    ep = compute_eef_from_pcd_grasp_object(
                        pts_for_fit, _shoulder_pos, seed=SEED + si, ctx=ctx,
                    )
                    hp = (
                        np.asarray(ep["gap_center"], dtype=np.float64)
                        if ep is not None and ep.get("gap_center") is not None
                        else None
                    )
                    meth = "grasp_object_auto"
                else:
                    hp, meth = resolve_hit_on_surface(
                        u_s, v_s, depth, pts_for_fit,
                        cam_pos_sv, cam_quat_sv, w_img, h_img, fl_m, ha_m,
                    )
                    if hp is not None and (_obj_z - 2.0) < hp[2] < (_obj_z + 2.0):
                        pass
                    elif hp is not None:
                        hp = None
                        meth = "failed_z_range"
                    ep = None
                    if hp is not None and _pcd_geom_mode == "grasp":
                        from behavior_interface.skills.plan_grasp_gripper_fit import (
                            compute_eef_from_pcd_gripper_fit,
                        )
                        ep = compute_eef_from_pcd_gripper_fit(
                            hp, pts_for_fit, cam_pos=cam_pos_sv, cam_quat=cam_quat_sv, ctx=ctx,
                        )
                    elif hp is not None:
                        ep = compute_eef_at_grasp(hp, geom)
                hit_positions.append(hp)
                hit_methods.append(meth)
                eef_poses.append(ep)
                ctx.log(
                    f"  [multi] sample {si}: ({u_s},{v_s}) → "
                    f"{hp.round(3).tolist() if hp is not None else None} [{meth}]"
                )

            # ── vlm_output 子文件夹（纯 cv2，无需 sim）──
            vlm_out_dir = os.path.join(case_dir, "vlm_output")
            os.makedirs(vlm_out_dir, exist_ok=True)
            src_img_path = vlm_data.get("source_image") or ""
            if not src_img_path or not os.path.isfile(src_img_path):
                src_img_path = os.path.join(os.path.dirname(vlm_path), f"{sv}.png")
            src_img = cv2.imread(src_img_path) if os.path.isfile(src_img_path) else None
            if src_img is None:
                src_img = np.zeros((h_img, w_img, 3), np.uint8)
            # point_on_image：在 VLM 输出点处画十字准星
            pt_img = src_img.copy()
            _label = item0.get("label", "?")[:40]
            R = 18
            cv2.circle(pt_img, (pu, pv), R, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(pt_img, (pu - R - 6, pv), (pu + R + 6, pv), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(pt_img, (pu, pv - R - 6), (pu, pv + R + 6), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(pt_img, (pu, pv), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(pt_img, _label, (pu + 24, pv - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(vlm_out_dir, "point_on_image.png"), pt_img)
            # 拷贝 VLM 原始文件
            shutil.copy(vlm_path, os.path.join(vlm_out_dir, "vlm_result.json"))
            marked = vlm_data.get("marked_image", "")
            if marked and os.path.isfile(marked):
                shutil.copy(marked, os.path.join(vlm_out_dir, "vlm_marked.png"))
            shutil.copy(src_img_path, os.path.join(vlm_out_dir, "vlm_input_image.png"))
            with open(os.path.join(vlm_out_dir, "grasp_point.json" if not _grasp_obj_mode else "object_label.json"), "w") as f:
                json.dump({
                    "point_px": {"u": pu, "v": pv},
                    "point_norm": item0.get("point_2d", [int(pu / w_img * 1000), int(pv / h_img * 1000)]),
                    "label": _label,
                    "semantic_only": _grasp_obj_mode,
                }, f, indent=2)

            # ── 诊断图：depth_vis、seg_vis、点云投影 ──
            # 1) depth_vis
            depth_vis_src = os.path.join(eff_init_dir, "depth_vis.png")
            if os.path.isfile(depth_vis_src):
                shutil.copy(depth_vis_src, os.path.join(vlm_out_dir, "depth_vis.png"))
            else:
                finite = depth[np.isfinite(depth) & (depth > 0)]
                vmax = float(np.percentile(finite, 99)) if len(finite) else 20.0
                dv = np.clip(depth / max(vmax, 0.1), 0, 1)
                cv2.imwrite(os.path.join(vlm_out_dir, "depth_vis.png"),
                            cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO))

            # 2) seg_vis
            seg_vis_src = os.path.join(eff_init_dir, "seg_vis.png")
            if os.path.isfile(seg_vis_src):
                shutil.copy(seg_vis_src, os.path.join(vlm_out_dir, "seg_vis.png"))

            # 3) 物体分割掩码覆盖图（绿色高亮物体像素）
            if obj_mask_2d is not None and os.path.isfile(src_img_path):
                mask_vis = src_img.copy()
                overlay = mask_vis.copy()
                overlay[obj_mask_2d] = (overlay[obj_mask_2d].astype(int) + [0, 60, 0]).clip(0, 255).astype(np.uint8)
                mask_vis_out = cv2.addWeighted(mask_vis, 0.5, overlay, 0.5, 0)
                cv2.putText(mask_vis_out, f"seg ids={obj_ids_used}", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vlm_out_dir, "seg_obj_mask.png"), mask_vis_out)

            # 4) 点云投影图：把 pts_near 投影回图像平面并叠加在 rgb 上
            if len(pts_near) > 0 and os.path.isfile(src_img_path):
                pcd_vis = src_img.copy()
                # 将世界坐标点转回相机坐标，再投影到像素
                R = _quat_to_mat(cam_quat_sv)
                pts_cam_frame = (R.T @ (pts_near - cam_pos_sv).T).T
                # 相机坐标系：z_c=-depth, x_c 向右, y_c 向上
                z_c = -pts_cam_frame[:, 2]
                valid_proj = z_c > 0.01
                pts_valid = pts_cam_frame[valid_proj]
                z_valid = z_c[valid_proj]
                fx = fl_m / ha_m * w_img
                u_proj = (pts_valid[:, 0] / z_valid * fx + w_img / 2.0).astype(int)
                v_proj = (-pts_valid[:, 1] / z_valid * fx + h_img / 2.0).astype(int)
                # 按深度着色（近=绿，远=红）
                z_min, z_max = z_valid.min(), z_valid.max()
                z_norm = ((z_valid - z_min) / max(z_max - z_min, 0.01)).clip(0, 1)
                for pi in range(len(u_proj)):
                    up, vp = int(u_proj[pi]), int(v_proj[pi])
                    if 0 <= up < w_img and 0 <= vp < h_img:
                        t = float(z_norm[pi])
                        c_bgr = (int((1 - t) * 255), int((1 - abs(2*t - 1)) * 255), int(t * 255))
                        cv2.circle(pcd_vis, (up, vp), 2, c_bgr, -1)
                # 叠加采样射线端点
                for si in range(N_SAMPLES):
                    cv2.circle(pcd_vis, (int(sample_us[si]), int(sample_vs[si])), 8,
                               COLORS_2D_BGR[si], 2)
                cv2.imwrite(os.path.join(vlm_out_dir, "pcd_proj_vis.png"), pcd_vis)

            ctx.log(f"  [multi] vlm_output → {vlm_out_dir}")

            # ── 阶段A：仅添加球（用于 ball_3d.png 可视化）──
            stage = og.sim.stage
            sphere_paths_all: List[Optional[str]] = []
            grip_paths_all: List[List[str]] = []
            for si in range(N_SAMPLES):
                if hit_positions[si] is None:
                    sphere_paths_all.append(None)
                    continue
                sp_path = f"/World/multi_sphere_{si}_{model}"
                _make_sphere(stage, sp_path, hit_positions[si], radius=VIZ_BALL_RADIUS, color=COLORS_3D[si])
                viz_paths.append(sp_path)
                sphere_paths_all.append(sp_path)

            # ── 统一渲染辅助：使用 init_render 保存的精确相机位置 ──
            def _render_view(vn, out_d, out_name):
                """使用保存的相机参数渲染，物体处于自然状态（不干预关节）。"""
                _capture_view_png(
                    gta, obj, obj_pos_render, obj_quat_render,
                    out_d, vn, cams, focus_saved, out_name, ctx,
                    cam_pos_override=saved_cam_pos.get(vn),
                    focus_override=focus_saved,
                )

            # 纯渲染刷新（不推进物理，防止关节因重力移动）
            obj.set_position_orientation(position=obj_pos_render, orientation=obj_quat_render)
            for _ in range(6):
                og.sim.render()
            ball3d_path = os.path.join(vlm_out_dir, "ball_3d.png")
            _render_view(sv, vlm_out_dir, "ball_3d.png")
            ctx.log(f"  [multi] ball_3d.png → {ball3d_path}")

            # ── 阶段B：添加夹爪 ──
            for si in range(N_SAMPLES):
                gp_list: List[str] = []
                if hit_positions[si] is not None and eef_poses[si] is not None:
                    gp_list = _create_gripper(
                        stage, f"/World/multi_grip_{si}_{model}",
                        np.asarray(eef_poses[si]["pos"]), np.asarray(eef_poses[si]["quat"]),
                        outward, COLORS_3D[si], opacity=0.90)
                    viz_paths.extend(gp_list)
                grip_paths_all.append(gp_list)

            # 纯渲染刷新（不推进物理）
            obj.set_position_orientation(position=obj_pos_render, orientation=obj_quat_render)
            for _ in range(6):
                og.sim.render()

            def _set_vis(prim_path: str, visible: bool):
                """USD 可见性切换。"""
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    lazy.pxr.UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                        "inherited" if visible else "invisible")

            # ── 渲染 all_grasps（全部可见）──
            all_dir = os.path.join(case_dir, "all_grasps")
            os.makedirs(all_dir, exist_ok=True)
            for vname in VIEWS:
                _render_view(vname, all_dir, f"{vname}.png")

            # ── 逐个渲染（隐藏其余）──
            grasp_info: List[Dict] = []
            for si in range(N_SAMPLES):
                if hit_positions[si] is None:
                    grasp_info.append({"idx": si, "ok": False})
                    continue
                g_dir = os.path.join(case_dir, f"grasp_{si}")
                os.makedirs(g_dir, exist_ok=True)

                # 隐藏其余球/夹爪
                for sj in range(N_SAMPLES):
                    if sj != si:
                        if sphere_paths_all[sj]:
                            _set_vis(sphere_paths_all[sj], False)
                        for gp in grip_paths_all[sj]:
                            _set_vis(gp, False)
                # 渲染（使用保存的相机参数）
                for vname in VIEWS:
                    _render_view(vname, g_dir, f"{vname}.png")
                # 恢复
                for sj in range(N_SAMPLES):
                    if sj != si:
                        if sphere_paths_all[sj]:
                            _set_vis(sphere_paths_all[sj], True)
                        for gp in grip_paths_all[sj]:
                            _set_vis(gp, True)

                handle_ref = np.asarray(cam_meta.get("handle_geom", handle.tolist()))
                grasp_info.append({
                    "idx": si,
                    "ok": True,
                    "u": int(sample_us[si]), "v": int(sample_vs[si]),
                    "hit_world": hit_positions[si].tolist(),
                    "hit_method": hit_methods[si],
                    "handle_dist_mm": float(np.linalg.norm(hit_positions[si] - handle_ref) * 1000),
                    "eef_pose": eef_poses[si],
                    "color_3d": list(COLORS_3D[si]),
                    "grasp_dir": g_dir,
                })

            result = {
                "ok": True, "mode": mode,
                "grasp_point_px": {"u": pu, "v": pv},
                "vlm_semantic_only": _grasp_obj_mode,
                "n_samples": N_SAMPLES,
                "grasps": grasp_info,
                "all_grasps_dir": all_dir,
                "vlm_output_dir": vlm_out_dir,
                "out_dir": case_dir,
            }
            with open(os.path.join(case_dir, _result_name), "w") as f:
                json.dump(result, f, indent=2)
            ctx.set_result(result)
            ctx.log(f"  [multi] 完成 → {case_dir}")
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # grasp_3d / eef_viz（旧流程兼容）
        # ══════════════════════════════════════════
        if mode in ("grasp_3d", "eef_viz"):
            sv = source_view if source_view in VIEWS else DEFAULT_SOURCE_VIEW
            vlm_path = _resolve_vlm_json(case_dir, vlm_json, sv)
            if not os.path.isfile(vlm_path):
                ctx.set_result({"ok": False, "error": f"缺少 {vlm_path}"})
                yield world.empty_action()
                return
            meta_path = os.path.join(case_dir, f"camera_meta_{sv}.json")
            if not os.path.isfile(meta_path):
                ctx.set_result({"ok": False, "error": f"缺少 {meta_path}"})
                yield world.empty_action()
                return
            with open(meta_path, encoding="utf-8") as f:
                cam_meta = json.load(f)
            with open(vlm_path, encoding="utf-8") as f:
                vlm = json.load(f)
            u_vlm = int(vlm["pixel"]["u"])
            v_vlm = int(vlm["pixel"]["v"])
            handle_geom = np.asarray(cam_meta.get("handle_geom", handle))
            hit_pos, hit_method = yield from _gen_resolve_vlm_3d(
                gta, obj, obj_pos_t, obj_quat_t, world,
                cam_meta, u_vlm, v_vlm, w_img, h_img, fl, ha, handle_geom, ctx)
            if hit_pos is None:
                ctx.set_result({"ok": False, "error": "深度与射线均未得到有效 3D 点"})
                yield world.empty_action()
                return
            stage = og.sim.stage
            red_path = f"/World/vlm_lawn_red_{model}"
            _make_sphere(stage, red_path, hit_pos, radius=0.024, color=(1.0, 0.0, 0.0))
            viz_paths.append(red_path)
            eef_pose = None
            grip_paths: List[str] = []
            if mode == "eef_viz":
                eef_pose = compute_eef_at_grasp(hit_pos, geom)
                if eef_pose is None:
                    ctx.set_result({"ok": False, "error": "EEF pose 计算失败"})
                    yield world.empty_action()
                    return
                grip_paths = _create_gripper(
                    stage, f"/World/vlm_lawn_grip_{model}",
                    np.asarray(eef_pose["pos"]), np.asarray(eef_pose["quat"]),
                    outward, (1.0, 0.0, 0.0), opacity=0.92)
                viz_paths.extend(grip_paths)
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
            saved_views: Dict[str, str] = {}
            prefix = "eef" if mode == "eef_viz" else "grasp_3d"
            for vname in VIEWS:
                saved_views[vname] = _capture_view_png(
                    gta, obj, obj_pos_t, obj_quat_t, case_dir,
                    vname, cams, focus, f"{prefix}_{vname}.png", ctx)
            import cv2
            cam_pos_sv = np.asarray(cam_meta["cam_pos"])
            cam_quat_sv = np.asarray(cam_meta["cam_quat_xyzw"])
            reproj = _world_to_pixel(cam_pos_sv, cam_quat_sv, hit_pos, w_img, h_img, fl, ha)
            err_px = float(np.hypot(reproj[0] - u_vlm, reproj[1] - v_vlm)) if reproj else None
            img_path = vlm.get("source_image", os.path.join(case_dir, f"{sv}.png"))
            base = cv2.imread(vlm.get("marked_image", "")) or cv2.imread(img_path)
            compare = base.copy() if base is not None else np.zeros((h_img, w_img, 3), np.uint8)
            cv2.circle(compare, (u_vlm, v_vlm), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            if reproj:
                cv2.circle(compare, reproj, 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(case_dir, f"vlm_compare_2d_{sv}.png"), compare)
            result = {
                "ok": True, "mode": mode, "source_view": sv,
                "vlm_pixel": {"u": u_vlm, "v": v_vlm},
                "hit_world": hit_pos.tolist(), "hit_method": hit_method,
                "handle_geom_world": handle_geom.tolist(),
                "handle_dist_mm": float(np.linalg.norm(hit_pos - handle_geom) * 1000),
                "pixel_error": err_px, "views": saved_views, "out_dir": case_dir,
            }
            if eef_pose:
                result["eef_pose"] = eef_pose
            with open(os.path.join(case_dir, f"{mode}_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            ctx.set_result(result)
            ctx.log(f"  [lawn] {mode} pe={err_px} hd={result['handle_dist_mm']:.0f}mm")
            yield world.empty_action()
            return

        # ══════════════════════════════════════════
        # verify（旧单视角，兼容）
        # ══════════════════════════════════════════
        if view not in VIEWS:
            ctx.set_result({"ok": False, "error": f"view 须为 {VIEWS}"})
            yield world.empty_action()
            return
        vlm_path = _resolve_vlm_json(case_dir, vlm_json, view)
        if not os.path.isfile(vlm_path):
            ctx.set_result({"ok": False, "error": f"缺少 {vlm_path}"})
            yield world.empty_action()
            return
        meta_path = os.path.join(case_dir, f"camera_meta_{view}.json")
        if not os.path.isfile(meta_path):
            ctx.set_result({"ok": False, "error": f"缺少 {meta_path}，请先 render"})
            yield world.empty_action()
            return
        with open(meta_path, encoding="utf-8") as f:
            cam_meta = json.load(f)
        with open(vlm_path, encoding="utf-8") as f:
            vlm = json.load(f)
        cam_pos = np.asarray(cam_meta["cam_pos"])
        cam_quat = np.asarray(cam_meta["cam_quat_xyzw"])
        u_vlm = int(vlm["pixel"]["u"])
        v_vlm = int(vlm["pixel"]["v"])
        cam_pos, cam_quat = _move_cam(gta, cam_pos, np.asarray(cam_meta["look_at"]))
        for _ in range(4):
            yield world.empty_action()
            obj.set_position_orientation(position=obj_pos_t, orientation=obj_quat_t)
        hit_pos = _depth_to_world_point(gta, u_vlm, v_vlm, cam_pos, cam_quat, w_img, h_img, fl, ha)
        hit_method = "depth_linear" if hit_pos is not None else None
        if hit_pos is None:
            origin, direction = _pixel_to_world_ray(cam_pos, cam_quat, u_vlm, v_vlm, w_img, h_img, fl, ha)
            end = origin + direction * 8.0
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
        handle_geom = np.asarray(cam_meta.get("handle_geom", handle))
        stage = og.sim.stage
        red_path = f"/World/vlm_lawn_red_{model}_{view}"
        _make_sphere(stage, red_path, hit_pos, radius=0.024, color=(1.0, 0.0, 0.0))
        viz_paths.append(red_path)
        reproj = _world_to_pixel(cam_pos, cam_quat, hit_pos, w_img, h_img, fl, ha)
        err_px = float(np.hypot(reproj[0] - u_vlm, reproj[1] - v_vlm)) if reproj else None
        import cv2
        img_path = vlm.get("source_image", os.path.join(case_dir, f"{view}.png"))
        base = cv2.imread(vlm.get("marked_image", "")) or cv2.imread(img_path)
        compare = base.copy() if base is not None else np.zeros((h_img, w_img, 3), np.uint8)
        cv2.circle(compare, (u_vlm, v_vlm), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if reproj:
            cv2.circle(compare, reproj, 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(case_dir, f"vlm_compare_2d_{view}.png"), compare)
        _capture_view_png(gta, obj, obj_pos_t, obj_quat_t, case_dir,
                          view, cams, focus, f"grasp_3d_{view}.png", ctx)
        result = {
            "ok": True, "view": view,
            "vlm_pixel": {"u": u_vlm, "v": v_vlm}, "hit_world": hit_pos.tolist(),
            "hit_method": hit_method, "handle_geom_world": handle_geom.tolist(),
            "handle_dist_mm": float(np.linalg.norm(hit_pos - handle_geom) * 1000),
            "pixel_error": err_px, "out_dir": case_dir,
        }
        with open(os.path.join(case_dir, f"verify_result_{view}.json"), "w") as f:
            json.dump(result, f, indent=2)
        ctx.set_result(result)

    finally:
        # 移除动态添加的 modality
        for mod in added_modalities:
            try:
                gta.remove_modality(mod)
                ctx.log(f"  [lawn] 移除 modality: {mod}")
            except Exception as e:
                ctx.log(f"  [lawn] 移除 modality {mod} 失败: {e}")
        for p in viz_paths:
            try:
                pr = og.sim.stage.GetPrimAtPath(p)
                if pr.IsValid():
                    og.sim.stage.RemovePrim(p)
            except Exception as e:
                ctx.log(f"  [lawn] 清除 viz prim {p} 失败: {e}")
        if obj is not None:
            try:
                world.env.scene.remove_object(obj)
                ctx.log(f"  [lawn] 对象 {getattr(obj,'name','?')} 已清除")
            except Exception as e:
                ctx.log(f"  [lawn] 清除对象失败: {e}")
        yield world.empty_action()
