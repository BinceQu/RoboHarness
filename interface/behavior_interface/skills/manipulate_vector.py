"""manipulate 族技能：add_vector_to_point / move_vector_to_vector。

add_vector_to_point:
  模型选点(可多点)→深度反投影到物体表面点→(有分割时)按 seg 过滤局部点云→
  PCA 估计局部坐标系：法向=最小特征值方向(翻转为“指向物体内部/夹爪接近向”),
  主轴=最大特征值方向(杆状物即为轴线)。落盘到 plans/vectors.json，下一轮
  capture 会把它画成局部坐标系(红=法向/接近向, 蓝=主轴双向, 绿=副法向)。

move_vector_to_vector:
  输入 from_vector(待移动)与 to_vector(目标)。先检查 from_vector 表面点所属
  物体是否被夹爪抓住(robot._ag_obj_in_hand + 几何 AABB)，未抓则报错；抓住则
  按 from→to 的刚体位移/旋转，把当前 EEF 位姿刚体变换到目标，驱动手臂到位
  (复用已验证的 _execute_one_eef)，最后更新 from_vector 的位姿。

投影约定与 vlm_grasp_verify 一致（USD 相机沿局部 -Z 看）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill
from behavior_interface.head_capture import (
    HEAD_FOCAL_LENGTH,
    HEAD_HORIZONTAL_APERTURE,
    HEAD_IMAGE_HEIGHT,
    HEAD_IMAGE_WIDTH,
)


# ============================ 向量持久化 ============================
def _vectors_path(session_id: str) -> str:
    return os.path.join(agent_runs.plans_dir(session_id), "vectors.json")


def _load_vectors(session_id: str) -> Dict[str, Any]:
    p = _vectors_path(session_id)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_vectors(session_id: str, vectors: Dict[str, Any]) -> None:
    os.makedirs(agent_runs.plans_dir(session_id), exist_ok=True)
    with open(_vectors_path(session_id), "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2, ensure_ascii=False)


def _next_vector_id(vectors: Dict[str, Any]) -> str:
    n = 1
    while f"vec_{n:04d}" in vectors:
        n += 1
    return f"vec_{n:04d}"


def _depth_modality_name(meta: Dict[str, Any], image_id: str) -> str:
    modalities = meta.get("modalities") or {}
    return str(
        modalities.get("depth")
        or modalities.get("depth_linear")
        or f"{image_id}.depth.npy"
    )


def _seg_modality_name(meta: Dict[str, Any], image_id: str) -> str:
    modalities = meta.get("modalities") or {}
    return str(
        modalities.get("seg")
        or modalities.get("seg_instance_id")
        or f"{image_id}.seg.npy"
    )


# ============================ 几何：投影 ============================
def _quat_to_mat(q) -> np.ndarray:
    """xyzw 四元数→旋转矩阵（与 vlm_grasp_verify 一致）。"""
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _load_image_geometry(session_id: str, image_id: str) -> Dict[str, Any]:
    """加载某帧的相机内外参与 depth/seg（与 plan_eef_v2 读取方式一致）。"""
    meta = agent_runs.load_image_meta(session_id, image_id)
    cam = meta.get("camera") or {}
    if not cam:
        raise ValueError(f"image {image_id} 无相机内外参（capture 时未取到）")
    cam_pos = np.asarray(cam["pos"], dtype=np.float64).reshape(3)
    cam_quat = np.asarray(cam["quat"], dtype=np.float64).reshape(4)  # xyzw
    w = int(cam.get("image_width", HEAD_IMAGE_WIDTH))
    h = int(cam.get("image_height", HEAD_IMAGE_HEIGHT))
    fl = float(cam.get("focal_length", HEAD_FOCAL_LENGTH))
    ha = float(cam.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    img_dir = agent_runs.images_dir(session_id)
    depth_name = _depth_modality_name(meta, image_id)
    seg_name = _seg_modality_name(meta, image_id)
    depth_p = os.path.join(img_dir, depth_name)
    seg_p = os.path.join(img_dir, seg_name)
    depth = np.load(depth_p) if os.path.isfile(depth_p) else None
    if depth is not None and getattr(depth, "ndim", 0) == 3:
        depth = depth[..., 0]
    seg = np.load(seg_p) if os.path.isfile(seg_p) else None
    if seg is not None and getattr(seg, "ndim", 0) == 3:
        seg = seg[..., 0]
    return {
        "cam_pos": cam_pos, "cam_quat": cam_quat,
        "w": w, "h": h, "fl": fl, "ha": ha,
        "depth": depth, "seg": seg,
    }


def _uvd_to_world(u, v, cam_pos, cam_quat, w, h, fl, ha, depth) -> Optional[np.ndarray]:
    """像素+深度反投影到世界坐标（USD -Z 视线，针孔模型）。"""
    if depth is None:
        return None
    ui = int(np.clip(int(round(u)), 0, w - 1))
    vi = int(np.clip(int(round(v)), 0, h - 1))
    d = float(depth[vi, ui])
    if not np.isfinite(d) or d <= 0.01 or d > 50.0:
        return None
    fx = fl / ha * w
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    x_c = (ui - cx) / fx * d
    y_c = -(vi - cy) / fy * d
    p_cam = np.array([x_c, y_c, -d], dtype=np.float64)
    R = _quat_to_mat(cam_quat)
    return np.asarray(cam_pos, dtype=np.float64).reshape(3) + R @ p_cam


def _surface_point_and_frame(pu, pv, cam_pos, cam_quat, w, h, fl, ha, depth, seg):
    """反投影中心点 + (seg 过滤的)局部点云 PCA → (hit, normal, axis, method)。

    normal：表面法向，翻转为“指向物体内部(背离相机)”，即二指夹爪接近向；
    axis：局部主延伸方向(杆状物即轴线)，已与 normal 正交化。
    """
    hit = _uvd_to_world(pu, pv, cam_pos, cam_quat, w, h, fl, ha, depth)
    if hit is None:
        return None, None, None, "depth_invalid"
    half = 6
    cu, cv = int(round(pu)), int(round(pv))
    seg_id = None
    if seg is not None:
        try:
            seg_id = seg[int(np.clip(cv, 0, h - 1)), int(np.clip(cu, 0, w - 1))]
        except Exception:
            seg_id = None
    pts: List[np.ndarray] = []
    pts_all: List[np.ndarray] = []
    for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
            uu, vv = cu + du, cv + dv
            if not (0 <= uu < w and 0 <= vv < h):
                continue
            p = _uvd_to_world(uu, vv, cam_pos, cam_quat, w, h, fl, ha, depth)
            if p is None:
                continue
            pts_all.append(p)
            if seg_id is None:
                continue
            try:
                same = bool(seg[vv, uu] == seg_id)
            except Exception:
                same = True
            if same:
                pts.append(p)
    used_seg = seg_id is not None and len(pts) >= 8
    if not used_seg:
        pts = pts_all
    if len(pts) < 6:
        return hit, None, None, "too_few_points"
    arr = np.asarray(pts, dtype=np.float64)
    c = arr.mean(axis=0)
    q = arr - c
    cov = q.T @ q
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0].astype(np.float64)
    axis = evecs[:, 2].astype(np.float64)
    # 法向翻转为“指向物体内部(背离相机)”=夹爪接近向。
    view = np.asarray(cam_pos, dtype=np.float64).reshape(3) - hit  # 点→相机
    if float(np.dot(normal, view)) > 0.0:
        normal = -normal
    nn = float(np.linalg.norm(normal))
    if nn < 1e-9:
        return hit, None, None, "degenerate_normal"
    normal = normal / nn
    # 主轴与法向正交化并归一化。
    axis = axis - float(np.dot(axis, normal)) * normal
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = tmp - float(np.dot(tmp, normal)) * normal
        an = float(np.linalg.norm(axis))
    axis = axis / max(an, 1e-9)
    return hit, normal, axis, ("pca_depth_seg" if used_seg else "pca_depth")


# ============================ add_vector ============================
@register_skill(
    "manipulate_add_vector_to_point",
    description=(
        "在所选 Qwen3-VL 0..1000 相对坐标点(可多点)反投影出物体表面点，"
        "估计该点局部坐标系并创建一个向量："
        "法向(接近向,垂直于表面指向物体内部)。下一轮 head 图会用坐标系叠加显示"
        "(红=法向/接近向, 蓝=主轴, 绿=副法向)，并返回每个向量的编号。"
    ),
)
def manipulate_add_vector_to_point(
    ctx,
    session_id: str,
    image_id: Optional[str] = None,
    u: Optional[float] = None,
    v: Optional[float] = None,
    points: Optional[List[Any]] = None,
    length_m: float = 0.1,
    **kwargs,
):
    world = ctx.world
    pts: List[Tuple[float, float]] = []
    if points:
        for pr in points:
            try:
                pts.append((float(pr[0]), float(pr[1])))
            except Exception:
                pass
    elif u is not None and v is not None:
        pts.append((float(u), float(v)))
    if not pts:
        ctx.set_result({"ok": False, "error": "未提供选点 (u,v) 或 points"})
        yield world.hold_action()
        return
    explicit_image_id = str(image_id or "").strip()
    resolved_image_id = explicit_image_id or agent_runs.latest_capture_image_id(
        session_id,
        require_camera=True,
        require_depth=True,
    )
    if not resolved_image_id:
        ctx.set_result({
            "ok": False,
            "error": (
                f"session {session_id!r} 中没有包含 camera pose 和 depth 的可用 capture；"
                "请先调用 capture_head_camera 或 wrist capture"
            ),
        })
        yield world.hold_action()
        return
    try:
        geom = _load_image_geometry(session_id, resolved_image_id)
    except Exception as e:
        ctx.set_result({"ok": False, "error": f"加载图像几何失败: {e}"})
        yield world.hold_action()
        return
    if geom["depth"] is None:
        ctx.set_result({"ok": False, "error": f"缺少 depth 数据: {resolved_image_id}"})
        yield world.hold_action()
        return
    cam_pos = geom["cam_pos"]; cam_quat = geom["cam_quat"]
    w = geom["w"]; h = geom["h"]; fl = geom["fl"]; ha = geom["ha"]
    depth = geom["depth"]; seg = geom["seg"]

    vectors = _load_vectors(session_id)
    created: List[Dict[str, Any]] = []
    errors: List[str] = []
    for (pu, pv) in pts:
        try:
            hit, normal, axis, method = _surface_point_and_frame(
                pu, pv, cam_pos, cam_quat, w, h, fl, ha, depth, seg,
            )
        except Exception as e:
            errors.append(f"({pu:.0f},{pv:.0f}) 反解异常: {e}")
            continue
        if hit is None or normal is None:
            errors.append(f"({pu:.0f},{pv:.0f}) 反解失败: {method}")
            continue
        base = np.asarray(hit, dtype=np.float64).reshape(3)
        n_dir = np.asarray(normal, dtype=np.float64).reshape(3)
        a_dir = np.asarray(axis if axis is not None else [0.0, 0.0, 0.0],
                           dtype=np.float64).reshape(3)
        b_dir = np.cross(n_dir, a_dir)
        bn = float(np.linalg.norm(b_dir))
        if bn > 1e-9:
            b_dir = b_dir / bn
        tip = base + n_dir * float(length_m)
        vid = _next_vector_id(vectors)
        rec = {
            "id": vid,
            "base_world": base.round(6).tolist(),
            "dir_world": n_dir.round(6).tolist(),
            "axis_world": a_dir.round(6).tolist(),
            "binormal_world": b_dir.round(6).tolist(),
            "length_m": float(length_m),
            "tip_world": tip.round(6).tolist(),
            "image_id": resolved_image_id,
            "uv": [float(pu), float(pv)],
            "hit_method": method,
        }
        vectors[vid] = rec
        created.append(rec)
    _save_vectors(session_id, vectors)
    ctx.set_result({
        "ok": True,
        "tool": "manipulate_add_vector_to_point",
        "image_id": resolved_image_id,
        "created": [
            {"vector_id": r["id"], "base_world": r["base_world"],
             "dir_world": r["dir_world"], "hit_method": r["hit_method"]}
            for r in created
        ],
        "vector_ids": [r["id"] for r in created],
        "errors": errors,
        "note": "下一轮 head 图将以坐标系叠加(红=法向/接近向, 蓝=主轴, 绿=副法向)",
    })
    yield world.hold_action()
    return


# ============================ move_vector 辅助 ============================
def _get_robot(world):
    r = getattr(world, "robot", None)
    if r is not None:
        return r
    env = getattr(world, "env", None)
    robots = getattr(env, "robots", None) if env is not None else None
    if robots:
        try:
            return robots[0]
        except Exception:
            return None
    return None


def _obj_aabb(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            return None
        lo, hi = aabb
        lo = np.asarray([float(x) for x in np.asarray(lo).reshape(-1)[:3]], dtype=np.float64)
        hi = np.asarray([float(x) for x in np.asarray(hi).reshape(-1)[:3]], dtype=np.float64)
        return lo, hi
    except Exception:
        return None


def _point_in_aabb(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.08) -> bool:
    return bool(np.all(p >= lo - margin) and np.all(p <= hi + margin))


def _held_arm_for_point(world, base_pt: np.ndarray) -> Tuple[Optional[str], Optional[Any], str]:
    """判断 base_pt 所属物体是否被某只手抓住。

    返回 (arm, held_obj, reason)。arm 为 None 表示未抓取。
    优先用 robot._ag_obj_in_hand（assisted grasp 权威）；再用 AABB 命中细化。
    """
    robot = _get_robot(world)
    if robot is None:
        return None, None, "无法获取 robot 句柄"
    ag = getattr(robot, "_ag_obj_in_hand", None)
    if not isinstance(ag, dict):
        return None, None, "机器人无 assisted-grasp 状态"
    held_map = {arm: ag.get(arm) for arm in ("left", "right")}
    holding = {a: o for a, o in held_map.items() if o is not None}
    if not holding:
        return None, None, "双手都没有抓取任何物体"
    for arm, obj in holding.items():
        box = _obj_aabb(obj)
        if box is not None and _point_in_aabb(base_pt, box[0], box[1]):
            return arm, obj, f"{arm} 手抓取的物体 AABB 命中该点"
    if len(holding) == 1:
        arm, obj = next(iter(holding.items()))
        return arm, obj, f"仅 {arm} 手在抓取（AABB 未命中，退化认定）"
    return None, None, "两只手都在抓取，但都不含该表面点"


def _read_eef_pose(world, arm: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取当前 EEF 位姿 (pos[3], quat_xyzw[4])。"""
    for getter in ("eef_pose", "get_eef_pose"):
        fn = getattr(world, getter, None)
        if not callable(fn):
            continue
        res = None
        try:
            res = fn(arm=arm)
        except TypeError:
            try:
                res = fn(arm)
            except Exception:
                res = None
        except Exception:
            res = None
        if res is None:
            continue
        pos = quat = None
        if isinstance(res, dict):
            pos = res.get("pos") or res.get("position")
            quat = res.get("quat") or res.get("quaternion") or res.get("orientation")
            if pos is None and all(k in res for k in ("x", "y", "z")):
                pos = [res["x"], res["y"], res["z"]]
        elif isinstance(res, (tuple, list)) and len(res) >= 2:
            pos, quat = res[0], res[1]
        if pos is not None and quat is not None:
            return (np.asarray(pos, dtype=np.float64).reshape(3),
                    np.asarray(quat, dtype=np.float64).reshape(4))
    return None


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """把单位向量 a 旋到 b 的最小旋转矩阵。"""
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    b = b / max(float(np.linalg.norm(b)), 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - float(np.dot(axis, a)) * a
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (vx @ vx)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    import math
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                         (R[0, 2] - R[2, 0]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
                     (R[1, 0] - R[0, 1]) / s])


# ============================ move_vector ============================
@register_skill(
    "manipulate_move_vector_to_vector",
    description=(
        "把 from_vector(待移动)移动到 to_vector(目标)：先检查 from_vector 表面点所属"
        "物体是否被夹爪抓住，未抓取则报错；抓住则按 from→to 的刚体位移与旋转，把当前"
        "EEF 位姿刚体变换到目标并驱动手臂到位（纯手臂，不动底盘）。"
    ),
)
def manipulate_move_vector_to_vector(
    ctx,
    session_id: str,
    from_vector: str,
    to_vector: str,
    back_m: float = 0.0,
    **kwargs,
):
    world = ctx.world
    vectors = _load_vectors(session_id)
    fv = vectors.get(from_vector)
    tv = vectors.get(to_vector)
    if fv is None or tv is None:
        miss = [k for k, val in (("from_vector", fv), ("to_vector", tv)) if val is None]
        ctx.set_result({"ok": False, "error": f"找不到向量: {', '.join(miss)}"})
        yield world.hold_action()
        return

    f_base = np.asarray(fv["base_world"], dtype=np.float64).reshape(3)
    f_dir = np.asarray(fv["dir_world"], dtype=np.float64).reshape(3)
    t_base = np.asarray(tv["base_world"], dtype=np.float64).reshape(3)
    t_dir = np.asarray(tv["dir_world"], dtype=np.float64).reshape(3)

    # 1) 抓取判定（用户明确要求：未抓取则报错）。
    arm, held_obj, reason = _held_arm_for_point(world, f_base)
    if arm is None:
        ctx.set_result({
            "ok": False,
            "tool": "manipulate_move_vector_to_vector",
            "error": f"from_vector 表面点所属物体未被夹爪抓住，无法移动：{reason}",
            "error_reason_included": True,
        })
        yield world.hold_action()
        return

    # 2) 刚体变换：位移 Δ + 把 f_dir 旋到 t_dir 的最小旋转。
    R_rel = _rotation_between(f_dir, t_dir)
    delta = t_base - f_base

    eef = _read_eef_pose(world, arm)
    if eef is None:
        ctx.set_result({
            "ok": False,
            "tool": "manipulate_move_vector_to_vector",
            "error": f"无法读取 {arm} 手当前 EEF 位姿，无法规划移动",
        })
        yield world.hold_action()
        return
    eef_pos, eef_quat = eef

    # 目标 EEF：绕 f_base 施加 R_rel，再整体平移 delta。
    new_eef_pos = R_rel @ (eef_pos - f_base) + f_base + delta
    R_eef = _quat_to_mat(eef_quat)
    new_R = R_rel @ R_eef
    new_eef_quat = _mat_to_quat_xyzw(new_R)

    # 3) 复用已验证的执行器把 EEF 移到目标位姿（纯手臂，不动底盘）。
    try:
        from behavior_interface.skills.eef import (
            _current_gripper_qpos_cmd,
            _ensure_world_pinned_actions,
            _execute_one_eef,
        )
    except Exception as e:
        ctx.set_result({"ok": False, "error": f"加载执行器失败: {e}"})
        yield world.hold_action()
        return

    _ensure_world_pinned_actions(world)
    try:
        lock_gripper_cmd = _current_gripper_qpos_cmd(world, arm)
    except Exception:
        lock_gripper_cmd = None

    obj_name = ""
    try:
        obj_name = str(getattr(held_obj, "name", "") or "")
    except Exception:
        obj_name = ""

    cand: Dict[str, Any] = {
        "arm": arm,
        "target": "grasp",
        "reachable": True,
        "eef_target": {
            "pos": [float(x) for x in new_eef_pos.tolist()],
            "quat": [float(x) for x in new_eef_quat.tolist()],
        },
        "meta": {"object_name": obj_name},
    }
    last = {
        "ok": True,
        "target": "grasp",
        "candidates": [cand],
        "object": {"input": obj_name},
    }

    try:
        result = yield from _execute_one_eef(
            ctx, last, cand, arm,
            back_m=float(back_m),
            skip_next_move=True,
            skip_grip_close=True,
            stop_after_safe=False,
            lock_gripper_cmd=lock_gripper_cmd,
        )
    except Exception as e:
        import traceback
        ctx.log(f"[manipulate_move] 执行异常: {e}\n{traceback.format_exc()}")
        ctx.set_result({
            "ok": False,
            "tool": "manipulate_move_vector_to_vector",
            "error": f"执行移动异常: {e}",
        })
        yield world.hold_action()
        return

    exec_ok = bool(result.get("ok", True)) if isinstance(result, dict) else True

    # 4) 成功则更新 from_vector 到新位姿（base 平移 delta，dir 旋到 t_dir）。
    if exec_ok:
        new_base = f_base + delta
        L = float(fv.get("length_m", 0.1))
        new_dir = t_dir / max(float(np.linalg.norm(t_dir)), 1e-9)
        new_axis = R_rel @ np.asarray(fv.get("axis_world", [0, 0, 0]), dtype=np.float64).reshape(3)
        an = float(np.linalg.norm(new_axis))
        if an > 1e-9:
            new_axis = new_axis / an
        new_binormal = np.cross(new_dir, new_axis)
        bn = float(np.linalg.norm(new_binormal))
        if bn > 1e-9:
            new_binormal = new_binormal / bn
        fv.update({
            "base_world": new_base.round(6).tolist(),
            "dir_world": new_dir.round(6).tolist(),
            "axis_world": new_axis.round(6).tolist(),
            "binormal_world": new_binormal.round(6).tolist(),
            "tip_world": (new_base + new_dir * L).round(6).tolist(),
        })
        vectors[from_vector] = fv
        _save_vectors(session_id, vectors)

    ctx.set_result({
        "ok": exec_ok,
        "tool": "manipulate_move_vector_to_vector",
        "arm": arm,
        "held_object": obj_name,
        "grasp_reason": reason,
        "from_vector": from_vector,
        "to_vector": to_vector,
        "target_eef_pos": [float(x) for x in new_eef_pos.tolist()],
        "result": result if isinstance(result, dict) else {"raw": str(result)},
        "error": None if exec_ok else (result.get("error") if isinstance(result, dict) else "执行失败"),
    })
    yield world.hold_action()
    return
