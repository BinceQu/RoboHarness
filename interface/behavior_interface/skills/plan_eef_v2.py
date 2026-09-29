"""plan_eef_v2 —— agent 工具 plan_*(img_id, u, v) 的仿真后端。

把 head-cam capture（capture skill 落盘的 img_NNNN：depth/seg/normal + 相机内外参）
桥接到既有 plan_eef_from_session 计算引擎：
  - push_* / grasp_point / grasp_obj / open / close  → 直接复用 plan_eef_from_session
  - press                                            → 沿相机视线（表面内法向）顶压（旧单姿态）
  - press_point                                      → 闭爪爪尖与点选点重合，严格 IK 后按外法向对齐筛选
  - place                                            → 仅手上有物体时，移到支撑面略上方后释放

产出统一记成 plan record（move_NNNN）落盘，返回 plan_id / eef_pose / next_move_world /
exec_sequence / 标注图（红接触点 + 绿 eef→next 向量）data URL，供 exec_move(plan_id) 执行。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill
from behavior_interface.skills.plan_eef_core import (
    GRASP_MODES,
    GRASP_POINT_MODES,
    HINGE_MODES,
    PUSH_MODES,
    PUSH_STEP_M,
)


def _plan_eef_core_api():
    """每次规划时取最新 plan_eef_core（热重载后避免仍用旧 2D 线框实现）。"""
    import importlib
    import behavior_interface.skills.plan_eef_core as pec
    importlib.reload(pec)
    return pec
from behavior_interface.skills.plan_grasp_core import session_dir
from behavior_interface.skills.plan_grasp_gripper_fit import resolve_hit_on_surface
from behavior_interface.skills.vlm_grasp_verify import _pixel_to_world_ray, _world_to_pixel

PRESS_STEP_M = 0.04          # press 沿内法向推进距离
PLACE_CLEARANCE_M = 0.08     # place eef 落在支撑点上方的余量

# v2 直接桥接 plan_eef_from_session 的 mode（其余 press/place 自定义）
_BRIDGE_MODES = set(PUSH_MODES) | set(GRASP_MODES) | set(HINGE_MODES)
_OBJECT_GRASP_MODES = {"grasp_obj", "grasp_obj_filter"}
_RGBD_GRASP_MODE = "grasp_point_filter_rgbd"
_IK_FILTER_MODES = {
    "grasp_obj_filter",
    "grasp_point_filter",
    _RGBD_GRASP_MODE,
    "press_point",
}
_VOLUME_GRASP_MODES = (
    set(_OBJECT_GRASP_MODES) | set(GRASP_POINT_MODES) | {_RGBD_GRASP_MODE}
)
RGBD_BATCH_MAX_POINTS = 16


def _object_root_pos_for_record(world, object_name: str) -> Optional[List[float]]:
    if not object_name:
        return None
    try:
        from behavior_interface.skills.exec_move_v2 import _object_root_pos

        pos = _object_root_pos(world, object_name)
        if pos is None:
            return None
        return np.asarray(pos, dtype=np.float64).reshape(3).tolist()
    except Exception:
        return None


def _selected_pose_ik_from_payload(payload: Dict[str, Any], cand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    plan_audit = payload.get("plan_audit") if isinstance(payload.get("plan_audit"), dict) else {}
    pick = plan_audit.get("pick") if isinstance(plan_audit.get("pick"), dict) else {}
    for src in (
        payload.get("selected_pose_ik"),
        cand.get("selected_pose_ik"),
        meta.get("selected_pose_ik"),
        plan_audit.get("selected_pose_ik"),
        pick.get("selected_pose_ik"),
    ):
        if isinstance(src, dict) and src:
            return src
    return None


def _selected_pose_ik_q_from_payload(payload: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    plan_audit = payload.get("plan_audit") if isinstance(payload.get("plan_audit"), dict) else {}
    pick = plan_audit.get("pick") if isinstance(plan_audit.get("pick"), dict) else {}
    selected = _selected_pose_ik_from_payload(payload, cand) or {}
    q_map: Dict[str, Any] = {}
    for src in (
        payload.get("selected_pose_ik_q"),
        cand.get("selected_pose_ik_q"),
        meta.get("selected_pose_ik_q"),
        plan_audit.get("selected_pose_ik_q"),
        pick.get("selected_pose_ik_q"),
    ):
        if isinstance(src, dict):
            for arm in ("left", "right"):
                if src.get(arm) is not None:
                    q_map[arm] = src[arm]
    if isinstance(selected, dict):
        for arm in ("left", "right"):
            q = selected.get(f"{arm}_q_arm")
            if q is not None:
                q_map.setdefault(arm, q)
    return q_map


def _ik_constraint_summary(selected_pose_ik: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(selected_pose_ik, dict) or not selected_pose_ik:
        return None
    tol = (
        selected_pose_ik.get("hard_constraint")
        or f"pos<={selected_pose_ik.get('pos_tol_mm', 10)}mm && "
           f"ori<={selected_pose_ik.get('ori_tol_deg', 3)}deg"
    )
    return (
        f"pose_ik_hard_constraint ({tol}): "
        f"left={'OK' if selected_pose_ik.get('left_ok') else 'NO'} "
        f"{selected_pose_ik.get('left_pos_mm', '?')}mm/"
        f"{selected_pose_ik.get('left_ori_deg', '?')}deg; "
        f"right={'OK' if selected_pose_ik.get('right_ok') else 'NO'} "
        f"{selected_pose_ik.get('right_pos_mm', '?')}mm/"
        f"{selected_pose_ik.get('right_ori_deg', '?')}deg; "
        f"both={'OK' if selected_pose_ik.get('both_ok') else 'NO'}"
    )


def normalize_rgbd_batch_points(
    points: Any,
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Validate native-pixel batch points after the public coordinate wrapper."""
    if not isinstance(points, list) or not points:
        raise ValueError("points 必须是非空数组")
    if len(points) > RGBD_BATCH_MAX_POINTS:
        raise ValueError(
            f"points 最多 {RGBD_BATCH_MAX_POINTS} 个，当前 {len(points)} 个"
        )
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(points):
        if not isinstance(raw, dict):
            raise ValueError(f"points[{index}] 必须是对象")
        if raw.get("u") is None or raw.get("v") is None:
            raise ValueError(f"points[{index}] 缺少 u/v")
        try:
            u = int(raw["u"])
            v = int(raw["v"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"points[{index}] u/v 必须是整数") from exc
        if u < 0 or v < 0:
            raise ValueError(
                f"points[{index}] u/v 必须非负，收到 ({u},{v})"
            )
        if image_width is not None and u >= int(image_width):
            raise ValueError(
                f"points[{index}].u 超出图像宽度 {image_width}: {u}"
            )
        if image_height is not None and v >= int(image_height):
            raise ValueError(
                f"points[{index}].v 超出图像高度 {image_height}: {v}"
            )
        plan_arm = str(raw.get("plan_arm") or "any").strip().lower()
        if plan_arm not in ("any", "left", "right"):
            raise ValueError(
                f"points[{index}].plan_arm 必须是 any/left/right"
            )
        key = (u, v, plan_arm)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "index": int(len(normalized)),
                "source_index": int(index),
                "u": u,
                "v": v,
                "plan_arm": plan_arm,
            }
        )
    if not normalized:
        raise ValueError("points 去重后为空")
    return normalized


def _rgbd_batch_plan_record(
    *,
    world,
    session_id: str,
    image_id: str,
    plan_id: str,
    point: Dict[str, Any],
    payload: Dict[str, Any],
    render_image_path: Optional[str],
    batch_id: str,
    seed: int,
) -> Dict[str, Any]:
    cand = payload["candidates"][0]
    selected_pose_ik = _selected_pose_ik_from_payload(payload, cand)
    selected_pose_ik_q = _selected_pose_ik_q_from_payload(payload, cand)
    ik_constraint_summary = (
        payload.get("ik_constraint_summary")
        or _ik_constraint_summary(selected_pose_ik)
    )
    ik_solution = (
        payload.get("ik_solution")
        or cand.get("ik_solution")
        or (selected_pose_ik or {}).get("solution")
    )
    return {
        "plan_id": plan_id,
        "session_id": session_id,
        "image_id": image_id,
        "skill": "plan_grasp_point_filter_rgbd",
        "mode": _RGBD_GRASP_MODE,
        "plan_arm": point["plan_arm"],
        "exec_sequence": payload.get("exec_sequence"),
        "target": payload.get("target"),
        "arm": (
            payload.get("recommended_arm")
            or payload.get("arm")
            or cand.get("arm")
            or "right"
        ),
        "recommended_arm": payload.get("recommended_arm"),
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "ik_constraint_summary": ik_constraint_summary,
        "ik_solution": ik_solution,
        "object_name": "",
        "object_root_pos": _object_root_pos_for_record(world, ""),
        "eef_pose": payload.get("eef_pose"),
        "next_move_world": payload.get("next_eef_move"),
        "hit_world": payload.get("hit_world"),
        "candidate_count": payload.get("candidate_count"),
        "candidate": cand,
        "render_image_path": render_image_path,
        "plan_audit": payload.get("plan_audit"),
        "grip_fit": payload.get("grip_fit"),
        "gap_center": payload.get("gap_center"),
        "grasp_vol_cm3": float(payload.get("grasp_vol_cm3") or 0.0),
        "overlap_vol_cm3": float(payload.get("overlap_vol_cm3") or 0.0),
        "inflated_overlap_vol_cm3": float(
            payload.get("inflated_overlap_vol_cm3") or 0.0
        ),
        "overlap_frac": float(payload.get("overlap_frac") or 0.0),
        "gripper_vol_cm3": float(
            (payload.get("grip_fit") or {}).get("gripper_vol_cm3") or 0.0
        ),
        "debug_images": payload.get("debug_images"),
        "seed": int(seed),
        "batch_id": batch_id,
        "batch_index": int(point["index"]),
        "batch_point": dict(point),
    }


# ──────────────────────────────────────────────────────────────────────────
# session 构造：head-cam capture meta → plan_eef_from_session 需要的 session dict
# ──────────────────────────────────────────────────────────────────────────

def _plan_session_dir(session_id: str, image_id: str) -> str:
    # 与 plan_grasp_core.session_dir 对齐：_save_plan_debug_images 会往这里写标注图
    return session_dir(f"{session_id}__{image_id}")


def _build_session(
    session_id: str, image_id: str, arm: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, int, int, float, float]:
    """从 capture 的 img_NNNN meta 构造 plan session（含 init_dir 下的 rgb/depth/seg）。"""
    meta = agent_runs.load_image_meta(session_id, image_id)
    cam = meta.get("camera") or {}
    if not cam:
        raise ValueError(f"image {image_id} 无相机内外参（capture 时未取到）")

    img_dir = agent_runs.images_dir(session_id)
    rgb_src = os.path.join(img_dir, meta.get("rgb", {}).get("head", f"{image_id}.png"))
    depth_src = os.path.join(img_dir, meta.get("modalities", {}).get("depth", f"{image_id}.depth.npy"))
    seg_src = os.path.join(img_dir, meta.get("modalities", {}).get("seg", f"{image_id}.seg.npy"))

    sdir = _plan_session_dir(session_id, image_id)
    init_dir = os.path.join(sdir, "init")
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(init_dir, exist_ok=True)
    if os.path.isfile(rgb_src):
        shutil.copy2(rgb_src, os.path.join(init_dir, "rgb.png"))
    if os.path.isfile(depth_src):
        shutil.copy2(depth_src, os.path.join(init_dir, "depth.npy"))
    if os.path.isfile(seg_src):
        shutil.copy2(seg_src, os.path.join(init_dir, "seg.npy"))

    from behavior_interface.skills.vlm_grasp_verify import _quat_to_mat

    from behavior_interface.head_capture import (
        HEAD_FOCAL_LENGTH,
        HEAD_HORIZONTAL_APERTURE,
        HEAD_IMAGE_HEIGHT,
        HEAD_IMAGE_WIDTH,
    )

    cam_pos = np.asarray(cam["pos"], dtype=np.float64)
    cam_quat = np.asarray(cam["quat"], dtype=np.float64)
    w = int(cam.get("image_width", HEAD_IMAGE_WIDTH))
    h = int(cam.get("image_height", HEAD_IMAGE_HEIGHT))
    fl = float(cam.get("focal_length", HEAD_FOCAL_LENGTH))
    ha = float(cam.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    # USD 相机沿局部 -Z 看；plan 标注时恢复 head 冻结外参重渲染
    cam_fwd = _quat_to_mat(cam_quat) @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    look_at = cam_pos + cam_fwd * 1.5

    cam_meta = {
        "view": "head",
        "cam_pos": cam_pos.tolist(),
        "cam_quat_xyzw": cam_quat.tolist(),
        "look_at": look_at.tolist(),
        "image_width": w,
        "image_height": h,
        "focal_length": fl,
        "horizontal_aperture": ha,
    }
    with open(os.path.join(init_dir, "camera_meta_head.json"), "w", encoding="utf-8") as f:
        json.dump(cam_meta, f, indent=2)

    session: Dict[str, Any] = {
        "session_id": f"{session_id}__{image_id}",
        "object_name": "",
        "grasp_mode": "grasp",
        "view": "head",
        "arm": arm,
        "image_width": w,
        "image_height": h,
        "focal_length": fl,
        "horizontal_aperture": ha,
        "handle_pos": cam_pos.tolist(),       # 占位，下面用 hit 覆盖
        "outward": [0.0, 1.0, 0.0],           # 占位，下面用 (cam→hit) 反向覆盖
        "door_meta": {},
        "object_info": {"input": "", "center": cam_pos.tolist()},
        "image_path": os.path.join(init_dir, "rgb.png"),
        "init_dir": init_dir,
        "has_depth": os.path.isfile(os.path.join(init_dir, "depth.npy")),
        "has_seg": os.path.isfile(os.path.join(init_dir, "seg.npy")),
    }
    return session, cam_pos, cam_quat, w, h, fl, ha


def _apply_object_info_to_session(session: Dict[str, Any], object_name: str, oi: Dict[str, Any]) -> None:
    """把 memory / 仿真解析出的物体信息写入 plan session。"""
    session["object_name"] = object_name
    session["object_info"]["input"] = object_name
    ab = oi.get("aabb") or {}
    if ab.get("min") and ab.get("max"):
        session["object_info"]["aabb_min"] = ab["min"]
        session["object_info"]["aabb_max"] = ab["max"]
        session["object_info"]["aabb"] = ab
    if ab.get("center"):
        session["object_info"]["center"] = ab["center"]
    if oi.get("pcd_centroid_world"):
        session["handle_pos"] = oi["pcd_centroid_world"]
        session["object_info"]["center"] = oi["pcd_centroid_world"]
    elif ab.get("center"):
        session["handle_pos"] = ab["center"]
    if oi.get("hit_world"):
        session["object_info"]["hit_world"] = oi["hit_world"]


def _object_info_from_handle(obj, object_name: str, hit: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """从仿真物体 handle 构建与 mark_object 一致的结构化信息。"""
    from behavior_interface.skills.mark_object_v2 import _gather_object_info, _aabb

    ab = _aabb(obj)
    if hit is None and ab is not None:
        lo, hi = ab
        hit = (lo + hi) / 2.0
    if hit is None:
        try:
            pos, _ = obj.get_position_orientation()
            hit = np.asarray(pos, dtype=np.float64).reshape(3)
        except Exception:
            hit = np.zeros(3, dtype=np.float64)
    return _gather_object_info(obj, object_name, np.asarray(hit, dtype=np.float64), None)


def _bind_object_from_uv_click(
    world,
    session: Dict[str, Any],
    hit: np.ndarray,
    *,
    ctx=None,
    tool_label: str = "plan",
) -> Tuple[bool, Optional[str]]:
    """有点选反解 hit 时，按几何（AABB 包含 hit）绑定目标物体，不用 memory 陈旧 mark。"""
    from behavior_interface.skills.mark_object_v2 import _find_object_at

    hit = np.asarray(hit, dtype=np.float64).reshape(3)
    bddl_name, obj = _find_object_at(world, hit)
    if bddl_name is None or obj is None:
        return False, (
            "点击处附近未找到物体；请对准目标再点选，或先 mark_object / 填 object_name"
        )
    oi = _object_info_from_handle(obj, bddl_name, hit)
    _apply_object_info_to_session(session, bddl_name, oi)
    session["object_info"]["hit_world"] = hit.tolist()
    if ctx is not None:
        ctx.log(
            f"  [{tool_label}] 点选绑定 object={bddl_name!r} "
            f"hit={hit.round(3).tolist()}"
        )
    return True, None


def _sync_session_object_target(
    session_id: str,
    session: Dict[str, Any],
    object_name: str = "",
    world=None,
) -> None:
    """按显式 object_name 绑定规划目标；未指定时沿用 memory 最后一次 mark。

    显式 object_name 必须优先读当前仿真世界，而不是 memory。memory 里的 mark/center
    可能来自旧 task 或物体被推动前的位置；再用 capture 时冻结的相机外参投影会直接点到图外。
    """
    name = (object_name or "").strip()
    mem_path = os.path.join(agent_runs.run_dir(session_id), "memory.json")
    mem: Dict[str, Any] = {"objects": {}, "marks": []}
    if os.path.isfile(mem_path):
        try:
            with open(mem_path, encoding="utf-8") as f:
                mem = json.load(f)
        except Exception:
            pass

    if name:
        if world is not None:
            from behavior_interface.skills.grasp import _resolve_object_handle

            obj = _resolve_object_handle(world, name)
            if obj is not None:
                oi = _object_info_from_handle(obj, name)
                _apply_object_info_to_session(session, name, oi)
                return
        oi = (mem.get("objects") or {}).get(name)
        if oi:
            _apply_object_info_to_session(session, name, oi)
            return
        session["object_name"] = name
        session["object_info"]["input"] = name
        return

    marks = mem.get("marks") or []
    if not marks:
        return
    last = marks[-1]
    oi = (mem.get("objects") or {}).get(last.get("bddl_name"))
    if not oi:
        return
    _apply_object_info_to_session(session, last.get("bddl_name", ""), oi)


def _hit_at_uv(session, u, v, cam_pos, cam_quat, w, h, fl, ha):
    """head capture 的 depth/seg 反解 (u,v) 处 3D 表面点（冻结拍照外参，不用 gta）。"""
    from behavior_interface.skills.plan_eef_core import _load_depth_seg
    from behavior_interface.skills.plan_grasp_gripper_fit import build_object_pointcloud

    depth, seg = _load_depth_seg(session)
    hit_ref = np.asarray(
        (session.get("object_info") or {}).get("center", session.get("handle_pos")),
        dtype=np.float64,
    )
    pts, _ = build_object_pointcloud(
        depth, seg, cam_pos, cam_quat, fl, ha, u, v, hit_ref=hit_ref,
    )
    hit, method = resolve_hit_on_surface(
        u, v, depth, pts, cam_pos, cam_quat, w, h, fl, ha, gta=None,
    )
    return hit, method


# ──────────────────────────────────────────────────────────────────────────
# open/close：解析 (u,v) 处铰链物体 → door_meta（hinge/axis/handle/tangent...）
# ──────────────────────────────────────────────────────────────────────────

def _resolve_articulated_at(world, hit: np.ndarray):
    """找包含 hit 点、且带 Open state（铰链）物体；返回 obj 或 None。"""
    from behavior_interface.skills.grasp import _aabb_of

    try:
        from omnigibson.object_states.open_state import Open
    except Exception:
        Open = None

    best = None
    best_d = 1e9
    try:
        objs = list(world.env.scene.objects)
    except Exception:
        objs = []
    for obj in objs:
        if Open is not None and Open not in getattr(obj, "states", {}):
            continue
        aabb = _aabb_of(obj)
        if aabb is None:
            continue
        lo, hi = aabb
        center = (lo + hi) / 2.0
        margin = 0.15
        inside = bool(np.all(hit >= lo - margin) and np.all(hit <= hi + margin))
        if inside:
            d = float(np.linalg.norm(hit - center))
            if d < best_d:
                best_d = d
                best = obj
    return best


def _door_meta_from_candidates(raw: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """从 _sample_open_candidates 输出抽 door_meta / handle_pos / outward（同 plan_eef capture）。"""
    open_cands = [c for c in raw if c.get("meta", {}).get("kind") == "door_arc"]
    best = open_cands[0] if open_cands else raw[0]
    meta = dict(best.get("meta", {}))
    handle_pos = np.asarray(
        meta.get("handle_closed_world", best["eef_target"]["pos"]), dtype=np.float64,
    )
    outward = np.asarray(meta.get("outward_normal_world", [0.0, -1.0, 0.0]), dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    return meta, handle_pos, outward


# ──────────────────────────────────────────────────────────────────────────
# 自定义 mode：press / place（不经 plan_eef_from_session）
# ──────────────────────────────────────────────────────────────────────────

def _build_custom_payload(
    session, mode, u, v, hit, eef_pos, eef_quat, approach, next_world,
    target, exec_seq, gripper_cmd, cam_pos, cam_quat, w, h, fl, ha, arm, ctx,
) -> Dict[str, Any]:
    """press/place：手工拼 candidate + 标注图，结构对齐 plan_eef_from_session 返回。"""
    sid = session["session_id"]
    contact_px = _world_to_pixel(cam_pos, cam_quat, hit, w, h, fl, ha) or (u, v)
    next_delta = (np.asarray(next_world) - np.asarray(eef_pos)).tolist()

    cand = {
        "id": 0,
        "target": target,
        "arm": arm,
        "label": f"plan_eef_v2_{mode}",
        "eef_target": {
            "pos": np.asarray(eef_pos).tolist(),
            "quat": np.asarray(eef_quat).tolist(),
            "approach": np.asarray(approach).tolist(),
            "gripper_cmd": gripper_cmd,
        },
        "next_eef_move": next_delta,
        "reachable": True,
        "reach_reason": "plan_eef_v2",
        "score": 1.0,
        "meta": {
            "plan_eef_v2": True,
            "mode": mode,
            "exec_sequence": exec_seq,
            "next_eef_move_world": np.asarray(next_world).tolist(),
            "hit_method": "depth_surface",
            "pixel": {"u": int(contact_px[0]), "v": int(contact_px[1])},
            "vlm_pixel": {"u": u, "v": v},
        },
    }
    pec = _plan_eef_core_api()
    debug_imgs = pec._save_plan_debug_images(
        session, sid, mode, u, v, int(contact_px[0]), int(contact_px[1]),
        np.asarray(eef_pos), np.asarray(next_world), cam_pos, cam_quat,
        w, h, fl, ha, np.asarray(approach), np.asarray(eef_quat),
        gta=None, world=ctx.world if ctx else None, ctx=ctx,
    )
    return {
        "ok": True,
        "step": "plan",
        "mode": mode,
        "exec_sequence": exec_seq,
        "target": target,
        "session_id": sid,
        "hit_world": np.asarray(hit).tolist(),
        "eef_pose": {
            "pos": np.asarray(eef_pos).tolist(),
            "quat": np.asarray(eef_quat).tolist(),
            "approach": np.asarray(approach).tolist(),
        },
        "next_eef_move": np.asarray(next_world).tolist(),
        "next_eef_move_delta": next_delta,
        "grasp_move_vis": debug_imgs.get("grasp_move_vis"),
        "candidates": [cand],
        "object": {"input": ""},
        "arm": arm,
    }


def _plan_press(session, u, v, hit, cam_pos, cam_quat, w, h, fl, ha, arm, ctx):
    from behavior_interface.skills.plan_grasp_gripper_fit import _mat_to_quat_xyzw

    _, ray = _pixel_to_world_ray(cam_pos, cam_quat, u, v, w, h, fl, ha)
    inward = np.asarray(ray, dtype=np.float64)
    inward /= np.linalg.norm(inward) + 1e-9
    eef_pos = np.asarray(hit, dtype=np.float64)
    next_world = eef_pos + inward * PRESS_STEP_M
    # eef 朝向：Z 轴（approach）沿 inward
    z = inward
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    eef_quat = _mat_to_quat_xyzw(R)
    return _build_custom_payload(
        session, "press", u, v, hit, eef_pos, eef_quat, inward, next_world,
        target="push", exec_seq="close_then_move", gripper_cmd=-1.0,
        cam_pos=cam_pos, cam_quat=cam_quat, w=w, h=h, fl=fl, ha=ha, arm=arm, ctx=ctx,
    )


def _plan_press_point(
    session,
    u,
    v,
    hit,
    hit_method,
    cam_pos,
    cam_quat,
    w,
    h,
    fl,
    ha,
    plan_arm,
    seed,
    ctx,
):
    from behavior_interface.errors import GraspObjPlanningError
    from behavior_interface.skills.plan_press_point import (
        PRESS_POINT_CLEARANCE_M,
        estimate_outward_normal_from_depth,
        plan_press_point_pose,
    )

    depth_path = os.path.join(session.get("init_dir", ""), "depth.npy")
    if not os.path.isfile(depth_path):
        raise GraspObjPlanningError(
            "plan_press_point 缺少冻结 head depth，无法估计目标点外法向"
        )
    depth = np.load(depth_path).astype(np.float64)
    normal_point, outward_normal, normal_audit = estimate_outward_normal_from_depth(
        depth,
        int(u),
        int(v),
        np.asarray(cam_pos, dtype=np.float64),
        np.asarray(cam_quat, dtype=np.float64),
        int(w),
        int(h),
        float(fl),
        float(ha),
    )
    if normal_point is not None:
        normal_audit["hit_point_delta_m"] = float(
            np.linalg.norm(
                np.asarray(hit, dtype=np.float64)
                - np.asarray(normal_point, dtype=np.float64)
            )
        )
    if outward_normal is None:
        raise GraspObjPlanningError(
            "plan_press_point 无法从目标点局部 depth 点云稳定估计外法向 "
            f"(reason={normal_audit.get('reason', 'unknown')})"
        )
    outward_normal = np.asarray(outward_normal, dtype=np.float64).reshape(3)
    session["outward"] = outward_normal.tolist()
    ctx.log(
        "  [plan_press_point] depth-only outward normal="
        f"{outward_normal.round(4).tolist()} "
        f"support={normal_audit.get('selected_support_points')} "
        f"rms_mm={float(normal_audit.get('selected_rms_m', 0.0)) * 1000.0:.3f}"
    )

    best, plan_audit = plan_press_point_pose(
        ctx.world,
        np.asarray(hit, dtype=np.float64),
        plan_arm=plan_arm,
        outward_normal_world=outward_normal,
        seed=int(seed) if seed is not None else 42,
        clearance_m=PRESS_POINT_CLEARANCE_M,
        ctx=ctx,
    )
    plan_audit["surface_normal_estimation"] = normal_audit
    eef_pos = np.asarray(best["eef_pos"], dtype=np.float64).reshape(3)
    eef_quat = np.asarray(best["quat"], dtype=np.float64).reshape(4)
    approach = np.asarray(best["R"], dtype=np.float64).reshape(3, 3)[:, 2]
    press_point = np.asarray(best["press_point_world"], dtype=np.float64).reshape(3)
    fingertip_target = np.asarray(
        best["fingertip_target_world"], dtype=np.float64
    ).reshape(3)
    fingertip_local = np.asarray(
        best["fingertip_local_eef"], dtype=np.float64
    ).reshape(3)
    selected_arm = str(best["recommended_arm"])
    selected_pose_ik = best["selected_pose_ik"]
    selected_pose_ik_q = best["selected_pose_ik_q"]
    move_only_trajectory = best.get("move_only_trajectory")
    safe_eef_pos = np.asarray(
        best["safe_eef_pos"], dtype=np.float64
    ).reshape(3)
    safe_back_m = float(best.get("safe_back_m", 0.10))
    gripper_vector = np.asarray(
        best["gripper_vector_world"], dtype=np.float64
    ).reshape(3)
    normal_gripper_angle_deg = float(best["normal_gripper_angle_deg"])
    exec_seq = "move_only"

    cand = {
        "id": 0,
        "target": "move_eef",
        "arm": selected_arm,
        "label": "plan_press_point",
        "eef_target": {
            "pos": eef_pos.tolist(),
            "quat": eef_quat.tolist(),
            "approach": approach.tolist(),
            "gripper_cmd": -1.0,
        },
        "next_eef_move": [0.0, 0.0, 0.0],
        "reachable": True,
        "reach_reason": "press_point_gpu_endpoint_safe_trajectory",
        "score": -normal_gripper_angle_deg,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "move_only_trajectory": move_only_trajectory,
        "ik_solution": best.get("ik_solution"),
        "meta": {
            "plan_eef_v2": True,
            "mode": "press_point",
            "skill": "plan_press_point",
            "exec_sequence": exec_seq,
            "next_eef_move_world": eef_pos.tolist(),
            "hit_method": hit_method,
            "pixel": {"u": int(u), "v": int(v)},
            "vlm_pixel": {"u": int(u), "v": int(v)},
            "recommended_arm": selected_arm,
            "required_gripper_state": "closed",
            "required_gripper_q": [0.0, 0.0],
            "press_point_world": press_point.tolist(),
            "fingertip_target_world": fingertip_target.tolist(),
            "fingertip_local_eef": fingertip_local.tolist(),
            "fingertip_clearance_m": float(PRESS_POINT_CLEARANCE_M),
            "safe_eef_pos": safe_eef_pos.tolist(),
            "safe_back_m": safe_back_m,
            "outward_normal_world": outward_normal.tolist(),
            "gripper_vector_world": gripper_vector.tolist(),
            "normal_gripper_angle_deg": normal_gripper_angle_deg,
            "selected_pose_ik": selected_pose_ik,
            "selected_pose_ik_q": selected_pose_ik_q,
            "move_only_trajectory": move_only_trajectory,
            "ik_solution": best.get("ik_solution"),
            "plan_audit": plan_audit,
        },
    }

    pec = _plan_eef_core_api()
    debug_imgs = pec._save_plan_debug_images(
        session,
        session["session_id"],
        "press_point",
        int(u),
        int(v),
        int(u),
        int(v),
        eef_pos,
        eef_pos,
        cam_pos,
        cam_quat,
        w,
        h,
        fl,
        ha,
        approach,
        eef_quat,
        hit_world=press_point,
        gap_center_world=fingertip_target,
        gta=None,
        world=ctx.world if ctx else None,
        ctx=ctx,
        gripper_q_m=0.0,
        force_frozen_gripper_overlay=True,
    )
    return {
        "ok": True,
        "step": "plan",
        "mode": "press_point",
        "exec_sequence": exec_seq,
        "target": "move_eef",
        "session_id": session["session_id"],
        "hit_world": press_point.tolist(),
        "hit_method": hit_method,
        "press_point_world": press_point.tolist(),
        "fingertip_target_world": fingertip_target.tolist(),
        "fingertip_local_eef": fingertip_local.tolist(),
        "fingertip_clearance_m": float(PRESS_POINT_CLEARANCE_M),
        "safe_eef_pos": safe_eef_pos.tolist(),
        "safe_back_m": safe_back_m,
        "outward_normal_world": outward_normal.tolist(),
        "gripper_vector_world": gripper_vector.tolist(),
        "normal_gripper_angle_deg": normal_gripper_angle_deg,
        "required_gripper_state": "closed",
        "required_gripper_q": [0.0, 0.0],
        "eef_pose": {
            "pos": eef_pos.tolist(),
            "quat": eef_quat.tolist(),
            "approach": approach.tolist(),
        },
        "next_eef_move": eef_pos.tolist(),
        "next_eef_move_delta": [0.0, 0.0, 0.0],
        "grasp_move_vis": debug_imgs.get("grasp_move_vis"),
        "render_image": debug_imgs.get("grasp_move_vis"),
        "debug_images": debug_imgs,
        "candidates": [cand],
        "arm": selected_arm,
        "recommended_arm": selected_arm,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "move_only_trajectory": move_only_trajectory,
        "ik_solution": best.get("ik_solution"),
        "plan_audit": plan_audit,
        "candidate_count": int(
            ((plan_audit.get("sampling") or {}).get("n_poses") or 0)
        ),
    }


def _plan_place(session, u, v, hit, cam_pos, cam_quat, w, h, fl, ha, arm, ctx):
    from behavior_interface.skills.plan_grasp_gripper_fit import _mat_to_quat_xyzw

    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 自上而下放置
    eef_pos = np.asarray(hit, dtype=np.float64) + np.array([0.0, 0.0, PLACE_CLEARANCE_M])
    next_world = np.asarray(hit, dtype=np.float64) + np.array([0.0, 0.0, 0.02])
    z = approach
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    eef_quat = _mat_to_quat_xyzw(R)
    return _build_custom_payload(
        session, "place", u, v, hit, eef_pos, eef_quat, approach, next_world,
        target="place", exec_seq="move_then_release", gripper_cmd=1.0,
        cam_pos=cam_pos, cam_quat=cam_quat, w=w, h=h, fl=fl, ha=ha, arm=arm, ctx=ctx,
    )


# ──────────────────────────────────────────────────────────────────────────
# skill 主体
# ──────────────────────────────────────────────────────────────────────────

@register_skill(
    "plan_eef_v2",
    description=(
        "agent plan_*(img_id,u,v) 后端：基于 head-cam capture 的 image_id 反投影规划 EEF。"
        "mode ∈ {push_*, grasp_point, grasp_point_filter_rgbd, grasp_obj, "
        "open, close, press, press_point, place}。"
        "落盘 plan record(move_NNNN)，返回 plan_id/eef_pose/next_move/标注图。"
    ),
)
def plan_eef_v2(
    ctx,
    session_id: str,
    image_id: str,
    mode: str,
    u: Optional[int] = None,
    v: Optional[int] = None,
    arm: str = "right",
    seed: Optional[int] = None,
    object_name: str = "",
    plan_arm: str = "any",
) -> Generator:
    from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims
    from behavior_interface.v2_display import mode_to_tool

    world = ctx.world
    clear_plan_viz_prims()
    mode = (mode or "").strip()
    tool_label = mode_to_tool(mode)
    plan_arm = str(plan_arm or "any").strip().lower()
    if plan_arm not in ("left", "right", "any"):
        plan_arm = "any"

    try:
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(session_id, image_id, arm)
    except (FileNotFoundError, ValueError) as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    explicit_obj = (object_name or "").strip()
    has_uv = u is not None and v is not None
    if has_uv:
        u = int(u)
        v = int(v)
    if explicit_obj:
        # Explicit object names are authoritative.  The UI may still send the
        # last clicked pixel; do not let that pixel override the named target.
        u, v = None, None
        has_uv = False

    # 显式 object_name 优先；仅无 object_name 时才允许点选决定目标。
    if explicit_obj:
        _sync_session_object_target(session_id, session, object_name, world=world)
    elif not has_uv:
        _sync_session_object_target(session_id, session, "", world=world)
    if not has_uv and session.get("object_name"):
        from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

        oi = session.get("object_info") or {}
        center = oi.get("center") or session.get("handle_pos")
        if center is not None:
            px = _world_to_pixel(
                cam_pos, cam_quat, np.asarray(center, dtype=np.float64), w, h, fl, ha,
            )
            if px is not None:
                u, v = int(px[0]), int(px[1])
                has_uv = True
                ctx.log(
                    f"  [{tool_label}] object_name 模式：AABB 中心投影像素=({u},{v})"
                )
    if not has_uv:
        ctx.set_result({
            "ok": False,
            "error": "需要 (u,v) 点选，或提供可投影的 object_name",
        })
        yield world.hold_action()
        return

    session["tool_label"] = tool_label
    ctx.log(
        f"[{tool_label}] image={image_id} pixel=({u},{v}) "
        f"object={session.get('object_name')!r} "
        f"cam=head fl={fl:.2f} ha={ha:.3f} res={w}x{h}"
    )

    # 反解 (u,v) 处 3D 表面点（冻结 head 外参 + depth，不用 gta）
    yield world.hold_action()
    if mode == _RGBD_GRASP_MODE:
        from behavior_interface.skills.grasp_point_filter_rgbd import (
            depth_hit_from_pixel,
        )

        depth_path = os.path.join(session["init_dir"], "depth.npy")
        try:
            depth = np.load(depth_path)
            hit, hit_audit = depth_hit_from_pixel(
                depth,
                u=int(u),
                v=int(v),
                camera_pos=cam_pos,
                camera_quat_xyzw=cam_quat,
                focal_length=fl,
                horizontal_aperture=ha,
            )
            hit_method = str(hit_audit["method"])
        except (FileNotFoundError, ValueError) as exc:
            ctx.set_result({
                "ok": False,
                "error": f"{tool_label} depth hit 失败: {exc}",
            })
            yield world.hold_action()
            return
    else:
        hit, hit_method = _hit_at_uv(
            session, u, v, cam_pos, cam_quat, w, h, fl, ha,
        )
    if hit is not None:
        ctx.log(f"  [{tool_label}] 3D hit={np.asarray(hit).round(3).tolist()} method={hit_method}")
    if hit is not None:
        outward = cam_pos - np.asarray(hit, dtype=np.float64)
        n = float(np.linalg.norm(outward))
        outward = (outward / n) if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        session["object_info"]["center"] = np.asarray(hit).tolist()
        session["handle_pos"] = np.asarray(hit).tolist()
        session["outward"] = outward.tolist()

    # grasp / push：无显式 object_name 时，用点击 hit 解析物体（勿沿用 memory）
    if (
        hit is not None
        and has_uv
        and not explicit_obj
        and mode in (set(GRASP_MODES) | set(PUSH_MODES))
    ):
        ok_bind, bind_err = _bind_object_from_uv_click(
            world, session, hit, ctx=ctx, tool_label=tool_label,
        )
        if not ok_bind:
            ctx.set_result({"ok": False, "error": bind_err, "hit_world": hit.tolist()})
            yield world.hold_action()
            return

    if mode in (set(GRASP_MODES) | {_RGBD_GRASP_MODE}) and hit is None:
        ctx.set_result({
            "ok": False,
            "error": (
                f"{tool_label} 无法在 head 图上反解 (u,v) 的 3D 点；"
                "请先 capture，并在 head 副视图点选（勿用 move 后仍点旧图）"
            ),
        })
        yield world.hold_action()
        return

    # ── press / press_point / place：自定义 ──
    if mode == _RGBD_GRASP_MODE:
        from behavior_interface.skills.grasp_point_filter_rgbd import (
            plan_grasp_point_filter_rgbd,
        )

        try:
            payload = plan_grasp_point_filter_rgbd(
                world=world,
                session=session,
                u=int(u),
                v=int(v),
                cam_pos=cam_pos,
                cam_quat=cam_quat,
                w=w,
                h=h,
                fl=fl,
                ha=ha,
                plan_arm=plan_arm,
                seed=int(seed) if seed is not None else 42,
                ctx=ctx,
            )
        except Exception as exc:
            from behavior_interface.errors import GraspObjPlanningError, SkillCancelled

            if isinstance(exc, SkillCancelled):
                raise
            if isinstance(exc, GraspObjPlanningError):
                ctx.set_result({
                    "ok": False,
                    "error": str(exc),
                    "hit_world": np.asarray(hit, dtype=np.float64).tolist(),
                    "plan_arm": plan_arm,
                })
                yield world.hold_action()
                return
            raise

    elif mode == "press_point":
        if hit is None:
            ctx.set_result({
                "ok": False,
                "error": "plan_press_point 无法反解 (u,v) 处 3D 点",
            })
            yield world.hold_action()
            return
        try:
            payload = _plan_press_point(
                session,
                u,
                v,
                hit,
                hit_method,
                cam_pos,
                cam_quat,
                w,
                h,
                fl,
                ha,
                plan_arm,
                seed,
                ctx,
            )
        except Exception as exc:
            from behavior_interface.errors import GraspObjPlanningError, SkillCancelled

            if isinstance(exc, SkillCancelled):
                raise
            if isinstance(exc, GraspObjPlanningError):
                ctx.set_result({
                    "ok": False,
                    "error": str(exc),
                    "hit_world": np.asarray(hit, dtype=np.float64).tolist(),
                    "plan_arm": plan_arm,
                })
                yield world.hold_action()
                return
            raise

    elif mode == "press":
        if hit is None:
            ctx.set_result({"ok": False, "error": "press 无法反解 (u,v) 处 3D 点"})
            yield world.hold_action()
            return
        payload = _plan_press(session, u, v, hit, cam_pos, cam_quat, w, h, fl, ha, arm, ctx)

    elif mode == "place":
        holding = _is_holding(world, arm)
        if not holding:
            ctx.set_result({"ok": False, "error": "place 仅在手上有物体时可用（当前未持物）"})
            yield world.hold_action()
            return
        if hit is None:
            ctx.set_result({"ok": False, "error": "place 无法反解 (u,v) 处支撑面 3D 点"})
            yield world.hold_action()
            return
        payload = _plan_place(session, u, v, hit, cam_pos, cam_quat, w, h, fl, ha, arm, ctx)

    elif mode in _BRIDGE_MODES:
        # open/close：先取铰链 door_meta
        if mode in HINGE_MODES:
            from behavior_interface.skills.eef import _sample_open_candidates

            obj = _resolve_articulated_at(world, np.asarray(hit)) if hit is not None else None
            if obj is None:
                ctx.set_result({"ok": False, "error": f"{mode}：(u,v) 处未找到铰链物体"})
                yield world.hold_action()
                return
            raw = yield from _sample_open_candidates(ctx, obj, opening=(mode == "open"), arm=arm)
            if not raw:
                ctx.set_result({"ok": False, "error": f"{mode}：{obj.name} 无门铰候选"})
                yield world.hold_action()
                return
            door_meta, handle_pos, hinge_outward = _door_meta_from_candidates(raw)
            session["door_meta"] = door_meta
            session["handle_pos"] = handle_pos.tolist()
            session["outward"] = hinge_outward.tolist()
            session["object_name"] = getattr(obj, "name", "")
            session["grasp_mode"] = "hinge"

        yield world.hold_action()
        if ctx.is_cancelled():
            from behavior_interface.errors import SkillCancelled
            raise SkillCancelled("plan 已取消")
        pec = _plan_eef_core_api()
        try:
            payload = pec.plan_eef_from_session(
                session, u, v, mode, gta=None, world=world, ctx=ctx,
                fast_plan=(mode == "open"),
                click_hit=hit,
                grasp_obj_seed=seed if mode in (_OBJECT_GRASP_MODES | set(GRASP_POINT_MODES)) else None,
                plan_arm=plan_arm if mode in _IK_FILTER_MODES else "any",
            )
        except Exception:
            if ctx.is_cancelled():
                from behavior_interface.errors import SkillCancelled
                raise SkillCancelled("plan 已取消") from None
            raise
    else:
        ctx.set_result({
            "ok": False,
            "error": (
                f"未知 mode={mode!r}；支持 "
                "push_*/grasp_point/grasp_point_filter_rgbd/grasp_obj/"
                "open/close/press/press_point/place"
            ),
        })
        yield world.hold_action()
        return

    if payload is None:
        ctx.set_result({
            "ok": False,
            "error": f"{tool_label} 规划返回空结果（内部错误，请查日志）",
        })
        yield world.hold_action()
        return
    if not payload.get("ok"):
        from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims
        clear_plan_viz_prims()
        ctx.log(f"  [{tool_label}] 规划失败: {payload.get('error', 'unknown')}")
        ctx.set_result(payload)
        yield world.hold_action()
        return

    # ── 落盘 plan record（move_NNNN）+ 标注图 data URL ──
    agent_runs.ensure_session(session_id)
    plan_id = agent_runs.next_plan_id(session_id)
    cand = payload["candidates"][0]
    render_path = payload.get("grasp_move_vis") or payload.get("render_image")
    plan_png = agent_runs.plan_path(session_id, plan_id, ".png")
    if render_path and os.path.isfile(render_path):
        shutil.copy2(render_path, plan_png)
    selected_pose_ik = _selected_pose_ik_from_payload(payload, cand)
    selected_pose_ik_q = _selected_pose_ik_q_from_payload(payload, cand)
    ik_constraint_summary = (
        payload.get("ik_constraint_summary")
        or _ik_constraint_summary(selected_pose_ik)
    )
    ik_solution = (
        payload.get("ik_solution")
        or cand.get("ik_solution")
        or ((cand.get("meta") or {}).get("ik_solution") if isinstance(cand.get("meta"), dict) else None)
        or (selected_pose_ik or {}).get("solution")
    )

    record = {
        "plan_id": plan_id,
        "session_id": session_id,
        "image_id": image_id,
        "skill": tool_label,
        "mode": mode,
        "plan_arm": plan_arm if mode in _IK_FILTER_MODES else "any",
        "exec_sequence": payload.get("exec_sequence"),
        "target": payload.get("target"),
        "arm": payload.get("recommended_arm") or payload.get("arm") or cand.get("arm") or arm,
        "recommended_arm": payload.get("recommended_arm"),
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "ik_constraint_summary": ik_constraint_summary,
        "ik_solution": ik_solution,
        "object_name": session.get("object_name") or payload.get("object_name"),
        "object_root_pos": _object_root_pos_for_record(
            world, session.get("object_name") or payload.get("object_name") or ""
        ),
        "eef_pose": payload.get("eef_pose"),
        "next_move_world": payload.get("next_eef_move"),
        "hit_world": payload.get("hit_world"),
        "press_point_world": payload.get("press_point_world"),
        "fingertip_target_world": payload.get("fingertip_target_world"),
        "fingertip_local_eef": payload.get("fingertip_local_eef"),
        "fingertip_clearance_m": payload.get("fingertip_clearance_m"),
        "required_gripper_state": payload.get("required_gripper_state"),
        "required_gripper_q": payload.get("required_gripper_q"),
        "candidate_count": payload.get("candidate_count"),
        "candidate": cand,
        "render_image_path": plan_png if os.path.isfile(plan_png) else render_path,
        "plan_audit": payload.get("plan_audit"),
        "grip_fit": payload.get("grip_fit"),
        "gap_center": payload.get("gap_center"),
        "grasp_vol_cm3": float(payload.get("grasp_vol_cm3") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "overlap_vol_cm3": float(payload.get("overlap_vol_cm3") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "inflated_overlap_vol_cm3": (
            float(payload.get("inflated_overlap_vol_cm3") or 0.0)
            if mode in _VOLUME_GRASP_MODES else None
        ),
        "overlap_frac": float(payload.get("overlap_frac") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "gripper_vol_cm3": (
            float((payload.get("grip_fit") or {}).get("gripper_vol_cm3") or 0.0)
            if mode in _VOLUME_GRASP_MODES else None
        ),
        "debug_images": payload.get("debug_images"),
        "seed": seed if mode in (_VOLUME_GRASP_MODES | {"press_point"}) else None,
    }
    agent_runs.save_plan_record(session_id, plan_id, record)

    render_meta = {}
    if record.get("render_image_path") and os.path.isfile(record["render_image_path"]):
        try:
            import cv2
            im = cv2.imread(record["render_image_path"])
            if im is not None:
                render_meta = {"image_width": int(im.shape[1]), "image_height": int(im.shape[0])}
        except Exception:
            pass

    from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims

    if mode in _VOLUME_GRASP_MODES:
        gv = float(payload.get("grasp_vol_cm3") or 0.0)
        ov = float(payload.get("overlap_vol_cm3") or 0.0)
        fr = float(payload.get("overlap_frac") or 0.0)
        gf = float((payload.get("grip_fit") or {}).get("gripper_vol_cm3") or 0.0)
        if mode == _RGBD_GRASP_MODE:
            inflated = float(payload.get("inflated_overlap_vol_cm3") or 0.0)
            ctx.log(
                f"  [{tool_label}] RGBD grasp_vol={gv:.2f}cm3 "
                f"overlap={ov:.2f}cm3 inflated_overlap={inflated:.2f}cm3 "
                f"({fr * 100:.0f}% of gripper {gf:.1f}cm3)"
            )
        else:
            ctx.log(
                f"  [{tool_label}] v7 grasp_vol={gv:.2f}cm3 "
                f"overlap_vol={ov:.2f}cm3 ({fr * 100:.0f}% of gripper {gf:.1f}cm3)"
            )

    ctx.set_result({
        "ok": True,
        "plan_id": plan_id,
        "tool": tool_label,
        "skill": tool_label,
        "mode": mode,
        "plan_arm": record.get("plan_arm"),
        "arm": record.get("arm"),
        "recommended_arm": record.get("recommended_arm"),
        "selected_pose_ik": record.get("selected_pose_ik"),
        "selected_pose_ik_q": record.get("selected_pose_ik_q"),
        "ik_constraint_summary": record.get("ik_constraint_summary"),
        "ik_solution": record.get("ik_solution"),
        "object_name": record.get("object_name"),
        "exec_sequence": payload.get("exec_sequence"),
        "eef_pose": payload.get("eef_pose"),
        "next_move_world": payload.get("next_eef_move"),
        "hit_world": payload.get("hit_world"),
        "press_point_world": payload.get("press_point_world"),
        "fingertip_target_world": payload.get("fingertip_target_world"),
        "fingertip_local_eef": payload.get("fingertip_local_eef"),
        "fingertip_clearance_m": payload.get("fingertip_clearance_m"),
        "required_gripper_state": payload.get("required_gripper_state"),
        "required_gripper_q": payload.get("required_gripper_q"),
        "candidate_count": payload.get("candidate_count"),
        "render_image": agent_runs.file_to_data_url(record["render_image_path"]),
        "render_image_path": record.get("render_image_path"),
        "plan_audit": payload.get("plan_audit"),
        "grip_fit": payload.get("grip_fit"),
        "gap_center": payload.get("gap_center"),
        "grasp_vol_cm3": float(payload.get("grasp_vol_cm3") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "overlap_vol_cm3": float(payload.get("overlap_vol_cm3") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "inflated_overlap_vol_cm3": (
            float(payload.get("inflated_overlap_vol_cm3") or 0.0)
            if mode in _VOLUME_GRASP_MODES else None
        ),
        "overlap_frac": float(payload.get("overlap_frac") or 0.0) if mode in _VOLUME_GRASP_MODES else None,
        "gripper_vol_cm3": (
            float((payload.get("grip_fit") or {}).get("gripper_vol_cm3") or 0.0)
            if mode in _VOLUME_GRASP_MODES else None
        ),
        "debug_images": payload.get("debug_images"),
        "seed": seed if mode in (_VOLUME_GRASP_MODES | {"press_point"}) else None,
        **render_meta,
    })
    clear_plan_viz_prims()
    yield world.hold_action()


@register_skill(
    "plan_eef_rgbd_batch",
    description=(
        "同一冻结 head RGBD 上批量规划 grasp_point_filter_rgbd；"
        "每点独立 plan_arm / plan_id，并输出同图多夹爪叠影。"
    ),
)
def plan_eef_rgbd_batch(
    ctx,
    session_id: str,
    image_id: str,
    points: List[Dict[str, Any]],
    arm: str = "right",
    seed: int = 42,
) -> Generator:
    from behavior_interface.errors import GraspObjPlanningError, SkillCancelled
    from behavior_interface.skills.grasp_point_filter_rgbd import (
        plan_grasp_point_filter_rgbd,
        prepare_rgbd_grasp_target,
        reconstruct_scene_mesh_from_session,
        render_batch_gripper_overlay,
    )
    from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims

    world = ctx.world
    clear_plan_viz_prims()
    try:
        normalized = normalize_rgbd_batch_points(points)
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(
            session_id,
            image_id,
            arm,
        )
        normalized = normalize_rgbd_batch_points(
            points,
            image_width=int(w),
            image_height=int(h),
        )
    except (FileNotFoundError, ValueError) as exc:
        ctx.set_result({"ok": False, "error": str(exc)})
        yield world.hold_action()
        return

    batch_id = f"rgbd_batch_{int(time.time() * 1000)}"
    session["tool_label"] = "plan_grasp_point_filter_rgbd"
    ctx.log(
        f"[plan_grasp_point_filter_rgbd] batch={batch_id} "
        f"image={image_id} points={len(normalized)}"
    )
    yield world.hold_action()
    if ctx.is_cancelled():
        raise SkillCancelled("RGBD batch plan 已取消")

    try:
        prepared_scene = reconstruct_scene_mesh_from_session(
            session,
            camera_pos=cam_pos,
            camera_quat_xyzw=cam_quat,
            focal_length=fl,
            horizontal_aperture=ha,
        )
    except Exception as exc:
        ctx.set_result(
            {
                "ok": False,
                "error": f"RGBD batch scene reconstruction failed: {exc}",
                "batch_id": batch_id,
            }
        )
        yield world.hold_action()
        return

    depth = prepared_scene[2]
    # Preparation is NumPy/OpenCV CPU work executed beside the simulator.
    # Eight threads per request multiplied across interfaces can starve Kit;
    # retain parallelism but make the host budget explicit and bounded.
    try:
        requested_workers = int(
            os.environ.get("BEHAVIOR_RGBD_POINT_WORKERS", "2")
        )
    except (TypeError, ValueError):
        requested_workers = 2
    analysis_workers = min(max(1, min(requested_workers, 8)), len(normalized))
    prepared_targets: Dict[int, Dict[str, Any]] = {}
    preparation_errors: Dict[int, str] = {}

    def prepare(point: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_rgbd_grasp_target(
            depth,
            u=int(point["u"]),
            v=int(point["v"]),
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w=int(w),
            h=int(h),
            fl=float(fl),
            ha=float(ha),
        )

    with ThreadPoolExecutor(max_workers=analysis_workers) as pool:
        future_to_index = {
            pool.submit(prepare, point): int(point["index"])
            for point in normalized
        }
        for future in as_completed(future_to_index):
            point_index = future_to_index[future]
            try:
                prepared_targets[point_index] = future.result()
            except Exception as exc:
                preparation_errors[point_index] = (
                    f"{type(exc).__name__}: {exc}"
                )

    rows: List[Dict[str, Any]] = []
    for point in normalized:
        point_index = int(point["index"])
        if ctx.is_cancelled():
            raise SkillCancelled("RGBD batch plan 已取消")
        if point_index in preparation_errors:
            rows.append(
                {
                    "point": dict(point),
                    "payload": None,
                    "error": preparation_errors[point_index],
                }
            )
            continue
        point_session = dict(session)
        point_session["session_id"] = (
            f"{session['session_id']}__{batch_id}_{point_index + 1}"
        )
        ctx.log(
            f"  [plan_grasp_point_filter_rgbd] batch point "
            f"#{point_index + 1}/{len(normalized)} "
            f"uv=({point['u']},{point['v']}) "
            f"plan_arm={point['plan_arm']}"
        )
        try:
            payload = plan_grasp_point_filter_rgbd(
                world=world,
                session=point_session,
                u=int(point["u"]),
                v=int(point["v"]),
                cam_pos=cam_pos,
                cam_quat=cam_quat,
                w=int(w),
                h=int(h),
                fl=float(fl),
                ha=float(ha),
                plan_arm=str(point["plan_arm"]),
                seed=int(seed) + point_index,
                ctx=ctx,
                prepared_scene=prepared_scene,
                prepared_target=prepared_targets[point_index],
                render_debug=False,
            )
            rows.append(
                {
                    "point": dict(point),
                    "payload": payload,
                    "error": None,
                }
            )
        except SkillCancelled:
            raise
        except GraspObjPlanningError as exc:
            rows.append(
                {
                    "point": dict(point),
                    "payload": None,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "point": dict(point),
                    "payload": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful_rows = [
        row
        for row in rows
        if isinstance(row.get("payload"), dict)
        and row["payload"].get("ok")
    ]
    for row in successful_rows:
        row["plan_id"] = agent_runs.next_plan_id(session_id)

    composite_path = os.path.join(
        agent_runs.plans_dir(session_id),
        f"{batch_id}.png",
    )
    rendered = render_batch_gripper_overlay(
        session,
        rows,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        w=int(w),
        h=int(h),
        fl=float(fl),
        ha=float(ha),
        output_path=composite_path,
    )

    plan_results: List[Dict[str, Any]] = []
    success_lookup = {id(row): row for row in successful_rows}
    for row in rows:
        point = row["point"]
        if id(row) not in success_lookup:
            plan_results.append(
                {
                    "ok": False,
                    "index": int(point["index"]),
                    "u": int(point["u"]),
                    "v": int(point["v"]),
                    "plan_arm": point["plan_arm"],
                    "error": row.get("error") or "planning failed",
                }
            )
            continue
        payload = row["payload"]
        plan_id = str(row["plan_id"])
        plan_png = agent_runs.plan_path(session_id, plan_id, ".png")
        if rendered and os.path.isfile(rendered):
            shutil.copy2(rendered, plan_png)
        record = _rgbd_batch_plan_record(
            world=world,
            session_id=session_id,
            image_id=image_id,
            plan_id=plan_id,
            point=point,
            payload=payload,
            render_image_path=(
                plan_png if os.path.isfile(plan_png) else rendered
            ),
            batch_id=batch_id,
            seed=int(seed) + int(point["index"]),
        )
        agent_runs.save_plan_record(session_id, plan_id, record)
        grip_fit = payload.get("grip_fit") or {}
        plan_results.append(
            {
                "ok": True,
                "index": int(point["index"]),
                "u": int(point["u"]),
                "v": int(point["v"]),
                "plan_arm": point["plan_arm"],
                "plan_id": plan_id,
                "arm": record["arm"],
                "recommended_arm": record["recommended_arm"],
                "eef_pose": payload.get("eef_pose"),
                "hit_world": payload.get("hit_world"),
                "selected_pose_ik": record["selected_pose_ik"],
                "ik_constraint_summary": record["ik_constraint_summary"],
                "grasp_vol_cm3": float(
                    payload.get("grasp_vol_cm3") or 0.0
                ),
                "overlap_vol_cm3": float(
                    payload.get("overlap_vol_cm3") or 0.0
                ),
                "inflated_overlap_vol_cm3": float(
                    payload.get("inflated_overlap_vol_cm3") or 0.0
                ),
                "gripper_vol_cm3": float(
                    grip_fit.get("gripper_vol_cm3") or 0.0
                ),
            }
        )

    successful = [row for row in plan_results if row.get("ok")]
    failed = [row for row in plan_results if not row.get("ok")]
    first = successful[0] if successful else {}
    result: Dict[str, Any] = {
        "ok": bool(successful),
        "all_ok": bool(successful) and not failed,
        "tool": "plan_grasp_point_filter_rgbd",
        "skill": "plan_grasp_point_filter_rgbd",
        "mode": _RGBD_GRASP_MODE,
        "batch_id": batch_id,
        "image_id": image_id,
        "point_count": int(len(normalized)),
        "success_count": int(len(successful)),
        "failure_count": int(len(failed)),
        "plans": plan_results,
        "plan_ids": [row["plan_id"] for row in successful],
        "plan_id": first.get("plan_id"),
        "arm": first.get("arm"),
        "recommended_arm": first.get("recommended_arm"),
        "eef_pose": first.get("eef_pose"),
        "render_image": (
            agent_runs.file_to_data_url(rendered)
            if rendered and os.path.isfile(rendered)
            else None
        ),
        "render_image_path": rendered,
        "image_width": int(w),
        "image_height": int(h),
        "execution_strategy": {
            "shared_rgbd_reconstruction": True,
            "point_observation_preparation": "parallel",
            "point_observation_workers": int(analysis_workers),
            "ik_fk_planning": "serialized_on_sim_thread",
            "reason": "OG FK temporarily writes robot joints",
        },
    }
    if not successful:
        result["error"] = "所有 RGBD 批量抓取点均规划失败"
    ctx.log(
        f"[plan_grasp_point_filter_rgbd] batch={batch_id} done "
        f"success={len(successful)}/{len(normalized)} "
        f"plans={result['plan_ids']}"
    )
    ctx.set_result(result)
    clear_plan_viz_prims()
    yield world.hold_action()


def _is_holding(world, arm: str) -> bool:
    try:
        ag = getattr(world.robot, "_ag_obj_in_hand", None)
        if isinstance(ag, dict):
            if arm in ("left", "right"):
                return ag.get(arm) is not None
            return any(v is not None for v in ag.values())
    except Exception:
        pass
    return False
