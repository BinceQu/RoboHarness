"""Move the base to a clicked floor point without changing body posture."""

from __future__ import annotations

import math
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.move_to import _drive_body_forward
from behavior_interface.skills.move_to_object_v2 import _norm_angle_rad
from behavior_interface.skills.move_to_point_v2 import _sphere_center_from_uv
from behavior_interface.skills.viz_base_path_overlay import (
    BASE_FRONT_OFFSET_M,
    PATH_Y_CENTER_M,
)


_MOVE_BASE_TO_POINT_BUILD = (
    "move_base_to_point_v2_floor_front_reference_direct_xy_ground_guard"
)
_MOVE_BASE_TO_POINT_DEFAULT_POS_TOL_M = 0.12
_BASE_FRONT_REFERENCE_LOCAL_XY = np.array(
    [BASE_FRONT_OFFSET_M, PATH_Y_CENTER_M],
    dtype=np.float64,
)
_FLOOR_CATEGORIES = {
    "floor",
    "floors",
    "flooring",
    "tile_floor",
    "tile_flooring",
    "paver",
    "pavers",
    "cobblestone",
    "lawn",
    "grass",
    "ground",
}


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_aabb(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        lo, hi = obj.aabb
        lo_arr = _to_np(lo).astype(np.float64).reshape(-1)
        hi_arr = _to_np(hi).astype(np.float64).reshape(-1)
        if lo_arr.size >= 3 and hi_arr.size >= 3:
            return lo_arr[:3], hi_arr[:3]
    except Exception:
        pass
    return None


def _iter_scene_objects(world: Any) -> Iterable[Tuple[str, Any]]:
    seen: set[int] = set()
    try:
        scope = getattr(getattr(world.env, "task", None), "object_scope", None) or {}
        for key, ent in scope.items():
            obj = getattr(ent, "unwrapped", ent)
            if obj is not None and id(obj) not in seen:
                seen.add(id(obj))
                yield str(key), obj
    except Exception:
        pass
    try:
        scene = getattr(getattr(world, "env", None), "scene", None)
        objects = getattr(scene, "objects", None) or []
        if isinstance(objects, dict):
            iterator = objects.items()
        else:
            iterator = ((getattr(obj, "name", f"obj_{i}"), obj) for i, obj in enumerate(objects))
        for key, obj in iterator:
            if obj is not None and id(obj) not in seen:
                seen.add(id(obj))
                yield str(key), obj
    except Exception:
        pass


def _category_is_floor(category: str) -> bool:
    cat = str(category or "").lower()
    return cat in _FLOOR_CATEGORIES or cat.startswith("floor") or cat.endswith("_floor")


def _check_hit_on_floor(world: Any, hit: np.ndarray, ground_tol_m: float) -> Dict[str, Any]:
    p = np.asarray(hit, dtype=np.float64).reshape(3)
    tol = max(0.0, float(ground_tol_m))
    candidates: list[Dict[str, Any]] = []
    nearest: Optional[Dict[str, Any]] = None
    for key, obj in _iter_scene_objects(world):
        category = str(getattr(obj, "category", "") or "")
        if not _category_is_floor(category):
            continue
        aabb = _safe_aabb(obj)
        if aabb is None:
            continue
        lo, hi = aabb
        inside_xy = bool(
            lo[0] - tol <= p[0] <= hi[0] + tol
            and lo[1] - tol <= p[1] <= hi[1] + tol
        )
        dx = 0.0 if lo[0] <= p[0] <= hi[0] else min(abs(p[0] - lo[0]), abs(p[0] - hi[0]))
        dy = 0.0 if lo[1] <= p[1] <= hi[1] else min(abs(p[1] - lo[1]), abs(p[1] - hi[1]))
        xy_gap = float(math.hypot(dx, dy))
        top_z = float(hi[2])
        dz_top = float(p[2] - top_z)
        rec = {
            "key": key,
            "name": str(getattr(obj, "name", key)),
            "category": category,
            "aabb_min": [round(float(v), 4) for v in lo.tolist()],
            "aabb_max": [round(float(v), 4) for v in hi.tolist()],
            "top_z": round(top_z, 4),
            "inside_xy": inside_xy,
            "xy_gap_m": round(xy_gap, 4),
            "dz_to_top_m": round(dz_top, 4),
        }
        candidates.append(rec)
        score = xy_gap + abs(dz_top)
        if nearest is None or score < float(nearest.get("_score", 1e9)):
            nearest = dict(rec)
            nearest["_score"] = score
        if inside_xy and abs(dz_top) <= tol:
            return {
                "ok": True,
                "ground_tol_m": float(tol),
                "floor": rec,
                "candidate_count": len(candidates),
            }
    if nearest is not None:
        nearest.pop("_score", None)
    return {
        "ok": False,
        "ground_tol_m": float(tol),
        "nearest_floor": nearest,
        "candidate_count": len(candidates),
        "reason": "hit point is not on a floor AABB top surface within tolerance",
    }


def _base_xy_yaw(world: Any) -> Tuple[np.ndarray, float]:
    pose = world.robot_pose()
    pos = np.asarray(pose.pos, dtype=np.float64).reshape(3)
    return pos[:2].copy(), float(pose.yaw)


def _base_front_reference_world_xy(
    base_xy: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Return the blue path start-line midpoint in world XY."""
    cos_y, sin_y = math.cos(float(yaw)), math.sin(float(yaw))
    offset_x, offset_y = _BASE_FRONT_REFERENCE_LOCAL_XY
    return np.asarray(base_xy, dtype=np.float64).reshape(2) + np.array(
        [
            cos_y * offset_x - sin_y * offset_y,
            sin_y * offset_x + cos_y * offset_y,
        ],
        dtype=np.float64,
    )


def _trunk_q(world: Any) -> list[float]:
    try:
        return [float(x) for x in np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(-1).tolist()]
    except Exception:
        return []


@register_skill(
    "move_base_to_point",
    description=(
        "点选 head 图地面点：反解 3D 必须落在 floor/floors 顶面；"
        "保持当前朝向，沿起始机体系 forward/leftward 向量直接平移，"
        "使蓝色路径起始线中点到达点选位置；"
        "躯干/双臂/夹爪保持当前姿态。"
    ),
)
def move_base_to_point(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    nav_timeout_s: float = 120.0,
    ground_tol_m: float = 0.04,
    pos_tol_m: float = _MOVE_BASE_TO_POINT_DEFAULT_POS_TOL_M,
    max_forward_m: Optional[float] = None,
) -> Generator:
    # Deprecated compatibility argument. Navigation range is governed by timeout
    # and runtime safety guards rather than a fixed target-distance cap.
    _ = max_forward_m
    world = ctx.world
    requested_pos_tol_m = float(pos_tol_m)
    effective_pos_tol_m = max(_MOVE_BASE_TO_POINT_DEFAULT_POS_TOL_M, requested_pos_tol_m)
    img = str(image_id or "").strip()
    if not img:
        ctx.set_result({"ok": False, "tool": "move_base_to_point", "error": "需要 image_id（先 capture）"})
        yield world.hold_action()
        return
    try:
        u_i, v_i = int(u), int(v)
    except Exception:
        ctx.set_result({"ok": False, "tool": "move_base_to_point", "error": "u/v 必须是有效图像坐标"})
        yield world.hold_action()
        return
    if u_i < 0 or v_i < 0:
        ctx.set_result({"ok": False, "tool": "move_base_to_point", "error": "需要 head 图上点选 (u,v)"})
        yield world.hold_action()
        return

    hit, hit_method, err = _sphere_center_from_uv(ctx, session_id, img, u_i, v_i)
    if err or hit is None:
        ctx.set_result({
            "ok": False,
            "tool": "move_base_to_point",
            "build": _MOVE_BASE_TO_POINT_BUILD,
            "image_id": img,
            "uv": [u_i, v_i],
            "error": err or "反解 3D 点失败",
        })
        yield world.hold_action()
        return
    hit = np.asarray(hit, dtype=np.float64).reshape(3)
    floor_check = _check_hit_on_floor(world, hit, ground_tol_m)
    if not bool(floor_check.get("ok")):
        ctx.log(
            "[move_base_to_point] 中止: hit "
            f"{hit.round(3).tolist()} 不在地面，nearest={floor_check.get('nearest_floor')}"
        )
        ctx.set_result({
            "ok": False,
            "tool": "move_base_to_point",
            "build": _MOVE_BASE_TO_POINT_BUILD,
            "image_id": img,
            "uv": [u_i, v_i],
            "hit_world": [float(x) for x in hit.tolist()],
            "hit_method": hit_method,
            "floor_check": floor_check,
            "error": (
                "反解 3D 点不是地面点：请点选可见 floor/floors 表面；"
                f"hit_z={hit[2]:.3f}m, ground_tol={float(ground_tol_m):.3f}m"
            ),
        })
        yield world.hold_action()
        return

    from behavior_interface.skills.eef import _freeze_world_limb_pins

    limb_pins = _freeze_world_limb_pins(world)
    trunk_q_start = _trunk_q(world)
    if hasattr(world, "set_trunk_pin_qpos") and trunk_q_start:
        world.set_trunk_pin_qpos(trunk_q_start)

    xy0, yaw0 = _base_xy_yaw(world)
    target_xy = hit[:2].copy()
    front_xy0 = _base_front_reference_world_xy(xy0, yaw0)
    clicked_delta = target_xy - xy0
    motion_delta = target_xy - front_xy0
    clicked_point_distance_m = float(np.linalg.norm(clicked_delta))
    target_distance_m = float(np.linalg.norm(motion_delta))
    cos_y0, sin_y0 = math.cos(yaw0), math.sin(yaw0)
    clicked_forward_m = float(
        cos_y0 * clicked_delta[0] + sin_y0 * clicked_delta[1]
    )
    clicked_leftward_m = float(
        -sin_y0 * clicked_delta[0] + cos_y0 * clicked_delta[1]
    )
    forward_m = float(
        cos_y0 * motion_delta[0] + sin_y0 * motion_delta[1]
    )
    translation_m = float(
        -sin_y0 * motion_delta[0] + cos_y0 * motion_delta[1]
    )
    clicked_bearing_deg = (
        math.degrees(math.atan2(clicked_leftward_m, clicked_forward_m))
        if clicked_point_distance_m > 1e-6
        else 0.0
    )
    motion_bearing_deg = (
        math.degrees(math.atan2(translation_m, forward_m))
        if target_distance_m > 1e-6
        else 0.0
    )
    ctx.log(
        "[move_base_to_point] floor hit="
        f"{hit.round(3).tolist()} direct_xy="
        f"({forward_m:+.3f},{translation_m:+.3f})m "
        f"motion_bearing={motion_bearing_deg:+.1f}deg "
        f"target=base_front_path_start_midpoint "
        f"yaw_hold={math.degrees(yaw0):+.1f}deg"
    )
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    world._codex_fast_motion_no_obs = True
    spin_stats: Dict[str, Any] = {
        "ok": True,
        "skipped": True,
        "reason": "holonomic_direct_xy",
        "action_only": True,
    }
    forward_ok = True
    try:
        if target_distance_m > max(0.0, effective_pos_tol_m):
            forward_result = yield from _drive_body_forward(
                ctx,
                forward_m,
                translation_m=translation_m,
                vmax=0.5,
                tol=effective_pos_tol_m,
                timeout_s=float(nav_timeout_s),
                log_prefix="move_base_to_point/direct_xy",
                nav_guard=True,
            )
            forward_ok = bool(forward_result)
        for _ in range(4):
            if hasattr(world, "set_trunk_pin_qpos") and trunk_q_start:
                world.set_trunk_pin_qpos(trunk_q_start)
            yield world.set_base_velocity(0.0, 0.0, 0.0)
    finally:
        world._codex_fast_motion_no_obs = old_no_obs

    xy1, yaw1 = _base_xy_yaw(world)
    front_xy1 = _base_front_reference_world_xy(xy1, yaw1)
    xy_err = float(np.linalg.norm(target_xy - front_xy1))
    base_center_xy_err = float(np.linalg.norm(target_xy - xy1))
    yaw_err_deg = abs(math.degrees(_norm_angle_rad(yaw0 - yaw1)))
    trunk_q_end = _trunk_q(world)
    ground_guard = dict(getattr(ctx, "_body_ground_guard_report", None) or {})
    forward_failure = str(getattr(ctx, "_nav_seg_fail", "") or "")
    spin_ok = True
    xy_success_tol_m = effective_pos_tol_m
    near_target_ok = bool(xy_err <= xy_success_tol_m)
    ok = bool(near_target_ok)
    if ok:
        error = None
        failure_reason = None
        status_reason = "ok" if forward_ok else "forward_timeout_but_near_target"
    elif not forward_ok:
        failure_reason = forward_failure or "forward_failed"
        status_reason = failure_reason
        if failure_reason == "stuck":
            error = (
                "平移阶段检测到底盘卡住，已急停："
                f"xy_err={xy_err:.3f}m > tol={xy_success_tol_m:.3f}m"
            )
        elif failure_reason == "ground_recovery_timeout":
            error = (
                "离地急停后未能在超时前稳定落地："
                f"xy_err={xy_err:.3f}m > tol={xy_success_tol_m:.3f}m"
            )
        else:
            error = (
                "平移阶段未到位或超时："
                f"xy_err={xy_err:.3f}m > tol={xy_success_tol_m:.3f}m"
            )
    else:
        failure_reason = "xy_residual_too_large"
        status_reason = failure_reason
        error = (
            "平移结束但目标残差过大："
            f"xy_err={xy_err:.3f}m > tol={xy_success_tol_m:.3f}m"
        )
    ctx.set_result({
        "ok": ok,
        "tool": "move_base_to_point",
        "build": _MOVE_BASE_TO_POINT_BUILD,
        "image_id": img,
        "uv": [u_i, v_i],
        "hit_world": [float(x) for x in hit.tolist()],
        "hit_method": hit_method,
        "floor_check": floor_check,
        "exec_order": "direct_xy_translation",
        "body_motion": (
            "simultaneous_robot_frame_xy; yaw held; target is base-front "
            "path-start midpoint; trunk/arms/grippers pinned"
        ),
        "linear_control": "simultaneous_robot_frame_xy",
        "target_reference": "base_front_path_start_midpoint",
        "target_reference_source": "viz_base_path_overlay_static_geometry",
        "front_reference_local_xy_m": [
            float(BASE_FRONT_OFFSET_M),
            float(PATH_Y_CENTER_M),
        ],
        "target_yaw_deg": round(math.degrees(yaw0), 3),
        "clicked_bearing_deg": round(clicked_bearing_deg, 3),
        "motion_bearing_deg": round(motion_bearing_deg, 3),
        "spin_deg": 0.0,
        "clicked_point_distance_from_base_center_m": round(
            clicked_point_distance_m,
            4,
        ),
        "target_distance_m": round(target_distance_m, 4),
        "commanded_base_displacement_m": round(target_distance_m, 4),
        "forward_m": round(forward_m, 4),
        "translation_m": round(translation_m, 4),
        "start_base_xy_yaw": [round(float(xy0[0]), 4), round(float(xy0[1]), 4), round(math.degrees(yaw0), 3)],
        "end_base_xy_yaw": [round(float(xy1[0]), 4), round(float(xy1[1]), 4), round(math.degrees(yaw1), 3)],
        "start_front_reference_xy": [
            round(float(front_xy0[0]), 4),
            round(float(front_xy0[1]), 4),
        ],
        "end_front_reference_xy": [
            round(float(front_xy1[0]), 4),
            round(float(front_xy1[1]), 4),
        ],
        "xy_err_m": round(xy_err, 4),
        "front_reference_xy_err_m": round(xy_err, 4),
        "base_center_xy_err_m": round(base_center_xy_err, 4),
        "yaw_err_deg": round(yaw_err_deg, 3),
        "requested_pos_tol_m": round(requested_pos_tol_m, 4),
        "effective_pos_tol_m": round(effective_pos_tol_m, 4),
        "xy_success_tol_m": round(xy_success_tol_m, 4),
        "near_target_ok": near_target_ok,
        "spin_result": spin_stats,
        "spin_ok": spin_ok,
        "forward_ok": bool(forward_ok),
        "ground_guard": ground_guard,
        "status_reason": status_reason,
        "failure_reason": failure_reason,
        "limb_pins": limb_pins,
        "trunk_q_start": [round(float(x), 5) for x in trunk_q_start],
        "trunk_q_end": [round(float(x), 5) for x in trunk_q_end],
        "error": error,
    })
    yield world.hold_action()
