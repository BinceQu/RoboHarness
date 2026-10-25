"""move_to_object_v2 —— 点选物体中心，升降后几何规划 xy/spin/q3 俯仰。

与 move_to_point 共用 _run_move_to_center(point_pipeline=True)；
目标 C = 物体 AABB 中心（非 uv 反解 3D 点）。
几何：move_to_object_geom.py（reach=0.60 统一）。
禁止 head 取景补偿；移动后拍 head 图并返回，全入画与否不判失败。
"""

from __future__ import annotations

# 进程日志里应出现此串；move_to_point 共用同一 pipeline build.
# Arms are initialized to grasp prep at reset/startup; lift itself does not move arms.
_NAV_PIPELINE_BUILD = "reach060_shoulder_forward_q3_v3_lowz_q1_repair_relative_lut_reset_grasp_only"
_MOVE_TO_OBJECT_BUILD = (
    f"{_NAV_PIPELINE_BUILD}_lift_xy_spin_pitch_v2_shoulder_dist_result_v1"
    "_eef_balance_v2_orientation_primary_goal_brake_guard_v1"
    "_reach_point_rgbd_pitch_recovery_v4"
    "_lift_fast_waypoints_by_step_limit_v1"
    "_reach_comp_fwd005_rollback_best_q3pitch_v2"
)

import json
import math
import os
import time
from typing import Any, Dict, Generator, Optional, Tuple

from behavior_interface.trunk_vertical_lift import POINT_PRE_DESCENT_DZ_MAX_M

# 闭环导航默认超时（秒）；可由 skill 参数 nav_timeout_s 覆盖
_NAV_TIMEOUT_S_DEFAULT = 120.0
_NAV_POS_TOL_M = 0.08
_NAV_YAW_TOL_DEG = 3.0
_NAV_Z_TOL_M = 0.04
_NAV_THETA_Z_TOL_DEG = 3.0
# 非 point 路径 dz 预降时锁定起点 θz 的守护容差
_PRE_DESCENT_THETA_Z_TOL_DEG = 2.0
# 执行期额外膨胀（米），叠加 Scene Graph robot_radius=0.25；过大易误报「远处被挡」
_NAV_EXTRA_INFLATE_M = 0.04
_NAV_STUCK_BACKUP_M = 0.32
# 底盘 footprint 到目标物体水平 AABB 的最小距离。
_BASE_TARGET_MIN_CLEARANCE_M = 0.10
# 弦球 xy 解如果让底盘太贴目标，不要求增大 R；保持 spin/yaw 不变，
# 沿机器人最终 backward 方向把底盘退到安全边界外。
_BASE_TARGET_BACKOFF_MARGIN_M = 0.005
_BASE_TARGET_BACKOFF_EXEC_GUARD_M = _NAV_POS_TOL_M
_BASE_TARGET_BACKOFF_MAX_M = 0.80
_BASE_TARGET_BACKOFF_STEP_M = 0.01
# 到位后、API 退出 capture 前 hold 稳定（秒），衰减底盘/躯干惯性晃动
# move_to_object_v2 / move_to_point_v2 共用（_run_move_to_center 末尾）
_NAV_EXIT_SETTLE_S = 5.0
_FAST_BASE_XY_YAW_MAX_S = 20.0
_FAST_BASE_PAYLOAD_MAX_FRAMES = 48
_FAST_TRUNK_MAX_S = 10.0
# —— 底盘贴地速度控制参数 ——
# 本模块使用物理单位生成速度，再反映射为 HolonomicBaseJointController 的输入 action。
# 当前 R1Pro 控制器把 [-1, 1] 映射到线速度 [-1.5, 1.5]m/s、角速度 [-pi, pi]rad/s，
# 不能直接把 m/s 或 rad/s 写进 action。
_BASE_MAX_LIN_VEL = 1.30
_BASE_MAX_ANG_VEL = 1.10
_BASE_PAYLOAD_MAX_LIN_VEL = 0.85
_BASE_PAYLOAD_MAX_ANG_VEL = 0.70
_BASE_MAX_LIN_ACCEL = 2.40
_BASE_MAX_ANG_ACCEL = 3.00
_BASE_PAYLOAD_MAX_LIN_ACCEL = 1.50
_BASE_PAYLOAD_MAX_ANG_ACCEL = 2.00
_BASE_LIN_SLOW_RADIUS = 0.55
_BASE_ANG_SLOW_RADIUS = math.radians(30.0)
# 终点制动使用低于命令减速度上限的保守值，确保 ramp 有余量跟随制动包络。
_BASE_BRAKE_DECEL_SAFETY = 0.75
_BASE_GOAL_STOP_MARGIN_M = 0.015
_BASE_GOAL_PLANE_TOL_M = 0.010
_BASE_GOAL_HARD_STOP_STEPS = 6

# 离地守护按仿真时间 20Hz 采样。触发后立即零速，确认稳定落地后自动续跑。
_BASE_GROUND_MONITOR_HZ = 20.0
_BASE_AIRBORNE_RISE_STOP_M = 0.018
_BASE_AIRBORNE_RISE_VZ_GATE_M = 0.012
_BASE_AIRBORNE_UP_VEL_STOP_MPS = 0.25
_BASE_AIRBORNE_TILT_RISE_STOP_DEG = 8.0
_BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M = 0.12
_BASE_GROUNDED_Z_MARGIN_M = 0.012
_BASE_GROUNDED_MAX_ABS_VZ_MPS = 0.08
_BASE_GROUNDED_MAX_TILT_RISE_DEG = 4.0
_BASE_GROUNDED_STABLE_SAMPLES = 3
_BASE_GROUNDED_MIN_CONTACT_LINKS = 3
# 骑在障碍棱边上时 z/tilt/contacts 三项都能满足，但底盘仍被侧向挤压着。
# 这种「假贴地」在 skill 退出后的自由物理步进里会滑落并自转（实测一次 45°，
# 导致随后基于旧位姿的点选全部指错方向）。因此判定必须同时确认水平已静止。
_BASE_GROUNDED_MAX_PLANAR_SPEED_MPS = 0.015
_BASE_GROUNDED_MAX_YAW_RATE_DPS = 3.0
# 水平残余受力的释放比垂直方向慢，3 个采样（20Hz 下 0.15s）不足以观察到滑移。
_BASE_GROUNDED_SETTLE_CONFIRM_SAMPLES = 8
_BASE_RESUME_SPEED_FACTOR = 0.72
_BASE_MIN_RESUME_SPEED_FACTOR = 0.35
# 水平碰撞 / 卡住守护同样按仿真时间 20Hz 运行。直接顶住家具时不能继续
# 把速度命令加到上限，更不能等完整 nav timeout。
_BASE_COLLISION_CONTACT_IMPULSE_MIN = 1e-4
_BASE_COLLISION_CONTACT_IMPULSE_HARD = 5e-2
_BASE_COLLISION_CONTACT_PERSIST_SAMPLES = 2
_BASE_STUCK_MIN_LIN_CMD_MPS = 0.05
_BASE_STUCK_MIN_ANG_CMD_RAD_S = 0.07
_BASE_STUCK_WINDOW_S = 0.80
_BASE_STUCK_XY_PROGRESS_M = 0.015
_BASE_STUCK_YAW_PROGRESS_DEG = 1.5
# 抓取判定用的肩距上限，同时是补偿的触发线与收手线
_REACH_OK_MAX_DIST_M = 0.70
# 主 move 跑完肩距仍 >0.7m 时的补偿补丁（不介入前面的弦球/升降规划）：
# ① 底盘每步前进 0.10m，直到卡住或肩距 ≤0.7；
# ② 仍偏远则每步压 q3 加深俯身 5°，肩距不再下降或累计 30° 就停，≤0.7 立即收手。
_REACH_COMP_FORWARD_STEP_M = 0.05
_REACH_COMP_FORWARD_MAX_STEPS = 15
_REACH_COMP_FORWARD_TIMEOUT_S = 20.0
_REACH_COMP_PITCH_STEP_DEG = 5.0
_REACH_COMP_PITCH_MAX_DEG = 30.0
# Phase2 只动 q3，不加 θz：θz=acos(forward·+Z) 封顶 180° 且 q3 继续下压后会折返，
# 地面目标需要的俯角就落在折返支上，用 θz 当控制量会顶在 165°/178° 不动。
# q3 下限取两个约束里更严的那个：
#   ① URDF 硬限位 -1.8326rad（-105°）；
#   ② 俯角 q1+q2-q3 ≤ 165°——补偿阶段允许在原上限基础上再俯身 15°。
# 实测两者谁更严取决于当前 q1+q2：站得直时（q1+q2 小）①先到，已深俯时②先到。
_REACH_COMP_Q3_MIN_RAD = -1.8326
_REACH_COMP_MAX_CHEST_PITCH_DEG = 165.0


def _challenge_action_only() -> bool:
    mode = str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE")
        or os.environ.get("INTERFACE_CHALLENGE_MODE")
        or ""
    ).strip().lower()
    return mode in {"train", "public_test", "hidden_test"}
import numpy as np

from behavior_interface.skills import register_skill
from behavior_interface.skills.capture import warm_obs_annotators
from behavior_interface.skills.base_chord_reach import (
    REACH_SPHERE_R_M,
    SHOULDER_TO_GRASP_WORK_M,
    reach_sphere_radius,
    shoulder_span_m,
    verify_chord_on_sphere,
    shoulder_positions_at_base,
)
from behavior_interface.skills.grasp import (
    _ARM_MAX_REACH,
    _ARM_MIN_REACH,
    _auto_pick_arm,
    _is_reachable,
)

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve(world, object_name: str):
    from behavior_interface.skills.grasp import _resolve_object_handle
    obj = _resolve_object_handle(world, object_name)
    if obj is not None:
        return obj
    try:
        target = object_name.split(".")[0].lower()
        for _, ent in (world.env.task.object_scope or {}).items():
            o = getattr(ent, "unwrapped", ent)
            if o is not None and (getattr(o, "category", "") or "").lower() == target:
                return o
    except Exception:
        pass
    return None


def _aabb(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        lo, hi = obj.aabb
        return _to_np(lo).reshape(-1), _to_np(hi).reshape(-1)
    except Exception:
        return None


def _ensure_plan_scene_graph(world, ctx, log_tag: str) -> bool:
    """规划前确保 Scene Graph 可用，优先复用 server 主循环缓存。"""
    sg_cached = getattr(world, "current_scene_graph", None)
    fr_cached = getattr(sg_cached, "free_region", None) if sg_cached is not None else None
    if fr_cached is not None:
        try:
            n_obs = len(fr_cached.obstacles)
        except Exception:
            n_obs = "?"
        ctx.log(f"[{log_tag}] 规划前 SG 使用缓存 obstacles={n_obs}")
        return True
    try:
        sg = world.build_scene_graph(robot_radius=0.25)
        if sg is not None:
            world.current_scene_graph = sg
            ctx.log(
                f"[{log_tag}] 规划前 SG 刷新 obstacles="
                f"{len(sg.free_region.obstacles)}"
            )
            return True
    except Exception as e:
        ctx.log(f"[{log_tag}] 规划前 SG 刷新失败: {e}")
    return False


def _nav_clearance_xy(world, x: float, y: float):
    from behavior_interface.scene_graph import point_clearance

    sg = getattr(world, "current_scene_graph", None)
    fr = getattr(sg, "free_region", None) if sg is not None else None
    if fr is None:
        return True, 999.0, None
    return point_clearance(fr, float(x), float(y))


def _is_free_xy(world, x: float, y: float) -> bool:
    free, _, _ = _nav_clearance_xy(world, x, y)
    return bool(free)


def _base_target_clearance_value(measure: Dict[str, Any]) -> float:
    return float(
        measure.get(
            "distance_aabb_m",
            measure.get("distance_point_m", float("inf")),
        )
    )


def _translate_plan_detail_xy(detail: Dict[str, Any], dx: float, dy: float) -> None:
    for key in ("shoulder_mid", "left_shoulder", "right_shoulder"):
        val = detail.get(key)
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        shifted = list(val)
        shifted[0] = float(shifted[0]) + float(dx)
        shifted[1] = float(shifted[1]) + float(dy)
        detail[key] = shifted


def _resolve_from_uv(
    ctx, session_id: str, image_id: str, u: int, v: int,
) -> Tuple[Optional[str], Any, Optional[str]]:
    """图像点选 → BDDL 名 + 物体 handle（逻辑同 mark_object_v2）。"""
    from behavior_interface.skills.mark_object_v2 import (
        _find_object_at,
        _gather_object_info,
        _write_memory,
    )
    from behavior_interface.skills.plan_eef_v2 import _build_session, _hit_at_uv

    world = ctx.world
    try:
        session, cam_pos, cam_quat, w, h, fl, ha = _build_session(
            session_id, image_id, "right",
        )
    except (FileNotFoundError, ValueError) as e:
        return None, None, str(e)

    u_i, v_i = int(u), int(v)
    if u_i >= w - 8 or u_i <= 7:
        ctx.log(
            f"[move_to_object_v2] 警告: u={u_i} 贴近图像左右边缘 (宽={w})，"
            "反投影/命中可能不准，建议点选物体中心附近"
        )

    hit, _ = _hit_at_uv(session, u_i, v_i, cam_pos, cam_quat, w, h, fl, ha)
    if hit is None:
        return None, None, f"无法反解 ({u_i},{v_i}) 的 3D 点"

    bddl_name, obj = _find_object_at(world, np.asarray(hit))
    if bddl_name is None or obj is None:
        return None, None, "该点附近未找到 BDDL 物体"

    info = _gather_object_info(obj, bddl_name, np.asarray(hit), None)
    _write_memory(session_id, info, image_id, int(u), int(v))
    ctx.log(f"[move_to_object_v2] 点选 ({u},{v}) → 物体「{bddl_name}」")
    return bddl_name, obj, None


def _grasp_proxy(center: np.ndarray) -> Dict[str, Any]:
    return {"pos": [float(center[0]), float(center[1]), float(center[2])]}


def _reach_report_live(world, center: np.ndarray) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    center = np.asarray(center, dtype=np.float64).reshape(3)
    pref = _auto_pick_arm(world, _grasp_proxy(center))
    for side in ("left", "right"):
        ok, reason = _is_reachable(world, _grasp_proxy(center), arm=side)
        sh_pos = None
        try:
            sh = world.shoulder_pose(arm=side)
            sh_arr = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
            sh_pos = [float(x) for x in sh_arr]
            d = float(np.linalg.norm(center - sh_arr))
        except Exception:
            d = -1.0
        arms[side] = {
            "reachable": ok,
            "reason": reason,
            "shoulder_world": sh_pos,
            "shoulder_to_object_m": round(d, 4) if d >= 0 else None,
        }
    left_ok = arms["left"]["reachable"]
    right_ok = arms["right"]["reachable"]
    pick = pref if arms.get(pref, {}).get("reachable") else (
        "left" if left_ok else ("right" if right_ok else None)
    )
    left_d = arms["left"]["shoulder_to_object_m"]
    right_d = arms["right"]["shoulder_to_object_m"]
    return {
        "left": arms["left"],
        "right": arms["right"],
        "reachable_left": left_ok,
        "reachable_right": right_ok,
        "reachable": left_ok or right_ok,
        "arms_reachable": [a for a in ("left", "right") if arms[a]["reachable"]],
        "arm": pick,
        "preferred_arm": pref,
        "target_center_world": [float(x) for x in center],
        "left_shoulder_world": arms["left"]["shoulder_world"],
        "right_shoulder_world": arms["right"]["shoulder_world"],
        "left_shoulder_to_object_m": left_d,
        "right_shoulder_to_object_m": right_d,
        "shoulder_to_object_m": (
            min(x for x in (left_d, right_d) if x is not None)
            if any(x is not None for x in (left_d, right_d))
            else None
        ),
    }


def _shoulder_distance_result(
    dual: Dict[str, Any],
    center: np.ndarray,
    *,
    reach_R_m: Optional[float] = None,
) -> Dict[str, Any]:
    center_list = [float(x) for x in np.asarray(center, dtype=np.float64).reshape(3)]
    left = dual.get("left") if isinstance(dual.get("left"), dict) else {}
    right = dual.get("right") if isinstance(dual.get("right"), dict) else {}
    left_d = dual.get("left_shoulder_to_object_m", left.get("shoulder_to_object_m"))
    right_d = dual.get("right_shoulder_to_object_m", right.get("shoulder_to_object_m"))
    left_sh = dual.get("left_shoulder_world", left.get("shoulder_world"))
    right_sh = dual.get("right_shoulder_world", right.get("shoulder_world"))
    d_vals = [float(x) for x in (left_d, right_d) if x is not None]
    nearest = round(min(d_vals), 4) if d_vals else None

    def _err_to_R(d):
        if d is None or reach_R_m is None:
            return None
        return round(float(d) - float(reach_R_m), 4)

    structured = {
        "target_center_world": center_list,
        "unit": "m",
        "reach_R_m": round(float(reach_R_m), 4) if reach_R_m is not None else None,
        "left": {
            "shoulder_world": left_sh,
            "distance_m": left_d,
            "error_to_reach_R_m": _err_to_R(left_d),
            "reachable": bool(left.get("reachable", dual.get("reachable_left", False))),
            "reason": left.get("reason"),
        },
        "right": {
            "shoulder_world": right_sh,
            "distance_m": right_d,
            "error_to_reach_R_m": _err_to_R(right_d),
            "reachable": bool(right.get("reachable", dual.get("reachable_right", False))),
            "reason": right.get("reason"),
        },
        "nearest_distance_m": nearest,
    }
    return {
        "left_shoulder_world": left_sh,
        "right_shoulder_world": right_sh,
        "left_shoulder_to_object_m": left_d,
        "right_shoulder_to_object_m": right_d,
        "shoulder_to_object_m": nearest,
        "left_shoulder_to_target_m": left_d,
        "right_shoulder_to_target_m": right_d,
        "shoulder_to_target_m": nearest,
        "left_shoulder_reach_error_m": _err_to_R(left_d),
        "right_shoulder_reach_error_m": _err_to_R(right_d),
        "shoulder_distance": structured,
    }


def _save_nav_reach(session_id: str, payload: Dict[str, Any]) -> None:
    if not session_id:
        return
    try:
        from behavior_interface import agent_runs
        path = os.path.join(agent_runs.run_dir(session_id), "navigation.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _head_view_live(world, lo: np.ndarray, hi: np.ndarray) -> Dict[str, Any]:
    from behavior_interface.head_capture import (
        get_head_sensor,
        head_intrinsics_tuple,
    )
    from behavior_interface.skills.head_view_offline import project_aabb_to_head

    head = get_head_sensor(world)
    if head is None:
        return {"ok": False, "reason": "no head sensor"}
    try:
        pos, quat = head.get_position_orientation()
        fl, ha, w, h = head_intrinsics_tuple(head)
        return project_aabb_to_head(
            np.asarray(pos).reshape(3), np.asarray(quat).reshape(4),
            lo, hi, w, h, fl, ha,
        )
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def _capture_head_snapshot(
    world,
    *,
    session_id: str,
    head_png_tag: str,
    log_tag: str,
    ctx=None,
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
    suffix: str = "failure",
) -> Dict[str, Any]:
    """Save a head snapshot for both successful and failed move_to* runs."""
    info: Dict[str, Any] = {
        "ok": False,
        "capture_ok": False,
        "reason": None,
        "head_png": None,
    }
    if lo is not None and hi is not None:
        try:
            info.update(_head_view_live(world, np.asarray(lo), np.asarray(hi)))
        except Exception as e:
            info.update({"ok": False, "reason": f"head_view_live: {e}"})
    safe_suffix = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in suffix)
    head_png = os.path.join(
        "/tmp", f"{head_png_tag}_{session_id or 'default'}_{safe_suffix}.png"
    )
    try:
        from behavior_interface.head_capture import capture_head_png

        cap = capture_head_png(world, head_png, n_render=12)
        if cap:
            info["capture_ok"] = True
            info["ok"] = bool(info.get("ok", True))
            info["head_png"] = head_png
            if isinstance(cap, dict):
                info["capture"] = cap
        else:
            info["reason"] = info.get("reason") or "capture_head_png returned falsy"
    except Exception as e:
        info["reason"] = f"capture_head_png: {e}"
    if ctx is not None:
        ctx.log(
            f"[{log_tag}] head capture ({suffix}) → {info.get('head_png')} "
            f"capture_ok={info.get('capture_ok')} view_ok={info.get('ok')} "
            f"reason={info.get('reason')}"
        )
    return info


def _read_current_head_rgbd_for_recovery(ctx, world, log_tag: str) -> Dict[str, Any]:
    """Read one synchronized head RGB-D frame plus local camera calibration."""
    import importlib

    from behavior_interface.camera_render_control import (
        set_robot_camera_render_updates,
    )
    from behavior_interface.skills.capture import (
        _ensure_modalities,
        _read_complete_frame,
    )

    head_capture = importlib.import_module("behavior_interface.head_capture")
    if not hasattr(head_capture, "head_factory_mount_pose"):
        head_capture = importlib.reload(head_capture)
    get_head_sensor = head_capture.get_head_sensor
    head_factory_mount_pose = head_capture.head_factory_mount_pose
    head_intrinsics_dict = head_capture.head_intrinsics_dict

    head = get_head_sensor(world)
    if head is None:
        return {"ok": False, "error": "未找到 head camera"}
    try:
        set_robot_camera_render_updates(world, True)
        changed = _ensure_modalities(head, ["rgb", "depth_linear"])
        if changed:
            try:
                env = getattr(world, "env", None)
                if env is not None:
                    env.load_observation_space()
            except Exception:
                pass
        try:
            import omnigibson as og

            for _ in range(5 if changed else 3):
                og.sim.render()
        except Exception:
            pass
        obs, _info, frame_errors = _read_complete_frame(
            head,
            ["rgb", "depth_linear"],
            ctx,
            log_tag=f"{log_tag}.collision_pitch_rgbd",
            n_attempts=4,
            allow_rebuild=True,
        )
        if frame_errors:
            return {
                "ok": False,
                "error": "当前 head RGB-D 未就绪: " + "; ".join(frame_errors),
            }
        rgb = _to_np(obs["rgb"])
        depth = _to_np(obs["depth_linear"]).squeeze()
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return {"ok": False, "error": f"head RGB shape 异常: {rgb.shape}"}
        if depth.ndim != 2:
            return {"ok": False, "error": f"head depth shape 异常: {depth.shape}"}
        mount = head_factory_mount_pose(world)
        if mount is None:
            return {"ok": False, "error": "缺少 head camera parent-frame 标定"}
        mount_pos, mount_quat = mount
        intrinsics = head_intrinsics_dict(head)
        return {
            "ok": True,
            "rgb": np.asarray(rgb[..., :3]).copy(),
            "depth": np.asarray(depth, dtype=np.float32).copy(),
            "camera_parent_pos": _to_np(mount_pos).reshape(3).astype(float).tolist(),
            "camera_parent_quat": _to_np(mount_quat).reshape(4).astype(float).tolist(),
            "intrinsics": intrinsics,
            "observation_sources": [
                "head_rgb",
                "head_depth_linear",
                "trunk_proprioception",
                "fixed_head_camera_parent_pose",
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": f"读取当前 head RGB-D 失败: {exc}"}


def _load_reference_head_rgb(session_id: str, image_id: str) -> np.ndarray:
    import cv2

    from behavior_interface import agent_runs

    path = agent_runs.image_path(session_id, image_id, ".png")
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取原始 capture RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _yield_reach_point_collision_pitch_recovery(
    ctx,
    world,
    *,
    session_id: str,
    image_id: str,
    source_uv_px,
    move_report: Dict[str, Any],
    log_tag: str,
    keep_ori_arm: str,
) -> Generator:
    """After grounded no-progress, pitch at the current base pose using RGB-D."""
    from behavior_interface.skills.move_to import yield_trunk_to_planned_pose
    from behavior_interface.skills.reach_point_pitch_recovery import (
        chest_targets_from_trunk_q,
        evaluate_target_at_trunk_q,
        grounded_abort_sample,
        head_camera_pose_robot_from_proprio,
        pitch_for_aligned_target,
        track_rgb_point,
        unproject_depth_pixel_robot,
    )

    report: Dict[str, Any] = {
        "ok": False,
        "attempted": False,
        "mode": "current_rgbd_pitch_at_stopped_base",
        "trigger": move_report.get("collision_guard_reason"),
        "observation_contract": (
            "head RGB + depth_linear + trunk proprioception + fixed camera mount; "
            "no robot/object/camera world pose"
        ),
    }
    reason = str(move_report.get("collision_guard_reason") or "")
    if reason != "no_progress":
        report["error"] = f"仅 no_progress 可俯身恢复，当前 reason={reason or 'unknown'}"
        return report
    try:
        abort_ground = grounded_abort_sample(
            z_m=move_report["abort_z_m"],
            ground_z_ref_m=move_report["ground_z_ref_m"],
            vz_mps=move_report["abort_vz_mps"],
            tilt_deg=move_report["abort_tilt_deg"],
            tilt_rise_deg=move_report["abort_tilt_rise_deg"],
            z_margin_m=_BASE_GROUNDED_Z_MARGIN_M,
            max_abs_vz_mps=_BASE_GROUNDED_MAX_ABS_VZ_MPS,
            max_tilt_deg=5.0,
            max_tilt_rise_deg=_BASE_GROUNDED_MAX_TILT_RISE_DEG,
        )
    except (KeyError, TypeError, ValueError) as exc:
        report["error"] = f"缺少急停即时贴地观测: {exc}"
        return report
    report["abort_ground_check"] = abort_ground
    if not abort_ground["ok"]:
        report["error"] = (
            "急停当下底盘未稳定贴地，不执行俯身恢复: "
            f"rise={abort_ground['ground_rise_m']:+.4f}m "
            f"vz={abort_ground['vz_mps']:+.3f}m/s "
            f"tilt={abort_ground['tilt_deg']:.1f}°"
        )
        return report

    report["attempted"] = True
    ctx.log(
        f"[{log_tag}] [碰撞后俯身恢复] 底盘保持当前位置；"
        "读取当前 RGB-D + trunk proprio + camera parent 标定"
    )
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    try:
        world._codex_fast_motion_no_obs = False
        yield from _hold(world, 4)
        current = _read_current_head_rgbd_for_recovery(ctx, world, log_tag)
    finally:
        world._codex_fast_motion_no_obs = old_no_obs
    report["current_observation"] = {
        key: value
        for key, value in current.items()
        if key not in {"rgb", "depth"}
    }
    if not current.get("ok"):
        report["error"] = current.get("error", "当前 RGB-D 读取失败")
        return report

    try:
        reference_rgb = _load_reference_head_rgb(session_id, image_id)
        frame_mean_abs_diff = float(
            np.mean(
                np.abs(
                    reference_rgb.astype(np.float32)
                    - current["rgb"].astype(np.float32)
                )
            )
        )
        source = np.asarray(source_uv_px, dtype=np.float64).reshape(2)
        tracking = track_rgb_point(
            reference_rgb,
            current["rgb"],
            source_px=(float(source[0]), float(source[1])),
        )
        tracked = np.asarray(tracking["tracked_pixel_uv"], dtype=np.float64)
        tracked_px = np.rint(tracked).astype(int)
        trunk_q_start = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        camera_relative = head_camera_pose_robot_from_proprio(
            trunk_q_start,
            camera_parent_pos=current["camera_parent_pos"],
            camera_parent_quat=current["camera_parent_quat"],
        )
        intrinsics = current["intrinsics"]
        target_robot, point_meta = unproject_depth_pixel_robot(
            current["depth"],
            px=int(tracked_px[0]),
            py=int(tracked_px[1]),
            camera_relative_pose=camera_relative,
            focal_length=float(intrinsics["focal_length"]),
            horizontal_aperture=float(intrinsics["horizontal_aperture"]),
        )
        horizontal_m = float(np.linalg.norm(target_robot[:2]))
        target_bearing_deg = math.degrees(
            math.atan2(float(target_robot[1]), float(target_robot[0]))
        )
        if abs(target_bearing_deg) > 20.0:
            raise ValueError(
                f"目标相对底盘偏航 {target_bearing_deg:+.1f}° > 20°，"
                "原地 trunk q4 不能可靠补偿"
            )
        pitch_solution = pitch_for_aligned_target(target_robot, trunk_q_start)
        target_q = np.asarray(
            pitch_solution["target_trunk_q"],
            dtype=np.float64,
        ).reshape(4)
        chest_targets = chest_targets_from_trunk_q(target_q)
        report.update({
            "tracking": tracking,
            "reference_current_rgb_mean_abs_diff": frame_mean_abs_diff,
            "camera_relative_pose": camera_relative,
            "point_observation": point_meta,
            "target_robot_m": target_robot.astype(float).tolist(),
            "target_horizontal_distance_m": horizontal_m,
            "target_bearing_deg": target_bearing_deg,
            "trunk_q_start": trunk_q_start.astype(float).tolist(),
            "pitch_solution": pitch_solution,
            "trunk_targets": chest_targets,
        })
        ctx.log(
            f"[{log_tag}] [碰撞后俯身恢复] tracked="
            f"({tracked[0]:.1f},{tracked[1]:.1f}) "
            f"source={tracking['source']} "
            f"score={tracking.get('confidence', tracking.get('template_score', 0.0)):.3f} "
            f"frame_mad={frame_mean_abs_diff:.1f} "
            f"target_robot=({target_robot[0]:.3f},{target_robot[1]:+.3f},"
            f"{target_robot[2]:.3f}) horizontal={horizontal_m:.3f}m "
            f"bearing={target_bearing_deg:+.1f}° "
            f"q3 {trunk_q_start[2]:+.3f}→{target_q[2]:+.3f}rad "
            f"q4 {trunk_q_start[3]:+.3f}→{target_q[3]:+.3f}rad "
            f"pitch_delta={pitch_solution['pitch_delta_deg']:+.1f}° "
            f"torso_yaw={pitch_solution['torso_yaw_deg']:+.1f}°"
        )
    except Exception as exc:
        report["error"] = f"合规 RGB-D 目标重定位/俯身求解失败: {exc}"
        ctx.log(f"[{log_tag}] [碰撞后俯身恢复] {report['error']}")
        return report

    trunk_exec = yield from yield_trunk_to_planned_pose(
        ctx,
        world,
        chest_z_tgt=float(chest_targets["chest_z_robot_m"]),
        theta_z_tgt=float(chest_targets["theta_z_deg"]),
        z_tol=0.05,
        theta_z_tol_deg=_NAV_THETA_Z_TOL_DEG,
        trunk_max_step_rad=0.08,
        trunk_timeout_s=45.0,
        log_prefix=f"{log_tag}/collision_pitch",
        object_z=None,
        reach_m=None,
        upward_require_theta_settle=False,
        target_trunk_q=target_q,
        keep_ori_arm=keep_ori_arm,
    )
    trunk_q_final = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    verification = evaluate_target_at_trunk_q(target_robot, trunk_q_final)
    nearest = float(verification["nearest_shoulder_to_target_m"])
    recovery_ok = bool(
        trunk_exec.get("ok")
        and float(verification["ray_forward_distance_m"]) > 0.0
        and float(verification["ray_residual_m"]) <= 0.06
        and _ARM_MIN_REACH <= nearest <= _ARM_MAX_REACH
    )
    report.update({
        "ok": recovery_ok,
        "trunk_exec": trunk_exec,
        "trunk_q_final": trunk_q_final.astype(float).tolist(),
        "verification": verification,
    })
    if recovery_ok:
        ctx.log(
            f"[{log_tag}] [碰撞后俯身恢复] 成功: "
            f"ray_residual={verification['ray_residual_m']:.3f}m "
            f"shoulder_nearest={nearest:.3f}m，底盘未继续前进"
        )
    else:
        report["error"] = (
            f"俯身后目标仍未进入工作空间: trunk_ok={trunk_exec.get('ok')} "
            f"ray_residual={verification['ray_residual_m']:.3f}m "
            f"shoulder_nearest={nearest:.3f}m"
        )
        ctx.log(f"[{log_tag}] [碰撞后俯身恢复] {report['error']}")
    return report


def _fallback_make_action_trunk_locked(world, trunk_q):
    """兼容热更新前的 WorldAPI：锁 base/arms，只改 trunk。"""
    if getattr(world, "dry_run", False):
        return world.empty_action()

    kw: Dict[str, Any] = {
        "base": [0.0, 0.0, 0.0],
        "trunk": np.asarray(trunk_q, dtype=np.float64).reshape(-1).tolist(),
    }
    for arm in ("left", "right"):
        try:
            world.controller_action_idx(f"arm_{arm}")
            if hasattr(world, "arm_qpos_list"):
                kw[f"arm_{arm}"] = world.arm_qpos_list(arm)
            else:
                names = list(world.robot.joints.keys())
                idx = [names.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
                qpos = world.robot.get_joint_positions()
                kw[f"arm_{arm}"] = [float(qpos[i]) for i in idx]
            try:
                if hasattr(world, "gripper_qpos_list"):
                    grip = world.gripper_qpos_list(arm)
                else:
                    gidx = world.controller_action_idx(f"gripper_{arm}")
                    qpos = world.robot.get_joint_positions()
                    grip = [float(qpos[int(i)]) for i in gidx]
                if grip is not None:
                    kw[f"gripper_{arm}"] = grip
            except Exception:
                pass
        except Exception:
            pass
    return world.make_action(**kw)


def _make_trunk_locked_action(world, trunk_q):
    fn = getattr(world, "make_action_trunk_locked", None)
    if callable(fn):
        return fn(trunk_q)
    return _fallback_make_action_trunk_locked(world, trunk_q)


def _ensure_world_action_compat(world):
    """Hot-reload skills 后，给旧 WorldAPI 实例补齐新 helper。"""
    if getattr(world, "dry_run", False):
        return
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
        return
    except Exception:
        pass
    if not callable(getattr(world, "make_action_trunk_locked", None)):
        import types

        def _compat_make_action_trunk_locked(self, trunk_q):
            return _fallback_make_action_trunk_locked(self, trunk_q)

        world.make_action_trunk_locked = types.MethodType(
            _compat_make_action_trunk_locked, world
        )
    if not callable(getattr(world, "hold_action_pinned", None)):
        import types

        def _compat_hold_action_pinned(self):
            return _fallback_make_action_trunk_locked(self, self.trunk_qpos())

        world.hold_action_pinned = types.MethodType(_compat_hold_action_pinned, world)


def _hold(world, n: int = 4):
    for _ in range(n):
        if hasattr(world, "hold_action_pinned"):
            yield world.hold_action_pinned()
        elif hasattr(world, "hold_action"):
            yield world.hold_action()
        else:
            yield _make_trunk_locked_action(world, world.trunk_qpos())


def _norm_angle_deg(a_deg: float) -> float:
    while a_deg > 180.0:
        a_deg -= 360.0
    while a_deg <= -180.0:
        a_deg += 360.0
    return a_deg


def _norm_angle_rad(a_rad: float) -> float:
    return (float(a_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _base_command_active_for_stuck(command) -> bool:
    cmd = np.asarray(command, dtype=np.float64).reshape(3)
    return bool(
        math.hypot(float(cmd[0]), float(cmd[1]))
        >= _BASE_STUCK_MIN_LIN_CMD_MPS
        or abs(float(cmd[2])) >= _BASE_STUCK_MIN_ANG_CMD_RAD_S
    )


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, math.sin(float(yaw) * 0.5), math.cos(float(yaw) * 0.5)],
        dtype=np.float64,
    )


def _quat_normalize_xyzw(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_inverse_xyzw(quat) -> np.ndarray:
    q = _quat_normalize_xyzw(quat)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_xyzw(q1, q2) -> np.ndarray:
    x1, y1, z1, w1 = _quat_normalize_xyzw(q1)
    x2, y2, z2, w2 = _quat_normalize_xyzw(q2)
    return _quat_normalize_xyzw(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64))


def _quat_rotate_xyzw(quat, vec) -> np.ndarray:
    q = _quat_normalize_xyzw(quat)
    u = q[:3]
    s = float(q[3])
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return (
        2.0 * float(np.dot(u, v)) * u
        + (s * s - float(np.dot(u, u))) * v
        + 2.0 * s * np.cross(u, v)
    )


def _quat_error_deg(q1, q2) -> float:
    a = _quat_normalize_xyzw(q1)
    b = _quat_normalize_xyzw(q2)
    dot = min(1.0, max(0.0, abs(float(np.dot(a, b)))))
    return math.degrees(2.0 * math.acos(dot))


def _array_like(reference, values):
    arr = np.asarray(values, dtype=np.float64)
    if hasattr(reference, "detach"):
        try:
            import torch as th

            return th.as_tensor(
                arr,
                dtype=getattr(reference, "dtype", th.float32),
                device=getattr(reference, "device", None),
            )
        except Exception:
            pass
    return arr.astype(np.float32)


def _zero_robot_velocities(world) -> None:
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return
    for setter_name in ("set_linear_velocity", "set_angular_velocity"):
        setter = getattr(robot, setter_name, None)
        if callable(setter):
            try:
                setter(np.zeros(3, dtype=np.float32))
            except Exception:
                pass
    try:
        vel = robot.get_joint_velocities()
        v_new = vel.clone() if hasattr(vel, "clone") else np.asarray(vel, dtype=np.float64).copy()
        v_new[...] = 0.0
        robot.set_joint_velocities(v_new)
    except Exception:
        pass


def _snapshot_assisted_payload_hold(world) -> Dict[str, Dict[str, Any]]:
    """Snapshot each assisted-grasp object's fixed transform in its EEF frame."""
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return {}
    held_map = getattr(robot, "_ag_obj_in_hand", None)
    constraint_map = getattr(robot, "_ag_obj_constraints", None)
    if not isinstance(held_map, dict):
        return {}

    hold: Dict[str, Dict[str, Any]] = {}
    for arm in ("left", "right"):
        obj = held_map.get(arm)
        if (
            obj is None
            or bool(getattr(obj, "fixed_base", False))
            or not isinstance(constraint_map, dict)
            or constraint_map.get(arm) is None
        ):
            continue
        try:
            eef_pos = _to_np(robot.get_eef_position(arm=arm)).astype(np.float64).reshape(3)
            eef_quat = _quat_normalize_xyzw(
                _to_np(robot.get_eef_orientation(arm=arm)).reshape(4)
            )
            obj_pos_raw, obj_quat_raw = obj.get_position_orientation()
            obj_pos = _to_np(obj_pos_raw).astype(np.float64).reshape(3)
            obj_quat = _quat_normalize_xyzw(_to_np(obj_quat_raw).reshape(4))
            inv_eef = _quat_inverse_xyzw(eef_quat)
            hold[arm] = {
                "object": obj,
                "name": str(
                    getattr(obj, "name", None)
                    or getattr(obj, "prim_path", None)
                    or type(obj).__name__
                ),
                "pos_eef": _quat_rotate_xyzw(inv_eef, obj_pos - eef_pos),
                "quat_eef": _quat_mul_xyzw(inv_eef, obj_quat),
            }
        except Exception:
            continue
    return hold


def _payload_expected_world_pose(robot, arm: str, spec: Dict[str, Any]):
    eef_pos = _to_np(robot.get_eef_position(arm=arm)).astype(np.float64).reshape(3)
    eef_quat = _quat_normalize_xyzw(
        _to_np(robot.get_eef_orientation(arm=arm)).reshape(4)
    )
    obj_pos = eef_pos + _quat_rotate_xyzw(eef_quat, spec["pos_eef"])
    obj_quat = _quat_mul_xyzw(eef_quat, spec["quat_eef"])
    return obj_pos, obj_quat


def _zero_payload_velocities(obj, position_reference=None) -> None:
    zero = _array_like(
        position_reference if position_reference is not None else np.zeros(3),
        np.zeros(3, dtype=np.float64),
    )
    for setter_name in ("set_linear_velocity", "set_angular_velocity"):
        setter = getattr(obj, setter_name, None)
        if callable(setter):
            try:
                setter(zero)
            except Exception:
                pass


def _force_apply_assisted_payload_hold(
    world,
    hold: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Restore held objects with the EEF so a joint snap cannot stretch the AG constraint."""
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    robot = getattr(world, "robot", None)
    hold = hold if hold is not None else getattr(world, "_codex_motion_payload_hold", None)
    if robot is None or not isinstance(hold, dict) or not hold:
        return False
    held_map = getattr(robot, "_ag_obj_in_hand", None)
    constraint_map = getattr(robot, "_ag_obj_constraints", None)
    if not isinstance(held_map, dict):
        return False

    restored = False
    max_drift = getattr(world, "_codex_motion_payload_max_drift", None)
    if not isinstance(max_drift, dict):
        max_drift = {}
        world._codex_motion_payload_max_drift = max_drift
    for arm, spec in hold.items():
        obj = spec.get("object")
        if (
            obj is None
            or bool(getattr(obj, "fixed_base", False))
            or held_map.get(arm) is not obj
            or not isinstance(constraint_map, dict)
            or constraint_map.get(arm) is None
        ):
            continue
        try:
            expected_pos, expected_quat = _payload_expected_world_pose(robot, arm, spec)
            pos_raw, quat_raw = obj.get_position_orientation()
            actual_pos = _to_np(pos_raw).astype(np.float64).reshape(3)
            actual_quat = _quat_normalize_xyzw(_to_np(quat_raw).reshape(4))
            pos_err = float(np.linalg.norm(actual_pos - expected_pos))
            ori_err = float(_quat_error_deg(actual_quat, expected_quat))
            prev = max_drift.get(arm) if isinstance(max_drift.get(arm), dict) else {}
            max_drift[arm] = {
                "name": spec.get("name"),
                "pos_m": max(float(prev.get("pos_m", 0.0)), pos_err),
                "ori_deg": max(float(prev.get("ori_deg", 0.0)), ori_err),
            }
            obj.set_position_orientation(
                position=_array_like(pos_raw, expected_pos),
                orientation=_array_like(quat_raw, expected_quat),
            )
            _zero_payload_velocities(obj, pos_raw)
            restored = True
        except Exception:
            continue
    return restored


def _fixed_base_assisted_grasps(world) -> Dict[str, str]:
    """Return fixed-base objects present in assisted-grasp bookkeeping."""
    robot = getattr(world, "robot", None)
    held_map = getattr(robot, "_ag_obj_in_hand", None)
    if not isinstance(held_map, dict):
        return {}
    out: Dict[str, str] = {}
    for arm in ("left", "right"):
        obj = held_map.get(arm)
        if obj is None or not bool(getattr(obj, "fixed_base", False)):
            continue
        out[arm] = str(
            getattr(obj, "name", None)
            or getattr(obj, "prim_path", None)
            or type(obj).__name__
        )
    return out


def _snapshot_fast_base_limb_hold(world) -> Dict[str, Dict[str, list]]:
    """Snapshot fixed articulation targets for the fast root-pose interpolation."""
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return {}
    try:
        q_now = robot.get_joint_positions()
    except Exception:
        return {}

    hold: Dict[str, Dict[str, list]] = {}
    for ctrl_name in (
        "trunk",
        "arm_left",
        "arm_right",
        "gripper_left",
        "gripper_right",
    ):
        idx = _controller_dof_indices(robot, ctrl_name)
        if idx.size <= 0:
            continue
        try:
            vals = _to_np(q_now)[idx].astype(np.float64).reshape(-1)
        except Exception:
            continue
        hold[ctrl_name] = {
            "indices": [int(i) for i in idx.tolist()],
            "qpos": [float(v) for v in vals.tolist()],
        }
    return hold


def _fast_base_limb_drift(
    world,
    hold: Optional[Dict[str, Dict[str, list]]] = None,
) -> Dict[str, float]:
    robot = getattr(world, "robot", None)
    hold = hold if hold is not None else getattr(world, "_codex_fast_base_limb_hold", None)
    if robot is None or not isinstance(hold, dict):
        return {}
    try:
        q_now = _to_np(robot.get_joint_positions()).astype(np.float64).reshape(-1)
    except Exception:
        return {}

    drift: Dict[str, float] = {}
    for ctrl_name, spec in hold.items():
        try:
            idx = np.asarray(spec["indices"], dtype=int).reshape(-1)
            q_tgt = np.asarray(spec["qpos"], dtype=np.float64).reshape(-1)
            if idx.size != q_tgt.size:
                continue
            drift[ctrl_name] = float(np.max(np.abs(q_now[idx] - q_tgt)))
        except Exception:
            continue
    return drift


def _force_apply_fast_base_limb_hold(
    world,
    hold: Optional[Dict[str, Dict[str, list]]] = None,
) -> bool:
    """Hard-restore pinned limbs after a physics step moves the robot root."""
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    robot = getattr(world, "robot", None)
    hold = hold if hold is not None else getattr(world, "_codex_fast_base_limb_hold", None)
    if robot is None or not isinstance(hold, dict) or not hold:
        return False
    try:
        q_now = robot.get_joint_positions()
        q_new = q_now.clone() if hasattr(q_now, "clone") else np.asarray(q_now, dtype=np.float64).copy()
        drift = _fast_base_limb_drift(world, hold)
        hold_indices: list[int] = []
        for spec in hold.values():
            idx = np.asarray(spec["indices"], dtype=int).reshape(-1)
            vals = np.asarray(spec["qpos"], dtype=np.float64).reshape(-1)
            if not _assign_q_values(q_new, idx, vals):
                continue
            hold_indices.extend(int(i) for i in idx.tolist())
        robot.set_joint_positions(q_new)

        try:
            vel = robot.get_joint_velocities()
            v_new = vel.clone() if hasattr(vel, "clone") else np.asarray(vel, dtype=np.float64).copy()
            for joint_i in hold_indices:
                v_new[int(joint_i)] = 0.0
            robot.set_joint_velocities(v_new)
        except Exception:
            pass

        max_drift = getattr(world, "_codex_fast_base_limb_max_drift", None)
        if not isinstance(max_drift, dict):
            max_drift = {}
            world._codex_fast_base_limb_max_drift = max_drift
        for ctrl_name, err in drift.items():
            max_drift[ctrl_name] = max(float(max_drift.get(ctrl_name, 0.0)), float(err))
        _force_apply_assisted_payload_hold(world)
        return True
    except Exception:
        return False


def _set_fast_base_pin(world, pos_xyz: np.ndarray, yaw_rad: float) -> None:
    pos_np = np.asarray(pos_xyz, dtype=np.float64).reshape(3).copy()
    world._codex_v1_fast_base_pin = {
        "pos": [float(pos_np[0]), float(pos_np[1]), float(pos_np[2])],
        "yaw": float(yaw_rad),
    }


def _force_apply_fast_base_pin(world) -> bool:
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    pin = getattr(world, "_codex_v1_fast_base_pin", None)
    if not isinstance(pin, dict):
        return False
    if getattr(world, "dry_run", False):
        try:
            pos = np.asarray(pin.get("pos"), dtype=np.float64).reshape(3)
            world._mock_base = np.array([pos[0], pos[1], float(pin.get("yaw", 0.0))], dtype=np.float64)
            return True
        except Exception:
            return False
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    pos_np = np.asarray(pin.get("pos"), dtype=np.float64).reshape(3)
    quat_np = _yaw_quat_xyzw(float(pin.get("yaw", 0.0)))
    try:
        try:
            import torch as th

            position = th.tensor(pos_np, dtype=th.float32)
            orientation = th.tensor(quat_np, dtype=th.float32)
        except Exception:
            position = pos_np
            orientation = quat_np
        robot.set_position_orientation(position=position, orientation=orientation)
        try:
            from behavior_interface.world_api import Pose

            world._last_pose = Pose(pos=pos_np.copy(), quat=quat_np.copy())
        except Exception:
            pass
        _force_apply_fast_base_limb_hold(world)
        _zero_robot_velocities(world)
        return True
    except Exception:
        return False


def _install_fast_base_post_step(
    world,
    limb_hold: Optional[Dict[str, Dict[str, list]]] = None,
    payload_hold: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return
    if getattr(world, "dry_run", False):
        return
    if getattr(world, "_codex_v1_fast_base_post_step_active", False):
        return
    try:
        import types

        prev = getattr(world, "shortcut_post_step_stabilize_now", None)
        world._codex_v1_fast_base_prev_post_step = prev

        def _fast_base_post_step(self):
            prev_cb = getattr(self, "_codex_v1_fast_base_prev_post_step", None)
            if callable(prev_cb):
                try:
                    prev_cb()
                except Exception:
                    pass
            return _force_apply_fast_base_pin(self)

        world.shortcut_post_step_stabilize_now = types.MethodType(_fast_base_post_step, world)
        world._codex_v1_fast_base_post_step_active = True
        world._codex_fast_base_limb_hold = dict(limb_hold or {})
        world._codex_fast_base_limb_max_drift = {}
        world._codex_motion_payload_hold = dict(payload_hold or {})
        world._codex_motion_payload_max_drift = {}
    except Exception:
        pass


def _restore_fast_base_post_step(world) -> None:
    if not getattr(world, "_codex_v1_fast_base_post_step_active", False):
        return
    try:
        prev = getattr(world, "_codex_v1_fast_base_prev_post_step", None)
        if callable(prev):
            world.shortcut_post_step_stabilize_now = prev
        else:
            try:
                delattr(world, "shortcut_post_step_stabilize_now")
            except Exception:
                world.shortcut_post_step_stabilize_now = None
        world._codex_v1_fast_base_post_step_active = False
        world._codex_v1_fast_base_prev_post_step = None
        world._codex_fast_base_limb_hold = None
        world._codex_fast_base_limb_max_drift = None
        world._codex_motion_payload_hold = None
        world._codex_motion_payload_max_drift = None
    except Exception:
        pass


def _install_base_motion_post_step(
    world,
    limb_hold: Optional[Dict[str, Dict[str, list]]] = None,
    payload_hold: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Hard-lock limbs and held payloads while the base itself remains velocity controlled."""
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    if getattr(world, "dry_run", False):
        return False
    if getattr(world, "_codex_base_motion_post_step_active", False):
        return False
    try:
        import types

        prev = getattr(world, "shortcut_post_step_stabilize_now", None)
        world._codex_base_motion_prev_post_step = prev

        def _base_motion_post_step(self):
            prev_cb = getattr(self, "_codex_base_motion_prev_post_step", None)
            if callable(prev_cb):
                try:
                    prev_cb()
                except Exception:
                    pass
            return _force_apply_fast_base_limb_hold(self)

        world.shortcut_post_step_stabilize_now = types.MethodType(
            _base_motion_post_step, world
        )
        world._codex_base_motion_post_step_active = True
        world._codex_fast_base_limb_hold = dict(limb_hold or {})
        world._codex_fast_base_limb_max_drift = {}
        world._codex_motion_payload_hold = dict(payload_hold or {})
        world._codex_motion_payload_max_drift = {}
        return True
    except Exception:
        return False


def _restore_base_motion_post_step(world) -> None:
    if not getattr(world, "_codex_base_motion_post_step_active", False):
        return
    try:
        prev = getattr(world, "_codex_base_motion_prev_post_step", None)
        if callable(prev):
            world.shortcut_post_step_stabilize_now = prev
        else:
            try:
                delattr(world, "shortcut_post_step_stabilize_now")
            except Exception:
                world.shortcut_post_step_stabilize_now = None
        world._codex_base_motion_post_step_active = False
        world._codex_base_motion_prev_post_step = None
        world._codex_fast_base_limb_hold = None
        world._codex_fast_base_limb_max_drift = None
        world._codex_motion_payload_hold = None
        world._codex_motion_payload_max_drift = None
    except Exception:
        pass


def _payload_end_drift(
    world,
    hold: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    robot = getattr(world, "robot", None)
    if robot is None or not isinstance(hold, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for arm, spec in hold.items():
        obj = spec.get("object")
        if obj is None:
            continue
        try:
            expected_pos, expected_quat = _payload_expected_world_pose(robot, arm, spec)
            pos_raw, quat_raw = obj.get_position_orientation()
            out[arm] = {
                "name": spec.get("name"),
                "pos_m": float(np.linalg.norm(
                    _to_np(pos_raw).astype(np.float64).reshape(3) - expected_pos
                )),
                "ori_deg": float(_quat_error_deg(
                    _to_np(quat_raw).reshape(4), expected_quat
                )),
            }
        except Exception:
            continue
    return out


def _rounded_payload_drift(stats: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for arm, spec in (stats or {}).items():
        if not isinstance(spec, dict):
            continue
        out[arm] = {
            "name": spec.get("name"),
            "pos_m": round(float(spec.get("pos_m", 0.0)), 5),
            "ori_deg": round(float(spec.get("ori_deg", 0.0)), 3),
        }
    return out


def _set_robot_base_pose_fast(world, pos_xyz: np.ndarray, yaw_rad: float) -> bool:
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    pos_np = np.asarray(pos_xyz, dtype=np.float64).reshape(3).copy()
    quat_np = _yaw_quat_xyzw(float(yaw_rad))
    _set_fast_base_pin(world, pos_np, float(yaw_rad))
    if getattr(world, "dry_run", False):
        try:
            world._mock_base = np.array([pos_np[0], pos_np[1], float(yaw_rad)], dtype=np.float64)
            return True
        except Exception:
            return False
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    q_saved = None
    try:
        q0 = robot.get_joint_positions()
        q_saved = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
    except Exception:
        q_saved = None
    try:
        try:
            import torch as th

            position = th.tensor(pos_np, dtype=th.float32)
            orientation = th.tensor(quat_np, dtype=th.float32)
        except Exception:
            position = pos_np
            orientation = quat_np
        robot.set_position_orientation(position=position, orientation=orientation)
        if q_saved is not None:
            try:
                robot.set_joint_positions(q_saved)
            except Exception:
                pass
        from behavior_interface.world_api import Pose

        world._last_pose = Pose(pos=pos_np.copy(), quat=quat_np.copy())
        _zero_robot_velocities(world)
        return True
    except Exception:
        return False


def _controller_dof_indices(robot, name: str) -> np.ndarray:
    ctrl = getattr(robot, "controllers", {}).get(name)
    if ctrl is None:
        return np.zeros(0, dtype=int)
    idx = getattr(ctrl, "dof_idx", None)
    if idx is None:
        return np.zeros(0, dtype=int)
    return _to_np(idx).astype(int).reshape(-1)


def _assign_q_values(q_target, indices: np.ndarray, values) -> bool:
    indices = np.asarray(indices, dtype=int).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if indices.size != vals.size:
        return False
    try:
        if hasattr(q_target, "detach"):
            import torch as th

            idx_t = th.as_tensor(indices, dtype=th.long, device=q_target.device)
            val_t = th.as_tensor(vals, dtype=q_target.dtype, device=q_target.device)
            q_target[idx_t] = val_t
        else:
            q_target[indices] = vals
        return True
    except Exception:
        return False


def _assign_action_values(action, indices: np.ndarray, values) -> bool:
    indices = np.asarray(indices, dtype=int).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if indices.size != vals.size:
        return False
    try:
        if hasattr(action, "detach"):
            import torch as th

            idx_t = th.as_tensor(indices, dtype=th.long, device=action.device)
            val_t = th.as_tensor(vals, dtype=action.dtype, device=action.device)
            action[idx_t] = val_t
        else:
            action[indices] = vals.astype(np.float32)
        return True
    except Exception:
        return False


def _base_target_q_for_pinned_world(world, bx: float, by: float, yaw_rad: float):
    """Build a full absolute-q target for robot.q_to_action while preserving limb pins."""
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return None
    q0 = robot.get_joint_positions()
    q_tgt = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()

    base_idx = _controller_dof_indices(robot, "base")
    if base_idx.size < 3:
        return None
    _assign_q_values(q_tgt, base_idx[:3], [float(bx), float(by), float(yaw_rad)])

    trunk_vals = None
    try:
        trunk_vals = world.trunk_pin_qpos_list()
    except Exception:
        trunk_vals = None
    if trunk_vals is None:
        try:
            trunk_vals = world.trunk_qpos()
        except Exception:
            trunk_vals = None
    if trunk_vals is not None:
        _assign_q_values(q_tgt, _controller_dof_indices(robot, "trunk"), trunk_vals)

    for arm in ("left", "right"):
        arm_vals = None
        try:
            arm_vals = world.arm_pin_qpos_list(arm)
        except Exception:
            arm_vals = None
        if arm_vals is None:
            try:
                arm_vals = world.arm_qpos_list(arm)
            except Exception:
                arm_vals = None
        if arm_vals is not None:
            _assign_q_values(q_tgt, _controller_dof_indices(robot, f"arm_{arm}"), arm_vals)

        grip_vals = None
        try:
            grip_vals = world.gripper_pin_qpos_list(arm)
        except Exception:
            grip_vals = None
        if grip_vals is None:
            try:
                grip_vals = world.gripper_qpos_list(arm)
            except Exception:
                grip_vals = None
        if grip_vals is not None:
            _assign_q_values(q_tgt, _controller_dof_indices(robot, f"gripper_{arm}"), grip_vals)

    return q_tgt


def _boosted_local_base_command(
    pose,
    *,
    bx: float,
    by: float,
    theta_x_deg: float,
    pos_tol: float,
    yaw_tol_deg: float,
    max_lin_vel: float = _BASE_MAX_LIN_VEL,
    max_ang_vel: float = _BASE_MAX_ANG_VEL,
) -> tuple[list[float], float, float]:
    """返回底盘期望速度（机体系 vx/vy/wz），已封顶到贴地稳定范围。

    返回值为物理单位 m/s、rad/s。线/角速度共享椭圆包络，避免横移和旋转同时
    满速；进入减速半径后按误差线性收敛。
    """
    max_lin_vel = max(0.0, float(max_lin_vel))
    max_ang_vel = max(0.0, float(max_ang_vel))
    dx = float(bx) - float(pose.pos[0])
    dy = float(by) - float(pose.pos[1])
    yaw = float(pose.yaw)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    err_f = cos_y * dx + sin_y * dy
    err_l = -sin_y * dx + cos_y * dy
    dist = float(math.hypot(dx, dy))
    yaw_err_rad = _norm_angle_rad(math.radians(float(theta_x_deg)) - yaw)

    # 线速度：朝目标方向，距离在减速半径内线性收敛，整体不超过 MAX_LIN_VEL
    if dist <= float(pos_tol):
        vx = vy = 0.0
    else:
        speed = max_lin_vel
        if dist < _BASE_LIN_SLOW_RADIUS:
            speed = max_lin_vel * (dist / _BASE_LIN_SLOW_RADIUS)
        inv = 1.0 / max(dist, 1e-9)
        vx = speed * err_f * inv
        vy = speed * err_l * inv

    # 角速度：误差在减速半径内线性收敛，不超过 MAX_ANG_VEL
    if abs(math.degrees(yaw_err_rad)) <= float(yaw_tol_deg):
        wz = 0.0
    else:
        w = max_ang_vel
        if abs(yaw_err_rad) < _BASE_ANG_SLOW_RADIUS:
            w = max_ang_vel * (abs(yaw_err_rad) / _BASE_ANG_SLOW_RADIUS)
        wz = math.copysign(w, yaw_err_rad)

    lin_ratio = math.hypot(vx, vy) / max(max_lin_vel, 1e-9)
    ang_ratio = abs(wz) / max(max_ang_vel, 1e-9)
    combined_ratio = math.hypot(lin_ratio, ang_ratio)
    if combined_ratio > 1.0:
        vx /= combined_ratio
        vy /= combined_ratio
        wz /= combined_ratio
    return [float(vx), float(vy), float(wz)], dist, abs(math.degrees(yaw_err_rad))


def _base_physics_dt(world) -> float:
    """Return simulation time advanced by one yielded controller action.

    The legacy function name is retained for callers, but this must be the
    action / sim-step duration, not the lower-level PhysX substep duration.
    """
    for attr_name in ("action_dt_s", "action_dt", "control_dt"):
        try:
            value = getattr(world, attr_name)
            dt = float(value() if callable(value) else value)
            if 1e-4 <= dt <= 0.2:
                return dt
        except Exception:
            pass
    try:
        env = getattr(world, "env", None)
        frequency = float((getattr(env, "env_config", None) or {})["action_frequency"])
        if frequency > 0.0:
            dt = 1.0 / frequency
            if 1e-4 <= dt <= 0.2:
                return dt
    except Exception:
        pass
    try:
        import omnigibson as og

        dt = float(og.sim.get_sim_step_dt())
        if 1e-4 <= dt <= 0.2:
            return dt
    except Exception:
        pass
    return 1.0 / 30.0


def _base_tilt_deg(quat_xyzw) -> float:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return 180.0
    x, y, _, _ = q / norm
    up_z = min(1.0, max(-1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(up_z))


def _base_yaw_rad(quat_xyzw) -> float:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return 0.0
    x, y, z, w = q / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _object_prim_paths(obj: Any) -> frozenset[str]:
    paths: set[str] = set()
    if obj is None:
        return frozenset()
    root = getattr(obj, "prim_path", None)
    if root:
        paths.add(str(root))
    try:
        for link in (getattr(obj, "links", None) or {}).values():
            path = getattr(link, "prim_path", None)
            if path:
                paths.add(str(path))
    except Exception:
        pass
    return frozenset(paths)


def _base_motion_collision_hits(
    world,
    *,
    ignored_external_paths: frozenset[str] = frozenset(),
) -> list[dict]:
    """Return non-ground external contacts on any robot link."""
    robot = getattr(world, "robot", None)
    if robot is None:
        return []
    try:
        contacts = robot.contact_list()
    except Exception:
        return []
    robot_paths = frozenset(
        str(path) for path in (getattr(robot, "link_prim_paths", None) or [])
    )
    if not robot_paths:
        try:
            robot_paths = frozenset(
                str(link.prim_path)
                for link in (getattr(robot, "links", None) or {}).values()
                if getattr(link, "prim_path", None)
            )
        except Exception:
            robot_paths = frozenset()

    ground_tokens = (
        "floor",
        "ground",
        "carpet",
        "rug",
        "tile_floor",
        "paver",
        "wheel",
    )
    hits: list[dict] = []
    for contact in contacts:
        body0 = str(getattr(contact, "body0", ""))
        body1 = str(getattr(contact, "body1", ""))
        body0_robot = body0 in robot_paths
        body1_robot = body1 in robot_paths
        if body0_robot == body1_robot:
            continue
        robot_path = body0 if body0_robot else body1
        other_path = body1 if body0_robot else body0
        if other_path in ignored_external_paths:
            continue
        other_lower = other_path.lower()
        if any(token in other_lower for token in ground_tokens):
            continue
        try:
            impulse = float(
                np.linalg.norm(
                    _to_np(getattr(contact, "impulse", [0.0, 0.0, 0.0]))
                )
            )
        except Exception:
            impulse = 0.0
        if impulse < _BASE_COLLISION_CONTACT_IMPULSE_MIN:
            continue
        hits.append(
            {
                "link": robot_path.rstrip("/").split("/")[-1],
                "link_path": robot_path,
                "other": other_path.rstrip("/").split("/")[-1],
                "other_path": other_path,
                "impulse": impulse,
            }
        )
    hits.sort(key=lambda item: float(item["impulse"]), reverse=True)
    return hits


def _base_ground_contact_link_count(world) -> Optional[int]:
    """Return grounded base / wheel link count, or None if contacts unavailable."""
    robot = getattr(world, "robot", None)
    if robot is None:
        return None
    try:
        contacts = robot.contact_list()
    except Exception:
        return None
    robot_paths = frozenset(
        str(path) for path in (getattr(robot, "link_prim_paths", None) or [])
    )
    if not robot_paths:
        try:
            robot_paths = frozenset(
                str(link.prim_path)
                for link in (getattr(robot, "links", None) or {}).values()
                if getattr(link, "prim_path", None)
            )
        except Exception:
            return None
    ground_tokens = (
        "floor",
        "ground",
        "carpet",
        "rug",
        "tile_floor",
        "paver",
    )
    base_tokens = (
        "base",
        "footprint",
        "wheel",
        "steer",
        "caster",
    )
    grounded_links: set[str] = set()
    for contact in contacts:
        body0 = str(getattr(contact, "body0", ""))
        body1 = str(getattr(contact, "body1", ""))
        body0_robot = body0 in robot_paths
        body1_robot = body1 in robot_paths
        if body0_robot == body1_robot:
            continue
        robot_path = body0 if body0_robot else body1
        other_path = body1 if body0_robot else body0
        if not any(token in other_path.lower() for token in ground_tokens):
            continue
        robot_link = robot_path.rstrip("/").split("/")[-1]
        if any(token in robot_link.lower() for token in base_tokens):
            grounded_links.add(robot_path)
    return len(grounded_links)


def _limit_vector(value, dim: int) -> np.ndarray:
    arr = np.asarray(_to_np(value), dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(dim, float(arr[0]), dtype=np.float64)
    if arr.size != dim:
        raise ValueError(f"limit dim={arr.size}, expected={dim}")
    return arr


def _base_physical_velocity_to_action(world, physical_cmd) -> list[float]:
    """把机体系物理速度反映射为 base controller 的输入 action。"""
    cmd = np.asarray(physical_cmd, dtype=np.float64).reshape(3)
    robot = getattr(world, "robot", None)
    controllers = getattr(robot, "controllers", None)
    controller = controllers.get("base") if isinstance(controllers, dict) else None
    input_limits = getattr(controller, "_command_input_limits", None)
    output_limits = getattr(controller, "_command_output_limits", None)
    if input_limits is None or output_limits is None:
        return [float(x) for x in cmd]
    try:
        in_lo = _limit_vector(input_limits[0], 3)
        in_hi = _limit_vector(input_limits[1], 3)
        out_lo = _limit_vector(output_limits[0], 3)
        out_hi = _limit_vector(output_limits[1], 3)
        cmd = np.clip(cmd, np.minimum(out_lo, out_hi), np.maximum(out_lo, out_hi))
        out_span = out_hi - out_lo
        if np.any(np.abs(out_span) <= 1e-12):
            raise ValueError("base controller output span is zero")
        action = in_lo + (cmd - out_lo) * (in_hi - in_lo) / out_span
        action = np.clip(action, np.minimum(in_lo, in_hi), np.maximum(in_lo, in_hi))
        return [float(x) for x in action]
    except Exception:
        return [float(x) for x in cmd]


def _base_velocity_action(world, physical_cmd):
    return world.set_base_velocity(
        *_base_physical_velocity_to_action(world, physical_cmd)
    )


def _ramp_base_cmd(
    last: list[float],
    desired: list[float],
    *,
    dt_s: float = 1.0 / 120.0,
    max_lin_accel: float = _BASE_MAX_LIN_ACCEL,
    max_ang_accel: float = _BASE_MAX_ANG_ACCEL,
) -> list[float]:
    """按物理时间限制线/角加速度，输入输出均为物理速度。"""
    last_np = np.asarray(last, dtype=np.float64).reshape(3)
    desired_np = np.asarray(desired, dtype=np.float64).reshape(3)
    out = last_np.copy()
    delta_xy = desired_np[:2] - last_np[:2]
    delta_xy_norm = float(np.linalg.norm(delta_xy))
    max_dv = max(0.0, float(max_lin_accel)) * max(1e-5, float(dt_s))
    if delta_xy_norm > max_dv > 0.0:
        delta_xy *= max_dv / delta_xy_norm
    out[:2] += delta_xy
    dw = float(desired[2]) - float(last[2])
    max_dw = max(0.0, float(max_ang_accel)) * max(1e-5, float(dt_s))
    dw = max(-max_dw, min(max_dw, dw))
    out[2] = float(last_np[2]) + dw
    return [float(x) for x in out]


def _cap_base_linear_speed(command, speed_limit_mps: float) -> list[float]:
    """Cap the physical XY command without changing its direction or yaw."""
    out = np.asarray(command, dtype=np.float64).reshape(3).copy()
    speed = float(np.linalg.norm(out[:2]))
    limit = max(0.0, float(speed_limit_mps))
    if speed > limit and speed > 1e-12:
        out[:2] *= limit / speed
    return [float(x) for x in out]


def _base_goal_brake_limit(
    *,
    distance_m: float,
    along_remaining_m: Optional[float],
    pos_tol_m: float,
    max_decel_mps2: float,
) -> tuple[float, float]:
    """Return a stop-safe linear speed and the remaining braking distance."""
    remaining = max(0.0, float(distance_m) - max(0.0, float(pos_tol_m)))
    if along_remaining_m is not None:
        remaining = min(
            remaining,
            max(0.0, float(along_remaining_m) - _BASE_GOAL_STOP_MARGIN_M),
        )
    brake_decel = max(
        1e-3,
        float(max_decel_mps2) * _BASE_BRAKE_DECEL_SAFETY,
    )
    return math.sqrt(2.0 * brake_decel * remaining), remaining


def _yield_base_xy_yaw_controller(
    ctx,
    world,
    *,
    bx: float,
    by: float,
    theta_x_deg: float,
    timeout_s: float,
    log_tag: str,
    pos_tol: float = _NAV_POS_TOL_M,
    yaw_tol_deg: float = _NAV_YAW_TOL_DEG,
) -> Generator:
    """Drive the holonomic base with closed-loop velocity controller commands.

    The server skips expensive camera obs while this loop is active, but every
    yielded value is still an ordinary controller action followed by physics.
    """
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return {
            "xy_ok": False,
            "yaw_ok": False,
            "trunk_ok": True,
            "path_status": "base_controller_unavailable",
            "waypoints_n": 1,
            "snap_goal": False,
            "stuck_recoveries": 0,
        }

    try:
        if getattr(world, "_codex_v1_fast_base_post_step_active", False):
            _restore_fast_base_post_step(world)
        if hasattr(world, "_codex_v1_fast_base_pin"):
            world._codex_v1_fast_base_pin = None
    except Exception:
        pass
    try:
        from behavior_interface.skills.eef import _freeze_world_limb_pins

        _freeze_world_limb_pins(world)
    except Exception as exc:
        ctx.log(f"[{log_tag}] WARN base controller freeze limb pins failed: {exc}")
    limb_hold = _snapshot_fast_base_limb_hold(world)
    payload_hold = _snapshot_assisted_payload_hold(world)
    payload_names = {
        arm: spec.get("name") for arm, spec in payload_hold.items()
    }
    ctx.log(
        f"[{log_tag}] 普通②xy③yaw action-only lock arms="
        f"{[name.removeprefix('arm_') for name in ('arm_left', 'arm_right') if name in limb_hold]} "
        f"payloads={payload_names}"
    )

    t0 = time.time()
    drive_steps = 0
    physics_steps = 0
    last_xy_err = float("inf")
    last_yaw_err = float("inf")
    last_log_t = 0.0
    last_cmd = [0.0, 0.0, 0.0]
    action_dt = _base_physics_dt(world)
    monitor_steps = max(
        1, int(math.floor(1.0 / (_BASE_GROUND_MONITOR_HZ * action_dt)))
    )
    monitor_actual_hz = 1.0 / (monitor_steps * action_dt)
    max_lin_vel = (
        _BASE_PAYLOAD_MAX_LIN_VEL if payload_hold else _BASE_MAX_LIN_VEL
    )
    max_ang_vel = (
        _BASE_PAYLOAD_MAX_ANG_VEL if payload_hold else _BASE_MAX_ANG_VEL
    )
    max_lin_accel = (
        _BASE_PAYLOAD_MAX_LIN_ACCEL if payload_hold else _BASE_MAX_LIN_ACCEL
    )
    max_ang_accel = (
        _BASE_PAYLOAD_MAX_ANG_ACCEL if payload_hold else _BASE_MAX_ANG_ACCEL
    )
    speed_scale = 1.0
    ground_recoveries = 0
    recovering_ground = False
    grounded_stable_samples = 0
    initial_pose = world.robot_pose()
    initial_xy = np.asarray(initial_pose.pos[:2], dtype=np.float64).reshape(2)
    goal_xy = np.asarray([float(bx), float(by)], dtype=np.float64)
    goal_delta = goal_xy - initial_xy
    goal_path_len = float(np.linalg.norm(goal_delta))
    goal_path_unit = (
        goal_delta / goal_path_len
        if goal_path_len > max(1e-6, float(pos_tol))
        else None
    )
    max_goal_overshoot_m = 0.0
    z0_base = float(initial_pose.pos[2])
    z_min = z0_base
    z_max = z0_base
    ground_z_ref = z0_base
    last_monitor_z = z0_base
    last_monitor_step = -monitor_steps
    max_abs_vz = 0.0
    initial_tilt_deg = _base_tilt_deg(initial_pose.quat)
    max_tilt_deg = initial_tilt_deg
    max_tilt_rise_deg = 0.0
    max_ground_rise = 0.0
    ignored_external_paths = frozenset(
        path
        for spec in payload_hold.values()
        for path in _object_prim_paths(spec.get("object"))
    )
    contact_streak = 0
    no_progress_s = 0.0
    progress_ref_xy_err: Optional[float] = None
    progress_ref_yaw_err: Optional[float] = None
    ctx.log(
        f"[{log_tag}] 普通②xy③yaw velocity 控制器目标: "
        f"xy=({float(bx):.2f},{float(by):.2f}) yaw={float(theta_x_deg):.1f}° "
        f"timeout={float(timeout_s):.1f}s "
        f"vmax={max_lin_vel:.2f}m/s wmax={max_ang_vel:.2f}rad/s "
        f"action_dt={action_dt:.4f}s "
        f"ground_guard={monitor_actual_hz:.1f}Hz"
    )

    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    try:
        world._codex_fast_motion_no_obs = True
    except Exception:
        pass
    try:
        while True:
            pose = world.robot_pose()
            z_now = float(pose.pos[2])
            sample_xy_err = float(
                math.hypot(float(bx) - float(pose.pos[0]), float(by) - float(pose.pos[1]))
            )
            sample_yaw_err = abs(
                _norm_angle_deg(
                    float(theta_x_deg) - math.degrees(float(pose.yaw))
                )
            )
            along_remaining = None
            if goal_path_unit is not None:
                current_xy = np.asarray(pose.pos[:2], dtype=np.float64).reshape(2)
                along_remaining = float(np.dot(goal_xy - current_xy, goal_path_unit))
                max_goal_overshoot_m = max(
                    max_goal_overshoot_m,
                    max(0.0, -along_remaining),
                )
                if along_remaining < -_BASE_GOAL_PLANE_TOL_M:
                    last_cmd = [0.0, 0.0, 0.0]
                    ctx.log(
                        f"[{log_tag}] [终点守护] 越过规划终点，立即零速且禁止倒车穿回 "
                        f"overshoot={-along_remaining:.3f}m "
                        f"xy_err={sample_xy_err:.3f}m yaw_err={sample_yaw_err:.1f}°"
                    )
                    for _ in range(_BASE_GOAL_HARD_STOP_STEPS):
                        physics_steps += 1
                        yield _base_velocity_action(world, last_cmd)
                    return {
                        "xy_ok": False,
                        "yaw_ok": False,
                        "trunk_ok": True,
                        "ok": False,
                        "status": "goal_overshoot",
                        "reason": "goal_overshoot",
                        "error": "底盘越过规划终点，已急停",
                        "path_status": "velocity_base_goal_overshoot",
                        "waypoints_n": 1,
                        "snap_goal": False,
                        "stuck_recoveries": 0,
                        "ground_recoveries": int(ground_recoveries),
                        "safety_abort": True,
                        "steps": int(drive_steps),
                        "physics_steps": int(physics_steps),
                        "xy_err_m": round(sample_xy_err, 4),
                        "yaw_err_deg": round(sample_yaw_err, 2),
                        "goal_overshoot_m": round(-along_remaining, 4),
                        "max_goal_overshoot_m": round(max_goal_overshoot_m, 4),
                        "action_dt_s": round(action_dt, 6),
                        "elapsed_s": round(float(time.time() - t0), 3),
                    }
            z_min = min(z_min, z_now)
            z_max = max(z_max, z_now)
            monitor_due = physics_steps % monitor_steps == 0
            just_recovered = False
            if monitor_due:
                sample_steps = max(1, physics_steps - last_monitor_step)
                sample_dt = max(action_dt, sample_steps * action_dt)
                vz = (z_now - last_monitor_z) / sample_dt
                tilt_deg = _base_tilt_deg(pose.quat)
                tilt_rise_deg = max(0.0, tilt_deg - initial_tilt_deg)
                ground_z_ref = min(ground_z_ref, z_now)
                ground_rise = z_now - ground_z_ref
                z_rise_from_start = z_now - z0_base
                max_abs_vz = max(max_abs_vz, abs(vz))
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                max_tilt_rise_deg = max(max_tilt_rise_deg, tilt_rise_deg)
                max_ground_rise = max(max_ground_rise, ground_rise)

                if recovering_ground:
                    contact_streak = 0
                    no_progress_s = 0.0
                    progress_ref_xy_err = None
                    progress_ref_yaw_err = None
                    grounded_now = bool(
                        z_now <= ground_z_ref + _BASE_GROUNDED_Z_MARGIN_M
                        and abs(vz) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                        and tilt_rise_deg <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
                    )
                    grounded_stable_samples = (
                        grounded_stable_samples + 1 if grounded_now else 0
                    )
                    if grounded_stable_samples >= _BASE_GROUNDED_STABLE_SAMPLES:
                        recovering_ground = False
                        grounded_stable_samples = 0
                        last_cmd = [0.0, 0.0, 0.0]
                        speed_scale = max(
                            _BASE_MIN_RESUME_SPEED_FACTOR,
                            _BASE_RESUME_SPEED_FACTOR ** ground_recoveries,
                        )
                        just_recovered = True
                        ctx.log(
                            f"[{log_tag}] [离地守护] 已稳定落地，resume 导航 "
                            f"recovery={ground_recoveries} z={z_now:.4f} "
                            f"vz={vz:+.3f}m/s tilt={tilt_deg:.1f}° "
                            f"tilt_rise={tilt_rise_deg:+.1f}° "
                            f"speed_scale={speed_scale:.2f}"
                        )
                else:
                    stop_reasons = []
                    if z_rise_from_start >= _BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M:
                        stop_reasons.append(
                            f"z_from_start={z_rise_from_start:.3f}"
                            f">={_BASE_AIRBORNE_Z_FROM_START_HARD_STOP_M:.3f}"
                        )
                    if ground_rise >= _BASE_AIRBORNE_RISE_STOP_M:
                        stop_reasons.append(
                            f"rise={ground_rise:.3f}>={_BASE_AIRBORNE_RISE_STOP_M:.3f}"
                        )
                    if (
                        ground_rise >= _BASE_AIRBORNE_RISE_VZ_GATE_M
                        and vz >= _BASE_AIRBORNE_UP_VEL_STOP_MPS
                    ):
                        stop_reasons.append(
                            f"rise={ground_rise:.3f},vz={vz:+.3f}"
                        )
                    if tilt_rise_deg >= _BASE_AIRBORNE_TILT_RISE_STOP_DEG:
                        stop_reasons.append(
                            f"tilt_rise={tilt_rise_deg:.1f}"
                            f">={_BASE_AIRBORNE_TILT_RISE_STOP_DEG:.1f}"
                        )
                    if stop_reasons:
                        recovering_ground = True
                        grounded_stable_samples = 0
                        ground_recoveries += 1
                        last_cmd = [0.0, 0.0, 0.0]
                        ctx.log(
                            f"[{log_tag}] [离地守护] 20Hz检测触发急停 "
                            f"recovery={ground_recoveries} "
                            f"reason={';'.join(stop_reasons)} "
                            f"z={z_now:.4f} ground_z={ground_z_ref:.4f} "
                            f"vz={vz:+.3f}m/s tilt={tilt_deg:.1f}° "
                            f"tilt0={initial_tilt_deg:.1f}° "
                            f"tilt_rise={tilt_rise_deg:+.1f}°"
                        )

                if not recovering_ground:
                    collision_hits = _base_motion_collision_hits(
                        world,
                        ignored_external_paths=ignored_external_paths,
                    )
                    contact_streak = contact_streak + 1 if collision_hits else 0
                    max_impulse = (
                        float(collision_hits[0]["impulse"]) if collision_hits else 0.0
                    )
                    contact_triggered = bool(
                        collision_hits
                        and (
                            max_impulse >= _BASE_COLLISION_CONTACT_IMPULSE_HARD
                            or contact_streak
                            >= _BASE_COLLISION_CONTACT_PERSIST_SAMPLES
                        )
                    )

                    command_active = _base_command_active_for_stuck(last_cmd)
                    if command_active:
                        if progress_ref_xy_err is None:
                            progress_ref_xy_err = sample_xy_err
                            progress_ref_yaw_err = sample_yaw_err
                            no_progress_s = 0.0
                        else:
                            ref_xy_err = float(progress_ref_xy_err)
                            ref_yaw_err = float(progress_ref_yaw_err or 0.0)
                            xy_progress = float(progress_ref_xy_err - sample_xy_err)
                            yaw_progress = float(
                                ref_yaw_err - sample_yaw_err
                            )
                            xy_needed = ref_xy_err > float(pos_tol)
                            yaw_needed = ref_yaw_err > float(yaw_tol_deg)
                            progressed = bool(
                                (
                                    xy_needed
                                    and xy_progress >= _BASE_STUCK_XY_PROGRESS_M
                                )
                                or (
                                    yaw_needed
                                    and yaw_progress
                                    >= _BASE_STUCK_YAW_PROGRESS_DEG
                                )
                            )
                            if not xy_needed and not yaw_needed:
                                no_progress_s = 0.0
                                progress_ref_xy_err = None
                                progress_ref_yaw_err = None
                            elif progressed:
                                progress_ref_xy_err = sample_xy_err
                                progress_ref_yaw_err = sample_yaw_err
                                no_progress_s = 0.0
                            else:
                                no_progress_s += sample_dt
                    else:
                        no_progress_s = 0.0
                        progress_ref_xy_err = None
                        progress_ref_yaw_err = None
                    stuck_triggered = no_progress_s >= _BASE_STUCK_WINDOW_S

                    if contact_triggered or stuck_triggered:
                        yield _base_velocity_action(world, [0.0, 0.0, 0.0])
                        reason = "contact" if contact_triggered else "no_progress"
                        top_hit = collision_hits[0] if collision_hits else None
                        grounded_at_abort = bool(
                            ground_rise <= _BASE_GROUNDED_Z_MARGIN_M
                            and abs(vz) <= _BASE_GROUNDED_MAX_ABS_VZ_MPS
                            and tilt_deg <= 5.0
                            and tilt_rise_deg
                            <= _BASE_GROUNDED_MAX_TILT_RISE_DEG
                        )
                        ctx.log(
                            f"[{log_tag}] [底盘碰撞守护] 20Hz急停 "
                            f"reason={reason} xy_err={sample_xy_err:.3f}m "
                            f"yaw_err={sample_yaw_err:.1f}° "
                            f"cmd=({last_cmd[0]:+.2f},{last_cmd[1]:+.2f},"
                            f"{last_cmd[2]:+.2f}) no_progress={no_progress_s:.2f}s "
                            f"contact={top_hit} grounded_now={grounded_at_abort} "
                            f"rise={ground_rise:+.4f}m vz={vz:+.3f}m/s "
                            f"tilt={tilt_deg:.1f}°"
                        )
                        return {
                            "xy_ok": False,
                            "yaw_ok": False,
                            "trunk_ok": True,
                            "ok": False,
                            "status": "stuck_or_collision",
                            "reason": "stuck_or_collision",
                            "error": "stuck或者碰撞",
                            "path_status": "velocity_base_stuck_or_collision",
                            "waypoints_n": 1,
                            "snap_goal": False,
                            "stuck_recoveries": 0,
                            "ground_recoveries": int(ground_recoveries),
                            "safety_abort": True,
                            "collision_guard_reason": reason,
                            "collision_contact": top_hit,
                            "grounded_at_abort": grounded_at_abort,
                            "abort_z_m": round(z_now, 4),
                            "ground_z_ref_m": round(ground_z_ref, 4),
                            "abort_ground_rise_m": round(ground_rise, 4),
                            "abort_vz_mps": round(vz, 3),
                            "abort_tilt_deg": round(tilt_deg, 2),
                            "abort_tilt_rise_deg": round(tilt_rise_deg, 2),
                            "steps": int(drive_steps),
                            "physics_steps": int(physics_steps),
                            "xy_err_m": round(sample_xy_err, 4),
                            "yaw_err_deg": round(sample_yaw_err, 2),
                            "max_ground_rise_m": round(max_ground_rise, 4),
                            "max_abs_vz_mps": round(max_abs_vz, 3),
                            "initial_tilt_deg": round(initial_tilt_deg, 2),
                            "max_tilt_deg": round(max_tilt_deg, 2),
                            "max_tilt_rise_deg": round(max_tilt_rise_deg, 2),
                            "elapsed_s": round(float(time.time() - t0), 3),
                        }

                last_monitor_z = z_now
                last_monitor_step = physics_steps

            if time.time() - t0 > float(timeout_s):
                yield _base_velocity_action(world, [0.0, 0.0, 0.0])
                path_status = (
                    "velocity_base_ground_recovery_timeout"
                    if recovering_ground
                    else "velocity_base_timeout"
                )
                ctx.log(
                    f"[{log_tag}] 普通②xy③yaw velocity TIMEOUT "
                    f"xy_err={last_xy_err:.3f}m yaw_err={last_yaw_err:.1f}° "
                    f"drive_steps={drive_steps} physics_steps={physics_steps} "
                    f"recovering_ground={recovering_ground} "
                    f"[Z监控] z0={z0_base:.4f} zmin={z_min:.4f} zmax={z_max:.4f} "
                    f"抬升={z_max - z0_base:+.4f}m 下沉={z_min - z0_base:+.4f}m"
                )
                return {
                    "xy_ok": False,
                    "yaw_ok": False,
                    "trunk_ok": True,
                    "path_status": path_status,
                    "waypoints_n": 1,
                    "snap_goal": False,
                    "stuck_recoveries": 0,
                    "ground_recoveries": int(ground_recoveries),
                    "safety_abort": bool(recovering_ground),
                    "steps": int(drive_steps),
                    "physics_steps": int(physics_steps),
                    "xy_err_m": round(last_xy_err, 4),
                    "yaw_err_deg": round(last_yaw_err, 2),
                    "max_ground_rise_m": round(max_ground_rise, 4),
                    "max_abs_vz_mps": round(max_abs_vz, 3),
                    "initial_tilt_deg": round(initial_tilt_deg, 2),
                    "max_tilt_deg": round(max_tilt_deg, 2),
                    "max_tilt_rise_deg": round(max_tilt_rise_deg, 2),
                    "elapsed_s": round(float(time.time() - t0), 3),
                }

            if recovering_ground or just_recovered:
                ctx.set_status(
                    f"base ground recovery #{ground_recoveries}"
                    if recovering_ground
                    else f"base resume after recovery #{ground_recoveries}"
                )
                physics_steps += 1
                yield _base_velocity_action(world, [0.0, 0.0, 0.0])
                continue

            desired_cmd, last_xy_err, last_yaw_err = _boosted_local_base_command(
                pose,
                bx=float(bx),
                by=float(by),
                theta_x_deg=float(theta_x_deg),
                pos_tol=float(pos_tol),
                yaw_tol_deg=float(yaw_tol_deg),
                max_lin_vel=max_lin_vel * speed_scale,
                max_ang_vel=max_ang_vel * speed_scale,
            )
            brake_speed_limit, brake_remaining = _base_goal_brake_limit(
                distance_m=last_xy_err,
                along_remaining_m=along_remaining,
                pos_tol_m=float(pos_tol),
                max_decel_mps2=max_lin_accel,
            )
            desired_cmd = _cap_base_linear_speed(
                desired_cmd,
                brake_speed_limit,
            )
            if last_xy_err <= float(pos_tol) and last_yaw_err <= float(yaw_tol_deg):
                # 进入终点容差后不再允许任何平移命令。
                desired_cmd = [0.0, 0.0, 0.0]
            if (
                last_xy_err <= float(pos_tol)
                and last_yaw_err <= float(yaw_tol_deg)
                and max(abs(v) for v in last_cmd) <= 1e-3
            ):
                last_cmd = [0.0, 0.0, 0.0]
                for _ in range(4):
                    yield _base_velocity_action(world, last_cmd)
                ctx.log(
                    f"[{log_tag}] 普通②xy③yaw velocity 到位 "
                    f"xy_err={last_xy_err:.3f}m yaw_err={last_yaw_err:.1f}° "
                    f"drive_steps={drive_steps} physics_steps={physics_steps} "
                    f"ground_recoveries={ground_recoveries} "
                    f"[Z监控] z0={z0_base:.4f} zmin={z_min:.4f} zmax={z_max:.4f} "
                    f"抬升={z_max - z0_base:+.4f}m 下沉={z_min - z0_base:+.4f}m"
                )
                return {
                    "xy_ok": True,
                    "yaw_ok": True,
                    "trunk_ok": True,
                    "path_status": (
                        "velocity_base_no_obs_ground_resume"
                        if ground_recoveries
                        else "velocity_base_no_obs"
                    ),
                    "waypoints_n": 1,
                    "snap_goal": False,
                    "stuck_recoveries": 0,
                    "ground_recoveries": int(ground_recoveries),
                    "safety_abort": False,
                    "steps": int(drive_steps),
                    "physics_steps": int(physics_steps),
                    "xy_err_m": round(last_xy_err, 4),
                    "yaw_err_deg": round(last_yaw_err, 2),
                    "max_ground_rise_m": round(max_ground_rise, 4),
                    "max_abs_vz_mps": round(max_abs_vz, 3),
                    "initial_tilt_deg": round(initial_tilt_deg, 2),
                    "max_tilt_deg": round(max_tilt_deg, 2),
                    "max_tilt_rise_deg": round(max_tilt_rise_deg, 2),
                    "max_goal_overshoot_m": round(max_goal_overshoot_m, 4),
                    "action_dt_s": round(action_dt, 6),
                    "elapsed_s": round(float(time.time() - t0), 3),
                }

            physical_cmd = _ramp_base_cmd(
                last_cmd,
                desired_cmd,
                dt_s=action_dt,
                max_lin_accel=max_lin_accel,
                max_ang_accel=max_ang_accel,
            )
            # Hard invariant: the issued velocity itself may never exceed the
            # remaining-distance stopping envelope, even after ramping.
            physical_cmd = _cap_base_linear_speed(
                physical_cmd,
                brake_speed_limit,
            )
            last_cmd = physical_cmd
            action_cmd = _base_physical_velocity_to_action(world, physical_cmd)

            now = time.time()
            if drive_steps == 0 or now - last_log_t >= 1.0:
                ctx.log(
                    f"[{log_tag}] 普通②xy③yaw velocity step={drive_steps} "
                    f"xy_err={last_xy_err:.3f}m yaw_err={last_yaw_err:.1f}° "
                    f"cmd_phys=({physical_cmd[0]:+.2f},{physical_cmd[1]:+.2f},"
                    f"{physical_cmd[2]:+.2f}) action=({action_cmd[0]:+.2f},"
                    f"{action_cmd[1]:+.2f},{action_cmd[2]:+.2f}) "
                    f"vstop={brake_speed_limit:.2f} rem_stop={brake_remaining:.3f}m "
                    f"z={z_now:.4f} dz_ground={z_now - ground_z_ref:+.4f}"
                )
                last_log_t = now
            drive_steps += 1
            physics_steps += 1
            ctx.set_status(
                f"base velocity xy_err={last_xy_err:.2f} yaw_err={last_yaw_err:.1f}"
            )
            yield world.set_base_velocity(*action_cmd)
    finally:
        limb_end_drift = _fast_base_limb_drift(world, limb_hold)
        payload_end_drift = _payload_end_drift(world, payload_hold)
        ctx.log(
            f"[{log_tag}] 普通②xy③yaw action-only summary "
            f"limb_end={{{', '.join(f'{name}: {float(err):.5f}' for name, err in limb_end_drift.items())}}} "
            f"payload_end={_rounded_payload_drift(payload_end_drift)}"
        )
        try:
            world._codex_fast_motion_no_obs = old_no_obs
        except Exception:
            pass


def _yield_fast_base_xy_yaw(
    ctx,
    world,
    *,
    bx: float,
    by: float,
    theta_x_deg: float,
    max_duration_s: float = _FAST_BASE_XY_YAW_MAX_S,
    controller_timeout_s: Optional[float] = None,
    log_tag: str,
) -> Generator:
    """Compatibility entry point backed only by controller actions.

    The server's no-observation stepping path keeps this efficient without
    bypassing collision detection or injecting simulator state.
    """
    timeout = (
        float(controller_timeout_s)
        if controller_timeout_s is not None
        else max(30.0, float(max_duration_s))
    )
    ctx.log(
        f"[{log_tag}] action-only base controller "
        f"(legacy fast root-pose path disabled) timeout={timeout:.1f}s"
    )
    report = yield from _yield_base_xy_yaw_controller(
        ctx,
        world,
        bx=float(bx),
        by=float(by),
        theta_x_deg=float(theta_x_deg),
        timeout_s=timeout,
        log_tag=log_tag,
    )
    report = dict(report or {})
    report["ok"] = bool(report.get("xy_ok")) and bool(report.get("yaw_ok"))
    report["mode"] = "base_velocity_action_only"
    report["action_only"] = True
    report["requested_fast_duration_s"] = round(float(max_duration_s), 3)
    return report


def _set_trunk_qpos_fast(world, target_q) -> bool:
    if _challenge_action_only() and not getattr(world, "dry_run", False):
        return False
    q_arr = np.asarray(target_q, dtype=np.float64).reshape(4)
    if getattr(world, "dry_run", False):
        return True
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    try:
        q_full = robot.get_joint_positions()
        q_new = q_full.clone() if hasattr(q_full, "clone") else np.asarray(q_full, dtype=np.float64).copy()
        tidx_raw = robot.trunk_control_idx
        if hasattr(tidx_raw, "detach"):
            tidx_raw = tidx_raw.detach().cpu().numpy()
        tidx = np.asarray(tidx_raw, dtype=int).reshape(-1)[:4]
        if len(tidx) < 4:
            return False
        for local_i, joint_i in enumerate(tidx):
            q_new[int(joint_i)] = float(q_arr[int(local_i)])
        robot.set_joint_positions(q_new)
        try:
            vel = robot.get_joint_velocities()
            v_new = vel.clone() if hasattr(vel, "clone") else np.asarray(vel, dtype=np.float64).copy()
            for joint_i in tidx:
                v_new[int(joint_i)] = 0.0
            robot.set_joint_velocities(v_new)
        except Exception:
            pass
        if hasattr(world, "set_trunk_pin_qpos"):
            world.set_trunk_pin_qpos(q_arr.tolist())
        return True
    except Exception:
        return False


def _yield_fast_trunk_qpos(
    ctx,
    world,
    target_q,
    *,
    max_duration_s: float = _FAST_TRUNK_MAX_S,
    log_prefix: str,
    q_tol_rad: float = 0.035,
) -> Generator:
    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    q_tgt = np.asarray(target_q, dtype=np.float64).reshape(4)
    dq_inf = float(np.max(np.abs(q_tgt - q0)))
    t0 = time.time()
    ctx.log(
        f"{log_prefix} fast trunk direct qpos "
        f"dq_inf={dq_inf:.3f}rad max={max_duration_s:.1f}s "
        f"target={[round(float(x), 4) for x in q_tgt.tolist()]}"
    )
    ok_set = _set_trunk_qpos_fast(world, q_tgt)
    hold_n = 1
    for _ in range(max(1, int(hold_n))):
        if hasattr(world, "make_action_trunk_locked"):
            yield world.make_action_trunk_locked(q_tgt.tolist())
        else:
            yield world.hold_action_pinned() if hasattr(world, "hold_action_pinned") else world.empty_action()
        # The absolute-position trunk controller can pull the joint state back
        # during the physics step. Re-apply the direct qpos after each step so
        # the final measured pose matches the fast target, not an intermediate
        # controller transient.
        ok_set = bool(_set_trunk_qpos_fast(world, q_tgt)) and bool(ok_set)
        _force_apply_fast_base_pin(world)
    ok_set = bool(_set_trunk_qpos_fast(world, q_tgt)) and bool(ok_set)
    _force_apply_fast_base_pin(world)
    q_fin = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    chest = world.chest_pose()
    q_err = float(np.max(np.abs(q_fin - q_tgt)))
    stats = {
        "reached": bool(ok_set and q_err <= float(q_tol_rad)),
        "mode": "fast_trunk_direct_qpos",
        "elapsed_s": round(float(time.time() - t0), 3),
        "q_start": [round(float(x), 4) for x in q0.tolist()],
        "q_target": [round(float(x), 4) for x in q_tgt.tolist()],
        "q_end": [round(float(x), 4) for x in q_fin.tolist()],
        "q_err_inf_rad": round(q_err, 4),
        "theta_z_deg": round(float(chest["theta_z_deg"]), 2),
        "chest_z": round(float(chest["z"]), 4),
        "set_ok": bool(ok_set),
    }
    ctx.log(
        f"{log_prefix} fast trunk done elapsed={stats['elapsed_s']:.1f}s "
        f"q_err={q_err:.4f}rad θz={chest['theta_z_deg']:.1f}° "
        f"z={chest['z']:.3f} ok={stats['reached']}"
    )
    return stats


def _yield_fast_trunk_theta_z(
    ctx,
    world,
    *,
    theta_z_tgt: float,
    max_duration_s: float = _FAST_TRUNK_MAX_S,
    log_prefix: str,
) -> Generator:
    from behavior_interface.trunk_vertical_lift import solve_q3_for_theta_z_holding_q12

    q_now = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    q3 = solve_q3_for_theta_z_holding_q12(
        float(theta_z_tgt),
        float(q_now[0]),
        float(q_now[1]),
        prefer_q3=float(q_now[2]),
    )
    if q3 is None:
        ctx.log(f"{log_prefix} fast trunk θz no q3 solution for {theta_z_tgt:.1f}°")
        return {
            "reached": False,
            "mode": "fast_trunk_theta_z_q3",
            "error": "solve_q3_for_theta_z_holding_q12 failed",
        }
    q_tgt = q_now.copy()
    q_tgt[2] = float(q3)
    stats = yield from _yield_fast_trunk_qpos(
        ctx,
        world,
        q_tgt,
        max_duration_s=max_duration_s,
        log_prefix=log_prefix,
        q_tol_rad=0.035,
    )
    chest = world.chest_pose()
    theta_err = abs(_norm_angle_deg(float(theta_z_tgt) - float(chest["theta_z_deg"])))
    stats.update({
        "mode": "fast_trunk_theta_z_q3",
        "theta_z_target_deg": round(float(theta_z_tgt), 2),
        "theta_z_err_deg": round(float(theta_err), 2),
        "reached": bool(stats.get("reached") and theta_err <= _NAV_THETA_Z_TOL_DEG),
    })
    return stats


def _pose_errors(
    world,
    bx: float,
    by: float,
    chest_z: float,
    theta_x_deg: float,
    theta_z_deg: float,
) -> Dict[str, float]:
    """当前位姿相对规划目标的误差。"""
    pose = world.robot_pose()
    chest = world.chest_pose()
    xy_err = float(math.hypot(float(pose.pos[0]) - bx, float(pose.pos[1]) - by))
    yaw_err = _norm_angle_deg(theta_x_deg - math.degrees(float(pose.yaw)))
    z_err = float(chest_z - chest["z"])
    tz_err = _norm_angle_deg(theta_z_deg - chest["theta_z_deg"])
    return {
        "xy_m": round(xy_err, 4),
        "yaw_deg": round(yaw_err, 2),
        "z_m": round(z_err, 4),
        "theta_z_deg": round(tz_err, 2),
    }


def _pose_within_tol(errors: Dict[str, float]) -> bool:
    return (
        errors["xy_m"] <= _NAV_POS_TOL_M
        and abs(errors["yaw_deg"]) <= _NAV_YAW_TOL_DEG
        and abs(errors["theta_z_deg"]) <= _NAV_THETA_Z_TOL_DEG
    )


def _aabb_around_point(center: np.ndarray, margin: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """以 3D 点为中心的小 AABB，供 head 取景投影用。"""
    m = float(margin)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    return c - m, c + m


def _yield_pre_descent_for_point(
    ctx,
    target_center: np.ndarray,
    *,
    dz_target_max_m: float = POINT_PRE_DESCENT_DZ_MAX_M,
    log_tag: str,
) -> Generator:
    """真机第 0 步：dz=肩高−目标z 过大时，协同 q1/q2/q3 垂直预降。"""
    from behavior_interface.skills.move_to import _base_link_z_world
    from behavior_interface.skills.move_to_object_geom import read_shoulder_head_pose
    from behavior_interface.trunk_vertical_lift import (
        plan_pre_descent_for_point_dz,
        solve_q3_for_theta_z_holding_q12,
    )

    world = ctx.world
    target_z = float(np.asarray(target_center, dtype=np.float64).reshape(3)[2])
    try:
        pose = read_shoulder_head_pose(world)
    except Exception as e:
        ctx.log(f"[{log_tag}] 预降跳过: 读肩位失败 {e}")
        return {"ok": True, "skipped": True, "reason": str(e)}

    shoulder_z = float(pose["shoulder_mid"][2])
    chest_z = float(pose["chest_z_now"])
    tz_hold = float(pose["theta_z_now_deg"])
    dz0 = shoulder_z - target_z

    if dz0 <= float(dz_target_max_m) + 1e-3:
        ctx.log(
            f"[{log_tag}] ①dz 跳过: dz={dz0:.3f}m ≤ {dz_target_max_m}m "
            f"(肩z={shoulder_z:.3f} 目标z={target_z:.3f})"
        )
        return {"ok": True, "skipped": True, "dz_start_m": round(dz0, 4)}

    base_z = _base_link_z_world(world)
    q0 = world.trunk_qpos()
    waypoints, meta = plan_pre_descent_for_point_dz(
        q0, chest_z, shoulder_z, target_z, base_z, dz_target_max_m=dz_target_max_m,
    )
    if meta.get("skipped"):
        ctx.log(f"[{log_tag}] ①dz 规划跳过: {meta.get('reason', '')}")
        return meta
    if not meta.get("ok") or not waypoints:
        ctx.log(f"[{log_tag}] ①dz 规划失败: {meta.get('error', '未知')}")
        return meta

    ctx.log(
        f"[{log_tag}] ①dz 协同预降: dz {meta['dz_start_m']:.3f}→目标≤{dz_target_max_m}m "
        f"胸口 {meta['chest_z_start']:.3f}→{meta['chest_z_end_plan']:.3f}m "
        f"θz守护={tz_hold:.1f}° waypoints={len(waypoints)}"
        + (f" [{meta.get('note')}]" if meta.get("note") else "")
    )
    for i, q_wp in enumerate(waypoints):
        q_hold = np.asarray(q_wp, dtype=np.float64).reshape(4).copy()
        # 保持 q1/q2 路点，反解 q3 锁住预降前俯仰，避免 dz 折腰叠加到末段俯仰
        nq3 = solve_q3_for_theta_z_holding_q12(
            tz_hold,
            float(q_hold[0]),
            float(q_hold[1]),
            prefer_q3=float(q_hold[2]),
        )
        if nq3 is not None:
            q_hold[2] = nq3
        for _ in range(3):
            yield _make_trunk_locked_action(world, q_hold.tolist())
        if i % 4 == 0 or i == len(waypoints) - 1:
            chest = world.chest_pose()
            sl = world.shoulder_pose("left")
            sr = world.shoulder_pose("right")
            sh_z = 0.5 * (float(sl["z"]) + float(sr["z"]))
            dz_now = sh_z - target_z
            ctx.set_status(
                f"pre_descent {i+1}/{len(waypoints)} "
                f"dz≈{dz_now:.3f} z_chest={chest['z']:.3f}"
            )

    chest1 = world.chest_pose()
    sl = world.shoulder_pose("left")
    sr = world.shoulder_pose("right")
    sh_z1 = 0.5 * (float(sl["z"]) + float(sr["z"]))
    tz_end = float(chest1["theta_z_deg"])
    meta["dz_end_m"] = round(sh_z1 - target_z, 4)
    meta["chest_z_end"] = round(float(chest1["z"]), 4)
    meta["theta_z_hold_deg"] = round(tz_hold, 2)
    meta["theta_z_end_deg"] = round(tz_end, 2)
    ctx.log(
        f"[{log_tag}] ①dz 完成: dz {meta['dz_start_m']:.3f}→{meta['dz_end_m']:.3f}m "
        f"胸口 {meta['chest_z_start']:.3f}→{meta['chest_z_end']:.3f}m "
        f"θz {tz_hold:.1f}°→{tz_end:.1f}°"
    )
    if abs(tz_end - tz_hold) > _PRE_DESCENT_THETA_Z_TOL_DEG:
        ctx.log(
            f"[{log_tag}] ①dz 警告: θz 偏离守护 "
            f"{tz_hold:.1f}°→{tz_end:.1f}° (>{_PRE_DESCENT_THETA_Z_TOL_DEG}°)"
        )
    return meta


def _yield_vertical_lift_before_nav(
    ctx,
    world,
    target_center: np.ndarray,
    chest_z_tgt: float,
    deltaz_m: float,
    log_tag: str,
) -> dict:
    """真机 ①升降：复用 move_in_robot_coord 的相对 LUT，先把肩高-目标z压到 0.4m 内。"""
    from behavior_interface.skills.move_to import yield_reverse_upward_relative_lut
    from behavior_interface.skills.move_to_object_geom import read_shoulder_head_pose

    target_z = float(np.asarray(target_center, dtype=np.float64).reshape(3)[2])
    pose0 = read_shoulder_head_pose(world)
    shoulder_z0 = float(pose0["shoulder_mid"][2])
    chest0 = world.chest_pose()
    dz0 = shoulder_z0 - target_z
    out = {
        "ok": True,
        "mode": "move_in_robot_coord_relative_lut",
        "z_reserved_m": round(float(deltaz_m), 4),
        "dz_start_m": round(float(dz0), 4),
        "shoulder_z_start_m": round(float(shoulder_z0), 4),
        "target_z_m": round(float(target_z), 4),
    }
    if dz0 <= POINT_PRE_DESCENT_DZ_MAX_M + 1e-3:
        ctx.log(
            f"[{log_tag}] ①升降跳过: dz={dz0:.3f}m ≤ {POINT_PRE_DESCENT_DZ_MAX_M:.2f}m "
            f"(肩z={shoulder_z0:.3f} 目标z={target_z:.3f})"
        )
        out["skipped"] = True
        return out

    desired_drop = max(0.0, float(dz0) - POINT_PRE_DESCENT_DZ_MAX_M)
    chest_z_target = float(chest0["z"]) - desired_drop
    ctx.log(
        f"[{log_tag}] ①升降复用 move_in_robot_coord: "
        f"dz {dz0:.3f}→≤{POINT_PRE_DESCENT_DZ_MAX_M:.2f}m "
        f"胸口目标 {chest0['z']:.3f}→{chest_z_target:.3f}m "
        f"drop={desired_drop:.3f}m"
    )
    lift = yield from yield_reverse_upward_relative_lut(
        ctx,
        world,
        chest_z_target=chest_z_target,
        theta_z_deg=90.0,
        z_tol=0.04,
        log_prefix=f"{log_tag}/lift",
        max_duration_s=15.0,
        max_fast_waypoints=2,
    )
    out["relative_lut_exec"] = lift
    if not lift.get("ok", False):
        out["ok"] = False
        out["error"] = lift.get("error", "相对 LUT 升降失败")
        return out

    pose1 = read_shoulder_head_pose(world)
    chest1 = world.chest_pose()
    shoulder_z1 = float(pose1["shoulder_mid"][2])
    dz1 = shoulder_z1 - target_z
    out.update({
        "dz_end_m": round(float(dz1), 4),
        "shoulder_z_end_m": round(float(shoulder_z1), 4),
        "chest_z_end_m": round(float(chest1["z"]), 4),
        "theta_z_end_deg": round(float(chest1["theta_z_deg"]), 2),
    })
    ctx.log(
        f"[{log_tag}] ①升降完成: dz {dz0:.3f}→{dz1:.3f}m "
        f"chest_z {chest0['z']:.3f}→{chest1['z']:.3f}m "
        f"θz={chest1['theta_z_deg']:.1f}°"
    )
    if dz1 > POINT_PRE_DESCENT_DZ_MAX_M + 0.08:
        out["note"] = "limited_by_lut_range"
        ctx.log(
            f"[{log_tag}] ①升降提示: LUT 已执行到可达附近，但 dz={dz1:.3f}m "
            f"仍大于 {POINT_PRE_DESCENT_DZ_MAX_M:.2f}m；后续先尝试 q3 俯仰完整几何"
        )
    return out


def adaptive_chord_reach_m(
    dz_m: float,
    reach_m: Optional[float] = None,
    *,
    margin_m: float = 0.04,
) -> float:
    """目标 dz 超过默认弦球时，自动放大 R_chord 使几何可解（点地面）。"""
    from behavior_interface.skills.base_chord_reach import (
        SHOULDER_HALF_WIDTH_M,
        chord_mid_horizontal_dist,
        reach_sphere_radius,
    )

    half_w = SHOULDER_HALF_WIDTH_M
    R_base = float(reach_m) if reach_m is not None else reach_sphere_radius()
    if chord_mid_horizontal_dist(R_base, half_w, float(dz_m)) is not None:
        return R_base
    R_need = math.sqrt(max(float(dz_m) ** 2 + half_w ** 2, 0.0)) + margin_m
    return max(R_base, R_need)


def _min_shoulder_distance_m(dual: Dict[str, Any]) -> Optional[float]:
    vals = [
        dual.get("left_shoulder_to_object_m"),
        dual.get("right_shoulder_to_object_m"),
    ]
    nums = [float(v) for v in vals if v is not None]
    if not nums:
        return None
    return float(min(nums))


def _yield_reach_compensation(
    ctx,
    world,
    *,
    C: np.ndarray,
    log_tag: str,
    keep_ori_arm: str = "none",
    max_dist_m: float = _REACH_OK_MAX_DIST_M,
) -> Generator:
    """肩距 >max_dist_m 时的补偿：先底盘前进，再压 q3 加深俯身。

    只在主 move（弦球规划 + 升降 + 底盘 + 俯仰）全部跑完、实测肩距仍够不到时才
    介入，且一旦 ≤max_dist_m 立即收手，不去追求把肩距压到规划的 R。
    """
    from behavior_interface.skills.move_to import yield_trunk_to_planned_pose
    from behavior_interface.trunk_vertical_lift import fk_torso_link4_theta_z_deg

    report: Dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "trigger_max_dist_m": float(max_dist_m),
        "min_distance_before_m": None,
        "min_distance_after_m": None,
        "phase1_forward": {
            "steps": [],
            "stopped_reason": None,
            "best_min_shoulder_m": None,
            "rolled_back_to_best": False,
        },
        "phase2_pitch": {"steps": [], "stopped_reason": None, "total_pitch_deg": 0.0},
    }
    dual = _reach_report_live(world, C)
    d0 = _min_shoulder_distance_m(dual)
    report["min_distance_before_m"] = None if d0 is None else round(float(d0), 4)
    if d0 is None or float(d0) <= float(max_dist_m):
        report["ok"] = True
        return report

    report["attempted"] = True
    d_cur = float(d0)
    ctx.log(
        f"[{log_tag}] [reach_comp] 肩距 min={d_cur:.3f}m > {max_dist_m:.2f}m，开始补偿"
    )

    # ── Phase1: 底盘每步前进 _REACH_COMP_FORWARD_STEP_M ──
    # 前进不一定单调拉近肩距（走过最近点、或俯身姿态下肩相对目标反而后移），所以
    # 记住肩距最小的那个落点；一旦某步没能让肩距继续下降就退回该落点再进 Phase2，
    # 避免把「白走的那一步」留在最终姿态里。
    pose0 = world.robot_pose()
    best_d = float(d_cur)
    best_xy = (float(pose0.pos[0]), float(pose0.pos[1]))
    best_yaw_deg = math.degrees(float(pose0.yaw))
    report["phase1_forward"]["best_min_shoulder_m"] = round(best_d, 4)
    for step_i in range(1, _REACH_COMP_FORWARD_MAX_STEPS + 1):
        pose = world.robot_pose()
        yaw = float(pose.yaw)
        travel = float(_REACH_COMP_FORWARD_STEP_M)
        bx = float(pose.pos[0]) + travel * math.cos(yaw)
        by = float(pose.pos[1]) + travel * math.sin(yaw)
        move_report = yield from _yield_base_xy_yaw_controller(
            ctx,
            world,
            bx=bx,
            by=by,
            theta_x_deg=math.degrees(yaw),
            timeout_s=_REACH_COMP_FORWARD_TIMEOUT_S,
            log_tag=f"{log_tag}/reach_comp",
            pos_tol=0.02,
            yaw_tol_deg=_NAV_YAW_TOL_DEG,
        )
        dual = _reach_report_live(world, C)
        d_new = _min_shoulder_distance_m(dual)
        stuck = bool(
            move_report.get("safety_abort")
            or not (move_report.get("xy_ok") and move_report.get("yaw_ok"))
        )
        report["phase1_forward"]["steps"].append({
            "step": step_i,
            "requested_travel_m": round(travel, 4),
            "target_xy": [round(bx, 4), round(by, 4)],
            "xy_ok": bool(move_report.get("xy_ok")),
            "yaw_ok": bool(move_report.get("yaw_ok")),
            "safety_abort": bool(move_report.get("safety_abort")),
            "collision_guard_reason": move_report.get("collision_guard_reason"),
            "min_shoulder_m": None if d_new is None else round(float(d_new), 4),
        })
        ctx.log(
            f"[{log_tag}] [reach_comp] Phase1 #{step_i}: forward={travel:.2f}m "
            f"min_d {d_cur:.3f}→{report['phase1_forward']['steps'][-1]['min_shoulder_m']} "
            f"xy_ok={move_report.get('xy_ok')} abort={move_report.get('safety_abort')}"
        )
        if d_new is not None and float(d_new) <= float(max_dist_m):
            report["phase1_forward"]["stopped_reason"] = "distance_ok"
            report["ok"] = True
            report["min_distance_after_m"] = round(float(d_new), 4)
            d_cur = float(d_new)
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase1 达标: min_d={d_new:.3f}m "
                f"≤{max_dist_m:.2f}m"
            )
            return report
        # 这一步是否让肩距实质下降（1mm 以内算没下降，避开测量噪声）
        improved = d_new is not None and float(d_new) < best_d - 1e-3
        if improved:
            best_d = float(d_new)
            best_xy = (float(world.robot_pose().pos[0]), float(world.robot_pose().pos[1]))
            best_yaw_deg = math.degrees(float(world.robot_pose().yaw))
            report["phase1_forward"]["best_min_shoulder_m"] = round(best_d, 4)
        if d_new is not None:
            d_cur = float(d_new)
        if not improved and not stuck:
            # 不降反增：退回肩距最小的落点，然后交给 Phase2 俯身
            report["phase1_forward"]["stopped_reason"] = "distance_not_decreasing"
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase1 #{step_i} 肩距未下降 "
                f"({d_new}m vs 最佳 {best_d:.3f}m)，回退到最佳落点后转 Phase2"
            )
            back = yield from _yield_base_xy_yaw_controller(
                ctx,
                world,
                bx=best_xy[0],
                by=best_xy[1],
                theta_x_deg=best_yaw_deg,
                timeout_s=_REACH_COMP_FORWARD_TIMEOUT_S,
                log_tag=f"{log_tag}/reach_comp_back",
                pos_tol=0.02,
                yaw_tol_deg=_NAV_YAW_TOL_DEG,
            )
            dual = _reach_report_live(world, C)
            d_back = _min_shoulder_distance_m(dual)
            report["phase1_forward"]["rolled_back_to_best"] = True
            report["phase1_forward"]["rollback"] = {
                "target_xy": [round(best_xy[0], 4), round(best_xy[1], 4)],
                "xy_ok": bool(back.get("xy_ok")),
                "min_shoulder_after_m": (
                    None if d_back is None else round(float(d_back), 4)
                ),
            }
            if d_back is not None:
                d_cur = float(d_back)
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase1 回退完成: min_d={d_cur:.3f}m "
                f"xy_ok={back.get('xy_ok')}"
            )
            if d_back is not None and float(d_back) <= float(max_dist_m):
                report["ok"] = True
                report["min_distance_after_m"] = round(float(d_back), 4)
                return report
            break
        if stuck:
            report["phase1_forward"]["stopped_reason"] = str(
                move_report.get("collision_guard_reason") or "stuck_or_not_reached"
            )
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase1 卡住，转入 Phase2 俯身: "
                f"min_d={d_cur:.3f}m"
            )
            break
    else:
        report["phase1_forward"]["stopped_reason"] = "max_steps"

    # ── Phase2: 每步压 q3 加深俯身 5°，最多 30° ──
    ctx.log(
        f"[{log_tag}] [reach_comp] Phase2 俯身: step={_REACH_COMP_PITCH_STEP_DEG:.0f}° "
        f"max={_REACH_COMP_PITCH_MAX_DEG:.0f}° min_d={d_cur:.3f}m"
    )
    total_pitch = 0.0
    max_steps = int(round(_REACH_COMP_PITCH_MAX_DEG / _REACH_COMP_PITCH_STEP_DEG))
    for step_i in range(1, max_steps + 1):
        chest = world.chest_pose()
        theta_z_now = float(chest["theta_z_deg"])
        q_now = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        q3_now = float(q_now[2])
        # 俯角上限换算成 q3 下界，与 URDF 硬限位取更严的一侧
        q3_floor_pitch = (
            float(q_now[0]) + float(q_now[1])
            - math.radians(float(_REACH_COMP_MAX_CHEST_PITCH_DEG))
        )
        q3_floor = max(float(_REACH_COMP_Q3_MIN_RAD), q3_floor_pitch)
        q3_tgt = max(q3_now - math.radians(float(_REACH_COMP_PITCH_STEP_DEG)), q3_floor)
        if q3_tgt >= q3_now - 1e-6:
            report["phase2_pitch"]["stopped_reason"] = (
                "q3_at_chest_pitch_limit"
                if q3_floor_pitch > float(_REACH_COMP_Q3_MIN_RAD)
                else "q3_at_urdf_limit"
            )
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase2 q3 已到下限 {q3_now:.4f}rad "
                f"(URDF {_REACH_COMP_Q3_MIN_RAD:.4f} / 俯角{_REACH_COMP_MAX_CHEST_PITCH_DEG:.0f}° "
                f"→ {q3_floor_pitch:.4f})，停止俯身"
            )
            break
        q_tgt = q_now.copy()
        q_tgt[2] = q3_tgt
        theta_z_tgt = float(
            fk_torso_link4_theta_z_deg(
                float(q_tgt[0]), float(q_tgt[1]), float(q_tgt[2]), float(q_tgt[3]),
            )
        )
        trunk_report = yield from yield_trunk_to_planned_pose(
            ctx,
            world,
            chest_z_tgt=float(chest["z"]),
            theta_z_tgt=theta_z_tgt,
            z_tol=0.05,
            theta_z_tol_deg=_NAV_THETA_Z_TOL_DEG,
            trunk_max_step_rad=0.08,
            trunk_timeout_s=30.0,
            log_prefix=f"{log_tag}/reach_comp_pitch",
            object_z=float(C[2]),
            reach_m=None,
            upward_require_theta_settle=False,
            target_trunk_q=q_tgt.tolist(),
            keep_ori_arm=keep_ori_arm,
        )
        dual = _reach_report_live(world, C)
        d_new = _min_shoulder_distance_m(dual)
        total_pitch += float(_REACH_COMP_PITCH_STEP_DEG)
        decreased = d_new is not None and float(d_new) < float(d_cur) - 1e-4
        report["phase2_pitch"]["steps"].append({
            "step": step_i,
            "q3_before_rad": round(q3_now, 4),
            "q3_target_rad": round(q3_tgt, 4),
            "q3_floor_rad": round(q3_floor, 4),
            "q3_floor_by": (
                "chest_pitch_165deg"
                if q3_floor_pitch > float(_REACH_COMP_Q3_MIN_RAD)
                else "urdf_limit"
            ),
            "chest_pitch_target_deg": round(
                math.degrees(float(q_tgt[0]) + float(q_tgt[1]) - q3_tgt), 1
            ),
            "theta_z_before_deg": round(theta_z_now, 2),
            "theta_z_target_deg": round(theta_z_tgt, 2),
            "trunk_ok": bool(trunk_report.get("ok")),
            "min_shoulder_m": None if d_new is None else round(float(d_new), 4),
            "decreased": bool(decreased),
        })
        report["phase2_pitch"]["total_pitch_deg"] = round(total_pitch, 1)
        ctx.log(
            f"[{log_tag}] [reach_comp] Phase2 #{step_i}: q3 {q3_now:.4f}→{q3_tgt:.4f}rad "
            f"(俯角{report['phase2_pitch']['steps'][-1]['chest_pitch_target_deg']:.0f}° "
            f"θz {theta_z_now:.1f}→{theta_z_tgt:.1f}°) "
            f"min_d {d_cur:.3f}→{report['phase2_pitch']['steps'][-1]['min_shoulder_m']} "
            f"decreased={decreased}"
        )
        if d_new is not None and float(d_new) <= float(max_dist_m):
            report["phase2_pitch"]["stopped_reason"] = "distance_ok"
            report["ok"] = True
            report["min_distance_after_m"] = round(float(d_new), 4)
            ctx.log(
                f"[{log_tag}] [reach_comp] Phase2 达标: min_d={d_new:.3f}m "
                f"≤{max_dist_m:.2f}m total_pitch=+{total_pitch:.0f}°"
            )
            return report
        if not trunk_report.get("ok", True):
            report["phase2_pitch"]["stopped_reason"] = "trunk_failed"
            break
        if d_new is None:
            report["phase2_pitch"]["stopped_reason"] = "measure_failed"
            break
        if not decreased:
            report["phase2_pitch"]["stopped_reason"] = "no_distance_decrease"
            ctx.log(f"[{log_tag}] [reach_comp] Phase2 肩距未下降，停止俯身")
            break
        d_cur = float(d_new)
    else:
        if report["phase2_pitch"]["stopped_reason"] is None:
            report["phase2_pitch"]["stopped_reason"] = "max_pitch"

    dual_f = _reach_report_live(world, C)
    d_f = _min_shoulder_distance_m(dual_f)
    report["min_distance_after_m"] = None if d_f is None else round(float(d_f), 4)
    report["ok"] = bool(d_f is not None and float(d_f) <= float(max_dist_m))
    ctx.log(
        f"[{log_tag}] [reach_comp] 结束 ok={report['ok']} "
        f"min_d {report['min_distance_before_m']}→{report['min_distance_after_m']}m "
        f"fwd={len(report['phase1_forward']['steps'])}步 "
        f"pitch=+{report['phase2_pitch']['total_pitch_deg']}° "
        f"p1={report['phase1_forward']['stopped_reason']} "
        f"p2={report['phase2_pitch']['stopped_reason']}"
    )
    return report


def _run_move_to_center(
    ctx,
    *,
    session_id: str,
    C: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    reach: float = 0.0,
    tool: str,
    build_id: str,
    log_tag: str,
    head_png_tag: str,
    result_extra: Optional[Dict[str, Any]] = None,
    travel_hint: str = "若仍不可达请调整目标点或 reach",
    nav_timeout_s: float = _NAV_TIMEOUT_S_DEFAULT,
    keep_ori_arm: str = "none",
) -> Generator:
    """以 C 为弦球球心：升降后离线几何规划 + 真机执行。

    几何：肩中点 forward+h3 的 q3-only 俯仰模型 → 俯仰后肩高切面弦球。
    真机：①升降 → ②xy → ③spin/yaw → ④q3俯仰。
    """
    from behavior_interface.skills.move_to import (
        _base_link_z_world,
        move_to as move_to_skill,
        yield_move_settle,
        yield_trunk_to_planned_pose,
    )

    world = ctx.world
    _ensure_world_action_compat(world)
    try:
        from behavior_interface.skills.eef import _freeze_world_limb_pins

        limb_pins = _freeze_world_limb_pins(world)
        ctx.log(
            f"[{log_tag}] limb pins frozen for move: "
            f"gripper_qpos={limb_pins.get('grippers', {})}"
        )
    except Exception as exc:
        ctx.log(f"[{log_tag}] WARN freeze limb pins failed: {exc}")
    C = np.asarray(C, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    extra = dict(result_extra or {})

    if extra.get("object_name"):
        ctx.log(f"[{log_tag}] 目标物体: {extra['object_name']}")

    def _fail(msg: str) -> Generator:
        ctx.log(f"[{log_tag}] 中止: {msg}")
        yield from _hold(world, 4)
        try:
            dual_fail = _reach_report_live(world, C)
            shoulder_distance_fail = _shoulder_distance_result(
                dual_fail,
                C,
                reach_R_m=reach_m if reach_m is not None else REACH_SPHERE_R_M,
            )
        except Exception as e:
            shoulder_distance_fail = {
                "shoulder_distance": {
                    "target_center_world": [float(x) for x in C],
                    "error": f"shoulder distance measurement failed: {e}",
                }
            }
        head_live = _capture_head_snapshot(
            world,
            session_id=session_id,
            head_png_tag=head_png_tag,
            log_tag=log_tag,
            ctx=ctx,
            lo=lo,
            hi=hi,
            suffix="failure",
        )
        result = {
            "ok": False,
            "error": msg,
            "tool": tool,
            "target_center_world": [float(x) for x in C],
            "visibility": head_live,
            "head_after_move_png": head_live.get("head_png"),
            "head_view_ok": bool(
                head_live.get("ok")
                and head_live.get("all_corners_in")
                and not head_live.get("overflow")
            ),
            **shoulder_distance_fail,
            **extra,
        }
        if tool == "move_to_object":
            result["object_center_world"] = result["target_center_world"]
        elif tool == "move_to_point":
            result["point_center_world"] = result["target_center_world"]
        ctx.set_result(result)
        _save_nav_reach(session_id, result)
        yield from _hold(world)
        return

    ctx.log(
        f"[{log_tag}] BUILD={build_id} "
        "| 无 head 补偿 | 无「步骤1躯干/步骤2底盘」旧流程"
    )
    if str(keep_ori_arm or "none").strip().lower() not in ("", "none", "false", "0", "no"):
        ctx.log(
            f"[{log_tag}] keep_ori_arm={keep_ori_arm}：仅④最终俯仰生效；"
            "原 J1-J4 轨迹不变，每帧仅 J5-J7 追踪入口世界系 EEF 姿态，"
            "EEF 平移只监控；①升降与②xy③spin 不补偿"
        )

    reach_m = float(reach) if float(reach) > 0.01 else None

    pose = world.robot_pose()
    robot_xy = np.array([float(pose.pos[0]), float(pose.pos[1])], dtype=np.float64)

    from behavior_interface.skills.move_to_object_geom import (
        plan_geom_from_pose,
        read_shoulder_head_pose,
        solve_q1_forward_height_with_current_q3,
        solve_theta_z_shoulder_forward_low_z_repair,
    )

    reach_plan = reach_m
    if reach_plan is not None:
        ctx.log(f"[{log_tag}] 自定义 reach={reach_plan:.3f}m（默认={REACH_SPHERE_R_M:.2f}m）")
    ctx.log(
        f"[{log_tag}] ── 真机①升降/预降：先把肩高-目标z压到0.4m内 ──"
    )
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    world._codex_fast_motion_no_obs = True
    try:
        lift_report = yield from _yield_vertical_lift_before_nav(
            ctx, world, C, float(world.chest_pose()["z"]), 0.0, log_tag,
        )
        if not lift_report.get("ok", True):
            ctx.log(f"[{log_tag}] 升降失败: {lift_report.get('error')}")

        pose_after_lift = read_shoulder_head_pose(world)
        sh_z_after_lift = float(pose_after_lift["shoulder_mid"][2])
        dz_after_lift = sh_z_after_lift - float(C[2])
        ctx.log(
            f"[{log_tag}] ①升降后: 肩z={sh_z_after_lift:.3f} "
            f"目标z={float(C[2]):.3f} dz={dz_after_lift:.3f}m"
        )

        ctx.log(f"[{log_tag}] ── 离线规划：肩中点 forward+h3 俯仰 → 弦球 xy/spin ──")
        _ensure_plan_scene_graph(world, ctx, log_tag)
        geom = plan_geom_from_pose(
            pose_after_lift,
            C,
            robot_xy,
            is_free_xy=lambda x, y: _is_free_xy(world, x, y),
            nav_clearance_fn=lambda x, y: _nav_clearance_xy(world, x, y),
            reach=reach_plan,
            base_link_z=_base_link_z_world(world),
        )
        if not geom.get("ok") and reach_m is None:
            sh_z_final = geom.get("shoulder_z")
            if sh_z_final is None:
                sh_z_final = (geom.get("base_geom") or {}).get("shoulder_z_slice_m")
            if sh_z_final is None:
                sh_z_final = sh_z_after_lift
            dz_final = float(sh_z_final) - float(C[2])
            reach_adapt = adaptive_chord_reach_m(dz_final, None)
            pitch_meta = (
                geom.get("pitch_geom")
                or (geom.get("trunk_meta") or {}).get("pitch_geom")
                or {}
            )
            ctx.log(
                f"[{log_tag}] 默认 R={REACH_SPHERE_R_M:.2f} 完整几何失败: "
                f"{geom.get('error', 'unknown')}；最终pose_after_pitch肩高 dz={dz_final:.3f}m "
                f"pitch_err={pitch_meta.get('err_z_m')}m → retry adaptive R={reach_adapt:.3f}m"
            )
            if reach_adapt > REACH_SPHERE_R_M + 1e-3:
                reach_plan = reach_adapt
                geom = plan_geom_from_pose(
                    pose_after_lift,
                    C,
                    robot_xy,
                    is_free_xy=lambda x, y: _is_free_xy(world, x, y),
                    nav_clearance_fn=lambda x, y: _nav_clearance_xy(world, x, y),
                    reach=reach_plan,
                    base_link_z=_base_link_z_world(world),
                )
        if not geom.get("ok"):
            yield from _fail(geom.get("error", "几何规划失败"))
            return

        chest_z = float(geom["chest_z"])
        theta_z = float(geom["theta_z_deg"])
        sh_z_plan = float(geom["shoulder_z"])
        trunk_meta = geom["trunk_meta"]
        pitch = trunk_meta.get("pitch_geom") or {}
        plan = geom["base_plan"]
        bx, by = plan["base_xy"]
        theta_x = float(geom["theta_x_deg"])
        detail = plan.get("plan_detail") or {}
        travel_m = float(detail.get("travel_m", 0))

        R_pitch = float(geom.get("reach_pitch_m", pitch.get("reach_m", reach_sphere_radius())))
        R_chord = float(geom.get("reach_chord_m", plan.get("reach_sphere_R_m", R_pitch)))
        if abs(R_pitch - REACH_SPHERE_R_M) > 1e-4 and reach_m is None:
            ctx.log(
                f"[{log_tag}] 警告: reach_pitch={R_pitch:.2f} "
                f"≠ REACH_SPHERE_R_M={REACH_SPHERE_R_M}，请检查 REACH_SPHERE_R_M"
            )
        ctx.log(
            f"[{log_tag}] 离线② 俯仰 R_pitch={R_pitch:.2f}m "
            f"h3={pitch.get('h3_m', pitch.get('h_m'))}m（h3=√(R²−(肩宽/2)²)）"
        )
        _relax = geom.get("reach_chord_relax") or trunk_meta.get("reach_chord_relax") or {}
        if _relax.get("applied"):
            ctx.log(
                f"[{log_tag}] 离线②b 弦球放松: R_chord {R_pitch:.2f}→{R_chord:.2f}m "
                f"(+{_relax.get('step_m')}m×{_relax.get('steps')}步, dz={_relax.get('dz_m')}m) "
                f"俯身θz不变={theta_z:.1f}°"
            )
        else:
            ctx.log(
                f"[{log_tag}] 离线②b 移动 R_chord={R_chord:.2f}m（弦球取点，R_chord=R_pitch）"
            )
        ctx.log(
            f"[{log_tag}] 离线③ 肩中点forward+h3 点z={pitch.get('intersection_z')} "
            f"目标z={pitch.get('object_z')} err={pitch.get('err_z_m')}m "
            f"→ theta_z {pitch.get('theta_z_now_deg')}°→{theta_z:.1f}° "
            f"[q3-only {pitch.get('pitch_q1_feasibility', {}).get('theta_z_achievable_lo_deg')}"
            f"°~{pitch.get('pitch_q1_feasibility', {}).get('theta_z_achievable_hi_deg')}°"
            f" sat={pitch.get('pitch_q1_feasibility', {}).get('theta_z_saturated')}]"
        )
        repair = pitch.get("low_z_q1_repair") or {}
        if repair.get("applied"):
            ctx.log(
                f"[{log_tag}] 离线③b 低位大俯角修补: "
                f"q3-only θz={repair.get('theta_z_q3_only_deg')}° "
                f"(lean={repair.get('lean_q3_only_deg')}°) → "
                f"q3 cap θz={repair.get('theta_z_q3_cap_deg')}°；"
                f"只调 q1 补高度 err_z={repair.get('err_z_m')}m "
                f"dq1={repair.get('q1_delta_rad')}rad dq3={repair.get('q3_delta_rad')}rad "
                f"target_q={pitch.get('target_trunk_q')}"
            )
        else:
            ctx.log(
                f"[{log_tag}] 离线③b 低位大俯角修补未启用: "
                f"{repair.get('reason')} chest_z={repair.get('chest_z_now')} "
                f"lean={repair.get('lean_q3_only_deg')}°"
            )
        nav_pick = detail.get("nav_clearance") or {}
        from behavior_interface.skills.base_footprint_distance import measure_base_distance

        def _measure_base_target_at(x: float, y: float) -> Tuple[Dict[str, Any], float]:
            m = measure_base_distance(
                base_pose={"x": float(x), "y": float(y), "yaw_deg": float(theta_x)},
                target_xy=C[:2],
                target_aabb_min=lo,
                target_aabb_max=hi,
            )
            return m, _base_target_clearance_value(m)

        base_target_clearance, base_target_clearance_m = _measure_base_target_at(bx, by)
        base_backoff = {
            "applied": False,
            "min_clearance_m": _BASE_TARGET_MIN_CLEARANCE_M,
            "target_clearance_m": (
                _BASE_TARGET_MIN_CLEARANCE_M + _BASE_TARGET_BACKOFF_MARGIN_M
                + _BASE_TARGET_BACKOFF_EXEC_GUARD_M
            ),
            "exec_guard_m": _BASE_TARGET_BACKOFF_EXEC_GUARD_M,
            "initial_clearance_m": base_target_clearance_m,
        }
        if base_target_clearance_m <= _BASE_TARGET_MIN_CLEARANCE_M:
            bx0, by0 = float(bx), float(by)
            yaw_rad = math.radians(theta_x)
            back_dir = np.array([-math.cos(yaw_rad), -math.sin(yaw_rad)], dtype=np.float64)
            target_clearance_m = (
                _BASE_TARGET_MIN_CLEARANCE_M + _BASE_TARGET_BACKOFF_MARGIN_M
                + _BASE_TARGET_BACKOFF_EXEC_GUARD_M
            )
            best = None
            prev_s = 0.0
            s = _BASE_TARGET_BACKOFF_STEP_M
            while s <= _BASE_TARGET_BACKOFF_MAX_M + 1e-9:
                cand = np.array([bx0, by0], dtype=np.float64) + back_dir * s
                cand_measure, cand_clear = _measure_base_target_at(float(cand[0]), float(cand[1]))
                if cand_clear > target_clearance_m:
                    lo_s, hi_s = prev_s, s
                    for _ in range(18):
                        mid_s = 0.5 * (lo_s + hi_s)
                        mid = np.array([bx0, by0], dtype=np.float64) + back_dir * mid_s
                        _, mid_clear = _measure_base_target_at(float(mid[0]), float(mid[1]))
                        if mid_clear > target_clearance_m:
                            hi_s = mid_s
                        else:
                            lo_s = mid_s
                    final_s = hi_s
                    final_xy = np.array([bx0, by0], dtype=np.float64) + back_dir * final_s
                    final_measure, final_clear = _measure_base_target_at(
                        float(final_xy[0]), float(final_xy[1])
                    )
                    best = (final_s, final_xy, final_measure, final_clear)
                    break
                prev_s = s
                s += _BASE_TARGET_BACKOFF_STEP_M

            if best is None:
                yield from _fail(
                    f"底盘 footprint 到目标物体水平 AABB 距离 "
                    f"{base_target_clearance_m*100:.1f}cm ≤ "
                    f"{_BASE_TARGET_MIN_CLEARANCE_M*100:.0f}cm；"
                    f"沿 yaw={theta_x:.1f}° 的 backward 后退 "
                    f"{_BASE_TARGET_BACKOFF_MAX_M:.2f}m 仍无法退出安全边界"
                )
                return

            shift_m, final_xy, base_target_clearance, base_target_clearance_m = best
            dx, dy = float(final_xy[0] - bx0), float(final_xy[1] - by0)
            bx, by = float(final_xy[0]), float(final_xy[1])
            plan["base_xy"] = [bx, by]
            detail["bx"] = bx
            detail["by"] = by
            _translate_plan_detail_xy(detail, dx, dy)
            detail["base_dist_to_object_m"] = round(float(np.linalg.norm(
                np.array([bx, by], dtype=np.float64) - C[:2]
            )), 4)
            travel_m = float(np.linalg.norm(np.array([bx, by], dtype=np.float64) - robot_xy))
            detail["travel_m"] = travel_m
            nav_free, nav_min_clear, nav_blocker = _nav_clearance_xy(world, bx, by)
            nav_pick = dict(nav_pick)
            nav_pick.update({
                "after_base_backoff": True,
                "all_free": bool(nav_free),
                "min_clearance_m": round(float(nav_min_clear), 4),
                "blocker": nav_blocker,
                "blockers": nav_blocker,
            })
            detail["nav_clearance"] = nav_pick
            base_backoff.update({
                "applied": True,
                "shift_m": float(shift_m),
                "direction_xy": back_dir.round(6).tolist(),
                "base_xy_before": [bx0, by0],
                "base_xy_after": [bx, by],
                "dx_m": dx,
                "dy_m": dy,
                "final_clearance_m": base_target_clearance_m,
                "final_distance_point_m": float(
                    base_target_clearance.get("distance_point_m", float("nan"))
                ),
                "final_distance_aabb_m": float(
                    base_target_clearance.get("distance_aabb_m", float("nan"))
                ),
            })
            ctx.log(
                f"[{log_tag}] 离线④b 底盘安全 backoff: "
                f"base_obj_clear {base_backoff['initial_clearance_m']*100:.1f}cm"
                f"→{base_target_clearance_m*100:.1f}cm "
                f"shift={shift_m:.3f}m dir=({back_dir[0]:+.2f},{back_dir[1]:+.2f})；"
                f"xy ({bx0:.2f},{by0:.2f})→({bx:.2f},{by:.2f})，yaw/spin 保持 {theta_x:.1f}°"
            )
        ctx.log(
            f"[{log_tag}] 离线④ 俯仰后肩高切面弦球 reach_R={plan['reach_sphere_R_m']}m "
            f"→ xy=({bx:.2f},{by:.2f}) yaw={theta_x:.1f}° "
            f"chest_z(model) {trunk_meta['chest_z_now']:.2f}→{chest_z:.2f} "
            f"q3_dz={geom.get('delta_chest_z_m', 0):.3f}m 肩切面z≈{sh_z_plan:.2f} "
            f"chord_ok={detail.get('chord_ok')} travel={travel_m:.2f}m "
            f"yaw_off={detail.get('yaw_offset_deg')}° "
            f"policy={detail.get('pick_policy', 'legacy')} "
            f"base_dist_O={detail.get('base_dist_to_object_m')}m "
            f"base_obj_clear={base_target_clearance_m:.3f}m "
            f"min_clr={nav_pick.get('min_clearance_m')}m "
            f"all_free={nav_pick.get('all_free')} "
            f"blockers={nav_pick.get('blockers')}"
        )
        if travel_m < 0.08:
            ctx.log(
                f"[{log_tag}] 提示: 规划底盘位移仅 {travel_m:.2f}m，"
                f"可能看不出移动；{travel_hint}"
            )

        trunk_meta["build"] = build_id
        trunk_meta["exec_order"] = "lift_xy_spin_pitch"
        trunk_meta["trunk_mode"] = (
            "q3_cap_q1_repair_after_nav"
            if trunk_meta.get("target_trunk_q") is not None
            else "q3_pitch_after_nav"
        )
        ctx.log(
            f"[{log_tag}] ── 真机执行 ── "
            "①升降已完成 → ②xy → ③spin → ④俯仰；"
            "导航锁躯干+双臂，躯干段锁底盘+双臂"
        )
        ctx.log(
            f"[{log_tag}] trunk: robot_coord 逻辑 "
            f"chest_z→{chest_z:.2f} theta_z→{theta_z:.1f}°"
        )
        ctx.log(
            f"[{log_tag}] 目标: xy=({bx:.2f},{by:.2f}) "
            f"yaw={theta_x:.1f}° chest_z={chest_z:.2f} theta_z={theta_z:.1f}°"
        )

        ctx.log(
            f"[{log_tag}] 执行模式: normal kinematic/controller；"
            "②xy③yaw 走 move_to，④trunk 走 yield_trunk_to_planned_pose"
        )

        nav_timeout_s = max(30.0, float(nav_timeout_s))
        deadline = time.time() + nav_timeout_s
        ctx.log(f"[{log_tag}] nav_timeout_s={nav_timeout_s:.0f}")

        last_path_status = None
        last_waypoints_n = 0
        last_trunk_report: Optional[Dict[str, Any]] = None
        last_base_report: Optional[Dict[str, Any]] = None
        collision_pitch_recovery: Optional[Dict[str, Any]] = None
        position_reached = False
        chest_z_exec_target = chest_z
        theta_z_exec_target = theta_z
        pose_errors = _pose_errors(
            world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
        )
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            remaining_s = max(1.0, deadline - time.time())
            ctx.log(
                f"[{log_tag}] 闭环尝试 #{attempt} "
                f"剩余 {remaining_s:.0f}s（normal controller path）"
            )
            move_report = yield from _yield_base_xy_yaw_controller(
                ctx,
                world,
                bx=float(bx),
                by=float(by),
                theta_x_deg=float(theta_x),
                timeout_s=remaining_s,
                log_tag=log_tag,
                pos_tol=_NAV_POS_TOL_M,
                yaw_tol_deg=_NAV_YAW_TOL_DEG,
            )
            last_base_report = move_report
            last_path_status = move_report.get("path_status")
            last_waypoints_n = int(move_report.get("waypoints_n") or 0)
            if move_report.get("safety_abort"):
                can_try_pitch_recovery = bool(
                    tool == "move_to_point"
                    and str(move_report.get("collision_guard_reason") or "") == "no_progress"
                    and str(extra.get("image_id") or "").strip()
                    and isinstance(extra.get("uv"), (list, tuple))
                    and len(extra.get("uv")) >= 2
                )
                if can_try_pitch_recovery:
                    collision_pitch_recovery = yield from (
                        _yield_reach_point_collision_pitch_recovery(
                            ctx,
                            world,
                            session_id=session_id,
                            image_id=str(extra["image_id"]),
                            source_uv_px=extra["uv"],
                            move_report=move_report,
                            log_tag=log_tag,
                            keep_ori_arm=keep_ori_arm,
                        )
                    )
                    if collision_pitch_recovery.get("ok"):
                        last_trunk_report = collision_pitch_recovery.get("trunk_exec")
                        chest_z_exec_target = float(
                            collision_pitch_recovery["trunk_targets"]["chest_z_robot_m"]
                        )
                        theta_z_exec_target = float(
                            collision_pitch_recovery["trunk_targets"]["theta_z_deg"]
                        )
                        pose_errors = {
                            "xy_m": round(float(move_report.get("xy_err_m") or 0.0), 4),
                            "yaw_deg": round(float(move_report.get("yaw_err_deg") or 0.0), 2),
                            "z_m": round(
                                float((last_trunk_report or {}).get("z_err_m") or 0.0),
                                4,
                            ),
                            "theta_z_deg": round(
                                float(
                                    (last_trunk_report or {}).get(
                                        "theta_z_err_deg",
                                        0.0,
                                    )
                                    or 0.0
                                ),
                                2,
                            ),
                        }
                        position_reached = True
                        ctx.log(
                            f"[{log_tag}] ②xy③yaw 被障碍挡住，但目标已通过"
                            f"当前 RGB-D 俯身恢复进入工作空间: err={pose_errors}"
                        )
                        break
                pose_errors = _pose_errors(
                    world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
                )
                position_reached = False
                ctx.log(
                    f"[{log_tag}] ②xy③yaw 安全守护中止，停止后续 trunk: "
                    f"{move_report} err={pose_errors}"
                )
                break
            if not (move_report.get("xy_ok", True) and move_report.get("yaw_ok", True)):
                pose_errors = _pose_errors(
                    world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
                )
                if _pose_within_tol(pose_errors):
                    ctx.log(
                        f"[{log_tag}] ②xy③yaw controller 返回未到位，"
                        f"但最终容差已满足，继续 trunk: {move_report} err={pose_errors}"
                    )
                else:
                    position_reached = False
                    ctx.log(f"[{log_tag}] ②xy③yaw 未到位: {move_report} err={pose_errors}")
                    break
            target_trunk_q = trunk_meta.get("target_trunk_q")
            target_trunk_q_exec = target_trunk_q
            last_trunk_report = yield from yield_trunk_to_planned_pose(
                ctx,
                world,
                chest_z_tgt=chest_z,
                theta_z_tgt=theta_z,
                z_tol=_NAV_Z_TOL_M,
                theta_z_tol_deg=_NAV_THETA_Z_TOL_DEG,
                trunk_max_step_rad=0.10,
                trunk_timeout_s=45.0,
                log_prefix=f"{log_tag}/trunk",
                object_z=float(C[2]),
                reach_m=R_pitch,
                upward_require_theta_settle=False,
                target_trunk_q=target_trunk_q,
                keep_ori_arm=keep_ori_arm,
            )
            theta_z_exec_target = float((last_trunk_report or {}).get("theta_z_target_deg", theta_z))
            chest_z_exec_target = chest_z
            target_trunk_q_exec = target_trunk_q
            if not last_trunk_report["ok"]:
                last_trunk_report["error"] = last_trunk_report.get("error") or "θz 俯仰未到位或超时"
            if not last_trunk_report.get("ok", True):
                ctx.log(
                    f"[{log_tag}] 躯干未到位: {last_trunk_report.get('error')}"
                )
                pose_errors = _pose_errors(
                    world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
                )
                position_reached = False
                break
            pose_errors = _pose_errors(
                world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
            )
            position_reached = _pose_within_tol(pose_errors)
            if position_reached:
                ctx.log(f"[{log_tag}] 闭环到位: {pose_errors}")
                break
            ctx.log(
                f"[{log_tag}] 未到位 err={pose_errors} "
                f"stage_ok={move_report} path={last_path_status}，继续驱动…"
            )
            yield from _hold(world, 6)
    finally:
        world._codex_fast_motion_no_obs = old_no_obs
    if not position_reached:
        ctx.log(
            f"[{log_tag}] {nav_timeout_s:.0f}s 内未到达规划位姿 "
            f"err={pose_errors}"
        )

    head_live = _head_view_live(world, lo, hi)
    head_ok = bool(
        head_live.get("ok") and head_live.get("all_corners_in")
        and not head_live.get("overflow")
    )
    if head_live.get("ok"):
        ctx.log(
            f"[{log_tag}] head 检查: all_in={head_live.get('all_corners_in')} "
            f"overflow={head_live.get('overflow')} fill={head_live.get('fill_visible')}"
        )

    dual = _reach_report_live(world, C)
    yaw_rad = math.radians(theta_x)
    sh_z = float(geom.get("shoulder_z", sh_z_plan))
    sm = (detail.get("shoulder_mid") or [bx, by, sh_z])
    left, right, _ = shoulder_positions_at_base(float(sm[0]), float(sm[1]), yaw_rad, sh_z)
    R_chord_verify = float(plan.get("reach_sphere_R_m", geom.get("reach_chord_m", R_pitch)))
    chord_ok, chord_dist = verify_chord_on_sphere(C, left, right, R_chord_verify)
    shoulder_distance = _shoulder_distance_result(dual, C, reach_R_m=R_chord_verify)
    reach_ok = bool(
        (_min_shoulder_distance_m(dual) or float("inf")) <= _REACH_OK_MAX_DIST_M
    )

    # 主 move 全部跑完后的补偿补丁：肩距仍 >0.7m 才介入，达标即收手
    reach_compensation: Optional[Dict[str, Any]] = None
    if tool == "move_to_point" and not reach_ok:
        old_no_obs_comp = bool(getattr(world, "_codex_fast_motion_no_obs", False))
        try:
            world._codex_fast_motion_no_obs = True
            reach_compensation = yield from _yield_reach_compensation(
                ctx,
                world,
                C=C,
                log_tag=log_tag,
                keep_ori_arm=keep_ori_arm,
                max_dist_m=_REACH_OK_MAX_DIST_M,
            )
        finally:
            world._codex_fast_motion_no_obs = old_no_obs_comp
        dual = _reach_report_live(world, C)
        shoulder_distance = _shoulder_distance_result(dual, C, reach_R_m=R_chord_verify)
        reach_ok = bool(
            (_min_shoulder_distance_m(dual) or float("inf")) <= _REACH_OK_MAX_DIST_M
        )
        # 补偿动过底盘/俯仰，刷新 head 投影与姿态误差读数
        head_live = _head_view_live(world, lo, hi)
        head_ok = bool(
            head_live.get("ok") and head_live.get("all_corners_in")
            and not head_live.get("overflow")
        )
        pose_errors = _pose_errors(
            world, bx, by, chest_z_exec_target, theta_x, theta_z_exec_target
        )

    final = world.robot_pose()
    moved_m = float(np.linalg.norm(
        np.array([final.pos[0], final.pos[1]]) - robot_xy
    ))

    head_png = os.path.join("/tmp", f"{head_png_tag}_{session_id or 'default'}.png")
    try:
        from behavior_interface.head_capture import capture_head_png
        cap = capture_head_png(world, head_png, n_render=12)
        if cap:
            head_live["head_png"] = head_png
    except Exception:
        pass

    nav_error = None
    base_failure_reason = str(
        (last_base_report or {}).get("reason")
        or (last_base_report or {}).get("status")
        or ""
    )
    arrival_mode = (
        "collision_pitch_recovery"
        if position_reached
        and collision_pitch_recovery
        and collision_pitch_recovery.get("ok")
        else "planned_base_and_trunk"
        if position_reached
        else None
    )
    if arrival_mode == "collision_pitch_recovery":
        base_failure_reason = ""
    if not position_reached and base_failure_reason == "stuck_or_collision":
        nav_error = "stuck或者碰撞"
    elif not position_reached and base_failure_reason == "goal_overshoot":
        nav_error = "底盘越过规划终点，已急停"
    elif not position_reached and last_trunk_report and not last_trunk_report.get("ok", True):
        nav_error = (
            f"躯干未到位: {last_trunk_report.get('error', 'unknown')} "
            f"xy={pose_errors['xy_m']:.3f}m yaw={pose_errors['yaw_deg']:+.1f}° "
            f"z={pose_errors['z_m']:+.3f}m θz={pose_errors['theta_z_deg']:+.1f}°"
        )
    elif not position_reached:
        nav_error = (
            f"位置未到达（{nav_timeout_s:.0f}s 超时）: "
            f"xy={pose_errors['xy_m']:.3f}m yaw={pose_errors['yaw_deg']:+.1f}° "
            f"z={pose_errors['z_m']:+.3f}m θz={pose_errors['theta_z_deg']:+.1f}°"
        )

    result = {
        "ok": position_reached,
        "position_reached": position_reached,
        "arrival_mode": arrival_mode,
        "tool": tool,
        "target_center_world": [float(x) for x in C],
        "base_target": [bx, by],
        "base_final": [float(final.pos[0]), float(final.pos[1])],
        "theta_x_deg": round(theta_x, 1),
        "chest_z": round(chest_z, 3),
        "theta_z_deg": round(theta_z, 1),
        "failure_reason": base_failure_reason or None,
        "trunk_plan": trunk_meta,
        "lift_report": lift_report,
        "head_view_ok": head_ok,
        "visibility": head_live,
        "head_after_move_png": head_live.get("head_png"),
        "measured_fill": round(float(head_live.get("fill_visible", 0)), 4) if head_live.get("ok") else None,
        "planned_travel_m": round(float(detail.get("travel_m", 0)), 4),
        "actual_travel_m": round(moved_m, 4),
        "chord_plan": plan,
        "base_target_clearance": base_target_clearance,
        "base_target_min_clearance_m": _BASE_TARGET_MIN_CLEARANCE_M,
        "base_target_backoff": base_backoff,
        "chord_ok_after_move": chord_ok,
        "chord_dist_after_move": chord_dist,
        "reachable": dual["reachable"],
        "reachable_left": dual["reachable_left"],
        "reachable_right": dual["reachable_right"],
        "arms_reachable": dual["arms_reachable"],
        "arm": dual.get("arm"),
        "preferred_arm": dual.get("preferred_arm"),
        "arm_left": dual.get("left"),
        "arm_right": dual.get("right"),
        "reach_sphere_R_m": plan["reach_sphere_R_m"],
        "reach_pitch_m": R_pitch,
        "reach_chord_m": R_chord,
        "theta_z_geom_deg": pitch.get("theta_z_geom_deg"),
        "arm_reach_note": (
            f"俯仰/移动 R={R_pitch:.2f}m（无 pad）；"
            f"抓取判定肩→物 [{_ARM_MIN_REACH},{_ARM_MAX_REACH}]m"
        ),
        **shoulder_distance,
        "error": nav_error,
        "pose_errors": pose_errors,
        "path_exec": {
            "mode": "normal_kinematic_controller",
            "snap_goal": False,
            "extra_inflate_m": _NAV_EXTRA_INFLATE_M,
            "nav_guard": "move_to_controller",
            "plan_nav_clearance": nav_pick,
            "stuck_backup_m": _NAV_STUCK_BACKUP_M,
            "last_status": last_path_status,
            "last_waypoints_n": last_waypoints_n,
            "move_to_report": last_base_report,
            "last_stuck_recoveries": int(
                (getattr(ctx, "_move_to_report", None) or {}).get("stuck_recoveries") or 0
            ),
        },
        "nav_timeout_s": nav_timeout_s,
        "nav_attempts": attempt,
        "trunk_exec": last_trunk_report,
        "collision_pitch_recovery": collision_pitch_recovery,
        "reach_ok": reach_ok,
        "reach_ok_threshold_m": _REACH_OK_MAX_DIST_M,
        "reach_compensation": reach_compensation,
        "keep_ori_requested_arm": (last_trunk_report or {}).get(
            "keep_ori_requested_arm",
            [],
        ),
        "keep_ori_arm": (last_trunk_report or {}).get("keep_ori_arm", []),
        "keep_ori_scope": "final_trunk_pitch_after_base_yaw",
        "keep_ori_ok": (last_trunk_report or {}).get(
            "keep_ori_ok",
            str(keep_ori_arm or "none").strip().lower() in ("", "none", "false", "0", "no"),
        ),
        "keep_ori_tracking_ok": (last_trunk_report or {}).get(
            "keep_ori_tracking_ok",
            True,
        ),
        "eef_pos_smooth_ok": (last_trunk_report or {}).get(
            "eef_pos_smooth_ok",
            True,
        ),
        "eef_pos_before": (last_trunk_report or {}).get("eef_pos_before", {}),
        "eef_pos_after": (last_trunk_report or {}).get("eef_pos_after", {}),
        "eef_pos_path_max_step_m": (last_trunk_report or {}).get(
            "eef_pos_path_max_step_m",
            {},
        ),
        "eef_pos_path_max_accel_step_m": (last_trunk_report or {}).get(
            "eef_pos_path_max_accel_step_m",
            {},
        ),
        "eef_balance_mode_counts": (last_trunk_report or {}).get(
            "eef_balance_mode_counts",
            {},
        ),
        "trunk_q_final": [
            round(float(x), 4)
            for x in np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4).tolist()
        ],
        "theta_z_offline_deg": round(theta_z, 2),
        "theta_z_exec_deg": round(
            float((last_trunk_report or {}).get("theta_z_target_deg", theta_z)), 2
        ),
        "reach_note": (
            None
            if reach_ok
            else (
                f"肩距仍>{_REACH_OK_MAX_DIST_M:.2f}m"
                + (
                    "（已尝试 forward/pitch 补偿）"
                    if reach_compensation and reach_compensation.get("attempted")
                    else ""
                )
            )
            if tool == "move_to_point"
            else (None if dual["reachable"] else "双臂均不可达（仅提示，不影响 ok）")
        ),
        **extra,
    }
    # 兼容各 tool 的球心字段名（几何/执行逻辑相同，仅 C 来源不同）
    if tool == "move_to_object":
        result["object_center_world"] = result["target_center_world"]
    elif tool == "move_to_point":
        result["point_center_world"] = result["target_center_world"]

    ctx.set_result(result)
    _save_nav_reach(session_id, result)

    ctx.log(
        f"[{log_tag}] 完成 ok={position_reached} reach_ok={reach_ok} "
        f"head_png={head_live.get('head_png')} "
        f"head_all_in={head_ok} reach L/R={dual['reachable_left']}/"
        f"{dual['reachable_right']} "
        f"shoulder_dist L/R={shoulder_distance['left_shoulder_to_object_m']}/"
        f"{shoulder_distance['right_shoulder_to_object_m']}m "
        f"fill={head_live.get('fill_visible')} "
        f"pose_err={pose_errors}"
    )
    try:
        grip_now = {
            arm: [round(float(x), 5) for x in (world.gripper_qpos_list(arm) or [])]
            for arm in ("left", "right")
        }
        grip_pin = {
            arm: [round(float(x), 5) for x in (world.gripper_pin_qpos_list(arm) or [])]
            for arm in ("left", "right")
        }
        ctx.log(f"[{log_tag}] gripper after move current={grip_now} pin={grip_pin}")
    except Exception as exc:
        ctx.log(f"[{log_tag}] WARN gripper after move read failed: {exc}")
    ctx.log(f"[{log_tag}] 退出 capture 前稳定 {_NAV_EXIT_SETTLE_S:.0f}s…")
    world._codex_fast_motion_no_obs = True
    try:
        yield from yield_move_settle(world, seconds=_NAV_EXIT_SETTLE_S)
        # 退出 no_obs 前预热相机 annotator：整段运动都在 no_obs，从未取 obs，
        # 分割 annotator 第一次取数会返回空张量，导致主循环恢复 normal env.step
        # 时 th.max(empty) 崩溃。此处在仍 no_obs 的窗口内预热（不计入运动耗时）。
        warm_obs_annotators(world, ctx, log_tag)
    finally:
        world._codex_fast_motion_no_obs = old_no_obs

@register_skill(
    "move_to_object_v2",
    description=(
        "点选或指定物体→AABB 中心为弦球球心；"
        "移动/姿态与 move_to_point 相同：①升降 ②xy ③spin/yaw ④纯q3俯仰；"
        f"reach 弦球半径（0={REACH_SPHERE_R_M:.2f}m）。"
    ),
)
def move_to_object_v2(
    ctx,
    session_id: str,
    object_name: str = "",
    image_id: str = "",
    u: Optional[int] = None,
    v: Optional[int] = None,
    standoff: float = 0.0,
    reach: float = 0.0,
    nav_timeout_s: float = _NAV_TIMEOUT_S_DEFAULT,
) -> Generator:
    """reach (m)：弦球半径。0 表示默认 REACH_SPHERE_R_M。standoff 保留兼容，当前未使用。"""
    world = ctx.world
    name = (object_name or "").strip()
    pick_uv = (
        not name
        and str(image_id or "").strip() != ""
        and u is not None
        and v is not None
    )

    def _fail(msg: str) -> Generator:
        ctx.log(f"[move_to_object_v2] 中止: {msg}")
        yield from _hold(world, 4)
        head_live = _capture_head_snapshot(
            world,
            session_id=session_id,
            head_png_tag="move_to_object_head",
            log_tag="move_to_object_v2",
            ctx=ctx,
            suffix="failure",
        )
        ctx.set_result({
            "ok": False,
            "error": msg,
            "tool": "move_to_object",
            "visibility": head_live,
            "head_after_move_png": head_live.get("head_png"),
        })
        yield from _hold(world)
        return

    if name:
        object_name = name
        obj = _resolve(world, name)
        ctx.log(f"[move_to_object_v2] 指定物体优先: {object_name}")
    elif pick_uv:
        bddl, obj, err = _resolve_from_uv(ctx, session_id, image_id.strip(), int(u), int(v))
        if err:
            yield from _fail(err)
            return
        object_name = bddl
        ctx.log(
            f"[move_to_object_v2] 点选物体: {object_name} "
            f"(image_id={image_id.strip()} u={int(u)} v={int(v)})"
        )
    else:
        yield from _fail("需要 object_name，或 image_id + u + v 点选物体")
        return

    if obj is None:
        yield from _fail(f"未找到物体: {object_name}")
        return

    ab = _aabb(obj)
    if ab is None:
        yield from _fail(f"物体 {object_name} 无 AABB")
        return
    lo, hi = ab
    C = (lo + hi) / 2.0
    ctx.log(
        f"[move_to_object_v2] 物体「{object_name}」弦球球心 C=aabb_center "
        f"{np.asarray(C).round(3).tolist()}"
    )

    yield from _run_move_to_center(
        ctx,
        session_id=session_id,
        C=C,
        lo=lo,
        hi=hi,
        reach=reach,
        tool="move_to_object",
        build_id=_MOVE_TO_OBJECT_BUILD,
        log_tag="move_to_object_v2",
        head_png_tag="move_to_object_head",
        result_extra={
            "object_name": object_name,
            "resolved_by": "uv_pick" if pick_uv else "name",
            "image_id": image_id.strip() if pick_uv else None,
            "uv": [int(u), int(v)] if pick_uv else None,
            "target_center_world": [float(x) for x in C],
            "sphere_center_source": "object_aabb_center",
            "exec_order": "lift_xy_spin_pitch",
            "trunk_mode": "q3_pitch_after_nav",
        },
        travel_hint="若仍不可达请换物体或重新点选",
        nav_timeout_s=nav_timeout_s,
    )
