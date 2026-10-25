"""grasp_point 规划核心（与 grasp_obj 共享 step5–8，step4 不同）。

与 plan_grasp_object 差异仅在锚点选取（step4）：
  - grasp_obj：100 表面点 → 127mm 球交占据体积升序 → top10
  - grasp_point：用户点击反解 3D 点 hit → 半径 1cm 球内随机取 10 个物体表面点
  - grasp_point_filter：同上，但半径 3cm，并在 GPU-v7 前插入 camera-face flip + 双臂 6D IK 过滤

后续与 unittest v3 同口径：
  5. 10 点 × 正20面体 20 朝向 × 绕爪轴 8 等分自转 = 1600 pose
  6. fast_overlap 初筛 → top60 GPU 批量 v7
  7. inflate overlap 3mm → overlap_vol < 1cm³ 中 grasp_vol 最大
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.skills.grasp_obj_pipeline_core import (
    ENV_COLLISION_MAX_CM3,
    N_ICOSA_FACES,
    N_ROLL,
    OVERLAP_INFLATE_MM,
    OVERLAP_VOL_MAX_CM3,
    PREFILTER_N,
    TOP_K_POINTS,
    VOXEL_MM,
    _format_ik_top20_for_error,
    _pose_ik_hard_constraint_summary,
    _selected_pose_ik_q_map,
    apply_camera_face_to_filter_poses,
    apply_overlap_inflate_to_poses,
    build_env_collision_tree,
    generate_icosa_roll_poses,
    pose_to_plan_candidate,
    select_best_pose_low_overlap,
    step6_prefilter_dual_ik_v7_metrics_gpu,
    step6_prefilter_v7_metrics_gpu,
)

GRASP_POINT_PIPELINE_BUILD = "v1_near_hit_1cm_random10"
GRASP_POINT_FILTER_PIPELINE_BUILD = "v2_near_hit_3cm_filter_external_gpu_ik"
NEAR_HIT_RADIUS_M = 0.01  # 1cm
NEAR_HIT_RADIUS_MM = 10.0
NEAR_HIT_FILTER_RADIUS_M = 0.03  # 3cm
NEAR_HIT_FILTER_RADIUS_MM = 30.0
TOP_K_ANCHORS = TOP_K_POINTS  # 10


def sample_surface_points_near_hit(
    object_tm,
    hit_world: np.ndarray,
    *,
    radius_m: float = NEAR_HIT_RADIUS_M,
    n_points: int = TOP_K_ANCHORS,
    rng: np.random.Generator,
    pool_multiplier: int = 80,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """在 hit 周围 radius 球内随机取 n 个物体 mesh 表面点（非占据体积排序）。"""
    import trimesh

    hit = np.asarray(hit_world, dtype=np.float64).reshape(3)
    r = float(radius_m)
    n_need = int(n_points)
    meta: Dict[str, Any] = {
        "radius_m": r,
        "radius_mm": r * 1000.0,
        "n_requested": n_need,
        "method": "mesh_surface_in_ball_random",
    }
    if object_tm is None:
        meta["error"] = "mesh 为空"
        return np.zeros((0, 3)), meta

    in_ball = np.zeros((0, 3), dtype=np.float64)
    n_pool = max(n_need * int(pool_multiplier), 800)
    for attempt in range(4):
        pts, _ = trimesh.sample.sample_surface(object_tm, n_pool)
        pts = np.asarray(pts, dtype=np.float64)
        d = np.linalg.norm(pts - hit, axis=1)
        in_ball = pts[d <= r + 1e-9]
        meta["pool_size"] = int(n_pool)
        meta["n_in_ball"] = int(len(in_ball))
        if len(in_ball) >= n_need:
            break
        n_pool *= 3

    if len(in_ball) == 0:
        meta["error"] = f"半径 {r*1000:.1f}mm 内无表面点"
        return np.zeros((0, 3)), meta

    if len(in_ball) <= n_need:
        chosen = in_ball
    else:
        idx = rng.choice(len(in_ball), size=n_need, replace=False)
        chosen = in_ball[idx]
    dist_mm = np.linalg.norm(chosen - hit, axis=1) * 1000.0
    meta["n_chosen"] = int(len(chosen))
    meta["dist_to_hit_mm"] = [float(x) for x in dist_mm]
    return np.asarray(chosen, dtype=np.float64), meta


def run_grasp_point_v1_pipeline(
    world,
    object_name: str,
    hit_world: np.ndarray,
    shoulder: np.ndarray,
    *,
    seed: int = 42,
    arm: str = "right",
    ctx=None,
    gpu_device_id: Optional[int] = None,
    near_radius_m: float = NEAR_HIT_RADIUS_M,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """grasp_point v1 全流程，返回 (best_candidate_dict, plan_audit)。"""
    from behavior_interface.skills.grasp import _resolve_object_handle
    from behavior_interface.skills.plan_grasp_opening_volume import (
        clear_opening_volume_caches,
        gripper_solid_voxels_eef,
        load_object_trimesh_world,
        object_centroid_for_volume,
    )

    clear_opening_volume_caches()
    hit = np.asarray(hit_world, dtype=np.float64).reshape(3)
    obj = _resolve_object_handle(world, object_name)
    tm = load_object_trimesh_world(world, object_name)
    obj_c = object_centroid_for_volume(world, object_name)
    if obj is None or tm is None or obj_c is None:
        if ctx:
            ctx.log("  [grasp_point/v1] 物体/mesh/物心失败")
        return None

    rng = np.random.default_rng(int(seed))
    voxel_m = VOXEL_MM / 1000.0

    if ctx:
        ctx.log(
            f"  [grasp_point/v1] build={GRASP_POINT_PIPELINE_BUILD} object={object_name} "
            f"hit={hit.round(4).tolist()} r={near_radius_m*1000:.1f}mm "
            f"→{TOP_K_ANCHORS}表面点×icosa×roll→GPU-v7→inflate→best")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_point 启动")

    top_pts, samp_meta = sample_surface_points_near_hit(
        tm, hit, radius_m=float(near_radius_m),
        n_points=TOP_K_ANCHORS, rng=rng)
    if len(top_pts) < 3:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            samp_meta.get("error") or f"hit 附近 {near_radius_m*1000:.0f}mm 内表面点不足")

    if ctx:
        ctx.log(
            f"  [grasp_point/v1] step4 球内随机 {len(top_pts)} 点 "
            f"dist_mm={samp_meta.get('dist_to_hit_mm')}")

    poses = generate_icosa_roll_poses(top_pts, n_roll=N_ROLL)
    if ctx:
        ctx.log(f"  [grasp_point/v1] pose={len(poses)} ({len(top_pts)}×20×{N_ROLL})")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("step6 GPU-v7")

    env_tree, env_used, env_npts = build_env_collision_tree(world, obj, voxel_m)
    gv_eef = gripper_solid_voxels_eef(voxel_m)
    pre = step6_prefilter_v7_metrics_gpu(
        poses, tm, world, object_name, gv_eef, env_tree,
        voxel_mm=VOXEL_MM,
        prefilter_n=PREFILTER_N,
        env_collision_max_cm3=ENV_COLLISION_MAX_CM3,
        ctx=ctx, gpu_device_id=gpu_device_id)

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("inflate overlap")
    apply_overlap_inflate_to_poses(
        pre, tm, obj_c, voxel_mm=VOXEL_MM, inflate_mm=OVERLAP_INFLATE_MM,
        ctx=ctx, use_gpu=True, gpu_device_id=gpu_device_id)

    best_pm, cand = select_best_pose_low_overlap(
        pre, overlap_vol_max_cm3=OVERLAP_VOL_MAX_CM3, ctx=ctx)
    if best_pm is None:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"无 overlap<{OVERLAP_VOL_MAX_CM3}cm³ 的抓取候选")

    best = pose_to_plan_candidate(best_pm, shoulder)
    best["reachable"] = True
    best["reach_reason"] = "v1_near_hit_overlap_best"
    pick_audit = {
        "pick_method": "v1_near_hit_overlap_best",
        "n_overlap_candidates": len(cand),
        "click_hit_world": hit.tolist(),
        "near_hit_sample": samp_meta,
    }

    plan_audit: Dict[str, Any] = {
        "grasp_point_build": GRASP_POINT_PIPELINE_BUILD,
        "pipeline": "v1_near_hit_gpu_inflate",
        "click_hit_world": hit.tolist(),
        "near_radius_mm": float(near_radius_m) * 1000.0,
        "overlap_inflate_mm": float(OVERLAP_INFLATE_MM),
        "overlap_vol_max_cm3": float(OVERLAP_VOL_MAX_CM3),
        "object_name": object_name,
        "seed": int(seed),
        "n_anchors": int(len(top_pts)),
        "n_poses": int(len(poses)),
        "n_v7_prefilter": int(len(pre)),
        "n_overlap_candidates": int(len(cand)),
        "env_collision_objects": [u["name"] for u in env_used],
        "pick": pick_audit,
        "best_pi": best_pm.get("pi"),
        "best_ni": best_pm.get("ni"),
        "best_ri": best_pm.get("ri"),
    }
    return best, plan_audit


def _log_point_filter_ik_top20(ctx, ik_meta: Dict[str, Any]) -> None:
    if not ctx:
        return
    ctx.log(
        f"  [grasp_point_filter] IK GPU rank ranked={ik_meta.get('n_ranked', 0)} "
        f"left={ik_meta.get('n_left_ok', 0)} right={ik_meta.get('n_right_ok', 0)} "
        f"both={ik_meta.get('n_both_ok', 0)} "
        f"elapsed={float(ik_meta.get('elapsed_s', 0.0)):.2f}s"
    )
    ctx.log(
        "  [grasp_point_filter] IK top20 columns: "
        "rank, IKallerr, L_pos_mm, L_ori_deg, R_pos_mm, R_ori_deg, both_strict"
    )
    for item in ik_meta.get("top20") or []:
        ctx.log(
            "  [grasp_point_filter] IK top20: "
            "rank={rank:02d}, IKallerr={allerr:.3f}, "
            "L_pos_mm={left_pos_mm:.2f}, L_ori_deg={left_ori_deg:.2f}, "
            "R_pos_mm={right_pos_mm:.2f}, R_ori_deg={right_ori_deg:.2f}, "
            "both_strict={both_strict}".format(**item)
        )


def run_grasp_point_filter_pipeline(
    world,
    object_name: str,
    hit_world: np.ndarray,
    shoulder: np.ndarray,
    *,
    seed: int = 42,
    arm: str = "right",
    ctx=None,
    gpu_device_id: Optional[int] = None,
    near_radius_m: float = NEAR_HIT_FILTER_RADIUS_M,
    plan_arm: str = "any",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """grasp_point_filter：3cm hit 邻域 + camera-face flip + 双臂 6D IK 过滤。"""
    from behavior_interface.skills.grasp import _resolve_object_handle
    from behavior_interface.skills.plan_grasp_opening_volume import (
        clear_opening_volume_caches,
        gripper_solid_voxels_eef,
        load_object_trimesh_world,
        object_centroid_for_volume,
    )

    clear_opening_volume_caches()
    hit = np.asarray(hit_world, dtype=np.float64).reshape(3)
    obj = _resolve_object_handle(world, object_name)
    tm = load_object_trimesh_world(world, object_name)
    obj_c = object_centroid_for_volume(world, object_name)
    if obj is None or tm is None or obj_c is None:
        if ctx:
            ctx.log("  [grasp_point_filter] 物体/mesh/物心失败")
        return None

    rng = np.random.default_rng(int(seed))
    voxel_m = VOXEL_MM / 1000.0
    plan_arm = str(plan_arm or "any").lower().strip()
    if plan_arm not in ("left", "right", "any"):
        plan_arm = "any"

    if ctx:
        ctx.log(
            f"  [grasp_point_filter] build={GRASP_POINT_FILTER_PIPELINE_BUILD} "
            f"object={object_name} hit={hit.round(4).tolist()} "
            f"r={near_radius_m*1000:.1f}mm "
            f"plan_arm={plan_arm} "
            "→10表面点×icosa×roll→camera-face flip→双臂GPU IK rank→top60 GPU-v7"
        )

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_point_filter 启动")

    top_pts, samp_meta = sample_surface_points_near_hit(
        tm, hit, radius_m=float(near_radius_m),
        n_points=TOP_K_ANCHORS, rng=rng)
    if len(top_pts) < 3:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            samp_meta.get("error") or f"hit 附近 {near_radius_m*1000:.0f}mm 内表面点不足")

    if ctx:
        ctx.log(
            f"  [grasp_point_filter] step4 球内随机 {len(top_pts)} 点 "
            f"dist_mm={samp_meta.get('dist_to_hit_mm')}")

    poses = generate_icosa_roll_poses(top_pts, n_roll=N_ROLL)
    apply_camera_face_to_filter_poses(poses, world, ctx=ctx)
    if ctx:
        ctx.log(
            f"  [grasp_point_filter] pose={len(poses)} "
            f"({len(top_pts)}×{N_ICOSA_FACES}×{N_ROLL})")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_point_filter dual IK")

    import time as _time
    _t_env = _time.perf_counter()
    env_tree, env_used, env_npts = build_env_collision_tree(world, obj, voxel_m)
    gv_eef = gripper_solid_voxels_eef(voxel_m)
    if ctx:
        ctx.log(
            f"  [grasp_point_filter] env树 {len(env_used)} 物体 {env_npts} 点 + 夹爪体素 {len(gv_eef)} "
            f"耗时 {_time.perf_counter() - _t_env:.1f}s")
    pre, ik_meta = step6_prefilter_dual_ik_v7_metrics_gpu(
        poses, tm, world, object_name, gv_eef, env_tree,
        voxel_mm=VOXEL_MM,
        prefilter_n=PREFILTER_N,
        env_collision_max_cm3=ENV_COLLISION_MAX_CM3,
        ctx=ctx, gpu_device_id=gpu_device_id,
        plan_arm=plan_arm)
    _log_point_filter_ik_top20(ctx, ik_meta)
    if not pre:
        from behavior_interface.errors import GraspObjPlanningError
        top20_s = _format_ik_top20_for_error(ik_meta)
        raise GraspObjPlanningError(
            "grasp_point_filter 没有可排序 IK pose "
            f"(ranked={ik_meta.get('n_ranked', 0)}, left={ik_meta.get('n_left_ok', 0)}, "
            f"right={ik_meta.get('n_right_ok', 0)}, both={ik_meta.get('n_both_ok', 0)}, "
            f"plan_arm={ik_meta.get('plan_arm', plan_arm)}, selection={ik_meta.get('selection')}, "
            f"elapsed={ik_meta.get('elapsed_s', 0.0):.2f}s)"
            + (f" top20={top20_s}" if top20_s else "")
        )

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_point_filter inflate overlap")
    apply_overlap_inflate_to_poses(
        pre, tm, obj_c, voxel_mm=VOXEL_MM, inflate_mm=OVERLAP_INFLATE_MM,
        ctx=ctx, use_gpu=True, gpu_device_id=gpu_device_id)

    best_pm, cand = select_best_pose_low_overlap(
        pre, overlap_vol_max_cm3=OVERLAP_VOL_MAX_CM3, ctx=ctx)
    if best_pm is None:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"grasp_point_filter IK过滤后无 overlap<{OVERLAP_VOL_MAX_CM3}cm³ 的抓取候选 "
            f"(IK both={ik_meta.get('n_both_ok', 0)}, plan_arm={ik_meta.get('plan_arm', plan_arm)})")

    selected_pose_ik = _pose_ik_hard_constraint_summary(best_pm)
    selected_pose_ik_q = _selected_pose_ik_q_map(selected_pose_ik)
    best = pose_to_plan_candidate(best_pm, shoulder)
    best_arm = ik_meta.get("recommended_arm") or str(arm)
    best["arm"] = best_arm
    best["recommended_arm"] = best_arm
    best["reachable"] = True
    best["reach_reason"] = "v2_near_hit_dual_arm_gpu_ik_rank_overlap_best"
    best["ik_filter"] = best_pm.get("ik_filter")
    best["ik_allerr"] = best_pm.get("ik_allerr")
    best["selected_pose_ik"] = selected_pose_ik
    best["selected_pose_ik_q"] = selected_pose_ik_q
    best["ik_solution"] = selected_pose_ik.get("solution")
    best["camera_face"] = best_pm.get("camera_face")
    pick_audit = {
        "pick_method": "v2_near_hit_dual_arm_gpu_ik_rank_overlap_best",
        "n_overlap_candidates": len(cand),
        "click_hit_world": hit.tolist(),
        "near_hit_sample": samp_meta,
        "ik_filter": ik_meta,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
    }

    plan_audit: Dict[str, Any] = {
        "grasp_point_build": GRASP_POINT_FILTER_PIPELINE_BUILD,
        "pipeline": "v2_near_hit_dual_arm_gpu_ik_rank_gpu_inflate_overlap",
        "click_hit_world": hit.tolist(),
        "near_radius_mm": float(near_radius_m) * 1000.0,
        "overlap_inflate_mm": float(OVERLAP_INFLATE_MM),
        "overlap_vol_max_cm3": float(OVERLAP_VOL_MAX_CM3),
        "env_collision_max_cm3": float(ENV_COLLISION_MAX_CM3),
        "use_gpu_step6": True,
        "dual_arm_ik_filter": True,
        "plan_arm": plan_arm,
        "recommended_arm": best_arm,
        "ik_filter": ik_meta,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "object_name": object_name,
        "seed": int(seed),
        "n_anchors": int(len(top_pts)),
        "n_poses": int(len(poses)),
        "n_v7_prefilter": int(len(pre)),
        "n_overlap_candidates": int(len(cand)),
        "env_collision_objects": [u["name"] for u in env_used],
        "env_collision_points": int(env_npts),
        "pick": pick_audit,
        "best_pi": best_pm.get("pi"),
        "best_ni": best_pm.get("ni"),
        "best_ri": best_pm.get("ri"),
    }
    if ctx:
        ctx.log(
            f"  [grasp_point_filter] 最终 grasp_vol={best['grasp_vol_cm3']:.2f} "
            f"overlap_vol={best['overlap_vol_cm3']:.3f} "
            f"pi={best_pm.get('pi')} ni={best_pm.get('ni')} ri={best_pm.get('ri')} "
            f"IK both={ik_meta.get('n_both_ok', 0)} topGPU={ik_meta.get('n_gpu_v7', 0)} "
            f"plan_arm={ik_meta.get('plan_arm')} selection={ik_meta.get('selection')} "
            f"selected_pose_ik={selected_pose_ik.get('solution')} "
            f"L={selected_pose_ik.get('left_pos_mm')}mm/{selected_pose_ik.get('left_ori_deg')}deg "
            f"R={selected_pose_ik.get('right_pos_mm')}mm/{selected_pose_ik.get('right_ori_deg')}deg "
            f"default_exec_arm={best_arm}"
        )
    return best, plan_audit
