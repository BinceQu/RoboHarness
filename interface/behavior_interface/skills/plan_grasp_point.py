"""grasp_point 模式：用户点击指定夹取邻域，在 hit 周围规划最优夹取 pose。

与 plan_grasp_object 仅 step4 不同；step5–8 与 unittest v3 / v13 grasp_obj 同口径。

输入：
  - head 冻结视图 + 点击 (u,v)：反解射线 3D 命中点 click_hit（黄球=点击邻域中心）
  - object_name：可选；未填时由 (u,v) 反解 hit 再解析点击处物体（同 mark_object）

规划流程（v1，与 assets/unittest/grasp_point_pipeline_v1 同口径）：
  1. 点击 (u,v) + depth/点云反解 3D hit（click_hit）
  2. 以 hit 为球心、半径 1cm 得邻域球（仅示意，不参与排序）
  3. mesh 表面采样：球内随机取 10 个表面点（非占据体积排序）
  4. 10 点作夹爪 1/3 黄点锚点 × 正20面体 20 朝向 × 绕爪轴 8 等分自转 → 1600 pose
  5. fast_overlap 初筛 top60 → GPU 批量 v7 grasp_vol + overlap_vol
  6. overlap 用夹爪法向膨胀 3mm 重算；overlap_vol < 1cm³ 中取 grasp_vol 最大
  7. 返回 best pose；head 主视图叠影：绿点=点击、黄球=click_hit、锚点=gap_center

与 plan_grasp_object：点击指定「在哪夹」，锚点来自 hit 1cm 球内随机表面点，非全物体最小开口球。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from behavior_interface.skills.grasp_point_pipeline_core import (
    GRASP_POINT_FILTER_PIPELINE_BUILD,
    GRASP_POINT_PIPELINE_BUILD,
    NEAR_HIT_FILTER_RADIUS_MM,
    NEAR_HIT_RADIUS_M,
    NEAR_HIT_RADIUS_MM,
    TOP_K_ANCHORS,
)
from behavior_interface.skills.gripper_camera_face import (
    apply_camera_face_forward_to_grasp_dict,
)

# 进程日志 / plan_audit 中应出现此串，用于确认代码版本已加载
GRASP_POINT_BUILD = GRASP_POINT_PIPELINE_BUILD


def _scalar_bool_field(d: Dict[str, Any], key: str) -> bool:
    v = d.get(key)
    if hasattr(v, "item"):
        try:
            return bool(v.item())
        except Exception:
            pass
    return bool(v)


def _build_grip_fit_from_best(
    best: Dict[str, Any],
    *,
    pcd_np: np.ndarray,
    click_hit: np.ndarray,
    build: str = GRASP_POINT_BUILD,
    near_radius_mm: float = NEAR_HIT_RADIUS_MM,
    sample_mode: str = "v1_near_hit_icosa",
) -> Dict[str, Any]:
    """与 plan_grasp_object.compute_eef_from_pcd_grasp_object 的 grip_fit 字段对齐。"""
    centroid = pcd_np.mean(axis=0) if len(pcd_np) >= 1 else click_hit
    gap_center = np.asarray(best["gap_center"], dtype=np.float64).reshape(3)
    return {
        "feasible": bool(best.get("feasible", True)),
        "has_volume": float(best.get("grasp_vol_cm3", 0.0)) > 0.0,
        "n_collision": int(best.get("ncol_exp", 0)),
        "n_collision_expanded": int(best.get("ncol_exp", 0)),
        "n_gap": int(best.get("grasp_vox", best.get("open_intersect_vox", 0))),
        "gap_voxel_n": int(best.get("grasp_vox", best.get("open_intersect_vox", 0))),
        "gap_vol_cm3": float(best.get("grasp_vol_cm3", 0.0)),
        "n_pos_side": 0,
        "n_neg_side": 0,
        "has_both_sides": True,
        "base_dist_mm": float(best.get("base_dist_mm", 0.0)),
        "y_balance": float(best.get("y_balance", 1.0)),
        "center_bias_mm": float(best.get("center_bias_mm", 0.0)),
        "min_rail_clear_mm": float(best.get("min_rail_clear_mm", 0.0)),
        "approach_label": best.get("approach_label", "v12_icosa"),
        "ref_y_label": best.get("ref_y_label", "v12"),
        "roll_deg": float(best.get("roll_deg", 0.0)),
        "reachable": _scalar_bool_field(best, "reachable"),
        "reach_reason": str(best.get("reach_reason", "")),
        "n_score_pcd": int(len(pcd_np)),
        "n_mesh_pts": 0,
        "volume_source": best.get("volume_source", "v12_opening_voxel"),
        "mesh_vol_tag": "v1_trimesh_near_hit",
        "n_pose_samples": 1600,
        "sample_mode": sample_mode,
        "obj_centroid": centroid.tolist(),
        "gap_dist_to_obj_mm": float(best.get("anchor_dist_mm", 0.0)),
        "opening_vol_build": best.get("opening_vol_build"),
        "open_intersect_vox": int(best.get("open_intersect_vox", 0)),
        "open_intersect_vol_cm3": float(best.get("open_intersect_vol_cm3", 0.0)),
        "open_vol_method": best.get("open_vol_method", ""),
        "open_voxel_mm": float(best.get("open_voxel_mm", 3.0)),
        "opening_voxel_total": int(best.get("opening_voxel_total", 0)),
        "grasp_vol_cm3": float(best.get("grasp_vol_cm3", 0.0)),
        "overlap_vol_cm3": float(best.get("overlap_vol_cm3", 0.0)),
        "gripper_vol_cm3": float(best.get("gripper_vol_cm3", 0.0)),
        "overlap_frac": float(best.get("overlap_frac", 0.0)),
        "anchor_dist_mm": float(best.get("anchor_dist_mm", 0.0)),
        "grasp_point_build": build,
        "near_hit_radius_mm": float(near_radius_mm),
        "n_near_anchors": int(TOP_K_ANCHORS),
        # 点击射线命中（黄球）；最佳夹取锚点见 anchor_marker_pt / gap_center
        "click_hit_world": click_hit.tolist(),
        "marker_pt": click_hit.tolist(),
        "anchor_marker_pt": gap_center.tolist(),
    }


def compute_eef_from_pcd_grasp_point(
    pcd: np.ndarray,
    shoulder: np.ndarray,
    *,
    hit_world: np.ndarray,
    seed: int = 42,
    world=None,
    arm: str = "right",
    session: Optional[Dict[str, Any]] = None,
    object_name: Optional[str] = None,
    ctx=None,
    gpu_device_id: Optional[int] = None,
    dual_arm_ik_filter: bool = False,
    plan_arm: str = "any",
) -> Optional[Dict[str, Any]]:
    """grasp_point：默认 1cm 老流程；filter 分支走 3cm + camera-face + 双臂 IK。"""
    shoulder = np.asarray(shoulder, dtype=np.float64).reshape(3)
    pcd_np = np.asarray(pcd, dtype=np.float64)
    hit = np.asarray(hit_world, dtype=np.float64).reshape(3)

    if world is not None and object_name and session is not None:
        from behavior_interface.skills.grasp_point_pipeline_core import (
            run_grasp_point_filter_pipeline,
            run_grasp_point_v1_pipeline,
        )

        is_filter = bool(dual_arm_ik_filter)
        if is_filter:
            result = run_grasp_point_filter_pipeline(
                world, str(object_name), hit, shoulder,
                seed=int(seed), arm=str(arm), ctx=ctx,
                gpu_device_id=gpu_device_id,
                plan_arm=plan_arm,
            )
        else:
            result = run_grasp_point_v1_pipeline(
                world, str(object_name), hit, shoulder,
                seed=int(seed), arm=str(arm), ctx=ctx,
                gpu_device_id=gpu_device_id,
            )
        if result is None:
            if ctx:
                label = "grasp_point_filter" if is_filter else "grasp_point"
                ctx.log(f"  [{label}] pipeline 无可用 pose")
            return None
        best, plan_audit = result
        if is_filter:
            camera_face_audit = dict(best.get("camera_face") or {"skipped": True, "reason": "prefilter_applied"})
        else:
            best, camera_face_audit = apply_camera_face_forward_to_grasp_dict(best, world=world)
        plan_audit = dict(plan_audit)
        plan_audit["camera_face"] = camera_face_audit
        build = GRASP_POINT_FILTER_PIPELINE_BUILD if is_filter else GRASP_POINT_BUILD
        gf = _build_grip_fit_from_best(
            best, pcd_np=pcd_np, click_hit=hit,
            build=build,
            near_radius_mm=NEAR_HIT_FILTER_RADIUS_MM if is_filter else NEAR_HIT_RADIUS_MM,
            sample_mode=(
                "v2_near_hit_3cm_icosa_dual_ik_filter"
                if is_filter
                else "v1_near_hit_icosa"
            ),
        )
        gf["camera_face"] = camera_face_audit
        if best.get("selected_pose_ik"):
            gf["selected_pose_ik"] = best.get("selected_pose_ik")
        if ctx:
            label = "grasp_point_filter" if is_filter else "grasp_point"
            near = (plan_audit.get("pick") or {}).get("near_hit_sample") or {}
            ctx.log(
                f"  [{label}] build={build} "
                f"n_anchors={plan_audit.get('n_anchors', 0)} "
                f"dist_mm={near.get('dist_to_hit_mm')}"
            )
            if not camera_face_audit.get("skipped"):
                ctx.log(
                    "  [grasp_point] wrist camera face "
                    f"flip={camera_face_audit.get('flipped')} "
                    f"angle {camera_face_audit.get('angle_before_deg', 0):.1f}°"
                    f"→{camera_face_audit.get('angle_after_deg', 0):.1f}°"
                )
        recommended_arm = best.get("recommended_arm") or best.get("arm")
        out = {
            "pos": np.asarray(best["pos"]).tolist(),
            "quat": np.asarray(best["quat"]).tolist(),
            "gap_center": np.asarray(best["gap_center"]).tolist(),
            "grip_fit": gf,
            "plan_audit": plan_audit,
            "camera_face": camera_face_audit,
        }
        if is_filter:
            out.update({
                "recommended_arm": recommended_arm,
                "arm": recommended_arm,
                "selected_pose_ik": best.get("selected_pose_ik"),
                "selected_pose_ik_q": best.get("selected_pose_ik_q"),
                "ik_solution": best.get("ik_solution"),
                "ik_filter": best.get("ik_filter"),
                "ik_allerr": best.get("ik_allerr"),
            })
        return out

    if ctx:
        ctx.log("  [grasp_point] WARN 缺少 world/session/object_name，无法走 v1 pipeline")
    return None
