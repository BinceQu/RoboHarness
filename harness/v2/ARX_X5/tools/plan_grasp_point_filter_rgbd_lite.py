from __future__ import annotations

from pathlib import Path
import numpy as np
from tools.util.storage import file_data_url
from tools.util.background_planning import (
    BUILD as RGBD_PLANNING_WORKER_BUILD,
    run_frozen_rgbd_planner,
)
from tools.util.x5_grasp_planner import (
    X5GraspPlanningError,
)
from tools.util.x5_grasp_planner import (
    plan_x5_grasp_point_filter_rgbd_lite,
)
from tools.util.x5_gripper_visualization import (
    render_x5_grasp_plan_three_views,
    render_x5_gripper_overlay,
)

from tools.util.motion_common import _current_snapshot, _save_pose_plan, _set_result


def plan_grasp_point_filter_rgbd_lite(
    ctx,
    session_id: str,
    image_id: str,
    u: float,
    v: float,
    arm: str = "right",
    seed: int = 42,
    plan_arm: str = "any",
    **kwargs,
):
    del arm
    try:
        del kwargs
        snapshot = _current_snapshot(
            ctx, ctx.snapshots.load(str(session_id), str(image_id))
        )
        if snapshot.role != "head":
            raise ValueError("selected image_id is not a head-camera capture")
        capture_epoch = int(snapshot.robot.get("motion_epoch", -1))
        if capture_epoch != ctx.world.motion_epoch():
            raise ValueError(
                "capture is stale because an evaluator action occurred after capture"
            )
        payload = yield from run_frozen_rgbd_planner(
            ctx,
            plan_x5_grasp_point_filter_rgbd_lite,
            snapshot=snapshot,
            u=float(u),
            v=float(v),
            plan_arm=str(plan_arm),
            seed=int(seed),
            return_scene_mesh_result=True,
        )
        payload["plan_audit"]["planning_execution"] = {
            "build": RGBD_PLANNING_WORKER_BUILD,
            "mode": "background_read_only_frozen_inputs",
            "waiting_action": "defer_no_evaluator_step",
            "episode_and_motion_epoch_revalidated": True,
            "evaluator_rpc_timeout_changed": False,
            "public_tool_arguments_changed": False,
        }
        scene_mesh_result = payload.pop("_scene_mesh_result", None)
        selected_arm = str(payload["recommended_arm"])
        safe_pose = dict(payload["safe_eef_pose"])
        final_pose = dict(payload["eef_pose"])
        ideal_safe_pose = dict(payload["ideal_safe_eef_pose"])
        ideal_final_pose = dict(payload["ideal_eef_pose"])
        parameters = {
            "point": payload["point_observation"],
            "seed": int(seed),
            "plan_arm": str(plan_arm),
            "pos_tol": 0.012,
            "ori_tol_deg": 5.0,
            "gripper_policy": "preserve",
            "compat_plan_mode": "grasp_point_filter_rgbd_lite",
            "compat_gripper_transition": "preserve",
            "exec_sequence": "move_only",
            "semantic_status": "rgbd_volume_and_paired_local_ik",
            "planner_variant": payload["planner_variant"],
            "execution_pose_source": payload["execution_pose_source"],
            "cartesian_metrics_semantics": (
                "reported pose and volume use capture-calibrated FK of signed joints; "
                "candidate selection uses minimum ideal/FK grasp volume"
            ),
            "selected_pose_ik_q": payload["selected_pose_ik_q"],
            "ik_solution": payload["ik_solution"],
            "grip_fit": payload["grip_fit"],
            "plan_audit": payload["plan_audit"],
        }
        result = _save_pose_plan(
            ctx,
            tool_name="plan_grasp_point_filter_rgbd_lite",
            snapshot=snapshot,
            arm=selected_arm,
            waypoints=[
                {
                    "label": "safe",
                    "pose": [*safe_pose["pos"], *safe_pose["quat"]],
                    "planned_joint_target": payload["ik_solution"]["safe_q"],
                    "pose_source": "capture_calibrated_fk_of_signed_safe_q",
                    "ideal_pose": [
                        *ideal_safe_pose["pos"],
                        *ideal_safe_pose["quat"],
                    ],
                    "planning_ik_position_residual_m": payload["ik_solution"][
                        "safe_position_error_m"
                    ],
                    "planning_ik_orientation_residual_deg": payload["ik_solution"][
                        "safe_orientation_error_deg"
                    ],
                },
                {
                    "label": "grasp",
                    "pose": [*final_pose["pos"], *final_pose["quat"]],
                    "planned_joint_target": payload["ik_solution"]["final_q"],
                    "pose_source": "capture_calibrated_fk_of_signed_final_q",
                    "ideal_pose": [
                        *ideal_final_pose["pos"],
                        *ideal_final_pose["quat"],
                    ],
                    "planning_ik_position_residual_m": payload["ik_solution"][
                        "final_position_error_m"
                    ],
                    "planning_ik_orientation_residual_deg": payload["ik_solution"][
                        "final_orientation_error_deg"
                    ],
                },
            ],
            parameters=parameters,
        )
        rgb_path = Path(snapshot.paths.get("rgb", snapshot.paths["rgb_array"]))
        overlay_path = rgb_path.with_name(
            f"{snapshot.image_id}.{result['plan_id']}.gripper.png"
        )
        three_view_path = rgb_path.with_name(
            f"{snapshot.image_id}.{result['plan_id']}.grasp_three_views.png"
        )
        visualization = render_x5_gripper_overlay(
            rgb=snapshot.rgb,
            depth_linear=snapshot.depth,
            intrinsic_matrix=snapshot.intrinsic_matrix,
            extrinsic_matrix=snapshot.extrinsic_matrix,
            eef_pose=final_pose,
            output_path=overlay_path,
        )
        three_views = render_x5_grasp_plan_three_views(
            rgb=snapshot.rgb,
            depth_linear=snapshot.depth,
            intrinsic_matrix=snapshot.intrinsic_matrix,
            extrinsic_matrix=snapshot.extrinsic_matrix,
            eef_pose=final_pose,
            hit_world=payload["hit_world"],
            grasp_vol_cm3=payload["grasp_vol_cm3"],
            overlap_vol_cm3=payload["overlap_vol_cm3"],
            inflated_overlap_vol_cm3=payload["inflated_overlap_vol_cm3"],
            arm=selected_arm,
            axial_point=str(payload["grip_fit"]["axial_point"]),
            output_path=three_view_path,
            support_plane_z=payload["grip_fit"].get("support_plane_z_m"),
            opening_surface_only=str(
                payload["grip_fit"].get("volume_source", "")
            ).endswith("surface_only"),
            opening_target_world=np.asarray(
                payload.get("grasp_anchor_world", payload["hit_world"]),
                dtype=np.float64,
            ),
            opening_target_radius_m=payload["grip_fit"].get("opening_target_radius_m"),
            scene_occupancy_route=payload["grip_fit"].get("mesh_effective_version"),
            scene_occupancy_fallback_reason=payload["grip_fit"].get(
                "mesh_fallback_reason"
            ),
            scene_mesh_result=scene_mesh_result,
            opening_query_version=payload["grip_fit"].get("opening_query_version"),
            surface_query_radius_cells=payload["grip_fit"].get(
                "surface_query_radius_cells"
            ),
        )
        debug_images = {}
        if visualization.get("ok"):
            debug_images["frozen_rgb_overlay"] = str(overlay_path)
        if three_views.get("ok"):
            debug_images["rgbd_mesh_three_views"] = str(three_view_path)
        payload["plan_audit"]["visualization"] = visualization
        payload["plan_audit"]["three_view_visualization"] = three_views
        result.update(payload)
        result.update(
            {
                "gripper_visualization": visualization,
                "grasp_plan_three_views_metadata": three_views,
                "debug_images": debug_images,
                "render_image": (
                    file_data_url(overlay_path)
                    if bool(visualization.get("ok"))
                    else None
                ),
                "render_image_path": (
                    str(overlay_path) if bool(visualization.get("ok")) else None
                ),
                "rgb_overlay_path": (
                    str(overlay_path) if bool(visualization.get("ok")) else None
                ),
                "render_image_3d": (
                    str(three_view_path) if bool(three_views.get("ok")) else None
                ),
                "render_image_3d_path": (
                    str(three_view_path) if bool(three_views.get("ok")) else None
                ),
                "3d_point_render": (
                    str(three_view_path) if bool(three_views.get("ok")) else None
                ),
                "grasp_volume_three_views": (
                    file_data_url(three_view_path)
                    if bool(three_views.get("ok"))
                    else None
                ),
                "grasp_volume_three_views_path": (
                    str(three_view_path) if bool(three_views.get("ok")) else None
                ),
                "grasp_volume_three_views_metadata": three_views,
                "identity_source": "agent_selected_rgbd_point",
                "identity_verified": False,
                "rgbd_collision_estimated": True,
                "collision_checked": False,
                "observation_contract": [
                    "frozen_evaluator_rgb",
                    "frozen_evaluator_depth_linear",
                    "frozen_evaluator_camera_matrices",
                    "capture_time_evaluator_proprioception",
                    "policy_local_static_official_x5_model",
                ],
                "segmentation_used": False,
                "object_identity_used": False,
                "simulator_mesh_used": False,
                "simulator_fk_ik_used": False,
                "direct_simulator_mutation": False,
            }
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        X5GraspPlanningError,
    ) as exc:
        result = {
            "ok": False,
            "error": str(exc),
            "plan_audit": getattr(exc, "audit", {}),
        }
    _set_result(ctx, "plan_grasp_point_filter_rgbd_lite", result)
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("plan_grasp_point_filter_rgbd_lite", plan_grasp_point_filter_rgbd_lite)
