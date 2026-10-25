"""Frozen RGB-D overlap filter for RoboDojo X5 move-tracked endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.util.storage import Snapshot
from tools.util.tracked_point_motion_x5_support import quat_to_mat_xyzw
from tools.util.x5_grasp_geometry import VOXEL_M, x5_gripper_geometry
from tools.util.x5_grasp_planner import (
    INFLATED_OVERLAP_MAX_CM3,
    ORIGINAL_OVERLAP_MAX_CM3,
    _build_scene_occupancy,
    estimate_support_plane_z,
)


BUILD = "official_v2_move_tracked_x5_frozen_rgbd_overlap_v2_request_scene_cache"
_PREPARED_SCENE_KEY = "_x5_move_tracked_prepared_overlap_scene_v2"


@dataclass(frozen=True)
class _PreparedOverlapScene:
    snapshot: Snapshot
    base_rotation: np.ndarray
    support_z: float | None
    support_report: Mapping[str, Any]
    occupancy: Any
    mesh_result: Any
    mesh_route: str
    mesh_fallback: str | None


def _load_snapshot(context: Mapping[str, Any]) -> Snapshot:
    rgb_path = Path(str(context["rgb_array_path"]))
    depth_path = Path(str(context["depth_path"]))
    rgb = np.load(rgb_path, allow_pickle=False)
    depth = np.load(depth_path, allow_pickle=False)
    intrinsic = np.asarray(context["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(context["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
    if (
        rgb.ndim != 3
        or rgb.shape[2] < 3
        or depth.shape != rgb.shape[:2]
        or not np.all(np.isfinite(intrinsic))
        or not np.all(np.isfinite(extrinsic))
    ):
        raise ValueError("frozen move-tracked RGB-D overlap input is invalid")
    return Snapshot(
        session_id=str(context.get("session_id") or "frozen"),
        image_id=str(context.get("image_id") or "frozen"),
        role="head",
        rgb=np.asarray(rgb[:, :, :3], dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=extrinsic,
        robot={},
        sequence=int(context.get("observation_sequence", -1)),
        episode_id=str(context.get("episode_id") or ""),
        paths={"rgb_array": str(rgb_path), "depth": str(depth_path)},
    )


def _prepare_scene(
    context: Mapping[str, Any],
    *,
    target_anchor: np.ndarray,
) -> tuple[_PreparedOverlapScene, bool]:
    """Build the immutable RGB-D occupancy once per isolated plan request."""

    cached = context.get(_PREPARED_SCENE_KEY)
    if isinstance(cached, _PreparedOverlapScene):
        return cached, True

    snapshot = _load_snapshot(context)
    if snapshot.depth is None:
        raise ValueError("frozen move-tracked overlap evaluation requires depth")
    base_rotation = np.asarray(context["base_rotation"], dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(base_rotation)):
        raise ValueError("move-tracked base rotation is invalid")
    support_z, support_report = estimate_support_plane_z(
        snapshot.depth,
        snapshot.intrinsic_matrix,
        snapshot.extrinsic_matrix,
    )
    occupancy, mesh_result, mesh_route, mesh_fallback = _build_scene_occupancy(
        snapshot,
        support_z=support_z,
        target_anchor=np.asarray(target_anchor, dtype=np.float64).reshape(3),
    )
    prepared = _PreparedOverlapScene(
        snapshot=snapshot,
        base_rotation=base_rotation.copy(),
        support_z=support_z,
        support_report=dict(support_report),
        occupancy=occupancy,
        mesh_result=mesh_result,
        mesh_route=str(mesh_route),
        mesh_fallback=(None if mesh_fallback is None else str(mesh_fallback)),
    )
    # The planner passes a private mutable copy of the wire-format context.
    # Keeping this object there bounds reuse to one worker process and one
    # frozen request; no scene state crosses evaluator observations.
    if isinstance(context, dict):
        context[_PREPARED_SCENE_KEY] = prepared
    return prepared, False


def evaluate_x5_move_endpoint_overlap(
    context: Mapping[str, Any],
    poses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the same two-stage Lite solid-overlap gates to X5 EEF poses."""

    requested = list(poses)
    if not requested:
        raise ValueError("move-tracked overlap evaluation requires at least one pose")
    converted: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    base_rotation = np.asarray(context["base_rotation"], dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(base_rotation)):
        raise ValueError("move-tracked base rotation is invalid")
    for pose in requested:
        position_base = np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
        quaternion_xyzw = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quaternion_xyzw))
        if (
            not np.all(np.isfinite(position_base))
            or not np.all(np.isfinite(quaternion_xyzw))
            or norm <= 1.0e-12
        ):
            raise ValueError("move-tracked overlap pose is invalid")
        quaternion_xyzw = quaternion_xyzw / norm
        position_env = base_rotation @ position_base
        rotation_env = base_rotation @ quat_to_mat_xyzw(quaternion_xyzw)
        converted.append((position_base, quaternion_xyzw, position_env, rotation_env))

    target_anchor = converted[0][2]
    prepared, scene_reused = _prepare_scene(
        context,
        target_anchor=target_anchor,
    )
    # The target anchor affects only opening-volume queries. This tool asks
    # exclusively for solid gripper overlap, so one frozen collision occupancy
    # is valid for every endpoint frontier in this request.
    occupancy = prepared.occupancy
    geometry = x5_gripper_geometry()
    voxel_cm3 = float(VOXEL_M**3 * 1.0e6)
    pose_reports: list[dict[str, Any]] = []
    passing_indices: list[int] = []
    for index, (
        position_base,
        quaternion_xyzw,
        position_env,
        rotation_env,
    ) in enumerate(converted):
        original_mask = occupancy.occupied_mask(
            geometry["gripper_voxels"], rotation_env, position_env
        )
        original_voxels = int(np.count_nonzero(original_mask))
        original_cm3 = float(original_voxels * voxel_cm3)
        original_ok = bool(original_cm3 <= ORIGINAL_OVERLAP_MAX_CM3 + 1.0e-12)
        inflated_voxels: int | None = None
        inflated_cm3: float | None = None
        inflated_ok = False
        if original_ok:
            inflated_mask = occupancy.occupied_mask(
                geometry["inflated_gripper_voxels"], rotation_env, position_env
            )
            inflated_voxels = int(np.count_nonzero(inflated_mask))
            inflated_cm3 = float(inflated_voxels * voxel_cm3)
            inflated_ok = bool(inflated_cm3 < INFLATED_OVERLAP_MAX_CM3)
        ok = bool(original_ok and inflated_ok)
        if ok:
            passing_indices.append(index)
        pose_reports.append(
            {
                "ok": ok,
                "evaluation_available": True,
                "reason": (
                    None
                    if ok
                    else "original_overlap"
                    if not original_ok
                    else "inflated_overlap"
                ),
                "pose_index": index,
                "eef_position_robot_base_m": position_base.astype(float).tolist(),
                "eef_quaternion_xyzw": quaternion_xyzw.astype(float).tolist(),
                "eef_position_robodojo_env_m": position_env.astype(float).tolist(),
                "overlap_vox": original_voxels,
                "overlap_vol_cm3": original_cm3,
                "original_overlap_ok": original_ok,
                "inflated_overlap_vox": inflated_voxels,
                "inflated_overlap_vol_cm3": inflated_cm3,
                "inflated_overlap_ok": inflated_ok,
                "inflated_query_skipped": not original_ok,
                "original_overlap_threshold_cm3": float(ORIGINAL_OVERLAP_MAX_CM3),
                "original_overlap_comparison": "<=",
                "inflated_overlap_threshold_cm3": float(INFLATED_OVERLAP_MAX_CM3),
                "inflated_overlap_comparison": "<",
                "voxel_m": float(VOXEL_M),
            }
        )

    return {
        "ok": bool(passing_indices),
        "build": BUILD,
        "evaluation_available": True,
        "pose_count": len(requested),
        "passing_pose_count": len(passing_indices),
        "rejected_pose_count": len(requested) - len(passing_indices),
        "passing_pose_indices": passing_indices,
        "poses": pose_reports,
        "original_overlap_threshold_cm3": float(ORIGINAL_OVERLAP_MAX_CM3),
        "original_overlap_comparison": "<=",
        "inflated_overlap_threshold_cm3": float(INFLATED_OVERLAP_MAX_CM3),
        "inflated_overlap_comparison": "<",
        "support_plane_z_m": prepared.support_z,
        "support_plane_estimation": prepared.support_report,
        "scene_occupancy_route": prepared.mesh_route,
        "scene_occupancy_fallback_reason": prepared.mesh_fallback,
        "scene_reconstruction_reused": scene_reused,
        "scene_reuse_scope": "single_frozen_isolated_planning_request",
        "scene_mesh_metadata": (
            None if prepared.mesh_result is None else prepared.mesh_result.metadata
        ),
        "geometry_source": ("policy_local_frozen_official_x5_open_gripper_3mm_voxels"),
        "segmentation_used": False,
        "simulator_mesh_used": False,
        "simulator_fk_ik_used": False,
        "direct_simulator_mutation": False,
    }


__all__ = ["BUILD", "evaluate_x5_move_endpoint_overlap"]
