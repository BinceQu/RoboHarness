"""Withhold observed X5 gripper rays from grasp material, not collision.

Only frozen public RGB-D/calibration/proprioception and the official policy-local
link6/7/8 visual meshes enter this conservative partition. A matching depth can
also be material touching the robot: it is ambiguous, never contact truth. The
unseen scene behind a finger is not reconstructed, and collision is unchanged.
Unmodelled arm links are explicitly outside this module's scope.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from tools.util.x5_kinematics import quaternion_to_matrix
from tools.util.rgbd_scene_mesh_robodojo_v1 import VoxelOccupancy
from tools.util.x5_grasp_geometry import x5_dynamic_visual_vertices, x5_gripper_geometry

BUILD = "public_rgbd_official_x5_gripper_material_partition_v1"
DEPTH_MATCH_M = 0.003
PIXEL_RADIUS = 1


def render_gripper_depth(snapshot, pose, gripper_qpos):
    # Local import avoids the historical visualization -> planner dependency
    # at import time. Reuse its perspective-correct triangle rasterizer.
    from tools.util.x5_gripper_visualization import _project_world, _rasterize_triangle

    depth = np.full(snapshot.depth.shape, np.inf, dtype=np.float32)
    color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    geometry = x5_gripper_geometry()
    vertices = x5_dynamic_visual_vertices(gripper_qpos)
    world = vertices @ quaternion_to_matrix(pose["quat"]).T + pose["pos"]
    uv, distance = _project_world(
        world, snapshot.intrinsic_matrix, snapshot.extrinsic_matrix
    )
    triangles = uv[geometry["visual_faces"]]
    distances = distance[geometry["visual_faces"]]
    height, width = depth.shape
    valid = (
        np.isfinite(triangles).all(axis=(1, 2))
        & np.isfinite(distances).all(axis=1)
        & (distances > 1e-5).all(axis=1)
    )
    valid &= (
        (triangles[:, :, 0].max(axis=1) >= 0)
        & (triangles[:, :, 0].min(axis=1) < width)
        & (triangles[:, :, 1].max(axis=1) >= 0)
        & (triangles[:, :, 1].min(axis=1) < height)
    )
    for index in np.flatnonzero(valid):
        _rasterize_triangle(
            triangles[index],
            distances[index],
            np.zeros(3, dtype=np.uint8),
            color,
            depth,
        )
    return depth


def matched_depth_mask(scene_depth, model_depth):
    from tools.util.x5_gripper_visualization import _nearby_depth_match_mask

    return _nearby_depth_match_mask(
        scene_depth,
        model_depth,
        radius_px=PIXEL_RADIUS,
        tolerance_m=DEPTH_MATCH_M,
    )


def observed_gripper_mask(snapshot):
    """Depth-register each known gripper, then apply a tighter final match."""
    from tools.util.x5_gripper_visualization import (
        _estimate_x5_gripper_joint_position_from_depth,
    )

    mask = np.zeros(snapshot.depth.shape, dtype=bool)
    arms = {}
    for arm in ("left", "right"):
        pose = snapshot.robot.get(f"eef_{arm}")
        qpos = snapshot.robot.get(f"gripper_{arm}_qpos")
        if not isinstance(pose, dict) or qpos is None:
            arms[arm] = {
                "applied": False,
                "reason": "capture_gripper_proprioception_missing",
            }
            continue
        values = np.r_[
            pose.get("pos", []), pose.get("quat", []), np.asarray(qpos).reshape(-1)
        ]
        if len(values) < 8 or not np.isfinite(values).all():
            arms[arm] = {
                "applied": False,
                "reason": "capture_gripper_proprioception_invalid",
            }
            continue
        selected, _, _, registration = _estimate_x5_gripper_joint_position_from_depth(
            scene_depth=snapshot.depth,
            intrinsic_matrix=snapshot.intrinsic_matrix,
            extrinsic_matrix=snapshot.extrinsic_matrix,
            eef_pose=pose,
            gripper_qpos=qpos,
        )
        model = render_gripper_depth(snapshot, pose, selected)
        matched = matched_depth_mask(snapshot.depth, model)
        mask |= matched
        arms[arm] = {
            "applied": bool(matched.any()),
            "reason": "depth_matched_gripper_or_contact_ambiguity"
            if matched.any()
            else "no_matched_gripper_surface",
            "model_pixel_count": int(np.isfinite(model).sum()),
            "withheld_pixel_count": int(matched.sum()),
            "geometry_qpos": selected,
            "opening_registration": registration,
        }
    return mask, {
        "build": BUILD,
        "scope": "both_public_x5_grippers_link6_link7_link8_only",
        "applied": bool(mask.any()),
        "arms": arms,
        "withheld_pixel_count": int(mask.sum()),
        "depth_match_tolerance_m": DEPTH_MATCH_M,
        "pixel_radius": PIXEL_RADIUS,
        "classification": "robot_surface_or_near_contact_ambiguous_not_grasp_material",
        "unmodelled_arm_links_removed": False,
        "collision_occupancy_changed": False,
        "occluded_space_declared_free": False,
        "segmentation_or_object_identity_used": False,
        "simulator_state_or_contact_truth_used": False,
        "inputs": [
            "frozen_metric_depth",
            "camera_calibration",
            "public_gripper_proprioception",
            "frozen_official_gripper_visual_mesh",
        ],
    }


def points_on_withheld_rays(points_world, mask, intrinsic, extrinsic):
    """Unknown/off-image projections are retained, not classified as robot."""
    from tools.util.x5_gripper_visualization import _project_world

    uv, distance = _project_world(points_world, intrinsic, extrinsic)
    height, width = mask.shape
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(distance)
        & (distance > 0.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    selected = np.flatnonzero(valid)
    pixels = np.rint(uv[selected]).astype(np.int64)
    result = np.zeros(len(uv), dtype=bool)
    result[selected] = mask[pixels[:, 1], pixels[:, 0]]
    return result


def partition_material(snapshot, result, mask, audit):
    """Conservatively withhold material cells intersecting matched image rays.

    Cell center/corner tests avoid retaining robot band votes merely because a
    voxel straddles a matched ray. This is a conservative material-score filter,
    not an exact semantic segmentation or a new scene completion. The original
    visible mesh (including robot/support) and collision object remain intact.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != snapshot.depth.shape:
        raise ValueError("gripper evidence mask must be boolean and match frozen depth")
    audit = dict(audit)
    surface = result.surface_occupancy
    pixels = np.asarray(result.observed_pixel_uv, dtype=np.int64).reshape(-1, 2)
    raw_keep = ~mask[pixels[:, 1], pixels[:, 0]]
    keep = np.ones(len(surface.occupied_indices), dtype=bool)
    if mask.any() and len(keep):
        lower = surface.origin_world + surface.occupied_indices * surface.voxel_size_m
        for offset in ((0.5, 0.5, 0.5), *product((0.0, 1.0), repeat=3)):
            selected = np.flatnonzero(keep)
            if not len(selected):
                break
            withheld = points_on_withheld_rays(
                lower[selected] + np.asarray(offset) * surface.voxel_size_m,
                mask,
                snapshot.intrinsic_matrix,
                snapshot.extrinsic_matrix,
            )
            keep[selected[withheld]] = False
    filtered = (
        VoxelOccupancy(
            voxel_size_m=surface.voxel_size_m,
            origin_world=surface.origin_world,
            occupied_indices=surface.occupied_indices[keep],
            source=surface.source + "_excluding_depth_matched_gripper_rays",
            query_radius_cells=surface.query_radius_cells,
        )
        if not keep.all()
        else surface
    )
    audit.update(
        raw_points_withheld=int((~raw_keep).sum()),
        surface_cells_before=len(keep),
        surface_cells_withheld=int((~keep).sum()),
        surface_cells_after=int(keep.sum()),
        material_filter="voxel_center_or_corner_projects_to_depth_matched_gripper_ray",
        collision_source_retained=result.collision_occupancy.source,
        collision_cells_retained=len(result.collision_occupancy.occupied_indices),
        full_visible_mesh_retained=True,
    )
    metadata = {
        **result.metadata,
        "planner_gripper_material_partition": audit,
        "surface_occupancy": filtered.metadata(),
        "full_view_observed_point_count_before_robot_material_partition": len(raw_keep),
        "observed_point_count": int(raw_keep.sum()),
        "observed_material_evidence_point_count": int(raw_keep.sum()),
    }
    return replace(
        result,
        surface_occupancy=filtered,
        metadata=metadata,
        observed_points_world=result.observed_points_world[raw_keep],
        observed_pixel_uv=pixels[raw_keep],
    )


def exclude_observed_gripper_material(snapshot, result):
    mask, audit = observed_gripper_mask(snapshot)
    return partition_material(snapshot, result, mask, audit), mask
