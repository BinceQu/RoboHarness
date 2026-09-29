"""Whole-view RGB-D reconstruction with separate structural priors.

Version 13 keeps the frozen v12 ordinary-surface path for views without thin
geometry. When thin structures are visible, it reconstructs those structures
with finite-tube ray chords and keeps the rest of the view as a shallow closed
surface shell outside a metric occlusion uncertainty band.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
import trimesh

from .rgbd_scene_mesh_v12 import (
    RGBDSceneMeshResult,
    reconstruct_rgbd_scene_mesh as _v12_reconstruct_rgbd_scene_mesh,
)
from .rgbd_thin_structure_v13 import (
    reconstruct_rgbd_thin_structure_mesh,
)


RGBD_SCENE_MESH_BUILD = "rgbd_whole_view_dual_geometry_priors_v13"


def _thin_occlusion_clearance_mask(
    structure_mask: np.ndarray,
    depth: np.ndarray,
    *,
    focal_px: float,
    clearance_mm: float,
) -> np.ndarray:
    selected = np.asarray(structure_mask, dtype=bool)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    if not selected.any():
        return selected.copy()
    distance_px = ndimage.distance_transform_edt(~selected)
    projected_distance_m = (
        distance_px
        * metric_depth
        / max(float(focal_px), 1e-12)
    )
    valid = np.isfinite(metric_depth) & (metric_depth > 0.0)
    return selected | (
        valid
        & (
            projected_distance_m
            <= float(clearance_mm) / 1000.0
        )
    )


def _with_v13_identity(
    result: RGBDSceneMeshResult,
    *,
    route: str,
) -> RGBDSceneMeshResult:
    return RGBDSceneMeshResult(
        mesh=result.mesh,
        metadata={
            **result.metadata,
            "parent_build": result.metadata.get("build"),
            "build": RGBD_SCENE_MESH_BUILD,
            "method": "whole_view_dual_geometry_prior_router",
            "selected_geometry_route": route,
            "used_click": False,
            "used_segmentation": False,
            "used_object_identity": False,
            "forbidden_inputs_used": [],
        },
    )


def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    **options: Any,
) -> RGBDSceneMeshResult:
    """Reconstruct a whole view using observable local geometry classes."""
    defer_topology_validation = bool(
        options.get("_defer_topology_validation_to_warp_cuda", False)
    )
    selected = str(
        options.get("completion_method", "organized_shell")
    ).strip().lower()
    if selected != "structural_hybrid":
        return _with_v13_identity(
            _v12_reconstruct_rgbd_scene_mesh(rgb, depth, **options),
            route="frozen_v12_requested_completion",
        )

    required_camera = {
        key: options[key]
        for key in (
            "camera_pos",
            "camera_quat_xyzw",
            "focal_length",
            "horizontal_aperture",
        )
    }
    thin = reconstruct_rgbd_thin_structure_mesh(
        rgb,
        depth,
        **required_camera,
    )
    if thin.mesh is None:
        return _with_v13_identity(
            _v12_reconstruct_rgbd_scene_mesh(rgb, depth, **options),
            route="ordinary_surface_v12",
        )

    clearance_mm = float(
        options.pop("structural_thin_clearance_mm", 8.0)
    )
    ordinary_shell_mm = float(
        options.pop("structural_ordinary_shell_mm", 0.1)
    )
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    focal_px = (
        float(required_camera["focal_length"])
        / float(required_camera["horizontal_aperture"])
        * float(metric_depth.shape[1])
    )
    exclusion = _thin_occlusion_clearance_mask(
        thin.mask,
        metric_depth,
        focal_px=focal_px,
        clearance_mm=clearance_mm,
    )
    ordinary_depth = metric_depth.copy()
    ordinary_depth[exclusion] = np.nan
    ordinary_options = dict(options)
    ordinary_options["completion_method"] = "organized_shell"
    ordinary_options["back_extrusion_mm"] = ordinary_shell_mm
    ordinary = _v12_reconstruct_rgbd_scene_mesh(
        rgb,
        ordinary_depth,
        **ordinary_options,
    )

    mesh = trimesh.util.concatenate((ordinary.mesh, thin.mesh))
    if defer_topology_validation:
        component_count = None
        mesh_watertight = None
        topology_validation = "deferred_to_warp_cuda"
    else:
        components = mesh.split(only_watertight=False)
        nonwatertight = sum(
            not bool(component.is_watertight)
            for component in components
        )
        if nonwatertight:
            raise ValueError(
                "v13 dual-prior reconstruction produced an open component"
            )
        component_count = int(len(components))
        mesh_watertight = bool(mesh.is_watertight)
        topology_validation = "trimesh_cpu"
    return RGBDSceneMeshResult(
        mesh=mesh,
        metadata={
            **ordinary.metadata,
            "parent_build": ordinary.metadata.get("build"),
            "build": RGBD_SCENE_MESH_BUILD,
            "method": "whole_view_dual_geometry_prior_router",
            "completion_method": "structural_hybrid",
            "selected_geometry_route": (
                "thin_tube_and_shallow_ordinary_surface"
            ),
            "thin_structure": thin.metadata,
            "thin_structure_detected": True,
            "thin_structure_pixels": int(thin.mask.sum()),
            "thin_occlusion_clearance_mm": clearance_mm,
            "thin_occlusion_excluded_pixels": int(exclusion.sum()),
            "ordinary_surface_prior": "shallow_organized_shell",
            "ordinary_surface_shell_mm": ordinary_shell_mm,
            "ordinary_surface_mesh_vertices": int(
                len(ordinary.mesh.vertices)
            ),
            "ordinary_surface_mesh_faces": int(
                len(ordinary.mesh.faces)
            ),
            "thin_mesh_vertices": int(len(thin.mesh.vertices)),
            "thin_mesh_faces": int(len(thin.mesh.faces)),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_components": component_count,
            "mesh_watertight": mesh_watertight,
            "mesh_topology_validation": topology_validation,
            "allowed_inputs": [
                "rgb",
                "metric_depth",
                "camera_intrinsics",
                "camera_extrinsics",
            ],
            "used_click": False,
            "used_segmentation": False,
            "used_object_identity": False,
            "forbidden_inputs_used": [],
        },
    )
