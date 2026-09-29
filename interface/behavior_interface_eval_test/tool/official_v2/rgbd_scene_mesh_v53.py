"""Representative RGB-D reconstruction with a visibility-safe solid prior.

V53 preserves the V52 expert and frozen V13 baseline, but rejects a planar
prism completion when it broadly occludes projectively observed free space.
The guard consumes only RGB-D and camera calibration.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image
import trimesh

from .rgbd_scene_mesh_v12 import RGBDSceneMeshResult
from . import rgbd_scene_mesh_v52 as _v52


MESH_VERSION = "v53"
RGBD_SCENE_MESH_BUILD = (
    "rgbd_representative_v53_visible_free_space_guard_runtime"
)
V13_RECONSTRUCTION_OVERRIDES: Dict[str, Any] = {
    "structural_planar_prism_visibility_guard": True,
    "structural_planar_prism_free_space_minimum_clearance_mm": 8.0,
    "structural_planar_prism_free_space_clearance_pixels": 3.0,
    "structural_planar_prism_free_space_maximum_violation_fraction": 0.05,
    "structural_planar_prism_free_space_minimum_violation_pixels": 64,
}
V13_RECONSTRUCTION_KWARGS: Dict[str, Any] = {
    **_v52.V13_RECONSTRUCTION_KWARGS,
    **V13_RECONSTRUCTION_OVERRIDES,
}


def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    defer_topology_validation_to_warp_cuda: bool = False,
) -> RGBDSceneMeshResult:
    """Reconstruct a scene with the V53 visible-free-space contract."""
    total_started = time.perf_counter()
    base_started = time.perf_counter()
    base_options = dict(V13_RECONSTRUCTION_KWARGS)
    if defer_topology_validation_to_warp_cuda:
        base_options[
            "_defer_topology_validation_to_warp_cuda"
        ] = True
    base = _v52._reconstruct_v13_scene_mesh(
        rgb,
        depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        **base_options,
    )
    base_elapsed_s = float(time.perf_counter() - base_started)
    camera = {
        "pos": np.asarray(camera_pos, dtype=np.float64).reshape(3).tolist(),
        "quat": np.asarray(
            camera_quat_xyzw,
            dtype=np.float64,
        ).reshape(4).tolist(),
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
    }

    def _workdir():
        # Re-select an existing temp root if an external owner removed its leaf.
        if tempfile.tempdir is not None and not Path(tempfile.tempdir).is_dir():
            tempfile.tempdir = None
        return tempfile.TemporaryDirectory(prefix="official_v2_rgbd_v53_")

    def _apply_v52_expert(
        temporary_dir: str,
    ) -> tuple[trimesh.Trimesh, dict[str, Any]]:
        rows = _v52._reconstruct_candidates(
            rgb=np.asarray(rgb),
            depth=np.asarray(depth),
            camera=camera,
            baseline_mesh=base.mesh,
            baseline_metadata=base.metadata,
            voxel_mm=1.5,
            surface_shell_mm=0.5,
            overlap_voxels=0.0,
            output_dir=Path(temporary_dir),
        )
        selected = next(
            row
            for row in rows
            if row["name"] == "v52_boundary_uncertainty_p90_band"
        )
        loaded = trimesh.load(
            selected["mesh_path"],
            force="mesh",
            process=False,
        )
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError("V53 expert output is not a Trimesh")
        selected_metadata = dict(selected)
        selected_metadata.pop("mesh_path", None)
        return loaded, selected_metadata

    expert_started = time.perf_counter()
    selected_metadata: dict[str, Any] | None = None
    try:
        try:
            with _workdir() as temporary_dir:
                mesh, selected_metadata = _apply_v52_expert(temporary_dir)
        except FileNotFoundError:
            with _workdir() as temporary_dir:
                mesh, selected_metadata = _apply_v52_expert(temporary_dir)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
        StopIteration,
    ) as error:
        mesh = base.mesh.copy()
        representative = {
            "representative_version": MESH_VERSION,
            "effective_mesh_version": "v13_visibility_guarded_fallback",
            "expert_applied": False,
            "applicability": "not_applicable",
            "fallback": "visibility_guarded_v13_mesh",
            "reason": f"{type(error).__name__}: {error}",
        }
    else:
        representative = {
            "representative_version": MESH_VERSION,
            "effective_mesh_version": MESH_VERSION,
            "expert_applied": True,
            "applicability": "applicable",
            "fallback": None,
            "reason": None,
            "selected_candidate": selected_metadata,
        }
    expert_elapsed_s = float(time.perf_counter() - expert_started)
    metadata = {
        **base.metadata,
        "parent_build": base.metadata.get("build"),
        "build": RGBD_SCENE_MESH_BUILD,
        "mesh_version": MESH_VERSION,
        **representative,
        "base_mesh_version": "v13_visibility_guarded",
        "base_reconstruction_build": base.metadata.get("build"),
        "base_reconstruction_kwargs": dict(base_options),
        "v53_parameters": {
            "visibility_guard": dict(V13_RECONSTRUCTION_OVERRIDES),
            "v52_expert_candidate": "v52_boundary_uncertainty_p90_band",
            "voxel_mm": 1.5,
            "surface_shell_mm": 0.5,
            "overlap_voxels": 0.0,
        },
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": (
            None
            if defer_topology_validation_to_warp_cuda
            else bool(mesh.is_watertight)
        ),
        "mesh_topology_validation": (
            "deferred_to_warp_cuda"
            if defer_topology_validation_to_warp_cuda
            else "trimesh_cpu"
        ),
        "timings_s": {
            "v13_visibility_guarded_base": base_elapsed_s,
            "v52_expert": expert_elapsed_s,
            "total": float(time.perf_counter() - total_started),
        },
        "allowed_inputs": [
            "rgb",
            "metric_depth",
            "camera_intrinsics",
            "camera_extrinsics",
        ],
        "used_click": False,
        "used_segmentation": False,
        "used_object_identity": False,
        "used_evaluator_centers": False,
        "used_reference_counts": False,
        "used_gt_mesh": False,
        "forbidden_inputs_used": [],
    }
    return RGBDSceneMeshResult(mesh=mesh, metadata=metadata)


def reconstruct_scene_mesh_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    defer_topology_validation_to_warp_cuda: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a frozen capture and reconstruct one reusable V53 scene mesh."""
    init_dir = str(session.get("init_dir") or "")
    rgb_path = str(
        session.get("rgb_path") or os.path.join(init_dir, "rgb.png")
    )
    depth_path = str(
        session.get("depth_path") or os.path.join(init_dir, "depth.npy")
    )
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"missing frozen RGB: {rgb_path}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"missing frozen depth: {depth_path}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    started = time.perf_counter()
    result = reconstruct_rgbd_scene_mesh(
        rgb,
        depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        defer_topology_validation_to_warp_cuda=bool(
            defer_topology_validation_to_warp_cuda
        ),
    )
    metadata = dict(result.metadata)
    metadata["elapsed_s"] = float(time.perf_counter() - started)
    return result.mesh, rgb, depth, metadata


__all__ = [
    "MESH_VERSION",
    "RGBD_SCENE_MESH_BUILD",
    "V13_RECONSTRUCTION_KWARGS",
    "V13_RECONSTRUCTION_OVERRIDES",
    "reconstruct_rgbd_scene_mesh",
    "reconstruct_scene_mesh_from_session",
]
