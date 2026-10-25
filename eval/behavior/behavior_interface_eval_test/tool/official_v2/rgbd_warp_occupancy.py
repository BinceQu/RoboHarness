"""CUDA occupancy sampling for a closed RGB-D scene mesh.

The planner needs a small regular occupancy grid, not a CPU point-in-mesh
service.  This module uploads the already reconstructed representative mesh to
Warp once and evaluates the complete local grid with a CUDA winding-number
kernel.  Runtime use is fail-closed: CPU execution is available only to unit
tests that opt in explicitly.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import tempfile
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from behavior_interface.runtime_tmp import nvrtc_compile_tmpdir

from .rgbd_projective_occupancy import (
    ProjectiveOccupancyContractError,
    ProjectiveOccupancyUnavailable,
    _hit_contract,
    local_grid_spec,
)


WARP_OCCUPANCY_VERSION = "v53_warp_winding_cuda_v1"
WARP_OCCUPANCY_BUILD = "rgbd_v53_local_warp_winding_occupancy_v1"
DEFAULT_MAX_GRID_VOXELS = 4_000_000
DEFAULT_MAX_MESH_TRIANGLES = 4_000_000
DEFAULT_MIN_FREE_CUDA_MIB = 512
DEFAULT_CACHE_SIZE = 2
DEFAULT_COMPONENT_LABEL_ITERATIONS = 64
DEFAULT_ORIENTATION_REPAIR_ITERATIONS = 2048


@dataclass(frozen=True)
class WarpMeshOccupancyGrid:
    origin: np.ndarray
    occupancy: np.ndarray
    voxel_m: float
    metadata: Dict[str, Any]
    device_dense: Any = None
    device_origin: Any = None
    device_resources: Tuple[Any, ...] = ()


@dataclass
class _CachedWarpMesh:
    source_mesh: Any
    warp_mesh: Any
    device: str
    vertices: int
    triangles: int
    bounds: np.ndarray
    build_elapsed_s: float
    topology_metadata: Dict[str, Any]
    device_resources: Tuple[Any, ...]


_WARP_MESH_CACHE: "OrderedDict[Tuple[Any, ...], _CachedWarpMesh]" = (
    OrderedDict()
)
_WARP_MESH_CACHE_LOCK = threading.Lock()
_WARP_RUNTIME_LOCK = threading.Lock()
_WARP_MODULE = None
_WARP_KERNEL = None


def _ensure_writable_warp_cache() -> str:
    configured = os.environ.get("WARP_CACHE_PATH")
    if configured:
        cache_path = Path(configured).expanduser()
    else:
        appdata = os.environ.get("OMNIGIBSON_APPDATA_PATH")
        if appdata:
            cache_path = Path(appdata) / "cache" / "warp"
        else:
            cache_path = (
                Path(tempfile.gettempdir())
                / f"behavior_eval_warp_{os.getuid()}"
            )
        os.environ["WARP_CACHE_PATH"] = str(cache_path)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        probe = cache_path / f".write_probe_{os.getpid()}"
        probe.touch(exist_ok=False)
        probe.unlink()
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"Warp kernel cache is not writable at {cache_path}: {exc}"
        ) from exc
    return str(cache_path)


def _reset_warp_runtime() -> None:
    global _WARP_MODULE, _WARP_KERNEL
    _WARP_MODULE = None
    _WARP_KERNEL = None


def _warp_runtime():
    global _WARP_MODULE, _WARP_KERNEL
    if _WARP_MODULE is not None and _WARP_KERNEL is not None:
        return _WARP_MODULE, _WARP_KERNEL
    with _WARP_RUNTIME_LOCK:
        if _WARP_MODULE is not None and _WARP_KERNEL is not None:
            return _WARP_MODULE, _WARP_KERNEL
        # NVRTC 编译读 TMPDIR；OG 的 USD 叶子可以很长，这里只切编译窗口。
        try:
            with nvrtc_compile_tmpdir():
                return _load_warp_runtime_locked()
        except Exception:
            _reset_warp_runtime()
            raise


def _load_warp_runtime_locked():
    global _WARP_MODULE, _WARP_KERNEL
    _ensure_writable_warp_cache()
    try:
        import warp as wp
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"Warp is required for CUDA mesh occupancy: {exc}"
        ) from exc

    @wp.kernel
    def fill_regular_grid_winding(
        mesh_id: wp.uint64,
        origin: wp.vec3,
        voxel_m: float,
        shape_y: int,
        shape_z: int,
        bounds_min: wp.vec3,
        bounds_max: wp.vec3,
        maximum_distance: float,
        output: wp.array(dtype=wp.uint8),
    ):
        linear = wp.tid()
        yz = shape_y * shape_z
        ix = linear // yz
        remainder = linear - ix * yz
        iy = remainder // shape_z
        iz = remainder - iy * shape_z
        point = origin + voxel_m * wp.vec3(
            float(ix),
            float(iy),
            float(iz),
        )
        if (
            point[0] < bounds_min[0]
            or point[1] < bounds_min[1]
            or point[2] < bounds_min[2]
            or point[0] > bounds_max[0]
            or point[1] > bounds_max[1]
            or point[2] > bounds_max[2]
        ):
            output[linear] = wp.uint8(0)
            return
        query = wp.mesh_query_point_sign_winding_number(
            mesh_id,
            point,
            maximum_distance,
            2.0,
            0.5,
        )
        if query.result and query.sign < 0.0:
            output[linear] = wp.uint8(1)
        else:
            output[linear] = wp.uint8(0)

    _WARP_MODULE = wp
    _WARP_KERNEL = fill_regular_grid_winding
    return wp, fill_regular_grid_winding


def _runtime_device(
    requested_device: Optional[str],
    *,
    allow_cpu_reference: bool,
):
    wp, kernel = _warp_runtime()
    requested = str(
        requested_device
        or os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", "cuda:0")
    ).strip()
    try:
        device = wp.get_device(requested)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"invalid Warp occupancy device {requested!r}: {exc}"
        ) from exc
    if device.is_cpu:
        if not bool(allow_cpu_reference):
            raise ProjectiveOccupancyUnavailable(
                "CPU Warp occupancy is disabled in runtime; CUDA is required"
            )
    elif not device.is_cuda:
        raise ProjectiveOccupancyUnavailable(
            f"unsupported Warp occupancy device {device}; expected cuda:*"
        )
    try:
        wp.zeros(1, dtype=wp.uint8, device=device)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"cannot initialize Warp occupancy on {device}: {exc}"
        ) from exc
    return wp, kernel, device


def _mesh_arrays(mesh: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.ascontiguousarray(
        np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    )
    faces = np.ascontiguousarray(
        np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    )
    if not len(vertices) or not len(faces):
        raise ProjectiveOccupancyContractError(
            "representative RGB-D mesh is empty"
        )
    if not np.all(np.isfinite(vertices)):
        raise ProjectiveOccupancyContractError(
            "representative RGB-D mesh has non-finite vertices"
        )
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ProjectiveOccupancyContractError(
            "representative RGB-D mesh has invalid face indices"
        )
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0))).astype(
        np.float32,
        copy=False,
    )
    return vertices, faces, bounds


def _mesh_cache_key(
    mesh: Any,
    *,
    device: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    bounds: np.ndarray,
) -> Tuple[Any, ...]:
    return (
        id(mesh),
        str(device),
        int(len(vertices)),
        int(len(faces)),
        bounds.tobytes(),
    )


def _repair_face_winding_torch(
    torch,
    triangles,
    component_labels,
    edge_faces,
    edge_requires_flip,
) -> Tuple[Any, Dict[str, int]]:
    """Solve face-winding parity constraints on the tensor device."""
    failure_count = int(edge_requires_flip.sum().item())
    if failure_count == 0:
        return triangles, {
            "orientation_repair_components": 0,
            "orientation_repair_affected_faces": 0,
            "orientation_repair_faces_flipped": 0,
            "orientation_repair_iterations": 0,
        }

    face_a = edge_faces[:, 0]
    face_b = edge_faces[:, 1]
    bad_roots = torch.unique(
        component_labels[face_a[edge_requires_flip]]
    )
    root_is_affected = torch.zeros(
        len(triangles),
        dtype=torch.bool,
        device=triangles.device,
    )
    root_is_affected[bad_roots] = True
    affected_faces = root_is_affected[component_labels]
    affected_edges = affected_faces[face_a]
    repair_a = face_a[affected_edges]
    repair_b = face_b[affected_edges]
    repair_constraint = edge_requires_flip[affected_edges]

    known = ~affected_faces
    known[bad_roots] = True
    face_flip = torch.zeros(
        len(triangles),
        dtype=torch.bool,
        device=triangles.device,
    )
    directed_source = torch.cat((repair_a, repair_b))
    directed_target = torch.cat((repair_b, repair_a))
    directed_constraint = torch.cat(
        (repair_constraint, repair_constraint)
    )
    maximum_iterations = max(
        1,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_WARP_ORIENTATION_REPAIR_ITERATIONS",
                str(DEFAULT_ORIENTATION_REPAIR_ITERATIONS),
            )
        ),
    )
    repair_iterations = 0
    for repair_iterations in range(1, maximum_iterations + 1):
        usable = known[directed_source]
        source = directed_source[usable]
        target = directed_target[usable]
        proposal = torch.logical_xor(
            face_flip[source],
            directed_constraint[usable],
        ).to(dtype=torch.int8)
        proposal_min = torch.full(
            (len(triangles),),
            2,
            dtype=torch.int8,
            device=triangles.device,
        )
        proposal_max = torch.full(
            (len(triangles),),
            -1,
            dtype=torch.int8,
            device=triangles.device,
        )
        proposal_min.scatter_reduce_(
            0,
            target,
            proposal,
            reduce="amin",
            include_self=True,
        )
        proposal_max.scatter_reduce_(
            0,
            target,
            proposal,
            reduce="amax",
            include_self=True,
        )
        has_proposal = proposal_min != 2
        conflicting_proposals = has_proposal & (
            proposal_min != proposal_max
        )
        conflicting_known = has_proposal & known & (
            proposal_min != face_flip.to(dtype=torch.int8)
        )
        if bool(torch.any(conflicting_proposals | conflicting_known).item()):
            raise ProjectiveOccupancyContractError(
                "representative RGB-D mesh is closed but not consistently "
                "orientable"
            )
        newly_known = has_proposal & ~known
        face_flip[newly_known] = proposal_min[newly_known].to(
            dtype=torch.bool
        )
        known |= newly_known
        if bool(torch.all(known[affected_faces]).item()):
            break
        if not bool(torch.any(newly_known).item()):
            raise ProjectiveOccupancyContractError(
                "face-winding repair could not traverse a closed component"
            )
    else:
        raise ProjectiveOccupancyContractError(
            "CUDA face-winding repair did not converge in "
            f"{maximum_iterations} iterations"
        )

    satisfied = (
        torch.logical_xor(face_flip[face_a], face_flip[face_b])
        == edge_requires_flip
    )
    if not bool(torch.all(satisfied).item()):
        raise ProjectiveOccupancyContractError(
            "face-winding repair left inconsistent shared edges"
        )

    component_face_count = torch.zeros(
        len(triangles),
        dtype=torch.int64,
        device=triangles.device,
    )
    component_flip_count = torch.zeros_like(component_face_count)
    component_face_count.scatter_add_(
        0,
        component_labels,
        torch.ones_like(component_labels),
    )
    component_flip_count.scatter_add_(
        0,
        component_labels,
        face_flip.to(dtype=torch.int64),
    )
    invert_roots = component_flip_count > (
        component_face_count - component_flip_count
    )
    face_flip = torch.logical_xor(
        face_flip,
        invert_roots[component_labels] & affected_faces,
    )
    repaired = triangles.clone()
    repaired[face_flip] = repaired[face_flip][:, (0, 2, 1)]
    return repaired, {
        "orientation_repair_components": int(len(bad_roots)),
        "orientation_repair_affected_faces": int(
            affected_faces.sum().item()
        ),
        "orientation_repair_faces_flipped": int(face_flip.sum().item()),
        "orientation_repair_iterations": int(repair_iterations),
    }


def _repair_face_winding_numpy(
    triangles: np.ndarray,
    component_labels: np.ndarray,
    edge_faces: np.ndarray,
    edge_requires_flip: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Small deterministic CPU reference for winding-repair tests."""
    failures = np.flatnonzero(edge_requires_flip)
    if not len(failures):
        return triangles, {
            "orientation_repair_components": 0,
            "orientation_repair_affected_faces": 0,
            "orientation_repair_faces_flipped": 0,
            "orientation_repair_iterations": 0,
        }

    face_flip = np.zeros(len(triangles), dtype=bool)
    affected_labels = np.unique(
        component_labels[edge_faces[failures, 0]]
    )
    affected_face_count = 0
    maximum_depth = 0
    for component in affected_labels:
        component_faces = np.flatnonzero(component_labels == component)
        affected_face_count += int(len(component_faces))
        component_edges = np.flatnonzero(
            component_labels[edge_faces[:, 0]] == component
        )
        adjacency: Dict[int, list[Tuple[int, bool]]] = {
            int(face): [] for face in component_faces
        }
        for edge_index in component_edges:
            first, second = edge_faces[edge_index]
            constraint = bool(edge_requires_flip[edge_index])
            adjacency[int(first)].append((int(second), constraint))
            adjacency[int(second)].append((int(first), constraint))
        assigned: Dict[int, bool] = {int(component_faces[0]): False}
        depths: Dict[int, int] = {int(component_faces[0]): 0}
        pending = deque((int(component_faces[0]),))
        while pending:
            first = pending.popleft()
            for second, constraint in adjacency[first]:
                proposed = bool(assigned[first] ^ constraint)
                if second in assigned:
                    if assigned[second] != proposed:
                        raise ProjectiveOccupancyContractError(
                            "representative RGB-D mesh is closed but not "
                            "consistently orientable"
                        )
                    continue
                assigned[second] = proposed
                depths[second] = depths[first] + 1
                pending.append(second)
        if len(assigned) != len(component_faces):
            raise ProjectiveOccupancyContractError(
                "face-winding repair could not traverse a closed component"
            )
        values = np.asarray(
            [assigned[int(face)] for face in component_faces],
            dtype=bool,
        )
        if int(values.sum()) > len(values) - int(values.sum()):
            values = ~values
        face_flip[component_faces] = values
        maximum_depth = max(maximum_depth, max(depths.values(), default=0))

    repaired = np.asarray(triangles, dtype=np.int64).copy()
    repaired[face_flip] = repaired[face_flip][:, (0, 2, 1)]
    final_constraint = np.logical_xor(
        face_flip[edge_faces[:, 0]],
        face_flip[edge_faces[:, 1]],
    )
    if not np.array_equal(final_constraint, edge_requires_flip):
        raise ProjectiveOccupancyContractError(
            "face-winding repair left inconsistent shared edges"
        )
    return repaired, {
        "orientation_repair_components": int(len(affected_labels)),
        "orientation_repair_affected_faces": int(affected_face_count),
        "orientation_repair_faces_flipped": int(face_flip.sum()),
        "orientation_repair_iterations": int(maximum_depth + 1),
    }


def _prepare_closed_outward_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str,
) -> Tuple[Any, Dict[str, Any], Any]:
    """Validate closed components and orient every component outwards.

    V53 can contain hundreds of individually closed shells.  Trimesh parity
    queries did not care whether an individual shell was inward-facing, but a
    generalized winding-number query does.  Runtime therefore computes face
    components and their stable signed volumes on CUDA, then flips complete
    inward components before the Warp BVH is built.  No component mesh objects
    are materialized.
    """
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch_device = torch.device(str(device))
            triangles = torch.as_tensor(
                faces,
                dtype=torch.int64,
                device=torch_device,
            )
            degenerate = (
                (triangles[:, 0] == triangles[:, 1])
                | (triangles[:, 1] == triangles[:, 2])
                | (triangles[:, 2] == triangles[:, 0])
            )
            degenerate_count = int(degenerate.sum().item())
            directed = torch.cat(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                ),
                dim=0,
            )
            low = torch.minimum(directed[:, 0], directed[:, 1])
            high = torch.maximum(directed[:, 0], directed[:, 1])
            keys = low * int(len(vertices)) + high
            sorted_keys, order = torch.sort(keys)
            _unique, counts = torch.unique_consecutive(
                sorted_keys,
                return_counts=True,
            )
            incidence_failures = int((counts != 2).sum().item())
            if incidence_failures:
                raise ProjectiveOccupancyContractError(
                    "representative RGB-D mesh failed closed-topology "
                    f"validation: degenerate_faces={degenerate_count}, "
                    f"edge_incidence_failures={incidence_failures}"
                )
            orientation = torch.where(
                directed[:, 0] == low,
                torch.ones_like(low),
                -torch.ones_like(low),
            )[order]
            orientation_sum = orientation.reshape(-1, 2).sum(dim=1)
            orientation_failures = int((orientation_sum != 0).sum().item())
            edge_count = int(len(counts))
            if degenerate_count:
                raise ProjectiveOccupancyContractError(
                    "representative RGB-D mesh failed closed-topology "
                    f"validation: degenerate_faces={degenerate_count}, "
                    "edge_incidence_failures=0, "
                    f"edge_orientation_failures={orientation_failures}"
                )

            sorted_face_ids = torch.arange(
                len(triangles),
                dtype=torch.int64,
                device=torch_device,
            ).repeat(3)[order]
            edge_faces = sorted_face_ids.reshape(-1, 2)
            face_a = edge_faces[:, 0]
            face_b = edge_faces[:, 1]
            component_labels = torch.arange(
                len(triangles),
                dtype=torch.int64,
                device=torch_device,
            )
            converged = False
            maximum_iterations = max(
                1,
                int(
                    os.environ.get(
                        "OFFICIAL_V2_LITE_WARP_COMPONENT_LABEL_ITERATIONS",
                        str(DEFAULT_COMPONENT_LABEL_ITERATIONS),
                    )
                ),
            )
            for component_iterations in range(1, maximum_iterations + 1):
                root_a = component_labels[face_a]
                root_b = component_labels[face_b]
                higher = torch.maximum(root_a, root_b)
                lower = torch.minimum(root_a, root_b)
                updated = component_labels.clone()
                updated.scatter_reduce_(
                    0,
                    higher,
                    lower,
                    reduce="amin",
                    include_self=True,
                )
                updated = updated[updated]
                if bool(torch.equal(updated, component_labels)):
                    component_labels = updated
                    converged = True
                    break
                component_labels = updated
            if not converged:
                raise ProjectiveOccupancyContractError(
                    "CUDA mesh component labeling did not converge in "
                    f"{maximum_iterations} iterations"
                )
            for _ in range(4):
                component_labels = component_labels[component_labels]
            component_roots = torch.unique(component_labels)
            component_count = int(len(component_roots))

            edge_requires_flip = (
                orientation.reshape(-1, 2)[:, 0]
                == orientation.reshape(-1, 2)[:, 1]
            )
            triangles, orientation_repair_metadata = (
                _repair_face_winding_torch(
                    torch,
                    triangles,
                    component_labels,
                    edge_faces,
                    edge_requires_flip,
                )
            )

            points = torch.as_tensor(
                vertices,
                dtype=torch.float32,
                device=torch_device,
            )
            points64 = points.to(dtype=torch.float64)
            p0 = points64[triangles[:, 0]]
            p1 = points64[triangles[:, 1]]
            p2 = points64[triangles[:, 2]]
            component_point_sum = torch.zeros(
                (len(triangles), 3),
                dtype=torch.float64,
                device=torch_device,
            )
            component_point_sum.index_add_(
                0,
                component_labels,
                p0 + p1 + p2,
            )
            component_face_count = torch.zeros(
                len(triangles),
                dtype=torch.float64,
                device=torch_device,
            )
            component_face_count.scatter_add_(
                0,
                component_labels,
                torch.ones(
                    len(triangles),
                    dtype=torch.float64,
                    device=torch_device,
                ),
            )
            references = component_point_sum / torch.clamp(
                3.0 * component_face_count[:, None],
                min=1.0,
            )
            reference = references[component_labels]
            signed_volume6_per_face = torch.sum(
                (p0 - reference)
                * torch.cross(
                    p1 - reference,
                    p2 - reference,
                    dim=1,
                ),
                dim=1,
            )
            component_volume6 = torch.zeros(
                len(triangles),
                dtype=torch.float64,
                device=torch_device,
            )
            component_volume6.scatter_add_(
                0,
                component_labels,
                signed_volume6_per_face,
            )
            root_volumes = component_volume6[component_roots] / 6.0
            negative_roots = root_volumes < 0.0
            negative_components = int(negative_roots.sum().item())
            flip_faces = component_volume6[component_labels] < 0.0
            flipped_faces = int(flip_faces.sum().item())
            outward_faces = triangles.clone()
            swapped = outward_faces[flip_faces][:, (0, 2, 1)]
            outward_faces[flip_faces] = swapped
            outward_faces = outward_faces.to(dtype=torch.int32).contiguous()
            minimum_abs_component_volume = float(
                torch.min(torch.abs(root_volumes)).item()
            )
            torch.cuda.synchronize(torch_device)
        except Exception as exc:
            if isinstance(exc, ProjectiveOccupancyContractError):
                raise
            raise ProjectiveOccupancyUnavailable(
                f"CUDA mesh topology validation failed on {device}: {exc}"
            ) from exc
        validation_device = str(torch_device)
        prepared_vertices = points
    else:
        triangles = np.asarray(faces, dtype=np.int64)
        degenerate_count = int(
            (
                (triangles[:, 0] == triangles[:, 1])
                | (triangles[:, 1] == triangles[:, 2])
                | (triangles[:, 2] == triangles[:, 0])
            ).sum()
        )
        directed = np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            ),
            axis=0,
        )
        low = np.minimum(directed[:, 0], directed[:, 1])
        high = np.maximum(directed[:, 0], directed[:, 1])
        keys = low * int(len(vertices)) + high
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(
            np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
        )
        stops = np.r_[starts[1:], len(sorted_keys)]
        counts = stops - starts
        incidence_failures = int(np.count_nonzero(counts != 2))
        if incidence_failures:
            raise ProjectiveOccupancyContractError(
                "representative RGB-D mesh failed closed-topology "
                f"validation: degenerate_faces={degenerate_count}, "
                f"edge_incidence_failures={incidence_failures}"
            )
        orientation = np.where(directed[:, 0] == low, 1, -1)[order]
        orientation_sum = np.add.reduceat(orientation, starts)
        orientation_failures = int(np.count_nonzero(orientation_sum != 0))
        edge_count = int(len(counts))
        if degenerate_count:
            raise ProjectiveOccupancyContractError(
                "representative RGB-D mesh failed closed-topology "
                f"validation: degenerate_faces={degenerate_count}, "
                "edge_incidence_failures=0, "
                f"edge_orientation_failures={orientation_failures}"
            )

        sorted_face_ids = np.tile(
            np.arange(len(triangles), dtype=np.int64),
            3,
        )[order]
        edge_faces = sorted_face_ids.reshape(-1, 2)
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            adjacency = coo_matrix(
                (
                    np.ones(2 * len(edge_faces), dtype=np.uint8),
                    (
                        np.concatenate(
                            (edge_faces[:, 0], edge_faces[:, 1])
                        ),
                        np.concatenate(
                            (edge_faces[:, 1], edge_faces[:, 0])
                        ),
                    ),
                ),
                shape=(len(triangles), len(triangles)),
            ).tocsr()
            component_count, component_labels = connected_components(
                adjacency,
                directed=False,
                return_labels=True,
            )
        except Exception as exc:
            raise ProjectiveOccupancyUnavailable(
                f"CPU topology reference requires scipy: {exc}"
            ) from exc
        component_iterations = 0
        edge_requires_flip = (
            orientation.reshape(-1, 2)[:, 0]
            == orientation.reshape(-1, 2)[:, 1]
        )
        triangles, orientation_repair_metadata = _repair_face_winding_numpy(
            triangles,
            component_labels,
            edge_faces,
            edge_requires_flip,
        )
        points64 = np.asarray(vertices, dtype=np.float64)
        p0 = points64[triangles[:, 0]]
        p1 = points64[triangles[:, 1]]
        p2 = points64[triangles[:, 2]]
        component_point_sum = np.zeros(
            (component_count, 3),
            dtype=np.float64,
        )
        np.add.at(
            component_point_sum,
            component_labels,
            p0 + p1 + p2,
        )
        component_face_count = np.bincount(
            component_labels,
            minlength=component_count,
        ).astype(np.float64)
        references = component_point_sum / (
            3.0 * component_face_count[:, None]
        )
        reference = references[component_labels]
        signed_volume6_per_face = np.einsum(
            "ij,ij->i",
            p0 - reference,
            np.cross(p1 - reference, p2 - reference),
        )
        component_volume6 = np.zeros(component_count, dtype=np.float64)
        np.add.at(
            component_volume6,
            component_labels,
            signed_volume6_per_face,
        )
        root_volumes = component_volume6 / 6.0
        negative_components = int(np.count_nonzero(root_volumes < 0.0))
        flip_faces = component_volume6[component_labels] < 0.0
        flipped_faces = int(np.count_nonzero(flip_faces))
        outward_faces = triangles.copy()
        outward_faces[flip_faces] = outward_faces[flip_faces][:, (0, 2, 1)]
        outward_faces = np.ascontiguousarray(outward_faces, dtype=np.int32)
        minimum_abs_component_volume = float(
            np.min(np.abs(root_volumes))
        )
        validation_device = "numpy_cpu_test_reference"
        prepared_vertices = vertices
    metadata = {
        "topology_validation": (
            "closed_edge_incidence_gpu_component_outward_orientation"
        ),
        "topology_validation_device": validation_device,
        "topology_validation_elapsed_s": float(
            time.perf_counter() - started
        ),
        "topology_edge_count": edge_count,
        "topology_degenerate_faces": degenerate_count,
        "topology_edge_incidence_failures": incidence_failures,
        "topology_edge_orientation_failures": orientation_failures,
        "topology_edge_orientation_failure_fraction": float(
            orientation_failures / max(edge_count, 1)
        ),
        "topology_edge_orientation_failures_after_repair": 0,
        **{
            f"topology_{key}": value
            for key, value in orientation_repair_metadata.items()
        },
        "topology_component_count": int(component_count),
        "topology_component_label_iterations": int(component_iterations),
        "topology_inward_components_flipped": int(negative_components),
        "topology_faces_flipped": int(flipped_faces),
        "topology_min_abs_component_volume_m3": (
            minimum_abs_component_volume
        ),
    }
    return outward_faces, metadata, prepared_vertices


def _validate_closed_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str,
) -> Dict[str, Any]:
    """Compatibility wrapper used by offline topology audits."""
    _prepared, metadata, _vertices = _prepare_closed_outward_topology(
        vertices,
        faces,
        device=device,
    )
    return metadata


def _cached_warp_mesh(
    mesh: Any,
    *,
    wp,
    device,
    max_mesh_triangles: int,
) -> Tuple[_CachedWarpMesh, bool]:
    vertices, faces, bounds = _mesh_arrays(mesh)
    if len(faces) > int(max_mesh_triangles):
        raise ProjectiveOccupancyContractError(
            f"representative mesh has {len(faces)} triangles, limit is "
            f"{int(max_mesh_triangles)}"
        )
    key = _mesh_cache_key(
        mesh,
        device=str(device),
        vertices=vertices,
        faces=faces,
        bounds=bounds,
    )
    limit = max(
        1,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_WARP_MESH_CACHE_SIZE",
                str(DEFAULT_CACHE_SIZE),
            )
        ),
    )
    with _WARP_MESH_CACHE_LOCK:
        cached = _WARP_MESH_CACHE.get(key)
        if cached is not None and cached.source_mesh is mesh:
            _WARP_MESH_CACHE.move_to_end(key)
            return cached, True

        started = time.perf_counter()
        try:
            prepared_faces, topology_metadata, prepared_vertices = (
                _prepare_closed_outward_topology(
                    vertices,
                    faces,
                    device=str(device),
                )
            )
            if device.is_cuda:
                points = wp.from_torch(
                    prepared_vertices,
                    dtype=wp.vec3,
                )
                indices = wp.from_torch(
                    prepared_faces.reshape(-1),
                    dtype=wp.int32,
                )
                device_resources = (prepared_vertices, prepared_faces)
            else:
                points = wp.array(
                    prepared_vertices,
                    dtype=wp.vec3,
                    device=device,
                )
                indices = wp.array(
                    prepared_faces.reshape(-1),
                    dtype=wp.int32,
                    device=device,
                )
                device_resources = ()
            warp_mesh = wp.Mesh(
                points=points,
                indices=indices,
                support_winding_number=True,
                bvh_constructor="lbvh" if device.is_cuda else "sah",
            )
            wp.synchronize_device(device)
        except ProjectiveOccupancyContractError:
            raise
        except Exception as exc:
            raise ProjectiveOccupancyUnavailable(
                f"failed to build Warp mesh on {device}: {exc}"
            ) from exc
        cached = _CachedWarpMesh(
            source_mesh=mesh,
            warp_mesh=warp_mesh,
            device=str(device),
            vertices=int(len(vertices)),
            triangles=int(len(faces)),
            bounds=bounds.copy(),
            build_elapsed_s=float(time.perf_counter() - started),
            topology_metadata=topology_metadata,
            device_resources=device_resources,
        )
        _WARP_MESH_CACHE[key] = cached
        _WARP_MESH_CACHE.move_to_end(key)
        while len(_WARP_MESH_CACHE) > limit:
            _WARP_MESH_CACHE.popitem(last=False)
        return cached, False


def clear_warp_mesh_cache() -> None:
    with _WARP_MESH_CACHE_LOCK:
        _WARP_MESH_CACHE.clear()


def _cuda_memory_preflight(
    *,
    device: str,
    vertices: int,
    triangles: int,
    grid_voxels: int,
) -> Dict[str, Optional[float]]:
    if not str(device).startswith("cuda"):
        return {
            "cuda_free_mib_before": None,
            "cuda_total_mib": None,
            "estimated_required_mib": None,
        }
    try:
        import torch

        torch_device = torch.device(str(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch_device)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"cannot query CUDA memory for Warp occupancy on {device}: {exc}"
        ) from exc
    reserve_mib = max(
        0,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MIN_FREE_MIB",
                str(DEFAULT_MIN_FREE_CUDA_MIB),
            )
        ),
    )
    # Vertex/triangle buffers, LBVH, fast-winding aggregates, output, and a
    # conservative allocator multiplier.  This is intentionally an upper
    # bound so an uncertain allocation fails before Warp starts building.
    estimated_bytes = int(
        vertices * 128
        + triangles * 384
        + grid_voxels * 8
        + 128 * 1024**2
    )
    if int(free_bytes) - reserve_mib * 1024**2 < estimated_bytes:
        raise ProjectiveOccupancyUnavailable(
            f"insufficient CUDA memory for Warp occupancy on {device}: "
            f"{free_bytes / 1024**2:.1f} MiB free, "
            f"{estimated_bytes / 1024**2:.1f} MiB estimated, "
            f"{reserve_mib} MiB reserved"
        )
    return {
        "cuda_free_mib_before": float(free_bytes / 1024**2),
        "cuda_total_mib": float(total_bytes / 1024**2),
        "estimated_required_mib": float(estimated_bytes / 1024**2),
    }


def build_warp_local_occupancy(
    mesh: Any,
    *,
    hit: Sequence[float],
    query_offsets: Iterable[np.ndarray],
    axial_z_m: np.ndarray,
    anchor_radius_m: float,
    voxel_m: float,
    ctx=None,
    requested_device: Optional[str] = None,
    allow_cpu_reference: bool = False,
    max_grid_voxels: Optional[int] = None,
    max_mesh_triangles: Optional[int] = None,
    validate_hit: bool = True,
) -> WarpMeshOccupancyGrid:
    """Evaluate exact local V53 occupancy on Warp CUDA."""
    started = time.perf_counter()
    query_parts = [
        np.asarray(part, dtype=np.float64).reshape(-1, 3)
        for part in query_offsets
    ]
    origin, upper, shape, radius = local_grid_spec(
        hit=hit,
        query_offsets=query_parts,
        axial_z_m=np.asarray(axial_z_m, dtype=np.float64),
        anchor_radius_m=float(anchor_radius_m),
        voxel_m=float(voxel_m),
    )
    total = int(np.prod(shape, dtype=np.int64))
    maximum_grid = int(
        max_grid_voxels
        if max_grid_voxels is not None
        else os.environ.get(
            "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MAX_VOXELS",
            str(DEFAULT_MAX_GRID_VOXELS),
        )
    )
    if maximum_grid <= 0 or total > maximum_grid:
        raise ProjectiveOccupancyContractError(
            f"local occupancy grid has {total} voxels, limit is "
            f"{maximum_grid}"
        )
    maximum_triangles = int(
        max_mesh_triangles
        if max_mesh_triangles is not None
        else os.environ.get(
            "OFFICIAL_V2_LITE_WARP_MAX_MESH_TRIANGLES",
            str(DEFAULT_MAX_MESH_TRIANGLES),
        )
    )
    if maximum_triangles <= 0:
        raise ValueError("max_mesh_triangles must be positive")

    wp, kernel, device = _runtime_device(
        requested_device,
        allow_cpu_reference=bool(allow_cpu_reference),
    )
    vertices, faces, _bounds = _mesh_arrays(mesh)
    memory = _cuda_memory_preflight(
        device=str(device),
        vertices=int(len(vertices)),
        triangles=int(len(faces)),
        grid_voxels=total,
    )
    cached, cache_hit = _cached_warp_mesh(
        mesh,
        wp=wp,
        device=device,
        max_mesh_triangles=maximum_triangles,
    )
    bounds = np.asarray(cached.bounds, dtype=np.float32)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    maximum_distance = max(diagonal + float(radius), float(voxel_m) * 4.0)
    query_started = time.perf_counter()
    try:
        with nvrtc_compile_tmpdir():
            output = wp.zeros(total, dtype=wp.uint8, device=device)
            wp.launch(
                kernel=kernel,
                dim=total,
                inputs=[
                    cached.warp_mesh.id,
                    wp.vec3(*np.asarray(origin, dtype=np.float32).tolist()),
                    float(voxel_m),
                    int(shape[1]),
                    int(shape[2]),
                    wp.vec3(*bounds[0].tolist()),
                    wp.vec3(*bounds[1].tolist()),
                    float(maximum_distance),
                    output,
                ],
                device=device,
            )
            wp.synchronize_device(device)
        device_dense = None
        device_origin = None
        device_resources: Tuple[Any, ...] = ()
        if device.is_cuda:
            try:
                import torch

                device_dense = wp.to_torch(output).reshape(-1).to(
                    dtype=torch.bool
                )
                device_origin = torch.as_tensor(
                    np.asarray(origin, dtype=np.float32),
                    dtype=torch.float32,
                    device=torch.device(str(device)),
                )
                device_resources = (output, device_dense, device_origin)
            except Exception as exc:
                raise ProjectiveOccupancyUnavailable(
                    "failed to retain Warp occupancy on CUDA for pose "
                    f"counting: {exc}"
                ) from exc
        occupancy = output.numpy().reshape(tuple(int(v) for v in shape)) > 0
    except Exception as exc:
        _reset_warp_runtime()
        raise ProjectiveOccupancyUnavailable(
            f"Warp winding occupancy failed on {device}: {exc}"
        ) from exc
    query_elapsed_s = float(time.perf_counter() - query_started)

    hit_array = np.asarray(hit, dtype=np.float64).reshape(3)
    maximum_hit_distance_m = max(float(voxel_m) * 2.5, 0.0075)
    hit_audit = _hit_contract(
        occupancy,
        origin=np.asarray(origin, dtype=np.float64),
        hit=hit_array,
        voxel_m=float(voxel_m),
        maximum_distance_m=maximum_hit_distance_m,
    )
    if validate_hit and not hit_audit["ok"]:
        raise ProjectiveOccupancyContractError(
            "clicked RGB-D surface is missing from V53 Warp occupancy "
            f"(nearest={hit_audit['nearest_occupied_distance_mm']}mm, "
            f"limit={hit_audit['maximum_distance_mm']:.3f}mm)"
        )

    metadata = {
        "geometry_version": WARP_OCCUPANCY_VERSION,
        "mesh_version": "v53",
        "occupancy_build": WARP_OCCUPANCY_BUILD,
        "occupancy_method": (
            "warp_cuda_fast_winding_number"
            if device.is_cuda
            else "warp_cpu_fast_winding_number_reference"
        ),
        "occupancy_device": str(device),
        "pose_count_device_required": str(device) if device.is_cuda else None,
        "pose_count_device_tensor_ready": bool(device_dense is not None),
        "cpu_runtime_fallback_allowed": bool(allow_cpu_reference),
        "occupancy_query_bounds": np.stack((origin, upper)).tolist(),
        "origin": np.asarray(origin, dtype=np.float64).tolist(),
        "shape": [int(value) for value in shape],
        "grid_voxels": int(total),
        "occupied_voxels": int(occupancy.sum()),
        "candidate_envelope_radius_m": float(radius),
        "voxel_m": float(voxel_m),
        "mesh_vertices": int(cached.vertices),
        "mesh_triangles": int(cached.triangles),
        "mesh_bounds": bounds.tolist(),
        "mesh_bvh_cache_hit": bool(cache_hit),
        "mesh_bvh_build_elapsed_s": (
            0.0 if cache_hit else float(cached.build_elapsed_s)
        ),
        "query_elapsed_s": query_elapsed_s,
        "elapsed_s": float(time.perf_counter() - started),
        "mesh_split_calls": 0,
        "trimesh_contains_calls": 0,
        "pyembree_used": False,
        "winding_accuracy": 2.0,
        "winding_inside_threshold": 0.5,
        "hit_contract": hit_audit,
        **cached.topology_metadata,
        **memory,
    }
    if ctx is not None and hasattr(ctx, "log"):
        ctx.log(
            "  [grasp_point_filter_rgbd] occupancy 3mm "
            f"shape={metadata['shape']} "
            f"occupied={metadata['occupied_voxels']}/{total} "
            f"method={metadata['occupancy_method']} "
            f"device={device} bvh_cache={cache_hit} "
            f"elapsed={metadata['elapsed_s']:.3f}s"
        )
    return WarpMeshOccupancyGrid(
        origin=np.asarray(origin, dtype=np.float64),
        occupancy=np.asarray(occupancy, dtype=bool),
        voxel_m=float(voxel_m),
        metadata=metadata,
        device_dense=device_dense,
        device_origin=device_origin,
        device_resources=device_resources,
    )


__all__ = [
    "WARP_OCCUPANCY_BUILD",
    "WARP_OCCUPANCY_VERSION",
    "WarpMeshOccupancyGrid",
    "build_warp_local_occupancy",
    "clear_warp_mesh_cache",
]
