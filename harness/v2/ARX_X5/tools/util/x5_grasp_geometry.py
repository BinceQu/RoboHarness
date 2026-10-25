"""Frozen official X5 gripper geometry used by the RGB-D lite planner."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ASSET_PATH = (
    Path(__file__).resolve().parent / "assets" / ("x5_gripper_geometry_open_3mm.npz")
)
VOXEL_M = 0.003
ROBODOJO_X5_GRIPPER_SCALE_M = (-0.01, 0.044)


def _readonly(array: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def x5_gripper_geometry() -> dict[str, Any]:
    """Load and validate the policy-local asset without simulator imports."""
    if not ASSET_PATH.is_file():
        raise FileNotFoundError(
            f"missing frozen official X5 gripper geometry: {ASSET_PATH}"
        )
    with np.load(ASSET_PATH, allow_pickle=False) as asset:
        metadata = json.loads(str(asset["metadata_json"]))
        result = {
            "gripper_voxels": _readonly(asset["gripper_voxels"], np.float64),
            "opening_voxels": _readonly(asset["opening_voxels"], np.float64),
            "opening_columns": _readonly(asset["opening_columns"], np.int64),
            "fixed_solid_voxels": _readonly(asset["fixed_solid_voxels"], np.float64),
            "finger_positive_voxels": _readonly(
                asset["finger_positive_voxels"], np.float64
            ),
            "finger_negative_voxels": _readonly(
                asset["finger_negative_voxels"], np.float64
            ),
            "inflated_gripper_voxels": _readonly(
                asset["inflated_gripper_voxels"], np.float64
            ),
            "axial_x_m": _readonly(asset["axial_x_m"], np.float64),
            "visual_vertices": _readonly(asset["visual_vertices"], np.float64),
            "visual_faces": _readonly(asset["visual_faces"], np.int64),
            "visual_face_component": _readonly(
                asset["visual_face_component"], np.int64
            ),
            "exact_opening_mesh_vertices": _readonly(
                asset["exact_opening_mesh_vertices"], np.float64
            ),
            "exact_opening_mesh_faces": _readonly(
                asset["exact_opening_mesh_faces"], np.int64
            ),
            "exact_opening_mesh_vertex_side": _readonly(
                asset["exact_opening_mesh_vertex_side"]
                if "exact_opening_mesh_vertex_side" in asset.files
                else np.where(
                    np.asarray(asset["exact_opening_mesh_vertices"])[:, 1] >= 0.0,
                    1,
                    -1,
                ),
                np.int8,
            ),
            "component_names": tuple(
                str(value) for value in asset["component_names"].tolist()
            ),
            "metadata": metadata,
        }
    if metadata.get("schema") != "robodojo_x5_gripper_geometry":
        raise ValueError("frozen X5 gripper geometry has an unexpected schema")
    if int(metadata.get("schema_version", 0)) < 3:
        raise ValueError(
            "frozen X5 gripper geometry does not contain the two-finger corridor"
        )
    if metadata.get("opening_geometry") != "two_finger_inner_surface_corridor":
        raise ValueError(
            "frozen X5 gripper opening is not the two-finger inner-surface corridor"
        )
    if metadata.get("exact_opening_component_selection") != (
        "fingertip_connected_common_xz_projection_component"
    ):
        raise ValueError(
            "frozen X5 exact opening does not identify its tip-connected component"
        )
    if metadata.get("exact_opening_metric_independent_of_voxel") is not True:
        raise ValueError(
            "frozen X5 exact opening must remain independent of voxel metrics"
        )
    if abs(float(metadata.get("voxel_m", 0.0)) - VOXEL_M) > 1e-9:
        raise ValueError("frozen X5 gripper geometry is not on the 3 mm grid")
    if metadata.get("eef_link") != "link6":
        raise ValueError("frozen X5 gripper geometry is not expressed at link6")
    if metadata.get("approach_axis_local") != [1.0, 0.0, 0.0]:
        raise ValueError("frozen X5 gripper geometry has the wrong approach axis")
    for key in (
        "gripper_voxels",
        "opening_voxels",
        "fixed_solid_voxels",
        "finger_positive_voxels",
        "finger_negative_voxels",
        "inflated_gripper_voxels",
        "visual_vertices",
    ):
        values = result[key]
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid frozen X5 geometry array {key}")
    columns = result["opening_columns"]
    if (
        columns.ndim != 2
        or columns.shape[1] != 4
        or len(columns) == 0
        or not np.all(np.isfinite(columns))
        or np.any(columns[:, 2] >= columns[:, 3])
    ):
        raise ValueError("invalid frozen X5 opening column boundaries")
    if len(np.unique(columns[:, :2], axis=0)) != len(columns):
        raise ValueError("frozen X5 opening columns contain duplicate X-Z keys")
    if int(metadata.get("opening_column_count", -1)) != len(columns):
        raise ValueError("frozen X5 opening column metadata does not match the asset")
    opening_keys = np.unique(
        np.rint(result["opening_voxels"] / VOXEL_M).astype(np.int64)[:, [0, 2]],
        axis=0,
    )
    if not set(map(tuple, columns[:, :2])).issubset(set(map(tuple, opening_keys))):
        raise ValueError("frozen X5 opening columns have no corresponding voxels")
    faces = result["visual_faces"]
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or int(faces.min()) < 0
        or int(faces.max()) >= len(result["visual_vertices"])
    ):
        raise ValueError("invalid frozen X5 visual mesh faces")
    if len(result["visual_face_component"]) != len(faces):
        raise ValueError("X5 visual face components do not match the mesh")
    exact_vertices = result["exact_opening_mesh_vertices"]
    exact_faces = result["exact_opening_mesh_faces"]
    if (
        exact_vertices.ndim != 2
        or exact_vertices.shape[1] != 3
        or len(exact_vertices) == 0
        or not np.all(np.isfinite(exact_vertices))
        or exact_faces.ndim != 2
        or exact_faces.shape[1] != 3
        or len(exact_faces) == 0
        or int(exact_faces.min()) < 0
        or int(exact_faces.max()) >= len(exact_vertices)
    ):
        raise ValueError("invalid frozen X5 exact opening mesh")
    vertex_side = result["exact_opening_mesh_vertex_side"]
    if (
        vertex_side.ndim != 1
        or len(vertex_side) != len(exact_vertices)
        or not np.all(np.isin(vertex_side, (-1, 1)))
    ):
        raise ValueError("invalid frozen X5 exact opening vertex-side labels")
    # The asset is generated as a closed triangle-surface result.  Check its
    # signed volume without importing trimesh in the evaluator process.
    exact_triangles = exact_vertices[exact_faces]
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            exact_triangles[:, 0],
            np.cross(exact_triangles[:, 1], exact_triangles[:, 2]),
        ).sum()
        / 6.0
    )
    if not np.isfinite(signed_volume) or signed_volume <= 0.0:
        raise ValueError("frozen X5 exact opening mesh has no positive volume")
    expected_exact_volume = float(metadata.get("exact_opening_mesh_volume_cm3", 0.0))
    if expected_exact_volume <= 0.0 or abs(
        signed_volume * 1e6 - expected_exact_volume
    ) > max(1e-3, expected_exact_volume * 1e-5):
        raise ValueError("X5 exact opening mesh volume metadata is inconsistent")
    if metadata.get("exact_opening_geometry") != (
        "variable_free_volume_between_official_link7_link8_inner_triangle_surfaces"
    ):
        raise ValueError("X5 exact opening is not bounded by variable inner triangles")
    if metadata.get("exact_opening_generation") != (
        "xz_triangle_arrangement_inner_surface_ray_intervals_closed_mesh"
    ):
        raise ValueError(
            "X5 exact opening generation is not the frozen triangle arrangement"
        )
    integrated_exact_volume = float(
        metadata.get("exact_opening_integrated_volume_cm3", 0.0)
    )
    if integrated_exact_volume <= 0.0 or abs(
        integrated_exact_volume - expected_exact_volume
    ) > max(1e-3, expected_exact_volume * 2e-7):
        raise ValueError("X5 exact opening integral metadata is inconsistent")
    arrangement_grid = float(metadata.get("exact_opening_arrangement_grid_m", 0.0))
    if not np.isfinite(arrangement_grid) or not (
        0.0 < arrangement_grid < VOXEL_M * 1e-3
    ):
        raise ValueError("X5 exact opening arrangement grid is not sub-voxel")
    if len(result["axial_x_m"]) != 7:
        raise ValueError("X5 frozen fingertip corridor must have seven samples")
    exact_y_bounds = np.asarray(
        metadata.get("exact_opening_inner_y_bounds_reference_m", []),
        dtype=np.float64,
    ).reshape(-1)
    if (
        exact_y_bounds.size != 2
        or not np.all(np.isfinite(exact_y_bounds))
        or exact_y_bounds[0] >= exact_y_bounds[1]
        or float(exact_vertices[:, 1].min()) < exact_y_bounds[0] - 1e-7
        or float(exact_vertices[:, 1].max()) > exact_y_bounds[1] + 1e-7
    ):
        raise ValueError("X5 exact opening mesh exceeds its inner jaw planes")
    # The lookup is deliberately kept private.  It makes wrist-image point
    # classification vectorizable without exposing another public API surface.
    result["_opening_column_lookup"] = {
        (int(ix), int(iz)): (int(inner_lo), int(inner_hi))
        for ix, iz, inner_lo, inner_hi in columns
    }
    result["_fixed_solid_keys"] = frozenset(
        map(
            tuple,
            np.rint(result["fixed_solid_voxels"] / VOXEL_M).astype(np.int64),
        )
    )
    result["_finger_positive_keys"] = np.rint(
        result["finger_positive_voxels"] / VOXEL_M
    ).astype(np.int64)
    result["_finger_negative_keys"] = np.rint(
        result["finger_negative_voxels"] / VOXEL_M
    ).astype(np.int64)
    return result


def x5_gripper_command_fraction(gripper_qpos: Any) -> float:
    """Return RoboDojo's normalized X5 gripper command in [0, 1]."""
    values = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.isfinite(values[0]):
        raise ValueError("X5 gripper_qpos must contain one finite normalized value")
    return float(np.clip(values[0], 0.0, 1.0))


def x5_gripper_joint_position_m(gripper_qpos: Any) -> float:
    """Map normalized proprioception to the URDF-limited finger joint position."""
    command = x5_gripper_command_fraction(gripper_qpos)
    scale_low, scale_high = ROBODOJO_X5_GRIPPER_SCALE_M
    commanded_m = scale_low + command * (scale_high - scale_low)
    return float(np.clip(commanded_m, 0.0, scale_high))


def x5_gripper_command_from_joint_position_m(position_m: Any) -> float:
    """Return the normalized command whose limited joint position is ``position_m``."""
    value = float(np.asarray(position_m, dtype=np.float64).reshape(()))
    if not math.isfinite(value):
        raise ValueError("X5 gripper joint position must be finite")
    scale_low, scale_high = ROBODOJO_X5_GRIPPER_SCALE_M
    limited = float(np.clip(value, 0.0, scale_high))
    return float((limited - scale_low) / (scale_high - scale_low))


def x5_gripper_open_fraction(gripper_qpos: Any) -> float:
    """Return the physical X5 opening fraction after RoboDojo/URDF limits."""
    _scale_low, scale_high = ROBODOJO_X5_GRIPPER_SCALE_M
    return x5_gripper_joint_position_m(gripper_qpos) / scale_high


def x5_legacy_opening_bounds(gripper_qpos: Any) -> np.ndarray:
    """Return a conservative AABB envelope for the current grasp corridor.

    The actual grasp volume is the tapered two-finger corridor represented by
    ``opening_columns``.  This legacy-compatible envelope is retained for
    metadata and callers that need a quick extent, but is never used as the
    volume membership test.
    """
    geometry = x5_gripper_geometry()
    bounds = np.asarray(
        geometry["metadata"]["opening_bounds_eef_m"], dtype=np.float64
    ).copy()
    finger_travel = float(geometry["metadata"]["gripper_open_m"])
    closure = finger_travel - x5_gripper_joint_position_m(gripper_qpos)
    bounds[0, 1] += closure
    bounds[1, 1] -= closure
    return bounds


def _x5_dynamic_solid_keys(
    geometry: dict[str, Any],
    closure_m: float,
) -> set[tuple[int, int, int]]:
    """Return fixed and translated finger solids on the 3 mm grid."""
    closure_cells = float(closure_m) / float(VOXEL_M)
    positive_keys = np.asarray(geometry["_finger_positive_keys"], dtype=np.int64)
    negative_keys = np.asarray(geometry["_finger_negative_keys"], dtype=np.int64)
    shifted_positive = positive_keys.copy()
    shifted_positive[:, 1] = np.rint(
        shifted_positive[:, 1].astype(np.float64) - closure_cells
    ).astype(np.int64)
    shifted_negative = negative_keys.copy()
    shifted_negative[:, 1] = np.rint(
        shifted_negative[:, 1].astype(np.float64) + closure_cells
    ).astype(np.int64)
    solids = set(geometry["_fixed_solid_keys"])
    solids.update(map(tuple, shifted_positive))
    solids.update(map(tuple, shifted_negative))
    return solids


def x5_legacy_opening_voxels(gripper_qpos: Any) -> np.ndarray:
    """Return current opening voxel centers between the two moving inner walls."""
    geometry = x5_gripper_geometry()
    command = x5_gripper_command_fraction(gripper_qpos)
    joint_position = x5_gripper_joint_position_m([command])
    finger_travel = float(geometry["metadata"]["gripper_open_m"])
    closure = max(0.0, finger_travel - joint_position)
    # Preserve the frozen asset exactly at its reference configuration.  Apart
    # from avoiding floating-point edge effects, this keeps planner/render
    # counts reproducible across processes.
    if closure <= 1e-12:
        return np.asarray(geometry["opening_voxels"], dtype=np.float64).copy()
    bounds = x5_legacy_opening_bounds(gripper_qpos)
    if bounds[0, 1] > bounds[1, 1]:
        return np.zeros((0, 3), dtype=np.float64)
    columns = np.asarray(geometry["opening_columns"], dtype=np.int64)
    pitch = float(VOXEL_M)
    # Work in integer voxel coordinates.  Using meter values here can move a
    # boundary by one cell when (for example) -0.048 / 0.003 is represented as
    # -16.000000000000004.
    closure_cells = closure / pitch
    y_low = int(math.ceil(bounds[0, 1] / pitch - 1e-9))
    y_high = int(math.floor(bounds[1, 1] / pitch + 1e-9))
    solid_keys = _x5_dynamic_solid_keys(geometry, closure)
    keys: list[tuple[int, int, int]] = []
    for ix, iz, inner_lo, inner_hi in columns:
        low_key = max(
            y_low,
            int(math.floor(float(inner_lo) + closure_cells + 1e-9)) + 1,
        )
        high_key = min(
            y_high,
            int(math.ceil(float(inner_hi) - closure_cells - 1e-9)) - 1,
        )
        if low_key <= high_key:
            keys.extend(
                (int(ix), iy, int(iz))
                for iy in range(low_key, high_key + 1)
                if (int(ix), iy, int(iz)) not in solid_keys
            )
    if not keys:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(keys, dtype=np.float64) * pitch


def x5_dynamic_visual_vertices(gripper_qpos: Any) -> np.ndarray:
    """Move the frozen official link6/link7/link8 mesh to the current qpos."""
    geometry = x5_gripper_geometry()
    vertices = np.asarray(geometry["visual_vertices"], dtype=np.float64).copy()
    faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
    components = np.asarray(geometry["visual_face_component"], dtype=np.int64)
    finger_travel = float(geometry["metadata"]["gripper_open_m"])
    closure = finger_travel - x5_gripper_joint_position_m(gripper_qpos)
    for component_index, y_shift in ((1, -closure), (2, closure)):
        vertex_indices = np.unique(faces[components == component_index].reshape(-1))
        vertices[vertex_indices, 1] += y_shift
    return vertices


def points_inside_legacy_x5_grasp_volume(
    points_eef: Any,
    gripper_qpos: Any,
) -> np.ndarray:
    """Classify link6-frame points against the tapered X5 grasp corridor."""
    points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    result = np.zeros(len(points), dtype=bool)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return result
    geometry = x5_gripper_geometry()
    bounds = x5_legacy_opening_bounds(gripper_qpos)
    if bounds[0, 1] > bounds[1, 1]:
        return result

    command = x5_gripper_command_fraction(gripper_qpos)
    joint_position = x5_gripper_joint_position_m([command])
    finger_travel = float(geometry["metadata"]["gripper_open_m"])
    closure = max(0.0, finger_travel - joint_position)
    lookup = geometry["_opening_column_lookup"]
    pitch = float(VOXEL_M)
    solid_keys = _x5_dynamic_solid_keys(geometry, closure)
    # A column is a square X-Z voxel footprint.  Querying the nearest center
    # gives the same stepped projection as the voxel volume while retaining a
    # continuous y test for RGB-D points.
    indices = np.flatnonzero(finite)
    query = points[indices]
    ix_values = np.rint(query[:, 0] / pitch).astype(np.int64)
    iz_values = np.rint(query[:, 2] / pitch).astype(np.int64)
    for output_index, point, ix, iz in zip(indices, query, ix_values, iz_values):
        column = lookup.get((int(ix), int(iz)))
        if column is None:
            continue
        if (
            abs(float(point[0]) - float(ix) * pitch) > 0.5 * pitch + 1e-9
            or abs(float(point[2]) - float(iz) * pitch) > 0.5 * pitch + 1e-9
        ):
            continue
        y_key = int(np.rint(float(point[1]) / pitch))
        if (int(ix), y_key, int(iz)) in solid_keys:
            continue
        inner_lo, inner_hi = column
        lower = max(
            float(bounds[0, 1]),
            float(inner_lo) * pitch + closure,
        )
        upper = min(
            float(bounds[1, 1]),
            float(inner_hi) * pitch - closure,
        )
        result[output_index] = lower < float(point[1]) < upper
    return result


def points_inside_x5_grasp_volume(points_eef: Any, gripper_qpos: Any) -> np.ndarray:
    """Continuous membership in the same inner-triangle volume as the views."""
    from tools.util.x5_exact_opening import points_inside_exact_x5_opening

    return points_inside_exact_x5_opening(points_eef, gripper_qpos)


def x5_dynamic_opening_voxels(gripper_qpos: Any) -> np.ndarray:
    """Quadrature centers inside the exact, un-eroded current opening."""
    from tools.util.x5_exact_opening import exact_x5_opening_voxels

    return exact_x5_opening_voxels(gripper_qpos).copy()


def x5_dynamic_opening_bounds(gripper_qpos: Any) -> np.ndarray:
    """AABB of the exact inner-triangle mesh; never a membership shortcut."""
    mesh = x5_dynamic_opening_mesh(gripper_qpos)
    vertices = np.asarray(mesh["vertices"])
    if not len(vertices):
        return np.zeros((2, 3), dtype=np.float64)
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def _empty_opening_mesh() -> dict[str, Any]:
    """Return a consistent empty diagnostic for a fully closed corridor."""
    return {
        "vertices": np.zeros((0, 3), dtype=np.float64),
        "faces": np.zeros((0, 3), dtype=np.int64),
        "volume_cm3": 0.0,
        "exact": True,
        "source": "empty_official_x5_inner_surface_corridor",
        "inner_y_bounds_m": np.zeros(2, dtype=np.float64),
        "boundary_definition": (
            "official_link8_to_link7_inner_triangle_planes_per_xz_cell"
        ),
    }


@lru_cache(maxsize=16)
def _translated_exact_opening_mesh(command_key: float) -> dict[str, Any]:
    """Translate the official inner-surface mesh for a prismatic X5 command.

    Both RoboDojo jaw joints are pure Y translations.  The frozen mesh stores
    which half of every vertex belongs to link8 (lower) or link7 (upper), so a
    dynamic pose is obtained exactly by translating those vertices; no cube,
    voxel lookup, simulator import, or runtime Boolean approximation is
    involved.
    """
    command = float(np.clip(command_key, 0.0, 1.0))
    geometry = x5_gripper_geometry()
    vertices = np.asarray(
        geometry["exact_opening_mesh_vertices"], dtype=np.float64
    ).copy()
    faces = np.asarray(geometry["exact_opening_mesh_faces"], dtype=np.int64).copy()
    side = np.asarray(geometry["exact_opening_mesh_vertex_side"], dtype=np.int8)
    closure = float(geometry["metadata"]["gripper_open_m"]) - (
        x5_gripper_joint_position_m([command])
    )
    # side=+1 is the positive-y/link7 surface, which moves toward -Y on
    # closure; side=-1 is link8 and moves toward +Y.
    vertices[:, 1] -= side.astype(np.float64) * closure
    triangles = vertices[faces]
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )
    if not np.isfinite(signed_volume) or abs(signed_volume) <= 1e-12:
        return _empty_opening_mesh()
    if signed_volume < 0.0:
        faces = faces[:, [0, 2, 1]]
        signed_volume = -signed_volume
    y_bounds = np.asarray(
        [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
        dtype=np.float64,
    )
    return {
        "vertices": vertices,
        "faces": faces,
        "volume_cm3": float(signed_volume * 1e6),
        "exact": True,
        "source": "frozen_official_x5_inner_surface_triangle_mesh_prismatic_translation",
        "inner_y_bounds_m": y_bounds,
        "boundary_definition": geometry["metadata"].get("exact_opening_y_boundary"),
        "arrangement_grid_m": geometry["metadata"].get(
            "exact_opening_arrangement_grid_m"
        ),
    }


def x5_dynamic_opening_mesh(gripper_qpos: Any) -> dict[str, Any]:
    """Return the exact triangulated free volume for the current X5 qpos.

    This physical-mesh diagnostic is independent of the 3 mm benchmark voxel
    occupancy.  It follows the official link7/link8 inner triangle surfaces
    under their measured prismatic joint motion.
    """
    command = x5_gripper_command_fraction(gripper_qpos)
    return _translated_exact_opening_mesh(round(command, 6))


def icosa_face_normals() -> np.ndarray:
    """Return the same 20 deterministic directions as the BEHAVIOR planner."""
    phi = (1.0 + 5.0**0.5) / 2.0
    b, c = 1.0 / phi, phi
    vertices: list[tuple[float, float, float]] = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                vertices.append((sx, sy, sz))
    for sy in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((0.0, sy * b, sz * c))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((sx * b, sz * c, 0.0))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((sx * c, 0.0, sz * b))
    result = np.asarray(vertices, dtype=np.float64)
    result /= np.linalg.norm(result, axis=1, keepdims=True) + 1e-12
    return result


def x5_eef_frame_from_approach_roll(
    approach: np.ndarray,
    roll: float,
) -> np.ndarray:
    """Build a right-handed EEF frame whose local +X is the approach axis."""
    x_axis = np.asarray(approach, dtype=np.float64).reshape(3)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_zero = np.cross(reference, x_axis)
    if float(np.linalg.norm(y_zero)) < 1e-3:
        y_zero = np.cross(np.array([0.0, 1.0, 0.0]), x_axis)
    y_zero /= np.linalg.norm(y_zero) + 1e-12
    z_zero = np.cross(x_axis, y_zero)
    z_zero /= np.linalg.norm(z_zero) + 1e-12
    cosine, sine = math.cos(float(roll)), math.sin(float(roll))
    y_axis = cosine * y_zero + sine * z_zero
    z_axis = -sine * y_zero + cosine * z_zero
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if float(np.linalg.det(rotation)) < 0.999:
        raise ValueError("failed to construct a right-handed X5 EEF frame")
    return rotation


__all__ = [
    "ASSET_PATH",
    "ROBODOJO_X5_GRIPPER_SCALE_M",
    "VOXEL_M",
    "icosa_face_normals",
    "points_inside_x5_grasp_volume",
    "x5_dynamic_opening_bounds",
    "x5_dynamic_opening_voxels",
    "x5_dynamic_opening_mesh",
    "x5_dynamic_visual_vertices",
    "x5_eef_frame_from_approach_roll",
    "x5_gripper_command_from_joint_position_m",
    "x5_gripper_command_fraction",
    "x5_gripper_joint_position_m",
    "x5_gripper_open_fraction",
    "x5_gripper_geometry",
]
