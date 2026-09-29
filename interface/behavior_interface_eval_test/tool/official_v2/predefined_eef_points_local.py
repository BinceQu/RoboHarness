"""Submission-local predefined points on either R1Pro gripper EEF.

These points are robot geometry, not observations of scene objects. Their
robot-base coordinates are obtained from evaluator proprioception through the
submission-local FK model. No simulator handle or runtime robot geometry is
used here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
import os

import numpy as np


PREDEFINED_EEF_POINT_NAMES = (
    "left_finger_tip",
    "right_finger_tip",
    "gripper_slide_center",
)
PREDEFINED_EEF_POINT_NAME_SET = frozenset(PREDEFINED_EEF_POINT_NAMES)

# The predefined fingertip points and the red plan overlay must use the same
# submission-local mesh.  In the EEF frame +Y is the left-finger side and +Z
# points from the palm toward the narrow physical tips.
_OPEN_GRIPPER_Q_M = 0.05
_FINGERTIP_CAP_BAND_M = 0.00025
_GRIPPER_VISUAL_MESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "r1pro_gripper_visual_mesh_open.npz",
)
_FINGER_COMPONENT_NAMES = (
    "right_gripper_finger_link1",
    "right_gripper_finger_link2",
)

# Center of the physical finger-slide rail surface in the red-overlay mesh.
# The prismatic joint origins are above the rail inside the open finger cavity,
# so their Z coordinate is not a visible point on the slide.  X is the midpoint
# of the two URDF finger-joint origins and Z is the broad central rail face.
_GRIPPER_SLIDE_CENTER_EEF_M = np.array(
    [-1.685e-7, 0.0, -0.062], dtype=np.float64
)


@lru_cache(maxsize=1)
def _open_fingertip_surface_centers_eef_m() -> tuple[np.ndarray, np.ndarray]:
    """Return the two physical tip-cap centers from the overlay mesh.

    The max-EEF-Z end is the narrow contact end of each finger.  The old
    implementation selected the min-Z, palm-side end and therefore displaced
    both named tips by roughly 97 mm in the EEF frame.
    """

    with np.load(_GRIPPER_VISUAL_MESH_PATH, allow_pickle=False) as asset:
        vertices = np.asarray(asset["vertices_eef"], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(asset["faces"], dtype=np.int64).reshape(-1, 3)
        face_component = np.asarray(
            asset["face_component"], dtype=np.int64
        ).reshape(-1)
        component_names = tuple(str(value) for value in asset["component_names"])
        gripper_q_m = float(
            np.asarray(asset["gripper_q_m"], dtype=np.float64).reshape(-1)[0]
        )
    if abs(gripper_q_m - _OPEN_GRIPPER_Q_M) > 1.0e-6:
        raise ValueError("predefined EEF point mesh is not the open reference geometry")
    if len(face_component) != len(faces):
        raise ValueError("predefined EEF point mesh has inconsistent face components")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("predefined EEF point mesh contains an invalid face index")

    points: list[np.ndarray] = []
    for component_name in _FINGER_COMPONENT_NAMES:
        try:
            component_id = component_names.index(component_name)
        except ValueError as exc:
            raise ValueError(
                f"predefined EEF point mesh lacks {component_name}"
            ) from exc
        vertex_ids = np.unique(faces[face_component == component_id])
        if vertex_ids.size == 0:
            raise ValueError(f"predefined EEF point mesh component {component_name} is empty")
        component_vertices = vertices[vertex_ids]
        tip_z = float(np.max(component_vertices[:, 2]))
        cap = component_vertices[
            component_vertices[:, 2] >= tip_z - _FINGERTIP_CAP_BAND_M
        ]
        if cap.shape[0] < 3:
            raise ValueError(
                f"predefined EEF point mesh component {component_name} has no tip cap"
            )
        point = np.asarray(cap.mean(axis=0), dtype=np.float64)
        point[2] = tip_z
        point.setflags(write=False)
        points.append(point)

    left, right = points
    if not (left[1] > 0.0 and right[1] < 0.0):
        raise ValueError("predefined EEF fingertip sides do not match the EEF +Y convention")
    return left, right


def is_predefined_eef_point_name(name: object) -> bool:
    """Return whether ``name`` selects submission-local EEF geometry."""

    return isinstance(name, str) and name in PREDEFINED_EEF_POINT_NAME_SET


def partition_predefined_eef_point_names(
    names: Iterable[object],
) -> tuple[list[str], list[str]]:
    """Split names into predefined and visual lists, preserving order."""

    predefined: list[str] = []
    tracked: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        (predefined if is_predefined_eef_point_name(name) else tracked).append(name)
    return predefined, tracked


def _gripper_q_pair(gripper_q_m: object) -> np.ndarray:
    q = np.asarray(gripper_q_m, dtype=np.float64).reshape(-1)
    if q.size < 2 or not np.all(np.isfinite(q[:2])):
        raise ValueError("gripper proprioception must contain two finite positions")
    return q[:2]


def predefined_eef_point_local_m(name: str, gripper_q_m: object) -> np.ndarray:
    """Return one predefined point in the selected gripper's EEF frame."""

    point_name = str(name)
    if point_name not in PREDEFINED_EEF_POINT_NAME_SET:
        raise ValueError(f"unknown predefined EEF point {point_name!r}")
    q = _gripper_q_pair(gripper_q_m)
    left_open, right_open = _open_fingertip_surface_centers_eef_m()
    if point_name == "left_finger_tip":
        point = left_open.copy()
        point[1] += float(q[0]) - _OPEN_GRIPPER_Q_M
        return point
    if point_name == "right_finger_tip":
        point = right_open.copy()
        point[1] -= float(q[1]) - _OPEN_GRIPPER_Q_M
        return point
    return _GRIPPER_SLIDE_CENTER_EEF_M.copy()


def quaternion_xyzw_to_matrix(quaternion_xyzw: object) -> np.ndarray:
    """Convert a finite XYZW quaternion to a proper rotation matrix."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError("EEF quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("EEF quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def predefined_eef_points_robot_base_m(
    names: Iterable[str],
    *,
    eef_position_robot_base_m: object,
    eef_quaternion_xyzw: object,
    gripper_q_m: object,
) -> dict[str, np.ndarray]:
    """Resolve predefined EEF points from one synchronized proprio/FK pose."""

    position = np.asarray(eef_position_robot_base_m, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("EEF position must contain three finite values")
    rotation = quaternion_xyzw_to_matrix(eef_quaternion_xyzw)
    result: dict[str, np.ndarray] = {}
    for raw_name in names:
        name = str(raw_name)
        result[name] = position + rotation @ predefined_eef_point_local_m(
            name, gripper_q_m
        )
    return result


def predefined_eef_point_catalog() -> list[Mapping[str, str]]:
    """Return JSON-ready public metadata for the three reserved names."""

    return [
        {
            "name": "left_finger_tip",
            "description": "center of the left finger narrow front contact cap",
        },
        {
            "name": "right_finger_tip",
            "description": "center of the right finger narrow front contact cap",
        },
        {
            "name": "gripper_slide_center",
            "description": "surface center of the fixed finger slide rail",
        },
    ]
