"""Submission-local predefined points on either RoboDojo X5 gripper.

The public names match BEHAVIOR official_v2. Their coordinates are derived
from the frozen official X5 visual mesh and current evaluator proprioception;
no simulator handles or runtime robot geometry are consulted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import numpy as np

from tools.util.x5_grasp_geometry import (
    ROBODOJO_X5_GRIPPER_SCALE_M,
    x5_gripper_geometry,
    x5_gripper_joint_position_m,
)


PREDEFINED_EEF_POINT_NAMES = (
    "left_finger_tip",
    "right_finger_tip",
    "gripper_slide_center",
)
PREDEFINED_EEF_POINT_NAME_SET = frozenset(PREDEFINED_EEF_POINT_NAMES)

# The frozen mesh is built at the official prismatic upper limit. X5 link7 is
# the +Y (left) finger and link8 is the -Y (right) finger in the link6 EEF.
_OPEN_GRIPPER_Q_M = float(ROBODOJO_X5_GRIPPER_SCALE_M[1])
_FINGERTIP_CAP_BAND_M = 0.00025
_FINGER_COMPONENT_NAMES = ("link7", "link8")

# Midpoint of joint7/joint8 rail origins in the official X5A.urdf, expressed
# in link6. The 2 um / 0.03 um asymmetry is retained instead of rounded away.
_GRIPPER_SLIDE_CENTER_EEF_M = np.array(
    [0.08657, -0.000002, -0.00024363], dtype=np.float64
)


@lru_cache(maxsize=1)
def _open_fingertip_surface_centers_eef_m() -> tuple[np.ndarray, np.ndarray]:
    """Return physical +X front-cap centers from the frozen official mesh."""

    geometry = x5_gripper_geometry()
    vertices = np.asarray(geometry["visual_vertices"], dtype=np.float64)
    faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
    face_component = np.asarray(geometry["visual_face_component"], dtype=np.int64)
    component_names = tuple(str(value) for value in geometry["component_names"])
    if len(face_component) != len(faces):
        raise ValueError("predefined X5 EEF mesh has inconsistent components")

    points: list[np.ndarray] = []
    for component_name in _FINGER_COMPONENT_NAMES:
        try:
            component_id = component_names.index(component_name)
        except ValueError as exc:
            raise ValueError(f"predefined X5 EEF mesh lacks {component_name}") from exc
        vertex_ids = np.unique(faces[face_component == component_id])
        if vertex_ids.size == 0:
            raise ValueError(
                f"predefined X5 EEF mesh component {component_name} is empty"
            )
        component_vertices = vertices[vertex_ids]
        tip_x = float(np.max(component_vertices[:, 0]))
        cap = component_vertices[
            component_vertices[:, 0] >= tip_x - _FINGERTIP_CAP_BAND_M
        ]
        if cap.shape[0] < 3:
            raise ValueError(
                f"predefined X5 EEF mesh component {component_name} has no tip cap"
            )
        point = np.asarray(cap.mean(axis=0), dtype=np.float64)
        point[0] = tip_x
        point.setflags(write=False)
        points.append(point)

    left, right = points
    if not (left[1] > 0.0 and right[1] < 0.0):
        raise ValueError("predefined X5 fingertips do not match the EEF +Y convention")
    return left, right


def is_predefined_eef_point_name(name: object) -> bool:
    return isinstance(name, str) and name in PREDEFINED_EEF_POINT_NAME_SET


def partition_predefined_eef_point_names(
    names: Iterable[object],
) -> tuple[list[str], list[str]]:
    predefined: list[str] = []
    tracked: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        (predefined if is_predefined_eef_point_name(name) else tracked).append(name)
    return predefined, tracked


def _gripper_q_m(gripper_qpos: object) -> float:
    return x5_gripper_joint_position_m(gripper_qpos)


def predefined_eef_point_local_m(
    name: str,
    gripper_q_m: object,
) -> np.ndarray:
    """Return one reserved point in the selected X5 link6 EEF frame."""

    point_name = str(name)
    if point_name not in PREDEFINED_EEF_POINT_NAME_SET:
        raise ValueError(f"unknown predefined EEF point {point_name!r}")
    q_m = _gripper_q_m(gripper_q_m)
    left_open, right_open = _open_fingertip_surface_centers_eef_m()
    if point_name == "left_finger_tip":
        point = left_open.copy()
        point[1] += q_m - _OPEN_GRIPPER_Q_M
        return point
    if point_name == "right_finger_tip":
        point = right_open.copy()
        point[1] -= q_m - _OPEN_GRIPPER_Q_M
        return point
    return _GRIPPER_SLIDE_CENTER_EEF_M.copy()


def quaternion_xyzw_to_matrix(quaternion_xyzw: object) -> np.ndarray:
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
    position = np.asarray(eef_position_robot_base_m, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("EEF position must contain three finite values")
    rotation = quaternion_xyzw_to_matrix(eef_quaternion_xyzw)
    return {
        str(name): position
        + rotation @ predefined_eef_point_local_m(str(name), gripper_q_m)
        for name in names
    }


def predefined_eef_point_catalog() -> list[Mapping[str, str]]:
    return [
        {
            "name": "left_finger_tip",
            "description": "center of the X5 link7 narrow front contact cap",
        },
        {
            "name": "right_finger_tip",
            "description": "center of the X5 link8 narrow front contact cap",
        },
        {
            "name": "gripper_slide_center",
            "description": "midpoint of the two X5 finger rail origins",
        },
    ]
