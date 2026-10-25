from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

URDF = Path(__file__).resolve().parents[2] / "robot_config/model.urdf"
ACTION_NAMES = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 9)]
LOCKED_INDICES = ()


def numbers(value):
    return np.array([float(x) for x in value.split()])


@lru_cache(maxsize=8)
def joints(urdf=URDF):
    result = {}
    for node in ET.parse(urdf).getroot().findall("joint"):
        origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
        result[node.get("name")] = {
            "type": node.get("type"),
            "parent": node.find("parent").get("link"),
            "child": node.find("child").get("link"),
            "xyz": numbers(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            ),
            "rpy": numbers(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            ),
            "axis": numbers(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0"),
            "limits": {k: float(v) for k, v in limit.attrib.items()}
            if limit is not None
            else {},
        }
    return result


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin(xyz, rpy):
    roll, pitch, yaw = rpy
    matrix = np.eye(4)
    matrix[:3, :3] = (
        rotation((0, 0, 1), yaw)
        @ rotation((0, 1, 0), pitch)
        @ rotation((1, 0, 0), roll)
    )
    matrix[:3, 3] = xyz
    return matrix


def forward(positions, urdf=URDF):
    frames, axes, pending = {"base_link": np.eye(4)}, {}, dict(joints(urdf))
    while pending:
        progressed = False
        for name, joint in list(pending.items()):
            if joint["parent"] not in frames:
                continue
            before = frames[joint["parent"]] @ origin(joint["xyz"], joint["rpy"])
            axes[name] = (before[:3, 3].copy(), before[:3, :3] @ joint["axis"])
            motion, value = np.eye(4), float(positions.get(name, 0))
            if joint["type"] in ("revolute", "continuous"):
                motion[:3, :3] = rotation(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * value
            frames[joint["child"]] = before @ motion
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError("URDF contains disconnected or cyclic joints")
    return frames, axes
