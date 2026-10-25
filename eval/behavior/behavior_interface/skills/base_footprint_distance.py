"""R1Pro base footprint distance helpers.

The footprint is built from the robot asset collision geometry:
  * base_link convex collision pieces from the converted USD asset;
  * three floor-touching wheel links from the URDF joint origins, treated as
    collision disks using the USD wheel collision radius.

All distance functions are 2D in world XY. They are read-only utilities and do
not touch the simulator.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
_R1PRO_ASSET = _ROOT / "datasets" / "omnigibson-robot-assets" / "models" / "r1pro"
_R1PRO_URDF = _R1PRO_ASSET / "urdf" / "r1pro.urdf"
_R1PRO_USD = _R1PRO_ASSET / "usd" / "r1pro.usda"


def _parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in re.split(r"[,\s]+", s.strip()) if x], dtype=np.float64)


def _numbers_in_points(points_blob: str) -> np.ndarray:
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", points_blob)]
    if len(nums) % 3:
        raise ValueError(f"USD points length is not divisible by 3: {len(nums)}")
    return np.asarray(nums, dtype=np.float64).reshape(-1, 3)


def _mesh_blocks(text: str, mesh_name_prefix: str) -> Iterable[str]:
    pat = re.compile(rf'def Mesh "{re.escape(mesh_name_prefix)}[^"]*"[\s\S]*?uniform token\[\] xformOpOrder')
    yield from (m.group(0) for m in pat.finditer(text))


def _extract_points_scale_translate(block: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pm = re.search(r"point3f\[\] points = \[(.*?)\]\s*(?:\(|\n)", block, re.S)
    sm = re.search(r"xformOp:scale = \(([^)]*)\)", block)
    tm = re.search(r"xformOp:translate = \(([^)]*)\)", block)
    if pm is None or sm is None:
        raise ValueError("USD mesh block missing points or scale")
    pts = _numbers_in_points(pm.group(1))
    scale = _parse_vec(sm.group(1))
    translate = _parse_vec(tm.group(1)) if tm is not None else np.zeros(3, dtype=np.float64)
    return pts, scale, translate


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in np.asarray(points_xy)[:, :2]})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside


def _dist_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-15:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / den)
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        abs(o1) <= 1e-12 and on_seg(a, c, b)
        or abs(o2) <= 1e-12 and on_seg(a, d, b)
        or abs(o3) <= 1e-12 and on_seg(c, a, d)
        or abs(o4) <= 1e-12 and on_seg(c, b, d)
    )


def point_to_polygon_distance(point_xy: Sequence[float], poly_xy: np.ndarray) -> float:
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    poly = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(poly) == 0:
        return float("inf")
    if len(poly) >= 3 and _point_in_poly(p, poly):
        return 0.0
    return min(_dist_point_segment(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def point_to_rect_distance(point_xy: Sequence[float], rect_min_xy: Sequence[float], rect_max_xy: Sequence[float]) -> float:
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    lo = np.asarray(rect_min_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_max_xy, dtype=np.float64).reshape(2)
    dx = max(float(lo[0] - p[0]), 0.0, float(p[0] - hi[0]))
    dy = max(float(lo[1] - p[1]), 0.0, float(p[1] - hi[1]))
    return float(math.hypot(dx, dy))


def polygon_to_rect_distance(poly_xy: np.ndarray, rect_min_xy: Sequence[float], rect_max_xy: Sequence[float]) -> float:
    poly = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    lo = np.asarray(rect_min_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_max_xy, dtype=np.float64).reshape(2)
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=np.float64)
    if any(_point_in_poly(v, poly) for v in rect) or any(lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] for p in poly):
        return 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        for j in range(4):
            if _segments_intersect(a, b, rect[j], rect[(j + 1) % 4]):
                return 0.0
    d = [point_to_polygon_distance(v, poly) for v in rect]
    d += [point_to_rect_distance(p, lo, hi) for p in poly]
    return float(min(d))


@lru_cache(maxsize=1)
def load_r1pro_base_footprint_local() -> Dict[str, Any]:
    """Load R1Pro base footprint in base_link XY coordinates."""
    usd_text = _R1PRO_USD.read_text(errors="ignore")
    base_pts: List[np.ndarray] = []
    for block in _mesh_blocks(usd_text, "base_link_col_"):
        pts, scale, translate = _extract_points_scale_translate(block)
        base_pts.append(pts * scale.reshape(1, 3) + translate.reshape(1, 3))
    if not base_pts:
        raise RuntimeError(f"no base_link collision meshes found in {_R1PRO_USD}")
    base_hull = _convex_hull(np.vstack(base_pts)[:, :2])

    root = ET.parse(_R1PRO_URDF).getroot()
    joint_origin: Dict[str, np.ndarray] = {}
    joint_parent: Dict[str, str] = {}
    joint_child: Dict[str, str] = {}
    for j in root.findall("joint"):
        name = j.attrib.get("name", "")
        child = j.find("child")
        parent = j.find("parent")
        origin = j.find("origin")
        if child is None or parent is None:
            continue
        joint_child[name] = child.attrib["link"]
        joint_parent[name] = parent.attrib["link"]
        xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        joint_origin[name] = _parse_vec(xyz)

    wheel_centers = []
    for i in (1, 2, 3):
        steer = joint_origin[f"steer_motor_joint{i}"]
        wheel = joint_origin[f"wheel_motor_joint{i}"]
        wheel_centers.append((steer + wheel)[:2])

    wheel_radii = []
    for block in _mesh_blocks(usd_text, "collisions"):
        # Wheel collisions are the only Mesh "collisions" blocks with boundingSphere
        # and a scale close to 0.14/0.055/0.14. Use the max horizontal/vertical radius.
        if "boundingSphere" not in block:
            continue
        pts, scale, _ = _extract_points_scale_translate(block)
        radius = float(np.max(np.linalg.norm((pts * scale.reshape(1, 3))[:, [0, 2]], axis=1)))
        if 0.04 <= radius <= 0.12:
            wheel_radii.append(radius)
    wheel_radius = float(np.median(wheel_radii)) if wheel_radii else 0.07

    return {
        "asset_urdf": str(_R1PRO_URDF),
        "asset_usd": str(_R1PRO_USD),
        "base_hull_xy": base_hull.tolist(),
        "base_hull_extent_xy": {
            "min": np.min(base_hull, axis=0).round(6).tolist(),
            "max": np.max(base_hull, axis=0).round(6).tolist(),
        },
        "wheel_centers_xy": [c.round(6).tolist() for c in wheel_centers],
        "wheel_radius_m": wheel_radius,
    }


def transform_xy_local_to_world(points_xy: np.ndarray, base_x: float, base_y: float, yaw_deg: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ rot.T + np.array([float(base_x), float(base_y)], dtype=np.float64)


def measure_base_distance(
    *,
    base_pose: Dict[str, float],
    target_xy: Sequence[float],
    target_aabb_min: Optional[Sequence[float]] = None,
    target_aabb_max: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    fp = load_r1pro_base_footprint_local()
    hull_world = transform_xy_local_to_world(
        np.asarray(fp["base_hull_xy"], dtype=np.float64),
        float(base_pose["x"]),
        float(base_pose["y"]),
        float(base_pose["yaw_deg"]),
    )
    wheel_centers_world = transform_xy_local_to_world(
        np.asarray(fp["wheel_centers_xy"], dtype=np.float64),
        float(base_pose["x"]),
        float(base_pose["y"]),
        float(base_pose["yaw_deg"]),
    )
    wheel_radius = float(fp["wheel_radius_m"])
    p = np.asarray(target_xy, dtype=np.float64).reshape(2)
    base_hull_point = point_to_polygon_distance(p, hull_world)
    wheel_point = max(0.0, min(float(np.linalg.norm(p - c)) - wheel_radius for c in wheel_centers_world))
    point_dist = min(base_hull_point, wheel_point)
    out: Dict[str, Any] = {
        "base_pose": {k: float(base_pose[k]) for k in ("x", "y", "yaw_deg") if k in base_pose},
        "target_xy": p.round(6).tolist(),
        "distance_point_m": point_dist,
        "distance_point_cm": point_dist * 100.0,
        "base_hull_point_distance_m": base_hull_point,
        "wheel_point_distance_m": wheel_point,
        "wheel_radius_m": wheel_radius,
        "wheel_centers_world_xy": wheel_centers_world.round(6).tolist(),
        "base_hull_world_xy": hull_world.round(6).tolist(),
        "footprint_source": {
            "urdf": fp["asset_urdf"],
            "usd": fp["asset_usd"],
            "base_hull_extent_xy": fp["base_hull_extent_xy"],
        },
    }
    if target_aabb_min is not None and target_aabb_max is not None:
        lo = np.asarray(target_aabb_min, dtype=np.float64).reshape(-1)[:2]
        hi = np.asarray(target_aabb_max, dtype=np.float64).reshape(-1)[:2]
        base_hull_rect = polygon_to_rect_distance(hull_world, lo, hi)
        wheel_rect = max(0.0, min(point_to_rect_distance(c, lo, hi) - wheel_radius for c in wheel_centers_world))
        rect_dist = min(base_hull_rect, wheel_rect)
        out.update({
            "target_aabb_min_xy": lo.round(6).tolist(),
            "target_aabb_max_xy": hi.round(6).tolist(),
            "distance_aabb_m": rect_dist,
            "distance_aabb_cm": rect_dist * 100.0,
            "base_hull_aabb_distance_m": base_hull_rect,
            "wheel_aabb_distance_m": wheel_rect,
        })
    return out


def find_scene_graph_object(scene_graph: Dict[str, Any], object_name: str) -> Optional[Dict[str, Any]]:
    for obj in scene_graph.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("bddl_key") == object_name or obj.get("name") == object_name or obj.get("scene_name") == object_name:
            return obj
    return None


def measure_from_state_and_scene_graph(
    state: Dict[str, Any],
    scene_graph: Dict[str, Any],
    object_name: str,
) -> Dict[str, Any]:
    obj = find_scene_graph_object(scene_graph, object_name)
    if obj is None:
        raise KeyError(f"object not found in scene graph: {object_name}")
    tro = state.get("tro") or {}
    target = (tro.get(object_name) or {}).get("pos") or obj.get("pos")
    if target is None:
        lo = np.asarray(obj["aabb_min"], dtype=np.float64)
        hi = np.asarray(obj["aabb_max"], dtype=np.float64)
        target = (0.5 * (lo + hi)).tolist()
    out = measure_base_distance(
        base_pose=state["base_pose"],
        target_xy=np.asarray(target, dtype=np.float64)[:2],
        target_aabb_min=obj.get("aabb_min"),
        target_aabb_max=obj.get("aabb_max"),
    )
    out.update({
        "object_name": object_name,
        "object_pos": [float(x) for x in target],
        "object_aabb_min": obj.get("aabb_min"),
        "object_aabb_max": obj.get("aabb_max"),
    })
    return out


def dumps_summary(measure: Dict[str, Any]) -> str:
    keys = [
        "object_name",
        "distance_point_cm",
        "distance_aabb_cm",
        "base_hull_point_distance_m",
        "wheel_point_distance_m",
        "base_hull_aabb_distance_m",
        "wheel_aabb_distance_m",
    ]
    return json.dumps({k: measure.get(k) for k in keys if k in measure}, ensure_ascii=False, indent=2)
