"""Numpy-only self-collision guard from the frozen public X5 robot model.

No task geometry, contact queries, object identities or simulator are used.
Convex separation is certified by a support plane; inconclusive GJK iterations
fail closed. This is a conservative robot envelope, not PhysX contact truth.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from tools.util.x5_kinematics import _axis_angle_matrix


BUILD = "official_x5_convex_envelope_sampled_self_collision_v1"
ASSET = Path(__file__).with_name("assets") / "x5_self_collision_v1.npz"
CLEARANCE_M = 0.0005
PATH_STEP_RAD = 0.08


def _closest_simplex(points):
    """Closest point to the origin in a <=4 point simplex, including faces."""
    values = np.asarray(points, dtype=np.float64)
    best_distance, best_point, best_indices = math.inf, None, None
    for size in range(1, len(values) + 1):
        for indices in combinations(range(len(values)), size):
            subset = values[list(indices)]
            if size == 1:
                weights = np.ones(1)
            else:
                basis = (subset[1:] - subset[0]).T
                coeff, _, rank, _ = np.linalg.lstsq(basis, -subset[0], rcond=1e-12)
                if rank != size - 1:
                    continue
                weights = np.r_[1.0 - np.sum(coeff), coeff]
                if np.min(weights) < -1e-10:
                    continue
                weights = np.maximum(weights, 0.0)
                weights /= weights.sum()
            point = weights @ subset
            distance = float(point @ point)
            if distance < best_distance:
                best_distance, best_point = distance, point
                best_indices = [indices[i] for i in np.flatnonzero(weights > 1e-10)]
    return best_point, [values[i] for i in best_indices]


def convex_separation(a, b, clearance_m=CLEARANCE_M):
    """Certify a gap > clearance, otherwise report contact or uncertainty.

    The reported separating lower bound is a projection gap, not a claimed
    exact penetration depth or volume. All finite vertex sets are supported.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (3,)
        or b.shape[1:] != (3,)
        or not len(a)
        or not len(b)
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or not math.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("convex vertices and clearance must be finite and valid")
    axis_gap = np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0))
    if float(axis_gap.max()) > clearance_m:
        return {
            "separated": True,
            "reason": "aabb_separating_plane",
            "separation_lower_bound_m": float(axis_gap.max()),
            "iterations": 0,
        }
    direction = a.mean(0) - b.mean(0)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    simplex = []
    upper = math.inf
    for iteration in range(64):
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return {
                "separated": False,
                "reason": "convex_envelopes_intersect",
                "distance_upper_bound_m": norm,
                "iterations": iteration,
            }
        unit = direction / norm
        support = a[np.argmin(a @ unit)] - b[np.argmax(b @ unit)]
        lower = float(unit @ support)
        if lower > clearance_m + 1e-10:
            return {
                "separated": True,
                "reason": "gjk_separating_plane",
                "separation_lower_bound_m": lower,
                "iterations": iteration + 1,
            }
        if any(np.linalg.norm(support - previous) < 1e-11 for previous in simplex):
            return {
                "separated": False,
                "reason": "clearance_not_certified",
                "distance_upper_bound_m": upper,
                "iterations": iteration + 1,
            }
        simplex.append(support)
        direction, simplex = _closest_simplex(simplex)
        upper = float(np.linalg.norm(direction))
        if upper <= clearance_m + 1e-10:
            return {
                "separated": False,
                "reason": "within_clearance_envelope",
                "distance_upper_bound_m": upper,
                "iterations": iteration + 1,
            }
        # An interior tetrahedron has zero distance. A nonzero closest point
        # lies on a face and must reduce to <=3 vertices before the next step.
        if len(simplex) >= 4:
            return {
                "separated": False,
                "reason": "numerically_uncertain_simplex",
                "distance_upper_bound_m": upper,
                "iterations": iteration + 1,
            }
    return {
        "separated": False,
        "reason": "gjk_iteration_limit",
        "distance_upper_bound_m": upper,
        "iterations": 64,
    }


@lru_cache(maxsize=1)
def robot_asset():
    with np.load(ASSET, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if (
            metadata.get("schema") != "robodojo_static_x5_self_collision"
            or metadata.get("version") != 1
        ):
            raise ValueError("unsupported static X5 collision asset")
        vertices = {}
        for collider in metadata["colliders"]:
            array = np.asarray(source[collider["key"]], dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                raise ValueError("invalid X5 collision hull")
            array.setflags(write=False)
            vertices[collider["key"]] = array
    groups = {"base_link": "base_link"}
    for joint in metadata["joints"]:
        groups[joint["child"]] = (
            groups[joint["parent"]] if joint["motion"] == "fixed" else joint["child"]
        )
    adjacent = {
        frozenset((groups[j["parent"]], groups[j["child"]]))
        for j in metadata["joints"]
        if j["motion"] != "fixed"
    }
    pairs = []
    for a, b in combinations(metadata["colliders"], 2):
        ga, gb = groups[a["body"]], groups[b["body"]]
        if ga != gb and frozenset((ga, gb)) not in adjacent:
            pairs.append((a, b))
    # Report an ordinary convex collider conflict before a looser outer bound.
    # Both remain enforced; this changes diagnostic priority, not acceptance.
    pairs.sort(
        key=lambda pair: bool(
            pair[0]["outer_bound_only"] or pair[1]["outer_bound_only"]
        )
    )
    return metadata, vertices, pairs


def link_transforms(joints, finger_position_m=0.044):
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.size != 6 or not np.all(np.isfinite(q)):
        raise ValueError("X5 self collision requires six finite joints")
    if not math.isfinite(finger_position_m) or not 0 <= finger_position_m <= 0.044:
        raise ValueError("X5 finger position outside official joint limits")
    transforms = {"base_link": np.eye(4)}
    for joint in robot_asset()[0]["joints"]:
        motion = np.eye(4)
        if joint["motion"] == "revolute":
            motion[:3, :3] = _axis_angle_matrix(
                np.asarray(joint["axis"]), q[int(joint["name"][5:]) - 1]
            )
        elif joint["motion"] == "prismatic":
            motion[:3, 3] = np.asarray(joint["axis"]) * finger_position_m
        transforms[joint["child"]] = (
            transforms[joint["parent"]]
            @ np.asarray(joint["local0"])
            @ motion
            @ np.asarray(joint["local1_inverse"])
        )
    return transforms


def check_configuration(joints, *, finger_position_m=0.044, clearance_m=CLEARANCE_M):
    metadata, vertices, pairs = robot_asset()
    transforms = link_transforms(joints, finger_position_m)
    transformed = {}
    for collider in metadata["colliders"]:
        t = transforms[collider["body"]]
        transformed[collider["key"]] = (
            vertices[collider["key"]] @ t[:3, :3].T + t[:3, 3]
        )
    checked = 0
    for a, b in pairs:
        checked += 1
        result = convex_separation(
            transformed[a["key"]], transformed[b["key"]], clearance_m
        )
        if not result["separated"]:
            return {
                "ok": False,
                "reason": "robot_self_collision_envelope",
                "pair": [a["body"], b["body"]],
                "pair_check": result,
                "pairs_checked": checked,
                "outer_bound_only": bool(
                    a["outer_bound_only"] or b["outer_bound_only"]
                ),
            }
    return {
        "ok": True,
        "reason": "all_nonadjacent_envelopes_separated",
        "pairs_checked": checked,
    }


def check_paired_path(current_q, safe_q, final_q, *, finger_position_m=0.044):
    """Check both targets and bounded joint-linear approach/descent samples.

    This matches nominal immutable-target replay, not a continuous swept-volume
    proof or a prediction of load-induced tracking error. Reverse descent is
    the nominal post-grasp lift branch; the actual EEF servo must still arrive.
    """
    points = [
        np.asarray(q, dtype=np.float64).reshape(6) for q in (current_q, safe_q, final_q)
    ]
    if not all(np.all(np.isfinite(q)) for q in points):
        raise ValueError("path joints must be finite")
    checks = 0
    audit = {
        "build": BUILD,
        "ok": False,
        "clearance_m": CLEARANCE_M,
        "path_max_joint_step_rad": PATH_STEP_RAD,
        "finger_position_m": finger_position_m,
        "scope": "single_arm_nonadjacent_robot_convex_envelopes",
        "continuous_swept_volume_checked": False,
        "other_arm_checked": False,
        "environment_collision_checked": False,
        "simulator_contact_truth_used": False,
        "conservative_outer_bound_bodies": ["link3"],
        "fixed_cluster_and_adjacent_pairs_excluded": True,
    }
    samples = [
        ("final", 1.0, points[2]),
        ("safe", 1.0, points[1]),
        ("current", 0.0, points[0]),
    ]
    for label, start, end in (
        ("approach", points[0], points[1]),
        ("descent_and_reverse_lift", points[1], points[2]),
    ):
        intervals = max(
            1, int(math.ceil(float(np.max(np.abs(end - start))) / PATH_STEP_RAD))
        )
        samples.extend(
            (label, i / intervals, start + (end - start) * i / intervals)
            for i in range(1, intervals)
        )
    for stage, fraction, q in samples:
        check = check_configuration(q, finger_position_m=finger_position_m)
        checks += 1
        if not check["ok"]:
            return {
                **audit,
                "reason": check["reason"],
                "failed_stage": stage,
                "failed_fraction": fraction,
                "failed_joints": q.tolist(),
                "sample_count": checks,
                "failure": check,
            }
    return {
        **audit,
        "ok": True,
        "reason": "sampled_self_collision_clear",
        "sample_count": checks,
    }
