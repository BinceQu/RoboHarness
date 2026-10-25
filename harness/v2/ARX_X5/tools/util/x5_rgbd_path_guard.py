"""NumPy-only non-gripper robot envelopes versus frozen public RGB-D points.

No extra perception input, scene/object identities, segmentation or simulator
queries. A hit is a conservative geometric rejection, NOT contact truth. This
supplements (does not replace or weaken) the existing exact gripper-volume
and robot self-collision gates. Unknown/withheld space is not certified free.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.util.geometry import OPTICAL_TO_USD_AXES
from tools.util.x5_kinematics import (
    invert_transform,
    pose_to_transform,
    x5_forward_kinematics,
)
from tools.util import x5_self_collision as robot_model
from tools.util.x5_grasp_geometry import x5_gripper_joint_position_m
from tools.util.x5_gripper_observation import DEPTH_MATCH_M, PIXEL_RADIUS
from tools.util.x5_robot_observation import observed_robot_mask


BUILD = "frozen_public_rgbd_non_gripper_path_envelopes_v4"
ASSET = Path(__file__).with_name("assets") / "x5_path_halfspaces_v1.npz"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INITIAL_WITHHOLD_MARGIN_M = 0.002
# These three bodies already have distinct exact gripper collision/volume
# gates. Do not silently replace their existing contact allowance with an
# any-point test. The camera and camera_base are NOT part of that geometry.
ENDPOINT_GRIPPER_BODIES = frozenset(("link6", "link7", "link8"))


def _rigid_transform(value):
    t = np.asarray(value, dtype=np.float64)
    if (
        t.shape != (4, 4)
        or not np.all(np.isfinite(t))
        or not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
        or np.linalg.det(t[:3, :3]) < 0
    ):
        raise ValueError("finite rigid proper transform required")
    return t


def _xyz(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("finite Nx3 observed world points required")
    return p


def inside_halfspaces(local_points, equations, margin_m=0.0):
    """Exact halfspace membership, bounded memory, no BLAS coefficient dot.

    Progressive plane chunks reject exterior points early; unlike subsampling
    planes, all survivors must pass EVERY halfspace. The explicit sum avoids
    host-BLAS batch/stride-dependent answers seen in the offline diagnostic.
    """
    points = _xyz(local_points)
    equations = np.asarray(equations, dtype=np.float64)
    margins = np.asarray(margin_m, dtype=float)
    if (
        equations.ndim != 2
        or equations.shape[1:] != (4,)
        or len(equations) < 4
        or not np.all(np.isfinite(equations))
        or margins.shape not in ((), (len(points),))
        or not np.isfinite(margins).all()
        or (margins < 0).any()
    ):
        raise ValueError(
            "finite convex equations and nonnegative metric margin required"
        )
    keep = np.arange(len(points), dtype=np.int64)
    for start in range(0, len(equations), 64):
        if not len(keep):
            break
        e = equations[start : start + 64]
        p = points[keep]
        distances = (
            p[:, 0, None] * e[None, :, 0]
            + p[:, 1, None] * e[None, :, 1]
            + p[:, 2, None] * e[None, :, 2]
            + e[None, :, 3]
        )
        threshold = margins if margins.ndim == 0 else margins[keep, None]
        keep = keep[np.all(distances <= threshold + 1e-10, axis=1)]
    return keep


class ObservedPointIndex:
    """Lossless x-sorted broad phase; never voxel-downsamples observed points."""

    def __init__(self, points):
        self.points = _xyz(points).copy()
        self.order = np.argsort(self.points[:, 0], kind="stable")
        self.sorted_x = self.points[self.order, 0]
        for array in (self.points, self.order, self.sorted_x):
            array.setflags(write=False)

    def inside(self, transform, vertices, equations, margin_m=0.0):
        transform = _rigid_transform(transform)
        vertices = _xyz(vertices)
        margins = np.asarray(margin_m, dtype=float)
        if (
            not len(vertices)
            or margins.shape not in ((), (len(self.points),))
            or not np.isfinite(margins).all()
            or (margins < 0).any()
        ):
            raise ValueError("nonempty envelope and finite nonnegative margin required")
        broad_margin = float(margins.max()) if margins.size else 0.0
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        low, high = world.min(0) - broad_margin, world.max(0) + broad_margin
        start = np.searchsorted(self.sorted_x, low[0] - 1e-10, side="left")
        end = np.searchsorted(self.sorted_x, high[0] + 1e-10, side="right")
        possible = self.order[start:end]
        p = self.points[possible]
        possible = possible[np.all((p >= low - 1e-10) & (p <= high + 1e-10), axis=1)]
        local = (self.points[possible] - transform[:3, 3]) @ transform[:3, :3]
        selected_margins = margins if margins.ndim == 0 else margins[possible]
        return possible[inside_halfspaces(local, equations, selected_margins)]


@lru_cache(maxsize=1)
def path_asset():
    metadata, vertices, _ = robot_model.robot_asset()
    source_sha = hashlib.sha256(robot_model.ASSET.read_bytes()).hexdigest()
    equations = {}
    with np.load(ASSET, allow_pickle=False) as data:
        audit = json.loads(str(data["metadata_json"]))
        if (
            audit.get("schema") != "robodojo_static_x5_path_halfspaces"
            or audit.get("version") != 1
            or audit.get("source_sha256") != source_sha
            or audit.get("task_scene_inputs") != []
        ):
            raise ValueError("unbound or unsupported static robot halfspaces")
        for collider in metadata["colliders"]:
            key = collider["key"]
            e = np.ascontiguousarray(data[key], dtype=np.float64)
            if (
                e.ndim != 2
                or e.shape[1:] != (4,)
                or len(e) < 4
                or not np.all(np.isfinite(e))
                or not np.allclose(np.linalg.norm(e[:, :3], axis=1), 1.0, atol=1e-8)
            ):
                raise ValueError("invalid static robot halfspaces")
            if len(inside_halfspaces(vertices[key], e)) != len(vertices[key]):
                raise ValueError("robot halfspaces do not contain their source model")
            e.setflags(write=False)
            equations[key] = e
    return (
        metadata,
        vertices,
        equations,
        {
            "robot_asset_sha256": source_sha,
            "halfspaces_asset_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
            "source_sha256": SOURCE_SHA256,
        },
    )


def sampled_path(current_q, safe_q, final_q):
    qs = [np.asarray(q, dtype=float) for q in (current_q, safe_q, final_q)]
    if any(q.shape != (6,) or not np.all(np.isfinite(q)) for q in qs):
        raise ValueError("six finite observed/IK joints required")
    yield "current", 0.0, qs[0].copy()
    for label, start, end in (("approach", qs[0], qs[1]), ("descent", qs[1], qs[2])):
        count = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(end - start))) / robot_model.PATH_STEP_RAD
                )
            ),
        )
        for i in range(1, count + 1):
            yield label, i / count, start + (end - start) * (i / count)


def withhold_pixel_boundary(depth, initial_mask):
    """One-pass existing pixel/depth uncertainty, NOT semantic dilation.

    Use the same 1-pixel/3-mm constants as the established gripper evidence
    partition. A neighbouring observed depth must agree. No propagation,
    hole-fill or removal of an entire projected robot silhouette is allowed.
    """
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(initial_mask)
    if depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != bool:
        raise ValueError("depth-shaped boolean robot ambiguity mask required")
    valid = np.isfinite(depth) & (depth > 0.02)
    source = mask & valid
    result = source.copy()
    height, width = depth.shape
    for dy in range(-PIXEL_RADIUS, PIXEL_RADIUS + 1):
        for dx in range(-PIXEL_RADIUS, PIXEL_RADIUS + 1):
            ya, yb = max(0, dy), min(height, height + dy)
            xa, xb = max(0, dx), min(width, width + dx)
            if ya >= yb or xa >= xb:
                continue
            a = (slice(ya, yb), slice(xa, xb))
            b = (slice(ya - dy, yb - dy), slice(xa - dx, xb - dx))
            with np.errstate(invalid="ignore"):
                match = np.abs(depth[a] - depth[b]) <= DEPTH_MATCH_M
            result[a] |= valid[a] & source[b] & match
    return result


def rgbd_point_uncertainty(x, y, depth, intrinsic):
    """Metric bound of the existing +/-1 pixel, +/-3 mm depth evidence box.

    For p=z*((u-cx)/fx,(v-cy)/fy,1), the triangle inequality gives
    dz*||ray|| + (z+dz)*pixel_radius*||(1/fx,1/fy)||. Camera rotation
    preserves this Euclidean bound. No class/scene/click-specific parameter.
    Used ONLY for initial robot/contact-ambiguous evidence, not to erase
    collision points throughout the candidate path or relax its zero margin.
    """
    k = np.asarray(intrinsic, dtype=float)
    ray_norm = np.sqrt(
        ((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0
    )
    return DEPTH_MATCH_M * ray_norm + (
        depth + DEPTH_MATCH_M
    ) * PIXEL_RADIUS * math.hypot(1.0 / k[0, 0], 1.0 / k[1, 1])


class FrozenRGBDPathGuard:
    """One lossless whole-view point set, reused across candidate paths."""

    def __init__(self, snapshot):
        self.metadata, self.vertices, self.equations, hashes = path_asset()
        depth = np.asarray(snapshot.depth, dtype=float)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            raise ValueError("frozen depth must be HxW")
        k = np.asarray(snapshot.intrinsic_matrix, dtype=float)
        transform = _rigid_transform(snapshot.extrinsic_matrix)
        if (
            k.shape != (3, 3)
            or not np.all(np.isfinite(k))
            or k[0, 0] <= 0
            or k[1, 1] <= 0
        ):
            raise ValueError("finite calibrated intrinsics required")
        y, x = np.nonzero(np.isfinite(depth) & (depth > 0.02))
        if not len(x):
            raise ValueError("no valid public depth for path guard")
        z = depth[y, x]
        optical = np.column_stack(
            ((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z)
        )
        points = (optical * OPTICAL_TO_USD_AXES) @ transform[:3, :3].T + transform[
            :3, 3
        ]
        all_points = ObservedPointIndex(points)
        uncertainty = np.maximum(
            INITIAL_WITHHOLD_MARGIN_M, rgbd_point_uncertainty(x, y, z, k)
        )
        self.current_q, self.bases, self.fingers = {}, {}, {}
        withheld = np.zeros(len(points), dtype=bool)
        # Both arms' observed robot/contact-ambiguous samples are withheld,
        # not turned into an assertion of free space behind a robot silhouette.
        for arm in ("left", "right"):
            q = np.asarray(snapshot.robot[f"arm_{arm}_qpos"], dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                raise ValueError("capture-time arm state required")
            pose = snapshot.robot[f"eef_{arm}"]
            self.current_q[arm] = q.copy()
            self.bases[arm] = _rigid_transform(
                pose_to_transform(np.r_[pose["pos"], pose["quat"]])
                @ invert_transform(x5_forward_kinematics(q))
            )
            gripper = np.asarray(
                snapshot.robot[f"gripper_{arm}_qpos"], dtype=float
            ).reshape(-1)
            if gripper.size != 1 or not np.isfinite(gripper[0]):
                raise ValueError("finite public gripper command state required")
            self.fingers[arm] = x5_gripper_joint_position_m(gripper)
            transforms = robot_model.link_transforms(q, self.fingers[arm])
            for c in self.metadata["colliders"]:
                key = c["key"]
                indices = all_points.inside(
                    self.bases[arm] @ transforms[c["body"]],
                    self.vertices[key],
                    self.equations[key],
                    uncertainty,
                )
                withheld[indices] = True
        initial_count = int(withheld.sum())
        visual_mask, visual_audit = observed_robot_mask(
            snapshot, self.bases, self.current_q, self.fingers
        )
        visual_added = int((visual_mask[y, x] & ~withheld).sum())
        withheld |= visual_mask[y, x]
        before_pixel_boundary = int(withheld.sum())
        pixel_mask = np.zeros(depth.shape, dtype=bool)
        pixel_mask[y[withheld], x[withheld]] = True
        withheld = withhold_pixel_boundary(depth, pixel_mask)[y, x]
        self.index = ObservedPointIndex(points[~withheld])
        if not len(self.index.points):
            raise ValueError("no_unambiguous_scene_depth_after_robot_withholding")
        self.uv = np.column_stack((x, y))[~withheld]
        self.uv.setflags(write=False)
        # Explicit monotonicity check, including fingers which the new path
        # test itself excludes. Numerical ambiguity must fail, not pass clear.
        for arm in ("left", "right"):
            transforms = robot_model.link_transforms(
                self.current_q[arm], self.fingers[arm]
            )
            for c in self.metadata["colliders"]:
                key = c["key"]
                if len(
                    self.index.inside(
                        self.bases[arm] @ transforms[c["body"]],
                        self.vertices[key],
                        self.equations[key],
                    )
                ):
                    raise ValueError("initial_self_withholding_invariant_failed")
        self.audit = dict(
            build=BUILD,
            **hashes,
            whole_view=True,
            pixel_stride=1,
            raw_point_count=len(points),
            retained_observed_point_count=len(self.index.points),
            initial_self_or_contact_ambiguous_point_count=int(withheld.sum()),
            initial_envelope_point_count=initial_count,
            depth_matched_visual_ambiguous_point_count=visual_added,
            visual_self_evidence=visual_audit,
            pixel_boundary_ambiguous_point_count=int(withheld.sum())
            - before_pixel_boundary,
            pixel_boundary_radius=PIXEL_RADIUS,
            pixel_boundary_depth_match_m=DEPTH_MATCH_M,
            initial_withhold_margin_m=INITIAL_WITHHOLD_MARGIN_M,
            initial_withhold_rule="max(existing_2mm_floor,calibrated_1pixel_3mm_depth_uncertainty_bound)",
            initial_per_point_uncertainty_quantiles_m=np.quantile(
                uncertainty, [0.0, 0.5, 1.0]
            ).tolist(),
            initial_self_withholding_invariant_passed=True,
            scope="sampled_non_gripper_robot_envelopes_vs_frozen_observed_points",
            guarded_bodies=[
                c["body"]
                for c in self.metadata["colliders"]
                if c["body"] not in ENDPOINT_GRIPPER_BODIES
            ],
            gripper_bodies_use_existing_volume_gate=sorted(ENDPOINT_GRIPPER_BODIES),
            path_max_joint_step_rad=robot_model.PATH_STEP_RAD,
            query_margin_m=0.0,
            allowed_inputs=[
                "frozen_depth",
                "camera_calibration",
                "capture_time_proprioception",
                "static_official_X5_model",
            ],
            forbidden_inputs_used=[],
            object_identity_or_click_filter_used=False,
            assumed_static_scene=True,
            observed_points_are_not_solid_volume=True,
            contact_truth_used=False,
            occluded_or_withheld_space_certified_free=False,
            continuous_swept_volume_proven=False,
            other_arm_trajectory_checked=False,
            actual_controller_trajectory_predicted=False,
        )

    def check(self, arm, safe_q, final_q):
        if arm not in self.current_q:
            raise ValueError("unknown X5 arm")
        sample_count = 0
        for stage, fraction, q in sampled_path(self.current_q[arm], safe_q, final_q):
            sample_count += 1
            transforms = robot_model.link_transforms(q, self.fingers[arm])
            for c in self.metadata["colliders"]:
                if c["body"] in ENDPOINT_GRIPPER_BODIES:
                    continue
                key = c["key"]
                indices = self.index.inside(
                    self.bases[arm] @ transforms[c["body"]],
                    self.vertices[key],
                    self.equations[key],
                )
                if len(indices):
                    pixels = self.uv[indices]
                    return dict(
                        build=BUILD,
                        ok=False,
                        reason="observed_surface_in_robot_path_envelope",
                        arm=arm,
                        sample_count=sample_count,
                        stage=stage,
                        fraction=fraction,
                        body=c["body"],
                        collider=key,
                        outer_bound_only=c["outer_bound_only"],
                        observed_point_count=len(indices),
                        pixel_bounds=[pixels.min(0).tolist(), pixels.max(0).tolist()],
                        sample_pixels=pixels[:: max(1, len(pixels) // 8)][:8].tolist(),
                        joints=q.tolist(),
                        contact_truth_used=False,
                    )
        return dict(
            build=BUILD,
            ok=True,
            reason="sampled_observed_non_gripper_envelopes_clear",
            arm=arm,
            sample_count=sample_count,
            contact_truth_used=False,
            physical_success_proven=False,
            occluded_space_certified_free=False,
        )
