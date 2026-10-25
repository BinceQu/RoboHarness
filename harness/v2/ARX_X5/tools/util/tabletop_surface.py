"""Conservative tabletop classification from public RGB-D, without object truth.

A height tolerance used for optical flow (several mm) is inappropriate here:
it would absorb a thin key or cloth into the table. Fit the dominant broad
horizontal plane tightly, then require an entire small pixel neighbourhood to
agree. Unresolved boundaries and noisy planes deliberately remain dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

BUILD = "rgbd_tabletop_pixel_center_submillimetre_anchor_v2"
PIXEL_CENTER_OFFSET = 0.5


@dataclass
class TabletopSurface:
    height_m: np.ndarray
    valid: np.ndarray
    fixed_mask: np.ndarray
    report: dict[str, Any]

    @classmethod
    def from_frame(cls, frame, *, reference_plane=None) -> "TabletopSurface":
        depth = np.asarray(frame.depth_linear, dtype=np.float64)
        intr = frame.intrinsics
        yy, xx = np.indices(depth.shape, dtype=np.float64)
        # RTX linear-depth samples lie at pixel centres. The public K uses
        # width/2,height/2, i.e. image-edge coordinates. Integer array indices
        # without +.5 invent a small tilted plane that differs across cameras.
        # This correction is local to surface classification; do not silently
        # change the existing public tracker/IK coordinate convention here.
        camera = np.stack(
            (
                (xx + PIXEL_CENTER_OFFSET - intr.cx) / intr.fx * depth,
                -(yy + PIXEL_CENTER_OFFSET - intr.cy) / intr.fy * depth,
                -depth,
            ),
            axis=-1,
        )
        transform = np.asarray(frame.camera_to_robot_base, dtype=np.float64)
        xyz = camera @ transform[:3, :3].T + transform[:3, 3]
        valid = np.isfinite(xyz).all(axis=-1) & np.isfinite(depth) & (depth > 0)
        report = dict(
            build=BUILD,
            source="public_rgbd_dominant_horizontal_plane",
            confident=False,
            uses_object_truth=False,
            uses_segmentation=False,
            depth_pixel_center_offset=PIXEL_CENTER_OFFSET,
        )
        blank = np.zeros(depth.shape, dtype=bool)

        def reject(reason):
            report["reason"] = reason
            return cls(np.full(depth.shape, np.nan), valid, blank, report)

        if reference_plane is not None:
            if not reference_plane.get("confident"):
                return reject("head_support_plane_unavailable")
            a, b, c = reference_plane["plane_coefficients_z_from_xy"]
            residual = (
                xyz[:, :, 2] - a * xyz[:, :, 0] - b * xyz[:, :, 1] - c
            ) / np.sqrt(1 + a * a + b * b)
            report = dict(
                reference_plane, source="same_episode_head_rgbd_support_plane"
            )
            support = valid & (np.abs(residual) <= report["table_tolerance_m"])
            fixed = cv2.erode(
                support.astype(np.uint8),
                np.ones((7, 7), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            report["fixed_pixel_count"] = int(fixed.sum())
            return cls(residual, valid, fixed, report)

        points = xyz[valid][::4]
        if len(points) < 500:
            return reject("insufficient_depth")
        # No task-specific height or object dimensions enter this estimator.
        bins, counts = np.unique(
            np.rint(points[:, 2] / 0.0005).astype(np.int64), return_counts=True
        )
        seed = bins[int(np.argmax(counts))] * 0.0005
        coefficients = np.array([0.0, 0.0, seed])
        design = np.column_stack((points[:, :2], np.ones(len(points))))
        for threshold in (0.0008, 0.0004, 0.0002, 0.00015, 0.00006, 0.00002):
            keep = np.abs(points[:, 2] - design @ coefficients) <= threshold
            if np.count_nonzero(keep) < 500:
                return reject("insufficient_plane_inliers")
            coefficients = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)[0]
        normal = np.r_[-coefficients[:2], 1.0]
        normal /= np.linalg.norm(normal)
        residual = (
            xyz[:, :, 2]
            - xyz[:, :, 0] * coefficients[0]
            - xyz[:, :, 1] * coefficients[1]
            - coefficients[2]
        ) * normal[2]
        errors = np.abs((points[:, 2] - design @ coefficients)[keep])
        p95 = float(np.quantile(errors, 0.95))
        tilt = float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1))))
        # A noisy plane is not grounds for widening the band until thin
        # objects disappear. Keep those registrations on the dynamic path.
        report.update(
            plane_coefficients_z_from_xy=coefficients.tolist(),
            normal_robot_base=normal.tolist(),
            residual_p95_m=p95,
            normal_tilt_deg=tilt,
        )
        if p95 > 0.00004 or tilt > 3.0:
            return reject("plane_not_precise_enough_for_thin_objects")
        tolerance = max(0.00002, 3 * p95)
        on_plane = valid & (np.abs(residual) <= tolerance)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            on_plane.astype(np.uint8), 8
        )
        if count < 2:
            return reject("no_connected_support")
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        support = labels == label
        support_xyz = xyz[support]
        fraction = float(np.mean(support))
        span = np.ptp(support_xyz[:, :2], axis=0)
        report.update(
            support_fraction=fraction,
            support_span_xy_m=span.tolist(),
            table_tolerance_m=tolerance,
        )
        if fraction < 0.12 or float(np.min(span)) < 0.60:
            return reject("plane_not_a_broad_support_surface")
        # Three pixels of clearance also protects subpixel clicks whose depth
        # sample falls just outside a narrow object or straddles its silhouette.
        # The largest connected patch establishes that this is the broad
        # support plane. An arm/cloth can split visible table into islands;
        # those islands still belong to this same precise geometric plane.
        # Erode every on-plane patch so narrow boundaries remain uncertain.
        fixed = cv2.erode(
            on_plane.astype(np.uint8),
            np.ones((7, 7), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        report.update(
            confident=True,
            reason="broad_precise_support_plane",
            boundary_clearance_px=3,
            fixed_pixel_count=int(fixed.sum()),
        )
        return cls(residual, valid, fixed, report)

    def classify(self, pixel_uv) -> dict[str, Any]:
        pixel = np.asarray(pixel_uv, dtype=float).reshape(2)
        height, width = self.valid.shape
        result = dict(
            tracking_mode="dynamic",
            classification="uncertain",
            fixed=False,
            plane=dict(self.report),
        )
        if not np.isfinite(pixel).all() or not (
            0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1
        ):
            result["reason"] = "pixel_outside_image"
            return result
        x, y = np.rint(pixel).astype(int)
        if not self.report["confident"]:
            result["reason"] = self.report["reason"]
        elif not self.valid[y, x]:
            result["reason"] = "invalid_depth_at_clicked_pixel"
        else:
            z = float(self.height_m[y, x])
            result["height_above_table_m"] = z
            if self.fixed_mask[y, x]:
                result.update(
                    tracking_mode="fixed_tabletop",
                    classification="tabletop",
                    fixed=True,
                    reason="on_plane_with_clear_neighbourhood",
                )
            elif abs(z) > self.report["table_tolerance_m"]:
                result.update(
                    classification="non_table_surface",
                    reason="separated_from_table_plane",
                )
            else:
                result["reason"] = "table_object_boundary_or_insufficient_neighbourhood"
        return result
