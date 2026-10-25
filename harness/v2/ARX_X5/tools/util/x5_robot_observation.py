"""Depth-matched robot/contact-ambiguous evidence from static X5 visuals.

Only existing public depth, calibration and proprioception are used. This
module never predicts hidden scene occupancy, declares withheld rays free or
reads segmentation/contact truth. Visual samples do not replace collision
envelopes. Command-versus-finger-position ambiguity uses the existing public
depth registration, while non-gripper links use capture-time observed joints.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.util import x5_self_collision as robot_model
from tools.util.x5_gripper_observation import matched_depth_mask, observed_gripper_mask

BUILD = "public_depth_registered_static_x5_robot_visual_samples_v1"
ASSET = Path(__file__).with_name("assets") / "x5_visual_surface_samples_v1.npz"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def visual_asset():
    with np.load(ASSET, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if (
            metadata.get("schema") != "robodojo_static_x5_visual_surface_samples"
            or metadata.get("version") != 1
            or metadata.get("task_scene_inputs") != []
            or metadata.get("robot_asset_sha256")
            != hashlib.sha256(robot_model.ASSET.read_bytes()).hexdigest()
        ):
            raise ValueError("unbound static robot visual asset")
        arrays = {}
        for component in metadata["components"]:
            p = np.asarray(data[component["key"]], dtype=float)
            if p.ndim != 2 or p.shape[1:] != (3,) or not np.isfinite(p).all():
                raise ValueError("invalid static robot visual samples")
            p.setflags(write=False)
            arrays[component["key"]] = p
    return metadata, arrays


def projected_depth(points_world, shape, intrinsic, extrinsic):
    """Nearest model sample per public camera pixel; no scene depth input."""
    k, camera = np.asarray(intrinsic, dtype=float), np.asarray(extrinsic, dtype=float)
    points = (np.asarray(points_world, dtype=float) - camera[:3, 3]) @ camera[:3, :3]
    z = -points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-5)
    points, z = points[valid], z[valid]
    uv = np.column_stack(
        (k[0, 0] * points[:, 0] / z + k[0, 2], -k[1, 1] * points[:, 1] / z + k[1, 2])
    )
    h, w = shape
    valid = (
        (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    )
    pixels = np.rint(uv[valid]).astype(np.int64)
    image = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(image, pixels[:, 1] * w + pixels[:, 0], z[valid])
    return image.reshape(h, w)


def observed_robot_mask(snapshot, bases, current_q, fingers):
    metadata, points = visual_asset()
    # Existing finger-depth registration has an applicability gate and reports
    # a command proxy fallback. Never claim its fit is a measured contact.
    mask, gripper_audit = observed_gripper_mask(snapshot)
    added, arms = 0, {}
    for arm in ("left", "right"):
        transforms = robot_model.link_transforms(current_q[arm], fingers[arm])
        world = []
        for component in metadata["components"]:
            if component["body"] in ("link6", "link7", "link8"):
                continue
            t = bases[arm] @ transforms[component["body"]]
            world.append(points[component["key"]] @ t[:3, :3].T + t[:3, 3])
        model_depth = projected_depth(
            np.concatenate(world),
            snapshot.depth.shape,
            snapshot.intrinsic_matrix,
            snapshot.extrinsic_matrix,
        )
        match = matched_depth_mask(snapshot.depth, model_depth)
        added += int((match & ~mask).sum())
        mask |= match
        arms[arm] = dict(
            model_pixel_count=int(np.isfinite(model_depth).sum()),
            depth_matched_pixel_count=int(match.sum()),
        )
    return mask, dict(
        build=BUILD,
        source_sha256=SOURCE_SHA256,
        visual_asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        gripper_registration=gripper_audit,
        non_gripper_visual_matches=arms,
        additional_non_gripper_pixels=added,
        total_withheld_pixel_count=int(mask.sum()),
        sampling_pitch_m=metadata["sampling_pitch_m"],
        depth_match_tolerance_m=0.003,
        pixel_radius=1,
        classification="depth_matched_robot_surface_or_contact_ambiguity",
        unknown_space_declared_free=False,
        simulator_state_or_contact_truth_used=False,
        used_segmentation_or_object_identity=False,
        observed_joint_positions_modified=False,
    )
