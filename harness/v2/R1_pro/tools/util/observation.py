"""One coherent allowed observation, with native RGB-D and measured joint state."""
from dataclasses import dataclass
import time

import numpy as np

from .model import ACTION_SLICES, PROPRIO_SLICES, LOCKED_INDICES, clip_position_targets

ALLOWED_SUFFIXES = ("::rgb", "::depth_linear", "::proprio", "::cam_rel_poses")


def filter_observation(raw):
    return {str(key): value for key, value in raw.items()
            if str(key) in ("task_id", "need_new_action") or str(key).endswith(ALLOWED_SUFFIXES)}


def camera_role(key):
    key = key.lower()
    for role, names in (("left_wrist", ("left_realsense", "left_wrist")),
                        ("right_wrist", ("right_realsense", "right_wrist")),
                        ("head", ("zed_link", "head"))):
        if any(name in key for name in names):
            return role
    return None


@dataclass
class Snapshot:
    observation: dict
    sequence: int
    received_ts: float
    episode_id: str


class Observation:
    _official_camera_arrays_owned = True

    def __init__(self):
        self.current = Snapshot({}, 0, 0, "")

    def update(self, payload):
        self.current = Snapshot(**payload)

    def snapshot(self):
        return self.current

    borrow_snapshot_for_reading = snapshot

    def observation_metadata(self):
        return self.current.sequence, self.current.received_ts

    def status(self):
        return {"sequence": self.current.sequence, "received_ts": self.current.received_ts,
                "episode_id": self.current.episode_id,
                "age_s": time.time() - self.current.received_ts}

    def proprio_vector(self):
        value = next((value for key, value in self.current.observation.items() if key.endswith("::proprio")), None)
        if value is None:
            raise ValueError("Evaluator has not provided proprioception")
        result = np.asarray(value, dtype=np.float32).reshape(-1)
        if result.shape != (65,) or not np.isfinite(result).all():
            raise ValueError("Expected 65 finite proprioception values")
        return result.copy()

    def hold_action(self):
        proprio = self.proprio_vector()
        action = np.zeros(27, dtype=np.float32)
        for group in ("trunk", "arm_left", "arm_right"):
            action[ACTION_SLICES[group]] = proprio[PROPRIO_SLICES[group + "_qpos"]]
        action[list(LOCKED_INDICES)] = 0
        return clip_position_targets(action)

    def eef_pose(self):
        proprio = self.proprio_vector()
        return {side: {"pos": proprio[PROPRIO_SLICES[f"eef_{side}_pos"]].tolist(),
                       "quat": proprio[PROPRIO_SLICES[f"eef_{side}_quat"]].tolist()}
                for side in ("left", "right")}

    def camera_rgb(self, role):
        value = next((value for key, value in self.current.observation.items()
                      if key.endswith("::rgb") and camera_role(key) == role), None)
        if value is None:
            return None
        value = np.asarray(value)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise ValueError(f"{role} RGB must be uint8 RGB or RGBA")
        return value[..., :3].copy()

    def camera_frame(self, role):
        rgb = self.camera_rgb(role)
        return None if rgb is None else rgb[..., ::-1].copy()

    def camera_depth_frame(self, role):
        value = next((value for key, value in self.current.observation.items()
                      if key.endswith("::depth_linear") and camera_role(key) == role), None)
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float32).squeeze()
        if result.ndim != 2:
            raise ValueError(f"{role} depth must be a 2D metric depth image")
        return result.copy()

    def camera_relative_poses(self):
        value = next((value for key, value in self.current.observation.items() if key.endswith("::cam_rel_poses")), None)
        if value is None:
            return {}
        poses = np.asarray(value, dtype=float).reshape(-1)
        if poses.shape != (21,) or not np.isfinite(poses).all():
            raise ValueError("Camera-relative poses require 21 finite values")
        return {role: {"pos": poses[i * 7:i * 7 + 3].tolist(), "quat": poses[i * 7 + 3:i * 7 + 7].tolist()}
                for i, role in enumerate(("left_wrist", "right_wrist", "head"))}

    def navigation_map_snapshot(self):
        return None

    spatial_map_snapshot = navigation_map_snapshot
