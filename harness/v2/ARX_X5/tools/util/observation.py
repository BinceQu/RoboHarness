"""Thread-safe adapter for the public RoboDojo observation dictionary."""

from __future__ import annotations

from copy import deepcopy
import math
import os
import threading
import time
import uuid
from typing import Any, Mapping

import numpy as np


DEFAULT_CAMERA_ROLES = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}

_ALLOWED_CAMERA_FIELDS = frozenset(
    {
        "color",
        "depth",
        "intrinsic_matrix",
        "extrinsic_matrix",
        "extrinsics_matrix",
        "shape",
    }
)
_ALLOWED_STATE_FIELDS = frozenset(
    {
        "left_arm_joint_state",
        "left_ee_joint_state",
        "left_ee_pose",
        "right_arm_joint_state",
        "right_ee_joint_state",
        "right_ee_pose",
    }
)


class ObservationAdapter:
    """Owns only observations delivered through the XPolicyLab boundary."""

    def __init__(self, camera_roles: Mapping[str, str] | None = None) -> None:
        self.camera_roles = {**DEFAULT_CAMERA_ROLES, **dict(camera_roles or {})}
        try:
            max_age_s = float(
                os.environ.get("ROBODOJO_INTERFACE_OBSERVATION_MAX_AGE_S", "5")
            )
        except ValueError:
            max_age_s = 5.0
        self.max_age_s = (
            max_age_s if math.isfinite(max_age_s) and max_age_s > 0 else 5.0
        )
        self._lock = threading.RLock()
        self._observation: dict[str, Any] | None = None
        self._sequence = 0
        self._episode_id = uuid.uuid4().hex
        self._observation_updated_at = 0.0
        self._heartbeat_at = 0.0

    def reset(self) -> None:
        with self._lock:
            self._observation = None
            self._sequence = 0
            self._episode_id = uuid.uuid4().hex
            self._observation_updated_at = 0.0
            self._heartbeat_at = 0.0

    def ingest(self, observation: Mapping[str, Any]) -> int:
        if not isinstance(observation, Mapping):
            raise TypeError("RoboDojo observation must be a mapping")
        vision = observation.get("vision", {})
        state = observation.get("state", {})
        if not isinstance(vision, Mapping) or not isinstance(state, Mapping):
            raise ValueError("observation vision and state fields must be mappings")
        allowed_camera_names = frozenset(self.camera_roles.values())
        sanitized_vision: dict[str, dict[str, Any]] = {}
        for camera_name, camera_data in vision.items():
            if camera_name not in allowed_camera_names or not isinstance(
                camera_data, Mapping
            ):
                continue
            sanitized_vision[str(camera_name)] = {
                str(key): deepcopy(value)
                for key, value in camera_data.items()
                if key in _ALLOWED_CAMERA_FIELDS
            }
        sanitized_state = {
            str(key): deepcopy(value)
            for key, value in state.items()
            if key in _ALLOWED_STATE_FIELDS
        }
        action = observation.get("action", {})
        sanitized_action = (
            {
                str(key): deepcopy(value)
                for key, value in action.items()
                if key in _ALLOWED_STATE_FIELDS
            }
            if isinstance(action, Mapping)
            else {}
        )
        sanitized = {
            "instruction": deepcopy(observation.get("instruction", "")),
            "vision": sanitized_vision,
            "state": sanitized_state,
            "action": sanitized_action,
        }
        with self._lock:
            self._observation = sanitized
            self._sequence += 1
            now = time.monotonic()
            self._observation_updated_at = now
            self._heartbeat_at = now
            return self._sequence

    def touch(self) -> None:
        """Record one evaluator policy poll without changing observation data."""
        with self._lock:
            if self._observation is not None:
                self._heartbeat_at = time.monotonic()

    def connected(self) -> bool:
        with self._lock:
            return self._observation is not None and (
                time.monotonic() - self._heartbeat_at <= self.max_age_s
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            heartbeat_age_s = (
                None
                if self._observation is None
                else max(0.0, time.monotonic() - self._heartbeat_at)
            )
            observation_age_s = (
                None
                if self._observation is None
                else max(0.0, time.monotonic() - self._observation_updated_at)
            )
            return {
                "connected": (
                    heartbeat_age_s is not None and heartbeat_age_s <= self.max_age_s
                ),
                "observation_available": self._observation is not None,
                "observation_age_s": observation_age_s,
                "heartbeat_age_s": heartbeat_age_s,
                "observation_max_age_s": self.max_age_s,
                "sequence": self._sequence,
                "episode_id": self._episode_id,
                "updated_at_monotonic": self._observation_updated_at,
            }

    def observation(self) -> dict[str, Any]:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("RoboDojo evaluator observation is unavailable")
            return deepcopy(self._observation)

    def instruction(self) -> str:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("RoboDojo evaluator observation is unavailable")
            value = deepcopy(self._observation.get("instruction", ""))
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item) for item in value)
        return str(value or "")

    def state(self, key: str, *, required: bool = True) -> np.ndarray | None:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("RoboDojo evaluator observation is unavailable")
            value = deepcopy(self._observation.get("state", {}).get(key))
        if value is None:
            if required:
                raise KeyError(f"RoboDojo observation is missing state.{key}")
            return None
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"state.{key} contains non-finite values")
        return result

    def action_state(self, key: str) -> np.ndarray | None:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("RoboDojo evaluator observation is unavailable")
            value = deepcopy(self._observation.get("action", {}).get(key))
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        return result if np.all(np.isfinite(result)) else None

    def camera(self, role: str, *, require_depth: bool = False) -> dict[str, Any]:
        camera_name = self.camera_roles.get(str(role), str(role))
        with self._lock:
            if self._observation is None:
                raise RuntimeError("RoboDojo evaluator observation is unavailable")
            data = self._observation.get("vision", {}).get(camera_name)
            if not isinstance(data, Mapping):
                raise KeyError(f"RoboDojo observation is missing vision.{camera_name}")
            selected = {
                key: deepcopy(value)
                for key, value in data.items()
                if key in _ALLOWED_CAMERA_FIELDS
            }
            sequence = self._sequence
            episode_id = self._episode_id
        color = selected.get("color")
        if color is None:
            raise KeyError(f"vision.{camera_name}.color is unavailable")
        rgb = np.asarray(color)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"vision.{camera_name}.color must be HxWx3")
        depth_value = selected.get("depth")
        depth = (
            None if depth_value is None else np.asarray(depth_value, dtype=np.float32)
        )
        if depth is not None and depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if require_depth and depth is None:
            raise KeyError(f"vision.{camera_name}.depth is unavailable")
        if depth is not None and depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"vision.{camera_name} RGB/depth mismatch: {rgb.shape[:2]} vs {depth.shape}"
            )
        intrinsic_value = selected.get("intrinsic_matrix")
        extrinsic_value = selected.get(
            "extrinsic_matrix", selected.get("extrinsics_matrix")
        )
        intrinsic = (
            None
            if intrinsic_value is None
            else np.asarray(intrinsic_value, dtype=np.float64).reshape(3, 3)
        )
        extrinsic = (
            None
            if extrinsic_value is None
            else np.asarray(extrinsic_value, dtype=np.float64).reshape(4, 4)
        )
        if require_depth and (intrinsic is None or extrinsic is None):
            raise KeyError(
                f"vision.{camera_name} depth tools require intrinsic_matrix and extrinsic_matrix"
            )
        return {
            "role": str(role),
            "camera_name": camera_name,
            "rgb": rgb[:, :, :3].copy(),
            "depth": None if depth is None else depth.copy(),
            "intrinsic_matrix": None if intrinsic is None else intrinsic.copy(),
            "extrinsic_matrix": None if extrinsic is None else extrinsic.copy(),
            "shape": list(rgb.shape),
            "sequence": sequence,
            "episode_id": episode_id,
        }

    def camera_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for role in self.camera_roles:
            try:
                frames[role] = self.camera(role)["rgb"]
            except (KeyError, ValueError):
                continue
        return frames

    def camera_depth_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for role in self.camera_roles:
            try:
                depth = self.camera(role)["depth"]
            except (KeyError, ValueError):
                continue
            if depth is not None:
                frames[role] = depth
        return frames
