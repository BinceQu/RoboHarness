"""Shared robot functions used by: capture_head_camera, capture_left_wrist_camera, capture_right_wrist_camera, move_tracked_point, plan_press_point.

All commands and geometry execute locally; no macro RPC or external project.
"""
from __future__ import annotations

from dataclasses import dataclass
from tools.util.geometry.capabilities import TOOL_CAPABILITIES
from tools.util.geometry.contract import PROPRIO_DIM
from tools.util.geometry.contract import PROPRIO_SLICES
from tools.util.geometry.tracked_object_distance import tracker_rgb_as_uint8
from typing import Any
from typing import Mapping
import cv2
import math
import numpy as np


def _capture_robot_state(
    ctx,
    *,
    proprio: Any | None = None,
) -> dict[str, Any]:
    adapter = getattr(ctx.world, "_official_adapter", None)
    if proprio is None:
        proprio = None if adapter is None else adapter.proprio_vector()
    else:
        proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio.size < PROPRIO_DIM or not np.all(
            np.isfinite(proprio[:PROPRIO_DIM])
        ):
            raise ValueError("frozen evaluator proprioception is invalid")
        proprio = proprio[:PROPRIO_DIM]

    if proprio is not None:
        def observed_values(name: str) -> list[float]:
            return np.asarray(
                proprio[PROPRIO_SLICES[name]],
                dtype=np.float64,
            ).astype(float).tolist()

        def observed_eef(side: str) -> dict[str, Any]:
            return ctx.world.local_pose_from_robot_relative(
                observed_values(f"eef_{side}_pos"),
                observed_values(f"eef_{side}_quat"),
            )

        observed = {
            "eef_left": observed_eef("left"),
            "eef_right": observed_eef("right"),
            "trunk_qpos": observed_values("trunk_qpos"),
            "trunk_qvel": observed_values("trunk_qvel"),
            "base_qvel": observed_values("base_qvel"),
            "arm_left_qpos": observed_values("arm_left_qpos"),
            "arm_left_qvel": observed_values("arm_left_qvel"),
            "arm_right_qpos": observed_values("arm_right_qpos"),
            "arm_right_qvel": observed_values("arm_right_qvel"),
            "gripper_left_qpos": observed_values("gripper_left_qpos"),
            "gripper_left_qvel": observed_values("gripper_left_qvel"),
            "gripper_right_qpos": observed_values("gripper_right_qpos"),
            "gripper_right_qvel": observed_values("gripper_right_qvel"),
        }
    else:
        observed = {
            "eef_left": ctx.world.eef_pose("left"),
            "eef_right": ctx.world.eef_pose("right"),
            "trunk_qpos": ctx.world.trunk_qpos().astype(float).tolist(),
            "trunk_qvel": [],
            "base_qvel": ctx.world.base_qvel().astype(float).tolist(),
            "arm_left_qpos": ctx.world.arm_qpos_list("left"),
            "arm_left_qvel": ctx.world.arm_qvel_list("left"),
            "arm_right_qpos": ctx.world.arm_qpos_list("right"),
            "arm_right_qvel": ctx.world.arm_qvel_list("right"),
            "gripper_left_qpos": ctx.world.gripper_qpos_list("left"),
            "gripper_left_qvel": ctx.world.gripper_qvel_list("left"),
            "gripper_right_qpos": ctx.world.gripper_qpos_list("right"),
            "gripper_right_qvel": ctx.world.gripper_qvel_list("right"),
        }
    return {
        "base_pose": ctx.world.robot_pose().as_dict(),
        **observed,
        "motion_epoch": int(ctx.world.motion_epoch()),
        "episode_id": str(ctx.world.episode_id()),
        "frame": "local_command_odometry",
        "proprio_source": (
            "frozen_evaluator_observation"
            if proprio is not None
            else "current_evaluator_observation"
        ),
    }



@dataclass(frozen=True)
class _FrozenWristObservation:
    frame: np.ndarray | None
    depth: np.ndarray | None
    camera_relative_pose: dict[str, list[float]] | None
    proprio: np.ndarray | None
    sequence: int | None
    received_ts: float | None



def _snapshot_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)



def _wrist_camera_role(key: str) -> str | None:
    lowered = str(key).lower()
    if "left_realsense" in lowered or "left_wrist" in lowered:
        return "left_wrist"
    if "right_realsense" in lowered or "right_wrist" in lowered:
        return "right_wrist"
    if "zed_link" in lowered or "head" in lowered:
        return "head"
    return None



def _freeze_wrist_observation(
    adapter: Any,
    role: str,
) -> _FrozenWristObservation:
    reader = getattr(adapter, "borrow_snapshot_for_reading", None)
    if not callable(reader):
        reader = getattr(adapter, "snapshot", None)
    if not callable(reader):
        raise RuntimeError(
            "official evaluator adapter does not expose an atomic observation snapshot"
        )
    snapshot = reader()
    observation = getattr(snapshot, "observation", None)
    if not isinstance(observation, Mapping):
        raise RuntimeError("official evaluator observation snapshot is invalid")

    rgb_value = next(
        (
            value
            for key, value in observation.items()
            if str(key).endswith("::rgb") and _wrist_camera_role(key) == role
        ),
        None,
    )
    frame = None
    if rgb_value is not None:
        canonical_rgb = tracker_rgb_as_uint8(
            _snapshot_array(rgb_value),
            "rgb",
        )
        frame = cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2BGR)

    depth_value = next(
        (
            value
            for key, value in observation.items()
            if str(key).endswith("::depth_linear")
            and _wrist_camera_role(key) == role
        ),
        None,
    )
    depth = None
    if depth_value is not None:
        candidate = _snapshot_array(depth_value, dtype=np.float32).squeeze()
        if candidate.ndim >= 2 and candidate.size:
            depth = candidate

    proprio_value = next(
        (
            value
            for key, value in observation.items()
            if str(key).endswith("::proprio")
        ),
        None,
    )
    proprio = None
    if proprio_value is not None:
        candidate = _snapshot_array(proprio_value, dtype=np.float64).reshape(-1)
        if candidate.size >= PROPRIO_DIM:
            proprio = candidate[:PROPRIO_DIM].copy()

    camera_relative_pose = None
    pose_value = next(
        (
            value
            for key, value in observation.items()
            if str(key).endswith("::cam_rel_poses")
        ),
        None,
    )
    if pose_value is not None:
        poses = _snapshot_array(pose_value, dtype=np.float64).reshape(-1)
        role_index = {
            "left_wrist": 0,
            "right_wrist": 1,
            "head": 2,
        }.get(str(role))
        if role_index is not None:
            start = 7 * role_index
            if poses.size >= start + 7:
                camera_relative_pose = {
                    "pos": poses[start : start + 3].astype(float).tolist(),
                    "quat": poses[start + 3 : start + 7].astype(float).tolist(),
                }

    raw_sequence = getattr(snapshot, "sequence", None)
    sequence = (
        int(raw_sequence)
        if isinstance(raw_sequence, (int, np.integer))
        and not isinstance(raw_sequence, bool)
        else None
    )
    raw_received_ts = getattr(snapshot, "received_ts", None)
    received_ts = None
    if raw_received_ts is not None:
        try:
            parsed_ts = float(raw_received_ts)
        except (TypeError, ValueError):
            parsed_ts = math.nan
        if math.isfinite(parsed_ts):
            received_ts = parsed_ts
    return _FrozenWristObservation(
        frame=frame,
        depth=depth,
        camera_relative_pose=camera_relative_pose,
        proprio=proprio,
        sequence=sequence,
        received_ts=received_ts,
    )



def _wrist_capture_alignment_report(
    *,
    proprio: np.ndarray | None,
    side: str,
    sequence: int | None,
) -> dict[str, Any]:
    qvel = (
        np.asarray([], dtype=np.float64)
        if proprio is None
        else np.asarray(
            proprio[PROPRIO_SLICES[f"gripper_{side}_qvel"]],
            dtype=np.float64,
        ).reshape(-1)
    )
    return {
        "settled": True,
        "settle_required": False,
        "action_steps": 0,
        "stable_steps": 0,
        "required_stable_steps": 0,
        "max_steps": 0,
        "qvel_limit_m_s": None,
        "initial_qvel_m_s": qvel.astype(float).tolist(),
        "final_qvel_m_s": qvel.astype(float).tolist(),
        "alignment": "single_evaluator_observation",
        "observation_sequence": sequence,
    }



def _blocked_payload(name: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "; ".join(
            TOOL_CAPABILITIES[name]["current_violations"]
        ),
        "blocked_by": "official evaluator observation/action boundary",
    }

