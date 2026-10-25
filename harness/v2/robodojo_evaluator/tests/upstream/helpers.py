from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from tools.util.x5_kinematics import (
    calibrated_pose_for_joints,
    transform_to_pose,
    x5_forward_kinematics,
)


JOINT_ACTION_KEYS = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}
EEF_ACTION_KEYS = {
    "left_ee_pose": 7,
    "left_ee_joint_state": 1,
    "right_ee_pose": 7,
    "right_ee_joint_state": 1,
}


def fake_observation(height: int = 24, width: int = 32) -> dict[str, Any]:
    y, x = np.mgrid[:height, :width]
    color = np.stack(
        [
            (x * 7) % 255,
            (y * 11) % 255,
            ((x + y) * 5) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    # A small 10 cm protrusion at the clicked center gives grasp-planner tests
    # a finite RGB-D target instead of an ungraspable infinite depth wall. The
    # exact center remains 1.0 m, preserving the public read_depth assertions.
    depth = np.full((height, width), 1.10, dtype=np.float32)
    center_u = width // 2
    center_v = height // 2
    depth[
        max(0, center_v - 1) : min(height, center_v + 2),
        max(0, center_u - 1) : min(width, center_u + 2),
    ] = 1.0
    intrinsic = np.array(
        [
            [160.0, 0.0, (width - 1) / 2.0],
            [0.0, 160.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    def camera(offset: float) -> dict[str, Any]:
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = np.diag([1.0, -1.0, -1.0])
        extrinsic[0, 3] = offset
        return {
            "color": color.copy(),
            "depth": depth.copy(),
            "intrinsic_matrix": intrinsic.copy(),
            "extrinsic_matrix": extrinsic,
            "shape": color.shape,
            "segmentation": np.ones((height, width), dtype=np.int32),
        }

    state = {
        "left_arm_joint_state": np.zeros(6, dtype=np.float32),
        "left_ee_joint_state": np.array([0.5], dtype=np.float32),
        "left_ee_pose": np.array(
            [-0.25, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "right_arm_joint_state": np.zeros(6, dtype=np.float32),
        "right_ee_joint_state": np.array([0.5], dtype=np.float32),
        "right_ee_pose": np.array(
            [0.25, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "privileged_object_pose": np.ones(7, dtype=np.float32),
    }
    return {
        "instruction": ["Pick the marked item", "Place it at the target"],
        "vision": {
            "cam_head": camera(0.0),
            "cam_left_wrist": camera(-0.25),
            "cam_right_wrist": camera(0.25),
            "oracle_camera": camera(4.0),
        },
        "state": state,
        "action": {
            key: np.asarray(value).copy()
            for key, value in state.items()
            if key != "privileged_object_pose"
        },
        "reward": 1.0,
        "success": True,
        "env_idx": 0,
        "task_object_handles": ["forbidden"],
    }


def top_down_grasp_observation(height: int = 24, width: int = 32) -> dict[str, Any]:
    """A visible object above a table, viewed along USD camera -Z.

    The general fake camera looks upward. Its shallower depth patch is below
    the world-Z support, so it cannot test a downward tabletop grasp.
    """
    observation = fake_observation(height, width)
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 2.0
    observation["vision"]["cam_head"]["extrinsic_matrix"] = transform
    # Unlike the general API fake, this fixture exercises physical IK and
    # robot self collision. Use upright bases behind the visible target and
    # derive the observed EEF from official FK. The old identity-quaternion
    # EEF at q=0 calibrated a different base rotation/position; its downward
    # solutions intersected link2 with a finger or folded the wrist backward.
    for arm, side in (("left", -1.), ("right", 1.)):
        base = np.eye(4)
        base[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        base[:3, 3] = [side * .3, -.25, .9]
        pose = transform_to_pose(base @ x5_forward_kinematics(np.zeros(6)))
        observation["state"][f"{arm}_ee_pose"] = pose.astype(np.float32)
        observation["action"][f"{arm}_ee_pose"] = pose.astype(np.float32)
    return observation


def validate_action(action: dict[str, Any]) -> str:
    keys = set(action)
    if keys == set(JOINT_ACTION_KEYS):
        expected = JOINT_ACTION_KEYS
        action_type = "joint"
    elif keys == set(EEF_ACTION_KEYS):
        expected = EEF_ACTION_KEYS
        action_type = "ee"
    else:
        raise AssertionError(f"unexpected RoboDojo action keys: {sorted(keys)}")
    for key, size in expected.items():
        value = np.asarray(action[key], dtype=np.float64)
        if value.shape != (size,):
            raise AssertionError(f"{key} has shape {value.shape}, expected {(size,)}")
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"{key} contains non-finite values")
    return action_type


def apply_action(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(observation)
    action_type = validate_action(action)
    reference = {
        arm: (
            np.asarray(observation["state"][f"{arm}_arm_joint_state"]),
            np.asarray(observation["state"][f"{arm}_ee_pose"]),
        )
        for arm in ("left", "right")
    }
    for key, value in action.items():
        copied = np.asarray(value, dtype=np.float32).copy()
        updated["state"][key] = copied
        updated["action"][key] = copied
    if action_type == "joint":
        for arm, (reference_q, reference_pose) in reference.items():
            target_q = np.asarray(action[f"{arm}_arm_joint_state"])
            observed_pose = calibrated_pose_for_joints(
                reference_q,
                reference_pose,
                target_q,
            ).astype(np.float32)
            updated["state"][f"{arm}_ee_pose"] = observed_pose
    return updated


def apply_action_fraction(
    observation: dict[str, Any],
    action: dict[str, Any],
    fraction: float = 0.4,
) -> dict[str, Any]:
    updated = deepcopy(observation)
    action_type = validate_action(action)
    alpha = float(fraction)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    reference = {
        arm: (
            np.asarray(observation["state"][f"{arm}_arm_joint_state"]),
            np.asarray(observation["state"][f"{arm}_ee_pose"]),
        )
        for arm in ("left", "right")
    }
    for key, value in action.items():
        target = np.asarray(value, dtype=np.float32)
        current = np.asarray(updated["state"][key], dtype=np.float32)
        observed = current + alpha * (target - current)
        if key.endswith("_pose"):
            quaternion = observed[3:]
            norm = float(np.linalg.norm(quaternion))
            if norm > 1e-8:
                observed[3:] = quaternion / norm
        updated["state"][key] = observed.copy()
        updated["action"][key] = target.copy()
    if action_type == "joint":
        for arm, (reference_q, reference_pose) in reference.items():
            observed_q = np.asarray(
                updated["state"][f"{arm}_arm_joint_state"]
            )
            observed_pose = calibrated_pose_for_joints(
                reference_q,
                reference_pose,
                observed_q,
            ).astype(np.float32)
            updated["state"][f"{arm}_ee_pose"] = observed_pose
    return updated


def drive_job(
    runtime,
    observation,
    name: str,
    arguments: dict[str, Any],
    *,
    action_updater=apply_action,
    max_cycles: int = 128,
    timeout_s: float = 120.0,
):
    import time

    request_id = runtime.submit(name, arguments)
    action_types: list[str] = []
    current = observation
    # Idle/deferred RPC polls are not evaluator actions or simulation steps.
    # Match the real deploy loop while retaining both action and wall bounds.
    deadline = time.monotonic() + timeout_s
    while len(action_types) < max_cycles and time.monotonic() < deadline:
        actions = runtime.get_action()
        if actions:
            if len(actions) != 1:
                raise AssertionError("Interface emitted more than one evaluator action")
            action_types.append(validate_action(actions[0]))
            current = action_updater(current, actions[0])
            runtime.update_observation(current)
        status = runtime.scheduler.job_status(request_id)
        if status["done"]:
            return runtime.wait(request_id, 0.1), current, action_types
        if not actions:
            time.sleep(0.001)
    raise AssertionError(f"tool did not complete: {name}")
