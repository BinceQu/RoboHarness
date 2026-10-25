"""Observation-backed dual-X5 world that emits legal RoboDojo actions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tools.util.contract import ARM_DOF, ROBOT_PROFILE
from tools.util.geometry import finite_vector, robodojo_base_rotation
from tools.util.observation import ObservationAdapter


@dataclass(frozen=True)
class WaitAction:
    """Internal sentinel; the deploy loop must not call take_action for it."""

    reason: str = "policy_wait"


@dataclass(frozen=True)
class DeferAction:
    """Suspend this generator until the next policy poll without stepping."""

    reason: str = "background_work_pending"


class RoboDojoActionWorld:
    """The subset of world operations needed by the public tool functions."""

    def __init__(self, adapter: ObservationAdapter) -> None:
        self._official_adapter = adapter
        self._motion_epoch = 0
        self._base_rotation = robodojo_base_rotation()
        self._gripper_command_latch: dict[str, float] = {}
        self._arm_motion_history: dict[str, deque[dict[str, Any]]] = {
            "left": deque(maxlen=128),
            "right": deque(maxlen=128),
        }

    @property
    def robot_profile(self) -> str:
        return ROBOT_PROFILE

    @property
    def base_rotation(self) -> np.ndarray:
        return self._base_rotation.copy()

    def reset(self) -> None:
        self._motion_epoch = 0
        self._gripper_command_latch.clear()
        for history in self._arm_motion_history.values():
            history.clear()

    def episode_id(self) -> str:
        return str(self._official_adapter.status()["episode_id"])

    def motion_epoch(self) -> int:
        return int(self._motion_epoch)

    def mark_action_dispatched(
        self,
        action: Mapping[str, Any] | None = None,
        *,
        tool_name: str = "",
    ) -> None:
        """Record the measured start of legal arm motion before dispatch.

        The history contains only observations already delivered through the
        evaluator boundary.  It lets a later neutral-return operation retrace
        a path that the simulator actually executed, without consulting
        simulator state, collision objects, or evaluator planners.
        """

        if action is not None and str(tool_name) != "set_arm_to_grasp_position":
            for side in ("left", "right"):
                measured_q = np.asarray(self.arm_qpos_list(side), dtype=np.float64)
                joint_key = f"{side}_arm_joint_state"
                pose_key = f"{side}_ee_pose"
                moves_arm = False
                if pose_key in action:
                    commanded_pose = finite_vector(
                        action[pose_key], 7, f"{side} EEF target"
                    )
                    measured_pose = np.asarray(
                        self.eef_pose_list(side), dtype=np.float64
                    )
                    moves_arm = bool(
                        np.max(np.abs(commanded_pose - measured_pose)) > 1.0e-6
                    )
                if joint_key in action:
                    commanded_q = finite_vector(
                        action[joint_key], ARM_DOF, f"{side} arm target"
                    )
                    moves_arm = bool(np.max(np.abs(commanded_q - measured_q)) > 1.0e-6)
                if not moves_arm:
                    continue
                history = self._arm_motion_history[side]
                if (
                    history
                    and np.max(
                        np.abs(
                            np.asarray(history[-1]["qpos"], dtype=np.float64)
                            - measured_q
                        )
                    )
                    <= 1.0e-6
                ):
                    continue
                history.append(
                    {
                        "qpos": measured_q.tolist(),
                        "motion_epoch": int(self._motion_epoch),
                        "tool": str(tool_name or "unknown"),
                    }
                )
        self._motion_epoch += 1

    def arm_retrace_waypoints(
        self,
        arm: str,
        target: Sequence[float],
        *,
        anchor_tolerance_rad: float,
        max_waypoints: int,
    ) -> list[dict[str, Any]]:
        """Return newest-to-oldest measured qpos back to a target anchor."""

        side = str(arm).strip().lower()
        if side not in self._arm_motion_history:
            raise ValueError(f"bad arm '{arm}'")
        target_q = finite_vector(target, ARM_DOF, f"{side} retrace target")
        history = list(self._arm_motion_history[side])
        if not history:
            return []
        anchor_index = None
        for index in range(len(history) - 1, -1, -1):
            qpos = np.asarray(history[index]["qpos"], dtype=np.float64)
            if float(np.max(np.abs(qpos - target_q))) <= anchor_tolerance_rad:
                anchor_index = index
                break
        if anchor_index is None:
            return []
        segment = history[anchor_index:]
        if len(segment) > max(1, int(max_waypoints)):
            return []
        return [dict(entry) for entry in reversed(segment)]

    def clear_arm_motion_history(self, arms: Sequence[str]) -> None:
        for arm in arms:
            side = str(arm).strip().lower()
            if side not in self._arm_motion_history:
                raise ValueError(f"bad arm '{arm}'")
            self._arm_motion_history[side].clear()

    def arm_qpos_list(self, arm: str) -> list[float]:
        values = self._official_adapter.state(f"{arm}_arm_joint_state")
        return finite_vector(values, ARM_DOF, f"{arm} arm state").tolist()

    def arm_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [0.0] * ARM_DOF

    def gripper_qpos_list(self, arm: str) -> list[float]:
        values = self._official_adapter.state(f"{arm}_ee_joint_state", required=False)
        if values is None:
            values = self._official_adapter.action_state(f"{arm}_ee_joint_state")
        if values is None or values.size < 1:
            return [0.0]
        return [float(np.clip(values[0], 0.0, 1.0))]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [0.0]

    def eef_pose_list(self, arm: str) -> list[float]:
        return finite_vector(
            self._official_adapter.state(f"{arm}_ee_pose"),
            7,
            f"{arm} EEF pose",
        ).tolist()

    def eef_pose(self, arm: str) -> dict[str, Any]:
        pose = self.eef_pose_list(arm)
        return {"pos": pose[:3], "quat": pose[3:], "frame": "robodojo_env"}

    def trunk_qpos(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def trunk_qvel(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def base_qvel(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def robot_pose(self) -> dict[str, Any]:
        return {
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "frame": "robodojo_fixed_base",
        }

    def hold_action(self) -> WaitAction:
        return WaitAction()

    def defer_action(self, reason: str = "background_work_pending") -> DeferAction:
        return DeferAction(reason=str(reason))

    def gripper_command_latch(self, arm: str) -> float | None:
        side = str(arm).strip().lower()
        if side not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        target = self._gripper_command_latch.get(side)
        return None if target is None else float(target)

    def latch_gripper_command(self, arm: str, value: Any) -> float:
        """Remember a satisfied gripper target without emitting an action."""

        side = str(arm).strip().lower()
        if side not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        target = self._normalize_gripper_target(value, f"{side} gripper target")
        self._gripper_command_latch[side] = target
        return target

    @staticmethod
    def _normalize_gripper_target(value: Any, label: str) -> float:
        target = np.asarray(value, dtype=np.float64).reshape(-1)
        if target.size < 1 or not np.isfinite(target[0]):
            raise ValueError(f"{label} must contain one finite value")
        return float(np.clip(target[0], 0.0, 1.0))

    def _gripper_target_for_action(self, arm: str, value: Any | None) -> float:
        if value is not None:
            return self._normalize_gripper_target(value, f"{arm} gripper target")
        latched = self._gripper_command_latch.get(arm)
        if latched is not None:
            return float(latched)
        return float(self.gripper_qpos_list(arm)[0])

    def make_joint_action(
        self,
        *,
        left_arm: Any | None = None,
        left_gripper: Any | None = None,
        right_arm: Any | None = None,
        right_gripper: Any | None = None,
    ) -> dict[str, list[float]]:
        left_q = (
            self.arm_qpos_list("left")
            if left_arm is None
            else finite_vector(left_arm, ARM_DOF, "left arm target").tolist()
        )
        right_q = (
            self.arm_qpos_list("right")
            if right_arm is None
            else finite_vector(right_arm, ARM_DOF, "right arm target").tolist()
        )
        left_g = self._gripper_target_for_action("left", left_gripper)
        right_g = self._gripper_target_for_action("right", right_gripper)
        action = {
            "left_arm_joint_state": left_q,
            "left_ee_joint_state": [left_g],
            "right_arm_joint_state": right_q,
            "right_ee_joint_state": [right_g],
        }
        if left_gripper is not None:
            self._gripper_command_latch["left"] = left_g
        if right_gripper is not None:
            self._gripper_command_latch["right"] = right_g
        return action

    def make_eef_action(
        self,
        *,
        left_pose: Any | None = None,
        left_gripper: Any | None = None,
        right_pose: Any | None = None,
        right_gripper: Any | None = None,
    ) -> dict[str, list[float]]:
        left = (
            self.eef_pose_list("left")
            if left_pose is None
            else finite_vector(left_pose, 7, "left EEF target").tolist()
        )
        right = (
            self.eef_pose_list("right")
            if right_pose is None
            else finite_vector(right_pose, 7, "right EEF target").tolist()
        )
        left_g = self._gripper_target_for_action("left", left_gripper)
        right_g = self._gripper_target_for_action("right", right_gripper)
        action = {
            "left_ee_pose": left,
            "left_ee_joint_state": [left_g],
            "right_ee_pose": right,
            "right_ee_joint_state": [right_g],
        }
        if left_gripper is not None:
            self._gripper_command_latch["left"] = left_g
        if right_gripper is not None:
            self._gripper_command_latch["right"] = right_g
        return action

    def make_action(self, **kwargs: Any) -> dict[str, list[float]]:
        if any(key.startswith("eef_") for key in kwargs):
            return self.make_eef_action(
                left_pose=kwargs.get("eef_left"),
                right_pose=kwargs.get("eef_right"),
                left_gripper=kwargs.get("gripper_left"),
                right_gripper=kwargs.get("gripper_right"),
            )
        return self.make_joint_action(
            left_arm=kwargs.get("arm_left"),
            right_arm=kwargs.get("arm_right"),
            left_gripper=kwargs.get("gripper_left"),
            right_gripper=kwargs.get("gripper_right"),
        )
