"""Observation-backed action construction for the official evaluator boundary.

This module deliberately has no OmniGibson import and no simulator handle. It
adapts the existing skill-facing WorldAPI action helpers to the stock v3.9.1
R1Pro controller layout using only evaluator-supplied proprioception.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Optional

import numpy as np

from behavior_interface.world_api import WorldAPI
from behavior_interface_eval_test.robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    ROBOT_MODEL,
    ROBOT_PROFILE,
    enforce_locked_trunk_joint,
)


# Static output limits from the submitted r1pro_8dof_hf250 controller profile.
# Evaluator actions are normalized; policy-local odometry is integrated in SI.
_BASE_CONTROLLER_OUTPUT_SCALE = np.array(
    [0.75, 0.75, 1.0],
    dtype=np.float64,
)
# Reject only observations that cannot be produced by the configured base
# velocity envelope while the submitted command is a stop. This guards
# policy-local odometry against sampled velocity-drive limit cycles without
# changing reconciliation during ordinary navigation or contact response.
_BASE_QVEL_ENVELOPE_FACTOR = 1.25
_GRIPPER_EFFORT_LIMIT_N = 20.0
# Keep the controller-limit preload across subsequent official actions. Before
# the evaluator-owned constraint exists this preserves the physical contacts
# needed to finish its grasp window; explicit open is the sole release path.
_GRIPPER_CARRY_EFFORT_N = _GRIPPER_EFFORT_LIMIT_N
_GRIPPER_OPEN_EFFORT_N = 3.0
_GRIPPER_GENTLE_CLOSE_EFFORT_N = 0.1
_GRIPPER_MIN_CLOSE_KEEPALIVE_EFFORT_N = 0.05
_GRIPPER_HOLD_KP_N_PER_M = 40.0
_GRIPPER_HOLD_KD_N_PER_MPS = 1.0
_GRIPPER_HOLD_MAX_EFFORT_N = 0.5
_GRIPPER_QPOS_LOWER_M = 0.0
_GRIPPER_QPOS_UPPER_M = 0.05
_GRIPPER_OPEN_HOLD_THRESHOLD_M = 0.049


class ObservationBackedActionWorld(WorldAPI):
    """WorldAPI action surface backed only by allowed evaluator observations."""

    def __init__(
        self,
        *,
        proprio_provider: Callable[[], Optional[np.ndarray]],
        hold_action_provider: Callable[[], np.ndarray],
        eef_pose_provider: Callable[[], dict[str, Any]],
        robot_dof: int = ARM_DOF,
    ) -> None:
        if int(robot_dof) != ARM_DOF:
            raise ValueError(
                f"robot profile {ROBOT_PROFILE!r} requires arm_dof={ARM_DOF}, "
                f"got {robot_dof}"
            )
        super().__init__(env=None, robot=None, dry_run=True, robot_dof=ARM_DOF)
        self.control_hz = 30.0
        self._proprio_provider = proprio_provider
        self._hold_action_provider = hold_action_provider
        self._eef_pose_provider = eef_pose_provider
        self._gripper_command_pins = {
            arm: self._default_gripper_command(arm)
            for arm in ("left", "right")
        }
        self._gripper_hold_qpos: dict[str, list[float]] = {}
        self._gripper_hold_allows_close_effort: dict[str, bool] = {}
        self._pending_base_odometry: Optional[dict[str, Any]] = None
        self._last_base_command_direction = np.array(
            [1.0, 0.0], dtype=np.float64
        )
        self._base_odometry_source = "command_prediction"
        self._base_odometry_corrections = 0
        self._base_odometry_rejections = 0
        self._motion_epoch = 0
        self._episode_id = uuid.uuid4().hex
        self._episode_initialized = False
        # Prevent legacy hot-reload helpers from replacing these action methods.
        self._codex_pinned_actions_v10 = True

    def reset_observation_state(self) -> None:
        self._mock_base = np.zeros(3, dtype=np.float64)
        self._arm_pin_qpos.clear()
        self._tool_roll_pin_qpos.clear()
        if ARM_DOF == 8:
            self._tool_roll_pin_qpos.update({"left": 0.0, "right": 0.0})
        self._tool_roll_motion_enabled.clear()
        self._trunk_pin_qpos = None
        self._gripper_pin_qpos.clear()
        self._gripper_pin_effort.clear()
        self._gripper_command_pins = {
            arm: self._default_gripper_command(arm)
            for arm in ("left", "right")
        }
        self._gripper_hold_qpos.clear()
        self._gripper_hold_allows_close_effort.clear()
        self._gripper_close_keepalive.clear()
        self._pending_base_odometry = None
        self._last_base_command_direction = np.array(
            [1.0, 0.0], dtype=np.float64
        )
        self._base_odometry_source = "command_prediction"
        self._base_odometry_corrections = 0
        self._base_odometry_rejections = 0
        self._motion_epoch += 1
        self._episode_id = uuid.uuid4().hex
        self._episode_initialized = False

    def _proprio(self) -> Optional[np.ndarray]:
        value = self._proprio_provider()
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr if arr.size >= PROPRIO_DIM else None

    def required_proprio_vector(self) -> np.ndarray:
        """Return one complete finite evaluator-proprio sample or fail closed."""

        value = self._proprio_provider()
        if value is None:
            raise RuntimeError("official evaluator proprioception is unavailable")
        try:
            proprio = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("official evaluator proprioception is invalid") from exc
        if proprio.size != PROPRIO_DIM or not np.all(np.isfinite(proprio)):
            raise RuntimeError("official evaluator proprioception is invalid")
        return proprio.copy()

    def pin_navigation_posture_from_proprio(self, proprio) -> None:
        """Freeze trunk and arm targets to one validated evaluator sample."""

        try:
            vector = np.asarray(proprio, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("navigation proprioception is invalid") from exc
        if vector.size != PROPRIO_DIM or not np.all(np.isfinite(vector)):
            raise RuntimeError("navigation proprioception is invalid")
        trunk = vector[PROPRIO_SLICES["trunk_qpos"]].copy()
        arms = {
            side: vector[PROPRIO_SLICES[f"arm_{side}_qpos"]].copy()
            for side in ("left", "right")
        }
        if trunk.size != 4 or any(arm.size != ARM_DOF for arm in arms.values()):
            raise RuntimeError("navigation proprioception dimensions are invalid")
        if abs(float(trunk[3])) > 1.0e-6:
            raise RuntimeError("navigation requires the locked trunk yaw at zero")
        if ARM_DOF == 8 and any(
            abs(float(arms[side][7])) > 1.0e-6
            for side in ("left", "right")
        ):
            raise RuntimeError("navigation requires both locked J8 joints at zero")
        trunk[3] = 0.0
        self._trunk_pin_qpos = trunk.astype(float).tolist()
        for side, arm in arms.items():
            if ARM_DOF == 8:
                arm[7] = 0.0
                self._tool_roll_pin_qpos[side] = 0.0
            self._arm_pin_qpos[side] = arm.astype(float).tolist()

    def robot_action_dim(self) -> int:
        return ACTION_DIM

    def controller_action_idx(self, controller_name: str) -> np.ndarray:
        action_slice = ACTION_SLICES.get(str(controller_name))
        if action_slice is None:
            return np.array([], dtype=int)
        return np.arange(ACTION_DIM, dtype=int)[action_slice]

    def _default_gripper_command(self, arm: str) -> list[float]:
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        return ([1.0] if dim == 1 else [0.0] * dim)

    def gripper_motor_type(self, arm: str) -> str:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        return (
            "effort"
            if len(self.controller_action_idx(f"gripper_{arm}")) == 2
            else "position"
        )

    def gripper_uses_effort(self, arm: str) -> bool:
        return self.gripper_motor_type(arm) == "effort"

    def trunk_qpos(self) -> np.ndarray:
        proprio = self._proprio()
        if proprio is None:
            return np.zeros(4, dtype=np.float64)
        return proprio[PROPRIO_SLICES["trunk_qpos"]].astype(np.float64, copy=True)

    def _raw_base_qvel(self) -> np.ndarray:
        proprio = self._proprio()
        if proprio is None:
            return np.zeros(3, dtype=np.float64)
        return proprio[PROPRIO_SLICES["base_qvel"]].astype(
            np.float64, copy=True
        )

    def raw_base_qvel(self) -> np.ndarray:
        """Return the untouched base_qvel slice from official proprio.

        Measured on a live evaluator (drive forward, spin 90 deg, drive
        forward): this vector stays [speed, 0] through every leg, so its x/y
        are expressed in the *robot-local* frame and rotate with the chassis.
        It is therefore useless as an absolute heading reference -- do not try
        to integrate it as if it were a fixed/canonical frame.  Combined with
        base_qpos being banned in the standard track and eef_*_quat being
        relative to the base, official proprio exposes no absolute yaw at all;
        heading drift has to be corrected from the map side instead.
        """
        return self._raw_base_qvel()

    def base_qvel(self) -> np.ndarray:
        """Return observation velocity in the commanded robot-local frame.

        The official proprio contract exposes virtual-base qvel but not the
        corresponding absolute base yaw.  Its x/y norm is frame invariant.
        During an official action step, use that observed norm along the
        submitted local command direction.  This preserves real stall and
        speed feedback without querying a simulator pose or inventing RGB-D
        odometry.
        """
        observed = self._raw_base_qvel()
        if observed.size < 3 or not np.all(np.isfinite(observed[:3])):
            return np.zeros(3, dtype=np.float64)
        command = None
        if self._pending_base_odometry is not None:
            command = np.asarray(
                self._pending_base_odometry.get("physical_velocity", []),
                dtype=np.float64,
            ).reshape(-1)
        if command is not None and command.size >= 2:
            command_speed = float(np.linalg.norm(command[:2]))
            if command_speed > 1e-9:
                self._last_base_command_direction = (
                    command[:2] / command_speed
                )
        linear_speed = float(np.linalg.norm(observed[:2]))
        local_linear = self._last_base_command_direction * linear_speed
        return np.array(
            [local_linear[0], local_linear[1], observed[2]],
            dtype=np.float64,
        )

    def arm_qpos_list(self, arm: str) -> list[float]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        proprio = self._proprio()
        if proprio is None:
            return [0.0] * ARM_DOF
        values = proprio[PROPRIO_SLICES[f"arm_{arm}_qpos"]]
        return values.astype(float).tolist()

    def arm_qvel_list(self, arm: str) -> list[float]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        proprio = self._proprio()
        if proprio is None:
            return [0.0] * ARM_DOF
        values = proprio[PROPRIO_SLICES[f"arm_{arm}_qvel"]]
        return values.astype(float).tolist()

    def tool_roll_qpos(self, arm: str) -> float:
        """Read J8 from evaluator proprioception, not the dry-run pin cache."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if ARM_DOF != 8:
            raise RuntimeError("independent J8 tool roll requires the 8DOF profile")
        values = np.asarray(self.arm_qpos_list(arm), dtype=np.float64).reshape(-1)
        if values.size != ARM_DOF or not np.all(np.isfinite(values)):
            return float(self.tool_roll_pin_qpos(arm))
        return float(values[7])

    def tool_roll_joint_limits(self, arm: str) -> tuple[float, float]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if ARM_DOF != 8:
            raise RuntimeError("independent J8 tool roll requires the 8DOF profile")
        return -math.pi, math.pi

    def motion_epoch(self) -> int:
        return int(self._motion_epoch)

    def episode_id(self) -> str:
        return str(self._episode_id)

    def set_episode_initialized(self, initialized: bool) -> None:
        self._episode_initialized = bool(initialized)

    def episode_initialized(self) -> bool:
        return bool(self._episode_initialized)

    def _advance_motion_epoch_for_explicit_overrides(
        self,
        overrides: dict[str, Any],
    ) -> None:
        moved = False
        if "base" in overrides:
            base = np.asarray(
                overrides["base"],
                dtype=np.float64,
            ).reshape(-1)
            moved = bool(base.size and float(np.max(np.abs(base))) > 1e-6)
        if "trunk" in overrides:
            trunk = np.asarray(
                overrides["trunk"],
                dtype=np.float64,
            ).reshape(-1)
            observed = self.trunk_qpos()
            moved = bool(
                moved
                or (
                    observed.size == trunk.size
                    and float(np.max(np.abs(observed - trunk))) > 1e-5
                )
            )
        if moved:
            self._motion_epoch += 1

    def set_arm_pin_qpos(self, arm: str, qpos) -> None:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            raise ValueError(f"arm pin qpos for {arm} expects at least 7 values, got {arr.size}")
        if arr.size < ARM_DOF:
            current = np.asarray(self.arm_qpos_list(arm), dtype=np.float64)
            current[: arr.size] = arr
            arr = current
        arr = arr[:ARM_DOF].copy()
        if ARM_DOF == 8:
            arr[7] = self.tool_roll_pin_qpos(arm)
        self._arm_pin_qpos[arm] = arr.astype(float).tolist()

    def gripper_qpos_list(self, arm: str) -> Optional[list[float]]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        proprio = self._proprio()
        if proprio is None:
            return None
        values = proprio[PROPRIO_SLICES[f"gripper_{arm}_qpos"]]
        return values.astype(float).tolist()

    def gripper_qvel_list(self, arm: str) -> Optional[list[float]]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        proprio = self._proprio()
        if proprio is None:
            return None
        values = proprio[PROPRIO_SLICES[f"gripper_{arm}_qvel"]]
        return values.astype(float).tolist()

    @staticmethod
    def _quat_multiply(left, right) -> np.ndarray:
        lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
        rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
        return np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )

    def local_pose_from_robot_relative(self, pos, quat) -> dict[str, Any]:
        """Convert an allowed robot-relative pose into local command odometry."""
        base = self.robot_pose()
        yaw = float(base.yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rel_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        local_pos = np.array(
            [
                base.pos[0] + cos_y * rel_pos[0] - sin_y * rel_pos[1],
                base.pos[1] + sin_y * rel_pos[0] + cos_y * rel_pos[1],
                base.pos[2] + rel_pos[2],
            ],
            dtype=np.float64,
        )
        base_quat = np.array(
            [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)],
            dtype=np.float64,
        )
        local_quat = self._quat_multiply(base_quat, quat)
        norm = float(np.linalg.norm(local_quat))
        if norm > 1e-12:
            local_quat /= norm
        return {
            "pos": local_pos.astype(float).tolist(),
            "quat": local_quat.astype(float).tolist(),
            "frame": "local_command_odometry",
        }

    def eef_pose(self, arm: str = "right") -> dict[str, Any]:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        pose = self._eef_pose_provider().get(arm)
        if pose is None:
            return {
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "frame": "local_command_odometry",
            }
        return self.local_pose_from_robot_relative(pose["pos"], pose["quat"])

    def set_gripper_pin_qpos(self, arm: str, qpos) -> None:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"gripper pin qpos for {arm} is empty")
        if self.gripper_uses_effort(arm):
            if self.gripper_close_keepalive_active(arm):
                return
            if arr.size == 1 and abs(float(arr[0])) >= 0.5:
                effort = (
                    _GRIPPER_OPEN_EFFORT_N
                    if float(arr[0]) > 0.0
                    else -_GRIPPER_GENTLE_CLOSE_EFFORT_N
                )
                self.set_gripper_pin_effort(arm, [effort, effort])
                return
            dim = len(self.controller_action_idx(f"gripper_{arm}"))
            if arr.size == 1 and dim > 1:
                arr = np.repeat(arr, dim)
            if arr.size != dim or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"gripper qpos hold for {arm} expects {dim} finite "
                    f"values, got {arr.size}"
                )
            target = np.clip(
                arr,
                _GRIPPER_QPOS_LOWER_M,
                _GRIPPER_QPOS_UPPER_M,
            )
            self._gripper_hold_qpos[arm] = target.astype(float).tolist()
            self._gripper_hold_allows_close_effort[arm] = True
            self._gripper_pin_effort.pop(arm, None)
            self._gripper_command_pins[arm] = self._default_gripper_command(arm)
            return
        command = float(np.clip(np.mean(arr), -1.0, 1.0))
        self._set_gripper_command_pin(arm, [command])

    def _set_gripper_command_pin(self, arm: str, command) -> None:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        arr = np.asarray(command, dtype=np.float64).reshape(-1)
        if arr.size == 1 and dim > 1:
            arr = np.repeat(arr, dim)
        if arr.size != dim or not np.all(np.isfinite(arr)):
            raise ValueError(
                f"gripper command for {arm} expects {dim} finite values, got {arr.size}"
            )
        limit = _GRIPPER_EFFORT_LIMIT_N if self.gripper_uses_effort(arm) else 1.0
        arr = np.clip(arr, -limit, limit)
        if self.gripper_close_keepalive_active(arm):
            return
        if self.gripper_uses_effort(arm):
            self._gripper_hold_qpos.pop(arm, None)
            self._gripper_hold_allows_close_effort.pop(arm, None)
        self._gripper_command_pins[arm] = arr.astype(float).tolist()
        if self.gripper_uses_effort(arm):
            self._gripper_pin_effort[arm] = arr.astype(float).tolist()

    def _legacy_gripper_effort_command(self, arm: str, command) -> np.ndarray:
        """Translate legacy open/close or qpos intent to bounded finger effort."""
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        arr = np.asarray(command, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"gripper command for {arm} is empty")
        if arr.size == 1:
            value = float(arr[0])
            effort = (
                _GRIPPER_OPEN_EFFORT_N
                if value > 0.0
                else (-_GRIPPER_GENTLE_CLOSE_EFFORT_N if value < 0.0 else 0.0)
            )
            return np.full(dim, effort, dtype=np.float32)
        if arr.size != dim:
            raise ValueError(
                f"legacy gripper command for {arm} expects 1 or {dim} values, got {arr.size}"
            )
        if bool(np.all(arr >= 0.0)) and bool(np.all(arr <= 0.051)):
            current_raw = self.gripper_qpos_list(arm)
            if current_raw is not None:
                current = np.asarray(current_raw, dtype=np.float64).reshape(-1)
                if current.size == dim:
                    error = arr - current
                    return np.where(
                        error > 0.001,
                        _GRIPPER_OPEN_EFFORT_N,
                        np.where(
                            error < -0.001,
                            -_GRIPPER_GENTLE_CLOSE_EFFORT_N,
                            0.0,
                        ),
                    ).astype(np.float32)
        return np.where(
            arr > 0.0,
            _GRIPPER_OPEN_EFFORT_N,
            np.where(arr < 0.0, -_GRIPPER_GENTLE_CLOSE_EFFORT_N, 0.0),
        ).astype(np.float32)

    def set_gripper_pin_effort(self, arm: str, effort) -> None:
        if not self.gripper_uses_effort(arm):
            raise ValueError(f"{arm} gripper is not effort controlled")
        self._set_gripper_command_pin(arm, effort)

    def gripper_hold_qpos_list(self, arm: str) -> Optional[list[float]]:
        target = self._gripper_hold_qpos.get(str(arm).lower().strip())
        return None if target is None else list(target)

    def capture_gripper_hold_qpos(
        self,
        arm: str,
        *,
        allow_close_effort: Optional[bool] = None,
    ) -> Optional[list[float]]:
        """Lock an effort gripper at its current allowed qpos observation."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if not self.gripper_uses_effort(arm):
            return self.gripper_pin_qpos_list(arm)
        if self.gripper_close_keepalive_active(arm):
            return None
        current = np.asarray(
            self.gripper_qpos_list(arm),
            dtype=np.float64,
        ).reshape(-1)
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        if current.size != dim or not np.all(np.isfinite(current)):
            return None
        previous_allow_close = self._gripper_hold_allows_close_effort.get(arm)
        self.set_gripper_pin_qpos(arm, current)
        self._gripper_hold_allows_close_effort[arm] = bool(
            previous_allow_close
            if allow_close_effort is None and previous_allow_close is not None
            else (
                True
                if allow_close_effort is None
                else allow_close_effort
            )
        )
        return self.gripper_hold_qpos_list(arm)

    def gripper_hold_allows_close_effort(self, arm: str) -> Optional[bool]:
        arm = str(arm).lower().strip()
        if arm not in self._gripper_hold_qpos:
            return None
        return bool(self._gripper_hold_allows_close_effort.get(arm, True))

    def clear_tool_gripper_holds(self) -> None:
        """Release unlatched tool-local holds before an external policy takes over."""
        for arm in ("left", "right"):
            if self.gripper_close_keepalive_active(arm):
                continue
            self._gripper_hold_qpos.pop(arm, None)
            self._gripper_hold_allows_close_effort.pop(arm, None)

    def _ensure_uncommanded_gripper_holds(
        self,
        overrides: dict[str, Any],
    ) -> None:
        for arm in ("left", "right"):
            if (
                not self.gripper_uses_effort(arm)
                or self.gripper_close_keepalive_active(arm)
                or arm in self._gripper_hold_qpos
                or f"gripper_{arm}" in overrides
                or f"gripper_effort_{arm}" in overrides
            ):
                continue
            self.capture_gripper_hold_qpos(arm)

    def _gripper_hold_effort(self, arm: str) -> Optional[np.ndarray]:
        target_raw = self._gripper_hold_qpos.get(arm)
        if target_raw is None:
            return None
        target = np.asarray(target_raw, dtype=np.float64).reshape(-1)
        current = np.asarray(
            self.gripper_qpos_list(arm),
            dtype=np.float64,
        ).reshape(-1)
        velocity = np.asarray(
            self.gripper_qvel_list(arm),
            dtype=np.float64,
        ).reshape(-1)
        if (
            current.size != target.size
            or velocity.size != target.size
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(velocity))
        ):
            return np.zeros_like(target)
        effort = (
            _GRIPPER_HOLD_KP_N_PER_M * (target - current)
            - _GRIPPER_HOLD_KD_N_PER_MPS * velocity
        )
        # An explicit open remains a release even if contact prevented the
        # fingers from reaching their upper limits. Negative action is the
        # evaluator's assisted-grasp intent signal, so open holds are outward
        # only. Generic intermediate-qpos holds remain bidirectional.
        outward_only = (
            not self._gripper_hold_allows_close_effort.get(arm, True)
            or target >= _GRIPPER_OPEN_HOLD_THRESHOLD_M
        )
        effort[outward_only] = np.maximum(effort[outward_only], 0.0)
        return np.clip(
            effort,
            -_GRIPPER_HOLD_MAX_EFFORT_N,
            _GRIPPER_HOLD_MAX_EFFORT_N,
        )

    def gripper_pin_effort_list(self, arm: str) -> Optional[list[float]]:
        pin = self._gripper_pin_effort.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def clear_gripper_pin_effort(self, arm: Optional[str] = None) -> None:
        if arm is None:
            for side in ("left", "right"):
                if not self.gripper_close_keepalive_active(side):
                    self._gripper_pin_effort.pop(side, None)
                    self._gripper_command_pins[side] = self._default_gripper_command(side)
            return
        arm = str(arm).lower().strip()
        if self.gripper_close_keepalive_active(arm):
            return
        self._gripper_pin_effort.pop(arm, None)
        self._gripper_command_pins[arm] = self._default_gripper_command(arm)

    def gripper_close_keepalive_active(self, arm: str) -> bool:
        return str(arm).lower().strip() in self._gripper_close_keepalive

    def latch_gripper_close_keepalive(self, arm: str, effort=None) -> None:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        if self.gripper_uses_effort(arm):
            source = (
                [-_GRIPPER_CARRY_EFFORT_N] * dim
                if effort is None
                else effort
            )
            arr = np.asarray(source, dtype=np.float64).reshape(-1)
            if arr.size == 1 and dim > 1:
                arr = np.repeat(arr, dim)
            if arr.size != dim or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"gripper keepalive for {arm} expects {dim} finite values, got {arr.size}"
                )
            arr = -np.clip(
                np.abs(arr),
                _GRIPPER_MIN_CLOSE_KEEPALIVE_EFFORT_N,
                _GRIPPER_EFFORT_LIMIT_N,
            )
            self._gripper_hold_qpos.pop(arm, None)
            self._gripper_hold_allows_close_effort.pop(arm, None)
            pin = arr.astype(float).tolist()
            self._gripper_pin_effort[arm] = pin
            self._gripper_command_pins[arm] = pin
        else:
            self._gripper_hold_qpos.pop(arm, None)
            self._gripper_hold_allows_close_effort.pop(arm, None)
            self._gripper_command_pins[arm] = [-1.0]
        self._gripper_close_keepalive.add(arm)

    def release_gripper_close_keepalive(self, arm: str) -> None:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        self._gripper_close_keepalive.discard(arm)
        self._gripper_pin_effort.pop(arm, None)
        self._gripper_command_pins[arm] = self._default_gripper_command(arm)

    def gripper_pin_qpos_list(self, arm: str) -> Optional[list[float]]:
        if self.gripper_uses_effort(arm):
            return None
        pin = self._gripper_command_pins.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def clear_gripper_pin_qpos(self, arm: Optional[str] = None) -> None:
        if arm is None:
            for side in ("left", "right"):
                if self.gripper_close_keepalive_active(side):
                    continue
                self._gripper_hold_qpos.pop(side, None)
                self._gripper_hold_allows_close_effort.pop(side, None)
                self._gripper_command_pins[side] = self._default_gripper_command(side)
                self._gripper_pin_effort.pop(side, None)
            return
        arm = str(arm).lower().strip()
        if self.gripper_close_keepalive_active(arm):
            return
        self._gripper_hold_qpos.pop(arm, None)
        self._gripper_hold_allows_close_effort.pop(arm, None)
        self._gripper_command_pins.pop(arm, None)
        self._gripper_pin_effort.pop(arm, None)

    def enforce_gripper_close_keepalive(self, action) -> np.ndarray:
        """Overlay only a latched close at the final policy action boundary."""
        out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if out.size != ACTION_DIM:
            raise ValueError(
                f"official action must have {ACTION_DIM} values, got {out.size}"
            )
        for arm in ("left", "right"):
            dim = len(self.controller_action_idx(f"gripper_{arm}"))
            if arm in self._gripper_close_keepalive:
                pin = (
                    self._gripper_pin_effort.get(
                        arm,
                        [-_GRIPPER_CARRY_EFFORT_N] * dim,
                    )
                    if self.gripper_uses_effort(arm)
                    else self._gripper_command_pins.get(arm, [-1.0])
                )
                if len(pin) != dim:
                    raise ValueError(
                        f"gripper keepalive for {arm} has {len(pin)} values, "
                        f"expected {dim}"
                    )
                out[ACTION_SLICES[f"gripper_{arm}"]] = np.asarray(
                    pin,
                    dtype=np.float32,
                )
        return out

    def enforce_tool_gripper_holds(self, action) -> np.ndarray:
        """Overlay tool-owned qpos holds, then the authoritative close latch."""
        out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if out.size != ACTION_DIM:
            raise ValueError(
                f"official action must have {ACTION_DIM} values, got {out.size}"
            )
        for arm in ("left", "right"):
            if (
                arm in self._gripper_close_keepalive
                or not self.gripper_uses_effort(arm)
            ):
                continue
            hold_effort = self._gripper_hold_effort(arm)
            if hold_effort is not None:
                out[ACTION_SLICES[f"gripper_{arm}"]] = hold_effort.astype(
                    np.float32,
                )
        return self.enforce_gripper_close_keepalive(out)

    def gripper_keepalive_status(self) -> dict[str, Any]:
        return {
            arm: {
                "active": arm in self._gripper_close_keepalive,
                "command": list(self._gripper_command_pins.get(arm, [])),
                "hold_qpos": self.gripper_hold_qpos_list(arm),
                "hold_allows_close_effort": (
                    self.gripper_hold_allows_close_effort(arm)
                ),
                "hold_mode": (
                    "close_keepalive"
                    if arm in self._gripper_close_keepalive
                    else (
                        "qpos_effort_servo"
                        if arm in self._gripper_hold_qpos
                        else "explicit_command"
                    )
                ),
                "motor_type": self.gripper_motor_type(arm),
                "observation_basis": "gripper_qpos_qvel",
            }
            for arm in ("left", "right")
        }

    def limb_pin_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trunk": self.trunk_pin_qpos_list(),
        }
        for arm in ("left", "right"):
            arm_q = self.arm_pin_qpos_list(arm) or self.arm_qpos_list(arm)
            arm_pin = np.asarray(arm_q, dtype=np.float64).reshape(-1).copy()
            if ARM_DOF == 8 and arm_pin.size >= ARM_DOF:
                arm_pin[7] = self.tool_roll_pin_qpos(arm)
            out[f"arm_{arm}"] = arm_pin.astype(float).tolist()
            gripper = self.gripper_pin_qpos_list(arm)
            effort = self.gripper_pin_effort_list(arm)
            if effort is not None:
                out[f"gripper_effort_{arm}"] = effort
            elif gripper is not None:
                out[f"gripper_{arm}"] = gripper
        return out

    def _integrate_base_command(
        self,
        action_vx: float,
        action_vy: float,
        action_wz: float,
        *,
        dt: float = 1.0 / 30.0,
    ) -> None:
        controller_command = np.array(
            [action_vx, action_vy, action_wz],
            dtype=np.float64,
        )
        vx, vy, wz = controller_command * _BASE_CONTROLLER_OUTPUT_SCALE
        linear_speed = float(math.hypot(vx, vy))
        if linear_speed > 1e-9:
            self._last_base_command_direction = np.array(
                [vx / linear_speed, vy / linear_speed],
                dtype=np.float64,
            )
        before = self._mock_base.copy()
        x, y, yaw = before
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = cos_y * vx - sin_y * vy
        wy = sin_y * vx + cos_y * vy
        self._mock_base = np.array(
            [x + wx * dt, y + wy * dt, yaw + wz * dt],
            dtype=np.float64,
        )
        self._pending_base_odometry = {
            "before": before,
            "controller_command": controller_command,
            "physical_velocity": np.array([vx, vy, wz], dtype=np.float64),
            "predicted_dt_s": float(dt),
        }
        self._base_odometry_source = "command_prediction"

    def reconcile_base_odometry_from_proprio(self, dt_s: float) -> dict[str, Any]:
        """Reconcile a command prediction with physically coherent base qvel."""
        pending = self._pending_base_odometry
        if pending is None:
            return {
                "corrected": False,
                "source": self._base_odometry_source,
            }
        dt = float(np.clip(float(dt_s), 1.0 / 240.0, 0.20))
        raw_observed = self._raw_base_qvel()
        observed = self.base_qvel()
        if (
            raw_observed.size < 3
            or observed.size < 3
            or not np.all(np.isfinite(raw_observed[:3]))
            or not np.all(np.isfinite(observed[:3]))
        ):
            self._pending_base_odometry = None
            return {
                "corrected": False,
                "source": self._base_odometry_source,
                "reason": "base_qvel_unavailable",
            }
        linear_envelope = float(
            np.linalg.norm(_BASE_CONTROLLER_OUTPUT_SCALE[:2])
            * _BASE_QVEL_ENVELOPE_FACTOR
        )
        yaw_envelope = float(
            abs(_BASE_CONTROLLER_OUTPUT_SCALE[2])
            * _BASE_QVEL_ENVELOPE_FACTOR
        )
        commanded_velocity = np.asarray(
            pending["physical_velocity"],
            dtype=np.float64,
        ).reshape(3)
        stop_was_commanded = bool(
            float(np.max(np.abs(commanded_velocity))) <= 1e-9
        )
        raw_linear_speed = float(np.linalg.norm(raw_observed[:2]))
        if stop_was_commanded and (
            raw_linear_speed > linear_envelope
            or abs(float(raw_observed[2])) > yaw_envelope
        ):
            # _integrate_base_command already installed the bounded command
            # prediction in _mock_base. Keep it instead of integrating a raw
            # sample outside the submitted controller's physical envelope.
            self._pending_base_odometry = None
            self._base_odometry_source = "command_prediction_incoherent_qvel"
            self._base_odometry_rejections += 1
            return {
                "corrected": False,
                "source": self._base_odometry_source,
                "reason": "base_qvel_outside_controller_envelope",
                "raw_observed_base_qvel": raw_observed[:3]
                .astype(float)
                .tolist(),
                "observed_base_qvel": observed[:3].astype(float).tolist(),
                "commanded_base_velocity": commanded_velocity.astype(float)
                .tolist(),
                "linear_speed_mps": raw_linear_speed,
                "linear_envelope_mps": linear_envelope,
                "yaw_envelope_rad_s": yaw_envelope,
                "rejections": int(self._base_odometry_rejections),
            }
        x, y, yaw = np.asarray(
            pending["before"],
            dtype=np.float64,
        ).reshape(3)
        vx, vy, wz = np.asarray(observed[:3], dtype=np.float64).reshape(3)
        cos_y, sin_y = math.cos(float(yaw)), math.sin(float(yaw))
        wx = cos_y * vx - sin_y * vy
        wy = sin_y * vx + cos_y * vy
        self._mock_base = np.array(
            [x + wx * dt, y + wy * dt, yaw + wz * dt],
            dtype=np.float64,
        )
        self._pending_base_odometry = None
        self._base_odometry_source = (
            "proprio_base_qvel_magnitude_command_direction"
        )
        self._base_odometry_corrections += 1
        return {
            "corrected": True,
            "source": self._base_odometry_source,
            "dt_s": dt,
            "raw_observed_base_qvel": raw_observed[:3].astype(float).tolist(),
            "observed_base_qvel": observed[:3].astype(float).tolist(),
            "corrections": int(self._base_odometry_corrections),
            "rejections": int(self._base_odometry_rejections),
        }

    def base_odometry_status(self) -> dict[str, Any]:
        raw = self.raw_base_qvel()
        body = self.base_qvel()
        raw_speed = float(np.linalg.norm(raw[:2]))
        # 两者的夹角：如果 base_qvel 在某个不随车转的系里，这个角会跟着底盘
        # 一起转，就能当绝对朝向用。实测下来它恒为 0（转 90° 也不变），说明
        # x/y 就在机体系。留着这个量是为了官方哪天改了合同能立刻发现。
        heading_deg = None
        if min(raw_speed, float(np.linalg.norm(body[:2]))) > 1e-3:
            heading_deg = float(
                math.degrees(
                    math.atan2(raw[1], raw[0]) - math.atan2(body[1], body[0])
                )
            )
        return {
            "frame": "policy_local_odometry",
            "source": self._base_odometry_source,
            "corrections": int(self._base_odometry_corrections),
            "rejections": int(self._base_odometry_rejections),
            "pending_command_prediction": self._pending_base_odometry is not None,
            "raw_qvel": [float(v) for v in raw[:3]],
            "body_qvel": [float(v) for v in body[:3]],
            "raw_speed_mps": raw_speed,
            "raw_vs_body_heading_deg": heading_deg,
        }

    def policy_local_base_pose(self) -> np.ndarray:
        """Return policy-owned odometry, never a simulator/world robot pose."""

        return np.asarray(self._mock_base, dtype=np.float64).reshape(3).copy()

    def navigation_action_prediction_checkpoint(self) -> dict[str, Any]:
        """Capture action-construction side effects before guarded prefetch."""

        pending = None
        if self._pending_base_odometry is not None:
            pending = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._pending_base_odometry.items()
            }
        return {
            "owner": id(self),
            "mock_base": self._mock_base.copy(),
            "pending_base_odometry": pending,
            "last_base_command_direction": (
                self._last_base_command_direction.copy()
            ),
            "base_odometry_source": str(self._base_odometry_source),
            "motion_epoch": int(self._motion_epoch),
        }

    def restore_navigation_action_prediction(self, checkpoint) -> None:
        """Discard an action that was constructed but never sent."""

        if not isinstance(checkpoint, dict) or checkpoint.get("owner") != id(self):
            raise RuntimeError("navigation action checkpoint is invalid")
        try:
            mock_base = np.asarray(
                checkpoint["mock_base"], dtype=np.float64
            ).reshape(3)
            direction = np.asarray(
                checkpoint["last_base_command_direction"], dtype=np.float64
            ).reshape(2)
            motion_epoch = int(checkpoint["motion_epoch"])
            source = str(checkpoint["base_odometry_source"])
            pending_raw = checkpoint["pending_base_odometry"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("navigation action checkpoint is invalid") from exc
        if (
            not np.all(np.isfinite(mock_base))
            or not np.all(np.isfinite(direction))
            or motion_epoch < 0
            or (pending_raw is not None and not isinstance(pending_raw, dict))
        ):
            raise RuntimeError("navigation action checkpoint is invalid")
        pending = None
        if pending_raw is not None:
            pending = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in pending_raw.items()
            }
        self._mock_base = mock_base.copy()
        self._pending_base_odometry = pending
        self._last_base_command_direction = direction.copy()
        self._base_odometry_source = source
        self._motion_epoch = motion_epoch

    def commit_navigation_action_prediction(self, action) -> None:
        """Commit base odometry only when a guarded action will be sent."""

        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.size != ACTION_DIM or not np.all(np.isfinite(vector)):
            raise RuntimeError("navigation action is invalid")
        base = vector[ACTION_SLICES["base"]]
        self._advance_motion_epoch_for_explicit_overrides({"base": base})
        self._integrate_base_command(
            float(base[0]),
            float(base[1]),
            float(base[2]),
        )

    def enforce_official_action(self, values) -> np.ndarray:
        """Apply immutable trunk lock plus world-owned per-arm J8 pins."""
        action = np.asarray(values, dtype=np.float32).reshape(-1).copy()
        if action.size != ACTION_DIM:
            raise ValueError(
                f"official action must have {ACTION_DIM} values, got {action.size}"
            )
        trunk_slice = ACTION_SLICES["trunk"]
        action[trunk_slice] = enforce_locked_trunk_joint(
            action[trunk_slice],
            copy=True,
        ).astype(np.float32, copy=False)
        if ARM_DOF == 8:
            for arm in ("left", "right"):
                arm_slice = ACTION_SLICES[f"arm_{arm}"]
                action[arm_slice.start + 7] = float(
                    self.tool_roll_pin_qpos(arm)
                )
        return self.enforce_tool_gripper_holds(action)

    def make_action_unpinned(self, **overrides: np.ndarray) -> np.ndarray:
        overrides = dict(overrides)
        self._ensure_uncommanded_gripper_holds(overrides)
        direct_gripper_effort: set[str] = set()
        for arm in ("left", "right"):
            alias = f"gripper_effort_{arm}"
            if alias not in overrides:
                continue
            ctrl = f"gripper_{arm}"
            if ctrl in overrides:
                raise ValueError(f"cannot specify both {ctrl} and {alias}")
            overrides[ctrl] = overrides.pop(alias)
            direct_gripper_effort.add(ctrl)
        action = np.asarray(
            self._hold_action_provider(),
            dtype=np.float32,
        ).reshape(-1).copy()
        if action.size != ACTION_DIM:
            raise ValueError(
                f"hold action must have {ACTION_DIM} values, got {action.size}"
            )

        for ctrl, value in overrides.items():
            idx = self.controller_action_idx(ctrl)
            if idx.size == 0:
                raise ValueError(
                    f"controller '{ctrl}' is not present in official "
                    f"{ROBOT_MODEL} action layout"
                )
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
            if ctrl.startswith("gripper_"):
                if vec.size == 0:
                    raise ValueError(f"action for {ctrl} is empty")
                arm = ctrl.split("_", 1)[1]
                if self.gripper_uses_effort(arm):
                    if ctrl in direct_gripper_effort:
                        if vec.size == 1 and idx.size > 1:
                            vec = np.repeat(vec, idx.size)
                        vec = np.clip(
                            vec,
                            -_GRIPPER_EFFORT_LIMIT_N,
                            _GRIPPER_EFFORT_LIMIT_N,
                        ).astype(np.float32, copy=False)
                    else:
                        vec = self._legacy_gripper_effort_command(arm, vec)
                    if (
                        self.gripper_close_keepalive_active(arm)
                        and bool(np.all(vec > 0.0))
                    ):
                        self.release_gripper_close_keepalive(arm)
                else:
                    vec = np.asarray([float(np.mean(vec))], dtype=np.float32)
                    if (
                        self.gripper_close_keepalive_active(arm)
                        and float(vec[0]) > 0.0
                    ):
                        self.release_gripper_close_keepalive(arm)
            elif ctrl.startswith("arm_") and vec.size != idx.size:
                if vec.size > idx.size:
                    vec = vec[: idx.size]
                else:
                    expanded = action[idx].copy()
                    expanded[: vec.size] = vec
                    vec = expanded
            if ctrl.startswith("arm_"):
                arm = ctrl.split("_", 1)[1]
                vec = vec.copy()
                if ARM_DOF == 8:
                    if arm in self._tool_roll_motion_enabled:
                        lower, upper = self.tool_roll_joint_limits(arm)
                        requested_j8 = float(vec[7])
                        if not lower <= requested_j8 <= upper:
                            raise ValueError(
                                f"{arm} J8 target {requested_j8:.6f}rad is outside "
                                f"[{lower:.6f}, {upper:.6f}]"
                            )
                        self.set_tool_roll_pin_qpos(arm, requested_j8)
                    else:
                        vec[7] = self.tool_roll_pin_qpos(arm)
            elif ctrl == "trunk":
                vec = enforce_locked_trunk_joint(vec, copy=True).astype(
                    np.float32,
                    copy=False,
                )
            if vec.size != idx.size:
                raise ValueError(
                    f"action length mismatch: {ctrl} expects {idx.size}, got {vec.size}"
                )
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"action for {ctrl} contains non-finite values")

            action[idx] = vec
            if ctrl in ("arm_left", "arm_right"):
                self._arm_pin_qpos[ctrl.split("_", 1)[1]] = vec.astype(float).tolist()
            elif ctrl == "trunk":
                self._trunk_pin_qpos = vec.astype(float).tolist()
            elif ctrl.startswith("gripper_"):
                arm = ctrl.split("_", 1)[1]
                self._set_gripper_command_pin(arm, vec)
            elif ctrl == "base":
                self._integrate_base_command(
                    float(vec[0]),
                    float(vec[1]),
                    float(vec[2]),
                )
        return self.enforce_official_action(action)

    def pinned_action(self, **overrides: np.ndarray) -> np.ndarray:
        self._advance_motion_epoch_for_explicit_overrides(overrides)
        # Inspect the caller's intent before adding historical pins. A stored
        # zero-effort command is not an explicit command from this action and
        # must be replaced by a hold of the current observed finger positions.
        self._ensure_uncommanded_gripper_holds(overrides)
        values = self.limb_pin_kwargs()
        values["base"] = [0.0, 0.0, 0.0]
        for arm in ("left", "right"):
            command_name = f"gripper_{arm}"
            effort_name = f"gripper_effort_{arm}"
            if command_name in overrides:
                values.pop(effort_name, None)
            elif effort_name in overrides:
                values.pop(command_name, None)
        values.update(overrides)
        return self.make_action_unpinned(**values)

    def make_action(self, **overrides: np.ndarray) -> np.ndarray:
        return self.pinned_action(**overrides)

    def empty_action(self) -> np.ndarray:
        return self.hold_action_pinned()

    def hold_action(self) -> np.ndarray:
        return self.hold_action_pinned()

    def hold_action_pinned(self) -> np.ndarray:
        self._ensure_uncommanded_gripper_holds({})
        values = self.limb_pin_kwargs()
        values["base"] = [0.0, 0.0, 0.0]
        return self.make_action_unpinned(**values)

    def make_action_trunk_locked(self, trunk_q) -> np.ndarray:
        self._advance_motion_epoch_for_explicit_overrides(
            {"trunk": trunk_q}
        )
        self._ensure_uncommanded_gripper_holds({})
        values = self.limb_pin_kwargs()
        values["base"] = [0.0, 0.0, 0.0]
        values["trunk"] = np.asarray(
            trunk_q,
            dtype=np.float64,
        ).reshape(-1).tolist()
        return self.make_action_unpinned(**values)

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> np.ndarray:
        return self.pinned_action(base=[float(vx), float(vy), float(wz)])
