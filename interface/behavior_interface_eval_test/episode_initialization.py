"""Observation-driven arm initialization for each official evaluator episode."""

from __future__ import annotations

import math
import threading
from copy import deepcopy
from typing import Any

import numpy as np


GRASP_PREP_Q = np.array(
    [
        0.0,
        0.0,
        0.0,
        -2.0943951024,
        0.0,
        -1.0471975512,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class EpisodeGraspPrepController:
    """Drive both arms to grasp-prep using one action per real observation."""

    def __init__(
        self,
        *,
        arm_dof: int,
        control_hz: float = 30.0,
        max_step_rad: float = 0.055,
        tolerance_rad: float = 0.035,
        timeout_s: float = 80.0,
        stable_steps: int = 3,
    ) -> None:
        self.arm_dof = int(arm_dof)
        if self.arm_dof not in (7, 8):
            raise ValueError(f"unsupported grasp-prep arm_dof={self.arm_dof}")
        self.target = GRASP_PREP_Q[: self.arm_dof].copy()
        self.control_hz = float(control_hz)
        self.max_step_rad = float(max_step_rad)
        self.tolerance_rad = float(tolerance_rad)
        self.timeout_s = float(timeout_s)
        self.required_stable_steps = int(stable_steps)
        if (
            self.control_hz <= 0.0
            or self.max_step_rad <= 0.0
            or self.tolerance_rad <= 0.0
            or self.timeout_s <= 0.0
            or self.required_stable_steps <= 0
        ):
            raise ValueError("episode grasp-prep configuration must be positive")
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._state = "waiting_for_proprio"
            self._starts: dict[str, np.ndarray] | None = None
            self._interpolation_steps = 0
            self._action_steps = 0
            self._stable_steps = 0
            self._max_abs_error_rad: float | None = None
            self._errors: dict[str, list[float]] = {}
            self._timed_out = False
            self._observed_hold: dict[str, np.ndarray] | None = None
            self._best_error = float('inf')
            self._last_progress_step = 0
            self._hold_reason = ""

    def ready(self) -> bool:
        with self._lock:
            return self._state in {"ready", "ready_observed_hold"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(
                {
                    "state": self._state,
                    "ready": self.ready(),
                    "grasp_prep_reached": self._state == "ready",
                    "hold_reason": self._hold_reason,
                    "warning": (
                        "Grasp-prep target was not reached; holding measured arm "
                        "positions. The agent must plan from the actual pose."
                        if self._observed_hold is not None else ""
                    ),
                    "hold_target_qpos": (
                        {side: q.tolist() for side, q in self._observed_hold.items()}
                        if self._observed_hold is not None else None
                    ),
                    "target_qpos": self.target.astype(float).tolist(),
                    "action_steps": int(self._action_steps),
                    "elapsed_control_s": float(
                        self._action_steps / self.control_hz
                    ),
                    "interpolation_steps": int(self._interpolation_steps),
                    "stable_steps": int(self._stable_steps),
                    "required_stable_steps": int(self.required_stable_steps),
                    "tolerance_rad": float(self.tolerance_rad),
                    "timeout_s": float(self.timeout_s),
                    "timed_out": bool(self._timed_out),
                    "max_abs_error_rad": self._max_abs_error_rad,
                    "error_rad": self._errors,
                    "observation_driven": True,
                    "direct_simulator_mutation": False,
                }
            )

    def note_missing_proprio(self) -> None:
        with self._lock:
            if not self.ready():
                self._state = "waiting_for_proprio"

    def _observed(self, world) -> dict[str, np.ndarray]:
        observed = {
            side: np.asarray(
                world.arm_qpos_list(side),
                dtype=np.float64,
            ).reshape(-1)
            for side in ("left", "right")
        }
        for side, values in observed.items():
            if values.size != self.arm_dof or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{side} arm proprio must contain {self.arm_dof} finite values"
                )
        return observed

    def step(self, world) -> np.ndarray:
        with self._lock:
            if self.ready():
                return world.hold_action()

            observed = self._observed(world)
            if self._observed_hold is not None:
                # A blocked joint must not be driven into contact forever.
                # Hold the finite measured pose; do not claim target convergence.
                error = max(float(np.max(np.abs(observed[side] - target)))
                            for side, target in self._observed_hold.items())
                self._stable_steps = self._stable_steps + 1 if error <= self.tolerance_rad else 0
                if self._stable_steps >= self.required_stable_steps:
                    self._state = "ready_observed_hold"
                return world.hold_action()
            if self._starts is None:
                self._starts = {
                    side: values.copy() for side, values in observed.items()
                }
                max_gap = max(
                    float(np.linalg.norm(self.target - values, ord=np.inf))
                    for values in observed.values()
                )
                self._interpolation_steps = max(
                    2,
                    int(math.ceil(max_gap / self.max_step_rad)),
                )
                self._state = "interpolating"

            if self._action_steps >= self._interpolation_steps:
                errors = {
                    side: self.target - values
                    for side, values in observed.items()
                }
                max_error = max(
                    float(np.max(np.abs(error)))
                    for error in errors.values()
                )
                self._max_abs_error_rad = max_error
                if max_error < self._best_error - 0.01:
                    self._best_error = max_error
                    self._last_progress_step = self._action_steps
                self._errors = {
                    side: error.astype(float).tolist()
                    for side, error in errors.items()
                }
                if max_error <= self.tolerance_rad:
                    self._stable_steps += 1
                else:
                    self._stable_steps = 0
                if self._stable_steps >= self.required_stable_steps:
                    for side in ("left", "right"):
                        world.set_arm_pin_qpos(side, self.target)
                    self._state = "ready"
                    return world.hold_action()
                if not self._timed_out:
                    self._state = "settling"

            timeout_steps = max(
                1,
                int(math.ceil(self.timeout_s * self.control_hz)),
            )
            stalled = (
                self._max_abs_error_rad is not None
                and self._max_abs_error_rad > self.tolerance_rad
                and self._action_steps - self._last_progress_step >= math.ceil(5 * self.control_hz)
            )
            if self._action_steps >= timeout_steps or stalled:
                self._timed_out = self._action_steps >= timeout_steps
                self._hold_reason = "timeout" if self._timed_out else "no_progress_for_5_control_seconds"
                self._state = "holding_after_timeout" if self._timed_out else "holding_after_stall"
                self._stable_steps = 0
                self._observed_hold = {side: q.copy() for side, q in observed.items()}
                for side, q in self._observed_hold.items():
                    world.set_arm_pin_qpos(side, q)
                return world.hold_action()

            if self._action_steps < self._interpolation_steps:
                alpha = _smoothstep(
                    (self._action_steps + 1) / self._interpolation_steps
                )
                commands = {
                    side: start + alpha * (self.target - start)
                    for side, start in self._starts.items()
                }
            else:
                commands = {
                    side: self.target.copy() for side in ("left", "right")
                }

            action = world.make_action(
                **{
                    f"arm_{side}": command.tolist()
                    for side, command in commands.items()
                }
            )
            self._action_steps += 1
            return action


__all__ = ["EpisodeGraspPrepController", "GRASP_PREP_Q"]
