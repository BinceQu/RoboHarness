"""Exercise chassis feedback offline, without starting an evaluator or GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.util.action_constants import (
    ADJUST_CHASSIS_BASE_MAX_ANG_RADPS,
    ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
    ADJUST_CHASSIS_RETREAT_MAX_MPS,
    CONTROL_HZ,
)
from tools.util.motion_common import _yield_adjust_chassis_controller


def execute(forward, feedback, *, timeout_s=120):
    sequence = 0
    velocity = np.zeros(3)
    commands = []
    world = SimpleNamespace(
        _official_adapter=SimpleNamespace(status=lambda: {"sequence": sequence}),
        base_qvel=lambda: velocity.copy(),
        set_base_velocity=lambda x, y, yaw: np.array([x, y, yaw]),
        hold_action=lambda: np.zeros(3),
    )
    context = SimpleNamespace(world=world, raise_if_cancelled=lambda _: None)
    motion = _yield_adjust_chassis_controller(context, forward=forward,
        timeout_s=timeout_s, _commit_result=False)
    while True:
        try:
            normalized = next(motion)
        except StopIteration as done:
            return done.value, commands
        requested_velocity = normalized * np.array([
            ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
            ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
            ADJUST_CHASSIS_BASE_MAX_ANG_RADPS,
        ])
        commands.append(requested_velocity.copy())
        velocity[:] = feedback(requested_velocity, len(commands))
        sequence += 1


def test_blocked_reverse_stops_instead_of_spending_entire_simulation_budget():
    result, commands = execute(-.7, lambda command, count: np.zeros(3))
    assert not result["ok"]
    assert result["obstacle_stop_reason"] == "recovery_qvel_stall"
    assert not result["timed_out"]
    # Three speed increases, then one bounded max-speed stall window.
    assert result["retreat_speed_max_mps"] == ADJUST_CHASSIS_RETREAT_MAX_MPS
    assert result["retreat_speed_escalations"] == 3
    assert len(commands) < CONTROL_HZ * 5
    assert np.array_equal(commands[-1], np.zeros(3))


def test_recovery_still_escapes_when_wheels_start_making_progress():
    def feedback(command, count):
        return command if count > 95 else np.zeros(3)
    result, commands = execute(-.7, feedback)
    assert result["recovery_attempted"] and result["recovery_ok"]
    assert result["retreat_m"] >= result["retreat_target_m"]
    assert result["ok"]
    assert len(commands) < CONTROL_HZ * 8
    assert np.array_equal(commands[-1], np.zeros(3))


@pytest.mark.parametrize("forward", [-.7, .7, 0.])
def test_unobstructed_translation_and_hold_remain_successful(forward):
    result, commands = execute(forward, lambda command, count: command)
    assert result["ok"] and result["linear_target_reached"]
    assert abs(result["actual"]["forward_m"] - forward) <= result["linear_tolerance_m"]
    assert not result["recovery_attempted"]
    assert np.array_equal(commands[-1], np.zeros(3))
