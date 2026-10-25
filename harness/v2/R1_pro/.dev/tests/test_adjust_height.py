"""Replay height status boundaries without a simulator or CUDA context."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.util.action_constants import ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD
from tools.util.geometry.trunk_vertical_lift_local import load_reverse_upward_lut
from tools.util.model import ACTION_SLICES, PROPRIO_SLICES
from tools.util.motion_common import _yield_adjust_height_motion


OFF_TABLE = [-0.9914950132369995, 2.5306994915008545, 0.3827572166919708, -5.006653282180196e-6]


def row_at(height):
    return min(load_reverse_upward_lut()["rows"], key=lambda row: abs(row["z_target_m"] - height))["trunk_q"]


def execute(start, upward, follow=True):
    measured = np.asarray(start, dtype=np.float64).copy()
    pins, commands = {}, []
    def proprio():
        values = np.zeros(65)
        values[PROPRIO_SLICES["trunk_qpos"]] = measured
        return values
    def action(q):
        values = np.zeros(27)
        values[ACTION_SLICES["trunk"]] = q
        return values
    world = SimpleNamespace(
        trunk_qpos=lambda: measured.copy(), arm_qpos_list=lambda _: [0.] * 8,
        set_arm_pin_qpos=lambda side, q: pins.update({side: q}),
        set_trunk_pin_qpos=lambda q: pins.update(trunk=q),
        make_action_trunk_locked=action,
        _official_adapter=SimpleNamespace(proprio_vector=proprio),
    )
    ctx = SimpleNamespace(world=world, raise_if_cancelled=lambda _: None)
    motion = _yield_adjust_height_motion(ctx, upward=upward, timeout_s=120,
        tolerance=ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD, stage_prefix="height regression")
    while True:
        try:
            command = next(motion)
        except StopIteration as stopped:
            return stopped.value, commands
        commands.append(command.copy())
        if follow:
            measured[:] = command[ACTION_SLICES["trunk"]]


def test_unsaturated_off_table_hold_does_not_report_height_success():
    result, commands = execute(OFF_TABLE, .05)
    assert not result["ok"] and not result["height_converged"]
    assert not result["requested_height_converged"]
    assert result["unfulfilled_hold"]
    assert not result["workspace_saturated"]
    assert abs(result["actual_upward_m"]) < 1e-5
    assert result["z_requested_error_m"] < -.05
    assert "no height change was commanded" in result["error"]
    # Correct the report, not the inherited trajectory: no new torso move.
    assert commands
    assert all(np.array_equal(command[3:6], np.asarray(OFF_TABLE[:3])) for command in commands)


@pytest.mark.parametrize("start,upward", [
    (OFF_TABLE, 0), (row_at(.67), -1), (row_at(1.15), 1),
    (row_at(.9), .003), (row_at(.9), .1), (row_at(.9), -.1),
])
def test_zero_saturated_small_and_normal_requests_keep_existing_behavior(start, upward):
    result, commands = execute(start, upward)
    assert result["ok"] and result["height_converged"], result
    assert not result["unfulfilled_hold"]
    assert commands
    if abs(upward) == .1:
        assert abs(result["actual_upward_m"] - upward) < .02
    else:
        assert abs(result["actual_upward_m"]) < 1e-5


def test_height_tracking_stall_still_returns_failure():
    result, _ = execute(row_at(.9), .1, follow=False)
    assert not result["ok"] and result["tracking_stalled"]
    assert not result["unfulfilled_hold"]
    assert "stalled" in result["error"]
