from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from behavior_interface_eval_test.robot_contract import ACTION_DIM, ACTION_SLICES
from behavior_interface_eval_test.tool.official_v2.tools import close_gripper
from behavior_interface_eval_test.tool.official_v2.visualization_local import (
    measure_wrist_opening_depth_evidence,
)


class _NoActuationWorld:
    def __init__(self) -> None:
        self._keepalive: dict[str, list[float]] = {}

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        return [0.05, 0.047]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [0.0, 0.0]

    def gripper_uses_effort(self, arm: str) -> bool:
        return ACTION_SLICES[f"gripper_{arm}"].stop - ACTION_SLICES[f"gripper_{arm}"].start == 2

    def make_action(self, **overrides) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for name, value in overrides.items():
            controller = name.replace("gripper_effort_", "gripper_")
            action[ACTION_SLICES[controller]] = value
        return self._overlay_keepalive(action)

    def hold_action(self) -> np.ndarray:
        return self._overlay_keepalive(np.zeros(ACTION_DIM, dtype=np.float32))

    def latch_gripper_close_keepalive(self, arm: str, effort=None) -> None:
        dim = ACTION_SLICES[f"gripper_{arm}"].stop - ACTION_SLICES[f"gripper_{arm}"].start
        self._keepalive[arm] = list(effort or ([-1.0] * dim))

    def gripper_close_keepalive_active(self, arm: str) -> bool:
        return arm in self._keepalive

    def _overlay_keepalive(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        for arm, command in self._keepalive.items():
            action[ACTION_SLICES[f"gripper_{arm}"]] = command
        return action


class _FullOpenNoisyVelocityWorld(_NoActuationWorld):
    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        return [0.05, 0.05]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [0.425, -0.425]

    def gripper_uses_effort(self, arm: str) -> bool:
        del arm
        return True

    def make_action(self, **overrides) -> dict:
        return dict(overrides)

    def hold_action(self) -> dict:
        return {"hold": True}


class _FreeClosingWorld(_FullOpenNoisyVelocityWorld):
    """An empty gripper keeps moving until both fingers reach the limit."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        qpos = max(0.001, 0.05 - 0.005 * self._action_count)
        return [qpos, qpos]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [-0.15, -0.15] if self._action_count < 10 else [0.0, 0.0]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        return dict(overrides)


class _AsymmetricClosingWorld(_FullOpenNoisyVelocityWorld):
    """Reproduce a first-contact finger moving farther than its mate."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        progress = min(self._action_count, 19) / 19.0
        before = np.asarray([0.05, 0.05])
        after = np.asarray([0.038, 0.045])
        return (before + progress * (after - before)).tolist()

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        # Persistent velocity prevents the surrogate from declaring a contact
        # window; the test is about force limiting, not a false success.
        return [-0.02, -0.02]

    def make_action(self, **overrides) -> np.ndarray:
        self._action_count += 1
        return dict(overrides)


class _NoisySettledHandleWorld(_FullOpenNoisyVelocityWorld):
    """Replay job-1787478667674: stable qpos with stale/noisy qvel."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        progress = min(self._action_count, 19) / 19.0
        before = np.asarray([0.0499998368, 0.0499243662])
        settled = np.asarray([0.0186960083, 0.0315870494])
        return (before + progress * (settled - before)).tolist()

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        if self._action_count < 19:
            return [-0.02, -0.02]
        return [-0.0007977381, -0.0083496086]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        return dict(overrides)


class _DelayedSlipAtMaxEffortWorld(_FullOpenNoisyVelocityWorld):
    """A false contact stays still during ramp-up, then closes at 20 N."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0
        self._max_effort_actions = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        if self._action_count == 0:
            return [0.05, 0.05]
        slip = 0.001 * max(0, self._max_effort_actions - 1)
        return [max(0.001, 0.04 - slip), max(0.001, 0.03 - slip)]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        velocity = -0.03 if self._max_effort_actions >= 2 else 0.0
        return [velocity, velocity]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        command = overrides.get("gripper_effort_right")
        if command is not None and max(abs(float(v)) for v in command) >= 19.9:
            self._max_effort_actions += 1
        return dict(overrides)


class _NearOpenAsymmetricBlockageWorld(_FullOpenNoisyVelocityWorld):
    """Replay the 15061 qpos equilibrium after the first low-force action."""

    def __init__(self, blocked_qpos=(0.0479, 0.05)) -> None:
        super().__init__()
        self._action_count = 0
        self._blocked_qpos = list(blocked_qpos)

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        if self._action_count == 0:
            return [0.05, 0.05]
        return list(self._blocked_qpos)

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        # The 15061 estimator remained noisy even though qpos was stationary.
        return [-0.04, 0.18]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        return dict(overrides)


class _NearOpenDriftingBlockageWorld(_NearOpenAsymmetricBlockageWorld):
    """Move farther than the evidence-window tolerance every four actions."""

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        if self._action_count == 0:
            return [0.05, 0.05]
        first_finger = (
            0.0479 if (self._action_count // 4) % 2 == 0 else 0.0472
        )
        return [first_finger, 0.05]


class _NearOpenConfirmationDriftWorld(_NearOpenAsymmetricBlockageWorld):
    """Shift qpos after depth contact but before its confirmation completes."""

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        if self._action_count == 0:
            return [0.05, 0.05]
        first_finger = 0.0479 if self._action_count <= 12 else 0.0472
        return [first_finger, 0.05]


class _SingleContactResumeWorld(_FullOpenNoisyVelocityWorld):
    """One finger stalls, receives preload, then resumes closing."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        step = self._action_count
        if step <= 4:
            return [0.05 - 0.0025 * step, 0.05 - 0.001 * step]
        if step <= 20:
            return [0.04, 0.046 - 0.0005 * (step - 4)]
        return [
            0.038 - 0.0003 * (step - 21),
            0.038 - 0.0003 * (step - 21),
        ]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        if 4 < self._action_count <= 20:
            return [0.0, -0.02]
        return [-0.02, -0.02]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        return dict(overrides)


class _OneFingerOnlyClosingWorld(_FullOpenNoisyVelocityWorld):
    """Replay the 15064 one-moving-finger false-positive geometry."""

    def __init__(self) -> None:
        super().__init__()
        self._action_count = 0

    def gripper_qpos_list(self, arm: str) -> list[float]:
        del arm
        progress = min(self._action_count, 5) / 5.0
        return [
            0.04981335997581482
            + (0.04980548471212387 - 0.04981335997581482) * progress,
            0.049999579787254333
            + (0.03654167801141739 - 0.049999579787254333) * progress,
        ]

    def gripper_qvel_list(self, arm: str) -> list[float]:
        del arm
        return [0.0, -0.02] if self._action_count < 5 else [0.0, 0.0]

    def make_action(self, **overrides) -> dict:
        self._action_count += 1
        return dict(overrides)


def _occlusion_evidence(side: str = "side_1", scale: float = 1.0) -> dict:
    bridge = {
        "point_count": int(round(7500 * scale)),
        "center_point_count": int(round(5300 * scale)),
        "side_0_point_count": 80 if side == "side_0" else 0,
        "side_1_point_count": 80 if side == "side_1" else 0,
        "normalized_lateral_q05": 0.48,
        "normalized_lateral_q95": 0.74,
    }
    boundary = {"point_count": int(round(1450 * scale))}
    return {
        "ok": True,
        "bilateral_depth_span_observed": False,
        "occlusion_limited_bilateral_blockage_observed": True,
        "occlusion_limited_candidate": {
            "visible_side": side,
            "bridge_component": bridge,
            "opposing_boundary_component": boundary,
            "bridge_inside_fraction": 0.84,
        },
    }


class CloseGripperDiagnosticsTest(unittest.TestCase):
    def test_asymmetric_closure_uses_fixed_half_newton_seek_force(self) -> None:
        result = {}
        world = _AsymmetricClosingWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        actions = list(close_gripper(ctx, arm="left", timeout_s=0.6))

        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_reason"], "timeout")
        self.assertFalse(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_moved"], [True, True])
        self.assertEqual(result["per_finger_blocked"], [False, False])
        np.testing.assert_allclose(
            result["gripper_qpos_before"],
            [0.05, 0.05],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result["gripper_qpos_after"],
            [0.038, 0.045],
            atol=1e-7,
        )
        self.assertEqual(result["peak_precontact_effort_n"], 0.5)
        commands = np.asarray(
            [
                action["gripper_effort_left"]
                for action in actions
                if isinstance(action, dict)
                and "gripper_effort_left" in action
            ]
        )
        self.assertEqual(float(np.max(np.abs(commands))), 0.5)

    def test_15064_stationary_qpos_ignores_noisy_qvel_and_preloads(self) -> None:
        result = {}
        ctx = SimpleNamespace(
            world=_NoisySettledHandleWorld(),
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        actions = list(close_gripper(ctx, arm="right", timeout_s=3.5))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_moved"], [True, True])
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertFalse(result["qvel_used_for_contact_detection"])
        self.assertEqual(result["peak_precontact_effort_n"], 0.5)
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)
        timeout_step_budget = int(np.ceil(3.5 * 30.0))
        self.assertGreaterEqual(
            timeout_step_budget / result["action_steps"],
            2.0,
        )
        commands = np.asarray(
            [
                action["gripper_effort_right"]
                for action in actions
                if "gripper_effort_right" in action
            ],
            dtype=np.float64,
        )
        confirmation = -commands[-12:]
        np.testing.assert_allclose(confirmation[:, 0], confirmation[:, 1])
        self.assertAlmostEqual(confirmation[0, 0], 1.0)
        self.assertAlmostEqual(confirmation[-1, 0], 20.0)
        np.testing.assert_allclose(confirmation[-4:, 0], 20.0)
        self.assertEqual(result["bilateral_ramp_steps_required"], 9)
        self.assertEqual(
            result["full_effort_confirmation_steps_required"],
            3,
        )

    def test_contact_that_slips_at_full_effort_cannot_confirm(self) -> None:
        result = {}
        ctx = SimpleNamespace(
            world=_DelayedSlipAtMaxEffortWorld(),
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        list(close_gripper(ctx, arm="right", timeout_s=1.5))

        self.assertFalse(result["ok"], result)
        self.assertFalse(result["grasp_confirmed"])
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertGreaterEqual(
            result["confirmation_motion_restart_count"],
            1,
        )
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)

    def test_single_contact_ramps_to_two_newtons_then_resets_on_motion(
        self,
    ) -> None:
        result = {}
        world = _SingleContactResumeWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        actions = list(close_gripper(ctx, arm="left", timeout_s=1.5))

        self.assertFalse(result["ok"])
        commands = np.asarray(
            [
                action["gripper_effort_left"]
                for action in actions
                if "gripper_effort_left" in action
            ],
            dtype=np.float64,
        )
        ramp_indices = np.flatnonzero(
            (np.abs(commands[:, 0]) > 0.5)
            & (np.abs(commands[:, 1]) == 0.5)
        )
        self.assertGreater(ramp_indices.size, 0, commands.tolist())
        peak_ramp_index = int(ramp_indices[-1])
        self.assertLessEqual(float(np.max(np.abs(commands[:, 0]))), 2.0)
        self.assertTrue(
            bool(
                np.any(
                    np.isclose(
                        np.abs(commands[peak_ramp_index + 1 :, 0]),
                        0.5,
                    )
                )
            ),
            commands.tolist(),
        )

    def test_initially_blocked_finger_and_later_stalled_finger_confirm(
        self,
    ) -> None:
        result = {}
        world = _OneFingerOnlyClosingWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )
        negative_depth = {
            "ok": True,
            "bilateral_depth_span_observed": False,
            "occlusion_limited_bilateral_blockage_observed": False,
        }

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=negative_depth,
        ):
            list(close_gripper(ctx, arm="right", timeout_s=1.0))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_moved"], [False, True])
        self.assertEqual(result["per_finger_blocked"], [True, True])
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [True, False],
        )
        self.assertTrue(result["grasp_confirmed"])
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)

    def test_short_timeout_after_initial_blockage_reports_confirmation_timeout(
        self,
    ) -> None:
        result = {}
        ctx = SimpleNamespace(
            world=_NoActuationWorld(),
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        actions = list(close_gripper(ctx, arm="right", timeout_s=0.3))

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["failure_stage"],
            "bilateral_contact_confirmation",
        )
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_blocked"], [True, True])
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [True, True],
        )
        self.assertFalse(result["actuation_detected"])
        self.assertEqual(result["max_abs_qpos_delta"], 0.0)
        self.assertEqual(result["max_closing_qpos_delta"], 0.0)
        self.assertEqual(result["peak_abs_qvel"], 0.0)
        effort_mode = (
            ACTION_SLICES["gripper_right"].stop
            - ACTION_SLICES["gripper_right"].start
            == 2
        )
        commanded = result["commanded_gripper_action"]
        if effort_mode:
            self.assertEqual(commanded[0], commanded[1])
            self.assertGreater(abs(float(commanded[0])), 0.5)
            self.assertLessEqual(abs(float(commanded[0])), 20.0)
        else:
            self.assertEqual(commanded, [-1.0])
        self.assertTrue(result["close_keepalive_active"])
        self.assertIn("did not remain stable", result["error"])
        self.assertEqual(len(actions), 11 if effort_mode else 10)
        if effort_mode:
            commands = np.asarray(
                [action[ACTION_SLICES["gripper_right"]] for action in actions]
            )
            self.assertTrue(np.all(commands[:9] <= 0.0))
            magnitudes = np.abs(commands[:9, 0])
            ramp_start = int(np.flatnonzero(magnitudes >= 1.0)[0])
            self.assertLessEqual(float(np.max(magnitudes[:ramp_start])), 0.5)
            self.assertTrue(bool(np.all(np.diff(magnitudes[ramp_start:]) >= 0.0)))
            self.assertEqual(result["peak_precontact_effort_n"], 0.5)
            np.testing.assert_allclose(
                commands[-2:],
                np.tile(np.asarray(commanded, dtype=np.float64), (2, 1)),
            )
        else:
            for action in actions[:-1]:
                np.testing.assert_allclose(
                    action[ACTION_SLICES["gripper_right"]],
                    [-1.0],
                )

    def test_initial_position_blockage_confirms_without_prior_travel(
        self,
    ) -> None:
        result = {}
        ctx = SimpleNamespace(
            world=_FullOpenNoisyVelocityWorld(),
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value={
                "ok": True,
                "bilateral_depth_span_observed": False,
            },
        ):
            list(close_gripper(ctx, arm="right", timeout_s=0.7))

        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["failure_stage"])
        self.assertTrue(result["grasp_confirmed"])
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_blocked"], [True, True])
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [True, True],
        )
        self.assertTrue(
            result["stationary_under_close_command_counts_as_contact"]
        )
        self.assertFalse(result["actuation_detected"])
        self.assertEqual(result["max_abs_qpos_delta"], 0.0)
        self.assertEqual(result["max_closing_qpos_delta"], 0.0)
        self.assertEqual(result["peak_abs_qvel"], 0.425)
        self.assertFalse(result["qvel_used_for_actuation_detection"])
        self.assertEqual(result["action_steps"], 16)
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)

    def test_empty_gripper_closing_to_lower_limits_is_not_contact(self) -> None:
        result = {}
        ctx = SimpleNamespace(
            world=_FreeClosingWorld(),
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        list(close_gripper(ctx, arm="right", timeout_s=1.0))

        self.assertFalse(result["ok"])
        self.assertFalse(result["grasp_confirmed"])
        self.assertEqual(result["failure_stage"], "empty_close")
        self.assertEqual(result["failure_reason"], "finger_lower_limits")
        self.assertEqual(result["per_finger_blocked"], [False, False])
        self.assertEqual(result["per_finger_moved"], [True, True])
        effort_mode = (
            ACTION_SLICES["gripper_right"].stop
            - ACTION_SLICES["gripper_right"].start
            == 2
        )
        if effort_mode:
            np.testing.assert_allclose(
                result["commanded_gripper_action"],
                [-0.5, -0.5],
            )

    def test_full_open_depth_blockage_completes_assisted_window(self) -> None:
        result = {}
        world = _FullOpenNoisyVelocityWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )
        evidence = {
            "ok": True,
            "bilateral_depth_span_observed": True,
            "inside_point_count": 26000,
            "spanning_component": {
                "side_0_point_count": 3000,
                "center_point_count": 10000,
                "side_1_point_count": 3000,
                "spans_opening": True,
            },
        }

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=evidence,
        ) as measure:
            actions = list(close_gripper(ctx, arm="right", timeout_s=3.5))

        self.assertTrue(result["ok"])
        self.assertTrue(result["grasp_confirmed"])
        self.assertTrue(result["assisted_grasp_window_completed"])
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertTrue(result["full_open_depth_contact_observed"])
        self.assertTrue(result["full_open_depth_contact_used"])
        self.assertFalse(result["constraint_directly_observed"])
        self.assertFalse(result["actuation_detected"])
        self.assertEqual(result["max_abs_qpos_delta"], 0.0)
        self.assertEqual(result["max_closing_qpos_delta"], 0.0)
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_plus_wrist_depth_"
            "full_open_blockage_window",
        )
        self.assertEqual(result["confirmation_steps_required"], 12)
        self.assertGreaterEqual(result["confirmation_steps"], 12)
        self.assertEqual(result["full_open_probe_steps_required"], 4)
        # Four low-force probe frames plus the unchanged 12-frame official
        # confirmation budget: the preload curve must not add latency.
        self.assertEqual(result["action_steps"], 16)
        self.assertEqual(result["peak_seek_effort_n"], 0.5)
        self.assertEqual(result["peak_precontact_effort_n"], 0.5)
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)
        commands = np.asarray(
            [
                action["gripper_effort_right"]
                for action in actions
                if "gripper_effort_right" in action
            ],
            dtype=np.float64,
        )
        confirmation = -commands[-12:]
        np.testing.assert_allclose(confirmation[:, 0], confirmation[:, 1])
        self.assertAlmostEqual(confirmation[0, 0], 1.0)
        self.assertAlmostEqual(confirmation[-1, 0], 20.0)
        self.assertTrue(bool(np.all(np.diff(confirmation[:, 0]) >= 0.0)))
        self.assertTrue(result["close_keepalive_active"])
        self.assertGreater(len(actions), 0)
        measure.assert_called_once()

    def test_qpos_blockage_does_not_wait_for_occlusion_evidence(self) -> None:
        result = {}
        world = _FullOpenNoisyVelocityWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )
        evidence = _occlusion_evidence()

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=evidence,
        ) as measure:
            list(close_gripper(ctx, arm="right", timeout_s=3.5))

        self.assertTrue(result["ok"])
        self.assertTrue(result["grasp_confirmed"])
        self.assertFalse(result["full_open_depth_contact_used"])
        self.assertEqual(result["full_open_occlusion_evidence_count"], 3)
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertFalse(result["constraint_directly_observed"])
        self.assertFalse(result["actuation_detected"])
        self.assertEqual(result["confirmation_steps_required"], 12)
        self.assertGreaterEqual(result["confirmation_steps"], 12)
        self.assertEqual(result["action_steps"], 16)
        self.assertTrue(result["close_keepalive_active"])
        self.assertEqual(measure.call_count, 3)

    def test_15061_near_open_asymmetric_blockage_confirms_without_extra_frames(
        self,
    ) -> None:
        result = {}
        world = _NearOpenAsymmetricBlockageWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=_occlusion_evidence(),
        ) as measure:
            list(close_gripper(ctx, arm="right", timeout_s=3.5))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertEqual(result["per_finger_moved"], [True, False])
        np.testing.assert_allclose(
            result["gripper_qpos_after"],
            [0.0479, 0.05],
            atol=1e-7,
        )
        self.assertTrue(result["actuation_detected"])
        self.assertTrue(result["wrist_depth_contact_observed"])
        self.assertFalse(result["wrist_depth_contact_used"])
        self.assertTrue(result["near_open_depth_contact_observed"])
        self.assertFalse(result["near_open_depth_contact_used"])
        self.assertFalse(result["full_open_depth_contact_observed"])
        self.assertFalse(result["full_open_depth_contact_used"])
        self.assertEqual(
            result["wrist_depth_contact_probe_mode"],
            "near_open",
        )
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertEqual(result["wrist_depth_evidence_count"], 3)
        self.assertEqual(result["confirmation_steps_required"], 12)
        # Five seek frames establish independent qpos stalls and the existing
        # twelve-frame window confirms while ramping 1 N -> 20 N. Wrist depth
        # remains diagnostic and is not a prerequisite for initial blockage.
        self.assertEqual(result["action_steps"], 17)
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)
        self.assertEqual(measure.call_count, 3)

    def test_near_open_qpos_drift_cannot_accumulate_depth_evidence(
        self,
    ) -> None:
        result = {}
        world = _NearOpenDriftingBlockageWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=_occlusion_evidence(),
        ):
            list(close_gripper(ctx, arm="right", timeout_s=1.0))

        self.assertFalse(result["ok"])
        self.assertFalse(result["wrist_depth_contact_observed"])
        self.assertFalse(result["near_open_depth_contact_observed"])
        self.assertLess(result["wrist_depth_evidence_count"], 3)

    def test_confirmation_drift_restarts_window_without_dropping_preload(
        self,
    ) -> None:
        result = {}
        world = _NearOpenConfirmationDriftWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=_occlusion_evidence(),
        ):
            actions = list(close_gripper(ctx, arm="right", timeout_s=1.2))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertEqual(result["candidate_reset_count"], 0)
        self.assertGreaterEqual(result["confirmation_motion_restart_count"], 1)
        self.assertGreater(
            result["max_contact_closure_drift_m"],
            0.0005,
        )
        commands = np.asarray(
            [
                action["gripper_effort_right"]
                for action in actions
                if "gripper_effort_right" in action
            ],
            dtype=np.float64,
        )
        bilateral = np.flatnonzero(
            np.all(np.abs(commands) >= 1.0 - 1e-7, axis=1)
        )
        self.assertGreater(len(bilateral), 0)
        self.assertTrue(
            bool(np.all(np.abs(commands[bilateral[0] :]) >= 1.0 - 1e-7)),
            commands.tolist(),
        )
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)

    def test_stationary_qpos_beyond_depth_margin_still_confirms_contact(
        self,
    ) -> None:
        result = {}
        world = _NearOpenAsymmetricBlockageWorld(
            blocked_qpos=(0.0455, 0.05)
        )
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=_occlusion_evidence(),
        ) as measure:
            list(close_gripper(ctx, arm="right", timeout_s=0.7))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertEqual(result["per_finger_blocked"], [True, True])
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [False, True],
        )
        self.assertFalse(result["wrist_depth_contact_observed"])
        self.assertFalse(result["near_open_depth_contact_observed"])
        measure.assert_not_called()

    def test_static_latched_retry_confirms_initial_position_blockage(
        self,
    ) -> None:
        result = {}
        world = _NearOpenAsymmetricBlockageWorld()
        world._action_count = 1
        world.latch_gripper_close_keepalive(
            "right",
            effort=[-0.5, -0.5],
        )
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )
        negative_depth = {
            "ok": True,
            "bilateral_depth_span_observed": False,
            "occlusion_limited_bilateral_blockage_observed": False,
        }

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            return_value=negative_depth,
        ) as measure:
            list(close_gripper(ctx, arm="right", timeout_s=3.5))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertTrue(result["close_started_with_keepalive"])
        self.assertFalse(result["latched_retry_recheck_completed"])
        self.assertEqual(result["action_steps"], 16)
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [True, True],
        )
        self.assertEqual(measure.call_count, 4)
        timeout_step_budget = int(np.ceil(3.5 * 30.0))
        self.assertGreaterEqual(
            timeout_step_budget / result["action_steps"],
            2.0,
        )

    def test_inconsistent_occlusion_does_not_override_qpos_blockage(
        self,
    ) -> None:
        result = {}
        world = _FullOpenNoisyVelocityWorld()
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda payload: result.update(payload),
            raise_if_cancelled=lambda where="": None,
        )

        with patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_live_wrist_opening_depth_evidence",
            side_effect=[
                _occlusion_evidence("side_0"),
                _occlusion_evidence("side_1"),
                _occlusion_evidence("side_0"),
                _occlusion_evidence("side_1"),
                _occlusion_evidence("side_0"),
            ],
        ) as measure:
            list(close_gripper(ctx, arm="right", timeout_s=1.0))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertFalse(result["full_open_depth_contact_observed"])
        self.assertEqual(result["full_open_occlusion_evidence_count"], 1)
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertEqual(measure.call_count, 4)


class WristOpeningDepthEvidenceTest(unittest.TestCase):
    @staticmethod
    def _measure(depth: np.ndarray) -> dict:
        return measure_wrist_opening_depth_evidence(
            depth_linear=depth,
            camera={
                "pos": [0.0, 0.0, 0.2],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "fx": 400.0,
                "fy": 400.0,
                "cx": 240.0,
                "cy": 240.0,
            },
            eef_pose={
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            gripper_qpos=[0.05, 0.05],
        )

    def test_connected_surface_spanning_opening_is_detected(self) -> None:
        depth = np.full((480, 480), 0.2, dtype=np.float32)

        result = self._measure(depth)

        self.assertTrue(result["ok"])
        self.assertTrue(result["bilateral_depth_span_observed"])
        self.assertTrue(result["spanning_component"]["spans_opening"])

    def test_center_only_surface_is_not_bilateral_evidence(self) -> None:
        depth = np.zeros((480, 480), dtype=np.float32)
        depth[230:251, :] = 0.2

        result = self._measure(depth)

        self.assertTrue(result["ok"])
        self.assertFalse(result["bilateral_depth_span_observed"])
        self.assertFalse(
            result["occlusion_limited_bilateral_blockage_observed"]
        )

    def test_opposed_occlusion_geometry_is_a_multiframe_candidate(self) -> None:
        depth = np.zeros((480, 480), dtype=np.float32)
        depth[155:271, 196:285] = 0.2
        depth[320:340, 196:285] = 0.2

        result = self._measure(depth)

        self.assertTrue(result["ok"])
        self.assertFalse(result["bilateral_depth_span_observed"])
        self.assertTrue(
            result["occlusion_limited_bilateral_blockage_observed"]
        )
        self.assertEqual(
            result["occlusion_limited_candidate"]["visible_side"],
            "side_1",
        )

    def test_one_sided_surface_without_opposing_boundary_is_rejected(self) -> None:
        depth = np.zeros((480, 480), dtype=np.float32)
        depth[155:271, 196:285] = 0.2

        result = self._measure(depth)

        self.assertTrue(result["ok"])
        self.assertFalse(result["bilateral_depth_span_observed"])
        self.assertFalse(
            result["occlusion_limited_bilateral_blockage_observed"]
        )


if __name__ == "__main__":
    unittest.main()
