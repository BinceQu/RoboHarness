from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from behavior_interface_eval_test.official_evaluator_v393 import route_evaluator_class
from behavior_interface_eval_test.rollout_budget import ROLLOUT_BUDGET_KEY
from behavior_interface_eval_test.operator_scene_control import (
    install_operator_reset_on_evaluator_class, write_finish_request, read_status,
)


class FakeBatchedEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []
        self.instance_eval_states = [SimpleNamespace(
            obs={"rgb": object(), "proprio": object()}, video_writer=None,
            metrics=[SimpleNamespace(aggregate=lambda: {"q_score": {"final": 0.25}})],
            env_accessor=SimpleNamespace(success=False))]

    def load_batch(self, mapping, **kwargs):
        self.calls.append(("load", dict(mapping), kwargs))

    def reset(self):
        self.calls.append(("reset",))

    def _step_fn(self, indices):
        self.calls.append(("step", list(indices)))
        return [False], [False]

    def _write_video(self, state):
        self.calls.append(("video", state))

    def _set_video_writer(self, state, writer):
        self.calls.append(("writer", writer))
        state.video_writer = writer


class RouteAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cls = route_evaluator_class(FakeBatchedEvaluator)
        self.route = self.cls({"num_envs": 1, "max_steps": 123})

    def test_preserves_timeout_and_native_observations(self):
        self.assertEqual(self.route.cfg["max_steps"], 123)
        obs = self.route._batch_obs()
        for key, value in self.route.route_state.obs.items():
            self.assertIs(obs[key], value)
        self.assertEqual(set(obs) - set(self.route.route_state.obs), {ROLLOUT_BUDGET_KEY})
        self.assertNotIn(ROLLOUT_BUDGET_KEY, self.route.route_state.obs)
        self.assertFalse(obs[ROLLOUT_BUDGET_KEY]['available'])

    def test_budget_uses_actual_timeout_and_episode_counter(self):
        self.route.env = SimpleNamespace(episode_steps=[60], task=SimpleNamespace(
            _termination_conditions={'timeout': SimpleNamespace(_max_steps=200)}))
        self.route.route_state.instance_id = 301
        budget = self.route._batch_obs()[ROLLOUT_BUDGET_KEY]
        self.assertEqual((budget['used_ticks'], budget['total_ticks'], budget['used_fraction']),
                         (60, 200, 0.3))
        self.assertEqual(budget['remaining_ticks'], 140)
        self.assertEqual(budget['instance_id'], 301)
        self.assertEqual(self.route.calls, [])
        self.route.env.episode_steps[0] = 201
        budget = self.route._rollout_budget()
        self.assertGreater(budget['used_fraction'], 1)
        self.assertEqual(budget['remaining_ticks'], 0)

    def test_resident_clock_does_not_report_idle_sentinel(self):
        self.route.route_state.instance_id = 301
        self.route._resident_tick_budget = (70, 200)
        self.assertEqual(self.route._rollout_budget()['used_ticks'], 70)
        old = self.route._rollout_budget()['episode_id']
        self.route.reset()
        now = self.route._rollout_budget()
        self.assertNotEqual(now['episode_id'], old)
        self.assertEqual(now['used_ticks'], 0)
        self.assertEqual(now['total_ticks'], 200)
        self.assertEqual(now['source'], 'resident_active_steps')

    def test_rejects_cross_environment_load_or_step(self):
        with self.assertRaises(ValueError):
            self.cls({"num_envs": 2})
        for mapping in ({1: 301}, {0: 301, 1: 302}):
            with self.assertRaises(ValueError):
                self.route.load_batch(mapping)
        with self.assertRaises(ValueError):
            self.route._step_fn([1])
        self.assertEqual(self.route.calls, [])

    def test_native_step_called_exactly_once(self):
        self.assertEqual(self.route._step_fn([0]), ([False], [False]))
        self.assertEqual(self.route.calls, [("step", [0])])

    def test_finish_does_not_send_another_policy_action(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "BEHAVIOR_EVAL_OPERATOR_DIR": temp, "BEHAVIOR_EVAL_TEST_PORT": "15010"
        }):
            install_operator_reset_on_evaluator_class(self.cls, 15010)
            self.route.load_batch({0: 304})
            self.assertEqual(read_status(15010)["current_instance_id"], 304)
            write_finish_request(15010)
            self.assertEqual(self.route._step_fn([0]), ([False], [True]))
            self.assertFalse(any(call[0] == "step" for call in self.route.calls))
            self.assertEqual(read_status(15010)["state"], "submitted")

    def test_metrics_use_native_v393_aggregation(self):
        self.assertEqual(self.route.resident_result(), (False, {"q_score": {"final": 0.25}}))

    def test_no_idle_video_until_writer_is_armed(self):
        self.route.record_resident_frame()
        self.assertEqual(self.route.calls, [])
        self.route.route_state.video_writer = object()
        self.route.record_resident_frame()
        self.assertEqual(len(self.route.calls), 1)
        self.route.stop_recording()
        self.assertIsNone(self.route.route_state.video_writer)


if __name__ == "__main__":
    unittest.main()
