"""Exercise resident rollout/reset and artifact boundaries without Isaac."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from behavior_interface_eval_test import official_evaluator_entrypoint as entry
from behavior_interface_eval_test.official_evaluator_v393 import route_evaluator_class


class ResidentResetLifecycleTest(unittest.TestCase):
    def run_driver(self, *, load_resets, rollouts=2, initially_idle=False):
        events = []

        class FakeEvaluator:
            load_task_instance_resets_rollout = load_resets

            def __init__(self, cfg):
                self.prepared = False
                self.steps = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def load_task_instance(self, instance):
                events.append(("load", instance))
                self.instance = instance
                if load_resets:
                    self.prepared = True
                    self.steps = 0

            def reset(self):
                events.append(("reset", self.instance))
                self.prepared = True
                self.steps = 0

            def step(self):
                if not self.prepared:
                    raise AssertionError("rollout entered without fresh initial metrics")
                self.steps += 1
                self.prepared = False
                events.append(("step", self.instance))
                return True, False

            def start_recording(self, path, rate):
                events.append(("record", self.instance))

            def stop_recording(self):
                events.append(("close_video", self.instance))

            def resident_result(self):
                return False, {"q_score": {"final": 0.25},
                               "time": {"simulator_steps": self.steps}}

        fake_modules = {
            "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda cfg: cfg)),
            "omnigibson.eval.evaluator": SimpleNamespace(
                Evaluator=FakeEvaluator, resolve_instance_ids=lambda *a, **k: [301, 302]),
            "omnigibson.eval.utils.eval_utils": SimpleNamespace(
                DEFAULT_EVAL_SEED=7, seed_everything=lambda seed: seed),
            "omnigibson.macros": SimpleNamespace(gm=SimpleNamespace()),
        }
        activity = iter([False, False] if initially_idle else [])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(headless=True, task_name="test_task",
                instance_indices=[0, 1], mode="public", robot_config=None,
                policy="websocket", host="127.0.0.1", port=8000,
                env_wrapper="test_wrapper", write_video=True, output_dir=tmp,
                num_rollouts=rollouts, video_fps=30, max_steps=3)
            module = SimpleNamespace(parse_args=lambda: args, logger=mock.Mock())
            with mock.patch.dict("sys.modules", fake_modules), \
                 mock.patch("itertools.cycle", side_effect=iter), \
                 mock.patch.object(entry, "resident_action_active",
                                   side_effect=lambda: next(activity, True)):
                entry.run_resident_evaluator(module, evaluator_cls=FakeEvaluator)
            metrics = [json.loads(p.read_text()) for p in sorted((Path(tmp) / "json").glob("*.json"))]
        return events, metrics

    def test_official_load_initializes_once_and_later_rollouts_reset(self):
        events, metrics = self.run_driver(load_resets=True)
        self.assertEqual([e for e in events if e[0] == "reset"],
                         [("reset", 301), ("reset", 302)])
        self.assertEqual([(m["instance_id"], m["rollout_id"]) for m in metrics],
                         [(301, 0), (301, 1), (302, 0), (302, 1)])
        for m in metrics:
            self.assertEqual(m["steps"], 1)
            self.assertEqual(m["time"]["simulator_steps"], 1)
            self.assertEqual(m["q_score"]["final"], 0.25)
        self.assertEqual(sum(e[0] == "record" for e in events), 4)
        self.assertEqual(sum(e[0] == "close_video" for e in events), 4)

    def test_legacy_evaluator_still_resets_before_every_rollout(self):
        events, metrics = self.run_driver(load_resets=False)
        self.assertEqual(sum(e[0] == "reset" for e in events), 4)
        self.assertEqual(len(metrics), 4)

    def test_inactive_initialization_does_not_emit_artifacts_or_double_reset(self):
        events, metrics = self.run_driver(load_resets=True, rollouts=1, initially_idle=True)
        self.assertEqual([e for e in events if e[0] == "load"],
                         [("load", 301), ("load", 301), ("load", 302)])
        self.assertFalse(any(e[0] == "reset" for e in events))
        self.assertEqual(len(metrics), 2)
        self.assertEqual(sum(e[0] == "record" for e in events), 2)

    def test_only_official_route_advertises_load_reset_contract(self):
        self.assertTrue(route_evaluator_class(object).load_task_instance_resets_rollout)


if __name__ == "__main__":
    unittest.main()
