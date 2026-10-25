"""Budget transport contract tests; never import or start Isaac/OmniGibson."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any
import unittest

from flask import Flask

from behavior_interface_eval_test.rollout_budget import (
    ROLLOUT_BUDGET_KEY, budget_snapshot, sanitize_budget, unavailable,
    install_rollout_budget_routes,
)


def payload(used=25, total=100, episode='episode-1'):
    return budget_snapshot(used, total, episode_id=episode, instance_id=301,
                           source='official_evaluator_episode_steps')


class BudgetTests(unittest.TestCase):
    def test_invalid_counters_fail_open_and_never_leak_privileged_fields(self):
        for used, total in [(-1, 100), (True, 100), ('1', 100), (1, 0), (1, float('inf'))]:
            got = payload(used, total)
            self.assertFalse(got['available'])
            self.assertIsNone(got['used_fraction'])
        for extra in [{'source': []}, {'episode_id': 'bad\nidentity'}]:
            self.assertFalse(sanitize_budget({**payload(), **extra})['available'])
        safe = sanitize_budget({**payload(), 'goal_truth': [True], 'q_score': .99})
        self.assertNotIn('goal_truth', safe)
        self.assertNotIn('q_score', safe)
        self.assertEqual(safe['used_fraction'], .25)

    def test_endpoint_is_read_only_cached_and_disconnect_aware(self):
        connected = [True]
        runtime = SimpleNamespace(server=SimpleNamespace(),
            adapter=SimpleNamespace(rollout_budget=lambda: payload()),
            evaluator_connections=SimpleNamespace(snapshot=lambda: (connected[0], 1)))
        app = Flask(__name__)
        install_rollout_budget_routes(app, runtime)
        client = app.test_client()
        response = client.get('/api/rollout_budget')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['used_ticks'], 25)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(client.post('/api/rollout_budget').status_code, 405)
        self.assertEqual(runtime.server.rollout_budget_snapshot(), payload())
        connected[0] = False
        self.assertFalse(client.get('/api/rollout_budget').json['available'])
        connected[0] = True
        runtime.adapter = None
        self.assertFalse(client.get('/api/rollout_budget').json['available'])

    def test_adapter_strips_operational_metadata_and_resets_between_rollouts(self):
        # Execute the real adapter class in a sensor-free namespace; importing
        # the full runtime would pull in GPU/tracker libraries unnecessarily.
        path = Path(__file__).with_name('official_policy_interface.py')
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
                 n.name in {'ObservationSnapshot', 'ObservationActionAdapter'}]
        namespace = {'__name__': __name__, 'Any': Any, 'dataclass': dataclass,
            'threading': threading, 'time': time, 'ROLLOUT_BUDGET_KEY': ROLLOUT_BUDGET_KEY,
            'sanitize_budget': sanitize_budget, 'unavailable_budget': unavailable,
            'UI_BDDL_PROGRESS_KEY': '__ui_goal__',
            'awaiting_bddl_progress': lambda *args: {},
            '_copy_observation': deepcopy,
            'filter_allowed_observation': lambda obs: ({k:v for k,v in obs.items() if k == 'rgb'},
                                                       [k for k in obs if k != 'rgb']),
            'np': SimpleNamespace(zeros=lambda *args, **kwargs: [0]*4, float32=float),
            'ACTION_DIM': 4, 'ACTION_SLICES': {'gripper_left': slice(0,2), 'gripper_right': slice(2,4)}}
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), 'exec'), namespace)
        adapter = namespace['ObservationActionAdapter']()
        obs = adapter.update({'rgb': [1,2,3], ROLLOUT_BUDGET_KEY: payload()})
        self.assertEqual(obs.observation, {'rgb': [1,2,3]})
        self.assertEqual(obs.rejected_keys, [])
        self.assertEqual(adapter.rollout_budget()['used_ticks'], 25)
        adapter.update({'rgb': [4,5,6]})
        self.assertFalse(adapter.rollout_budget()['available'])
        adapter.update({'rgb': [7], ROLLOUT_BUDGET_KEY: payload(0, episode='episode-2')})
        self.assertEqual(adapter.rollout_budget()['episode_id'], 'episode-2')
        adapter.reset()
        self.assertFalse(adapter.rollout_budget()['available'])


if __name__ == '__main__':
    unittest.main()
