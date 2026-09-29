import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from behavior_interface.agent_monitor import challenge_instance_max_ticks
from official_eval_harness.catalog import challenge_2025_max_ticks, resolve_task
from roboharness import runner


class ArchivedBudget(unittest.TestCase):
    def test_runner_catalog_and_monitor_use_archived_limits(self):
        expected = {
            0: 4299, 1: 10535, 2: 27664, 3: 27392, 5: 20343,
            6: 15239, 7: 37781, 8: 17886, 9: 27437,
        }
        # Exercise the monitor's fallback without the launcher's override.
        with patch.dict(os.environ, {}, clear=True):
            for index, ticks in expected.items():
                with self.subTest(task=index):
                    task = runner.load_task(f'task{index:02d}')
                    self.assertEqual(task['max_steps'], ticks)
                    self.assertEqual(task['challenge_year'], 2025)
                    self.assertEqual(task['budget_multiplier'], 2)
                    self.assertEqual(resolve_task(task['task']).timeout_steps, ticks)
                    self.assertEqual(challenge_2025_max_ticks(task_id=index), ticks)
                    self.assertEqual(
                        challenge_instance_max_ticks(task_name=task['task_name']), ticks)

    def test_launch_rejects_changed_budget_or_protocol(self):
        task = runner.load_task('task01')
        variants = [
            {'max_steps': int(task['max_steps'] * 0.75)},
            {'max_steps': task['max_steps'] + 1},
            {'max_steps': float(task['max_steps'])},
            {'challenge_year': 2026},
            {'budget_multiplier': 1.5},
            {'evaluator_commit': 'different-evaluator'},
            {'protocol': 'challenge-2026'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tasks').mkdir()
            for change in variants:
                with self.subTest(change=change):
                    (root / 'tasks/task01.json').write_text(json.dumps({**task, **change}))
                    with patch.object(runner, 'ROOT', root):
                        with self.assertRaisesRegex(ValueError, 'Challenge 2025'):
                            runner.load_task('task01')


if __name__ == '__main__':
    unittest.main()
