import contextlib
import io
import json
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from roboharness import runner


class ArchivedCases(unittest.TestCase):
    def test_model_or_harness_overrides_are_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{}')
            for args, diagnostic in (([], False), (['--model', 'different-model'], True),
                                     (['--harness', 'codex'], True)):
                with self.subTest(args=args):
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        code = runner.main(['--task', 'task01', '--config', str(config),
                                            '--dry-run', *args])
                    self.assertEqual(code, 0)
                    self.assertEqual(json.loads(stream.getvalue())['diagnostic_only'], diagnostic)

    def test_all_cases_have_trajectory_authoritative_scores(self):
        for path in sorted((runner.ROOT / 'tasks').glob('*.json')):
            task = runner.load_task(path.stem)
            with self.subTest(task=path.stem):
                self.assertTrue(all('archive_reported_q' in case for case in task['cases']))
                self.assertTrue(all(case.get('archive_score_basis') == 'trajectory-directory official score'
                                    for case in task['cases']
                                    if path.stem not in {'task00', 'task05'}))

    def test_all_cases_have_exact_prompts_and_unmodified_scores(self):
        used = set()
        count = 0
        contexts = runner.read_json(runner.ROOT /
            'harness/claude_code/tests/fixtures/archive_context.json')['sources']
        contexts = {(row['task'], row['instance_id']): row for row in contexts}
        for path in sorted((runner.ROOT / 'tasks').glob('*.json')):
            task = runner.load_task(path.stem)
            self.assertEqual([301, 304, 306, 308, 310], [c['instance_id'] for c in task['cases']])
            for case in task['cases']:
                result = runner.official_result(runner.ROOT / case['reference_result'], task, case['instance_id'])
                self.assertEqual(result['q_score']['final'], case['reference_q'])
                self.assertEqual(case['slot'] + 301, case['instance_id'])
                context = contexts[(task['task'], case['instance_id'])]
                self.assertEqual(case['claude_mcp_name'], context['mcp_server_name'])
                self.assertEqual(case['prompt_sha256'], context['prompt_sha256'])
                self.assertEqual(case['prompt_source_sha256'], context['sha256'])
                used.add(case['prompt'])
                count += 1
        self.assertEqual(count, 45)
        self.assertEqual(len(contexts), count)
        self.assertEqual(used, {str(p.relative_to(runner.ROOT)) for p in (runner.ROOT / 'prompt').rglob('*.txt')})

    def test_instance_ids_are_not_silently_interpreted_as_slots(self):
        task = runner.load_task('task01')
        with self.assertRaises(ValueError):
            runner.select_cases(task, '0,3,5')
        with self.assertRaises(ValueError):
            runner.select_cases(task, '301,301')
        self.assertEqual([310, 301], [c['instance_id'] for c in runner.select_cases(task, '310,301')])

    def test_stale_or_nonfinite_score_is_rejected(self):
        task = runner.load_task('task01')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'score.json'
            body = {'task': task['task_name'], 'instance_id': 304, 'rollout_id': 0,
                    'steps': 2, 'q_score': {'final': 1}}
            path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                runner.official_result(path, task, 301)
            body['instance_id'] = 301
            body['q_score']['final'] = float('nan')
            path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                runner.official_result(path, task, 301)


class WallClockBudget(unittest.TestCase):
    def test_cap_requires_explicit_finite_non_negative_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{}')
            self.assertEqual(runner.read_config(path)['session_timeout_s'], 0)
            for value in (0, 86400):
                path.write_text(json.dumps({'session_timeout_s': value}))
                self.assertEqual(runner.read_config(path)['session_timeout_s'], value)
            for value in (-1, True, None, '86400', float('nan'), float('inf')):
                with self.subTest(value=value):
                    path.write_text(json.dumps({'session_timeout_s': value}))
                    with self.assertRaisesRegex(ValueError, 'session_timeout_s'):
                        runner.read_config(path)

    def exercise_wait(self, timeout, clock_step):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {'q_score': {'final': 1.0}, 'steps': 7, 'success': True}
            output = root / 'output/json/sample_301_0.json'
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps(result))
            case_dir = root / 'instance_301'
            case_dir.mkdir()
            run = runner.Run({'run_dir': directory, 'run_id': 'test-wall-clock',
                              'task_config': {'task_name': 'sample'}, 'port': 15071,
                              'harness': 'codex'}, {'session_timeout_s': timeout})
            run.assert_alive = Mock()
            run.finish_episode = Mock()
            run.log = Mock()
            run.write_summary = Mock()
            agent = Mock()
            agent.poll.side_effect = [None, None, 0]
            with patch.object(runner, 'official_result', side_effect=[None, None, result]), \
                 patch.object(runner.time, 'monotonic', side_effect=itertools.count(0, clock_step)), \
                 patch.object(runner.time, 'sleep'), \
                 patch.object(runner, 'request_json', return_value={'session_ticks': 7}):
                run.wait_score({'instance_id': 301, 'reference_q': 1.0}, agent, case_dir)
            return run

    def test_disabled_wall_cap_does_not_submit_after_large_clock_advances(self):
        run = self.exercise_wait(0, 100000)
        run.finish_episode.assert_not_called()
        self.assertEqual(run.results[0]['finish_reason'], 'evaluator_end')

    def test_opted_in_wall_cap_retains_forced_submission_reason(self):
        run = self.exercise_wait(1, 1)
        run.finish_episode.assert_called_once_with('wall_timeout')
        self.assertEqual(run.results[0]['finish_reason'], 'wall_timeout')


class Ownership(unittest.TestCase):
    def test_case_handoff_waits_for_reset_and_request_acknowledgement(self):
        health = {'evaluator_connected': True, 'episode_initialization': {'ready': True},
                  'action_source': 'hold'}
        status = {'current_instance_id': 304, 'state': 'ready'}
        self.assertFalse(runner.handoff_ready(301, health, status, {}))
        request = {'op': 'finish', 'request_id': 'previous-case'}
        self.assertFalse(runner.handoff_ready(304, health, status, request))
        status['applied_request_id'] = request['request_id']
        self.assertTrue(runner.handoff_ready(304, health, status, request))
        for state in ('resetting', 'error', 'queued', 'finish_queued'):
            self.assertFalse(runner.handoff_ready(304, health, {**status, 'state': state}, request))
        self.assertFalse(runner.handoff_ready(304, {**health, 'action_source': 'tool'}, status, {}))
        self.assertFalse(runner.handoff_ready(304, {**health, 'episode_initialization': {'ready': False}}, status, {}))

    def test_cleanup_does_not_kill_another_run_with_similar_token(self):
        token = 'test-' + uuid.uuid4().hex
        children = []
        try:
            for value in (token, token + '-other'):
                children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                    env={**os.environ, 'ROBOHARNESS_RUN_TOKEN': value}))
            runner.cleanup(token, grace=0.1)
            self.assertIsNotNone(children[0].wait(timeout=3))
            self.assertIsNone(children[1].poll())
        finally:
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=3)

    def test_reused_pid_is_never_signalled(self):
        with patch.object(runner, 'owned_pids', return_value=[1234]), \
             patch.object(runner, 'process_start', side_effect=['old', 'new', 'new']), \
             patch.object(runner.os, 'kill') as kill:
            runner.cleanup('example', grace=0)
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
