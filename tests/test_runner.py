import contextlib
import hashlib
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


class SupplementalCases(unittest.TestCase):
    def test_new_instances_inherit_only_the_selected_setup(self):
        task = runner.load_task('task01')
        cases = runner.supplemental_cases(task, '302,303,305,307,309', 301)
        self.assertEqual([c['slot'] for c in cases], [1, 2, 4, 6, 8])
        for case in cases:
            self.assertEqual(case['context_reference_instance_id'], 301)
            self.assertEqual(case['prompt_sha256'], task['cases'][0]['prompt_sha256'])
            self.assertIsNone(case['reference_q'])
            self.assertIsNone(case['archive_reported_q'])
            self.assertNotIn('reference_result', case)
        self.assertEqual(task['cases'][0]['reference_q'], 1.0)
        for ids, template in [('301', 301), ('302,302', 301), ('321', 301),
                              ('2', 301), ('302', None), ('302', 302)]:
            with self.subTest(ids=ids, template=template), self.assertRaises(ValueError):
                runner.supplemental_cases(task, ids, template)

    def test_supplemental_score_has_no_fabricated_archive_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = runner.load_task('task03')
            cases = runner.supplemental_cases(task, '302', 301)
            plan = {'run_dir': directory, 'run_id': 'supplemental-score',
                    'task_config': task, 'cases': cases, 'port': 15073,
                    'harness': 'claude_code', 'evaluation_scope': 'supplemental'}
            run = runner.Run(plan, {'session_timeout_s': 0, 'model': task['model']})
            case_dir = root / 'instance_302'
            case_dir.mkdir()
            score = {'task': task['task_name'], 'instance_id': 302, 'rollout_id': 0,
                     'q_score': {'final': 2/7}, 'steps': 20, 'success': False}
            path = root / 'output/json' / f'{task["task_name"]}_302_0.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(score))
            agent = Mock()
            agent.poll.return_value = 0
            with patch.object(runner, 'native_contract', return_value={}) as contract, \
                    patch.object(runner, 'inspect_native_case', return_value={'state': 'match'}):
                run.wait_score(cases[0], agent, case_dir)
            contract.assert_called_once_with('task03', 301)
            run.write_summary('complete')
            summary = runner.read_json(root / 'summary.json')
            self.assertEqual(summary['mean_q'], 2/7)
            self.assertEqual(summary['evaluation_scope'], 'supplemental')
            self.assertIsNone(summary['reference_mean_q'])
            self.assertIsNone(summary['archive_reported_mean_q'])
            self.assertIsNone(summary['delta_archive_mean_q'])
            self.assertIsNone(summary['cases'][0]['delta_q'])


class FileDescriptorLimit(unittest.TestCase):
    def test_limit_is_inherited_without_changing_the_parent_or_hard_limit(self):
        import resource
        before = resource.getrlimit(resource.RLIMIT_NOFILE)
        code = '''
import json, resource, subprocess, sys
from roboharness.runner import ensure_file_limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
record = ensure_file_limit(4096)
assert record['before_soft'] == 1024 and record['soft'] == 4096 and record['hard'] == hard
subprocess.run([sys.executable, '-c', 'import resource; assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 4096'], check=True)
assert ensure_file_limit(2048)['soft'] == 4096
'''
        subprocess.run([sys.executable, '-c', code], cwd=runner.ROOT, check=True, timeout=10)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_NOFILE), before)

    def test_insufficient_hard_limit_is_rejected_before_launch(self):
        with patch.object(runner.resource, 'getrlimit', return_value=(1024, 2048)), \
                patch.object(runner.resource, 'setrlimit') as setter:
            with self.assertRaisesRegex(ValueError, 'hard limit'):
                runner.ensure_file_limit(4096)
            setter.assert_not_called()


class NativeContextRuntime(unittest.TestCase):
    def test_agent_launch_uses_private_workspace_without_inherited_git_settings(self):
        with tempfile.TemporaryDirectory(dir='/var/tmp') as directory:
            root = Path(directory)
            task = runner.load_task('task01')
            config = root / 'config.json'
            config.write_text('{}')
            run = runner.Run({'run_dir': directory, 'run_id': 'test-isolated-workspace',
                              'task_config': task, 'port': 15079, 'harness': 'claude_code'},
                             runner.read_config(config))
            run.env = {'XDG_RUNTIME_DIR': str(root / 'runtime'), 'GIT_DIR': '/borrowed/git',
                       'GIT_WORK_TREE': '/borrowed/worktree', 'GIT_INDEX_FILE': '/borrowed/index'}
            run.spawn = Mock()
            run.write_summary = Mock()
            run.log = Mock()
            with patch.object(runner, 'request_json', side_effect=lambda port, route, body, **kw: {'ok': True, **body}):
                run.start_agent(task['cases'][0])
            actual = run.spawn.call_args
            self.assertEqual(actual.kwargs['cwd'], root / 'runtime/agent-workspaces/instance_301/embodied_claude_code')
            self.assertFalse(any(k.startswith('GIT_') for k in actual.args[2]))
            self.assertEqual(run.env['GIT_DIR'], '/borrowed/git')
            self.assertEqual(json.loads((root / 'instance_301/case.json').read_text())['agent_cwd'], str(actual.kwargs['cwd']))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.case_dir = self.root / 'instance_301'
        folder = self.case_dir / 'claude-home/projects/project'
        folder.mkdir(parents=True)
        self.transcript = folder / 'one.jsonl'
        self.content = '- robot-skill: archived description'
        self.expected = {'pattern': {'sha256': hashlib.sha256(self.content.encode()).hexdigest(),
                                     'names': ['robot-skill']}}
        self.case = {'instance_id': 301, 'reference_q': 1.0}
        self.result = {'q_score': {'final': 1.0}, 'steps': 7, 'success': True}
        output = self.root / 'output/json/sample_301_0.json'
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(self.result))
        self.run = runner.Run({'run_dir': directory.name, 'run_id': 'test-native-context',
                               'task_config': {'task': 'task01', 'task_name': 'sample'},
                               'port': 15071, 'harness': 'claude_code'}, {'session_timeout_s': 0})
        self.run.assert_alive = Mock()
        self.run.finish_episode = Mock()
        self.run.log = Mock()
        self.run.write_summary = Mock()
        self.agent = Mock()
        patcher = patch.object(runner, 'native_contract', return_value=self.expected)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_listing(self, content=None):
        self.transcript.write_text(json.dumps({'attachment': {'type': 'skill_listing',
            'isInitial': True, 'content': self.content if content is None else content,
            'names': ['robot-skill']}}) + '\n')

    def test_divergence_stops_before_waiting_for_an_official_score(self):
        self.write_listing('changed description')
        with patch.object(runner, 'official_result', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'native context mismatch'):
                self.run.wait_score(self.case, self.agent, self.case_dir)
        self.agent.poll.assert_not_called()
        self.run.finish_episode.assert_not_called()
        self.assertEqual(self.run.results, [])
        audit = json.loads((self.case_dir / 'native-context.check.json').read_text())
        self.assertEqual(audit['state'], 'mismatch')
        self.assertFalse((self.case_dir / 'comparison.json').exists())

    def test_a_score_without_native_evidence_does_not_count_as_a_completed_case(self):
        with patch.object(runner, 'official_result', return_value=self.result):
            with self.assertRaisesRegex(RuntimeError, 'native context missing'):
                self.run.wait_score(self.case, self.agent, self.case_dir)
        self.assertEqual(self.run.results, [])

    def test_incomplete_startup_waits_and_completion_rechecks_the_transcript(self):
        self.transcript.write_text('{"attachment":')
        self.agent.poll.side_effect = [None, None, 0]
        with patch.object(runner, 'official_result', side_effect=[None, None, self.result]), \
             patch.object(runner.time, 'sleep', side_effect=lambda _: self.write_listing()), \
             patch.object(runner, 'request_json', return_value={'session_ticks': 7}), \
             patch.object(runner, 'inspect_native_case', wraps=runner.inspect_native_case) as inspect:
            self.run.wait_score(self.case, self.agent, self.case_dir)
        self.assertEqual(inspect.call_count, 3)
        self.assertTrue(inspect.call_args.kwargs['finished'])
        self.run.finish_episode.assert_not_called()
        self.assertEqual(self.run.results[0]['q'], 1.0)
        self.assertEqual(json.loads((self.case_dir / 'native-context.check.json').read_text())['state'], 'match')

    def test_a_previous_verified_audit_cannot_hide_changed_completion_evidence(self):
        self.write_listing()
        self.agent.poll.return_value = None
        with patch.object(runner, 'official_result', side_effect=[None, self.result]), \
             patch.object(runner.time, 'sleep', side_effect=lambda _: self.write_listing('changed')), \
             patch.object(runner, 'request_json', return_value={'session_ticks': 7}):
            with self.assertRaisesRegex(RuntimeError, 'native context mismatch'):
                self.run.wait_score(self.case, self.agent, self.case_dir)
        self.assertEqual(self.run.results, [])
        self.assertEqual(json.loads((self.case_dir / 'native-context.check.json').read_text())['state'], 'mismatch')

    def test_agent_exit_without_initial_context_does_not_force_a_submission(self):
        (self.case_dir / 'agent.json').write_text(json.dumps({'type': 'result', 'is_error': False}))
        self.agent.poll.return_value = 0
        with patch.object(runner, 'official_result', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'native context missing'):
                self.run.wait_score(self.case, self.agent, self.case_dir)
        self.run.finish_episode.assert_not_called()


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
