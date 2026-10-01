import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from roboharness import runner


class ListenerPorts(unittest.TestCase):
    def setUp(self):
        self.task = runner.load_task('task01')
        self.ports = {'http': 15071, 'policy': 15070, 'gate': 15072}
        self.config = {'task_ports': {'task01': self.ports}}

    def test_defaults_and_explicit_http_override_remain_compatible(self):
        self.assertEqual(runner.listener_ports(self.task, {}),
                         {'http': 15071, 'policy': 16071, 'gate': 17071})
        self.assertEqual(runner.listener_ports(self.task, {}, 15079),
                         {'http': 15079, 'policy': 16079, 'gate': 17079})
        self.assertEqual(runner.listener_ports(self.task, self.config, 15079),
                         {'http': 15079, 'policy': 15070, 'gate': 15072})
        self.assertEqual(self.config['task_ports']['task01'], self.ports)

    def test_incomplete_invalid_and_colliding_ports_fail_before_launch(self):
        for invalid in (None, [], {}, {'http': 15071},
                        {**self.ports, 'gate': 15070},
                        {**self.ports, 'gate': '15072'},
                        {**self.ports, 'gate': True},
                        {**self.ports, 'gate': 1023},
                        {**self.ports, 'gate': 65536}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.listener_ports(self.task, {'task_ports': {'task01': invalid}})
        for invalid in (None, []):
            with self.assertRaises(ValueError):
                runner.listener_ports(self.task, {'task_ports': invalid})
        for invalid in (0, 63536):
            with self.assertRaises(ValueError):
                runner.listener_ports(self.task, {}, invalid)
        with self.assertRaises(ValueError):
            runner.listener_ports(self.task, self.config, 15070)

    def test_session_wrapper_preserves_mapping_and_does_not_rewrite_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'session-config.json'
            # A non-default HTTP selection catches the old wrapper override.
            selected = {**self.ports, 'http': 15079}
            config.write_text(json.dumps({'task_ports': {'task01': selected}}))
            before = config.read_bytes()
            process = subprocess.run(
                [str(runner.ROOT / 'scripts/reproduce_task.sh'), 'task01', '--dry-run'],
                cwd=runner.ROOT, env={**os.environ, 'ROBOHARNESS_SESSION_CONFIG': str(config)},
                text=True, capture_output=True, check=True)
            plan = json.loads(process.stdout)
            self.assertEqual([plan['port'], plan['policy_port'], plan['gate_port']],
                             [15079, 15070, 15072])
            self.assertEqual(config.read_bytes(), before)

    def test_real_prepare_and_stack_commands_share_the_selected_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'session.json'
            config_path.write_text(json.dumps({**self.config, 'cache_dir': str(root / 'cache')}))
            plan = {'run_dir': str(root / 'run'), 'run_id': 'test-session-ports',
                    'task_config': self.task, 'cases': self.task['cases'],
                    'gpu': 5, 'port': 15071, 'policy_port': 15070, 'gate_port': 15072,
                    'write_video': False}
            run = runner.Run(plan, runner.read_config(config_path))
            run.spawn = Mock()
            run.assert_alive = Mock()
            with patch.object(runner.subprocess, 'check_output', return_value=str(root / 'data')):
                run.prepare()
            self.assertEqual(run.env['BEHAVIOR_EVAL_TEST_POLICY_PORT'], '15070')
            self.assertEqual(json.loads((run.path / 'plan.json').read_text())['gate_port'], 15072)
            with patch.object(runner, 'request_json', side_effect=[
                    {'session_isolation': {'enabled': True}},
                    {'fail_closed': True, 'max_idle_wait_s': 0}]) as request:
                run.start_stack()
            commands = {call.args[0]: call.args[1] for call in run.spawn.call_args_list}
            def argument(role, option):
                args = commands[role]
                return args[args.index(option) + 1]
            self.assertEqual(argument('interface', '--http-port'), '15071')
            self.assertEqual(argument('interface', '--policy-port'), '15070')
            self.assertEqual(argument('gate', '--listen-port'), '15072')
            self.assertEqual(argument('gate', '--backend-policy-uri'), 'ws://127.0.0.1:15070')
            self.assertEqual(argument('gate', '--backend-http-url'), 'http://127.0.0.1:15071')
            self.assertEqual(argument('evaluator', '--port'), '15072')
            self.assertEqual([call.args for call in request.call_args_list],
                             [(15071, '/__official__/healthz'), (15072, '/status')])

    def test_mapped_policy_lock_prevents_starting_an_overlapping_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / 'locks'
            locks.mkdir()
            config = root / 'session.json'
            config.write_text(json.dumps({**self.config, 'cache_dir': str(root)}))
            with (locks / 'port-15070.lock').open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                error = io.StringIO()
                with patch.object(runner, 'preflight'), \
                     patch('roboharness.assets.prepare_robot_asset'), \
                     patch.object(runner, 'Run') as run, contextlib.redirect_stderr(error):
                    code = runner.main(['--task', 'task01', '--config', str(config)])
                self.assertEqual(code, 1)
                self.assertIn('owns port 15070', error.getvalue())
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
