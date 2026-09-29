import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'interface'), str(ROOT / 'harness/claude_code/src')]
from flask import Flask
from official_eval_harness.session_guard import install_session_guard
from embodied_claude_code.config import validate_owned_origin, ConfigurationError
from roboharness.runner import process_start


class SessionOwnership(unittest.TestCase):
    def test_only_active_session_can_mutate_the_assigned_port(self):
        env = {'ROBOHARNESS_HTTP_PORT': '16061', 'ROBOHARNESS_TASK_ID': '1'}
        sid = 't01p16061i0-test'
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env):
            guard = Path(directory) / 'guard.json'
            guard.write_text(json.dumps({'port': 16061, 'session_id': sid,
                'owner_pid': os.getpid(), 'owner_start': process_start(os.getpid())}))
            app = Flask(__name__)
            effects = []
            @app.post('/api/v2/action')
            def action():
                effects.append('action')
                return {'ok': True}
            install_session_guard(app, 16061, str(guard))
            client = app.test_client()
            for rejected in ['t01p16061i0-stale', 't08p16068i0-test', '']:
                self.assertEqual(client.post('/api/v2/action', json={'session_id': rejected}).status_code, 409)
            self.assertEqual(effects, [])
            self.assertEqual(client.post('/api/v2/action', json={'session_id': sid}).status_code, 200)
            self.assertEqual(effects, ['action'])
            guard.write_text('{}')
            self.assertEqual(client.post('/api/v2/action', json={'session_id': sid}).status_code, 409)

    def test_custom_port_does_not_allow_cross_origin_or_cross_task(self):
        with patch.dict(os.environ, {'ROBOHARNESS_HTTP_PORT': '16061', 'ROBOHARNESS_TASK_ID': '1',
                                     'BEHAVIOR_EVAL_OWNER_PORT': '16061'}):
            validate_owned_origin('http://127.0.0.1:16061', 't01p16061i0-test')
            for url, sid in [('http://127.0.0.1:16068', 't01p16061i0-test'),
                             ('http://example.org:16061', 't01p16061i0-test'),
                             ('http://127.0.0.1:16061', 't08p16061i0-test')]:
                with self.assertRaises(ConfigurationError):
                    validate_owned_origin(url, sid)
