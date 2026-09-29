from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code.client import RestClient
from embodied_claude_code.config import Settings
from embodied_claude_code.errors import ConfigurationError, TransportError
from embodied_claude_code.rollout_budget import normalize_budget
from embodied_claude_code.server import create_mcp_server
from embodied_claude_code.service import EmbodiedService
from fake_behavior import FakeBehaviorClient


def snapshot(used=40, total=100, episode='episode-1'):
    return {'available': True, 'used_ticks': used, 'total_ticks': total,
            'episode_id': episode, 'instance_id': 301, 'observation_sequence': 5,
            'source': 'official_evaluator_episode_steps', 'observation_age_s': 0.125,
            'goal_truth': 'must-not-leak', 'q_score': 0.8, 'total_objects': 7}


class BudgetClient(FakeBehaviorClient):
    def __init__(self):
        super().__init__()
        self.budget = snapshot()
        self.memory['rollout_budget'] = self.budget
        self.budget_reads = []

    def get_rollout_budget(self, *, timeout_s):
        self.budget_reads.append(timeout_s)
        return self.budget


class BudgetSanitizerTests(unittest.TestCase):
    def test_exact_ratio_zero_and_overrun(self):
        for used, ratio, remaining in [(0, 0, 100), (40, .4, 60), (110, 1.1, 0)]:
            got = normalize_budget(snapshot(used))
            self.assertTrue(got['available'])
            self.assertEqual((got['used_fraction'], got['remaining_ticks']), (ratio, remaining))
            self.assertNotIn('goal_truth', got)
            self.assertNotIn('q_score', got)

    def test_missing_or_invalid_is_unknown_not_zero(self):
        cases = [None, {}, {'available': False}, snapshot(-1), snapshot(True),
                 snapshot('4'), snapshot(1.5), snapshot(total=0), snapshot(total=float('nan')),
                 snapshot(episode='bad\nidentity'), {**snapshot(), 'source': []}]
        for value in cases:
            with self.subTest(value=value):
                got = normalize_budget(value)
                self.assertFalse(got['available'])
                self.assertIsNone(got['used_ticks'])
                self.assertIsNone(got['used_fraction'])

    def test_budget_endpoint_remains_same_origin_and_read_only(self):
        RestClient._validate_api_path('/api/rollout_budget', 'GET')
        for path, method in [('/api/rollout_budget', 'POST'),
                             ('http://other:15011/api/rollout_budget', 'GET')]:
            with self.assertRaises(ConfigurationError):
                RestClient._validate_api_path(path, method)


class BudgetMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_reply_including_image_failure_skills_and_sdk_errors(self):
        from mcp.client import Client
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'XDG_RUNTIME_DIR': tmp, 'BEHAVIOR_MINIMAP_HUD': '0'
        }), patch('embodied_claude_code.skills.publish_loaded_skills', return_value=True), \
                patch('embodied_claude_code.service.publish_from_tool_result', return_value=None):
            fake = BudgetClient()
            service = EmbodiedService(Settings(base_url='http://127.0.0.1:5011',
                record=False, record_root=Path(tmp), session_id='budget-test'), client=fake)
            async with Client(create_mcp_server(service), mode='legacy') as client:
                operations = [('capture_head_camera', {}), ('adjust_chassis', {'forward': .1}),
                              ('measure_shoulder_distance', {}), ('activate_skill', {}),
                              ('activate_skill', {'name': 'missing-skill'}),
                              ('deactivate_skill', {'name': 'missing-skill'}),
                              ('deactivate_skill', {}), ('tool-does-not-exist', {})]
                for name, args in operations:
                    with self.subTest(name=name, args=args):
                        result = await asyncio.wait_for(client.call_tool(name, args), 5)
                        blocks = [x.text for x in result.content
                                  if getattr(x, 'type', None) == 'text' and x.text.startswith('rollout_budget=')]
                        self.assertEqual(len(blocks), 1)
                        budget = json.loads(blocks[0].split('=', 1)[1])
                        self.assertEqual(budget['used_fraction'], .4)
                        self.assertNotIn('must-not-leak', blocks[0])
                        if result.structured_content is not None:
                            self.assertEqual(result.structured_content['rollout_budget']['used_ticks'], 40)
                        if name == 'capture_head_camera':
                            self.assertTrue(any(getattr(x, 'type', None) == 'image' for x in result.content))
                self.assertTrue(all(0 < x <= 1 for x in fake.budget_reads))

    async def test_failed_action_reads_post_action_usage_and_records_it(self):
        class FailedClient(BudgetClient):
            def post_json(self, path, payload):
                if path == '/api/v2/adjust_chassis':
                    self.budget['used_ticks'] = 60
                    return {'ok': False, 'error': 'blocked'}
                return super().post_json(path, payload)
        with tempfile.TemporaryDirectory() as tmp:
            fake = FailedClient()
            service = EmbodiedService(Settings(base_url='http://127.0.0.1:5011',
                record_root=Path(tmp)), client=fake)
            service.prepare_episode(session_id='record-budget', record=True)
            result = service.call(tool_name='adjust_chassis', arguments={'forward': .1})
            self.assertTrue(result.is_error)
            self.assertEqual(result.data['rollout_budget']['used_ticks'], 60)
            recorded = ''.join(p.read_text() for p in Path(tmp).rglob('turns.jsonl'))
            self.assertIn('rollout_budget', recorded)
            self.assertEqual(fake.budget_reads, [])  # reused the existing memory read
            with patch.object(fake, 'get_memory', side_effect=TransportError('offline')):
                unavailable = service.call(tool_name='adjust_chassis', arguments={'forward': .1})
            self.assertFalse(unavailable.data['rollout_budget']['available'])
            self.assertIsNone(unavailable.data['rollout_budget']['used_ticks'])
            fake.memory['rollout_budget'] = snapshot(0, episode='episode-2')
            with patch.object(fake, 'post_json', return_value={'ok': True}):
                reset = service.call(tool_name='adjust_chassis', arguments={'forward': .1})
            self.assertEqual(reset.data['rollout_budget']['episode_id'], 'episode-2')
            self.assertEqual(reset.data['rollout_budget']['used_ticks'], 0)
            service.finish_episode()


if __name__ == '__main__':
    unittest.main()
