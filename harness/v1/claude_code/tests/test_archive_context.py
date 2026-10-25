import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code.config import Settings
from embodied_claude_code.server import create_mcp_server
from embodied_claude_code.service import EmbodiedService
from embodied_claude_code.skills import discover_task_skills
from fake_behavior import FakeBehaviorClient
from test_hooks import SESSION_START


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / 'tests/fixtures/archive_context.json').read_text())


class ArchiveContext(unittest.TestCase):
    def test_startup_context_matches_archived_transcripts_byte_for_byte(self):
        with patch.dict(os.environ, {'ROBOHARNESS_PROTOCOL': REFERENCE['protocol']}):
            context = SESSION_START.build_context(ROOT, revision=REFERENCE['startup_revision'])
            self.assertEqual(hashlib.sha256(context.encode()).hexdigest(),
                             REFERENCE['startup_context_sha256'])
            self.assertEqual([skill.name for skill in discover_task_skills(ROOT)],
                             REFERENCE['available_task_skills'])


class ArchiveMCP(unittest.IsolatedAsyncioTestCase):
    async def test_archived_tools_keep_images_skills_and_omit_later_budget_telemetry(self):
        from mcp.client import Client

        class ArchiveClient(FakeBehaviorClient):
            budget_reads = 0

            def get_rollout_budget(self, **kwargs):
                self.budget_reads += 1
                raise AssertionError('The archived protocol has no budget telemetry endpoint')

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'ROBOHARNESS_PROTOCOL': REFERENCE['protocol'],
            'XDG_RUNTIME_DIR': tmp, 'BEHAVIOR_MINIMAP_HUD': '0',
        }), patch('embodied_claude_code.skills.publish_loaded_skills', return_value=True), \
                patch('embodied_claude_code.service.publish_from_tool_result', return_value=None):
            fake = ArchiveClient()
            fake.memory['rollout_budget'] = {'available': True, 'used_ticks': 100}
            service = EmbodiedService(Settings(base_url='http://127.0.0.1:5011',
                record=False, record_root=Path(tmp), session_id='archive-context-test'), client=fake)
            server = create_mcp_server(service)
            self.assertNotIn('rollout_budget', server.instructions)
            prefix = server.instructions[:REFERENCE['mcp_instructions_prefix_length']]
            self.assertEqual(hashlib.sha256(prefix.encode()).hexdigest(),
                             REFERENCE['mcp_instructions_prefix_sha256'])
            async with Client(server, mode='legacy') as client:
                tools = (await client.list_tools()).tools
                names = {tool.name for tool in tools}
                self.assertTrue({"plan_press_point", "adjust_plan_pose", "cut_object"}
                                .issubset(names))
                self.assertTrue({"read_depth", "move_point_to_point",
                                 "plan_grasp_point_filter"}.isdisjoint(names))
                for tool in tools:
                    self.assertNotIn('rollout_budget', tool.description)
                operations = [('capture_head_camera', {}), ('measure_shoulder_distance', {}),
                              ('activate_skill', {}), ('activate_skill', {'name': 'pick-up-object'}),
                              ('deactivate_skill', {'name': 'pick-up-object'}),
                              ('activate_skill', {'name': 'place-object-in-container'}),
                              ('deactivate_skill', {'name': 'place-object-in-container'}),
                              ('activate_skill', {'name': 'open-doors-and-drawers'}),
                              ('deactivate_skill', {'name': 'open-doors-and-drawers'}),
                              ('activate_skill', {'name': 'traverse-narrow-passages'}),
                              ('deactivate_skill', {'name': 'traverse-narrow-passages'}),
                              ('activate_skill', {'name': 'missing-skill'})]
                for name, args in operations:
                    with self.subTest(name=name, args=args):
                        result = await client.call_tool(name, args)
                        for block in result.content:
                            if getattr(block, 'type', None) == 'text':
                                self.assertNotIn('rollout_budget', block.text)
                        payload = result.structured_content
                        if payload is not None:
                            self.assertNotIn('rollout_budget', payload)
                        if name == 'capture_head_camera':
                            self.assertTrue(any(getattr(b, 'type', None) == 'image' for b in result.content))
                        elif name == 'activate_skill' and not args:
                            self.assertEqual([skill['name'] for skill in payload['skills']],
                                             REFERENCE['available_task_skills'])
                        elif name == 'activate_skill' and args.get('name') in REFERENCE['activated_skill_body_sources']:
                            self.assertIn(hashlib.sha256(payload['body'].encode()).hexdigest(),
                                          REFERENCE['activated_skill_body_sources'][args['name']])
            self.assertEqual(fake.budget_reads, 0)


if __name__ == '__main__':
    unittest.main()
