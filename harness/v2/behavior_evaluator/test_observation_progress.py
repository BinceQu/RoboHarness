"""Receipts can explain a read-only loop without inventing actions or task scores."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'R1_pro'))
from tools.util import observation_hook


class ObservationProgressTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.workspace = Path(self.folder.name)
        self.budget = {'available': True, 'episode_id': 'same-scene', 'instance_id': 302,
            'used_ticks': 11326, 'total_ticks': 11589, 'remaining_ticks': 263}

    def receipt(self, number, **changes):
        result = {'tool_name': 'capture_head_camera', 'joint_commands_sent': 0,
            'rollout_budget': copy.deepcopy(self.budget), 'images': [{'path': 'head.png'}],
            'perception': 'perception/latest.json', 'latest_clickable_image_id': 'img_1'}
        result.update(changes)
        path = self.workspace / f'tool_call_history/{number:06d}_capture_head_camera/result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        return path

    def test_observation_loop_explained_from_exact_unchanged_budget(self):
        paths = [self.receipt(number) for number in (1, 2, 3)]
        before = [path.read_bytes() for path in paths]
        notice = observation_hook.repeated_capture_notice(self.workspace, 3)
        self.assertIn('11326/11589', notice)
        self.assertIn('No keep-alive needed', notice)
        self.assertIn('no motion', notice)
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_no_notice_for_motion_unknown_budget_new_episode_or_exhaustion(self):
        for changes in (
            {'joint_commands_sent': 1}, {'tool_name': 'measure_shoulder_distance'},
            {'error': 'capture failed'}, {'images': []}, {'rollout_budget': {'available': False}},
            {'rollout_budget': {**self.budget, 'episode_id': 'other-scene'}},
            {'rollout_budget': {**self.budget, 'remaining_ticks': 0}},
            {'rollout_budget': {**self.budget, 'used_ticks': 11327, 'remaining_ticks': 262}},
        ):
            with self.subTest(changes=changes):
                for number in (1, 2):
                    self.receipt(number)
                self.receipt(3, **changes)
                self.assertIsNone(observation_hook.repeated_capture_notice(self.workspace, 3))
        self.assertIsNone(observation_hook.repeated_capture_notice(self.workspace, 2))

    def test_notice_keeps_original_result_and_image_bytes(self):
        for number in (1, 2, 3):
            self.receipt(number, images=[{'path': 'fixture', 'sha256': 'different', 'caption': 'original'}])
        pending = self.workspace / 'pending.json'
        pending.write_text(json.dumps({'counter': 2}))
        picture = {'type': 'image', 'source': {
            'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}}
        event = {'tool_name': 'Bash', 'hook_event_name': 'PostToolUse',
            'tool_response': {'stdout': 'unchanged command output', 'stderr': '', 'interrupted': False}}
        with patch.object(observation_hook, 'binding', return_value=(
                self.workspace, pending, {'latest_image_id': 'img_3'}, 3)), \
                patch.object(observation_hook, 'image_content', return_value=picture):
            result = observation_hook.process(event)
        response = result['hookSpecificOutput']['updatedToolOutput']
        self.assertEqual(response['stdout'], event['tool_response']['stdout'])
        self.assertIn(picture, response['structuredContent'])
        self.assertIn('observation-only captures', response['structuredContent'][-1]['text'])
        # Old receipt damage must not discard the current camera's pixels.
        self.receipt(4, images=[{'path': 'fixture', 'sha256': 'different', 'caption': 'original'}])
        with patch.object(observation_hook, 'binding', return_value=(
                self.workspace, pending, {'latest_image_id': 'img_4'}, 4)), \
                patch.object(observation_hook, 'image_content', return_value=picture), \
                patch.object(observation_hook, 'repeated_capture_notice', side_effect=ValueError('bad history')):
            fallback = observation_hook.process(event)
        self.assertIn(picture, fallback['hookSpecificOutput']['updatedToolOutput']['structuredContent'])

    def test_interrupted_capture_batch_also_explains_read_only_timing(self):
        path = self.receipt(1, images=[{'path': '/immutable/head.png'}])
        pending = self.workspace / 'pending.json'
        pending.write_text(json.dumps({'counter': 0}))
        event = {'tool_name': 'Bash', 'hook_event_name': 'PostToolUseFailure'}
        with patch.object(observation_hook, 'binding', return_value=(self.workspace, pending, {}, 2)):
            result = observation_hook.process(event)
        notice = result['hookSpecificOutput']['additionalContext']
        self.assertIn(str(path), notice)
        self.assertIn('No keep-alive needed', notice)
        self.assertIn('/immutable/head.png', notice)


if __name__ == '__main__':
    unittest.main()
