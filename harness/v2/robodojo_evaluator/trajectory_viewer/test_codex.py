"""Codex regression checks against its saved Responses and execution schemas."""
import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import ProxyHandler, build_opener

from server import Viewer, conversation, handler


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / 'astra/ARX_X5'
        (self.workspace / 'tool_call_history/000001_capture_head_camera').mkdir(parents=True)
        (self.workspace / 'tools').mkdir()
        (self.workspace / 'tools/capture_head_camera.py').write_text('# tool')
        self.path = self.workspace / '.runtime/codex-home/sessions/2026/10/15/rollout-test.jsonl'
        self.path.parent.mkdir(parents=True)
        self.viewer = Viewer(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def event(kind, **values):
        return {'type': 'response_item', 'payload': {'type': kind, **values}}

    def write(self, events):
        self.path.write_text(''.join(json.dumps(e) + '\n' for e in events))

    def test_saved_reasoning_is_not_decrypted_or_replaced_by_commentary(self):
        self.write([
            self.event('message', role='developer', content=[{'type': 'input_text', 'text': 'Saved instructions'}]),
            self.event('message', role='assistant', content=[{'type': 'output_text', 'text': 'Public update'}]),
            self.event('reasoning', summary=[], encrypted_content='PRIVATE CIPHERTEXT'),
            self.event('custom_tool_call', call_id='a', name='exec', input='text("one");'),
            self.event('custom_tool_call_output', call_id='a', output=[{'type': 'input_text', 'text': 'one'}]),
            self.event('custom_tool_call', call_id='b', name='exec', input='text("two");'),
        ])
        log = self.viewer.native('astra')
        data = conversation(log)
        calls = [e for e in data['entries'] if e['kind'] == 'tool']
        self.assertEqual([x['block']['type'] for x in calls[0]['before']], ['text', 'redacted_thinking'])
        self.assertEqual(calls[0]['before'][0]['block']['text'], 'Public update')
        self.assertEqual(calls[1]['before'], [])
        self.assertNotIn('PRIVATE CIPHERTEXT', json.dumps(log.entry(3)))
        self.assertIn('PRIVATE CIPHERTEXT', json.dumps(log.raw(3)))
        cap = self.viewer.session('astra')['native_capabilities']
        self.assertEqual(cap['reasoning_items'], 1)
        self.assertEqual(cap['readable_reasoning_items'], 0)
        self.assertEqual(cap['encrypted_reasoning_items'], 1)
        self.assertEqual(len(self.viewer.catalog()[0]['native_sessions']), 1)

    def test_exact_commands_returns_and_actual_image_bytes(self):
        raw = b'exact saved image, not latest.png'
        command = '.venv/bin/python tools/capture_head_camera.py \'{}\''
        code = 'text(await tools.exec_command({cmd:' + json.dumps(command) + '}));'
        output = json.dumps({'chunk_id': 'id', 'exit_code': 0, 'output': '{"history":' +
                            json.dumps(str(self.workspace / 'tool_call_history/000001_capture_head_camera')) + '}'})
        self.write([
            self.event('custom_tool_call', call_id='a', name='exec', input=code),
            {'type': 'event_msg', 'payload': {'type': 'item_completed', 'item': {
                'type': 'CommandExecution', 'command': ['/bin/bash', '-lc', command],
                'cwd': self.workspace.as_uri(), 'aggregated_output': 'UNEXPOSED FULL STDOUT'}}},
            self.event('custom_tool_call_output', call_id='a', output=[
                {'type': 'input_text', 'text': output},
                {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}]),
        ])
        log = self.viewer.native('astra')
        data = conversation(log)
        call = data['entries'][0]
        self.assertEqual(call['call']['input']['code'], code)
        self.assertEqual(call['call']['input']['commands'], [command])
        self.assertTrue(call['action'])
        self.assertEqual(call['robot_records'], ['000001_capture_head_camera'])
        self.assertNotIn('UNEXPOSED FULL STDOUT', json.dumps(data))
        self.assertEqual(log.reads[0]['kind'], 'execute')
        self.assertEqual(log.receipts['000001_capture_head_camera'], {'tool_line': 1, 'result_line': 3})
        source = call['results'][0]['block']['content'][1]['source']
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.viewer))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{http.server_port}' + source['image_url']) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/png')
                self.assertEqual(response.read(), raw)
        finally:
            http.shutdown()
            http.server_close()
            thread.join()

    def test_live_append_pairs_by_call_id_and_does_not_duplicate_nested_events(self):
        self.write([
            self.event('reasoning', summary=[{'type': 'summary_text', 'text': 'Saved summary'}]),
            self.event('function_call', call_id='a', name='sleep', arguments='{"duration_ms":1}'),
            self.event('function_call', call_id='b', name='sleep', arguments='{"duration_ms":2}'),
        ])
        first = conversation(self.viewer.native('astra'))
        with self.path.open('a') as stream:
            stream.write(json.dumps(self.event('function_call_output', call_id='b', output='second')) + '\n')
            stream.write(json.dumps(self.event('function_call_output', call_id='a', output='first')) + '\n')
        second = conversation(self.viewer.native('astra'))
        self.assertEqual(second['tool_count'], 2)
        self.assertEqual(first['entries'][0]['id'], second['entries'][0]['id'])
        self.assertEqual(second['entries'][0]['results'][0]['block']['content'][0]['text'], 'first')
        self.assertEqual(second['entries'][1]['results'][0]['block']['content'][0]['text'], 'second')
        self.assertEqual(second['entries'][1]['shared_before'], second['entries'][0]['id'])
        self.assertEqual(second['entries'][0]['before'][0]['block']['label'], 'Reasoning summary')

    def test_schema_read_is_not_robot_action_and_error_is_preserved(self):
        self.write([
            self.event('custom_tool_call', call_id='a', name='exec', input=
                'text(await tools.exec_command({cmd:"python tools/capture_head_camera.py --schema"}));'),
            self.event('custom_tool_call_output', call_id='a', output=[{'type': 'input_text', 'text':
                json.dumps({'chunk_id': 'id', 'exit_code': 1, 'output': 'exact failure'})}]),
        ])
        call = conversation(self.viewer.native('astra'))['entries'][0]
        self.assertFalse(call['action'])
        self.assertEqual(call['status'], 'error')
        self.assertEqual(call['results'][0]['block']['content'][0]['text'], 'exact failure')

    def test_image_access_is_evidence_not_an_extra_model_image(self):
        self.write([{'type': 'event_msg', 'payload': {'type': 'item_completed', 'item': {
            'type': 'ImageView', 'path': (self.workspace / 'perception/current.png').as_uri()}}}])
        log = self.viewer.native('astra')
        self.assertEqual(log.reads[0]['kind'], 'view_image')
        self.assertEqual(log.reads[0]['path'], 'perception/current.png')
        self.assertEqual(conversation(log)['entries'], [])
        self.assertEqual(self.viewer.session('astra')['native_capabilities']['format'], 'codex')


if __name__ == '__main__':
    unittest.main()
