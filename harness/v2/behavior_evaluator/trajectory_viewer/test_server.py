"""Read-only viewer checks: live append, exact images, access evidence and confinement."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

from server import NativeLog, Viewer, accesses, conversation


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key = "task01-test/instance_301"
        self.case = self.root / self.key
        self.workspace = self.case / "workspace"
        self.history = self.workspace / "tool_call_history/000001_capture_head_camera"
        self.history.mkdir(parents=True)
        self.native = self.workspace / ".runtime/claude-home/projects/project/session.jsonl"
        self.native.parent.mkdir(parents=True)
        self.native.write_text("")
        self.viewer = Viewer(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def event(self, command):
        return {"type": "assistant", "cwd": str(self.workspace), "message": {"content": [
            {"type": "tool_use", "name": "Bash", "id": "use-1", "input": {"command": command}}]}}

    def test_execution_does_not_count_as_reading(self):
        rows = accesses(self.event("python tools/close_gripper.py '{\"arm\":\"left\"}'"), self.workspace)
        self.assertEqual([(r["kind"], r["path"]) for r in rows], [("execute", "tools/close_gripper.py")])
        rows = accesses(self.event("python tools/close_gripper.py --schema"), self.workspace)
        self.assertEqual(rows[0]["kind"], "schema")

    def test_explicit_read_and_program_image_processing_are_distinct(self):
        rows = accesses(self.event("sed -n '1,80p' robot_config/control.py"), self.workspace)
        self.assertEqual(rows[0]["kind"], "read")
        rows = accesses(self.event("python -c \"from PIL import Image; Image.open('tool_call_history/a.png')\""), self.workspace)
        self.assertEqual(rows[0]["kind"], "process_image")
        rows = accesses(self.event("python -c \"open('out.json', 'w')\""), self.workspace)
        self.assertEqual(rows, [])

    def test_live_partial_line_is_not_lost_or_duplicated(self):
        first = json.dumps(self.event("cat AGENTS.md")) + "\n"
        second = json.dumps(self.event("cat robot_config/joints.md")) + "\n"
        self.native.write_text(first + second[:20])
        log = self.viewer.native(self.key)
        self.assertEqual(len(log.rows), 1)
        with self.native.open("a") as stream:
            stream.write(second[20:])
        self.assertEqual(len(self.viewer.native(self.key).rows), 2)
        self.assertEqual(len(self.viewer.native(self.key).rows), 2)

    def test_shell_separators_and_literal_python_heredoc(self):
        rows = accesses(self.event('grep -n torso robot_config/joints.md | head -30; grep limit tools/util/model.py'), self.workspace)
        self.assertEqual([(r['kind'], r['path']) for r in rows],
                         [('search', 'robot_config/joints.md'), ('search', 'tools/util/model.py')])
        rows = accesses(self.event("python - <<'EOF'\nfrom tools.util.model import validate_action\nEOF"), self.workspace)
        self.assertEqual([(r['kind'], r['path']) for r in rows], [('import', 'tools/util/model.py')])

    def test_receipt_mapping_and_original_image_bytes(self):
        raw = b"original image bytes"
        event = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "use-1", "content": [
            {"type": "text", "text": json.dumps({"record": str(self.history / "result.json")})},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(raw).decode()}}]}]}}
        self.native.write_text(json.dumps(self.event("python tools/capture_head_camera.py")) + "\n" + json.dumps(event) + "\n")
        log = self.viewer.native(self.key)
        self.assertEqual(log.receipts[self.history.name], {"tool_line": 1, "result_line": 2})
        sanitized = log.entry(2)["event"]["message"]["content"][0]["content"][1]["source"]
        self.assertNotIn("data", sanitized)
        self.assertIn("image_url", sanitized)
        actual = log.raw(2)["message"]["content"][0]["content"][1]["source"]["data"]
        self.assertEqual(base64.b64decode(actual), raw)

    def test_file_boundary_and_symlink_escape(self):
        (self.case / "prompt.txt").write_text("task")
        self.assertEqual(self.viewer.file(self.key, "prompt.txt").read_text(), "task")
        for path in ["../../etc/passwd", "workspace/.session.json", "workspace/tool_call_history/../../.runtime/secret"]:
            with self.assertRaises(ValueError):
                self.viewer.file(self.key, path)
        secret = self.root / "secret"
        secret.write_text("private")
        (self.history / "escape.json").symlink_to(secret)
        with self.assertRaises(ValueError):
            self.viewer.file(self.key, "workspace/tool_call_history/000001_capture_head_camera/escape.json")
        with self.assertRaises(ValueError):
            self.viewer.case("../instance_301")

    def test_incomplete_case_has_no_invented_score(self):
        (self.case / "case.json").write_text(json.dumps({"instance_id": 301}))
        (self.case.parent / "summary.json").write_text(json.dumps({"status": "failed", "cases": []}))
        row = self.viewer.catalog()[0]
        self.assertIsNone(row["q"])
        self.assertEqual(row["status"], "incomplete")

    def test_flag_free_receipts_show_errors_and_confirmed_skill_state(self):
        skill = self.workspace / "tool_call_history/000002_activate_skill"
        skill.mkdir()
        (skill / "input.json").write_text(json.dumps({"tool": "activate_skill",
            "arguments": {"name": "pick-up-object"}}))
        (skill / "result.json").write_text(json.dumps({"tool_name": "activate_skill",
            "task_skill_state": {"active_skill": "pick-up-object"}}))
        (self.history / "result.json").write_text(json.dumps({"error": "Camera disconnected"}))
        session = self.viewer.session(self.key)
        self.assertEqual(session["calls"][0]["error"], "Camera disconnected")
        self.assertEqual([item["name"] for item in session["activated_skills"]], ["pick-up-object"])
        (skill / "result.json").write_text(json.dumps({"error": "Could not load skill"}))
        self.assertEqual(self.viewer.session(self.key)["activated_skills"], [])

    def test_resident_test_status_binding_and_ten_instance_count(self):
        resident = self.root / 'resident-task30'
        resident.mkdir()
        binding = resident / 'active_session.json'
        binding.write_text(json.dumps({'workspace': str(self.workspace)}))
        (self.case / 'case.json').write_text(json.dumps({'instance_id': 301}))
        summary = self.case.parent / 'summary.json'
        summary.write_text(json.dumps({'phase': 'running', 'resident': str(resident),
            'instances': list(range(301, 311)), 'cases': [], 'n_finished': 0}))
        row = self.viewer.catalog()[0]
        self.assertEqual(row['status'], 'running')
        self.assertEqual(row['run_status'], 'running')
        self.assertEqual(row['run_expected'], 10)
        self.assertIsNone(row['q'])
        binding.write_text(json.dumps({'workspace': str(self.workspace) + '-other'}))
        self.assertEqual(self.viewer.catalog()[0]['status'], 'incomplete')

    def test_legacy_task01_status_and_instance_count_remain_supported(self):
        (self.case / 'case.json').write_text(json.dumps({'instance_id': 301}))
        (self.case.parent / 'active_session.json').write_text(json.dumps({'workspace': str(self.workspace)}))
        (self.case.parent / 'summary.json').write_text(json.dumps({'status': 'running',
            'n_expected': 5, 'cases': [], 'n_finished': 0}))
        row = self.viewer.catalog()[0]
        self.assertEqual(row['status'], 'running')
        self.assertEqual(row['run_expected'], 5)

    def transcript(self, events):
        self.native.write_text(''.join(json.dumps(event) + '\n' for event in events))
        return conversation(self.viewer.native(self.key))

    @staticmethod
    def assistant(response, *content):
        return {'type': 'assistant', 'message': {'id': response, 'content': list(content)}}

    @staticmethod
    def tool(tool_id, name='Bash', **args):
        return {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': args}

    @staticmethod
    def result(tool_id, content, error=False):
        return {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': tool_id, 'content': content, 'is_error': error}]}}

    def test_every_tool_and_shared_thinking_with_out_of_order_results(self):
        command = "python - <<'PY'\nprint('literal `command` $value')\nPY"
        d = self.transcript([
            self.assistant('reply-a', {'type': 'thinking', 'thinking': 'Inspect the current image.'}),
            self.assistant('reply-a', self.tool('first', command=command)),
            {'type': 'attachment', 'attachment': {'type': 'hook_success', 'stdout': 'INTERNAL AUDIT'}},
            self.assistant('reply-a', self.tool('second', 'Read', file_path='/tmp/image.png')),
            self.result('second', 'second result'),
            self.result('first', 'first result', error=True),
            self.assistant('reply-b', self.tool('third', 'Glob', pattern='**/*.py')),
        ])
        tools = [entry for entry in d['entries'] if entry['kind'] == 'tool']
        self.assertEqual([t['call']['name'] for t in tools], ['Bash', 'Read', 'Glob'])
        self.assertEqual(tools[0]['call']['input']['command'], command)
        self.assertEqual(tools[1]['before'], tools[0]['before'])
        self.assertEqual(tools[1]['shared_before'], tools[0]['id'])
        self.assertEqual([t['results'][0]['block']['content'] for t in tools[:2]], ['first result', 'second result'])
        self.assertEqual(tools[0]['status'], 'error')
        self.assertEqual(tools[2]['before'], [])
        self.assertEqual(tools[2]['status'], 'pending')
        self.assertNotIn('INTERNAL AUDIT', json.dumps(d))

    def test_future_thinking_never_moves_before_an_earlier_tool(self):
        first = {'type': 'thinking', 'thinking': 'Before tool one.'}
        later = {'type': 'text', 'text': 'Additional assistant text before tool two.'}
        d = self.transcript([
            self.assistant('reply', first, self.tool('one', command='pwd')),
            self.result('one', 'workspace'),
            self.assistant('reply', later, self.tool('two', command='ls')),
        ])
        tools = [e for e in d['entries'] if e['kind'] == 'tool']
        self.assertEqual([x['block'] for x in tools[0]['before']], [first])
        self.assertEqual([x['block'] for x in tools[1]['before']], [first, later])
        self.assertIsNone(tools[1]['shared_before'])

    def test_prompt_injected_context_compaction_and_standalone_thinking_survive(self):
        d = self.transcript([
            {'type': 'user', 'message': {'content': 'Exact task prompt'}},
            {'type': 'attachment', 'attachment': {'type': 'hook_additional_context', 'content': ['Actual baseline instruction']}},
            {'type': 'attachment', 'attachment': {'type': 'skill_listing', 'content': 'skill-a: first skill'}},
            self.assistant('only-thinking', {'type': 'thinking', 'thinking': 'Reasoning with no tool yet.'}),
            {'type': 'system', 'subtype': 'compact_boundary', 'content': 'Conversation compacted'},
            {'type': 'user', 'isCompactSummary': True, 'message': {'content': 'Exact saved summary'}},
            {'type': 'attachment', 'attachment': {'type': 'file', 'displayPath': 'tools/README.md', 'content': {'type': 'text', 'file': {'content': 'Saved file contents'}}}},
            self.assistant('after-compact', self.tool('new', command='pwd')),
            self.result('new', 'workspace'),
            {'type': 'attachment', 'attachment': {'type': 'total_tokens_reminder', 'text': 'Exact budget reminder'}},
        ])
        self.assertEqual([e['kind'] for e in d['entries']], ['user', 'context', 'assistant', 'boundary', 'summary', 'context', 'tool'])
        self.assertEqual(d['entries'][1]['chunks'][0]['blocks'][0]['text'], 'Actual baseline instruction')
        self.assertEqual(d['entries'][1]['chunks'][1]['blocks'][0]['text'], 'skill-a: first skill')
        self.assertEqual(d['entries'][4]['chunks'][0]['blocks'][0]['text'], 'Exact saved summary')
        self.assertEqual(d['entries'][5]['chunks'][0]['blocks'][0]['text'], 'Saved file contents')
        self.assertEqual(d['entries'][-1]['before'], [])
        self.assertEqual(d['entries'][-1]['after'][0]['blocks'][0]['text'], 'Exact budget reminder')

    def test_action_filter_distinguishes_execution_from_reading_and_schema(self):
        (self.workspace / 'tools').mkdir()
        (self.workspace / 'tools/capture_head_camera.py').write_text('# entry point')
        events = []
        commands = ['cat tools/capture_head_camera.py', 'python tools/capture_head_camera.py --schema',
                    'python tools/capture_head_camera.py --help', "python tools/capture_head_camera.py '{}'",
                    'python robot_config/control.py --action 0', 'python -c "print(1)"']
        for i, command in enumerate(commands):
            events.append(self.assistant(str(i), self.tool(str(i), command=command)))
        d = self.transcript(events)
        self.assertEqual([e['action'] for e in d['entries']], [False, False, False, True, True, False])
        self.assertEqual(d['action_count'], 2)
        self.assertEqual(d['tool_count'], 6)

    def test_live_return_updates_pending_call_without_truncating_thought_or_output(self):
        thought = 'Long reasoning 中文\n' * 6000
        events = [self.assistant('reply', {'type': 'thinking', 'thinking': thought}, self.tool('one', command='pwd'))]
        before = self.transcript(events)['entries'][0]
        returned = 'Exact stdout \n' * 4000
        with self.native.open('a') as stream:
            stream.write(json.dumps(self.result('one', returned)) + '\n')
        after = conversation(self.viewer.native(self.key))['entries'][0]
        self.assertEqual(after['id'], before['id'])
        self.assertNotEqual(after['revision'], before['revision'])
        self.assertEqual(before['status'], 'pending')
        self.assertEqual(after['status'], 'returned')
        self.assertEqual(after['before'][0]['block']['thinking'], thought)
        self.assertEqual(after['results'][0]['block']['content'], returned)

    def test_images_are_only_the_actual_return_blocks_and_receipts_classify_robot(self):
        image = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': base64.b64encode(b'original jpeg').decode()}}
        d = self.transcript([
            self.assistant('reply', self.tool('one', command='python tools/capture_head_camera.py')),
            self.result('one', [{'type': 'text', 'text': json.dumps({'record': str(self.history / 'result.json')})}, image]),
        ])
        entry = d['entries'][0]
        self.assertEqual(entry['robot_records'], [self.history.name])
        self.assertTrue(entry['action'])
        self.assertEqual([b['type'] for b in entry['results'][0]['block']['content']], ['text', 'image'])
        source = entry['results'][0]['block']['content'][1]['source']
        self.assertIn('/api/native-image?', source['image_url'])
        self.assertIn('line=2', source['image_url'])
        self.assertNotIn('data', source)

    def test_tool_ids_are_scoped_to_native_session_files(self):
        self.native.write_text(json.dumps(self.assistant('same-reply', self.tool('same-id', command='first-session'))) + '\n')
        second = self.native.with_name('session2.jsonl')
        second.write_text(json.dumps(self.assistant('same-reply', self.tool('same-id', command='second-session'))) + '\n' + json.dumps(self.result('same-id', 'second return')) + '\n')
        d = conversation(self.viewer.native(self.key))
        self.assertEqual(d['entries'][0]['status'], 'pending')
        self.assertEqual(d['entries'][0]['before'], [])
        self.assertEqual(d['entries'][1]['results'][0]['block']['content'], 'second return')

    def test_malformed_native_call_is_retained_with_its_validation_error(self):
        arguments = {'__unparsedToolInput': {'raw': '{"command": "unfinished', 'len': 10000}}
        d = self.transcript([
            self.assistant('broken', self.tool('bad', **arguments)),
            self.result('bad', 'InputValidationError: command required', error=True),
        ])
        self.assertEqual(d['tool_count'], 1)
        self.assertEqual(d['entries'][0]['call']['input'], arguments)
        self.assertEqual(d['entries'][0]['status'], 'error')
        self.assertFalse(d['entries'][0]['action'])

    def test_pi_session_calls_results_images_and_compaction(self):
        self.native.unlink()
        raw = b'original pi jpeg'
        command = 'python tools/capture_head_camera.py'
        lines = [
            {'type': 'session', 'version': 3, 'id': 's', 'cwd': str(self.workspace)},
            {'type': 'message', 'id': 'a1', 'timestamp': '2026-10-17T12:31:29Z', 'message': {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'look first'},
                {'type': 'toolCall', 'id': 'call-1', 'name': 'read', 'arguments': {'path': str(self.workspace / 'AGENTS.md')}},
                {'type': 'toolCall', 'id': 'call-2', 'name': 'bash', 'arguments': {'command': command}}]}},
            {'type': 'message', 'id': 'r1', 'message': {'role': 'toolResult', 'toolCallId': 'call-1', 'content': [{'type': 'text', 'text': 'doc'}]}},
            {'type': 'message', 'id': 'r2', 'message': {'role': 'toolResult', 'toolCallId': 'call-2', 'isError': False, 'content': [
                {'type': 'text', 'text': json.dumps({'record': str(self.history / 'result.json')})},
                {'type': 'image', 'data': base64.b64encode(raw).decode(), 'mimeType': 'image/jpeg'}]}},
            {'type': 'compaction', 'id': 'c1', 'summary': '## Goal\nthree cans', 'tokensBefore': 43863},
        ]
        sessions = self.case / 'pi-sessions'
        sessions.mkdir()
        (sessions / 'one.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))
        log = self.viewer.native(self.key)
        self.assertEqual(log.receipts[self.history.name], {'tool_line': 2, 'result_line': 4})
        self.assertEqual([(r['kind'], r['path']) for r in log.reads], [('read', 'AGENTS.md'), ('execute', 'tools/capture_head_camera.py')])
        d = conversation(log)
        tools = [e for e in d['entries'] if e['kind'] == 'tool']
        self.assertEqual([t['call']['name'] for t in tools], ['read', 'bash'])
        self.assertEqual(tools[0]['before'][0]['block']['thinking'], 'look first')
        self.assertEqual(tools[1]['shared_before'], tools[0]['id'])
        self.assertTrue(tools[1]['action'])
        self.assertEqual([t['status'] for t in tools], ['returned', 'returned'])
        source = tools[1]['results'][0]['block']['content'][1]['source']
        self.assertNotIn('data', source)
        self.assertEqual(base64.b64decode(log.raw(4)['message']['content'][0]['content'][1]['source']['data']), raw)
        self.assertEqual(d['entries'][-1]['kind'], 'summary')
        self.assertEqual(self.viewer.catalog()[0]['native_sessions'], ['one'])

    def test_multiple_returns_without_saved_calls_remain_distinct(self):
        d = self.transcript([{'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'missing-a', 'content': 'First return'},
            {'type': 'tool_result', 'tool_use_id': 'missing-b', 'content': 'Second return'},
        ]}}])
        self.assertEqual(len({e['id'] for e in d['entries']}), 2)
        self.assertEqual([e['results'][0]['block']['content'] for e in d['entries']], ['First return', 'Second return'])


if __name__ == "__main__":
    unittest.main()
