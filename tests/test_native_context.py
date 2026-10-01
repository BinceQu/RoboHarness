import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from roboharness.native_context import ROOT, PROFILE, contract, create_workspace, inspect_listing, seed


class NativeContextTests(unittest.TestCase):
    def test_workspace_is_private_and_never_inherits_git_metadata(self):
        with tempfile.TemporaryDirectory(dir='/var/tmp') as directory:
            root = Path(directory)
            workspace = create_workspace(root / 'runtime', 301)
            self.assertEqual(workspace.name, 'embodied_claude_code')
            self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                create_workspace(root / 'runtime', 301)
            # A .git file marks a worktree just as a .git directory marks a checkout.
            git_root = root / 'checkout'
            git_root.mkdir()
            (git_root / '.git').write_text('gitdir: /not-needed-for-this-check')
            with self.assertRaisesRegex(ValueError, 'outside Git worktrees'):
                create_workspace(git_root / 'cache', 301)
            from roboharness.runner import preflight
            with self.assertRaisesRegex(ValueError, 'outside Git worktrees'):
                preflight({'cache_dir': str(git_root / 'cache')}, 'claude_code')
            link = root / 'linked-cache'
            link.symlink_to(git_root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'outside Git worktrees'):
                create_workspace(link, 304)
            self.assertFalse((git_root / 'cache').exists())
        with self.assertRaisesRegex(ValueError, 'absolute runtime'):
            create_workspace(Path('relative'), 301)

    def test_matching_skill_listing_does_not_hide_a_changed_native_workspace(self):
        content = '- robot-skill: archived description'
        expected = {'case': {'git_branch': 'HEAD'},
                    'pattern': {'sha256': hashlib.sha256(content.encode()).hexdigest(), 'names': ['robot-skill']}}
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / 'one.jsonl'
            for branch, state in [('HEAD', 'match'), ('main', 'mismatch'), (None, 'mismatch')]:
                row = {'gitBranch': branch, 'attachment': {'type': 'skill_listing',
                       'isInitial': True, 'content': content, 'names': ['robot-skill']}}
                transcript.write_text(json.dumps(row) + '\n')
                audit = inspect_listing(transcript, expected)
                self.assertEqual(audit['state'], state)
                self.assertEqual(audit['actual_git_branch'], branch)

    def test_all_selected_cases_have_one_recorded_listing(self):
        index = json.loads((ROOT / PROFILE).read_text())
        self.assertEqual(len(index['cases']), 45)
        counts = {}
        for case in index['cases']:
            item = contract(case['task'], case['instance_id'])
            self.assertEqual(item['case']['git_branch'], 'HEAD')
            digest = item['pattern']['sha256']
            counts[digest] = counts.get(digest, 0) + 1
        self.assertEqual(sorted(counts.values()), [7, 38])
        with self.assertRaises(ValueError):
            contract('task04', 301)

    def test_seed_is_case_local_and_never_overwrites_existing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'case-home'
            env = seed(home, 'task03', 301)
            self.assertEqual(env, {'SLASH_COMMAND_TOOL_CHAR_BUDGET': '5987'})
            original = (home / '.claude.json').read_bytes()
            self.assertEqual(list(json.loads(original)['skillUsage']),
                             ['embodied-claude-code:navigate-to-target'])
            self.assertEqual((home / '.claude.json').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                seed(home, 'task03', 301)
            self.assertEqual((home / '.claude.json').read_bytes(), original)

    def test_distinct_archived_pattern_and_symlink_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'case-home'
            env = seed(home, 'task08', 304)
            self.assertEqual(env['SLASH_COMMAND_TOOL_CHAR_BUDGET'], '5959')
            self.assertEqual(list(json.loads((home / '.claude.json').read_text())['skillUsage']),
                             ['embodied-claude-code:behavior-v2-baseline',
                              'embodied-claude-code:close-box'])
            link = Path(directory) / 'linked-home'
            link.symlink_to(home, target_is_directory=True)
            with self.assertRaises(ValueError):
                seed(link, 'task08', 304)

    def test_names_alone_do_not_satisfy_listing_fidelity(self):
        text = '- robot-skill: archived description'
        expected = {'pattern': {'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                'names': ['robot-skill']}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transcript.jsonl'
            for content, state in [(text, 'match'), ('- robot-skill', 'mismatch')]:
                path.write_text(json.dumps({'attachment': {'type': 'skill_listing',
                    'isInitial': True, 'content': content, 'names': ['robot-skill']}}) + '\n')
                self.assertEqual(inspect_listing(path, expected)['state'], state)
            path.write_text('{"partial":')
            self.assertEqual(inspect_listing(path, expected)['state'], 'pending')
            path.write_text('{"type":"assistant"}\n')
            self.assertEqual(inspect_listing(path, expected)['state'], 'missing')

    def test_report_requires_actual_case_transcript_and_full_listing(self):
        from scripts.report_validation import native_contexts
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertEqual(native_contexts(run, 'task03', {301: {}}, set())[0]['state'], 'not_started')
            self.assertEqual(native_contexts(run, 'task03', {301: {}}, {301})[0]['state'], 'missing')
            folder = run / 'instance_301/claude-home/projects/project'
            folder.mkdir(parents=True)
            transcript = folder / 'one.jsonl'
            transcript.write_text('{"type":"assistant"}\n')
            self.assertEqual(native_contexts(run, 'task03', {301: {}}, {301})[0]['state'], 'missing')
            text = '- robot-skill: archived description'
            expected = {'pattern': {'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                    'names': ['robot-skill']}}
            with mock.patch('scripts.report_validation.native_contract', return_value=expected):
                for content, state in [(text, 'match'), ('- robot-skill', 'mismatch')]:
                    transcript.write_text(json.dumps({'attachment': {'type': 'skill_listing',
                        'isInitial': True, 'content': content, 'names': ['robot-skill']}}) + '\n')
                    self.assertEqual(native_contexts(run, 'task03', {301: {}}, {301})[0]['state'], state)
            (folder / 'two.jsonl').write_text('{}\n')
            self.assertEqual(native_contexts(run, 'task03', {301: {}}, {301})[0]['state'], 'ambiguous')


@unittest.skipUnless(os.environ.get('ROBOHARNESS_NATIVE_CLAUDE_TESTS') == '1',
                     'requires the pinned real CLI and harness test dependencies')
class NativeContextCLITests(unittest.TestCase):
    def test_both_recorded_listings_with_real_cli_hooks_mcp_and_compaction(self):
        harness_tests = str(ROOT / 'harness/claude_code/tests')
        sys.path.insert(0, harness_tests)
        try:
            from test_native_skill_lifecycle import NativeSkillLifecycleTests
            for task, iid, namespace in [('task03', 301, 'behavior-v2'),
                                         ('task08', 304, 'plugin:embodied-claude-code:behavior-v2')]:
                with self.subTest(task=task, instance_id=iid):
                    expected = contract(task, iid)
                    def setup(runtime, env):
                        env.update(seed(runtime / 'claude', task, iid))
                        return create_workspace(runtime / 'native-runtime', iid)
                    NativeSkillLifecycleTests().run_transport(
                        bridge=False, archived=True, mcp_name=namespace,
                        context_setup=setup,
                        expected_listing_sha256=expected['pattern']['sha256'],
                        expect_non_git_workspace=True,
                        endpoint_port=int(os.environ.get('ROBOHARNESS_NATIVE_TEST_PORT', '0')))
        finally:
            sys.path.remove(harness_tests)


if __name__ == '__main__':
    unittest.main()
