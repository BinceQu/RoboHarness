import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.with_evaluator_sources import source_environment


@unittest.skipUnless(shutil.which('git'), 'Git is required for the installation check')
class EvaluatorSourceTests(unittest.TestCase):
    def test_moving_branch_and_offline_cache_keep_recorded_source(self):
        with tempfile.TemporaryDirectory(prefix='roboharness sources ') as tmp:
            root = Path(tmp)
            upstream = root / 'upstream'
            upstream.mkdir()

            def git(*args, env=None):
                return subprocess.check_output(
                    ['git', *map(str, args)], env=env, text=True,
                    stderr=subprocess.PIPE,
                ).strip()

            git('init', '--quiet', upstream)
            git('-C', upstream, 'checkout', '--quiet', '-b', 'release/b1k')
            for content in ('recorded', 'later update'):
                (upstream / 'value.txt').write_text(content)
                git('-C', upstream, 'add', 'value.txt')
                git('-C', upstream, '-c', 'user.name=Source check',
                    '-c', 'user.email=source-check@example.invalid',
                    'commit', '--quiet', '-m', content)
                if content == 'recorded':
                    recorded = git('-C', upstream, 'rev-parse', 'HEAD')
            self.assertNotEqual(git('-C', upstream, 'rev-parse', 'HEAD'), recorded)

            sources = {'fixture': {'url': upstream.as_uri(), 'revision': recorded,
                                   'upstream_ref': 'release/b1k'}}
            parent_env = dict(os.environ, GIT_CONFIG_COUNT='1',
                              GIT_CONFIG_KEY_0='core.abbrev', GIT_CONFIG_VALUE_0='12')
            unchanged = dict(parent_env)
            cache = root / 'cache'
            child_env = source_environment(sources, cache, parent_env)
            self.assertEqual(parent_env, unchanged)
            self.assertEqual(git('config', '--get', 'core.abbrev', env=child_env), '12')

            # Reusing a prepared pin must not consult a now-unavailable upstream.
            upstream.rename(root / 'upstream unavailable')
            child_env = source_environment(sources, cache, parent_env)
            installed = root / 'installed'
            git('clone', '--quiet', '--filter=blob:none', sources['fixture']['url'],
                installed, env=child_env)
            self.assertEqual(git('-C', installed, 'rev-parse', 'HEAD'), recorded)
            self.assertEqual((installed / 'value.txt').read_text(), 'recorded')


if __name__ == '__main__':
    unittest.main()
