#!/usr/bin/env python3
"""Install upstream dependencies from the recorded commits without editing BEHAVIOR."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def source_environment(sources, cache, env):
    env = dict(env)
    slot = int(env.get('GIT_CONFIG_COUNT', '0'))
    for name, source in sources.items():
        revision = source['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError(f'{name}: a complete Git commit is required')
        repo = cache / f'{name}-{revision}.git'
        ref = f"refs/heads/{source['upstream_ref']}"
        run_git('check-ref-format', ref)
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            run_git('init', '--bare', '--quiet', repo)
        if run_git('--git-dir', repo, 'rev-parse', '--is-bare-repository') != 'true':
            raise ValueError(f'Expected a private bare source cache: {repo}')
        present = subprocess.run(
            ['git', '--git-dir', str(repo), 'cat-file', '-e', f'{revision}^{{commit}}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        shallow = run_git('--git-dir', repo, 'rev-parse', '--is-shallow-repository') == 'true'
        if not present or shallow:
            # Older Git versions can recurse while serving a filtered clone
            # from a shallow local repository. Keep the complete Git history;
            # the separate LFS test/training assets are not needed to install.
            unshallow = ['--unshallow'] if shallow else []
            run_git('--git-dir', repo, 'fetch', '--no-tags', *unshallow, source['url'], revision)
        run_git('--git-dir', repo, 'update-ref', ref, revision)
        run_git('--git-dir', repo, 'symbolic-ref', 'HEAD', ref)
        if run_git('--git-dir', repo, 'rev-parse', f'{ref}^{{commit}}') != revision:
            raise ValueError(f'{name}: source cache does not match the recorded commit')
        # Only children of this installer see the mapping. Pip still resolves
        # the upstream URL/ref, but Git reads our immutable local snapshot.
        env[f'GIT_CONFIG_KEY_{slot}'] = f'url.{repo.as_uri()}.insteadOf'
        env[f'GIT_CONFIG_VALUE_{slot}'] = source['url']
        slot += 1
    env['GIT_CONFIG_COUNT'] = str(slot)
    env['GIT_LFS_SKIP_SMUDGE'] = '1'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify installed source commits using this Python')
    parser.add_argument('--cache', type=Path, default=ROOT / '.local' / 'setup-sources')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sources = json.loads((ROOT / 'configs' / 'evaluator-sources.json').read_text())
    if args.check:
        if args.command:
            parser.error('--check does not take a command')
        for name, source in sources.items():
            dist = importlib.metadata.distribution(name)
            direct = json.loads(dist.read_text('direct_url.json') or '{}')
            actual = direct.get('vcs_info', {}).get('commit_id')
            if actual != source['revision']:
                raise SystemExit(f"{name}: expected {source['revision']}, installed source is {actual!r}")
            print(f'{name} {dist.version}: verified commit {actual}')
        return
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide the installer command after --')
    env = source_environment(sources, args.cache.resolve(), os.environ)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
