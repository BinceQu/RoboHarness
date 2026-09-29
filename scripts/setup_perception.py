#!/usr/bin/env python3
"""Fetch or verify the external SAM 2 source and checkpoint used by this release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify existing files without downloading')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'configs/perception.json').read_text())
    data = ROOT / 'data'
    source = data / 'sam2'
    if not source.exists() and not args.check:
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.sam2-install-', dir=data) as temporary:
            checkout = Path(temporary) / 'source'
            subprocess.run(['git', 'init', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', spec['repository']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth', '1', 'origin', spec['commit']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
            checkout.rename(source)
    for relative, expected in spec['source_sha256'].items():
        path = source / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f'SAM 2 source missing or different: {path}; expected commit {spec["commit"]}.')
    checkpoint = spec['checkpoint']
    target = data / checkpoint['filename']
    if not target.exists() and not args.check:
        partial = target.with_suffix('.part')
        with urllib.request.urlopen(checkpoint['url'], timeout=60) as src, partial.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if digest(partial) != checkpoint['sha256']:
            partial.unlink()
            raise SystemExit('Downloaded SAM 2 checkpoint checksum mismatch.')
        partial.replace(target)
    if not target.is_file() or digest(target) != checkpoint['sha256']:
        raise SystemExit(f'SAM 2 checkpoint missing or different: {target}')
    print(f'Verified {len(spec["source_sha256"])} SAM 2 source files and {target.name}.')


if __name__ == '__main__':
    main()
