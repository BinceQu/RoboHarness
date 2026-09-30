#!/usr/bin/env python3
"""Recover initial native Skill listing metadata from the 45 selected transcripts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recover(source: Path) -> dict:
    index = json.loads((ROOT / 'harness/claude_code/tests/fixtures/archive_context.json').read_text())
    patterns, cases = {}, []
    for item in index['sources']:
        path = (source / item['path']).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            raise ValueError(f'Archived transcript checksum mismatch: {path}')
        prefix, listing, usage = bytearray(), None, {}
        with path.open('rb') as stream:
            for line in stream:
                prefix.extend(line)
                row = json.loads(line)
                attachment = row.get('attachment', {})
                if attachment.get('type') == 'skill_listing' and attachment.get('isInitial'):
                    listing = attachment
                if row.get('type') == 'assistant':
                    usage = row.get('message', {}).get('usage', {})
                    break
        if listing is None:
            raise ValueError(f'No initial native Skill listing: {path}')
        text = listing['content']
        listing_sha = hashlib.sha256(text.encode()).hexdigest()
        plugin = [line for line in text.splitlines() if line.startswith('- embodied-claude-code:')]
        native = '\n'.join(line for line in text.splitlines() if not line.startswith('- embodied-claude-code:'))
        patterns[listing_sha] = {
            'sha256': listing_sha, 'chars': len(text), 'names': listing['names'],
            'plugin_listing_lines': plugin,
            'native_listing_sha256': hashlib.sha256(native.encode()).hexdigest(),
            'plugin_descriptions': [line.split(': ', 1)[0][2:] for line in plugin if ': ' in line],
        }
        cases.append({'task': item['task'], 'instance_id': item['instance_id'],
                      'listing_sha256': listing_sha, 'source': item['path'],
                      'source_sha256': item['sha256'], 'prefix_bytes': len(prefix),
                      'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                      'first_reported_input_tokens': usage.get('input_tokens')})
    return {
        'schema_version': 1, 'claude_version': '2.1.259',
        'scope': 'Recorded initial native Skill listing metadata. Original skillUsage counters and complete model request bodies were not archived. Bundled CLI description text is represented only by hashes.',
        'patterns': patterns, 'cases': cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--behavior-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'harness/claude_code/profiles/archived-native-context.json')
    args = parser.parse_args()
    result = recover(args.behavior_source.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Recovered {len(result["cases"])} cases with {len(result["patterns"])} listing patterns')


if __name__ == '__main__':
    main()
