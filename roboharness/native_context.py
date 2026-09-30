"""Recover the recorded native Skill listing in a fresh, case-local CLI home.

Claude 2.1.259 ranks descriptions using persisted skillUsage. The archive did
not save the original counters. We derive the minimum priority seed from the
recorded listing, then validate the emitted listing separately. This does not
claim to recover the unrecorded complete model request.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'harness/claude_code/profiles/archived-native-context.json'


def contract(task: str, instance_id: int, *, root: Path = ROOT) -> dict:
    index = json.loads((root / PROFILE).read_text())
    if index.get('schema_version') != 1 or index.get('claude_version') != '2.1.259':
        raise ValueError('Unsupported archived native context profile')
    matches = [case for case in index['cases']
               if case['task'] == task and case['instance_id'] == instance_id]
    if len(matches) != 1:
        raise ValueError(f'No unique archived native context for {task}/{instance_id}')
    case = matches[0]
    pattern = index['patterns'][case['listing_sha256']]
    if pattern['sha256'] != case['listing_sha256']:
        raise ValueError('Native context profile checksum key mismatch')
    return {'case': case, 'pattern': pattern, 'claude_version': index['claude_version']}


def seed(config_dir: Path, task: str, instance_id: int, *, root: Path = ROOT) -> dict:
    """Create a new isolated config; never merge or overwrite existing history."""
    expected = contract(task, instance_id, root=root)
    if not config_dir.is_absolute() or config_dir.is_symlink():
        raise ValueError('Native context requires an absolute, non-symlink case directory')
    config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pattern = expected['pattern']
    now_ms = time.time_ns() // 1_000_000
    usage = {name: {'usageCount': 1, 'lastUsedAt': now_ms}
             for name in pattern['plugin_descriptions']}
    encoded = (json.dumps({'skillUsage': usage}, indent=2) + '\n').encode()
    # O_EXCL also rejects symlinks and protects global or resumed CLI homes.
    target = config_dir / '.claude.json'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded)
    record = {
        'schema_version': 1, 'task': task, 'instance_id': instance_id,
        'expected_listing_sha256': pattern['sha256'],
        'character_budget': pattern['chars'],
        'prioritized_plugin_descriptions': pattern['plugin_descriptions'],
        'initial_config_sha256': hashlib.sha256(encoded).hexdigest(),
        'seeded_at_unix_ms': now_ms,
        'source': expected['case']['source'],
        'source_sha256': expected['case']['source_sha256'],
        'scope': 'Derived priority seed, not a historical skillUsage snapshot. '
                 'The expected emitted listing is the archived acceptance criterion.',
    }
    (config_dir / 'native-context.seed.json').write_text(json.dumps(record, indent=2) + '\n')
    return {'SLASH_COMMAND_TOOL_CHAR_BUDGET': str(pattern['chars'])}


def inspect_listing(transcript: Path, expected: dict) -> dict:
    """Compare the full recorded listing, including descriptions and builtins."""
    with transcript.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live final line may still be incomplete.
            attachment = row.get('attachment', {})
            if attachment.get('type') == 'skill_listing' and attachment.get('isInitial'):
                content = attachment.get('content', '')
                digest = hashlib.sha256(content.encode()).hexdigest()
                matches = (digest == expected['pattern']['sha256']
                           and attachment.get('names') == expected['pattern']['names'])
                return {'state': 'match' if matches else 'mismatch',
                        'expected_sha256': expected['pattern']['sha256'],
                        'actual_sha256': digest, 'actual_chars': len(content)}
            if row.get('type') == 'assistant':
                return {'state': 'missing', 'reason': 'No initial Skill listing before first assistant'}
    return {'state': 'pending'}
