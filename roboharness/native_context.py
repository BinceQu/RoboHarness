"""Recover and verify native Skill context and the recorded MCP tool profile.

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
    if case.get('git_branch') != 'HEAD':
        raise ValueError('Missing or unsupported archived native workspace metadata')
    pattern = index['patterns'][case['listing_sha256']]
    if pattern['sha256'] != case['listing_sha256']:
        raise ValueError('Native context profile checksum key mismatch')
    tool_profile = index.get('mcp_tool_profile')
    if not isinstance(tool_profile, dict) or not tool_profile.get('required_tools'):
        raise ValueError('Missing archived MCP tool profile')
    return {'case': case, 'pattern': pattern, 'claude_version': index['claude_version'],
            'tool_profile': tool_profile}


def validate_workspace_root(path: Path) -> Path:
    """Reject Git ancestors before starting a simulator or creating a workspace."""
    if not path.is_absolute():
        raise ValueError('Native workspace requires an absolute runtime directory')
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if os.path.lexists(parent / '.git'):
            raise ValueError(
                f'Native workspace would inherit Git metadata from {parent}; '
                'configure cache_dir outside Git worktrees for archived reproduction.'
            )
    return resolved


def create_workspace(runtime_dir: Path, instance_id: int) -> Path:
    """Keep release Git state out of the model's native startup context.

    The pinned CLI discovers Git metadata independently of Git's ceiling env
    variable. Use a real non-Git directory, not an environment-only override.
    Plugin, prompt and recorder paths remain absolute and case-local.
    """
    workspace = validate_workspace_root(
        runtime_dir / 'agent-workspaces' / f'instance_{instance_id}' / 'embodied_claude_code')
    workspace.mkdir(parents=True, exist_ok=False, mode=0o700)
    return workspace


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
                result = {'state': 'match' if matches else 'mismatch',
                          'expected_sha256': expected['pattern']['sha256'],
                          'actual_sha256': digest, 'actual_chars': len(content)}
                expected_branch = expected.get('case', {}).get('git_branch')
                if expected_branch is not None:
                    result.update(expected_git_branch=expected_branch,
                                  actual_git_branch=row.get('gitBranch'))
                    if row.get('gitBranch') != expected_branch:
                        result.update(state='mismatch', reason='Native workspace Git metadata differs from archive')
                return result
            if row.get('type') == 'assistant':
                return {'state': 'missing', 'reason': 'No initial Skill listing before first assistant'}
    return {'state': 'pending'}


def inspect_tool_profile(case_dir: Path, expected: dict, *, finished: bool = False) -> dict:
    """Verify the profile/catalog recorded by the actual MCP service.

    Native skill-listing equality cannot reveal filtered-out direct tools. The
    recorder writes this manifest when MCP loads its live interface catalog.
    """
    manifests = list((case_dir / 'trajectory').glob('*/manifest.json'))
    if not manifests:
        return {'state': 'missing' if finished else 'pending',
                'reason': 'No MCP tool manifest for the case'}
    if len(manifests) != 1:
        return {'state': 'ambiguous', 'reason': 'Multiple MCP tool manifests for one case'}
    try:
        manifest = json.loads(manifests[0].read_text())
        profile = manifest['profile']
        catalog = manifest['tool_catalog']['tools']
        names = [item['name'] for item in catalog]
        if not isinstance(profile, dict) or any(not isinstance(name, str) for name in names):
            raise ValueError('Invalid MCP profile or tool names')
        differences = []
        for key in ('allow_tools', 'deny_tools'):
            if sorted(profile[key]) != sorted(expected[key]):
                differences.append(key)
        if profile.get('fixed_arguments') != expected['fixed_arguments']:
            differences.append('fixed_arguments')
        missing = sorted(set(expected['required_tools']) - set(names))
        forbidden = sorted(set(expected['deny_tools']) & set(names))
        if missing or forbidden or len(names) != len(set(names)):
            differences.append('tool_catalog')
        case_path = case_dir / 'case.json'
        if case_path.exists():
            case = json.loads(case_path.read_text())
            if not isinstance(case, dict):
                raise ValueError('Invalid case session metadata')
            session_id = case.get('session_id')
            if not session_id or manifest.get('session_id') != session_id:
                differences.append('session_id')
        return {'state': 'mismatch' if differences else 'match',
                'manifest': str(manifests[0]), 'profile': profile,
                'tool_names': names, 'missing_tools': missing,
                'forbidden_tools': forbidden, 'differences': differences,
                **({'reason': 'Recorded MCP tool profile differs from archive'} if differences else {})}
    except FileNotFoundError:
        return {'state': 'missing' if finished else 'pending'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'unreadable', 'reason': f'Invalid MCP tool manifest: {error}'}


def inspect_case(config_dir: Path, expected: dict, *, finished: bool = False) -> dict:
    """Inspect the case's unique native transcript without trusting a cached audit.

    A startup with no transcript or an incomplete first line is still pending.
    Once the agent/evaluator finishes, the same absence is a failed contract.
    """
    transcripts = list(config_dir.glob('projects/*/*.jsonl'))
    if len(transcripts) > 1:
        return {'state': 'ambiguous', 'reason': 'Multiple native transcripts for one case'}
    if not transcripts:
        return {'state': 'missing' if finished else 'pending' if config_dir.exists() else 'not_started'}
    try:
        result = inspect_listing(transcripts[0], expected)
        result['transcript'] = str(transcripts[0])
    except FileNotFoundError:
        # A live CLI can publish or move its transcript between glob and open.
        result = {'state': 'missing' if finished else 'pending'}
    except (OSError, ValueError) as error:
        result = {'state': 'unreadable', 'reason': str(error)}
    if finished and result['state'] == 'pending':
        result['state'] = 'missing'
    if expected.get('tool_profile') is not None:
        tools = inspect_tool_profile(config_dir.parent, expected['tool_profile'], finished=finished)
        result['mcp_tool_profile'] = tools
        if result['state'] == 'match' and tools['state'] != 'match':
            result.update(state=tools['state'], reason=tools.get('reason', 'MCP tool profile not verified'))
    return result
