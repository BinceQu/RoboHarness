"""Native delivery at asynchronous receipt boundaries, without a simulator."""
import base64
from hashlib import sha256
import json

from PIL import Image
import pytest

from tools.util.files import atomic_json
from tools.util.observation_hook import process


@pytest.fixture
def history(tmp_path, monkeypatch):
    state = tmp_path / '.runtime'
    atomic_json(tmp_path / '.session.json', {'workspace': str(tmp_path), 'state_dir': str(state)})
    atomic_json(state / 'state.json', {'counter': 0, 'latest_image_id': 'head-current'})
    monkeypatch.setenv('ROBOT_WORKSPACE', str(tmp_path))
    return tmp_path, state


def event(kind, call='call1', tool='Bash', session='native1', **extra):
    return dict(hook_event_name=kind, tool_name=tool, session_id=session, tool_use_id=call, **extra)


def reserve(history, n):
    root, state = history
    atomic_json(state / 'state.json', {'counter': n, 'latest_image_id': f'head-{n}'})
    record = root / 'tool_call_history' / f'{n:06d}_capture_head_camera'
    record.mkdir(parents=True)
    return record


def finish(record, n):
    picture = record / 'returned.png'
    Image.new('RGB', (720, 720), (n, 20, 30)).save(picture)
    digest = sha256(picture.read_bytes()).hexdigest()
    atomic_json(record / 'result.json', {'ok': True, 'tool_name': 'capture_head_camera',
        'images': [{'path': str(picture), 'sha256': digest, 'mime_type': 'image/png',
                    'caption': f'head-{n}'}]})
    return digest


def hashes(result):
    blocks = result.get('hookSpecificOutput', {}).get('updatedToolOutput', {}).get('structuredContent', [])
    return [sha256(base64.b64decode(b['source']['data'])).hexdigest()
            for b in blocks if b['type'] == 'image']


@pytest.mark.parametrize('next_tool', ['Bash', 'TaskOutput'])
def test_background_completion_is_delivered_once(history, next_tool):
    process(event('PreToolUse'))
    record = reserve(history, 1)
    assert process(event('PostToolUse', tool_response={'stdout': 'Running in background'})) == {}
    expected = finish(record, 1)
    if next_tool == 'Bash':
        process(event('PreToolUse', call='call2'))
    response = {'stdout': 'shell text', 'stderr': 'warning'} if next_tool == 'Bash' else {
        'retrieval_status': 'success', 'task': {'output': 'background text', 'status': 'completed'}}
    output = process(event('PostToolUse', call='call2', tool=next_tool, tool_response=response))
    if next_tool == 'TaskOutput':
        assert 'updatedToolOutput' not in output['hookSpecificOutput']
        assert 'Use native Read' in output['hookSpecificOutput']['additionalContext']
        assert str(record / 'returned.png') in output['hookSpecificOutput']['additionalContext']
        process(event('PreToolUse', call='call3'))
        response = {'stdout': 'shell text', 'stderr': 'warning'}
        output = process(event('PostToolUse', call='call3', tool_response=response))
    assert hashes(output) == [expected]
    updated = output['hookSpecificOutput']['updatedToolOutput']
    assert all(updated[k] == v for k, v in response.items())
    assert 'shell text' in json.dumps(updated['structuredContent'])
    assert process(event('PostToolUse', call='call3', tool='TaskOutput', tool_response=response)) == {}


def test_unfinished_hole_does_not_block_newer_receipts_or_late_completion(history):
    process(event('PreToolUse'))
    first = reserve(history, 1)
    assert process(event('PostToolUse', tool_response={'stdout': 'background'})) == {}
    process(event('PreToolUse', call='call2'))
    second = reserve(history, 2)
    h2 = finish(second, 2)
    assert hashes(process(event('PostToolUse', call='call2', tool_response={'stdout': 'done'}))) == [h2]
    h1 = finish(first, 1)
    result = process(event('PostToolUse', call='call3', tool_response={'stdout': 'late'}))
    assert hashes(result) == [h1]
    text = json.dumps(result)
    assert 'Current image=\\"head-2\\"' in text


def test_failure_retains_bytes_for_next_success_and_validates_before_commit(history):
    process(event('PreToolUse'))
    record = reserve(history, 1)
    expected = finish(record, 1)
    failure = process(event('PostToolUseFailure', error='shell failed'))
    assert 'updatedToolOutput' not in failure['hookSpecificOutput']
    picture = record / 'returned.png'
    good = picture.read_bytes()
    picture.write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash differs'):
        process(event('PostToolUse', call='call2', tool_response={}))
    picture.write_bytes(good)
    assert hashes(process(event('PostToolUse', call='call3', tool_response={}))) == [expected]


def test_sessions_are_isolated_and_do_not_replay_existing_history(history):
    finish(reserve(history, 1), 1)
    process(event('PreToolUse', session='native1'))
    process(event('PreToolUse', session='native2'))
    expected = finish(reserve(history, 2), 2)
    for session in ['native1', 'native2']:
        output = process(event('PostToolUse', session=session, tool_response={'stdout': 'ok'}))
        assert hashes(output) == [expected]
        assert process(event('PostToolUse', session=session, tool_response={})) == {}
