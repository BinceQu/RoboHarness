import base64
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from PIL import Image
import pytest

from tools.util.runtime import Runtime
from tools.util.observation import ObservationAdapter
from tools.util.geometry import relative_uv_to_pixel
from tools.joint_command import validate_command
from tools.util.receipt import without_ok
from tests.upstream.helpers import fake_observation, apply_action

TEMPLATE = Path(__file__).resolve().parents[2]/'ARX_X5'

@pytest.fixture
def robot(tmp_path):
    root = tmp_path/'ARX_X5'
    root.mkdir()
    (root/'.session.json').write_text(json.dumps({'session_id':'unit','task':'unit'}))
    shutil.copytree(TEMPLATE/'robot_config', root/'robot_config')
    shutil.copytree(TEMPLATE/'skills', root/'skills')
    runtime = Runtime(root)
    observation = fake_observation(48,64)
    runtime.update_obs(observation)
    stop = threading.Event()
    failures = []
    def evaluator():
        nonlocal observation
        try:
            while not stop.is_set():
                for action in runtime.get_action():
                    observation = apply_action(observation, action)
                    runtime.update_obs(observation)
                stop.wait(.005)
        except Exception as exc:
            failures.append(exc)
    worker = threading.Thread(target=evaluator, daemon=True)
    worker.start()
    yield runtime
    stop.set()
    worker.join(10)
    runtime.close()
    assert not failures
    for record in (root / 'tool_call_history').glob('*/result.json'):
        result = json.loads(record.read_text())
        assert result == without_ok(result), str(record)


def test_cli_removes_nested_flags_and_preserves_failures(robot, monkeypatch, capsys):
    import sys
    from tools.util import cli
    payload = {'ok': True, 'actual': .058, 'linear_target_reached': False,
               'rows': [{'ok': False, 'error': 'blocked'}]}
    result = robot.history  # Worker has no active control operation.
    assert result is None
    from tools.util.history import History
    returned = History(robot.root, 'adjust_chassis', {}).finish(payload)
    assert returned == without_ok(returned)
    assert returned['actual'] == .058 and returned['linear_target_reached'] is False
    assert returned['rows'] == [{'error': 'blocked'}]
    monkeypatch.setattr(cli, 'request', lambda payload: robot.call(payload['tool'], payload['arguments']))
    monkeypatch.setattr(sys, 'argv', ['capture_head_camera.py', '{}'])
    cli.run_cli('capture_head_camera')
    capture = json.loads(capsys.readouterr().out)
    assert capture == without_ok(capture) and capture['images']
    monkeypatch.setattr(sys, 'argv', ['capture_head_camera.py', '{"session_id":"foreign"}'])
    with pytest.raises(SystemExit) as exit_info:
        cli.run_cli('capture_head_camera')
    assert exit_info.value.code == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure == without_ok(failure) and 'another session' in failure['error']

def test_capture_receipt_and_native_image_hook(robot, monkeypatch):
    from tools.util import observation_hook as hook
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError('Ordinary camera tools must not render the robot body')
    monkeypatch.setattr('tools.util.render.BodyViewRenderer', forbidden_renderer)
    settings = lambda: {'workspace':str(robot.root),'state_dir':str(robot.root/'.runtime')}
    monkeypatch.setattr(hook, 'settings', settings)
    event = {'tool_name':'Bash','session_id':'claude-test','tool_use_id':'tool-test','hook_event_name':'PreToolUse'}
    assert hook.process(event) == {}
    result = robot.call('capture_head_camera', {})
    assert not result.get('error'), result
    assert result['evaluator_action_count'] == 0
    assert Image.open(result['images'][0]['path']).size == (64,48)
    response = hook.process({**event,'hook_event_name':'PostToolUse','tool_response':{'stdout':json.dumps(result),'stderr':''}})
    blocks = response['hookSpecificOutput']['updatedToolOutput']['structuredContent']
    images = [block for block in blocks if block['type']=='image']
    assert len(images) == 1
    raw = base64.b64decode(images[0]['source']['data'])
    assert hashlib.sha256(raw).hexdigest() == result['images'][0]['sha256']
    assert robot.body_renderer is None
    assert not (robot.root/'perception/proprioception/three_views.png').exists()
    assert (robot.root/'perception/proprioception/state.json').is_file()
    assert json.loads((robot.root/'perception/latest.json').read_text())['body_image'] is None
    assert (robot.root/'perception/exteroception/head/depth.npy').is_file()
    assert (robot.root/'perception/exteroception/tracks.json').is_file()
    # A summarized/truncated Bash result must not be expanded by the image hook.
    record = Path(result['history'])/'result.json'
    receipt = json.loads(record.read_text())
    receipt['plan_audit'] = {'candidates':['large diagnostic '*20000]}
    receipt['plan_id'] = 'plan_test'
    record.write_text(json.dumps(receipt))
    response = hook.process({**event,'hook_event_name':'PostToolUse',
                            'tool_response':{'stdout':'plan_id: plan_test','stderr':''}})
    bounded = response['hookSpecificOutput']['updatedToolOutput']['structuredContent']
    assert sum(len(b.get('text','')) for b in bounded) < 10000
    summary = json.loads(bounded[1]['text'])
    assert summary['plan_id']=='plan_test'
    assert summary['full_result_path']==str(record)
    assert 'candidates' not in summary
    assert len(record.read_text()) > 300000
    assert [b for b in bounded if b['type']=='image']==images

def test_click_geometry_history_and_zero_ticks(robot):
    capture = robot.call('capture_head_camera', {})
    assert not capture.get('error'), capture
    result = robot.call('read_depth', {'image_id':capture['image_id'],'u':500,'v':500})
    assert not result.get('error'), result
    assert result['evaluator_action_count'] == 0
    clicks = json.loads((Path(result['history'])/'clicks.json').read_text())
    assert clicks[0]['pixel_uv'] == [32,24]
    assert list(Path(result['history']).glob('*_clicks.png'))

def test_exact_one_joint_step_and_latched_grippers(robot):
    action = {side+'_arm_joint_state':[0]*6 for side in ('left','right')}
    action.update(left_ee_joint_state=[.3],right_ee_joint_state=[.6])
    action['right_arm_joint_state'][0]=.02
    before = robot.adapter.status()['sequence']
    result = robot.call('joint_command', {'action':action})
    assert not result.get('error'), result
    assert result['evaluator_action_count'] == 1
    assert robot.body_renderer is None
    assert not result.get('capture_error'), result
    assert robot.adapter.status()['sequence'] == before+1
    saved = json.loads((Path(result['history'])/'joint_commands.jsonl').read_text())
    assert saved['action'] == action
    next_action = robot.world.make_joint_action(right_arm=[0]*6)
    assert next_action['left_ee_joint_state'] == [.3]
    assert next_action['right_ee_joint_state'] == [.6]
    np.testing.assert_allclose(robot.world.arm_qpos_list('right')[0], .02)


def test_on_demand_body_view_is_read_only_and_reaches_native_hook(robot, monkeypatch):
    from tools.util import observation_hook as hook
    capture = robot.call('capture_head_camera', {})
    assert (not capture.get('error')), capture
    monkeypatch.setattr(hook, 'settings', lambda: {
        'workspace': str(robot.root), 'state_dir': str(robot.root/'.runtime')})
    event = {'tool_name': 'Bash', 'session_id': 'body-test', 'tool_use_id': 'body-call',
             'hook_event_name': 'PreToolUse'}
    hook.process(event)
    latest_before = json.loads((robot.root/'perception/latest.json').read_text())
    state_before = json.loads((robot.root/'.runtime/state.json').read_text())
    sequence = robot.adapter.status()['sequence']
    result = robot.call('read_proprioception', {})
    assert (not result.get('error')), result
    assert result['evaluator_action_count'] == 0
    assert robot.adapter.status()['sequence'] == sequence
    assert result['sequence'] == sequence
    assert len(result['images']) == 1
    media = result['images'][0]
    assert Image.open(media['path']).size == (1320, 650)
    current = json.loads((robot.root/'perception/latest.json').read_text())
    assert current['cameras'] == latest_before['cameras']
    assert current['tracks'] == latest_before['tracks']
    assert json.loads((robot.root/'.runtime/state.json').read_text())['latest_image_id'] == state_before['latest_image_id']
    assert json.loads(Path(result['state_path']).read_text()) == result['state']
    assert not list(Path(result['history']).glob('observation/head/*'))
    assert not (Path(result['history'])/'joint_commands.jsonl').exists()
    response = hook.process({**event, 'hook_event_name': 'PostToolUse',
        'tool_response': {'stdout': json.dumps(result), 'stderr': ''}})
    blocks = response['hookSpecificOutput']['updatedToolOutput']['structuredContent']
    images = [block for block in blocks if block['type'] == 'image']
    assert len(images) == 1
    assert hashlib.sha256(base64.b64decode(images[0]['source']['data'])).hexdigest() == media['sha256']

    # Motion updates numeric state, expires the current view, and keeps history.
    def forbidden_render(*args, **kwargs):
        raise AssertionError('An ordinary action must not invoke body rendering')
    monkeypatch.setattr(robot.body_renderer, 'save', forbidden_render)
    action = {side+'_arm_joint_state': [0]*6 for side in ('left', 'right')}
    action.update(left_ee_joint_state=[.3], right_ee_joint_state=[.6])
    action['right_arm_joint_state'][0] = .15
    moved = robot.call('joint_command', {'action': action})
    assert (not moved.get('error')), moved
    assert moved['evaluator_action_count'] == 1
    body = json.loads((robot.root/'perception/proprioception/state.json').read_text())
    assert body['sequence'] == sequence + 1
    assert body['render_joint_positions']['right_joint1'] == pytest.approx(.15)
    assert not (robot.root/'perception/proprioception/three_views.png').exists()
    assert json.loads((robot.root/'perception/latest.json').read_text())['body_image'] is None
    assert hashlib.sha256(Path(media['path']).read_bytes()).hexdigest() == media['sha256']


def test_body_read_before_camera_and_render_failure(robot, monkeypatch):
    class Renderer:
        def __init__(self, root):
            self.calls = 0
        def save(self, joints, path, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError('render unavailable')
            Image.new('RGB', (12, 8)).save(path)
        def close(self):
            pass
    monkeypatch.setattr('tools.util.render.BodyViewRenderer', Renderer)
    first = robot.call('read_proprioception', {})
    assert (not first.get('error')) and first['evaluator_action_count'] == 0, first
    assert first['perception'] is None
    assert not (robot.root/'perception/latest.json').exists()
    assert json.loads((robot.root/'.runtime/state.json').read_text())['latest_image_id'] is None
    failed = robot.call('read_proprioception', {})
    assert failed.get('error') and 'render unavailable' in failed['error']
    assert failed['evaluator_action_count'] == 0
    assert not failed.get('images')
    assert not (robot.root/'perception/proprioception/three_views.png').exists()
    assert Path(first['images'][0]['path']).exists()
    invalid = robot.call('read_proprioception', {'unexpected': True})
    assert invalid.get('error') and invalid['evaluator_action_count'] == 0
    robot.reset()
    disconnected = robot.call('read_proprioception', {})
    assert disconnected.get('error') and 'not connected' in disconnected['error']
    assert not disconnected.get('images')

def test_skill_activation_and_free_deactivation(robot):
    result = robot.call('activate_skill', {'name':'pick-up-object'})
    assert result['active_skill'] == 'pick-up-object'
    assert 'instructions' in result
    end = robot.call('deactivate_skill', {})
    assert (not end.get('error')) and end['active_skill'] is None
    assert end['evaluator_action_count'] == 0

def test_no_privileged_observation_fields():
    adapter = ObservationAdapter()
    adapter.ingest(fake_observation())
    value = adapter.observation()
    assert set(value) == {'instruction','vision','state','action'}
    assert 'privileged_object_pose' not in value['state']
    assert 'oracle_camera' not in value['vision']
    assert 'segmentation' not in value['vision']['cam_head']

@pytest.mark.parametrize('shape', [(640,480),(720,720),(1920,1080)])
def test_normalized_contract_preserves_original_resolution(shape):
    w,h=shape
    assert relative_uv_to_pixel(0,0,w,h) == (0,0)
    assert relative_uv_to_pixel(1000,1000,w,h) == (w-1,h-1)
    assert relative_uv_to_pixel(345,548,w,h) == (round(.345*(w-1)),round(.548*(h-1)))

@pytest.mark.parametrize('bad', [None,{}, {'left_arm_joint_state':[0]*6},
    {'left_arm_joint_state':[float('nan')]*6,'left_ee_joint_state':[0],
     'right_arm_joint_state':[0]*6,'right_ee_joint_state':[1]},
    {'left_arm_joint_state':[0]*6,'left_ee_joint_state':[2],
     'right_arm_joint_state':[0]*6,'right_ee_joint_state':[1]}])
def test_reject_malformed_atomic_commands(bad):
    with pytest.raises(ValueError):
        validate_command(bad)

def test_foreign_session_cannot_send_commands(robot):
    result = robot.call('capture_head_camera', {'session_id':'foreign'})
    assert bool(result.get('error'))
    assert 'another session' in result['error']

@pytest.mark.parametrize('name', [
    'adjust_left_eef_pose_in_head_frame',
    'adjust_right_eef_pose_in_head_frame',
    'adjust_left_eef_pose_in_wrist_frame',
    'adjust_right_eef_pose_in_wrist_frame',
])
def test_eef_motion_strips_metadata_after_ownership_check(robot, name):
    arguments = {'y': .015} if name.endswith('head_frame') else {'leftward': .015}
    sequence = robot.adapter.status()['sequence']
    result = robot.call(name, arguments)
    assert (not result.get('error')), result
    assert result['evaluator_action_count'] > 0
    assert robot.adapter.status()['sequence'] > sequence
    assert result['position_error_m'] <= result['position_tolerance_m']
    assert result['observation']['session_id'] == 'unit'
    denied = robot.call(name, {**arguments, 'session_id': 'another-session'})
    assert denied.get('error') and 'another session' in denied['error']

def test_wrist_images_and_diagnostics_reach_native_hook(robot, monkeypatch):
    from tools.util import observation_hook as hook
    monkeypatch.setattr(hook, 'settings', lambda: {
        'workspace':str(robot.root), 'state_dir':str(robot.root/'.runtime')})
    event = {'tool_name':'Bash','session_id':'wrist-test','tool_use_id':'wrist-call',
             'hook_event_name':'PreToolUse'}
    hook.process(event)
    result = robot.call('capture_left_wrist_camera', {})
    assert (not result.get('error')), result
    assert result['feed'] == 'left_wrist'
    assert result['observation']['feed'] == 'left_wrist'
    assert len(result['images']) >= 3  # Grasp volume, wrist overlay, fresh head.
    assert result['images'][-1]['caption'].startswith('CURRENT head')
    assert any(image['caption'].startswith('CURRENT left_wrist') for image in result['images'])
    response = hook.process({**event, 'hook_event_name':'PostToolUse',
        'tool_response':{'stdout':json.dumps(result),'stderr':''}})
    images = [part for part in response['hookSpecificOutput']['updatedToolOutput']['structuredContent']
              if part['type'] == 'image']
    delivered = {hashlib.sha256(base64.b64decode(part['source']['data'])).hexdigest() for part in images}
    assert delivered == {image['sha256'] for image in result['images']}
