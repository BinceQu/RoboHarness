"""Explicit development acceptance on a connected, disposable X5 episode.

Consumes two evaluator actions: a 0.015 rad right-base target adjustment and
restoration. This is not a scored task run. All inputs/results are retained
by the same workspace history mechanism used by native agents.
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    sys.path.insert(0, str(root))
    from tools.util.cli import request
    from robot_config.control import Robot
    from PIL import Image
    robot = Robot()
    cases = []

    def call(name, arguments):
        value = request({'method':'call', 'tool':name, 'arguments':arguments})
        assert not value.get('error'), value
        cases.append({'tool':name, 'history':value['history'],
                      'actions':value['evaluator_action_count'], 'error':value.get('error')})
        return value

    before = robot.status()['evaluator']['sequence']
    capture = call('capture_head_camera', {})
    assert Image.open(capture['images'][-1]['path']).size == (640,480)
    # The center of this image is an ordinary depth-bearing tabletop surface;
    # this is a depth/coordinate transport check, not a semantic target test.
    call('read_depth', {'image_id':capture['image_id'], 'u':500, 'v':500})
    for side in ('left','right'):
        wrist = call(f'capture_{side}_wrist_camera', {})
        assert wrist['feed'] == side+'_wrist'
        assert any(image['caption'].startswith('CURRENT '+side+'_wrist') for image in wrist['images'])
    assert robot.status()['evaluator']['sequence'] == before
    call('activate_skill', {'name':'pick-up-object'})
    call('deactivate_skill', {})
    state = json.loads((root/'perception/proprioception/state.json').read_text())['public_state']
    keys = ('left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state')
    original = {key:list(state[key]) for key in keys}
    adjusted = {key:list(value) for key,value in original.items()}
    adjusted['right_arm_joint_state'][0] += .015
    try:
        first = call('joint_command', {'action':adjusted})
        assert first['evaluator_action_count'] == 1
        assert robot.status()['evaluator']['sequence'] == before+1
    finally:
        restored = call('joint_command', {'action':original})
        assert restored['evaluator_action_count'] == 1
    assert robot.status()['evaluator']['sequence'] == before+2
    report = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'kind':'live transport/control smoke; not an official task score', 'passed':len(cases),
        'workspace':str(root), 'cases':cases,
        'evaluator_actions':2, 'original_camera_size':[640,480],
        'final_sequence':robot.status()['evaluator']['sequence']}
    output = root.parent/'live-smoke.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
