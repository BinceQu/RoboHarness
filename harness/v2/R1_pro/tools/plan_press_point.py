"""plan_press_point

Unavailable in this evaluator profile. The legacy press planner requires
privileged geometry and has not been ported to observation-only planning.
This compatibility entry returns a failure without sending any joint command
or consuming a simulation tick. It cannot create a press plan. Use documented
joint_control or end-effector movement tools for observable, bounded motions.
"""
from __future__ import annotations

from tools.util.camera_frames import _blocked_payload
from tools.util.motion_common import _set_result
from tools.util.tool import main

NAME = 'plan_press_point'
SCHEMA = {'type': 'object',
 'properties': {'image_id': {'type': 'string',
                             'description': 'The latest clickable head-camera image_id, actually '
                                            '720 x 720; re-ground the point when image_id changes. '
                                            'Wrist images are not clickable.'},
                'u': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel column in the latest 720 x 720 '
                                     'head image: 0 is the left edge and 719 is the right edge. Do '
                                     'not normalize or rescale.'},
                'v': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel row in the latest 720 x 720 '
                                     'head image: 0 is the top edge and 719 is the bottom edge. Do '
                                     'not normalize or rescale.'},
                'plan_arm': {'type': 'string', 'enum': ['any', 'left', 'right'], 'default': 'any'},
                'seed': {'type': 'number', 'default': 42, 'description': '姿态采样种子'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(robot, **kwargs):
    del kwargs
    _set_result(
        robot,
        "plan_press_point",
        _blocked_payload("plan_press_point"),
    )
    yield from ()




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
