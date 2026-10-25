"""adjust_left_eef_pose_in_head_frame

Adjust the left EEF by incremental translation and local-gripper RPY. For translation, use only x/y/z in the current robot base frame (+X forward, +Y left, +Z up), or only forward/leftward/upward in the starting head-camera frame. Never mix the two parameter families, even when one value is zero. y and leftward have the same positive-left sign but belong to different frames. For a horizontal push, issue short +x increments and verify external-object motion from the next evaluator RGB-D observation; tool success itself means the requested EEF pose converged.

How this tool works:
Use either base-frame x/y/z or starting-head-camera forward/leftward/upward for translation (metres); never mix them. Rotations are degrees about the gripper's local axes. Transform the offset, solve left-arm IK, and interpolate joint targets while retaining the other joints and any carrying finger effort.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.motion_common import _adjust_eef_pose_in_head_frame_local
from tools.util.tool import main

NAME = 'adjust_left_eef_pose_in_head_frame'
SCHEMA = {'type': 'object',
 'properties': {'x': {'type': 'number',
                      'description': 'Robot-base delta: +X is chassis-forward. Use short +X '
                                     'increments for chassis-horizontal pushing. Use only with '
                                     'y/z; never combine with camera-frame parameters.; m'},
                'y': {'type': 'number',
                      'description': 'Robot-base delta: +Y is chassis-left. Same positive-left '
                                     'sign as leftward, but base-fixed rather than '
                                     'camera-rotated.; m'},
                'z': {'type': 'number',
                      'description': 'Robot-base delta: +Z is chassis-up. Use only with x/y.; m'},
                'forward': {'type': 'number',
                            'description': 'Starting-head-camera delta along the viewing '
                                           'direction. This includes a vertical component when the '
                                           'camera is pitched and is not chassis-horizontal '
                                           'forward. Use only with leftward/upward; never combine '
                                           'with x/y/z.; m'},
                'leftward': {'type': 'number',
                             'description': 'Starting-head-camera delta toward image left. Same '
                                            'positive-left sign as y, but follows camera '
                                            'orientation.; m'},
                'upward': {'type': 'number',
                           'description': 'Starting-head-camera delta toward image up. Use only '
                                          'with forward/leftward.; m'},
                'roll': {'type': 'number',
                         'default': 0.0,
                         'description': 'deg(left gripper local axis)'},
                'pitch': {'type': 'number', 'default': 0.0, 'description': 'deg(+ fingertips-up)'},
                'yaw': {'type': 'number',
                        'default': 0.0,
                        'description': 'deg(left gripper local axis)'},
                'pos_tol': {'type': 'number', 'default': 0.012, 'description': 'm'},
                'ori_tol_deg': {'type': 'number', 'default': 3.0, 'description': 'deg'},
                'max_steps': {'type': 'number',
                              'default': 240,
                              'description': 'observation/action steps'},
                'timeout_s': {'type': 'number', 'default': 60.0, 'description': 's'}},
 'additionalProperties': False}


def actions(
    robot,
    forward: float | None = None,
    upward: float | None = None,
    leftward: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    actions = _adjust_eef_pose_in_head_frame_local(
        robot,
        arm="left",
        tool_name="adjust_left_eef_pose_in_head_frame",
        forward=forward,
        upward=upward,
        leftward=leftward,
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
    )
    for action in actions:
        yield action




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
