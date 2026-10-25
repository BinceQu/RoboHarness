"""adjust_left_eef_pose_in_wrist_frame

以实时 left wrist camera 为机体系调整左 EEF；平移单位统一为米，roll/pitch/yaw 为夹爪局部轴增量。旋转阶段固定 J1-J4，仅求解和驱动 J5-J7；J8 不参与并保持锁定。执行后返回左腕视图。

How this tool works:
Express translation (metres) in the starting wrist-camera frame. Rotations are degrees about the gripper's local axes. Transform the offset, solve left-arm IK, and interpolate joint targets while retaining the other joints and any carrying finger effort.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.motion_common import _adjust_eef_pose_in_wrist_frame_local
from tools.util.tool import main

NAME = 'adjust_left_eef_pose_in_wrist_frame'
SCHEMA = {'type': 'object',
 'properties': {'forward': {'type': 'number', 'default': 0.0, 'description': 'm(+沿左腕视线向前/-向后)'},
                'leftward': {'type': 'number',
                             'default': 0.0,
                             'description': 'm(+画面向左/-向右，与底部x标尺方向相反)'},
                'upward': {'type': 'number', 'default': 0.0, 'description': 'm(+画面向上/-向下)'},
                'roll': {'type': 'number',
                         'default': 0.0,
                         'description': 'deg(夹爪局部 roll 增量；仅 J5-J7 求解)'},
                'pitch': {'type': 'number',
                          'default': 0.0,
                          'description': 'deg(夹爪局部 pitch 增量；+为指尖上抬)'},
                'yaw': {'type': 'number',
                        'default': 0.0,
                        'description': 'deg(夹爪局部 yaw 增量；仅 J5-J7 求解)'},
                'pos_tol': {'type': 'number', 'default': 0.012, 'description': 'm'},
                'ori_tol_deg': {'type': 'number',
                                'default': 3.0,
                                'description': 'deg(最终夹爪姿态收敛容差)'}},
 'additionalProperties': False}


def actions(
    robot,
    forward: float = 0.0,
    leftward: float = 0.0,
    upward: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pos_tol: float = 0.012,
    ori_tol_deg: float = 3.0,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    actions = _adjust_eef_pose_in_wrist_frame_local(
        robot,
        arm="left",
        tool_name="adjust_left_eef_pose_in_wrist_frame",
        forward=forward,
        leftward=leftward,
        upward=upward,
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
