from __future__ import annotations


from tools.util.motion_common import _eef_adjust


def adjust_left_eef_pose_in_wrist_frame(
    ctx,
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
    yield from _eef_adjust(
        ctx,
        tool_name="adjust_left_eef_pose_in_wrist_frame",
        arm="left",
        frame="wrist",
        forward=forward,
        upward=upward,
        leftward=leftward,
        x=None,
        y=None,
        z=None,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("adjust_left_eef_pose_in_wrist_frame", adjust_left_eef_pose_in_wrist_frame)
