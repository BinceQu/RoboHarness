from __future__ import annotations


from tools.util.motion_common import _eef_adjust


def adjust_left_eef_pose_in_head_frame(
    ctx,
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
    yield from _eef_adjust(
        ctx,
        tool_name="adjust_left_eef_pose_in_head_frame",
        arm="left",
        frame="head",
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


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("adjust_left_eef_pose_in_head_frame", adjust_left_eef_pose_in_head_frame)
