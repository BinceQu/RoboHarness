from __future__ import annotations


from tools.util.motion_common import _capture


def capture_left_wrist_camera(ctx, session_id: str, **kwargs):
    del kwargs
    yield from _capture(
        ctx,
        tool_name="capture_left_wrist_camera",
        role="left_wrist",
        session_id=session_id,
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("capture_left_wrist_camera", capture_left_wrist_camera)
