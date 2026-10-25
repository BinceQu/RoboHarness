from __future__ import annotations


from tools.util.motion_common import _capture


def capture_head_camera(ctx, session_id: str, **kwargs):
    del kwargs
    yield from _capture(
        ctx,
        tool_name="capture_head_camera",
        role="head",
        session_id=session_id,
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("capture_head_camera", capture_head_camera)
