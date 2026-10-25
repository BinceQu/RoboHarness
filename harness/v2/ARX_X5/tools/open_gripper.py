from __future__ import annotations


from tools.util.motion_common import _gripper


def open_gripper(ctx, arm: str = "right", **kwargs):
    yield from _gripper(
        ctx,
        "open_gripper",
        arm,
        1.0,
        float(kwargs.get("timeout_s", 3.5)),
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("open_gripper", open_gripper)
