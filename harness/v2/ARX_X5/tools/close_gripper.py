from __future__ import annotations


from tools.util.motion_common import _gripper


def close_gripper(ctx, arm: str = "right", **kwargs):
    yield from _gripper(
        ctx,
        "close_gripper",
        arm,
        0.0,
        float(kwargs.get("timeout_s", 3.5)),
        confirm_close_preload=True,
    )
    if ctx.result is not None:
        ctx.result["grasp_detected"] = "unknown"
        ctx.result["contact_truth_used"] = False


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("close_gripper", close_gripper)
