from __future__ import annotations


from tools.util.motion_common import _unsupported


def reset_body(
    ctx,
    keep_ori_arm: str = "none",
    timeout_s: float = 45.0,
    trunk_max_step: float | None = None,
    shoulder_iters: int = 4,
    pitch_deg: float = 0.0,
    **kwargs,
):
    yield from _unsupported(
        ctx,
        "reset_body",
        {
            "keep_ori_arm": keep_ori_arm,
            "timeout_s": timeout_s,
            "trunk_max_step": trunk_max_step,
            "shoulder_iters": shoulder_iters,
            "pitch_deg": pitch_deg,
            **kwargs,
        },
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("reset_body", reset_body)
