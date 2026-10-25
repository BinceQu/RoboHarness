from __future__ import annotations


from tools.util.motion_common import _unsupported


def adjust_chassis(
    ctx,
    forward: float = 0.0,
    translation: float = 0.0,
    spin: float = 0.0,
    timeout_s: float = 120.0,
    **kwargs,
):
    yield from _unsupported(
        ctx,
        "adjust_chassis",
        {
            "forward": forward,
            "translation": translation,
            "spin": spin,
            "timeout_s": timeout_s,
            **kwargs,
        },
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("adjust_chassis", adjust_chassis)
