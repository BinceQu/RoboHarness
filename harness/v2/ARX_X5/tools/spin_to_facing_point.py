from __future__ import annotations


from tools.util.motion_common import _unsupported


def spin_to_facing_point(
    ctx,
    u: float,
    v: float,
    image_id: str = "",
    max_abs_spin: float = 60.0,
    timeout_s: float = 150.0,
    session_id: str = "",
    **kwargs,
):
    yield from _unsupported(
        ctx,
        "spin_to_facing_point",
        {
            "u": u,
            "v": v,
            "image_id": image_id,
            "max_abs_spin": max_abs_spin,
            "timeout_s": timeout_s,
            "session_id": session_id,
            **kwargs,
        },
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("spin_to_facing_point", spin_to_facing_point)
