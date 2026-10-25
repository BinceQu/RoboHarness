from __future__ import annotations


from tools.util.motion_common import _unsupported


def move_chassis_to_floor_point(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    nav_timeout_s: float = 120.0,
    ground_tol_m: float = 0.04,
    pos_tol_m: float = 0.12,
    **kwargs,
):
    yield from _unsupported(
        ctx,
        "move_chassis_to_floor_point",
        {
            "session_id": session_id,
            "image_id": image_id,
            "u": u,
            "v": v,
            "nav_timeout_s": nav_timeout_s,
            "ground_tol_m": ground_tol_m,
            "pos_tol_m": pos_tol_m,
            **kwargs,
        },
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("move_chassis_to_floor_point", move_chassis_to_floor_point)
