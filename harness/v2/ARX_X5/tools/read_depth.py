from __future__ import annotations

import math
from tools.util.contract import (
    validate_read_depth_args,
)
from tools.util.geometry import (
    relative_uv_to_pixel,
)

from tools.util.motion_common import _current_snapshot, _set_result, _snapshot_geometry


def read_depth(ctx, session_id: str, image_id: str, u: float, v: float):
    common = {
        "depth_m": None,
        "unit": "m",
        "image_id": str(image_id or ""),
        "source": "policy_owned_frozen_robodojo_depth",
    }
    try:
        request = validate_read_depth_args(
            {"session_id": session_id, "image_id": image_id, "u": u, "v": v}
        )
        snapshot = _current_snapshot(
            ctx,
            ctx.snapshots.load(request["session_id"], request["image_id"]),
        )
        depth, _intrinsic, _extrinsic = _snapshot_geometry(snapshot)
        px, py = relative_uv_to_pixel(
            request["u"], request["v"], depth.shape[1], depth.shape[0]
        )
        value = float(depth[py, px])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"depth is invalid at selected pixel ({px}, {py})")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        _set_result(ctx, "read_depth", {**common, "ok": False, "error": str(exc)})
        yield ctx.world.hold_action()
        return
    _set_result(
        ctx,
        "read_depth",
        {
            **common,
            "ok": True,
            "error": None,
            "depth_m": value,
            "pixel_uv": [px, py],
            "neighborhood_interpolation": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("read_depth", read_depth)
