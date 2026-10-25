"""Rotate the robot base so a head-camera point lands on the image centerline."""

from __future__ import annotations

import math
from typing import Any, Dict

from behavior_interface.skills import SKILL_REGISTRY, register_skill


_BUILD = "face_to_point_v1_pinhole_yaw_only"


class _ChildCtx:
    """Capture a nested skill result while delegating runtime hooks to parent ctx."""

    def __init__(self, parent):
        self.parent = parent
        self.world = parent.world
        self.result = None

    def log(self, msg: str) -> None:
        self.parent.log(msg)

    def set_status(self, msg: str) -> None:
        self.parent.set_status(msg)

    def set_result(self, payload: Dict[str, Any]) -> None:
        self.result = dict(payload or {})

    def get_last_result(self, skill_name: str):
        return self.parent.get_last_result(skill_name)

    def is_cancelled(self) -> bool:
        return self.parent.is_cancelled()

    def raise_if_cancelled(self, where: str = "") -> None:
        self.parent.raise_if_cancelled(where)


def _head_intrinsics(world) -> tuple[float, float, int, int, str]:
    try:
        from behavior_interface.head_capture import get_head_sensor, head_intrinsics_tuple

        head = get_head_sensor(world)
        if head is not None:
            fl, ha, w, h = head_intrinsics_tuple(head)
            return float(fl), float(ha), int(w), int(h), "head_sensor"
    except Exception:
        pass
    from behavior_interface.head_capture import head_intrinsics_fallback

    fl, ha, w, h = head_intrinsics_fallback()
    return float(fl), float(ha), int(w), int(h), "fallback"


def _image_intrinsics(
    world,
    *,
    session_id: str = "",
    image_id: str = "",
) -> tuple[float, float, int, int, str]:
    session = str(session_id or "").strip()
    image = str(image_id or "").strip()
    if session and image:
        try:
            from behavior_interface import agent_runs

            meta = agent_runs.load_image_meta(session, image)
            camera = dict(meta.get("camera") or {})
            fl = float(camera["focal_length"])
            ha = float(camera["horizontal_aperture"])
            w = int(camera["image_width"])
            h = int(camera["image_height"])
            if fl > 0.0 and ha > 0.0 and w > 0 and h > 0:
                return fl, ha, w, h, "capture_meta"
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            pass
    return _head_intrinsics(world)


def compute_face_to_point_spin_deg(
    *,
    u: float,
    image_width: int,
    focal_length: float,
    horizontal_aperture: float,
) -> float:
    """Return move_in_robot_coord spin that centers pixel u horizontally."""
    w = max(1, int(image_width))
    fx = float(focal_length) / float(horizontal_aperture) * float(w)
    if not math.isfinite(fx) or fx <= 1e-9:
        raise ValueError(f"invalid head camera fx from fl={focal_length}, ha={horizontal_aperture}, w={w}")
    cx = (float(w) - 1.0) * 0.5
    ray_angle_deg = math.degrees(math.atan((float(u) - cx) / fx))
    return -ray_angle_deg


@register_skill(
    "face_to_point",
    description=(
        "输入 Qwen3-VL 0..1000 相对坐标 u/v，按 head 相机内参计算仅水平旋转的 spin，"
        "调用 move_in_robot_coord(spin=...)，使该点落到画面中线；不执行俯仰。"
    ),
)
def face_to_point(
    ctx,
    u: float,
    v: float,
    image_id: str = "",
    max_abs_spin: float = 60.0,
    min_abs_spin: float = 0.25,
    timeout_s: float = 150.0,
    session_id: str = "",
):
    world = ctx.world
    fl, ha, w, h, intr_source = _image_intrinsics(
        world,
        session_id=session_id,
        image_id=image_id,
    )
    u_f = float(u)
    v_f = float(v)
    if not (math.isfinite(u_f) and math.isfinite(v_f)):
        ctx.set_result({
            "ok": False,
            "tool": "face_to_point",
            "build": _BUILD,
            "error": f"non-finite uv: u={u!r}, v={v!r}",
        })
        yield world.hold_action()
        return

    spin = compute_face_to_point_spin_deg(
        u=u_f,
        image_width=w,
        focal_length=fl,
        horizontal_aperture=ha,
    )
    unclamped_spin = float(spin)
    limit = abs(float(max_abs_spin))
    if limit > 0.0:
        spin = max(-limit, min(limit, spin))
    if abs(spin) < abs(float(min_abs_spin)):
        spin = 0.0

    report: Dict[str, Any] = {
        "ok": True,
        "tool": "face_to_point",
        "build": _BUILD,
        "session_id": str(session_id or ""),
        "image_id": str(image_id or ""),
        "input_uv_px": [round(u_f, 3), round(v_f, 3)],
        "image_width": int(w),
        "image_height": int(h),
        "center_u_px": round((float(w) - 1.0) * 0.5, 3),
        "focal_length": float(fl),
        "horizontal_aperture": float(ha),
        "intrinsics_source": intr_source,
        "ray_angle_deg": round(-float(unclamped_spin), 4),
        "spin_deg": round(float(spin), 4),
        "unclamped_spin_deg": round(float(unclamped_spin), 4),
        "max_abs_spin": float(max_abs_spin),
        "min_abs_spin": float(min_abs_spin),
        "exec_args": {
            "forward": 0.0,
            "spin": float(spin),
            "pitch": 0.0,
            "upward": 0.0,
            "timeout_s": float(timeout_s),
        },
    }
    ctx.log(
        f"[face_to_point] uv=({u_f:.1f},{v_f:.1f}) center_u={(w - 1) * 0.5:.1f} "
        f"ray={-unclamped_spin:+.2f}deg -> spin={spin:+.2f}deg"
    )
    ctx.set_status(f"face_to_point spin={spin:+.1f}deg")

    if abs(spin) <= 1e-9:
        report["move_result"] = {
            "ok": True,
            "skipped": True,
            "reason": "point already on horizontal centerline",
        }
        ctx.set_result(report)
        yield world.hold_action()
        return

    spec = SKILL_REGISTRY.get("move_in_robot_coord")
    move_fn = getattr(spec, "fn", None)
    if move_fn is None:
        report["ok"] = False
        report["error"] = "move_in_robot_coord is not registered"
        ctx.set_result(report)
        yield world.hold_action()
        return

    child = _ChildCtx(ctx)
    yield from move_fn(
        child,
        forward=0.0,
        spin=float(spin),
        pitch=0.0,
        upward=0.0,
        timeout_s=float(timeout_s),
    )
    move_result = dict(child.result or {})
    report["move_result"] = move_result
    report["ok"] = bool(move_result.get("ok", False))
    if not report["ok"]:
        report["error"] = move_result.get("error") or "move_in_robot_coord failed"
    ctx.set_result(report)
