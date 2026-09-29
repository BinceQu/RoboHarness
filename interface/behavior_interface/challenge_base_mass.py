"""将 R1Pro chassis（base_footprint）质量对齐到 Challenge 数据采集配置。

官方 HeavyRobotWrapper 把 base mass 设为 250kg。该设置只能在环境初始化
期间执行；代码热加载不得修改任何运行中的物理状态。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

# 与 OmniGibson HeavyRobotWrapper / 数据采集配置一致
DEFAULT_CHALLENGE_BASE_MASS_KG = 250.0
CHALLENGE_BASE_MASS_KG = float(
    os.environ.get("BEHAVIOR_BASE_MASS_KG", DEFAULT_CHALLENGE_BASE_MASS_KG)
)
# PhysX/USD 读回常为 249.99998；1e-6 会反复误判为未对齐并 stop/play
ALIGN_TOLERANCE_KG = float(os.environ.get("BEHAVIOR_BASE_MASS_TOL_KG", "0.1"))


def is_challenge_base_mass_aligned(
    mass_kg: Optional[float],
    *,
    target_kg: Optional[float] = None,
    tol_kg: float = ALIGN_TOLERANCE_KG,
) -> bool:
    if mass_kg is None:
        return False
    target = float(CHALLENGE_BASE_MASS_KG if target_kg is None else target_kg)
    return abs(float(mass_kg) - target) <= float(tol_kg)


def apply_challenge_base_mass(
    robot: Any,
    *,
    mass_kg: Optional[float] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    preserve_sim_state: bool = True,
) -> dict[str, Any]:
    """把 robot.base_footprint_link.mass 设为 Challenge 对齐值（须在仿真线程调用）。

    已对齐（默认 |Δ| <= 0.1kg）时绝不调用 og.sim.stopped()。
    真正改质量时，默认 dump_state → stop/set → play → load_state，
    避免 play() 里的 robot.reset() 冲掉现场。
    """
    target = float(CHALLENGE_BASE_MASS_KG if mass_kg is None else mass_kg)
    if robot is None:
        return {"ok": False, "error": "robot is None", "target_kg": target}

    link = getattr(robot, "base_footprint_link", None)
    if link is None:
        return {"ok": False, "error": "robot has no base_footprint_link", "target_kg": target}

    before = float(getattr(link, "mass", float("nan")))
    if is_challenge_base_mass_aligned(before, target_kg=target):
        try:
            robot._challenge_base_mass_pending = False
        except Exception:
            pass
        result = {
            "ok": True,
            "changed": False,
            "skipped": True,
            "reason": "already_aligned",
            "before_kg": before,
            "after_kg": before,
            "target_kg": target,
            "tol_kg": ALIGN_TOLERANCE_KG,
        }
        if log_fn is not None:
            log_fn(
                f"challenge base mass already aligned: {before:.6f}kg "
                f"(target={target:.1f}kg tol={ALIGN_TOLERANCE_KG}kg); skip sim.stopped()"
            )
        return result

    import omnigibson as og

    saved_state = None
    used_stopped = False
    if getattr(og, "sim", None) is not None:
        used_stopped = True
        # play() 在 was_stopped 时会 robot.reset()；先 dump 再 load 保住场景/机器人状态
        if preserve_sim_state and og.sim.is_playing():
            try:
                saved_state = og.sim.dump_state()
            except Exception as exc:
                saved_state = None
                if log_fn is not None:
                    log_fn(f"WARN challenge base mass dump_state failed: {exc}")
        with og.sim.stopped():
            link.mass = target
        if saved_state is not None:
            try:
                og.sim.load_state(saved_state)
            except Exception as exc:
                if log_fn is not None:
                    log_fn(f"WARN challenge base mass load_state failed: {exc}")
                saved_state = "load_failed"
    else:
        link.mass = target

    after = float(getattr(link, "mass", float("nan")))
    try:
        robot._challenge_base_mass_pending = False
    except Exception:
        pass
    result = {
        "ok": True,
        "changed": True,
        "skipped": False,
        "before_kg": before,
        "after_kg": after,
        "target_kg": target,
        "tol_kg": ALIGN_TOLERANCE_KG,
        "used_sim_stopped": used_stopped,
        "state_restored": saved_state is not None and saved_state != "load_failed",
    }
    if log_fn is not None:
        log_fn(
            f"challenge base mass aligned: {before:.6f}kg -> {after:.6f}kg "
            f"(target={target:.1f}kg stopped={used_stopped} "
            f"state_restored={result['state_restored']})"
        )
    return result


def read_challenge_base_mass(robot: Any) -> Optional[float]:
    link = getattr(robot, "base_footprint_link", None) if robot is not None else None
    if link is None:
        return None
    try:
        return float(link.mass)
    except Exception:
        return None


def _running_server_from_flask() -> Any:
    from flask import current_app, has_app_context

    from behavior_interface.web_runtime_reload import _running_server

    if not has_app_context():
        return None
    return _running_server(current_app._get_current_object())


def _patch_version_endpoint_with_base_mass(server: Any) -> bool:
    """热补丁 /api/skills/version，回报当前 base mass。"""
    try:
        from flask import current_app, has_app_context, jsonify

        if not has_app_context():
            return False
        app = current_app._get_current_object()
        old = app.view_functions.get("api_skills_version")
        if old is None:
            return False
        if getattr(old, "_challenge_base_mass_patched", False):
            return True

        def api_skills_version_with_mass():
            payload = old()
            try:
                data = payload.get_json() if hasattr(payload, "get_json") else None
            except Exception:
                data = None
            if not isinstance(data, dict):
                return payload
            robot = getattr(server, "robot", None)
            live_kg = read_challenge_base_mass(robot)
            base_mass = getattr(server, "challenge_base_mass", None)
            if not isinstance(base_mass, dict):
                base_mass = {}
            if live_kg is not None:
                base_mass = {
                    **base_mass,
                    "ok": True,
                    "after_kg": live_kg,
                    "target_kg": CHALLENGE_BASE_MASS_KG,
                    "aligned": is_challenge_base_mass_aligned(live_kg),
                }
            data["challenge_base_mass"] = base_mass
            data["challenge_base_mass_kg"] = live_kg
            return jsonify(data)

        api_skills_version_with_mass._challenge_base_mass_patched = True  # type: ignore[attr-defined]
        app.view_functions["api_skills_version"] = api_skills_version_with_mass
        return True
    except Exception as exc:
        print(f"[skills.reload] version patch skipped: {exc}", flush=True)
        return False


def hot_apply_challenge_base_mass() -> dict[str, Any]:
    """兼容旧调用方：热加载只补 version，绝不修改或排队修改物理状态。"""
    try:
        server = _running_server_from_flask()
    except Exception as exc:
        return {"ok": False, "error": f"no running server: {exc}", "installed": False}

    if server is None:
        return {"ok": False, "error": "no Flask app context", "installed": False}
    if getattr(server, "dry_run", False):
        return {"ok": True, "skipped": True, "reason": "dry_run", "installed": False}

    version_patched = _patch_version_endpoint_with_base_mass(server)
    live_kg = read_challenge_base_mass(getattr(server, "robot", None))

    result = {
        "ok": True,
        "installed": True,
        "queued": False,
        "skipped": True,
        "reason": (
            "already_aligned"
            if is_challenge_base_mass_aligned(live_kg)
            else "restart_required"
        ),
        "target_kg": CHALLENGE_BASE_MASS_KG,
        "version_patched": version_patched,
        "live_kg": live_kg,
        "aligned": is_challenge_base_mass_aligned(live_kg),
    }
    print(f"[skills.reload] challenge_base_mass_queue={result}", flush=True)
    return result
