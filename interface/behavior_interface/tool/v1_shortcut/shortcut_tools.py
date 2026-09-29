"""Shortcut tool overrides for the v1_shortcut profile.

These wrappers intentionally keep planning/capture tools on the normal code
path.  They override motion/execution tools so the final target is converted to
joint-space once, then played directly as absolute joint targets.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Generator, Optional

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill

_SHORTCUT_BUILD = "v1_shortcut_direct_joint_targets"
_BAD_BASE_ABS_XY_M = 100.0


def _arm_eff(arm: str | None) -> str:
    a = str(arm or "right").strip().lower()
    return a if a in ("left", "right") else "right"


def _as_vec(x, n: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(n)


def _quat_err_deg(q0, q1) -> float:
    q0 = _as_vec(q0, 4)
    q1 = _as_vec(q1, 4)
    n0 = float(np.linalg.norm(q0))
    n1 = float(np.linalg.norm(q1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    dot = abs(float(np.dot(q0 / n0, q1 / n1)))
    return float(math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))))


def _gripper_keep(world, arm: str) -> Optional[list[float]]:
    try:
        vals = world.gripper_qpos_list(arm)
        if vals is not None:
            vals_f = [float(x) for x in vals]
            if vals_f:
                world.set_gripper_pin_qpos(arm, vals_f)
                return vals_f
    except Exception:
        pass
    return None


def _gripper_open_cmd(world, arm: str) -> list[float]:
    try:
        from behavior_interface.skills.arm_reset import _gripper_open_override

        return list(_gripper_open_override(world, arm))
    except Exception:
        return [1.0]


def _gripper_cmd_for_mode(world, arm: str, mode: str | None) -> Optional[list[float]]:
    m = str(mode or "keep").strip().lower()
    if m in ("open", "opened", "true", "1", "yes"):
        return _gripper_open_cmd(world, arm)
    if m in ("close", "closed", "-1"):
        return [-1.0]
    return _gripper_keep(world, arm)


def _force_open_if_needed(world, arm: str, mode: str | None) -> None:
    if str(mode or "").strip().lower() not in ("open", "opened", "true", "1", "yes"):
        return
    try:
        from behavior_interface.skills.arm_reset import _force_open_gripper_qpos

        _force_open_gripper_qpos(world, arm)
    except Exception:
        pass


def _force_set_trunk_qpos(world, target_qpos) -> None:
    _ensure_shortcut_hard_pin(world)
    q_target = np.asarray(target_qpos, dtype=np.float64).reshape(4)
    try:
        world.set_trunk_pin_qpos(q_target.tolist())
    except Exception:
        pass
    if getattr(world, "dry_run", False):
        return
    robot = getattr(world, "robot", None)
    if robot is None:
        return
    try:
        raw_idx = robot.trunk_control_idx
        if hasattr(raw_idx, "detach"):
            raw_idx = raw_idx.detach().cpu().numpy()
        idx = np.asarray(raw_idx, dtype=int).reshape(-1)[:4]
        q0 = robot.get_joint_positions()
        q_new = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        for local_i, joint_i in enumerate(idx):
            q_new[int(joint_i)] = float(q_target[int(local_i)])
        robot.set_joint_positions(q_new)
        try:
            v0 = robot.get_joint_velocities()
            v_new = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
            for joint_i in idx:
                v_new[int(joint_i)] = 0.0
            robot.set_joint_velocities(v_new)
        except Exception:
            pass
    except Exception:
        pass


def _joint_indices_by_names(robot, joint_names) -> list[int]:
    try:
        names = list(robot.joints.keys())
    except Exception:
        return []
    out: list[int] = []
    for name in joint_names:
        try:
            out.append(int(names.index(str(name))))
        except Exception:
            pass
    return out


def _gripper_joint_names(world, arm: str) -> list[str]:
    robot = getattr(world, "robot", None)
    names: list[str] = []
    try:
        names.extend(list(getattr(robot, "finger_joint_names", {}).get(arm, [])))
    except Exception:
        pass
    names.extend([
        f"{arm}_gripper_finger_joint1",
        f"{arm}_gripper_finger_joint2",
    ])
    out: list[str] = []
    seen = set()
    try:
        robot_names = set(robot.joints.keys())
    except Exception:
        robot_names = set()
    for name in names:
        name = str(name)
        if name in seen or (robot_names and name not in robot_names):
            continue
        seen.add(name)
        out.append(name)
    return out


def _force_apply_shortcut_pin_qpos(world) -> None:
    """Directly snap the robot to pinned qpos at explicit shortcut target boundaries."""
    if getattr(world, "dry_run", False):
        return
    robot = getattr(world, "robot", None)
    if robot is None:
        return
    indices_to_zero: list[int] = []
    try:
        q0 = robot.get_joint_positions()
        q_new = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
    except Exception:
        return

    try:
        raw_idx = robot.trunk_control_idx
        if hasattr(raw_idx, "detach"):
            raw_idx = raw_idx.detach().cpu().numpy()
        idx = np.asarray(raw_idx, dtype=int).reshape(-1)[:4]
        tq = np.asarray(world.trunk_pin_qpos_list(), dtype=np.float64).reshape(4)
        for local_i, joint_i in enumerate(idx):
            q_new[int(joint_i)] = float(tq[int(local_i)])
            indices_to_zero.append(int(joint_i))
    except Exception:
        pass

    for arm in ("left", "right"):
        try:
            aq = world.arm_pin_qpos_list(arm)
            if aq is not None:
                idx = _joint_indices_by_names(
                    robot, [f"{arm}_arm_joint{i + 1}" for i in range(7)]
                )
                arr = np.asarray(aq, dtype=np.float64).reshape(-1)
                if len(idx) == 7 and arr.size >= 7:
                    for local_i, joint_i in enumerate(idx):
                        q_new[int(joint_i)] = float(arr[int(local_i)])
                        indices_to_zero.append(int(joint_i))
        except Exception:
            pass
        try:
            gq = world.gripper_pin_qpos_list(arm)
            gnames = _gripper_joint_names(world, arm)
            idx = _joint_indices_by_names(robot, gnames)
            arr = np.asarray(gq, dtype=np.float64).reshape(-1) if gq is not None else np.zeros(0)
            # Only hard-write true finger qpos. A single value can be an action
            # semantic command (open/close), not a joint position.
            if idx and arr.size == len(idx):
                for local_i, joint_i in enumerate(idx):
                    q_new[int(joint_i)] = float(arr[int(local_i)])
                    indices_to_zero.append(int(joint_i))
        except Exception:
            pass

    try:
        robot.set_joint_positions(q_new)
    except Exception:
        return
    _zero_joint_velocities(world, indices_to_zero or None)


def _shortcut_pin_drift(world) -> Dict[str, Any]:
    """Measure drift from shortcut pin targets without changing the robot."""
    out: Dict[str, Any] = {"max_err_rad": 0.0, "parts": {}}
    if getattr(world, "dry_run", False):
        return out
    try:
        tq = np.asarray(world.trunk_pin_qpos_list(), dtype=np.float64).reshape(4)
        q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        err = float(np.linalg.norm(q - tq, ord=np.inf))
        out["parts"]["trunk"] = err
        out["max_err_rad"] = max(float(out["max_err_rad"]), err)
    except Exception:
        pass
    for arm in ("left", "right"):
        try:
            pin = world.arm_pin_qpos_list(arm)
            if pin is None:
                continue
            target = np.asarray(pin, dtype=np.float64).reshape(7)
            q = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7)
            err = float(np.linalg.norm(q - target, ord=np.inf))
            out["parts"][f"arm_{arm}"] = err
            out["max_err_rad"] = max(float(out["max_err_rad"]), err)
        except Exception:
            pass
    return out


def _maybe_correct_shortcut_pin_qpos(
    world,
    *,
    trunk_tol_rad: float = 0.015,
    arm_tol_rad: float = 0.030,
    min_interval_s: float = 0.12,
) -> Optional[Dict[str, Any]]:
    """Low-rate pin correction for idle/capture holds in v1_shortcut.

    Shortcut tools finish by directly placing the trunk / arms at their target
    qpos.  The normal position controllers can still sag a little during idle
    render or capture frames, so we only snap back when measured drift crosses
    a small threshold.  This avoids the old per-physics-step hard pin that
    fought the controller continuously.
    """
    if getattr(world, "dry_run", False):
        return None
    now = time.monotonic()
    last = float(getattr(world, "_codex_v1_shortcut_last_pin_snap_t", 0.0) or 0.0)
    if now - last < float(min_interval_s):
        return None
    drift = _shortcut_pin_drift(world)
    parts = dict(drift.get("parts") or {})
    trunk_err = float(parts.get("trunk", 0.0))
    arm_err = max(
        [float(v) for k, v in parts.items() if str(k).startswith("arm_")] or [0.0]
    )
    if trunk_err <= float(trunk_tol_rad) and arm_err <= float(arm_tol_rad):
        return None
    _force_apply_shortcut_pin_qpos(world)
    report = {
        "trunk_err_rad": trunk_err,
        "arm_err_rad": arm_err,
        "parts": {k: round(float(v), 5) for k, v in parts.items()},
        "t": now,
    }
    world._codex_v1_shortcut_last_pin_snap_t = now
    world._codex_v1_shortcut_last_pin_snap = report
    world._codex_v1_shortcut_pin_snap_count = int(
        getattr(world, "_codex_v1_shortcut_pin_snap_count", 0) or 0
    ) + 1
    return report


def _shortcut_base_pin_error(world) -> Optional[Dict[str, float]]:
    pin = getattr(world, "_codex_v1_shortcut_base_pin", None)
    if not isinstance(pin, dict):
        return None
    try:
        pose = world.robot_pose()
        p = np.asarray(pose.pos, dtype=np.float64).reshape(3)
        pin_pos = np.asarray(pin.get("pos"), dtype=np.float64).reshape(3)
        pin_yaw = float(pin.get("yaw", 0.0))
        yaw_err = (float(pose.yaw) - pin_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return {
            "xy_m": float(np.linalg.norm(p[:2] - pin_pos[:2])),
            "yaw_deg": float(math.degrees(abs(yaw_err))),
        }
    except Exception:
        return None


def _post_step_stabilize_shortcut_pins(world) -> Optional[Dict[str, Any]]:
    """Keep rendered shortcut base, trunk, and arms aligned after PhysX integration."""
    if getattr(world, "dry_run", False):
        return None
    drift = _shortcut_pin_drift(world)
    parts = dict(drift.get("parts") or {})
    trunk_err = float(parts.get("trunk", 0.0))
    arm_err = max(
        [float(v) for k, v in parts.items() if str(k).startswith("arm_")] or [0.0]
    )
    base_err = _shortcut_base_pin_error(world)
    base_reseeded = False
    if base_err is not None and (
        float(base_err.get("xy_m", 0.0)) > 0.25
        or float(base_err.get("yaw_deg", 0.0)) > 45.0
    ):
        # env.reset/task switch/manual teleports leave the old shortcut base pin
        # in world state.  Re-seed instead of pulling the robot back to history.
        try:
            pose = world.robot_pose()
            _set_shortcut_base_pin(world, pose.pos, pose.yaw)
            base_reseeded = True
        except Exception:
            pass
        base_pinned = False
    else:
        base_pinned = _force_apply_shortcut_base_pin(world)
    _force_apply_shortcut_pin_qpos(world)
    report = {
        "base_pinned": bool(base_pinned),
        "base_reseeded": bool(base_reseeded),
        "base_err": {
            k: round(float(v), 5)
            for k, v in dict(base_err or {}).items()
        },
        "trunk_err_rad": trunk_err,
        "arm_err_rad": arm_err,
        "parts": {k: round(float(v), 6) for k, v in parts.items()},
    }
    world._codex_v1_shortcut_post_step_stabilize_count = int(
        getattr(world, "_codex_v1_shortcut_post_step_stabilize_count", 0) or 0
    ) + 1
    world._codex_v1_shortcut_last_post_step_stabilize = report
    return report


def _ensure_shortcut_hard_pin(world) -> None:
    if getattr(world, "dry_run", False) or getattr(world, "_codex_v1_shortcut_hard_pin_v4", False):
        return
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
    except Exception:
        pass
    try:
        import types

        def _shortcut_hold_action_pinned(self):
            _maybe_correct_shortcut_pin_qpos(self)
            return self.pinned_action()

        def _shortcut_empty_action(self):
            return self.hold_action_pinned()

        def _shortcut_set_base_velocity(self, vx: float, vy: float, wz: float):
            if abs(float(vx)) + abs(float(vy)) + abs(float(wz)) <= 1e-9:
                _maybe_correct_shortcut_pin_qpos(self)
            return self.pinned_action(base=[float(vx), float(vy), float(wz)])

        world.hold_action_pinned = types.MethodType(_shortcut_hold_action_pinned, world)
        world.hold_action = world.hold_action_pinned
        world.empty_action = types.MethodType(_shortcut_empty_action, world)
        world.set_base_velocity = types.MethodType(_shortcut_set_base_velocity, world)
        world.shortcut_post_step_stabilize_now = types.MethodType(
            lambda self: _post_step_stabilize_shortcut_pins(self),
            world,
        )
        if hasattr(world, "shortcut_hard_pin_now"):
            try:
                delattr(world, "shortcut_hard_pin_now")
            except Exception:
                world.shortcut_hard_pin_now = None
        world._codex_v1_shortcut_hard_pin_v4 = True
    except Exception:
        pass


def _zero_joint_velocities(world, indices=None) -> None:
    if getattr(world, "dry_run", False):
        return
    robot = getattr(world, "robot", None)
    if robot is None:
        return
    try:
        v0 = robot.get_joint_velocities()
        v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
        if indices is None:
            v[...] = 0.0
        else:
            for j in np.asarray(indices, dtype=int).reshape(-1):
                v[int(j)] = 0.0
        robot.set_joint_velocities(v)
    except Exception:
        pass


def _sync_shortcut_pins(world, *, include_current: bool = True) -> Dict[str, Any]:
    """Refresh all hold targets from the current robot state after shortcut motion."""
    _ensure_shortcut_hard_pin(world)
    report: Dict[str, Any] = {"ok": True}
    if getattr(world, "dry_run", False):
        return report
    try:
        pose = world.robot_pose()
        _set_shortcut_base_pin(world, pose.pos, pose.yaw)
        report["base"] = [round(float(pose.pos[0]), 5), round(float(pose.pos[1]), 5), round(float(pose.yaw), 5)]
    except Exception as exc:
        report["base_error"] = str(exc)
    try:
        tq = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
        world.set_trunk_pin_qpos(tq.tolist())
        report["trunk"] = [round(float(x), 5) for x in tq.tolist()]
    except Exception as exc:
        report["ok"] = False
        report["trunk_error"] = str(exc)
    for arm in ("left", "right"):
        try:
            aq = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7)
            world.set_arm_pin_qpos(arm, aq)
            report[f"arm_{arm}"] = [round(float(x), 5) for x in aq.tolist()]
        except Exception as exc:
            report["ok"] = False
            report[f"arm_{arm}_error"] = str(exc)
        try:
            gq = world.gripper_qpos_list(arm)
            if gq is not None:
                vals = [float(x) for x in gq]
                world.set_gripper_pin_qpos(arm, vals)
                report[f"gripper_{arm}"] = [round(float(x), 5) for x in vals]
        except Exception as exc:
            report[f"gripper_{arm}_error"] = str(exc)
    _zero_joint_velocities(world)
    if include_current:
        try:
            world.hold_action()
        except Exception:
            pass
    return report


def _sync_shortcut_pins_to_targets(
    world,
    *,
    trunk_q=None,
    arm_hold: Optional[Dict[str, np.ndarray]] = None,
    gripper_hold: Optional[Dict[str, Optional[list[float]]]] = None,
    include_current: bool = True,
) -> Dict[str, Any]:
    """Write shortcut pin targets from known command targets, not drifting state."""
    _ensure_shortcut_hard_pin(world)
    report: Dict[str, Any] = {"ok": True, "source": "explicit_targets"}
    if getattr(world, "dry_run", False):
        return report
    try:
        pose = world.robot_pose()
        _set_shortcut_base_pin(world, pose.pos, pose.yaw)
        report["base"] = [round(float(pose.pos[0]), 5), round(float(pose.pos[1]), 5), round(float(pose.yaw), 5)]
    except Exception as exc:
        report["base_error"] = str(exc)
    try:
        if trunk_q is None:
            tq = np.asarray(world.trunk_pin_qpos_list(), dtype=np.float64).reshape(4)
        else:
            tq = np.asarray(trunk_q, dtype=np.float64).reshape(4)
        world.set_trunk_pin_qpos(tq.tolist())
        report["trunk"] = [round(float(x), 5) for x in tq.tolist()]
    except Exception as exc:
        report["ok"] = False
        report["trunk_error"] = str(exc)
    for arm, q in dict(arm_hold or {}).items():
        try:
            aq = np.asarray(q, dtype=np.float64).reshape(7)
            world.set_arm_pin_qpos(arm, aq)
            report[f"arm_{arm}"] = [round(float(x), 5) for x in aq.tolist()]
        except Exception as exc:
            report["ok"] = False
            report[f"arm_{arm}_error"] = str(exc)
    for arm, q in dict(gripper_hold or {}).items():
        if q is None:
            continue
        try:
            vals = [float(x) for x in q]
            world.set_gripper_pin_qpos(arm, vals)
            report[f"gripper_{arm}"] = [round(float(x), 5) for x in vals]
        except Exception as exc:
            report[f"gripper_{arm}_error"] = str(exc)
    _zero_joint_velocities(world)
    if include_current:
        try:
            world.hold_action()
        except Exception:
            pass
    return report


def _base_pose_sane(pose) -> bool:
    try:
        p = np.asarray(pose.pos, dtype=np.float64).reshape(3)
        return bool(
            np.all(np.isfinite(p[:2]))
            and abs(float(p[0])) <= _BAD_BASE_ABS_XY_M
            and abs(float(p[1])) <= _BAD_BASE_ABS_XY_M
            and np.isfinite(float(pose.yaw))
        )
    except Exception:
        return False


def _snapshot_limb_holds(world) -> tuple[Dict[str, np.ndarray], Dict[str, Optional[list[float]]]]:
    _sync_shortcut_pins(world, include_current=False)
    arm_hold: Dict[str, np.ndarray] = {}
    gripper_hold: Dict[str, Optional[list[float]]] = {}
    for arm in ("left", "right"):
        try:
            q = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7).copy()
            arm_hold[arm] = q
            world.set_arm_pin_qpos(arm, q)
        except Exception:
            pass
        gripper_hold[arm] = _gripper_keep(world, arm)
    return arm_hold, gripper_hold


def _limb_hold_kwargs(
    arm_hold: Dict[str, np.ndarray],
    gripper_hold: Dict[str, Optional[list[float]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm, q in arm_hold.items():
        out[f"arm_{arm}"] = np.asarray(q, dtype=np.float64).reshape(7).tolist()
        grip = gripper_hold.get(arm)
        if grip is not None:
            out[f"gripper_{arm}"] = list(grip)
    return out


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(float(yaw) * 0.5), math.cos(float(yaw) * 0.5)], dtype=np.float64)


def _norm_angle_rad(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _set_shortcut_base_pin(world, pos, yaw: float) -> None:
    pos_np = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    world._codex_v1_shortcut_base_pin = {
        "pos": [float(pos_np[0]), float(pos_np[1]), float(pos_np[2])],
        "yaw": float(yaw),
    }


def _force_apply_shortcut_base_pin(world) -> bool:
    pin = getattr(world, "_codex_v1_shortcut_base_pin", None)
    if not isinstance(pin, dict):
        return False
    pos = np.asarray(pin.get("pos"), dtype=np.float64).reshape(3)
    yaw = float(pin.get("yaw", 0.0))
    robot = getattr(world, "robot", None)
    if robot is None or getattr(world, "dry_run", False):
        return False
    quat_np = _yaw_quat_xyzw(yaw)
    try:
        import torch as th

        position = th.tensor(pos, dtype=th.float32)
        orientation = th.tensor(quat_np, dtype=th.float32)
    except Exception:
        position = pos
        orientation = quat_np
    try:
        robot.set_position_orientation(position=position, orientation=orientation)
        try:
            from behavior_interface.world_api import Pose

            world._last_pose = Pose(pos=pos.copy(), quat=quat_np.copy())
        except Exception:
            pass
        for setter_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(robot, setter_name, None)
            if callable(setter):
                try:
                    setter(np.zeros(3, dtype=np.float32))
                except Exception:
                    pass
        return True
    except Exception:
        return False


def _set_robot_base_pose(world, pos, yaw: float) -> bool:
    pos_np = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    quat_np = _yaw_quat_xyzw(float(yaw))
    if getattr(world, "dry_run", False):
        try:
            world._mock_base = np.array([pos_np[0], pos_np[1], float(yaw)], dtype=np.float64)
        except Exception:
            pass
        return True
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    try:
        q_saved = None
        try:
            q0 = robot.get_joint_positions()
            q_saved = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        except Exception:
            q_saved = None
        try:
            import torch as th

            position = th.tensor(pos_np, dtype=th.float32)
            orientation = th.tensor(quat_np, dtype=th.float32)
        except Exception:
            position = pos_np
            orientation = quat_np
        robot.set_position_orientation(position=position, orientation=orientation)
        if q_saved is not None:
            try:
                q_now = robot.get_joint_positions()
                q_new = q_now.clone() if hasattr(q_now, "clone") else np.asarray(q_now, dtype=np.float64).copy()
                names = list(robot.joints.keys())
                keep_indices = []
                try:
                    raw_idx = robot.trunk_control_idx
                    if hasattr(raw_idx, "detach"):
                        raw_idx = raw_idx.detach().cpu().numpy()
                    keep_indices.extend([int(i) for i in np.asarray(raw_idx, dtype=int).reshape(-1)[:4]])
                except Exception:
                    pass
                keep_names = []
                for a in ("left", "right"):
                    keep_names.extend([f"{a}_arm_joint{i + 1}" for i in range(7)])
                    keep_names.extend([
                        f"{a}_gripper_finger_joint1",
                        f"{a}_gripper_finger_joint2",
                    ])
                for name in keep_names:
                    if name in names:
                        keep_indices.append(int(names.index(name)))
                for idx in sorted(set(keep_indices)):
                    q_new[int(idx)] = q_saved[int(idx)]
                robot.set_joint_positions(q_new)
            except Exception:
                pass
        try:
            from behavior_interface.world_api import Pose

            world._last_pose = Pose(pos=pos_np.copy(), quat=quat_np.copy())
        except Exception:
            pass
        _set_shortcut_base_pin(world, pos_np, float(yaw))
        for setter_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(robot, setter_name, None)
            if callable(setter):
                try:
                    setter(np.zeros(3, dtype=np.float32))
                except Exception:
                    pass
        _zero_joint_velocities(world)
        try:
            pos_check, quat_check = robot.get_position_orientation()
            pos_arr = np.asarray(
                pos_check.detach().cpu().numpy() if hasattr(pos_check, "detach") else pos_check,
                dtype=np.float64,
            ).reshape(3)
            quat_arr = np.asarray(
                quat_check.detach().cpu().numpy() if hasattr(quat_check, "detach") else quat_check,
                dtype=np.float64,
            ).reshape(4)
            qn = float(np.linalg.norm(quat_arr))
            if (
                not np.all(np.isfinite(pos_arr))
                or not np.all(np.isfinite(quat_arr))
                or qn < 0.5
                or qn > 1.5
                or abs(float(pos_arr[0])) > _BAD_BASE_ABS_XY_M
                or abs(float(pos_arr[1])) > _BAD_BASE_ABS_XY_M
            ):
                return False
        except Exception:
            return False
        return True
    except Exception:
        return False


def _shortcut_upward_target(ctx, upward: float, z_tol: float) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    world = ctx.world
    q0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    chest0 = world.chest_pose()
    z_curr = float(chest0["z"])
    z_tgt_raw = z_curr + float(upward)
    report: Dict[str, Any] = {
        "ok": True,
        "mode": "direct_reverse_upward_lut_final_q",
        "z_start_m": round(z_curr, 4),
        "z_target_raw_m": round(z_tgt_raw, 4),
        "upward_delta_cmd_m": round(float(upward), 4),
        "q_start": [round(float(x), 4) for x in q0.tolist()],
    }
    if abs(float(upward)) <= max(0.004, float(z_tol) * 0.25):
        report["skipped"] = True
        report["q_target"] = [round(float(x), 4) for x in q0.tolist()]
        return q0, report
    try:
        from behavior_interface.skills.move_to import _base_link_z_world, _relative_reverse_upward_waypoints
        from behavior_interface.trunk_vertical_lift import (
            get_reverse_upward_combined_lut,
            plan_reverse_upward_trajectory_from_upward,
        )

        base_z = _base_link_z_world(world)
        lut = get_reverse_upward_combined_lut(
            base_z,
            q4=float(q0[3]),
            theta_z_deg=90.0,
        )
        if not lut.get("ok"):
            report["ok"] = False
            report["error"] = lut.get("error", "反向 upward 查表失败")
            return None, report
        upward_for_planner = z_tgt_raw - float(lut["z_upright_m"])
        waypoints_abs, vmeta = plan_reverse_upward_trajectory_from_upward(
            z_curr,
            upward_for_planner,
            base_z,
            q4=float(q0[3]),
            theta_z_deg=90.0,
        )
        report["reverse_upward"] = vmeta
        if vmeta.get("direction") == "hold" and not waypoints_abs:
            report["skipped"] = True
            report["reason"] = vmeta.get("note", "hold")
            report["q_target"] = [round(float(x), 4) for x in q0.tolist()]
            return q0, report
        if not vmeta.get("ok") or not waypoints_abs:
            report["ok"] = False
            report["error"] = vmeta.get("error", "反向 upward 规划失败")
            return None, report
        waypoints, rel_meta = _relative_reverse_upward_waypoints(world, waypoints_abs, vmeta)
        report["relative_lut"] = rel_meta
        if not waypoints:
            report["ok"] = False
            report["error"] = "反向 upward 没有可用路点"
            return None, report
        q_target = np.asarray(waypoints[-1], dtype=np.float64).reshape(4)
        report["q_target"] = [round(float(x), 4) for x in q_target.tolist()]
        report["n_waypoints_collapsed"] = len(waypoints)
        return q_target, report
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"shortcut upward final q failed: {exc}"
        return None, report


def _play_arm_q(
    ctx,
    arm: str,
    q_target,
    *,
    gripper_mode: str = "keep",
    max_dq_per_frame: float = 0.16,
    hold_frames: int = 12,
    label: str = "shortcut",
) -> Generator[Any, None, Dict[str, Any]]:
    world = ctx.world
    q1 = _as_vec(q_target, 7)
    q0 = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7)
    gap = float(np.linalg.norm(q1 - q0, ord=np.inf))
    n_frames = max(2, int(math.ceil(gap / max(1e-6, float(max_dq_per_frame)))) + 2)
    n_frames = min(80, n_frames)
    grip = _gripper_cmd_for_mode(world, arm, gripper_mode)
    ctx.log(
        f"[v1_shortcut] {label} arm={arm} direct joint play "
        f"gap={gap:.3f}rad frames={n_frames} gripper={gripper_mode}"
    )
    for fi in range(1, n_frames + 1):
        s = fi / n_frames
        ss = s * s * (3.0 - 2.0 * s)
        q = q0 + (q1 - q0) * float(ss)
        _force_open_if_needed(world, arm, gripper_mode)
        action = {f"arm_{arm}": q.tolist()}
        if grip is not None:
            action[f"gripper_{arm}"] = grip
        yield world.make_action(**action)
    for _ in range(max(1, int(hold_frames))):
        _force_open_if_needed(world, arm, gripper_mode)
        action = {f"arm_{arm}": q1.tolist()}
        if grip is not None:
            action[f"gripper_{arm}"] = grip
        yield world.make_action(**action)
    try:
        world.set_arm_pin_qpos(arm, q1)
        if grip is not None:
            world.set_gripper_pin_qpos(arm, grip)
    except Exception:
        pass
    sync_report = _sync_shortcut_pins_to_targets(
        world,
        arm_hold={arm: q1},
        gripper_hold={arm: grip} if grip is not None else None,
    )
    q_now = np.asarray(world.arm_qpos_list(arm), dtype=np.float64).reshape(7)
    return {
        "frames": int(n_frames + max(1, int(hold_frames))),
        "q_gap_start_rad": gap,
        "q_track_err_rad": float(np.linalg.norm(q_now - q1, ord=np.inf)),
        "target_qpos": [round(float(x), 6) for x in q1.tolist()],
        "final_pin_sync": sync_report,
    }


def _solve_final_q(ctx, arm: str, target_pos, target_quat, *, seed_q=None):
    from behavior_interface.skills.eef import _eef_solve_6d_dls_arm_q

    return _eef_solve_6d_dls_arm_q(
        ctx.world,
        arm,
        target_pos,
        target_quat,
        pos_tol=0.010,
        ori_tol_deg=5.0,
        max_steps=360,
        max_dq_per_step=0.070,
        max_dx_per_step=0.024,
        max_dw_per_step=0.10,
        ori_weight=0.70,
        lam=0.08,
        seed_q=seed_q,
    )


def _stored_q(cand: Dict[str, Any], arm: str):
    try:
        from behavior_interface.skills.eef import _stored_filter_q_for_arm

        return _stored_filter_q_for_arm(cand, arm)
    except Exception:
        return None


def _resolve_exec_arm(world, record: Dict[str, Any], cand: Dict[str, Any], arm: Optional[str]) -> str:
    a = str(arm or "").strip().lower()
    if a in ("left", "right"):
        return a
    sol = str(cand.get("ik_solution") or "").strip().lower()
    if sol in ("left", "right"):
        return sol
    rec = str(record.get("arm") or cand.get("arm") or "").strip().lower()
    return rec if rec in ("left", "right") else "right"


def _candidate_next_move(record: Dict[str, Any], cand: Dict[str, Any], target_pos: np.ndarray) -> np.ndarray:
    """Resolve the post-grasp Cartesian lift in world coordinates."""
    for value in (
        cand.get("next_eef_move"),
        (cand.get("meta") or {}).get("next_eef_move"),
    ):
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(arr)) > 1e-9:
                return arr
        except Exception:
            pass
    for value in (
        record.get("next_move_world"),
        (cand.get("meta") or {}).get("next_eef_move_world"),
    ):
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(3) - np.asarray(target_pos, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(arr)) > 1e-9:
                return arr
        except Exception:
            pass
    return np.array([0.0, 0.0, 0.10], dtype=np.float64)


def _shortcut_object_snapshot(world, object_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"object_name": object_name or "", "object": None, "z": None, "pos": None}
    if not object_name:
        return out
    try:
        from behavior_interface.skills.eef import _read_obj_z, _resolve_object_handle, _to_np

        obj = _resolve_object_handle(world, object_name)
        out["object"] = obj
        out["z"] = _read_obj_z(obj)
        if obj is not None:
            try:
                p, _ = obj.get_position_orientation()
                out["pos"] = _to_np(p)
            except Exception:
                out["pos"] = None
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _exec_plan_shortcut(
    ctx,
    *,
    session_id: str,
    plan_id: str,
    arm: Optional[str],
    close_and_lift: bool,
    reset_tool_roll_at_start: bool = False,
) -> Generator:
    world = ctx.world
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions
        from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims

        _ensure_world_pinned_actions(world)
        clear_plan_viz_prims()
    except Exception:
        pass
    try:
        record = agent_runs.load_plan_record(session_id, plan_id)
    except FileNotFoundError as exc:
        ctx.set_result({"ok": False, "error": str(exc), "plan_id": plan_id})
        yield world.hold_action()
        return
    cand: Dict[str, Any] = record.get("candidate") or {}
    et = cand.get("eef_target") or {}
    if not et.get("pos") or not et.get("quat"):
        ctx.set_result({"ok": False, "error": f"plan {plan_id} 无 eef_target.pos/quat"})
        yield world.hold_action()
        return
    arm_eff = _resolve_exec_arm(world, record, cand, arm)
    tool_roll_reset = None
    if bool(reset_tool_roll_at_start):
        try:
            from behavior_interface.skills.eef import (
                _reset_selected_tool_roll_to_zero,
            )

            tool_roll_reset = _reset_selected_tool_roll_to_zero(
                world,
                arm_eff,
                ctx=ctx,
                stage_name="exec_plan_pose.start",
            )
        except Exception as exc:
            ctx.set_result({
                "ok": False,
                "tool": "exec_eef_pose",
                "plan_id": plan_id,
                "arm": arm_eff,
                "error": f"exec_plan_pose J8 reset failed: {exc}",
            })
            yield world.hold_action()
            return
    target_pos = _as_vec(et["pos"], 3)
    target_quat = _as_vec(et["quat"], 4)
    target = str(cand.get("target") or record.get("target") or "").strip().lower()
    object_name = (
        record.get("object_name")
        or (cand.get("meta") or {}).get("object_name")
        or ""
    )
    obj0 = _shortcut_object_snapshot(world, str(object_name)) if close_and_lift else {}
    q = _stored_q(cand, arm_eff)
    q_source = "stored_filter_q"
    if q is None:
        q_source = "offline_dls_final_pose"
        q, solve_pos_err, solve_ori_err = _solve_final_q(ctx, arm_eff, target_pos, target_quat)
    else:
        solve_pos_err, solve_ori_err = 0.0, 0.0
    if q is None:
        ctx.set_result({
            "ok": False,
            "tool": "exec_move" if close_and_lift else "exec_eef_pose",
            "plan_id": plan_id,
            "arm": arm_eff,
            "error": "shortcut_final_pose_ik_failed",
            "solve_pos_err_m": float(solve_pos_err),
            "solve_ori_err_deg": float(solve_ori_err),
        })
        yield world.hold_action()
        return
    play = yield from _play_arm_q(
        ctx,
        arm_eff,
        q,
        gripper_mode="keep",
        max_dq_per_frame=0.16,
        hold_frames=18,
        label=f"exec {'move' if close_and_lift else 'eef_pose'} {plan_id}",
    )
    eef = world.eef_pose(arm=arm_eff)
    pos_now = _as_vec(eef["pos"], 3)
    quat_now = _as_vec(eef["quat"], 4)
    pos_err = float(np.linalg.norm(pos_now - target_pos))
    ori_err = _quat_err_deg(quat_now, target_quat)
    result = {
        "ok": bool(pos_err <= 0.010 and ori_err <= 5.0),
        "build": _SHORTCUT_BUILD,
        "execution": "direct_joint_target",
        "q_source": q_source,
        "arm": arm_eff,
        "plan_id": plan_id,
        "final_pos_err_m": pos_err,
        "final_ori_err_deg": ori_err,
        "eef_target_pos": target_pos.tolist(),
        "eef_actual_pos": pos_now.tolist(),
        "play": play,
    }
    if tool_roll_reset is not None:
        result["tool_roll_reset"] = tool_roll_reset
    if close_and_lift:
        grip = _gripper_cmd_for_mode(world, arm_eff, "close")
        for _ in range(18):
            action = {f"arm_{arm_eff}": _as_vec(q, 7).tolist()}
            if grip is not None:
                action[f"gripper_{arm_eff}"] = grip
            yield world.make_action(**action)
        result["final_pin_sync_after_close"] = _sync_shortcut_pins_to_targets(
            world,
            arm_hold={arm_eff: _as_vec(q, 7)},
            gripper_hold={arm_eff: grip} if grip is not None else None,
        )
        result["closed_gripper"] = True
        next_move = _candidate_next_move(record, cand, target_pos)
        lift_target_pos = target_pos + next_move
        result["next_eef_move"] = [float(x) for x in next_move.tolist()]
        result["lift_target_pos"] = [float(x) for x in lift_target_pos.tolist()]
        if target == "grasp":
            q_lift, lift_solve_pos_err, lift_solve_ori_err = _solve_final_q(
                ctx, arm_eff, lift_target_pos, target_quat, seed_q=_as_vec(q, 7)
            )
            result["lift_solve_pos_err_m"] = float(lift_solve_pos_err)
            result["lift_solve_ori_err_deg"] = float(lift_solve_ori_err)
            if q_lift is None:
                result["lift_ok"] = False
                result["lift_error"] = "shortcut_lift_pose_ik_failed"
            else:
                lift_play = yield from _play_arm_q(
                    ctx,
                    arm_eff,
                    q_lift,
                    gripper_mode="close",
                    max_dq_per_frame=0.08,
                    hold_frames=24,
                    label=f"exec move {plan_id} lift",
                )
                eef_lift = world.eef_pose(arm=arm_eff)
                pos_lift_now = _as_vec(eef_lift["pos"], 3)
                quat_lift_now = _as_vec(eef_lift["quat"], 4)
                result["lift_play"] = lift_play
                result["lift_final_pos_err_m"] = float(np.linalg.norm(pos_lift_now - lift_target_pos))
                result["lift_final_ori_err_deg"] = _quat_err_deg(quat_lift_now, target_quat)
                result["lift_actual_pos"] = [float(x) for x in pos_lift_now.tolist()]
                result["lift_ok"] = bool(result["lift_final_pos_err_m"] <= 0.05)
        else:
            result["lift_ok"] = False
            result["lift_skipped"] = "non_grasp_target"

        obj1 = _shortcut_object_snapshot(world, str(object_name))
        obj_z0 = obj0.get("z")
        obj_z1 = obj1.get("z")
        dz = (float(obj_z1) - float(obj_z0)) if obj_z0 is not None and obj_z1 is not None else None
        obj_pos0 = obj0.get("pos")
        obj_pos1 = obj1.get("pos")
        obj_displ = None
        obj_moved_dist = None
        if obj_pos0 is not None and obj_pos1 is not None:
            displ = np.asarray(obj_pos1, dtype=np.float64) - np.asarray(obj_pos0, dtype=np.float64)
            obj_displ = [float(x) for x in displ.tolist()]
            obj_moved_dist = float(np.linalg.norm(displ))
        holding = {}
        try:
            holding = dict(world.holding_summary())
        except Exception:
            try:
                holding = {"left": world.held_object_name("left"), "right": world.held_object_name("right")}
            except Exception:
                holding = {}
        held = holding.get(arm_eff) if isinstance(holding, dict) else None
        grasped = bool((dz is not None and dz > 0.04) or (held and str(held) == str(object_name)))
        result.update({
            "object_name": str(object_name),
            "object_z_before": obj_z0,
            "object_z_after": obj_z1,
            "object_dz": dz,
            "object_pos_before": [float(x) for x in np.asarray(obj_pos0).reshape(3).tolist()] if obj_pos0 is not None else None,
            "object_pos_after": [float(x) for x in np.asarray(obj_pos1).reshape(3).tolist()] if obj_pos1 is not None else None,
            "object_displacement": obj_displ,
            "object_moved_dist": obj_moved_dist,
            "holding": holding,
            "grasped": grasped,
            "note": "v1_shortcut closes gripper then lifts; grasp success requires object dz > 4cm or holding.",
        })
        result["ok"] = bool(grasped)
        if not grasped:
            dz_mm = "?" if dz is None else f"{float(dz) * 1000:.1f}"
            result["error"] = f"抓取失败：{object_name or 'object'} Δz={dz_mm}mm (需 >40mm)"
        else:
            result.pop("error", None)
    ok_out = bool(result["ok"])
    err_out = result.get("error")
    if close_and_lift and not ok_out and not err_out:
        err_out = "shortcut grasp failed"
    elif not close_and_lift and not ok_out:
        err_out = "shortcut target EEF not reached"
    ctx.set_result({
        "ok": ok_out,
        "tool": "exec_move" if close_and_lift else "exec_eef_pose",
        "plan_id": plan_id,
        "arm": arm_eff,
        "skill": record.get("skill") or record.get("mode"),
        "result": result,
        "error": None if ok_out else err_out,
    })
    yield world.hold_action()


@register_skill("exec_eef_pose_v2", description="v1_shortcut: direct joint target to plan eef_pose.")
def exec_eef_pose_v2(
    ctx,
    session_id: str,
    plan_id: str,
    arm: Optional[str] = None,
    back_m: float = 0.10,
    stop_after_safe: bool = False,
    reset_tool_roll_at_start: bool = False,
):
    yield from _exec_plan_shortcut(
        ctx,
        session_id=session_id,
        plan_id=plan_id,
        arm=arm,
        close_and_lift=False,
        reset_tool_roll_at_start=reset_tool_roll_at_start,
    )


@register_skill("exec_move_v2", description="v1_shortcut: normal safe/contact/close/lift execution.")
def exec_move_v2(ctx, session_id: str, plan_id: str, arm: Optional[str] = None, back_m: float = 0.10):
    from behavior_interface.skills.exec_move_v2 import exec_move_v2 as _exec_move_v2

    yield from _exec_move_v2(
        ctx,
        session_id=session_id,
        plan_id=plan_id,
        arm=arm,
        back_m=back_m,
    )


@register_skill("capture", description="v1_shortcut: same capture implementation, routed through the shortcut profile.")
def capture(ctx, **kwargs):
    from behavior_interface.skills.capture import capture as _capture

    yield from _capture(ctx, **kwargs)


@register_skill(
    "plan_move_eef",
    description="v1_shortcut: same plan_move_eef implementation, including optional u/v/depth target.",
)
def plan_move_eef(
    ctx,
    session_id: str,
    upward: float = 0.0,
    forward: float = 0.0,
    leftward: float = 0.0,
    u: Optional[float] = None,
    v: Optional[float] = None,
    depth: Optional[float] = None,
    gripper: str = "keep",
    arm: str = "right",
    pos_tol: float = 0.012,
):
    from behavior_interface.skills.plan_move_eef import plan_move_eef as _plan_move_eef

    yield from _plan_move_eef(
        ctx,
        session_id=session_id,
        upward=upward,
        forward=forward,
        leftward=leftward,
        u=u,
        v=v,
        depth=depth,
        gripper=gripper,
        arm=arm,
        pos_tol=pos_tol,
    )


@register_skill("adjust_plan_pose", description="v1_shortcut: same adjust_plan_pose implementation.")
def adjust_plan_pose(ctx, **kwargs):
    from behavior_interface.skills.adjust_plan_pose import adjust_plan_pose as _adjust_plan_pose

    yield from _adjust_plan_pose(ctx, **kwargs)


@register_skill("plan_eef_v2", description="v1_shortcut: same plan_eef_v2 implementation.")
def plan_eef_v2(ctx, **kwargs):
    from behavior_interface.skills.plan_eef_v2 import plan_eef_v2 as _plan_eef_v2

    yield from _plan_eef_v2(ctx, **kwargs)


@register_skill("move_in_robot_coord", description="v1_shortcut: direct final base/trunk target, with arms and grippers locked.")
def move_in_robot_coord(
    ctx,
    forward: float = 0.0,
    spin: float = 0.0,
    pitch: float = 0.0,
    upward: float = 0.0,
    vmax: float = 0.5,
    wmax: float = 1.0,
    k_ang: float = 2.0,
    yaw_tol_deg: float = 3.0,
    z_tol: float = 0.05,
    theta_z_tol_deg: float = 5.0,
    trunk_max_step_rad: float = 0.10,
    trunk_timeout_s: float = 30.0,
    timeout_s: float = 120.0,
    nav_guard: bool = True,
    extra_inflate: float = 0.04,
):
    world = ctx.world
    _sync_shortcut_pins(world, include_current=False)
    arm_hold, gripper_hold = _snapshot_limb_holds(world)
    pose0 = world.robot_pose()
    if not _base_pose_sane(pose0):
        ctx.set_result({
            "ok": False,
            "tool": "move_in_robot_coord",
            "build": _SHORTCUT_BUILD,
            "error": "shortcut base pose is invalid; restart world before moving",
            "robot_pose_start": pose0.as_dict(),
        })
        yield world.hold_action()
        return
    trunk0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    report: Dict[str, Any] = {
        "ok": True,
        "tool": "move_in_robot_coord",
        "build": _SHORTCUT_BUILD,
        "execution": "direct_base_and_trunk_targets",
        "forward": float(forward),
        "spin": float(spin),
        "pitch": float(pitch),
        "upward": float(upward),
        "robot_pose_start": pose0.as_dict(),
        "trunk_q_start": [round(float(x), 5) for x in trunk0.tolist()],
        "arms_locked": list(arm_hold.keys()),
        "nav_guard_ignored": bool(nav_guard),
    }

    q_target = trunk0.copy()
    if abs(float(upward)) > 1e-9:
        q_up, up_report = _shortcut_upward_target(ctx, float(upward), float(z_tol))
        report["upward_exec"] = up_report
        if q_up is None or not up_report.get("ok", False):
            report["ok"] = False
            report["error"] = up_report.get("error", "shortcut upward target failed")
            ctx.set_result(report)
            yield world.hold_action()
            return
        q_target = np.asarray(q_up, dtype=np.float64).reshape(4)

    if abs(float(pitch)) > 1e-9:
        try:
            from behavior_interface.skills.move_to import _feasible_trunk_q3_delta

            ok_pitch, why, q3_tgt = _feasible_trunk_q3_delta(q_target, float(pitch))
        except Exception as exc:
            ok_pitch, why, q3_tgt = False, str(exc), q_target[2]
        report["pitch_q3"] = {
            "mode": "direct_q3_delta",
            "pitch_deg": round(float(pitch), 3),
            "q3_start_rad": round(float(q_target[2]), 5),
            "q3_target_rad": round(float(q3_tgt), 5),
        }
        if not ok_pitch:
            report["ok"] = False
            report["error"] = why
            ctx.set_result(report)
            yield world.hold_action()
            return
        q_target[2] = float(q3_tgt)

    ctx.log(
        f"[v1_shortcut] move_in_robot_coord direct trunk "
        f"[{','.join(f'{v:+.3f}' for v in trunk0)}] -> "
        f"[{','.join(f'{v:+.3f}' for v in q_target)}]"
    )
    _force_set_trunk_qpos(world, q_target)

    yaw0 = float(pose0.yaw)
    yaw1 = _norm_angle_rad(yaw0 + math.radians(float(spin)))
    pos1 = np.asarray(pose0.pos, dtype=np.float64).reshape(3).copy()
    if abs(float(forward)) > 1e-9:
        pos1[0] += float(forward) * math.cos(yaw0)
        pos1[1] += float(forward) * math.sin(yaw0)
    base_set = True
    if abs(float(forward)) > 1e-9 or abs(float(spin)) > 1e-9:
        base_set = _set_robot_base_pose(world, pos1, yaw1)
        if not base_set:
            report["ok"] = False
            report["error"] = "shortcut base pose set failed"

    for _ in range(10):
        action: Dict[str, Any] = {"trunk": q_target.tolist()}
        action.update(_limb_hold_kwargs(arm_hold, gripper_hold))
        yield world.make_action(**action)
    _force_set_trunk_qpos(world, q_target)
    final_pin_sync = _sync_shortcut_pins_to_targets(
        world,
        trunk_q=q_target,
        arm_hold=arm_hold,
        gripper_hold=gripper_hold,
    )
    pose1 = world.robot_pose()
    trunk1 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    trunk_err = float(np.linalg.norm(trunk1 - q_target, ord=np.inf))
    xy_err = float(np.linalg.norm(np.asarray(pose1.pos[:2]) - pos1[:2]))
    yaw_err = abs(_norm_angle_rad(float(pose1.yaw) - yaw1))
    report.update({
        "robot_pose_target": {
            "pos": [round(float(x), 5) for x in pos1.tolist()],
            "quat": [round(float(x), 6) for x in _yaw_quat_xyzw(yaw1).tolist()],
            "yaw_deg": round(math.degrees(yaw1), 3),
        },
        "robot_pose_end": pose1.as_dict(),
        "trunk_q_target": [round(float(x), 5) for x in q_target.tolist()],
        "trunk_q_end": [round(float(x), 5) for x in trunk1.tolist()],
        "trunk_err_inf_rad": round(trunk_err, 5),
        "base_xy_err_m": round(xy_err, 5),
        "base_yaw_err_deg": round(math.degrees(yaw_err), 4),
        "base_pose_set": bool(base_set),
        "final_pin_sync": final_pin_sync,
    })
    if report.get("ok", True):
        report["ok"] = bool(trunk_err <= 0.030 and xy_err <= 0.030 and math.degrees(yaw_err) <= 3.0)
        report["error"] = None if report["ok"] else "shortcut move_in_robot_coord target not reached"
    ctx.set_result(report)
    yield world.hold_action()


def _move_to_fail(ctx, *, session_id: str, tool: str, msg: str, C=None, lo=None, hi=None, extra=None):
    world = ctx.world
    extra = dict(extra or {})
    ctx.log(f"[v1_shortcut] {tool} 中止: {msg}")
    head_live = {}
    try:
        from behavior_interface.skills.move_to_object_v2 import _capture_head_snapshot

        head_live = _capture_head_snapshot(
            world,
            session_id=session_id,
            head_png_tag=f"{tool}_head",
            log_tag=f"{tool}_shortcut",
            ctx=ctx,
            lo=lo,
            hi=hi,
            suffix="failure",
        )
    except Exception:
        head_live = {"ok": False, "reason": "capture failed"}
    result = {
        "ok": False,
        "tool": tool,
        "build": _SHORTCUT_BUILD,
        "execution": "direct_base_and_trunk_targets",
        "error": msg,
        "visibility": head_live,
        "head_after_move_png": head_live.get("head_png"),
        **extra,
    }
    if C is not None:
        result["target_center_world"] = [float(x) for x in np.asarray(C, dtype=np.float64).reshape(3)]
    ctx.set_result(result)
    yield world.hold_action()


def _shortcut_pre_lift_for_target(ctx, C: np.ndarray, log_tag: str) -> Dict[str, Any]:
    from behavior_interface.trunk_vertical_lift import POINT_PRE_DESCENT_DZ_MAX_M
    from behavior_interface.skills.move_to_object_geom import read_shoulder_head_pose

    world = ctx.world
    arm_hold, gripper_hold = _snapshot_limb_holds(world)
    target_z = float(np.asarray(C, dtype=np.float64).reshape(3)[2])
    pose0 = read_shoulder_head_pose(world)
    shoulder_z0 = float(pose0["shoulder_mid"][2])
    chest0 = world.chest_pose()
    dz0 = shoulder_z0 - target_z
    out: Dict[str, Any] = {
        "ok": True,
        "mode": "direct_final_relative_lut_q",
        "dz_start_m": round(float(dz0), 4),
        "shoulder_z_start_m": round(float(shoulder_z0), 4),
        "target_z_m": round(float(target_z), 4),
    }
    stage_t0 = time.time()
    if dz0 <= POINT_PRE_DESCENT_DZ_MAX_M + 1e-3:
        out["skipped"] = True
        out["elapsed_s"] = round(time.time() - stage_t0, 3)
        return out
    desired_drop = max(0.0, float(dz0) - float(POINT_PRE_DESCENT_DZ_MAX_M))
    ctx.log(
        f"[v1_shortcut] {log_tag} direct pre-lift drop={desired_drop:.3f}m "
        f"dz {dz0:.3f}-><={POINT_PRE_DESCENT_DZ_MAX_M:.2f}m"
    )
    q_up, up_report = _shortcut_upward_target(ctx, -desired_drop, 0.04)
    out["relative_lut_target"] = up_report
    if q_up is None or not up_report.get("ok", False):
        out["ok"] = False
        out["error"] = up_report.get("error", "direct pre-lift target failed")
        out["elapsed_s"] = round(time.time() - stage_t0, 3)
        return out
    _force_set_trunk_qpos(world, q_up)
    action: Dict[str, Any] = {"trunk": np.asarray(q_up, dtype=np.float64).reshape(4).tolist()}
    action.update(_limb_hold_kwargs(arm_hold, gripper_hold))
    yield world.make_action(**action)
    _force_apply_shortcut_base_pin(world)
    _force_set_trunk_qpos(world, q_up)
    out["pin_sync_after_lift"] = _sync_shortcut_pins_to_targets(
        world,
        trunk_q=q_up,
        arm_hold=arm_hold,
        gripper_hold=gripper_hold,
    )
    pose1 = read_shoulder_head_pose(world)
    chest1 = world.chest_pose()
    out.update({
        "dz_end_m": round(float(np.asarray(pose1["shoulder_mid"])[2]) - target_z, 4),
        "chest_z_start_m": round(float(chest0["z"]), 4),
        "chest_z_end_m": round(float(chest1["z"]), 4),
        "theta_z_end_deg": round(float(chest1["theta_z_deg"]), 2),
        "elapsed_s": round(time.time() - stage_t0, 3),
    })
    ctx.log(f"[v1_shortcut] {log_tag} direct pre-lift done elapsed={out['elapsed_s']:.3f}s")
    return out


def _move_to_center_shortcut(
    ctx,
    *,
    session_id: str,
    C,
    lo,
    hi,
    reach: float = 0.0,
    tool: str,
    log_tag: str,
    result_extra: Optional[Dict[str, Any]] = None,
):
    world = ctx.world
    C = np.asarray(C, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    extra = dict(result_extra or {})
    _sync_shortcut_pins(world, include_current=False)
    arm_hold, gripper_hold = _snapshot_limb_holds(world)
    try:
        from behavior_interface.skills.move_to import _base_link_z_world
        from behavior_interface.skills.move_to_object_v2 import (
            _BASE_TARGET_BACKOFF_EXEC_GUARD_M,
            _BASE_TARGET_BACKOFF_MARGIN_M,
            _BASE_TARGET_BACKOFF_MAX_M,
            _BASE_TARGET_BACKOFF_STEP_M,
            _BASE_TARGET_MIN_CLEARANCE_M,
            _capture_head_snapshot,
            _ensure_plan_scene_graph,
            _head_view_live,
            _is_free_xy,
            _nav_clearance_xy,
            _pose_errors,
            _pose_within_tol,
            _reach_report_live,
            _save_nav_reach,
            _shoulder_distance_result,
            adaptive_chord_reach_m,
        )
        from behavior_interface.skills.base_chord_reach import (
            REACH_SPHERE_R_M,
            reach_sphere_radius,
        )
        from behavior_interface.skills.base_footprint_distance import measure_base_distance
        from behavior_interface.skills.move_to_object_geom import (
            plan_geom_from_pose,
            read_shoulder_head_pose,
        )
        from behavior_interface.trunk_vertical_lift import solve_q3_for_theta_z_holding_q12
    except Exception as exc:
        yield from _move_to_fail(
            ctx,
            session_id=session_id,
            tool=tool,
            msg=f"shortcut imports failed: {exc}",
            C=C,
            lo=lo,
            hi=hi,
            extra=extra,
        )
        return

    pose_start = world.robot_pose()
    if not _base_pose_sane(pose_start):
        yield from _move_to_fail(
            ctx,
            session_id=session_id,
            tool=tool,
            msg="shortcut base pose is invalid; restart world before move_to",
            C=C,
            lo=lo,
            hi=hi,
            extra={**extra, "robot_pose_start": pose_start.as_dict()},
        )
        return
    robot_xy = np.array([float(pose_start.pos[0]), float(pose_start.pos[1])], dtype=np.float64)
    ctx.log(
        f"[v1_shortcut] {log_tag} direct move_to center={C.round(3).tolist()} "
        f"start_xy={robot_xy.round(3).tolist()}"
    )

    lift_report = yield from _shortcut_pre_lift_for_target(ctx, C, log_tag)
    if not lift_report.get("ok", True):
        yield from _move_to_fail(
            ctx,
            session_id=session_id,
            tool=tool,
            msg=lift_report.get("error", "direct pre-lift failed"),
            C=C,
            lo=lo,
            hi=hi,
            extra={**extra, "lift_report": lift_report},
        )
        return
    arm_hold, gripper_hold = _snapshot_limb_holds(world)

    _ensure_plan_scene_graph(world, ctx, f"{log_tag}_shortcut")
    pose_after_lift = read_shoulder_head_pose(world)
    reach_m = float(reach) if float(reach) > 0.01 else None
    reach_plan = reach_m
    geom = plan_geom_from_pose(
        pose_after_lift,
        C,
        robot_xy,
        is_free_xy=lambda x, y: _is_free_xy(world, x, y),
        nav_clearance_fn=lambda x, y: _nav_clearance_xy(world, x, y),
        reach=reach_plan,
        base_link_z=_base_link_z_world(world),
    )
    if not geom.get("ok") and reach_m is None:
        sh_z_final = geom.get("shoulder_z")
        if sh_z_final is None:
            sh_z_final = (geom.get("base_geom") or {}).get("shoulder_z_slice_m")
        if sh_z_final is None:
            sh_z_final = float(pose_after_lift["shoulder_mid"][2])
        reach_adapt = adaptive_chord_reach_m(float(sh_z_final) - float(C[2]), None)
        if reach_adapt > REACH_SPHERE_R_M + 1e-3:
            reach_plan = reach_adapt
            geom = plan_geom_from_pose(
                pose_after_lift,
                C,
                robot_xy,
                is_free_xy=lambda x, y: _is_free_xy(world, x, y),
                nav_clearance_fn=lambda x, y: _nav_clearance_xy(world, x, y),
                reach=reach_plan,
                base_link_z=_base_link_z_world(world),
            )
    if not geom.get("ok"):
        yield from _move_to_fail(
            ctx,
            session_id=session_id,
            tool=tool,
            msg=geom.get("error", "几何规划失败"),
            C=C,
            lo=lo,
            hi=hi,
            extra={**extra, "lift_report": lift_report, "geom": geom},
        )
        return

    plan = geom["base_plan"]
    detail = plan.get("plan_detail") or {}
    bx, by = [float(x) for x in plan["base_xy"]]
    theta_x = float(geom["theta_x_deg"])
    theta_z = float(geom["theta_z_deg"])
    chest_z = float(geom["chest_z"])
    trunk_meta = dict(geom["trunk_meta"])

    def _measure_base_target_at(x: float, y: float):
        m = measure_base_distance(
            base_pose={"x": float(x), "y": float(y), "yaw_deg": float(theta_x)},
            target_xy=C[:2],
            target_aabb_min=lo,
            target_aabb_max=hi,
        )
        clearance = float(m.get("distance_aabb_m", m.get("distance_point_m", float("inf"))))
        return m, clearance

    base_target_clearance, base_target_clearance_m = _measure_base_target_at(bx, by)
    base_backoff = {
        "applied": False,
        "min_clearance_m": _BASE_TARGET_MIN_CLEARANCE_M,
        "initial_clearance_m": base_target_clearance_m,
    }
    if base_target_clearance_m <= _BASE_TARGET_MIN_CLEARANCE_M:
        bx0, by0 = bx, by
        yaw_rad = math.radians(theta_x)
        back_dir = np.array([-math.cos(yaw_rad), -math.sin(yaw_rad)], dtype=np.float64)
        target_clearance = (
            _BASE_TARGET_MIN_CLEARANCE_M
            + _BASE_TARGET_BACKOFF_MARGIN_M
            + _BASE_TARGET_BACKOFF_EXEC_GUARD_M
        )
        best = None
        s = _BASE_TARGET_BACKOFF_STEP_M
        while s <= _BASE_TARGET_BACKOFF_MAX_M + 1e-9:
            cand = np.array([bx0, by0], dtype=np.float64) + back_dir * s
            cand_measure, cand_clear = _measure_base_target_at(float(cand[0]), float(cand[1]))
            if cand_clear > target_clearance:
                best = (s, cand, cand_measure, cand_clear)
                break
            s += _BASE_TARGET_BACKOFF_STEP_M
        if best is not None:
            shift_m, final_xy, base_target_clearance, base_target_clearance_m = best
            dx, dy = float(final_xy[0] - bx0), float(final_xy[1] - by0)
            bx, by = float(final_xy[0]), float(final_xy[1])
            plan["base_xy"] = [bx, by]
            detail["bx"] = bx
            detail["by"] = by
            try:
                from behavior_interface.skills.move_to_object_v2 import _translate_plan_detail_xy

                _translate_plan_detail_xy(detail, dx, dy)
            except Exception:
                pass
            base_backoff.update({
                "applied": True,
                "shift_m": float(shift_m),
                "base_xy_before": [bx0, by0],
                "base_xy_after": [bx, by],
                "final_clearance_m": base_target_clearance_m,
            })

    q_now = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    target_trunk_q = trunk_meta.get("target_trunk_q")
    if target_trunk_q is not None:
        q_target = np.asarray(target_trunk_q, dtype=np.float64).reshape(4)
        pitch_mode = "direct_target_trunk_q"
    else:
        q3 = solve_q3_for_theta_z_holding_q12(
            theta_z,
            float(q_now[0]),
            float(q_now[1]),
            prefer_q3=float(q_now[2]),
        )
        q_target = q_now.copy()
        if q3 is not None:
            q_target[2] = float(q3)
        pitch_mode = "direct_q3_for_theta_z"

    motion_t0 = time.time()
    base_t0 = time.time()
    pos_target = np.asarray(world.robot_pose().pos, dtype=np.float64).reshape(3).copy()
    pos_target[0] = bx
    pos_target[1] = by
    base_set = _set_robot_base_pose(world, pos_target, math.radians(theta_x))
    base_pin_applied = _force_apply_shortcut_base_pin(world)
    base_elapsed_s = round(time.time() - base_t0, 3)

    trunk_t0 = time.time()
    _force_set_trunk_qpos(world, q_target)
    arm_hold, gripper_hold = _snapshot_limb_holds(world)
    world.set_trunk_pin_qpos(q_target.tolist())
    action: Dict[str, Any] = {"trunk": q_target.tolist()}
    action.update(_limb_hold_kwargs(arm_hold, gripper_hold))
    yield world.make_action(**action)
    _force_apply_shortcut_base_pin(world)
    _force_set_trunk_qpos(world, q_target)
    trunk_elapsed_s = round(time.time() - trunk_t0, 3)
    final_pin_sync = _sync_shortcut_pins_to_targets(
        world,
        trunk_q=q_target,
        arm_hold=arm_hold,
        gripper_hold=gripper_hold,
    )
    motion_elapsed_s = round(time.time() - motion_t0, 3)

    pose_errors = _pose_errors(world, bx, by, chest_z, theta_x, theta_z)
    position_reached = bool(_pose_within_tol(pose_errors))
    base_stage_report = {
        "ok": bool(base_set and abs(float(pose_errors.get("xy_m", 999.0))) <= 0.05 and abs(float(pose_errors.get("yaw_deg", 999.0))) <= 3.0),
        "mode": "v1_shortcut_direct_base_pose",
        "elapsed_s": base_elapsed_s,
        "base_pose_set": bool(base_set),
        "base_pin_applied": bool(base_pin_applied),
        "target_xy": [round(float(bx), 4), round(float(by), 4)],
        "target_yaw_deg": round(float(theta_x), 2),
        "xy_err_m": pose_errors.get("xy_m"),
        "yaw_err_deg": pose_errors.get("yaw_deg"),
    }
    trunk_stage_report = {
        "ok": bool(abs(float(pose_errors.get("theta_z_deg", 999.0))) <= 3.0),
        "mode": pitch_mode,
        "elapsed_s": trunk_elapsed_s,
        "target_trunk_q": [round(float(x), 5) for x in q_target.tolist()],
    }
    ctx.log(
        f"[v1_shortcut] {log_tag} direct stages: "
        f"lift={float(lift_report.get('elapsed_s', 0.0)):.3f}s "
        f"base_xy_yaw={base_elapsed_s:.3f}s trunk={trunk_elapsed_s:.3f}s "
        f"pose_err={pose_errors}"
    )
    head_live = _head_view_live(world, lo, hi)
    head_png = _capture_head_snapshot(
        world,
        session_id=session_id,
        head_png_tag=f"{tool}_head",
        log_tag=f"{log_tag}_shortcut",
        ctx=ctx,
        lo=lo,
        hi=hi,
        suffix="success" if position_reached else "not_reached",
    )
    if head_png.get("head_png"):
        head_live["head_png"] = head_png.get("head_png")
    final_hold_t0 = time.time()
    _force_set_trunk_qpos(world, q_target)
    action = {"trunk": q_target.tolist()}
    action.update(_limb_hold_kwargs(arm_hold, gripper_hold))
    yield world.make_action(**action)
    _force_apply_shortcut_base_pin(world)
    _force_set_trunk_qpos(world, q_target)
    final_hold_elapsed_s = round(time.time() - final_hold_t0, 3)
    final_pin_sync = _sync_shortcut_pins_to_targets(
        world,
        trunk_q=q_target,
        arm_hold=arm_hold,
        gripper_hold=gripper_hold,
    )
    pose_errors = _pose_errors(world, bx, by, chest_z, theta_x, theta_z)
    position_reached = bool(_pose_within_tol(pose_errors))
    trunk_q_live = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    trunk_stage_report["trunk_q_end"] = [
        round(float(x), 5)
        for x in trunk_q_live.tolist()
    ]
    try:
        trunk_stage_report["q_err_inf_rad"] = round(float(np.linalg.norm(trunk_q_live - q_target, ord=np.inf)), 6)
    except Exception:
        pass
    trunk_stage_report["theta_z_err_deg"] = pose_errors.get("theta_z_deg")
    base_stage_report["xy_err_m"] = pose_errors.get("xy_m")
    base_stage_report["yaw_err_deg"] = pose_errors.get("yaw_deg")
    base_stage_report["ok"] = bool(
        base_set
        and abs(float(pose_errors.get("xy_m", 999.0))) <= 0.05
        and abs(float(pose_errors.get("yaw_deg", 999.0))) <= 3.0
    )
    trunk_stage_report["ok"] = bool(abs(float(pose_errors.get("theta_z_deg", 999.0))) <= 3.0)
    try:
        chest_live = world.chest_pose()
    except Exception:
        chest_live = {}
    live_pose_errors = dict(pose_errors)
    if chest_live:
        live_pose_errors["live_chest_z"] = round(float(chest_live.get("z", 0.0)), 4)
        live_pose_errors["live_theta_z_deg"] = round(float(chest_live.get("theta_z_deg", 0.0)), 2)
    dual = _reach_report_live(world, C)
    shoulder_distance = _shoulder_distance_result(
        dual,
        C,
        reach_R_m=float(plan.get("reach_sphere_R_m", geom.get("reach_chord_m", reach_sphere_radius()))),
    )
    final = world.robot_pose()
    result = {
        "ok": bool(position_reached),
        "position_reached": bool(position_reached),
        "tool": tool,
        "build": _SHORTCUT_BUILD,
        "execution": "shortcut_direct_base_trunk_target",
        "target_center_world": [float(x) for x in C],
        "base_target": [float(bx), float(by)],
        "base_final": [float(final.pos[0]), float(final.pos[1])],
        "base_pose_set": bool(base_set),
        "theta_x_deg": round(theta_x, 1),
        "chest_z": round(chest_z, 3),
        "theta_z_deg": round(theta_z, 1),
        "trunk_plan": trunk_meta,
        "lift_report": lift_report,
        "timing": {
            "lift_elapsed_s": lift_report.get("elapsed_s"),
            "base_xy_yaw_elapsed_s": base_elapsed_s,
            "trunk_elapsed_s": trunk_elapsed_s,
            "motion_elapsed_s": motion_elapsed_s,
            "final_hold_elapsed_s": final_hold_elapsed_s,
        },
        "visibility": head_live,
        "head_after_move_png": head_live.get("head_png"),
        "planned_travel_m": round(float(detail.get("travel_m", 0.0)), 4),
        "actual_travel_m": round(float(np.linalg.norm(np.asarray(final.pos[:2]) - robot_xy)), 4),
        "chord_plan": plan,
        "base_target_clearance": base_target_clearance,
        "base_target_min_clearance_m": _BASE_TARGET_MIN_CLEARANCE_M,
        "base_target_backoff": base_backoff,
        "reachable": dual["reachable"],
        "reachable_left": dual["reachable_left"],
        "reachable_right": dual["reachable_right"],
        "arms_reachable": dual["arms_reachable"],
        "arm": dual.get("arm"),
        "preferred_arm": dual.get("preferred_arm"),
        "pose_errors": pose_errors,
        "path_exec": {
            "mode": "v1_shortcut_direct_set_base_pose_and_trunk_q",
            "nav_guard": "not_run",
            "astar": "not_run",
            "base_xy_yaw": base_stage_report,
        },
        "trunk_exec": trunk_stage_report,
        "final_pin_sync": final_pin_sync,
        "final_live_chest_pose": chest_live,
        "final_live_pose_errors": live_pose_errors,
        "error": None if position_reached else "shortcut direct target pose not reached",
        **shoulder_distance,
        **extra,
    }
    if tool == "move_to_object":
        result["object_center_world"] = result["target_center_world"]
    elif tool == "move_to_point":
        result["point_center_world"] = result["target_center_world"]
    ctx.log(
        f"[v1_shortcut] {log_tag} done ok={position_reached} "
        f"base=({bx:.3f},{by:.3f}) yaw={theta_x:.1f} pose_err={pose_errors} "
        f"timing={result['timing']}"
    )
    ctx.set_result(result)
    _save_nav_reach(session_id, result)
    yield world.hold_action()


@register_skill("move_to_object_v2", description="v1_shortcut: resolve object, plan final target, then direct-set base/trunk.")
def move_to_object_v2(
    ctx,
    session_id: str,
    object_name: str = "",
    image_id: str = "",
    u: Optional[int] = None,
    v: Optional[int] = None,
    standoff: float = 0.0,
    reach: float = 0.0,
    nav_timeout_s: float = 120.0,
):
    world = ctx.world
    try:
        from behavior_interface.skills.move_to_object_v2 import _aabb, _resolve, _resolve_from_uv
    except Exception as exc:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_object", msg=str(exc))
        return
    name = str(object_name or "").strip()
    pick_uv = bool(
        not name
        and str(image_id or "").strip()
        and u is not None
        and v is not None
    )
    if name:
        object_name = name
        obj = _resolve(world, name)
        ctx.log(f"[v1_shortcut] move_to_object_v2 object_name priority: {object_name}")
    elif pick_uv:
        bddl, obj, err = _resolve_from_uv(ctx, session_id, image_id.strip(), int(u), int(v))
        if err:
            yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_object", msg=err)
            return
        object_name = bddl or ""
    else:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_object", msg="需要 object_name，或 image_id + u + v 点选物体")
        return
    if obj is None:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_object", msg=f"未找到物体: {object_name}")
        return
    ab = _aabb(obj)
    if ab is None:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_object", msg=f"物体 {object_name} 无 AABB")
        return
    lo, hi = ab
    C = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2.0
    yield from _move_to_center_shortcut(
        ctx,
        session_id=session_id,
        C=C,
        lo=lo,
        hi=hi,
        reach=reach,
        tool="move_to_object",
        log_tag="move_to_object_v2",
        result_extra={
            "object_name": object_name,
            "resolved_by": "uv_pick" if pick_uv else "name",
            "image_id": image_id.strip() if pick_uv else None,
            "uv": [int(u), int(v)] if pick_uv else None,
            "sphere_center_source": "object_aabb_center",
        },
    )


def _move_to_point_common(ctx, *, session_id: str, image_id: str, u: int, v: int, reach: float, version: str):
    try:
        from behavior_interface.skills.move_to_object_v2 import _aabb_around_point
        from behavior_interface.skills.move_to_point_v2 import _sphere_center_from_uv
    except Exception as exc:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_point", msg=str(exc))
        return
    img = str(image_id or "").strip()
    if not img:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_point", msg="需要 image_id（先 capture）")
        return
    if int(u) < 0 or int(v) < 0:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_point", msg="需要 head 图上点选 (u,v)")
        return
    C, hit_method, err = _sphere_center_from_uv(ctx, session_id, img, int(u), int(v))
    if err:
        yield from _move_to_fail(ctx, session_id=session_id, tool="move_to_point", msg=err)
        return
    lo, hi = _aabb_around_point(np.asarray(C, dtype=np.float64), margin=0.03)
    yield from _move_to_center_shortcut(
        ctx,
        session_id=session_id,
        C=C,
        lo=lo,
        hi=hi,
        reach=reach,
        tool="move_to_point",
        log_tag=version,
        result_extra={
            "image_id": img,
            "uv": [int(u), int(v)],
            "hit_world": [float(x) for x in np.asarray(C, dtype=np.float64).reshape(3)],
            "hit_method": hit_method,
            "sphere_center_source": "uv_hit",
        },
    )


@register_skill("move_to_point_v2", description="v1_shortcut: resolve point, plan final target, then direct-set base/trunk.")
def move_to_point_v2(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    reach: float = 0.0,
    nav_timeout_s: float = 120.0,
):
    yield from _move_to_point_common(
        ctx,
        session_id=session_id,
        image_id=image_id,
        u=u,
        v=v,
        reach=reach,
        version="move_to_point_v2",
    )


@register_skill("move_to_point_v3", description="v1_shortcut: resolve point, plan final target, then direct-set base/trunk.")
def move_to_point_v3(
    ctx,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    reach: float = 0.0,
    nav_timeout_s: float = 120.0,
):
    yield from _move_to_point_common(
        ctx,
        session_id=session_id,
        image_id=image_id,
        u=u,
        v=v,
        reach=reach,
        version="move_to_point_v3",
    )


@register_skill(
    "move_eef",
    description="v1_shortcut: solve final EEF pose once, including optional u/v/depth target.",
)
def move_eef(
    ctx,
    upward: float = 0.0,
    forward: float = 0.0,
    leftward: float = 0.0,
    u: Optional[float] = None,
    v: Optional[float] = None,
    depth: Optional[float] = None,
    gripper: str = "keep",
    arm: str = "right",
    pos_tol: float = 0.012,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    from behavior_interface.skills.move_eef import (
        _has_value,
        _head_cam_pose,
        camera_delta_from_world,
        camera_delta_to_world,
        target_from_uv_depth,
    )

    world = ctx.world
    _sync_shortcut_pins(world, include_current=False)
    arm_eff = _arm_eff(arm)
    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系增量"})
        yield world.hold_action()
        return
    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = _as_vec(eef0["pos"], 3)
    quat0 = _as_vec(eef0["quat"], 4)
    uv_depth_supplied = [_has_value(u), _has_value(v), _has_value(depth)]
    use_uv_depth = any(uv_depth_supplied)
    if use_uv_depth and not all(uv_depth_supplied):
        ctx.set_result({
            "ok": False,
            "tool": "move_eef",
            "build": _SHORTCUT_BUILD,
            "arm": arm_eff,
            "error": "u, v, depth 必须同时提供；depth 单位 m，语义为 head-camera forward z-depth",
        })
        yield world.hold_action()
        return
    target_source = "camera_delta_cm"
    uv_depth_target = None
    if use_uv_depth:
        try:
            target_pos, uv_depth_target = target_from_uv_depth(
                world,
                cam_pos,
                cam_quat,
                u=float(u),
                v=float(v),
                depth=float(depth),
            )
        except Exception as exc:
            ctx.set_result({
                "ok": False,
                "tool": "move_eef",
                "build": _SHORTCUT_BUILD,
                "arm": arm_eff,
                "error": f"u/v/depth 反解 EEF 目标失败: {exc}",
            })
            yield world.hold_action()
            return
        delta = target_pos - pos0
        delta_cam = camera_delta_from_world(cam_quat, delta)
        upward_eff = float(delta_cam["upward"])
        forward_eff = float(delta_cam["forward"])
        leftward_eff = float(delta_cam["leftward"])
        target_source = "uv_depth"
    else:
        upward_eff = float(upward)
        forward_eff = float(forward)
        leftward_eff = float(leftward)
        delta = camera_delta_to_world(
            cam_quat,
            upward_cm=upward_eff,
            forward_cm=forward_eff,
            leftward_cm=leftward_eff,
        )
        target_pos = pos0 + delta
    if float(np.linalg.norm(delta)) < 1e-9:
        q = np.asarray(world.arm_qpos_list(arm_eff), dtype=np.float64).reshape(7)
    else:
        q, solve_pos_err, solve_ori_err = _solve_final_q(ctx, arm_eff, target_pos, quat0)
        if q is None:
            ctx.set_result({
                "ok": False,
                "tool": "move_eef",
                "build": _SHORTCUT_BUILD,
                "arm": arm_eff,
                "error": "shortcut_move_eef_ik_failed",
                "solve_pos_err_m": float(solve_pos_err),
                "solve_ori_err_deg": float(solve_ori_err),
            })
            yield world.hold_action()
            return
    play = yield from _play_arm_q(ctx, arm_eff, q, gripper_mode=gripper, label="move_eef")
    eef1 = world.eef_pose(arm=arm_eff)
    pos1 = _as_vec(eef1["pos"], 3)
    quat1 = _as_vec(eef1["quat"], 4)
    pos_err = float(np.linalg.norm(pos1 - target_pos))
    ori_err = _quat_err_deg(quat1, quat0)
    ok = bool(pos_err <= max(float(pos_tol) * 2.5, 0.020) and ori_err <= 6.0)
    ctx.set_result({
        "ok": ok,
        "tool": "move_eef",
        "build": _SHORTCUT_BUILD,
        "execution": "direct_joint_target",
        "arm": arm_eff,
        "target_source": target_source,
        "uv_depth_target": uv_depth_target,
        "delta_cam_cm": {
            "upward": upward_eff,
            "forward": forward_eff,
            "leftward": leftward_eff,
        },
        "delta_world_m": delta.round(6).tolist(),
        "target_pos": target_pos.tolist(),
        "actual_pos": pos1.tolist(),
        "pos_err_m": pos_err,
        "ori_err_deg": ori_err,
        "play": play,
    })
    yield world.hold_action()


@register_skill("rotate_eef", description="v1_shortcut: solve final rotated EEF pose once, then play target arm joints.")
def rotate_eef(
    ctx,
    rotate_forward_deg: float = 0.0,
    rotate_upward_deg: float = 0.0,
    rotate_leftward_deg: float = 0.0,
    arm: str = "right",
    ori_tol_deg: float = 3.0,
    pos_tol: float = 0.015,
    max_steps: int = 240,
    timeout_s: float = 60.0,
):
    from behavior_interface.skills.move_eef import _head_cam_pose
    from behavior_interface.skills.rotate_eef import camera_frame_delta_quat, _quat_mul

    world = ctx.world
    _sync_shortcut_pins(world, include_current=False)
    arm_eff = _arm_eff(arm)
    cam_pos, cam_quat = _head_cam_pose(world)
    if cam_pos is None or cam_quat is None:
        ctx.set_result({"ok": False, "error": "未找到 head 相机，无法换算相机系旋转"})
        yield world.hold_action()
        return
    eef0 = world.eef_pose(arm=arm_eff)
    pos0 = _as_vec(eef0["pos"], 3)
    quat0 = _as_vec(eef0["quat"], 4)
    q_delta = camera_frame_delta_quat(
        cam_quat,
        rotate_forward_deg=float(rotate_forward_deg),
        rotate_upward_deg=float(rotate_upward_deg),
        rotate_leftward_deg=float(rotate_leftward_deg),
    )
    quat_tgt = _as_vec(_quat_mul(q_delta, quat0), 4)
    n = float(np.linalg.norm(quat_tgt))
    if n > 1e-9:
        quat_tgt = quat_tgt / n
    q, solve_pos_err, solve_ori_err = _solve_final_q(ctx, arm_eff, pos0, quat_tgt)
    if q is None:
        ctx.set_result({
            "ok": False,
            "tool": "rotate_eef",
            "build": _SHORTCUT_BUILD,
            "arm": arm_eff,
            "error": "shortcut_rotate_eef_ik_failed",
            "solve_pos_err_m": float(solve_pos_err),
            "solve_ori_err_deg": float(solve_ori_err),
        })
        yield world.hold_action()
        return
    play = yield from _play_arm_q(ctx, arm_eff, q, gripper_mode="keep", label="rotate_eef")
    eef1 = world.eef_pose(arm=arm_eff)
    pos1 = _as_vec(eef1["pos"], 3)
    quat1 = _as_vec(eef1["quat"], 4)
    pos_err = float(np.linalg.norm(pos1 - pos0))
    ori_err = _quat_err_deg(quat1, quat_tgt)
    ok = bool(pos_err <= max(float(pos_tol) * 3.0, 0.040) and ori_err <= max(float(ori_tol_deg) * 2.0, 6.0))
    ctx.set_result({
        "ok": ok,
        "tool": "rotate_eef",
        "build": _SHORTCUT_BUILD,
        "execution": "direct_joint_target",
        "arm": arm_eff,
        "pos_err_m": pos_err,
        "ori_err_deg": ori_err,
        "quat_target": quat_tgt.tolist(),
        "quat_after": quat1.tolist(),
        "play": play,
    })
    yield world.hold_action()


@register_skill(
    "set_arm_to_grasp_position",
    description=(
        "v1_shortcut: direct grasp-prep joint target; "
        "gripper=open|keep (default keep)."
    ),
)
def set_arm_to_grasp_position(
    ctx,
    arm: str = "right",
    gripper: str | None = None,
    open_gripper: bool | None = None,
    max_dq_per_step: float = 0.24,
    tol: float = 0.08,
    timeout_s: float = 35.0,
    force_jointspace: bool = True,
):
    from behavior_interface.skills.arm_reset import (
        _force_open_gripper_qpos,
        _force_set_arm_qpos,
        _grasp_prep_shoulder_hang_target_qpos,
        _grasp_prep_verify,
        _gripper_open_override,
        _gripper_qpos_map,
        _normalize_grasp_gripper_mode,
        _phase1_v2_grasp_target_qpos,
    )

    arm_norm = str(arm or "right").strip().lower()
    if arm_norm not in ("left", "right", "both"):
        ctx.set_result({"ok": False, "error": f"bad arm '{arm}'"})
        yield ctx.world.hold_action()
        return
    arms = ["left", "right"] if arm_norm == "both" else [arm_norm]
    mode = _normalize_grasp_gripper_mode(gripper, open_gripper)

    world = ctx.world
    if getattr(world, "dry_run", False):
        ctx.set_result({
            "ok": True,
            "dry_run": True,
            "build": _SHORTCUT_BUILD,
            "execution": "direct_grasp_prep_joint_target",
            "arms": arms,
            "gripper": mode,
        })
        yield world.hold_action()
        return
    try:
        from behavior_interface.skills.eef import _ensure_world_pinned_actions

        _ensure_world_pinned_actions(world)
    except Exception:
        pass
    from behavior_interface.skills.eef import _reset_selected_tool_roll_to_zero

    for selected_arm in arms:
        _reset_selected_tool_roll_to_zero(
            world,
            selected_arm,
            ctx=ctx,
            stage_name="set_arm_to_grasp_position.start",
        )
    _sync_shortcut_pins(world, include_current=False)

    target_map: Dict[str, np.ndarray] = {}
    phase1_map: Dict[str, np.ndarray] = {}
    keep_grip: Dict[str, Optional[list[float]]] = {}
    for a in arms:
        phase1 = _phase1_v2_grasp_target_qpos(world, a)
        final = _grasp_prep_shoulder_hang_target_qpos(phase1)
        phase1_map[a] = np.asarray(phase1, dtype=np.float64).reshape(7)
        target_map[a] = np.asarray(final, dtype=np.float64).reshape(7)
        keep_grip[a] = _gripper_keep(world, a)
        ctx.log(
            f"[v1_shortcut] set_arm_to_grasp_position[{a}] "
            f"direct grasp-prep q=[{','.join(f'{v:+.3f}' for v in target_map[a])}] "
            f"gripper={mode}"
        )

    action = {}
    for a in arms:
        _force_set_arm_qpos(world, a, target_map[a])
        action[f"arm_{a}"] = target_map[a].tolist()
        if mode == "open":
            _force_open_gripper_qpos(world, a)
            action[f"gripper_{a}"] = _gripper_open_override(world, a)
        elif keep_grip.get(a) is not None:
            try:
                world.set_gripper_pin_qpos(a, keep_grip[a])
            except Exception:
                pass
            action[f"gripper_{a}"] = keep_grip[a]
    # Shortcut profile: qpos + pin is the execution primitive.  A single action
    # tick is enough to refresh controller targets; extra ticks dominate wall
    # time on 15051 without changing the verified final q.
    yield world.make_action(**action)

    verify: Dict[str, Dict[str, Any]] = {}
    ok = True
    for a in arms:
        _force_set_arm_qpos(world, a, target_map[a])
        v = _grasp_prep_verify(world, a, phase1_map[a], target_map[a], tol=float(tol))
        verify[a] = v
        ok = bool(ok and v.get("ok", False))
        try:
            world.set_arm_pin_qpos(a, target_map[a])
            if mode == "keep" and keep_grip.get(a) is not None:
                world.set_gripper_pin_qpos(a, keep_grip[a])
        except Exception:
            pass
        ctx.log(
            f"[v1_shortcut] set_arm_to_grasp_position[{a}] "
            f"verify ok={v.get('ok')} err={v.get('err_final_rad')}rad"
        )

    final_pin_sync = _sync_shortcut_pins_to_targets(
        world,
        arm_hold={a: target_map[a] for a in arms},
        gripper_hold={
            a: (_gripper_open_cmd(world, a) if mode == "open" else keep_grip.get(a))
            for a in arms
        },
    )
    ctx.set_result({
        "ok": bool(ok),
        "build": _SHORTCUT_BUILD,
        "execution": "direct_grasp_prep_joint_target",
        "arms": arms,
        "pose": "grasp_prep_shoulder_hang_elbow_wrist_folded",
        "gripper": mode,
        "open_gripper": mode == "open",
        "keep_gripper_qpos": {
            a: [round(float(x), 5) for x in (keep_grip.get(a) or [])]
            for a in arms
        },
        "gripper_qpos": _gripper_qpos_map(world, arms),
        "target_qpos": {
            a: [round(float(x), 5) for x in target_map[a].tolist()]
            for a in arms
        },
        "verify": verify,
        "tol_rad": float(tol),
        "final_pin_sync": final_pin_sync,
        "error": None if ok else "grasp prep 关节未到位",
    })
    return


@register_skill("reset_body", description="v1_shortcut: direct trunk upright q target, optionally compensate wrist to keep EEF orientation.")
def reset_body(
    ctx,
    timeout_s: float = 45.0,
    trunk_max_step: float = 0.24,
    shoulder_iters: int = 4,
    keep_ori_arm: str = "none",
):
    from behavior_interface.skills.arm_reset import _arm_qpos
    from behavior_interface.skills.reset_body import (
        _arm_locked_drift_report,
        _normalize_keep_ori_arm,
        _quat_err_deg,
        _quat_normalize,
        _robot_joint_state_clone,
        _set_full_joint_positions,
        _solve_wrist_for_eef_ori_at_trunk,
    )

    world = ctx.world
    if getattr(world, "dry_run", False):
        ctx.set_result({"ok": True, "dry_run": True, "build": _SHORTCUT_BUILD})
        yield world.hold_action()
        return
    _sync_shortcut_pins(world, include_current=False)
    trunk0 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    target = np.zeros(4, dtype=np.float64)
    keep_ori_arms = _normalize_keep_ori_arm(keep_ori_arm)
    arm_hold: Dict[str, np.ndarray] = {}
    gripper_hold: Dict[str, Optional[list[float]]] = {}
    keep_ori_quat0: Dict[str, np.ndarray] = {}
    keep_ori_arm_q: Dict[str, np.ndarray] = {}
    keep_ori_solve_err: Dict[str, float] = {}
    keep_ori_solve_ok: Dict[str, bool] = {}
    for a in ("left", "right"):
        try:
            arm_hold[a] = np.asarray(_arm_qpos(world, a), dtype=np.float64).reshape(7).copy()
            gripper_hold[a] = _gripper_keep(world, a)
            world.set_arm_pin_qpos(a, arm_hold[a])
            if a in keep_ori_arms:
                keep_ori_quat0[a] = _quat_normalize(world.eef_pose(arm=a)["quat"])
                keep_ori_arm_q[a] = arm_hold[a].copy()
        except Exception as exc:
            ctx.log(f"[v1_shortcut] reset_body cannot lock {a}: {exc}")
            keep_ori_arms.discard(a)

    ctx.log(
        f"[v1_shortcut] reset_body direct trunk q "
        f"[{','.join(f'{v:+.3f}' for v in trunk0)}] -> [0,0,0,0] "
        f"arms_locked={list(arm_hold.keys())} keep_ori_arm={sorted(keep_ori_arms)}"
    )
    saved_probe = None
    saved_vel = None
    if keep_ori_arms:
        try:
            saved_probe = _robot_joint_state_clone(world.robot)
            try:
                saved_vel = world.robot.get_joint_velocities()
                try:
                    saved_vel = saved_vel.clone()
                except Exception:
                    saved_vel = np.asarray(saved_vel, dtype=np.float64).copy()
            except Exception:
                saved_vel = None
            for a in sorted(keep_ori_arms):
                wrist, err_deg, ok_w = _solve_wrist_for_eef_ori_at_trunk(
                    world,
                    a,
                    trunk_q=target,
                    shoulder_elbow_q=arm_hold[a][:4],
                    wrist_seed_q=arm_hold[a][4:7],
                    target_quat=keep_ori_quat0[a],
                    max_steps=36,
                    ori_tol_deg=0.25,
                )
                q_cmd = arm_hold[a].copy()
                q_cmd[4:7] = wrist
                keep_ori_arm_q[a] = q_cmd
                keep_ori_solve_err[a] = float(err_deg)
                keep_ori_solve_ok[a] = bool(ok_w)
                ctx.log(
                    f"[v1_shortcut] reset_body keep_ori {a}: "
                    f"wrist=[{','.join(f'{v:+.4f}' for v in wrist)}] "
                    f"solve_ori_err={err_deg:.3f}deg ok={ok_w}"
                )
        except Exception as exc:
            ctx.log(f"[v1_shortcut] reset_body keep_ori compensation failed: {type(exc).__name__}: {exc}")
        finally:
            if saved_probe is not None:
                try:
                    _set_full_joint_positions(world.robot, saved_probe)
                except Exception:
                    pass
            if saved_vel is not None:
                try:
                    world.robot.set_joint_velocities(saved_vel)
                except Exception:
                    pass

    _force_set_trunk_qpos(world, target)
    for _ in range(12):
        action: Dict[str, Any] = {"trunk": target.tolist()}
        for a, q in arm_hold.items():
            action[f"arm_{a}"] = (keep_ori_arm_q.get(a, q) if a in keep_ori_arms else q).tolist()
            if gripper_hold.get(a) is not None:
                action[f"gripper_{a}"] = gripper_hold[a]
        yield world.make_action(**action)
    _force_set_trunk_qpos(world, target)
    final_arm_targets: Dict[str, np.ndarray] = {
        a: (keep_ori_arm_q.get(a, q) if a in keep_ori_arms else q)
        for a, q in arm_hold.items()
    }
    final_pin_sync = _sync_shortcut_pins_to_targets(
        world,
        trunk_q=target,
        arm_hold=final_arm_targets,
        gripper_hold=gripper_hold,
    )

    trunk1 = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    arm_drift: Dict[str, Optional[float]] = {}
    for a, q in arm_hold.items():
        try:
            arm_drift[a] = float(np.linalg.norm(np.asarray(_arm_qpos(world, a)) - q, ord=np.inf))
        except Exception:
            arm_drift[a] = None
    locked_drift = _arm_locked_drift_report(world, arm_hold, keep_ori_arms)
    keep_ori_final_err: Dict[str, Optional[float]] = {}
    keep_ori_quat_after: Dict[str, list[float]] = {}
    for a in sorted(keep_ori_arms):
        try:
            qf = _quat_normalize(world.eef_pose(arm=a)["quat"])
            keep_ori_quat_after[a] = [round(float(x), 6) for x in qf.tolist()]
            keep_ori_final_err[a] = float(_quat_err_deg(keep_ori_quat0[a], qf))
        except Exception:
            keep_ori_final_err[a] = None
    trunk_err = float(np.linalg.norm(trunk1 - target, ord=np.inf))
    finite_drifts = [float(x) for x in locked_drift.values() if x is not None and np.isfinite(float(x))]
    max_arm_drift = float(max(finite_drifts)) if finite_drifts else 0.0
    keep_ori_ok = all(v is None or float(v) <= 3.0 for v in keep_ori_final_err.values())
    ok = bool(trunk_err <= 0.030 and max_arm_drift <= 0.080 and keep_ori_ok)
    ctx.set_result({
        "ok": ok,
        "build": _SHORTCUT_BUILD,
        "execution": "direct_trunk_joint_target_keep_ori_wrist_comp",
        "trunk_before": [round(float(x), 5) for x in trunk0.tolist()],
        "trunk_after": [round(float(x), 5) for x in trunk1.tolist()],
        "trunk_err_inf_rad": round(trunk_err, 5),
        "arms_locked": list(arm_hold.keys()),
        "keep_ori_arm": sorted(keep_ori_arms),
        "keep_ori_quat_before": {
            a: [round(float(x), 6) for x in q.tolist()] for a, q in keep_ori_quat0.items()
        },
        "keep_ori_quat_after": keep_ori_quat_after,
        "keep_ori_solve_err_deg": {
            a: round(float(v), 5) for a, v in keep_ori_solve_err.items()
        },
        "keep_ori_solve_ok": dict(keep_ori_solve_ok),
        "keep_ori_err_deg": {
            a: (round(float(v), 5) if v is not None and np.isfinite(float(v)) else None)
            for a, v in keep_ori_final_err.items()
        },
        "keep_ori_wrist_qpos": {
            a: [round(float(x), 5) for x in q.tolist()] for a, q in keep_ori_arm_q.items()
        },
        "arm_drift_inf_rad": {
            a: (round(float(d), 5) if d is not None and np.isfinite(float(d)) else None)
            for a, d in arm_drift.items()
        },
        "locked_drift_inf_rad": {
            a: (round(float(d), 5) if np.isfinite(float(d)) else None)
            for a, d in locked_drift.items()
        },
        "max_arm_drift_inf_rad": round(max_arm_drift, 5),
        "gripper_mode": "keep",
        "final_pin_sync": final_pin_sync,
        "error": None if ok else "reset_body shortcut target not reached",
    })
    yield world.hold_action()
