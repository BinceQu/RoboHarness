from __future__ import annotations

from typing import Any
import numpy as np
from tools.util.geometry import (
    finite_vector,
    optical_vector_to_world,
    surface_normal_at_uv,
)

from tools.util.motion_constants import (
    X5_SHOULDERS,
    _COMPAT_PLAN_MODES,
    _COMPAT_PUSH_OPTICAL_DIRECTIONS,
)
from tools.util.motion_common import (
    _finite_number,
    _head_snapshot,
    _save_pose_plan,
    _select_arm,
    _set_result,
    _snapshot_geometry,
    _target_from_click,
)


def _horizontal_camera_direction(extrinsic: Any, optical: Any) -> np.ndarray:
    optical_direction = finite_vector(optical, 3, "optical interaction direction")
    direction = optical_vector_to_world(extrinsic, optical_direction)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        # A camera axis can become vertical after table-plane projection. The
        # signed camera-right axis keeps opposite directions distinct.
        sign = -1.0 if float(np.sum(optical_direction)) < 0.0 else 1.0
        direction = optical_vector_to_world(extrinsic, [sign, 0.0, 0.0])
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        norm = 1.0
    return direction / norm


def _contact_plan(
    ctx,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    contact_kind: str,
) -> dict[str, Any]:
    snapshot = _head_snapshot(ctx, arguments)
    target, point_meta = _target_from_click(
        snapshot, arguments.get("u"), arguments.get("v")
    )
    depth, intrinsic, extrinsic = _snapshot_geometry(snapshot)
    normal = surface_normal_at_uv(
        depth,
        intrinsic,
        extrinsic,
        _finite_number(arguments.get("u"), "u"),
        _finite_number(arguments.get("v"), "v"),
    )
    requested = arguments.get("plan_arm", arguments.get("arm", "any"))
    arm = _select_arm(ctx, target, requested)
    current = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
    compat_mode = str(arguments.get("_compat_plan_mode", "") or "").strip()
    if compat_mode and compat_mode not in _COMPAT_PLAN_MODES:
        raise ValueError(f"unsupported compatibility plan mode {compat_mode!r}")
    contact_offset = 0.08 if contact_kind == "press" else 0.10
    safe_offset = contact_offset + (0.10 if contact_kind == "press" else 0.08)
    safe = current.copy()
    contact = current.copy()
    safe[:3] = target + normal * safe_offset
    contact[:3] = target + normal * contact_offset
    waypoints = [
        {"label": "safe", "pose": safe.tolist()},
        {"label": contact_kind, "pose": contact.tolist()},
    ]
    interaction_direction = None
    interaction_distance = None
    semantic_status = "rgbd_policy_approximation"

    if compat_mode in _COMPAT_PUSH_OPTICAL_DIRECTIONS:
        interaction_distance = _finite_number(
            arguments.get("_compat_distance_m", 0.10), "interaction distance"
        )
        interaction_direction = _horizontal_camera_direction(
            extrinsic, _COMPAT_PUSH_OPTICAL_DIRECTIONS[compat_mode]
        )
        interaction = contact.copy()
        interaction[:3] += interaction_direction * interaction_distance
        waypoints.append({"label": compat_mode, "pose": interaction.tolist()})
        semantic_status = "rgbd_directional_approximation"
    elif compat_mode == "open":
        interaction_distance = _finite_number(
            arguments.get("_compat_distance_m", 0.12), "interaction distance"
        )
        interaction_direction = normal.copy()
        interaction = contact.copy()
        interaction[:3] += interaction_direction * interaction_distance
        waypoints.append({"label": "pull", "pose": interaction.tolist()})
        semantic_status = "rgbd_articulation_agnostic_approximation"
    elif compat_mode == "close":
        interaction_distance = _finite_number(
            arguments.get("_compat_distance_m", 0.10), "interaction distance"
        )
        interaction_direction = -normal
        interaction = contact.copy()
        interaction[:3] += interaction_direction * interaction_distance
        waypoints.append({"label": "push_surface", "pose": interaction.tolist()})
        semantic_status = "rgbd_articulation_agnostic_approximation"

    if interaction_distance is not None and not 0.02 <= interaction_distance <= 0.25:
        raise ValueError("interaction distance must be in 0.02..0.25 m")
    for waypoint in waypoints:
        pose = finite_vector(waypoint["pose"], 7, "planned EEF waypoint")
        shoulder_distance = float(np.linalg.norm(pose[:3] - X5_SHOULDERS[arm]))
        if shoulder_distance > 1.05:
            raise ValueError(
                f"{waypoint['label']} waypoint is outside the conservative X5 reach bound"
            )

    parameters = {
        "contact_kind": contact_kind,
        "point": point_meta,
        "surface_normal_robodojo_env": normal.tolist(),
        "seed": int(arguments.get("seed", 42)),
        "gripper_policy": "preserve",
        "compat_plan_mode": compat_mode or None,
        "compat_gripper_transition": "preserve",
        "exec_sequence": "move_only",
        "semantic_status": semantic_status,
    }
    if interaction_direction is not None:
        parameters["interaction_direction_robodojo_env"] = (
            interaction_direction.tolist()
        )
        parameters["interaction_distance_m"] = interaction_distance
    result = _save_pose_plan(
        ctx,
        tool_name=tool_name,
        snapshot=snapshot,
        arm=arm,
        waypoints=waypoints,
        parameters=parameters,
    )
    result.update(
        {
            "target_robodojo_env_m": target.tolist(),
            "surface_normal_robodojo_env": normal.tolist(),
            "identity_source": "agent_selected_rgbd_point",
            "identity_verified": False,
            "compat_plan_mode": compat_mode or None,
            "semantic_status": semantic_status,
            "interaction_direction_robodojo_env": None
            if interaction_direction is None
            else interaction_direction.tolist(),
            "interaction_distance_m": interaction_distance,
        }
    )
    return result


def plan_press_point(ctx, **kwargs):
    try:
        result = _contact_plan(
            ctx,
            tool_name="plan_press_point",
            arguments=kwargs,
            contact_kind="press",
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    _set_result(ctx, "plan_press_point", result)
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("plan_press_point", plan_press_point)
