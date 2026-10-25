"""Test-owned two-phase R1Pro vertical torso trajectory planner.

The planner consumes only evaluator proprioception and static submission
assets.  It intentionally has no dependency on the simulator-backed Interface
implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


REVERSE_UPWARD_Z_PHASE_BOUNDARY_M = 0.73
REVERSE_UPWARD_Z_CM_STEP_M = 0.01
TRUNK_LUT_HOLD_ACTIONS = 3
TRUNK_LUT_MAX_EXEC_WAYPOINTS = 18
TRUNK_LUT_JOINT_MATCH_TOLERANCE_RAD = 1e-4

TRUNK_JOINT_LIMITS = np.array(
    [
        [-1.1345, 1.8326],
        [-2.7925, 2.5307],
        [-1.8326, 1.5708],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)

_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "trunk_vertical_lift"
_TABLE_ASSETS = (
    (
        "phase1_sine_manifold_table.json",
        "70d981ab976ac06c4ecfd5805d675e06a07323d6d4d083a54aebc5c25a6b4d4a",
        1,
    ),
    (
        "phase2_theta90_table.json",
        "9b937d25f0748ac5b0840851b4f483972bc36ae48bc1f64f60536902b18ca4a3",
        2,
    ),
)


def torso_chest_height_local_m(trunk_q) -> float:
    """Return torso_link4 height in the submitted robot base frame."""
    q1, q2, q3, _q4 = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    return float(
        0.34265
        + 0.40000 * math.cos(float(q1))
        + 0.30000 * math.cos(float(q1 + q2))
        + 0.09962 * math.cos(float(q1 + q2 - q3))
    )


def torso_theta_z_deg(trunk_q) -> float:
    """Return the local-FK forward-axis polar angle used by Interface v2."""
    q1, q2, q3, _q4 = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    forward_z = -math.sin(float(q1 + q2 - q3))
    return math.degrees(math.acos(float(np.clip(forward_z, -1.0, 1.0))))


def snap_z_to_nearest_cm(
    z_value: float,
    step_m: float = REVERSE_UPWARD_Z_CM_STEP_M,
) -> float:
    step = max(0.001, float(step_m))
    return round(round(float(z_value) / step) * step, 4)


def _load_table_asset(
    filename: str,
    expected_digest: str,
) -> tuple[dict[str, Any], str]:
    path = _ASSET_DIR / filename
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_digest:
        raise RuntimeError(
            f"validated torso LUT digest mismatch for {filename}: {digest}"
        )
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise RuntimeError(f"validated torso LUT is invalid: {filename}")
    return parsed, digest


def _validated_row(
    sample: dict[str, Any],
    *,
    phase: int,
) -> dict[str, Any]:
    q_raw = sample.get("trunk_q")
    if not isinstance(q_raw, list) or len(q_raw) < 3:
        q_raw = [
            sample.get("q1_rad"),
            sample.get("q2_rad"),
            sample.get("q3_rad"),
        ]
    q = np.asarray([q_raw[0], q_raw[1], q_raw[2], 0.0], dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise RuntimeError(f"phase{phase} torso LUT contains non-finite joints")
    if bool(
        np.any(q < TRUNK_JOINT_LIMITS[:, 0] - 1e-5)
        or np.any(q > TRUNK_JOINT_LIMITS[:, 1] + 1e-5)
    ):
        raise RuntimeError(f"phase{phase} torso LUT contains out-of-limit joints")
    z_target = float(sample["z_target_m"])
    z_observed = float(sample.get("chest_z_world_m", z_target))
    return {
        "phase": int(phase),
        "z_target_m": z_target,
        "z_observed_m": z_observed,
        "trunk_q": q,
        "theta_z_deg": float(
            sample.get("theta_z_deg", torso_theta_z_deg(q))
        ),
    }


@lru_cache(maxsize=1)
def load_reverse_upward_lut() -> dict[str, Any]:
    """Load and validate the submission-local copy of the v2 torso LUT."""
    tables: dict[int, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for filename, expected_digest, phase in _TABLE_ASSETS:
        table, digest = _load_table_asset(filename, expected_digest)
        tables[phase] = table
        digests[filename] = digest

    rows: list[dict[str, Any]] = []
    for sample in tables[1].get("samples", []):
        if sample.get("ok") is not True:
            continue
        row = _validated_row(sample, phase=1)
        if row["z_target_m"] + 1e-4 >= REVERSE_UPWARD_Z_PHASE_BOUNDARY_M:
            rows.append(row)
    for sample in tables[2].get("samples", []):
        if sample.get("ok") is not True:
            continue
        row = _validated_row(sample, phase=2)
        if row["z_target_m"] < REVERSE_UPWARD_Z_PHASE_BOUNDARY_M - 1e-4:
            rows.append(row)
    rows.sort(key=lambda row: float(row["z_target_m"]), reverse=True)
    if not rows:
        raise RuntimeError("validated two-phase torso LUT contains no rows")
    if any(
        rows[index]["z_target_m"] <= rows[index + 1]["z_target_m"]
        for index in range(len(rows) - 1)
    ):
        raise RuntimeError("validated torso LUT heights are not strictly descending")

    frame_offsets = [
        float(row["z_observed_m"])
        - torso_chest_height_local_m(row["trunk_q"])
        for row in rows
    ]
    height_frame_offset = float(np.median(frame_offsets))
    if float(np.max(np.abs(np.asarray(frame_offsets) - height_frame_offset))) > 0.002:
        raise RuntimeError("validated torso LUT base-height frame is inconsistent")

    return {
        "model": "reverse_upward_phase1_manifold_phase2_theta90",
        "source": "submission_static_validated_tables",
        "rows": rows,
        "z_upright_m": float(tables[1]["z_start_m"]),
        "z_min_m": float(min(row["z_target_m"] for row in rows)),
        "z_max_m": float(max(row["z_target_m"] for row in rows)),
        "z_boundary_m": REVERSE_UPWARD_Z_PHASE_BOUNDARY_M,
        "dz_step_m": REVERSE_UPWARD_Z_CM_STEP_M,
        "n_phase1": sum(row["phase"] == 1 for row in rows),
        "n_phase2": sum(row["phase"] == 2 for row in rows),
        "phase2_limited_by": tables[2].get("limited_by"),
        "height_frame_offset_m": height_frame_offset,
        "asset_digests": digests,
    }


def chest_height_lut_frame_m(trunk_q) -> float:
    lut = load_reverse_upward_lut()
    return float(
        torso_chest_height_local_m(trunk_q)
        + float(lut["height_frame_offset_m"])
    )


def _nearest_row_index(rows: list[dict[str, Any]], z_value: float) -> int:
    return int(
        min(
            range(len(rows)),
            key=lambda index: abs(
                float(rows[index]["z_target_m"]) - float(z_value)
            ),
        )
    )


def _nearest_joint_row_index(
    rows: list[dict[str, Any]],
    trunk_q: np.ndarray,
) -> tuple[int, float]:
    errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(row["trunk_q"], dtype=np.float64) - trunk_q
                )
            )
        )
        for row in rows
    ]
    index = int(np.argmin(errors))
    return index, errors[index]


def plan_absolute_height_trajectory(
    trunk_q,
    upward_m: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Plan the test-owned v2 1 cm LUT as absolute trunk joint targets."""
    start = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    if not np.all(np.isfinite(start)):
        raise ValueError("trunk proprioception contains non-finite values")
    start[3] = 0.0
    if bool(
        np.any(start < TRUNK_JOINT_LIMITS[:, 0] - 1e-4)
        or np.any(start > TRUNK_JOINT_LIMITS[:, 1] + 1e-4)
    ):
        raise ValueError("observed trunk proprioception is outside profile limits")
    requested = float(upward_m)
    if not math.isfinite(requested):
        raise ValueError("upward must be finite")

    lut = load_reverse_upward_lut()
    rows = lut["rows"]
    z_current = chest_height_lut_frame_m(start)
    z_target_raw = z_current + requested
    z_target_snapped = snap_z_to_nearest_cm(z_target_raw)
    z_target = float(
        np.clip(z_target_snapped, lut["z_min_m"], lut["z_max_m"])
    )
    z_clamped = abs(z_target - z_target_snapped) > 1e-6

    height_start_index = _nearest_row_index(rows, z_current)
    joint_start_index, joint_start_error = _nearest_joint_row_index(rows, start)
    starts_on_lut = bool(
        joint_start_error <= TRUNK_LUT_JOINT_MATCH_TOLERANCE_RAD
    )
    # LUT z labels and submission-local FK differ by up to a few millimeters at
    # the phase-2 limit.  When proprioception is already on a validated LUT row,
    # the joints are the authoritative progress marker.  Height matching remains
    # the fallback for evaluator reset postures that are not on the trajectory.
    start_index = joint_start_index if starts_on_lut else height_start_index
    end_index = _nearest_row_index(rows, z_target)
    lut_start_q = np.asarray(
        rows[start_index]["trunk_q"], dtype=np.float64
    )
    if start_index == end_index:
        selected_indices: list[int] = []
    elif start_index < end_index:
        first_index = start_index + (1 if starts_on_lut else 0)
        selected_indices = list(range(first_index, end_index + 1))
    else:
        first_index = start_index - (1 if starts_on_lut else 0)
        selected_indices = list(range(first_index, end_index - 1, -1))

    # The validated v2 tables are joint-position trajectories, not delta
    # templates. Translating them by the observed q start changes their FK and
    # can reverse the requested vertical motion on the evaluator reset branch.
    # A hold remains at the observed pose; every real move commands table rows
    # verbatim, with the custom trunk yaw joint locked at zero.
    if selected_indices:
        waypoints = []
        for index in selected_indices:
            lut_q = np.asarray(rows[index]["trunk_q"], dtype=np.float64).copy()
            lut_q[3] = 0.0
            waypoints.append(lut_q)
    else:
        held = start.copy()
        held[3] = 0.0
        waypoints = [held]

    target_q = waypoints[-1].copy()
    z_predicted = chest_height_lut_frame_m(target_q)
    direction = (
        "hold"
        if start_index == end_index
        else ("down" if end_index > start_index else "up")
    )
    requested_direction = (
        "hold"
        if abs(requested) <= 1e-12
        else ("down" if requested < 0.0 else "up")
    )
    saturated_hold = bool(
        direction == "hold" and requested_direction != "hold" and z_clamped
    )
    phases = (
        [int(rows[index]["phase"]) for index in selected_indices]
        if selected_indices
        else [int(rows[start_index]["phase"])]
    )
    meta: dict[str, Any] = {
        "ok": True,
        "model": lut["model"],
        "source": lut["source"],
        "direction": direction,
        "requested_direction": requested_direction,
        "upward_cmd_m": requested,
        "z_curr_m": z_current,
        "z_tgt_raw_m": z_target_raw,
        "z_tgt_snapped_m": z_target_snapped,
        "z_tgt_m": z_target,
        "z_predicted_m": z_predicted,
        "predicted_upward_m": z_predicted - z_current,
        "z_clamped": bool(z_clamped),
        "clamp_note": (
            None
            if not z_clamped
            else (
                f"requested height {z_target_raw:.3f}m exceeds LUT range "
                f"[{lut['z_min_m']:.3f}, {lut['z_max_m']:.3f}]m; "
                f"clamped to {z_target:.3f}m"
            )
        ),
        "z_min_m": lut["z_min_m"],
        "z_max_m": lut["z_max_m"],
        "z_boundary_m": lut["z_boundary_m"],
        "dz_step_m": lut["dz_step_m"],
        "phase2_limited_by": lut["phase2_limited_by"],
        "i_start": start_index,
        "i_start_height_nearest": height_start_index,
        "i_start_joint_nearest": joint_start_index,
        "i_end": end_index,
        "z_start_row_m": float(rows[start_index]["z_target_m"]),
        "z_end_row_m": float(rows[end_index]["z_target_m"]),
        "phases_used": phases,
        "n_waypoints": len(waypoints),
        "n_lut_rows": len(rows),
        "n_phase1": lut["n_phase1"],
        "n_phase2": lut["n_phase2"],
        "q_actual_start": start.astype(float).tolist(),
        "q_lut_start": lut_start_q.astype(float).tolist(),
        "q_lut_end": np.asarray(
            rows[end_index]["trunk_q"], dtype=np.float64
        ).astype(float).tolist(),
        "q_target": target_q.astype(float).tolist(),
        "clipped_waypoints": 0,
        "absolute_lut": True,
        "relative_lut": False,
        "holds_observed_pose": not bool(selected_indices),
        "starts_on_lut": starts_on_lut,
        "q_lut_match_error_rad": joint_start_error,
        "q_lut_match_tolerance_rad": TRUNK_LUT_JOINT_MATCH_TOLERANCE_RAD,
        "saturated_hold": saturated_hold,
        "connects_observed_pose_to_lut": bool(
            selected_indices and not starts_on_lut
        ),
        "trunk_yaw_locked": True,
        "asset_digests": dict(lut["asset_digests"]),
    }
    return waypoints, meta


def sample_lut_execution_waypoints(
    waypoints,
    phases,
    max_waypoints: int = TRUNK_LUT_MAX_EXEC_WAYPOINTS,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Select source LUT rows while minimizing the largest joint-space step."""
    source = [
        np.asarray(waypoint, dtype=np.float64).reshape(4).copy()
        for waypoint in waypoints
    ]
    limit = int(max_waypoints)
    if limit < 1:
        raise ValueError("max_waypoints must be positive")
    if not source:
        return [], {
            "policy": "minimax_joint_linf_with_phase_boundaries",
            "source_waypoint_count": 0,
            "execution_waypoint_count": 0,
            "max_execution_waypoints": limit,
            "sampled": False,
            "source_indices": [],
            "phase_transition_source_indices": [],
            "mandatory_source_indices": [],
            "execution_phases_used": [],
            "preserves_source_endpoint": True,
            "preserves_phase_boundaries": True,
            "source_adjacent_max_joint_step_rad": 0.0,
            "execution_max_joint_step_rad": 0.0,
        }

    phase_values = [int(phase) for phase in phases]
    if len(phase_values) != len(source):
        raise ValueError("phases must have one entry per LUT waypoint")

    transitions = [
        index
        for index in range(1, len(source))
        if phase_values[index] != phase_values[index - 1]
    ]
    mandatory = {0, len(source) - 1}
    for index in transitions:
        mandatory.update((index - 1, index))
    if len(mandatory) > limit:
        raise ValueError(
            "max_waypoints is too small to preserve all LUT phase boundaries"
        )

    if len(source) <= limit:
        selected_indices = list(range(len(source)))
    else:
        source_matrix = np.vstack(source)
        costs = np.full(
            (limit + 1, len(source)),
            np.inf,
            dtype=np.float64,
        )
        predecessors = np.full(
            (limit + 1, len(source)),
            -1,
            dtype=np.int64,
        )
        costs[1, 0] = 0.0
        for selected_count in range(2, limit + 1):
            for end_index in range(1, len(source)):
                for start_index in range(end_index):
                    previous_cost = float(
                        costs[selected_count - 1, start_index]
                    )
                    if not math.isfinite(previous_cost):
                        continue
                    if any(
                        start_index < required < end_index
                        for required in mandatory
                    ):
                        continue
                    step_cost = float(
                        np.max(
                            np.abs(
                                source_matrix[end_index]
                                - source_matrix[start_index]
                            )
                        )
                    )
                    candidate_cost = max(previous_cost, step_cost)
                    if candidate_cost < costs[selected_count, end_index]:
                        costs[selected_count, end_index] = candidate_cost
                        predecessors[selected_count, end_index] = start_index

        end_index = len(source) - 1
        if not math.isfinite(float(costs[limit, end_index])):
            raise RuntimeError("unable to sample LUT while preserving phase boundaries")
        selected_indices = []
        for selected_count in range(limit, 0, -1):
            selected_indices.append(end_index)
            if selected_count > 1:
                end_index = int(predecessors[selected_count, end_index])
        selected_indices.reverse()
        if selected_indices[0] != 0 or len(selected_indices) != limit:
            raise RuntimeError("LUT waypoint sampling did not produce the requested path")

    sampled = [source[index].copy() for index in selected_indices]
    source_matrix = np.vstack(source)
    sampled_matrix = np.vstack(sampled)
    source_max_step = (
        0.0
        if len(source) < 2
        else float(np.max(np.abs(np.diff(source_matrix, axis=0))))
    )
    execution_max_step = (
        0.0
        if len(sampled) < 2
        else float(np.max(np.abs(np.diff(sampled_matrix, axis=0))))
    )
    preserves_phase_boundaries = all(
        transition - 1 in selected_indices and transition in selected_indices
        for transition in transitions
    )
    return sampled, {
        "policy": "minimax_joint_linf_with_phase_boundaries",
        "source_waypoint_count": len(source),
        "execution_waypoint_count": len(sampled),
        "max_execution_waypoints": limit,
        "sampled": len(sampled) < len(source),
        "source_indices": selected_indices,
        "phase_transition_source_indices": transitions,
        "mandatory_source_indices": sorted(mandatory),
        "execution_phases_used": [
            phase_values[index] for index in selected_indices
        ],
        "preserves_source_endpoint": bool(
            selected_indices[-1] == len(source) - 1
        ),
        "preserves_phase_boundaries": bool(preserves_phase_boundaries),
        "source_adjacent_max_joint_step_rad": source_max_step,
        "execution_max_joint_step_rad": execution_max_step,
    }


__all__ = [
    "REVERSE_UPWARD_Z_CM_STEP_M",
    "REVERSE_UPWARD_Z_PHASE_BOUNDARY_M",
    "TRUNK_JOINT_LIMITS",
    "TRUNK_LUT_HOLD_ACTIONS",
    "TRUNK_LUT_MAX_EXEC_WAYPOINTS",
    "chest_height_lut_frame_m",
    "load_reverse_upward_lut",
    "plan_absolute_height_trajectory",
    "sample_lut_execution_waypoints",
    "snap_z_to_nearest_cm",
    "torso_chest_height_local_m",
    "torso_theta_z_deg",
]
