"""close_gripper

闭合指定 left/right 夹爪；不移动 EEF，完成后返回对应 wrist camera 视图。

How this tool works:
Ramp negative effort on the two fingers while checking measured positions and velocities. Detect each finger's contact/stall, hold for the confirmation window, and retain a negative carrying effort. Report measured evidence; inspect the returned wrist image to verify the object.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
from tools.util.action_constants import GRIPPER_CLOSE_UPPER_LIMIT_M
from tools.util.geometry.contract import ACTION_DIM
from tools.util.geometry.visualization_local import measure_wrist_opening_depth_evidence
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _set_result
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'close_gripper'
SCHEMA = {'type': 'object',
 'properties': {'arm': {'type': 'string', 'enum': ['right', 'left'], 'default': 'right'}},
 'additionalProperties': False,
 'required': ['arm']}


def actions(robot, arm: str = "right", **kwargs):
    arm_eff = str(arm).strip().lower()
    if arm_eff not in ("left", "right"):
        _set_result(
            robot,
            "close_gripper",
            {"ok": False, "error": "arm must be left or right"},
        )
        yield robot.hold_action()
        return
    timeout_s = float(
        kwargs.get("timeout_s", GRIPPER_CLOSE_DEFAULT_TIMEOUT_S)
    )
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        _set_result(
            robot,
            "close_gripper",
            {"ok": False, "error": "timeout_s must be positive and finite"},
        )
        yield robot.hold_action()
        return
    before = robot.gripper_qpos_list(arm_eff)
    before_qpos = np.asarray(before, dtype=np.float64).reshape(-1)
    before_qvel = np.asarray(
        robot.gripper_qvel_list(arm_eff),
        dtype=np.float64,
    ).reshape(-1)
    if (
        before_qpos.size != 2
        or before_qvel.size != 2
        or not np.all(np.isfinite(before_qpos))
        or not np.all(np.isfinite(before_qvel))
    ):
        _set_result(
            robot,
            "close_gripper",
            {
                "ok": False,
                "arm": arm_eff,
                "failure_stage": "gripper_proprioception",
                "error": "two finite gripper qpos and qvel values are required",
                "gripper_qpos_before": before,
                "implementation": f"official_action_{ACTION_DIM}d_gripper",
            },
        )
        yield robot.hold_action()
        return
    # A prior close may already be protecting an official constraint. Keep that
    # latch active until the first new negative command atomically replaces it;
    # an immediate cancellation must never create a zero-effort release frame.
    close_started_with_keepalive = bool(
        getattr(
            robot,
            "gripper_close_keepalive_active",
            lambda _arm: False,
        )(arm_eff)
    )
    latch_keepalive = getattr(
        robot,
        "latch_gripper_close_keepalive",
        None,
    )
    max_steps = max(1, int(math.ceil(timeout_s * CONTROL_HZ)))
    uses_effort = bool(
        getattr(robot, "gripper_uses_effort", lambda _arm: False)(arm_eff)
    )
    confirmation_window_s = (
        GRIPPER_CLOSE_BILATERAL_STABILITY_WINDOW_S
        if uses_effort
        else ASSISTED_GRASP_WINDOW_S + ASSISTED_GRASP_FALLBACK_MARGIN_S
    )
    confirmation_steps_required = int(
        math.ceil(confirmation_window_s * CONTROL_HZ)
    )
    bilateral_ramp_steps_required = max(
        1,
        int(
            math.ceil(
                GRIPPER_CLOSE_BILATERAL_RAMP_WINDOW_S * CONTROL_HZ
            )
        ),
    )
    single_contact_ramp_steps_required = max(
        1,
        int(math.ceil(GRIPPER_CLOSE_SINGLE_CONTACT_RAMP_S * CONTROL_HZ)),
    )
    action_steps = 0
    confirmation_steps = 0
    preload_steps = 0
    blocked = np.zeros(2, dtype=bool)
    moved = np.zeros(2, dtype=bool)
    blocked_without_prior_travel = np.zeros(2, dtype=bool)
    stall_count = np.zeros(2, dtype=np.int64)
    stall_anchor_qpos = before_qpos.copy()
    last_qpos = before_qpos.copy()
    bilateral_contact = False
    bilateral_contact_ever_observed = False
    grasp_window_completed = False
    failure_reason = "timeout"
    contact_qpos = [None, None]
    single_contact_ramp_steps = np.zeros(2, dtype=np.int64)
    confirmation_attempts = 0
    candidate_reset_count = 0
    confirmation_motion_restart_count = 0
    peak_seek_effort_n = 0.0
    peak_precontact_effort_n = 0.0
    peak_preload_effort_n = 0.0
    max_contact_closure_drift_m = 0.0
    max_contact_motion_m = 0.0
    max_abs_qpos_delta = 0.0
    max_closing_qpos_delta = 0.0
    peak_abs_qvel = float(np.max(np.abs(before_qvel)))
    full_open_depth_evidence: dict[str, Any] | None = None
    full_open_depth_evidence_attempts = 0
    valid_depth_evidence_attempts = 0
    full_open_depth_contact = False
    full_open_depth_contact_used = False
    full_open_depth_contact_mode: str | None = None
    full_open_occlusion_evidence_count = 0
    full_open_occlusion_evidence_side: str | None = None
    full_open_occlusion_signature: dict[str, Any] | None = None
    continuous_depth_evidence_count = 0
    depth_probe_stability_count = 0
    depth_probe_stability_anchor = before_qpos.copy()
    depth_probe_evidence_qpos: np.ndarray | None = None
    depth_probe_evidence_mode: str | None = None
    depth_contact_probe_mode: str | None = None
    depth_probe_last_attempt_step = -GRIPPER_CLOSE_STALL_STEPS
    latched_retry_recheck_completed = False

    def occlusion_signature(evidence: dict[str, Any]) -> dict[str, Any] | None:
        detail = evidence.get("occlusion_limited_candidate")
        if not isinstance(detail, dict):
            return None
        side = detail.get("visible_side")
        bridge = detail.get("bridge_component")
        boundary = detail.get("opposing_boundary_component")
        if side not in {"side_0", "side_1"}:
            return None
        if not isinstance(bridge, dict) or not isinstance(boundary, dict):
            return None
        try:
            signature = {
                "visible_side": str(side),
                "bridge_point_count": int(bridge["point_count"]),
                "bridge_center_point_count": int(
                    bridge["center_point_count"]
                ),
                "bridge_visible_side_point_count": int(
                    bridge[f"{side}_point_count"]
                ),
                "opposing_boundary_point_count": int(
                    boundary["point_count"]
                ),
                "bridge_inside_fraction": float(
                    detail["bridge_inside_fraction"]
                ),
                "bridge_q05": float(bridge["normalized_lateral_q05"]),
                "bridge_q95": float(bridge["normalized_lateral_q95"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
        numeric_values = [
            value
            for key, value in signature.items()
            if key != "visible_side"
        ]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            return None
        return signature

    def occlusion_signatures_match(
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        if previous["visible_side"] != current["visible_side"]:
            return False
        for key in (
            "bridge_point_count",
            "bridge_center_point_count",
            "bridge_visible_side_point_count",
            "opposing_boundary_point_count",
        ):
            before_value = float(previous[key])
            current_value = float(current[key])
            allowed_delta = max(
                8.0,
                GRIPPER_CLOSE_OCCLUSION_COUNT_REL_TOLERANCE
                * max(before_value, current_value),
            )
            if abs(before_value - current_value) > allowed_delta:
                return False
        if (
            abs(
                float(previous["bridge_inside_fraction"])
                - float(current["bridge_inside_fraction"])
            )
            > GRIPPER_CLOSE_OCCLUSION_FRACTION_TOLERANCE
        ):
            return False
        for key in ("bridge_q05", "bridge_q95"):
            if (
                abs(float(previous[key]) - float(current[key]))
                > GRIPPER_CLOSE_OCCLUSION_QUANTILE_TOLERANCE
            ):
                return False
        return True

    def close_action(command: np.ndarray):
        if uses_effort:
            if callable(latch_keepalive):
                latch_keepalive(arm_eff, effort=command.astype(float).tolist())
            return robot.make_action(
                **{f"gripper_effort_{arm_eff}": command.astype(float).tolist()}
            )
        if callable(latch_keepalive):
            latch_keepalive(arm_eff)
        return robot.make_action(**{f"gripper_{arm_eff}": [-1.0]})

    def linear_effort(
        start_n: float,
        target_n: float,
        step: int,
        total_steps: int,
    ) -> float:
        if total_steps <= 1:
            return float(target_n)
        progress = min(1.0, max(0.0, float(step) / float(total_steps - 1)))
        return float(start_n + (target_n - start_n) * progress)

    while action_steps < max_steps:
        _check_cancelled(robot, "close gripper")
        qpos = np.asarray(
            robot.gripper_qpos_list(arm_eff),
            dtype=np.float64,
        ).reshape(-1)
        qvel = np.asarray(
            robot.gripper_qvel_list(arm_eff),
            dtype=np.float64,
        ).reshape(-1)
        if (
            qpos.size != 2
            or qvel.size != 2
            or not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(qvel))
        ):
            failure_reason = "invalid_gripper_proprioception"
            break

        max_abs_qpos_delta = max(
            max_abs_qpos_delta,
            float(np.max(np.abs(qpos - before_qpos))),
        )
        max_closing_qpos_delta = max(
            max_closing_qpos_delta,
            float(np.max(np.maximum(before_qpos - qpos, 0.0))),
        )
        peak_abs_qvel = max(
            peak_abs_qvel,
            float(np.max(np.abs(qvel))),
        )
        moved |= (before_qpos - qpos) >= GRIPPER_CLOSE_MIN_FINGER_TRAVEL_M
        at_lower_limit = qpos <= GRIPPER_CLOSE_LOWER_LIMIT_MARGIN_M

        qpos_step_drift = float(np.max(np.abs(qpos - last_qpos)))
        qpos_anchor_drift = float(
            np.max(np.abs(qpos - depth_probe_stability_anchor))
        )
        if (
            qpos_step_drift < GRIPPER_CLOSE_DEPTH_PROBE_QPOS_DRIFT_M
            and qpos_anchor_drift < GRIPPER_CLOSE_DEPTH_PROBE_QPOS_DRIFT_M
        ):
            depth_probe_stability_count += 1
        else:
            depth_probe_stability_anchor = qpos.copy()
            depth_probe_stability_count = 1

        full_open_stable_probe = bool(
            uses_effort
            and action_steps >= GRIPPER_CLOSE_FULL_OPEN_PROBE_STEPS
            and depth_probe_stability_count
            >= GRIPPER_CLOSE_DEPTH_PROBE_STABILITY_STEPS
            and np.all(
                qpos
                >= GRIPPER_CLOSE_UPPER_LIMIT_M
                - GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
            )
            and max_abs_qpos_delta
            < GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
        )
        near_open_stable_probe = bool(
            uses_effort
            and action_steps >= GRIPPER_CLOSE_FULL_OPEN_PROBE_STEPS
            and depth_probe_stability_count
            >= GRIPPER_CLOSE_DEPTH_PROBE_STABILITY_STEPS
            and np.all(
                qpos
                >= GRIPPER_CLOSE_UPPER_LIMIT_M
                - GRIPPER_CLOSE_NEAR_OPEN_DEPTH_MARGIN_M
            )
            and max_closing_qpos_delta
            <= GRIPPER_CLOSE_NEAR_OPEN_DEPTH_MARGIN_M
            and not bool(np.any(at_lower_limit))
        )
        current_depth_probe_mode = (
            "full_open"
            if full_open_stable_probe
            else ("near_open" if near_open_stable_probe else None)
        )
        should_measure_wrist_depth = bool(
            current_depth_probe_mode is not None
            and not full_open_depth_contact
            and full_open_depth_evidence_attempts
            < GRIPPER_CLOSE_DEPTH_EVIDENCE_MAX_ATTEMPTS
            and action_steps - depth_probe_last_attempt_step
            >= GRIPPER_CLOSE_STALL_STEPS
        )
        if should_measure_wrist_depth:
            depth_probe_last_attempt_step = action_steps
            full_open_depth_evidence = _live_wrist_opening_depth_evidence(
                robot,
                arm_eff,
                qpos,
            )
            full_open_depth_evidence_attempts += 1
            if bool(full_open_depth_evidence.get("ok", False)):
                valid_depth_evidence_attempts += 1
            continuous_span = bool(
                full_open_depth_evidence.get(
                    "bilateral_depth_span_observed",
                    False,
                )
            )
            occlusion_limited = bool(
                full_open_depth_evidence.get(
                    "occlusion_limited_bilateral_blockage_observed",
                    False,
                )
            )
            current_occlusion_signature = occlusion_signature(
                full_open_depth_evidence
            )

            same_probe_geometry = bool(
                depth_probe_evidence_qpos is not None
                and depth_probe_evidence_mode == current_depth_probe_mode
                and float(
                    np.max(np.abs(qpos - depth_probe_evidence_qpos))
                )
                < GRIPPER_CLOSE_DEPTH_PROBE_QPOS_DRIFT_M
            )
            if not same_probe_geometry:
                continuous_depth_evidence_count = 0
                full_open_occlusion_evidence_count = 0
                full_open_occlusion_evidence_side = None
                full_open_occlusion_signature = None
                depth_probe_evidence_qpos = qpos.copy()
                depth_probe_evidence_mode = current_depth_probe_mode

            if continuous_span:
                continuous_depth_evidence_count += 1
                full_open_occlusion_evidence_count = 0
                full_open_occlusion_evidence_side = None
                full_open_occlusion_signature = None
                required_depth_frames = (
                    1
                    if current_depth_probe_mode == "full_open"
                    else GRIPPER_CLOSE_DEPTH_EVIDENCE_FRAMES
                )
                if continuous_depth_evidence_count >= required_depth_frames:
                    full_open_depth_contact = True
                    full_open_depth_contact_mode = "continuous_surface_span"
                    depth_contact_probe_mode = current_depth_probe_mode
            elif occlusion_limited and current_occlusion_signature is not None:
                continuous_depth_evidence_count = 0
                if (
                    full_open_occlusion_signature is not None
                    and occlusion_signatures_match(
                        full_open_occlusion_signature,
                        current_occlusion_signature,
                    )
                ):
                    full_open_occlusion_evidence_count += 1
                else:
                    full_open_occlusion_evidence_count = 1
                full_open_occlusion_signature = current_occlusion_signature
                full_open_occlusion_evidence_side = str(
                    current_occlusion_signature["visible_side"]
                )
                if (
                    full_open_occlusion_evidence_count
                    >= GRIPPER_CLOSE_OCCLUSION_EVIDENCE_FRAMES
                ):
                    full_open_depth_contact = True
                    full_open_depth_contact_mode = (
                        "stable_occlusion_limited_opposed_components"
                    )
                    depth_contact_probe_mode = current_depth_probe_mode
            else:
                continuous_depth_evidence_count = 0
                full_open_occlusion_evidence_count = 0
                full_open_occlusion_evidence_side = None
                full_open_occlusion_signature = None

        if bilateral_contact:
            contact_reference = np.asarray(
                [
                    qpos[i] if contact_qpos[i] is None else contact_qpos[i]
                    for i in range(2)
                ],
                dtype=np.float64,
            )
            contact_closure_drift = np.maximum(contact_reference - qpos, 0.0)
            contact_motion = np.abs(contact_reference - qpos)
            max_contact_closure_drift_m = max(
                max_contact_closure_drift_m,
                float(np.max(contact_closure_drift)),
            )
            max_contact_motion_m = max(
                max_contact_motion_m,
                float(np.max(contact_motion)),
            )
            contact_drift_limit_m = (
                GRIPPER_CLOSE_DEPTH_PROBE_QPOS_DRIFT_M
                if full_open_depth_contact_used
                else GRIPPER_CLOSE_CONTACT_DRIFT_M
            )
            moving_contact = (
                ~at_lower_limit
                & (contact_motion > contact_drift_limit_m)
            )
            if bool(np.any(at_lower_limit)):
                bilateral_contact = False
                failure_reason = "contact_collapsed_to_lower_limit"
                break
            if bool(np.any(moving_contact)):
                # Increasing preload commonly compresses or recenters a real
                # object by more than the stationary-window tolerance. That
                # motion invalidates elapsed *stability time*, not the fact
                # that both fingers were independently blocked. Preserve the
                # synchronized ramp and acquire a fresh stationary window at
                # the new qpos. If contact was actually lost, the freely
                # closing finger will reach its lower limit and fail below.
                contact_qpos = qpos.astype(float).tolist()
                confirmation_steps = 0
                confirmation_attempts += 1
                confirmation_motion_restart_count += 1
                if full_open_depth_contact_used:
                    full_open_depth_evidence = None
                    full_open_depth_evidence_attempts = 0
                    valid_depth_evidence_attempts = 0
                    full_open_depth_contact = False
                    full_open_depth_contact_used = False
                    full_open_depth_contact_mode = None
                    full_open_occlusion_evidence_count = 0
                    full_open_occlusion_evidence_side = None
                    full_open_occlusion_signature = None
                    continuous_depth_evidence_count = 0
                    depth_probe_evidence_qpos = None
                    depth_probe_evidence_mode = None
                    depth_contact_probe_mode = None
                    depth_probe_last_attempt_step = action_steps
                failure_reason = "contact_compliance_settling"
            else:
                if preload_steps > 0:
                    confirmation_steps += 1
                if confirmation_steps >= confirmation_steps_required:
                    grasp_window_completed = True
                    break

            confirmation_effort_n = linear_effort(
                GRIPPER_CLOSE_CONFIRM_EFFORT_N,
                GRIPPER_CLOSE_PRELOAD_EFFORT_N,
                preload_steps,
                bilateral_ramp_steps_required,
            )
            command = np.full(2, -confirmation_effort_n, dtype=np.float64)
            peak_preload_effort_n = max(
                peak_preload_effort_n,
                confirmation_effort_n,
            )
            yield close_action(command)
            action_steps += 1
            preload_steps += 1
            last_qpos = qpos.copy()
            continue

        for finger_i in range(2):
            if blocked[finger_i]:
                reference_qpos = (
                    qpos[finger_i]
                    if contact_qpos[finger_i] is None
                    else float(contact_qpos[finger_i])
                )
                closure_drift = max(
                    0.0,
                    float(reference_qpos - qpos[finger_i]),
                )
                contact_motion = abs(float(reference_qpos - qpos[finger_i]))
                max_contact_closure_drift_m = max(
                    max_contact_closure_drift_m,
                    closure_drift,
                )
                max_contact_motion_m = max(
                    max_contact_motion_m,
                    contact_motion,
                )
                resumed = contact_motion > GRIPPER_CLOSE_CONTACT_DRIFT_M
                if resumed or at_lower_limit[finger_i]:
                    blocked[finger_i] = False
                    blocked_without_prior_travel[finger_i] = False
                    stall_count[finger_i] = 0
                    contact_qpos[finger_i] = None
                    single_contact_ramp_steps[finger_i] = 0
                    stall_anchor_qpos[finger_i] = qpos[finger_i]
                continue

            full_open_blockage = bool(
                current_depth_probe_mode is not None
                and full_open_depth_contact
            )
            # The evaluator's instantaneous qvel is noisy enough to report
            # millimetres per second while consecutive qpos samples are
            # stationary. Once at least one close action has actually been
            # issued, a bounded stationary qpos window is direct blockage
            # evidence even when the object was already touching a finger at
            # its initial position. Requiring prior travel would make that
            # physically valid contact impossible to observe. qvel remains
            # diagnostic, while resumed qpos motion and the lower-limit check
            # below still reject a free-closing finger.
            is_stalled = bool(
                action_steps > 0
                and not at_lower_limit[finger_i]
                and abs(float(qpos[finger_i] - last_qpos[finger_i])) < 0.0005
                and abs(
                    float(qpos[finger_i] - stall_anchor_qpos[finger_i])
                )
                < GRIPPER_CLOSE_STALL_WINDOW_DRIFT_M
            )
            if is_stalled:
                if full_open_blockage:
                    # The probe itself already supplied the required history:
                    # both qpos values stayed at the open limit for this many
                    # low-effort actions, and live depth spans the whole gap.
                    stall_count[finger_i] = GRIPPER_CLOSE_STALL_STEPS
                else:
                    stall_count[finger_i] += 1
            else:
                stall_count[finger_i] = 0
                stall_anchor_qpos[finger_i] = qpos[finger_i]
            if stall_count[finger_i] >= GRIPPER_CLOSE_STALL_STEPS:
                blocked[finger_i] = True
                contact_qpos[finger_i] = float(qpos[finger_i])
                blocked_without_prior_travel[finger_i] = not moved[finger_i]
                single_contact_ramp_steps[finger_i] = 0

        if bool(np.all(blocked)):
            bilateral_contact = True
            bilateral_contact_ever_observed = True
            full_open_depth_contact_used = bool(
                current_depth_probe_mode is not None
                and full_open_depth_contact
            )
            confirmation_attempts += 1
            failure_reason = "bilateral_contact"
            confirmation_steps = 0
            single_contact_ramp_steps[:] = 0
            last_qpos = qpos.copy()
            continue
        if bool(np.all(at_lower_limit)):
            failure_reason = "finger_lower_limits"
            break

        # Reissuing close cannot create a fresh stroke while the previous
        # effort latch is still active. Once legal proprioception and wrist
        # depth have observed a stationary near-open state for a complete
        # recheck window, another 81 identical actions add no information.
        # Dynamic retries still retain the full caller-provided timeout.
        if (
            close_started_with_keepalive
            and uses_effort
            and action_steps >= GRIPPER_CLOSE_LATCHED_RECHECK_STEPS
            and depth_probe_stability_count
            >= GRIPPER_CLOSE_LATCHED_RECHECK_STEPS
            and valid_depth_evidence_attempts
            >= GRIPPER_CLOSE_DEPTH_EVIDENCE_FRAMES
            and current_depth_probe_mode is not None
            and not full_open_depth_contact
        ):
            failure_reason = "latched_retry_static_equilibrium"
            latched_retry_recheck_completed = True
            break

        if uses_effort:
            command = np.zeros(2, dtype=np.float64)
            for finger_i in range(2):
                if at_lower_limit[finger_i]:
                    command[finger_i] = 0.0
                elif blocked[finger_i]:
                    single_contact_effort_n = linear_effort(
                        GRIPPER_CLOSE_INITIAL_EFFORT_N,
                        GRIPPER_CLOSE_SINGLE_CONTACT_MAX_EFFORT_N,
                        int(single_contact_ramp_steps[finger_i]),
                        single_contact_ramp_steps_required,
                    )
                    command[finger_i] = -single_contact_effort_n
                    single_contact_ramp_steps[finger_i] += 1
                else:
                    seek_effort = min(
                        GRIPPER_CLOSE_MAX_SEEK_EFFORT_N,
                        GRIPPER_CLOSE_PRECONTACT_MAX_EFFORT_N,
                    )
                    command[finger_i] = -float(seek_effort)
            peak_seek_effort_n = max(
                peak_seek_effort_n,
                float(np.max(np.abs(command))),
            )
            peak_precontact_effort_n = max(
                peak_precontact_effort_n,
                float(np.max(np.abs(command))),
            )
        else:
            command = np.array([-1.0, -1.0], dtype=np.float64)
        yield close_action(command)
        action_steps += 1
        last_qpos = qpos.copy()

    fallback_effort_n = GRIPPER_CLOSE_INITIAL_EFFORT_N
    if grasp_window_completed:
        carry_command = np.full(
            2,
            -GRIPPER_CLOSE_CARRY_EFFORT_N,
            dtype=np.float64,
        )
        if callable(latch_keepalive):
            if uses_effort:
                latch_keepalive(arm_eff, effort=carry_command.tolist())
            else:
                latch_keepalive(arm_eff)
    else:
        if (
            uses_effort
            and bilateral_contact_ever_observed
            and failure_reason
            not in {"finger_lower_limits", "contact_collapsed_to_lower_limit"}
        ):
            # Do not undo an in-progress bilateral preload at the timeout
            # boundary. Preserve the last level reached by the monotone ramp;
            # explicit open remains the only release operation. A finger that
            # subsequently reached its lower limit disproved the candidate,
            # so that case falls back to the low-force close latch below.
            fallback_effort_n = linear_effort(
                GRIPPER_CLOSE_CONFIRM_EFFORT_N,
                GRIPPER_CLOSE_PRELOAD_EFFORT_N,
                max(0, preload_steps - 1),
                bilateral_ramp_steps_required,
            )
        fallback_command = np.full(
            2,
            -fallback_effort_n,
            dtype=np.float64,
        )
        if uses_effort:
            yield close_action(fallback_command)
        if callable(latch_keepalive):
            if uses_effort:
                latch_keepalive(
                    arm_eff,
                    effort=fallback_command.tolist(),
                )
            else:
                latch_keepalive(arm_eff)

    after = robot.gripper_qpos_list(arm_eff)
    after_qvel = robot.gripper_qvel_list(arm_eff)
    after_qpos = np.asarray(after, dtype=np.float64).reshape(-1)
    if (
        after_qpos.size == before_qpos.size
        and after_qpos.size > 0
        and np.all(np.isfinite(after_qpos))
        and np.all(np.isfinite(before_qpos))
    ):
        max_abs_qpos_delta = max(
            max_abs_qpos_delta,
            float(np.max(np.abs(after_qpos - before_qpos))),
        )
        max_closing_qpos_delta = max(
            max_closing_qpos_delta,
            float(
                np.max(
                    np.maximum(before_qpos - after_qpos, 0.0)
                )
            ),
        )
    actuation_detected = bool(
        max_closing_qpos_delta >= GRIPPER_CLOSE_MIN_ACTUATION_DELTA
    )
    started_away_from_closed_limit = bool(
        np.all(before_qpos > GRIPPER_CLOSE_LOWER_LIMIT_MARGIN_M)
    )
    failure_stage = (
        None
        if grasp_window_completed
        else (
            "empty_close"
            if failure_reason in {
                "finger_lower_limits",
                "contact_collapsed_to_lower_limit",
            }
            else (
                "bilateral_contact_confirmation"
                if bilateral_contact_ever_observed
                else (
                    "gripper_actuation"
                    if started_away_from_closed_limit
                    and not actuation_detected
                    else "bilateral_contact_confirmation"
                )
            )
        )
    )
    commanded_action = (
        [-GRIPPER_CLOSE_CARRY_EFFORT_N] * 2
        if grasp_window_completed and uses_effort
        else (
            [-fallback_effort_n] * 2
            if uses_effort
            else [-1.0]
        )
    )
    keepalive_active = bool(
        getattr(
            robot,
            "gripper_close_keepalive_active",
            lambda _arm: False,
        )(arm_eff)
    )
    _set_result(
        robot,
        "close_gripper",
        {
            "ok": grasp_window_completed,
            "arm": arm_eff,
            "gripper_qpos_before": before,
            "gripper_qpos_after": after,
            "gripper_qvel_before": before_qvel.tolist(),
            "gripper_qvel_after": after_qvel,
            "commanded_gripper_action": commanded_action,
            "control_mode": "independent_effort" if uses_effort else "position",
            "per_finger_contact_qpos_m": contact_qpos,
            "per_finger_blocked": blocked.astype(bool).tolist(),
            "per_finger_blocked_without_prior_travel": (
                blocked_without_prior_travel.astype(bool).tolist()
            ),
            "stationary_under_close_command_counts_as_contact": True,
            "contact_evidence_policy": (
                "four_frame_qpos_stall_after_issued_close_command_"
                "with_no_prior_travel_requirement"
                if uses_effort
                else "official_negative_close_qpos_stall"
            ),
            "peak_seek_effort_n": peak_seek_effort_n if uses_effort else None,
            "peak_precontact_effort_n": (
                peak_precontact_effort_n if uses_effort else None
            ),
            "precontact_effort_policy": (
                "fixed_0.5N_seek_then_single_contact_linear_ramp_to_2N"
                if uses_effort
                else "official_negative_close"
            ),
            "single_contact_ramp_start_effort_n": (
                GRIPPER_CLOSE_INITIAL_EFFORT_N if uses_effort else None
            ),
            "single_contact_ramp_target_effort_n": (
                GRIPPER_CLOSE_SINGLE_CONTACT_MAX_EFFORT_N
                if uses_effort
                else None
            ),
            "single_contact_ramp_steps": (
                single_contact_ramp_steps.astype(int).tolist()
                if uses_effort
                else None
            ),
            "single_contact_ramp_steps_required": (
                single_contact_ramp_steps_required if uses_effort else None
            ),
            "peak_preload_effort_n": (
                peak_preload_effort_n if uses_effort else None
            ),
            "peak_confirmation_effort_n": (
                peak_preload_effort_n if uses_effort else None
            ),
            "preload_effort_n": (
                GRIPPER_CLOSE_PRELOAD_EFFORT_N if uses_effort else None
            ),
            "confirmation_effort_n": (
                GRIPPER_CLOSE_CONFIRM_EFFORT_N if uses_effort else None
            ),
            "carry_effort_n": (
                GRIPPER_CLOSE_CARRY_EFFORT_N if uses_effort else None
            ),
            "bilateral_effort_policy": (
                "synchronized_linear_1N_to_20N_over_9_frames_then_"
                "full_effort_stability_inside_12_frame_confirmation_"
                "then_persistent_20N_carry_until_explicit_open"
                if uses_effort
                else "official_negative_close"
            ),
            "bilateral_ramp_steps_required": (
                bilateral_ramp_steps_required if uses_effort else None
            ),
            "full_effort_confirmation_steps_required": (
                max(
                    1,
                    confirmation_steps_required
                    - bilateral_ramp_steps_required,
                )
                if uses_effort
                else None
            ),
            "bilateral_stability_window_s": (
                GRIPPER_CLOSE_BILATERAL_STABILITY_WINDOW_S
                if uses_effort
                else ASSISTED_GRASP_WINDOW_S
                + ASSISTED_GRASP_FALLBACK_MARGIN_S
            ),
            "actuation_detected": actuation_detected,
            "max_abs_qpos_delta": max_abs_qpos_delta,
            "max_closing_qpos_delta": max_closing_qpos_delta,
            "peak_abs_qvel": peak_abs_qvel,
            "failure_stage": failure_stage,
            "bilateral_contact_candidate": bilateral_contact,
            "bilateral_contact_ever_observed": bilateral_contact_ever_observed,
            "grasp_confirmed": grasp_window_completed,
            "assisted_grasp_window_completed": grasp_window_completed,
            "constraint_directly_observed": False,
            "confirmation_source": (
                (
                    "legal_gripper_qpos_plus_wrist_depth_"
                    f"{depth_contact_probe_mode}_occlusion_window"
                    if full_open_depth_contact_mode
                    == "stable_occlusion_limited_opposed_components"
                    else (
                        "legal_gripper_qpos_plus_wrist_depth_"
                        f"{depth_contact_probe_mode}_blockage_window"
                    )
                )
                if full_open_depth_contact_used
                else "legal_gripper_qpos_bilateral_stall_window"
            ),
            "full_open_probe_steps_required": (
                GRIPPER_CLOSE_FULL_OPEN_PROBE_STEPS if uses_effort else None
            ),
            "full_open_depth_evidence_attempts": (
                full_open_depth_evidence_attempts if uses_effort else None
            ),
            "valid_wrist_depth_evidence_attempts": (
                valid_depth_evidence_attempts if uses_effort else None
            ),
            "full_open_depth_contact_observed": (
                bool(
                    full_open_depth_contact
                    and depth_contact_probe_mode == "full_open"
                )
                if uses_effort
                else None
            ),
            "full_open_depth_contact_used": (
                bool(
                    full_open_depth_contact_used
                    and depth_contact_probe_mode == "full_open"
                )
                if uses_effort
                else None
            ),
            "full_open_depth_contact_mode": (
                (
                    full_open_depth_contact_mode
                    if depth_contact_probe_mode == "full_open"
                    else None
                )
                if uses_effort
                else None
            ),
            "full_open_occlusion_evidence_count": (
                full_open_occlusion_evidence_count if uses_effort else None
            ),
            "full_open_occlusion_evidence_frames_required": (
                GRIPPER_CLOSE_OCCLUSION_EVIDENCE_FRAMES
                if uses_effort
                else None
            ),
            "full_open_depth_evidence_max_attempts": (
                GRIPPER_CLOSE_DEPTH_EVIDENCE_MAX_ATTEMPTS
                if uses_effort
                else None
            ),
            "full_open_occlusion_evidence_side": (
                full_open_occlusion_evidence_side if uses_effort else None
            ),
            "full_open_occlusion_signature": (
                full_open_occlusion_signature if uses_effort else None
            ),
            "wrist_opening_depth_evidence": full_open_depth_evidence,
            "wrist_depth_contact_observed": (
                full_open_depth_contact if uses_effort else None
            ),
            "wrist_depth_contact_used": (
                full_open_depth_contact_used if uses_effort else None
            ),
            "wrist_depth_contact_mode": (
                full_open_depth_contact_mode if uses_effort else None
            ),
            "wrist_depth_contact_probe_mode": (
                depth_contact_probe_mode if uses_effort else None
            ),
            "wrist_depth_evidence_count": (
                max(
                    continuous_depth_evidence_count,
                    full_open_occlusion_evidence_count,
                )
                if uses_effort
                else None
            ),
            "wrist_depth_evidence_frames_required": (
                GRIPPER_CLOSE_DEPTH_EVIDENCE_FRAMES
                if uses_effort
                else None
            ),
            "near_open_depth_contact_observed": (
                bool(
                    full_open_depth_contact
                    and depth_contact_probe_mode == "near_open"
                )
                if uses_effort
                else None
            ),
            "near_open_depth_contact_used": (
                bool(
                    full_open_depth_contact_used
                    and depth_contact_probe_mode == "near_open"
                )
                if uses_effort
                else None
            ),
            "near_open_depth_margin_m": (
                GRIPPER_CLOSE_NEAR_OPEN_DEPTH_MARGIN_M
                if uses_effort
                else None
            ),
            "depth_probe_stability_steps": (
                depth_probe_stability_count if uses_effort else None
            ),
            "depth_probe_stability_steps_required": (
                GRIPPER_CLOSE_DEPTH_PROBE_STABILITY_STEPS
                if uses_effort
                else None
            ),
            "close_started_with_keepalive": close_started_with_keepalive,
            "latched_retry_recheck_completed": (
                latched_retry_recheck_completed if uses_effort else None
            ),
            "latched_retry_recheck_steps": (
                GRIPPER_CLOSE_LATCHED_RECHECK_STEPS
                if uses_effort
                else None
            ),
            "actuation_evidence_source": "gripper_qpos_closing_displacement",
            "qvel_used_for_actuation_detection": False,
            "qvel_used_for_contact_detection": False,
            "per_finger_moved": moved.astype(bool).tolist(),
            "per_finger_stall_count": stall_count.astype(int).tolist(),
            "confirmation_steps": confirmation_steps,
            "confirmation_steps_required": confirmation_steps_required,
            "preload_steps": preload_steps,
            "contact_hold_steps": preload_steps,
            "confirmation_attempts": confirmation_attempts,
            "candidate_reset_count": candidate_reset_count,
            "confirmation_motion_restart_count": (
                confirmation_motion_restart_count if uses_effort else None
            ),
            "max_contact_closure_drift_m": max_contact_closure_drift_m,
            "max_contact_motion_m": max_contact_motion_m,
            "failure_reason": failure_reason,
            "action_steps": action_steps,
            "timeout_s": timeout_s,
            "timed_out": not grasp_window_completed and action_steps >= max_steps,
            "close_keepalive_active": keepalive_active,
            "close_intent_persists_until_explicit_open": keepalive_active,
            "implementation": f"official_action_{ACTION_DIM}d_gripper",
            **(
                {}
                if grasp_window_completed
                else {
                    "error": (
                        (
                            "close was already latched; a complete legal "
                            "proprioception and wrist-depth recheck observed "
                            "the same stationary near-open state without "
                            "bilateral blockage; explicit open is required "
                            "before another closing stroke"
                            if latched_retry_recheck_completed
                            else (
                                "close command produced no observable finger "
                                "displacement and no stable wrist-depth evidence "
                                "of a surface blocking the open or near-open "
                                "finger gap; close effort remains latched "
                                "until explicit open"
                            )
                        )
                        if failure_stage == "gripper_actuation"
                        else (
                            "both fingers reached their lower limits without "
                            "bilateral object contact"
                            if failure_stage == "empty_close"
                            else (
                                (
                                    "bilateral finger blockage was observed but "
                                    "did not remain stable for the complete "
                                    "assisted-grasp confirmation window; close "
                                    "remains latched until explicit open"
                                    if bilateral_contact_ever_observed
                                    else (
                                        "both fingers did not remain "
                                        "independently blocked away from their "
                                        "lower limits for the assisted-grasp "
                                        "confirmation window; close effort "
                                        "remains latched until explicit open"
                                    )
                                )
                            )
                        )
                    )
                }
            ),
        },
    )
    yield robot.hold_action()


GRIPPER_CLOSE_DEFAULT_TIMEOUT_S = 3.5



GRIPPER_CLOSE_LOWER_LIMIT_MARGIN_M = 0.002



GRIPPER_CLOSE_INITIAL_EFFORT_N = 0.5



GRIPPER_CLOSE_MAX_SEEK_EFFORT_N = 0.5



GRIPPER_CLOSE_PRECONTACT_MAX_EFFORT_N = GRIPPER_CLOSE_INITIAL_EFFORT_N



GRIPPER_CLOSE_SINGLE_CONTACT_MAX_EFFORT_N = 2.0



GRIPPER_CLOSE_SINGLE_CONTACT_RAMP_S = 0.5



GRIPPER_CLOSE_CONFIRM_EFFORT_N = 1.0



GRIPPER_CLOSE_PRELOAD_EFFORT_N = 20.0



GRIPPER_CLOSE_CARRY_EFFORT_N = GRIPPER_CLOSE_PRELOAD_EFFORT_N



GRIPPER_CLOSE_BILATERAL_STABILITY_WINDOW_S = 0.4



GRIPPER_CLOSE_BILATERAL_RAMP_WINDOW_S = 0.3



GRIPPER_CLOSE_MIN_FINGER_TRAVEL_M = 0.0008



GRIPPER_CLOSE_STALL_STEPS = 4



GRIPPER_CLOSE_STALL_WINDOW_DRIFT_M = 0.0005



GRIPPER_CLOSE_CONTACT_DRIFT_M = 0.0005



GRIPPER_CLOSE_MIN_ACTUATION_DELTA = 1.0e-4



GRIPPER_CLOSE_FULL_OPEN_PROBE_STEPS = GRIPPER_CLOSE_STALL_STEPS



GRIPPER_CLOSE_NEAR_OPEN_DEPTH_MARGIN_M = 0.004



GRIPPER_CLOSE_DEPTH_PROBE_QPOS_DRIFT_M = 0.0005



GRIPPER_CLOSE_DEPTH_PROBE_STABILITY_STEPS = GRIPPER_CLOSE_STALL_STEPS



GRIPPER_CLOSE_DEPTH_EVIDENCE_FRAMES = 3



GRIPPER_CLOSE_LATCHED_RECHECK_STEPS = 24



GRIPPER_CLOSE_OCCLUSION_EVIDENCE_FRAMES = 3



GRIPPER_CLOSE_DEPTH_EVIDENCE_MAX_ATTEMPTS = 5



GRIPPER_CLOSE_OCCLUSION_COUNT_REL_TOLERANCE = 0.25



GRIPPER_CLOSE_OCCLUSION_QUANTILE_TOLERANCE = 0.08



GRIPPER_CLOSE_OCCLUSION_FRACTION_TOLERANCE = 0.08



ASSISTED_GRASP_WINDOW_S = 0.3



ASSISTED_GRASP_FALLBACK_MARGIN_S = 0.1



def _live_wrist_opening_depth_evidence(
    ctx,
    arm: str,
    gripper_qpos: np.ndarray,
) -> dict[str, Any]:
    role = f"{arm}_wrist"
    adapter = getattr(ctx.world, "_official_adapter", None)
    if adapter is None:
        return {
            "ok": False,
            "bilateral_depth_span_observed": False,
            "error": "official evaluator adapter is unavailable",
        }
    try:
        depth = adapter.camera_depth_frame(role)
        relative_pose = adapter.camera_relative_poses().get(role)
        if depth is None:
            raise ValueError(f"{role} depth_linear is unavailable")
        if relative_pose is None:
            raise ValueError(f"{role} relative pose is unavailable")
        depth_array = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_array.ndim != 2:
            raise ValueError(
                f"{role} depth_linear must be HxW, got {depth_array.shape}"
            )
        height, width = depth_array.shape
        camera = {
            **ctx.world.local_pose_from_robot_relative(
                relative_pose["pos"],
                relative_pose["quat"],
            ),
            **_camera_intrinsics(role, width, height),
        }
        return measure_wrist_opening_depth_evidence(
            depth_linear=depth_array,
            camera=camera,
            eef_pose=ctx.world.eef_pose(arm),
            gripper_qpos=gripper_qpos,
        )
    except Exception as exc:
        return {
            "ok": False,
            "bilateral_depth_span_observed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
