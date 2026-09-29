"""exec_eef_pose_v2 —— 仅将 EEF 移到 plan 的 eef_pose（无 next_move、不合爪）。

入参与 exec_move_v2 相同（session_id, plan_id, arm?, back_m?），
读取同一 plan record，只用 eef_target.pos/quat：
  当前 EEF → safe → contact；press_point 直接回放规划阶段 GPU IK anchors
忽略 next_eef_move 与 gripper_cmd；合爪由用户手动完成。

与 plan_grasp_point / plan_grasp_object 落盘格式兼容。
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Generator, Optional

import math

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill
from behavior_interface.skills.exec_move_v2 import (
    _candidate_with_record_root,
    _compensate_candidate_for_object_drift,
    _pick_better_arm,
    log_final_eef_error,
    log_plan_eef_input,
)


_EXEC_EEF_POSE_MAX_ATTEMPTS = 3
_EXEC_EEF_POSE_RETRY_HOLD_FRAMES = 2


def _strict_filtered_plan_exec_arm(
    cand: Dict[str, Any],
    record_arm: str,
    requested_arm: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the arm for plans whose final IK branch was validated at plan time."""
    from behavior_interface.skills.eef import (
        _candidate_requires_stored_final_q,
        _stored_filter_q_for_arm,
    )

    if not _candidate_requires_stored_final_q(cand):
        return None, None
    arm = str(requested_arm or record_arm or "").lower().strip()
    if arm not in ("left", "right"):
        return None, "严格 IK 计划没有有效的规划手臂"
    if _stored_filter_q_for_arm(cand, arm) is None:
        return None, f"严格 IK 计划没有 arm={arm} 的存储最终关节解，拒绝换手执行"
    return arm, None


def _exec_eef_pose_result_status(
    result: Any,
) -> tuple[bool, Optional[str], Optional[float]]:
    exec_ok = True
    err_msg = None
    moved_f = None
    if not isinstance(result, dict):
        return exec_ok, err_msg, moved_f

    exec_ok = bool(result.get("ok", True))
    err_msg = result.get("error")
    if exec_ok and (ferr := result.get("final_pos_err_m", result.get("cnt_err"))) is not None:
        try:
            ferr_f = float(ferr)
        except (TypeError, ValueError):
            ferr_f = None
        if ferr_f is not None and math.isfinite(ferr_f) and ferr_f > 0.06:
            exec_ok = False
            err_msg = f"末端未到达规划位姿（位置偏差 {ferr_f * 1000:.0f}mm）"
    if exec_ok and (ori := result.get("final_ori_err_deg")) is not None:
        try:
            ori_f = float(ori)
        except (TypeError, ValueError):
            ori_f = None
        app = result.get("final_approach_err_deg")
        try:
            app_f = float(app) if app is not None else ori_f
        except (TypeError, ValueError):
            app_f = ori_f
        if (
            ori_f is not None
            and math.isfinite(ori_f)
            and ori_f > 40.0
            and app_f is not None
            and math.isfinite(app_f)
            and app_f > 40.0
        ):
            exec_ok = False
            err_msg = (
                f"末端未到达规划位姿（姿态偏差 {ori_f:.1f}°，"
                f"夹爪指向偏差 {app_f:.1f}°）"
            )
    moved = result.get("object_moved_dist")
    try:
        moved_f = float(moved) if moved is not None else None
    except (TypeError, ValueError):
        moved_f = None
    return bool(exec_ok), err_msg, moved_f


def _exec_eef_pose_result_retryable(result: Any, *, exec_ok: bool) -> bool:
    if exec_ok or not isinstance(result, dict):
        return False
    if result.get("safe_plan_failed"):
        return False
    status = str(result.get("status") or result.get("reason") or "").strip().lower()
    if status in ("stuck_or_collision", "collision", "force_estop"):
        return False
    error = str(result.get("error") or "")
    non_retryable_markers = (
        "离线规划失败",
        "超出当前肩膀明显可达范围",
        "requires stored",
        "没有有效的规划手臂",
        "没有 arm=",
        "J8 reset failed",
    )
    if any(marker in error for marker in non_retryable_markers):
        return False
    return any(
        key in result
        for key in (
            "pre_err",
            "cnt_err",
            "final_pos_err_m",
            "final_ori_err_deg",
            "final_approach_err_deg",
        )
    )


def _exec_eef_pose_attempt_summary(
    attempt: int,
    result: Any,
    *,
    exec_ok: bool,
    retryable: bool,
    err_msg: Optional[str],
) -> dict:
    summary = {
        "attempt": int(attempt),
        "ok": bool(exec_ok),
        "retryable": bool(retryable),
        "error": err_msg,
    }
    if isinstance(result, dict):
        for key in (
            "pre_err",
            "cnt_err",
            "final_pos_err_m",
            "final_ori_err_deg",
            "final_approach_err_deg",
        ):
            if key in result:
                summary[key] = result.get(key)
    return summary


def _run_exec_eef_pose_attempts(
    ctx,
    execute_once: Callable[[int], Generator],
    *,
    max_attempts: int = _EXEC_EEF_POSE_MAX_ATTEMPTS,
):
    max_attempts = max(1, int(max_attempts))
    attempts: list[dict] = []
    result = None
    exec_ok = False
    err_msg = None
    moved_f = None
    for attempt in range(1, max_attempts + 1):
        ctx.log(
            f"[exec_eef_pose_v2] exec attempt {attempt}/{max_attempts}: "
            "replay current→safe→final with identical parameters"
        )
        result = yield from execute_once(attempt)
        exec_ok, err_msg, moved_f = _exec_eef_pose_result_status(result)
        retryable = _exec_eef_pose_result_retryable(result, exec_ok=exec_ok)
        attempts.append(
            _exec_eef_pose_attempt_summary(
                attempt,
                result,
                exec_ok=exec_ok,
                retryable=retryable,
                err_msg=err_msg,
            )
        )
        if exec_ok or not retryable or attempt >= max_attempts:
            break
        ctx.log(
            f"[exec_eef_pose_v2] attempt {attempt}/{max_attempts} failed: "
            f"{err_msg or 'trajectory execution did not reach final'}; "
            "restart exec with identical parameters"
        )
        for _ in range(_EXEC_EEF_POSE_RETRY_HOLD_FRAMES):
            yield ctx.world.hold_action()
    return result, exec_ok, err_msg, moved_f, attempts


@register_skill(
    "exec_eef_pose_v2",
    description=(
        "执行 plan_id：仅移动到 eef_pose，忽略 next_eef_move，不合爪。"
        "当前→safe→contact；press_point 回放规划阶段 GPU IK anchors。"
    ),
)
def exec_eef_pose_v2(
    ctx,
    session_id: str,
    plan_id: str,
    arm: Optional[str] = None,
    back_m: float = 0.10,
    stop_after_safe: bool = False,
    reset_tool_roll_at_start: bool = False,
) -> Generator:
    from behavior_interface.skills.eef import (
        _current_gripper_qpos_cmd,
        _ensure_world_pinned_actions,
        _execute_one_eef,
        _reset_selected_tool_roll_to_zero,
    )
    from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims

    world = ctx.world
    _ensure_world_pinned_actions(world)
    clear_plan_viz_prims()

    try:
        record = agent_runs.load_plan_record(session_id, plan_id)
    except FileNotFoundError as e:
        ctx.set_result({"ok": False, "error": str(e)})
        yield world.hold_action()
        return

    cand: Dict[str, Any] = record.get("candidate") or {}
    if not cand.get("eef_target"):
        ctx.set_result({"ok": False, "error": f"plan {plan_id} 无可执行 candidate"})
        yield world.hold_action()
        return

    if cand.get("reachable") is False:
        reason = cand.get("reach_reason") or "肩距不可达"
        ctx.log(
            f"[exec_eef_pose_v2] WARN plan={plan_id}: 规划标注不可达 ({reason})，仍尝试执行"
        )

    object_name = (
        record.get("object_name")
        or cand.get("meta", {}).get("object_name")
        or ""
    )
    cand = _candidate_with_record_root(cand, record)
    cand = _compensate_candidate_for_object_drift(world, cand, object_name, ctx)

    record_arm = record.get("arm") or cand.get("arm") or "right"
    arm_req = (arm or "").lower().strip() if isinstance(arm, str) else None
    target_pos_pick = (cand.get("eef_target") or {}).get("pos") or [0.0, 0.0, 0.0]
    strict_arm, strict_arm_error = _strict_filtered_plan_exec_arm(
        cand, record_arm, arm_req
    )
    if strict_arm_error:
        ctx.set_result({
            "ok": False,
            "error": strict_arm_error,
            "plan_id": plan_id,
            "record_arm": record_arm,
            "requested_arm": arm_req,
        })
        yield world.hold_action()
        return
    if strict_arm is not None:
        arm_eff = strict_arm
        ctx.log(
            f"[exec_eef_pose_v2] 严格 IK 计划锁定 arm={arm_eff}，"
            "复用规划阶段验证的最终关节解"
        )
    elif arm_req in ("left", "right"):
        arm_eff = arm_req
        ctx.log(f"[exec_eef_pose_v2] 使用指定 arm={arm_eff}")
    else:
        from behavior_interface.skills.plan_grasp_object import _is_shoulder_reachable

        tp = np.asarray(target_pos_pick, dtype=np.float64).reshape(3)
        sh = world.shoulder_pose(arm=record_arm)
        shoulder = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
        ok_plan, reason_plan = _is_shoulder_reachable(tp, shoulder)
        if ok_plan:
            arm_eff = record_arm
            ctx.log(f"[exec_eef_pose_v2] 使用规划臂 arm={arm_eff} ({reason_plan})")
        else:
            arm_eff = _pick_better_arm(world, target_pos_pick, record_arm, ctx)
            ctx.log(
                f"[exec_eef_pose_v2] 规划臂 {record_arm} 不可达({reason_plan}) "
                f"→ 自动换 arm={arm_eff}"
            )

    tool_roll_reset = None
    if bool(reset_tool_roll_at_start):
        try:
            tool_roll_reset = yield from _reset_selected_tool_roll_to_zero(
                world,
                arm_eff,
                ctx=ctx,
                stage_name="exec_plan_pose.start",
            )
            if not tool_roll_reset.get("ok", True):
                raise RuntimeError(
                    f"J8 controller reset did not converge: {tool_roll_reset}"
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

    last = {
        "ok": True,
        "target": cand.get("target", "grasp"),
        "candidates": [cand],
        "object": {
            "input": object_name,
        },
    }

    exec_seq = (cand.get("meta") or {}).get("exec_sequence")
    ctx.log(
        f"[exec_eef_pose_v2] plan={plan_id} skill={record.get('skill')} "
        f"target={cand.get('target')} exec_seq={exec_seq} "
        f"(忽略 next_eef_move；不合爪)"
    )
    if cand.get("target") == "move_eef" or exec_seq == "move_only":
        from behavior_interface.skills.eef import (
            _stored_press_gpu_move_trajectory,
        )

        stored_press_trajectory = _stored_press_gpu_move_trajectory(
            cand, arm_eff
        )
        if stored_press_trajectory is not None:
            stored_back_m = float(
                stored_press_trajectory.get("back_m", back_m)
            )
            ctx.log(
                "[exec_eef_pose_v2] 回放规划阶段 GPU "
                f"current→safe→press anchors，back={stored_back_m*100:.0f}cm"
            )
        else:
            ctx.log(
                f"[exec_eef_pose_v2] back_m={back_m*100:.0f}cm "
                "(legacy move_only 当前→目标直线分支)"
            )
    else:
        ctx.log(f"[exec_eef_pose_v2] back_m={back_m*100:.0f}cm")
    if stop_after_safe:
        ctx.log("[exec_eef_pose_v2] stop_after_safe=True，仅到 safe pose 后停止")

    lock_gripper_cmd = _current_gripper_qpos_cmd(world, arm_eff)
    ctx.log(
        f"[exec_eef_pose_v2] 锁定夹爪原位置 arm={arm_eff} "
        f"finger_qpos={lock_gripper_cmd if lock_gripper_cmd is not None else 'n/a'}；"
        "不执行开爪/合爪"
    )
    input_pose_report = log_plan_eef_input(
        ctx, prefix="exec_eef_pose_v2", plan_id=plan_id, arm=arm_eff, cand=cand,
    )

    def execute_once(_attempt: int):
        attempt_cand = copy.deepcopy(cand)
        attempt_last = dict(last)
        attempt_last["candidates"] = [attempt_cand]
        return (yield from _execute_one_eef(
            ctx, attempt_last, attempt_cand, arm_eff,
            back_m=float(back_m),
            skip_next_move=True,
            skip_grip_close=True,
            stop_after_safe=bool(stop_after_safe),
            lock_gripper_cmd=copy.deepcopy(lock_gripper_cmd),
        ))

    try:
        result, exec_ok, err_msg, moved_f, exec_attempts = yield from (
            _run_exec_eef_pose_attempts(ctx, execute_once)
        )
    except Exception as e:
        import traceback
        ctx.log(f"[exec_eef_pose_v2] 执行异常: {e}\n{traceback.format_exc()}")
        ctx.set_result({
            "ok": False, "error": f"exec 异常: {e}", "plan_id": plan_id,
        })
        yield world.hold_action()
        return

    from behavior_interface.v2_display import mode_to_tool
    plan_skill = record.get("skill") or record.get("mode") or ""
    tool = plan_skill if str(plan_skill).startswith("plan_") else mode_to_tool(str(plan_skill))

    final_pose_report = None
    if isinstance(result, dict):
        try:
            final_pose_report = log_final_eef_error(
                ctx,
                prefix="exec_eef_pose_v2",
                plan_id=plan_id,
                arm=arm_eff,
                cand=cand,
                result=result,
            )
        except Exception as diag_exc:
            ctx.log(f"[exec_eef_pose_v2] WARN final EEF error diag failed: {diag_exc}")
        if moved_f is not None and math.isfinite(moved_f) and moved_f > 0.02:
            ctx.log(
                f"[exec_eef_pose_v2] WARN pose-only touched {object_name or 'object'}: "
                f"object_move={moved_f * 1000:.1f}mm；不作为 IK/EEF 到位失败"
            )

    out = {
        "ok": exec_ok,
        "tool": "exec_eef_pose",
        "plan_id": plan_id,
        "arm": arm_eff,
        "skill": tool,
        "skip_next_move": True,
        "skip_grip_close": True,
        "result": result if isinstance(result, dict) else {"raw": str(result)},
        "input_eef_pose": input_pose_report,
        "attempt_count": len(exec_attempts),
        "max_attempts": _EXEC_EEF_POSE_MAX_ATTEMPTS,
        "attempts": exec_attempts,
    }
    if final_pose_report is not None:
        out["eef_pose_error"] = final_pose_report
    if tool_roll_reset is not None:
        out["tool_roll_reset"] = tool_roll_reset
    if moved_f is not None and math.isfinite(moved_f) and moved_f > 0.02:
        out["warning"] = (
            f"pose-only object_move={moved_f * 1000:.1f}mm "
            "(exec_eef_pose 只按 EEF 6D 到位判定成功)"
        )
    if err_msg:
        out["error"] = err_msg
    if not exec_ok:
        ctx.log(f"[exec_eef_pose_v2] FAIL plan={plan_id}: {out.get('error', err_msg)}")
    ctx.set_result(out)
    yield world.hold_action()
