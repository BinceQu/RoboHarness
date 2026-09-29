"""exec_move_v2 —— agent 工具 exec_move(plan_id) 的仿真后端。

读取 plan_eef_v2 落盘的 plan record（move_NNNN）里的可执行 candidate，
复用既有 _execute_one_eef：
  - IK / cuRobo 到 eef_pose（pre-approach → contact）
  - 按 exec_sequence 处理夹爪：
      push/press（close_then_move）：先合爪再移动到 eef_pose，再沿 next_eef_move 推
      grasp/open/close（move_then_close）：先到 eef_pose 再合爪，再沿 next_eef_move 挪
      place（move_then_release）：到位后张爪释放
然后回传执行结果。后续由 web 端点自动 capture 主视图。
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional

import copy
import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill


# 肩部最舒适抓取距离（m）：偏离越多打分越差
_COMFORT_REACH_M = 0.55


def _object_root_pos(world, object_name: str) -> Optional[np.ndarray]:
    """Return object root position in world coordinates; fall back to AABB center."""
    if not object_name:
        return None
    from behavior_interface.skills.grasp import _resolve_object_handle, _to_np

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        return None
    try:
        return np.asarray(obj.get_position_orientation()[0], dtype=np.float64).reshape(3)
    except Exception:
        try:
            lo, hi = obj.aabb
            return (_to_np(lo).reshape(3) + _to_np(hi).reshape(3)) * 0.5
        except Exception:
            return None


def _planned_object_root_pos(cand: Dict[str, Any]) -> Optional[np.ndarray]:
    # 只能使用规划时保存的 object root pose。obj_centroid / gap_center /
    # surface centroid 都不是 root，用它们和当前 root 相减会制造假漂移。
    for path in (
        ("object_root_pos",),
        ("meta", "object_root_pos"),
        ("meta", "exec_plan_object_root_pos"),
    ):
        cur: Any = cand
        for key in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        if cur is not None:
            try:
                return np.asarray(cur, dtype=np.float64).reshape(3)
            except Exception:
                pass
    return None


def _offset_vec3_in_dict(dct: Dict[str, Any], key: str, delta: np.ndarray) -> None:
    val = dct.get(key)
    if val is None:
        return
    try:
        arr = np.asarray(val, dtype=np.float64).reshape(3)
    except Exception:
        return
    dct[key] = (arr + delta).tolist()


def _compensate_candidate_for_object_drift(
    world,
    cand: Dict[str, Any],
    object_name: str,
    ctx,
) -> Dict[str, Any]:
    """Object drift compensation is disabled; replay the saved plan pose."""
    if cand.get("target") != "grasp":
        return cand
    planned = _planned_object_root_pos(cand)
    current = _object_root_pos(world, object_name)
    if planned is None or current is None:
        if planned is None and current is not None and (cand.get("meta") or {}).get("grip_fit", {}).get("obj_centroid") is not None:
            ctx.log(
                "[exec_move_v2] skip object drift compensation: plan 缺少 object_root_pos "
                "（obj_centroid 不是 root，不能用于漂移补偿）"
            )
        return cand
    delta = current - planned
    drift = float(np.linalg.norm(delta))
    out = copy.deepcopy(cand)
    if drift >= 0.015:
        ctx.log(
            "[exec_move_v2] object drift compensation disabled: "
            f"detected Δ=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})m "
            f"|Δ|={drift*1000:.0f}mm; abort exec and re-plan from current image"
        )
        out.setdefault("meta", {})["object_drift_before_exec"] = {
            "abort": True,
            "threshold_m": 0.015,
            "planned_root_pos": planned.tolist(),
            "current_root_pos": current.tolist(),
            "delta": delta.tolist(),
            "dist_m": drift,
        }
    return out


def _candidate_with_record_root(cand: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    root = record.get("object_root_pos")
    if root is None:
        return cand
    out = copy.deepcopy(cand)
    out["object_root_pos"] = root
    out.setdefault("meta", {})["object_root_pos"] = root
    return out


def _pick_better_arm(world, target_pos, default_arm: str, ctx) -> str:
    """arm=None 时按两臂肩距择优：可达优先，再选最接近舒适距离的臂。"""
    from behavior_interface.skills.plan_grasp_object import _is_shoulder_reachable

    tp = np.asarray(target_pos, dtype=np.float64).reshape(3)
    best = None
    detail = []
    for a in ("left", "right"):
        try:
            sh = world.shoulder_pose(arm=a)
            shoulder = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
        except Exception:
            continue
        d = float(np.linalg.norm(tp - shoulder))
        ok, reason = _is_shoulder_reachable(tp, shoulder)
        # 可达 +0 罚分，不可达 +10 罚分；再加偏离舒适距离的惩罚
        score = abs(d - _COMFORT_REACH_M) + (0.0 if ok else 10.0)
        detail.append(f"{a}: d={d:.2f}m {reason} score={score:.2f}")
        cand = (score, a)
        if best is None or cand < best:
            best = cand
    if ctx is not None:
        ctx.log(f"[exec_move_v2] 自动选臂 {' | '.join(detail)}")
    if best is None:
        return default_arm
    return best[1]


def _vec_or_none(value, n: int) -> Optional[np.ndarray]:
    try:
        return np.asarray(value, dtype=np.float64).reshape(n)
    except Exception:
        return None


def _fmt_vec(value, digits: int = 3) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return "(" + ",".join(f"{float(x):+.{digits}f}" for x in arr.tolist()) + ")"


def _fmt_m(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return str(v)
    return f"{v * 1000.0:.1f}mm"


def _candidate_target_pose(cand: Dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    et = cand.get("eef_target") or {}
    return _vec_or_none(et.get("pos"), 3), _vec_or_none(et.get("quat"), 4)


def _candidate_expected_end_pos(cand: Dict[str, Any]) -> Optional[np.ndarray]:
    """Expected EEF endpoint after exec_move's post-contact motion, when simple."""
    target_pos, _target_quat = _candidate_target_pose(cand)
    if target_pos is None:
        return None
    target = str(cand.get("target") or "").strip().lower()
    next_move = _vec_or_none(cand.get("next_eef_move", [0.0, 0.0, 0.0]), 3)
    if next_move is None:
        return None
    if target == "push":
        return target_pos + next_move
    if target == "grasp":
        et = cand.get("eef_target") or {}
        # Current topdown/filter grasp path lifts by target + next_move.  Old
        # non-quat grasp path adds a tiny approach offset before lift.
        if et.get("quat") is not None:
            return target_pos + next_move
        approach = _vec_or_none(et.get("approach", [0.0, 0.0, 0.0]), 3)
        if approach is None:
            return target_pos + next_move
        n = float(np.linalg.norm(approach))
        if n > 1e-9:
            approach = approach / n
        return target_pos + 0.03 * approach + next_move
    return None


def _finger_center_report(world, arm: str, target_pos: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    try:
        link_names = list(world.robot.finger_link_names.get(arm, []))
    except Exception:
        return None
    centers = []
    out_links = []
    for name in link_names:
        try:
            link = world.robot.links.get(name)
            if link is None:
                continue
            p, _ = link.get_position_orientation()
            pos = np.asarray(p, dtype=np.float64).reshape(3)
        except Exception:
            continue
        centers.append(pos)
        item = {
            "name": str(name),
            "pos": [float(x) for x in pos.tolist()],
        }
        if target_pos is not None:
            d = pos - target_pos
            item["delta_to_target"] = [float(x) for x in d.tolist()]
            item["err_m"] = float(np.linalg.norm(d))
        out_links.append(item)
    if not centers:
        return None
    center = np.mean(np.stack(centers, axis=0), axis=0)
    report: Dict[str, Any] = {
        "center": [float(x) for x in center.tolist()],
        "links": out_links,
    }
    if target_pos is not None:
        delta = center - target_pos
        report["delta_to_target"] = [float(x) for x in delta.tolist()]
        report["err_m"] = float(np.linalg.norm(delta))
    return report


def _pose_error_against_target(world, arm: str, target_pos: np.ndarray, target_quat: Optional[np.ndarray]) -> Dict[str, Any]:
    eef = world.eef_pose(arm=arm)
    actual_pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    actual_quat = _vec_or_none(eef.get("quat"), 4)
    delta = actual_pos - target_pos
    report: Dict[str, Any] = {
        "target_pos": [float(x) for x in target_pos.tolist()],
        "actual_pos": [float(x) for x in actual_pos.tolist()],
        "delta": [float(x) for x in delta.tolist()],
        "pos_err_m": float(np.linalg.norm(delta)),
    }
    if target_quat is not None:
        report["target_quat"] = [float(x) for x in target_quat.tolist()]
    if actual_quat is not None:
        report["actual_quat"] = [float(x) for x in actual_quat.tolist()]
    if target_quat is not None:
        try:
            from behavior_interface.skills.eef import _eef_pose_err

            pos_err, ori_err, app_err = _eef_pose_err(world, arm, target_pos, target_quat)
            report["pos_err_m"] = float(pos_err)
            report["ori_err_deg"] = float(ori_err)
            report["approach_err_deg"] = float(app_err)
        except Exception as exc:
            report["ori_error_unavailable"] = str(exc)
    return report


def log_plan_eef_input(ctx, *, prefix: str, plan_id: str, arm: str, cand: Dict[str, Any]) -> Dict[str, Any]:
    target_pos, target_quat = _candidate_target_pose(cand)
    et = cand.get("eef_target") or {}
    gripper_cmd = et.get("gripper_cmd", cand.get("gripper_cmd"))
    next_move = _vec_or_none(cand.get("next_eef_move", [0.0, 0.0, 0.0]), 3)
    report: Dict[str, Any] = {
        "plan_id": plan_id,
        "arm": arm,
        "target": cand.get("target"),
        "exec_sequence": (cand.get("meta") or {}).get("exec_sequence"),
        "gripper_cmd": gripper_cmd,
    }
    if target_pos is not None:
        report["target_pos"] = [float(x) for x in target_pos.tolist()]
    if target_quat is not None:
        report["target_quat"] = [float(x) for x in target_quat.tolist()]
    if next_move is not None:
        report["next_eef_move"] = [float(x) for x in next_move.tolist()]
    msg = (
        f"[{prefix}] input_eef_pose plan={plan_id} arm={arm} "
        f"target={cand.get('target')} exec_seq={(cand.get('meta') or {}).get('exec_sequence')} "
        f"pos={_fmt_vec(target_pos) if target_pos is not None else 'n/a'} "
        f"quat={_fmt_vec(target_quat) if target_quat is not None else 'n/a'} "
        f"next_move={_fmt_vec(next_move) if next_move is not None else 'n/a'} "
        f"gripper_cmd={gripper_cmd}"
    )
    ctx.log(msg)
    return report


def log_final_eef_error(
    ctx,
    *,
    prefix: str,
    plan_id: str,
    arm: str,
    cand: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    include_expected_end: bool = False,
) -> Dict[str, Any]:
    world = ctx.world
    target_pos, target_quat = _candidate_target_pose(cand)
    report: Dict[str, Any] = {
        "plan_id": plan_id,
        "arm": arm,
        "target": cand.get("target"),
    }
    if target_pos is not None:
        target_report = _pose_error_against_target(world, arm, target_pos, target_quat)
        report["vs_input_eef_pose"] = target_report
        ori_s = (
            f" ori={target_report.get('ori_err_deg', float('nan')):.2f}°"
            if target_report.get("ori_err_deg") is not None else ""
        )
        app_s = (
            f" approach={target_report.get('approach_err_deg', float('nan')):.2f}°"
            if target_report.get("approach_err_deg") is not None else ""
        )
        ctx.log(
            f"[{prefix}] final_eef_vs_input plan={plan_id} arm={arm} "
            f"target_pos={_fmt_vec(target_pos)} actual_pos={_fmt_vec(target_report['actual_pos'])} "
            f"delta={_fmt_vec(target_report['delta'])} "
            f"|delta|={_fmt_m(target_report.get('pos_err_m'))}{ori_s}{app_s}"
        )
        finger = _finger_center_report(world, arm, target_pos)
        if finger is not None:
            report["finger_center_vs_input_eef_pose"] = finger
            ctx.log(
                f"[{prefix}] finger_center_vs_input plan={plan_id} arm={arm} "
                f"center={_fmt_vec(finger['center'])} "
                f"delta={_fmt_vec(finger.get('delta_to_target', [0, 0, 0]))} "
                f"|delta|={_fmt_m(finger.get('err_m'))}"
            )
    if include_expected_end:
        expected_end = _candidate_expected_end_pos(cand)
        if expected_end is not None:
            end_report = _pose_error_against_target(world, arm, expected_end, target_quat)
            report["vs_expected_exec_end"] = end_report
            ctx.log(
                f"[{prefix}] final_eef_vs_expected_end plan={plan_id} arm={arm} "
                f"expected_pos={_fmt_vec(expected_end)} actual_pos={_fmt_vec(end_report['actual_pos'])} "
                f"delta={_fmt_vec(end_report['delta'])} "
                f"|delta|={_fmt_m(end_report.get('pos_err_m'))}"
            )
    if isinstance(result, dict):
        summary_keys = (
            "pre_err", "cnt_err", "end_err", "final_pos_err_m",
            "final_ori_err_deg", "final_approach_err_deg",
            "skip_next_move", "skip_grip_close",
        )
        summary = {k: result.get(k) for k in summary_keys if k in result}
        report["result_error_summary"] = summary
        ctx.log(f"[{prefix}] result_error_summary plan={plan_id}: {summary}")
    return report


@register_skill(
    "exec_move_v2",
    description=(
        "执行 plan_id 对应的 EEF 动作：复用 _execute_one_eef（IK/cuRobo 到 eef_pose，"
        "按 skill 处理合爪/释放，再沿 next_eef_move 挪动）。"
        "arm='left'|'right' 指定手臂；不填则按肩距自动选更好解的一只。"
    ),
)
def exec_move_v2(
    ctx, session_id: str, plan_id: str, arm: Optional[str] = None,
    back_m: float = 0.10,
) -> Generator:
    from behavior_interface.skills.eef import _ensure_world_pinned_actions, _execute_one_eef
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
            f"[exec_move_v2] WARN plan={plan_id}: 规划标注不可达 ({reason})，仍尝试执行"
        )

    object_name = (
        record.get("object_name")
        or cand.get("meta", {}).get("object_name")
        or ""
    )
    cand = _candidate_with_record_root(cand, record)
    cand = _compensate_candidate_for_object_drift(world, cand, object_name, ctx)
    drift_meta = (cand.get("meta") or {}).get("object_drift_before_exec")
    if isinstance(drift_meta, dict) and drift_meta.get("abort"):
        dist_mm = float(drift_meta.get("dist_m", 0.0)) * 1000.0
        msg = (
            f"物体位置已与 plan 时不同：root drift={dist_mm:.1f}mm "
            "(阈值 15mm)。请 restart/move_to_object/plan 后再 exec。"
        )
        ctx.set_result({
            "ok": False,
            "tool": "exec_move",
            "plan_id": plan_id,
            "error": msg,
            "object_drift_before_exec": drift_meta,
        })
        yield world.hold_action()
        return

    record_arm = record.get("arm") or cand.get("arm") or "right"
    arm_req = (arm or "").lower().strip() if isinstance(arm, str) else None
    target_pos_pick = (cand.get("eef_target") or {}).get("pos") or [0.0, 0.0, 0.0]
    if arm_req in ("left", "right"):
        arm = arm_req
        ctx.log(f"[exec_move_v2] 使用指定 arm={arm}")
    else:
        # 默认跟规划臂一致；仅当规划臂肩距不可达时才自动换另一只
        from behavior_interface.skills.plan_grasp_object import _is_shoulder_reachable
        tp = np.asarray(target_pos_pick, dtype=np.float64).reshape(3)
        sh = world.shoulder_pose(arm=record_arm)
        shoulder = np.array([sh["x"], sh["y"], sh["z"]], dtype=np.float64)
        ok_plan, reason_plan = _is_shoulder_reachable(tp, shoulder)
        if ok_plan:
            arm = record_arm
            ctx.log(f"[exec_move_v2] 使用规划臂 arm={arm} ({reason_plan})")
        else:
            arm = _pick_better_arm(world, target_pos_pick, record_arm, ctx)
            ctx.log(
                f"[exec_move_v2] 规划臂 {record_arm} 不可达({reason_plan}) "
                f"→ 自动换 arm={arm}"
            )
    last = {
        "ok": True,
        "target": cand.get("target", "grasp"),
        "candidates": [cand],
        "object": {
            "input": object_name,
        },
    }

    ctx.log(f"[exec_move_v2] plan={plan_id} skill={record.get('skill')} "
            f"target={cand.get('target')} exec_seq={cand.get('meta', {}).get('exec_sequence')}")

    ctx.log(f"[exec_move_v2] back_m={back_m*100:.0f}cm")
    input_pose_report = log_plan_eef_input(
        ctx, prefix="exec_move_v2", plan_id=plan_id, arm=arm, cand=cand,
    )
    try:
        result = yield from _execute_one_eef(ctx, last, cand, arm, back_m=float(back_m))
    except Exception as e:
        import traceback
        ctx.log(f"[exec_move_v2] 执行异常: {e}\n{traceback.format_exc()}")
        ctx.set_result({"ok": False, "error": f"exec 异常: {e}", "plan_id": plan_id})
        yield world.hold_action()
        return

    from behavior_interface.v2_display import mode_to_tool
    plan_skill = record.get("skill") or record.get("mode") or ""
    tool = plan_skill if str(plan_skill).startswith("plan_") else mode_to_tool(str(plan_skill))

    exec_ok = True
    err_msg = None
    final_pose_report = None
    if isinstance(result, dict):
        try:
            final_pose_report = log_final_eef_error(
                ctx,
                prefix="exec_move_v2",
                plan_id=plan_id,
                arm=arm,
                cand=cand,
                result=result,
                include_expected_end=True,
            )
        except Exception as diag_exc:
            ctx.log(f"[exec_move_v2] WARN final EEF error diag failed: {diag_exc}")
        exec_ok = bool(result.get("ok", True))
        err_msg = result.get("error")
        object_label = object_name or cand.get("meta", {}).get("object_name") or "object"
        # grasp 成功以物体 z 上升为准（object Δz>4cm），不因末端残差否定
        if result.get("grasped"):
            exec_ok = True
            err_msg = None
        elif cand.get("target") == "grasp":
            dz = result.get("object_dz")
            if dz is not None and float(dz) <= 0.04:
                exec_ok = False
                err_msg = (
                    f"抓取失败：{object_label} Δz={float(dz)*1000:.1f}mm "
                    f"(需 >40mm)"
                )
            moved = result.get("object_moved_dist")
            try:
                moved_f = float(moved) if moved is not None else None
            except (TypeError, ValueError):
                moved_f = None
            if exec_ok and moved_f is not None and moved_f <= 0.02:
                exec_ok = False
                err_msg = (
                    f"抓取失败：{object_label} 位移={moved_f*1000:.1f}mm "
                    "(需能观测到物体移动)"
                )
        elif exec_ok and float(result.get("final_pos_err_m", result.get("cnt_err", 0))) > 0.06:
            exec_ok = False
            err_msg = (
                f"末端未到达规划位姿（偏差 "
                f"{float(result.get('final_pos_err_m', 0)) * 1000:.0f}mm）"
            )

    out = {
        "ok": exec_ok,
        "tool": "exec_move",
        "plan_id": plan_id,
        "skill": tool,
        "result": result if isinstance(result, dict) else {"raw": str(result)},
        "input_eef_pose": input_pose_report,
    }
    if final_pose_report is not None:
        out["eef_pose_error"] = final_pose_report
    if err_msg:
        out["error"] = err_msg
    if not exec_ok:
        ctx.log(f"[exec_move_v2] FAIL plan={plan_id}: {out.get('error', err_msg)}")
    ctx.set_result(out)
    yield world.hold_action()
