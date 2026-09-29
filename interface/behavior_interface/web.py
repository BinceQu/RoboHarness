"""Flask Web 前端。
- /                主页（四路 MJPEG + 状态 + skill 控制 + 日志）
- /video/<feed>    MJPEG 流，feed in {main, head, left_wrist, right_wrist}
- /api/state       JSON：完整状态快照
- /api/skills      JSON：注册的 skill 列表
- /api/skill       POST {name, args}：提交一个 skill 执行
- /api/skill/cancel POST：取消当前
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import os
import socket

# Keep web-side JPEG/resize helpers from creating an unbounded OpenCV pool.
try:
    _opencv_threads = int(os.environ.get("BEHAVIOR_OPENCV_THREADS", "1"))
except (TypeError, ValueError):
    _opencv_threads = 1
cv2.setNumThreads(max(1, min(_opencv_threads, 16)))

from flask import Flask, Response, g, jsonify, render_template, request, send_file

from .agent_monitor import (
    AgentMonitorStore,
    challenge_instance_max_ticks,
    is_external_agent_session,
    resolve_v2_tool_name,
)
from .operator_controls import (
    eval_control_snapshot,
    stop_agent_sessions,
)
from .coordinate_contract import COORDINATE_MAX, COORDINATE_SYSTEM
from .recording import (
    HumanTrajectoryRecorder,
    RecordingConflict,
    RecordingNotFound,
)
from .skills import list_skills
from .tool import active_tool_version, active_v2_tool_names


_POST_ACTION_OBSERVATION_KEYS = (
    "image_id",
    "rgb_main",
    "rgb_main_path",
    "rgb_overlay_path",
    "rgb_path",
    "image_width",
    "image_height",
    "camera",
    "tro",
    "robot",
    "base_path_overlay",
    "chassis_forward_2m",
    "eef_near_0.1m",
    "nearby_object_warning",
    "track_object_distance_binding",
    "memory",
    "memory_text",
    "feed",
)
_CLOSE_GRIPPER_CONTROL_HORIZON_S = 3.5
_CLOSE_GRIPPER_MIN_WAIT_TIMEOUT_S = 180.0
_CLOSE_GRIPPER_MAX_WAIT_TIMEOUT_S = 600.0
_CLOSE_GRIPPER_MIN_OBSERVED_HZ = 0.25
_CLOSE_GRIPPER_WAIT_MARGIN = 1.5
_CLOSE_GRIPPER_WAIT_OVERHEAD_S = 30.0


def _promote_post_action_observation_fields(result: dict) -> dict:
    """Make top-level observation fields one coherent post-action frame."""
    observation = result.get("observation")
    if not isinstance(observation, dict):
        return result

    input_image_id = str(result.get("image_id") or "").strip()
    output_image_id = str(observation.get("image_id") or "").strip()
    if input_image_id and output_image_id and input_image_id != output_image_id:
        result.setdefault("input_image_id", input_image_id)

    for key in _POST_ACTION_OBSERVATION_KEYS:
        if key in observation:
            result[key] = observation[key]
    return result


def _maybe_attach_spatial_map(
    session_id: str,
    tool: str,
    args: dict | None,
    result: dict | None,
) -> dict:
    """底盘工具退出后更新测试口小地图；官方评测默认不附图。"""
    payload = result if isinstance(result, dict) else {}
    try:
        from behavior_interface.rtabmap_slam.live import (
            get_live_mapper,
            live_backend_selected,
        )

        if live_backend_selected():
            mapper = get_live_mapper()
            if mapper is not None:
                return mapper.attach_to_result(session_id, tool, args, payload)
        from behavior_interface.spatial_map import attach_to_result

        return attach_to_result(session_id, tool, args, payload)
    except Exception:
        return payload


def _default_close_gripper_wait_timeout(server: Any) -> float:
    """Budget wall time for a control-step horizon on a slow evaluator loop."""
    try:
        control_horizon_s = float(
            getattr(
                server,
                "gripper_close_control_horizon_s",
                _CLOSE_GRIPPER_CONTROL_HORIZON_S,
            )
        )
    except (TypeError, ValueError):
        control_horizon_s = _CLOSE_GRIPPER_CONTROL_HORIZON_S
    try:
        control_hz = float(
            getattr(
                server,
                "gripper_control_hz",
                getattr(server, "target_hz", 30.0),
            )
        )
    except (TypeError, ValueError):
        control_hz = 30.0
    try:
        observed_hz = float(getattr(server, "fps", 0.0))
    except (TypeError, ValueError):
        observed_hz = 0.0

    if not math.isfinite(control_horizon_s) or control_horizon_s <= 0.0:
        control_horizon_s = _CLOSE_GRIPPER_CONTROL_HORIZON_S
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        control_hz = 30.0
    if not math.isfinite(observed_hz) or observed_hz <= 0.0:
        observed_hz = 1.0
    effective_hz = max(
        _CLOSE_GRIPPER_MIN_OBSERVED_HZ,
        min(observed_hz, control_hz),
    )
    control_steps = math.ceil(control_horizon_s * control_hz)
    estimated_wall_s = control_steps / effective_hz
    return min(
        _CLOSE_GRIPPER_MAX_WAIT_TIMEOUT_S,
        max(
            _CLOSE_GRIPPER_MIN_WAIT_TIMEOUT_S,
            estimated_wall_s * _CLOSE_GRIPPER_WAIT_MARGIN
            + _CLOSE_GRIPPER_WAIT_OVERHEAD_S,
        ),
    )


_AGENT_MONITOR_PUBLISH_LOCK = threading.Lock()
_AGENT_MONITOR_PUBLISHED_PAYLOAD: dict[str, Any] | None = None
_DEFAULT_AGENT_MONITOR_PUBLISH_MAX_BYTES = 8 * 1024 * 1024
def _agent_monitor_publish_max_bytes() -> int:
    try:
        return max(
            1024 * 1024,
            int(
                os.environ.get(
                    "BEHAVIOR_AGENT_MONITOR_MAX_BYTES",
                    str(_DEFAULT_AGENT_MONITOR_PUBLISH_MAX_BYTES),
                )
            ),
        )
    except (TypeError, ValueError):
        return _DEFAULT_AGENT_MONITOR_PUBLISH_MAX_BYTES


def _published_agent_monitor_payload() -> dict[str, Any] | None:
    with _AGENT_MONITOR_PUBLISH_LOCK:
        if _AGENT_MONITOR_PUBLISHED_PAYLOAD is None:
            return None
        return dict(_AGENT_MONITOR_PUBLISHED_PAYLOAD)


def _store_published_agent_monitor_payload(payload: dict[str, Any]) -> None:
    global _AGENT_MONITOR_PUBLISHED_PAYLOAD
    with _AGENT_MONITOR_PUBLISH_LOCK:
        _AGENT_MONITOR_PUBLISHED_PAYLOAD = payload


def _compact_agent_monitor_payload(payload: dict) -> dict:
    """Return the monitor fields the browser needs without base64/raw dumps."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid agent monitor payload"}
    compact: dict[str, Any] = {}
    for key in (
        "ok",
        "error",
        "path",
        "preview",
        "blank_until_run",
        "stale_reason",
        "monitor_source",
        "published_at",
        "heartbeat_ts",
        "version",
        "session_id",
        "started_at",
        "attempt_id",
        "live_attempt_id",
        "live_session_id",
        "viewing_history",
        "agent_id",
        "task",
        "port",
        "host",
        "interface_pid",
        "log_dir",
        "turn",
        "updated_at",
        "model",
        "server_pid",
        "server_started_ts",
        "server_reset_count",
        "prompt",
        "active_skill",
        "skill_selection_mode",
        "skill_selection_output",
        "skill_decision",
        "point",
        "chosen",
        "user_prompt",
        "loaded_skills",
        "active",
        "card_count",
        "run_dir",
        "tick_origin",
        "session_ticks",
        "max_ticks",
        "task_id",
        "task_name",
        "world_reset_seq",
        "live_world_reset_seq",
    ):
        if key in payload:
            compact[key] = payload.get(key)
    if compact.get("max_ticks") in (None, "", 0, 10000):
        compact["max_ticks"] = challenge_instance_max_ticks(
            compact.get("task_id") if compact.get("task_id") is not None else payload.get("task_id"),
            compact.get("task_name") or payload.get("task_name") or payload.get("task"),
        )
    if "active_skill" not in compact:
        compact["active_skill"] = ""
    if "skill_selection_output" not in compact:
        compact["skill_selection_output"] = ""
    compact["user_prompt"] = (
        payload.get("user_prompt") or payload.get("prompt") or compact.get("user_prompt") or ""
    )
    compact["prompt"] = compact["user_prompt"]
    # 有卡片就带上。以前只认 embodied_keyframes，published/空 source 会被裁成
    # cards=[]，页面一直 waiting。
    if (
        payload.get("monitor_source") == "embodied_keyframes"
        or payload.get("cards")
        or payload.get("sessions")
    ):
        compact["cards"] = _compact_agent_monitor_cards(payload.get("cards"))
        compact["sessions"] = _compact_agent_monitor_sessions(payload.get("sessions"))
        compact["active"] = bool(payload.get("active"))
    else:
        compact.setdefault("cards", [])
        compact.setdefault("active", False)

    if "tool_result" in payload:
        compact["tool_result"] = _compact_agent_monitor_tool_result(payload.get("tool_result"))

    image = dict(payload.get("image") or {})
    if image:
        image["has_data_url"] = bool(image.get("data_url"))
        image.pop("data_url", None)
        compact["image"] = image

    response = payload.get("response")
    if isinstance(response, dict):
        compact["response"] = {
            key: response.get(key)
            for key in ("text", "thinking", "action", "status")
            if key in response
        }
    elif isinstance(response, str):
        compact["response"] = {"text": response}
    return compact


def _compact_agent_monitor_sessions(sessions: Any) -> list[dict[str, Any]]:
    """本端口历史 attempt 列表；每项带精简卡片，供下拉切换。"""
    compact: list[dict[str, Any]] = []
    if not isinstance(sessions, list):
        return compact
    for item in sessions:
        if not isinstance(item, dict):
            continue
        compact.append({
            "attempt_id": item.get("attempt_id") or "",
            "session_id": item.get("session_id") or "",
            "started_at": item.get("started_at") or "",
            "updated_at": item.get("updated_at") or "",
            "card_count": item.get("card_count") or 0,
            "status": item.get("status") or "",
            "live": bool(item.get("live")),
            "user_prompt": item.get("user_prompt") or "",
            "loaded_skills": item.get("loaded_skills") or [],
            "cards": _compact_agent_monitor_cards(item.get("cards")),
        })
    return compact


def _compact_agent_monitor_cards(cards: Any) -> list[dict[str, Any]]:
    """直播卡片只给前端工具名 / 参数 / 是否有图，不带磁盘路径。"""
    compact: list[dict[str, Any]] = []
    if not isinstance(cards, list):
        return compact
    for card in cards:
        if not isinstance(card, dict):
            continue
        compact.append({
            "card_id": card.get("card_id"),
            "ts": card.get("ts"),
            "tool": card.get("tool"),
            "args": card.get("args") or {},
            "image_id": card.get("image_id") or "",
            "input_image_id": card.get("input_image_id") or "",
            "output_image_id": card.get("output_image_id") or "",
            "has_input": bool(card.get("has_input")),
            "has_output": bool(card.get("has_output")),
            "has_image": bool(card.get("has_image") or card.get("has_input") or card.get("has_output")),
            "points": card.get("points") or [],
            "ok": card.get("ok"),
            "loaded_skills": card.get("loaded_skills") or [],
            "model_image_v": card.get("model_image_v") or "",
            "session_ticks": int(card.get("session_ticks") or 0),
        })
    return compact


def _compact_agent_monitor_tool_result(tool_result: Any) -> Any:
    if not isinstance(tool_result, dict):
        return tool_result
    compact: dict[str, Any] = {}
    for key in ("tool", "args"):
        if key in tool_result:
            compact[key] = tool_result.get(key)
    result = tool_result.get("result")
    if isinstance(result, dict):
        keep = (
            "ok",
            "error",
            "status",
            "message",
            "image_id",
            "plan_id",
            "object_name",
            "arm",
            "distance_m",
            "left_shoulder_distance_m",
            "right_shoulder_distance_m",
            "min_shoulder_distance_m",
            "overlap_vol_cm3",
            "grasp_vol_cm3",
        )
        compact["result"] = {key: result.get(key) for key in keep if key in result}
    elif result is not None:
        compact["result"] = result
    return compact


def _agent_monitor_payload_is_current(payload: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, dict) or not payload.get("ok"):
        return False, str((payload or {}).get("error") or "monitor payload unavailable")
    try:
        updated_at = float(payload.get("updated_at") or 0.0)
    except Exception:
        updated_at = 0.0
    try:
        started_ts = float(state.get("started_ts") or 0.0)
    except Exception:
        started_ts = 0.0
    if updated_at and started_ts and updated_at < started_ts - 1.0:
        return False, "monitor file predates current interface process"
    if payload.get("server_pid") is not None and state.get("pid") is not None:
        try:
            if int(payload.get("server_pid")) != int(state.get("pid")):
                return False, "monitor file belongs to a different interface pid"
        except Exception:
            return False, "monitor pid metadata invalid"
    state_reset = state.get("reset_count")
    payload_reset = payload.get("server_reset_count")
    if payload_reset is None:
        if int(state_reset or 0) > 0:
            return False, "monitor file lacks reset metadata after a reset"
    else:
        try:
            if int(payload_reset) != int(state_reset or 0):
                return False, "monitor reset_count does not match current reset"
        except Exception:
            return False, "monitor reset metadata invalid"
    return True, ""


def _monitor_task_fields(server, state: dict[str, Any] | None = None) -> tuple[Any, str, Any]:
    state = state or {}
    task_id = state.get("task_id")
    if task_id is None:
        task_id = getattr(server, "task_id", None)
    task_name = str(
        state.get("task")
        or state.get("task_name")
        or getattr(server, "task_name", None)
        or ""
    ).strip()
    port = getattr(server, "web_port", None)
    return task_id, task_name, port


def _empty_agent_monitor_payload(
    server,
    state: dict[str, Any] | None = None,
    *,
    stale_reason: str = "",
) -> dict[str, Any]:
    state = state or server.snapshot_state()
    task_id, task_name, port = _monitor_task_fields(server, state)
    return {
        "ok": True,
        "preview": True,
        "blank_until_run": True,
        "stale_reason": stale_reason,
        "updated_at": time.time(),
        "session_id": "",
        "turn": None,
        "server_pid": state.get("pid"),
        "server_started_ts": state.get("started_ts"),
        "server_reset_count": state.get("reset_count"),
        "world_reset_seq": 0,
        "live_world_reset_seq": 0,
        "task_id": task_id,
        "task_name": task_name,
        "max_ticks": challenge_instance_max_ticks(task_id, task_name, port=port),
        "active_skill": "",
        "skill_selection_output": "",
        "skill_decision": None,
        "image": {},
        "aux_images": [],
        "response": {
            "text": "",
            "thinking": None,
            "action": None,
            "status": "awaiting_external_agent_publish",
        },
        "chosen": None,
        "tool_result": None,
        "point": None,
        "user_prompt": "",
        "cards": [],
        "active": False,
        "card_count": 0,
    }


def _agent_monitor_payload_for_server(
    server,
    state: dict[str, Any] | None = None,
    store: AgentMonitorStore | None = None,
) -> dict[str, Any]:
    # None builds the browser snapshot.  {} is the poll: do not replace it
    # with snapshot_state(), which is what piled Flask threads into CLOSE_WAIT.
    if state is None:
        state = server.snapshot_state()
    if store is not None:
        task_id, task_name, _port = _monitor_task_fields(server, state)
        store.bind_task(task_id, task_name)
        snapshot = store.snapshot(state)
        if (
            snapshot.get("active")
            or snapshot.get("user_prompt")
            or snapshot.get("cards")
            or snapshot.get("sessions")
        ):
            return snapshot
    published = _published_agent_monitor_payload()
    if published is not None:
        current, reason = _agent_monitor_payload_is_current(published, state)
        if current:
            return published
        return _empty_agent_monitor_payload(server, state, stale_reason=reason)
    return _empty_agent_monitor_payload(
        server,
        state,
        stale_reason="no current agent monitor payload has been published",
    )


def _json_no_store(payload: dict) -> Response:
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


def _json_safe(obj: Any) -> Any:
    """把 inf/nan 换成 null，避免前端 JSON.parse 失败。"""
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

# ── v2 工具元数据：agent 与网页下拉菜单共用的统一接口描述 ──────────────────────
# session_id 由后端隐式注入（网页用固定 "web"），不在 args 暴露。
# widget: number=数值框 / image=自动填最近 image_id / plan=自动填最近 plan_id /
#         uv=点图取 (u,v) / select=下拉枚举 / hidden=固定常量(不显示)。
def _F(n, d=0.0, req=False, unit=""):
    return {
        "name": n, "type": "number", "default": d, "required": req,
        "widget": "number", "unit": unit,
    }


def _m_to_legacy_cm(value) -> float:
    """Convert public v2 linear meters to legacy skill centimeters."""
    return float(value) * 100.0


# move_in_robot_coord upward：两阶段垂直升降 FK 直立总降幅约 0.54m，API 留余量；不可达由 skill 回报
_ROBOT_UPWARD_MAX_M = 0.60


V2_TOOLS = [
    {"name": "capture_head_camera", "endpoint": "/api/v2/capture_head_camera", "args": [],
     "desc": "拍一张 head camera 主视图，返回 image_id（后续点选/规划都基于它）。"},
    {"name": "capture", "endpoint": "/api/v2/capture", "args": [],
     "desc": "拍一张 head 主视图，返回 image_id（后续点选/规划都基于它）。"},
    {"name": "capture_left_wrist_camera", "endpoint": "/api/v2/capture_left_wrist_camera", "args": [],
     "desc": (
         "拍左腕 RGB + depth，不追加 head capture。图中红色图层对应左夹爪两爪中间区域；"
         "红色像素表示非机器人可见 3D 点严格落在该区域内；底边 x(m) 标尺以夹爪开口中心为 0，"
         "向右为正，范围 ±0.02m，每格 0.01m。"
     )},
    {"name": "capture_right_wrist_camera", "endpoint": "/api/v2/capture_right_wrist_camera", "args": [],
     "desc": (
         "拍右腕 RGB + depth，不追加 head capture。图中红色图层对应右夹爪两爪中间区域；"
         "红色像素表示非机器人可见 3D 点严格落在该区域内；底边 x(m) 标尺以夹爪开口中心为 0，"
         "向右为正，范围 ±0.02m，每格 0.01m。"
     )},
    {"name": "read_depth", "endpoint": "/api/v2/read_depth",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True,
          "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "required": True,
          "unit": "Qwen3-VL 相对坐标 0..1000"},
     ],
     "pick_like": "mark_object",
     "desc": "读取冻结 capture 中所选 UV 单个像素的 depth_linear，返回 depth_m，单位 m。"},
    {"name": "adjust_chassis", "endpoint": "/api/v2/adjust_chassis",
     "args": [
         _F("forward", unit="m(+前进/-倒车)"),
         _F("translation", unit="m(+左移/-右移)"),
         _F("spin", unit="deg(+左转CCW/-右转)"),
     ],
     "desc": "只调整底盘：机体系 forward/translation(+左/-右) 同帧合成为二维直移，随后执行水平 spin；不改 pitch/upward，完成后拍 head camera。"},
    {"name": "navigate_to", "endpoint": "/api/v2/navigate_to",
     "args": [
         {"name": "name", "widget": "text", "required": True,
          "placeholder": "已通过 mark_on_map 保存的地点名"},
         _F("arrival_tolerance_m", 0.25, unit="m"),
     ],
     "desc": (
         "导航到已标记地点。读取当前合规小地图的占用栅格，在未知区外按底盘"
         "静态尺寸膨胀障碍，并优先规划远离墙面的折线路径；随后只通过官方"
         "底盘 action 分段执行。地图后端可替换，不读取场景或全局位姿真值。"
     )},
    {"name": "adjust_pitch", "endpoint": "/api/v2/adjust_pitch",
     "args": [_F("degree", unit="deg(+仰头/-俯身，纯q3)")],
     "desc": "只调整机器人俯仰 pitch：正值仰头，负值俯身；完成后拍 head camera。"},
    {"name": "adjust_height", "endpoint": "/api/v2/adjust_height",
     "args": [_F("upward", unit="m(+升/-降)")],
     "desc": "只调整机器人高度 upward，单位米；完成后拍 head camera。"},
    {"name": "move_in_world_coord", "endpoint": "/api/v2/move_in_world_coord",
     "args": [
         _F("dx", unit="m"),
         _F("dy", unit="m"),
         _F("dz", unit="m(胸口升降，有效约±0.3，勿填角度)"),
         _F("dthetax", unit="deg(水平转)"),
         _F("dthetaz", unit="deg(俯仰)"),
     ],
     "desc": "世界系胸口 5D 增量（dx/dy 世界轴，A* 避障，不倒车）；完成后自动拍主视图。"},
    {"name": "move_in_robot_coord", "endpoint": "/api/v2/move_in_robot_coord",
     "args": [
         _F("forward", unit="m(+前进/-倒车)"),
         _F("spin", unit="deg(+左转CCW/-右转)"),
         _F("pitch", unit="deg(+仰头/-俯身,纯q3,q1/q2锁定)"),
         _F("upward", unit="m(+升/-降,反向查表θz≈90°,z≥0.73流形/<0.73探底,1cm格点)"),
     ],
     "desc": "机体系：反向 upward 查表垂直升降+pitch/forward/spin；无解返回不可达；完成后拍主视图。"},
    {"name": "face_to_point", "endpoint": "/api/v2/face_to_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         _F("max_abs_spin", 60.0, unit="deg(安全限幅)"),
     ],
     "desc": "仅水平旋转底盘，把选中的 head 图点转到画面中线；不俯仰、不平移，完成后自动拍主视图。"},
    {"name": "spin_to_facing_point", "endpoint": "/api/v2/spin_to_facing_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         _F("max_abs_spin", 60.0, unit="deg(安全限幅)"),
     ],
     "desc": "只水平旋转底盘，让选中的 head 图点落到画面中线；不俯仰、不平移。"},
    {"name": "query_map", "endpoint": "/api/v2/query_map",
    "args": [
        {"name": "fixed_north_up", "widget": "select", "required": False,
         "options": ["", "1"],
         "placeholder": "填 1 改成起点朝向朝上的固定俯视图（默认车头朝上）"},
    ],
    "desc": (
        "查询场景地图：车头朝上的俯视小地图，用 depth 建出的占用栅格画出"
        "墙与家具、走过的空地，叠加最近一段行进路线、起点和自己标记过的"
        "地点，并返回每个已标记地点的方位与距离。"
        "只用 depth 观测与合规底盘完成量，不读世界真值。"
    )},
   {"name": "mark_on_map", "endpoint": "/api/v2/mark_on_map",
    "args": [
        {"name": "name", "widget": "text", "required": True,
         "placeholder": "起个名字，如「车库门口」「目标箱子」"},
        {"name": "image_id", "widget": "image", "required": False,
         "placeholder": "要点选的那张 head 图；不填就标脚下"},
        {"name": "u", "widget": "uv", "required": False,
         "unit": "Qwen3-VL 相对坐标 0..1000"},
        {"name": "v", "widget": "uv", "required": False,
         "unit": "Qwen3-VL 相对坐标 0..1000"},
    ],
    "pick_like": "mark_object",
    "one_of": [
        {"fields": ["name"], "label": "标记脚下位置"},
        {"fields": ["name", "image_id", "u", "v"], "label": "点选 head 图标记物体"},
    ],
    "desc": (
        "在地图上标记一个位置并起名，之后一直画在小地图上，随时能查到它相对"
        "当前底盘的方位。两种用法："
        "(1) 只给 name，标记机器人脚下的当前位置，例如「车库门口」；"
        "(2) 给 name + image_id + u,v，点选那张 head 图上的一个像素，用该帧 depth "
        "反解出它的三维位置，把地面投影标到地图上，例如点中远处的箱子标成「目标箱子」。"
        "u,v 是 Qwen3-VL 相对坐标 0..1000，与 track_object_distance 等点选工具一致，"
        "image_id 必须是 capture_head_camera 返回的那张图。"
        "返回所有已标记地点相对当前底盘的方位：距离多少米、spin 多少度可以正对过去"
        "（左正右负，可直接填给 adjust_chassis 的 spin）。"
        "迷路时先标记关键位置（门口、目标物所在房间），再靠这些方位走回去。"
    )},
   {"name": "open_gripper", "endpoint": "/api/v2/open_gripper",
     "args": [{"name": "arm", "widget": "select", "required": True,
               "options": ["right", "left"], "default": "right"}],
     "desc": "打开指定 left/right 夹爪；不移动 EEF。"},
    {"name": "close_gripper", "endpoint": "/api/v2/close_gripper",
     "args": [{"name": "arm", "widget": "select", "required": True,
               "options": ["right", "left"], "default": "right"}],
     "desc": "闭合指定 left/right 夹爪；不移动 EEF，完成后返回对应 wrist camera 视图。"},
    {"name": "adjust_left_eef_pose_in_head_frame",
     "endpoint": "/api/v2/adjust_left_eef_pose_in_head_frame",
     "args": [
         _F("forward", unit="m(head相机系向前，可负)"),
         _F("upward", unit="m(head相机系向上，可负)"),
         _F("leftward", unit="m(head相机系向左，可负)"),
         _F("roll", unit="deg(左夹爪局部轴，与adjust_plan_pose一致)"),
         _F("pitch", unit="deg(+翘头/fingertips-up)"),
         _F("yaw", unit="deg(左夹爪局部轴，与adjust_plan_pose一致)"),
         _F("pos_tol", 0.012, unit="m"),
         _F("ori_tol_deg", 3.0, unit="deg"),
     ],
     "desc": (
         "调整左 EEF 6D pose：平移按 head camera，旋转按左夹爪局部 "
         "roll/pitch/yaw；平移单位为米，完成后拍 head camera。"
     )},
    {"name": "adjust_right_eef_pose_in_head_frame",
     "endpoint": "/api/v2/adjust_right_eef_pose_in_head_frame",
     "args": [
         _F("forward", unit="m(head相机系向前，可负)"),
         _F("upward", unit="m(head相机系向上，可负)"),
         _F("leftward", unit="m(head相机系向左，可负)"),
         _F("roll", unit="deg(右夹爪局部轴，与adjust_plan_pose一致)"),
         _F("pitch", unit="deg(+翘头/fingertips-up)"),
         _F("yaw", unit="deg(右夹爪局部轴，与adjust_plan_pose一致)"),
         _F("pos_tol", 0.012, unit="m"),
         _F("ori_tol_deg", 3.0, unit="deg"),
     ],
     "desc": (
         "调整右 EEF 6D pose：平移按 head camera，旋转按右夹爪局部 "
         "roll/pitch/yaw；平移单位为米，完成后拍 head camera。"
     )},
    {"name": "adjust_left_eef_pose_in_wrist_frame",
     "endpoint": "/api/v2/adjust_left_eef_pose_in_wrist_frame",
     "args": [
         _F("forward", unit="m(+沿左腕视线向前/-向后)"),
         _F("leftward", unit="m(+画面向左/-向右，与底部x标尺方向相反)"),
         _F("upward", unit="m(+画面向上/-向下)"),
         _F("roll", unit="deg(夹爪局部 roll 增量；仅 J5-J7 求解)"),
         _F("pitch", unit="deg(夹爪局部 pitch 增量；+为指尖上抬)"),
         _F("yaw", unit="deg(夹爪局部 yaw 增量；仅 J5-J7 求解)"),
         _F("pos_tol", 0.012, unit="m"),
         _F("ori_tol_deg", 3.0, unit="deg(最终夹爪姿态收敛容差)"),
     ],
     "desc": (
         "以实时 left wrist camera 为机体系调整左 EEF；平移单位统一为米，"
         "roll/pitch/yaw 为夹爪局部轴增量。旋转阶段固定 J1-J4，仅求解和驱动 "
         "J5-J7；J8 不参与并保持锁定。"
         "执行后返回左腕视图。"
     )},
    {"name": "adjust_right_eef_pose_in_wrist_frame",
     "endpoint": "/api/v2/adjust_right_eef_pose_in_wrist_frame",
     "args": [
         _F("forward", unit="m(+沿右腕视线向前/-向后)"),
         _F("leftward", unit="m(+画面向左/-向右，与底部x标尺方向相反)"),
         _F("upward", unit="m(+画面向上/-向下)"),
         _F("roll", unit="deg(夹爪局部 roll 增量；仅 J5-J7 求解)"),
         _F("pitch", unit="deg(夹爪局部 pitch 增量；+为指尖上抬)"),
         _F("yaw", unit="deg(夹爪局部 yaw 增量；仅 J5-J7 求解)"),
         _F("pos_tol", 0.012, unit="m"),
         _F("ori_tol_deg", 3.0, unit="deg(最终夹爪姿态收敛容差)"),
     ],
     "desc": (
         "以实时 right wrist camera 为机体系调整右 EEF；平移单位统一为米，"
         "roll/pitch/yaw 为夹爪局部轴增量。旋转阶段固定 J1-J4，仅求解和驱动 "
         "J5-J7；J8 不参与并保持锁定。"
         "执行后返回右腕视图。"
     )},
    {"name": "move_point_to_point",
     "endpoint": "/api/v2/move_point_to_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "points", "widget": "multi_uv_arm", "required": True,
          "min_points": 2, "max_points": 2, "arm_options": ["any"]},
         _F("above_target_point_m", 0.0,
            unit="m(目标点沿机器人基座 +Z 的抬高量，0 表示目标点本身)"),
         _F("pos_tol", 0.012, unit="m"),
         _F("ori_tol_deg", 5.0, unit="deg"),
     ],
     "desc": (
         "在同一张 head RGB-D 图上依次点选待移动点和目标点。选择离待移动点最近且夹爪"
         "稳定处于非全开/非全合状态的手，保持 EEF 姿态和夹爪开度不变，把 EEF 按两点"
         "三维差值平移，使待移动点到达目标点沿机器人基座 +Z 上方指定距离；默认 0m。"
     )},
    {"name": "move_eef", "endpoint": "/api/v2/move_eef",
     "args": [
         _F("upward", unit="m(相机系向上，可负)"),
         _F("forward", unit="m(相机系向前/进场景，可负)"),
         _F("leftward", unit="m(相机系向左，可负)"),
         {"name": "u", "widget": "uv", "type": "integer", "required": False,
          "unit": "Qwen3-VL 相对坐标 0..1000（与 v/depth 同填）"},
         {"name": "v", "widget": "uv", "type": "integer", "required": False,
          "unit": "Qwen3-VL 相对坐标 0..1000（与 u/depth 同填）"},
         {"name": "depth", "widget": "number", "type": "number", "required": False,
          "unit": "m(可选；head-camera forward z-depth，与 Memory 一致)"},
         _F("pos_tol", 0.012, unit="m(与 move_eef 默认一致，小于此位移会判定已到位)"),
         {"name": "gripper", "widget": "select", "required": False,
          "options": ["keep", "open", "close"], "default": "keep"},
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left"], "default": "right"},
     ],
     "desc": "head 相机系 EEF 平移（m，姿态不变）；若同时输入 u/v/depth，则反解为目标 EEF 三维位置并执行。"},
    {"name": "plan_move_eef", "endpoint": "/api/v2/plan_move_eef",
     "args": [
         _F("upward", unit="m(相机系向上，可负)"),
         _F("forward", unit="m(相机系向前/进场景，可负)"),
         _F("leftward", unit="m(相机系向左，可负)"),
         {"name": "u", "widget": "uv", "type": "integer", "required": False,
          "unit": "Qwen3-VL 相对坐标 0..1000（与 v/depth 同填）"},
         {"name": "v", "widget": "uv", "type": "integer", "required": False,
          "unit": "Qwen3-VL 相对坐标 0..1000（与 u/depth 同填）"},
         {"name": "depth", "widget": "number", "type": "number", "required": False,
          "unit": "m(可选；head-camera forward z-depth，与 Memory 一致)"},
         _F("pos_tol", 0.012, unit="m(与 move_eef 默认一致，小于此位移会判定已到位)"),
         {"name": "gripper", "widget": "select", "required": False,
          "options": ["keep", "open", "close"], "default": "keep"},
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left"], "default": "right"},
     ],
     "desc": "规划并预览 move_eef：默认按相机系 m 增量；若同时输入 u/v/depth，则反解目标位置；返回前验证同分支6D直线路径。"},
    {"name": "plan_eef_translation_to_uvd_point", "endpoint": "/api/v2/plan_eef_translation_to_uvd_point",
     "args": [
         {"name": "u", "widget": "uv", "type": "integer", "required": True,
          "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "type": "integer", "required": True,
          "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "depth", "widget": "number", "type": "number", "required": True,
          "unit": "m(head-camera forward z-depth)"},
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left"], "default": "right"},
         _F("pos_tol", 0.012, unit="m"),
     ],
     "desc": "规划并预览 EEF 平移到 u/v/depth 反解出的3D点；返回前验证同分支6D直线路径；不接受增量参数。"},
    {"name": "plan_move_eef_to_point", "endpoint": "/api/v2/plan_move_eef_to_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True},
         {"name": "v", "widget": "uv", "required": True},
         _F("upward", 0.05, unit="m(世界Z正上方，可负)"),
         _F("pos_tol", 0.012, unit="m(传给 move_eef)"),
         {"name": "gripper", "widget": "select", "required": False,
          "options": ["keep", "open", "close"], "default": "keep"},
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left"], "default": "right"},
     ],
     "desc": "点选 head 图反解 3D，预览夹爪到该点世界Z正上方 upward m，并返回 move_eef 参数。"},
    {"name": "adjust_plan_pose", "endpoint": "/api/v2/adjust_plan_pose",
     "args": [
         {"name": "plan_id", "widget": "plan", "required": True},
         _F("forward", unit="m(head相机系平移，只改pos)"),
         _F("upward", unit="m(head相机系平移，只改pos)"),
         _F("leftward", unit="m(head相机系平移，只改pos)"),
         _F("roll", unit="deg(+按夹爪爪尖轴正向)"),
         _F("pitch", unit="deg(+翘头/fingertips-up)"),
         _F("yaw", unit="deg(+向左转，绕wrist camera轴)"),
     ],
     "desc": "调整已有 move 的 eef 6D pose：平移按 head 相机系，旋转按 move 的夹爪局部轴；返回新 plan_id 和红爪预览。"},
    {"name": "rotate_eef", "endpoint": "/api/v2/rotate_eef",
     "args": [
         _F("rotate_forward_deg", 0.0, unit="deg(+CCW沿视线进场景)"),
         _F("rotate_upward_deg", 0.0, unit="deg(+CCW从上往下看)"),
         _F("rotate_leftward_deg", 0.0, unit="deg(+CCW沿画面左轴)"),
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left"], "default": "right"},
     ],
     "desc": "head 相机系 EEF 旋转（deg，位置不变）；正=右手定则/CCW；顺序 fwd→up→left；完成后拍主视图。"},
    {"name": "move_to", "endpoint": "/api/v2/move_to",
     "args": [
         _F("x", 0.0, True, unit="m"),
         _F("y", 0.0, True, unit="m"),
         _F("z", unit="m"),
         _F("thetax", unit="deg"),
         _F("thetaz", unit="deg"),
     ],
     "desc": "世界系绝对 5D 移动；完成后自动拍主视图。"},
    {"name": "plan_push", "endpoint": "/api/v2/plan", "mode_prefix": "push_",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True},
              {"name": "direction", "widget": "select", "required": True,
               "options": ["left", "right", "forward", "backward", "up", "down"]}],
     "desc": "对 (u,v) 物体沿 direction 规划推动（闭合夹爪）。"},
    {"name": "plan_grasp_point", "endpoint": "/api/v2/plan", "mode": "grasp_point",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "对 (u,v) 点规划抓取（张开夹爪）。"},
    {"name": "plan_grasp_point_filter", "endpoint": "/api/v2/plan", "mode": "grasp_point_filter",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True},
              {"name": "plan_arm", "widget": "select", "required": False,
               "options": ["any", "left", "right"], "default": "any"},
              _F("seed", 42, unit="grasp 随机种子")],
     "desc": "对 (u,v) 点 3cm 邻域规划抓取；plan_arm=left/right 时只从对应手臂 IK 严格可达池继续 overlap/graspvol，any 保持自动策略。"},
    {"name": "plan_grasp_point_filter_rgbd", "endpoint": "/api/v2/plan",
     "mode": "grasp_point_filter_rgbd",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "points", "widget": "multi_uv_arm", "required": True,
               "max_points": 16,
               "arm_options": ["any", "left", "right"]},
              _F("seed", 42, unit="姿态采样种子")],
     "desc": "同一张冻结 head RGBD 可点选多个位置；每点独立选择 plan_arm 并生成可执行 plan_id，所有成功夹爪合并显示在同一张图。"},
    {"name": "plan_grasp_point_filter_rgbd_lite", "endpoint": "/api/v2/plan",
     "mode": "grasp_point_filter_rgbd_lite",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True},
              {"name": "plan_arm", "widget": "select", "required": False,
               "options": ["any", "left", "right"], "default": "any"},
              _F("seed", 42, unit="姿态采样种子")],
     "desc": "轻量 RGBD 抓取：候选、过滤和输出保持完整版本逻辑；冷启动 IK 使用 split8，完整 warm-start 阶段使用等状态量 warm32x6；不启用候选压缩。"},
    {"name": "plan_grasp_object", "endpoint": "/api/v2/plan", "mode": "grasp_obj",
     "args": [
         {"name": "object_name", "widget": "text", "required": False},
         {"name": "image_id", "widget": "image", "required": False},
         {"name": "u", "widget": "uv", "required": False},
         {"name": "v", "widget": "uv", "required": False},
         _F("seed", 42, unit="grasp 随机种子"),
     ],
     "one_of": [
         {"fields": ["object_name"], "label": "按物体名"},
         {"fields": ["image_id", "u", "v"], "label": "点选 head 图 (u,v)"},
     ],
     "desc": "整体抓取规划（需先 capture）：填 object_name，或在 head 图上点选 (u,v)。"},
    {"name": "plan_grasp_object_filter", "endpoint": "/api/v2/plan", "mode": "grasp_obj_filter",
     "args": [
         {"name": "object_name", "widget": "text", "required": False},
         {"name": "image_id", "widget": "image", "required": False},
         {"name": "u", "widget": "uv", "required": False},
         {"name": "v", "widget": "uv", "required": False},
         {"name": "plan_arm", "widget": "select", "required": False,
          "options": ["any", "left", "right"], "default": "any"},
         _F("seed", 42, unit="grasp 随机种子"),
     ],
     "one_of": [
         {"fields": ["object_name"], "label": "按物体名"},
         {"fields": ["image_id", "u", "v"], "label": "点选 head 图 (u,v)"},
     ],
     "desc": "整体抓取规划 + 左右臂严格 6D IK 过滤；plan_arm=left/right 时只从对应手臂 IK 严格可达池继续 overlap/graspvol，any 保持自动策略。"},
    {"name": "plan_open", "endpoint": "/api/v2/plan", "mode": "open",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "对 (u,v) 铰链物体规划开门。"},
    {"name": "plan_close", "endpoint": "/api/v2/plan", "mode": "close",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "对 (u,v) 铰链物体规划关门。"},
    {"name": "plan_press", "endpoint": "/api/v2/plan", "mode": "press",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "对 (u,v) 规划按压（沿表面内法向，闭合夹爪）。"},
    {"name": "plan_press_point", "endpoint": "/api/v2/plan", "mode": "press_point",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True},
              {"name": "plan_arm", "widget": "select", "required": False,
               "options": ["any", "left", "right"], "default": "any"},
              _F("seed", 42, unit="姿态采样种子")],
     "desc": (
         "点选 head 图反解 3D press 点；把闭爪真实爪尖严格约束到该点，"
         "以80个近似均匀夹爪轴方向×8个绕轴 roll 采样640个空间姿态；"
         "先用 GPU IK 硬过滤末端点与沿夹爪向量回退10cm的 safe 点，"
         "两点均须满足位置误差≤10mm、姿态误差≤3°；"
         "再按目标点外法向与爪尖→O_EEF夹爪向量夹角从小到大，"
         "用 GPU IK 逐个验证 current→safe→press 完整轨迹；"
         "plan_arm=any 时任意手臂产生完整轨迹即可，指定 left/right 时必须由指定手臂"
         "产生完整轨迹；候选排序不使用阈值内 IK 残差，只取夹角顺序中第一个"
         "轨迹有解的 pose 作为 move；"
         "执行前先对返回 arm 调用 close_gripper。"
     )},
    {"name": "plan_place", "endpoint": "/api/v2/plan", "mode": "place",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "手上有物体时，对 (u,v) 支撑面规划放置（张开释放）。"},
    {"name": "mark_object", "endpoint": "/api/v2/mark_object",
     "args": [{"name": "image_id", "widget": "image", "required": True},
              {"name": "u", "widget": "uv", "required": True},
              {"name": "v", "widget": "uv", "required": True}],
     "desc": "标记 (u,v) 指向的物体，写 memory.md，返回 bddl name。"},
    {"name": "move_to_object", "endpoint": "/api/v2/move_to_object",
     "args": [
         {"name": "image_id", "widget": "image", "required": False},
         {"name": "u", "widget": "uv", "required": False},
         {"name": "v", "widget": "uv", "required": False},
         {"name": "object_name", "widget": "text", "required": False},
         _F("reach", 0.0, unit="m(弦球半径，0=默认0.60m)"),
         _F("nav_timeout_s", 120.0, unit="s(底盘闭环导航超时，默认120)"),
     ],
     "pick_like": "mark_object",
     "one_of": [
         {"fields": ["object_name"], "label": "按 bddl 名称"},
         {"fields": ["image_id", "u", "v"], "label": "点选（同 mark_object，自动解析 bddl name）"},
     ],
     "desc": "object_name 或点选物体；球心=物体 AABB 中心；移动/姿态与 move_to_point 相同（xy→yaw→upward→q3俯仰）。"},
    {"name": "move_to_point", "endpoint": "/api/v2/move_to_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True},
         {"name": "v", "widget": "uv", "required": True},
         _F("reach", 0.0, unit="m(弦球半径，0=默认0.60m)"),
         _F("nav_timeout_s", 120.0, unit="s(底盘闭环导航超时，默认120)"),
     ],
     "pick_like": "mark_object",
     "desc": "点选 (u,v) 光束反解 3D 点为弦球球心；移动/姿态与 move_to_object 相同（xy→yaw→upward→q3俯仰）。"},
    {"name": "move_to_reach_point", "endpoint": "/api/v2/move_to_reach_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True},
         {"name": "v", "widget": "uv", "required": True},
         _F("reach", 0.0, unit="m(弦球半径，0=默认0.60m)"),
         _F("nav_timeout_s", 120.0, unit="s(底盘闭环导航超时，默认120)"),
         {"name": "keep_ori_arm", "widget": "select", "required": False,
          "options": ["none", "left", "right", "both"], "default": "none"},
     ],
     "pick_like": "mark_object",
     "desc": "点选 (u,v) 反解 3D reach point，并移动到底盘/肩部可达位置；keep_ori_arm 仅在底盘旋转完成后的最终俯仰阶段生效，保持原 J1-J4 轨迹不变并逐帧仅用 J5-J7 追踪入口世界系 EEF 姿态；EEF 平移只监控。"},
    {"name": "move_base_to_point", "endpoint": "/api/v2/move_base_to_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         _F("nav_timeout_s", 120.0, unit="s(XY 平移阶段超时)"),
         _F("ground_tol_m", 0.04, unit="m(反解点离地面顶面的容差)"),
         _F("pos_tol_m", 0.12, unit="m(XY 到点容差)"),
     ],
     "pick_like": "mark_object",
     "desc": "点选可见地面点：反解 3D 必须在 floor/floors 顶面；保持当前朝向，以 forward/leftward 同步平移，使蓝色路径起始线中点到达点选位置；躯干/双臂/夹爪保持不动。"},
    {"name": "move_chassis_to_floor_point", "endpoint": "/api/v2/move_chassis_to_floor_point",
     "args": [
         {"name": "image_id", "widget": "image", "required": True},
         {"name": "u", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         {"name": "v", "widget": "uv", "required": True, "unit": "Qwen3-VL 相对坐标 0..1000"},
         _F("nav_timeout_s", 120.0, unit="s(XY 平移阶段超时)"),
         _F("ground_tol_m", 0.04, unit="m(反解点离地面顶面的容差)"),
         _F("pos_tol_m", 0.12, unit="m(XY 到点容差)"),
     ],
     "pick_like": "mark_object",
     "desc": "点选可见地板/地面点：保持当前 yaw，按起始机体系 forward/leftward 向量直接同步平移，使蓝色路径起始线中点到达点选位置；身体不动。"},
    {"name": "measure_shoulder_distance", "endpoint": "/api/v2/measure_shoulder_distance",
     "args": [
         {"name": "object_name", "widget": "text", "required": False},
         {"name": "image_id", "widget": "image", "required": False},
         {"name": "u", "widget": "uv", "required": False},
         {"name": "v", "widget": "uv", "required": False},
     ],
     "pick_like": "mark_object",
     "one_of": [
         {"fields": ["object_name"], "label": "按物体名"},
         {"fields": ["image_id", "u", "v"], "label": "点选 head 图反解 3D"},
     ],
     "desc": "测量当前左右肩到目标距离；object_name 优先，使用 AABB 中心；否则点选 head 图反解 3D。"},
    {"name": "exec_move", "endpoint": "/api/v2/exec_move",
     "args": [{"name": "plan_id", "widget": "plan", "required": True},
              {"name": "arm", "widget": "select", "required": False,
               "options": ["", "left", "right"]},
              {"name": "back_m", "widget": "text", "required": False,
               "default": "0.10",
               "placeholder": "后退安全距离(m)，默认0.10"}],
     "desc": "执行 plan_id 对应的动作；arm 可选 left/right；back_m 控制安全后退距离(默认0.10m)；完成后自动拍主视图。"},
    {"name": "exec_eef_pose", "endpoint": "/api/v2/exec_eef_pose",
     "args": [{"name": "plan_id", "widget": "plan", "required": True},
              {"name": "arm", "widget": "select", "required": False,
               "options": ["", "left", "right"]},
              {"name": "back_m", "widget": "text", "required": False,
               "default": "0.10",
               "placeholder": "后退安全距离(m)，默认0.10"}],
     "desc": "仅移动到 eef_pose：当前→safe→目标，不合爪；press_point 回放规划阶段 GPU IK anchors；完成后拍主视图。"},
    {"name": "exec_plan_pose", "endpoint": "/api/v2/exec_plan_pose",
     "args": [{"name": "plan_id", "widget": "plan", "required": True},
              {"name": "arm", "widget": "select", "required": False,
               "options": ["", "left", "right"]},
              {"name": "back_m", "widget": "text", "required": False,
               "default": "0.10",
               "placeholder": "后退安全距离(m)，默认0.10"}],
     "desc": "执行 plan_id 对应的 EEF pose：当前→safe→目标 pose，不合爪；press_point 直接回放规划阶段已验证的 GPU IK anchors；完成后拍实际执行臂的 wrist camera。"},
    {"name": "arm_reset", "endpoint": "/api/v2/arm_reset",
     "invoke": "skill",
     "skill_name": "arm_reset",
     "args": [
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left", "both"], "default": "right"},
         {"name": "mode", "widget": "select", "required": False,
          "options": ["grasp", "hang", "ready"], "default": "grasp"},
         {"name": "open_gripper", "widget": "select", "required": False,
          "options": ["true", "false"], "default": "true"},
     ],
     "desc": "手臂复位；默认 grasp prep，避免升降时自然下垂手臂撑到底盘；不动底盘/腰/躯干。"},
    {"name": "set_arm_to_grasp_position", "endpoint": "/api/v2/set_arm_to_grasp_position",
     "invoke": "skill",
     "skill_name": "set_arm_to_grasp_position",
     "args": [
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left", "both"], "default": "right"},
         {"name": "keep_ori_arm", "widget": "select", "required": False,
          "options": ["none", "left", "right", "both"], "default": "none"},
         {"name": "gripper", "widget": "select", "required": False,
          "options": ["open", "keep"], "default": "keep"},
         _F("max_dq_per_step", 0.30, unit="rad/frame"),
         _F("tol", 0.08, unit="rad"),
         _F("timeout_s", 15.0, unit="s"),
     ],
     "desc": "抓取预备：普通规划器照常生成J1-J4运动；keep_ori_arm记录入口世界系夹爪RPY，并在每个执行帧仅反解J5-J7追踪该RPY；gripper=open/keep，默认keep。"},
    {"name": "set_arm_to_grasp_position_shortcut", "endpoint": "/api/v2/set_arm_to_grasp_position_shortcut",
     "invoke": "skill",
     "skill_name": "set_arm_to_grasp_position_shortcut",
     "args": [
         {"name": "arm", "widget": "select", "required": False,
          "options": ["right", "left", "both"], "default": "right"},
         {"name": "open_gripper", "widget": "select", "required": False,
          "options": ["true", "false"], "default": "true"},
         _F("max_dq_per_step", 1.2, unit="rad/step"),
         _F("tol", 0.08, unit="rad"),
         _F("timeout_s", 35.0, unit="s"),
     ],
     "desc": "旧版shortcut：直接设定并锁定grasp prep关节，不做EEF直线IK规划。"},
    {"name": "reset_body", "endpoint": "/api/v2/reset_body",
     "args": [
         {"name": "keep_ori_arm", "widget": "select", "required": False,
          "options": ["none", "left", "right", "both"], "default": "none"},
         _F("timeout_s", 45.0, unit="s"),
         _F("trunk_max_step", 0.06, unit="rad/步"),
         _F("shoulder_iters", 4, unit="兼容保留/已忽略"),
     ],
     "desc": "腰部三关节复位直立；keep_ori_arm 保持入口 J1-J4 命令不变，每帧仅用 J5-J7 追踪入口世界系 EEF 姿态；EEF 平移只监控；其余手臂全关节锁定；完成后拍主视图。"},
    {"name": "manipulate_add_vector_to_point", "endpoint": "/api/v2/manipulate_add_vector_to_point",
     "args": [
         {"name": "image_id", "widget": "text", "required": False,
          "placeholder": "可选；留空自动使用本 session 最新可反投影 capture"},
         {"name": "u", "widget": "text", "required": False, "placeholder": "单点 u(0..1000)"},
         {"name": "v", "widget": "text", "required": False, "placeholder": "单点 v(0..1000)"},
         {"name": "points", "widget": "text", "required": False,
          "placeholder": "多点 JSON: [[u1,v1],[u2,v2]]"},
         {"name": "length_m", "widget": "text", "required": False, "default": "0.1",
          "placeholder": "向量长度(m)，默认0.1"},
     ],
     "desc": "在所选 0..1000 相对坐标点(可多点)反投影出物体表面点；image_id 可省略，自动使用本 session 最新且含 depth 的 capture。PCA 估计局部坐标系并创建向量(法向=接近向)，返回向量编号。"},
    {"name": "manipulate_move_vector_to_vector", "endpoint": "/api/v2/manipulate_move_vector_to_vector",
     "args": [
         {"name": "from_vector", "widget": "text", "required": True,
          "placeholder": "待移动向量 id (如 vec_0001)"},
         {"name": "to_vector", "widget": "text", "required": True,
          "placeholder": "目标向量 id (如 vec_0002)"},
         {"name": "back_m", "widget": "text", "required": False, "default": "0.0",
          "placeholder": "安全后退距离(m)，默认0"},
     ],
     "desc": "把 from_vector 移到 to_vector：先检查其表面点所属物体是否被夹爪抓住，未抓取报错；抓住则按刚体位移/旋转把当前 EEF 位姿变换到目标并驱动手臂到位(纯手臂)。"},
]


# 只在 BEHAVIOR_SPATIAL_MAP=1 的测试口暴露
SPATIAL_MAP_TOOLS = ("query_map", "mark_on_map", "navigate_to")


def reload_v2_tools_metadata() -> list:
    """Return current v2 metadata without reloading this web module.

    Reloading `behavior_interface.web` from `/api/v2/tools` resets module
    globals, including the live Run/Stop agent Popen handle.  Tool metadata can
    still hot-reload the tool-version modules via `visible_v2_tools()`, while
    changes to this web file should be applied by restarting the interface (or
    by one deliberate reload when no agent is running).
    """
    return list(visible_v2_tools())


def _public_coordinate_tool_metadata(tools: list) -> list:
    normalized = []
    for tool in tools:
        item = dict(tool)
        args = []
        for arg in tool.get("args") or []:
            param = dict(arg)
            if str(param.get("widget") or "") == "uv":
                param["type"] = "integer"
                param["minimum"] = 0
                param["maximum"] = COORDINATE_MAX
                param["unit"] = "Qwen3-VL relative coordinate 0..1000"
                param["coordinate_system"] = COORDINATE_SYSTEM
            args.append(param)
        item["args"] = args
        item["coordinate_system"] = COORDINATE_SYSTEM
        item["coordinate_range"] = {
            "u": [0, COORDINATE_MAX],
            "v": [0, COORDINATE_MAX],
        }
        normalized.append(item)
    return normalized


def visible_v2_tools() -> list:
    """Return v2 tool metadata for the active tool-version profile."""
    import importlib
    import sys

    for mod_name in ("behavior_interface.tool", f"behavior_interface.tool.{active_tool_version()}"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            importlib.reload(mod)
    names = active_v2_tool_names()
    if names is None:
        from behavior_interface.spatial_map import spatial_map_enabled

        tools = list(V2_TOOLS)
        if not spatial_map_enabled():
            tools = [
                item for item in tools if item.get("name") not in SPATIAL_MAP_TOOLS
            ]
        return _public_coordinate_tool_metadata(tools)
    by_name = {str(tool.get("name")): tool for tool in V2_TOOLS}
    aliases = {
        "mesure_shoulder_distance": (
            "measure_shoulder_distance",
            "/api/v2/mesure_shoulder_distance",
        ),
    }
    selected = []
    for name in names:
        tool = by_name.get(name)
        if tool is None and name in aliases:
            canonical_name, endpoint = aliases[name]
            canonical = by_name.get(canonical_name)
            if canonical is not None:
                tool = {
                    **canonical,
                    "name": name,
                    "endpoint": endpoint,
                }
        if tool is not None:
            selected.append(tool)
    from behavior_interface.spatial_map import spatial_map_enabled

    if spatial_map_enabled():
        present = {str(item.get("name")) for item in selected}
        for extra_name in SPATIAL_MAP_TOOLS:
            extra = by_name.get(extra_name)
            if extra is not None and extra_name not in present:
                selected.append(extra)
    return _public_coordinate_tool_metadata(
        selected
    )


def _encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def build_app(server) -> Flask:
    """server: BehaviorInterface 实例（在 server.py 定义）"""
    app = Flask(__name__, template_folder="templates")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    human_recorder = HumanTrajectoryRecorder()
    app.extensions["human_trajectory_recorder"] = human_recorder
    monitor_store = AgentMonitorStore(
        port=getattr(server, "web_port", None),
        task_id=getattr(server, "task_id", None),
        task_name=getattr(server, "task_name", None),
        tick_reader=lambda: int(getattr(server, "tick", 0) or 0),
    )
    app.extensions["agent_monitor_store"] = monitor_store

    def _current_server_tick() -> int:
        try:
            return int(getattr(server, "tick", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _on_monitor_world_reset() -> None:
        try:
            monitor_store.on_world_reset(sim_tick=_current_server_tick())
        except Exception:
            return

    hooks = getattr(server, "_after_world_reset_hooks", None)
    if not isinstance(hooks, list):
        server._after_world_reset_hooks = []
        hooks = server._after_world_reset_hooks
    hooks.append(_on_monitor_world_reset)

    def _privileged_scene_api_disabled() -> bool:
        from behavior_interface.memory import task_goals_only_memory_enabled

        return task_goals_only_memory_enabled()

    def _recording_state_snapshot() -> dict[str, Any]:
        try:
            state = server.snapshot_state()
        except Exception as exc:
            return {"snapshot_error": str(exc)}
        keep = (
            "pid",
            "started_ts",
            "task",
            "scene",
            "instance_id",
            "mode",
            "tick",
            "fps",
            "reset_count",
            "task_switch_count",
            "base_pose",
            "eef_pose",
            "active_skill",
            "tro",
            "goals",
            "memory",
        )
        return {
            key: state.get(key)
            for key in keep
            if key in state
        }

    def _recording_memory_text() -> str:
        try:
            return str(server.get_memory_text() or "")
        except Exception as exc:
            return f"[memory unavailable: {exc}]"

    def _start_continuous_capture(status: dict[str, Any]) -> dict[str, Any]:
        """点 record 的同时开连续录制，产出离线 SLAM 能直接重放的 bundle。

        写到会话目录下的 capture/，与 turns/、images/ 并列。
        Flask 线程立刻 prime：目录和 origin 帧马上落盘，不等下一拍 act()。
        """
        try:
            from behavior_interface.continuous_capture import (
                CAPTURE,
                continuous_capture_enabled,
            )
            from behavior_interface.rtabmap_slam.capture import CAPTURE_SCHEMA
        except Exception as exc:
            return {"ok": False, "error": f"unavailable: {exc}"}
        if not continuous_capture_enabled():
            return {"ok": False, "enabled": False}
        run_dir = str(status.get("run_dir") or "")
        if not run_dir:
            return {"ok": False, "error": "recording has no run_dir"}
        capture_dir = os.path.join(run_dir, "capture")
        started = CAPTURE.request_start(capture_dir)
        if not started.get("ok"):
            return started
        primed = CAPTURE.prime(getattr(server, "world", None))
        pointer = {
            "capture_dir": capture_dir,
            "schema": CAPTURE_SCHEMA,
            "status": primed,
        }
        try:
            with open(os.path.join(run_dir, "continuous_capture.json"), "w", encoding="utf-8") as handle:
                json.dump(pointer, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except Exception:
            pass
        return primed

    def _stop_continuous_capture() -> dict[str, Any]:
        """当场收尾：manifest 没有 closed 就无法离线重放。"""
        try:
            from behavior_interface.continuous_capture import CAPTURE
        except Exception as exc:
            return {"ok": False, "error": f"unavailable: {exc}"}
        return CAPTURE.stop_now(getattr(server, "world", None))

    def _attach_continuous_capture(recording: dict[str, Any]) -> dict[str, Any]:
        recording["continuous_capture"] = _continuous_capture_status()
        return recording

    @app.before_request
    def _begin_human_v2_recording():
        if request.method != "POST" or not request.path.startswith("/api/v2/"):
            return None
        if request.headers.get("X-Behavior-Source", "").strip() != "human-ui":
            return None
        record_id = request.headers.get("X-Behavior-Record-Id", "").strip()
        if not record_id:
            return None

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        status = human_recorder.status(record_id)
        if not status.get("active"):
            return jsonify({
                "ok": False,
                "error": f"human recording is not active: {record_id}",
                "recording": status,
            }), 409
        if str(body.get("session_id") or "").strip() != status.get("session_id"):
            return jsonify({
                "ok": False,
                "error": "recording session_id does not match request session_id",
                "expected_session_id": status.get("session_id"),
            }), 409

        tool_name = (
            request.headers.get("X-Behavior-Tool-Name", "").strip()
            or request.path.rsplit("/", 1)[-1]
        )
        input_media = {
            "kind": request.headers.get("X-Behavior-Input-Media-Kind", "").strip(),
            "image_id": (
                request.headers.get("X-Behavior-Input-Image-Id", "").strip()
                or str(body.get("image_id") or "").strip()
            ),
            "plan_id": (
                request.headers.get("X-Behavior-Input-Plan-Id", "").strip()
                or str(body.get("plan_id") or "").strip()
            ),
            "path": request.headers.get(
                "X-Behavior-Input-Media-Path", ""
            ).strip(),
        }
        input_media = {key: value for key, value in input_media.items() if value}
        try:
            g.human_recording_context = human_recorder.begin_tool(
                record_id=record_id,
                tool_name=tool_name,
                endpoint=request.path,
                request_args=body,
                input_media=input_media,
                pre_state=_recording_state_snapshot(),
                pre_memory_text=_recording_memory_text(),
                bootstrap=request.headers.get(
                    "X-Behavior-Record-Bootstrap", ""
                ).strip() == "1",
            )
            g.human_recording_finished = False
        except (RecordingConflict, RecordingNotFound) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return None

    @app.after_request
    def _finish_human_v2_recording(response):
        context = getattr(g, "human_recording_context", None)
        if context is None or getattr(g, "human_recording_finished", False):
            return response
        payload = response.get_json(silent=True)
        if payload is None:
            payload = {
                "ok": response.status_code < 400,
                "body": response.get_data(as_text=True),
            }
        human_recorder.finish_tool(
            context,
            response=payload,
            http_status=response.status_code,
            post_state=_recording_state_snapshot(),
            post_memory_text=_recording_memory_text(),
        )
        g.human_recording_finished = True
        return response

    @app.teardown_request
    def _fail_human_v2_recording(exc):
        context = getattr(g, "human_recording_context", None)
        if (
            exc is None
            or context is None
            or getattr(g, "human_recording_finished", False)
        ):
            return
        human_recorder.finish_tool(
            context,
            response={"ok": False, "error": str(exc)},
            http_status=500,
            post_state=_recording_state_snapshot(),
            post_memory_text=_recording_memory_text(),
            exception=repr(exc),
        )
        g.human_recording_finished = True

    @app.before_request
    def _begin_agent_monitor_card():
        if request.method != "POST" or not request.path.startswith("/api/v2/"):
            return None
        if request.path.rstrip("/").endswith("/tools"):
            return None
        if request.headers.get("X-Behavior-Source", "").strip() == "human-ui":
            return None
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return None
        session_id = str(body.get("session_id") or "").strip()
        if not is_external_agent_session(session_id):
            return None
        tool_name = (
            request.headers.get("X-Behavior-Tool-Name", "").strip()
            or resolve_v2_tool_name(request.path, body, V2_TOOLS)
        )
        g.agent_monitor_card = monitor_store.begin_card(
            session_id=session_id,
            tool=tool_name,
            args=body,
        )
        g.agent_monitor_card_finished = False
        return None

    @app.after_request
    def _finish_agent_monitor_card(response):
        context = getattr(g, "agent_monitor_card", None)
        if context is None or getattr(g, "agent_monitor_card_finished", False):
            return response
        payload = response.get_json(silent=True)
        monitor_store.finish_card(
            context,
            payload if isinstance(payload, dict) else {},
            sim_tick=_current_server_tick(),
        )
        g.agent_monitor_card_finished = True
        return response

    @app.teardown_request
    def _fail_agent_monitor_card(exc):
        context = getattr(g, "agent_monitor_card", None)
        if (
            context is None
            or getattr(g, "agent_monitor_card_finished", False)
        ):
            return
        monitor_store.finish_card(
            context,
            {"ok": False, "error": str(exc) if exc else "request failed"},
            sim_tick=_current_server_tick(),
        )
        g.agent_monitor_card_finished = True

    @app.route("/")
    def index():
        from flask import make_response
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/api/human_recording/start", methods=["POST"])
    def api_human_recording_start():
        payload = request.get_json(force=True, silent=True) or {}
        tools = reload_v2_tools_metadata()
        try:
            status = human_recorder.start(
                state=_recording_state_snapshot(),
                memory_text=_recording_memory_text(),
                tool_version=active_tool_version(),
                tools=tools,
                label=str(payload.get("label") or ""),
                note=str(payload.get("note") or ""),
            )
        except RecordingConflict as exc:
            return jsonify({
                "ok": False,
                "error": str(exc),
                "recording": human_recorder.status(),
            }), 409
        status["continuous_capture"] = _start_continuous_capture(status)
        return _json_no_store({"ok": True, "recording": status})

    def _continuous_capture_status() -> dict[str, Any]:
        try:
            from behavior_interface.continuous_capture import CAPTURE

            return CAPTURE.status()
        except Exception as exc:
            return {"ok": False, "error": f"unavailable: {exc}"}

    @app.route("/api/human_recording/status", methods=["GET"])
    def api_human_recording_status():
        record_id = request.args.get("record_id", "").strip() or None
        recording = _attach_continuous_capture(human_recorder.status(record_id))
        return _json_no_store({
            "ok": True,
            "recording": recording,
        })

    @app.route("/api/human_recording/stop", methods=["POST"])
    def api_human_recording_stop():
        payload = request.get_json(force=True, silent=True) or {}
        record_id = str(payload.get("record_id") or "").strip()
        outcome = str(payload.get("outcome") or "partial").strip().lower()
        if not record_id:
            return jsonify({"ok": False, "error": "missing record_id"}), 400
        if outcome not in {"success", "partial", "failure", "aborted"}:
            return jsonify({
                "ok": False,
                "error": f"invalid outcome: {outcome}",
            }), 400
        try:
            status = human_recorder.stop(
                record_id,
                outcome=outcome,
                reason=str(payload.get("reason") or "user_stop"),
                state=_recording_state_snapshot(),
                memory_text=_recording_memory_text(),
            )
        except RecordingNotFound:
            return jsonify({
                "ok": False,
                "error": f"unknown recording: {record_id}",
            }), 404
        status["continuous_capture"] = _stop_continuous_capture()
        return _json_no_store({"ok": True, "recording": status})

    @app.route("/video/<feed>")
    def video(feed: str):
        boundary = b"--frame"

        def gen():
            last_id = -1
            while True:
                with server.frame_lock:
                    item = server.frames.get(feed)
                    fid = int(getattr(server, "frame_ids", {}).get(feed, server.frame_id))
                if item is None or fid == last_id:
                    time.sleep(0.01)
                    continue
                last_id = fid
                jpg = _encode_jpeg(item)
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode()
                    + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    _FRAME_FEEDS = frozenset({"main", "head", "left_wrist", "right_wrist"})

    @app.route("/api/frame/<feed>.jpg")
    def api_frame_jpg(feed: str):
        """单帧 JPEG 快照（短连接），供副视图轮询；避免 4 路 MJPEG 占满浏览器 HTTP/1.1 连接。"""
        if feed not in _FRAME_FEEDS:
            return jsonify({"ok": False, "error": f"unknown feed {feed!r}"}), 404
        with server.frame_lock:
            item = server.frames.get(feed)
            fid = int(getattr(server, "frame_ids", {}).get(feed, server.frame_id))
            updated_ts = float(getattr(server, "frame_updated_ts", {}).get(feed, 0.0))
        if item is None:
            return Response(status=503)
        jpg = _encode_jpeg(item)
        if not jpg:
            return Response(status=503)
        return Response(
            jpg,
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Frame-Id": str(fid),
                "X-Frame-Age-Ms": str(max(0, int((time.time() - updated_ts) * 1000))) if updated_ts else "0",
            },
        )

    @app.route("/api/spatial_map.png")
    def api_spatial_map_png():
        """实时俯视地图快照，供 UI 小地图槽位轮询。只读，不改地图状态。"""
        from behavior_interface.spatial_map import (
            active_session_id,
            map_snapshot_png,
            spatial_map_enabled,
        )

        if not spatial_map_enabled():
            return Response(status=404)
        # 实时建图在 agent session 建立之前就开始了，UI 只看「当前活跃的那张」；
        # 用 UI 自己的 session id 去取会永远是空图。
        session_id = (
            active_session_id()
            or (request.args.get("session_id") or "").strip()
            or "default"
        )
        heading_up = request.args.get("north_up") not in {"1", "true", "on"}
        try:
            from behavior_interface.rtabmap_slam.live import (
                get_live_mapper,
                live_backend_selected,
            )

            if live_backend_selected():
                mapper = get_live_mapper()
                if mapper is None:
                    return Response(status=503)
                mapper.adopt_session(session_id)
                png, version = mapper.map_snapshot_png(heading_up=heading_up)
            else:
                png, version = map_snapshot_png(session_id, heading_up=heading_up)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503
        except Exception as exc:  # 地图渲染失败不该拖垮 UI 轮询
            return jsonify({"ok": False, "error": str(exc)}), 500
        return Response(
            png,
            mimetype="image/png",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Map-Version": version,
            },
        )

    @app.route("/api/spatial_map/status")
    def api_spatial_map_status():
        """只读后端诊断，明确当前是否真的在跑独立 RTAB-Map worker。"""
        try:
            from behavior_interface.rtabmap_slam.live import (
                get_live_mapper,
                live_backend_selected,
            )

            if live_backend_selected():
                mapper = get_live_mapper()
                return _json_no_store(
                    mapper.status() if mapper is not None else {"enabled": False}
                )
            mapper = server._live_mapper()
            return _json_no_store(
                mapper.status() if mapper is not None else {"enabled": False}
            )
        except Exception as exc:
            return _json_no_store({"enabled": False, "error": str(exc)})

    @app.route("/api/ui_poll")
    def api_ui_poll():
        """合并 state + log + memory 为一次 JSON，减少并行 HTTP 占用连接槽。"""
        with server.state_lock:
            log_tail = list(server.log_lines)[-40:]
        state_payload = dict(server.snapshot_state())
        # 前端不用这两坨，但里面有 data-url，ui_poll 能到 5MB+，
        # 远程 4s 超时后 monitor 根本不会画。
        state_payload.pop("last_skill_results", None)
        state_payload.pop("skill_history", None)
        eval_control = eval_control_snapshot(
            server,
            official="official" in str(state_payload.get("mode") or ""),
        )
        if eval_control.get("current_instance_id") is not None:
            state_payload["instance_id"] = eval_control["current_instance_id"]
        memory_payload = {
            "summary": server.get_memory_summary(),
            "text": server.get_memory_text(),
        }
        return _json_no_store({
            "state": state_payload,
            "log": log_tail,
            "memory": memory_payload,
            "agent_monitor": _compact_agent_monitor_payload(
                _agent_monitor_payload_for_server(server, state_payload, monitor_store)
            ),
            "human_recording": _attach_continuous_capture(human_recorder.status()),
            "eval_control": eval_control,
            "tool_version": active_tool_version(),
            # Compatibility for browser tabs that loaded the previous index.html.
            # New UI reads "memory"; old UI reads "scene_graph" into the same
            # right-side panel, so feed it memory instead of stale scene graph.
            "scene_graph": memory_payload,
        })

    @app.route("/api/agent_monitor/session_begin", methods=["POST"])
    def api_agent_monitor_session_begin():
        """测试脚本开 Claude 前调用：钉住本 session 的 tick 零点。"""
        body = request.get_json(silent=True) or {}
        session_id = str(body.get("session_id") or "").strip()
        if not is_external_agent_session(session_id):
            return jsonify({"ok": False, "error": "invalid session_id"}), 400
        monitor_store.bind_task(
            getattr(server, "task_id", None),
            getattr(server, "task_name", None),
        )
        info = monitor_store.begin_session_clock(
            session_id,
            sim_tick=_current_server_tick(),
        )
        return jsonify({"ok": True, **info})

    @app.route("/api/agent_monitor/publish", methods=["POST"])
    @app.route("/api/agent_monitor", methods=["GET", "POST"])
    def api_agent_monitor():
        if request.method == "POST":
            max_bytes = _agent_monitor_publish_max_bytes()
            content_length = request.content_length
            if content_length is not None and content_length > max_bytes:
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload too large",
                    "max_bytes": max_bytes,
                    "content_length": content_length,
                }), 413
            raw = request.get_data(cache=False)
            if len(raw) > max_bytes:
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload too large",
                    "max_bytes": max_bytes,
                    "content_length": len(raw),
                }), 413
            try:
                payload = json.loads(raw)
            except Exception as exc:
                return jsonify({
                    "ok": False,
                    "error": f"invalid JSON: {exc}",
                }), 400
            if not isinstance(payload, dict):
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload must be a JSON object",
                }), 400

            session_id = str(payload.get("session_id") or "").strip()
            if not session_id:
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload requires session_id",
                }), 400

            state = server.snapshot_state()
            if payload.get("server_pid") is None:
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload requires server_pid",
                }), 400
            if payload.get("server_reset_count") is None:
                return jsonify({
                    "ok": False,
                    "error": "agent monitor payload requires server_reset_count",
                }), 400
            try:
                if int(payload["server_pid"]) != int(state.get("pid")):
                    return jsonify({
                        "ok": False,
                        "error": "agent monitor server_pid does not match current interface",
                        "expected_server_pid": state.get("pid"),
                    }), 409
                if int(payload["server_reset_count"]) != int(state.get("reset_count") or 0):
                    return jsonify({
                        "ok": False,
                        "error": "agent monitor reset_count does not match current interface",
                        "expected_server_reset_count": state.get("reset_count"),
                    }), 409
            except (TypeError, ValueError):
                return jsonify({
                    "ok": False,
                    "error": "agent monitor server_pid/reset_count metadata is invalid",
                }), 400

            now = time.time()
            payload = dict(payload)
            payload.setdefault("ok", True)
            payload.setdefault("updated_at", now)
            current, reason = _agent_monitor_payload_is_current(payload, state)
            if not current:
                return jsonify({
                    "ok": False,
                    "error": reason,
                }), 409

            payload["session_id"] = session_id
            payload["monitor_source"] = "published"
            payload["published_at"] = now
            payload["heartbeat_ts"] = now
            payload.setdefault(
                "agent_id",
                "skillopt" if session_id.startswith("skillopt-") else "external",
            )
            payload.setdefault("task", state.get("task"))
            payload.setdefault("port", getattr(server, "web_port", None))
            payload.setdefault("host", socket.gethostname())
            payload.setdefault("interface_pid", state.get("pid"))
            _store_published_agent_monitor_payload(payload)
            return _json_no_store({
                "ok": True,
                "accepted": True,
                "session_id": session_id,
                "server_pid": state.get("pid"),
                "server_reset_count": state.get("reset_count"),
                "published_at": now,
                "bytes": len(raw),
                "max_bytes": max_bytes,
            })

        # The pool polls this every monitor tick.  ``snapshot_state`` builds
        # the full browser status (tracker replay, tool registry, skill lock).
        # A 2s client timeout abandons the socket and this thread stays in
        # CLOSE_WAIT; three lanes passed 300 threads and stopped serving
        # tools.  The poll only needs session id and tick count, which the
        # monitor store already keeps.  ``?attempt=`` still takes the snapshot.
        payload = _agent_monitor_payload_for_server(
            server, state={"poll": True}, store=monitor_store
        )
        attempt_id = str(request.args.get("attempt") or "").strip()
        if attempt_id:
            payload = monitor_store.snapshot(
                server.snapshot_state(),
                attempt_id=attempt_id,
            )
        if request.args.get("full") in {"1", "true", "yes"}:
            return _json_no_store(payload)
        return _json_no_store(_compact_agent_monitor_payload(payload))

    @app.route("/api/agent_monitor/model_image", methods=["POST"])
    def api_agent_monitor_model_image():
        """MCP 把发给模型的那张 JPEG 回写到卡片，监视器和 Codex 看同一张图。"""
        body = request.get_json(force=True, silent=True) or {}
        raw_b64 = str(body.get("image_b64") or body.get("image") or "").strip()
        if raw_b64.startswith("data:") and "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            data = base64.b64decode(raw_b64, validate=False) if raw_b64 else b""
        except Exception:
            data = b""
        if not data:
            return jsonify({"ok": False, "error": "model image required"}), 400
        if len(data) > 512 * 1024:
            return jsonify({"ok": False, "error": "model image too large"}), 413
        snapshot = monitor_store.apply_model_image(
            str(body.get("session_id") or "").strip(),
            str(body.get("image_id") or "").strip(),
            data,
            tool=str(body.get("tool") or "").strip(),
        )
        status = 200 if snapshot.get("ok") else 400
        return _json_no_store(snapshot), status

    @app.route("/api/agent_monitor/prompt", methods=["POST"])
    def api_agent_monitor_prompt():
        body = request.get_json(force=True, silent=True) or {}
        prompt = str(body.get("user_prompt") or body.get("prompt") or "").strip()
        loaded_skills = body.get("loaded_skills") or body.get("skills")
        if not prompt and not loaded_skills:
            return jsonify({"ok": False, "error": "prompt required"}), 400
        session_id = str(body.get("session_id") or "").strip()
        snapshot = monitor_store.set_user_prompt(
            prompt,
            session_id=session_id or None,
            loaded_skills=loaded_skills,
            new_attempt=body.get("new_attempt"),
        )
        return _json_no_store({
            "ok": True,
            "accepted": True,
            "session_id": snapshot.get("session_id") or "",
            "user_prompt": snapshot.get("user_prompt") or prompt,
            "loaded_skills": snapshot.get("loaded_skills") or [],
        })

    @app.route("/api/agent_monitor/image")
    def api_agent_monitor_image():
        card_id = request.args.get("card_id")
        if card_id not in {None, ""}:
            path = monitor_store.card_image_path(
                request.args.get("session") or "",
                card_id,
                request.args.get("kind") or "output",
            )
            if not path:
                return jsonify({"ok": False, "error": "agent monitor card image not found"}), 404
            mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            return send_file(
                path,
                mimetype=mime,
                conditional=True,
                max_age=60,
            ), {"Cache-Control": "private, max-age=60"}
        payload = _agent_monitor_payload_for_server(server, store=monitor_store)
        image = payload.get("image") if isinstance(payload, dict) else None
        path = (image or {}).get("rgb_path") if isinstance(image, dict) else None
        if not path or not os.path.isfile(path):
            return jsonify({"ok": False, "error": "agent monitor image not found"}), 404
        return send_file(
            path,
            mimetype="image/png",
            conditional=False,
            max_age=0,
        ), {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

    @app.route("/api/state")
    def api_state():
        # snapshot_state 内部已经持锁，外层不能再持同一把锁（非 reentrant Lock 会死锁）
        return jsonify(server.snapshot_state())

    @app.route("/api/views", methods=["GET", "POST"])
    def api_views():
        """左栏四路视图的按需开关。默认只开 head；开某路以省资源换实时性。

        GET  -> {"ok": True, "feeds": {main/head/left_wrist/right_wrist: bool}}
        POST {"feed": "main", "active": true}  或  {"feeds": {"main": true, ...}}
        """
        getter = getattr(server, "get_feed_active", None)
        setter = getattr(server, "set_feed_active", None)
        if not callable(getter) or not callable(setter):
            return jsonify({"ok": False, "error": "view toggle not supported"}), 501
        if request.method == "GET":
            return _json_no_store({"ok": True, "feeds": getter()})
        payload = request.get_json(force=True, silent=True) or {}
        updates = {}
        if isinstance(payload.get("feeds"), dict):
            updates.update(payload["feeds"])
        elif "feed" in payload:
            updates[str(payload.get("feed"))] = payload.get("active", True)
        else:
            return jsonify({"ok": False, "error": "missing feed/feeds"}), 400
        last = {"ok": True, "feeds": getter()}
        for feed, active in updates.items():
            last = setter(str(feed), bool(active))
            if not last.get("ok"):
                return jsonify(last), 400
        return _json_no_store({"ok": True, "feeds": getter()})

    @app.route("/api/tasks")
    def api_tasks():
        from .challenge_tasks import challenge_tasks

        return jsonify({
            "ok": True,
            "challenge_year": 2026,
            "current_id": server.task_id,
            "current": server.task_name,
            "current_scene": server.scene_model,
            "tasks": challenge_tasks(),
        })

    @app.route("/api/task/switch", methods=["POST"])
    def api_task_switch():
        from .challenge_tasks import challenge_task_names, challenge_task_scene

        payload = request.get_json(force=True, silent=True) or {}
        task = str(payload.get("task") or "").strip()
        requested_scene = str(payload.get("scene") or "").strip() or None
        if not task:
            return jsonify({"ok": False, "error": "missing task"}), 400
        valid = challenge_task_names()
        if task not in valid:
            return jsonify({
                "ok": False,
                "error": f"unknown task: {task}",
                "valid_tasks": valid,
            }), 400
        scene = requested_scene or challenge_task_scene(task)
        if not scene:
            return jsonify({"ok": False, "error": f"missing scene for task: {task}"}), 400
        expected_scene = challenge_task_scene(task)
        if expected_scene and scene != expected_scene:
            return jsonify({
                "ok": False,
                "error": f"scene mismatch for {task}: expected {expected_scene}, got {scene}",
                "task": task,
                "scene": scene,
                "expected_scene": expected_scene,
            }), 400
        try:
            if (
                task == server.task_name
                and scene == server.scene_model
                and not getattr(server, "task_switch_request", None)
            ):
                return jsonify({
                    "ok": True,
                    "already_current": True,
                    "task": task,
                    "scene": server.scene_model,
                })
            pending = server.request_task_switch(task, scene_model=scene)
            human_recorder.stop_all(
                outcome="aborted",
                reason="task_switch",
                state=_recording_state_snapshot(),
                memory_text=_recording_memory_text(),
            )
            return jsonify({
                "ok": True,
                "pending": pending,
                "task": task,
                "scene": scene,
            })
        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 409
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/memory")
    def api_memory():
        """返回右侧 Memory 面板文本 + 结构化数据。"""
        budget_reader = getattr(server, 'rollout_budget_snapshot', None)
        return jsonify({
            "summary": server.get_memory_summary(),
            "text": server.get_memory_text(),
            "raw": server.get_memory(),
            'rollout_budget': budget_reader() if callable(budget_reader) else {
                'available': False, 'reason': 'evaluator_budget_not_published'},
        })

    @app.route("/api/skills")
    def api_skills():
        return jsonify({
            "tool_version": active_tool_version(),
            "skills": list_skills(),
        })

    @app.route("/api/skill", methods=["POST"])
    def api_skill():
        payload = request.get_json(force=True, silent=True) or {}
        name = payload.get("name")
        args = payload.get("args") or {}
        if not name:
            return jsonify({"ok": False, "error": "missing 'name'"}), 400
        try:
            req_id = server.submit_skill(name, args)
            return jsonify({"ok": True, "request_id": req_id})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/skill/cancel", methods=["POST"])
    def api_skill_cancel():
        server.cancel_current_skill()
        return jsonify({"ok": True})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        # 只设置标志，真正的 env.reset() 由主 sim 线程下一拍执行（PhysX 必须主线程操作）。
        # 可选 body:
        #   不带 instance_id              -> 普通 reset，回到当前 instance 的初始状态
        #   {"instance_id": <int>}       -> reset 时切换到指定 instance
        #   {"instance_id": "random"} 或 null -> reset 时随机换一个评测 instance
        # 注意：不带 instance_id 时必须用「无参」调用 request_reset()，让它使用 server
        # 模块自身的 _INSTANCE_UNSET 默认值（避免跨模块哨兵对象身份不一致的问题）。
        body = request.get_json(force=True, silent=True) or {}
        try:
            if "instance_id" in body:
                v = body.get("instance_id")
                if v is None or str(v).strip().lower() in ("random", "rand", ""):
                    pending = server.request_reset(None)
                else:
                    try:
                        pending = server.request_reset(int(v))
                    except (TypeError, ValueError):
                        return jsonify({"ok": False, "error": f"instance_id 非法: {v!r}"}), 400
            else:
                pending = server.request_reset()
        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 409
        human_recorder.stop_all(
            outcome="aborted",
            reason="world_reset",
            state=_recording_state_snapshot(),
            memory_text=_recording_memory_text(),
        )
        return jsonify({"ok": True, "pending": pending, "instance_id": server.current_instance_id})

    @app.route("/api/eval_instances")
    def api_eval_instances():
        """右上角 instance 下拉：本口已加载 / 可 reset 的 instance。"""
        official = False
        try:
            official = "official" in str((server.snapshot_state() or {}).get("mode") or "")
        except Exception:
            official = False
        return _json_no_store(eval_control_snapshot(server, official=official))

    @app.route("/api/agent_session")
    def api_agent_session():
        return _json_no_store(eval_control_snapshot(server).get("agent_session") or {})

    @app.route("/api/agent_session/stop", methods=["POST"])
    def api_agent_session_stop():
        """一键停本口 Claude/agent，不拆 interface / evaluator。"""
        port = eval_control_snapshot(server).get("port")
        return jsonify(stop_agent_sessions(port))

    @app.route("/api/scene_graph")
    def api_scene_graph():
        """兼容旧前端：右侧信息面板现在显示 Memory。"""
        return jsonify({
            "summary": server.get_memory_summary(),
            "text": server.get_memory_text(),
            "raw": server.get_memory(),
        })

    @app.route("/api/scene_graph_raw")
    def api_scene_graph_raw():
        """返回完整结构化 Scene Graph（与 /api/scene_graph 的 text 内容等价，
        只是 JSON 形式不带美化缩进，方便程序消费）。"""
        if _privileged_scene_api_disabled():
            return jsonify({
                "ok": False,
                "error": "scene graph is disabled in task_goals_only ablation mode",
            }), 403
        sg = server.get_scene_graph()
        if sg is None:
            return jsonify({"ok": False, "error": "Scene Graph 未构建"}), 503
        from behavior_interface.scene_graph import to_dict
        return jsonify(to_dict(sg))

    @app.route("/api/plan", methods=["POST"])
    def api_plan():
        """POST {goal:[x,y]} 或 {start:[x,y], goal:[x,y]}，返回 A* 规划结果。
        仅规划，不执行。用于 agent 询问"能否到 X" / 调试避障"""
        if _privileged_scene_api_disabled():
            return jsonify({
                "ok": False,
                "error": "world-coordinate planning is disabled in task_goals_only ablation mode",
            }), 403
        body = request.get_json(force=True, silent=True) or {}
        try:
            gx, gy = float(body["goal"][0]), float(body["goal"][1])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "缺少 goal=[x,y]"}), 400
        extra_inflate = float(body.get("extra_inflate", 0.0))

        sg = server.get_scene_graph()
        if sg is None:
            return jsonify({"ok": False, "error": "Scene Graph 未构建"}), 503

        # start 可显式给，缺省用机器人当前位置
        if "start" in body:
            sx, sy = float(body["start"][0]), float(body["start"][1])
        else:
            rp = server._cached_robot_pose
            sx, sy = float(rp["x"]), float(rp["y"])

        from behavior_interface.scene_graph import (
            plan_path, is_point_free, is_segment_free,
        )
        start_ok, start_blk = is_point_free(sg.free_region, sx, sy,
                                            extra_inflate=extra_inflate)
        goal_ok, goal_blk = is_point_free(sg.free_region, gx, gy,
                                          extra_inflate=extra_inflate)
        seg_ok, seg_blk = is_segment_free(sg.free_region, (sx, sy), (gx, gy),
                                          extra_inflate=extra_inflate)
        waypoints, status = plan_path(sg.free_region, (sx, sy), (gx, gy),
                                      extra_inflate=extra_inflate)
        return jsonify({
            "ok": True,
            "start": [sx, sy], "goal": [gx, gy],
            "start_free": start_ok, "start_blocked_by": start_blk,
            "goal_free": goal_ok, "goal_blocked_by": goal_blk,
            "direct_segment_free": seg_ok, "segment_blocked_by": seg_blk,
            "plan_status": status,
            "waypoints": [list(p) for p in waypoints],
        })

    def _nav_pipeline_build_id() -> str:
        try:
            import behavior_interface.skills.move_to_object_v2 as mto
            return str(getattr(mto, "_MOVE_TO_OBJECT_BUILD", "unknown"))
        except Exception:
            return "import_failed"

    def _move_to_point_skill_name() -> str:
        return os.environ.get("MOVE_TO_POINT_SKILL", "move_to_point_v3").strip()

    def _move_to_point_build_id() -> str:
        mod = _move_to_point_skill_name()
        try:
            import importlib
            mtp = importlib.import_module(f"behavior_interface.skills.{mod}")
            return str(getattr(mtp, "_MOVE_TO_POINT_BUILD", mod))
        except Exception:
            return "import_failed"

    def _adjust_plan_pose_build_id() -> str:
        try:
            import behavior_interface.skills.adjust_plan_pose as appose
            return str(getattr(appose, "ADJUST_PLAN_POSE_BUILD", "unknown"))
        except Exception:
            return "import_failed"

    def _spatial_map_build_id() -> str:
        try:
            from behavior_interface.rtabmap_slam.live import (
                BUILD as RTABMAP_BUILD,
                live_backend_selected,
            )

            if live_backend_selected():
                return RTABMAP_BUILD
            from behavior_interface import spatial_map

            return str(getattr(spatial_map, "BUILD", "unknown"))
        except Exception:
            return "import_failed"

    def _spatial_map_backend_id() -> str:
        try:
            from behavior_interface.rtabmap_slam.live import (
                BACKEND as RTABMAP_BACKEND,
                live_backend_selected,
            )

            if live_backend_selected():
                return RTABMAP_BACKEND
            from behavior_interface import spatial_map

            return str(getattr(spatial_map, "BACKEND", "unknown"))
        except Exception:
            return "import_failed"

    def _eef_adjust_build_ids() -> dict[str, str]:
        builds = {
            "adjust_head_frame_build": "import_failed",
            "adjust_wrist_frame_build": "import_failed",
        }
        try:
            import behavior_interface.skills.adjust_eef_pose as head_adjust
            builds["adjust_head_frame_build"] = str(
                getattr(head_adjust, "ADJUST_EEF_POSE_BUILD", "unknown")
            )
        except Exception:
            pass
        try:
            import behavior_interface.skills.adjust_eef_pose_in_wrist_frame as wrist_adjust
            builds["adjust_wrist_frame_build"] = str(
                getattr(
                    wrist_adjust,
                    "ADJUST_EEF_WRIST_FRAME_BUILD",
                    "unknown",
                )
            )
        except Exception:
            pass
        return builds

    @app.route("/api/skills/version", methods=["GET"])
    def api_skills_version():
        """确认当前进程加载的 move_to_object / move_to_point 代码版本。"""
        build = _nav_pipeline_build_id()
        base_mass = getattr(server, "challenge_base_mass", None)
        live_base_mass = None
        try:
            from behavior_interface.challenge_base_mass import read_challenge_base_mass

            live_base_mass = read_challenge_base_mass(getattr(server, "robot", None))
        except Exception:
            pass
        if live_base_mass is not None:
            base_mass = {
                **(base_mass if isinstance(base_mass, dict) else {}),
                "ok": True,
                "after_kg": live_base_mass,
            }
        return jsonify({
            "ok": True,
            "tool_version": active_tool_version(),
            "robot": getattr(server, "robot_name", None),
            "robot_dof": getattr(server, "robot_dof", None),
            "challenge_base_mass_kg": live_base_mass,
            "challenge_base_mass": base_mass,
            "nav_pipeline_build": build,
            "spatial_map_backend": _spatial_map_backend_id(),
            "spatial_map_build": _spatial_map_build_id(),
            "move_to_object_build": build,
            "move_to_point_build": _move_to_point_build_id(),
            "adjust_plan_pose_build": _adjust_plan_pose_build_id(),
            "pinned_actions_v11": bool(
                getattr(
                    getattr(server, "world", None),
                    "_codex_pinned_actions_v11",
                    False,
                )
            ),
            "runtime_action_policy": (
                "controller_action_only"
                if getattr(
                    getattr(server, "world", None),
                    "_codex_pinned_actions_v11",
                    False,
                )
                else "legacy_or_unverified"
            ),
            **_eef_adjust_build_ids(),
        })

    @app.route("/api/skills/reload", methods=["POST"])
    def api_skills_reload():
        """开发时热更新 skill。POST 一下就会 reload behavior_interface.skills.*"""
        import importlib
        body = request.get_json(force=True, silent=True) or {}
        requested_tool_version = str(body.get("tool_version") or "").strip().lower()
        if requested_tool_version:
            from behavior_interface.tool import ENV_TOOL_VERSION, VALID_TOOL_VERSIONS
            if requested_tool_version not in VALID_TOOL_VERSIONS:
                return jsonify({
                    "ok": False,
                    "error": f"unknown tool_version={requested_tool_version}",
                    "valid": list(VALID_TOOL_VERSIONS),
                }), 400
            os.environ[ENV_TOOL_VERSION] = requested_tool_version
        from behavior_interface.skills import reload_all_skills
        loaded = reload_all_skills()
        import behavior_interface.skills as skills_pkg
        import behavior_interface.memory as memory_mod
        importlib.reload(memory_mod)
        memory_reloaded = True
        try:
            import behavior_interface.agent_monitor as monitor_mod
            monitor_mod = importlib.reload(monitor_mod)
            store = app.extensions.get("agent_monitor_store")
            if store is not None:
                store.__class__ = monitor_mod.AgentMonitorStore
        except Exception:
            pass
        tools = reload_v2_tools_metadata()
        return jsonify({
            "ok": True,
            "tool_version": active_tool_version(),
            "modules_reloaded": loaded,
            "memory_reloaded": memory_reloaded,
            "skills_now": sorted(skills_pkg.SKILL_REGISTRY.keys()),
            "v2_tools_reloaded": True,
            "v2_tools_now": [str(t.get("name")) for t in tools],
            "nav_pipeline_build": _nav_pipeline_build_id(),
            "move_to_object_build": _nav_pipeline_build_id(),
            "move_to_point_build": _move_to_point_build_id(),
            **_eef_adjust_build_ids(),
            "move_to_object_args": next(
                (t.get("args") for t in tools if t.get("name") == "move_to_object"), None
            ),
            "move_to_point_args": next(
                (t.get("args") for t in tools if t.get("name") == "move_to_point"), None
            ),
        })

    @app.route("/api/robot_info")
    def api_robot_info():
        """读 robot 的 controller_action_idx + 每个 controller 的 command_dim。
        调试用：写 grasp / move 时需要知道每个 controller slice 维度。
        """
        if server.dry_run or server.robot is None:
            return jsonify({"ok": False, "error": "dry-run or robot not loaded"})
        try:
            out = {}
            for name in server.robot.controller_order:
                idx = server.robot.controller_action_idx[name]
                if hasattr(idx, "detach"):
                    idx = idx.detach().cpu().numpy().tolist()
                else:
                    idx = list(idx)
                ctl = server.robot.controllers[name]
                out[name] = {
                    "indices": idx,
                    "command_dim": int(ctl.command_dim),
                    "class": type(ctl).__name__,
                    "mode": getattr(ctl, "mode", None),
                }
            return jsonify({"ok": True, "action_dim": int(server.robot.action_dim),
                            "controllers": out})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

    @app.route("/api/log")
    def api_log():
        with server.state_lock:
            return jsonify({"log": list(server.log_lines)})

    @app.route("/api/camera", methods=["GET", "POST"])
    def api_camera():
        if request.method == "GET":
            return jsonify(server.get_camera_state())
        payload = request.get_json(force=True, silent=True) or {}
        deltas = payload.get("deltas") or None
        overrides = payload.get("overrides") or None
        reset = bool(payload.get("reset", False))
        params = server.adjust_camera(deltas=deltas, overrides=overrides, reset=reset)
        return jsonify({"ok": True, "params": params})

    # ── v2：agent ↔ 仿真统一接口（人机共用同一服务/env/skill 队列）──────────────
    # 每个端点 = 提交对应 skill → 同步等完成 → 返回结果（含主视图 data URL）。
    # 详见 embodied-agent/BEHAVIOR_AGENT_DESIGN.md §3。
    def _v2_run(
        skill_name: str,
        args: dict,
        timeout_s: float,
        tool: str | None = None,
        allow_ok_false: bool = False,
    ):
        """提交 skill 并同步等待其 result。返回 (result, err_response)。"""
        from behavior_interface.v2_display import skill_display_name

        try:
            req_id = server.submit_skill(skill_name, args)
            result = server.wait_for_skill_result(
                skill_name, timeout_s=timeout_s, request_id=req_id,
            )
            if tool and "tool" not in result:
                result["tool"] = tool
            if "tool" not in result:
                result["tool"] = skill_display_name(skill_name, args)
            if not result.get("ok") and not allow_ok_false:
                return None, (jsonify(_json_safe(result)), 400)
            return _json_safe(result), None
        except TimeoutError as e:
            server.cancel_current_skill()
            return None, (jsonify({"ok": False, "error": str(e)}), 504)
        except Exception as e:
            return None, (jsonify({"ok": False, "error": str(e)}), 400)

    def _v2_capture(
        session_id: str,
        timeout_s: float = 60.0,
        *,
        skill_name: str = "capture",
        extra_args: dict | None = None,
    ):
        args = {"session_id": session_id}
        if extra_args:
            args.update(extra_args)
        return _v2_run(skill_name, args, timeout_s, tool=skill_name)

    def _v2_attach_observation(
        session_id: str,
        result: dict,
        *,
        capture_timeout_s: float = 60.0,
        capture_skill_name: str = "capture",
        capture_extra_args: dict | None = None,
    ) -> dict:
        """skill 退出后拍指定视图，写入 result['observation']。"""
        if not session_id:
            result.setdefault("observation", None)
            return result
        obs, cerr = _v2_capture(
            session_id,
            capture_timeout_s,
            skill_name=capture_skill_name,
            extra_args=capture_extra_args,
        )
        if cerr:
            result["observation"] = None
            try:
                err_body = cerr[0].get_json(silent=True) or {}
                result["capture_error"] = err_body.get("error", "capture 失败")
            except Exception:
                result["capture_error"] = "capture 失败"
        else:
            result["observation"] = obs
            result["exit_capture_skill"] = capture_skill_name
        return _promote_post_action_observation_fields(result)

    def _v2_run_with_exit_capture(
        session_id: str,
        skill_name: str,
        args: dict,
        timeout_s: float,
        tool: str | None = None,
        capture_timeout_s: float = 60.0,
        exit_capture_skill_name: str = "capture",
        exit_capture_extra_args: dict | None = None,
        exit_capture_by_result_arm: bool = False,
    ):
        """运行 skill；无论成败（含超时）在退出时 capture 并附 observation。"""
        force_exit_capture = active_tool_version() in ("v1", "v1_shortcut", "v2", "v3")

        def capture_skill_for(result_payload: dict) -> str:
            if not exit_capture_by_result_arm:
                return exit_capture_skill_name
            nested = result_payload.get("result")
            nested_arm = nested.get("arm") if isinstance(nested, dict) else None
            arm = str(
                result_payload.get("arm")
                or nested_arm
                or (args or {}).get("arm")
                or ""
            ).strip().lower()
            if arm in ("left", "right"):
                return f"capture_{arm}_wrist_camera"
            return exit_capture_skill_name

        result, err = _v2_run(
            skill_name, args, timeout_s, tool=tool, allow_ok_false=True,
        )
        if result is not None:
            if skill_name != "capture":
                plan_input_image = str((args or {}).get("image_id") or "").strip()
                if plan_input_image:
                    result.setdefault("input_image_id", plan_input_image)
                skip_exit_capture = (
                    not force_exit_capture
                    and
                    skill_name == "plan_eef_v2"
                    and result.get("ok") is False
                    and bool(plan_input_image)
                )
                if skip_exit_capture:
                    result.setdefault("image_id", plan_input_image)
                    result.setdefault("input_image_id", plan_input_image)
                    result.setdefault("observation", None)
                    result["exit_capture_skipped"] = True
                else:
                    _v2_attach_observation(
                        session_id,
                        result,
                        capture_timeout_s=capture_timeout_s,
                        capture_skill_name=capture_skill_for(result),
                        capture_extra_args=exit_capture_extra_args,
                    )
            _maybe_attach_spatial_map(session_id, tool or skill_name, args, result)
            return result, None
        payload: dict = {"ok": False, "error": "skill 无结果", "skill": skill_name}
        if err:
            try:
                payload = err[0].get_json(silent=True) or payload
            except Exception:
                pass
            if tool and "tool" not in payload:
                payload["tool"] = tool
            plan_input_image = str((args or {}).get("image_id") or "").strip()
            if plan_input_image:
                payload.setdefault("input_image_id", plan_input_image)
            skip_exit_capture = (
                not force_exit_capture
                and
                skill_name == "plan_eef_v2"
                and bool(plan_input_image)
            )
            if skip_exit_capture:
                payload.setdefault("image_id", plan_input_image)
                payload.setdefault("input_image_id", plan_input_image)
                payload.setdefault("observation", None)
                payload["exit_capture_skipped"] = True
            elif session_id and skill_name != "capture":
                _v2_attach_observation(
                    session_id,
                    payload,
                    capture_timeout_s=capture_timeout_s,
                    capture_skill_name=capture_skill_for(payload),
                    capture_extra_args=exit_capture_extra_args,
                )
            _maybe_attach_spatial_map(session_id, tool or skill_name, args, payload)
            return None, (jsonify(payload), err[1] if err and len(err) > 1 else 500)
        return None, (jsonify(payload), 500)

    def _v2_run_with_required_v1_capture(
        session_id: str,
        skill_name: str,
        args: dict,
        timeout_s: float,
        tool: str | None = None,
        allow_ok_false: bool = False,
        capture_timeout_s: float = 60.0,
    ):
        """For compact public tool profiles, always attach a post-run head observation."""
        if active_tool_version() in ("v1", "v1_shortcut", "v2", "v3") and skill_name != "capture":
            return _v2_run_with_exit_capture(
                session_id,
                skill_name,
                args,
                timeout_s,
                tool=tool,
                capture_timeout_s=capture_timeout_s,
            )
        return _v2_run(
            skill_name,
            args,
            timeout_s,
            tool=tool,
            allow_ok_false=allow_ok_false,
        )

    def _parse_plan_move_eef_args(body: dict, session_id: str) -> dict:
        args = {
            "session_id": session_id,
            "upward": _m_to_legacy_cm(body.get("upward", 0.0)),
            "forward": _m_to_legacy_cm(body.get("forward", 0.0)),
            "leftward": _m_to_legacy_cm(body.get("leftward", 0.0)),
            "gripper": str(body.get("gripper", "keep")),
            "pos_tol": float(body.get("pos_tol", 0.012)),
        }
        for k in ("u", "v", "depth"):
            if body.get(k) is not None and str(body.get(k)).strip() != "":
                args[k] = float(body[k])
        if body.get("arm") is not None:
            args["arm"] = str(body["arm"])
        return args

    def _parse_plan_move_eef_to_point_args(body: dict, session_id: str) -> dict:
        args = {
            "session_id": session_id,
            "image_id": str(body.get("image_id", "")).strip(),
            "u": int(body.get("u", -1)),
            "v": int(body.get("v", -1)),
            "upward": _m_to_legacy_cm(body.get("upward", 0.05)),
            "gripper": str(body.get("gripper", "keep")),
            "pos_tol": float(body.get("pos_tol", 0.012)),
        }
        if body.get("arm") is not None:
            args["arm"] = str(body["arm"])
        return args

    def _parse_adjust_plan_pose_args(body: dict, session_id: str) -> dict:
        args = {
            "session_id": session_id,
            "plan_id": str(body.get("plan_id", "")).strip(),
            "forward": _m_to_legacy_cm(body.get("forward", 0.0)),
            "upward": _m_to_legacy_cm(body.get("upward", 0.0)),
            "leftward": _m_to_legacy_cm(body.get("leftward", 0.0)),
            "roll": float(body.get("roll", 0.0)),
            "pitch": float(body.get("pitch", 0.0)),
            "yaw": float(body.get("yaw", 0.0)),
        }
        if isinstance(body.get("move"), dict):
            args["move"] = body["move"]
        return args

    def _api_v2_adjust_eef_in_head_frame(body: dict, side: str, tool_name: str):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        timeout_s = float(body.get("timeout_s", 90.0))
        capture_timeout_s = float(
            body.get("capture_timeout_s", max(120.0, min(timeout_s, 180.0)))
        )
        args = {
            "forward": float(body.get("forward", 0.0)),
            "upward": float(body.get("upward", 0.0)),
            "leftward": float(body.get("leftward", 0.0)),
            "roll": float(body.get("roll", 0.0)),
            "pitch": float(body.get("pitch", 0.0)),
            "yaw": float(body.get("yaw", 0.0)),
            "pos_tol": float(body.get("pos_tol", 0.012)),
            "ori_tol_deg": float(body.get("ori_tol_deg", 3.0)),
        }
        result, err = _v2_run_with_exit_capture(
            session_id,
            tool_name,
            args,
            timeout_s,
            tool=tool_name,
            capture_timeout_s=capture_timeout_s,
        )
        if result is not None:
            result["tool"] = tool_name
            result.setdefault("arm", side)
        return err[0] if err else jsonify(result)

    def _api_v2_adjust_eef_in_wrist_frame(body: dict, side: str, tool_name: str):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        timeout_s = float(body.get("timeout_s", 90.0))
        capture_timeout_s = float(
            body.get("capture_timeout_s", max(120.0, min(timeout_s, 180.0)))
        )
        args = {
            "forward": float(body.get("forward", 0.0)),
            "leftward": float(body.get("leftward", 0.0)),
            "upward": float(body.get("upward", 0.0)),
            "roll": float(body.get("roll", 0.0)),
            "pitch": float(body.get("pitch", 0.0)),
            "yaw": float(body.get("yaw", 0.0)),
            "pos_tol": float(body.get("pos_tol", 0.012)),
            "ori_tol_deg": float(body.get("ori_tol_deg", 3.0)),
        }
        result, err = _v2_run_with_exit_capture(
            session_id,
            tool_name,
            args,
            timeout_s,
            tool=tool_name,
            capture_timeout_s=capture_timeout_s,
            exit_capture_skill_name=f"capture_{side}_wrist_camera",
        )
        if result is not None:
            result["tool"] = tool_name
        return err[0] if err else jsonify(result)

    def _lazy_register_v2_routes() -> None:
        """热更新 web.py 后无需重启仿真，也能挂上新增的 v2 路由。"""

        def _parse_arm_open_gripper(body: dict) -> tuple:
            arm = (body.get("arm") or "right").strip()
            og = body.get("open_gripper", True)
            if isinstance(og, str):
                og = og.lower() in ("true", "1", "yes")
            return arm, bool(og)

        def _parse_gripper_mode(body: dict) -> str:
            g = body.get("gripper", None)
            if g is None:
                og = body.get("open_gripper", None)
                if og is None:
                    g = "keep"
                else:
                    if isinstance(og, str):
                        og = og.lower() in ("true", "1", "yes")
                    g = "open" if bool(og) else "keep"
            g = str(g).strip().lower()
            return g if g in ("open", "keep") else "keep"

        for side in ("left", "right"):
            tool_name = f"adjust_{side}_eef_pose_in_wrist_frame"
            endpoint_name = f"api_v2_{tool_name}"
            if endpoint_name in app.view_functions:
                continue

            def api_v2_adjust_eef_in_wrist_frame_lazy(side=side, tool_name=tool_name):
                body = request.get_json(force=True, silent=True) or {}
                return _api_v2_adjust_eef_in_wrist_frame(body, side, tool_name)

            app.add_url_rule(
                f"/api/v2/{tool_name}",
                endpoint_name,
                api_v2_adjust_eef_in_wrist_frame_lazy,
                methods=["POST"],
            )

        if "api_v2_arm_reset" not in app.view_functions:
            def api_v2_arm_reset_lazy():
                body = request.get_json(force=True, silent=True) or {}
                arm, og = _parse_arm_open_gripper(body)
                mode = (body.get("mode") or "grasp").strip()
                args = {
                    "arm": arm,
                    "mode": mode,
                    "open_gripper": og,
                    "timeout_s": float(body.get("timeout_s", 60.0)),
                }
                session_id = (body.get("session_id") or "web").strip()
                t_skill = float(body.get("timeout_s", 60.0))
                result, err = _v2_run_with_exit_capture(
                    session_id, "arm_reset", args,
                    max(t_skill + 10.0, 70.0), tool="arm_reset",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule("/api/v2/arm_reset", "api_v2_arm_reset", api_v2_arm_reset_lazy, methods=["POST"])

        if "api_v2_set_arm_to_grasp_position" not in app.view_functions:
            def api_v2_set_arm_to_grasp_position_lazy():
                body = request.get_json(force=True, silent=True) or {}
                arm = (body.get("arm") or "right").strip()
                gripper = _parse_gripper_mode(body)
                keep_ori_arm = str(body.get("keep_ori_arm", "none"))
                keep_ori_enabled = keep_ori_arm.strip().lower() not in {
                    "", "none", "false", "0", "no",
                }
                requested_timeout = float(body.get("timeout_s", 15.0))
                t_skill = (
                    min(150.0, max(60.0, requested_timeout))
                    if keep_ori_enabled
                    else min(15.0, max(6.0, requested_timeout))
                )
                args = {
                    "arm": arm,
                    "keep_ori_arm": keep_ori_arm,
                    "gripper": gripper,
                    "open_gripper": gripper == "open",
                    "timeout_s": t_skill,
                    "max_dq_per_step": float(body.get("max_dq_per_step", 0.30)),
                    "tol": float(body.get("tol", 0.08)),
                }
                session_id = (body.get("session_id") or "web").strip()
                result, err = _v2_run_with_exit_capture(
                    session_id, "set_arm_to_grasp_position", args,
                    max(t_skill + 30.0, 90.0),
                    tool="set_arm_to_grasp_position",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule(
                "/api/v2/set_arm_to_grasp_position",
                "api_v2_set_arm_to_grasp_position",
                api_v2_set_arm_to_grasp_position_lazy,
                methods=["POST"],
            )

        if "api_v2_set_arm_to_grasp_position_shortcut" not in app.view_functions:
            def api_v2_set_arm_to_grasp_position_shortcut_lazy():
                body = request.get_json(force=True, silent=True) or {}
                arm, og = _parse_arm_open_gripper(body)
                t_skill = float(body.get("timeout_s", 35.0))
                args = {
                    "arm": arm,
                    "open_gripper": og,
                    "timeout_s": t_skill,
                    "max_dq_per_step": float(body.get("max_dq_per_step", 1.2)),
                    "tol": float(body.get("tol", 0.08)),
                }
                session_id = (body.get("session_id") or "web").strip()
                result, err = _v2_run_with_exit_capture(
                    session_id, "set_arm_to_grasp_position_shortcut", args,
                    max(t_skill + 30.0, 90.0),
                    tool="set_arm_to_grasp_position_shortcut",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule(
                "/api/v2/set_arm_to_grasp_position_shortcut",
                "api_v2_set_arm_to_grasp_position_shortcut",
                api_v2_set_arm_to_grasp_position_shortcut_lazy,
                methods=["POST"],
            )

        if "api_v2_plan_move_eef" not in app.view_functions:
            def api_v2_plan_move_eef_lazy():
                body = request.get_json(force=True, silent=True) or {}
                session_id = (body.get("session_id") or "").strip()
                if not session_id:
                    return jsonify({"ok": False, "error": "缺少 session_id"}), 400
                result, err = _v2_run_with_required_v1_capture(
                    session_id,
                    "plan_move_eef",
                    _parse_plan_move_eef_args(body, session_id),
                    float(body.get("timeout_s", 60.0)),
                    tool="plan_move_eef",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule(
                "/api/v2/plan_move_eef",
                "api_v2_plan_move_eef",
                api_v2_plan_move_eef_lazy,
                methods=["POST"],
            )

        if "api_v2_plan_move_eef_to_point" not in app.view_functions:
            def api_v2_plan_move_eef_to_point_lazy():
                body = request.get_json(force=True, silent=True) or {}
                session_id = (body.get("session_id") or "").strip()
                image_id = (body.get("image_id") or "").strip()
                if not session_id or not image_id:
                    return jsonify({"ok": False, "error": "缺少 session_id / image_id"}), 400
                if body.get("u") is None or body.get("v") is None:
                    return jsonify({"ok": False, "error": "u, v 必填"}), 400
                result, err = _v2_run_with_required_v1_capture(
                    session_id,
                    "plan_move_eef_to_point",
                    _parse_plan_move_eef_to_point_args(body, session_id),
                    float(body.get("timeout_s", 90.0)),
                    tool="plan_move_eef_to_point",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule(
                "/api/v2/plan_move_eef_to_point",
                "api_v2_plan_move_eef_to_point",
                api_v2_plan_move_eef_to_point_lazy,
                methods=["POST"],
            )

        if "api_v2_adjust_plan_pose" not in app.view_functions:
            def api_v2_adjust_plan_pose_lazy():
                body = request.get_json(force=True, silent=True) or {}
                session_id = (body.get("session_id") or "").strip()
                if not session_id:
                    return jsonify({"ok": False, "error": "缺少 session_id"}), 400
                if not (body.get("plan_id") or isinstance(body.get("move"), dict)):
                    return jsonify({"ok": False, "error": "需要 plan_id 或 move"}), 400
                result, err = _v2_run_with_required_v1_capture(
                    session_id,
                    "adjust_plan_pose",
                    _parse_adjust_plan_pose_args(body, session_id),
                    float(body.get("timeout_s", 90.0)),
                    tool="adjust_plan_pose",
                )
                return err[0] if err else jsonify(result)

            app.add_url_rule(
                "/api/v2/adjust_plan_pose",
                "api_v2_adjust_plan_pose",
                api_v2_adjust_plan_pose_lazy,
                methods=["POST"],
            )

        if "api_v2_mesure_shoulder_distance" not in app.view_functions:
            def api_v2_mesure_shoulder_distance_lazy():
                body = request.get_json(force=True, silent=True) or {}
                session_id = (body.get("session_id") or "").strip()
                object_name = (body.get("object_name") or "").strip()
                image_id = (body.get("image_id") or "").strip()
                has_uv = body.get("u") is not None and body.get("v") is not None
                if not session_id:
                    return jsonify({"ok": False, "error": "需要 session_id"}), 400
                if not object_name and not (image_id and has_uv):
                    return jsonify({
                        "ok": False,
                        "error": "需要 object_name，或 image_id + u + v",
                    }), 400
                args = {"session_id": session_id}
                if object_name:
                    args["object_name"] = object_name
                else:
                    args["image_id"] = image_id
                    args["u"] = int(body["u"])
                    args["v"] = int(body["v"])
                result, err = _v2_run_with_required_v1_capture(
                    session_id,
                    "mesure_shoulder_distance",
                    args,
                    float(body.get("timeout_s", 45.0)),
                    tool="measure_shoulder_distance",
                    allow_ok_false=True,
                )
                return err[0] if err else jsonify(result)

            for route, endpoint in (
                ("/api/v2/measure_shoulder_distance", "api_v2_measure_shoulder_distance"),
                ("/api/v2/mesure_shoulder_distance", "api_v2_mesure_shoulder_distance"),
            ):
                app.add_url_rule(
                    route,
                    endpoint,
                    api_v2_mesure_shoulder_distance_lazy,
                    methods=["POST"],
                )

    @app.route("/api/v2/capture", methods=["POST"])
    def api_v2_capture():
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        result, err = _v2_capture(session_id, float(body.get("timeout_s", 60.0)))
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/capture_head_camera", methods=["POST"])
    def api_v2_capture_head_camera():
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        result, err = _v2_capture(session_id, float(body.get("timeout_s", 60.0)))
        if result is not None:
            result["tool"] = "capture_head_camera"
        return err[0] if err else jsonify(result)

    def _api_v2_wrist_capture(body: dict, side: str):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        n_settle = int(body.get("n_settle", 2))
        skill_name = f"capture_{side}_wrist_camera"
        result, err = _v2_run(
            skill_name,
            {"session_id": session_id, "n_settle": n_settle},
            float(body.get("timeout_s", 60.0)),
            tool=skill_name,
            allow_ok_false=True,
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/capture_left_wrist_camera", methods=["POST"])
    def api_v2_capture_left_wrist_camera():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_wrist_capture(body, "left")

    @app.route("/api/v2/capture_right_wrist_camera", methods=["POST"])
    def api_v2_capture_right_wrist_camera():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_wrist_capture(body, "right")

    @app.route("/api/v2/read_depth", methods=["POST"])
    def api_v2_read_depth():
        body = request.get_json(force=True, silent=True) or {}
        session_id = str(body.get("session_id") or "").strip()
        image_id = str(body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not image_id or not has_uv:
            return jsonify({
                "ok": False,
                "error": "需要 image_id + u + v（先 capture camera 再点选）",
            }), 400
        allowed_id_chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789_.-"
        )
        if (
            len(session_id) > 128
            or len(image_id) > 128
            or not session_id[0].isalnum()
            or not image_id[0].isalnum()
            or any(char not in allowed_id_chars for char in session_id)
            or any(char not in allowed_id_chars for char in image_id)
        ):
            return jsonify({
                "ok": False,
                "error": "session_id 或 image_id 含不支持的字符",
            }), 400
        result, err = _v2_run(
            "read_depth",
            {
                "session_id": session_id,
                "image_id": image_id,
                "u": body["u"],
                "v": body["v"],
            },
            float(body.get("timeout_s", 10.0)),
            tool="read_depth",
            allow_ok_false=True,
        )
        return err[0] if err else jsonify(result)

    def _api_v2_move_world(body: dict):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        args = {k: float(body.get(k, 0.0))
                for k in ("dx", "dy", "dz", "dthetax", "dthetaz")}
        dz = args["dz"]
        if abs(dz) > 0.35:
            return jsonify({
                "ok": False,
                "error": (
                    f"dz={dz} 的单位是米（胸口高度变化），有效约 ±0.3m。"
                    "若要做俯仰请填 dthetaz（度）；若要走远请填 dx/dy。"
                ),
            }), 400
        if abs(args["dthetaz"]) > 120.0:
            return jsonify({
                "ok": False,
                "error": (
                    f"dthetaz={args['dthetaz']}° 过大；"
                    "胸口 θz 允许约 [60,165]°，单次增量请控制在该区间内"
                ),
            }), 400
        result, err = _v2_run_with_exit_capture(
            session_id, "move_in_world_coord", args,
            float(body.get("timeout_s", 150.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/move_in_world_coord", methods=["POST"])
    def api_v2_move_in_world_coord():
        """世界系 5D 增量；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_move_world(body)

    @app.route("/api/v2/move", methods=["POST"])
    def api_v2_move():
        """兼容旧路径，等同 move_in_world_coord。"""
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_move_world(body)

    @app.route("/api/v2/move_in_robot_coord", methods=["POST"])
    def api_v2_move_in_robot_coord():
        """机体系 forward/spin/pitch；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        args = {k: float(body.get(k, 0.0))
                for k in ("forward", "spin", "pitch", "upward")}
        # 不再用 |pitch|≤45 硬卡；真正约束在 skill 内：结果 θz∈[60,165]，只动 q3
        if abs(args["pitch"]) > 120.0:
            return jsonify({
                "ok": False,
                "error": (
                    f"pitch={args['pitch']}° 过大；"
                    "胸口 θz 允许 [60,165]°，单次 pitch 请控制在该可达区间内"
                ),
            }), 400
        if abs(args["upward"]) > _ROBOT_UPWARD_MAX_M:
            return jsonify({
                "ok": False,
                "error": (
                    f"upward={args['upward']}m 超出 API 上限 |upward|≤{_ROBOT_UPWARD_MAX_M}m"
                    "（两阶段垂直升降；直立姿态约可降 0.55m，具体由 skill 校验）"
                ),
            }), 400
        result, err = _v2_run_with_exit_capture(
            session_id, "move_in_robot_coord", args,
            float(body.get("timeout_s", 150.0)),
        )
        return err[0] if err else jsonify(result)

    def _api_v2_move_in_robot_coord_named(
        body: dict,
        args: dict,
        tool_name: str,
        timeout_default: float = 150.0,
        *,
        observation_only_chassis: bool = False,
    ):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        result, err = _v2_run_with_exit_capture(
            session_id,
            "move_in_robot_coord",
            {
                "forward": float(args.get("forward", 0.0)),
                "translation": float(args.get("translation", 0.0)),
                "spin": float(args.get("spin", 0.0)),
                "pitch": float(args.get("pitch", 0.0)),
                "upward": float(args.get("upward", 0.0)),
                "observation_only_chassis": bool(
                    observation_only_chassis
                ),
            },
            float(body.get("timeout_s", timeout_default)),
            tool=tool_name,
        )
        if result is not None:
            result["tool"] = tool_name
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/adjust_chassis", methods=["POST"])
    def api_v2_adjust_chassis():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_move_in_robot_coord_named(
            body,
            {
                "forward": float(body.get("forward", 0.0)),
                "translation": float(
                    body.get("translation", body.get("leftward", 0.0))
                ),
                "spin": float(body.get("spin", 0.0)),
            },
            "adjust_chassis",
        )

    @app.route("/api/v2/adjust_pitch", methods=["POST"])
    def api_v2_adjust_pitch():
        body = request.get_json(force=True, silent=True) or {}
        degree = float(body.get("degree", body.get("pitch", 0.0)))
        # 只动 q3；可达性由 skill 按结果 θz∈[60,165] 校验，不再硬卡 |degree|≤45
        if abs(degree) > 120.0:
            return jsonify({
                "ok": False,
                "error": (
                    f"degree={degree}° 过大；"
                    "胸口 θz 允许 [60,165]°，单次 degree 请控制在该可达区间内"
                ),
            }), 400
        return _api_v2_move_in_robot_coord_named(
            body,
            {"pitch": degree},
            "adjust_pitch",
            timeout_default=180.0,
        )

    @app.route("/api/v2/adjust_hight", methods=["POST"])
    @app.route("/api/v2/adjust_height", methods=["POST"])
    def api_v2_adjust_hight():
        """调整身体高度；保留 adjust_hight 路由兼容旧客户端。"""
        body = request.get_json(force=True, silent=True) or {}
        upward = float(body.get("upward", body.get("upwardm", body.get("upward_m", 0.0))))
        if abs(upward) > _ROBOT_UPWARD_MAX_M:
            return jsonify({
                "ok": False,
                "error": f"upward={upward}m 超出 API 上限 |upward|≤{_ROBOT_UPWARD_MAX_M}m",
            }), 400
        return _api_v2_move_in_robot_coord_named(
            body,
            {"upward": upward},
            "adjust_height",
            timeout_default=120.0,
        )

    @app.route("/api/v2/face_to_point", methods=["POST"])
    def api_v2_face_to_point():
        """只旋转底盘，把公开 0..1000 head 图像点水平居中。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        try:
            u = float(body["u"])
            v = float(body["v"])
        except Exception:
            return jsonify({"ok": False, "error": "需要 0..1000 的 u/v 坐标"}), 400
        max_abs_spin = float(body.get("max_abs_spin", 60.0))
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": u,
            "v": v,
            "max_abs_spin": max_abs_spin,
        }
        result, err = _v2_run_with_exit_capture(
            session_id,
            "face_to_point",
            args,
            float(body.get("timeout_s", 180.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/spin_to_facing_point", methods=["POST"])
    def api_v2_spin_to_facing_point():
        """v2 名字：只旋转底盘，把 head 图像点水平居中。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        try:
            u = float(body["u"])
            v = float(body["v"])
        except Exception:
            return jsonify({"ok": False, "error": "需要 0..1000 的 u/v 坐标"}), 400
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": u,
            "v": v,
            "max_abs_spin": float(body.get("max_abs_spin", 60.0)),
        }
        result, err = _v2_run_with_exit_capture(
            session_id,
            "face_to_point",
            args,
            float(body.get("timeout_s", 180.0)),
            tool="spin_to_facing_point",
        )
        if result is not None:
            result["tool"] = "spin_to_facing_point"
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/move_eef", methods=["POST"])
    def api_v2_move_eef():
        """head 相机系 EEF 米制增量平移；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        args = {
            "upward": _m_to_legacy_cm(body.get("upward", 0.0)),
            "forward": _m_to_legacy_cm(body.get("forward", 0.0)),
            "leftward": _m_to_legacy_cm(body.get("leftward", 0.0)),
            "gripper": str(body.get("gripper", "keep")),
            "pos_tol": float(body.get("pos_tol", 0.012)),
        }
        for k in ("u", "v", "depth"):
            if body.get(k) is not None and str(body.get(k)).strip() != "":
                args[k] = float(body[k])
        if body.get("arm") is not None:
            args["arm"] = str(body["arm"])
        result, err = _v2_run_with_exit_capture(
            session_id, "move_eef", args, float(body.get("timeout_s", 90.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/move_point_to_point", methods=["POST"])
    def api_v2_move_point_to_point():
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        points = body.get("points")
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not image_id:
            return jsonify({"ok": False, "error": "缺少 image_id"}), 400
        if not isinstance(points, list) or len(points) != 2:
            return jsonify({
                "ok": False,
                "error": "points 必须恰好包含两个点：[待移动点, 目标点]",
            }), 400
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "points": points,
            "above_target_point_m": float(
                body.get("above_target_point_m", 0.0)
            ),
            "pos_tol": float(body.get("pos_tol", 0.012)),
            "ori_tol_deg": float(body.get("ori_tol_deg", 5.0)),
            "max_steps": int(body.get("max_steps", 360)),
            "timeout_s": float(body.get("timeout_s", 90.0)),
        }
        result, err = _v2_run_with_exit_capture(
            session_id,
            "move_point_to_point",
            args,
            max(float(body.get("timeout_s", 90.0)) + 30.0, 120.0),
            tool="move_point_to_point",
        )
        return err[0] if err else jsonify(result)

    def _api_v2_gripper(body: dict, gripper: str, tool_name: str):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        arm = str(body.get("arm", "right")).strip().lower()
        if arm not in ("left", "right"):
            return jsonify({"ok": False, "error": "arm 必须是 left/right"}), 400
        args = {
            "upward": 0.0,
            "forward": 0.0,
            "leftward": 0.0,
            "gripper": gripper,
            "arm": arm,
            "pos_tol": float(body.get("pos_tol", 0.012)),
        }
        wait_timeout_s = (
            float(body["timeout_s"])
            if body.get("timeout_s") is not None
            else (
                _default_close_gripper_wait_timeout(server)
                if gripper == "close"
                else 90.0
            )
        )
        result, err = _v2_run_with_exit_capture(
            session_id,
            "move_eef",
            args,
            wait_timeout_s,
            tool=tool_name,
            exit_capture_skill_name=(
                f"capture_{arm}_wrist_camera" if gripper == "close" else "capture"
            ),
        )
        if result is not None:
            result["tool"] = tool_name
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/open_gripper", methods=["POST"])
    def api_v2_open_gripper():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_gripper(body, "open", "open_gripper")

    @app.route("/api/v2/close_gripper", methods=["POST"])
    def api_v2_close_gripper():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_gripper(body, "close", "close_gripper")

    @app.route("/api/v2/adjust_left_eef_pose_in_head_frame", methods=["POST"])
    def api_v2_adjust_left_eef_pose_in_head_frame():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_adjust_eef_in_head_frame(
            body,
            "left",
            "adjust_left_eef_pose_in_head_frame",
        )

    @app.route("/api/v2/adjust_right_eef_pose_in_head_frame", methods=["POST"])
    def api_v2_adjust_right_eef_pose_in_head_frame():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_adjust_eef_in_head_frame(
            body,
            "right",
            "adjust_right_eef_pose_in_head_frame",
        )

    @app.route("/api/v2/adjust_left_eef_pose_in_wrist_frame", methods=["POST"])
    def api_v2_adjust_left_eef_pose_in_wrist_frame():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_adjust_eef_in_wrist_frame(
            body,
            "left",
            "adjust_left_eef_pose_in_wrist_frame",
        )

    @app.route("/api/v2/adjust_right_eef_pose_in_wrist_frame", methods=["POST"])
    def api_v2_adjust_right_eef_pose_in_wrist_frame():
        body = request.get_json(force=True, silent=True) or {}
        return _api_v2_adjust_eef_in_wrist_frame(
            body,
            "right",
            "adjust_right_eef_pose_in_wrist_frame",
        )

    @app.route("/api/v2/plan_move_eef", methods=["POST"])
    def api_v2_plan_move_eef():
        """仅预览 move_eef 的目标 EEF 红爪叠影；不执行动作。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "plan_move_eef", _parse_plan_move_eef_args(body, session_id),
            float(body.get("timeout_s", 60.0)),
            tool="plan_move_eef",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/plan_eef_translation_to_uvd_point", methods=["POST"])
    def api_v2_plan_eef_translation_to_uvd_point():
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        missing = [k for k in ("u", "v", "depth") if body.get(k) is None or str(body.get(k)).strip() == ""]
        if missing:
            return jsonify({"ok": False, "error": f"缺少 {', '.join(missing)}"}), 400
        args = {
            "session_id": session_id,
            "upward": 0.0,
            "forward": 0.0,
            "leftward": 0.0,
            "u": float(body["u"]),
            "v": float(body["v"]),
            "depth": float(body["depth"]),
            "gripper": "keep",
            "pos_tol": float(body.get("pos_tol", 0.012)),
        }
        if body.get("arm") is not None:
            args["arm"] = str(body["arm"])
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "plan_move_eef",
            args,
            float(body.get("timeout_s", 60.0)),
            tool="plan_eef_translation_to_uvd_point",
        )
        if result is not None:
            result["tool"] = "plan_eef_translation_to_uvd_point"
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/plan_move_eef_to_point", methods=["POST"])
    def api_v2_plan_move_eef_to_point():
        """点选 head 图反解 3D，预览 EEF 到该点世界Z正上方；不执行动作。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        if not session_id or not image_id:
            return jsonify({"ok": False, "error": "缺少 session_id / image_id"}), 400
        if body.get("u") is None or body.get("v") is None:
            return jsonify({"ok": False, "error": "u, v 必填"}), 400
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "plan_move_eef_to_point",
            _parse_plan_move_eef_to_point_args(body, session_id),
            float(body.get("timeout_s", 90.0)),
            tool="plan_move_eef_to_point",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/adjust_plan_pose", methods=["POST"])
    def api_v2_adjust_plan_pose():
        """调整已有 move/eef_pose，返回新的 plan_id 和 head 红爪预览；不执行动作。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not (body.get("plan_id") or isinstance(body.get("move"), dict)):
            return jsonify({"ok": False, "error": "需要 plan_id 或 move"}), 400
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "adjust_plan_pose",
            _parse_adjust_plan_pose_args(body, session_id),
            float(body.get("timeout_s", 90.0)),
            tool="adjust_plan_pose",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/rotate_eef", methods=["POST"])
    def api_v2_rotate_eef():
        """head 相机系 EEF 增量旋转（位置不变）；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        args = {
            "rotate_forward_deg": float(body.get("rotate_forward_deg", 0.0)),
            "rotate_upward_deg": float(body.get("rotate_upward_deg", 0.0)),
            "rotate_leftward_deg": float(body.get("rotate_leftward_deg", 0.0)),
        }
        if body.get("arm") is not None:
            args["arm"] = str(body["arm"])
        result, err = _v2_run_with_exit_capture(
            session_id, "rotate_eef", args, float(body.get("timeout_s", 90.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/move_to", methods=["POST"])
    def api_v2_move_to():
        """move_to(x,y,z,thetax,thetaz)：世界系绝对 5D；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if body.get("x") is None or body.get("y") is None:
            return jsonify({"ok": False, "error": "x, y 必填"}), 400
        args = {"x": float(body["x"]), "y": float(body["y"])}
        if body.get("z") is not None:
            args["z"] = float(body["z"])
        if body.get("thetax") is not None:
            args["theta_x_deg"] = float(body["thetax"])
        if body.get("thetaz") is not None:
            args["theta_z_deg"] = float(body["thetaz"])
        result, err = _v2_run_with_exit_capture(
            session_id, "move_to", args, float(body.get("timeout_s", 150.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/plan", methods=["POST"])
    def api_v2_plan():
        """plan_*(image_id,u,v,mode[,object_name])：返回 plan_id/eef_pose/next_move/标注图(data url)。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        mode = (body.get("mode") or "").strip()
        obj_name = (body.get("object_name") or "").strip()
        batch_points = body.get("points")
        is_rgbd_batch = bool(
            mode == "grasp_point_filter_rgbd"
            and isinstance(batch_points, list)
            and batch_points
        )
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id or not image_id or not mode:
            return jsonify({"ok": False, "error": "需要 session_id / image_id / mode"}), 400
        if mode in ("grasp_obj", "grasp_obj_filter"):
            if not obj_name and not has_uv:
                return jsonify({
                    "ok": False,
                    "error": f"{mode} 需要 object_name，或 image_id + u + v",
                }), 400
        elif not has_uv and not is_rgbd_batch:
            return jsonify({"ok": False, "error": "u, v 必填"}), 400
        args = {
            "session_id": session_id, "image_id": image_id, "mode": mode,
            "arm": (body.get("arm") or "right"),
        }
        if is_rgbd_batch:
            args["points"] = batch_points
            args.pop("mode", None)
        elif has_uv and not obj_name:
            args["u"] = int(body["u"])
            args["v"] = int(body["v"])
        if body.get("seed") is not None and mode in (
            "grasp_obj", "grasp_obj_filter", "grasp_point_filter",
            "grasp_point_filter_rgbd", "grasp_point_filter_rgbd_lite",
            "press_point",
        ):
            args["seed"] = int(body["seed"])
        if not is_rgbd_batch and mode in (
            "grasp_obj_filter",
            "grasp_point_filter",
            "grasp_point_filter_rgbd",
            "grasp_point_filter_rgbd_lite",
            "press_point",
        ):
            plan_arm = str(body.get("plan_arm", "any")).strip().lower()
            args["plan_arm"] = plan_arm if plan_arm in ("left", "right", "any") else "any"
        if obj_name:
            args["object_name"] = obj_name
        from behavior_interface.v2_display import mode_to_tool
        tool = mode_to_tool(mode)
        plan_default_timeout = 1800.0 if mode in (
            "grasp_obj", "grasp_obj_filter", "grasp_point_filter",
            "grasp_point_filter_rgbd", "grasp_point_filter_rgbd_lite",
            "press_point",
        ) else 200.0
        if is_rgbd_batch:
            plan_default_timeout *= max(1, len(batch_points))
        plan_skill = (
            "plan_eef_rgbd_batch"
            if is_rgbd_batch
            else (
                "plan_eef_rgbd_lite"
                if mode == "grasp_point_filter_rgbd_lite"
                else "plan_eef_v2"
            )
        )
        result, err = _v2_run_with_exit_capture(
            session_id,
            plan_skill,
            args,
            float(body.get("timeout_s", plan_default_timeout)), tool=tool,
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/arm_reset", methods=["POST"])
    def api_v2_arm_reset():
        """arm_reset：手臂复位，不动底盘/躯干。"""
        body = request.get_json(force=True, silent=True) or {}
        arm = (body.get("arm") or "right").strip()
        mode = (body.get("mode") or "grasp").strip()
        og = body.get("open_gripper", True)
        if isinstance(og, str):
            og = og.lower() in ("true", "1", "yes")
        t_skill = float(body.get("timeout_s", 60.0))
        args = {
            "arm": arm,
            "mode": mode,
            "open_gripper": bool(og),
            "timeout_s": t_skill,
        }
        session_id = (body.get("session_id") or "web").strip()
        result, err = _v2_run_with_exit_capture(
            session_id, "arm_reset", args,
            max(t_skill + 10.0, 70.0), tool="arm_reset",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/set_arm_to_grasp_position", methods=["POST"])
    def api_v2_set_arm_to_grasp_position():
        """set_arm_to_grasp_position：EEF 直线 IK 同分支过滤到 grasp prep。"""
        body = request.get_json(force=True, silent=True) or {}
        arm = (body.get("arm") or "right").strip()
        gripper = body.get("gripper", None)
        if gripper is None:
            og = body.get("open_gripper", None)
            if og is None:
                gripper = "keep"
            else:
                if isinstance(og, str):
                    og = og.lower() in ("true", "1", "yes")
                gripper = "open" if bool(og) else "keep"
        gripper = str(gripper).strip().lower()
        if gripper not in ("open", "keep"):
            gripper = "keep"
        keep_ori_arm = str(body.get("keep_ori_arm", "none"))
        keep_ori_enabled = keep_ori_arm.strip().lower() not in {
            "", "none", "false", "0", "no",
        }
        requested_timeout = float(body.get("timeout_s", 15.0))
        t_skill = (
            min(150.0, max(60.0, requested_timeout))
            if keep_ori_enabled
            else min(15.0, max(6.0, requested_timeout))
        )
        args = {
            "arm": arm,
            "keep_ori_arm": keep_ori_arm,
            "gripper": gripper,
            "open_gripper": gripper == "open",
            "timeout_s": t_skill,
            "max_dq_per_step": float(body.get("max_dq_per_step", 0.30)),
            "tol": float(body.get("tol", 0.08)),
        }
        session_id = (body.get("session_id") or "web").strip()
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "set_arm_to_grasp_position", args,
            max(t_skill + 5.0, 20.0),
            tool="set_arm_to_grasp_position",
            allow_ok_false=True,
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/set_arm_to_grasp_position_shortcut", methods=["POST"])
    def api_v2_set_arm_to_grasp_position_shortcut():
        """set_arm_to_grasp_position_shortcut：旧版直接设定并锁定 grasp prep。"""
        body = request.get_json(force=True, silent=True) or {}
        arm = (body.get("arm") or "right").strip()
        og = body.get("open_gripper", True)
        if isinstance(og, str):
            og = og.lower() in ("true", "1", "yes")
        t_skill = float(body.get("timeout_s", 35.0))
        args = {
            "arm": arm,
            "open_gripper": bool(og),
            "timeout_s": t_skill,
            "max_dq_per_step": float(body.get("max_dq_per_step", 1.2)),
            "tol": float(body.get("tol", 0.08)),
        }
        session_id = (body.get("session_id") or "web").strip()
        result, err = _v2_run_with_exit_capture(
            session_id, "set_arm_to_grasp_position_shortcut", args,
            max(t_skill + 30.0, 90.0),
            tool="set_arm_to_grasp_position_shortcut",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/reset_body", methods=["POST"])
    def api_v2_reset_body():
        """reset_body：腰部直立复位；手臂/夹爪锁进入 skill 时的当前关节角。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "web").strip()
        args = {
            "timeout_s": float(body.get("timeout_s", 45.0)),
            "trunk_max_step": float(body.get("trunk_max_step", 0.06)),
            "shoulder_iters": int(body.get("shoulder_iters", 4)),
            "keep_ori_arm": str(body.get("keep_ori_arm", "none")),
            "pitch_deg": float(body.get("pitch_deg", 0.0)),
        }
        if isinstance(body.get("keep_ori_target_quat"), dict):
            args["keep_ori_target_quat"] = body["keep_ori_target_quat"]
        result, err = _v2_run_with_exit_capture(
            session_id, "reset_body", args,
            max(float(body.get("timeout_s", 60.0)) + 15.0, 75.0), tool="reset_body",
        )
        if err:
            return err
        response = jsonify(result)
        if isinstance(result, dict) and result.get("timed_out"):
            return response, 504
        return response

    def _parse_exec_plan_body(body: dict) -> tuple[dict, tuple | None]:
        """exec_move / exec_eef_pose 共用入参解析。"""
        session_id = (body.get("session_id") or "").strip()
        plan_id = (body.get("plan_id") or "").strip()
        if not session_id or not plan_id:
            return {}, (jsonify({"ok": False, "error": "需要 session_id / plan_id"}), 400)
        exec_args = {"session_id": session_id, "plan_id": plan_id}
        arm_sel = (body.get("arm") or "").strip().lower()
        if arm_sel in ("left", "right"):
            exec_args["arm"] = arm_sel
        try:
            raw_back_m = body.get("back_m")
            if raw_back_m is not None and raw_back_m != "":
                back_m_val = float(raw_back_m)
                if back_m_val == 0.0 or 0.02 <= back_m_val <= 0.50:
                    exec_args["back_m"] = back_m_val
        except (TypeError, ValueError):
            pass
        return exec_args, None

    @app.route("/api/v2/exec_move", methods=["POST"])
    def api_v2_exec_move():
        """exec_move(plan_id)：执行已规划动作；完成后自动 capture 主视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        exec_args, perr = _parse_exec_plan_body(body)
        if perr:
            return perr[0]
        exec_res, err = _v2_run_with_exit_capture(
            session_id, "exec_move_v2", exec_args,
            float(body.get("timeout_s", 300.0)),
        )
        return err[0] if err else jsonify(exec_res)

    @app.route("/api/v2/exec_eef_pose", methods=["POST"])
    def api_v2_exec_eef_pose():
        """exec_eef_pose(plan_id)：当前→safe→grasp，不合爪、无避障。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        exec_args, perr = _parse_exec_plan_body(body)
        if perr:
            return perr[0]
        if "stop_after_safe" in body:
            sas = body.get("stop_after_safe")
            exec_args["stop_after_safe"] = (
                str(sas).strip().lower() in ("1", "true", "yes", "on")
                if isinstance(sas, str)
                else bool(sas)
            )
        exec_res, err = _v2_run_with_exit_capture(
            session_id, "exec_eef_pose_v2", exec_args,
            float(body.get("timeout_s", 300.0)),
        )
        return err[0] if err else jsonify(exec_res)

    @app.route("/api/v2/exec_plan_pose", methods=["POST"])
    def api_v2_exec_plan_pose():
        """v2 名字：执行 plan_id 对应的 EEF pose，不合爪；返回实际执行臂 wrist 视图。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        exec_args, perr = _parse_exec_plan_body(body)
        if perr:
            return perr[0]
        if "stop_after_safe" in body:
            sas = body.get("stop_after_safe")
            exec_args["stop_after_safe"] = (
                str(sas).strip().lower() in ("1", "true", "yes", "on")
                if isinstance(sas, str)
                else bool(sas)
            )
        exec_args["reset_tool_roll_at_start"] = True
        exec_res, err = _v2_run_with_exit_capture(
            session_id,
            "exec_eef_pose_v2",
            exec_args,
            float(body.get("timeout_s", 300.0)),
            tool="exec_plan_pose",
            exit_capture_by_result_arm=True,
        )
        if exec_res is not None:
            exec_res["tool"] = "exec_plan_pose"
        return err[0] if err else jsonify(exec_res)

    @app.route("/api/v2/manipulate_add_vector_to_point", methods=["POST"])
    def api_v2_manipulate_add_vector_to_point():
        """在所选 0..1000 相对坐标点反投影并创建局部坐标系向量。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        args: dict = {"session_id": session_id}
        if image_id:
            args["image_id"] = image_id
        pts = body.get("points")
        if isinstance(pts, str) and pts.strip():
            try:
                pts = json.loads(pts)
            except Exception:
                pts = None
        if isinstance(pts, list) and pts:
            args["points"] = pts
        elif body.get("u") is not None and body.get("v") is not None:
            try:
                args["u"] = float(body["u"])
                args["v"] = float(body["v"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "u, v 需为数值"}), 400
        else:
            return jsonify({"ok": False, "error": "需要 points 或 (u, v)"}), 400
        try:
            args["length_m"] = float(body.get("length_m", 0.1))
        except (TypeError, ValueError):
            args["length_m"] = 0.1
        result, err = _v2_run_with_exit_capture(
            session_id, "manipulate_add_vector_to_point", args,
            float(body.get("timeout_s", 60.0)),
            tool="manipulate_add_vector_to_point",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/manipulate_move_vector_to_vector", methods=["POST"])
    def api_v2_manipulate_move_vector_to_vector():
        """把 from_vector 移动到 to_vector（需其表面点物体被夹爪抓住），纯手臂到位。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        from_vector = (body.get("from_vector") or "").strip()
        to_vector = (body.get("to_vector") or "").strip()
        if not session_id or not from_vector or not to_vector:
            return jsonify({"ok": False, "error": "缺少 session_id / from_vector / to_vector"}), 400
        args: dict = {
            "session_id": session_id,
            "from_vector": from_vector,
            "to_vector": to_vector,
        }
        try:
            args["back_m"] = float(body.get("back_m", 0.0))
        except (TypeError, ValueError):
            args["back_m"] = 0.0
        result, err = _v2_run_with_exit_capture(
            session_id, "manipulate_move_vector_to_vector", args,
            float(body.get("timeout_s", 300.0)),
            tool="manipulate_move_vector_to_vector",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/mark_object", methods=["POST"])
    def api_v2_mark_object():
        """mark_object(image_id,u,v)：点→物体，写 session memory.md，返回 bddl name。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        if not session_id or not image_id:
            return jsonify({"ok": False, "error": "需要 session_id / image_id"}), 400
        object_name = (body.get("object_name") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not object_name and not has_uv:
            return jsonify({"ok": False, "error": "需要 object_name 或 u + v"}), 400
        args = {"session_id": session_id, "image_id": image_id}
        if object_name:
            args["object_name"] = object_name
        if has_uv and not object_name:
            args["u"] = int(body["u"])
            args["v"] = int(body["v"])
        result, err = _v2_run_with_exit_capture(
            session_id, "mark_object_v2", args, float(body.get("timeout_s", 60.0)),
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/move_to_object", methods=["POST"])
    def api_v2_move_to_object():
        """move_to_object：与 move_to_point 同款移动（球心=物体 AABB 中心）。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        object_name = (body.get("object_name") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not object_name and not (image_id and has_uv):
            return jsonify({
                "ok": False,
                "error": "需要 object_name，或 image_id + u + v",
            }), 400
        args = {"session_id": session_id, "object_name": object_name, "image_id": image_id}
        if has_uv and not object_name:
            args["u"] = int(body["u"])
            args["v"] = int(body["v"])
        if body.get("standoff") is not None:
            args["standoff"] = float(body["standoff"])
        if body.get("reach") is not None:
            args["reach"] = float(body["reach"])
        nav_timeout_s = float(body.get("nav_timeout_s", 120.0))
        args["nav_timeout_s"] = nav_timeout_s
        queue_timeout = float(body.get("timeout_s", max(320.0, nav_timeout_s + 120.0)))
        mv_res, err = _v2_run_with_exit_capture(
            session_id, "move_to_object_v2", args, queue_timeout,
        )
        return err[0] if err else jsonify(mv_res)

    @app.route("/api/v2/move_to_point", methods=["POST"])
    def api_v2_move_to_point():
        """move_to_point：与 move_to_object 同款移动（球心=uv 反解 3D 点）。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not image_id or not has_uv:
            return jsonify({
                "ok": False,
                "error": "需要 image_id + u + v（先 capture 再点选）",
            }), 400
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": int(body["u"]),
            "v": int(body["v"]),
        }
        if body.get("reach") is not None:
            args["reach"] = float(body["reach"])
        nav_timeout_s = float(body.get("nav_timeout_s", 120.0))
        args["nav_timeout_s"] = nav_timeout_s
        queue_timeout = float(body.get("timeout_s", max(320.0, nav_timeout_s + 120.0)))
        mv_res, err = _v2_run_with_exit_capture(
            session_id, _move_to_point_skill_name(), args, queue_timeout,
        )
        return err[0] if err else jsonify(mv_res)

    @app.route("/api/v2/move_to_reach_point", methods=["POST"])
    def api_v2_move_to_reach_point():
        """v2 名字：move_to_point → move_to_reach_point。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not image_id or not has_uv:
            return jsonify({
                "ok": False,
                "error": "需要 image_id + u + v（先 capture_head_camera 再点选）",
            }), 400
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": int(body["u"]),
            "v": int(body["v"]),
        }
        if body.get("reach") is not None:
            args["reach"] = float(body["reach"])
        nav_timeout_s = float(body.get("nav_timeout_s", 120.0))
        args["nav_timeout_s"] = nav_timeout_s
        args["keep_ori_arm"] = str(body.get("keep_ori_arm", "none"))
        queue_timeout = float(body.get("timeout_s", max(320.0, nav_timeout_s + 120.0)))
        mv_res, err = _v2_run_with_exit_capture(
            session_id,
            _move_to_point_skill_name(),
            args,
            queue_timeout,
            tool="move_to_reach_point",
        )
        if mv_res is not None:
            mv_res["tool"] = "move_to_reach_point"
        return err[0] if err else jsonify(mv_res)

    @app.route("/api/v2/move_base_to_point", methods=["POST"])
    def api_v2_move_base_to_point():
        """move_base_to_point：点选地面点，底盘保持 yaw 并直接做 XY 平移。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not image_id or not has_uv:
            return jsonify({
                "ok": False,
                "error": "需要 image_id + u + v（先 capture 再点选地面）",
            }), 400
        nav_timeout_s = float(body.get("nav_timeout_s", 120.0))
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": int(body["u"]),
            "v": int(body["v"]),
            "nav_timeout_s": nav_timeout_s,
            "ground_tol_m": float(body.get("ground_tol_m", 0.04)),
            "pos_tol_m": float(body.get("pos_tol_m", 0.12)),
        }
        queue_timeout = float(body.get("timeout_s", max(220.0, nav_timeout_s + 100.0)))
        mv_res, err = _v2_run_with_required_v1_capture(
            session_id,
            "move_base_to_point",
            args,
            queue_timeout,
            tool="move_base_to_point",
            allow_ok_false=True,
        )
        return err[0] if err else jsonify(mv_res)

    @app.route("/api/v2/move_chassis_to_floor_point", methods=["POST"])
    def api_v2_move_chassis_to_floor_point():
        """v2 名字：move_base_to_point → move_chassis_to_floor_point。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not image_id or not has_uv:
            return jsonify({
                "ok": False,
                "error": "需要 image_id + u + v（先 capture_head_camera 再点选地面）",
            }), 400
        nav_timeout_s = float(body.get("nav_timeout_s", 120.0))
        args = {
            "session_id": session_id,
            "image_id": image_id,
            "u": int(body["u"]),
            "v": int(body["v"]),
            "nav_timeout_s": nav_timeout_s,
            "ground_tol_m": float(body.get("ground_tol_m", 0.04)),
            "pos_tol_m": float(body.get("pos_tol_m", 0.12)),
        }
        queue_timeout = float(body.get("timeout_s", max(220.0, nav_timeout_s + 100.0)))
        mv_res, err = _v2_run_with_required_v1_capture(
            session_id,
            "move_base_to_point",
            args,
            queue_timeout,
            tool="move_chassis_to_floor_point",
            allow_ok_false=True,
        )
        if mv_res is not None:
            mv_res["tool"] = "move_chassis_to_floor_point"
        return err[0] if err else jsonify(mv_res)

    @app.route("/api/v2/navigate_to", methods=["POST"])
    def api_v2_navigate_to():
        """Plan and execute a high-clearance route to a marked map place."""
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "error": "navigate_to 请求必须是 JSON object",
            }), 400
        from behavior_interface_eval_test.tool.official_v2.contract import (
            NAVIGATE_TO_MAX_TIMEOUT_S,
            validate_navigate_to_args,
        )

        try:
            args = validate_navigate_to_args(body)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        session_id = str(args["session_id"])
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        nav_timeout_s = float(args["timeout_s"])
        result, err = _v2_run_with_exit_capture(
            session_id,
            "navigate_to",
            args,
            # The controller may extend a short request after it knows the
            # route length.  Leave the server wait above the public ceiling so
            # that dynamic navigation timeouts are not cancelled by Flask.
            max(240.0, nav_timeout_s + 60.0, NAVIGATE_TO_MAX_TIMEOUT_S + 60.0),
            tool="navigate_to",
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/query_map", methods=["POST"])
    def api_v2_query_map():
        """测试口查询当前选中的合规 SLAM 后端 + 场景俯视图。"""
        from behavior_interface.spatial_map import (
            build_query_result,
            spatial_map_enabled,
        )

        if not spatial_map_enabled():
            return jsonify({
                "ok": False,
                "error": "query_map 仅测试口开启（BEHAVIOR_SPATIAL_MAP=1）",
            }), 404
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        heading_up = not bool(body.get("fixed_north_up"))
        from behavior_interface.rtabmap_slam.live import (
            BUILD as RTABMAP_BUILD,
            get_live_mapper,
            live_backend_selected,
        )

        if live_backend_selected():
            mapper = get_live_mapper()
            if mapper is None:
                return jsonify({"ok": False, "error": "RTAB-Map 后端未初始化"}), 503
            query = mapper.query(session_id)
            result = {
                "ok": True,
                "tool": "query_map",
                "build": RTABMAP_BUILD,
                "spatial_map": query,
                "hint": query.get("hint"),
                "image_id": "minimap_current",
            }
            try:
                png, _version = mapper.map_snapshot_png(
                    heading_up=heading_up, size=640
                )
                result["rgb_main"] = (
                    "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                )
                from behavior_interface import agent_runs

                agent_runs.ensure_session(session_id)
                path = agent_runs.image_path(
                    session_id, "minimap_current", ".rtabmap.png"
                )
                with open(path, "wb") as handle:
                    handle.write(png)
                result["rgb_main_path"] = path
            except Exception:
                pass
            return jsonify(_json_safe(result))
        return jsonify(_json_safe(build_query_result(
            session_id, heading_up=heading_up
        )))

    @app.route("/api/v2/mark_position", methods=["POST"])
    @app.route("/api/v2/mark_on_map", methods=["POST"])
    def api_v2_mark_on_map():
        """在地图上标记一个具名地点：点选 head 图上的物体，或标记脚下。"""
        from behavior_interface.spatial_map import (
            build_mark_on_map_result,
            spatial_map_enabled,
        )

        if not spatial_map_enabled():
            return jsonify({
                "ok": False,
                "error": "mark_on_map 仅测试口开启（BEHAVIOR_SPATIAL_MAP=1）",
            }), 404
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        name = str(body.get("name") or body.get("label") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "缺少 name"}), 400
        from behavior_interface.rtabmap_slam.live import (
            get_live_mapper,
            live_backend_selected,
        )

        if live_backend_selected():
            mapper = get_live_mapper()
            if mapper is None:
                result = {"ok": False, "error": "RTAB-Map 后端未初始化"}
            else:
                evaluator_sequence = None
                try:
                    from behavior_interface.continuous_capture import peek_sequence

                    evaluator_sequence = peek_sequence(getattr(server, "world", None))
                except Exception:
                    pass
                result = mapper.mark_on_map(
                    session_id,
                    name,
                    image_id=str(body.get("image_id") or "").strip(),
                    u=body.get("u"),
                    v=body.get("v"),
                    evaluator_sequence=evaluator_sequence,
                )
        else:
            result = build_mark_on_map_result(
                session_id,
                name,
                image_id=str(body.get("image_id") or "").strip(),
                u=body.get("u"),
                v=body.get("v"),
            )
        return jsonify(_json_safe(result)), (200 if result.get("ok") else 400)

    @app.route("/api/v2/mesure_shoulder_distance", methods=["POST"])
    @app.route("/api/v2/measure_shoulder_distance", methods=["POST"])
    def api_v2_mesure_shoulder_distance():
        """测量肩部到目标点的距离；保留旧拼写路由兼容旧客户端。"""
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        object_name = (body.get("object_name") or "").strip()
        image_id = (body.get("image_id") or "").strip()
        has_uv = body.get("u") is not None and body.get("v") is not None
        if not session_id:
            return jsonify({"ok": False, "error": "需要 session_id"}), 400
        if not object_name and not (image_id and has_uv):
            return jsonify({
                "ok": False,
                "error": "需要 object_name，或 image_id + u + v",
            }), 400
        args = {"session_id": session_id}
        if object_name:
            args["object_name"] = object_name
        else:
            args["image_id"] = image_id
            args["u"] = int(body["u"])
            args["v"] = int(body["v"])
        result, err = _v2_run_with_required_v1_capture(
            session_id,
            "mesure_shoulder_distance",
            args,
            float(body.get("timeout_s", 45.0)),
            tool="measure_shoulder_distance",
            allow_ok_false=True,
        )
        return err[0] if err else jsonify(result)

    @app.route("/api/v2/tools", methods=["GET"])
    def api_v2_tools():
        """agent ↔ 网页共用的 v2 工具元数据（下拉菜单 + 动态 args 表单）。
        session_id 由后端隐式注入，不在此暴露。"""
        _lazy_register_v2_routes()
        tools = reload_v2_tools_metadata()
        return jsonify({
            "tool_version": active_tool_version(),
            "tools": tools,
        })

    # ── plan_grasp 交互 API ────────────────────────────────────────────────────
    @app.route("/api/plan_grasp/views", methods=["GET"])
    def api_plan_grasp_views():
        from behavior_interface.skills.plan_grasp_core import PLAN_GRASP_VIEWS

        labels = {
            "left_upper_front": "左上前",
            "left_upper_back": "左上后",
            "right_upper_front": "右上前",
            "right_upper_back": "右上后",
        }
        return jsonify({
            "ok": True,
            "views": [{"id": v, "label": labels.get(v, v)} for v in PLAN_GRASP_VIEWS],
        })

    @app.route("/api/plan_grasp/capture", methods=["POST"])
    def api_plan_grasp_capture():
        """Step1: object_name + view → 照片 URL + session_id。"""
        body = request.get_json(force=True, silent=True) or {}
        object_name = (body.get("object_name") or body.get("object") or "").strip()
        if not object_name:
            return jsonify({"ok": False, "error": "缺少 object_name"}), 400
        args = {
            "step": "capture",
            "object_name": object_name,
            "arm": body.get("arm", "right"),
            "view": body.get("view", "left_upper_front"),
        }
        if body.get("cam_dist") is not None:
            args["cam_dist"] = float(body["cam_dist"])
        sync = bool(body.get("sync", True))
        try:
            req_id = server.submit_skill("plan_grasp", args)
            if not sync:
                return jsonify({"ok": True, "request_id": req_id, "async": True})
            result = server.wait_for_skill_result("plan_grasp", timeout_s=180.0)
            if not result.get("ok"):
                return jsonify(result), 400
            return jsonify(result)
        except TimeoutError as e:
            return jsonify({"ok": False, "error": str(e)}), 504
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    def _plan_grasp_submit(body, step: str, default_timeout: float = 180.0):
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return None, (jsonify({"ok": False, "error": "缺少 session_id"}), 400)
        args = {
            "step": step,
            "session_id": session_id,
            "point_source": body.get("point_source", "user"),
        }
        if body.get("u") is not None:
            args["u"] = float(body["u"])
        if body.get("v") is not None:
            args["v"] = float(body["v"])
        sync = bool(body.get("sync", True))
        timeout = 240.0 if args["point_source"] == "vlm" else default_timeout
        try:
            req_id = server.submit_skill("plan_grasp", args)
            if not sync:
                return {"ok": True, "request_id": req_id, "async": True}, None
            result = server.wait_for_skill_result("plan_grasp", timeout_s=timeout)
            if not result.get("ok"):
                return None, (jsonify(result), 400)
            return result, None
        except TimeoutError as e:
            return None, (jsonify({"ok": False, "error": str(e)}), 504)
        except Exception as e:
            return None, (jsonify({"ok": False, "error": str(e)}), 400)

    @app.route("/api/plan_grasp/resolve_3d", methods=["POST"])
    def api_plan_grasp_resolve_3d():
        """Step2a: 2D 点 → 3D 击中点 + 仿真红球。"""
        body = request.get_json(force=True, silent=True) or {}
        result, err = _plan_grasp_submit(body, "resolve_3d")
        if err:
            return err
        return jsonify(result)

    @app.route("/api/plan_grasp/resolve_grasp", methods=["POST"])
    def api_plan_grasp_resolve_grasp():
        """Step2b: 由上次 3D 点 → EEF pose + 仿真夹爪。"""
        body = request.get_json(force=True, silent=True) or {}
        result, err = _plan_grasp_submit(body, "resolve_grasp", default_timeout=120.0)
        if err:
            return err
        return jsonify(result)

    @app.route("/api/plan_grasp/resolve", methods=["POST"])
    def api_plan_grasp_resolve():
        """兼容：一步完成 3D + grasp。"""
        body = request.get_json(force=True, silent=True) or {}
        result, err = _plan_grasp_submit(body, "resolve")
        if err:
            return err
        return jsonify(result)

    @app.route("/api/plan_grasp/session/<session_id>/image")
    def api_plan_grasp_image(session_id: str):
        from behavior_interface.skills.plan_grasp_core import load_session

        try:
            sess = load_session(session_id)
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "session 不存在"}), 404
        path = sess.get("image_path")
        if not path or not os.path.isfile(path):
            return jsonify({"ok": False, "error": "image 不存在"}), 404
        return send_file(path, mimetype="image/png")

    @app.route("/api/plan_grasp/session/<session_id>/marked")
    def api_plan_grasp_marked(session_id: str):
        from behavior_interface.skills.plan_grasp_core import session_dir

        path = os.path.join(session_dir(session_id), "point_on_image.png")
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": "marked 图不存在，请先 resolve_3d"}), 404
        return send_file(path, mimetype="image/png")

    @app.route("/api/plan_grasp/session/<session_id>/viz_3d")
    def api_plan_grasp_viz_3d(session_id: str):
        from behavior_interface.skills.plan_grasp_core import session_dir

        path = os.path.join(session_dir(session_id), "viz_3d.png")
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": "viz_3d 不存在，请先 resolve_3d"}), 404
        return send_file(path, mimetype="image/png")

    @app.route("/api/plan_grasp/session/<session_id>/viz_grasp")
    def api_plan_grasp_viz_grasp(session_id: str):
        from behavior_interface.skills.plan_grasp_core import session_dir

        path = os.path.join(session_dir(session_id), "viz_grasp.png")
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": "viz_grasp 不存在，请先 resolve_grasp"}), 404
        return send_file(path, mimetype="image/png")

    return app


def serve_bounded_http(app, host: str, port: int) -> None:
    """Serve ``app`` with a fixed worker pool.

    Werkzeug's ``threaded=True`` starts one thread per accepted socket and
    never reaps it.  A client that times out leaves that thread blocked and
    the socket in CLOSE_WAIT; the next poll adds another.  That is how one
    interface grew past 900 threads and 50 GB.  Past the pool size the
    acceptor answers 503 and closes the socket itself, so a slow handler
    cannot create another thread.  A request whose client has already gone
    returns before the view runs.
    """
    import select
    import socket as socket_module
    from concurrent.futures import ThreadPoolExecutor

    from werkzeug.serving import ThreadedWSGIServer

    try:
        pool_size = int(os.environ.get("BEHAVIOR_INTERFACE_HTTP_THREADS", "24"))
    except (TypeError, ValueError):
        pool_size = 24
    pool_size = max(4, min(pool_size, 32))

    @app.before_request
    def _stop_if_client_gone():
        sock = request.environ.get("werkzeug.socket")
        if sock is None:
            return None
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return None
            peeked = sock.recv(1, socket_module.MSG_PEEK | socket_module.MSG_DONTWAIT)
        except BlockingIOError:
            return None
        except OSError:
            return ("", 499)
        if peeked == b"":
            return ("", 499)
        return None

    class _BoundedServer(ThreadedWSGIServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._slots = threading.BoundedSemaphore(pool_size)
            self._pool = ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="iface-http",
            )

        def process_request(self, request, client_address):  # noqa: A002
            # Blocked forever is how a timed-out client kept the thread.
            # Compute does not touch the socket, so this only fires when a
            # read or write actually stalls.  A client that has gone away
            # fails the write and the worker returns to the pool.
            try:
                request.settimeout(
                    float(os.environ.get("BEHAVIOR_INTERFACE_HTTP_SOCKET_TIMEOUT_S", "20"))
                )
            except (OSError, TypeError, ValueError):
                pass
            if not self._slots.acquire(blocking=False):
                try:
                    request.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\n"
                        b"Content-Length: 0\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                except OSError:
                    pass
                self.shutdown_request(request)
                return

            def _run() -> None:
                try:
                    self.process_request_thread(request, client_address)
                finally:
                    self._slots.release()

            try:
                self._pool.submit(_run)
            except RuntimeError:
                self._slots.release()
                self.shutdown_request(request)

        def server_close(self) -> None:
            self._pool.shutdown(wait=False)
            super().server_close()

    httpd = _BoundedServer(host, int(port), app)
    httpd.serve_forever()


def run_web_in_thread(server, host: str, port: int) -> threading.Thread:
    """开一个守护线程跑 Flask。线程数有上限，客户端断开后不再继续处理。"""
    setattr(server, "web_host", host)
    setattr(server, "web_port", port)
    app = build_app(server)

    def _run():
        serve_bounded_http(app, host, port)

    t = threading.Thread(target=_run, name="flask-web", daemon=True)
    t.start()
    return t
