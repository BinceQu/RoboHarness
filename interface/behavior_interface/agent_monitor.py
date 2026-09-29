"""Embodied Codex 关键帧监视：一次工具调用一张卡片。

卡片规则：
- 每次外部 agent 的 /api/v2 调用生成一张卡片；
- 工具名和参数在上，图片在下；
- 只有返回图：下方放输出图；
- 既有输入图（args.image_id）又有返回图：输入图（含选点红十字）和输出图并排；
- 输入图与输出图是同一张时，只显示一张（有选点则叠红十字）。

直播规则：
- 同一 BEHAVIOR_SESSION_ID（含 Codex resume / compact）接着往旧卡片后面写，不切盘；
- 场景重置会结束当前 attempt：tips 从新 session 从 0 再计，旧卡片留在下拉历史里；
- 只有换了 session_id 或发生场景重置才另开直播；
- 没有新 session 时，刷新或重启 interface 都恢复当前正在看的那一轮，不能空掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any

import cv2


SCHEMA_VERSION = "behavior.agent_monitor.keyframes.v1"
_SKIP_SESSION_PREFIXES = ("web-", "human-")
BASELINE_SKILL = "behavior-v2-baseline"
_SKIP_ARG_KEYS = {"session_id", "timeout_s"}
_IMAGE_PATH_KEYS = ("rgb_path", "rgb_main_path", "rgb_overlay_path")
_MAX_ARG_CHARS = 120
_MAX_ARGS = 10
_THUMB_MAX_W = 320
_THUMB_JPEG_QUALITY = 70
_COORDINATE_MAX = 1000.0
_RED_BGR = (0, 0, 255)
_CROSS_OUTLINE_BGR = (0, 0, 0)
# 缩略图十字只要标点，不能盖住目标物体。
_CROSS_ARM_MIN_PX = 4
_CROSS_ARM_DIV = 40
_CROSS_THICKNESS_PX = 1


def default_monitor_root() -> str:
    """监视记录根目录。默认放本机 /tmp。

    仓库在 NFS 上。九路同时 session_begin 时，把 session.json fsync 到
    那块盘会超过控制器的 5 秒超时，会话时钟建不起来，Claude 也就不会开工。
    BEHAVIOR_AGENT_MONITOR_ROOT 仍可把目录指回别处。
    """
    env = os.environ.get("BEHAVIOR_AGENT_MONITOR_ROOT", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    user = os.environ.get("USER") or "bince"
    port = os.environ.get("BEHAVIOR_EVAL_TEST_PORT") or os.environ.get("PORT") or "shared"
    return f"/tmp/behavior_agent_monitor_{user}_p{port}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _as_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _attempt_stamp(value: str) -> str:
    """把 started_at 收成目录后缀，避免同一 session_id 两轮写进同一个文件夹。"""
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) >= 17:
        return f"{digits[:8]}T{digits[8:17]}Z"
    if len(digits) >= 14:
        return f"{digits[:8]}T{digits[8:14]}Z"
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_attempt_id(session_id: str, started_at: str, run_dir: str = "") -> str:
    """一轮 attempt 的稳定 id；同一 session_id 再开一轮也必须能区分。"""
    sid = str(session_id or "").strip()
    stamp = _attempt_stamp(started_at)
    if sid and stamp:
        return f"{sid}-{stamp}"
    base = os.path.basename(str(run_dir or "").rstrip("/"))
    return base or sid or stamp


MONITOR_ATTEMPT_SUFFIX_RE = re.compile(r"-\d{8}T\d{6,9}Z(?:-\d+)?$")


def monitor_token_aliases(token: str) -> list[str]:
    """页面可能传 attempt_id，目录名仍是 session_id。两种都认。"""
    text = str(token or "").strip()
    if not text:
        return []
    aliases = [text]
    stripped = MONITOR_ATTEMPT_SUFFIX_RE.sub("", text)
    if stripped and stripped != text:
        aliases.append(stripped)
    return aliases


def convention_card_thumb_names(card_id: int, role: str) -> list[str]:
    stem = f"card_{int(card_id):04d}_{'in' if role == 'input' else 'out'}"
    return [f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"]


def is_external_agent_session(session_id: str) -> bool:
    """浏览器 Human Involve / 默认 web session 不进监视器。"""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    return not sid.startswith(_SKIP_SESSION_PREFIXES)


_HARNESS_TASK_SID_RE = re.compile(r"(?:^|[^0-9a-z])t(\d{2})p\d{4,5}", re.IGNORECASE)
_TASK_BUCKET_RE = re.compile(r"^t(\d{2})_(.+)$")
_PORT_BUCKET_RE = re.compile(r"^p(\d{4,5})$")
_TASK_MATCHERS: list[tuple[str, int, str]] | None = None


def normalize_task_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


def task_bucket_name(task_id: int, task_name: str) -> str:
    name = str(task_name or "").strip() or "unknown"
    return f"t{int(task_id):02d}_{name}"


def parse_task_bucket(name: str) -> tuple[int | None, str | None]:
    match = _TASK_BUCKET_RE.fullmatch(str(name or "").strip())
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def _task_matchers() -> list[tuple[str, int, str]]:
    """长名字优先，避免 picking_up 误伤 picking_up_trash / picking_up_toys。"""
    global _TASK_MATCHERS
    if _TASK_MATCHERS is not None:
        return _TASK_MATCHERS
    from behavior_interface.challenge_tasks import BEHAVIOR_CHALLENGE_2026_TASKS

    items: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for task in BEHAVIOR_CHALLENGE_2026_TASKS:
        task_id = int(task["id"])
        task_name = str(task["name"])
        display = str(task.get("display_name") or "")
        for raw in (task_name, display):
            key = normalize_task_key(raw)
            if len(key) < 6 or key in seen:
                continue
            seen.add(key)
            items.append((key, task_id, task_name))
    items.sort(key=lambda item: len(item[0]), reverse=True)
    _TASK_MATCHERS = items
    return items


def infer_monitor_task(
    payload: dict[str, Any] | None = None,
    *,
    session_id: str = "",
    prompt: str = "",
    dir_name: str = "",
    parent_name: str = "",
) -> tuple[int | None, str | None]:
    """从 session.json / 目录名 / session_id / prompt 推断挑战任务。"""
    from behavior_interface.challenge_tasks import challenge_task_id, challenge_task_name

    data = payload if isinstance(payload, dict) else {}
    raw_id = data.get("task_id")
    raw_name = str(data.get("task_name") or data.get("task") or "").strip()
    if raw_name:
        mapped = challenge_task_id(raw_name)
        if mapped is not None:
            return mapped, str(challenge_task_name(mapped) or raw_name)
    if raw_id is not None and str(raw_id).strip() != "":
        try:
            task_id = int(raw_id)
        except (TypeError, ValueError):
            task_id = None
        if task_id is not None:
            name = challenge_task_name(task_id)
            if name:
                return task_id, name
    bucket_id, bucket_name = parse_task_bucket(parent_name)
    if bucket_id is not None and bucket_name:
        return bucket_id, bucket_name

    sid = str(session_id or data.get("session_id") or dir_name or "").strip()
    harness = _HARNESS_TASK_SID_RE.search(sid)
    if harness:
        task_id = int(harness.group(1))
        name = challenge_task_name(task_id)
        if name:
            return task_id, name

    blob = normalize_task_key(
        " ".join(
            (
                sid,
                str(prompt or data.get("user_prompt") or data.get("prompt") or ""),
                str(dir_name or ""),
            )
        )
    )
    if not blob:
        return None, None
    for key, task_id, task_name in _task_matchers():
        if key in blob:
            return task_id, task_name
    return None, None


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def normalize_loaded_skills(value: Any) -> list[str]:
    """hook / launcher 传来的 loaded_skills，去空去重并保持顺序。"""
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        raw = [str(item or "").strip() for item in value]
    else:
        raw = []
    names: list[str] = []
    for item in raw:
        if item and item not in names:
            names.append(item)
    return names


def infer_loaded_skills_from_prompt(prompt: str) -> list[str]:
    """没有 hook 上报时只承认 baseline；任务 skill 必须由原生 invoke 上报。"""
    del prompt
    return [BASELINE_SKILL]


def compact_args(args: Any) -> dict[str, Any]:
    """给卡片底部用的短参数，去掉 session / timeout 和过长字段。"""
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if len(out) >= _MAX_ARGS:
            break
        name = str(key)
        if name in _SKIP_ARG_KEYS:
            continue
        item = _safe_json(value)
        if isinstance(item, str) and len(item) > _MAX_ARG_CHARS:
            item = item[: _MAX_ARG_CHARS - 3] + "..."
        out[name] = item
    return out


def _as_coord(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def extract_model_points(args: Any) -> list[tuple[float, float]]:
    """从工具参数取出模型选点，坐标为 Qwen 相对系 0..1000。"""
    if not isinstance(args, dict):
        return []
    points: list[tuple[float, float]] = []

    def add(u_value: Any, v_value: Any) -> None:
        u_coord = _as_coord(u_value)
        v_coord = _as_coord(v_value)
        if u_coord is None or v_coord is None:
            return
        points.append((u_coord, v_coord))

    add(args.get("u"), args.get("v"))
    raw_points = args.get("points")
    if isinstance(raw_points, list):
        for item in raw_points:
            if isinstance(item, dict):
                add(item.get("u"), item.get("v"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                add(item[0], item[1])
    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        key = (round(point[0], 3), round(point[1], 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return unique


def extract_input_image_id(args: Any) -> str:
    """工具调用传入的 image_id；没有则空字符串。"""
    if not isinstance(args, dict):
        return ""
    return str(args.get("image_id") or "").strip()


def resolve_recorded_image(session_id: str, image_id: str) -> str:
    """按 session + image_id 在 agent_runs 落盘里找到 RGB 文件。"""
    sid = str(session_id or "").strip()
    iid = str(image_id or "").strip()
    if not sid or not iid:
        return ""
    try:
        from behavior_interface import agent_runs
    except Exception:
        return ""
    suffixes = (
        ".png",
        ".raw.png",
        ".left_wrist.png",
        ".right_wrist.png",
        ".left_wrist.raw.png",
        ".right_wrist.raw.png",
    )
    for suffix in suffixes:
        try:
            path = agent_runs.image_path(sid, iid, suffix)
        except (OSError, ValueError):
            continue
        if path and os.path.isfile(path):
            return path
    try:
        directory = agent_runs.images_dir(sid)
    except (OSError, ValueError):
        return ""
    if not os.path.isdir(directory):
        return ""
    skip = ("depth", "seg", "mask", "normal", "vector", "path")
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return ""
    for name in names:
        lower = name.lower()
        if not name.startswith(iid) or not lower.endswith(".png"):
            continue
        if any(token in lower for token in skip):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return ""


def extract_result_image(payload: Any) -> dict[str, str] | None:
    """从工具返回里取出结果图；plan 叠加图单独放 overlay_path。"""
    if not isinstance(payload, dict):
        return None
    candidates: list[dict[str, Any]] = []
    observation = payload.get("observation")
    if isinstance(observation, dict):
        candidates.append(observation)
    candidates.append(payload)
    for obj in candidates:
        image_id = str(obj.get("image_id") or "").strip()
        rgb_path = ""
        overlay_path = ""
        raw = str(obj.get("rgb_path") or obj.get("rgb_main_path") or "").strip()
        if raw and os.path.isfile(raw):
            rgb_path = raw
        for key in ("rgb_overlay_path", "render_image", "rgb_main_path"):
            path = str(obj.get(key) or "").strip()
            if path and os.path.isfile(path) and path != rgb_path:
                overlay_path = path
                break
        if not rgb_path:
            for key in _IMAGE_PATH_KEYS:
                path = str(obj.get(key) or "").strip()
                if path and os.path.isfile(path):
                    rgb_path = path
                    break
        if image_id or rgb_path or overlay_path:
            return {
                "image_id": image_id,
                "rgb_path": overlay_path or rgb_path,
                "overlay_path": overlay_path,
                "raw_path": rgb_path,
            }
    return None


def resolve_v2_tool_name(path: str, body: dict[str, Any] | None, catalog: list | None = None) -> str:
    """/api/v2/plan + mode 还原成 plan_* 工具名。"""
    slug = str(path or "").rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug:
        return "unknown"
    if slug != "plan":
        return slug
    mode = str((body or {}).get("mode") or "").strip()
    for tool in catalog or []:
        if not isinstance(tool, dict):
            continue
        endpoint = str(tool.get("endpoint") or "")
        if not endpoint.rstrip("/").endswith("/plan"):
            continue
        tool_mode = str(tool.get("mode") or "")
        if tool_mode and tool_mode == mode:
            return str(tool.get("name") or slug)
        prefix = str(tool.get("mode_prefix") or "")
        if prefix and mode.startswith(prefix):
            return str(tool.get("name") or slug)
    return f"plan:{mode}" if mode else "plan"


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".monitor-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# 一条车道的监视目录会留下几百个旧 session。session_begin 和每次
# GET /api/agent_monitor 都持锁扫它们，九路一起做就把 HTTP 线程占满，
# 新请求只能回 503，时钟也就建不起来。浏览器历史仍走全量扫描；
# 时钟和轮询只认 current.json。
_POLL_STATE_KEYS = frozenset({"poll", "monitor_poll"})


def _state_is_poll(state: dict[str, Any] | None) -> bool:
    return isinstance(state, dict) and bool(_POLL_STATE_KEYS.intersection(state))


def _draw_red_crosses(image: Any, points: list[tuple[float, float]]) -> Any:
    """在缩略图上画红色十字；坐标按 0..1000 映射到当前像素。"""
    if image is None or not points:
        return image
    height, width = image.shape[:2]
    if width <= 1 or height <= 1:
        return image
    arm = max(_CROSS_ARM_MIN_PX, min(width, height) // _CROSS_ARM_DIV)
    thickness = _CROSS_THICKNESS_PX
    for u_coord, v_coord in points:
        x = int(round(max(0.0, min(_COORDINATE_MAX, u_coord)) / _COORDINATE_MAX * (width - 1)))
        y = int(round(max(0.0, min(_COORDINATE_MAX, v_coord)) / _COORDINATE_MAX * (height - 1)))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        for color, width_px in (
            (_CROSS_OUTLINE_BGR, thickness + 1),
            (_RED_BGR, thickness),
        ):
            cv2.line(
                image,
                (x, max(0, y - arm)),
                (x, min(height - 1, y + arm)),
                color,
                width_px,
                cv2.LINE_AA,
            )
            cv2.line(
                image,
                (max(0, x - arm), y),
                (min(width - 1, x + arm), y),
                color,
                width_px,
                cv2.LINE_AA,
            )
    return image


def _is_model_image_path(path: str) -> bool:
    """MCP 回写的模型 JPEG，已经叠过 HUD，不能再叠一层。"""
    name = os.path.basename(str(path or ""))
    return name.startswith("model_") and name.lower().endswith((".jpg", ".jpeg", ".png"))


def _load_minimap_png() -> bytes | None:
    """和 /api/spatial_map.png 同一份实时小地图。失败就让卡片先用原图。"""
    try:
        from behavior_interface.spatial_map import (
            active_session_id,
            map_snapshot_png,
            spatial_map_enabled,
        )

        if not spatial_map_enabled():
            return None
        session_id = active_session_id() or "default"
        from behavior_interface.rtabmap_slam.live import (
            get_live_mapper,
            live_backend_selected,
        )

        if live_backend_selected():
            mapper = get_live_mapper()
            if mapper is None:
                return None
            mapper.adopt_session(session_id)
            png, _version = mapper.map_snapshot_png(heading_up=True)
        else:
            png, _version = map_snapshot_png(session_id, heading_up=True)
        return png or None
    except Exception:
        return None


def _overlay_minimap_hud_bgr(image: Any) -> Any:
    """按插件同一套几何：短边 30% 小地图 + 标题条，贴右上角。"""
    if image is None:
        return image
    raw = _load_minimap_png()
    if not raw:
        return image
    try:
        import numpy as np
    except Exception:
        return image
    mini = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if mini is None or mini.size == 0:
        return image
    height, width = image.shape[:2]
    edge = min(int(height), int(width))
    if edge <= 0:
        return image
    mini_px = max(64, int(round(edge * 0.30)))
    title_h = max(32, int(round(mini_px * 0.16)))
    hud_h = title_h + mini_px
    if hud_h > height or mini_px > width:
        return image
    mini = cv2.resize(mini, (mini_px, mini_px), interpolation=cv2.INTER_AREA)
    # #10141a → BGR
    bar = np.full((title_h, mini_px, 3), (0x1A, 0x14, 0x10), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale = max(0.28, title_h * 0.018)
    sub_scale = max(0.22, title_h * 0.014)
    cv2.putText(
        bar,
        "Live SLAM minimap",
        (4, max(12, int(title_h * 0.45))),
        font,
        main_scale,
        (0xF8, 0xF6, 0xF4),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "heading up",
        (4, max(18, int(title_h * 0.82))),
        font,
        sub_scale,
        (0xF8, 0xF6, 0xF4),
        1,
        cv2.LINE_AA,
    )
    hud = np.vstack((bar, mini))
    out = image.copy()
    x0 = width - mini_px
    out[0:hud_h, x0:width] = hud
    return out


def _write_bytes_atomic(path: str, data: bytes) -> bool:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        return False


def _write_thumb(
    src: str,
    dest: str,
    points: list[tuple[float, float]] | None = None,
    overlay_hud: bool = False,
) -> bool:
    """把关键帧缩成 JPEG；有选点时叠红色十字。失败则不写，前端回退原图。"""
    try:
        image = cv2.imread(src)
        if image is None:
            return False
        if overlay_hud and not _is_model_image_path(src):
            image = _overlay_minimap_hud_bgr(image)
        height, width = image.shape[:2]
        if width > _THUMB_MAX_W and width > 0:
            new_h = max(1, int(height * _THUMB_MAX_W / width))
            image = cv2.resize(image, (_THUMB_MAX_W, new_h), interpolation=cv2.INTER_AREA)
        image = _draw_red_crosses(image, points or [])
        ok, buf = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), _THUMB_JPEG_QUALITY],
        )
        if not ok:
            return False
        directory = os.path.dirname(dest)
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".thumb-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(buf.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, dest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True
    except Exception:
        return False


def _as_nonneg_int(value: Any) -> int | None:
    """把仿真 tick 收成 >=0 的整数；读不到就当未知。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def challenge_instance_max_ticks(
    task_id: int | str | None = None,
    task_name: str | None = None,
    port: int | str | None = None,
) -> int | None:
    """本任务 Challenge 2026 最大 ticks；查不到就空着，不猜预算。"""
    if os.environ.get('ROBOHARNESS_MAX_STEPS'):
        return int(os.environ['ROBOHARNESS_MAX_STEPS'])
    try:
        from official_eval_harness.catalog import challenge_2026_max_ticks

        value = challenge_2026_max_ticks(
            task_id=task_id,
            task_name=task_name,
            port=port,
        )
    except Exception:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def session_used_ticks(origin: Any, current: Any) -> int:
    """本 instance 已用 ticks：当前仿真 tick 减第一次 tool call 原点，从 0 起算。"""
    start = _as_nonneg_int(origin)
    now = _as_nonneg_int(current)
    if start is None or now is None:
        return 0
    return max(0, now - start)


@dataclass
class MonitorCard:
    card_id: int
    ts: str
    tool: str
    args: dict[str, Any]
    input_image_id: str = ""
    input_rgb_path: str = ""
    input_thumb_path: str = ""
    output_image_id: str = ""
    output_rgb_path: str = ""
    output_thumb_path: str = ""
    points: list[tuple[float, float]] = field(default_factory=list)
    ok: bool | None = None
    loaded_skills: list[str] = field(default_factory=list)
    model_image_v: str = ""
    session_ticks: int = 0

    def public(self) -> dict[str, Any]:
        has_input = bool(
            self.input_thumb_path
            or (self.input_rgb_path and os.path.isfile(self.input_rgb_path))
        )
        has_output = bool(
            self.output_thumb_path
            or (self.output_rgb_path and os.path.isfile(self.output_rgb_path))
        )
        return {
            "card_id": self.card_id,
            "ts": self.ts,
            "tool": self.tool,
            "args": dict(self.args),
            "input_image_id": self.input_image_id,
            "output_image_id": self.output_image_id,
            "image_id": self.output_image_id or self.input_image_id,
            "has_input": has_input,
            "has_output": has_output,
            "has_image": has_input or has_output,
            "points": [list(point) for point in self.points],
            "ok": self.ok,
            "loaded_skills": list(self.loaded_skills),
            "model_image_v": self.model_image_v,
            "session_ticks": int(self.session_ticks),
        }

    def persistent(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "ts": self.ts,
            "tool": self.tool,
            "args": dict(self.args),
            "input_image_id": self.input_image_id,
            "input_rgb_path": self.input_rgb_path,
            "input_thumb_path": self.input_thumb_path,
            "output_image_id": self.output_image_id,
            "output_rgb_path": self.output_rgb_path,
            "output_thumb_path": self.output_thumb_path,
            "image_id": self.output_image_id or self.input_image_id,
            "rgb_path": self.output_rgb_path or self.input_rgb_path,
            "thumb_path": self.output_thumb_path or self.input_thumb_path,
            "points": [list(point) for point in self.points],
            "ok": self.ok,
            "loaded_skills": list(self.loaded_skills),
            "model_image_v": self.model_image_v,
            "session_ticks": int(self.session_ticks),
        }


@dataclass
class MonitorSession:
    session_id: str
    run_dir: str
    prompt: str = ""
    loaded_skills: list[str] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""
    attempt_id: str = ""
    cards: list[MonitorCard] = field(default_factory=list)
    last_image_id: str = ""
    last_rgb_path: str = ""
    images: dict[str, str] = field(default_factory=dict)
    status: str = "recording"
    tick_origin: int | None = None
    world_reset_seq: int = 0

    def ensure_attempt_id(self) -> str:
        if not self.attempt_id:
            self.attempt_id = make_attempt_id(
                self.session_id, self.started_at, self.run_dir
            )
        return self.attempt_id


@dataclass
class CardContext:
    session_id: str
    tool: str
    args: dict[str, Any]


class AgentMonitorStore:
    """进程内直播状态 + NAS 上按 session 追加的卡片记录。"""

    def __init__(
        self,
        root: str | None = None,
        port: int | str | None = None,
        task_id: int | str | None = None,
        task_name: str | None = None,
        tick_reader: Any = None,
    ) -> None:
        base = os.path.abspath(root) if root else default_monitor_root()
        self.port = port
        self._base = base
        self.task_id: int | None = None
        self.task_name: str = ""
        self.root = base
        self._lock = threading.RLock()
        self._tick_reader = tick_reader
        self._live: MonitorSession | None = None
        self._pending_prompt = ""
        self._pending_skills: list[str] = []
        self._history_cache: list[dict[str, Any]] | None = None
        self._run_dir_index: dict[str, str] = {}
        self._world_reset_seq = 0
        self.bind_task(task_id, task_name, restore=False)
        if self.root == base and port is not None:
            self.root = os.path.join(base, f"p{port}")
        with self._lock:
            self._restore_live_locked()

    def bind_task(
        self,
        task_id: int | str | None,
        task_name: str | None = None,
        *,
        restore: bool = True,
    ) -> None:
        """下拉和落盘都跟当前任务走；换任务就换桶，不再按端口过滤。"""
        inferred_id, inferred_name = infer_monitor_task(
            {"task_id": task_id, "task_name": task_name or ""},
        )
        name = inferred_name or str(task_name or "").strip()
        tid = inferred_id
        if tid is None and str(task_id or "").strip() != "":
            try:
                tid = int(task_id)
            except (TypeError, ValueError):
                tid = None
        if tid == self.task_id and name == self.task_name:
            return
        self.task_id = tid
        self.task_name = name
        if tid is not None and name:
            self.root = os.path.join(self._base, task_bucket_name(tid, name))
        elif self.port is not None:
            self.root = os.path.join(self._base, f"p{self.port}")
        else:
            self.root = self._base
        with self._lock:
            self._history_cache = None
            self._run_dir_index.clear()
            live = self._live
            if live is not None:
                self._rehome_session_locked(live)
                self._write_session_locked(live)
            elif restore:
                self._restore_live_locked()

    def set_tick_reader(self, tick_reader: Any) -> None:
        self._tick_reader = tick_reader

    def _current_sim_tick(self, override: Any = None) -> int:
        pinned = _as_nonneg_int(override)
        if pinned is not None:
            return pinned
        reader = self._tick_reader
        if callable(reader):
            try:
                value = _as_nonneg_int(reader())
            except Exception:
                value = None
            if value is not None:
                return value
        return 0

    def _pin_tick_origin_locked(
        self,
        session: MonitorSession,
        sim_tick: Any = None,
    ) -> int:
        if session.tick_origin is None:
            session.tick_origin = self._current_sim_tick(sim_tick)
        return int(session.tick_origin)

    def begin_session_clock(
        self,
        session_id: str,
        sim_tick: Any = None,
    ) -> dict[str, Any]:
        """保证 session 存在。ticks 原点留给本 instance 第一次 tool call。"""
        sid = str(session_id or "").strip()
        if not is_external_agent_session(sid):
            raise ValueError("session_id 不能用于 agent monitor")
        with self._lock:
            live = self._ensure_session_locked(sid)
            current = self._current_sim_tick(sim_tick)
            origin = live.tick_origin
            self._write_session_locked(live)
            return {
                "session_id": live.session_id,
                "tick_origin": origin,
                "sim_tick": current,
                "session_ticks": session_used_ticks(origin, current),
                "world_reset_seq": int(live.world_reset_seq or 0),
                "max_ticks": challenge_instance_max_ticks(self.task_id, self.task_name, port=self.port),
            }

    def on_world_reset(self, sim_tick: Any = None) -> dict[str, Any]:
        """场景重置后 ticks 从 0 再计。同一 session 可以继续，只清原点。

        下一次 tool call（无论谁发）重新钉原点。compact / resume 不得走这条路。
        """
        with self._lock:
            self._world_reset_seq = int(self._world_reset_seq or 0) + 1
            seq = self._world_reset_seq
            live = self._live
            if live is None:
                return {
                    "ok": True,
                    "reset": False,
                    "world_reset_seq": seq,
                    "reason": "no_live_session",
                    "session_ticks": 0,
                    "max_ticks": challenge_instance_max_ticks(self.task_id, self.task_name, port=self.port),
                }
            live.tick_origin = None
            live.world_reset_seq = seq
            self._write_session_locked(live)
            current = self._current_sim_tick(sim_tick)
            return {
                "ok": True,
                "reset": True,
                "session_id": live.session_id,
                "attempt_id": live.ensure_attempt_id(),
                "tick_origin": None,
                "sim_tick": current,
                "session_ticks": 0,
                "world_reset_seq": seq,
                "max_ticks": challenge_instance_max_ticks(self.task_id, self.task_name, port=self.port),
            }

    def set_user_prompt(
        self,
        prompt: str,
        session_id: str | None = None,
        loaded_skills: Any = None,
        new_attempt: Any = False,
    ) -> dict[str, Any]:
        text = str(prompt or "").strip()
        sid = str(session_id or "").strip()
        skills = normalize_loaded_skills(loaded_skills)
        if not skills and text:
            skills = infer_loaded_skills_from_prompt(text)
        with self._lock:
            if text:
                self._pending_prompt = text
            if skills:
                self._pending_skills = list(skills)
            live = self._live
            if sid:
                # resume / compact 会复用同一个 BEHAVIOR_SESSION_ID。
                # 即使 launcher 仍带 new_attempt，也必须接着旧卡片写，不能切盘。
                live = self._ensure_session_locked(sid)
            if live is not None and (not sid or sid == live.session_id):
                if text:
                    live.prompt = text
                if skills:
                    self._apply_skills_locked(live, skills)
                if text or skills:
                    self._write_session_locked(live)
            return self.snapshot()

    def _apply_skills_locked(self, live: MonitorSession, skills: list[str]) -> None:
        live.loaded_skills = list(skills)
        for card in live.cards:
            if not card.loaded_skills:
                card.loaded_skills = list(skills)

    def begin_card(
        self,
        *,
        session_id: str,
        tool: str,
        args: dict[str, Any] | None,
    ) -> CardContext | None:
        sid = str(session_id or "").strip()
        name = str(tool or "").strip()
        if not is_external_agent_session(sid) or not name:
            return None
        with self._lock:
            live = self._ensure_session_locked(sid)
            # 本 instance 第一次 tool call 钉 ticks 原点；场景重置后原点为空会再钉。
            self._pin_tick_origin_locked(live)
            return CardContext(session_id=sid, tool=name, args=compact_args(args or {}))

    def finish_card(
        self,
        context: CardContext | None,
        response: Any,
        sim_tick: Any = None,
    ) -> dict[str, Any] | None:
        if context is None:
            return None
        payload = response if isinstance(response, dict) else {}
        with self._lock:
            live = self._live
            if live is None or live.session_id != context.session_id:
                return None
            images = self._images_locked(live)
            card_id = len(live.cards) + 1
            points = extract_model_points(context.args)
            input_id = extract_input_image_id(context.args)
            input_path = self._lookup_image_locked(live, input_id)
            seen = extract_result_image(payload) or {}
            output_id = str(seen.get("image_id") or "").strip()
            overlay_path = str(seen.get("overlay_path") or "").strip()
            raw_path = str(seen.get("raw_path") or seen.get("rgb_path") or "").strip()
            output_path = overlay_path if overlay_path and os.path.isfile(overlay_path) else raw_path
            if output_id and (not output_path or not os.path.isfile(output_path)):
                output_path = self._lookup_image_locked(live, output_id)
            if output_id and raw_path and os.path.isfile(raw_path):
                images[output_id] = raw_path
                live.last_image_id = output_id
                live.last_rgb_path = raw_path
            same_file = bool(
                input_path
                and output_path
                and os.path.isfile(input_path)
                and os.path.isfile(output_path)
                and os.path.samefile(input_path, output_path)
            )
            # 选点工具（plan / move_to_reach_point）左右两张：输入带十字，输出放结果或叠加图。
            show_output = bool(
                output_path
                and os.path.isfile(output_path)
                and (not same_file or overlay_path or points)
            )
            input_thumb = ""
            output_thumb = ""
            if input_path and os.path.isfile(input_path):
                name = f"card_{card_id:04d}_in.jpg"
                dest = os.path.join(live.run_dir, "thumbs", name)
                if _write_thumb(
                    input_path,
                    dest,
                    points,
                    overlay_hud=not _is_model_image_path(input_path),
                ):
                    input_thumb = os.path.join("thumbs", name)
            if show_output:
                name = f"card_{card_id:04d}_out.jpg"
                dest = os.path.join(live.run_dir, "thumbs", name)
                if _write_thumb(
                    output_path,
                    dest,
                    None,
                    overlay_hud=not _is_model_image_path(output_path),
                ):
                    output_thumb = os.path.join("thumbs", name)
            ok = payload.get("ok")
            if not isinstance(ok, bool):
                ok = None
            origin = self._pin_tick_origin_locked(live, sim_tick)
            card = MonitorCard(
                card_id=card_id,
                ts=utc_now(),
                tool=context.tool,
                args=dict(context.args),
                input_image_id=input_id,
                input_rgb_path=input_path if os.path.isfile(input_path) else "",
                input_thumb_path=input_thumb,
                output_image_id=output_id if show_output else "",
                output_rgb_path=output_path if show_output else "",
                output_thumb_path=output_thumb,
                points=points,
                ok=ok,
                loaded_skills=list(live.loaded_skills or self._pending_skills),
                session_ticks=session_used_ticks(origin, self._current_sim_tick(sim_tick)),
            )
            live.cards.append(card)
            self._append_card_locked(live, card)
            self._write_session_locked(live)
            return card.public()

    def apply_model_image(
        self,
        session_id: str,
        image_id: str,
        data: bytes,
        tool: str = "",
    ) -> dict[str, Any]:
        """用模型实际看到的 JPEG 覆盖卡片图，保证监视器和 Codex 是同一张。"""
        payload = data if isinstance(data, (bytes, bytearray)) else b""
        if not payload:
            return {"ok": False, "error": "empty image"}
        is_jpeg = bytes(payload[:3]) == b"\xff\xd8\xff"
        is_png = bytes(payload[:8]) == b"\x89PNG\r\n\x1a\n"
        if not is_jpeg and not is_png:
            return {"ok": False, "error": "model image must be jpeg or png"}
        sid = str(session_id or "").strip()
        iid = str(image_id or "").strip()
        tool_name = str(tool or "").strip()
        ext = ".jpg" if is_jpeg else ".png"
        with self._lock:
            live = self._live
            if sid:
                live = self._ensure_session_locked(sid)
            if live is None:
                return {"ok": False, "error": "no live monitor session"}
            thumbs_dir = os.path.join(live.run_dir, "thumbs")
            os.makedirs(thumbs_dir, exist_ok=True)
            model_name = f"model_{iid or 'latest'}{ext}"
            model_rel = os.path.join("thumbs", model_name)
            model_abs = os.path.join(live.run_dir, model_rel)
            if not _write_bytes_atomic(model_abs, bytes(payload)):
                return {"ok": False, "error": "failed to write model image"}
            if iid:
                self._images_locked(live)[iid] = model_abs
            rev = str(int(time.time() * 1000))
            matched = 0

            def _stamp_output(card: MonitorCard) -> None:
                nonlocal matched
                dest = card.output_thumb_path
                if not dest:
                    dest = os.path.join("thumbs", f"card_{card.card_id:04d}_out.jpg")
                    card.output_thumb_path = dest
                if not os.path.isabs(dest):
                    dest = os.path.join(live.run_dir, dest)
                _write_bytes_atomic(dest, bytes(payload))
                card.output_rgb_path = model_abs
                card.model_image_v = rev
                matched += 1

            for card in reversed(live.cards):
                if iid and card.output_image_id == iid:
                    _stamp_output(card)
                elif not iid and tool_name and card.tool == tool_name:
                    _stamp_output(card)
                    break
                if iid and card.input_image_id == iid and card.input_thumb_path:
                    dest = card.input_thumb_path
                    if not os.path.isabs(dest):
                        dest = os.path.join(live.run_dir, dest)
                    _write_thumb(model_abs, dest, card.points, overlay_hud=False)
                    card.input_rgb_path = model_abs
                    if not card.model_image_v:
                        card.model_image_v = rev
                    matched += 1
            if matched == 0:
                for card in reversed(live.cards):
                    if card.output_image_id or card.output_rgb_path or card.output_thumb_path:
                        _stamp_output(card)
                        break
            live.updated_at = utc_now()
            self._write_session_locked(live)
            return {
                "ok": True,
                "session_id": live.session_id,
                "image_id": iid,
                "model_image_v": rev,
                "updated_cards": matched,
            }

    def snapshot(
        self,
        state: dict[str, Any] | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        # None means "build the browser snapshot".  An explicit empty dict is
        # the poll path: session id and ticks do not need that payload, and
        # building it under load leaves one Flask thread per timed-out GET.
        if state is None:
            state_provider = getattr(self, "_state_provider", None)
            if callable(state_provider):
                try:
                    state = state_provider() or {}
                except Exception:
                    state = {}
            else:
                state = {}
        wanted = str(attempt_id or "").strip()
        with self._lock:
            if self._live is None:
                self._restore_live_locked()
            live = self._live
            # 轮询只问当前 session 和 tick。全量历史要把几百个旧目录的
            # session.json 从共享盘读进来，持锁期间 session_begin 进不来。
            if _state_is_poll(state) and not wanted:
                sessions = (
                    [self._session_public_locked(live, live=True)]
                    if live is not None
                    else []
                )
            else:
                sessions = self._history_locked()
            live_attempt = live.ensure_attempt_id() if live is not None else ""
            viewed = live
            viewing_history = False
            if wanted and wanted != live_attempt:
                viewed = self._load_attempt_locked(wanted)
                viewing_history = viewed is not None
                if viewed is None:
                    viewed = live
                    viewing_history = False
            prompt = (viewed.prompt if viewed is not None else "") or self._pending_prompt
            skills = list(
                (viewed.loaded_skills if viewed is not None else []) or self._pending_skills
            )
            cards = [card.public() for card in (viewed.cards if viewed is not None else [])]
            return {
                "ok": True,
                "active": live is not None,
                "preview": live is None,
                "blank_until_run": live is None and not prompt,
                "monitor_source": "embodied_keyframes",
                "task_id": self.task_id,
                "task_name": self.task_name,
                "user_prompt": prompt,
                "prompt": prompt,
                "loaded_skills": skills,
                "session_id": viewed.session_id if viewed is not None else "",
                "started_at": viewed.started_at if viewed is not None else "",
                "attempt_id": viewed.ensure_attempt_id() if viewed is not None else "",
                "live_attempt_id": live_attempt,
                "live_session_id": live.session_id if live is not None else "",
                "viewing_history": viewing_history,
                "sessions": sessions,
                "cards": cards,
                "card_count": len(cards),
                "updated_at": time.time(),
                "server_pid": state.get("pid"),
                "server_started_ts": state.get("started_ts"),
                "server_reset_count": state.get("reset_count"),
                "run_dir": viewed.run_dir if viewed is not None else "",
                "tick_origin": viewed.tick_origin if viewed is not None else None,
                "session_ticks": (
                    session_used_ticks(viewed.tick_origin, self._current_sim_tick())
                    if viewed is not None
                    else 0
                ),
                "world_reset_seq": (
                    int(viewed.world_reset_seq or 0) if viewed is not None else self._world_reset_seq
                ),
                "live_world_reset_seq": (
                    int(live.world_reset_seq or 0) if live is not None else self._world_reset_seq
                ),
                "max_ticks": challenge_instance_max_ticks(self.task_id, self.task_name, port=self.port),
            }

    def card_image_path(
        self,
        session_id: str,
        card_id: Any,
        kind: str | None = None,
    ) -> str | None:
        try:
            index = int(card_id)
        except (TypeError, ValueError):
            return None
        role = str(kind or "output").strip().lower()
        if role not in {"input", "output"}:
            role = "output"
        sid = str(session_id or "").strip()
        # 历史缩略图按约定文件名直取，不拿直播锁、不扫整棵 NAS。
        # 旧实现每张图都 _iter_session_dirs + 读 cards.jsonl，单张 10s+，页面裂开。
        found = self._convention_card_image(sid, index, role)
        if found:
            return found
        with self._lock:
            run_dir = self._resolve_run_dir_locked(sid)
            if not run_dir:
                return None
            live = self._live
            if (
                live is not None
                and os.path.isdir(live.run_dir)
                and os.path.abspath(live.run_dir) == os.path.abspath(run_dir)
            ):
                card = next((item for item in live.cards if item.card_id == index), None)
                if card is not None:
                    return self._card_file_locked(run_dir, card.persistent(), role)
            persisted = self._load_persisted_card(run_dir, index)
            if persisted is None:
                return None
            return self._card_file_locked(run_dir, persisted, role)

    def _lookup_roots(self) -> list[str]:
        roots: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            text = str(path or "").strip()
            if not text:
                return
            key = os.path.abspath(text)
            if key in seen:
                return
            seen.add(key)
            roots.append(text)

        _add(self.root)
        _add(self._base)
        if self.port is not None:
            _add(os.path.join(self._base, f"p{self.port}"))
        if self.task_id is not None and self.task_name:
            _add(os.path.join(self._base, task_bucket_name(self.task_id, self.task_name)))
        return roots

    def _candidate_run_dirs(self, token: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for alias in monitor_token_aliases(token):
            for root in self._lookup_roots():
                path = os.path.join(root, alias)
                if not os.path.isdir(path):
                    continue
                key = os.path.abspath(path)
                if key in seen:
                    continue
                seen.add(key)
                out.append(path)
        return out

    def _convention_card_image(self, token: str, card_id: int, role: str) -> str | None:
        roles = [role] if role == "input" else [role, "input"]
        for run_dir in self._candidate_run_dirs(token):
            thumbs = os.path.join(run_dir, "thumbs")
            for current in roles:
                for name in convention_card_thumb_names(card_id, current):
                    path = os.path.join(thumbs, name)
                    if os.path.isfile(path):
                        return path
        return None

    def _index_run_dir_locked(self, session: MonitorSession) -> None:
        path = str(session.run_dir or "").strip()
        if not path:
            return
        for key in (session.session_id, session.attempt_id, os.path.basename(path)):
            text = str(key or "").strip()
            if text:
                self._run_dir_index[text] = path

    def _refresh_history_entry_locked(self, session: MonitorSession) -> None:
        cache = self._history_cache
        if cache is None:
            return
        public = self._session_public_locked(
            session,
            live=session.status == "recording",
        )
        aid = public.get("attempt_id")
        sid = public.get("session_id")
        for item in cache:
            if item.get("attempt_id") == aid or (
                item.get("session_id") == sid and bool(item.get("live"))
            ):
                item.update(public)
                return
        cache.insert(0, public)

    def _images_locked(self, session: MonitorSession) -> dict[str, str]:
        images = getattr(session, "images", None)
        if not isinstance(images, dict):
            session.images = {}
            images = session.images
        return images

    def _lookup_image_locked(self, session: MonitorSession, image_id: str) -> str:
        iid = str(image_id or "").strip()
        if not iid:
            return ""
        images = self._images_locked(session)
        cached = str(images.get(iid) or "").strip()
        if cached and os.path.isfile(cached):
            return cached
        model_guess = os.path.join(session.run_dir, "thumbs", f"model_{iid}.jpg")
        if os.path.isfile(model_guess):
            images[iid] = model_guess
            return model_guess
        found = resolve_recorded_image(session.session_id, iid)
        if found:
            images[iid] = found
            return found
        if session.last_image_id == iid and session.last_rgb_path and os.path.isfile(session.last_rgb_path):
            images[iid] = session.last_rgb_path
            return session.last_rgb_path
        return ""

    def _card_file_locked(self, run_dir: str, card: dict[str, Any], role: str) -> str | None:
        if role == "input":
            thumbs = [card.get("input_thumb_path") or ""]
            rgbs = [card.get("input_rgb_path") or ""]
        else:
            thumbs = [card.get("output_thumb_path") or "", card.get("thumb_path") or ""]
            rgbs = [card.get("output_rgb_path") or "", card.get("rgb_path") or ""]
        for thumb in thumbs:
            path = self._resolve_existing(run_dir, str(thumb), "")
            if path:
                return path
        for rgb in rgbs:
            path = self._resolve_existing(run_dir, "", str(rgb))
            if path:
                return path
        if role == "output":
            return self._card_file_locked(run_dir, card, "input")
        return None

    def _resolve_existing(self, run_dir: str, thumb_rel: str, rgb_path: str) -> str | None:
        if thumb_rel:
            thumb_abs = thumb_rel if os.path.isabs(thumb_rel) else os.path.join(run_dir, thumb_rel)
            if os.path.isfile(thumb_abs):
                return thumb_abs
        if rgb_path and os.path.isfile(rgb_path):
            return rgb_path
        return None

    def _restore_live_locked(self) -> MonitorSession | None:
        """没有新 session 时，把本端口上一轮 attempt 读回内存，刷新/重启都还在。"""
        if self._live is not None:
            return self._live
        session_id = self._discover_restore_session_id_locked()
        if not session_id:
            return None
        session = self._load_session_dir_locked(session_id)
        if session is None:
            return None
        session.status = "recording"
        self._live = session
        self._world_reset_seq = max(
            int(self._world_reset_seq or 0),
            int(session.world_reset_seq or 0),
        )
        if session.prompt:
            self._pending_prompt = session.prompt
        if session.loaded_skills:
            self._pending_skills = list(session.loaded_skills)
        return session

    def _discover_restore_session_id_locked(self) -> str:
        pointer = self._read_json_file(os.path.join(self.root, "current.json"))
        sid = str((pointer or {}).get("session_id") or "").strip()
        if sid and (
            str((pointer or {}).get("run_dir") or "").strip()
            or self._session_dir(self.root, sid)
        ):
            return sid
        # 指针文件在，但没有可用 session 时不再扫旧目录。一条车道有
        # 几百个旧 session.json，持锁打开它们会让 session_begin 超时。
        if pointer is not None:
            return ""
        newest = ("", "")
        for root, require_port in self._scan_roots():
            if not os.path.isdir(root):
                continue
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for name in names:
                if name in {"current.json", "thumbs"} or name.startswith("."):
                    continue
                payload = self._read_json_file(os.path.join(root, name, "session.json"))
                if not isinstance(payload, dict):
                    continue
                if require_port and not self._session_matches_store(
                    payload=payload,
                    session_id=str(payload.get("session_id") or name),
                    dir_name=name,
                    parent_name=os.path.basename(root),
                    prompt=str(payload.get("user_prompt") or payload.get("prompt") or ""),
                ):
                    continue
                sid = str(payload.get("session_id") or name).strip()
                if not sid:
                    continue
                updated = str(payload.get("updated_at") or "")
                status = str(payload.get("status") or "")
                key = ("1" if status == "recording" else "0") + updated
                if key > newest[0]:
                    newest = (key, sid)
        return newest[1]

    def _scan_roots(self) -> list[tuple[str, bool]]:
        if self.task_id is not None or self.task_name:
            return [(path, True) for path in self._iter_bucket_dirs()]
        roots = [(self.root, False)]
        if self.port is not None and self._base != self.root:
            roots.append((self._base, True))
        return roots

    def _iter_bucket_dirs(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for path in (self.root, self._base):
            key = os.path.abspath(path)
            if key in seen or not os.path.isdir(path):
                continue
            seen.add(key)
            out.append(path)
        if not os.path.isdir(self._base):
            return out
        try:
            names = os.listdir(self._base)
        except OSError:
            return out
        for name in names:
            if name in {"current.json", "thumbs"} or name.startswith("."):
                continue
            path = os.path.join(self._base, name)
            if not os.path.isdir(path):
                continue
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    def _session_belongs_here(self, session_id: str) -> bool:
        return self._session_matches_store(session_id=session_id, dir_name=session_id)

    def _session_matches_store(
        self,
        *,
        payload: dict[str, Any] | None = None,
        session_id: str = "",
        dir_name: str = "",
        parent_name: str = "",
        prompt: str = "",
    ) -> bool:
        sid = str(session_id or "").strip()
        if self.task_id is None and not self.task_name:
            if self.port is None:
                return True
            return bool(sid) and str(self.port) in sid
        inferred_id, inferred_name = infer_monitor_task(
            payload,
            session_id=sid,
            prompt=prompt,
            dir_name=dir_name,
            parent_name=parent_name,
        )
        if inferred_id is not None and self.task_id is not None:
            return inferred_id == self.task_id
        if inferred_name and self.task_name:
            return inferred_name == self.task_name
        return False

    def _session_dir(self, root: str, session_id: str) -> str:
        path = os.path.join(root, session_id)
        return path if os.path.isdir(path) else ""

    def _read_json_file(self, path: str) -> dict[str, Any] | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _load_session_dir_locked(self, session_id: str) -> MonitorSession | None:
        sid = str(session_id or "").strip()
        candidates: list[str] = []
        pointer = self._read_json_file(os.path.join(self.root, "current.json"))
        if pointer and str(pointer.get("session_id") or "").strip() == sid:
            pointed = str(pointer.get("run_dir") or "").strip()
            if pointed:
                candidates.append(pointed)
        # 指针已经给出这一轮的目录时，不再把任务桶里的旧 session 全打开。
        if not candidates:
            for root, _require_port in self._scan_roots():
                candidate = self._session_dir(root, sid)
                if candidate:
                    candidates.append(candidate)
        seen: set[str] = set()
        for run_dir in candidates:
            key = os.path.abspath(run_dir)
            if key in seen or not os.path.isdir(run_dir):
                continue
            seen.add(key)
            found = self._read_json_file(os.path.join(run_dir, "session.json"))
            if found is None:
                continue
            found_sid = str(found.get("session_id") or "").strip()
            if found_sid and found_sid != sid:
                continue
            return self._session_from_payload_locked(run_dir, found)
        return None

    def _session_from_payload_locked(
        self,
        run_dir: str,
        payload: dict[str, Any],
    ) -> MonitorSession:
        images = payload.get("images")
        started = str(payload.get("started_at") or "")
        sid = str(payload.get("session_id") or os.path.basename(run_dir)).strip()
        session = MonitorSession(
            session_id=sid,
            run_dir=run_dir,
            prompt=str(payload.get("user_prompt") or payload.get("prompt") or ""),
            loaded_skills=normalize_loaded_skills(payload.get("loaded_skills")),
            started_at=started,
            updated_at=str(payload.get("updated_at") or started),
            attempt_id=str(payload.get("attempt_id") or "").strip(),
            cards=self._load_persisted_cards(run_dir),
            last_image_id=str(payload.get("last_image_id") or ""),
            last_rgb_path=str(payload.get("last_rgb_path") or ""),
            images=dict(images) if isinstance(images, dict) else {},
            status=str(payload.get("status") or "recording") or "recording",
            tick_origin=_as_nonneg_int(payload.get("tick_origin")),
            world_reset_seq=int(_as_nonneg_int(payload.get("world_reset_seq")) or 0),
        )
        session.ensure_attempt_id()
        return session

    def _session_public_locked(self, session: MonitorSession, *, live: bool) -> dict[str, Any]:
        return {
            "attempt_id": session.ensure_attempt_id(),
            "session_id": session.session_id,
            "started_at": session.started_at,
            "updated_at": session.updated_at,
            "card_count": len(session.cards),
            "status": session.status,
            "live": live,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "user_prompt": session.prompt,
            "loaded_skills": list(session.loaded_skills),
            "tick_origin": session.tick_origin,
            "world_reset_seq": int(session.world_reset_seq or 0),
            "cards": [card.public() for card in session.cards],
        }

    def _history_locked(self) -> list[dict[str, Any]]:
        if self._history_cache is not None:
            live = self._live
            if live is not None:
                live_id = live.ensure_attempt_id()
                for item in self._history_cache:
                    if item.get("attempt_id") == live_id:
                        item.update(self._session_public_locked(live, live=True))
                        break
                else:
                    self._history_cache.insert(0, self._session_public_locked(live, live=True))
            return self._history_cache
        found: dict[str, dict[str, Any]] = {}
        live = self._live
        live_id = live.ensure_attempt_id() if live is not None else ""
        live_dir = os.path.abspath(live.run_dir) if live is not None else ""
        if live is not None:
            found[live_id] = self._session_public_locked(live, live=True)
        for run_dir, payload in self._iter_session_dirs_locked():
            if live_dir and os.path.abspath(run_dir) == live_dir:
                continue
            session = self._session_from_payload_locked(run_dir, payload)
            item = self._session_public_locked(session, live=False)
            found[item["attempt_id"]] = item
        sessions = list(found.values())
        sessions.sort(
            key=lambda item: (
                1 if item.get("live") else 0,
                str(item.get("updated_at") or ""),
                str(item.get("started_at") or ""),
            ),
            reverse=True,
        )
        self._history_cache = sessions
        return sessions

    def _iter_session_dirs_locked(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for root, require_port in self._scan_roots():
            if not os.path.isdir(root):
                continue
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for name in names:
                if name in {"current.json", "thumbs"} or name.startswith("."):
                    continue
                run_dir = os.path.join(root, name)
                key = os.path.abspath(run_dir)
                if key in seen or not os.path.isdir(run_dir):
                    continue
                payload = self._read_json_file(os.path.join(run_dir, "session.json"))
                if not isinstance(payload, dict):
                    continue
                sid = str(payload.get("session_id") or name).strip()
                if require_port and not self._session_matches_store(
                    payload=payload,
                    session_id=sid,
                    dir_name=name,
                    parent_name=os.path.basename(root),
                    prompt=str(payload.get("user_prompt") or payload.get("prompt") or ""),
                ):
                    continue
                seen.add(key)
                out.append((run_dir, payload))
        return out

    def _load_attempt_locked(self, attempt_id: str) -> MonitorSession | None:
        token = str(attempt_id or "").strip()
        if not token:
            return None
        live = self._live
        if live is not None and live.ensure_attempt_id() == token:
            return live
        run_dir = self._resolve_run_dir_locked(token)
        if not run_dir:
            return None
        payload = self._read_json_file(os.path.join(run_dir, "session.json"))
        if payload is None:
            return None
        return self._session_from_payload_locked(run_dir, payload)

    def _resolve_run_dir_locked(self, token: str) -> str:
        text = str(token or "").strip()
        if not text:
            return ""
        live = self._live
        if live is not None and text in {
            live.ensure_attempt_id(),
            live.session_id,
            os.path.basename(live.run_dir),
        }:
            return live.run_dir
        for alias in monitor_token_aliases(text):
            indexed = str(self._run_dir_index.get(alias) or "").strip()
            if indexed and os.path.isdir(indexed):
                return indexed
        candidates = self._candidate_run_dirs(text)
        if candidates:
            return candidates[0]
        for run_dir, payload in self._iter_session_dirs_locked():
            sid = str(payload.get("session_id") or "").strip()
            attempt = str(payload.get("attempt_id") or "").strip()
            if text in {os.path.basename(run_dir), sid, attempt}:
                if not attempt:
                    attempt = make_attempt_id(sid, str(payload.get("started_at") or ""), run_dir)
                if text in {os.path.basename(run_dir), attempt} or (
                    text == sid and (live is None or live.session_id != sid)
                ):
                    return run_dir
                if text == sid:
                    return run_dir
        return ""

    def _load_persisted_cards(self, run_dir: str) -> list[MonitorCard]:
        path = os.path.join(run_dir, "cards.jsonl")
        if not os.path.isfile(path):
            return []
        cards: list[MonitorCard] = []
        try:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    card = self._card_from_persistent(item)
                    if card is not None:
                        cards.append(card)
        except OSError:
            return []
        cards.sort(key=lambda item: item.card_id)
        return cards

    def _card_from_persistent(self, data: dict[str, Any]) -> MonitorCard | None:
        try:
            card_id = int(data.get("card_id") or 0)
        except (TypeError, ValueError):
            return None
        if card_id <= 0:
            return None
        points: list[tuple[float, float]] = []
        for item in data.get("points") or []:
            if isinstance(item, dict):
                u_coord = _as_coord(item.get("u"))
                v_coord = _as_coord(item.get("v"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                u_coord = _as_coord(item[0])
                v_coord = _as_coord(item[1])
            else:
                continue
            if u_coord is None or v_coord is None:
                continue
            points.append((u_coord, v_coord))
        args = data.get("args")
        return MonitorCard(
            card_id=card_id,
            ts=str(data.get("ts") or ""),
            tool=str(data.get("tool") or ""),
            args=dict(args) if isinstance(args, dict) else {},
            input_image_id=str(data.get("input_image_id") or ""),
            input_rgb_path=str(data.get("input_rgb_path") or ""),
            input_thumb_path=str(data.get("input_thumb_path") or ""),
            output_image_id=str(data.get("output_image_id") or data.get("image_id") or ""),
            output_rgb_path=str(data.get("output_rgb_path") or data.get("rgb_path") or ""),
            output_thumb_path=str(data.get("output_thumb_path") or data.get("thumb_path") or ""),
            points=points,
            ok=data.get("ok") if isinstance(data.get("ok"), bool) else None,
            loaded_skills=normalize_loaded_skills(data.get("loaded_skills")),
            model_image_v=str(data.get("model_image_v") or ""),
            session_ticks=session_used_ticks(0, data.get("session_ticks")),
        )

    def _ensure_session_locked(self, session_id: str) -> MonitorSession:
        live = self._live
        if live is not None and live.session_id == session_id:
            # 场景重置后的空 live 不能再把归档卡片捡回来，否则 tips 原点会回到旧局。
            if live.cards or int(live.world_reset_seq or 0) > 0:
                return live
        existing = self._load_session_dir_locked(session_id)
        if existing is not None and existing.cards:
            return self._activate_session_locked(existing, previous=live)
        archived = self._latest_session_with_cards_locked(session_id)
        if archived is not None:
            if live is not None and live.session_id == session_id and not live.cards:
                self._discard_empty_session_dir_locked(live)
                live = None
            return self._activate_session_locked(archived, previous=live)
        if live is not None and live.session_id == session_id:
            return live
        if existing is not None:
            return self._activate_session_locked(existing, previous=live)
        if live is not None:
            live.status = "completed"
            self._write_session_locked(live)
        return self._create_session_locked(session_id)

    def _activate_session_locked(
        self,
        session: MonitorSession,
        previous: MonitorSession | None,
    ) -> MonitorSession:
        if previous is not None and previous is not session:
            if previous.session_id != session.session_id:
                previous.status = "completed"
                self._write_session_locked(previous)
        session.status = "recording"
        self._rehome_session_locked(session)
        self._live = session
        self._write_session_locked(session)
        return session

    def _latest_session_with_cards_locked(self, session_id: str) -> MonitorSession | None:
        sid = str(session_id or "").strip()
        best: MonitorSession | None = None
        best_key = ""
        for run_dir, payload in self._iter_session_dirs_locked():
            found_sid = str(payload.get("session_id") or "").strip()
            if found_sid != sid:
                continue
            session = self._session_from_payload_locked(run_dir, payload)
            if not session.cards:
                continue
            key = str(session.updated_at or session.started_at or "")
            if key >= best_key:
                best_key = key
                best = session
        return best

    def _rehome_session_locked(self, session: MonitorSession) -> None:
        dest = os.path.join(self.root, session.session_id)
        src = str(session.run_dir or "").strip()
        if not src or os.path.abspath(src) == os.path.abspath(dest):
            return
        if os.path.isdir(dest):
            payload = self._read_json_file(os.path.join(dest, "session.json"))
            dest_cards = self._load_persisted_cards(dest) if payload else []
            if dest_cards:
                return
            shutil.rmtree(dest, ignore_errors=True)
        parent = os.path.dirname(dest)
        os.makedirs(parent, exist_ok=True)
        os.rename(src, dest)
        session.run_dir = dest

    def _discard_empty_session_dir_locked(self, session: MonitorSession) -> None:
        if session.cards:
            return
        path = str(session.run_dir or "").strip()
        if path and os.path.isdir(path) and not self._load_persisted_cards(path):
            shutil.rmtree(path, ignore_errors=True)

    def _start_new_attempt_locked(self, session_id: str) -> MonitorSession:
        """兼容旧调用：同一 session_id 继续写，不再切盘。"""
        return self._ensure_session_locked(session_id)

    def _create_session_locked(self, session_id: str) -> MonitorSession:
        started = utc_now()
        run_dir = os.path.join(self.root, session_id)
        os.makedirs(os.path.join(run_dir, "thumbs"), exist_ok=True)
        os.makedirs(self.root, exist_ok=True)
        session = MonitorSession(
            session_id=session_id,
            run_dir=run_dir,
            prompt=self._pending_prompt,
            loaded_skills=list(self._pending_skills),
            started_at=started,
            updated_at=started,
            attempt_id=make_attempt_id(session_id, started, run_dir),
            world_reset_seq=int(self._world_reset_seq or 0),
        )
        self._live = session
        self._write_session_locked(session)
        return session

    def _archive_run_dir_locked(self, session: MonitorSession) -> str:
        src = str(session.run_dir or "").strip()
        if not src or not os.path.isdir(src):
            return ""
        parent = os.path.dirname(src)
        base = f"{session.session_id}-{_attempt_stamp(session.started_at)}"
        dest = os.path.join(parent, base)
        suffix = 2
        while os.path.exists(dest):
            dest = os.path.join(parent, f"{base}-{suffix}")
            suffix += 1
        os.rename(src, dest)
        session.run_dir = dest
        self._write_session_locked(session)
        return dest

    def _write_session_locked(self, session: MonitorSession) -> None:
        session.updated_at = utc_now()
        session.ensure_attempt_id()
        self._index_run_dir_locked(session)
        self._refresh_history_entry_locked(session)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session.session_id,
            "attempt_id": session.attempt_id,
            "status": session.status,
            "started_at": session.started_at,
            "updated_at": session.updated_at,
            "user_prompt": session.prompt,
            "loaded_skills": list(session.loaded_skills),
            "card_count": len(session.cards),
            "last_image_id": session.last_image_id,
            "last_rgb_path": session.last_rgb_path,
            "images": dict(self._images_locked(session)),
            "run_dir": session.run_dir,
            "tick_origin": session.tick_origin,
            "world_reset_seq": int(session.world_reset_seq or 0),
            "task_id": self.task_id,
            "task_name": self.task_name,
        }
        _write_json_atomic(os.path.join(session.run_dir, "session.json"), payload)
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session.session_id if session.status == "recording" else "",
            "run_dir": session.run_dir,
            "updated_at": payload["updated_at"],
        }
        _write_json_atomic(os.path.join(self.root, "current.json"), pointer)

    def _append_card_locked(self, session: MonitorSession, card: MonitorCard) -> None:
        path = os.path.join(session.run_dir, "cards.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(card.persistent(), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _load_persisted_card(self, run_dir: str, card_id: int) -> dict[str, Any] | None:
        path = os.path.join(run_dir, "cards.jsonl")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(item.get("card_id") or 0) == card_id:
                        return item
        except OSError:
                    return None
        return None


def _iter_existing_session_dirs(base: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    """列出 (run_dir, dir_name, parent_name, payload)。"""
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    if not os.path.isdir(base):
        return out
    try:
        names = os.listdir(base)
    except OSError:
        return out

    def _read(path: str) -> dict[str, Any] | None:
        if not os.path.isfile(os.path.join(path, "session.json")):
            return None
        try:
            with open(os.path.join(path, "session.json"), encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    for name in names:
        if name in {"current.json", "thumbs"} or name.startswith("."):
            continue
        path = os.path.join(base, name)
        if not os.path.isdir(path) or os.path.islink(path):
            # 旧 p{port} 软链先跳过，避免把任务桶扫两遍。
            if os.path.islink(path):
                continue
            continue
        payload = _read(path)
        if payload is not None:
            out.append((path, name, "", payload))
            continue
        if not (_PORT_BUCKET_RE.fullmatch(name) or _TASK_BUCKET_RE.fullmatch(name)):
            continue
        try:
            children = os.listdir(path)
        except OSError:
            continue
        for child in children:
            if child in {"current.json", "thumbs"} or child.startswith("."):
                continue
            run_dir = os.path.join(path, child)
            if not os.path.isdir(run_dir):
                continue
            child_payload = _read(run_dir)
            if child_payload is None:
                continue
            out.append((run_dir, child, name, child_payload))
    return out


def rebind_monitor_sessions_by_task(
    base: str | None = None,
    *,
    dry_run: bool = True,
    symlink_official_ports: bool = True,
) -> dict[str, Any]:
    """把旧 p{port} 记录迁到 t{id}_{name}，官方口再链回当前任务桶。"""
    from behavior_interface.challenge_tasks import challenge_task_name

    root = os.path.abspath(base or default_monitor_root())
    official = {15060 + index: index for index in range(10)}
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for run_dir, dir_name, parent_name, payload in _iter_existing_session_dirs(root):
        if parse_task_bucket(parent_name)[0] is not None:
            skipped.append({"run_dir": run_dir, "reason": "already_task_bucket"})
            continue
        task_id, task_name = infer_monitor_task(
            payload,
            session_id=str(payload.get("session_id") or dir_name),
            prompt=str(payload.get("user_prompt") or payload.get("prompt") or ""),
            dir_name=dir_name,
            parent_name=parent_name,
        )
        if task_id is None or not task_name:
            unknown.append({"run_dir": run_dir, "session_id": payload.get("session_id")})
            continue
        dest_parent = os.path.join(root, task_bucket_name(task_id, task_name))
        dest = os.path.join(dest_parent, dir_name)
        if os.path.abspath(run_dir) == os.path.abspath(dest):
            skipped.append({"run_dir": run_dir, "reason": "same_path"})
            continue
        record = {
            "src": run_dir,
            "dest": dest,
            "task_id": task_id,
            "task_name": task_name,
        }
        if dry_run:
            moved.append(record)
            continue
        os.makedirs(dest_parent, exist_ok=True)
        final = dest
        suffix = 2
        while os.path.exists(final):
            final = f"{dest}-from-{parent_name or 'root'}-{suffix}"
            suffix += 1
        os.rename(run_dir, final)
        payload = dict(payload)
        payload["task_id"] = task_id
        payload["task_name"] = task_name
        payload["run_dir"] = final
        _write_json_atomic(os.path.join(final, "session.json"), payload)
        record["dest"] = final
        moved.append(record)

    if symlink_official_ports:
        for port, task_id in official.items():
            name = challenge_task_name(task_id)
            if not name:
                continue
            bucket = os.path.join(root, task_bucket_name(task_id, name))
            port_dir = os.path.join(root, f"p{port}")
            os.makedirs(bucket, exist_ok=True)
            if os.path.islink(port_dir):
                current = os.path.realpath(port_dir)
                if os.path.abspath(current) == os.path.abspath(bucket):
                    continue
                if not dry_run:
                    os.unlink(port_dir)
            elif os.path.isdir(port_dir):
                leftovers = [
                    item
                    for item in os.listdir(port_dir)
                    if item not in {"current.json", "thumbs"} and not item.startswith(".")
                ]
                if leftovers:
                    parking = os.path.join(root, f"p{port}_unassigned")
                    record = {"port": port, "park": parking, "count": len(leftovers)}
                    if dry_run:
                        unknown.append(record)
                        continue
                    os.makedirs(parking, exist_ok=True)
                    for item in leftovers:
                        src = os.path.join(port_dir, item)
                        dest = os.path.join(parking, item)
                        suffix = 2
                        while os.path.exists(dest):
                            dest = f"{dest}-{suffix}"
                            suffix += 1
                        os.rename(src, dest)
                if not dry_run:
                    shutil.rmtree(port_dir, ignore_errors=True)
            if dry_run:
                links.append({"port": port, "target": bucket})
                continue
            os.symlink(bucket, port_dir)
            links.append({"port": port, "target": bucket})

    return {
        "base": root,
        "dry_run": dry_run,
        "moved": moved,
        "skipped": skipped,
        "unknown": unknown,
        "links": links,
    }
