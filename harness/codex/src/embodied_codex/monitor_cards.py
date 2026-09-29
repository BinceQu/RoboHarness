"""把模型实际看到的图回写到 Agent Monitor 卡片。"""
from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def resolve_base_url(explicit: str = "") -> str:
    for key in ("BEHAVIOR_BASE_URL", "BEHAVIOR_INTERFACE_URL"):
        text = str(explicit or os.environ.get(key) or "").strip().rstrip("/")
        if text:
            return text
        explicit = ""
    return ""


def _image_id_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    image_id = str(response.get("image_id") or "").strip()
    if image_id:
        return image_id
    observation = response.get("observation")
    if isinstance(observation, dict):
        return str(observation.get("image_id") or "").strip()
    return ""


def _atomic_write(path: str, data: bytes) -> bool:
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except OSError:
        return False


def _write_monitor_files(
    snapshot: dict[str, Any],
    *,
    image_id: str,
    data: bytes,
    tool: str,
) -> bool:
    """接口还没新路由时，直接覆盖直播缩略图，让卡片立刻换成模型图。"""
    run_dir = str(snapshot.get("run_dir") or "").strip()
    if not run_dir or not os.path.isdir(run_dir):
        return False
    thumbs = os.path.join(run_dir, "thumbs")
    wrote = False
    if image_id:
        wrote = _atomic_write(os.path.join(thumbs, f"model_{image_id}.jpg"), data) or wrote
    cards = snapshot.get("cards") if isinstance(snapshot.get("cards"), list) else []
    target = None
    for card in reversed(cards):
        if not isinstance(card, dict):
            continue
        if image_id and str(card.get("output_image_id") or "") == image_id:
            target = card
            break
        if tool and str(card.get("tool") or "") == tool and card.get("has_output"):
            target = card
            break
    if target is None:
        for card in reversed(cards):
            if isinstance(card, dict) and card.get("has_output"):
                target = card
                break
    if target is None:
        return wrote
    try:
        card_id = int(target.get("card_id") or 0)
    except (TypeError, ValueError):
        card_id = 0
    if card_id <= 0:
        return wrote
    wrote = _atomic_write(
        os.path.join(thumbs, f"card_{card_id:04d}_out.jpg"), data
    ) or wrote
    return wrote


def publish_model_card_image(
    *,
    data: bytes,
    image_id: str = "",
    tool: str = "",
    session_id: str = "",
    base_url: str = "",
    opener=urlopen,
) -> bool:
    """把 MCP 发给模型的那张图打到监视器。失败不影响工具结果。"""
    payload = data if isinstance(data, (bytes, bytearray)) else b""
    if not payload:
        return False
    origin = resolve_base_url(base_url)
    sid = str(session_id or os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
    iid = str(image_id or "").strip()
    name = str(tool or "").strip()
    if origin:
        body: dict[str, Any] = {
            "image_b64": base64.b64encode(bytes(payload)).decode("ascii"),
            "image_id": iid,
            "tool": name,
        }
        if sid:
            body["session_id"] = sid
        request = Request(
            f"{origin}/api/agent_monitor/model_image",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener(request, timeout=3.0) as response:
                if 200 <= int(getattr(response, "status", 200)) < 300:
                    return True
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        try:
            with opener(
                Request(f"{origin}/api/agent_monitor", method="GET"),
                timeout=3.0,
            ) as response:
                snapshot = json.loads(response.read().decode("utf-8"))
            if isinstance(snapshot, dict):
                return _write_monitor_files(
                    snapshot, image_id=iid, data=bytes(payload), tool=name
                )
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return False
    return False


def publish_from_tool_result(
    result: Any,
    *,
    session_id: str = "",
    base_url: str = "",
    opener=urlopen,
) -> bool:
    media = getattr(result, "media", None) or []
    if not media:
        return False
    first = media[0]
    data = getattr(first, "data", b"") or b""
    payload = result.data if isinstance(getattr(result, "data", None), dict) else {}
    response = payload.get("response")
    return publish_model_card_image(
        data=data,
        image_id=_image_id_from_response(response),
        tool=str(payload.get("tool_name") or "").strip(),
        session_id=session_id,
        base_url=base_url,
        opener=opener,
    )
