"""Local Anthropic Messages to OpenAI Chat Completions protocol bridge.

Claude Code speaks the Anthropic Messages API. The Qwen endpoint used by this
plugin speaks the OpenAI-compatible Chat Completions API. This module translates
only that protocol boundary; it does not proxy or modify BEHAVIOR traffic.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from .coordinates import (
    LATEST_IMAGE_REMINDER_PREFIX,
    VLM_IMAGE_COORDINATE_CONTRACT,
    VLM_IMAGE_COORDINATE_REMINDER,
)


DEFAULT_UPSTREAM = "http://127.0.0.1:31000/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.8-27B"
MAX_REQUEST_BYTES = 64 * 1024 * 1024

# A tool result can contain a navigation/path overlay that is useful for a
# chassis decision but is not the object being picked. Keep this instruction
# beside the new image so Qwen resolves the current task target before choosing
# a coordinate or an action.
VISUAL_GROUNDING_GATE = (
    "Visual grounding gate: re-read the currently attached image and ground only "
    "the physical object requested by the current task or active Skill. During a "
    "pickup stage, click the requested object's visible graspable body (for "
    "example, the soda can), never its destination container, floor, furniture, "
    "navigation path, map, or overlay text. Use this image_id only; do not copy "
    "coordinates from an older image."
)

class BridgeError(RuntimeError):
    def __init__(self, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BridgeConfig:
    upstream_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 1900.0
    prompt_images: tuple[Path, ...] = ()


def normalize_upstream_url(value: str) -> str:
    url = str(value or DEFAULT_UPSTREAM).strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Qwen upstream must be an absolute http(s) URL.")
    if parsed.path.endswith("/chat/completions"):
        return url
    if parsed.path.endswith("/v1"):
        return url + "/chat/completions"
    if not parsed.path or parsed.path == "/":
        return url + "/v1/chat/completions"
    raise ValueError(
        "Qwen upstream must end in /v1 or /v1/chat/completions."
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _system_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n\n".join(
        str(block.get("text") or "")
        for block in value
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type") or "")
    if source_type == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        if not data:
            return None
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = str(source.get("url") or "")
        if not url:
            return None
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _openai_user_content(blocks: list[Any]) -> str | list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for raw in blocks:
        if isinstance(raw, str):
            if raw:
                content.append({"type": "text", "text": raw})
            continue
        if not isinstance(raw, dict):
            continue
        block_type = str(raw.get("type") or "")
        if block_type == "text":
            content.append({"type": "text", "text": str(raw.get("text") or "")})
        elif block_type == "image":
            image = _anthropic_image_to_openai(raw)
            if image is not None:
                content.append(image)
        elif block_type == "image_url" and isinstance(raw.get("image_url"), dict):
            content.append(raw)
    if not content:
        return ""
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0].get("text") or "")
    return content


def _tool_result_parts(
    block: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    value = block.get("content")
    if isinstance(value, str):
        return value, [], []
    if not isinstance(value, list):
        if value is None:
            return "", [], []
        return json.dumps(value, ensure_ascii=False), [], []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    visual_reminders: list[str] = []
    for item in value:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text") or "")
            if text.startswith(LATEST_IMAGE_REMINDER_PREFIX):
                visual_reminders.append(text)
            else:
                texts.append(text)
        elif isinstance(item, dict) and item.get("type") == "image":
            image = _anthropic_image_to_openai(item)
            if image is not None:
                images.append(image)
    if not images:
        texts.extend(visual_reminders)
        visual_reminders = []
    return "\n".join(texts), images, visual_reminders


def _translate_user_message(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    blocks = content if isinstance(content, list) else []
    tool_messages: list[dict[str, Any]] = []
    ordinary: list[Any] = []
    visual_results: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            ordinary.append(block)
            continue
        call_id = str(block.get("tool_use_id") or "")
        text, images, visual_reminders = _tool_result_parts(block)
        if block.get("is_error"):
            text = "Tool error:\n" + text
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": text or "(tool returned no text)",
            }
        )
        if images:
            # Qwen's grounding reference sends the image before its grounding
            # instruction. Keep the tool metadata in the preceding tool message.
            visual_results.extend(images)
            visual_results.append(
                {
                    "type": "text",
                    "text": (
                        f"Visual output from tool call {call_id}. "
                        f"{VLM_IMAGE_COORDINATE_REMINDER}"
                    ),
                }
            )
            visual_results.extend(
                {"type": "text", "text": reminder}
                for reminder in visual_reminders or [VISUAL_GROUNDING_GATE]
            )
    translated = list(tool_messages)
    user_content = _openai_user_content(ordinary + visual_results)
    if user_content != "":
        translated.append({"role": "user", "content": user_content})
    return translated


def _translate_assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "text":
            texts.append(str(block.get("text") or ""))
        elif block_type == "tool_use":
            arguments = block.get("input")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(texts) if texts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _prompt_image_blocks(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in paths:
        data = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{media_type};base64,"
                        + base64.b64encode(data).decode("ascii")
                    )
                },
            }
        )
    return blocks


def _log_prompt_image_injection(
    messages: list[dict[str, Any]],
    config: BridgeConfig,
    injected_images: bool,
) -> None:
    """把首轮是否把 prompt 附图送进 Qwen 请求记下来，便于核对。"""
    if getattr(_log_prompt_image_injection, "_logged", False):
        return
    _log_prompt_image_injection._logged = True
    dump = os.environ.get("EMBODIED_PROMPT_IMAGE_INJECT_LOG", "").strip()
    images: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = str((item.get("image_url") or {}).get("url") or "")
            raw = b""
            if "base64," in url:
                try:
                    raw = base64.b64decode(url.split("base64,", 1)[1], validate=False)
                except Exception:
                    raw = b""
            images.append(
                {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
                    "media_prefix": url.split(";", 1)[0][:80],
                }
            )
        if images:
            break
    configured: list[dict[str, Any]] = []
    for path in config.prompt_images:
        raw = path.read_bytes() if path.is_file() else b""
        configured.append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
            }
        )
    payload = {
        "injected_flag": injected_images,
        "configured_paths": [item["path"] for item in configured],
        "configured_files": configured,
        "upstream_user_images": images,
        "upstream_user_image_count": len(images),
        "sha256_match": bool(
            images
            and configured
            and images[0].get("sha256")
            and images[0]["sha256"] == configured[0]["sha256"]
        ),
    }
    line = (
        "qwen-bridge: prompt-image inject="
        f"{injected_images} configured={len(config.prompt_images)} "
        f"upstream_images={len(images)}\n"
    )
    sys.stderr.write(line)
    sys.stderr.flush()
    if dump:
        Path(dump).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") == "image_url"
        for item in content
    )


def _retain_latest_visual_observation(messages: list[dict[str, Any]]) -> None:
    """Remove stale image bytes while preserving the newest visual observation.

    Robot poses and camera frames change after nearly every tool call. Keeping all
    prior images gives the VLM several equally shaped coordinate canvases and can
    make a current image_id inherit a point from an older frame. Textual tool
    evidence remains in the conversation; only superseded image blocks and their
    now-invalid attachment reminders are removed.
    """
    image_message_indexes = [
        index for index, message in enumerate(messages) if _message_has_image(message)
    ]
    if len(image_message_indexes) <= 1:
        return
    latest_index = image_message_indexes[-1]
    remove_messages: set[int] = set()
    for index in image_message_indexes[:-1]:
        content = messages[index].get("content")
        if not isinstance(content, list):
            continue
        filtered: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                continue
            text = str(item.get("text") or "")
            if (
                item.get("type") == "text"
                and (
                    text.startswith("Visual output from tool call ")
                    or text.startswith(LATEST_IMAGE_REMINDER_PREFIX)
                    or text == VISUAL_GROUNDING_GATE
                )
            ):
                continue
            filtered.append(item)
        if filtered:
            messages[index]["content"] = filtered
        else:
            remove_messages.add(index)
    if remove_messages:
        messages[:] = [
            message
            for index, message in enumerate(messages)
            if index not in remove_messages or index == latest_index
        ]


def _current_turn_has_image(messages: list[dict[str, Any]]) -> bool:
    """Return whether new visual input arrived after the latest assistant turn."""
    latest_assistant = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ),
        default=-1,
    )
    return any(_message_has_image(message) for message in messages[latest_assistant + 1 :])


def translate_messages_request(
    payload: dict[str, Any], config: BridgeConfig
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = _system_text(payload.get("system"))
    if VLM_IMAGE_COORDINATE_CONTRACT not in system:
        system = "\n\n".join(
            part for part in (system, VLM_IMAGE_COORDINATE_CONTRACT) if part
        )
    messages.append({"role": "system", "content": system})

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise BridgeError("messages must be an array", status=400)
    has_assistant = any(
        isinstance(item, dict) and item.get("role") == "assistant"
        for item in raw_messages
    )
    injected_images = False
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "")
        if role == "assistant":
            messages.append(_translate_assistant_message(raw.get("content")))
        elif role == "user":
            translated = _translate_user_message(raw.get("content"))
            if config.prompt_images and not has_assistant and not injected_images:
                for message in translated:
                    if message.get("role") != "user":
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    elif not isinstance(content, list):
                        content = []
                    message["content"] = _prompt_image_blocks(
                        config.prompt_images
                    ) + content
                    injected_images = True
                    break
            messages.extend(translated)

    _retain_latest_visual_observation(messages)
    _log_prompt_image_injection(messages, config, injected_images)

    tools: list[dict[str, Any]] = []
    for raw in payload.get("tools") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        schema = raw.get("input_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(raw["name"]),
                    "description": str(raw.get("description") or ""),
                    "parameters": schema,
                },
            }
        )

    request: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max(1, int(payload.get("max_tokens") or 1024)),
        # Claude Code expects an SSE response, but buffering one upstream turn
        # gives deterministic conversion for both text and tool calls.
        "stream": False,
    }
    if tools:
        request["tools"] = tools
    current_turn_has_image = _current_turn_has_image(messages)
    if current_turn_has_image:
        # Qwen's official grounding implementation uses non-thinking mode for
        # perception. Limit it to the turn that receives a new image so ordinary
        # Claude Code planning turns retain Qwen's reasoning mode.
        request["chat_template_kwargs"] = {"enable_thinking": False}
        # Coordinate selection is not a creative task. Server-default sampling
        # makes identical images produce materially different tool arguments.
        request["temperature"] = 0.0
        request["top_p"] = 1.0
    # Claude Code owns tool selection, including text-only compaction turns.
    # Translate its explicit choice; omitted choices must remain omitted.
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "auto")
        if choice_type == "any":
            request["tool_choice"] = "required"
        elif choice_type == "tool" and tool_choice.get("name"):
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": str(tool_choice["name"])},
            }
        elif choice_type in {"auto", "none"}:
            request["tool_choice"] = choice_type
    if not current_turn_has_image:
        for name in ("temperature", "top_p"):
            if payload.get(name) is not None:
                request[name] = payload[name]
    stop = payload.get("stop_sequences")
    if isinstance(stop, list) and stop:
        request["stop"] = stop
    return request


def call_upstream(request_payload: dict[str, Any], config: BridgeConfig) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        # This gateway's edge rejects urllib's default Python user agent
        # with Cloudflare 1010 before the request reaches the model.
        "User-Agent": "Mozilla/5.0",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = Request(
        config.upstream_url,
        data=_json_bytes(request_payload),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_s) as response:
            raw = response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:4000]
        raise BridgeError(
            f"Qwen upstream HTTP {exc.code}: {details}",
            status=exc.code if 400 <= exc.code < 500 else 502,
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise BridgeError(f"Qwen upstream unavailable: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BridgeError("Qwen upstream returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeError("Qwen upstream returned a non-object response")
    if payload.get("error"):
        raise BridgeError(f"Qwen upstream error: {payload['error']}")
    return payload


def _tool_input(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        return value, json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raw = str(value or "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw_arguments": raw}, json.dumps(
            {"_raw_arguments": raw}, ensure_ascii=False, separators=(",", ":")
        )
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return parsed, json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def translate_chat_response(
    payload: dict[str, Any], model: str
) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise BridgeError("Qwen upstream response has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise BridgeError("Qwen upstream choice has no message")
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(
            str(item.get("text") or "")
            for item in text
            if isinstance(item, dict)
        )
    if not isinstance(text, str) or not text:
        reasoning = message.get("reasoning_content")
        text = reasoning if isinstance(reasoning, str) else ""
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    function_call = message.get("function_call")
    if not tool_calls and isinstance(function_call, dict):
        tool_calls = [{"type": "function", "function": function_call}]
    for raw in tool_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        parsed, partial_json = _tool_input(function.get("arguments"))
        content.append(
            {
                "type": "tool_use",
                "id": str(raw.get("id") or f"toolu_{uuid.uuid4().hex}"),
                "name": str(function["name"]),
                "input": parsed,
                "partial_json": partial_json,
            }
        )

    finish_reason = str(choice.get("finish_reason") or "stop")
    if any(block.get("type") == "tool_use" for block in content):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    usage_raw = payload.get("usage")
    usage_raw = usage_raw if isinstance(usage_raw, dict) else {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens") or 0),
        "output_tokens": int(usage_raw.get("completion_tokens") or 0),
    }
    if not content:
        content.append({"type": "text", "text": ""})
    return content, stop_reason, usage


def anthropic_response(payload: dict[str, Any], config: BridgeConfig) -> dict[str, Any]:
    request_payload = translate_messages_request(payload, config)
    upstream = call_upstream(request_payload, config)
    content, stop_reason, usage = translate_chat_response(upstream, config.model)
    for block in content:
        block.pop("partial_json", None)
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": config.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _estimate_tokens(payload: dict[str, Any]) -> int:
    text = json.dumps(
        {
            "system": payload.get("system"),
            "messages": payload.get("messages"),
            "tools": payload.get("tools"),
        },
        ensure_ascii=False,
    )
    # This endpoint is used for context budgeting, not billing. Bias upward so
    # Claude Code compacts before Qwen's real context boundary.
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: BridgeConfig) -> None:
        self.config = config
        super().__init__(address, BridgeHandler)


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EmbodiedQwenBridge/1"

    @property
    def bridge(self) -> BridgeServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[qwen-bridge] " + (fmt % args), file=sys.stderr, flush=True)

    def _send_json(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise BridgeError("invalid Content-Length", status=400) from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise BridgeError("request body size is invalid", status=413)
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BridgeError("request body is not valid JSON", status=400) from exc
        if not isinstance(payload, dict):
            raise BridgeError("request body must be an object", status=400)
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health", "/v1/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "model": self.bridge.config.model,
                    "upstream": self.bridge.config.upstream_url,
                },
            )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/api/hello", "/health", "/v1/models"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/messages/count_tokens":
                self._send_json(HTTPStatus.OK, {"input_tokens": _estimate_tokens(payload)})
                return
            if path != "/v1/messages":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            request_payload = translate_messages_request(payload, self.bridge.config)
            upstream = call_upstream(request_payload, self.bridge.config)
            content, stop_reason, usage = translate_chat_response(
                upstream, self.bridge.config.model
            )
            if bool(payload.get("stream")):
                self._send_stream(content, stop_reason, usage)
            else:
                for block in content:
                    block.pop("partial_json", None)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "id": f"msg_{uuid.uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "model": self.bridge.config.model,
                        "content": content,
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                        "usage": usage,
                    },
                )
        except BridgeError as exc:
            self._send_json(
                exc.status,
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # keep the local gateway fail-closed
            self.log_error("unexpected bridge error: %s", exc)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"local protocol bridge failed: {exc}",
                    },
                },
            )

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        data = _json_bytes(payload)
        self.wfile.write(b"event: " + event.encode("ascii") + b"\n")
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _send_stream(
        self,
        content: list[dict[str, Any]],
        stop_reason: str,
        usage: dict[str, int],
    ) -> None:
        message_id = f"msg_{uuid.uuid4().hex}"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.bridge.config.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": 0,
                    },
                },
            },
        )
        for index, block in enumerate(content):
            if block.get("type") == "tool_use":
                self._event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    },
                )
                self._event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": block.get("partial_json") or "{}",
                        },
                    },
                )
            else:
                self._event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                self._event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": str(block.get("text") or ""),
                        },
                    },
                )
            self._event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        self._event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
        self._event("message_stop", {"type": "message_stop"})


def _paths_from_environment() -> tuple[Path, ...]:
    raw = os.environ.get("QWEN_PROMPT_IMAGES_JSON", "").strip()
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("QWEN_PROMPT_IMAGES_JSON is not valid JSON") from exc
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("QWEN_PROMPT_IMAGES_JSON must be a string array")
    return tuple(Path(item).resolve() for item in values)


def _write_ready(path: Path, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(port) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Translate Anthropic Messages requests to Qwen Chat Completions."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument(
        "--upstream",
        default=(
            os.environ.get("QWEN_CHAT_COMPLETIONS_URL")
            or os.environ.get("QWEN_OPENAI_URL")
            or os.environ.get("QWEN_OPENAI_BASE_URL")
            or DEFAULT_UPSTREAM
        ),
    )
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    args = parser.parse_args(argv)
    timeout = float(os.environ.get("QWEN_HTTP_TIMEOUT_S", "1900"))
    config = BridgeConfig(
        upstream_url=normalize_upstream_url(args.upstream),
        model=str(args.model),
        api_key=os.environ.get("QWEN_API_KEY", "").strip(),
        timeout_s=timeout,
        prompt_images=_paths_from_environment(),
    )
    server = BridgeServer((args.host, args.port), config)
    port = int(server.server_address[1])
    if args.ready_file:
        _write_ready(args.ready_file, port)
    print(
        f"[qwen-bridge] listening on http://{args.host}:{port}; model={config.model}",
        file=sys.stderr,
        flush=True,
    )

    def stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
