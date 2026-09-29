"""把首轮 prompt 附图注入 Anthropic /v1/messages，再原样转到上游。

Claude 直连 31000 时不会走 qwen_bridge，--prompt-image 必须在这里喂进模型。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
from typing import Any
from urllib.parse import urlparse


MAX_REQUEST_BYTES = 64 * 1024 * 1024
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
    "expect",
}


def image_blocks(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in paths:
        data = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
    return blocks


def inject_prompt_images(
    payload: dict[str, Any], paths: tuple[Path, ...]
) -> tuple[dict[str, Any], bool]:
    """仅在还没有 assistant 消息时，把附图插到第一条 user content 前面。"""
    if not paths:
        return payload, False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, False
    if any(
        isinstance(item, dict) and item.get("role") == "assistant" for item in messages
    ):
        return payload, False
    blocks = image_blocks(paths)
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = []
        message["content"] = blocks + content
        return payload, True
    return payload, False


def _paths_from_environment() -> tuple[Path, ...]:
    raw = os.environ.get("QWEN_PROMPT_IMAGES_JSON", "").strip()
    if not raw:
        return ()
    values = json.loads(raw)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("QWEN_PROMPT_IMAGES_JSON must be a string array")
    return tuple(Path(item).resolve() for item in values)


def _write_ready(path: Path, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(port) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _log_injection(
    payload: dict[str, Any], paths: tuple[Path, ...], injected: bool
) -> None:
    if getattr(_log_injection, "_logged", False):
        return
    _log_injection._logged = True
    images: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            source = item.get("source") or {}
            raw = b""
            data = source.get("data")
            if isinstance(data, str) and data:
                try:
                    raw = base64.b64decode(data, validate=False)
                except Exception:
                    raw = b""
            images.append(
                {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
                    "media_type": str(source.get("media_type") or ""),
                }
            )
        if images:
            break
    configured: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_bytes() if path.is_file() else b""
        configured.append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
            }
        )
    record = {
        "injected_flag": injected,
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
    sys.stderr.write(
        "anthropic-inject: prompt-image inject="
        f"{injected} configured={len(paths)} upstream_images={len(images)}\n"
    )
    sys.stderr.flush()
    dump = os.environ.get("EMBODIED_PROMPT_IMAGE_INJECT_LOG", "").strip()
    if dump:
        Path(dump).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class InjectServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        upstream: str,
        timeout_s: float,
        prompt_images: tuple[Path, ...],
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.timeout_s = timeout_s
        self.prompt_images = prompt_images
        parsed = urlparse(self.upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("upstream must be an absolute http(s) URL")
        self.upstream_scheme = parsed.scheme
        self.upstream_host = parsed.hostname or ""
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        super().__init__(address, InjectHandler)


class InjectHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EmbodiedAnthropicInject/1"

    @property
    def inject(self) -> InjectServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[anthropic-inject] " + (fmt % args), file=sys.stderr, flush=True)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        return self.rfile.read(length) if length else b""

    def _forward(self, method: str, body: bytes | None) -> None:
        if self.inject.upstream_scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self.inject.upstream_host,
                self.inject.upstream_port,
                timeout=self.inject.timeout_s,
            )
        else:
            conn = http.client.HTTPConnection(
                self.inject.upstream_host,
                self.inject.upstream_port,
                timeout=self.inject.timeout_s,
            )
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP:
                continue
            headers[key] = value
        headers["Host"] = (
            f"{self.inject.upstream_host}:{self.inject.upstream_port}"
            if self.inject.upstream_port not in {80, 443}
            else self.inject.upstream_host
        )
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP:
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()

    def do_HEAD(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"", "/health", "/api/hello", "/v1/models"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        try:
            self._forward("HEAD", None)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"", "/health"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "upstream": self.inject.upstream,
                    "prompt_images": [str(path) for path in self.inject.prompt_images],
                },
            )
            return
        try:
            self._forward("GET", None)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            raw = self._read_body()
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/messages":
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body must be object"})
                    return
                payload, injected = inject_prompt_images(
                    payload, self.inject.prompt_images
                )
                _log_injection(payload, self.inject.prompt_images, injected)
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._forward("POST", raw)
        except Exception as exc:
            self.log_message("forward failed: %s", exc)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inject first-turn prompt images into Anthropic Messages."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument(
        "--upstream",
        default=os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip(),
    )
    args = parser.parse_args(argv)
    if not args.upstream:
        raise SystemExit("anthropic-inject: --upstream is required")
    timeout = float(os.environ.get("QWEN_HTTP_TIMEOUT_S", "1900"))
    server = InjectServer(
        (args.host, args.port),
        upstream=args.upstream,
        timeout_s=timeout,
        prompt_images=_paths_from_environment(),
    )
    host, port = server.server_address[:2]
    if args.ready_file:
        _write_ready(args.ready_file, int(port))
    print(
        f"[anthropic-inject] listening on http://{host}:{port}; upstream={args.upstream}",
        file=sys.stderr,
        flush=True,
    )

    def _stop(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
