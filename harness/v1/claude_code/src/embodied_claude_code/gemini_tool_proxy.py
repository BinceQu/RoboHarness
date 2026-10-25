"""Gemini-only Anthropic proxy.

Gemini emits a real tool_use for a short tool name and writes an XML tag when
the name contains ``mcp__``.  This proxy shortens ``mcp__behavior-v2__`` and
``mcp__plugin_embodied-claude-code_behavior-v2__`` on the way out, then puts
the prefix back on tool_use blocks in the response.  Qwen never uses it.
"""
from __future__ import annotations

import argparse
import http.client
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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
_PREFIXES = (
    "mcp__plugin_embodied-claude-code_behavior-v2__",
    "mcp__behavior-v2__",
)


def shorten_name(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def restore_name(name: str) -> str:
    if name.startswith("mcp__"):
        return name
    return "mcp__behavior-v2__" + name


def _rewrite_tools(payload: dict[str, Any]) -> set[str]:
    shortened: set[str] = set()
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return shortened
    seen: set[str] = set()
    kept: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            kept.append(tool)
            continue
        name = tool.get("name")
        if isinstance(name, str):
            short = shorten_name(name)
            if short in seen:
                continue
            seen.add(short)
            if short != name:
                shortened.add(short)
            tool = dict(tool)
            tool["name"] = short
        kept.append(tool)
    payload["tools"] = kept
    return shortened


def _accept_reasoning_effort(payload: dict[str, Any], upstream: str) -> None:
    """The 27B vLLM rejects Claude's default effort ``high``.

    Its Messages endpoint accepts ``xhigh``, ``medium``, and ``low``.
    Flash on the other port accepts ``high``, so this rewrite stays on
    the 27B origin.
    """
    if ":30000" not in upstream and not str(payload.get("model") or "").endswith("27B-FP8"):
        return
    config = payload.get("output_config")
    if isinstance(config, dict) and config.get("effort") == "high":
        config["effort"] = "xhigh"


def _rewrite_response(payload: dict[str, Any], shortened: set[str]) -> None:
    content = payload.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if isinstance(name, str) and name in shortened:
            block["name"] = restore_name(name)


def _write_ready(path: Path, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(port) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], upstream: str, timeout_s: float) -> None:
        self.upstream = upstream.rstrip("/")
        self.timeout_s = timeout_s
        parsed = urlparse(self.upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("upstream must be an absolute http(s) URL")
        self.upstream_scheme = parsed.scheme
        self.upstream_host = parsed.hostname or ""
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        super().__init__(address, ProxyHandler)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GeminiToolProxy/1"

    @property
    def proxy(self) -> ProxyServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        return self.rfile.read(length) if length else b""

    def _upstream(self, method: str, body: bytes | None) -> http.client.HTTPResponse:
        if self.proxy.upstream_scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self.proxy.upstream_host, self.proxy.upstream_port, timeout=self.proxy.timeout_s
            )
        else:
            conn = http.client.HTTPConnection(
                self.proxy.upstream_host, self.proxy.upstream_port, timeout=self.proxy.timeout_s
            )
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP:
                continue
            headers[key] = value
        headers["Host"] = (
            f"{self.proxy.upstream_host}:{self.proxy.upstream_port}"
            if self.proxy.upstream_port not in {80, 443}
            else self.proxy.upstream_host
        )
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn.request(method, self.path, body=body, headers=headers)
        return conn.getresponse()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"", "/health"}:
            self._send_json(HTTPStatus.OK, {"ok": True, "upstream": self.proxy.upstream})
            return
        try:
            response = self._upstream("GET", None)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._relay(response)

    def do_POST(self) -> None:  # noqa: N802
        try:
            raw = self._read_body()
            path = self.path.split("?", 1)[0].rstrip("/")
            shortened: set[str] = set()
            if path == "/v1/messages" and raw:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    shortened = _rewrite_tools(payload)
                    _accept_reasoning_effort(payload, self.proxy.upstream)
                    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            response = self._upstream("POST", raw)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        if path == "/v1/messages" and response.status == 200:
            body = response.read()
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                _rewrite_response(payload, shortened)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        self._relay(response)

    def _relay(self, response: http.client.HTTPResponse) -> None:
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shorten Gemini MCP tool names.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--upstream", default=os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip())
    args = parser.parse_args(argv)
    if not args.upstream:
        raise SystemExit("gemini-tool-proxy: --upstream is required")
    timeout = float(os.environ.get("QWEN_HTTP_TIMEOUT_S", "1900"))
    server = ProxyServer((args.host, args.port), upstream=args.upstream, timeout_s=timeout)
    host, port = server.server_address[:2]
    if args.ready_file:
        _write_ready(args.ready_file, int(port))
    print(
        f"[gemini-tool-proxy] listening on http://{host}:{port}; upstream={args.upstream}",
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
