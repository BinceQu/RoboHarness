"""Gemini-only stdio MCP filter in front of the embodied server.

The antigravity proxy returns HTTP 400 Invalid request once Claude attaches
the embodied tool catalog.  That catalog names the coordinate system
``qwen3vl_relative_0_1000``.  This process rewrites those strings on the
tools/list response only.  Qwen never starts this module.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

_REPLACEMENTS = (
    ("qwen3vl_relative_0_1000", "relative_image_0_1000"),
    ("Qwen3-VL relative image coordinates 0..1000", "relative image coordinates 0..1000"),
    ("Qwen3-VL relative coordinate 0..1000", "relative image coordinate 0..1000"),
    ("Qwen3-VL", "vision"),
)


def _strip_empty_enums(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "enum" and isinstance(item, list):
                item = [entry for entry in item if entry != ""]
            cleaned[key] = _strip_empty_enums(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_empty_enums(item) for item in value]
    return value


def _rewrite(text: str) -> str:
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    # Gemini rejects an enum that contains an empty string.  exec_plan_pose
    # advertises arm=["", "left", "right"].  Drop only that empty entry.
    if '"enum"' in text and text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        return json.dumps(_strip_empty_enums(payload), ensure_ascii=False, separators=(",", ":"))
    return text


def _rewrite_payload(payload: bytes) -> bytes:
    if b"tools/list" not in payload and b"qwen" not in payload and b"Qwen" not in payload:
        return payload
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    rewritten = _rewrite(text)
    if rewritten == text:
        return payload
    return rewritten.encode("utf-8")


def _pump(src, dst, rewrite: bool) -> None:
    # The embodied MCP server speaks newline-delimited JSON, not Content-Length.
    while True:
        line = src.readline()
        if not line:
            return
        if rewrite:
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                dst.write(line)
                dst.flush()
                continue
            rewritten = _rewrite(text)
            line = rewritten.encode("utf-8")
            if not line.endswith(b"\n"):
                line += b"\n"
        dst.write(line)
        dst.flush()


def main() -> int:
    command = sys.argv[1:]
    if not command:
        sys.stderr.write("usage: python -m embodied_claude_code.gemini_mcp_wrap COMMAND...\n")
        return 2
    env = os.environ.copy()
    err_path = os.environ.get("GEMINI_MCP_WRAP_LOG") or "/tmp/gemini_mcp_wrap.log"
    err_file = open(err_path, "ab", buffering=0)
    child = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err_file,
        env=env,
    )
    assert child.stdin is not None and child.stdout is not None
    to_child = threading.Thread(
        target=_pump, args=(sys.stdin.buffer, child.stdin, False), daemon=True
    )
    from_child = threading.Thread(
        target=_pump, args=(child.stdout, sys.stdout.buffer, True), daemon=True
    )
    to_child.start()
    from_child.start()
    try:
        return int(child.wait())
    except KeyboardInterrupt:
        child.terminate()
        return int(child.wait())


if __name__ == "__main__":
    raise SystemExit(main())
