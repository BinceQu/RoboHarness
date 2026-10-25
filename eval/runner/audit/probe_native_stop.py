"""Offline Claude Code Stop-hook compatibility probe; no robot or model service.

A local deterministic Messages fixture returns a plan-only end_turn. The
unmodified CLI should stop without a hook, and execute one harmless Bash call
after native Stop feedback. This is evaluator-side test code, never installed
in a robot workspace or used to influence an evaluation episode.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from urllib.parse import urlsplit


SENTINEL = "OFFLINE_STOP_PROBE_FEEDBACK"
TOKEN = "offline-stop-probe-tool-executed"
MODEL = "Qwen3.8-Flash-Next-FP8"
HOOK = '''import json
import pathlib
import sys

mode, record = sys.argv[1:]
data = json.load(sys.stdin)
path = pathlib.Path(record)
first = not path.exists()
with path.open("a") as stream:
    stream.write(json.dumps({"event": data.get("hook_event_name"),
        "stop_hook_active": data.get("stop_hook_active"),
        "last_assistant_message": data.get("last_assistant_message")}) + "\\n")
if first and not data.get("stop_hook_active"):
    text = "OFFLINE_STOP_PROBE_FEEDBACK: Execute the pending harmless Bash step before ending this offline probe."
    output = ({"decision": "block", "reason": text} if mode == "block" else
        {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": text}})
    print(json.dumps(output))
'''


def successful_probe_result(block):
    if (not isinstance(block, dict) or block.get("type") != "tool_result"
            or block.get("tool_use_id") != "toolu_offline_stop_probe"
            or block.get("is_error", False)):
        return False
    content = block.get("content", "")
    if isinstance(content, list):
        content = "\n".join(item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text")
    return isinstance(content, str) and content.strip() == TOKEN


class Fixture(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, feedback_text=SENTINEL):
        super().__init__(("127.0.0.1", 0), Handler)
        self.feedback_text = feedback_text
        self.requests = []
        self.tool_result_seen = False
        self.feedback_seen = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        route = urlsplit(self.path).path
        if route.endswith("/count_tokens"):
            self.json_response({"input_tokens": 100})
            return
        if not route.endswith("/messages"):
            self.send_error(404)
            return
        messages = body.get("messages", [])
        serialized = json.dumps(messages)
        feedback = self.server.feedback_text in serialized
        tool_result = any(
            isinstance(message.get("content"), list)
            and any(successful_probe_result(block) for block in message["content"])
            for message in messages
        )
        self.server.feedback_seen |= feedback
        self.server.tool_result_seen |= tool_result
        self.server.requests.append({"feedback_seen": feedback,
            "tool_result_seen": tool_result, "stream": body.get("stream", False)})
        if tool_result or len(self.server.requests) > 5:
            content = [{"type": "text", "text": "Offline probe complete."}]
            reason = "end_turn"
        elif feedback:
            content = [{"type": "tool_use", "id": "toolu_offline_stop_probe", "name": "Bash",
                "input": {"command": "printf 'offline-stop-probe-tool-executed\\n'",
                          "description": "Complete the harmless offline probe"}}]
            reason = "tool_use"
        else:
            content = [{"type": "text", "text": "Next I will execute the pending step."}]
            reason = "end_turn"
        message = {"id": f"msg_probe_{len(self.server.requests)}", "type": "message",
            "role": "assistant", "model": MODEL, "content": content, "stop_reason": reason,
            "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}}
        if not body.get("stream"):
            self.json_response(message)
            return
        events = [("message_start", {"type": "message_start", "message": {
            **message, "content": [], "stop_reason": None,
            "usage": {"input_tokens": 100, "output_tokens": 0}}})]
        block = content[0]
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        events.extend([
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {
                "stop_reason": reason, "stop_sequence": None}, "usage": {"output_tokens": 20}}),
            ("message_stop", {"type": "message_stop"}),
        ])
        payload = "".join(f"event: {name}\ndata: {json.dumps(event)}\n\n" for name, event in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def json_response(self, value):
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_case(binary, root, mode):
    case = root / mode
    case.mkdir()
    script = case / "fixture_hook.py"
    script.write_text(HOOK)
    settings = {"hooks": {}}
    feedback_text = SENTINEL
    if mode == "workspace":
        from tools.util.stop_check import FEEDBACK
        from tools.util.files import atomic_json
        feedback_text = FEEDBACK
        template = Path(__file__).resolve().parents[3] / "harness/v2/R1_pro"
        # Exercise the actual module and project hook registration using only
        # a local fixture state; no evaluator, physics, or production session.
        for relative in ("tools/__init__.py", "tools/util/__init__.py",
                         "tools/util/settings.py", "tools/util/files.py", "tools/util/stop_check.py"):
            target = case / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template / relative, target)
        settings["hooks"]["Stop"] = json.loads((template / ".claude/settings.json").read_text())["hooks"]["Stop"]
        state_dir = case / ".runtime/state"
        atomic_json(case / ".session.json", {"workspace": str(case), "state_dir": str(state_dir), "session_id": "stop-probe"})
        atomic_json(state_dir / "state.json", {"session_id": "stop-probe", "counter": 0})
    elif mode != "none":
        command = shlex.join([sys.executable, str(script), mode, str(case / "hook-inputs.jsonl")])
        settings["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command": command, "timeout": 10}]}]
    settings_path = case / "settings.json"
    settings_path.write_text(json.dumps(settings))
    server = Fixture(feedback_text)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ANTHROPIC_", "CLAUDE_", "GIT_"))
           and key not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PYTHONPATH", "PYTHONHOME", "ROBOT_WORKSPACE"}}
    env.update(ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
        ANTHROPIC_API_KEY="offline-test-only", ANTHROPIC_AUTH_TOKEN="offline-test-only",
        CLAUDE_CONFIG_DIR=str(case / "claude-home"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", CUDA_VISIBLE_DEVICES="",
        ENABLE_TOOL_SEARCH="false")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    command = [binary, "--model", MODEL, "--permission-mode", "dontAsk", "--tools", "Bash",
        "--allowedTools", "Bash", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "", "--settings", str(settings_path), "-p", "--output-format", "json"]
    try:
        with (case / "agent.json").open("w") as stdout, (case / "agent.stderr.log").open("w") as stderr:
            process = subprocess.run(command, cwd=case, env=env, input="Complete the pending harmless offline probe step.",
                text=True, stdout=stdout, stderr=stderr, timeout=90)
        final = json.loads((case / "agent.json").read_text())
        passed = (process.returncode == 0 and final.get("is_error") is False
            and server.tool_result_seen == (mode != "none")
            and server.feedback_seen == (mode != "none")
            and len(server.requests) == (1 if mode == "none" else 3)
            and final.get("result") == ("Next I will execute the pending step."
                                       if mode == "none" else "Offline probe complete."))
        record = {"mode": mode, "passed": passed, "returncode": process.returncode,
            "requests": server.requests, "result": final.get("result"), "is_error": final.get("is_error"),
            "feedback_seen": server.feedback_seen, "tool_result_seen": server.tool_result_seen}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    (case / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for mode in ("none", "feedback", "block", "workspace"):
        record = run_case(args.claude, root, mode)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"passed": all(row["passed"] for row in records), "cases": records,
        "claude_version": subprocess.check_output([args.claude, "--version"], text=True).strip(),
        "scope": "Native CLI lifecycle only, including the actual workspace Stop module and registration, with deterministic loopback responses. No inference service, evaluator, GPU, or production session was used.",
        "limitation": "Does not prove that Qwen will obey feedback or complete a physical task.",
        "documentation": "https://code.claude.com/docs/en/hooks#stop-decision-control"}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
