"""Real Claude CLI + real plugin/MCP/hooks, with local deterministic model replies.

No real robot or external model is contacted. Opt in with
EMBODIED_NATIVE_CLAUDE_TESTS=1. Exercise direct Anthropic and bridge transports.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from fake_behavior import FakeBehaviorClient
from embodied_claude_code.launcher import EMBODIED_TOOL_GLOBS

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen3.8-Flash-Next-FP8"
SUMMARY_REQUEST = "Your task is to create a detailed summary of the conversation so far"


@unittest.skipUnless(os.environ.get("EMBODIED_NATIVE_CLAUDE_TESTS") == "1",
                     "native CLI test requires EMBODIED_NATIVE_CLAUDE_TESTS=1")
class NativeSkillLifecycleTests(unittest.TestCase):
    def test_direct_anthropic_native_skill_deactivate_and_compaction(self):
        self.run_transport(bridge=False)

    def test_bridge_native_skill_deactivate_and_compaction(self):
        self.run_transport(bridge=True)

    def test_native_slash_skill_requires_explicit_activation(self):
        self.run_transport(bridge=False, slash=True)

    def run_transport(self, *, bridge: bool, slash: bool = False):
        claude = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        self.assertTrue(claude)
        fake = FakeBehaviorClient()
        requests, snapshots, summaries, monitor = [], [], [], []
        unexpected = []
        with tempfile.TemporaryDirectory(prefix="cc-native-lifecycle-") as directory:
            runtime = Path(directory)
            state_path = runtime / "runtime/embodied-claude-code/task_skill_state.native-lifecycle.json"

            class Endpoint(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def send_json(self, body, status=200):
                    encoded = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

                def do_GET(self):
                    if self.path in ("/api/state", "/api/v2/tools"):
                        self.send_json(fake.get_json(self.path))
                    else:
                        unexpected.append(self.path)
                        self.send_json({"error": "unexpected request"}, 404)

                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if self.path == "/api/agent_monitor/prompt":
                        monitor.append(payload)
                        self.send_json({"ok": True})
                        return
                    if not self.path.startswith("/v1/"):
                        unexpected.append(self.path)
                        self.send_json({"error": "robot actions are forbidden in this test"}, 400)
                        return
                    if "/count_tokens" in self.path:
                        self.send_json({"input_tokens": 1000})
                        return
                    last_user = next((message for message in reversed(payload.get("messages", []))
                                      if message.get("role") == "user"), {})
                    summary = SUMMARY_REQUEST in json.dumps(last_user)
                    if summary:
                        summaries.append(payload)
                        content = [{"type": "text", "text": (
                            "<summary>This is a local lifecycle test. Native pick-up-object and "
                            "close-box were loaded and explicitly deactivated. No robot motion "
                            "occurred. Current task Skill is null. List Skills, then report "
                            "NATIVE_LIFECYCLE_OK. Do not reactivate historical Skills.</summary>"
                        )}]
                    else:
                        requests.append(payload)
                        snapshots.append(json.loads(state_path.read_text()) if state_path.exists() else None)
                        step = len(requests)
                        tool_names = [tool.get("function", tool).get("name", "")
                                      for tool in payload.get("tools", [])]
                        mcp_prefix = next((name.removesuffix("deactivate_skill") for name in tool_names
                                           if name.endswith("__deactivate_skill")), "missing__")
                        plan = {
                            1: ((mcp_prefix + "activate_skill", {"name": "pick-up-object"}) if slash else
                                ("Skill", {"skill": "embodied-claude-code:pick-up-object"})),
                            2: (mcp_prefix + "deactivate_skill", {"name": "pick-up-object", "reason": "cancelled; local test"}),
                            3: ("Skill", {"skill": "embodied-claude-code:close-box"}),
                            4: (mcp_prefix + "deactivate_skill", {"name": "close-box", "reason": "cancelled; local test"}),
                            5: (mcp_prefix + "activate_skill", {}),
                        }
                        if step in plan:
                            name, arguments = plan[step]
                            content = [{"type": "tool_use", "id": f"toolu_lifecycle_{step}",
                                        "name": name, "input": arguments}]
                        else:
                            content = [{"type": "text", "text": "NATIVE_LIFECYCLE_OK"}]
                    usage_input = 195_000 if not summary and len(requests) == 4 else 1000
                    stop = "tool_use" if content[0]["type"] == "tool_use" else "end_turn"
                    if self.path.startswith("/v1/chat/completions"):
                        if stop == "tool_use":
                            block = content[0]
                            message = {"role": "assistant", "content": None, "tool_calls": [{
                                "id": block["id"], "type": "function", "function": {
                                    "name": block["name"], "arguments": json.dumps(block["input"]),
                                },
                            }]}
                        else:
                            message = {"role": "assistant", "content": content[0]["text"]}
                        self.send_json({"id": "test", "object": "chat.completion", "model": MODEL,
                                        "choices": [{"index": 0, "message": message,
                                                     "finish_reason": "tool_calls" if stop == "tool_use" else "stop"}],
                                        "usage": {"prompt_tokens": usage_input, "completion_tokens": 100}})
                        return
                    message = {"id": f"msg_{len(requests)}_{len(summaries)}", "type": "message",
                               "role": "assistant", "model": MODEL, "content": [],
                               "stop_reason": None, "stop_sequence": None,
                               "usage": {"input_tokens": usage_input, "output_tokens": 0}}
                    if not payload.get("stream"):
                        message.update(content=content, stop_reason=stop)
                        message["usage"]["output_tokens"] = 100
                        self.send_json(message)
                        return
                    events = [("message_start", {"message": message})]
                    for index, block in enumerate(content):
                        if block["type"] == "tool_use":
                            start = {**block, "input": {}}
                            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
                        else:
                            start = {"type": "text", "text": ""}
                            delta = {"type": "text_delta", "text": block["text"]}
                        events.append(("content_block_start", {"index": index, "content_block": start}))
                        events.append(("content_block_delta", {"index": index, "delta": delta}))
                        events.append(("content_block_stop", {"index": index}))
                    events.extend([
                        ("message_delta", {"delta": {"stop_reason": stop, "stop_sequence": None},
                                           "usage": {"output_tokens": 100}}),
                        ("message_stop", {}),
                    ])
                    encoded = "".join("event: " + kind + "\ndata: " + json.dumps({"type": kind, **value}) + "\n\n"
                                      for kind, value in events).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

            endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
            worker = threading.Thread(target=endpoint.serve_forever, daemon=True)
            worker.start()
            origin = f"http://127.0.0.1:{endpoint.server_port}"
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("ANTHROPIC_", "CLAUDE_", "BEHAVIOR_", "QWEN_", "EMBODIED_"))
                   and key not in {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY",
                                   "all_proxy", "DISABLE_COMPACT", "DISABLE_AUTO_COMPACT", "CLAUDECODE"}}
            env.update({
                "CLAUDE_CONFIG_DIR": str(runtime / "claude"), "XDG_RUNTIME_DIR": str(runtime / "runtime"),
                "BEHAVIOR_SESSION_ID": "native-lifecycle", "BEHAVIOR_RECORD": "0",
                "BEHAVIOR_BASE_URL": origin,
                "BEHAVIOR_RECORD_ROOT": str(runtime / "records"),
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "100000", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",
                "EMBODIED_CLAUDE_PYTHON": sys.executable,
                "ANTHROPIC_API_KEY": "local-test-only", "ANTHROPIC_AUTH_TOKEN": "local-test-only",
                "ANTHROPIC_BASE_URL": origin,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
                "ENABLE_TOOL_SEARCH": "false", "MCP_CONNECTION_NONBLOCKING": "0",
                "PYTHONPATH": str(ROOT / "src"),
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            })
            bridge_process = None
            try:
                if bridge:
                    ready = runtime / "bridge-port"
                    bridge_process = subprocess.Popen([
                        sys.executable, "-m", "embodied_claude_code.qwen_bridge",
                        "--host", "127.0.0.1", "--port", "0", "--ready-file", str(ready),
                        "--model", MODEL, "--upstream", origin + "/v1/chat/completions",
                    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    deadline = time.monotonic() + 8
                    while not ready.exists() and bridge_process.poll() is None and time.monotonic() < deadline:
                        threading.Event().wait(0.05)
                    self.assertTrue(ready.exists(), "local bridge did not start")
                    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:" + ready.read_text().strip()
                prompt = (
                    "/embodied-claude-code:pick-up-object" if slash else
                    "This is a local lifecycle diagnostic. Activate then cancel pick-up-object and "
                    "close-box, list Skills, then say NATIVE_LIFECYCLE_OK. Do not call robot tools."
                )
                completed = subprocess.run([
                    claude, "--print", "--verbose", "--output-format", "stream-json", "--max-turns", "8",
                    "--plugin-dir", str(ROOT), "--settings", str(ROOT / "profiles/claude-settings.json"),
                    "--permission-mode", "dontAsk", "--allowedTools", "Skill", *EMBODIED_TOOL_GLOBS,
                    "--tools", "Skill", "--model", MODEL,
                    prompt,
                ], cwd=runtime, env=env, text=True, capture_output=True, timeout=60)
            finally:
                if bridge_process is not None:
                    bridge_process.terminate()
                    try:
                        bridge_process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        bridge_process.kill()
                        bridge_process.communicate(timeout=5)
                endpoint.shutdown()
                endpoint.server_close()
                worker.join(2)
            diagnostics = json.dumps({"first_tool_names": [tool.get("function", tool).get("name")
                                       for tool in requests[0].get("tools", [])] if requests else [],
                                      "states": snapshots, "summary_requests": len(summaries),
                                      "request_tails": [json.dumps(request.get("messages", [])[-1:])[-1500:]
                                                        for request in requests]}, default=str)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout[:3000] + completed.stdout[-4000:] + diagnostics)
            self.assertEqual(unexpected, [])
            events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
            statuses = [event for event in events if event.get("subtype") == "status"]
            self.assertTrue(any(event.get("compact_result") == "success" for event in statuses), statuses)
            self.assertFalse(any(event.get("compact_result") == "failed" for event in statuses), statuses)
            self.assertTrue(any(event.get("subtype") == "compact_boundary" for event in events))
            self.assertEqual(len(requests), 6, completed.stdout[-5000:])
            self.assertEqual([state["active_task_skill"] if state else None for state in snapshots],
                             [None, "pick-up-object", None, "close-box", None, None],
                             completed.stderr + completed.stdout[:3000] + completed.stdout[-4000:] + diagnostics)
            self.assertEqual(json.loads(state_path.read_text())["active_task_skill"], None)
            self.assertEqual(monitor[-1]["loaded_skills"], ["behavior-v2-baseline"])
            self.assertTrue(summaries, "native compact did not run")
            self.assertIn("NATIVE_LIFECYCLE_OK", completed.stdout)
            self.assertIn("active_skill", json.dumps(requests[-1]))
            if slash:
                self.assertIn("This expansion alone does not change the active task Skill.",
                              json.dumps(requests[0]))
            print(json.dumps({"native_skill_lifecycle": "bridge" if bridge else "direct",
                              "activation": "slash" if slash else "native_Skill",
                              "requests": len(requests), "compactions": len(summaries),
                              "active_states": [state["active_task_skill"] if state else None for state in snapshots],
                              "robot_actions": len(unexpected), "result": "passed"}), flush=True)


if __name__ == "__main__":
    unittest.main()
