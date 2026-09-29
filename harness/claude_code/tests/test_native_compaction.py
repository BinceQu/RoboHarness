"""Opt-in native Claude regression using a real bridge and harmless Read calls.

The fake upstream reports high token usage to trigger native compaction without
generating a huge conversation. Set EMBODIED_COMPACTION_QWEN_URL explicitly to
route only the native summary request to a real Qwen Chat Completions endpoint.
All task turns and tools remain local; no BEHAVIOR interface is contacted.
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

from embodied_claude_code.qwen_bridge import BridgeConfig, call_upstream


ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "tests" / "fixtures" / "native_compaction_note.txt"
MODEL = "Qwen3.8-Flash-Next-FP8"
SUMMARY_REQUEST = "Your task is to create a detailed summary of the conversation so far"


def _text(messages: list[dict]) -> str:
    return "\n".join(
        value if isinstance(value := message.get("content"), str) else "\n".join(
            block.get("text", "") for block in value or [] if isinstance(block, dict)
        )
        for message in messages
    )


@unittest.skipUnless(
    os.environ.get("EMBODIED_NATIVE_CLAUDE_TESTS") == "1",
    "native CLI test requires EMBODIED_NATIVE_CLAUDE_TESTS=1",
)
class NativeCompactionTests(unittest.TestCase):
    def test_native_summary_compacts_and_tools_resume(self) -> None:
        claude = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        self.assertTrue(claude, "Claude Code must be installed for this opt-in test")
        qwen_url = os.environ.get("EMBODIED_COMPACTION_QWEN_URL", "")
        task_requests: list[dict] = []
        summary_requests: list[dict] = []
        summary_responses: list[dict] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):  # noqa: N802
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = payload["messages"]
                last_user = next(
                    (message for message in reversed(messages) if message["role"] == "user"),
                    {},
                )
                is_summary = SUMMARY_REQUEST in _text([last_user])
                if is_summary:
                    summary_requests.append(payload)
                    if qwen_url:
                        response = call_upstream(payload, BridgeConfig(
                            upstream_url=qwen_url, model=MODEL, timeout_s=120,
                        ))
                        summary_responses.append(response)
                        self._send(response)
                        return
                    # A conforming model obeys required tool choice, reproducing
                    # the old empty-summary failure when the bridge overrides it.
                    if payload.get("tool_choice") == "required":
                        message = self._read_call()
                        finish_reason = "tool_calls"
                    else:
                        message = {"role": "assistant", "content": (
                            "<summary>The user requested repeated reads of the synthetic "
                            f"note at {NOTE}. Four reads completed. The marker is "
                            "COMPACTION_NOTE_720. Read once more, then report completion. "
                            "No robot or external application is involved.</summary>"
                        )}
                        finish_reason = "stop"
                    tokens = 91000
                else:
                    task_requests.append(payload)
                    if len(task_requests) <= 5:
                        message = self._read_call()
                        finish_reason = "tool_calls"
                    else:
                        message = {"role": "assistant", "content": "NATIVE_COMPACTION_OK"}
                        finish_reason = "stop"
                    tokens = 90000 if len(task_requests) == 4 else 2000
                response = {
                    "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
                    "usage": {"prompt_tokens": tokens, "completion_tokens": 50},
                }
                if is_summary:
                    summary_responses.append(response)
                self._send(response)

            def _read_call(self):
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_read_{len(task_requests)}_{len(summary_requests)}",
                    "type": "function",
                    "function": {"name": "Read", "arguments": json.dumps({"file_path": str(NOTE)})},
                }]}

            def _send(self, response):
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with tempfile.TemporaryDirectory(prefix="cc-native-compaction-") as directory:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(upstream.server_close)
            self.addCleanup(thread.join, 2)
            self.addCleanup(upstream.shutdown)
            env = {
                key: value for key, value in os.environ.items()
                if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "QWEN_"))
                and key not in {"DISABLE_COMPACT", "DISABLE_AUTO_COMPACT", "CLAUDECODE"}
            }
            env.update({
                "CLAUDE_CONFIG_DIR": directory,
                "ANTHROPIC_API_KEY": "local-test-only",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "100000",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",
                # Existing shells must not reactivate the removed override.
                "EMBODIED_FORCE_TOOL_CHOICE": "1",
                "PYTHONPATH": str(ROOT / "src"),
                "NO_PROXY": "127.0.0.1,localhost,100.101.73.1",
                "no_proxy": "127.0.0.1,localhost,100.101.73.1",
            })
            ready = Path(directory) / "bridge-port"
            with subprocess.Popen([
                sys.executable, "-m", "embodied_claude_code.qwen_bridge",
                "--host", "127.0.0.1", "--port", "0", "--ready-file", str(ready),
                "--model", MODEL,
                "--upstream", f"http://127.0.0.1:{upstream.server_address[1]}/v1/chat/completions",
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as bridge:
                try:
                    deadline = time.monotonic() + 8
                    while not ready.is_file() and bridge.poll() is None and time.monotonic() < deadline:
                        threading.Event().wait(0.05)
                    self.assertTrue(ready.is_file(), "local bridge did not start")
                    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{ready.read_text().strip()}"
                    completed = subprocess.run([
                        claude, "--print", "--verbose", "--output-format", "stream-json",
                        "--no-session-persistence", "--strict-mcp-config",
                        "--mcp-config", '{"mcpServers":{}}',
                        "--tools", "Read", "--allowedTools", "Read", "--permission-mode", "dontAsk",
                        "--model", MODEL, "--max-turns", "8",
                        "--settings", '{"autoCompactEnabled":true}',
                        f"Read {NOTE} five times. Then say NATIVE_COMPACTION_OK. This is a local diagnostic.",
                    ], cwd=directory, env=env, text=True, capture_output=True, timeout=150 if qwen_url else 45)
                finally:
                    bridge.terminate()
                    try:
                        bridge.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.communicate(timeout=5)

        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout[-2500:])
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
        statuses = [event for event in events if event.get("subtype") == "status"]
        self.assertTrue(any(event.get("compact_result") == "success" for event in statuses), statuses)
        self.assertFalse(any(event.get("compact_result") == "failed" for event in statuses), statuses)
        self.assertTrue(any(event.get("subtype") == "compact_boundary" for event in events))
        self.assertEqual(len(summary_requests), 1)
        summary = summary_requests[0]
        self.assertNotIn("tool_choice", summary)
        self.assertTrue(summary["tools"], "native compaction keeps tool schemas")
        self.assertTrue(any(message["role"] == "tool" for message in summary["messages"]))
        answer = summary_responses[0]["choices"][0]["message"]
        self.assertTrue((answer.get("content") or "").strip(), answer)
        self.assertFalse(answer.get("tool_calls"), answer)
        self.assertEqual(len(task_requests), 6, "one Read and a final response must follow compaction")
        self.assertIn("COMPACTION_NOTE_720", _text(task_requests[4]["messages"]))
        self.assertLess(len(task_requests[4]["messages"]), len(task_requests[3]["messages"]))
        result = next(event for event in events if event.get("type") == "result")
        self.assertFalse(result.get("is_error"), result)
        self.assertEqual(result["result"], "NATIVE_COMPACTION_OK")
        print(json.dumps({
            "compaction_test": "live_qwen_summary" if qwen_url else "fake_qwen",
            "compact_result": "success", "summary_tool_choice": summary.get("tool_choice"),
            "summary_characters": len(answer["content"]),
            "summary_usage": summary_responses[0].get("usage"),
            "messages_before": len(task_requests[3]["messages"]),
            "messages_after": len(task_requests[4]["messages"]),
            "post_compaction_read": "passed", "result": result["result"],
        }), flush=True)


if __name__ == "__main__":
    unittest.main()
