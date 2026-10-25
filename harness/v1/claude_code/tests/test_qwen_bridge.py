from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from embodied_claude_code.qwen_bridge import (
    BridgeConfig,
    BridgeServer,
    VISUAL_GROUNDING_GATE,
    normalize_upstream_url,
    translate_chat_response,
    translate_messages_request,
)
from embodied_claude_code.coordinates import (
    LATEST_IMAGE_REMINDER_PREFIX,
    VLM_IMAGE_COORDINATE_CONTRACT,
    VLM_IMAGE_COORDINATE_REMINDER,
    latest_image_grounding_reminder,
)


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BridgeConfig(
            upstream_url="http://127.0.0.1:9/v1/chat/completions",
            model="Qwen3.8-27B",
        )

    def test_upstream_url_accepts_base_or_completion_endpoint(self) -> None:
        self.assertEqual(
            normalize_upstream_url("http://model:30000/v1"),
            "http://model:30000/v1/chat/completions",
        )
        self.assertEqual(
            normalize_upstream_url("http://model:30000/v1/chat/completions"),
            "http://model:30000/v1/chat/completions",
        )

    def test_system_tools_and_tool_choice_are_translated(self) -> None:
        request = translate_messages_request(
            {
                "system": [
                    {"type": "text", "text": "system one"},
                    {"type": "text", "text": "system two"},
                ],
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "name": "capture_head_camera",
                        "description": "capture",
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "capture_head_camera"},
                "max_tokens": 256,
            },
            self.config,
        )
        self.assertEqual(request["model"], "Qwen3.8-27B")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("system one\n\nsystem two", request["messages"][0]["content"])
        self.assertIn(
            VLM_IMAGE_COORDINATE_CONTRACT, request["messages"][0]["content"]
        )
        self.assertEqual(
            request["tools"][0]["function"]["parameters"]["additionalProperties"],
            False,
        )
        self.assertEqual(
            request["tool_choice"]["function"]["name"], "capture_head_camera"
        )
        self.assertFalse(request["stream"])

    def test_existing_coordinate_contract_is_not_duplicated(self) -> None:
        request = translate_messages_request(
            {
                "system": f"system policy\n\n{VLM_IMAGE_COORDINATE_CONTRACT}",
                "messages": [{"role": "user", "content": "hello"}],
            },
            self.config,
        )
        self.assertEqual(
            request["messages"][0]["content"].count(
                VLM_IMAGE_COORDINATE_CONTRACT
            ),
            1,
        )

    def test_client_tool_choice_is_preserved_after_tool_results(self) -> None:
        choices = (
            (None, None),
            ({"type": "auto"}, "auto"),
            ({"type": "none"}, "none"),
            ({"type": "any"}, "required"),
            (
                {"type": "tool", "name": "capture_head_camera"},
                {"type": "function", "function": {"name": "capture_head_camera"}},
            ),
        )
        for suffix in (
            [],
            [{"role": "user", "content": "Stop operating and report what happened."}],
            [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n"
                        "Your task is to create a detailed summary of the conversation."
                    )},
                    {"type": "text", "text": "<total_tokens>14800000 tokens left</total_tokens>"},
                ],
            }],
        ):
            for incoming, outgoing in choices:
                with self.subTest(suffix=suffix, tool_choice=incoming):
                    payload = self._tool_continuation_payload()
                    payload["messages"].extend(suffix)
                    if incoming is not None:
                        payload["tool_choice"] = incoming
                    request = translate_messages_request(payload, self.config)
                    if outgoing is None:
                        self.assertNotIn("tool_choice", request)
                    else:
                        self.assertEqual(request["tool_choice"], outgoing)

    @staticmethod
    def _tool_continuation_payload() -> dict:
        return {
            "messages": [
                {"role": "user", "content": "operate the robot"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use", "id": "toolu_1",
                        "name": "capture_head_camera", "input": {},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result", "tool_use_id": "toolu_1",
                        "content": "observation ready",
                    }],
                },
            ],
            "tools": [{
                "name": "capture_head_camera", "description": "capture",
                "input_schema": {"type": "object"},
            }],
        }

    def test_initial_turn_keeps_automatic_tool_choice(self) -> None:
        request = translate_messages_request(
            {
                "messages": [{"role": "user", "content": "operate"}],
                "tools": [
                    {
                        "name": "capture_head_camera",
                        "description": "capture",
                        "input_schema": {"type": "object"},
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
            self.config,
        )
        self.assertEqual(request["tool_choice"], "auto")

    def test_assistant_tool_use_and_user_tool_result_round_trip(self) -> None:
        image_data = base64.b64encode(b"image").decode("ascii")
        request = translate_messages_request(
            {
                "messages": [
                    {"role": "user", "content": "look"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "capture_head_camera",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "camera result"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": image_data,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ],
                "max_tokens": 64,
            },
            self.config,
        )
        messages = request["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(VLM_IMAGE_COORDINATE_CONTRACT, messages[0]["content"])
        assistant = messages[2]
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "toolu_1")
        self.assertIn("camera result", messages[3]["content"])
        self.assertEqual(messages[4]["role"], "user")
        self.assertEqual(messages[4]["content"][0]["type"], "image_url")
        self.assertIn(
            VLM_IMAGE_COORDINATE_REMINDER,
            messages[4]["content"][1]["text"],
        )
        visual = messages[4]["content"][0]["image_url"]["url"]
        self.assertEqual(visual, f"data:image/jpeg;base64,{image_data}")
        self.assertEqual(
            request["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["top_p"], 1.0)

    def test_launcher_image_is_injected_only_on_first_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            image.write_bytes(b"png-data")
            config = BridgeConfig(
                upstream_url=self.config.upstream_url,
                model=self.config.model,
                prompt_images=(image,),
            )
            first = translate_messages_request(
                {"messages": [{"role": "user", "content": "first"}]}, config
            )
            later = translate_messages_request(
                {
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "done"},
                        {"role": "user", "content": "next"},
                    ]
                },
                config,
            )
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertEqual(first["messages"][1]["content"][0]["type"], "image_url")
        self.assertEqual(
            first["chat_template_kwargs"], {"enable_thinking": False}
        )
        serialized_later = json.dumps(later)
        self.assertNotIn("base64", serialized_later)
        self.assertNotIn("chat_template_kwargs", later)

    def test_historical_image_does_not_disable_thinking_on_later_turn(self) -> None:
        image_data = base64.b64encode(b"image").decode("ascii")
        request = translate_messages_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": "inspect"},
                        ],
                    },
                    {"role": "assistant", "content": "inspected"},
                    {"role": "user", "content": "plan the next step"},
                ]
            },
            self.config,
        )
        self.assertNotIn("chat_template_kwargs", request)
        self.assertIn(image_data, json.dumps(request))

    def test_text_turn_preserves_requested_sampling(self) -> None:
        request = translate_messages_request(
            {
                "messages": [{"role": "user", "content": "plan"}],
                "temperature": 0.7,
                "top_p": 0.8,
            },
            self.config,
        )
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["top_p"], 0.8)

    def test_visual_turn_forces_deterministic_sampling(self) -> None:
        image_data = base64.b64encode(b"image").decode("ascii")
        request = translate_messages_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": "locate the point"},
                        ],
                    }
                ],
                "temperature": 1.0,
                "top_p": 0.5,
            },
            self.config,
        )
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["top_p"], 1.0)

    def test_only_latest_visual_observation_is_sent_upstream(self) -> None:
        old_image = base64.b64encode(b"old-image").decode("ascii")
        new_image = base64.b64encode(b"new-image").decode("ascii")
        request = translate_messages_request(
            {
                "messages": [
                    {"role": "user", "content": "inspect successive frames"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_old",
                                "name": "capture_head_camera",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_old",
                                "content": [
                                    {"type": "text", "text": "image_id=img_old"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": old_image,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                    {"role": "assistant", "content": "old frame inspected"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_new",
                                "name": "capture_head_camera",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_new",
                                "content": [
                                    {"type": "text", "text": "image_id=img_new"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": new_image,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ]
            },
            self.config,
        )

        serialized = json.dumps(request)
        self.assertNotIn(old_image, serialized)
        self.assertIn(new_image, serialized)
        self.assertIn("image_id=img_old", serialized)
        self.assertIn("image_id=img_new", serialized)
        self.assertEqual(serialized.count("data:image/png;base64,"), 1)
        self.assertEqual(serialized.count(VLM_IMAGE_COORDINATE_REMINDER), 1)
        self.assertEqual(serialized.count(VISUAL_GROUNDING_GATE), 1)
        self.assertEqual(
            request["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_latest_image_reminder_moves_with_image_and_old_reminder_is_removed(self) -> None:
        messages = [{"role": "user", "content": "inspect successive frames"}]
        for image_id in ("img_old", "img_new"):
            messages.extend([
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": f"toolu_{image_id}",
                        "name": "capture_head_camera",
                        "input": {},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{image_id}",
                        "content": [
                            {"type": "text", "text": f"image_id={image_id}"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(image_id.encode()).decode(),
                                },
                            },
                            {
                                "type": "text",
                                "text": latest_image_grounding_reminder(image_id),
                            },
                        ],
                    }],
                },
            ])

        request = translate_messages_request({"messages": messages}, self.config)
        serialized = json.dumps(request)
        self.assertNotIn(latest_image_grounding_reminder("img_old"), serialized)
        self.assertIn(latest_image_grounding_reminder("img_new"), serialized)
        self.assertEqual(serialized.count(LATEST_IMAGE_REMINDER_PREFIX), 1)
        self.assertEqual(serialized.count("data:image/png;base64,"), 1)
        self.assertNotIn(VISUAL_GROUNDING_GATE, serialized)
        tool_messages = [m for m in request["messages"] if m["role"] == "tool"]
        self.assertEqual(
            [m["content"] for m in tool_messages],
            ["image_id=img_old", "image_id=img_new"],
        )
        latest_visual = request["messages"][-1]
        self.assertEqual(latest_visual["role"], "user")
        self.assertEqual(
            latest_visual["content"][0]["image_url"]["url"],
            "data:image/png;base64," + base64.b64encode(b"img_new").decode(),
        )
        self.assertEqual(
            latest_visual["content"][-1]["text"],
            latest_image_grounding_reminder("img_new"),
        )

    def test_reminder_without_image_is_not_silently_dropped(self) -> None:
        reminder = latest_image_grounding_reminder("")
        for extra_blocks in ([], [{"type": "image", "source": {}}]):
            with self.subTest(extra_blocks=extra_blocks):
                request = translate_messages_request(
                    {"messages": [{
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": "toolu_missing_image",
                            "content": [
                                {"type": "text", "text": reminder},
                                *extra_blocks,
                            ],
                        }],
                    }]},
                    self.config,
                )
                self.assertEqual(request["messages"][-1]["role"], "tool")
                self.assertEqual(request["messages"][-1]["content"], reminder)

    def test_tool_call_response_becomes_anthropic_tool_use(self) -> None:
        content, stop_reason, usage = translate_chat_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "private planning",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "adjust_chassis",
                                        "arguments": "{\"forward_m\":0.1}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            self.config.model,
        )
        self.assertEqual(stop_reason, "tool_use")
        self.assertEqual(content[-1]["type"], "tool_use")
        self.assertEqual(content[-1]["input"], {"forward_m": 0.1})
        self.assertEqual(usage, {"input_tokens": 10, "output_tokens": 4})

    def test_reasoning_is_fallback_only_when_no_public_content(self) -> None:
        content, stop_reason, _ = translate_chat_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": None,
                            "reasoning_content": "fallback response",
                        },
                    }
                ]
            },
            self.config.model,
        )
        self.assertEqual(content, [{"type": "text", "text": "fallback response"}])
        self.assertEqual(stop_reason, "end_turn")


class _UpstreamHandler(BaseHTTPRequestHandler):
    request_payload: dict = {}

    def log_message(self, _format, *_args):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).request_payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "你好"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(
    os.environ.get("EMBODIED_NETWORK_TESTS") == "1",
    "socket tests require EMBODIED_NETWORK_TESTS=1",
)
class BridgeHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        upstream_port = cls.upstream.server_address[1]
        cls.bridge = BridgeServer(
            ("127.0.0.1", 0),
            BridgeConfig(
                upstream_url=f"http://127.0.0.1:{upstream_port}/v1/chat/completions",
                model="Qwen3.8-27B",
            ),
        )
        cls.upstream_thread = threading.Thread(
            target=cls.upstream.serve_forever, daemon=True
        )
        cls.bridge_thread = threading.Thread(target=cls.bridge.serve_forever, daemon=True)
        cls.upstream_thread.start()
        cls.bridge_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bridge.shutdown()
        cls.upstream.shutdown()
        cls.bridge.server_close()
        cls.upstream.server_close()
        cls.bridge_thread.join(timeout=2)
        cls.upstream_thread.join(timeout=2)

    def _post(self, path: str, payload: dict) -> tuple[str, str]:
        port = self.bridge.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.headers.get_content_type(), response.read().decode("utf-8")

    def test_nonstream_messages_endpoint(self) -> None:
        media_type, body = self._post(
            "/v1/messages",
            {
                "model": "ignored-claude-name",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            },
        )
        response = json.loads(body)
        self.assertEqual(media_type, "application/json")
        self.assertEqual(response["model"], "Qwen3.8-27B")
        self.assertEqual(response["content"], [{"type": "text", "text": "你好"}])
        self.assertEqual(_UpstreamHandler.request_payload["model"], "Qwen3.8-27B")

    def test_stream_messages_endpoint_emits_complete_anthropic_sse(self) -> None:
        media_type, body = self._post(
            "/v1/messages?beta=true",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
                "stream": True,
            },
        )
        self.assertEqual(media_type, "text/event-stream")
        for event in (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ):
            self.assertIn(f"event: {event}\n", body)
        self.assertIn('"text":"你好"', body)

    def test_count_tokens_endpoint_is_available(self) -> None:
        media_type, body = self._post(
            "/v1/messages/count_tokens",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(media_type, "application/json")
        self.assertGreater(json.loads(body)["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
