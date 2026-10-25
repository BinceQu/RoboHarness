from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code import launcher


ROOT = Path(__file__).resolve().parents[1]


class LauncherArgumentTests(unittest.TestCase):
    def test_defaults_target_qwen_and_behavior(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            options, remaining = launcher._parse([])
        self.assertEqual(options.port, 15060)
        self.assertEqual(options.qwen_model, "Qwen3.8-27B")
        self.assertEqual(
            options.qwen_url,
            "http://127.0.0.1:31000/v1/chat/completions",
        )
        self.assertEqual(remaining, [])

    def test_wrapper_options_and_separator(self) -> None:
        options, remaining = launcher._parse(
            [
                "--port=15063",
                "--qwen-url",
                "http://model.local:8000/v1",
                "--qwen-model=custom-qwen",
                "--",
                "exec",
                "hello",
            ]
        )
        self.assertEqual(options.port, 15063)
        self.assertEqual(
            options.qwen_url,
            "http://model.local:8000/v1/chat/completions",
        )
        self.assertEqual(options.qwen_model, "custom-qwen")
        self.assertEqual(remaining, ["exec", "hello"])

    def test_environment_options_are_supported(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BEHAVIOR_PORT": "15063",
                "QWEN_CHAT_COMPLETIONS_URL": "http://qwen.local:8000/v1",
            },
            clear=True,
        ):
            options, remaining = launcher._parse([])
        self.assertEqual(options.port, 15063)
        self.assertEqual(
            options.qwen_url,
            "http://qwen.local:8000/v1/chat/completions",
        )
        self.assertEqual(remaining, [])

    def test_invalid_port_and_missing_value_fail_before_launch(self) -> None:
        for args in (["--port", "0"], ["--port", "65536"], ["--port"]):
            with self.subTest(args=args), self.assertRaises(launcher.UsageError):
                launcher._parse(args)

    def test_security_options_cannot_override_boundary(self) -> None:
        for option in (
            "--tools=Bash",
            "--permission-mode=bypassPermissions",
            "--plugin-dir=/tmp/other",
            "--model=other",
            "--bare",
        ):
            with self.subTest(option=option), self.assertRaises(launcher.UsageError):
                launcher._assert_safe_passthrough([option])

    def test_session_id_matches_interface_contract(self) -> None:
        launcher._validate_session_id("tvdraw15063-git-a25-20260903")
        for value in ("-bad", "bad id", "x" * 65, "session:one", "任务"):
            with self.subTest(value=value), self.assertRaises(launcher.UsageError):
                launcher._validate_session_id(value)

    def test_exec_stdin_is_translated_without_losing_prompt(self) -> None:
        with patch("sys.stdin", StringIO("你好，观察机器人。\n")):
            args, stdin_prompt = launcher._translate_exec(["exec", "-"])
        self.assertEqual(args, ["--print"])
        self.assertEqual(stdin_prompt, "你好，观察机器人。\n")

    def test_exec_inline_prompt_is_translated(self) -> None:
        args, stdin_prompt = launcher._translate_exec(["exec", "inspect scene"])
        self.assertEqual(args, ["--print", "inspect scene"])
        self.assertIsNone(stdin_prompt)
        self.assertEqual(launcher._prompt_from_args(args, None), "inspect scene")


class LauncherImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _png(self, name: str = "reference.png") -> Path:
        path = self.root / name
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
            )
        )
        return path

    def test_image_is_validated_and_deduplicated(self) -> None:
        image = self._png()
        with patch.dict(os.environ, {"BEHAVIOR_PROMPT_IMAGES": str(image)}, clear=True):
            result = launcher._validate_images([str(image)])
        self.assertEqual(result, [image.resolve()])

    def test_missing_or_forged_image_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(launcher.UsageError):
                launcher._validate_images([str(self.root / "missing.png")])
            fake = self.root / "fake.png"
            fake.write_text("not an image", encoding="utf-8")
            with self.assertRaises(launcher.UsageError):
                launcher._validate_images([str(fake)])


class LauncherIntegrationTests(unittest.TestCase):
    def test_launch_pins_plugin_tools_model_and_local_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            calls: dict[str, object] = {}

            class BridgeProcess:
                returncode = None

                def poll(self):
                    return None

                def terminate(self):
                    calls["terminated"] = True

                def wait(self, timeout=None):
                    self.returncode = 0
                    return 0

                def kill(self):
                    calls["killed"] = True

            def fake_run(command, **kwargs):
                calls["command"] = command
                calls["env"] = kwargs["env"]
                calls["input"] = kwargs.get("input")
                return SimpleNamespace(returncode=0)

            with (
                patch.dict(
                    os.environ,
                    {"XDG_RUNTIME_DIR": str(runtime)},
                    clear=True,
                ),
                patch.object(launcher, "_get_json") as get_json,
                patch.object(launcher, "_post_monitor") as post_monitor,
                patch.object(launcher, "_claude_executable", return_value="/fake/claude"),
                patch.object(launcher.subprocess, "Popen", return_value=BridgeProcess()),
                patch.object(launcher, "_wait_for_ready", return_value=18123),
                patch.object(launcher.subprocess, "run", side_effect=fake_run),
                patch.object(launcher, "Path", side_effect=lambda value: (
                    Path(directory) / "fallback" if str(value) == "/tmp/embodied-claude-code" else Path(value)
                )),
            ):
                status = launcher.launch(["--port", "15063", "--", "exec", "hello"])

        self.assertEqual(status, 0)
        self.assertEqual(
            [call.args[0] for call in get_json.call_args_list],
            ["http://127.0.0.1:15063" + path for path in
             ("/__official__/idle_probe", "/api/state", "/api/v2/tools")],
        )
        post_monitor.assert_called_once()
        command = calls["command"]
        self.assertIn("--plugin-dir", command)
        self.assertIn("--permission-mode", command)
        self.assertIn("dontAsk", command)
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "")
        for allowed in launcher.EMBODIED_TOOL_GLOBS:
            self.assertIn(allowed, command)
        env = calls["env"]
        self.assertRegex(env["BEHAVIOR_SESSION_ID"], r"^claude-[a-f0-9]{16}$")
        self.assertEqual(post_monitor.call_args.args[2], env["BEHAVIOR_SESSION_ID"])
        self.assertFalse(post_monitor.call_args.kwargs["resume"])
        self.assertEqual(env["BEHAVIOR_BASE_URL"], "http://127.0.0.1:15063")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:18123")
        self.assertEqual(env["QWEN_MODEL"], "Qwen3.8-27B")
        self.assertEqual(
            env["CLAUDE_PLUGIN_DATA"],
            str(
                (
                    ROOT
                    / ".claude-home"
                    / "plugins"
                    / "data"
                    / "embodied-claude-code-inline"
                ).resolve()
            ),
        )
        self.assertEqual(env["ENABLE_TOOL_SEARCH"], "false")
        self.assertNotIn("EMBODIED_FORCE_TOOL_CHOICE", env)
        self.assertEqual(env["MCP_TIMEOUT"], "120000")
        self.assertEqual(env["MCP_CONNECT_TIMEOUT_MS"], "120000")
        self.assertEqual(env["MCP_CONNECTION_NONBLOCKING"], "0")
        self.assertTrue(calls["terminated"])

    def test_shell_entrypoints_are_executable(self) -> None:
        for name in (
            "run",
            "embodied-claude-code",
            "embodied-claude-code-mcp",
            "qwen-anthropic-bridge",
            "bootstrap",
        ):
            path = ROOT / "scripts" / name
            self.assertTrue(path.is_file())
            self.assertTrue(os.access(path, os.X_OK))


if __name__ == "__main__":
    unittest.main()
