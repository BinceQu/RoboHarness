from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "embodied-codex"


@unittest.skipUnless(shutil.which("sh"), "launcher tests require a POSIX shell")
class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.codex_args = self.root / "codex-args.txt"
        self.curl_args = self.root / "curl-args.txt"

        self._write_executable(
            "codex",
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" >\"$CODEX_ARGS_FILE\"\n"
            "printf '%s\\n' \"$BEHAVIOR_BASE_URL\" >\"$CODEX_ENV_FILE\"\n"
            "if [ -n \"${CODEX_STDIN_FILE:-}\" ]; then cat >\"$CODEX_STDIN_FILE\"; fi\n",
        )
        self._write_executable(
            "curl",
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" >>\"$CURL_ARGS_FILE\"\n"
            "exit \"${FAKE_CURL_EXIT:-0}\"\n",
        )

    def _write_executable(self, name: str, contents: str) -> None:
        path = self.bin_dir / name
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)

    def _run(
        self,
        *args: str,
        curl_exit: int = 0,
        extra_env: dict[str, str] | None = None,
        stdin_path: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}",
                "CODEX_ARGS_FILE": str(self.codex_args),
                "CODEX_ENV_FILE": str(self.root / "codex-env.txt"),
                "CODEX_STDIN_FILE": str(self.root / "codex-stdin.txt"),
                "CURL_ARGS_FILE": str(self.curl_args),
                "FAKE_CURL_EXIT": str(curl_exit),
                "XDG_RUNTIME_DIR": str(self.root / "runtime"),
            }
        )
        if extra_env:
            env.update(extra_env)
        stdin = None
        if stdin_path is not None:
            stdin = stdin_path.open(encoding="utf-8")
        try:
            return subprocess.run(
                ["sh", str(LAUNCHER), *args],
                capture_output=True,
                check=False,
                env=env,
                stdin=stdin,
                text=True,
            )
        finally:
            if stdin is not None:
                stdin.close()

    def _captured_codex_args(self) -> list[str]:
        return self.codex_args.read_text(encoding="utf-8").splitlines()

    def test_default_port_and_codex_arguments(self) -> None:
        result = self._run("--", "--model", "gpt-5.6-sol")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._captured_codex_args(),
            [
                "--profile",
                "embodied",
                "-c",
                (
                    "mcp_servers.behavior-v2.env.BEHAVIOR_BASE_URL="
                    "http://127.0.0.1:15060"
                ),
                "--model",
                "gpt-5.6-sol",
            ],
        )
        curl_args = self.curl_args.read_text(encoding="utf-8")
        self.assertIn("http://127.0.0.1:15060/api/state", curl_args)
        self.assertIn("http://127.0.0.1:15060/api/v2/tools", curl_args)
        stamp = self.root / "runtime" / "embodied-codex" / "behavior_base_url"
        self.assertEqual(
            stamp.read_text(encoding="utf-8").strip(),
            "http://127.0.0.1:15060",
        )
        self.assertEqual(
            (self.root / "codex-env.txt").read_text(encoding="utf-8").strip(),
            "http://127.0.0.1:15060",
        )

    def test_explicit_port_is_session_scoped_config_override(self) -> None:
        result = self._run("--port", "15062")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            (
                "mcp_servers.behavior-v2.env.BEHAVIOR_BASE_URL="
                "http://127.0.0.1:15062"
            ),
            self._captured_codex_args(),
        )
        self.assertIn("Starting embodied Codex on http://127.0.0.1:15062", result.stderr)

    def test_equals_port_form(self) -> None:
        result = self._run("--port=15080")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "http://127.0.0.1:15080/api/v2/tools",
            self.curl_args.read_text(encoding="utf-8"),
        )

    def test_overlong_session_id_is_rejected_before_health_check(self) -> None:
        long_id = (
            "search-pick-place-15060-prompt-gates-sol-attempt12-continue-20260820"
        )
        result = self._run(
            "--port",
            "15060",
            extra_env={"BEHAVIOR_SESSION_ID": long_id},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("official max is 64", result.stderr)
        self.assertFalse(self.codex_args.exists())
        self.assertFalse(self.curl_args.exists())

    def test_invalid_session_id_characters_are_rejected_before_health_check(
        self,
    ) -> None:
        result = self._run(
            "--port",
            "15060",
            extra_env={"BEHAVIOR_SESSION_ID": "bad id/with space"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported characters", result.stderr)
        self.assertFalse(self.codex_args.exists())
        self.assertFalse(self.curl_args.exists())

    def test_invalid_ports_are_rejected_before_health_check(self) -> None:
        for port in ("", "abc", "0", "65536", "12/34"):
            with self.subTest(port=port):
                self.codex_args.unlink(missing_ok=True)
                self.curl_args.unlink(missing_ok=True)
                result = self._run("--port", port)

                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.codex_args.exists())
                self.assertFalse(self.curl_args.exists())

    def test_missing_port_value_is_rejected(self) -> None:
        result = self._run("--port")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--port requires a value", result.stderr)
        self.assertFalse(self.codex_args.exists())

    def test_unhealthy_endpoint_does_not_start_codex(self) -> None:
        result = self._run("--port", "15080", curl_exit=7)

        self.assertEqual(result.returncode, 1)
        self.assertIn("BEHAVIOR endpoint is not healthy", result.stderr)
        self.assertIn("No Codex session was started", result.stderr)
        self.assertFalse(self.codex_args.exists())

    def test_help_has_no_runtime_side_effects(self) -> None:
        result = self._run("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage: embodied-codex", result.stdout)
        self.assertFalse(self.curl_args.exists())
        self.assertFalse(self.codex_args.exists())

    def test_exec_stdin_prompt_is_published_before_codex_starts(self) -> None:
        prompt = self.root / "prompt.txt"
        prompt.write_text(
            "Find one ungrasped can of soda. Follow pick-up-object and "
            "place-object-in-container.\n",
            encoding="utf-8",
        )
        result = self._run(
            "--port",
            "15060",
            "exec",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "-C",
            str(self.root / "workspace"),
            "-",
            extra_env={"BEHAVIOR_SESSION_ID": "luna-attempt6"},
            stdin_path=prompt,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        curl_args = self.curl_args.read_text(encoding="utf-8")
        self.assertIn("/api/agent_monitor/prompt", curl_args)
        self.assertIn("Published user prompt to http://127.0.0.1:15060", result.stderr)
        stamp = self.root / "runtime" / "embodied-codex" / "user_prompt.luna-attempt6"
        self.assertIn("Find one ungrasped can of soda", stamp.read_text(encoding="utf-8"))
        body = (self.root / "runtime" / "embodied-codex" / "prompt.body.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("luna-attempt6", body)
        self.assertIn("Find one ungrasped can of soda", body)
        self.assertIn("pick-up-object", body)
        self.assertIn("loaded_skills", body)
        self.assertIn("new_attempt", body)
        # 只复制 stdin 文件，不能把它读空，否则 codex exec - 会丢原文。
        self.assertIn(
            "Find one ungrasped can of soda",
            (self.root / "codex-stdin.txt").read_text(encoding="utf-8"),
        )

    def test_exec_cli_prompt_is_published(self) -> None:
        result = self._run(
            "exec",
            "--json",
            "只捡起地上的罐子",
            extra_env={"BEHAVIOR_SESSION_ID": "sess-cli"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/api/agent_monitor/prompt", self.curl_args.read_text(encoding="utf-8"))
        stamp = self.root / "runtime" / "embodied-codex" / "user_prompt.sess-cli"
        self.assertEqual(stamp.read_text(encoding="utf-8").strip(), "只捡起地上的罐子")

    def test_exec_clears_stale_active_task_skill_stamp(self) -> None:
        stamp_dir = self.root / "runtime" / "embodied-codex"
        stamp_dir.mkdir(parents=True)
        leftover = stamp_dir / "active_task_skill.unit-clear-stamp"
        leftover.write_text("open-doors-and-drawers\n", encoding="utf-8")
        prompt = self.root / "prompt.txt"
        prompt.write_text("Find one ungrasped can of soda.\n", encoding="utf-8")
        result = self._run(
            "--port",
            "15060",
            "exec",
            "--json",
            "-",
            extra_env={"BEHAVIOR_SESSION_ID": "unit-clear-stamp"},
            stdin_path=prompt,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(leftover.exists())

    def test_tui_launch_does_not_publish_empty_prompt(self) -> None:
        result = self._run("--port", "15062")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(
            "/api/agent_monitor/prompt",
            self.curl_args.read_text(encoding="utf-8"),
        )

    def _write_min_png(self, path: Path) -> Path:
        # 1x1 透明 PNG，只用于校验魔数，不依赖外部文件。
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
            )
        )
        return path

    def test_help_documents_prompt_image(self) -> None:
        result = self._run("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prompt-image FILE", result.stdout)
        self.assertIn("BEHAVIOR_PROMPT_IMAGES", result.stdout)
        self.assertFalse(self.codex_args.exists())

    def test_text_only_exec_does_not_inject_image_flag(self) -> None:
        prompt = self.root / "prompt.txt"
        prompt.write_text("Find one ungrasped can of soda.\n", encoding="utf-8")
        result = self._run(
            "--port",
            "15060",
            "exec",
            "--json",
            "-",
            extra_env={"BEHAVIOR_SESSION_ID": "text-only"},
            stdin_path=prompt,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--image", " ".join(self._captured_codex_args()))
        self.assertFalse(
            any(arg.startswith("--image=") for arg in self._captured_codex_args())
        )

    def test_prompt_image_is_injected_after_exec_as_equals_form(self) -> None:
        png = self._write_min_png(self.root / "wrist-bad.png")
        prompt = self.root / "prompt.txt"
        prompt.write_text("Open the TV-stand drawer.\n", encoding="utf-8")
        result = self._run(
            "--port",
            "15063",
            "--prompt-image",
            str(png),
            "--",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(self.root / "workspace"),
            "-",
            extra_env={"BEHAVIOR_SESSION_ID": "img-exec"},
            stdin_path=prompt,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self._captured_codex_args()
        self.assertIn("exec", args)
        image_flag = f"--image={png.resolve()}"
        self.assertIn(image_flag, args)
        self.assertEqual(args[args.index("exec") + 1], image_flag)
        self.assertEqual(args[-1], "-")
        self.assertIn("attaching 1 prompt image(s)", result.stderr)
        body = (self.root / "runtime" / "embodied-codex" / "prompt.body.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("attached prompt images", body)
        self.assertIn(str(png.resolve()), body)
        self.assertIn(
            "Open the TV-stand drawer.",
            (self.root / "codex-stdin.txt").read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "attached prompt images",
            (self.root / "codex-stdin.txt").read_text(encoding="utf-8"),
        )

    def test_behavior_prompt_images_env_is_opt_in(self) -> None:
        png = self._write_min_png(self.root / "ref.png")
        result = self._run(
            "--port",
            "15060",
            "exec",
            "--json",
            "只捡起地上的罐子",
            extra_env={
                "BEHAVIOR_SESSION_ID": "img-env",
                "BEHAVIOR_PROMPT_IMAGES": str(png),
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--image={png.resolve()}", self._captured_codex_args())

    def test_missing_prompt_image_does_not_start_codex(self) -> None:
        result = self._run(
            "--port",
            "15060",
            "--prompt-image",
            str(self.root / "missing.png"),
            "--",
            "exec",
            "-",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("prompt image not found", result.stderr)
        self.assertFalse(self.codex_args.exists())
        self.assertFalse(self.curl_args.exists())

    def test_wrong_extension_is_rejected_before_health_check(self) -> None:
        txt = self.root / "notes.txt"
        txt.write_text("not an image\n", encoding="utf-8")
        result = self._run(
            "--port",
            "15060",
            "--prompt-image",
            str(txt),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported prompt image type", result.stderr)
        self.assertFalse(self.codex_args.exists())
        self.assertFalse(self.curl_args.exists())

    def test_existing_exec_image_flag_is_not_treated_as_prompt_text(self) -> None:
        png = self._write_min_png(self.root / "already.png")
        result = self._run(
            "exec",
            "--json",
            "--image",
            str(png),
            "只捡起地上的罐子",
            extra_env={"BEHAVIOR_SESSION_ID": "img-passthrough"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stamp = self.root / "runtime" / "embodied-codex" / "user_prompt.img-passthrough"
        self.assertEqual(stamp.read_text(encoding="utf-8").strip(), "只捡起地上的罐子")
        self.assertNotIn(str(png), stamp.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
