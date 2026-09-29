from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code.config import Settings
from embodied_claude_code.errors import ConfigurationError, ToolPolicyError
from embodied_claude_code.profile import MCP_EXCLUDED_TOOLS, ToolProfile


ROOT = Path(__file__).resolve().parents[1]
class SettingsTests(unittest.TestCase):
    def test_model_image_limits_have_bounded_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.max_images_per_call, 1)
        self.assertEqual(settings.model_image_max_bytes, 256 * 1024)
        self.assertEqual(settings.model_image_max_edge, 720)
        self.assertEqual(settings.model_image_jpeg_quality, 95)
        self.assertEqual(settings.image_converter, "/usr/bin/convert")

    def test_model_image_limits_are_read_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BEHAVIOR_MODEL_IMAGE_MAX_BYTES": "131072",
                "BEHAVIOR_MODEL_IMAGE_MAX_EDGE": "640",
                "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY": "75",
                "BEHAVIOR_IMAGE_CONVERTER": "/opt/imagemagick/convert",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.model_image_max_bytes, 131072)
        self.assertEqual(settings.model_image_max_edge, 640)
        self.assertEqual(settings.model_image_jpeg_quality, 75)
        self.assertEqual(settings.image_converter, "/opt/imagemagick/convert")

    def test_model_image_limits_reject_invalid_values(self) -> None:
        invalid = {
            "BEHAVIOR_MODEL_IMAGE_MAX_BYTES": ("0", "bad"),
            "BEHAVIOR_MODEL_IMAGE_MAX_EDGE": ("-1", "bad"),
            "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY": ("0", "96", "bad"),
            "BEHAVIOR_IMAGE_CONVERTER": ("",),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {name: value}, clear=True):
                        with self.assertRaises(ConfigurationError):
                            Settings.from_env()

    def test_memory_timeout_defaults_to_one_second(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.memory_timeout_s, 1.0)

    def test_memory_timeout_is_read_from_environment(self) -> None:
        with patch.dict(
            os.environ, {"BEHAVIOR_MEMORY_TIMEOUT_S": "0.25"}, clear=True
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.memory_timeout_s, 0.25)

    def test_memory_timeout_rejects_non_numeric_and_out_of_bounds_values(self) -> None:
        for value in ("not-a-number", "0", "-0.1", "5.01", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"BEHAVIOR_MEMORY_TIMEOUT_S": value}, clear=True
                ):
                    with self.assertRaises(ConfigurationError):
                        Settings.from_env()

    def test_remote_origin_is_rejected_by_default(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings(base_url="http://example.com:5011")

    def test_remote_origin_can_be_explicitly_allowed(self) -> None:
        settings = Settings(
            base_url="https://robot.example.com", allow_remote=True
        )
        self.assertEqual(settings.base_url, "https://robot.example.com")


class ProfileTests(unittest.TestCase):
    def test_mcp_exclusions_cannot_be_reenabled_by_profile(self) -> None:
        excluded = set(MCP_EXCLUDED_TOOLS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "embodied_claude_code.profile.v1",
                        "name": "allow-all",
                        "allow_tools": ["*"],
                        "deny_tools": [],
                        "fixed_arguments": {},
                    }
                ),
                encoding="utf-8",
            )
            profile = ToolProfile.load(path)

        self.assertTrue(all(not profile.allows(name) for name in excluded))
        self.assertTrue(excluded.issubset(profile.deny_tools))
        self.assertTrue(profile.allows("plan_grasp_point_filter_rgbd_lite"))
        self.assertTrue(profile.allows("control_wrist_roll"))
        self.assertFalse(profile.allows("cut_object"))
        self.assertFalse(profile.allows("plan_press_point"))
        self.assertFalse(profile.allows("adjust_plan_pose"))

    def test_profile_filters_and_fixes_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "embodied_claude_code.profile.v1",
                        "name": "test",
                        "description": "test profile",
                        "allow_tools": ["capture_head_camera", "adjust_chassis"],
                        "deny_tools": ["adjust_chassis"],
                        "fixed_arguments": {
                            "capture_head_camera": {"timeout_s": 10}
                        },
                    }
                ),
                encoding="utf-8",
            )
            profile = ToolProfile.load(path)

        self.assertTrue(profile.allows("capture_head_camera"))
        self.assertFalse(profile.allows("adjust_chassis"))
        self.assertEqual(
            profile.apply_arguments("capture_head_camera", {}),
            {"timeout_s": 10},
        )
        with self.assertRaises(ToolPolicyError):
            profile.apply_arguments(
                "capture_head_camera", {"timeout_s": 20}
            )


class ClaudeCodeSessionProfileTests(unittest.TestCase):
    def test_embodied_profile_is_fail_closed(self) -> None:
        settings = json.loads(
            (ROOT / "profiles" / "claude-settings.json").read_text(
                encoding="utf-8"
            )
        )
        permissions = settings["permissions"]
        self.assertEqual(permissions["defaultMode"], "dontAsk")
        self.assertEqual(
            set(permissions["allow"]),
            {
                "mcp__behavior-robot__*",
                "mcp__plugin_embodied-claude-code_behavior-robot__*",
                "mcp__behavior-v2__*",
                "mcp__plugin_embodied-claude-code_behavior-v2__*",
            },
        )

        plugin = json.loads(
            (ROOT / ".claude-plugin" / "plugin.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(plugin["name"], "embodied-claude-code")
        self.assertEqual(plugin["displayName"], "Embodied Claude Code")

        plugin_mcp = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
        bundled = plugin_mcp["mcpServers"]["behavior-v2"]
        self.assertEqual(
            bundled["command"],
            "/bin/sh",
        )
        self.assertEqual(bundled["args"], ["${CLAUDE_PLUGIN_ROOT}/scripts/embodied-claude-code-mcp"])
        self.assertIs(bundled["alwaysLoad"], True)
        self.assertEqual(
            bundled["env"]["EMBODIED_PLUGIN_ROOT"], "${CLAUDE_PLUGIN_ROOT}"
        )
        # Claude exports CLAUDE_PLUGIN_DATA to plugin subprocesses itself.
        # Re-declaring it here can shadow that value with the literal token in
        # --plugin-dir development sessions.
        self.assertNotIn("CLAUDE_PLUGIN_DATA", bundled["env"])
        self.assertNotIn("BEHAVIOR_BASE_URL", bundled["env"])

    def test_hooks_use_claude_plugin_root_and_pretool_boundary(self) -> None:
        hooks = json.loads(
            (ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8")
        )["hooks"]
        self.assertEqual(set(hooks), {"SessionStart", "UserPromptSubmit", "PreToolUse",
                                      "PostToolUse", "UserPromptExpansion"})
        commands = [
            hook["command"]
            for groups in hooks.values()
            for group in groups
            for hook in group["hooks"]
        ]
        self.assertTrue(all("${CLAUDE_PLUGIN_ROOT}" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
