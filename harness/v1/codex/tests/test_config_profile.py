from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 does not provide tomllib.
    tomllib = None

from embodied_codex.config import Settings
from embodied_codex.errors import ConfigurationError, ToolPolicyError
from embodied_codex.profile import MCP_EXCLUDED_TOOLS, ToolProfile


ROOT = Path(__file__).resolve().parents[1]
MCP_COMMAND = "@MCP_COMMAND@"


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
                        "schema_version": "embodied_codex.profile.v1",
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

    def test_profile_filters_and_fixes_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "embodied_codex.profile.v1",
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


class CodexSessionProfileTests(unittest.TestCase):
    @unittest.skipIf(tomllib is None, "tomllib requires Python 3.11+")
    def test_embodied_profile_is_fail_closed(self) -> None:
        with (ROOT / "profiles" / "embodied.config.toml").open("rb") as file:
            profile = tomllib.load(file)

        self.assertEqual(profile["model"], "gpt-5.6-sol")
        self.assertEqual(profile["model_reasoning_effort"], "high")
        self.assertEqual(profile["service_tier"], "fast")
        self.assertNotIn("model_auto_compact_token_limit", profile)
        self.assertNotIn("model_auto_compact_token_limit_scope", profile)
        self.assertFalse(profile["include_apps_instructions"])
        self.assertFalse(profile["include_collaboration_mode_instructions"])
        self.assertFalse(profile["include_permissions_instructions"])
        self.assertFalse(profile["agents"]["enabled"])

        features = profile["features"]
        self.assertEqual(
            features["code_mode"],
            {
                "enabled": True,
                "direct_only_tool_namespaces": ["mcp__behavior_v2"],
            },
        )
        self.assertTrue(features["code_mode_host"])

        plugins = profile["plugins"]
        self.assertFalse(plugins["visualize@openai-bundled"]["enabled"])
        embodied = plugins["embodied-codex-plugin@roboharness"]
        self.assertTrue(embodied["enabled"])

        server = profile["mcp_servers"]["behavior-v2"]
        self.assertEqual(server["command"], MCP_COMMAND)
        self.assertTrue(server["enabled"])
        self.assertTrue(server["required"])
        self.assertEqual(server["default_tools_approval_mode"], "approve")
        read_only_tools = {
            "capture_head_camera",
            "capture_left_wrist_camera",
            "capture_right_wrist_camera",
            "measure_shoulder_distance",
        }
        self.assertEqual(set(server["tools"]), read_only_tools)
        self.assertTrue(
            all(
                tool["approval_mode"] == "approve"
                for tool in server["tools"].values()
            )
        )
        self.assertEqual(
            server["env"]["BEHAVIOR_BASE_URL"],
            "http://127.0.0.1:15060",
        )
        self.assertEqual(
            server["env"]["EMBODIED_PLUGIN_ROOT"],
            "@PLUGIN_ROOT@",
        )
        self.assertNotIn("enabled_tools", server)
        self.assertIn("BEHAVIOR_RECORD", server["env_vars"])
        self.assertIn("EMBODIED_PLUGIN_ROOT", server["env_vars"])
        self.assertIn("BEHAVIOR_MEMORY_TIMEOUT_S", server["env_vars"])
        for name in (
            "BEHAVIOR_MODEL_IMAGE_MAX_BYTES",
            "BEHAVIOR_MODEL_IMAGE_MAX_EDGE",
            "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY",
            "BEHAVIOR_IMAGE_CONVERTER",
        ):
            self.assertIn(name, server["env_vars"])

        plugin_mcp = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
        bundled = plugin_mcp["mcpServers"]["behavior-v2"]
        self.assertEqual(bundled["command"], "/bin/sh")
        self.assertEqual(bundled["args"], ["${CODEX_PLUGIN_ROOT}/scripts/embodied-codex-mcp"])
        self.assertEqual(
            bundled["env"]["EMBODIED_PLUGIN_ROOT"],
            "${CODEX_PLUGIN_ROOT}",
        )
        self.assertIn("EMBODIED_PLUGIN_ROOT", bundled["env_vars"])
        self.assertEqual(bundled["default_tools_approval_mode"], "approve")
        self.assertIn("BEHAVIOR_MEMORY_TIMEOUT_S", bundled["env_vars"])
        for name in (
            "BEHAVIOR_MODEL_IMAGE_MAX_BYTES",
            "BEHAVIOR_MODEL_IMAGE_MAX_EDGE",
            "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY",
            "BEHAVIOR_IMAGE_CONVERTER",
        ):
            self.assertIn(name, bundled["env_vars"])
        self.assertEqual(set(bundled["tools"]), read_only_tools)
        self.assertTrue(
            all(
                tool["approval_mode"] == "approve"
                for tool in bundled["tools"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
