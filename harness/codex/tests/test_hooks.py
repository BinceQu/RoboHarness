from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tool_policy", ROOT / "hooks" / "tool_policy.py"
)
assert SPEC is not None and SPEC.loader is not None
TOOL_POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL_POLICY)


def load_hook(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "hooks" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SESSION_START = load_hook("session_start")
SKILL_ROUTER = load_hook("skill_router")


class ToolPolicyHookTests(unittest.TestCase):
    def test_policy_denies_non_embodied_tool(self) -> None:
        result = TOOL_POLICY.decision({"tool_name": "Bash"})
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_allows_behavior_mcp_tool(self) -> None:
        self.assertIsNone(
            TOOL_POLICY.decision(
                {"tool_name": "mcp__behavior_v2__adjust_chassis"},
            )
        )

    def test_policy_rejects_noncanonical_behavior_server_name(self) -> None:
        result = TOOL_POLICY.decision(
            {"tool_name": "mcp__behavior-v2__adjust_chassis"}
        )
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_rejects_same_tool_name_from_other_server(self) -> None:
        result = TOOL_POLICY.decision({"tool_name": "mcp__other__adjust_chassis"})
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_rejects_server_name_with_embodied_substring(self) -> None:
        result = TOOL_POLICY.decision(
            {"tool_name": "mcp__other_behavior_v2__adjust_chassis"}
        )
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_denies_malformed_input(self) -> None:
        result = TOOL_POLICY.decision({})
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_denies_malformed_function_name(self) -> None:
        result = TOOL_POLICY.decision(
            {"tool_name": "mcp__behavior-v2__../../shell"}
        )
        self.assertEqual(
            result["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_policy_denies_every_non_behavior_tool_family(self) -> None:
        denied_tools = (
            "shell",
            "Bash",
            "functions.exec",
            "functions.wait",
            "apply_patch",
            "view_image",
            "web_search",
            "browser.open",
            "mcp__filesystem__write_file",
            "mcp__other__capture_head_camera",
            "multi_agent_v1__spawn_agent",
        )
        for tool_name in denied_tools:
            with self.subTest(tool_name=tool_name):
                result = TOOL_POLICY.decision({"tool_name": tool_name})
                self.assertEqual(
                    result["hookSpecificOutput"]["permissionDecision"],
                    "deny",
                )

    def test_policy_allows_native_skill_tools(self) -> None:
        allowed = (
            "Skill",
            "skill",
            "skills.list",
            "skills.read",
            "functions.skill",
            "functions.skills.read",
        )
        for tool_name in allowed:
            with self.subTest(tool_name=tool_name):
                self.assertIsNone(TOOL_POLICY.decision({"tool_name": tool_name}))


class SkillContextHookTests(unittest.TestCase):
    def test_session_context_includes_baseline_skill_body(self) -> None:
        context = SESSION_START.build_context(ROOT)
        self.assertIn(
            '<activated_skill name="behavior-v2-baseline">', context
        )
        self.assertIn("For each decision:", context)
        self.assertIn("Call `activate_skill` with exactly one `name`", context)
        self.assertIn(
            "End this Codex session only when (1) the robot has fallen",
            context,
        )
        self.assertIn("forbid `reset_body`", context)
        self.assertIn("not an episode reset", context)
        self.assertIn("one exit: its subtask is complete", context)
        self.assertIn("exits that Skill only", context)
        self.assertNotIn("Stop acting on transport failure", context)
        self.assertIn("`pick-up-object`:", context)
        self.assertIn("`place-object-in-container`:", context)
        self.assertNotIn("`stand-trash-can-upright`:", context)
        self.assertIn("`cut-object`:", context)
        self.assertIn("`navigate-to-target`:", context)
        self.assertIn("`close-box`:", context)
        self.assertNotIn("`look-around-find-object`:", context)
        self.assertNotIn('<activated_skill name="pick-up-object">', context)
        self.assertNotIn('<activated_skill name="open-doors-and-drawers">', context)

    def test_session_start_reinjects_active_task_skill_body(self) -> None:
        context = SESSION_START.build_context(
            ROOT, active_skill="open-doors-and-drawers"
        )
        self.assertIn(
            '<activated_skill name="behavior-v2-baseline">', context
        )
        self.assertIn(
            '<activated_skill name="open-doors-and-drawers">', context
        )
        self.assertIn("open-doors-and-drawers", context)

    def test_session_start_does_not_cut_monitor_cards(self) -> None:
        source = (ROOT / "hooks" / "session_start.py").read_text(encoding="utf-8")
        self.assertNotIn("new_attempt=True", source)
        self.assertNotIn("publish_user_prompt", source)



    def test_discover_skills_lists_task_skills_only(self) -> None:
        names = [skill.name for skill in SKILL_ROUTER.discover_skills(ROOT)]
        self.assertEqual(
            names,
            [
                "close-box",
                "cut-object",
                "navigate-to-target",
                "open-doors-and-drawers",
                "pick-up-object",
                "place-object-in-container",
                "traverse-narrow-passages",
            ],
        )
        self.assertNotIn("find-object", names)
        self.assertNotIn("look-around-find-object", names)
        self.assertNotIn("stand-trash-can-upright", names)

    def test_user_prompt_does_not_inject_task_skills(self) -> None:
        prompts = (
            "Put the three cans of soda from the living room inside the trash "
            "can in the kitchen. The robot already holds one can; put it in "
            "the bin next.",
            "Pick up every toy, game, puzzle, and ball and put each one "
            "inside the toy box.",
            "Open the refrigerator door so the robot can access it.",
            "Navigate the robot fully through the open doorway directly ahead.",
            "Use place-object-in-container for this turn.",
            "Capture the head camera once and report success.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt[:48]):
                self.assertIsNone(SKILL_ROUTER.decision({"prompt": prompt}, ROOT))

















































    def test_publish_user_prompt_posts_plain_user_text(self) -> None:
        captured: dict[str, object] = {}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=0):
            captured["url"] = request.full_url
            captured["body"] = request.data
            captured["timeout"] = timeout
            return _Response()

        ok = SKILL_ROUTER.publish_user_prompt(
            "只捡起地上的罐子",
            base_url="http://127.0.0.1:15051",
            opener=opener,
        )
        self.assertTrue(ok)
        self.assertEqual(
            captured["url"],
            "http://127.0.0.1:15051/api/agent_monitor/prompt",
        )
        payload = __import__("json").loads(captured["body"])
        self.assertEqual(payload["prompt"], "只捡起地上的罐子")
        self.assertEqual(payload["user_prompt"], "只捡起地上的罐子")
        self.assertEqual(payload["loaded_skills"], ["behavior-v2-baseline"])
        self.assertNotIn("new_attempt", payload)

    def test_publish_user_prompt_can_mark_new_attempt(self) -> None:
        captured: dict[str, object] = {}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=0):
            captured["body"] = request.data
            return _Response()

        ok = SKILL_ROUTER.publish_user_prompt(
            "只捡起地上的罐子",
            base_url="http://127.0.0.1:15051",
            new_attempt=True,
            opener=opener,
        )
        self.assertTrue(ok)
        payload = __import__("json").loads(captured["body"])
        self.assertTrue(payload["new_attempt"])

    def test_publish_user_prompt_skips_when_base_url_missing(self) -> None:
        old = os.environ.pop("BEHAVIOR_BASE_URL", None)
        old_iface = os.environ.pop("BEHAVIOR_INTERFACE_URL", None)
        old_runtime = os.environ.get("XDG_RUNTIME_DIR")
        empty = tempfile.TemporaryDirectory()
        os.environ["XDG_RUNTIME_DIR"] = empty.name
        try:
            self.assertFalse(SKILL_ROUTER.publish_user_prompt("hello", base_url=""))
        finally:
            empty.cleanup()
            if old is None:
                os.environ.pop("BEHAVIOR_BASE_URL", None)
            else:
                os.environ["BEHAVIOR_BASE_URL"] = old
            if old_iface is None:
                os.environ.pop("BEHAVIOR_INTERFACE_URL", None)
            else:
                os.environ["BEHAVIOR_INTERFACE_URL"] = old_iface
            if old_runtime is None:
                os.environ.pop("XDG_RUNTIME_DIR", None)
            else:
                os.environ["XDG_RUNTIME_DIR"] = old_runtime

    def test_extract_user_prompt_accepts_nested_fields(self) -> None:
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt({"user_prompt": "只捡罐子"}),
            "只捡罐子",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt({"prompt": {"text": " nested "}}),
            "nested",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt(
                {"prompt": [{"type": "input_text", "text": "从床边捡起盒子"}]}
            ),
            "从床边捡起盒子",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt(
                {"event": {"user_prompt": "只捡红盒子"}}
            ),
            "只捡红盒子",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "01abc",
                    "turn_id": "t1",
                    "prompt": "只捡罐子",
                    "cwd": "/tmp",
                    "model": "gpt",
                    "permission_mode": "bypassPermissions",
                }
            ),
            "只捡罐子",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt(
                {
                    "messages": [
                        {"role": "system", "content": "ignore"},
                        {"role": "user", "content": [{"type": "text", "text": "从床边捡起盒子"}]},
                    ]
                }
            ),
            "从床边捡起盒子",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_user_prompt(
                {
                    "hook_event_name": "SessionStart",
                    "source": "startup",
                    "session_id": "01abc",
                    "cwd": "/tmp",
                }
            ),
            "",
        )

    def test_resolve_active_task_skill_uses_stamp_then_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp_dir = Path(tmp) / "embodied-codex"
            stamp_dir.mkdir(parents=True)
            (stamp_dir / "active_task_skill.unit-hook-stamp").write_text(
                "open-doors-and-drawers\n", encoding="utf-8"
            )
            old_sid = os.environ.get("BEHAVIOR_SESSION_ID")
            old_runtime = os.environ.get("XDG_RUNTIME_DIR")
            os.environ["BEHAVIOR_SESSION_ID"] = "unit-hook-stamp"
            os.environ["XDG_RUNTIME_DIR"] = tmp
            try:
                self.assertEqual(
                    SKILL_ROUTER.resolve_active_task_skill(ROOT),
                    "open-doors-and-drawers",
                )
                self.assertEqual(
                    SKILL_ROUTER.loaded_skill_names(
                        "",
                        ROOT,
                        invoked=SKILL_ROUTER.resolve_active_task_skill(ROOT),
                    ),
                    ["behavior-v2-baseline", "open-doors-and-drawers"],
                )
            finally:
                if old_sid is None:
                    os.environ.pop("BEHAVIOR_SESSION_ID", None)
                else:
                    os.environ["BEHAVIOR_SESSION_ID"] = old_sid
                if old_runtime is None:
                    os.environ.pop("XDG_RUNTIME_DIR", None)
                else:
                    os.environ["XDG_RUNTIME_DIR"] = old_runtime

        class _Response:
            status = 200

            def read(self):
                return (
                    b'{"loaded_skills":["behavior-v2-baseline",'
                    b'"pick-up-object"]}'
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=0):
            self.assertIn("/api/agent_monitor", request.full_url)
            return _Response()

        with tempfile.TemporaryDirectory() as tmp:
            old_sid = os.environ.pop("BEHAVIOR_SESSION_ID", None)
            old_runtime = os.environ.get("XDG_RUNTIME_DIR")
            os.environ["XDG_RUNTIME_DIR"] = tmp
            try:
                self.assertEqual(
                    SKILL_ROUTER.resolve_active_task_skill(
                        ROOT,
                        query_monitor=True,
                        base_url="http://127.0.0.1:15060",
                        opener=opener,
                    ),
                    "pick-up-object",
                )
            finally:
                if old_sid is None:
                    os.environ.pop("BEHAVIOR_SESSION_ID", None)
                else:
                    os.environ["BEHAVIOR_SESSION_ID"] = old_sid
                if old_runtime is None:
                    os.environ.pop("XDG_RUNTIME_DIR", None)
                else:
                    os.environ["XDG_RUNTIME_DIR"] = old_runtime

    def test_loaded_skill_names_are_baseline_until_native_invoke(self) -> None:
        trash = (
            "Find one ungrasped can of soda. Follow the activated "
            "search/navigation, pick-up-object, and place-object-in-container "
            "policies."
        )
        self.assertEqual(
            SKILL_ROUTER.loaded_skill_names(trash, ROOT),
            ["behavior-v2-baseline"],
        )
        self.assertEqual(
            SKILL_ROUTER.loaded_skill_names("", ROOT),
            ["behavior-v2-baseline"],
        )
        self.assertEqual(
            SKILL_ROUTER.loaded_skill_names(
                trash, ROOT, invoked="pick-up-object"
            ),
            ["behavior-v2-baseline", "pick-up-object"],
        )

    def test_extract_invoked_skill_name_from_native_tool_payload(self) -> None:
        self.assertEqual(
            SKILL_ROUTER.extract_invoked_skill_name(
                {
                    "tool_name": "skills.read",
                    "tool_input": {"package": "pick-up-object"},
                },
                ROOT,
            ),
            "pick-up-object",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_invoked_skill_name(
                {
                    "tool_name": "Skill",
                    "tool_input": {"skill": "place-object-in-container"},
                },
                ROOT,
            ),
            "place-object-in-container",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_invoked_skill_name(
                {
                    "tool_name": "skills.read",
                    "tool_input": {
                        "package": "embodied-codex-plugin",
                        "resource": "skills/open-doors-and-drawers/SKILL.md",
                    },
                },
                ROOT,
            ),
            "open-doors-and-drawers",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_invoked_skill_name(
                {"tool_name": "skills.list", "tool_input": {}},
                ROOT,
            ),
            "",
        )
        self.assertEqual(
            SKILL_ROUTER.extract_invoked_skill_name(
                {
                    "tool_name": "Skill",
                    "tool_input": {"skill": "find-object"},
                },
                ROOT,
            ),
            "",
        )

    def test_publish_user_prompt_can_omit_loaded_skills(self) -> None:
        captured: dict[str, object] = {}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=0):
            captured["body"] = request.data
            return _Response()

        ok = SKILL_ROUTER.publish_user_prompt(
            "继续开门",
            base_url="http://127.0.0.1:15060",
            plugin_root=ROOT,
            include_skills=False,
            opener=opener,
        )
        self.assertTrue(ok)
        payload = __import__("json").loads(captured["body"])
        self.assertEqual(payload["prompt"], "继续开门")
        self.assertNotIn("loaded_skills", payload)

    def test_publish_user_prompt_includes_loaded_skills(self) -> None:
        captured: dict[str, object] = {}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=0):
            captured["body"] = request.data
            return _Response()

        ok = SKILL_ROUTER.publish_user_prompt(
            "Use pick-up-object then place-object-in-container.",
            base_url="http://127.0.0.1:15060",
            plugin_root=ROOT,
            invoked="pick-up-object",
            opener=opener,
        )
        self.assertTrue(ok)
        payload = __import__("json").loads(captured["body"])
        self.assertEqual(
            payload["loaded_skills"],
            [
                "behavior-v2-baseline",
                "pick-up-object",
            ],
        )

    def test_stamped_prompt_fills_empty_session_start_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp_dir = Path(tmp) / "embodied-codex"
            stamp_dir.mkdir(parents=True)
            prompt_file = stamp_dir / "user_prompt.luna-attempt6"
            prompt_file.write_text("Find one ungrasped can of soda.\n", encoding="utf-8")
            old_sid = os.environ.get("BEHAVIOR_SESSION_ID")
            old_file = os.environ.get("BEHAVIOR_USER_PROMPT_FILE")
            old_runtime = os.environ.get("XDG_RUNTIME_DIR")
            os.environ["BEHAVIOR_SESSION_ID"] = "luna-attempt6"
            os.environ["BEHAVIOR_USER_PROMPT_FILE"] = str(prompt_file)
            os.environ["XDG_RUNTIME_DIR"] = tmp
            try:
                self.assertEqual(
                    SKILL_ROUTER.resolve_prompt_for_publish(
                        {
                            "hook_event_name": "SessionStart",
                            "source": "startup",
                        }
                    ),
                    "Find one ungrasped can of soda.",
                )
            finally:
                if old_sid is None:
                    os.environ.pop("BEHAVIOR_SESSION_ID", None)
                else:
                    os.environ["BEHAVIOR_SESSION_ID"] = old_sid
                if old_file is None:
                    os.environ.pop("BEHAVIOR_USER_PROMPT_FILE", None)
                else:
                    os.environ["BEHAVIOR_USER_PROMPT_FILE"] = old_file
                if old_runtime is None:
                    os.environ.pop("XDG_RUNTIME_DIR", None)
                else:
                    os.environ["XDG_RUNTIME_DIR"] = old_runtime

    def test_resolve_base_url_reads_stamp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp_dir = Path(tmp) / "embodied-codex"
            stamp_dir.mkdir(parents=True)
            (stamp_dir / "behavior_base_url").write_text(
                "http://127.0.0.1:15061\n", encoding="utf-8"
            )
            old = os.environ.pop("BEHAVIOR_BASE_URL", None)
            old_iface = os.environ.pop("BEHAVIOR_INTERFACE_URL", None)
            old_runtime = os.environ.get("XDG_RUNTIME_DIR")
            os.environ["XDG_RUNTIME_DIR"] = tmp
            try:
                self.assertEqual(
                    SKILL_ROUTER.resolve_base_url(),
                    "http://127.0.0.1:15061",
                )
            finally:
                if old is None:
                    os.environ.pop("BEHAVIOR_BASE_URL", None)
                else:
                    os.environ["BEHAVIOR_BASE_URL"] = old
                if old_iface is None:
                    os.environ.pop("BEHAVIOR_INTERFACE_URL", None)
                else:
                    os.environ["BEHAVIOR_INTERFACE_URL"] = old_iface
                if old_runtime is None:
                    os.environ.pop("XDG_RUNTIME_DIR", None)
                else:
                    os.environ["XDG_RUNTIME_DIR"] = old_runtime


if __name__ == "__main__":
    unittest.main()
