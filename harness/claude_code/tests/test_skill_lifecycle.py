from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from embodied_claude_code import launcher, skills, skill_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hooks"))
import session_start
import skill_lifecycle
import skill_router
import tool_policy


class SkillLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {
            "XDG_RUNTIME_DIR": self.tmp.name,
            "BEHAVIOR_SESSION_ID": "lifecycle-test",
            "BEHAVIOR_BASE_URL": "",
            "BEHAVIOR_INTERFACE_URL": "",
            "CLAUDE_PLUGIN_ROOT": str(ROOT),
            "PYTHONPATH": str(ROOT / "src"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        monitor_patch = patch.object(skills, "publish_loaded_skills", return_value=True)
        hook_patch = patch.object(skill_router, "publish_user_prompt", return_value=True)
        self.monitor = monitor_patch.start()
        self.hook_monitor = hook_patch.start()
        self.addCleanup(monitor_patch.stop)
        self.addCleanup(hook_patch.stop)

    def native(self, name, **response):
        return skill_lifecycle.handle({
            "hook_event_name": "PostToolUse", "tool_name": "Skill",
            "tool_input": {"skill": "embodied-claude-code:" + name},
            "tool_response": response or {"success": True},
        }, ROOT)

    def test_every_task_can_complete_and_cancel_with_full_document(self):
        for doc in skills.discover_task_skills(ROOT):
            with self.subTest(name=doc.name):
                active = skills.activate_skill(doc.name, ROOT)
                self.assertTrue(active["ok"])
                self.assertEqual(active["body"], doc.path.read_text().strip())
                self.assertIn(f'name="{doc.name}"', doc.text)
                self.assertIn("do\nnot prohibit explicit cancellation or handoff", doc.text)
                result = skills.deactivate_skill(doc.name, "Cancelled; work remains", ROOT)
                self.assertTrue(result["ok"])
                self.assertTrue(result["changed"])
                self.assertIsNone(result["active_skill"])
                self.assertNotIn("success", result)
                self.assertEqual(skills.catalog_payload(ROOT)["active_skill"], None)
                self.assertEqual(skill_router.resolve_active_task_skill(ROOT), "")
                self.monitor.assert_called_with(session_id="", base_url="")

    def test_native_and_mcp_share_state_without_process_cache(self):
        skills.activate_skill("pick-up-object", ROOT)
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")
        context = self.native("place-object-in-container")
        self.assertEqual(skills.current_task_skill(ROOT), "place-object-in-container")
        self.assertIn('"active_task_skill":"place-object-in-container"', json.dumps(context).replace('\\"', '"'))
        result = skills.deactivate_skill("place-object-in-container", root=ROOT)
        self.assertTrue(result["ok"])
        self.assertEqual(skill_router.resolve_active_task_skill(ROOT), "")
        self.native("pick-up-object")
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")

    def test_deactivation_is_idempotent_and_mismatch_cannot_close_another_skill(self):
        skills.activate_skill("pick-up-object", ROOT)
        self.assertFalse(skills.deactivate_skill("place-object-in-container", root=ROOT)["ok"])
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")
        first = skills.deactivate_skill("pick-up-object", root=ROOT)
        second = skills.deactivate_skill("pick-up-object", root=ROOT)
        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(first["revision"], second["revision"])
        for name in ("behavior-v2-baseline", "unknown", "", "../pick-up-object"):
            self.assertFalse(skills.deactivate_skill(name, root=ROOT)["ok"])
        self.assertIsNone(skill_state.read_state().active_task_skill)

    def test_explicit_inactive_survives_stale_legacy_monitor_and_fresh_process(self):
        legacy = skill_state.legacy_stamp_paths()[0]
        legacy.parent.mkdir(parents=True)
        legacy.write_text("pick-up-object\n")
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")
        skills.deactivate_skill("pick-up-object", root=ROOT)
        legacy.write_text("place-object-in-container\n")
        def reject_monitor(*args, **kwargs):
            raise AssertionError("must not restore from monitor")

        self.assertEqual(skill_router.resolve_active_task_skill(
            ROOT, query_monitor=True, opener=reject_monitor
        ), "")
        for event in ("compact", "resume", "startup", "clear"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "hooks/session_start.py")],
                input=json.dumps({"hook_event_name": "SessionStart", "source": event}),
                text=True, capture_output=True, env=os.environ.copy(), timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            context = json.loads(completed.stdout)["hookSpecificOutput"]["additionalContext"]
            self.assertTrue(context.startswith('<task_skill_state>{"active_task_skill":null'))
            self.assertNotIn('<activated_skill name="pick-up-object">', context)
            self.assertNotIn('<activated_skill name="place-object-in-container">', context)

    def test_session_isolation_and_launcher_new_attempt(self):
        skills.activate_skill("pick-up-object", ROOT, session_id="one")
        skills.activate_skill("place-object-in-container", ROOT, session_id="two")
        skills.deactivate_skill("pick-up-object", root=ROOT, session_id="one")
        self.assertEqual(skills.current_task_skill(ROOT, "two"), "place-object-in-container")
        with patch.object(launcher, "Path", side_effect=lambda value: (
            Path(self.tmp.name) / "fallback" if str(value) == "/tmp/embodied-claude-code" else Path(value)
        )):
            launcher._stamp_runtime("http://127.0.0.1:1", "two", "new task", [])
        self.assertIsNone(skill_state.read_state("two").active_task_skill)
        self.assertIsNone(skill_state.read_state("one").active_task_skill)

    def test_failed_native_invocation_and_pre_hook_do_not_activate(self):
        for response in ({"isError": True}, {"is_error": True}, {"success": False}, {"ok": False}):
            self.assertIsNone(self.native("pick-up-object", **response))
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps({"tool_name": "Skill", "tool_input": {"skill": "pick-up-object"}}))), patch("sys.stdout", output):
            tool_policy.main()
        self.assertIsNone(skill_state.read_state())
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(skill_router.extract_invoked_skill_name({
            "tool_input": {"skill": "other", "args": "pick-up-object"},
            "tool_response": "pick-up-object",
        }, ROOT), "")

    def test_launcher_resume_preserves_state_without_new_monitor_attempt(self):
        skills.activate_skill("pick-up-object", ROOT)
        with (
            patch.dict(os.environ, {"EMBODIED_ANTHROPIC_BASE_URL": "http://model.local"}),
            patch.object(launcher, "Path", side_effect=lambda value: (
                Path(self.tmp.name) / "fallback" if str(value) == "/tmp/embodied-claude-code" else Path(value)
            )),
            patch.object(launcher, "_get_json"),
            patch.object(launcher, "_post_monitor") as monitor,
            patch.object(launcher, "_claude_executable", return_value="/fake/claude"),
            patch.object(launcher.subprocess, "run") as run,
        ):
            run.return_value.returncode = 0
            for option in ("--resume=session-uuid", "--continue", "-c", "-r"):
                self.assertEqual(launcher.launch(["--", "exec", option, "continue task"]), 0)
                self.assertTrue(monitor.call_args.kwargs["resume"])
                self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")
            with patch.dict(os.environ, {"BEHAVIOR_SESSION_ID": ""}):
                with self.assertRaises(launcher.UsageError):
                    launcher.launch(["--", "exec", "--continue", "continue task"])

        with patch.object(launcher, "_no_proxy_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.status = 200
            launcher._post_monitor("http://local", "continue task", "lifecycle-test", [], resume=True)
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertFalse(body["new_attempt"])
            self.assertEqual(body["loaded_skills"], ["behavior-v2-baseline", "pick-up-object"])

    def test_slash_expansion_requests_explicit_commit_without_false_activation(self):
        result = skill_lifecycle.handle({
            "hook_event_name": "UserPromptExpansion", "expansion_type": "slash_command",
            "command_name": "embodied-claude-code:pick-up-object", "command_source": "plugin",
        }, ROOT)
        self.assertIn('call activate_skill with name="pick-up-object"', result["hookSpecificOutput"]["additionalContext"])
        self.assertIsNone(skill_state.read_state())
        skills.activate_skill("pick-up-object", ROOT)
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")

    def test_failed_monitor_publication_retries_baseline_before_next_robot_tool(self):
        skills.activate_skill("pick-up-object", ROOT)
        self.monitor.return_value = False
        result = skills.deactivate_skill("pick-up-object", root=ROOT)
        self.assertTrue(result["ok"])
        self.assertFalse(result["monitor_synced"])
        self.assertFalse(skill_state.read_state().monitor_synced)
        with patch("sys.stdin", io.StringIO('{"tool_name":"mcp__behavior-v2__capture_head_camera"}')):
            tool_policy.main()
        self.hook_monitor.assert_called_with("", invoked="")
        self.assertTrue(skill_state.read_state().monitor_synced)
        self.assertIsNone(skill_state.read_state().active_task_skill)

    def test_atomic_write_failure_preserves_prior_state(self):
        skills.activate_skill("pick-up-object", ROOT)
        original = skill_state.state_path().read_bytes()
        with patch.object(skill_state.os, "replace", side_effect=OSError("disk full")):
            result = skills.deactivate_skill("pick-up-object", root=ROOT)
        self.assertFalse(result["ok"])
        self.assertEqual(skill_state.state_path().read_bytes(), original)
        self.assertEqual(skills.current_task_skill(ROOT), "pick-up-object")

    def test_corrupt_state_does_not_fall_back_to_old_skill(self):
        skills.activate_skill("pick-up-object", ROOT)
        skill_state.state_path().write_text('{"version": 1}')
        with self.assertRaises(ValueError):
            skill_router.resolve_active_task_skill(ROOT, query_monitor=True)
        self.assertFalse(skills.deactivate_skill("pick-up-object", root=ROOT)["ok"])

    def test_lifecycle_publications_are_ordered_across_concurrent_calls(self):
        entered, release = threading.Event(), threading.Event()
        published = []

        def monitor(name="", **kwargs):
            if name == "pick-up-object":
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test did not release publisher")
            published.append(name)
            return True

        with patch.object(skills, "publish_loaded_skills", side_effect=monitor):
            with ThreadPoolExecutor(max_workers=2) as pool:
                activation = pool.submit(skills.activate_skill, "pick-up-object", ROOT)
                self.assertTrue(entered.wait(5))
                deactivation = pool.submit(skills.deactivate_skill, "pick-up-object", "handoff", ROOT)
                release.set()
                self.assertTrue(activation.result(5)["ok"])
                self.assertTrue(deactivation.result(5)["ok"])
        self.assertEqual(published, ["pick-up-object", ""])
        self.assertIsNone(skill_state.read_state().active_task_skill)


if __name__ == "__main__":
    unittest.main()
