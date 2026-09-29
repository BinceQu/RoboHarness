from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_claude_code import skills


ROOT = Path(__file__).resolve().parents[1]


class ActivateSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_sid = os.environ.get("BEHAVIOR_SESSION_ID")
        self._old_xdg = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["BEHAVIOR_SESSION_ID"] = "unit-test-skills"
        os.environ["XDG_RUNTIME_DIR"] = self._tmp.name
        publisher = patch.object(skills, "publish_loaded_skills", return_value=True)
        publisher.start()
        self.addCleanup(publisher.stop)

    def tearDown(self) -> None:
        if self._old_sid is None:
            os.environ.pop("BEHAVIOR_SESSION_ID", None)
        else:
            os.environ["BEHAVIOR_SESSION_ID"] = self._old_sid
        if self._old_xdg is None:
            os.environ.pop("XDG_RUNTIME_DIR", None)
        else:
            os.environ["XDG_RUNTIME_DIR"] = self._old_xdg
        self._tmp.cleanup()

    def test_empty_name_lists_task_skills_only(self) -> None:
        payload = skills.activate_skill("", root=ROOT)
        names = [item["name"] for item in payload["skills"]]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "list")
        self.assertEqual(
            names,
            [
                "open-doors-and-drawers",
                "pick-up-object",
                "place-object-in-container",
            ],
        )
        self.assertNotIn("behavior-v2-baseline", names)
        self.assertNotIn("find-object", names)
        self.assertNotIn("look-around-find-object", names)
        self.assertNotIn("stand-trash-can-upright", names)
        self.assertNotIn("body", payload)

    def test_activate_loads_one_skill_body(self) -> None:
        payload = skills.activate_skill("pick-up-object", root=ROOT)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "activated")
        self.assertEqual(payload["name"], "pick-up-object")
        self.assertIn("set_arm_to_grasp_position", payload["body"])
        self.assertNotIn("Release and verify", payload["body"])

    def test_disabled_skills_cannot_activate(self) -> None:
        # Retained source documents can be disabled by their frontmatter.
        for name in ("cut-object", "navigate-to-target", "close-box",
                     "traverse-narrow-passages", "look-around-find-object"):
            with self.subTest(name=name):
                payload = skills.activate_skill(name, root=ROOT)
                self.assertFalse(payload["ok"])
                self.assertIn(f"Unknown task Skill `{name}`", payload["error"])
                self.assertEqual(skills.current_task_skill(ROOT), "")

    def test_hidden_stand_trash_cannot_activate(self) -> None:
        payload = skills.activate_skill("stand-trash-can-upright", root=ROOT)
        self.assertFalse(payload["ok"])
        self.assertIn("Unknown task Skill `stand-trash-can-upright`", payload["error"])
        # 文件仍在磁盘上，只是不进目录、不能 activate。
        skill_path = ROOT / "skills" / "stand-trash-can-upright" / "SKILL.md"
        self.assertTrue(skill_path.is_file())
        body = skill_path.read_text(encoding="utf-8")
        self.assertIn("disable-model-invocation: true", body)
        self.assertIn("`z=0.4`", body)

    def test_activate_rejects_baseline_and_unknown(self) -> None:
        baseline = skills.activate_skill("behavior-v2-baseline", root=ROOT)
        unknown = skills.activate_skill("search-navigation", root=ROOT)
        old_find = skills.activate_skill("find-object", root=ROOT)
        self.assertFalse(baseline["ok"])
        self.assertFalse(unknown["ok"])
        self.assertFalse(old_find["ok"])
        self.assertIn("Unknown task Skill `find-object`", old_find["error"])
        self.assertEqual(skills.current_task_skill(ROOT), "")

    def test_catalog_markdown_is_name_and_description_only(self) -> None:
        text = skills.render_catalog_markdown(ROOT)
        self.assertIn("`pick-up-object`:", text)
        self.assertNotIn("stand-trash-can-upright", text)
        self.assertNotIn("set_arm_to_grasp_position", text)

    def test_activate_stamps_active_task_skill_for_hooks(self) -> None:
        payload = skills.activate_skill("open-doors-and-drawers", root=ROOT)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            skills.load_active_task_skill_stamp(),
            "open-doors-and-drawers",
        )
        self.assertEqual(
            skills.current_task_skill(ROOT),
            "open-doors-and-drawers",
        )
