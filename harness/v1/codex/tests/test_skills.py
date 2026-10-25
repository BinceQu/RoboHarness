from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from embodied_codex import skills


ROOT = Path(__file__).resolve().parents[1]


class ActivateSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        skills._ACTIVE_TASK_SKILL = ""
        self._tmp = tempfile.TemporaryDirectory()
        self._old_sid = os.environ.get("BEHAVIOR_SESSION_ID")
        self._old_xdg = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["BEHAVIOR_SESSION_ID"] = "unit-test-skills"
        os.environ["XDG_RUNTIME_DIR"] = self._tmp.name

    def tearDown(self) -> None:
        for path in skills._active_skill_stamp_paths():
            try:
                path.unlink()
            except OSError:
                pass
        skills._ACTIVE_TASK_SKILL = ""
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
                "close-box",
                "cut-object",
                "navigate-to-target",
                "open-doors-and-drawers",
                "pick-up-object",
                "place-object-in-container",
                "traverse-narrow-passages",
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

    def test_activate_loads_new_cut_and_find_skills(self) -> None:
        cut = skills.activate_skill("cut-object", root=ROOT)
        self.assertTrue(cut["ok"])
        self.assertIn("cutting_tool_point", cut["body"])
        self.assertIn("target_object_point", cut["body"])
        self.assertIn("Call `track_object_distance`", cut["body"])
        self.assertIn("xyz_in_robot_base_coord_m", cut["body"])
        self.assertIn("`adjust_left_eef_pose_in_head_frame`", cut["body"])
        self.assertIn("`adjust_right_eef_pose_in_head_frame`", cut["body"])
        self.assertNotIn("`cut_object`", cut["body"])

        gone = skills.activate_skill("look-around-find-object", root=ROOT)
        self.assertFalse(gone["ok"])
        self.assertIn("Unknown task Skill `look-around-find-object`", gone["error"])

        nav = skills.activate_skill("navigate-to-target", root=ROOT)
        self.assertTrue(nav["ok"])
        self.assertEqual(nav["name"], "navigate-to-target")
        nav_body = " ".join(nav["body"].split())
        self.assertIn("## A. Marked object or place", nav["body"])
        self.assertIn("## B. Unseen object or named place", nav["body"])
        self.assertIn("no clear structure", nav_body)
        self.assertIn("`spin=90`", nav_body)
        self.assertIn("`mark_on_map`", nav_body)
        self.assertIn("traverse-narrow-passages", nav_body)
        self.assertNotIn("look-around-find-object", nav_body)
        self.assertIn("top-right of that head image", nav_body)

    def test_activate_loads_close_box(self) -> None:
        payload = skills.activate_skill("close-box", root=ROOT)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["name"], "close-box")
        body = " ".join(payload["body"].split())
        self.assertIn(
            "Close a box that has a lid. Do not use on objects without a lid.",
            payload["body"],
        )
        self.assertIn("`lid_edge`", body)
        self.assertIn("`body_edge`", body)
        self.assertIn("`track_object_distance`", body)
        self.assertIn("`plan_grasp_point_filter_rgbd_lite`", body)
        self.assertIn("`move_to_reach_point`", body)
        self.assertIn("`exec_plan_pose`", body)
        self.assertIn("`close_gripper`", body)
        self.assertIn("xyz_in_robot_base_coord_m", body)
        self.assertIn("judge which way", body)
        self.assertIn("Do not follow a fixed", body)
        self.assertIn("acute angle", body)
        self.assertIn("`open_gripper`", body)
        self.assertIn("go back to step 1", body)
        self.assertNotIn("`z=0.2`", body)
        self.assertNotIn("`z=-0.2`", body)
        self.assertNotIn("0.5*", body)
        self.assertNotIn("open-doors-and-drawers", body)

    def test_hidden_stand_trash_cannot_activate(self) -> None:
        payload = skills.activate_skill("stand-trash-can-upright", root=ROOT)
        self.assertFalse(payload["ok"])
        self.assertIn("Unknown task Skill `stand-trash-can-upright`", payload["error"])
        skill_path = ROOT / "skills" / "stand-trash-can-upright" / "SKILL.md"
        self.assertTrue(skill_path.is_file())
        body = skill_path.read_text(encoding="utf-8")
        self.assertIn("`z=0.4`", body)

    def test_activate_rejects_baseline_and_unknown(self) -> None:
        baseline = skills.activate_skill("behavior-v2-baseline", root=ROOT)
        unknown = skills.activate_skill("search-navigation", root=ROOT)
        old_find = skills.activate_skill("find-object", root=ROOT)
        self.assertFalse(baseline["ok"])
        self.assertFalse(unknown["ok"])
        self.assertFalse(old_find["ok"])
        self.assertIn("Unknown task Skill `find-object`", old_find["error"])
        self.assertEqual(skills._ACTIVE_TASK_SKILL, "")

    def test_catalog_markdown_is_name_and_description_only(self) -> None:
        text = skills.render_catalog_markdown(ROOT)
        self.assertIn("`pick-up-object`:", text)
        self.assertNotIn("set_arm_to_grasp_position", text)

    def test_activate_stamps_active_task_skill_for_hooks(self) -> None:
        payload = skills.activate_skill("open-doors-and-drawers", root=ROOT)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            skills.load_active_task_skill_stamp(),
            "open-doors-and-drawers",
        )
        skills._ACTIVE_TASK_SKILL = ""
        self.assertEqual(
            skills.current_task_skill(ROOT),
            "open-doors-and-drawers",
        )
