from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


CLAUDE_ROOT = Path(__file__).resolve().parents[1]
CODEX_ROOT = CLAUDE_ROOT.parent / "embodied-codex"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _adapt_names(text: str) -> str:
    return (
        text.replace("Embodied Codex", "Embodied Claude Code")
        .replace("`behavior_v2`", "`behavior-v2`")
        .replace("Codex session", "Claude Code session")
    )


def _collapsed(text: str) -> str:
    text = re.sub(r"<!-- claude-task-lifecycle:start -->.*?<!-- claude-task-lifecycle:end -->",
                  "", text, flags=re.DOTALL)
    text = text.replace(
        "material margin. Re-locate that candidate in the latest clickable head image\n"
        "   after the measurement, then pass its current `image_id`, `u`, and `v` pixel\n"
        "   tuple to grasp planning. Do not reuse the pre-measurement image tuple when\n"
        "   tracking has returned a newer image. If no candidate passes both visual identity and",
        "material margin, and pass that candidate's exact original `image_id`, `u`,\n"
        "   and `v` to grasp planning.    If no candidate passes both visual identity and",
    )
    text = text.replace(
        "`observation_sequence` and `xyz_in_robot_base_coord_m` describe the current\n"
        "metric update. Image `u`/`v` is intentionally omitted from persistent tracking;\n"
        "re-locate any new image point in the latest clickable 720 x 720 head image.",
        "`observation_sequence`, current `u`/`v`, and\n"
        "`xyz_in_robot_base_coord_m` describe the current update.",
    )
    text = text.replace(
        "bind, if the holding EEF moves but that object's measured track\n"
        "(`xyz_in_robot_base_coord_m`) stays essentially unchanged, that mismatch is",
        "bind, if the holding EEF moves but that object's track (`u`/`v` and\n"
        "`xyz_in_robot_base_coord_m`) stays essentially unchanged, that mismatch is",
    )
    text = re.sub(r"<!-- claude-pixel-contract:start -->.*?<!-- claude-pixel-contract:end -->",
                  "", text, flags=re.DOTALL)
    text = text.replace(
        "Use integer original-image pixels in the latest clickable 720 x 720 head\n"
        "   image: `u` and `v` range from 0 through 719. Do not normalize, rescale, or\n"
        "   use coordinates relative to the appliance, target panel, handle, or a crop.\n"
        "   Treat `image_id`, `u`,",
        "Obey the current image's advertised coordinate system. For relative\n"
        "   coordinates from 0 through 1000, normalize `u` and `v` against the\n"
        "   entire attached image, including all background margins, never against the\n"
        "   appliance, target panel, handle, or an imagined crop. Treat `image_id`, `u`,",
    )
    text = re.sub(r"\s+", " ", text).strip()
    # Explicit lifecycle-only deltas. All physical procedures still compare
    # exactly against Codex after these reviewed substitutions.
    replacements = (
        ("Successful completion requires a fresh head image that shows the lid seated shut",
         "The only Skill exit is a fresh head image that shows the lid seated shut"),
        ('7. Recapture the head camera. When the lid sits shut on the body, call `deactivate_skill` with `name="close-box"` and return to the parent task. If the box is still open, go back to step 1.',
         "7. Recapture the head camera. Exit only when the lid sits shut on the body. If the box is still open, go back to step 1."),
        ('Then call `deactivate_skill` with `name="cut-object"` and return control to the parent task.', ""),
        ("4. Success requires a fresh `capture_head_camera` that shows the", "4. Stop this Skill only when a fresh `capture_head_camera` shows the"),
        ('After either case reaches its goal and required marking is done, call `deactivate_skill` with `name="navigate-to-target"` and return to the parent task.',
         "Either exit leaves this Skill only."),
        ("motion. Successful completion requires a hinged door opened fully", "motion. The only Skill exits are: a hinged door opened fully"),
        ('After the applicable completion and cleanup conditions are satisfied, call `deactivate_skill` with `name="open-doors-and-drawers"` and return to the parent task.', ""),
        ("Skill. Treat this Skill as a picking subtask. Successful completion requires a visually",
         "Skill. Treat this Skill as a picking subtask. The only exit is a visually"),
        ('8. Only after that stow returns, call `deactivate_skill` with `name="pick-up-object"` and return control to the parent task. Follow the parent\'s placement instructions, activating a placement Skill only if needed. That is Skill exit,',
         "8. Only after that stow returns, exit this Skill and return control to the parent task so a matching placement Skill can run. That is Skill exit,"),
        ("controller status labels. Success requires a fresh observation that the", "controller status labels. The only Skill exit is a fresh observation that the"),
        ('Then call `deactivate_skill` with `name="place-object-in-container"`. Return to the parent task; this does not end the Claude Code session. Activate another Skill only when its procedure is needed.',
         "That report is this Skill's only exit. It does not end the Claude Code session. Return to the parent task; if more requested objects remain, activate the next matching Skill."),
        ('Then report this object\'s placement complete and call `deactivate_skill` with `name="place-object-in-container"`. That exits this Skill only.',
         "Then report this object's placement complete. That exits this Skill only."),
        ("- Do not treat the object as fallen if this session has not called `open_gripper` on the holding arm, or if a current wrist image of that gripper shows a red grasp-volume overlay. - If the object falls from the gripper, this Skill cannot finish placement while empty-handed.",
         "- If the object falls from the gripper, this Skill cannot finish placement while empty-handed."),
        ('If it is no longer held, first stow the unused placement arm with `set_arm_to_grasp_position` as required above, then deactivate this Skill with a reason stating that placement is unfinished and the object is outside. Return to the parent task.',
         "If it is no longer held, first stow the unused placement arm with `set_arm_to_grasp_position` as required above, then return to the parent task."),
        ("Success requires that a fresh `capture_head_camera` image shows the", "This Skill has one exit only: a fresh `capture_head_camera` image shows the"),
        ('`mode="reset"` on the holding arm, then call `deactivate_skill` with `name="stand-trash-can-upright"` and return to the parent task. Activate `place-object-in-container` only if its placement procedure is needed.',
         '`mode="reset"` on the holding arm, return to the parent task, and activate `place-object-in-container` if placement still remains.'),
        ('After successful crossing and any required post-exit checks, call `deactivate_skill` with `name="traverse-narrow-passages"` before returning to the parent task or activating a manipulation Skill.', ""),
    )
    for current, previous in replacements:
        text = text.replace(current, previous)
    return re.sub(r"\s+", " ", text).strip()


@unittest.skipUnless(CODEX_ROOT.is_dir(), "sibling embodied-codex source is absent")
class CodexParityTests(unittest.TestCase):
    def test_shared_runtime_modules_are_byte_identical(self) -> None:
        for name in ("client.py", "errors.py", "monitor_cards.py", "py.typed"):
            with self.subTest(name=name):
                self.assertEqual(
                    (CODEX_ROOT / "src" / "embodied_codex" / name).read_bytes(),
                    (CLAUDE_ROOT / "src" / "embodied_claude_code" / name).read_bytes(),
                )

    def test_machine_policy_differs_only_by_profile_namespace(self) -> None:
        codex = json.loads(_read(CODEX_ROOT / "profiles" / "baseline.json"))
        claude = json.loads(_read(CLAUDE_ROOT / "profiles" / "baseline.json"))
        codex["schema_version"] = codex["schema_version"].replace(
            "embodied_codex", "embodied_claude_code"
        )
        self.assertEqual(codex, claude)

    def test_task_skill_catalog_matches_codex(self) -> None:
        codex_names = {
            path.parent.name for path in (CODEX_ROOT / "skills").glob("*/SKILL.md")
        }
        claude_names = {
            path.parent.name for path in (CLAUDE_ROOT / "skills").glob("*/SKILL.md")
        }
        self.assertEqual(codex_names, claude_names)
        self.assertEqual(len(claude_names), 9)

    def test_task_procedures_match_except_claude_grounding_and_lifecycle(self) -> None:
        names = (
            "close-box",
            "cut-object",
            "navigate-to-target",
            "open-doors-and-drawers",
            "place-object-in-container",
            "stand-trash-can-upright",
            "traverse-narrow-passages",
        )
        for name in names:
            with self.subTest(name=name):
                codex = _adapt_names(
                    _collapsed(_read(CODEX_ROOT / "skills" / name / "SKILL.md"))
                )
                claude = _collapsed(
                    _read(CLAUDE_ROOT / "skills" / name / "SKILL.md")
                )
                self.assertEqual(codex, claude)

    def test_baseline_delta_is_only_claude_and_coordinate_contract(self) -> None:
        codex = _collapsed(
            _adapt_names(
                _read(CODEX_ROOT / "skills" / "behavior-v2-baseline" / "SKILL.md")
            )
        )
        claude = _collapsed(
            _read(CLAUDE_ROOT / "skills" / "behavior-v2-baseline" / "SKILL.md")
        )
        claude_catalog = (
            "The `behavior-v2` MCP server derives its tools from the active "
            "interface catalog after machine policy. It also restores the stable "
            "`mark_on_map` contract when an interface profile omits that catalog "
            "entry. Call the resulting tools directly using their advertised "
            "descriptions and input schemas."
        )
        codex_catalog = (
            "The `behavior-v2` MCP server mirrors the active interface tool "
            "catalog. Its MCP tool names and count are the interface tool names "
            "and count after machine policy. Call those tools directly using their "
            "advertised descriptions and input schemas."
        )
        coordinate_contract = (
            "Image coordinates are one shared transport contract for every task "
            "Skill. For visual grounding, the full-image resolution attached to "
            "every `image_id` is declared as exactly 1000 x 1000. Read coordinates "
            "directly on that declared canvas: its top-left is `u=0,v=0` and its "
            "bottom-right is `u=1000,v=1000`. Do not infer, mention, or convert "
            "through encoded pixel dimensions, and never send encoded-image pixel "
            "indices. On every new image, visually locate and verify the intended "
            "point again; never copy coordinates from a different `image_id` or "
            "assume that a previous point is still correct. Numbers printed on "
            "navigation overlays and coordinates quoted by path or obstacle "
            "telemetry are annotations, not locations of the requested physical "
            "object."
        )
        held_object_evidence = (
            "Held-object evidence. Empirically, an object already in a gripper "
            "generally does not drop unless that gripper has executed `open_gripper`. "
            "Do not infer "
            "a drop from finger `qpos`, chassis motion, a lost track, or a head "
            "image that no longer shows the object. If a current wrist image of "
            "that gripper shows a red grasp-volume overlay, the object has not "
            "dropped. Missing red is not a drop: first treat the wrist view as "
            "badly angled or occluded, and recapture or change viewpoint. Do not "
            "`open_gripper`, and do not call `set_arm_to_grasp_position` with "
            '`gripper="open"` on that arm, while the hold still stands.'
        )
        restored = claude.replace(claude_catalog, codex_catalog)
        restored = restored.replace(coordinate_contract, "")
        restored = restored.replace(held_object_evidence, "")
        restored = restored.replace("object-specific point", "coordinates")
        self.assertEqual(_collapsed(restored), codex)

    def test_pick_skill_delta_is_only_grounding_and_distance_recovery(self) -> None:
        codex = _collapsed(
            _adapt_names(_read(CODEX_ROOT / "skills" / "pick-up-object" / "SKILL.md"))
        )
        claude = _collapsed(
            _read(CLAUDE_ROOT / "skills" / "pick-up-object" / "SKILL.md")
        )
        additions = (
            "The adapter normally labels this precision view `role=raw_rgb`; "
            "locate the physical object itself and ignore navigation lines, "
            "distance text, and coordinates from earlier images.",
            "If the plan says the point is too far to plan, do not jog the EEF or "
            "`close_gripper`: call `adjust_pitch` or `adjust_chassis` to close "
            "range, or call `move_to_reach_point` again on a fresh faced head image "
            "and target, then recapture, reidentify, and plan again. A manual "
            "`adjust_pitch` with a negative degree can reach a lower torso pitch "
            "than `move_to_reach_point`; that extra lean is often useful for a "
            "floor object.",
            "- If `plan_grasp_point_filter_rgbd_lite` returns that the point is too "
            "far to plan, close range first: `adjust_pitch`, `adjust_chassis`, or a "
            "new `move_to_reach_point` on a fresh faced head image. Then recapture, "
            "reidentify, and plan again. A manual `adjust_pitch` with a negative "
            "degree can reach a lower torso pitch than `move_to_reach_point`; that "
            "extra lean is often useful for a floor object. Do not replace that "
            "approach with `adjust_*_eef` or `close_gripper`.",
            "A current wrist image with a red grasp-volume overlay means that "
            "gripper still holds an object: do not start a new pick on it and do "
            "not call `set_arm_to_grasp_position` with `gripper=\"open\"` for that "
            "arm.",
        )
        restored = claude
        for addition in additions:
            self.assertIn(addition, restored)
            restored = restored.replace(addition, "")
        self.assertEqual(_collapsed(restored), codex)

    def test_hook_event_contract_matches_except_claude_lifecycle(self) -> None:
        codex = json.loads(_read(CODEX_ROOT / "hooks" / "hooks.json"))["hooks"]
        claude = json.loads(_read(CLAUDE_ROOT / "hooks" / "hooks.json"))["hooks"]
        self.assertEqual(set(codex) | {"PostToolUse", "UserPromptExpansion"}, set(claude))
        for event_name in codex:
            with self.subTest(event=event_name):
                self.assertEqual(len(codex[event_name]), len(claude[event_name]))
                for codex_group, claude_group in zip(
                    codex[event_name], claude[event_name], strict=True
                ):
                    self.assertEqual(codex_group["matcher"], claude_group["matcher"])
                    self.assertEqual(len(codex_group["hooks"]), len(claude_group["hooks"]))
                    for codex_hook, claude_hook in zip(
                        codex_group["hooks"], claude_group["hooks"], strict=True
                    ):
                        for field in ("type", "timeout", "additionalContextLimit"):
                            if event_name == "PreToolUse" and field == "timeout":
                                self.assertEqual(claude_hook[field], 5)
                                continue
                            self.assertEqual(codex_hook.get(field), claude_hook.get(field))


if __name__ == "__main__":
    unittest.main()
