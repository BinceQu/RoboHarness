from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from behavior_interface_eval_test.official_bddl_progress import (
    UI_BDDL_PROGRESS_KEY,
    build_evaluator_bddl_progress,
    decode_bddl_progress,
    encode_bddl_progress,
    sanitize_bddl_progress,
)


class OfficialBddlProgressTest(unittest.TestCase):
    @staticmethod
    def _task():
        conditions = SimpleNamespace(
            parsed_goal_conditions=[
                ["real", "cooked__popcorn.n.01_1"],
                [
                    "contains",
                    "popcorn__bag.n.01_1",
                    "cooked__popcorn.n.01_1",
                ],
            ]
        )
        return SimpleNamespace(
            compiled_task=SimpleNamespace(conditions=conditions),
            activity_natural_language_goal_conditions=[],
        )

    def test_builds_normal_interface_goal_schema_from_evaluator_status(self) -> None:
        progress = build_evaluator_bddl_progress(
            self._task(),
            {"satisfied": [1], "unsatisfied": [0]},
        )

        self.assertTrue(progress["ok"])
        self.assertEqual(progress["satisfied"], 1)
        self.assertEqual(progress["total"], 2)
        self.assertFalse(progress["complete"])
        self.assertFalse(progress["items"][0]["satisfied"])
        self.assertTrue(progress["items"][1]["satisfied"])
        self.assertIn("popcorn", progress["items"][0]["full"])
        self.assertIn("含有", progress["items"][1]["full"])

    def test_json_envelope_roundtrip_is_primitive_and_bounded(self) -> None:
        encoded = encode_bddl_progress(
            build_evaluator_bddl_progress(
                self._task(),
                {"satisfied": [0, 1]},
            )
        )

        self.assertIsInstance(encoded, str)
        self.assertIsInstance(json.loads(encoded), dict)
        decoded = decode_bddl_progress(encoded)
        self.assertTrue(decoded["complete"])
        self.assertEqual(decoded["source"], "evaluator-ui-only")

    def test_sanitizer_discards_non_ui_fields_and_recomputes_counts(self) -> None:
        payload = sanitize_bddl_progress(
            {
                "items": [
                    {
                        "index": 99,
                        "label": "goal",
                        "full": "goal full",
                        "satisfied": True,
                        "object_pose": [1, 2, 3],
                    }
                ],
                "satisfied": 0,
                "total": 100,
                "complete": False,
                "scene": {"hidden": True},
                "ok": True,
            }
        )

        self.assertEqual(payload["satisfied"], 1)
        self.assertEqual(payload["total"], 1)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["items"][0]["index"], 0)
        self.assertNotIn("object_pose", payload["items"][0])
        self.assertNotIn("scene", payload)

    def test_reserved_key_is_explicitly_namespaced(self) -> None:
        self.assertEqual(
            UI_BDDL_PROGRESS_KEY,
            "__behavior_interface_ui_bddl_progress__",
        )

    def test_stack_selects_physical_gpu_and_bounded_tasking_entrypoint(self) -> None:
        package_dir = Path(__file__).resolve().parent
        stack_script = (
            package_dir / "start_official_test_stack_v391.sh"
        ).read_text(encoding="utf-8")
        evaluator_script = (
            package_dir / "launch_official_evaluator_v391.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU"', stack_script)
        self.assertIn('IK_FILTER_CUDA_VISIBLE_DEVICES="$GPU"', stack_script)
        self.assertIn('BEHAVIOR_RTABMAP_CUDA_DEVICE="$GPU"', stack_script)
        self.assertIn(
            'BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE="$GPU"', stack_script
        )
        self.assertIn('BEHAVIOR_SLAM_CUDA_DEVICE="cuda:0"', stack_script)
        self.assertIn(
            'OFFICIAL_V2_LITE_OCCUPANCY_DEVICE="cuda:0"', stack_script
        )
        self.assertIn("OMNIGIBSON_GPU_ID=0", stack_script)
        self.assertIn("official_port_gpu.sh", stack_script)
        self.assertIn('EXPECTED_GPU="$(official_expected_gpu "$HTTP_PORT")"', stack_script)
        helper = (package_dir / "official_port_gpu.sh").read_text(encoding="utf-8")
        self.assertIn("15060|15061|15062) echo 0", helper)
        self.assertIn("15063|15064|15065) echo 1", helper)
        self.assertIn("15066|15067|15068) echo 2", helper)
        self.assertIn("15069) echo 3", helper)
        self.assertIn('BEHAVIOR_EVAL_TEST_PHYSICAL_GPU="$GPU"', stack_script)
        self.assertIn("BEHAVIOR_EVAL_TEST_UNMASK_CUDA=0", stack_script)
        self.assertIn("-u OMNIGIBSON_APPDATA_PATH", stack_script)
        self.assertEqual(
            stack_script.count(
                'OMNIGIBSON_APPDATA_PATH="$EVALUATOR_APPDATA_PATH"'
            ),
            1,
        )
        self.assertEqual(
            stack_script.count(
                'BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS="$EVALUATOR_APPDATA_PATH"'
            ),
            2,
        )
        self.assertIn('gpu${PHYSICAL_GPU}', evaluator_script)
        self.assertIn(
            "behavior_interface_eval_test.official_evaluator_entrypoint",
            evaluator_script,
        )

    def test_legacy_interface_launchers_keep_cuda_consumers_on_owned_card(self) -> None:
        package_dir = Path(__file__).resolve().parent.parent
        launcher = (package_dir / "scripts" / "launch_interface.sh").read_text(
            encoding="utf-8"
        )
        starter = (package_dir / "scripts" / "start_interface.sh").read_text(
            encoding="utf-8"
        )
        six_gpu = (
            package_dir / "scripts" / "start_interface_6gpu.sh"
        ).read_text(encoding="utf-8")
        for source in (launcher, starter):
            self.assertIn("BEHAVIOR_RTABMAP_CUDA_DEVICE", source)
            self.assertIn("BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE", source)
            self.assertIn("BEHAVIOR_SLAM_CUDA_DEVICE", source)
            self.assertIn("OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", source)
        self.assertIn('"BEHAVIOR_RTABMAP_CUDA_DEVICE=$gpu"', six_gpu)
        self.assertIn('"BEHAVIOR_SLAM_CUDA_DEVICE=cuda:0"', six_gpu)


if __name__ == "__main__":
    unittest.main()
