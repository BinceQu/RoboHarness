from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from behavior_interface_eval_test.tool.official_v2 import task_memory
from behavior_interface_eval_test.tool.official_v2.task_memory import (
    install_official_task_memory,
    load_official_task_memory,
    task_memory_fields,
)


class OfficialMemoryTest(unittest.TestCase):
    def test_public_task_definition_contains_requested_fields(self) -> None:
        memory = load_official_task_memory("make_microwave_popcorn")
        raw = memory.raw()

        self.assertEqual(raw["task"], "make_microwave_popcorn")
        self.assertIn("popcorn bag", raw["task_objective"])
        self.assertIn("Task objective:", raw["instruction"])
        self.assertIn("task BDDL", raw["instruction"])
        self.assertEqual(raw["task_instruction"], raw["instruction"])
        self.assertEqual(
            raw["bddl_conditions"],
            [
                "cooked popcorn 1 exists",
                "popcorn bag 1 contains cooked popcorn 1",
            ],
        )
        self.assertEqual(raw["goal_conditions"], raw["bddl_conditions"])
        self.assertFalse(raw["bddl_live_state_available"])
        self.assertIn("definition only", memory.text())
        self.assertIn("live predicate satisfaction", memory.text())

    def test_public_fields_are_copy_safe(self) -> None:
        fields = task_memory_fields("make_microwave_popcorn")
        fields["bddl_conditions"].append("mutated")

        fresh = task_memory_fields("make_microwave_popcorn")
        self.assertNotIn("mutated", fresh["bddl_conditions"])

    def test_installs_test_owned_server_accessors(self) -> None:
        server = SimpleNamespace()
        memory = install_official_task_memory(
            server,
            "make_microwave_popcorn",
        )

        self.assertEqual(server.get_memory(), memory.raw())
        self.assertEqual(server.get_memory_text(), memory.text())
        self.assertEqual(server.get_memory_summary(), memory.summary())

    def test_dynamic_distance_fields_extend_static_task_memory(self) -> None:
        dynamic = SimpleNamespace(
            memory_fields=lambda: {
                "tracked_object_distances": {
                    "can": {"depth_m": 1.25, "unit": "m"}
                }
            },
            summary_fields=lambda: {"tracked_object_distances": 1},
            text=lambda: "Tracked object distances:\n- can: depth=1.2500 m",
        )
        server = SimpleNamespace()
        memory = install_official_task_memory(
            server,
            "make_microwave_popcorn",
            dynamic_memory=dynamic,
        )

        raw = server.get_memory()
        self.assertEqual(raw["task"], memory.task)
        self.assertEqual(
            raw["tracked_object_distances"]["can"]["depth_m"],
            1.25,
        )
        self.assertEqual(
            server.get_memory_summary()["tracked_object_distances"],
            1,
        )
        self.assertIn("can: depth=1.2500 m", server.get_memory_text())

    def test_unknown_or_incomplete_task_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            path = Path(temp_root) / "task_data.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "incomplete_task",
                                "name": "Incomplete Task",
                                "instruction": "Do the task.",
                                "goal_conditions": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no BDDL conditions"):
                load_official_task_memory(
                    "incomplete_task",
                    task_data_path=path,
                )
            with self.assertRaisesRegex(KeyError, "unknown public evaluator"):
                load_official_task_memory(
                    "missing_task",
                    task_data_path=path,
                )

    def test_module_has_no_original_interface_dependency(self) -> None:
        source_path = Path(task_memory.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        self.assertFalse(
            any(
                name == "behavior_interface"
                or name.startswith("behavior_interface.")
                for name in imports
            )
        )


if __name__ == "__main__":
    unittest.main()
