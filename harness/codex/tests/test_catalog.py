from __future__ import annotations

import unittest

from embodied_codex.catalog import SURFACE_FACING_FALLBACK, ToolCatalog
from embodied_codex.errors import ToolPolicyError
from embodied_codex.profile import ToolProfile


class ToolCatalogTests(unittest.TestCase):
    def test_surface_facing_is_restored_when_interface_omits_it(self) -> None:
        catalog = ToolCatalog.from_payload(
            {"tool_version": "v2", "tools": []},
            ToolProfile.baseline(),
        )

        spec = catalog.get("move_chassis_to_directly_facing_surface")
        self.assertEqual(
            spec.endpoint, "/api/v2/move_chassis_to_directly_facing_surface"
        )
        self.assertEqual(spec.metadata, SURFACE_FACING_FALLBACK)
        points = spec.input_schema["properties"]["points"]
        self.assertEqual(points["minItems"], 3)
        self.assertEqual(points["maxItems"], 3)
        spec.validate_arguments(
            {
                "image_id": "img-1",
                "points": [
                    {"u": 100, "v": 200},
                    {"u": 200, "v": 300},
                    {"u": 300, "v": 120},
                ],
            }
        )

    def test_advertised_surface_facing_wins_without_duplicate(self) -> None:
        advertised = {
            **SURFACE_FACING_FALLBACK,
            "desc": "Interface-owned surface facing contract.",
        }
        catalog = ToolCatalog.from_payload(
            {"tool_version": "v2", "tools": [advertised]},
            ToolProfile.baseline(),
        )

        self.assertEqual(list(catalog.tools), ["move_chassis_to_directly_facing_surface"])
        self.assertEqual(
            catalog.get("move_chassis_to_directly_facing_surface").description,
            "Interface-owned surface facing contract.",
        )

    def test_multi_uv_arm_uses_interface_point_payload(self) -> None:
        catalog = ToolCatalog.from_payload(
            {
                "tool_version": "v2",
                "tools": [
                    {
                        "name": "plan_rgbd",
                        "endpoint": "/api/v2/plan",
                        "mode": "rgbd",
                        "args": [
                            {
                                "name": "points",
                                "widget": "multi_uv_arm",
                                "required": True,
                                "min_points": 1,
                                "max_points": 16,
                                "arm_options": ["any", "left", "right"],
                            }
                        ],
                    }
                ],
            },
            ToolProfile.baseline(),
        )
        spec = catalog.get("plan_rgbd")
        points = spec.input_schema["properties"]["points"]
        self.assertEqual(points["minItems"], 1)
        self.assertEqual(points["maxItems"], 16)
        self.assertEqual(
            points["items"]["properties"]["plan_arm"]["enum"],
            ["any", "left", "right"],
        )
        self.assertEqual(
            points["items"]["properties"]["plan_arm"]["default"], "any"
        )
        self.assertNotIn("mode", spec.input_schema["properties"])
        spec.validate_arguments(
            {"points": [{"u": 250, "v": 750, "plan_arm": "left"}]}
        )
        with self.assertRaises(ToolPolicyError):
            spec.validate_arguments(
                {"points": [{"u": 250, "v": 750, "arm": "left"}]}
            )

    def test_one_of_interface_alternatives_are_preserved(self) -> None:
        catalog = ToolCatalog.from_payload(
            {
                "tools": [
                    {
                        "name": "measure_target",
                        "endpoint": "/api/v2/measure_target",
                        "args": [
                            {
                                "name": "object_name",
                                "widget": "text",
                                "required": False,
                            },
                            {
                                "name": "image_id",
                                "widget": "image",
                                "required": False,
                            },
                            {"name": "u", "widget": "uv", "required": False},
                            {"name": "v", "widget": "uv", "required": False},
                        ],
                        "one_of": [
                            {"fields": ["object_name"]},
                            {"fields": ["image_id", "u", "v"]},
                        ],
                    }
                ]
            },
            ToolProfile.baseline(),
        )
        spec = catalog.get("measure_target")
        spec.validate_arguments({"object_name": "can"})
        spec.validate_arguments({"image_id": "img", "u": 1, "v": 2})
        spec.validate_arguments(
            {
                "object_name": "can",
                "image_id": "img",
                "u": 1,
                "v": 2,
            }
        )
        with self.assertRaises(ToolPolicyError):
            spec.validate_arguments({})


if __name__ == "__main__":
    unittest.main()
