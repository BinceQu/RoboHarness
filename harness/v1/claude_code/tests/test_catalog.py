from __future__ import annotations

import unittest

from embodied_claude_code.catalog import (
    MARK_ON_MAP_FALLBACK,
    SURFACE_FACING_FALLBACK,
    ToolCatalog,
)
from embodied_claude_code.errors import ToolPolicyError
from embodied_claude_code.profile import ToolProfile


class ToolCatalogTests(unittest.TestCase):
    def test_mark_on_map_is_restored_when_interface_profile_omits_it(self) -> None:
        catalog = ToolCatalog.from_payload(
            {"tool_version": "v2", "tools": []},
            ToolProfile.baseline(),
        )

        spec = catalog.get("mark_on_map")
        self.assertEqual(spec.endpoint, "/api/v2/mark_on_map")
        self.assertNotEqual(spec.metadata, MARK_ON_MAP_FALLBACK)
        self.assertEqual(MARK_ON_MAP_FALLBACK["args"][2]["maximum"], 1000)
        self.assertEqual(spec.input_schema["required"], ["name"])
        self.assertEqual(spec.input_schema["properties"]["u"]["maximum"], 719)
        self.assertIn("original-image pixel", spec.input_schema["properties"]["u"]["description"])
        spec.validate_arguments({"name": "target_box"})
        spec.validate_arguments(
            {"name": "target_box", "image_id": "img-1", "u": 500, "v": 600}
        )

    def test_advertised_mark_on_map_wins_without_duplicate(self) -> None:
        advertised = {
            **MARK_ON_MAP_FALLBACK,
            "desc": "Interface-owned mark contract.",
        }
        catalog = ToolCatalog.from_payload(
            {"tool_version": "v2", "tools": [advertised]},
            ToolProfile.baseline(),
        )

        self.assertEqual(
            list(catalog.tools),
            ["mark_on_map", "move_chassis_to_directly_facing_surface"],
        )
        self.assertEqual(
            catalog.get("mark_on_map").description,
            "Interface-owned mark contract.",
        )

    def test_surface_facing_is_restored_when_interface_omits_it(self) -> None:
        catalog = ToolCatalog.from_payload(
            {"tool_version": "v2", "tools": []},
            ToolProfile.baseline(),
        )

        spec = catalog.get("move_chassis_to_directly_facing_surface")
        self.assertEqual(
            spec.endpoint, "/api/v2/move_chassis_to_directly_facing_surface"
        )
        self.assertNotEqual(spec.metadata, SURFACE_FACING_FALLBACK)
        points = spec.input_schema["properties"]["points"]
        self.assertEqual(points["minItems"], 3)
        self.assertEqual(points["maxItems"], 3)
        self.assertEqual(points["items"]["properties"]["u"]["maximum"], 719)
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
            {"image_id": "img-1", "points": [{"u": 250, "v": 650, "plan_arm": "left"}]}
        )
        with self.assertRaises(ToolPolicyError):
            spec.validate_arguments(
                {"image_id": "img-1", "points": [{"u": 250, "v": 650, "arm": "left"}]}
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

    def test_python_str_annotation_is_coerced_to_json_schema_string(self) -> None:
        catalog = ToolCatalog.from_payload(
            {
                "tool_version": "v2",
                "tools": [
                    {
                        "name": "move_tracked_point",
                        "endpoint": "/api/v2/move_tracked_point",
                        "args": [
                            {
                                "name": "execution_mode",
                                "type": "str",
                                "required": False,
                                "default": "exec",
                            }
                        ],
                    }
                ],
            },
            ToolProfile.baseline(),
        )
        spec = catalog.get("move_tracked_point")
        self.assertEqual(
            spec.input_schema["properties"]["execution_mode"]["type"],
            "string",
        )
        spec.validate_arguments({"execution_mode": "exec"})

    def test_move_tracked_point_keeps_items_when_catalog_type_is_any(self) -> None:
        catalog = ToolCatalog.from_payload(
            {
                "tool_version": "v2",
                "tools": [
                    {
                        "name": "move_tracked_point",
                        "endpoint": "/api/v2/move_tracked_point",
                        "args": [
                            {
                                "name": "execution_mode",
                                "type": "string",
                                "required": False,
                                "default": "exec",
                                "options": ["exec", "plan"],
                            },
                            {
                                "name": "points",
                                "type": "any",
                                "widget": "tracked_target_points",
                                "required": True,
                                "min_points": 1,
                                "max_points": 6,
                                "items": {
                                    "type": "object",
                                    "required": ["name"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "role": {
                                            "type": "string",
                                            "enum": ["on_hand", "off_hand"],
                                        },
                                        "target_xyz_m": {
                                            "type": "object",
                                            "required": ["x", "y", "z"],
                                            "additionalProperties": False,
                                            "properties": {
                                                "x": {"type": "string"},
                                                "y": {"type": "string"},
                                                "z": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        ],
                    }
                ],
            },
            ToolProfile.baseline(),
        )
        spec = catalog.get("move_tracked_point")
        points = spec.input_schema["properties"]["points"]
        self.assertEqual(points["type"], "array")
        self.assertEqual(
            set(points["items"]["properties"]),
            {"name", "role", "target_xyz_m"},
        )
        spec.validate_arguments(
            {
                "execution_mode": "plan",
                "points": [
                    {
                        "name": "left_finger_tip",
                        "role": "on_hand",
                        "target_xyz_m": {"x": "x", "y": "y", "z": "a"},
                    },
                    {
                        "name": "handle",
                        "role": "off_hand",
                        "target_xyz_m": {"x": "x", "y": "y", "z": "z"},
                    },
                ],
            }
        )
        with self.assertRaises(ToolPolicyError) as raised:
            spec.validate_arguments(
                {
                    "execution_mode": "plan",
                    "points": [
                        {
                            "name": "handle",
                            "role": "off_hand",
                            "x": "x",
                            "y": "y",
                            "z": "z",
                        }
                    ],
                }
            )
        self.assertIn("target_xyz_m", str(raised.exception))

    def test_generic_python_annotation_is_dropped_not_copied(self) -> None:
        catalog = ToolCatalog.from_payload(
            {
                "tool_version": "v2",
                "tools": [
                    {
                        "name": "move_tracked_point",
                        "endpoint": "/api/v2/move_tracked_point",
                        "args": [
                            {
                                "name": "relations",
                                "type": "Sequence[Mapping[str, Any]] | None",
                                "required": False,
                            }
                        ],
                    }
                ],
            },
            ToolProfile.baseline(),
        )
        spec = catalog.get("move_tracked_point")
        self.assertNotIn("type", spec.input_schema["properties"]["relations"])
        spec.validate_arguments({"relations": [{"a": 1}]})


if __name__ == "__main__":
    unittest.main()
