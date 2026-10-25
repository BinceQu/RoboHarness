from __future__ import annotations

import unittest

from embodied_claude_code.coordinates import (
    LATEST_IMAGE_REMINDER_PREFIX,
    latest_image_grounding_reminder,
    schema_uses_image_coordinates,
    with_image_coordinate_descriptions,
)


class LatestImageReminderTests(unittest.TestCase):
    def test_english_reminder_is_bound_to_the_supplied_image(self) -> None:
        for image_id in ("img_0051", "img_0053", "img_0056"):
            with self.subTest(image_id=image_id):
                reminder = latest_image_grounding_reminder(image_id)
                self.assertTrue(reminder.isascii())
                self.assertTrue(reminder.startswith(LATEST_IMAGE_REMINDER_PREFIX))
                self.assertIn(f"image_id={image_id}", reminder)
                self.assertIn("inspect the latest image actually attached", reminder)
                self.assertIn("derive u/v from that image only", reminder)
                self.assertIn("never copy coordinates", reminder)

    def test_missing_id_requests_a_fresh_image_without_inventing_an_id(self) -> None:
        reminder = latest_image_grounding_reminder("")
        self.assertTrue(reminder.isascii())
        self.assertIn("obtain a fresh image", reminder)
        self.assertNotIn("image_id=", reminder)


class CoordinateSchemaTests(unittest.TestCase):
    def test_flat_point_is_annotated_without_mutating_interface_schema(self) -> None:
        original = {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "u": {"type": "integer", "minimum": 0, "maximum": 1000},
                "v": {"type": "integer", "minimum": 0, "maximum": 1000},
            },
        }

        adapted = with_image_coordinate_descriptions(original)

        self.assertTrue(schema_uses_image_coordinates(original))
        self.assertNotIn("description", original["properties"]["u"])
        self.assertIn("0 is the left edge", adapted["properties"]["u"]["description"])
        self.assertIn("original-image pixel", adapted["properties"]["u"]["description"])
        self.assertEqual(adapted["properties"]["u"]["maximum"], 719)
        self.assertEqual(original["properties"]["u"]["maximum"], 1000)
        self.assertIn("0 is the top edge", adapted["properties"]["v"]["description"])
        self.assertIn(
            "re-ground the point when image_id changes",
            adapted["properties"]["image_id"]["description"],
        )

    def test_nested_multi_point_items_are_annotated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "u": {"type": "integer"},
                            "v": {"type": "integer"},
                        },
                    },
                },
            },
        }

        adapted = with_image_coordinate_descriptions(schema)
        point = adapted["properties"]["points"]["items"]["properties"]

        self.assertTrue(schema_uses_image_coordinates(schema))
        self.assertIn("720 x 720", point["u"]["description"])
        self.assertIn("720 x 720", point["v"]["description"])
        self.assertIn(
            "720 x 720",
            adapted["properties"]["image_id"]["description"],
        )

    def test_unrelated_schema_is_copied_without_coordinate_text(self) -> None:
        schema = {
            "type": "object",
            "properties": {"spin": {"type": "number", "description": "degrees"}},
        }

        adapted = with_image_coordinate_descriptions(schema)

        self.assertFalse(schema_uses_image_coordinates(schema))
        self.assertEqual(adapted, schema)
        self.assertIsNot(adapted, schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
