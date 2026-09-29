from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from embodied_claude_code.monitor_cards import (
    _image_id_from_response,
    _write_monitor_files,
    publish_model_card_image,
)


class MonitorCardPublishTests(unittest.TestCase):
    def test_image_id_prefers_top_level_then_observation(self) -> None:
        self.assertEqual(
            _image_id_from_response({"image_id": "img_0008"}),
            "img_0008",
        )
        self.assertEqual(
            _image_id_from_response({"observation": {"image_id": "img_0018"}}),
            "img_0018",
        )
        self.assertEqual(_image_id_from_response({}), "")

    def test_publish_posts_model_jpeg(self) -> None:
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
            return _Response()

        ok = publish_model_card_image(
            data=b"\xff\xd8\xff" + b"x" * 16,
            image_id="img_0008",
            tool="capture_head_camera",
            session_id="sol15062-mbox19",
            base_url="http://127.0.0.1:15062",
            opener=opener,
        )
        self.assertTrue(ok)
        self.assertEqual(
            captured["url"],
            "http://127.0.0.1:15062/api/agent_monitor/model_image",
        )
        payload = json.loads(captured["body"])
        self.assertEqual(payload["image_id"], "img_0008")
        self.assertEqual(payload["session_id"], "sol15062-mbox19")
        self.assertIn("image_b64", payload)

    def test_file_fallback_overwrites_output_thumb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            thumbs = run_dir / "thumbs"
            thumbs.mkdir()
            old = thumbs / "card_0008_out.jpg"
            old.write_bytes(b"old-thumb")
            jpeg = b"\xff\xd8\xff" + b"model"
            ok = _write_monitor_files(
                {
                    "run_dir": str(run_dir),
                    "cards": [
                        {
                            "card_id": 8,
                            "tool": "capture_head_camera",
                            "output_image_id": "img_0008",
                            "has_output": True,
                        }
                    ],
                },
                image_id="img_0008",
                data=jpeg,
                tool="capture_head_camera",
            )
            self.assertTrue(ok)
            self.assertEqual(old.read_bytes(), jpeg)
            self.assertEqual((thumbs / "model_img_0008.jpg").read_bytes(), jpeg)


if __name__ == "__main__":
    unittest.main()
