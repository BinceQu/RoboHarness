from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from behavior_interface_eval_test.interface_gateway import build_app


class GatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("http://127.0.0.1:18080").test_client()

    @patch("requests.Session.request")
    def test_forwards_post_body_and_path(self, request_mock: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "application/json"}
        response.content = b'{"ok":true}'
        response.close = Mock()
        request_mock.return_value = response

        result = self.app.post(
            "/api/v2/adjust_chassis?mode=test",
            data=b'{"forward":0.1}',
            content_type="application/json",
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {"ok": True})
        kwargs = request_mock.call_args.kwargs
        self.assertEqual(kwargs["url"], "http://127.0.0.1:18080/api/v2/adjust_chassis")
        self.assertEqual(kwargs["data"], b'{"forward":0.1}')
        self.assertEqual(kwargs["params"], [("mode", "test")])

    @patch("requests.Session.request")
    def test_marks_gateway_response(self, request_mock: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html", "Content-Length": "4"}
        response.content = b"test"
        response.close = Mock()
        request_mock.return_value = response

        result = self.app.get("/")

        self.assertEqual(result.data, b"test")
        self.assertEqual(
            result.headers["X-Behavior-Interface-Layer"],
            "evaluator-separated-test",
        )
        self.assertEqual(result.headers["Content-Type"], "text/html")


if __name__ == "__main__":
    unittest.main()
