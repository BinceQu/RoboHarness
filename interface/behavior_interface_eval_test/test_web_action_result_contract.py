from __future__ import annotations

import pytest

import behavior_interface.web as web


class _ActionServer:
    target_hz = 30.0

    def __init__(self, *, fps: float = 1.0) -> None:
        self.fps = fps
        self.calls: list[tuple[str, dict]] = []
        self.waits: list[tuple[str, float]] = []

    def submit_skill(self, name: str, args: dict) -> str:
        self.calls.append((name, dict(args)))
        return f"request-{len(self.calls)}"

    def wait_for_skill_result(
        self,
        skill_name: str,
        *,
        timeout_s: float,
        request_id: str,
    ) -> dict:
        del request_id
        self.waits.append((skill_name, timeout_s))
        if skill_name == "capture":
            return {
                "ok": True,
                "tool": "capture",
                "feed": "head",
                "image_id": "img-after",
                "rgb_main": "data:image/png;base64,after",
                "rgb_overlay_path": "/tmp/img-after.path.png",
                "camera": {"frame": "post-action"},
            }
        if skill_name == "capture_left_wrist_camera":
            return {
                "ok": True,
                "tool": skill_name,
                "feed": "left_wrist",
                "image_id": "left-wrist-after",
                "rgb_main": "data:image/png;base64,left-after",
                "camera": {"frame": "post-action"},
            }
        if skill_name == "move_eef":
            return {"ok": True, "arm": "left"}
        if skill_name.startswith("move_to_point"):
            return {
                "ok": True,
                "image_id": "img-before",
                "rgb_overlay_path": "/tmp/img-before.path.png",
                "camera": {"frame": "input"},
                "shoulder_distance_estimate": {
                    "left_m": 0.68,
                    "right_m": 0.69,
                },
            }
        raise AssertionError(f"unexpected skill: {skill_name}")

    def cancel_current_skill(self) -> None:
        raise AssertionError("request should not be cancelled")


def test_move_to_reach_point_promotes_one_coherent_post_action_frame() -> None:
    server = _ActionServer()
    client = web.build_app(server).test_client()

    response = client.post(
        "/api/v2/move_to_reach_point",
        json={
            "session_id": "human-test",
            "image_id": "img-before",
            "u": 469,
            "v": 548,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["input_image_id"] == "img-before"
    assert payload["image_id"] == "img-after"
    assert payload["rgb_main"] == "data:image/png;base64,after"
    assert payload["rgb_overlay_path"] == "/tmp/img-after.path.png"
    assert payload["camera"] == {"frame": "post-action"}
    assert payload["observation"]["image_id"] == payload["image_id"]
    assert payload["shoulder_distance_estimate"]["left_m"] == 0.68


@pytest.mark.parametrize(
    ("fps", "expected_timeout_s"),
    [(1.0, 187.5), (0.5, 345.0)],
)
def test_close_gripper_default_wait_covers_slow_evaluator_control_horizon(
    fps: float,
    expected_timeout_s: float,
) -> None:
    server = _ActionServer(fps=fps)
    client = web.build_app(server).test_client()

    response = client.post(
        "/api/v2/close_gripper",
        json={"session_id": "human-test", "arm": "left"},
    )

    assert response.status_code == 200
    assert server.waits[0][0] == "move_eef"
    assert server.waits[0][1] == pytest.approx(expected_timeout_s)


def test_close_gripper_explicit_wait_timeout_is_preserved() -> None:
    server = _ActionServer(fps=0.5)
    client = web.build_app(server).test_client()

    response = client.post(
        "/api/v2/close_gripper",
        json={
            "session_id": "human-test",
            "arm": "left",
            "timeout_s": 95.0,
        },
    )

    assert response.status_code == 200
    assert server.waits[0] == ("move_eef", 95.0)
