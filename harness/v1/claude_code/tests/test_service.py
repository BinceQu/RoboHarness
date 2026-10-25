from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

from embodied_claude_code.config import Settings
from embodied_claude_code.errors import CameraFrameError, ToolPolicyError, TransportError
from embodied_claude_code.service import (
    OFFICIAL_SESSION_ID_MAX_LEN,
    PERSISTENT_TRACKING_MAX_STRING_CHARS,
    PERSISTENT_TRACKING_MAX_TRACKS,
    PERSISTENT_TRACKING_UNAVAILABLE_WARNING,
    EmbodiedService,
    _compact_persistent_tracking,
    _validate_official_session_id,
)

from fake_behavior import FakeBehaviorClient, PNG_1X1, PNG_720


def wrist_frame(image_id: str, valid_depth_pixel_count: int) -> dict:
    return {
        "ok": True,
        "image_id": image_id,
        "image_width": 480,
        "image_height": 480,
        "valid_depth_pixel_count": valid_depth_pixel_count,
    }


def tracking_memory(*, object_x: float, observation_sequence: int) -> dict[str, Any]:
    return {
        "raw": {
            "tracked_object_distance_tracking": {
                "camera": "head",
                "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
                "depth_unit": "m",
                "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                "xyz_in_robot_base_coord_frame": "current_robot_base",
                "identity_source": "must-not-leak",
            },
            "tracked_object_distances": {
                "held_object": {
                    "camera": "head",
                    "confidence": 0.91,
                    "depth_m": 0.58,
                    "episode_id": "episode-1",
                    "identity_verified": False,
                    "observation_sequence": observation_sequence,
                    "source_image_id": "img-grounding",
                    "source_observation_sequence": 10,
                    "source_session_id": "session-1",
                    "status": "observed",
                    "track_id": "distance_point_001",
                    "u": 505.0,
                    "v": 768.0,
                    "xyz_in_robot_base_coord_m": [object_x, -0.01, 0.13],
                    "identity_source": "must-not-leak",
                },
                "container_opening": {
                    "camera": "head",
                    "confidence": 0.92,
                    "depth_m": 0.52,
                    "episode_id": "episode-1",
                    "observation_sequence": observation_sequence,
                    "source_image_id": "img-grounding",
                    "source_observation_sequence": 10,
                    "source_session_id": "session-1",
                    "status": "observed",
                    "track_id": "distance_point_002",
                    "u": 560.0,
                    "v": 678.0,
                    "xyz_in_robot_base_coord_m": [0.66, -0.07, 0.21],
                },
            },
            "instruction": "must-not-leak",
        },
        "text": "must-not-leak",
    }


class MemorySequenceClient(FakeBehaviorClient):
    def __init__(self, memories: list[Any]) -> None:
        super().__init__()
        self.memories = list(memories)

    def get_memory(self, *, timeout_s: float) -> Any:
        self.memory_timeouts.append(timeout_s)
        if not self.memories:
            raise AssertionError("No memory response remains")
        memory = self.memories.pop(0)
        if isinstance(memory, Exception):
            raise memory
        return memory


class FixedActionClient(MemorySequenceClient):
    def __init__(self, response: dict[str, Any], memory: Any) -> None:
        super().__init__([memory])
        self.response = response

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        if path != "/api/v2/adjust_chassis":
            return super().post_json(path, payload)
        self.requests.append({"path": path, "body": dict(payload)})
        return self.response


class WristSequenceClient(FakeBehaviorClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.tools.append(
            {
                "name": "capture_right_wrist_camera",
                "endpoint": "/api/v2/capture_right_wrist_camera",
                "args": [],
                "desc": "Capture the right wrist RGB camera.",
            }
        )

    def post_json(self, path: str, payload: dict) -> dict:
        if path != "/api/v2/capture_right_wrist_camera":
            return super().post_json(path, payload)
        self.requests.append({"path": path, "body": dict(payload)})
        if not self.responses:
            raise AssertionError("No wrist response remains")
        return self.responses.pop(0)


class EmbodiedServiceTests(unittest.TestCase):
    def test_adapter_lifecycle_and_state_reads_do_not_fetch_memory(self) -> None:
        fake = FakeBehaviorClient()
        service = EmbodiedService(Settings(record=False), client=fake)

        service.start_episode(session_id="no-unrelated-memory-read", record=False)
        service.list_tools()
        service.get_state()
        service.stop_episode()

        self.assertEqual(fake.memory_timeouts, [])

    def test_live_tracking_updates_after_actions_without_regrounding(self) -> None:
        fake = MemorySequenceClient(
            [
                tracking_memory(object_x=0.61, observation_sequence=101),
                tracking_memory(object_x=0.49, observation_sequence=102),
            ]
        )
        service = EmbodiedService(
            Settings(
                base_url="http://127.0.0.1:5011",
                http_timeout_s=2,
                memory_timeout_s=0.2,
                record=False,
            ),
            client=fake,
        )
        service.start_episode(session_id="tracking-updates", record=False)

        first = service.call(
            tool_name="adjust_chassis", arguments={"forward": 0.05}
        )
        second = service.call(
            tool_name="adjust_chassis", arguments={"forward": 0.05}
        )

        first_track = first.data["persistent_tracking"]["tracks"]["held_object"]
        second_track = second.data["persistent_tracking"]["tracks"]["held_object"]
        self.assertEqual(first_track["xyz_in_robot_base_coord_m"][0], 0.61)
        self.assertEqual(second_track["xyz_in_robot_base_coord_m"][0], 0.49)
        self.assertEqual(first_track["observation_sequence"], 101)
        self.assertEqual(second_track["observation_sequence"], 102)
        self.assertEqual(fake.memory_timeouts, [0.2, 0.2])
        self.assertEqual(
            [request["path"] for request in fake.requests],
            ["/api/v2/adjust_chassis", "/api/v2/adjust_chassis"],
        )
        self.assertTrue(
            all(
                "track_object_distance" not in request["path"]
                for request in fake.requests
            )
        )

    def test_expected_memory_failures_preserve_action_result_and_status(self) -> None:
        cases = (
            (
                "transport-after-success",
                {"ok": True, "completed": "unchanged"},
                TransportError("memory timed out"),
                False,
            ),
            (
                "malformed-after-explicit-failure",
                {"ok": False, "failure_reason": "blocked", "detail": {"x": 1}},
                {"raw": {"tracked_object_distances": []}},
                True,
            ),
        )
        for name, response, memory, expected_error in cases:
            with self.subTest(name=name):
                fake = FixedActionClient(response, memory)
                service = EmbodiedService(
                    Settings(
                        base_url="http://127.0.0.1:5011",
                        http_timeout_s=2,
                        memory_timeout_s=0.15,
                        record=False,
                    ),
                    client=fake,
                )
                service.start_episode(session_id=name, record=False)

                result = service.call(
                    tool_name="adjust_chassis", arguments={"forward": 0.05}
                )

                self.assertEqual(result.data["response"], response)
                self.assertEqual(result.is_error, expected_error)
                self.assertEqual(
                    result.data["persistent_tracking"],
                    {
                        "available": False,
                        "warning": PERSISTENT_TRACKING_UNAVAILABLE_WARNING,
                    },
                )
                self.assertEqual(
                    result.data["warnings"],
                    [PERSISTENT_TRACKING_UNAVAILABLE_WARNING],
                )
                self.assertEqual(fake.memory_timeouts, [0.15])

    def test_unexpected_memory_programming_error_is_not_hidden(self) -> None:
        fake = FixedActionClient(
            {"ok": True}, RuntimeError("unexpected test-double failure")
        )
        service = EmbodiedService(
            Settings(record=False),
            client=fake,
        )
        service.start_episode(session_id="unexpected-memory-error", record=False)

        with self.assertRaisesRegex(RuntimeError, "unexpected test-double failure"):
            service.call(tool_name="adjust_chassis", arguments={"forward": 0.05})

    def test_compact_tracking_is_whitelisted_finite_and_bounded(self) -> None:
        tracks: dict[str, Any] = {}
        for index in range(PERSISTENT_TRACKING_MAX_TRACKS + 2):
            tracks[f"track-{index:02d}"] = {
                "camera": "head",
                "confidence": 0.9,
                "depth_m": 0.5,
                "episode_id": "e" * (PERSISTENT_TRACKING_MAX_STRING_CHARS + 20),
                "identity_verified": False,
                "lost": False,
                "observation_sequence": index,
                "source_image_id": f"image-{index}",
                "source_observation_sequence": 1,
                "source_session_id": "session",
                "status": "observed",
                "track_id": f"point-{index}",
                "u": 500.0,
                "v": 600.0,
                "valid": True,
                "xyz_in_robot_base_coord_m": [0.5, 0.0, 0.2],
                "identity_source": "must-not-leak",
                "diagnostic_payload": "must-not-leak",
            }
        tracks["track-00"]["depth_m"] = float("nan")
        tracks["track-01"]["xyz_in_robot_base_coord_m"] = [0.5, float("inf"), 0.2]
        memory = {
            "raw": {
                "tracked_object_distance_tracking": {
                    "camera": "head",
                    "coordinate_system": "relative",
                    "depth_unit": "m",
                    "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                    "xyz_in_robot_base_coord_frame": "current_robot_base",
                    "replay_storage": "must-not-leak",
                },
                "tracked_object_distances": tracks,
                "instruction": "must-not-leak",
            },
            "text": "must-not-leak",
        }

        compact = _compact_persistent_tracking(memory)

        self.assertTrue(compact["available"])
        self.assertEqual(len(compact["tracks"]), PERSISTENT_TRACKING_MAX_TRACKS)
        self.assertEqual(
            list(compact["tracks"]),
            [f"track-{index:02d}" for index in range(PERSISTENT_TRACKING_MAX_TRACKS)],
        )
        self.assertEqual(
            compact["total_track_count"], PERSISTENT_TRACKING_MAX_TRACKS + 2
        )
        self.assertTrue(compact["tracks_truncated"])
        self.assertEqual(
            len(compact["tracks"]["track-00"]["episode_id"]),
            PERSISTENT_TRACKING_MAX_STRING_CHARS,
        )
        self.assertNotIn("depth_m", compact["tracks"]["track-00"])
        self.assertNotIn(
            "xyz_in_robot_base_coord_m", compact["tracks"]["track-01"]
        )
        serialized = json.dumps(compact, ensure_ascii=True)
        self.assertNotIn("must-not-leak", serialized)
        self.assertEqual(compact["map_marks"], [])

    def test_compact_tracking_includes_map_marks(self) -> None:
        memory = tracking_memory(object_x=0.61, observation_sequence=7)
        memory["raw"]["marked_places"] = [
            {
                "name": "box1",
                "direction": "ahead",
                "range_m": 2.1,
                "spin_deg_to_face": -17.0,
                "secret": "must-not-leak",
            },
            {
                "name": "garage door",
                "direction": "left",
                "range_m": 8.3,
                "spin_deg_to_face": 91.0,
            },
        ]
        compact = _compact_persistent_tracking(memory)
        self.assertEqual(
            compact["map_marks"],
            [
                {
                    "name": "box1",
                    "direction": "ahead",
                    "range_m": 2.1,
                    "spin_deg_to_face": -17.0,
                },
                {
                    "name": "garage door",
                    "direction": "left",
                    "range_m": 8.3,
                    "spin_deg_to_face": 91.0,
                },
            ],
        )
        self.assertNotIn("must-not-leak", json.dumps(compact))

    def test_episode_calls_are_visual_recorded_and_session_scoped(self) -> None:
        fake = FakeBehaviorClient()
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            started = service.start_episode(
                session_id="unit-session", record=True, label="baseline"
            )
            self.assertEqual(started.data["session_id"], "unit-session")
            self.assertEqual(len(started.data["available_tools"]), 5)
            self.assertIn("mark_on_map", started.data["available_tools"])

            catalog = service.list_tools()
            self.assertEqual(catalog.data["tool_version"], "v2")

            captured = service.call(tool_name="capture_head_camera", arguments={})
            planned = service.call(
                tool_name="plan_grasp_point_filter_rgbd_lite",
                arguments={"image_id": "img-1", "u": 400, "v": 500},
            )
            received = planned.data["response"]["received"]
            self.assertEqual(received["session_id"], "unit-session")
            self.assertEqual(
                received["mode"], "grasp_point_filter_rgbd_lite"
            )

            self.assertEqual(len(captured.media), 1)
            self.assertEqual(captured.media[0].data, PNG_720)
            self.assertNotIn(
                "data:image", json.dumps(captured.data, ensure_ascii=True)
            )

            with self.assertRaises(ToolPolicyError):
                service.call(tool_name="not_a_tool", arguments={})

            state = service.get_state()
            self.assertEqual(state.data["state"]["task"], "unit_test_task")
            stopped = service.stop_episode(outcome="success", note="done")
            run_dir = Path(stopped.data["recording"]["run_dir"])

            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            turns = [
                json.loads(line)
                for line in (run_dir / "turns.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["outcome"], "success")
            self.assertEqual(manifest["turn_count"], 7)
            self.assertEqual(len(turns), 7)
            self.assertTrue(any(turn["tool_result"]["is_error"] for turn in turns))
            self.assertNotIn(
                "data:image", (run_dir / "turns.jsonl").read_text(encoding="utf-8")
            )
            media_files = list((run_dir / "media").iterdir())
            self.assertEqual(len(media_files), 1)

        self.assertEqual(
            [request["path"] for request in fake.requests],
            ["/api/v2/capture_head_camera", "/api/v2/plan"],
        )

    def test_session_id_rejects_official_overlong_id(self) -> None:
        long_id = (
            "search-pick-place-15060-prompt-gates-sol-attempt12-continue-20260820"
        )
        self.assertGreater(len(long_id), OFFICIAL_SESSION_ID_MAX_LEN)
        with self.assertRaises(ToolPolicyError) as ctx:
            _validate_official_session_id(long_id)
        self.assertIn("64", str(ctx.exception))
        self.assertIn(str(len(long_id)), str(ctx.exception))

        max_id = "a" * OFFICIAL_SESSION_ID_MAX_LEN
        self.assertEqual(_validate_official_session_id(max_id), max_id)
        generated = EmbodiedService._new_session_id()
        self.assertLessEqual(len(generated), OFFICIAL_SESSION_ID_MAX_LEN)
        _validate_official_session_id(generated)

        fake = FakeBehaviorClient()
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            with self.assertRaises(ToolPolicyError):
                service.start_episode(session_id=long_id, record=False)
            self.assertEqual(fake.requests, [])

    def test_session_id_cannot_be_overridden(self) -> None:
        fake = FakeBehaviorClient()
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            service.start_episode(session_id="owned", record=False)
            with self.assertRaises(ToolPolicyError):
                service.call(
                    tool_name="adjust_chassis",
                    arguments={"session_id": "other", "spin": 10},
                )
            service.stop_episode()

    def test_wrist_capture_retries_low_valid_depth_frame(self) -> None:
        fake = WristSequenceClient(
            [
                wrist_frame("bad-frame", 20_650),
                wrist_frame("good-frame", 230_400),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            service.start_episode(session_id="wrist-retry", record=False)
            result = service.call(
                tool_name="capture_right_wrist_camera", arguments={}
            )

        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(result.data["response"]["image_id"], "good-frame")
        self.assertEqual(result.data["capture_quality"]["attempts"], 2)
        self.assertEqual(
            result.data["capture_quality"]["discarded_bad_frames"], 1
        )
        self.assertAlmostEqual(
            result.data["capture_quality"]["discarded_frames"][0][
                "valid_depth_ratio"
            ],
            20_650 / 230_400,
            places=6,
        )
        self.assertEqual(len(result.data["warnings"]), 1)

    def test_wrist_capture_fails_closed_after_three_bad_frames(self) -> None:
        fake = WristSequenceClient(
            [
                wrist_frame("bad-1", 20_650),
                wrist_frame("bad-2", 20_650),
                wrist_frame("bad-3", 20_650),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            service.start_episode(session_id="wrist-fail-closed", record=False)
            with self.assertRaises(CameraFrameError) as raised:
                service.call(
                    tool_name="capture_right_wrist_camera", arguments={}
                )

        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(raised.exception.code, "camera_frame_error")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.details["attempts"], 3)
        self.assertEqual(len(raised.exception.details["discarded_frames"]), 3)

    def test_non_wrist_tool_is_never_retried_by_frame_guard(self) -> None:
        class BadActionClient(FakeBehaviorClient):
            def post_json(self, path: str, payload: dict) -> dict:
                if path != "/api/v2/adjust_chassis":
                    return super().post_json(path, payload)
                self.requests.append({"path": path, "body": dict(payload)})
                return wrist_frame("not-a-camera", 20_650)

        fake = BadActionClient()
        with tempfile.TemporaryDirectory() as directory:
            service = EmbodiedService(
                Settings(
                    base_url="http://127.0.0.1:5011",
                    http_timeout_s=2,
                    record_root=Path(directory),
                ),
                client=fake,
            )
            service.start_episode(session_id="no-action-retry", record=False)
            result = service.call(
                tool_name="adjust_chassis", arguments={"forward": 0.05}
            )

        self.assertEqual(len(fake.requests), 1)
        self.assertNotIn("capture_quality", result.data)


if __name__ == "__main__":
    unittest.main()
