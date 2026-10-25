"""Public receipts preserve physical evidence without generic controller flags."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from tools.util.files import atomic_json
from tools.util.history import CallHistory, record_failure
from tools.util.receipt import camera_observation_ready, tool_error, without_ok


def test_partial_move_keeps_all_measurements_and_internal_result(tmp_path):
    state = tmp_path / ".runtime"
    atomic_json(state / "state.json", {"counter": 0})
    (tmp_path / "tool_call_history").mkdir()
    robot = SimpleNamespace(workspace=tmp_path, state_dir=state, session_id="test",
        config={"episode_id": "episode", "trajectory_dir": str(tmp_path / "trajectory")}, logs=[],
        _official_adapter=SimpleNamespace(current=SimpleNamespace(sequence=1)))
    evidence = {"ok": True, "requested": {"forward": 2.5, "translation": -.9},
        "actual": {"forward": .058, "translation": -.021},
        "linear_target_reached": False, "obstacle_limited": True,
        "stop": {"ok": True, "settled": True},
        "diagnostics": [{"ok": False, "error": "blocked"}]}
    original = copy.deepcopy(evidence)
    result = {"ok": True, "response": evidence, "images": [], "joint_commands_sent": 20}
    history = CallHistory(robot, "adjust_chassis", {"forward": 2.5, "translation": -.9})
    public = history.finish(result, {"operation_result": evidence})
    assert evidence == original and evidence["ok"] is True
    assert "ok" not in public and "ok" not in public["response"]
    assert public["response"]["actual"] == original["actual"]
    assert public["response"]["linear_target_reached"] is False
    assert public["response"]["obstacle_limited"] is True
    assert public["response"]["diagnostics"] == [{"error": "blocked"}]
    for path in (history.calls / "output.json", history.path / "result.json", state / "last_result.json"):
        stored = json.loads(path.read_text())
        assert stored == without_ok(stored)
    assert public["joint_commands_sent"] == 20


def test_failed_cli_without_initialized_workspace_does_not_leak_ok(monkeypatch):
    def unavailable():
        raise RuntimeError("no session")
    monkeypatch.setattr("tools.util.settings.settings", unavailable)
    result = record_failure("capture_head_camera", {}, {"ok": False, "error": "disconnected", "images": []})
    assert "ok" not in result and result["error"] == "disconnected"
    assert result["images"] == [] and "no session" in result["history_error"]


def test_startup_requires_camera_evidence_and_accepts_old_receipts():
    good = {"tool_name": "capture_head_camera", "images": [{"path": "head.png"}],
            "perception": "latest.json", "latest_clickable_image_id": "img_1"}
    assert camera_observation_ready(good)
    assert camera_observation_ready({**good, "ok": True})
    for bad in ({}, {"ok": True}, {**good, "ok": False}, {**good, "error": "failed"},
                {**good, "images": []}, {**good, "response": {"error": "capture failed"}}):
        assert not camera_observation_ready(bad)
    assert tool_error({"response": {"error": "empty grasp", "grasp_confirmed": False}}) == "empty grasp"
