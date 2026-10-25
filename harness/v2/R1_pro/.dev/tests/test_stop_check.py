"""Native completion feedback is isolated, bounded, and never controls physics."""
import json

import pytest

from tools.util.files import atomic_json
from tools.util.stop_check import FEEDBACK, process


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    state_dir = tmp_path / ".runtime/state"
    state_dir.mkdir(parents=True)
    atomic_json(tmp_path / ".session.json", {"workspace": str(tmp_path),
        "state_dir": str(state_dir), "session_id": "robot-test"})
    atomic_json(state_dir / "state.json", {"session_id": "robot-test", "counter": 12})
    monkeypatch.setenv("ROBOT_WORKSPACE", str(tmp_path))
    return state_dir


def test_feedback_once_per_call_counter_and_claude_session(workspace):
    event = {"hook_event_name": "Stop", "session_id": "claude-a", "stop_hook_active": False}
    response = {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": FEEDBACK}}
    assert process(event) == response
    assert process(event) == {}
    # The native flag remains true after a tool call on some CLI versions.
    # Bound by robot-call state, so a later premature stop can still be checked.
    event["stop_hook_active"] = True
    assert process(event) == {}
    atomic_json(workspace / "state.json", {"session_id": "robot-test", "counter": 13})
    assert process(event) == response
    assert process(event) == {}
    assert process({**event, "session_id": "claude-b"}) == response
    assert process(event) == {}
    assert json.loads((workspace / "state.json").read_text()) == {"session_id": "robot-test", "counter": 13}
    assert sorted(path.name for path in workspace.iterdir()) == ["native-stop", "state.json"]


def test_other_native_events_never_receive_stop_feedback(workspace):
    for name in ("StopFailure", "SessionStart", "SubagentStop", "PostToolUse"):
        assert process({"hook_event_name": name}) == {}
    assert not (workspace / "native-stop").exists()


@pytest.mark.parametrize("state", [
    {"session_id": "another-robot", "counter": 1},
    {"session_id": "robot-test", "counter": -1},
    {"session_id": "robot-test", "counter": True},
    {"session_id": "robot-test", "counter": "1"},
])
def test_invalid_state_does_not_create_a_continuation_marker(workspace, state):
    atomic_json(workspace / "state.json", state)
    with pytest.raises(ValueError):
        process({"hook_event_name": "Stop", "session_id": "claude-a"})
    assert not (workspace / "native-stop").exists()


def test_marker_cannot_escape_workspace(workspace):
    with pytest.raises(ValueError, match="session id"):
        process({"hook_event_name": "Stop", "session_id": "../../escape"})
    config_path = workspace.parents[1] / ".session.json"
    config = json.loads(config_path.read_text())
    config["state_dir"] = str(workspace.parents[2])
    atomic_json(config_path, config)
    with pytest.raises(ValueError, match="inside"):
        process({"hook_event_name": "Stop", "session_id": "claude-a"})
