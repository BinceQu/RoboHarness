"""Compact presentation must preserve decisions and native image delivery."""
import base64
import copy
from hashlib import sha256
import json

from PIL import Image
import pytest

from tools.util.files import atomic_json
from tools.util.observation_hook import process, without_receipt_echoes
from tools.util.receipt import model_receipt, model_receipt_text


def make_receipt(root, *, number=1, role="path_overlay", feed="head"):
    folder = root / "tool_call_history" / f"{number:06d}_adjust_chassis"
    folder.mkdir(parents=True)
    path = folder / "returned_image_0.png"
    Image.new("RGB", (720, 720), (number, 45, 90)).save(path)
    return {"tool_name": "adjust_chassis", "response": {
        "requested": {"forward_m": 2.5, "translation_m": -.9},
        "actual": {"forward_m": .058, "translation_m": -.021},
        "linear_target_reached": False, "obstacle_limited": True,
        "recovery_ok": False, "reason": "base_qvel_stall", "future_measurement": [0.00000000007]*40},
        "images": [{"image_id": f"img_{number}", "feed": feed, "role": role,
                    "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                    "mime_type": "image/png", "width": 720, "height": 720,
                    "source_width": 720, "source_height": 720, "bytes": path.stat().st_size,
                    "caption": "very long historical instruction "*50}],
        "persistent_tracking": {"tracks": {"target": {"status": "lost", "identity_verified": False,
            "source_image_id": "old", "u": 20, "v": 100, "confidence": .011111111,
            "xyz_in_robot_base_coord_m": [.05555, -.3333333333, .202020202]}}},
        "rollout_budget": {"available": True, "used_ticks": 123, "remaining_ticks": 77,
                           "total_ticks": 200, "episode_id": "e", "instance_id": 301},
        "joint_commands_sent": 15, "latest_clickable_image_id": f"img_{number}",
        "task_skill_state": {"baseline": "behavior-v2-baseline", "active_skill": None},
        "observation_instruction": "old redundant guidance", "duration_s": 2.22,
        "perception": str(root / "perception/latest.json"), "record": str(folder / "result.json")}


def test_projection_preserves_every_measurement_error_and_current_binding(tmp_path):
    full = make_receipt(tmp_path)
    full["error"] = "Observation failed after motion: " + "specific detail "*800
    before = copy.deepcopy(full)
    compact = model_receipt(full)
    for key in ("tool_name", "response", "error", "persistent_tracking", "rollout_budget",
                "joint_commands_sent", "latest_clickable_image_id", "task_skill_state"):
        assert compact[key] == full[key]
    assert full == before
    assert model_receipt(compact) == compact
    for key in ("image_id", "feed", "role", "width", "height"):
        assert compact["images"][0][key] == full["images"][0][key]
    assert tmp_path / compact["images"][0]["path"] == full_path(full)
    assert tmp_path / compact["record"] == tmp_path / "tool_call_history/000001_adjust_chassis/result.json"
    assert "sha256" not in compact["images"][0]
    assert compact == json.loads(model_receipt_text(full))


def full_path(receipt):
    from pathlib import Path
    return Path(receipt["images"][0]["path"])


@pytest.mark.parametrize("feed,role,words", [
    ("head", "path_overlay", "0..719"),
    ("head", "raw_rgb", "ORIGINAL PIXELS"),
    ("head", "planning_overlay", "Plan preview only; not clickable"),
    ("left_wrist", "grasp_volume_overlay", "observation only"),
    ("right_wrist", "grasp_volume_overlay", "observation only"),
    ("proprioception", None, "no scene/held objects"),
])
def test_image_role_and_coordinate_limits_remain_explicit(tmp_path, feed, role, words):
    full = make_receipt(tmp_path, feed=feed, role=role)
    compact = model_receipt(full)
    assert words in compact["images"][0]["caption"]
    assert "native Read" in compact["observation_instruction"]


def test_unknown_images_and_source_dimensions_are_not_silently_rewritten(tmp_path):
    full = make_receipt(tmp_path, feed="new_sensor", role="unknown")
    full["images"][0].update(source_width=1440, source_height=1080)
    compact = model_receipt(full)["images"][0]
    assert compact["caption"] == full["images"][0]["caption"]
    assert (compact["source_width"], compact["source_height"]) == (1440, 1080)


def test_skills_and_direct_joint_vectors_survive_whole(tmp_path):
    full = make_receipt(tmp_path)
    full.update(tool_name="joint_control", response={"action": list(range(27)),
        "joint_positions": {f"joint{i}": float(i) for i in range(24)}})
    assert model_receipt(full)["response"] == full["response"]
    skill = {"instructions": "# Skill\n" + "Exact full procedure.\n"*500,
             "images": [], "skill_path": str(tmp_path / "skills/pick-up-object/SKILL.md")}
    assert model_receipt(skill, tmp_path)["instructions"] == skill["instructions"]


@pytest.mark.parametrize("form", ["full", "compact", "pretty", "warnings", "filtered", "redirected", "twice"])
def test_native_hook_one_receipt_exact_pixels_no_loss_of_shell_errors(tmp_path, monkeypatch, form):
    state = tmp_path / ".runtime"
    monkeypatch.setenv("ROBOT_WORKSPACE", str(tmp_path))
    atomic_json(tmp_path / ".session.json", {"workspace": str(tmp_path), "state_dir": str(state)})
    atomic_json(state / "state.json", {"counter": 0, "latest_image_id": ""})
    event = dict(tool_name="Bash", session_id="test", tool_use_id="one")
    process({**event, "hook_event_name": "PreToolUse"})
    receipt = make_receipt(tmp_path)
    atomic_json(tmp_path / "tool_call_history/000001_adjust_chassis/result.json", receipt)
    atomic_json(state / "state.json", {"counter": 1, "latest_image_id": "img_1", "active_skill": None})
    raw = json.dumps(receipt)
    stdout = {"full": raw, "compact": model_receipt_text(receipt),
        "pretty": json.dumps(receipt, indent=2), "warnings": "compiler warning\n"+raw+"\nEXIT=0",
        "filtered": '"linear_target_reached": false', "redirected": "",
        "twice": raw+"\n"+model_receipt_text(receipt)}[form]
    output = process({**event, "hook_event_name": "PostToolUse",
        "tool_response": {"stdout": stdout, "stderr": "shell warning"}})["hookSpecificOutput"]["updatedToolOutput"]
    assert output["stdout"] == stdout and output["stderr"] == "shell warning"
    blocks = output["structuredContent"]
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    expected = model_receipt(receipt)
    expected.pop("observation_instruction")
    caption = expected["images"][0].pop("caption")
    assert sum(json.loads(t) == expected for t in texts if t.startswith('{')) == 1
    assert texts.count(caption) == 1
    assert texts[-1] == caption
    assert "Saved robot observation:" not in "\n".join(texts)
    assert "shell warning" in texts
    if form == "warnings":
        assert "compiler warning" in "\n".join(texts) and "EXIT=0" in "\n".join(texts)
    if form == "filtered":
        assert stdout in texts
    images = [b for b in blocks if b["type"] == "image"]
    assert len(images) == 1
    assert sha256(base64.b64decode(images[0]["source"]["data"])).hexdigest() == receipt["images"][0]["sha256"]
    assert process({**event, "hook_event_name": "PostToolUse", "tool_response": {}}) == {}


def test_unmatched_or_incomplete_json_is_never_silently_discarded(tmp_path):
    receipt = make_receipt(tmp_path)
    altered = copy.deepcopy(receipt)
    altered["response"]["actual"]["forward_m"] = 100
    blocks = [{"type": "text", "text": value} for value in (
        json.dumps(altered), json.dumps(receipt)[:300], '{"other":"data"}', 'compiler error {bad}')]
    assert without_receipt_echoes(blocks, [(1, None, receipt)], tmp_path) == blocks


def test_multiple_receipts_are_adjacent_to_corresponding_images(tmp_path, monkeypatch):
    state = tmp_path / ".runtime"
    monkeypatch.setenv("ROBOT_WORKSPACE", str(tmp_path))
    atomic_json(tmp_path / ".session.json", {"workspace": str(tmp_path), "state_dir": str(state)})
    atomic_json(state / "state.json", {"counter": 0, "latest_image_id": ""})
    event = dict(tool_name="Bash", session_id="test", tool_use_id="batch")
    process({**event, "hook_event_name": "PreToolUse"})
    receipts = [make_receipt(tmp_path, number=n) for n in (1, 2)]
    for receipt in receipts:
        atomic_json(tmp_path / receipt["record"], receipt)
    atomic_json(state / "state.json", {"counter": 2, "latest_image_id": "img_2", "active_skill": None})
    output = process({**event, "hook_event_name": "PostToolUse", "tool_response": {
        "stdout": "\n".join(json.dumps(r) for r in receipts)}})
    blocks = output["hookSpecificOutput"]["updatedToolOutput"]["structuredContent"]
    assert [b["type"] for b in blocks] == ["text", "image", "text", "text", "image", "text"]
    for i, receipt in enumerate(receipts):
        expected = model_receipt(receipt)
        expected.pop("observation_instruction")
        media = expected["images"][0]
        caption = media.pop("caption")
        assert json.loads(blocks[3*i]["text"]) == expected
        assert blocks[3*i+2]["text"] == caption
        assert base64.b64decode(blocks[3*i+1]["source"]["data"]) == full_path(receipt).read_bytes()


def test_late_image_keeps_newer_head_and_skill_binding(tmp_path, monkeypatch):
    state = tmp_path / ".runtime"
    monkeypatch.setenv("ROBOT_WORKSPACE", str(tmp_path))
    atomic_json(tmp_path / ".session.json", {"workspace": str(tmp_path), "state_dir": str(state)})
    atomic_json(state / "state.json", {"counter": 0, "latest_image_id": ""})
    event = dict(tool_name="Bash", session_id="test", tool_use_id="late")
    process({**event, "hook_event_name": "PreToolUse"})
    receipt = make_receipt(tmp_path)
    atomic_json(tmp_path / receipt["record"], receipt)
    atomic_json(state / "state.json", {"counter": 2, "latest_image_id": "newer-head",
                                       "active_skill": "pick-up-object"})
    output = process({**event, "hook_event_name": "PostToolUse", "tool_response": {"stdout": ""}})
    blocks = output["hookSpecificOutput"]["updatedToolOutput"]["structuredContent"]
    assert sum(b["type"] == "image" for b in blocks) == 1
    assert 'Current image="newer-head"' in blocks[-1]["text"]
    assert 'skill="pick-up-object"' in blocks[-1]["text"]
    assert 'Older images are historical' in blocks[-1]["text"]


def test_only_expected_metadata_and_empty_tracking_are_pruned(tmp_path):
    full = make_receipt(tmp_path)
    full["response"].update(tool="adjust_chassis", tool_name="adjust_chassis", tool_version="official_v2",
        dynamic_collision_truth=False, global_base_pose_truth=False, forbidden_observation_reads=[],
        observation_contract=["base_qvel_proprioception", "action_history", "static_r1pro_action_limits"])
    full["persistent_tracking"] = {"available": True, "tracks": {}, "map_marks": [],
        "camera": "head", "coordinate_system": "image_pixels_720x720"}
    full["rollout_budget"].update(source="official_evaluator_episode_steps", sampling="latest_evaluator_observation")
    compact = model_receipt(full)
    assert compact["persistent_tracking"] == {"available": True, "tracks": {}}
    assert compact["response"]["actual"] == full["response"]["actual"]
    assert compact["response"]["linear_target_reached"] is False
    assert "tool_version" not in compact["response"]
    assert "source" not in compact["rollout_budget"]
    for key in ("used_ticks", "remaining_ticks", "total_ticks", "episode_id", "instance_id"):
        assert compact["rollout_budget"][key] == full["rollout_budget"][key]
    full["response"].update(tool="different_operation", tool_version="unexpected", dynamic_collision_truth=True,
        forbidden_observation_reads=["unexpected_sensor"])
    full["persistent_tracking"]["error"] = "tracking failed"
    full["rollout_budget"]["source"] = "unexpected_source"
    compact = model_receipt(full)
    for key in ("tool", "tool_version", "dynamic_collision_truth", "forbidden_observation_reads"):
        assert compact["response"][key] == full["response"][key]
    assert compact["persistent_tracking"] == full["persistent_tracking"]
    assert compact["rollout_budget"]["source"] == "unexpected_source"


def test_missing_head_id_and_missing_pixels_cannot_authorize_clicks(tmp_path):
    full = make_receipt(tmp_path)
    full["images"][0]["image_id"] = ""
    assert "capture a new head image" in model_receipt(full)["images"][0]["caption"]
    full.update(images=[], error="Observation failed", latest_clickable_image_id="")
    compact = model_receipt(full)
    assert compact["error"] == full["error"] and compact["latest_clickable_image_id"] == ""
    assert "observation_instruction" not in compact


def test_required_navigation_tracking_budget_plan_and_record_evidence_survives(tmp_path):
    full = make_receipt(tmp_path)
    full["response"].update(actual={"forward_m": 0.21622831691399758},
        chassis_forward_2m="No object on the chassis-forward path within 2m",
        plan_id="plan_007", verification="direct_evaluator_base_qvel_integration")
    compact = model_receipt(full)
    for key in ("actual", "chassis_forward_2m", "plan_id"):
        assert compact["response"][key] == full["response"][key]
    for key in ("persistent_tracking", "rollout_budget", "joint_commands_sent"):
        assert compact[key] == full[key]
    assert "verification" not in compact["response"]
    assert tmp_path / compact["record"] == tmp_path / full["record"]
    assert tmp_path / compact["images"][0]["path"] == full_path(full)
    # An unfamiliar structured diagnostic is not static prose.
    full["response"]["verification"] = {"error": "unexpected sensor disagreement", "delta_m": .03}
    assert model_receipt(full)["response"]["verification"] == full["response"]["verification"]
    del full["record"]
    full["response"]["verification"] = "unarchived result"
    assert model_receipt(full)["response"]["verification"] == "unarchived result"


@pytest.mark.parametrize("relative", [False, True])
def test_delivery_auditor_accepts_public_paths_and_uses_saved_hashes(tmp_path, monkeypatch, relative):
    from pathlib import Path
    evaluator = Path(__file__).resolve().parents[5] / "eval/runner/audit"
    monkeypatch.syspath_prepend(str(evaluator))
    from audit_episode_delivery import audit
    root = tmp_path / "workspace"
    receipt = make_receipt(root, number=2)
    atomic_json(Path(receipt["record"]), receipt)
    public = model_receipt(receipt, root) if relative else receipt
    event = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "one",
        "content": [{"type": "text", "text": json.dumps(public)}, {"type": "image", "source": {
            "type": "base64", "data": base64.b64encode(full_path(receipt).read_bytes()).decode()}}]}]}}
    log = root / ".runtime/claude-home/projects/offline/session.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps(event) + "\n")
    report = audit(tmp_path)
    assert report["total_image_results"] == report["verified_image_results"] == 1
