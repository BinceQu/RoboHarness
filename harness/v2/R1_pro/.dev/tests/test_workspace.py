"""Boundary regression tests: no simulator, no model and no external mutation."""
import ast
import asyncio
import base64
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import pytest

from tools.util.driver import Driver, ControlServer, ControlHandler, serve
from tools.util.model import ACTION_NAMES, ACTION_SLICES, PROPRIO_SLICES, validate_action, joints, forward
from tools.util.protocol import request, pack, unpack
from tools.util.session import create
from tools.util.tool import run
from tools.util.receipt import tool_error, without_ok

TEMPLATE = Path(__file__).resolve().parents[2]
MESH_ROOT = TEMPLATE / "robot_config"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[path.stem] = result
    spec.loader.exec_module(result)
    return result


def bulk_record(receipt, name):
    """Bulk evidence moved out of the model workspace in bbd8074."""
    record = Path(receipt["record"])
    config = json.loads((record.parents[2] / ".session.json").read_text())
    return Path(config["trajectory_dir"]) / "calls" / record.parent.name / name


def raw_observation():
    proprio = np.zeros(65, dtype=np.float32)
    proprio[3:11] = proprio[30:38] = [0, 0, 0, -1, 0, -.5, 0, 0]
    proprio[19:22], proprio[46:49] = [.45, .3, .7], [.45, -.3, .7]
    proprio[25] = proprio[52] = 1
    rgb = np.zeros((720, 720, 3), dtype=np.uint8)
    rgb[..., :] = [40, 100, 160]
    rgb[360:450, 430:520] = [220, 40, 15]
    raw = {"robot_r1::proprio": proprio,
           "robot_r1::cam_rel_poses": np.array([.5, .3, .7, 0, 0, 0, 1,
                .5, -.3, .7, 0, 0, 0, 1, 0, 0, 1.2, 0, 0, 0, 1], dtype=np.float32),
           "privileged_object_poses": {"must_not_reach_agent": True}, "task_id": 1,
           "__behavior_rollout_budget__": {"max_steps": 10535, "remaining_steps": 10400}}
    for role in ("head", "left_wrist", "right_wrist"):
        raw[f"robot_r1::{role}::rgb"] = rgb.copy()
        raw[f"robot_r1::{role}::depth_linear"] = np.ones((720, 720), dtype=np.float32)
    return raw


def test_official_websocket_health_reset_and_action():
    """Use the official wire order, including its required plain HTTP probe."""
    import socket
    from urllib.request import urlopen
    from websockets.sync.client import connect

    async def scenario():
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        driver = Driver()
        server = asyncio.create_task(serve(driver, "127.0.0.1", port))

        def client():
            deadline = time.monotonic() + 10
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as reply:
                        assert reply.status == 200 and reply.read() == b"OK\n"
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as connection:
                assert unpack(connection.recv())["action_dim"] == 27
                connection.send(pack({"reset": True}))
                connection.send(pack(raw_observation()))
                action = unpack(connection.recv(timeout=10))["action"]
                assert action.shape == (27,) and np.isfinite(action).all()
                assert driver.sequence == 1 and driver.ticks == 1

        try:
            await asyncio.to_thread(client)
        finally:
            server.cancel()
            try:
                await server
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("EEF_NEAR_SAM2", "0")
    # AF_UNIX names are limited to 108 bytes; tmp_path can exceed that.
    import tempfile
    socket_dir = Path(tempfile.mkdtemp(prefix="r1-test-"))
    socket = socket_dir / "control.sock"
    driver = Driver()
    driver.connected = True
    driver.initializer._state = "ready"
    server = ControlServer(str(socket), ControlHandler)
    server.driver = driver
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    observed = raw_observation()
    errors = []

    def evaluator():
        try:
            while driver.connected:
                action = driver.observe(observed)
                for name in ("trunk", "arm_left", "arm_right"):
                    observed["robot_r1::proprio"][PROPRIO_SLICES[name + "_qpos"]] = action[ACTION_SLICES[name]]
                observed["robot_r1::proprio"][:3] = action[:3] * [.75, .75, 1]
                for name in ("gripper_left", "gripper_right"):
                    positions = observed["robot_r1::proprio"][PROPRIO_SLICES[name + "_qpos"]]
                    positions[:] = np.clip(positions + action[ACTION_SLICES[name]] * .005, 0, .05)
        except RuntimeError as exc:
            if driver.connected:
                errors.append(exc)

    evaluating = threading.Thread(target=evaluator, daemon=True)
    evaluating.start()
    with driver.condition:
        driver.condition.wait_for(lambda: driver.sequence >= 1, timeout=5)
    config = create(TEMPLATE, tmp_path / "workspace", socket, session_id="unit-session", task_name="picking_up_trash", with_environment=False)
    monkeypatch.setenv("ROBOT_WORKSPACE", config["workspace"])
    try:
        yield driver, config, observed
        # Every public history written by any tool/skill in these integration tests
        # must use the same flag-free contract, including nested diagnostic results.
        for record in (Path(config["workspace"]) / "tool_call_history").glob("*/*.json"):
            if record.name in {"result.json", "output.json"}:
                value = json.loads(record.read_text())
                assert value == without_ok(value), str(record)
    finally:
        driver.disconnected()
        server.shutdown()
        server.server_close()
        evaluating.join(5)
    assert not evaluating.is_alive()
    assert not errors


def invoke(name, arguments):
    tool = module(TEMPLATE / "tools" / f"{name}.py")
    result = run(name, tool.SCHEMA, tool.actions, arguments)
    assert result == without_ok(result)
    return result


def test_partial_chassis_progress_survives_public_cli(live, monkeypatch, capsys):
    import sys
    from tools.util.tool import main
    def stopped(robot, forward, translation):
        robot.set_result({"ok": True, "requested": {"forward_m": forward, "translation_m": translation},
            "actual": {"forward_m": .058, "translation_m": -.021},
            "linear_target_reached": False, "obstacle_limited": True})
        yield from ()
    monkeypatch.setattr(sys, "argv", ["adjust_chassis.py", '{"forward":2.5,"translation":-0.9}'])
    main("adjust_chassis", {"type": "object"}, stopped)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == without_ok(receipt)
    assert receipt["response"]["requested"] == {"forward_m": 2.5, "translation_m": -.9}
    assert receipt["response"]["actual"] == {"forward_m": .058, "translation_m": -.021}
    assert receipt["response"]["linear_target_reached"] is False
    assert receipt["response"]["obstacle_limited"] is True
    assert receipt["images"] and receipt["joint_commands_sent"] == 0


@pytest.mark.parametrize("failure,tracking", [
    ("tracking_error", {"max_tracking_error_rad": .05, "stall_steps": 20,
                        "waypoint_timeout_s": .5}),
    ("tracking_stall", {"max_tracking_error_rad": .5, "stall_steps": 2,
                        "waypoint_timeout_s": .5}),
    ("tracking_timeout", {"max_tracking_error_rad": .5, "stall_steps": 100,
                          "waypoint_timeout_s": .067}),
])
def test_exec_plan_recovery_pins_observed_joints_across_ipc(live, monkeypatch, failure, tracking):
    """Exercise the real recovery branches that failed in task01 v5 call 36."""
    from tools.util.robot import Robot
    driver, _, _ = live
    robot = Robot()
    tool = module(TEMPLATE / "tools/exec_plan_pose.py")
    observed_arm = driver.world.arm_qpos_list("right")
    target = list(observed_arm)
    target[0] += .1
    getter = driver.world.arm_qpos_list
    monkeypatch.setattr(driver.world, "arm_qpos_list",
        lambda arm: list(observed_arm) if arm == "right" else getter(arm))
    trajectory = {
        "active_arm": "right",
        "parameters": {"reset_tool_roll_at_start": True, "back_m": .1},
        "tracking": {"stationary_qvel_tolerance_rad_s": 1.,
                     "start_trunk_tolerance_rad": .025,
                     "settle_tolerance_rad": .001, **tracking},
        "start_state": {"episode_id": robot.episode_id(),
                        "motion_epoch": robot.motion_epoch(),
                        "trunk_qpos": robot.trunk_qpos().tolist()},
    }
    # Planning and initial gripper preparation are independent of this
    # regression. Keep the actual execution loop, controller, IPC and receipt.
    monkeypatch.setattr(tool, "_load_plan_trajectory",
        lambda *args, **kwargs: (trajectory, "offline-plan.json", {}))
    monkeypatch.setattr(tool, "_resolve_exec_final_q",
        lambda **kwargs: (np.asarray(target), {"source": "signed_final_q"}))
    monkeypatch.setattr(tool, "_prepare_exec_segments",
        lambda **kwargs: ([{"name": "approach", "waypoints": [target],
                            "endpoint_q": target}], {}))
    def prepared(*args, result_out, **kwargs):
        result_out.update(ok=True, action_steps=0)
        return iter(())
    monkeypatch.setattr(tool, "_ensure_exec_selected_gripper_open", prepared)
    monkeypatch.setattr(tool, "_reset_and_lock_selected_j8", prepared)
    result = run(tool.NAME, tool.SCHEMA, tool.actions, {"plan_id": "offline-plan"}, robot=robot)
    assert tool_error(result)
    assert "read-only" not in str(result)
    full = json.loads(bulk_record(result, "output.json").read_text())
    assert full["error"] is None
    assert full["operation_result"]["execution"] == "stopped_on_" + failure
    assert result["images"], "A stopped motion must still return a fresh observation"
    records = [json.loads(line) for line in
        bulk_record(result, "joint_commands.jsonl").read_text().splitlines()]
    assert len(records) == full["operation_result"]["action_steps"] + 1
    last = records[-1]["action"]
    assert np.allclose(last[ACTION_SLICES["arm_right"]], observed_arm)
    assert last[6] == 0


def test_end_to_end_perception_pixels_actions_and_hooks(live, monkeypatch):
    driver, config, raw = live
    workspace = Path(config["workspace"])
    # None of these ordinary reads/actions may load or render robot meshes.
    import tools.util.render as rendering
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError("Ordinary tools must not load/render robot meshes")
    monkeypatch.setattr(rendering, "RobotRenderer", forbidden_renderer)
    from tools.util.observation_hook import process
    event = {"tool_name": "Bash", "hook_event_name": "PreToolUse", "session_id": "claude-test", "tool_use_id": "call-test"}
    assert process(event) == {}
    # Session initialization uses this read-only path without an action generator.
    capture = run("capture_head_camera", {"type": "object", "additionalProperties": False}, None, {})
    assert (not tool_error(capture)), capture
    assert driver.ticks == 0, "read-only capture must not burn physics budget"
    manifest = json.loads((workspace / "perception/latest.json").read_text())
    assert manifest["sequence"] == driver.sequence
    assert Image.open(workspace / "perception/exteroception/head_rgb.png").size == (720, 720)
    assert np.array_equal(np.asarray(Image.open(manifest["images"]["head_rgb"]["path"])), raw["robot_r1::head::rgb"])
    archived = json.loads(bulk_record(capture, "observation/manifest.json").read_text())
    depth = np.load(archived["archive"]["head_depth"]["path"])["depth_mm"]
    assert np.array_equal(depth, raw["robot_r1::head::depth_linear"] * 1000)
    assert Image.open(workspace / "perception/exteroception/head_depth.png").size == (720, 720)
    assert "robot_three_views" not in manifest
    assert not (workspace / "perception/proprioception/robot_three_views.png").exists()
    assert manifest["proprioception_render_tool"] == "python tools/read_proprioception.py"
    assert "privileged_object_poses" not in driver.payload()["snapshot"]["observation"]
    event.update(hook_event_name="PostToolUse", tool_response={"stdout": json.dumps(capture), "stderr": "", "interrupted": False})
    output = process(event)["hookSpecificOutput"]["updatedToolOutput"]
    images = [block for block in output["structuredContent"] if block["type"] == "image"]
    assert len(images) == 1
    assert sha256(base64.b64decode(images[0]["source"]["data"])).hexdigest() == capture["images"][0]["sha256"]
    first_id = manifest["head_image_id"]
    # The following direct Python calls represent one additional native Bash
    # invocation. Bracket the batch so all its receipts are delivered once.
    event.update(hook_event_name='PreToolUse', tool_use_id='batch')
    process(event)
    point = invoke("measure_shoulder_distance", {"image_id": first_id, "u": 475, "v": 405})
    assert (not tool_error(point)), point
    record = Path(point["record"]).parent
    assert Image.open(record / f"click_{first_id}_crosshair.png").size == (720, 720)
    assert json.loads(bulk_record(point, "selected_points.json").read_text())[0]["u"] == 475
    stale = invoke("measure_shoulder_distance", {"image_id": first_id, "u": 475, "v": 405})
    assert tool_error(stale) and "latest" in stale["error"]
    action = driver.world.hold_action().tolist()
    action[22] = -.6
    action[14], action[24] = .2, -.3
    direct = invoke("robot_step", {"action": action, "steps": 2})
    assert (not tool_error(direct)), direct
    assert direct["joint_commands_sent"] == 2
    assert len(direct["response"]["action"]) == 27
    assert len(direct["response"]["joint_positions"]) == 24
    assert np.isclose(driver.world.arm_pin_qpos_list("right")[5], -.6)
    assert np.isclose(driver.world.arm_qpos_list("left")[7], .2)
    assert np.isclose(driver.world.arm_qpos_list("right")[7], -.3)
    assert np.isclose(json.loads((workspace / "perception/proprioception/joint_state.json").read_text())["joint_positions"]["right_arm_joint6"], -.6)
    records = bulk_record(direct, "joint_commands.jsonl").read_text().splitlines()
    assert len(records) == 2 and all(len(json.loads(row)["action"]) == 27 for row in records)
    assert all(np.array_equal(np.asarray(json.loads(row)["action"], dtype=np.float32),
                              np.asarray(action, dtype=np.float32)) for row in records)
    bad = action.copy()
    bad[6] = .1
    before = driver.ticks
    invalid = invoke("robot_step", {"action": bad})
    assert tool_error(invalid) and driver.ticks == before
    # Custom Python scripts use the same session-local source and history.
    from tools.util.robot import Robot
    robot = Robot()
    repeated = robot.command(action, steps=1)
    assert (not tool_error(repeated)) and repeated["joint_commands_sent"] == 1
    assert Path(repeated["record"]).is_relative_to(workspace)
    assert robot.capture("head")["images"]
    event.update(hook_event_name='PostToolUse', tool_response={'stdout': 'batch complete'})
    assert process(event)['hookSpecificOutput']['updatedToolOutput']['structuredContent']
    # Neither an old episode nor a second session may take over this driver.
    for change in ({"session_id": "other-session"}, {"episode_id": "expired"}):
        with pytest.raises(ValueError):
            request(config["socket"], {"operation": "observe", "session_id": config["session_id"],
                "episode_id": config["episode_id"], **change})
    # A normal Bash command must never reattach an old robot image.
    event.update(hook_event_name="PreToolUse", tool_use_id="unrelated")
    assert process(event) == {}
    event.update(hook_event_name="PostToolUse", tool_response={"stdout": "hello"})
    assert process(event) == {}


def test_proprioception_on_demand_measured_joints_and_native_image(live, monkeypatch):
    from tools.read_proprioception import run as read_proprioception
    from tools.util.observation_hook import process
    from tools.util.render import RobotRenderer
    driver, config, _ = live
    workspace = Path(config["workspace"])
    # A real prior command changes measured posture; render must use that qpos.
    action = driver.world.hold_action().tolist()
    action[22] = -.7
    moved = invoke("robot_step", {"action": action, "steps": 1})
    assert (not tool_error(moved)), moved
    latest = (workspace / "perception/latest.json").read_bytes()
    head = (workspace / "perception/exteroception/head_rgb.png").resolve()
    state_path = Path(config["state_dir"]) / "state.json"
    state_before = json.loads(state_path.read_text())
    before_ticks, sequence = driver.ticks, driver.sequence
    calls = []
    render = RobotRenderer.three_views
    def measured(renderer, positions, **kwargs):
        calls.append((dict(positions), kwargs["sequence"]))
        return render(renderer, positions, **kwargs)
    monkeypatch.setattr(RobotRenderer, "three_views", measured)
    event = {"tool_name": "Bash", "hook_event_name": "PreToolUse", "session_id": "native-pose",
             "tool_use_id": "read-pose"}
    process(event)
    result = read_proprioception({})
    assert (not tool_error(result)), result
    assert result["joint_commands_sent"] == 0 and driver.ticks == before_ticks
    assert driver.sequence == sequence
    assert len(calls) == 1 and calls[0][1] == sequence
    assert np.isclose(calls[0][0]["right_arm_joint6"], -.7)
    assert result["latest_clickable_image_id"] == moved["latest_clickable_image_id"]
    record = Path(result["record"]).parent
    assert not bulk_record(result, "joint_commands.jsonl").exists()
    assert not bulk_record(result, "observation/head_rgb.png").exists()
    assert (workspace / "perception/latest.json").read_bytes() == latest
    assert (workspace / "perception/exteroception/head_rgb.png").resolve() == head
    state_after = json.loads(state_path.read_text())
    assert state_after.pop("counter") == state_before.pop("counter") + 1
    assert state_after == state_before
    metadata = json.loads((workspace / "perception/proprioception/render.json").read_text())
    assert metadata["sequence"] == sequence
    assert json.loads(Path(metadata["joint_state"]).read_text())["joint_positions"] == calls[0][0]
    picture = result["images"][0]
    assert Image.open(picture["path"]).size == (1320, 650)
    assert picture["clickable"] is False
    assert (workspace / "perception/proprioception/robot_three_views.png").read_bytes() == Path(picture["path"]).read_bytes()
    assert sha256(Path(picture["path"]).read_bytes()).hexdigest() == picture["sha256"]
    output = process({**event, "hook_event_name": "PostToolUse",
        "tool_response": {"stdout": json.dumps(result)}})["hookSpecificOutput"]["updatedToolOutput"]
    images = [block for block in output["structuredContent"] if block["type"] == "image"]
    assert len(images) == 1
    assert sha256(base64.b64decode(images[0]["source"]["data"])).hexdigest() == picture["sha256"]

    # Later actions update measured joints, but never rewrite or present the old
    # posture render as part of the new observation's coherent manifest.
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError("Unexpected mesh loading")
    monkeypatch.setattr("tools.util.render.RobotRenderer", forbidden_renderer)
    action[22] = -.5
    next_action = invoke("robot_step", {"action": action, "steps": 1})
    assert (not tool_error(next_action)), next_action
    assert not bulk_record(next_action, "observation/robot_three_views.png").exists()
    current = json.loads((workspace / "perception/proprioception/joint_state.json").read_text())
    assert current["sequence"] > metadata["sequence"]
    assert np.isclose(current["joint_positions"]["right_arm_joint6"], -.5)
    assert json.loads((workspace / "perception/proprioception/render.json").read_text()) == metadata
    assert sha256(Path(picture["path"]).read_bytes()).hexdigest() == picture["sha256"]
    assert "robot_three_views" not in json.loads((workspace / "perception/latest.json").read_text())

    # Rendering failure must not attach that earlier PNG or move the robot.
    ticks = driver.ticks
    failed = read_proprioception({})
    assert tool_error(failed) and "Unexpected mesh loading" in failed["error"]
    assert failed["images"] == [] and driver.ticks == ticks
    assert json.loads(Path(failed["record"]).read_text())["error"] == failed["error"]
    assert json.loads((workspace / "perception/proprioception/render.json").read_text()) == metadata


def test_proprioception_validates_before_render_and_works_without_camera(live, monkeypatch):
    from tools.read_proprioception import run as read_proprioception
    driver, config, _ = live
    called = []
    class Renderer:
        def __init__(self, root):
            called.append(root)
        def three_views(self, positions, **kwargs):
            return Image.new("RGB", (1320, 650))
        def close(self):
            pass
    monkeypatch.setattr("tools.util.render.RobotRenderer", Renderer)
    invalid = read_proprioception({"steps": 1})
    assert tool_error(invalid) and "ValidationError" in invalid["error"]
    assert not called and driver.ticks == 0
    valid = read_proprioception({})
    assert (not tool_error(valid)), valid
    assert len(called) == 1 and driver.ticks == 0
    assert valid["latest_clickable_image_id"] == ""
    assert not (Path(config["workspace"]) / "perception/latest.json").exists()


def test_named_joint_control_moves_j8_and_holds_omitted_targets(live):
    driver, _, _ = live
    tool = module(TEMPLATE / "robot_config/control.py")
    initial = {"joints": {"left_arm_joint8": .25, "right_arm_joint8": -.4},
               "left_finger_effort": -5, "right_finger_effort": -3, "steps": 2}
    first = run("joint_control", tool.SCHEMA, tool.actions, initial)
    assert (not tool_error(first)), first
    assert np.isclose(driver.world.arm_qpos_list("left")[7], .25)
    assert np.isclose(driver.world.arm_qpos_list("right")[7], -.4)
    for role in ("head", "left_wrist", "right_wrist"):
        before = driver.ticks
        captured = invoke(f"capture_{role}_camera", {})
        assert (not tool_error(captured)), captured
        assert driver.ticks == before
        assert driver.world.gripper_pin_effort_list("left") == [-5, -5]
        assert driver.world.gripper_pin_effort_list("right") == [-3, -3]
    second = run("joint_control", tool.SCHEMA, tool.actions,
                 {"joints": {"right_arm_joint6": -.6}, "steps": 1})
    assert (not tool_error(second)), second
    assert len(second["response"]["action"]) == 27
    assert np.isclose(second["response"]["joint_positions"]["right_arm_joint8"], -.4)
    action = json.loads(bulk_record(second, "joint_commands.jsonl").read_text())["action"]
    assert np.allclose([action[14], action[24], action[22]], [.25, -.4, -.6])
    assert action[15:17] == [-5, -5] and action[25:27] == [-3, -3]


@pytest.mark.parametrize("primitive", ["named_omitted", "named_explicit", "raw_explicit"])
def test_direct_finger_targets_survive_following_tool_hold(live, primitive):
    """A later tool must not silently turn carrying effort into a qpos servo."""
    driver, _, _ = live
    with driver.condition:
        driver.world.latch_gripper_close_keepalive("left", [-5., -5.])
        driver.world.latch_gripper_close_keepalive("right", [-20., -20.])
    expected = [-5., -5., -20., -20.]
    if primitive == "raw_explicit":
        action = driver.world.hold_action().tolist()
        action[15:17], action[25:27] = [-.01, -.02], [2., 3.]
        expected = [-.01, -.02, 2., 3.]
        first = invoke("robot_step", {"action": action, "steps": 2})
    else:
        tool = module(TEMPLATE / "robot_config/control.py")
        arguments = {"joints": {"right_arm_joint6": -.6}, "steps": 2}
        if primitive == "named_explicit":
            arguments.update(left_finger_effort=0., right_finger_effort=-.02)
            expected = [0., 0., -.02, -.02]
        first = run("joint_control", tool.SCHEMA, tool.actions, arguments)
    assert (not tool_error(first)), first

    def hold(robot):
        for _ in range(2):
            yield robot.hold_action()
        robot.set_result({"ok": True})

    follow = run("following_tool_hold", {"type": "object"}, hold, {})
    assert (not tool_error(follow)), follow
    records = bulk_record(follow, "joint_commands.jsonl").read_text().splitlines()
    assert len(records) == 2
    for row in records:
        command = json.loads(row)["action"]
        assert np.allclose(command[15:17] + command[25:27], expected, atol=1e-8)
    opened = invoke("open_gripper", {"arm": "right"})
    assert (not tool_error(opened)), opened
    after_open = run("following_open_hold", {"type": "object"}, hold, {})
    assert (not tool_error(after_open)), after_open
    for row in bulk_record(after_open, "joint_commands.jsonl").read_text().splitlines():
        command = json.loads(row)["action"]
        assert np.allclose(command[15:17], expected[:2], atol=1e-8)
        assert all(value >= 0 for value in command[25:27])
    assert not driver.world.gripper_keepalive_status()["right"]["direct_effort"]
    with driver.condition:
        driver.world.reset_observation_state()
        assert all(not value["direct_effort"] and not value["active"]
                   for value in driver.world.gripper_keepalive_status().values())


def test_nominal_and_wire_rotary_endpoints_without_broad_tolerance():
    from tools.util.model import position_command_bounds, LOCKED_INDICES
    indices, _ = position_command_bounds()
    for index in indices:
        if index in LOCKED_INDICES:
            continue
        for side, direction in (("lower", -np.inf), ("upper", np.inf)):
            limit = joints()[ACTION_NAMES[index]]["limits"][side]
            wire = float(np.float32(limit))
            action = [0.] * 27
            for endpoint in (limit, wire):
                action[int(index)] = endpoint
                assert validate_action(action)[index] == np.float32(limit)
            envelope = min(limit, wire) if side == "lower" else max(limit, wire)
            action[int(index)] = float(np.nextafter(envelope, direction))
            with pytest.raises(ValueError):
                validate_action(action)


@pytest.mark.parametrize("primitive", ["named", "raw"])
def test_documented_torso_endpoint_through_public_api(live, primitive):
    if primitive == "named":
        tool = module(TEMPLATE / "robot_config/control.py")
        result = run("joint_control", tool.SCHEMA, tool.actions,
                     {"joints": {"torso_joint2": 2.5307}, "steps": 1})
    else:
        action = live[0].world.hold_action().tolist()
        action[4] = 2.5307
        result = invoke("robot_step", {"action": action, "steps": 1})
    assert (not tool_error(result)), result
    assert result["joint_commands_sent"] == 1
    command = json.loads(bulk_record(result, "joint_commands.jsonl").read_text())["action"]
    assert command[4] == float(np.float32(2.5307))


def test_measured_limit_residuals_hold_valid_targets_without_relaxing_agent_api(live):
    """Replay the actual task01 failures through the local controller and wire."""
    from tools.util.robot import Robot
    driver, _, observed = live
    residuals = {4: 2.5307018756866455, 12: -1.0471982955932617,
                 18: .17450237274169922, 22: -1.0471981763839722}
    with driver.condition:
        for group in ("trunk", "arm_left", "arm_right"):
            for action_index, value in residuals.items():
                slot = ACTION_SLICES[group]
                if slot.start <= action_index < slot.stop:
                    proprio_index = PROPRIO_SLICES[group + "_qpos"].start + action_index - slot.start
                    observed["robot_r1::proprio"][proprio_index] = value
                    driver.adapter.current.observation["robot_r1::proprio"][proprio_index] = value
    robot = Robot()
    original = robot._official_adapter.proprio_vector()
    command = robot.hold_action().tolist()
    validate_action(command)
    assert np.array_equal(robot._official_adapter.proprio_vector(), original)

    # Public raw/named targets still reject even these small violations.
    invalid = command.copy()
    invalid[18] = residuals[18]
    before = driver.ticks
    raw = invoke("robot_step", {"action": invalid})
    assert tool_error(raw) and raw["joint_commands_sent"] == 0
    named_tool = module(TEMPLATE / "robot_config/control.py")
    named = run("joint_control", named_tool.SCHEMA, named_tool.actions,
                {"joints": {"right_arm_joint2": residuals[18]}})
    assert tool_error(named) and named["joint_commands_sent"] == 0
    assert driver.ticks == before
    assert np.array_equal(driver.adapter.proprio_vector(), original)

    def actions(robot):
        # Internal baseline tools pin measured positions before holding or
        # interpolating. Exercise both fresh observations and these pins.
        robot.set_trunk_pin_qpos(robot.trunk_qpos())
        for arm in ("left", "right"):
            robot.set_arm_pin_qpos(arm, robot.arm_qpos_list(arm))
        for _ in range(2):
            yield robot.make_action(base=[.2, -.1, .3],
                gripper_effort_left=[-5., -5.], gripper_effort_right=[-3., -3.])
        robot.set_result({"ok": True})

    result = run("limit_residual_regression", {"type": "object"}, actions, {})
    assert (not tool_error(result)), result
    assert result["joint_commands_sent"] == 2
    rows = bulk_record(result, "joint_commands.jsonl").read_text().splitlines()
    assert len(rows) == 2
    for row in rows:
        action = json.loads(row)["action"]
        validate_action(action)
        for index, residual in residuals.items():
            bound = joints()[ACTION_NAMES[index]]["limits"]["lower" if residual < 0 else "upper"]
            assert action[index] == float(np.float32(bound))
        assert action[15:17] == [-5, -5] and action[25:27] == [-3, -3]
        assert np.allclose(action[:3], [.2, -.1, .3])


def test_height_hold_failure_reaches_agent_receipt_with_current_image(live):
    driver, _, observed = live
    start = [-.9914950132369995, 2.5306994915008545, .3827572166919708, -5.006653282180196e-6]
    with driver.condition:
        observed["robot_r1::proprio"][PROPRIO_SLICES["trunk_qpos"]] = start
        driver.adapter.current.observation["robot_r1::proprio"][PROPRIO_SLICES["trunk_qpos"]] = start
    result = invoke("adjust_height", {"upward": .05})
    assert tool_error(result), result
    assert "no height change was commanded" in result["response"]["error"]
    assert result["images"] and result["latest_clickable_image_id"]
    assert result["joint_commands_sent"] > 0
    full = json.loads(bulk_record(result, "output.json").read_text())["operation_result"]
    assert full["unfulfilled_hold"] and not full["height_converged"]
    assert abs(full["actual_upward_m"]) < 1e-4


def test_multi_point_tracking_keeps_original_clicks_and_replaces_named_set(live):
    _, config, raw = live
    workspace = Path(config["workspace"])
    captured = invoke("capture_head_camera", {})
    image_id = captured["latest_clickable_image_id"]
    selected = [{"name": "held", "u": 445, "v": 380},
                {"name": "interior", "u": 500, "v": 430}]
    tracked = invoke("track_object_distance", {"image_id": image_id, "points": selected})
    assert (not tool_error(tracked)), tracked
    record = Path(tracked["record"]).parent
    stored = json.loads(bulk_record(tracked, "selected_points.json").read_text())
    assert stored == [{"image_id": image_id, "u": point["u"], "v": point["v"],
                       "label": point["name"]} for point in selected]
    original = np.asarray(Image.open(Path(config["trajectory_dir"]) / "recordings"
        / config["session_id"] / "images" / f"{image_id}.png"))
    marked = np.asarray(Image.open(record / f"click_{image_id}_crosshair.png"))
    assert np.array_equal(original, raw["robot_r1::head::rgb"])
    assert marked.shape == original.shape
    for point in selected:
        u, v = point["u"], point["v"]
        assert not np.array_equal(marked[v-16:v+17, u-16:u+17], original[v-16:v+17, u-16:u+17])
    result = json.loads(bulk_record(tracked, "output.json").read_text())["operation_result"]
    assert set(result["tracked_names"]) == {"held", "interior"}
    assert result["image_id"] == image_id
    assert (workspace / "perception/exteroception/tracks.json").read_bytes() == bulk_record(tracked, "observation/tracks.json").read_bytes()
    rebound = invoke("track_object_distance", {
        "image_id": tracked["latest_clickable_image_id"], "points": selected[:1]})
    assert (not tool_error(rebound)), rebound
    result = json.loads(bulk_record(rebound, "output.json").read_text())["operation_result"]
    assert result["tracked_names"] == ["held"]
    assert set(result["cleared_names"]) == {"held", "interior"}
    # The same transport also serves generators with no result callback, and a
    # cancelled local commit guard must not change the driver's named set.
    from contextlib import nullcontext
    from tools.util.coordinates import arguments_to_interface
    from tools.util import motion_common as p
    from tools.util.robot import Robot
    binding = p._load_tracker_bound_capture(config["session_id"], rebound["latest_clickable_image_id"])
    capture = binding["capture"]
    kwargs = {"session_id": config["session_id"], "image_id": rebound["latest_clickable_image_id"],
              "capture_observation_sequence": capture.evaluator_sequence,
              "capture_episode_id": binding["episode_id"], "capture_image_shape": capture.depth.shape,
              "capture_frame_digest": binding["frame_digest"]}
    robot = Robot()
    with robot.exclusive():
        tracker = robot._official_tracked_object_distances
        before = tracker.memory_fields()
        with pytest.raises(ValueError, match="cancelled before commit"):
            tracker.replace_points(arguments_to_interface(selected), **kwargs,
                                   commit_guard=lambda: nullcontext(False))
        assert tracker.memory_fields() == before
        checked = []
        bound = tracker.replace_points(arguments_to_interface(selected), **kwargs,
                                       check_cancelled=lambda: checked.append(True))
        assert checked == [True]
        assert set(bound["entries"]) == {"held", "interior"}


def test_skills_and_workspace_isolation(live, tmp_path):
    driver, config, _ = live
    from tools.util.skills import run as run_skill
    from tools.activate_skill import activate
    from tools.deactivate_skill import deactivate
    def transition(name, arguments):
        return run_skill(name, activate if name == "activate_skill" else deactivate, arguments)
    from tools.util.session_context import build_context, context_parts, current_state
    result = transition("activate_skill", {"name": "place-object-in-container"})
    assert (not tool_error(result))
    context = build_context(current_state())
    assert "".join(context_parts(context)) == context
    assert result["instructions"].strip() in context
    assert transition("deactivate_skill", {"name": "place-object-in-container"})["task_skill_state"]["active_skill"] is None
    workspace = Path(config["workspace"])
    assert not (workspace / "runtime").exists()
    assert (workspace / ".claude/skills").resolve() == workspace / "skills"
    assert (workspace / "tools/adjust_chassis.py").stat().st_ino != (TEMPLATE / "tools/adjust_chassis.py").stat().st_ino
    for path in workspace.rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(workspace)
    with pytest.raises(FileExistsError):
        create(TEMPLATE, workspace, config["socket"], session_id="another", task_name="picking_up_trash", with_environment=False)


def test_gripper_feedback_generators_use_real_context_signature(live):
    driver, _, _ = live
    opened = invoke("open_gripper", {"arm": "right"})
    assert (not tool_error(opened)), opened
    assert np.allclose(driver.world.gripper_qpos_list("right"), [.05, .05])
    closed = invoke("close_gripper", {"arm": "right"})
    # The fixture contains no object: reaching both lower stops must remain
    # an explicit empty grasp, not be misreported as a successful grasp.
    assert tool_error(closed) and "error" not in closed, closed
    assert closed["response"]["failure_stage"] == "empty_close"
    assert closed["response"]["actuation_detected"]
    assert closed["response"]["grasp_confirmed"] is False
    assert np.allclose(driver.world.gripper_qpos_list("right"), [0, 0])
    reopened = invoke("open_gripper", {"arm": "right"})
    assert (not tool_error(reopened)), reopened
    assert np.allclose(driver.world.gripper_qpos_list("right"), [.05, .05])


def test_rejected_cli_calls_leave_history_without_stale_images(live, monkeypatch, capsys):
    import sys
    from tools.util import tool, skills
    driver, config, _ = live
    before = driver.ticks

    def disconnected():
        raise ConnectionError("Evaluator unavailable")

    monkeypatch.setattr(tool, "Robot", disconnected)
    for value in ("{broken", "{}"):
        monkeypatch.setattr(sys, "argv", ["capture_head_camera.py", value])
        tool.main("capture_head_camera", {"type": "object"}, lambda robot: None)
        result = json.loads(capsys.readouterr().out)
        assert tool_error(result) and result["images"] == []
        assert "history_error" not in result
        record = Path(config["workspace"]) / result["record"]
        assert record.is_file()
        saved = json.loads(record.with_name("input.json").read_text())
        assert saved["arguments"] == (value if value == "{broken" else {})
        assert saved["observation_sequence"] is None
    monkeypatch.setattr(sys, "argv", ["activate_skill.py", "[]"])
    from tools.activate_skill import activate
    skills.main("activate_skill", activate)
    result = json.loads(capsys.readouterr().out)
    assert tool_error(result) and (Path(config["workspace"]) / result["record"]).is_file()
    assert driver.ticks == before
    state = json.loads((Path(config["state_dir"]) / "state.json").read_text())
    assert state["counter"] == 3 and state["active_skill"] is None


@pytest.mark.parametrize("arm,residual", [("left", 2e-8), ("right", -3e-8)])
def test_baseline_j8_reset_handles_measured_residual_but_commands_only_zero(live, arm, residual):
    from tools.exec_plan_pose import _reset_and_lock_selected_j8
    driver, _, observed = live
    index = PROPRIO_SLICES[f"arm_{arm}_qpos"].stop - 1
    with driver.condition:
        observed["robot_r1::proprio"][index] = residual
        driver.adapter.current.observation["robot_r1::proprio"][index] = residual

    def actions(robot, arm):
        result = {}
        yield from _reset_and_lock_selected_j8(robot, arm, stage="exec_plan_pose.start", result_out=result)
        robot.set_result(result)

    result = run("j8_reset_regression", {"type": "object"}, actions, {"arm": arm})
    assert (not tool_error(result)), result
    assert result["joint_commands_sent"] > 0
    assert driver.world.tool_roll_pin_qpos(arm) == 0
    assert arm not in driver.world._tool_roll_motion_enabled
    rows = bulk_record(result, "joint_commands.jsonl").read_text().splitlines()
    for row in rows:
        action = json.loads(row)["action"]
        assert action[14] == action[24] == 0
        validate_action(action)
    assert np.isclose(driver.world.arm_qpos_list(arm)[7], 0)


def test_tool_catalog_and_joint_manual():
    catalog = json.loads((TEMPLATE / "tools/util/assets/tool_catalog.json").read_text())
    for item in catalog:
        path = TEMPLATE / "tools" / (item["name"] + ".py")
        tool = module(path)
        assert tool.NAME == item["name"]
        # Required inputs/enumerations must survive extraction.
        assert set(tool.SCHEMA.get("required", [])) == set(item["input_schema"].get("required", []))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("roboharness_workspace", "behavior_interface"))
            assert not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "p"), "No hidden planning facade"
    assert {p.parent.name for p in (TEMPLATE / "skills").glob("*/SKILL.md")} == {
        "behavior-v2-baseline", "pick-up-object", "place-object-in-container", "open-doors-and-drawers"}
    pictures = list((TEMPLATE / "robot_config/joints").glob("*.png"))
    assert len(pictures) == 24
    assert all(Image.open(path).size == (1320, 735) for path in pictures)
    before, _ = forward({})
    after, _ = forward({"right_arm_joint4": -math.pi / 2})
    assert not np.allclose(before["right_arm_link4"], after["right_arm_link4"])
    for index in (6,):
        command = [0.] * 27
        command[index] = .1
        with pytest.raises(ValueError):
            validate_action(command)
    for index in (14, 24):
        command = [0.] * 27
        command[index] = math.pi / 2
        assert np.isclose(validate_action(command)[index], math.pi / 2)
    # Sensor/controller values at a valid static limit survive float32 wire
    # conversion. The next representable value outside must still be rejected.
    command = [0.] * 27
    for index in (12, 22):
        low = np.float32(joints()[ACTION_NAMES[index]]["limits"]["lower"])
        command[index] = float(low)
        validate_action(command)
        command[index] = float(np.nextafter(low, np.float32(-np.inf)))
        with pytest.raises(ValueError):
            validate_action(command)
        command[index] = 0.


def test_action_algorithms_match_pinned_task01_source():
    """Pin extracted actions and explicitly recorded, reviewed bug fixes."""
    fixture = json.loads((Path(__file__).parent / "fixtures/baseline_action_hashes.json").read_text())
    functions = fixture["functions"]
    for name, update in fixture.get("reviewed_updates", {}).items():
        assert update["baseline_sha256"] == functions[name]
        assert update["reason"] and update["date"]
        functions[name] = update["sha256"]

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id="robot", ctx=node.ctx), node) if node.id == "ctx" else node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "p":
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            if isinstance(node.value, ast.Name) and node.value.id in {"ctx", "robot"} and node.attr == "world":
                return ast.copy_location(ast.Name(id="robot", ctx=node.ctx), node)
            return self.generic_visit(node)

        def visit_arg(self, node):
            if node.arg == "ctx":
                node.arg = "robot"
            return self.generic_visit(node)

    for path in (TEMPLATE / "tools").glob("*.py"):
        if path.stem.startswith("capture_") or path.stem not in functions or path.stem == "robot_step":
            continue
        after = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "actions")
        normalized = ast.dump(Normalize().visit(after), include_attributes=False)
        assert sha256(normalized.encode()).hexdigest() == functions[path.stem], path.name


def test_native_hook_failure_and_hash_validation(live):
    _, config, _ = live
    from tools.util.observation_hook import process
    event = {"tool_name": "Bash", "hook_event_name": "PreToolUse", "session_id": "test-uuid", "tool_use_id": "test-failure"}
    process(event)
    capture = invoke("capture_head_camera", {})
    failure = process({**event, "hook_event_name": "PostToolUseFailure", "error": "shell failed after moving"})
    assert "updatedToolOutput" not in failure["hookSpecificOutput"]
    assert capture["images"][0]["path"] in failure["hookSpecificOutput"]["additionalContext"]
    Path(capture["images"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        process({**event, "hook_event_name": "PostToolUse", "tool_response": {"stdout": "abbreviated output"}})


def test_wrong_resolution_and_unknown_point_are_rejected(live):
    driver, config, _ = live
    capture = invoke("capture_head_camera", {})
    state = json.loads((Path(config["state_dir"]) / "state.json").read_text())
    # Deliberately corrupt only this test's latest view. Reject before physics.
    Image.new("RGB", (640, 480)).save(state["latest_head_path"])
    before = driver.ticks
    result = invoke("measure_shoulder_distance", {"image_id": capture["latest_clickable_image_id"], "u": 500, "v": 400})
    assert tool_error(result) and "720" in result["error"]
    assert driver.ticks == before


def test_measure_requires_pixel_contract_before_motion(live):
    driver, _, _ = live
    before = driver.ticks
    result = invoke('measure_shoulder_distance', {'object_name': 'hallstand'})
    assert tool_error(result) and 'ValidationError' in result['error']
    assert 'TypeError' not in result['error']
    assert result['joint_commands_sent'] == 0 and driver.ticks == before
    entry = next(x for x in json.loads((TEMPLATE / 'tools/util/assets/tool_catalog.json').read_text())
                 if x['name'] == 'measure_shoulder_distance')
    from tools.measure_shoulder_distance import SCHEMA
    assert entry['input_schema'] == SCHEMA


def test_unavailable_tools_send_no_commands_and_return_current_head(live):
    name = 'plan_press_point'
    driver, _, _ = live
    capture = invoke('capture_head_camera', {})
    before = driver.ticks
    arguments = {'image_id': capture['latest_clickable_image_id'], 'u': 475, 'v': 405}
    result = invoke(name, arguments)
    assert tool_error(result)
    assert result['joint_commands_sent'] == 0 and driver.ticks == before
    assert result['images'][0]['image_id'] == result['latest_clickable_image_id']
    assert 'wrist observation or plan preview' not in result['images'][0]['caption']
    assert 'Unavailable' in (TEMPLATE / 'tools' / (name + '.py')).read_text().split('"""')[1]


def test_budget_contains_no_goal_state():
    from tools.util.budget import sanitize_budget
    result = sanitize_budget({"available": True, "used_ticks": 30, "total_ticks": 100,
        "episode_id": "episode-1", "instance_id": 301, "source": "official_evaluator_episode_steps",
        "q_score": 1., "objects": ["secret"]})
    assert result["remaining_ticks"] == 70
    assert "q_score" not in result and "objects" not in result
