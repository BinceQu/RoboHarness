"""Diagnostic-only physical smoke, with the same Python tools as the agent."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

from tools.util.settings import settings
from tools.util.tool import run
from tools.activate_skill import activate
from tools.deactivate_skill import deactivate
from tools.util.skills import run as run_skill
from tools.util.robot import Robot
from tools.util.receipt import tool_error


def bulk_record(receipt, name):
    """Bulk per-call files live in <case>/trajectory/calls, outside the workspace."""
    record = Path(receipt["record"])
    return record.parents[2].parent / "trajectory/calls" / record.parent.name / name


def main():
    config = settings()
    workspace = Path(config["workspace"])
    checks = []
    direct_effort_checks = []

    def call(name, arguments, *, expect_empty_close=False):
        path = (workspace / "robot_config/control.py" if name == "joint_control"
                else workspace / "tools" / (name + ".py"))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result = run(name, module.SCHEMA, module.actions, arguments)
        checks.append({"tool": name, "arguments": arguments, "result": result})
        print(json.dumps({"tool": name, "error": tool_error(result)}), file=sys.stderr, flush=True)
        if expect_empty_close:
            evidence = result["response"]
            if ("error" in result
                    or evidence.get("failure_stage") != "empty_close"
                    or evidence.get("grasp_confirmed") is not False
                    or not evidence.get("actuation_detected")
                    or not np.allclose(evidence["gripper_qpos_after"], [0, 0], atol=.003)):
                raise RuntimeError(f"Unexpected empty-close feedback: {result}")
        elif tool_error(result):
            raise RuntimeError(f"{name}: {result.get('error') or result['response']}")
        return result

    try:
        call("capture_head_camera", {})
        call("capture_right_wrist_camera", {})
        before = Robot().hold_action().tolist()
        moved = before.copy()
        moved[22] = min(1.04, max(-1.04, before[22] + .04))
        moved[14], moved[24] = .10, -.12
        call("robot_step", {"action": moved, "steps": 15})
        measured = Robot().arm_qpos_list("right")[5]
        if abs(measured - before[22]) < .005:
            raise RuntimeError("Command did not produce measurable wrist motion")
        j8_changed = [Robot().arm_qpos_list(arm)[7] for arm in ("left", "right")]
        if not np.allclose(j8_changed, [.10, -.12], atol=.015):
            raise RuntimeError(f"Raw commands did not move both J8 joints: {j8_changed}")
        call("robot_step", {"action": before, "steps": 15})
        if abs(Robot().arm_qpos_list("right")[5] - before[22]) > .015:
            raise RuntimeError("Wrist did not return to its initial position")
        call("joint_control", {"joints": {"left_arm_joint8": -.08, "right_arm_joint8": .09}, "steps": 15})
        named_j8 = [Robot().arm_qpos_list(arm)[7] for arm in ("left", "right")]
        if not np.allclose(named_j8, [-.08, .09], atol=.015):
            raise RuntimeError(f"Named joint commands did not reach both J8 targets: {named_j8}")
        call("robot_step", {"action": before, "steps": 15})
        j8_restored = [Robot().arm_qpos_list(arm)[7] for arm in ("left", "right")]
        if not np.allclose(j8_restored, [0, 0], atol=.015):
            raise RuntimeError(f"J8 joints were not restored before baseline tools: {j8_restored}")
        capture = call("capture_head_camera", {})
        tracked = call("track_object_distance", {"image_id": capture["latest_clickable_image_id"],
            "points": [{"name": "diagnostic_a", "u": 300, "v": 500},
                       {"name": "diagnostic_b", "u": 400, "v": 500}]})
        tracking_names = json.loads(bulk_record(tracked, "output.json").read_text())["operation_result"]["tracked_names"]
        if set(tracking_names) != {"diagnostic_a", "diagnostic_b"}:
            raise RuntimeError(f"Unexpected tracked names: {tracking_names}")
        call("adjust_chassis", {"forward": 0, "translation": 0, "spin": 0})
        call("open_gripper", {"arm": "right"})
        call("close_gripper", {"arm": "right"}, expect_empty_close=True)
        # A direct torso/arm command must preserve an already latched close
        # through the next ordinary tool, not just through its own steps.
        carried_effort = Robot().gripper_pin_effort_list("right")
        if carried_effort is None or not all(value < 0 for value in carried_effort):
            raise RuntimeError("Empty-close diagnostic did not retain closing intent")
        call("joint_control", {"joints": {"right_arm_joint8": 0}, "steps": 3})

        def check_following_hold(label, expected):
            held = call("adjust_chassis", {"forward": 0, "translation": 0, "spin": 0})
            commands = [json.loads(row)["action"] for row in
                        bulk_record(held, "joint_commands.jsonl").read_text().splitlines()]
            if not commands or not all(np.allclose(command[25:27], expected, atol=1e-8) for command in commands):
                raise RuntimeError(f"{label}: later tool lost right finger force {expected}")
            direct_effort_checks.append({"case": label, "expected_right_effort": expected,
                                         "commands_checked": len(commands), "passed": True})

        check_following_hold("named_omitted_preserves_close", carried_effort)
        call("joint_control", {"right_finger_effort": -.02, "steps": 3})
        check_following_hold("named_explicit_small_force", [-.02, -.02])
        asymmetric = Robot().hold_action().tolist()
        asymmetric[25:27] = [2., 3.]
        call("robot_step", {"action": asymmetric, "steps": 3})
        check_following_hold("raw_asymmetric_open_force", [2., 3.])
        call("open_gripper", {"arm": "right"})
        if Robot().gripper_keepalive_status()["right"]["direct_effort"]:
            raise RuntimeError("Explicit open did not release the direct effort target")
        # Exercise the relocated IK worker and its unchanged controller on a
        # small reachable hand offset. This diagnostic is never a scored run.
        call("adjust_right_eef_pose_in_head_frame", {"x": .02, "max_steps": 180, "timeout_s": 120})
        call("robot_step", {"action": before, "steps": 30})
        for operation, handler in (("activate_skill", activate), ("deactivate_skill", deactivate)):
            value = run_skill(operation, handler, {"name": "pick-up-object"})
            if tool_error(value):
                raise RuntimeError(value)
        report = {"type": "smoke_result", "is_error": False, "checks": checks,
            "wrist_before": before[22], "wrist_changed": measured,
            "wrist_restored": Robot().arm_qpos_list("right")[5],
            "j8_raw_targets": [.10, -.12], "j8_raw_measured": j8_changed,
            "j8_named_targets": [-.08, .09], "j8_named_measured": named_j8,
            "j8_restored": j8_restored, "tracking_names": tracking_names,
            "direct_effort_followup_checks": direct_effort_checks}
    except Exception as exc:
        report = {"type": "smoke_result", "is_error": True, "error": str(exc), "checks": checks}
    print(json.dumps(report), flush=True)
    return 1 if report["is_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
