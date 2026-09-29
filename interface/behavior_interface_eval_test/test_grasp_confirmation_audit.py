from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from behavior_interface_eval_test.grasp_confirmation_audit import (
    ORACLE_ENABLE_ENV,
    ORACLE_TRACE_ENV,
    PREDICTION_TRACE_ENV,
    analyze_grasp_confirmation,
    privileged_oracle_trace_path,
    read_jsonl,
    record_compliant_grasp_prediction,
)
from behavior_interface_eval_test.privileged_grasp_oracle import (
    PrivilegedGraspOracleTrace,
    privileged_arm_truth,
)


class _FakeAttribute:
    def __init__(self, value):
        self.value = value

    def IsValid(self):
        return True

    def Get(self):
        return self.value


class _FakeConstraint:
    def __init__(self, path: str, *, valid=True, enabled=True):
        self.path = path
        self.valid = valid
        self.enabled = enabled

    def IsValid(self):
        return self.valid

    def GetPath(self):
        return self.path

    def GetAttribute(self, name):
        self.last_attribute = name
        return _FakeAttribute(self.enabled)


class GraspConfirmationAuditTest(unittest.TestCase):
    def test_policy_import_path_contains_no_private_assisted_grasp_reader(self):
        root = Path(__file__).resolve().parent
        common_source = (root / "grasp_confirmation_audit.py").read_text(
            encoding="utf-8"
        )
        policy_tools_source = (
            root / "tool" / "official_v2" / "tools.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_ag_obj_in_hand", common_source)
        self.assertNotIn("_ag_obj_constraints", common_source)
        self.assertNotIn("privileged_grasp_oracle", policy_tools_source)

    def test_privileged_oracle_is_disabled_by_default_and_requires_path(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(privileged_oracle_trace_path())
        with mock.patch.dict(
            os.environ,
            {ORACLE_ENABLE_ENV: "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, ORACLE_TRACE_ENV):
                privileged_oracle_trace_path()
        with mock.patch.dict(
            os.environ,
            {ORACLE_ENABLE_ENV: "yes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                privileged_oracle_trace_path()

    def test_truth_requires_object_and_live_constraint(self):
        obj = SimpleNamespace(name="can", prim_path="/World/can")
        constraint = _FakeConstraint("/World/robot/eef/ag_constraint")
        robot = SimpleNamespace(
            _ag_obj_in_hand={"right": obj},
            _ag_obj_constraints={"right": constraint},
            _ag_obj_constraint_params={
                "right": {
                    "ag_joint_prim_path": "/World/robot/eef/ag_constraint"
                }
            },
            _ag_freeze_gripper={"right": True},
        )

        established = privileged_arm_truth(robot, "right")
        self.assertTrue(established["established"])
        self.assertEqual(established["phase"], "established")
        self.assertEqual(established["object_name"], "can")

        robot._ag_obj_constraints["right"] = None
        releasing = privileged_arm_truth(robot, "right")
        self.assertFalse(releasing["established"])
        self.assertEqual(
            releasing["phase"],
            "release_window_or_stale_object_reference",
        )

        robot._ag_obj_constraints["right"] = _FakeConstraint(
            "/World/robot/eef/ag_constraint",
            enabled=False,
        )
        disabled = privileged_arm_truth(robot, "right")
        self.assertTrue(disabled["established"])
        self.assertTrue(disabled["constraint_live"])
        self.assertFalse(disabled["constraint_enabled"])

        robot._ag_obj_constraints["right"] = _FakeConstraint(
            "/World/robot/eef/ag_constraint",
            valid=False,
        )
        invalid = privileged_arm_truth(robot, "right")
        self.assertFalse(invalid["established"])
        self.assertFalse(invalid["constraint_live"])

    def test_oracle_writes_private_state_only_to_separate_jsonl(self):
        obj = SimpleNamespace(name="box", prim_path="/World/box")
        robot = SimpleNamespace(
            name="robot_r1",
            arm_names=("left", "right"),
            _ag_obj_in_hand={"left": obj, "right": None},
            _ag_obj_constraints={
                "left": _FakeConstraint("/World/robot/left/ag_constraint"),
                "right": None,
            },
            _ag_obj_constraint_params={"left": {}, "right": {}},
            _ag_freeze_gripper={"left": True, "right": False},
        )
        observation = {"robot_r1::proprio": [0.0]}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "oracle.jsonl")
            trace = PrivilegedGraspOracleTrace(path)
            returned = trace.record(robot, event="step", action=[0.0, -0.1])
            trace.close()
            record = read_jsonl(path)[0]

        self.assertIs(returned["policy_visible"], False)
        self.assertTrue(record["arms"]["left"]["established"])
        self.assertFalse(record["arms"]["right"]["established"])
        self.assertEqual(observation, {"robot_r1::proprio": [0.0]})
        self.assertNotIn("arms", observation)

    def test_prediction_trace_contains_only_existing_close_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "predictions.jsonl")
            with mock.patch.dict(
                os.environ,
                {PREDICTION_TRACE_ENV: path},
                clear=True,
            ):
                self.assertIsNone(
                    record_compliant_grasp_prediction("open_gripper", {"ok": True})
                )
                record = record_compliant_grasp_prediction(
                    "close_gripper",
                    {
                        "ok": False,
                        "arm": "right",
                        "grasp_confirmed": False,
                        "failure_reason": "timeout",
                        "gripper_qpos_after": [0.038, 0.045],
                    },
                )
            written = read_jsonl(path)

        self.assertFalse(record["privileged_test_only"])
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["record_type"], "compliant_prediction")
        self.assertNotIn("constraint_prim_path", written[0])
        self.assertNotIn("object_name", written[0])

    def test_offline_alignment_reports_false_positive_and_false_negative(self):
        def oracle(time_ns, sequence, left, right):
            return {
                "record_type": "privileged_oracle_sample",
                "time_ns": time_ns,
                "sequence": sequence,
                "arms": {
                    "left": {"established": left, "object_name": "left_obj"},
                    "right": {"established": right, "object_name": "right_obj"},
                },
            }

        def prediction(time_ns, arm, value):
            return {
                "record_type": "compliant_prediction",
                "time_ns": time_ns,
                "arm": arm,
                "grasp_confirmed": value,
            }

        report = analyze_grasp_confirmation(
            [
                oracle(1_000_000_000, 1, False, False),
                oracle(2_000_000_000, 2, True, False),
                oracle(3_000_000_000, 3, False, True),
            ],
            [
                prediction(1_100_000_000, "left", False),
                prediction(2_100_000_000, "left", False),
                prediction(3_100_000_000, "left", True),
                prediction(3_100_000_000, "right", True),
            ],
            max_age_s=0.5,
        )

        self.assertEqual(
            report["confusion_matrix"],
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertEqual(
            [item["outcome"] for item in report["mismatches"]],
            ["fn", "fp"],
        )

    def test_stale_oracle_sample_is_not_mislabeled_as_ground_truth(self):
        report = analyze_grasp_confirmation(
            [
                {
                    "record_type": "privileged_oracle_sample",
                    "time_ns": 1,
                    "sequence": 1,
                    "arms": {"right": {"established": True}},
                }
            ],
            [
                {
                    "record_type": "compliant_prediction",
                    "time_ns": 10_000_000_001,
                    "arm": "right",
                    "grasp_confirmed": True,
                }
            ],
            max_age_s=1.0,
        )
        self.assertEqual(report["paired_predictions"], 0)
        self.assertEqual(report["unpaired"][0]["reason"], "stale_oracle_sample")


if __name__ == "__main__":
    unittest.main()
