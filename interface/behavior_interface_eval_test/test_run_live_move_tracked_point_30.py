from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from behavior_interface_eval_test import run_live_move_tracked_point_30 as benchmark


def _capabilities_payload():
    return {
        "ok": True,
        "mode": "official-strict",
        "tool_version": "official_v2",
        "allowed_observations": [
            "task_id",
            "need_new_action",
            "*::rgb",
            "*::depth_linear",
            "*::proprio",
            "*::cam_rel_poses",
        ],
        "tools": {
            "move_tracked_point": {
                "status": "conditional",
                "feasibility": "implemented_now",
                "semantics": "N-point rigid tracked motion",
                "constraints": "terminal live observations are authoritative",
                "observations": [
                    "*::rgb",
                    "*::depth_linear",
                    "*::proprio",
                    "*::cam_rel_poses",
                ],
            }
        },
    }


def _public_schema_payload():
    coordinate = {
        "oneOf": [
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {"type": "number"}},
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        ]
    }
    return {
        "ok": True,
        "tool_version": "official_v2",
        "tools": [
            {
                "name": "move_tracked_point",
                "endpoint": "/api/v2/move_tracked_point",
                "args": [
                    {
                        "name": "points",
                        "type": "array",
                        "widget": "tracked_target_points",
                        "min_points": 1,
                        "max_points": 6,
                        "items": {
                            "properties": {
                                "name": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": ["on_hand", "off_hand"],
                                },
                                "target_xyz_m": coordinate,
                            }
                        },
                    },
                    {"name": "relations", "type": "array"},
                    {
                        "name": "quick_constraint",
                        "type": "object",
                        "enum_types": sorted(
                            benchmark.REQUIRED_MOVE_TRACKED_QUICK_PRESETS
                        ),
                    },
                    {"name": "pos_tol", "type": "number"},
                    {"name": "ori_tol_deg", "type": "number"},
                    {"name": "max_steps", "type": "integer"},
                    {"name": "timeout_s", "type": "number"},
                ],
            }
        ],
    }


def _reload_receipt(build="arbitrary-future-build"):
    return {
        "ok": True,
        "generation": 41,
        "stack_digest": "stack-digest-41",
        "builds": {
            "move_tracked_point": build,
            "live_runner": "arbitrary-live-build",
        },
        "trajectory": {"schema": "trajectory-runtime", "version": 17},
        "registry": {"installed": True},
        "rollback": False,
    }


def _reload_status(
    build="arbitrary-future-build",
    generation=41,
    *,
    stack_digest="stack-digest-41",
):
    payload = _reload_receipt(build)
    payload["generation"] = generation
    payload["stack_digest"] = stack_digest
    return payload


class RuntimeBuildGateTest(unittest.TestCase):
    def test_batch_start_accepts_any_server_build_after_semantic_validation(self):
        responses = {
            benchmark.RUNTIME_RELOAD_PATH: _reload_receipt("build-2037-no-edit"),
            benchmark.RUNTIME_RELOAD_STATUS_PATH: _reload_status(
                "build-2037-no-edit"
            ),
            benchmark.RUNTIME_CAPABILITIES_PATH: _capabilities_payload(),
            benchmark.RUNTIME_PUBLIC_SCHEMA_PATH: _public_schema_payload(),
        }

        def request(_base_url, path, **_kwargs):
            return responses[path]

        with mock.patch.object(benchmark, "_http_json", side_effect=request) as http:
            identity, _capabilities = benchmark._load_batch_runtime_identity(
                "http://runtime"
            )

        self.assertEqual(identity["move_tracked_point_build"], "build-2037-no-edit")
        self.assertEqual(identity["generation"], 41)
        self.assertEqual(identity["stack_digest"], "stack-digest-41")
        self.assertEqual(identity["trajectory_schema_version"], 17)
        self.assertEqual(len(identity["semantic_fingerprint"]), 64)
        reload_calls = [
            call
            for call in http.call_args_list
            if call.args[1] == benchmark.RUNTIME_RELOAD_PATH
        ]
        self.assertEqual(len(reload_calls), 1)
        self.assertEqual(reload_calls[0].kwargs["payload"], {})

    def test_semantic_gate_rejects_missing_role_and_symbolic_coordinate_support(self):
        schema = _public_schema_payload()
        points = schema["tools"][0]["args"][0]
        points["items"]["properties"]["role"]["enum"] = ["on_hand"]
        points["items"]["properties"]["target_xyz_m"] = {
            "type": "array",
            "items": {"type": "number"},
        }

        report = benchmark._runtime_semantic_snapshot(
            _capabilities_payload(), schema
        )

        self.assertFalse(report["ok"])
        joined = "\n".join(report["errors"])
        self.assertIn("on_hand and off_hand", joined)
        self.assertIn("numeric, symbolic", joined)

    def test_read_only_consistency_rejects_mid_batch_generation_change(self):
        capabilities = _capabilities_payload()
        schema = _public_schema_payload()
        semantic = benchmark._runtime_semantic_snapshot(capabilities, schema)
        expected = {
            "generation": 41,
            "stack_digest": "stack-digest-41",
            "move_tracked_point_build": "same-build",
            "live_runner_build": "arbitrary-live-build",
            "trajectory_schema": "trajectory-runtime",
            "trajectory_schema_version": 17,
            "semantic_fingerprint": semantic["fingerprint"],
        }
        responses = {
            benchmark.RUNTIME_RELOAD_STATUS_PATH: _reload_status(
                "same-build", generation=42
            ),
            benchmark.RUNTIME_CAPABILITIES_PATH: capabilities,
            benchmark.RUNTIME_PUBLIC_SCHEMA_PATH: schema,
        }

        with mock.patch.object(
            benchmark,
            "_http_json",
            side_effect=lambda _base_url, path, **_kwargs: responses[path],
        ):
            report = benchmark._runtime_batch_consistency(
                "http://runtime", expected
            )

        self.assertFalse(report["ok"])
        self.assertIn("reload generation changed", "\n".join(report["errors"]))

    def test_read_only_consistency_rejects_schema_change_without_build_change(self):
        capabilities = _capabilities_payload()
        original_schema = _public_schema_payload()
        expected_semantic = benchmark._runtime_semantic_snapshot(
            capabilities, original_schema
        )
        changed_schema = _public_schema_payload()
        changed_schema["tools"][0]["args"][0]["coordinate_help"] = "new semantics"
        expected = {
            "generation": 41,
            "stack_digest": "stack-digest-41",
            "move_tracked_point_build": "same-build",
            "live_runner_build": "arbitrary-live-build",
            "trajectory_schema": "trajectory-runtime",
            "trajectory_schema_version": 17,
            "semantic_fingerprint": expected_semantic["fingerprint"],
        }
        responses = {
            benchmark.RUNTIME_RELOAD_STATUS_PATH: _reload_status("same-build"),
            benchmark.RUNTIME_CAPABILITIES_PATH: capabilities,
            benchmark.RUNTIME_PUBLIC_SCHEMA_PATH: changed_schema,
        }

        with mock.patch.object(
            benchmark,
            "_http_json",
            side_effect=lambda _base_url, path, **_kwargs: responses[path],
        ):
            report = benchmark._runtime_batch_consistency(
                "http://runtime", expected
            )

        self.assertFalse(report["ok"])
        self.assertIn("capability/schema changed", "\n".join(report["errors"]))

    def test_result_identity_rejects_build_or_generation_changed_during_case(self):
        result = {
            "build": "build-after-reload",
            "reload_generation": 42,
            "official_v2_stack_digest": "stack-after-reload",
            "trajectory_schema": "trajectory-runtime",
            "trajectory_schema_version": 17,
        }
        expected = {
            "generation": 41,
            "stack_digest": "stack-digest-41",
            "move_tracked_point_build": "build-before-reload",
            "live_runner_build": "arbitrary-live-build",
            "trajectory_schema": "trajectory-runtime",
            "trajectory_schema_version": 17,
        }

        report = benchmark._execution_evidence(
            result,
            requested_names=("head", "tail"),
            start_q=None,
            final_q=None,
            expected_runtime=expected,
        )

        joined = "\n".join(report["errors"])
        self.assertIn("build changed during benchmark", joined)
        self.assertIn("reload generation does not match", joined)
        self.assertIn("stack digest does not match", joined)

    def test_run_never_submits_a_case_after_pre_case_reload_change(self):
        semantic = benchmark._runtime_semantic_snapshot(
            _capabilities_payload(), _public_schema_payload()
        )
        identity = {
            **benchmark._reload_receipt_identity(_reload_receipt("batch-build")),
            "semantic_fingerprint": semantic["fingerprint"],
            "semantic_fingerprint_algorithm": semantic[
                "fingerprint_algorithm"
            ],
            "reload_receipt": _reload_receipt("batch-build"),
            "semantic_snapshot": semantic["snapshot"],
        }
        point = [[0.4, -0.2, 1.0]]
        entry = {
            "name": "head",
            "status": "observed",
            "observation_sequence": 8,
            "xyz_in_robot_base_coord_m": point[0],
        }
        snapshot = (
            {"raw": {"tracked_object_distances": {"head": entry}}},
            {"head": entry},
            point,
            {"attempts": 1},
        )
        mismatch = {
            "ok": False,
            "errors": ["official-v2 reload generation changed during benchmark"],
            "generation": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark._parser().parse_args(
                [
                    "--count",
                    "1",
                    "--minimum-required-case-count",
                    "1",
                    "--single-point-only",
                    "--output",
                    str(Path(directory) / "report.json"),
                ]
            )
            with mock.patch.object(
                benchmark,
                "_load_batch_runtime_identity",
                return_value=(identity, _capabilities_payload()),
            ), mock.patch.object(
                benchmark,
                "_http_json",
                return_value={"evaluator_connected": True},
            ), mock.patch.object(
                benchmark,
                "_wait_for_tracked_memory",
                return_value=snapshot,
            ), mock.patch.object(
                benchmark,
                "_runtime_batch_consistency",
                return_value=mismatch,
            ), mock.patch.object(benchmark, "_run_case") as submit:
                report = benchmark.run(args)

        self.assertFalse(report["ok"])
        self.assertEqual(report["cases"][0]["failure_stage"], "runtime_consistency")
        self.assertEqual(report["runtime_batch"]["move_tracked_point_build"], "batch-build")
        submit.assert_not_called()

    def test_legacy_bootstrap_retries_once_and_pins_status_receipt(self):
        legacy = {
            "ok": True,
            "tool": "live_move_tracked_point_test",
            "build": "legacy-live-handler",
        }
        status_receipt = _reload_receipt("bootstrapped-formal-build")
        calls = {"reload": 0}

        def request(_base_url, path, **_kwargs):
            if path == benchmark.RUNTIME_RELOAD_PATH:
                calls["reload"] += 1
                return dict(legacy)
            if path == benchmark.RUNTIME_RELOAD_STATUS_PATH:
                raise RuntimeError("HTTP 404 for dedicated status")
            if path == benchmark.RUNTIME_LIVE_STATUS_PATH:
                return {
                    "ok": True,
                    "state": "idle",
                    "reload_receipt": dict(status_receipt),
                }
            if path == benchmark.RUNTIME_CAPABILITIES_PATH:
                return _capabilities_payload()
            if path == benchmark.RUNTIME_PUBLIC_SCHEMA_PATH:
                return _public_schema_payload()
            raise AssertionError(path)

        with mock.patch.object(benchmark, "_http_json", side_effect=request):
            identity, _capabilities = benchmark._load_batch_runtime_identity(
                "http://runtime"
            )

        self.assertEqual(calls["reload"], 2)
        self.assertEqual(identity["reload_attempt_count"], 2)
        self.assertEqual(identity["reload_receipt_protocol"], "legacy_bootstrap_status")
        self.assertEqual(
            identity["reload_status_endpoint"],
            f"{benchmark.RUNTIME_LIVE_STATUS_PATH}.reload_receipt",
        )
        self.assertEqual(
            identity["move_tracked_point_build"], "bootstrapped-formal-build"
        )

    def test_read_only_consistency_rejects_same_build_with_changed_stack(self):
        capabilities = _capabilities_payload()
        schema = _public_schema_payload()
        semantic = benchmark._runtime_semantic_snapshot(capabilities, schema)
        expected = {
            **benchmark._reload_receipt_identity(_reload_receipt("same-build")),
            "semantic_fingerprint": semantic["fingerprint"],
        }
        responses = {
            benchmark.RUNTIME_RELOAD_STATUS_PATH: _reload_status(
                "same-build", stack_digest="different-stack"
            ),
            benchmark.RUNTIME_CAPABILITIES_PATH: capabilities,
            benchmark.RUNTIME_PUBLIC_SCHEMA_PATH: schema,
        }

        with mock.patch.object(
            benchmark,
            "_http_json",
            side_effect=lambda _base_url, path, **_kwargs: responses[path],
        ):
            report = benchmark._runtime_batch_consistency(
                "http://runtime", expected
            )

        self.assertFalse(report["ok"])
        self.assertIn("stack digest changed", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
