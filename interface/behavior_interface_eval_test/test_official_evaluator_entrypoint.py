from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from behavior_interface_eval_test.official_evaluator_entrypoint import (
    configure_simulation_app_defaults,
    evaluator_action_trace_path,
    evaluator_gpu_mapping,
    evaluator_extra_args,
    install_evaluator_action_trace,
    install_effort_gripper_no_op_compat,
    install_eager_assisted_grasp_quat2mat,
    install_prepared_usd_cache,
    install_split_gpu_mapping,
    kit_renderer_gpu_ordinal,
    tasking_thread_limit,
)


class OfficialEvaluatorEntrypointTest(unittest.TestCase):
    def test_assisted_grasp_quat2mat_uses_compiled_functions_eager_body(self) -> None:
        def eager(value):
            return value + 1

        def compiled(_value):
            raise AssertionError("compiled wrapper should not run")

        compiled._torchdynamo_orig_callable = eager
        transform_utils = SimpleNamespace(quat2mat=compiled)

        self.assertTrue(
            install_eager_assisted_grasp_quat2mat(transform_utils)
        )
        self.assertEqual(transform_utils.quat2mat(4), 5)
        self.assertIs(transform_utils.quat2mat, eager)
        self.assertFalse(
            install_eager_assisted_grasp_quat2mat(transform_utils)
        )

    def test_prepared_usd_is_reused_between_prebuild_and_load(self) -> None:
        class FakeUSDObject:
            calls = 0

            def _prepare_to_load(self):
                type(self).calls += 1
                return self.path

        with tempfile.TemporaryDirectory() as tmp:
            prepared = os.path.join(tmp, "prepared.usd")
            with open(prepared, "wb") as stream:
                stream.write(b"usd")
            obj = FakeUSDObject()
            obj.path = prepared
            self.assertTrue(install_prepared_usd_cache(FakeUSDObject))
            self.assertFalse(install_prepared_usd_cache(FakeUSDObject))
            self.assertEqual(obj._prepare_to_load(), prepared)
            self.assertEqual(obj._prepare_to_load(), prepared)
            self.assertEqual(FakeUSDObject.calls, 1)

    def test_effort_gripper_no_op_compat_is_scoped_and_idempotent(self) -> None:
        class Backend:
            @staticmethod
            def zeros(size):
                return np.zeros(size, dtype=np.float32)

        class FakeController:
            def __init__(self, motor_type, mode="independent"):
                self._motor_type = motor_type
                self._mode = mode
                self.command_dim = 2

            def compute_no_op_goal(self, controller_idx):
                return {"target": f"goal-{controller_idx}"}

            def _compute_no_op_command(self, controller_idx):
                return f"command-{controller_idx}"

        self.assertTrue(
            install_effort_gripper_no_op_compat(FakeController, Backend)
        )
        self.assertFalse(
            install_effort_gripper_no_op_compat(FakeController, Backend)
        )

        effort = FakeController("effort")
        np.testing.assert_array_equal(
            effort.compute_no_op_goal(3)["target"],
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            effort._compute_no_op_command(3),
            [0.0, 0.0],
        )
        self.assertEqual(
            FakeController("position").compute_no_op_goal(4),
            {"target": "goal-4"},
        )
        self.assertEqual(
            FakeController("velocity")._compute_no_op_command(5),
            "command-5",
        )
        self.assertEqual(
            FakeController("effort", mode="binary").compute_no_op_goal(6),
            {"target": "goal-6"},
        )

    def test_action_trace_is_opt_in(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(evaluator_action_trace_path())
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_EVAL_TEST_ACTION_TRACE_PATH": "trace.jsonl"},
            clear=True,
        ):
            self.assertEqual(
                evaluator_action_trace_path(),
                os.path.abspath("trace.jsonl"),
            )

    def test_action_trace_records_received_gripper_command_and_goal(self) -> None:
        class FakeControllerView:
            last_action = None

            @classmethod
            def get_command_dim(cls, group_key):
                return 1

            @classmethod
            def get_goal(cls, group_key, controller_idx):
                del controller_idx
                target = 0.05 if group_key == "left" else 0.0
                return {"target": np.asarray([target, target])}

            @classmethod
            def get_control(cls, group_key, controller_idx):
                del controller_idx
                target = 0.05 if group_key == "left" else 0.0
                return np.asarray([target, target])

        class FakeRobot:
            name = "robot_r1"
            controllers = {
                "gripper_left": ("left", 0),
                "gripper_right": ("right", 0),
            }
            gripper_control_idx = {
                "left": np.asarray([0, 1]),
                "right": np.asarray([2, 3]),
            }

            def apply_action(self, action):
                FakeControllerView.last_action = action
                return "applied"

            def get_joint_positions(self):
                return np.asarray([0.05, 0.05, 0.05, 0.047])

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "action.jsonl")
            self.assertTrue(
                install_evaluator_action_trace(
                    FakeRobot,
                    FakeControllerView,
                    trace_path,
                )
            )
            robot = FakeRobot()
            self.assertEqual(robot.apply_action(np.asarray([1.0, -1.0])), "applied")
            with open(trace_path, encoding="utf-8") as trace_file:
                record = json.loads(trace_file.readline())

        right = next(
            item
            for item in record["controllers"]
            if item["name"] == "gripper_right"
        )
        self.assertEqual(record["action"], [1.0, -1.0])
        self.assertEqual(right["action_start"], 1)
        self.assertEqual(right["action_stop"], 2)
        self.assertEqual(right["command"], [-1.0])
        self.assertEqual(right["goal_after_update"], {"target": [0.0, 0.0]})
        self.assertEqual(
            record["gripper_qpos_before_physics"]["right"],
            [0.05, 0.047],
        )

    def test_defaults_to_eight_carbonite_tasking_threads(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tasking_thread_limit(), 8)

    def test_accepts_explicit_bounded_limit(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_EVAL_TEST_TASKING_THREADS": "4"},
            clear=True,
        ):
            self.assertEqual(tasking_thread_limit(), 4)

    def test_splits_physical_renderer_and_visible_physics_gpu(self) -> None:
        class FakeSimulationApp:
            def __init__(self, launch_config=None):
                self.launch_config = launch_config

        with mock.patch.dict(
            os.environ,
            {
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "5",
                "OMNIGIBSON_GPU_ID": "0",
            },
            clear=True,
        ):
            self.assertEqual(evaluator_gpu_mapping(), (5, 0))
            install_split_gpu_mapping(FakeSimulationApp)

        app = FakeSimulationApp({"active_gpu": 0, "physics_gpu": 7})
        self.assertEqual(app.launch_config["active_gpu"], 10)
        self.assertEqual(app.launch_config["physics_gpu"], 0)

    def test_masked_routes_use_duplicate_icd_renderer_ordinal(self) -> None:
        class FakeSimulationApp:
            def __init__(self, launch_config=None):
                self.launch_config = launch_config

        with mock.patch.dict(
            os.environ,
            {
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "5",
                "CUDA_VISIBLE_DEVICES": "5",
                "OMNIGIBSON_GPU_ID": "0",
            },
            clear=True,
        ):
            install_split_gpu_mapping(FakeSimulationApp)

        app = FakeSimulationApp({"active_gpu": 7, "physics_gpu": 7})
        self.assertEqual(app.launch_config["active_gpu"], 10)
        self.assertEqual(app.launch_config["physics_gpu"], 0)

    def test_kit_renderer_gpu_stride_is_configurable(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_EVAL_TEST_KIT_GPU_STRIDE": "1"},
            clear=True,
        ):
            self.assertEqual(kit_renderer_gpu_ordinal(5), 5)

    def test_rejects_invalid_gpu_mapping(self) -> None:
        for value in ("-1", "gpu1"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {"BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": value},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        evaluator_gpu_mapping()

    def test_rejects_physical_gpu_that_disagrees_with_single_cuda_mask(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "2",
                "OMNIGIBSON_GPU_ID": "0",
                "CUDA_VISIBLE_DEVICES": "5",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                evaluator_gpu_mapping()

    def test_rejects_multi_gpu_cuda_mask(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "2",
                "OMNIGIBSON_GPU_ID": "0",
                "CUDA_VISIBLE_DEVICES": "2,3",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                evaluator_gpu_mapping()

    def test_rejects_invalid_limits(self) -> None:
        for value in ("0", "33", "many"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {"BEHAVIOR_EVAL_TEST_TASKING_THREADS": value},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        tasking_thread_limit()

    def test_defaults_to_synchronous_rendering_and_legacy_job_backend(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            args = evaluator_extra_args()
        self.assertIn("--/app/asyncRendering=false", args)
        self.assertIn("--/app/asyncRenderingLowLatency=false", args)
        self.assertIn("--/omni/replicator/asyncRendering=false", args)
        self.assertIn(
            "--/plugins/carb.tasking.plugin/stuckCheckSeconds=0",
            args,
        )
        self.assertIn(
            "--/plugins/carb.tasking.plugin/useOmniJob=false",
            args,
        )
        # These only cover extension/MDL reload. Native texture subscriptions
        # require the separately tested native_asset_watches workaround.
        self.assertIn("--/app/extensions/fsWatcherEnabled=false", args)
        self.assertIn("--/app/material/disableMdlReload=true", args)
        # SimulationApp synthesizes threadCount/maxThreadCount from
        # ``limit_cpu_threads``; the wrapper must not add a duplicate.
        self.assertFalse(
            any("carb.tasking.plugin/threadCount=" in arg for arg in args)
        )

    def test_forces_synchronous_asset_loading_before_kit_startup(self) -> None:
        class FakeSimulationApp:
            DEFAULT_LAUNCHER_CONFIG = {
                "limit_cpu_threads": 32,
                "sync_loads": False,
                "extra_args": ["--existing-arg=true"],
            }

        with mock.patch.dict(os.environ, {}, clear=True):
            configure_simulation_app_defaults(FakeSimulationApp, 8)

        config = FakeSimulationApp.DEFAULT_LAUNCHER_CONFIG
        self.assertEqual(config["limit_cpu_threads"], 8)
        self.assertIs(config["sync_loads"], True)
        self.assertEqual(config["extra_args"][0], "--existing-arg=true")
        self.assertIn("--/app/asyncRendering=false", config["extra_args"])
        self.assertEqual(config["limit_cpu_threads"], 8)

    def test_allows_omni_job_fallback_and_rejects_invalid_value(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_EVAL_TEST_USE_OMNI_JOB": "0"},
            clear=True,
        ):
            self.assertIn(
                "--/plugins/carb.tasking.plugin/useOmniJob=false",
                evaluator_extra_args(),
            )
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_EVAL_TEST_USE_OMNI_JOB": "maybe"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                evaluator_extra_args()

    def test_main_reserves_current_writer_before_importing_omnigibson(self) -> None:
        code = (
            "import json,os,sys,types\n"
            "fake_tmp=types.ModuleType('behavior_interface.runtime_tmp')\n"
            "fake_tmp.configure_process_runtime_tmp=lambda _port:'managed-tmp'\n"
            "sys.modules['behavior_interface.runtime_tmp']=fake_tmp\n"
            "fake_storage=types.ModuleType('behavior_interface.runtime_storage')\n"
            "def preflight(path,**kwargs):\n"
            " print(json.dumps({'path':path,'reserve_for_pid':kwargs.get('reserve_for_pid'),"
            "'pid':os.getpid(),'og_loaded':'omnigibson' in sys.modules}),flush=True)\n"
            " raise SystemExit(42)\n"
            "fake_storage.preflight_storage=preflight\n"
            "sys.modules['behavior_interface.runtime_storage']=fake_storage\n"
            "from behavior_interface_eval_test import official_evaluator_entrypoint as entry\n"
            "entry.main()\n"
        )
        env = os.environ.copy()
        env["OMNIGIBSON_APPDATA_PATH"] = "/var/tmp/fake-official-appdata"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 42, proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["path"], env["OMNIGIBSON_APPDATA_PATH"])
        self.assertEqual(payload["reserve_for_pid"], payload["pid"])
        self.assertFalse(payload["og_loaded"])


if __name__ == "__main__":
    unittest.main()
