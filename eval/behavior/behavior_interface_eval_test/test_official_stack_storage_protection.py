from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OfficialStackStorageProtectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stack_script = (
            Path(__file__).resolve().parent / "start_official_test_stack_v391.sh"
        ).read_text(encoding="utf-8")
        cls.stack_prefix = cls.stack_script.split("\nread_live_pid() {", 1)[0]

    def _run_prefix(
        self,
        body: str,
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH",
            "BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE",
            "BEHAVIOR_EVAL_TEST_GRASP_AUDIT_DIR",
            "BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE",
        ):
            env.pop(key, None)
        # Keep prefix tests deterministic even when the developer shell has a
        # CUDA mask left over from another stack.
        env["BEHAVIOR_EVAL_TEST_PORT"] = "15060"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        if environment:
            env.update(environment)

        with tempfile.TemporaryDirectory() as temporary:
            package_dir = Path(temporary) / "behavior_interface_eval_test"
            package_dir.mkdir()
            harness = package_dir / "stack_harness.sh"
            harness.write_text(
                f"{self.stack_prefix}\n{body}\n",
                encoding="utf-8",
            )
            return subprocess.run(
                ["bash", str(harness)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_default_appdata_is_fixed_from_user_and_physical_gpu(self) -> None:
        # GPU 3 belongs to the 15063 official port.
        result = self._run_prefix(
            'printf "%s" "$EVALUATOR_APPDATA_PATH"',
            environment={
                "USER": "stack-test",
                "BEHAVIOR_EVAL_TEST_PORT": "15063",
                "CUDA_VISIBLE_DEVICES": "3",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "/var/tmp/og_appdata_stack-test_official_v391_gpu3",
        )

    def test_explicit_appdata_is_preserved(self) -> None:
        appdata = "/var/tmp/custom official appdata"
        result = self._run_prefix(
            'printf "%s" "$EVALUATOR_APPDATA_PATH"',
            environment={
                "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, appdata)

    def test_appdata_with_path_separator_is_rejected(self) -> None:
        result = self._run_prefix(
            ":",
            environment={
                "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": "/var/tmp/a:b",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot contain ':'", result.stderr)

    def test_rgbd_annotator_device_defaults_normalizes_and_validates(self) -> None:
        default = self._run_prefix('printf "%s" "$RGBD_ANNOTATOR_DEVICE"')
        explicit = self._run_prefix(
            'printf "%s" "$RGBD_ANNOTATOR_DEVICE"',
            environment={"BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE": "CUDA:0"},
        )
        invalid = self._run_prefix(
            ":",
            environment={"BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE": "gpu"},
        )
        invalid_ordinal = self._run_prefix(
            ":",
            environment={"BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE": "cuda:1"},
        )

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout, "cpu")
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(explicit.stdout, "cuda:0")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("must be cpu, cuda", invalid.stderr)
        self.assertEqual(invalid_ordinal.returncode, 2)
        self.assertIn("single-GPU managed stack", invalid_ordinal.stderr)

    def test_evaluator_reuse_requires_matching_rgbd_device(self) -> None:
        appdata = "/var/tmp/official-stack-rgbd-device-test"
        process_env = os.environ.copy()
        for key in (
            "BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE",
            "BEHAVIOR_EVAL_TEST_PRIVILEGED_GRASP_ORACLE_PATH",
            "BEHAVIOR_EVAL_TEST_GRASP_PREDICTION_TRACE_PATH",
        ):
            process_env.pop(key, None)
        process_env.update(
            {
                "OMNIGIBSON_APPDATA_PATH": appdata,
                "BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS": appdata,
                "BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE": "cuda:0",
                "CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "0",
                "OMNIGIBSON_GPU_ID": "0",
            }
        )
        sleeper = subprocess.Popen(["sleep", "30"], env=process_env)
        try:
            requested_cuda = {
                "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                "BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE": "cuda:0",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            matching = self._run_prefix(
                f'evaluator_has_requested_audit_env "{sleeper.pid}"',
                environment=requested_cuda,
            )
            mismatch = self._run_prefix(
                f'evaluator_has_requested_audit_env "{sleeper.pid}"',
                environment={
                    "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                },
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

        self.assertEqual(matching.returncode, 0, matching.stderr)
        self.assertNotEqual(mismatch.returncode, 0)

    def test_policy_reuse_requires_matching_storage_protection(self) -> None:
        appdata = "/var/tmp/official-stack-test-appdata"
        process_env = os.environ.copy()
        process_env.update(
            {
                "BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS": appdata,
                "CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "0",
                "OMNIGIBSON_GPU_ID": "0",
                "IK_FILTER_CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_RTABMAP_CUDA_DEVICE": "0",
                "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE": "0",
                "BEHAVIOR_SLAM_CUDA_DEVICE": "cuda:0",
                "BEHAVIOR_SLAM_GEOMETRY_DEVICE": "cuda:0",
                "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cuda:0",
            }
        )
        sleeper = subprocess.Popen(["sleep", "30"], env=process_env)
        try:
            matching = self._run_prefix(
                f'interface_has_requested_audit_env "{sleeper.pid}"',
                environment={
                    "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                    "CUDA_VISIBLE_DEVICES": "0",
                },
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

        self.assertEqual(matching.returncode, 0, matching.stderr)

        legacy_env = os.environ.copy()
        legacy_env.pop("BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS", None)
        sleeper = subprocess.Popen(["sleep", "30"], env=legacy_env)
        try:
            legacy = self._run_prefix(
                f'interface_has_requested_audit_env "{sleeper.pid}"',
                environment={
                    "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                },
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

        self.assertNotEqual(legacy.returncode, 0)

        mismatch_env = os.environ.copy()
        mismatch_env.update(
            {
                "BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS": (
                    "/var/tmp/a-different-appdata"
                ),
                "CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "0",
                "OMNIGIBSON_GPU_ID": "0",
                "IK_FILTER_CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_RTABMAP_CUDA_DEVICE": "0",
                "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE": "0",
                "BEHAVIOR_SLAM_CUDA_DEVICE": "cuda:0",
                "BEHAVIOR_SLAM_GEOMETRY_DEVICE": "cuda:0",
                "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cuda:0",
            }
        )
        sleeper = subprocess.Popen(["sleep", "30"], env=mismatch_env)
        try:
            mismatch = self._run_prefix(
                f'interface_has_requested_audit_env "{sleeper.pid}"',
                environment={
                    "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                },
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

        self.assertNotEqual(mismatch.returncode, 0)

        writer_env = os.environ.copy()
        writer_env.update(
            {
                "OMNIGIBSON_APPDATA_PATH": appdata,
                "BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS": appdata,
                "CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU": "0",
                "OMNIGIBSON_GPU_ID": "0",
                "IK_FILTER_CUDA_VISIBLE_DEVICES": "0",
                "BEHAVIOR_RTABMAP_CUDA_DEVICE": "0",
                "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE": "0",
                "BEHAVIOR_SLAM_CUDA_DEVICE": "cuda:0",
                "BEHAVIOR_SLAM_GEOMETRY_DEVICE": "cuda:0",
                "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cuda:0",
            }
        )
        sleeper = subprocess.Popen(["sleep", "30"], env=writer_env)
        try:
            writer = self._run_prefix(
                f'interface_has_requested_audit_env "{sleeper.pid}"',
                environment={
                    "BEHAVIOR_EVAL_TEST_OMNIGIBSON_APPDATA_PATH": appdata,
                },
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

        self.assertNotEqual(writer.returncode, 0)

    def test_port_gpu_contract_defaults_and_rejects_mismatch(self) -> None:
        default = self._run_prefix(
            'printf "%s" "$GPU"',
            environment={
                "BEHAVIOR_EVAL_TEST_PORT": "15061",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        mismatch = self._run_prefix(
            ":",
            environment={
                "BEHAVIOR_EVAL_TEST_PORT": "15061",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout, "1")
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("assigned to physical GPU 1", mismatch.stderr)

    def test_unmapped_port_requires_explicit_override(self) -> None:
        rejected = self._run_prefix(
            ":",
            environment={
                "BEHAVIOR_EVAL_TEST_PORT": "15066",
                "CUDA_VISIBLE_DEVICES": "5",
            },
        )
        accepted = self._run_prefix(
            'printf "%s" "$GPU"',
            environment={
                "BEHAVIOR_EVAL_TEST_PORT": "15066",
                "CUDA_VISIBLE_DEVICES": "5",
                "BEHAVIOR_EVAL_TEST_ALLOW_UNMAPPED_PORT": "1",
            },
        )

        self.assertEqual(rejected.returncode, 2)
        self.assertIn("unsupported official port 15066", rejected.stderr)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stdout, "5")


if __name__ == "__main__":
    unittest.main()
