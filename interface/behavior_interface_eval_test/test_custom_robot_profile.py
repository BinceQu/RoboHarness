from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from behavior_interface_eval_test.robot_contract import load_robot_contract
from behavior_interface_eval_test.robot_profiles.r1pro_8dof_hf250.validate import (
    validate,
)
from behavior_interface_eval_test.robot_profiles.r1pro_8dof_hf250.install import (
    install,
    restore,
)


ROOT = Path(__file__).resolve().parents[1]


class CustomRobotProfileTest(unittest.TestCase):
    def test_static_profile_validation(self) -> None:
        report = validate()
        self.assertTrue(report["ok"])
        self.assertEqual(report["action_dim"], 27)
        self.assertEqual(report["proprio_dim"], 65)
        self.assertEqual(report["low_level_dof"], 30)
        self.assertEqual(report["base_mass_kg"], 250.0)
        self.assertTrue(report["base_inertia_mass_scaled"])
        self.assertEqual(report["arm_j1_j7_effort_multiplier"], 2.0)
        self.assertEqual(
            report["arm_max_effort_by_joint"],
            {
                "j1": 110.0,
                "j2": 110.0,
                "j3": 50.0,
                "j4": 50.0,
                "j5": 36.0,
                "j6": 36.0,
                "j7": 36.0,
                "j8": 20.0,
            },
        )
        self.assertEqual(report["model"], "r1pro")
        self.assertEqual(report["baked_base_joint_effort"], 10000.0)
        self.assertEqual(report["base_passive_lock_friction"], 1_000_000_000.0)
        self.assertEqual(report["base_locked_axes"], ["z", "rx", "ry"])
        self.assertEqual(
            report["base_velocity_drive_kd"],
            [30000.0, 30000.0, 1700.0],
        )
        self.assertEqual(report["runtime_base_motion_effort"], 10000.0)
        self.assertFalse(report["runtime_effort_mode_switching"])
        self.assertFalse(report["runtime_joint_state_reads"])
        self.assertFalse(report["runtime_joint_effort_mutation"])
        self.assertFalse(report["runtime_direct_state_mutation"])
        self.assertFalse(report["has_assisted_grasp_truth"])
        self.assertEqual(report["gripper_motor_type"], "effort")
        self.assertEqual(report["gripper_mode"], "independent")
        self.assertEqual(report["gripper_command_dim"], 2)
        self.assertEqual(
            report["gripper_command_input_limits"],
            [[-20.0, -20.0], [20.0, 20.0]],
        )
        self.assertEqual(
            report["gripper_command_output_limits"],
            [[-20.0, -20.0], [20.0, 20.0]],
        )
        self.assertEqual(report["gripper_limit_tolerance"], 0.0)
        self.assertFalse(report["gripper_inverted"])

    def test_install_never_mutates_shared_root(self) -> None:
        """Live Kit processes watch the shared root; install() must not touch it."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data_v391"
            assets = data_root / "omnigibson-robot-assets"
            models = assets / "models"
            stock = models / "r1pro"
            stock.mkdir(parents=True)
            stock_marker = stock / "stock.txt"
            stock_marker.write_text("stock", encoding="utf-8")
            (models / "tiago").mkdir()
            (assets / "metadata.json").write_text("{}", encoding="utf-8")
            (data_root / "og_dataset").mkdir()
            (data_root / "og_dataset" / "scene.usd").write_text("usd", encoding="utf-8")

            def snapshot() -> dict[str, tuple[int, float]]:
                return {
                    path.relative_to(data_root).as_posix(): (
                        path.stat().st_ino,
                        path.stat().st_mtime_ns,
                    )
                    for path in data_root.rglob("*")
                }

            before = snapshot()
            destination = install(data_root)
            self.assertEqual(snapshot(), before)

            # The overlay is a sibling root, named by content, holding a real
            # copy of the profile and symlinks for everything else.
            variant_root = destination.parents[2]
            self.assertEqual(variant_root.parent, data_root.parent)
            self.assertTrue(variant_root.name.startswith("data_v391__r1pro-"))
            self.assertEqual(destination, variant_root / "omnigibson-robot-assets" / "models" / "r1pro")
            self.assertFalse(destination.is_symlink())
            self.assertTrue((destination / "r1pro.yaml").is_file())
            self.assertFalse((destination / "stock.txt").exists())
            self.assertTrue(
                (destination / "curobo" / "r1pro_description_curobo_default.yaml").is_file()
            )
            self.assertEqual(
                os.readlink(variant_root / "og_dataset"), str(data_root / "og_dataset")
            )
            self.assertEqual(
                os.readlink(variant_root / "omnigibson-robot-assets" / "metadata.json"),
                str(assets / "metadata.json"),
            )
            self.assertEqual(
                os.readlink(destination.parent / "tiago"), str(models / "tiago")
            )
            self.assertTrue((variant_root / "og_dataset" / "scene.usd").is_file())

            # Reinstall is a no-op: same root, same inodes, nothing rewritten.
            installed_directory_inode = destination.stat().st_ino
            installed_asset_inode = (destination / "r1pro.yaml").stat().st_ino
            reinstalled = install(data_root)
            self.assertEqual(reinstalled, destination)
            self.assertEqual(reinstalled.stat().st_ino, installed_directory_inode)
            self.assertEqual(
                (reinstalled / "r1pro.yaml").stat().st_ino,
                installed_asset_inode,
            )
            self.assertEqual(snapshot(), before)

            # Stock is untouched, so restore points straight back at the root.
            restored = restore(data_root)
            self.assertEqual(restored, stock)
            self.assertEqual(stock_marker.read_text(encoding="utf-8"), "stock")

    def test_restore_on_legacy_rewritten_root_uses_backup_overlay(self) -> None:
        """Roots rewritten by the old in-place installer expose stock via overlay."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data_v391"
            assets = data_root / "omnigibson-robot-assets"
            models = assets / "models"
            (models / "r1pro").mkdir(parents=True)
            (models / "r1pro" / "r1pro.yaml").write_text("profile", encoding="utf-8")
            backup = assets / ".behavior_interface_eval_test_backups" / "r1pro"
            backup.mkdir(parents=True)
            (backup / "stock.txt").write_text("stock", encoding="utf-8")

            before = {p: p.stat().st_mtime_ns for p in data_root.rglob("*")}
            restored = restore(data_root)
            self.assertEqual({p: p.stat().st_mtime_ns for p in data_root.rglob("*")}, before)
            self.assertTrue(restored.parents[2].name.startswith("data_v391__r1pro-stock-"))
            self.assertEqual((restored / "stock.txt").read_text(encoding="utf-8"), "stock")
            self.assertFalse((restored / "r1pro.yaml").exists())
            self.assertEqual((models / "r1pro" / "r1pro.yaml").read_text(encoding="utf-8"), "profile")

    def test_install_cli_prints_variant_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data_v391"
            (data_root / "omnigibson-robot-assets" / "models" / "r1pro").mkdir(parents=True)
            output = subprocess.check_output(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "behavior_interface_eval_test"
                        / "robot_profiles"
                        / "r1pro_8dof_hf250"
                        / "install.py"
                    ),
                    "--data-root",
                    str(data_root),
                ],
                text=True,
            ).strip()
            self.assertEqual(Path(output), install(data_root).parents[2])
            self.assertTrue((Path(output) / "omnigibson-robot-assets" / "models" / "r1pro" / "r1pro.yaml").is_file())

    def test_contract_dimensions_and_slices(self) -> None:
        contract = load_robot_contract("r1pro_8dof_hf250")
        self.assertEqual(contract.arm_dof, 8)
        self.assertEqual(contract.action_dim, 27)
        self.assertEqual(contract.proprio_dim, 65)
        self.assertEqual(contract.base_footprint_radius_m, 0.42)
        self.assertEqual(
            contract.action_slices["arm_left"],
            slice(7, 15),
        )
        self.assertEqual(
            contract.proprio_slices["arm_right_qpos"],
            slice(30, 38),
        )
        self.assertEqual(contract.action_slices["gripper_left"], slice(15, 17))
        self.assertEqual(contract.action_slices["arm_right"], slice(17, 25))
        self.assertEqual(contract.action_slices["gripper_right"], slice(25, 27))
        self.assertNotIn("grasp_left", contract.proprio_slices)
        self.assertNotIn("grasp_right", contract.proprio_slices)

    def test_profile_selected_at_process_import_time(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["BEHAVIOR_EVAL_TEST_ROBOT_PROFILE"] = "r1pro_8dof_hf250"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "from behavior_interface_eval_test.official_action_world "
                    "import ACTION_DIM, ARM_DOF, PROPRIO_DIM; "
                    "print(json.dumps([ACTION_DIM, ARM_DOF, PROPRIO_DIM]))"
                ),
            ],
            cwd=ROOT,
            env=env,
            text=True,
        )
        self.assertEqual(json.loads(output), [27, 8, 65])


if __name__ == "__main__":
    unittest.main()
