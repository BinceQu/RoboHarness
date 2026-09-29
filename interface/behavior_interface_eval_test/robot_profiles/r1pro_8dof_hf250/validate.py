#!/usr/bin/env python3
"""Static validation for the official v3.9.1 custom robot package."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PROFILE_DIR = Path(__file__).resolve().parent
PROFILE_NAME = "r1pro_8dof_hf250"
MODEL = "r1pro"
MODEL_DIR = PROFILE_DIR / "assets" / "models" / MODEL
DEFINITION = MODEL_DIR / f"{MODEL}.yaml"
EVALUATOR_CONFIG = PROFILE_DIR / "evaluator_robot.yaml"
CONTRACT = PROFILE_DIR / "robot_contract.json"
USD = MODEL_DIR / "usd" / f"{PROFILE_NAME}.usda"
URDF = MODEL_DIR / "urdf" / f"{PROFILE_NAME}.urdf"
TARGET_MASS_KG = 250.0
MASS_SCALE = TARGET_MASS_KG / 47.63
EXPECTED_USD_DIAGONAL_INERTIA = tuple(
    value * MASS_SCALE for value in (1.5789126, 1.4648, 2.8372872)
)
EXPECTED_URDF_INERTIA = {
    name: value * MASS_SCALE
    for name, value in {
        "ixx": 1.5812,
        "ixy": -1.0625e-5,
        "ixz": 0.053601,
        "iyy": 1.4648,
        "iyz": 5.7054e-6,
        "izz": 2.835,
    }.items()
}
EXPECTED_JOINT_EFFORTS = {
    **{f"torso_joint{i}": 1000.0 for i in range(1, 5)},
    **{
        f"{arm}_arm_joint{i}": effort
        for arm in ("left", "right")
        for i, effort in {
            1: 110.0,
            2: 110.0,
            3: 50.0,
            4: 50.0,
            5: 36.0,
            6: 36.0,
            7: 36.0,
            8: 20.0,
        }.items()
    },
    **{
        f"{arm}_gripper_finger_joint{i}": 100.0
        for arm in ("left", "right")
        for i in range(1, 3)
    },
}
BASE_JOINTS = {
    f"base_footprint_{component}_joint"
    for component in ("x", "y", "z", "rx", "ry", "rz")
}
BASE_CONTROLLED_JOINTS = {
    "base_footprint_x_joint",
    "base_footprint_y_joint",
    "base_footprint_rz_joint",
}
BASE_UNUSED_JOINTS = {
    "base_footprint_z_joint",
    "base_footprint_rx_joint",
    "base_footprint_ry_joint",
}
BASE_CONTROLLED_EFFORT = 10000.0
BASE_LOCK_FRICTION = 1000000000.0
BASE_PHYSICS_FREQUENCY_HZ = 120.0
BASE_LINEAR_COMMAND_LIMIT_MPS = 0.75
BASE_VELOCITY_DRIVE_KD = [30000.0, 30000.0, 1700.0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_close(actual: float, expected: float, message: str) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6),
        f"{message}: expected {expected}, got {actual}",
    )


def _usd_joint_block(usd_text: str, joint_name: str) -> str:
    match = re.search(
        rf'(?ms)^[ ]{{8}}def Physics(?:Revolute|Prismatic)Joint "{re.escape(joint_name)}".*?^[ ]{{8}}\}}',
        usd_text,
    )
    if match is None:
        raise AssertionError(f"USD joint block is missing: {joint_name}")
    return match.group(0)


def _usd_drive_value(block: str, attribute: str) -> float:
    match = re.search(
        rf"(?m)^\s+float drive:(?:angular|linear):physics:{re.escape(attribute)} = ([^\s]+)$",
        block,
    )
    if match is None:
        raise AssertionError(f"USD drive attribute is missing: {attribute}")
    return float(match.group(1))


def _usd_joint_friction(block: str) -> float:
    match = re.search(
        r"(?m)^\s+float physxJoint:jointFriction = ([^\s]+)$",
        block,
    )
    if match is None:
        raise AssertionError("USD joint friction is missing")
    return float(match.group(1))


def validate(profile_dir: Path = PROFILE_DIR) -> dict:
    del profile_dir
    definition = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    evaluator = yaml.safe_load(EVALUATOR_CONFIG.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    usd_text = USD.read_text(encoding="utf-8")
    urdf_root = ET.parse(URDF).getroot()

    _require(evaluator["model"] == MODEL, "evaluator model is not canonical")
    _require(evaluator["name"] == "robot_r1", "evaluator robot name changed")
    _require("type" not in evaluator, "official evaluator rejects robot type")
    _require(
        definition["usd_path"] == f"models/{MODEL}/usd/{PROFILE_NAME}.usda",
        "RobotDefinition USD path is not self-contained under the R1Pro variant",
    )
    _require(
        definition["urdf_path"] == f"models/{MODEL}/urdf/{PROFILE_NAME}.urdf",
        "RobotDefinition URDF path is not self-contained under the R1Pro variant",
    )
    _require(len(evaluator["reset_joint_pos"]) == 30, "reset_joint_pos must have 30 values")
    _require(contract["action_dim"] == 27, "custom action dimension must be 27")
    _require(contract["proprio_dim"] == 65, "custom proprio dimension must be 65")
    _require(
        "grasp_left" not in evaluator["proprio_obs"]
        and "grasp_right" not in evaluator["proprio_obs"],
        "non-standard assisted-grasp truth must not be requested as proprioception",
    )
    _require(contract["arm_dof"] == 8, "custom arm dimension must be 8")
    base_output_limits = evaluator["controller_config"]["base"][
        "command_output_limits"
    ]
    base_velocity_drive_kd = evaluator["controller_config"]["base"].get(
        "isaac_kd"
    )
    _require(
        isinstance(base_velocity_drive_kd, list)
        and len(base_velocity_drive_kd) == 3,
        "base velocity drive damping must be specified per controlled axis",
    )
    for index, expected in enumerate(BASE_VELOCITY_DRIVE_KD):
        _require_close(
            float(base_velocity_drive_kd[index]),
            expected,
            f"base velocity drive damping axis {index}",
        )
    for index in (0, 1):
        _require(
            float(base_velocity_drive_kd[index])
            <= TARGET_MASS_KG * BASE_PHYSICS_FREQUENCY_HZ,
            f"base linear damping axis {index} exceeds the monotone "
            "small-error bound",
        )
        _require(
            float(base_velocity_drive_kd[index])
            * BASE_LINEAR_COMMAND_LIMIT_MPS
            >= BASE_CONTROLLED_EFFORT,
            f"base linear damping axis {index} no longer reaches the "
            "existing max-force acceleration at a full command",
        )
    _require(
        float(base_velocity_drive_kd[2])
        <= EXPECTED_USD_DIAGONAL_INERTIA[2] * BASE_PHYSICS_FREQUENCY_HZ,
        "base yaw damping exceeds the monotone small-error bound",
    )
    _require(
        float(base_output_limits[0][1]) == -0.75
        and float(base_output_limits[1][1]) == 0.75,
        "base lateral velocity output must match the official R1Pro default",
    )
    trunk_output_limits = evaluator["controller_config"]["trunk"][
        "command_output_limits"
    ]
    _require(
        float(trunk_output_limits[0][3]) == 0.0
        and float(trunk_output_limits[1][3]) == 0.0,
        "torso_joint4 yaw output must be locked at zero",
    )
    _require(
        float(evaluator["reset_joint_pos"][9]) == 0.0,
        "torso_joint4 reset position must be zero",
    )

    for arm in ("left", "right"):
        names = definition["manipulation"]["arm_joint_names"][arm]
        _require(len(names) == 8, f"{arm} arm definition does not contain 8 joints")
        _require(names[-1] == f"{arm}_arm_joint8", f"{arm} J8 is not in the arm controller")
        gripper = evaluator["controller_config"][f"gripper_{arm}"]
        _require(
            gripper.get("name") == "MultiFingerGripperController"
            and gripper.get("motor_type") == "effort"
            and gripper.get("mode") == "independent",
            f"{arm} gripper must use the official independent effort controller",
        )
        _require(
            gripper.get("command_input_limits")
            == [[-20.0, -20.0], [20.0, 20.0]]
            and gripper.get("command_output_limits")
            == [[-20.0, -20.0], [20.0, 20.0]],
            f"{arm} gripper effort limits must be direct per-finger +/-20N",
        )
        _require_close(
            float(gripper.get("limit_tolerance")),
            0.0,
            f"{arm} gripper limit tolerance",
        )
        _require(
            gripper.get("inverted") is False,
            f"{arm} gripper effort direction must use negative-to-close",
        )

    base_block_match = re.search(
        r'(?ms)^[ ]{4}def Xform "base_link" \(.*?^[ ]{4}\}',
        usd_text,
    )
    _require(base_block_match is not None, "USD base_link block is missing")
    base_block = base_block_match.group(0)
    mass_match = re.search(r"(?m)^\s+float physics:mass = ([^\s]+)$", base_block)
    _require(mass_match is not None, "USD base mass is missing")
    _require_close(float(mass_match.group(1)), TARGET_MASS_KG, "USD base mass")
    diagonal_match = re.search(
        r"(?m)^\s+float3 physics:diagonalInertia = \(([^)]+)\)$",
        base_block,
    )
    _require(diagonal_match is not None, "USD base diagonal inertia is missing")
    usd_diagonal_inertia = tuple(
        float(value.strip()) for value in diagonal_match.group(1).split(",")
    )
    _require(len(usd_diagonal_inertia) == 3, "USD base diagonal inertia is malformed")
    for actual, expected in zip(usd_diagonal_inertia, EXPECTED_USD_DIAGONAL_INERTIA):
        _require_close(actual, expected, "USD base diagonal inertia")

    usd_joints = set(
        re.findall(r'def Physics(?:Revolute|Prismatic)Joint "([^"]+)"', usd_text)
    )
    urdf_joints = {joint.get("name") for joint in urdf_root.findall("joint")}
    required_usd = set(EXPECTED_JOINT_EFFORTS) | BASE_JOINTS
    required_urdf = set(EXPECTED_JOINT_EFFORTS)
    _require(
        required_usd <= usd_joints,
        f"USD joints missing: {sorted(required_usd - usd_joints)}",
    )
    _require(
        required_urdf <= urdf_joints,
        f"URDF joints missing: {sorted(required_urdf - urdf_joints)}",
    )

    for joint_name, expected_effort in {
        **EXPECTED_JOINT_EFFORTS,
        **{joint_name: BASE_CONTROLLED_EFFORT for joint_name in BASE_CONTROLLED_JOINTS},
    }.items():
        joint_block = _usd_joint_block(usd_text, joint_name)
        _require_close(
            _usd_drive_value(joint_block, "maxForce"),
            expected_effort,
            f"USD {joint_name} maxForce",
        )

    for joint_name in BASE_UNUSED_JOINTS:
        joint_block = _usd_joint_block(usd_text, joint_name)
        _require(
            "PhysicsDriveAPI:" not in joint_block
            and "drive:" not in joint_block,
            f"USD unused base joint must not be driven: {joint_name}",
        )
        _require_close(
            _usd_joint_friction(joint_block),
            BASE_LOCK_FRICTION,
            f"USD passive base lock friction {joint_name}",
        )

    base_mass = urdf_root.find("./link[@name='base_link']/inertial/mass")
    _require(base_mass is not None, "URDF base mass is missing")
    _require_close(float(base_mass.get("value")), TARGET_MASS_KG, "URDF base mass")
    base_inertia = urdf_root.find("./link[@name='base_link']/inertial/inertia")
    _require(base_inertia is not None, "URDF base inertia is missing")
    for name, expected in EXPECTED_URDF_INERTIA.items():
        _require_close(float(base_inertia.get(name)), expected, f"URDF base inertia {name}")

    urdf_joint_by_name = {
        joint.get("name"): joint for joint in urdf_root.findall("joint")
    }
    for joint_name, expected_effort in EXPECTED_JOINT_EFFORTS.items():
        limit = urdf_joint_by_name[joint_name].find("limit")
        _require(limit is not None, f"URDF joint limit is missing: {joint_name}")
        _require_close(
            float(limit.get("effort")),
            expected_effort,
            f"URDF {joint_name} effort",
        )

    for path in sorted((MODEL_DIR / "curobo").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        cspace = payload["robot_cfg"]["kinematics"]["cspace"]
        _require(len(cspace["joint_names"]) == 30, f"{path.name} cspace is not 30DOF")
        _require(
            len(cspace["cspace_distance_weight"]) == 30,
            f"{path.name} cspace weights do not match",
        )
        _require(
            len(cspace["null_space_weight"]) == 30,
            f"{path.name} null-space weights do not match",
        )

    return {
        "ok": True,
        "model": MODEL,
        "embodiment_variant": PROFILE_NAME,
        "action_dim": 27,
        "proprio_dim": 65,
        "low_level_dof": 30,
        "base_mass_kg": 250.0,
        "base_inertia_mass_scaled": True,
        "baked_base_joint_effort": BASE_CONTROLLED_EFFORT,
        "base_passive_lock_friction": BASE_LOCK_FRICTION,
        "base_locked_axes": ["z", "rx", "ry"],
        "arm_j1_j7_effort_multiplier": 2.0,
        "arm_max_effort_by_joint": {
            "j1": 110.0,
            "j2": 110.0,
            "j3": 50.0,
            "j4": 50.0,
            "j5": 36.0,
            "j6": 36.0,
            "j7": 36.0,
            "j8": 20.0,
        },
        "base_lateral_velocity_range_mps": [-0.75, 0.75],
        "base_velocity_drive_kd": BASE_VELOCITY_DRIVE_KD,
        "torso_joint4_yaw_locked": True,
        "runtime_base_motion_effort": BASE_CONTROLLED_EFFORT,
        "runtime_effort_mode_switching": False,
        "runtime_joint_state_reads": False,
        "runtime_joint_effort_mutation": False,
        "runtime_direct_state_mutation": False,
        "gripper_motor_type": "effort",
        "gripper_mode": "independent",
        "gripper_command_dim": 2,
        "gripper_command_input_limits": [[-20.0, -20.0], [20.0, 20.0]],
        "gripper_command_output_limits": [[-20.0, -20.0], [20.0, 20.0]],
        "gripper_limit_tolerance": 0.0,
        "gripper_inverted": False,
        "gripper_effort_limit_n": 20.0,
        "assisted_grasp_truth_observation": False,
        "has_assisted_grasp_truth": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
