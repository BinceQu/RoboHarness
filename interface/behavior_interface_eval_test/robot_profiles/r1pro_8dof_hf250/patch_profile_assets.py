#!/usr/bin/env python3
"""Apply the reproducible physical and kinematic edits for this robot profile."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PROFILE_DIR = Path(__file__).resolve().parent
PROFILE_NAME = "r1pro_8dof_hf250"
SOURCE_MODEL = "r1pro"
MODEL_DIR = PROFILE_DIR / "assets" / "models" / SOURCE_MODEL
USD_PATH = MODEL_DIR / "usd" / f"{PROFILE_NAME}.usda"
URDF_PATH = MODEL_DIR / "urdf" / f"{PROFILE_NAME}.urdf"
CUROBO_DIR = MODEL_DIR / "curobo"
TARGET_MASS_KG = 250.0
SOURCE_MASS_KG = 47.63
MASS_SCALE = TARGET_MASS_KG / SOURCE_MASS_KG
SOURCE_USD_DIAGONAL_INERTIA = (1.5789126, 1.4648, 2.8372872)
SOURCE_URDF_INERTIA = {
    "ixx": 1.5812,
    "ixy": -1.0625e-5,
    "ixz": 0.053601,
    "iyy": 1.4648,
    "iyz": 5.7054e-6,
    "izz": 2.835,
}

TORSO_EFFORT = 1000.0
ARM_EFFORT_BY_JOINT = {
    1: 110.0,
    2: 110.0,
    3: 50.0,
    4: 50.0,
    5: 36.0,
    6: 36.0,
    7: 36.0,
    8: 20.0,
}
GRIPPER_EFFORT = 100.0
BASE_CONTROLLED_EFFORT = 10000.0
BASE_LOCK_FRICTION = 1000000000.0

JOINT_EFFORTS = {
    **{f"torso_joint{i}": TORSO_EFFORT for i in range(1, 5)},
    **{
        f"{arm}_arm_joint{i}": effort
        for arm in ("left", "right")
        for i, effort in ARM_EFFORT_BY_JOINT.items()
    },
    **{
        f"{arm}_gripper_finger_joint{i}": GRIPPER_EFFORT
        for arm in ("left", "right")
        for i in range(1, 3)
    },
}
BASE_CONTROLLED_JOINTS = {
    "base_footprint_x_joint",
    "base_footprint_y_joint",
    "base_footprint_rz_joint",
}
BASE_LOCK_JOINTS = {
    "base_footprint_z_joint",
    "base_footprint_rx_joint",
    "base_footprint_ry_joint",
}


def _format_number(value: float) -> str:
    return f"{value:.9g}"


def _patch_usd_base_inertial(text: str) -> str:
    pattern = re.compile(r'(?ms)(^[ ]{4}def Xform "base_link" \(.*?^[ ]{4}\})')
    match = pattern.search(text)
    if match is None:
        raise RuntimeError("USD base_link block was not found")
    block = match.group(1)
    mass_pattern = re.compile(r"(?m)^([ ]+float physics:mass = ).*$")
    block, mass_count = mass_pattern.subn(
        rf"\g<1>{_format_number(TARGET_MASS_KG)}",
        block,
        count=1,
    )
    inertia_pattern = re.compile(r"(?m)^([ ]+float3 physics:diagonalInertia = ).*$")
    target_inertia = tuple(value * MASS_SCALE for value in SOURCE_USD_DIAGONAL_INERTIA)
    inertia_text = ", ".join(_format_number(value) for value in target_inertia)
    block, inertia_count = inertia_pattern.subn(
        rf"\g<1>({inertia_text})",
        block,
        count=1,
    )
    if mass_count != 1 or inertia_count != 1:
        raise RuntimeError("USD base_link mass or diagonal inertia was not found")
    return text[: match.start()] + block + text[match.end() :]


def _set_usd_drive_attribute(block: str, drive_kind: str, name: str, value: float) -> str:
    pattern = re.compile(
        rf"(?m)^([ ]+float drive:{drive_kind}:physics:{re.escape(name)} = ).*$"
    )
    replacement = rf"\g<1>{_format_number(value)}"
    if pattern.search(block):
        return pattern.sub(replacement, block, count=1)

    marker = f'            uniform token physics:axis = '
    if marker not in block:
        raise RuntimeError("USD joint block has no physics:axis attribute")
    return block.replace(
        marker,
        f"            float drive:{drive_kind}:physics:{name} = {_format_number(value)}\n{marker}",
        1,
    )


def _remove_usd_drive(block: str, drive_kind: str) -> str:
    schema = f"PhysicsDriveAPI:{drive_kind}"
    schema_pattern = re.compile(r'(?m)^([ ]+(?:prepend )?apiSchemas = \[)([^\]]*)(\])$')
    schema_match = schema_pattern.search(block)
    if schema_match is None:
        raise RuntimeError("USD joint block has no apiSchemas declaration")
    schemas = re.findall(r'"([^"]+)"', schema_match.group(2))
    schemas = [item for item in schemas if item != schema]
    replacement = (
        schema_match.group(1)
        + ", ".join(f'"{item}"' for item in schemas)
        + schema_match.group(3)
    )
    block = block[: schema_match.start()] + replacement + block[schema_match.end() :]
    block = re.sub(
        rf"(?m)^[ ]+[^\n]*drive:{re.escape(drive_kind)}:[^\n]*\n?",
        "",
        block,
    )
    return block


def _set_usd_joint_friction(block: str, friction: float) -> str:
    pattern = re.compile(r"(?m)^([ ]+float physxJoint:jointFriction = ).*$")
    replacement = rf"\g<1>{_format_number(friction)}"
    if pattern.search(block):
        return pattern.sub(replacement, block, count=1)

    marker = "            uniform token physics:axis = "
    if marker not in block:
        raise RuntimeError("USD joint block has no physics:axis attribute")
    return block.replace(
        marker,
        f"            float physxJoint:jointFriction = {_format_number(friction)}\n{marker}",
        1,
    )


def _patch_usd_joint(
    text: str,
    joint_name: str,
    effort: float | None = None,
    *,
    passive_lock_friction: float | None = None,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf'(?ms)(^[ ]{{8}}def Physics(?:Revolute|Prismatic)Joint "{re.escape(joint_name)}".*?^[ ]{{8}}\}})'
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"USD joint block not found: {joint_name}")
    block = match.group(1)
    drive_kind = "linear" if "PhysicsPrismaticJoint" in block else "angular"
    patched = block
    if effort is not None:
        patched = _set_usd_drive_attribute(patched, drive_kind, "maxForce", effort)
    if passive_lock_friction is not None:
        patched = _remove_usd_drive(patched, drive_kind)
        patched = _set_usd_joint_friction(patched, passive_lock_friction)
    return text[: match.start()] + patched + text[match.end() :], patched != block


def patch_usd() -> None:
    text = USD_PATH.read_text(encoding="utf-8")
    text = _patch_usd_base_inertial(text)
    for joint_name, effort in JOINT_EFFORTS.items():
        text, _ = _patch_usd_joint(text, joint_name, effort)
    for joint_name in BASE_CONTROLLED_JOINTS:
        text, _ = _patch_usd_joint(text, joint_name, BASE_CONTROLLED_EFFORT)
    for joint_name in BASE_LOCK_JOINTS:
        text, _ = _patch_usd_joint(
            text,
            joint_name,
            passive_lock_friction=BASE_LOCK_FRICTION,
        )
    USD_PATH.write_text(text, encoding="utf-8")


def patch_urdf() -> None:
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    root.set("name", PROFILE_NAME)
    base_mass = root.find("./link[@name='base_link']/inertial/mass")
    if base_mass is None:
        raise RuntimeError("base_link inertial mass was not found in URDF")
    base_mass.set("value", _format_number(TARGET_MASS_KG))
    base_inertia = root.find("./link[@name='base_link']/inertial/inertia")
    if base_inertia is None:
        raise RuntimeError("base_link inertia tensor was not found in URDF")
    for name, source_value in SOURCE_URDF_INERTIA.items():
        base_inertia.set(name, _format_number(source_value * MASS_SCALE))
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    for joint_name, effort in JOINT_EFFORTS.items():
        joint = by_name.get(joint_name)
        if joint is None:
            raise RuntimeError(f"URDF joint not found: {joint_name}")
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF joint has no limit: {joint_name}")
        limit.set("effort", _format_number(effort))
    ET.indent(tree, space="    ")
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)


def patch_curobo() -> None:
    for path in sorted(CUROBO_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        kinematics = payload["robot_cfg"]["kinematics"]
        cspace = kinematics["cspace"]
        names = list(cspace["joint_names"])
        for arm in ("left", "right"):
            joint7 = f"{arm}_arm_joint7"
            joint8 = f"{arm}_arm_joint8"
            if joint8 not in names:
                insert_at = names.index(joint7) + 1
                names.insert(insert_at, joint8)
                for key in ("cspace_distance_weight", "null_space_weight"):
                    values = list(cspace[key])
                    values.insert(insert_at, 1)
                    cspace[key] = values
        cspace["joint_names"] = names

        lock_joints = kinematics["lock_joints"]
        if path.name.endswith("_base.yaml"):
            lock_joints["left_arm_joint8"] = None
            lock_joints["right_arm_joint8"] = None
        else:
            lock_joints.pop("left_arm_joint8", None)
            lock_joints.pop("right_arm_joint8", None)

        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=120),
            encoding="utf-8",
        )


def main() -> None:
    patch_usd()
    patch_urdf()
    patch_curobo()
    print(f"Patched {MODEL_DIR}")


if __name__ == "__main__":
    main()
