"""Relocate evaluator robot paths without changing source assets or physics."""
import os
from pathlib import Path
import yaml


def prepare(root: Path, destination: Path) -> Path:
    source = root.resolve() / 'Assets/Robots'
    target = destination / 'Robots'
    x5 = target / 'x5'
    x5.mkdir(parents=True)
    for folder, output in [(source, target), (source/'x5', x5)]:
        for item in folder.iterdir():
            if (folder == source and item.name == 'x5') or item.name == 'curobo.yml':
                continue
            (output/item.name).symlink_to(os.path.relpath(item, output))
    config = yaml.safe_load((source/'x5/curobo.yml').read_text())
    config['robot_cfg']['kinematics'].update(
        asset_root_path=str(source/'x5'), urdf_path=str(source/'x5/X5A.urdf'))
    (x5/'curobo.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    return target.resolve()


def prepare_joint_config(root: Path, destination: Path) -> Path:
    """Use the evaluator's existing need_planner switch for joint actions.

    This only omits evaluator-side IK/CuRobo warmup. Workspace tools compute
    their own joint commands. All other configuration files are identical.
    """
    source = root.resolve()/'env_cfg'
    robot_target = destination/'robot'
    robot_target.mkdir(parents=True, exist_ok=False)
    for folder, output in [(source, destination), (source/'robot', robot_target)]:
        for item in folder.iterdir():
            if (folder == source and item.name == 'robot') or item.name == 'dual_x5.yml':
                continue
            (output/item.name).symlink_to(os.path.relpath(item, output))
    config = yaml.safe_load((source/'robot/dual_x5.yml').read_text())
    for robot in config['robots']:
        if robot['robot_name'] != 'x5' or robot.get('type', 'target') != 'target':
            raise ValueError('Joint-only configuration requires target X5 arms')
        robot['need_planner'] = False
    (robot_target/'dual_x5.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    return destination.resolve()
