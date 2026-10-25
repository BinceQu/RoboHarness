"""Evaluator-only relocation of the checked-out X5 robot assets."""
import os
from pathlib import Path

if os.environ.get('ROBODOJO_HARNESS_RUN_DIR'):
    from robodojo_eval_harness.selection import install
    install()

robot_path = os.environ.get('ROBOHARNESS_ROBOTS_PATH')
if robot_path:
    resolved = Path(robot_path).resolve(strict=True)
    from env import global_configs
    global_configs.ROBOTS_PATH = str(resolved)

config_path = os.environ.get('ROBOHARNESS_ENV_CONFIG_PATH')
if config_path:
    from env import global_configs
    global_configs.ENV_CONFIG_PATH = str(Path(config_path).resolve(strict=True))
