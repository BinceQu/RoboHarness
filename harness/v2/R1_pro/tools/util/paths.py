"""Locations relative to this copy of the complete R1 Pro harness."""
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROBOT_CONFIG = WORKSPACE / "robot_config"
ASSETS = ROBOT_CONFIG / "assets"
PYTHON = WORKSPACE / ".venv/bin/python"
