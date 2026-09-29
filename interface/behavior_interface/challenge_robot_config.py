"""Challenge robot-config loading shared by submission artifacts and the interface."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .robot_variant import (
    normalize_robot_dof,
    robot_config_path_for_dof,
    robot_dof_from_type,
)

DEFAULT_CHALLENGE_ROBOT_CONFIG = robot_config_path_for_dof(8)


def resolve_challenge_robot_config_path(
    path: Optional[str] = None,
    *,
    robot_dof: Optional[int] = None,
) -> Path:
    raw = (
        path
        or os.environ.get("BEHAVIOR_ROBOT_CONFIG")
        or str(robot_config_path_for_dof(robot_dof))
    )
    return Path(raw).expanduser().resolve()


def load_challenge_robot_config(
    path: Optional[str] = None,
    *,
    robot_dof: Optional[int] = None,
) -> Dict[str, Any]:
    config_path = resolve_challenge_robot_config_path(path, robot_dof=robot_dof)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"robot config must be a mapping: {config_path}")

    model = str(config.get("model", "")).strip().lower()
    name = str(config.get("name", "")).strip()
    if model != "r1pro":
        raise ValueError(f"expected model=r1pro in {config_path}, got {model!r}")
    if not name:
        raise ValueError(f"missing robot name in {config_path}")

    configured_type = str(config.get("robot_type", "")).strip()
    if not configured_type:
        raise ValueError(f"missing robot_type in {config_path}")
    configured_dof = normalize_robot_dof(config.get("arm_dof"))
    if robot_dof is not None and configured_dof != normalize_robot_dof(robot_dof):
        raise ValueError(
            f"robot config arm_dof={configured_dof} does not match requested "
            f"robot_dof={normalize_robot_dof(robot_dof)} in {config_path}"
        )
    if robot_dof_from_type(configured_type) != configured_dof:
        raise ValueError(
            f"robot_type={configured_type!r} does not match arm_dof={configured_dof} "
            f"in {config_path}"
        )

    if config.get("self_collisions") is not True:
        raise ValueError(
            f"challenge robot config must enable self_collisions in {config_path}"
        )
    overrides = (config.get("load_config") or {}).get("joint_effort_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError(
                f"load_config.joint_effort_overrides must be a non-empty mapping in {config_path}"
            )
        for joint_name, effort in overrides.items():
            value = float(effort)
            if not joint_name or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"invalid effort override {joint_name!r}={effort!r} in {config_path}"
                )

    return config


def build_interface_robot_config(
    *,
    path: Optional[str],
    robot_type: Optional[str],
    robot_dof: Optional[int] = None,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Convert the Challenge `model` config to this branch's legacy `type` config."""
    config = deepcopy(load_challenge_robot_config(path, robot_dof=robot_dof))
    model = str(config.pop("model")).strip().lower()
    configured_type = str(config.pop("robot_type")).strip()
    configured_dof = normalize_robot_dof(config.pop("arm_dof"))
    requested = str(robot_type or configured_type).strip()
    if model != "r1pro":
        raise ValueError(
            f"robot config model={model!r} does not match expected model='r1pro'"
        )
    if requested.lower() != configured_type.lower():
        raise ValueError(
            f"robot config robot_type={configured_type!r} does not match "
            f"requested robot={requested!r}"
        )
    if robot_dof is not None and configured_dof != normalize_robot_dof(robot_dof):
        raise ValueError(
            f"robot config arm_dof={configured_dof} does not match requested "
            f"robot_dof={normalize_robot_dof(robot_dof)}"
        )

    config.pop("eval", None)
    config["type"] = configured_type
    sensor_kwargs = (
        config.setdefault("sensor_config", {})
        .setdefault("VisionSensor", {})
        .setdefault("sensor_kwargs", {})
    )
    sensor_kwargs["image_width"] = int(image_width)
    sensor_kwargs["image_height"] = int(image_height)
    return config
