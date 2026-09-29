"""Read one metric depth value from a frozen capture artifact."""

from __future__ import annotations

import math
import os
import re
from typing import Any

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.skills import register_skill


_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ReadDepthInputError(ValueError):
    """The public request is malformed."""


class ReadDepthObservationError(ValueError):
    """The frozen capture cannot provide the requested depth value."""


def _validated_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ReadDepthInputError(f"{label} is required")
    if not _POLICY_ID_RE.fullmatch(text):
        raise ReadDepthInputError(f"{label} contains unsupported characters")
    return text


def _policy_artifact_path(session_id: str, artifact: Any) -> str:
    name = str(artifact or "").strip()
    if not name:
        raise ReadDepthObservationError("capture has no depth_linear artifact")
    image_dir = os.path.realpath(agent_runs.images_dir(session_id))
    candidate = os.path.realpath(
        name if os.path.isabs(name) else os.path.join(image_dir, name)
    )
    try:
        inside_image_dir = os.path.commonpath([image_dir, candidate]) == image_dir
    except ValueError:
        inside_image_dir = False
    if not inside_image_dir:
        raise ReadDepthObservationError(
            "depth_linear artifact is outside policy-owned capture storage"
        )
    return candidate


def _depth_plane(path: str) -> np.ndarray:
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReadDepthObservationError(
            f"cannot load frozen depth_linear artifact: {exc}"
        ) from exc
    if raw.ndim == 2:
        plane = raw
    elif raw.ndim == 3 and raw.shape[-1] == 1:
        plane = raw[..., 0]
    else:
        raise ReadDepthObservationError(
            f"depth_linear must be HxW or HxWx1, got shape {raw.shape}"
        )
    return np.asarray(plane)


def read_depth_from_capture(
    *,
    session_id: str,
    image_id: str,
    u: float,
    v: float,
) -> float:
    """Return the exact native-pixel depth from one frozen capture."""
    session = _validated_id(session_id, label="session_id")
    image = _validated_id(image_id, label="image_id")
    try:
        u_value = float(u)
        v_value = float(v)
    except (TypeError, ValueError) as exc:
        raise ReadDepthInputError("u/v must be numeric") from exc
    if not (math.isfinite(u_value) and math.isfinite(v_value)):
        raise ReadDepthInputError("u/v must be finite")

    try:
        meta = agent_runs.load_image_meta(session, image)
    except (OSError, ValueError) as exc:
        raise ReadDepthObservationError(
            f"cannot load frozen capture {image!r}: {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise ReadDepthObservationError("capture metadata must be an object")

    modalities = meta.get("modalities") or {}
    if not isinstance(modalities, dict):
        raise ReadDepthObservationError("capture modalities metadata must be an object")
    depth_path = _policy_artifact_path(
        session,
        modalities.get("depth_linear") or modalities.get("depth"),
    )
    depth = _depth_plane(depth_path)
    height, width = depth.shape

    camera = meta.get("camera") or {}
    if not isinstance(camera, dict):
        raise ReadDepthObservationError("capture camera metadata must be an object")
    try:
        capture_width = int(camera["image_width"])
        capture_height = int(camera["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReadDepthObservationError(
            "capture is missing image_width/image_height metadata"
        ) from exc
    if (capture_height, capture_width) != (height, width):
        raise ReadDepthObservationError(
            "RGB/depth resolution mismatch: "
            f"capture={capture_width}x{capture_height}, depth={width}x{height}"
        )

    px = int(round(u_value))
    py = int(round(v_value))
    if not (0 <= px < width and 0 <= py < height):
        raise ReadDepthInputError(
            f"pixel ({px}, {py}) is outside depth image {width}x{height}"
        )
    depth_m = float(depth[py, px])
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ReadDepthObservationError(
            f"depth_linear is invalid at pixel ({px}, {py})"
        )
    return depth_m


@register_skill(
    "read_depth",
    description=(
        "Read the exact depth_linear value at one Qwen3-VL 0..1000 UV point "
        "from a frozen capture image_id and return depth_m in meters."
    ),
)
def read_depth(
    ctx,
    session_id: str,
    image_id: str,
    u: float,
    v: float,
):
    """Read one frozen capture pixel without querying or mutating the simulator."""
    try:
        depth_m = read_depth_from_capture(
            session_id=session_id,
            image_id=image_id,
            u=u,
            v=v,
        )
    except (ReadDepthInputError, ReadDepthObservationError) as exc:
        ctx.set_result(
            {
                "ok": False,
                "error": str(exc),
                "failure_stage": (
                    "input validation"
                    if isinstance(exc, ReadDepthInputError)
                    else "observation validation"
                ),
                "depth_m": None,
                "unit": "m",
                "image_id": str(image_id or ""),
                "execution": "hold_action_only",
                "observation_contract": ["frozen capture depth_linear"],
                "direct_simulator_mutation": False,
            }
        )
        yield ctx.world.hold_action()
        return

    ctx.set_result(
        {
            "ok": True,
            "error": None,
            "depth_m": depth_m,
            "unit": "m",
            "image_id": str(image_id),
            "source": "policy_owned_frozen_depth_linear",
            "execution": "hold_action_only",
            "observation_contract": ["frozen capture depth_linear"],
            "direct_simulator_mutation": False,
        }
    )
    yield ctx.world.hold_action()


__all__ = [
    "ReadDepthInputError",
    "ReadDepthObservationError",
    "read_depth",
    "read_depth_from_capture",
]
