"""Independent RTAB-Map backend for official BEHAVIOR RGB-D observations.

This package intentionally has no dependency on ``spatial_map`` or
``pose_graph``.  The only public input types are defined in ``official``.
"""

from .client import RtabmapClient, WorkerRequestTimeout, WorkerUnavailable
from .official import (
    BodyOdometry,
    CameraIntrinsics,
    CameraRelativePose,
    OfficialObservation,
    SE2Pose,
)
from .protocol import MapResult, PoseRecord
from .render import render_heading_up
from .validation import ThreeTurnReport, ThreeTurnValidator, TurnMetrics

__all__ = [
    "BodyOdometry",
    "CAPTURE_SCHEMA",
    "CameraIntrinsics",
    "CameraRelativePose",
    "CaptureBundleWriter",
    "CaptureWriterError",
    "MapResult",
    "OfficialObservation",
    "PoseRecord",
    "RtabmapClient",
    "WorkerRequestTimeout",
    "SE2Pose",
    "ThreeTurnReport",
    "ThreeTurnValidator",
    "TurnMetrics",
    "WorkerUnavailable",
    "render_heading_up",
]


def __getattr__(name: str):
    """Load capture exports lazily so ``python -m ...offline`` stays acyclic."""

    if name in {"CAPTURE_SCHEMA", "CaptureBundleWriter", "CaptureWriterError"}:
        from .capture import CAPTURE_SCHEMA, CaptureBundleWriter, CaptureWriterError

        exports = {
            "CAPTURE_SCHEMA": CAPTURE_SCHEMA,
            "CaptureBundleWriter": CaptureBundleWriter,
            "CaptureWriterError": CaptureWriterError,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
