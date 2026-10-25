"""Live named depth memory backed only by evaluator head RGB-D observations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import zlib
from collections import OrderedDict, deque
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

from .contract import (
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    TRACK_OBJECT_DISTANCE_MAX_POINTS,
    validate_track_object_distance_args,
)
from .dynamic_point_tracker import (
    OBSERVED,
    CameraIntrinsics,
    DynamicPointSnapshot,
    DynamicPointTracker,
    TrackerFrame,
    camera_to_policy_from_odometry,
    transform_from_pose,
)
from .eef_adjustment_local import local_robot_state
from .grasp_geometry_local import quat_to_mat_xyzw
from .grasp_kinematics_local import eef_pose
from .replay_frame_archive import ReplayFrameArchive, ReplayFramePayload


TRACKED_OBJECT_DISTANCES_MEMORY_KEY = "tracked_object_distances"
TRACKING_POLICY_MEMORY_KEY = "tracked_object_distance_tracking"
# Retain every observation through ten minutes at 30 Hz, including capture.
# These are capacity bounds, not a wall-clock TTL: lower observation rates
# allow longer thinking. High-entropy streams still fail closed at the byte
# bound. Keep the previous 640 MiB RAM budget; older payloads read from disk.
TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES = 30 * 600 + 1
TRACK_OBJECT_DISTANCE_REPLAY_MAX_BYTES = 24 * 1024 * 1024 * 1024
TRACK_OBJECT_DISTANCE_REPLAY_MEMORY_MAX_BYTES = 640 * 1024 * 1024
TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S = 600.0
TRACK_OBJECT_DISTANCE_MAX_CAPTURE_BINDINGS = 8
TRACK_OBJECT_DISTANCE_MAX_BINDING_SESSIONS = 8
TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK = (
    TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES * TRACK_OBJECT_DISTANCE_MAX_POINTS
)
TRACK_OBJECT_DISTANCE_MAX_BINDING_FAILURES = 64
TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL = 1
TRACK_OBJECT_DISTANCE_REPLAY_MAX_PENDING_FRAMES = 32
# One 720p RGB-D zlib job can take roughly 90-110 ms on the evaluator CPU.
# Four independent workers sustain the 30 Hz observation stream while keeping
# archival work bounded.  The environment override is intentionally capped so
# a malformed deployment setting cannot create an unbounded thread fan-out.
TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS_DEFAULT = 4
TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS_MAX = 8
TRACK_OBJECT_DISTANCE_MOTION_RETENTION_MAX_S = 660.0
TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD = (
    "official_rgbd_registered_npoint_on_hand_eef_gate_v7"
)
# A two-point rigid model is an observation-side estimator, not permission to
# manufacture an arbitrarily different point cloud.  Reject the fused sample
# if either point would need a large correction from current evaluator RGB-D.
TRACK_OBJECT_DISTANCE_RIGID_PAIR_MAX_POINT_CORRECTION_M = 0.035
TRACK_OBJECT_DISTANCE_RIGID_PAIR_MIN_DISTANCE_M = 0.005


def _replay_compression_initializer() -> None:
    """Keep archival compression below the evaluator callback priority."""

    # Linux applies nice values per thread.  If the platform does not expose
    # that facility, the normal executor behavior remains a valid fallback.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass


def _replay_compression_worker_count() -> int:
    raw = os.environ.get(
        "BEHAVIOR_OFFICIAL_REPLAY_COMPRESSION_WORKERS",
        str(TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS_DEFAULT),
    )
    try:
        requested = int(str(raw).strip())
    except (TypeError, ValueError):
        requested = TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS_DEFAULT
    return max(
        1,
        min(
            requested,
            TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS_MAX,
        ),
    )


TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS = (
    _replay_compression_worker_count()
)


def _get_or_upgrade_replay_executor() -> ThreadPoolExecutor:
    """Reuse a hot-reload executor while raising its lazy thread ceiling."""

    existing = globals().get("_REPLAY_COMPRESSION_EXECUTOR")
    if existing is not None and not bool(
        getattr(existing, "_shutdown", False)
    ):
        # ThreadPoolExecutor creates workers lazily. Increasing this private
        # ceiling is safe for queued work and preserves the executor identity
        # required by the transactional hot-reload bridge.
        try:
            existing._max_workers = max(  # type: ignore[attr-defined]
                int(getattr(existing, "_max_workers")),
                int(TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS),
            )
        except (AttributeError, TypeError, ValueError):
            pass
        return existing
    return ThreadPoolExecutor(
        max_workers=TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS,
        thread_name_prefix="official-replay-compress",
        initializer=_replay_compression_initializer,
    )


_REPLAY_COMPRESSION_EXECUTOR = _get_or_upgrade_replay_executor()


def _replay_executor_worker_count() -> int:
    try:
        return int(getattr(_REPLAY_COMPRESSION_EXECUTOR, "_max_workers"))
    except (AttributeError, TypeError, ValueError):
        return int(TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_WORKERS)

# Optional local diagnostic.  Keep it disabled in normal operation: this
# module runs on every evaluator observation, so even a small JSONL append per
# frame would add avoidable CPU and filesystem contention.
_CAPTURE_TRACE_ENABLED = (
    os.environ.get("BEHAVIOR_OFFICIAL_CAPTURE_TRACE", "0")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)


def _capture_trace(stage: str, **fields: Any) -> None:
    if not _CAPTURE_TRACE_ENABLED:
        return
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "thread": threading.get_native_id(),
        "module": "tracked_object_distance",
        "stage": str(stage),
        **fields,
    }
    try:
        with open(
            f"/tmp/official_capture_trace_{os.getpid()}.jsonl",
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass
_ALLOWED_EXACT_OBSERVATION_KEYS = frozenset({"task_id", "need_new_action"})
_ALLOWED_OBSERVATION_SUFFIXES = (
    "::rgb",
    "::depth_linear",
    "::proprio",
    "::cam_rel_poses",
)


def project_registered_rigid_pair(
    measured_points_robot_base_m: Any,
    *,
    reference_distance_m: float,
    max_point_correction_m: float = (
        TRACK_OBJECT_DISTANCE_RIGID_PAIR_MAX_POINT_CORRECTION_M
    ),
) -> dict[str, Any]:
    """Project a measured point pair onto its registration-time distance.

    The midpoint and measured pair direction are preserved, so this is the
    unique minimum-L2 correction of the two current RGB-D points.  It uses no
    scene identity or simulator state; the only extra premise is the caller's
    declaration that the two named points belong to one rigid object.
    """

    measured = np.asarray(measured_points_robot_base_m, dtype=np.float64)
    if measured.shape != (2, 3) or not np.all(np.isfinite(measured)):
        raise ValueError("measured rigid pair must be a finite 2x3 array")
    reference = float(reference_distance_m)
    maximum = float(max_point_correction_m)
    if not math.isfinite(reference) or (
        reference < TRACK_OBJECT_DISTANCE_RIGID_PAIR_MIN_DISTANCE_M
    ):
        raise ValueError("registration rigid-pair distance is invalid")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_point_correction_m must be positive and finite")

    delta = measured[1] - measured[0]
    measured_distance = float(np.linalg.norm(delta))
    if (
        not math.isfinite(measured_distance)
        or measured_distance < TRACK_OBJECT_DISTANCE_RIGID_PAIR_MIN_DISTANCE_M
    ):
        return {
            "ok": False,
            "reason": "measured_pair_collapsed",
            "reference_distance_m": reference,
            "measured_distance_m": measured_distance,
            "distance_error_m": abs(measured_distance - reference),
            "max_point_correction_m": float("inf"),
            "correction_limit_m": maximum,
            "fused_points_robot_base_m": None,
        }

    direction = delta / measured_distance
    midpoint = np.mean(measured, axis=0)
    half = 0.5 * reference * direction
    fused = np.asarray([midpoint - half, midpoint + half], dtype=np.float64)
    corrections = np.linalg.norm(fused - measured, axis=1)
    max_correction = float(np.max(corrections))
    ok = bool(max_correction <= maximum + 1e-12)
    return {
        "ok": ok,
        "reason": None if ok else "rigid_pair_rgbd_correction_exceeded",
        "reference_distance_m": reference,
        "measured_distance_m": measured_distance,
        "fused_distance_m": float(np.linalg.norm(fused[1] - fused[0])),
        "distance_error_m": abs(measured_distance - reference),
        "point_correction_m": corrections.astype(float).tolist(),
        "max_point_correction_m": max_correction,
        "correction_limit_m": maximum,
        "midpoint_preserved": True,
        "measured_direction_preserved": True,
        "projection": "minimum_l2_registered_rigid_pair_distance",
        "fused_points_robot_base_m": fused.astype(float).tolist() if ok else None,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def head_camera_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """Return the static evaluator head-camera calibration at any resolution."""

    image_width = int(width)
    image_height = int(height)
    fx = 306.0 * float(image_width) / 720.0
    return CameraIntrinsics(
        width=image_width,
        height=image_height,
        fx=fx,
        fy=fx,
        cx=float(image_width) * 0.5,
        cy=float(image_height) * 0.5,
    )


def _is_head_key(key: str) -> bool:
    lowered = str(key).lower()
    return "zed_link" in lowered or "head" in lowered


def tracker_frame_from_allowed_observation(
    observation: Mapping[str, Any],
    *,
    sequence: int,
    timestamp_s: float,
    episode_id: str,
    base_xy_yaw: Any,
    copy_arrays: bool = True,
) -> TrackerFrame:
    """Assemble one synchronized tracker frame from the official allowlist.

    ``copy_arrays=False`` is reserved for an internally owned immutable
    observation snapshot that is handed directly to ``ingest()``, which takes
    the tracker's single ownership copy.  The public/default behavior remains
    defensive so standalone callers cannot mutate a returned frame indirectly.
    """

    rejected = sorted(
        str(key)
        for key in observation
        if str(key) not in _ALLOWED_EXACT_OBSERVATION_KEYS
        and not str(key).endswith(_ALLOWED_OBSERVATION_SUFFIXES)
    )
    if rejected:
        raise ValueError(
            "observation contains non-allowlisted keys: " + ", ".join(rejected)
        )

    rgb = None
    depth = None
    camera_poses = None
    proprio = None
    for key, value in observation.items():
        name = str(key)
        if name.endswith("::rgb") and _is_head_key(name):
            rgb = _as_numpy(value)
        elif name.endswith("::depth_linear") and _is_head_key(name):
            depth = _as_numpy(value)
        elif name.endswith("::cam_rel_poses"):
            camera_poses = _as_numpy(value)
        elif name.endswith("::proprio"):
            proprio = _as_numpy(value)

    if rgb is None:
        raise ValueError("head RGB is missing from evaluator observation")
    if depth is None:
        raise ValueError("head depth_linear is missing from evaluator observation")
    if camera_poses is None:
        raise ValueError("camera-relative poses are missing from evaluator observation")

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"head RGB must be HxWx3 or HxWx4, got {image.shape}")
    height, width = image.shape[:2]
    depth_image = np.asarray(depth, dtype=np.float32).squeeze()
    if depth_image.ndim != 2:
        raise ValueError(
            f"head depth_linear must be HxW or HxWx1, got {np.asarray(depth).shape}"
        )
    if depth_image.shape != (height, width):
        raise ValueError("head RGB and depth_linear resolutions do not match")

    poses = np.asarray(camera_poses, dtype=np.float64).reshape(-1)
    head_offset = 14
    if poses.size < head_offset + 7:
        raise ValueError("camera-relative poses do not contain the head camera")
    camera_position_robot = poses[head_offset : head_offset + 3]
    camera_quaternion_robot = poses[head_offset + 3 : head_offset + 7]
    camera_to_robot_base = transform_from_pose(
        camera_position_robot,
        camera_quaternion_robot,
    )
    camera_to_policy = camera_to_policy_from_odometry(
        base_xy_yaw,
        camera_position_robot,
        camera_quaternion_robot,
    )
    frame_rgb = image.copy() if copy_arrays else image
    frame_depth = depth_image.copy() if copy_arrays else depth_image
    proprio_vector = None
    if proprio is not None:
        proprio_vector = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio_vector.size != PROPRIO_DIM:
            raise ValueError(
                f"proprioception must contain {PROPRIO_DIM} values, "
                f"got {proprio_vector.size}"
            )
        if not np.all(np.isfinite(proprio_vector)):
            raise ValueError("proprioception contains non-finite values")
    return TrackerFrame(
        rgb=frame_rgb,
        depth_linear=frame_depth,
        intrinsics=head_camera_intrinsics(width, height),
        camera_to_robot_base=camera_to_robot_base,
        camera_to_policy=camera_to_policy,
        sequence=int(sequence),
        timestamp_s=float(timestamp_s),
        episode_id=str(episode_id),
        camera_role="head",
        color_order="rgb",
        proprio=(
            None
            if proprio_vector is None
            else proprio_vector.copy()
            if copy_arrays
            else proprio_vector
        ),
    )


def _copy_frame(frame: TrackerFrame) -> TrackerFrame:
    return TrackerFrame(
        rgb=np.asarray(frame.rgb).copy(),
        depth_linear=np.asarray(frame.depth_linear).copy(),
        intrinsics=frame.intrinsics,
        camera_to_robot_base=np.asarray(
            frame.camera_to_robot_base,
            dtype=np.float64,
        ).copy(),
        camera_to_policy=np.asarray(frame.camera_to_policy, dtype=np.float64).copy(),
        sequence=int(frame.sequence),
        timestamp_s=float(frame.timestamp_s),
        episode_id=str(frame.episode_id),
        camera_role=str(frame.camera_role),
        color_order=str(frame.color_order),
        proprio=(
            None
            if frame.proprio is None
            else np.asarray(frame.proprio, dtype=np.float64).copy()
        ),
    )


def _compact_replay_frame(frame: TrackerFrame) -> TrackerFrame:
    """Keep exactly the grayscale and geometry consumed by the tracker."""

    gray = DynamicPointTracker._to_gray(frame.rgb, frame.color_order)
    return TrackerFrame(
        rgb=gray.copy(),
        depth_linear=np.asarray(frame.depth_linear, dtype=np.float32).copy(),
        intrinsics=frame.intrinsics,
        camera_to_robot_base=np.asarray(
            frame.camera_to_robot_base,
            dtype=np.float64,
        ).copy(),
        camera_to_policy=np.asarray(frame.camera_to_policy, dtype=np.float64).copy(),
        sequence=int(frame.sequence),
        timestamp_s=float(frame.timestamp_s),
        episode_id=str(frame.episode_id),
        camera_role=str(frame.camera_role),
        color_order="rgb",
    )


def tracker_rgb_as_uint8(rgb: Any, color_order: str) -> np.ndarray:
    """Canonicalize evaluator RGB exactly once for capture and tracking."""

    image = np.asarray(rgb)
    order = str(color_order or "").strip().lower()
    if order not in ("rgb", "bgr"):
        raise ValueError("color_order must be rgb or bgr")
    if image.ndim == 2:
        gray = DynamicPointTracker._as_uint8_image(image)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"RGB must be HxWx3 or HxWx4, got {image.shape}")
    color = DynamicPointTracker._as_uint8_image(image[..., :3])
    if order == "bgr":
        color = color[..., ::-1]
    return np.ascontiguousarray(color, dtype=np.uint8)


def tracker_frame_content_digest(frame: TrackerFrame) -> str:
    """Digest the exact RGB-D and camera geometry consumed by the tracker."""

    canonical_rgb = tracker_rgb_as_uint8(frame.rgb, frame.color_order)
    gray = np.ascontiguousarray(
        DynamicPointTracker._to_gray(canonical_rgb, "rgb"),
        dtype=np.uint8,
    )
    depth = np.ascontiguousarray(
        np.asarray(frame.depth_linear, dtype="<f4").squeeze(),
    )
    if depth.ndim != 2 or depth.shape != gray.shape:
        raise ValueError("RGB and depth_linear resolutions do not match")
    intrinsics = frame.intrinsics
    intrinsics.validate()
    if (intrinsics.height, intrinsics.width) != gray.shape:
        raise ValueError("camera intrinsics resolution does not match RGB-D")
    camera_to_robot_base = np.ascontiguousarray(
        np.asarray(frame.camera_to_robot_base, dtype="<f8"),
    )
    if camera_to_robot_base.shape != (4, 4):
        raise ValueError("camera_to_robot_base must be a 4x4 transform")
    if not np.all(np.isfinite(camera_to_robot_base)):
        raise ValueError("camera_to_robot_base must be finite")

    digest = hashlib.sha256()
    digest.update(b"official-track-object-distance-frame-v1\0")
    digest.update(
        np.asarray([gray.shape[0], gray.shape[1]], dtype="<u4").tobytes()
    )
    digest.update(canonical_rgb.tobytes(order="C"))
    digest.update(gray.tobytes(order="C"))
    digest.update(depth.tobytes(order="C"))
    digest.update(
        np.asarray(
            [
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
            ],
            dtype="<f8",
        ).tobytes()
    )
    digest.update(camera_to_robot_base.tobytes(order="C"))
    return digest.hexdigest()


class TrackObjectDistanceBindingError(ValueError):
    """A fail-closed frame-binding error with a stable machine reason."""

    def __init__(self, reason: str, message: str) -> None:
        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("binding error reason is required")
        self.reason = normalized_reason
        super().__init__(str(message))


@dataclass(frozen=True)
class _CaptureFrameBinding:
    session_id: str
    image_id: str
    episode_id: str
    observation_sequence: int
    image_shape: tuple[int, int]
    frame_digest: str


@dataclass(frozen=True)
class _StoredReplayFrame:
    """Losslessly compressed tracker input retained under a byte budget."""

    gray_zlib: bytes
    depth_zlib: bytes
    image_shape: tuple[int, int]
    intrinsics: CameraIntrinsics
    camera_to_robot_base: np.ndarray
    camera_to_policy: np.ndarray
    sequence: int
    timestamp_s: float
    episode_id: str
    camera_role: str
    storage_bytes: int
    archive_payload: ReplayFramePayload | None = None


@dataclass(frozen=True)
class _PendingReplayFrame:
    """One ordered compression job that has not entered replay history yet."""

    generation: int
    sequence: int
    episode_id: str
    input_bytes: int
    future: Future[_StoredReplayFrame]


def _store_replay_frame(frame: TrackerFrame) -> _StoredReplayFrame:
    # Compression consumes these arrays synchronously and retains only bytes.
    # Avoid copying the already-owned frozen tracker frame into a temporary
    # TrackerFrame first; the canonicalization and resulting zlib payload stay
    # byte-for-byte identical to _compact_replay_frame().
    gray = np.ascontiguousarray(
        DynamicPointTracker._to_gray(frame.rgb, frame.color_order),
        dtype=np.uint8,
    )
    depth = np.ascontiguousarray(frame.depth_linear, dtype="<f4")
    if gray.ndim != 2 or depth.shape != gray.shape:
        raise ValueError("compact replay RGB-D must have matching HxW shapes")
    gray_zlib = zlib.compress(
        memoryview(gray).cast("B"),
        TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL,
    )
    depth_zlib = zlib.compress(
        memoryview(depth).cast("B"),
        TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL,
    )
    camera_to_robot_base = np.array(
        frame.camera_to_robot_base,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    camera_to_policy = np.array(
        frame.camera_to_policy,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    storage_bytes = (
        len(gray_zlib)
        + len(depth_zlib)
        + int(camera_to_robot_base.nbytes)
        + int(camera_to_policy.nbytes)
        + 256
    )
    return _StoredReplayFrame(
        gray_zlib=gray_zlib,
        depth_zlib=depth_zlib,
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        intrinsics=frame.intrinsics,
        camera_to_robot_base=camera_to_robot_base,
        camera_to_policy=camera_to_policy,
        sequence=int(frame.sequence),
        timestamp_s=float(frame.timestamp_s),
        episode_id=str(frame.episode_id),
        camera_role=str(frame.camera_role),
        storage_bytes=storage_bytes,
    )


def _archive_replay_frame(
    frame: TrackerFrame, archive: ReplayFrameArchive,
) -> _StoredReplayFrame:
    stored = _store_replay_frame(frame)
    try:
        payload = archive.store(stored.gray_zlib, stored.depth_zlib)
    except OSError as exc:
        raise TrackObjectDistanceBindingError(
            "replay_storage_unavailable", f"lossless replay archive unavailable: {exc}",
        ) from exc
    return replace(stored, gray_zlib=b"", depth_zlib=b"", archive_payload=payload)


def _restore_replay_frame(stored: _StoredReplayFrame) -> TrackerFrame:
    height, width = stored.image_shape
    expected_gray_bytes = height * width
    expected_depth_bytes = expected_gray_bytes * np.dtype("<f4").itemsize
    try:
        payload = getattr(stored, "archive_payload", None)
        gray_zlib, depth_zlib = (
            payload.read() if payload is not None
            else (stored.gray_zlib, stored.depth_zlib)
        )
        gray_bytes = zlib.decompress(gray_zlib)
        depth_bytes = zlib.decompress(depth_zlib)
    except (OSError, zlib.error) as exc:
        raise TrackObjectDistanceBindingError(
            "replay_history_corrupt",
            "capture replay history failed its lossless decompression check",
        ) from exc
    if (
        len(gray_bytes) != expected_gray_bytes
        or len(depth_bytes) != expected_depth_bytes
    ):
        raise TrackObjectDistanceBindingError(
            "replay_history_corrupt",
            "capture replay history has an invalid decoded frame size",
        )
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(height, width).copy()
    depth = (
        np.frombuffer(depth_bytes, dtype="<f4")
        .reshape(height, width)
        .astype(np.float32, copy=True)
    )
    return TrackerFrame(
        rgb=gray,
        depth_linear=depth,
        intrinsics=stored.intrinsics,
        camera_to_robot_base=stored.camera_to_robot_base.copy(),
        camera_to_policy=stored.camera_to_policy.copy(),
        sequence=int(stored.sequence),
        timestamp_s=float(stored.timestamp_s),
        episode_id=str(stored.episode_id),
        camera_role=str(stored.camera_role),
        color_order="rgb",
    )


def _entry_with_binding(
    entry: dict[str, Any],
    binding: _CaptureFrameBinding | None,
) -> dict[str, Any]:
    if binding is None:
        return entry
    entry.update(
        {
            "source_session_id": binding.session_id,
            "source_image_id": binding.image_id,
            "source_observation_sequence": int(
                binding.observation_sequence
            ),
            "initial_grounding": "policy_owned_frozen_head_capture",
        }
    )
    return entry


def _memory_entry(snapshot: DynamicPointSnapshot) -> dict[str, Any]:
    if (
        snapshot.status != OBSERVED
        or snapshot.depth_m is None
        or snapshot.uv is None
        or snapshot.point_robot_base_m is None
    ):
        raise ValueError("only currently observed points may enter distance memory")
    depth_m = float(snapshot.depth_m)
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("tracked depth must be positive and finite")
    xyz_in_robot_base_coord_m = [
        float(value) for value in snapshot.point_robot_base_m
    ]
    if len(xyz_in_robot_base_coord_m) != 3 or not all(
        math.isfinite(value) for value in xyz_in_robot_base_coord_m
    ):
        raise ValueError(
            "tracked XYZ in robot base coord must contain three finite values"
        )
    return {
        "name": snapshot.label,
        "depth_m": depth_m,
        "unit": "m",
        "u": float(snapshot.uv[0]),
        "v": float(snapshot.uv[1]),
        "xyz_in_robot_base_coord_m": xyz_in_robot_base_coord_m,
        "xyz_frame": "current_robot_base",
        "xyz_axes": "x_forward_y_left_z_up",
        "track_id": snapshot.track_id,
        "status": OBSERVED,
        "camera": snapshot.camera_role,
        "confidence": float(snapshot.confidence),
        "observation_sequence": int(snapshot.observation_sequence),
        "episode_id": snapshot.episode_id,
        "depth_source": "current_evaluator_depth_linear",
        "xyz_source": "current_evaluator_depth_linear_and_cam_rel_poses",
        "identity_source": "model_annotation",
        "identity_verified": False,
    }


def _unobserved_memory_entry(
    snapshot: DynamicPointSnapshot,
    name: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "depth_m": None,
        "unit": "m",
        "u": None,
        "v": None,
        "xyz_in_robot_base_coord_m": None,
        "xyz_frame": "current_robot_base",
        "xyz_axes": "x_forward_y_left_z_up",
        "track_id": snapshot.track_id,
        "status": "temporarily_unobserved",
        "tracker_status": snapshot.status,
        "camera": snapshot.camera_role,
        "confidence": float(snapshot.confidence),
        "observation_sequence": int(snapshot.observation_sequence),
        "last_seen_sequence": int(snapshot.last_seen_sequence),
        "episode_id": snapshot.episode_id,
        "depth_source": None,
        "xyz_source": None,
        "identity_source": "model_annotation",
        "identity_verified": False,
    }


def _missing_tracker_memory_entry(
    *,
    name: str,
    track_id: str,
    previous: Mapping[str, Any] | None,
    observation_sequence: int,
    episode_id: str,
) -> dict[str, Any]:
    """Keep only registration identity when a retained tracker omits a row."""

    prior = dict(previous or {})
    return {
        "name": str(name),
        "depth_m": None,
        "unit": "m",
        "u": None,
        "v": None,
        "xyz_in_robot_base_coord_m": None,
        "xyz_frame": "current_robot_base",
        "xyz_axes": "x_forward_y_left_z_up",
        "track_id": str(track_id),
        "status": "temporarily_unobserved",
        "tracker_status": "missing",
        "camera": str(prior.get("camera") or "head"),
        "confidence": 0.0,
        "observation_sequence": int(observation_sequence),
        "last_seen_sequence": int(
            prior.get("last_seen_sequence")
            or prior.get("observation_sequence")
            or observation_sequence
        ),
        "episode_id": str(episode_id),
        "depth_source": None,
        "xyz_source": None,
        "identity_source": "model_annotation",
        "identity_verified": False,
    }


class TrackedObjectDistanceMemory:
    """Own the live tracker and expose only currently measured named depths."""

    # The official policy adapter publishes immutable-by-replacement
    # observation snapshots.  Its hot path may opt into the borrowed-frame
    # ingest below; all public/standalone callers retain the defensive-copy
    # default.
    _supports_borrowed_ingest = True

    def __init__(
        self,
        *,
        tracker_factory: Callable[[], DynamicPointTracker] = DynamicPointTracker,
        replay_max_frames: int = TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES,
        replay_max_bytes: int = TRACK_OBJECT_DISTANCE_REPLAY_MAX_BYTES,
        replay_memory_max_bytes: int = TRACK_OBJECT_DISTANCE_REPLAY_MEMORY_MAX_BYTES,
        max_capture_bindings: int = TRACK_OBJECT_DISTANCE_MAX_CAPTURE_BINDINGS,
    ) -> None:
        if int(replay_max_frames) < 2:
            raise ValueError("replay_max_frames must be at least 2")
        if int(replay_max_bytes) < 1:
            raise ValueError("replay_max_bytes must be positive")
        if int(replay_memory_max_bytes) < 0:
            raise ValueError("replay_memory_max_bytes must be nonnegative")
        if int(max_capture_bindings) < 1:
            raise ValueError("max_capture_bindings must be positive")
        self._lock = threading.RLock()
        self._tracker_factory = tracker_factory
        self._replay_max_frames = int(replay_max_frames)
        self._replay_max_bytes = int(replay_max_bytes)
        self._replay_memory_max_bytes = min(int(replay_memory_max_bytes), self._replay_max_bytes)
        self._replay_archive = ReplayFrameArchive(self._replay_memory_max_bytes)
        self._max_capture_bindings = int(max_capture_bindings)
        self._max_capture_bindings_total = (
            self._max_capture_bindings * TRACK_OBJECT_DISTANCE_MAX_BINDING_SESSIONS
        )
        self._tracker = tracker_factory()
        self._latest_frame: TrackerFrame | None = None
        self._frame_history: deque[_StoredReplayFrame] = deque()
        self._replay_history_bytes = 0
        self._replay_generation = 0
        self._pending_replay: deque[_PendingReplayFrame] = deque()
        self._pending_replay_bytes = 0
        self._replay_async_submitted = 0
        self._replay_async_committed = 0
        self._replay_async_discarded = 0
        self._capture_bindings: OrderedDict[
            tuple[str, str], _CaptureFrameBinding
        ] = OrderedDict()
        self._binding_failures: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._registration_binding: _CaptureFrameBinding | None = None
        self._name_by_track_id: dict[str, str] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._registration_entries: dict[str, dict[str, Any]] = {}
        self._active_rigid_pair: dict[str, Any] | None = None
        self._active_rigid_pair_report: dict[str, Any] = {}
        self._active_on_hand_eef_prior: dict[str, Any] | None = None
        self._on_hand_eef_prior_counter = 0
        self._temporarily_unobserved: dict[str, dict[str, Any]] = {}
        self._motion_retention_leases: dict[str, dict[str, Any]] = {}
        self._motion_retention_counter = 0
        self._last_deleted: list[dict[str, Any]] = []
        self._last_error: str | None = None

    def reset(self) -> None:
        with self._lock:
            for key in list(self._capture_bindings):
                self._remember_binding_failure_locked(key, "episode_changed")
            self._tracker = self._tracker_factory()
            self._latest_frame = None
            self._clear_replay_history_locked()
            self._discard_pending_replay_locked()
            self._capture_bindings.clear()
            self._registration_binding = None
            self._name_by_track_id.clear()
            self._entries.clear()
            self._registration_entries.clear()
            self._active_rigid_pair = None
            self._active_rigid_pair_report = {}
            self._active_on_hand_eef_prior = None
            self._temporarily_unobserved.clear()
            self._motion_retention_leases.clear()
            self._last_deleted.clear()
            self._last_error = None

    def clear_for_reselection(self) -> dict[str, Any]:
        """Drop the current named set before a new public selection starts.

        This is deliberately narrower than ``reset()``: the synchronized
        evaluator frame, replay history, and capture bindings remain available
        for the replacement request.  Only the live point set and any
        pair/prior state derived from it are retired.  Callers use the returned
        names to make the replacement visible in structured results.
        """

        with self._lock:
            # Upgrade a live development instance before touching its state so
            # an old class layout cannot make the reselection reset partial.
            self.upgrade_runtime_state()
            previous_names = [
                str(name) for name in self._name_by_track_id.values()
            ]
            replacement_tracker = self._tracker_factory()
            deactivate = getattr(self._tracker, "deactivate_rigid_pair", None)
            if callable(deactivate):
                deactivate()
            self._tracker = replacement_tracker
            self._name_by_track_id.clear()
            self._entries.clear()
            self._registration_entries.clear()
            self._registration_binding = None
            self._active_rigid_pair = None
            self._active_rigid_pair_report = {}
            self._active_on_hand_eef_prior = None
            self._temporarily_unobserved.clear()
            self._motion_retention_leases.clear()
            self._last_deleted = []
            self._last_error = None
            return {
                "cleared_names": previous_names,
                "cleared_count": len(previous_names),
                "replacement": "complete_named_set",
            }

    def upgrade_runtime_state(self) -> dict[str, Any]:
        """Initialize fields added by a hot-reloaded test-local build.

        Fresh evaluator processes never need this path.  It exists so the
        development live runner can upgrade its policy-owned memory instance
        without restarting or reaching into the simulator process.
        """

        with self._lock:
            # Existing instances retain their configured limits during reload.
            # New default capacity takes effect on construction; old in-memory
            # frames remain readable alongside newly archived payloads.
            if not hasattr(self, "_replay_archive"):
                self._replay_memory_max_bytes = min(
                    self._replay_max_bytes, TRACK_OBJECT_DISTANCE_REPLAY_MEMORY_MAX_BYTES,
                )
                self._replay_archive = ReplayFrameArchive(self._replay_memory_max_bytes)
            tracker_factory_upgraded = False
            tracker_instance_upgraded = False
            tracker_upgrade_report: dict[str, Any] = {}
            factory = getattr(self, "_tracker_factory", None)
            if (
                factory is not DynamicPointTracker
                and getattr(factory, "__module__", None)
                == DynamicPointTracker.__module__
                and getattr(factory, "__name__", None)
                == DynamicPointTracker.__name__
            ):
                # importlib.reload() keeps old class objects alive in existing
                # policy memory. Their methods then resolve the module's new
                # _TrackState global, which is an invalid old/new hybrid.
                self._tracker_factory = DynamicPointTracker
                tracker_factory_upgraded = True
            tracker = getattr(self, "_tracker", None)
            if (
                tracker is not None
                and tracker.__class__ is not DynamicPointTracker
                and getattr(tracker.__class__, "__module__", None)
                == DynamicPointTracker.__module__
                and getattr(tracker.__class__, "__name__", None)
                == DynamicPointTracker.__name__
            ):
                tracker.__class__ = DynamicPointTracker
                tracker_instance_upgraded = True
            tracker_upgrade = getattr(tracker, "upgrade_runtime_state", None)
            if callable(tracker_upgrade):
                tracker_upgrade_report = dict(tracker_upgrade())
            if not hasattr(self, "_registration_entries"):
                self._registration_entries = {}
            if not hasattr(self, "_active_rigid_pair"):
                self._active_rigid_pair = None
            if not hasattr(self, "_active_rigid_pair_report"):
                self._active_rigid_pair_report = {}
            if not hasattr(self, "_active_on_hand_eef_prior"):
                self._active_on_hand_eef_prior = None
            if not hasattr(self, "_on_hand_eef_prior_counter"):
                self._on_hand_eef_prior_counter = 0
            if not hasattr(self, "_motion_retention_leases"):
                self._motion_retention_leases = {}
            if not hasattr(self, "_motion_retention_counter"):
                self._motion_retention_counter = 0
            # Hot-reloaded/live instances can retain a registration entry or
            # public memory row after the underlying tracker has retired its
            # ID.  Reconcile those name maps before any pair is reactivated.
            registered_names = {
                str(name) for name in self._name_by_track_id.values()
            }
            for name in list(self._entries):
                if str(name) not in registered_names:
                    self._entries.pop(name, None)
            for name in list(self._registration_entries):
                if str(name) not in registered_names:
                    self._registration_entries.pop(name, None)
            tracker_pair_report: dict[str, Any] = {}
            if self._active_rigid_pair is not None:
                pair_names = tuple(
                    str(name) for name in self._active_rigid_pair.get("names", ())
                )
                missing_pair_names = [
                    name for name in pair_names if name not in registered_names
                ]
                if not missing_pair_names:
                    track_by_name = {
                        str(name): str(track_id)
                        for track_id, name in self._name_by_track_id.items()
                    }
                    list_points = getattr(self._tracker, "list_points", None)
                    if callable(list_points):
                        tracker_ids = {
                            str(snapshot.track_id) for snapshot in list_points()
                        }
                        missing_pair_names = [
                            name
                            for name in pair_names
                            if track_by_name[name] not in tracker_ids
                        ]
                if missing_pair_names:
                    # A member can have been retired between evaluator
                    # observations while another named point remained alive.
                    # Retire the stale wrapper pair and let callers establish
                    # a fresh full registration instead of raising before the
                    # new points are even inspected.
                    tracker_pair_report = self._retire_active_rigid_pair_locked(
                        reason="rigid_pair_track_removed",
                        unavailable=missing_pair_names,
                    )
                else:
                    try:
                        tracker_pair_report = (
                            self._activate_tracker_rigid_pair_locked(
                                self._active_rigid_pair
                            )
                        )
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        # A hot-reloaded tracker may retain the names while its
                        # internal IDs have already been retired.  Treat that
                        # state exactly like a removed member so a new complete
                        # selection can recover instead of inheriting the old
                        # pair error.
                        tracker_pair_report = (
                            self._retire_active_rigid_pair_locked(
                                reason="rigid_pair_track_removed",
                                unavailable=pair_names,
                            )
                        )
                        tracker_pair_report["activation_error"] = str(exc)
                        self._active_rigid_pair_report = tracker_pair_report
            return {
                "ok": True,
                "build": TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD,
                "tracker_factory_upgraded": tracker_factory_upgraded,
                "tracker_instance_upgraded": tracker_instance_upgraded,
                "tracker_upgrade": tracker_upgrade_report,
                "tracker_rigid_pair": tracker_pair_report,
                "registration_reference_available": bool(
                    self._registration_entries
                ),
                "requires_point_reregistration": bool(
                    self._name_by_track_id and not self._registration_entries
                ),
                "active_motion_retention_leases": len(
                    self._motion_retention_leases
                ),
                "active_on_hand_eef_prior": deepcopy(
                    self._active_on_hand_eef_prior
                ),
            }

    def _purge_motion_retention_locked(
        self,
        *,
        episode_id: str | None = None,
    ) -> None:
        now = time.monotonic()
        for lease_id, lease in list(self._motion_retention_leases.items()):
            if (
                float(lease["deadline_monotonic_s"]) <= now
                or (
                    episode_id is not None
                    and str(lease["episode_id"]) != str(episode_id)
                )
            ):
                self._motion_retention_leases.pop(lease_id, None)

    def _motion_retention_active_locked(
        self,
        name: str,
        *,
        episode_id: str,
    ) -> bool:
        return any(
            str(lease["episode_id"]) == str(episode_id)
            and str(name) in lease["names"]
            for lease in self._motion_retention_leases.values()
        )

    def begin_motion_retention(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Retain visual identity state while a preplanned motion is executing.

        Retention never publishes predicted or stale coordinates. It only keeps
        the policy-owned registration/template alive so later evaluator RGB-D
        frames can reacquire the same named points before final validation.
        """

        requested = tuple(str(name or "").strip() for name in names)
        expected_episode = str(episode_id or "").strip()
        duration = float(timeout_s)
        if not requested or any(not name for name in requested):
            raise ValueError("motion retention names must be non-empty")
        if len(set(requested)) != len(requested):
            raise ValueError("motion retention names must be unique")
        if not expected_episode:
            raise ValueError("motion retention episode_id is required")
        if (
            not math.isfinite(duration)
            or duration <= 0.0
            or duration > TRACK_OBJECT_DISTANCE_MOTION_RETENTION_MAX_S
        ):
            raise ValueError(
                "motion retention timeout_s must be finite and in "
                f"(0, {TRACK_OBJECT_DISTANCE_MOTION_RETENTION_MAX_S:g}]"
            )

        with self._lock:
            self.upgrade_runtime_state()
            latest = self._latest_frame
            if latest is None:
                raise ValueError("no synchronized evaluator frame is available")
            if str(latest.episode_id) != expected_episode:
                raise ValueError("motion retention episode changed")
            registered = set(self._name_by_track_id.values())
            missing = [name for name in requested if name not in registered]
            if missing:
                raise ValueError(
                    "motion retention points are not registered: "
                    + ", ".join(missing)
                )
            self._purge_motion_retention_locked(episode_id=expected_episode)
            superseded_lease_ids: list[str] = []
            requested_set = set(requested)
            for existing_lease_id, existing_lease in list(
                self._motion_retention_leases.items()
            ):
                if (
                    str(existing_lease["episode_id"]) == expected_episode
                    and {
                        str(name) for name in existing_lease["names"]
                    }
                    == requested_set
                ):
                    self._motion_retention_leases.pop(existing_lease_id, None)
                    superseded_lease_ids.append(str(existing_lease_id))
            self._motion_retention_counter += 1
            lease_id = (
                f"motion-retention-{self._motion_retention_counter:08d}"
            )
            self._motion_retention_leases[lease_id] = {
                "lease_id": lease_id,
                "names": tuple(requested),
                "episode_id": expected_episode,
                "deadline_monotonic_s": time.monotonic() + duration,
                "duration_s": duration,
            }
            return {
                "ok": True,
                "lease_id": lease_id,
                "names": list(requested),
                "episode_id": expected_episode,
                "duration_s": duration,
                "superseded_lease_ids": superseded_lease_ids,
                "coordinates_published_while_unobserved": False,
            }

    def end_motion_retention(self, lease_id: str) -> dict[str, Any]:
        token = str(lease_id or "").strip()
        if not token:
            raise ValueError("motion retention lease_id is required")
        with self._lock:
            lease = self._motion_retention_leases.pop(token, None)
            return {
                "ok": lease is not None,
                "lease_id": token,
                "released": lease is not None,
            }

    def _activate_tracker_rigid_pair_locked(
        self,
        pair: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Push the registered pair into the current observation-only tracker."""

        names = tuple(str(name) for name in pair["names"])
        track_by_name = {
            name: track_id for track_id, name in self._name_by_track_id.items()
        }
        missing = [name for name in names if name not in track_by_name]
        if missing:
            raise ValueError(
                "rigid-pair tracker ids are unavailable for "
                + ", ".join(missing)
            )
        activate = getattr(self._tracker, "activate_rigid_pair", None)
        if not callable(activate):
            raise RuntimeError(
                "dynamic tracker does not expose shared rigid-pair tracking"
            )
        prior_required = bool(
            self._active_on_hand_eef_prior is not None
            and tuple(self._active_on_hand_eef_prior["names"]) == names
        )
        try:
            result = activate(
                [track_by_name[name] for name in names],
                reference_distance_m=float(pair["reference_distance_m"]),
                observation_prior_required=prior_required,
            )
        except TypeError:
            if prior_required:
                raise
            result = activate(
                [track_by_name[name] for name in names],
                reference_distance_m=float(pair["reference_distance_m"]),
            )
        return dict(result)

    def _retire_active_rigid_pair_locked(
        self,
        *,
        reason: str = "rigid_pair_track_removed",
        unavailable: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Retire wrapper pair state when one of its tracker members is gone.

        ``DynamicPointTracker.remove_point()`` retires its own pair, but the
        manager also keeps a name-based pair for RGB-D fusion and the optional
        on-hand prior.  Keeping those layers in lockstep is important: a
        subsequent full ``replace_points`` call must never be blocked by an
        obsolete name-to-track-id mapping.
        """

        pair = self._active_rigid_pair
        if pair is None:
            self._active_rigid_pair_report = {}
            return {}
        names = tuple(str(name) for name in pair.get("names", ()))
        missing = (
            [str(name) for name in unavailable]
            if unavailable is not None
            else [
                name
                for name in names
                if name
                not in {
                    str(candidate_name)
                    for candidate_name in self._name_by_track_id.values()
                }
            ]
        )
        visual_report = (
            self._tracker.rigid_pair_report()
            if callable(getattr(self._tracker, "rigid_pair_report", None))
            else {}
        )
        try:
            reference_distance = float(pair.get("reference_distance_m"))
        except (TypeError, ValueError):
            reference_distance = None
        if reference_distance is not None and not math.isfinite(
            reference_distance
        ):
            reference_distance = None
        entry_retirement = self._restore_raw_rigid_pair_entries_locked(names)
        report = {
            "ok": False,
            "active": False,
            "reason": str(reason),
            "names": list(names),
            "unavailable": missing,
            "reference_distance_m": reference_distance,
            "visual_tracker_pair": deepcopy(visual_report),
            "fusion_applied": False,
            "raw_rgbd_measurement_published": bool(
                entry_retirement["restored_raw_names"]
                or entry_retirement["unchanged_raw_names"]
            ),
            "prediction_values_published_as_observation": False,
            "stale_depth_published": False,
            "stale_xyz_in_robot_base_coord_published": False,
            "entry_retirement": entry_retirement,
            "build": TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD,
        }

        deactivate = getattr(self._tracker, "deactivate_rigid_pair", None)
        if callable(deactivate):
            deactivate()
        self._active_rigid_pair = None
        self._active_rigid_pair_report = report
        prior = self._active_on_hand_eef_prior
        if prior is not None and set(names) & {
            str(name) for name in prior.get("names", ())
        }:
            self._active_on_hand_eef_prior = None
        return deepcopy(report)

    def _restore_raw_rigid_pair_entries_locked(
        self,
        names: tuple[str, ...] | list[str],
    ) -> dict[str, Any]:
        """Remove pair fusion from public rows without exposing stale XYZ.

        Successful rigid-pair projection replaces the public XYZ while keeping
        that observation's direct RGB-D XYZ in ``raw_xyz_*``.  Pair retirement
        must reverse that replacement in the same critical section.  A row
        that is no longer observed, or a legacy fused row whose raw value is
        unavailable, is made explicitly unavailable instead of retaining the
        projected coordinate.
        """

        restored: list[str] = []
        fail_closed: list[str] = []
        unchanged: list[str] = []
        missing: list[str] = []
        for candidate_name in names:
            name = str(candidate_name)
            entry = self._entries.get(name)
            if entry is None:
                missing.append(name)
                continue

            raw_present = "raw_xyz_in_robot_base_coord_m" in entry
            raw_xyz = entry.pop("raw_xyz_in_robot_base_coord_m", None)
            raw_source = entry.pop("raw_xyz_source", None)
            xyz_source = str(entry.get("xyz_source") or "")
            fused_source = "registration_rigid_pair" in xyz_source
            entry.pop("rigid_pair_fusion", None)

            status = str(entry.get("status") or "")
            valid_raw: list[float] | None = None
            if raw_present:
                try:
                    raw_array = np.asarray(raw_xyz, dtype=np.float64).reshape(3)
                except (TypeError, ValueError):
                    raw_array = np.empty(0, dtype=np.float64)
                if (
                    raw_array.shape == (3,)
                    and np.all(np.isfinite(raw_array))
                    and "registration_rigid_pair" not in str(raw_source or "")
                ):
                    valid_raw = raw_array.astype(float).tolist()

            if status == OBSERVED and valid_raw is not None:
                entry["xyz_in_robot_base_coord_m"] = valid_raw
                entry["xyz_source"] = str(
                    raw_source
                    or "current_evaluator_depth_linear_and_cam_rel_poses"
                )
                self._temporarily_unobserved.pop(name, None)
                restored.append(name)
                continue

            if status == OBSERVED and not raw_present and not fused_source:
                # Activation can fail before applying fusion.  In that case the
                # public row is already the direct current RGB-D measurement.
                self._temporarily_unobserved.pop(name, None)
                unchanged.append(name)
                continue

            entry.update(
                {
                    "depth_m": None,
                    "u": None,
                    "v": None,
                    "xyz_in_robot_base_coord_m": None,
                    "status": "temporarily_unobserved",
                    "depth_source": None,
                    "xyz_source": None,
                }
            )
            if status == OBSERVED:
                entry["tracker_status"] = "rigid_pair_raw_rgbd_unavailable"
            temporary = dict(self._temporarily_unobserved.get(name) or {})
            temporary.update(
                {
                    "name": name,
                    "status": str(
                        entry.get("tracker_status")
                        or "rigid_pair_raw_rgbd_unavailable"
                    ),
                    "missed_steps": max(
                        1,
                        int(temporary.get("missed_steps", 0)),
                    ),
                    "observation_sequence": int(
                        entry.get("observation_sequence", -1)
                    ),
                    "motion_retention_active": bool(
                        temporary.get("motion_retention_active", False)
                    ),
                }
            )
            self._temporarily_unobserved[name] = temporary
            fail_closed.append(name)

        return {
            "restored_raw_names": restored,
            "fail_closed_names": fail_closed,
            "unchanged_raw_names": unchanged,
            "missing_names": missing,
            "stale_fused_coordinates_published": False,
        }

    def _discard_pending_replay_locked(self) -> None:
        pending = list(self._pending_replay)
        self._pending_replay.clear()
        self._pending_replay_bytes = 0
        self._replay_generation += 1
        self._replay_async_discarded += len(pending)
        for item in pending:
            item.future.cancel()

    def _invalidate_replay_locked(self, reason: str, error: str) -> None:
        for key in list(self._capture_bindings):
            self._remove_binding_locked(key, reason)
        self._clear_replay_history_locked()
        self._discard_pending_replay_locked()
        self._last_error = str(error)

    def _drain_ready_replay_locked(self) -> None:
        """Commit only already-finished compression jobs without blocking."""

        while self._pending_replay and self._pending_replay[0].future.done():
            pending = self._pending_replay.popleft()
            self._pending_replay_bytes = max(
                0,
                self._pending_replay_bytes - int(pending.input_bytes),
            )
            if pending.generation != self._replay_generation:
                self._replay_async_discarded += 1
                continue
            try:
                stored = pending.future.result()
                if (
                    stored.sequence != pending.sequence
                    or stored.episode_id != pending.episode_id
                ):
                    raise ValueError(
                        "asynchronous replay compression changed frame identity"
                    )
                self._append_stored_replay_frame_locked(stored)
                self._replay_async_committed += 1
            except Exception as exc:
                self._invalidate_replay_locked(
                    exc.reason if isinstance(exc, TrackObjectDistanceBindingError)
                    else "replay_history_inconsistent",
                    f"{type(exc).__name__}: {exc}",
                )
                return
            if not self._capture_bindings and self._pending_replay:
                self._discard_pending_replay_locked()
                return

    def _schedule_replay_locked(self, frame: TrackerFrame) -> None:
        self._drain_ready_replay_locked()
        if not self._capture_bindings:
            return
        if len(self._pending_replay) >= min(
            self._replay_max_frames,
            TRACK_OBJECT_DISTANCE_REPLAY_MAX_PENDING_FRAMES,
        ):
            self._invalidate_replay_locked(
                "replay_history_inconsistent",
                "asynchronous replay compression backlog exceeded its bound",
            )
            return
        input_bytes = (
            int(np.asarray(frame.rgb).nbytes)
            + int(np.asarray(frame.depth_linear).nbytes)
            + int(np.asarray(frame.camera_to_robot_base).nbytes)
            + int(np.asarray(frame.camera_to_policy).nbytes)
        )
        future = _REPLAY_COMPRESSION_EXECUTOR.submit(
            _archive_replay_frame, frame, self._replay_archive,
        )
        self._pending_replay.append(
            _PendingReplayFrame(
                generation=self._replay_generation,
                sequence=int(frame.sequence),
                episode_id=str(frame.episode_id),
                input_bytes=input_bytes,
                future=future,
            )
        )
        self._pending_replay_bytes += input_bytes
        self._replay_async_submitted += 1

    def _apply_replay_backpressure(self, frame: TrackerFrame) -> None:
        """Bound retained raw RGB-D without dropping an exact replay frame."""

        input_bytes = (
            int(np.asarray(frame.rgb).nbytes)
            + int(np.asarray(frame.depth_linear).nbytes)
            + int(np.asarray(frame.camera_to_robot_base).nbytes)
            + int(np.asarray(frame.camera_to_policy).nbytes)
        )
        max_pending_frames = min(
            self._replay_max_frames,
            TRACK_OBJECT_DISTANCE_REPLAY_MAX_PENDING_FRAMES,
        )
        while True:
            with self._lock:
                self._drain_ready_replay_locked()
                over_limit = bool(
                    self._capture_bindings
                    and self._pending_replay
                    and (
                        len(self._pending_replay) >= max_pending_frames
                        or self._pending_replay_bytes + input_bytes
                        > self._replay_max_bytes
                    )
                )
                oldest_sequence = (
                    None
                    if not over_limit
                    else int(self._pending_replay[0].sequence)
                )
            if oldest_sequence is None:
                return
            # This is exceptional backpressure only. The normal observation
            # path returns immediately while compression stays ahead.
            self._flush_replay_through(oldest_sequence)

    def _flush_replay_through(
        self,
        sequence: int,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        """Wait outside the manager lock until exact replay reaches sequence."""

        target = int(sequence)
        flush_started = time.perf_counter()
        _capture_trace(
            "flush_start",
            target=target,
            pending=len(self._pending_replay),
            history=len(self._frame_history),
            bindings=len(self._capture_bindings),
        )
        while True:
            with self._lock:
                self._drain_ready_replay_locked()
                pending = next(
                    (
                        item
                        for item in reversed(self._pending_replay)
                        if item.generation == self._replay_generation
                        and item.sequence <= target
                    ),
                    None,
                )
            if pending is None:
                _capture_trace(
                    "flush_done",
                    target=target,
                    elapsed_ms=(time.perf_counter() - flush_started) * 1000.0,
                    pending=len(self._pending_replay),
                    history=len(self._frame_history),
                )
                return
            if check_cancelled is not None:
                check_cancelled()
            try:
                pending.future.result(timeout=0.05)
            except FutureTimeoutError:
                continue
            except Exception:
                # The next drain records the exact failure and invalidates all
                # affected bindings instead of publishing a replay gap.
                continue

    def _remember_binding_failure_locked(
        self,
        key: tuple[str, str],
        reason: str,
    ) -> None:
        self._binding_failures[key] = str(reason)
        self._binding_failures.move_to_end(key)
        while len(self._binding_failures) > TRACK_OBJECT_DISTANCE_MAX_BINDING_FAILURES:
            self._binding_failures.popitem(last=False)

    def _remove_binding_locked(
        self,
        key: tuple[str, str],
        reason: str,
    ) -> None:
        if self._capture_bindings.pop(key, None) is not None:
            self._remember_binding_failure_locked(key, reason)

    def _popleft_replay_frame_locked(self) -> None:
        stored = self._frame_history.popleft()
        self._replay_history_bytes = max(
            0,
            self._replay_history_bytes - int(stored.storage_bytes),
        )

    def _clear_replay_history_locked(self) -> None:
        self._frame_history.clear()
        self._replay_history_bytes = 0
        self._replay_archive = ReplayFrameArchive(self._replay_memory_max_bytes)

    def _trim_replay_history_locked(self) -> None:
        while self._frame_history and (
            len(self._frame_history) > self._replay_max_frames
            or self._replay_history_bytes > self._replay_max_bytes
        ):
            self._popleft_replay_frame_locked()
        if not self._frame_history:
            for key in list(self._capture_bindings):
                self._remove_binding_locked(key, "replay_history_evicted")
            return

        oldest = self._frame_history[0]
        newest = self._frame_history[-1]
        for key, binding in list(self._capture_bindings.items()):
            if binding.episode_id != newest.episode_id:
                self._remove_binding_locked(key, "episode_changed")
            elif (
                binding.observation_sequence < oldest.sequence
                or binding.observation_sequence > newest.sequence
            ):
                self._remove_binding_locked(key, "replay_history_evicted")
        if not self._capture_bindings:
            self._clear_replay_history_locked()
            return

        first_needed_sequence = min(
            binding.observation_sequence
            for binding in self._capture_bindings.values()
        )
        while (
            self._frame_history
            and self._frame_history[0].sequence < first_needed_sequence
        ):
            self._popleft_replay_frame_locked()

    def _append_replay_frame_locked(self, frame: TrackerFrame) -> None:
        if not self._capture_bindings:
            return
        self._append_stored_replay_frame_locked(
            _archive_replay_frame(frame, self._replay_archive)
        )

    def _append_stored_replay_frame_locked(
        self,
        stored: _StoredReplayFrame,
    ) -> None:
        if not self._capture_bindings:
            return
        if self._frame_history:
            previous = self._frame_history[-1]
            if previous.episode_id != stored.episode_id:
                for key in list(self._capture_bindings):
                    self._remove_binding_locked(key, "episode_changed")
                self._clear_replay_history_locked()
                return
            if stored.sequence <= previous.sequence:
                raise ValueError(
                    "replay observation sequence must increase strictly"
                )
        self._frame_history.append(stored)
        self._replay_history_bytes += int(stored.storage_bytes)
        self._trim_replay_history_locked()

    def _replay_sequence_present_locked(
        self,
        *,
        episode_id: str,
        sequence: int,
    ) -> bool:
        """Return whether one exact frame is already queued or committed."""

        episode = str(episode_id)
        target = int(sequence)
        if any(
            stored.episode_id == episode and stored.sequence == target
            for stored in self._frame_history
        ):
            return True
        return any(
            pending.episode_id == episode and pending.sequence == target
            for pending in self._pending_replay
        )

    def register_capture(
        self,
        *,
        session_id: str,
        image_id: str,
        observation_sequence: int,
        episode_id: str,
        image_shape: tuple[int, int],
        defer_replay_compression: bool = False,
    ) -> dict[str, Any]:
        """Pin one policy-owned capture to the exact synchronized tracker frame.

        ``defer_replay_compression`` is an internal live-runtime fast path. It
        publishes the same binding immediately and queues the lossless replay
        encoding on the existing bounded executor. Consumers that need replay
        (``replace_points``) already cross ``_flush_replay_through`` and thus
        retain the original ordering and failure semantics.
        """

        register_started = time.perf_counter()
        _capture_trace(
            "register_start",
            session=str(session_id),
            image=str(image_id),
            sequence=observation_sequence,
            history=len(self._frame_history),
            pending=len(self._pending_replay),
            bindings=len(self._capture_bindings),
        )

        session = str(session_id or "").strip()
        image = str(image_id or "").strip()
        episode = str(episode_id or "").strip()
        if not session:
            raise ValueError("session_id is required for capture binding")
        if not image:
            raise ValueError("image_id is required for capture binding")
        if not episode:
            raise ValueError("episode_id is required for capture binding")
        if isinstance(observation_sequence, bool):
            raise ValueError("observation_sequence must be an integer")
        try:
            sequence = int(observation_sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_sequence must be an integer") from exc
        if sequence < 0 or sequence != observation_sequence:
            raise ValueError("observation_sequence must be a non-negative integer")
        try:
            height, width = (int(value) for value in image_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_shape must contain height and width") from exc
        if height <= 1 or width <= 1:
            raise ValueError("capture image dimensions must be greater than one")

        # The live callback only needs to validate and publish the binding.
        # Joining an older compression future here turns a one-shot capture
        # into an evaluator-frame stall. Consumers that need replay bytes
        # still cross the exact barrier in ``replace_points``. Keep the
        # historical synchronous path unchanged for standalone callers.
        if defer_replay_compression:
            with self._lock:
                self._drain_ready_replay_locked()
            _capture_trace(
                "register_nonblocking_drain",
                sequence=sequence,
                elapsed_ms=(time.perf_counter() - register_started) * 1000.0,
                history=len(self._frame_history),
                pending=len(self._pending_replay),
            )
        else:
            # An older capture may already be collecting asynchronous replay.
            # The synchronous API retains its exact historical barrier.
            self._flush_replay_through(sequence)
            _capture_trace(
                "register_after_flush",
                sequence=sequence,
                elapsed_ms=(time.perf_counter() - register_started) * 1000.0,
                history=len(self._frame_history),
                pending=len(self._pending_replay),
            )
        with self._lock:
            lock_acquired = time.perf_counter()
            latest = self._latest_frame
            if latest is None:
                raise TrackObjectDistanceBindingError(
                    "no_synchronized_frame",
                    "no synchronized evaluator head RGB-D frame is available "
                    "for capture binding",
                )
            if latest.episode_id != episode:
                raise TrackObjectDistanceBindingError(
                    "episode_mismatch",
                    "capture episode does not match the synchronized tracker frame",
                )
            if latest.sequence != sequence:
                raise TrackObjectDistanceBindingError(
                    "sequence_mismatch",
                    "capture observation sequence does not match the synchronized "
                    "tracker frame",
                )
            if latest.camera_role != "head":
                raise TrackObjectDistanceBindingError(
                    "unsupported_camera_role",
                    "track_object_distance requires a head capture",
                )
            latest_shape = (
                int(latest.intrinsics.height),
                int(latest.intrinsics.width),
            )
            if latest_shape != (height, width):
                raise TrackObjectDistanceBindingError(
                    "image_shape_mismatch",
                    "capture image resolution does not match the synchronized "
                    "tracker frame",
                )

            history_latest = (
                None if not self._frame_history else self._frame_history[-1]
            )
            if history_latest is not None and (
                history_latest.episode_id != episode
                or history_latest.sequence > sequence
            ):
                raise TrackObjectDistanceBindingError(
                    "replay_history_inconsistent",
                    "capture cannot be aligned with replay frame history",
                )
            if not defer_replay_compression:
                if history_latest is None or history_latest.sequence < sequence:
                    store_started = time.perf_counter()
                    _capture_trace(
                        "register_store_start",
                        sequence=sequence,
                        shape=[
                            int(latest.depth_linear.shape[0]),
                            int(latest.depth_linear.shape[1]),
                        ],
                    )
                    stored = _archive_replay_frame(latest, self._replay_archive)
                    _capture_trace(
                        "register_store_done",
                        sequence=sequence,
                        elapsed_ms=(time.perf_counter() - store_started) * 1000.0,
                        storage_bytes=int(stored.storage_bytes),
                    )
                    self._frame_history.append(stored)
                    self._replay_history_bytes += int(stored.storage_bytes)

            binding = _CaptureFrameBinding(
                session_id=session,
                image_id=image,
                episode_id=episode,
                observation_sequence=sequence,
                image_shape=(height, width),
                frame_digest=tracker_frame_content_digest(latest),
            )
            digest_done = time.perf_counter()
            _capture_trace(
                "register_digest_done",
                sequence=sequence,
                elapsed_ms=(digest_done - register_started) * 1000.0,
                lock_wait_ms=(digest_done - lock_acquired) * 1000.0,
            )
            key = (session, image)
            self._binding_failures.pop(key, None)
            self._capture_bindings[key] = binding
            self._capture_bindings.move_to_end(key)
            session_keys = [
                candidate_key
                for candidate_key in self._capture_bindings
                if candidate_key[0] == session
            ]
            while len(session_keys) > self._max_capture_bindings:
                evicted_key = session_keys.pop(0)
                self._remove_binding_locked(evicted_key, "binding_evicted")
            while len(self._capture_bindings) > self._max_capture_bindings_total:
                evicted_key = next(iter(self._capture_bindings))
                self._remove_binding_locked(evicted_key, "binding_evicted")
            if not defer_replay_compression:
                self._trim_replay_history_locked()
            if key not in self._capture_bindings:
                reason = self._binding_failures.get(
                    key,
                    "replay_history_evicted",
                )
                raise TrackObjectDistanceBindingError(
                    reason,
                    "capture frame exceeds the bounded replay history capacity",
                )

            if defer_replay_compression and not self._replay_sequence_present_locked(
                episode_id=episode,
                sequence=sequence,
            ):
                schedule_started = time.perf_counter()
                _capture_trace(
                    "register_deferred_schedule_start",
                    sequence=sequence,
                )
                try:
                    # The binding is installed first so the normal ordered
                    # replay scheduler accepts this first frame. The future is
                    # joined only by the existing replay barrier.
                    self._schedule_replay_locked(latest)
                except Exception as exc:
                    self._remove_binding_locked(
                        key,
                        "replay_history_inconsistent",
                    )
                    raise TrackObjectDistanceBindingError(
                        "replay_history_inconsistent",
                        f"capture replay scheduling failed: {exc}",
                    ) from exc
                _capture_trace(
                    "register_deferred_schedule_done",
                    sequence=sequence,
                    elapsed_ms=(time.perf_counter() - schedule_started) * 1000.0,
                    pending=len(self._pending_replay),
                )
            result = {
                "ok": True,
                "trackable": True,
                "reason": None,
                "session_id": session,
                "image_id": image,
                "episode_id": episode,
                "observation_sequence": sequence,
                "image_shape": [height, width],
                "frame_digest": binding.frame_digest,
                "frame_digest_algorithm": "sha256",
                "frame_binding": "exact_synchronized_evaluator_tracker_frame",
                "binding_capacity_per_session": self._max_capture_bindings,
                "replay_capacity_frames": self._replay_max_frames,
                "replay_capacity_bytes": self._replay_max_bytes,
                "replay_capacity_point_frames": TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK,
                **self._replay_archive.status(),
                "replay_storage": "lossless_zlib_gray_and_float32_depth",
            }
            _capture_trace(
                "register_done",
                sequence=sequence,
                elapsed_ms=(time.perf_counter() - register_started) * 1000.0,
                history=len(self._frame_history),
                bindings=len(self._capture_bindings),
            )
            return result

    def register_capture_deferred(self, **kwargs: Any) -> dict[str, Any]:
        """Register a live capture without joining its replay compression.

        This narrow wrapper keeps the historical ``register_capture`` API
        synchronous for standalone callers and makes the live-only choice
        explicit at the tool boundary.
        """

        return self.register_capture(
            defer_replay_compression=True,
            **kwargs,
        )

    def _bound_replay_frames_locked(
        self,
        *,
        session_id: str,
        image_id: str,
        capture_observation_sequence: int,
        capture_episode_id: str,
        capture_image_shape: tuple[int, int],
        capture_frame_digest: str,
    ) -> tuple[_CaptureFrameBinding, list[_StoredReplayFrame]]:
        key = (str(session_id), str(image_id))
        binding = self._capture_bindings.get(key)
        if binding is None:
            reason = self._binding_failures.get(key)
            if reason is None and any(
                candidate_image == key[1] and candidate_session != key[0]
                for candidate_session, candidate_image in (
                    list(self._capture_bindings) + list(self._binding_failures)
                )
            ):
                reason = "session_mismatch"
            reason = reason or "binding_not_registered"
            messages = {
                "binding_evicted": (
                    "capture frame binding was evicted by the bounded binding set"
                ),
                "replay_history_evicted": (
                    "capture replay history was evicted by its bounded storage "
                    "policy; capture a new head image and select the points on "
                    "that new image"
                ),
                "episode_changed": (
                    "evaluator episode changed after the frozen capture"
                ),
                "session_mismatch": (
                    "image_id belongs to a different policy session"
                ),
                "binding_not_registered": (
                    "capture frame binding was not registered"
                ),
            }
            raise TrackObjectDistanceBindingError(
                reason,
                messages.get(reason, "capture frame binding is unavailable"),
            )
        expected_shape = tuple(int(value) for value in capture_image_shape)
        if binding.observation_sequence != int(capture_observation_sequence):
            raise TrackObjectDistanceBindingError(
                "sequence_mismatch",
                "frozen capture observation sequence was modified",
            )
        if binding.episode_id != str(capture_episode_id):
            raise TrackObjectDistanceBindingError(
                "episode_mismatch",
                "frozen capture episode does not match its frame binding",
            )
        if binding.image_shape != expected_shape:
            raise TrackObjectDistanceBindingError(
                "image_shape_mismatch",
                "frozen capture image resolution was modified",
            )
        if binding.frame_digest != str(capture_frame_digest or "").strip():
            raise TrackObjectDistanceBindingError(
                "digest_mismatch",
                "frozen capture RGB-D or camera pose does not match its frame binding",
            )
        latest = self._latest_frame
        if latest is None:
            raise TrackObjectDistanceBindingError(
                "no_synchronized_frame",
                "no synchronized evaluator head RGB-D observation available",
            )
        if latest.episode_id != binding.episode_id:
            raise TrackObjectDistanceBindingError(
                "episode_changed",
                "evaluator episode changed after the frozen capture",
            )

        replay = [
            frame
            for frame in self._frame_history
            if frame.episode_id == binding.episode_id
            and frame.sequence >= binding.observation_sequence
            and frame.sequence <= latest.sequence
        ]
        if (
            not replay
            or replay[0].sequence != binding.observation_sequence
            or replay[-1].sequence != latest.sequence
        ):
            raise TrackObjectDistanceBindingError(
                "replay_history_evicted",
                "capture replay history is incomplete or stale; capture a new "
                "head image and select the points again",
            )
        for previous, current in zip(replay, replay[1:]):
            if current.sequence != previous.sequence + 1:
                raise TrackObjectDistanceBindingError(
                    "replay_gap",
                    "capture replay history has an observation gap; capture a new "
                    "head image and select the points again",
                )
        return binding, list(replay)

    def _apply_active_rigid_pair_locked(self) -> dict[str, Any]:
        pair = self._active_rigid_pair
        if not pair:
            self._active_rigid_pair_report = {}
            return {}
        names = tuple(str(name) for name in pair["names"])
        unavailable = [
            name
            for name in names
            if name not in self._entries
            or self._entries[name].get("status") != OBSERVED
        ]
        if unavailable:
            visual_report = (
                self._tracker.rigid_pair_report()
                if callable(getattr(self._tracker, "rigid_pair_report", None))
                else {}
            )
            report = {
                "ok": False,
                "reason": "rigid_pair_point_unobserved",
                "names": list(names),
                "unavailable": unavailable,
                "reference_distance_m": float(pair["reference_distance_m"]),
                "visual_tracker_pair": visual_report,
                "fusion_applied": False,
                "build": TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD,
            }
            self._active_rigid_pair_report = report
            return deepcopy(report)

        visual_report = (
            self._tracker.rigid_pair_report()
            if callable(getattr(self._tracker, "rigid_pair_report", None))
            else {}
        )
        if visual_report and not bool(visual_report.get("ok")):
            report = {
                "ok": False,
                "reason": str(
                    visual_report.get("reason")
                    or "visual_rigid_pair_candidate_rejected"
                ),
                "names": list(names),
                "reference_distance_m": float(pair["reference_distance_m"]),
                "observation_sequence": (
                    None
                    if self._latest_frame is None
                    else int(self._latest_frame.sequence)
                ),
                "visual_tracker_pair": deepcopy(visual_report),
                "fusion_applied": False,
                "raw_rgbd_measurement_published": False,
                "prediction_values_published_as_observation": False,
                "stale_depth_published": False,
                "stale_xyz_in_robot_base_coord_published": False,
                "build": TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD,
            }
            for name in names:
                entry = self._entries[name]
                entry.update(
                    {
                        "depth_m": None,
                        "u": None,
                        "v": None,
                        "xyz_in_robot_base_coord_m": None,
                        "status": "temporarily_unobserved",
                        "tracker_status": "candidate_rejected",
                        "depth_source": None,
                        "xyz_source": None,
                    }
                )
                entry.pop("raw_xyz_in_robot_base_coord_m", None)
                entry.pop("raw_xyz_source", None)
                entry["rigid_pair_fusion"] = deepcopy(report)
                self._temporarily_unobserved[name] = {
                    "name": name,
                    "status": "candidate_rejected",
                    "missed_steps": 1,
                    "observation_sequence": int(
                        entry.get("observation_sequence", -1)
                    ),
                    "motion_retention_active": self._motion_retention_active_locked(
                        name,
                        episode_id=str(entry.get("episode_id") or ""),
                    ),
                }
            self._active_rigid_pair_report = report
            return deepcopy(report)

        measured = np.asarray(
            [
                self._entries[name].get(
                    "raw_xyz_in_robot_base_coord_m",
                    self._entries[name]["xyz_in_robot_base_coord_m"],
                )
                for name in names
            ],
            dtype=np.float64,
        )
        report = project_registered_rigid_pair(
            measured,
            reference_distance_m=float(pair["reference_distance_m"]),
            max_point_correction_m=float(pair["max_point_correction_m"]),
        )
        report.update(
            {
                "names": list(names),
                "observation_sequence": (
                    None
                    if self._latest_frame is None
                    else int(self._latest_frame.sequence)
                ),
                "registration_observation_sequence": int(
                    pair["registration_observation_sequence"]
                ),
                "build": TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD,
                "observation_inputs": [
                    "current_evaluator_rgb",
                    "current_evaluator_depth_linear",
                    "current_evaluator_camera_relative_pose",
                    "policy_owned_registration_distance",
                ],
                "direct_simulator_mutation": False,
                "visual_tracker_pair": visual_report,
                "fusion_applied": bool(report.get("ok")),
            }
        )
        if not bool(report.get("ok")):
            for index, name in enumerate(names):
                entry = self._entries[name]
                if "raw_xyz_in_robot_base_coord_m" in entry:
                    entry["xyz_in_robot_base_coord_m"] = deepcopy(
                        entry["raw_xyz_in_robot_base_coord_m"]
                    )
                    entry["xyz_source"] = (
                        "current_evaluator_depth_linear_and_cam_rel_poses"
                    )
                entry["rigid_pair_fusion"] = {
                    key: deepcopy(value)
                    for key, value in report.items()
                    if key != "fused_points_robot_base_m"
                }
            self._active_rigid_pair_report = report
            return deepcopy(report)

        fused = np.asarray(
            report["fused_points_robot_base_m"], dtype=np.float64
        ).reshape(2, 3)
        public_report = {
            key: deepcopy(value)
            for key, value in report.items()
            if key != "fused_points_robot_base_m"
        }
        for index, name in enumerate(names):
            entry = self._entries[name]
            if "raw_xyz_in_robot_base_coord_m" not in entry:
                entry["raw_xyz_in_robot_base_coord_m"] = deepcopy(
                    entry["xyz_in_robot_base_coord_m"]
                )
                entry["raw_xyz_source"] = str(entry.get("xyz_source") or "")
            entry["xyz_in_robot_base_coord_m"] = fused[index].astype(float).tolist()
            entry["xyz_source"] = (
                "current_evaluator_rgbd_projected_to_policy_owned_"
                "registration_rigid_pair_distance"
            )
            entry["rigid_pair_fusion"] = deepcopy(public_report)
        self._active_rigid_pair_report = report
        return deepcopy(report)

    def activate_rigid_pair(
        self,
        names: list[str] | tuple[str, str],
        *,
        episode_id: str,
        max_point_correction_m: float = (
            TRACK_OBJECT_DISTANCE_RIGID_PAIR_MAX_POINT_CORRECTION_M
        ),
    ) -> dict[str, Any]:
        """Fuse two registered names using their frozen registration distance."""

        requested = tuple(str(name or "").strip() for name in names)
        if len(requested) != 2 or any(not name for name in requested):
            raise ValueError("rigid pair requires exactly two non-empty names")
        if requested[0] == requested[1]:
            raise ValueError("rigid pair names must be unique")
        expected_episode = str(episode_id or "").strip()
        if not expected_episode:
            raise ValueError("episode_id is required")
        maximum = float(max_point_correction_m)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_point_correction_m must be positive and finite")

        with self._lock:
            self.upgrade_runtime_state()
            latest = self._latest_frame
            if latest is None:
                raise ValueError("no synchronized evaluator frame is available")
            if latest.episode_id != expected_episode:
                raise ValueError("rigid pair episode changed")
            missing = [
                name for name in requested if name not in self._registration_entries
            ]
            if missing:
                raise ValueError(
                    "rigid-pair registration reference is unavailable for "
                    + ", ".join(missing)
                    + "; register the tracked points again"
                )
            reference_points = np.asarray(
                [
                    self._registration_entries[name][
                        "xyz_in_robot_base_coord_m"
                    ]
                    for name in requested
                ],
                dtype=np.float64,
            )
            reference_distance = float(
                np.linalg.norm(reference_points[1] - reference_points[0])
            )
            if reference_distance < TRACK_OBJECT_DISTANCE_RIGID_PAIR_MIN_DISTANCE_M:
                raise ValueError("registered rigid pair is collapsed")
            candidate_pair = {
                "names": list(requested),
                "reference_distance_m": reference_distance,
                "registration_observation_sequence": int(
                    self._registration_entries[requested[0]][
                        "observation_sequence"
                    ]
                ),
                "max_point_correction_m": maximum,
            }
            tracker_activation = self._activate_tracker_rigid_pair_locked(
                candidate_pair
            )
            self._active_rigid_pair = candidate_pair
            self._active_rigid_pair_report = {
                "ok": True,
                "tracker_activation": tracker_activation,
            }
            return self._apply_active_rigid_pair_locked()

    def deactivate_rigid_pair(
        self,
        *,
        reason: str = "rigid_pair_request_scope_ended",
    ) -> dict[str, Any]:
        """Retire pair-only tracker state without deleting registered points."""

        retirement_reason = str(reason or "rigid_pair_request_scope_ended")
        with self._lock:
            previous_before_upgrade = deepcopy(self._active_rigid_pair)
            upgrade = self.upgrade_runtime_state()
            active_after_upgrade = deepcopy(self._active_rigid_pair)
            previous = active_after_upgrade or previous_before_upgrade
            if active_after_upgrade is not None:
                retired = self._retire_active_rigid_pair_locked(
                    reason=retirement_reason,
                )
                entry_retirement = dict(retired.get("entry_retirement") or {})
            elif previous_before_upgrade is not None:
                # Runtime upgrade may already have retired an invalid old pair.
                # Reuse that transactional report instead of deactivating the
                # underlying tracker a second time.
                retired = deepcopy(upgrade.get("tracker_rigid_pair") or {})
                entry_retirement = dict(retired.get("entry_retirement") or {})
            else:
                deactivate = getattr(self._tracker, "deactivate_rigid_pair", None)
                if callable(deactivate):
                    deactivate()
                stale_entry_names = [
                    str(name)
                    for name, entry in self._entries.items()
                    if (
                        "raw_xyz_in_robot_base_coord_m" in entry
                        or "rigid_pair_fusion" in entry
                        or "registration_rigid_pair"
                        in str(entry.get("xyz_source") or "")
                    )
                ]
                entry_retirement = self._restore_raw_rigid_pair_entries_locked(
                    stale_entry_names
                )
                self._active_rigid_pair_report = {}
                self._active_on_hand_eef_prior = None
                retired = {}
            return {
                "ok": True,
                "active": False,
                "reason": retirement_reason,
                "previous_pair": previous,
                "retired_report": retired,
                "entry_retirement": entry_retirement,
            }

    def activate_on_hand_eef_observation_prior(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
        session_id: str,
        image_id: str,
        arm: str,
        anchors_eef_m: Any,
    ) -> dict[str, Any]:
        """Gate 1-6 current RGB-D candidates with explicit on-hand local FK."""

        requested = tuple(str(name or "").strip() for name in names)
        if (
            not 1 <= len(requested) <= 6
            or any(not name for name in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError(
                "on-hand EEF prior requires one to six unique point names"
            )
        selected_arm = str(arm or "").strip().lower()
        if selected_arm not in ("left", "right"):
            raise ValueError("on-hand EEF prior arm must be left or right")
        anchors = np.asarray(anchors_eef_m, dtype=np.float64)
        if anchors.shape != (len(requested), 3) or not np.all(
            np.isfinite(anchors)
        ):
            raise ValueError(
                "on-hand EEF anchors must be a finite point_count x 3 array"
            )
        expected_episode = str(episode_id or "").strip()
        expected_session = str(session_id or "").strip()
        expected_image = str(image_id or "").strip()
        if not expected_episode or not expected_session or not expected_image:
            raise ValueError("episode_id, session_id, and image_id are required")

        with self._lock:
            latest = self._latest_frame
            binding = self._registration_binding
            pair = self._active_rigid_pair
            if latest is None or latest.episode_id != expected_episode:
                raise ValueError("on-hand EEF prior episode changed")
            if binding is None or (
                binding.session_id != expected_session
                or binding.image_id != expected_image
                or binding.episode_id != expected_episode
            ):
                raise ValueError("on-hand EEF prior registration binding changed")
            exact_rigid_pair = bool(
                len(requested) == 2
                and pair is not None
                and tuple(pair["names"]) == requested
            )
            if len(requested) == 2 and not exact_rigid_pair:
                raise ValueError(
                    "on-hand EEF prior requires the same active rigid pair"
                )
            if len(requested) != 2 and pair is not None:
                raise ValueError(
                    "multi-point on-hand EEF prior requires rigid-pair tracking "
                    "to be inactive"
                )
            track_by_name = {
                name: track_id for track_id, name in self._name_by_track_id.items()
            }
            if any(name not in track_by_name for name in requested):
                raise ValueError("on-hand EEF prior track identity is unavailable")
            require_prior = None
            if exact_rigid_pair:
                require_prior = getattr(
                    self._tracker,
                    "require_active_rigid_pair_observation_prior",
                    None,
                )
                if not callable(require_prior):
                    raise RuntimeError(
                        "dynamic tracker does not expose an on-hand observation gate"
                    )
            self._on_hand_eef_prior_counter += 1
            lease_id = f"on-hand-eef-prior-{self._on_hand_eef_prior_counter:08d}"
            tracker_config = getattr(self._tracker, "config", None)
            candidate = {
                "lease_id": lease_id,
                "names": list(requested),
                "track_ids": [track_by_name[name] for name in requested],
                "episode_id": expected_episode,
                "session_id": expected_session,
                "image_id": expected_image,
                "arm": selected_arm,
                "anchors_eef_m": anchors.astype(float).tolist(),
                "activation_observation_sequence": int(latest.sequence),
                "source": "submission_local_fk_from_current_evaluator_proprio",
                "prediction_values_published_as_observation": False,
                "tracking_mode": (
                    "rigid_pair_joint_eef_candidate_gate"
                    if exact_rigid_pair
                    else "independent_npoint_eef_candidate_gates"
                ),
                "max_pixel_error_px": float(
                    getattr(tracker_config, "rigid_pair_prior_max_pixel_error_px", 24.0)
                ),
                "max_depth_error_m": float(
                    getattr(tracker_config, "rigid_pair_prior_max_depth_error_m", 0.025)
                ),
                "max_point_error_m": float(
                    getattr(tracker_config, "rigid_pair_prior_max_point_error_m", 0.030)
                ),
            }
            if require_prior is not None:
                require_prior(tuple(candidate["track_ids"]), required=True)
            self._active_on_hand_eef_prior = candidate
            return {"ok": True, **deepcopy(candidate)}

    def deactivate_on_hand_eef_observation_prior(
        self,
        lease_id: str,
    ) -> dict[str, Any]:
        token = str(lease_id or "").strip()
        if not token:
            raise ValueError("on-hand EEF prior lease_id is required")
        with self._lock:
            active = self._active_on_hand_eef_prior
            if active is None or str(active["lease_id"]) != token:
                return {"ok": False, "released": False, "lease_id": token}
            require_prior = getattr(
                self._tracker,
                "require_active_rigid_pair_observation_prior",
                None,
            )
            if (
                callable(require_prior)
                and len(active.get("track_ids", ())) == 2
                and self._active_rigid_pair is not None
                and tuple(self._active_rigid_pair.get("names", ()))
                == tuple(active.get("names", ()))
            ):
                require_prior(tuple(active["track_ids"]), required=False)
            self._active_on_hand_eef_prior = None
            return {"ok": True, "released": True, "lease_id": token}

    def _on_hand_eef_prior_for_frame_locked(
        self,
        frame: TrackerFrame,
    ) -> dict[str, Any] | None:
        active = self._active_on_hand_eef_prior
        if active is None:
            return None
        if str(frame.episode_id) != str(active["episode_id"]):
            return None
        if frame.proprio is None:
            return None
        proprio = np.asarray(frame.proprio, dtype=np.float64).reshape(-1)
        if proprio.size != PROPRIO_DIM or not np.all(np.isfinite(proprio)):
            return None
        try:
            left_q = proprio[PROPRIO_SLICES["arm_left_qpos"]]
            right_q = proprio[PROPRIO_SLICES["arm_right_qpos"]]
            trunk_q = proprio[PROPRIO_SLICES["trunk_qpos"]]
            left_gripper = proprio[PROPRIO_SLICES["gripper_left_qpos"]]
            right_gripper = proprio[PROPRIO_SLICES["gripper_right_qpos"]]
            state = local_robot_state(
                trunk_q=trunk_q,
                arm_left_q=left_q,
                arm_right_q=right_q,
                gripper_left_q=left_gripper,
                gripper_right_q=right_gripper,
            )
            arm = str(active["arm"])
            selected_q = left_q if arm == "left" else right_q
            position, quaternion = eef_pose(state, arm, selected_q)
            anchors = np.asarray(active["anchors_eef_m"], dtype=np.float64)
            points = np.asarray(position, dtype=np.float64)[None, :] + (
                quat_to_mat_xyzw(quaternion) @ anchors.T
            ).T
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        if points.shape != (len(active["track_ids"]), 3) or not np.all(
            np.isfinite(points)
        ):
            return None
        return {
            "track_ids": list(active["track_ids"]),
            "observation_sequence": int(frame.sequence),
            "episode_id": str(frame.episode_id),
            "source": "submission_local_fk_from_current_evaluator_proprio",
            "points_robot_base_m": points.astype(float).tolist(),
        }

    def ingest(
        self,
        frame: TrackerFrame,
        *,
        copy_frame: bool = True,
    ) -> dict[str, Any]:
        """Publish current depths and retire only confirmed off-screen/lost tracks.

        ``copy_frame=False`` is an internal ownership fast path for the
        official adapter.  That adapter replaces its complete snapshot on
        every observation, so retaining the borrowed arrays is safe while the
        tracker and replay workers consume them.  The default remains a full
        defensive copy for every existing caller.
        """

        if not isinstance(copy_frame, bool):
            raise TypeError("copy_frame must be a bool")
        frozen = frame if not copy_frame else _copy_frame(frame)
        self._apply_replay_backpressure(frozen)
        with self._lock:
            self._drain_ready_replay_locked()
            previous_episode = (
                None if self._latest_frame is None else self._latest_frame.episode_id
            )
            if previous_episode is not None and previous_episode != frozen.episode_id:
                self._last_deleted = [
                    {
                        "name": name,
                        "reason": "episode_changed",
                        "observation_sequence": int(frozen.sequence),
                    }
                    for name in self._name_by_track_id.values()
                ]
                self._name_by_track_id.clear()
                self._entries.clear()
                self._registration_entries.clear()
                self._active_rigid_pair = None
                self._active_rigid_pair_report = {}
                self._active_on_hand_eef_prior = None
                self._temporarily_unobserved.clear()
                self._motion_retention_leases.clear()
                for key in list(self._capture_bindings):
                    self._remove_binding_locked(key, "episode_changed")
                self._clear_replay_history_locked()
                self._discard_pending_replay_locked()
                self._tracker = self._tracker_factory()
                self._registration_binding = None
                self._latest_frame = frozen
                self._last_error = None
                return self.status()

            if self._capture_bindings:
                # Compression is ordered but no longer joined by the action
                # path. replace_points() provides the exact replay barrier.
                self._schedule_replay_locked(frozen)
            try:
                if self._name_by_track_id:
                    observation_prior = self._on_hand_eef_prior_for_frame_locked(
                        frozen
                    )
                    snapshots = (
                        self._tracker.ingest(
                            frozen,
                            rigid_pair_observation_prior=observation_prior,
                        )
                        if self._active_on_hand_eef_prior is not None
                        else self._tracker.ingest(frozen)
                    )
                else:
                    snapshots = ()
            except (TypeError, ValueError) as exc:
                self._invalidate_replay_locked(
                    "replay_history_inconsistent",
                    f"{type(exc).__name__}: {exc}",
                )
                return self.status()

            self._latest_frame = frozen
            self._last_error = None
            self._purge_motion_retention_locked(episode_id=frozen.episode_id)
            snapshots_by_id = {snapshot.track_id: snapshot for snapshot in snapshots}
            deleted_this_frame: list[dict[str, Any]] = []
            for track_id, name in list(self._name_by_track_id.items()):
                snapshot = snapshots_by_id.get(track_id)
                if snapshot is None or snapshot.status != OBSERVED:
                    status = "missing" if snapshot is None else snapshot.status
                    predicted_uv = None if snapshot is None else snapshot.predicted_uv
                    permanently_lost = snapshot is None or status == "lost"
                    out_of_view = snapshot is not None and predicted_uv is None
                    retained_for_motion = bool(
                        self._motion_retention_active_locked(
                            name,
                            episode_id=frozen.episode_id,
                        )
                    )
                    if (permanently_lost or out_of_view) and not retained_for_motion:
                        deleted_this_frame.append(
                            {
                                "name": name,
                                "reason": (
                                    "out_of_view" if out_of_view else "tracking_lost"
                                ),
                                "tracker_status": status,
                                "observation_sequence": int(frozen.sequence),
                            }
                        )
                        self._tracker.remove_point(track_id)
                        self._name_by_track_id.pop(track_id, None)
                        self._temporarily_unobserved.pop(name, None)
                        self._entries.pop(name, None)
                        self._registration_entries.pop(name, None)
                    else:
                        previous_entry = self._entries.get(name)
                        unobserved_entry = (
                            _unobserved_memory_entry(snapshot, name)
                            if snapshot is not None
                            else _missing_tracker_memory_entry(
                                name=name,
                                track_id=track_id,
                                previous=previous_entry,
                                observation_sequence=int(frozen.sequence),
                                episode_id=frozen.episode_id,
                            )
                        )
                        self._entries[name] = _entry_with_binding(
                            unobserved_entry, self._registration_binding
                        )
                        self._temporarily_unobserved[name] = {
                            "name": name,
                            "status": status,
                            "missed_steps": int(
                                snapshot.missed_steps
                                if snapshot is not None
                                else int(
                                    (self._temporarily_unobserved.get(name) or {}).get(
                                        "missed_steps", 0
                                    )
                                )
                                + 1
                            ),
                            "observation_sequence": int(frozen.sequence),
                            "motion_retention_active": retained_for_motion,
                        }
                    continue
                try:
                    self._entries[name] = _entry_with_binding(
                        _memory_entry(snapshot),
                        self._registration_binding,
                    )
                    self._temporarily_unobserved.pop(name, None)
                except ValueError:
                    deleted_this_frame.append(
                        {
                            "name": name,
                            "reason": "invalid_current_depth",
                            "observation_sequence": int(frozen.sequence),
                        }
                    )
                    self._tracker.remove_point(track_id)
                    self._name_by_track_id.pop(track_id, None)
                    self._entries.pop(name, None)
                    self._registration_entries.pop(name, None)
                    self._temporarily_unobserved.pop(name, None)
            if deleted_this_frame:
                self._last_deleted = deleted_this_frame
            retired_pair_this_frame = False
            if self._active_rigid_pair is not None:
                pair_names = tuple(
                    str(name) for name in self._active_rigid_pair.get("names", ())
                )
                registered_names = {
                    str(name) for name in self._name_by_track_id.values()
                }
                missing_pair_names = [
                    name for name in pair_names if name not in registered_names
                ]
                if missing_pair_names:
                    self._retire_active_rigid_pair_locked(
                        reason="rigid_pair_track_removed",
                        unavailable=missing_pair_names,
                    )
                    retired_pair_this_frame = True
            if not self._name_by_track_id:
                self._motion_retention_leases.clear()
                self._registration_binding = None
                self._registration_entries.clear()
                self._active_rigid_pair = None
                if not retired_pair_this_frame:
                    self._active_rigid_pair_report = {}
                self._active_on_hand_eef_prior = None
            elif self._active_rigid_pair is not None:
                self._apply_active_rigid_pair_locked()
            return self.status()

    def replace_points(
        self,
        points: list[dict[str, Any]],
        *,
        session_id: str,
        image_id: str,
        capture_observation_sequence: int,
        capture_episode_id: str,
        capture_image_shape: tuple[int, int],
        capture_frame_digest: str,
        check_cancelled: Callable[[], None] | None = None,
        commit_guard: Callable[[], Any] | None = None,
        commit_result: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Replace the complete named set on the frozen image and replay it.

        A successful call commits one isolated candidate and drops every
        previous name; this operation never merges a new selection with old
        points.  Runtime compatibility repair is deliberately performed only
        after request-shape validation so an obsolete pair cannot mask a bad
        new request.
        """

        request = validate_track_object_distance_args(
            {
                "session_id": session_id,
                "image_id": image_id,
                "points": points,
            }
        )
        if check_cancelled is not None and not callable(check_cancelled):
            raise TypeError("check_cancelled must be callable")
        if commit_guard is not None and not callable(commit_guard):
            raise TypeError("commit_guard must be callable")
        if commit_result is not None and not callable(commit_result):
            raise TypeError("commit_result must be callable")
        # Registration is also used by the normal UI endpoint, so it must not
        # depend on a development reload endpoint having run beforehand.  The
        # upgrade path now retires incomplete legacy pairs instead of raising
        # before this request can build its replacement candidate.
        self.upgrade_runtime_state()
        if check_cancelled is not None:
            check_cancelled()
        with self._lock:
            latest_sequence = (
                None
                if self._latest_frame is None
                else int(self._latest_frame.sequence)
            )
        if latest_sequence is not None:
            self._flush_replay_through(
                latest_sequence,
                check_cancelled=check_cancelled,
            )
        with self._lock:
            binding, replay_frames = self._bound_replay_frames_locked(
                session_id=request["session_id"],
                image_id=request["image_id"],
                capture_observation_sequence=capture_observation_sequence,
                capture_episode_id=capture_episode_id,
                capture_image_shape=capture_image_shape,
                capture_frame_digest=capture_frame_digest,
            )
            replay_work = len(request["points"]) * max(
                1,
                len(replay_frames),
            )
            if replay_work > TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK:
                raise TrackObjectDistanceBindingError(
                    "replay_work_exceeded",
                    "capture replay workload is too large; capture a new head "
                    "image and select the points again",
                )

            candidate = self._tracker_factory()
            if check_cancelled is not None:
                check_cancelled()
            candidate.ingest(_restore_replay_frame(replay_frames[0]))
            candidate_names: dict[str, str] = {}
            for index, point in enumerate(request["points"], start=1):
                snapshot = candidate.mark_point(
                    u=point["u"],
                    v=point["v"],
                    label=point["name"],
                    track_id=f"distance_point_{index:03d}",
                )
                candidate_names[snapshot.track_id] = point["name"]

            candidate_registration_entries: dict[str, dict[str, Any]] = {}
            for track_id, name in candidate_names.items():
                registration_snapshot = candidate.get_point(track_id)
                candidate_registration_entries[name] = _entry_with_binding(
                    _memory_entry(registration_snapshot),
                    binding,
                )

            for stored_replay_frame in replay_frames[1:]:
                if check_cancelled is not None:
                    check_cancelled()
                replay_frame = _restore_replay_frame(stored_replay_frame)
                snapshots = {
                    snapshot.track_id: snapshot
                    for snapshot in candidate.ingest(replay_frame)
                }
                for track_id, name in candidate_names.items():
                    snapshot = snapshots.get(track_id)
                    if snapshot is None:
                        raise ValueError(
                            f"point {name!r} disappeared during frozen-frame replay"
                        )
                    if snapshot.status == "lost" or (
                        snapshot.status != OBSERVED
                        and snapshot.predicted_uv is None
                    ):
                        raise ValueError(
                            f"point {name!r} left view or tracking was lost during "
                            "frozen-frame replay"
                        )

            candidate_entries: dict[str, dict[str, Any]] = {}
            for track_id, name in candidate_names.items():
                snapshot = candidate.get_point(track_id)
                if snapshot.status != OBSERVED:
                    raise ValueError(
                        f"point {name!r} is not observed in the current evaluator frame"
                    )
                candidate_entries[name] = _entry_with_binding(
                    _memory_entry(snapshot),
                    binding,
                )

            latest = replay_frames[-1]
            replaced_names = list(self._name_by_track_id.values())
            committed = {
                "entries": deepcopy(candidate_entries),
                "replaced_names": replaced_names,
                "session_id": binding.session_id,
                "image_id": binding.image_id,
                "capture_observation_sequence": int(
                    binding.observation_sequence
                ),
                "observation_sequence": int(latest.sequence),
                "replayed_observation_count": len(replay_frames) - 1,
                "episode_id": latest.episode_id,
                "frame_binding": (
                    "policy_owned_frozen_head_capture_then_ordered_live_replay"
                ),
                "frame_digest": binding.frame_digest,
                "registration_entries": deepcopy(
                    candidate_registration_entries
                ),
            }
            guard = commit_guard() if commit_guard is not None else nullcontext(True)
            with guard as commit_allowed:
                if commit_allowed is False:
                    raise ValueError(
                        "track_object_distance request was cancelled before commit"
                    )
                if commit_guard is None and check_cancelled is not None:
                    check_cancelled()
                previous_state = (
                    self._tracker,
                    self._name_by_track_id,
                    self._entries,
                    self._registration_entries,
                    self._active_rigid_pair,
                    self._active_rigid_pair_report,
                    self._active_on_hand_eef_prior,
                    self._temporarily_unobserved,
                    self._motion_retention_leases,
                    self._registration_binding,
                    self._last_deleted,
                    self._last_error,
                )
                try:
                    self._tracker = candidate
                    self._name_by_track_id = candidate_names
                    self._entries = candidate_entries
                    self._registration_entries = candidate_registration_entries
                    self._active_rigid_pair = None
                    self._active_rigid_pair_report = {}
                    self._active_on_hand_eef_prior = None
                    self._temporarily_unobserved = {}
                    self._motion_retention_leases = {}
                    self._registration_binding = binding
                    self._last_deleted = []
                    self._last_error = None
                    if commit_result is not None:
                        commit_result(deepcopy(committed))
                except Exception:
                    (
                        self._tracker,
                        self._name_by_track_id,
                        self._entries,
                        self._registration_entries,
                        self._active_rigid_pair,
                        self._active_rigid_pair_report,
                        self._active_on_hand_eef_prior,
                        self._temporarily_unobserved,
                        self._motion_retention_leases,
                        self._registration_binding,
                        self._last_deleted,
                        self._last_error,
                    ) = previous_state
                    raise
            return committed

    def observed_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        session_id: str,
        image_id: str,
        episode_id: str,
    ) -> dict[str, Any]:
        """Return an atomic same-observation snapshot for a live controller.

        Predicted or stale entries are deliberately excluded.  Consumers such
        as ``cut_object`` must stop instead of steering from an older depth.
        """

        requested = [str(name or "").strip() for name in names]
        if not requested or any(not name for name in requested):
            raise ValueError("names must contain non-empty point names")
        if len(set(requested)) != len(requested):
            raise ValueError("names must be unique")
        expected_session = str(session_id or "").strip()
        expected_image = str(image_id or "").strip()
        expected_episode = str(episode_id or "").strip()
        if not expected_session or not expected_image or not expected_episode:
            raise ValueError("session_id, image_id, and episode_id are required")

        with self._lock:
            latest = self._latest_frame
            binding = self._registration_binding
            unavailable: dict[str, str] = {}
            reason: str | None = None
            if latest is None:
                reason = "no_synchronized_frame"
            elif latest.episode_id != expected_episode:
                reason = "episode_changed"
            elif binding is None:
                reason = "tracking_not_registered"
            elif (
                binding.session_id != expected_session
                or binding.image_id != expected_image
            ):
                reason = "tracking_replaced"
            elif binding.episode_id != expected_episode:
                reason = "episode_changed"

            entries: dict[str, dict[str, Any]] = {}
            if reason is None and latest is not None:
                for name in requested:
                    entry = self._entries.get(name)
                    if entry is None:
                        unavailable[name] = "point_not_registered_or_retired"
                        continue
                    status = str(entry.get("status") or "")
                    if status != OBSERVED:
                        unavailable[name] = "point_unobserved"
                        continue
                    if int(entry.get("observation_sequence", -1)) != int(
                        latest.sequence
                    ):
                        unavailable[name] = "stale_tracking_measurement"
                        continue
                    try:
                        xyz = np.asarray(
                            entry["xyz_in_robot_base_coord_m"],
                            dtype=np.float64,
                        ).reshape(3)
                        depth_m = float(entry["depth_m"])
                    except (KeyError, TypeError, ValueError):
                        unavailable[name] = "invalid_tracking_measurement"
                        continue
                    if (
                        not np.all(np.isfinite(xyz))
                        or not math.isfinite(depth_m)
                        or depth_m <= 0.0
                    ):
                        unavailable[name] = "invalid_tracking_measurement"
                        continue
                    entries[name] = deepcopy(entry)
                if unavailable:
                    reason = "tracked_point_unavailable"

            rigid_report: dict[str, Any] = {}
            active_pair = self._active_rigid_pair
            if active_pair is not None:
                active_names = set(active_pair["names"])
                if active_names.issubset(set(requested)):
                    rigid_report = deepcopy(self._active_rigid_pair_report)
                    if reason is None and not bool(rigid_report.get("ok")):
                        reason = str(
                            rigid_report.get("reason")
                            or "rigid_pair_measurement_invalid"
                        )

            return {
                "ok": reason is None,
                "reason": reason,
                "entries": entries,
                "unavailable": unavailable,
                "requested_names": requested,
                "observation_sequence": (
                    None if latest is None else int(latest.sequence)
                ),
                "episode_id": None if latest is None else latest.episode_id,
                "session_id": (
                    None if binding is None else binding.session_id
                ),
                "image_id": None if binding is None else binding.image_id,
                "frame": "current_robot_base",
                "axes": "x_forward_y_left_z_up",
                "measurement_source": (
                    "current_evaluator_rgbd_and_cam_rel_poses_with_"
                    "registration_rigid_pair_projection"
                    if rigid_report.get("ok")
                    else "current_evaluator_depth_linear_and_cam_rel_poses"
                ),
                "rigid_pair_fusion": rigid_report,
                "predictions_published": False,
            }

    def registered_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        """Return immutable registration-time RGB-D points for local FK gating.

        Registration entries are policy-owned observations captured by
        ``replace_points``.  They are intentionally separate from the live
        tracker entries, which may be temporarily mis-associated when a rigid
        pair first switches from independent to joint tracking.
        """

        requested = [str(name or "").strip() for name in names]
        if not requested or any(not name for name in requested):
            raise ValueError("names must contain non-empty point names")
        if len(set(requested)) != len(requested):
            raise ValueError("names must be unique")
        expected_episode = str(episode_id or "").strip()
        if not expected_episode:
            raise ValueError("episode_id is required")

        with self._lock:
            self.upgrade_runtime_state()
            binding = self._registration_binding
            latest = self._latest_frame
            reason: str | None = None
            if binding is None:
                reason = "tracking_not_registered"
            elif binding.episode_id != expected_episode:
                reason = "episode_changed"
            elif latest is not None and latest.episode_id != expected_episode:
                reason = "episode_changed"

            entries: dict[str, dict[str, Any]] = {}
            unavailable: dict[str, str] = {}
            if reason is None and binding is not None:
                for name in requested:
                    entry = self._registration_entries.get(name)
                    if not isinstance(entry, Mapping):
                        unavailable[name] = "registration_point_unavailable"
                        continue
                    try:
                        xyz = np.asarray(
                            entry["xyz_in_robot_base_coord_m"],
                            dtype=np.float64,
                        ).reshape(3)
                        entry_sequence = int(entry["observation_sequence"])
                    except (KeyError, TypeError, ValueError):
                        unavailable[name] = "invalid_registration_point"
                        continue
                    if (
                        not np.all(np.isfinite(xyz))
                        or entry_sequence != int(binding.observation_sequence)
                        or str(entry.get("source_session_id") or "")
                        != str(binding.session_id)
                        or str(entry.get("source_image_id") or "")
                        != str(binding.image_id)
                    ):
                        unavailable[name] = "invalid_registration_point"
                        continue
                    entries[name] = deepcopy(dict(entry))
                if unavailable:
                    reason = "registration_point_unavailable"

            return {
                "ok": reason is None,
                "reason": reason,
                "entries": entries,
                "unavailable": unavailable,
                "requested_names": requested,
                "session_id": None if binding is None else binding.session_id,
                "image_id": None if binding is None else binding.image_id,
                "episode_id": None if binding is None else binding.episode_id,
                "registration_observation_sequence": (
                    None
                    if binding is None
                    else int(binding.observation_sequence)
                ),
                "image_shape": (
                    None if binding is None else list(binding.image_shape)
                ),
                "frame_digest": None if binding is None else binding.frame_digest,
                "measurement_source": "policy_owned_frozen_head_rgbd_registration",
                "predictions_published": False,
            }

    def observed_active_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        """Read named points from the active registration without rebinding it."""

        with self._lock:
            binding = self._registration_binding
            if binding is None:
                latest = self._latest_frame
                return {
                    "ok": False,
                    "reason": "tracking_not_registered",
                    "entries": {},
                    "unavailable": {},
                    "requested_names": [str(name) for name in names],
                    "observation_sequence": (
                        None if latest is None else int(latest.sequence)
                    ),
                    "episode_id": None if latest is None else latest.episode_id,
                    "session_id": None,
                    "image_id": None,
                    "frame": "current_robot_base",
                    "axes": "x_forward_y_left_z_up",
                    "measurement_source": (
                        "current_evaluator_depth_linear_and_cam_rel_poses"
                    ),
                    "predictions_published": False,
                }
            session_id = str(binding.session_id)
            image_id = str(binding.image_id)
        return self.observed_points_snapshot(
            names,
            session_id=session_id,
            image_id=image_id,
            episode_id=episode_id,
        )

    def memory_fields(self) -> dict[str, Any]:
        with self._lock:
            return {
                TRACKED_OBJECT_DISTANCES_MEMORY_KEY: deepcopy(self._entries),
                TRACKING_POLICY_MEMORY_KEY: {
                    "camera": "head",
                    "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
                    "depth_unit": "m",
                    "depth_source": "current_evaluator_depth_linear",
                    "xyz_in_robot_base_coord_unit": "m",
                    "xyz_in_robot_base_coord_frame": "current_robot_base",
                    "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                    "xyz_in_robot_base_coord_source": (
                        "current_evaluator_depth_linear_and_cam_rel_poses"
                    ),
                    "identity_source": "model_annotation",
                    "identity_verified": False,
                    "initial_grounding": "policy_owned_frozen_head_capture",
                    "frame_alignment": "required_session_id_and_image_id",
                    "replay_policy": "strict_consecutive_evaluator_observations",
                    "replay_storage": (
                        "lossless_zlib_gray_and_float32_depth_with_byte_budget"
                    ),
                    "deletion_policy": (
                        "remove_measurement_when_unobserved_and_retire_track_"
                        "when_out_of_view_or_lost"
                    ),
                    "stale_depth_published": False,
                    "stale_xyz_in_robot_base_coord_published": False,
                    "rigid_pair_fusion_build": (
                        TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD
                    ),
                    "active_rigid_pair": deepcopy(self._active_rigid_pair),
                    "active_rigid_pair_report": deepcopy(
                        self._active_rigid_pair_report
                    ),
                    "active_on_hand_eef_observation_prior": deepcopy(
                        self._active_on_hand_eef_prior
                    ),
                    "observation_contract": [
                        "*::rgb",
                        "*::depth_linear",
                        "*::cam_rel_poses",
                        "*::proprio",
                    ],
                    "direct_simulator_mutation": False,
                },
            }

    def note_observation_error(self, error: Exception | str) -> None:
        """Record a modality/shape failure without deleting valid live entries."""

        with self._lock:
            self._last_error = (
                str(error)
                if isinstance(error, str)
                else f"{type(error).__name__}: {error}"
            )

    def summary_fields(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_object_distances": sum(
                    entry.get("status") == OBSERVED
                    for entry in self._entries.values()
                ),
                "tracked_object_distance_names": list(self._entries),
                "tracked_object_distance_registrations": len(
                    self._name_by_track_id
                ),
            }

    def text(self) -> str:
        with self._lock:
            if not self._entries:
                return "Tracked object distances: none."
            lines = [
                "Tracked object distances and XYZ in current robot base coord "
                "(live head RGB-D):"
            ]
            for name, entry in self._entries.items():
                if entry.get("status") == OBSERVED:
                    x, y, z = entry["xyz_in_robot_base_coord_m"]
                    lines.append(
                        f"- {name}: depth={entry['depth_m']:.4f} m, "
                        f"uv=({entry['u']:.1f}, {entry['v']:.1f}), "
                        "xyz_in_robot_base_coord="
                        f"({x:.4f}, {y:.4f}, {z:.4f}) m"
                    )
                else:
                    lines.append(
                        f"- {name}: depth and xyz_in_robot_base_coord unavailable "
                        "(temporarily unobserved)"
                    )
            return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._drain_ready_replay_locked()
            latest_episode = (
                None if self._latest_frame is None else self._latest_frame.episode_id
            )
            self._purge_motion_retention_locked(episode_id=latest_episode)
            retained_names = sorted(
                {
                    str(name)
                    for lease in self._motion_retention_leases.values()
                    for name in lease["names"]
                }
            )
            return {
                "active": sum(
                    entry.get("status") == OBSERVED
                    for entry in self._entries.values()
                ),
                "names": list(self._entries),
                "registered": len(self._name_by_track_id),
                "registered_names": list(self._name_by_track_id.values()),
                "temporarily_unobserved": deepcopy(
                    self._temporarily_unobserved
                ),
                "motion_retention": {
                    "active_leases": len(self._motion_retention_leases),
                    "retained_names": retained_names,
                    "coordinates_published_while_unobserved": False,
                },
                "observation_sequence": (
                    None
                    if self._latest_frame is None
                    else int(self._latest_frame.sequence)
                ),
                "episode_id": (
                    None
                    if self._latest_frame is None
                    else self._latest_frame.episode_id
                ),
                "last_deleted": deepcopy(self._last_deleted),
                "last_error": self._last_error,
                "capture_bindings": len(self._capture_bindings),
                "capture_bindings_by_session": {
                    session: sum(
                        candidate_session == session
                        for candidate_session, _ in self._capture_bindings
                    )
                    for session in dict.fromkeys(
                        candidate_session
                        for candidate_session, _ in self._capture_bindings
                    )
                },
                "binding_failure_tombstones": len(self._binding_failures),
                "replay_history_frames": len(self._frame_history),
                "replay_history_bytes": int(self._replay_history_bytes),
                "replay_history_max_frames": int(self._replay_max_frames),
                "replay_history_max_bytes": int(self._replay_max_bytes),
                "replay_work_max_point_frames": TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK,
                **self._replay_archive.status(),
                "replay_history_encoding": (
                    "lossless_zlib_gray_and_float32_depth"
                ),
                "replay_async": True,
                "replay_compression_workers": _replay_executor_worker_count(),
                "replay_pending_frames": len(self._pending_replay),
                "replay_pending_bytes": int(self._pending_replay_bytes),
                "replay_pending_max_frames": min(
                    self._replay_max_frames,
                    TRACK_OBJECT_DISTANCE_REPLAY_MAX_PENDING_FRAMES,
                ),
                "replay_async_submitted": int(self._replay_async_submitted),
                "replay_async_committed": int(self._replay_async_committed),
                "replay_async_discarded": int(self._replay_async_discarded),
                "idle_tracker_fast_path": not bool(self._name_by_track_id),
                "tracker_acceleration": (
                    self._tracker.acceleration_status()
                    if callable(
                        getattr(self._tracker, "acceleration_status", None)
                    )
                    else {"backend": "cpu", "state": "unavailable"}
                ),
                "replay_history_oldest_sequence": (
                    None
                    if not self._frame_history
                    else int(self._frame_history[0].sequence)
                ),
                "replay_history_newest_sequence": (
                    None
                    if not self._frame_history
                    else int(self._frame_history[-1].sequence)
                ),
                "source_session_id": (
                    None
                    if self._registration_binding is None
                    else self._registration_binding.session_id
                ),
                "source_image_id": (
                    None
                    if self._registration_binding is None
                    else self._registration_binding.image_id
                ),
                "registration_reference_names": list(
                    self._registration_entries
                ),
                "active_rigid_pair": deepcopy(self._active_rigid_pair),
                "active_rigid_pair_report": deepcopy(
                    self._active_rigid_pair_report
                ),
                "active_on_hand_eef_observation_prior": deepcopy(
                    self._active_on_hand_eef_prior
                ),
                "deletion_policy": (
                    "remove_measurement_when_unobserved_and_retire_track_"
                    "when_out_of_view_or_lost"
                ),
            }


__all__ = [
    "TRACKED_OBJECT_DISTANCES_MEMORY_KEY",
    "TRACKING_POLICY_MEMORY_KEY",
    "TRACK_OBJECT_DISTANCE_REPLAY_MAX_BYTES",
    "TRACK_OBJECT_DISTANCE_REPLAY_MEMORY_MAX_BYTES",
    "TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S",
    "TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES",
    "TRACK_OBJECT_DISTANCE_MAX_CAPTURE_BINDINGS",
    "TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK",
    "TRACK_OBJECT_DISTANCE_RIGID_PAIR_FUSION_BUILD",
    "TRACK_OBJECT_DISTANCE_RIGID_PAIR_MAX_POINT_CORRECTION_M",
    "TrackObjectDistanceBindingError",
    "TrackedObjectDistanceMemory",
    "head_camera_intrinsics",
    "tracker_frame_content_digest",
    "tracker_frame_from_allowed_observation",
    "tracker_rgb_as_uint8",
    "project_registered_rigid_pair",
]
