"""Live adapter for the independent evaluator-safe RTAB-Map backend.

The simulation thread only snapshots allowed observations and queues them.
All native RTAB-Map work runs on one background thread so a slow registration
cannot stretch a physics step.  Per-tick body-frame qvel is never discarded:
when the bounded image queue is full, ticks remain pending and are attached to
the next camera frame that can be accepted.
"""

from __future__ import annotations

import atexit
import base64
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image

from behavior_interface.coordinate_contract import relative_to_pixel

from .client import (
    RtabmapClient,
    WorkerRequestTimeout,
    default_worker_path,
    worker_request_timeout_s,
)
from .official import (
    CameraIntrinsics,
    CameraRelativePose,
    BodyOdometry,
    OfficialObservation,
    SE2Pose,
    align_odometry_path_to_endpoint,
)
from .protocol import MapResult, PoseRecord, VERSION as PROTOCOL_VERSION
from .render import render_heading_up


BACKEND = "behavior_interface.rtabmap_slam.LiveMapper"


def _worker_artifact_build_id(
    worker_path: Optional[os.PathLike[str] | str] = None,
) -> str:
    """Identify the exact native artifact selected when this module loads."""

    prefix = f"rtabmap_querysubmap_protocol{PROTOCOL_VERSION}"
    selected = Path(worker_path) if worker_path is not None else default_worker_path()
    digest = hashlib.sha256()
    try:
        resolved = selected.expanduser().resolve(strict=True)
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except (OSError, RuntimeError):
        return f"{prefix}_worker_unavailable"
    return f"{prefix}_worker_{digest.hexdigest()[:12]}"


BUILD = _worker_artifact_build_id()
ENV_BACKEND = "BEHAVIOR_SPATIAL_MAP_BACKEND"
BACKEND_NAME = "rtabmap"

AUTO_START_NAME = "start"
AUTO_START_SOURCE = "automatic_initial_observation"

FRAME_INTERVAL_S = 0.20
FRAME_QUEUE_SIZE = 6
MAX_FAILURES = 3
MAX_MARK_MAP_SOURCE_AGE_S = 5.0
PRODUCER_POSE_HISTORY_SIZE = 8192
DATABASE_STAT_INTERVAL_S = 5.0
WORKER_STALL_WARNING_S = 10.0
DEFAULT_MAP_SIZE_PX = 490
DEFAULT_MAP_SPAN_M: Optional[float] = None
HEAD_MINIMAP_FRACTION = 0.32
HEAD_MINIMAP_MARGIN_FRAC = 0.016

# A minimap is a raster diagnostic, not the navigation source of truth.  A
# long-running episode can accumulate hundreds of thousands of qvel samples in
# ``_trail``; feeding every sample to PIL on the Flask thread makes a read-only
# image request pause the evaluator.  Keep the full anchored trail for map
# export/navigation and bound only the points copied into a display render.
try:
    _display_trail_limit = int(
        os.environ.get("BEHAVIOR_RTABMAP_DISPLAY_TRAIL_MAX_POINTS", "8192")
    )
except (TypeError, ValueError):
    _display_trail_limit = 8192
DISPLAY_TRAIL_MAX_POINTS = max(1024, min(_display_trail_limit, 65536))

_STOP = object()


def live_backend_selected() -> bool:
    value = str(os.environ.get(ENV_BACKEND, "") or "").strip().lower()
    return value in {"rtabmap", "rtab-map", "rtabmap_native", "native_rtabmap"}


def _navigation_route_snapshot(session_id: str):
    """Read an optional transient route; RTAB state remains backend-owned."""

    try:
        from behavior_interface_eval_test.navigation_route_overlay import (
            get_route_for_display,
        )

        return get_route_for_display(session_id)
    except (ImportError, RuntimeError, ValueError):
        return None


@dataclass(frozen=True)
class _FrameJob:
    sequence: int
    observation: OfficialObservation
    ticks: tuple[tuple[tuple[float, float, float], float], ...]
    enqueued_wall_s: float
    enqueued_monotonic_s: float
    producer_odometry_pose: SE2Pose


@dataclass(frozen=True)
class _ImagePose:
    anchor_node_id: Optional[int]
    local_pose: SE2Pose
    fallback_pose: SE2Pose
    frame_id: int
    source_sequence: Optional[int] = None


@dataclass(frozen=True)
class _PendingImagePose:
    source_sequence: int
    producer_odometry_pose: SE2Pose


@dataclass(frozen=True)
class _AnchoredTrailPose:
    anchor_node_id: Optional[int]
    local_pose: SE2Pose
    fallback_pose: SE2Pose


@dataclass(frozen=True)
class _Place:
    name: str
    x_m: float
    y_m: float
    source: str
    image_id: str = ""
    # Keep user landmarks in graph-local coordinates so loop-closure graph
    # optimization moves them together with the node that observed them.
    anchor_node_id: Optional[int] = None
    anchor_local_pose: Optional[SE2Pose] = None


def _finite_qvel(value: Any) -> Optional[tuple[float, float, float]]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return tuple(float(item) for item in array[:3])


def _camera_intrinsics(camera: dict[str, Any], width: int, height: int) -> CameraIntrinsics:
    try:
        fx = float(camera.get("fx") or 0.0)
        fy = float(camera.get("fy") or 0.0)
        cx = float(camera.get("cx") if camera.get("cx") is not None else width * 0.5)
        cy = float(camera.get("cy") if camera.get("cy") is not None else height * 0.5)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera intrinsics are invalid") from exc
    if fx <= 0.0:
        focal = float(camera.get("focal_length") or 0.0)
        aperture = float(camera.get("horizontal_aperture") or 0.0)
        if focal <= 0.0 or aperture <= 0.0:
            raise ValueError("camera focal length/aperture are missing")
        fx = focal * float(width) / aperture
    if fy <= 0.0:
        fy = fx
    return CameraIntrinsics(width, height, fx, fy, cx, cy)


def _official_observation(
    timestamp_s: float,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    camera: dict[str, Any],
) -> OfficialObservation:
    depth = np.asarray(depth_m).squeeze()
    image = np.asarray(rgb)
    if depth.ndim != 2 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("head RGB-D dimensions are invalid")
    height, width = depth.shape
    if image.shape[:2] != (height, width):
        raise ValueError("head RGB and depth dimensions differ")
    relative = dict(camera.get("robot_relative_pose") or {})
    if not relative and camera.get("pos") is not None and camera.get("quat") is not None:
        relative = {"pos": camera.get("pos"), "quat": camera.get("quat")}
    camera_pose = CameraRelativePose.create(
        relative.get("pos") or (), relative.get("quat") or ()
    )
    return OfficialObservation.create(
        timestamp_s=timestamp_s,
        head_rgb=np.ascontiguousarray(image, dtype=np.uint8),
        head_depth_m=np.ascontiguousarray(depth, dtype=np.float32),
        intrinsics=_camera_intrinsics(camera, width, height),
        camera_relative_pose=camera_pose,
    )


def _quat_matrix_xyzw(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.all(np.isfinite(array)):
        raise ValueError("camera quaternion must contain four finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("camera quaternion has zero norm")
    x, y, z, w = array / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_capture_bundle(session_id: str, image_id: str) -> Optional[dict[str, Any]]:
    try:
        from behavior_interface import agent_runs

        meta = agent_runs.load_image_meta(session_id, image_id)
        depth_path = Path(agent_runs.image_path(session_id, image_id, ".depth.npy"))
        if not depth_path.is_file():
            return None
        return {
            "depth": np.load(depth_path, allow_pickle=False),
            "camera": dict((meta or {}).get("camera") or {}),
            "evaluator_sequence": (meta or {}).get("evaluator_sequence"),
        }
    except Exception:
        return None


def _unproject_uv_robot_xy(
    bundle: dict[str, Any], u: Any, v: Any
) -> Optional[tuple[float, float]]:
    depth = np.asarray(bundle.get("depth"), dtype=np.float32).squeeze()
    camera = dict(bundle.get("camera") or {})
    if depth.ndim != 2:
        return None
    height, width = depth.shape
    try:
        column = relative_to_pixel(u, width)
        row = relative_to_pixel(v, height)
    except Exception:
        return None
    column = min(max(int(column), 0), width - 1)
    row = min(max(int(row), 0), height - 1)
    distance = float(depth[row, column])
    if not math.isfinite(distance) or distance <= 1e-4:
        return None
    try:
        intrinsics = _camera_intrinsics(camera, width, height)
        relative = dict(camera.get("robot_relative_pose") or {})
        if not relative and camera.get("pos") is not None and camera.get("quat") is not None:
            relative = {"pos": camera.get("pos"), "quat": camera.get("quat")}
        position = np.asarray(relative.get("pos"), dtype=np.float64).reshape(3)
        rotation = _quat_matrix_xyzw(relative.get("quat"))
    except Exception:
        return None
    camera_point = np.asarray(
        [
            (column - intrinsics.cx) / intrinsics.fx * distance,
            -(row - intrinsics.cy) / intrinsics.fy * distance,
            -distance,
        ],
        dtype=np.float64,
    )
    point = position + rotation @ camera_point
    return float(point[0]), float(point[1])


class LiveMapper:
    """Process-global live RTAB-Map session fed only by official observations."""

    def __init__(
        self,
        *,
        client_factory: Optional[Callable[[], RtabmapClient]] = None,
        frame_interval_s: float = FRAME_INTERVAL_S,
        queue_size: int = FRAME_QUEUE_SIZE,
        worker_timeout_s: Optional[float] = None,
    ) -> None:
        self.disabled = False
        self.failures = 0
        self.last_error = ""
        self.integrated = 0
        self._frame_interval_s = float(frame_interval_s)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._client_factory = client_factory or self._make_client
        self._input_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._sim_clock_s = 0.0
        self._pending_ticks: list[tuple[tuple[float, float, float], float]] = []
        self._producer_odometry = BodyOdometry()
        self._producer_poses: OrderedDict[int, SE2Pose] = OrderedDict()
        self._origin_enqueued = False
        self._last_enqueued_clock_s = -math.inf
        self._last_sequence: Optional[int] = None
        self._latest: Optional[MapResult] = None
        self._trail: list[_AnchoredTrailPose] = []
        # The start landmark is kept separate from user places, matching the
        # existing query_map contract and the renderer's first-trail marker.
        self._start_place: Optional[_Place] = None
        self._places: list[_Place] = []
        self._image_poses: dict[str, _ImagePose] = {}
        self._pending_image_poses: dict[str, _PendingImagePose] = {}
        self._session_id = "default"
        self._places_version = 0
        self._cache: dict[
            tuple[int, bool, int, Optional[float], int, int], tuple[bytes, str]
        ] = {}
        self.frames_enqueued = 0
        self.frames_processed = 0
        self.frames_skipped_queue_full = 0
        self.tracking_lost = 0
        self.worker_latency_s = 0.0
        self.queue_wait_s = 0.0
        self.worker_processing_s = 0.0
        self.worker_timeout_s = float(
            worker_request_timeout_s()
            if worker_timeout_s is None
            else worker_timeout_s
        )
        if not 0.0 < self.worker_timeout_s <= 3600.0:
            raise ValueError("worker_timeout_s must be in (0, 3600]")
        self.worker_stall_warning_s = min(
            WORKER_STALL_WARNING_S, self.worker_timeout_s
        )
        self.worker_timeout_count = 0
        self.worker_timed_out = False
        self._worker_started_monotonic_s: Optional[float] = None
        self._last_result_monotonic_s: Optional[float] = None
        self._last_result_source_monotonic_s: Optional[float] = None
        self._last_processed_sequence: Optional[int] = None
        self._last_processed_producer_pose: Optional[SE2Pose] = None
        self._worker_database_path: Optional[Path] = None
        self._worker_database_size_bytes: Optional[int] = None
        self._last_database_stat_monotonic_s: Optional[float] = None
        self.started_wall_s = time.time()
        atexit.register(self.close)

    @staticmethod
    def _make_client() -> RtabmapClient:
        raw = str(os.environ.get("BEHAVIOR_RTABMAP_LIVE_LOG", "") or "").strip()
        log_path = Path(raw).expanduser() if raw else Path(
            f"/tmp/behavior_rtabmap_live_{os.getpid()}.log"
        )
        database_path = None
        database_directory = str(
            os.environ.get("BEHAVIOR_RTABMAP_DATABASE_DIR", "") or ""
        ).strip()
        if database_directory:
            directory = Path(database_directory).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            database_path = directory / (
                f"rtabmap_worker_{os.getpid()}_{time.time_ns()}.db"
            )
        return RtabmapClient(
            database_path=database_path,
            log_path=log_path,
            profile="official",
        )

    def _enabled(self) -> bool:
        if self.disabled or self._closed or not live_backend_selected():
            return False
        try:
            from behavior_interface.spatial_map import spatial_map_enabled

            return bool(spatial_map_enabled())
        except Exception:
            return False

    def _ensure_thread(self) -> None:
        # Steady-state producers do not need the map publication/render lock.
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker_main,
                name="behavior-rtabmap-live",
                daemon=True,
            )
            self._thread.start()

    def odom_tick(self, world: Any, dt: float) -> None:
        self._check_worker_watchdog()
        if not self._enabled():
            return
        try:
            step = float(dt)
            if not math.isfinite(step) or step <= 0.0 or step > 1.0:
                return
            qvel = _finite_qvel(world.base_qvel())
            if qvel is None:
                return
            with self._input_lock:
                self._sim_clock_s += step
                self._pending_ticks.append((qvel, step))
                self._producer_odometry.advance(qvel, step)
        except Exception as exc:
            self._note_failure(f"odom_tick: {exc}", fatal=False)

    def map_tick(self, world: Any, now: Optional[float] = None) -> bool:
        del now
        self._check_worker_watchdog()
        if not self._enabled():
            return False
        try:
            from behavior_interface.continuous_capture import peek_sequence, read_frame

            sequence_hint = peek_sequence(world)
        except Exception as exc:
            self._note_failure(f"map_tick: {exc}", fatal=False)
            return False
        with self._input_lock:
            if sequence_hint is not None:
                sequence_key = int(sequence_hint)
                self._producer_poses[sequence_key] = self._producer_odometry.snapshot()
                self._producer_poses.move_to_end(sequence_key)
                while len(self._producer_poses) > PRODUCER_POSE_HISTORY_SIZE:
                    self._producer_poses.popitem(last=False)
            sim_clock = self._sim_clock_s
            if (
                self._origin_enqueued
                and self._pending_ticks
                and sim_clock - self._last_enqueued_clock_s < self._frame_interval_s
            ):
                return False
            if self._queue.full():
                self.frames_skipped_queue_full += 1
                return False
        try:
            if sequence_hint is not None and sequence_hint == self._last_sequence:
                return False
            frame = read_frame(world)
            if frame is None:
                return False
            sequence, rgb, depth_m, camera = frame
            with self._input_lock:
                if sequence == self._last_sequence or self._queue.full():
                    if self._queue.full():
                        self.frames_skipped_queue_full += 1
                    return False
                timestamp_s = max(
                    self._sim_clock_s,
                    0.0 if not self._origin_enqueued else self._last_enqueued_clock_s + 1e-6,
                )
                observation = _official_observation(
                    timestamp_s, rgb, depth_m, dict(camera or {})
                )
                if self._origin_enqueued:
                    ticks = tuple(self._pending_ticks)
                else:
                    # The first camera frame defines the episode origin. Motion
                    # before that frame cannot be registered to an observation.
                    ticks = ()
                self._pending_ticks.clear()
                self._origin_enqueued = True
                self._last_enqueued_clock_s = timestamp_s
                self._last_sequence = int(sequence)
                job = _FrameJob(
                    int(sequence),
                    observation,
                    ticks,
                    time.time(),
                    time.monotonic(),
                    self._producer_odometry.snapshot(),
                )
                self._queue.put_nowait(job)
                self.frames_enqueued += 1
            self._ensure_thread()
            return True
        except Exception as exc:
            self._note_failure(f"map_tick: {exc}", fatal=False)
            return False

    def _worker_main(self) -> None:
        client: Optional[RtabmapClient] = None
        pending_odometry_poses: list[SE2Pose] = []
        try:
            client = self._client_factory()
            database_path = getattr(client, "database_path", None)
            if database_path is not None:
                with self._state_lock:
                    self._worker_database_path = Path(database_path)
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    assert isinstance(item, _FrameJob)
                    with self._state_lock:
                        self._worker_started_monotonic_s = time.monotonic()
                        self.queue_wait_s = max(
                            0.0,
                            self._worker_started_monotonic_s - item.enqueued_monotonic_s,
                        )
                    for qvel, dt in item.ticks:
                        client.advance_odometry(qvel, dt)
                        pending_odometry_poses.append(client.odometry.snapshot())
                    result = client.submit(item.observation)
                    self._sample_worker_database_size(client)
                    if result.tracking_ok:
                        endpoint = client.odometry.snapshot()
                        self._accept_result(
                            item,
                            result,
                            tuple(pending_odometry_poses),
                            endpoint,
                        )
                        pending_odometry_poses.clear()
                    else:
                        # A failed RGB-D frame still has a valid body-velocity
                        # path.  Publish it provisionally, anchored to the
                        # last mapped pose, then clear it so recovery cannot
                        # append the same ticks twice.
                        endpoint = client.odometry.snapshot()
                        self._accept_result(
                            item,
                            result,
                            tuple(pending_odometry_poses),
                            endpoint,
                        )
                        pending_odometry_poses.clear()
                except Exception as exc:
                    message = f"worker: {type(exc).__name__}: {exc}"
                    if isinstance(exc, WorkerRequestTimeout):
                        self._note_worker_timeout(message)
                    else:
                        self._note_failure(message, fatal=True)
                    return
                finally:
                    with self._state_lock:
                        self._worker_started_monotonic_s = None
                    self._queue.task_done()
        finally:
            if client is not None:
                client.close()
                self._sample_worker_database_size(client, force=True)

    def _sample_worker_database_size(
        self,
        client: RtabmapClient,
        *,
        force: bool = False,
    ) -> None:
        """Sample storage telemetry only from the asynchronous worker thread."""

        database_path = getattr(client, "database_path", None)
        if database_path is None:
            return
        now = time.monotonic()
        with self._state_lock:
            previous = self._last_database_stat_monotonic_s
        if (
            not force
            and previous is not None
            and now - previous < DATABASE_STAT_INTERVAL_S
        ):
            return
        path = Path(database_path)
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        with self._state_lock:
            self._worker_database_path = path
            self._worker_database_size_bytes = size_bytes
            self._last_database_stat_monotonic_s = now

    @staticmethod
    def _anchor_trail_pose(result: MapResult, pose: SE2Pose) -> _AnchoredTrailPose:
        anchors = [item for item in result.poses if item.node_id > 0]
        if not anchors:
            return _AnchoredTrailPose(None, SE2Pose(), pose)
        by_id = {item.node_id: item for item in anchors}
        anchor_record = by_id.get(result.ref_node_id)
        if anchor_record is None:
            anchor_record = max(anchors, key=lambda item: item.node_id)
        anchor_pose = SE2Pose(
            anchor_record.x_m,
            anchor_record.y_m,
            anchor_record.yaw_rad,
        )
        return _AnchoredTrailPose(
            anchor_record.node_id,
            anchor_pose.relative_to(pose),
            pose,
        )

    @staticmethod
    def _resolve_anchored_pose(
        result: MapResult,
        anchor_node_id: Optional[int],
        local_pose: SE2Pose,
        fallback_pose: SE2Pose,
    ) -> SE2Pose:
        if anchor_node_id is not None:
            for record in result.poses:
                if record.node_id == anchor_node_id:
                    return SE2Pose(
                        record.x_m,
                        record.y_m,
                        record.yaw_rad,
                    ).compose(local_pose)
        return fallback_pose

    @classmethod
    def _image_pose_from_odometry(
        cls,
        result: MapResult,
        result_producer_pose: SE2Pose,
        image_producer_pose: SE2Pose,
        source_sequence: int,
    ) -> _ImagePose:
        capture_pose = result.current_pose.compose(
            result_producer_pose.relative_to(image_producer_pose)
        )
        anchored = cls._anchor_trail_pose(result, capture_pose)
        return _ImagePose(
            anchored.anchor_node_id,
            anchored.local_pose,
            anchored.fallback_pose,
            result.frame_id,
            int(source_sequence),
        )

    @classmethod
    def _anchor_place(
        cls,
        result: MapResult,
        pose: SE2Pose,
        preferred_node_id: Optional[int] = None,
    ) -> tuple[Optional[int], Optional[SE2Pose]]:
        if preferred_node_id is not None:
            for record in result.poses:
                if record.node_id == preferred_node_id:
                    anchor = SE2Pose(record.x_m, record.y_m, record.yaw_rad)
                    return record.node_id, anchor.relative_to(pose)
        anchored = cls._anchor_trail_pose(result, pose)
        if anchored.anchor_node_id is None:
            return None, None
        return anchored.anchor_node_id, anchored.local_pose

    @classmethod
    def _resolve_place(cls, result: MapResult, place: _Place) -> _Place:
        if place.anchor_node_id is None or place.anchor_local_pose is None:
            return place
        pose = cls._resolve_anchored_pose(
            result,
            place.anchor_node_id,
            place.anchor_local_pose,
            SE2Pose(place.x_m, place.y_m, 0.0),
        )
        return replace(place, x_m=pose.x_m, y_m=pose.y_m)

    def _resolved_places(
        self,
        result: MapResult,
        *,
        include_start: bool = False,
    ) -> tuple[_Place, ...]:
        places: list[_Place] = []
        if include_start and self._start_place is not None:
            places.append(self._resolve_place(result, self._start_place))
        places.extend(self._resolve_place(result, item) for item in self._places)
        return tuple(places)

    def _accept_result(
        self,
        job: _FrameJob,
        result: MapResult,
        odometry_poses: Sequence[SE2Pose],
        odometry_endpoint: Optional[SE2Pose],
    ) -> None:
        with self._state_lock:
            # The first worker result is the first map-frame observation that
            # this session can legally anchor.  It is deliberately derived
            # from RTAB-Map's reported pose, never from simulator state or a
            # benchmark mark.  Keep it exactly once, including across repeated
            # failed/tracking-lost observations of the same episode.
            if self._start_place is None:
                pose = result.current_pose
                if all(
                    math.isfinite(float(value))
                    for value in (pose.x_m, pose.y_m, pose.yaw_rad)
                ):
                    anchor_node_id, anchor_local_pose = self._anchor_place(
                        result, pose
                    )
                    self._start_place = _Place(
                        AUTO_START_NAME,
                        float(pose.x_m),
                        float(pose.y_m),
                        AUTO_START_SOURCE,
                        anchor_node_id=anchor_node_id,
                        anchor_local_pose=anchor_local_pose,
                    )
                    self._places_version += 1
                    self._cache.clear()
            self._latest = result
            self.frames_processed += 1
            self.integrated = self.frames_processed
            completed = time.monotonic()
            self._last_result_monotonic_s = completed
            self._last_result_source_monotonic_s = job.enqueued_monotonic_s
            self._last_processed_sequence = job.sequence
            producer_pose = getattr(job, "producer_odometry_pose", None)
            self._last_processed_producer_pose = (
                producer_pose if isinstance(producer_pose, SE2Pose) else None
            )
            self.worker_latency_s = max(0.0, completed - job.enqueued_monotonic_s)
            started = self._worker_started_monotonic_s
            self.worker_processing_s = (
                0.0 if started is None else max(0.0, completed - started)
            )
            # 每个 qvel tick 都进入轨迹。整段只用成功帧的 SLAM 端点做刚体重锚，
            # 再保存到图节点的局部位姿，使后续回环同步修正蓝线；轨迹从不反馈建图。
            if result.recovery_hold:
                # Local RGB-D odometry can remain healthy while the absolute
                # map transform is unknown. Do not turn that provisional chain
                # into a globally anchored blue trail; a later relocalization
                # cannot reconstruct its exact historical global positions.
                pass
            elif result.tracking_ok:
                if odometry_poses and odometry_endpoint is not None:
                    mapped_poses = align_odometry_path_to_endpoint(
                        odometry_poses,
                        odometry_endpoint,
                        result.current_pose,
                    )
                else:
                    mapped_poses = (result.current_pose,)
                self._trail.extend(
                    self._anchor_trail_pose(result, pose)
                    for pose in mapped_poses
                )
            else:
                # Tracking loss must not erase motion from the displayed
                # trajectory.  No map/graph state is changed here: these are
                # only provisional trail samples and will be re-anchored by
                # the next successful result.
                if odometry_poses and odometry_endpoint is not None:
                    with self._state_lock:
                        latest = self._latest
                    reference = (
                        latest.current_pose
                        if latest is not None and latest.tracking_ok
                        else result.current_pose
                    )
                    mapped_poses = align_odometry_path_to_endpoint(
                        odometry_poses,
                        odometry_endpoint,
                        reference,
                    )
                else:
                    mapped_poses = (result.current_pose,)
                self._trail.extend(
                    self._anchor_trail_pose(result, pose)
                    for pose in mapped_poses
                )
            if not result.tracking_ok:
                self.tracking_lost += 1
            if (
                result.tracking_ok
                and not result.recovery_hold
                and self._last_processed_producer_pose is not None
            ):
                ready = [
                    (key, pending)
                    for key, pending in self._pending_image_poses.items()
                    if pending.source_sequence <= job.sequence
                ]
                for key, pending in ready:
                    self._image_poses[key] = self._image_pose_from_odometry(
                        result,
                        self._last_processed_producer_pose,
                        pending.producer_odometry_pose,
                        pending.source_sequence,
                    )
                    self._pending_image_poses.pop(key, None)
            if not self.worker_timed_out:
                self.failures = 0
                self.last_error = ""
            self._cache.clear()

    def _note_failure(self, message: str, *, fatal: bool) -> None:
        with self._state_lock:
            self.failures += 1
            self.last_error = str(message)[:600]
            if fatal or self.failures >= MAX_FAILURES:
                self.disabled = True

    def _note_worker_timeout(self, message: str) -> None:
        with self._state_lock:
            if not self.worker_timed_out:
                self.worker_timeout_count += 1
            self.worker_timed_out = True
            self.failures += 1
            self.last_error = str(message)[:600]
            self.disabled = True

    def _worker_busy_s_locked(self, now: float) -> float:
        if self._worker_started_monotonic_s is None:
            return 0.0
        return max(0.0, now - self._worker_started_monotonic_s)

    def _worker_stalled_locked(self, now: float) -> bool:
        return bool(
            self.worker_timed_out
            or self._worker_busy_s_locked(now) >= self.worker_stall_warning_s
        )

    def _check_worker_watchdog(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            busy_s = self._worker_busy_s_locked(now)
            if (
                self._worker_started_monotonic_s is not None
                and busy_s >= self.worker_timeout_s
                and not self.worker_timed_out
            ):
                self.worker_timeout_count += 1
                self.worker_timed_out = True
                self.failures += 1
                self.last_error = (
                    "worker: RTAB-Map asynchronous request exceeded total "
                    f"watchdog deadline ({busy_s:.3f}s >= "
                    f"{self.worker_timeout_s:.3f}s)"
                )[:600]
                self.disabled = True

    def wait_until_idle(self, timeout_s: float = 10.0) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                # The worker will drain at least one item, then this bounded
                # wait can place the shutdown marker without dropping a frame.
                try:
                    self._queue.put(_STOP, timeout=2.0)
                except queue.Full:
                    return
            thread.join(timeout=5.0)

    def adopt_session(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if key:
            with self._state_lock:
                self._session_id = key

    def note_image_pose(
        self,
        image_id: str,
        evaluator_sequence: Optional[int] = None,
    ) -> None:
        key = str(image_id or "").strip()
        if not key:
            return
        with self._state_lock:
            if key in self._image_poses or self.disabled:
                return
            session_id = self._session_id
        source_sequence: Optional[int]
        try:
            source_sequence = (
                int(evaluator_sequence)
                if evaluator_sequence is not None
                else None
            )
        except (TypeError, ValueError, OverflowError):
            source_sequence = None
        if source_sequence is None:
            try:
                from behavior_interface import agent_runs

                meta = agent_runs.load_image_meta(session_id, key)
                value = (meta or {}).get("evaluator_sequence")
                if isinstance(value, int) and not isinstance(value, bool):
                    source_sequence = int(value)
            except Exception:
                pass

        producer_pose: Optional[SE2Pose] = None
        if source_sequence is not None:
            with self._input_lock:
                producer_pose = self._producer_poses.get(source_sequence)
        with self._state_lock:
            result = self._latest
            if (
                result is None
                or self.disabled
                or result.recovery_hold
                or not result.tracking_ok
            ):
                return
            if source_sequence is not None:
                if producer_pose is None:
                    return
                if (
                    self._last_processed_sequence is not None
                    and self._last_processed_sequence >= source_sequence
                    and self._last_processed_producer_pose is not None
                ):
                    self._image_poses[key] = self._image_pose_from_odometry(
                        result,
                        self._last_processed_producer_pose,
                        producer_pose,
                        source_sequence,
                    )
                else:
                    self._pending_image_poses[key] = _PendingImagePose(
                        source_sequence,
                        producer_pose,
                    )
            else:
                anchored = self._anchor_trail_pose(
                    result,
                    result.current_pose,
                )
                self._image_poses[key] = _ImagePose(
                    anchored.anchor_node_id,
                    anchored.local_pose,
                    anchored.fallback_pose,
                    result.frame_id,
                )

    @staticmethod
    def _bearing(pose: SE2Pose, x_m: float, y_m: float) -> tuple[float, float]:
        dx = float(x_m) - pose.x_m
        dy = float(y_m) - pose.y_m
        distance = math.hypot(dx, dy)
        world = math.atan2(dy, dx) if distance > 1e-9 else pose.yaw_rad
        bearing = (world - pose.yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
        return distance, math.degrees(bearing)

    def _display_result(
        self,
        heading_up: bool,
        *,
        max_points: Optional[int] = None,
    ) -> Optional[MapResult]:
        with self._state_lock:
            result = self._latest
            if result is None:
                return None
            graph = {
                item.node_id: SE2Pose(item.x_m, item.y_m, item.yaw_rad)
                for item in result.poses
            }
            trail = self._trail
            limit = None
            if max_points is not None:
                try:
                    limit = max(1, int(max_points))
                except (TypeError, ValueError, OverflowError):
                    limit = None
            if limit is not None and len(trail) > limit:
                # Preserve both endpoints and use deterministic nearest-index
                # sampling.  This changes only raster detail; the anchored
                # source trail remains untouched for all control/map logic.
                indices = (
                    round(index * (len(trail) - 1) / (limit - 1))
                    for index in range(limit)
                ) if limit > 1 else (len(trail) - 1,)
                selected = (trail[index] for index in indices)
            else:
                selected = iter(trail)
            resolved = []
            for item in selected:
                anchor = graph.get(item.anchor_node_id)
                resolved.append(
                    anchor.compose(item.local_pose)
                    if anchor is not None
                    else item.fallback_pose
                )
            trail = tuple(
                PoseRecord(index + 1, pose.x_m, pose.y_m, pose.yaw_rad)
                for index, pose in enumerate(resolved)
            )
            return replace(result, poses=trail)

    def map_snapshot_png(
        self,
        *,
        heading_up: bool = True,
        size: int = DEFAULT_MAP_SIZE_PX,
        span_m: Optional[float] = DEFAULT_MAP_SPAN_M,
    ) -> tuple[bytes, str]:
        with self._state_lock:
            native_result = self._latest
            if native_result is None:
                raise RuntimeError(
                    "RTAB-Map has not processed its first official RGB-D frame"
                )
            result = self._display_result(
                bool(heading_up),
                max_points=DISPLAY_TRAIL_MAX_POINTS,
            )
            assert result is not None
            places = tuple(
                (item.x_m, item.y_m, item.name)
                for item in self._resolved_places(native_result)
            )
            route = _navigation_route_snapshot(self._session_id)
            route_revision = 0 if route is None else int(route.revision)
            view_span = None if span_m is None else float(span_m)
            key = (
                result.frame_id,
                bool(heading_up),
                int(size),
                view_span,
                self._places_version,
                route_revision,
            )
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        image = render_heading_up(
            result,
            size_px=int(size),
            span_m=view_span,
            heading_up=bool(heading_up),
            places=places,
            route_xy_m=(None if route is None else route.points_xy_m),
        )
        output = BytesIO()
        image.save(output, format="PNG")
        payload = output.getvalue()
        version = (
            f"{BUILD}:{result.frame_id}:{result.loop_count}:"
            f"{self._places_version}:{route_revision}:{int(bool(heading_up))}"
        )
        with self._state_lock:
            self._cache[key] = (payload, version)
        return payload, version

    def _place_reports(
        self,
        result: MapResult,
        pose: SE2Pose,
    ) -> list[dict[str, Any]]:
        reports = []
        for item in self._resolved_places(result):
            distance, bearing = self._bearing(pose, item.x_m, item.y_m)
            reports.append(
                {
                    "name": item.name,
                    "range_m": round(distance, 2),
                    "spin_deg_to_face": round(bearing, 1),
                    "source": item.source,
                    "image_id": item.image_id,
                }
            )
        return reports

    def query(self, session_id: str = "") -> dict[str, Any]:
        self._check_worker_watchdog()
        self.adopt_session(session_id)
        with self._state_lock:
            result = self._latest
            session = self._session_id
            if result is None:
                return {
                    "ok": True,
                    "backend": BACKEND,
                    "build": BUILD,
                    "session_id": session,
                    "empty": True,
                    "hint": "RTAB-Map 正在等待第一帧官方 RGB-D。",
                }
            pose = result.current_pose
            start = (
                None
                if self._start_place is None
                else self._resolve_place(result, self._start_place)
            )
            # A valid result normally always has a start landmark.  Keep the
            # origin fallback for defensive compatibility with old in-memory
            # mapper instances created before this field was introduced.
            start_x = 0.0 if start is None else start.x_m
            start_y = 0.0 if start is None else start.y_m
            start_range, start_bearing = self._bearing(pose, start_x, start_y)
            return {
                "ok": True,
                "backend": BACKEND,
                "build": BUILD,
                "session_id": session,
                "empty": False,
                "pose": {
                    "x": round(pose.x_m, 3),
                    "y": round(pose.y_m, 3),
                    "yaw_deg": round(math.degrees(pose.yaw_rad), 1),
                    "frame": "rtabmap_contact_aware_q",
                    "global_confident": not result.recovery_hold,
                },
                "start": {
                    "name": AUTO_START_NAME,
                    "range_m": round(start_range, 3),
                    "bearing_deg": round(start_bearing, 1),
                    "source": (
                        AUTO_START_SOURCE if start is None else start.source
                    ),
                    "note": "本局第一次官方 RGB-D 观测",
                },
                "places": self._place_reports(result, pose),
                "mapped_frames": self.frames_processed,
                "worker_stalled": self._worker_stalled_locked(time.monotonic()),
                "worker_timed_out": self.worker_timed_out,
                "tracking_lost": self.tracking_lost,
                "loop_count": result.loop_count,
                "node_count": result.node_count,
                "slam_mode": (
                    "relocalizing"
                    if result.recovery_hold
                    else "mapping"
                    if result.mapping_active
                    else "localization"
                ),
                "recovery_hold": result.recovery_hold,
                "localized_this_frame": result.localized,
                "visual_localized_this_frame": result.visual_localized,
                "geometric_localized_this_frame": result.geometric_localized,
                "live_pose_role": "Q",
                "hint": (
                    "RTAB-Map 正在把恢复后的局部轨迹重新定位到冻结历史图；"
                    "当前绝对 x/y 和地图标记暂停授权。"
                    if result.recovery_hold
                    else
                    f"RTAB-Map 已处理 {self.frames_processed} 帧，"
                    f"当前为{'建图' if result.mapping_active else '冻结地图定位'}模式，"
                    f"回环/定位匹配累计 {result.loop_count} 次；冻结后不会恢复结构建图。"
                ),
            }

    def mark_on_map(
        self,
        session_id: str,
        name: str,
        *,
        image_id: str = "",
        u: Any = None,
        v: Any = None,
        evaluator_sequence: Optional[int] = None,
    ) -> dict[str, Any]:
        label = str(name or "").strip()
        if not label:
            return {"ok": False, "tool": "mark_on_map", "error": "需要 name"}
        self.adopt_session(session_id)
        picked = str(image_id or "").strip()
        if bool(picked) or u is not None or v is not None:
            if not (picked and u is not None and v is not None):
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": "点选要同时给 image_id、u、v；只标脚下就三个都别给",
                }
        try:
            current_sequence = (
                int(evaluator_sequence)
                if evaluator_sequence is not None
                else None
            )
        except (TypeError, ValueError, OverflowError):
            current_sequence = None

        bundle: Optional[dict[str, Any]] = None
        robot_xy: Optional[tuple[float, float]] = None
        image_sequence: Optional[int] = None
        if picked:
            bundle = _load_capture_bundle(session_id, picked)
            if bundle is None:
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": f"{picked} 没有可用的 depth/cam_rel_pose",
                }
            value = bundle.get("evaluator_sequence")
            if isinstance(value, int) and not isinstance(value, bool):
                image_sequence = int(value)
            self.note_image_pose(picked, image_sequence)
            robot_xy = _unproject_uv_robot_xy(bundle, u, v)
            if robot_xy is None:
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": f"{picked} 的 ({u},{v}) 没有可用 depth/cam_rel_pose",
                }

        current_producer_pose: Optional[SE2Pose] = None
        if current_sequence is not None:
            with self._input_lock:
                current_producer_pose = self._producer_poses.get(current_sequence)
        with self._state_lock:
            result = self._latest
            if result is None:
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": "RTAB-Map 尚无位姿",
                }
            confident_image_pose = self._image_poses.get(picked) if picked else None
            existing_auto = (
                self._start_place
                if label == AUTO_START_NAME and not picked
                else None
            )
            thread = self._thread
            if existing_auto is None and (
                self.disabled or thread is None or not thread.is_alive()
            ):
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": "RTAB-Map worker 不健康；拒绝用旧位姿写入地图标记",
                    "disabled": self.disabled,
                    "worker_thread_alive": bool(
                        thread is not None and thread.is_alive()
                    ),
                    "last_processed_evaluator_sequence": self._last_processed_sequence,
                }
            if existing_auto is None and not result.tracking_ok:
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": "RTAB-Map 当前跟踪未确认；暂不写入地图标记",
                }
            if (
                existing_auto is None
                and current_sequence is not None
                and self._last_processed_sequence is not None
                and current_sequence > self._last_processed_sequence
            ):
                source_age = (
                    math.inf
                    if self._last_result_source_monotonic_s is None
                    else max(
                        0.0,
                        time.monotonic() - self._last_result_source_monotonic_s,
                    )
                )
                if source_age > MAX_MARK_MAP_SOURCE_AGE_S:
                    return {
                        "ok": False,
                        "tool": "mark_on_map",
                        "error": "RTAB-Map 地图源已过期；拒绝用旧位姿写入地图标记",
                        "map_source_age_s": round(source_age, 3),
                        "evaluator_sequence": current_sequence,
                        "last_processed_evaluator_sequence": self._last_processed_sequence,
                    }
            if picked and confident_image_pose is None:
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": (
                        f"RTAB-Map 尚未可靠绑定 {picked} 的采集帧；"
                        "拒绝退回到当前位姿，请在地图处理追上后重试"
                    ),
                    "image_evaluator_sequence": image_sequence,
                    "last_processed_evaluator_sequence": self._last_processed_sequence,
                }
            if (
                result.recovery_hold
                and existing_auto is None
                and confident_image_pose is None
            ):
                return {
                    "ok": False,
                    "tool": "mark_on_map",
                    "error": (
                        "RTAB-Map 正在恢复全局定位；当前绝对位置未确认，"
                        "暂不写入地图标记"
                    ),
                    "slam_mode": "relocalizing",
                }
            pose = result.current_pose
            if (
                existing_auto is None
                and not picked
                and current_sequence is not None
                and current_sequence != self._last_processed_sequence
            ):
                if (
                    current_producer_pose is None
                    or self._last_processed_producer_pose is None
                ):
                    return {
                        "ok": False,
                        "tool": "mark_on_map",
                        "error": "当前 evaluator 帧没有可审计的 qvel 位姿绑定",
                        "evaluator_sequence": current_sequence,
                        "last_processed_evaluator_sequence": self._last_processed_sequence,
                    }
                pose = result.current_pose.compose(
                    self._last_processed_producer_pose.relative_to(
                        current_producer_pose
                    )
                )
            source = "robot_position"
            x_m, y_m = pose.x_m, pose.y_m
            place_anchor_node_id: Optional[int] = None
            place_anchor_local_pose: Optional[SE2Pose] = None
            already_marked = False
            # ``start`` is the canonical automatic landmark.  Treat an
            # explicit foot-point request with that reserved name as an
            # idempotent lookup; a pixel pick remains a normal user place.
            if existing_auto is not None:
                place = self._resolve_place(result, existing_auto)
                x_m, y_m = place.x_m, place.y_m
                source = place.source
                already_marked = True
            elif picked:
                image_pose = confident_image_pose
                assert image_pose is not None
                assert robot_xy is not None
                capture_pose = self._resolve_anchored_pose(
                    result,
                    image_pose.anchor_node_id,
                    image_pose.local_pose,
                    image_pose.fallback_pose,
                )
                world = capture_pose.compose(SE2Pose(robot_xy[0], robot_xy[1], 0.0))
                x_m, y_m = world.x_m, world.y_m
                source = "image_pick"
                place_anchor_node_id, place_anchor_local_pose = self._anchor_place(
                    result,
                    world,
                    None if image_pose is None else image_pose.anchor_node_id,
                )
            if not already_marked:
                if not picked:
                    (
                        place_anchor_node_id,
                        place_anchor_local_pose,
                    ) = self._anchor_place(result, pose)
                place = _Place(
                    label,
                    float(x_m),
                    float(y_m),
                    source,
                    picked,
                    anchor_node_id=place_anchor_node_id,
                    anchor_local_pose=place_anchor_local_pose,
                )
                self._places.append(place)
                self._places_version += 1
                self._cache.clear()
            distance, bearing = self._bearing(pose, place.x_m, place.y_m)
            places = self._place_reports(result, pose)
        return {
            "ok": True,
            "tool": "mark_on_map",
            "build": BUILD,
            "marked": label,
            "marked_from": source,
            "already_marked": already_marked,
            "marked_range_m": round(distance, 2),
            "marked_spin_deg_to_face": round(bearing, 1),
            "places": places,
            "hint": f"已在 RTAB-Map 标记「{label}」。",
        }

    def attach_to_result(
        self,
        session_id: str,
        tool: str,
        args: Optional[dict[str, Any]],
        result: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        del args
        payload = dict(result or {})
        self.adopt_session(session_id)
        image_id = str(payload.get("image_id") or "").strip()
        if image_id:
            self.note_image_pose(
                image_id,
                payload.get("evaluator_sequence"),
            )
        chassis_tools = {
            "adjust_chassis",
            "spin_to_facing_point",
            "face_to_point",
            "move_chassis_to_floor_point",
            "move_base_to_point",
            "move_to_reach_point",
        }
        if str(tool or "").strip() not in chassis_tools:
            return payload
        payload["spatial_map"] = self.query(session_id)
        try:
            png, _version = self.map_snapshot_png(size=640)
            payload["rgb_minimap"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            if session_id:
                from behavior_interface import agent_runs

                path = agent_runs.image_path(
                    session_id, image_id or "map", ".minimap.rtabmap.png"
                )
                Path(path).write_bytes(png)
                payload["minimap_path"] = path
        except Exception:
            pass
        return payload

    def status(self) -> dict[str, Any]:
        self._check_worker_watchdog()
        with self._state_lock:
            result = self._latest
            thread = self._thread
            now = time.monotonic()
            result_age = (
                None if self._last_result_monotonic_s is None
                else max(0.0, now - self._last_result_monotonic_s)
            )
            source_age = (
                None if self._last_result_source_monotonic_s is None
                else max(0.0, now - self._last_result_source_monotonic_s)
            )
            worker_busy_s = self._worker_busy_s_locked(now)
            return {
                "backend": BACKEND,
                "build": BUILD,
                "selected": live_backend_selected(),
                "enabled": self._enabled(),
                "disabled": self.disabled,
                "worker_thread_alive": bool(thread is not None and thread.is_alive()),
                "frames_enqueued": self.frames_enqueued,
                "frames_processed": self.frames_processed,
                "frames_skipped_queue_full": self.frames_skipped_queue_full,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "pending_qvel_ticks": len(self._pending_ticks),
                "last_evaluator_sequence": self._last_sequence,
                "last_processed_evaluator_sequence": self._last_processed_sequence,
                "last_result_age_s": None if result_age is None else round(result_age, 3),
                "map_source_age_s": None if source_age is None else round(source_age, 3),
                "worker_busy": self._worker_started_monotonic_s is not None,
                "worker_busy_s": round(worker_busy_s, 3),
                "worker_stalled": self._worker_stalled_locked(now),
                "worker_timed_out": self.worker_timed_out,
                "worker_timeout_s": self.worker_timeout_s,
                "worker_timeout_count": self.worker_timeout_count,
                "queue_wait_s": round(self.queue_wait_s, 3),
                "worker_processing_s": round(self.worker_processing_s, 3),
                "worker_database_path": (
                    None
                    if self._worker_database_path is None
                    else str(self._worker_database_path)
                ),
                "worker_database_size_bytes": self._worker_database_size_bytes,
                "tracking_lost": self.tracking_lost,
                "loop_count": 0 if result is None else result.loop_count,
                "node_count": 0 if result is None else result.node_count,
                "slam_mode": (
                    "waiting"
                    if result is None
                    else "relocalizing"
                    if result.recovery_hold
                    else "mapping"
                    if result.mapping_active
                    else "localization"
                ),
                "recovery_hold": bool(
                    result is not None and result.recovery_hold
                ),
                "localized_this_frame": bool(
                    result is not None and result.localized
                ),
                "visual_localized_this_frame": bool(
                    result is not None and result.visual_localized
                ),
                "geometric_localized_this_frame": bool(
                    result is not None and result.geometric_localized
                ),
                "worker_latency_s": round(self.worker_latency_s, 3),
                "failures": self.failures,
                "last_error": self.last_error,
                "allowed_inputs": [
                    "head_rgb",
                    "head_depth_m",
                    "cam_rel_pose",
                    "base_qvel",
                    "evaluator_sequence",
                ],
                "forbidden_inputs_consumed": [],
            }


LiveMapper.map_snapshot_png._navigation_route_overlay_native = True


_SINGLETON_LOCK = threading.Lock()
_SINGLETON: Optional[LiveMapper] = None


def get_live_mapper(*, create: bool = True) -> Optional[LiveMapper]:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None and create:
            _SINGLETON = LiveMapper()
        return _SINGLETON


def reset_live_mapper() -> None:
    """Drop the process-global mapper at an evaluator episode boundary."""

    global _SINGLETON
    with _SINGLETON_LOCK:
        previous = _SINGLETON
        _SINGLETON = None
    if previous is not None:
        previous.close()
    try:
        from behavior_interface_eval_test.navigation_route_overlay import (
            clear_all_routes,
        )

        clear_all_routes()
    except (ImportError, RuntimeError):
        pass


def stamp_minimap_on_bgr(
    head_bgr: np.ndarray,
    *,
    mapper: Optional[LiveMapper] = None,
) -> np.ndarray:
    """Overlay the selected live RTAB map in the head frame's top-right corner."""

    head = np.asarray(head_bgr)
    if head.ndim != 3 or head.shape[2] < 3:
        raise ValueError("head_bgr must be HxWx3")
    out = np.ascontiguousarray(head[:, :, :3].copy())
    live = mapper or get_live_mapper()
    if live is None:
        return out
    try:
        payload, _version = live.map_snapshot_png(heading_up=True)
        with Image.open(BytesIO(payload)) as image:
            minimap = image.convert("RGBA")
    except Exception:
        return out

    height, width = out.shape[:2]
    side = max(96, int(round(min(height, width) * HEAD_MINIMAP_FRACTION)))
    margin = max(6, int(round(min(height, width) * HEAD_MINIMAP_MARGIN_FRAC)))
    x0 = width - margin - side
    y0 = margin
    if x0 < 0 or y0 + side > height:
        return out
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    source = np.asarray(minimap.resize((side, side), resample), dtype=np.uint8)
    # The renderer is RGBA; the official live frame and return value are BGR.
    source_bgr = source[:, :, 2::-1].astype(np.float32)
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    patch = out[y0 : y0 + side, x0 : x0 + side].astype(np.float32)
    out[y0 : y0 + side, x0 : x0 + side] = np.clip(
        source_bgr * alpha + patch * (1.0 - alpha), 0.0, 255.0
    ).astype(np.uint8)
    return out
