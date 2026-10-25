"""Versioned binary IPC used by the Python client and native RTAB-Map worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import io
import math
import struct
from typing import BinaryIO

import numpy as np

from .official import OfficialObservation, SE2Pose


VERSION = 16
REQUEST_MAGIC = b"B1RQ"
RESPONSE_MAGIC = b"B1RS"
MAX_PACKET_BYTES = 128 * 1024 * 1024

PREFIX = struct.Struct("<4sHHQ")
FRAME_META = struct.Struct("<QdII4d12d3dIi12dQ")
RESPONSE_META = struct.Struct("<QBBBBBBBQII12d6IiI")
POSE_RECORD = struct.Struct("<i3d")

FRAME_FLAG_RECOVERY_HOLD = 1 << 0
FRAME_FLAG_RECOVERY_RELOCALIZATION = 1 << 1
FRAME_FLAG_STRUCTURAL_USABLE = 1 << 2
FRAME_FLAG_VERIFIED_GRAPH_BRIDGE = 1 << 3
FRAME_FLAG_NORMAL_GLOBAL_NO_MODE = 1 << 4
FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING = 1 << 5
FRAME_FLAG_RECOVERY_GLOBAL_NO_MODE = 1 << 6
FRAME_FLAGS_KNOWN = (
    FRAME_FLAG_RECOVERY_HOLD
    | FRAME_FLAG_RECOVERY_RELOCALIZATION
    | FRAME_FLAG_STRUCTURAL_USABLE
    | FRAME_FLAG_VERIFIED_GRAPH_BRIDGE
    | FRAME_FLAG_NORMAL_GLOBAL_NO_MODE
    | FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING
    | FRAME_FLAG_RECOVERY_GLOBAL_NO_MODE
)


class Command(IntEnum):
    RESET = 1
    FRAME = 2
    PING = 3
    SHUTDOWN = 4


class Status(IntEnum):
    OK = 0
    TRACKING_LOST = 1
    BAD_REQUEST = 2
    INTERNAL_ERROR = 3


class QueryOutcome(IntEnum):
    NONE = 0
    HOLDING = 1
    COMMITTED = 2
    ALL_KNOWN = 3
    BRIDGE_ONLY_DISCARDED = 4
    NOVELTY_RESUMED = 5


class QueryScope(IntEnum):
    NONE = 0
    RECOVERY = 1
    NORMAL = 2


@dataclass(frozen=True)
class PoseRecord:
    node_id: int
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class ExternalLoopHypothesis:
    """One RGB-D metric revisit submitted for native RTAB-Map validation.

    ``candidate_to_current`` is the current base pose expressed in the
    historical candidate base frame, i.e. the transform that maps current-frame
    points into the candidate frame. Appearance retrieval is not represented
    here: callers may construct this object only after depth-backed SE(2)
    registration and temporal consensus.
    """

    candidate_id: int
    candidate_to_current: SE2Pose
    covariance: tuple[float, ...]
    recovery_relocalization: bool = False
    verified_graph_bridge: bool = False

    def __post_init__(self) -> None:
        if int(self.candidate_id) <= 0:
            raise ValueError("external loop candidate id must be positive")
        pose = self.candidate_to_current
        if not all(
            math.isfinite(float(value))
            for value in (pose.x_m, pose.y_m, pose.yaw_rad)
        ):
            raise ValueError("external loop pose must be finite")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.size != 9:
            raise ValueError("external loop covariance must contain 9 values")
        covariance = covariance.reshape(3, 3)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("external loop covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
            raise ValueError("external loop covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("external loop covariance must be positive semidefinite")
        if np.any(np.diag(covariance) <= 0.0):
            raise ValueError("external loop covariance diagonal must be positive")
        object.__setattr__(
            self,
            "covariance",
            tuple(float(value) for value in covariance.reshape(-1)),
        )
        if self.recovery_relocalization and self.verified_graph_bridge:
            raise ValueError(
                "recovery relocalization and normal verified graph bridge are exclusive"
            )


@dataclass(frozen=True)
class MapResult:
    frame_id: int
    tracking_ok: bool
    map_updated: bool
    loop_closed: bool
    occupancy: np.ndarray
    low_obstacles: np.ndarray
    high_obstacles: np.ndarray
    x_min_m: float
    y_min_m: float
    cell_size_m: float
    current_pose: SE2Pose
    node_count: int
    loop_count: int
    inliers: int
    features: int
    poses: tuple[PoseRecord, ...]
    native_pose: SE2Pose | None = None
    fused_odometry_pose: SE2Pose | None = None
    mapping_active: bool = True
    localized: bool = False
    visual_localized: bool = False
    geometric_localized: bool = False
    optimizer_backend: str = "ceres"
    ref_node_id: int = 0
    # A mapping session can temporarily stage a geometrically plausible revisit
    # or handle a confirmed revisit without entering irreversible global
    # frozen-map mode. ``localized`` distinguishes confirmed pose correction
    # from staging. This bit stays separate from ``mapping_active`` so new,
    # genuinely unknown areas can still be added after the candidate exits.
    read_only_match: bool = False
    # The graph and occupancy stayed immutable while global relocalization
    # gathered enough post-interruption evidence.
    recovery_hold: bool = False
    # The worker suppressed graph/raster writes while keeping the one native
    # RTAB-Map instance in incremental-memory mode. This is independent from
    # ``mapping_active``, which remains the native lifecycle invariant.
    soft_localization: bool = False
    # Localization evidence is currently insufficient. The map is still held
    # read-only until either a verified old-place anchor or sustained novelty.
    uncertain_hold: bool = False
    query_outcome: QueryOutcome = QueryOutcome.NONE
    # Query acknowledgements are transactional: a terminal outcome is valid
    # only when both values echo the hold episode sent on this frame.
    query_scope: QueryScope = QueryScope.NONE
    query_generation: int = 0
    # Native-authoritative progress through the post-novelty occupancy
    # reconciliation window. Stationary duplicate mapping nodes do not consume
    # this counter.
    novelty_resume_reconciliation_viewpoints_remaining: int = 0

    @property
    def map_write_enabled(self) -> bool:
        return bool(
            self.mapping_active
            and not self.soft_localization
            and not self.recovery_hold
        )


@dataclass(frozen=True)
class DecodedFrame:
    frame_id: int
    observation: OfficialObservation
    camera_to_base: np.ndarray
    odometry_delta: SE2Pose
    external_loop_hypothesis: ExternalLoopHypothesis | None = None
    recovery_hold: bool = False
    structural_usable: bool = True
    normal_global_no_mode: bool = False
    normal_global_search_pending: bool = False
    recovery_global_no_mode: bool = False
    query_generation: int = 0

    @property
    def external_loop_candidate_id(self) -> int:
        hypothesis = self.external_loop_hypothesis
        return 0 if hypothesis is None else int(hypothesis.candidate_id)


def _packet(magic: bytes, code: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds safety limit")
    return PREFIX.pack(magic, VERSION, int(code), len(payload)) + payload


def command_packet(command: Command) -> bytes:
    return _packet(REQUEST_MAGIC, command)


def frame_packet(
    observation: OfficialObservation,
    odometry_delta: SE2Pose,
    frame_id: int,
    external_loop_hypothesis: ExternalLoopHypothesis | None = None,
    recovery_hold: bool = False,
    structural_usable: bool = True,
    normal_global_no_mode: bool = False,
    normal_global_search_pending: bool = False,
    recovery_global_no_mode: bool = False,
    query_generation: int = 0,
) -> bytes:
    if normal_global_no_mode and not normal_global_search_pending:
        raise ValueError("normal global no-mode requires a pending search")
    if recovery_global_no_mode and not recovery_hold:
        raise ValueError("recovery global no-mode requires a recovery hold")
    if recovery_hold and normal_global_search_pending:
        raise ValueError("recovery and normal query holds are exclusive")
    if (recovery_hold or normal_global_search_pending) and int(query_generation) <= 0:
        raise ValueError("query hold requires a positive generation")
    if int(query_generation) < 0 or int(query_generation) >= 1 << 64:
        raise ValueError("query generation is outside uint64 range")
    local = observation.camera_relative_pose.rtabmap_local_transform().reshape(-1)
    intrinsics = observation.intrinsics
    hypothesis = external_loop_hypothesis
    if hypothesis is None:
        candidate_id = 0
        metric_pose = SE2Pose()
        metric_covariance = (0.0,) * 9
    else:
        candidate_id = int(hypothesis.candidate_id)
        metric_pose = hypothesis.candidate_to_current
        metric_covariance = hypothesis.covariance
    frame_flags = (
        (FRAME_FLAG_RECOVERY_HOLD if recovery_hold else 0)
        | (
            FRAME_FLAG_RECOVERY_RELOCALIZATION
            if hypothesis is not None
            and bool(hypothesis.recovery_relocalization)
            else 0
        )
        | (FRAME_FLAG_STRUCTURAL_USABLE if structural_usable else 0)
        | (
            FRAME_FLAG_VERIFIED_GRAPH_BRIDGE
            if hypothesis is not None and bool(hypothesis.verified_graph_bridge)
            else 0
        )
        | (FRAME_FLAG_NORMAL_GLOBAL_NO_MODE if normal_global_no_mode else 0)
        | (
            FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING
            if normal_global_search_pending else 0
        )
        | (
            FRAME_FLAG_RECOVERY_GLOBAL_NO_MODE
            if recovery_global_no_mode else 0
        )
    )
    meta = FRAME_META.pack(
        int(frame_id),
        float(observation.timestamp_s),
        int(intrinsics.width),
        int(intrinsics.height),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.cx),
        float(intrinsics.cy),
        *(float(value) for value in local),
        float(odometry_delta.x_m),
        float(odometry_delta.y_m),
        float(odometry_delta.yaw_rad),
        frame_flags,
        candidate_id,
        float(metric_pose.x_m),
        float(metric_pose.y_m),
        float(metric_pose.yaw_rad),
        *(float(value) for value in metric_covariance),
        int(query_generation),
    )
    payload = meta + observation.rgb.tobytes(order="C") + observation.depth_m.tobytes(order="C")
    return _packet(REQUEST_MAGIC, Command.FRAME, payload)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    if size < 0 or size > MAX_PACKET_BYTES:
        raise ValueError("invalid packet length")
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError(f"worker closed after {len(chunks)} of {size} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def read_prefix(stream: BinaryIO, expected_magic: bytes) -> tuple[int, int]:
    magic, version, code, payload_size = PREFIX.unpack(read_exact(stream, PREFIX.size))
    if magic != expected_magic:
        raise ValueError(f"invalid packet magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    if payload_size > MAX_PACKET_BYTES:
        raise ValueError("worker packet exceeds safety limit")
    return int(code), int(payload_size)


def decode_frame_packet(packet: bytes) -> DecodedFrame:
    """Decode a frame for protocol tests and non-RTAB fake workers."""

    stream = io.BytesIO(packet)
    command, payload_size = read_prefix(stream, REQUEST_MAGIC)
    if command != Command.FRAME:
        raise ValueError("packet is not a frame")
    payload = read_exact(stream, payload_size)
    if len(payload) < FRAME_META.size:
        raise ValueError("truncated frame metadata")
    values = FRAME_META.unpack_from(payload)
    frame_id, stamp, width, height = values[:4]
    fx, fy, cx, cy = values[4:8]
    local = np.asarray(values[8:20], dtype=np.float64).reshape(3, 4)
    delta = SE2Pose(*values[20:23])
    frame_flags = int(values[23])
    if frame_flags & ~FRAME_FLAGS_KNOWN:
        raise ValueError("frame packet contains unknown flags")
    if (
        frame_flags & FRAME_FLAG_NORMAL_GLOBAL_NO_MODE
        and not frame_flags & FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING
    ):
        raise ValueError("normal global no-mode requires a pending search")
    if (
        frame_flags & FRAME_FLAG_RECOVERY_GLOBAL_NO_MODE
        and not frame_flags & FRAME_FLAG_RECOVERY_HOLD
    ):
        raise ValueError("recovery global no-mode requires a recovery hold")
    query_generation = int(values[37])
    if (
        frame_flags
        & (FRAME_FLAG_RECOVERY_HOLD | FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING)
        and query_generation <= 0
    ):
        raise ValueError("query hold requires a positive generation")
    external_loop_candidate_id = int(values[24])
    external_loop_hypothesis = None
    if external_loop_candidate_id > 0:
        external_loop_hypothesis = ExternalLoopHypothesis(
            external_loop_candidate_id,
            SE2Pose(*values[25:28]),
            tuple(float(value) for value in values[28:37]),
            recovery_relocalization=bool(
                frame_flags & FRAME_FLAG_RECOVERY_RELOCALIZATION
            ),
            verified_graph_bridge=bool(
                frame_flags & FRAME_FLAG_VERIFIED_GRAPH_BRIDGE
            ),
        )
    pixels = int(width) * int(height)
    expected = FRAME_META.size + pixels * 3 + pixels * 4
    if len(payload) != expected:
        raise ValueError("frame payload size does not match dimensions")
    offset = FRAME_META.size
    rgb = np.frombuffer(
        payload, dtype=np.uint8, count=pixels * 3, offset=offset
    ).reshape(height, width, 3).copy()
    offset += pixels * 3
    depth = np.frombuffer(
        payload, dtype="<f4", count=pixels, offset=offset
    ).reshape(height, width).copy()

    # Reconstruct a lightweight observation for tests.  The exact native
    # transform is returned separately because matrix-to-quaternion recovery
    # would add a second convention to this protocol decoder.
    from .official import CameraIntrinsics, CameraRelativePose

    observation = OfficialObservation.create(
        timestamp_s=stamp,
        head_rgb=rgb,
        head_depth_m=depth,
        intrinsics=CameraIntrinsics(width, height, fx, fy, cx, cy),
        camera_relative_pose=CameraRelativePose.create((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    local.setflags(write=False)
    return DecodedFrame(
        int(frame_id), observation, local, delta, external_loop_hypothesis,
        recovery_hold=bool(frame_flags & FRAME_FLAG_RECOVERY_HOLD),
        structural_usable=bool(frame_flags & FRAME_FLAG_STRUCTURAL_USABLE),
        normal_global_no_mode=bool(
            frame_flags & FRAME_FLAG_NORMAL_GLOBAL_NO_MODE
        ),
        normal_global_search_pending=bool(
            frame_flags & FRAME_FLAG_NORMAL_GLOBAL_SEARCH_PENDING
        ),
        recovery_global_no_mode=bool(
            frame_flags & FRAME_FLAG_RECOVERY_GLOBAL_NO_MODE
        ),
        query_generation=query_generation,
    )


def response_packet(result: MapResult, status: Status = Status.OK) -> bytes:
    occupancy = np.ascontiguousarray(result.occupancy, dtype=np.int8)
    low = np.ascontiguousarray(result.low_obstacles, dtype=np.uint8)
    high = np.ascontiguousarray(result.high_obstacles, dtype=np.uint8)
    if occupancy.ndim != 2 or low.shape != occupancy.shape or high.shape != occupancy.shape:
        raise ValueError("all map layers must be equally sized 2D arrays")
    height, width = occupancy.shape
    grid_bytes = width * height
    pose_bytes = b"".join(
        POSE_RECORD.pack(int(p.node_id), float(p.x_m), float(p.y_m), float(p.yaw_rad))
        for p in result.poses
    )
    meta = RESPONSE_META.pack(
        int(result.frame_id),
        int(bool(result.tracking_ok)),
        int(bool(result.map_updated)),
        int(bool(result.loop_closed)),
        (0 if result.mapping_active else 1)
        | (2 if result.localized else 0)
        | (4 if result.visual_localized else 0)
        | (8 if result.geometric_localized else 0)
        | (16 if result.read_only_match else 0)
        | (32 if result.recovery_hold else 0)
        | (64 if result.soft_localization else 0)
        | (128 if result.uncertain_hold else 0),
        int(result.query_outcome),
        3 if result.optimizer_backend == "ceres" else 0,
        int(result.query_scope),
        int(result.query_generation),
        int(width),
        int(height),
        float(result.x_min_m),
        float(result.y_min_m),
        float(result.cell_size_m),
        float(result.current_pose.x_m),
        float(result.current_pose.y_m),
        float(result.current_pose.yaw_rad),
        float((result.native_pose or result.current_pose).x_m),
        float((result.native_pose or result.current_pose).y_m),
        float((result.native_pose or result.current_pose).yaw_rad),
        float((result.fused_odometry_pose or result.current_pose).x_m),
        float((result.fused_odometry_pose or result.current_pose).y_m),
        float((result.fused_odometry_pose or result.current_pose).yaw_rad),
        int(result.node_count),
        int(result.loop_count),
        int(result.inliers),
        int(result.features),
        int(grid_bytes),
        len(result.poses),
        int(result.ref_node_id),
        int(result.novelty_resume_reconciliation_viewpoints_remaining),
    )
    payload = meta + occupancy.tobytes() + low.tobytes() + high.tobytes() + pose_bytes
    return _packet(RESPONSE_MAGIC, status, payload)


def read_result(stream: BinaryIO) -> MapResult:
    status_value, payload_size = read_prefix(stream, RESPONSE_MAGIC)
    try:
        status = Status(status_value)
    except ValueError as exc:
        raise ValueError(f"unknown worker status {status_value}") from exc
    payload = read_exact(stream, payload_size)
    if status not in (Status.OK, Status.TRACKING_LOST):
        message = payload.decode("utf-8", errors="replace")
        raise RuntimeError(f"RTAB-Map worker error ({status.name}): {message}")
    if len(payload) < RESPONSE_META.size:
        raise ValueError("truncated result metadata")
    values = RESPONSE_META.unpack_from(payload)
    (
        frame_id,
        tracking_ok,
        map_updated,
        loop_closed,
        mode_flags,
        query_outcome,
        optimizer_backend,
        query_scope,
        query_generation,
        width,
        height,
        x_min,
        y_min,
        cell_size,
        pose_x,
        pose_y,
        pose_yaw,
        native_pose_x,
        native_pose_y,
        native_pose_yaw,
        fused_odom_pose_x,
        fused_odom_pose_y,
        fused_odom_pose_yaw,
        node_count,
        loop_count,
        inliers,
        features,
        grid_bytes,
        pose_count,
        ref_node_id,
        novelty_resume_reconciliation_viewpoints_remaining,
    ) = values
    try:
        decoded_query_outcome = QueryOutcome(query_outcome)
    except ValueError as exc:
        raise ValueError(
            f"worker returned unknown query outcome {query_outcome}"
        ) from exc
    try:
        decoded_query_scope = QueryScope(query_scope)
    except ValueError as exc:
        raise ValueError(f"worker returned unknown query scope {query_scope}") from exc
    if decoded_query_outcome is not QueryOutcome.NONE and (
        decoded_query_scope is QueryScope.NONE or int(query_generation) <= 0
    ):
        raise ValueError("worker returned an unbound query acknowledgement")
    if width * height != grid_bytes:
        raise ValueError("worker grid dimensions are inconsistent")
    expected = RESPONSE_META.size + grid_bytes * 3 + pose_count * POSE_RECORD.size
    if len(payload) != expected:
        raise ValueError("worker result payload size is inconsistent")
    offset = RESPONSE_META.size
    occupancy = np.frombuffer(
        payload, dtype=np.int8, count=grid_bytes, offset=offset
    ).reshape(height, width).copy()
    offset += grid_bytes
    low = np.frombuffer(
        payload, dtype=np.uint8, count=grid_bytes, offset=offset
    ).reshape(height, width).copy()
    offset += grid_bytes
    high = np.frombuffer(
        payload, dtype=np.uint8, count=grid_bytes, offset=offset
    ).reshape(height, width).copy()
    offset += grid_bytes
    poses = []
    for _ in range(pose_count):
        node_id, x_m, y_m, yaw_rad = POSE_RECORD.unpack_from(payload, offset)
        poses.append(PoseRecord(node_id, x_m, y_m, yaw_rad))
        offset += POSE_RECORD.size
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        if grid_bytes:
            raise ValueError("worker returned an invalid cell size")
        cell_size = 0.05
    return MapResult(
        frame_id=int(frame_id),
        tracking_ok=bool(tracking_ok) and status != Status.TRACKING_LOST,
        map_updated=bool(map_updated),
        loop_closed=bool(loop_closed),
        occupancy=occupancy,
        low_obstacles=low,
        high_obstacles=high,
        x_min_m=float(x_min),
        y_min_m=float(y_min),
        cell_size_m=float(cell_size),
        current_pose=SE2Pose(float(pose_x), float(pose_y), float(pose_yaw)),
        node_count=int(node_count),
        loop_count=int(loop_count),
        inliers=int(inliers),
        features=int(features),
        poses=tuple(poses),
        native_pose=SE2Pose(
            float(native_pose_x),
            float(native_pose_y),
            float(native_pose_yaw),
        ),
        fused_odometry_pose=SE2Pose(
            float(fused_odom_pose_x),
            float(fused_odom_pose_y),
            float(fused_odom_pose_yaw),
        ),
        mapping_active=not bool(mode_flags & 1),
        localized=bool(mode_flags & 2),
        visual_localized=bool(mode_flags & 4),
        geometric_localized=bool(mode_flags & 8),
        read_only_match=bool(mode_flags & 16),
        recovery_hold=bool(mode_flags & 32),
        soft_localization=bool(mode_flags & 64),
        uncertain_hold=bool(mode_flags & 128),
        query_outcome=decoded_query_outcome,
        query_scope=decoded_query_scope,
        query_generation=int(query_generation),
        novelty_resume_reconciliation_viewpoints_remaining=int(
            novelty_resume_reconciliation_viewpoints_remaining
        ),
        optimizer_backend={0: "toro", 3: "ceres"}.get(
            optimizer_backend, f"unknown:{optimizer_backend}"
        ),
        ref_node_id=int(ref_node_id),
    )
