"""Offline RGB-D replay and deformation release gate.

The reader intentionally exposes only evaluator-compliant products from a
persisted capture. Global camera/base poses may coexist in legacy metadata,
but no code path below names or copies those fields.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterator, Optional, Sequence
import zlib

import numpy as np
from PIL import Image

from .client import RtabmapClient
from .map_audit import audit_map, render_trajectory_audit
from .official import (
    CameraIntrinsics,
    CameraRelativePose,
    OfficialObservation,
    SE2Pose,
    align_odometry_path_to_endpoint,
)
from .protocol import MapResult, QueryOutcome
from .render import render_floorplan, render_heading_up
from .validation import ThreeTurnValidator


MOTION_SCHEMA = "behavior.rtabmap_qvel_batch.v1"
LEGACY_CAPTURE_SCHEMA = "behavior.rtabmap_capture.v1"
CAPTURE_SCHEMA = "behavior.rtabmap_capture.v2"
SUPPORTED_CAPTURE_SCHEMAS = frozenset((LEGACY_CAPTURE_SCHEMA, CAPTURE_SCHEMA))
LEGACY_DEPTH_DELTA_ENCODING = "depth-xor-zlib-v1"
DEPTH_DELTA_ENCODING = "depth-xor-byte-shuffle-zlib-v1"
SUPPORTED_DEPTH_DELTA_ENCODINGS = frozenset(
    (LEGACY_DEPTH_DELTA_ENCODING, DEPTH_DELTA_ENCODING)
)
REPORT_SCHEMA = "behavior.rtabmap_offline_report.v19"
LANDMARK_REPORT_SCHEMA = "behavior.rtabmap_replay_landmarks.v1"
ALLOWED_OBSERVATION_POLICY = "official_allowlist"
SQLITE_HOT_JOURNAL_MAGIC = b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7"

class OfflineDatasetError(ValueError):
    """A persisted capture cannot be replayed without guessing its meaning."""


def _frozen_map_changed(
    frozen_occupancy: np.ndarray,
    frozen_low: np.ndarray,
    frozen_high: np.ndarray,
    current_occupancy: np.ndarray,
    current_low: np.ndarray,
    current_high: np.ndarray,
) -> bool:
    """Return whether any frozen structural layer changed."""

    if (
        frozen_occupancy.shape != current_occupancy.shape
        or frozen_low.shape != current_low.shape
        or frozen_high.shape != current_high.shape
    ):
        return True
    return not (
        np.array_equal(frozen_occupancy, current_occupancy)
        and np.array_equal(frozen_low, current_low)
        and np.array_equal(frozen_high, current_high)
    )


@dataclass
class _FrozenMapAuditState:
    """Count unauthorized frozen-map transitions, not repeated changed frames."""

    occupancy: Optional[np.ndarray] = None
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    node_count: Optional[int] = None
    x_min_m: Optional[float] = None
    y_min_m: Optional[float] = None
    cell_size_m: Optional[float] = None
    authorized_query_transactions: set[tuple[int, int, int]] = field(
        default_factory=set
    )

    def clear(self) -> None:
        self.occupancy = None
        self.low = None
        self.high = None
        self.node_count = None
        self.x_min_m = None
        self.y_min_m = None
        self.cell_size_m = None

    def observe(self, result: MapResult) -> tuple[int, int, int]:
        terminal_outcomes = {
            QueryOutcome.COMMITTED,
            QueryOutcome.BRIDGE_ONLY_DISCARDED,
        }
        first_terminal_ack = False
        if (
            result.query_outcome in terminal_outcomes
            and int(result.query_scope) != 0
            and int(result.query_generation) > 0
        ):
            transaction = (
                int(result.query_scope),
                int(result.query_generation),
                int(result.query_outcome),
            )
            first_terminal_ack = transaction not in self.authorized_query_transactions
            self.authorized_query_transactions.add(transaction)

        if not result.soft_localization:
            self.clear()
            return 0, 0, 0

        authorized = bool(
            (result.loop_closed and result.localized)
            or first_terminal_ack
        )
        map_mutations = 0
        detail_cells_added = 0
        node_growth_events = 0
        if self.occupancy is not None and not authorized:
            assert self.low is not None
            assert self.high is not None
            same_geometry = bool(
                self.occupancy.shape == result.occupancy.shape
                and self.low.shape == result.low_obstacles.shape
                and self.high.shape == result.high_obstacles.shape
                and self.x_min_m == result.x_min_m
                and self.y_min_m == result.y_min_m
                and self.cell_size_m == result.cell_size_m
            )
            base_changed = bool(
                not same_geometry
                or not np.array_equal(self.occupancy, result.occupancy)
                or not np.array_equal(self.low, result.low_obstacles)
            )
            high_changed_illegally = False
            if same_geometry:
                removed_or_rewritten = (self.high != 0) & (
                    result.high_obstacles != self.high
                )
                high_changed_illegally = bool(np.any(removed_or_rewritten))
                if not base_changed and not high_changed_illegally:
                    detail_cells_added = int(
                        np.count_nonzero(
                            (self.high == 0) & (result.high_obstacles != 0)
                        )
                    )
            map_mutations = int(base_changed or high_changed_illegally)
            node_growth_events = int(result.node_count != self.node_count)

        # Advance after every observation so one transition is never counted on
        # every subsequent frame. Authorized graph commits establish a new
        # frozen baseline; unauthorized transitions are still counted above.
        self.occupancy = result.occupancy.copy()
        self.low = result.low_obstacles.copy()
        self.high = result.high_obstacles.copy()
        self.node_count = result.node_count
        self.x_min_m = result.x_min_m
        self.y_min_m = result.y_min_m
        self.cell_size_m = result.cell_size_m
        return map_mutations, detail_cells_added, node_growth_events


def _finite_float(value: object, name: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise OfflineDatasetError(f"{name} must be numeric") from exc
    if not math.isfinite(output):
        raise OfflineDatasetError(f"{name} must be finite")
    return output


def _finite_vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise OfflineDatasetError(f"{name} must contain {size} values")
    return tuple(_finite_float(item, name) for item in value)


def _safe_relative_file(directory: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OfflineDatasetError(f"{name} must be a relative file name")
    relative = Path(value)
    if relative.is_absolute():
        raise OfflineDatasetError(f"{name} must be relative to the images directory")
    root = directory.resolve()
    candidate = (directory / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise OfflineDatasetError(f"{name} escapes the images directory")
    if not candidate.is_file():
        raise OfflineDatasetError(f"{name} does not exist: {candidate}")
    return candidate


@dataclass(frozen=True)
class MotionTick:
    dt_s: float
    base_qvel: tuple[float, float, float]

    @classmethod
    def create(cls, dt_s: object, base_qvel: object) -> "MotionTick":
        dt = _finite_float(dt_s, "motion tick dt_s")
        if dt <= 0.0 or dt > 1.0:
            raise OfflineDatasetError("motion tick dt_s must be within (0, 1]")
        velocity = _finite_vector(base_qvel, 3, "motion tick base_qvel")
        return cls(dt, (velocity[0], velocity[1], velocity[2]))


@dataclass(frozen=True)
class MotionBatch:
    ticks: tuple[MotionTick, ...]
    complete: bool
    source: str

    @property
    def duration_s(self) -> float:
        return sum(tick.dt_s for tick in self.ticks)


@dataclass(frozen=True)
class _DepthPayloadSpec:
    path: Path
    encoding: str
    shape: tuple[int, int]
    base_image_id: Optional[str] = None


class _DepthPayloadReader:
    """Decode full or lossless delta depth frames with a tiny sequential cache."""

    def __init__(self, *, cache_frames: int = 3) -> None:
        self._specs: dict[str, _DepthPayloadSpec] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_frames = max(1, int(cache_frames))

    def add(self, image_id: str, spec: _DepthPayloadSpec) -> None:
        if image_id in self._specs:
            raise OfflineDatasetError(f"duplicate depth payload for {image_id}")
        if spec.encoding in SUPPORTED_DEPTH_DELTA_ENCODINGS:
            base = spec.base_image_id
            if not base or base not in self._specs:
                raise OfflineDatasetError(
                    f"{image_id} depth delta base must be an earlier capture frame"
                )
            if self._specs[base].shape != spec.shape:
                raise OfflineDatasetError(
                    f"{image_id} depth delta shape differs from its base frame"
                )
        self._specs[image_id] = spec

    def _remember(self, image_id: str, depth: np.ndarray) -> np.ndarray:
        depth.setflags(write=False)
        self._cache[image_id] = depth
        self._cache.move_to_end(image_id)
        while len(self._cache) > self._cache_frames:
            self._cache.popitem(last=False)
        return depth

    @staticmethod
    def _inflate_exact(path: Path, expected_bytes: int) -> bytes:
        try:
            encoded = path.read_bytes()
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(encoded, expected_bytes + 1)
        except (OSError, zlib.error) as exc:
            raise OfflineDatasetError(f"cannot decode depth delta {path}: {exc}") from exc
        if (
            len(decoded) != expected_bytes
            or not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
        ):
            raise OfflineDatasetError(
                f"depth delta {path} does not decode to exactly {expected_bytes} bytes"
            )
        return decoded

    def _full_depth(self, image_id: str, spec: _DepthPayloadSpec) -> np.ndarray:
        try:
            depth = np.load(spec.path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise OfflineDatasetError(f"cannot load depth frame {spec.path}: {exc}") from exc
        if depth.shape != spec.shape or not np.issubdtype(depth.dtype, np.floating):
            raise OfflineDatasetError(
                f"{image_id} depth payload has shape {depth.shape} and dtype {depth.dtype}; "
                f"expected floating {spec.shape}"
            )
        return np.ascontiguousarray(depth)

    def load(self, image_id: str, *, copy: bool) -> np.ndarray:
        if image_id not in self._specs:
            raise OfflineDatasetError(f"unknown depth payload {image_id}")
        cached = self._cache.get(image_id)
        if cached is not None:
            self._cache.move_to_end(image_id)
            return cached.copy() if copy else cached

        chain: list[tuple[str, _DepthPayloadSpec]] = []
        cursor = image_id
        while cursor not in self._cache:
            spec = self._specs[cursor]
            if spec.encoding not in SUPPORTED_DEPTH_DELTA_ENCODINGS:
                depth = self._remember(cursor, self._full_depth(cursor, spec))
                break
            chain.append((cursor, spec))
            assert spec.base_image_id is not None
            cursor = spec.base_image_id
        else:
            depth = self._cache[cursor]
            self._cache.move_to_end(cursor)

        for delta_id, spec in reversed(chain):
            base = np.ascontiguousarray(depth, dtype=np.dtype("<f4"))
            expected_bytes = int(base.nbytes)
            delta = np.frombuffer(
                self._inflate_exact(spec.path, expected_bytes), dtype=np.uint8
            )
            if spec.encoding == DEPTH_DELTA_ENCODING:
                delta = delta.reshape(4, -1).T.reshape(-1)
            decoded_bytes = np.bitwise_xor(base.view(np.uint8).reshape(-1), delta)
            depth = decoded_bytes.view(np.dtype("<f4")).reshape(spec.shape).copy()
            depth = self._remember(delta_id, depth)

        return depth.copy() if copy else depth


@dataclass(frozen=True)
class CaptureFrame:
    image_id: str
    evaluator_sequence: int
    timestamp_s: float
    rgb_path: Path
    depth_path: Path
    intrinsics: CameraIntrinsics
    camera_relative_pose: CameraRelativePose
    sampled_base_qvel: tuple[float, float, float]
    motion: MotionBatch
    _depth_reader: Optional[_DepthPayloadReader] = field(
        default=None, repr=False, compare=False
    )

    def load_rgb(self) -> np.ndarray:
        with Image.open(self.rgb_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def load_depth(self) -> np.ndarray:
        if self._depth_reader is None:
            return np.load(self.depth_path, allow_pickle=False)
        return self._depth_reader.load(self.image_id, copy=True)

    def observation(self) -> OfficialObservation:
        rgb = self.load_rgb()
        depth = (
            np.load(self.depth_path, allow_pickle=False)
            if self._depth_reader is None
            else self._depth_reader.load(self.image_id, copy=False)
        )
        return OfficialObservation.create(
            timestamp_s=self.timestamp_s,
            head_rgb=rgb,
            head_depth_m=depth,
            intrinsics=self.intrinsics,
            camera_relative_pose=self.camera_relative_pose,
        )


@dataclass(frozen=True)
class ReplayFrameRecord:
    image_id: str
    evaluator_sequence: int
    timestamp_s: float
    motion_source: str
    motion_complete: bool
    motion_ticks: int
    tracking_ok: bool
    map_updated: bool
    loop_closed: bool
    x_m: float
    y_m: float
    yaw_rad: float
    node_count: int
    loop_count: int
    inliers: int
    features: int
    mapping_active: bool = True
    localized: bool = False
    visual_localized: bool = False
    geometric_localized: bool = False
    # True when this frame was used for transient read-only revisit tracking
    # while the session remained capable of mapping genuinely new space.
    read_only_match: bool = False
    soft_localization: bool = False
    uncertain_hold: bool = False


@dataclass(frozen=True)
class ReplayFootLandmark:
    """A user foot-point event bound to an exact evaluator tick."""

    name: str
    evaluator_sequence: int

    @classmethod
    def create(cls, name: object, evaluator_sequence: object) -> "ReplayFootLandmark":
        label = str(name or "").strip()
        if not label:
            raise OfflineDatasetError("foot landmark name must not be empty")
        if isinstance(evaluator_sequence, bool):
            raise OfflineDatasetError("foot landmark evaluator sequence must be an integer")
        try:
            sequence = int(evaluator_sequence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OfflineDatasetError(
                "foot landmark evaluator sequence must be an integer"
            ) from exc
        if sequence < 0:
            raise OfflineDatasetError(
                "foot landmark evaluator sequence must be non-negative"
            )
        return cls(label, sequence)


@dataclass(frozen=True)
class _AnchoredReplayLandmark:
    spec: ReplayFootLandmark
    anchor_node_id: Optional[int]
    anchor_local_pose: SE2Pose
    fallback_pose: SE2Pose


@dataclass(frozen=True)
class OfflineReplayReport:
    qualified_no_deformation: bool
    reasons: tuple[str, ...]
    dataset_path: str
    profile: str
    optimizer_backend: str
    trajectory_source: str
    frames_total: int
    tracking_accepted: int
    tracking_lost: int
    map_updated_frames: int
    read_only_matches: int
    mapping_frames: int
    localization_frames: int
    soft_localization_frames: int
    uncertain_hold_frames: int
    rtab_incremental_memory_inactive_frames: int
    localization_matches: int
    visual_localization_matches: int
    geometric_localization_matches: int
    mapping_mode_transitions: int
    mapping_freeze_count: int
    mapping_resume_count: int
    freeze_frame_id: Optional[int]
    localization_map_mutations: int
    localization_detail_cells_added: int
    localization_node_growth_events: int
    final_nodes: int
    final_loops: int
    known_cells: int
    motion_intervals_total: int
    motion_intervals_complete: int
    motion_coverage_ratio: float
    qvel_tick_count: int
    tool_completion_intervals: int
    missing_motion_intervals: tuple[str, ...]
    max_static_translation_drift_m: float
    max_static_yaw_drift_deg: float
    map_audit: dict[str, object]
    three_turn_report: dict[str, object]
    place_retrieval: dict[str, object]
    allowed_inputs: tuple[str, ...]
    forbidden_inputs_consumed: tuple[str, ...]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["schema_version"] = REPORT_SCHEMA
        return output


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineDatasetError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OfflineDatasetError(f"JSON root must be an object: {path}")
    return value


def _metadata_header(
    path: Path,
    images_dir: Path,
    first_sequence: int,
    physics_hz: float,
) -> dict[str, object]:
    data = _load_json(path)
    schema = data.get("schema_version")
    if schema is not None and schema not in SUPPORTED_CAPTURE_SCHEMAS:
        raise OfflineDatasetError(f"{path.name} has an unsupported capture schema")
    if data.get("observation_policy") != ALLOWED_OBSERVATION_POLICY:
        raise OfflineDatasetError(
            f"{path.name} is not marked {ALLOWED_OBSERVATION_POLICY!r}"
        )
    image_id = data.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise OfflineDatasetError(f"{path.name} has no image_id")
    sequence_value = data.get("evaluator_sequence")
    if isinstance(sequence_value, bool) or not isinstance(sequence_value, int):
        raise OfflineDatasetError(f"{path.name} has an invalid evaluator_sequence")
    camera = data.get("camera")
    rgb = data.get("rgb")
    modalities = data.get("modalities")
    robot = data.get("robot")
    if not all(isinstance(item, dict) for item in (camera, rgb, modalities, robot)):
        raise OfflineDatasetError(f"{path.name} is missing compliant observation products")
    assert isinstance(camera, dict)
    assert isinstance(rgb, dict)
    assert isinstance(modalities, dict)
    assert isinstance(robot, dict)
    relative = camera.get("robot_relative_pose")
    if not isinstance(relative, dict) or relative.get("frame") != "robot_base":
        raise OfflineDatasetError(f"{path.name} has no robot-relative head camera pose")
    width = int(_finite_float(camera.get("image_width"), "camera.image_width"))
    height = int(_finite_float(camera.get("image_height"), "camera.image_height"))
    intrinsics = CameraIntrinsics(
        width,
        height,
        _finite_float(camera.get("fx"), "camera.fx"),
        _finite_float(camera.get("fy"), "camera.fy"),
        _finite_float(camera.get("cx"), "camera.cx"),
        _finite_float(camera.get("cy"), "camera.cy"),
    )
    position = _finite_vector(relative.get("pos"), 3, "camera.robot_relative_pose.pos")
    quaternion = _finite_vector(relative.get("quat"), 4, "camera.robot_relative_pose.quat")
    sampled_qvel = _finite_vector(robot.get("base_qvel"), 3, "robot.base_qvel")
    raw_depth = modalities.get("depth_linear")
    if isinstance(raw_depth, str):
        depth_spec = _DepthPayloadSpec(
            path=_safe_relative_file(
                images_dir, raw_depth, "modalities.depth_linear"
            ),
            encoding="npy",
            shape=(height, width),
        )
    elif isinstance(raw_depth, dict):
        if schema != CAPTURE_SCHEMA:
            raise OfflineDatasetError(
                f"{path.name} uses an encoded depth payload without capture schema v2"
            )
        encoding = raw_depth.get("encoding")
        if encoding not in SUPPORTED_DEPTH_DELTA_ENCODINGS:
            raise OfflineDatasetError(f"{path.name} has an unsupported depth encoding")
        if raw_depth.get("decoded_dtype") != "<f4":
            raise OfflineDatasetError(f"{path.name} depth delta must decode to <f4")
        base_image_id = raw_depth.get("base_image_id")
        if not isinstance(base_image_id, str) or not base_image_id:
            raise OfflineDatasetError(f"{path.name} depth delta has no base_image_id")
        depth_spec = _DepthPayloadSpec(
            path=_safe_relative_file(
                images_dir, raw_depth.get("file"), "modalities.depth_linear.file"
            ),
            encoding=str(encoding),
            shape=(height, width),
            base_image_id=base_image_id,
        )
    else:
        raise OfflineDatasetError(f"{path.name} has no supported depth payload")

    return {
        "image_id": image_id,
        "evaluator_sequence": sequence_value,
        "timestamp_s": _finite_float(data["timestamp_s"], "timestamp_s")
        if "timestamp_s" in data
        else (sequence_value - first_sequence) / physics_hz,
        "rgb_path": _safe_relative_file(images_dir, rgb.get("head"), "rgb.head"),
        "depth_path": depth_spec.path,
        "depth_spec": depth_spec,
        "intrinsics": intrinsics,
        "camera_relative_pose": CameraRelativePose.create(position, quaternion),
        "sampled_base_qvel": (sampled_qvel[0], sampled_qvel[1], sampled_qvel[2]),
    }


def _load_motion_manifest(path: Optional[Path]) -> dict[str, MotionBatch]:
    if path is None:
        return {}
    batches: dict[str, MotionBatch] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OfflineDatasetError(f"cannot read motion manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfflineDatasetError(f"invalid motion JSONL line {line_number}: {exc}") from exc
        if not isinstance(item, dict) or item.get("schema_version") != MOTION_SCHEMA:
            raise OfflineDatasetError(f"motion line {line_number} has an unsupported schema")
        image_id = item.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in batches:
            raise OfflineDatasetError(f"motion line {line_number} has an invalid image_id")
        raw_ticks = item.get("ticks")
        if not isinstance(raw_ticks, list):
            raise OfflineDatasetError(f"motion line {line_number} ticks must be a list")
        complete = item.get("complete")
        if not isinstance(complete, bool):
            raise OfflineDatasetError(
                f"motion line {line_number} complete must be a boolean"
            )
        ticks = []
        for raw_tick in raw_ticks:
            if not isinstance(raw_tick, dict):
                raise OfflineDatasetError(f"motion line {line_number} has a non-object tick")
            ticks.append(MotionTick.create(raw_tick.get("dt_s"), raw_tick.get("base_qvel")))
        batches[image_id] = MotionBatch(
            tuple(ticks), complete, "qvel_manifest"
        )
    return batches


def _tool_completion_batches(run_dir: Path) -> dict[str, MotionBatch]:
    """Extract only officially returned completion quantities from tool logs."""

    batches: dict[str, MotionBatch] = {}
    turns_dir = run_dir / "turns"
    if not turns_dir.is_dir():
        return batches
    for path in sorted(turns_dir.glob("turn_*.json")):
        data = _load_json(path)
        if data.get("status") != "succeeded":
            continue
        output = data.get("output")
        if not isinstance(output, dict):
            continue
        response = output.get("response")
        if not isinstance(response, dict) or response.get("ok") is not True:
            continue
        image_id = response.get("image_id")
        actual = response.get("actual")
        if not isinstance(image_id, str) or not isinstance(actual, dict):
            continue
        forward = _finite_float(actual.get("forward_m", 0.0), "actual.forward_m")
        lateral = _finite_float(actual.get("translation_m", 0.0), "actual.translation_m")
        spin = math.radians(_finite_float(actual.get("spin_deg", 0.0), "actual.spin_deg"))
        ticks: list[MotionTick] = []
        if math.hypot(forward, lateral) > 1e-12:
            ticks.append(MotionTick.create(1.0, (forward, lateral, 0.0)))
        if abs(spin) > 1e-12:
            ticks.append(MotionTick.create(1.0, (0.0, 0.0, spin)))
        batches[image_id] = MotionBatch(tuple(ticks), True, "tool_actual")
    return batches


class BehaviorCaptureDataset:
    """Strict reader for persisted BEHAVIOR head RGB-D observations."""

    def __init__(
        self,
        path: Path | str,
        *,
        physics_hz: float = 60.0,
        motion_jsonl: Optional[Path | str] = None,
        use_tool_completions: bool = False,
    ) -> None:
        run_dir = Path(path).expanduser().resolve()
        recording_manifest = None
        recording_path = run_dir / "recording.json"
        if recording_path.is_file():
            candidate_manifest = _load_json(recording_path)
            if candidate_manifest.get("schema_version") in SUPPORTED_CAPTURE_SCHEMAS:
                if candidate_manifest.get("observation_policy") != ALLOWED_OBSERVATION_POLICY:
                    raise OfflineDatasetError("capture manifest has an invalid observation policy")
                if candidate_manifest.get("closed") is not True:
                    raise OfflineDatasetError("continuous capture was not closed cleanly")
                recording_manifest = candidate_manifest
        images_dir = run_dir / "images" if (run_dir / "images").is_dir() else run_dir
        if not images_dir.is_dir():
            raise OfflineDatasetError(f"capture images directory does not exist: {images_dir}")
        hz = _finite_float(physics_hz, "physics_hz")
        if hz <= 0.0:
            raise OfflineDatasetError("physics_hz must be positive")
        metadata_paths = sorted(images_dir.glob("img_*.meta.json"))
        if not metadata_paths:
            raise OfflineDatasetError(f"no img_*.meta.json files found in {images_dir}")
        preliminary = []
        for metadata_path in metadata_paths:
            data = _load_json(metadata_path)
            sequence = data.get("evaluator_sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise OfflineDatasetError(f"{metadata_path.name} has an invalid evaluator_sequence")
            preliminary.append((sequence, metadata_path))
        preliminary.sort(key=lambda item: item[0])
        sequences = [item[0] for item in preliminary]
        if len(set(sequences)) != len(sequences):
            raise OfflineDatasetError("evaluator_sequence values must be unique")
        first_sequence = sequences[0]

        if motion_jsonl:
            manifest_path = Path(motion_jsonl).expanduser().resolve()
        else:
            default_manifest = run_dir / "motion.jsonl"
            manifest_path = default_manifest if default_manifest.is_file() else None
        motion = _load_motion_manifest(manifest_path)
        completions = _tool_completion_batches(run_dir) if use_tool_completions else {}
        frames = []
        known_ids = set()
        depth_reader = _DepthPayloadReader()
        for index, (_sequence, metadata_path) in enumerate(preliminary):
            header = _metadata_header(metadata_path, images_dir, first_sequence, hz)
            image_id = str(header["image_id"])
            if image_id in known_ids:
                raise OfflineDatasetError(f"duplicate image_id {image_id}")
            known_ids.add(image_id)
            if index == 0:
                batch = MotionBatch((), True, "origin")
            elif image_id in motion:
                batch = motion[image_id]
            elif image_id in completions:
                batch = completions[image_id]
            else:
                batch = MotionBatch((), False, "missing")
            depth_spec = header.pop("depth_spec")
            assert isinstance(depth_spec, _DepthPayloadSpec)
            depth_reader.add(image_id, depth_spec)
            frames.append(
                CaptureFrame(motion=batch, _depth_reader=depth_reader, **header)
            )
        unknown_motion = set(motion) - known_ids
        if unknown_motion:
            raise OfflineDatasetError(
                "motion manifest references unknown images: " + ", ".join(sorted(unknown_motion))
            )
        self.path = run_dir
        self.images_dir = images_dir
        self.frames = tuple(frames)
        if recording_manifest is not None and recording_manifest.get("frame_count") != len(frames):
            raise OfflineDatasetError("capture manifest frame count does not match its images")
        timestamps = [frame.timestamp_s for frame in self.frames]
        if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
            raise OfflineDatasetError("capture timestamps must increase strictly")

    def __iter__(self) -> Iterator[CaptureFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)


def _pose_delta(first: SE2Pose, second: SE2Pose) -> SE2Pose:
    return first.relative_to(second)


def _anchor_replay_landmark(
    result: MapResult,
    spec: ReplayFootLandmark,
    pose: SE2Pose,
) -> _AnchoredReplayLandmark:
    anchors = tuple(record for record in result.poses if record.node_id > 0)
    if not anchors:
        return _AnchoredReplayLandmark(spec, None, SE2Pose(), pose)
    anchor = next(
        (record for record in anchors if record.node_id == result.ref_node_id),
        max(anchors, key=lambda record: record.node_id),
    )
    anchor_pose = SE2Pose(anchor.x_m, anchor.y_m, anchor.yaw_rad)
    return _AnchoredReplayLandmark(
        spec,
        anchor.node_id,
        anchor_pose.relative_to(pose),
        pose,
    )


def _resolve_replay_landmark(
    result: MapResult,
    landmark: _AnchoredReplayLandmark,
) -> SE2Pose:
    if landmark.anchor_node_id is not None:
        for record in result.poses:
            if record.node_id == landmark.anchor_node_id:
                return SE2Pose(
                    record.x_m,
                    record.y_m,
                    record.yaw_rad,
                ).compose(landmark.anchor_local_pose)
    return landmark.fallback_pose


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_sqlite_database_clean(database_path: Path) -> None:
    """Reject a replay DB that still needs recovery after worker exit."""

    database = database_path.resolve()
    if not database.is_file():
        raise OfflineDatasetError(f"RTAB-Map database is missing after shutdown: {database}")
    journal = Path(f"{database}-journal")
    try:
        if journal.is_file():
            with journal.open("rb") as stream:
                journal_header = stream.read(8)
        else:
            journal_header = b""
    except OSError as exc:
        raise OfflineDatasetError(
            f"cannot inspect RTAB-Map SQLite journal after shutdown: {journal}"
        ) from exc
    if journal_header == SQLITE_HOT_JOURNAL_MAGIC:
        raise OfflineDatasetError(
            "RTAB-Map worker exited with a hot SQLite journal; replay artifacts "
            "are not crash-clean"
        )
    sqlite_sidecars = tuple(
        Path(f"{database}{suffix}") for suffix in ("-journal", "-wal", "-shm")
    )
    residual_sidecars = tuple(path for path in sqlite_sidecars if path.exists())
    if residual_sidecars:
        raise OfflineDatasetError(
            "RTAB-Map worker exited with residual SQLite sidecars in DELETE mode: "
            + ", ".join(str(path) for path in residual_sidecars)
        )

    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        rows = connection.execute("PRAGMA quick_check").fetchall()
    except sqlite3.Error as exc:
        raise OfflineDatasetError(
            f"RTAB-Map database failed read-only integrity verification: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if rows != [("ok",)]:
        raise OfflineDatasetError(
            f"RTAB-Map database failed PRAGMA quick_check: {rows!r}"
        )


def replay_capture(
    dataset: BehaviorCaptureDataset,
    output_dir: Path | str,
    *,
    database_path: Optional[Path | str] = None,
    size_px: int = 720,
    span_m: float = 16.0,
    save_every: int = 0,
    profile: str = "official",
    feature_backend: Optional[str] = None,
    cuda_device: Optional[str | int] = None,
    foot_landmarks: Sequence[ReplayFootLandmark] = (),
    fixed_north_up: bool = False,
    overwrite: bool = False,
) -> OfflineReplayReport:
    """Replay one capture without importing or calling the web interface."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "minimap_frames"
    log_path = output / "worker.log"
    db_path = Path(database_path).expanduser().resolve() if database_path else output / "rtabmap.db"
    minimap_path = output / "final_minimap.png"
    floorplan_path = output / "final_floorplan.png"
    evidence_floorplan_path = output / "final_floorplan_evidence.png"
    audit_path = output / "map_audit.json"
    arrays_path = output / "map_arrays.npz"
    trajectory_path = output / "trajectory.json"
    online_trajectory_path = output / "trajectory_online_frames.json"
    tick_trajectory_path = output / "trajectory_qvel_ticks.json"
    graph_trajectory_path = output / "graph_nodes.json"
    landmarks_path = output / "landmarks.json"
    report_path = output / "report.json"
    database_sidecars = tuple(
        Path(f"{db_path}{suffix}") for suffix in ("-journal", "-wal", "-shm")
    )
    query_terminal_ledger_path = Path(f"{db_path}.query-terminals-v1")
    deprecated_outputs = (
        output / "trajectory_pre_global_loop.json",
        output / "trajectory_pre_wall.json",
        output / "final_floorplan_raw.png",
        output / "floorplan_audit.json",
    )
    known_outputs = (
        log_path,
        db_path,
        *database_sidecars,
        query_terminal_ledger_path,
        minimap_path,
        floorplan_path,
        evidence_floorplan_path,
        audit_path,
        arrays_path,
        trajectory_path,
        online_trajectory_path,
        tick_trajectory_path,
        graph_trajectory_path,
        landmarks_path,
        report_path,
        *deprecated_outputs,
    )
    existing = [path for path in known_outputs if path.exists()]
    if frame_dir.is_dir() and any(frame_dir.glob("*.png")):
        existing.append(frame_dir)
    if existing and not overwrite:
        raise OfflineDatasetError(
            "output already contains a prior replay; choose a new directory or pass overwrite=True"
        )
    if overwrite:
        for path in known_outputs:
            if path.is_file():
                path.unlink()
        if frame_dir.is_dir():
            for path in frame_dir.glob("*.png"):
                path.unlink()
    if save_every > 0:
        frame_dir.mkdir(parents=True, exist_ok=True)
    landmark_specs = tuple(foot_landmarks)
    if any(not isinstance(item, ReplayFootLandmark) for item in landmark_specs):
        raise OfflineDatasetError(
            "foot_landmarks must contain ReplayFootLandmark values"
        )
    landmark_names = [item.name for item in landmark_specs]
    if len(set(landmark_names)) != len(landmark_names):
        raise OfflineDatasetError("foot landmark names must be unique")
    if landmark_specs and len(dataset):
        first_sequence = dataset.frames[0].evaluator_sequence
        last_sequence = dataset.frames[-1].evaluator_sequence
        outside = tuple(
            item
            for item in landmark_specs
            if not first_sequence <= item.evaluator_sequence <= last_sequence
        )
        if outside:
            raise OfflineDatasetError(
                "foot landmark evaluator sequence is outside the replay: "
                + ", ".join(
                    f"{item.name}@{item.evaluator_sequence}" for item in outside
                )
            )
    landmark_specs_by_sequence: dict[int, list[ReplayFootLandmark]] = {}
    for item in landmark_specs:
        landmark_specs_by_sequence.setdefault(item.evaluator_sequence, []).append(item)
    pending_landmark_odometry: dict[str, tuple[ReplayFootLandmark, SE2Pose]] = {}
    anchored_landmarks: dict[str, _AnchoredReplayLandmark] = {}
    landmark_samples: dict[str, list[dict[str, object]]] = {
        item.name: [] for item in landmark_specs
    }
    previous_evaluator_sequence: Optional[int] = None
    validator = ThreeTurnValidator()
    records: list[ReplayFrameRecord] = []
    trajectory_samples: list[SE2Pose] = []
    pending_odometry_poses: list[SE2Pose] = []
    last_tracked_odometry_pose: Optional[SE2Pose] = None
    last_tracked_map_pose: Optional[SE2Pose] = None
    final_result: Optional[MapResult] = None
    previous_result: Optional[MapResult] = None
    accepted = 0
    map_updates = 0
    read_only_matches = 0
    mapping_frames = 0
    localization_frames = 0
    soft_localization_frames = 0
    uncertain_hold_frames = 0
    rtab_incremental_memory_inactive_frames = 0
    localization_matches = 0
    visual_localization_matches = 0
    geometric_localization_matches = 0
    mapping_mode_transitions = 0
    mapping_freeze_count = 0
    mapping_resume_count = 0
    freeze_frame_id: Optional[int] = None
    previous_soft_localization: Optional[bool] = None
    frozen_map_audit = _FrozenMapAuditState()
    localization_map_mutations = 0
    localization_detail_cells_added = 0
    localization_node_growth_events = 0
    complete_intervals = 0
    qvel_ticks = 0
    tool_intervals = 0
    missing_intervals = []
    max_static_translation = 0.0
    max_static_yaw = 0.0
    place_retrieval: dict[str, object] = {"enabled": False}

    with RtabmapClient(
        database_path=db_path,
        log_path=log_path,
        profile=profile,
        feature_backend=feature_backend,
        cuda_device=cuda_device,
    ) as client:
        for index, frame in enumerate(dataset):
            odom_before = client.odometry.snapshot()
            if previous_evaluator_sequence is None:
                for item in landmark_specs_by_sequence.get(
                    frame.evaluator_sequence, ()
                ):
                    pending_landmark_odometry[item.name] = (item, odom_before)
            else:
                crossed = tuple(
                    item
                    for sequence, items in landmark_specs_by_sequence.items()
                    if previous_evaluator_sequence < sequence <= frame.evaluator_sequence
                    for item in items
                    if item.name not in anchored_landmarks
                    and item.name not in pending_landmark_odometry
                )
                sequence_span = frame.evaluator_sequence - previous_evaluator_sequence
                if crossed and (
                    not frame.motion.complete
                    or len(frame.motion.ticks) != sequence_span
                ):
                    raise OfflineDatasetError(
                        "cannot bind foot landmark to an exact evaluator tick: "
                        f"{frame.image_id} spans {sequence_span} ticks but records "
                        f"{len(frame.motion.ticks)}"
                    )
            for tick_index, tick in enumerate(frame.motion.ticks, 1):
                client.advance_odometry(tick.base_qvel, tick.dt_s)
                tick_pose = client.odometry.snapshot()
                pending_odometry_poses.append(tick_pose)
                if previous_evaluator_sequence is not None:
                    sequence = previous_evaluator_sequence + tick_index
                    for item in landmark_specs_by_sequence.get(sequence, ()):
                        pending_landmark_odometry[item.name] = (item, tick_pose)
            odom_after = client.odometry.snapshot()
            if index > 0:
                if frame.motion.complete:
                    complete_intervals += 1
                else:
                    missing_intervals.append(frame.image_id)
                if frame.motion.source == "qvel_manifest":
                    qvel_ticks += len(frame.motion.ticks)
                elif frame.motion.source == "tool_actual":
                    tool_intervals += 1
            result = client.submit(frame.observation())
            if result.tracking_ok and not result.recovery_hold:
                for name, (spec, event_odometry_pose) in tuple(
                    pending_landmark_odometry.items()
                ):
                    mapped_pose = result.current_pose.compose(
                        odom_after.relative_to(event_odometry_pose)
                    )
                    anchored_landmarks[name] = _anchor_replay_landmark(
                        result,
                        spec,
                        mapped_pose,
                    )
                    pending_landmark_odometry.pop(name, None)
            if result.tracking_ok:
                if pending_odometry_poses:
                    mapped_ticks = align_odometry_path_to_endpoint(
                        pending_odometry_poses,
                        odom_after,
                        result.current_pose,
                    )
                elif not trajectory_samples:
                    mapped_ticks = (result.current_pose,)
                else:
                    mapped_ticks = ()
                trajectory_samples.extend(mapped_ticks)
                pending_odometry_poses.clear()
                last_tracked_odometry_pose = odom_after
                last_tracked_map_pose = result.current_pose
            validator.observe(result, odom_after, frame.timestamp_s)
            accepted += int(result.tracking_ok)
            map_updates += int(result.map_updated)
            read_only_matches += int(result.read_only_match)
            mapping_frames += int(result.map_write_enabled)
            localization_frames += int(result.soft_localization)
            soft_localization_frames += int(result.soft_localization)
            uncertain_hold_frames += int(result.uncertain_hold)
            rtab_incremental_memory_inactive_frames += int(
                not result.mapping_active
            )
            localization_matches += int(result.localized)
            visual_localization_matches += int(result.visual_localized)
            geometric_localization_matches += int(result.geometric_localized)
            if (
                previous_soft_localization is not None
                and result.soft_localization != previous_soft_localization
            ):
                mapping_mode_transitions += 1
                if result.soft_localization:
                    mapping_freeze_count += 1
                else:
                    mapping_resume_count += 1
            previous_soft_localization = result.soft_localization
            if result.soft_localization and freeze_frame_id is None:
                freeze_frame_id = result.frame_id
            map_mutations, detail_added, node_growth_events = (
                frozen_map_audit.observe(result)
            )
            localization_map_mutations += map_mutations
            localization_detail_cells_added += detail_added
            localization_node_growth_events += node_growth_events
            motion_delta = _pose_delta(odom_before, odom_after)
            is_static = (
                frame.motion.complete
                and math.hypot(motion_delta.x_m, motion_delta.y_m) <= 1e-4
                and abs(motion_delta.yaw_rad) <= math.radians(0.01)
            )
            if (
                is_static
                and previous_result is not None
                and result.tracking_ok
                and previous_result.tracking_ok
            ):
                drift = _pose_delta(previous_result.current_pose, result.current_pose)
                max_static_translation = max(
                    max_static_translation, math.hypot(drift.x_m, drift.y_m)
                )
                max_static_yaw = max(max_static_yaw, abs(math.degrees(drift.yaw_rad)))
            records.append(
                ReplayFrameRecord(
                    frame.image_id,
                    frame.evaluator_sequence,
                    frame.timestamp_s,
                    frame.motion.source,
                    frame.motion.complete,
                    len(frame.motion.ticks),
                    result.tracking_ok,
                    result.map_updated,
                    result.loop_closed,
                    result.current_pose.x_m,
                    result.current_pose.y_m,
                    result.current_pose.yaw_rad,
                    result.node_count,
                    result.loop_count,
                    result.inliers,
                    result.features,
                    result.mapping_active,
                    result.localized,
                    result.visual_localized,
                    result.geometric_localized,
                    result.read_only_match,
                    result.soft_localization,
                    result.uncertain_hold,
                )
            )
            if save_every > 0 and (index % save_every == 0 or index + 1 == len(dataset)):
                resolved_landmarks = {
                    name: _resolve_replay_landmark(result, landmark)
                    for name, landmark in anchored_landmarks.items()
                }
                for name, pose in resolved_landmarks.items():
                    landmark_samples[name].append(
                        {
                            "image_id": frame.image_id,
                            "evaluator_sequence": frame.evaluator_sequence,
                            "frame_id": result.frame_id,
                            "loop_count": result.loop_count,
                            "x_m": pose.x_m,
                            "y_m": pose.y_m,
                        }
                    )
                render_heading_up(
                    result,
                    size_px=size_px,
                    span_m=span_m,
                    heading_up=not fixed_north_up,
                    places=(
                        (pose.x_m, pose.y_m, name)
                        for name, pose in resolved_landmarks.items()
                    ),
                ).save(
                    frame_dir / f"{index + 1:06d}_{frame.image_id}.png"
                )
            previous_result = result
            final_result = result
            previous_evaluator_sequence = frame.evaluator_sequence
        place_retrieval = client.place_retrieval_statistics

    if final_result is None:
        raise OfflineDatasetError("capture contains no frames")
    if pending_landmark_odometry or len(anchored_landmarks) != len(landmark_specs):
        unresolved = set(pending_landmark_odometry) | (
            set(landmark_names) - set(anchored_landmarks)
        )
        raise OfflineDatasetError(
            "SLAM produced no reliable graph anchor for foot landmark(s): "
            + ", ".join(sorted(unresolved))
        )
    _verify_sqlite_database_clean(db_path)
    if pending_odometry_poses:
        if last_tracked_odometry_pose is None or last_tracked_map_pose is None:
            raise OfflineDatasetError(
                "SLAM never accepted a pose to anchor the complete qvel trajectory"
            )
        trailing_ticks = tuple(
            last_tracked_map_pose.compose(
                last_tracked_odometry_pose.relative_to(pose)
            )
            for pose in pending_odometry_poses
        )
        trajectory_samples.extend(trailing_ticks)
    # This is the pose returned online at each tick endpoint. Do not rewrite
    # history using the final graph: that would hide whether localization was
    # actually stable when the agent consumed the map.
    complete_trajectory = tuple(trajectory_samples)
    trajectory_source = "online_rtabmap_pose_qvel_prior_with_rgbd_localization"
    map_audit = audit_map(final_result, complete_trajectory)
    final_timestamp = dataset.frames[-1].timestamp_s
    three_turn = validator.report(final_timestamp)
    intervals = max(0, len(dataset) - 1)
    coverage = 1.0 if intervals == 0 else complete_intervals / intervals
    known_cells = int(np.count_nonzero(final_result.occupancy >= 0))
    reasons = []
    if complete_intervals != intervals:
        reasons.append(
            f"motion coverage {complete_intervals}/{intervals}; every frame interval "
            "needs qvel ticks or an official completion"
        )
    reasons.extend(three_turn.session_reasons)
    if three_turn.complete_in_place_sequence_observed:
        reasons.extend(
            reason
            for reason in three_turn.reasons
            if reason not in three_turn.session_reasons
        )
    if accepted == 0 or known_cells == 0:
        reasons.append("SLAM produced no tracked occupied/free map")
    if rtab_incremental_memory_inactive_frames:
        reasons.append(
            "native RTAB-Map left incremental memory on "
            f"{rtab_incremental_memory_inactive_frames} frames"
        )
    if localization_map_mutations:
        reasons.append(
            "frozen structural map changed in "
            f"{localization_map_mutations} unauthorized localization events"
        )
    if localization_node_growth_events:
        reasons.append(
            "permanent node count changed in "
            f"{localization_node_growth_events} unauthorized localization events"
        )
    if max_static_translation > 0.01:
        reasons.append(
            f"static-frame translation drift {max_static_translation:.4f}m > 0.01m"
        )
    if max_static_yaw > 0.2:
        reasons.append(f"static-frame yaw drift {max_static_yaw:.3f}deg > 0.2deg")
    reasons.extend(f"map audit: {reason}" for reason in map_audit.reasons)

    final_landmarks = {
        name: _resolve_replay_landmark(final_result, landmark)
        for name, landmark in anchored_landmarks.items()
    }
    render_heading_up(
        final_result,
        size_px=size_px,
        span_m=span_m,
        heading_up=not fixed_north_up,
        places=(
            (pose.x_m, pose.y_m, name)
            for name, pose in final_landmarks.items()
        ),
    ).save(minimap_path)
    render_floorplan(final_result, size_px=size_px).save(evidence_floorplan_path)
    render_trajectory_audit(
        final_result, complete_trajectory, size_px=size_px
    ).save(floorplan_path)
    _write_json(audit_path, map_audit.to_dict())
    np.savez_compressed(
        arrays_path,
        occupancy=final_result.occupancy,
        low_obstacles=final_result.low_obstacles,
        high_obstacles=final_result.high_obstacles,
        x_min_m=np.float64(final_result.x_min_m),
        y_min_m=np.float64(final_result.y_min_m),
        cell_size_m=np.float64(final_result.cell_size_m),
        current_pose=np.asarray(
            [
                final_result.current_pose.x_m,
                final_result.current_pose.y_m,
                final_result.current_pose.yaw_rad,
            ],
            dtype=np.float64,
        ),
    )
    _write_json(
        trajectory_path,
        [asdict(record) for record in records],
    )
    _write_json(online_trajectory_path, [asdict(record) for record in records])
    _write_json(
        tick_trajectory_path,
        [asdict(pose) for pose in complete_trajectory],
    )
    _write_json(
        graph_trajectory_path,
        [asdict(pose) for pose in final_result.poses],
    )
    if landmark_specs:
        _write_json(
            landmarks_path,
            {
                "schema_version": LANDMARK_REPORT_SCHEMA,
                "landmarks": [
                    {
                        "name": item.name,
                        "evaluator_sequence": item.evaluator_sequence,
                        "anchor_node_id": anchored_landmarks[item.name].anchor_node_id,
                        "anchor_local_pose": asdict(
                            anchored_landmarks[item.name].anchor_local_pose
                        ),
                        "fallback_pose": asdict(
                            anchored_landmarks[item.name].fallback_pose
                        ),
                        "resolved_final_pose": asdict(final_landmarks[item.name]),
                        "samples": landmark_samples[item.name],
                    }
                    for item in landmark_specs
                ],
            },
        )
    three_turn_dict = asdict(three_turn)
    report = OfflineReplayReport(
        qualified_no_deformation=not reasons,
        reasons=tuple(reasons),
        dataset_path=str(dataset.path),
        profile=profile,
        optimizer_backend=final_result.optimizer_backend,
        trajectory_source=trajectory_source,
        frames_total=len(dataset),
        tracking_accepted=accepted,
        tracking_lost=len(dataset) - accepted,
        map_updated_frames=map_updates,
        read_only_matches=read_only_matches,
        mapping_frames=mapping_frames,
        localization_frames=localization_frames,
        soft_localization_frames=soft_localization_frames,
        uncertain_hold_frames=uncertain_hold_frames,
        rtab_incremental_memory_inactive_frames=(
            rtab_incremental_memory_inactive_frames
        ),
        localization_matches=localization_matches,
        visual_localization_matches=visual_localization_matches,
        geometric_localization_matches=geometric_localization_matches,
        mapping_mode_transitions=mapping_mode_transitions,
        mapping_freeze_count=mapping_freeze_count,
        mapping_resume_count=mapping_resume_count,
        freeze_frame_id=freeze_frame_id,
        localization_map_mutations=localization_map_mutations,
        localization_detail_cells_added=localization_detail_cells_added,
        localization_node_growth_events=localization_node_growth_events,
        final_nodes=final_result.node_count,
        final_loops=final_result.loop_count,
        known_cells=known_cells,
        motion_intervals_total=intervals,
        motion_intervals_complete=complete_intervals,
        motion_coverage_ratio=coverage,
        qvel_tick_count=qvel_ticks,
        tool_completion_intervals=tool_intervals,
        missing_motion_intervals=tuple(missing_intervals),
        max_static_translation_drift_m=max_static_translation,
        max_static_yaw_drift_deg=max_static_yaw,
        map_audit=map_audit.to_dict(),
        three_turn_report=three_turn_dict,
        place_retrieval=place_retrieval,
        allowed_inputs=(
            "head_rgb",
            "head_depth_m",
            "camera_intrinsics",
            "camera.robot_relative_pose",
            "robot.base_qvel",
            "evaluator_sequence",
            "tool.response.actual.forward_m",
            "tool.response.actual.translation_m",
            "tool.response.actual.spin_deg",
        ),
        forbidden_inputs_consumed=(),
        artifacts={
            "minimap": str(minimap_path),
            "floorplan": str(floorplan_path),
            "floorplan_evidence": str(evidence_floorplan_path),
            "map_audit": str(audit_path),
            "map_arrays": str(arrays_path),
            "trajectory": str(trajectory_path),
            "trajectory_online_frames": str(online_trajectory_path),
            "trajectory_qvel_ticks": str(tick_trajectory_path),
            "graph_nodes": str(graph_trajectory_path),
            **({"landmarks": str(landmarks_path)} if landmark_specs else {}),
            "worker_log": str(log_path),
            "database": str(db_path),
            **(
                {"query_terminal_ledger": str(query_terminal_ledger_path)}
                if query_terminal_ledger_path.is_file()
                else {}
            ),
            "report": str(report_path),
        },
    )
    _write_json(report_path, report.to_dict())
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline compliant RGB-D replay through the independent RTAB-Map worker"
    )
    parser.add_argument("capture", type=Path, help="run directory or its images directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-jsonl", type=Path)
    parser.add_argument("--use-tool-completions", action="store_true")
    parser.add_argument("--physics-hz", type=float, default=60.0)
    parser.add_argument("--size-px", type=int, default=720)
    parser.add_argument("--span-m", type=float, default=16.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="replay only the first N capture frames (for reproducible prefixes)",
    )
    parser.add_argument(
        "--foot-landmark",
        action="append",
        default=[],
        metavar="NAME@EVALUATOR_SEQUENCE",
        help="render a graph-anchored foot mark at an exact recorded evaluator tick",
    )
    parser.add_argument(
        "--fixed-north-up",
        action="store_true",
        help="render replay minimaps in the stable map frame",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "official",
            "sparse-rgbd",
            "sparse-icp",
            "native-robust",
            "native-robust-ceres",
        ),
        default="official",
    )
    parser.add_argument(
        "--feature-backend",
        choices=("cpu", "kornia-sift"),
        help="visual feature backend (default: client environment or cpu)",
    )
    parser.add_argument(
        "--cuda-device",
        help="physical GPU id exposed only to the RTAB-Map worker",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-stable-three-turns",
        action="store_true",
        help="exit with status 2 unless the complete no-deformation gate passes",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    dataset = BehaviorCaptureDataset(
        args.capture,
        physics_hz=args.physics_hz,
        motion_jsonl=args.motion_jsonl,
        use_tool_completions=args.use_tool_completions,
    )
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise OfflineDatasetError("--max-frames must be a positive integer")
        if args.max_frames > len(dataset):
            raise OfflineDatasetError(
                f"--max-frames {args.max_frames} exceeds capture length {len(dataset)}"
            )
        dataset.frames = dataset.frames[: args.max_frames]
    landmarks = []
    for raw in args.foot_landmark:
        name, separator, sequence = str(raw).rpartition("@")
        if not separator:
            raise OfflineDatasetError(
                "--foot-landmark must use NAME@EVALUATOR_SEQUENCE"
            )
        landmarks.append(ReplayFootLandmark.create(name, sequence))
    report = replay_capture(
        dataset,
        args.output_dir,
        size_px=args.size_px,
        span_m=args.span_m,
        save_every=max(0, args.save_every),
        profile=args.profile,
        feature_backend=args.feature_backend,
        cuda_device=args.cuda_device,
        foot_landmarks=landmarks,
        fixed_north_up=args.fixed_north_up,
        overwrite=args.overwrite,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.require_stable_three_turns and (
        not report.qualified_no_deformation
        or report.three_turn_report.get("passed") is not True
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
