"""Crash-evident writer for evaluator-compliant offline RGB-D captures.

The API deliberately has no pose, scene, or simulator handle.  A producer can
only append official head-camera observations and body-frame velocity ticks.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Sequence
import zlib

import numpy as np
from PIL import Image

from .official import OfficialObservation
from .offline import (
    ALLOWED_OBSERVATION_POLICY,
    CAPTURE_SCHEMA,
    DEPTH_DELTA_ENCODING,
    MOTION_SCHEMA,
    MotionTick,
)


class CaptureWriterError(ValueError):
    """The requested write would make the capture incomplete or ambiguous."""


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy_payload_size(array: np.ndarray) -> int:
    """Return the exact v1 NPY size without copying the array payload."""

    header = {
        "descr": np.lib.format.dtype_to_descr(array.dtype),
        "fortran_order": bool(array.flags.f_contiguous and not array.flags.c_contiguous),
        "shape": array.shape,
    }
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, header)
    return int(stream.tell() + array.nbytes)


class CaptureBundleWriter:
    """Persist a continuous, directly replayable official-observation bundle.

    Call :meth:`append_frame` once for the origin, then call :meth:`append_tick`
    for every evaluator physics tick before appending the next camera frame.
    The writer refuses a frame interval with no ticks or with inconsistent
    elapsed time, so a partial recording cannot silently claim full coverage.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        timing_tolerance_s: float = 0.008,
        idle_depth_compression: bool = True,
        idle_qvel_threshold: float = 0.001,
        delta_keyframe_interval: int = 30,
        delta_min_savings_ratio: float = 0.05,
    ) -> None:
        root = Path(path).expanduser().resolve()
        tolerance = float(timing_tolerance_s)
        if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > 0.1:
            raise CaptureWriterError("timing_tolerance_s must be within [0, 0.1]")
        if root.exists() and any(root.iterdir()):
            raise CaptureWriterError(f"capture directory is not empty: {root}")
        qvel_threshold = float(idle_qvel_threshold)
        if not math.isfinite(qvel_threshold) or not 0.0 <= qvel_threshold <= 0.1:
            raise CaptureWriterError("idle_qvel_threshold must be within [0, 0.1]")
        keyframe_interval = int(delta_keyframe_interval)
        if keyframe_interval < 1 or keyframe_interval > 300:
            raise CaptureWriterError("delta_keyframe_interval must be within [1, 300]")
        minimum_savings = float(delta_min_savings_ratio)
        if not math.isfinite(minimum_savings) or not 0.0 <= minimum_savings < 1.0:
            raise CaptureWriterError("delta_min_savings_ratio must be within [0, 1)")
        self.path = root
        self.images_dir = root / "images"
        self.motion_path = root / "motion.jsonl"
        self.manifest_path = root / "recording.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._timing_tolerance_s = tolerance
        self._idle_depth_compression = bool(idle_depth_compression)
        self._idle_qvel_threshold = qvel_threshold
        self._delta_keyframe_interval = keyframe_interval
        self._delta_min_savings_ratio = minimum_savings
        self._pending_ticks: list[MotionTick] = []
        self._frame_count = 0
        self._last_sequence: int | None = None
        self._last_timestamp_s: float | None = None
        self._previous_depth: np.ndarray | None = None
        self._previous_image_id: str | None = None
        self._delta_chain_length = 0
        self._depth_delta_frames = 0
        self._depth_keyframes = 0
        self._depth_raw_bytes = 0
        self._depth_npy_baseline_bytes = 0
        self._depth_stored_bytes = 0
        self._closed = False
        self._write_manifest()

    @property
    def payload_stats(self) -> dict[str, int | float | str]:
        saved = max(0, self._depth_npy_baseline_bytes - self._depth_stored_bytes)
        ratio = (
            float(self._depth_stored_bytes) / float(self._depth_npy_baseline_bytes)
            if self._depth_npy_baseline_bytes
            else 1.0
        )
        return {
            "depth_encoding": (
                DEPTH_DELTA_ENCODING if self._idle_depth_compression else "npy"
            ),
            "depth_delta_frames": self._depth_delta_frames,
            "depth_keyframes": self._depth_keyframes,
            "depth_raw_bytes": self._depth_raw_bytes,
            "depth_npy_baseline_bytes": self._depth_npy_baseline_bytes,
            "depth_stored_bytes": self._depth_stored_bytes,
            "depth_bytes_saved": saved,
            "depth_storage_ratio": round(ratio, 6),
        }

    def append_tick(self, base_qvel: Sequence[float], dt_s: float) -> None:
        """Append one actual body-frame velocity sample from an evaluator tick."""

        if self._closed:
            raise CaptureWriterError("capture writer is closed")
        if self._frame_count == 0:
            raise CaptureWriterError("append the origin camera frame before velocity ticks")
        self._pending_ticks.append(MotionTick.create(dt_s, base_qvel))

    def append_frame(
        self,
        observation: OfficialObservation,
        *,
        evaluator_sequence: int,
        sampled_base_qvel: Sequence[float] = (0.0, 0.0, 0.0),
        allow_idle_compression: bool = True,
    ) -> str:
        """Persist one frame and bind all pending ticks to its incoming interval."""

        if self._closed:
            raise CaptureWriterError("capture writer is closed")
        if isinstance(evaluator_sequence, bool) or not isinstance(evaluator_sequence, int):
            raise CaptureWriterError("evaluator_sequence must be an integer")
        if self._last_sequence is not None and evaluator_sequence <= self._last_sequence:
            raise CaptureWriterError("evaluator_sequence must increase strictly")
        sampled = MotionTick.create(1.0, sampled_base_qvel).base_qvel
        timestamp_s = float(observation.timestamp_s)
        if self._last_timestamp_s is None:
            if self._pending_ticks:
                raise CaptureWriterError("origin frame cannot have incoming velocity ticks")
        else:
            if timestamp_s <= self._last_timestamp_s:
                raise CaptureWriterError("camera timestamps must increase strictly")
            if not self._pending_ticks:
                raise CaptureWriterError(
                    "every camera-frame interval must contain all body qvel ticks"
                )
            frame_duration = timestamp_s - self._last_timestamp_s
            tick_duration = sum(tick.dt_s for tick in self._pending_ticks)
            if abs(frame_duration - tick_duration) > self._timing_tolerance_s:
                raise CaptureWriterError(
                    "qvel tick duration does not match the camera-frame interval: "
                    f"ticks={tick_duration:.6f}s frames={frame_duration:.6f}s"
                )

        image_id = f"img_{self._frame_count + 1:06d}"
        rgb_name = f"{image_id}.png"
        full_depth_name = f"{image_id}.depth.npy"
        delta_depth_name = f"{image_id}.depth.xor.zlib"
        metadata_name = f"{image_id}.meta.json"
        rgb_path = self.images_dir / rgb_name
        metadata_path = self.images_dir / metadata_name
        rgb_temporary = self.images_dir / f".{rgb_name}.tmp"
        depth = observation.depth_m
        delta_payload: bytes | None = None
        idle_interval = bool(self._pending_ticks) and all(
            max(abs(value) for value in tick.base_qvel) <= self._idle_qvel_threshold
            for tick in self._pending_ticks
        )
        can_delta = bool(
            self._idle_depth_compression
            and allow_idle_compression
            and idle_interval
            and self._previous_depth is not None
            and self._previous_image_id is not None
            and self._delta_chain_length < self._delta_keyframe_interval
            and self._previous_depth.shape == depth.shape
            and self._previous_depth.dtype == depth.dtype
        )
        if can_delta:
            delta = np.bitwise_xor(
                depth.view(np.uint8), self._previous_depth.view(np.uint8)
            )
            # Byte shuffle groups the four IEEE-754 byte lanes before DEFLATE.
            # Millimeter-scale depth jitter then compresses much better, while
            # XOR and the inverse transpose remain bit-exact.
            shuffled = delta.reshape(-1, 4).T
            compressor = zlib.compressobj(
                1,
                zlib.DEFLATED,
                zlib.MAX_WBITS,
                zlib.DEF_MEM_LEVEL,
                zlib.Z_RLE,
            )
            raw_delta = shuffled.tobytes()
            candidate = compressor.compress(raw_delta) + compressor.flush()
            maximum_bytes = int(depth.nbytes * (1.0 - self._delta_min_savings_ratio))
            if len(candidate) < maximum_bytes:
                delta_payload = candidate

        if delta_payload is None:
            depth_name = full_depth_name
            depth_metadata: str | dict[str, object] = depth_name
        else:
            depth_name = delta_depth_name
            depth_metadata = {
                "encoding": DEPTH_DELTA_ENCODING,
                "file": depth_name,
                "base_image_id": self._previous_image_id,
                "decoded_dtype": "<f4",
            }
        depth_path = self.images_dir / depth_name
        depth_temporary = self.images_dir / f".{depth_name}.tmp"
        try:
            Image.fromarray(observation.rgb, mode="RGB").save(rgb_temporary, format="PNG")
            with depth_temporary.open("wb") as stream:
                if delta_payload is None:
                    np.save(stream, depth, allow_pickle=False)
                else:
                    stream.write(delta_payload)
            os.replace(rgb_temporary, rgb_path)
            os.replace(depth_temporary, depth_path)
        finally:
            rgb_temporary.unlink(missing_ok=True)
            depth_temporary.unlink(missing_ok=True)

        intrinsics = observation.intrinsics
        relative_pose = observation.camera_relative_pose
        metadata = {
            "schema_version": CAPTURE_SCHEMA,
            "observation_policy": ALLOWED_OBSERVATION_POLICY,
            "image_id": image_id,
            "evaluator_sequence": evaluator_sequence,
            "timestamp_s": timestamp_s,
            "camera": {
                "image_width": intrinsics.width,
                "image_height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "robot_relative_pose": {
                    "frame": "robot_base",
                    "pos": list(relative_pose.position_m),
                    "quat": list(relative_pose.quaternion_xyzw),
                },
            },
            "rgb": {"head": rgb_name},
            "modalities": {"depth_linear": depth_metadata},
            # This snapshot is diagnostic only. Replay integrates the complete
            # tick batch below and never treats this value as interval motion.
            "robot": {"base_qvel": list(sampled)},
        }
        _atomic_json(metadata_path, metadata)

        if self._frame_count > 0:
            motion_record = {
                "schema_version": MOTION_SCHEMA,
                "image_id": image_id,
                "complete": True,
                "ticks": [
                    {"dt_s": tick.dt_s, "base_qvel": list(tick.base_qvel)}
                    for tick in self._pending_ticks
                ],
            }
            with self.motion_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(motion_record, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        self._frame_count += 1
        self._last_sequence = evaluator_sequence
        self._last_timestamp_s = timestamp_s
        self._previous_depth = depth
        self._previous_image_id = image_id
        self._depth_raw_bytes += int(depth.nbytes)
        self._depth_npy_baseline_bytes += _npy_payload_size(depth)
        self._depth_stored_bytes += int(depth_path.stat().st_size)
        if delta_payload is None:
            self._depth_keyframes += 1
            self._delta_chain_length = 0
        else:
            self._depth_delta_frames += 1
            self._delta_chain_length += 1
        self._pending_ticks.clear()
        self._write_manifest()
        return image_id

    def close(self) -> None:
        if self._closed:
            return
        if self._pending_ticks:
            raise CaptureWriterError(
                "capture has velocity ticks after the final persisted camera frame"
            )
        self._closed = True
        self._write_manifest()

    def _write_manifest(self) -> None:
        _atomic_json(
            self.manifest_path,
            {
                "schema_version": CAPTURE_SCHEMA,
                "observation_policy": ALLOWED_OBSERVATION_POLICY,
                "frame_count": self._frame_count,
                "motion_manifest": self.motion_path.name,
                "closed": self._closed,
                "payload_storage": self.payload_stats,
                "allowed_inputs": [
                    "head_rgb",
                    "head_depth_m",
                    "camera_intrinsics",
                    "camera.robot_relative_pose",
                    "robot.base_qvel",
                    "evaluator_sequence",
                    "observation_timestamp_s",
                ],
            },
        )

    def __enter__(self) -> "CaptureBundleWriter":
        return self

    def __exit__(self, exception_type, _value, _traceback) -> None:
        if exception_type is None:
            self.close()
