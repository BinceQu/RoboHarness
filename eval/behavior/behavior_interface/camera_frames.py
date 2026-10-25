"""Small, simulator-independent helpers for validating camera buffers."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np


_SEGMENTATION_KEYS = ("seg_instance_id", "instance_id_segmentation")
_SECRET_QUERY_KEYS = ("token", "key", "secret", "password", "passwd", "auth", "signature")


def to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def rgb_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = to_numpy(value)
    if (
        arr.size == 0
        or arr.ndim != 3
        or arr.shape[0] == 0
        or arr.shape[1] == 0
        or arr.shape[2] not in (3, 4)
    ):
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr[..., :3]


def convert_rgb_frame(value: Any, converter) -> Any:
    """Run a color converter only for a non-empty, well-formed RGB frame."""
    arr = rgb_array(value)
    if arr is None:
        return None
    return converter(arr)


def scalar_image_array(value: Any, *, dtype=None) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = to_numpy(value)
    if arr.size == 0:
        return None
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def vector_image_array(value: Any, *, min_channels: int = 3) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = to_numpy(value)
    if (
        arr.size == 0
        or arr.ndim != 3
        or arr.shape[0] == 0
        or arr.shape[1] == 0
        or arr.shape[2] < int(min_channels)
    ):
        return None
    return arr


def observation_value(obs: Mapping[str, Any], modality: str) -> Any:
    if modality == "seg_instance_id":
        for key in _SEGMENTATION_KEYS:
            value = obs.get(key)
            if value is not None:
                return value
        return None
    return obs.get(modality)


def observation_frame_errors(
    obs: Any,
    required_modalities: Sequence[str],
) -> list[str]:
    """Return missing, malformed, empty, or shape-mismatched modalities."""
    if not isinstance(obs, Mapping):
        return ["observation is not a mapping"]

    errors: list[str] = []
    expected_shape = None
    for modality in required_modalities:
        value = observation_value(obs, modality)
        if value is None:
            errors.append(f"{modality} missing")
            continue
        if modality == "rgb":
            arr = rgb_array(value)
        elif modality in {"depth_linear", "seg_instance_id"}:
            arr = scalar_image_array(value)
        elif modality == "normal":
            arr = vector_image_array(value)
        else:
            arr = vector_image_array(value, min_channels=1)
            if arr is None:
                arr = scalar_image_array(value)
        if arr is None:
            errors.append(f"{modality} empty or malformed")
            continue
        shape = tuple(int(v) for v in arr.shape[:2])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            errors.append(f"{modality} shape {shape} != {expected_shape}")
    return errors


def redact_camera_source(source: Any) -> str:
    """Return a log-safe camera source without credentials or secret query values."""
    text = str(source or "<unknown>")
    try:
        parts = urlsplit(text)
    except Exception:
        return text
    if not parts.scheme or not parts.netloc:
        return text

    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    if parts.username is not None:
        host = f"{parts.username}:***@{host}"
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if any(secret in key.lower() for secret in _SECRET_QUERY_KEYS):
            value = "***"
        query.append((key, value))
    return urlunsplit((parts.scheme, host, parts.path, urlencode(query), parts.fragment))


class CameraFeedHealth:
    """Thread-safe camera failure, stale-frame, and reconnect state."""

    def __init__(
        self,
        source: Any,
        backend: str,
        *,
        failure_threshold: int = 3,
        stale_after_s: float = 5.0,
        reconnect_backoff_s: float = 0.5,
        reconnect_backoff_max_s: float = 8.0,
        clock=time.monotonic,
        wall_clock=time.time,
    ):
        self.source = redact_camera_source(source)
        self.backend = str(backend or "<unknown>")
        self.failure_threshold = max(1, int(failure_threshold))
        self.stale_after_s = max(0.0, float(stale_after_s))
        self.reconnect_backoff_s = max(0.0, float(reconnect_backoff_s))
        self.reconnect_backoff_max_s = max(
            self.reconnect_backoff_s,
            float(reconnect_backoff_max_s),
        )
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._running = True
        self._opened = False
        self._degraded = False
        self._consecutive_failures = 0
        self._reconnect_count = 0
        self._reconnect_in_progress = False
        self._next_reconnect_at = 0.0
        self._last_success_mono: Optional[float] = None
        self._last_success_wall: Optional[float] = None
        self._last_error = ""
        self._last_frame_meta: dict[str, Any] = {}

    def start(self) -> None:
        with self._lock:
            self._running = True
            self._opened = False
            self._degraded = False
            self._consecutive_failures = 0
            self._reconnect_in_progress = False
            self._next_reconnect_at = 0.0
            self._last_success_mono = None
            self._last_success_wall = None
            self._last_error = ""
            self._last_frame_meta = {}

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._reconnect_in_progress = False

    def record_open(self, ok: bool, error: str = "") -> None:
        now = self._clock()
        with self._lock:
            if not self._running:
                return
            self._opened = bool(ok)
            if ok:
                self._last_error = ""
                return
            self._last_error = str(error or "camera initialization failed")
            self._consecutive_failures = max(
                self._consecutive_failures + 1,
                self.failure_threshold,
            )
            self._degraded = True
            self._next_reconnect_at = min(self._next_reconnect_at or now, now)

    def record_success(self, frame: Any) -> bool:
        arr = rgb_array(frame)
        if arr is None:
            return False
        now = self._clock()
        with self._lock:
            if not self._running:
                return False
            self._opened = True
            self._degraded = False
            self._consecutive_failures = 0
            self._reconnect_in_progress = False
            self._next_reconnect_at = 0.0
            self._last_success_mono = now
            self._last_success_wall = self._wall_clock()
            self._last_error = ""
            self._last_frame_meta = {
                "type": type(frame).__name__,
                "shape": tuple(int(v) for v in arr.shape),
                "size": int(arr.size),
                "dtype": str(arr.dtype),
            }
        return True

    def record_failure(self, reason: str) -> bool:
        """Record one failed read and return whether the feed entered degraded state."""
        now = self._clock()
        with self._lock:
            if not self._running:
                return False
            was_degraded = self._degraded
            self._consecutive_failures += 1
            self._last_error = str(reason)
            if self._consecutive_failures >= self.failure_threshold:
                self._degraded = True
                if self._next_reconnect_at <= 0.0:
                    self._next_reconnect_at = now
            return self._degraded and not was_degraded

    def stale(self, now: Optional[float] = None) -> bool:
        with self._lock:
            if self._last_success_mono is None:
                return self._opened and self._consecutive_failures > 0
            current = self._clock() if now is None else float(now)
            return current - self._last_success_mono > self.stale_after_s

    def should_reconnect(self, now: Optional[float] = None) -> bool:
        with self._lock:
            current = self._clock() if now is None else float(now)
            return bool(
                self._running
                and self._degraded
                and not self._reconnect_in_progress
                and current >= self._next_reconnect_at
            )

    def begin_reconnect(self) -> Optional[int]:
        with self._lock:
            if not self.should_reconnect():
                return None
            self._reconnect_in_progress = True
            self._reconnect_count += 1
            return self._reconnect_count

    def finish_reconnect(self, ok: bool, error: str = "") -> None:
        now = self._clock()
        with self._lock:
            if not self._running:
                self._reconnect_in_progress = False
                return
            self._reconnect_in_progress = False
            if error:
                self._last_error = str(error)
            exponent = max(0, self._reconnect_count - 1)
            delay = min(
                self.reconnect_backoff_max_s,
                self.reconnect_backoff_s * (2**exponent),
            )
            self._next_reconnect_at = now + delay
            if ok:
                self._opened = True
            else:
                self._opened = False
                self._degraded = True

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            age_s = (
                None
                if self._last_success_mono is None
                else max(0.0, now - self._last_success_mono)
            )
            return {
                "source": self.source,
                "backend": self.backend,
                "running": self._running,
                "opened": self._opened,
                "degraded": self._degraded,
                "stale": (
                    self._last_success_mono is not None
                    and age_s is not None
                    and age_s > self.stale_after_s
                ),
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "reconnect_count": self._reconnect_count,
                "reconnect_in_progress": self._reconnect_in_progress,
                "last_success_ts": self._last_success_wall,
                "last_success_age_s": age_s,
                "last_error": self._last_error,
                "frame": dict(self._last_frame_meta),
            }


def frame_description(frame: Any) -> str:
    if frame is None:
        return "read returned None"
    try:
        arr = to_numpy(frame)
        return (
            f"invalid frame type={type(frame).__name__} "
            f"shape={tuple(int(v) for v in arr.shape)} size={int(arr.size)} "
            f"dtype={arr.dtype}"
        )
    except Exception as exc:
        return f"invalid frame type={type(frame).__name__}: {type(exc).__name__}: {exc}"
