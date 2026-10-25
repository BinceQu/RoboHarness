"""Policy-side writer for the optional compliant grasp audit."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Mapping


PREDICTION_TRACE_ENV = "BEHAVIOR_EVAL_TEST_GRASP_PREDICTION_TRACE_PATH"
AUDIT_SCHEMA = "behavior_grasp_confirmation_audit_v1"


def _prediction_trace_path() -> str | None:
    raw = os.environ.get(PREDICTION_TRACE_ENV, "").strip()
    if not raw:
        return None
    return os.path.abspath(os.path.expanduser(raw))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _JSONLWriter:
    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                _jsonable(record),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            if self._fd is None:
                raise RuntimeError("grasp prediction audit writer is closed")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError("short write while appending grasp audit JSONL")
                remaining = remaining[written:]

    def close(self) -> bool:
        """Close the raw descriptor once; return whether this call closed it."""

        with self._lock:
            if self._fd is None:
                return False
            os.close(self._fd)
            self._fd = None
            return True


_writers_lock = threading.Lock()
_writers: dict[str, _JSONLWriter] = {}


def _writer(path: str) -> _JSONLWriter:
    with _writers_lock:
        writer = _writers.get(path)
        if writer is None:
            writer = _JSONLWriter(path)
            _writers[path] = writer
        return writer


def close_prediction_audit_writers() -> dict[str, Any]:
    """Close cached writers before committing a new hot-reload generation."""

    with _writers_lock:
        stale = list(_writers.items())
        _writers.clear()
    closed = 0
    errors: list[str] = []
    for path, writer in stale:
        try:
            close = getattr(writer, "close", None)
            if callable(close):
                closed += int(bool(close()))
                continue
            # A writer created by a pre-hot-reload generation has no close()
            # method. It has the same private layout, so close its descriptor
            # without retaining an old-class instance in the new generation.
            lock = getattr(writer, "_lock", None)
            with lock if lock is not None else threading.Lock():
                fd = getattr(writer, "_fd", None)
                if fd is not None:
                    os.close(fd)
                    writer._fd = None
                    closed += 1
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {"closed_count": closed, "close_errors": errors}


def record_compliant_grasp_prediction(
    tool_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Persist only the public close result without inspecting simulator state."""
    if str(tool_name) != "close_gripper":
        return None
    path = _prediction_trace_path()
    if path is None:
        return None
    record = {
        "schema": AUDIT_SCHEMA,
        "record_type": "compliant_prediction",
        "privileged_test_only": False,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "arm": str(result.get("arm") or ""),
        "grasp_confirmed": bool(result.get("grasp_confirmed", False)),
        "ok": bool(result.get("ok", False)),
        "failure_stage": result.get("failure_stage"),
        "failure_reason": result.get("failure_reason"),
        "confirmation_source": result.get("confirmation_source"),
        "action_steps": result.get("action_steps"),
        "gripper_qpos_before": result.get("gripper_qpos_before"),
        "gripper_qpos_after": result.get("gripper_qpos_after"),
        "bilateral_contact_ever_observed": result.get(
            "bilateral_contact_ever_observed"
        ),
    }
    _writer(path).append(record)
    return record


__all__ = [
    "close_prediction_audit_writers",
    "record_compliant_grasp_prediction",
]
