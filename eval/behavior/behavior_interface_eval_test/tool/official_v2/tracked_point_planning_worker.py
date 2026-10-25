"""Bounded, simulator-free process for frozen tracked-point plan requests."""

from __future__ import annotations

import contextlib
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np


SCHEMA = "official_frozen_tracked_point_worker_v1"
_ARRAY_TAG = "__planning_ndarray__"
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _encode(value: Any, *, strict: bool = False) -> Any:
    if isinstance(value, np.ndarray):
        return {_ARRAY_TAG: _encode(value.tolist(), strict=strict)}
    if isinstance(value, np.generic):
        return _encode(value.item(), strict=strict)
    if isinstance(value, Mapping):
        return {str(key): _encode(item, strict=strict) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, strict=strict) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if strict:
            raise ValueError("frozen planning input contains a non-finite number")
        return None
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_ARRAY_TAG}:
            array = np.asarray(value[_ARRAY_TAG], dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("planning worker returned a non-finite array")
            return array
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=2.0)


def run_plan_worker(
    kwargs: Mapping[str, Any], *, cancel_event=None,
    hard_deadline_monotonic: float,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run only local planning; the parent retains all action and plan signing."""
    from ...robot_contract import ACTION_DIM, ROBOT_PROFILE
    from .tracked_point_motion_local import PlanningDeadlineExceeded

    deadline = float(hard_deadline_monotonic)
    if not math.isfinite(deadline):
        raise ValueError("planning worker requires a finite deadline")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("frozen tracked-point planning was cancelled")
    if time.monotonic() >= deadline:
        raise PlanningDeadlineExceeded("frozen planning worker deadline expired before launch")
    arguments = dict(kwargs)
    arguments["state"] = vars(arguments["state"])
    arguments["hard_deadline_monotonic"] = deadline
    payload = {"schema": SCHEMA, "robot_profile": ROBOT_PROFILE,
               "action_dim": ACTION_DIM, "kwargs": arguments}
    wire = json.dumps(_encode(payload, strict=True), allow_nan=False).encode("utf-8")
    if len(wire) > _MAX_MESSAGE_BYTES:
        raise ValueError("frozen planning request exceeds its bounded IPC contract")
    context = arguments.get("overlap_filter_context") or {}
    artifact_root = (context.get("session") or {}).get("init_dir")
    artifact_dir = Path(tempfile.mkdtemp(prefix="move_tracked_planning_", dir=artifact_root))
    request_path = artifact_dir / "request.json"
    request_path.write_bytes(wire)
    request_digest = hashlib.sha256(wire).hexdigest()
    diagnostics: dict[str, Any] = {
        "worker_schema": SCHEMA, "worker_isolated": True,
        "request_path": str(request_path), "request_sha256": request_digest,
        "stderr_path": str(artifact_dir / "worker.stderr.log"),
        "stage": "worker_startup", "stages": [],
    }

    def publish(report: Mapping[str, Any]) -> None:
        diagnostics.update(dict(report))
        if progress_callback is not None:
            progress_callback(dict(diagnostics))

    started = time.monotonic()
    with (artifact_dir / "worker.stderr.log").open("wb") as error_stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", __name__, str(request_path), request_digest],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error_stream,
            cwd=str(Path(__file__).resolve().parents[3]),
            env={**os.environ, "BEHAVIOR_EVAL_TEST_ROBOT_PROFILE": ROBOT_PROFILE},
        )
        publish({"worker_pid": process.pid})
        result = None
        buffer = b""
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while result is None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("frozen tracked-point planning was cancelled")
                    # Consume a completed result before checking elapsed time.
                    events = selector.select(timeout=0.02)
                    if events:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        buffer += chunk
                        if len(buffer) > _MAX_MESSAGE_BYTES:
                            raise RuntimeError("planning worker response exceeds its bounded IPC contract")
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            message = _decode(json.loads(line))
                            if message.get("kind") == "progress":
                                publish(message["report"])
                            elif message.get("kind") == "result":
                                result = message["result"]
                            elif message.get("kind") == "error":
                                if message.get("error_type") == "PlanningDeadlineExceeded":
                                    raise PlanningDeadlineExceeded(message["error"])
                                raise RuntimeError(message["error"])
                            else:
                                raise RuntimeError("planning worker returned an unknown message")
                        if not chunk and result is None:
                            raise RuntimeError("planning worker exited without a result")
                    if result is not None:
                        break
                    if time.monotonic() >= deadline:
                        # Capture the owned child's blocked native/Python stage,
                        # then terminate it. Never leave a timed-out GPU task alive.
                        if process.poll() is None and hasattr(signal, "SIGUSR1"):
                            process.send_signal(signal.SIGUSR1)
                            time.sleep(0.05)
                        raise PlanningDeadlineExceeded(
                            "frozen planning worker timed out in "
                            + str(diagnostics.get("stage", "unknown"))
                        )
            if not isinstance(result, dict):
                raise RuntimeError("planning worker result must be an object")
            (artifact_dir / "result.json").write_text(
                json.dumps(_encode(result), allow_nan=False), encoding="utf-8"
            )
            return result
        finally:
            _stop_worker(process)
            process.stdout.close()
            publish({"worker_reaped": True, "worker_elapsed_s": time.monotonic() - started})
            (artifact_dir / "diagnostics.json").write_text(
                json.dumps(_encode(diagnostics), allow_nan=False), encoding="utf-8"
            )


def main() -> None:
    protocol_stream = sys.stdout

    def emit(message: dict[str, Any]) -> None:
        protocol_stream.write(json.dumps(_encode(message), allow_nan=False) + "\n")
        protocol_stream.flush()

    faulthandler.enable(file=sys.stderr)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from ...robot_contract import ACTION_DIM, ROBOT_PROFILE
            from .grasp_kinematics_local import LocalRobotState
            from .tools import _move_tracked_plan_frozen

            request_path = Path(sys.argv[1])
            if request_path.stat().st_size > _MAX_MESSAGE_BYTES:
                raise ValueError("frozen planning request is too large")
            request_bytes = request_path.read_bytes()
            if hashlib.sha256(request_bytes).hexdigest() != sys.argv[2]:
                raise ValueError("frozen planning request integrity mismatch")
            payload = _decode(json.loads(request_bytes))
            if (payload.get("schema") != SCHEMA
                    or payload.get("robot_profile") != ROBOT_PROFILE
                    or payload.get("action_dim") != ACTION_DIM):
                raise ValueError("frozen planning worker robot/schema contract mismatch")
            kwargs = payload["kwargs"]
            kwargs["state"] = LocalRobotState(**kwargs["state"])
            kwargs["progress_callback"] = lambda report: emit({"kind": "progress", "report": report})
            result = _move_tracked_plan_frozen(**kwargs)
            emit({"kind": "result", "result": result})
        except Exception as exc:
            emit({"kind": "error", "error_type": type(exc).__name__, "error": str(exc)})


if __name__ == "__main__":
    main()
