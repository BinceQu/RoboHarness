"""Human-operated public tool trajectory recording.

The recorder intentionally sits above the skill layer: one browser-initiated
public tool call becomes one training turn even when the route performs
additional internal captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import tempfile
import threading
import time
from typing import Any
import uuid

from . import agent_runs


SCHEMA_VERSION = "behavior.human_tool_trajectory.v1"


class RecordingConflict(RuntimeError):
    """Raised when a second human recording would overlap the active one."""


class RecordingNotFound(KeyError):
    """Raised for an unknown or inactive recording id."""


@dataclass
class _Recording:
    record_id: str
    session_id: str
    run_dir: str
    manifest: dict[str, Any]
    tool_specs: dict[str, dict[str, Any]]
    lease: agent_runs.ActiveSessionLease | None
    status: str = "recording"
    turn_count: int = 0
    in_flight: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_json(value: Any) -> Any:
    """Return JSON-safe data while replacing inline binary payloads."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:128]:
            header, _, encoded = value.partition(",")
            mime = header[5:].split(";", 1)[0] or "application/octet-stream"
            return {
                "omitted": "inline_data_url",
                "mime": mime,
                "encoded_chars": len(encoded),
            }
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)


def _artifact_refs(value: Any) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {
        "image_ids": [],
        "plan_ids": [],
        "paths": [],
    }

    def add(bucket: str, item: Any) -> None:
        if not isinstance(item, str):
            return
        item = item.strip()
        if item and item not in refs[bucket]:
            refs[bucket].append(item)

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, str(child_key).lower())
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child, key)
            return
        if key == "image_id" or key.endswith("_image_id"):
            add("image_ids", node)
        elif key == "plan_id" or key.endswith("_plan_id"):
            add("plan_ids", node)
        elif key == "path" or key.endswith("_path"):
            add("paths", node)

    walk(value)
    return refs


class HumanTrajectoryRecorder:
    """Thread-safe process-local registry backed by append-only JSONL files."""

    def __init__(self, root: str | None = None) -> None:
        self.root = os.path.abspath(root or agent_runs.RUNS_ROOT)
        self._lock = threading.RLock()
        self._records: dict[str, _Recording] = {}
        self._active_id: str | None = None

    def start(
        self,
        *,
        state: dict[str, Any],
        memory_text: str,
        tool_version: str,
        tools: list[dict[str, Any]],
        label: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            active = self._active_locked()
            if active is not None:
                raise RecordingConflict(
                    f"recording already active: {active.record_id}"
                )

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            record_id = f"human-{stamp}-{uuid.uuid4().hex[:8]}"
            session_id = record_id
            run_dir = agent_runs.managed_session_dir(self.root, session_id)
            lease = None
            try:
                self._ensure_session(run_dir, session_id)
                lease = agent_runs.acquire_session_lease(
                    session_id,
                    root=self.root,
                )
                os.makedirs(os.path.join(run_dir, "turns"), exist_ok=True)

                started_at = _utc_now()
                safe_tools = _safe_json(tools)
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": record_id,
                    "session_id": session_id,
                    "source": "human_involve",
                    "status": "recording",
                    "started_at": started_at,
                    "updated_at": started_at,
                    "completed_at": None,
                    "outcome": None,
                    "stop_reason": None,
                    "label": str(label or "")[:200],
                    "note": str(note or "")[:2000],
                    "tool_version": str(tool_version or ""),
                    "tool_schema_snapshot": safe_tools,
                    "initial": {
                        "state": _safe_json(state),
                        "memory_text": str(memory_text or ""),
                    },
                    "bootstrap": None,
                    "turn_count": 0,
                    "failed_turn_count": 0,
                    "in_flight": 0,
                }
                recording = _Recording(
                    record_id=record_id,
                    session_id=session_id,
                    run_dir=run_dir,
                    manifest=manifest,
                    tool_specs={
                        str(tool.get("name")): dict(tool)
                        for tool in tools
                        if isinstance(tool, dict) and tool.get("name")
                    },
                    lease=lease,
                )
                self._records[record_id] = recording
                self._active_id = record_id
                self._write_manifest_locked(recording)
                self._append_event_locked(
                    recording,
                    {
                        "event": "recording_started",
                        "at": started_at,
                        "record_id": record_id,
                        "session_id": session_id,
                        "tool_version": str(tool_version or ""),
                    },
                )
                return self._status_locked(recording)
            except BaseException:
                self._records.pop(record_id, None)
                if self._active_id == record_id:
                    self._active_id = None
                if lease is not None:
                    agent_runs.release_session_lease(lease)
                raise

    def begin_tool(
        self,
        *,
        record_id: str,
        tool_name: str,
        endpoint: str,
        request_args: dict[str, Any],
        input_media: dict[str, Any] | None,
        pre_state: dict[str, Any],
        pre_memory_text: str,
        bootstrap: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            recording = self._require_active_locked(record_id)
            started_at = _utc_now()
            started_monotonic = time.monotonic()
            safe_request_args = _safe_json(request_args)
            model_args = self._model_args_locked(
                recording, tool_name, request_args
            )
            recording.in_flight += 1
            recording.manifest["in_flight"] = recording.in_flight
            recording.manifest["updated_at"] = started_at

            if bootstrap:
                context = {
                    "record_id": record_id,
                    "bootstrap": True,
                    "started_at": started_at,
                    "started_monotonic": started_monotonic,
                    "tool_name": tool_name,
                    "endpoint": endpoint,
                    "request_args": safe_request_args,
                    "model_args": model_args,
                    "input_media": _safe_json(input_media or {}),
                    "pre_state": _safe_json(pre_state),
                    "pre_memory_text": str(pre_memory_text or ""),
                }
                self._append_event_locked(
                    recording,
                    {
                        "event": "bootstrap_started",
                        "at": started_at,
                        "tool_name": tool_name,
                        "endpoint": endpoint,
                    },
                )
                self._write_manifest_locked(recording)
                return context

            recording.turn_count += 1
            turn_index = recording.turn_count
            turn_relpath = f"turns/turn_{turn_index:04d}.json"
            turn = {
                "schema_version": SCHEMA_VERSION,
                "record_id": record_id,
                "session_id": recording.session_id,
                "turn_index": turn_index,
                "status": "in_progress",
                "started_at": started_at,
                "completed_at": None,
                "duration_ms": None,
                "action": {
                    "tool_name": tool_name,
                    "endpoint": endpoint,
                    "args": model_args,
                    "request_args": safe_request_args,
                },
                "input": {
                    "displayed_media": _safe_json(input_media or {}),
                    "state": _safe_json(pre_state),
                    "memory_text": str(pre_memory_text or ""),
                },
                "output": None,
            }
            context = {
                "record_id": record_id,
                "bootstrap": False,
                "turn_index": turn_index,
                "turn_relpath": turn_relpath,
                "started_at": started_at,
                "started_monotonic": started_monotonic,
                "turn": turn,
            }
            recording.manifest["turn_count"] = turn_index
            self._atomic_json(
                os.path.join(recording.run_dir, turn_relpath),
                turn,
            )
            self._append_event_locked(
                recording,
                {
                    "event": "tool_call_started",
                    "at": started_at,
                    "turn_index": turn_index,
                    "turn_path": turn_relpath,
                    "action": turn["action"],
                },
            )
            self._write_manifest_locked(recording)
            return context

    def finish_tool(
        self,
        context: dict[str, Any],
        *,
        response: Any,
        http_status: int,
        post_state: dict[str, Any],
        post_memory_text: str,
        exception: str | None = None,
    ) -> None:
        with self._lock:
            record_id = str(context.get("record_id") or "")
            recording = self._records.get(record_id)
            if recording is None:
                return

            completed_at = _utc_now()
            duration_ms = max(
                0,
                int(
                    (
                        time.monotonic()
                        - float(context.get("started_monotonic") or time.monotonic())
                    )
                    * 1000
                ),
            )
            safe_response = _safe_json(response)
            artifacts = _artifact_refs(response)
            response_ok = (
                isinstance(response, dict) and response.get("ok") is not False
            )
            succeeded = (
                exception is None
                and int(http_status) < 400
                and response_ok
            )
            status = "succeeded" if succeeded else "failed"

            recording.in_flight = max(0, recording.in_flight - 1)
            recording.manifest["in_flight"] = recording.in_flight
            recording.manifest["updated_at"] = completed_at

            if context.get("bootstrap"):
                recording.manifest["bootstrap"] = {
                    "status": status,
                    "started_at": context.get("started_at"),
                    "completed_at": completed_at,
                    "duration_ms": duration_ms,
                    "action": {
                        "tool_name": context.get("tool_name"),
                        "endpoint": context.get("endpoint"),
                        "args": context.get("model_args") or {},
                        "request_args": context.get("request_args") or {},
                    },
                    "input": {
                        "displayed_media": context.get("input_media") or {},
                        "state": context.get("pre_state") or {},
                        "memory_text": context.get("pre_memory_text") or "",
                    },
                    "output": {
                        "http_status": int(http_status),
                        "response": safe_response,
                        "artifacts": artifacts,
                        "state": _safe_json(post_state),
                        "memory_text": str(post_memory_text or ""),
                        "exception": exception,
                    },
                }
                self._append_event_locked(
                    recording,
                    {
                        "event": "bootstrap_finished",
                        "at": completed_at,
                        "status": status,
                        "duration_ms": duration_ms,
                        "artifacts": artifacts,
                    },
                )
                self._write_manifest_locked(recording)
                return

            turn = dict(context.get("turn") or {})
            turn["status"] = status
            turn["completed_at"] = completed_at
            turn["duration_ms"] = duration_ms
            turn["output"] = {
                "http_status": int(http_status),
                "response": safe_response,
                "artifacts": artifacts,
                "state": _safe_json(post_state),
                "memory_text": str(post_memory_text or ""),
                "exception": exception,
            }
            turn_relpath = str(context.get("turn_relpath") or "")
            if turn_relpath:
                self._atomic_json(
                    os.path.join(recording.run_dir, turn_relpath),
                    turn,
                )
            if not succeeded:
                recording.manifest["failed_turn_count"] = int(
                    recording.manifest.get("failed_turn_count") or 0
                ) + 1
            self._append_event_locked(
                recording,
                {
                    "event": "tool_call_finished",
                    "at": completed_at,
                    "turn_index": context.get("turn_index"),
                    "turn_path": turn_relpath,
                    "status": status,
                    "http_status": int(http_status),
                    "duration_ms": duration_ms,
                    "artifacts": artifacts,
                },
            )
            self._write_manifest_locked(recording)

    def stop(
        self,
        record_id: str,
        *,
        outcome: str,
        reason: str,
        state: dict[str, Any],
        memory_text: str,
    ) -> dict[str, Any]:
        with self._lock:
            recording = self._records.get(str(record_id or ""))
            if recording is None:
                raise RecordingNotFound(record_id)
            if recording.status != "recording":
                return self._status_locked(recording)
            return self._stop_locked(
                recording,
                outcome=outcome,
                reason=reason,
                state=state,
                memory_text=memory_text,
            )

    def stop_all(
        self,
        *,
        outcome: str,
        reason: str,
        state: dict[str, Any],
        memory_text: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            active = self._active_locked()
            if active is None:
                return []
            return [
                self._stop_locked(
                    active,
                    outcome=outcome,
                    reason=reason,
                    state=state,
                    memory_text=memory_text,
                )
            ]

    def status(self, record_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if record_id:
                recording = self._records.get(str(record_id))
                if recording is None:
                    return {"active": False, "record_id": str(record_id), "status": "unknown"}
                return self._status_locked(recording)
            active = self._active_locked()
            if active is None:
                return {"active": False, "status": "idle"}
            return self._status_locked(active)

    def _stop_locked(
        self,
        recording: _Recording,
        *,
        outcome: str,
        reason: str,
        state: dict[str, Any],
        memory_text: str,
    ) -> dict[str, Any]:
        completed_at = _utc_now()
        recording.status = "stopped"
        recording.manifest.update(
            {
                "status": "stopped",
                "updated_at": completed_at,
                "completed_at": completed_at,
                "outcome": str(outcome or "partial"),
                "stop_reason": str(reason or "user_stop"),
                "final": {
                    "state": _safe_json(state),
                    "memory_text": str(memory_text or ""),
                },
            }
        )
        if self._active_id == recording.record_id:
            self._active_id = None
        try:
            self._append_event_locked(
                recording,
                {
                    "event": "recording_stopped",
                    "at": completed_at,
                    "outcome": recording.manifest["outcome"],
                    "reason": recording.manifest["stop_reason"],
                    "turn_count": recording.turn_count,
                    "in_flight": recording.in_flight,
                },
            )
            self._write_manifest_locked(recording)
            return self._status_locked(recording)
        finally:
            if recording.lease is not None:
                agent_runs.release_session_lease(recording.lease)
                recording.lease = None

    def _active_locked(self) -> _Recording | None:
        if not self._active_id:
            return None
        recording = self._records.get(self._active_id)
        if recording is None or recording.status != "recording":
            self._active_id = None
            return None
        return recording

    def _require_active_locked(self, record_id: str) -> _Recording:
        recording = self._records.get(str(record_id or ""))
        if recording is None:
            raise RecordingNotFound(record_id)
        if recording.status != "recording" or self._active_id != recording.record_id:
            raise RecordingConflict(f"recording is not active: {record_id}")
        return recording

    def _model_args_locked(
        self,
        recording: _Recording,
        tool_name: str,
        request_args: dict[str, Any],
    ) -> dict[str, Any]:
        spec = recording.tool_specs.get(str(tool_name or "")) or {}
        allowed = {
            str(arg.get("name"))
            for arg in (spec.get("args") or [])
            if isinstance(arg, dict) and arg.get("name")
        }
        return {
            key: _safe_json(request_args[key])
            for key in sorted(allowed)
            if key in request_args
        }

    def _status_locked(self, recording: _Recording) -> dict[str, Any]:
        return {
            "active": recording.status == "recording",
            "status": recording.status,
            "record_id": recording.record_id,
            "session_id": recording.session_id,
            "started_at": recording.manifest.get("started_at"),
            "completed_at": recording.manifest.get("completed_at"),
            "turn_count": recording.turn_count,
            "failed_turn_count": int(
                recording.manifest.get("failed_turn_count") or 0
            ),
            "in_flight": recording.in_flight,
            "outcome": recording.manifest.get("outcome"),
            "stop_reason": recording.manifest.get("stop_reason"),
            "run_dir": recording.run_dir,
            "manifest_path": os.path.join(recording.run_dir, "recording.json"),
        }

    def _ensure_session(self, run_dir: str, session_id: str) -> None:
        if os.path.realpath(self.root) == os.path.realpath(agent_runs.RUNS_ROOT):
            agent_runs.ensure_session(session_id)
            return
        os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plans"), exist_ok=True)
        self._atomic_json(
            os.path.join(run_dir, "session.json"),
            {"session_id": session_id, "image_counter": 0, "plan_counter": 0},
        )

    def _write_manifest_locked(self, recording: _Recording) -> None:
        self._atomic_json(
            os.path.join(recording.run_dir, "recording.json"),
            recording.manifest,
        )

    def _append_event_locked(
        self,
        recording: _Recording,
        event: dict[str, Any],
    ) -> None:
        path = os.path.join(recording.run_dir, "events.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            json.dump(_safe_json(event), stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_json(path: str, payload: Any) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    _safe_json(payload),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
