from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import uuid

from .media import Media, safe_json


SCHEMA_VERSION = "embodied_codex.trajectory.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _extension(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(mime_type, ".bin")


class TrajectoryRecorder:
    """Append-only model-visible tool trace with content-addressed media."""

    def __init__(
        self,
        *,
        root: Path,
        session_id: str,
        label: str,
        profile: dict[str, Any],
        tool_catalog: dict[str, Any],
        initial_state: Any,
    ) -> None:
        self._lock = threading.RLock()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.record_id = f"codex-{stamp}-{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.run_dir = root.expanduser().resolve() / self.record_id
        self.media_dir = self.run_dir / "media"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.media_dir.mkdir()
        self.turns_path = self.run_dir / "turns.jsonl"
        started_at = utc_now()
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "session_id": session_id,
            "source": "codex_mcp",
            "status": "recording",
            "started_at": started_at,
            "updated_at": started_at,
            "completed_at": None,
            "outcome": None,
            "note": "",
            "label": str(label)[:200],
            "turn_count": 0,
            "profile": safe_json(profile),
            "tool_catalog": safe_json(tool_catalog),
            "initial_state": safe_json(initial_state),
        }
        self._write_manifest()

    def record(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        started_monotonic: float,
        result: Any = None,
        media: list[Media] | None = None,
        error: Any = None,
        is_error: bool = False,
    ) -> None:
        with self._lock:
            index = int(self.manifest["turn_count"]) + 1
            descriptors = [self._store_media(item) for item in media or []]
            event = {
                "schema_version": SCHEMA_VERSION,
                "record_id": self.record_id,
                "session_id": self.session_id,
                "turn_index": index,
                "completed_at": utc_now(),
                "duration_ms": max(
                    0, int((time.monotonic() - started_monotonic) * 1000)
                ),
                "tool_call": {
                    "name": tool_name,
                    "arguments": safe_json(arguments),
                },
                "tool_result": {
                    "is_error": bool(is_error or error is not None),
                    "data": safe_json(result),
                    "error": safe_json(error),
                    "media": descriptors,
                },
            }
            with self.turns_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.manifest["turn_count"] = index
            self.manifest["updated_at"] = event["completed_at"]
            self._write_manifest()

    def finish(self, *, outcome: str, note: str = "") -> None:
        with self._lock:
            completed_at = utc_now()
            self.manifest.update(
                {
                    "status": "completed",
                    "updated_at": completed_at,
                    "completed_at": completed_at,
                    "outcome": str(outcome)[:100],
                    "note": str(note)[:2000],
                }
            )
            self._write_manifest()

    def status(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "run_dir": str(self.run_dir),
            "status": self.manifest["status"],
            "turn_count": self.manifest["turn_count"],
        }

    def _store_media(self, media: Media) -> dict[str, Any]:
        filename = media.sha256 + _extension(media.mime_type)
        path = self.media_dir / filename
        if not path.exists():
            fd, temporary = tempfile.mkstemp(prefix=".media-", dir=self.media_dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(media.data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        descriptor = media.public()
        descriptor["path"] = f"media/{filename}"
        return descriptor

    def _write_manifest(self) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=self.run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.manifest, stream, ensure_ascii=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.run_dir / "manifest.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
