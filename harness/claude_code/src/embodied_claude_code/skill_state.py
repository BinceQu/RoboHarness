"""Session-scoped task workflow state, shared by MCP and Claude hooks.

This state does not describe which documents remain in Claude's history.
An explicit null survives restarts and always takes precedence over old stamps.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator


BASELINE_SKILL = "behavior-v2-baseline"
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class SkillState:
    active_task_skill: str | None = None
    revision: int = 0
    version: int = 1
    monitor_synced: bool = False


def _session_id(session_id: str = "") -> str:
    sid = session_id or os.environ.get("BEHAVIOR_SESSION_ID", "")
    if not SESSION_ID_RE.fullmatch(sid):
        raise ValueError("Skill lifecycle requires a valid BEHAVIOR_SESSION_ID (1..64 characters).")
    return sid


def state_path(session_id: str = "") -> Path:
    folder = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "embodied-claude-code"
    return folder / f"task_skill_state.{_session_id(session_id)}.json"


def legacy_stamp_paths(session_id: str = "") -> list[Path]:
    sid = _session_id(session_id)
    primary = state_path(sid).parent / f"active_task_skill.{sid}"
    if os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        return [primary]
    fallback = Path("/tmp/embodied-claude-code") / primary.name
    return list(dict.fromkeys((primary, fallback)))


def read_state(session_id: str = "") -> SkillState | None:
    if not (session_id or os.environ.get("BEHAVIOR_SESSION_ID")):
        return None
    path = state_path(session_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Read old installations once; subsequent writes use only the JSON state.
        for legacy in legacy_stamp_paths(session_id):
            try:
                name = legacy.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                continue
            if name and name != BASELINE_SKILL and SKILL_NAME_RE.fullmatch(name):
                return SkillState(active_task_skill=name)
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid task Skill state version; refusing to restore an old Skill.")
    name = payload.get("active_task_skill")
    revision = payload.get("revision")
    if (
        "active_task_skill" not in payload
        or (name is not None and (not isinstance(name, str) or not SKILL_NAME_RE.fullmatch(name)
                                 or name == BASELINE_SKILL))
        or type(revision) is not int or revision < 0
    ):
        raise ValueError("Invalid task Skill state; refusing to restore an old Skill.")
    return SkillState(active_task_skill=name, revision=revision,
                      monitor_synced=payload.get("monitor_synced") is True)


class StateWriter:
    def __init__(self, path: Path, state: SkillState | None):
        self.path = path
        self.state = state or SkillState()

    def set_active(self, name: str | None) -> SkillState:
        if name is not None and (not SKILL_NAME_RE.fullmatch(name) or name == BASELINE_SKILL):
            raise ValueError("Only a task Skill or null may be stored as active.")
        if name == self.state.active_task_skill and self.path.exists():
            return self.state
        updated = SkillState(name, self.state.revision + 1)
        self._write(updated)
        return updated

    def _write(self, updated: SkillState) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(asdict(updated), stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.state = updated

    def mark_monitor_synced(self, synced: bool) -> None:
        if self.state.monitor_synced != synced:
            try:
                self._write(replace(self.state, monitor_synced=synced))
            except OSError:
                # Workflow state is already committed. Retry telemetry next time.
                pass


@contextmanager
def locked_state(session_id: str = "") -> Iterator[StateWriter]:
    path = state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep this lock through the monitor POST so older publications cannot
    # overtake a newer lifecycle transition in another hook/MCP process.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield StateWriter(path, read_state(session_id))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_active(name: str | None, session_id: str = "") -> SkillState:
    with locked_state(session_id) as writer:
        return writer.set_active(name)


def state_notice(state: SkillState) -> str:
    payload = asdict(state)
    payload.pop("monitor_synced")
    return (
        "<task_skill_state>" + json.dumps(payload, separators=(",", ":"))
        + "</task_skill_state>\n"
        "This is the current task Skill state. Earlier Skill bodies and activation "
        "messages are historical unless named active here. Null means follow the "
        "baseline and parent task. Loading history after compaction does not activate a Skill."
    )
