"""Claude Code 原生 skill 的 embodied 实现：目录可见，invoke 后才给出一份 SKILL.md。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import skill_state


BASELINE_SKILL = "behavior-v2-baseline"
# 磁盘上保留 SKILL.md，但不进目录、也不能 activate。
# 本次评测只开放三个任务 skill，其余仍隐藏。baseline 由 SessionStart 单独注入。
HIDDEN_TASK_SKILLS = frozenset({
    "close-box",
    "cut-object",
    "navigate-to-target",
    "pick-up-object-on-ground",
    "stand-trash-can-upright",
    "traverse-narrow-passages",
})
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    text: str
    path: Path


def _active_skill_stamp_paths(session_id: str = "") -> list[Path]:
    return skill_state.legacy_stamp_paths(session_id)


def save_active_task_skill_stamp(name: str, session_id: str = "") -> None:
    skill_state.save_active(name or None, session_id)


def load_active_task_skill_stamp(session_id: str = "") -> str:
    state = skill_state.read_state(session_id)
    return (state.active_task_skill or "") if state else ""


def current_task_skill(root: Path | None = None, session_id: str = "") -> str:
    """Read shared state on every call; native hooks run in other processes."""
    name = load_active_task_skill_stamp(session_id)
    if not name:
        return ""
    known = {document.name for document in discover_task_skills(root)}
    if name in known:
        return name
    return ""


def plugin_root() -> Path:
    for key in ("CLAUDE_PLUGIN_ROOT", "EMBODIED_PLUGIN_ROOT", "PLUGIN_ROOT"):
        raw = str(os.environ.get(key) or "").strip()
        if raw:
            return Path(raw)
    return Path(__file__).resolve().parents[2]


def _frontmatter_scalar(lines: list[str], key: str) -> str:
    prefix = key + ":"
    for line in lines:
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return ""


def read_skill(path: Path, skill_root: Path) -> SkillDocument | None:
    resolved = path.resolve()
    try:
        resolved.relative_to(skill_root.resolve())
    except ValueError:
        return None
    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 4 or lines[0].strip() != "---":
        return None
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None
    frontmatter = lines[1:end]
    name = _frontmatter_scalar(frontmatter, "name")
    description = _frontmatter_scalar(frontmatter, "description")
    if not name or not description:
        return None
    return SkillDocument(
        name=name,
        description=description,
        text=text,
        path=resolved,
    )


def discover_task_skills(root: Path | None = None) -> list[SkillDocument]:
    skill_root = (root or plugin_root()) / "skills"
    documents = []
    if not skill_root.is_dir():
        return documents
    for path in sorted(skill_root.glob("*/SKILL.md")):
        document = read_skill(path, skill_root)
        if (
            document is not None
            and document.name != BASELINE_SKILL
            and document.name not in HIDDEN_TASK_SKILLS
        ):
            documents.append(document)
    return documents


def catalog_payload(root: Path | None = None, session_id: str = "") -> dict[str, Any]:
    documents = discover_task_skills(root)
    return {
        "ok": True,
        "mode": "list",
        "active_skill": current_task_skill(root, session_id) or None,
        "baseline_already_active": True,
        "instruction": (
            "Call activate_skill again with exactly one `name` to load that "
            "SKILL.md. Keep at most one task Skill active. Call deactivate_skill "
            "to return to the baseline and parent task."
        ),
        "skills": [
            {"name": document.name, "description": document.description}
            for document in documents
        ],
    }


def render_catalog_markdown(root: Path | None = None) -> str:
    documents = discover_task_skills(root)
    lines = [
        "Available task Skills (name and description only). Call "
        "`activate_skill` with exactly one `name` to load that SKILL.md. "
        "The baseline Skill is already active; do not activate it again. "
        "Use `deactivate_skill` to finish, cancel, or hand control back to the parent task."
    ]
    for document in documents:
        lines.append(f"- `{document.name}`: {document.description}")
    return "\n".join(lines)


def activate_skill(name: str = "", root: Path | None = None, *,
                   session_id: str = "", base_url: str = "") -> dict[str, Any]:
    """空名字返回目录；有名字则载入那一份任务 SKILL.md。"""
    requested = str(name or "").strip().replace("_", "-")
    if not requested:
        return catalog_payload(root, session_id)
    if not SKILL_NAME_RE.fullmatch(requested):
        return {
            "ok": False,
            "error": "Skill name must be hyphen-case, such as pick-up-object.",
            "skills": catalog_payload(root, session_id)["skills"],
        }
    if requested == BASELINE_SKILL:
        return {
            "ok": False,
            "error": "behavior-v2-baseline is already active. Activate a task Skill.",
            "skills": catalog_payload(root, session_id)["skills"],
        }
    documents = {document.name: document for document in discover_task_skills(root)}
    document = documents.get(requested)
    if document is None:
        return {
            "ok": False,
            "error": f"Unknown task Skill `{requested}`.",
            "skills": catalog_payload(root, session_id)["skills"],
        }
    try:
        with skill_state.locked_state(session_id) as writer:
            previous = writer.state.active_task_skill
            state = writer.set_active(document.name)
            synced = publish_loaded_skills(document.name, session_id=session_id, base_url=base_url)
            writer.mark_monitor_synced(synced)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"Could not activate task Skill: {exc}"}
    return {
        "ok": True,
        "mode": "activated",
        "name": document.name,
        "active_skill": document.name,
        "previous_skill": previous,
        "revision": state.revision,
        "monitor_synced": synced,
        "state_notice": skill_state.state_notice(state),
        "instruction": (
            "This is the only active task Skill. After satisfying its success "
            "conditions and required post-actions, call deactivate_skill. You may "
            "also deactivate to cancel or hand control back without claiming success."
        ),
        "body": document.text.strip(),
    }


def deactivate_skill(name: str, reason: str = "", root: Path | None = None, *,
                     session_id: str = "", base_url: str = "") -> dict[str, Any]:
    requested = str(name or "").strip().replace("_", "-")
    if not SKILL_NAME_RE.fullmatch(requested):
        return {"ok": False, "error": "Skill name must be hyphen-case, such as pick-up-object."}
    if requested == BASELINE_SKILL:
        return {"ok": False, "error": "The baseline remains active throughout the session."}
    known = {doc.name for doc in discover_task_skills(root)}
    try:
        leftover = load_active_task_skill_stamp(session_id)
        # A disabled skill may still have an active stamp from an older session.
        if requested not in known and requested != leftover:
            return {"ok": False, "error": f"Unknown task Skill `{requested}`."}
        with skill_state.locked_state(session_id) as writer:
            previous = writer.state.active_task_skill
            if previous is not None and previous != requested:
                return {"ok": False, "error": "Task Skill name does not match the active Skill.",
                        "active_skill": previous}
            state = writer.set_active(None)
            synced = publish_loaded_skills(session_id=session_id, base_url=base_url)
            writer.mark_monitor_synced(synced)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"Could not deactivate task Skill: {exc}"}
    return {
        "ok": True,
        "mode": "deactivated",
        "name": requested,
        "reason": reason,
        "changed": previous is not None,
        "active_skill": None,
        "active_skills": [BASELINE_SKILL],
        "revision": state.revision,
        "monitor_synced": synced,
        "state_notice": skill_state.state_notice(state),
        "instruction": (
            "Control has returned to the baseline and parent task. Stop following "
            "this Skill's action sequence. Deactivation does not establish subtask "
            "success or parent-task completion, or change the robot's physical state."
        ),
    }


def resolve_base_url() -> str:
    for key in ("BEHAVIOR_BASE_URL", "BEHAVIOR_INTERFACE_URL"):
        text = str(os.environ.get(key) or "").strip().rstrip("/")
        if text:
            return text
    return ""


def publish_loaded_skills(invoked: str = "", opener=urlopen, *,
                          session_id: str = "", base_url: str = "") -> bool:
    """把当前任务 skill 打到 Agent Monitor；失败不影响 activate。"""
    origin = base_url.rstrip("/") or resolve_base_url()
    if not origin:
        return False
    skills = [BASELINE_SKILL]
    name = str(invoked or "").strip()
    if name and name != BASELINE_SKILL and name not in skills:
        skills.append(name)
    body: dict[str, Any] = {"loaded_skills": skills}
    sid = str(session_id or os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
    if sid:
        body["session_id"] = sid
    request = Request(
        f"{origin}/api/agent_monitor/prompt",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=3.0) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except (URLError, TimeoutError, OSError, ValueError):
        return False
