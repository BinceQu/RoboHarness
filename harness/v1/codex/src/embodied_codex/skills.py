"""Codex 原生 skill 的 embodied 实现：目录可见，invoke 后才给出一份 SKILL.md。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


BASELINE_SKILL = "behavior-v2-baseline"
MAX_SKILL_CHARS = 32_000
# 磁盘上保留 SKILL.md，但不进目录、也不能 activate。
HIDDEN_TASK_SKILLS = frozenset({"stand-trash-can-upright"})
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    text: str
    path: Path


_ACTIVE_TASK_SKILL = ""


def _active_skill_stamp_paths(session_id: str = "") -> list[Path]:
    """与 hook 约定同一份 stamp：active_task_skill.<BEHAVIOR_SESSION_ID>。"""
    sid = str(session_id or os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
    if not sid:
        return []
    name = f"active_task_skill.{sid}"
    folders: list[Path] = []
    xdg = str(os.environ.get("XDG_RUNTIME_DIR") or "").strip()
    if xdg:
        folders.append(Path(xdg) / "embodied-codex")
    tmp = Path("/tmp/embodied-codex")
    if tmp not in folders:
        folders.append(tmp)
    return [folder / name for folder in folders]


def save_active_task_skill_stamp(name: str, session_id: str = "") -> None:
    """activate 成功后记下：这份 SKILL.md 现在在上下文里。"""
    skill = str(name or "").strip()
    if not skill or skill == BASELINE_SKILL:
        return
    for path in _active_skill_stamp_paths(session_id):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(skill + "\n", encoding="utf-8")
        except OSError:
            continue


def load_active_task_skill_stamp(session_id: str = "") -> str:
    for path in _active_skill_stamp_paths(session_id):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text and text != BASELINE_SKILL:
            return text
    return ""


def current_task_skill(root: Path | None = None) -> str:
    """内存优先，MCP 重启后从 stamp 恢复。"""
    global _ACTIVE_TASK_SKILL
    if _ACTIVE_TASK_SKILL:
        return _ACTIVE_TASK_SKILL
    name = load_active_task_skill_stamp()
    if not name:
        return ""
    known = {document.name for document in discover_task_skills(root)}
    if name in known:
        _ACTIVE_TASK_SKILL = name
        return name
    return ""


def plugin_root() -> Path:
    for key in ("EMBODIED_PLUGIN_ROOT", "PLUGIN_ROOT"):
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
        text=text[:MAX_SKILL_CHARS],
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


def catalog_payload(root: Path | None = None) -> dict[str, Any]:
    documents = discover_task_skills(root)
    return {
        "ok": True,
        "mode": "list",
        "active_skill": current_task_skill(root) or None,
        "baseline_already_active": True,
        "instruction": (
            "Call activate_skill again with exactly one `name` to load that "
            "SKILL.md. Keep only one task Skill active."
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
        "The baseline Skill is already active; do not activate it again."
    ]
    for document in documents:
        lines.append(f"- `{document.name}`: {document.description}")
    return "\n".join(lines)


def activate_skill(name: str = "", root: Path | None = None) -> dict[str, Any]:
    """空名字返回目录；有名字则载入那一份任务 SKILL.md。"""
    global _ACTIVE_TASK_SKILL
    requested = str(name or "").strip().replace("_", "-")
    if not requested:
        return catalog_payload(root)
    if not SKILL_NAME_RE.fullmatch(requested):
        return {
            "ok": False,
            "error": "Skill name must be hyphen-case, such as pick-up-object.",
            "skills": catalog_payload(root)["skills"],
        }
    if requested == BASELINE_SKILL:
        return {
            "ok": False,
            "error": "behavior-v2-baseline is already active. Activate a task Skill.",
            "skills": catalog_payload(root)["skills"],
        }
    documents = {document.name: document for document in discover_task_skills(root)}
    document = documents.get(requested)
    if document is None:
        return {
            "ok": False,
            "error": f"Unknown task Skill `{requested}`.",
            "skills": catalog_payload(root)["skills"],
        }
    _ACTIVE_TASK_SKILL = document.name
    save_active_task_skill_stamp(document.name)
    publish_loaded_skills(document.name)
    return {
        "ok": True,
        "mode": "activated",
        "name": document.name,
        "active_skill": document.name,
        "instruction": (
            "This is the only active task Skill. Follow it until its exit "
            "condition, then activate the next Skill if needed."
        ),
        "body": document.text.strip(),
    }


def resolve_base_url() -> str:
    for key in ("BEHAVIOR_BASE_URL", "BEHAVIOR_INTERFACE_URL"):
        text = str(os.environ.get(key) or "").strip().rstrip("/")
        if text:
            return text
    return ""


def publish_loaded_skills(invoked: str = "", opener=urlopen) -> bool:
    """把当前任务 skill 打到 Agent Monitor；失败不影响 activate。"""
    origin = resolve_base_url()
    if not origin:
        return False
    skills = [BASELINE_SKILL]
    name = str(invoked or "").strip()
    if name and name != BASELINE_SKILL and name not in skills:
        skills.append(name)
    body: dict[str, Any] = {"loaded_skills": skills}
    sid = str(os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
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
