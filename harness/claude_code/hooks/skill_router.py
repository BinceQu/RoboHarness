#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import Request

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from embodied_claude_code import skill_state
from embodied_claude_code.config import behavior_urlopen as urlopen, validate_owned_origin


BASELINE_SKILL = "behavior-v2-baseline"
# Claude Code 原生 skill 工具名；主机读 SKILL.md，不是给模型开 shell/文件系统。
NATIVE_SKILL_TOOL_RE = re.compile(
    r"^(skill|skills(?:\.[A-Za-z][A-Za-z0-9_]*)?|functions\.skills?(?:\.[A-Za-z][A-Za-z0-9_]*)?)$",
    re.IGNORECASE,
)
# 目录扫描失败时只回退到这次允许的三个任务 skill，不要把未成熟 skill 再列出来。
DEFAULT_TASK_SKILLS = (
    "open-doors-and-drawers",
    "pick-up-object",
    "place-object-in-container",
)
HIDDEN_TASK_SKILLS = frozenset({
    "close-box",
    "cut-object",
    "navigate-to-target",
    "pick-up-object-on-ground",
    "stand-trash-can-upright",
    "traverse-narrow-passages",
})


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    text: str
    path: Path


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


def discover_skills(plugin_root: Path) -> list[SkillDocument]:
    skill_root = plugin_root / "skills"
    documents = []
    for path in sorted(skill_root.glob("*/SKILL.md")):
        document = read_skill(path, skill_root)
        if (
            document is not None
            and document.name != BASELINE_SKILL
            and document.name not in HIDDEN_TASK_SKILLS
        ):
            documents.append(document)
    return documents


def render_task_skill_catalog(plugin_root: Path | None = None) -> str:
    """SessionStart 只注入任务 skill 的名字和简介，正文留给 activate_skill。"""
    documents = discover_skills(plugin_root or Path("."))
    lines = [
        "Available task Skills (name and description only). Call "
        "`activate_skill` with exactly one `name` to load that SKILL.md. "
        "Keep at most one task Skill active. Use `deactivate_skill` to finish, "
        "cancel, or return control to the parent task. Do not activate "
        "behavior-v2-baseline again."
    ]
    for document in documents:
        lines.append(f"- `{document.name}`: {document.description}")
    return "\n".join(lines)


def known_task_skill_names(plugin_root: Path | None = None) -> tuple[str, ...]:
    """磁盘上的任务 skill 名；baseline 不算任务 skill。"""
    if plugin_root is None:
        return DEFAULT_TASK_SKILLS
    names = tuple(document.name for document in discover_skills(Path(plugin_root)))
    return names or DEFAULT_TASK_SKILLS


def is_native_skill_tool(tool_name: str) -> bool:
    """Claude Code loads a plugin Skill through its host Skill tool."""
    return NATIVE_SKILL_TOOL_RE.fullmatch(str(tool_name or "").strip()) is not None


def _normalized_skill_name(value: Any) -> str:
    text = str(value or "").strip().replace("_", "-")
    return text.lower()


def extract_invoked_skill_name(
    payload: dict[str, Any],
    plugin_root: Path | None = None,
) -> str:
    """从 Claude Code 原生 skill 工具参数里取出当前 invoke 的那一个任务 skill。"""
    known = set(known_task_skill_names(plugin_root))
    tool_input = payload.get("tool_input")
    if tool_input is None:
        tool_input = payload.get("arguments")
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
    candidates: list[str] = []
    if isinstance(tool_input, dict):
        for key in ("name", "skill", "skill_name", "package", "resource"):
            raw = _normalized_skill_name(tool_input.get(key))
            if not raw:
                continue
            for skill in known:
                if (raw in (skill, "embodied-claude-code:" + skill)
                    or raw.endswith("/" + skill) or raw.endswith("/" + skill + "/skill.md")):
                    candidates.append(skill)
    unique: list[str] = []
    for name in candidates:
        if name not in unique:
            unique.append(name)
    if len(unique) == 1:
        return unique[0]
    return ""


def _active_skill_stamp_paths(session_id: str = "") -> list[Path]:
    return skill_state.legacy_stamp_paths(session_id)


def save_stamped_active_task_skill(name: str, session_id: str = "") -> None:
    skill_state.save_active(name or None, session_id)


def load_stamped_active_task_skill(session_id: str = "") -> str:
    state = skill_state.read_state(session_id)
    return (state.active_task_skill or "") if state else ""


def resolve_active_task_skill(
    plugin_root: Path | None = None,
    invoked: str = "",
    *,
    query_monitor: bool = False,
    base_url: str = "",
    opener=urlopen,
) -> str:
    """The monitor is an output, never an authority for workflow restoration."""
    del query_monitor, base_url, opener
    known = set(known_task_skill_names(plugin_root))
    candidates = [str(invoked or "").strip(), load_stamped_active_task_skill()]
    for candidate in candidates:
        skill = str(candidate or "").strip()
        if skill and skill != BASELINE_SKILL and skill in known:
            return skill
    return ""


def render_activated_task_skill(plugin_root: Path, name: str) -> str:
    """把已激活任务 skill 的 SKILL.md 包成 SessionStart 注入块。"""
    skill = str(name or "").strip()
    if not skill or skill == BASELINE_SKILL:
        return ""
    for document in discover_skills(plugin_root):
        if document.name != skill:
            continue
        return (
            f'<activated_skill name="{document.name}">\n'
            f"{document.text.strip()}\n"
            "</activated_skill>"
        )
    return ""


def loaded_skill_names(
    prompt: str = "",
    plugin_root: Path | None = None,
    invoked: str = "",
) -> list[str]:
    """baseline 始终在；任务 skill 只追加当前上下文里那一份。"""
    del prompt, plugin_root
    names = [BASELINE_SKILL]
    skill = str(invoked or "").strip()
    if skill and skill != BASELINE_SKILL and skill not in names:
        names.append(skill)
    return names


def _text_from_value(value: Any) -> str:
    """从字符串 / 嵌套对象 / content 数组里抽出一段用户原文。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "prompt", "user_prompt", "content", "input"):
            found = _text_from_value(value.get(key))
            if found:
                return found
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                found = _text_from_value(
                    item.get("text")
                    or item.get("prompt")
                    or item.get("user_prompt")
                    or item.get("content")
                )
                if found:
                    parts.append(found)
        return "\n".join(parts).strip()
    return ""


def extract_user_prompt(payload: Any) -> str:
    """从 UserPromptSubmit / SessionStart stdin 取出用户原文。"""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    for key in ("prompt", "user_prompt", "text", "input", "user_input"):
        found = _text_from_value(payload.get(key))
        if found:
            return found
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    event = payload.get("event")
    if isinstance(event, dict):
        found = _text_from_value(event.get("user_prompt") or event.get("prompt"))
        if found:
            return found
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"", "user"}:
                continue
            found = _text_from_value(
                item.get("content") or item.get("text") or item.get("prompt")
            )
            if found:
                return found
    return ""


def _stamp_dirs() -> list[Path]:
    """hook 若仍带着 XDG，只读自己的 runtime，避免误读别人的 /tmp stamp。"""
    xdg = str(os.environ.get("XDG_RUNTIME_DIR") or "").strip()
    if xdg:
        return [Path(xdg) / "embodied-claude-code"]
    if os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        raise ValueError("Official hook requires its isolated XDG_RUNTIME_DIR.")
    return [Path("/tmp/embodied-claude-code")]


def load_stamped_user_prompt(session_id: str = "") -> str:
    """launcher 在 exec 启动前落下的原文；按 session / 显式文件对齐，避免串台。"""
    sid = str(session_id or os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
    paths: list[Path] = []
    explicit = str(os.environ.get("BEHAVIOR_USER_PROMPT_FILE") or "").strip()
    if explicit:
        paths.append(Path(explicit))
    if sid:
        for folder in _stamp_dirs():
            paths.append(folder / f"user_prompt.{sid}")
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def resolve_prompt_for_publish(payload: Any) -> str:
    """先从 hook payload 抽，抽不到再读 launcher stamp。"""
    found = extract_user_prompt(payload)
    if found:
        return found
    stamped = load_stamped_user_prompt()
    if (
        not stamped
        and isinstance(payload, dict)
        and str(payload.get("hook_event_name") or "") == "UserPromptSubmit"
    ):
        print(
            "[embodied-claude-code] UserPromptSubmit missing prompt; keys="
            + ",".join(sorted(str(key) for key in payload)),
            file=sys.stderr,
        )
    return stamped


def resolve_base_url(explicit: str = "") -> str:
    """hook 进程可能拿不到 launcher 的 env，再读一份 stamp。"""
    for candidate in (
        explicit,
        os.environ.get("BEHAVIOR_BASE_URL"),
        os.environ.get("BEHAVIOR_INTERFACE_URL"),
    ):
        text = str(candidate or "").strip().rstrip("/")
        if text:
            validate_owned_origin(text)
            return text
    if os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        raise ValueError("Official hook requires its bound BEHAVIOR_BASE_URL.")
    for folder in _stamp_dirs():
        stamp = folder / "behavior_base_url"
        try:
            text = stamp.read_text(encoding="utf-8").strip().rstrip("/")
        except OSError:
            continue
        if text:
            return text
    return ""


def publish_user_prompt(
    prompt: str,
    *,
    base_url: str = "",
    session_id: str = "",
    plugin_root: Path | None = None,
    invoked: str = "",
    new_attempt: bool = False,
    include_skills: bool = True,
    opener=urlopen,
) -> bool:
    """把用户原文和当前上下文里的 skill（baseline + 至多一份任务 SKILL.md）发到监视器。"""
    text = str(prompt or "").strip()
    skills = loaded_skill_names(text, plugin_root, invoked=invoked)
    origin = resolve_base_url(base_url)
    if (not text and not (include_skills and skills)) or not origin:
        if (text or (include_skills and skills)) and not origin:
            print(
                "[embodied-claude-code] skip prompt publish: BEHAVIOR_BASE_URL missing",
                file=sys.stderr,
            )
        return False
    body: dict[str, Any] = {}
    if include_skills:
        body["loaded_skills"] = skills
    if text:
        body["prompt"] = text
        body["user_prompt"] = text
    # Use the BEHAVIOR session, not Claude Code's conversation UUID.
    sid = str(session_id or os.environ.get("BEHAVIOR_SESSION_ID") or "").strip()
    if sid:
        body["session_id"] = sid
    if new_attempt:
        body["new_attempt"] = True
    request = Request(
        f"{origin}/api/agent_monitor/prompt",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=3.0) as response:
            ok = 200 <= int(getattr(response, "status", 200)) < 300
            if not ok:
                print(
                    f"[embodied-claude-code] prompt publish HTTP {getattr(response, 'status', '?')}",
                    file=sys.stderr,
                )
            return ok
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"[embodied-claude-code] prompt publish failed: {exc}", file=sys.stderr)
        return False


def decision(payload: dict[str, Any], plugin_root: Path) -> dict[str, Any] | None:
    """不再按 prompt 注入任务 SKILL.md；任务 skill 走 Claude Code 原生 invoke。"""
    del payload, plugin_root
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    plugin_root = Path(
        os.environ.get("CLAUDE_PLUGIN_ROOT")
        or os.environ.get("EMBODIED_PLUGIN_ROOT")
        or os.environ.get("PLUGIN_ROOT")
        or Path(__file__).resolve().parents[1]
    )
    if isinstance(payload, dict):
        if os.environ.get("BEHAVIOR_SESSION_ID"):
            with skill_state.locked_state() as writer:
                state = writer.set_active(writer.state.active_task_skill)
                synced = publish_user_prompt(
                    resolve_prompt_for_publish(payload), plugin_root=plugin_root,
                    invoked=state.active_task_skill or "",
                )
                writer.mark_monitor_synced(synced)
        else:
            publish_user_prompt(resolve_prompt_for_publish(payload), include_skills=False)
    result = decision(payload if isinstance(payload, dict) else {}, plugin_root)
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
